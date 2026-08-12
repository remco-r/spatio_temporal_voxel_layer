"""Main window for the FOV Boundary Tool."""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QListWidget, QMessageBox,
    QInputDialog, QAction, QToolBar, QStatusBar, QSplitter,
    QGroupBox, QCheckBox, QUndoStack, QUndoCommand
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush

from .range_image_widget import RangeImageWidget
from .cloud_loader import OrganizedCloud, load_pcd, unfold_cloud
from .convex_decomposition import hertel_mehlhorn, is_convex_angular_polygon, angular_to_dir as _angular_to_dir
from .config_io import BlindSpotConfig, BlindSpotRegion, save_config, load_config, format_polygons_as_param_string


def _dir_to_angular(d: np.ndarray) -> Tuple[float, float]:
    """Unit 3D direction -> (azimuth in [0, 2pi], elevation)."""
    az = float(np.arctan2(d[1], d[0]))
    if az < 0:
        az += 2.0 * np.pi
    el = float(np.arctan2(d[2], np.hypot(d[0], d[1])))
    return az, el


def _slerp(d0: np.ndarray, d1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit directions (the great-circle path)."""
    dot = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-6:
        d = (1.0 - t) * d0 + t * d1
    else:
        so = np.sin(omega)
        d = (np.sin((1.0 - t) * omega) / so) * d0 + (np.sin(t * omega) / so) * d1
    n = np.linalg.norm(d)
    return d / n if n > 0 else d0


class AddRegionCommand(QUndoCommand):
    """Undo command for adding a polygon region."""
    def __init__(self, widget: RangeImageWidget, vertices: List[Tuple[int, int]], name: str):
        super().__init__(f"Add region '{name}'")
        self._widget = widget
        self._vertices = vertices
        self._name = name

    def redo(self):
        self._widget._regions.append(list(self._vertices))
        self._widget._region_names.append(self._name)
        self._widget.update()

    def undo(self):
        if self._widget._regions:
            self._widget._regions.pop()
            self._widget._region_names.pop()
            self._widget.update()


class RemoveRegionCommand(QUndoCommand):
    """Undo command for removing a polygon region."""
    def __init__(self, widget: RangeImageWidget, index: int):
        super().__init__(f"Remove region '{widget._region_names[index]}'")
        self._widget = widget
        self._index = index
        self._vertices = list(widget._regions[index])
        self._name = widget._region_names[index]

    def redo(self):
        self._widget._regions.pop(self._index)
        self._widget._region_names.pop(self._index)
        self._widget.update()

    def undo(self):
        self._widget._regions.insert(self._index, self._vertices)
        self._widget._region_names.insert(self._index, self._name)
        self._widget.update()


class MoveVertexCommand(QUndoCommand):
    """Undo command for moving a vertex."""
    def __init__(self, widget: RangeImageWidget, region_idx: int, vertex_idx: int,
                 old_pos: Tuple[int, int], new_pos: Tuple[int, int]):
        super().__init__("Move vertex")
        self._widget = widget
        self._ri = region_idx
        self._vi = vertex_idx
        self._old = old_pos
        self._new = new_pos

    def redo(self):
        self._widget._regions[self._ri][self._vi] = self._new
        self._widget.update()

    def undo(self):
        self._widget._regions[self._ri][self._vi] = self._old
        self._widget.update()


class MainWindow(QMainWindow):
    """Main application window for the FOV Boundary Tool."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FOV Boundary Tool")
        self.setMinimumSize(1200, 700)

        self._cloud: Optional[OrganizedCloud] = None
        self._undo_stack = QUndoStack(self)
        self._depth_min = 0.0
        self._depth_max = 50.0
        self._threshold = 5.0

        # Monotonic angular<->pixel axes (built lazily from the loaded cloud) used
        # to draw polygon edges as their true great-circle arcs.
        self._az_col_x: Optional[np.ndarray] = None
        self._az_col_y: Optional[np.ndarray] = None
        self._el_row_x: Optional[np.ndarray] = None
        self._el_row_y: Optional[np.ndarray] = None
        # Smooth per-index calibration (azimuth per column, elevation per row).
        # Defined over the whole grid, including empty (nan/inf) pixels.
        self._col_az: Optional[np.ndarray] = None
        self._row_el: Optional[np.ndarray] = None

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

        # Load section
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout(load_group)
        self._btn_load_pcd = QPushButton("Load PCD File")
        load_layout.addWidget(self._btn_load_pcd)
        left_layout.addWidget(load_group)

        # Depth threshold
        threshold_group = QGroupBox("Depth Threshold")
        threshold_layout = QVBoxLayout(threshold_group)
        self._lbl_threshold = QLabel("Threshold: 5.0 m")
        self._slider_threshold = QSlider(Qt.Horizontal)
        self._slider_threshold.setRange(0, 1000)
        self._slider_threshold.setValue(100)
        threshold_layout.addWidget(self._lbl_threshold)
        threshold_layout.addWidget(self._slider_threshold)
        left_layout.addWidget(threshold_group)

        # Drawing controls
        draw_group = QGroupBox("Draw Polygons")
        draw_layout = QVBoxLayout(draw_group)
        self._btn_new_polygon = QPushButton("New Polygon (click vertices)")
        self._lbl_draw_help = QLabel("Enter = finish, Esc = cancel · may click just outside FOV")
        self._lbl_draw_help.setStyleSheet("color: gray; font-size: 10px;")
        draw_layout.addWidget(self._btn_new_polygon)
        draw_layout.addWidget(self._lbl_draw_help)
        left_layout.addWidget(draw_group)

        # Regions list
        regions_group = QGroupBox("Regions")
        regions_layout = QVBoxLayout(regions_group)
        self._list_regions = QListWidget()
        self._btn_rename_region = QPushButton("Rename")
        self._btn_delete_region = QPushButton("Delete")
        self._btn_decompose = QPushButton("Decompose All")
        self._chk_show_decomp = QCheckBox("Show Decomposition")
        self._chk_show_decomp.setChecked(True)
        self._chk_show_arcs = QCheckBox("Show Great-Circle Arcs")
        self._chk_show_arcs.setChecked(True)
        self._chk_show_arcs.setToolTip(
            "Draw polygon edges as the true great-circle arcs that STVL's isInside "
            "test uses, instead of straight lines in the range image.")
        regions_layout.addWidget(self._list_regions)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_rename_region)
        btn_row.addWidget(self._btn_delete_region)
        regions_layout.addLayout(btn_row)
        regions_layout.addWidget(self._btn_decompose)
        regions_layout.addWidget(self._chk_show_decomp)
        regions_layout.addWidget(self._chk_show_arcs)
        left_layout.addWidget(regions_group)

        # Save/Load
        io_group = QGroupBox("Save / Load")
        io_layout = QVBoxLayout(io_group)
        self._btn_save = QPushButton("Save YAML")
        self._btn_load_yaml = QPushButton("Load YAML")
        io_layout.addWidget(self._btn_save)
        io_layout.addWidget(self._btn_load_yaml)
        left_layout.addWidget(io_group)

        left_layout.addStretch()

        # Right panel: range image
        self._range_widget = RangeImageWidget()
        self._range_widget.set_edge_interpolator(self._edge_arc_pixels)
        self._range_widget.set_convexity_checker(self._is_region_convex)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(self._range_widget, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Load a PCD file or grab from ROS topic to begin.")

    def _setup_menu(self):
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction("Load PCD...", self._on_load_pcd, "Ctrl+O")
        file_menu.addAction("Save YAML...", self._on_save, "Ctrl+S")
        file_menu.addAction("Load YAML...", self._on_load_yaml, "Ctrl+L")
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close, "Ctrl+Q")

        edit_menu = menu.addMenu("&Edit")
        undo_action = self._undo_stack.createUndoAction(self, "&Undo")
        undo_action.setShortcut("Ctrl+Z")
        redo_action = self._undo_stack.createRedoAction(self, "&Redo")
        redo_action.setShortcut("Ctrl+Y")
        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)

    def _connect_signals(self):
        self._btn_load_pcd.clicked.connect(self._on_load_pcd)
        self._slider_threshold.valueChanged.connect(self._on_threshold_changed)
        self._btn_new_polygon.clicked.connect(self._on_new_polygon)
        self._btn_rename_region.clicked.connect(self._on_rename_region)
        self._btn_delete_region.clicked.connect(self._on_delete_region)
        self._btn_decompose.clicked.connect(self._on_decompose)
        self._chk_show_decomp.toggled.connect(self._range_widget.toggle_decomposition_display)
        self._chk_show_arcs.toggled.connect(self._range_widget.toggle_arc_display)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_load_yaml.clicked.connect(self._on_load_yaml)
        self._range_widget.polygon_finished.connect(self._on_polygon_finished)
        self._range_widget.polygon_cancelled.connect(self._on_polygon_cancelled)
        self._range_widget.vertex_moved.connect(self._on_vertex_moved)
        self._range_widget.vertex_deleted.connect(self._on_vertex_deleted)

    def _on_load_pcd(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open PCD File", "", "PCD Files (*.pcd);;All Files (*)")
        if not filepath:
            return

        try:
            self._cloud = load_pcd(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PCD:\n{e}")
            return

        if self._cloud.height == 1 and self._cloud.width > 1:
            self._prompt_unfold_cloud()

        self._az_col_x = None
        self._update_range_image()
        self._status.showMessage(
            f"Loaded: {filepath} ({self._cloud.width}×{self._cloud.height})")

    def _prompt_unfold_cloud(self):
        """Ask the user for the number of channels/rings and reshape an
        unorganized (flat) cloud into an organized (height, width) grid.
        Loops on an invalid height so the user can retry without re-reading
        the file; cancelling leaves the cloud as its 1-row fallback."""
        total_points = self._cloud.width
        while True:
            height, ok = QInputDialog.getInt(
                self, "Unorganized Cloud",
                f"This cloud has {total_points} points and no organized grid.\n"
                "Enter the number of channels/rings (height) to unfold it into "
                "a matrix (Cancel to leave it as a single row):",
                value=1, min=1, max=total_points)
            if not ok:
                return
            try:
                self._cloud = unfold_cloud(self._cloud, height)
                return
            except ValueError as e:
                QMessageBox.warning(self, "Invalid Height", str(e))

    def _on_threshold_changed(self, value: int):
        # Map slider 0-1000 to depth range
        if self._cloud is not None:
            ranges = self._cloud.ranges
            valid = ranges[ranges > 0]
            if valid.size > 0:
                self._depth_max = float(np.percentile(valid, 99))
        self._threshold = (value / 1000.0) * self._depth_max
        self._lbl_threshold.setText(f"Threshold: {self._threshold:.1f} m")
        if self._cloud is not None:
            self._update_range_image()

    def _update_range_image(self):
        """Render the range image with depth threshold coloring."""
        if self._cloud is None:
            return

        if self._az_col_x is None:
            self._build_angular_axes()

        ranges = self._cloud.ranges  # (H, W)
        h, w = ranges.shape

        # Create RGB image
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Valid mask (non-zero, non-nan)
        valid = (ranges > 0) & np.isfinite(ranges)

        # Normalize ranges for coloring
        valid_ranges = ranges[valid]
        if valid_ranges.size == 0:
            self._range_widget.set_image(image)
            return

        max_r = float(np.percentile(valid_ranges, 99))
        if max_r <= 0:
            max_r = 1.0
        self._depth_max = max_r

        normalized = np.clip(ranges / max_r, 0, 1)

        # Apply threshold coloring:
        # Below threshold: warm colors (close objects - likely robot body)
        # Above threshold: cool colors (far objects - environment)
        below = valid & (ranges <= self._threshold)
        above = valid & (ranges > self._threshold)

        # Below threshold: yellow-red gradient
        if np.any(below):
            t = np.clip(ranges[below] / max(self._threshold, 0.01), 0, 1)
            image[below, 0] = 255  # R
            image[below, 1] = (255 * (1 - t)).astype(np.uint8)  # G: bright to dark
            image[below, 2] = 0  # B

        # Above threshold: blue-cyan gradient
        if np.any(above):
            t = np.clip((ranges[above] - self._threshold) / max(max_r - self._threshold, 0.01), 0, 1)
            image[above, 0] = 0  # R
            image[above, 1] = (100 * (1 - t)).astype(np.uint8)  # G
            image[above, 2] = (100 + 155 * (1 - t)).astype(np.uint8)  # B

        # Ensure contiguous
        image = np.ascontiguousarray(image)
        self._range_widget.set_image(image)

    def _on_new_polygon(self):
        if self._cloud is None:
            QMessageBox.information(self, "No Data", "Load a point cloud first.")
            return
        self._range_widget.start_drawing()
        self._status.showMessage("Drawing: click to place vertices. Enter = finish, Esc = cancel.")

    def _is_region_convex(self, region_pixels: List[Tuple[int, int]]) -> bool:
        """True iff the region, projected to (azimuth, elevation), is a valid
        convex half-plane cone - i.e. what STVL's `ConvexCone::fromAngularVertices`
        would accept as a single half-plane region, not merely flat-pixel convex."""
        if len(region_pixels) < 3:
            return False
        angular = [self._pixel_to_angular_model(px, py) for px, py in region_pixels]
        return is_convex_angular_polygon(angular)

    def _refresh_region_list(self):
        """Rebuild the region list from the widget's current regions, marking
        any that are not (spherically) convex."""
        idx = self._list_regions.currentRow()
        self._list_regions.clear()
        for name, region in zip(self._range_widget.region_names, self._range_widget.regions):
            convex = self._is_region_convex(region)
            label = name if convex else f"{name} ⚠ non-convex"
            self._list_regions.addItem(label)
            if not convex:
                self._list_regions.item(self._list_regions.count() - 1).setForeground(
                    QBrush(QColor(255, 80, 80)))
        if 0 <= idx < self._list_regions.count():
            self._list_regions.setCurrentRow(idx)

    def _on_polygon_finished(self):
        """Called when user finishes a polygon (Enter key)."""
        regions = self._range_widget.regions
        if regions:
            name = f"region_{len(regions)}"
            self._refresh_region_list()
            self._status.showMessage(f"Region '{name}' created with {len(regions[-1])} vertices.")
        # Stay in drawing mode for next polygon
        self._range_widget.start_drawing()

    def _on_polygon_cancelled(self):
        self._status.showMessage("Polygon cancelled.")

    def _on_rename_region(self):
        idx = self._list_regions.currentRow()
        if idx < 0:
            return
        old_name = self._range_widget.region_names[idx]
        new_name, ok = QInputDialog.getText(self, "Rename Region", "New name:", text=old_name)
        if ok and new_name:
            self._range_widget._region_names[idx] = new_name
            self._refresh_region_list()

    def _on_delete_region(self):
        idx = self._list_regions.currentRow()
        if idx < 0:
            return
        cmd = RemoveRegionCommand(self._range_widget, idx)
        self._undo_stack.push(cmd)
        self._refresh_region_list()
        self._status.showMessage("Region deleted.")

    def _on_decompose(self):
        """Run Hertel-Mehlhorn convex decomposition on all regions."""
        if not self._range_widget.regions:
            QMessageBox.information(self, "No Regions", "Draw some polygon regions first.")
            return

        all_decomposed = []
        total_convex = 0
        non_convex_inputs = 0

        for region_pixels in self._range_widget.regions:
            if not self._is_region_convex(region_pixels):
                non_convex_inputs += 1
            # Decomposition needs to test convexity in the (azimuth, elevation)
            # half-plane sense, not flat pixel space - pass the conversion in.
            polygon = [(float(x), float(y)) for x, y in region_pixels]
            convex_parts = hertel_mehlhorn(polygon, self._pixel_to_angular_model)
            # Convert back to int pixel coords for display
            convex_parts_int = [
                [(int(round(x)), int(round(y))) for x, y in poly]
                for poly in convex_parts
            ]
            all_decomposed.append(convex_parts_int)
            total_convex += len(convex_parts_int)

        self._range_widget.set_decomposition(all_decomposed)
        self._status.showMessage(
            f"Decomposition complete: {len(self._range_widget.regions)} regions → "
            f"{total_convex} convex polygons ({non_convex_inputs} were non-convex and got split).")

    def _on_vertex_moved(self, ri: int, vi: int, new_x: int, new_y: int):
        # Could track for undo - simplified here
        self._refresh_region_list()
        self._status.showMessage(f"Vertex moved in region {ri}.")

    def _on_vertex_deleted(self, ri: int, vi: int):
        self._refresh_region_list()
        self._status.showMessage(f"Vertex deleted from region {ri}.")

    def _pixel_to_angular(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to (azimuth, elevation) in radians.

        Azimuth is in [0, 2pi] range. Elevation is in [-pi/2, pi/2].
        If the point at (px, py) is invalid (NaN/zero), searches nearby pixels
        for a valid point, falling back to linear interpolation from the cloud's
        angular extent.
        """
        if self._cloud is None:
            return (0.0, 0.0)

        # Out-of-FOV vertex (drawn beyond the image): use the extrapolating model
        # so boundaries can extend past the grid instead of being clamped to it.
        if self._col_az is not None and not (
                0 <= px < self._cloud.width and 0 <= py < self._cloud.height):
            return self._pixel_to_angular_model(px, py)

        # Clamp to valid range
        py = max(0, min(py, self._cloud.height - 1))
        px = max(0, min(px, self._cloud.width - 1))

        point = self._cloud.xyz[py, px]
        x, y, z = float(point[0]), float(point[1]), float(point[2])

        if self._is_valid_point(x, y, z):
            az = float(np.arctan2(y, x))
            if az < 0:
                az += 2.0 * np.pi
            el = float(np.arctan2(z, np.sqrt(x**2 + y**2)))
            return (az, el)

        # Point is invalid (empty/nan/inf pixel). Prefer the smooth calibration
        # model, which is defined everywhere, over snapping to a nearby valid
        # return (snapping warps boundaries drawn through empty space).
        if self._col_az is not None:
            return self._pixel_to_angular_model(px, py)

        # Point is invalid — search neighbors in expanding radius
        for radius in range(1, 20):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) != radius and abs(dx) != radius:
                        continue  # only check border of the search square
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self._cloud.height and 0 <= nx < self._cloud.width:
                        p = self._cloud.xyz[ny, nx]
                        xn, yn, zn = float(p[0]), float(p[1]), float(p[2])
                        if self._is_valid_point(xn, yn, zn):
                            az = float(np.arctan2(yn, xn))
                            if az < 0:
                                az += 2.0 * np.pi
                            el = float(np.arctan2(zn, np.sqrt(xn**2 + yn**2)))
                            return (az, el)

        # Last resort: linear interpolation from cloud angular extent
        return self._pixel_to_angular_fallback(px, py)

    @staticmethod
    def _is_valid_point(x: float, y: float, z: float) -> bool:
        """Check if a 3D point is valid (not NaN, not inf, not all zeros)."""
        if x == 0.0 and y == 0.0 and z == 0.0:
            return False
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            return False
        return True

    def _pixel_to_angular_fallback(self, px: int, py: int) -> Tuple[float, float]:
        """Estimate angular coords from the cloud's valid angular extent."""
        azimuths = self._cloud.azimuths
        elevations = self._cloud.elevations
        valid = np.isfinite(azimuths) & np.isfinite(elevations) & (self._cloud.ranges > 0)

        if not np.any(valid):
            # Absolute fallback
            az = (px / self._cloud.width) * 2 * np.pi - np.pi
            el = (py / self._cloud.height) * (np.pi / 4) - (np.pi / 8)
            return (az, el)

        # Use per-column median azimuth and per-row median elevation
        col_azimuths = azimuths[:, px]
        col_valid = np.isfinite(col_azimuths) & (self._cloud.ranges[:, px] > 0)
        if np.any(col_valid):
            az = float(np.median(col_azimuths[col_valid]))
        else:
            valid_az = azimuths[valid]
            az_min, az_max = float(valid_az.min()), float(valid_az.max())
            az = az_min + (px / self._cloud.width) * (az_max - az_min)

        row_elevations = elevations[py, :]
        row_valid = np.isfinite(row_elevations) & (self._cloud.ranges[py, :] > 0)
        if np.any(row_valid):
            el = float(np.median(row_elevations[row_valid]))
        else:
            valid_el = elevations[valid]
            el_min, el_max = float(valid_el.min()), float(valid_el.max())
            el = el_min + (py / self._cloud.height) * (el_max - el_min)

        return (az, el)

    def _build_angular_axes(self):
        """Precompute monotonic per-column azimuth and per-row elevation lookup
        axes so angular coords can be inverted back to pixel coords for arc drawing."""
        if self._cloud is None:
            return
        az = self._cloud.azimuths
        el = self._cloud.elevations
        rng = self._cloud.ranges
        valid = np.isfinite(az) & np.isfinite(el) & (rng > 0)

        az_masked = np.where(valid, az, np.nan)
        el_masked = np.where(valid, el, np.nan)
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                col_vals = np.nanmedian(az_masked, axis=0)  # (W,)
                row_vals = np.nanmedian(el_masked, axis=1)  # (H,)

        col_vals = np.unwrap(self._fill_nan(col_vals))
        row_vals = self._fill_nan(row_vals)

        # Smooth per-index calibration (used for a consistent pixel<->angle model).
        self._col_az = col_vals
        self._row_el = row_vals

        self._az_col_x, self._az_col_y = self._as_increasing(
            col_vals, np.arange(self._cloud.width, dtype=float))
        self._el_row_x, self._el_row_y = self._as_increasing(
            row_vals, np.arange(self._cloud.height, dtype=float))

    @staticmethod
    def _fill_nan(a: np.ndarray) -> np.ndarray:
        """Linearly fill NaN entries in a 1D array along its index."""
        a = np.asarray(a, dtype=float).copy()
        idx = np.arange(len(a))
        good = np.isfinite(a)
        if not np.any(good):
            return np.zeros_like(a)
        a[~good] = np.interp(idx[~good], idx[good], a[good])
        return a

    @staticmethod
    def _as_increasing(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort (x, y) so x is strictly increasing (required by np.interp)."""
        order = np.argsort(x, kind='stable')
        return x[order], y[order]

    @staticmethod
    def _interp_extrap(x: float, xp: np.ndarray, fp: np.ndarray) -> float:
        """Linear interpolation that also linearly *extrapolates* beyond the ends
        (np.interp clamps). Lets vertices/arcs extend just outside the FOV grid."""
        x = float(x)
        if x < xp[0]:
            slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
            return float(fp[0] + (x - xp[0]) * slope)
        if x > xp[-1]:
            slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            return float(fp[-1] + (x - xp[-1]) * slope)
        return float(np.interp(x, xp, fp))

    def _pixel_to_angular_model(self, px: float, py: float) -> Tuple[float, float]:
        """Smooth, hole-free pixel -> (azimuth, elevation) using the per-column /
        per-row calibration. Unlike `_pixel_to_angular`, this is defined everywhere
        (also over nan/inf pixels) and is the exact inverse of `_angular_to_pixel`,
        so arcs traced with it are clean and independent of local data occupancy.
        Pixels outside the grid are linearly extrapolated so boundaries can be drawn
        slightly beyond the FOV (e.g. to fully enclose the outermost pixel row)."""
        if self._col_az is None:
            return self._pixel_to_angular(int(round(px)), int(round(py)))
        W = len(self._col_az)
        H = len(self._row_el)
        az = self._interp_extrap(px, np.arange(W, dtype=float), self._col_az) % (2.0 * np.pi)
        el = self._interp_extrap(py, np.arange(H, dtype=float), self._row_el)
        return (az, el)

    def _angular_to_pixel(self, az: float, el: float) -> Tuple[float, float]:
        """Inverse of `_pixel_to_angular`: (azimuth, elevation) -> pixel coords.
        Extrapolates beyond the grid so out-of-FOV angles map to out-of-image pixels."""
        if self._az_col_x is None:
            return (0.0, 0.0)
        # Bring azimuth into the same (unwrapped) branch as the column axis.
        center = 0.5 * (self._az_col_x[0] + self._az_col_x[-1])
        a = az
        while a - center > np.pi:
            a -= 2.0 * np.pi
        while center - a > np.pi:
            a += 2.0 * np.pi
        px = self._interp_extrap(a, self._az_col_x, self._az_col_y)
        py = self._interp_extrap(el, self._el_row_x, self._el_row_y)
        return (px, py)

    def _angular_to_pixel_int(self, az: float, el: float) -> Tuple[int, int]:
        """Integer-pixel variant of `_angular_to_pixel` for placing loaded vertices."""
        px, py = self._angular_to_pixel(az, el)
        return (int(round(px)), int(round(py)))

    def _edge_arc_pixels(self, v0: Tuple[int, int], v1: Tuple[int, int],
                         n: int = 24) -> List[Tuple[float, float]]:
        """Trace the great-circle arc between two pixel vertices, returning a list
        of intermediate pixel points. This matches STVL's `isInside` edge semantics
        (a half-plane test against the great circle through the two edge directions)."""
        if self._cloud is None or self._az_col_x is None:
            return [(float(v0[0]), float(v0[1])), (float(v1[0]), float(v1[1]))]
        a0 = self._pixel_to_angular_model(v0[0], v0[1])
        a1 = self._pixel_to_angular_model(v1[0], v1[1])
        d0 = _angular_to_dir(*a0)
        d1 = _angular_to_dir(*a1)
        pts: List[Tuple[float, float]] = []
        for t in np.linspace(0.0, 1.0, n):
            d = _slerp(d0, d1, float(t))
            az, el = _dir_to_angular(d)
            pts.append(self._angular_to_pixel(az, el))
        # Pin endpoints to the exact clicked pixels so arcs meet the drawn vertices.
        pts[0] = (float(v0[0]), float(v0[1]))
        pts[-1] = (float(v1[0]), float(v1[1]))
        return pts

    def _on_save(self):
        if not self._range_widget.regions:
            QMessageBox.information(self, "Nothing to Save", "No regions defined.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Blind Spot Config", "fov_blind_spots.yaml",
            "YAML Files (*.yaml *.yml);;All Files (*)")
        if not filepath:
            return

        # Run decomposition if not done
        if not self._range_widget._decomposed:
            self._on_decompose()

        config = BlindSpotConfig(
            sensor_frame=self._cloud.frame_id if self._cloud else "sensor",
            cloud_width=self._cloud.width if self._cloud else 0,
            cloud_height=self._cloud.height if self._cloud else 0,
        )

        for i, (region_pixels, name) in enumerate(
                zip(self._range_widget.regions, self._range_widget.region_names)):
            # Convert pixel vertices to angular
            orig_angular = [self._pixel_to_angular(px, py) for px, py in region_pixels]

            # Convert decomposed polygons to angular
            convex_angular = []
            if i < len(self._range_widget._decomposed):
                for poly_pixels in self._range_widget._decomposed[i]:
                    poly_angular = [self._pixel_to_angular(px, py) for px, py in poly_pixels]
                    convex_angular.append(poly_angular)

            region = BlindSpotRegion(
                name=name,
                original_vertices=orig_angular,
                convex_polygons=convex_angular,
            )
            config.regions.append(region)

        try:
            save_config(config, filepath)
            param_str = format_polygons_as_param_string(config)
            print(f"\n=== STVL obstruction_polygons parameter ===\n{param_str}\n")
            self._status.showMessage(f"Saved to {filepath}")
            QMessageBox.information(
                self, "Saved",
                f"Config saved to:\n{filepath}\n\n"
                f"STVL parameter string (also printed to terminal):\n\n{param_str}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")

    def _on_load_yaml(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Blind Spot Config", "",
            "YAML Files (*.yaml *.yml);;All Files (*)")
        if not filepath:
            return

        try:
            config = load_config(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{e}")
            return

        # Convert angular coordinates back to pixel coordinates
        # This requires knowing the cloud dimensions and angular mapping
        if self._cloud is None:
            QMessageBox.warning(self, "No Cloud Loaded",
                                "Load a point cloud first so polygon coordinates "
                                "can be mapped to pixels.")
            return

        regions_pixels = []
        region_names = []
        decomposed_pixels = []

        for region in config.regions:
            # Convert angular vertices to pixel coords (smooth calibration model)
            pixels = [self._angular_to_pixel_int(az, el)
                      for az, el in region.original_vertices]
            regions_pixels.append(pixels)
            region_names.append(region.name)

            # Convert decomposed polygons
            decomp = []
            for poly in region.convex_polygons:
                poly_pixels = [self._angular_to_pixel_int(az, el) for az, el in poly]
                decomp.append(poly_pixels)
            decomposed_pixels.append(decomp)

        self._range_widget.set_regions(regions_pixels, region_names)
        self._range_widget.set_decomposition(decomposed_pixels)

        self._refresh_region_list()

        self._status.showMessage(
            f"Loaded {len(config.regions)} regions from {filepath}")

