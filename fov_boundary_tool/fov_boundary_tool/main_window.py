"""Main window for the FOV Boundary Tool."""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QListWidget, QMessageBox,
    QInputDialog, QAction, QToolBar, QStatusBar, QSplitter,
    QGroupBox, QCheckBox, QApplication, QUndoStack, QUndoCommand
)
from PyQt5.QtCore import Qt

from .range_image_widget import RangeImageWidget
from .cloud_loader import OrganizedCloud, load_pcd
from .convex_decomposition import hertel_mehlhorn
from .config_io import BlindSpotConfig, BlindSpotRegion, save_config, load_config, format_polygons_as_param_string


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
        self._btn_load_topic = QPushButton("Grab from ROS Topic")
        load_layout.addWidget(self._btn_load_pcd)
        load_layout.addWidget(self._btn_load_topic)
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
        self._lbl_draw_help = QLabel("Enter = finish, Esc = cancel")
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
        regions_layout.addWidget(self._list_regions)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_rename_region)
        btn_row.addWidget(self._btn_delete_region)
        regions_layout.addLayout(btn_row)
        regions_layout.addWidget(self._btn_decompose)
        regions_layout.addWidget(self._chk_show_decomp)
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
        self._btn_load_topic.clicked.connect(self._on_load_topic)
        self._slider_threshold.valueChanged.connect(self._on_threshold_changed)
        self._btn_new_polygon.clicked.connect(self._on_new_polygon)
        self._btn_rename_region.clicked.connect(self._on_rename_region)
        self._btn_delete_region.clicked.connect(self._on_delete_region)
        self._btn_decompose.clicked.connect(self._on_decompose)
        self._chk_show_decomp.toggled.connect(self._range_widget.toggle_decomposition_display)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_load_yaml.clicked.connect(self._on_load_yaml)
        self._range_widget.polygon_finished.connect(self._on_polygon_finished)
        self._range_widget.polygon_cancelled.connect(self._on_polygon_cancelled)
        self._range_widget.vertex_moved.connect(self._on_vertex_moved)

    def _on_load_pcd(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open PCD File", "", "PCD Files (*.pcd);;All Files (*)")
        if not filepath:
            return

        try:
            self._cloud = load_pcd(filepath)
            self._update_range_image()
            self._status.showMessage(
                f"Loaded: {filepath} ({self._cloud.width}×{self._cloud.height})")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PCD:\n{e}")

    def _on_load_topic(self):
        """Grab a single frame from a ROS2 PointCloud2 topic."""
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import PointCloud2
            from .cloud_loader import load_from_ros_msg_fast
        except ImportError:
            QMessageBox.warning(self, "ROS2 Not Available",
                                "rclpy is not available. Please source your ROS2 workspace.")
            return

        topic, ok = QInputDialog.getText(self, "ROS Topic", "PointCloud2 topic:",
                                         text="/points3d")
        if not ok or not topic:
            return

        self._status.showMessage(f"Waiting for message on {topic}...")
        QApplication.processEvents()

        try:
            rclpy.init()
            node = Node('fov_boundary_tool_grabber')

            msg_received = [None]

            def callback(msg):
                msg_received[0] = msg

            sub = node.create_subscription(PointCloud2, topic, callback, 1)

            # Spin until we get a message (with timeout)
            import time
            start = time.time()
            while msg_received[0] is None and (time.time() - start) < 10.0:
                rclpy.spin_once(node, timeout_sec=0.1)

            node.destroy_node()
            rclpy.shutdown()

            if msg_received[0] is None:
                QMessageBox.warning(self, "Timeout",
                                    f"No message received on {topic} within 10 seconds.")
                return

            self._cloud = load_from_ros_msg_fast(msg_received[0])
            self._update_range_image()
            self._status.showMessage(
                f"Grabbed from {topic} ({self._cloud.width}×{self._cloud.height})")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to grab from topic:\n{e}")
            try:
                rclpy.shutdown()
            except:
                pass

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

    def _on_polygon_finished(self):
        """Called when user finishes a polygon (Enter key)."""
        regions = self._range_widget.regions
        if regions:
            name = f"region_{len(regions)}"
            self._list_regions.addItem(name)
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
            self._list_regions.item(idx).setText(new_name)

    def _on_delete_region(self):
        idx = self._list_regions.currentRow()
        if idx < 0:
            return
        cmd = RemoveRegionCommand(self._range_widget, idx)
        self._undo_stack.push(cmd)
        self._list_regions.takeItem(idx)
        self._status.showMessage("Region deleted.")

    def _on_decompose(self):
        """Run Hertel-Mehlhorn convex decomposition on all regions."""
        if not self._range_widget.regions:
            QMessageBox.information(self, "No Regions", "Draw some polygon regions first.")
            return

        all_decomposed = []
        total_convex = 0

        for region_pixels in self._range_widget.regions:
            # Convert pixel coords to (azimuth, elevation) for decomposition
            # But decomposition works in any 2D space, so we can work in pixel space
            polygon = [(float(x), float(y)) for x, y in region_pixels]
            convex_parts = hertel_mehlhorn(polygon)
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
            f"{total_convex} convex polygons.")

    def _on_vertex_moved(self, ri: int, vi: int, new_x: int, new_y: int):
        # Could track for undo - simplified here
        self._status.showMessage(f"Vertex moved in region {ri}.")

    def _pixel_to_angular(self, px: int, py: int) -> Tuple[float, float]:
        """Convert pixel coordinates to (azimuth, elevation) in radians.

        Azimuth is in [0, 2pi] range. Elevation is in [-pi/2, pi/2].
        If the point at (px, py) is invalid (NaN/zero), searches nearby pixels
        for a valid point, falling back to linear interpolation from the cloud's
        angular extent.
        """
        if self._cloud is None:
            return (0.0, 0.0)

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
            # Convert angular vertices to pixel coords
            pixels = [self._angular_to_pixel(az, el)
                      for az, el in region.original_vertices]
            regions_pixels.append(pixels)
            region_names.append(region.name)

            # Convert decomposed polygons
            decomp = []
            for poly in region.convex_polygons:
                poly_pixels = [self._angular_to_pixel(az, el) for az, el in poly]
                decomp.append(poly_pixels)
            decomposed_pixels.append(decomp)

        self._range_widget.set_regions(regions_pixels, region_names)
        self._range_widget.set_decomposition(decomposed_pixels)

        # Update list widget
        self._list_regions.clear()
        for name in region_names:
            self._list_regions.addItem(name)

        self._status.showMessage(
            f"Loaded {len(config.regions)} regions from {filepath}")

    def _angular_to_pixel(self, azimuth: float, elevation: float) -> Tuple[int, int]:
        """Convert (azimuth, elevation) in radians to pixel coordinates.

        Uses the cloud's actual angular layout to find the nearest pixel.
        """
        if self._cloud is None:
            return (0, 0)

        # Compute angular maps
        azimuths = self._cloud.azimuths  # (H, W)
        elevations = self._cloud.elevations  # (H, W)

        # Find nearest valid pixel
        valid = self._cloud.ranges > 0
        if not np.any(valid):
            return (0, 0)

        # Angular distance
        az_diff = np.abs(azimuths - azimuth)
        # Handle wraparound
        az_diff = np.minimum(az_diff, 2 * np.pi - az_diff)
        el_diff = np.abs(elevations - elevation)

        dist = az_diff + el_diff
        dist[~valid] = np.inf

        idx = np.unravel_index(np.argmin(dist), dist.shape)
        return (int(idx[1]), int(idx[0]))  # (x=col, y=row)
