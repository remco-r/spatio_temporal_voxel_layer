"""Range image widget for displaying and interacting with the depth image."""

import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF, QRectF
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QPolygonF,
                          QBrush, QMouseEvent, QKeyEvent)
from typing import List, Optional, Tuple, Callable


class RangeImageWidget(QWidget):
    """Widget that displays a range image and handles polygon drawing interaction."""

    vertex_placed = pyqtSignal(int, int)  # pixel x, y
    polygon_finished = pyqtSignal()
    polygon_cancelled = pyqtSignal()
    vertex_moved = pyqtSignal(int, int, int, int)  # region_idx, vertex_idx, new_x, new_y
    vertex_deleted = pyqtSignal(int, int)  # region_idx, vertex_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 200)

        self._pixmap: Optional[QPixmap] = None
        self._image_data: Optional[np.ndarray] = None

        # Drawing state
        self._current_vertices: List[Tuple[int, int]] = []
        self._mouse_pos: Optional[Tuple[int, int]] = None
        self._is_drawing = False

        # Completed regions (original user-drawn polygons)
        self._regions: List[List[Tuple[int, int]]] = []
        self._region_names: List[str] = []

        # Convex decomposition results
        self._decomposed: List[List[List[Tuple[int, int]]]] = []

        # Edit state
        self._dragging: Optional[Tuple[int, int]] = None  # (region_idx, vertex_idx)
        self._drag_start: Optional[Tuple[int, int]] = None

        # Display
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._show_decomposition = True

        # Great-circle arc rendering. When set, `_edge_interp(v0, v1)` returns a
        # list of image-space pixel points tracing the true great-circle arc that
        # STVL's `isInside` half-plane test actually uses for the edge v0->v1.
        self._edge_interp: Optional[Callable[[Tuple[int, int], Tuple[int, int]],
                                             List[Tuple[float, float]]]] = None
        self._show_arcs = True

        # When set, `_is_convex_fn(region)` reports whether a completed region is
        # a valid convex half-plane cone (spherical convexity, not flat-pixel).
        # Non-convex regions get a warning outline in paintEvent.
        self._is_convex_fn: Optional[Callable[[List[Tuple[int, int]]], bool]] = None

        # Allow vertices to be placed this fraction beyond each image edge, so a
        # boundary can be drawn slightly outside the FOV (e.g. to fully enclose the
        # outermost pixel row/column).
        self._oob_margin_frac = 0.02

        # Width in pixels of a full 2pi azimuth turn, when the cloud covers a full
        # panorama. Arc geometry is periodic in x with this period, so a region whose
        # edge crosses the 0/2pi seam is also drawn one period to each side, showing
        # it leave one edge of the image and re-enter the other. None for a partial
        # FOV, which is not periodic and must not be replicated.
        self._wrap_period: Optional[float] = None

    def set_image(self, image_data: np.ndarray):
        """Set the range image to display. Expects RGB uint8 array (H, W, 3)."""
        self._image_data = image_data
        h, w = image_data.shape[:2]
        if image_data.ndim == 2:
            # Grayscale
            qimg = QImage(image_data.data, w, h, w, QImage.Format_Grayscale8)
        else:
            bytes_per_line = 3 * w
            qimg = QImage(image_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._fit_to_widget()
        self.update()

    def _fit_to_widget(self):
        """Compute scale and offset to fit image in widget."""
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return
        scale_x = ww / pw
        scale_y = wh / ph
        # Shrink so there is a clickable margin all around the image, sized to the
        # allowed out-of-FOV placement margin. This lets vertices be placed outside
        # the FOV on every side (including the azimuth edges), not just in letterbox.
        m = self._oob_margin_frac
        self._scale = min(scale_x, scale_y) / (1.0 + 2.0 * m)
        self._offset_x = (ww - pw * self._scale) / 2
        self._offset_y = (wh - ph * self._scale) / 2

    def _widget_to_image(self, wx: int, wy: int) -> Tuple[int, int]:
        """Convert widget coordinates to image pixel coordinates."""
        ix = int((wx - self._offset_x) / self._scale)
        iy = int((wy - self._offset_y) / self._scale)
        return ix, iy

    def _image_to_widget(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert image pixel coordinates to widget coordinates."""
        wx = ix * self._scale + self._offset_x
        wy = iy * self._scale + self._offset_y
        return wx, wy

    def _clamp_image_coords(self, ix: int, iy: int) -> Tuple[int, int]:
        """Clamp image coords to the allowed range, permitting a margin outside the
        image so vertices can be placed slightly beyond the FOV."""
        if self._pixmap is None:
            return ix, iy
        w, h = self._pixmap.width(), self._pixmap.height()
        mx = max(2, int(round(self._oob_margin_frac * w)))
        my = max(2, int(round(self._oob_margin_frac * h)))
        ix = max(-mx, min(ix, w - 1 + mx))
        iy = max(-my, min(iy, h - 1 + my))
        return ix, iy

    def start_drawing(self):
        """Enter polygon drawing mode."""
        self._is_drawing = True
        self._current_vertices = []
        self.setCursor(Qt.CrossCursor)

    def stop_drawing(self):
        """Exit polygon drawing mode."""
        self._is_drawing = False
        self._current_vertices = []
        self._mouse_pos = None
        self.setCursor(Qt.ArrowCursor)
        self.update()

    @property
    def regions(self) -> List[List[Tuple[int, int]]]:
        return self._regions

    @property
    def region_names(self) -> List[str]:
        return self._region_names

    def set_regions(self, regions: List[List[Tuple[int, int]]], names: List[str]):
        """Set the polygon regions (for loading)."""
        self._regions = regions
        self._region_names = names
        self.update()

    def set_decomposition(self, decomposed: List[List[List[Tuple[int, int]]]]):
        """Set the convex decomposition results for visualization."""
        self._decomposed = decomposed
        self.update()

    def toggle_decomposition_display(self, show: bool):
        self._show_decomposition = show
        self.update()

    def set_edge_interpolator(self, fn: Optional[Callable[[Tuple[int, int], Tuple[int, int]],
                                                          List[Tuple[float, float]]]]):
        """Provide a callback mapping an edge (v0, v1) in image pixels to a list of
        intermediate image-pixel points tracing the true great-circle arc."""
        self._edge_interp = fn
        self.update()

    def toggle_arc_display(self, show: bool):
        self._show_arcs = show
        self.update()

    def set_wrap_period(self, period: Optional[float]):
        """Set the pixel period of a full 2pi azimuth turn, or None when the cloud is
        not a full panorama and so must not be drawn wrapped."""
        self._wrap_period = period
        self.update()

    def _wrap_offsets(self) -> List[float]:
        """Image-space x offsets at which periodic arc geometry is redrawn."""
        if self._wrap_period is None:
            return [0.0]
        return [-self._wrap_period, 0.0, self._wrap_period]

    def set_convexity_checker(self, fn: Optional[Callable[[List[Tuple[int, int]]], bool]]):
        """Provide a callback reporting whether a completed region (list of pixel
        vertices) is a valid convex half-plane cone. Used to flag non-convex
        regions in paintEvent."""
        self._is_convex_fn = fn
        self.update()

    def _edge_image_points(self, v0: Tuple[int, int],
                           v1: Tuple[int, int]) -> List[Tuple[float, float]]:
        """Image-space points along edge v0->v1, following the great-circle arc if an
        interpolator is available, else a straight chord."""
        if self._show_arcs and self._edge_interp is not None:
            try:
                return [(float(px), float(py)) for px, py in self._edge_interp(v0, v1)]
            except Exception:
                pass
        return [(float(v0[0]), float(v0[1])), (float(v1[0]), float(v1[1]))]

    def _chained_image_points(self, verts: List[Tuple[int, int]],
                              closed: bool) -> List[Tuple[float, float]]:
        """One continuous image-space polyline along a vertex chain.

        Each edge's arc is unwrapped independently, starting from its own first
        vertex, so two consecutive edges can legitimately end up a whole azimuth turn
        apart. Chain them here - shift every edge by whole periods so it continues
        where the previous one ended - otherwise a seam-crossing ring would still
        break at an edge junction even though no single edge does."""
        n = len(verts)
        edges = n if closed else n - 1
        out: List[Tuple[float, float]] = []
        prev_end: Optional[Tuple[float, float]] = None
        for i in range(edges):
            pts = self._edge_image_points(verts[i], verts[(i + 1) % n])
            if self._wrap_period and prev_end is not None:
                turns = round((pts[0][0] - prev_end[0]) / self._wrap_period)
                if turns:
                    pts = [(px - turns * self._wrap_period, py) for px, py in pts]
            out.extend(pts[:-1])  # drop last to avoid duplicating shared vertices
            prev_end = pts[-1]
        if not closed and prev_end is not None:
            out.append(prev_end)
        return out

    def _ring_polygons(self, verts: List[Tuple[int, int]]) -> List[QPolygonF]:
        """Closed QPolygonFs tracing arcs around the vertex ring - one per wrap
        offset, so a ring whose edges cross the 0/2pi seam is drawn leaving one side
        of the panorama and re-entering the other, instead of as one streak straight
        across it. Copies that fall entirely off-canvas simply do not show."""
        ring = self._chained_image_points(verts, closed=True)
        polys: List[QPolygonF] = []
        for dx in self._wrap_offsets():
            poly = QPolygonF()
            for px, py in ring:
                poly.append(QPointF(*self._image_to_widget(px + dx, py)))
            polys.append(poly)
        return polys

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_to_widget()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self._pixmap:
            # Draw scaled image
            painter.save()
            painter.translate(self._offset_x, self._offset_y)
            painter.scale(self._scale, self._scale)
            painter.drawPixmap(0, 0, self._pixmap)
            painter.restore()

            # Outline the FOV (image extent) so the clickable out-of-FOV margin is visible.
            x0, y0 = self._image_to_widget(0, 0)
            x1, y1 = self._image_to_widget(self._pixmap.width(), self._pixmap.height())
            painter.setPen(QPen(QColor(180, 180, 180, 160), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        # Draw completed regions
        colors = [
            QColor(255, 100, 100, 80), QColor(100, 255, 100, 80),
            QColor(100, 100, 255, 80), QColor(255, 255, 100, 80),
            QColor(255, 100, 255, 80), QColor(100, 255, 255, 80),
        ]

        for ri, region in enumerate(self._regions):
            color = colors[ri % len(colors)]
            is_convex = (len(region) < 3 or self._is_convex_fn is None
                         or self._is_convex_fn(region))
            if is_convex:
                outline_pen = QPen(color.darker(150), 2)
                vertex_color = color.darker(120)
            else:
                # Warn: this region isn't a valid single half-plane cone and
                # will be split into multiple convex parts on decomposition.
                outline_pen = QPen(QColor(255, 40, 40), 3, Qt.DashLine)
                vertex_color = QColor(255, 40, 40)
            painter.setPen(outline_pen)

            # Draw filled polygon (edges follow great-circle arcs)
            if len(region) >= 3:
                painter.setBrush(QBrush(color))
                for poly in self._ring_polygons(region):
                    painter.drawPolygon(poly)

            # Draw vertices
            painter.setBrush(QBrush(vertex_color))
            for vx, vy in region:
                wx, wy = self._image_to_widget(vx, vy)
                painter.drawEllipse(QPointF(wx, wy), 4, 4)

        # Draw convex decomposition overlay
        if self._show_decomposition and self._decomposed:
            decomp_colors = [
                QColor(255, 50, 50, 40), QColor(50, 255, 50, 40),
                QColor(50, 50, 255, 40), QColor(255, 200, 50, 40),
                QColor(200, 50, 255, 40), QColor(50, 255, 200, 40),
            ]
            ci = 0
            for region_decomp in self._decomposed:
                for convex_poly in region_decomp:
                    dc = decomp_colors[ci % len(decomp_colors)]
                    painter.setPen(QPen(dc.darker(100), 1.5, Qt.DashLine))
                    painter.setBrush(QBrush(dc))
                    for poly in self._ring_polygons(convex_poly):
                        painter.drawPolygon(poly)
                    ci += 1

        # Draw current polygon being drawn
        if self._is_drawing and self._current_vertices:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            # Draw edges (following great-circle arcs)
            if len(self._current_vertices) >= 2:
                run = self._chained_image_points(self._current_vertices, closed=False)
                for dx in self._wrap_offsets():
                    pts = [QPointF(*self._image_to_widget(px + dx, py))
                           for px, py in run]
                    for j in range(len(pts) - 1):
                        painter.drawLine(pts[j], pts[j + 1])

            # Draw closing line preview (last vertex -> mouse -> first vertex)
            if self._mouse_pos and len(self._current_vertices) >= 1:
                last = self._current_vertices[-1]
                lx, ly = self._image_to_widget(*last)
                mx, my = self._mouse_pos

                # Line from last vertex to cursor
                painter.setPen(QPen(QColor(0, 255, 0, 180), 1.5, Qt.DashLine))
                painter.drawLine(QPointF(lx, ly), QPointF(mx, my))

                # Closing line: cursor to first vertex
                if len(self._current_vertices) >= 2:
                    first = self._current_vertices[0]
                    fx, fy = self._image_to_widget(*first)
                    painter.setPen(QPen(QColor(0, 200, 0, 120), 1.5, Qt.DotLine))
                    painter.drawLine(QPointF(mx, my), QPointF(fx, fy))

            # Draw vertices
            painter.setPen(QPen(QColor(0, 255, 0), 1))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
            for vx, vy in self._current_vertices:
                wx, wy = self._image_to_widget(vx, vy)
                painter.drawEllipse(QPointF(wx, wy), 5, 5)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if self._is_drawing:
                if self._pixmap is None:
                    return
                ix, iy = self._widget_to_image(event.x(), event.y())
                ix, iy = self._clamp_image_coords(ix, iy)
                self._current_vertices.append((ix, iy))
                self.vertex_placed.emit(ix, iy)
                self.update()
            else:
                # Check if clicking near a vertex for dragging
                self._check_vertex_drag(event.x(), event.y())

        elif event.button() == Qt.RightButton and not self._is_drawing:
            # Delete vertex
            self._check_vertex_delete(event.x(), event.y())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_drawing:
            self._mouse_pos = (event.x(), event.y())
            self.update()
        elif self._dragging is not None:
            ix, iy = self._widget_to_image(event.x(), event.y())
            ix, iy = self._clamp_image_coords(ix, iy)
            ri, vi = self._dragging
            if self._pixmap:
                self._regions[ri][vi] = (ix, iy)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging is not None:
            ri, vi = self._dragging
            ix, iy = self._regions[ri][vi]
            self.vertex_moved.emit(ri, vi, ix, iy)
            self._dragging = None

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self._is_drawing and len(self._current_vertices) >= 3:
                self._regions.append(list(self._current_vertices))
                self._region_names.append(f"region_{len(self._regions)}")
                self._current_vertices = []
                self._mouse_pos = None
                self.polygon_finished.emit()
                self.update()
        elif event.key() == Qt.Key_Escape:
            if self._is_drawing:
                self._current_vertices = []
                self._mouse_pos = None
                self.polygon_cancelled.emit()
                self.stop_drawing()
                self.update()

    def _check_vertex_drag(self, wx: int, wy: int):
        """Check if click is near a vertex and start dragging."""
        threshold = 8  # pixels
        for ri, region in enumerate(self._regions):
            for vi, (vx, vy) in enumerate(region):
                vwx, vwy = self._image_to_widget(vx, vy)
                if abs(wx - vwx) < threshold and abs(wy - vwy) < threshold:
                    self._dragging = (ri, vi)
                    return

    def _check_vertex_delete(self, wx: int, wy: int):
        """Check if right-click is near a vertex and delete it."""
        threshold = 8
        for ri, region in enumerate(self._regions):
            for vi, (vx, vy) in enumerate(region):
                vwx, vwy = self._image_to_widget(vx, vy)
                if abs(wx - vwx) < threshold and abs(wy - vwy) < threshold:
                    if len(region) > 3:
                        region.pop(vi)
                        self.vertex_deleted.emit(ri, vi)
                        self.update()
                    return

    def remove_region(self, index: int):
        """Remove a region by index."""
        if 0 <= index < len(self._regions):
            self._regions.pop(index)
            self._region_names.pop(index)
            if index < len(self._decomposed):
                self._decomposed.pop(index)
            self.update()

    def clear_all(self):
        """Remove all regions."""
        self._regions.clear()
        self._region_names.clear()
        self._decomposed.clear()
        self.update()
