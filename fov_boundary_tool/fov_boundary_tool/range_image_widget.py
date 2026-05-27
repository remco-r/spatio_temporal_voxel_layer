"""Range image widget for displaying and interacting with the depth image."""

import numpy as np
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QPointF
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QPolygonF,
                          QBrush, QMouseEvent, QKeyEvent)
from typing import List, Optional, Tuple


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
        self._scale = min(scale_x, scale_y)
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

        # Draw completed regions
        colors = [
            QColor(255, 100, 100, 80), QColor(100, 255, 100, 80),
            QColor(100, 100, 255, 80), QColor(255, 255, 100, 80),
            QColor(255, 100, 255, 80), QColor(100, 255, 255, 80),
        ]

        for ri, region in enumerate(self._regions):
            color = colors[ri % len(colors)]
            pen = QPen(color.darker(150), 2)
            painter.setPen(pen)

            # Draw filled polygon
            if len(region) >= 3:
                poly = QPolygonF()
                for vx, vy in region:
                    wx, wy = self._image_to_widget(vx, vy)
                    poly.append(QPointF(wx, wy))
                painter.setBrush(QBrush(color))
                painter.drawPolygon(poly)

            # Draw vertices
            painter.setBrush(QBrush(color.darker(120)))
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
                    poly = QPolygonF()
                    for vx, vy in convex_poly:
                        wx, wy = self._image_to_widget(vx, vy)
                        poly.append(QPointF(wx, wy))
                    painter.drawPolygon(poly)
                    ci += 1

        # Draw current polygon being drawn
        if self._is_drawing and self._current_vertices:
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            # Draw edges
            for i in range(len(self._current_vertices) - 1):
                x1, y1 = self._image_to_widget(*self._current_vertices[i])
                x2, y2 = self._image_to_widget(*self._current_vertices[i + 1])
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

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
                ix, iy = self._widget_to_image(event.x(), event.y())
                if self._pixmap and 0 <= ix < self._pixmap.width() and 0 <= iy < self._pixmap.height():
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
            ri, vi = self._dragging
            if self._pixmap and 0 <= ix < self._pixmap.width() and 0 <= iy < self._pixmap.height():
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
