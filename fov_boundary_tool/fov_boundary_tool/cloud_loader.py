"""Point cloud loading utilities for organized clouds."""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrganizedCloud:
    """An organized point cloud with xyz data and metadata."""
    xyz: np.ndarray  # shape (height, width, 3)
    frame_id: str = "sensor"
    width: int = 0
    height: int = 0

    def __post_init__(self):
        self.height, self.width = self.xyz.shape[:2]

    @property
    def ranges(self) -> np.ndarray:
        """Compute range for each point."""
        return np.linalg.norm(self.xyz, axis=2)

    @property
    def azimuths(self) -> np.ndarray:
        """Compute azimuth (atan2(y, x)) for each point, in [0, 2pi] range."""
        az = np.arctan2(self.xyz[:, :, 1], self.xyz[:, :, 0])
        az[az < 0] += 2.0 * np.pi
        return az

    @property
    def elevations(self) -> np.ndarray:
        """Compute elevation (atan2(z, sqrt(x²+y²))) for each point."""
        xy = np.sqrt(self.xyz[:, :, 0] ** 2 + self.xyz[:, :, 1] ** 2)
        return np.arctan2(self.xyz[:, :, 2], xy)


def load_pcd(filepath: str) -> OrganizedCloud:
    """Load an organized PCD file and return an OrganizedCloud.

    Supports ASCII and binary PCD formats.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PCD file not found: {filepath}")

    with open(filepath, 'rb') as f:
        header = {}
        fields = []
        sizes = []
        types = []
        counts = []

        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            if line.startswith('#'):
                continue
            if line.startswith('DATA'):
                header['data_format'] = line.split()[1]
                break

            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            if key == 'FIELDS':
                fields = parts[1:]
            elif key == 'SIZE':
                sizes = [int(s) for s in parts[1:]]
            elif key == 'TYPE':
                types = parts[1:]
            elif key == 'COUNT':
                counts = [int(c) for c in parts[1:]]
            elif key == 'WIDTH':
                header['width'] = int(parts[1])
            elif key == 'HEIGHT':
                header['height'] = int(parts[1])
            elif key == 'VIEWPOINT':
                header['viewpoint'] = [float(v) for v in parts[1:]]
            elif key == 'POINTS':
                header['points'] = int(parts[1])

        width = header.get('width', 0)
        height = header.get('height', 1)
        n_points = header.get('points', width * height)
        data_format = header.get('data_format', 'ascii')

        # Find x, y, z field indices
        try:
            x_idx = fields.index('x')
            y_idx = fields.index('y')
            z_idx = fields.index('z')
        except ValueError:
            raise ValueError(f"PCD file must have x, y, z fields. Found: {fields}")

        if data_format == 'ascii':
            data = np.loadtxt(f, max_rows=n_points)
            xyz = np.column_stack([data[:, x_idx], data[:, y_idx], data[:, z_idx]])
        elif data_format == 'binary':
            # Build numpy dtype for the full point
            dtype_map = {'F': 'f', 'U': 'u', 'I': 'i'}
            dt_fields = []
            for i, (name, size, typ, count) in enumerate(zip(fields, sizes, types, counts)):
                np_type = f"{dtype_map.get(typ, 'f')}{size}"
                if count == 1:
                    dt_fields.append((name, np_type))
                else:
                    dt_fields.append((name, np_type, (count,)))

            dtype = np.dtype(dt_fields)
            raw = np.frombuffer(f.read(n_points * dtype.itemsize), dtype=dtype, count=n_points)
            xyz = np.column_stack([raw['x'], raw['y'], raw['z']])
        elif data_format == 'binary_compressed':
            import struct
            import lzf
            compressed_size = struct.unpack('<I', f.read(4))[0]
            uncompressed_size = struct.unpack('<I', f.read(4))[0]
            compressed_data = f.read(compressed_size)
            raw_data = lzf.decompress(compressed_data, uncompressed_size)

            # Build dtype same as binary
            dtype_map = {'F': 'f', 'U': 'u', 'I': 'i'}
            field_sizes = []
            for size, typ, count in zip(sizes, types, counts):
                field_sizes.append(size * count)

            # Data is stored column-major in binary_compressed
            offset = 0
            columns = {}
            for i, (name, size, typ, count) in enumerate(zip(fields, sizes, types, counts)):
                np_type = f"{dtype_map.get(typ, 'f')}{size}"
                col_bytes = size * count * n_points
                columns[name] = np.frombuffer(raw_data[offset:offset + col_bytes], dtype=np_type)
                offset += col_bytes

            xyz = np.column_stack([columns['x'], columns['y'], columns['z']])
        else:
            raise ValueError(f"Unsupported PCD data format: {data_format}")

        # Reshape to organized if height > 1
        xyz = xyz.astype(np.float32)
        if height > 1:
            xyz = xyz.reshape(height, width, 3)
        else:
            # Unorganized cloud - treat as single row (still "organized" for our purposes)
            xyz = xyz.reshape(1, width, 3)

    return OrganizedCloud(xyz=xyz, frame_id="sensor")


def unfold_cloud(cloud: OrganizedCloud, height: int) -> OrganizedCloud:
    """Reshape a flat/unorganized cloud (height == 1) into an organized
    (height, width, 3) grid. `height` is the number of channels/rings -
    lidar drivers typically emit unorganized points azimuth-major,
    channel-minor (one full vertical column across all channels per azimuth
    step, then the next azimuth step), so `height` is the fast-varying run
    length in the raw flat order. The reshape uses it as the fast axis, then
    transposes so the result matches this tool's row=elevation-channel /
    column=azimuth convention (see _build_angular_axes in main_window.py).

    The raw channel index's direction (does it run top-to-bottom or
    bottom-to-top of the sensor's vertical FOV?) is sensor/driver-specific
    and unknown here, so the row order is corrected using the actual
    elevation of the data itself: row 0 (displayed at the top of the range
    image) is oriented to be the highest-elevation row, matching the
    intuitive "up" of a picture, regardless of which way the raw channel
    index happened to run.
    """
    total_points = cloud.xyz.shape[1]
    if height <= 0 or total_points % height != 0:
        raise ValueError(f"{total_points} points is not evenly divisible by height {height}.")
    width = total_points // height
    xyz = cloud.xyz.reshape(width, height, 3).transpose(1, 0, 2)

    ranges = np.linalg.norm(xyz, axis=2)
    elevations = np.arctan2(xyz[:, :, 2], np.hypot(xyz[:, :, 0], xyz[:, :, 1]))
    valid = (ranges > 0) & np.isfinite(elevations)

    def row_median_elevation(row: int) -> float:
        vals = elevations[row][valid[row]]
        return float(np.median(vals)) if vals.size else float('nan')

    first_el = row_median_elevation(0)
    last_el = row_median_elevation(height - 1)
    if np.isfinite(first_el) and np.isfinite(last_el) and first_el < last_el:
        xyz = xyz[::-1, :, :].copy()

    return OrganizedCloud(xyz=xyz, frame_id=cloud.frame_id)
