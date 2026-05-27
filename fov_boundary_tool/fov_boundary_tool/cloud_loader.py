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


def load_from_ros_msg(msg) -> OrganizedCloud:
    """Convert a sensor_msgs/PointCloud2 message to OrganizedCloud.

    Requires the message to be organized (height > 1).
    """
    import struct
    from collections import namedtuple

    width = msg.width
    height = msg.height

    if height <= 1:
        raise ValueError(
            f"Expected organized cloud (height > 1), got height={height}. "
            "Unorganized clouds are not supported."
        )

    # Find x, y, z field offsets
    field_map = {f.name: f for f in msg.fields}
    for required in ('x', 'y', 'z'):
        if required not in field_map:
            raise ValueError(f"PointCloud2 missing required field: {required}")

    x_field = field_map['x']
    y_field = field_map['y']
    z_field = field_map['z']

    point_step = msg.point_step
    data = np.frombuffer(msg.data, dtype=np.uint8)

    # Extract x, y, z as float32 arrays
    xyz = np.zeros((height * width, 3), dtype=np.float32)
    for i, field in enumerate([x_field, y_field, z_field]):
        offset = field.offset
        # Interpret as float32 at each point's offset
        values = np.frombuffer(msg.data, dtype=np.float32,
                               count=height * width,
                               offset=0)
        # Need to handle stride properly
        all_data = np.frombuffer(msg.data, dtype=np.uint8)
        float_data = np.zeros(height * width, dtype=np.float32)
        for idx in range(height * width):
            start = idx * point_step + offset
            float_data[idx] = np.frombuffer(all_data[start:start + 4], dtype=np.float32)[0]
        xyz[:, i] = float_data

    xyz = xyz.reshape(height, width, 3)
    frame_id = msg.header.frame_id if hasattr(msg, 'header') else "sensor"

    return OrganizedCloud(xyz=xyz, frame_id=frame_id)


def load_from_ros_msg_fast(msg) -> OrganizedCloud:
    """Fast conversion of sensor_msgs/PointCloud2 to OrganizedCloud using numpy structured arrays."""
    width = msg.width
    height = msg.height

    if height <= 1:
        raise ValueError(
            f"Expected organized cloud (height > 1), got height={height}."
        )

    field_map = {f.name: f for f in msg.fields}
    for required in ('x', 'y', 'z'):
        if required not in field_map:
            raise ValueError(f"PointCloud2 missing required field: {required}")

    point_step = msg.point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height * width, point_step)

    x_off = field_map['x'].offset
    y_off = field_map['y'].offset
    z_off = field_map['z'].offset

    x = raw[:, x_off:x_off + 4].view(np.float32).flatten()
    y = raw[:, y_off:y_off + 4].view(np.float32).flatten()
    z = raw[:, z_off:z_off + 4].view(np.float32).flatten()

    xyz = np.column_stack([x, y, z]).reshape(height, width, 3)
    frame_id = msg.header.frame_id if hasattr(msg, 'header') else "sensor"

    return OrganizedCloud(xyz=xyz, frame_id=frame_id)
