"""YAML save/load for FOV blind spot polygon definitions."""

import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class BlindSpotRegion:
    """A single blind spot region with user-drawn and decomposed polygons."""
    name: str
    original_vertices: List[tuple]  # (azimuth_rad, elevation_rad)
    convex_polygons: List[List[tuple]] = field(default_factory=list)


@dataclass
class BlindSpotConfig:
    """Complete blind spot configuration for a sensor."""
    sensor_frame: str = "sensor"
    cloud_width: int = 0
    cloud_height: int = 0
    regions: List[BlindSpotRegion] = field(default_factory=list)


def save_config(config: BlindSpotConfig, filepath: str) -> None:
    """Save blind spot configuration to YAML file."""
    data = {
        'sensor_frame': config.sensor_frame,
        'cloud_width': config.cloud_width,
        'cloud_height': config.cloud_height,
        'regions': []
    }

    for region in config.regions:
        region_data = {
            'name': region.name,
            'original_vertices': [
                [float(az), float(el)] for az, el in region.original_vertices
            ],
            'convex_polygons': [
                [[float(az), float(el)] for az, el in poly]
                for poly in region.convex_polygons
            ]
        }
        data['regions'].append(region_data)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def format_polygons_as_param_string(config: BlindSpotConfig) -> str:
    """Format all convex polygons as a single STVL-compatible parameter string.

    Output format: "[[az1,el1, az2,el2, ...], [az3,el3, az4,el4, ...]]"
    """
    poly_strings = []
    for region in config.regions:
        for poly in region.convex_polygons:
            coords = ", ".join(
                f"{az:.6f},{el:.6f}" for az, el in poly
            )
            poly_strings.append(f"[{coords}]")
    return "[" + ", ".join(poly_strings) + "]"


def load_config(filepath: str) -> BlindSpotConfig:
    """Load blind spot configuration from YAML file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)

    config = BlindSpotConfig(
        sensor_frame=data.get('sensor_frame', 'sensor'),
        cloud_width=data.get('cloud_width', 0),
        cloud_height=data.get('cloud_height', 0),
    )

    for region_data in data.get('regions', []):
        region = BlindSpotRegion(
            name=region_data['name'],
            original_vertices=[tuple(v) for v in region_data['original_vertices']],
            convex_polygons=[
                [tuple(v) for v in poly]
                for poly in region_data.get('convex_polygons', [])
            ]
        )
        config.regions.append(region)

    return config
