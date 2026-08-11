# FOV Boundary Tool

GUI tool for defining convex polygon FOV blind spot boundaries on lidar range images.

## Purpose

Define angular blind spots (e.g., robot body, mast) in a lidar's field of view. These blind
spots are saved as convex polygon regions that can later be used by STVL to suppress voxel
clearing in those directions.

## Usage

```bash
# Run directly
python -m fov_boundary_tool.main

# Or if installed
fov_boundary_tool
```

## Workflow

1. **Load** an organized point cloud (PCD file)
2. **Adjust** the depth threshold slider to highlight close objects (robot body)
3. **Draw** polygon regions around blind spots:
   - Click to place vertices
   - Closing line preview shown while drawing
   - **Enter** to finish polygon
   - **Esc** to cancel current polygon
4. **Edit** polygons:
   - Drag vertices to move them
   - Right-click to delete a vertex
   - Ctrl+Z / Ctrl+Y for undo/redo
5. **Decompose** regions into convex polygons (Hertel-Mehlhorn algorithm)
6. **Save** to YAML file (coordinates in radians, sensor frame)
7. **Load** previously saved configurations for editing

## Output Format

YAML file with convex polygons in spherical angular coordinates (azimuth, elevation) in radians:

```yaml
sensor_frame: "os_sensor"
cloud_width: 1024
cloud_height: 128
regions:
  - name: "mast"
    original_vertices:
      - [0.785, -0.174]
      - [0.873, -0.174]
      - [0.873, 0.087]
      - [0.785, 0.087]
    convex_polygons:
      - [[0.785, -0.174], [0.873, -0.174], [0.873, 0.087], [0.785, 0.087]]
```

## Dependencies

- Python 3
- PyQt5
- NumPy
- SciPy
- PyYAML
