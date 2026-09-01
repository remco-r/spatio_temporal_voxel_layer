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

## Reading what you have drawn

Polygon edges are **great-circle arcs**, not straight lines in the range image, because
that is what STVL's `isInside` half-plane test uses. The "Show Great-Circle Arcs" toggle
(on by default) draws the real thing, so what you see is what will be masked. Two effects
are worth recognising, both of which STVL accepts silently:

- An edge spanning a wide range of azimuth **bows away from the equator**, reaching
  `atan(tan(el) / cos(dAz / 2))` at its midpoint. Drawn on a narrow-vFOV panorama the arc
  will visibly leave the top or bottom of the image. Keep each edge's azimuth step to
  about 0.5 rad or less, and split a wide blind spot into several regions.
- An edge whose azimuth step exceeds pi takes the short way round the **other** side, so
  the region masks the complement of what it appears to enclose. Such an arc leaves one
  side of the panorama and re-enters the far side; for a full 360 deg cloud the tool draws
  that wrap, so the region shows up on both edges of the image.

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
