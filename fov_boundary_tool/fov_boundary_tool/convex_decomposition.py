"""Hertel-Mehlhorn convex decomposition algorithm."""

import numpy as np
from typing import List, Tuple
from scipy.spatial import Delaunay


Polygon = List[Tuple[float, float]]


def cross_2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """2D cross product of vectors OA and OB."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def is_convex_polygon(vertices: List[np.ndarray]) -> bool:
    """Check if a polygon (list of 2D points) is convex."""
    n = len(vertices)
    if n < 3:
        return False

    sign = None
    for i in range(n):
        o = vertices[i]
        a = vertices[(i + 1) % n]
        b = vertices[(i + 2) % n]
        c = cross_2d(o, a, b)
        if abs(c) < 1e-10:
            continue
        if sign is None:
            sign = c > 0
        elif (c > 0) != sign:
            return False
    return True


def ear_clip_triangulate(polygon: Polygon) -> List[Tuple[int, int, int]]:
    """Triangulate a simple polygon using ear clipping.

    Returns list of triangles as tuples of vertex indices.
    """
    n = len(polygon)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    vertices = [np.array(p, dtype=np.float64) for p in polygon]
    indices = list(range(n))
    triangles = []

    # Determine winding order
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    ccw = area > 0

    def is_ear(idx: int, idx_list: List[int], verts: List[np.ndarray]) -> bool:
        n_v = len(idx_list)
        pos = idx_list.index(idx)
        prev_idx = idx_list[(pos - 1) % n_v]
        next_idx = idx_list[(pos + 1) % n_v]

        a = verts[prev_idx]
        b = verts[idx]
        c = verts[next_idx]

        # Check if angle at b is convex
        cross = cross_2d(a, b, c)
        if ccw and cross <= 1e-10:
            return False
        if not ccw and cross >= -1e-10:
            return False

        # Check no other vertex is inside triangle abc
        for other in idx_list:
            if other in (prev_idx, idx, next_idx):
                continue
            if point_in_triangle(verts[other], a, b, c):
                return False
        return True

    remaining = list(indices)
    max_iter = n * n  # safety limit

    iter_count = 0
    while len(remaining) > 3 and iter_count < max_iter:
        iter_count += 1
        found_ear = False
        for i in range(len(remaining)):
            if is_ear(remaining[i], remaining, vertices):
                pos = i
                prev_pos = (pos - 1) % len(remaining)
                next_pos = (pos + 1) % len(remaining)
                triangles.append((remaining[prev_pos], remaining[pos], remaining[next_pos]))
                remaining.pop(pos)
                found_ear = True
                break
        if not found_ear:
            # Fallback: just make a fan triangulation
            for i in range(1, len(remaining) - 1):
                triangles.append((remaining[0], remaining[i], remaining[i + 1]))
            break

    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))

    return triangles


def point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Check if point p is strictly inside triangle abc."""
    d1 = cross_2d(a, b, p)
    d2 = cross_2d(b, c, p)
    d3 = cross_2d(c, a, p)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def _can_merge(poly_a: List[int], poly_b: List[int], shared_edge: Tuple[int, int],
               vertices: List[np.ndarray]) -> bool:
    """Check if merging two polygons along a shared edge produces a convex polygon."""
    merged = _merge_polygons(poly_a, poly_b, shared_edge)
    if merged is None:
        return False
    return is_convex_polygon([vertices[i] for i in merged])


def _merge_polygons(poly_a: List[int], poly_b: List[int],
                    shared_edge: Tuple[int, int]) -> List[int]:
    """Merge two polygons that share an edge, removing the shared edge."""
    e0, e1 = shared_edge

    # Find the shared edge in poly_a (e0 -> e1 direction)
    n_a = len(poly_a)
    edge_pos_a = None
    for i in range(n_a):
        if poly_a[i] == e0 and poly_a[(i + 1) % n_a] == e1:
            edge_pos_a = i
            break
        if poly_a[i] == e1 and poly_a[(i + 1) % n_a] == e0:
            edge_pos_a = i
            e0, e1 = e1, e0
            break

    if edge_pos_a is None:
        return None

    # Find e1 -> e0 in poly_b
    n_b = len(poly_b)
    edge_pos_b = None
    for i in range(n_b):
        if poly_b[i] == e1 and poly_b[(i + 1) % n_b] == e0:
            edge_pos_b = i
            break

    if edge_pos_b is None:
        return None

    # Build merged polygon: walk poly_a skipping e0->e1 edge, insert poly_b vertices
    merged = []
    for i in range(n_a):
        idx = (edge_pos_a + 1 + i) % n_a
        if idx == edge_pos_a:
            break
        merged.append(poly_a[idx])

    # Insert poly_b vertices (excluding the shared edge endpoints if they're just connectors)
    for i in range(1, n_b - 1):
        idx = (edge_pos_b + 1 + i) % n_b
        merged.append(poly_b[idx])

    return merged if len(merged) >= 3 else None


def hertel_mehlhorn(polygon: Polygon) -> List[Polygon]:
    """Decompose a simple polygon into convex parts using Hertel-Mehlhorn.

    1. Triangulate using ear clipping
    2. Iteratively remove internal diagonals that keep both sides convex

    Args:
        polygon: List of (azimuth, elevation) tuples defining a simple polygon.

    Returns:
        List of convex polygons, each a list of (azimuth, elevation) tuples.
    """
    if len(polygon) < 3:
        return []

    vertices = [np.array(p, dtype=np.float64) for p in polygon]

    # Check if already convex
    if is_convex_polygon(vertices):
        return [polygon]

    # Step 1: Triangulate
    triangles = ear_clip_triangulate(polygon)
    if not triangles:
        return [polygon]

    # Step 2: Build adjacency - each polygon starts as a triangle
    polys = [list(tri) for tri in triangles]

    # Build internal edge map: edge -> list of polygon indices that share it
    def get_edges(poly_indices: List[int]) -> List[Tuple[int, int]]:
        edges = []
        n = len(poly_indices)
        for i in range(n):
            e = (poly_indices[i], poly_indices[(i + 1) % n])
            edges.append(e)
        return edges

    # Original polygon edges (not internal diagonals)
    n = len(polygon)
    boundary_edges = set()
    for i in range(n):
        e = (i, (i + 1) % n)
        boundary_edges.add(e)
        boundary_edges.add((e[1], e[0]))

    # Step 3: Iteratively merge
    changed = True
    while changed:
        changed = False
        # Build adjacency
        edge_to_poly = {}
        for pi, poly in enumerate(polys):
            for edge in get_edges(poly):
                canonical = (min(edge), max(edge))
                if canonical in boundary_edges or (canonical[1], canonical[0]) in boundary_edges:
                    # Check if it's actually a boundary edge
                    if edge in boundary_edges or (edge[1], edge[0]) in boundary_edges:
                        continue
                if canonical not in edge_to_poly:
                    edge_to_poly[canonical] = []
                edge_to_poly[canonical].append((pi, edge))

        # Try removing each internal diagonal
        for canonical, poly_list in edge_to_poly.items():
            if len(poly_list) != 2:
                continue
            pi_a, edge_a = poly_list[0]
            pi_b, edge_b = poly_list[1]
            if pi_a == pi_b:
                continue

            shared = (canonical[0], canonical[1])
            if _can_merge(polys[pi_a], polys[pi_b], shared, vertices):
                merged = _merge_polygons(polys[pi_a], polys[pi_b], shared)
                if merged and is_convex_polygon([vertices[i] for i in merged]):
                    # Replace pi_a with merged, remove pi_b
                    polys[pi_a] = merged
                    polys.pop(pi_b)
                    changed = True
                    break

    # Convert back to coordinate tuples
    result = []
    for poly_indices in polys:
        result.append([polygon[i] for i in poly_indices])

    return result
