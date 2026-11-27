"""Small, dependency-free sculpting tools for architectural mesh placements.

Coordinates follow BlockN's convention: X is width, Y is height, and Z is
depth.  These helpers produce real solids, including open arch silhouettes;
they do not fill a concave profile's opening with a convex hull.  The returned
meshes and serialized parameters never share writable arrays with inputs.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh


MAX_MESH_VERTICES = 65_536
MAX_MESH_FACES = 131_072
_EPS = 1e-12


def _contains_boolean(values: Any) -> bool:
    # A mixed Python list otherwise coerces True to 1 before dtype inspection.
    if isinstance(values, np.ndarray):
        return values.dtype.kind == "b"
    return any(isinstance(value, (bool, np.bool_)) for row in values for value in row)


def _points(values: Any, columns: int, name: str) -> np.ndarray:
    """Copy a finite, real numeric array without silently accepting strings."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if raw.ndim != 2 or raw.shape[1] != columns:
        raise ValueError(f"{name} must be an N by {columns} array")
    if raw.dtype.kind not in "fiu" or _contains_boolean(values):
        raise ValueError(f"{name} must contain real numbers")
    result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return result


def _mesh_arrays(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate the canonical mesh payload used by the MESH primitive."""
    points = _points(vertices, 3, "MESH vertices")
    if not 4 <= len(points) <= MAX_MESH_VERTICES:
        raise ValueError(f"MESH requires 4..{MAX_MESH_VERTICES} vertices")
    try:
        indices = np.asarray(faces)
    except (TypeError, ValueError) as exc:
        raise ValueError("MESH faces must be a rectangular integer array") from exc
    if (indices.ndim != 2 or indices.shape[1] != 3 or indices.dtype.kind not in "iu"
            or _contains_boolean(faces)):
        raise ValueError("MESH faces must be an M by 3 integer array")
    if not 4 <= len(indices) <= MAX_MESH_FACES:
        raise ValueError(f"MESH requires 4..{MAX_MESH_FACES} triangle faces")
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("MESH face indices must refer to existing vertices")
    indices = np.array(indices, dtype=np.int64, copy=True)
    if len(np.unique(indices)) != len(points):
        raise ValueError("MESH contains unreferenced vertices")
    canonical_faces = np.sort(indices, axis=1)
    if len(np.unique(canonical_faces, axis=0)) != len(indices):
        raise ValueError("MESH contains duplicate triangle faces")
    triangles = points[indices]
    area_vectors = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(area_vectors, axis=1) <= _EPS):
        raise ValueError("MESH triangles must be nondegenerate")
    return points, indices


def _normalized_mesh(
    position: np.ndarray,
    dimensions: np.ndarray,
    parameters: Mapping[str, Any],
) -> trimesh.Trimesh:
    """Compile a validated canonical payload; called by the primitive registry."""
    if "vertices" not in parameters or "faces" not in parameters:
        raise ValueError("MESH requires vertices and faces parameters")
    vertices, faces = _mesh_arrays(parameters["vertices"], parameters["faces"])
    if np.any(vertices < -0.5 - 1e-9) or np.any(vertices > 0.5 + 1e-9):
        raise ValueError("MESH vertices must lie within [-0.5, 0.5]")
    if not (
        np.allclose(vertices.min(axis=0), -0.5, rtol=0.0, atol=1e-9)
        and np.allclose(vertices.max(axis=0), 0.5, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("MESH geometry must span its full placement envelope")
    # Remove accepted roundoff at the extremes without collapsing close points.
    minimum = vertices.min(axis=0)
    vertices = (vertices - minimum) / (vertices.max(axis=0) - minimum) - 0.5
    vertices = vertices * dimensions + position
    if not np.all(np.isfinite(vertices)):
        raise ValueError("MESH transformed vertices must remain finite")
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def mesh_parameters(
    mesh: trimesh.Trimesh,
) -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, Any]]:
    """Normalize a world-space mesh for a deterministic ``MESH`` placement.

    Returns ``(position, dimensions, {"vertices": ..., "faces": ...})``.
    Vertex coordinates are normalized per axis into [-0.5, 0.5].  Unused
    vertices, duplicate faces, and degenerate triangles are rejected so the
    recorded envelope describes actual geometry rather than dummy points.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh must be a trimesh.Trimesh")
    vertices = _points(mesh.vertices, 3, "mesh vertices")
    if not len(vertices):
        raise ValueError("mesh must not be empty")
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    dimensions = maximum - minimum
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions <= 0):
        raise ValueError("mesh must have a finite, positive three-dimensional envelope")
    position = minimum + dimensions / 2.0
    normalized = (vertices - minimum) / dimensions - 0.5
    normalized, faces = _mesh_arrays(normalized, mesh.faces)
    return (
        tuple(float(value) for value in position),
        tuple(float(value) for value in dimensions),
        {"vertices": normalized.tolist(), "faces": faces.tolist()},
    )


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _profile_points(profile: Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    points = _points(profile, 2, "profile")
    if not 3 <= len(points) <= 2048:
        raise ValueError("profile requires 3..2048 vertices")
    minimum = points.min(axis=0)
    scale = float(np.max(points.max(axis=0) - minimum))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("profile must enclose a nonzero area")
    normalized = (points - minimum) / scale
    for index, point in enumerate(normalized):
        if np.any(np.linalg.norm(normalized[index + 1:] - point, axis=1) <= _EPS):
            raise ValueError("profile vertices must be distinct; omit the closing duplicate")

    # Straight collinear samples are harmless but would form zero-area ears.
    # Remove those samples while rejecting an edge that doubles back on itself.
    changed = True
    while changed and len(normalized) > 3:
        changed = False
        for index in range(len(normalized)):
            previous = normalized[index - 1] - normalized[index]
            following = normalized[(index + 1) % len(normalized)] - normalized[index]
            if abs(float(_cross2(previous, following))) <= _EPS:
                if float(np.dot(previous, following)) >= 0:
                    raise ValueError("profile edges must not double back")
                points = np.delete(points, index, axis=0)
                normalized = np.delete(normalized, index, axis=0)
                changed = True
                break

    ends = np.roll(normalized, -1, axis=0)
    for index, (a, b) in enumerate(zip(normalized, ends)):
        other_indices = np.arange(index + 2, len(normalized))
        if index == 0:
            other_indices = other_indices[other_indices != len(normalized) - 1]
        if not len(other_indices):
            continue
        c = normalized[other_indices]
        d = ends[other_indices]
        overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d))
                         <= np.minimum(np.maximum(a, b), np.maximum(c, d)) + _EPS, axis=1)
        orientation_c = _cross2(b - a, c - a)
        orientation_d = _cross2(b - a, d - a)
        orientation_a = _cross2(d - c, a - c)
        orientation_b = _cross2(d - c, b - c)
        crosses = (
            ((orientation_c >= -_EPS) & (orientation_d <= _EPS)
             | (orientation_c <= _EPS) & (orientation_d >= -_EPS))
            & ((orientation_a >= -_EPS) & (orientation_b <= _EPS)
               | (orientation_a <= _EPS) & (orientation_b >= -_EPS))
        )
        if np.any(overlap & crosses):
            raise ValueError("profile must be a simple polygon without crossing or touching edges")

    area_twice = float(np.sum(_cross2(normalized, np.roll(normalized, -1, axis=0))))
    if abs(area_twice) <= _EPS:
        raise ValueError("profile must enclose a nonzero area")
    if area_twice < 0:
        points = points[::-1].copy()
        normalized = normalized[::-1].copy()
    return points, normalized


def _triangulate_polygon(points: np.ndarray) -> np.ndarray:
    """Ear clip a validated CCW simple polygon without optional backends."""
    remaining = list(range(len(points)))
    faces = []
    while len(remaining) > 3:
        for index, current in enumerate(remaining):
            previous = remaining[index - 1]
            following = remaining[(index + 1) % len(remaining)]
            a, b, c = points[[previous, current, following]]
            if float(_cross2(b - a, c - b)) <= _EPS:
                continue
            others = [value for value in remaining if value not in (previous, current, following)]
            candidates = points[others]
            inside = (
                (_cross2(b - a, candidates - a) >= -_EPS)
                & (_cross2(c - b, candidates - b) >= -_EPS)
                & (_cross2(a - c, candidates - c) >= -_EPS)
            )
            if np.any(inside):
                continue
            faces.append((previous, current, following))
            del remaining[index]
            break
        else:
            raise ValueError("profile cannot be triangulated without degenerate triangles")
    faces.append(tuple(remaining))
    return np.asarray(faces, dtype=np.int64)


def extrude_profile(profile: Sequence[Sequence[float]], depth: float) -> trimesh.Trimesh:
    """Extrude a simple XY polygon through a centered, positive Z depth.

    Pass the polygon's ordered boundary without a duplicate closing vertex.
    Both winding directions and concave boundaries are supported.  Repeated,
    crossing, or touching edges are rejected; nested holes should instead be
    represented as an open-bottom arch boundary or separate solid members.
    """
    depth = _positive(depth, "depth")
    points, normalized = _profile_points(profile)
    cap = _triangulate_polygon(normalized)
    count = len(points)
    vertices = np.concatenate([
        np.column_stack([points, np.full(count, -depth / 2.0)]),
        np.column_stack([points, np.full(count, depth / 2.0)]),
    ])
    sides = []
    for index in range(count):
        following = (index + 1) % count
        sides.extend([(index, following, following + count),
                      (index, following + count, index + count)])
    faces = np.concatenate([cap[:, ::-1], cap + count, np.asarray(sides, dtype=np.int64)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _rotate(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    return (vector * cosine + np.cross(axis, vector) * np.sin(angle)
            + axis * np.dot(axis, vector) * (1.0 - cosine))


def _transport(normal: np.ndarray, previous: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    cross = np.cross(previous, tangent)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(previous, tangent), -1.0, 1.0))
    if sine > _EPS:
        normal = _rotate(normal, cross / sine, float(np.arctan2(sine, cosine)))
    elif cosine < 0:
        raise ValueError("path cannot reverse its tangent direction")
    normal = normal - tangent * np.dot(normal, tangent)
    return normal / np.linalg.norm(normal)


def tube_path(
    points: Sequence[Sequence[float]],
    radius: float,
    sections: int = 8,
    closed: bool = False,
) -> trimesh.Trimesh:
    """Sweep a constant-radius polygonal rib along a 3D centerline.

    Parallel-transported frames avoid the flips of a fixed up vector.  Closed
    loops spread residual frame rotation around the loop and share their seam
    vertices; open tubes have solid end caps.  Omit a duplicate final point for
    a closed loop.  Radius should be smaller than local curvature and spacing;
    this function does not resolve self-intersections of thick or crossing ribs.
    """
    path = _points(points, 3, "path")
    radius = _positive(radius, "radius")
    sections = _integer(sections, 3, 128, "sections")
    if not isinstance(closed, (bool, np.bool_)):
        raise ValueError("closed must be a boolean")
    closed = bool(closed)
    minimum = 3 if closed else 2
    if len(path) < minimum or len(path) * sections + (0 if closed else 2) > MAX_MESH_VERTICES:
        raise ValueError(f"path requires at least {minimum} points and at most {MAX_MESH_VERTICES} mesh vertices")
    differences = np.roll(path, -1, axis=0) - path if closed else np.diff(path, axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("path segments must have finite, nonzero length; omit a duplicate closing point")
    directions = differences / lengths[:, None]
    if closed:
        tangents = np.roll(directions, 1, axis=0) + directions
    else:
        tangents = np.concatenate([directions[:1], directions[:-1] + directions[1:], directions[-1:]])
    tangent_lengths = np.linalg.norm(tangents, axis=1)
    if np.any(tangent_lengths <= _EPS):
        raise ValueError("path must not double back on itself")
    tangents /= tangent_lengths[:, None]
    reference = np.eye(3)[int(np.argmin(np.abs(tangents[0])))]
    normal = reference - tangents[0] * np.dot(reference, tangents[0])
    normal /= np.linalg.norm(normal)
    normals = [normal]
    for index in range(1, len(path)):
        normals.append(_transport(normals[-1], tangents[index - 1], tangents[index]))
    normals = np.asarray(normals)
    if closed:
        seam = _transport(normals[-1], tangents[-1], tangents[0])
        correction = float(np.arctan2(np.dot(tangents[0], np.cross(seam, normals[0])),
                                      np.dot(seam, normals[0])))
        distances = np.concatenate([[0.0], np.cumsum(lengths[:-1])])
        for index, fraction in enumerate(distances / lengths.sum()):
            normals[index] = _rotate(normals[index], tangents[index], correction * fraction)
    binormals = np.cross(tangents, normals)
    angles = np.arange(sections) * (2.0 * np.pi / sections)
    vertices = (path[:, None, :] + radius * (
        np.cos(angles)[None, :, None] * normals[:, None, :]
        + np.sin(angles)[None, :, None] * binormals[:, None, :]
    )).reshape(-1, 3)
    faces = []
    segment_count = len(path) if closed else len(path) - 1
    for index in range(segment_count):
        following = (index + 1) % len(path)
        for section in range(sections):
            next_section = (section + 1) % sections
            a, b = index * sections + section, index * sections + next_section
            c, d = following * sections + section, following * sections + next_section
            faces.extend([(a, b, d), (a, d, c)])
    if not closed:
        start, end = len(vertices), len(vertices) + 1
        vertices = np.concatenate([vertices, path[[0, -1]]])
        last_ring = (len(path) - 1) * sections
        for section in range(sections):
            following = (section + 1) % sections
            faces.extend([(start, following, section),
                          (end, last_ring + section, last_ring + following)])
    if not np.all(np.isfinite(vertices)):
        raise ValueError("tube vertices must remain finite")
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=False)


__all__ = ["extrude_profile", "tube_path", "mesh_parameters"]
