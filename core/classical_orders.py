"""Sculpted, reusable column solids in the application's Y-up coordinates.

The Greek Doric and Ionic choices follow the Met's *Architecture in Ancient
Greece* and Vitruvius III.5 / IV.3: baseless Doric, twenty hollow flutes, and
twenty-four filleted Ionic flutes. Profiles are design interpretations rather
than measured reconstructions of one monument. Every returned part is a
closed solid; ornament is assembled without boolean or triangulation engines.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from numbers import Real

import numpy as np
import trimesh

from .architectural_geometry import extrude_profile
from .architecture.masonry import MAX_COURSE, MAX_STRETCHER


@dataclass(frozen=True)
class ColumnPart:
    element: str
    mesh: trimesh.Trimesh


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def _closed_loft(rings: np.ndarray) -> trimesh.Trimesh:
    """Close corresponding, nondegenerate rings, with outward winding."""
    levels, sections, _ = rings.shape
    vertices = np.concatenate([rings.reshape(-1, 3), rings[0].mean(axis=0)[None],
                               rings[-1].mean(axis=0)[None]])
    faces = []
    for level in range(levels - 1):
        for index in range(sections):
            a = level * sections + index
            b = level * sections + (index + 1) % sections
            faces.extend([[a, a + sections, b + sections], [a, b + sections, b]])
    bottom, top = len(vertices) - 2, len(vertices) - 1
    for index in range(sections):
        following = (index + 1) % sections
        faces.extend([[bottom, index, following],
                      [top, (levels - 1) * sections + following,
                       (levels - 1) * sections + index]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Some uses sweep along X instead of Y; give all callers positive winding.
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _lathe(profile: list[tuple[float, float]], sections: int = 48) -> trimesh.Trimesh:
    """Revolve (height, radius) stations; duplicate heights form real ledges."""
    profile = np.asarray(profile, dtype=float)
    theta = np.arange(sections) * (2 * np.pi / sections)
    rings = np.empty((len(profile), sections, 3), dtype=float)
    rings[:, :, 0] = profile[:, 1, None] * np.cos(theta)
    rings[:, :, 1] = profile[:, 0, None]
    rings[:, :, 2] = profile[:, 1, None] * np.sin(theta)
    return _closed_loft(rings)


def _box(width: float, height: float, depth: float, bottom: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[width, height, depth])
    mesh.apply_translation([0, bottom + height / 2, 0])
    return mesh


def _span_count(span: float, limit: float) -> int:
    span = float(span)
    if span <= 1e-9:
        return 1
    count = max(1, int(math.ceil(span / limit - 1e-9)))
    while span / count > limit + 1e-9:
        count += 1
    return count


def _course_box_mesh(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    lo, hi = mesh.bounds
    dx, dy, dz = (float(v) for v in (hi - lo))
    nx, ny, nz = _span_count(dx, MAX_STRETCHER), _span_count(dy, MAX_COURSE), _span_count(dz, MAX_STRETCHER)
    if nx * ny * nz == 1:
        return [mesh]
    sx, sy, sz = dx / nx, dy / ny, dz / nz
    pieces = []
    for iy in range(ny):
        for ix in range(nx):
            for iz in range(nz):
                piece = trimesh.creation.box(extents=[sx, sy, sz])
                piece.apply_translation([
                    lo[0] + (ix + 0.5) * sx,
                    lo[1] + (iy + 0.5) * sy,
                    lo[2] + (iz + 0.5) * sz,
                ])
                pieces.append(piece)
    return pieces


def _course_lathe_mesh(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    verts = np.asarray(mesh.vertices, dtype=float)
    if len(verts) < 10:
        return [mesh]
    body = verts[:-2]
    sections = None
    for candidate in (48, 32, 24, 16):
        if len(body) % candidate == 0:
            levels = len(body) // candidate
            rings = body.reshape(levels, candidate, 3)
            if levels >= 2 and np.ptp(rings[:, :, 1], axis=1).max() <= 1e-6:
                sections = candidate
                break
    if sections is None:
        return [mesh]
    ys = rings[:, 0, 1]
    bands = []
    start = 0
    for index in range(1, levels):
        if ys[index] - ys[start] > MAX_COURSE + 1e-6:
            if index - 1 <= start:
                start = index
                continue
            bands.append((start, index))
            start = index - 1
    bands.append((start, levels))
    pieces = []
    for a, b in bands:
        if b - a < 2:
            continue
        part = _closed_loft(rings[a:b])
        extent = part.bounds[1] - part.bounds[0]
        if float(extent[1]) <= MAX_COURSE + 1e-6 and float(part.volume) > 1e-8:
            pieces.append(part)
    return pieces or [mesh]


def course_part_mesh(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    """Split a column part until its AABB is a cubit ashlar. Local Y-up."""
    lo, hi = mesh.bounds
    dx, dy, dz = (float(v) for v in (hi - lo))
    if dy <= MAX_COURSE + 1e-6 and max(dx, dz) <= MAX_STRETCHER + 1e-6:
        return [mesh]
    if len(mesh.vertices) <= 24:
        return _course_box_mesh(mesh)
    return _course_lathe_mesh(mesh)


def _fluted_section(flutes: int, fillet: float) -> np.ndarray:
    """Circular hollows cut inward between arrises or narrow flat fillets."""
    half_angle = np.pi / flutes
    channel_angle = half_angle * (1 - fillet)
    half_chord = np.sin(channel_angle)
    edge_radius = np.cos(channel_angle)
    hollow_bottom = 0.945 if flutes == 20 else 0.940
    sag = edge_radius - hollow_bottom
    circle_radius = (half_chord ** 2 + sag ** 2) / (2 * sag)
    circle_center = hollow_bottom + circle_radius
    result = []
    for index in range(flutes):
        center_angle = (2 * index + 1) * half_angle
        radial = np.array([np.cos(center_angle), np.sin(center_angle)])
        tangent = np.array([-np.sin(center_angle), np.cos(center_angle)])
        # Seven samples give six visibly concave faces within each channel.
        for u in np.linspace(-half_chord, half_chord, 7)[:(-1 if fillet == 0 else None)]:
            v = circle_center - np.sqrt(max(0, circle_radius ** 2 - u ** 2))
            result.append(v * radial + u * tangent)
        if fillet:
            angle = center_angle + half_angle
            result.append(np.array([np.cos(angle), np.sin(angle)]))
    return np.asarray(result)


_SHAFT_STATIONS = np.array([0.0, 0.055, 0.16, 0.32, 0.52, 0.73, 0.90, 1.0])


def _shaft_section(flutes: int, fillet: float) -> np.ndarray:
    if flutes:
        return _fluted_section(flutes, float(fillet))
    theta = np.arange(48) * (2 * np.pi / 48)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _validate_shaft(radius, height, flutes, fillet, top_ratio):
    radius, height = _positive(radius, "radius"), _positive(height, "height")
    if isinstance(flutes, bool) or flutes not in (0, 20, 24):
        raise ValueError("flutes must be 0, 20, or 24")
    if not isinstance(fillet, Real) or not np.isfinite(fillet) or not 0 <= fillet <= 0.4:
        raise ValueError("fillet must be in [0, 0.4]")
    if not isinstance(top_ratio, Real) or not np.isfinite(top_ratio) or not 0.65 <= top_ratio <= 0.95:
        raise ValueError("top_ratio must be in [0.65, 0.95]")
    return radius, height, int(flutes), float(fillet), float(top_ratio)


def _shaft_rings(radius: float, height: float, flutes: int, fillet: float,
                 top_ratio: float, t: np.ndarray) -> np.ndarray:
    section = _shaft_section(flutes, fillet)
    swelling = min(0.026, (1 - top_ratio) / np.pi * 0.70)
    scale = radius * (1 - (1 - top_ratio) * t + swelling * np.sin(np.pi * t))
    rings = np.empty((len(t), len(section), 3), dtype=float)
    rings[:, :, 0] = section[None, :, 0] * scale[:, None]
    rings[:, :, 1] = height * t[:, None]
    rings[:, :, 2] = section[None, :, 1] * scale[:, None]
    return rings


def fluted_shaft(radius: float, height: float, *, flutes: int = 20,
                 fillet: float = 0.0, top_ratio: float = 0.83) -> trimesh.Trimesh:
    """A tapered solid with restrained entasis and actual concave fluting.

    ``flutes=0`` produces the smooth Tuscan shaft. The lower radius is exact;
    entasis bows the profile outside a straight taper without exceeding it.
    """
    radius, height, flutes, fillet, top_ratio = _validate_shaft(
        radius, height, flutes, fillet, top_ratio)
    return _closed_loft(_shaft_rings(radius, height, flutes, fillet, top_ratio, _SHAFT_STATIONS))


def fluted_shaft_courses(radius: float, height: float, *, flutes: int = 20,
                         fillet: float = 0.0, top_ratio: float = 0.83,
                         max_height: float = 0.80) -> tuple[trimesh.Trimesh, ...]:
    """Stack fluted drums whose AABB height never exceeds ``max_height``."""
    radius, height, flutes, fillet, top_ratio = _validate_shaft(
        radius, height, flutes, fillet, top_ratio)
    max_height = _positive(max_height, "max_height")
    count = max(1, int(np.ceil(height / max_height - 1e-9)))
    while height / count > max_height + 1e-9:
        count += 1
    drums = []
    for index in range(count):
        t0, t1 = index / count, (index + 1) / count
        stations = _SHAFT_STATIONS[(_SHAFT_STATIONS > t0 + 1e-12) & (_SHAFT_STATIONS < t1 - 1e-12)]
        t = np.unique(np.concatenate(([t0, t1], stations)))
        drums.append(_closed_loft(_shaft_rings(radius, height, flutes, fillet, top_ratio, t)))
    return tuple(drums)


def spiral_volute(radius: float, depth: float, *, hand: int = 1) -> trimesh.Trimesh:
    """An extruded two-turn spiral ribbon with an open, readable channel."""
    radius, depth = _positive(radius, "radius"), _positive(depth, "depth")
    if hand not in (-1, 1) or isinstance(hand, bool):
        raise ValueError("hand must be -1 or 1")
    theta = np.linspace(np.pi / 2, np.pi / 2 + 4 * np.pi, 73)
    r = np.linspace(1.0, 0.12, len(theta))
    centerline = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    derivative = np.column_stack([-0.88 / (4 * np.pi) * np.cos(theta) - r * np.sin(theta),
                                 -0.88 / (4 * np.pi) * np.sin(theta) + r * np.cos(theta)])
    normals = np.column_stack([-derivative[:, 1], derivative[:, 0]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    width = 0.060
    boundary = np.concatenate([centerline + width * normals,
                               (centerline - width * normals)[::-1]])
    boundary[:, 0] *= hand
    # Full requested envelope makes placement and corner clearance predictable.
    boundary -= (boundary.max(axis=0) + boundary.min(axis=0)) / 2
    boundary *= 2 * radius / np.ptp(boundary, axis=0)
    return extrude_profile(boundary, depth)


def _leaf(width: float, height: float, outward: float, bottom: float,
          angle: float, reach: float) -> trimesh.Trimesh:
    """A thick, lobed acanthus leaf with a raised midrib and recurved tip."""
    t = np.array([0, 0.16, 0.28, 0.40, 0.51, 0.63, 0.74, 0.85, 0.94, 1.0])
    spread = np.array([0.24, 0.75, 0.57, 1.0, 0.69, 0.88, 0.56, 0.55, 0.20, 0.055])
    theta = np.arange(12) * (2 * np.pi / 12)
    rings = np.empty((len(t), len(theta), 3))
    rings[:, :, 0] = width * spread[:, None] * np.cos(theta)[None]
    rings[:, :, 1] = bottom + height * t[:, None]
    bow = outward + reach * (t ** 1.7) - reach * 0.18 * np.maximum(0, (t - 0.88) / 0.12)
    rings[:, :, 2] = bow[:, None] + height * (0.035 + 0.024 * np.sin(np.pi * t[:, None])) * np.sin(theta)[None]
    mesh = _closed_loft(rings)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
    return mesh


@lru_cache(maxsize=48)
def _column_template(order: str, radius: float, shaft_height: float,
                     base_height: float, base_width: float, capital_height: float,
                     capital_width: float) -> tuple[ColumnPart, ...]:
    parts: list[ColumnPart] = []

    def add(element: str, mesh: trimesh.Trimesh) -> None:
        parts.append(ColumnPart(element, mesh))

    y = 0.0
    if order == "Doric":
        shaft_height += base_height
    elif base_height:
        h, w = base_height, base_width
        add("base", _box(w, h * 0.22, w, 0))
        if order == "Tuscan":
            add("base_upper", _lathe([(h * t, r) for t, r in
                [(0.22, w * 0.39), (0.30, w * 0.44), (0.46, w * 0.47),
                 (0.63, w * 0.44), (0.76, radius * 1.05), (1.0, radius)]]))
        else:
            # Lower torus, hollow scotia and upper torus form an Attic profile.
            add("base_lower", _lathe([(h * t, w * r) for t, r in
                [(0.22, 0.40), (0.25, 0.455), (0.31, 0.48), (0.38, 0.47), (0.44, 0.42)]]))
            add("base_scotia", _lathe([(h * t, w * r) for t, r in
                [(0.44, 0.42), (0.48, 0.42), (0.51, 0.35), (0.59, 0.31),
                 (0.68, 0.33), (0.73, 0.38), (0.76, 0.38)]]))
            add("base_upper", _lathe([(h * t, r) for t, r in
                [(0.76, w * 0.38), (0.80, w * 0.415), (0.87, w * 0.425),
                 (0.94, w * 0.39), (1.0, radius)]]))
        y = h
    flutes = 20 if order == "Doric" else (0 if order == "Tuscan" else 24)
    taper = 0.83 if order == "Doric" else 0.86
    shaft = fluted_shaft(radius, shaft_height, flutes=flutes,
                         fillet=0 if flutes != 24 else 0.18, top_ratio=taper)
    shaft.apply_translation([0, y, 0])
    add("shaft", shaft)
    y += shaft_height
    h, w, neck = capital_height, capital_width, radius * taper
    if order in {"Doric", "Tuscan"}:
        add("capital_neck", _lathe([(y, neck), (y + h * 0.17, neck)]))
        profile = [(y + h * 0.12, neck)]
        for start in (0.14, 0.21, 0.28):
            profile.extend([(y + h * start, neck),
                            (y + h * (start + 0.015), neck * 1.055),
                            (y + h * (start + 0.045), neck * 1.055),
                            (y + h * (start + 0.060), neck)])
        add("capital_annuli", _lathe(profile))
        add("capital_echinus", _lathe([(y + h * t, r) for t, r in
            [(0.31, neck), (0.38, neck * 1.09), (0.47, w * 0.39),
             (0.57, w * 0.455), (0.66, w * 0.475), (0.70, w * 0.478)]]))
        add("capital_abacus", _box(w, h * 0.30, w, y + h * 0.70))
    else:
        add("shaft_astragal", _lathe([(y + h * t, neck * r) for t, r in
            [(0, 1.0), (0.025, 1.07), (0.055, 1.09), (0.085, 1.03)]]))
        ornate = order in {"Corinthian", "Composite"}
        if ornate:
            add("capital_bell", _lathe([(y + h * t, r) for t, r in
                [(0.06, neck), (0.21, neck * 1.04), (0.48, w * 0.30),
                 (0.70, w * 0.35), (0.89, w * 0.42)]]))
            for tier in range(2):
                for index in range(8):
                    add("capital_foliage", _leaf(w * 0.10, h * (0.46 if tier == 0 else 0.52),
                        neck * (0.97 if tier == 0 else 1.13), y + h * (0.10 + tier * 0.23),
                        index * np.pi / 4 + tier * np.pi / 8, w * (0.20 if tier == 0 else 0.19)))
        else:
            add("capital_neck", _lathe([(y + h * t, r) for t, r in
                [(0.05, neck), (0.15, neck * 1.05), (0.27, w * 0.285),
                 (0.40, w * 0.31), (0.54, w * 0.31)]]))
        if order in {"Ionic", "Composite"}:
            theta = np.arange(32) * 2 * np.pi / 32
            stations = np.array([-0.40, -0.34, -0.21, 0, 0.21, 0.34, 0.40])
            fullness = np.array([0.82, 1, 0.95, 0.90, 0.95, 1, 0.82])
            center = 0.68 if order == "Ionic" else 0.77
            # The Ionic bolster enters the abacus slightly; its bearing must
            # not depend solely on tangent contact at the ornamental scrolls.
            cushion_height = 0.21 if order == "Ionic" else 0.10
            rings = np.empty((len(stations), len(theta), 3))
            rings[:, :, 0] = w * stations[:, None]
            rings[:, :, 1] = y + h * center + h * cushion_height * fullness[:, None] * np.cos(theta)[None]
            rings[:, :, 2] = w * 0.275 * fullness[:, None] * np.sin(theta)[None]
            add("capital_cushion", _closed_loft(rings))
            vr = min(h * (0.34 if order == "Ionic" else 0.18), w * 0.21)
            for side in (-1, 1):
                for face in (-1, 1):
                    volute = spiral_volute(vr, w * 0.075, hand=-side)
                    volute.apply_translation([side * (w * 0.5 - vr),
                                              y + h * (0.54 if order == "Ionic" else 0.72),
                                              face * w * 0.2925])
                    add("capital_volute", volute)
        else:
            # Eight small scrolls turn toward the four abacus corners.
            for index in range(8):
                angle = index * np.pi / 4
                helix = spiral_volute(min(h * 0.14, w * 0.12), w * 0.052,
                                      hand=1 if index % 2 else -1)
                helix.apply_translation([0, y + h * 0.72, w * 0.36])
                helix.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
                add("capital_helix", helix)
        add("capital_abacus", _box(w, h * 0.12, w * (0.68 if order == "Ionic" else 1.0),
                                    y + h * 0.88))
    return tuple(parts)


def column_meshes(order: str, radius: float, shaft_height: float,
                  base_height: float, base_width: float, capital_height: float,
                  capital_width: float) -> tuple[ColumnPart, ...]:
    """Return detached local solids spanning exactly the supplied total height.

    Greek Doric consumes a caller's legacy base allowance in the shaft. Cached
    templates are never returned directly, so editing one column cannot mutate
    another column or a later export.
    """
    if order not in {"Doric", "Ionic", "Tuscan", "Corinthian", "Composite"}:
        raise ValueError(f"unsupported classical order: {order}")
    values = tuple(_positive(value, name, allow_zero=name == "base_height")
                   for name, value in [("radius", radius), ("shaft_height", shaft_height),
                   ("base_height", base_height), ("base_width", base_width),
                   ("capital_height", capital_height), ("capital_width", capital_width)])
    return tuple(ColumnPart(part.element, part.mesh.copy())
                 for part in _column_template(order, *values))
