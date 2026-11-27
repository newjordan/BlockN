"""Per-member colour, weathering, and real-scale UVs for compiled builds.

Architects keep the palette colour they authored. This module paints the
compiled meshes from that finish and the substance registry: a deterministic
tone, then the weathering profile driven by ``age`` and by exposure measured
per face. Mass and centroid are read from the physical report and are not
measured again.

The compiled solid stays welded, so it remains watertight. Vertex colours on
that mesh are only a display average. Exports split vertices at hard edges so
each flat face keeps its own exposure and its box-projected UVs seam there.
Curved members weld back together where the normals agree.

glTF multiplies ``baseColorFactor`` by the vertex colour, both in linear
space. In memory the factor and the multipliers stay sRGB, which is what OBJ
and PLY store. The glTF writer converts on the way out, and
``absolute_vertex_colors`` decodes a loaded linear pair back to sRGB. The
placement's alpha is kept, so a pane authored below 255 stays translucent.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from trimesh.visual.color import linear_to_srgb, srgb_to_linear

from .physics import PhysicalReport
from .substances import (
    FINISH_TONE_COUNT,
    SUBSTANCES,
    WEATHERING,
    Substance,
    WeatheringProfile,
    finish_tone,
)


TONE_COUNT = 8
# One UV unit is one metre, so a later texture tiles at real-world scale.
UV_SCALE_M = 1.0
# An overhang that stands p metres proud of a wall shades points up to
# RAIN_SHADOW_RATIO * p below the overhang's base. Flush masonry has p = 0 and
# casts no shadow. 2.0 is wind-driven rain at one unit out for two units down.
RAIN_SHADOW_RATIO = 2.0
# A lip shallower than this does not count as a projection.
PROJECTION_MIN = 0.05
_SHELTER_TOLERANCE = 1e-3
_SAMPLE_GRID = 3
_GRID_CELL = 1.0
# Faces steeper than this are walls. Shallower faces see the sky.
_UP_LIMIT = 0.5
# Weld export vertices whose normals agree within about 40 degrees so a
# cylinder stays smooth and a box stays faceted.
_HARD_EDGE_DOT = math.cos(math.radians(40.0))
# Box UVs are in metres. A seam jumps by much more than this.
_UV_WELD_M = 1e-3

if FINISH_TONE_COUNT != TONE_COUNT:
    raise RuntimeError("finish tones and surface tones must use the same count")


def tone_index(seed: int, member_id: str) -> int:
    """Return a stable tone in ``0 .. TONE_COUNT-1`` for one member."""
    digest = hashlib.sha256(f"{int(seed)}:{member_id}".encode("utf-8")).digest()
    return digest[0] % TONE_COUNT


def tone_color(substance: Substance, index: int) -> Tuple[int, int, int]:
    """Quantised sRGB colour between the substance's low and high ends."""
    span = max(TONE_COUNT - 1, 1)
    blend = float(int(index) % TONE_COUNT) / float(span)
    return tuple(
        int(round(low + (high - low) * blend))
        for low, high in zip(substance.color_low, substance.color_high)
    )


def face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Weathered sRGB for each face, or an empty array when the mesh is unpainted."""
    stored = getattr(mesh, "_face_srgb", None)
    if stored is None:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(stored, dtype=np.uint8)


def face_shelter(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-face shelter in 0..1. 1 is fully out of the rain."""
    stored = getattr(mesh, "_face_shelter", None)
    if stored is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(stored, dtype=np.float64)


def _member_grid(lo: np.ndarray, hi: np.ndarray, cell: float) -> Dict[Tuple[int, int], List[int]]:
    grid: Dict[Tuple[int, int], List[int]] = {}
    for index in range(len(lo)):
        ix0 = int(np.floor(lo[index, 0] / cell))
        ix1 = int(np.floor(hi[index, 0] / cell))
        iz0 = int(np.floor(lo[index, 2] / cell))
        iz1 = int(np.floor(hi[index, 2] / cell))
        for ix in range(ix0, ix1 + 1):
            for iz in range(iz0, iz1 + 1):
                grid.setdefault((ix, iz), []).append(index)
    return grid


def _shelter_fractions(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Fraction of each member's top that has another solid anywhere above it.

    Samples a 3 by 3 grid on the top face. A sample is covered when some other
    member's box contains that plan point and its base is at or above this
    member's top. Upward faces use this as their sky cover.
    """
    count = len(lo)
    shelter = np.zeros(count, dtype=np.float64)
    if count == 0:
        return shelter

    grid = _member_grid(lo, hi, _GRID_CELL)
    for index in range(count):
        width = float(hi[index, 0] - lo[index, 0])
        depth = float(hi[index, 2] - lo[index, 2])
        if width <= 1e-8 or depth <= 1e-8:
            samples = [(float(lo[index, 0] + hi[index, 0]) * 0.5, float(lo[index, 2] + hi[index, 2]) * 0.5)]
        else:
            pad_x = min(width * 0.05, width * 0.49)
            pad_z = min(depth * 0.05, depth * 0.49)
            xs = np.linspace(lo[index, 0] + pad_x, hi[index, 0] - pad_x, _SAMPLE_GRID)
            zs = np.linspace(lo[index, 2] + pad_z, hi[index, 2] - pad_z, _SAMPLE_GRID)
            samples = [(float(x), float(z)) for x in xs for z in zs]
        top = float(hi[index, 1])
        hits = 0
        for x, z in samples:
            ix = int(np.floor(x / _GRID_CELL))
            iz = int(np.floor(z / _GRID_CELL))
            for other in grid.get((ix, iz), ()):
                if other == index or float(lo[other, 1]) < top - _SHELTER_TOLERANCE:
                    continue
                if lo[other, 0] - 1e-9 <= x <= hi[other, 0] + 1e-9 and lo[other, 2] - 1e-9 <= z <= hi[other, 2] + 1e-9:
                    hits += 1
                    break
        shelter[index] = hits / float(len(samples))
    return shelter


def _inset_linspace(start: float, end: float) -> np.ndarray:
    span = float(end - start)
    pad = min(abs(span) * 0.08, abs(span) * 0.49) if abs(span) > 1e-8 else 0.0
    if span >= 0.0:
        return np.linspace(start + pad, end - pad, _SAMPLE_GRID)
    return np.linspace(start - pad, end + pad, _SAMPLE_GRID)


def _rain_shadow(points: np.ndarray, axis: np.ndarray, sign: np.ndarray, plane: np.ndarray,
                 index: int, grid: Dict[Tuple[int, int], List[int]], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """True where a point on a side face sits in a projecting member's rain shadow."""
    count = len(points)
    hit = np.zeros(count, dtype=bool)
    if count == 0:
        return hit
    ix = np.floor(points[:, 0] / _GRID_CELL).astype(np.int64)
    iz = np.floor(points[:, 2] / _GRID_CELL).astype(np.int64)
    seen: set[int] = set()
    chosen: List[int] = []
    for key in set(zip(ix.tolist(), iz.tolist())):
        for other in grid.get(key, ()):
            if other != index and other not in seen:
                seen.add(other)
                chosen.append(other)
    if not chosen:
        return hit
    clo = lo[np.asarray(chosen, dtype=np.int64)]
    chi = hi[np.asarray(chosen, dtype=np.int64)]
    for ax, sg in ((0, 1.0), (0, -1.0), (2, 1.0), (2, -1.0)):
        selected = (axis == ax) & (sign == sg)
        if not np.any(selected):
            continue
        pts = points[selected]
        pl = plane[selected]
        along = 2 if ax == 0 else 0
        if sg > 0.0:
            projection = chi[None, :, ax] - pl[:, None]
            reaches = clo[None, :, ax] <= (pl[:, None] + PROJECTION_MIN)
        else:
            projection = pl[:, None] - clo[None, :, ax]
            reaches = chi[None, :, ax] >= (pl[:, None] - PROJECTION_MIN)
        along_value = pts[:, along]
        covers = (
            (clo[None, :, along] - 1e-4 <= along_value[:, None])
            & (along_value[:, None] <= chi[None, :, along] + 1e-4)
        )
        gap = clo[None, :, 1] - pts[:, None, 1]
        shadowed = (
            (projection >= PROJECTION_MIN)
            & reaches
            & covers
            & (gap >= -1e-3)
            & (gap <= RAIN_SHADOW_RATIO * np.maximum(projection, 0.0))
        )
        hit[selected] = np.any(shadowed, axis=1)
    return hit


def _box_side_shelter(origin: np.ndarray, extent: np.ndarray, index: int,
                      grid: Dict[Tuple[int, int], List[int]], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Shelter fraction for the four vertical faces of an axis-aligned box, +X -X +Z -Z."""
    specs = (
        (0, 1.0, 1, 2, float(extent[0])),
        (0, -1.0, 1, 2, float(origin[0])),
        (2, 1.0, 0, 1, float(extent[2])),
        (2, -1.0, 0, 1, float(origin[2])),
    )
    points: List[np.ndarray] = []
    axes: List[np.ndarray] = []
    signs: List[np.ndarray] = []
    planes: List[np.ndarray] = []
    for axis, sign, vary_a, vary_b, plane in specs:
        aa = _inset_linspace(float(origin[vary_a]), float(extent[vary_a]))
        bb = _inset_linspace(float(origin[vary_b]), float(extent[vary_b]))
        grid_a, grid_b = np.meshgrid(aa, bb, indexing="ij")
        flat_a = grid_a.reshape(-1)
        flat_b = grid_b.reshape(-1)
        sample = np.zeros((flat_a.size, 3), dtype=np.float64)
        sample[:, axis] = plane
        sample[:, vary_a] = flat_a
        sample[:, vary_b] = flat_b
        points.append(sample)
        axes.append(np.full(flat_a.size, axis, dtype=np.int64))
        signs.append(np.full(flat_a.size, sign, dtype=np.float64))
        planes.append(np.full(flat_a.size, plane, dtype=np.float64))
    stacked = np.vstack(points)
    shadowed = _rain_shadow(
        stacked, np.concatenate(axes), np.concatenate(signs), np.concatenate(planes),
        index, grid, lo, hi,
    )
    per_face = stacked.shape[0] // 4
    return np.array([float(np.mean(shadowed[i * per_face:(i + 1) * per_face])) for i in range(4)], dtype=np.float64)


def _triangle_side_shelter(tris: np.ndarray, normals: np.ndarray, index: int,
                           grid: Dict[Tuple[int, int], List[int]], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    count = len(tris)
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    centroid = tris.mean(axis=1)
    order = np.argsort(tris[:, :, 1], axis=1)
    rows = np.arange(count)
    low_pt = tris[rows, order[:, 0]]
    high_pt = tris[rows, order[:, 2]]
    samples = np.stack((
        centroid,
        0.45 * centroid + 0.55 * high_pt,
        0.45 * centroid + 0.55 * low_pt,
    ), axis=1)
    use_x = np.abs(normals[:, 0]) >= np.abs(normals[:, 2])
    axis = np.where(use_x, 0, 2).astype(np.int64)
    component = np.where(use_x, normals[:, 0], normals[:, 2])
    sign = np.where(component >= 0.0, 1.0, -1.0)
    plane = centroid[rows, axis]
    flat = samples.reshape(-1, 3)
    shadowed = _rain_shadow(
        flat, np.repeat(axis, 3), np.repeat(sign, 3), np.repeat(plane, 3),
        index, grid, lo, hi,
    )
    return shadowed.reshape(count, 3).mean(axis=1)


def _is_placement_box(mesh: trimesh.Trimesh, origin: np.ndarray, extent: np.ndarray) -> bool:
    if len(mesh.faces) != 12 or len(mesh.vertices) > 8:
        return False
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return bool(
        np.all(np.abs(vertices.min(axis=0) - origin) < 1e-4)
        and np.all(np.abs(vertices.max(axis=0) - extent) < 1e-4)
    )


def _face_shelters(
    meshes: Sequence[Optional[trimesh.Trimesh]],
    lo: np.ndarray,
    hi: np.ndarray,
    sky: np.ndarray,
) -> List[Optional[np.ndarray]]:
    """Shelter per face. Upward faces use sky cover, soffits are covered, sides use rain shadow."""
    grid = _member_grid(lo, hi, _GRID_CELL)
    shelters: List[Optional[np.ndarray]] = []
    for index, mesh in enumerate(meshes):
        if mesh is None or mesh.is_empty or len(mesh.faces) == 0:
            shelters.append(None)
            continue
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        shelter = np.zeros(len(normals), dtype=np.float64)
        upward = normals[:, 1] >= _UP_LIMIT
        downward = normals[:, 1] <= -_UP_LIMIT
        shelter[upward] = float(sky[index])
        shelter[downward] = 1.0
        side = ~(upward | downward)
        if not np.any(side):
            shelters.append(shelter)
            continue
        if _is_placement_box(mesh, lo[index], hi[index]):
            per_side = _box_side_shelter(lo[index], hi[index], index, grid, lo, hi)
            labels = np.full(len(normals), -1, dtype=np.int64)
            labels[normals[:, 0] > 0.5] = 0
            labels[normals[:, 0] < -0.5] = 1
            labels[normals[:, 2] > 0.5] = 2
            labels[normals[:, 2] < -0.5] = 3
            for label, fraction in enumerate(per_side):
                shelter[(labels == label) & side] = fraction
            missed = side & (labels < 0)
            if np.any(missed):
                faces = np.asarray(mesh.faces)[missed]
                shelter[missed] = _triangle_side_shelter(
                    np.asarray(mesh.vertices, dtype=np.float64)[faces],
                    normals[missed], index, grid, lo, hi,
                )
        else:
            faces = np.asarray(mesh.faces)[side]
            shelter[side] = _triangle_side_shelter(
                np.asarray(mesh.vertices, dtype=np.float64)[faces],
                normals[side], index, grid, lo, hi,
            )
        shelters.append(shelter)
    return shelters


def _box_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    """Box-project vertices onto the plane of their dominant normal, in metres."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    return _project_uv(vertices, normals)


def _project_uv(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    dominant = np.argmax(np.abs(normals), axis=1)
    uv = np.empty((len(points), 2), dtype=np.float64)
    on_x = dominant == 0
    on_y = dominant == 1
    on_z = dominant == 2
    uv[on_x] = points[on_x][:, [1, 2]]
    uv[on_y] = points[on_y][:, [0, 2]]
    uv[on_z] = points[on_z][:, [0, 1]]
    return uv / UV_SCALE_M


def _mix(color: np.ndarray, tint: Sequence[int], amount: np.ndarray) -> None:
    weight = np.clip(amount, 0.0, 1.0)[:, None]
    target = np.asarray(tint, dtype=np.float64)
    color *= (1.0 - weight)
    color += weight * target


def _reference_rgb(tone: Sequence[int], profile: WeatheringProfile) -> np.ndarray:
    """Brightest sRGB the tone and its tints can reach, per channel."""
    columns = [
        np.asarray(tone, dtype=np.float64),
        np.asarray(profile.biofilm.color, dtype=np.float64),
        np.asarray(profile.soot.color, dtype=np.float64),
        np.asarray(profile.patina.color, dtype=np.float64),
        np.asarray(profile.erosion_color, dtype=np.float64),
    ]
    return np.maximum.reduce(columns)


def _channel(value: Any, count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(count, float(array[0]), dtype=np.float64)
    if array.size != count:
        raise ValueError("weathering channel length does not match the samples")
    return array


def _weather_vertices(
    tone: Sequence[int],
    profile: WeatheringProfile,
    age: float,
    upward: np.ndarray,
    shelter: float,
    height: np.ndarray,
) -> np.ndarray:
    """Return weathered sRGB for each sample. Age 0 leaves the tone untouched.

    ``upward`` is 1 on faces that see the sky and 0 on walls and soffits, so
    biofilm and erosion stay off vertical faces. ``shelter`` is 1 in rain
    shadow, under a cover, or on a soffit. Open patina follows the unshaded
    fraction. The age term of biofilm and soot is small, so a rain-washed wall
    only drifts.
    """
    count = len(height)
    color = np.tile(np.asarray(tone, dtype=np.float64), (count, 1))
    if age <= 0.0 or count == 0:
        return np.clip(np.rint(color), 0, 255).astype(np.uint8)

    upward_channel = _channel(upward, count)
    sheltered = _channel(shelter, count)
    rain = upward_channel * (1.0 - sheltered)
    if profile.patina_on == "shelter":
        patina_channel = sheltered
    else:
        patina_channel = 1.0 - sheltered

    _mix(color, profile.biofilm.color, age * (profile.biofilm.age + profile.biofilm.exposure * rain))
    _mix(color, profile.soot.color, age * (profile.soot.age + profile.soot.exposure * sheltered))
    _mix(color, profile.patina.color, age * (profile.patina.age + profile.patina.exposure * patina_channel))
    _mix(color, profile.erosion_color, age * profile.erosion_strength * rain)
    if profile.grime_height > 0.0 and profile.grime_strength > 0.0:
        splash = np.clip((profile.grime_height - height) / profile.grime_height, 0.0, 1.0)
        _mix(color, profile.soot.color, age * profile.grime_strength * splash)
    return np.clip(np.rint(color), 0, 255).astype(np.uint8)


def _multipliers(absolute: np.ndarray, reference: np.ndarray, alpha: int) -> np.ndarray:
    """sRGB vertex colours that reconstruct ``absolute`` when multiplied by ``reference``."""
    scale = np.maximum(reference.astype(np.float64), 1.0)
    rgb = np.clip(np.rint(absolute.astype(np.float64) / scale * 255.0), 0, 255).astype(np.uint8)
    # The factor's alpha stays 1, so this channel is the authored translucency
    # and is not applied twice.
    alpha_column = np.full((len(absolute), 1), int(alpha), dtype=np.uint8)
    return np.concatenate([rgb, alpha_column], axis=1)


def _factor_rgb(factor: Any) -> np.ndarray:
    base = np.asarray(factor, dtype=np.float64).reshape(-1)
    if base.size < 3:
        return np.ones(3, dtype=np.float64)
    rgb = base[:3]
    if float(rgb.max()) > 1.0:
        rgb = rgb / 255.0
    return rgb


def absolute_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """sRGB the product ``baseColorFactor * COLOR_0`` resolves to.

    uint8 multipliers are the in-memory and OBJ/PLY encoding: they are sRGB
    ratios. Float multipliers in 0..1 are the glTF encoding: linear ratios
    against a linear factor, decoded back to sRGB.
    """
    raw = None
    visual = getattr(mesh, "visual", None)
    if visual is not None and hasattr(visual, "vertex_attributes"):
        raw = visual.vertex_attributes.get("color")
    if raw is None and visual is not None and getattr(visual, "kind", None) in {"vertex", "face"}:
        return np.asarray(visual.vertex_colors)
    if raw is None:
        return np.tile(np.array([255, 255, 255, 255], dtype=np.uint8), (len(mesh.vertices), 1))
    material = getattr(visual, "material", None)
    factor = getattr(material, "baseColorFactor", None)
    colors = np.asarray(raw)
    if factor is None:
        return colors
    base = _factor_rgb(factor)
    if np.issubdtype(colors.dtype, np.floating):
        linear = np.clip(colors[:, :3].astype(np.float64) * base, 0.0, None)
        rgb = np.clip(np.rint(linear_to_srgb(linear) * 255.0), 0, 255).astype(np.uint8)
        if colors.shape[1] >= 4:
            alpha = np.clip(np.rint(colors[:, 3].astype(np.float64) * 255.0), 0, 255).astype(np.uint8)
        else:
            alpha = np.full(len(colors), 255, dtype=np.uint8)
        return np.column_stack([rgb, alpha])
    rgb = np.clip(np.rint(colors[:, :3].astype(np.float64) * base), 0, 255).astype(np.uint8)
    alpha = colors[:, 3] if colors.shape[1] >= 4 else np.full(len(colors), 255, dtype=np.uint8)
    return np.column_stack([rgb, np.asarray(alpha, dtype=np.uint8)])


def _paint(
    mesh: trimesh.Trimesh,
    material: trimesh.visual.material.PBRMaterial,
    absolute: np.ndarray,
    reference: np.ndarray,
    alpha: int,
) -> None:
    uv = _box_uv(mesh)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.visual.vertex_attributes["color"] = _multipliers(absolute, reference, alpha)
    if alpha < 255:
        material.alphaMode = "BLEND"


def _material(
    cache: Dict[Tuple, trimesh.visual.material.PBRMaterial],
    name: str,
    reference: np.ndarray,
    roughness: float,
    metallic: float,
) -> trimesh.visual.material.PBRMaterial:
    key = (name, tuple(int(channel) for channel in reference), round(roughness, 4), round(metallic, 4))
    material = cache.get(key)
    if material is None:
        material = trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=[*(float(channel) / 255.0 for channel in reference), 1.0],
            roughnessFactor=float(roughness),
            metallicFactor=float(metallic),
            doubleSided=True,
        )
        cache[key] = material
    return material


def _roughness(substance: Substance, profile: WeatheringProfile, age: float) -> float:
    return float(np.clip(substance.roughness + profile.roughness_age * age, 0.0, 1.0))


def _member_metadata(
    *,
    substance: Optional[str],
    mass_kg: float,
    density: float,
    role: str,
    kind: str,
    centroid: Sequence[float],
    roughness: float,
    color: Sequence[int],
    tone: Optional[int],
    friction: Optional[float],
) -> Dict[str, Any]:
    return {
        "substance": substance,
        "mass_kg": round(float(mass_kg), 3),
        "density": round(float(density), 3),
        "role": str(role),
        "kind": str(kind),
        "centroid": [round(float(value), 6) for value in centroid],
        "roughness": round(float(roughness), 4),
        "color": [int(channel) for channel in color],
        "tone": None if tone is None else int(tone),
        "friction": None if friction is None else round(float(friction), 4),
        "uv_scale_m": UV_SCALE_M,
    }


def _vertices_from_faces(mesh: trimesh.Trimesh, face_srgb: np.ndarray) -> np.ndarray:
    """Average incident face colours. Export splits the hard edges this average blends."""
    absolute = np.zeros((len(mesh.vertices), 3), dtype=np.float64)
    counts = np.zeros(len(mesh.vertices), dtype=np.float64)
    faces = np.asarray(mesh.faces)
    for corner in range(faces.shape[1]):
        np.add.at(absolute, faces[:, corner], face_srgb)
        np.add.at(counts, faces[:, corner], 1.0)
    counts = np.maximum(counts, 1.0)
    return np.clip(np.rint(absolute / counts[:, None]), 0, 255).astype(np.uint8)


def _split_hard_edges(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Export copy with one vertex group per flat face. The source mesh is unchanged.

    Corners weld when they coincide, their normals stay within ``_HARD_EDGE_DOT``,
    and their box UVs agree. A curved member therefore keeps smooth shading, and
    a box keeps a seam on every edge.
    """
    face_srgb = getattr(mesh, "_face_srgb", None)
    if face_srgb is None or len(face_srgb) != len(mesh.faces) or len(mesh.faces) == 0:
        copied = mesh.copy()
        stored = None
        visual = getattr(mesh, "visual", None)
        if visual is not None and hasattr(visual, "vertex_attributes"):
            raw = visual.vertex_attributes.get("color")
            if raw is not None and len(raw) == len(mesh.vertices):
                stored = np.array(raw, copy=True)
        if stored is not None and hasattr(copied.visual, "vertex_attributes"):
            copied.visual.vertex_attributes["color"] = stored
        return copied

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    corners = vertices[faces]
    face_count = len(faces)
    corner_uv = _project_uv(corners.reshape(-1, 3), np.repeat(normals, 3, axis=0)).reshape(face_count, 3, 2)
    corner_color = np.repeat(np.asarray(face_srgb, dtype=np.uint8)[:, None, :], 3, axis=1)
    quant = np.round(corners / 1e-5).astype(np.int64)

    groups: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = {}
    for face_index in range(face_count):
        for corner_index in range(3):
            key = (
                int(quant[face_index, corner_index, 0]),
                int(quant[face_index, corner_index, 1]),
                int(quant[face_index, corner_index, 2]),
            )
            groups.setdefault(key, []).append((face_index, corner_index))

    new_vertices: List[np.ndarray] = []
    new_normals: List[np.ndarray] = []
    new_uv: List[np.ndarray] = []
    new_color: List[np.ndarray] = []
    indexed = np.empty((face_count, 3), dtype=np.int64)
    for items in groups.values():
        clusters: List[dict] = []
        for face_index, corner_index in items:
            normal = normals[face_index]
            uv = corner_uv[face_index, corner_index]
            matched = None
            for cluster in clusters:
                if float(np.dot(cluster["normal"], normal)) < _HARD_EDGE_DOT:
                    continue
                if float(np.linalg.norm(cluster["uv"] - uv)) > _UV_WELD_M:
                    continue
                # Grime and rain shadow can differ across one flat face. Keep the
                # first colour so the exported vertex stays on an authored sample.
                if float(np.max(np.abs(cluster["color"] - corner_color[face_index, corner_index]))) > 2.0:
                    continue
                matched = cluster
                break
            if matched is None:
                matched = {
                    "normal": normal.astype(np.float64).copy(),
                    "uv": uv.astype(np.float64).copy(),
                    "color": corner_color[face_index, corner_index].astype(np.float64).copy(),
                    "count": 0,
                    "items": [],
                }
                clusters.append(matched)
            matched["count"] += 1
            weight = 1.0 / float(matched["count"])
            matched["normal"] = matched["normal"] * (1.0 - weight) + normal * weight
            matched["items"].append((face_index, corner_index))
        for cluster in clusters:
            slot = len(new_vertices)
            face_index, corner_index = cluster["items"][0]
            new_vertices.append(corners[face_index, corner_index])
            new_uv.append(cluster["uv"])
            new_color.append(cluster["color"])
            length = float(np.linalg.norm(cluster["normal"]))
            new_normals.append(cluster["normal"] / length if length > 1e-8 else cluster["normal"])
            for face_index, corner_index in cluster["items"]:
                indexed[face_index, corner_index] = slot

    exported = trimesh.Trimesh(
        vertices=np.asarray(new_vertices, dtype=np.float64),
        faces=indexed,
        vertex_normals=np.asarray(new_normals, dtype=np.float64),
        process=False,
    )
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    reference = _factor_rgb(getattr(material, "baseColorFactor", (1.0, 1.0, 1.0, 1.0))) * 255.0
    absolute = np.clip(np.rint(np.asarray(new_color, dtype=np.float64)), 0, 255).astype(np.uint8)
    alpha = 255
    if visual is not None and hasattr(visual, "vertex_attributes"):
        stored = visual.vertex_attributes.get("color")
        if stored is not None and len(stored) and np.shape(stored)[1] >= 4:
            alpha = int(stored[0, 3])
    exported.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(new_uv, dtype=np.float64),
        material=material,
    )
    exported.visual.vertex_attributes["color"] = _multipliers(absolute, reference, alpha)
    exported.metadata = dict(getattr(mesh, "metadata", {}) or {})
    return exported


def srgb_u8_to_linear(values: np.ndarray) -> np.ndarray:
    """Linear RGB in 0..1 from sRGB bytes or 0..1 sRGB fractions."""
    array = np.asarray(values, dtype=np.float64)
    if array.size and float(np.max(array)) > 1.0:
        array = array / 255.0
    return srgb_to_linear(array)


def apply_surfaces(document: Any, scene: Any, physical: PhysicalReport) -> None:
    """Paint compiled meshes. Geometry, placements, and the physical report stay put."""
    placements = list(getattr(document, "placements", ()) or ())
    blocks = list(getattr(scene, "blocks", ()) or ())
    if not placements or physical is None:
        scene.export_metadata = {
            "uv_scale_m": UV_SCALE_M,
            "age": float(getattr(getattr(document, "request", None), "age", 0.0) or 0.0),
            "total_mass_kg": 0.0,
            "bill_of_materials": {},
        }
        return

    age = float(getattr(document.request, "age", 0.0))
    seed = int(getattr(document.request, "seed", 0))
    lo = np.empty((len(placements), 3), dtype=np.float64)
    hi = np.empty((len(placements), 3), dtype=np.float64)
    for index, placement in enumerate(placements):
        center = np.asarray(placement.position, dtype=np.float64)
        half = np.asarray(placement.dimensions, dtype=np.float64) / 2.0
        lo[index] = center - half
        hi[index] = center + half
    sky = _shelter_fractions(lo, hi)

    by_id = {member.primitive_id: member for member in physical.members}
    materials: Dict[Tuple, trimesh.visual.material.PBRMaterial] = {}
    block_by_id = {}
    for block in blocks:
        parameters = getattr(block, "parameters", {}) or {}
        primitive_id = parameters.get("primitive_id")
        if primitive_id is not None:
            block_by_id[str(primitive_id)] = block

    member_meshes: List[Optional[trimesh.Trimesh]] = []
    for placement in placements:
        block = block_by_id.get(str(placement.id))
        mesh = getattr(block, "mesh", None) if block is not None else None
        if mesh is not None and getattr(mesh, "is_empty", True):
            mesh = None
        member_meshes.append(mesh)
    shelters = _face_shelters(member_meshes, lo, hi, sky)

    for index, placement in enumerate(placements):
        primitive_id = str(placement.id)
        block = block_by_id.get(primitive_id)
        mesh = member_meshes[index]
        member = by_id.get(primitive_id)
        substance_id = str(placement.substance) if getattr(placement, "substance", None) else ""
        authored = tuple(int(channel) for channel in placement.material)
        alpha = authored[3] if len(authored) >= 4 else 255
        substance = SUBSTANCES.get(substance_id)
        profile = WEATHERING.get(substance_id)
        mass_kg = float(member.mass_kg) if member is not None else 0.0
        centroid = member.centroid_m if member is not None else tuple(float(value) for value in placement.position)

        if substance is None or profile is None or mesh is None:
            rgba = tuple(int(channel) for channel in placement.material)
            tone_rgb = rgba[:3]
            roughness = 0.85
            metallic = 0.0
            name = f"unassigned.{tone_rgb[0]}.{tone_rgb[1]}.{tone_rgb[2]}"
            reference = np.asarray(tone_rgb, dtype=np.float64)
            absolute = (
                np.tile(np.asarray(tone_rgb, dtype=np.uint8), (len(mesh.vertices), 1))
                if mesh is not None else None
            )
            tone = None
            friction = None
            density = 0.0
            mean = tone_rgb
            face_srgb = None
        else:
            tone = tone_index(seed, primitive_id)
            tone_rgb = finish_tone(substance, authored[:3], tone)
            roughness = _roughness(substance, profile, age)
            metallic = float(substance.metallic)
            name = f"{substance.id}.tone{tone}"
            # Age 0 keeps the factor equal to the tone, so the vertex
            # multiplier is identity and the surface is that finish.
            reference = (
                np.asarray(tone_rgb, dtype=np.float64)
                if age <= 0.0
                else _reference_rgb(tone_rgb, profile)
            )
            normals = np.asarray(mesh.face_normals, dtype=np.float64)
            upward = (normals[:, 1] >= _UP_LIMIT).astype(np.float64)
            face_shelter_values = shelters[index]
            if face_shelter_values is None:
                face_shelter_values = np.zeros(len(mesh.faces), dtype=np.float64)
            face_srgb = _weather_vertices(
                tone_rgb, profile, age, upward, face_shelter_values,
                np.asarray(mesh.triangles_center, dtype=np.float64)[:, 1],
            )
            mesh._face_srgb = face_srgb
            mesh._face_shelter = np.asarray(face_shelter_values, dtype=np.float64)
            absolute = _vertices_from_faces(mesh, face_srgb)
            friction = float(substance.friction)
            density = float(substance.density)
            mean = tuple(int(channel) for channel in np.rint(face_srgb.mean(axis=0)))

        if mesh is not None and absolute is not None:
            material = _material(materials, name, reference, roughness, metallic)
            _paint(mesh, material, absolute, reference, alpha)
        if block is not None:
            block.color = (int(mean[0]), int(mean[1]), int(mean[2]), alpha)
            metadata = _member_metadata(
                substance=substance.id if substance is not None else None,
                mass_kg=mass_kg,
                density=density,
                role=str(placement.role),
                kind=str(placement.kind),
                centroid=centroid,
                roughness=roughness,
                color=mean,
                tone=tone,
                friction=friction,
            )
            authored_parameters = getattr(placement, "parameters", {}) or {}
            for key in ("plot_id", "building_product", "street_id"):
                if key in authored_parameters:
                    metadata[key] = authored_parameters[key]
            existing = dict(getattr(mesh, "metadata", {}) or {}) if mesh is not None else {}
            existing.update(metadata)
            if mesh is not None:
                mesh.metadata = existing
            parameters = dict(getattr(block, "parameters", {}) or {})
            parameters["surface"] = metadata
            block.parameters = parameters

    bill = physical.to_dict().get("bill_of_materials", {})
    scene.export_metadata = {
        "uv_scale_m": UV_SCALE_M,
        "age": round(age, 4),
        "total_mass_kg": round(float(physical.total_mass_kg), 3),
        "bill_of_materials": bill,
    }


__all__ = [
    "TONE_COUNT",
    "UV_SCALE_M",
    "RAIN_SHADOW_RATIO",
    "PROJECTION_MIN",
    "absolute_vertex_colors",
    "apply_surfaces",
    "face_colors",
    "face_shelter",
    "tone_color",
    "tone_index",
]
