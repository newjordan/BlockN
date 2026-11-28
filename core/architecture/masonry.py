"""Human-scale ashlar: walls are stacked stones, not slabs.

Ancient builders did not author a "wall" solid and cut holes in it. They set
courses of lift-able blocks in running bond; a doorway is the void they left,
closed by a lintel stone. This module is the only place that decides stone
size. Callers emit the returned stones as ordinary placements.

The cubit used here (0.444 m) is a shared Attic/Roman approximation. It is a
design module, not a claim that any monument was laid to that measure.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.architectural_geometry import extrude_profile

from .substances import SUBSTANCES


CUBIT = 0.444
MIN_STONE = 0.08
MAX_COURSE = 0.80
MAX_STRETCHER = 1.60
MAX_LINTEL = 2.40
MAX_STONES = 12_000
EPS = 1e-9

COURSE_TARGET = {"concept": CUBIT * 4 / 3, "standard": CUBIT, "presentation": CUBIT * 2 / 3}
STRETCHER_TARGET = {"concept": CUBIT * 8 / 3, "standard": CUBIT * 2, "presentation": CUBIT * 1.5}

Opening = Dict[str, Any]


@dataclass(frozen=True)
class AshlarStone:
    """One rectangular stone, center-based, Y-up."""

    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    course: int
    index: int
    unit: str
    axis: str
    opening_id: Optional[str] = None
    opening_index: Optional[int] = None


def _r(value: float) -> float:
    result = round(float(value), 6)
    return 0.0 if result == -0.0 else result


def _vec(values: Iterable[float]) -> Tuple[float, float, float]:
    out = tuple(_r(value) for value in values)
    if len(out) != 3:
        raise ValueError("expected a three-component vector")
    return out  # type: ignore[return-value]


def _detail(detail: str) -> str:
    key = str(detail or "standard").strip().lower()
    return key if key in COURSE_TARGET else "standard"


def course_target(detail: str = "standard") -> float:
    return COURSE_TARGET[_detail(detail)]


def stretcher_target(detail: str = "standard") -> float:
    return STRETCHER_TARGET[_detail(detail)]


def wall_axis(dimensions: Sequence[float]) -> str:
    """The long face of a box wall runs along X or Z."""
    width, _, depth = (float(value) for value in dimensions)
    return "x" if width >= depth else "z"


def _divisions(span: float, target: float, limit: Optional[float] = None) -> int:
    if span <= MIN_STONE * 1.5:
        return 1
    count = max(1, int(round(span / max(target, MIN_STONE))))
    cap = MAX_STRETCHER if limit is None else limit
    while span / count > cap + 1e-9:
        count += 1
    return count


def _uniform_edges(start: float, end: float, count: int) -> List[float]:
    if count <= 1:
        return [start, end]
    step = (end - start) / count
    return [start + step * index for index in range(count)] + [end]


def _offset_edges(start: float, end: float, count: int, offset: bool) -> List[float]:
    """Running-bond edges: odd courses start with a half-stone closer."""
    span = end - start
    if count <= 1 or span < MIN_STONE * 2 or not offset:
        return _uniform_edges(start, end, count)
    module = span / count
    half = module / 2.0
    if half < MIN_STONE:
        return _uniform_edges(start, end, count)
    edges = [start, start + half]
    x = start + half
    remaining = count - 1
    while remaining > 1 and x + module < end - MIN_STONE:
        x += module
        edges.append(x)
        remaining -= 1
    if edges[-1] < end - EPS:
        edges.append(end)
    else:
        edges[-1] = end
    return edges


def _pairs(edges: Sequence[float]) -> List[Tuple[float, float]]:
    pieces = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end - start >= MIN_STONE * 0.5:
            pieces.append((start, end))
        elif pieces:
            previous = pieces[-1]
            pieces[-1] = (previous[0], end)
        else:
            pieces.append((start, end))
    return [(a, b) for a, b in pieces if b - a > EPS]


def _subtract(intervals: Sequence[Tuple[float, float]], cut_start: float, cut_end: float) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for start, end in intervals:
        if cut_end <= start + EPS or cut_start >= end - EPS:
            result.append((start, end))
            continue
        if cut_start - start >= MIN_STONE * 0.5:
            result.append((start, min(end, cut_start)))
        if end - cut_end >= MIN_STONE * 0.5:
            result.append((max(start, cut_end), end))
    return [(a, b) for a, b in result if b - a > EPS]


def _merge_marks(marks: Iterable[float], protected: Iterable[float], height: float) -> List[float]:
    protected_set = {0.0, float(height), *(float(value) for value in protected)}
    ordered = sorted({float(value) for value in marks if -EPS <= value <= height + EPS})
    if not ordered:
        return [0.0, float(height)]
    merged = [ordered[0]]
    for mark in ordered[1:]:
        previous = merged[-1]
        if mark - previous >= MIN_STONE * 0.6:
            merged.append(mark)
            continue
        prev_protected = any(abs(previous - value) <= EPS for value in protected_set)
        mark_protected = any(abs(mark - value) <= EPS for value in protected_set)
        if mark_protected and not prev_protected:
            merged[-1] = mark
        elif mark_protected and prev_protected and mark - previous > EPS:
            merged.append(mark)
    if abs(merged[-1] - height) > EPS:
        merged.append(float(height))
    merged[0] = 0.0
    merged[-1] = float(height)
    return merged


def _stone(position: Sequence[float], dimensions: Sequence[float], course: int, index: int,
           unit: str, axis: str, opening_id: Optional[str] = None,
           opening_index: Optional[int] = None) -> AshlarStone:
    dims = _vec(dimensions)
    if any(value <= EPS for value in dims):
        raise ValueError(f"ashlar stone has non-positive dimensions {dims}")
    return AshlarStone(
        position=_vec(position), dimensions=dims, course=int(course), index=int(index),
        unit=str(unit), axis=str(axis), opening_id=opening_id, opening_index=opening_index,
    )


def iter_stack(bottom: float, height: float, detail: str = "standard") -> List[Tuple[float, float, int]]:
    """Split a vertical member into cubit-height drums. Returns (center_y, height, course)."""
    if height <= EPS:
        raise ValueError("stack height must be positive")
    count = _divisions(height, course_target(detail), MAX_COURSE)
    size = height / count
    return [(bottom + (index + 0.5) * size, size, index) for index in range(count)]


def iter_masonry_solid(
    center: Sequence[float],
    dimensions: Sequence[float],
    detail: str = "standard",
    max_stones: Optional[int] = None,
) -> List[AshlarStone]:
    """Fill a rectangular mass with running-bond ashlars.

    Y is always the bed direction. The longer horizontal axis carries the
    half-stone stagger. A thin wall becomes one wythe; a pier or podium
    becomes several stones in plan. Stone size coarsens toward MAX_COURSE
    and MAX_STRETCHER when the count would exceed ``max_stones``.
    """
    cx, cy, cz = (float(value) for value in center)
    dx, dy, dz = (float(value) for value in dimensions)
    if min(dx, dy, dz) <= EPS:
        raise ValueError("masonry solid dimensions must be positive")
    course_h = course_target(detail)
    stretcher = stretcher_target(detail)
    ny = _divisions(dy, course_h, MAX_COURSE)
    nx = _divisions(dx, stretcher, MAX_STRETCHER)
    nz = _divisions(dz, stretcher, MAX_STRETCHER)
    limit = MAX_STONES if max_stones is None else max(1, min(int(max_stones), MAX_STONES))
    while nx * ny * nz > limit and (course_h < MAX_COURSE or stretcher < MAX_STRETCHER):
        course_h = min(MAX_COURSE, course_h * 1.15)
        stretcher = min(MAX_STRETCHER, stretcher * 1.15)
        ny = _divisions(dy, course_h, MAX_COURSE)
        nx = _divisions(dx, stretcher, MAX_STRETCHER)
        nz = _divisions(dz, stretcher, MAX_STRETCHER)
    x0, y0, z0 = cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0
    y_edges = _uniform_edges(y0, y0 + dy, ny)
    along_x = dx >= dz
    axis = "x" if along_x else "z"
    x_module = dx / nx
    z_module = dz / nz
    stones: List[AshlarStone] = []
    for course, (y1, y2) in enumerate(_pairs(y_edges)):
        if along_x:
            x_spans = _clip_grid(x0, x0 + dx, x0, x_module, course % 2 == 1) if nx > 1 else [(x0, x0 + dx)]
            z_spans = _pairs(_uniform_edges(z0, z0 + dz, nz))
        else:
            x_spans = _pairs(_uniform_edges(x0, x0 + dx, nx))
            z_spans = _clip_grid(z0, z0 + dz, z0, z_module, course % 2 == 1) if nz > 1 else [(z0, z0 + dz)]
        index = 0
        for xa, xb in x_spans:
            for za, zb in z_spans:
                stones.append(_stone(
                    ((xa + xb) / 2.0, (y1 + y2) / 2.0, (za + zb) / 2.0),
                    (xb - xa, y2 - y1, zb - za),
                    course, index, "ashlar", axis,
                ))
                index += 1
    if not stones:
        raise ValueError("masonry solid produced no stones")
    return stones


def _clip_grid(start: float, end: float, origin: float, module: float, offset: bool) -> List[Tuple[float, float]]:
    """Clip a running-bond grid to a solid interval so joints continue through openings."""
    span = end - start
    if span <= MIN_STONE or module <= MIN_STONE:
        return [(start, end)] if span > EPS else []
    shift = (module / 2.0) if offset else 0.0
    edges = [start]
    k = int(math.floor((start - origin - shift) / module)) - 1
    while True:
        grid = origin + shift + k * module
        k += 1
        if grid <= start + EPS:
            continue
        if grid >= end - EPS:
            break
        edges.append(grid)
        if k > 10_000:
            break
    edges.append(end)
    return _pairs(edges)


def iter_ashlar(
    *,
    length: float,
    height: float,
    thickness: float,
    axis: str,
    base_y: float = 0.0,
    cross: float = 0.0,
    along_offset: float = 0.0,
    openings: Sequence[Opening] = (),
    detail: str = "standard",
) -> List[AshlarStone]:
    """Course a single-wythe wall. Openings are voids; lintels and sills are stones."""
    if axis not in {"x", "z"}:
        raise ValueError("ashlar axis must be 'x' or 'z'")
    if min(length, height, thickness) <= EPS:
        raise ValueError("ashlar wall dimensions must be positive")
    specs = list(openings or ())
    course_h = course_target(detail)
    stretcher = stretcher_target(detail)
    ny = _divisions(height, course_h, MAX_COURSE)
    nx = _divisions(length, stretcher, MAX_STRETCHER)
    while ny * nx > MAX_STONES:
        if course_h >= MAX_COURSE and stretcher >= MAX_STRETCHER:
            break
        course_h = min(MAX_COURSE, course_h * 1.15)
        stretcher = min(MAX_STRETCHER, stretcher * 1.15)
        ny = _divisions(height, course_h, MAX_COURSE)
        nx = _divisions(length, stretcher, MAX_STRETCHER)
    module = length / nx

    protected = [float(spec["sill"]) for spec in specs]
    protected += [float(spec["sill"]) + float(spec["height"]) for spec in specs]
    y_marks = [index * height / ny for index in range(ny + 1)]
    y_marks.extend(protected)
    bands = _pairs(_merge_marks(y_marks, protected, height))

    origin = along_offset - length / 2.0
    wall_end = along_offset + length / 2.0
    stones: List[AshlarStone] = []
    for course, (local_y0, local_y1) in enumerate(bands):
        y0, y1 = base_y + local_y0, base_y + local_y1
        intervals: List[Tuple[float, float]] = [(origin, wall_end)]
        for spec in specs:
            sill = float(spec["sill"])
            head = sill + float(spec["height"])
            left = along_offset + float(spec["center"]) - float(spec["width"]) / 2.0
            right = along_offset + float(spec["center"]) + float(spec["width"]) / 2.0
            if local_y0 >= sill - EPS and local_y1 <= head + EPS:
                intervals = _subtract(intervals, left, right)
        index = 0
        for start, end in intervals:
            for a, b in _clip_grid(start, end, origin, module, course % 2 == 1):
                unit, opening_id, opening_index = _classify_stone(
                    a, b, local_y0, local_y1, along_offset, specs,
                )
                along = (a + b) / 2.0
                if axis == "x":
                    position = (along, (y0 + y1) / 2.0, cross)
                    dims = (b - a, y1 - y0, thickness)
                else:
                    position = (cross, (y0 + y1) / 2.0, along)
                    dims = (thickness, y1 - y0, b - a)
                stones.append(_stone(
                    position, dims, course, index, unit, axis, opening_id, opening_index,
                ))
                index += 1
    if not stones:
        raise ValueError("ashlar wall produced no stones")
    return stones


def _classify_stone(
    start: float,
    end: float,
    y0: float,
    y1: float,
    along_offset: float,
    specs: Sequence[Opening],
) -> Tuple[str, Optional[str], Optional[int]]:
    mid = (start + end) / 2.0
    for index, spec in enumerate(specs):
        left = along_offset + float(spec["center"]) - float(spec["width"]) / 2.0
        right = along_offset + float(spec["center"]) + float(spec["width"]) / 2.0
        sill = float(spec["sill"])
        head = sill + float(spec["height"])
        opening_id = str(spec.get("id", f"opening.{index:03d}"))
        overlaps_x = not (end < left - EPS or start > right + EPS)
        if overlaps_x:
            if sill > EPS and abs(y1 - sill) <= 0.12:
                return "sill", opening_id, index
            if abs(y0 - head) <= 0.12:
                return "lintel", opening_id, index
        overlaps_y = y0 < head - EPS and y1 > sill + EPS
        on_jamb = (
            abs(start - left) <= 0.12 or abs(end - left) <= 0.12
            or abs(start - right) <= 0.12 or abs(end - right) <= 0.12
        )
        if overlaps_y and on_jamb:
            return "jamb", opening_id, index
    return "ashlar", None, None


def iter_ashlar_box(
    center: Sequence[float],
    dimensions: Sequence[float],
    axis: str,
    openings: Sequence[Opening] = (),
    detail: str = "standard",
) -> List[AshlarStone]:
    cx, cy, cz = (float(value) for value in center)
    dx, dy, dz = (float(value) for value in dimensions)
    if axis == "x":
        length, thickness, along_offset, cross = dx, dz, cx, cz
    elif axis == "z":
        length, thickness, along_offset, cross = dz, dx, cz, cx
    else:
        raise ValueError("ashlar axis must be 'x' or 'z'")
    return iter_ashlar(
        length=length, height=dy, thickness=thickness, axis=axis,
        base_y=cy - dy / 2.0, cross=cross, along_offset=along_offset,
        openings=openings, detail=detail,
    )


def radial_divisions(
    start: float,
    end: float,
    bottom: float,
    top: float,
    radius: float,
    detail: str = "standard",
    facets: Optional[int] = None,
) -> List[Tuple[float, float, float, float, int, int]]:
    """Yield (theta0, theta1, y0, y1, course, index) for stacked radial ashlars."""
    height = top - bottom
    sweep = end - start
    if height <= EPS or sweep <= EPS:
        raise ValueError("radial masonry needs positive height and sweep")
    ny = _divisions(height, course_target(detail), MAX_COURSE)
    if facets is None:
        arc = abs(sweep) * max(radius, MIN_STONE)
        facets = _divisions(arc, stretcher_target(detail), MAX_STRETCHER)
    facets = max(1, int(facets))
    y_edges = _uniform_edges(bottom, top, ny)
    rows: List[Tuple[float, float, float, float, int, int]] = []
    for course, (y0, y1) in enumerate(_pairs(y_edges)):
        for index, (a0, a1) in enumerate(_pairs(_offset_edges(start, end, facets, course % 2 == 1))):
            rows.append((a0, a1, y0, y1, course, index))
    return rows


def radial_ashlar_mesh(
    rx: float,
    rz: float,
    inner: float,
    outer: float,
    start: float,
    end: float,
    bottom: float,
    top: float,
    segments: int = 1,
):
    """One annular wedge; same construction as classical sector solids."""
    samples = max(1, int(segments))
    angles = np.linspace(start, end, samples + 1)
    profile = [(rx * outer * math.cos(t), rz * outer * math.sin(t)) for t in angles]
    profile += [(rx * inner * math.cos(t), rz * inner * math.sin(t)) for t in reversed(angles)]
    mesh = extrude_profile(profile, top - bottom)
    transform = np.array([[1., 0., 0., 0.], [0., 0., -1., (top + bottom) / 2],
                          [0., 1., 0., 0.], [0., 0., 0., 1.]])
    mesh.apply_transform(transform)
    return mesh


def _cross2(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _almost_same(a: Sequence[float], b: Sequence[float], tol: float = EPS) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _collinear(a: Sequence[float], b: Sequence[float], c: Sequence[float], tol: float) -> bool:
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    span = math.hypot(*ab) * math.hypot(*bc)
    if span <= tol:
        return True
    return abs(_cross2(ab, bc)) <= tol * max(span, 1.0)


def _ring_area(points: Sequence[Sequence[float]]) -> float:
    area = 0.0
    count = len(points)
    for index, point in enumerate(points):
        other = points[(index + 1) % count]
        area += _cross2(point, other)
    return 0.5 * area


def _clean_ring(points: Sequence[Sequence[float]], tol: float = EPS) -> List[Tuple[float, float]]:
    """Drop closing duplicates, coincident samples, and collinear vertices."""
    raw = [(float(point[0]), float(point[1])) for point in points]
    if not raw:
        return []
    ring: List[Tuple[float, float]] = [raw[0]]
    for point in raw[1:]:
        if not _almost_same(point, ring[-1], tol):
            ring.append(point)
    if len(ring) >= 2 and _almost_same(ring[0], ring[-1], tol):
        ring.pop()
    changed = True
    while changed and len(ring) > 3:
        changed = False
        for index in range(len(ring)):
            previous = ring[index - 1]
            current = ring[index]
            following = ring[(index + 1) % len(ring)]
            if _collinear(previous, current, following, tol) or _almost_same(previous, following, tol):
                ring.pop(index)
                changed = True
                break
    if len(ring) < 3:
        return []
    area = _ring_area(ring)
    if abs(area) <= max(tol, EPS):
        return []
    if area < 0:
        ring = list(reversed(ring))
    return ring


def _finalize_ring(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    cleaned = _clean_ring(points, EPS)
    if not cleaned:
        return []
    rounded = [(_r(x), _r(y)) for x, y in cleaned]
    return _clean_ring(rounded, 1e-6)


def _profile_ring(polygon_xy: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    try:
        points = [(float(row[0]), float(row[1])) for row in polygon_xy]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("profile polygon must be an N by 2 array") from exc
    if len(points) < 3:
        raise ValueError("profile polygon must be an N by 2 array")
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in points):
        raise ValueError("profile polygon must contain finite vertices")
    ring = _finalize_ring(points)
    if len(ring) < 3:
        raise ValueError("profile polygon must be a simple closed ring")
    return ring


def _intersect_h(p: Sequence[float], q: Sequence[float], y: float) -> Tuple[float, float]:
    dy = q[1] - p[1]
    if abs(dy) <= EPS:
        return (float(p[0]), float(y))
    t = (y - p[1]) / dy
    t = min(1.0, max(0.0, t))
    return (p[0] + t * (q[0] - p[0]), float(y))


def _intersect_v(p: Sequence[float], q: Sequence[float], x: float) -> Tuple[float, float]:
    dx = q[0] - p[0]
    if abs(dx) <= EPS:
        return (float(x), float(p[1]))
    t = (x - p[0]) / dx
    t = min(1.0, max(0.0, t))
    return (float(x), p[1] + t * (q[1] - p[1]))


def _clip_halfplane(polygon: Sequence[Sequence[float]], inside, intersect) -> List[Tuple[float, float]]:
    if len(polygon) < 3:
        return []
    output: List[Tuple[float, float]] = []
    previous = (float(polygon[-1][0]), float(polygon[-1][1]))
    prev_in = inside(previous)
    for raw in polygon:
        current = (float(raw[0]), float(raw[1]))
        curr_in = inside(current)
        if curr_in:
            if not prev_in:
                output.append(intersect(previous, current))
            output.append(current)
        elif prev_in:
            output.append(intersect(previous, current))
        previous = current
        prev_in = curr_in
    return output


def _clip_horizontal_band(
    polygon: Sequence[Sequence[float]],
    y0: float,
    y1: float,
) -> List[Tuple[float, float]]:
    """Sutherland–Hodgman clip of an elevation ring to y0 <= y <= y1."""
    lo, hi = (y0, y1) if y1 >= y0 else (y1, y0)
    lower = _clip_halfplane(
        polygon,
        lambda point: point[1] >= lo - EPS,
        lambda a, b: _intersect_h(a, b, lo),
    )
    clipped = _clip_halfplane(
        lower,
        lambda point: point[1] <= hi + EPS,
        lambda a, b: _intersect_h(a, b, hi),
    )
    return _finalize_ring(clipped)


def _clip_vertical_band(
    polygon: Sequence[Sequence[float]],
    x0: float,
    x1: float,
) -> List[Tuple[float, float]]:
    """Sutherland–Hodgman clip of an elevation ring to x0 <= x <= x1."""
    lo, hi = (x0, x1) if x1 >= x0 else (x1, x0)
    left = _clip_halfplane(
        polygon,
        lambda point: point[0] >= lo - EPS,
        lambda a, b: _intersect_v(a, b, lo),
    )
    clipped = _clip_halfplane(
        left,
        lambda point: point[0] <= hi + EPS,
        lambda a, b: _intersect_v(a, b, hi),
    )
    return _finalize_ring(clipped)


def _profile_bands(y_min: float, y_max: float, detail: str) -> List[Tuple[float, float]]:
    height = float(y_max) - float(y_min)
    if height <= EPS:
        raise ValueError("profile must have positive height")
    count = max(1, _divisions(height, course_target(detail)))
    count = max(count, int(math.ceil(height / MAX_COURSE - EPS)))
    if count > 1 and height / count < MIN_STONE:
        count = max(1, int(math.floor(height / MIN_STONE + EPS)))
        count = max(count, int(math.ceil(height / MAX_COURSE - EPS)))
    edges = _uniform_edges(float(y_min), float(y_max), count)
    bands: List[Tuple[float, float]] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end - start >= MIN_STONE or not bands:
            bands.append((start, end))
            continue
        merged = (bands[-1][0], end)
        if merged[1] - merged[0] <= MAX_COURSE + EPS:
            bands[-1] = merged
        else:
            bands.append((start, end))
    if len(bands) >= 2 and bands[0][1] - bands[0][0] < MIN_STONE:
        merged = (bands[0][0], bands[1][1])
        if merged[1] - merged[0] <= MAX_COURSE + EPS:
            bands[1] = merged
            bands.pop(0)
    return [(a, b) for a, b in bands if b - a > EPS]


def _usable_band(polygon: Sequence[Sequence[float]], y0: float, y1: float) -> bool:
    if len(polygon) < 3 or y1 - y0 <= EPS:
        return False
    ys = [point[1] for point in polygon]
    xs = [point[0] for point in polygon]
    if max(ys) - min(ys) <= EPS or max(xs) - min(xs) <= EPS:
        return False
    return abs(_ring_area(polygon)) > EPS


def iter_profile_courses(
    polygon_xy: Sequence[Sequence[float]],
    depth: float,
    *,
    bottom: float = 0.0,
    detail: str = "standard",
) -> List[Tuple[List[Tuple[float, float]], float, float, int]]:
    """Split an elevation polygon into cubit-height horizontal courses.

    ``polygon_xy`` is a simple (x, y) ring in the profile plane, Y up, with
    y = 0 at ``bottom`` or already in the same frame as ``bottom``. Each
    item is ``(band_polygon, y0, y1, course)``. Band rings are simple, CCW,
    and omit a closing duplicate so ``extrude_profile`` can consume them.
    Vertical coursing only: these lifts are carved plates, not running bond.
    """
    if isinstance(depth, bool) or not isinstance(depth, Real):
        raise ValueError("profile course depth must be a finite positive number")
    depth = float(depth)
    if not math.isfinite(depth) or depth <= EPS:
        raise ValueError("profile course depth must be a finite positive number")
    ring = _profile_ring(polygon_xy)
    offset = float(bottom)
    if not math.isfinite(offset):
        raise ValueError("profile course bottom must be finite")
    y_min = min(point[1] for point in ring)
    y_max = max(point[1] for point in ring)
    rows: List[Tuple[List[Tuple[float, float]], float, float, int]] = []
    pending: Optional[float] = None
    for y0, y1 in _profile_bands(y_min, y_max, detail):
        lo = y0 if pending is None else pending
        polygon = _clip_horizontal_band(ring, lo, y1)
        height = y1 - lo
        sliver = height < MIN_STONE
        if _usable_band(polygon, lo, y1) and not sliver:
            rows.append((polygon, lo, y1, len(rows)))
            pending = None
            continue
        if _usable_band(polygon, lo, y1) and sliver and not rows:
            pending = lo
            continue
        if sliver and rows:
            _prev_polygon, prev0, _prev1, prev_course = rows[-1]
            if y1 - prev0 <= MAX_COURSE + EPS:
                merged = _clip_horizontal_band(ring, prev0, y1)
                if _usable_band(merged, prev0, y1):
                    rows[-1] = (merged, prev0, y1, prev_course)
                    pending = None
                    continue
        if _usable_band(polygon, lo, y1):
            rows.append((polygon, lo, y1, len(rows)))
            pending = None
        else:
            pending = lo
    if pending is not None:
        hi = y_max
        if rows:
            _prev_polygon, prev0, _prev1, prev_course = rows[-1]
            if hi - prev0 <= MAX_COURSE + EPS:
                merged = _clip_horizontal_band(ring, prev0, hi)
                if _usable_band(merged, prev0, hi):
                    rows[-1] = (merged, prev0, hi, prev_course)
                    pending = None
        if pending is not None:
            polygon = _clip_horizontal_band(ring, pending, hi)
            if _usable_band(polygon, pending, hi):
                rows.append((polygon, pending, hi, len(rows)))
    if not rows:
        raise ValueError("profile produced no masonry courses")
    if abs(offset) > EPS:
        shifted: List[Tuple[List[Tuple[float, float]], float, float, int]] = []
        for polygon, y0, y1, course in rows:
            band = [(_r(x), _r(y + offset)) for x, y in polygon]
            band = _finalize_ring(band)
            if not _usable_band(band, y0 + offset, y1 + offset):
                continue
            shifted.append((band, _r(y0 + offset), _r(y1 + offset), course))
        if not shifted:
            raise ValueError("profile produced no masonry courses")
        rows = [(band, y0, y1, index) for index, (band, y0, y1, _) in enumerate(shifted)]
    else:
        rows = [(polygon, _r(y0), _r(y1), course) for polygon, y0, y1, course in rows]
    return rows


def course_profile_mesh(
    polygon: Sequence[Sequence[float]],
    depth: float,
    detail: str = "standard",
    *,
    bottom: float = 0.0,
) -> List[Tuple[Any, int]]:
    """Extrude cubit-scale profile tiles as Y-up solids.

    Elevation rings already live in XY with Y up. Each tile is a course
    band clipped in X so the extruded AABB stays under MAX_STRETCHER.
    """
    return [(mesh, course) for mesh, course, _tile in course_profile_tiles(
        polygon, depth, detail, bottom=bottom,
    )]


def course_profile_tiles(
    polygon: Sequence[Sequence[float]],
    depth: float,
    detail: str = "standard",
    *,
    bottom: float = 0.0,
) -> List[Tuple[Any, int, int]]:
    """Course a plate, then tile each lift so the extruded AABB stays human-scale.

    Yields ``(mesh, course, tile)``. The mesh lives in the profile frame:
    polygon XY, extrusion centered on Z, and depth tiles centered on the
    original plate. Callers rotate/translate into world space.
    """
    if isinstance(depth, bool) or not isinstance(depth, Real):
        raise ValueError("profile course depth must be a finite positive number")
    depth = float(depth)
    if not math.isfinite(depth) or depth <= EPS:
        raise ValueError("profile course depth must be a finite positive number")
    stretcher = stretcher_target(detail)
    n_depth = _divisions(depth, stretcher, MAX_STRETCHER)
    chunk = depth / n_depth
    tiles: List[Tuple[Any, int, int]] = []
    tile = 0
    for depth_index in range(n_depth):
        zc = -depth / 2.0 + (depth_index + 0.5) * chunk
        for band, y0, y1, course in iter_profile_courses(
            polygon, chunk, bottom=bottom, detail=detail,
        ):
            xs = [point[0] for point in band]
            x0, x1 = min(xs), max(xs)
            nx = _divisions(x1 - x0, stretcher, MAX_STRETCHER)
            x_edges = _uniform_edges(x0, x1, nx)
            for xa, xb in _pairs(x_edges):
                piece = band if nx == 1 else _clip_vertical_band(band, xa, xb)
                if not _usable_band(piece, y0, y1):
                    continue
                mesh = extrude_profile(piece, chunk)
                if abs(zc) > EPS:
                    mesh.apply_translation((0.0, 0.0, zc))
                tiles.append((mesh, course, tile))
                tile += 1
    if not tiles:
        raise ValueError("profile produced no masonry tiles")
    return tiles


def _lerp3(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, float, float]:
    return (float(a[0] + (b[0] - a[0]) * t),
            float(a[1] + (b[1] - a[1]) * t),
            float(a[2] + (b[2] - a[2]) * t))


def _dist3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _envelope6(pts: Sequence[Sequence[float]]) -> Tuple[float, ...]:
    xs, ys, zs = zip(*pts)
    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    return tuple(round((lo[i] + hi[i]) / 2.0, 6) for i in range(3)) + tuple(
        round(hi[i] - lo[i], 6) for i in range(3))


def iter_shell_wedges(
    apex: Sequence[float],
    p: Sequence[float],
    q: Sequence[float],
    apex_lo: Sequence[float],
    p_lo: Sequence[float],
    q_lo: Sequence[float],
    max_course: float = MAX_COURSE,
    max_stretch: float = MAX_STRETCHER,
) -> List[Tuple[Tuple[float, float, float], ...]]:
    """Split a thick triangular roof shell into cubit-scale cells."""
    stack = [(tuple(map(float, apex)), tuple(map(float, p)), tuple(map(float, q)),
              tuple(map(float, apex_lo)), tuple(map(float, p_lo)), tuple(map(float, q_lo)))]
    cells: List[Tuple[Tuple[float, float, float], ...]] = []
    while stack:
        a, b, c, al, bl, cl = stack.pop()
        pts = (a, b, c, al, bl, cl)
        xs, ys, zs = zip(*pts)
        if (max(ys) - min(ys) <= max_course + 1e-6
                and max(max(xs) - min(xs), max(zs) - min(zs)) <= max_stretch + 1e-6):
            cells.append(pts)
            continue
        if len(cells) + len(stack) > 4000:
            cells.append(pts)
            continue
        d_ab, d_ac, d_bc = _dist3(a, b), _dist3(a, c), _dist3(b, c)
        m_bc, ml_bc = _lerp3(b, c, 0.5), _lerp3(bl, cl, 0.5)
        m_ab, ml_ab = _lerp3(a, b, 0.5), _lerp3(al, bl, 0.5)
        m_ac, ml_ac = _lerp3(a, c, 0.5), _lerp3(al, cl, 0.5)
        splits = [
            (d_bc, (a, b, m_bc, al, bl, ml_bc), (a, m_bc, c, al, ml_bc, cl)),
            (d_ab, (a, m_ab, c, al, ml_ab, cl), (m_ab, b, c, ml_ab, bl, cl)),
            (d_ac, (a, b, m_ac, al, bl, ml_ac), (m_ac, b, c, ml_ac, bl, cl)),
        ]
        splits.sort(key=lambda item: -item[0])
        chosen = None
        for _length, child_a, child_b in splits:
            if _envelope6(child_a) != _envelope6(child_b):
                chosen = (child_a, child_b)
                break
        if chosen is None:
            g = ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0)
            gl = ((al[0] + bl[0] + cl[0]) / 3.0, (al[1] + bl[1] + cl[1]) / 3.0, (al[2] + bl[2] + cl[2]) / 3.0)
            stack.extend(((a, b, g, al, bl, gl), (b, c, g, bl, cl, gl), (c, a, g, cl, al, gl)))
        else:
            stack.extend(chosen)
    return cells


def shell_wedge_mesh(apex, p, q, apex_lo, p_lo, q_lo):
    """Closed thick triangle; winding is corrected to a positive volume."""
    import trimesh
    mesh = trimesh.Trimesh(
        vertices=[apex, p, q, apex_lo, p_lo, q_lo],
        faces=[(0, 1, 2), (3, 5, 4), (0, 4, 1), (0, 3, 4),
               (1, 5, 2), (1, 4, 5), (2, 0, 3), (2, 5, 3)],
        process=False)
    if mesh.volume < 0:
        mesh.invert()
    mesh.fix_normals(multibody=False)
    return mesh


def shell_cell_meshes(cells: Sequence[Sequence[Sequence[float]]]):
    """One mesh per unique AABB so diagonal pairs become a single ashlar."""
    import trimesh
    groups: Dict[Tuple[float, ...], list] = {}
    for cell in cells:
        groups.setdefault(_envelope6(cell), []).append(cell)
    meshes = []
    for group in groups.values():
        pieces = []
        for cell in group:
            mesh = shell_wedge_mesh(*cell)
            if mesh.volume > 0:
                pieces.append(mesh)
        if not pieces:
            continue
        meshes.append(pieces[0] if len(pieces) == 1 else trimesh.util.concatenate(pieces))
    return meshes


def stone_is_human_scale(stone: AshlarStone) -> bool:
    width, height, depth = stone.dimensions
    along = max(width, depth)
    if stone.unit == "lintel":
        return height <= MAX_COURSE * 1.2 + 1e-6 and along <= MAX_LINTEL + 1e-6
    if stone.unit == "drum":
        return height <= MAX_COURSE + 1e-6
    if stone.unit == "profile_course":
        return height <= MAX_COURSE + 1e-6 and along <= MAX_STRETCHER + 1e-6
    return height <= MAX_COURSE + 1e-6 and along <= MAX_STRETCHER + 1e-6


def fill_box(
    center: Sequence[float],
    dimensions: Sequence[float],
    detail: str = "standard",
    openings: Sequence[Opening] = (),
    axis: Optional[str] = None,
) -> List[AshlarStone]:
    """Course a box as ashlars; openings force a single-wythe wall runner."""
    if openings:
        return iter_ashlar_box(center, dimensions, axis or wall_axis(dimensions), openings, detail)
    return iter_masonry_solid(center, dimensions, detail)


def placement_is_masonry_cube(placement: Any) -> bool:
    """CUBEs/cylinders that must be human-scale stones: walls, piers, stacked drums."""
    kind = str(getattr(placement, "kind", ""))
    role = str(getattr(placement, "role", ""))
    block_type = str(getattr(placement, "block_type", ""))
    parameters = getattr(placement, "parameters", {}) or {}
    unit = str(parameters.get("unit", ""))
    construction = str(parameters.get("construction", ""))
    if unit in {"ashlar", "sill", "lintel", "jamb", "drum", "paving", "radial_ashlar", "profile_course"}:
        return True
    if block_type == "MESH" and "ashlar" in construction.lower():
        return True
    if block_type not in {"CUBE", "CYLINDER"}:
        return False
    if construction in {"ashlar running bond", "ashlar stack"}:
        return True
    # Coarse roles also describe timber infill and continuous cast walls.
    # Explicit masonry units above always take precedence over material.
    substance = SUBSTANCES.get(getattr(placement, "substance", None))
    if substance is not None and (substance.category == "timber" or substance.id == "cast-concrete"):
        return False
    if kind in {"wall", "wall_segment", "pier", "bridge_pier", "hall_pier"}:
        return True
    if "wall" in role and block_type == "CUBE":
        return True
    return False


def placement_is_oversized(placement: Any) -> bool:
    parameters = getattr(placement, "parameters", {}) or {}
    unit = str(parameters.get("unit", "") or "")
    role = str(getattr(placement, "role", ""))
    dims = tuple(float(value) for value in placement.dimensions)
    if len(dims) != 3:
        return True
    width, height, depth = dims
    along = max(width, depth)
    if unit == "lintel" or role.endswith(".header"):
        return height > MAX_COURSE * 1.25 + 1e-6 or along > MAX_LINTEL + 1e-6
    if unit == "profile_course":
        return height > MAX_COURSE + 1e-6 or along > MAX_STRETCHER + 1e-6
    if unit == "drum" or str(getattr(placement, "block_type", "")) == "CYLINDER":
        return height > MAX_COURSE * 1.05 + 1e-6
    return height > MAX_COURSE + 1e-6 or along > MAX_STRETCHER + 1e-6


def ashlar_checks(document) -> list:
    """Fail walls and piers that are still megalith slabs instead of stones."""
    from .quality import _check

    placements = list(document.placements)
    masonry = [placement for placement in placements if placement_is_masonry_cube(placement)]
    if not masonry:
        return []
    oversized = [placement.id for placement in masonry if placement_is_oversized(placement)]
    checks = [_check(
        "craft.human_scale_blocks", not oversized,
        "Masonry members are human-scale stones.",
        "Masonry members include a slab larger than a human-scale ashlar.",
        primitive_ids=oversized, observed=len(oversized), expected=0,
    )]
    hosts: Dict[str, List[AshlarStone]] = {}
    for placement in masonry:
        parameters = placement.parameters or {}
        host = parameters.get("host_id")
        unit = str(parameters.get("unit", ""))
        if not host or unit not in {"ashlar", "sill", "lintel", "jamb"}:
            continue
        axis = str(parameters.get("axis") or wall_axis(placement.dimensions))
        hosts.setdefault(str(host), []).append(AshlarStone(
            tuple(float(value) for value in placement.position),  # type: ignore[arg-type]
            tuple(float(value) for value in placement.dimensions),  # type: ignore[arg-type]
            int(parameters.get("course", 0)),
            int(parameters.get("stone_index", 0)),
            unit, axis,
        ))
    aligned = []
    for host, stones in hosts.items():
        if len({stone.course for stone in stones}) < 2:
            continue
        if not running_bond_holds(stones):
            aligned.append(host)
    checks.append(_check(
        "craft.running_bond", not aligned,
        "Ashlar courses stagger their vertical joints.",
        "Ashlar courses stack joints in a continuous slab-like grid.",
        primitive_ids=aligned, observed=len(aligned), expected=0,
    ))
    return checks


def running_bond_holds(stones: Sequence[AshlarStone], tolerance: float = 0.04) -> bool:
    """True when interior vertical joints stagger between adjacent courses.

    Jambs at openings are allowed to stack; they are a vertical cut, not a
    failed bond in the field of the wall.
    """
    if not stones:
        return True
    axis = stones[0].axis
    courses: Dict[int, List[AshlarStone]] = {}
    jambs: List[float] = []
    for stone in stones:
        courses.setdefault(stone.course, []).append(stone)
        if stone.unit in {"sill", "lintel", "jamb"}:
            center = stone.position[0] if axis == "x" else stone.position[2]
            length = stone.dimensions[0] if axis == "x" else stone.dimensions[2]
            jambs.extend((center - length / 2.0, center + length / 2.0))
    if len(courses) < 2:
        return True

    def joints(course_stones: Sequence[AshlarStone]) -> List[float]:
        along = []
        for stone in course_stones:
            center = stone.position[0] if axis == "x" else stone.position[2]
            length = stone.dimensions[0] if axis == "x" else stone.dimensions[2]
            along.append(center - length / 2.0)
            along.append(center + length / 2.0)
        lo, hi = min(along), max(along)
        return sorted(value for value in along if value - lo > tolerance and hi - value > tolerance)

    def is_jamb(joint: float) -> bool:
        return any(abs(joint - jamb) <= tolerance for jamb in jambs)

    ordered = [courses[key] for key in sorted(courses)]
    for lower, upper in zip(ordered, ordered[1:]):
        low_joints = joints(lower)
        high_joints = joints(upper)
        if not low_joints or not high_joints:
            continue
        for joint in low_joints:
            if is_jamb(joint):
                continue
            if any(abs(joint - other) <= tolerance for other in high_joints):
                return False
    return True
