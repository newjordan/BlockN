"""A proportioned Gothic architect: plan, elevation, openings, and carved members.

The mesh is authored from profiles and spatial ribs, with genuinely open portals
and windows. Dimensions describe the whole composition, including its spires.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import random

import numpy as np
import trimesh

from core.architectural_geometry import extrude_profile, mesh_parameters, tube_path
from .masonry import (AshlarStone, MAX_COURSE, MAX_STRETCHER, MIN_STONE, fill_box,
                      iter_shell_wedges, shell_cell_meshes, wall_axis)
from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement

ARCHITECT = ArchitectInfo("cathedral", "1.2.0", "Cathedral Architect")
PRODUCT = "gothic-cathedral"
VARIANTS = ("high-gothic", "early-english", "rayonnant")
DEFAULTS = (32, 50, 64)
REQUIRED_ROLES = {
    "foundation", "nave_pier", "nave_arcade", "clerestory_frame",
    "aisle_window", "flying_buttress", "buttress_pier", "west_portal",
    "west_tower", "spire", "rose_tracery", "transept", "apse_window",
    "vault_rib", "nave_roof",
}
KINDS = {
    "foundation": "foundation", "floor": "slab", "approach_step": "circulation",
    "nave_pier": "column", "pier_shaft": "column", "pier_capital": "column",
    "nave_arcade": "arch", "clerestory_frame": "wall", "aisle_window": "wall",
    "window_tracery": "wall", "sill": "wall", "wall_spandrel": "wall",
    "flying_buttress": "arch", "buttress_pier": "pier", "buttress_cap": "roof",
    "pinnacle": "roof", "finial": "roof", "west_portal": "arch",
    "portal_gable": "roof", "west_tower": "wall", "tower_lancet": "wall",
    "tower_shaft": "pier", "tower_cornice": "entablature", "spire": "roof",
    "spire_rib": "roof", "rose_tracery": "wall", "rose_surround": "wall",
    "facade_gallery": "arch", "facade_gable": "roof", "transept": "wall",
    "apse_window": "wall", "apse_pier": "pier", "apse_roof": "roof",
    "vault_rib": "roof", "vault_boss": "roof", "nave_roof": "roof",
    "roof_seam": "roof", "aisle_roof": "roof", "transept_roof": "roof",
    "ridge": "roof", "crossing_lantern": "wall",
    "glazing": "wall", "aisle_dado": "wall", "facade_mullion": "wall",
}

# The physical substance behind every cathedral role.  High Gothic work is
# limestone ashlar throughout; the steep roofs are laid as slate courses and
# the apse hips and eave rolls are lead, exactly as the geometry courses them.
CATHEDRAL_SUBSTANCES = {
    "foundation": "limestone", "floor": "limestone", "approach_step": "limestone",
    "nave_pier": "limestone", "pier_shaft": "limestone", "pier_capital": "limestone",
    "nave_arcade": "limestone", "clerestory_frame": "limestone", "aisle_window": "limestone",
    "window_tracery": "limestone", "sill": "limestone", "wall_spandrel": "limestone",
    "flying_buttress": "limestone", "buttress_pier": "limestone", "buttress_cap": "limestone",
    "pinnacle": "limestone", "finial": "limestone", "west_portal": "limestone",
    "portal_gable": "limestone", "west_tower": "limestone", "tower_lancet": "limestone",
    "tower_shaft": "limestone", "tower_cornice": "limestone", "spire": "limestone",
    "spire_rib": "limestone", "rose_tracery": "limestone", "rose_surround": "limestone",
    "facade_gallery": "limestone", "facade_gable": "limestone", "transept": "limestone",
    "apse_window": "limestone", "apse_pier": "limestone",
    "vault_rib": "limestone", "vault_boss": "limestone",
    "nave_roof": "slate", "transept_roof": "slate", "aisle_roof": "slate",
    "apse_roof": "slate", "roof_seam": "lead-sheet", "ridge": "limestone",
    "crossing_lantern": "limestone", "aisle_dado": "limestone", "facade_mullion": "limestone",
    "glazing": "window-glass",
}
CATHEDRAL_DEFAULT_SUBSTANCE = "limestone"

# Tall thin piers and wall ribs use fewer drums than the cubit runner.
# Stay at the human-scale course cap so an 8 m pier is ~10 stones, not 20.
TALL_COURSE = MAX_COURSE
_TALL_MIN_HEIGHT = 3.0
_EPS = 1e-9


def _r(value):
    result = round(float(value), 6)
    return 0.0 if result == -0.0 else result


def _tall_course_count(span, limit=TALL_COURSE):
    span = float(span)
    if span <= _EPS:
        raise ValueError("stack height must be positive")
    count = max(1, math.ceil(span / limit - 1e-9))
    while span / count > limit + 1e-9:
        count += 1
    return count


def _iter_tall_stack(bottom, height, limit=TALL_COURSE):
    """Split a vertical member into ~0.75–0.80 m drums. Returns (center_y, height, course)."""
    count = _tall_course_count(height, limit)
    size = float(height) / count
    bottom = float(bottom)
    return [(bottom + (index + 0.5) * size, size, index) for index in range(count)]


def _is_tall_thin(dimensions):
    """True for a shaft or wall rib whose height dominates its plan, not a long low band."""
    width, height, depth = (float(value) for value in dimensions)
    return height >= _TALL_MIN_HEIGHT and height >= max(width, depth) * 1.25


def _bond_edges(start, end, count, offset):
    start, end = float(start), float(end)
    if count <= 1:
        return [start, end]
    module = (end - start) / count
    if not offset or module / 2.0 < 0.08:
        return [start + module * index for index in range(count)] + [end]
    half = module / 2.0
    edges = [start, start + half]
    x = start + half
    remaining = count - 1
    while remaining > 1 and x + module < end - 0.08:
        x += module
        edges.append(x)
        remaining -= 1
    if edges[-1] < end - _EPS:
        edges.append(end)
    else:
        edges[-1] = end
    return edges


def _iter_tall_box(center, dimensions):
    """Course a tall thin box as stacked ashlars; stagger only if a face needs two stretchers."""
    cx, cy, cz = (float(value) for value in center)
    dx, dy, dz = (float(value) for value in dimensions)
    axis = wall_axis(dimensions)
    ny = _tall_course_count(dy)
    course_h = dy / ny
    along = dx if axis == "x" else dz
    n_along = _tall_course_count(along, MAX_STRETCHER) if along > MIN_STONE * 1.5 else 1
    origin = (cx if axis == "x" else cz) - along / 2.0
    y0 = cy - dy / 2.0
    stones = []
    for course in range(ny):
        edges = _bond_edges(origin, origin + along, n_along, course % 2 == 1)
        y = y0 + (course + 0.5) * course_h
        index = 0
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a <= _EPS:
                continue
            if axis == "x":
                position = (_r((a + b) / 2.0), _r(y), _r(cz))
                dims = (_r(b - a), _r(course_h), _r(dz))
            else:
                position = (_r(cx), _r(y), _r((a + b) / 2.0))
                dims = (_r(dx), _r(course_h), _r(b - a))
            stones.append(AshlarStone(position, dims, course, index, "ashlar", axis))
            index += 1
    if not stones:
        raise ValueError("tall masonry produced no stones")
    return stones


def _clean_ring(points, eps=1e-9):
    """Drop consecutive (and closing) duplicates so extrude_profile stays simple."""
    if not points:
        return []
    out = [points[0]]
    for point in points[1:]:
        if abs(point[0] - out[-1][0]) > eps or abs(point[1] - out[-1][1]) > eps:
            out.append(point)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def _intersect_x(a, b, x):
    dx = b[0] - a[0]
    if abs(dx) <= _EPS:
        return (x, a[1])
    t = (x - a[0]) / dx
    return (x, a[1] + t * (b[1] - a[1]))


def _intersect_y(a, b, y):
    dy = b[1] - a[1]
    if abs(dy) <= _EPS:
        return (a[0], y)
    t = (y - a[1]) / dy
    return (a[0] + t * (b[0] - a[0]), y)


def _clip_side(points, inside, intersect):
    if len(points) < 3:
        return []
    out = []
    prev = points[-1]
    prev_in = inside(prev)
    for curr in points:
        curr_in = inside(curr)
        if curr_in:
            if not prev_in:
                out.append(intersect(prev, curr))
            out.append(curr)
        elif prev_in:
            out.append(intersect(prev, curr))
        prev, prev_in = curr, curr_in
    return _clean_ring(out)


def _clip_poly(polygon, x0=None, x1=None, y0=None, y1=None):
    """Sutherland–Hodgman clip of an elevation polygon to an axis-aligned band."""
    pts = _clean_ring([(float(p[0]), float(p[1])) for p in np.asarray(polygon, dtype=float)])
    if y0 is not None:
        cut = float(y0)
        pts = _clip_side(pts, lambda p: p[1] >= cut - _EPS, lambda a, b: _intersect_y(a, b, cut))
    if y1 is not None:
        cut = float(y1)
        pts = _clip_side(pts, lambda p: p[1] <= cut + _EPS, lambda a, b: _intersect_y(a, b, cut))
    if x0 is not None:
        cut = float(x0)
        pts = _clip_side(pts, lambda p: p[0] >= cut - _EPS, lambda a, b: _intersect_x(a, b, cut))
    if x1 is not None:
        cut = float(x1)
        pts = _clip_side(pts, lambda p: p[0] <= cut + _EPS, lambda a, b: _intersect_x(a, b, cut))
    return pts


def _clip_convex(subject, clipper):
    """Sutherland–Hodgman clip of ``subject`` to a convex ``clipper`` of either winding."""
    clip = [(float(p[0]), float(p[1])) for p in clipper]
    area = sum(clip[i][0] * clip[(i + 1) % len(clip)][1] - clip[(i + 1) % len(clip)][0] * clip[i][1]
               for i in range(len(clip)))
    sign = 1.0 if area >= 0 else -1.0
    pts = _clean_ring([(float(p[0]), float(p[1])) for p in subject])
    for a, b in zip(clip, clip[1:] + clip[:1]):
        ex, ey = b[0] - a[0], b[1] - a[1]
        if abs(ex) + abs(ey) <= _EPS:
            continue

        def side(p, a=a, ex=ex, ey=ey):
            return sign * (ex * (p[1] - a[1]) - ey * (p[0] - a[0]))

        def cut(p, q, side=side):
            sp, sq = side(p), side(q)
            t = sp / (sp - sq)
            return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)

        pts = _clip_side(pts, lambda p, side=side: side(p) >= -_EPS, cut)
        if len(pts) < 3:
            return []
    return pts


def _point_in_poly(x, z, poly):
    inside = False
    previous = poly[-1]
    for current in poly:
        x0, z0 = previous
        x1, z1 = current
        if (z0 > z) != (z1 > z) and x < (x1 - x0) * (z - z0) / (z1 - z0 + 1e-30) + x0:
            inside = not inside
        previous = current
    return inside


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _reverse_overlap(a, b, c, d, eps=1e-8):
    """True when AB and CD are collinear, opposite, and overlap in their interiors."""
    ab = (b[0] - a[0], b[1] - a[1])
    cd = (d[0] - c[0], d[1] - c[1])
    lab, lcd = math.hypot(*ab), math.hypot(*cd)
    if lab < eps or lcd < eps:
        return False
    if abs(_cross2(ab, cd)) > eps * lab * lcd:
        return False
    ac = (c[0] - a[0], c[1] - a[1])
    if abs(_cross2(ab, ac)) > eps * lab:
        return False
    if ab[0] * cd[0] + ab[1] * cd[1] >= 0:
        return False
    axis = 0 if abs(ab[0]) >= abs(ab[1]) else 1
    a0, b0, c0, d0 = a[axis], b[axis], c[axis], d[axis]
    if a0 > b0:
        a0, b0 = b0, a0
    if c0 > d0:
        c0, d0 = d0, c0
    return min(b0, d0) - max(a0, c0) > eps


def _simple_rings(ring):
    """Split a clip result that joined disjoint pieces with a doubled clip edge."""
    ring = _clean_ring([(float(p[0]), float(p[1])) for p in ring])
    if len(ring) < 3:
        return []
    n = len(ring)
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        for j in range(i + 2, n if i else n - 1):
            if (j + 1) % n == i:
                continue
            c, d = ring[j], ring[(j + 1) % n]
            if not _reverse_overlap(a, b, c, d):
                continue
            left = [ring[k % n] for k in range(i + 1, j + 1)]
            right = [ring[k % n] for k in range(j + 1, i + n + 1)]
            pieces = []
            for part in (left, right):
                pieces.extend(_simple_rings(part))
            return pieces
    if _poly_area(ring) < MIN_STONE * MIN_STONE * 0.25:
        return []
    return [ring]


def _poly_area(points):
    if len(points) < 3:
        return 0.0
    x, y = zip(*points)
    return abs(sum(x[i] * y[(i + 1) % len(points)] - x[(i + 1) % len(points)] * y[i]
                   for i in range(len(points)))) * 0.5


def _strict_courses(span, limit):
    """Divide span into equal parts that never exceed limit (no +eps slack)."""
    span = float(span)
    if span <= _EPS:
        return 1
    count = max(1, math.ceil(span / limit))
    while span / count > limit:
        count += 1
    return count


def _polyline_samples(points, spacing, closed=False):
    """Resample a 3D polyline so consecutive samples are at most ``spacing`` apart."""
    pts = [np.asarray(p, dtype=float).reshape(3) for p in points]
    if closed and len(pts) >= 2 and float(np.linalg.norm(pts[0] - pts[-1])) <= 1e-9:
        pts = pts[:-1]
    if closed and pts:
        pts = pts + [np.array(pts[0], dtype=float, copy=True)]
    if len(pts) < 2:
        return pts
    spacing = max(float(spacing), MIN_STONE)
    samples = [np.array(pts[0], dtype=float, copy=True)]
    remaining = spacing
    for start, end in zip(pts[:-1], pts[1:]):
        vec = end - start
        dist = float(np.linalg.norm(vec))
        if dist <= 1e-12:
            continue
        direction = vec / dist
        consumed = 0.0
        while dist - consumed >= remaining - 1e-12:
            consumed += remaining
            samples.append(start + direction * consumed)
            remaining = spacing
        remaining -= dist - consumed
    end = pts[-1]
    if float(np.linalg.norm(samples[-1] - end)) > 1e-9:
        samples.append(np.array(end, dtype=float, copy=True))
    return samples


_SLATED = {"nave_roof", "transept_roof", "aisle_roof"}


def _cubit_runs(samples, radius):
    """Group consecutive rib samples into curved runs that each fit one stone.

    The tube adds at most ``radius`` on every side of its centerline, so the
    padded point bounds are a conservative envelope for the swept member.
    """
    pad = 2.0 * float(radius)

    def fits(points):
        extent = np.ptp(np.asarray(points), axis=0) + pad
        return (float(extent[1]) <= MAX_COURSE + 1e-6
                and max(float(extent[0]), float(extent[2])) <= MAX_STRETCHER + 1e-6)

    runs, run = [], [samples[0]]
    for point in samples[1:]:
        if float(np.linalg.norm(point - run[-1])) <= 1e-9:
            continue
        if len(run) == 1 or fits([*run, point]):
            run.append(point)
        else:
            runs.append(run)
            run = [run[-1], point]
    if len(run) >= 2:
        runs.append(run)
    return runs


def _octagon_ring(x, z, y0, y1, outer0, outer1, inner0, inner1, angles):
    """A closed, hollow frustum course with outward-facing normals."""
    k = len(angles)
    rings = ((outer0, y0), (outer1, y1), (inner1, y1), (inner0, y0))
    verts = [[x + r * math.cos(a), y, z + r * math.sin(a)] for r, y in rings for a in angles]
    faces = []
    for ring in range(4):
        nxt = (ring + 1) % 4
        for i in range(k):
            j = (i + 1) % k
            a, b, c, d = ring * k + i, ring * k + j, nxt * k + j, nxt * k + i
            faces.extend(((a, b, c), (a, c, d)))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _xz_along(width, depth, angle):
    cosine, sine = abs(math.cos(angle)), abs(math.sin(angle))
    return max(width * cosine + depth * sine, width * sine + depth * cosine)


def _rect_world_box(polygon, depth, x, y, z, angle):
    """World center/dims for an axis-aligned elevation rectangle; otherwise None.

    Cubes cannot follow an oblique apse facet, so only k·π/2 walls become ashlars.
    """
    poly = np.asarray(polygon, dtype=float)
    if len(poly) < 4:
        return None
    rolled = np.roll(poly, -1, axis=0)
    edges = rolled - poly
    if not np.all((np.abs(edges[:, 0]) <= 1e-8) | (np.abs(edges[:, 1]) <= 1e-8)):
        return None
    if np.any((np.abs(edges[:, 0]) <= 1e-8) & (np.abs(edges[:, 1]) <= 1e-8)):
        return None
    ux = np.unique(np.round(poly[:, 0], 8))
    uy = np.unique(np.round(poly[:, 1], 8))
    if len(ux) != 2 or len(uy) != 2:
        return None
    quarter = angle / (math.pi / 2.0)
    if abs(quarter - round(quarter)) > 1e-8:
        return None
    k = int(round(quarter)) % 4
    px0, px1 = float(ux.min()), float(ux.max())
    py0, py1 = float(uy.min()), float(uy.max())
    dx, dy = px1 - px0, py1 - py0
    if min(dx, dy, float(depth)) < MIN_STONE * 0.5:
        return None
    px_c, py_c = (px0 + px1) / 2.0, (py0 + py1) / 2.0
    cosine, sine = math.cos(angle), math.sin(angle)
    center = (x + px_c * cosine, y + py_c, z - px_c * sine)
    dims = (dx, dy, float(depth)) if k % 2 == 0 else (float(depth), dy, dx)
    return center, dims


def iter_profile_pieces(polygon, depth, angle=0, max_course=MAX_COURSE, max_stretcher=MAX_STRETCHER):
    """Yield (ring, course) bands that stay inside a cubit ashlar AABB."""
    polygon = np.asarray(polygon, dtype=float)
    max_course = float(max_course)
    max_stretcher = float(max_stretcher)
    y0, y1 = float(polygon[:, 1].min()), float(polygon[:, 1].max())
    ny = _strict_courses(y1 - y0, max_course) if y1 - y0 > _EPS else 1
    y_edges = _bond_edges(y0, y1, ny, False)
    course = 0
    depth = float(depth)
    for ya, yb in zip(y_edges[:-1], y_edges[1:]):
        yb = min(yb, ya + max_course - 1e-9)
        if yb - ya < MIN_STONE * 0.5:
            continue
        for band in _simple_rings(_clip_poly(polygon, y0=ya, y1=yb)):
            if len(band) < 3 or _poly_area(band) < MIN_STONE * MIN_STONE * 0.25:
                continue
            xs = [p[0] for p in band]
            x0, x1 = min(xs), max(xs)
            if x1 - x0 < MIN_STONE * 0.5:
                continue
            nx = 1
            while _xz_along((x1 - x0) / nx, depth, angle) > max_stretcher and (x1 - x0) / nx > MIN_STONE:
                nx += 1
            x_edges = _bond_edges(x0, x1, nx, False)
            for xa, xb in zip(x_edges[:-1], x_edges[1:]):
                piece = band if nx == 1 else _clip_poly(band, x0=xa, x1=xb)
                rings = [piece] if nx == 1 else _simple_rings(piece)
                for ring in rings:
                    if len(ring) < 3 or _poly_area(ring) < MIN_STONE * MIN_STONE * 0.25:
                        continue
                    yield ring, course
                    course += 1


def _profile_mesh(polygon, depth, angle=0, x=0, y=0, z=0):
    mesh = extrude_profile(polygon, depth)
    if angle:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
    mesh.apply_translation([x, y, z])
    return mesh


def _cubit_aabb(mesh):
    extent = mesh.bounds[1] - mesh.bounds[0]
    return (float(extent[1]) <= MAX_COURSE + 1e-6
            and max(float(extent[0]), float(extent[2])) <= MAX_STRETCHER + 1e-6)


def iter_fitted_profile_pieces(polygon, depth, angle=0, x=0, y=0, z=0):
    """Yield elevation rings whose extruded AABB stays inside a cubit ashlar.

    ``iter_profile_pieces`` can drop thin pointed-arch slivers; the leftover
    whole plate used to be emitted as one MESH. Binary-split only rings that
    actually extrude oversized — do not subdivide failed slivers into dust.
    """
    rings = []
    try:
        rings = [np.asarray(ring, dtype=float) for ring, _ in iter_profile_pieces(polygon, depth, angle)]
    except ValueError:
        rings = []
    if not rings:
        rings = [np.asarray(polygon, dtype=float)]

    def try_mesh(ring):
        try:
            return _profile_mesh(ring, depth, angle, x, y, z)
        except ValueError:
            return None

    pending = [(np.asarray(ring, dtype=float), 0) for ring in rings if len(ring) >= 3]
    course = 0
    yielded = False
    while pending:
        ring, level = pending.pop()
        if len(ring) < 3:
            continue
        mesh = try_mesh(ring)
        if mesh is not None and _cubit_aabb(mesh):
            yield ring, course
            course += 1
            yielded = True
            continue
        if level >= 6:
            continue
        xs = [float(p[0]) for p in ring]
        ys = [float(p[1]) for p in ring]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if span_y < MIN_STONE * 2 and span_x < MIN_STONE * 2:
            continue
        if mesh is None and span_y <= MAX_COURSE + 1e-6 and span_x <= MAX_STRETCHER + 1e-6:
            continue
        if span_y >= span_x and span_y > MIN_STONE * 2:
            mid = (min(ys) + max(ys)) / 2.0
            parts = (_clip_poly(ring, y1=mid), _clip_poly(ring, y0=mid))
        else:
            mid = (min(xs) + max(xs)) / 2.0
            parts = (_clip_poly(ring, x1=mid), _clip_poly(ring, x0=mid))
        for part in parts:
            if len(part) < 3:
                continue
            for simple in _simple_rings(part):
                pending.append((np.asarray(simple, dtype=float), level + 1))
    if not yielded:
        mesh = try_mesh(polygon)
        if mesh is not None and _cubit_aabb(mesh):
            yield np.asarray(polygon, dtype=float), 0


def _annulus_strip_polygon(outer, inner, start, n_edges):
    """Simple elevation ring for one radial masonry strip (void stays outside)."""
    count = len(outer)
    idxs = [int((start + k) % count) for k in range(n_edges + 1)]
    ring_o = [(float(outer[i][0]), float(outer[i][1])) for i in idxs]
    ring_i = [(float(inner[i][0]), float(inner[i][1])) for i in reversed(idxs)]
    return _clean_ring(ring_o + ring_i)


def _annulus_strip_mesh(outer, inner, depth, start, n_edges):
    """Closed radial strip of an outer/inner elevation ring, extruded through depth."""
    count = len(outer)
    n = n_edges + 1
    idxs = [int((start + k) % count) for k in range(n)]
    ring_o = [np.asarray(outer[i], dtype=float) for i in idxs]
    ring_i = [np.asarray(inner[i], dtype=float) for i in idxs]
    vertices = [[float(u), float(v), zz]
                for zz in (depth / 2.0, -depth / 2.0)
                for boundary in (ring_o, ring_i)
                for u, v in boundary]
    faces = []

    def quad(a, b, c, d):
        faces.extend(((a, b, c), (a, c, d)))

    for i in range(n_edges):
        j = i + 1
        fo, fi, bo, bi = i, n + i, 2 * n + i, 3 * n + i
        foj, fij, boj, bij = j, n + j, 2 * n + j, 3 * n + j
        quad(fo, foj, fij, fi)
        quad(bo, bi, bij, boj)
        quad(fo, bo, boj, foj)
        quad(fi, fij, bij, bi)
    quad(0, n, 3 * n, 2 * n)
    end = n_edges
    quad(end, 2 * n + end, 3 * n + end, n + end)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals(multibody=False)
    return mesh


def list_products():
    return [{"name": PRODUCT, "category": "assembly", "architect": ARCHITECT.to_dict(),
        "architect_id": ARCHITECT.id, "architect_title": ARCHITECT.title,
        "architect_version": ARCHITECT.version,
        "description": "Cruciform Gothic cathedral with open tracery, rib vaults, paired west spires and flying buttresses",
        "defaults": dict(zip(("width", "height", "depth"), DEFAULTS), variant="high-gothic", detail="standard", bays=7),
        "variants": ["auto", "random", *VARIANTS], "detail_levels": ["concept", "standard", "presentation"],
        "opening_modes": ["auto"],
        "parameters": ["choir_bays", "apse_sides", "spire_ratio"],
        "substances": sorted(set(CATHEDRAL_SUBSTANCES.values()) | {CATHEDRAL_DEFAULT_SUBSTANCE}),
        "rules": {"required_roles": sorted(REQUIRED_ROLES), "bays": [4, 10],
            "choir_bays": [2, 4], "apse_sides": [5, 7, 9], "spire_ratio": [0.22, 0.36],
            "depth_to_width": [1.6, 2.6], "height_to_width": [1.2, 1.9],
            "scope": "architectural visualization and modular asset generation"}}]


def pointed_arc(half, rise, segments=16):
    """Two circular arcs meet at a true Gothic point, left spring to right."""
    circle_rise = max(rise, half * 1.05)
    c = (circle_rise * circle_rise - half * half) / (2 * half)
    radius = half + c
    theta = np.linspace(0, math.acos(c / radius), segments + 1)
    left = np.column_stack((c - radius * np.cos(theta), radius * np.sin(theta)))
    left[:, 1] *= rise / circle_rise
    return np.vstack((left, left[-2::-1] * [-1, 1]))


def cusped_loop(rx, ry=None, lobes=4, segments=32, phase=0):
    """A foil boundary with deliberate inward cusps, rather than a round badge."""
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    radius = .72 + .28 * np.abs(np.cos(lobes * (angles - phase) / 2))
    return np.column_stack((rx * radius * np.cos(angles),
                            (rx if ry is None else ry) * radius * np.sin(angles)))


class Cathedral:
    def __init__(self, request):
        self.request = request
        self.w, self.h, self.d = [float(default if value is None else value)
            for default, value in zip(DEFAULTS, (request.width, request.height, request.depth))]
        if not all(math.isfinite(x) and 12 <= x <= 200 for x in (self.w, self.h, self.d)):
            raise ValueError("cathedral dimensions must be finite and between 12 and 200")
        if not 1.6 <= self.d / self.w <= 2.6 or not 1.2 <= self.h / self.w <= 1.9:
            raise ValueError("cathedral requires depth/width 1.6..2.6 and height/width 1.2..1.9")
        if request.detail not in {"concept", "standard", "presentation"}:
            raise ValueError("cathedral detail must be concept, standard, or presentation")
        if request.openings != "auto" or request.levels is not None:
            raise ValueError("cathedrals author their own openings and elevations; use openings=auto and omit levels")
        self.rng = random.Random(request.seed)
        self.variant = request.variant
        if self.variant == "auto": self.variant = "high-gothic"
        if self.variant == "random": self.variant = self.rng.choice(VARIANTS)
        if self.variant not in VARIANTS:
            raise ValueError(f"cathedral variant must be auto, random, or one of {', '.join(VARIANTS)}")
        self.bays = self.integer(request.bays if request.bays is not None else 7, "bays", 4, 10)
        params = request.parameters
        if self.variant=="early-english" and "spire_ratio" in params:
            raise ValueError("early-english uses a dominant crossing spire; omit west spire_ratio")
        if set(params) - {"choir_bays", "apse_sides", "spire_ratio"}:
            raise ValueError("cathedral parameters: choir_bays, apse_sides, spire_ratio")
        self.choir_bays = self.integer(params.get("choir_bays", 3), "choir_bays", 2, 4)
        self.apse_sides = self.integer(params.get("apse_sides", 7), "apse_sides", 5, 9)
        if self.apse_sides % 2 != 1: raise ValueError("apse_sides must be 5, 7 or 9")
        ratio = params.get("spire_ratio", .30 if self.variant != "early-english" else .34)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not .22 <= ratio <= .36:
            raise ValueError("spire_ratio must be between 0.22 and 0.36")
        self.spire_ratio = float(ratio)
        self.detail = request.detail
        self.segments = {"concept": 8, "standard": 14, "presentation": 14}[self.detail]
        self.placements, self.counts = [], Counter()
        self.root = None
        self.b = self.h * .009
        self.n = self.w * .162
        self.outer = self.w * .326
        self.front, self.back = -self.d * .418, self.d * .30
        self.cross_width = self.w * .255
        # Nave and choir share one bay pitch, so the elevation keeps its rhythm.
        start = self.front + self.w * .20
        pitch = (self.back - start - self.cross_width) / (self.bays + self.choir_bays)
        self.cross = start + pitch * self.bays + self.cross_width / 2
        self.eave = self.h * .435
        self.aisle_h = self.h * .23
        self.arcade_h = self.h * .245
        self.ridge_h = self.h * .58
        self.t = self.w * .014

    @staticmethod
    def integer(value, name, low, high):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}")
        return value

    def add(self, role, position, dimensions, block_type="CUBE", **params):
        pid = f"cathedral.{role}.{self.counts[role]:04d}"
        self.counts[role] += 1
        self.placements.append(Placement(id=pid, kind=KINDS[role], role=role, block_type=block_type,
            position=tuple(float(x) for x in position), dimensions=tuple(float(x) for x in dimensions),
            material=(0, 55, 14, 90) if role=="glazing" else (0, 255, 65, 255), parent_id=self.root,
            parameters={"style": self.variant, **params},
            substance=CATHEDRAL_SUBSTANCES.get(role, CATHEDRAL_DEFAULT_SUBSTANCE)))
        return pid

    def ashlar(self, role, position, dimensions, **params):
        host = f"cathedral.{role}.{self.counts[role]:04d}"
        last = None
        for stone in fill_box(position, dimensions, "standard" if self.detail == "presentation" else self.detail):
            last = self.add(role, stone.position, stone.dimensions,
                            construction="ashlar running bond", unit=stone.unit,
                            course=stone.course, host_id=host, axis=stone.axis, **params)
        return last

    def stack_member(self, role, x, bottom, z, width, depth, height, block_type="CUBE", **params):
        host = f"cathedral.{role}.{self.counts[role]:04d}"
        last = None
        for center_y, course_h, index in _iter_tall_stack(bottom, height):
            extra = {"construction": "ashlar stack", "unit": "drum", "course": index, "host_id": host, **params}
            if block_type == "CYLINDER":
                extra["radius"] = extra.get("radius", min(width, depth) / 2)
                extra["height"] = course_h
            last = self.add(role, (x, center_y, z), (width, course_h, depth), block_type, **extra)
        return last

    def mesh(self, role, mesh, **params):
        position, dimensions, data = mesh_parameters(mesh)
        return self.add(role, position, dimensions, "MESH", **data, **params)

    def profile(self, role, polygon, depth, x=0, y=0, z=0, angle=0, **params):
        if role == "glazing":
            mesh = extrude_profile(polygon, depth)
            if angle:
                mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
            mesh.apply_translation([x, y, z])
            dims = mesh.bounds[1] - mesh.bounds[0]
            # Glass stays a thin removable pane, never ashlar. Split a light
            # that is taller than a cubit so phosphor shows leads, not one
            # 2 m plate. Span may run longer so presentation stays under cap.
            glass_h, glass_span = MAX_COURSE, MAX_STRETCHER * 2.5
            if (float(dims[1]) <= glass_h + 1e-6
                    and max(float(dims[0]), float(dims[2])) <= glass_span + 1e-6):
                return self.mesh(role, mesh, **params)
            host = f"cathedral.glazing.{self.counts['glazing']:04d}"
            last = None
            try:
                pieces = list(iter_profile_pieces(
                    polygon, depth, angle, max_course=glass_h, max_stretcher=glass_span))
            except ValueError:
                pieces = []
            if not pieces:
                return self.mesh(role, mesh, **params)
            for piece, course in pieces:
                pane = extrude_profile(piece, depth)
                if angle:
                    pane.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
                pane.apply_translation([x, y, z])
                last = self.mesh(role, pane, course=course, host_id=host, **params)
            return last
        mesh = extrude_profile(polygon, depth)
        if angle:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
        mesh.apply_translation([x, y, z])
        return self.mesh(role, mesh, **params)

    def plate(self, role, polygon, depth, x=0, y=0, z=0, angle=0, **params):
        """Course a wall plate: ashlars when the elevation is a facing rectangle."""
        if abs(float(depth)) > MAX_STRETCHER + 1e-6:
            return self.tiles(role, polygon, depth, x, y, z, angle, **params)
        box = _rect_world_box(polygon, depth, x, y, z, angle)
        if box is not None:
            return self.ashlar(role, box[0], box[1], **params)
        return self._course_profile(role, polygon, depth, x, y, z, angle, **params)

    def _iter_profile_pieces(self, polygon, depth, angle=0):
        yield from iter_profile_pieces(polygon, depth, angle)

    def _course_profile(self, role, polygon, depth, x=0, y=0, z=0, angle=0, **params):
        """Split a non-rectangular (or rotated) elevation into human-scale bands."""
        meta = {"construction": "ashlar running bond", "unit": "profile_course", **params}
        host = meta.get("host_id") or f"cathedral.{role}.{self.counts[role]:04d}"
        meta["host_id"] = host
        last = None
        for piece, course in iter_fitted_profile_pieces(polygon, depth, angle, x, y, z):
            try:
                last = self.profile(
                    role, piece, depth, x, y, z, angle, **{**meta, "course": course})
            except ValueError:
                continue
        if last is None:
            try:
                mesh = _profile_mesh(polygon, depth, angle, x, y, z)
            except ValueError:
                return None
            if _cubit_aabb(mesh):
                last = self.mesh(role, mesh, **params)
        return last

    def tiles(self, role, polygon, depth, x=0, y=0, z=0, angle=0, **params):
        """Course a long extruded plate along the run so a nave roof is tiles, not a slab."""
        run = float(depth)
        count = max(1, int(math.ceil(abs(run) / MAX_STRETCHER - 1e-9)))
        chunk = run / count
        axis = np.array([math.sin(angle), 0.0, math.cos(angle)])
        origin = np.array([float(x), float(y), float(z)]) - axis * (run / 2.0)
        last = None
        if role in _SLATED and count > 1:
            # Slates break joint: odd courses shift half a stone along the run,
            # so a roof reads as laid courses rather than stack-bonded paper.
            host = f"cathedral.{role}.{self.counts[role]:04d}"
            pieces = list(iter_fitted_profile_pieces(polygon, chunk, angle, x, y, z))
            levels = sorted({round(float(np.min(np.asarray(piece)[:, 1])), 6) for piece, _ in pieces})
            for piece, course in pieces:
                band = levels.index(round(float(np.min(np.asarray(piece)[:, 1])), 6))
                shift = chunk / 2 if band % 2 else 0.0
                cuts = sorted({0.0, run, *[shift + chunk * k for k in range(count + 1)
                                            if 0.0 < shift + chunk * k < run]})
                for s0, s1 in zip(cuts, cuts[1:]):
                    if s1 - s0 < MIN_STONE:
                        continue
                    center = origin + axis * ((s0 + s1) / 2)
                    try:
                        last = self.profile(role, piece, s1 - s0, float(center[0]), float(center[1]),
                                            float(center[2]), angle, construction="ashlar running bond",
                                            unit="profile_course", course=course, host_id=host, **params)
                    except ValueError:
                        continue
            if last is not None:
                return last
        for index in range(count):
            center = origin + axis * ((index + 0.5) * chunk)
            last = self.plate(
                role, polygon, chunk, float(center[0]), float(center[1]), float(center[2]),
                angle, **params)
        return last

    def rib(self, role, points, radius, closed=False, **params):
        sections = 6 if self.detail == "concept" else 8
        radius = float(radius)
        spacing = max(MIN_STONE * 2, min(MAX_STRETCHER, MAX_COURSE) - 2.0 * radius - 0.05)
        pts = [np.asarray(p, dtype=float).reshape(3) for p in points]
        if closed:
            if len(pts) >= 2 and float(np.linalg.norm(pts[0] - pts[-1])) <= 1e-9:
                pts = pts[:-1]
            loop = tube_path(pts, radius, sections=sections, closed=True)
            extent = loop.bounds[1] - loop.bounds[0]
            if (extent[1] <= MAX_COURSE + 1e-6
                    and max(float(extent[0]), float(extent[2])) <= MAX_STRETCHER + 1e-6):
                return self.mesh(role, loop, construction="ashlar stack",
                                 unit="profile_course", **params)
            pts = pts + [pts[0]]
        # Keep every authored point, so a foil stays round rather than becoming
        # a chorded hexagon. Only overlong straight segments are subdivided.
        samples = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            vec = b - a
            dist = float(np.linalg.norm(vec))
            if dist < 1e-9:
                continue
            steps = max(1, int(math.ceil(dist / spacing - 1e-9)))
            for k in range(1, steps + 1):
                samples.append(a + vec * (k / steps))
        host = f"cathedral.{role}.{self.counts[role]:04d}"
        last = None
        for index, run in enumerate(_cubit_runs(samples, radius)):
            try:
                mesh = tube_path(run, radius, sections=sections, closed=False)
            except ValueError:
                continue
            last = self.mesh(
                role, mesh, construction="ashlar stack", unit="profile_course", course=index,
                host_id=host, **params)
        if last is not None:
            return last
        path = pts if closed else [np.asarray(p, dtype=float).reshape(3) for p in points]
        if len(path) >= 2:
            denser = _polyline_samples(path, max(MIN_STONE, spacing * 0.5), closed=closed)
            for index, (a, b) in enumerate(zip(denser[:-1], denser[1:])):
                if float(np.linalg.norm(b - a)) <= 1e-9:
                    continue
                last = self.mesh(
                    role, tube_path([a, b], radius, sections=sections, closed=False),
                    construction="ashlar stack", unit="profile_course", course=index,
                    host_id=host, **params)
            if last is not None:
                return last
        if closed:
            loop = tube_path(pts, radius, sections=sections, closed=True)
            extent = loop.bounds[1] - loop.bounds[0]
            if (extent[1] <= MAX_COURSE + 1e-6
                    and max(float(extent[0]), float(extent[2])) <= MAX_STRETCHER + 1e-6):
                return self.mesh(role, loop, **params)
        return self.mesh(role, tube_path(points, radius, sections=sections, closed=False), **params)

    def profiled_prism(self, role, x, bottom, z, width, depth, height, profile=None, round_section=False):
        """One carved member, with continuous beveled bases, necks and capitals."""
        profile = profile or ((0, .84), (.17, 1), (.46, 1), (1, .75))
        max_scale = max(abs(float(scale)) for _, scale in profile)
        if (height > MAX_COURSE + 1e-6
                or max(width, depth) > MAX_STRETCHER + 1e-6
                or max(width, depth) * max_scale > MAX_STRETCHER + 1e-6):
            if round_section:
                return self.stack_member(role, x, bottom, z, width, depth, height, "CYLINDER",
                                         radius=min(width, depth) / 2, sections=8)
            return self.ashlar(role, (x, bottom + height / 2.0, z), (width, height, depth))
        if round_section:
            footprint = np.array([(math.cos(a) / 2, math.sin(a) / 2)
                                  for a in np.linspace(math.pi / 8, 2 * math.pi + math.pi / 8, 8, endpoint=False)])
        else:
            footprint = np.array([(-.34, -.5), (.34, -.5), (.5, -.34), (.5, .34),
                                  (.34, .5), (-.34, .5), (-.5, .34), (-.5, -.34)])
        vertices = [[x + width * scale * a, bottom + height * y, z + depth * scale * b]
                    for y, scale in profile for a, b in footprint]
        faces = []
        for level in range(len(profile) - 1):
            for i in range(8):
                a = level * 8 + i
                b = level * 8 + (i + 1) % 8
                faces.extend(((a, b, b + 8), (a, b + 8, a + 8)))
        top = (len(profile) - 1) * 8
        for i in range(1, 7):
            faces.extend(((0, i + 1, i), (top, top + i, top + i + 1)))
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.fix_normals(multibody=False)
        return self.mesh(role, mesh)

    def molded_shaft(self, role, x, bottom, z, height, radius):
        neck = min(self.t * .55 / height, .15)
        profile = ((0, 1.85), (neck * .20, 1.85), (neck * .55, 1.38), (neck, 1),
                   (1 - neck, 1), (1 - neck * .55, 1.30), (1 - neck * .22, 1.75), (1, 1.75))
        return self.profiled_prism(role, x, bottom, z, radius * 2, radius * 2, height,
                                   profile, round_section=True)

    def course(self, role, x, bottom, z, width, height, projection, angle=0, normal=1):
        # Split the run so a cornice is stretchers, not one 10 m extrusion.
        section = np.array([(-.42, 0), (.48, 0), (.48, .19), (.27, .40),
                            (.27, .69), (.10, 1), (-.42, 1)])
        section *= [projection * normal, height]
        return self.tiles(role, section, width, x, bottom, z, angle - math.pi / 2)

    def foil(self, role, x, y, z, rx, ry=None, angle=0, lobes=4, radius=None, phase=0):
        points = cusped_loop(rx, ry, lobes, 32 if self.detail != "concept" else 24, phase)
        return self.rib(role, [[x + u * math.cos(angle), y + v, z - u * math.sin(angle)]
                               for u, v in points], radius or self.t * .095, True)

    def pierced_tympanum(self, polygon, center, rx, ry, depth, x, bottom, z, angle):
        """Extrude a tympanum around a trefoil void without a boolean backend."""
        polygon = np.asarray(polygon, dtype=float)
        center = np.asarray(center, dtype=float)
        # Include every outer corner, so the panel still meets its archivolt.
        angles = np.concatenate((np.linspace(0, 2 * math.pi, 48, endpoint=False),
            np.mod(np.arctan2((polygon[:, 1] - center[1]) / ry,
                             (polygon[:, 0] - center[0]) / rx), 2 * math.pi)))
        angles = np.unique(np.round(angles, 12))
        outer, inner = [], []
        edges = np.roll(polygon, -1, axis=0) - polygon
        relative = polygon - center
        for a in angles:
            ray = np.array([rx * math.cos(a), ry * math.sin(a)])
            divisor = ray[0] * edges[:, 1] - ray[1] * edges[:, 0]
            valid = np.abs(divisor) > 1e-12
            distances = np.full(len(edges), np.inf)
            edge_u = np.full(len(edges), np.inf)
            distances[valid] = (relative[valid, 0] * edges[valid, 1]
                                - relative[valid, 1] * edges[valid, 0]) / divisor[valid]
            edge_u[valid] = (relative[valid, 0] * ray[1] - relative[valid, 1] * ray[0]) / divisor[valid]
            hit = (distances > 0) & (edge_u >= -1e-9) & (edge_u <= 1 + 1e-9)
            distance = float(np.min(distances[hit]))
            foil_radius = .72 + .28 * abs(math.cos(3 * (a - math.pi / 2) / 2))
            if distance <= foil_radius:
                raise ValueError("tympanum trefoil must remain inside its masonry surround")
            outer.append(center + distance * ray)
            inner.append(center + foil_radius * ray)
        count = len(outer)
        host = f"cathedral.portal_gable.{self.counts['portal_gable']:04d}"
        remaining, start = count, 0
        while remaining > 0:
            n_edges = 1
            while n_edges < remaining:
                trial = _annulus_strip_mesh(outer, inner, depth, start, n_edges + 1)
                if angle:
                    trial.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0]))
                if _cubit_aabb(trial):
                    n_edges += 1
                else:
                    break
            # Sector polygons keep the trefoil outside the ring; Y-clip cannot
            # bridge the portal void the way a U-surround plate would.
            self._course_profile(
                "portal_gable", _annulus_strip_polygon(outer, inner, start, n_edges),
                depth, x, bottom, z, angle, host_id=host)
            start = (start + n_edges) % count
            remaining -= n_edges
        return np.asarray(inner)

    def arch(self, role, x, bottom, z, width, height, depth, thickness=None, angle=0, **params):
        a = width / 2
        t = thickness or self.t * .38
        rise = min(height * .50, a * 1.58)
        spring = height - rise
        outer = pointed_arc(a, rise, self.segments) + [0, spring]
        inner = pointed_arc(a - t, rise - t, self.segments) + [0, spring]
        if params.get("cusped"):
            # The intrados itself forms the cusps; they are not overlaid marks.
            station = max(2, round(self.segments * .55))
            for index in (station, len(inner) - station - 1):
                weights = np.maximum(0, 1 - np.abs(np.arange(len(inner)) - index) / 2)
                inner[:, 0] -= np.sign(inner[:, 0]) * weights * min(t * 1.15, (a - t) * .20)
        # Jambs and two half-heads; a U-surround would bridge the void when Y-clipped.
        if spring > MIN_STONE * 0.5:
            self.plate(role, [(-a, 0), (-a + t, 0), (-a + t, spring), (-a, spring)],
                       depth, x, bottom, z, angle, **params)
            self.plate(role, [(a - t, 0), (a, 0), (a, spring), (a - t, spring)],
                       depth, x, bottom, z, angle, **params)
        apex = self.segments
        self.plate(role, np.vstack((outer[:apex + 1], inner[apex::-1])),
                   depth, x, bottom, z, angle, **params)
        return self.plate(role, np.vstack((outer[apex:], inner[:apex - 1:-1])),
                          depth, x, bottom, z, angle, **params)

    def portal_infill(self, x, z, width, height, wall_half, top, angle=0, depth=None):
        """Solid wall around a portal's outer archivolt, up to the register above.

        Without it the pointed head leaves open corners (and slots beside the
        jambs) between the doorway and the next masonry course.
        """
        # The fill tucks a little behind the outer moulding, so no sliver
        # opens at the crown where the archivolt is coursed.
        a = width / 2
        rise = min(height * .50, a * 1.58)
        spring = height - rise
        tuck = self.t * .3
        arc = pointed_arc(a - tuck, rise - tuck, self.segments) + [0, self.b + spring]
        a -= tuck
        left = arc[:self.segments + 1]
        for side in (-1, 1):
            curve = left if side == -1 else left[::-1] * [-1, 1]
            if side == -1:
                ring = [(-wall_half, self.b), (-a, self.b), *map(tuple, curve), (0, top), (-wall_half, top)]
            else:
                ring = [(0, top), *map(tuple, curve), (a, self.b), (wall_half, self.b), (wall_half, top)]
            ring = _clip_poly(ring, y1=top)
            if len(ring) >= 3 and _poly_area(ring) > MIN_STONE ** 2:
                self.plate("wall_spandrel", ring, depth or self.t, x, 0, z, angle, register="portal_infill")

    def window(self, role, x, bottom, z, width, height, depth, angle=0, lights=2, glazed=True, **params):
        """A pierced wall bay: side jambs, curved spandrel, bars and open lights."""
        t = self.t * .44
        self.arch(role, x, bottom, z, width, height, depth, t, angle, **params)
        half=width/2;rise=min(height*.50,half*1.58);spring=height-rise
        curve=pointed_arc(half,rise,self.segments)+[0,spring]
        # Two solid shoulders bind the pointed opening into a rectangular bay.
        # Splitting at the apex avoids a self-touching polygon at the crown.
        left=curve[:self.segments+1]
        right=curve[self.segments:]
        # Shoulders sit a hair behind the archivolt face, so the two never
        # share a coplanar face or an identical sliver envelope.
        self.plate("wall_spandrel",np.vstack((left,[-half,height])),depth*.96,x,bottom,z,angle)
        self.plate("wall_spandrel",np.vstack((right,[half,height])),depth*.96,x,bottom,z,angle)
        if not glazed:
            # An open internal archway: jambs and head only, no lights.
            return
        pane=np.vstack(([-half+t,0],pointed_arc(half-t,rise-t,self.segments)+[0,spring],[half-t,0]))
        self.profile("glazing",pane,self.t*.035,x,bottom,z,angle)
        # A small ledge binds the light to its bay; no opaque window infill.
        self.plate("sill", [(-width*.53,-t),(width*.53,-t),(width*.53,0),(-width*.53,0)], depth*1.1,
                     x, bottom, z, angle)
        if self.detail == "concept": return
        # A central mullion and a foil. Inner light arches explode the
        # placement budget past MAX_PLACEMENTS at presentation.
        mullion = t * .70
        # The foil sits inside the pointed head, below its apex.
        r = min(width * .155, (rise - t) * .27, height * .095)
        center_y = bottom + spring + (rise - t) * .40
        if lights == 2:
            self.plate("facade_mullion", [(-mullion/2, 0), (mullion/2, 0),
                         (mullion/2, center_y-bottom-r*.86), (-mullion/2, center_y-bottom-r*.86)],
                         depth*.68, x, bottom, z, angle)
        elif lights >= 3:
            # Mullions divide the lights evenly and die into the arch head.
            head = pointed_arc(half-t, rise-t, self.segments) + [0, spring]
            left = head[:self.segments+1]
            for k in range(1, lights):
                u = -half + width * k / lights
                top = float(np.interp(-abs(u), left[:, 0], left[:, 1])) - t * .2
                self.plate("facade_mullion", [(u-mullion/2, 0), (u+mullion/2, 0),
                             (u+mullion/2, top), (u-mullion/2, top)],
                             depth*.68, x, bottom, z, angle)
        # A trefoil stands on two lobes with the third pointing up.
        self.foil("window_tracery", x, center_y, z, r, angle=angle,
                  lobes=3 if self.variant=="early-english" else 4, radius=t*.42,
                  phase=math.pi/2 if self.variant=="early-english" else 0)

    def drum(self, x, z, bottom, top, r_out, r_in):
        """A hollow octagonal drum laid as eight facet stones per course."""
        host = f"cathedral.tower_shaft.{self.counts['tower_shaft']:04d}"
        angles = np.linspace(math.pi / 8, 2 * math.pi + math.pi / 8, 8, endpoint=False)
        n = _strict_courses(top - bottom, MAX_COURSE)
        for j in range(n):
            y0 = bottom + (top - bottom) * j / n
            y1 = bottom + (top - bottom) * (j + 1) / n
            for i in range(8):
                a0, a1 = angles[i], angles[(i + 1) % 8]
                pts = [[x + r * math.cos(a), y, z + r * math.sin(a)]
                       for y in (y0, y1) for r in (r_out, r_in) for a in (a0, a1)]
                self.mesh("tower_shaft", trimesh.Trimesh(vertices=pts).convex_hull,
                          construction="ashlar stack", unit="profile_course", course=j,
                          facet=i, host_id=host, register="spire_drum")

    def gabled_shoulder(self, x, z, bottom, eave, peak, width, depth, tip_radius):
        """Four pierced, receding gables carry the tower directly into its spire."""
        height=peak-bottom
        shoulder=eave-bottom
        for angle in (0,math.pi/2,math.pi,math.pi*1.5):
            face_width=(depth if abs(math.cos(angle))>.5 else width)*.82
            base_radius=(width if abs(math.cos(angle))>.5 else depth)*.455
            opening_half=face_width*.32
            opening_height=height*.83
            rise=min(opening_height*.58,opening_half*1.55)
            spring=opening_height-rise
            segments=min(self.segments,14)
            curve=pointed_arc(opening_half,rise,segments)+[0,spring]
            apex_i=segments
            # Two halves; a U-panel would bridge the void when Y-clipped.
            left_panel=np.vstack((
                [(-face_width/2,0),(-face_width/2,shoulder),(0,height)],
                curve[apex_i::-1],[(-opening_half,0)]))
            right_panel=np.vstack((
                [(0,height),(face_width/2,shoulder),(face_width/2,0),(opening_half,0)],
                curve[:apex_i-1:-1]))

            def place(role,profile,thickness,recess=0,register="crown_panel"):
                host=f"cathedral.{role}.{self.counts[role]:04d}"
                last=None
                for piece, course in self._iter_profile_pieces(profile, thickness, angle):
                    mesh=extrude_profile(piece,thickness)
                    # The face leans inward continuously. Gable tips meet the
                    # lower taper's apothem instead of stopping at another shelf.
                    mesh.vertices[:,2] += base_radius+(tip_radius-base_radius)*mesh.vertices[:,1]/height+recess
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2-angle,[0,1,0]))
                    mesh.apply_translation([x,bottom,z])
                    last=self.mesh(role,mesh,register=register,construction="ashlar running bond",
                                   unit="profile_course",course=course,host_id=host)
                return last

            place("tower_lancet",left_panel,self.t*.54)
            place("tower_lancet",right_panel,self.t*.54)
            section=self.t*.22
            outer=pointed_arc(opening_half+section,rise+section,segments)+[0,spring]
            inner=pointed_arc(opening_half-section*.35,rise-section*.35,segments)+[0,spring]
            left_reveal=np.vstack((
                [(-opening_half-section,0)],outer[:apex_i+1],inner[apex_i::-1],
                [(-opening_half+section*.35,0)]))
            right_reveal=np.vstack((
                [(opening_half+section,0)],outer[:apex_i-1:-1],inner[apex_i:],
                [(opening_half-section*.35,0)]))
            place("tower_lancet",left_reveal,self.t*.30,-self.t*.27,"crown_reveal")
            place("tower_lancet",right_reveal,self.t*.30,-self.t*.27,"crown_reveal")
            # Two rakes; a chevron coping double-backs when Y-clipped.
            place("tower_cornice",[(-face_width/2,shoulder),(0,height),
                    (0,height-section*1.5),(-face_width/2+section,shoulder)],
                  self.t*.70,self.t*.035,"crown_gable")
            place("tower_cornice",[(0,height),(face_width/2,shoulder),
                    (face_width/2-section,shoulder),(0,height-section*1.5)],
                  self.t*.70,self.t*.035,"crown_gable")

    def spire(self, x, bottom, z, width, height, role="spire", shoulder=None):
        # A carved lower shoulder gives way to a deliberately quiet taper.
        if role=="pinnacle":
            foot=height*.26
            self.profiled_prism("buttress_cap",x,bottom,z,width*.96,width*.96,foot,
                                ((0,1),(.16,1),(.27,.74),(.78,.74),(1,.88)))
            for angle in (0,math.pi/2):
                for side in (-1,1):
                    self.plate("pinnacle",[(-width*.5,0),(0,foot*.62),(width*.5,0)],self.t*.17,
                        x+side*width*.46 if angle else x,bottom+foot,
                        z if angle else z+side*width*.46,angle)
            bottom+=foot;height-=foot
        finial_height = min(self.t * .72, height * .08)
        taper_height = height - finial_height
        open_height = taper_height * .18 if role=="spire" and (height > width*.9 or shoulder) else 0
        base_radius = .5
        if open_height:
            lower_radius, upper_radius = width*.5, width*.405
            if shoulder:
                self.gabled_shoulder(x,z,shoulder["bottom"],shoulder["eave"],bottom+open_height,
                                     shoulder["width"],shoulder["depth"],upper_radius*math.cos(math.pi/8))
                # A solid inner drum carries the spire, so only the near
                # openings read instead of a tangle of far-side arches.
                self.drum(x,z,shoulder["bottom"],bottom+open_height,upper_radius*.80,
                          upper_radius*.80-self.t*.8)
            else:
                # The crossing lantern retains its smaller octagonal broach.
                face_width = 2*lower_radius*math.sin(math.pi/8)
                opening_half = face_width*.35
                opening_height = open_height*.85
                rise = min(opening_height*.64, opening_half*1.55)
                spring = opening_height-rise
                segs=max(6,self.segments//2)
                curve=pointed_arc(opening_half,rise,segs)+[0,spring]
                apex_i=segs
                left_panel=np.vstack((
                    [(-face_width/2,0),(-face_width/2,open_height),(0,open_height)],
                    curve[apex_i::-1],[(-opening_half,0)]))
                right_panel=np.vstack((
                    [(0,open_height),(face_width/2,open_height),(face_width/2,0),(opening_half,0)],
                    curve[:apex_i-1:-1]))
                for a in np.linspace(0,2*math.pi,8,endpoint=False):
                    for panel in (left_panel, right_panel):
                        host=f"cathedral.tower_lancet.{self.counts['tower_lancet']:04d}"
                        for piece, course in self._iter_profile_pieces(panel, self.t*.27, a):
                            mesh=extrude_profile(piece,self.t*.27)
                            fraction=mesh.vertices[:,1]/open_height
                            mesh.vertices[:,0] *= 1-fraction*(1-upper_radius/lower_radius)
                            mesh.vertices[:,2] += (lower_radius+(upper_radius-lower_radius)*fraction)*math.cos(math.pi/8)
                            mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi/2-a,[0,1,0]))
                            mesh.apply_translation([x,bottom,z])
                            self.mesh("tower_lancet",mesh,register="spire_openwork",
                                      construction="ashlar running bond",unit="profile_course",
                                      course=course,host_id=host)
            base_radius = .405
            points=[[x+upper_radius*math.cos(a),bottom+open_height,z+upper_radius*math.sin(a)]
                    for a in np.linspace(math.pi/8,2*math.pi+math.pi/8,8,endpoint=False)]
            self.rib("spire_rib",points,self.t*.115,True,register="shoulder_collar")
        start_fraction = open_height/taper_height if taper_height > _EPS else 0.0
        rings = [(start_fraction, base_radius),
                 (max(start_fraction+.025,.10),base_radius*.96),(.94,.035),(1,.012)]
        n = _strict_courses(taper_height, MAX_COURSE) if taper_height > _EPS else 1
        angles = np.linspace(math.pi/8,2*math.pi+math.pi/8,8,endpoint=False)
        ys = np.linspace(start_fraction, 1.0, n + 1)
        rs = np.interp(ys, [y for y, _ in rings], [r for _, r in rings])
        host = f"cathedral.{role}.{self.counts[role]:04d}"
        shell = min(self.t * .22, width * .08)
        for j in range(n):
            y_a, y_b = float(ys[j]), float(ys[j + 1])
            r_a, r_b = float(rs[j]), float(rs[j + 1])
            y0, y1 = bottom + taper_height * y_a, bottom + taper_height * y_b
            inner_scale_a = max(r_a - shell / max(width, MIN_STONE), r_a * .72)
            inner_scale_b = max(r_b - shell / max(width, MIN_STONE), r_b * .72)
            if 2 * width * r_a <= MAX_STRETCHER + 1e-6:
                # A slender course is one carved ring, not eight facet chips.
                self.mesh(role, _octagon_ring(x, z, y0, y1, width * r_a, width * r_b,
                                              width * inner_scale_a, width * inner_scale_b, angles),
                          construction="ashlar stack", unit="profile_course",
                          course=j, host_id=host)
                continue
            # A wide course splits each flat facet into stone-length pieces.
            chord = 2 * width * r_a * math.sin(math.pi / 8)
            pieces = max(1, int(math.ceil(chord / (MAX_STRETCHER - 1e-4))))
            for i in range(8):
                a0, a1 = float(angles[i]), float(angles[(i + 1) % 8])

                def on_face(radius, s, a0=a0, a1=a1):
                    p0 = np.array([math.cos(a0), math.sin(a0)]) * width * radius
                    p1 = np.array([math.cos(a1), math.sin(a1)]) * width * radius
                    return p0 + (p1 - p0) * s

                for piece in range(pieces):
                    s0, s1 = piece / pieces, (piece + 1) / pieces
                    # Each course is a true frustum segment: its outer face runs
                    # from the lower radius to the upper one, so the taper reads
                    # as one continuous slope instead of stacked flaring tiers.
                    verts = []
                    for y, r_out, r_in in ((y0, r_a, inner_scale_a), (y1, r_b, inner_scale_b)):
                        for radius, frac in ((r_out, s0), (r_out, s1), (r_in, s1), (r_in, s0)):
                            u, v = on_face(radius, frac)
                            verts.append([x + u, y, z + v])
                    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                             (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
                             (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    mesh.fix_normals(multibody=False)
                    self.mesh(role, mesh, construction="ashlar stack", unit="profile_course",
                              course=j, host_id=host, facet=i, **({"piece": piece} if pieces > 1 else {}))
        if shoulder and open_height:
            # Gable heads continue into short meridian ribs; the upper three
            # quarters of the taper retain their quiet, uninterrupted outline.
            fractions=np.array([start_fraction,rings[1][0],min(.32,rings[1][0]+.08)])
            radii=np.interp(fractions,[y for y,r in rings],[r for y,r in rings])*width*math.cos(math.pi/8)
            for a in (0,math.pi/2,math.pi,math.pi*1.5):
                self.rib("spire_rib",[[x+r*math.cos(a),bottom+taper_height*y,z+r*math.sin(a)]
                                     for y,r in zip(fractions,radii)],self.t*.10,
                         register="gable_meridian")
        if role=="spire":
            self.profiled_prism("finial",x,bottom+taper_height,z,self.t*.32,self.t*.32,finial_height,
                                ((0,.62),(.22,1),(.46,.62),(.70,.38),(1,.10)),round_section=True)
            points=[[x+self.t*.19*math.cos(a),bottom+taper_height+finial_height*.23,
                     z+self.t*.19*math.sin(a)] for a in np.linspace(0,2*math.pi,12,endpoint=False)]
            self.rib("finial",points,self.t*.055,True)
        else:
            self.profiled_prism("finial",x,bottom+taper_height,z,self.t*.16,self.t*.16,finial_height,
                                ((0,.75),(.25,1),(.5,.55),(1,.12)),round_section=True)

    def foundation(self):
        # Cruciform terrace as paving, not one plan-sized MESH.
        # Wide enough to seat the outer buttress caps on the plinth.
        x=self.outer*1.2+self.w*.05;tx=self.w*.51;z0=self.front-self.w*.08;z1=self.back
        c0=self.cross-self.cross_width*.64;c1=self.cross+self.cross_width*.64
        # Disjoint convex pieces: nave terrace, two transept arms and the
        # eastern apse plinth, a constant margin outside the apse facets.
        # Each paving cell is clipped to them, so edges stay straight.
        rect=lambda a,b,c,d:[(a,c),(b,c),(b,d),(a,d)]
        half=math.pi/(2*self.apse_sides)
        ring=(self.outer+self.w*.06)/math.cos(half)
        apse=[(x,z1),*[(ring*math.cos(a),z1+ring*math.sin(a))
                       for a in np.linspace(0,math.pi,self.apse_sides+1)[1:-1]],(-x,z1)]
        pieces=[rect(-x,x,z0,z1),rect(x,tx,c0,c1),rect(-tx,-x,c0,c1),apse[::-1]]
        xs=[-tx,tx]; zs=[z0,z1+ring]
        module=min(MAX_STRETCHER, 1.5)
        x0, x1, z_lo, z_hi = min(xs), max(xs), min(zs), max(zs)
        nx=max(1, int(math.ceil((x1-x0)/module-1e-9)))
        nz=max(1, int(math.ceil((z_hi-z_lo)/module-1e-9)))
        dx, dz = (x1-x0)/nx, (z_hi-z_lo)/nz
        self.root=None
        host=f"cathedral.foundation.{self.counts['foundation']:04d}"
        for i in range(nx):
            for j in range(nz):
                cell=rect(x0+i*dx,x0+(i+1)*dx,z_lo+j*dz,z_lo+(j+1)*dz)
                for piece in pieces:
                    clipped=_clip_convex(cell,piece)
                    if len(clipped)<3 or _poly_area(clipped)<MIN_STONE**2:
                        continue
                    cx,cz=np.mean(clipped,axis=0)
                    kw=dict(construction="crepidoma paving",unit="paving",
                            course=0,bond="running",host_id=host)
                    if abs(_poly_area(clipped)-dx*dz)<1e-9*max(1.0,dx*dz):
                        pid=self.add("foundation",(cx,self.b/2,cz),(dx,self.b,dz),**kw)
                    else:
                        slab=extrude_profile(clipped,self.b)
                        slab.vertices=slab.vertices[:,[0,2,1]];slab.invert()
                        slab.apply_translation([0,self.b/2,0])
                        pid=self.mesh("foundation",slab,**kw)
                    if self.root is None:
                        self.root=pid
        if self.root is None:
            raise ValueError("cathedral foundation produced no paving")
        for placement in self.placements:
            if placement.parent_id is None and placement.id != self.root:
                placement.parent_id=self.root
        for i in range(3):
            self.ashlar("approach_step", (0,self.b*(i+1)/6,self.front-self.w*.08-self.t*(3-i)*.8),
                (self.w*(.66-i*.03),self.b*(i+1)/3,self.t*1.7))

    def pier(self,x,z,bottom,height,role="nave_pier",radius=None):
        r=radius or self.w*.014
        self.stack_member(role,x,bottom,z,r*2,r*2,height,"CYLINDER",radius=r,
                          sections=8 if role=="apse_pier" else 12)
        self.add("pier_capital",(x,bottom+self.t*.25,z),(r*3.5,self.t*.5,r*3.5))
        self.add("pier_capital",(x,bottom+height-self.t*.22,z),(r*3.2,self.t*.44,r*3.2))
        if self.detail != "concept" and role!="apse_pier":
            for a in (0,math.pi/2,math.pi,math.pi*1.5):
                self.stack_member("pier_shaft",x+r*math.cos(a),bottom,z+r*math.sin(a),
                    r*.54,r*.54,height,"CYLINDER",radius=r*.27,sections=8)

    def buttress(self,x,z,sign):
        outer=self.outer*1.20*sign;bottom=self.b
        pier_h=self.h*.31
        pitch=min((self.cross-self.cross_width/2-self.front-self.w*.20)/self.bays,
                  (self.back-self.cross-self.cross_width/2)/self.choir_bays)
        max_depth=pitch*.64
        self.profiled_prism("buttress_cap",outer,bottom,z,self.w*.084,min(self.w*.049,max_depth),self.t*.82,
                            ((0,1),(.26,1),(.56,.83),(1,.74)))
        # Stepped masonry piers narrow toward their cap.
        for i in range(3):
            height= pier_h/3
            width=self.w*(.041-i*.006);depth=self.w*(.036-i*.005)
            a=width/2;b=depth/2;c=min(a,b)*.26
            footprint=[(-a+c,-b),(a-c,-b),(a,-b+c),(a,b-c),(a-c,b),(-a+c,b),(-a,b-c),(-a,-b+c)]
            lift_h=height+self.t*.05
            lift_bottom=bottom+height*(i+.5)-lift_h/2
            host=f"cathedral.buttress_pier.{self.counts['buttress_pier']:04d}"
            # A wide pier is laid as several stones per course.
            bands=max(1,int(math.ceil(width/(MAX_STRETCHER-1e-4))))
            parts=[_clip_poly(footprint,x0=-a+2*a*k/bands,x1=-a+2*a*(k+1)/bands) for k in range(bands)]
            for center_y, course_h, index in _iter_tall_stack(lift_bottom, lift_h):
                for part in parts:
                    member=extrude_profile(part,course_h)
                    member.vertices=member.vertices[:,[0,2,1]];member.invert()
                    member.apply_translation([outer,center_y,z])
                    self.mesh("buttress_pier",member,construction="ashlar stack",
                              unit="profile_course",course=index,host_id=host)
            if i:
                self.profiled_prism("buttress_cap",outer,bottom+height*i-self.t*.04,z,
                                    width+self.t*.54,depth+self.t*.54,self.t*.38)
        for level in (0,1):
            start=np.array([self.n*sign,self.eave-self.h*(.03+level*.075),z])
            end_y=(bottom+pier_h+self.t*.76) if level==0 else pier_h-self.h*.06
            end=np.array([outer-sign*self.t*1.15,end_y,z])
            # A firm descending coping bridges between masonry abutments.
            # The soffit rises toward it, leaving a narrow positive web at midspan.
            us=np.linspace(0,1,self.segments+1)
            thickness=self.t*.56+self.h*.0015
            end_depth=self.h*(.025 if level==0 else .020)+self.t*.35
            line_y=start[1]+(end[1]-start[1])*us
            top=np.column_stack((start[0]+(end[0]-start[0])*us,line_y))
            lower=top.copy();lower[:,1]-=end_depth-(end_depth-thickness)*np.sin(us*math.pi)
            poly=np.vstack((top,lower[::-1]))
            self.plate("flying_buttress",poly,self.t*.61,z=z,side=sign,bay_z=z)
            coping=top.copy();coping[:,1]+=self.t*.24
            self.plate("buttress_cap",np.vstack((coping,top[::-1])),self.t*.88,z=z,
                         register="flying_coping",side=sign,bay_z=z)
            for point in (start,end):
                block_height=end_depth+self.t*.36
                self.ashlar("buttress_cap",(point[0],point[1]-end_depth/2+self.t*.10,z),
                         (self.t*1.6,block_height,self.t*.75),
                         register="flying_abutment")
            if level:
                self.ashlar("buttress_cap",(outer,end_y-end_depth-self.t*.13,z),
                         (self.w*.029+self.t*.3,self.t*.26,min(self.w*.024+self.t*.3,max_depth)),
                         register="flying_seat")
        cap_bottom=bottom+pier_h-self.t*.18
        self.profiled_prism("buttress_cap",outer,cap_bottom,z,self.w*.029+self.t*.4,
                            min(self.w*.026+self.t*.4,max_depth),self.t*.66,
                            ((0,.70),(.27,.83),(.55,1),(.80,1),(1,.88)))
        self.spire(outer,cap_bottom+self.t*.66,z,self.w*.03,self.h*.092,"pinnacle")

    def longitudinal(self):
        start=self.front+self.w*.20
        c0=self.cross-self.cross_width/2;c1=self.cross+self.cross_width/2
        stations=[*np.linspace(start,c0,self.bays+1),*np.linspace(c1,self.back,self.choir_bays+1)]
        segments=[(a,b) for a,b in zip(stations,stations[1:]) if not (abs(a-c0)<1e-5 and abs(b-c1)<1e-5)]
        pier_h=self.h*.158
        for sign in (-1,1):
            for z in stations:
                self.pier(sign*self.n,z,self.b,pier_h)
                # Transept walls brace the crossing; the apse closes the east end.
                if min(abs(z-c0),abs(z-c1),abs(z-self.back))>1e-5:
                    self.buttress(sign*self.n,z,sign)
                self.ashlar("wall_spandrel",(sign*self.outer,self.b+self.aisle_h*.5,z),
                    (self.t*1.25,self.aisle_h,self.t*1.1))
                self.ashlar("wall_spandrel",(sign*self.n,(self.eave+self.arcade_h)/2,z),
                    (self.t*.9,self.eave-self.arcade_h,self.t))
            for z0,z1 in segments:
                z=(z0+z1)/2;span=z1-z0
                self.arch("nave_arcade",sign*self.n,self.b,z,span*.94,self.arcade_h,self.t*.75,
                          self.t*.57,math.pi/2,bay_z=z)
                self.window("aisle_window",sign*self.outer,self.b+self.h*.036,z,span*.86,self.aisle_h*.77,
                            self.t*.7,math.pi/2,lights=1 if self.variant=="early-english" else 2)
                self.ashlar("aisle_dado",(sign*self.outer,self.b+self.h*.018,z),
                         (self.t*.8,self.h*.036,span))
                # A wall plate carries the aisle eave over the window heads.
                window_top=self.b+self.h*.036+self.aisle_h*.77
                self.ashlar("wall_spandrel",(sign*self.outer,(window_top+self.aisle_h+self.t*.1)/2,z),
                         (self.t*.9,self.aisle_h+self.t*.1-window_top,span))
                # Clerestory heads stop under a wall band and eave cornice.
                self.window("clerestory_frame",sign*self.n,self.h*.285,z,span*.86,
                            self.eave-self.h*.305,self.t*.62,math.pi/2,
                            lights=3 if self.variant=="rayonnant" else 2)
                self.ashlar("wall_spandrel",(sign*self.n,self.eave-self.h*.01,z),
                            (self.t*.8,self.h*.02,span))
                self.ashlar("wall_spandrel",(sign*self.n,self.h*.264,z),(self.t,self.h*.042,span))
                if self.detail != "concept":
                    # The triforium arcade sits clear of the aisle roof.
                    gallery=self.aisle_h+self.h*.026+self.t*.15
                    for j in range(3):
                        self.arch("facade_gallery",sign*self.n,gallery,z+(j-1)*span*.25,
                                  span*.20,self.h*.285-gallery-self.t*.12,self.t*1.12,self.t*.12,math.pi/2)
        for sign in (-1,1):
            for z0,z1 in ((start,c0),(c1,self.back)):
                self.course("facade_mullion",sign*self.n,self.eave-self.t*.08,(z0+z1)/2,z1-z0,
                            self.t*.38,self.t*1.3,math.pi/2,normal=sign)
        # Transverse pointed ribs and crossing diagonals in every vaulted bay.
        all_segments=[*segments,(c0,c1)]
        for z0,z1 in all_segments:
            for z in (z0,):
                arc=pointed_arc(self.n,self.h*.245,self.segments)
                self.rib("vault_rib",[[x,self.b+pier_h+y,z] for x,y in arc],self.t*.22)
            # Four half-ribs rise from the springers and meet at the boss.
            for flip in (-1,1):
                points=[]
                for t in np.linspace(0,1,self.segments*2+1):
                    points.append([-self.n+2*self.n*t,self.b+pier_h+self.h*.245*math.sin(math.pi*t)**.78,
                                   z0+(z1-z0)*(t if flip==1 else 1-t)])
                self.rib("vault_rib",points[:self.segments+1],self.t*.19)
                self.rib("vault_rib",points[:self.segments-1:-1],self.t*.19)
            self.add("vault_boss",(0,self.b+pier_h+self.h*.245,(z0+z1)/2),
                     (self.t*.9,self.t*.45,self.t*.9),"CYLINDER",radius=self.t*.45,height=self.t*.45,sections=10)
        # Two slope quads, tiled along the nave — a chevron shell double-backs when Y-clipped.
        rise = self.ridge_h - self.eave
        thick = self.t * .35
        left = [(-self.n-self.t, 0), (0, rise), (0, rise-thick), (-self.n-self.t, -thick)]
        right = [(0, rise), (self.n+self.t, 0), (self.n+self.t, -thick), (0, rise-thick)]
        run, zc = self.back-self.front, (self.back+self.front)/2
        self.tiles("nave_roof", left, run, 0, self.eave, zc)
        self.tiles("nave_roof", right, run, 0, self.eave, zc)
        self.ashlar("ridge",(0,self.ridge_h,(self.back+self.front)/2),(self.t*.3,self.t*.38,self.back-self.front))
        for sign in (-1,1):
            poly=[(self.n*sign,self.aisle_h+self.h*.026),(self.outer*sign,self.aisle_h),
                  (self.outer*sign,self.aisle_h-self.t*.25),(self.n*sign,self.aisle_h+self.h*.026-self.t*.25)]
            for z0,z1 in ((start,c0),(c1,self.back)):
                self.tiles("aisle_roof",poly,z1-z0,z=(z1+z0)/2)
    def rose(self,x,y,z,r,angle=0,role="rose_tracery"):
        def point(u,v):return [x+u*math.cos(angle),y+v,z-u*math.sin(angle)]
        # The load-bearing outer roll, radial mullions, and outer foil band
        # occupy separate registers. Fine ornament does not repeat every spoke.
        rings=((r,self.t*.29),(r*.92,self.t*.09),(r*.68,self.t*.13),(r*.22,self.t*.15))
        for radius,thick in rings:
            self.rib(role,[point(radius*math.cos(a),radius*math.sin(a))
                          for a in np.linspace(0,2*math.pi,48,endpoint=False)],thick,True)
        petals=12 if self.variant=="rayonnant" else 10 if self.variant=="high-gothic" else 8
        for a in np.linspace(0,2*math.pi,petals,endpoint=False):
            radial=np.array([math.cos(a),math.sin(a)]); tangent=np.array([-math.sin(a),math.cos(a)])
            # Mullions meet the outer roll so coursed rings stay envelope-connected.
            self.rib(role,[point(*(radial*r*.22)),point(*(radial*r))],self.t*.14)
            if self.detail!="concept":
                # Foils sit between the spokes, never bisected by them.
                b=a+math.pi/petals
                radial=np.array([math.cos(b),math.sin(b)]); tangent=np.array([-math.sin(b),math.cos(b)])
                foil=cusped_loop(r*.115,r*.125,lobes=3,segments=24)
                self.rib(role,[point(*(radial*(r*.795+u)+tangent*v)) for u,v in foil],
                         self.t*.095,True)
        if self.detail!="concept":
            self.foil(role,x,y,z,r*.18,angle=angle,radius=self.t*.08)

    def rose_wall(self,x,bottom,z,width,height,angle=0):
        r=width*.35; cy=height*.57
        register_height=cy-r
        pierced_register=bool(angle)
        # Four curved spandrels leave a true circular aperture through the wall.
        for i in range(4):
            a0=i*math.pi/2; a1=(i+1)*math.pi/2
            arc=np.array([[r*math.cos(a),cy+r*math.sin(a)] for a in np.linspace(a0,a1,18)])
            register_bottom=register_height if pierced_register else 0
            corners=[[(width/2,cy),(width/2,height),(0,height)],
                     [(0,height),(-width/2,height),(-width/2,cy)],
                     [(-width/2,cy),(-width/2,register_bottom),(0,register_bottom)],
                     [(0,register_bottom),(width/2,register_bottom),(width/2,cy)]][i]
            if pierced_register and i==2:
                corners=corners[:-1]
            elif pierced_register and i==3:
                corners=corners[1:]
            self.plate("rose_surround",np.vstack((arc[::-1],corners)),self.t,x,bottom,z,angle)
        circle=[[r*math.cos(a),cy+r*math.sin(a)] for a in np.linspace(0,2*math.pi,64,endpoint=False)]
        self.profile("glazing",circle,self.t*.035,x,bottom,z,angle)
        normal=-1 if angle==0 or x<0 else 1
        self.rose(x+math.sin(angle)*self.t*.6*normal,bottom+cy,z+math.cos(angle)*self.t*.6*normal,r,angle)
        for j in range(3 if self.detail!="concept" else 1):
            r_min, r_max = r*(1.01+j*.036), r*(1.048+j*.036)
            thick=self.t*.22
            offset=self.t*(.45+j*.23)*normal
            cx=x+math.sin(angle)*offset; cz=z+math.cos(angle)*offset
            # Chord ≤ course so a face-mounted roll is ashlars, not one 7 m ring.
            # The ring's own width adds to each facet's height near the sides.
            module=min(MAX_COURSE, MAX_STRETCHER)-(r_max-r_min)-.02
            facets=max(8, int(math.ceil(math.tau * r_max / module - 1e-9)))
            host=f"cathedral.rose_surround.{self.counts['rose_surround']:04d}"
            for i in range(facets):
                a0=math.tau*i/facets; a1=math.tau*(i+1)/facets
                samples=max(2, int(round((a1-a0)/(math.pi/12))))
                angs=np.linspace(a0, a1, samples+1)
                poly=[(r_max*math.cos(t), r_max*math.sin(t)) for t in angs]
                poly+=[(r_min*math.cos(t), r_min*math.sin(t)) for t in reversed(angs)]
                mesh=extrude_profile(poly, thick)
                if angle:
                    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle,[0,1,0]))
                mesh.apply_translation([cx, bottom+cy, cz])
                self.mesh("rose_surround", mesh, construction="ashlar running bond",
                          unit="radial_ashlar", course=j, facet=i, host_id=host)
        if pierced_register:
            # The lower transept register is cut through the wall, with two
            # paired lights and continuous masonry at their sill and lintel.
            window_bottom=bottom+self.t*.72
            window_height=register_height-self.t*.76
            for side in (-1,1):
                offset=side*width*.25
                self.window("transept",x+offset*math.cos(angle),window_bottom,
                            z-offset*math.sin(angle),width*.43,window_height,self.t*.85,angle,lights=2,
                            register="transept_lower")
            for offset,jamb_width in ((-width*.477,width*.046),(0,width*.07),(width*.477,width*.046)):
                self.plate("wall_spandrel",[(offset-jamb_width/2,0),(offset+jamb_width/2,0),
                             (offset+jamb_width/2,register_height),(offset-jamb_width/2,register_height)],
                             self.t,x,bottom,z,angle)
            self.plate("wall_spandrel",[(-width/2,0),(width/2,0),
                         (width/2,self.t*.74),(-width/2,self.t*.74)],self.t,x,bottom,z,angle)
        for offset,scale in ((0,1.0),(register_height,.80),(height,1.20)):
            self.course("facade_mullion",x,bottom+offset-self.t*.08,z,width*1.04,
                        self.t*.43*scale,self.t*(1.6 if offset==height else 1.35),angle,normal)
        # Recessed panel borders give the facade a legible stone frame.
        for side in (-1,1):
            self.plate("facade_mullion",[(side*width*.46-self.t*.11,0),(side*width*.46-self.t*.11,height),
                         (side*width*.46+self.t*.11,height),(side*width*.46+self.t*.11,0)],self.t*1.3,
                         x,bottom,z,angle)

    def portal(self,x,z,width,height,role="west_portal",angle=0,wall_face=0.0,gable_top=None):
        """``wall_face``: outward offset of the host wall's face from ``x, z``."""
        layers=3
        normal=-1 if angle==0 or x<0 else 1
        opening_id=f"{role}:{_r(x)}:{_r(z)}:{_r(angle)}"
        for i in range(layers):
            width_i=width*(1-i*.050);height_i=height-i*self.t*.34
            offset=(layers-1-i)*self.t*.28*normal
            px=x+math.sin(angle)*offset;pz=z+math.cos(angle)*offset
            substantial=i%2==0
            section=self.t*(.32 if substantial else .15)
            member_depth=self.t*(.93 if substantial else .50)
            self.arch(role,px,self.b,pz,width_i,height_i,member_depth,section,angle,
                      section="chamfered_roll" if substantial else "fillet",
                      opening_id=opening_id)
        if self.detail!="concept":
            rise=min(height_i*.50,width_i*.79);spring=height_i-rise
            panel=pointed_arc(width_i/2-self.t*.25,rise,self.segments)+[0,spring]
            inner=self.pierced_tympanum(panel,[0,spring+rise*.40],width_i*.30,rise*.38,
                                       self.t*.42,x,self.b,z,angle)
            offset=self.t*.25*normal
            self.rib("window_tracery",[[x+u*math.cos(angle)+math.sin(angle)*offset,self.b+v,
                      z-u*math.sin(angle)+math.cos(angle)*offset] for u,v in inner],self.t*.105,True)
            # A blind, recessed trefoil: tympana are carved stone, not glass.
            back=-self.t*.12*normal
            self.plate("portal_gable",inner,self.t*.3,x+math.sin(angle)*back,self.b,
                       z+math.cos(angle)*back,angle,register="tympanum_field")
        # Two rakes; a chevron double-backs when Y-clipped and would seal the portal.
        normal=-1 if angle==0 or x<0 else 1
        # Rakes bed into the wall face rather than standing proud of it.
        front=(layers-1)*self.t*.28+self.t*.465
        gable_depth=front-wall_face+self.t*.1
        seat=(wall_face-self.t*.1+gable_depth/2)*normal
        gx=x+math.sin(angle)*seat
        gz=z+math.cos(angle)*seat
        # Bold rakes (a wimperg) bedded on the wall, crowned by a finial.
        outer_w, inner_w = width*.70, width*.60
        finial=self.t*1.1 if self.detail=="presentation" else 0.0
        y0=height*.61
        y_outer=height*1.19 if gable_top is None else min(height*1.19,gable_top-self.b-finial)
        y_inner=y_outer-height*.12
        self.plate("portal_gable", [(-outer_w, y0), (0, y_outer), (0, y_inner), (-inner_w, y0)],
                   gable_depth, gx, self.b, gz, angle)
        self.plate("portal_gable", [(0, y_outer), (outer_w, y0), (inner_w, y0), (0, y_inner)],
                   gable_depth, gx, self.b, gz, angle)
        if self.detail=="presentation":
            self.profiled_prism("finial",gx,self.b+y_outer-self.t*.1,gz,self.t*.42,self.t*.42,self.t*1.1,
                                ((0,.6),(.18,1),(.42,.55),(1,.1)),round_section=True)
            # Each rake lands on a small pinnacle stop rather than in mid-air.
            for sign in (-1,1):
                u=sign*(outer_w+inner_w)/2
                self.profiled_prism("finial",gx+u*math.cos(angle),self.b+y0-self.t*.55,gz-u*math.sin(angle),
                                    self.t*.5,self.t*.5,self.t*1.6,
                                    ((0,.8),(.3,1),(.5,.7),(1,.12)),round_section=True)

    def tower(self,x,z):
        tw=self.w*.184; td=self.w*.193
        # Early English towers still clear the nave ridge before their caps.
        top=self.h*(.69 if self.variant=="early-english" else 1-self.spire_ratio)
        bottom=self.b
        elevations=[bottom,self.h*.25,self.h*.45]
        if self.variant!="early-english": elevations.append(top-self.h*.13)
        elevations.append(top)
        for level,(lo,hi) in enumerate(zip(elevations,elevations[1:])):
            for dx in (-1,1):
                for dz in (-1,1):
                    self.stack_member("tower_shaft",x+dx*tw*.44,lo,z+dz*td*.44,
                             self.t*.95,self.t*.95,hi-lo)
            # Windows stop under the frieze, so their pointed heads stay whole.
            head=hi-(self.h*.037+self.t*.2 if self.detail!="concept" else self.t*.9)
            belfry=level>=2
            lights=(3 if self.variant=="rayonnant" else 2) if belfry else 2 if level else 1
            for side in (-1,1):
                if level==0 and side==-1:
                    # The tower's entrance remains a doorway, set in solid wall.
                    self.portal_infill(x,z-td*.5,tw*.72,self.h*.145,tw*.5,self.h*.184,depth=self.t*.8)
                    self.ashlar("wall_spandrel",(x,(self.h*.184+hi)/2,z-td*.5),
                                (tw*.79,hi-self.h*.184,self.t*.8))
                elif level==0:
                    # At ground level the inner face opens into the aisle.
                    self.window("west_tower",x,lo+self.t,z+side*td*.5,tw*.79,head-lo-self.t,self.t*.8,
                                lights=1,glazed=False,tower_x=x)
                    self.ashlar("wall_spandrel",(x,(head+hi)/2,z+side*td*.5),(tw*.79,hi-head,self.t*.8))
                else:
                    self.tower_face("west_tower",x,z+side*td*.5,0,tw*.79,lo,head,hi,
                                    .64 if belfry else .52,.06 if belfry else .16,lights,tower_x=x)
                # Lower stages are mostly wall; only the belfry opens wide.
                self.tower_face("tower_lancet",x+side*tw*.5,z,math.pi/2,td*.79,lo,head,hi,
                                .64 if belfry else .52 if level else .30,
                                .06 if belfry else .16 if level else .42,lights)
            if self.detail!="concept":
                # Small blind arcades below each cornice establish a second
                # scale of detail against the tall principal lancets.
                self.ashlar("tower_cornice",(x,hi-self.h*.037,z),
                         (tw*1.03,self.t*.25,td*1.03))
                for angle in (0,math.pi/2):
                    for side in (-1,1):
                        for j in range(3):
                            u=(j-1)*tw*.27
                            self.arch("facade_gallery",x+(side*tw*.515 if angle else u),
                                      hi-self.h*.037,z+(-u if angle else side*td*.515),
                                      tw*.24,self.h*.033,self.t*.30,self.t*.105,angle)
            for j in range(3):
                self.ashlar("tower_cornice",(x,hi-self.t*.12+j*self.t*.16,z),
                         (tw+self.t*(.2+j*.13),self.t*.15,td+self.t*(.2+j*.13)))
        spire_height=self.h*(.075 if self.variant=="early-english" else self.spire_ratio)-self.t
        crown_height=min(self.h*.046,spire_height*.22)
        # One base cornice anchors the gabled shoulder; its upper courses are
        # receding pointed face panels instead of shelves beneath a pyramid.
        self.profiled_prism("tower_cornice",x,top,z,tw*1.09,td*1.09,self.t*.50)
        for dx in (-1,1):
            for dz in (-1,1):
                self.molded_shaft("tower_shaft",x+dx*tw*.43,top,z+dz*td*.43,
                                  crown_height+self.t*.35,self.t*.30)
        # A short Early English cap is a plain octagon; gabled shoulders only
        # read on a tall spire and splay into a starfish on a low one.
        capped=self.variant=="early-english"
        # The plain cap sits directly on its base cornice.
        cap_base=top+self.t*.45 if capped else top+crown_height
        self.spire(x,cap_base,z,tw*.80,spire_height-(cap_base-top),
                   shoulder=None if capped else
                   {"bottom":top+self.t*.09,"eave":top+crown_height*.88,"width":tw,"depth":td})
        for dx in (-1,1):
            for dz in (-1,1):
                self.spire(x+dx*tw*.45,top+crown_height*.43,z+dz*td*.45,
                           self.t*1.47,self.h*.096,"pinnacle")
        for dx,dz in (() if capped else ((-1,0),(1,0),(0,-1),(0,1))):
            self.molded_shaft("tower_shaft",x+dx*tw*.32,top,z+dz*td*.32,
                              crown_height*.90,self.t*.19)
            self.spire(x+dx*tw*.32,top+crown_height*.90,z+dz*td*.32,
                       self.t*.82,self.h*.055,"pinnacle")
        self.portal(x,z-td*.5-self.t*.7,tw*.72,self.h*.145,wall_face=-self.t*.3)

    def tower_face(self,role,x,z,angle,face,lo,head,hi,frac,sill,lights,**params):
        """One tower face: a centered lancet bay set in solid coursed wall."""
        bottom=lo+self.t+(head-lo-self.t)*sill
        width=face*frac
        self.window(role,x,bottom,z,width,head-bottom,self.t*.8,angle,lights=lights,**params)
        half=face/2;inner=width/2
        panels=[(-half,-inner,lo,hi),(inner,half,lo,hi),(-inner,inner,lo,bottom),(-inner,inner,head,hi)]
        for u0,u1,y0,y1 in panels:
            if u1-u0>MIN_STONE and y1-y0>MIN_STONE:
                self.plate("wall_spandrel",[(u0,y0),(u1,y0),(u1,y1),(u0,y1)],self.t*.8,x,0,z,angle)

    def west(self):
        z=self.front; top=self.h*.44
        for x in (-self.w*.253,self.w*.253):self.tower(x,z+self.w*.097)
        self.portal(0,z-self.t,self.n*1.23,self.h*.182,wall_face=-self.t*.5)
        # The central portal has open flanks tied into the tower plinths.
        portal_top=self.h*.19
        self.portal_infill(0,z,self.n*1.23,self.h*.182,self.n,portal_top)
        self.rose_wall(0,portal_top,z,self.n*2,top-portal_top)
        for i in range(9):
            self.arch("facade_gallery",(i-4)*self.n*.215,top-self.t*.5,z-self.t*.7,
                      self.n*.20,self.h*.048,self.t*.5,self.t*.12)
        gable_bottom=top+self.h*.052
        # The gallery is a blind arcade: a wall behind it closes the nave end.
        self.ashlar("wall_spandrel",(0,(top+gable_bottom)/2,z+self.t*.1),
                    (self.n*2,gable_bottom-top+self.t*.2,self.t*.8))
        # A screen gable rises above the ridge, at least as steep as the roof,
        # so it reads as Gothic rather than as a low classical pediment.
        apex=self.ridge_h+self.h*.04
        self.plate("facade_gable",[(-self.n-self.t,0),(0,apex-gable_bottom),(self.n+self.t,0)],self.t,0,gable_bottom,z)
        self.rib("ridge",[[-self.n-self.t,gable_bottom,z-self.t*.6],[0,apex,z-self.t*.6],
                         [self.n+self.t,gable_bottom,z-self.t*.6]],self.t*.18)
        if self.detail!="concept":self.rose(0,gable_bottom+(apex-gable_bottom)*.33,z-self.t*.6,self.n*.19)

    def transept(self):
        span=self.w*.49
        for sign in (-1,1):
            x=span*sign
            # The doorway is sized so its full wimperg clears the lights above.
            finial=self.t*1.1 if self.detail=="presentation" else 0.0
            door=min(self.h*.17,(self.h*.19+self.t*.5-self.b-finial)/1.19)
            self.portal(x,self.cross,self.cross_width*.59,door,"west_portal",math.pi/2,
                        wall_face=self.t*.5)
            self.rose_wall(x,self.h*.19,self.cross,self.cross_width,self.h*.245,math.pi/2)
            # Masonry flanks join the entrance, rose registers and outer piers.
            self.portal_infill(x,self.cross,self.cross_width*.59,door,
                               self.cross_width/2,self.h*.19,math.pi/2)
            for z in (self.cross-self.cross_width/2,self.cross+self.cross_width/2):
                self.ashlar("transept",(x,(self.b+self.eave)/2,z),
                            (self.t*1.18*1.34,self.eave-self.b,self.t*1.65*1.34))
                self.transept_side(sign,z,span)
            # The gable stands a coping's height proud of the roof slopes.
            proud=self.t*.7
            self.plate("facade_gable",[(-self.cross_width/2,0),(0,self.ridge_h-self.eave+proud),
                         (self.cross_width/2,0)],self.t,x,self.eave,self.cross,math.pi/2)
            self.rib("ridge",[[x+sign*self.t*.62,self.eave,self.cross-self.cross_width/2],
                     [x+sign*self.t*.62,self.ridge_h+proud,self.cross],
                     [x+sign*self.t*.62,self.eave,self.cross+self.cross_width/2]],self.t*.17)
            if self.detail!="concept":
                for j in (-1,0,1):
                    self.arch("facade_gallery",x+sign*self.t*.5,self.eave+self.t*.2,
                              self.cross+j*self.cross_width*.23,self.cross_width*.19,
                              self.h*(.072 if j==0 else .047),self.t*.3,self.t*.16,math.pi/2)
        rise = self.ridge_h - self.eave
        thick = self.t * .3
        left = [(-self.cross_width/2, 0), (0, rise), (0, rise-thick), (-self.cross_width/2, -thick)]
        right = [(0, rise), (self.cross_width/2, 0), (self.cross_width/2, -thick), (0, rise-thick)]
        self.tiles("transept_roof", left, span*2, y=self.eave, z=self.cross, angle=math.pi/2)
        self.tiles("transept_roof", right, span*2, y=self.eave, z=self.cross, angle=math.pi/2)
        # A slim crossing lantern gives a third, secondary skyline accent.
        # Early English carries its dominant spire on a broader tower.
        width=self.n*(1.05 if self.variant=="early-english" else .72)
        hi=self.h*(.73 if self.variant=="early-english" else .70)
        # A solid tambour seats the lantern below the lowest roof valley.
        valley=self.ridge_h-(width/2)*(self.ridge_h-self.eave)/(self.cross_width/2)
        lo=self.ridge_h+self.t*.5
        self.ashlar("crossing_lantern",(0,(valley-self.t*.3+lo)/2,self.cross),
                    (width,lo-valley+self.t*.3,width))
        self.ashlar("tower_cornice",(0,lo-self.t*.15,self.cross),(width+self.t*.4,self.t*.3,width+self.t*.4))
        for angle in (0,math.pi/2):
            for sign in (-1,1):
                fx,fz=(sign*width*.5 if angle else 0),self.cross+(0 if angle else sign*width*.5)
                if self.variant=="early-english":
                    # A broad crossing tower: paired lights set in masonry.
                    self.tower_face("crossing_lantern",fx,fz,angle,width,lo,hi-self.t*.3,hi,
                                    .62,.0,2)
                else:
                    self.window("crossing_lantern",fx,lo,fz,width,hi-lo,self.t*.55,angle)
        self.ashlar("tower_cornice",(0,hi,self.cross),(width*1.14,self.t*.35,width*1.14))
        self.spire(0,hi,self.cross,width*(1.0 if self.variant=="early-english" else 1.1),
                   self.h*(.27 if self.variant=="early-english" else .14))

    def transept_side(self,sign,z,span):
        """An arm's flank: aisle-height lights outside the aisle, a solid
        triforium band, and clerestory lights on the nave's bay rhythm."""
        pitch=(self.cross-self.cross_width/2-self.front-self.w*.20)/self.bays
        face=span-self.t*1.18*.67
        sill=self.b+self.h*.036; low_top=sill+self.aisle_h*.77
        upper=self.h*.285

        def bays(u0,u1,bottom,height,lights):
            count=max(1,int(round((u1-u0)/pitch)))
            bay=(u1-u0)/count
            for k in range(count):
                u=u0+bay*(k+.5)
                self.window("transept",sign*u,bottom,z,bay*.86,height,self.t,lights=lights)
            for k in range(count+1):
                u=u0+bay*k
                lo,hi=(u-bay*.07,u+bay*.07)
                lo,hi=max(lo,u0),min(hi,u1)
                if hi-lo>MIN_STONE:
                    self.ashlar("wall_spandrel",(sign*(lo+hi)/2,bottom+height/2,z),(hi-lo,height,self.t))
        # Below the aisle roof the arm is interior; only its outer reach is lit.
        bays(self.outer,face,sill,low_top-sill,1 if self.variant=="early-english" else 2)
        self.ashlar("wall_spandrel",(sign*(self.n+self.outer)/2,(self.b+low_top)/2,z),
                    (self.outer-self.n,low_top-self.b,self.t))
        self.ashlar("aisle_dado",(sign*(self.outer+face)/2,(self.b+sill)/2,z),
                    (face-self.outer,sill-self.b,self.t*.8))
        self.ashlar("wall_spandrel",(sign*(self.n+face)/2,(low_top+upper)/2,z),
                    (face-self.n,upper-low_top,self.t))
        bays(self.n,face,upper,self.eave-upper-self.h*.02,3 if self.variant=="rayonnant" else 2)
        self.ashlar("wall_spandrel",(sign*(self.n+face)/2,self.eave-self.h*.01,z),
                    (face-self.n,self.h*.02,self.t))
        self.course("facade_mullion",sign*(self.n+face)/2,self.eave-self.t*.08,z,face-self.n,
                    self.t*.38,self.t*1.3,normal=1 if z>self.cross else -1)
        self.course("facade_mullion",sign*(self.n+face)/2,upper-self.t*.08,z,face-self.n,
                    self.t*.40,self.t*1.3,normal=1 if z>self.cross else -1)

    def apse(self):
        radius=self.outer
        angles=np.linspace(0,math.pi,self.apse_sides+1)
        window_bottom=self.b+self.h*.04
        window_top=self.aisle_h-self.t*.22
        roof_edge=self.aisle_h+self.t*.52
        for a0,a1 in zip(angles,angles[1:]):
            mid=(a0+a1)/2
            r=radius*math.cos((a1-a0)/2);width=2*radius*math.sin((a1-a0)/2)
            # Local facade horizontal vector is tangent to the apse.
            angle=math.pi/2-mid
            x=r*math.cos(mid);z=self.back+r*math.sin(mid)
            self.window("apse_window",x,window_bottom,z,width*.80,window_top-window_bottom,self.t*.86,angle,
                        register="apse_facet")
            self.plate("wall_spandrel",[(-width/2,self.b),(width/2,self.b),
                         (width/2,window_bottom),(-width/2,window_bottom)],self.t,x,0,z,angle)
            for side in (-1,1):
                lo,hi=sorted((side*width*.40,side*width*.5))
                self.plate("wall_spandrel",[(lo,window_bottom),(hi,window_bottom),
                             (hi,self.aisle_h),(lo,self.aisle_h)],self.t,x,0,z,angle)
            self.plate("wall_spandrel",[(-width/2,window_top),(width/2,window_top),
                         (width/2,self.aisle_h+self.t*.04),(-width/2,self.aisle_h+self.t*.04)],
                         self.t,x,0,z,angle)
            for bottom,height,projection in ((self.b,self.t*.38,self.t*1.3),
                    (window_bottom-self.t*.23,self.t*.29,self.t*1.20),
                    (self.aisle_h,self.t*.55,self.t*1.62)):
                self.course("facade_mullion",x,bottom,z,width+self.t*.24,height,projection,angle)
        # Radiating buttresses stand just proud of each interior apse corner
        # and fly to the hemicycle, matching the straight choir bays. The end
        # corners are left to the last choir buttresses beside them.
        pier_top=self.h*.29
        for a in angles[1:-1]:
            reach=radius+self.t*1.0
            x=reach*math.cos(a);z=self.back+reach*math.sin(a)
            self.pier(x,z,self.b,pier_top-self.b,"apse_pier",self.t*.9)
            self.profiled_prism("buttress_cap",x,pier_top,z,self.t*2.2,self.t*2.2,self.t*.6,
                                ((0,.75),(.4,1),(1,.85)))
            self.spire(x,pier_top+self.t*.6,z,self.t*1.5,self.h*.092,"pinnacle")
            self.radial_flyer(a,reach,pier_top)
        # A chevet: the low ring is the ambulatory under a lean-to, and an
        # upper hemicycle on the nave wall line carries the clerestory round
        # the east end under a half-cone that continues the nave roof.
        amb_in=self.aisle_h+self.h*.026
        upper_sill=self.h*.285
        head=self.eave-self.h*.02
        lights=3 if self.variant=="rayonnant" else 2
        for i,(a0,a1) in enumerate(zip(angles,angles[1:])):
            mid=(a0+a1)/2;half=(a1-a0)/2
            self.slate_facet("apse_roof",
                [self.n*math.cos(a0),amb_in,self.back+self.n*math.sin(a0)],
                [self.n*math.cos(a1),amb_in,self.back+self.n*math.sin(a1)],
                [radius*math.cos(a0),roof_edge,self.back+radius*math.sin(a0)],
                [radius*math.cos(a1),roof_edge,self.back+radius*math.sin(a1)],self.t*.3,i)
            width=2*self.n*math.sin(half);r=self.n*math.cos(half)
            angle=math.pi/2-mid
            x=r*math.cos(mid);z=self.back+r*math.sin(mid)
            self.plate("wall_spandrel",[(-width/2,self.aisle_h),(width/2,self.aisle_h),
                         (width/2,upper_sill-self.t),(-width/2,upper_sill-self.t)],self.t*.8,x,0,z,angle)
            self.tower_face("apse_window",x,z,angle,width,upper_sill-self.t,head,self.eave,.86,0,lights,
                            register="hemicycle")
            self.course("facade_mullion",x,self.eave-self.t*.08,z,width+self.t*.24,self.t*.38,self.t*1.3,angle)
            self.slate_facet("apse_roof",[0,self.ridge_h,self.back],[0,self.ridge_h,self.back],
                [(self.n+self.t)*math.cos(a0),self.eave,self.back+(self.n+self.t)*math.sin(a0)],
                [(self.n+self.t)*math.cos(a1),self.eave,self.back+(self.n+self.t)*math.sin(a1)],
                self.t*.35,i)
        for a in angles[1:-1]:
            self.stack_member("apse_pier",self.n*math.cos(a),self.aisle_h,self.back+self.n*math.sin(a),
                              self.t*.9,self.t*.9,self.eave-self.aisle_h,"CYLINDER",radius=self.t*.45,
                              sections=8,register="hemicycle_shaft")
            # Explicit, shallow hip rolls survive hidden-line feature thresholds.
            reach=self.n+self.t
            self.rib("roof_seam",[[reach*.03*math.cos(a),self.ridge_h-(self.ridge_h-self.eave)*.03+self.t*.025,
                                   self.back+reach*.03*math.sin(a)],
                                  [reach*math.cos(a),self.eave+self.t*.025,self.back+reach*math.sin(a)]],
                     self.t*.065,register="apse_hip")
        self.rib("roof_seam",[[radius*math.cos(a),roof_edge+self.t*.03,self.back+radius*math.sin(a)]
                              for a in angles],self.t*.075,register="apse_eave")
        # The hemicycle vault: half-ribs from each pier to one boss.
        spring=self.b+self.h*.158;crown=spring+self.h*.245
        for a in angles:
            self.rib("vault_rib",[[self.n*(1-u)*math.cos(a),spring+(crown-spring)*math.sin(math.pi/2*u)**.78,
                                   self.back+self.n*(1-u)*math.sin(a)]
                                  for u in np.linspace(0,.97,self.segments+1)],self.t*.19,register="hemicycle")

    def radial_flyer(self,a,reach,pier_top):
        """One flying buttress from a chevet pier to the hemicycle wall."""
        us=np.linspace(0,1,self.segments+1)
        start=np.array([self.n+self.t*.45,self.eave-self.h*.03])
        end=np.array([reach-self.t*.8,pier_top+self.t*.9])
        thickness=self.t*.56+self.h*.0015
        end_depth=self.h*.025+self.t*.35
        top=start+(end-start)*us[:,None]
        lower=top.copy();lower[:,1]-=end_depth-(end_depth-thickness)*np.sin(us*math.pi)
        side=1 if math.cos(a)>=0 else -1
        # Stones are cut once for the canonical side and reused mirrored, so
        # the pair matches exactly instead of differing by float noise.
        poly=np.vstack((top,lower[::-1]));depth=self.t*.61
        canonical=-(a if side>0 else math.pi-a)
        host=f"cathedral.flying_buttress.{self.counts['flying_buttress']:04d}"
        for piece,course in iter_fitted_profile_pieces(poly,depth,canonical):
            mesh=extrude_profile(piece,depth)
            mesh.apply_transform(trimesh.transformations.rotation_matrix(canonical,[0,1,0]))
            mesh.apply_translation([0,0,self.back])
            if side<0:
                # An exact reflection of the canonical stone, not a re-rotation.
                mesh.vertices[:,0]*=-1;mesh.invert()
            self.mesh("flying_buttress",mesh,side=side,register="chevet",
                      construction="ashlar running bond",unit="profile_course",course=course,
                      host_id=host)
        coping=top.copy();coping[:,1]+=self.t*.24
        self.plate("buttress_cap",np.vstack((coping,top[::-1])),self.t*.88,0,0,self.back,-a,
                   register="flying_coping")

    def slate_facet(self,role,in_a,in_b,out_a,out_b,thick,facet):
        """Course one roof facet in eave-parallel slates, alternate courses
        broken half a stone, each stone a convex cubit-scale hull."""
        host=f"cathedral.{role}.{self.counts[role]:04d}"
        top=[np.asarray(p,dtype=float) for p in (in_a,in_b,out_a,out_b)]
        low=[p-[0,thick,0] for p in top]

        def at(surface,f,s):
            inner=surface[0]+(surface[1]-surface[0])*s
            outer=surface[2]+(surface[3]-surface[2])*s
            return inner+(outer-inner)*f
        rise=abs(float(top[0][1]-top[2][1]))
        run=float(np.linalg.norm(((top[2]+top[3])/2-(top[0]+top[1])/2)[[0,2]]))
        # Course rise plus slate thickness must stay within one course height,
        # and a shallow facet still needs courses no deeper than a stone.
        rings=max(1,int(math.ceil(rise/(MAX_COURSE*.95-thick))),
                  int(math.ceil(run/(MAX_STRETCHER*.6))))
        for ring in range(rings):
            f0,f1=ring/rings,(ring+1)/rings
            chord=float(np.linalg.norm(at(top,f1,1)-at(top,f1,0)))
            count=max(1,int(math.ceil(chord/(MAX_STRETCHER*.6))))
            while True:
                cuts=sorted({0.0,1.0,*[min(1.0,max(0.0,(k+.5*(ring%2))/count)) for k in range(1,count+1)]})
                stones=[]
                for s0,s1 in zip(cuts,cuts[1:]):
                    if s1-s0<1e-6: continue
                    pts=[at(surface,f,s) for surface in (top,low) for f in (f0,f1) for s in (s0,s1)]
                    stones.append(trimesh.Trimesh(vertices=np.unique(np.round(pts,9),axis=0)).convex_hull)
                if all(_cubit_aabb(m) for m in stones) or count>64: break
                count+=1
            for k,mesh in enumerate(stones):
                self.mesh(role,mesh,facet=facet,construction="ashlar running bond",
                          unit="profile_course",course=ring,stone=k,host_id=host)

    def compose(self):
        self.foundation();self.longitudinal();self.west();self.transept();self.apse()
        return BuildDocument(ARCHITECT,self.request,self.placements,metadata={
            "cathedral_rules": {"variant":self.variant,"nave_bays":self.bays,"choir_bays":self.choir_bays,
                "apse_sides":self.apse_sides,"spire_ratio":self.spire_ratio,
                "plan":"cruciform nave, flanking aisles, transept, chevet with ambulatory and clerestoried hemicycle",
                "openings":"pierced surrounds with separate removable glazing; portals remain open",
                "vaults":"transverse pointed ribs and diagonal ribs at every nave and choir bay",
                "scope":"visual architecture; geometric checks are not structural engineering"},
            "invariants":{"required_roles":sorted(REQUIRED_ROLES)}})


def build_document(request: BuildRequest):
    return Cathedral(request).compose()


def random_controls(rng=None):
    rng=rng or random
    w=rng.randint(26,42)
    return {"width":w,"height":round(w*rng.uniform(1.4,1.7)),"depth":round(w*rng.uniform(1.9,2.3)),
            "bays":rng.randint(5,9),"detail":rng.choice(("concept","standard","presentation")),"levels":None}
