"""A closed aqueduct-and-spline canal. A marble rides it forever.

The course is a seeded loop of human-scale channel blocks on piers, with
arches on the straighter spans. The marble path is a closed Catmull-Rom
spline resampled by arc length. Animation wraps; the run does not end.
This is a visual architecture loop, not a physics integrator.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

import numpy as np
import trimesh

from core.architectural_geometry import extrude_profile, mesh_parameters, tube_path
from .masonry import MAX_COURSE, MAX_STRETCHER, MIN_STONE, fill_box, iter_stack
from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement


ARCHITECT = ArchitectInfo("marble", "1.0.0", "Marble Run Architect")
PRODUCT = "marble-run"
DEFAULTS = (36, 10, 28)
DETAIL_LEVELS = ("concept", "standard", "presentation")
REQUIRED_ROLES = {"foundation", "pier", "arch", "channel_bed", "channel_wall", "marble"}
KINDS = {
    "foundation": "foundation",
    "pier": "pier",
    "arch": "arch",
    "channel_bed": "slab",
    "channel_wall": "wall",
    "guide_spline": "circulation",
    "marble": "circulation",
}
# An aqueduct tradition in the round: limestone piers, arcades and channel
# walls over a concrete waterway, iron rails for the riding spline, and a
# marble sphere for the marble.
SUBSTANCES = {
    "foundation": "limestone",
    "pier": "limestone",
    "arch": "limestone",
    "channel_bed": "roman-concrete",
    "channel_wall": "limestone",
    "guide_spline": "wrought-iron",
    "marble": "marble",
}
DEFAULT_SUBSTANCE = "limestone"


def list_products():
    return [{
        "name": PRODUCT, "category": "assembly", "architect": ARCHITECT.to_dict(),
        "architect_id": ARCHITECT.id, "architect_title": ARCHITECT.title,
        "architect_version": ARCHITECT.version,
        "description": "Closed aqueduct-and-spline canal with a looping marble ride through procedural masonry",
        "defaults": dict(zip(("width", "height", "depth"), DEFAULTS), variant="auto",
                         detail="standard", bays=8, levels=2),
        "variants": ["auto"], "detail_levels": list(DETAIL_LEVELS),
        "opening_modes": ["auto"], "parameters": [],
        "substances": sorted(set(SUBSTANCES.values()) | {DEFAULT_SUBSTANCE}),
        "rules": {"required_roles": sorted(REQUIRED_ROLES), "bays": [4, 12], "levels": [1, 3],
                  "scope": "looping visual architecture; not a physics simulation"},
    }]


def catmull_rom(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def closed_spline(controls, samples_per_span=8):
    points = np.asarray(controls, dtype=float)
    if len(points) < 4:
        raise ValueError("a marble-run loop needs at least four control points")
    samples_per_span = max(4, int(samples_per_span))
    n = len(points)
    rows = []
    for i in range(n):
        p0, p1, p2, p3 = (points[(i + k) % n] for k in (-1, 0, 1, 2))
        for t in np.linspace(0.0, 1.0, samples_per_span, endpoint=False):
            rows.append(catmull_rom(p0, p1, p2, p3, t))
    return np.asarray(rows, dtype=float)


def resample_closed(points, spacing=0.7):
    """Return arc-length samples of a closed polyline and its length."""
    closed = np.vstack((points, points[:1]))
    segments = np.diff(closed, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("marble-run path has a degenerate segment")
    total = float(np.sum(lengths))
    count = max(24, int(round(total / max(spacing, 0.25))))
    distances = np.linspace(0.0, total, count, endpoint=False)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    samples = []
    for distance in distances:
        index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        index = min(max(index, 0), len(points) - 1)
        span = lengths[index]
        t = 0.0 if span <= 1e-12 else (distance - cumulative[index]) / span
        samples.append(points[index] + t * (points[(index + 1) % len(points)] - points[index]))
    return np.asarray(samples, dtype=float), total


def point_on_loop(samples, length, distance):
    """Position and unit tangent on a closed resampled loop, wrapping forever."""
    if length <= 1e-12 or len(samples) < 2:
        raise ValueError("marble-run path is empty")
    wrapped = float(distance) % length
    if wrapped < 0:
        wrapped += length
    spacing = length / len(samples)
    index = int(wrapped / spacing) % len(samples)
    nxt = (index + 1) % len(samples)
    t = (wrapped - index * spacing) / spacing
    position = samples[index] * (1.0 - t) + samples[nxt] * t
    tangent = samples[nxt] - samples[index]
    norm = np.linalg.norm(tangent)
    if norm <= 1e-12:
        tangent = samples[(nxt + 1) % len(samples)] - samples[index]
        norm = np.linalg.norm(tangent)
    tangent = tangent / max(norm, 1e-12)
    return position, tangent


def _frame(tangent):
    tangent = np.asarray(tangent, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)
    up = np.array([0.0, 1.0, 0.0])
    binormal = np.cross(up, tangent)
    length = np.linalg.norm(binormal)
    if length < 1e-8:
        binormal = np.array([1.0, 0.0, 0.0])
    else:
        binormal = binormal / length
    normal = np.cross(tangent, binormal)
    normal = normal / np.linalg.norm(normal)
    frame = np.eye(4)
    frame[:3, 0] = tangent
    frame[:3, 1] = normal
    frame[:3, 2] = binormal
    return frame


def _horizontal(vector):
    """Unit XZ direction and length; None when the run is vertical."""
    vector = np.asarray(vector, dtype=float)
    horizontal = np.array([float(vector[0]), 0.0, float(vector[2])], dtype=float)
    length = float(np.linalg.norm(horizontal))
    if length < 1e-8:
        return None, 0.0
    return horizontal / length, length


def oriented_box(center, extents, tangent):
    """Box with local X along tangent, Y up, Z to the right."""
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_transform(_frame(tangent))
    mesh.apply_translation(center)
    return mesh


def oriented_ashlars(center, extents, tangent):
    """Split an oriented canal box until its AABB is a human-scale stretcher.

    Local X follows the spline. Boxes longer than ``MAX_STRETCHER`` split
    along that axis first; yawed AABBs that still exceed the cap split again.
    """
    along, height, width = (float(value) for value in extents)
    center = np.asarray(center, dtype=float)
    horiz, _span = _horizontal(tangent)
    if horiz is None:
        horiz = np.array([float(tangent[0]), 0.0, float(tangent[2])], dtype=float)
        length = float(np.linalg.norm(horiz))
        if length < 1e-8:
            return []
        horiz = horiz / length
    side = np.cross(np.array([0.0, 1.0, 0.0]), horiz)
    side_len = float(np.linalg.norm(side))
    if side_len < 1e-8:
        return []
    side = side / side_len

    up = np.array([0.0, 1.0, 0.0])

    def envelope(position, span, rise, across):
        mesh = oriented_box(position, (span, rise, across), horiz)
        dims = mesh.bounds[1] - mesh.bounds[0]
        return mesh, dims

    def too_big(dims):
        return (
            float(dims[1]) > MAX_COURSE + 1e-6
            or max(float(dims[0]), float(dims[2])) > MAX_STRETCHER + 1e-6
        )

    stack = [(center.copy(), along, height, width)]
    meshes = []
    while stack:
        position, span, rise, across = stack.pop()
        mesh, dims = envelope(position, span, rise, across)
        if not too_big(dims):
            meshes.append(mesh)
            continue
        if rise > MAX_COURSE + 1e-9 and rise > MIN_STONE * 2:
            half = rise / 2.0
            stack.append((position - up * (half / 2.0), span, half, across))
            stack.append((position + up * (half / 2.0), span, half, across))
        elif span > MAX_STRETCHER + 1e-9 or (too_big(dims) and span > MIN_STONE * 2 and span + 1e-9 >= across * 0.5):
            half = span / 2.0
            stack.append((position - horiz * (half / 2.0), half, rise, across))
            stack.append((position + horiz * (half / 2.0), half, rise, across))
        elif across > MIN_STONE * 2:
            half = across / 2.0
            stack.append((position - side * (half / 2.0), span, rise, half))
            stack.append((position + side * (half / 2.0), span, rise, half))
        else:
            meshes.append(mesh)
    return meshes


def _arch_profile(width, height, thickness, segments):
    """Jamb, bearing and half-head polygons for an open-bottom arcade panel."""
    half = width / 2.0
    thickness = min(max(float(thickness), width * 0.08), half * 0.42)
    inner = max(half - thickness, width * 0.14)
    bearing = min(max(thickness, height * 0.08), height * 0.24)
    spring = min(max(height * 0.20, 0.16), height - bearing - 0.22)
    inner_rise = max(height - spring - bearing, 0.2)
    segments = max(8, int(segments))
    n_arc = max(4, segments // 2)
    left = [(-half, spring), (-half, height - bearing), (0.0, height - bearing)]
    left.extend(
        (inner * math.cos(t), spring + inner_rise * math.sin(t))
        for t in np.linspace(math.pi / 2.0, math.pi, n_arc + 1)[1:]
    )
    right = [(half, spring), (half, height - bearing), (0.0, height - bearing)]
    right.extend(
        (inner * math.cos(t), spring + inner_rise * math.sin(t))
        for t in np.linspace(math.pi / 2.0, 0.0, n_arc + 1)[1:]
    )
    return {
        "half": half, "inner": inner, "bearing": bearing, "spring": spring,
        "left": left, "right": right, "jamb_along": half - inner,
    }


def oriented_arch(mid, tangent, width, foot_y, crown_y, depth, thickness=0.28, segments=14):
    """Upright arcade panel whose flat top carries the canal.

    Local X is the span, Y is world-up, Z is the aqueduct thickness. The
    profile is an open-bottom arch: solid spandrels, bearing under the bed,
    legs that land on the piers. `mid` is the span midpoint at bed height;
    the mesh is planted so y=foot_y is the feet and y=crown_y is the bearing.
    """
    pieces = oriented_arch_pieces(mid, tangent, width, foot_y, crown_y, depth, thickness, segments)
    if not pieces:
        raise ValueError("arcade panel produced no masonry")
    return trimesh.util.concatenate(pieces)


def oriented_arch_pieces(mid, tangent, width, foot_y, crown_y, depth, thickness=0.28, segments=14):
    """Jambs, bearing and two half-heads as cubit-scale solids. The void stays open."""
    from .cathedral import iter_profile_pieces

    height = float(crown_y) - float(foot_y)
    width = float(width)
    depth = float(depth)
    if height < 0.5 or width < 0.5:
        raise ValueError("arcade panel needs a positive span and rise")
    horiz, _span = _horizontal(tangent)
    if horiz is None:
        raise ValueError("arcade panel needs a horizontal span")
    spec = _arch_profile(width, height, thickness, segments)
    origin = np.array([float(mid[0]), float(foot_y), float(mid[2])])
    meshes = []
    if spec["jamb_along"] > MIN_STONE * 0.5 and spec["spring"] > MIN_STONE * 0.5:
        for sign in (-1.0, 1.0):
            center = origin + horiz * (sign * (spec["half"] + spec["inner"]) / 2.0)
            center = center + np.array([0.0, spec["spring"] / 2.0, 0.0])
            meshes.extend(oriented_ashlars(
                center, (spec["jamb_along"], spec["spring"], depth), horiz))
    if spec["bearing"] > MIN_STONE * 0.5:
        center = origin + np.array([0.0, height - spec["bearing"] / 2.0, 0.0])
        meshes.extend(oriented_ashlars(center, (width, spec["bearing"], depth), horiz))
    frame = _frame(horiz)
    from .cathedral import _clip_poly, _simple_rings

    def emit_ring(ring, run=None, z_off=0.0, splits_left=8):
        run = float(depth if run is None else run)
        mesh = extrude_profile(ring, run)
        if abs(z_off) > 1e-12:
            mesh.apply_translation([0.0, 0.0, z_off])
        mesh.apply_transform(frame)
        mesh.apply_translation(origin)
        dims = mesh.bounds[1] - mesh.bounds[0]
        if (float(dims[1]) <= MAX_COURSE + 1e-6
                and max(float(dims[0]), float(dims[2])) <= MAX_STRETCHER + 1e-6) or splits_left <= 0:
            meshes.append(mesh)
            return
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        if run + 1e-9 >= span_x and run + 1e-9 >= span_y and run > MIN_STONE * 2:
            half = run / 2.0
            emit_ring(ring, half, z_off - half / 2.0, splits_left - 1)
            emit_ring(ring, half, z_off + half / 2.0, splits_left - 1)
            return
        if span_x >= span_y:
            mid = (min(xs) + max(xs)) / 2.0
            parts = (_clip_poly(ring, x1=mid), _clip_poly(ring, x0=mid))
        else:
            mid = (min(ys) + max(ys)) / 2.0
            parts = (_clip_poly(ring, y1=mid), _clip_poly(ring, y0=mid))
        emitted = False
        for part in parts:
            if len(part) < 3:
                continue
            try:
                rings = _simple_rings(part)
            except ValueError:
                rings = [part]
            for simple in rings:
                if len(simple) < 3:
                    continue
                emit_ring(simple, run, z_off, splits_left - 1)
                emitted = True
        if not emitted:
            meshes.append(mesh)

    for polygon in (spec["left"], spec["right"]):
        try:
            pieces = list(iter_profile_pieces(polygon, depth))
        except ValueError:
            pieces = []
        if not pieces:
            pieces = [(polygon, 0)]
        for ring, _course in pieces:
            emit_ring(ring)
    return meshes


class _Composer:
    def __init__(self, request):
        self.request = request
        self.w, self.h, self.d = [
            float(default if value is None else value)
            for default, value in zip(DEFAULTS, (request.width, request.height, request.depth))
        ]
        if not all(math.isfinite(value) and 8 <= value <= 200 for value in (self.w, self.h, self.d)):
            raise ValueError("marble-run dimensions must be finite and between 8 and 200")
        if request.detail not in DETAIL_LEVELS:
            raise ValueError("marble-run detail must be concept, standard, or presentation")
        if request.openings != "auto":
            raise ValueError("marble-run authors its own channel; use openings=auto")
        if request.parameters:
            raise ValueError("marble-run does not accept extra parameters")
        variant = str(request.variant or "auto").strip().lower()
        if variant not in {"auto", "random"}:
            raise ValueError("marble-run uses a masonry canal, not a column order; use variant=auto")
        self.bays = 8 if request.bays is None else request.bays
        if isinstance(self.bays, bool) or not isinstance(self.bays, int) or not 4 <= self.bays <= 12:
            raise ValueError("bays must be an integer between 4 and 12")
        self.levels = 2 if request.levels is None else request.levels
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or not 1 <= self.levels <= 3:
            raise ValueError("levels must be an integer between 1 and 3")
        self.rng = np.random.RandomState(int(request.seed) & 0x7FFFFFFF)
        self.detail = request.detail
        self.placements = []
        self.counts = Counter()
        self.base = max(0.35, self.h * 0.04)
        self.channel_w = min(1.85, max(1.1, min(self.w, self.d) * 0.052))
        self.wall_h = min(0.48, max(0.3, self.h * 0.042))
        self.bed_h = min(0.26, max(0.16, self.h * 0.022))
        self.pier_w = min(1.1, max(0.52, self.channel_w * 0.64))
        self.marble_r = min(0.9, max(0.58, self.channel_w * 0.52))
        self.root = None

    def add(self, role, position, dimensions, block_type="CUBE", **params):
        pid = f"marble-run.{role}.{self.counts[role]:04d}"
        self.counts[role] += 1
        color = (0, 255, 80, 255) if role == "marble" else (222, 213, 191, 255)
        if role == "guide_spline":
            color = (0, 180, 70, 255)
        self.placements.append(Placement(
            id=pid, kind=KINDS[role], role=role, block_type=block_type,
            position=tuple(float(value) for value in position),
            dimensions=tuple(float(value) for value in dimensions),
            material=color, parent_id=self.root,
            parameters={"product": PRODUCT, **params},
            substance=SUBSTANCES.get(role, DEFAULT_SUBSTANCE),
        ))
        return pid

    def sculpt(self, role, mesh, **params):
        position, dimensions, payload = mesh_parameters(mesh)
        return self.add(role, position, dimensions, "MESH", **payload, **params)

    def course_oriented(self, role, center, extents, tangent, **params):
        """Emit an oriented canal box as human-scale stretchers along the spline."""
        host = f"marble-run.{role}.{self.counts[role]:04d}"
        last = None
        for index, mesh in enumerate(oriented_ashlars(center, extents, tangent)):
            last = self.sculpt(
                role, mesh, **{
                    "construction": "ashlar running bond", **params,
                    "course": index, "host_id": host,
                })
        return last

    def stack_pier(self, x, z, height):
        host = f"marble-run.pier.{self.counts['pier']:04d}"
        for center_y, course_h, index in iter_stack(0.0, height, self.detail):
            self.add("pier", (x, center_y, z), (self.pier_w, course_h, self.pier_w),
                     construction="ashlar stack", unit="drum", course=index, host_id=host)

    def compose(self):
        phase = float(self.rng.uniform(0, math.tau))
        rise = self.h * 0.22
        controls = []
        for i in range(self.bays):
            angle = math.tau * i / self.bays
            jitter = 0.10 * float(self.rng.uniform(-1.0, 1.0))
            radius_x = self.w * 0.36 * (1.0 + jitter)
            radius_z = self.d * 0.36 * (1.0 - jitter * 0.6)
            x = radius_x * math.cos(angle)
            z = radius_z * math.sin(angle)
            y = self.base + self.h * 0.42 + rise * math.sin(2.0 * angle + phase)
            controls.append(np.array([x, y, z], dtype=float))
        samples_per = {"concept": 6, "standard": 8, "presentation": 10}[self.detail]
        spline = closed_spline(controls, samples_per)
        spacing = {"concept": 1.05, "standard": 0.78, "presentation": 0.62}[self.detail]
        path, length = resample_closed(spline, spacing)

        # Piers follow the canal so each arcade panel can sit under the bed
        # and land on the neighboring feet. Chord midpoints fall inside the
        # loop and cannot carry the specus.
        n_piers = int(max(self.bays * 2, round(length / 6.25)))
        n_piers = min(max(n_piers, 8), 28)
        stations = []
        for i in range(n_piers):
            pos, _tan = point_on_loop(path, length, i * length / n_piers)
            stations.append(pos)

        pad_h = self.base
        for i, pos in enumerate(stations):
            host = f"marble-run.foundation.{self.counts['foundation']:04d}"
            stones = fill_box(
                (float(pos[0]), pad_h / 2.0, float(pos[2])),
                (self.pier_w * 2.4, pad_h, self.pier_w * 2.4),
                self.detail,
            )
            for stone in stones:
                pid = self.add(
                    "foundation", stone.position, stone.dimensions,
                    construction="ashlar running bond", unit="paving",
                    course=stone.course, bond="running", host_id=host,
                )
                if self.root is None:
                    self.root = pid
                    for placement in self.placements:
                        if placement.parent_id is None and placement.id != pid:
                            placement.parent_id = pid
            pier_top = max(float(pos[1]) - self.bed_h * 0.55, pad_h + 0.45)
            self.stack_pier(float(pos[0]), float(pos[2]), pier_top)

        arch_segments = 10 if self.detail == "concept" else 14
        for i, pos in enumerate(stations):
            nxt = stations[(i + 1) % n_piers]
            tangent, span = _horizontal(nxt - pos)
            if tangent is None or span < self.pier_w * 1.6:
                continue
            mid, _tan = point_on_loop(path, length, (i + 0.5) * length / n_piers)
            crown_y = float(mid[1])
            if crown_y < 1.0:
                continue
            host = f"marble-run.arch.{self.counts['arch']:04d}"
            for index, mesh in enumerate(oriented_arch_pieces(
                    mid, tangent, span + self.pier_w * 0.4,
                    0.0, crown_y + self.bed_h * 0.2,
                    max(self.pier_w * 1.3, self.channel_w),
                    thickness=min(max(self.pier_w * 0.75, 0.32), span * 0.22),
                    segments=arch_segments,
            )):
                self.sculpt(
                    "arch", mesh, construction="ashlar running bond",
                    unit="profile_course", course=index, host_id=host,
                )

        half_bed = (self.channel_w - self.wall_h * 0.2) * 0.5
        wall_t = min(0.3, self.channel_w * 0.24)
        along = spacing * 1.05
        for i, point in enumerate(path):
            tangent = path[(i + 1) % len(path)] - path[i - 1]
            if np.linalg.norm(tangent) < 1e-8:
                tangent = path[(i + 1) % len(path)] - point
            horiz, _span = _horizontal(tangent)
            if horiz is None:
                continue
            self.course_oriented(
                "channel_bed",
                (point[0], point[1] - self.bed_h / 2.0, point[2]),
                (along, self.bed_h, self.channel_w), horiz,
                unit="paving",
            )
            wall_center_y = point[1] + self.wall_h / 2.0 - self.bed_h * 0.2
            side = np.cross([0.0, 1.0, 0.0], horiz)
            side_len = np.linalg.norm(side)
            if side_len < 1e-8:
                continue
            side = side / side_len
            for sign in (-1.0, 1.0):
                center = point + side * sign * half_bed
                self.course_oriented(
                    "channel_wall",
                    (center[0], wall_center_y, center[2]),
                    (along, self.wall_h, wall_t), horiz,
                    unit="ashlar",
                )

        ride = np.array(path, dtype=float, copy=True)
        ride[:, 1] += self.marble_r
        if self.detail != "concept":
            from .cathedral import _polyline_samples
            lip = path + np.array([0.0, self.wall_h * 0.85, 0.0])
            radius = max(0.03, self.marble_r * 0.16)
            spacing = max(MIN_STONE * 2, min(MAX_STRETCHER, MAX_COURSE) - 2.0 * radius - 0.05)
            samples = _polyline_samples(lip, spacing, closed=True)
            host = f"marble-run.guide_spline.{self.counts['guide_spline']:04d}"
            for index, (a, b) in enumerate(zip(samples[:-1], samples[1:])):
                if float(np.linalg.norm(np.asarray(b) - np.asarray(a))) < MIN_STONE * 0.5:
                    continue
                self.sculpt(
                    "guide_spline",
                    tube_path([a, b], radius, sections=6, closed=False),
                    construction="closed riding spline", course=index, host_id=host,
                )

        start, _tangent = point_on_loop(ride, length, 0.0)
        # Icosahedron keeps phosphor feature edges; a smooth sphere vanishes.
        marble = trimesh.creation.icosphere(subdivisions=0, radius=self.marble_r)
        marble.apply_translation(tuple(float(value) for value in start))
        self.sculpt("marble", marble, construction="riding marble")

        path_list = [[round(float(c), 6) for c in row] for row in ride]
        return BuildDocument(ARCHITECT, self.request, self.placements, metadata={
            "marble_run": {
                "closed": True, "looping": True, "infinite": True,
                "length": round(length, 6),
                "speed": 6.0,
                "marble_radius": round(self.marble_r, 6),
                "samples": len(path_list),
                "path": path_list,
            },
            "invariants": {"required_roles": sorted(REQUIRED_ROLES), "min_placements": 8},
            "scope": "Looping visual canal; the marble follows the spline, it is not a rigid-body solve",
        })


def random_controls(rng=None):
    rng = rng if isinstance(rng, np.random.RandomState) else np.random.RandomState()
    return {
        "width": int(rng.randint(24, 49)),
        "height": int(rng.randint(8, 17)),
        "depth": int(rng.randint(20, 41)),
        "bays": int(rng.randint(4, 13)),
        "detail": str(rng.choice(DETAIL_LEVELS)),
        "levels": int(rng.randint(1, 4)),
    }


def build_document(request: BuildRequest) -> BuildDocument:
    return _Composer(request).compose()


def marble_run_checks(document):
    from .quality import _check, _envelope_connectivity_checks

    observed = {placement.role for placement in document.placements}
    checks = [_check(
        "marble.required_roles", REQUIRED_ROLES <= observed,
        "The marble run has its canal, piers and marble.",
        "The marble run is missing a defining member.",
        observed=sorted(observed), expected=sorted(REQUIRED_ROLES),
    )]
    record = (document.metadata or {}).get("marble_run") or {}
    path = np.asarray(record.get("path", []), dtype=float)
    closed = bool(record.get("closed")) and len(path) >= 8
    if closed:
        gap = float(np.linalg.norm(path[0] - path[-1]))
        spacing = float(record.get("length", 1.0)) / len(path)
        closed = gap < max(2.5 * spacing, 2.0)
    checks.append(_check(
        "marble.closed_path", closed,
        "The riding spline is a closed loop.",
        "The marble path is missing or is not a closed loop.",
        observed={"samples": int(len(path)), "closed": bool(record.get("closed"))},
        expected={"samples": ">=8", "closed": True},
    ))
    length = float(record.get("length") or 0.0)
    checks.append(_check(
        "marble.path_length", length >= min(document.request.width or DEFAULTS[0], 20.0),
        "The canal is long enough to ride.",
        "The marble path is too short to loop.",
        observed=length, expected=min(float(document.request.width or DEFAULTS[0]), 20.0),
    ))
    marbles = [p for p in document.placements if p.role == "marble"]
    on_path = False
    if marbles and len(path):
        start = np.asarray(marbles[0].position, dtype=float)
        on_path = bool(np.min(np.linalg.norm(path - start, axis=1)) < 1.5)
    checks.append(_check(
        "marble.on_path", on_path,
        "The marble starts on the canal spline.",
        "The marble is not on the riding path.",
    ))
    checks.extend(_envelope_connectivity_checks(document, "marble"))
    return checks
