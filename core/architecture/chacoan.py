"""Chacoan great house: curved rear range, stepped roofs and recessed plaza kivas."""
from __future__ import annotations

from collections import defaultdict
import math
import random
import sys

import numpy as np
import trimesh

from .historic import Composer, Form, TAU, Voids, extent, hull, number
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("chacoan", "1.0.0", "Chacoan Great House Architect")
PRODUCT = "chacoan-great-house"
STYLE = "chacoan-sandstone"
REGION = "Chaco Canyon, San Juan Basin (New Mexico, United States)"
PERIOD = "Ancestral Pueblo Chacoan great houses, 9th–12th centuries"
VARIANTS = ("single-kiva", "paired-kivas")
KINDS = {
    "site_earth": "foundation", "plaza_surface": "slab", "room_wall": "wall", "back_wall": "wall",
    "partition": "wall", "viga": "beam", "terrace_deck": "roof", "access_ladder": "circulation",
    "kiva_floor": "slab", "kiva_wall": "wall", "kiva_bench": "seating",
}
FORMS = {PRODUCT: Form(
    PRODUCT, "Chacoan great house with curved sandstone masonry, stepped viga roofs and recessed plaza kivas",
    (32, 9, 28), VARIANTS, frozenset(KINDS), {"bays": 5},
    rules={"region": REGION, "period": PERIOD, "width": [26, 42], "height": [8, 11], "depth": [23, 37],
           "depth/width": [.72, 1.03], "bays": [4, 7], "bays_meaning": "plaza-facing rooms per terrace",
           "kivas_by_variant": dict(zip(VARIANTS, (1, 2))), "tiers_by_variant": dict(zip(VARIANTS, (2, 3)))},
)}
DEFAULT_SUBSTANCE = "sandstone"
SUBSTANCES = {"site_earth": "earth-fill", "plaza_surface": "earth-fill", "terrace_deck": "earth-fill",
              "kiva_floor": "earth-fill", "viga": "softwood", "access_ladder": "softwood"}
STONE = ((191, 144, 100, 255), (205, 163, 117, 255), (176, 127, 88, 255),
         (199, 154, 108, 255), (186, 139, 100, 255))
TOL = 1e-5
KIVA_WALL = .48
FLOOR_LAP = .01


def plan_counts(variant):
    kivas = VARIANTS.index(variant) + 1
    return kivas, kivas + 1


class GreatHouse(Composer):
    PALETTE = {"site_earth": (151, 111, 79, 255), "plaza_surface": (204, 165, 119, 255), "wall": STONE[0],
               "kiva_floor": (177, 141, 104, 255), "kiva_bench": STONE[1], "terrace_deck": (186, 151, 113, 255),
               "viga": (105, 70, 45, 255), "access_ladder": (122, 81, 53, 255)}
    TIMBER_KINDS = frozenset({"beam"})
    WALL = .53
    DECK = .18

    def compose(self):
        for name, value in zip(("width", "height", "depth"), (self.w, self.h, self.d)):
            number(value, f"{PRODUCT} {name}", *self.form.rules[name])
        self.ratio(self.d / self.w, "depth/width", *self.form.rules["depth/width"])
        self.bays = self.control("bays", *self.form.rules["bays"])
        self.kivas, self.tiers = plan_counts(self.variant)
        self.plaza = 1.72 * self.vary("kiva-depth", .10)
        self.floor = .19
        self.rx, self.rz = .476 * self.w, .425 * self.d
        self.spring = .053 * self.d
        self.rise = (self.h - self.plaza) / self.tiers
        self.setback = .116 * self.d * self.vary("setback", .07)
        self.course, self.stone_length, self.facets = {"concept": (.50, 1.33, 10), "standard": (.38, 1.06, 14),
                                                       "presentation": (.28, .88, 18)}[self.detail]
        radius = min(self.w * (.126 if self.kivas == 1 else .102), self.d * .153) * self.vary("kiva-radius", .08)
        kz = -.239 * self.d * self.vary("kiva-position", .08)
        centers = [0.] if self.kivas == 1 else [-.222 * self.w, .222 * self.w]
        self.site(centers, kz, radius + .36)
        for k, x in enumerate(centers):
            self.kiva(k, x, kz, radius)
        for tier in range(self.tiers):
            self.range(tier)
        return self.document({"region": REGION, "period": PERIOD, "plaza_kivas": self.kivas,
                              "room_tiers": self.tiers, "bays_per_tier": self.bays})

    def panel(self, role, polygon, bottom, top, **params):
        """Extrude one convex, horizontal tile; triangular end tiles are allowed."""
        self.mesh(role, hull([(x, y, z) for y in (bottom, top) for x, z in polygon]), **params)

    def site(self, centers, z, hole):
        """Tile straight bands outside a polygonal excavation at each kiva."""
        x0, x1 = -.5 * self.w, .5 * self.w
        z0, z1 = -.5 * self.d, .5 * self.d
        for role, lo, hi in (("site_earth", 0., self.floor), ("site_earth", self.floor, self.plaza - .13),
                             ("plaza_surface", self.plaza - .13, self.plaza)):
            for a, b in ((z0, z - hole), (z + hole, z1)):
                self.slab(role, x0, x1, a, b, lo, hi, module=3.3)
            if hi <= self.floor:
                self.slab(role, x0, x1, z - hole, z + hole, lo, hi, module=3.3)
                continue
            # Same angular stations are used on both sides, so the row strips
            # share exact seams. A small embed at the outer wall closes the cut.
            cuts = [(z + hole * math.sin(a), hole * math.cos(a))
                    for a in np.linspace(-math.pi / 2, math.pi / 2, 25)]
            for (a, ra), (b, rb) in zip(cuts, cuts[1:]):
                left = [(x0, x0)] + [(x + ra, x + rb) for x in centers]
                right = [(x - ra, x - rb) for x in centers] + [(x1, x1)]
                for (l0, l1), (r0, r1) in zip(left, right):
                    n = math.ceil(max(r0 - l0, r1 - l1) / 3.3)
                    xa, xb = np.linspace(l0, r0, n + 1), np.linspace(l1, r1, n + 1)
                    for i in range(n):
                        self.panel(role, [(xa[i], a), (xa[i+1], a), (xb[i+1], b), (xb[i], b)], lo, hi)

    def kiva(self, k, x, z, radius):
        disk = trimesh.creation.cylinder(radius=radius + FLOOR_LAP, height=.10, sections=48)
        disk.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))
        disk.apply_translation((x, self.floor + .05, z))
        self.mesh("kiva_floor", disk, kiva=k)
        start = len(self.placements)
        self.ring("kiva_wall", x, z, radius + KIVA_WALL, KIVA_WALL, self.floor, self.plaza, kiva=k)
        self.tint(start)
        self.ring("kiva_bench", x, z, radius + .045, .42, self.floor + .10, self.floor + .43, kiva=k)

    def tint(self, start):
        for i, p in enumerate(self.placements[start:]):
            p.material = STONE[(p.parameters["course"] * 3 + i * 7) % len(STONE)]

    def rear_curve(self, front):
        # A different chord of the same ellipse for every floor. The upper
        # chord recedes without letting any upper plan overhang its lower roof.
        angles = np.linspace(math.pi, 0, self.facets + 1)
        curve = np.column_stack((self.rx * np.cos(angles), self.spring + self.rz * np.sin(angles)))
        crest = self.facets // 2
        edge = np.interp(front, curve[:crest + 1, 1], curve[:crest + 1, 0])
        return np.vstack(((-abs(edge), front), curve[curve[:, 1] > front + TOL], (abs(edge), front)))

    def curved_wall(self, curve, low, top, tier):
        start = len(self.placements)
        beds = np.linspace(low, top, math.ceil((top - low) / self.course) + 1)
        for j, (ya, yb) in enumerate(zip(beds, beds[1:])):
            for a, b in zip(curve, curve[1:]):
                # The same radial thickness at each vertex gives adjoining
                # facets a shared mitre instead of letting rectangular stones overlap.
                inner = [np.array((0., self.spring)) + (p - (0., self.spring)) *
                         (1 - self.WALL / np.linalg.norm(p - (0., self.spring))) for p in (a, b)]
                n = math.ceil(np.linalg.norm(b - a) / self.stone_length)
                cuts = np.linspace(0., 1., n + 1)
                if j % 2:
                    cuts = np.r_[0., (cuts[:-1] + cuts[1:]) / 2, 1.]
                for u, v in zip(cuts, cuts[1:]):
                    c, d = (a + (b - a) * t for t in (u, v))
                    e, f = (inner[0] + (inner[1] - inner[0]) * t for t in (v, u))
                    mesh = hull([(x, y, z) for y in (ya, yb) for x, z in (c, d, e, f)])
                    self.stone("back_wall", mesh, f"rear.{tier}", j, tier=tier)
        self.tint(start)

    def range(self, tier):
        low = self.plaza + tier * self.rise
        top = self.plaza + (tier + 1) * self.rise
        # All front walls cut a chord and lock into the ends of the rear arc.
        front = self.spring + self.setback * tier
        curve = self.rear_curve(front)
        left, right = curve[0, 0], curve[-1, 0]
        dividers = np.linspace(left + .24, right - .24, self.bays + 1)
        midpoints = (dividers[:-1] + dividers[1:]) / 2
        doors = Voids(low)
        for i, x in enumerate(midpoints):
            doors.rect(x, .82, low, min(self.rise * .68, 1.62), f"tier-{tier}.room-{i}")
        start = len(self.placements)
        self.run("room_wall", "x", 0., front + self.WALL / 2, right - left,
                 low, top - self.DECK, self.WALL, voids=doors, tier=tier)
        self.tint(start)
        self.curved_wall(curve, low, top - self.DECK, tier)
        # Retain one stone partition between adjoining rooms, but seat the
        # vigas on a secondary rear bearing wall to shorten the longer spans.
        for x in dividers[1:-1]:
            back = float(np.interp(x, *curve.T)) - .08
            a, b = front + self.WALL, back
            self.run("partition", "z", (a + b) / 2, x, b - a, low, top - self.DECK, .34, tier=tier)
        beam_line = front + (curve[:, 1].max() - front) * .48
        bx = float(np.interp(beam_line, curve[:len(curve)//2 + 1, 1], curve[:len(curve)//2 + 1, 0]))
        self.run("partition", "x", 0., beam_line, -2 * bx, low, top - self.DECK, .36, tier=tier)
        for x in np.linspace(left + 1., right - 1., math.ceil((right - left) / .86)):
            back = float(np.interp(x, *curve.T)) - .02
            starts = (front - .17, beam_line) if back > beam_line + .3 else (front - .17,)
            for a, b in zip(starts, (*starts[1:], back)):
                self.member("viga", x - .105, x + .105, top - self.DECK - .19, top - self.DECK + .02,
                            a, b, tier=tier)
        # Clip terrace tiles to the faceted rear plan. Build in strips so every
        # tile remains a closed primitive and no Boolean engine is required.
        for a, b in zip(curve, curve[1:]):
            n = math.ceil((b[0] - a[0]) / 1.6)
            points = np.linspace(a, b, n + 1)
            for (xa, za), (xb, zb) in zip(points, points[1:]):
                slices = math.ceil(max(za - front, zb - front) / 2.3)
                ya, yb = np.linspace(front, za, slices + 1), np.linspace(front, zb, slices + 1)
                for i in range(slices):
                    footprint = [(xa, ya[i]), (xb, yb[i]), (xb, yb[i+1]), (xa, ya[i+1])]
                    if za - front < TOL:
                        footprint.pop(3)
                    elif zb - front < TOL:
                        footprint.pop(2)
                    self.mesh("terrace_deck", hull([(x, y, z) for y in (top - self.DECK, top)
                                                    for x, z in footprint]), tier=tier)
        if tier:
            self.ladder(dividers[1] if tier % 2 else dividers[-2], low, top, front)

    def ladder(self, x, low, top, front):
        heel = front - min(.92, self.setback * .65)
        head = front - .02
        for side in (-1, 1):
            c = x + side * .285
            self.mesh("access_ladder", hull([(c + dx, y, z + dz)
                                             for y, z in ((low, heel), (top + .35, head))
                                             for dx in (-.047, .047) for dz in (-.047, .047)]))
        count = math.ceil((top - low) / .33)
        for i in range(1, count + 1):
            y = low + i * (top - low) / (count + 1)
            z = heel + (head - heel) * (y - low) / (top + .35 - low)
            self.member("access_ladder", x - .285, x + .285, y - .042, y + .042, z - .06, z + .06)


GreatHouse.module = sys.modules[__name__]


def build_document(request):
    return GreatHouse(request).compose()


def random_controls(product, rng):
    width = rng.randint(28, 36)
    return {"width": width, "height": rng.randint(8, 10), "depth": round(width * rng.uniform(.79, .97)),
            "bays": rng.randint(4, 7)}


def bound(p, axis, side):
    return p.position[axis] + side * p.dimensions[axis] / 2


def world_vertices(p):
    return np.asarray(p.parameters["vertices"]) * p.dimensions + p.position


def crosses_circle(part, center, radius):
    """Intersect the inner opening against the placed mesh's XZ footprint."""
    # Reject most members by their cheap envelope first; it is exact for boxes.
    center = np.asarray(center)
    delta = np.maximum(np.abs(np.asarray(part.position)[[0, 2]] - center) - np.asarray(part.dimensions)[[0, 2]] / 2, 0)
    if np.dot(delta, delta) >= radius ** 2:
        return False
    if part.block_type != "MESH":
        return True
    triangles = world_vertices(part)[:, [0, 2]][np.asarray(part.parameters["faces"])] - center
    edges = np.roll(triangles, -1, axis=1) - triangles
    turn = edges[:, :, 0] * -triangles[:, :, 1] + edges[:, :, 1] * triangles[:, :, 0]
    contains = (np.all(turn > TOL, axis=1) | np.all(turn < -TOL, axis=1)).any()
    length = np.sum(edges ** 2, axis=2)
    t = np.clip(np.sum(-triangles * edges, axis=2) / np.maximum(length, TOL ** 2), 0, 1)
    closest = triangles + t[:, :, None] * edges
    return bool(contains or (np.sum(closest ** 2, axis=2) < radius ** 2).any())


def courses_close(parts, center, radius, bottom, top):
    """Check every bed and angular joint against the actual stone vertices."""
    beds = defaultdict(list)
    for part in parts:
        if part.block_type != "MESH":
            return False
        points = world_vertices(part)[:, [0, 2]] - center
        radii = np.linalg.norm(points, axis=1)
        if abs(radii.min() - radius) > TOL or abs(radii.max() - radius - KIVA_WALL) > TOL:
            return False
        angles = np.arctan2(points[:, 1], points[:, 0])
        u = angles[0]
        local = (angles - u + math.pi) % TAU - math.pi
        start = (u + local.min()) % TAU
        span = local.max() - local.min()
        if not TOL < span < math.pi / 2:
            return False
        band = round(bound(part, 1, -1), 6), round(bound(part, 1, 1), 6)
        beds[band].append((start, start + span))
    y = bottom
    for (a, b), arcs in sorted(beds.items()):
        if abs(a - y) > TOL:
            return False
        arcs.sort()
        following = [c for c, _ in arcs[1:]] + [arcs[0][0] + TAU]
        if any(abs(d - c) > TOL for (_, d), c in zip(arcs, following)):
            return False
        y = b
    return abs(y - top) <= TOL


def rule_checks(document, form):
    request = document.request
    variant = request.variant
    if variant in {"auto", "random"}:
        variant = form.variants[0] if variant == "auto" else random.Random(request.seed).choice(form.variants)
    kivas, tiers = plan_counts(variant)
    roles = defaultdict(list)
    for p in document.placements:
        roles[p.role].append(p)
    floors = {p.parameters.get("kiva"): p for p in roles["kiva_floor"]}
    walls = defaultdict(list)
    for p in roles["kiva_wall"]:
        walls[p.parameters.get("kiva")].append(p)
    plaza = max((bound(p, 1, 1) for p in roles["plaza_surface"]), default=0.)
    open_ok = set(floors) == set(range(kivas)) and len(roles["kiva_floor"]) == kivas
    wall_ok = open_ok and set(walls) == set(floors)
    blocked, broken = [], []
    for index, floor in floors.items():
        center = np.asarray(floor.position)[[0, 2]]
        r = min(floor.dimensions[0], floor.dimensions[2]) / 2
        floor_top = bound(floor, 1, 1)
        if floor_top >= plaza - .8 or abs(floor.dimensions[0] - floor.dimensions[2]) > TOL:
            open_ok = False
        blocked.extend(p.id for p in document.placements if bound(p, 1, 1) > floor_top + TOL
                       and crosses_circle(p, center, r * .77))
        if not courses_close(walls[index], center, r - FLOOR_LAP, bound(floor, 1, -1), plaza):
            wall_ok = False
            broken.extend(p.id for p in walls[index])
    yield ("open_kivas", open_ok and not blocked, "The circular plaza floors are recessed and open above.",
           "A plaza kiva is missing, too shallow or blocked above.", sorted(set(blocked)))
    yield ("kiva_courses", wall_ok, "Continuous radial stone courses line each excavation.",
           "A circular lining has a missing or displaced stone or bed.", broken)

    decks, faces = defaultdict(list), defaultdict(list)
    for p in roles["terrace_deck"]:
        decks[p.parameters.get("tier")].append(p)
    for p in roles["room_wall"]:
        faces[p.parameters.get("tier")].append(p)
    ok = decks.keys() == faces.keys() == set(range(tiers))
    if ok:
        limits = [extent(decks[t]) for t in range(tiers)]
        for t, (lo, hi) in enumerate(limits):
            wlo, whi = extent(faces[t])
            ok &= (abs(wlo[2] - lo[2]) < TOL and abs(whi[1] - lo[1]) < TOL
                   and all(abs(bound(p, 1, 1) - hi[1]) < TOL for p in decks[t]))
            if t:
                prev, last = limits[t-1]
                ok &= (lo[2] - prev[2] > 1.2 and hi[1] - last[1] > 1.55 and hi[0] < last[0]
                       and abs(wlo[1] - last[1]) < TOL and abs(hi[2] - last[2]) < .35)
            else:
                ok &= abs(wlo[1] - plaza) < TOL
    yield ("terraced_range", ok, "Room fronts bear successive flat roofs that step back from the plaza.",
           "A room range does not bear or recede with its terrace.", [] if ok else [p.id for p in roles["terrace_deck"]])
