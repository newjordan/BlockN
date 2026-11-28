"""A Djenné-inspired Sudano-Sahelian mosque, not a survey of the monument.

The present Great Mosque at Djenné dates to 1907. A raised platform keeps
its earthen fabric above the ground. An open court leads to a pillared prayer
hall under a flat earth roof. On the qibla side, tapering towers and engaged
buttresses carry protruding palm timbers (toron) used during replastering.
BlockN models the qibla facade on +Z and the courtyard approach on -Z.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np
import trimesh

from .historic import Composer, Form, Voids, box_mesh, number, oriented_profile, taper
from .masonry import iter_ashlar, iter_masonry_solid
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("sahelian", "1.0.0", "Sudano-Sahelian Architect")
STYLE = "sudano-sahelian"
REGION = "Djenné, Mali (West Africa)"
PERIOD = "Early 20th century; present Great Mosque completed in 1907"
PRODUCT = "sahelian-mosque"
KINDS = {
    "plinth": "foundation", "plinth_core": "foundation", "entry_step": "circulation",
    "court_floor": "slab", "hall_floor": "slab",
    "court_wall": "wall", "hall_wall": "wall", "qibla_wall": "wall", "lintel": "beam",
    "hall_pier": "pier", "roof_beam": "beam", "roof_joist": "beam", "roof_deck": "roof",
    "parapet": "wall", "pinnacle": "ornament", "qibla_tower": "pier", "buttress": "pier",
    "tower_crown": "ornament", "toron": "ornament",
}
FORMS = {PRODUCT: Form(
    PRODUCT, "Djenné-inspired earthen mosque with three battered towers, palm toron and an open courtyard",
    (42, 17, 46), ("djenne",), frozenset(KINDS), {"bays": 5},
    rules={"region": REGION, "period": PERIOD, "bays": [3, 7], "bays_step": 2,
           "bays_meaning": "aisles across the prayer hall (odd count: 3, 5 or 7)",
           "width": [24, 56], "height": [8, 28], "depth": [24, 64],
           "height/width": [.30, .55], "depth/width": [.84, 1.2], "qibla_axis": "+Z"},
)}
DEFAULT_SUBSTANCE = "mud-brick"
SUBSTANCES = dict.fromkeys(("lintel", "roof_beam", "roof_joist", "toron"), "palm-timber")
SUBSTANCES["plinth_core"] = "earth-fill"
EARTH = (190, 141, 92, 255)
PALM = (94, 69, 47, 255)
TOL = 1e-4


class Mosque(Composer):
    PALETTE = {**dict.fromkeys(KINDS, EARTH),
               **dict.fromkeys(("lintel", "roof_beam", "roof_joist", "toron"), PALM),
               "plinth": (163, 118, 76, 255), "court_floor": (207, 163, 113, 255),
               "hall_floor": (182, 135, 90, 255), "roof_deck": (202, 155, 106, 255)}
    TIMBER_KINDS = frozenset({"beam"})

    def block_count(self):
        counts = super().block_count()
        counts["earthen_blocks"] = sum(p.parameters.get("unit") == "adobe_block" for p in self.placements)
        counts["timbers"] = counts.get("timbers", 0) + self.counts["toron"]
        counts["other_members"] -= counts["earthen_blocks"] + self.counts["toron"]
        return counts

    def earth(self, role, center, dimensions, *, axis=None, openings=(), warp=None):
        """Reuse the running-bond layout, without calling the units cut stone."""
        host = self.host(role)
        x, y, z = center
        w, h, d = dimensions
        if axis:
            # The wall runner's openings use heights relative to the base.
            along, cross, length, depth = (x, z, w, d) if axis == "x" else (z, x, d, w)
            blocks = iter_ashlar(length=length, height=h, thickness=depth, axis=axis,
                                 base_y=y - h / 2, cross=cross, along_offset=along,
                                 openings=openings, detail=self.detail)
        else:
            blocks = iter_masonry_solid(center, dimensions, detail=self.detail)
        for block in blocks:
            tags = {"host_id": host, "course": block.course, "stone_index": block.index,
                    "axis": block.axis, "construction": "earthen running bond", "unit": "adobe_block"}
            if block.opening_id:
                tags["opening_id"] = block.opening_id
            if warp is None:
                self.add(role, block.position, block.dimensions, **tags)
            else:
                mesh = box_mesh(block.position, block.dimensions)
                mesh.vertices = warp(mesh.vertices)
                self.mesh(role, mesh, **tags)
        return host

    def wall(self, role, axis, a0, a1, cross, top, doors=()):
        voids = Voids(self.base)
        midpoint = (a0 + a1) / 2
        for i, (mark, clear) in enumerate(doors):
            voids.rect(mark - midpoint, clear, self.base, self.door_top, f"{self.host(role)}.door-{i + 1}")
        center = (midpoint, self.base + top / 2, cross) if axis == "x" else (cross, self.base + top / 2, midpoint)
        dims = (a1 - a0, top, self.thick) if axis == "x" else (self.thick, top, a1 - a0)
        host = self.earth(role, center, dims, axis=axis, openings=voids.specs)
        for mark, clear in doors:
            along = clear / 2 + .22
            # Bear each jamb beyond the actual opening, not on a floating insert.
            x0, x1, z0, z1 = (mark - along, mark + along, cross - .24, cross + .24) if axis == "x" else (
                cross - .24, cross + .24, mark - along, mark + along)
            self.member("lintel", x0, x1, self.base + self.door_top - .2, self.base + self.door_top,
                        z0, z1, host_id=host)

    def compose(self):
        w, h, d = self.w, self.h, self.d
        for name, size in zip(("width", "height", "depth"), (w, h, d)):
            number(size, f"{PRODUCT} {name}", *self.form.rules[name])
        bays = self.control("bays", *self.form.rules["bays"])
        if bays % 2 == 0:
            raise ValueError(f"{PRODUCT} requires an odd number of bays (3, 5 or 7)")
        self.ratio(h / w, "height/width", *self.form.rules["height/width"])
        self.ratio(d / w, "depth/width", *self.form.rules["depth/width"])
        x, z = .48 * w, .44 * d
        split = -.02 * d
        self.thick = .64
        self.base = min(.76, .035 * h + .23)
        self.door_top = min(2.85, .21 * h + .65)
        t, b = self.thick, self.base
        roof = max(b + self.door_top + .8, .49 * h)
        court_head = b + self.door_top + .48
        bay_pitch = 2 * (x - t) / bays
        door_width = min(1.7, .58 * bay_pitch)

        tower_depth = .049 * d + .8
        back = z + tower_depth - .15
        edge = .42
        for cross in (-z + edge / 2, back - edge / 2):
            self.earth("plinth", (0., b / 2, cross), (w, b, edge), axis="x")
        for cross in (-w / 2 + edge / 2, w / 2 - edge / 2):
            self.earth("plinth", (cross, b / 2, (back - z) / 2), (edge, b, back + z - 2 * edge), axis="z")
        self.slab("plinth_core", -w / 2 + edge, w / 2 - edge, -z + edge, back - edge,
                  0, b, module=3, unit="core", construction="compacted earth fill")
        self.slab("court_floor", -x + t, x - t, -z + t, split - t / 2, b, b + .07, module=2.5)
        self.slab("hall_floor", -x + t, x - t, split + t / 2, z - t, b, b + .07, module=2.5)
        for i in range(3):
            self.member("entry_step", -door_width, door_width, 0, b * (i + 1) / 3,
                        -z - .38 * (3 - i), -z - .38 * (2 - i))

        self.wall("court_wall", "x", -x, x, -z + t / 2, court_head - b, doors=[(0., door_width)])
        self.wall("hall_wall", "x", -x, x, split, roof - b,
                  doors=[(mark, door_width) for mark in (-bay_pitch, 0., bay_pitch)])
        self.wall("qibla_wall", "x", -x, x, z - t / 2, roof - b)
        for side in (-1, 1):
            cross = side * (x - t / 2)
            self.wall("court_wall", "z", -z + t, split - t / 2, cross, court_head - b)
            self.wall("hall_wall", "z", split + t / 2, z - t, cross, roof - b)

        xs = [-x + t + bay_pitch * i for i in range(1, bays)]
        zs = np.linspace(split, z - t / 2, max(3, round((z - split) / 5)) + 2)[1:-1]
        x_bearings = [-x, *xs, x]
        for row, mark in enumerate(zs):
            for cx in xs:
                h_pier = roof - .5 - b - .07
                self.earth("hall_pier", (cx, b + .07 + h_pier / 2, mark), (.68, h_pier, .68))
            for a, c in zip(x_bearings, x_bearings[1:]):
                self.member("roof_beam", a, c, roof - .5, roof - .24, mark - .14, mark + .14, row=row)
        z_bearings = [split, *zs, z]
        for mark in np.linspace(-x + t, x - t, 2 * bays + 1):
            for a, c in zip(z_bearings, z_bearings[1:]):
                self.member("roof_joist", mark - .085, mark + .085, roof - .24, roof, a, c)
        roof_top = roof + .24
        self.slab("roof_deck", -x, x, split - t / 2, z, roof, roof_top, module=2.5)

        self.parapet("x", -x, x, z - t / 2, roof_top)
        self.parapet("x", -x, x, split, roof_top)
        self.parapet("x", -x, x, -z + t / 2, court_head)
        for side in (-1, 1):
            cross = side * (x - t / 2)
            self.parapet("z", split + t / 2, z - t, cross, roof_top)
            self.parapet("z", -z + t, split - t / 2, cross, court_head)

        # The central tower is higher than the flanks. All towers and buttresses
        # overlap the qibla fabric behind them at both foot and head.
        batter = self.vary("tower-batter", .10)
        for side in (-1, 0, 1):
            self.tower(side * .27 * w, z, h if not side else .83 * h,
                       (.139 if not side else .12) * w, tower_depth, batter)
        for cx in (-.423 * w, -.138 * w, .138 * w, .423 * w):
            head = roof_top + .66
            warp = taper(cx, z - .25, b, head, .60, .75)
            self.earth("buttress", (cx, (b + head) / 2, z - .25 + 1.55 / 2),
                       (.052 * w, head - b, 1.55), warp=warp)
        return self.document({"region": REGION, "period": PERIOD, "bays": bays, "qibla_axis": "+Z"})

    def parapet(self, axis, a0, a1, cross, y):
        span = a1 - a0
        center = (a0 + a1) / 2
        xyz = (center, y + .16, cross) if axis == "x" else (cross, y + .16, center)
        dims = (span, .32, .42) if axis == "x" else (.42, .32, span)
        self.earth("parapet", xyz, dims, axis=axis)
        count = max(2, round(span / 1.8))
        for k in range(count):
            mark = a0 + span * (k + .5) / count
            rise = .52 + .12 * (k % 2)
            profile = [(-.32, y + .32), (.32, y + .32), (.14, y + .52),
                       (0., y + .32 + rise), (-.14, y + .52)]
            self.mesh("pinnacle", oriented_profile(profile, .33, axis, mark, cross))

    def tower(self, x, wall_front, height, width, depth, batter):
        crown = .078 * height
        head = height - crown
        inner = wall_front - .25
        z = inner + depth / 2
        shrink_x, shrink_z = 1 - .52 * batter, 1 - .37 * batter
        warp = taper(x, inner, self.base, head, shrink_x, shrink_z)
        host = self.earth("qibla_tower", (x, (head + self.base) / 2, z),
                          (width, head - self.base, depth), warp=warp)
        self.mesh("tower_crown", _spire(x, inner + depth * shrink_z / 2, head, height,
                                        width * shrink_x, depth * shrink_z), host_id=host)
        rows = max(4, round((head - self.base) / (1.45 if self.detail == "concept" else 1.18)))
        for i, fraction in enumerate(np.linspace(.16, .94, rows)):
            y = self.base + (head - self.base) * fraction
            for col in ((-.33, 0., .33) if i % 2 == 0 else (-.21, .21)):
                cx, _, face = warp(np.array([[x + col * width, y, z + depth / 2]]))[0]
                reach = .49 * self.vary(f"toron-{x:.3f}-{i}-{col}", .2)
                mesh = trimesh.creation.cylinder(radius=.081 if self.detail == "concept" else .072,
                                                 height=reach + .27, sections=8 if self.detail == "concept" else 12)
                mesh.apply_translation((cx, y, face + (reach - .27) / 2))
                self.mesh("toron", mesh, host_id=host)


def _spire(x, z, y0, y1, width, depth):
    mesh = box_mesh((x, (y0 + y1) / 2, z), (width, y1 - y0, depth))
    mesh.vertices = taper(x, z, y0, y1, .16, .16)(mesh.vertices)
    return mesh


def build_document(request):
    builder = Mosque(request)
    return builder.compose()


def random_controls(product, rng):
    w = rng.uniform(28, 48)
    form = FORMS[product]
    low, high = form.rules["bays"]
    return {"variant": rng.choice(form.variants), "width": w,
            "depth": w * rng.uniform(.9, 1.15), "height": w * rng.uniform(.33, .48),
            "bays": rng.randrange(low, high + 1, form.rules["bays_step"]), "levels": None,
            "detail": rng.choice(("concept", "standard", "presentation"))}


def _bounds(member):
    center, radius = np.asarray(member.position), np.asarray(member.dimensions) / 2
    return center - radius, center + radius


def _groups(members):
    groups = defaultdict(list)
    for p in members:
        groups[p.parameters.get("host_id")].append(p)
    return list(groups.values())


def _profile(members):
    """Measure the bottom and top courses; do not trust saved design ratios."""
    bounds = [_bounds(p) for p in members]
    low, high = min(lo[1] for lo, _ in bounds), max(hi[1] for _, hi in bounds)
    foot = [(lo, hi) for lo, hi in bounds if lo[1] < low + TOL]
    head = [(lo, hi) for lo, hi in bounds if hi[1] > high - TOL]
    return (np.min([lo for lo, _ in foot], axis=0), np.max([hi for _, hi in foot], axis=0),
            np.min([lo for lo, _ in head], axis=0), np.max([hi for _, hi in head], axis=0))


def rule_checks(document, form):
    roles = defaultdict(list)
    for p in document.placements:
        roles[p.role].append(p)

    towers = _groups(roles["qibla_tower"])
    profiles = sorted((_profile(group) for group in towers), key=lambda profile: profile[0][0])
    heads = [hi[1] for _, _, _, hi in profiles]
    central_height = heads[1] - profiles[1][0][1] if len(heads) == 3 else 0.
    silhouette = (len(heads) == 3 and heads[1] > max(heads[0], heads[2]) + .08 * central_height
                  and abs(heads[0] - heads[2]) < .02 * central_height
                  and len(roles["tower_crown"]) == 3
                  and all(lo[2] > 0 for lo, _, _, _ in profiles))
    yield ("three_qibla_towers", silhouette, "Three towers face the qibla, with a taller central tower.",
           "The qibla tower silhouette is incomplete or out of order.",
           [] if silhouette else [p.id for p in roles["qibla_tower"] + roles["tower_crown"]])

    piers = _groups(roles["buttress"])
    qibla = [_bounds(p) for p in roles["qibla_wall"]]
    bad_batter = []
    for group in towers + piers:
        foot_lo, foot_hi, head_lo, head_hi = _profile(group)
        attached = any(hi[2] > foot_lo[2] + TOL and hi[2] > head_lo[2] + TOL and lo[2] < foot_hi[2]
                       and lo[0] < head_hi[0] and hi[0] > head_lo[0] for lo, hi in qibla)
        if not attached or any((head_hi - head_lo)[i] >= .90 * (foot_hi - foot_lo)[i] for i in (0, 2)):
            bad_batter.append(group[0].id)
    yield ("battered_facade", len(piers) == 4 and not bad_batter,
           "Battered towers and four engaged buttresses narrow toward the head.",
           "A tower or buttress is missing, detached, or lacks batter.", bad_batter)

    tower_bounds = [np.array([_bounds(p) for p in group]) for group in towers]
    reached = defaultdict(list)
    bad_toron = []
    for p in roles["toron"]:
        lo, hi = _bounds(p)
        x, y, _ = p.position
        for i, bounds in enumerate(tower_bounds):
            mask = ((bounds[:, 0, 0] <= x + TOL) & (bounds[:, 1, 0] >= x - TOL) &
                    (bounds[:, 0, 1] <= y + TOL) & (bounds[:, 1, 1] >= y - TOL))
            if not np.any(mask):
                continue
            front = float(np.max(bounds[mask, 1, 2]))
            if p.substance == "palm-timber" and lo[2] < front - .09 and hi[2] > front + .24:
                reached[i].append(round(y, 3))
                break
        else:
            bad_toron.append(p.id)
    yield ("embedded_toron", bool(towers) and not bad_toron
           and all(len(set(reached[i])) >= 3 for i in range(len(towers))),
           "Palm toron are embedded in each tower and project in rows for replastering.",
           "A tower is missing toron rows, or a timber does not engage and project from the earth.", bad_toron)

    court = [_bounds(p) for p in roles["court_floor"]]
    covers = []
    if court:
        low, high = np.min([lo for lo, _ in court], axis=0), np.max([hi for _, hi in court], axis=0)
        for p in roles["roof_deck"] + roles["roof_beam"] + roles["roof_joist"]:
            lo, hi = _bounds(p)
            if hi[1] > high[1] and all(min(high[i], hi[i]) - max(low[i], lo[i]) > TOL for i in (0, 2)):
                covers.append(p.id)
    yield ("open_courtyard", bool(court and roles["roof_deck"]) and not covers,
           "The roof covers the prayer hall, not the courtyard.",
           "The courtyard is missing or spanned by roof members.", covers)

    piers = _groups(roles["hall_pier"])
    beams = [_bounds(p) for p in roles["roof_beam"]]
    bad_bearings = []
    for group in piers:
        _, _, lo, hi = _profile(group)
        if not any(abs(beam_lo[1] - hi[1]) < TOL
                   and min(hi[0], beam_hi[0]) - max(lo[0], beam_lo[0]) > TOL
                   and min(hi[2], beam_hi[2]) - max(lo[2], beam_lo[2]) > TOL for beam_lo, beam_hi in beams):
            bad_bearings.append(group[0].id)
    yield ("pillared_hall", bool(piers and beams) and not bad_bearings,
           "The interior earthen piers meet the flat roof's palm beams.",
           "A prayer hall pier has no beam bearing above it.", bad_bearings)


Mosque.module = sys.modules[__name__]
