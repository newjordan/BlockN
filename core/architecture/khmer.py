"""Sandstone temple-mountain of Angkor's 12th-century Khmer tradition.

A quincunx puts the tallest sanctuary tower between four corner towers.
Terraces and stairs lift the sanctuaries above open, corbel-vaulted galleries.
This is an Angkor Wat-inspired study, not a survey of the monument.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np

from .historic import Composer, Form, Voids, number, oriented_profile, taper
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("khmer", "1.0.0", "Angkor Khmer Architect")
STYLE = "angkor-temple-mountain"
REGION = "Angkor, Cambodia (mainland Southeast Asia)"
PERIOD = "12th century, Angkor Wat period of the Khmer Empire"
PRODUCT = "khmer-temple-mountain"
KINDS = {
    "core_fill": "core", "terrace_face": "foundation", "terrace_paving": "slab",
    "stair": "circulation", "gallery_pier": "pier", "gallery_architrave": "beam",
    "gallery_corbel": "roof", "gallery_ridge": "roof", "shrine_wall": "wall",
    "shrine_threshold": "slab", "shrine_frame": "ornament", "shrine_pediment": "ornament",
    "tower_bell": "roof", "tower_aedicule": "ornament", "tower_finial": "ornament",
}
FORMS = {PRODUCT: Form(
    PRODUCT, "Angkor Wat-inspired sandstone temple-mountain with five sanctuary towers and corbelled galleries",
    (44, 24, 44), ("angkor-wat",), frozenset(KINDS), {"bays": 4},
    rules={"region": REGION, "period": PERIOD, "bays": [3, 6],
           "bays_meaning": "column bays in each gallery wing",
           "width": [34, 60], "depth": [34, 60], "height": [18, 34],
           "height/short_side": [.45, .70], "width/depth": [.78, 1.28]},
)}
DEFAULT_SUBSTANCE = "sandstone"
SUBSTANCES = {"core_fill": "laterite"}
TOL = 1e-4
DIRECTIONS = (("x", -1), ("x", 1), ("z", -1), ("z", 1))


class Temple(Composer):
    PALETTE = {"terrace_face": (169, 133, 96, 255), "terrace_paving": (192, 159, 118, 255),
               "core_fill": (146, 88, 62, 255), "stair": (181, 150, 110, 255),
               "gallery_pier": (181, 149, 111, 255), "gallery_architrave": (210, 179, 133, 255),
               "gallery_corbel": (168, 138, 104, 255), "gallery_ridge": (192, 155, 112, 255),
               "shrine_wall": (190, 160, 122, 255), "shrine_threshold": (174, 148, 115, 255),
               "shrine_frame": (217, 188, 143, 255), "shrine_pediment": (213, 177, 129, 255),
               "tower_bell": (179, 145, 107, 255), "tower_aedicule": (206, 171, 121, 255),
               "tower_finial": (204, 167, 111, 255)}

    def compose(self):
        for name, value in zip(("width", "height", "depth"), (self.w, self.h, self.d)):
            number(value, f"{PRODUCT} {name}", *self.form.rules[name])
        self.m = min(self.w, self.d)
        self.ratio(self.h / self.m, "height/short_side", *self.form.rules["height/short_side"])
        self.ratio(self.w / self.d, "width/depth", *self.form.rules["width/depth"])
        self.bays = self.control("bays", *self.form.rules["bays"])
        y = 0.
        previous = self.w / 2, self.d / 2
        for tier, (span, rise) in enumerate(((.88, .060), (.77, .066), (.59, .080)), 1):
            x, z = self.w * span / 2, self.d * span / 2
            head = y + rise * self.h
            self.terrace(tier, x, z, y, head)
            for axis, sign in DIRECTIONS:
                i = int(axis == "z")
                self.stair(tier, axis, sign, (x, z)[i], previous[i], y, head)
            if tier == 2:
                self.galleries(x, z, head)
            y, previous = head, (x, z)

        for name, x, z in (("center", 0., 0.), *((f"{a}{b}", a * .203 * self.w, b * .203 * self.d)
                              for a in (-1, 1) for b in (-1, 1))):
            central = name == "center"
            self.shrine(name, x, z, y, self.h * (1 if central else .78),
                        self.m * (.173 if central else .127), central)
        return self.document({"region": REGION, "period": PERIOD, "terraces": 3, "towers": 5,
                              "gallery_bays": self.bays, "approach_axis": "-Z"})

    def terrace(self, tier, x, z, base, head):
        t, rim = .48, .18
        self.shell("terrace_face", -x, x, -z, z, base, head - rim, t, tier=tier)
        self.slab("core_fill", -x + t, x - t, -z + t, z - t, base, head - rim, module=2.1,
                  unit="core", construction="laterite platform fill", tier=tier)
        paving_module = {"concept": 2.3, "standard": 1.9, "presentation": 1.55}[self.detail]
        self.slab("terrace_paving", -x, x, -z, z, head - rim, head, module=paving_module, tier=tier)

    def stair(self, tier, axis, sign, edge, outer, base, head):
        # Stairs sit within the setback, with the upper riser reaching the
        # next terrace top. Split the climbing surface into solid treads.
        run = min(outer - edge, (head - base) * (.97 if tier < 3 else .82) + .62)
        count = math.ceil((head - base) / .255)
        width = self.m * (.103 - .014 * tier)
        for k in range(count):
            near, far = sorted((sign * (edge + run * k / count), sign * (edge + run * (k + 1) / count)))
            top = head - (head - base) * k / count
            x0, x1, z0, z1 = (near, far, -width / 2, width / 2) if axis == "x" else (
                -width / 2, width / 2, near, far)
            self.member("stair", x0, x1, base, top, z0, z1, tier=tier, axis=axis, sign=sign, tread=k)

    def galleries(self, x, z, base):
        # Eight wings leave axial breaks for access. The corridor stays open
        # under pairs of sandstone courses that step inward without a true arch.
        half = self.m * .028
        ridge_x, ridge_z = x - half - .32, z - half - .32
        pier_h = self.h * .096 * self.vary("gallery-pier", .055)
        for axis, facing in DIRECTIONS:
            along = ridge_x if axis == "x" else ridge_z
            cross = facing * (ridge_z if axis == "x" else ridge_x)
            a, b = self.m * .061, along - half - .24
            for side in (-1, 1):
                wing = f"{axis}.{facing}.{side}"
                begin, end = sorted((a * side, b * side))
                marks = np.linspace(begin + .23, end - .23, self.bays + 1)
                beam_top = base + pier_h + .28
                for row in (-1, 1):
                    line = cross + row * (half - .26)
                    for mark in marks:
                        x0, z0 = (mark, line) if axis == "x" else (line, mark)
                        self.stack("gallery_pier", x0, base, z0, .46, .52, pier_h,
                                   wing=wing, row=row)
                    divisions = [begin, *marks[1:-1], end]
                    for l, r in zip(divisions, divisions[1:]):
                        self.gallery_block("gallery_architrave", axis, l, r, line - .28, line + .28,
                                           base + pier_h, beam_top, wing=wing, row=row)
                self.corbel(wing, axis, begin, end, cross, half, beam_top)

    def gallery_block(self, role, direction, a, b, c, d, y0, y1, **tags):
        if direction == "x":
            return self.member(role, a, b, y0, y1, c, d, **tags)
        return self.member(role, c, d, y0, y1, a, b, **tags)

    def corbel(self, wing, axis, begin, end, cross, half, base):
        count = 5
        rise = half * .20
        length = end - begin
        pitch = {"concept": 1.45, "standard": 1.21, "presentation": .91}[self.detail]
        divisions = max(1, math.ceil(length / pitch))
        depth, advance = half * .45, half * .136
        for k in range(count):
            edges = np.linspace(begin, end, divisions + 1)
            if k % 2:
                edges[1:-1] -= length / divisions * .32
            y0 = base + k * rise
            for row in (-1, 1):
                c, d = sorted((cross + row * (half - k * advance),
                               cross + row * (half - k * advance - depth)))
                for a, b in zip(edges, edges[1:]):
                    self.gallery_block("gallery_corbel", axis, a, b, c, d, y0, y0 + rise,
                                       wing=wing, row=row, course=k, axis=axis)
        y = base + count * rise
        edges = np.linspace(begin, end, divisions + 1)
        for a, b in zip(edges, edges[1:]):
            self.gallery_block("gallery_ridge", axis, a, b, cross - half * .22, cross + half * .22,
                               y, y + rise * .78, wing=wing, axis=axis)

    def shrine(self, name, x, z, base, height, width, central):
        half, t = width / 2, .62
        room_top = base + (height - base) * .28
        door_height = min(2.4, (room_top - base) * .73)
        head = base + door_height
        half_door = min(.74, width * .13)
        self.slab("shrine_threshold", x - half, x + half, z - half, z + half, base, base + .18,
                  module=1.5, shrine=name)
        bottom = base + .18
        doors = "fblr" if central else "f"
        openings = {}
        for face in doors:
            voids = Voids(bottom)
            voids.rect(0, 2 * half_door, bottom, head - bottom, f"{name}.{face}")
            openings[face] = voids.specs
        self.shell("shrine_wall", x - half, x + half, z - half, z + half, bottom, room_top,
                   t, openings=openings, shrine=name)
        for face in doors:
            axis = "x" if face in "fb" else "z"
            along, cross = (x, z) if axis == "x" else (z, x)
            sign = -1 if face in "fl" else 1
            plane = cross + sign * (half - .07)
            for hand in (-1, 1):
                center = along + hand * (half_door + .12)
                self.gallery_block("shrine_frame", axis, center - .15, center + .15, plane - .19, plane + .19,
                                   bottom, head, shrine=name)
            self.gallery_block("shrine_frame", axis, along - half_door - .34, along + half_door + .34,
                               plane - .26, plane + .26, head, head + .21, shrine=name)
            self.pediment("shrine_pediment", axis, along, plane, head + .21, half_door + .4,
                          min(width * .18, room_top - head - .11), .49, shrine=name)
        self.bell(name, x, z, room_top, height, width)

    def pediment(self, role, axis, along, cross, y, half, rise, depth, **tags):
        # Stylised stepped ogive, like the nested frontons on tower doorways.
        profile = [(-half, y), (half, y), (half * .92, y + rise * .22),
                   (half * .68, y + rise * .22), (half * .51, y + rise * .54),
                   (half * .29, y + rise * .64), (0., y + rise),
                   (-half * .29, y + rise * .64), (-half * .51, y + rise * .54),
                   (-half * .68, y + rise * .22), (-half * .92, y + rise * .22)]
        self.mesh(role, oriented_profile(profile, depth, axis, along, cross), **tags)

    def bell(self, name, x, z, base, height, width):
        rise = height - base
        crown = base + rise * .865
        shoulder = self.vary("lotus-shoulder", .046)
        sizes = tuple(v * shoulder if i in (1, 2, 3) else v for i, v in enumerate((1., .90, .76, .58, .39, .22)))
        fractions = (0., .18, .38, .58, .76, 1.)
        for i, ((r, next_r), (a, b)) in enumerate(zip(zip(sizes, sizes[1:]), zip(fractions, fractions[1:]))):
            low, high = base + (crown - base) * a, base + (crown - base) * b
            sill = min(.24, (high - low) * .21)
            size = width * r
            # A square moulding below each receding bell stage. The sandstone
            # units in the upper course follow the sloping tower profile.
            self.ashlar("tower_bell", (x, low + sill / 2, z), (size, sill, size), shrine=name, tier=i)
            self.ashlar("tower_bell", (x, (low + sill + high) / 2, z),
                        (size * .94, high - low - sill, size * .94),
                        warp=taper(x, z, low + sill, high, next_r / (r * .94), next_r / (r * .94)),
                        shrine=name, tier=i)
            if i < 4:
                for axis, sign in DIRECTIONS:
                    along, cross = (x, z) if axis == "x" else (z, x)
                    self.pediment("tower_aedicule", axis, along, cross + sign * size * .451, low + sill,
                                  size * .25, (high - low - sill) * .68, size * .14, shrine=name, tier=i)
        r = width * sizes[-1] / 2
        knots = ((0., .91), (.27, .65), (.52, .76), (.69, .56), (.90, .26))
        self.lathe("tower_bell", x, z, [(crown + (height - crown) * u, r * v) for u, v in knots],
                   sides=12, shrine=name, tier=5)
        self.drum("tower_finial", x, z, crown + .9 * (height - crown), height, r * .26, r * .12,
                  sides=12, shrine=name)


Temple.module = sys.modules[__name__]


def build_document(request):
    return Temple(request).compose()


def random_controls(product, rng):
    w, d = rng.randint(37, 49), rng.randint(39, 46)
    return {"variant": FORMS[product].variants[0], "width": w, "depth": d,
            "height": round(min(w, d) * rng.uniform(.52, .63)), "bays": rng.randint(3, 6),
            "levels": None, "detail": rng.choice(("concept", "standard", "presentation"))}


def _bounds(p):
    center = np.asarray(p.position)
    half = np.asarray(p.dimensions) / 2
    return center - half, center + half


def rule_checks(document, form):
    roles = defaultdict(list)
    for p in document.placements:
        roles[p.role].append(p)
    terraces, stairs = defaultdict(list), defaultdict(list)
    for p in roles["terrace_paving"]:
        terraces[p.parameters.get("tier")].append(_bounds(p))
    for p in roles["stair"]:
        stairs[p.parameters.get("tier"), p.parameters.get("axis"), p.parameters.get("sign")].append(p)
    levels = [(np.min([lo for lo, _ in terraces[i]], axis=0),
               np.max([hi for _, hi in terraces[i]], axis=0)) for i in (1, 2, 3) if terraces[i]]
    nested = len(levels) == 3 and all((a[0][[0, 2]] < b[0][[0, 2]]).all() and
                                     (a[1][[0, 2]] > b[1][[0, 2]]).all() and a[1][1] < b[1][1]
                                     for a, b in zip(levels, levels[1:]))
    yield ("receding_terraces", nested, "Three terraces rise and recede toward the sanctuary platform.",
           "A terrace is missing or extends beyond the terrace below.", [])
    bad_stairs = []
    if nested:
        for tier, (lo, hi) in enumerate(levels, 1):
            base = levels[tier - 2][1][1] if tier > 1 else 0.
            for axis, sign in DIRECTIONS:
                dim = 0 if axis == "x" else 2
                ordered = sorted(stairs[tier, axis, sign], key=lambda p: p.position[dim] * sign)
                extent = (hi[dim] if sign == 1 else -lo[dim])
                previous, top = extent, hi[1]
                for i, p in enumerate(ordered):
                    a, b = _bounds(p)
                    inner, outer = (a[dim], b[dim]) if sign == 1 else (-b[dim], -a[dim])
                    rise = top - b[1]
                    if (abs(inner - previous) > TOL or abs(a[1] - base) > TOL or
                            (not TOL < rise <= .29 if i else abs(rise) > TOL)):
                        bad_stairs.append(p.id)
                    previous, top = outer, b[1]
    stair_count = sum(bool(stairs[t, a, s]) for t in (1, 2, 3) for a, s in DIRECTIONS)
    yield ("axial_stairways", nested and stair_count == 12 and not bad_stairs,
           "Stair flights on four axes meet each of the three raised terraces.",
           "A stair is missing, doesn't meet a terrace, or has a broken tread sequence.", bad_stairs)

    finials = roles["tower_finial"]
    central = max(finials, key=lambda p: p.position[1], default=None)
    towers_ok = False
    if central is not None and len(finials) == 5 and nested:
        tx, _, tz = central.position
        lo, hi = levels[-1]
        mid = (lo + hi) / 2
        others = [p for p in finials if p is not central]
        offsets = np.array([[p.position[0] - tx, p.position[2] - tz] for p in others])
        ref = np.abs(offsets[0])
        towers_ok = (np.linalg.norm([tx - mid[0], tz - mid[2]]) < TOL and
                     (ref > 1).all() and np.allclose(np.abs(offsets), ref, rtol=0, atol=TOL) and
                     len({tuple(np.sign(pair)) for pair in offsets}) == 4 and
                     all(abs(p.position[1] - others[0].position[1]) < TOL and
                         central.position[1] > p.position[1] + 1 and
                         lo[0] < p.position[0] < hi[0] and lo[2] < p.position[2] < hi[2] for p in others))
    yield ("five_tower_quincunx", towers_ok, "A taller central sanctuary stands between four equal corner towers.",
           "The central peak or one of the four corner towers is missing or displaced.",
           [] if towers_ok else [p.id for p in finials])

    courses = defaultdict(list)
    ridges = defaultdict(list)
    for p in roles["gallery_corbel"]:
        courses[p.parameters.get("wing"), p.parameters.get("row"), p.parameters.get("course")].append(p)
    for p in roles["gallery_ridge"]:
        ridges[p.parameters.get("wing")].append(p)
    bad_corbel = []
    wings = {wing for wing, _, _ in courses}
    complete = len(wings) == 8 and wings == set(ridges)
    for wing in sorted(wings, key=str):
        axis = 2 if ridges[wing] and ridges[wing][0].parameters.get("axis") == "x" else 0
        for row in (-1, 1):
            previous = None
            for i in range(5):
                band = courses[wing, row, i]
                if not band:
                    complete = False
                    break
                first = band[0]
                lo, hi = _bounds(first)
                bad_corbel.extend(p.id for p in band if abs(p.position[axis] - first.position[axis]) > TOL or
                                  abs(p.position[1] - first.position[1]) > TOL or
                                  abs(p.dimensions[1] - first.dimensions[1]) > TOL)
                if previous is not None:
                    a, b = previous
                    if abs(b[1] - lo[1]) > TOL or (first.position[axis] - (a[axis] + b[axis]) / 2) * row >= -TOL or (
                            min(b[axis], hi[axis]) - max(a[axis], lo[axis]) <= TOL):
                        bad_corbel.append(first.id)
                previous = lo, hi
            if previous is not None:
                for p in ridges[wing]:
                    a, b = _bounds(p)
                    if abs(a[1] - previous[1][1]) > TOL or (
                            min(b[axis], previous[1][axis]) - max(a[axis], previous[0][axis]) <= TOL):
                        bad_corbel.append(p.id)
    yield ("inward_corbel_steps", complete and not bad_corbel,
           "Each gallery ceiling closes on overlapping inward-stepped courses and a ridge cap.",
           "A gallery is missing, or its corbel steps or ridge are out of bearing.", sorted(set(bad_corbel)))
