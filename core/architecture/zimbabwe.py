"""Great Zimbabwe architect: the dry-stone enclosure.

The builders of Great Zimbabwe laid granite blocks dry, in level courses, with
no mortar anywhere. Out of that one technique come the rules of the form. A
wall never runs straight and never turns a corner: it curves all the way round.
It is battered, wide at the foot and narrower at the head. An entrance is a gap
in the wall with a stepped threshold. A solid conical tower stands free inside,
and a decorated band crowns the highest stretch of the outer wall, with upright
monoliths on the wall head above it. The houses inside are round, of daga
(clay), under conical thatch.

``great-enclosure`` has the inner wall that runs beside the outer one and makes
the narrow passage to the tower. ``valley-enclosure`` is a single wall, lower
at the front.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np

from .historic import TAU, clamp, extent, hull
from .schema import ArchitectInfo
from .sonnet import Poem, controls_for, form

ARCHITECT = ArchitectInfo("zimbabwe", "1.0.0", "Great Zimbabwe Architect")
STYLE = "zimbabwe-dry-stone"
PRODUCT = "zimbabwe-enclosure"
REGION = "Great Zimbabwe, Masvingo (Zimbabwe plateau, southern Africa)"
PERIOD = "Shona state of Great Zimbabwe, about 1100–1450 CE"
VARIANTS = ("great-enclosure", "valley-enclosure")
KINDS = {
    "outer_wall": "wall", "wall_lining": "wall", "wall_core": "core", "passage_wall": "wall",
    "threshold": "circulation", "decorated_band": "ornament", "monolith": "ornament",
    "tower_stone": "wall", "tower_core": "core",
    "hut_floor": "slab", "hut_wall": "wall", "hut_roof": "roof",
}
REQUIRED = set(KINDS) - {"passage_wall"}
FORMS = {PRODUCT: form(
    PRODUCT, "Curved dry-stone granite enclosure with battered walls, a solid conical tower and daga huts",
    (44, 9, 36), VARIANTS, REQUIRED,
    lines={"band": ("chevron", "herringbone", "dentelle"), "tower": ("single", "twin"),
           "entrances": ("three", "two"), "huts": ("ring", "cluster")},
    size={"width": (36, 60), "height": (7, 11), "depth": (30, 50)},
    ratios={"depth/width": (.7, .95), "height/width": (.15, .26)},
    region=REGION, period=PERIOD, mortar="none; every stone is laid dry",
    plan="one closed curve; no straight run and no corner")}
DEFAULT_SUBSTANCE = "granite"
SUBSTANCES = {"hut_floor": "mud-brick", "hut_wall": "mud-brick", "hut_roof": "thatch"}
STONE = ((150, 140, 128, 255), (164, 153, 138, 255), (139, 130, 120, 255), (157, 146, 130, 255))
SKIN = .5                        # depth of a facing stone
BAND_ROW, MONOLITH = .28, .9     # two rows of band, then the upright stones
CROWN = 2 * BAND_ROW + MONOLITH
WALLS = ("outer_wall", "wall_lining", "wall_core", "passage_wall")
TOL = 1e-4


class Enclosure(Poem):
    PALETTE = {"wall": STONE[0], "core": (128, 120, 110, 255), "threshold": (144, 134, 122, 255),
               "decorated_band": (176, 165, 148, 255), "monolith": (132, 124, 116, 255),
               "hut_floor": (168, 120, 84, 255), "hut_wall": (184, 132, 92, 255), "hut_roof": (186, 164, 104, 255)}

    def compose(self):
        a, b = self.w / 2, self.d / 2
        course, facet = {"concept": (.62, 2.2), "standard": (.46, 1.7), "presentation": (.33, 1.3)}[self.detail]
        self.head = self.h - CROWN                                   # the outer wall's head at the rear
        self.courses = max(4, round(self.head / course))
        self.course = self.head / self.courses
        perimeter = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        # A facet count of 4n + 2 puts a vertex on each end of the width and a facet square on the axis.
        self.facets = count = 4 * max(3, round((perimeter / facet - 2) / 4)) + 2
        step = TAU / count
        angles = -math.pi / 2 + step / 2 + step * np.arange(count)
        stretch = 1 / math.cos(step / 2)                             # the axial facets lie on the plot edge
        self.vertices = np.column_stack((a * np.cos(angles), b * stretch * np.sin(angles)))
        normals = np.column_stack((np.cos(angles) / a, np.sin(angles) / b))
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        self.mid = -math.pi / 2 + step * (np.arange(count) + 1)      # the angle at the middle of each facet
        self.foot = clamp(.085 * min(self.w, self.d), 2.6, 3.8) * self.vary("wall", .08)
        self.top = .55 * self.foot
        front = (.6 if self.variant == "great-enclosure" else .46) * self.vary("front", .06)
        rise = np.clip(((np.sin(self.mid) + 1) / 2 - .2) / .5, 0, 1)
        rise = rise * rise * (3 - 2 * rise)
        self.heights = [max(3, round(self.courses * (front + (1 - front) * s))) for s in rise]
        gaps = {count - 1, self.facet_at(-math.pi / 2 - 1.2 * self.vary("west-entrance", .12))}
        if self.line("entrances") == "three":
            gaps.add(self.facet_at(-math.pi / 2 + .95 * self.vary("east-entrance", .12)))
        self.outer_wall(gaps)
        inside = self.foot + 1.8
        if self.variant == "great-enclosure":
            inside = self.passage()
        self.towers(inside)
        self.huts()
        return self.document({"region": REGION, "period": PERIOD, "entrances": len(gaps),
                              "wall_courses": self.courses, "course_height": round(self.course, 3),
                              "wall_foot": round(self.foot, 3), "wall_head": round(self.top, 3),
                              "construction": "coursed granite laid dry; two faces over a packed core"})

    # ------------------------------------------------------------ the curve
    def facet_at(self, angle):
        return int(round((angle % TAU + math.pi / 2) / (TAU / self.facets) - 1)) % self.facets

    def point(self, facet, along, inset):
        """A plan point on a facet: ``along`` runs 0..1 between its vertices, ``inset`` is metres inward."""
        k, n = facet, (facet + 1) % self.facets
        p = self.vertices[k] - self.normals[k] * inset
        q = self.vertices[n] - self.normals[n] * inset
        return p + (q - p) * along

    def piece(self, facet, u, v, low, high, y0, y1):
        """A convex stone on a facet. ``low`` and ``high`` are its (near, far) insets at y0 and y1."""
        return hull([(x, y, z) for y, (near, far) in ((y0, low), (y1, high))
                     for along, inset in ((u, near), (v, near), (v, far), (u, far))
                     for x, z in (self.point(facet, along, inset),)])

    def lean(self, y):
        """Insets of the outer face and the inner face of the outer wall at height ``y``."""
        t = y / self.head
        outer = .6 * (self.foot - self.top) * t
        return outer, outer + self.foot - (self.foot - self.top) * t

    @staticmethod
    def bond(course):
        return (0., .5, 1.) if course % 2 == 0 else (0., .25, .75, 1.)

    def tint(self, start):
        for i, p in enumerate(self.placements[start:]):
            p.material = STONE[(int(p.parameters.get("course", 0)) * 3 + i * 7) % len(STONE)]

    # ------------------------------------------------------------ the walls
    def outer_wall(self, gaps):
        start = len(self.placements)
        for facet in range(self.facets):
            if facet in gaps:
                self.threshold(facet)
                continue
            for j in range(self.heights[facet]):
                y0, y1 = j * self.course, (j + 1) * self.course
                (o0, i0), (o1, i1) = self.lean(y0), self.lean(y1)
                cuts = self.bond(j)
                for u, v in zip(cuts, cuts[1:]):
                    self.stone("outer_wall", self.piece(facet, u, v, (o0, o0 + SKIN), (o1, o1 + SKIN), y0, y1),
                               f"outer.{facet}", j, facet=facet)
                    self.stone("wall_lining", self.piece(facet, u, v, (i0 - SKIN, i0), (i1 - SKIN, i1), y0, y1),
                               f"lining.{facet}", j, facet=facet)
            # The core is packed behind the faces in lifts of two courses.
            for j in range(0, self.heights[facet], 2):
                y0, y1 = j * self.course, min(j + 2, self.heights[facet]) * self.course
                (o0, i0), (o1, i1) = self.lean(y0), self.lean(y1)
                self.mesh("wall_core", self.piece(facet, 0, 1, (o0 + SKIN, i0 - SKIN), (o1 + SKIN, i1 - SKIN), y0, y1),
                          construction="dry rubble core", unit="core", course=j // 2, facet=facet)
        self.tint(start)
        self.crown()

    def threshold(self, facet):
        """An entrance is a gap in the wall, crossed by two rising steps."""
        half = self.foot / 2
        for step, (near, far) in enumerate(((0., half), (half, self.foot))):
            self.mesh("threshold", self.piece(facet, 0, 1, (near, far), (near, far), 0, .22 * (step + 1)),
                      facet=facet, step=step, construction="dry-stone step")

    def crown(self):
        """The decorated band and the monoliths, on the stretch of wall that stands to full height."""
        pattern = self.line("band")
        near = self.lean(self.head)[0]
        high = [f for f in range(self.facets) if self.heights[f] == self.courses]
        for n, facet in enumerate(high):
            length = float(np.linalg.norm(self.point(facet, 1, near) - self.point(facet, 0, near)))
            for row in range(2):
                y0 = self.head + row * BAND_ROW
                if pattern == "dentelle":           # upright blocks, every other one set back
                    count = max(2, round(length / .34))
                    for k in range(count):
                        back = .14 if (k + row) % 2 else 0.
                        self.mesh("decorated_band", self.piece(facet, k / count, (k + 1) / count, (near + back, near + SKIN),
                                                               (near + back, near + SKIN), y0, y0 + BAND_ROW),
                                  facet=facet, row=row, pattern=pattern)
                    continue
                # Thin slabs leaning at 45 degrees: by rows for a chevron, by neighbours for a herringbone.
                shift = BAND_ROW / length
                count = max(2, round((length - 2 * BAND_ROW) / .3))
                pitch = (1 - 2 * shift) / count
                for k in range(count):
                    sign = (1 if row == 0 else -1) if pattern == "chevron" else (1 if (k + row) % 2 == 0 else -1)
                    u = shift + k * pitch + pitch * .2 - sign * shift / 2
                    v = u + pitch * .6
                    points = [(x, y, z) for y, slide in ((y0, 0.), (y0 + BAND_ROW, sign * shift))
                              for along in (u + slide, v + slide) for inset in (near, near + SKIN)
                              for x, z in (self.point(facet, along, inset),)]
                    self.mesh("decorated_band", hull(points), facet=facet, row=row, pattern=pattern)
            if n % 3 == 1:
                x, z = self.point(facet, .5, near + SKIN / 2)
                self.member("monolith", x - .15, x + .15, self.head + 2 * BAND_ROW, self.h, z - .15, z + .15,
                            facet=facet, construction="upright monolith")

    def passage(self):
        """The inner wall beside the outer one; returns the inset of its inner face."""
        clear = 1.2 * self.vary("passage", .15)
        near, thick, lean = self.foot + clear, 1.1, .12
        courses = max(3, round(.85 * self.courses))
        start = len(self.placements)
        for facet in range(self.facets):
            if not math.radians(34) < self.mid[facet] < math.radians(152):
                continue
            length = float(np.linalg.norm(self.point(facet, 1, near) - self.point(facet, 0, near)))
            stones = max(2, math.ceil(length / 1.0))
            for j in range(courses):
                y0, y1 = j * self.course, (j + 1) * self.course
                l0, l1 = lean * j / courses, lean * (j + 1) / courses
                cuts = [k / stones for k in range(stones + 1)]
                if j % 2:
                    cuts = [0.] + [(k + .5) / stones for k in range(stones)] + [1.]
                for u, v in zip(cuts, cuts[1:]):
                    self.stone("passage_wall", self.piece(facet, u, v, (near + l0, near + thick - l0),
                                                          (near + l1, near + thick - l1), y0, y1),
                               f"passage.{facet}", j, facet=facet)
        self.tint(start)
        return near + thick

    # ------------------------------------------------------------ tower and huts
    def towers(self, inside):
        radius = clamp(.062 * min(self.w, self.d), 1.9, 2.8) * self.vary("tower", .08)
        height = (self.h - .35) * (1 - .06 * abs(self.vary("tower-height", 1) - 1))
        facet = self.facet_at(math.radians(58) * self.vary("tower-bearing", .1))
        centre = self.vertices[facet] - self.normals[facet] * (inside + 1.1 + radius)
        sizes = [(radius, height, centre)]
        if self.line("tower") == "twin":
            small = .46 * radius
            tangent = np.array((-self.normals[facet][1], self.normals[facet][0]))
            if tangent[0] > 0:
                tangent = -tangent                                   # the small tower stands toward the middle
            sizes.append((small, .45 * height, centre + tangent * (radius + small + .8)))
        for index, (r, top, (cx, cz)) in enumerate(sizes):
            # A slightly swelling cone: it draws in slowly at first and faster toward the head.
            profile = [(top * u, r * (1 - .58 * u ** 1.3)) for u in np.linspace(0, 1, 7)]
            start = len(self.placements)
            self.revolve("tower_stone", profile, thickness=.55, core="tower_core", cx=float(cx), cz=float(cz),
                         tower=index)
            self.tint(start)

    def huts(self):
        a, b = self.w / 2, self.d / 2
        r = clamp(.055 * min(self.w, self.d), 1.6, 2.4) * self.vary("hut", .07)
        if self.line("huts") == "ring":
            turn = math.radians(8) * (self.vary("hut-turn", 1) - 1)
            centres = [(.42 * a * math.cos(q + turn), .42 * b * math.sin(q + turn) - .12 * b)
                       for q in np.radians(np.linspace(165, 375, 5))]
        else:
            pitch = 2 * (r + .5) + 1.0
            cx, cz = -.34 * a * self.vary("hut-x", .08), -.1 * b * self.vary("hut-z", .2)
            centres = [(cx + sx * pitch / 2, cz + sz * pitch / 2) for sx in (-1, 1) for sz in (-1, 1)]
        for index, (x, z) in enumerate(centres):
            size = r * (1 + .12 * ((index * 5) % 3 - 1))
            wall, floor = 1.95, .15
            self.drum("hut_floor", x, z, 0, floor, size + .35, sides=16, hut=index)
            door = math.atan2(-z - .12 * b, -x)                       # the door looks to the middle of the yard
            self.ring("hut_wall", x, z, size, .3, floor, wall, gaps=[(door, .45 / size, floor, 1.6)],
                      hut=index, finish="daga")
            self.conic("hut_roof", x, z, wall, wall + size * .95, size + .5, 0., sides=10, hut=index)


Enclosure.module = sys.modules[__name__]
random_controls = controls_for(sys.modules[__name__])


def build_document(request):
    return Enclosure(request).compose()


def _box(p):
    half = np.asarray(p.dimensions) / 2
    return np.asarray(p.position) - half, np.asarray(p.position) + half


def rule_checks(document, form):
    roles = defaultdict(list)
    for p in document.placements:
        roles[p.role].append(p)
    outer = roles["outer_wall"]
    centre = half = None
    stray, curved = [], False
    if outer:
        lo, hi = extent(outer)
        centre, half = (lo + hi) / 2, (hi - lo) / 2
        for p in outer:
            radius = math.hypot((p.position[0] - centre[0]) / half[0], (p.position[2] - centre[2]) / half[2])
            if not .86 <= radius <= 1.02:
                stray.append(p.id)
        foot = sorted((p for p in outer if p.parameters.get("course") == 0),
                      key=lambda p: math.atan2((p.position[2] - centre[2]) / half[2],
                                               (p.position[0] - centre[0]) / half[0]))
        if len(foot) >= 12:
            points = [np.array((p.position[0], p.position[2])) for p in foot]
            steps = [q - p for p, q in zip(points, points[1:] + points[:1])]
            bearings = [math.atan2(s[1], s[0]) for s in steps]
            turns = [(b - a + math.pi) % TAU - math.pi for a, b in zip(bearings, bearings[1:] + bearings[:1])]
            runs = [sum(turns[(i + k) % len(turns)] for k in range(3)) for i in range(len(turns))]
            curved = min(turns) > -.02 and max(turns) < .5 and min(runs) > .02 and abs(sum(turns) - TAU) < .01
    yield ("curved_walls", bool(outer) and curved and not stray,
           "The outer wall follows one closed curve, with no straight run and no corner.",
           "The outer wall leaves its curve, runs straight, or turns a corner.", stray)

    battered = False
    if outer and roles["wall_lining"]:
        rear = [p for p in (*outer, *roles["wall_lining"])
                if abs(p.position[0] - centre[0]) < 2.0 and p.position[2] > centre[2]]
        levels = sorted({p.parameters.get("course") for p in rear if isinstance(p.parameters.get("course"), int)})
        if len(levels) >= 3:
            spans = []
            for level in (levels[0], levels[-1]):
                boxes = [_box(p) for p in rear if p.parameters.get("course") == level]
                spans.append((min(lo[2] for lo, _ in boxes), max(hi[2] for _, hi in boxes)))
            (f0, f1), (h0, h1) = spans
            battered = h1 - h0 < f1 - f0 - .2 and h1 < f1 - .05 and h0 > f0 + .02
    yield ("battered_walls", battered, "The outer wall is wide at the foot and draws in toward the head.",
           "The outer wall is plumb, or wider at the head than at the foot.", [])

    towers = defaultdict(list)
    for p in (*roles["tower_stone"], *roles["tower_core"]):
        towers[p.parameters.get("tower")].append(p)
    walls = [_box(p) for role in WALLS for p in roles[role]]
    bad = []
    for members in towers.values():
        lo, hi = extent(members)
        axis = (lo + hi) / 2
        base = extent([p for p in members if _box(p)[0][1] < lo[1] + .9])
        head = extent([p for p in members if _box(p)[1][1] > hi[1] - .9])
        tapers = head[1][0] - head[0][0] < .75 * (base[1][0] - base[0][0])
        # Solid: at every height some member holds the tower's own axis.
        spans = sorted((b[0][1], b[1][1]) for b in map(_box, members)
                       if all(b[0][i] < axis[i] < b[1][i] for i in (0, 2)))
        reach = lo[1]
        for y0, y1 in spans:
            if y0 > reach + .02:
                break
            reach = max(reach, y1)
        solid = lo[1] < .02 and reach > hi[1] - .02
        # Free: no wall stone comes within a hand of the round foot.
        reach_out = (hi[0] - lo[0]) / 2 + .3
        free = not any(math.hypot(*(max(wlo[i] - axis[i], 0., axis[i] - whi[i]) for i in (0, 2))) < reach_out
                       for wlo, whi in walls)
        if not (tapers and solid and free):
            bad += [p.id for p in members]
    yield ("solid_tower", bool(towers) and not bad,
           "Each conical tower is solid, tapers to its head, and stands free of the walls.",
           "A tower is missing, hollow, untapered, or built into a wall.", bad)

    band, low = roles["decorated_band"], []
    crowned = False
    if band and outer:
        wall_top = max(_box(p)[1][1] for p in outer)
        low = [p.id for p in band if _box(p)[0][1] < wall_top - .02 or p.position[2] < centre[2]]
        full = [_box(p) for p in outer if _box(p)[1][1] > wall_top - .02]
        # Every band stone stands over a facing stone of the full-height wall.
        for p in band:
            lo, hi = _box(p)
            if p.id not in low and not any(all(a[i] < hi[i] + .05 and b[i] > lo[i] - .05 for i in (0, 2)) for a, b in full):
                low.append(p.id)
        crowned = not low
    yield ("crowning_band", crowned, "The decorated band sits on the highest stretch of the outer wall.",
           "The decorated band is missing or does not sit on the highest stretch of the wall.", low)
