"""Tibetan architect: the fortress-palace.

A Tibetan palace is a hill made taller. Its walls are stone, whitewashed, and
every face leans inward. The lowest storeys are blind; the windows begin higher
up and grow toward the roof, each under a timber lintel and inside a black
surround that is wider at the sill than at the head. The block of highest rank
is red, and a dark band of bundled tamarisk (penbey) runs under its parapet.
Roofs are flat. What stands on the highest roof is gilded: small hipped
pavilion roofs (gyaphib) or victory banners (gyaltsen).

``potala`` follows the Potala Palace at Lhasa: white wings step up on either
side to a taller red palace. ``dzong`` follows the fortress-monasteries of
Tibet and Bhutan: ranges under pitched timber roofs close a court around a
central tower (utse). Both are reached by a broad stair up a faced base.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np

from .historic import Voids, clamp, extent, hull, taper
from .schema import ArchitectInfo
from .sonnet import Poem, controls_for, form

ARCHITECT = ArchitectInfo("tibetan", "1.0.0", "Tibetan Fortress-Palace Architect")
STYLE = "tibetan-fortress-palace"
PRODUCT = "tibetan-palace"
REGION = "Tibetan Plateau and the eastern Himalaya (Lhasa, Shigatse, Bhutan)"
PERIOD = "17th century, the Potala of the Fifth Dalai Lama and the dzongs of Zhabdrung Ngawang Namgyal"
VARIANTS = ("potala", "dzong")
KINDS = {
    "core_fill": "core", "base_wall": "wall", "terrace_paving": "slab",
    "stair_step": "circulation", "stair_landing": "slab",
    "wing_wall": "wall", "palace_wall": "wall", "frieze": "ornament", "parapet": "wall", "roof_deck": "roof",
    "window_frame": "ornament", "portal_frame": "ornament", "portal_lintel": "entablature",
    "range_roof": "roof", "gallery_post": "column", "gallery_beam": "beam", "pavilion": "ornament", "pavilion_roof": "roof", "banner": "ornament",
    "finial": "ornament",
}
REQUIRED = {"core_fill", "base_wall", "terrace_paving", "stair_step", "wing_wall", "palace_wall", "frieze",
            "parapet", "roof_deck", "window_frame", "portal_frame", "portal_lintel", "finial"}
FORMS = {PRODUCT: form(
    PRODUCT, "Tibetan fortress-palace of battered whitewashed stone with a red central block and gilded roof crowns",
    (42, 29, 27), VARIANTS, REQUIRED,
    lines={"wings": ("stepped", "level"), "roofs": ("gilded-pavilions", "banners"),
           "stair": ("zigzag", "axial"), "frieze": ("single", "double")},
    size={"width": (36, 52), "height": (25, 36), "depth": (26, 36)},
    ratios={"depth/width": (.6, .75), "height/width": (.6, .8)},
    region=REGION, period=PERIOD,
    openings="timber-lintelled windows in black trapezoid surrounds; the lowest storey is blind")}
DEFAULT_SUBSTANCE = "granite"
SUBSTANCES = {"frieze": "painted-timber", "window_frame": "painted-timber", "portal_frame": "painted-timber",
              "portal_lintel": "softwood", "roof_deck": "earth-fill", "range_roof": "softwood",
              "gallery_post": "softwood", "gallery_beam": "softwood",
              "pavilion": "painted-timber", "pavilion_roof": "copper-sheet", "banner": "copper-sheet",
              "finial": "copper-sheet"}
FACES = {"f": "front", "b": "back", "l": "left", "r": "right"}
STOREY = 3.0        # one storey of windows
PARAPET = .6
WALLS = ("wing_wall", "palace_wall")


class Palace(Poem):
    PALETTE = {"core": (150, 140, 128, 255), "base_wall": (226, 221, 208, 255),
               "terrace_paving": (188, 180, 166, 255), "stair_step": (206, 200, 188, 255),
               "stair_landing": (200, 194, 182, 255), "wing_wall": (238, 234, 224, 255),
               "palace_wall": (142, 42, 38, 255), "frieze": (84, 26, 30, 255), "parapet": (232, 226, 212, 255),
               "roof_deck": (176, 160, 138, 255), "window_frame": (26, 22, 20, 255),
               "portal_frame": (26, 22, 20, 255), "portal_lintel": (96, 60, 38, 255),
               "range_roof": (120, 72, 44, 255), "gallery_post": (132, 44, 36, 255),
               "gallery_beam": (112, 66, 40, 255), "pavilion": (150, 36, 34, 255),
               "pavilion_roof": (214, 172, 62, 255), "banner": (214, 172, 62, 255), "finial": (226, 188, 76, 255)}
    TIMBER_KINDS = frozenset({"entablature", "column", "beam"})
    T = .7

    def compose(self):
        w, d, h = self.w, self.d, self.h
        self.base = base = clamp(.14 * h, 3.6, 5.0)
        self.crown_h = clamp(.12 * h, 3.4, 4.6)
        self.top = top = h - self.crown_h                           # roof of the highest block
        lean = .1 * base
        lane = clamp(.06 * w, 2.4, 3.2)
        zigzag = self.line("stair") == "zigzag"
        front = -d / 2 + (2 * lane if zigzag else base / .75)       # foot of the base behind the stair
        hz = (d / 2 - front) / 2
        self.base_warp = taper(0, front + hz, 0, base, (w / 2 - lean) / (w / 2), (hz - lean) / hz)
        self.podium("base_wall", -w / 2, w / 2, front, d / 2, 0, base, paving="terrace_paving",
                    warp=self.base_warp, grade="concept")
        if zigzag:
            self.zigzag(front, lane)
        else:
            self.flight("stair_step", clamp(.14 * w, 4.5, 7.0), -d / 2, front, 0, base)
            self.member("stair_landing", -2.0, 2.0, base - .3, base, front, front + lean + .3)
        # The plan of everything above the base, before the walls lean.
        X = w / 2 - lean - .9
        z0, z1 = front + lean + clamp(.1 * 2 * hz, 2.0, 3.0), d / 2 - lean - .9
        self.cz = (z0 + z1) / 2
        batter = .07 * (h - base) * self.vary("batter", .15)
        self.shrink = (1 - batter / X, 1 - batter / ((z1 - z0) / 2))
        self.warp = taper(0, self.cz, base, h, *self.shrink)
        self.frieze_h = .8 if self.line("frieze") == "single" else 1.3
        rise = top - base
        if self.variant == "potala":
            massing = self.potala(X, z0, z1, rise)
        else:
            massing = self.dzong(X, z0, z1, rise)
        return self.document({"region": REGION, "period": PERIOD, "base_height": round(base, 3),
                              "roof_height": round(top, 3), "wall_batter": round(batter / (h - base), 4),
                              "blind_storeys": 1, **massing,
                              "construction": "battered stone walls, whitewashed; flat earth roofs on timber"})

    # ------------------------------------------------------------ massing
    def potala(self, X, z0, z1, rise):
        cw = X * .36 * self.vary("central", .08)
        setback = 1.4 * self.vary("wing-setback", .2)
        if self.line("wings") == "stepped":
            mid = cw + (X - cw) * .5 * self.vary("wing-step", .08)
            parts = [(cw, mid, .78 * self.vary("inner-wing", .04)), (mid, X, .56 * self.vary("outer-wing", .05))]
        else:
            parts = [(cw, X, .68 * self.vary("wing", .05))]
        tops = [self.base + rise * share for _, _, share in parts]
        self.block("central", "palace_wall", -cw, cw, z0, z1, self.top, frieze=True, door=True,
                   hidden={"left": tops[0], "right": tops[0]})
        for side, outer in ((-1, "left"), (1, "right")):
            for i, ((a, b, _), top) in enumerate(zip(parts, tops)):
                xa, xb = sorted((side * a, side * b))
                below = {outer: tops[i + 1]} if i + 1 < len(parts) else {}
                self.block(f"{outer}-wing-{i}", "wing_wall", xa, xb, z0 + setback, z1, top,
                           faces="fb" + outer[0], hidden=below)
        self.crown(cw, (z1 - z0) / 2)
        return {"wings_per_side": len(parts), "wing_heights": [round(t, 3) for t in tops]}

    def dzong(self, X, z0, z1, rise):
        t = self.T
        depth = clamp(.16 * min(2 * X, z1 - z0), 3.2, 5.5)
        high = self.base + rise * .46 * self.vary("range", .05)
        low = self.base + rise * .36 * self.vary("front-range", .05) if self.line("wings") == "stepped" else high
        # Stone faces the world; the court side of every range is an open timber gallery.
        ranges = (("front-range", -X, X, z0, z0 + depth, low, "flr", "b"),
                  ("back-range", -X, X, z1 - depth, z1, high, "blr", "f"),
                  ("left-range", -X, -X + depth, z0 + depth - t, z1 - depth + t, high, "l", "r"),
                  ("right-range", X - depth, X, z0 + depth - t, z1 - depth + t, high, "r", "l"))
        for name, xa, xb, za, zb, top, faces, court in ranges:
            side = len(faces) == 1
            self.block(name, "wing_wall", xa, xb, za, zb, top, faces=faces, parapet=False, door=name == "front-range")
            self.gallery(name, (xa, xb, za, zb), court, top)
            self.pitched(name, xa, xb, za + (t if side else 0), zb - (t if side else 0), top, "z" if side else "x")
        uz = (z1 - z0) / 2 - depth - 1.6
        ux = min(X - depth - 1.6, uz * 1.5)
        self.block("central", "palace_wall", -ux, ux, self.cz - uz, self.cz + uz, self.top, frieze=True)
        self.crown(ux, uz)
        return {"range_depth": round(depth, 3), "range_heights": [round(low, 3), round(high, 3)],
                "utse_plan": [round(2 * ux, 3), round(2 * uz, 3)]}

    # ------------------------------------------------------------ members
    @staticmethod
    def at(key, plan, along, y, out):
        """A point on a block's face: ``along`` the face, at height y, ``out`` from its surface."""
        x0, x1, z0, z1 = plan
        return {"f": (along, y, z0 - out), "b": (along, y, z1 + out),
                "l": (x0 - out, y, along), "r": (x1 + out, y, along)}[key]

    def solid(self, role, points, **params):
        """A convex member given by its corners before the walls lean."""
        return self.mesh(role, hull(self.warp(np.asarray(points, dtype=float))), **params)

    def plate(self, role, key, plan, a0, a1, y0, y1, o0, o1, **params):
        return self.solid(role, [self.at(key, plan, a, y, o) for a in (a0, a1) for y in (y0, y1) for o in (o0, o1)],
                          **params)

    def block(self, name, role, x0, x1, z0, z1, top, faces="fblr", hidden=None, frieze=False, parapet=True,
              door=False):
        """One battered mass: blind foot, windows that grow upward, flat roof, parapet."""
        t, y0, plan, hidden = self.T, self.base, (x0, x1, z0, z1), hidden or {}
        wall_top = top - (self.frieze_h if frieze else 0.)
        rows = int((wall_top - y0 - .6) / STOREY)
        blind = max(1, rows // 3)
        lights = []
        for key in faces:
            a0, a1 = (x0, x1) if key in "fb" else (z0 + t, z1 - t)
            length, mid = a1 - a0, (a0 + a1) / 2
            bays = int((length - 1.6) / 2.7)
            for row in range(blind, rows):
                grown = row - blind
                width, height = min(.7 + .18 * grown, 1.3), min(1.0 + .28 * grown, STOREY - 1.2)
                sill = y0 + row * STOREY + .8
                if sill < hidden.get(FACES[key], 0.) + .3:
                    continue
                for bay in range(bays):
                    offset = (bay + .5) * length / bays - length / 2
                    lights.append((key, offset, mid + offset, width, sill, height, row, bay))
        # A face that stands behind a lower neighbour is not built below that neighbour's roof.
        split = y0 + int((min(hidden.values(), default=y0) - y0) / STOREY) * STOREY
        covered = "".join(key for key in faces if FACES[key] in hidden)
        lifts = [(y0, wall_top, faces)] if split <= y0 else \
            [(y0, split, "".join(k for k in faces if k not in covered)), (split, wall_top, faces)]
        tags = {"block": name, "warp": self.warp, "grade": "concept"}
        for lift, (ya, yb, built) in enumerate(lifts):
            openings = {}
            for key in built:
                voids = Voids(ya)
                for face, offset, _, width, sill, height, row, bay in lights:
                    if face == key and ya <= sill < yb:
                        opening = f"{name}.{key}.{row}.{bay}"
                        voids.rect(offset, width, sill, height, opening)
                        voids.rect(offset, width + .2, sill + height, .2, opening + ".seat")
                if door and key == "f" and not lift:
                    voids.rect(0., 1.5, y0, 2.4, f"{name}.door")
                    voids.rect(0., 2.0, y0 + 2.4, .25, f"{name}.door.seat")
                openings[FACES[key]] = voids.specs
            self.shell(role, x0, x1, z0, z1, ya, yb, t, openings, built, lift=lift, **tags)
        if frieze:
            bands = ((0., .8),) if self.line("frieze") == "single" else ((0., .5), (.8, 1.3))
            for a, b in bands:
                self.shell("frieze", x0, x1, z0, z1, wall_top + a, wall_top + b, t, faces=faces, **tags)
            if len(bands) == 2:                                  # a whitewashed course parts the two bands
                self.shell("parapet", x0, x1, z0, z1, wall_top + .5, wall_top + .8, t, faces=faces,
                           course_between="frieze", **tags)
        if parapet:
            self.shell("parapet", x0, x1, z0, z1, top, top + PARAPET, t, faces=faces, **tags)
        self.slab("roof_deck", x0 + t, x1 - t, z0 + t, z1 - t, top - .25, top, module=2.4, warp=self.warp,
                  block=name)
        for key, _, at, width, sill, height, row, _ in lights:
            tags = {"block": name, "row": row, "face": key}
            half = width / 2
            self.plate("window_frame", key, plan, at - half - .1, at + half + .1, sill + height, sill + height + .2,
                       -t, .08, member="head", **tags)
            self.plate("window_frame", key, plan, at - half - .3, at + half + .3, sill - .12, sill, 0., .1,
                       member="sill", **tags)
            for side in (-1, 1):                                 # the surround spreads toward the sill
                inner = at + side * half
                corners = [(inner, sill), (inner + side * .3, sill), (inner, sill + height),
                           (inner + side * .1, sill + height)]
                self.solid("window_frame", [self.at(key, plan, a, y, o) for a, y in corners for o in (0., .08)],
                           member="jamb", **tags)
        if door:
            head = y0 + 2.4
            self.plate("portal_lintel", "f", plan, -1.0, 1.0, head, head + .25, -t, .1, block=name)
            for side in (-1, 1):
                corners = [(side * .75, y0), (side * 1.15, y0), (side * .75, head), (side * .9, head)]
                self.solid("portal_frame", [self.at("f", plan, a, y, o) for a, y in corners for o in (0., .1)],
                           member="jamb", block=name)

    def gallery(self, name, plan, key, top):
        """The court side of a range: timber posts, and a beam at every floor."""
        x0, x1, z0, z1 = plan
        a0, a1 = (x0, x1) if key in "fb" else (z0, z1)
        bays = max(1, round((a1 - a0) / 3.0))
        marks = [a0 + .12 + (a1 - a0 - .24) * i / bays for i in range(bays + 1)]
        floors = [self.base + STOREY * k for k in range(1, int((top - .5 - self.base) / STOREY) + 1)] + [top - .25]
        for at in marks:
            self.plate("gallery_post", key, plan, at - .12, at + .12, self.base, top - .25, -.24, 0., block=name)
        for a, b in zip(marks, marks[1:]):
            for level, y in enumerate(floors):
                self.plate("gallery_beam", key, plan, a + .12, b - .12, y - .22, y, -.22, -.02, block=name, level=level)

    def pitched(self, name, x0, x1, z0, z1, eave, ridge_axis):
        """A low timber roof over one range, laid in lengths; its eaves overhang the wall."""
        over, rise = .5, 1.5
        (a0, a1), (c0, c1) = ((x0, x1), (z0, z1)) if ridge_axis == "x" else ((z0, z1), (x0, x1))
        if ridge_axis == "x":
            a0, a1 = a0 - over, a1 + over
        pieces = max(1, math.ceil((a1 - a0) / 3.2))
        for i in range(pieces):
            ends = (a0 + (a1 - a0) * i / pieces, a0 + (a1 - a0) * (i + 1) / pieces)
            section = ((c0 - over, eave), (c1 + over, eave), ((c0 + c1) / 2, eave + rise))
            points = [(a, y, c) if ridge_axis == "x" else (c, y, a) for a in ends for c, y in section]
            self.solid("range_roof", points, block=name)

    def crown(self, half_x, half_z):
        """What stands on the highest roof; the tip of the tallest finial is the full height."""
        y, c, cz = self.top, self.crown_h, self.cz
        u = (y - self.base) / (self.h - self.base)
        ax, az = (half * (1 + (scale - 1) * u) - self.T for half, scale in zip((half_x, half_z), self.shrink))
        a = min(ax, az)
        if self.line("roofs") == "banners":
            self.drum("banner", 0, cz, y, y + .86 * c, .5, sides=10, rank=0)
            self.drum("finial", 0, cz, y + .86 * c, y + c, .16, .04, sides=8)
            for i, (sx, sz) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1)), 1):
                self.drum("banner", sx * (ax - .7), cz + sz * (az - .7), y, y + .6 * c, .32, sides=10, rank=i)
            return
        spots = [(0., 1.0)] + ([(-.62 * ax, .58), (.62 * ax, .58)] if ax > 1.9 * a * .36 + 1.2 else [])
        for i, (x, scale) in enumerate(spots):
            half, tall = a * .36 * scale, c * scale
            self.member("pavilion", x - half, x + half, y, y + .38 * tall, cz - half, cz + half, rank=i)
            self.conic("pavilion_roof", x, cz, y + .38 * tall, y + .66 * tall, (half + .7 * scale) * math.sqrt(2),
                       half * .6 * math.sqrt(2), sides=4, rot=math.pi / 4, rank=i, tier=0)
            self.conic("pavilion_roof", x, cz, y + .66 * tall, y + .9 * tall, half * .75 * math.sqrt(2), .12,
                       sides=4, rot=math.pi / 4, rank=i, tier=1)
            self.drum("finial", x, cz, y + .9 * tall, y + tall, .14, .04, sides=8, rank=i)

    def zigzag(self, front, lane):
        """Two ramped flights along the face of the base: out along the plot edge, then back to the gate."""
        base, edge = self.base, -self.d / 2
        mid = edge + lane
        reach = min(.3 * self.w, self.w / 2 - lane - 1.0)
        steps = math.ceil(base / .2)
        turn = steps // 2

        def tread(x0, x1, z0, z1, top, **params):
            x0, x1 = sorted((x0, x1))
            points = np.array([(x, y, z) for x in (x0, x1) for y in (0., top) for z in (z0, z1)])
            self.mesh("stair_step", hull(self.base_warp(points)), construction="solid stair", **params)

        for j in range(turn):                                    # the outer flight climbs to the right
            tread(-reach + 2 * reach * j / turn, -reach + 2 * reach * (j + 1) / turn, edge, mid,
                  base * (j + 1) / steps, step=j, flight="outer")
        tread(reach, reach + lane, edge, front, base * turn / steps, step=turn, flight="turn")
        upper = steps - turn - 1
        for j in range(upper):                                   # the inner flight returns to the gate
            xa = reach - (reach - 1.5) * j / upper
            xb = reach - (reach - 1.5) * (j + 1) / upper
            tread(xa, xb, mid, front, base * (turn + j + 1) / steps, step=turn + 1 + j, flight="inner")
        tread(-1.5, 1.5, mid, front, base, step=steps, flight="gate")


Palace.module = sys.modules[__name__]
random_controls = controls_for(sys.modules[__name__])


def build_document(request):
    return Palace(request).compose()


def _blocks(placements, roles):
    groups = defaultdict(list)
    for p in placements:
        if p.role in roles and p.parameters.get("block"):
            groups[p.parameters["block"]].append(p)
    return groups


def _top(p):
    return p.position[1] + p.dimensions[1] / 2


def rule_checks(document, form):
    placements = document.placements
    blocks = _blocks(placements, WALLS)
    boxes = {name: extent(stones) for name, stones in blocks.items()}

    plumb = []
    for name, stones in blocks.items():
        lo, hi = boxes[name]
        foot = extent([p for p in stones if p.position[1] < lo[1] + 1.0])
        head = extent([p for p in stones if p.position[1] > hi[1] - 1.0])
        spans = [(foot[1][a] - foot[0][a]) - (head[1][a] - head[0][a]) for a in (0, 2)]
        if min(spans) < -.01 or max(spans) < .05:
            plumb += [p.id for p in stones[:1]]
    yield ("battered_walls", len(blocks) >= 2 and not plumb, "Every block is narrower at its head than at its foot.",
           "A block is missing, or one of its walls is plumb or leans outward.", plumb)

    frames = _blocks(placements, {"window_frame"})
    wrong = []
    for name, pieces in frames.items():
        if name not in boxes:
            wrong += [p.id for p in pieces]
            continue
        rows = defaultdict(lambda: [0., 0.])
        for p in pieces:
            member, row = p.parameters.get("member"), p.parameters.get("row")
            if member == "jamb":
                rows[row][1] = max(rows[row][1], p.dimensions[1])
                if p.position[1] - p.dimensions[1] / 2 < boxes[name][0][1] + 2.5:
                    wrong.append(p.id)                           # a window in the blind storey
            elif member == "head":
                rows[row][0] = max(rows[row][0], max(p.dimensions[0], p.dimensions[2]))
        areas = [rows[row][0] * rows[row][1] for row in sorted(rows, key=lambda r: (r is None, r))]
        if any(b < .9 * a for a, b in zip(areas, areas[1:])):          # the lean itself narrows a capped light
            wrong += [p.id for p in pieces[:1]]
    yield ("windows_grow", bool(frames) and not wrong, "The lowest storey is blind and windows grow toward the roof.",
           "A window opens in the lowest storey, or windows shrink going up.", wrong)

    central = blocks.get("central", [])
    crest = max(map(_top, central), default=0.)
    rivals = [name for name, stones in blocks.items() if name != "central" and max(map(_top, stones)) > crest - 1.5]
    yield ("central_block", bool(central) and len(blocks) >= 2 and not rivals,
           "The central block rises above every wing.", "The central block is missing or a wing rises as high.",
           [blocks[name][0].id for name in rivals])

    frieze = [p for p in placements if p.role == "frieze" and p.parameters.get("block") == "central"]
    parapet = [p for p in placements if p.role == "parapet" and p.parameters.get("block") == "central"
               and not p.parameters.get("course_between")]
    banded = bool(frieze and parapet and central)
    if banded:
        foot = min(p.position[1] - p.dimensions[1] / 2 for p in parapet)
        banded = abs(max(map(_top, frieze)) - foot) < .05 and \
            min(p.position[1] - p.dimensions[1] / 2 for p in frieze) > crest - .05
    yield ("frieze_band", banded, "The dark frieze lies above the wall and directly under the parapet.",
           "The frieze is missing or does not lie directly under the central parapet.", [p.id for p in frieze[:3]])

    tops = sorted(_top(p) for p in placements if p.role == "stair_step")
    sills = [p.position[1] - p.dimensions[1] / 2 for p in placements if p.role == "portal_frame"]
    climbs = bool(tops and sills) and tops[0] < .35 and all(b - a < .35 for a, b in zip(tops, tops[1:])) \
        and abs(tops[-1] - min(sills)) < .06
    yield ("approach_stair", climbs, "The stair climbs without a break from the ground to the entrance.",
           "The stair is broken, or stops short of the entrance.", [])
