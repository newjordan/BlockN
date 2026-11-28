"""Lalibela architect: the rock-hewn church.

A church at Lalibela is not built. A trench is cut down into a slope of volcanic
rock until a block stands free in a pit, and the block is then hollowed into a
church: plinth, walls, pillars and roof are one stone, still joined to the floor
of the pit. The roof is the old ground surface, so the building is seen first
from above. Its rules follow from that: it stands wholly inside its pit, it
never rises above the rock around it, and the only way to it is down.

``cruciform`` follows Bete Giyorgis: a Greek cross of four equal arms on a
stepped plinth, with nested crosses in relief on the roof. ``colonnaded`` follows
Bete Medhane Alem: a rectangular church ringed by square piers under its roof.

Here y=0 is the floor of the pit and y=height is the natural ground.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

from .historic import clamp, extent, hull
from .schema import ArchitectInfo
from .sonnet import Poem, controls_for, form

ARCHITECT = ArchitectInfo("lalibela", "1.0.0", "Lalibela Rock-Hewn Architect")
STYLE = "lalibela-rock-hewn"
PRODUCT = "lalibela-church"
REGION = "Lalibela, Lasta, Amhara highlands (Ethiopia)"
PERIOD = "Zagwe dynasty, traditionally the reign of King Lalibela, about 1200 CE"
VARIANTS = ("cruciform", "colonnaded")
KINDS = {
    "bedrock": "foundation", "pit_floor": "slab", "stair_step": "circulation", "tunnel_lintel": "monolith",
    "plinth_step": "platform", "entry_step": "circulation", "hewn_band": "monolith", "blind_panel": "ornament",
    "window_frame": "ornament", "pillar": "column", "colonnade_pier": "column", "church_roof": "roof",
    "roof_relief": "ornament",
}
REQUIRED = {"bedrock", "pit_floor", "stair_step", "plinth_step", "hewn_band", "pillar", "church_roof", "roof_relief"}
CHURCH = {"plinth_step", "entry_step", "hewn_band", "blind_panel", "window_frame", "pillar", "colonnade_pier",
          "church_roof", "roof_relief"}
FORMS = {PRODUCT: form(
    PRODUCT, "Monolithic church hewn downward from living rock, standing in its pit below the ground surface",
    (30, 12, 40), VARIANTS, REQUIRED,
    lines={"windows": ("ogee", "cross", "aksumite"), "relief": ("nested-crosses", "arcade"),
           "approach": ("open-stair", "tunnel"), "plinth": ("three-step", "two-step")},
    size={"width": (26, 38), "height": (10, 14), "depth": (36, 52)},
    ratios={"depth/width": (1.25, 1.6), "height/width": (.3, .45)},
    region=REGION, period=PERIOD, datum="y=0 is the pit floor; y=height is the natural ground surface",
    openings="hewn windows and one door; nothing is arched or assembled")}
DEFAULT_SUBSTANCE = "tuff"
SUBSTANCES = {}
FLOOR = .3          # the dressed floor of the pit
WALL = 1.2          # what is left of the block as its wall
OUT, RISE = .4, .5  # tread and rise of a plinth step
ROOF, RELIEF = .6, .25
TRENCH = 2.0        # the least clear trench between church and pit face
HEWN = "living rock monolith"


def cross_pieces(length, half):
    """A Greek cross as three rectangles (x0, x1, z0, z1) that do not overlap."""
    return [(-half, half, -length, length), (-length, -half, -half, half), (half, length, -half, half)]


def run(axis, a0, a1, inner, outer, e0=0, e1=0, door=False):
    return {"axis": axis, "a0": a0, "a1": a1, "inner": inner, "outer": outer, "e0": e0, "e1": e1, "door": door}


def cross_runs(length, half, t):
    """The wall of a hollow Greek cross as twelve straight runs that meet without overlapping."""
    runs = [run("x", -half, half, -length + t, -length, 1, 1, door=True), run("x", -half, half, length - t, length, 1, 1),
            run("z", -half, half, -length + t, -length, 1, 1), run("z", -half, half, length - t, length, 1, 1)]
    for side in (-1, 1):
        runs += [run("z", half - t, length - t, side * (half - t), side * half),
                 run("z", -(length - t), -(half - t), side * (half - t), side * half),
                 run("x", half, length - t, side * (half - t), side * half, e0=-1),
                 run("x", -(length - t), -half, side * (half - t), side * half, e1=-1)]
    return runs


def rect_runs(hx, hz, t):
    return [run("x", -hx, hx, -hz + t, -hz, 1, 1, door=True), run("x", -hx, hx, hz - t, hz, 1, 1),
            run("z", -hz + t, hz - t, -hx + t, -hx), run("z", -hz + t, hz - t, hx - t, hx)]


def cuts(low, high, marks, module=4.0):
    """Joints from low to high that fall on every mark and are never further apart than a module."""
    edges = sorted({low, high, *(m for m in marks if low < m < high)})
    out = [low]
    for a, b in zip(edges, edges[1:]):
        count = max(1, math.ceil((b - a) / module - 1e-9))
        out += [a + (b - a) * (k + 1) / count for k in range(count)]
    return out


class Church(Poem):
    PALETTE = {"bedrock": (150, 96, 72, 255), "pit_floor": (132, 86, 66, 255), "stair_step": (140, 90, 68, 255),
               "tunnel_lintel": (144, 92, 70, 255), "plinth_step": (160, 100, 76, 255),
               "entry_step": (150, 96, 72, 255), "hewn_band": (172, 108, 82, 255), "blind_panel": (128, 78, 60, 255),
               "window_frame": (186, 122, 94, 255), "pillar": (160, 100, 76, 255),
               "colonnade_pier": (176, 112, 86, 255), "church_roof": (166, 104, 80, 255),
               "roof_relief": (184, 120, 92, 255)}

    def compose(self):
        w, h, d = self.w, self.h, self.d
        margin = clamp(.13 * w, 3.2, 4.5)
        stretch = 1.3 if self.variant == "colonnaded" else 1.0
        length = min(d - margin - h / .88, 2 * (w / 2 - margin) * stretch)
        px = min(w / 2 - margin, length / 2)
        trench = max(TRENCH + .1, clamp(.1 * w, 2.2, 3.2) * self.vary("trench", .1))
        if px - trench < 5.5:
            raise ValueError(f"{PRODUCT} has no room for a church and its trench on this plot")
        z1 = d / 2 - margin
        z0 = z1 - length
        stair = clamp(.09 * w, 2.4, 3.0)
        self.steps = 3 if self.line("plinth") == "three-step" else 2
        self.rock(px, z0, z1, stair)
        self.slab("pit_floor", -px, px, z0, z1, 0, FLOOR, module=3.5, construction=HEWN)
        self.flight("stair_step", stair, z0, -d / 2, FLOOR, h, base=0)
        if self.line("approach") == "tunnel":
            self.tunnel(stair, z0, d / 2 + z0)
        start = len(self.placements)
        if self.variant == "cruciform":
            arm = self.cruciform(px - trench)
        else:
            arm = self.colonnaded(px - trench, length / 2 - trench)
        self.shift(start, 0, (z0 + z1) / 2)
        return self.document({"region": REGION, "period": PERIOD, "pit_depth": h, "trench": round(trench, 3),
                              "plinth_steps": self.steps, "arm_width": arm,
                              "construction": "hewn downward from living rock; nothing is assembled"})

    # ------------------------------------------------------------ the rock
    def rock(self, px, z0, z1, stair):
        """The living rock round the pit and the stair trench, in cells that reach the pit floor level."""
        w, h, d = self.w, self.h, self.d
        xs = cuts(-w / 2, w / 2, (-px, px, -stair / 2, stair / 2))
        zs = cuts(-d / 2, d / 2, (z0, z1))
        lifts = max(2, math.ceil(h / 4.0))
        for xa, xb in zip(xs, xs[1:]):
            for za, zb in zip(zs, zs[1:]):
                cx, cz = (xa + xb) / 2, (za + zb) / 2
                if (abs(cx) < px and z0 < cz < z1) or (abs(cx) < stair / 2 and cz < z0):
                    continue
                # A cell that faces the pit or the stair trench is seen; the rest is the mass behind.
                faces = any(min(xb - x0, x1 - xa) > -1e-6 and min(zb - c0, c1 - za) > -1e-6 and
                            max(min(xb, x1) - max(xa, x0), min(zb, c1) - max(za, c0)) > 1e-6
                            for x0, x1, c0, c1 in ((-px, px, z0, z1), (-stair / 2, stair / 2, -d / 2, z0)))
                for j in range(lifts):
                    hidden = not faces and j < lifts - 1
                    self.member("bedrock", xa, xb, h * j / lifts, h * (j + 1) / lifts, za, zb, lift=j,
                                **({"construction": "living rock mass", "unit": "core"} if hidden else
                                   {"construction": HEWN}))

    def tunnel(self, stair, z0, run_length):
        """Rock left standing over the foot of the stair, so the pit is entered through a tunnel."""
        covered = run_length * .38
        count = max(1, math.ceil(covered / 1.6))
        for k in range(count):
            za, zb = z0 - covered * (k + 1) / count, z0 - covered * k / count
            clear = FLOOR + (self.h - FLOOR) * (z0 - za) / run_length + 2.3
            if self.h - clear >= .8:
                self.member("tunnel_lintel", -stair / 2, stair / 2, clear, self.h, za, zb, construction=HEWN)

    # ------------------------------------------------------------ the church
    def box(self, role, axis, a0, a1, y0, y1, c0, c1, **params):
        if axis == "x":
            return self.member(role, a0, a1, y0, y1, c0, c1, construction=HEWN, **params)
        return self.member(role, c0, c1, y0, y1, a0, a1, construction=HEWN, **params)

    def band(self, runs, y0, y1, proud=0., gaps=None, role="hewn_band", **params):
        """One horizontal band of every wall run; ``gaps`` names which openings are cut through it."""
        for index, r in enumerate(runs):
            sign = 1 if r["outer"] > r["inner"] else -1
            a0, a1 = r["a0"] - r["e0"] * proud, r["a1"] + r["e1"] * proud
            c0, c1 = sorted((r["inner"], r["outer"] + sign * proud))
            voids = []
            if gaps in ("door", "base") and r["door"]:
                voids.append((0., 1.2))
            if gaps in ("lower", "upper"):
                voids += [(c, .8) for c in r["windows"]]
            edge = a0
            for centre, width in sorted(voids):
                if centre - width / 2 - edge > .05:
                    self.box(role, r["axis"], edge, centre - width / 2, y0, y1, c0, c1, run=index, **params)
                edge = centre + width / 2
            if a1 - edge > .05:
                self.box(role, r["axis"], edge, a1, y0, y1, c0, c1, run=index, **params)

    def windows(self, runs):
        for r in runs:
            length = r["a1"] - r["a0"]
            count = int(length // 2.8) or (1 if length >= 1.8 else 0)
            centres = [r["a0"] + (i + .5) * length / count for i in range(count)]
            r["windows"] = [c for c in centres if not r["door"] or abs(c) > 1.5]

    def walls(self, runs, y0, y1):
        """An Aksumite facade is banded: plain base, blind windows, string course, lights, cornice."""
        self.windows(runs)
        height = y1 - y0
        marks = [y0 + height * f for f in (0, .28, .45, .50, .78, .90, 1.0)]
        door = min(2.3, marks[1] - y0)
        self.band(runs, y0, y0 + door, gaps="door", register="base")
        if marks[1] - y0 - door > .25:
            self.band(runs, y0 + door, marks[1], register="base-head")
        self.band(runs, marks[1], marks[2], gaps="lower", register="blind-windows")
        self.band(runs, marks[2], marks[3], proud=.12, register="string-course")
        self.band(runs, marks[3], marks[4], gaps="upper", register="windows")
        self.band(runs, marks[4], marks[5], register="frieze")
        self.band(runs, marks[5], marks[6], proud=.1, register="cornice")
        style = self.line("windows")
        for r in runs:
            sign = 1 if r["outer"] > r["inner"] else -1
            axis, inner, outer = r["axis"], r["inner"], r["outer"]
            mid = (inner + outer) / 2
            for c in r["windows"]:
                # The lower window is blind: the rock is left in it, set back from the face.
                self.box("blind_panel", axis, c - .4, c + .4, marks[1], marks[2], *sorted((inner, mid)))
                lo, hi = marks[3], marks[4]
                if style == "cross":
                    third = sorted((inner + (outer - inner) / 3, inner + 2 * (outer - inner) / 3))
                    centre = (lo + hi) / 2
                    self.box("window_frame", axis, c - .4, c + .4, centre - .09, centre + .09, *third, member="bar")
                    self.box("window_frame", axis, c - .09, c + .09, lo, centre - .09, *third, member="stem")
                    self.box("window_frame", axis, c - .09, c + .09, centre + .09, hi, *third, member="stem")
                elif style == "ogee":
                    for side in (-1, 1):
                        profile = [(c + side * .4, hi), (c + side * .4, hi - .7), (c, hi)]
                        points = [(a, y, k) if axis == "x" else (k, y, a) for a, y in profile for k in (inner, outer)]
                        self.mesh("window_frame", hull(points), member="ogee", construction=HEWN)
                else:
                    for along in (c - .51, c + .51):
                        for y in (lo + .11, hi - .11):
                            self.box("window_frame", axis, along - .11, along + .11, y - .11, y + .11,
                                     *sorted((outer, outer + sign * .15)), member="monkey-head")

    def entry(self, fronts):
        """Half-steps at the door, one in front of each plinth step."""
        for k, z in enumerate(fronts):
            self.member("entry_step", -.8, .8, FLOOR + k * RISE, FLOOR + (k + .5) * RISE, z - OUT / 2, z,
                        step=k, construction=HEWN)

    def ribs(self, axis, a0, a1, c0, c1, y0, y1):
        count = max(1, round((a1 - a0) / 1.6))
        for i in range(count):
            mid = a0 + (i + .5) * (a1 - a0) / count
            self.box("roof_relief", axis, mid - .15, mid + .15, y0, y1, c0, c1, member="rib")

    def cruciform(self, half):
        n = self.steps
        length = half - n * OUT
        arm = max(2.5, length * .40 * self.vary("arm", .08))
        for k in range(n):
            for x0, x1, za, zb in cross_pieces(length + (n - k) * OUT, arm + (n - k) * OUT):
                self.member("plinth_step", x0, x1, FLOOR + k * RISE, FLOOR + (k + 1) * RISE, za, zb, step=k,
                            construction=HEWN)
        self.entry([-(length + (n - k) * OUT) for k in range(n)])
        base, eaves, top = FLOOR + n * RISE, self.h - ROOF - RELIEF, self.h - RELIEF
        self.walls(cross_runs(length, arm, WALL), base, eaves)
        inner = arm - WALL
        for sx in (-1, 1):                      # pilasters stand in the four corners of the crossing
            for sz in (-1, 1):
                self.member("pillar", *sorted((sx * inner, sx * (inner - .6))), base, eaves,
                            *sorted((sz * inner, sz * (inner - .6))), construction=HEWN)
        lr, hr = length + .1, arm + .1
        for x0, x1, za, zb in cross_pieces(lr, hr):
            self.member("church_roof", x0, x1, eaves, top, za, zb, construction=HEWN)
        self.band(cross_runs(lr, hr, .5), top, self.h, role="roof_relief", member="border")
        if self.line("relief") == "nested-crosses":
            self.band(cross_runs(lr - 1.1, hr - 1.1, .45), top, self.h, role="roof_relief", member="cross")
            for x0, x1, za, zb in cross_pieces(lr - 2.4, .3):
                self.member("roof_relief", x0, x1, top, self.h, za, zb, member="cross", construction=HEWN)
        else:
            self.ribs("z", -(lr - .5), lr - .5, -(hr - .5), hr - .5, top, self.h)
            for side in (-1, 1):
                self.ribs("x", *sorted((side * (hr + .2), side * (lr - .5))), -(hr - .5), hr - .5, top, self.h)
        return round(2 * arm, 3)

    def colonnaded(self, hx, hz):
        n = self.steps
        for k in range(n):
            self.member("plinth_step", -(hx - k * OUT), hx - k * OUT, FLOOR + k * RISE, FLOOR + (k + 1) * RISE,
                        -(hz - k * OUT), hz - k * OUT, step=k, construction=HEWN)
        self.entry([-(hz - k * OUT) for k in range(n)])
        base, eaves, top = FLOOR + n * RISE, self.h - ROOF - RELIEF, self.h - RELIEF
        rx, rz = hx - (n - 1) * OUT - .3, hz - (n - 1) * OUT - .3
        cx, cz = rx - .55, rz - .55             # the line of the colonnade
        for axis, span, cross in (("x", cx, cz), ("z", cz, cx)):
            count = max(2, round(2 * span / 2.4))
            for i in range(count + 1) if axis == "x" else range(1, count):
                along = -span + 2 * span * i / count
                for side in (-1, 1):
                    self.box("colonnade_pier", axis, along - .4, along + .4, base, eaves,
                             side * cross - .4, side * cross + .4, side=f"{axis}{side:+d}")
        aisle = 1.4 * self.vary("aisle", .1)
        wx, wz = cx - .4 - aisle, cz - .4 - aisle
        self.walls(rect_runs(wx, wz, WALL), base, eaves)
        rows = max(1, round(2 * (wz - WALL - 1.0) / 3.2))
        for i in range(rows):
            z = -(wz - WALL - 1.0) + (i + .5) * 2 * (wz - WALL - 1.0) / rows
            for side in (-1, 1):
                self.member("pillar", side * wx * .38 - .4, side * wx * .38 + .4, base, eaves, z - .4, z + .4,
                            construction=HEWN)
        self.member("church_roof", -rx, rx, eaves, top, -rz, rz, construction=HEWN)
        self.band(rect_runs(rx, rz, .5), top, self.h, role="roof_relief", member="border")
        if self.line("relief") == "nested-crosses":
            size = min(rx - 1.3, 3.0)
            count = max(1, int((2 * rz - 2.0) // (2 * size + 1.0)))
            for i in range(count):
                z = -(rz - 1.0) + (i + .5) * 2 * (rz - 1.0) / count
                for x0, x1, za, zb in cross_pieces(size, .3):
                    self.member("roof_relief", x0, x1, top, self.h, za + z, zb + z, member="cross", construction=HEWN)
        else:
            self.ribs("z", -(rz - .5), rz - .5, -(rx - .5), rx - .5, top, self.h)
        return None


Church.module = sys.modules[__name__]
random_controls = controls_for(sys.modules[__name__])


def build_document(request):
    return Church(request).compose()


def rule_checks(document, form):
    roles = defaultdict(list)
    for p in document.placements:
        roles[p.role].append(p)
    church = [p for p in document.placements if p.role in CHURCH]
    rock, pit = roles["bedrock"], roles["pit_floor"]
    surface = extent(rock)[1][1] if rock else 0.
    variant = church[0].parameters.get("style") if church else None

    free, intruders = False, []
    if church and pit and rock:
        (lo, hi), (pit_lo, pit_hi) = extent(church), extent(pit)
        free = min(min(lo[a] - pit_lo[a], pit_hi[a] - hi[a]) for a in (0, 2)) >= TRENCH - 1e-6
        for p in (*rock, *roles["tunnel_lintel"], *roles["stair_step"]):
            if all(p.position[a] - p.dimensions[a] / 2 < pit_hi[a] - 1e-4 and
                   p.position[a] + p.dimensions[a] / 2 > pit_lo[a] + 1e-4 for a in (0, 2)):
                intruders.append(p.id)
    yield ("freestanding_in_pit", free and not intruders,
           "The church stands wholly inside its pit with a clear trench on every side.",
           "The church is missing, crowds the pit face, or rock is left standing in the trench.", intruders)

    roof = roles["church_roof"] + roles["roof_relief"]
    high = [p.id for p in church if p.position[1] + p.dimensions[1] / 2 > surface + 1e-4]
    level = bool(roof) and abs(extent(roof)[1][1] - surface) < .02
    yield ("roof_at_ground", level and not high, "The roof is level with the surrounding rock and nothing rises above it.",
           "The church rises above the rock it was cut from, or its roof lies below the ground surface.", high)

    steps = defaultdict(list)
    for p in roles["plinth_step"]:
        steps[p.parameters.get("step")].append(p)
    boxes = [extent(steps[k]) for k in sorted(k for k in steps if isinstance(k, int))]
    bad = [k for k, ((lo, hi), (lo2, hi2)) in enumerate(zip(boxes, boxes[1:]), 1)
           if not (abs(lo2[1] - hi[1]) < 1e-4 and all(lo2[a] > lo[a] + .05 and hi2[a] < hi[a] - .05 for a in (0, 2)))]
    yield ("stepped_plinth", len(boxes) >= 2 and not bad, "Each plinth step stands on the one below and inside its edge.",
           "A plinth step is missing, overhangs the step below, or does not rest on it.",
           [p.id for k in bad for p in steps[k]])

    bands = roles["hewn_band"]
    if variant == "colonnaded":
        piers = roles["colonnade_pier"]
        ringed = False
        if piers and bands:
            xs, zs = [p.position[0] for p in piers], [p.position[2] for p in piers]
            ringed = True
            for axis, edge in ((0, min(xs)), (0, max(xs)), (2, min(zs)), (2, max(zs))):
                other = 2 - axis
                line = sorted(p.position[other] for p in piers if abs(p.position[axis] - edge) < .05)
                ends = (min(zs), max(zs)) if axis == 0 else (min(xs), max(xs))
                ringed &= len(line) >= 3 and abs(line[0] - ends[0]) < .05 and abs(line[-1] - ends[1]) < .05 \
                    and all(b - a <= 3.2 for a, b in zip(line, line[1:]))
            lo, hi = extent(bands)
            ringed &= lo[0] > min(xs) and hi[0] < max(xs) and lo[2] > min(zs) and hi[2] < max(zs)
        yield ("plan", ringed, "An unbroken colonnade of piers rings the church on all four sides.",
               "The colonnade is missing, broken, or does not enclose the church.", [])
    else:
        cross = False
        if bands:
            lo, hi = extent(bands)
            cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
            widths = []
            for axis, edge, sign in ((2, hi[2], 1), (2, lo[2], -1), (0, hi[0], 1), (0, lo[0], -1)):
                tip = [p for p in bands if abs(p.position[axis] + sign * p.dimensions[axis] / 2 - edge) < .2]
                tip_lo, tip_hi = extent(tip)
                widths.append(tip_hi[2 - axis] - tip_lo[2 - axis])
            arm = max(widths)
            corners = [p for p in bands if abs(p.position[0] - cx) > arm / 2 + .2 and abs(p.position[2] - cz) > arm / 2 + .2]
            cross = abs((hi[0] - lo[0]) - (hi[2] - lo[2])) < .05 and arm - min(widths) < .05 \
                and arm < .7 * (hi[0] - lo[0]) and not corners
        yield ("plan", cross, "The plan is a Greek cross of four equal arms.",
               "The arms of the cross are unequal, missing, or the corners between them are filled.", [])

    tops = sorted(p.position[1] + p.dimensions[1] / 2 for p in roles["stair_step"])
    floor = extent(pit)[1][1] if pit else 0.
    descends = bool(tops) and tops[0] - floor < .26 and abs(tops[-1] - surface) < .02 and \
        all(b - a < .26 for a, b in zip(tops, tops[1:]))
    yield ("descending_stair", descends, "The rock-cut stair descends without a break from the ground to the pit floor.",
           "The stair is broken or does not reach from the ground to the pit floor.", [])
