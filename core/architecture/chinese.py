"""Chinese hall architect: timber halls on stone terraces, Ming and Qing.

A hall is a frame standing clear of the ground on a stone terrace. Bays are
counted across the front and are always odd, so the door sits on the axis; the
middle bay is the widest and they shorten outward. Columns on stone bases carry
architraves and a flat plate; on the plate stand the bracket sets (dougong)
that throw the eaves far past the columns. The roof is laid in courses whose
slope steepens toward the ridge, and every eave lifts and shoots out at its
corners.

Proportions follow the Song rule that a column stands no higher than its bay is
wide, and the Qing doukou scale for the bracket work: a bracket set steps out
three doukou at a jump, rafters reach 21 and the fliers 28 doukou past the eave
column, and a corner shoots three doukou out and four doukou up.

Timber members are long pieces, as timber is. Only the terrace is coursed stone.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .historic import (Composer, Form, TAU, clamp, extent, flare, hull, oriented_profile, taper, unmirrored)
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("chinese", "1.0.0", "Chinese Architect")
STYLE = "chinese"
KINDS = {
    "core_fill": "core",
    "terrace_face": "foundation", "terrace_paving": "slab", "step": "circulation", "ramp": "circulation",
    "baluster_post": "ornament", "baluster_panel": "ornament", "spout": "ornament",
    "column_base": "foundation", "column": "column", "inner_column": "column",
    "architrave": "entablature", "plate": "entablature", "cross_beam": "beam",
    "bracket_block": "bracket", "bracket_arm": "bracket", "purlin": "beam", "king_post": "column",
    "rafter": "beam", "flying_rafter": "beam", "corner_beam": "beam",
    "roof_tile": "roof", "main_ridge": "roof", "hip_ridge": "roof", "surround_ridge": "roof",
    "chiwen": "ornament", "beast": "ornament",
    "end_wall": "wall", "sill_wall": "wall", "upper_wall": "wall",
    "door": "infill", "lattice": "infill", "gable_panel": "infill", "bargeboard": "infill",
}
FORMS = {form.name: form for form in (
    Form("chinese-hall", "Timber hall on a marble terrace: bracket sets under a concave glazed-tile roof with flying corners",
         (44, 17, 24), ("hip-roof", "hip-and-gable"),
         frozenset({"terrace_face", "step", "baluster_post", "column", "architrave", "bracket_arm", "purlin",
                    "roof_tile", "main_ridge", "hip_ridge", "chiwen"}),
         {"bays": 7},
         rules={"bays": [3, 9], "bays_meaning": "across the front, and always odd",
                "bays_step": 2, "column": "an eave column stands no higher than its bay is wide",
                "roof": "slope steepens toward the ridge; corners shoot out three doukou and up four"}),
    Form("chinese-palace-hall", "Double-eave hall: a lower eave roof around a clerestory wall, an upper roof over it",
         (54, 22, 27), ("double-hip-roof", "double-hip-and-gable"),
         frozenset({"terrace_face", "step", "column", "inner_column", "architrave", "bracket_arm", "purlin",
                    "roof_tile", "surround_ridge", "upper_wall", "main_ridge", "hip_ridge", "chiwen"}),
         {"bays": 9},
         rules={"bays": [5, 9], "bays_meaning": "across the front, and always odd",
                "bays_step": 2, "column": "an eave column stands no higher than its bay is wide",
                "eaves": "the lower eave springs from the eave columns, the upper from the inner columns"}),
)}
VERMILION = (158, 34, 28, 255)
PAINT = (44, 92, 112, 255)
GLAZE, GLAZE_DARK = (222, 168, 48, 255), (178, 122, 30, 255)
MARBLE, MARBLE_GREY = (232, 228, 216, 255), (196, 194, 184, 255)
TIMBER, BRICK = (168, 132, 90, 255), (176, 112, 90, 255)
PALETTE = {
    "core_fill": (150, 138, 120, 255), "foundation": MARBLE, "slab": MARBLE_GREY, "circulation": MARBLE,
    "column": VERMILION, "entablature": PAINT, "bracket": PAINT, "beam": TIMBER, "roof": GLAZE,
    "ornament": GLAZE, "wall": BRICK, "infill": VERMILION,
    "architrave": PAINT, "plate": (58, 84, 112, 255), "purlin": PAINT, "cross_beam": (52, 104, 124, 255),
    "king_post": TIMBER, "rafter": VERMILION, "flying_rafter": (196, 170, 120, 255),
    "main_ridge": GLAZE_DARK, "chiwen": GLAZE_DARK, "beast": GLAZE_DARK, "hip_ridge": GLAZE_DARK,
    "surround_ridge": GLAZE_DARK, "lattice": (208, 168, 72, 255),
    "gable_panel": (58, 104, 142, 255), "bargeboard": (58, 104, 142, 255),
}
# Fir is the structural timber of the official halls: purlins, beams and posts
# stay bare softwood inside the roof. What the visitor sees is painted or
# lacquered: vermilion columns, doors and rafters, blue-green architraves,
# plates, purlins and bracket sets. The terrace, stairs and balustrade are white
# marble over a rammed earth and rubble core, the enclosing walls are fired
# brick, and the roof and its ridges are lead-glazed tile.
PAINTED = ("column", "inner_column", "architrave", "plate", "bracket_block", "bracket_arm", "purlin",
           "rafter", "flying_rafter", "door", "lattice", "gable_panel", "bargeboard")
SUBSTANCES = {
    "core_fill": "earth-fill",
    "terrace_face": "marble", "terrace_paving": "marble", "step": "marble", "ramp": "marble",
    "baluster_post": "marble", "baluster_panel": "marble", "spout": "marble", "column_base": "marble",
    "cross_beam": "softwood", "king_post": "softwood", "corner_beam": "softwood",
    "roof_tile": "glazed-tile", "main_ridge": "glazed-tile", "hip_ridge": "glazed-tile",
    "surround_ridge": "glazed-tile", "chiwen": "glazed-tile", "beast": "glazed-tile",
    "end_wall": "fired-clay-brick", "sill_wall": "fired-clay-brick", "upper_wall": "fired-clay-brick",
    **{role: "painted-timber" for role in PAINTED},
}
DEFAULT_SUBSTANCE = "softwood"
# The eave turns in plan and lifts in elevation only near the corner itself.
PLAN_TURN, LIFT_TURN = 2.6, 3.0
PAVING = {"concept": 3.0, "standard": 1.6, "presentation": 1.2}


def bar(a, b, width, height):
    """A straight member whose underside centre line runs from ``a`` to ``b``."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    along = b - a
    side = np.array((-along[2], 0.0, along[0]))
    norm = np.linalg.norm(side)
    side = side / norm if norm > 1e-9 else np.array((1.0, 0.0, 0.0))
    return hull([end + side * across + (0, lift, 0) for end in (a, b)
                 for across in (-width / 2, width / 2) for lift in (0.0, height)])


def round_timber(a, b, radius, sides=12, top_radius=None):
    """A round member as a hull of two polygon rings; a column is one tapered piece."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    axis = (b - a) / np.linalg.norm(b - a)
    seed = np.array((0.0, 1.0, 0.0)) if abs(axis[1]) < .9 else np.array((1.0, 0.0, 0.0))
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    # Lay a vertex (not a flat) against the underside, so a horizontal member beds on what carries it.
    down = np.array((0.0, -1.0, 0.0)) if abs(axis[1]) < .9 else v
    first = math.atan2(float(np.dot(down, v)), float(np.dot(down, u)))
    return hull([centre + r * (math.cos(t) * u + math.sin(t) * v)
                 for centre, r in ((a, radius), (b, radius if top_radius is None else top_radius))
                 for t in (first + TAU * k / sides for k in range(sides))])


def interpolate(marks, u):
    """Height on a piecewise profile given as (position, height) marks."""
    for (u0, y0), (u1, y1) in zip(marks, marks[1:]):
        if u <= u1 + 1e-12:
            return y0 if u1 - u0 < 1e-12 else y0 + (y1 - y0) * (u - u0) / (u1 - u0)
    return marks[-1][1]


def split(a, b, length):
    """``a`` to ``b`` in pieces no longer than ``length``."""
    pieces = max(1, math.ceil(np.linalg.norm(b - a) / length))
    return [(a + (b - a) * k / pieces, a + (b - a) * (k + 1) / pieces) for k in range(pieces)]


class Pitch:
    """One roof over a rectangular eave.

    ``front(s, u)`` is the tile surface of the slope facing -Z at eave
    parameter ``s`` in [-1, 1] and height parameter ``u`` (0 at the eave, 1 at
    the ridge or the wall the slope dies into). ``end(s, u)`` is the slope
    facing -X. A ridge roof closes on a ridge; an apron on an inner rectangle.
    """

    def __init__(self, ex, ez, y0, rise, marks, lift, shoot, crown, rings, gable=None, zr=None):
        self.ex, self.ez, self.y0, self.rise, self.marks = ex, ez, y0, rise, marks
        self.lift, self.shoot, self.crown, self.rings, self.gable = lift, shoot, crown, rings, gable
        self.zr = ez if zr is None else zr               # half-depth of the columns it covers

    def step(self, u):
        """Index of the profile step that carries height ``u``."""
        return next(k for mark, k in self.rings if mark >= u - 1e-9)

    def edge(self, s):
        """Plan reach of the eave: straight until it shoots out at the corner."""
        return 1 + self.shoot * abs(s) ** PLAN_TURN / self.ex

    def height(self, s, u):
        return self.y0 + self.rise * interpolate(self.marks, u) \
            + self.lift * abs(s) ** LIFT_TURN * (1 - u) ** 2

    def width(self, s, u):
        """Half-width of the front slope at ``u``, measured from its middle."""
        if self.gable and u > self.gable[0]:
            u_g, over, _ = self.gable
            return (1 - u_g) * self.edge(s) * self.ex + u_g * self.crown[0] + over
        return (1 - u) * self.edge(s) * self.ex + u * self.crown[0]

    def front(self, s, u):
        return np.array((s * self.width(s, u), self.height(s, u),
                         -((1 - u) * self.edge(s) * self.ez + u * self.crown[1])))

    def end(self, s, u):
        return np.array((-((1 - u) * self.edge(1) * self.ex + u * self.crown[0]), self.height(s, u),
                         s * ((1 - u) * self.edge(s) * self.ez + u * self.crown[1])))

    def rafter(self, side, v, u):
        """Tile surface above the rafter whose eave is ``v`` from the axis."""
        s = v / (self.width(0, u) if side == "long" else (1 - u) * self.ez + u * self.crown[1])
        if side == "long":
            return np.array((v, self.height(s, u), -((1 - u) * self.edge(s) * self.ez + u * self.crown[1])))
        return np.array((-((1 - u) * self.edge(s) * self.ex + u * self.crown[0]), self.height(s, u), v))

    def reaches(self, side, v):
        """Where a rafter at ``v`` meets its hip (or the ridge) along ``u``."""
        span, crown = (self.ex, self.crown[0]) if side == "long" else (self.ez, self.crown[1])
        if self.gable and side == "long" and abs(v) <= self.gable[2]:
            return 1.0
        u = min(1.0, (span * (1 + self.shoot / self.ex) - abs(v)) / (span * (1 + self.shoot / self.ex) - crown))
        return min(u, self.gable[0]) if self.gable and side == "end" else u


class Hall(Composer):
    PALETTE = PALETTE
    T = .5                   # ashlar facing skin of the terrace
    TIMBER_KINDS = frozenset({"beam", "entablature", "bracket", "column", "infill"})

    def compose(self):
        palace, self.gabled = "palace" in self.form.name, self.variant.endswith("hip-and-gable")
        self.ratio(self.d / self.w, "depth/width", .32, .8 if palace else .9)
        bays = self.control("bays", *self.form.rules["bays"])
        if bays % 2 == 0:
            raise ValueError(f"{self.form.name} takes an odd number of bays, so the door stands on the axis")
        self.jumps = 3 if bays >= 9 else 2                 # five or nine steps (wucai, jiucai), by rank
        self.sets = 0

        # Plan: bay widths, the stair that climbs the terrace, and the terrace that holds both.
        middle = bays // 2
        # Bays shorten outward by a modest ratio. It compounds, so it is drawn narrowly and never above one.
        step = .95 * self.vary("bays", .03)
        weights = [step ** abs(k - middle) for k in range(bays)]
        riser, tread = .15 * self.vary("riser", .1), .32 * self.vary("tread", .08)
        share = clamp(.2 * self.vary("terrace", .2), .15, .25)   # terrace as a share of the eave column
        reach = clamp(.9 * self.vary("column", .04), .84, .96)   # eave column as a share of the widest bay
        self.side_stairs = bays >= 5 and self.pick("side_stairs", (True, False))
        hc = .08 * self.w
        for _ in range(6):
            d = hc / 66                                    # one doukou: the column is sixty-six of them
            steps = math.ceil(share * hc / riser)
            spare = 28 * d / 3 + 3 * d + .3                 # a corner reaches this far past the terrace edge
            clear = self.w - 2 * max(tread * steps, spare)
            hc = reach * clear / sum(weights)
        self.dk, self.dc = hc / 66, hc / 11
        self.overhang = 28 * self.dk                        # eave tip past the eave column
        self.arm_reach = (3 * self.jumps + .2) * self.dk    # eave purlin past the eave column
        ht = max(share * hc, .8)
        steps = math.ceil(ht / riser)
        riser, tread = ht / steps, clamp(riser * 2.15, .28, .36)     # a comfortable going: twice the rise and a tread
        run = tread * steps
        spare = self.overhang / 3 + 3 * self.dk + .3
        self.tx, self.tz = self.w / 2 - (max(run, spare) if self.side_stairs else spare), self.d / 2 - max(run, spare)
        self.cap = .12 * ht                                   # paving thickness on the terrace
        x, z = self.tx - self.overhang * 2 / 3, self.tz - self.overhang * 2 / 3     # eave column lines
        self.ratio(z / x, "frame depth/frame width", .2, .85)
        bay = [2 * x * k / sum(weights) for k in weights]
        self.xs = [-x] + [-x + sum(bay[:k + 1]) for k in range(bays)]
        self.xs[-1] = x
        self.shape = ((.3, 1.4, .3) if palace else (.62, .85, .62)) if palace or 2 * z < 4.3 * bay[middle] \
            else (.5, .65, .85, .65, .5)
        self.zs = [-z] + [-z + 2 * z * sum(self.shape[:k]) / sum(self.shape) for k in range(1, len(self.shape))] + [z]
        self.count = len(self.shape)                       # purlin steps from the eave purlin to the ridge
        roof_depth = 2 * (-self.zs[1] if palace else z) + 2 * self.overhang
        self.ridge_height = clamp(.05 * .35 * roof_depth + .22, .25, .7)
        self.chiwen = clamp(.09 * roof_depth, .9, 3.4)     # an owl-tail stands in proportion to the span

        # Heights. The eave columns obey the bay rule. Inner columns rise to the
        # cross-beams; in a double-eave hall they rise a storey to the upper eave.
        self.rafter_t, self.tile_t = 1.5 * self.dk, .06 + clamp(.2 * self.vary("tiles", .1), .14, .3)
        self.thickness = self.rafter_t + self.tile_t       # tiles, boarding and the rafters under them
        self.lift, self.shoot = 4 * self.dk * self.vary("lift", .2), 3 * self.dk
        self.ridge_width = clamp(1.1 * self.dc, .35, .9)
        self.bays, self.beasts = bays, min(9, bays)     # an odd line of beasts, by rank
        self.tile_pitch = {"concept": 9 * self.dk, "standard": 4.5 * self.dk, "presentation": 3 * self.dk}[self.detail]
        self.rafter_pitch = {"standard": 4.5 * self.dk, "presentation": 2.25 * self.dk}.get(self.detail)
        hL = hc
        self.beam_rise = 2 * self.dk + (self.jumps - 1) * 3.4 * self.dk + 5.4 * self.dk + 1.25 * self.dc   # plate, set, beam
        self.yt, self.yp = ht + hL, ht + hL + 2 * self.dk
        if palace:
            # The lower roof is one shallow step (wuju) from the eave purlin in to the clerestory wall.
            self.lower_rise = .5 * self.vary("lower_slope", .08) * (
                self.shape[0] * 2 * z / sum(self.shape) + self.overhang - self.arm_reach * .5)
            hU = hL + self.eave(self.yt) - self.yt + self.rafter_t + self.tile_t + self.lower_rise \
                + clamp(.9 * self.vary("clerestory", .15), .7, 1.5) + 6.4 * self.dk
        else:
            hU = hL + self.beam_rise
        self.ratio(hU / bay[middle], "inner column/middle bay", .6, 2.5)
        self.yU = ht + hU

        self.x, self.z = x, z

        self.terrace(ht)
        self.stairs(ht, steps, riser, tread)
        self.rails(ht, hL)
        for x_ in self.xs:
            for z_ in self.zs:
                edge = z_ in (self.zs[0], self.zs[-1]) or x_ in (self.xs[0], self.xs[-1])
                self.column(x_, z_, ht, self.yt if edge else self.yU, "column" if edge else "inner_column")
        self.walls(ht)
        self.eave_ring(x, z, self.xs, self.zs, self.yt, "lower" if palace else "main")
        self.corners(x, z, self.yt)
        if not palace:
            ex, ez = x + self.overhang, z + self.overhang
            g = self.gabled and clamp(.5 * ez * self.vary("setback", .12), self.overhang + .6, .75 * ez)
            p = self.pitch(ex, ez, self.eave(self.yt) + self.rafter_t + self.tile_t, (ex - ez, 0), g, z)
            self.frame(self.xs, self.zs, self.yp, p, "main")
            self.roof(p, self.yt, "main")
        else:
            self.double_eave(x, z)
        return self.document(self.rules(bays, bay, hL, hU, ht))

    def rules(self, bays, bay, hL, hU, ht):
        """The proportions the hall was built to, for the catalog and the tests to read back."""
        return {
            "typology": f"{'double-eave ' if 'palace' in self.form.name else ''}{self.variant.replace('-', ' ')} hall",
            "bays": bays, "depth_bays": len(self.shape), "middle_bay": round(bay[bays // 2], 3),
            "eave_column_height": round(hL, 3), "inner_column_height": round(hU, 3), "terrace_height": round(ht, 3),
            "bracket_jumps": self.jumps, "bracket_sets": self.sets, "beasts": self.beasts,
            "eave_overhang": round(self.overhang, 3), "corner_shoot": round(self.shoot, 3),
            "corner_lift": round(self.lift, 3), "canonical_height": round(self.canonical, 3),
            "purlin_slopes": self.slopes,          # ju: rise over run in tenths, from the eave to the ridge
            "roof": "concave tile courses steepening toward the ridge; corners shoot out and lift",
            "structure": "timber post-and-beam frame on a stone terrace; bracket sets carry the eaves",
        }

    # ------------------------------------------------------------ proportions
    def eave(self, top):
        """Top of the eave purlins above a column top at ``top``."""
        return top + 2 * self.dk + (self.jumps - 1) * 3.4 * self.dk + 13.8 * self.dk

    def ring_marks(self, marks, ez, length=None):
        """Profile marks cut into tile courses no longer than ``length`` along the slope."""
        length = length or {"concept": 2.4, "standard": 1.2, "presentation": .7}[self.detail]
        rings = [(marks[0][0], 0)]
        for k, ((u0, y0), (u1, y1)) in enumerate(zip(marks, marks[1:])):
            pieces = max(1, math.ceil(math.hypot((u1 - u0) * ez, (y1 - y0)) / length))
            rings += [(u0 + (u1 - u0) * j / pieces, k) for j in range(1, pieces + 1)]
        return rings

    def pitch(self, ex, ez, y0, crown, g, zr):
        """A ridge roof from its eave to its ridge, fitted to the requested height."""
        reach, count = self.arm_reach, self.count
        runs = [self.overhang - reach, reach] + [(ez - self.overhang) / count] * count
        # Jujia, in ju (rise over run in tenths): a flier at 3.5, the eave step at 5, then the purlin steps
        # steepening to the ridge: 5, 7, 9 for a seven-purlin roof, 5 to 9 one at a time for eleven.
        slopes = [3.5, 5.0] + list({3: (5.0, 7.0, 9.0), 5: (5.0, 6.0, 7.0, 8.0, 9.0)}[count])
        canonical = sum(p / 10 * r for p, r in zip(slopes, runs))
        gable = None
        if g:
            # The lower hips are a hip roof's hips: they run at 45 degrees toward that roof's ridge end
            # and stop where they meet the gable plane, set in by ``g`` from the eave corner.
            gable = ((g + self.shoot) / (ez + self.shoot), clamp(.12 * ez, .7, 1.6), ex - g)
        self.canonical = y0 + canonical + self.chiwen     # the height this plan has at canonical slopes
        lo, hi = y0 + .7 * canonical + self.chiwen, y0 + 1.3 * canonical + self.chiwen
        if not lo - 1e-9 <= self.h <= hi + 1e-9:
            raise ValueError(f"{self.form.name} {self.w:g} x {self.d:g} needs a height between {lo:.1f} and {hi:.1f}")
        rise = self.h - self.chiwen - y0
        self.ratio(rise / (2 * ez), "roof rise/roof depth", .25, .48)
        marks, u, y = [(0.0, 0.0)], 0.0, 0.0
        for p, r in zip(slopes, runs):
            u += r / ez
            y += p / 10 * r / canonical
            marks.append((u, y))
        self.slopes = [round(p * rise / canonical, 2) for p in slopes]
        return Pitch(ex, ez, y0, rise, marks, self.lift, self.shoot, crown, self.ring_marks(marks, ez), gable, zr)

    # ---------------------------------------------------------------- terrace
    def terrace(self, ht):
        """The terrace as a xumi throne: plinth, friezes, cavettos and a recessed waist."""
        t, a, cap = self.T, clamp(.16 * ht, .12, .3), self.cap
        tx, tz = self.tx, self.tz
        layers = [("plinth", 0, .2, -.8, -.8), ("lower frieze", .2, .34, 0, 0), ("lower cavetto", .34, .47, 1, 0),
                  ("waist", .47, .6, 1, 1), ("upper cavetto", .6, .73, 1, 0), ("upper frieze", .73, .88, 0, 0)]
        for name, y0, y1, foot, head in layers:
            wide, narrow = min(foot, head), max(foot, head)        # the insets at the wide end and the narrow one
            hx, hz = tx - a * wide, tz - a * wide                  # every course is set to its wide end
            if foot == head:
                warp = None
            elif foot > head:                                      # wider at the head: draw the foot in
                warp = flare(0, 0, y0 * ht, y1 * ht, (tx - a * narrow) / hx, (tz - a * narrow) / hz)
            else:                                                  # wider at the foot: draw the head in
                warp = taper(0, 0, y0 * ht, y1 * ht, (tx - a * narrow) / hx, (tz - a * narrow) / hz)
            self.shell("terrace_face", -hx, hx, -hz, hz, y0 * ht, y1 * ht, t, warp=warp, layer=name)
            self.core(-hx + t, hx - t, y0 * ht, y1 * ht, -hz + t, hz - t, warp=warp, layer=name)
        self.slab("terrace_paving", -tx, tx, -tz, tz, ht - cap, ht, PAVING[self.detail])

    def stairs(self, ht, steps, riser, tread):
        """Front and back stairs on the axis; larger halls also have one on each side."""
        half, ramp = clamp(.11 * self.w, 1.7, 6.0), clamp(.028 * self.w, 0.0, 1.3)
        if self.w < 20:
            ramp = 0.0
        for quarter in (0, 2):
            start = len(self.placements)
            self.flight(self.tz, half, ramp, ht, steps, riser, tread)
            self.turn(start, quarter)
        if self.side_stairs:
            for quarter in (1, 3):
                start = len(self.placements)
                self.flight(self.tx, min(half, self.tz * .35), 0.0, ht, steps, riser, tread)
                self.turn(start, quarter)

    def flight(self, edge, half, ramp, ht, steps, riser, tread):
        """Steps facing -Z from the terrace edge at -edge; a carved ramp sits between the two flights."""
        spans = [(-half, half)] if not ramp else [(-half, -ramp), (ramp, half)]
        thick = clamp(ht / steps * 2.4, .25, .5)
        for j in range(steps):
            z1, z0 = -edge - (steps - j) * tread, -edge - (steps - j - 1) * tread
            for a, b in spans:
                pieces = max(1, math.ceil((b - a) / 1.5))
                for k in range(pieces):
                    x0 = a + (b - a) * k / pieces
                    self.member("step", x0, x0 + (b - a) / pieces, j * riser, (j + 1) * riser, z0, z1, step=j)
        if ramp:
            pieces = max(1, math.ceil(steps * tread / 1.4))
            for k in range(pieces):
                points = []
                for t in (k / pieces, (k + 1) / pieces):
                    y, z = riser + t * (ht - riser), -edge - steps * tread * (1 - t)
                    points += [(sx * ramp, max(0.0, y - thick), z) for sx in (-1, 1) for _ in (0,)] + \
                              [(sx * ramp, y, z) for sx in (-1, 1)]
                self.mesh("ramp", hull(points), piece=k, construction="carved dragon ramp")

    def rails(self, ht, hL):
        """Marble balustrade (wangzhu posts and lanban panels) with dragon-head spouts beneath."""
        inset, ph = .34, clamp(.11 * hL, .9, 1.3)
        half = clamp(.11 * self.w, 1.7, 6.0)
        side_half = min(half, self.tz * .35) if self.side_stairs else 0.0
        # Each edge is laid out along its own run: front and back span the width, the sides the depth.
        for edge, extent, gap, quarters in ((self.tz, self.tx, half, (0, 2)), (self.tx, self.tz, side_half, (1, 3))):
            shared = edge == self.tx                       # the sides borrow the corners' posts
            for quarter in quarters:
                start = len(self.placements)
                spans = [(-extent + inset, -gap - .35), (gap + .35, extent - inset)] if gap else [(-extent + inset, extent - inset)]
                for a, b in spans:
                    self.rail((a, b), -edge + inset, ht - self.cap, ph, edge, shared)
                self.turn(start, quarter)

    def rail(self, span, cross, paving, ph, edge, shared_ends=False):
        """Posts, panels and spouts along one run at ``cross``; ``shared_ends`` leaves its end posts to the
        run that meets it at the corner."""
        a, b = span
        posts = max(2, round((b - a) / 1.45) + 1)
        marks = [a + (b - a) * k / (posts - 1) for k in range(posts)]
        for x in marks[1:-1] if shared_ends else marks:
            self.member("baluster_post", x - .12, x + .12, paving, paving + ph, cross - .12, cross + .12)
            self.member("baluster_post", x - .15, x + .15, paving + ph, paving + ph + .26, cross - .15, cross + .15, part="head")
            points = [(x + sx * w, yy, z) for z, w, top in ((cross + .1, .15, paving), (-edge - .62, .1, paving - .12))
                      for sx in (-1, 1) for yy in (paving - .42, top)]
            self.mesh("spout", hull(points), construction="carved dragon head")
        for a_, b_ in zip(marks, marks[1:]):
            self.member("baluster_panel", a_ + .12, b_ - .12, paving, paving + ph - .2, cross - .09, cross + .09)
            self.member("baluster_panel", a_ + .12, b_ - .12, paving + ph - .2, paving + ph - .08, cross - .11, cross + .11, part="rail")

    # ------------------------------------------------------------------ frame
    def column(self, x, z, bottom, top, role):
        dc = self.dc
        self.mesh("column_base", round_timber((x, bottom, z), (x, bottom + .3 * dc + .08, z), .68 * dc, 12, .6 * dc),
                  construction="drum mirror base")
        # Shoufen: the shaft narrows by a hundredth of its height; shengtou: it leans a hundredth inward.
        self.mesh(role, round_timber((x, bottom + .3 * dc, z), (x * .99, top, z * .99), dc / 2, 12,
                                     dc / 2 - (top - bottom) / 100), register=role, construction="lacquered fir column")

    def beam(self, role, a0, a1, cross, y0, y1, thick, **params):
        """A long timber along X, spliced every six metres."""
        pieces = max(1, math.ceil((a1 - a0) / 6.0))
        for k in range(pieces):
            x0 = a0 + (a1 - a0) * k / pieces
            self.member(role, x0, x0 + (a1 - a0) / pieces, y0, y1, cross - thick / 2, cross + thick / 2, piece=k, **params)

    def purlin(self, role, a0, a1, cross, y, radius, **params):
        pieces = max(1, math.ceil((a1 - a0) / 7.0))
        for k in range(pieces):
            x0 = a0 + (a1 - a0) * k / pieces
            self.mesh(role, round_timber((x0, y, cross), (x0 + (a1 - a0) / pieces, y, cross), radius, 12), piece=k, **params)

    def eave_ring(self, x, z, xs, zs, yt, tag):
        """Plate, bracket sets, architrave beams and eave purlins around one storey."""
        d, reach, dc = self.dk, self.arm_reach, self.dc
        yp, base = yt + 2 * d, (self.jumps - 1) * 3.4 * d
        for lines, run, line, quarters in ((xs, x, z, (0, 2)), (zs, z, x, (1, 3))):
            for quarter in quarters:
                start = len(self.placements)
                cross = -line
                self.beam("plate", -run - 1.75 * d, run + 1.75 * d, cross, yt, yp, 3.5 * d, storey=tag)
                places = list(lines[1:-1]) + [a + (b - a) * k / (count + 1) for a, b in zip(lines, lines[1:])
                                              for count in [max(1, round((b - a) / (11 * d)) - 1)] for k in range(1, count + 1)]
                for along in sorted(places):
                    self.bracket_set(along, yp, cross, tag)
                outer, inner = yp + base + 8.4 * d, yp + base + 8.4 * d + .5 * reach   # tops of the two purlins
                self.beam("architrave", -run - reach, run + reach, cross - reach, yp + base + 3.4 * d, yp + base + 5.4 * d,
                          2 * d, part="eave beam", storey=tag)
                self.beam("architrave", -run - 1.2 * d, run + 1.2 * d, cross, yp + base, inner - 4.5 * d,
                          1.25 * d, part="wall-line beam", storey=tag)
                self.purlin("purlin", -run - reach, run + reach, cross - reach, outer - 1.5 * d, 1.5 * d, part="eave purlin", storey=tag)
                self.purlin("purlin", -run - 1.5 * d, run + 1.5 * d, cross, inner - 2.25 * d, 2.25 * d, part="wall-line purlin", storey=tag)
                self.turn(start, quarter)

    def bracket_set(self, along, yp, cross, tag):
        """One pingshen bracket set facing -Z, measured in doukou."""
        d, jumps = self.dk, self.jumps
        place = lambda dx, dy, dz: (along + dx, yp + dy, cross + dz)   # noqa: E731
        self.sets += 1
        set_id = self.sets
        self.add("bracket_block", place(0, d, 0), (3 * d, 2 * d, 3 * d), tier=0, name="ludou", set=set_id, storey=tag)
        for tier in range(1, jumps + 1):
            base, reach = (tier - 1) * 3.4 * d, (3 * tier + .2) * d
            arm = (9.2 if tier == 1 else 12.4) * d
            self.add("bracket_arm", place(0, base + 3 * d, 0), (arm, 2 * d, d), tier=tier, name="gong", set=set_id, storey=tag)
            self.add("bracket_arm", place(0, base + 3 * d, 0), (d, 2 * d, 2 * reach), tier=tier, name="qiao", set=set_id, storey=tag)
            for sign in (-1, 1):
                self.add("bracket_block", place(sign * (arm / 2 - .7 * d), base + 4.7 * d, 0), (1.4 * d,) * 3,
                         tier=tier, name="shengdou", set=set_id, storey=tag)
                self.add("bracket_block", place(0, base + 4.7 * d, sign * (reach - .7 * d)), (1.4 * d,) * 3,
                         tier=tier, name="sandou", set=set_id, storey=tag)
        base = (jumps - 1) * 3.4 * d + 5.4 * d          # top of the last tier's blocks
        self.add("bracket_arm", place(0, base + d, -reach), (9.2 * d, 2 * d, d), tier=jumps + 1, name="linggong", set=set_id, storey=tag)
        self.add("bracket_arm", place(0, base + d, -.25 * reach - 1.2 * d), (d, 2 * d, 1.5 * reach + 2.4 * d),
                 tier=jumps + 1, name="yaotou", set=set_id, storey=tag)
        for sign in (-1, 1):
            self.add("bracket_block", place(sign * 3.9 * d, base + 2.7 * d, -reach), (1.4 * d,) * 3,
                     tier=jumps + 1, name="shengdou", set=set_id, storey=tag)

    def corners(self, x, z, yt):
        """The four corner bracket sets, each with arms along both walls and a diagonal jump."""
        start = len(self.placements)
        d, jumps = self.dk, self.jumps
        yp = yt + 2 * d
        self.sets += 1
        set_id = self.sets
        self.add("bracket_block", (0, yp + d, 0), (3 * d, 2 * d, 3 * d), tier=0, name="ludou", set=set_id, corner=True)
        for tier in range(1, jumps + 1):
            reach, low = (3 * tier + .2) * d, (tier - 1) * 3.4 * d
            arm = (9.2 if tier == 1 else 12.4) * d
            for axis in (0, 2):
                dims = (arm, 2 * d, d) if axis == 0 else (d, 2 * d, arm)
                centre = [0.0, yp + low + 3 * d, 0.0]
                centre[axis] = arm / 4
                self.add("bracket_arm", centre, dims, tier=tier, name="gong", set=set_id, corner=True)
                for sign in (-1, 1):
                    at = [0.0, yp + low + 4.7 * d, 0.0]
                    at[axis] = arm / 4 + sign * (arm / 2 - .7 * d)
                    self.add("bracket_block", at, (1.4 * d,) * 3, tier=tier, name="shengdou", set=set_id, corner=True)
            a, b = (reach * .45, yp + low + 2 * d, reach * .45), (-reach * .85, yp + low + 2 * d, -reach * .85)
            self.mesh("bracket_arm", bar(a, b, d, 2 * d), tier=tier, name="diagonal qiao", set=set_id, corner=True)
        reach = (3 * jumps + .2) * d
        low = (jumps - 1) * 3.4 * d + 5.4 * d
        for axis in (0, 2):
            dims = (7.8 * d, 2 * d, d) if axis == 0 else (d, 2 * d, 7.8 * d)
            centre = [0.0, yp + low + d, 0.0]
            centre[axis] = -reach + .7 * d
            self.add("bracket_arm", centre, dims, tier=jumps + 1, name="linggong", set=set_id, corner=True)
        a, b = (reach * .3, yp + low, reach * .3), (-reach - 2.4 * d, yp + low, -reach - 2.4 * d)
        self.mesh("bracket_arm", bar(a, b, d, 2 * d), tier=jumps + 1, name="diagonal yaotou", set=set_id, corner=True)
        self.shift(start, -x, -z)
        self.mirror(self.clone(start))
        self.turn(self.clone(start), 2)


    # ------------------------------------------------------------------ walls
    def walls(self, ht):
        """Brick gable-end walls, sill walls under lattice windows, and doors on the central bays."""
        d, dc, x, z = self.dk, self.dc, self.x, self.z
        top = self.yt - 6.4 * d                               # under the architrave
        for sign in (-1, 1):
            self.run("end_wall", "z", 0, sign * x, 2 * z - dc, ht, top, .8, face="left" if sign < 0 else "right")
        doors = {self.bays // 2}
        if self.bays >= 7 and self.pick("side_doors", (True, False)):
            doors |= {1, self.bays - 2}                       # the two bays beside the middle open as doors
        sill = ht + clamp(.9 * self.vary("sill", .1), .7, 1.1)
        for face, sign in (("front", -1), ("back", 1)):
            for k, (a, b) in enumerate(zip(self.xs, self.xs[1:])):
                lo, hi = a + dc / 2, b - dc / 2
                if k in doors:
                    self.leaves(lo, hi, ht, top, sign * z, "door", face, k)
                else:
                    self.run("sill_wall", "x", (lo + hi) / 2, sign * z, hi - lo, ht, sill, .6, face=face, bay=k)
                    self.leaves(lo, hi, sill, top, sign * z, "window", face, k)

    def leaves(self, lo, hi, y0, y1, cross, part, face, bay):
        """Geshan leaves between two columns: a solid panel below a lattice light."""
        count = 4 if hi - lo < 3.4 else 6
        width = (hi - lo) / count
        for k in range(count):
            a = lo + k * width
            self.member("door", a, a + width - .04, y0, y1, cross - .04, cross + .04, part=part, face=face, bay=bay, leaf=k)
            if self.detail == "concept":
                continue                                      # the lattice is standard-detail work
            low = y0 + .45 * (y1 - y0) if part == "door" else y0 + .12
            for j in range(4):
                v = a + width * (j + .5) / 4
                self.member("lattice", v - .018, v + .018, low, y1 - .12, cross - .075, cross - .04, part=part, face=face, bay=bay, leaf=k)
            self.member("lattice", a + .08, a + width - .12, (low + y1) / 2 - .025, (low + y1) / 2 + .025, cross - .075, cross - .04,
                        part=part, face=face, bay=bay, leaf=k, rail=True)

    def double_eave(self, x, z):
        """A palace hall: the lower eave springs from the eave columns and dies into a clerestory wall.

        The wall stands on the inner columns, a storey higher. On it the upper
        eave repeats the bracket work, and the upper roof is a whole ridge roof.
        """
        d, dc = self.dk, self.dc
        xw, zw, ez = -self.xs[1], -self.zs[1], z + self.overhang      # inner column lines; depth of the lower eave
        tw = clamp(1.6 * dc, .6, 1.2)                                  # clerestory wall thickness
        marks = [(0.0, 0.0), (1.0, 1.0)]
        lower = Pitch(x + self.overhang, ez, self.eave(self.yt) + self.rafter_t + self.tile_t, self.lower_rise, marks,
                      self.lift, self.shoot, (xw + tw / 2, zw + tw / 2), self.ring_marks(marks, ez), None, z)
        self.frame(self.xs, self.zs, self.yp, lower, "lower", beams_only=True)
        for side in ("long", "end"):
            start = len(self.placements)
            self.slope(lower, side, 1.0, "lower")
            self.turn(self.clone(start), 2)
        start = len(self.placements)
        self.hip(lower, 1.0, self.yt, "lower", (x, z))
        self.mirror(self.clone(start))
        self.turn(self.clone(start), 2)
        # The surround ridge (weiji) runs round the foot of the wall, where the lower roof meets it.
        top, rw, rh = lower.height(0, 1), self.ridge_width, self.ridge_height
        for half, cross in ((xw, zw), (zw, xw)):
            pieces = max(1, math.ceil(2 * half / 1.5))
            for sign in (-1, 1):
                for k in range(pieces):
                    a0, a1 = -half + 2 * half * k / pieces, -half + 2 * half * (k + 1) / pieces
                    if cross == zw:
                        self.member("surround_ridge", a0, a1, top - .08, top + rh, sign * cross - rw / 2, sign * cross + rw / 2,
                                    part="surround ridge", piece=k)
                    else:
                        self.member("surround_ridge", sign * cross - rw / 2, sign * cross + rw / 2, top - .08, top + rh, a0, a1,
                                    part="surround ridge", piece=k)
        self.shell("upper_wall", -xw, xw, -zw, zw, top - .25, self.yU - 6.4 * d, tw, construction_note="clerestory wall")
        # The upper storey is a hall of its own, standing on the inner columns.
        ez = zw + self.overhang
        g = self.gabled and clamp(.5 * ez * self.vary("setback", .12), self.overhang + .6, .75 * ez)
        upper = self.pitch(xw + self.overhang, ez, self.eave(self.yU) + self.rafter_t + self.tile_t, (xw - zw, 0), g, zw)
        self.eave_ring(xw, zw, self.xs[1:-1], self.zs[1:-1], self.yU, "upper")
        self.corners(xw, zw, self.yU)
        self.frame(self.xs[1:-1], self.zs[1:-1], self.yU + 2 * d, upper, "upper")
        self.roof(upper, self.yU, "upper", (xw, zw))

    # ------------------------------------------------------------------- roof
    def frame(self, xs, zs, yp, p, tag, beams_only=False):
        """Cross-beams on the column heads, the short posts on them, and the purlins those carry."""
        d, dc = self.dk, self.dc
        bottom = yp + (self.jumps - 1) * 3.4 * d + 5.4 * d       # on the blocks of each column-head set
        stops = [zs[0] - .6 * dc] + [v for line in zs[1:-1] for v in (line - .5 * dc, line + .5 * dc)] + [zs[-1] + .6 * dc]
        lines = []
        for k in range(1, self.count + 1):                   # the last of them is the ridge line
            zk = -p.zr + k * 2 * p.zr / self.count
            u = (p.ez - abs(zk)) / p.ez
            # A gabled roof stops its purlins at the gable board, not at the overhang beyond it.
            lines.append((zk, p.height(0, u) - self.tile_t - self.rafter_t,
                          min(p.width(0, u), p.gable[2]) if p.gable else p.width(0, u)))
        for x in xs:
            for a, b in zip(stops[::2], stops[1::2]):
                self.member("cross_beam", x - .4 * dc, x + .4 * dc, bottom, bottom + 1.25 * dc, a, b, storey=tag)
            if beams_only:
                continue
            # A post stands only where this purlin line is really the roof above it: inside the hip.
            # Past the hip the end slope is lower, and the purlin simply spans on to the next post.
            for k, (zk, top, reach) in enumerate(lines, 1):
                if abs(x) <= reach - .3 and top - 4.5 * d > bottom + 1.25 * dc + .1:
                    self.member("king_post", x - .25 * dc, x + .25 * dc, bottom + 1.25 * dc, top - 4.5 * d,
                                zk - .25 * dc, zk + .25 * dc, storey=tag, line=k)
        if beams_only:
            return
        for k, (zk, top, reach) in enumerate(lines, 1):
            self.purlin("purlin", -reach + .2, reach - .2, zk, top - 2.25 * d, 2.25 * d, part="frame purlin",
                        storey=tag, line=k)

    def slope(self, p, side, u_end, tag):
        """Tile one slope from the eave up to ``u_end``, then lay its rafters and eave fliers."""
        point = p.front if side == "long" else p.end
        rings = [(u, k) for u, k in p.rings if u < u_end - 1e-9] + [(u_end, p.step(u_end))]
        columns = max(2, math.ceil(2 * (p.ex if side == "long" else p.ez) / self.tile_pitch))
        cover = .45 * self.tile_t                     # convex cover tiles stand above the pan tiles between them
        for course, ((u0, _), (u1, step)) in enumerate(zip(rings, rings[1:])):
            steep = p.rise * (interpolate(p.marks, u1) - interpolate(p.marks, u0)) / ((u1 - u0) * p.ez)
            for column in range(columns):
                s0, s1 = -1 + 2 * column / columns, -1 + 2 * (column + 1) / columns
                top = [point(s, u) + (0, cover * (column % 2 == 0), 0) for s, u in ((s0, u0), (s1, u0), (s0, u1), (s1, u1))]
                mesh = hull(top + [v - (0, self.thickness * math.sqrt(1 + steep * steep), 0) for v in top])
                if min(mesh.extents) >= 1e-6:
                    self.mesh("roof_tile", mesh, slope=side, course=course, step=step, column=column, eave=tag)
        if not self.rafter_pitch:
            return                                    # concept detail shows the tiles, not what carries them
        d, rt, tt, dc = self.dk, self.rafter_t, self.tile_t, self.dc
        span = p.ex if side == "long" else p.ez
        fan_count = max(4, round(6 * 4.5 * d / self.rafter_pitch))
        fan_zone = fan_count * self.rafter_pitch
        under = lambda v, u: p.rafter(side, v, u) - (0, tt + rt, 0)   # noqa: E731  (under a rafter)
        count = int((span - fan_zone) / self.rafter_pitch)
        for k in range(-count, count):
            v = (k + .5) * self.rafter_pitch
            u_hit = p.reaches(side, v)
            if u_hit <= 7 * d / p.ez + .02:
                continue
            marks = [7 * d / p.ez] + [u for u, _ in rings if 7 * d / p.ez < u < u_hit - 1e-9] + [u_hit]
            for a, b in zip(marks, marks[1:]):
                for start, end in split(under(v, a), under(v, b), 4.5):
                    self.mesh("rafter", bar(start, end, rt, rt), rafter="field", slope=side, eave=tag)
            self.mesh("flying_rafter", bar(under(v, 21 * d / p.ez), under(v, 0), rt, rt), rafter="field", slope=side, eave=tag)
        # Wing rafters: they fan from one point on the hip out to the eave, and rise with it.
        gather = rings[max(1, len(rings) // 3)][0]
        for k in range(fan_count):
            s = -1 + (k + .5) / fan_count * fan_zone / span
            tip = point(s, 0) - (0, tt, 0)
            upper = point(-1, gather) + (.3 * dc, -tt - 2 * rt - .45 * dc, .3 * dc)
            back = (tip - upper) * (1, 0, 1)
            back /= np.linalg.norm(back)
            self.mesh("rafter", bar(upper, tip - 7 * d * back - (0, rt, 0), rt, rt), rafter="wing", slope=side, eave=tag)
            self.mesh("flying_rafter", bar(tip - 21 * d * back - (0, rt, 0), tip - (0, rt, 0), rt, rt), rafter="wing", slope=side, eave=tag)

    def hip(self, p, u_end, yt, tag, lines):
        """The hip at the front-left corner: its ridge tiles, the beasts along it, and the corner beam.

        ``lines`` are the half-width and half-depth of the column rectangle the corner stands on.
        """
        d, rt, tt, dc = self.dk, self.rafter_t, self.tile_t, self.dc
        rw, rh = self.ridge_width, self.ridge_height
        line = [p.front(-1, u) - (0, .06, 0) for u in [0.0] + [u for u, _ in p.rings if 0 < u < u_end - 1e-9] + [u_end]]
        for k, (a, b) in enumerate(zip(line, line[1:])):
            self.mesh("hip_ridge", bar(a, b, rw, rh), segment=k, eave=tag)
        # An odd line of beasts sits down the lower hip, led by the immortal on his phoenix at the tip.
        lengths, walked = [float(np.linalg.norm(b - a)) for a, b in zip(line, line[1:])], 0.0
        total = sum(lengths)
        for index in range(self.beasts):
            distance = (index + .6) / self.beasts * .4 * total
            k = next(k for k, length in enumerate(lengths) if walked + sum(lengths[:k]) + length > distance)
            a, b = line[k], line[k + 1]
            at = a + (b - a) * (distance - walked - sum(lengths[:k])) / lengths[k]
            self.add("beast", at + (0, rh + .18, 0), (.28, .36, .28), beast=index, eave=tag)
        self.add("beast", line[0] + (0, rh + .3, 0), (.34, .62, .34), figure="immortal", eave=tag)
        base = (self.jumps - 1) * 3.4 * d + 5.4 * d
        a = (-lines[0], yt + 2 * d + base, -lines[1])
        b = line[0] - (0, tt + rt + .9 * dc, 0) + (.25, 0, .25)
        self.mesh("corner_beam", bar(a, b, .6 * dc, .9 * dc), eave=tag, construction="jiaoliang hip rafter")

    def gable(self, p, tag):
        """The upper gable of the hip-and-gable roof, where the hips stop and the main roof runs on.

        The end slope ends at a ridge (boji); on that ridge stands the gable board
        (shanhua) under the two slopes; bargeboards (bofengban) hang below the
        overhanging edges, and the hip ridges climb those edges to the ridge end.
        """
        u_g, over, gx = p.gable
        yg, zg = p.height(1, u_g), (1 - u_g) * p.ez         # where the hips end, and how deep the gable is there
        rw, rh, tt = self.ridge_width, self.ridge_height, self.thickness
        strips = max(2, math.ceil(2 * zg / .5))
        for k in range(strips):
            z0, z1 = -zg + 2 * zg * k / strips, -zg + 2 * zg * (k + 1) / strips
            top = p.height(0, 1 - min(abs(z0), abs(z1)) / p.ez) - tt      # the tiles over the nearer end of the strip
            if top > yg + .03:
                self.member("gable_panel", -gx - .08, -gx + .08, yg - .05, top, z0, z1, strip=k, eave=tag)
        pieces = max(1, math.ceil(2 * zg / 1.5))
        for k in range(pieces):
            z0 = -zg + 2 * zg * k / pieces
            self.member("surround_ridge", -gx - rw / 2, -gx + rw / 2, yg - .06, yg + rh, z0, z0 + 2 * zg / pieces,
                        piece=k, eave=tag, part="gable ridge")
        marks = [u_g] + [u for u, _ in p.rings if u_g + 1e-9 < u < 1 - 1e-9] + [1.0]     # hips' end up to the ridge
        for sign in (-1, 1):
            edge = [np.array((-(gx + over), p.height(1, u), sign * (1 - u) * (p.ez + p.shoot))) for u in marks]
            for a, b in zip(edge, edge[1:]):
                self.mesh("bargeboard", hull([q + (sx * .05, dy, 0) for q in (a, b) for sx in (-1, 1) for dy in (-tt, -tt - .55)]),
                          side=sign, eave=tag, construction="bofengban bargeboard")
            for k, (a, b) in enumerate(zip(edge, edge[1:])):
                self.mesh("hip_ridge", bar(a - (0, .06, 0), b - (0, .06, 0), rw, rh), segment=k, side=sign, upper=True, eave=tag)

    def ridge(self, p, tag):
        """The main ridge, and the owl-tails that bite its two ends."""
        half = p.gable[2] if p.gable else p.crown[0]
        rw, rh, yr = self.ridge_width, self.ridge_height, p.height(0, 1)
        pieces = max(1, math.ceil(2 * half / 1.5))
        for k in range(pieces):
            x0 = -half + 2 * half * k / pieces
            self.member("main_ridge", x0, x0 + 2 * half / pieces, yr - .08, yr + rh, -rw / 2, rw / 2, piece=k, eave=tag)
        start = len(self.placements)
        w, h = clamp(.9 * self.chiwen, .5, 3.0), self.chiwen
        # A hooked silhouette: the mouth grips the ridge end and the tail curls back over it.
        outline = [(-.5, 0), (1.25, 0), (1.25, .2), (.55, .26), (.28, .6), (.8, .86), (.5, 1), (-.2, 1), (-.52, .66), (-.48, .28)]
        polygon = [(-half + a * w, yr - .05 + b * h) for a, b in outline]
        self.mesh("chiwen", oriented_profile(polygon, rw + .16, "x", 0.0, 0.0), eave=tag, construction="glazed owl-tail")
        self.mirror(self.clone(start))

    def roof(self, p, yt, tag, lines=None):
        """A whole ridge roof: four tiled slopes, four hips, the ridge and its owl-tails."""
        start = len(self.placements)
        self.slope(p, "long", 1.0, tag)
        self.turn(self.clone(start), 2)                      # the back slope
        u_end = p.gable[0] if p.gable else 1.0
        start = len(self.placements)
        self.slope(p, "end", u_end, tag)
        if p.gable:
            self.gable(p, tag)
        self.turn(self.clone(start), 2)                      # the far end
        start = len(self.placements)
        self.hip(p, u_end, yt, tag, lines or (self.x, self.z))
        self.mirror(self.clone(start))
        self.turn(self.clone(start), 2)
        self.ridge(p, tag)


Hall.module = sys.modules[__name__]


def build_document(request):
    return Hall(request).compose()


def random_controls(product, rng):
    """Keep the plan proportions and sample a centered odd bay grid."""
    form = FORMS[product]
    width, height, depth = form.defaults
    scale = rng.uniform(.85, 1.2)
    bays = rng.choice((7, 9) if product == "chinese-palace-hall" else (5, 7))
    # A wider middle bay raises the eave columns and the required plot height.
    height *= 1 + .055 * (form.controls["bays"] - bays)
    return {"width": round(width * scale), "height": round(height * scale), "depth": round(depth * scale),
            "bays": bays,
            "detail": rng.choice(("concept", "standard"))}


def _top(p):
    return p.position[1] + p.dimensions[1] / 2


def _bottom(p):
    return p.position[1] - p.dimensions[1] / 2


def _grades(tiles, slope):
    """Slope of each purlin step up the middle tile column of one roof slope, read from the eave inward.

    The roof faces -Z for a long slope and -X for an end slope. Each step is measured
    between the ridge ends of its own last course and of the step below it.
    """
    axis = 2 if slope == "long" else 0
    facing = [p for p in tiles if p.parameters["slope"] == slope and p.position[axis] < 0]
    first = [p for p in facing if p.parameters["course"] == 0]
    if not first:
        return []
    column = min(first, key=lambda p: abs(p.position[0 if slope == "long" else 2])).parameters["column"]
    line = [p for p in facing if p.parameters["column"] == column]
    crests = {}
    for p in line:
        step = p.parameters["step"]
        if step not in crests or p.parameters["course"] > crests[step].parameters["course"]:
            crests[step] = p
    ordered = [crests[step] for step in sorted(crests)]
    far = lambda p: p.position[axis] + p.dimensions[axis] / 2     # noqa: E731  (its ridge end)
    return [(_top(above) - _top(below)) / (far(above) - far(below))
            for below, above in zip(ordered, ordered[1:]) if far(above) - far(below) > 1e-6]


def rule_checks(document, form):
    """Rules read from the members themselves, not from metadata."""
    placements = document.placements
    by_role = {}
    for p in placements:
        by_role.setdefault(p.role, []).append(p)
    eave = by_role.get("column", [])
    tiles = [p for p in by_role.get("roof_tile", []) if p.parameters.get("eave") != "lower"]
    checks = []

    # Bays: odd in number across the front, the middle one widest, shortening outward on both hands.
    if eave:
        line = min(p.position[2] for p in eave)
        marks = sorted({round(p.position[0], 3) for p in eave if abs(p.position[2] - line) < 1e-3})
        widths = [b - a for a, b in zip(marks, marks[1:])]
        middle = len(widths) // 2
        outward = [widths[middle - k] / widths[middle - k - 1] for k in range(1, middle)]     # the left half, outward
        ok = len(widths) >= 3 and len(widths) % 2 == 1 and all(
            ratio > 1.0 for ratio in outward) and all(
            abs(a - b) < 1e-3 for a, b in zip(widths, widths[::-1]))
        checks.append(("wider_middle_bay", ok, "Bays are odd in number, the middle one widest, and they shorten outward.",
                       "The bays no longer shorten outward from a widest middle bay.", [p.id for p in eave] if not ok else []))
        # The Song rule: a column stands no higher than its bay is wide.
        tall = [p.id for p in eave if middle >= len(widths) or p.dimensions[1] > widths[middle] + 1e-6]
        checks.append(("column_within_bay", not tall, "No eave column stands higher than the middle bay is wide.",
                       "An eave column stands higher than the middle bay is wide.", tall))

    # The hall stands on its terrace: every column foot rests on the paving, well above the ground.
    paving, feet = by_role.get("terrace_paving", []), by_role.get("column_base", [])
    if paving and feet:
        level = max(_top(p) for p in paving)
        off = [p.id for p in feet if abs(_bottom(p) - level) > 1e-3]
        ok = not off and level >= .5
        checks.append(("hall_on_terrace", ok, "Every column stands on the terrace paving, clear of the ground.",
                       "A column has left the terrace paving, or the terrace has sunk to the ground.", off))

    # Every eave column carries a bracket set, and every set steps out the same number of jumps.
    blocks = by_role.get("bracket_block", [])
    if eave and blocks:
        heads = [p for p in blocks if p.parameters.get("name") == "ludou"]
        bare = [c.id for c in eave if not any(
            abs(h.position[0] - c.position[0]) < h.dimensions[0] and abs(h.position[2] - c.position[2]) < h.dimensions[2]
            and 0 <= _bottom(h) - _top(c) < 1.0 for h in heads)]
        jumps = {}
        for p in by_role.get("bracket_arm", []):
            if p.parameters.get("name") == "qiao":
                jumps[p.parameters.get("set")] = jumps.get(p.parameters.get("set"), 0) + 1
        uneven = sorted(key for key, count in jumps.items() if count != next(iter(jumps.values())))
        ok = not bare and bool(jumps) and not uneven
        checks.append(("brackets_carry_eaves", ok,
                       "Every eave column carries a bracket set, and every set steps out the same number of jumps.",
                       "An eave column carries no bracket set, or the sets step out unevenly.", bare))

    # The roof steepens toward the ridge, purlin step by purlin step, on both kinds of slope.
    flat = []
    for slope in ("long", "end"):
        grades = _grades(tiles, slope)
        if len(grades) < 2 or not all(a < b - 1e-9 for a, b in zip(grades, grades[1:])):
            flat += [p.id for p in tiles if p.parameters["slope"] == slope]
    checks.append(("roof_steepens_to_ridge", not flat, "The roof steepens with every purlin step toward the ridge.",
                   "A slope no longer steepens toward the ridge.", flat))

    # Flying corners: at the eave, the corner tile stands above the middle of the same eave.
    front = [p for p in tiles if p.parameters["course"] == 0 and p.parameters["slope"] == "long" and p.position[2] < 0]
    if front:
        columns = max(p.parameters["column"] for p in front)
        corner = min(front, key=lambda p: p.parameters["column"])
        centre = min((p for p in front if p.parameters["column"] % 2 == 0), key=lambda p: abs(p.position[0]))
        rise = _top(corner) - _top(centre)
        checks.append(("flying_corners", rise > .05, "The eave lifts toward each corner.",
                       "An eave corner no longer lifts above the middle of the eave.",
                       [p.id for p in front if p.parameters["column"] in (0, columns)] if rise <= .05 else []))

    # An owl-tail bites each end of the ridge, each with its twin.
    tails = by_role.get("chiwen", [])
    lorn = unmirrored(tails, {"chiwen"}) if tails else []
    ok = len(tails) == 2 and not lorn
    checks.append(("owl_tail_each_end", ok, "An owl-tail bites each end of the ridge.",
                   "The ridge has lost an owl-tail, or its twin is missing.", [p.id for p in tails] if not ok else []))

    # The same odd line of beasts sits down every hip, led by the immortal on the corner tip.
    beasts = by_role.get("beast", [])
    hips = {}
    for p in beasts:
        hips.setdefault((p.position[0] > 0, p.position[2] > 0, p.parameters.get("eave")), []).append(p)
    lines = {key: sum(1 for q in members if "beast" in q.parameters) for key, members in hips.items()}
    ok = len(hips) >= 4 and len(set(lines.values())) == 1 and next(iter(lines.values())) % 2 == 1
    checks.append(("beast_line_each_hip", ok, "The same odd line of beasts sits down every hip.",
                   "A hip has lost its line of beasts, or the lines no longer match.",
                   [p.id for p in beasts] if not ok else []))

    # A hip roof closes on its hips. A hip-and-gable roof closes each end with a gable standing on the hips.
    panels = by_role.get("gable_panel", [])
    if document.request.variant.endswith("hip-and-gable"):
        ends = [p for p in tiles if p.parameters["slope"] == "end"]
        ok = bool(panels) and bool(ends) and min(_bottom(p) for p in panels) >= max(_top(p) for p in ends) - .35
        checks.append(("gable_closes_hips", ok, "A gable board closes each end above the hips.",
                       "A gable no longer stands on the hips.", [p.id for p in panels] if not ok else []))
    else:
        checks.append(("no_gable", not panels, "The hip roof closes on its hips without a gable.",
                       "A hip roof has grown a gable.", [p.id for p in panels]))

    # A double-eave hall: the lower eave runs round beneath the upper one, on a clerestory wall between them.
    if "palace" in form.name:
        purlins = [p for p in by_role.get("purlin", []) if p.parameters.get("part") == "eave purlin"]
        lower = [_top(p) for p in purlins if p.parameters.get("storey") == "lower"]
        upper = [_bottom(p) for p in purlins if p.parameters.get("storey") == "upper"]
        wall = by_role.get("upper_wall", [])
        ok = bool(lower and upper and wall) and max(lower) < min(_bottom(p) for p in wall) \
            and max(_top(p) for p in wall) < min(upper)
        checks.append(("two_eaves", ok, "A lower eave runs round beneath the upper one, on a clerestory wall.",
                       "The lower eave no longer runs beneath the upper one on a wall.",
                       [p.id for p in purlins] if not ok else []))
    return checks
