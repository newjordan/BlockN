"""Byzantine architect: the cross-in-square church of the Middle Byzantine centuries.

A cross-in-square church is planned on a square naos. Columns divide it into nine
bays: a domed centre, four barrel-vaulted arms that make a cross, and four low
corner bays. Four arches span the centre, four pendentives turn their square
into the circle of a drum, and a hemispherical dome on the drum crowns the
church. The sanctuary at the east end is three rooms under three apses, the
middle one for the altar, and a narthex closes the west.

Four plans are built. ``four-column`` sets the dome on four columns, as at the
Lips church, the Myrelaion and the Chalkeon. ``two-column`` rests the western
arches on two columns and the eastern ones on the piers that divide the
sanctuary into three, a variant of the Greek school. ``five-dome`` raises a
lesser dome over each corner bay, as at the Cattolica di Stilo. ``four-pier``
has no columns: solid masonry fills the corners and four piers stand in from
the walls, the atrophied Greek cross of the Chora.

This is an original composition built from the type's rules, not a survey of
one monument. Walls alternate layers of ashlar with layers of brick, the
Constantinopolitan fashion (the Greek lands more often framed each stone in
brick), and every course of the building lies on one set of lines so the bands
run unbroken round it. The rubble core of a wall is not modelled apart from its
facing, the apses are round rather than three-sided, and the corner bays are
open to their roofs where a church would groin-vault them.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import math
import sys

import numpy as np
from .coursed import Coursed, Lights, arc_x, clip, solid
from .historic import DETAILS, Form, TAU, clamp, hull, number, unmirrored
from .masonry import course_target, stretcher_target
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("byzantine", "1.0.0", "Byzantine Architect")
STYLE = "byzantine"
KINDS = {
    "core_fill": "core", "cell": "core", "platform": "foundation", "floor": "slab",
    "naos_wall": "wall", "apse_wall": "wall", "narthex_wall": "wall", "partition": "wall",
    "tympanum": "wall", "haunch": "wall", "pier": "pier",
    "great_arch": "arch", "window_arch": "arch", "door_arch": "arch", "arcade_arch": "arch",
    "column_base": "column", "column_shaft": "column", "capital": "column",
    "pendentive": "vault", "vault": "vault", "conch": "vault",
    "drum": "drum", "dome": "roof", "roof": "roof", "deck": "roof", "cornice": "entablature",
    "finial": "ornament", "tie_beam": "beam",
}
FORMS = {form.name: form for form in (
    Form("byzantine-church",
         "Middle Byzantine cross-in-square church: a pendentive dome on a lit drum over four arches, three apses and a narthex",
         (15, 20, 23), ("four-column", "two-column", "five-dome", "four-pier"),
         frozenset({"naos_wall", "great_arch", "pendentive", "drum", "dome", "apse_wall",
                    "narthex_wall", "arcade_arch", "window_arch", "roof"}),
         {"bays": 12},
         rules={"bays": [4, 20], "bays_step": 2, "bays_meaning": "even number of arched lights in the drum",
                "width": [9, 30], "height": [10, 52], "depth": [12, 56],
                "height/width": [1.1, 1.7], "depth/width": [1.3, 1.8],
                "period": "Middle Byzantine, 9th to 12th century",
                "region": "Constantinople, Macedonia and Greece",
                "plan": "a square naos in nine bays: a domed centre, four arms and four corner bays",
                "structure": "four arches on columns carry pendentives, a drum and a hemispherical dome"}),
)}
TYPLOGY = {"four-column": "cross-in-square church with the dome on four columns",
           "two-column": "cross-in-square church: two columns and two sanctuary piers carry the dome",
           "five-dome": "cross-in-square church with a lesser dome over each corner bay",
           "four-pier": "atrophied Greek-cross church: four piers in solid corners carry the dome"}
BRICK, BRICK_DARK = (174, 106, 78, 255), (152, 90, 66, 255)
ASHLAR, ASHLAR_DARK = (210, 200, 180, 255), (194, 184, 164, 255)
MARBLE, LEAD, TILE, OAK, IRON = (232, 230, 224, 255), (92, 100, 112, 255), (172, 98, 72, 255), (112, 82, 56, 255), (62, 64, 68, 255)
PALETTE = {
    "core_fill": (150, 138, 120, 255), "cell": (150, 138, 120, 255), "foundation": (196, 188, 172, 255),
    "slab": MARBLE, "wall": ASHLAR, "pier": ASHLAR, "arch": BRICK, "vault": BRICK, "drum": BRICK,
    "roof": TILE, "deck": LEAD, "dome": LEAD, "conch": BRICK, "column": MARBLE,
    "beam": OAK, "entablature": ASHLAR, "ornament": IRON,
}
# Brick and limestone in lime mortar; reused marble columns; lead on the dome and
# the decks, tile on the timber roofs over the vaults; oak ties; an iron cross.
SUBSTANCES = {"core_fill": "lime-mortar", "cell": "lime-mortar", "platform": "limestone", "floor": "marble",
              "column_base": "marble", "column_shaft": "marble", "capital": "marble", "pier": "limestone",
              "great_arch": "fired-clay-brick", "vault": "fired-clay-brick", "conch": "fired-clay-brick",
              "pendentive": "fired-clay-brick", "tympanum": "fired-clay-brick", "haunch": "fired-clay-brick",
              "drum": "fired-clay-brick", "roof": "clay-roof-tile", "dome": "lead-sheet", "deck": "lead-sheet",
              "tie_beam": "oak", "finial": "wrought-iron", "cornice": "limestone"}
DEFAULT_SUBSTANCE = "limestone"
TOL = 1e-4


def substances_for(form):
    return set(SUBSTANCES.values()) | {"fired-clay-brick"}


class Church(Coursed):
    """One cross-in-square church, raised from one set of course lines up to the cross on the dome."""
    PALETTE = PALETTE
    TIMBER_KINDS = frozenset({"beam"})
    BANDED = frozenset({"naos_wall", "apse_wall", "narthex_wall", "partition", "tympanum", "haunch",
                        "window_arch", "door_arch", "arcade_arch"})

    def __init__(self, request):
        super().__init__(request)
        self.stone_courses = self.pick("bands", (2, 3))

    def compose(self):
        self.plan()
        self.platform()
        self.walls()
        self.partitions()
        self.supports()
        self.apses()
        self.crown()
        self.vaults()
        self.roofs()
        self.floors()
        self.bands(0)
        self.shift(0, 0, -(self.z0 - self.narthex - self.outer + self.z1 + self.rb) / 2)
        return self.document({
            "typology": TYPLOGY[self.variant], "plan": "cross-in-square",
            "column_square": round(2 * self.a, 3), "dome_diameter": round(2 * self.rd, 3),
            "springing": round(self.ys, 3), "drum_height": round(self.hd, 3), "drum_sides": self.sides or "round",
            "drum_lights": self.lights, "drum_light_angles": [round(TAU * k / self.lights, 4) for k in range(self.lights)],
            "arches": 4, "pendentives": 4, "apses": 1 if self.solid else 3,
            "narthex": round(self.narthex, 3), "exonarthex": round(self.outer, 3),
            "arm_roof_pitch": round(self.pitch_r, 4), "ridge": round(self.ridge, 3),
            "bands": f"{self.brick_courses} courses of brick to {self.stone_courses} of stone",
            "wall_thickness": round(self.T, 3), "spire_tip": round(self.h, 3),
        })

    # ---------------------------------------------------------------- plan
    def plan(self):
        name = self.form.name
        for key, size in (("width", "w"), ("height", "h"), ("depth", "d")):
            number(getattr(self, size), f"{name} {key}", *self.form.rules[key])
        self.ratio(self.h / self.w, "height/width", *self.form.rules["height/width"])
        self.ratio(self.d / self.w, "depth/width", *self.form.rules["depth/width"])
        self.lights = self.control("bays", *self.form.rules["bays"])
        if self.lights % 2:
            raise ValueError(f"{name} needs an even number of lights in the drum (4 to 20)")
        self.solid = self.variant == "four-pier"
        self.five = self.variant == "five-dome"

        self.pitch, self.stretch = course_target(self.grade), stretcher_target(self.grade)
        self.y_f = .3                                       # the floor stands one step above the ground
        e = self.e = clamp(.018 * self.w, .22, .38)         # eaves and the step stay inside the plot
        W = self.W = self.w - 2 * e                         # wall to wall
        T = self.T = clamp(.056 * W * self.vary("wall", .1), .8, 1.3)
        t = self.t = T / 2
        a = self.a = (W / 2 - T) * .6 * self.vary("bay", .06)   # the columns stand this far from the axes
        r = self.r = a - t                                     # the clear half-span of the arches
        self.c = a + t
        self.rd = a * min(.84 * self.vary("dome", .05), 1 - 1.5 * t / a)   # the drum's inner radius
        self.rise = math.sqrt(a * a + r * r - self.rd * self.rd)           # springing to the drum's base
        self.star = math.sqrt(a * a - r * r)          # where an arch's extrados meets its pier's inner face
        self.deck, self.cornice = .14, .22
        finial = clamp(.055 * self.h, .8, 2.0)
        rest = self.rise + self.deck + self.cornice + self.rd + T + finial
        self.hd = clamp(self.h - rest - 1.6 * a * self.vary("column", .12), 1.1 * self.rd, 2.5 * self.rd)
        drawn = self.h - rest - self.hd                     # where the columns would end
        self.ys = self.line(drawn)                          # the arches spring on a course line
        self.hd += drawn - self.ys                          # the drum takes up the difference
        self.y_db = self.ys + self.rise
        self.y_dt = self.y_db + self.deck + self.hd
        self.y_e = self.ys + .55 * a                        # the top of every wall round the naos
        if not self.y_f + 2.9 <= self.ys <= 2.3 * a:
            raise ValueError(f"{name} requires height/width between "
                             f"{(self.y_f + 2.9 + rest + 1.1 * self.rd) / self.w:.2f} and "
                             f"{(2.3 * a + rest + 2.5 * self.rd) / self.w:.2f}")

        def arms(narthex, outer, apse):                    # the western and sanctuary arms together
            return self.d - 2 * e - narthex - outer - apse - 2 * a - 3 * T

        narthex = clamp(.3 * W * self.vary("narthex", .1), 2.8, 6.4)
        self.rb = .8 * r * self.vary("apse", .08)
        outer = .75 * narthex if self.pick("exonarthex", (False, True)) else 0.0
        least, most = a, 3.2 * a
        if arms(narthex, outer, self.rb) < least:          # a shallow plot: shorten the narthex first
            outer, narthex = 0.0, max(2.8, narthex)
        if arms(narthex, outer, self.rb) < least:
            self.rb = max(.62 * r, self.rb - (least - arms(narthex, outer, self.rb)))
        if not least <= arms(narthex, outer, self.rb) <= most:
            raise ValueError(f"{name} requires depth/width between "
                             f"{(2 * e + 2.8 + .62 * r + 2 * a + 3 * T + least) / self.w:.2f} and "
                             f"{(2 * e + 6.4 + 4.8 + .86 * r + 2 * a + 3 * T + most) / self.w:.2f}")
        span = arms(narthex, outer, self.rb)
        share = self.vary("bema", .12)                     # the sanctuary arm runs longer than the western one
        west, east = span / (1 + share), span - span / (1 + share)
        self.z0, self.z1 = -(a + t + west + T), a + t + east + T
        self.narthex, self.outer = narthex, outer

        # The narthex is three barrel-vaulted bays that rest on the triple arcade; its roofs show
        # as three gables on the west front, and its outer slopes run down to the side walls.
        self.aw = clamp(2 * self.c * .56, 2.6, 7.0)        # the triple arcade's width
        self.lite = (self.aw - 1.0) / 3                    # the clear width of each of its three lights
        self.hb = (self.lite + .5) / 2                     # half a bay: the columns stand a bay apart
        self.arcade_spring = max(self.line(self.y_f + 1.9), self.line(self.ys - .3 * a))
        self.y_nv = self.arcade_spring + .3                # the narthex vaults spring just above the arcade
        self.y_ne = self.y_nv + .05
        self.ridge_n = self.y_nv + self.hb + .28
        self.pitch_n = (self.ridge_n - self.y_ne) / self.hb
        self.bays_x = (-2 * self.hb, 0.0, 2 * self.hb)
        self.pitch_out = min(self.pitch_n, (self.ridge_n - self.y_f - 2.9) / max(.2, W / 2 - 3 * self.hb))
        self.y_xe = self.y_ne - max(.7, .25 * outer) if outer else None
        self.slope_x = (self.y_ne - .1 - self.y_xe) / outer if outer else 0.0
        self.rs = min(.5 * self.rb, .5 * (W / 2 - T - self.c) - .25)   # a side room's apse, if its wall has room
        self.rs = self.rs if self.rs >= .7 and not self.solid else 0.0
        self.y_as = self.line(min(self.ys - .3 * r, self.y_e - .6 - self.rs))
        self.y_cw = self.y_e + .4 * a if self.five else self.y_e
        self.top_cell = self.y_cw if self.five else self.y_e
        self.pitch_r = (r + .32 - .55 * a) / self.c        # the arm roofs clear their vaults by a course of tile
        self.ridge = self.y_e + self.pitch_r * self.c
        self.rays = {"concept": 7, "standard": 9, "presentation": 11}[self.detail]
        sides = self.pick("drum", ("round", 8, 12, 14))
        self.sides = sides if isinstance(sides, int) and self.lights % sides == 0 and self.lights // sides in (1, 2) else None

    # ---------------------------------------------------------- course lines
    # --------------------------------------------------------------- ground
    def platform(self):
        """One step of a crepis under the whole church: coursed ashlar facing over a rubble core."""
        half = self.W / 2
        west, east = self.z0 - self.narthex - self.outer - self.e, self.z1 + self.rb + self.e
        self.shell("platform", -half, half, west, east, 0, self.y_f - .1, .5)
        self.core(-half + .5, half - .5, 0, self.y_f - .1, west + .5, east - .5)

    def floors(self):
        """A floor of opus sectile from the west door to the sanctuary, laid on the platform."""
        self.slab("floor", -self.W / 2, self.W / 2, self.z0 - self.narthex - self.outer, self.z1,
                  self.y_f - .1, self.y_f, 1.6, construction="opus sectile")

    # --------------------------------------------------------------- walls
    def light(self, voids, centre, sill, half, rise, ring, opening_id):
        """An arched light whose sill falls on a course line."""
        low = self.line(sill)
        voids.arch(centre, half, low, low + rise, ring, opening_id, "window_arch")
    def fit_light(self, voids, centre, top, opening_id):
        """An arched light as tall as the courses allow under a wall's top."""
        half = clamp(.035 * self.W, .25, .5)
        spring = self.line(top - half - .35)
        sill = max(self.line(spring - 1.0), self.line(self.y_f + 1.0, True))
        if spring - sill >= .3:
            voids.arch(centre, half, sill, spring, .2, opening_id, "window_arch")

    def gable(self, axis, cross, over=None):
        """The triangular wall that closes the end of an arm's pitched roof."""
        self.coursed("naos_wall", axis, 0, cross, self.T, -self.c, self.c, self.y_e, self.ridge,
                     lower=lambda u: max(self.y_e, over(u)) if over else self.y_e,
                     upper=lambda u: self.ridge - self.pitch_r * abs(u))

    def walls(self):
        """The walls round the nine bays and the narthexes behind them, with the gables over them."""
        W, T, c = self.W, self.T, self.c
        inner = W / 2 - T
        rooms = not self.solid
        for sign in (-1, 1):
            cross = sign * (inner + T / 2)
            for lo, hi, top, lit in ((self.z0, -c, self.top_cell, rooms), (-c, c, self.y_e, True),
                                     (c, self.z1, self.top_cell, rooms)):
                voids = Lights(self.y_f)
                if lit:
                    arm = lo == -c
                    self.light(voids, 0, self.ys - (.35 if arm else .6) * self.a,
                               clamp((.09 if arm else .075) * self.a, .25, .6), (.3 if arm else .22) * self.a, .22,
                               f"side.{sign:+d}.{lo:.0f}")
                self.wall("naos_wall", "z", (lo + hi) / 2, cross, hi - lo, self.y_f, top, voids, side=sign)
            # The narthex ends under the lowest of its roofs; the outer narthex under its lean-to.
            eave = self.ridge_n - self.pitch_out * (W / 2 - 3 * self.hb)
            self.wall("narthex_wall", "z", self.z0 - self.narthex / 2, cross, self.narthex, self.y_f, eave, side=sign)
            if self.outer:
                self.wall("narthex_wall", "z", self.z0 - self.narthex - self.outer / 2, cross, self.outer,
                          self.y_f, self.y_xe, side=sign)

        # The naos wall behind the narthex: the triple arcade under the gable of the western arm.
        arcade = Lights(self.y_f)
        parapet = self.line(self.y_f + .95)
        if self.arcade_spring - parapet < self.lite / 2 + .5:
            parapet = self.y_f
        arcade.arch(0, self.lite / 2, self.y_f, self.arcade_spring, .25, "arcade.0", "arcade_arch")
        for sign in (-1, 1):
            arcade.arch(sign * (self.lite + .5), self.lite / 2, parapet, self.arcade_spring, .25,
                        f"arcade.{sign:+d}", "arcade_arch")
            arcade.rect(sign * (self.lite / 2 + .25), .5, self.y_f, self.arcade_spring - self.y_f, f"arcade.column.{sign:+d}")
        self.wall("naos_wall", "x", 0, self.z0 + T / 2, 2 * c, self.y_f, self.y_e, arcade)
        self.gable("x", self.z0 + T / 2, lambda u: self.arcade_spring + arc_x(self.lite / 2 + .25, u))
        for sign in (-1, 1):
            lo, hi = sorted((sign * c, sign * inner))
            voids = Lights(self.y_f)
            if rooms:
                self.door(voids, 0, .55, self.line(self.y_f + 1.9), .22, f"west.door.{sign:+d}")
            self.wall("naos_wall", "x", (lo + hi) / 2, self.z0 + T / 2, hi - lo, self.y_f, self.top_cell, voids)

        # The east wall: the sanctuary's apse arch under the eastern gable, and a lesser apse to each side room.
        east = Lights(self.y_f)
        east.arch(0, self.rb, self.y_f, self.ys, .3, "apse.0", "door_arch")
        self.wall("naos_wall", "x", 0, self.z1 - T / 2, 2 * c, self.y_f, self.y_e, east)
        self.gable("x", self.z1 - T / 2, lambda u: self.ys + arc_x(self.rb + .3, u))
        for sign in (-1, 1):
            lo, hi = sorted((sign * c, sign * inner))
            voids = Lights(self.y_f)
            if self.rs:
                voids.arch(0, self.rs, self.y_f, self.y_as, .22, f"apse.{sign:+d}", "door_arch")
            self.wall("naos_wall", "x", (lo + hi) / 2, self.z1 - T / 2, hi - lo, self.y_f, self.top_cell, voids)

        # The narthex fronts: a door in the middle bay and a light in each outer one, under three gables.
        for cross, top, depth, gables in ((self.z0 - self.narthex + T / 2, self.y_ne, self.narthex, True),
                                          (self.z0 - self.narthex - self.outer + T / 2, self.y_xe, self.outer, False)):
            if not depth:
                continue
            voids = Lights(self.y_f)
            self.door(voids, 0, .7, min(self.line(self.y_f + 1.9), top - 1.35), .25, f"narthex.door.{depth:.0f}")
            for sign in (-1, 1):
                self.fit_light(voids, sign * W * .3, top, f"narthex.light.{sign:+d}.{depth:.0f}")
            self.wall("narthex_wall", "x", 0, cross, W - 2 * T, self.y_f, top, voids)
            if gables:
                for xc in self.bays_x:
                    self.coursed("narthex_wall", "x", 0, cross, T, xc - self.hb, xc + self.hb, top, self.ridge_n,
                                 upper=lambda u, xc=xc: self.ridge_n - self.pitch_n * abs(u - xc))

    def door(self, voids, centre, half, spring, ring, opening_id):
        """An arched doorway from the floor."""
        voids.arch(centre, half, self.y_f, spring, ring, opening_id, "door_arch")
    def partition(self, lo, hi, cross, axis, door):
        """A wall between an arm and a corner bay: solid to the springing, then shouldered round the vault."""
        voids = Lights(self.y_f)
        if door:
            self.door(voids, door * .2 * (hi - lo), .55, self.line(self.y_f + 1.9), .22, f"partition.{axis}.{cross:+.0f}.{lo:.0f}")
        self.wall("partition", axis, (lo + hi) / 2, cross, hi - lo, self.y_f, self.ys, voids)
        self.shoulder(axis, lo, hi, cross)
        self.wall("partition", axis, (lo + hi) / 2, cross, hi - lo, self.ys + self.star, self.top_cell)

    def shoulder(self, axis, lo, hi, cross):
        """The courses of a partition that rise beside its vault's haunch, cut back to the vault's extrados."""
        sign = 1 if cross > 0 else -1
        edges, first = self.lines(self.ys, self.ys + self.star)
        host = self.host("partition")
        count = max(1, math.ceil((hi - lo) / (1.15 * self.stretch)))
        step = (hi - lo) / count
        for course, (y0, y1) in enumerate(zip(edges, edges[1:])):
            shift = .5 * ((first + course) % 2)
            marks = sorted({lo, hi, *[lo + step * (k + shift) for k in range(count) if lo < lo + step * (k + shift) < hi]})
            xa, xb = (clamp(arc_x(self.a, y - self.ys), self.r, self.c) for y in (y0, y1))
            polygon = [(sign * xa, y0), (sign * self.c, y0), (sign * self.c, y1), (sign * xb, y1)]
            for p, q in zip(marks, marks[1:]):
                profile = self.plate(polygon, q - p, "x" if axis == "z" else "z", 0, (p + q) / 2)
                if profile is not None:
                    self.stone("partition", profile, host, first + course)

    def partitions(self):
        """Eight walls split the arms from the corner bays; in ``four-pier`` the bays themselves are filled."""
        T, a, c = self.T, self.a, self.c
        inner = self.W / 2 - T
        rooms = not self.solid
        for sign in (-1, 1):
            self.partition(self.z0 + T, -c, sign * a, "z", -1 if rooms else 0)
            self.partition(c, self.z1 - T, sign * a, "z", 1 if rooms else 0)
            self.partition(c, inner, sign * a, "x", 1 if rooms else 0)
        if not rooms:
            for sx in (-1, 1):
                for lo, hi in ((self.z0 + T, -c), (c, self.z1 - T)):
                    x0, x1 = sorted((sx * c, sx * inner))
                    self.core(x0, x1, self.y_f, self.top_cell - .3, lo, hi, role="cell")

    # ------------------------------------------------------------ supports
    def column(self, x, z, bottom, top, radius, width):
        """A reused marble column: a plinth and torus base, a shaft with a gentle swelling, an impost block."""
        self.add("column_base", (x, bottom + .08, z), (2.7 * radius, .16, 2.7 * radius), construction="marble plinth")
        self.drum("column_base", x, z, bottom + .16, bottom + .3, 1.34 * radius, 1.1 * radius, sides=16,
                  construction="ashlar stack", unit="drum")
        foot, head = bottom + .3, top - .4
        self.lathe("column_shaft", x, z,
                   [(foot + (head - foot) * u, radius * (1.06 - .16 * u)) for u in (0, .34, .67, 1)], sides=16)
        self.frustum("capital", x, z, head, top, 1.12 * radius, width / math.sqrt(2), sides=4, rot=math.pi / 4, rise=.42,
                     construction="impost block")

    def supports(self):
        """Marble columns carry the arches; in ``two-column`` the eastern pair are the ends of the sanctuary walls."""
        T = self.T
        if self.solid:                                       # the atrophied cross: four piers stand in from the corners
            for sx in (-1, 1):
                for sz in (-1, 1):
                    self.stack("pier", sx * self.a, self.y_f, sz * self.a, T, T, self.ys - self.y_f)
        else:
            radius = clamp(.33 * T, .24, .42)
            for sx in (-1, 1):
                for sz in (-1, 1):
                    if self.variant == "two-column" and sz > 0:
                        self.stack("pier", sx * self.a, self.y_f, sz * self.a, T, min(1.6 * T, 1.5), self.ys - self.y_f)
                    else:
                        self.column(sx * self.a, sz * self.a, self.y_f, self.ys, radius, T)
        # The narthex arcade stands on two lesser columns, with pilasters behind them on the west wall.
        small = clamp(.24 * T, .17, .3)
        west = self.z0 - self.narthex + T
        for sign in (-1, 1):
            self.column(sign * self.hb, self.z0 + T / 2, self.y_f, self.arcade_spring, small, max(.9, 2.2 * small))
            self.stack("pier", sign * self.hb, self.y_f, west + .15, .5, .3, self.y_nv - self.y_f)

    # --------------------------------------------------------------- apses
    def apses(self):
        """Round apses under half-domes: one for the sanctuary, a lesser one for each side room."""
        self.apse((0.0, self.z1), self.rb, self.ys, 3)
        if self.rs:
            for sign in (-1, 1):
                self.apse((sign * (self.c + self.W / 2 - self.T) / 2, self.z1), self.rs, self.y_as, 1)

    def apse(self, centre, radius, spring, count):
        """An apse: a drum of brick with arched lights, closed by a half-dome under a skin of lead."""
        half, ring = clamp(.17 * radius, .2, .42), .22
        outer = half + ring
        top = self.line(spring - .2) - outer                # the head of each light stops on a course line
        low = self.line(top - .6)
        gap = max(.75, 2.4 * outer / (radius - self.T / 2))
        lights = [(math.pi / 2 + (k - (count - 1) / 2) * gap, low, half, top, ring, f"apse.{centre[0]:+.1f}.{k}")
                  for k in range(count)]
        self.curved("apse_wall", centre, radius, self.y_f - .1, spring, lights, arc=(0.0, math.pi),
                    thick=min(self.T, .6 * radius))                  # a lesser apse is too slender for the full thickness
        self.dome("conch", centre, spring, radius, .36, arc=(0.0, math.pi))
        self.dome("roof", centre, spring, radius + .08, .12, arc=(0.0, math.pi), construction="lead on the conch")
    # --------------------------------------------------------------- crown
    def arches(self):
        """Four arches between the capitals, each a ring one brick thick.

        The northern and southern arches hand their springers over to the eastern and western ones,
        so no stone is laid twice where two arches meet above a capital.
        """
        r, t, T = self.r, self.t, self.T
        count = max(13, int(math.ceil(math.pi * r / .4)) | 1)
        for axis in ("x", "z"):
            for sign in (-1, 1):
                host = self.host("great_arch")
                for index in range(count):
                    q0, qm, q1 = math.pi * index / count, math.pi * (index + .5) / count, math.pi * (index + 1) / count
                    polygon = [((r + t) * math.cos(q), self.ys + (r + t) * math.sin(q)) for q in (q0, qm, q1)]
                    polygon += [(r * math.cos(q), self.ys + r * math.sin(q)) for q in (q1, qm, q0)]
                    if axis == "z":
                        for bound in (1, -1):
                            polygon = clip(polygon, lambda p, bound=bound: bound * p[0] <= r + 1e-9,
                                            lambda p, q, bound=bound: (bound * r, p[1] + (q[1] - p[1]) * (bound * r - p[0]) / (q[0] - p[0])))
                    profile = self.plate(polygon, T, axis, 0, sign * self.a) if len(polygon) >= 3 else None
                    if profile is not None:
                        self.mesh("great_arch", profile, unit="voussoir",
                                  construction="ashlar voussoir arch", host_id=host, course=index,
                                  keystone=index == count // 2, arch=f"{axis}{sign:+d}")

    def tympana(self):
        """The walls above the arches up to the drum, and the haunch over each capital between them."""
        for axis in ("x", "z"):
            for sign in (-1, 1):
                self.coursed("tympanum", axis, 0, sign * self.a, self.T, -self.r, self.r, self.ys, self.y_db,
                             lower=lambda u: self.ys + arc_x(self.a, u), arch=f"{axis}{sign:+d}")
                if axis == "x":
                    for side in (-1, 1):
                        lo, hi = sorted((side * self.r, side * self.c))
                        self.coursed("haunch", "x", 0, sign * self.a, self.T, lo, hi, self.ys, self.y_db,
                                     lower=lambda u: self.ys + arc_x(self.a, u), corner=f"{side:+d}{sign:+d}")

    def pendentives(self):
        """The spherical triangles that turn the square of the arches into the circle of the drum.

        Each corner rises in the courses above it. Every course is a fan of bricks set out from the
        pier's inner corner, and each course reaches further under the dome than the one below it,
        as the corbelled courses of a built pendentive do. A brick is one cell of the fan, cut
        back where the spherical surface crosses it.
        """
        r, big = self.r, self.a * self.a + self.r * self.r     # the sphere is centred on the springing level
        edges, first = self.lines(self.ys, self.y_db)
        rays = [(math.cos(math.pi * j / (2 * self.rays)), math.sin(math.pi * j / (2 * self.rays)))
                for j in range(self.rays + 1)]
        reach = .95 * self.stretch
        for sx in (-1, 1):
            for sz in (-1, 1):
                host = self.host("pendentive")
                corner = f"{sx:+d}{sz:+d}"

                def far(ray, rise):
                    """How far a course reaches along a ray before the sphere stops it, or the middle of a side."""
                    k = r * (ray[0] + ray[1])
                    disc = k * k - 2 * r * r + (big - rise * rise)
                    return r / max(ray) if disc <= 0 else min(k - math.sqrt(disc), r / max(ray))

                def at(ray, s, y):
                    return (sx * (r - s * ray[0]), y, sz * (r - s * ray[1]))

                def corners(ra, rb, s0, s1, y):
                    """The corners of one cell that are masonry at this height, and where the surface crosses it."""
                    rise, points = y - self.ys, []
                    fa, fb = far(ra, rise), far(rb, rise)
                    for ray, f in ((ra, fa), (rb, fb)):
                        points += [at(ray, s, y) for s in (s0, s1) if f >= s - 1e-9]
                    for s in (s0, s1):
                        if min(fa, fb) < s - 1e-9 < max(fa, fb):
                            inside, outside = (ra, rb) if fa >= fb else (rb, ra)
                            for _ in range(24):                # the ray along which the surface meets this edge of the cell
                                norm = math.hypot(inside[0] + outside[0], inside[1] + outside[1])
                                mid = ((inside[0] + outside[0]) / norm, (inside[1] + outside[1]) / norm)
                                inside, outside = (mid, outside) if far(mid, rise) >= s else (inside, mid)
                            points.append(at(inside, s, y))
                    return points

                for course, (y0, y1) in enumerate(zip(edges, edges[1:])):
                    for ra, rb in zip(rays, rays[1:]):
                        top = max(far(ra, y1 - self.ys), far(rb, y1 - self.ys))
                        count = max(1, math.ceil(top / reach))
                        for k in range(count):
                            s0, s1 = top * k / count, top * (k + 1) / count
                            points = corners(ra, rb, s0, s1, y0) + corners(ra, rb, s0, s1, y1)
                            mesh = solid(points) if len(points) >= 4 else None
                            if mesh is not None and max(mesh.extents) <= 1.5:
                                self.stone("pendentive", mesh, host, first + course, corner=corner)

    def raise_drum(self):
        """The drum stands on a ledge round the square, lit by arched lights set evenly round it."""
        T = self.T
        y0, y1 = self.y_db + self.deck, self.y_dt
        for k in range(8):                                   # the lead deck from the drum to the edge of the square
            q0, q1 = TAU * k / 8, TAU * (k + 1) / 8
            inner = [(self.rd * math.cos(q), self.rd * math.sin(q)) for q in (q0, q1)]
            outer = [(self.c / max(abs(math.cos(q)), abs(math.sin(q))) * math.cos(q),
                      self.c / max(abs(math.cos(q)), abs(math.sin(q))) * math.sin(q)) for q in (q0, q1)]
            self.mesh("deck", hull([(x, y, z) for y in (self.y_db - .01, y0) for x, z in inner + outer]),
                      construction="lead on timber decking", tier="drum")
        half, ring = clamp(.21 * TAU * (self.rd + T / 2) / self.lights, .18, .5), .25
        sill = self.line(y0 + .25 * (y1 - y0))
        spring = self.line(min(y1 - .25, sill + .62 * (y1 - y0)) + half + ring) - half - ring   # the head stops on a line
        sides = self.sides or self.lights
        lights = [(TAU * ((k + .5) / sides + (j + .5) / self.lights) if self.sides else TAU * k / self.lights,
                   sill, half, spring, ring, f"drum.{k}.{j}")
                  for k in range(sides) for j in range(self.lights // sides)]
        self.curved("drum", (0.0, 0.0), self.rd + T, y0, y1, lights, faces=self.sides or 0)
        self.radial("cornice", (0.0, 0.0), self.rd, self.rd + T + .2, y1 - .01, y1 + self.cornice,
                    construction="cornice ring")

    def finial(self, x, z, bottom, cap):
        """A lead cap over the oculus, the pomum, and the iron cross that tops the church."""
        top = self.h
        self.drum("cornice", x, z, bottom - .02, bottom + .16, cap + .1, cap, sides=12, construction="lead cap")
        base = bottom + .16
        ball = base + .45 * (top - base)
        shape = [(base, .14), (base + .12, .3), (ball - .34, .36), (ball - .18, .3), (ball - .05, .1)]
        for (ya, ra), (yb, rb) in zip(shape, shape[1:]):
            self.drum("finial", x, z, ya, yb, ra, rb, sides=12, construction="pomum")
        self.member("finial", x - .045, x + .045, ball - .06, top, z - .045, z + .045, construction="wrought-iron cross")
        arm = ball + .5 * (top - ball)
        self.member("finial", x - .3, x + .3, arm - .04, arm + .04, z - .04, z + .04, construction="wrought-iron cross")

    def lesser_domes(self):
        """In the five-dome plan each corner room carries a drum and a dome beneath the central one."""
        T, c = self.T, self.c
        inner = self.W / 2 - T
        for sx in (-1, 1):
            for lo, hi in ((self.z0 + T, -c), (c, self.z1 - T)):
                x0, x1 = sorted((sx * c, sx * inner))
                centre = ((x0 + x1) / 2, (lo + hi) / 2)
                self.slab("deck", x0, x1, lo, hi, self.y_cw - .02, self.y_cw + self.deck, 1.6,
                          construction="lead on timber decking")
                rq = min(.44 * min(x1 - x0, hi - lo), (self.y_dt - self.y_cw - .77) / 1.9)
                if rq >= .5:
                    base = self.y_cw + self.deck
                    self.radial("drum", centre, rq, rq + .35, base, base + .9 * rq, lesser=True)
                    top, cap = self.dome("dome", centre, base + .9 * rq, rq + .35, .35, lesser=True)
                    self.drum("cornice", centre[0], centre[1], top - .02, top + .12, cap + .08, cap, sides=12,
                              construction="lead cap")

    def crown(self):
        """Everything above the capitals: the arches, their walls and pendentives, the drum, the dome and the cross."""
        self.arches()
        self.tympana()
        self.pendentives()
        self.raise_drum()
        top, cap = self.dome("dome", (0.0, 0.0), self.y_dt + self.cornice, self.rd + self.T, self.T,
                             construction="lead on a brick shell")
        self.finial(0.0, 0.0, top, cap)
        for axis, sign in (("x", -1), ("x", 1), ("z", -1), ("z", 1)):
            a, r, y = sign * self.a, self.r, self.ys
            box = (-r, r, y - .3, y - .04, a - .14, a + .14) if axis == "x" else (a - .14, a + .14, y - .3, y - .04, -r, r)
            self.member("tie_beam", *box, construction="oak tie across the arch")
        if self.five:
            self.lesser_domes()

    # --------------------------------------------------------------- vaults
    @staticmethod
    def sectors(outer, thick):
        """Angles that divide a semicircle into sectors no stone's envelope makes too large."""
        def extent(q0, q1):
            points = [(r * math.cos(q), r * math.sin(q)) for r in (outer, outer - thick) for q in (q0, (q0 + q1) / 2, q1)]
            return max(x for x, _ in points) - min(x for x, _ in points), max(y for _, y in points) - min(y for _, y in points)

        def split(q0, q1):
            wide, tall = extent(q0, q1)
            if (wide > .95 or tall > .7) and q1 - q0 > .02:
                mid = (q0 + q1) / 2
                return split(q0, mid)[:-1] + split(mid, q1)
            return [q0, q1]
        angles = []
        for q0, q1 in zip([math.pi * k / 4 for k in range(4)], [math.pi * k / 4 for k in range(1, 5)]):
            angles += split(q0, q1)[:-1]
        return angles + [math.pi]

    def barrel(self, axis, along, lo, hi, radius, thick, spring=None, **params):
        """A barrel vault: courses of brick that run the length of its tunnel, each a sector of a semicircle."""
        host = self.host("vault")
        lo, hi = sorted((lo, hi))
        spring = self.ys if spring is None else spring
        angles = self.sectors(radius + thick, thick)
        count = max(1, math.ceil((hi - lo) / (1.15 * self.stretch)))
        step = (hi - lo) / count
        for k, (q0, q1) in enumerate(zip(angles, angles[1:])):
            arc = lambda r, q: (r * math.cos(q), spring + r * math.sin(q))
            polygon = [arc(radius + thick, q) for q in (q0, (q0 + q1) / 2, q1)]
            polygon += [arc(radius, q) for q in (q1, (q0 + q1) / 2, q0)]
            marks = sorted({lo, hi, *[lo + step * (j + .5 * (k % 2)) for j in range(count) if lo < lo + step * (j + .5 * (k % 2)) < hi]})
            for p, q in zip(marks, marks[1:]):
                profile = self.plate(polygon, q - p, axis, along, (p + q) / 2)
                if profile is not None:
                    self.stone("vault", profile, host, k, **params)

    def vaults(self):
        """Barrel vaults continue each arch down its arm; three more roof the bays of the narthex."""
        r, t, T, c = self.r, self.t, self.T, self.c
        self.barrel("x", 0.0, self.z0 + T, -c, r, t)
        self.barrel("x", 0.0, c, self.z1 - T, r, t)
        for sign in (-1, 1):
            self.barrel("z", 0.0, sign * c, sign * (self.W / 2 - T), r, t)
        for xc in self.bays_x:
            self.barrel("x", xc, self.z0 - self.narthex + T, self.z0, self.hb - .3, .3, spring=self.y_nv)

    # ---------------------------------------------------------------- roofs
    def roofs(self):
        """Tile roofs that show the cross: pitched over the arms and lower over the corner bays,
        three gables and two long slopes over the narthex, a lean-to over the outer narthex."""
        W, c, e, T = self.W, self.c, self.e, self.T
        tile = {"concept": 1.5, "standard": 1.15, "presentation": .8}[self.detail]
        inner = W / 2 - T

        def pitched(axis):
            def warp(vertices):
                out = vertices.copy()
                out[:, 1] += self.pitch_r * (c - np.abs(vertices[:, axis]))
                return out
            return warp

        self.slab("roof", -c, c, self.z0, -c, self.y_e, self.y_e + .22, tile, pitched(0), arm="west")
        self.slab("roof", -c, c, c, self.z1 + e, self.y_e, self.y_e + .22, tile, pitched(0), arm="east")
        for sign, arm in ((-1, "north"), (1, "south")):
            x0, x1 = sorted((sign * c, sign * (W / 2 + e)))
            self.slab("roof", x0, x1, -c, c, self.y_e, self.y_e + .22, tile, pitched(2), arm=arm)
        cells = ((self.z0, -c), (c, self.z1))
        if not self.five:
            for sx in (-1, 1):
                for lo, hi in cells:
                    x0, x1 = sorted((sx * c, sx * (W / 2 + e)))
                    xc, zc, hx, hz = (x0 + x1) / 2, (lo + hi) / 2, (x1 - x0) / 2, (hi - lo) / 2
                    rise = .5 * min(hx, hz)

                    def hip(vertices, xc=xc, zc=zc, hx=hx, hz=hz, rise=rise):
                        out = vertices.copy()
                        out[:, 1] += rise * (1 - np.maximum(np.abs(vertices[:, 0] - xc) / hx,
                                                            np.abs(vertices[:, 2] - zc) / hz))
                        return out
                    self.slab("roof", x0, x1, lo, hi, self.y_e, self.y_e + .2, tile, hip, corner=True)

        west = self.z0 - self.narthex

        def tent(vertices):
            """Three ridges over the bays, and slopes that run out to the side walls."""
            out = vertices.copy()
            near = np.minimum.reduce([np.abs(vertices[:, 0] - xc) for xc in self.bays_x])
            beyond = np.maximum(near - self.hb, 0.0)
            drop = self.pitch_n * np.minimum(near, self.hb) + self.pitch_out * beyond
            out[:, 1] += (self.ridge_n - self.y_ne) - drop
            return out
        self.slab("roof", -(W / 2 + e), W / 2 + e, west - e, self.z0, self.y_ne, self.y_ne + .2, tile, tent)
        if self.outer:

            def lean(vertices):
                out = vertices.copy()
                out[:, 1] += self.slope_x * (vertices[:, 2] - (west - self.outer))
                return out
            self.slab("roof", -(W / 2 + e), W / 2 + e, west - self.outer - e, west, self.y_xe, self.y_xe + .18, tile, lean)

    # --------------------------------------------------------------- totals
    def block_count(self):
        """How many pieces it takes to raise the church: brick, cut stone, core, timber and the rest."""
        bill = Counter()
        for p in self.placements:
            unit = p.parameters.get("unit")
            if unit == "core":
                bill["core_blocks"] += 1
            elif p.substance == "fired-clay-brick":
                bill["bricks"] += 1
            elif "ashlar" in str(p.parameters.get("construction", "")) or unit in {"drum", "voussoir", "radial_ashlar"}:
                bill["cut_stones"] += 1
            elif p.kind in self.TIMBER_KINDS:
                bill["timbers"] += 1
            else:
                bill["other_members"] += 1
        return {**dict(sorted(bill.items())), "total": len(self.placements)}


Church.module = sys.modules[__name__]


def build_document(request):
    return Church(request).compose()


def random_controls(product, rng):
    """Sample the interior of the coupled envelope on the desktop metre grid."""
    width = rng.randint(13, 24)
    return {"variant": rng.choice(FORMS[product].variants), "width": width,
            "height": rng.randint(math.ceil(1.27 * width), math.floor(1.55 * width)),
            "depth": rng.randint(math.ceil(1.54 * width), math.floor(1.70 * width)),
            "bays": rng.choice((8, 12, 16)), "levels": None, "detail": rng.choice(DETAILS)}


# ----------------------------------------------------------------- rule checks
def _groups(members, key="host_id"):
    found = defaultdict(list)
    for p in members:
        found[p.parameters.get(key)].append(p)
    return list(found.values())


def _lo(p):
    return np.asarray(p.position, dtype=float) - np.asarray(p.dimensions, dtype=float) / 2


def _hi(p):
    return np.asarray(p.position, dtype=float) + np.asarray(p.dimensions, dtype=float) / 2


def _points(p):
    """The corners of a member in the world: a mesh's own vertices, or a box's."""
    if p.block_type == "MESH":
        return np.asarray(p.parameters["vertices"], dtype=float) * p.dimensions + p.position
    return np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]) * p.dimensions + _lo(p)


def rule_checks(document, form):
    """Rules of the cross-in-square church, measured from the members after any metadata is stripped."""
    roles = defaultdict(list)
    for p in document.placements:
        roles[p.role].append(p)
    placements = document.placements
    variant = next((p.parameters.get("style") for p in placements if p.parameters.get("style")), None)

    # 1. Four arches of one span and one crown stand on the edges of a single square.
    arches = _groups(roles["great_arch"])
    boxes = [[(_lo(q), _hi(q)) for q in group] for group in arches]
    along = [max(hi[0] for _, hi in box) - min(lo[0] for lo, _ in box) >
             max(hi[2] for _, hi in box) - min(lo[2] for lo, _ in box) for box in boxes]
    crowns = [max(hi[1] for _, hi in box) for box in boxes]
    # The dome's axis: where the arches that run east and west are centred, which need not be the middle of the plot.
    axis_z = (np.mean([(min(lo[2] for lo, _ in box) + max(hi[2] for _, hi in box)) / 2
                       for box, runs in zip(boxes, along) if runs]) if sum(along) == 2 else 0.0)
    middle = [((min(lo[0] for lo, _ in box) + max(hi[0] for _, hi in box)) / 2,
               (min(lo[2] for lo, _ in box) + max(hi[2] for _, hi in box)) / 2) for box in boxes]
    planes = [max(abs(x), abs(z - axis_z)) for x, z in middle]
    centred = [min(abs(x), abs(z - axis_z)) for x, z in middle]
    ok = (len(arches) == 4 and max(crowns) - min(crowns) < .05 and max(planes) - min(planes) < .05
          and max(centred) < .05 and sum(along) == 2)
    yield ("arches_on_one_bay", ok, "Four arches of one height stand on the edges of a single square.",
           "The arches differ in height, or no longer share one square.",
           [q.id for group in arches for q in group] if not ok else [])

    def radial(points):
        return np.hypot(points[:, 0], points[:, 2] - axis_z)

    main = [p for p in roles["drum"] if not p.parameters.get("lesser")]
    drum_lo = min((float(_lo(p)[1]) for p in main), default=0.0)
    drum_radius = min((float(radial(_points(p)).min()) for p in main), default=0.0)

    # 2. Four pendentives mirror across both axes and carry the drum's circle.
    corners = defaultdict(list)
    for p in roles["pendentive"]:
        corners[p.parameters.get("corner")].append(p)
    loose = (unmirrored(placements, {"pendentive"}, axis=0)
             + unmirrored(placements, {"pendentive"}, axis=2, centre=axis_z))
    reach = [min(float(radial(_points(p)).min()) for p in group if p.parameters.get("course") == top)
             for group in corners.values() for top in [max(q.parameters.get("course", 0) for q in group)]]
    tops = [max(float(_hi(p)[1]) for p in group) for group in corners.values()]
    ok = (len(corners) == 4 and not loose and main and all(abs(r - drum_radius) < .3 for r in reach)
          and all(0 <= drum_lo - top <= .3 for top in tops))
    yield ("pendentives_carry_drum", ok, "Four pendentives mirror across both axes and carry the drum's circle.",
           "A pendentive is missing, unmirrored, or no longer meets the drum.",
           loose + [p.id for group in corners.values() for p in group] * (not ok and not loose))

    # 3. Courses step inward without overshooting the drum's inner face.
    lazy = []
    for corner, group in corners.items():
        by_course = defaultdict(list)
        for p in group:
            by_course[p.parameters.get("course")].append(p)
        nearest = [min(float(radial(_points(p)).min()) for p in by_course[k]) for k in sorted(by_course)]
        if (len(nearest) < 2 or any(b > a + 1e-6 for a, b in zip(nearest, nearest[1:]))
                or nearest[-1] > nearest[0] - .3 or nearest[-1] < drum_radius - .3):
            lazy += [p.id for p in group]
    yield ("pendentives_corbelled", bool(corners) and not lazy,
           "Each course of a pendentive reaches further under the dome than the one below it.",
           "A pendentive has lost its inward step or projects into the drum opening.", lazy)

    # 4. The drum carries the requested lights, set evenly round it.
    heads = defaultdict(list)
    for p in roles["window_arch"]:
        oid = str(p.parameters.get("opening_id", ""))
        if oid.startswith("drum.") and p.parameters.get("unit") == "voussoir":
            heads[oid].append(p)
    wanted = document.request.bays if document.request.bays is not None else form.controls["bays"]
    centres = [np.mean([p.position for p in group], axis=0) for group in heads.values()]
    angles = sorted(math.atan2(c[2] - axis_z, c[0]) % TAU for c in centres)
    gaps = [b - a for a, b in zip(angles, angles[1:])] + ([angles[0] + TAU - angles[-1]] if angles else [])
    ok = len(heads) == wanted and gaps and max(gaps) - min(gaps) < .03
    yield ("drum_lights", ok, f"The drum carries {wanted} arched lights, set evenly round it.",
           f"The drum should carry {wanted} evenly spaced lights.",
           [p.id for group in heads.values() for p in group] + [p.id for p in main] * (not ok))

    # 5. The apses stand on the east wall: three for a church with side rooms, the middle one widest and tallest.
    apses = [[(_lo(q), _hi(q)) for q in group] for group in _groups(roles["apse_wall"])]
    count = 1 if variant == "four-pier" else 3
    if apses:
        foot = [min(lo[2] for lo, _ in box) for box in apses]
        wide = [max(hi[0] for _, hi in box) - min(lo[0] for lo, _ in box) for box in apses]
        tall = [max(hi[1] for _, hi in box) for box in apses]
        east = [abs((min(lo[0] for lo, _ in box) + max(hi[0] for _, hi in box)) / 2) for box in apses]
        middle = int(np.argmin(east))
        ok = (len(apses) == count and max(foot) - min(foot) < .05 and all(wide[middle] > w + .2 for i, w in enumerate(wide) if i != middle)
              and all(tall[middle] > t + .2 for i, t in enumerate(tall) if i != middle))
    else:
        ok = False
    yield ("apses_at_east", ok, "Round apses stand on the east wall, the sanctuary's wider and taller than the rest.",
           "An apse is missing from the east wall, or the sanctuary's is no longer the greatest.",
           [q.id for q in roles["apse_wall"]] * (not ok))

    # 6. Straight and radial walls share one material schedule on their bed joints.
    schedule = Church(document.request)
    mixed = []
    checked = 0
    for role in ("naos_wall", "narthex_wall", "partition", "apse_wall"):
        for p in roles[role]:
            if p.parameters.get("unit") in {"core", "voussoir", "drum"} or p.parameters.get("register") == "spandrel":
                continue
            checked += 1
            course = p.parameters.get("course")
            band = schedule.band_for_course(course) if isinstance(course, int) else None
            substance = {"brick": "fired-clay-brick", "stone": "limestone"}.get(band)
            if band is None or p.parameters.get("band") != band or p.substance != substance:
                mixed.append(p.id)
    yield ("banded_walls", checked > 0 and not mixed,
           "Courses of brick alternate with courses of stone, two or more at a time, round the church.",
           "A wall course does not match the shared brick-and-stone schedule.", mixed)

    # 7. A narthex spans the west front and opens into the naos through a triple arcade.
    narthex = [(_lo(p), _hi(p)) for p in roles["narthex_wall"]]
    naos = [(_lo(p), _hi(p)) for p in roles["naos_wall"]]
    arcade = _groups(roles["arcade_arch"], "opening_id")
    crowns = [max(float(_hi(p)[1]) for p in group) for group in arcade]
    frames = [np.mean([p.position[2] for p in group]) for group in arcade]
    ok = (len(arcade) == 3 and max(crowns) - min(crowns) < .05 and max(frames) - min(frames) < .05
          and narthex and naos and min(lo[2] for lo, _ in narthex) < min(lo[2] for lo, _ in naos) - 1.0
          and max(hi[0] for _, hi in narthex) - min(lo[0] for lo, _ in narthex) >=
          max(hi[0] for _, hi in naos) - min(lo[0] for lo, _ in naos) - .1)
    yield ("narthex_and_arcade", ok, "A narthex spans the full west front and opens into the naos through three arches.",
           "The narthex no longer spans the west front, or its arcade is not three arches in a row.",
           [p.id for group in arcade for p in group] + [q.id for q in roles["narthex_wall"]] * (not ok))

    # 8. Four pitched roofs define the cross. Five-dome churches use
    # lesser domes instead of hipped roofs to cover the corner bays.
    arms = defaultdict(list)
    for p in roles["roof"]:
        if p.parameters.get("arm"):
            arms[p.parameters["arm"]].append(p)
    ridges = [max(float(_hi(p)[1]) for p in group) for group in arms.values()]
    arms_ok = len(arms) == 4 and max(ridges) - min(ridges) < .05
    dome = [p for p in roles["dome"] if not p.parameters.get("lesser")]
    apex = max((float(_hi(p)[1]) for p in dome), default=0.0)
    if variant == "five-dome":
        covers = [p for p in roles["dome"] if p.parameters.get("lesser")]
        groups = _groups(covers)
        quadrants = {(sum(p.position[0] for p in group) > 0,
                      sum(p.position[2] - axis_z for p in group) > 0) for group in groups}
        corners_ok = len(groups) == len(quadrants) == 4 and max(_hi(p)[1] for p in covers) < apex - .3
    else:
        covers = [p for p in roles["roof"] if p.parameters.get("corner")]
        corners_ok = (arms_ok and len(_groups(covers)) == 4
                      and max(_hi(p)[1] for p in covers) < min(ridges) - .3)
    ok = arms_ok and corners_ok
    yield ("cross_in_the_roofs", ok, "Four equal arm ridges frame the corner roofs or lesser domes.",
           "An arm ridge is out of level or the corner coverings no longer match the plan.",
           [p.id for group in arms.values() for p in group] + [p.id for p in covers] if not ok else [])

    # 9. Follow the central crown from the shell through the oculus cap to the finial.
    roofs = max((float(_points(p)[:, 1].max()) for p in roles["roof"]), default=0.0)
    cross = [p for p in roles["finial"] if "cross" in str(p.parameters.get("construction"))]
    tip = max(cross, key=lambda p: _hi(p)[1], default=None)
    caps = [p for p in roles["cornice"] if p.parameters.get("construction") == "lead cap"
            and abs(p.position[0]) < .05 and abs(p.position[2] - axis_z) < .05]
    bases = [p for p in roles["finial"] if p.parameters.get("construction") == "pomum"]
    ok = (bool(dome and caps and bases) and tip is not None and apex > roofs + .5
          and abs(apex - min(_lo(p)[1] for p in caps) - .02) < .05
          and abs(min(_lo(p)[1] for p in bases) - max(_hi(p)[1] for p in caps)) < .05
          and abs(tip.position[0]) < .05 and abs(tip.position[2] - axis_z) < .05)
    yield ("dome_crowns_the_church", ok, "The dome rises above the roofs and meets the finial cap on its axis.",
           "The dome or finial is missing, displaced from the crown joint, or off-axis.",
           [p.id for p in dome + caps + bases + cross] if not ok else [])
