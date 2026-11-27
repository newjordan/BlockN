"""Dravidian architect: the tiered temple towers of South India.

A Dravidian tower is a stack of talas (storeys), each standing inside the one
below. Every tala is a wall of pilasters and niches under a prastara, a flat
roof slab that overhangs as an eave and is carved with kudu arches. On the
slab stands a hara, a parapet of miniature shrines: kutas (square, with a
stepped cap) at the corners, a shala (barrel-roofed) along each side, and
panjaras between them. Above the last tala the crown is finished with kalasas,
the pots that top every Dravidian tower.

A gopuram is the gateway tower: a stone base pierced by a passage, brick and
lime plaster above, and a barrel crown with gilded pots. A vimana is the
shrine tower itself, stone throughout, raised over the sanctum and its
pillared hall (mandapa), and crowned by a dome or a barrel.

The talas shorten as they rise, which gives the tower its concave profile, and
the summit of a tall gopuram is about a third of its base.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .historic import Composer, Form, Voids, clamp, extent, flare, oriented_profile, taper, unmirrored
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("dravidian", "1.0.0", "Dravidian Architect")
STYLE = "dravidian"
KINDS = {
    "core_fill": "core", "base_core": "core",
    "plinth": "foundation", "floor": "slab", "step": "circulation",
    "base_wall": "wall", "passage_wall": "wall", "passage_ceiling": "slab", "string_course": "entablature",
    "sanctum_wall": "wall", "mandapa_pier": "column", "mandapa_beam": "entablature", "mandapa_roof": "slab",
    "parapet": "wall",
    "tala_wall": "wall", "pilaster": "pier", "niche": "ornament", "figure": "ornament",
    "prastara": "entablature", "kudu": "ornament",
    "kuta_base": "platform", "kuta_post": "column", "kuta_roof": "roof",
    "shala_wall": "wall", "shala_roof": "roof", "panjara": "ornament",
    "griva": "wall", "sikhara": "roof", "kalasa": "ornament",
}
FORMS = {form.name: form for form in (
    Form("dravidian-gopuram", "Tiered gateway tower: a stone base pierced by a passage, brick talas and a barrel crown",
         (26, 46, 13), ("rajagopuram", "early-gopuram"),
         frozenset({"plinth", "base_wall", "passage_wall", "tala_wall", "pilaster", "prastara", "kudu",
                    "kuta_roof", "shala_roof", "kalasa"}),
         {"bays": 7, "levels": 7},
         rules={"bays": [3, 11], "bays_meaning": "pilaster bays across each half of the front of the base",
                "levels": [5, 13], "levels_meaning": "talas (storeys) above the base",
                "profile": "each tala stands inside the one below; the summit is about a third of the base"}),
    Form("dravidian-vimana", "Tiered shrine tower over its sanctum and pillared hall, crowned by a dome or a barrel",
         (20, 36, 42), ("kuta-vimana", "shala-vimana"),
         frozenset({"plinth", "sanctum_wall", "mandapa_pier", "tala_wall", "prastara", "kuta_roof", "griva",
                    "sikhara", "kalasa"}),
         {"bays": 5, "levels": 5},
         rules={"bays": [3, 9], "bays_meaning": "pillars in each row of the hall", "levels": [3, 9],
                "levels_meaning": "talas (storeys) above the sanctum",
                "plan": "the tower is square, and the pillared hall (mandapa) stands before it"}),
)}
GRANITE, BRICK, PLASTER, GILT = (140, 136, 130, 255), (200, 160, 120, 255), (240, 234, 220, 255), (224, 172, 64, 255)
PALETTE = {
    "core_fill": (150, 120, 96, 255), "base_core": (120, 116, 110, 255),
    "foundation": GRANITE, "slab": (176, 170, 158, 255), "circulation": GRANITE,
    "wall": BRICK, "pier": BRICK, "entablature": BRICK, "ornament": PLASTER, "roof": BRICK, "platform": BRICK,
    "column": BRICK, "figure": PLASTER, "kudu": PLASTER, "kalasa": GILT,
    "base_wall": GRANITE, "passage_wall": GRANITE, "passage_ceiling": GRANITE, "string_course": GRANITE,
    "sanctum_wall": GRANITE, "mandapa_pier": GRANITE, "mandapa_beam": GRANITE, "mandapa_roof": GRANITE,
    "griva": GRANITE, "sikhara": BRICK, "floor": GRANITE,
}
VIMANA_PALETTE = {**PALETTE, "core_fill": (120, 116, 110, 255), "wall": GRANITE, "pier": GRANITE,
                  "entablature": GRANITE, "ornament": (226, 216, 198, 255), "roof": (170, 164, 154, 255),
                  "platform": GRANITE, "column": GRANITE, "figure": GRANITE, "kudu": (226, 216, 198, 255),
                  "sikhara": GRANITE}
# A gopuram is granite below and brick above. The plinth, the base, the passage
# and its floor are dressed granite; the talas, their pilasters, the frames of
# their niches, their prastaras and haras are fired brick laid in lime mortar;
# the figures in the niches and the kudu arches are lime plaster (stucco) laid
# on brick; and the pots are gilded copper. A vimana is one material throughout: the Chola masters raised theirs
# in granite, pot included.
SUBSTANCES = {
    "core_fill": "fired-clay-brick", "base_core": "granite",
    "plinth": "granite", "base_wall": "granite", "passage_wall": "granite", "passage_ceiling": "granite",
    "string_course": "granite", "floor": "granite", "step": "granite",
    "figure": "lime-plaster", "niche": "fired-clay-brick", "kudu": "lime-plaster",
    "kalasa": "copper-sheet",
    "default": "fired-clay-brick",
}
VIMANA = {"core_fill": "granite", "base_core": "granite", "kalasa": "granite", "default": "granite"}
DEFAULT_SUBSTANCE = "fired-clay-brick"


def substances_for(form):
    """What a tower is built of, so the catalog describes the vimana and the gopuram each on its own terms."""
    return set(VIMANA.values()) if "vimana" in form.name else set(SUBSTANCES.values())


def arch_band(radius, thickness, spring, count=12):
    """The section of a barrel vault: a semicircular band standing on its springing line."""
    outer = [(radius * math.cos(math.pi * k / count), spring + radius * math.sin(math.pi * k / count))
             for k in range(count + 1)]
    inner = [((radius - thickness) * math.cos(math.pi * k / count),
              spring + (radius - thickness) * math.sin(math.pi * k / count)) for k in range(count, -1, -1)]
    return outer + inner


def arch_head(half, rise):
    """A kudu arch: a horseshoe outline, wide at the shoulders and pinched at the foot."""
    shoulder = math.asin(.45)
    arc = [(half * math.cos(t), rise * math.sin(t)) for t in np.linspace(math.pi - shoulder, shoulder, 10)]
    return [(-.62 * half, 0.0), (-half, .45 * rise)] + arc + [(half, .45 * rise), (.62 * half, 0.0)]


def arch_fill(radius, spring, count=12):
    """A filled semicircular head: the tympanum that closes the end of a barrel."""
    return [(radius * math.cos(math.pi * k / count), spring + radius * math.sin(math.pi * k / count))
            for k in range(count + 1)]


class Tower(Composer):
    PALETTE = PALETTE
    T = .6                   # ashlar facing skin

    def __init__(self, request):
        super().__init__(request)
        self.gopuram = self.form.name == "dravidian-gopuram"
        if not self.gopuram:
            self.PALETTE = VIMANA_PALETTE

    def substance(self, role, kind):
        table = VIMANA if not self.gopuram else SUBSTANCES
        return table.get(role) or table.get(kind) or table["default"]

    def compose(self):
        self.count = self.control("levels", *self.form.rules["levels"])
        self.bays = self.control("bays", *self.form.rules["bays"])
        self.thick = .5                                       # thickness of a prastara
        self.eave = clamp(.028 * self.w, .35, .7)            # how far a prastara overhangs its wall
        self.pitch = 1.9 * self.vary("pitch", .1)            # pilaster bays of the talas
        if self.gopuram:
            self.ratio(self.d / self.w, "depth/width", .3, .95)
            return self.gate()
        self.ratio(self.d / self.w, "depth/width", 1.5, 2.8)
        return self.shrine()

    def storeys(self, room, fall, least):
        """Tala heights from the first to the last, falling evenly to ``fall`` of the first.

        ``fall`` is the last tala as a share of the first. With many talas in a tower of a given
        height it is raised until no tala is lower than ``least``: the profile straightens rather
        than a storey shrinking out of reason.
        """
        least *= self.count
        if 2 * room <= least + .05:
            raise ValueError(f"needs a height at least {(least + .05) / 2:.1f} m taller than its plinth, "
                             f"base and crown for {self.count} talas")
        weights = [1 - (1 - max(fall, least / (2 * room - least))) * k / (self.count - 1) for k in range(self.count)]
        return [room * k / sum(weights) for k in weights], sum(weights)

    # --------------------------------------------------------------- the gate
    def gate(self):
        """A gopuram: plinth, stone base with its passage, brick talas, and the barrel crown."""
        early = self.variant == "early-gopuram"
        w, d, h = self.w, self.d, self.h
        self.ratio(h / w, "height/width", 1.3, 3.4)
        bx, bz = w / 2 - self.eave - .1, d / 2 - self.eave - .1   # faces of the base walls
        plinth = clamp(.035 * h, 1.0, 2.0)
        base = clamp((.33 if early else .46) * self.vary("base", .1), .25, .58) * h
        self.wall = clamp(.06 * bx, 1.2, 2.6)                # structural thickness of the base walls
        self.passage = clamp(.2 * w * self.vary("passage", .12), 2.4, .5 * w - 2 * self.wall - 3.0)
        self.head = base - clamp(.12 * base, 1.3, 3.4)       # ceiling of the passage
        self.taper = (1 / 3) ** (1 / self.count) * self.vary("taper", .04)
        x = bx - clamp(.045 * w, .8, 1.6)                    # wall faces of the first tala
        z = bz - clamp(.045 * w, .5, 1.2)
        self.ratio(2 * z * self.taper ** (self.count - 1), "summit depth", 2.0, 60)
        top = self.taper ** (self.count - 1)
        crown = self.crown_height(x * top, z * top, h)
        heights, total = self.storeys(h - base - crown, .6 if early else .45, .95)
        # The mouldings of the plinth are cut back at the jambs: their sloping courses would otherwise
        # squeeze the gateway narrower than the passage above it.
        door = Voids(0)
        door.rect(0, self.passage + .44, 0, plinth, "passage")
        self.plinth(bx + .3, bz + .3, plinth, door)
        self.slab("floor", -self.passage / 2, self.passage / 2, -d / 2, d / 2, 0, .12, 1.2)
        self.gate_base(plinth, base - self.thick, bx, bz)
        self.roof(-1, bx, bz, base - self.thick, self.thick, (x, z), heights[0] * .8, early=early)
        y, levels = base, []
        for k in range(self.count):
            share = self.taper ** k
            last = k == self.count - 1
            inner = (0.0, 0.0) if last else (x * self.taper ** (k + 1), z * self.taper ** (k + 1))
            self.tala(k, x * share, z * share, y, heights[k], inner, None if last else heights[k + 1])
            levels.append(round(y + heights[k], 3))
            y += heights[k]
        self.crown(x * top, z * top, y, h)
        return self.document({
            "typology": f"{'early' if early else 'great'} gopuram", "talas": self.count,
            "base_height": round(base, 3), "passage_width": round(self.passage, 3),
            "tala_heights": [round(v, 3) for v in heights], "tala_levels": levels,
            "summit_share": round(top, 3),
            "tala_widths": [round(2 * x * self.taper ** k, 3) for k in range(self.count)],
            "hara": "kutas at the corners, shalas along each side, panjaras between",
            "crown": "a barrel (shala) over the top tala, with gilded pots along its ridge",
            "structure": "stone base on a moulded plinth; brick and lime-plaster talas, each roofed by a prastara"})

    # ------------------------------------------------------------- the shrine
    def shrine(self):
        """A vimana: plinth, sanctum and pillared hall, stone talas, and the crown with its pot."""
        barrel = self.variant == "shala-vimana"
        w, d, h = self.w, self.d, self.h
        self.ratio(h / w, "height/width", 1.4, 3.2)
        bx = w / 2 - self.eave - .1                       # faces of the sanctum walls
        hall = d - w                                       # the pillared hall stands before the tower
        zt = hall / 2                                      # the tower's centre; the hall's is the mirror of it
        plinth = clamp(.04 * h, .8, 1.8)
        steps = math.ceil(plinth / .18)
        run = steps * .3                                   # the stair that climbs to the plinth
        sanctum = clamp(.2 * h * self.vary("sanctum", .1), 5.0, .34 * h)
        self.wall = clamp(.08 * bx, 1.1, 2.0)              # how thick the walls of the cell are
        self.taper = .58 ** (1 / self.count) * self.vary("taper", .04)
        x = bx - clamp(.05 * w, .7, 1.3)                   # wall faces of the first tala
        top = self.taper ** (self.count - 1)
        crown = 1.16 * x * top * self.vary("crown", .08)
        heights, total = self.storeys(h - plinth - sanctum - crown, .62, 1.1)
        self.door = clamp(.11 * w, 1.2, 2.4)
        start = len(self.placements)
        self.plinth(bx + .3, (d - run - .1) / 2, plinth)   # one plinth runs under tower and hall
        self.shift(start, 0, (run - .1) / 2)
        self.stairs(plinth, steps)
        self.hall(plinth, min(clamp(.62 * sanctum, 2.7, 40), sanctum - 2.25), bx, zt, run)
        start = len(self.placements)
        self.sanctum(bx, plinth, sanctum, x, heights[0] * .8)
        y = plinth + sanctum
        levels = []
        for k in range(self.count):
            share = self.taper ** k
            last = k == self.count - 1
            inner = (x * top * .58, x * top * .58) if last else (x * self.taper ** (k + 1),) * 2
            self.tala(k, x * share, x * share, y, heights[k], inner, None if last else heights[k + 1])
            levels.append(round(y + heights[k], 3))
            y += heights[k]
        self.sikhara(x * top, y, crown, barrel)
        self.shift(start, 0, zt)
        return self.document({
            "typology": f"{'barrel' if barrel else 'domed'} vimana", "talas": self.count,
            "sanctum_height": round(sanctum, 3), "tala_heights": [round(v, 3) for v in heights],
            "tala_levels": levels, "summit_share": round(top, 3), "crown_height": round(crown, 3),
            "hall": "a pillared mandapa before the sanctum, roofed with a slab under its own hara",
            "hara": "kutas at the corners, shalas along each side, panjaras between",
            "crown": "a neck (griva) under a barrel (shala), with the kalasa on top" if barrel else
                     "a neck (griva) under a stepped dome, with the kalasa on top",
            "structure": "granite throughout: a walled cell, talas of pilastered wall, each roofed by a prastara"})

    def stairs(self, plinth, steps):
        """The stair down from the plinth at the front of the hall, on the axis."""
        riser, tread = plinth / steps, .3
        for j in range(steps):
            z1 = -self.d / 2 + (j + 1) * tread
            self.member("step", -1.5, 1.5, 0, (j + 1) * riser, z1 - tread, z1, step=j)

    def sanctum(self, bx, plinth, sanctum, first, after):
        """The walled cell (garbhagriha) that the tower is raised over, open to the hall by one door."""
        t, thick = self.T, self.thick
        ix, top = bx - self.wall, plinth + sanctum - thick
        door = Voids(plinth)
        door.rect(0, self.door, plinth, plinth + .62 * sanctum, "door")
        self.shell("sanctum_wall", -bx, bx, -bx, bx, plinth, top, t, {"front": door.specs}, sanctum=True)
        self.shell("sanctum_wall", -ix, ix, -ix, ix, plinth, top, t, {"front": door.specs}, sanctum=True, register="cell")
        slot = [(-self.door / 2 - .05, self.door / 2 + .05, plinth, plinth + .62 * sanctum, "x")]
        for sign in (-1, 1):                               # the core between the two skins
            lo, hi = sorted((sign * (bx - t), sign * (ix + t)))
            self.core(-(bx - t), bx - t, plinth, top, lo, hi, voids=slot if sign < 0 else (), sanctum=True)
            self.core(*sorted((sign * (bx - t), sign * (ix + t))), plinth, top, -(ix + t), ix + t, sanctum=True)
        self.slab("floor", -ix + t, ix - t, -ix + t, ix - t, plinth, plinth + .12, 1.2)
        self.dress(bx, bx, plinth, top, self.pitch, door=self.door, sanctum=True)
        self.roof(-1, bx, bx, top, thick, (first, first), after)

    def sikhara(self, x, y, height, barrel):
        """The neck (griva) of a vimana and its crown (shikhara), with the kalasa on top."""
        g, t = x * .58, self.T                       # the neck is set well inside the top tala
        neck, base = .3 * height, y + .3 * height
        self.shell("griva", -g, g, -g, g, y, base - self.thick, t)
        self.core(-g + t, g - t, y, base - self.thick, -g + t, g - t)
        self.dress(g, g, y, base - self.thick, 1.2, neck=True)
        self.prastara(g, g, base - self.thick, self.thick, (0.0, 0.0), neck=True)
        self.kudus(g + self.eave, g + self.eave, base - self.thick, neck=True)
        if barrel:
            radius = g * .9
            self.shala(0, 0, base, 2 * g - .3, 2 * radius + .08, clamp(.45 * height - radius, .4, 1.4), "x",
                      roof_role="sikhara", wall_role="griva", crown=True)
            tip = base + 1.3 * clamp(.45 * height - radius, .4, 1.4) + radius
        else:
            for k in range(5):                       # a dome raised in five steps
                r0, r1 = g * .92 * (1 - .17 * k), g * .92 * (1 - .17 * (k + 1))
                ya, yb = base + .45 * height * k / 5, base + .45 * height * (k + 1) / 5
                self.frustum("sikhara", 0, 0, ya, yb, r0, r1, sides=8, rot=math.pi / 8, rise=yb - ya, step=k)
            tip = base + .45 * height
        self.kalasa(0, 0, tip - .08, .25 * height, .2 * height, crown=True)

    def hall(self, plinth, height, bx, zt, run):
        """The pillared hall (mandapa) before the sanctum: piers in two rows under a stone roof with its own hara."""
        p = clamp(.055 * self.w, .5, .9)
        front, back = -self.d / 2 + run + p / 2 + .4, zt - bx - p / 2 - .5
        top = plinth + height
        reach = bx - p / 2 - .5                         # the outer pier centres
        xs = [-reach + 2 * reach * k / (self.bays - 1) for k in range(self.bays)]
        for z in (front, back):
            for x in xs:
                self.stack("mandapa_pier", x, plinth, z, p, p, height - .55, mandapa=True)
                self.member("mandapa_beam", x - p * .68, x + p * .68, top - .55, top - .42, z - p * .68, z + p * .68,
                            part="capital", mandapa=True)
            for a, b in zip(xs, xs[1:]):                # architraves from pier to pier
                self.member("mandapa_beam", a + p * .6, b - p * .6, top - .42, top, z - p * .3, z + p * .3, mandapa=True)
        for x in (xs[0], xs[-1]):                       # and down each side
            self.member("mandapa_beam", x - p * .3, x + p * .3, top - .42, top, front + p * .6, back - p * .6, mandapa=True)
        z0, z1, half = -self.d / 2 + .3, zt - bx - self.eave, bx + self.eave - .2
        self.ashlar("mandapa_roof", (0, top + .25, (z0 + z1) / 2), (2 * half, .5, z1 - z0), mandapa=True)
        start = len(self.placements)
        self.hara(half, (z1 - z0) / 2, top + .5, .85, sides="flr", mandapa=True)
        self.shift(start, 0, (z0 + z1) / 2)             # no parapet along the back: the sanctum rises there
        self.slab("floor", -bx, bx, front - p, back + p, plinth, plinth + .12, 1.2)

    def plinth(self, hx, hz, height, door=None):
        """The adhishthana: a broad foot, a rounded cushion, a recessed band and a projecting roll."""
        t, depth = self.T, clamp(.2 * height, .25, .5)
        layers = (("foot", 0, .28, 0, 0), ("cushion", .28, .46, 1, .1), ("cushion", .46, .64, .1, 1),
                  ("band", .64, .86, 1.1, 1.1), ("roll", .86, 1, 1.1, -.2))
        faces = {face: door.specs for face in ("front", "back")} if door else None
        slot = [(-self.passage / 2 - .27, self.passage / 2 + .27, 0, height, "x")] if door else []
        for name, a, b, foot, head in layers:
            y0, y1 = a * height, b * height
            wide, narrow = min(foot, head), max(foot, head)      # the insets at the wide end and the narrow one
            fx, fz = hx - wide * depth, hz - wide * depth        # every moulding is set to its wide end
            if foot == head:
                warp = None
            elif foot > head:                                    # wider at the head: draw the foot in
                warp = flare(0, 0, y0, y1, (hx - narrow * depth) / fx, (hz - narrow * depth) / fz)
            else:                                                # wider at the foot: draw the head in
                warp = taper(0, 0, y0, y1, (hx - narrow * depth) / fx, (hz - narrow * depth) / fz)
            self.shell("plinth", -fx, fx, -fz, fz, y0, y1, t, faces, warp=warp, layer=name)
            self.core(-fx + t, fx - t, y0, y1, -fz + t, fz - t, voids=slot, warp=warp, role="base_core", layer=name)

    def gate_base(self, y0, top, bx, bz):
        """The stone base of a gopuram: coursed granite skins on a thick core, pierced by the passage."""
        t, thick, passage = self.T, self.wall, self.passage
        door = Voids(y0)
        door.rect(0, passage, y0, self.head - y0, "passage")
        self.shell("base_wall", -bx, bx, -bz, bz, y0, top, t, {"front": door.specs, "back": door.specs})
        for sign in (-1, 1):
            self.core(*sorted((sign * (bx - t), sign * (passage / 2 + t))), y0, top, -bz + t, bz - t, role="base_core")
            self.run("passage_wall", "z", 0, sign * (passage / 2 + t / 2), 2 * bz - 2 * t, y0, self.head - .3, t)
        self.core(-passage / 2, passage / 2, self.head, top, -bz + t, bz - t, role="base_core")
        self.ashlar("passage_ceiling", (0, self.head - .15, 0), (passage + 2 * t, .3, 2 * bz - 2 * t))
        # Pilasters flank the passage and then march at even spacing to the corners. The base is one
        # storey in the earliest gateways and two in the great ones; a string course marks the division.
        first = passage / 2 + .3
        pitch = (bx - first) / max(2, self.bays - 1)
        width = clamp(.16 * pitch, .5, 1.1)
        storeys = 1 if self.variant == "early-gopuram" else 2
        bands = [(y0 + (top - y0) * k / storeys, y0 + (top - y0) * (k + 1) / storeys) for k in range(storeys)]
        for face, sign in (("front", -1), ("back", 1)):
            reach = [first + width / 2 + k * pitch for k in range(40) if first + width + k * pitch <= bx - .25]
            marks = sorted({-v for v in reach} | set(reach))
            self.pilaster_line("x", marks, bz, sign, y0, top, width, base=True)
            edges = sorted({-bx, *marks, bx})
            for a, b in zip(edges, edges[1:]):
                middle = (a + b) / 2
                if abs(middle) < passage / 2 + .5:
                    continue                                  # the passage has no niche
                for low, high in bands:
                    self.niche("x", middle, sign * bz, sign, low, high, b - a - width, base=True)
        for sign in (-1, 1):
            count = max(1, round(2 * bz / pitch))
            self.pilaster_line("z", [-bz + 2 * bz * k / count for k in range(count + 1)], bx, sign, y0, top, width, base=True)
            for k in range(count):
                for low, high in bands:
                    self.niche("z", -bz + 2 * bz * (k + .5) / count, sign * bx, sign, low, high, 2 * bz / count - width, base=True)
        for low, _ in bands[1:]:
            for sign in (-1, 1):
                a, b = sorted((sign * passage / 2, sign * bx))
                self.run("string_course", "x", (a + b) / 2, sign * (bz + .12), b - a, low - .14, low + .14, .3)
                self.run("string_course", "z", 0, sign * (bx + .12), 2 * bz, low - .14, low + .14, .3)

    def pilaster_line(self, axis, marks, face, sign, y0, y1, width, **params):
        """Pilasters of ``width`` on ``marks`` along a wall face, keyed into it and standing 0.22 proud."""
        for along in marks:
            self.run("pilaster", axis, along, face + sign * (.11 - .25), width, y0, y1, .5,
                      face=(axis, sign), **params)

    def niche(self, axis, along, cross, sign, y0, y1, width, **params):
        """A koshta niche: jambs, a sill and a pediment around a standing figure, set forward of the wall."""
        height = y1 - y0
        span = max(.4, width - .9)
        sill = clamp(height * .16, .3, .9)
        head = clamp(height * .1, .2, .5)
        stand = clamp(height * .58, .9, 2.6)

        def box(role, a0, a1, ya, yb, depth, **tag):
            c0, c1 = sorted((cross, cross + sign * depth))
            if axis == "x":
                self.member(role, along + a0, along + a1, ya, yb, c0, c1, **tag)
            else:
                self.member(role, c0, c1, ya, yb, along + a0, along + a1, **tag)

        for side in (-1, 1):
            box("niche", side * span / 2 - .09, side * span / 2 + .09, y0 + sill, y1 - head, .16, **params)
        box("niche", -span / 2 - .09, span / 2 + .09, y0, y0 + sill, .22, **params)
        box("niche", -span / 2 - .18, span / 2 + .18, y1 - head, y1, .34, part="pediment", **params)
        radius = clamp(span * .2, .18, .42)
        fx, fz = (along, cross + sign * radius * .55) if axis == "x" else (cross + sign * radius * .55, along)
        self.add("figure", (fx, y0 + sill + stand / 2, fz), (2 * radius, stand, 2 * radius), "CYLINDER",
                 radius=radius, height=stand, sections=12, **params)

    # --------------------------------------------------------------- the talas
    def tala(self, k, x, z, y, height, inner, after):
        """One storey: coursed wall between pilasters, roofed by a prastara that carries the hara."""
        t = self.T
        top = y + height - self.thick
        self.shell("tala_wall", -x, x, -z, z, y, top, t, tala=k)
        self.core(-x + t, x - t, y, top, -z + t, z - t, tala=k)
        self.dress(x, z, y, top, self.pitch, tala=k)
        self.roof(k, x, z, top, self.thick, inner, after)

    def dress(self, x, z, y0, y1, pitch, door=0.0, **tag):
        """Pilasters at even spacing along each face of a rectangle, and a niche in every bay wide enough.

        ``door`` clears the middle of the front face for a doorway.
        """
        for axis, half, across in (("x", x, z), ("z", z, x)):
            count = max(1, round(2 * half / pitch))
            bay, width = 2 * half / count, clamp(.14 * 2 * half / count, .28, .9)
            for sign in (-1, 1):
                blocked = lambda a: bool(door) and axis == "x" and sign < 0 and abs(a) < door / 2 + .3   # noqa: E731
                self.pilaster_line(axis, [a for a in (-half + bay * j for j in range(count + 1))
                                          if not blocked(a) or abs(a) >= half - 1e-9], across, sign, y0, y1, width, **tag)
                for j in range(count):
                    middle = -half + bay * (j + .5)
                    if bay >= 1.25 and y1 - y0 >= 1.5 and not blocked(middle):
                        self.niche(axis, middle, sign * across, sign, y0, y1, bay - width, **tag)

    def roof(self, k, x, z, y, thick, inner, after, **params):
        """What caps a storey: its prastara with kudu arches along the eave, and the hara on the slab."""
        self.prastara(x, z, y, thick, inner, tala=k, **params)
        self.kudus(x + self.eave, z + self.eave, y, tala=k, **params)
        depth = (x - inner[0]) + self.eave
        u = min(.62 * depth - .08, (after - thick) / 1.35 if after else 1.15, 2.2)
        if u >= .32:
            self.hara(x + self.eave, z + self.eave, y + thick, u, tala=k, **params)

    def prastara(self, x, z, y, thick, inner, **params):
        """The flat roof slab of a storey: out past the eave, and in to the foot of the next storey."""
        ix, iz = max(inner[0] - .4, 0.0), max(inner[1] - .4, 0.0)
        ox, oz = x + self.eave, z + self.eave
        for sign in (-1, 1):                       # front and back strips span the full width
            edge, inside = sign * oz, sign * iz
            strips = max(1, math.ceil(abs(edge - inside) / 1.4))
            for k in range(strips):
                a, b = edge + (inside - edge) * k / strips, edge + (inside - edge) * (k + 1) / strips
                self.run("prastara", "x", 0, (a + b) / 2, 2 * ox, y, y + thick, abs(b - a), **params)
        if iz > .2:                                # side strips fill in between them
            for sign in (-1, 1):
                edge, inside = sign * ox, sign * ix
                strips = max(1, math.ceil(abs(edge - inside) / 1.4))
                for k in range(strips):
                    a, b = edge + (inside - edge) * k / strips, edge + (inside - edge) * (k + 1) / strips
                    self.run("prastara", "z", 0, (a + b) / 2, 2 * iz, y, y + thick, abs(b - a), **params)

    def kudus(self, x, z, y, **params):
        """Kudu arches in relief along the eave face of a prastara."""
        half, rise, depth = .2, self.thick * .78, .12
        for axis, span, across in (("x", x, z), ("z", z, x)):
            count = max(1, round(2 * span / .95))
            for sign in (-1, 1):
                cross = sign * across - sign * (depth / 2 - .02)
                for k in range(count):
                    along = -span + 2 * span * (k + .5) / count
                    head = [(along + a, y + .04 + b) for a, b in arch_head(half, rise)]
                    self.mesh("kudu", oriented_profile(head, depth, axis, 0.0, cross), face=(axis, sign), **params)


    # --------------------------------------------------------------- the hara
    def crown_height(self, x, z, h):
        """How high the barrel and its pots rise above the top tala."""
        return 1.3 * clamp(.05 * h, 1.0, 2.6) + (z - clamp(.34 * z, .45, 1.3) - .26) + clamp(.035 * h, .9, 2.2)

    def crown_plan(self, x, z, h):
        """The barrel over the top tala, between kutas at its corners, and the pots on its ridge."""
        return (clamp(.34 * z, .45, 1.3), 2 * (x - clamp(.34 * z, .45, 1.3) - .22),
                2 * (z - clamp(.34 * z, .45, 1.3) - .22), clamp(.05 * h, 1.0, 2.6), clamp(.035 * h, .9, 2.2))

    def crown(self, x, z, y, h, **params):
        """The crown of a gopuram: kutas at the corners of the top slab, a barrel between, and pots on its ridge."""
        u, length, width, wall, pot = self.crown_plan(x, z, h)
        for sx in (-1, 1):
            for sz in (-1, 1):
                self.kuta(sx * (x - u / 2 - .12), sz * (z - u / 2 - .12), y, u, u * 1.35, crown=True)
        self.shala(0, 0, y, length, width, wall, "x", crown=True)
        for along in (-1, 0, 1):                  # a pot at each end of the ridge and one on the crown
            self.kalasa(along * (length / 2 - width * .6), 0, y + 1.3 * wall + width / 2 - .16, pot, pot * .38,
                        crown=True, ridge=along)

    def hara(self, x, z, y, u, sides="fblr", **params):
        """The parapet on a prastara. ``(x, z)`` bound its outer edge and ``u`` is the size of a kuta.

        A kuta stands at each corner. Between them, along each side, run panjaras and shalas in
        alternation: a panjara next to each kuta, then a shala, and so on to the far kuta.
        """
        inset = u / 2 + .15
        for sx in (-1, 1):
            for sz in (-1, 1):
                self.kuta(sx * (x - inset), sz * (z - inset), y, u, u * 1.35, corner=True, **params)
        for axis, half, across in (("x", x, z), ("z", z, x)):
            free = 2 * (half - u - .15)          # along the side, between the corner kutas
            if free < 3.4 * u:
                continue
            count = max(1, round((free / u - .9) / 3.6))

            def gaps(count):
                return (free - count * 4.0 * u) / (3 * count - 1)

            if gaps(count) < .05 * u and count > 1:
                count -= 1
            if gaps(count) > 1.4 * u:
                count += 1
            gap = gaps(count)

            def place(role, along, size, height=None, length=None, **tag):
                centre = (along, sign * (across - inset)) if axis == "x" else (sign * (across - inset), along)
                if role == "panjara":
                    self.panjara(*centre, y, size, axis, **params, **tag)
                else:
                    self.shala(*centre, y, length, size, height, axis, **params, **tag)

            for sign in (-1, 1):
                if ("f" if sign < 0 else "b") not in (sides if axis == "x" else sides.translate(str.maketrans("fb", "lr"))):
                    continue
                along = -free / 2
                for k in range(count):
                    place("panjara", along + .45 * u, .9 * u)
                    along += .9 * u + gap
                    place("shala", along + 1.1 * u, 1.1 * u, .7 * u, 2.2 * u)
                    along += 2.2 * u + gap
                    if k < count - 1:
                        place("panjara", along + .45 * u, .9 * u)
                        along += .9 * u + gap
                place("panjara", along + .45 * u, .9 * u)

    def kuta(self, cx, cz, y, size, height, **params):
        """A square miniature shrine of the hara: a seat, four posts, a plate, a stepped cap and a pot."""
        s, post = size / 2, size * .18
        if size < .5:                              # too small to stand open: a block will do
            self.ashlar("kuta_base", (cx, y + height * .29, cz), (size, height * .58, size), **params)
        else:
            self.ashlar("kuta_base", (cx, y + height * .09, cz), (size, height * .18, size), **params)
            for sx in (-1, 1):
                for sz in (-1, 1):
                    self.stack("kuta_post", cx + sx * (s - post / 2), y + height * .18, cz + sz * (s - post / 2),
                               post, post, height * .32, **params)
            self.ashlar("kuta_roof", (cx, y + height * .54, cz), (size, height * .08, size), **params)
        for k in range(3):
            width = size * (1 - .27 * k)
            self.member("kuta_roof", cx - width / 2, cx + width / 2, y + height * (.58 + .1 * k),
                        y + height * (.68 + .1 * k), cz - width / 2, cz + width / 2, step=k, **params)
        self.kalasa(cx, cz, y + height * .88, height * .12, size * .14, **params)

    def panjara(self, cx, cz, y, size, axis, **params):
        """A low niched block that fills the hara between the kutas and the shalas."""
        along, across = (.9 * size, .72 * size) if axis == "x" else (.72 * size, .9 * size)
        self.ashlar("panjara", (cx, y + size * .3, cz), (along, size * .6, across), **params)
        self.member("kuta_roof", cx - along * .58, cx + along * .58, y + size * .6, y + size * .76,
                    cz - across * .58, cz + across * .58, **params)

    def shala(self, cx, cz, y, length, width, wall, axis, roof_role="shala_roof", wall_role="shala_wall", **params):
        """An oblong pavilion roofed with a barrel that runs along ``axis``.

        In a hara it is the shala proper. A vimana may be crowned by one, and names its parts otherwise.
        """
        def at(a, c):
            return (cx + a, cz + c) if axis == "x" else (cx + c, cz + a)

        def dims(along, up, across):
            return (along, up, across) if axis == "x" else (across, up, along)

        def block(role, a, c, along, up, across, base, **tag):
            x, z = at(a, c)
            self.ashlar(wall_role if role == "shala_wall" else role, (x, base + up / 2, z), dims(along, up, across), **tag)

        def tympanum(profile, depth, a):
            if axis == "x":
                self.mesh(wall_role, oriented_profile(profile, depth, "z", cz, cx + a), **params)
            else:
                self.mesh(wall_role, oriented_profile(profile, depth, "x", cx, cz + a), **params)

        seat, spring = .3 * wall, y + 1.3 * wall
        radius = width / 2 - .04
        post, end = clamp(width * .16, .1, .4), clamp(width * .14, .1, .4)
        block("shala_wall", 0, 0, length, seat, width, y, **params)                # the seat
        for sign in (-1, 1):                                                        # the two end walls
            block("shala_wall", sign * (length / 2 - end / 2), 0, end, wall, width, y + seat, **params)
            tympanum(arch_fill(radius, spring), end, sign * (length / 2 - end / 2))
        bays = max(1, round((length - 2 * end) / (width * .9)))
        for sign in (-1, 1):                                                        # a row of posts down each side
            for k in range(bays + 1):
                x, z = at(-length / 2 + end + (length - 2 * end) * k / bays, sign * (width / 2 - post / 2))
                self.stack("kuta_post", x, y + seat, z, post, post, wall, **params)
            block("shala_wall", 0, sign * (width / 2 - post / 2), length - 2 * end, .16 * wall, post, spring - .16 * wall,
                  **params)                                                         # the wall plate they carry
        a0, a1, cross = (cx - length / 2 + end, cx + length / 2 - end, cz) if axis == "x" else \
            (cz - length / 2 + end, cz + length / 2 - end, cx)
        self.vault(roof_role, axis, a0, a1, cross, spring, radius, clamp(width * .1, .06, .3), **params)
        if not params.get("crown"):
            for sign in (-1, 1):                   # a small pot over each end of the ridge
                x, z = at(sign * (length / 2 - end / 2), 0)
                self.kalasa(x, z, spring + radius - .04, radius * .5, radius * .24, **params)

    def vault(self, role, axis, a0, a1, cross, spring, radius, thickness, **params):
        """A barrel vault of arch bands from ``a0`` to ``a1`` along ``axis``, springing at ``spring``."""
        band = arch_band(radius, thickness, spring)
        pieces = max(1, math.ceil((a1 - a0) / 1.2))
        for k in range(pieces):
            middle = a0 + (a1 - a0) * (k + .5) / pieces
            if axis == "x":
                self.mesh(role, oriented_profile(band, (a1 - a0) / pieces, "z", cross, middle), band=k, **params)
            else:
                self.mesh(role, oriented_profile(band, (a1 - a0) / pieces, "x", cross, middle), band=k, **params)

    def kalasa(self, cx, cz, y, height, radius, **params):
        """A pot finial: a swelling body on a foot, a neck, and a flared lip."""
        shape = (((0, .5), (.3, 1.0), (.58, .9), (.74, .46), (.84, .42), (1.0, .6)) if height < 1.2 else
                 ((0, .5), (.12, .74), (.3, 1.0), (.48, .99), (.62, .84), (.72, .55), (.8, .44), (.88, .58), (1.0, .62)))
        self.lathe("kalasa", cx, cz, [(y + a * height, b * radius) for a, b in shape], sides=12, **params)

Tower.module = sys.modules[__name__]


def build_document(request):
    return Tower(request).compose()


def random_controls(product, rng):
    """A tower's height suits its plan and its talas, so a random one is the default grown or shrunk whole."""
    width, height, depth = FORMS[product].defaults
    scale = rng.uniform(.85, 1.2)
    return {"width": round(width * scale), "height": round(height * scale), "depth": round(depth * scale),
            "bays": FORMS[product].controls.get("bays"), "levels": FORMS[product].controls.get("levels"),
            "detail": rng.choice(("concept", "standard"))}


def _roof(p):
    """Which roof a member of a prastara, its kudus or its hara belongs to, read from its tags."""
    return tuple(p.parameters.get(key) for key in ("tala", "mandapa", "neck"))


def _corners(members, centre):
    """Members grouped by the corner of the plan they stand in, about ``centre``: (x > 0, z > 0)."""
    groups = {}
    for p in members:
        groups.setdefault((p.position[0] > centre[0], p.position[2] > centre[2]), []).append(p)
    return groups


def rule_checks(document, form):
    """Rules read from the members themselves, not from metadata."""
    placements = document.placements
    by_role = {}
    for p in placements:
        by_role.setdefault(p.role, []).append(p)
    gopuram = "gopuram" in form.name
    checks = []

    def top(p):
        return p.position[1] + p.dimensions[1] / 2

    def bottom(p):
        return p.position[1] - p.dimensions[1] / 2

    # Every tala stands inside the one below and is lower than it; the summit is a set share of the base.
    talas = {}
    for p in by_role.get("tala_wall", []):
        talas.setdefault(p.parameters.get("tala"), []).append(p)
    order = sorted(k for k in talas if k is not None)
    if len(order) >= 3:
        lo = [extent(talas[k])[0] for k in order]
        hi = [extent(talas[k])[1] for k in order]
        wide = [b[0] - a[0] for a, b in zip(lo, hi)]
        deep = [b[2] - a[2] for a, b in zip(lo, hi)]
        tall = [b[1] - a[1] for a, b in zip(lo, hi)]
        ok = all(a > b + .05 for a, b in zip(wide, wide[1:])) and all(a > b + .05 for a, b in zip(deep, deep[1:])) \
            and all(a > b + .02 for a, b in zip(tall, tall[1:]))
        checks.append(("diminishing_talas", ok, "Every tala stands inside the one below it and is lower than it.",
                       "A tala no longer stands inside the one below it, or is taller than it.",
                       [p.id for k in order for p in talas[k]] if not ok else []))
        share = wide[-1] / wide[0]
        ok = (.25 if gopuram else .4) <= share <= (.6 if gopuram else .75)
        checks.append(("summit_share", ok, "The summit is about a third of the base." if gopuram else "The summit is about three fifths of the base.",
                       "The summit is no longer in proportion to the base.", [p.id for p in talas[order[-1]]] if not ok else []))

    # Every prastara carries its hara: a kuta at each corner, and shalas along each side wide enough for them.
    roofs = {}
    for p in by_role.get("prastara", []):
        roofs.setdefault(_roof(p), []).append(p)
    bare, uneven = [], []
    for k, members in sorted(roofs.items(), key=lambda item: str(item[0])):
        if k[2]:                                     # the neck's cap carries the crown, not a parapet
            continue
        kutas = [p for p in by_role.get("kuta_base", []) if _roof(p) == k and p.parameters.get("corner")]
        shalas = [p for p in by_role.get("shala_roof", []) if _roof(p) == k]
        lo, hi = extent(members)
        if len(_corners(kutas, (lo + hi) / 2)) != 4:
            bare += [p.id for p in members]
        u = max((p.dimensions[0] for p in kutas), default=0.0)
        sides = 3 if k[1] else 4                          # a hall roof has no parapet along its back
        if u and min(hi[0] - lo[0], hi[2] - lo[2]) >= 5.5 * u + .3 and len(shalas) < sides:
            uneven += [p.id for p in members]
    checks.append(("hara_each_tala", not bare and not uneven, "Every prastara carries a hara: kutas at its corners and shalas along its sides.",
                   "A prastara has lost the kutas at its corners, or the shalas along its sides.", bare + uneven))

    # Kudu arches run along the eave of every prastara, on all four faces.
    plain = []
    for k, members in sorted(roofs.items(), key=lambda item: str(item[0])):
        faces = {tuple(p.parameters.get("face", ())) for p in by_role.get("kudu", []) if _roof(p) == k}
        if len(faces) < 4:
            plain += [p.id for p in members[:4]]
    checks.append(("kudu_each_eave", not plain, "Kudu arches run along every eave.",
                   "An eave has lost its kudu arches.", plain))

    # The pilasters of each tala march at even spacing along every face.
    ragged = []
    pitched = {}
    for p in by_role.get("pilaster", []):
        face = p.parameters.get("face")
        if face and p.parameters.get("tala") is not None:
            pitched.setdefault((p.parameters["tala"], tuple(face), p.parameters.get("host_id")), []).append(p)
    faces = {}
    for (tala, face, _), stones in pitched.items():
        axis = 0 if face[0] == "x" else 2
        centre = sum(p.position[axis] for p in stones) / len(stones)
        faces.setdefault((tala, face), []).append((centre, stones))
    for marks in faces.values():
        gaps = [b[0] - a[0] for a, b in zip(sorted(marks), sorted(marks)[1:])]
        if len(gaps) >= 3 and max(gaps) > 1.15 * min(gaps):
            ragged += [p.id for _, stones in marks for p in stones]
    checks.append(("pilasters_march", not ragged, "The pilasters stand at even spacing along every face.",
                   "The pilasters no longer stand at even spacing.", ragged))

    # The pots that finish the tower: mirrored on both axes, and the highest thing on it.
    pots = by_role.get("kalasa", [])
    lorn = unmirrored(pots, {"kalasa"}) + (unmirrored(pots, {"kalasa"}, axis=2) if gopuram else [])
    highest = max((top(p) for p in placements), default=0.0)
    ok = bool(pots) and not lorn and abs(max(top(p) for p in pots) - highest) < 1e-3
    checks.append(("kalasa_crowns_tower", ok, "The kalasas are mirrored across the tower, and crown it.",
                   "A kalasa has lost its twin, or the tower rises above its kalasas.", sorted(set(lorn)) if not ok else []))

    # A gopuram is a gate: its passage is open from front to back.
    if gopuram:
        walls = by_role.get("passage_wall", [])
        if walls:
            half = min(abs(p.position[0]) - p.dimensions[0] / 2 for p in walls)
            head = max(top(p) for p in walls)                # the walls run up to the passage's ceiling
            depth = max(abs(p.position[2]) + p.dimensions[2] / 2 for p in by_role.get("base_wall", []))
            blocked = []
            for p in placements:
                x0, x1 = p.position[0] - p.dimensions[0] / 2, p.position[0] + p.dimensions[0] / 2
                y0, y1 = bottom(p), top(p)
                z0, z1 = p.position[2] - p.dimensions[2] / 2, p.position[2] + p.dimensions[2] / 2
                if p.role != "floor" and x0 < half - .02 and x1 > -half + .02 and y0 < head - .02 and y1 > .14 \
                        and z0 < depth - .05 and z1 > -depth + .05:
                    blocked.append(p.id)
            checks.append(("passage_open", not blocked and half >= 1.0, "The passage runs open through the base from front to back.",
                           "Something stands in the passage, or the passage has closed.", blocked))
    return checks
