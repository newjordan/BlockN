"""Lamu-inspired Swahili stone houses with inward-facing courts and rooms."""
from __future__ import annotations

import math
import sys
from collections import defaultdict

from .historic import Composer, Form, Voids, extent, number
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("swahili", "1.0.0", "Swahili Coast Architect")
STYLE = "swahili-stone-house"
PRODUCT = "swahili-courtyard-house"
REGION = "Lamu, Kenya (Swahili Coast)"
PERIOD = "Swahili stone-town domestic tradition, 18th–19th centuries"
VARIANTS = ("single-court", "double-court")
KINDS = {
    "foundation": "foundation", "side_wall": "wall", "room_wall": "wall", "entry_screen": "wall",
    "entry_lintel": "beam", "carved_frame": "ornament", "baraza": "slab", "court_paving": "slab",
    "floor_joist": "beam", "upper_floor": "slab", "roof_joist": "beam", "roof_deck": "roof",
    "gallery_post": "column", "gallery_beam": "beam", "gallery_canopy": "roof",
}
FORMS = {PRODUCT: Form(
    PRODUCT, "Lamu-inspired coral-rag courtyard house with a screened entrance and carved timber portal",
    (12, 7, 28), VARIANTS, frozenset(KINDS), {"bays": 3},
    rules={"region": REGION, "period": PERIOD, "bays": [2, 4],
           "bays_meaning": "doorways across each court-facing room front",
           "width": [10, 18], "height": [6, 10], "depth": [23, 45],
           "depth/width": [1.65, 3.9], "courts": {"single-court": 1, "double-court": 2}},
)}
DEFAULT_SUBSTANCE = "coral-limestone"
SUBSTANCES = {**dict.fromkeys(("entry_lintel", "carved_frame", "floor_joist", "roof_joist",
                               "gallery_post", "gallery_beam", "gallery_canopy"), "hardwood"),
              "upper_floor": "lime-mortar", "roof_deck": "lime-mortar"}
TOL = 1e-5


class House(Composer):
    BASE, WALL, DOOR = .16, .48, 2.20
    TIMBER_KINDS = frozenset({"beam", "column"})
    PALETTE = {"wall": (217, 205, 181, 255), "foundation": (159, 145, 129, 255),
               "court_paving": (177, 156, 132, 255), "baraza": (202, 187, 154, 255),
               "roof_deck": (230, 217, 179, 255), "upper_floor": (203, 178, 143, 255),
               "carved_frame": (102, 57, 37, 255), "gallery_canopy": (165, 118, 75, 255),
               "beam": (110, 74, 51, 255), "column": (107, 69, 45, 255)}

    def box(self, role, x0, x1, y0, y1, z0, z1, **tags):
        self.add(role, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 (x1 - x0, y1 - y0, z1 - z0), **tags)

    def compose(self):
        for name, value in zip(("width", "height", "depth"), (self.w, self.h, self.d)):
            number(value, f"{PRODUCT} {name}", *self.form.rules[name])
        self.ratio(self.d / self.w, "depth/width", *self.form.rules["depth/width"])
        self.bays = self.control("bays", 2, 4)
        self.clear = self.w / 2 - self.WALL
        self.front, self.back = -self.d / 2 + .9, self.d / 2
        self.eaves = self.h - .52
        self.upper = (self.h - self.BASE) * .49 + self.BASE
        span = self.back - self.front
        double = self.variant == "double-court"
        # Keep passages and courts legible at both ends of the size envelope.
        a = self.front + span * (.225 if double else .25) * self.vary("front-range", .10)
        b = self.back - span * (.24 if double else .40) * self.vary("rear-range", .08)
        if double:
            middle = self.front + span * .53 * self.vary("middle-range", .025)
            ranges = [(self.front, a), (middle - span * .075, middle + span * .075), (b, self.back)]
        else:
            ranges = [(self.front, a), (b, self.back)]
        self.slab("foundation", -self.w / 2, self.w / 2, -self.d / 2, self.back,
                  0, self.BASE, module=2.0)
        for side in (-1, 1):
            cross = side * (self.clear + self.WALL / 2)
            self.ashlar("side_wall", (cross, (self.BASE + self.h) / 2, (self.front + self.back) / 2),
                        (self.WALL, self.h - self.BASE, span), axis="z")
        for i, (lo, hi) in enumerate(ranges):
            self.range(i, lo, hi, i == len(ranges) - 1)
        for i, ((_, lo), (hi, _)) in enumerate(zip(ranges, ranges[1:])):
            self.court(i, lo, hi)
        self.entry(a)
        return self.document({"region": REGION, "period": PERIOD, "open_courts": len(ranges) - 1,
                              "court_door_bays": self.bays, "street_axis": "-Z"})

    def crosswall(self, role, z, top, entries=(), windows=(), **tags):
        openings = Voids(self.BASE)
        for j, x in enumerate(entries):
            openings.rect(x, 1.16, self.BASE, self.DOOR - self.BASE, f"{role}.{self.counts[role]}.d{j}")
        for j, x in enumerate(windows):
            openings.rect(x, .68, self.upper + .38, .87, f"{role}.{self.counts[role]}.w{j}")
        lights = tuple(openings.specs)
        for op in lights:
            y = self.BASE + op["sill"] + op["height"]
            openings.rect(op["center"], op["width"] + .32, y, .18, op["id"] + ".seat")
        self.ashlar(role, (0., (self.BASE + top) / 2, z), (2 * self.clear, top - self.BASE, self.WALL),
                    axis="x", openings=openings.specs, **tags)
        for op in lights:
            y = self.BASE + op["sill"] + op["height"]
            x, half = op["center"], op["width"] / 2 + .16
            self.box("entry_lintel", x - half, x + half, y, y + .18,
                     z - self.WALL / 2, z + self.WALL / 2, opening=op["id"], **tags)

    def range(self, i, lo, hi, rear):
        bays = tuple(-self.clear + (k + .5) * 2 * self.clear / self.bays for k in range(self.bays))
        west, east = -self.clear, self.clear
        front = (-self.clear * .47,) if i == 0 else bays
        self.crosswall("room_wall", lo + self.WALL / 2, self.h, front, front, range_id=i,
                       edge="street" if i == 0 else "court")
        self.crosswall("room_wall", hi - self.WALL / 2, self.h, () if rear else bays,
                       () if rear else bays, range_id=i, edge="back" if rear else "court")
        joists = math.ceil((east - west) / .65)
        for n in range(joists):
            x = west + (n + .5) * (east - west) / joists
            for role, top in (("floor_joist", self.upper), ("roof_joist", self.eaves)):
                self.box(role, x - .075, x + .075, top - .20, top, lo + .10, hi - .10, range_id=i)
        for role, top in (("upper_floor", self.upper), ("roof_deck", self.eaves)):
            self.slab(role, west, east, lo + .15, hi - .15, top, top + .15, module=1.55, range_id=i)

    def court(self, i, lo, hi):
        self.slab("court_paving", -self.clear, self.clear, lo, hi, self.BASE, self.BASE + .07,
                  module=1.55, court=i)
        overhang = min(.86, (hi - lo) * .23)
        head = self.BASE + 2.52
        for side, z in ((-1, lo), (1, hi)):
            edge = z - side * overhang
            a, b = sorted((edge, z))
            self.box("gallery_canopy", -self.clear, self.clear, head, head + .13, a, b, court=i)
            self.box("gallery_beam", -self.clear, self.clear, head - .16, head,
                     edge - .07, edge + .07, court=i)
            for x in (-self.clear + .20, self.clear - .20):
                self.box("gallery_post", x - .09, x + .09, self.BASE + .07, head - .16,
                         edge - .09, edge + .09, court=i)

    def entry(self, court_start):
        street_x = -self.clear * .47
        screen_x = self.clear * .48 * self.vary("entry-offset", .18)
        screen = self.front + (court_start - self.front) * .51
        self.crosswall("entry_screen", screen, self.upper - .20, (screen_x,), edge="screen")
        jamb = .58
        for side in (-1, 1):
            x = street_x + side * (jamb + .11)
            self.box("carved_frame", x - .11, x + .11, self.BASE, self.DOOR + .23,
                     self.front - .12, self.front, member="jamb")
        self.box("carved_frame", street_x - .8, street_x + .8, self.DOOR, self.DOOR + .25,
                 self.front - .12, self.front, member="head")
        for j in range(9):
            x = street_x + (j - 4) * .16
            self.box("carved_frame", x - .054, x + .054, self.DOOR + .067, self.DOOR + .183,
                     self.front - .15, self.front - .12, member="relief")
        for x0, x1 in ((-self.clear + .13, street_x - .86), (street_x + .86, self.clear - .13)):
            self.slab("baraza", x0, x1, self.front - .53, self.front, self.BASE, self.BASE + .42, module=1.1)


House.module = sys.modules[__name__]


def build_document(request):
    return House(request).compose()


def random_controls(product, rng):
    w = rng.randint(10, 15)
    return {"width": w, "height": rng.randint(6, 9), "depth": round(w * rng.uniform(2.35, 2.9)),
            "bays": rng.randint(2, 4), "detail": rng.choice(("concept", "standard"))}


def rule_checks(document, form):
    roles, courts, roofs = defaultdict(list), defaultdict(list), defaultdict(list)
    for member in document.placements:
        roles[member.role].append(member)
        if member.role == "court_paving":
            courts[member.parameters.get("court")].append(member)
        elif member.role == "roof_deck":
            roofs[member.parameters.get("range_id")].append(member)
    expected = 2 if roles["court_paving"] and roles["court_paving"][0].parameters.get("style") == "double-court" else 1
    valid = set(courts) == set(range(expected)) and set(roofs) == set(range(expected + 1))
    blocked = []
    if valid:
        for i, paving in sorted(courts.items()):
            lo, hi = extent(paving)
            z0, z1 = lo[2] + 1., hi[2] - 1.
            valid &= hi[2] - lo[2] >= 2.5
            left, right = extent(roofs[i]), extent(roofs[i + 1])
            valid &= abs(lo[2] - left[1][2] - .15) < TOL and abs(right[0][2] - hi[2] - .15) < TOL
            for p in (*roles["roof_deck"], *roles["gallery_canopy"]):
                half = p.dimensions[2] / 2
                if p.position[2] + half > z0 + TOL and p.position[2] - half < z1 - TOL:
                    blocked.append(p.id)
    yield ("open_courts", valid and not blocked, "Unroofed courts sit between enclosed room ranges.",
           "A courtyard is missing, obstructed overhead, or separated from its room ranges.", blocked)

    street = [p for p in roles["entry_lintel"] if p.parameters.get("edge") == "street" and p.position[1] < 2.5]
    screens = [p for p in roles["entry_lintel"] if p.parameters.get("edge") == "screen"]
    clear = len(street) == len(screens) == 1
    obstructions = []
    if clear:
        front, inner = street[0], screens[0]
        clear = inner.position[2] > front.position[2] + 1 and abs(inner.position[0] - front.position[0]) > 1.16
        for head, panels in ((front, roles["room_wall"]), (inner, roles["entry_screen"])):
            x, y, z = head.position
            for p in panels:
                dx, dy, dz = (d / 2 for d in p.dimensions)
                if (abs(z - p.position[2]) < dz - TOL and
                        abs(x - p.position[0]) < dx + .53 - TOL and p.position[1] - dy < y - .12):
                    obstructions.append(p.id)
    yield ("screened_entry", clear and not obstructions, "Offset doors preserve the entrance dogleg.",
           "The street and screen openings align, are missing, or are blocked.", obstructions)
