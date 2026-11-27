"""Egyptian architect: the New Kingdom cult temple.

The temple is a processional axis. A pylon of two battered towers guards the
gate; behind it an open court with shaded colonnades leads to a hypostyle hall
whose taller central nave is lit by a clerestory, and then to low, dark
sanctuary chambers. Roofs fall and floors rise toward the god.

Construction is trabeated: stone beams on stone columns, flat slab roofs, no
arch. Walls are coursed blocks, but architraves, roof slabs, lintels and
obelisks are single great stones, because the span of a slab sets the spacing
of every column in the building.
"""
from __future__ import annotations

import math
import sys

from .historic import Composer, Form, Voids, batter, clamp, extent, hull, taper, unmirrored
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("egyptian", "1.0.0", "Egyptian Architect")
STYLE = "egyptian"
KINDS = {
    "core_fill": "core",
    "pylon": "wall", "pylon_cornice": "entablature", "pylon_roof": "slab", "gate_jamb": "wall",
    "gate_lintel": "entablature", "enclosure_wall": "wall", "wall_coping": "entablature",
    "hall_wall": "wall", "clerestory": "wall", "sanctuary_wall": "wall",
    "column_drum": "column", "abacus": "column", "architrave": "entablature", "door_lintel": "entablature",
    "court_roof": "roof", "nave_roof": "roof", "aisle_roof": "roof", "sanctuary_roof": "roof",
    "floor": "slab", "ramp": "circulation", "obelisk": "ornament", "flag_mast": "ornament",
    "pennant": "ornament", "shrine": "ornament",
}
FORMS = {form.name: form for form in (
    Form("egyptian-temple", "Axial cult temple: pylon, colonnaded court, clerestoried hypostyle hall, sanctuary",
         (30, 24, 72), ("papyrus-bud", "open-papyrus"),
         frozenset({"pylon", "pylon_cornice", "gate_lintel", "enclosure_wall", "column_drum", "architrave",
                    "clerestory", "nave_roof", "aisle_roof", "sanctuary_roof", "obelisk"}),
         rules={"height_to_width": [0.55, 1.1], "depth_to_width": [1.8, 3.2],
                "structure": "trabeated; monolithic architraves and roof slabs; no arch",
                "section": "roofs fall and floors rise from pylon to sanctuary"}),
)}
SAND = (226, 204, 160, 255)
PALETTE = {"core_fill": (176, 156, 120, 255), "wall": SAND, "column": (232, 214, 176, 255), "entablature": (214, 190, 146, 255),
           "roof": (204, 184, 146, 255), "slab": (196, 180, 150, 255), "obelisk": (206, 150, 132, 255),
           "flag_mast": (120, 86, 58, 255), "pennant": (170, 30, 36, 255), "shrine": (120, 92, 62, 255),
           "foundation": SAND, "circulation": SAND}
# New Kingdom work: coursed sandstone with granite reserved for the members
# that do the structural work -- architraves, lintels, abaci and obelisks.
# Sandstone rubble set in mortar fills the wall and pylon cores, cedar the
# masts and the naos.
SUBSTANCES = {
    "core_fill": "sandstone",
    "pylon": "sandstone", "pylon_cornice": "sandstone", "pylon_roof": "sandstone",
    "gate_jamb": "sandstone", "gate_lintel": "granite",
    "enclosure_wall": "sandstone", "wall_coping": "sandstone",
    "hall_wall": "sandstone", "clerestory": "sandstone", "sanctuary_wall": "sandstone",
    "column_drum": "sandstone", "abacus": "granite", "architrave": "granite",
    "door_lintel": "granite",
    "court_roof": "sandstone", "nave_roof": "sandstone", "aisle_roof": "sandstone",
    "sanctuary_roof": "sandstone",
    "floor": "limestone", "ramp": "limestone",
    "obelisk": "granite", "flag_mast": "cedar", "pennant": "painted-linen",
    "shrine": "cedar",
}
DEFAULT_SUBSTANCE = "sandstone"


class Temple(Composer):
    PALETTE = PALETTE
    T = 0.6

    def compose(self):
        self.ratio(self.h / self.w, "height/width", .55, 1.1)
        self.ratio(self.d / self.w, "depth/width", 1.8, 3.2)
        t, w, h = self.T, self.w, self.h
        self.grade = {"concept": "concept", "standard": "concept", "presentation": "standard"}[self.detail]
        apron = clamp(.07 * self.d, 3.5, 7.0)
        front, back = -self.d / 2 + apron, self.d / 2
        pylon_d = clamp(.27 * h * self.vary("pylon", .14), 3.5, 9.0)
        gate = clamp(.07 * w * self.vary("gate", .15), 1.6, 3.0)       # half the gap between the towers
        nave_h = .58 * h * self.vary("nave", .06)
        aisle_h = .7 * nave_h * self.vary("aisle", .06)
        low, high = aisle_h + 1.4, nave_h + 1.4          # roof tops
        inner = front + pylon_d                           # court begins
        length = back - inner
        hall0 = inner + .34 * length * self.vary("court", .14)
        hall1 = hall0 + .34 * length * self.vary("hall", .1)
        half, wall_t = w / 2, 1.6
        clear = half - wall_t
        floors = (.3, .6, .9)

        self.pylon(front, pylon_d, gate, h, half)
        self.forecourt(front, gate, h, half)

        # Enclosure: outer face battered, inner face plumb.
        sanct_h = .8 * low
        for z0, z1, top, role in ((inner, hall1, low, "enclosure_wall"), (hall1, back, sanct_h, "sanctuary_wall")):
            for sign in (-1, 1):
                self.run(role, "z", (z0 + z1) / 2, sign * (half - t / 2), z1 - z0, 0, top, t,
                         warp=batter(0, sign, .5, top), register="outer facing")
                self.run(role, "z", (z0 + z1) / 2, sign * (clear + t / 2), z1 - z0, 0, top, t, register="inner facing")
                self.core(*sorted((sign * (clear + t), sign * (half - t))), 0, top, z0, z1)
                self.slab("wall_coping", *sorted((sign * (clear), sign * (half + .15))), z0, z1, top, top + .3, 1.6)
        self.run("sanctuary_wall", "x", 0, back - t / 2, w, 0, sanct_h, t, warp=batter(2, 1, .5, sanct_h))
        self.slab("wall_coping", -half - .15, half + .15, back - wall_t, back + .15, sanct_h, sanct_h + .3, 1.6)

        radius = clamp(nave_h / 11 * self.vary("column", .1), .45, 1.6)
        # Court: a shaded walk down each side.
        court_r = radius * .8
        walk = clamp(.16 * w * self.vary("walk", .12), 3.0, 5.0)
        self.colonnade(clear - walk, inner + 1.5, hall0 - 1.5, low - 1.4, court_r, "papyrus-bud", floors[0])
        for sign in (-1, 1):
            self.slab("court_roof", *sorted((sign * (clear - walk - court_r), sign * clear)), inner, hall0,
                      low - .5, low, 2.2, construction="monolithic roof slabs")
        self.slab("floor", -clear, clear, inner, hall0, 0, floors[0], 2.0)

        # Hypostyle hall: tall nave, lower aisles, clerestory between.
        door = Voids(0)
        door.rect(0, 2 * gate * .8, 0, .62 * low, "hall.door")
        self.run("hall_wall", "x", 0, hall0 + t / 2, 2 * clear, 0, low, t, door)
        self.lintel(0, hall0 + t / 2, 2 * gate * .8, .62 * low, t)
        nave = clamp(.12 * w, 2.4, 4.0)
        pitch = clamp(4.6 * radius, 3.0, 5.5)
        rows = max(1, int((clear - nave - 1.2) / pitch))
        aisle_x = [nave + (k + 1) * (clear - nave) / (rows + 1) for k in range(rows)]
        z0, z1 = hall0 + t + 1.6, hall1 - 1.6
        self.colonnade(nave, z0, z1, nave_h, radius, self.variant, floors[1], beam_to=(hall0 + t, hall1))
        for x in aisle_x:
            self.colonnade(x, z0, z1, aisle_h, radius * .78, "papyrus-bud", floors[1], beam_to=(hall0 + t, hall1))
        first = aisle_x[0]
        for sign in (-1, 1):
            lights = Voids(aisle_h + .9)
            count = max(2, int((hall1 - hall0) / pitch))
            for k in range(count):
                lights.rect(-(hall1 - hall0 - t) / 2 + (k + .5) * (hall1 - hall0 - t) / count, 1.0,
                            aisle_h + .9 + (nave_h - aisle_h) * .2, (nave_h - aisle_h) * .55, f"grille.{k}")
            self.run("clerestory", "z", (hall0 + t + hall1) / 2, sign * first, hall1 - hall0 - t, aisle_h + .9, nave_h + .9, t,
                     lights, construction_note="stone grilles light the nave")
        for z in (hall0 + t / 2, hall1 - t / 2):
            self.run("clerestory", "x", 0, z, 2 * first + t, low, nave_h + .9, t, register="nave end wall")
        for x0, x1 in ((-first, -nave), (-nave, nave), (nave, first)):
            self.slab("nave_roof", x0 - .3, x1 + .3, hall0, hall1, nave_h + .9, high, 2.2, construction="monolithic roof slabs")
        edges = [first] + aisle_x[1:] + [clear]
        for sign in (-1, 1):
            for x0, x1 in zip(edges, edges[1:]):
                self.slab("aisle_roof", *sorted((sign * (x0 - .2), sign * (x1 + .2))), hall0, hall1, aisle_h + .9, low,
                          2.2, construction="monolithic roof slabs")
        self.slab("floor", -clear, clear, hall0, hall1, 0, floors[1], 2.0)
        self.ramp(hall0, floors[0], floors[1], gate)

        # Sanctuary: parallel chambers, the god's on the axis.
        chambers = max(3, int(math.ceil(2 * clear / 5.5)) | 1)
        width = 2 * clear / chambers
        doors = Voids(0)
        for k in range(chambers):
            doors.rect(-clear + (k + .5) * width, 2.0 if k == chambers // 2 else 1.2, 0, .55 * sanct_h, f"chamber.{k}")
        self.run("hall_wall", "x", 0, hall1 + t / 2, 2 * clear, 0, low, t, doors)
        for k in range(chambers):
            self.lintel(-clear + (k + .5) * width, hall1 + t / 2, 2.0 if k == chambers // 2 else 1.2, .55 * sanct_h, t)
        for k in range(1, chambers):
            self.run("sanctuary_wall", "z", (hall1 + t + back - wall_t) / 2, -clear + k * width, back - wall_t - hall1 - t,
                     0, sanct_h - .5, t, register="chamber partition")
        for k in range(chambers):
            self.slab("sanctuary_roof", -clear + k * width - .3, -clear + (k + 1) * width + .3, hall1 + t, back - wall_t + .2,
                      sanct_h - .5, sanct_h, 2.2, construction="monolithic roof slabs")
        self.slab("floor", -clear, clear, hall1, back - wall_t, 0, floors[2], 2.0)
        self.ramp(hall1, floors[1], floors[2], 1.2)
        zc = (hall1 + back) / 2
        self.member("shrine", -.9, .9, floors[2], floors[2] + 2.4, zc - 1.1, zc + 1.1, construction="gilded cedar naos")
        return self.document({"typology": "axial cult temple", "pylon_height": h, "pylon_batter": "about 1 in 9",
                              "nave_column_height": round(nave_h, 3), "aisle_column_height": round(aisle_h, 3),
                              "column_slenderness": round(nave_h / (2 * radius), 2), "aisle_rows": rows,
                              "sanctuary_chambers": chambers, "floor_levels": list(floors),
                              "roof_levels": [round(high, 3), round(low, 3), round(sanct_h, 3)],
                              "structure": "trabeated; monolithic architraves, lintels and roof slabs"})

    # ------------------------------------------------------------- parts
    def pylon(self, front, depth, gate, top, half):
        t = self.T
        lean = min(top / 9, depth * .3)
        for sign in (-1, 1):
            x0, x1 = sorted((sign * gate, sign * half))
            cx, cz, width = (x0 + x1) / 2, front + depth / 2, x1 - x0
            sx, sz = (width - 2 * lean) / width, (depth - 2 * lean) / depth
            body = top - 1.5
            self.shell("pylon", x0, x1, front, front + depth, 0, body, t, warp=taper(cx, cz, 0, body, sx, sz))
            self.core(x0 + t, x1 - t, 0, body, front + t, front + depth - t, warp=taper(cx, cz, 0, body, sx, sz))
            hx, hz = width / 2 * sx, depth / 2 * sz
            # Torus roll, then a cavetto that flares in two lifts.
            self.shell("pylon_cornice", cx - hx - .12, cx + hx + .12, cz - hz - .12, cz + hz + .12, body, body + .3,
                       t, register="torus")
            for y0, y1, a, b in ((body + .3, body + .9, 0.0, .3), (body + .9, top, .3, .75)):
                self.shell("pylon_cornice", cx - hx - a, cx + hx + a, cz - hz - a, cz + hz + a, y0, y1, t,
                           warp=taper(cx, cz, y0, y1, (hx + b) / (hx + a), (hz + b) / (hz + a)), register="cavetto", grade="presentation")
            self.slab("pylon_roof", cx - hx - .3, cx + hx + .3, cz - hz - .3, cz + hz + .3, top - .3, top, 1.6)
        clear, height = gate * .8, .5 * top
        for sign in (-1, 1):
            x0, x1 = sorted((sign * clear, sign * (gate + .6)))
            self.shell("gate_jamb", x0, x1, front + depth * .12, front + depth * .88, 0, height, min(t, (x1 - x0) / 2 - .05))
        count = max(1, int(math.ceil(depth * .76 / 1.6)))
        for k in range(count):
            z0 = front + depth * .12 + depth * .76 * k / count
            self.member("gate_lintel", -gate - .6, gate + .6, height, height + .9, z0, z0 + depth * .76 / count,
                        construction="monolithic lintel")
        y0 = height + .9
        self.shell("pylon_cornice", -gate - .6, gate + .6, front + depth * .12, front + depth * .88, y0, y0 + .8, t,
                   warp=taper(0, front + depth / 2, y0, y0 + .8, (gate + 1.1) / (gate + .6), 1.12), register="gate cavetto", grade="presentation")
        self.slab("pylon_roof", -gate - 1.1, gate + 1.1, front + depth * .08, front + depth * .92, y0 + .8, y0 + 1.05, 1.6)

    def forecourt(self, front, gate, top, half):
        tower = half - gate
        for sign in (-1, 1):
            for share in (.3, .72):
                x = sign * (gate + tower * share)
                self.drum("flag_mast", x, front - .55, 0, top + 2.0, .28, .12, sides=8, construction="cedar flagstaff")
                self.member("pennant", x, x + sign * 1.4, top + 1.2, top + 1.9, front - .57, front - .53,
                            construction="linen pennant")
            # Obelisk: one stone, tapering shaft and a pyramidion.
            x, z, tall = sign * (gate + 2.4), front - 2.6, .6 * top
            base = tall / 10
            self.member("obelisk", x - base * .75, x + base * .75, 0, .6, z - base * .75, z + base * .75,
                        construction="granite pedestal")
            shaft = tall - .6 - base * 1.2
            points = [(x + sx * r, y, z + sz * r) for y, r in ((.6, base / 2), (.6 + shaft, base * .36))
                      for sx in (-1, 1) for sz in (-1, 1)]
            self.mesh("obelisk", hull(points), construction="monolithic granite shaft")
            points = [(x + sx * base * .36, .6 + shaft, z + sz * base * .36) for sx in (-1, 1) for sz in (-1, 1)]
            self.mesh("obelisk", hull(points + [(x, tall, z)]), construction="gilded pyramidion")

    def column(self, x, z, floor, height, radius, order):
        r, h = radius, height - floor
        if order == "open-papyrus":
            profile = [(0, 1.3 * r), (.3, 1.3 * r), (.3, r), (.8 * h, .84 * r), (.84 * h, .9 * r),
                       (.95 * h, 1.75 * r), (.97 * h, 1.75 * r)]
            block = .62 * r
        else:
            profile = [(0, 1.3 * r), (.3, 1.3 * r), (.3, .88 * r), (.12 * h, r), (.76 * h, .8 * r),
                       (.8 * h, .98 * r), (.86 * h, r), (.97 * h, .7 * r)]
            block = .8 * r
        profile = [(floor + y, radius_) for y, radius_ in profile]
        steps = [(a, b) for a, b in zip(profile, profile[1:]) if b[0] - a[0] > 1e-9]
        for a, b in steps:
            self.lathe("column_drum", x, z, [a, b], order=order)
        self.member("abacus", x - block, x + block, floor + .97 * h, height, z - block, z + block, order=order)

    def colonnade(self, x, z0, z1, height, radius, order, floor, beam_to=None):
        """A mirrored pair of column rows along Z carrying architraves."""
        pitch = clamp(4.6 * radius, 2.6, 5.5)
        count = max(2, int(round((z1 - z0) / pitch)) + 1)
        zs = [z0 + (z1 - z0) * k / (count - 1) for k in range(count)]
        ends = beam_to or (z0 - radius, z1 + radius)
        for sign in (-1, 1):
            for z in zs:
                self.column(sign * x, z, floor, height, radius, order)
            marks = [ends[0]] + [(a + b) / 2 for a, b in zip(zs, zs[1:])] + [ends[1]]
            marks = [ends[0]] + zs[1:-1] + [ends[1]] if count > 2 else list(ends)
            for a, b in zip(marks, marks[1:]):
                self.member("architrave", sign * x - radius * .8, sign * x + radius * .8, height, height + .9, a, b,
                            construction="monolithic architrave")

    def lintel(self, x, z, width, height, thickness):
        self.member("door_lintel", x - width / 2 - .5, x + width / 2 + .5, height, height + .6,
                    z - thickness / 2 - .08, z + thickness / 2 + .08, construction="monolithic lintel")

    def ramp(self, z, low, high, half):
        """A shallow ramp on the axis lifts the procession to the next floor."""
        run = 2.4

        def slope(vertices):
            out = vertices.copy()
            lifted = vertices[:, 1] > low + 1e-9
            out[lifted, 1] = low + .02 + (high - low - .02) * (vertices[lifted, 2] - (z - run)) / run
            return out
        self.slab("ramp", -half, half, z - run, z, 0, low + .1, 1.2, warp=slope)


Temple.module = sys.modules[__name__]


def build_document(request):
    return Temple(request).compose()


def random_controls(product, rng):
    width = rng.randint(26, 36)
    return {"width": width, "height": round(width * rng.uniform(.72, .9)), "depth": round(width * rng.uniform(2.2, 2.7)),
            "bays": None, "levels": None, "detail": rng.choice(("concept", "standard"))}


def rule_checks(document, form):
    placements = document.placements
    by_role = {}
    for p in placements:
        by_role.setdefault(p.role, []).append(p)

    def top(role):
        return max((p.position[1] + p.dimensions[1] / 2 for p in by_role.get(role, [])), default=None)
    checks = []
    pylon = by_role.get("pylon", [])
    if pylon:
        low, high = min(p.position[1] for p in pylon), max(p.position[1] for p in pylon)
        foot, head = ([p for p in pylon if abs(p.position[1] - y) < 1e-6] for y in (low, high))
        spread = extent(foot)[1][2] - extent(foot)[0][2] > extent(head)[1][2] - extent(head)[0][2] + .2
        checks.append(("battered_pylon", spread, "The pylon towers lean inward as they rise.",
                       "The pylon has lost its batter.", []))
    levels = [top(role) for role in ("pylon", "nave_roof", "aisle_roof", "sanctuary_roof")]
    falling = all(v is not None for v in levels) and all(a > b + .05 for a, b in zip(levels, levels[1:]))
    checks.append(("falling_roofs", falling, "Roofs step down from pylon to sanctuary.",
                   "The roofs no longer fall toward the sanctuary.", []))
    arches = [p.id for p in placements if p.kind == "arch"]
    checks.append(("trabeated", not arches and bool(by_role.get("architrave")), "Every span is a beam on posts.",
                   "The temple contains an arch or has lost its architraves.", arches))
    missing = unmirrored(placements, {"column_drum", "abacus", "obelisk"})
    checks.append(("processional_axis", not missing, "Columns and obelisks mirror across the axis.",
                   "A column or obelisk has no twin across the axis.", missing))
    drums, beams = by_role.get("column_drum", []), by_role.get("architrave", [])
    if drums and beams:
        bare = []
        for p in by_role.get("abacus", []):
            y = p.position[1] + p.dimensions[1] / 2
            if not any(abs(b.position[1] - b.dimensions[1] / 2 - y) < 1e-3
                       and abs(b.position[0] - p.position[0]) < b.dimensions[0] / 2
                       and abs(b.position[2] - p.position[2]) <= b.dimensions[2] / 2 + 1e-6 for b in beams):
                bare.append(p.id)
        checks.append(("columns_carry_beams", not bare, "Every abacus bears an architrave.",
                       "A column carries nothing.", bare))
    return checks
