"""Muscovite architect: the Old Russian masonry church.

The church is a tall white cube, the chetverik, on a low podium.
Its rules are strict. Each wall bay ends in a gable instead of a level eave.
The altar is in one or three half-round apses on the east, and they stay below
the eaves. Nothing but drums stands on the vaults, every drum carries a dome,
and every dome carries a cross. The west door is reached through a porch.

``five-domed`` follows the cathedrals of the Moscow Kremlin: a great drum over
the crossing and four lesser drums on the diagonals. ``tent-roof`` follows the
votive churches of the sixteenth century, such as the Ascension at Kolomenskoye:
an octagon on the cube, a ring of kokoshniki, and a steep eight-sided tent that
closes to one small drum.

The walls are modelled in white limestone, as in the Dormition Cathedral and the
older white-stone churches. Brick under whitewash is not modelled as a finish.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np

from .historic import Voids, clamp, extent, hull, oriented_profile, rotation
from .schema import ArchitectInfo
from .sonnet import Poem, controls_for, form

ARCHITECT = ArchitectInfo("muscovite", "1.0.0", "Muscovite Church Architect")
STYLE = "muscovite-church"
PRODUCT = "muscovite-church"
REGION = "Muscovy (Moscow, Vladimir and the Russian north)"
PERIOD = "Late 15th to 17th centuries"
VARIANTS = ("five-domed", "tent-roof")
KINDS = {
    "core_fill": "core", "podium_wall": "wall", "floor_paving": "slab", "porch_step": "circulation",
    "porch_pier": "pier", "porch_beam": "beam", "porch_roof": "roof",
    "body_wall": "wall", "window_arch": "arch", "portal_arch": "arch", "gable": "gable",
    "interior_pier": "pier", "vault_deck": "slab", "hip_roof": "roof",
    "apse_wall": "wall", "apse_roof": "roof",
    "octagon": "tower", "tent": "roof", "drum": "wall", "dome": "roof", "cross": "ornament",
}
REQUIRED = {"core_fill", "podium_wall", "body_wall", "gable", "vault_deck", "apse_wall", "apse_roof",
            "drum", "dome", "cross", "porch_roof", "portal_arch"}
FORMS = {PRODUCT: form(
    PRODUCT, "White-stone church with gabled wall bays, eastern apses and onion domes on drums, or a tent roof",
    (16, 26, 24), VARIANTS, REQUIRED,
    lines={"domes": ("onion", "helmet"), "gables": ("zakomary", "kokoshniki"),
           "apses": ("three", "one"), "porch": ("gabled", "tent")},
    size={"width": (12, 22), "height": (18, 40), "depth": (17, 36)},
    ratios={"depth/width": (1.35, 1.65), "height/width": (1.45, 1.9)},
    controls={"bays": (3, 2, 4)},
    region=REGION, period=PERIOD, bays_meaning="gabled wall bays on each face of the cube",
    orientation="the porch is west (-Z) and the apses east (+Z)",
    openings="round arches over the slit windows and the west portal")}
DEFAULT_SUBSTANCE = "limestone"
SUBSTANCES = {"core_fill": "earth-fill", "porch_beam": "oak", "porch_roof": "softwood",
              "interior_pier": "fired-clay-brick", "vault_deck": "fired-clay-brick",
              "hip_roof": "copper-sheet", "apse_roof": "copper-sheet", "dome": "copper-sheet", "cross": "wrought-iron"}
TOL = .05
T = .8              # wall of the cube
LEDGE = .35         # the podium shows this much beyond the wall
STEPS = 1.2         # run of the porch steps
DECK = .3
SHEET = .1          # dome covering


def outline(shape, half, rise, count=4):
    """Half-outline of a wall gable from its foot to its crown: a round zakomara or a keeled kokoshnik."""
    points = []
    for k in range(count + 1):
        s = k / count
        x = half * (math.cos(math.pi / 2 * s) if shape == "zakomary" else (1 - s) * (1 + 1.2 * s))
        y = rise * (math.sin(math.pi / 2 * s) if shape == "zakomary" else s)
        points.append((x, y))
    return points


def bulb_profile(shape, radius, base, height, steps=10):
    """(height, radius) stations of a Russian dome from its springing to its tip.

    ``onion`` swells to a third again of its springing radius and draws in to a
    point through a hollow curve. ``helmet`` is the older form: it does not swell.
    """
    points = []
    for k in range(steps + 1):
        u = k / steps
        if shape == "helmet":
            r = (1 - u) ** .55 * (1 + .5 * u)
        elif u <= .35:
            r = 1 + .36 * math.sin(math.pi / 2 * u / .35)
        else:
            v = (u - .35) / .65
            r = 1.36 * (1 - v) ** 1.4 * (1 + 1.4 * v)
        points.append((base + u * height, max(radius * r, .12)))
    return points


class Church(Poem):
    PALETTE = {"core": (140, 128, 112, 255), "podium_wall": (206, 200, 186, 255), "floor_paving": (190, 184, 170, 255),
               "porch_step": (198, 192, 178, 255), "pier": (236, 232, 222, 255), "beam": (104, 72, 48, 255),
               "porch_roof": (120, 92, 66, 255), "wall": (240, 237, 228, 255), "arch": (226, 220, 208, 255),
               "gable": (244, 241, 233, 255), "vault_deck": (214, 208, 196, 255), "hip_roof": (88, 132, 112, 255),
               "apse_roof": (88, 132, 112, 255), "octagon": (240, 237, 228, 255), "tent": (232, 226, 214, 255),
               "dome": (214, 170, 62, 255), "cross": (226, 190, 84, 255)}
    TIMBER_KINDS = frozenset({"beam"})

    # ------------------------------------------------------------ plan
    def compose(self):
        w, h, d = self.w, self.h, self.d
        self.base = clamp(.03 * h, .5, .9)
        self.sx = w - 2 * LEDGE
        three = self.line("apses") == "three"
        self.apse_r = (.19 if three else .26) * self.sx * self.vary("apse", .05)
        porch = clamp(.14 * d, 2.6, 4.0)
        self.zp = -d / 2 + STEPS                    # podium front
        self.zw = self.zp + porch                   # west face of the cube
        self.ze = d / 2 - self.apse_r               # east face; the main apse fills the rest of the plot
        self.sz = self.ze - self.zw
        self.cz = (self.zw + self.ze) / 2
        side = min(self.sx, self.sz)
        self.cross_h = clamp(.06 * h, 1.3, 2.0)
        self.swell = 2.6 if self.line("domes") == "onion" else 1.6     # dome height over springing radius
        self.gable_rise = .5 * min(self.sx, self.sz - 2 * T) / self.bays
        tent = self.variant == "tent-roof"
        if tent:
            self.eaves = max(.36 * h, self.base + 6.0)
        else:
            self.rc = .17 * side * self.vary("drum", .05)
            self.dome_h = self.swell * (self.rc + .1)
            drum = min(clamp(.15 * h, 2.2, 5.5) * self.vary("drum-height", .1),
                       h - self.cross_h - self.dome_h - DECK - self.base - 6.5)
            if drum < 1.6:
                raise ValueError(f"{PRODUCT} is too low for a drum and dome over walls this wide")
            self.drum_h = drum
            self.eaves = h - self.cross_h - self.dome_h - drum - DECK
        self.deck = self.eaves + DECK
        self.podium("podium_wall", -w / 2, w / 2, self.zp, self.ze, 0, self.base, paving="floor_paving")
        self.porch(porch)
        self.walls()
        self.gables()
        self.apses(three)
        pier = self.rc if not tent else .2 * side
        for sx in (-1, 1):
            for sz in (-1, 1):
                self.stack("interior_pier", sx * pier, self.base, self.cz + sz * pier, .9, .9, self.eaves - self.base)
        self.slab("vault_deck", -self.sx / 2 + T, self.sx / 2 - T, self.zw + T, self.ze - T,
                  self.eaves, self.deck, module=1.6)
        rules = self.tent_crown(side) if tent else self.five_domes(side)
        return self.document({"region": REGION, "period": PERIOD, "wall_bays": self.bays,
                              "apses": 3 if three else 1, "eaves_height": round(self.eaves, 3), **rules,
                              "construction": "white limestone walls on a faced podium; brick vaults; sheet-covered timber domes"})

    # ------------------------------------------------------------ cube
    def walls(self):
        base, eaves, sx, sz = self.base, self.eaves, self.sx, self.sz
        tall = eaves - base
        sill = base + .6 * tall
        light = min(1.7 * self.vary("window", .1), eaves - sill - 1.3)
        for face, axis, length, along, cross in (("west", "x", sx, 0., self.zw + T / 2),
                                                 ("east", "x", sx, 0., self.ze - T / 2),
                                                 ("north", "z", sz - 2 * T, self.cz, -sx / 2 + T / 2),
                                                 ("south", "z", sz - 2 * T, self.cz, sx / 2 - T / 2)):
            voids = Voids(base)
            if face != "east":                      # the apses stand against the east wall
                for i in range(self.bays):
                    centre = (i + .5) * length / self.bays - length / 2
                    if face == "west" and abs(centre) < 1e-6:
                        continue                    # the porch roof stands in front of this bay
                    voids.arch(centre, .3, sill, sill + light, .3, f"{face}.window.{i}", role="window_arch")
            if face == "west":
                voids.arch(0, .9, base, base + 2.0, .36, "portal", role="portal_arch")
            self.run("body_wall", axis, along, cross, length, base, eaves, T, voids=voids, face=face)

    def gable(self, role, axis, along, cross, thickness, half, y0, rise, shape, turn=None, **params):
        """One wall gable, cut as a fan of pieces about the middle of its foot."""
        right = outline(shape, half, rise)
        points = [(x, y0 + y) for x, y in right] + [(-x, y0 + y) for x, y in right[-2::-1]]
        for a, b in zip(points, points[1:]):
            mesh = oriented_profile([(0., y0), a, b], thickness, axis, along, cross)
            if turn is not None:
                mesh.vertices = turn(np.asarray(mesh.vertices))
            self.mesh(role, mesh, construction="carved gable", shape=shape, **params)

    def gables(self):
        shape, rise = self.line("gables"), self.gable_rise
        for face, axis, length, along, cross, inward in (
                ("west", "x", self.sx, 0., self.zw + T / 2, 1), ("east", "x", self.sx, 0., self.ze - T / 2, -1),
                ("north", "z", self.sz - 2 * T, self.cz, -self.sx / 2 + T / 2, 1),
                ("south", "z", self.sz - 2 * T, self.cz, self.sx / 2 - T / 2, -1)):
            bay = length / self.bays
            for i in range(self.bays):
                self.gable("gable", axis, along + (i + .5) * bay - length / 2, cross, T,
                           bay / 2, self.eaves, rise, shape, face=face, tier=0)
            if shape == "kokoshniki":               # a second tier stands on the vault, over the piers between bays
                for i in range(1, self.bays):
                    self.gable("gable", axis, along + i * bay - length / 2, cross + inward * (T / 2 + .2), .4,
                               bay * .34, self.deck, rise * .8, shape, face=face, tier=1)

    def apses(self, three):
        base, r = self.base, self.apse_r
        top = base + .62 * (self.eaves - base)
        plan = [(0., r)] + ([(side * .33 * self.sx, .13 * self.sx) for side in (-1, 1)] if three else [])
        for i, (x, radius) in enumerate(plan):
            height = top if i == 0 else base + .54 * (self.eaves - base)
            sill = base + .45 * (height - base)
            self.ring("apse_wall", x, self.ze, radius, .6, 0, height, arc=(0., math.pi),
                      gaps=[(math.pi / 2, .26 / radius, sill, sill + 1.5)], apse=i)
            self.conic("apse_roof", x, self.ze, height, height + .55 * radius, radius, 0., sides=8,
                       arc=(0., math.pi), apse=i, construction="sheet over timber")

    def porch(self, depth):
        base, zp, zw = self.base, self.zp, self.zw
        width = clamp(.3 * self.w * self.vary("porch", .08), 3.4, 5.0)
        self.flight("porch_step", width, -self.d / 2, zp, 0, base)
        head, half = base + 3.5, width / 2
        for side in (-1, 1):
            self.stack("porch_pier", side * (half - .3), base, zp + .3, .6, .6, head - base)
            self.member("porch_beam", side * (half - .3) - .2, side * (half - .3) + .2, head, head + .25, zp + .6, zw)
        self.member("porch_beam", -half, half, head, head + .25, zp, zp + .6)
        eave = head + .25
        corners = [(x, eave, z) for x in (-half, half) for z in (zp, zw)]
        if self.line("porch") == "gabled":
            crown = [(0., eave + .5 * width, z) for z in (zp, zw)]
        else:
            crown = [(0., eave + .85 * width, (zp + zw) / 2)]
        self.mesh("porch_roof", hull(corners + crown), construction="boarded roof", shape=self.line("porch"))

    # ------------------------------------------------------------ crowns
    def bulb(self, cx, cz, radius, y, height, gores, crown):
        """A dome as a timber cradle under sheet: gores between the stations of its profile."""
        profile = bulb_profile(self.line("domes"), radius, y, height)
        for course, ((ya, ra), (yb, rb)) in enumerate(zip(profile, profile[1:])):
            for g in range(gores):
                a0, a1 = math.tau * g / gores, math.tau * (g + 1) / gores
                points = []
                for yy, r in ((ya, ra), (yb, rb)):
                    inner = r - SHEET if r > .6 else 0.
                    for rr in (r, inner):
                        points += [(cx + rr * math.cos(a), yy, cz + rr * math.sin(a)) for a in (a0, a1)]
                self.mesh("dome", hull(points), construction="sheet over timber cradle", course=course, crown=crown)
        return y + height

    def cross(self, cx, cz, y, crown):
        top = y + self.cross_h
        arm = .28 * self.cross_h
        self.member("cross", cx - .07, cx + .07, y, top, cz - .07, cz + .07, crown=crown, part="mast")
        for height, reach in ((.72, arm), (.5, arm * .55)):
            for side in (-1, 1):
                x0, x1 = sorted((cx + side * .07, cx + side * reach))
                self.member("cross", x0, x1, y + height * self.cross_h - .06, y + height * self.cross_h + .06,
                            cz - .07, cz + .07, crown=crown, part="bar")

    def crown(self, index, cx, cz, radius, y, drum, gores):
        if radius > 1.4:
            lights = 8 if index == 0 else 4
            gaps = [(math.tau * (k + .5) / lights, .2 / radius, y + .25 * drum, y + .8 * drum) for k in range(lights)]
            self.ring("drum", cx, cz, radius, .45, y, y + drum, gaps=gaps, crown=index)
        else:
            self.lathe("drum", cx, cz, [(y, radius), (y + drum, radius)], sides=gores, crown=index)
        swell = self.swell * (radius + .1)
        self.cross(cx, cz, self.bulb(cx, cz, radius + .1, y + drum, swell, gores, index), index)
        return y + drum + swell + self.cross_h

    def five_domes(self, side):
        deck, rc = self.deck, self.rc
        # A low hipped roof sheds the vaults between the drums.
        hx, hz, rise = self.sx / 2 - T, self.sz / 2 - T, .5 * self.gable_rise
        low = [(sx * hx, deck, self.cz + sz * hz) for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        high = [(x * .3, deck + rise, self.cz + (z - self.cz) * .3) for x, _, z in low]
        for i in range(4):
            j = (i + 1) % 4
            self.mesh("hip_roof", hull([low[i], low[j], high[i], high[j], (0, deck, self.cz), (0, deck + rise, self.cz)]),
                      construction="sheet over timber")
        self.crown(0, 0., self.cz, rc, deck, self.drum_h, 16)
        lesser, reach = .58 * rc, .31 * side
        for index, (sx, sz) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1)), 1):
            self.crown(index, sx * reach, self.cz + sz * reach, lesser, deck, .62 * self.drum_h, 12)
        return {"domes": 5, "central_drum_radius": round(rc, 3), "drum_height": round(self.drum_h, 3)}

    def tent_crown(self, side):
        deck, h = self.deck, self.h
        radius = .36 * side * self.vary("octagon", .04)
        octagon = clamp(.12 * h, 2.4, 4.5)
        neck_r, neck = clamp(.07 * side, .8, 1.3), 1.2
        tent = h - self.cross_h - self.swell * (neck_r + .1) - neck - octagon - deck
        if tent < .9 * radius:
            raise ValueError(f"{PRODUCT} is too low for a tent roof over walls this wide")
        rot = math.pi / 8                           # faces of the octagon square to the walls
        tiers = max(1, math.ceil(octagon / 1.6))
        for k in range(tiers):
            self.conic("octagon", 0., self.cz, deck + octagon * k / tiers, deck + octagon * (k + 1) / tiers,
                       radius, radius, sides=8, rot=rot, tier=k, construction="octagon")
        # A kokoshnik against each face of the octagon hides the foot of the tent.
        apothem, face = radius * math.cos(math.pi / 8), radius * math.sin(math.pi / 8)
        for k in range(8):
            self.gable("gable", "x", 0., -apothem - .175, .35, face * .92, deck, octagon * .85, "kokoshniki",
                       turn=rotation(math.pi / 4 * k, 0., self.cz), face="octagon", tier=2, side=k)
        foot = deck + octagon
        tiers = max(2, math.ceil(tent / 2.4))
        for k in range(tiers):
            ra, rb = (radius + (neck_r - radius) * u / tiers for u in (k, k + 1))
            self.conic("tent", 0., self.cz, foot + tent * k / tiers, foot + tent * (k + 1) / tiers, ra, rb,
                       sides=8, rot=rot, tier=k, construction="masonry tent")
        self.crown(0, 0., self.cz, neck_r, foot + tent, neck, 12)
        return {"domes": 1, "tent_height": round(tent, 3), "octagon_radius": round(radius, 3)}


Church.module = sys.modules[__name__]


def build_document(request):
    return Church(request).compose()


random_controls = controls_for(sys.modules[__name__])


# ---------------------------------------------------------------- rules
def _crowns(placements, role):
    groups = defaultdict(list)
    for p in placements:
        if p.role == role and isinstance(p.parameters.get("crown"), int):
            groups[p.parameters["crown"]].append(p)
    return groups


def _centre(lo, hi):
    return (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2


def rule_checks(document, form):
    placements = document.placements
    drums, domes, crosses = (_crowns(placements, role) for role in ("drum", "dome", "cross"))
    drum_box = {k: extent(v) for k, v in drums.items()}
    dome_box = {k: extent(v) for k, v in domes.items()}
    seats = [p for p in placements if p.role in ("vault_deck", "tent")]
    bad = []
    for k in sorted(set(drums) | set(domes)):
        if k not in drums or k not in domes:
            bad += [p.id for p in drums.get(k, []) + domes.get(k, [])]
            continue
        (lo, hi), (dlo, dhi) = drum_box[k], dome_box[k]
        cx, cz = _centre(lo, hi)
        springing = [p for p in domes[k] if p.position[1] - p.dimensions[1] / 2 < dlo[1] + TOL]
        slo, shi = extent(springing)
        carried = abs(dlo[1] - hi[1]) < TOL and math.dist(_centre(dlo, dhi), (cx, cz)) < TOL and \
            all(shi[a] - slo[a] > hi[a] - lo[a] - TOL for a in (0, 2))
        seated = any(abs(p.position[1] + p.dimensions[1] / 2 - lo[1]) < TOL
                     and abs(p.position[0] - cx) <= p.dimensions[0] / 2 + TOL
                     and abs(p.position[2] - cz) <= p.dimensions[2] / 2 + TOL for p in seats)
        if not (carried and seated):
            bad += [p.id for p in drums[k]]
    yield ("domes_on_drums", bool(drums) and not bad,
           "Every dome springs from its own drum, and every drum stands on the vault or the tent.",
           "A dome has no drum beneath it, or a drum does not stand on the vault or the tent.", bad)

    tents = defaultdict(list)
    for p in placements:
        if p.role == "tent":
            tents[p.parameters.get("tier")].append(p)
    ordered = False
    if tents:
        tiers = [extent(tents[k]) for k in sorted(tents)]
        widths = [hi[0] - lo[0] for lo, hi in tiers]
        ordered = set(drums) == {0} and all(b <= a + TOL for a, b in zip(widths, widths[1:])) and \
            abs(tiers[-1][1][1] - drum_box[0][0][1]) < TOL and widths[0] > 2.5 * (drum_box[0][1][0] - drum_box[0][0][0])
    elif set(drums) == set(domes) == set(crosses) == {0, 1, 2, 3, 4}:
        cx, cz = _centre(*drum_box[0])
        tops = {k: extent(crosses[k])[1][1] for k in crosses}
        sizes = [dome_box[k][1] - dome_box[k][0] for k in range(1, 5)]
        spots = sorted((round(_centre(*drum_box[k])[0] - cx, 2), round(_centre(*drum_box[k])[1] - cz, 2))
                       for k in range(1, 5))
        reach = abs(spots[0][0])
        ordered = all(tops[0] > tops[k] + 1.0 for k in range(1, 5)) and \
            all(drum_box[0][1][0] - drum_box[0][0][0] > drum_box[k][1][0] - drum_box[k][0][0] + .2 for k in range(1, 5)) and \
            all(np.allclose(size, sizes[0], atol=TOL) for size in sizes) and reach > 1.0 and \
            spots == sorted((sx * reach, sz * reach) for sx in (-1, 1) for sz in (-1, 1))
    yield ("crown_hierarchy", ordered,
           "The central crown is the highest; four equal lesser domes stand on the diagonals, or the tent narrows to one drum.",
           "The crowns are not ordered about the centre: a lesser dome is missing, unequal or too high, or the tent does not narrow.",
           [])

    walls = [p for p in placements if p.role == "body_wall"]
    apses = defaultdict(list)
    for p in placements:
        if p.role == "apse_wall":
            apses[p.parameters.get("apse")].append(p)
    east, low = [], bool(walls) and len(apses) in (1, 3)
    if low:
        wlo, whi = extent(walls)
        for k, members in apses.items():
            lo, hi = extent(members)
            if not (lo[2] > whi[2] - TOL and lo[2] < whi[2] + TOL and hi[1] < whi[1] - .5
                    and lo[0] > wlo[0] - TOL and hi[0] < whi[0] + TOL):
                east += [p.id for p in members]
        centred = [abs(sum(_centre(*extent(m))[:1])) for m in apses.values()]
        low = min(centred) < TOL
    yield ("eastern_apses", low and not east, "One or three apses project from the east wall and stay below the eaves.",
           "An apse is missing, is not against the east wall, or rises to the eaves.", east)

    bare = []
    for k, members in domes.items():
        (dlo, dhi) = dome_box[k]
        marks = crosses.get(k, [])
        masts = [p for p in marks if p.parameters.get("part") == "mast"] or marks
        if not marks or abs(extent(marks)[0][1] - dhi[1]) > TOL or \
                math.dist(_centre(*extent(masts)), _centre(dlo, dhi)) > TOL or \
                extent(marks)[1][0] - extent(marks)[0][0] < .4:
            bare += [p.id for p in members[:1]]
    yield ("crosses", bool(domes) and not bare, "Every dome carries a cross on its axis.",
           "A dome has no cross, or its cross is off the axis of the dome.", bare)
