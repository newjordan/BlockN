"""Mughal architect: the garden tomb.

A Mughal tomb is one figure turned four times. A broad square plinth carries a
square building whose corners are cut off, so that eight faces look out. Each
main face is the same: a tall framed portal, the pishtaq, rises above the wall
head and holds a deep pointed arch, the iwan; smaller pointed niches stand in
storeys beside it and on the cut corners. Over the middle a drum lifts the dome,
kiosks called chhatris stand on the roof at the four diagonals, and a finial
closes the axis.

``taj`` follows the tomb of Mumtaz Mahal at Agra: white marble, a high drum, a
swelling dome and four free-standing minarets at the plinth corners. ``humayun``
follows the earlier tomb of Humayun at Delhi: red sandstone trimmed with white
marble, a taller plinth, a broad low dome and no minarets.

One face, one cut corner, one chhatri and one minaret are composed, then turned
onto the other three sides. The symmetry is built, not asserted.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np

from .coursed import clean, clip
from .historic import TAU, box_mesh, clamp, extent, hull, oriented_profile, rotation
from .schema import ArchitectInfo
from .sonnet import Poem, dome_profile, form

ARCHITECT = ArchitectInfo("mughal", "1.0.0", "Mughal Garden Tomb Architect")
STYLE = "mughal-garden-tomb"
PRODUCT = "mughal-tomb"
REGION = "North India (Agra and Delhi)"
PERIOD = "Mughal Empire, about 1560–1650"
VARIANTS = ("taj", "humayun")
KINDS = {
    "core_fill": "core", "plinth_wall": "wall", "plinth_paving": "slab", "plinth_stair": "circulation",
    "tomb_wall": "wall", "chamfer_wall": "wall", "niche_arch": "arch", "niche_back": "ornament",
    "pishtaq": "wall", "iwan_arch": "arch", "iwan_back": "ornament", "door_lintel": "entablature",
    "guldasta": "ornament", "chamber_pier": "pier", "roof_deck": "roof", "drum": "wall", "dome": "roof",
    "lotus": "ornament", "finial": "ornament",
    "chhatri_base": "slab", "chhatri_column": "column", "chhatri_eave": "ornament", "chhatri_dome": "roof",
    "minaret_shaft": "column", "minaret_balcony": "ornament", "minaret_base": "slab",
    "minaret_column": "column", "minaret_eave": "ornament", "minaret_dome": "roof",
}
REQUIRED = {"core_fill", "plinth_wall", "plinth_stair", "tomb_wall", "chamfer_wall", "pishtaq", "iwan_arch",
            "niche_arch", "drum", "dome", "finial", "chhatri_column", "chhatri_dome"}
FORMS = {PRODUCT: form(
    PRODUCT, "Four-fold garden tomb on a square plinth: pishtaq portals, a drum and dome, and roof chhatris",
    (44, 34, 44), VARIANTS, REQUIRED,
    lines={"dome": ("onion", "bulbous", "pointed"), "niches": ("two-storey", "single"),
           "chhatris": ("octagonal", "square"), "finial": ("lotus-kalash", "spire")},
    size={"width": (36, 56), "height": (24, 47), "depth": (36, 56)},
    ratios={"depth/width": (1.0, 1.0), "height/width": (.66, .84)},
    region=REGION, period=PERIOD, symmetry="four-fold about the vertical axis",
    minarets_by_variant={"taj": 4, "humayun": 0},
    openings="pointed arches of cut voussoirs; the door in each iwan is lintelled")}
DEFAULT_SUBSTANCE = "marble"
SUBSTANCES = {"core_fill": "fired-clay-brick", "finial": "copper-sheet"}
# In the Delhi tomb these are red sandstone; the arches, niche backs, dome and kiosk domes stay marble.
RED = frozenset({"plinth_wall", "plinth_paving", "plinth_stair", "tomb_wall", "chamfer_wall", "pishtaq", "iwan_back",
                 "door_lintel", "guldasta", "chamber_pier", "roof_deck", "drum", "chhatri_base", "chhatri_column",
                 "chhatri_eave"})
WHITE, SHADE, ROSE, GOLD = (238, 234, 226, 255), (222, 217, 207, 255), (168, 86, 64, 255), (196, 158, 74, 255)
POINT = .2          # how far each arch centre sits past the axis, as a share of the half-span
WALL = .6
PROUD = .5          # the pishtaq stands this far in front of the wall face
DECK = .3
# Roles that are composed once and turned onto the other three sides.
TURNED = frozenset({"tomb_wall", "chamfer_wall", "pishtaq", "iwan_arch", "niche_arch", "guldasta",
                    "chhatri_column", "minaret_shaft"})
TOMB = ("tomb_wall", "chamfer_wall", "pishtaq")


def substances_for(form):
    return {"marble", "sandstone", "fired-clay-brick", "copper-sheet"}


def arch_shape(half, ring):
    """A two-centred pointed arch: offset, inner and outer radius, the two closing angles, outer rise."""
    e = POINT * half
    inner, outer = half + e, half + e + ring
    return e, inner, outer, math.acos(e / inner), math.acos(e / outer), math.sqrt(outer * outer - e * e)


def bulb(shape, radius, base, height, steps=12, fullness=1.):
    """(height, radius) stations of the great dome; it closes to a collar that takes the lotus.

    ``fullness`` scales the swell: a low dome that swelled like a tall one would overhang its drum.
    """
    swell, power = {"onion": (.15, 2.6), "bulbous": (.27, 3.0), "pointed": (0., 1.9)}[shape]
    swell *= fullness
    points = []
    for k in range(steps + 1):
        u = k / steps
        r = (1 + swell * math.sin(math.pi * min(1., u / .55)) ** 1.2) * (1 - .922 * u ** power) ** .8
        points.append((base + u * height, radius * r))
    return points


class Tomb(Poem):
    def compose(self):
        w, h = self.w, self.h
        taj = self.taj = self.variant == "taj"
        self.PALETTE = ({"wall": WHITE, "arch": SHADE, "roof": WHITE, "slab": SHADE, "column": WHITE} if taj else
                        {"wall": ROSE, "arch": WHITE, "roof": WHITE, "slab": (150, 80, 62, 255), "column": ROSE,
                         "roof_deck": (176, 96, 72, 255), "niche_back": WHITE, "entablature": WHITE})
        self.PALETTE.update({"core": (150, 104, 80, 255), "finial": GOLD, "ornament": WHITE,
                             "circulation": SHADE if taj else (156, 84, 64, 255)})
        self.T = T = .31 * w * self.vary("tomb", .03)                   # half-side of the tomb
        self.c = .24 * T * self.vary("chamfer", .08)                    # leg of the cut corner
        self.Wp = .34 * T * self.vary("pishtaq", .05)                   # half-width of the portal
        self.P = P = clamp(.07 * h, 2., 3.2) if taj else clamp(.11 * h, 3., 4.5)
        radius = (.36 if taj else .46) * T * self.vary("dome", .04)
        drum = (.13 if taj else .07) * h * self.vary("drum", .08)
        finial = clamp(.075 * h, 2.2, 3.2)
        low, high = (1.75, 2.2) if taj else (1.15, 1.5)                 # dome height over its base radius
        wall = (.37 if taj else .40) * h
        dome = clamp(h - P - wall - drum - finial, low * radius, high * radius)
        wall = h - P - drum - finial - dome                             # the wall takes up what the dome leaves
        self.ratio(wall / h, "wall height/height", .26, .54)
        self.head, self.crown = P + wall, P + wall * (1.2 if taj else 1.14)
        self.plinth()
        start = len(self.placements)
        self.face()
        self.corner()
        q = T - self.c / 2 - 2.2
        self.kiosk("chhatri", -q, -q, self.head, clamp(.11 * T, 1.2, 1.9), 1.)
        minaret = self.minaret() if taj else None
        self.quarter_turns(start)
        self.roof(radius)
        top = self.head + drum
        self.ring("drum", 0, 0, radius, .7, self.head, top)
        self.shell_dome(bulb(self.line("dome"), radius, top, dome, fullness=1. if taj else .4))
        self.finial(top + dome, .13 * radius)
        return self.document({"region": REGION, "period": PERIOD, "plan": "square with cut corners (hasht-bihisht)",
                              "tomb_side": round(2 * T, 3), "plinth_height": round(P, 3),
                              "wall_head": round(self.head, 3), "pishtaq_head": round(self.crown, 3),
                              "dome_radius": round(radius, 3), "dome_height": round(dome, 3),
                              "minarets": 4 if taj else 0, "minaret_height": minaret,
                              "construction": "cut-stone facing and voussoir arches; brick core in the plinth"})

    def quarter_turns(self, start):
        """Repeat everything placed since ``start`` on the other three sides, each copy its own wall."""
        count = len(self.placements) - start
        for quarter in (1, 2, 3):
            begin = self.clone(start, count)
            for p in self.placements[begin:]:
                if "host_id" in p.parameters:
                    p.parameters["host_id"] = f"{p.parameters['host_id']}.q{quarter}"
            self.turn(begin, quarter)

    def shell_dome(self, profile):
        """Lay the dome in rings of radial stones; where it closes, in solid drums.

        Where the profile swells or closes fast a stone leans a long way in one course,
        so those courses are cut shorter and lower to stay within a mason's stone.
        """
        for (ya, ra), (yb, rb) in zip(profile, profile[1:]):
            if min(ra, rb) < 2.4:
                self.lathe("dome", 0, 0, [(ya, ra), (yb, rb)], sides=20)
            else:
                steep = abs(rb - ra) > .45 * (yb - ya)
                self.ring("dome", 0, 0, ra, .5, ya, yb, top_radius=rb, grade="standard" if steep else "concept")

    def panels(self, role, along, cross, width, y0, y1, thickness, warp=None, skip=None, **params):
        """A thin revetment of slabs on X, each about a metre, breaking joint row by row."""
        rows = max(1, math.ceil((y1 - y0) / 1.25))
        for j in range(rows):
            ya, yb = y0 + (y1 - y0) * j / rows, y0 + (y1 - y0) * (j + 1) / rows
            count = max(1, math.ceil(width / 1.3))
            cuts = [-width / 2 + width * k / count for k in range(count + 1)]
            if j % 2 and count > 1:
                cuts = [cuts[0]] + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])] + [cuts[-1]]
            for xa, xb in zip(cuts, cuts[1:]):
                if skip and xa < skip[1] - 1e-6 and xb > skip[0] + 1e-6 and ya < skip[2] - 1e-6:
                    lo, hi = max(xa, skip[0]), min(xb, skip[1])
                    spans = [(xa, lo), (hi, xb)]
                else:
                    spans = [(xa, xb)]
                for a, b in spans:
                    if b - a < .08:
                        continue
                    centre, dims = (along + (a + b) / 2, (ya + yb) / 2, cross), (b - a, yb - ya, thickness)
                    if warp is None:
                        self.add(role, centre, dims, construction="stone revetment", **params)
                    else:
                        mesh = box_mesh(centre, dims)
                        mesh.vertices = warp(np.asarray(mesh.vertices))
                        self.mesh(role, mesh, construction="stone revetment", **params)

    def substance(self, role, kind):
        if not self.taj and role in RED:
            return "sandstone"
        return super().substance(role, kind)

    # ------------------------------------------------------------- plinth
    def plinth(self):
        """The square platform fills the plot; its stair is cut into the front so nothing leaves the square."""
        hw, hd, P, t = self.w / 2, self.d / 2, self.P, .5
        top, run = P - .2, P * 1.5
        sw = clamp(.13 * self.w, 4., 7.) / 2
        inner = -hd + run
        self.shell("plinth_wall", -hw, hw, -hd, hd, 0, top, t, faces="blr", grade="concept")
        for side in (-1, 1):
            a, b = sorted((side * sw, side * hw))
            self.ashlar("plinth_wall", ((a + b) / 2, top / 2, -hd + t / 2), (b - a, top, t), axis="x",
                        grade="concept", face="front")
            self.ashlar("plinth_wall", (side * (sw + .2), top / 2, (-hd + t + inner) / 2), (.4, top, inner + hd - t),
                        axis="z", grade="concept", face="stair cheek")
            a, b = sorted((side * (sw + .4), side * (hw - t)))
            self.fill(a, b, 0, top, -hd + t, inner, module=4.2)
            a, b = sorted((side * sw, side * hw))
            self.slab("plinth_paving", a, b, -hd, inner, top, P, module=3.2)
        self.fill(-hw + t, hw - t, 0, top, inner, hd - t, module=4.2)
        self.slab("plinth_paving", -hw, hw, inner, hd, top, P, module=3.2)
        self.flight("plinth_stair", 2 * sw, -hd, inner, 0, P)

    # --------------------------------------------------------------- arches
    def pointed_arch(self, role, wall_role, along, cross, thickness, half, spring, ring, warp=None, **params):
        """Voussoirs of a two-centred arch and the coursed spandrels that square it off."""
        e, inner, outer, close_in, close_out, top = arch_shape(half, ring)

        def at(radius, angle):
            return max(0., -e + radius * math.cos(angle)), radius * math.sin(angle)

        def built(outline, side):
            points = [(side * x, spring + y) for x, y in outline]
            mesh = oriented_profile(points[::side], thickness, "x", along, cross)
            if warp is not None:
                mesh.vertices = warp(np.asarray(mesh.vertices))
            return mesh
        count = max(3, math.ceil(inner * close_in / .5))
        host = self.host(role)
        for side in (1, -1):
            for i in range(count):
                a0, a1 = close_in * i / count, close_in * (i + 1) / count
                o1 = close_out if i == count - 1 else a1                 # the last stone closes on the axis
                outline = [at(outer, a0), at(outer, (a0 + o1) / 2), at(outer, o1),
                           at(inner, a1), at(inner, (a0 + a1) / 2), at(inner, a0)]
                self.mesh(role, built(outline, side), construction="ashlar voussoir arch", unit="voussoir",
                          course=i, host_id=host, **params)
        reach = half + ring

        def edge(y):
            return max(0., -e + math.sqrt(max(0., outer * outer - y * y)))
        bands = math.ceil(top / .7)
        host = self.host(wall_role)
        for band in range(bands):
            ya, yb = top * band / bands, top * (band + 1) / bands
            outline = [(edge(ya), ya), (reach, ya), (reach, yb), (edge(yb), yb), (edge((ya + yb) / 2), (ya + yb) / 2)]
            pieces = max(1, math.ceil((reach - edge(yb)) / 1.25))
            step = (reach - edge(yb)) / pieces
            cuts = [edge(yb) + step * k for k in range(pieces + 1)]
            if band % 2 and pieces > 1:                                   # break joint with the band below
                cuts = [cuts[0]] + [x + step / 2 for x in cuts[:-1]] + [cuts[-1]]
            for xa, xb in zip(cuts, cuts[1:]):
                piece = outline
                for bound, keep in ((xa, 1), (xb, -1)):
                    piece = clip(piece, lambda p, b=bound, k=keep: k * (p[0] - b) >= -1e-9,
                                 lambda p, q, b=bound: (b, p[1] + (q[1] - p[1]) * (b - p[0]) / (q[0] - p[0])))
                piece = clean(piece)
                if piece:
                    for side in (1, -1):
                        self.stone(wall_role, built(piece, side), host, band, register="spandrel", **params)

    def arcade(self, role, arch_role, along, cross, length, y0, y1, thickness, arches, back=None, warp=None,
               ring=.35, **params):
        """A straight wall on X with pointed openings; ``back`` closes each one as a blind niche."""
        specs = []
        for i, (centre, half, sill, spring) in enumerate(arches):
            name = f"{role}.{self.counts[role]}.{i}"
            specs += [{"id": name, "center": centre, "width": 2 * half, "sill": sill - y0, "height": spring - sill},
                      {"id": name + ".head", "center": centre, "width": 2 * (half + ring), "sill": spring - y0,
                       "height": arch_shape(half, ring)[5]}]
        self.ashlar(role, (along, (y0 + y1) / 2, cross), (length, y1 - y0, thickness), specs, "x", None, warp, **params)
        for centre, half, sill, spring in arches:
            self.pointed_arch(arch_role, role, along + centre, cross, thickness, half, spring, ring, warp, **params)
            if back:
                rise = arch_shape(half, 0.)[5]
                self.panels(back, along + centre, cross + thickness / 2 + .09, 2 * half + .5, sill, spring + rise + .1,
                            .18, warp, **params)

    def niches(self, length):
        """(half-span, sill, spring) for each storey of niches in a wall bay of this length."""
        base, wall = self.P, self.head - self.P
        storeys = 2 if self.line("niches") == "two-storey" else 1
        height = wall / storeys
        half = min(.3 * length, 1.5 if storeys == 2 else 2.1, (height - 1.7) / 1.3)
        top = arch_shape(half, .35)[5]
        out = []
        for s in range(storeys):
            low = base + s * height
            spring = low + height - .7 - top
            out.append((half, low + (0. if s == 0 else min(.9, (spring - low) * .3)), spring))
        return out

    # ----------------------------------------------------------- one side
    def face(self):
        """The front face: the pishtaq and its iwan, the door behind it, and a niche bay each side."""
        T, Wp, base, head, crown = self.T, self.Wp, self.P, self.head, self.crown
        depth = PROUD + WALL
        ring, half = .45, Wp - 1.0
        spring = crown - arch_shape(half, ring)[5] - max(1.2, .1 * (crown - base))
        self.arcade("pishtaq", "iwan_arch", 0, -T - PROUD + depth / 2, 2 * Wp, base, crown, depth,
                    [(0, half, base, spring)], ring=ring)
        # The iwan is closed by a screen wall a portal's depth behind the face, with one lintelled door.
        back = min(head - DECK, spring + arch_shape(half, 0.)[5] + .4)
        self.panels("iwan_back", 0, -T + WALL + .15, 2 * half + 1., base, back, .3, skip=(-.7, .7, base + 2.4))
        self.member("door_lintel", -1., 1., base + 2.4, base + 2.7, -T + WALL, -T + WALL + .3)
        bay = T - self.c - Wp
        for side in (-1, 1):
            self.arcade("tomb_wall", "niche_arch", side * (Wp + bay / 2), -T + WALL / 2, bay, base, head, WALL,
                        [(0, *niche) for niche in self.niches(bay)], back="niche_back")
            # A slender shaft at each edge of the portal, rising past its head.
            self.lathe("guldasta", side * (Wp + .3), -T - .3,
                       [(base, .3), (crown, .3), (crown + .25, .42), (crown + .6, .28), (crown + 1.4, .04)], sides=8)

    def corner(self):
        """The cut corner between the front and left faces, built on X and swung through 45 degrees."""
        T, c = self.T, self.c
        length = c * math.sqrt(2)
        self.arcade("chamfer_wall", "niche_arch", 0, WALL / 2, length, self.P, self.head, WALL,
                    [(0, *niche) for niche in self.niches(length)], back="niche_back",
                    warp=rotation(-math.pi / 4, -(T - c / 2), -(T - c / 2)))

    def kiosk(self, prefix, cx, cz, y, radius, scale):
        """A chhatri: a slab, a ring of columns, a wide eave and a small dome."""
        octagonal = self.line("chhatris") == "octagonal"
        count, reach = (8, radius - .22) if octagonal else (4, radius - .22)
        tall, post = 2.1 * scale, .3 * scale
        side = reach / math.sqrt(2) + post
        if octagonal:
            self.drum(f"{prefix}_base", cx, cz, y, y + .2, radius + .15, sides=8)
        else:
            self.member(f"{prefix}_base", cx - side, cx + side, y, y + .2, cz - side, cz + side)
        for i in range(count):
            angle = TAU * (i + .5) / count
            self.stack(f"{prefix}_column", cx + reach * math.cos(angle), y + .2, cz + reach * math.sin(angle),
                       post, post, tall, "CYLINDER" if octagonal else "CUBE")
        eave = y + .2 + tall
        if octagonal:
            self.drum(f"{prefix}_eave", cx, cz, eave, eave + .16, radius + .4, sides=8)
        else:
            self.member(f"{prefix}_eave", cx - side - .4, cx + side + .4, eave, eave + .16,
                        cz - side - .4, cz + side + .4)
        self.lathe(f"{prefix}_dome", cx, cz, dome_profile("onion" if self.taj else "pointed", radius * .8, eave + .16,
                                                           radius * 1.3, steps=5), sides=10, course=.5)
        return eave + .16 + radius * 1.3

    def minaret(self):
        """One tapering minaret at the front-left corner of the plinth, with three balconies and a chhatri."""
        P = self.P
        foot = clamp(.034 * self.w, 1.3, 1.9)
        head = .72 * foot
        m = self.w / 2 - foot - .6
        top = P + .6 * (self.h - P) * self.vary("minaret", .04)
        for k in range(3):
            y0, y1 = P + (top - P) * k / 3, P + (top - P) * (k + 1) / 3
            r0, r1 = foot + (head - foot) * k / 3, foot + (head - foot) * (k + 1) / 3
            self.lathe("minaret_shaft", -m, -m, [(y0, r0), (y1 - .28, r1)], sides=12, stage=k)
            self.drum("minaret_balcony", -m, -m, y1 - .28, y1, r1 + .45, sides=12, stage=k)
        return round(self.kiosk("minaret", -m, -m, top, head + .1, .7), 3)

    # ---------------------------------------------------------- the centre
    def roof(self, radius):
        """The inner chamber that carries the drum, and the flat roof over the eight-sided plan."""
        P, deck = self.P, self.head - DECK
        for k in range(8):                                       # eight piers stand under the drum
            angle = TAU * (k + .5) / 8
            self.stack("chamber_pier", (radius - .35) * math.cos(angle), P, (radius - .35) * math.sin(angle),
                       1.4, 1.4, deck - P)
        half = self.T - WALL
        cut = self.c - WALL * (2 - math.sqrt(2))
        flat = half - cut
        self.slab("roof_deck", -flat, flat, -half, half, deck, self.head, module=3.2)
        for side in (-1, 1):
            a, b = sorted((side * flat, side * half))
            self.slab("roof_deck", a, b, -flat, flat, deck, self.head, module=3.2)
            for end in (-1, 1):
                corners = [(side * flat, end * flat), (side * half, end * flat), (side * flat, end * half)]
                self.mesh("roof_deck", _prism(corners, deck, self.head))

    def finial(self, y, collar):
        """The lotus over the dome's collar and the gilded finial; its tip is the height of the building."""
        rise = self.h - y
        self.lathe("lotus", 0, 0, [(y, collar * 1.6), (y + .22 * rise, collar * 1.15)], sides=16, course=.4)
        if self.line("finial") == "lotus-kalash":
            shape = ((.22, .5), (.38, .85), (.5, .35), (.62, .62), (.74, .25), (1., .06))
        else:
            shape = ((.22, .5), (.34, .3), (1., .06))
        self.lathe("finial", 0, 0, [(y + u * rise, collar * r) for u, r in shape], sides=12, course=.4)


def _prism(corners, y0, y1):
    return hull([(x, y, z) for y in (y0, y1) for x, z in corners])


Tomb.module = sys.modules[__name__]


def random_controls(product, rng):
    """The plot is square, so the depth follows the width instead of being drawn apart from it."""
    width = rng.randint(36, 56)
    return {"width": width, "depth": width, "height": clamp(round(width * rng.uniform(.67, .83)), 24, 47)}


def build_document(request):
    return Tomb(request).compose()


def _side(p):
    """Which of the four sides a member belongs to: 0 front, 1 right, 2 back, 3 left."""
    x, z = p.position[0], p.position[2]
    return (0 if z < 0 else 2) if abs(z) >= abs(x) else (1 if x > 0 else 3)


def _quadrants(members):
    groups = defaultdict(list)
    for p in members:
        groups[(p.position[0] > 0, p.position[2] > 0)].append(p)
    return groups


def _on_diagonals(members, ceiling):
    groups = _quadrants(members)
    if len(groups) != 4:
        return False
    for group in groups.values():
        lo, hi = extent(group)
        cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        if abs(abs(cx) - abs(cz)) > .15 or hi[1] >= ceiling:
            return False
    return True


def rule_checks(document, form):
    placements = document.placements
    roles = defaultdict(list)
    for p in placements:
        roles[p.role].append(p)

    def key(p, turned):
        x, y, z = p.position
        dx, dy, dz = p.dimensions
        if turned:
            x, z, dx, dz = -z, x, dz, dx
        return (p.role, round(x, 2) + 0., round(y, 2), round(z, 2) + 0., round(dx, 2), round(dy, 2), round(dz, 2))
    chosen = [p for p in placements if p.role in TURNED]
    seen = {key(p, False) for p in chosen}
    lone = [p.id for p in chosen if key(p, True) not in seen]
    yield ("fourfold_symmetry", bool(chosen) and not lone,
           "Every face, cut corner, kiosk and minaret has its twin a quarter turn away.",
           "A member has no twin a quarter turn about the vertical axis.", lone)

    crowned = False
    dome_top = 0.
    if roles["dome"] and roles["drum"] and roles["finial"]:
        (dlo, dhi), (rlo, rhi), (flo, fhi) = (extent(roles[name]) for name in ("dome", "drum", "finial"))
        dome_top = dhi[1]
        centred = all(abs(lo[a] + hi[a]) < .1 for lo, hi in ((dlo, dhi), (rlo, rhi), (flo, fhi)) for a in (0, 2))
        highest = max(p.position[1] + p.dimensions[1] / 2 for p in placements)
        below = max(p.position[1] + p.dimensions[1] / 2 for p in placements if p.role not in {"dome", "lotus", "finial"})
        crowned = centred and abs(dlo[1] - rhi[1]) < .05 and abs(fhi[1] - highest) < 1e-6 and below < dome_top
    yield ("crowning_dome", crowned, "The dome sits on its drum over the centre of the plan and rises above all else.",
           "The dome is missing, off the axis, off its drum, or overtopped.",
           [] if crowned else [p.id for p in roles["dome"][:3]])

    portals, walls = defaultdict(list), defaultdict(list)
    for p in roles["pishtaq"]:
        portals[_side(p)].append(p)
    for p in roles["tomb_wall"]:
        walls[_side(p)].append(p)
    low = [side for side in range(4) if not portals[side] or not walls[side]
           or extent(portals[side])[1][1] < extent(walls[side])[1][1] + 1.]
    yield ("pishtaq_rises", not low, "On each of the four faces the portal rises above the wall head.",
           "A face has no portal, or its portal does not rise above the wall head.",
           [p.id for side in low for p in portals[side][:2]])

    kiosks = _on_diagonals(roles["chhatri_dome"], dome_top)
    style = next((p.parameters.get("style") for p in placements if p.parameters.get("style")), None)
    shafts = roles["minaret_shaft"]
    towers = _on_diagonals(shafts + roles["minaret_dome"], dome_top) if style == "taj" or shafts else True
    yield ("diagonal_kiosks", kiosks and towers,
           "Chhatris, and minarets where there are any, stand at the four diagonals below the dome.",
           "A chhatri or minaret is missing, off its diagonal, or as high as the dome.",
           [] if kiosks and towers else [p.id for p in (roles["chhatri_dome"] + shafts)[:3]])

    seated = False
    tomb = [p for name in TOMB for p in roles[name]]
    if tomb and roles["plinth_wall"] and roles["plinth_paving"]:
        (tlo, thi), (plo, phi) = extent(tomb), extent(roles["plinth_wall"])
        deck = extent(roles["plinth_paving"])[1][1]
        seated = abs(tlo[1] - deck) < .05 and all(tlo[a] > plo[a] + 1. and thi[a] < phi[a] - 1. for a in (0, 2))
    yield ("tomb_on_plinth", seated, "The tomb stands on the plinth, well inside its edge.",
           "The tomb does not rest on the plinth or reaches its edge.", [] if seated else [p.id for p in tomb[:3]])
