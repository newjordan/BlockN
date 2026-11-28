"""Buddhist stupa architect: the relic mound and the path around it.

A stupa is not entered. It is a solid mound over a relic, and the act of
worship is to walk round it keeping it on the right. So its rules are about a
centre and a circle: a dome on a round drum, a clear path all the way round,
gateways on the four cardinal axes, and one vertical axis that rises through
the dome to a square railing, a mast and its parasols.

``great-stupa`` follows the Great Stupa at Sanchi: a stone railing with four
toranas, a terrace reached by a pair of stairs laid against the drum, and a
mast of parasol discs. ``dagoba`` follows the brick dagobas of Anuradhapura:
basal rings instead of a walked terrace, a frontispiece (vahalkada) on each
axis, and the parasols fused into a ringed cone.
"""
from __future__ import annotations

import math
import sys

from .historic import TAU, clamp, extent, sector_mesh
from .schema import ArchitectInfo
from .sonnet import Poem, controls_for, form

ARCHITECT = ArchitectInfo("stupa", "1.0.0", "Buddhist Stupa Architect")
STYLE = "buddhist-stupa"
PRODUCT = "buddhist-stupa"
REGION = "Central India (Sanchi) and Sri Lanka (Anuradhapura)"
PERIOD = "Maurya to Satavahana, 3rd–1st century BCE, and the Anuradhapura period"
VARIANTS = ("great-stupa", "dagoba")
KINDS = {
    "ground_paving": "slab", "core_fill": "core",
    "vedika_post": "railing", "vedika_rail": "railing", "vedika_coping": "railing",
    "torana_pillar": "column", "torana_architrave": "beam",
    "vahalkada": "wall", "vahalkada_stele": "column",
    "medhi_wall": "wall", "terrace_paving": "slab", "terrace_stair": "circulation",
    "terrace_railing": "railing",
    "anda": "dome", "harmika": "wall", "harmika_coping": "slab", "yasti": "mast", "chattra": "ornament",
}
REQUIRED = {"ground_paving", "core_fill", "vedika_rail", "vedika_coping", "medhi_wall", "terrace_paving",
            "anda", "harmika", "harmika_coping", "yasti", "chattra"}
FORMS = {PRODUCT: form(
    PRODUCT, "Solid relic mound: a dome on a round drum inside a railed circumambulation path with four gateways",
    (36, 17, 36), VARIANTS, REQUIRED,
    lines={"anda": ("hemisphere", "bubble", "bell"), "parasols": ("triple", "single", "seven"),
           "torana": ("three-architrave", "two-architrave"), "terrace": ("single", "double")},
    size={"width": (28, 48), "height": (12, 24), "depth": (28, 48)},
    ratios={"depth/width": (.9, 1.1), "height/width": (.4, .52)},
    region=REGION, period=PERIOD,
    plan="circular, set out from the shorter side of the plot",
    openings="none; the mound is solid")}
DEFAULT_SUBSTANCE = "sandstone"
SUBSTANCES = {"core_fill": "fired-clay-brick"}
# A dagoba is brick throughout; what shows of it is the coat of lime. Only its frontispieces are stone.
DAGOBA_LIME = {"medhi_wall", "anda", "harmika", "harmika_coping", "yasti", "chattra"}
DAGOBA_BRICK = {"vedika_rail", "vedika_coping", "terrace_paving"}
GROUND = .2         # the paved court the whole monument stands on
PAVING = .25
GATE = 1.85         # half the break in the railing at each gateway
PATH = 1.5          # the least clear width of the circumambulation path
MASS = {"medhi_wall", "terrace_paving", "terrace_stair", "terrace_railing", "vahalkada", "vahalkada_stele",
        "anda"}
HIDDEN = {"ground_paving", "core_fill"}     # under the path, or inside the mound
RAILING = {"vedika_post", "vedika_rail", "vedika_coping", "torana_pillar"}
GATEWAY = {"torana_pillar", "torana_architrave", "vahalkada", "vahalkada_stele"}


def substances_for(form):
    return {DEFAULT_SUBSTANCE, "fired-clay-brick", "lime-plaster", "limestone"}


def anda_radius(shape, u):
    """Radius of the dome, as a share of its greatest radius, at share ``u`` of the way to its apex."""
    if shape == "hemisphere":
        return math.sqrt(max(0., 1 - u * u))
    if shape == "bubble":                       # more than half a sphere: it swells above its foot
        return math.sqrt(max(0., 1 - (1.22 * u - .22) ** 2))
    return ((1 - u ** 2.6) ** .55 + .08 * (1 - u) ** 5) / 1.08      # bell: a flared foot, a long shoulder


def anda_stations(shape, radius, base, height, cut=.93):
    """(height, radius) stations up to the flat crown, close enough that every course is a liftable stone."""
    points = [(base, radius * anda_radius(shape, 0.))]
    for i in range(1, 401):
        y, r = base + height * i / 400, radius * anda_radius(shape, cut * i / 400)
        if abs(r - points[-1][1]) >= .33 or y - points[-1][0] >= .55 or i == 400:
            if i == 400 and y - points[-1][0] < .16 and len(points) > 1:
                points.pop()
            points.append((y, r))
    return points


class Stupa(Poem):
    PALETTE = {"ground_paving": (186, 170, 146, 255), "core": (158, 104, 78, 255),
               "railing": (196, 172, 140, 255), "column": (200, 176, 142, 255), "beam": (204, 180, 146, 255),
               "wall": (194, 168, 134, 255), "terrace_paving": (184, 164, 136, 255),
               "terrace_stair": (188, 166, 136, 255), "dome": (206, 186, 156, 255),
               "harmika_coping": (200, 178, 146, 255), "mast": (190, 166, 132, 255),
               "chattra": (208, 186, 150, 255)}
    @property
    def dagoba(self):
        return self.variant == "dagoba"

    def substance(self, role, kind):
        if self.dagoba and role in DAGOBA_LIME:
            return "lime-plaster"
        if self.dagoba and role in DAGOBA_BRICK:
            return "fired-clay-brick"
        if self.dagoba and role.startswith("vahalkada"):
            return "limestone"
        return super().substance(role, kind)

    def compose(self):
        h, radius = self.h, min(self.w, self.d) / 2
        double = self.line("terrace") == "double"
        spire = (.30 if self.dagoba else .17) * h
        harmika = clamp(.065 * h, 1.0, 1.6)
        self.rail = radius - .8
        drum = .56 * radius * self.vary("drum", .04)
        walk = clamp(.11 * radius, 1.5, 2.1) * self.vary("walk", .08)
        dome = drum - walk
        rest = h - GROUND - spire - harmika
        rise = clamp(rest - (.19 if double else .13) * h, .8 * dome, 1.3 * dome)
        terrace = rest - rise
        if not 1.2 <= terrace <= .36 * h:
            raise ValueError(f"{PRODUCT} cannot seat a dome of this width at this height")
        if self.dagoba:         # three basal rings; ``double`` broadens the lowest into a walk
            steps, shares = ((1.5, .45, 0.) if double else (.9, .45, 0.)), (1 / 3, 2 / 3, 1.)
        else:
            steps, shares = ((1.3, 0.) if double else (0.,)), ((.42, 1.) if double else (1.,))
        self.slab("ground_paving", -self.w / 2, self.w / 2, -self.d / 2, self.d / 2, 0, GROUND, module=3.0)
        tiers, y = [], GROUND
        for step, share in zip(steps, shares):
            tiers.append((drum + step, y, GROUND + terrace * share))
            y = GROUND + terrace * share
        for k, (r, y0, y1) in enumerate(tiers):
            inner = (tiers[k + 1][0] if k + 1 < len(tiers) else dome) - .3
            self.tier(k, r, y0, y1, inner)
        self.enclosure(tiers[0][0], terrace)
        crown = self.dome(dome, GROUND + terrace, rise)
        self.axis(crown, GROUND + terrace + rise, harmika)
        return self.document({"region": REGION, "period": PERIOD, "dome_radius": round(dome, 3),
                              "dome_rise": round(rise, 3), "terrace_height": round(terrace, 3),
                              "railing_radius": round(self.rail, 3), "gateways": 4,
                              "construction": "stone or plastered brick facing over a solid brick core"})

    # ------------------------------------------------------------ the drum
    def tier(self, k, r, y0, y1, inner):
        top = y1 - PAVING
        self.ring("medhi_wall", 0, 0, r, .6, y0, top, tier=k)
        for radius, lo, hi in ((r - .6, y0, top), (inner, top, y1)):
            self.drum("core_fill", 0, 0, lo, hi, radius, radius, 24, construction="rammed fill", unit="core", tier=k)
        bands = max(1, math.ceil((r - inner) / .65 - 1e-9))
        for i in range(bands):
            self.ring("terrace_paving", 0, 0, r - (r - inner) * i / bands, (r - inner) / bands, top, y1,
                      grade="standard", tier=k)
        if self.dagoba:
            return
        # Two flights laid against the drum, meeting on the south axis, so the ground path stays clear.
        width = 1.3 if k == 0 else 1.1
        steps = math.ceil((y1 - y0) / .19)
        turn = min(.30 / (r + width / 2), 1.15 / steps)
        for side in (-1, 1):
            for j in range(steps):
                a0, a1 = sorted(-math.pi / 2 + side * turn * (steps - j - n) for n in (0, 1))
                self.mesh("terrace_stair", sector_mesh(0, 0, r, r + width, a0, a1, y0, y0 + (y1 - y0) * (j + 1) / steps),
                          step=j, tier=k, flight="west" if side < 0 else "east", construction="solid stair")
        self.ring("terrace_railing", 0, 0, r - .05, .22, y1, y1 + .75, tier=k,
                  gaps=((3 * math.pi / 2, max(1.0 / r, 2 * turn), y1, y1 + .75),))

    def dome(self, radius, base, rise):
        stations = anda_stations(self.line("anda"), radius, base, rise)
        self.revolve("anda", stations, thickness=.5, core="core_fill", sides=24, grade="standard")
        return stations[-1][1]

    # ------------------------------------------------------------ the axis
    def axis(self, crown, y, height):
        half = .8 * crown
        body = y + height - .2
        self.ashlar("harmika", (0, (y + body) / 2, 0), (2 * half, body - y, 2 * half))
        self.member("harmika_coping", -half - .15, half + .15, body, y + height, -half - .15, half + .15)
        y, spire = y + height, self.h - y - height
        if self.dagoba:
            count = {"single": 3, "triple": 7, "seven": 13}[self.line("parasols")]
            neck = y + .2 * spire
            self.lathe("yasti", 0, 0, [(y, .5 * half), (neck, .5 * half)])
            foot, tip = .42 * half * self.vary("spire", .08), .1
            pitch = (self.h - .5 - neck) / count
            for k in range(count):
                r0, r1 = (foot + (tip - foot) * n / count for n in (k, k + 1))
                self.drum("chattra", 0, 0, neck + k * pitch, neck + (k + 1) * pitch, r0, r0 - (r0 - r1) * .45,
                          ring=k)
            self.drum("yasti", 0, 0, self.h - .5, self.h, tip * .9, .03, member="pinnacle")
            return
        count = {"triple": 3, "single": 1, "seven": 7}[self.line("parasols")]
        widest = min(1.9, .62 * half) * self.vary("parasol", .08)
        thick = min(.24, spire / (2.6 * count))
        pitch = (spire - .35) / count
        for k in range(count):
            seat = y + pitch * (k + 1) - thick
            self.drum("yasti", 0, 0, y + pitch * k, seat, .15, .15, 10, member="mast", ring=k)
            r = widest * (1 - .7 * k / count)
            self.drum("chattra", 0, 0, seat, seat + thick, r, .55 * r, ring=k)
        self.drum("yasti", 0, 0, self.h - .35, self.h, .12, .03, 10, member="finial")

    # ------------------------------------------------------------ the path and its gates
    def enclosure(self, foot, terrace):
        rail = self.rail
        gaps = lambda y0, y1: tuple((k * math.pi / 2, GATE / rail, y0, y1) for k in range(4))
        if self.dagoba:
            self.ring("vedika_rail", 0, 0, rail - .05, .45, GROUND, GROUND + 1.1, gaps=gaps(GROUND, GROUND + 1.1))
            self.ring("vedika_coping", 0, 0, rail, .55, GROUND + 1.1, GROUND + 1.3, gaps=gaps(GROUND + 1.1, GROUND + 1.3),
                      grade="standard")
        else:
            top = GROUND + 2.1
            count = round(TAU * (rail - .2) / .8)
            for i in range(count):
                a = TAU * (i + .5) / count
                off = abs((a + math.pi / 4) % (math.pi / 2) - math.pi / 4)      # angle to the nearest axis
                if off * rail > GATE + .25:
                    self.drum("vedika_post", (rail - .2) * math.cos(a), (rail - .2) * math.sin(a), GROUND, top,
                              .2, .2, 8, post=i)
            for n, y in enumerate((.55, 1.1, 1.65)):
                self.ring("vedika_rail", 0, 0, rail - .13, .14, GROUND + y, GROUND + y + .28,
                          gaps=gaps(GROUND + y, GROUND + y + .28), grade="standard", rail=n)
            self.ring("vedika_coping", 0, 0, rail, .4, top, top + .3, gaps=gaps(top, top + .3), grade="standard")
        start = len(self.placements)
        if self.dagoba:
            self.vahalkada(foot, terrace)
        else:
            self.torana()
        count = len(self.placements) - start
        for quarter in (1, 2, 3):
            copy = self.clone(start, count)
            self.turn(copy, quarter)
            for p in self.placements[copy:]:            # each gateway is its own piece of masonry
                if "host_id" in p.parameters:
                    p.parameters["host_id"] = f"{p.parameters['host_id']}.q{quarter}"

    def torana(self):
        """One gateway on the south axis: two pillars under two or three architraves."""
        count = 3 if self.line("torana") == "three-architrave" else 2
        total = clamp(.42 * self.h, 5.5, 8.5) * self.vary("torana", .06)
        z, beam = -(self.rail - .2), .5
        shaft = total - (2 * count - 1) * beam
        for side in (-1, 1):
            x = side * (GATE - .35)
            self.stack("torana_pillar", x, GROUND, z, .7, .7, shaft)
            for k in range(1, count):
                y = GROUND + shaft + (2 * k - 1) * beam
                self.member("torana_pillar", x - .25, x + .25, y, y + beam, z - .25, z + .25, member="die", tier=k)
        for k in range(count):
            y = GROUND + shaft + 2 * k * beam
            reach = GATE + 1.2 - .12 * k
            self.member("torana_architrave", -reach, reach, y, y + beam, z - .3, z + .3, tier=k)

    def vahalkada(self, foot, terrace):
        """One frontispiece on the south axis: stepped registers against the basal rings, between two stelae."""
        count = 3 if self.line("torana") == "three-architrave" else 2
        half = 1.9
        back = -(foot + .02 + foot - math.sqrt(foot * foot - half * half))    # clear of the curve of the ring
        rise = (terrace + 1.0) / count
        for k in range(count):
            wide, deep = half - .35 * k, 1.0 - .22 * k
            self.ashlar("vahalkada", (0, GROUND + (k + .5) * rise, back - deep / 2), (2 * wide, rise, deep), axis="x",
                        register=k)
        for side in (-1, 1):
            self.stack("vahalkada_stele", side * (half + .3), GROUND, back - .55, .45, .45, terrace + 1.6)


Stupa.module = sys.modules[__name__]
random_controls = controls_for(sys.modules[__name__])


def build_document(request):
    return Stupa(request).compose()


def _reach(p):
    """Nearest and farthest plan distance from the axis to a member's envelope."""
    x, _, z = p.position
    if p.block_type == "MESH":      # a curved stone is measured at its own corners, not its bounding box
        radii = [math.hypot(x + v[0] * p.dimensions[0], z + v[2] * p.dimensions[2]) for v in p.parameters["vertices"]]
        return min(radii), max(radii)
    hx, hz = p.dimensions[0] / 2, p.dimensions[2] / 2
    near = math.hypot(max(abs(x) - hx, 0.), max(abs(z) - hz, 0.))
    return near, math.hypot(abs(x) + hx, abs(z) + hz)


def _turned(p):
    """A member's place and size as plain numbers, and the same after a quarter turn about the axis."""
    def key(x, z, dx, dz):
        return (p.role, round(x, 3) + 0., round(p.position[1], 3), round(z, 3) + 0.,
                round(dx, 3), round(p.dimensions[1], 3), round(dz, 3))
    (x, _, z), (dx, _, dz) = p.position, p.dimensions
    return key(x, z, dx, dz), key(-z, x, dz, dx)


def rule_checks(document, form):
    placements = document.placements
    by = lambda roles: [p for p in placements if p.role in roles]
    ground = max((p.position[1] + p.dimensions[1] / 2 for p in by({"ground_paving"})), default=0.)

    mass, railing = by(MASS), by(RAILING)
    inner = max((_reach(p)[1] for p in mass if p.position[1] - p.dimensions[1] / 2 < ground + 2.2), default=0.)
    outer = min((_reach(p)[0] for p in railing), default=0.)
    blocked = [p.id for p in placements if p.role not in MASS | RAILING | HIDDEN
               and p.position[1] - p.dimensions[1] / 2 < ground + 2.2
               and _reach(p)[0] < outer - 1e-4 and _reach(p)[1] > inner + 1e-4]
    yield ("clear_path", bool(mass and railing) and outer - inner >= PATH and not blocked,
           "A clear path circles the mound between the drum and the railing.",
           "The circumambulation path is missing, too narrow, or blocked.", blocked)

    gates = by(GATEWAY)
    here = {_turned(p)[0] for p in gates}
    lone = [p.id for p in gates if _turned(p)[1] not in here]
    on_axis = [p for p in gates if min(abs(p.position[0]), abs(p.position[2])) < .01]
    yield ("cardinal_gateways", len(on_axis) >= 4 and not lone,
           "A gateway stands on each of the four cardinal axes, and all four are alike.",
           "A gateway is missing, off its axis, or unlike the others.", lone)

    dome, drum = by({"anda"}), by({"medhi_wall"})
    seated, centre = False, (0., 0.)
    if dome and drum:
        (lo, hi), (drum_lo, drum_hi) = extent(dome), extent(drum)
        centre = ((lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2)
        walk = max(p.position[1] + p.dimensions[1] / 2 for p in by({"terrace_paving"}) or drum)
        seated = abs(lo[1] - walk) < .05 and all(
            lo[a] > drum_lo[a] + .8 and hi[a] < drum_hi[a] - .8 for a in (0, 2)) and \
            abs(hi[0] - lo[0] - (hi[2] - lo[2])) < .1
    yield ("dome_on_drum", seated, "The dome sits on the drum, inside its edge, and is round in plan.",
           "The dome is missing, off the drum, or as wide as the drum.", [] if seated else [p.id for p in dome[:3]])

    crown = by({"harmika_coping", "yasti", "chattra"})
    stray = [p.id for p in crown if math.hypot(p.position[0] - centre[0], p.position[2] - centre[1]) > .05]
    box = by({"harmika"})
    discs = sorted(by({"chattra"}), key=lambda p: p.position[1])
    widths = [p.dimensions[0] for p in discs]
    upright = bool(dome and box and discs and by({"yasti"})) and not stray
    if upright:
        (lo, hi), top = extent(box), extent(dome)[1][1]
        upright = abs(lo[1] - top) < .05 and abs((lo[0] + hi[0]) / 2 - centre[0]) < .05 and \
            abs((lo[2] + hi[2]) / 2 - centre[1]) < .05 and discs[0].position[1] > hi[1] and \
            all(b < a - 1e-4 for a, b in zip(widths, widths[1:]))
    yield ("axis", upright, "Harmika, mast and parasols stand on the dome's axis, and the parasols diminish upward.",
           "The crowning members are missing, off the dome's axis, or do not diminish upward.", stray)
