"""Mesoamerican architect: stepped temple pyramids.

Three building traditions share one logic: a rubble-and-earth core faced with
cut stone, raised in receding terraces, climbed by a steep balustraded stair
to a small temple. Nothing is arched; every opening is spanned by a lintel.

``talud-tablero`` follows Teotihuacan: each terrace is a sloping talud under a
framed vertical tablero. ``radial`` follows the Postclassic Maya calendar
pyramid: nine terraces and four stairs of 91 steps, which with the temple
platform count the 365 days of the year. ``maya-temple`` follows Classic Tikal:
nine steep terraces, one stair, and a temple under a tall roof comb.
"""
from __future__ import annotations

import math
import sys

from .historic import Composer, Form, Voids, clamp, extent, shear_y, taper
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("mesoamerican", "1.0.0", "Mesoamerican Architect")
STYLE = "mesoamerican"
KINDS = {
    "core_fill": "core",
    "terrace_face": "wall", "talud": "wall", "tablero": "wall", "tablero_frame": "wall",
    "terrace_paving": "slab", "stair_step": "circulation", "alfarda": "slab",
    "temple_wall": "wall", "temple_lintel": "entablature", "temple_cornice": "entablature",
    "temple_roof": "roof", "roof_comb": "wall", "almena": "ornament", "foundation": "foundation",
}
FORMS = {form.name: form for form in (
    Form("mesoamerican-pyramid", "Stepped temple pyramid with balustraded stairs and a summit temple",
         (36, 24, 36), ("talud-tablero", "radial", "maya-temple"),
         frozenset({"terrace_paving", "stair_step", "alfarda", "temple_wall", "temple_lintel", "temple_roof"}),
         {"levels": 9},
         rules={"levels": [3, 9], "levels_meaning": "terraces; talud-tablero defaults to 4",
                "height_to_width": [0.4, 1.3], "depth_to_width": [0.8, 1.25],
                "openings": "lintelled only; no true arch", "radial_steps": "4 x 91 + platform = 365"}),
)}
PALETTE = {"core_fill": (150, 132, 108, 255), "temple_roof": (188, 176, 150, 255), "tablero_frame": (196, 120, 96, 255),
           "almena": (196, 120, 96, 255), "alfarda": (176, 166, 146, 255)}
# Cut limestone facing and core, plastered where the tradition finished and
# painted its surfaces: the framed tablero panel, the roof comb, and the lime
# concrete slab over the corbelled temple vault.
SUBSTANCES = {
    "core_fill": "limestone",
    "terrace_face": "limestone", "talud": "limestone",
    "tablero": "lime-plaster", "tablero_frame": "limestone",
    "terrace_paving": "limestone", "stair_step": "limestone", "alfarda": "limestone",
    "temple_wall": "limestone", "temple_lintel": "limestone", "temple_cornice": "limestone",
    "temple_roof": "lime-mortar", "roof_comb": "lime-plaster",
    "almena": "limestone", "foundation": "limestone",
}
DEFAULT_SUBSTANCE = "limestone"
# variant -> (summit share of the base, temple share of the height, stairs)
SHAPE = {"talud-tablero": (.42, .2, 1), "radial": (.36, .22, 4), "maya-temple": (.34, .36, 1)}


class Pyramid(Composer):
    PALETTE = PALETTE
    T = 0.5

    def compose(self):
        self.ratio(self.h / self.w, "height/width", .4, 1.3)
        self.ratio(self.d / self.w, "depth/width", .8, 1.25)
        variant = self.variant
        default = 4 if variant == "talud-tablero" else 9
        levels = self.control("levels", 3, 9) if self.request.levels is not None else default
        share, temple_share, stairs = SHAPE[variant]
        share *= self.vary("summit", .12)
        t = self.T
        temple_h = clamp(temple_share * self.h * self.vary("temple", .15), 3.0, 20.0)
        height = self.h - temple_h
        ax, az = self.w / 2 - (1.0 if stairs == 4 else 0), self.d / 2 - 1.0   # stairs stand a metre proud
        bx, bz = ax * share, az * share
        rise = height / levels
        self.grade = {"concept": "concept", "standard": "concept", "presentation": "standard"}[self.detail]
        feet = []
        for k in range(levels):
            # Each terrace: a facing that leans back, then a level walk to the next foot.
            x0, z0 = ax - (ax - bx) * k / levels, az - (az - bz) * k / levels
            x1, z1 = ax - (ax - bx) * (k + 1) / levels, az - (az - bz) * (k + 1) / levels
            y0, y1 = k * rise, (k + 1) * rise
            feet.append((x0, z0))
            if variant == "talud-tablero":
                slope_h = rise * .32
                lean_x, lean_z = min((x0 - x1) * .5, slope_h * .7), min((z0 - z1) * .5, slope_h * .7)
                tx, tz = x0 - lean_x, z0 - lean_z
                self.shell("talud", -x0, x0, -z0, z0, y0, y0 + slope_h, t,
                           warp=taper(0, 0, y0, y0 + slope_h, tx / x0, tz / z0), terrace=k)
                out, band = .3, (rise - slope_h) * .22
                px, pz = tx + out, tz + out
                self.shell("tablero_frame", -px, px, -pz, pz, y0 + slope_h, y0 + slope_h + band, t, terrace=k)
                self.shell("tablero", -px + .2, px - .2, -pz + .2, pz - .2, y0 + slope_h + band, y1 - band, t,
                           terrace=k)
                self.shell("tablero_frame", -px, px, -pz, pz, y1 - band, y1, t, terrace=k)
                top_x, top_z = px, pz
            else:
                lean = (.55 if variant == "radial" else .7) * self.vary("lean", .12)
                top_x, top_z = x0 - (x0 - x1) * lean, z0 - (z0 - z1) * lean
                self.shell("terrace_face", -x0, x0, -z0, z0, y0, y1, t,
                           warp=taper(0, 0, y0, y1, top_x / x0, top_z / z0), terrace=k)
            last = k == levels - 1
            ix, iz = (0, 0) if last else (x1 - t, z1 - t)
            self.core(-(top_x - t), top_x - t, y0, y1 - .3, -(top_z - t), top_z - t, terrace=k)
            if last:
                self.ashlar("terrace_paving", (0, y1 - .15, 0), (2 * top_x, .3, 2 * top_z), terrace=k)
            else:
                for za, zb in ((-top_z, -iz), (iz, top_z)):
                    self.ashlar("terrace_paving", (0, y1 - .15, (za + zb) / 2), (2 * top_x, .3, zb - za), terrace=k)
                for xa, xb in ((-top_x, -ix), (ix, top_x)):
                    self.ashlar("terrace_paving", ((xa + xb) / 2, y1 - .15, 0), (xb - xa, .3, 2 * iz), terrace=k)
        steps = 91 if variant == "radial" else int(math.ceil(height / .3))
        stair_w = clamp(.2 * self.w * self.vary("stair", .18), 3.0, 9.0)
        for quarter in range(stairs):
            start = len(self.placements)
            half, summit = (az, bz) if quarter % 2 == 0 else (ax, bx)
            self.stair(stair_w, half + 1.0, summit + .2, height, steps, quarter)
            self.turn(start, quarter)
        self.temple(bx, bz, height, temple_h, stairs)
        slope = math.degrees(math.atan2(height, min(ax - bx, az - bz)))
        return self.document({"typology": f"{variant} pyramid", "terraces": levels,
                              "terrace_rise": round(rise, 3), "stairs": stairs, "steps_per_stair": steps,
                              "calendar_count": steps * stairs + 1 if variant == "radial" else None,
                              "mean_slope_degrees": round(slope, 1), "temple_height": round(temple_h, 3),
                              "construction": "cut-stone facing over a rubble and earth core",
                              "openings": "lintelled; no true arch"})

    def stair(self, width, foot, head, height, steps, quarter):
        """One flight facing -Z, climbing from ``foot`` to ``head`` (distances from the centre)."""
        tread, riser = (foot - head) / steps, height / steps
        pieces = max(1, int(math.ceil(width / 1.5)))
        for j in range(steps):
            z1 = -foot + (j + 1) * tread
            for k in range(pieces):
                x0 = -width / 2 + width * k / pieces
                self.member("stair_step", x0, x0 + width / pieces, max(0.0, (j + 1) * riser - 1.0), (j + 1) * riser,
                            z1 - tread, z1 + (.02 if j < steps - 1 else 0), step=j, flight=quarter,
                            construction="cut stone tread")
        # Alfardas: the sloping balustrades that contain the flight.
        runs = max(1, int(math.ceil((foot - head) / 1.2)))
        for side in (-1, 1):
            x0, x1 = sorted((side * width / 2, side * (width / 2 + .9)))
            for k in range(runs):
                za, zb = -foot + (foot - head) * k / runs, -foot + (foot - head) * (k + 1) / runs
                base = height * k / runs
                self.slab("alfarda", x0, x1, za, zb, max(0.0, base - 1.2), base + .45,
                          warp=shear_y(2, za, height / (foot - head)) if base > 1.2 else self.wedge(za, height / (foot - head), base + .45),
                          flight=quarter, construction="sloping balustrade slab")

    @staticmethod
    def wedge(origin, slope, top):
        """Warp for the lowest balustrade stones: level bed, sloping top."""
        def warp(vertices):
            out = vertices.copy()
            high = vertices[:, 1] > top - 1e-9
            out[high, 1] += (vertices[high, 2] - origin) * slope
            return out
        return warp

    def temple(self, bx, bz, y, height, stairs):
        t, variant = self.T, self.variant
        self.grade = self.detail
        hx, hz = bx * .62 * self.vary("shrine", .1), bz * (.5 if variant == "maya-temple" else .62) * self.vary("shrine_depth", .1)
        back = bz * .12 if variant == "maya-temple" else 0      # the shrine sits toward the rear
        comb = height * .45 * self.vary("comb", .15) if variant == "maya-temple" else 0
        wall = (height - comb) * (.62 if variant != "talud-tablero" else .7)
        door_w, door_h = clamp(.3 * 2 * hx, 1.2, 2.2), min(wall - .7, 2.6)
        openings = {}
        for face, half in (("front", hx), ("back", hx), ("left", hz), ("right", hz)):
            if face == "front" or stairs == 4:
                voids = Voids(y)
                voids.rect(0, door_w, y, door_h, f"temple.{face}")
                openings[face] = voids.specs
        self.shell("temple_wall", -hx, hx, back - hz, back + hz, y, y + wall, t, openings)
        for face in openings:
            # A single stone or sapodilla beam spans each doorway.
            along = (door_w + .8, .35, t + .1) if face in ("front", "back") else (t + .1, .35, door_w + .8)
            centre = {"front": (0, back - hz + t / 2), "back": (0, back + hz - t / 2),
                      "left": (-hx + t / 2, back), "right": (hx - t / 2, back)}[face]
            self.add("temple_lintel", (centre[0], y + door_h + .175, centre[1]), along,
                     construction="monolithic lintel", face=face)
        top = y + wall
        self.shell("temple_cornice", -hx - .25, hx + .25, back - hz - .25, back + hz + .25, top, top + .35, t + .25)
        upper = height - comb - wall - .35 - .3
        self.shell("temple_wall", -hx, hx, back - hz, back + hz, top + .35, top + .35 + upper, t,
                   warp=taper(0, back, top + .35, top + .35 + upper, .93, .93), register="upper zone")
        roof = top + .35 + upper
        self.slab("temple_roof", -hx * .93, hx * .93, back - hz * .93, back + hz * .93, roof, roof + .3,
                  construction="lime concrete on a corbelled vault")
        if comb:
            depth, lights = clamp(.28 * 2 * hz, 1.0, 2.2), Voids(roof + .3)
            for k in (-1, 0, 1):
                lights.rect(k * hx * .5, .5, roof + .3 + comb * .3, comb * .35, f"comb.{k}")
            cz = back + hz * .93 - depth / 2
            self.shell("roof_comb", -hx * .8, hx * .8, cz - depth / 2, cz + depth / 2, roof + .3, roof + .3 + comb, t,
                       {"front": lights.specs, "back": lights.specs},
                       warp=taper(0, cz, roof + .3, roof + .3 + comb, .8, .7))
        elif variant == "talud-tablero":
            count = max(3, int(round(2 * hx / 1.4)))
            for k in range(count):
                x = -hx * .9 + 2 * hx * .9 * k / (count - 1)
                for z in (back - hz * .9, back + hz * .9):
                    self.member("almena", x - .22, x + .22, roof + .3, roof + .85, z - .12, z + .12,
                                construction="carved roof merlon")


Pyramid.module = sys.modules[__name__]


def build_document(request):
    return Pyramid(request).compose()


def random_controls(product, rng):
    width = rng.randint(30, 44)
    return {"width": width, "height": round(width * rng.uniform(.55, .75)), "depth": width,
            "bays": None, "levels": 9, "detail": rng.choice(("concept", "standard"))}


def rule_checks(document, form):
    placements = document.placements
    faces = [p for p in placements if p.role in {"terrace_face", "talud"} and "terrace" in p.parameters]
    tiers = {}
    for p in faces:
        tiers.setdefault(int(p.parameters["terrace"]), []).append(p)
    spans = [extent(tiers[k])[1][0] - extent(tiers[k])[0][0] for k in sorted(tiers)]
    recede = len(spans) >= 3 and all(a > b + .05 for a, b in zip(spans, spans[1:]))
    checks = [("receding_terraces", recede, "Every terrace stands inside the one below.",
               "The terraces no longer recede.", [])]
    arches = [p.id for p in placements if p.kind == "arch"]
    lintels = [p for p in placements if p.role == "temple_lintel"]
    checks.append(("lintelled_openings", not arches and bool(lintels), "Openings are spanned by lintels.",
                   "An opening is arched or has lost its lintel.", arches))
    flights = {}
    for p in placements:
        if p.role == "stair_step":
            flights.setdefault(p.parameters.get("flight"), set()).add(p.parameters.get("step"))
    if len(flights) == 4:
        counts = sorted(len(steps) for steps in flights.values())
        checks.append(("calendar_stairs", counts == [91] * 4, "Four stairs of 91 steps and the platform count 365.",
                       "The radial stairs no longer count the year.", []))
    steps = [p for p in placements if p.role == "stair_step"]
    temple = [p for p in placements if p.role == "temple_wall"]
    if steps and temple:
        reach = max(p.position[1] + p.dimensions[1] / 2 for p in steps)
        floor = min(p.position[1] - p.dimensions[1] / 2 for p in temple)
        checks.append(("stair_reaches_temple", abs(reach - floor) < .05, "The stair arrives at the temple floor.",
                       "The stair stops short of the temple platform.", []))
    return checks
