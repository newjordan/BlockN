"""Timber pagoda architect: the Japanese multi-storey tower (tō).

A pagoda is a frame, not a wall. Each storey is a ring of twelve posts tied by
beams; bracket arms step outward from every post to carry eaves far deeper
than the body is wide. Storeys shrink as they rise, each standing on beams
laid across the frame below. Through the middle runs the central pillar
(shinbashira): it stands on its own foundation stone, passes every storey
without bearing on them, and carries the bronze finial (sōrin).

Timber members are long pieces, as timber is. Only the podium is coursed stone.
"""
from __future__ import annotations

import math
import sys

from .historic import Composer, Form, clamp, extent, hull
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("pagoda", "1.0.0", "Pagoda Architect")
STYLE = "japanese"
KINDS = {
    "core_fill": "core",
    "podium": "foundation", "podium_paving": "slab", "step": "circulation", "foundation_stone": "foundation",
    "post": "column", "inner_post": "column", "central_pillar": "column", "strut": "column",
    "sill_beam": "beam", "tie_beam": "beam", "head_plate": "beam", "eave_purlin": "beam", "cross_beam": "beam",
    "rafter": "beam", "bracket_block": "bracket", "bracket_arm": "bracket",
    "panel": "infill", "door": "infill", "lattice": "infill",
    "eave_roof": "roof", "balcony": "platform", "railing": "ornament", "sorin": "ornament",
}
FORMS = {form.name: form for form in (
    Form("japanese-pagoda", "Timber pagoda: diminishing storeys, bracketed deep eaves, free central pillar and finial",
         (15, 33, 15), ("five-storey", "three-storey"),
         frozenset({"podium", "post", "tie_beam", "bracket_arm", "eave_roof", "central_pillar", "cross_beam", "sorin"}),
         rules={"height_to_width": [1.3, 2.6], "depth_to_width": [0.95, 1.05],
                "width_meaning": "span of the lowest eaves", "plan": "three bays square",
                "structure": "post-and-beam timber frame with bracket sets; central pillar bears only the finial"}),
)}
TIMBER, PLASTER, TILE, BRONZE = (150, 62, 44, 255), (236, 228, 208, 255), (72, 78, 86, 255), (120, 150, 120, 255)
PALETTE = {"column": TIMBER, "beam": TIMBER, "bracket": TIMBER, "infill": PLASTER, "door": (110, 48, 36, 255),
           "lattice": (64, 104, 72, 255), "roof": TILE, "platform": TIMBER, "railing": TIMBER, "sorin": BRONZE,
           "central_pillar": (132, 96, 62, 255)}
# A pagoda is a cedar frame (hinoki posts, beams, brackets and doors) with
# lime-plaster infill panels, clay tiles on the eaves, a granite step and
# foundation stone on a coursed stone podium, and a bronze finial -- a copper
# alloy, so the substance registry records it as copper sheet.
SUBSTANCES = {
    "core_fill": "mud-brick",
    "podium": "limestone", "podium_paving": "limestone",
    "step": "granite", "foundation_stone": "granite",
    "post": "cedar", "inner_post": "cedar", "central_pillar": "cedar", "strut": "cedar",
    "sill_beam": "cedar", "tie_beam": "cedar", "head_plate": "cedar",
    "eave_purlin": "cedar", "cross_beam": "cedar", "rafter": "cedar",
    "bracket_block": "cedar", "bracket_arm": "cedar",
    "panel": "lime-plaster", "door": "cedar", "lattice": "cedar",
    "eave_roof": "clay-roof-tile",
    "balcony": "cedar", "railing": "cedar", "sorin": "copper-sheet",
}
DEFAULT_SUBSTANCE = "cedar"


class Pagoda(Composer):
    PALETTE = PALETTE

    def compose(self):
        self.ratio(self.h / self.w, "height/width", 1.3, 2.6)
        self.ratio(self.d / self.w, "depth/width", .95, 1.05)
        storeys = 5 if self.variant == "five-storey" else 3
        span = min(self.w, self.d)
        body = .44 * span                                   # first-storey frame, post to post
        body *= self.vary("body", .06)
        shrink = (.52 if storeys == 5 else .64) * self.vary("taper", .08)   # top storey as a share of the first
        hang = (span - body) / 2
        podium = clamp(.04 * self.h, .8, 1.6)
        finial = (.3 if storeys == 5 else .27) * self.h * self.vary("finial", .08)
        weights = [1.35 * self.vary("first_storey", .1)] + [1.0] * (storeys - 1)
        self.curve, self.lift = 1.55 * self.vary("roof_curve", .12), .16 * self.vary("corner_lift", .35)
        zone = self.h - podium - finial
        bases = [podium + zone * sum(weights[:k]) / sum(weights) for k in range(storeys + 1)]
        widths = [body * (1 - (1 - shrink) * k / (storeys - 1)) for k in range(storeys)]
        hangs = [hang * (1 - .22 * self.vary("eave_taper", .3) * k / (storeys - 1)) for k in range(storeys)]
        self.post_r = clamp(.03 * body, .14, .32)

        self.base(body / 2 + 1.7, podium)
        for k in range(storeys):
            above = widths[k + 1] / 2 if k + 1 < storeys else None
            self.storey(k, widths[k] / 2, hangs[k], bases[k], bases[k + 1], above)
        top = bases[-1]
        # The central pillar: spliced lengths, clear of every floor.
        joints = [podium + .3] + [bases[k] + .6 for k in sorted({storeys // 2, storeys - 1})] + [top + finial * .55]
        for index, (y0, y1) in enumerate(zip(joints, joints[1:])):
            self.add("central_pillar", (0, (y0 + y1) / 2, 0), (.7, y1 - y0, .7), "CYLINDER",
                     radius=.35 - .05 * index, height=y1 - y0, sections=12, splice=index,
                     construction="hinoki cypress log")
        self.sorin(top, finial)
        return self.document({"typology": f"{storeys}-storey pagoda", "storeys": storeys,
                              "body_widths": [round(v, 3) for v in widths],
                              "eave_overhangs": [round(v, 3) for v in hangs],
                              "eave_to_body": round(span / body, 2), "taper": shrink,
                              "finial_share": round(finial / self.h, 2), "bays": 3,
                              "structure": "post-and-beam frame; storeys stand on cross beams; "
                                           "central pillar is independent and carries the finial"})

    def base(self, half, top):
        self.shell("podium", -half, half, -half, half, 0, top, .5)
        self.core(-half + .5, half - .5, 0, top - .2, -half + .5, half - .5, construction_note="rammed earth and rubble")
        self.slab("podium_paving", -half + .5, half - .5, -half + .5, half - .5, top - .2, top, 1.2)
        steps = int(math.ceil(top / .2))
        for j in range(steps):
            self.member("step", -1.6, 1.6, 0, (j + 1) * top / steps, -half - (steps - j) * .32, -half - (steps - j - 1) * .32,
                        construction="granite step")
        self.member("foundation_stone", -.8, .8, top - .2, top + .3, -.8, .8, construction="shinso foundation stone")

    def ring(self, role, half, y0, y1, section, **params):
        """Four beams around a square, lapped at the corners."""
        s = section / 2
        for sign in (-1, 1):
            self.member(role, -half - s, half + s, y0, y1, sign * half - s, sign * half + s, **params)
            self.member(role, sign * half - s, sign * half + s, y0, y1, -half + s, half - s, **params)

    def storey(self, k, c, hang, y0, y1, above):
        r = self.post_r
        height = y1 - y0
        sill = y0 + .22
        head = sill + .52 * height                         # top of the posts
        plate = head + .62                                 # top of the bracket zone
        marks = [-c, -c / 3, c / 3, c]
        edge = [(x, z) for x in marks for z in marks if abs(x) == c or abs(z) == c]
        self.ring("sill_beam", c, y0, sill, 2 * r, storey=k)
        for x, z in edge:
            self.add("post", (x, (sill + head) / 2, z), (2 * r, head - sill, 2 * r), "CYLINDER",
                     radius=r, height=head - sill, sections=12, storey=k, construction="hinoki post")
        self.ring("tie_beam", c, head - .34, head - .04, r * 1.2, storey=k, construction="head tie through the posts")
        if k == 0:
            for x in (-c / 3, c / 3):
                for z in (-c / 3, c / 3):
                    self.add("inner_post", (x, (y0 + plate) / 2, z), (2.4 * r, plate - y0, 2.4 * r), "CYLINDER",
                             radius=1.2 * r, height=plate - y0, sections=12, storey=k)
        # Infill between posts: doors on the axes, plaster and lattice beside them.
        start = len(self.placements)
        for bay, (a, b) in enumerate(zip(marks, marks[1:])):
            x0, x1 = a + r, b - r
            role = "door" if bay == 1 else "panel"
            self.member(role, x0, x1, sill, head - .34, -c - .04, -c + .04, storey=k)
            if bay != 1 and k == 0:
                bars = 7
                for j in range(bars):
                    x = x0 + (x1 - x0) * (j + .5) / bars
                    self.member("lattice", x - .035, x + .035, sill + .9, head - .9, -c - .07, -c - .04, storey=k)
        # Bracket sets: a bearing block, an arm along the wall and an arm stepping out.
        reach = hang * .36
        for x in marks:
            if x != c:                                      # each side owns its left corner
                self.member("bracket_block", x - 1.2 * r, x + 1.2 * r, head, head + .22, -c - 1.2 * r, -c + 1.2 * r, storey=k)
            for y, out in ((head + .22, reach * .55), (head + .42, reach)):
                self.member("bracket_arm", x - .8 * r, x + .8 * r, y, y + .2, -c - out, -c + .5 * r, storey=k,
                            construction="stepped bracket arm")
            if abs(x) != c:
                self.member("bracket_arm", x - 2.6 * r, x + 2.6 * r, head + .22, head + .42, -c - .7 * r, -c + .7 * r,
                            storey=k, construction="wall bracket arm")
        rise = y1 - (head + .3)
        inner = (above + .12) if above else .55
        eave = c + hang
        count = int(2 * eave / (.9 if self.detail == "standard" else .5)) if self.detail != "concept" else 0
        for j in range(count):
            x = -eave + .2 + (2 * eave - .4) * j / max(1, count - 1)
            if abs(x) > c + reach:
                continue                                    # corner rafters fan from the hip instead
            # Rafters follow the sag of the roof in three scarfed lengths.
            reach_u = (eave - c) / (eave - inner)
            for part in range(3):
                pts = []
                for u in (reach_u * part / 3, reach_u * (part + 1) / 3):
                    half = eave + (inner - eave) * u
                    y = self.roof_y(u, x / eave, head + .3, rise, hang) - .15
                    pts += [(x * half / eave + sx * .05, y + dy, -half) for sx in (-1, 1) for dy in (0, -.1)]
                self.mesh("rafter", hull(pts), storey=k)
        self.roof_side(k, eave, inner, head + .3, rise, hang)
        self.spin(start)
        self.ring("head_plate", c, plate - .16, plate, 1.6 * r, storey=k)
        self.ring("eave_purlin", c + reach, plate - .16, plate, 1.4 * r, storey=k)
        if above:
            # Beams across the frame carry the storey above on short struts.
            for sign in (-1, 1):
                self.member("cross_beam", -c, c, plate, plate + .26, sign * above - .13, sign * above + .13, storey=k)
                self.member("cross_beam", sign * above - .13, sign * above + .13, plate, plate + .26, -c, c, storey=k)
            upper = [-above, -above / 3, above / 3, above]
            for x in upper:
                for z in upper:
                    if abs(x) == above or abs(z) == above:
                        self.member("strut", x - .8 * r, x + .8 * r, plate + .26, y1, z - .8 * r, z + .8 * r, storey=k)
        if k:
            deck = c + .75
            for sign in (-1, 1):
                self.member("balcony", -deck, deck, y0 + .12, y0 + .2, *sorted((sign * (c + r), sign * deck)), storey=k)
                self.member("balcony", *sorted((sign * (c + r), sign * deck)), y0 + .12, y0 + .2, -c - r, c + r, storey=k)
            self.ring("railing", deck - .06, y0 + .82, y0 + .9, .08, storey=k)
            for x in (-deck + .06, 0, deck - .06):
                for z in (-deck + .06, 0, deck - .06):
                    if abs(x) > 1 or abs(z) > 1:
                        self.member("railing", x - .04, x + .04, y0 + .2, y0 + .82, z - .04, z + .04, storey=k)

    def roof_y(self, u, s, eave_y, rise, hang):
        """Concave slope from eave (u=0) to body (u=1); corners (|s|=1) lift."""
        return eave_y + rise * u ** self.curve + self.lift * hang * abs(s) ** 3 * (1 - u) ** 2

    def roof_side(self, k, eave, inner, eave_y, rise, hang):
        """One of four roof slopes, facing -Z, as courses of tile."""
        rings = 5
        cuts = max(6, int(math.ceil(2 * eave / 1.3)) // 2 * 2)
        for i in range(rings):
            for j in range(cuts):
                pts = []
                for u in (i / rings, (i + 1) / rings):
                    half = eave + (inner - eave) * u
                    for s in (-1 + 2 * j / cuts, -1 + 2 * (j + 1) / cuts):
                        y = self.roof_y(u, s, eave_y, rise, hang)
                        pts += [(s * half, y, -half), (s * half, y - .14, -half)]
                self.mesh("eave_roof", hull(pts), storey=k, course=i, construction="tile on boarded rafters")

    def sorin(self, base, height):
        """Bronze finial: dew basin, bowl, nine rings, water-flame and jewel."""
        self.member("sorin", -.75, .75, base - .05, base + .3, -.75, .75, part="roban")
        y = base + .3
        self.lathe("sorin", 0, 0, [(y, .62), (y + .25, .5), (y + .42, .2)], sides=12, part="fukubachi")
        self.lathe("sorin", 0, 0, [(y + .42, .2), (y + .5, .5), (y + .58, .2)], sides=12, part="ukebana")
        mast_top = base + height
        self.add("sorin", (0, (y + .58 + mast_top - .5) / 2, 0), (.2, mast_top - .5 - y - .58, .2), "CYLINDER",
                 radius=.1, height=mast_top - .5 - y - .58, sections=8, part="mast")
        rings0, rings1 = y + .8, base + height * .68
        for n in range(9):
            centre = rings0 + (rings1 - rings0) * n / 8
            radius = .56 - .03 * n
            self.drum("sorin", 0, 0, centre - .05, centre + .05, radius, radius, 12, part="kurin", ring=n)
        flame = base + height * .72
        for sign in (-1, 1):
            pts = [(sign * a, flame + b, c) for a, b in ((.1, 0), (.5, height * .06), (.1, height * .15))
                   for c in (-.02, .02)] + [(sign * .1, flame + height * .15, 0)]
            self.mesh("sorin", hull(pts), part="suien")
        self.lathe("sorin", 0, 0, [(mast_top - .5, .1), (mast_top - .3, .24), (mast_top, .03)], sides=10, part="hoju")


Pagoda.module = sys.modules[__name__]
Pagoda.TIMBER_KINDS = frozenset({"beam", "bracket", "column", "infill", "platform"})


def build_document(request):
    return Pagoda(request).compose()


def random_controls(product, rng):
    width = rng.randint(12, 18)
    return {"width": width, "height": round(width * rng.uniform(1.9, 2.4)), "depth": width,
            "bays": None, "levels": None, "detail": rng.choice(("concept", "standard"))}


def rule_checks(document, form):
    placements = document.placements
    tiers, frames = {}, {}
    for p in placements:
        if p.role == "eave_roof":
            tiers.setdefault(int(p.parameters.get("storey", -1)), []).append(p)
        if p.role == "post":
            frames.setdefault(int(p.parameters.get("storey", -1)), []).append(p)
    spans = [extent(tiers[k])[1][0] - extent(tiers[k])[0][0] for k in sorted(tiers)]
    shrinking = len(spans) in (3, 5) and all(a > b + .05 for a, b in zip(spans, spans[1:]))
    checks = [("diminishing_storeys", shrinking, "An odd number of roofs, each narrower than the one below.",
               "The roofs no longer diminish in an odd-numbered series.", [])]
    if spans and 0 in frames:
        body = extent(frames[0])[1][0] - extent(frames[0])[0][0]
        checks.append(("deep_eaves", spans[0] >= 1.8 * body, "The lowest eaves span nearly twice the body.",
                       "The eaves are too shallow for a bracketed pagoda.", []))
    pillar = sorted((p for p in placements if p.role == "central_pillar"), key=lambda p: p.position[1])
    roofs = [p for p in placements if p.role == "eave_roof"]
    whole = bool(pillar) and all(abs((a.position[1] + a.dimensions[1] / 2) - (b.position[1] - b.dimensions[1] / 2)) < 1e-3
                                 for a, b in zip(pillar, pillar[1:]))
    if pillar and roofs:
        whole = whole and pillar[-1].position[1] + pillar[-1].dimensions[1] / 2 > extent(roofs)[1][1] \
            and all(abs(p.position[0]) < 1e-6 and abs(p.position[2]) < 1e-6 for p in pillar)
    checks.append(("continuous_central_pillar", whole, "The central pillar runs unbroken from its stone to the finial.",
                   "The central pillar is broken, off axis, or stops below the top roof.", [p.id for p in pillar] if not whole else []))
    posts = {}
    for k, members in frames.items():
        posts[k] = len(members)
    checks.append(("three_bay_frames", bool(posts) and set(posts.values()) == {12}, "Every storey is a ring of twelve posts.",
                   "A storey has lost a post.", []))
    return checks
