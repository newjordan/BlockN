"""Art Deco architect: the setback skyscraper.

The form is a law before it is a style. New York's 1916 Zoning Resolution let a
building fill its lot up to a street-wall height. Above that, every part had to
stay under a sky-exposure plane leaning in from the lot line, except a tower
covering no more than a quarter of the lot, which could rise without limit. A
tall building on a full lot therefore steps back in tiers and ends in a slender
shaft. The architects of the 1920s dressed that envelope in unbroken vertical
piers with recessed spandrels and glass between them, and gave it a crown.

``spire-crown`` ends in diminishing metal tiers and a needle. ``stepped-crown``
ends in a ziggurat of clad tiers, a glazed lantern and a mast.

The model is a full-block site: the plane applies on all four lot lines. The
stone piers are cladding one tier high. Each stands on the podium or on a
concealed iron frame column, so no pier carries the tiers above it.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

from .historic import clamp, extent, hull
from .schema import ArchitectInfo
from .sonnet import Poem, controls_for, form

ARCHITECT = ArchitectInfo("artdeco", "1.0.0", "Art Deco Skyscraper Architect")
STYLE = "artdeco-setback"
PRODUCT = "artdeco-tower"
REGION = "New York City and the North American cities that copied its zoning"
PERIOD = "Setback skyscrapers under the 1916 Zoning Resolution, about 1923–1940"
VARIANTS = ("spire-crown", "stepped-crown")
KINDS = {
    "plinth": "foundation", "core_fill": "core", "plaza_paving": "slab",
    "pier": "column", "frame_column": "column", "mullion": "column",
    "spandrel": "panel", "glazing": "window", "parapet": "trim", "setback_ornament": "ornament",
    "setback_terrace": "slab", "shaft_cap": "slab",
    "portal_glazing": "window", "portal_surround": "trim", "portal_frame": "trim",
    "crown_tier": "roof", "crown_ornament": "ornament", "lantern_post": "column", "lantern_glazing": "window",
    "finial": "roof",
}
REQUIRED = {"plinth", "core_fill", "pier", "frame_column", "spandrel", "glazing", "parapet", "setback_terrace",
            "portal_glazing", "crown_tier", "finial"}
FORMS = {PRODUCT: form(
    PRODUCT, "Setback skyscraper under a sky-exposure plane: stepped tiers, a quarter-lot tower, unbroken piers and a crown",
    (36, 120, 36), VARIANTS, REQUIRED,
    lines={"ornament": ("chevron", "sunburst", "ziggurat"), "piers": ("single", "paired", "finned"),
           "setbacks": ("symmetric", "street-stepped"), "portal": ("triple-bay", "stepped-arch")},
    size={"width": (28, 44), "height": (84, 180), "depth": (28, 44)},
    ratios={"depth/width": (.8, 1.25), "height/width": (2.6, 4.6)},
    region=REGION, period=PERIOD,
    sky_exposure="one metre of setback for every four metres of rise above the street wall",
    tower_share="the tower covers at most a quarter of the lot and is exempt from the plane",
    site="a full block: the plane applies on all four lot lines")}
DEFAULT_SUBSTANCE = "limestone"
SUBSTANCES = {"plinth": "granite", "plaza_paving": "granite", "core_fill": "cast-concrete",
              "frame_column": "wrought-iron", "spandrel": "terracotta", "glazing": "window-glass",
              "setback_terrace": "cast-concrete", "shaft_cap": "cast-concrete",
              "portal_glazing": "window-glass", "portal_frame": "copper-sheet",
              "lantern_post": "wrought-iron", "lantern_glazing": "window-glass", "finial": "wrought-iron"}
TOL = 1e-4
SLOPE = 4.0         # metres of rise the plane allows for each metre of setback
PIER = .8           # a pier is square in plan
RECESS = .3         # spandrels and glass stand this far behind the pier face
SKIN = .2
PODIUM = .6
ROOF = .3
random_controls = controls_for(sys.modules[__name__])


def _odd(value, low):
    return max(low, 2 * round((value - 1) / 2) + 1)


def _spread(total, steps):
    """Cumulative setbacks, in bays, that reach ``total`` in ``steps`` steps of at least one bay."""
    return [round(total * j / steps) for j in range(steps + 1)]


class Tower(Poem):
    PALETTE = {"plinth": (96, 92, 90, 255), "core_fill": (150, 148, 142, 255), "plaza_paving": (120, 116, 112, 255),
               "pier": (214, 206, 188, 255), "mullion": (204, 196, 178, 255), "frame_column": (70, 72, 78, 255),
               "spandrel": (120, 86, 66, 255), "glazing": (48, 66, 78, 255), "parapet": (206, 198, 180, 255),
               "setback_ornament": (188, 150, 84, 255), "setback_terrace": (150, 146, 138, 255),
               "shaft_cap": (150, 146, 138, 255), "portal_glazing": (40, 54, 64, 255),
               "portal_surround": (220, 212, 194, 255), "portal_frame": (164, 122, 62, 255),
               "crown_ornament": (188, 150, 84, 255), "lantern_post": (70, 72, 78, 255),
               "lantern_glazing": (226, 206, 130, 255), "finial": (176, 180, 186, 255)}
    TIMBER_KINDS = frozenset()

    def compose(self):
        w, d, h = self.w, self.d, self.h
        bay = 4.0 * self.vary("bay", .1)
        self.nx, self.nz = _odd((w - PIER) / bay, 7), _odd((d - PIER) / bay, 7)
        self.xs = [-(w - PIER) / 2 + i * (w - PIER) / self.nx for i in range(self.nx + 1)]
        self.zs = [-(d - PIER) / 2 + i * (d - PIER) / self.nz for i in range(self.nz + 1)]
        mx, mz = self.xs[1] - self.xs[0], self.zs[1] - self.zs[0]
        # The tower keeps at most half of each side, so at most a quarter of the lot.
        side, deep = math.ceil(self.nx / 4), math.ceil(self.nz / 4)
        shift = 1 if self.line("setbacks") == "street-stepped" else 0     # the tower stands back from the street
        steps = min(side, deep - shift)
        sides, fronts, backs = _spread(side, steps), _spread(deep + shift, steps), _spread(deep - shift, steps)
        self.plans = list(zip(sides, fronts, backs))                      # bays set back, tier by tier
        self.storey = 3.7 * self.vary("storey", .04)
        self.crest = 2.6 if self.line("piers") == "finned" else 1.4       # what stands above a tier's roof
        crown = clamp(.17 * h * self.vary("crown", .1), 14.0, 34.0)
        top = math.floor((h - crown - PODIUM) / self.storey)
        counts = [max(5, round(.2 * h * self.vary("street-wall", .1) / self.storey))]
        for j, (s, f, b) in enumerate(self.plans[1:-1], 1):
            reach = min(s * mx, f * mz, b * mz) * (.92 + (self.vary(f"tier-{j}", .08) - 1)) - .02
            counts.append(min(counts[0] + math.floor(SLOPE * reach / self.storey), top - 4 * (steps - j)))
        counts.append(top)
        if any(b - a < 2 for a, b in zip(counts, counts[1:])):
            raise ValueError(f"{PRODUCT} is too low for its setbacks on this lot")
        self.levels_at = [PODIUM + n * self.storey for n in counts]       # roof level of each tier
        self.portal_bays = {(self.nx - 1) // 2 + k for k in ((-1, 0, 1) if self.line("portal") == "triple-bay" else (0,))}
        self.portal_storeys = 3 if self.line("portal") == "triple-bay" else 4
        self.podium("plinth", -w / 2, w / 2, -d / 2, d / 2, 0, PODIUM, paving="plaza_paving", thickness=.4)
        base = PODIUM
        for tier, roof in enumerate(self.levels_at):
            self.tier(tier, base, roof, counts[tier - 1] if tier else 0, counts[tier])
            base = roof
        self.portal()
        self.crown(self.levels_at[-1])
        (x0, x1), (z0, z1) = self.footprint(len(self.plans) - 1)
        return self.document({"region": REGION, "period": PERIOD, "tiers": len(self.plans),
                              "storeys": counts[-1], "storey_height": round(self.storey, 3),
                              "street_wall": round(self.levels_at[0] + self.crest, 3),
                              "tower_share_of_lot": round((x1 - x0) * (z1 - z0) / (w * d), 4),
                              "sky_exposure_slope": SLOPE,
                              "construction": "stone piers one tier high on a concealed iron frame; "
                                              "terracotta spandrels and glass between them"})

    # ------------------------------------------------------------ plan
    def lines(self, tier):
        s, f, b = self.plans[tier]
        return self.xs[s:self.nx - s + 1], self.zs[f:self.nz - b + 1]

    def footprint(self, tier):
        xs, zs = self.lines(tier)
        return (xs[0] - PIER / 2, xs[-1] + PIER / 2), (zs[0] - PIER / 2, zs[-1] + PIER / 2)

    def faces(self, tier):
        """(name, axis the face runs along, its pier lines, its outer plane, the way in)."""
        xs, zs = self.lines(tier)
        (x0, x1), (z0, z1) = self.footprint(tier)
        return (("front", "x", xs, z0, 1), ("back", "x", xs, z1, -1), ("left", "z", zs, x0, 1), ("right", "z", zs, x1, -1))

    def panel(self, role, axis, a0, a1, y0, y1, plane, inward, depth, thickness, **params):
        """A member in a face: ``depth`` behind the face plane, ``thickness`` thick."""
        c0, c1 = sorted((plane + inward * depth, plane + inward * (depth + thickness)))
        if axis == "x":
            return self.member(role, a0, a1, y0, y1, c0, c1, **params)
        return self.member(role, c0, c1, y0, y1, a0, a1, **params)

    # ------------------------------------------------------------ tiers
    def tier(self, tier, base, roof, first, last):
        xs, zs = self.lines(tier)
        last_tier = tier == len(self.plans) - 1
        for x in xs:
            for z in zs:
                if x in (xs[0], xs[-1]) or z in (zs[0], zs[-1]):
                    if tier:
                        self.member("frame_column", x - PIER / 2, x + PIER / 2, PODIUM, base,
                                    z - PIER / 2, z + PIER / 2, tier=tier, construction="riveted iron frame")
                    self.member("pier", x - PIER / 2, x + PIER / 2, base, roof + self.crest,
                                z - PIER / 2, z + PIER / 2, tier=tier, construction="stone cladding pier")
        paired = self.line("piers") == "paired"
        for face, axis, marks, plane, inward in self.faces(tier):
            for bay, (m0, m1) in enumerate(zip(marks, marks[1:])):
                a0, a1 = m0 + PIER / 2, m1 - PIER / 2
                portal = tier == 0 and face == "front" and bay in self.portal_bays
                for n in range(self.portal_storeys if portal else first, last):
                    y = PODIUM + n * self.storey
                    self.panel("spandrel", axis, a0, a1, y, y + 1.3, plane, inward, RECESS, SKIN,
                               tier=tier, face=face, bay=bay, storey=n)
                    self.panel("glazing", axis, a0, a1, y + 1.3, y + self.storey, plane, inward, RECESS, SKIN,
                               tier=tier, face=face, bay=bay, storey=n)
                self.panel("parapet", axis, a0, a1, roof, roof + 1.1, plane, inward, RECESS, SKIN,
                           tier=tier, face=face, bay=bay)
                self.ornament("setback_ornament", axis, (a0 + a1) / 2, (a1 - a0) * .5, roof + 1.1, .3,
                              plane, inward, RECESS, tier=tier, face=face, bay=bay)
                if paired and not portal:
                    mid = (a0 + a1) / 2
                    self.panel("mullion", axis, mid - .2, mid + .2, base, roof + 1.1, plane, inward, 0, RECESS,
                               tier=tier, face=face, bay=bay)
        (x0, x1), (z0, z1) = self.footprint(tier)
        inner = RECESS + SKIN
        if last_tier:
            self.member("shaft_cap", x0 + inner, x1 - inner, roof - ROOF, roof, z0 + inner, z1 - inner,
                        construction="cast concrete")
            return
        (u0, u1), (v0, v1) = self.footprint(tier + 1)       # the open terrace round the foot of the next tier
        for a0, a1, c0, c1 in ((x0 + inner, x1 - inner, z0 + inner, v0), (x0 + inner, x1 - inner, v1, z1 - inner),
                               (x0 + inner, u0, v0, v1), (u1, x1 - inner, v0, v1)):
            self.member("setback_terrace", a0, a1, roof - ROOF, roof, c0, c1, tier=tier, construction="cast concrete")

    def ornament(self, role, axis, centre, width, y, height, plane, inward, depth, **params):
        """The building's one motif, set on a parapet or a crown tier: chevron, sunburst or ziggurat."""
        motif = self.line("ornament")
        c0, c1 = sorted((plane + inward * depth, plane + inward * (depth + .15)))

        def gablet(a0, a1, rise):
            points = [(a, y, c) if axis == "x" else (c, y, a) for a in (a0, a1) for c in (c0, c1)]
            points += [((a0 + a1) / 2, y + rise, c) if axis == "x" else (c, y + rise, (a0 + a1) / 2) for c in (c0, c1)]
            self.mesh(role, hull(points), motif=motif, **params)
        if motif == "chevron":
            gablet(centre - width / 2, centre + width / 2, height)
        elif motif == "sunburst":
            for k, rise in enumerate((.55, 1.0, .55)):
                gablet(centre + (k - 1.5) * width / 3, centre + (k - .5) * width / 3, height * rise)
        else:
            for k in range(2):
                half = width / 2 * (1 - .45 * k)
                self.panel(role, axis, centre - half, centre + half, y + height * k / 2, y + height * (k + 1) / 2,
                           plane, inward, depth, .15, motif=motif, **params)

    # ------------------------------------------------------------ portal
    def portal(self):
        _, _, marks, plane, inward = self.faces(0)[0]
        head = PODIUM + self.portal_storeys * self.storey
        for bay in sorted(self.portal_bays):
            a0, a1 = marks[bay] + PIER / 2, marks[bay + 1] - PIER / 2
            clear = a1 - a0
            if self.line("portal") == "triple-bay":
                self.panel("portal_glazing", "x", a0, a1, PODIUM, head - 1.2, plane, inward, PIER, SKIN, bay=bay)
                self.panel("portal_surround", "x", a0, a1, head - 1.2, head, plane, inward, RECESS, PIER - RECESS + SKIN,
                           bay=bay, member="lintel")
                for x in (a0 + clear / 3, a0 + 2 * clear / 3):
                    self.panel("portal_frame", "x", x - .06, x + .06, PODIUM, head - 1.2, plane, inward, PIER - .1, .1,
                               bay=bay, member="mullion")
                continue
            # A stepped arch: corbels close the bay in three steps under a deep lintel.
            spring, step = PODIUM + 2.5 * self.storey, .4 * self.storey
            self.panel("portal_glazing", "x", a0, a1, PODIUM, head - .3 * self.storey, plane, inward, PIER, SKIN, bay=bay)
            self.panel("portal_surround", "x", a0, a1, head - .3 * self.storey, head, plane, inward, RECESS,
                       PIER - RECESS + SKIN, bay=bay, member="lintel")
            for q in range(3):
                reach = clear * (q + 1) / 9
                for b0, b1 in ((a0, a0 + reach), (a1 - reach, a1)):
                    self.panel("portal_surround", "x", b0, b1, spring + q * step, spring + (q + 1) * step, plane, inward,
                               RECESS, PIER - RECESS, bay=bay, member="corbel", step=q)
            self.panel("portal_frame", "x", (a0 + a1) / 2 - .06, (a0 + a1) / 2 + .06, PODIUM, spring, plane, inward,
                       PIER - .1, .1, bay=bay, member="mullion")

    # ------------------------------------------------------------ crown
    def box_tier(self, role, hx, hz, cz, y0, y1, index, plate=.15, **params):
        """A hollow tier: four plates and a lid, so the crown weighs what sheet and cladding weigh."""
        self.member(role, -hx, hx, y0, y1, cz - hz, cz - hz + plate, crown_tier=index, face="front", **params)
        self.member(role, -hx, hx, y0, y1, cz + hz - plate, cz + hz, crown_tier=index, face="back", **params)
        self.member(role, -hx, -hx + plate, y0, y1, cz - hz + plate, cz + hz - plate, crown_tier=index, face="left", **params)
        self.member(role, hx - plate, hx, y0, y1, cz - hz + plate, cz + hz - plate, crown_tier=index, face="right", **params)
        self.member(role, -hx + plate, hx - plate, y1 - .2, y1, cz - hz + plate, cz + hz - plate,
                    crown_tier=index, face="lid", **params)

    def crown(self, roof):
        (x0, x1), (z0, z1) = self.footprint(len(self.plans) - 1)
        cz, hx, hz = (z0 + z1) / 2, (x1 - x0) / 2 - 1.0, (z1 - z0) / 2 - 1.0
        rise = self.h - roof
        spire = self.variant == "spire-crown"
        tiers, shrink, share = (4, .78, .42) if spire else (3, .72, .5)
        substance = "copper-sheet" if spire else "limestone"
        y = roof
        for q in range(tiers):
            height = rise * share / tiers
            self.box_tier("crown_tier", hx, hz, cz, y, y + height, q, construction="hollow crown tier")
            for axis, half, plane, inward in (("x", hx, cz - hz, 1), ("x", hx, cz + hz, -1),
                                              ("z", hz, -hx, 1), ("z", hz, hx, -1)):
                self.ornament("crown_ornament", axis, 0 if axis == "x" else cz, half * 1.2, y + height, height * .55,
                              plane, inward, 0, crown_tier=q, face=(axis, inward))
            hx, hz, y = hx * shrink, hz * shrink, y + height
        for p in self.placements:
            if p.role in ("crown_tier", "crown_ornament"):
                p.substance = substance
                p.material = (112, 170, 150, 255) if spire else (222, 214, 196, 255)
        if spire:
            self.conic("finial", 0, cz, y, self.h, min(hx, hz) * .8, 0, sides=4, rot=math.pi / 4,
                       crown_tier=tiers, construction="needle")
            return
        # A glazed lantern on four posts, then the mast.
        top = y + rise * .26
        lx, lz, post = hx * .8, hz * .8, .3
        for sx in (-1, 1):
            for sz in (-1, 1):
                x, z = sx * (lx - post / 2), cz + sz * (lz - post / 2)
                self.member("lantern_post", x - post / 2, x + post / 2, y, top - .3, z - post / 2, z + post / 2, crown_tier=tiers)
        for sz in (-1, 1):
            z = cz + sz * (lz - post / 2)
            self.member("lantern_glazing", -lx + post, lx - post, y, top - .3, z - .05, z + .05, crown_tier=tiers)
        for sx in (-1, 1):
            x = sx * (lx - post / 2)
            self.member("lantern_glazing", x - .05, x + .05, y, top - .3, cz - lz + post, cz + lz - post, crown_tier=tiers)
        self.member("crown_tier", -lx, lx, top - .3, top, cz - lz, cz + lz, crown_tier=tiers, face="lantern-lid",
                    construction="hollow crown tier")
        self.placements[-1].substance = "copper-sheet"
        self.member("finial", -.14, .14, top, self.h, cz - .14, cz + .14, crown_tier=tiers + 1, construction="mast")


Tower.module = sys.modules[__name__]


def build_document(request):
    return Tower(request).compose()


# ---------------------------------------------------------------- rules
def _box(p):
    return [(p.position[a] - p.dimensions[a] / 2, p.position[a] + p.dimensions[a] / 2) for a in range(3)]


def _tiers(placements, role="pier"):
    groups = defaultdict(list)
    for p in placements:
        if p.role == role and isinstance(p.parameters.get("tier"), int):
            groups[p.parameters["tier"]].append(p)
    return [groups[k] for k in sorted(groups)]


def rule_checks(document, form):
    placements = document.placements
    tiers = _tiers(placements)
    boxes = [extent(tier) for tier in tiers]
    known = len(tiers) >= 2
    lot = boxes[0] if known else None
    tower = boxes[-1] if known else None
    wall = float(lot[1][1]) if known else 0.

    # 1. Above the street wall: under the plane, or inside the tower.
    over = []
    if known:
        for p in placements:
            (x0, x1), (_, y1), (z0, z1) = _box(p)
            if y1 <= wall + TOL:
                continue
            if x0 >= tower[0][0] - TOL and x1 <= tower[1][0] + TOL and z0 >= tower[0][2] - TOL and z1 <= tower[1][2] + TOL:
                continue
            setback = min(x0 - lot[0][0], lot[1][0] - x1, z0 - lot[0][2], lot[1][2] - z1)
            if y1 > wall + SLOPE * setback + .05:
                over.append(p.id)
    yield ("sky_exposure", known and not over,
           "Everything above the street wall stays under the sky-exposure plane or inside the tower.",
           "A member above the street wall breaks through the sky-exposure plane outside the tower.", over)

    share = 1.
    if known:
        share = float((tower[1][0] - tower[0][0]) * (tower[1][2] - tower[0][2])
                      / ((lot[1][0] - lot[0][0]) * (lot[1][2] - lot[0][2])))
    yield ("quarter_lot_tower", known and share <= .25 + 1e-6, "The tower covers no more than a quarter of the lot.",
           "The tower is missing or covers more than a quarter of the lot.",
           [p.id for p in tiers[-1]] if known and share > .25 + 1e-6 else [])

    # 3. Each tier stands on the roof of the one below, inside its edge.
    terraces = {k: extent(group)[1][1] for k, group in enumerate(_tiers(placements, "setback_terrace"))}
    loose = []
    for k, ((lo, hi), (lo2, hi2)) in enumerate(zip(boxes, boxes[1:]), 1):
        inside = all(lo2[a] > lo[a] + .5 and hi2[a] < hi[a] - .5 for a in (0, 2))
        seated = k - 1 in terraces and abs(lo2[1] - terraces[k - 1]) < .05 and hi2[1] > hi[1] + 1.
        if not (inside and seated):
            loose += [p.id for p in tiers[k]]
    yield ("receding_tiers", known and not loose, "Each tier stands on the roof of the tier below and inside its edge.",
           "A tier overhangs the tier below, does not rise above it, or does not start at its roof.", loose)

    # 4. One pier per bay line per tier, base to crest, on a support, on the bay lines of the tier below.
    broken = []
    supports = [p for p in placements if p.role in ("frame_column", "plaza_paving")]
    for k, tier in enumerate(tiers):
        lo, hi = boxes[k]
        below = None
        if k:
            lines = [{round(p.position[a], 3) for p in tiers[k - 1]} for a in (0, 2)]
            below = lines
        seen = set()
        for p in tier:
            (x0, x1), (y0, y1), (z0, z1) = _box(p)
            spot = (round(p.position[0], 3), round(p.position[2], 3))
            whole = abs(y0 - lo[1]) < .02 and abs(y1 - hi[1]) < .02 and spot not in seen
            seen.add(spot)
            seated = any(abs(_box(s)[1][1] - y0) < .02 and abs(s.position[0] - p.position[0]) < s.dimensions[0] / 2 + TOL
                         and abs(s.position[2] - p.position[2]) < s.dimensions[2] / 2 + TOL for s in supports)
            aligned = below is None or (spot[0] in below[0] and spot[1] in below[1])
            if not (whole and seated and aligned):
                broken.append(p.id)
    yield ("unbroken_piers", known and not broken,
           "Every pier runs unbroken through its tier, stands on a support, and keeps the bay lines of the tier below.",
           "A pier is cut short, unsupported, or off the bay lines of the tier below.", broken)

    # 5. The crown narrows tier by tier to a finial at the very top.
    crown = defaultdict(list)
    for p in placements:
        if isinstance(p.parameters.get("crown_tier"), int):
            crown[p.parameters["crown_tier"]].append(p)
    spans = [extent(crown[k]) for k in sorted(crown)]
    widths = [float(hi[0] - lo[0]) for lo, hi in spans]
    feet = [float(lo[1]) for lo, _ in spans]
    finials = [p for p in placements if p.role == "finial"]
    summit = max((_box(p)[1][1] for p in placements), default=0.)
    tapers = (len(spans) >= 3 and all(b < a - .05 for a, b in zip(widths, widths[1:]))
              and all(b > a for a, b in zip(feet, feet[1:])) and bool(finials)
              and abs(max(_box(p)[1][1] for p in finials) - summit) < TOL
              and all(p.parameters.get("crown_tier") == max(crown) for p in finials)
              and all(float(lo[a]) >= tower[0][a] - TOL and float(hi[a]) <= tower[1][a] + TOL
                      for lo, hi in spans for a in (0, 2)) if known else False)
    yield ("tapering_crown", bool(tapers), "The crown narrows tier by tier to a finial at the very top.",
           "The crown widens as it rises, leaves the tower, or does not end in a finial at the top.",
           [p.id for p in finials] if not tapers else [])
