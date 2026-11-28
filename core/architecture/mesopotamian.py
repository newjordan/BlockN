"""Mesopotamian architect: the ziggurat.

A ziggurat is a solid mountain of sun-dried brick in a jacket of baked brick.
It has no interior. Its rules are few and strict: the stages recede, every face
leans in and is ribbed with shallow buttresses, the only way up is by open
stairs on the front, and the one room stands on the summit.

``triple-stair`` follows the Ziggurat of Ur as Ur-Namma built it: one stair on
the axis and two laid against the front face, all three meeting at a gate on the
first stage. ``axial-stair`` keeps the single processional flight.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np

from .historic import Voids, box_mesh, clamp, extent, taper
from .schema import ArchitectInfo
from .sonnet import Poem, form

ARCHITECT = ArchitectInfo("mesopotamian", "1.0.0", "Mesopotamian Architect")
STYLE = "mesopotamian-ziggurat"
PRODUCT = "mesopotamian-ziggurat"
REGION = "Southern Mesopotamia (Ur, Uruk, Babylon; modern Iraq)"
PERIOD = "Ur III to Neo-Babylonian, about 2100–550 BCE"
VARIANTS = ("triple-stair", "axial-stair")
KINDS = {
    "core_fill": "core", "core_footing": "core", "stage_face": "wall", "buttress": "wall", "terrace_paving": "slab",
    "stair_step": "circulation", "stair_landing": "slab", "gate_wall": "wall", "gate_arch": "arch",
    "shrine_wall": "wall", "shrine_lintel": "entablature", "shrine_roof": "roof", "shrine_parapet": "wall",
}
REQUIRED = {"core_fill", "stage_face", "buttress", "terrace_paving", "stair_step", "shrine_wall", "shrine_roof"}
FORMS = {PRODUCT: form(
    PRODUCT, "Stepped ziggurat of buttressed baked brick over a mud-brick core, with open stairs and a summit shrine",
    (48, 22, 46), VARIANTS, REQUIRED,
    lines={"stages": ("even", "diminishing"), "buttresses": ("broad", "close", "paired"),
           "shrine": ("cella", "twin-cella"), "crown": ("stepped-merlons", "plain")},
    size={"width": (40, 64), "height": (18, 30), "depth": (38, 60)},
    ratios={"depth/width": (.8, 1.1), "height/width": (.36, .52)},
    controls={"levels": (3, 3, 5)},
    region=REGION, period=PERIOD, levels_meaning="receding stages",
    openings="one true arch at the stair gate; the shrine door is lintelled")}
DEFAULT_SUBSTANCE = "fired-clay-brick"
SUBSTANCES = {"core_fill": "mud-brick", "core_footing": "fired-clay-brick", "shrine_wall": "glazed-tile", "shrine_parapet": "glazed-tile",
              "shrine_lintel": "cedar", "shrine_roof": "earth-fill"}
TOL = 1e-4
PAVING = .3
PROUD = .3          # how far a buttress stands out from its face
MUD_DEPTH = 15.0    # sun-dried brick is not laid deeper than this below the summit


class Ziggurat(Poem):
    PALETTE = {"core": (150, 122, 92, 255), "wall": (188, 142, 98, 255), "buttress": (198, 152, 106, 255),
               "terrace_paving": (176, 136, 100, 255), "stair_step": (182, 140, 100, 255),
               "stair_landing": (176, 136, 100, 255), "arch": (170, 120, 84, 255),
               "shrine_wall": (58, 104, 150, 255), "shrine_parapet": (66, 116, 164, 255),
               "shrine_lintel": (96, 62, 40, 255), "shrine_roof": (160, 128, 96, 255)}
    T = .5

    def compose(self):
        levels, w, d = self.levels, self.w, self.d
        crown = .8 if self.line("crown") == "stepped-merlons" else .4
        shrine_h = clamp(.17 * self.h, 3.2, 5.5)
        weights = [1.0 if self.line("stages") == "even" else .8 ** k for k in range(levels)]
        rises = [(self.h - shrine_h) * weight / sum(weights) for weight in weights]
        run = rises[0] / (.74 * self.vary("stair", .06))
        batters = [rise * .11 * self.vary("batter", .15) for rise in rises]
        climbs = [rise / 1.05 + 1.2 for rise in rises[1:]]            # an upper flight and its landing
        spare_z = d - run - sum(climbs) - 2 * sum(batters) - 7.3
        spare_x = w - 2 * sum(batters) - 9.6
        side = clamp(.075 * w, 2.4, 4.2) * self.vary("setback", .12)
        side_z, side_x = min(side, spare_z / (levels - 1)), min(side, spare_x / (2 * (levels - 1)))
        if min(side_z, side_x) < 1.0:
            raise ValueError(f"{PRODUCT} has no room for {levels} stages on this plot")
        hx, za, zb, y = w / 2 - PROUD, -d / 2 + run, d / 2 - PROUD, 0.0      # buttresses stand proud of the face
        self.stages = []
        for k, (rise, lean) in enumerate(zip(rises, batters)):
            if k:
                hx, za, zb = hx - side_x, za + climbs[k - 1], zb - side_z
            self.stage(k, hx, za, zb, y, y + rise, lean)
            self.stages.append((hx, za, zb, y, y + rise, lean))
            hx, za, zb, y = hx - lean, za + lean, zb - lean, y + rise
        stair = clamp(.11 * w * self.vary("stair-width", .1), 3.6, 6.0)
        self.stairs(stair)
        self.shrine(hx, za, zb, y, shrine_h - crown, crown)
        return self.document({"region": REGION, "period": PERIOD, "stages": levels,
                              "stage_rises": [round(r, 3) for r in rises], "stairs": 3 if self.triple else 1,
                              "construction": "baked-brick facing over a sun-dried brick core"})

    @property
    def triple(self):
        return self.variant == "triple-stair"

    def stage(self, k, hx, za, zb, y0, y1, lean):
        cz, hz, top, t = (za + zb) / 2, (zb - za) / 2, y1 - PAVING, self.T
        warp = taper(0, cz, y0, y1, (hx - lean) / hx, (hz - lean) / hz)
        self.shell("stage_face", -hx, hx, za, zb, y0, top, t, warp=warp, grade="concept", terrace=k)
        # Sun-dried brick crushes under too much of itself, so the deepest lifts are baked brick.
        split = clamp(self.h - MUD_DEPTH, y0, top)
        for role, lo, hi in (("core_footing", y0, split), ("core_fill", split, top)):
            self.fill(-(hx - lean - t), hx - lean - t, lo, hi, za + lean + t, zb - lean - t, module=4.0,
                      role=role, terrace=k)
        self.slab("terrace_paving", -(hx - lean), hx - lean, za + lean, zb - lean, top, y1, module=3.0, terrace=k)
        # Shallow buttresses rib every face that is not behind a stair.
        spacing, width = {"broad": (5.2, 2.2), "close": (3.4, 1.3), "paired": (6.4, 1.1)}[self.line("buttresses")]
        offsets = (-.95, .95) if self.line("buttresses") == "paired" else (0.,)
        for axis, half, faces in (("x", hx, ((za, -1), (zb, 1))), ("z", hz, ((-hx, -1), (hx, 1)))):
            count = max(2, round(2 * (half - 1.2) / spacing))
            for cross, sign in faces:
                if axis == "x" and sign < 0 and k == 0 and self.triple:
                    continue                                    # the flank stairs lie against this face
                for i in range(count):
                    for offset in offsets:
                        along = (i + .5) * 2 * (half - 1.2) / count - (half - 1.2) + offset
                        if axis == "x" and sign < 0 and abs(along) < 4.2 + width / 2:
                            continue                            # the axial stair
                        centre = (along, (y0 + top) / 2, cross + sign * PROUD / 2) if axis == "x" else \
                            (cross + sign * PROUD / 2, (y0 + top) / 2, cz + along)
                        dims = (width, top - y0, PROUD) if axis == "x" else (PROUD, top - y0, width)
                        self.ashlar("buttress", centre, dims, axis=axis, grade="concept", warp=warp, terrace=k)

    def stairs(self, width):
        hx, za, zb, _, top, lean = self.stages[0]
        self.flight("stair_step", width, -self.d / 2, za, 0, top, flight="axial", terrace=0)
        self.member("stair_landing", -width / 2, width / 2, top - PAVING, top, za, za + lean + .4, terrace=0)
        for k, (_, foot, _, y0, y1, lean) in enumerate(self.stages[1:], 1):
            upper = width * .8
            self.flight("stair_step", upper, foot - (y1 - y0) / 1.05, foot, y0, y1, flight="axial", terrace=k)
            self.member("stair_landing", -upper / 2, upper / 2, y1 - PAVING, y1, foot, foot + lean + .4, terrace=k)
        # The gate where the stairs meet: a brick wall across the first stage with a true arch.
        half, ring = 1.2, .36
        outer = half + ring
        gate, span = top + 2.3 + outer + .5, min(hx - lean, width / 2 + 2.6)
        voids = Voids(top)
        voids.arch(0, half, top, top + 2.3, ring, "gate", role="gate_arch")
        self.run("gate_wall", "x", 0, za + lean + .4 + .45, 2 * span, top, gate, .9, voids=voids)
        if not self.triple:
            return
        flank = clamp(.065 * self.w, 2.4, 3.2)
        steps = math.ceil(top / .2)
        warp = taper(0, (za + zb) / 2, 0, top, (hx - lean) / hx, ((zb - za) / 2 - lean) / ((zb - za) / 2))
        for side in (-1, 1):
            foot, head = side * (hx - 1.0), side * (width / 2 + .1)
            for j in range(steps):
                xa, xb = sorted((foot + (head - foot) * j / steps, foot + (head - foot) * (j + 1) / steps))
                y = top * (j + 1) / steps
                mesh = box_mesh(((xa + xb) / 2, y / 2, za - flank / 2), (xb - xa, y, flank))
                mesh.vertices = warp(np.asarray(mesh.vertices))
                self.mesh("stair_step", mesh, step=j, flight="left" if side < 0 else "right", terrace=0,
                          construction="solid stair")

    def shrine(self, hx, za, zb, y, height, crown):
        twin = self.line("shrine") == "twin-cella"
        sx = min(hx * .62, 6.5 if twin else 4.6) * self.vary("shrine", .08)
        sz = min((zb - za) * .3, 3.6) * self.vary("shrine-depth", .08)
        cz = zb - (zb - za) * .12 - sz                              # the shrine stands toward the back
        wall, roof = y + height - .45, y + height
        doors = Voids(y)
        centres = (-sx / 2, sx / 2) if twin else (0.,)
        for i, x in enumerate(centres):
            doors.rect(x, 1.3, y, min(2.3, height - 1.5), f"shrine.{i}")
            doors.rect(x, 1.9, y + min(2.3, height - 1.5), .25, f"shrine.{i}.seat")
        self.shell("shrine_wall", -sx, sx, cz - sz, cz + sz, y, wall, .6, {"front": doors.specs})
        if twin:
            self.ashlar("shrine_wall", (0, (y + wall) / 2, cz), (.5, wall - y, 2 * sz - 1.2), axis="z", face="party")
        for x in centres:
            head = y + min(2.3, height - 1.5)
            self.member("shrine_lintel", x - .95, x + .95, head, head + .25, cz - sz, cz - sz + .6)
        self.slab("shrine_roof", -sx, sx, cz - sz, cz + sz, wall, roof, module=1.6)
        t = .45
        if self.line("crown") == "plain":
            self.shell("shrine_parapet", -sx, sx, cz - sz, cz + sz, roof, roof + crown, t)
            return
        # Stepped merlons: a low course, then merlons, each topped by a narrower block.
        self.shell("shrine_parapet", -sx, sx, cz - sz, cz + sz, roof, roof + .2, t)
        for axis, a0, a1, crosses in (("x", -sx, sx, (cz - sz + t / 2, cz + sz - t / 2)),
                                      ("z", cz - sz + t, cz + sz - t, (-sx + t / 2, sx - t / 2))):
            count = max(2, round((a1 - a0) / 1.5))
            pitch = (a1 - a0) / count
            for cross in crosses:
                for i in range(count):
                    mid = a0 + (i + .5) * pitch
                    for size, y0, y1 in ((pitch * .62, roof + .2, roof + .5), (pitch * .3, roof + .5, roof + crown)):
                        self.ashlar("shrine_parapet", (mid, (y0 + y1) / 2, cross) if axis == "x" else
                                    (cross, (y0 + y1) / 2, mid), (size, y1 - y0, t) if axis == "x" else
                                    (t, y1 - y0, size), axis=axis, merlon=True)


Ziggurat.module = sys.modules[__name__]


def build_document(request):
    return Ziggurat(request).compose()


def random_controls(product, rng):
    from .sonnet import roll
    controls = roll(FORMS[product], rng)
    controls["levels"] = rng.randint(3, 4 if controls["depth"] < 48 else 5)
    return controls


def _terraces(placements, role):
    groups = defaultdict(list)
    for p in placements:
        if p.role == role and isinstance(p.parameters.get("terrace"), int):
            groups[p.parameters["terrace"]].append(p)
    return [groups[k] for k in sorted(groups)]


def rule_checks(document, form):
    placements = document.placements
    stages = _terraces(placements, "stage_face")
    boxes = [extent(stage) for stage in stages]
    bad = []
    for k, ((lo, hi), (lo2, hi2)) in enumerate(zip(boxes, boxes[1:]), 1):
        inside = all(lo2[a] > lo[a] + .5 and hi2[a] < hi[a] - .5 for a in (0, 2))
        seated = abs(lo2[1] - hi[1] - PAVING) < .05
        if not (inside and seated):
            bad += [p.id for p in stages[k]]
    yield ("receding_stages", len(stages) >= 3 and not bad, "Each stage stands on the one below and inside its edge.",
           "A stage is missing, overhangs the stage below, or does not rest on it.", bad)

    plumb = []
    for stage, (lo, hi) in zip(stages, boxes):
        head = [p for p in stage if p.position[1] > hi[1] - .9]
        if not head or extent(head)[0][0] < lo[0] + .1 or extent(head)[1][0] > hi[0] - .1:
            plumb += [p.id for p in stage[:1]]
    yield ("battered_faces", bool(stages) and not plumb, "Every stage face leans inward as it rises.",
           "A stage face is plumb or leans outward.", plumb)

    steps = [p for p in placements if p.role == "stair_step" and p.parameters.get("flight") == "axial"]
    tops = sorted(p.position[1] + p.dimensions[1] / 2 for p in steps)
    summit = boxes[-1][1][1] + PAVING if boxes else 0.
    climbs = bool(tops) and tops[0] < .35 and abs(tops[-1] - summit) < .05 and \
        all(b - a < .35 for a, b in zip(tops, tops[1:]))
    yield ("processional_stair", climbs, "The axial stair climbs without a break from the ground to the summit.",
           "The axial stair is broken or stops short of the summit.", [])

    shrine = [p for p in placements if p.role == "shrine_wall"]
    crowned = False
    if shrine and boxes:
        (lo, hi), (top_lo, top_hi) = extent(shrine), boxes[-1]
        crowned = abs(lo[1] - summit) < .05 and all(lo[a] > top_lo[a] and hi[a] < top_hi[a] for a in (0, 2))
    yield ("summit_shrine", crowned, "The shrine stands on the highest stage, inside its edge.",
           "The shrine is missing or does not stand on the highest stage.", [p.id for p in shrine] if not crowned else [])
