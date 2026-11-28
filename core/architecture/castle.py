"""Norman castle architect: great towers, shell keeps and walled wards.

The forms follow English work of about 1070-1220. A great tower stands on a
battered plinth, is entered at the first floor through a forebuilding, is
stiffened by shallow pilasters and clasping corner turrets, and hides twin
roofs below its wall-walk either side of a spine wall. Curtain towers are
rectangular in Norman work and become drums around 1200.

The massing descends from the castle kit in angelX (keep, round tower, curtain
run, gatehouse), rebuilt here at full scale, stone by stone.
"""
from __future__ import annotations

import math
import sys

from .historic import (Composer, Form, TAU, Voids, batter, clamp, extent, hull, rotation, sector_mesh,
                       taper, unmirrored)
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("castle", "1.0.0", "Castle Architect")
STYLE = "norman"
KINDS = {
    "core_fill": "core",
    "plinth": "foundation", "motte": "foundation",
    "keep_wall": "wall", "inner_wall": "wall", "cross_wall": "wall", "corner_turret": "wall",
    "pilaster": "pier", "buttress": "pier",
    "window_arch": "arch", "door_arch": "arch", "gate_arch": "arch",
    "corbel": "entablature", "wall_walk": "slab", "parapet": "wall", "merlon": "wall",
    "keep_roof": "roof", "turret_roof": "roof", "tower_roof": "roof",
    "turret_deck": "slab", "tower_deck": "slab",
    "floor_joist": "beam", "floor_boards": "slab",
    "forebuilding": "wall", "stair": "circulation", "bridge": "circulation",
    "shell_wall": "wall", "curtain_wall": "wall", "mural_tower": "wall",
    "gate_tower": "wall", "gate_passage": "wall",
    "portcullis": "ornament", "mast": "ornament", "banner": "ornament",
}
FORMS = {form.name: form for form in (
    Form("norman-keep", "Great tower on a battered plinth with forebuilding, pilasters and corner turrets",
         (20, 30, 25), ("square", "round"),
         frozenset({"plinth", "keep_wall", "door_arch", "wall_walk", "parapet", "merlon", "keep_roof", "stair"}),
         {"bays": 3, "levels": 4},
         rules={"bays": [2, 4], "levels": [3, 5], "height_to_width": [0.9, 2.2], "depth_to_width": [0.9, 1.6],
                "entrance": "first floor, by an external stair"}),
    Form("norman-shell-keep", "Ring wall crowning an earth motte, entered through a gate tower",
         (46, 20, 46), ("shell",),
         frozenset({"motte", "shell_wall", "gate_tower", "gate_arch", "wall_walk", "parapet", "merlon", "stair"}),
         rules={"height_to_width": [0.3, 0.6], "depth_to_width": [0.85, 1.2], "motte_slope_degrees": 36.5}),
    Form("norman-castle", "Keep inside a curtain wall with a gate and mural towers",
         (48, 28, 60), ("square-towers", "drum-towers"),
         frozenset({"curtain_wall", "mural_tower", "gate_tower", "gate_arch", "keep_wall", "door_arch",
                    "wall_walk", "parapet", "merlon", "keep_roof"}),
         {"bays": 1},
         rules={"bays": [0, 3], "height_to_width": [0.4, 0.9], "depth_to_width": [1.0, 1.8],
                "bays_meaning": "mural towers between the corners of each long side"}),
)}
STONE_DARK = (150, 142, 128, 255)
PALETTE = {
    "core_fill": (132, 124, 110, 255), "plinth": STONE_DARK, "corbel": STONE_DARK, "motte": (96, 112, 72, 255),
    "roof": (86, 92, 104, 255), "portcullis": (60, 56, 52, 255), "mast": (120, 86, 58, 255),
    "bridge": (120, 86, 58, 255), "banner": (170, 30, 36, 255),
    "floor_joist": (120, 86, 58, 255), "floor_boards": (150, 112, 76, 255),
}
# Norman work: limestone ashlar facings over a flint-and-mortar rubble core,
# oak floors, stairs and drawbridges, lead on the boarded keep roofs, slated
# tower cones, and a shingled cedar cap on the clasping turrets.
SUBSTANCES = {
    "core_fill": "flint",
    "plinth": "limestone",
    "motte": "earth-fill",
    "keep_wall": "limestone", "inner_wall": "limestone", "cross_wall": "limestone",
    "corner_turret": "limestone",
    "pilaster": "limestone", "buttress": "limestone",
    "window_arch": "limestone", "door_arch": "limestone", "gate_arch": "limestone",
    "corbel": "limestone", "wall_walk": "limestone", "parapet": "limestone", "merlon": "limestone",
    "keep_roof": "lead-sheet", "turret_roof": "cedar", "tower_roof": "slate",
    "turret_deck": "limestone", "tower_deck": "limestone",
    "floor_joist": "oak", "floor_boards": "oak",
    "forebuilding": "limestone", "stair": "limestone", "bridge": "oak",
    "shell_wall": "limestone", "curtain_wall": "limestone", "mural_tower": "limestone",
    "gate_tower": "limestone", "gate_passage": "limestone",
    "portcullis": "oak", "mast": "oak", "banner": "painted-linen",
}
DEFAULT_SUBSTANCE = "limestone"
FRONT = -math.pi / 2          # the angle that faces -Z


class Castle(Composer):
    PALETTE = PALETTE
    T = 0.5                   # ashlar facing skin
    PARAPET = 0.45

    def compose(self):
        return {"norman-keep": self.keep_site, "norman-shell-keep": self.shell_site,
                "norman-castle": self.castle_site}[self.form.name]()

    # ------------------------------------------------------------- sites
    def keep_site(self):
        self.ratio(self.h / self.w, "height/width", .9, 2.2)
        self.ratio(self.d / self.w, "depth/width", .9, 1.6)
        levels, bays = self.control("levels", 3, 5), self.control("bays", 2, 4)
        if self.variant == "square":
            fore = clamp(.24 * self.d, 4.2, 8.0)
            rules = self.square_keep(0, fore / 2, self.w, self.d - fore, self.h, levels, bays, fore)
        else:
            radius = min(self.w / 2, .36 * self.d)
            if self.player is not None and self.player.height > 2.0:
                # As in the castle: a drum for a bigger body is a bigger drum,
                # so its stair can climb from the basement with the player's
                # headroom; it stays inside the plot.
                radius = min(self.w / 2, self.d / 2, radius * self.player.height / 1.8)
            rules = self.round_keep(radius, self.d, self.h, levels)
        return self.document({"typology": f"{self.variant} great tower", **rules})

    def shell_site(self):
        self.ratio(self.h / self.w, "height/width", .3, .6)
        self.ratio(self.d / self.w, "depth/width", .85, 1.2)
        t, base = self.T, min(self.w, self.d) / 2
        mound = clamp(.42 * self.h * self.vary("motte", .14), 3.5, 16.0)
        summit = base - 1.35 * mound
        if summit < 7.5:
            raise ValueError("norman-shell-keep needs a wider motte for this height")
        radius, thick, rise = summit - 1.0, 2.4 * self.vary("wall", .12), 3.0 * self.vary("gate_rise", .2)
        wall = self.h - mound - rise - 1.8
        top = mound + wall
        self.frustum("motte", 0, 0, 0, mound, base, summit, sides=32, rise=mound / 3,
                     construction="rammed earth motte")
        gate_w, gate_d = 5.6, thick + 2.4
        gap = [(FRONT, math.asin((gate_w / 2 - .05) / radius), 0, 1e9)]
        self.ring("shell_wall", 0, 0, radius, t, mound, top, gap, register="outer facing")
        self.ring("shell_wall", 0, 0, radius - thick + t, t, mound, top, gap, register="inner facing")
        for k in range(2):
            self.ring("core_fill", 0, 0, radius - t - k * (thick - 2 * t) / 2, (thick - 2 * t) / 2, mound, top - .3,
                      gap, grade="concept", core=True)
        self.walk_rings(0, 0, radius - t, radius - thick + t, top, gap)
        self.ring("parapet", 0, 0, radius, self.PARAPET, top, top + .9, gap)
        crenels = max(8, int(round(TAU * radius / 2.0)))
        self.ring("merlon", 0, 0, radius, self.PARAPET, top + .9, top + 1.8,
                  gap + [(FRONT + TAU * (k + .5) / crenels, .35 / radius, 0, 1e9) for k in range(crenels)])
        self.gate_tower(0, -(radius + .8), gate_w, gate_d, mound, top + rise, 1.1)
        steps = int(math.ceil(mound / .24))
        tread, riser = (base - summit) / steps, mound / steps
        for j in range(steps):
            self.member("stair", -1.3, 1.3, max(0.0, j * riser - .45), (j + 1) * riser,
                        -base + j * tread, -base + (j + 1) * tread + .05, construction="stone step in the motte")
        self.banner(0, -(radius + .8) + gate_d / 2, top + rise)
        return self.document({"typology": "shell keep on a motte", "motte_height": mound,
                              "motte_slope_degrees": 36.5, "shell_radius": radius, "wall_thickness": thick,
                              "wall_height": wall, "construction": "ashlar facings over a rubble core"})

    def castle_site(self):
        self.ratio(self.h / self.w, "height/width", .4, .9)
        self.ratio(self.d / self.w, "depth/width", 1.0, 1.8)
        bays = self.control("bays", 0, 3)
        drums = self.variant == "drum-towers"
        # Large fields of walling use the next coarser stone so the ward stays legible.
        coarse = {"concept": "concept", "standard": "concept", "presentation": "standard"}[self.detail]
        wall_h, thick = clamp(.28 * self.h * self.vary("curtain", .12), 6.0, 11.0), 2.2
        size = (2.9 if drums else 5.0) * self.vary("tower", .1)   # drum radius or square tower side
        out = size * 1.34 if drums else 2.5     # how far a tower (and its eave) stands proud
        a, b = self.w / 2 - out, self.d / 2 - out
        tower_h = wall_h + (3.6 if drums else 3.2)
        cut = size * .85 if drums else size - out

        gate_w, gate_d = 6.4, 6.5
        gate_front = -b - (1.0 if drums else 1.8)
        gate_top = wall_h + (3.4 if drums else 4.6)
        self.gate_tower(0, gate_front, gate_w, gate_d, 0, gate_top, 1.25, grade=coarse)
        towers = [(sx * (a if drums else a - size / 2 + out), sz * (b if drums else b - size / 2 + out))
                  for sx in (-1, 1) for sz in (-1, 1)]
        mids = [-b + (k + 1) * 2 * b / (bays + 1) for k in range(bays)]
        towers += [(sx * (a if drums else a - size / 2 + out), z) for sx in (-1, 1) for z in mids]
        if drums:
            towers += [(sx * (gate_w / 2 + size * .7), -b) for sx in (-1, 1)]
        for index, (x, z) in enumerate(towers):
            gate_drum = drums and index >= len(towers) - 2
            if drums:
                self.drum_tower(x, z, size, tower_h + (1.6 if gate_drum else 0), coarse, banner=not gate_drum)
            else:
                self.square_tower(x, z, size, tower_h, coarse)
        # Curtain runs stop at the towers that interrupt them.
        half_mid = size * .8 if drums else size / 2
        for sign in (-1, 1):
            stops = [-b + cut] + [v for z in mids for v in (z - half_mid, z + half_mid)] + [b - cut]
            for z0, z1 in zip(stops[::2], stops[1::2]):
                self.curtain("z", z0, z1, sign * a, sign, wall_h, thick, coarse)
            gate_edge = gate_w / 2 + (size * 1.5 if drums else 0)
            lo, hi = sorted((sign * gate_edge, sign * (a - cut)))
            self.curtain("x", lo, hi, -b, -1, wall_h, thick, coarse)
        self.curtain("x", -a + cut, a - cut, b, 1, wall_h, thick, coarse)
        keep = clamp(.36 * min(2 * a, 2 * b) * self.vary("keep", .1), 12.0, 20.0)
        if self.player is not None:
            # A keep for a bigger body is a bigger keep: its newel stairs need
            # a turn wide enough for the player's headroom. Scale it by their
            # height against a person's, and at least to the size those stairs
            # fit (a 21 m keep, measured), within the bailey.
            keep = min(max(keep * self.player.height / 1.8, 21.0 if self.player.height > 2.0 else keep),
                       2 * (a - thick - 3.0), (2 * b - 2 * thick - 6.0 - 4.2) / 1.1)
        fore = clamp(.26 * keep, 4.2, 6.0)
        centre = b - thick - 3.0 - (keep * 1.1 + fore) / 2
        start = self.grade
        self.grade = coarse
        aside = (self.vary("keep_place", 1) - 1) * (a - thick - 3.0 - keep / 2)     # keeps rarely sit on the axis
        rules = self.square_keep(aside * .6, centre + fore / 2, keep, keep * 1.1, self.h, 4, 2, fore, capped=drums)
        self.grade = start
        return self.document({"typology": "keep and bailey", "curtain_height": wall_h,
                              "curtain_thickness": thick, "mural_towers": len(towers),
                              "tower_plan": "drum" if drums else "rectangular", "keep": rules,
                              "construction": "ashlar facings over a rubble core"})

    # ------------------------------------------------------------- keeps
    def square_keep(self, cx, cz, bx, bz, top, levels, bays, fore, capped=False):
        start, t, q, corbel = len(self.placements), self.T, .35, .25
        p = clamp(.045 * min(bx, bz), .5, 1.1)
        hx, hz = bx / 2 - p, bz / 2 - p                     # turret faces
        thick = clamp(.17 * 2 * min(hx, hz) * self.vary("wall", .14), 2.0, 4.5)      # structural wall
        s = clamp(.19 * 2 * min(hx, hz) * self.vary("turret", .14), 2.6, thick + q)  # corner turret
        tt = q + t
        rise = clamp(.1 * top * self.vary("turret_rise", .3), 1.6, 4.5)
        yw = top - 1.8 - rise - (1.1 * s if capped else 1.5)
        yp, yt = clamp(.09 * yw * self.vary("plinth", .25), 1.2, 3.2), yw + 1.8 + rise
        usable = yw - 1.2
        weights = [1.5 * self.vary("hall", .15) if k == 2 else self.vary(f"storey{k}", .08) for k in range(levels)]
        floors = [usable * sum(weights[:k]) / sum(weights) for k in range(levels + 1)]
        f1 = floors[1]
        ix, iz = hx - q - thick, hz - q - thick             # clear interior
        fw = clamp(.42 * 2 * hx * self.vary("forebuilding", .15), 4.5, 9.0)
        yf = max(.62 * yw * self.vary("forebuilding_height", .1), f1 + 4.0)
        stair_w = self.passage(clamp(fore - p - 1.6, 1.2, 2.0))

        pitch = min(.75 * ix / 2, 2.4)
        ridge = yw - .6
        eave = ridge - pitch
        # Battered plinth: the foot spreads the load and throws off the ram.
        self.shell("plinth", -hx - p, hx + p, -hz - p, hz + p, 0, yp, t,
                   warp=taper(0, 0, 0, yp, hx / (hx + p), hz / (hz + p)))

        def lights(length, face):
            voids = Voids(yp)
            for bay in range(bays):
                centre = -length / 2 + (bay + .5) * length / bays
                for level in range(levels):
                    base, storey = floors[level], floors[level + 1] - floors[level]
                    oid = f"{face}.{level}.{bay}"
                    if face == "front" and abs(centre - (hx - fw / 2)) < fw / 2 + .7 and base < yf:
                        continue                         # behind the forebuilding
                    if level == 0:
                        sill = max(yp + .5, base + .5 * storey)
                        if sill + 1.3 < floors[1]:
                            voids.rect(centre, .22, sill, 1.0, oid)
                        continue
                    sill, tall = base + .38 * storey, clamp(.3 * storey, .8, 1.5)
                    if level == 2 and length / bays >= 3.0:
                        for side in (-1, 1):             # paired lights of the hall
                            voids.arch(centre + side * .48, .28, sill, sill + tall, .2,
                                       f"{oid}.{'ab'[side > 0]}", "window_arch")
                    else:
                        voids.arch(centre, .3, sill, sill + tall, .22, oid, "window_arch")
            return voids

        front = lights(2 * (hx - s), "front")
        half, spring = self.door(.8, f1, f1 + 1.7)
        reach = hx - s - 1.4 if self.player is None else hx - s - .6 - half
        front.arch(clamp(hx - fw / 2, 0, reach), half, f1, spring, .3, "keep.door", "door_arch")
        faces = (("front", "x", -1, hx, hz, front), ("back", "x", 1, hx, hz, None),
                 ("left", "z", -1, hz, hx, None), ("right", "z", 1, hz, hx, None))
        for face, axis, sign, half, depth, voids in faces:
            length, outer = 2 * (half - s), sign * (depth - q)
            voids = voids or lights(length, face)
            self.run("keep_wall", axis, 0, outer - sign * t / 2, length, yp, yw, t, voids, face=face)
            # Rubble core behind the facing, kept open behind every light.
            inner = sign * (depth - q - thick)
            for y0, y1, back in ((0, eave - .4, inner), (eave - .4, yw - .3, inner + sign * t)):
                lo, hi = sorted((outer - sign * t, back))
                box = (-length / 2, length / 2, y0, y1, lo, hi) if axis == "x" else (lo, hi, y0, y1, -length / 2, length / 2)
                self.core(*box, voids=voids.slots(0, axis), face=face)
            for bay in range(1, bays):
                width = clamp(.12 * length, 1.2, 2.0)
                self.run("pilaster", axis, -length / 2 + bay * length / bays, outer + sign * q / 2,
                         width, yp, yw, q, face=face)
            count = max(2, int(round(length / .8)))
            for k in range(count):
                along, cross = -length / 2 + (k + .5) * length / count, outer + sign * corbel / 2
                self.add("corbel", (along, yw - .175, cross) if axis == "x" else (cross, yw - .175, along),
                         (.3, .35, corbel) if axis == "x" else (corbel, .35, .3), face=face)
            cross = outer + sign * (corbel - self.PARAPET / 2)
            self.run("parapet", axis, 0, cross, length, yw, yw + .9, self.PARAPET, face=face)
            self.crenellate("merlon", axis, -length / 2, length / 2, cross, yw + .9, self.PARAPET, face=face)

        # Clasping corner turrets carry the stairs and rise clear of the parapet.
        for sx in (-1, 1):
            for sz in (-1, 1):
                loops = Voids(yp)
                for level in range(1, levels):
                    loops.rect(0, .22, floors[level] + 1.2, 1.0, f"turret.{level}")
                self.run("corner_turret", "x", sx * (hx - s / 2), sz * (hz - tt / 2), s, yp, yw, tt, loops)
                self.run("corner_turret", "z", sz * (hz - tt - (s - tt) / 2), sx * (hx - tt / 2),
                         s - tt, yp, yw, tt)
                x0, x1 = sorted((sx * (hx - s), sx * hx))
                z0, z1 = sorted((sz * (hz - s), sz * hz))
                openings = {}
                if self.player is not None:
                    # The turrets stand across the corners of the wall-walk; a
                    # door through each of their two inner faces carries the
                    # walk round the corner for the player.
                    clear = self.player.clearances()
                    tall = min(clear.door_height + .2, yt - yw - .5)
                    if tall >= self.player.height + .1:
                        wide = min(clear.door_width + .2, s - 2 * t - .2, hx - tt - (hx - q - thick) - .1)
                        walk_x = sx * ((hx - q - thick) + (hx - tt)) / 2 - (x0 + x1) / 2
                        walk_z = sz * ((hz - q - thick) + (hz - tt)) / 2 - (z0 + z1) / 2
                        openings = {("front" if sz > 0 else "back"): [
                                        {"id": f"turret.walk.{sx}.{sz}.z", "center": walk_x, "width": wide,
                                         "sill": 0.0, "height": tall}],
                                    ("left" if sx > 0 else "right"): [
                                        {"id": f"turret.walk.{sx}.{sz}.x", "center": walk_z, "width": wide,
                                         "sill": 0.0, "height": tall}]}
                self.shell("corner_turret", x0, x1, z0, z1, yw, yt, t, openings)
                self.ashlar("turret_deck", ((x0 + x1) / 2, yt - .125, (z0 + z1) / 2), (s - 2 * t, .25, s - 2 * t))
                if capped:
                    self.frustum("turret_roof", (x0 + x1) / 2, (z0 + z1) / 2, yt, yt + 1.1 * s, s * .72, 0,
                                 sides=8, rot=math.pi / 8, construction="shingled timber cap")
                else:
                    self.battlement(x0, x1, z0, z1, yt, .75, .9, .55)

        # Wall-walk on the wall head, and the inner skin that rises above the roofs.
        edge_x, edge_z = hx - tt, hz - tt
        for z0, z1 in ((-edge_z, -iz), (iz, edge_z)):
            self.ashlar("wall_walk", (0, yw - .15, (z0 + z1) / 2), (2 * edge_x, .3, z1 - z0))
        for x0, x1 in ((-edge_x, -ix), (ix, edge_x)):
            self.ashlar("wall_walk", ((x0 + x1) / 2, yw - .15, 0), (x1 - x0, .3, 2 * iz))
        self.shell("inner_wall", -ix - t, ix + t, -iz - t, iz + t, eave - .4, yw - .3, t)

        # Spine wall halves the floor spans; a countersunk roof sits either side of it.
        doors = Voids(0)
        for level in range(0 if self.player is not None else 1, levels):
            half, spring = self.door(.6, floors[level], floors[level] + 1.6)
            if self.player is None:
                doors.arch(0, half, floors[level], spring, .25, f"spine.{level}", "door_arch")
                continue
            # For a player the stair wells take the middle of each half, so a
            # door at the front and one at the back join the open floor either
            # side of them; the ground floor gets them too.
            edge = iz - half - .6
            doors.arch(-edge, half, floors[level], spring, .25, f"spine.{level}", "door_arch")
            doors.arch(edge, half, floors[level], spring, .25, f"spine.{level}.back", "door_arch")
        self.run("cross_wall", "z", 0, 0, 2 * iz, 0, eave - .16, .9, doors)
        # For a player, straight flights between the floors, along the back
        # wall, each in a stairwell trimmed out of the floor above.
        wells = self.inner_stairs(ix, iz, floors, levels, top=yw) if self.player is not None else {}
        # Timber floors: oak joists bear on the spine wall and in pockets in the outer wall.
        count = max(2, int(round(2 * iz / .9)))
        for level in range(1, levels):
            y = floors[level]
            for sign in (-1, 1):
                well = wells.get((level, sign), 0.0)
                z_lo, z_hi, x_well = -well / 2, well / 2, ix - well
                x0, x1 = sorted((sign * .45, sign * (ix + .3)))
                for k in range(count):
                    z = -iz + (k + .5) * 2 * iz / count
                    if well and z + .11 > z_lo and z - .11 < z_hi:
                        # Cut short at the well, onto a trimmer.
                        x0, x1 = sorted((sign * .45, sign * x_well))
                    self.member("floor_joist", x0, x1, y - .34, y - .06, z - .11, z + .11, level=level,
                                construction="oak joist")
                    x0, x1 = sorted((sign * .45, sign * (ix + .3)))
                if well:
                    # The trimmers that frame the well: two from the spine wall
                    # to the outer wall, and one across between them.
                    span = sorted((sign * .45, sign * (ix + .3)))
                    self.member("floor_joist", *span, y - .34, y - .06, z_lo - .22, z_lo, level=level,
                                construction="oak trimmer")
                    self.member("floor_joist", *span, y - .34, y - .06, z_hi, z_hi + .22, level=level,
                                construction="oak trimmer")
                    self.member("floor_joist", *sorted((sign * (x_well - .22), sign * x_well)), y - .34, y - .06,
                                z_lo, z_hi, level=level, construction="oak trimmer")
                    full = sorted((sign * .45, sign * ix))
                    self.slab("floor_boards", *full, -iz, z_lo, y - .06, y, 1.5, level=level,
                              construction="oak boards")
                    self.slab("floor_boards", *full, z_hi, iz, y - .06, y, 1.5, level=level,
                              construction="oak boards")
                    self.slab("floor_boards", *sorted((sign * .45, sign * x_well)), z_lo, z_hi, y - .06, y, 1.5,
                              level=level, construction="oak boards")
                    continue
                self.slab("floor_boards", *sorted((sign * .45, sign * ix)), -iz, iz, y - .06, y, 1.5, level=level,
                          construction="oak boards")
        for valley, crest in ((0, ix / 2), (ix, ix / 2), (0, -ix / 2), (-ix, -ix / 2)):
            def slope(vertices, valley=valley):
                out = vertices.copy()
                out[:, 1] += abs(vertices[:, 0] - valley) * pitch / (ix / 2)
                return out
            x0, x1 = sorted((valley, crest))
            sign = 1 if valley + crest > 0 else -1
            well = wells.get((levels, sign), 0.0)
            if not well:
                self.slab("keep_roof", x0, x1, -iz, iz, eave - .16, eave, warp=slope,
                          construction="lead on oak boarding, below the wall-walk")
                continue
            # The top flight comes up through the roof to the wall-walk: leave
            # its hatch open, the roof whole either side of it.
            hole = sorted((sign * (ix - well), sign * (ix + .3)))
            parts = [(x0, x1, -iz, -well / 2), (x0, x1, well / 2, iz)]
            parts += [(a, b, -well / 2, well / 2) for a, b in ((x0, min(x1, hole[0])), (max(x0, hole[1]), x1))]
            for a, b, z0, z1 in parts:
                if b - a > .3 and z1 - z0 > .3:
                    self.slab("keep_roof", a, b, z0, z1, eave - .16, eave, warp=slope,
                              construction="lead on oak boarding, below the wall-walk")

        if fore:
            self.forebuilding(hx, hz, fw, fore, p, q, yp, yf, f1, stair_w, s)
        self.banner(-(hx - s / 2), hz - s / 2, yt + (1.1 * s if capped else 0))
        if self.pick("handed", (False, True)):
            self.mirror(start)                          # the forebuilding may stand on either hand
        self.shift(start, cx, cz)
        return {"wall_thickness": round(thick, 3), "wall_walk_height": round(yw, 3),
                "plinth_height": round(yp, 3), "plinth_projection": round(p, 3),
                "entrance_height": round(f1, 3), "floors": [round(v, 3) for v in floors],
                "turret_side": round(s, 3), "pilaster_projection": q, "bays": bays,
                "roof": "twin countersunk roofs either side of a spine wall",
                "floor_structure": "oak joists from the outer wall to the spine wall",
                "joist_span": round(ix - .45, 3),
                "construction": "ashlar facings over a rubble core"}

    def inner_stairs(self, ix, iz, floors, levels, top=None):
        """A newel stair from each floor to the next, sized to the player.

        Keeps climbed by newel stairs in a turret, too tight for most game
        bodies. A sized keep gets its newel inside, against the outer wall of one
        half: risers no taller than the player's step, goings no shorter than
        their stride at the walking line, and a turn wide enough that the
        player's full height clears the flight above. It starts from open
        floor and lands at an edge of its well. Successive stairs take
        alternate halves, so no stair stands in the well of another.
        With ``top``, one more flight climbs from the top floor to that
        height, the wall-walk, and lands facing the outer wall so the player
        steps off onto the wall head.
        Returns ``{(level, side): size}``: the square well to leave open in
        that floor, its side on the outer wall, centred midway along it.
        """
        clear = self.player.clearances()
        width = self.passage(1.0)
        newel = .3
        # The tallest riser that still reads as one step once cut into cells
        # at the player's default pitch, where the step is a whole number of
        # cells.
        from .voxels import profile_pitch
        cell = profile_pitch(self.player)
        tallest = min(clear.stair_max_rise, math.floor(clear.stair_max_rise / cell + .5) * cell)
        wells = {}
        heights = list(floors[:levels]) + ([top] if top is not None else [])
        for level in range(len(heights) - 1):
            y0, y1 = heights[level], heights[level + 1]
            last = top is not None and level == len(heights) - 2
            # Risers a fifth under the step, so none reads taller than it in voxels.
            n = max(2, int(math.ceil((y1 - y0) / (.8 * clear.stair_max_rise) - 1e-9)))
            going = max(clear.stair_min_going, .28)
            # The walking line's radius, below: one turn must rise the body, a
            # slab, and the fall of the flight above across the body's width,
            # with two cells to spare for the voxels.
            most = (min(ix - .45, 2 * iz - 2 * clear.door_width) - .1 - width) / 2
            # The well sits against the outer wall, midway along it. The stair
            # fills the half from that wall to the spine, so the floor in front
            # of it and the floor behind it only meet round the spine side,
            # when the body fits there. Otherwise the stair must start and land
            # facing the front, where the entrance and the spine doors join
            # every floor. Try landings turned a little off square, shallower
            # steps, both ways round and a wider turn; take the first that fits.
            fewest = max(2, int(math.ceil((y1 - y0) / tallest - 1e-9)))
            counts = [n + extra for extra in range(4)] + list(range(n - 1, fewest - 1, -1))

            def faces_floor(angle, wide):
                c, s_ = math.cos(angle), math.sin(angle)
                return s_ < -.8 or (wide and (s_ > .8 or c < -.8))

            plan = None
            for off in (0.0, .15, -.15, .3, -.3, .45, -.45, .6, -.6):
                for steps in counts:
                    lift = (y1 - y0) / steps
                    deck = max(lift, .25)
                    turn = self.player.height + deck + lift * self.player.radius / going + .4
                    walk_r = max(newel + width / 2, turn * going / (2 * math.pi * lift))
                    while plan is None and walk_r <= most + 1e-9:
                        step = going / walk_r
                        wide = ix - .5 - (2 * walk_r + width + .1) >= 2 * self.player.radius + .2
                        ends = (off,) if last else (-math.pi / 2 + off, math.pi / 2 + off, math.pi + off)
                        for end in ends:
                            # The last flight lands facing the outer wall,
                            # where the wall-walk is; the others face floor.
                            if not (math.cos(end) > .8 if last else faces_floor(end, wide)):
                                continue
                            for turning in (1, -1):
                                if faces_floor(end - turning * steps * step, wide):
                                    plan = (steps, lift, walk_r, step, end, turning)
                                    break
                            if plan:
                                break
                        walk_r += .05
                    if plan:
                        break
                if plan:
                    break
            if plan is None:
                continue
            n, rise, walk_r, step, end, turning = plan
            slab = max(rise, .25)
            inner, outer = walk_r - width / 2, walk_r + width / 2
            size = 2 * outer + .1
            side = -1 if level % 2 == 0 else 1
            cx, cz = side * (ix - size / 2), 0.0
            for k in range(n):
                a = end - turning * (n - 1 - k) * step
                top = y0 + (k + 1) * rise
                a0, a1 = (a - step / 2, a + step / 2)
                if side < 0:
                    # Mirrored half: x flips, so the angle reflects about the z axis.
                    a0, a1 = math.pi - a1, math.pi - a0
                # The top step runs on under the edge of the floor above, so
                # no gap opens between them when both are cut into voxels.
                reach = outer
                if k == n - 1:
                    edge = (size / 2) / max(abs(math.cos(end)), abs(math.sin(end)))
                    reach = edge + .3
                self.mesh("stair", sector_mesh(cx, cz, max(inner, .05), reach, a0, a1,
                                               max(y0, top - slab), top, segments=3),
                          level=level, construction="stone newel step")
            self.frustum("stair", cx, cz, y0, y1, newel, newel, sides=8, construction="stone newel")
            wells[(level + 1, side)] = size
        return wells

    def drum_stair(self, inside, heights, whole=False, deck=None, walk=None):
        """A newel stair up a round keep, sized to the player, and a floor at
        each storey it reaches.

        The stair stands against the back of the drum, opposite the door, as
        one continuous helix with its newel through every storey, so no flight
        stands under another. Each floor is a crescent round it. Every flight
        starts and lands where that crescent is wide enough for the body, so
        a player steps off at each storey onto open floor. With ``whole``
        nothing is built unless the stair reaches the top floor. With ``deck``
        a last flight climbs on, with no floor at its head, to the wall-walk
        that rings the drum at that height from ``walk`` outwards; its steps
        are kept in ``self.roof_hatch`` so the roof can open over them.
        Returns the number of flights built.
        """
        clear = self.player.clearances()
        body = 2 * self.player.radius + .3
        width = self.passage(1.0)
        newel = .3
        from .voxels import profile_pitch
        cell = profile_pitch(self.player)
        tallest = min(clear.stair_max_rise, math.floor(clear.stair_max_rise / cell + .5) * cell)
        going = max(clear.stair_min_going, .28)

        def needed(lift):
            # The rise of one turn that clears the body over the flight below.
            return self.player.height + max(lift, .25) + lift * self.player.radius / going + .4

        self.roof_hatch = []
        floors = len(heights) - 1
        if deck is not None:
            heights = list(heights) + [deck]
        flights = []
        for y0, y1 in zip(heights, heights[1:]):
            n = max(2, int(math.ceil((y1 - y0) / (.8 * tallest) - 1e-9)))
            fewest = max(2, int(math.ceil((y1 - y0) / tallest - 1e-9)))
            flights.append((y0, y1, [n + e for e in range(4)] + list(range(n - 1, fewest - 1, -1))))
        if not flights:
            return 0
        walk_r = max([newel + width / 2] + [needed((y1 - y0) / counts[0]) * going / (TAU * (y1 - y0) / counts[0])
                                            for y0, y1, counts in flights])
        outer = walk_r + width / 2
        c = inside - outer - .05                    # the newel stands this far behind the middle
        if c < 0 or 2 * inside - 2 * outer - .05 < body:
            return 0
        cx, cz = 0.0, -c * math.sin(FRONT)

        def reach(angle):
            # From the newel to the drum's inner face, looking along ``angle``.
            u = angle - FRONT
            return c * math.cos(u) + math.sqrt(max(0.0, inside ** 2 - (c * math.sin(u)) ** 2))

        def open_floor(angle):
            return reach(angle) - outer >= body

        # Search every start the player can walk to and every flight that
        # lands on open floor, carrying each reachable landing angle (to a
        # twentieth of a radian) up a storey at a time; keep the first way up
        # that climbs furthest, preferring the shallowest steps.
        reached = {i: (.05 * i, .05 * i, []) for i in range(int(TAU / .05) + 1)}   # key -> (angle, start, plan)
        best = None

        def approach(start, step, lift):
            # A player walks to the first step along the walking line, under the
            # first turn wherever it clears their head, from open floor.
            under = TAU - (self.player.height + max(lift, .25) + .2) * step / lift
            return any(open_floor(start - .1 * i) for i in range(int(max(0.0, under) / .1) + 1))

        for flight, (y0, y1, counts) in enumerate(flights):
            onward = {}
            for angle, start, plan in reached.values():
                for steps in counts:
                    lift = (y1 - y0) / steps
                    widest = TAU * lift / needed(lift)      # the step angle the headroom allows
                    narrowest = going / walk_r              # the step angle the stride needs
                    k = 0
                    while narrowest + k * .01 <= widest + 1e-9:
                        step = narrowest + k * .01
                        k += 1
                        # The wall-walk rings the whole drum, so the last
                        # flight may land anywhere round it.
                        if flight < floors and not open_floor(angle + (steps - 1) * step):
                            continue
                        if flight == 0 and not approach(angle, step, lift):
                            continue
                        after = angle + steps * step
                        key = round((after % TAU) / .05)
                        if key not in onward:
                            onward[key] = (after, start, plan + [(y0, y1, steps, lift, step)])
            if not onward:
                break
            reached = onward
            best = min(reached.values(), key=lambda v: sum(-f[2] for f in v[2]))
        if best is None:
            return 0
        start, plan = best[1], best[2]
        if not plan or (whole and len(plan) < floors):
            return 0
        angle = start
        for level, (y0, y1, steps, lift, step) in enumerate(plan):
            slab = max(lift, .25)
            for k in range(steps):
                a = angle + k * step
                top = y0 + (k + 1) * lift
                rim = outer
                last = level >= floors
                if k == steps - 1:
                    # The top step runs on under the floor's edge, or out
                    # across the drum's inner face onto the wall-walk.
                    rim = max(outer, reach(a) + (walk - inside) + .3) if last else outer + .3
                piece = (cx, cz, max(walk_r - width / 2, .05), rim, a - step / 2, a + step / 2,
                         max(y0, top - slab), top)
                self.mesh("stair", sector_mesh(*piece, segments=3),
                          level=level + 1, construction="stone newel step")
                if last:
                    self.roof_hatch.append(piece)
            angle += steps * step
            if level >= floors:
                continue
            # The floor at the top of the flight: a crescent round the stair.
            bins = 48
            for j in range(bins):
                a0, a1 = TAU * j / bins, TAU * (j + 1) / bins
                far = min(reach(a) for a in (a0, (a0 + a1) / 2, a1)) + .1   # tucked into the wall, as boards are
                if far - outer < .4:
                    continue
                rings = max(1, int(math.ceil((far - outer) / 1.2)))
                for i in range(rings):
                    lo = outer + (far - outer) * i / rings
                    hi = outer + (far - outer) * (i + 1) / rings
                    self.mesh("floor_boards", sector_mesh(cx, cz, lo, hi, a0, a1, y1 - .1, y1),
                              level=level + 2, construction="oak boards")
        self.frustum("stair", cx, cz, plan[0][0], plan[-1][1], newel, newel, sides=8, construction="stone newel")
        return len(plan)

    def forebuilding(self, hx, hz, fw, fore, p, q, yp, yf, f1, stair_w, s=None):
        t, x0, x1, z0 = self.T, hx - fw, hx, -(hz + fore)
        self.shell("plinth", x0 - p, x1 + p, z0 - p, -hz, 0, yp, t, faces="flr",
                   warp=taper((x0 + x1) / 2, -hz, 0, yp, fw / (fw + 2 * p), fore / (fore + p)))
        chapel = Voids(yp)
        chapel.arch(0, .35, f1 + .45 * (yf - f1), f1 + .45 * (yf - f1) + 1.1, .22, "fore.light", "window_arch")
        self.run("forebuilding", "x", (x0 + x1) / 2, z0 + t / 2, fw, yp, yf, t, chapel, face="front")
        door_z = -(hz + p + stair_w / 2)
        left_z0, left_z1 = z0 + t, -(hz - q)
        entry = Voids(yp)
        half, spring = self.door(.7, f1, f1 + 1.5)
        entry.arch(door_z - (left_z0 + left_z1) / 2, half, f1, spring, .28, "fore.door", "door_arch")
        self.run("forebuilding", "z", (left_z0 + left_z1) / 2, x0 + t / 2, left_z1 - left_z0, yp, yf, t,
                 entry, face="left")
        self.run("forebuilding", "z", (z0 + t - hz) / 2, x1 - t / 2, fore - t, yp, yf, t, face="right")
        self.ashlar("tower_deck", ((x0 + x1) / 2, yf - .15, (z0 + t - hz) / 2), (fw - 2 * t, .3, fore - t))
        if self.player is not None:
            # A vestibule floor at the entrance level, so the stair, the
            # forebuilding door and the keep door join up for a walking player.
            self.ashlar("tower_deck", ((x0 + x1) / 2, f1 - .15, (z0 + t - hz) / 2), (fw - 2 * t, .3, fore - t),
                        register="vestibule floor")
            # The keep door's sill bridges the recess between the turrets.
            half = self.door(.8, f1, f1 + 1.7)[0]
            door_x = clamp(hx - fw / 2, 0, hx - s - .6 - half)
            self.ashlar("tower_deck", (door_x, f1 - .15, -(hz - q / 2)), (2 * half, .3, q),
                        register="door sill")
        self.battlement(x0, x1, z0, -hz, yf, .9, 1.3, .7, faces="flr")
        # The stair climbs along the keep to the forebuilding door at the first floor.
        steps = int(math.ceil(f1 / .2))
        tread = clamp((x0 + hx + p) / steps, .24, .3)
        for j in range(steps):
            xa = x0 - (steps - j) * tread
            self.ashlar("stair", (xa + tread / 2, (j + 1) * f1 / steps / 2, door_z),
                        (tread, (j + 1) * f1 / steps, stair_w))

    def round_keep(self, radius, depth, top, levels):
        start, t = len(self.placements), self.T
        p, proj = clamp(.09 * radius, .6, 1.4), clamp(.2 * radius * self.vary("buttress", .2), 1.2, 2.2)
        r = radius - p - proj
        thick = clamp(.3 * r * self.vary("wall", .12), 2.2, 4.6)
        rise = clamp(.08 * top * self.vary("turret_rise", .3), 1.5, 3.6)
        yw = top - 1.8 - rise
        yp = clamp(.12 * yw * self.vary("plinth", .25), 1.8, 4.0)
        usable = yw - 1.2
        floors = [usable * k / levels for k in range(levels + 1)]
        f1 = max(floors[1], yp + .4)
        beta = clamp(.26 * r * self.vary("buttress_width", .15), 1.1, 1.8) / r   # buttress half-angle
        spurs = [FRONT + math.pi / 6 + k * math.pi / 3 for k in range(6)]
        solid = [(a, beta, 0, 1e9) for a in spurs]

        self.ring("plinth", 0, 0, r + p, t, 0, yp, solid, top_radius=r)
        gaps = list(solid)
        for level in range(1, levels):
            for k in range(6):
                if (k + level) % 2 == 0 and not (k == 0 and level == 1):
                    gaps.append((FRONT + k * math.pi / 3, .14 / r, floors[level] + 1.1, floors[level] + 2.2))
        (half, spring), ring = self.door(.7, f1, f1 + 1.5), .28
        if self.player is not None:
            # The gap narrows through the drum's curved wall; widen it so the
            # door is still door-wide at the inner face.
            half, spring = self.door(half * r / (r - t), f1, spring)
        gaps += [(FRONT, half / r, f1, spring), (FRONT, (half + ring) / r, spring, spring + half + ring)]
        self.ring("keep_wall", 0, 0, r, t, yp, yw, gaps)
        open_slots = [g for g in gaps if g not in solid]
        layers = max(1, int(math.ceil((thick - t) / .8)))
        for k in range(layers):
            # For a player, the slots keep their width through the core instead
            # of narrowing with the radius, so the door passage stays door-wide.
            scale = 1.0 if self.player is None else r / (r - t - (k + 1) * (thick - t) / layers)
            self.ring("core_fill", 0, 0, r - t - k * (thick - t) / layers, (thick - t) / layers, 0, yw - .3,
                      [(a, h * scale + .1 / r, y0, y1) for a, h, y0, y1 in open_slots], grade="concept", core=True)
        self.arch_head("keep_wall", "x", 0, -(r - t / 2), t, half, spring, ring, "keep.door", "door_arch")
        if self.player is not None:
            # The drum is hollow behind its door; a boarded floor at the
            # entrance level gives a walking player somewhere to go, and a
            # newel stair against the back wall climbs to a floor at each
            # storey above it.
            inside = r - thick
            upper = [y for y in floors[2:levels] if y > f1 + 2.5]
            # Down to the basement too, when one stair can climb the whole drum.
            walk = r - thick + t
            if not self.drum_stair(inside, [0.0, f1] + upper, whole=True, deck=yw, walk=walk):
                self.member("floor_boards", -.35, .35, f1 - .1, f1, -.35, .35, construction="oak boards")
                rings = max(1, int(math.ceil((inside - .5) / 1.0)))
                for k in range(rings):
                    lo, hi = .5 + (inside - .5) * k / rings, .5 + (inside - .5) * (k + 1) / rings
                    count = max(6, int(math.ceil(TAU * hi / 2.0)))
                    for j in range(count):
                        self.mesh("floor_boards", sector_mesh(0, 0, lo, hi, TAU * j / count, TAU * (j + 1) / count,
                                                              f1 - .1, f1), construction="oak boards")
                self.drum_stair(inside, [f1] + upper, deck=yw, walk=walk)
        # Six wedge buttresses rise above the parapet as turrets.
        for a in spurs:
            self.ring("plinth", 0, 0, r + proj + p, t, 0, yp, arc=(a - beta, a + beta), top_radius=r + proj)
            self.ring("buttress", 0, 0, r + proj, t, yp, top, arc=(a - beta, a + beta))
            self.ring("buttress", 0, 0, r, t, yw, top, arc=(a - beta, a + beta), register="gorge")
            for side in (-1, 1):
                self.ashlar("buttress", (r - t + (proj + t) / 2, top / 2, 0), (proj + t, top, t),
                            axis="x", warp=rotation(a + side * (beta - t / 2 / r)), register="cheek")
            for k in range(2):
                self.mesh("tower_deck", sector_mesh(0, 0, r - t, r + proj, a - beta + k * beta,
                                                    a - beta + (k + 1) * beta, top - .3, top))
        self.walk_rings(0, 0, r - t, r - thick + t, yw, ())
        self.ring("parapet", 0, 0, r, self.PARAPET, yw, yw + .9, solid)
        crenels = []
        for k in range(6):
            free = math.pi / 3 - 2 * beta
            count = max(1, int(round(free * r / 2.0)))
            crenels += [(spurs[k] + beta + free * (i + .5) / count, .35 / r, 0, 1e9) for i in range(count)]
        self.ring("merlon", 0, 0, r, self.PARAPET, yw + .9, yw + 1.8, solid + crenels)
        ridge = yw - .6
        eave = ridge - min(.45 * (r - thick), 2.2)
        self.ring("inner_wall", 0, 0, r - thick + t, t, eave - .4, yw - .3)
        if self.player is not None and getattr(self, "roof_hatch", None):
            self.hatched_roof(r - thick, eave, ridge)
        else:
            self.frustum("keep_roof", 0, 0, eave, ridge, r - thick, 0, sides=12, rise=(ridge - eave) / 2,
                         construction="lead on oak boarding, below the wall-walk")
        # A straight stair stops short of the door; a timber bridge closes the gap.
        steps = int(math.ceil(f1 / .2))
        landing = -(r + p + 1.2)
        tread = clamp((depth - 2 * radius + proj - 1.2) / steps, .2, .3)     # the flight fits the plot
        for j in range(steps):
            self.ashlar("stair", (0, (j + 1) * f1 / steps / 2, landing - (steps - j - .5) * tread),
                        (self.passage(1.6), (j + 1) * f1 / steps, tread))
        deck = self.passage(1.6) / 2
        self.member("bridge", -deck, deck, f1 - .2, f1, landing, -(r - t), construction="oak drawbridge")
        self.banner((r + proj / 2) * math.cos(spurs[2]), (r + proj / 2) * math.sin(spurs[2]), top)
        low, high = extent(self.placements[start:])
        self.shift(start, 0, -(low[2] + high[2]) / 2)
        return {"wall_thickness": round(thick, 3), "wall_walk_height": round(yw, 3),
                "plinth_height": round(yp, 3), "entrance_height": round(f1, 3), "buttresses": 6,
                "floors": [round(v, 3) for v in floors], "roof": "low cone below the wall-walk",
                "construction": "ashlar facings over a rubble core"}

    def hatched_roof(self, edge, eave, ridge):
        """The round keep's low cone, laid in panels, with a hatch left open
        wherever a panel would stand over the last flight's steps within a
        body's height (and the body's half-width either side), so the stair
        climbs through the roof onto the wall-walk."""
        room, margin = self.player.height + .2, self.player.radius + .2

        def roof(rho):
            return ridge - (ridge - eave) * min(rho, edge) / edge

        samples = []
        for cx, cz, r0, r1, a0, a1, y0, y1 in self.roof_hatch:
            for i in range(6):
                for j in range(6):
                    ra, a = r0 + (r1 - r0) * i / 5, a0 + (a1 - a0) * j / 5
                    x, z = cx + ra * math.cos(a), cz + ra * math.sin(a)
                    if eave < y1 + room and roof(math.hypot(x, z)) > y1 - .1:
                        samples.append((x, z))
        host = self.host("keep_roof")
        bands = max(2, int(math.ceil(edge / .8)))
        for b in range(bands):
            ra, rb = edge * b / bands, edge * (b + 1) / bands
            count = max(6, int(math.ceil(TAU * rb / 1.0)))
            for j in range(count):
                a0, a1 = TAU * j / count, TAU * (j + 1) / count
                corners = [(rho * math.cos(a), rho * math.sin(a)) for rho in (ra, rb) for a in (a0, a1)]
                mx = sum(c[0] for c in corners) / 4
                mz = sum(c[1] for c in corners) / 4
                size = max(math.hypot(c[0] - mx, c[1] - mz) for c in corners)
                if any(math.hypot(x - mx, z - mz) < size + margin for x, z in samples):
                    continue
                points = [(x, y, z) for rho, (x, z) in zip((ra, ra, rb, rb), corners)
                          for y in (eave, roof(rho))]
                self.mesh("keep_roof", hull(points), course=b, host_id=host,
                          construction="lead on oak boarding, below the wall-walk")

    # ------------------------------------------------------------- parts
    def walk_rings(self, cx, cz, outer, inner, y, gaps):
        count = max(1, int(math.ceil((outer - inner) / 1.1 - 1e-9)))
        width = (outer - inner) / count
        for k in range(count):
            self.ring("wall_walk", cx, cz, outer - k * width, width, y - .3, y, gaps)

    def battlement(self, x0, x1, z0, z1, y, height, merlon, crenel, faces="fblr", grade=None):
        """Breast wall and merlons around a rectangular deck."""
        tp = self.PARAPET
        self.shell("parapet", x0, x1, z0, z1, y, y + height, tp, faces=faces, grade=grade)
        for key in faces:
            if key in "fb":
                cross = z0 + tp / 2 if key == "f" else z1 - tp / 2
                self.crenellate("merlon", "x", x0, x1, cross, y + height, tp, merlon, crenel, height, grade=grade)
            else:
                cross = x0 + tp / 2 if key == "l" else x1 - tp / 2
                self.crenellate("merlon", "z", z0 + tp, z1 - (tp if "b" in faces else 0), cross, y + height,
                                tp, merlon, crenel, height, grade=grade)

    def gate_tower(self, cx, front, width, depth, y0, top, half, grade=None):
        """Rectangular gate tower with a round-arched, barrel-vaulted passage along Z."""
        start, t, ring = len(self.placements), self.T, .32
        half, spring = self.door(half, y0, y0 + 2.3)
        x0, x1, back = -width / 2, width / 2, front + depth
        for z, face in ((front + t / 2, "front"), (back - t / 2, "back")):
            voids = Voids(y0)
            voids.arch(0, half, y0, spring, ring, f"gate.{face}", "gate_arch")
            if face == "front":
                sill = y0 + .62 * (top - y0)
                voids.arch(0, .3, sill, sill + 1.0, .22, "gate.chamber", "window_arch")
            self.run("gate_tower", "x", 0, z, width, y0, top, t, voids, grade, face=face)
        for sign in (-1, 1):
            self.run("gate_tower", "z", front + depth / 2, sign * (width / 2 - t / 2), depth - 2 * t,
                     y0, top, t, None, grade, face="left" if sign < 0 else "right")
            self.run("gate_passage", "z", front + depth / 2, sign * (half + t / 2), depth - 2 * t,
                     y0, spring, t, None, grade)
        bays = max(1, int(math.ceil((depth - 2 * t) / 1.2)))
        length = (depth - 2 * t) / bays
        for k in range(bays):
            self.arch_head("gate_passage", "x", 0, front + t + (k + .5) * length, length, half, spring, ring,
                           register="barrel vault")
        self.ashlar("tower_deck", (0, top - .15, front + depth / 2), (width - 2 * t, .3, depth - 2 * t), grade=grade)
        self.battlement(x0, x1, front, back, top, .9, 1.3, .7, grade=grade)
        # Portcullis hangs half raised in its slot behind the outer arch.
        z, foot = front + t + .25, y0 + 1.3
        if self.player is not None:
            # For a player it is drawn up clear of their door height.
            foot = min(max(foot, y0 + self.player.clearances().door_height), spring + half - .3)
        for k in range(7):
            x = -half + .1 + k * (2 * half - .2) / 6
            self.member("portcullis", x - .04, x + .04, foot, spring + half, z - .04, z + .04,
                        construction="oak shod with iron")
        for k in range(4):
            y = foot + .25 + k * .6
            if self.player is not None and y + .04 > spring + half - .05:
                break
            self.member("portcullis", -half, half, y - .04, y + .04, z - .04, z + .04,
                        construction="oak shod with iron")
        self.shift(start, cx, 0)

    def square_tower(self, cx, cz, side, top, grade):
        start, t, h = len(self.placements), self.T, side / 2
        foot, lean = 1.6, .45
        self.shell("plinth", -h - lean, h + lean, -h - lean, h + lean, 0, foot, t, grade=grade,
                   warp=taper(0, 0, 0, foot, h / (h + lean), h / (h + lean)))
        loops = {}
        for face in ("front", "back", "left", "right"):
            voids = Voids(foot)
            for y in (foot + .3 * (top - foot), foot + .68 * (top - foot)):
                voids.rect(0, .22, y, 1.1, f"loop.{face}.{int(y * 10)}")
            loops[face] = voids.specs
        self.shell("mural_tower", -h, h, -h, h, foot, top, t, loops, grade=grade)
        self.ashlar("tower_deck", (0, top - .15, 0), (side - 2 * t, .3, side - 2 * t), grade=grade)
        self.battlement(-h, h, -h, h, top, .9, 1.1, .6, grade=grade)
        self.shift(start, cx, cz)

    def drum_tower(self, cx, cz, radius, top, grade, banner=True):
        """Battered drum, corbelled eave and a steep cone that oversails the shaft."""
        t, foot = self.T, .22 * top
        turn = self.rng.uniform(0, math.pi / 4)
        gaps = [(turn + k * TAU / 3, .13 / radius, 1.4 + k * (top - 3.4) / 4, 2.5 + k * (top - 3.4) / 4)
                for k in range(5)]
        self.ring("plinth", cx, cz, radius * 1.12, t, 0, foot, top_radius=radius, grade=grade)
        self.ring("mural_tower", cx, cz, radius, t, foot, top, gaps, grade=grade)
        self.ring("corbel", cx, cz, radius * 1.14, radius * .14 + t, top, top + .35, grade=grade)
        eave, tip = radius * 1.34, top + .35 + 1.5 * radius + .4
        self.frustum("tower_roof", cx, cz, top + .35, tip, eave, 0, sides=12, rot=turn,
                     construction="slated timber cone")
        if banner and self.rng.random() < .55:
            self.banner(cx, cz, tip)

    def curtain(self, axis, a0, a1, face, sign, top, thick, grade):
        """One curtain run: battered outer facing, inner facing, wall-walk and parapet."""
        if a1 - a0 < .6:
            return
        t, along, length = self.T, (a0 + a1) / 2, a1 - a0
        lean = batter(2 if axis == "x" else 0, sign, .6, 1.8)
        self.run("curtain_wall", axis, along, face - sign * t / 2, length, 0, top, t, None, grade, lean,
                 register="outer facing")
        self.run("curtain_wall", axis, along, face - sign * (thick - t / 2), length, 0, top, t, None, grade,
                 register="inner facing")
        lo, hi = sorted((face - sign * t, face - sign * (thick - t)))
        self.core(*((a0, a1, 0, top - .3, lo, hi) if axis == "x" else (lo, hi, 0, top - .3, a0, a1)))
        centre = (along, top - .15, (lo + hi) / 2) if axis == "x" else ((lo + hi) / 2, top - .15, along)
        dims = (length, .3, hi - lo) if axis == "x" else (hi - lo, .3, length)
        self.ashlar("wall_walk", centre, dims, grade=grade)
        cross = face - sign * self.PARAPET / 2
        self.run("parapet", axis, along, cross, length, top, top + .9, self.PARAPET, None, grade)
        self.crenellate("merlon", axis, a0, a1, cross, top + .9, self.PARAPET, grade=grade)
        self.run("parapet", axis, along, face - sign * (thick - .2), length, top, top + .5, .4, None, grade,
                 register="inner breastwork")

    def banner(self, x, z, y):
        self.member("mast", x - .06, x + .06, y, y + 2.8, z - .06, z + .06, construction="ash pole")
        self.member("banner", x + .06, x + 1.16, y + 1.9, y + 2.7, z - .02, z + .02, construction="painted cloth")


Castle.module = sys.modules[__name__]


def build_document(request):
    return Castle(request).compose()


def random_controls(product, rng):
    width, height, depth = FORMS[product].defaults
    scale = rng.uniform(.9, 1.2)
    controls = {}
    if product == "norman-keep":
        controls = {"bays": rng.randint(2, 4), "levels": rng.randint(3, 5)}
    elif product == "norman-castle":
        controls = {"bays": rng.randint(0, 2)}
    return {"width": round(width * scale), "height": round(height * scale * rng.uniform(.95, 1.08)),
            "depth": round(depth * scale), "detail": rng.choice(("concept", "standard")), **controls}


def rule_checks(document, form):
    """Rules read from the stones themselves, not from metadata."""
    placements = document.placements
    by_role = {}
    for p in placements:
        by_role.setdefault(p.role, []).append(p)
    checks = []
    faces = {}
    for p in by_role.get("plinth", []):
        faces.setdefault(p.parameters.get("host_id"), []).append(p)
    if faces:
        plumb = []
        for stones in faces.values():
            # Each plinth face is measured on its own: foot course against head course.
            low, high = min(p.position[1] for p in stones), max(p.position[1] for p in stones)
            if high - low < 1e-6:
                continue
            foot, head = ([p for p in stones if abs(p.position[1] - y) < 1e-6] for y in (low, high))
            spread = extent(foot)[1] - extent(foot)[0] - (extent(head)[1] - extent(head)[0])
            if max(spread[0], spread[2]) < .02:
                plumb += [p.id for p in head]
        checks.append(("battered_plinth", not plumb, "The plinth spreads toward its foot.",
                       "The plinth has lost its batter.", plumb))
    doors = [p for p in by_role.get("door_arch", []) if str(p.parameters.get("opening_id", "")).endswith("door")]
    if doors:
        low = [p.id for p in doors if p.position[1] - p.dimensions[1] / 2 < 2.5]
        checks.append(("raised_entrance", not low, "The keep is entered at the first floor.",
                       "A keep doorway opens at ground level.", low))
    turrets = [p for p in by_role.get("corner_turret", []) if p.block_type == "CUBE"]
    if turrets:
        missing = unmirrored(turrets, {"corner_turret"},
                             centre=(extent(turrets)[0][0] + extent(turrets)[1][0]) / 2)
        checks.append(("paired_turrets", not missing, "Corner turrets balance about the keep's axis.",
                       "A corner turret has no twin.", missing))
    merlons, walk = by_role.get("merlon", []), by_role.get("wall_walk", [])
    if merlons and walk:
        cover = max(p.position[1] + p.dimensions[1] / 2 for p in merlons) - \
            max(p.position[1] + p.dimensions[1] / 2 for p in walk)
        checks.append(("guarded_wall_walk", cover >= 1.5, "Merlons cover a standing defender on the wall-walk.",
                       "The parapet no longer covers the wall-walk.", []))
    if form.name == "norman-shell-keep" and by_role.get("motte") and by_role.get("shell_wall"):
        summit = max(p.position[1] + p.dimensions[1] / 2 for p in by_role["motte"])
        low = [p.id for p in by_role["shell_wall"] if p.position[1] - p.dimensions[1] / 2 < summit - 1e-3]
        checks.append(("shell_crowns_motte", not low, "The shell wall stands on the motte's summit.",
                       "Shell wall stones sit below the summit.", low))
    if form.name == "norman-castle" and by_role.get("curtain_wall") and by_role.get("keep_wall"):
        commands = max(p.position[1] for p in by_role["keep_wall"]) > \
            1.3 * max(p.position[1] for p in by_role["curtain_wall"])
        checks.append(("keep_commands_ward", commands, "The keep overlooks its curtain.",
                       "The keep no longer rises above the curtain.", []))
    return checks
