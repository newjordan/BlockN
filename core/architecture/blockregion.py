"""Regional town layouts: towns laid out the way their own tradition lays
them out, not on the English street-and-plot plan.

First: a Pueblo, after Taos and Acoma. Room blocks are built contiguous,
party wall to party wall, in rows round a plaza, each stepping back storey by
storey from the plaza so every roof is the terrace of the rooms above. A
second ring of blocks stands outside a lane that runs round the first, facing
it. The plaza holds a kiva (a round ceremonial chamber, sunk in the ground and
entered by a ladder through its roof) and bread ovens; the fields lie outside,
maize, beans and squash. One passage leads in from the south.

The layout scores against the same town rules as the English town
(street enclosure, frontage, square enclosure, size and rim) through
``blockrules.score_town``, and is walked from its entrance by
``region_walk``.
"""
from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import blockstyles
from .blockmode import BlockBuild, state
from .blocktown import FIELD, FREE, GREEN, HOUSE, PAD, PLAZA, STREET, _Town, turned

Cell = Tuple[int, int]
_FACING = {(0, -1): "north", (0, 1): "south", (1, 0): "east", (-1, 0): "west"}


def _pool(player: Any, seed: int, want_depth: int = 16) -> List[BlockBuild]:
    """Pueblo room blocks of one depth (three storeys), so the ring's back
    line stays straight against the lane."""
    out: List[BlockBuild] = []
    k = 0
    while len(out) < 12 and k < 200:
        p = blockstyles.pueblo_terrace(player, seed * 1000 + 300 + k)
        base = {(x, z) for (x, y, z) in p.blocks if y == 0}
        if max(z for _, z in base) - min(z for _, z in base) + 1 == want_depth:
            out.append(p)
        k += 1
    return out


class _Pueblo:
    def __init__(self, b: BlockBuild, t: _Town, pool: List[BlockBuild], rng: random.Random):
        self.b, self.t, self.pool, self.rng = b, t, pool, rng
        self.turns: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def variant(self, i: int, facing: Cell) -> Dict[str, Any]:
        key = (i, facing)
        if key not in self.turns:
            for q in range(4):
                tb = turned(self.pool[i], q)
                f = tb.ports[0]["facing"]
                if (int(round(f[0])), int(round(f[2]))) == facing:
                    break
            d = tb.ports[0]
            dc = (int(math.floor(d["position"][0])), int(math.floor(d["position"][2])))
            base = {(x, z) for (x, y, z) in tb.blocks if y == 0}
            self.turns[key] = {"t": tb, "stair": (dc[0] + facing[0], dc[1] + facing[1]), "base": base,
                               "w": max(x for x, _ in base) + 1, "d": max(z for _, z in base) + 1}
        return self.turns[key]

    def place(self, facing: Cell, along: int, line: int, P: int) -> bool:
        """One block facing ``facing``: its door's step on row (or column)
        ``line``, its extent along the row starting at ``along``."""
        return self.place_at(facing, along, line, P) is not None

    def place_at(self, facing: Cell, along: int, line: int, P: int, i: Optional[int] = None,
                 kind: str = "row-house") -> Optional[Dict[str, Any]]:
        """As ``place``; returns the house's record (with its x/z extent) or
        None where it did not fit. ``i`` picks the pool entry."""
        if i is None:
            i = self.rng.randrange(len(self.pool))
        v = self.variant(i, facing)
        if facing[1] != 0:          # a row along x
            ox, oz = along, line - v["stair"][1]
        else:                       # a row along z
            ox, oz = line - v["stair"][0], along
        cells = {(x + ox, z + oz) for (x, z) in v["base"]}
        if not all(self.t.inside(x, z, 1) and self.t.use[x, z] in (FREE, HOUSE) for (x, z) in cells):
            return None
        for c in cells:
            self.t.use[c] = HOUSE
        n_ports = len(self.b.ports)
        self.t.absorb(v["t"], ox, oz, P)
        floor = sorted(c for c in cells
                       if all((c[0] + a, c[1] + d) in cells for a, d in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        rec = {"cells": floor, "floor": P, "door": self.b.ports[n_ports]["position"],
               "variant": i, "kind": kind, "facing": _FACING[facing],
               "storeys": self.pool[i].notes[-1] if self.pool[i].notes else "",
               # one floor for each level a door opens onto
               "floors": len({int(round(q["position"][1])) for q in self.pool[i].ports}),
               "box": (min(x for x, _ in cells), max(x for x, _ in cells),
                       min(z for _, z in cells), max(z for _, z in cells))}
        self.t.houses.append(rec)
        return rec


def pueblo(player: Any = "blocky", seed: int = 0, k: int = 5, j: int = 4) -> BlockBuild:
    """A Pueblo of two rings of contiguous room blocks round a plaza of
    ``k`` blocks by ``j`` blocks, on level ground."""
    rng = random.Random(seed * 7919 + 211)
    pool = _pool(player, seed)
    D = 16                                     # a ring's depth: one room block
    L = 4                                      # the lane between the rings
    W, H = 7 * k + 1, 7 * j + 1                # the plaza
    margin = 8
    n = W + 2 * (2 * D + L) + 2 * margin
    m = H + 2 * (2 * D + L) + 2 * margin
    G0 = 20
    heights = np.full((n, m), G0, dtype=int)
    b = BlockBuild("pueblo", player, seed)
    t = _Town(b, heights)
    pb = _Pueblo(b, t, pool, rng)
    px0, pz0 = margin + 2 * D + L, margin + 2 * D + L
    px1, pz1 = px0 + W - 1, pz0 + H - 1
    P = G0

    # The plaza, the lane round the first ring and the passage in from the south.
    t.use[px0:px1 + 1, pz0:pz1 + 1] = PLAZA
    ix0, ix1, iz0, iz1 = px0 - D, px1 + D, pz0 - D, pz1 + D            # the first ring's outer edge
    lx0, lx1, lz0, lz1 = ix0 - L, ix1 + L, iz0 - L, iz1 + L
    t.use[lx0:lx1 + 1, lz0:lz1 + 1] = np.where(t.use[lx0:lx1 + 1, lz0:lz1 + 1] == PLAZA, PLAZA, STREET)
    t.use[ix0:ix1 + 1, iz0:iz1 + 1] = np.where(t.use[ix0:ix1 + 1, iz0:iz1 + 1] == PLAZA, PLAZA, FREE)
    gate = px0 + 7 * (k // 2) + 3              # the middle of the south row's middle block
    passage = {(x, z) for x in range(gate - 1, gate + 2) for z in range(pz1 + 1, m)}
    for c in passage:
        t.use[c] = STREET

    placed = 0
    # First ring, facing the plaza: north and south rows across the plaza's
    # width, east and west rows down its depth.
    for i in range(k):
        placed += pb.place((0, 1), px0 + 7 * i, pz0, P)
        placed += pb.place((0, -1), px0 + 7 * i, pz1, P)
    for i in range(j):
        placed += pb.place((-1, 0), pz0 + 7 * i, px1, P)
        placed += pb.place((1, 0), pz0 + 7 * i, px0, P)
    # The corners of the first ring face out onto the lane.
    for i in range(2):
        placed += pb.place((0, -1), px1 + 1 + 7 * i, iz0 - 1, P)
        placed += pb.place((0, -1), px0 - 15 + 7 * i, iz0 - 1, P)
        placed += pb.place((0, 1), px1 + 1 + 7 * i, iz1 + 1, P)
        placed += pb.place((0, 1), px0 - 15 + 7 * i, iz1 + 1, P)
    # Second ring, facing the lane.
    nx = (lx1 - lx0) // 7
    x_start = lx0 + ((lx1 - lx0 + 1) - (7 * nx + 1)) // 2
    for i in range(nx):
        placed += pb.place((0, 1), x_start + 7 * i, lz0, P)
        placed += pb.place((0, -1), x_start + 7 * i, lz1, P)
    nz = (lz1 - lz0) // 7
    z_start = lz0 + ((lz1 - lz0 + 1) - (7 * nz + 1)) // 2
    for i in range(nz):
        placed += pb.place((-1, 0), z_start + 7 * i, lx1, P)
        placed += pb.place((1, 0), z_start + 7 * i, lx0, P)

    # The kiva: a round chamber sunk in the plaza's north-west quarter, its
    # roof level with the plaza, a low wall round the hatch and the ladder
    # standing up out of it.
    kx, kz = px0 + W // 3, pz0 + H // 3
    R = 4
    for x in range(kx - R - 1, kx + R + 2):
        for z in range(kz - R - 1, kz + R + 2):
            r = math.hypot(x - kx, z - kz)
            if r <= R + .5:
                for y in range(P - 5, P):
                    if r > R - .5:
                        b.blocks[(x, y, z)] = state("smooth_sandstone")      # the masonry ring
                    elif y == P - 5:
                        b.blocks[(x, y, z)] = state("packed_mud")            # the floor
                    elif y == P - 4 and r > R - 1.5:
                        b.blocks[(x, y, z)] = state("smooth_sandstone_slab", type="bottom")  # the bench
                    else:
                        b.blocks[(x, y, z)] = state("air")
                if r > R - .5:
                    b.blocks[(x, P, z)] = state("mud_brick_slab", type="bottom")  # the low kerb round the roof
                elif not (abs(x - kx) <= 1 and abs(z - kz) <= 1):
                    b.blocks[(x, P - 1, z)] = state("stripped_spruce_log", axis="x")  # the roof
                    b.blocks[(x, P - 1, z)] = state("packed_mud") if (x + z) % 3 else b.blocks[(x, P - 1, z)]
                t.use[x, z] = PAD
    for y in range(P - 4, P + 4):
        b.blocks[(kx, y, kz)] = state("ladder", facing="north")
    b.blocks[(kx, P - 4, kz + 1)] = state("packed_mud")
    for y in range(P - 4, P + 3):
        b.blocks[(kx, y, kz + 1)] = state("packed_mud") if y < P else b.blocks.get((kx, y, kz + 1), state("air"))
    b.blocks[(kx, P - 4, kz - 2)] = state("campfire", lit="true")              # the fire pit
    # Bread ovens (hornos) on the plaza's edge.
    for (hx, hz) in ((px1 - 3, pz0 + 3), (px0 + 3, pz1 - 3), (px1 - 3, pz1 - 4)):
        for x in range(hx - 1, hx + 2):
            for z in range(hz - 1, hz + 2):
                b.blocks[(x, P, z)] = state("mud_bricks")
                if abs(x - hx) + abs(z - hz) <= 1:
                    b.blocks[(x, P + 1, z)] = state("packed_mud")
                t.use[x, z] = PAD
        b.blocks[(hx, P + 2, hz)] = state("packed_mud")
        b.blocks[(hx, P, hz - 1)] = state("furnace", facing="north", lit="true")
    # Ladders against the plaza walls are part of each block; drying racks
    # of chillies (red banners on fences) stand along the south row.
    for i in range(k):
        x = px0 + 7 * i + 1
        if t.use[x, pz1] == PLAZA:
            b.blocks[(x, P, pz1)] = state("spruce_fence")
            b.blocks[(x, P + 1, pz1)] = state("spruce_fence")

    # Fields outside the second ring: maize (wheat as its stand-in), beans
    # and squash in plots, ditches between.
    for x in range(n):
        for z in range(m):
            if t.use[x, z] != FREE or not t.inside(x, z, 1):
                continue
            if lx0 - D <= x <= lx1 + D and lz0 - D <= z <= lz1 + D:
                continue
            t.use[x, z] = FIELD
            if (x // 6 + z // 6) % 4 == 3 and x % 6 == 0:
                t.surface[(x, z)] = state("water")
                continue
            t.surface[(x, z)] = state("farmland", moisture="7")
            crop = ("wheat", "wheat", "beetroots", "pumpkin_stem")[(x // 6 + 2 * (z // 6)) % 4]
            b.blocks[(x, P, z)] = state(crop, age="3" if crop == "beetroots" else "7")
    # Ground: the plaza and lanes beaten earth, the rest dry grass.
    for x in range(n):
        for z in range(m):
            u = t.use[x, z]
            if u in (PLAZA, STREET):
                t.surface.setdefault((x, z), state(("coarse_dirt", "dirt", "packed_mud", "coarse_dirt")[(x * 7 + z * 3) % 4]))
            elif u in (FREE, PAD, GREEN):
                t.surface.setdefault((x, z), state("coarse_dirt") if ((x * 73856093) ^ (z * 19349663)) % 6 == 0 else state("grass_block"))
    ground = t.lay_ground()
    b.town = {  # type: ignore[attr-defined]
        "layout": "pueblo", "tradition": "pueblo", "plaza": [W, H], "ring_depth": D, "lane": L,
        "houses": t.houses, "heights": t.G, "heights_in": t.H0, "use": t.use,
        "entrance": (gate, m - 2), "kiva": (kx, kz),
        "counts": {"houses": placed, "blocks_wanted": 2 * k + 2 * j + 8 + 2 * nx + 2 * nz, "ground_blocks": ground},
    }
    b.notes.append(f"pueblo: {placed} room blocks in two rings round a {W} by {H} plaza, a kiva and ovens")
    return b


def _norse_pool(player: Any, seed: int, n: int = 10) -> List[BlockBuild]:
    return [blockstyles.norse_longhouse(player, seed * 1000 + 500 + k) for k in range(n)]


def _pit_house(b: BlockBuild, t: _Town, x0: int, z0: int, P: int, rng: random.Random) -> None:
    """A sunken-floored hut (a grubenhaus), six by four: its floor dug a
    block into the ground, a thatched roof coming down to the ground on its
    long sides, plank gable ends, the door in the west gable. Weaving and
    small crafts were done in them."""
    for x in range(x0 + 1, x0 + 5):
        for z in (z0 + 1, z0 + 2):
            t.G[x, z] = P - 1                       # the dug floor
            t.surface[(x, z)] = state("coarse_dirt")
    for x in range(x0, x0 + 6):
        b.blocks[(x, P, z0)] = state("spruce_stairs", facing="south", half="bottom")
        b.blocks[(x, P, z0 + 3)] = state("spruce_stairs", facing="north", half="bottom")
        for z in (z0 + 1, z0 + 2):
            b.blocks[(x, P + 1, z)] = state("hay_block")
    for x in (x0, x0 + 5):
        for z in (z0 + 1, z0 + 2):
            b.blocks[(x, P, z)] = state("spruce_planks")
            b.blocks[(x, P - 1, z)] = state("spruce_planks")
    b.blocks[(x0, P, z0 + 1)] = state("air")             # the door, a step down
    b.blocks[(x0, P - 1, z0 + 1)] = state("coarse_dirt")
    b.blocks[(x0 + 4, P - 1, z0 + 2)] = state("loom", facing="west")
    for x in range(x0, x0 + 6):
        for z in range(z0, z0 + 4):
            t.use[x, z] = PAD


def _farmyard(b: BlockBuild, t: _Town, box: Tuple[int, int, int, int], back: int, P: int,
              rng: random.Random) -> int:
    """Fence a farm's yard from its longhouse back ``back`` cells (signed:
    the yard runs north for negative), and set its outbuildings: a pit
    house or two, a haystack, a woodpile, a well. Returns cells fenced."""
    x0, x1, z0, z1 = box
    if back < 0:
        za, zb = z0 + back, z0 - 1
        gate_z = z0 - 1
    else:
        za, zb = z1 + 1, z1 + back
        gate_z = z1 + 1
    xa, xb = x0 - 1, x1 + 1
    fence = 0
    for x in range(xa, xb + 1):
        for z in range(za, zb + 1):
            if not t.inside(x, z, 1) or t.use[x, z] not in (FREE,):
                continue
            t.use[x, z] = GREEN
            # Trodden earth near the house and gate, grass further back.
            near = abs(z - (z0 - 1 if back < 0 else z1 + 1)) <= 2
            worn = ((x * 73856093) ^ (z * 19349663)) % 7 < (4 if near else 1)
            t.surface[(x, z)] = state("coarse_dirt") if worn else state("grass_block")
            edge = x in (xa, xb) or z == (za if back < 0 else zb)
            if edge:
                b.blocks[(x, P, z)] = state("spruce_fence")
                fence += 1
    # The yard's own path from the longhouse's back door to the gate side.
    # Outbuildings: pit houses along the far fence, haystack and woodpile.
    far = za + 1 if back < 0 else zb - 4
    xs = list(range(xa + 2, xb - 6, 9))
    rng.shuffle(xs)
    for x in xs[:rng.choice((1, 2, 2))]:
        if all(t.use[x + dx, far + dz] == GREEN for dx in range(6) for dz in range(4)):
            _pit_house(b, t, x, far, P, rng)
    hx = xb - 4
    hz = (za + zb) // 2
    if all(t.use[hx + dx, hz + dz] == GREEN for dx in range(3) for dz in range(3)):
        for dx in range(3):
            for dz in range(3):
                b.blocks[(hx + dx, P, hz + dz)] = state("hay_block")
                t.use[hx + dx, hz + dz] = PAD
        b.blocks[(hx + 1, P + 1, hz + 1)] = state("hay_block")
    wx, wz = xa + 2, (za + zb) // 2 + (1 if back < 0 else -1)
    if t.use[wx, wz] == GREEN and t.use[wx + 1, wz] == GREEN:
        for dx in range(2):
            b.blocks[(wx + dx, P, wz)] = state("stripped_spruce_log", axis="x")
        b.blocks[(wx, P + 1, wz)] = state("stripped_spruce_log", axis="x")
    return fence


def norse(player: Any = "blocky", seed: int = 0, farms: int = 4) -> BlockBuild:
    """A Viking-age village after Vorbasse in Jutland: fenced farmsteads in
    two rows either side of one village street, every longhouse long side to
    the street and its yard behind it, with pit houses, a haystack and a
    woodpile in each yard. Halfway along, a green opens off the street with
    the chieftain's hall at its head and two farms facing in from its sides.
    Strip fields and pasture lie beyond the yards. ``farms`` farms per row
    on each side of the green."""
    rng = random.Random(seed * 7919 + 977)
    pool = _norse_pool(player, seed)
    lens = []
    for k, p in enumerate(pool):
        base = {(x, z) for (x, y, z) in p.blocks if y == 0}
        lens.append(max(x for x, _ in base) - min(x for x, _ in base) + 1)
    hall_i = max(range(len(pool)), key=lambda k: lens[k])
    S = 5                                          # the street's width
    Y = 26                                         # a yard's depth behind its house
    D = 11                                         # a longhouse's depth
    GW, GH = 24, 23                                # the green
    margin = 10
    seg = farms * 27 + 14                          # street on either side of the green
    n = 2 * margin + 2 * seg + GW
    m = 2 * margin + 2 * (D + Y) + S + GH + D + 4
    G0 = 20
    P = G0
    heights = np.full((n, m), G0, dtype=int)
    b = BlockBuild("norse-village", player, seed)
    t = _Town(b, heights)
    pb = _Pueblo(b, t, pool, rng)
    zs0 = margin + D + Y + GH + D + 2               # the street's north edge
    zs1 = zs0 + S - 1
    gx0 = margin + seg
    gx1 = gx0 + GW - 1
    t.use[margin:n - margin, zs0:zs1 + 1] = STREET
    t.use[gx0:gx1 + 1, zs0 - GH:zs0] = PLAZA

    placed = 0
    # The hall at the head of the green, facing down it.
    hall = pb.place_at((0, 1), gx0 + (GW - lens[hall_i]) // 2, zs0 - GH - 1, P, i=hall_i, kind="hall")
    placed += hall is not None
    # The farms facing in from the green's sides, their yards behind them.
    for facing, line, sign in (((1, 0), gx0 - 1, -1), ((-1, 0), gx1 + 1, 1)):
        for i in sorted(range(len(pool)), key=lambda k: lens[k]):
            rec = pb.place_at(facing, zs0 - GH + 1, line, P, i=i)
            if rec is not None:
                placed += 1
                x0, x1, z0, z1 = rec["box"]
                # The yard runs back from the house, away from the green.
                if sign < 0:
                    _farmyard(b, t, (x0 - 14, x0 - 1, z0, z1), 0, P, rng)
                else:
                    _farmyard(b, t, (x1 + 1, x1 + 14, z0, z1), 0, P, rng)
                break
    # The two rows of farmsteads along the street.
    rows = 0
    for facing, line, back in (((0, 1), zs0 - 1, -Y), ((0, -1), zs1 + 1, Y)):
        x = margin
        while x < n - margin - 16:
            if facing == (0, 1) and gx0 - 14 <= x + 22 and x <= gx1 + 14:
                x = gx1 + 15                       # past the green and the farms that face it
                continue
            rec = None
            for _ in range(len(pool)):
                rec = pb.place_at(facing, x, line, P)
                if rec is not None:
                    break
            if rec is None:
                x += 1
                continue
            placed += 1
            rows += 1
            _farmyard(b, t, rec["box"], back, P, rng)
            x = rec["box"][1] + 3                  # a two-block gap: the yard's gate
    # The street runs only as far as the last farms; beyond is open land.
    xs_ = [h["box"][0] for h in t.houses] + [h["box"][1] for h in t.houses]
    sx0, sx1 = min(xs_) - 1, max(xs_) + 1
    t.use[margin:sx0, zs0:zs1 + 1] = FREE
    t.use[sx1 + 1:n - margin, zs0:zs1 + 1] = FREE
    # The green: a runestone, a well, an old ash tree.
    cx, cz = (gx0 + gx1) // 2, zs0 - GH // 2
    for y in range(P, P + 3):
        b.blocks[(cx, y, cz)] = state("chiseled_stone_bricks" if y == P + 1 else "stone_bricks")
    b.blocks[(cx, P + 3, cz)] = state("stone_brick_slab", type="bottom")
    wx, wz = gx0 + 5, zs0 - 6
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            b.blocks[(wx + dx, P, wz + dz)] = state("water") if dx == dz == 0 else state("cobblestone_wall")
            t.use[wx + dx, wz + dz] = PAD
    tx, tz = gx1 - 6, zs0 - GH + 8
    for y in range(P, P + 6):
        b.blocks[(tx, y, tz)] = state("oak_log", axis="y")
    for dx in range(-3, 4):
        for dz in range(-3, 4):
            for y in range(P + 4, P + 8):
                if dx * dx + dz * dz + (y - P - 6) ** 2 * 2 <= 12 and (dx, dz) != (0, 0) or y >= P + 6 and abs(dx) + abs(dz) <= 2:
                    b.blocks.setdefault((tx + dx, y, tz + dz), state("oak_leaves", persistent="true"))
    t.use[tx, tz] = PAD
    # Fields and pasture beyond the yards: long strips of barley (wheat as its
    # stand-in) and flax, a fenced pasture with sheep, the rest grass.
    for x in range(n):
        for z in range(m):
            if t.use[x, z] != FREE or not t.inside(x, z, 2):
                continue
            if abs(z - (zs0 + zs1) / 2) < 6:
                continue
            band = (x // 8) % 5
            if band in (0, 1, 3):
                t.use[x, z] = FIELD
                if x % 8 == 7:
                    t.surface[(x, z)] = state("grass_block")
                    continue
                t.surface[(x, z)] = state("farmland", moisture="7")
                crop = ("wheat", "wheat", "beetroots", "carrots")[(x // 8 + z // 30) % 4]
                b.blocks[(x, P, z)] = state(crop, age="3" if crop == "beetroots" else "7")
    for x in range(n):
        for z in range(m):
            u = t.use[x, z]
            if u == STREET:
                t.surface.setdefault((x, z), state(("dirt_path", "coarse_dirt", "dirt_path", "gravel")[(x * 5 + z * 3) % 4]))
            elif u == PLAZA:
                t.surface.setdefault((x, z), state("coarse_dirt") if ((x * 73856093) ^ (z * 19349663)) % 9 == 0
                                     else state("grass_block"))
    ground = t.lay_ground()
    b.town = {  # type: ignore[attr-defined]
        "layout": "norse-village", "tradition": "norse", "green": [GW, GH], "street": S, "yard": Y,
        "houses": t.houses, "heights": t.G, "heights_in": t.H0, "use": t.use,
        "entrance": (sx0 - 2, (zs0 + zs1) // 2),
        "counts": {"houses": placed, "street_farms": rows, "ground_blocks": ground},
    }
    b.notes.append(f"norse village: {placed} longhouses (a hall, {rows} farms on the street), "
                   f"fenced yards with pit houses, a green of {GW} by {GH}")
    return b


def mediterranean(player: Any = "blocky", seed: int = 0, bx: int = 3, bz: int = 3,
                  per_row: int = 4) -> BlockBuild:
    """A Mediterranean town of courtyard houses built wall to wall in
    blocks, as in a Roman insula grid or an old Andalusian or south Italian
    town: each block two rows of houses back to back, every house turned in
    on its own court with only its door and small shuttered windows to the
    street. Narrow streets run between the blocks; the middle block is left
    open as the piazza, paved, with a fountain, its rim lined by the house
    fronts of the blocks round it."""
    rng = random.Random(seed * 7919 + 1231)
    pool = [blockstyles.mediterranean_courtyard_house(player, seed * 1000 + 700 + k) for k in range(10)]
    St = 5                                          # a street's width
    widths = []
    for p in pool:
        base = {(x, z) for (x, y, z) in p.blocks if y == 0}
        widths.append(max(x for x, _ in base) - min(x for x, _ in base) + 1)
    BW = per_row * 16 + 1                           # a block's length, house plinths shared
    BD = 2 * max(widths) - 1                        # two rows back to back
    margin = 8
    n = 2 * margin + bx * BW + (bx + 1) * St
    m = 2 * margin + bz * BD + (bz + 1) * St
    G0 = 20
    P = G0
    heights = np.full((n, m), G0, dtype=int)
    b = BlockBuild("mediterranean-town", player, seed)
    t = _Town(b, heights)
    pb = _Pueblo(b, t, pool, rng)
    x_of = [margin + St + i * (BW + St) for i in range(bx)]
    z_of = [margin + St + j * (BD + St) for j in range(bz)]
    # Streets between the blocks; the outer rows look out over the gardens.
    t.use[x_of[0]:x_of[-1] + BW, z_of[0]:z_of[-1] + BD] = STREET
    for x0 in x_of:
        for z0 in z_of:
            t.use[x0:x0 + BW, z0:z0 + BD] = FREE
    ci, cj = bx // 2, bz // 2
    sx0, sz0 = x_of[ci] - St, z_of[cj] - St
    sx1, sz1 = x_of[ci] + BW + St - 1, z_of[cj] + BD + St - 1
    t.use[sx0:sx1 + 1, sz0:sz1 + 1] = PLAZA

    placed = 0
    for i, x0 in enumerate(x_of):
        for j, z0 in enumerate(z_of):
            if (i, j) == (ci, cj):
                continue
            for facing, line in (((0, -1), z0), ((0, 1), z0 + BD - 1)):
                x = x0
                while x < x0 + BW - 12:
                    order = list(range(len(pool)))
                    rng.shuffle(order)
                    rec = None
                    for k in order:
                        if x + widths[k] - 1 > x0 + BW - 1:
                            continue
                        rec = pb.place_at(facing, x, line, P, i=k, kind="courtyard-house")
                        if rec is not None:
                            break
                    if rec is None:
                        break
                    placed += 1
                    x = rec["box"][1]                   # the next house shares its plinth
    # The piazza: a fountain in the middle, a stone bench round it, and
    # cypresses in pots at the corners.
    cx, cz = (sx0 + sx1) // 2, (sz0 + sz1) // 2
    for dx in range(-3, 4):
        for dz in range(-3, 4):
            r = max(abs(dx), abs(dz))
            if r == 3:
                b.blocks[(cx + dx, P, cz + dz)] = state("smooth_stone_slab", type="bottom")
            elif r == 2:
                b.blocks[(cx + dx, P, cz + dz)] = state("stone_bricks")
            else:
                b.blocks[(cx + dx, P, cz + dz)] = state("water")
            t.use[cx + dx, cz + dz] = PAD if r <= 2 else PLAZA
    for y in range(P, P + 3):
        b.blocks[(cx, y, cz)] = state("stone_brick_wall") if y < P + 2 else state("water")
    for (px, pz) in ((sx0 + St + 2, sz0 + St + 2), (sx1 - St - 2, sz0 + St + 2),
                     (sx0 + St + 2, sz1 - St - 2), (sx1 - St - 2, sz1 - St - 2)):
        b.blocks[(px, P, pz)] = state("decorated_pot")
        for y in range(P + 1, P + 5):
            b.blocks[(px, y, pz)] = state("spruce_leaves", persistent="true")
        b.blocks[(px, P + 5, pz)] = state("spruce_leaves", persistent="true")
        t.use[px, pz] = PAD
    for x in range(n):
        for z in range(m):
            u = t.use[x, z]
            if u == STREET:
                t.surface.setdefault((x, z), state(("cobblestone", "stone", "andesite", "cobblestone")[(x * 5 + z * 3) % 4]))
            elif u == PLAZA:
                t.surface.setdefault((x, z), state("smooth_stone") if (x + z) % 2 else state("polished_andesite"))
            elif u == FREE:
                t.surface.setdefault((x, z), state("coarse_dirt") if ((x * 73856093) ^ (z * 19349663)) % 6 == 0 else state("grass_block"))
    ground = t.lay_ground()
    mid_z = z_of[cj] - St // 2 - 1
    b.town = {  # type: ignore[attr-defined]
        "layout": "mediterranean-blocks", "tradition": "mediterranean", "blocks": [bx, bz], "street": St,
        "houses": t.houses, "heights": t.G, "heights_in": t.H0, "use": t.use,
        "entrance": (x_of[0] - 2, mid_z),
        "counts": {"houses": placed, "ground_blocks": ground},
    }
    b.notes.append(f"mediterranean town: {placed} courtyard houses wall to wall in {bx * bz - 1} blocks "
                   f"round a piazza")
    return b


def region_walk(b: BlockBuild) -> Tuple[Any, Dict[str, Any]]:
    """Walk a regional town from its entrance and say which floors of each
    building the player reaches."""
    from .gameplay import walk as walk_grid
    from .blockmode import kind

    sub = b.walk_sub()
    g = b.grid(sub)
    ox, oy, oz = g.origin
    ex, ez = b.town["entrance"]
    G = b.town["heights"]
    y0 = int(G[ex, ez])
    while b.blocks.get((ex, y0, ez)) is not None and kind(b.blocks[(ex, y0, ez)]) != "air":
        y0 += 1
    seed = (int((ex + .5 - ox) * sub), int((y0 - oy) * sub), int((ez + .5 - oz) * sub))
    w = walk_grid(g, b.player, seeds=[seed], climbs=b.climbs(sub, g.origin))
    nodes = w.nodes
    wx = nodes[:, 0] / sub + ox
    wy = nodes[:, 1] / sub + oy
    wz = nodes[:, 2] / sub + oz
    S = b.sizes.storey
    cell_of = np.floor(wx).astype(int) * 100003 + np.floor(wz).astype(int)
    houses_ok = floors_ok = floors_total = 0
    for h in b.town["houses"]:
        keys = np.array([x * 100003 + z for (x, z) in h["cells"]])
        inside = np.isin(cell_of, keys)
        floors = ok = 0
        for kk in range(h.get("floors", 6)):
            here = inside & (np.abs(wy - (h["floor"] + 1 + kk * S)) < .3)
            if not here.any():
                break
            floors += 1
            ok += int(w.reached[here].any())
        floors_total += floors
        floors_ok += ok
        houses_ok += int(floors > 0 and ok == floors)
    kx, kz = b.town.get("kiva", (None, None))
    kiva = False
    if kx is not None:
        P = int(G[kx, kz])
        here = (np.abs(wx - (kx + .5)) < 3) & (np.abs(wz - (kz + .5)) < 3) & (np.abs(wy - (P - 4)) < .3)
        kiva = bool(here.any() and w.reached[here].any())
    return w, {"houses": len(b.town["houses"]), "houses_every_floor_reached": houses_ok,
               "floors_reached": f"{floors_ok}/{floors_total}", "kiva_floor_reached": kiva}
