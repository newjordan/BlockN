"""Block-mode towns fitted to a given height map.

The terrain is an input, not something this module makes: a 2-D array of
ground heights in whole blocks (``heights[x, z]`` blocks of ground under
column (x, z)), such as one read from a Minecraft map. ``sample_heights``
gives stand-in maps (a hill, a hillside, a valley) for tests and pictures.

``town`` fits a castle and a dense town to that ground:

- the castle goes where the ground is high and level near the middle of
  the map, turned so its gate has room in front, on a platform cut and
  filled to one level;
- a market square is levelled in front of the gate;
- a main street runs from the square down to the edge of the map, and side
  streets branch from it and from the square; each street is routed around
  steep ground, its surface eased so no step is more than one block, and a
  stair block is set at each rise;
- houses front onto every street and the square, each on its own levelled
  pad at the street's height, with retaining walls where a pad is cut into
  the slope or built out over it;
- fields go on gentle ground near the edge of town, then lamp posts, trees,
  flowers and grass.

Everything is sized to the player through ``BlockSizes``, and every block is
a Minecraft block state, as in ``blockmode``.
"""

from __future__ import annotations

import heapq
from collections import Counter
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .blockcraft import INTERIOR_FLOORS, _turn_xz, castle, house, lamp_post, paste, tree, well
from .blockkinds import apartment, mine, tower, windmill, workshop
from . import blockstyles, lore
from .blockmode import DIRS, BlockBuild, disc, kind, state

Cell = Tuple[int, int]

FREE, CASTLE, STREET, PLAZA, HOUSE, PAD, FIELD, GREEN = range(8)
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_N8 = _N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
_FRONTS = ("north", "east", "south", "west")      # quarter-turn q turns north to _FRONTS[q]

_MAIN = (("cobblestone", 5), ("stone_bricks", 2), ("andesite", 2), ("gravel", 1), ("mossy_cobblestone", 1))
_SIDE = (("dirt_path", 7), ("coarse_dirt", 2), ("gravel", 2), ("cobblestone", 1))
_SQUARE = (("stone_bricks", 4), ("polished_andesite", 3), ("cobblestone", 2), ("andesite", 1))
_RETAIN = (("stone_bricks", 4), ("cobblestone", 3), ("mossy_cobblestone", 2), ("mossy_stone_bricks", 1))
_ROCK = (("stone", 6), ("andesite", 2), ("tuff", 1), ("cobblestone", 1))
# A cliff's face in bands, by height, so a tall face reads as strata.
_STRATA = ((("stone", 6), ("andesite", 2), ("cobblestone", 1)), (("andesite", 4), ("stone", 3), ("diorite", 1)),
           (("tuff", 4), ("stone", 3), ("andesite", 1)), (("stone", 5), ("granite", 1), ("andesite", 2)))
# Trim on a house's outside that a landing may clear.
_TRIM = ("trapdoor", "potted_", "flower_pot", "_slab", "lantern", "banner", "vine", "leaves", "fern")
# How many of each building type a town takes, and how far apart they stand.
_KIND_QUOTA = {"tower": 4, "workshop": 8, "windmill": 2, "apartment": 14, "townhouse": 30, "shop": 90}
_KIND_SPACING = {"tower": 22, "workshop": 8, "windmill": 30, "apartment": 0, "townhouse": 0, "shop": 0}


# ----------------------------------------------------------- sample ground
_FACE = {(1, 0): "east", (-1, 0): "west", (0, 1): "south", (0, -1): "north"}


def fountain(part: BlockBuild, mx: int, mz: int, rng: random.Random) -> None:
    """A 5 by 5 stone basin of water round a carved column with a spout of
    water on top and lanterns on the corners, centred on (mx, mz)."""
    for x in range(mx - 2, mx + 3):
        for z in range(mz - 2, mz + 3):
            ring = max(abs(x - mx), abs(z - mz)) == 2
            if ring:
                corner = abs(x - mx) == 2 and abs(z - mz) == 2
                part.set(x, 0, z, state("chiseled_stone_bricks" if corner else "stone_bricks"))
                if corner:
                    part.set(x, 1, z, state("lantern", hanging="false"))
                else:
                    part.set(x, 1, z, state("stone_brick_slab", type="bottom"))
                part.path(x, z, "stone_bricks")
            else:
                part.path(x, z, "water")
                if (x, z) != (mx, mz):
                    part.set(x, 0, z, state("water"))
    part.set(mx, 0, mz, state("chiseled_stone_bricks"))
    part.set(mx, 1, mz, state("stone_brick_wall"))
    part.set(mx, 2, mz, state("stone_brick_wall"))
    part.set(mx, 3, mz, state("chiseled_stone_bricks"))
    part.set(mx, 4, mz, state("water"))


def stall(part: BlockBuild, sx: int, sz: int, along: Cell, facing: Cell, side: int, colour: str, good: str,
          rng: random.Random) -> None:
    """A 3 by 3 market stall on fence posts under a striped wool canopy,
    its open front toward the middle of the square, a counter of slabs with
    wares on it, a crate of ``good`` and a lantern hanging under the canopy."""
    ax, az = along
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            x, z = sx + ax * di + facing[0] * dj, sz + az * di + facing[1] * dj
            if abs(di) == 1 and abs(dj) == 1:
                part.set(x, 0, z, state("spruce_fence"))
                part.set(x, 1, z, state("spruce_fence"))
            part.set(x, 2, z, state(f"{colour}_wool" if dj != 0 else "white_wool"))
    # Front (toward the square's middle) is -side along ``along``.
    fx_, fz_ = sx - ax * side, sz - az * side
    part.set(fx_, 0, fz_, state("spruce_slab", type="top"))
    part.set(fx_, 1, fz_, state(rng.choice(("potted_red_tulip", "candle", "potted_fern", "flower_pot"))))
    part.set(sx, 0, sz, state(good) if good != "barrel" else state("barrel", facing="up"))
    bx, bz = sx + ax * side, sz + az * side
    part.set(bx, 0, bz, state("barrel", facing="up"))
    part.set(bx, 1, bz, state(rng.choice(("hay_block", "barrel", "composter", "pumpkin"))))
    part.set(sx, 1, sz, state("lantern", hanging="true"))


def sample_heights(kind: str = "hill", size: int = 112, seed: int = 0) -> np.ndarray:
    """A stand-in height map of ``size`` by ``size`` columns, for tests and
    pictures when no real map is given. ``kind`` is "hill" (a rise in the
    middle), "slope" (a hillside rising to the north), "valley" (a valley
    running north to south) or "valley-wall" (a valley floor under a steep
    wall to the north). Heights are whole blocks, at least 2."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, size)
    X, Z = np.meshgrid(t, t, indexing="ij")
    noise = np.zeros_like(X)
    for k in range(6):
        f = 1.2 + 1.1 * k
        th, ph = rng.uniform(0, 2 * np.pi, 2)
        noise += np.sin(2 * np.pi * f * (X * np.cos(th) + Z * np.sin(th)) + ph) / (1 + k)
    noise /= max(1e-9, np.abs(noise).max())
    if kind == "hill":
        base = 4 + 24 * np.exp(-((X - .5) ** 2 + (Z - .42) ** 2) / .07)
    elif kind == "slope":
        base = 4 + 32 * (1 - Z) ** 1.15 + 3 * np.exp(-((X - .5) ** 2 + (Z - .3) ** 2) / .03)
    elif kind == "valley":
        base = 3 + 24 * np.abs(2 * (X - .5)) ** 1.6
    elif kind in ("valley-wall", "valley_wall"):
        # A valley floor to the south and a steep wall rising about 44 blocks
        # to the north, curving round at both ends like the side of a gorge.
        edge = .60 + .12 * ((X - .5) / .5) ** 2
        u = np.clip((edge - Z) / (edge - .07), 0.0, 1.0)
        base = 5 + 3 * (1 - Z) + 62 * u ** 1.5
    else:
        raise ValueError(f"unknown sample terrain '{kind}'")
    return np.clip(np.rint(base + 2.5 * noise), 2, None).astype(int)


def load_heights(path: str) -> np.ndarray:
    """A height map from a ``.npy`` or ``.json`` file (a 2-D list of whole
    block heights, indexed [x][z])."""
    if str(path).endswith(".npy"):
        a = np.load(path)
    else:
        import json
        from pathlib import Path
        a = np.array(json.loads(Path(path).read_text()))
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("a height map is a 2-D array of heights")
    return np.clip(np.rint(a), 1, None).astype(int)


# ----------------------------------------------------------------- helpers
def turned(part: BlockBuild, quarter: int) -> BlockBuild:
    """``part`` turned ``quarter`` quarter-turns, moved so its lowest x and z
    are 0."""
    tmp = BlockBuild(part.product, part.player, part.seed)
    paste(tmp, part, 0, 0, quarter)
    (lx, _, lz), _hi = tmp.bounds()
    out = BlockBuild(part.product, part.player, part.seed)
    paste(out, tmp, -lx, -lz)
    out.notes = list(part.notes)
    return out


class _Town:
    def __init__(self, b: BlockBuild, heights: np.ndarray):
        self.b = b
        self.rng = b.rng
        self.n = heights.shape[0]
        self.m = heights.shape[1]
        self.H0 = heights.astype(int).copy()
        self.G = heights.astype(int).copy()
        self.use = np.zeros(heights.shape, dtype=np.uint8)
        self.surface: Dict[Cell, str] = {}
        self.streets: List[Dict[str, Any]] = []
        self.houses: List[Dict[str, Any]] = []
        self.fixed: Set[Cell] = set()     # cells whose height must not change again
        self.eaves: Dict[Cell, int] = {}  # cells under a house's eaves only, with its floor height
        self.steep: Set[Cell] = set()     # land climbed by terraces, where houses may stand on deeper cuts
        self.terrace_rows = 2             # rows of houses between one terrace street and the next
        self.max_cut = 14                 # deepest cut or fill under the castle's platform, in blocks
        self.under: Dict[Cell, int] = {}  # columns over a mine: the top layer of its rock skin
        self.kinds: Dict[str, List[Cell]] = {}
        self.stilts = 0
        self.embedded = 0
        self.crags = 0
        self.back_doors = 0
        self.cliff_fronts = 0
        self.rejects: Counter = Counter()   # (kind, reason) -> plots a building did not fit

    # ------------------------------------------------------------ basics
    def inside(self, x: int, z: int, margin: int = 0) -> bool:
        return margin <= x < self.n - margin and margin <= z < self.m - margin

    def absorb(self, part: BlockBuild, dx: int = 0, dz: int = 0, dy: int = 0) -> None:
        """Copy a part already in world x/z (plus dx, dz) with its layer 0 at
        height ``dy``; its ground layer becomes the top block of the ground."""
        for (x, y, z), block in part.blocks.items():
            self.b.blocks[(x + dx, y + dy, z + dz)] = block
        for (x, y, z), data in part.nbt.items():
            self.b.nbt[(x + dx, y + dy, z + dz)] = data
        for e in part.entities:
            ne = dict(e)
            ne["pos"] = [e["pos"][0] + dx, e["pos"][1] + dy, e["pos"][2] + dz]
            self.b.entities.append(ne)
        for (x, z), block in part.ground.items():
            if self.inside(x + dx, z + dz):
                self.surface[(x + dx, z + dz)] = block
        for port in part.ports:
            p = dict(port)
            p["position"] = [port["position"][0] + dx, port["position"][1] + dy, port["position"][2] + dz]
            self.b.ports.append(p)
        self.b.notes.extend(part.notes)

    def level(self, cells: Sequence[Cell], h: int) -> None:
        for (x, z) in cells:
            if self.inside(x, z):
                self.G[x, z] = h

    def pick(self, choices: Sequence[Tuple[str, float]]) -> str:
        names = [c[0] for c in choices]
        weights = [c[1] for c in choices]
        return state(self.rng.choices(names, weights)[0])

    # ------------------------------------------------------------ castle
    def place_castle(self, keep: BlockBuild) -> Dict[str, Any]:
        best = None
        n, m = self.n, self.m
        for q in range(4):
            t = turned(keep, q)
            (lx, _, lz), (hx, _, hz) = t.bounds()
            w, d = hx + 1, hz + 1
            gate = next(p for p in t.ports if p["id"].startswith("port.gate."))
            fx, fz = int(round(gate["facing"][0])), int(round(gate["facing"][2]))
            gx, gz = int(math.floor(gate["position"][0])), int(math.floor(gate["position"][2]))
            for sx in range(3, n - w - 3, 2):
                for sz in range(3, m - d - 3, 2):
                    # Room ahead of the gate for the square and a street.
                    ax, az = sx + gx + fx * 30, sz + gz + fz * 30
                    if not self.inside(ax, az, 1):
                        continue
                    rect = self.H0[sx - 1:sx + w + 1, sz - 1:sz + d + 1]
                    T = int(np.median(rect))
                    earth = float(np.abs(rect - T).mean())
                    if float(np.abs(rect - T).max()) > self.max_cut:
                        continue          # no castle on a platform cut or built up more than that
                    bx, bz = sx + gx + fx * 10, sz + gz + fz * 10
                    ahead = self.H0[max(0, bx - 6):bx + 6, max(0, bz - 6):bz + 6]
                    down = T - float(ahead.mean()) if ahead.size else 0.0
                    cx, cz = sx + w / 2, sz + d / 2
                    centre = math.hypot(cx - n / 2, cz - m / 2) / max(n, m)
                    score = .6 * T - 2.0 * earth - 14 * centre + .3 * min(max(down, -4), 8)
                    if best is None or score > best[0]:
                        best = (score, q, sx, sz, T)
        if best is None:
            raise ValueError("the map is too small for the castle and a town in front of it")
        _, q, sx, sz, T = best
        t = turned(keep, q)
        (_, _, _), (hx, _, hz) = t.bounds()
        rect = [(x, z) for x in range(sx - 1, sx + hx + 2) for z in range(sz - 1, sz + hz + 2)]
        self.level(rect, T)
        for (x, z) in rect:
            self.use[x, z] = CASTLE
            self.fixed.add((x, z))
        self.absorb(t, sx, sz, T)
        gate = [p for p in self.b.ports if p["id"].startswith("port.gate.")][-1]
        f = (int(round(gate["facing"][0])), int(round(gate["facing"][2])))
        g = (int(math.floor(gate["position"][0])), int(math.floor(gate["position"][2])))
        self.b.notes.append(f"town: castle at ({sx}, {sz}) turned {q}, on a platform at height {T}")
        return {"q": q, "x": sx, "z": sz, "w": hx + 1, "d": hz + 1, "T": T, "gate": g, "facing": f}

    # ------------------------------------------------------------ square
    def place_square(self, c: Dict[str, Any]) -> Dict[str, Any]:
        (gx, gz), (fx, fz) = c["gate"], c["facing"]
        ax, az = -fz, fx                          # along the castle front
        half, depth = 10, 15
        # Start the square at the castle's outer face, not inside its gate.
        for j0 in range(0, 12):
            x, z = gx + fx * j0, gz + fz * j0
            if self.inside(x, z, 1) and self.use[x, z] == FREE:
                break
        gx, gz = gx + fx * j0, gz + fz * j0
        cells = []
        for i in range(-half, half + 1):
            for j in range(0, depth):
                x, z = gx + ax * i + fx * j, gz + az * i + fz * j
                if self.inside(x, z, 1) and self.use[x, z] == FREE:
                    cells.append((x, z))
        T = c["T"]
        self.level(cells, T)
        for cell in cells:
            self.use[cell] = PLAZA
            self.fixed.add(cell)
            self.surface[cell] = self.pick(_SQUARE)
        # A fountain in the middle with benches round it, striped market
        # stalls down both sides with counters and hanging lanterns, lamp posts.
        part = BlockBuild("square", self.b.player, self.b.seed)
        drng = random.Random(self.b.seed * 7919 + 101)   # its own stream: dressing never moves the plan
        mx, mz = gx + fx * (depth // 2), gz + fz * (depth // 2)
        fountain(part, mx, mz, drng)
        for k in range(4):
            dx, dz = _N4[k]
            bx, bz = mx + dx * 4, mz + dz * 4
            part.set(bx, 0, bz, state("spruce_stairs", facing=_FACE[(dx, dz)], half="bottom"))
        colours = ("red", "blue", "yellow", "orange", "light_blue", "green", "cyan", "lime", "purple")
        goods = ("barrel", "hay_block", "pumpkin", "melon", "composter", "chest", "bee_nest", "carved_pumpkin")
        for side in (-1, 1):
            for j in (2, 6, 10, 13):
                sx0, sz0 = gx + ax * side * (half - 3) + fx * j, gz + az * side * (half - 3) + fz * j
                stall(part, sx0, sz0, (ax, az), (fx, fz), side, drng.choice(colours),
                      drng.choice(goods), drng)
        for i in (-half, half):
            for j in (0, depth - 1):
                lamp_post(part, gx + ax * i + fx * j, gz + az * i + fz * j, 4)
        for i in (-half // 2, half // 2):
            lamp_post(part, gx + ax * i + fx * (depth - 1), gz + az * i + fz * (depth - 1), 4)
        # Crates and sacks between the stalls along the square's sides.
        for side in (-1, 1):
            for j in (4, 8, 12):
                x, z = gx + ax * side * (half - 1) + fx * j, gz + az * side * (half - 1) + fz * j
                part.set(x, 0, z, state(drng.choice(("barrel", "hay_block", "composter", "chest", "pumpkin"))))
                if drng.random() < .5:
                    part.set(x, 1, z, state(drng.choice(("barrel", "hay_block", "melon"))))
        self.absorb(part, 0, 0, T)
        far = (gx + fx * (depth - 1), gz + fz * (depth - 1))
        sides = [(gx + ax * s * half + fx * (depth // 2), gz + az * s * half + fz * (depth // 2)) for s in (-1, 1)]
        corners = [(gx + ax * s * half + fx * (depth - 1), gz + az * s * half + fz * (depth - 1)) for s in (-1, 1)]
        # How far the square reaches out beyond the castle's platform edge.
        edge = {(1, 0): c["x"] + c["w"] - 1, (-1, 0): -c["x"], (0, 1): c["z"] + c["d"] - 1,
                (0, -1): -c["z"]}[(fx, fz)]
        reach = (far[0] * fx + far[1] * fz) - edge
        return {"cells": cells, "far": far, "sides": sides, "corners": corners, "along": (ax, az), "T": T,
                "reach": reach}

    # ------------------------------------------------------------ streets
    def route(self, start: Cell, goal: Cell, through: Set[int]) -> List[Cell]:
        """A* over the ground: cheap on the level, dear up or down, never
        through anything but ``through`` uses."""
        G = self.G
        open_: List[Tuple[float, float, Cell]] = [(0.0, 0.0, start)]
        came: Dict[Cell, Optional[Cell]] = {start: None}
        cost: Dict[Cell, float] = {start: 0.0}
        while open_:
            _, g, cur = heapq.heappop(open_)
            if cur == goal:
                break
            if g > cost.get(cur, 1e18) + 1e-9:
                continue
            for dx, dz in _N8:
                nx, nz = cur[0] + dx, cur[1] + dz
                if not self.inside(nx, nz, 2):
                    continue
                if (nx, nz) != goal and self.use[nx, nz] not in through:
                    continue
                if dx and dz and (self.use[cur[0] + dx, cur[1]] not in through or
                                  self.use[cur[0], cur[1] + dz] not in through):
                    continue
                dh = abs(int(G[nx, nz]) - int(G[cur]))
                step = (1.414 if dx and dz else 1.0) * (1 + 3.0 * dh + (25.0 if dh > 1 else 0.0))
                if self.use[nx, nz] == STREET:
                    step *= .6
                ng = g + step
                if ng < cost.get((nx, nz), 1e18):
                    cost[(nx, nz)] = ng
                    came[(nx, nz)] = cur
                    hh = math.hypot(goal[0] - nx, goal[1] - nz)
                    heapq.heappush(open_, (ng + hh, ng, (nx, nz)))
        if goal not in came:
            return []
        out = []
        cur: Optional[Cell] = goal
        while cur is not None:
            out.append(cur)
            cur = came[cur]
        return out[::-1]

    def lay_street(self, line: List[Cell], radius: float, mix: Sequence[Tuple[str, float]], h0: int,
                   name: str, heights: Optional[Sequence[int]] = None, trim: int = 0) -> Dict[str, Any]:
        """Stamp a street of half-width ``radius`` along ``line``, its surface
        eased from height ``h0`` to follow the ground with no step over one
        (or at ``heights``, when given). With ``trim``, the street stops
        where its surface would stand more than ``trim`` blocks off the
        ground: steep ground is left to the climbing streets."""
        if not line:
            return {}
        if heights is not None:
            hs = [int(h) for h in heights]
        else:
            raw = np.array([self.G[c] for c in line], dtype=float)
            k = 4
            smooth = np.array([raw[max(0, i - k):i + k + 1].mean() for i in range(len(raw))])
            hs = []
            prev = h0
            for i, c in enumerate(line):
                if c in self.fixed:
                    h = int(self.G[c])
                else:
                    h = int(round(smooth[i]))
                h = max(prev - 1, min(prev + 1, h)) if i else (int(self.G[c]) if c in self.fixed else h0)
                hs.append(h)
                prev = h
        if trim:
            for i, (c, h) in enumerate(zip(line, hs)):
                if c not in self.fixed and c in self.steep and abs(h - int(self.G[c])) > trim:
                    line, hs = line[:i], hs[:i]
                    break
            if len(line) < 6:
                return {}
        cells: Dict[Cell, int] = {}
        offs = sorted(disc(0.5, 0.5, radius + .5), key=lambda o: o[0] ** 2 + o[1] ** 2)
        for (cx, cz), h in zip(line, hs):
            for (ox, oz) in offs:
                c = (cx + ox, cz + oz)
                if not self.inside(*c, 1) or c in cells or c in self.fixed:
                    continue
                if self.use[c] not in (FREE, STREET):
                    continue
                cells[c] = h
        for c, h in cells.items():
            self.G[c] = h
            self.use[c] = STREET
            self.fixed.add(c)
            self.surface[c] = self.pick(mix)
        for c in line:
            self.fixed.add(c)
        st = {"name": name, "line": line, "heights": hs, "cells": cells}
        self.streets.append(st)
        return st

    def ring_lines(self, c: Dict[str, Any], gap: int = 2, margin: int = 2) -> List[List[Cell]]:
        """The loop of cells ``gap`` outside the castle platform, split into
        two lines that start at the gate and meet behind the castle, each cut
        where it leaves the map."""
        x0, z0 = c["x"] - 1 - gap, c["z"] - 1 - gap
        x1, z1 = c["x"] + c["w"] + gap, c["z"] + c["d"] + gap
        loop = ([(x, z0) for x in range(x0, x1)] + [(x1, z) for z in range(z0, z1)] +
                [(x, z1) for x in range(x1, x0, -1)] + [(x0, z) for z in range(z1, z0, -1)])
        g = c["gate"]
        i0 = min(range(len(loop)), key=lambda i: (loop[i][0] - g[0]) ** 2 + (loop[i][1] - g[1]) ** 2)
        loop = loop[i0:] + loop[:i0]
        half = len(loop) // 2
        out = []
        for line in (loop[:half + 1], [loop[0]] + loop[:half - 1:-1]):
            seg: List[Cell] = []
            for cell in line:
                if self.inside(*cell, margin):
                    seg.append(cell)
                else:
                    break
            out.append(seg)
        return out

    def climb(self, c: Dict[str, Any], radius: float, depth: int = 11) -> Dict[str, Any]:
        """Terraced streets up steep ground beside the castle, joined by
        switchback ramps: a street along the contour every two plot depths
        up the slope, so a row of houses can stand on each side of it, and a
        ramp at alternate ends from each one to the next, its surface rising
        no more than a block a step. Returns the plan, empty when no side of
        the castle rises steeply enough to need it."""
        n, m = self.n, self.m
        cx, cz = c["x"] + c["w"] / 2.0, c["z"] + c["d"] / 2.0
        k = 2
        pad = np.pad(self.H0.astype(float), k, mode="edge")
        Hs = sum(pad[k + i:k + i + n, k + j:k + j + m] for i in range(-k, k + 1) for j in range(-k, k + 1))
        Hs = Hs / float((2 * k + 1) ** 2)
        gap = self.terrace_rows * depth + 2 * int(math.ceil(radius)) + 3 + (2 if self.terrace_rows == 1 else 0)

        def cell(o: Cell, a: Cell, u: Cell, sa: int, su: int) -> Cell:
            return (o[0] + a[0] * sa + u[0] * su, o[1] + a[1] * sa + u[1] * su)

        # Which side rises: the one whose ground climbs most over two terraces.
        best = None
        for name in _FRONTS:
            u = DIRS[name]
            u = (int(u[0]), int(u[1]))
            a = (-u[1], u[0])
            # The first cell past the castle's platform, mid-way along that side.
            if u[0]:
                o = (c["x"] - 2 if u[0] < 0 else c["x"] + c["w"] + 1, int(math.floor(cz)))
            else:
                o = (int(math.floor(cx)), c["z"] - 2 if u[1] < 0 else c["z"] + c["d"] + 1)
            span = int((c["d"] if u[0] else c["w"]) / 2)
            near, far = [], []
            for sa in range(-span, span + 1, 3):
                p0, p1 = cell(o, a, u, sa, 2), cell(o, a, u, sa, 2 * gap)
                if self.inside(*p0) and self.inside(*p1):
                    near.append(Hs[p0])
                    far.append(Hs[p1])
            if len(near) < 4:
                continue
            rise = float(np.mean(far) - np.mean(near))
            if best is None or rise > best[0]:
                best = (rise, u, a, o)
        if best is None or best[0] < 14:
            return {}
        _, u, a, o = best

        legs: List[Dict[str, Any]] = []
        prev_su: Dict[int, int] = {}
        for kk in range(12):
            su_k = depth + 2 * int(math.ceil(radius)) + 2 + kk * gap
            ref = [Hs[cell(o, a, u, sa, su_k)] for sa in range(-20, 21, 2) if self.inside(*cell(o, a, u, sa, su_k), 4)]
            if len(ref) < 8:
                break
            h_k = int(round(float(np.median(ref))))
            # Follow the contour at h_k out from the middle both ways.
            cols: Dict[int, int] = {}
            for step in (1, -1):
                sa = 0 if step == 1 else -1
                while True:
                    if sa in prev_su:     # keep a full two plots above the terrace below
                        lo, hi = prev_su[sa] + gap - 1, prev_su[sa] + gap + 5
                    else:
                        lo, hi = su_k - gap // 3, su_k + gap // 3
                    found = None
                    for su in range(lo, hi + 1):
                        p = cell(o, a, u, sa, su)
                        if not self.inside(*p, 4):
                            break
                        if Hs[p] >= h_k:
                            found = su
                            break
                    if found is None:
                        break
                    p = cell(o, a, u, sa, found)
                    if self.use[p] != FREE:
                        break
                    cols[sa] = found
                    sa += step
            if len(cols) < 24:
                break
            keys = sorted(cols)
            vals = np.array([cols[q] for q in keys], dtype=float)
            w = 4
            sm = [int(round(vals[max(0, i - w):i + w + 1].mean())) for i in range(len(vals))]
            line: List[Cell] = []
            for q, su in zip(keys, sm):
                if line:
                    last_su = prev_line_su
                    while abs(su - last_su) > 1:
                        last_su += 1 if su > last_su else -1
                        line.append(cell(o, a, u, q, last_su))
                line.append(cell(o, a, u, q, su))
                prev_line_su = su
            line = [p for p in line if self.inside(*p, 3) and self.use[p] == FREE]
            if len(line) < 24:
                break
            st = self.lay_street(line, radius, _SIDE, h_k, f"terrace {kk}", heights=[h_k] * len(line))
            if not st:
                break
            legs.append({"street": st, "h": h_k, "line": line})
            for q, su in zip(keys, sm):
                prev_su[q] = su
        if not legs:
            return {}

        # Switchbacks: a ramp at alternate ends from each terrace to the next.
        ramps = []
        for kk in range(len(legs) - 1):
            lo_leg, hi_leg = legs[kk], legs[kk + 1]
            end = kk % 2
            l0 = lo_leg["line"] if end == 0 else lo_leg["line"][::-1]
            l1 = hi_leg["line"] if end == 0 else hi_leg["line"][::-1]
            dh = hi_leg["h"] - lo_leg["h"]
            done = False
            off = int(math.ceil(radius)) + 1
            for i in range(2, min(len(l0), 60), 3):
                for j in (2, 5, 8):
                    if j >= len(l1):
                        continue
                    # From just above the lower terrace to just below the upper.
                    s0 = (l0[i][0] + u[0] * off, l0[i][1] + u[1] * off)
                    s1 = (l1[j][0] - u[0] * off, l1[j][1] - u[1] * off)
                    if not (self.inside(*s0, 2) and self.inside(*s1, 2)):
                        continue
                    if self.use[s0] != FREE or self.use[s1] != FREE:
                        continue
                    path = self.route(s0, s1, {FREE})
                    if len(path) - 1 >= abs(dh) + 2:
                        hs = [int(round(lo_leg["h"] + dh * t / (len(path) - 1))) for t in range(len(path))]
                        st = self.lay_street(path, radius, _SIDE, lo_leg["h"], f"switchback {kk}", heights=hs)
                        if st:
                            ramps.append(st)
                            done = True
                        break
                if done:
                    break
        # The lowest terrace joins the streets below at its far end, the
        # highest at the other end goes on round if it can.
        first = legs[0]
        joined = False
        ends = [first["line"][-3], first["line"][2]] if len(legs) > 1 and (len(legs) - 1) % 2 == 0 else \
            [first["line"][2], first["line"][-3]]
        others = [(int(cc[0]), int(cc[1])) for cc in zip(*np.nonzero(self.use == STREET))]
        others = [cc for cc in others if cc not in first["street"]["cells"] and
                  any(self.inside(cc[0] + dx, cc[1] + dz) and self.use[cc[0] + dx, cc[1] + dz] == FREE
                      for dx, dz in _N4)]
        off = int(math.ceil(radius)) + 1
        for e in ends:
            if joined or not others:
                break
            e = (e[0] - u[0] * off, e[1] - u[1] * off)
            if not self.inside(*e, 2) or self.use[e] != FREE:
                continue
            cand = sorted(others, key=lambda cc: (cc[0] - e[0]) ** 2 + (cc[1] - e[1]) ** 2)[:6]
            for cc in cand:
                goal = cc
                path = self.route(e, goal, {FREE})
                if len(path) < 3:
                    continue
                dh = int(self.G[goal]) - first["h"]
                if len(path) - 1 < abs(dh) + 1:
                    continue
                hs = [int(round(first["h"] + dh * t / (len(path) - 1))) for t in range(len(path))]
                st = self.lay_street(path[:-1], radius, _SIDE, first["h"], "terrace stair", heights=hs[:-1])
                if st:
                    ramps.append(st)
                    joined = True
                    break
        # The land the terraces climb: houses there may stand on deeper cuts.
        zone = set()
        top = max(max(abs((p[0] - o[0]) * u[0] + (p[1] - o[1]) * u[1]) for p in L["line"]) for L in legs)
        for x in range(n):
            for z in range(m):
                su = (x - o[0]) * u[0] + (z - o[1]) * u[1]
                if -2 <= su <= top + gap // 2:
                    zone.add((x, z))
        self.steep |= zone
        self.b.notes.append(f"town: {len(legs)} terraces up the {_FRONTS[[DIRS[f] for f in _FRONTS].index(u)]} "
                            f"side, {len(ramps)} switchbacks")
        return {"legs": legs, "ramps": ramps, "up": u, "joined": joined}

    def street_stairs(self) -> int:
        """A stair block on every street or square cell that sits one block
        below its neighbour, facing up the rise, so the street is walked
        rather than jumped."""
        placed = 0
        for x in range(self.n):
            for z in range(self.m):
                if self.use[x, z] not in (STREET, PLAZA):
                    continue
                h = int(self.G[x, z])
                for name in _FRONTS:
                    dx, dz = DIRS[name]
                    nx, nz = x + dx, z + dz
                    if not self.inside(nx, nz) or self.use[nx, nz] not in (STREET, PLAZA, CASTLE, PAD):
                        continue
                    if int(self.G[nx, nz]) == h + 1 and (x, h, z) not in self.b.blocks:
                        block = "cobblestone_stairs" if self.use[x, z] == STREET else "stone_brick_stairs"
                        self.b.blocks[(x, h, z)] = state(block, facing=name, half="bottom")
                        placed += 1
                        break
        return placed

    # ------------------------------------------------------------ houses
    def place_houses(self, pool: List[BlockBuild], limit: int, gate: Cell, only: Optional[str] = None) -> int:
        cache: Dict[Tuple[int, int], Dict[str, Any]] = {}

        def variant(i: int, q: int) -> Dict[str, Any]:
            if (i, q) not in cache:
                t = turned(pool[i], q)
                door = t.ports[0]
                fx, fz = int(round(door["facing"][0])), int(round(door["facing"][2]))
                dcx, dcz = int(math.floor(door["position"][0])), int(math.floor(door["position"][2]))
                base = {(x, z) for (x, y, z) in t.blocks if y == 0}
                foot = {(x, z) for (x, _, z) in t.blocks}
                floor = {(x, z) for (x, y, z), blk in t.blocks.items()
                         if y == 0 and ("_planks" in blk or blk.split("[")[0].split(":")[-1] in INTERIOR_FLOORS)}
                # A shop with its door on the street line is set with the
                # door itself on the first cell off the street.
                k_ = 0 if getattr(pool[i], "street_door", 0) else 1
                cache[(i, q)] = {"t": t, "stair": (dcx + fx * k_, dcz + fz * k_), "base": base, "foot": foot,
                                 "floor": floor}
            return cache[(i, q)]

        fronts = []
        for x in range(self.n):
            for z in range(self.m):
                if self.use[x, z] not in (STREET, PLAZA):
                    continue
                for name in _FRONTS:
                    dx, dz = DIRS[name]
                    s = (x + dx, z + dz)
                    if self.inside(*s, 2) and self.use[s] == FREE:
                        d = math.hypot(x - gate[0], z - gate[1])
                        fronts.append((d + self.rng.random() * 6, (x, z), name))
        fronts.sort()
        placed = 0
        # Plot roles: a workshop near the square, a tower high over the
        # streets, a windmill at the edge of town; houses everywhere.
        kinds = [getattr(p, "kind", "house") for p in pool]
        plaza_d = self.distance_to((PLAZA,))
        street_h = sorted(int(self.G[c]) for c in zip(*np.nonzero(self.use == STREET)))
        high = street_h[int(.8 * (len(street_h) - 1))] if street_h else 0
        far = sorted(f[0] for f in fronts)[int(.5 * (len(fronts) - 1))] if fronts else 0
        main_cells = set(self.streets[0]["cells"]) if self.streets else set()
        lane = self.b.sizes.lane

        S = self.b.sizes.storey

        def wanted(e: Cell, s: Cell, d: float, out: str) -> List[str]:
            want = []
            # Flats where a cliff rises behind the plot: the ground a house's
            # depth back stands two storeys or more over the street.
            ox_, oz_ = DIRS[out]
            behind = [int(self.G[s[0] + ox_ * k, s[1] + oz_ * k]) for k in range(5, 11)
                      if self.inside(s[0] + ox_ * k, s[1] + oz_ * k)]
            if behind and max(behind) - int(self.G[e]) >= 2 * S:
                want.append("apartment")
                self.cliff_fronts += 1
            if plaza_d[e] <= 26 or e in main_cells or (s not in self.steep and self.rng.random() < .3):
                want.append("workshop")
            # Merchants' houses line the square and the main street, gable
            # to the street in a continuous front, as in any market town.
            if plaza_d[e] <= 30 or e in main_cells:
                want.append("townhouse")
            if int(self.G[e]) >= high:
                want.append("tower")
            if d >= far and s not in self.steep:
                want.append("windmill")
            # A seld fits any street front left over: one bay deep, it stands
            # in a strip too shallow for a house, its back to the bank.
            want.append("shop")
            return [k for k in want if len(self.kinds.get(k, [])) < _KIND_QUOTA[k] and
                    all(math.hypot(s[0] - o[0], s[1] - o[1]) >= _KIND_SPACING[k] for o in self.kinds.get(k, []))]

        for dist, e, out in fronts:
            if placed >= limit:
                break
            dx, dz = DIRS[out]
            s = (e[0] + dx, e[1] + dz)
            if self.use[s] != FREE:
                continue
            q = _FRONTS.index({"north": "south", "south": "north", "east": "west", "west": "east"}[out])
            P = int(self.G[e])
            want = wanted(e, s, dist, out)
            if only is not None:
                # A first pass for one type: only fronts that want it.
                if only not in want:
                    continue
                want = [only]
            else:
                want = [k for k in want if k != "shop"]   # selds only fill what is left
            first = [i for i in self.rng.sample(range(len(pool)), len(pool)) if kinds[i] in want]
            rest = [] if only is not None else [i for i in self.rng.sample(range(len(pool)), len(pool))
                                                 if kinds[i] == "house"]
            for i in first + rest:
                v = variant(i, q)
                ox, oz = s[0] - v["stair"][0], s[1] - v["stair"][1]
                foot = {(x + ox, z + oz) for (x, z) in v["foot"]}
                base = {(x + ox, z + oz) for (x, z) in v["base"]}
                if not all(self.inside(x, z, 1) for (x, z) in foot):
                    self.rejects[(kinds[i], "map edge")] += 1
                    continue
                if not all(self.use[c] == FREE for c in base):
                    self.rejects[(kinds[i], "ground taken")] += 1
                    continue
                # A door on the street line opens straight onto the street:
                # nothing may stand on the street cell before it (a lamp post
                # or a stall on the square).
                if getattr(pool[i], "street_door", 0) and any(
                        self.b.blocks.get((e[0], y, e[1])) is not None for y in range(P + 1, P + 4)):
                    self.rejects[(kinds[i], "door blocked")] += 1
                    continue
                # Eaves may hang over a street or square, or meet the eaves of a
                # house on the same terrace (a row: the roofs run on together
                # over a drip gap), never over another house's walls.
                if not all((self.use[c] in (FREE, STREET, PLAZA) and
                            (self.use[c] == FREE or int(self.G[c]) <= P + 1)) or abs(self.eaves.get(c, -99) - P) <= 2
                           for c in foot - base):
                    self.rejects[(kinds[i], "overhang")] += 1
                    continue
                pad = set()
                for (x, z) in base:
                    for ddx in (-1, 0, 1):
                        for ddz in (-1, 0, 1):
                            pad.add((x + ddx, z + ddz))
                pad = {c for c in pad if self.inside(*c, 1) and (c in foot or self.use[c] == FREE)}
                # The ground is cut to the floor under the house itself and
                # levelled before its door. Elsewhere it is left alone where
                # it stands off the floor by more than two blocks: rising
                # ground stays as rock against the walls (the house is set
                # into the slope), falling ground stays below it, and the
                # house stands out over it on timber stilts, or, more than
                # five blocks down, on a crag of rock raised to its floor.
                front = {c for c in pad | foot if max(abs(c[0] - s[0]), abs(c[1] - s[1])) <= lane + 1}
                cut_lim = 13 if s in self.steep else 6
                if kinds[i] == "apartment":
                    # Flats are dug into the face they stand against.
                    cut_lim = max(cut_lim, 3 * S)
                levelled, stilted, embedded, crag = set(), set(), set(), set()
                ok = True
                for c in pad | foot:
                    g = int(self.G[c])
                    if c in front or abs(g - P) <= 2:
                        if abs(g - P) > cut_lim:
                            ok = False
                            break
                        levelled.add(c)
                    elif g > P:
                        if c in base:
                            if g - P > cut_lim:
                                ok = False
                                break
                            levelled.add(c)
                        else:
                            embedded.add(c)
                    elif P - g <= 5:
                        stilted.add(c)
                    else:
                        # Too far down for timber legs: the house sits on a
                        # crag of the hill's rock raised to its floor.
                        if P - g > 18:
                            ok = False
                            break
                        levelled.add(c)
                        crag.add(c)
                if not ok:
                    self.rejects[(kinds[i], "slope")] += 1
                    continue
                if any(c in self.under and P - 1 < self.under[c] + 2 for c in levelled):
                    continue
                for c in foot - base:
                    if self.use[c] == FREE:
                        self.eaves[c] = P
                foot = {c for c in foot if self.use[c] == FREE}
                self.level(list(levelled), P)
                for c in pad:
                    if self.use[c] in (STREET, PLAZA):
                        continue          # an eave over the street leaves it a street
                    self.use[c] = PAD
                    self.fixed.add(c)
                for c in foot:
                    self.use[c] = HOUSE
                    self.fixed.add(c)
                if not getattr(pool[i], "street_door", 0):
                    self.surface[s] = state("dirt_path")
                n_ports = len(self.b.ports)
                self.absorb(v["t"], ox, oz, P)
                # Stilts: a log post down to the ground at the corners and
                # every third block round a stilted floor, braced with a
                # fence rail under its edge.
                legs = 0
                for c in sorted(stilted & base):
                    edge = any((c[0] + a, c[1] + b_) not in base for a, b_ in _N4)
                    if not edge:
                        continue
                    corner = sum((c[0] + a, c[1] + b_) not in base for a, b_ in _N4) >= 2
                    g = int(self.G[c])
                    if corner or (c[0] + c[1]) % 3 == 0:
                        for y in range(g, P):
                            self.b.blocks[(c[0], y, c[1])] = state("spruce_log", axis="y")
                        legs += 1
                    else:
                        self.b.blocks.setdefault((c[0], P - 1, c[1]), state("spruce_fence"))
                k = kinds[i]
                if k == "apartment":
                    self.back_door(base, out, P, getattr(pool[i], "storeys", 1), foot)
                self.kinds.setdefault(k, []).append(s)
                self.stilts += int(legs > 0)
                self.embedded += int(bool(embedded & pad))
                self.crags += int(bool(crag))
                self.houses.append({"cells": sorted((x + ox, z + oz) for (x, z) in v["floor"]), "floor": P,
                                    "door": self.b.ports[n_ports]["position"], "variant": i, "kind": k,
                                    "stilts": legs, "embedded": len(embedded), "crag": len(crag),
                                    "storeys": pool[i].notes[-1] if pool[i].notes else "",
                                    "front": e, "street_door": getattr(pool[i], "street_door", 0)})
                placed += 1
                break
        return placed

    def back_door(self, base: Set[Cell], out: str, P: int, storeys: int, own: Set[Cell] = frozenset()) -> int:
        """A door in the back wall of a house of flats built against a rock
        face, on the highest storey whose floor meets the ground on top of the
        face, with a landing in front of it (on its own eaves' cells or open ground, cut or filled
        two blocks at most). Returns the storey, or 0 for none."""
        sz = self.b.sizes
        S, dw, dh = sz.storey, sz.door_width, sz.door_height
        ox, oz = DIRS[out]
        proj = {c: c[0] * ox + c[1] * oz for c in base}
        back = max(proj.values())
        row = sorted((c for c in base if proj[c] == back), key=lambda c: c[0] * -oz + c[1] * ox)
        blocks = self.b.blocks
        # Windows in the back wall that the rock now covers are walled up.
        for c in row:
            g = int(self.G[c[0] + ox, c[1] + oz]) if self.inside(c[0] + ox, c[1] + oz) else 0
            for y in range(P, g):
                blk = blocks.get((c[0], y, c[1]))
                if blk is not None and "glass" in blk:
                    blocks[(c[0], y, c[1])] = state("stone_bricks")
        for k in range(storeys - 1, 0, -1):
            stand = P + k * S + 1
            for j in range(1, len(row) - dw):
                cells = row[j:j + dw]
                inside = [(c[0] - ox, c[1] - oz) for c in cells]
                land = [(c[0] + ox * d, c[1] + oz * d) for c in cells for d in (1, 2)]
                if not all(self.inside(*c, 1) for c in land):
                    continue
                # The landing is open ground at the floor (cut or filled a
                # block at most), or a street or square that is already
                # there: level with the floor or a block below it.
                road = [self.use[c] in (STREET, PLAZA) for c in land]
                if any(road[:dw]):
                    if not all(r and int(self.G[c]) in (stand - 1, stand) for r, c in zip(road[:dw], land[:dw])):
                        continue
                    land = land[:dw]
                elif not all((self.use[c] in (FREE, PAD) or c in own) and c not in self.under and
                             abs(int(self.G[c]) - stand) <= 2 for c in land):
                    continue
                # The room inside is clear floor at the door, and nothing of
                # the house stands outside it.
                if not all(blocks.get((x, stand - 1, z)) is not None and
                           all((x, y, z) not in blocks for y in range(stand, stand + dh))
                           for (x, z) in inside):
                    continue
                # Shutters and window boxes on a wall that now faces the rock
                # make way for the landing; anything else built there stops it.
                outside = [(x, y, z) for (x, z) in land for y in range(stand, stand + dh) if (x, y, z) in blocks]
                if not all(any(w in blocks[p_] for w in _TRIM) for p_ in outside):
                    continue
                for p_ in outside:
                    blocks.pop(p_)
                for (x, z) in cells:
                    for y in range(stand, stand + dh):
                        blocks.pop((x, y, z), None)
                    blocks[(x, stand - 1, z)] = state("spruce_planks")
                if not any(road[:dw]):
                    self.level(land, stand)
                    for c in land:
                        if c not in own:
                            self.use[c] = PAD
                        self.fixed.add(c)
                        self.surface[c] = state("dirt_path")
                self.back_doors += 1
                return k
        return 0

    # ------------------------------------------------------------ the mine
    def place_mine(self, part: BlockBuild) -> Dict[str, Any]:
        """Dig ``part`` (a ``blockkinds.mine``, entrance facing north at
        z = 0) into the hill off a street: the entrance one cell back from
        the street, its floor at the street's height, the rock over every
        passage at least a block under the ground. Picks the street end
        where the hill rises highest over it. Returns {} if none will do."""
        info = part.mine
        w, H = info["width"], info["clear"]
        cols: Dict[Cell, Tuple[int, int]] = {}
        for (x, y, z) in part.blocks:
            lo, hi = cols.get((x, z), (y, y))
            cols[(x, z)] = (min(lo, y), max(hi, y))
        ax = w // 2
        best: Optional[Tuple[float, Cell, int, int, int, int]] = None
        for x in range(2, self.n - 2):
            for z in range(2, self.m - 2):
                if self.use[x, z] != STREET:
                    continue
                for name in _FRONTS:
                    dx, dz = DIRS[name]
                    s = (x + dx, z + dz)
                    if not self.inside(*s, 3) or self.use[s] != FREE:
                        continue
                    P = int(self.G[x, z])
                    q = _FRONTS.index({"north": "south", "south": "north", "east": "west", "west": "east"}[name])
                    tx, tz = _turn_xz(ax + .5, .5, q)
                    ox = s[0] + dx - int(math.floor(tx))
                    oz = s[1] + dz - int(math.floor(tz))
                    margin, ok = 10 ** 6, True
                    for (lx, lz), (lo, hi) in cols.items():
                        cx, cz = _turn_xz(lx + .5, lz + .5, q)
                        c = (int(math.floor(cx)) + ox, int(math.floor(cz)) + oz)
                        if not self.inside(*c, 2):
                            ok = False
                            break
                        if lz < 0:
                            if self.use[c] != FREE or abs(int(self.G[c]) - P) > 2:
                                ok = False
                                break
                            continue
                        if self.use[c] in (CASTLE, PLAZA, HOUSE, PAD) or c in self.under:
                            ok = False
                            break
                        if lz < 4:
                            if self.use[c] != FREE:
                                ok = False
                                break
                            continue
                        top = P - 1 + hi
                        if self.use[c] == STREET and int(self.G[c]) < top + 3:
                            ok = False
                            break
                        margin = min(margin, int(self.G[c]) - (top + 2))
                        if margin < 0:
                            ok = False
                            break
                    if ok and (best is None or margin > best[0]):
                        best = (margin, s, q, ox, oz, P)
        if best is None:
            self.b.notes.append("town: no hillside off a street high enough over it for a mine")
            return {}
        margin, s, q, ox, oz, P = best
        paste(self.b, part, ox, oz, q, dy=P - 1)
        for (lx, lz), (lo, hi) in cols.items():
            cx, cz = _turn_xz(lx + .5, lz + .5, q)
            c = (int(math.floor(cx)) + ox, int(math.floor(cz)) + oz)
            if lz < 0:
                self.G[c] = P
                self.use[c] = PAD
                self.fixed.add(c)
                self.surface[c] = state("gravel")
            else:
                self.under[c] = P - 1 + hi
                if lz < 4 and self.use[c] == FREE:
                    self.use[c] = PAD
        self.G[s] = P
        self.use[s] = PAD
        self.surface[s] = state("gravel")
        walk = []
        for (x, y, z) in info["walk"]:
            cx, cz = _turn_xz(x + .5, z + .5, q)
            walk.append((int(math.floor(cx)) + ox, y + P - 1, int(math.floor(cz)) + oz))
        out = {"entrance": s, "floor": P, "walk": walk, "cover_margin": margin, "levels": info["levels"],
               "galleries": info["galleries"], "stairs": info["stairs"]}
        self.b.notes.append(f"town: a mine dug into the hill off the street at {s}, {info['levels']} level(s), "
                            f"rock at least {margin + 1} block(s) thick over it")
        return out

    # ------------------------------------------------------------ the rest
    def distance_to(self, uses: Sequence[int]) -> np.ndarray:
        dist = np.full(self.G.shape, 10 ** 6, dtype=int)
        q: List[Cell] = []
        for x in range(self.n):
            for z in range(self.m):
                if self.use[x, z] in uses:
                    dist[x, z] = 0
                    q.append((x, z))
        head = 0
        while head < len(q):
            x, z = q[head]
            head += 1
            for dx, dz in _N4:
                nx, nz = x + dx, z + dz
                if self.inside(nx, nz) and dist[nx, nz] > dist[x, z] + 1:
                    dist[nx, nz] = dist[x, z] + 1
                    q.append((nx, nz))
        return dist

    def place_town_wall(self, reach: int = 10, height: int = 5, tower_every: int = 16) -> Dict[str, int]:
        """A town wall where the town meets the edge of the build area.

        Where a street runs so near the edge that no plot fits between it and
        the edge (open ground runs out within ``reach`` blocks), that side of
        the street becomes a wall street: the town wall stands on the strip
        right beside it, two blocks thick, its walk ``height`` blocks up and
        merlons on the outer face, with a tower every ``tower_every`` blocks.
        Streets that run out through the strip leave gaps for gates."""
        strip: Dict[Cell, Tuple[int, Cell]] = {}
        for x in range(self.n):
            for z in range(self.m):
                if self.use[x, z] != STREET:
                    continue
                for d in _N4:
                    run = []
                    k = 1
                    while True:
                        c = (x + d[0] * k, z + d[1] * k)
                        if not self.inside(*c, 1):
                            break
                        if self.use[c] != FREE or k > reach:
                            run = []
                            break
                        run.append(c)
                        k += 1
                    for k, c in enumerate(run[:2], 1):
                        if c not in strip or strip[c][0] > k:
                            strip[c] = (k, d)
        if not strip:
            return {"cells": 0, "towers": 0}
        body = (("stone_bricks", 6), ("cobblestone", 2), ("mossy_stone_bricks", 1), ("cracked_stone_bricks", 1))
        towers = set()
        for (x, z), (k, d) in strip.items():
            along = x if d[0] == 0 else z
            if along % tower_every in (tower_every // 2, tower_every // 2 + 1):
                towers.add((x, z))
        for (x, z), (k, d) in strip.items():
            g = int(self.G[x, z])
            # The outer face: no wall cell further out on this line.
            outer = (x + d[0], z + d[1]) not in strip
            top = g + height + (3 if (x, z) in towers else 0)
            for y in range(g, top):
                self.b.blocks[(x, y, z)] = self.pick(body)
            # A string course of slabs under the walk shows the wall's line from the street.
            if not outer:
                self.b.blocks[(x, g + 2, z)] = state("stone_bricks")
            if outer and (x + z) % 2 == 0:
                self.b.blocks[(x, top, z)] = state("stone_bricks")
            elif outer:
                self.b.blocks[(x, top, z)] = state("stone_brick_slab", type="bottom")
            self.use[x, z] = CASTLE
            self.fixed.add((x, z))
            self.surface[(x, z)] = state("stone_bricks")
        return {"cells": len(strip), "towers": len(towers)}

    def place_fields(self, limit: int = 8) -> int:
        dist = self.distance_to((STREET,))
        placed = 0
        crops = ("wheat", "wheat", "carrots", "potatoes")
        for _ in range(400):
            if placed >= limit:
                break
            w, d = self.rng.randint(7, 10), self.rng.randint(7, 11)
            x0, z0 = self.rng.randint(2, self.n - w - 3), self.rng.randint(2, self.m - d - 3)
            box = [(x, z) for x in range(x0 - 1, x0 + w + 1) for z in range(z0 - 1, z0 + d + 1)]
            if not all(self.use[c] == FREE for c in box):
                continue
            if min(dist[c] for c in box) > 10:
                continue
            hs = [int(self.G[c]) for c in box]
            if max(hs) - min(hs) > 3:
                continue
            h = int(np.median(hs))
            if any(c in self.under and h - 1 < self.under[c] + 2 for c in box):
                continue
            self.level(box, h)
            crop = self.rng.choice(crops)
            part = BlockBuild("field", self.b.player, self.b.seed)
            mid = z0 + d // 2
            for (x, z) in box:
                self.use[x, z] = FIELD
                if x in (x0 - 1, x0 + w) or z in (z0 - 1, z0 + d):
                    if not (z == mid and x in (x0 - 1, x0 + w)):
                        part.set(x, 0, z, state("oak_fence"))
                    continue
                if z == mid:
                    part.path(x, z, "water")
                    continue
                part.path(x, z, "farmland")
                part.set(x, 0, z, state(crop, age=self.rng.choice((4, 5, 6, 7, 7))))
            self.absorb(part, 0, 0, h)
            placed += 1
        return placed

    def place_lamps(self, spacing: int = 8) -> int:
        lamps: List[Cell] = []
        for st in self.streets:
            for c in st["line"][2::spacing]:
                for dx, dz in _N4:
                    for r in range(1, 5):
                        s = (c[0] + dx * r, c[1] + dz * r)
                        if not self.inside(*s, 1):
                            break
                        if self.use[s] == STREET:
                            continue
                        if self.use[s] in (FREE, PAD) and all(math.hypot(s[0] - l[0], s[1] - l[1]) > spacing - 2
                                                              for l in lamps):
                            part = BlockBuild("lamp", self.b.player, self.b.seed)
                            lamp_post(part, s[0], s[1], 4)
                            self.absorb(part, 0, 0, int(self.G[s]))
                            self.use[s] = GREEN
                            lamps.append(s)
                        break
                    else:
                        continue
                    break
        return len(lamps)

    def dress_streets(self, spacing: int = 5, wells: int = 3) -> Dict[str, int]:
        """Street life on the free ground beside the streets, never on a
        street or a door's landing: stacked barrels and crates, benches
        facing the street, hay and carts at workshops' ends, flower tubs, and
        a few wells on level ground off the streets."""
        # A door's landing and the cells round it stay clear; the rest of a
        # house's yard may take a bench or a barrel against its wall.
        near_door = set()
        for h in self.houses:
            dx_, dz_ = int(math.floor(h["door"][0])), int(math.floor(h["door"][2]))
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    near_door.add((dx_ + dx, dz_ + dz))
        placed: List[Cell] = []
        counts = {"props": 0, "benches": 0, "wells": 0}
        flowers = ("red_tulip", "dandelion", "cornflower", "oxeye_daisy", "azure_bluet", "fern")
        for st in self.streets:
            for c in st["line"][1::spacing]:
                for dx, dz in self.rng.sample(_N4, 4):
                    for r in range(1, 5):
                        cell = (c[0] + dx * r, c[1] + dz * r)
                        if not self.inside(*cell, 1) or self.use[cell] == STREET:
                            continue
                        break
                    else:
                        continue
                    if (not self.inside(*cell, 1) or self.use[cell] not in (FREE, PAD) or cell in near_door
                            or any(abs(cell[0] - p_[0]) + abs(cell[1] - p_[1]) < 3 for p_ in placed[-40:])):
                        continue
                    # The street cell beside it must be level with it, so a bench faces a pavement.
                    st_cell = (cell[0] - dx, cell[1] - dz)
                    if abs(int(self.G[cell]) - int(self.G[st_cell])) > 0:
                        continue
                    part = BlockBuild("prop", self.b.player, self.b.seed)
                    x, z = cell
                    roll = self.rng.random()
                    if roll < .3:
                        part.set(x, 0, z, state("spruce_stairs", facing=_FACE[(dx, dz)], half="bottom"))
                        counts["benches"] += 1
                    elif roll < .55:
                        part.set(x, 0, z, state("barrel", facing="up"))
                        if self.rng.random() < .6:
                            part.set(x, 1, z, state(self.rng.choice(("barrel", "hay_block", "composter"))))
                    elif roll < .75:
                        part.set(x, 0, z, state("hay_block", axis="y"))
                        if self.rng.random() < .5:
                            part.set(x, 1, z, state("hay_block", axis="x"))
                    else:
                        part.path(x, z, "coarse_dirt")
                        part.set(x, 0, z, state("potted_" + self.rng.choice(flowers))
                                 if self.rng.random() < .3 else state(self.rng.choice(flowers)))
                    self.absorb(part, 0, 0, int(self.G[cell]))
                    self.use[cell] = GREEN
                    placed.append(cell)
                    counts["props"] += 1
                    break
        # Wells: a 5 by 5 patch of free, level-ish ground beside a street.
        for st in self.streets:
            if counts["wells"] >= wells:
                break
            line = st["line"]
            for c in line[len(line) // 3: len(line) - 3: 7]:
                done = False
                for dx, dz in _N4:
                    cx, cz = c[0] + dx * 4, c[1] + dz * 4
                    patch = [(cx + i, cz + j) for i in range(-2, 3) for j in range(-2, 3)]
                    if not all(self.inside(*p_, 2) and self.use[p_] == FREE and p_ not in near_door for p_ in patch):
                        continue
                    hs = [int(self.G[p_]) for p_ in patch]
                    h = int(self.G[(cx - dx * 2, cz - dz * 2)])
                    if max(hs) - min(hs) > 1 or abs(h - int(self.G[c])) > 1:
                        continue
                    self.level(patch, h)
                    for p_ in patch:
                        self.surface[p_] = self.pick(_SIDE)
                        self.use[p_] = GREEN
                    part = BlockBuild("well", self.b.player, self.b.seed)
                    well(part, cx - 1, cz - 1)
                    self.absorb(part, 0, 0, h)
                    counts["wells"] += 1
                    done = True
                    break
                if done:
                    break
        return counts

    def place_greenery(self) -> Tuple[int, int]:
        dist = self.distance_to((STREET, PLAZA, HOUSE, PAD, CASTLE))
        trees: List[Cell] = []
        plants = 0
        flowers = ("poppy", "dandelion", "cornflower", "oxeye_daisy", "allium", "azure_bluet")
        hi = float(np.percentile(self.G, 80))
        for x in range(2, self.n - 2):
            for z in range(2, self.m - 2):
                if self.use[x, z] != FREE:
                    continue
                d = dist[x, z]
                p = .015 if d < 3 else (.05 if d < 8 else .11)
                if self.rng.random() < p and all(math.hypot(x - t[0], z - t[1]) >= 4 for t in trees[-60:]):
                    near = [(x + dx, z + dz) for dx in range(-2, 3) for dz in range(-2, 3)]
                    if all(self.inside(*c) and self.use[c] in (FREE, GREEN) for c in near):
                        wood = "spruce" if self.G[x, z] >= hi else self.rng.choice(("oak", "oak", "birch", "spruce"))
                        part = BlockBuild("tree", self.b.player, self.b.seed * 7 + x * 131 + z)
                        tree(part, x, z, wood=wood)
                        h = int(self.G[x, z])
                        part.blocks = {(bx, by, bz): blk for (bx, by, bz), blk in part.blocks.items()
                                       if not self.inside(bx, bz) or by + h >= self.G[bx, bz]}
                        self.absorb(part, 0, 0, h)
                        self.use[x, z] = GREEN
                        trees.append((x, z))
                        continue
                if (x, z) in self.steep and self.rng.random() < .22 and (x, int(self.G[x, z]), z) not in self.b.blocks:
                    # Terrace gardens: bushes of leaves along the walls.
                    leaf = self.rng.choice(("oak_leaves", "azalea_leaves", "flowering_azalea_leaves",
                                            "dark_oak_leaves", "spruce_leaves"))
                    self.b.blocks[(x, int(self.G[x, z]), z)] = state(leaf, persistent="true", distance="1")
                    plants += 1
                    continue
                if self.rng.random() < .10 and (x, int(self.G[x, z]), z) not in self.b.blocks:
                    plant = self.rng.choice(flowers) if self.rng.random() < .35 else "short_grass"
                    self.b.blocks[(x, int(self.G[x, z]), z)] = state(plant)
                    plants += 1
        return len(trees), plants

    # ------------------------------------------------------------ ground
    def lay_ground(self) -> int:
        """Turn the height map into blocks: a top block, dirt under it, stone
        under that; columns that stand two or more above a neighbour show
        rock, or a retaining wall where either side is built on."""
        # Retaining walls hold up streets, the square, fields and the castle;
        # beside houses (set into the slope or standing out on stilts) and on
        # open ground a face is left as the hill's own rock, in strata.
        held = (CASTLE, STREET, PLAZA, FIELD)
        strata = [self.b.mix(*band) for band in _STRATA]
        n = 0
        for x in range(self.n):
            for z in range(self.m):
                g = int(self.G[x, z])
                drop, by_built = 0, self.use[x, z] in held
                for dx, dz in _N4:
                    nx, nz = x + dx, z + dz
                    if self.inside(nx, nz):
                        dd = g - int(self.G[nx, nz])
                        if dd > drop:
                            drop = dd
                        if dd >= 2 and self.use[nx, nz] in held:
                            by_built = True
                top = self.surface.get((x, z), state("grass_block"))
                for y in range(g):
                    if (x, y, z) in self.b.blocks:
                        continue
                    if y == g - 1:
                        block = top
                    elif drop >= 2 and y >= g - drop:
                        block = self.pick(_RETAIN) if by_built else strata[(y // 4) % len(strata)]()
                    elif y >= g - 4:
                        block = state("dirt")
                    else:
                        block = state("stone")
                    self.b.blocks[(x, y, z)] = block
                    n += 1
        return n


# ------------------------------------------------------------------- town
def town(player: Any = "blocky", seed: int = 0, heights: Optional[np.ndarray] = None, terrain: str = "hill",
         size: int = 144, houses: int = 140, with_mine: bool = True,
         street_style: Optional[str] = None, climate: Optional[str] = None) -> BlockBuild:
    """A castle in a dense town, fitted to ``heights`` (or to a sample
    ``terrain`` of ``size`` columns when no map is given). Returns a
    ``BlockBuild`` whose layer 0 is the bottom of the ground; ``b.town``
    holds the plan (castle, square, streets, houses, heights).
    ``street_style`` names the tradition of the merchants' houses on the
    square and main street (one of the style sub-architects); by default the
    seed picks one of the traditions that suit ``climate`` ("temperate",
    "cold", "warm" or "dry"; temperate when not given)."""
    b = BlockBuild("town", player, seed)
    if heights is None:
        heights = sample_heights(terrain, size, seed)
    heights = np.asarray(heights).astype(int)
    if min(heights.shape) < 80:
        raise ValueError("a town needs a height map at least 80 by 80")
    t = _Town(b, heights)
    lane = b.sizes.lane
    c = t.place_castle(castle(player, seed))
    sq = t.place_square(c)

    # The main street from the square down to the edge of the map.
    fx, fz = c["facing"]
    start = (sq["far"][0] + fx, sq["far"][1] + fz)
    edge = []
    for x in range(2, t.n - 2):
        for z in range(2, t.m - 2):
            if x in (2, t.n - 3) or z in (2, t.m - 3):
                vx, vz = x - start[0], z - start[1]
                L = math.hypot(vx, vz) or 1.0
                if (vx * fx + vz * fz) / L > .35:
                    edge.append((int(t.H0[x, z]) + .02 * L, (x, z)))
    edge.sort()
    main: Dict[str, Any] = {}
    for _, goal in edge[:6]:
        line = t.route(start, goal, {FREE, STREET})
        if line:
            main = t.lay_street(line, max(2, lane + 1), _MAIN, sq["T"], "main street")
            b.town_entrance = goal  # type: ignore[attr-defined]
            break
    if not main:
        raise ValueError("no route for a main street from the castle gate")

    # A street round the castle, in two halves from the square to the back,
    # hugging the platform's foot.
    ring_lines = t.ring_lines(c)
    for k, ring in enumerate(ring_lines):
        if len(ring) >= 8:
            t.lay_street(ring, max(1, lane), _SIDE, sq["T"], f"castle lane {k}", trim=4)

    # Where the ground beside the castle rises steeply, terraces climb it,
    # joined by switchbacks.
    climb = t.climb(c, max(1, lane))

    # An outer ring a plot's depth twice over beyond the castle lane, so the
    # streets that radiate out meet it and the land between is built up.
    # It also stays a merchant's plot (14 deep) clear of the square's far side,
    # so the square's rim can be built up with houses that face it.
    outer_gap = max(2 + 2 * 11 + t.rng.randint(0, 3), sq["reach"] + 15)
    for k, ring in enumerate(t.ring_lines(c, gap=outer_gap)):
        if len(ring) >= 12:
            t.lay_street(ring, max(1, lane), _SIDE, int(t.G[ring[0]]), f"outer ring {k}", trim=4)

    # Side streets: along the castle front from the square, and off the main
    # street every dozen blocks, roughly along the slope.
    side_r = max(1, lane)
    branches: List[Tuple[Cell, Cell, int]] = []
    ax, az = sq["along"]
    # They leave from the square's far corners, so its two long sides stay
    # whole for houses that face it.
    for s, (px, pz) in zip((-1, 1), sq["corners"]):
        a0 = (px + ax * s, pz + az * s)
        goal = (a0[0] + ax * s * 26 + fx * 4, a0[1] + az * s * 26 + fz * 4)
        branches.append((a0, goal, sq["T"]))
    for ring in ring_lines:
        for i in range(10, len(ring) - 4, 13):
            r0 = ring[i]
            cx, cz = c["x"] + c["w"] / 2, c["z"] + c["d"] / 2
            vx, vz = r0[0] - cx, r0[1] - cz
            L = math.hypot(vx, vz) or 1.0
            reach = t.rng.randint(30, 44)
            goal = (int(round(r0[0] + vx / L * reach)), int(round(r0[1] + vz / L * reach)))
            if t.use[r0] == STREET:
                branches.append((r0, goal, int(t.G[r0])))
    line = main["line"]
    for i in range(8, len(line) - 6, 11):
        (x0, z0), (x1, z1) = line[max(0, i - 3)], line[min(len(line) - 1, i + 3)]
        dx, dz = x1 - x0, z1 - z0
        L = math.hypot(dx, dz) or 1.0
        for side in (-1, 1):
            nx, nz = -dz / L * side, dx / L * side
            reach = t.rng.randint(18, 30)
            drift = t.rng.uniform(-6, 6)
            goal = (int(round(line[i][0] + nx * reach + dx / L * drift)),
                    int(round(line[i][1] + nz * reach + dz / L * drift)))
            branches.append((line[i], goal, main["heights"][i]))
    count = 0
    for a0, goal, h0 in branches:
        goal = (min(max(goal[0], 3), t.n - 4), min(max(goal[1], 3), t.m - 4))
        if t.use[goal] != FREE or not t.inside(*a0, 1):
            continue
        if t.use[a0] not in (FREE, STREET):
            continue
        line = t.route(a0, goal, {FREE, STREET})
        if len(line) < 8:
            continue
        st = t.lay_street(line, side_r, _SIDE, int(t.G[a0]) if t.use[a0] == STREET else h0, f"street {count}",
                          trim=4)
        count += 1
        # One lane off each side street, halfway along.
        for j in [len(line) // 2]:
            if not (st and len(line) > 16 and t.rng.random() < .8):
                continue
            (x0, z0), (x1, z1) = line[j - 2], line[j + 2]
            dx, dz = x1 - x0, z1 - z0
            L = math.hypot(dx, dz) or 1.0
            side = t.rng.choice((-1, 1))
            goal2 = (int(round(line[j][0] - dz / L * side * 16)), int(round(line[j][1] + dx / L * side * 16)))
            goal2 = (min(max(goal2[0], 3), t.n - 4), min(max(goal2[1], 3), t.m - 4))
            if t.use[goal2] == FREE:
                line2 = t.route(line[j], goal2, {FREE, STREET})
                if len(line2) >= 8:
                    t.lay_street(line2, side_r, _SIDE, st["heights"][j], f"lane {count}", trim=4)
    stairs = t.street_stairs()

    # A mine into the hill, before the houses take the hillside.
    dug = t.place_mine(mine(player, seed * 1000 + 977, hill=False)) if with_mine else {}

    # Houses, most varied near the castle.
    # Half the pool are narrow town houses at the smallest size their stair
    # allows (width and depth 1 ask for the least), so they fill tight plots
    # between the roomier ones. The rest of the pool are the town's other
    # buildings, each placed where its role fits: workshops by the square,
    # towers high over the streets, windmills at the edge of town.
    roofs = ("gable", "cross", "gable", "hip", "cross", "gable", "hip")
    pool = [house(player, seed * 1000 + k, storeys=t.rng.choice((1, 2, 2, 2, 3)), name=f"house{k:02d}",
                  roof=roofs[(seed * 3 + k) % len(roofs)], **({"width": 1, "depth": 1} if k % 2 else {}))
            for k in range(28)]
    for k in range(3):
        pool.append(tower(player, seed * 1000 + 100 + k, name=f"tower{k:02d}"))
    for k in range(4):
        pool.append(workshop(player, seed * 1000 + 200 + k, name=f"workshop{k:02d}"))
    for k in range(2):
        pool.append(windmill(player, seed * 1000 + 300 + k, name=f"windmill{k:02d}"))
    for k in range(8):
        pool.append(apartment(player, seed * 1000 + 400 + k, name=f"flats{k:02d}"))
    # The merchants' houses of the square and main street, from a style
    # sub-architect: one tradition for the town's front.
    if not street_style:
        styles = blockstyles.for_climate(climate or "temperate")
        street_style = styles[seed % len(styles)]
    purpose = blockstyles.purpose_of(street_style)
    for k in range(10):
        p = blockstyles.build(street_style, purpose, player, seed * 1000 + 500 + k)
        p.kind = "townhouse"  # type: ignore[attr-defined]
        pool.append(p)
    # A town of another region is built in that region's way throughout:
    # its houses are the tradition's own, not the English medieval house,
    # and it keeps only the other buildings that belong there.
    home = getattr(lore.TRADITIONS.get(street_style), "climate", "temperate")
    if home != "temperate":
        keep = {"cold": ("tower", "workshop", "windmill", "apartment"),
                "warm": ("workshop", "windmill", "apartment"),
                "dry": ("apartment",)}.get(home, ())
        pool = [p for p in pool if getattr(p, "kind", "house") in keep or getattr(p, "kind", "") == "townhouse"]
        for k in range(20):
            p = blockstyles.build(street_style, purpose, player, seed * 1000 + 600 + k)
            p.kind = "house"  # type: ignore[attr-defined]
            pool.append(p)
    if home == "temperate":
        for k in range(36):
            p = blockstyles.english_shop(player, seed * 1000 + 700 + k, row=k % 3 != 0, street_door=k >= 18)
            p.kind = "shop"  # type: ignore[attr-defined]
            pool.append(p)
    for p in pool:
        if not hasattr(p, "kind"):
            p.kind = "house"  # type: ignore[attr-defined]
    # Flats go up first against the faces that rise behind street fronts,
    # before the houses on the terrace above take the ground at their backs.
    # The merchants' houses take the square's rim and the main street first,
    # while the deep plots behind them are still open.
    n_houses = t.place_houses(pool, houses, c["gate"], only="townhouse")
    n_houses += t.place_houses(pool, houses - n_houses, c["gate"], only="apartment")
    # A windmill wants open ground at the edge of town: give it its pick
    # before the houses spread out there.
    n_houses += t.place_houses(pool, houses - n_houses, c["gate"], only="windmill")
    n_houses += t.place_houses(pool, houses - n_houses, c["gate"])
    # Selds fill the street fronts the houses could not.
    n_houses += t.place_houses(pool, houses - n_houses, c["gate"], only="shop")
    # Where the town runs up against the edge of the build area, it is walled.
    walled = t.place_town_wall()
    n_fields = t.place_fields()
    n_lamps = t.place_lamps(spacing=5)
    street_life = t.dress_streets()
    n_trees, n_plants = t.place_greenery()
    ground = t.lay_ground()
    b.town = {  # type: ignore[attr-defined]
        "castle": c, "square": {"T": sq["T"], "cells": len(sq["cells"])},
        "streets": [{"name": s["name"], "length": len(s["line"]), "cells": len(s["cells"]),
                     "max_step": int(max((abs(a - b_) for a, b_ in zip(s["heights"], s["heights"][1:])), default=0))}
                    for s in t.streets],
        "terraces": len(climb.get("legs", [])), "switchbacks": len(climb.get("ramps", [])),
        "mine": dug, "street_style": street_style, "climate": climate or "temperate", "town_wall": walled, "rejects": dict(t.rejects), "kinds": {k: len(v) for k, v in t.kinds.items()}, "stilted": t.stilts,
        "set_into_rock": t.embedded, "on_crags": t.crags, "back_doors": t.back_doors, "cliff_fronts": t.cliff_fronts,
        "houses": t.houses, "heights": t.G, "heights_in": t.H0, "use": t.use, "entrance": getattr(b, "town_entrance", None),
        "counts": {"houses": n_houses, "streets": len(t.streets), "street_stairs": stairs, "fields": n_fields,
                   "lamps": n_lamps, "street_props": street_life["props"], "benches": street_life["benches"],
                   "wells": street_life["wells"], "trees": n_trees, "plants": n_plants, "ground_blocks": ground},
    }
    b.notes.append(f"town: {n_houses} houses on {len(t.streets)} streets, {n_fields} fields, {n_trees} trees, "
                   f"{stairs} street stairs, on a {t.n}x{t.m} height map")
    return b


def town_walk(b: BlockBuild):
    """Walk the town from where the main street leaves the map, and say what
    the player reaches: each house's floors, the castle gate, wall-walk and
    keep roof. Returns (walk, report)."""
    from .gameplay import walk as walk_grid

    sub = b.walk_sub()
    g = b.grid(sub)
    ox, oy, oz = g.origin
    ex, ez = b.town["entrance"]
    G = b.town["heights"]
    y0 = int(G[ex, ez])
    while b.blocks.get((ex, y0, ez)) is not None and kind(b.blocks[(ex, y0, ez)]) != "air":
        y0 += 1                           # a street stair at the map's edge: start on its tread
    seed = (int((ex + .5 - ox) * sub), int((y0 - oy) * sub), int((ez + .5 - oz) * sub))
    w = walk_grid(g, b.player, seeds=[seed])
    nodes = w.nodes
    wx = nodes[:, 0] / sub + ox
    wy = nodes[:, 1] / sub + oy
    wz = nodes[:, 2] / sub + oz
    reached = w.reached
    S = b.sizes.storey
    houses_ok = 0
    floors_total = floors_ok = 0
    cell_of = np.floor(wx).astype(int) * 100003 + np.floor(wz).astype(int)
    for h in b.town["houses"]:
        keys = np.array([x * 100003 + z for (x, z) in h["cells"]])
        inside = np.isin(cell_of, keys)
        floors = 0
        ok = 0
        for k in range(4):
            lvl = h["floor"] + 1 + k * S
            here = inside & (np.abs(wy - lvl) < .3)
            if not here.any():
                break
            floors += 1
            if reached[here].any():
                ok += 1
        floors_total += floors
        floors_ok += ok
        houses_ok += int(floors > 0 and ok == floors)
    c = b.town["castle"]
    T = c["T"]
    in_castle = (wx >= c["x"]) & (wx < c["x"] + c["w"]) & (wz >= c["z"]) & (wz < c["z"] + c["d"])
    hw = max(8, 2 * S)
    on_walk = in_castle & (np.abs(wy - (T + hw)) < .3)
    top = wy[in_castle & reached].max() - T if (in_castle & reached).any() else 0.0
    mine_ok, mine_n = 0, 0
    dug = b.town.get("mine") or {}
    if dug:
        key = np.floor(wx).astype(int) * 100003 + np.floor(wz).astype(int)
        for (x, y, z) in dug["walk"]:
            here = (key == x * 100003 + z) & (np.abs(wy - y) < .3)
            mine_n += 1
            mine_ok += int(bool(here.any() and reached[here].any()))
    report = {
        "houses": len(b.town["houses"]),
        "mine_floors_reached": f"{mine_ok}/{mine_n}",
        "houses_every_floor_reached": houses_ok,
        "house_floors_reached": f"{floors_ok}/{floors_total}",
        "castle_bailey_reached": bool((in_castle & reached & (np.abs(wy - T) < .3)).any()),
        "castle_wall_walk_reached_share": round(float(reached[on_walk].mean()), 3) if on_walk.any() else 0.0,
        "castle_highest_point_reached_m": round(float(top), 2),
        "summary": w.summary(),
    }
    return w, report


__all__ = ["town", "town_walk", "sample_heights", "load_heights", "turned"]
