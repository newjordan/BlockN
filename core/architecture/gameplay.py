"""Playability of a build for a player profile, measured on its voxel grid.

Doors in BlockN are voids between stones, not placements, so whether a player
can get in is a question about the empty space. This module walks that space:

1. **Capsule fit.** A cell is a place the player can stand when the body (a
   capsule ``radius`` wide and ``height`` tall, footprint a disc of ``k`` cells)
   fits above it and there is ground or a solid cell under its centre. The rim
   may overlap anything a step tall, and the rounded ends lift and lower the
   rim further, as a character controller's capsule does on stairs.
2. **Moves.** From a standing cell the player walks to any of the 4 neighbours,
   stepping up at most ``step_height`` or dropping at most ``max_drop``, with
   headroom for the move, and walks off a ledge to land up to a body width out.
   Ballistic jumps (``_jump_edges``) add moves up ledges and across gaps; the
   walk is also run without them, so a check can tell "walks in" from "only
   jumps in".
3. **Inside and outside.** At the player's waist height the solid slice is
   closed by the widest door gap; air the outside cannot reach is interior.
4. **Ports.** Standing cells that lie in a gap the closing filled are doorways.
   Each connected group is one port, with a facing (interior to exterior), a
   measured clear width and height, and a world position on its floor.

The checks are added to the quality report only when the request names a
profile, so builds without one are audited exactly as before.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .profile import PlayerProfile, resolve
from .voxels import VoxelGrid, voxelize

_OPEN = 1 << 20          # headroom above the grid top: open sky
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class Body:
    k: int               # footprint radius in cells
    h: int               # height in cells
    step: int            # tallest step up, in cells
    drop: int            # tallest drop, in cells
    waist: int           # cells from the feet to the waist

    @classmethod
    def of(cls, player: PlayerProfile, pitch: float) -> "Body":
        return cls(
            k=max(0, int(math.ceil(player.radius / pitch - 0.5))),
            # A surface voxelizes to within half a cell of where it is, so the
            # height and the step round to whole cells rather than up or down.
            h=max(1, int(math.floor(player.height / pitch + 0.5))),
            step=max(0, int(math.floor(player.step_height / pitch + 0.5))),
            drop=max(0, int(math.floor(player.max_drop / pitch + 1e-9))),
            waist=max(0, int(round(player.height * 0.5 / pitch))),
        )

    def disc(self) -> List[Tuple[int, int]]:
        r2 = self.k * self.k + self.k
        return [(dx, dz) for dx in range(-self.k, self.k + 1) for dz in range(-self.k, self.k + 1)
                if dx * dx + dz * dz <= r2]


def _shift(a: np.ndarray, dx: int, dz: int, fill: Any) -> np.ndarray:
    """out[x, ..., z] = a[x + dx, ..., z + dz], with ``fill`` outside the grid."""
    out = np.full_like(a, fill)
    nx, nz = a.shape[0], a.shape[-1]
    xs_dst = slice(max(0, -dx), nx - max(0, dx))
    xs_src = slice(max(0, dx), nx - max(0, -dx))
    zs_dst = slice(max(0, -dz), nz - max(0, dz))
    zs_src = slice(max(0, dz), nz - max(0, -dz))
    out[xs_dst, ..., zs_dst] = a[xs_src, ..., zs_src]
    return out


def headroom(solid: np.ndarray) -> np.ndarray:
    """Clear cells straight up from each cell, counting the cell itself (0 if solid)."""
    nx, ny, nz = solid.shape
    out = np.zeros(solid.shape, dtype=np.int32)
    above = np.full((nx, nz), _OPEN, dtype=np.int32)
    for y in range(ny - 1, -1, -1):
        above = np.where(solid[:, y, :], 0, above + 1)
        out[:, y, :] = above
    return out


def _min_filter(a: np.ndarray, disc: Sequence[Tuple[int, int]]) -> np.ndarray:
    out = a.copy()
    for dx, dz in disc:
        if dx or dz:
            np.minimum(out, _shift(a, dx, dz, _OPEN), out=out)
    return out


def _capsule_fit(up: np.ndarray, body: Body, radius: float, pitch: float) -> np.ndarray:
    """Clear height for the whole body standing with its feet in each cell.

    Like a character controller, the body may overlap anything no taller than
    a step, and its rounded ends pull in toward the rim: at distance d from
    the centre, a capsule of radius r is c = r - sqrt(r^2 - d^2) above its feet
    and the same below its top. So an offset column only has to be clear from
    ``step + c`` up to ``height - c``; the centre column from the feet up.
    """
    ny = up.shape[1]
    groups: Dict[int, List[Tuple[int, int]]] = {}
    for dx, dz in body.disc():
        if not (dx or dz):
            continue
        d = min(radius, math.hypot(dx, dz) * pitch)
        rise = radius - math.sqrt(max(0.0, radius * radius - d * d))
        lift = body.step + int(math.floor(rise / pitch + 1e-9))
        groups.setdefault(lift, []).append((dx, dz))
    fit = up.copy()
    for lift, offsets in groups.items():
        cap = lift - body.step
        lifted = np.full_like(up, _OPEN)
        if lift < ny:
            lifted[:, :ny - lift, :] = up[:, lift:, :]
        only = np.full_like(up, _OPEN)
        for dx, dz in offsets:
            np.minimum(only, _shift(lifted, dx, dz, _OPEN), out=only)
        np.minimum(fit, only + lift + cap, out=fit)
    return fit


def _dilate2(a: np.ndarray, r: int) -> np.ndarray:
    out = a.copy()
    for d in range(1, r + 1):
        for dx, dz in ((d, 0), (-d, 0)):
            out |= _shift(a, dx, dz, False)
    tmp = out.copy()
    for d in range(1, r + 1):
        for dx, dz in ((0, d), (0, -d)):
            out |= _shift(tmp, dx, dz, False)
    return out


def _erode2(a: np.ndarray, r: int) -> np.ndarray:
    return ~_dilate2(~a, r)


def _flood2(free: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """4-connected flood over ``free`` from ``seeds`` (2D boolean arrays)."""
    reached = seeds & free
    while True:
        grown = reached.copy()
        for dx, dz in _DIRS:
            grown |= _shift(reached, dx, dz, False)
        grown &= free
        if (grown == reached).all():
            return reached
        reached = grown


def _border(shape: Tuple[int, int], width: int = 1) -> np.ndarray:
    b = np.zeros(shape, dtype=bool)
    b[:width, :] = b[-width:, :] = True
    b[:, :width] = b[:, -width:] = True
    return b


@dataclass
class Walk:
    """Everything the checks need, kept for tests and tools."""

    grid: VoxelGrid
    player: PlayerProfile
    body: Body
    nodes: np.ndarray                    # N x 3 standing cells
    reached: np.ndarray                  # N bool, reachable from outside
    interior: np.ndarray                 # N bool
    ports: List[Dict[str, Any]] = field(default_factory=list)
    close_radius: int = 0
    reached_walking: Optional[np.ndarray] = None   # N bool, reachable without jumping
    jump_edges: int = 0

    def summary(self) -> Dict[str, Any]:
        n_in = int(self.interior.sum())
        walking = self.reached if self.reached_walking is None else self.reached_walking
        return {
            "profile": self.player.name,
            "pitch": self.grid.pitch,
            "standing_cells": int(len(self.nodes)),
            "reached_cells": int(self.reached.sum()),
            "interior_cells": n_in,
            "interior_reached": int((self.interior & self.reached).sum()),
            "interior_reached_share": None if not n_in else round(float((self.interior & self.reached).sum()) / n_in, 4),
            "interior_reached_walking": int((self.interior & walking).sum()),
            "reached_walking_cells": int(walking.sum()),
            "jump_edges": self.jump_edges,
            "ports": len(self.ports),
        }


def walk(grid: VoxelGrid, player: Any, *, close_width: Optional[float] = None, jumps: bool = True,
         seeds: Optional[Sequence[Sequence[int]]] = None,
         climbs: Optional[Sequence[Tuple[Sequence[int], Sequence[int]]]] = None) -> Walk:
    """Find standing cells, what the outside reaches, the interior and the ports.

    ``close_width`` is the widest opening treated as a doorway rather than open
    space (default: 1.5 times the player's door width, at least 2 m). A wider
    opening makes the space behind it count as outside.

    ``jumps`` adds ballistic jumps (see ``_jump_edges``). ``seeds`` are cells
    to start from; by default, the ground on the grid's open outer ring.

    ``climbs`` are ladders: boxes ``(lo, hi)`` of grid cells (hi exclusive)
    round a ladder and the wall top it leads to; every standing cell in a box
    is joined to every other, both ways, as a climb.
    """
    player = resolve(player) or PlayerProfile()
    p = grid.pitch
    body = Body.of(player, p)
    solid = grid.solid
    nx, ny, nz = solid.shape

    up = headroom(solid)                                  # clear run above each cell
    fit = _capsule_fit(up, body, player.radius, p)
    support = np.zeros_like(solid)
    support[:, 0, :] = True
    support[:, 1:, :] = solid[:, :-1, :]
    standing = (fit >= body.h) & support
    nodes = np.argwhere(standing)
    ids = np.full(solid.shape, -1, dtype=np.int64)
    ids[tuple(nodes.T)] = np.arange(len(nodes))

    # Edges: walk to a 4-neighbour d cells up (d <= step) or down (d <= drop),
    # with headroom for the body over the height it passes through.
    src_list, dst_list = [], []
    xs, ys, zs = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    for dx, dz in _DIRS:
        tx, tz = xs + dx, zs + dz
        inside = (tx >= 0) & (tx < nx) & (tz >= 0) & (tz < nz)
        for d in range(-body.drop, body.step + 1):
            ty = ys + d
            ok = inside & (ty >= 0) & (ty < ny)
            if not ok.any():
                continue
            i = np.nonzero(ok)[0]
            t = ids[tx[i], ty[i], tz[i]]
            good = t >= 0
            # The whole body fits at both ends; on the way, the centre column
            # it rises in (or falls through) must clear the body as well.
            if d > 0:     # rise inside the current column
                good &= up[xs[i], ys[i], zs[i]] >= body.h + d
            elif d < 0:   # step out over the edge and fall in the next column
                good &= up[tx[i], ty[i], tz[i]] >= body.h - d
            src_list.append(i[good])
            dst_list.append(t[good])
        # Walk off a ledge: the body clears the edge at its own height and
        # lands up to a body width out, where the rim no longer overhangs.
        for m in range(2, body.k + 2):
            tx, tz = xs + m * dx, zs + m * dz
            inside = (tx >= 0) & (tx < nx) & (tz >= 0) & (tz < nz)
            over = inside.copy()
            for j in range(1, m):
                px, pz = xs + j * dx, zs + j * dz
                over[inside] &= fit[px[inside], ys[inside], pz[inside]] >= body.h
            for d in range(-body.drop, 0):
                ty = ys + d
                ok = over & (ty >= 0)
                if not ok.any():
                    continue
                i = np.nonzero(ok)[0]
                t = ids[tx[i], ty[i], tz[i]]
                good = (t >= 0)
                good[good] &= up[tx[i][good], ty[i][good], tz[i][good]] >= body.h - d
                src_list.append(i[good])
                dst_list.append(t[good])
    for lo, hi in climbs or ():
        box = np.all((nodes >= np.asarray(lo)) & (nodes < np.asarray(hi)), axis=1)
        idx = np.nonzero(box)[0]
        if len(idx) >= 2:
            hub = np.full(len(idx) - 1, idx[0])
            src_list += [hub, idx[1:]]
            dst_list += [idx[1:], hub]
    walk_src = np.concatenate(src_list) if src_list else np.zeros(0, dtype=np.int64)
    walk_dst = np.concatenate(dst_list) if dst_list else np.zeros(0, dtype=np.int64)
    if jumps:
        jump_src, jump_dst = _jump_edges(player, body, p, standing, fit, nodes, ids)
    else:
        jump_src = jump_dst = np.zeros(0, dtype=np.int64)

    # Seeds: ground cells on the outer ring of the grid, which the voxelizer
    # leaves as open margin around the build, or the cells asked for.
    if seeds is None:
        ring = body.k + 2
        seed_idx = np.nonzero((ys == 0) & ((xs < ring) | (xs >= nx - ring) | (zs < ring) | (zs >= nz - ring)))[0]
    else:
        seed_idx = np.array([_settle(ids, cell) for cell in seeds], dtype=np.int64)
        seed_idx = seed_idx[seed_idx >= 0]
    reached_walk = _bfs(len(nodes), walk_src, walk_dst, seed_idx)
    reached = _bfs(len(nodes), np.concatenate([walk_src, jump_src]), np.concatenate([walk_dst, jump_dst]), seed_idx)

    # Interior and doorways, one waist slice per floor level in use.
    if close_width is None:
        close_width = max(2.0, 1.5 * player.clearances().door_width)
    r = max(1, int(math.ceil(close_width / 2.0 / p)))
    interior = np.zeros(len(nodes), dtype=bool)
    gap_nodes = np.zeros(len(nodes), dtype=bool)
    for y in np.unique(ys).tolist():
        wy = min(ny - 1, y + body.waist)
        s = solid[:, wy, :]
        closed = _erode2(_dilate2(s, r), r)
        # Closing never grows a solid into the grid edge; keep the ring open.
        closed &= ~_border(s.shape)
        # Closing leaves straight walls where they are and plugs gaps narrower
        # than 2r, so the air the border reaches around the plugs is outside,
        # the plugs are doorways and the rest of the air is interior.
        outside = _flood2(~closed, _border(s.shape))
        idx = np.nonzero(ys == y)[0]
        interior[idx] = ~outside[xs[idx], zs[idx]] & ~closed[xs[idx], zs[idx]]
        gap_nodes[idx] = closed[xs[idx], zs[idx]] & ~s[xs[idx], zs[idx]]

    ports = _ports(grid, player, body, nodes, gap_nodes & reached, interior, up)
    return Walk(grid, player, body, nodes, reached, interior, ports, r, reached_walk, int(len(jump_src)))


def _settle(ids: np.ndarray, cell: Sequence[int]) -> int:
    """The standing cell at ``cell`` or the first one below it, or -1."""
    x, y, z = (int(v) for v in cell)
    if not (0 <= x < ids.shape[0] and 0 <= z < ids.shape[2]):
        return -1
    for yy in range(min(y, ids.shape[1] - 1), -1, -1):
        if ids[x, yy, z] >= 0:
            return int(ids[x, yy, z])
    return -1


def _bfs(n: int, src: np.ndarray, dst: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    indptr = np.searchsorted(src, np.arange(n + 1))
    reached = np.zeros(n, dtype=bool)
    reached[seeds] = True
    queue = deque(np.unique(seeds).tolist())
    while queue:
        node = queue.popleft()
        for m in dst[indptr[node]:indptr[node + 1]].tolist():
            if not reached[m]:
                reached[m] = True
                queue.append(m)
    return reached


def _jump_edges(player: PlayerProfile, body: Body, p: float, standing: np.ndarray, fit: np.ndarray,
                nodes: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Straight ballistic jumps from ledge and gap edges.

    Take-off speed v0 = sqrt(2 g J) straight up, any horizontal speed up to
    ``run_speed``. Landing dh above the take-off (negative for below) happens on
    the way down at t = (v0 + sqrt(v0^2 - 2 g dh)) / g, so the farthest landing
    is run_speed * t. A jump of n cells takes t, so its horizontal speed is
    n p / t, and at column i the feet are v0 t_i - g t_i^2 / 2 above take-off,
    with t_i = t i / n. The body must fit (plus one cell for rounding) at every
    column on the way, and land on a standing cell no farther below than the
    safe drop.

    Only jumps the walk can't already make are kept: up a ledge taller than a
    step, or across a column with nowhere to stand within a step of the
    take-off height. They start from edge cells, where some neighbour has no
    standing cell within a step, which keeps the count of candidates small.
    """
    nx, ny, nz = standing.shape
    g, v = player.gravity, player.run_speed
    v0 = math.sqrt(2.0 * g * player.jump_height)
    if v0 <= 0 or not len(nodes):
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # near[x, y, z]: somewhere to stand in this column within a step of y.
    near = standing.copy()
    for k in range(1, body.step + 1):
        near[:, k:, :] |= standing[:, :-k, :]
        near[:, :-k, :] |= standing[:, k:, :]
    xs, ys, zs = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    up_max = int(math.floor(player.jump_height / p + 1e-9))
    src_list, dst_list = [], []
    for dx, dz in _DIRS:
        nxt_x, nxt_z = xs + dx, zs + dz
        inside = (nxt_x >= 0) & (nxt_x < nx) & (nxt_z >= 0) & (nxt_z < nz)
        edge = ~inside.copy()
        edge[inside] = ~near[nxt_x[inside], ys[inside], nxt_z[inside]]
        cand = np.nonzero(edge)[0]
        if not len(cand):
            continue
        cx, cy, cz = xs[cand], ys[cand], zs[cand]
        for dy in range(-body.drop, up_max + 1):
            dh = dy * p
            disc = v0 * v0 - 2.0 * g * dh
            if disc < 0:
                continue
            t_land = (v0 + math.sqrt(disc)) / g
            n_max = int(math.floor(v * t_land / p + 1e-9))
            n_min = 1 if dy > body.step else 2
            for n in range(n_min, n_max + 1):
                lx, ly, lz = cx + n * dx, cy + dy, cz + n * dz
                ok = (lx >= 0) & (lx < nx) & (lz >= 0) & (lz < nz) & (ly >= 0) & (ly < ny)
                if not ok.any():
                    continue
                sel = np.nonzero(ok)[0]
                land = ids[lx[sel], ly[sel], lz[sel]]
                keep = land >= 0
                if not keep.any():
                    continue
                sel, land = sel[keep], land[keep]
                gap = np.full(len(sel), dy > body.step)
                for i in range(1, n):
                    ti = t_land * i / n
                    off = int(math.floor((v0 * ti - 0.5 * g * ti * ti) / p + 1e-9))
                    px, pz = cx[sel] + i * dx, cz[sel] + i * dz
                    py = cy[sel] + off
                    clear = (py >= 0) & (py < ny)
                    good = np.zeros(len(sel), dtype=bool)
                    good[clear] = fit[px[clear], py[clear], pz[clear]] >= body.h + 1
                    gap |= ~near[px, cy[sel], pz]
                    sel, land, gap = sel[good], land[good], gap[good]
                    if not len(sel):
                        break
                if not len(sel):
                    continue
                sel, land = sel[gap], land[gap]
                src_list.append(cand[sel])
                dst_list.append(land)
    if not src_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(src_list), np.concatenate(dst_list)


def _ports(grid: VoxelGrid, player: PlayerProfile, body: Body, nodes: np.ndarray, gap: np.ndarray,
           interior: np.ndarray, up: np.ndarray) -> List[Dict[str, Any]]:
    """Group doorway cells (8-connected, same floor) into ports."""
    idx = np.nonzero(gap)[0]
    if not len(idx):
        return []
    key = {tuple(nodes[i]): i for i in idx}
    seen, groups = set(), []
    for i in idx:
        if i in seen:
            continue
        group, queue = [], deque([i])
        seen.add(i)
        while queue:
            n = queue.popleft()
            group.append(n)
            x, y, z = nodes[n]
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    m = key.get((x + dx, y, z + dz))
                    if m is not None and m not in seen:
                        seen.add(m)
                        queue.append(m)
        groups.append(np.array(group))

    solid = grid.solid
    p = grid.pitch
    all_ids = {tuple(c): j for j, c in enumerate(nodes)}
    out = []
    for gi, group in enumerate(sorted(groups, key=lambda g: tuple(nodes[g].min(0)))):
        cells = nodes[group]
        span_x = cells[:, 0].max() - cells[:, 0].min() + 1
        span_z = cells[:, 2].max() - cells[:, 2].min() + 1
        # The wall runs along the longer span; the port faces across it.
        wall_axis = "x" if span_x >= span_z else "z"
        span = span_x if wall_axis == "x" else span_z
        normal = (0, 1) if wall_axis == "x" else (1, 0)
        # Facing: towards the side whose standing cells are not interior.
        votes = 0
        for c in cells:
            for sign in (1, -1):
                for step in range(1, body.k + 6):
                    j = all_ids.get((c[0] + sign * step * normal[0], c[1], c[2] + sign * step * normal[1]))
                    if j is not None and not gap[j]:
                        votes += sign if not interior[j] else -sign
                        break
        sign = 1 if votes >= 0 else -1
        facing = [0.0, 0.0, 0.0]
        facing[0 if wall_axis == "z" else 2] = float(sign)
        centre = cells.mean(0)
        floor_y = int(cells[:, 1].min())
        height_cells = int(min(up[c[0], c[1], c[2]] for c in cells))
        out.append({
            "id": f"port.{gi:03d}",
            "source": "measured",
            "position": [round(float(grid.origin[0] + (centre[0] + 0.5) * p), 4), round(floor_y * p + grid.origin[1], 4),
                         round(float(grid.origin[2] + (centre[2] + 0.5) * p), 4)],
            "facing": facing,
            "wall_axis": wall_axis,
            # Standing centres in the gap plus the body each side; the true gap
            # is up to one cell wider.
            "width": round(float((span + 2 * body.k + 0.5) * p), 4),
            "height": None if height_cells >= _OPEN else round(float(height_cells * p), 4),
            "cells": int(len(cells)),
            "profile": player.name,
        })
    return out


# ------------------------------------------------------------------- checks
def _check(code: str, status: str, message: str, *, severity: Optional[str] = None,
           ids: Sequence[str] = (), observed: Any = None, expected: Any = None):
    from .quality import QualityCheck

    severity = severity or {"pass": "info", "warning": "warning", "fail": "error"}[status]
    return QualityCheck(code, status, severity, message, list(ids), observed, expected)


def stair_checks(document: Any, player: PlayerProfile) -> List[Any]:
    risers = [(pl.id, float(pl.parameters.get("riser")))
              for pl in document.placements
              if isinstance(pl.parameters, dict) and pl.parameters.get("riser") is not None]
    if not risers:
        return [_check("game.stair_rise", "pass", "No stairs to check.", observed={"stairs": 0})]
    over = [(i, r) for i, r in risers if r > player.step_height + 1e-6]
    worst = max(r for _, r in risers)
    if over:
        return [_check("game.stair_rise", "fail",
                       f"{len(over)} stair placement(s) rise more than a '{player.name}' can step "
                       f"({worst:.3f} > {player.step_height:.3f}).",
                       ids=[i for i, _ in over], observed={"max_riser": round(worst, 4)},
                       expected={"max_riser": player.step_height})]
    return [_check("game.stair_rise", "pass", f"Every stair riser is within the '{player.name}' step height.",
                   observed={"max_riser": round(worst, 4)}, expected={"max_riser": player.step_height})]


def _grid_checks(grid: VoxelGrid, result: Walk, player: PlayerProfile, declared: Sequence[Dict[str, Any]]
                 ) -> List[Dict[str, Any]]:
    """Entry, reach, doorway and headroom findings for one walked grid."""
    clear = player.clearances()
    summary = result.summary()
    out: List[Dict[str, Any]] = []
    n_in = summary["interior_cells"]
    if n_in == 0:
        out.append(dict(code="game.entry", status="pass", message="The build has no enclosed floor to enter.",
                        observed=summary))
    elif summary["interior_reached_walking"] > 0:
        out.append(dict(code="game.entry", status="pass", message=f"A '{player.name}' can walk in from outside.",
                        observed=summary))
    elif summary["interior_reached"] > 0:
        out.append(dict(code="game.entry", status="warning",
                        message=f"A '{player.name}' can only get inside by jumping.", observed=summary))
    else:
        out.append(dict(
            code="game.entry", status="fail",
            message=(f"A '{player.name}' cannot get inside from the ground outside: no doorway fits "
                     f"a body {2 * player.radius:.2f} wide and {player.height:.2f} tall."),
            ids=[d["host"] for d in declared], observed=summary,
            expected={"door_width": clear.door_width, "door_height": clear.door_height}))
    if n_in and summary["interior_reached"]:
        share = summary["interior_reached_share"] or 0.0
        out.append(dict(code="game.interior_reach", status="pass" if share >= 0.9 else "warning",
                        message=f"{share:.0%} of the floor inside is reachable.", observed=summary,
                        expected={"interior_reached_share": 0.9}))
    tight = [port for port in result.ports
             # A measured size can be up to one cell short of the true one.
             if port["width"] + grid.pitch < clear.door_width
             or (port["height"] is not None and port["height"] + grid.pitch < clear.door_height)]
    if result.ports:
        out.append(dict(
            code="game.port_clearance", status="warning" if tight else "pass",
            message=(f"{len(tight)} of {len(result.ports)} doorway(s) are tighter than the '{player.name}' clearance."
                     if tight else f"All {len(result.ports)} doorway(s) clear the '{player.name}' with margin."),
            observed=[{k: port[k] for k in ("id", "width", "height")} for port in result.ports],
            expected={"door_width": clear.door_width, "door_height": clear.door_height}))
    if n_in and clear.ceiling_height:
        need = int(math.ceil(clear.ceiling_height / grid.pitch - 1e-9))
        inside = result.nodes[result.interior]
        low = [v for v in _headroom_at(grid, inside) if v < need]
        out.append(dict(
            code="game.headroom", status="warning" if low else "pass",
            message=(f"{len(low)} interior cell(s) have less than {clear.ceiling_height:.2f} of headroom "
                     f"for a {player.camera.mode.replace('_', ' ')} camera." if low else
                     f"Interior headroom clears {clear.ceiling_height:.2f} for the camera."),
            observed={"low_cells": len(low)}, expected={"ceiling_height": clear.ceiling_height}))
    return out


def _as_check(finding: Dict[str, Any]):
    return _check(finding["code"], finding["status"], finding["message"], ids=finding.get("ids", ()),
                  observed=finding.get("observed"), expected=finding.get("expected"))


_RANK = {"pass": 0, "warning": 1, "fail": 2}


def audit(document: Any, scene: Any, profile: Any = None, *, pitch: Optional[float] = None
          ) -> Tuple[List[Any], Dict[str, Any], Optional[VoxelGrid]]:
    """Playability checks for a compiled build. Returns (checks, metrics, grid).

    A settlement is walked one building at a time, each on its own grid, and
    the checks name the plots that fail; its grid is returned as None, since a
    whole town at body-sized cells is too big to be useful in one array.
    """
    player = resolve(profile if profile is not None else document.request.profile)
    if player is None:
        raise ValueError("a playability audit needs a player profile")
    clear = player.clearances()
    checks = stair_checks(document, player)
    plots = (document.metadata.get("settlement") or {}).get("plots")
    if plots:
        findings, per_plot, ports = _settlement_findings(document, scene, player, plots, pitch)
        checks.extend(findings)
        metrics = {"player": player.to_dict(), "clearances": clear.to_dict(), "buildings": per_plot,
                   "ports": ports}
        return checks, metrics, None
    grid = voxelize(document, scene, pitch=pitch, profile=player)
    result = walk(grid, player)
    grid.ports = result.ports
    declared = list(document.metadata.get("ports") or [])
    checks.extend(_as_check(f) for f in _grid_checks(grid, result, player, declared))
    metrics = {"player": player.to_dict(), "clearances": clear.to_dict(), "walk": result.summary(),
               "ports": result.ports, "declared_ports": declared}
    return checks, metrics, grid


def _settlement_findings(document: Any, scene: Any, player: PlayerProfile, plots: Sequence[Dict[str, Any]],
                         pitch: Optional[float]):
    from .voxels import voxelize_parts

    pairs = list(zip(scene.blocks, document.placements))
    per_plot, ports = [], []
    worst: Dict[str, Dict[str, Any]] = {}
    for record in plots:
        product = record.get("product")
        if not product:
            continue
        prefix = f"{record['id']}.{product}."
        members = [(b, pl) for b, pl in pairs if pl.id.startswith(prefix)]
        if not members:
            continue
        grid = voxelize_parts([b for b, _ in members], [pl for _, pl in members], pitch=pitch, player=player,
                              source={"product": product, "plot": record["id"]})
        result = walk(grid, player)
        found = _grid_checks(grid, result, player, [])
        status = {f["code"]: f["status"] for f in found}
        per_plot.append({"plot": record["id"], "product": product, "pitch": grid.pitch,
                         "checks": status, "walk": result.summary()})
        for port in result.ports:
            ports.append({**port, "id": f"{record['id']}.{port['id']}", "plot": record["id"]})
        for f in found:
            entry = worst.setdefault(f["code"], {"status": "pass", "plots": {"pass": [], "warning": [], "fail": []}})
            entry["plots"][f["status"]].append(record["id"])
            if _RANK[f["status"]] > _RANK[entry["status"]]:
                entry["status"] = f["status"]
    findings = []
    words = {
        "game.entry": ("get into", "only by jumping into"),
        "game.interior_reach": ("reach most of the floor of", "reach most of the floor of"),
        "game.port_clearance": ("clear every doorway of", "clear every doorway of"),
        "game.headroom": ("have headroom in", "have headroom in"),
    }
    total = len(per_plot)
    for code, entry in sorted(worst.items()):
        bad = entry["plots"]["fail"] + entry["plots"]["warning"]
        verb = words.get(code, ("pass", "pass"))[0]
        if entry["status"] == "pass":
            message = f"A '{player.name}' can {verb} all {total} building(s) checked."
        else:
            counts = [f"{len(entry['plots'][s])} {word}" for s, word in (("fail", "fail"), ("warning", "warn"))
                      if entry["plots"][s]]
            message = (f"{' and '.join(counts)} of {total} building(s) on "
                       f"{code.split('.', 1)[1].replace('_', ' ')} for a '{player.name}': {', '.join(bad[:8])}"
                       + (" and more." if len(bad) > 8 else "."))
        findings.append(_check(code, entry["status"], message, ids=bad,
                               observed={k: len(v) for k, v in entry["plots"].items()}))
    return findings, per_plot, ports


def _headroom_at(grid: VoxelGrid, cells: np.ndarray) -> List[int]:
    if not len(cells):
        return []
    up = headroom(grid.solid)
    return up[cells[:, 0], cells[:, 1], cells[:, 2]].tolist()


__all__ = ["Body", "Walk", "walk", "audit", "stair_checks", "headroom"]
