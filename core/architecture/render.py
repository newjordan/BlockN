"""Pictures of a voxel grid, for a person to look at.

Every playability result should come with an image, so this draws a
``VoxelGrid`` as a shaded orthographic view with Pillow and numpy only (no GPU,
no OpenGL). Three things can be layered on it:

* a cut, which drops every cell above a height or in front of a plane, so the
  inside of a building shows;
* a walk overlay, which tints the floor under each standing cell the walk
  found: green where the player walks to, amber where they only get by a
  jump, red for enclosed floor they never reach;
* a figure, a capsule of the player's own size stood at a doorway, so the door
  can be judged against the body by eye.

Faces are drawn back to front (painter's order). For axis-aligned unit cubes
seen orthographically that order is exact.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .voxels import AIR, VoxelGrid

_FALLBACK = (170, 160, 150)
_GROUND = (92, 104, 84)
_FIGURE = (40, 110, 220)
_REACH = (60, 200, 80)
_JUMP = (240, 170, 40)
_SHUT = (220, 50, 50)


def _substance_rgb(name: Optional[str]) -> Tuple[int, int, int]:
    try:
        from .substances import get_substance
        s = get_substance(name or "")
        lo, hi = s.color_low, s.color_high
        return tuple(int((a + b) / 2) for a, b in zip(lo, hi))  # type: ignore[return-value]
    except Exception:
        pass
    key = (name or "").lower()
    for word, rgb in (("glass", (150, 190, 210)), ("oak", (150, 105, 60)), ("timber", (150, 105, 60)),
                      ("wood", (150, 105, 60)), ("board", (150, 105, 60)), ("thatch", (200, 170, 90)),
                      ("tile", (170, 80, 60)), ("brick", (160, 70, 55)), ("iron", (70, 70, 75)),
                      ("plaster", (225, 220, 205)), ("marble", (230, 228, 220)), ("stone", (165, 160, 150))):
        if word in key:
            return rgb
    return _FALLBACK


def _view(azimuth: float, elevation: float):
    az, el = math.radians(azimuth), math.radians(elevation)
    # Direction from the scene toward the camera.
    d = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])
    right = np.cross([0.0, 1.0, 0.0], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    up /= np.linalg.norm(up)
    return d, right, up


def render(grid: VoxelGrid, *, azimuth: float = 35.0, elevation: float = 30.0, width: int = 1100,
           cut_above: Optional[float] = None, cut_front_z: Optional[float] = None,
           walk: Any = None, figure: Optional[Dict[str, Any]] = None, title: Optional[str] = None,
           ground: bool = True, max_cell: Optional[float] = 0.3, block: Optional[int] = None) -> Image.Image:
    """Draw ``grid``. Heights and planes are in world metres.

    ``block`` is for a block-mode grid, ``block`` cells to a Minecraft block:
    edges are drawn only along block boundaries and each block's shade varies
    a little, so the picture reads as blocks rather than as small cells. A
    palette entry with an ``rgb`` is drawn in that colour.

    ``walk`` is a ``gameplay.Walk`` for the same grid. ``figure`` is a player
    profile plus a world position, ``{"player": PlayerProfile, "at": (x, z)}``.
    """
    labels = grid.labels.copy()
    p, origin = grid.pitch, np.array(grid.origin, dtype=float)
    nx, ny, nz = labels.shape
    if cut_above is not None:
        k = int(math.floor((cut_above - origin[1]) / p))
        labels[:, max(0, k):, :] = AIR
    if cut_front_z is not None:
        k = int(math.floor((cut_front_z - origin[2]) / p))
        labels[:, :, :max(0, min(nz, k))] = AIR

    colours = {e["label"]: (tuple(e["rgb"]) if e.get("rgb") else _substance_rgb(e.get("substance")))
               for e in grid.palette}
    tint = np.zeros(labels.shape, dtype=np.uint8)   # 0 none, 1 reach, 2 jump, 3 shut
    if walk is not None:
        nodes = walk.nodes
        walking = walk.reached if walk.reached_walking is None else walk.reached_walking
        kind = np.where(walking, 1, np.where(walk.reached, 2, np.where(walk.interior, 3, 0)))
        keep = (kind > 0) & (nodes[:, 1] > 0)
        sx, sy, sz = nodes[keep, 0], nodes[keep, 1] - 1, nodes[keep, 2]
        ok = labels[sx, sy, sz] != AIR
        tint[sx[ok], sy[ok], sz[ok]] = kind[keep][ok]

    fig = np.zeros(labels.shape, dtype=bool)
    if figure is not None:
        player, at = figure["player"], figure["at"]
        fx, fz = at[0], at[-1]
        foot = at[1] if len(at) == 3 else 0.0
        cx, cz = (fx - origin[0]) / p, (fz - origin[2]) / p
        rr = player.radius / p
        top = int(math.ceil(player.height / p))
        xs, zs = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        disc = (xs + .5 - cx) ** 2 + (zs + .5 - cz) ** 2 <= rr * rr
        y0 = max(0, int(math.floor((foot - origin[1]) / p + 1e-6)))
        for y in range(y0, min(ny, y0 + top)):
            # Round the head and feet like a capsule.
            h = (y - y0 + .5) * p
            r_here = player.radius
            for end in (h, player.height - h):
                if end < player.radius:
                    r_here = min(r_here, math.sqrt(max(0.0, player.radius ** 2 - (player.radius - end) ** 2)))
            sl = (xs + .5 - cx) ** 2 + (zs + .5 - cz) ** 2 <= (r_here / p) ** 2
            free = sl & disc & (labels[:, y, :] == AIR)
            fig[:, y, :] |= free

    # Draw at no finer than ``max_cell``: blocks of k cells become one, taking
    # the highest label, the strongest tint and the figure if any cell has it.
    k = max(1, int(round(max_cell / p))) if max_cell else 1
    if block:
        k = 1
    if k > 1:
        def pool(a, fn):
            sx, sy, sz = (-(-n // k) * k for n in a.shape)
            b = np.zeros((sx, sy, sz), dtype=a.dtype)
            b[:a.shape[0], :a.shape[1], :a.shape[2]] = a
            return fn(fn(fn(b.reshape(sx // k, k, sy, sz), axis=1).reshape(sx // k, sy // k, k, sz), axis=2)
                      .reshape(sx // k, sy // k, sz // k, k), axis=3)
        labels, tint, fig = pool(labels, np.max), pool(tint, np.max), pool(fig, np.max)
        # Keep the overlay on top faces only where the pooled cell is floor.
        p = p * k
        nx, ny, nz = labels.shape

    solid = (labels != AIR) | fig
    d, right, up = _view(azimuth, elevation)
    axes = ((0, 1 if d[0] > 0 else -1), (1, 1 if d[1] > 0 else -1), (2, 1 if d[2] > 0 else -1))
    shade = {0: .78, 1: 1.0, 2: .62}
    top_label = int(labels.max()) if labels.size else 0
    lut = np.array([colours.get(i, _FALLBACK) for i in range(top_label + 1)], dtype=float).reshape(-1, 3)
    batches = []   # (depth, world quads, rgb, edge mask or None)

    def quad_for(axis: int, sign: int) -> np.ndarray:
        o = np.zeros(3)
        o[axis] = 1.0 if sign > 0 else 0.0
        u_ax, v_ax = [a for a in range(3) if a != axis]
        out = []
        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            c = o.copy()
            c[u_ax], c[v_ax] = du, dv
            out.append(c)
        return np.array(out)

    def exposed_of(mask: np.ndarray, axis: int, sign: int) -> np.ndarray:
        nb = np.zeros_like(mask)
        sl_from = [slice(None)] * 3
        sl_to = [slice(None)] * 3
        if sign > 0:
            sl_from[axis], sl_to[axis] = slice(1, None), slice(None, -1)
        else:
            sl_from[axis], sl_to[axis] = slice(None, -1), slice(1, None)
        nb[tuple(sl_to)] = mask[tuple(sl_from)]
        return mask & ~nb

    def jitter(bcells: np.ndarray) -> np.ndarray:
        bx, by, bz = (bcells[:, i].astype(np.int64) for i in range(3))
        return 1.0 + (((bx * 73856093) ^ (by * 19349663) ^ (bz * 83492791)) % 11 - 5) / 100.0

    # Blocks that fill their whole cube in one label are drawn a block face
    # at a time rather than cell by cell: the same picture, far fewer faces.
    skip = None
    if block and walk is None and figure is None and all(n % block == 0 for n in labels.shape):
        kb = block
        bx_, by_, bz_ = (n // kb for n in labels.shape)
        cube = labels.reshape(bx_, kb, by_, kb, bz_, kb)
        first = cube[:, 0, :, 0, :, 0]
        full = (first != AIR)
        for i in range(kb):
            for j in range(kb):
                for k2 in range(kb):
                    full &= cube[:, i, :, j, :, k2] == first
        skip = full.repeat(kb, 0).repeat(kb, 1).repeat(kb, 2)
        for axis, sign in axes:
            cells = np.argwhere(exposed_of(full, axis, sign))
            if not len(cells):
                continue
            quad = quad_for(axis, sign)
            rgb = lut[first[tuple(cells.T)]] * (shade[axis] * jitter(cells))[:, None]
            world = origin + (cells[:, None, :] + quad[None]) * (p * kb)
            depth = (origin + (cells + .5) * (p * kb)) @ d
            batches.append((depth, world, np.minimum(255, rgb).astype(int), np.ones((len(cells), 4), bool)))

    for axis, sign in axes:
        exposed = exposed_of(solid, axis, sign)
        if skip is not None:
            exposed &= ~skip
        cells = np.argwhere(exposed)
        if not len(cells):
            continue
        quad = quad_for(axis, sign)
        idx = tuple(cells.T)
        rgb = lut[labels[idx]]
        tinted = np.zeros(len(cells), bool)
        if axis == 1 and sign > 0:
            t = tint[idx]
            tinted = t > 0
            for k_, c_ in ((1, _REACH), (2, _JUMP), (3, _SHUT)):
                rgb[t == k_] = c_
        f_ = fig[idx]
        rgb[f_] = _FIGURE
        sv = np.full(len(cells), shade[axis])
        if block:
            jit = ~f_ & ~tinted
            sv[jit] *= jitter(cells[jit] // block)
        rgb = np.minimum(255, rgb * sv[:, None]).astype(int)
        world = origin + (cells[:, None, :] + quad[None]) * p
        depth = (origin + (cells + .5) * p) @ d
        edges = None
        if block:
            # The quad's edges that lie on block boundaries.
            corner = cells[:, None, :] + quad[None]
            u_ax, v_ax = [a for a in range(3) if a != axis]
            edges = np.zeros((len(cells), 4), bool)
            for i in range(4):
                for a_ in (u_ax, v_ax):
                    same = corner[:, i, a_] == corner[:, (i + 1) % 4, a_]
                    edges[:, i] |= same & (np.mod(corner[:, i, a_], block) == 0)
        batches.append((depth, world, rgb, edges))

    # Project and fit to the image.
    if batches:
        depth = np.concatenate([b_[0] for b_ in batches])
        world = np.concatenate([b_[1] for b_ in batches])
        rgb = np.concatenate([b_[2] for b_ in batches])
        edge_any = any(b_[3] is not None for b_ in batches)
        edges = (np.concatenate([b_[3] if b_[3] is not None else np.zeros((len(b_[0]), 4), bool) for b_ in batches])
                 if edge_any else None)
        outline = np.concatenate([np.full(len(b_[0]), b_[3] is None) for b_ in batches])
    else:
        depth, world, rgb, edges, outline = (np.zeros(0), np.zeros((0, 4, 3)), np.zeros((0, 3), int), None,
                                             np.zeros(0, bool))
    pts = world.reshape(-1, 3) if len(world) else np.zeros((1, 3))
    if ground:
        g0 = origin + np.array([0, -origin[1], 0])
        gx, gz = nx * p, nz * p
        ground_quad = np.array([g0, g0 + [gx, 0, 0], g0 + [gx, 0, gz], g0 + [0, 0, gz]])
        pts = np.concatenate([pts, ground_quad])
    u, v = pts @ right, pts @ up
    span = max(u.max() - u.min(), (v.max() - v.min()) * 1.0, 1e-6)
    pad = 30
    scale = (width - 2 * pad) / span
    height = int((v.max() - v.min()) * scale) + 2 * pad + (40 if title else 0)
    img = Image.new("RGB", (width, max(200, height)), (246, 244, 239))
    draw = ImageDraw.Draw(img)
    top_off = 40 if title else 0
    umin, vmax = u.min(), v.max()

    def to_px(q: np.ndarray) -> np.ndarray:
        return np.stack([(q @ right - umin) * scale + pad, (vmax - q @ up) * scale + pad + top_off], axis=-1)

    if ground:
        draw.polygon([tuple(x) for x in to_px(ground_quad).tolist()], fill=_GROUND)
    order = np.argsort(depth, kind="stable")
    px_all = to_px(world[order]).tolist() if len(order) else []
    cols = rgb[order].tolist()
    lines = np.maximum(0, rgb[order] - 28).tolist()
    ed = edges[order].tolist() if edges is not None else None
    ol = outline[order].tolist()
    for n_, quad_px in enumerate(px_all):
        pts_ = [tuple(q) for q in quad_px]
        col, line = tuple(cols[n_]), tuple(lines[n_])
        if ol[n_]:
            draw.polygon(pts_, fill=col, outline=line)
            continue
        draw.polygon(pts_, fill=col)
        e_ = ed[n_]
        for i in range(4):
            if e_[i]:
                draw.line([pts_[i], pts_[(i + 1) % 4]], fill=line, width=1)
    if title:
        try:
            font = ImageFont.load_default(size=22)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((pad, 10), title, fill=(30, 30, 30), font=font)
    return img


def legend(walk: bool = True, figure: bool = True) -> str:
    parts = []
    if figure:
        parts.append("blue figure = the player at full size")
    if walk:
        parts.append("green floor = walks to it, amber = only by a jump, red = enclosed and never reached")
    return "; ".join(parts)
