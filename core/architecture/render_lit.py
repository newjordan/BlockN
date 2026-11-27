"""In-game-style pictures of a block-mode build: a perspective view from a
player's eye, lit the way Minecraft lights a world.

``render.py`` draws orthographic views with flat shading, which suit plans
and walk overlays but make a poor comparison with screenshots taken inside a
game. This module renders what a player standing in the build would see:

* **Perspective rays.** One ray per pixel walks the 0.25 m visual grid cell
  by cell (a 3-D DDA) until it meets a block, so visibility is exact and
  every hit knows its face and where on the block it landed.
* **Minecraft's light.** Light is worked out per block the way the game does
  it: block light spreads from lanterns, torches, candles, campfires and lit
  furnaces, losing one level per block through anything that is not opaque;
  sky light pours down open columns and spreads in through windows and
  doors. A face is lit by the light in front of it, interpolated smoothly
  between block centres, which also darkens inside corners (Minecraft's
  smooth lighting). Block light is warm, sky light neutral.
* **Textures.** Each block face is drawn as a 16 x 16 texel pattern made from
  its block's colour: plank seams, log bark, brick and stone courses, cobble
  and leaf noise, so a face reads as a block rather than a flat swatch.

Pictures only; nothing here changes a build. Numpy and Pillow only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .blockmode import SUB, BlockBuild, kind, parse

_EMIT = {"lantern": 15, "soul_lantern": 10, "torch": 14, "wall_torch": 14, "campfire": 15, "glowstone": 15,
         "sea_lantern": 15, "jack_o_lantern": 15, "shroomlight": 15}
_SKY = np.array([0.62, 0.78, 0.95])
_GRASS = np.array([0.38, 0.55, 0.24])


def _emission(block: str) -> int:
    name, props = parse(block)
    if name in _EMIT:
        if name == "campfire" and props.get("lit", "true") != "true":
            return 0
        return _EMIT[name]
    if name.endswith("candle") and props.get("lit") == "true":
        return 3 * int(props.get("candles", "1"))
    if name in ("furnace", "smoker", "blast_furnace") and props.get("lit") == "true":
        return 13
    return 0


def _opaque(block: str) -> bool:
    name, _ = parse(block)
    if kind(block) != "full":
        return False
    return not (name.endswith("glass") or name.endswith("leaves") or name in ("barrier",))


def light_levels(b: BlockBuild, shape: Tuple[int, int, int], origin: Tuple[int, int, int],
                 sky: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(block light, sky light), 0 to 15 per block, over a block-aligned box
    of ``shape`` blocks whose first block is at ``origin``."""
    opaque = np.zeros(shape, bool)
    leafy = np.zeros(shape, bool)
    block = np.zeros(shape, np.int16)
    ox, oy, oz = origin
    for (x, y, z), blk in b.blocks.items():
        i, j, k = x - ox, y - oy, z - oz
        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            continue
        if _opaque(blk):
            opaque[i, j, k] = True
        elif parse(blk)[0].endswith("leaves"):
            leafy[i, j, k] = True
        e = _emission(blk)
        if e:
            block[i, j, k] = max(block[i, j, k], e)

    def spread(level: np.ndarray, down_free: bool = False) -> np.ndarray:
        level = level.copy()
        cost = 1 + leafy.astype(np.int16)
        for _ in range(15):
            best = np.zeros_like(level)
            for axis in range(3):
                for sign in (1, -1):
                    sh = np.roll(level, sign, axis=axis)
                    sl = [slice(None)] * 3
                    sl[axis] = 0 if sign == 1 else -1
                    sh[tuple(sl)] = 0
                    if down_free and axis == 1 and sign == -1:
                        # Full sky light falls straight down without fading.
                        sh = np.where(sh == 15, 16, sh)
                    best = np.maximum(best, sh)
            new = np.where(opaque, 0, np.maximum(level, best - cost))
            new = np.minimum(new, 15)
            if np.array_equal(new, level):
                break
            level = new
        return level

    blight = spread(block)
    if sky:
        # Open to the sky: every cell with nothing opaque above it.
        above = np.flip(np.logical_or.accumulate(np.flip(opaque | leafy, axis=1), axis=1), axis=1)
        covered = np.zeros_like(above)
        covered[:, :-1, :] = above[:, 1:, :]
        s0 = np.where(~covered & ~opaque, 15, 0).astype(np.int16)
        slight = spread(s0, down_free=True)
    else:
        slight = np.zeros(shape, np.int16)
    return blight, slight


def _brightness(level: np.ndarray) -> np.ndarray:
    """Minecraft's light curve, level 0 to 15 to brightness 0 to 1, with a
    little ambient so the darkest corner is not black."""
    f = np.clip(level, 0, 15) / 15.0
    return 0.035 + 0.965 * f / (4.0 - 3.0 * f)


def _hash(*a: np.ndarray) -> np.ndarray:
    h = np.zeros(np.broadcast(*a).shape, dtype=np.int64)
    for k, v in enumerate(a):
        h = h ^ (np.asarray(v, dtype=np.int64) * (73856093, 19349663, 83492791, 2654435761, 97531)[k % 5])
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 1023) / 1023.0


def _texture(names: Sequence[str], lab: np.ndarray, u: np.ndarray, v: np.ndarray, bx: np.ndarray,
             by: np.ndarray, bz: np.ndarray, axis: np.ndarray, log_axis: np.ndarray) -> np.ndarray:
    """A brightness factor per pixel from a 16 x 16 texel pattern chosen by
    the block's name. ``u, v`` are 0..1 across the face, v up the face (or
    along z on top faces)."""
    tu = np.floor(u * 16).astype(np.int64)
    tv = np.floor(v * 16).astype(np.int64)
    noise = _hash(bx, by, bz, tu, tv)
    f = 1.0 + (noise - 0.5) * 0.10
    fam = np.array([_family(n) for n in names] + ["plain"])[np.minimum(lab, len(names))]
    planks = fam == "planks"
    if planks.any():
        row = tv // 4
        seam = (tv % 4 == 3)
        shift = (_hash(bx, by, bz, row) * 16).astype(np.int64)
        joint = ((tu + shift) % 16 == 0)
        grain = 1.0 + (_hash(bx, by, bz, row, tu // 3) - 0.5) * 0.10
        f = np.where(planks, grain * np.where(seam | joint, 0.80, 1.0) * (1 + (noise - .5) * .05), f)
    logs = fam == "log"
    if logs.any():
        la = log_axis[np.minimum(lab, len(log_axis) - 1)]
        end = logs & (axis == la)
        along_v = ((axis != 1) & (la == 1)) | ((axis == 1) & (la == 2))
        stripe = np.where(along_v, tu, tv)
        bark = 1.0 + (_hash(bx, by, bz, stripe) - 0.5) * 0.30
        f = np.where(logs & ~end, bark * (1 + (noise - .5) * .08), f)
        ring = np.hypot(u - .5, v - .5)
        f = np.where(end, 0.9 + 0.12 * np.cos(ring * 28), f)
    bricks = fam == "bricks"
    if bricks.any():
        row = tv // 4
        off = (row % 2) * 4
        mortar = (tv % 4 == 0) | ((tu + off) % 8 == 0)
        f = np.where(bricks, np.where(mortar, 1.25, 0.95 + (noise - .5) * .12), f)
    stone = fam == "stonebricks"
    if stone.any():
        row = tv // 8
        off = (row % 2) * 8
        mortar = (tv % 8 == 0) | ((tu + off) % 16 == 0)
        f = np.where(stone, np.where(mortar, 0.78, 1.0 + (noise - .5) * .16), f)
    rough = fam == "rough"
    if rough.any():
        blot = _hash(bx, by, bz, tu // 3, tv // 3)
        f = np.where(rough, 0.82 + 0.3 * blot + (noise - .5) * .1, f)
    leaves = fam == "leaves"
    if leaves.any():
        f = np.where(leaves, np.where(noise > .72, 0.55, 0.9 + (noise - .5) * .4), f)
    return f


def _family(name: str) -> str:
    if name.endswith("|d"):
        return "plain"    # painted detail (books, lids, flames): flat colour, light noise
    n = name.split("|")[0].split(":")[-1].split("[")[0]
    if n.endswith("_planks") or n.endswith("_stairs") and "stone" not in n and "brick" not in n \
            and "cobble" not in n or n.endswith("_slab") and "stone" not in n and "brick" not in n:
        return "planks"
    if n.endswith("_log") or n.endswith("_wood"):
        return "log"
    if n in ("bricks", "brick_stairs", "brick_slab", "brick_wall", "mud_bricks", "chimney"):
        return "bricks"
    if "stone_brick" in n or "deepslate_brick" in n or "deepslate_tile" in n or "polished" in n:
        return "stonebricks"
    if "cobble" in n or n in ("andesite", "stone", "tuff", "gravel", "coarse_dirt", "dirt", "podzol",
                              "calcite", "white_terracotta", "terracotta", "moss_block"):
        return "rough"
    if n.endswith("leaves") or n in ("grass_block", "vine"):
        return "leaves"
    return "plain"


def _opaque_grid(b: BlockBuild, shape: Tuple[int, int, int], origin: Tuple[int, int, int]) -> np.ndarray:
    occ = np.zeros(shape, bool)
    for (x, y, z), blk in b.blocks.items():
        i, j, k = x - origin[0], y - origin[1], z - origin[2]
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2] and _opaque(blk):
            occ[i, j, k] = True
    return occ


def _occlusion(occ: np.ndarray, pt: np.ndarray, nrm: np.ndarray, ax: np.ndarray,
               origin: Tuple[int, int, int]) -> np.ndarray:
    """A factor 0.55 to 1 per hit: darker near edges and corners of the face
    where opaque blocks stand beside the cell the face looks into."""
    cell = np.floor(pt + nrm * 0.5).astype(np.int64) - np.array(origin) + 1
    frac = pt - np.floor(pt + nrm * 0.5)
    lim = np.array(occ.shape) - 1
    out = np.ones(len(pt))
    tang = np.where(ax[:, None] == 0, [1, 2], np.where(ax[:, None] == 1, [0, 2], [0, 1]))
    r = np.arange(len(pt))
    shade = []
    for t in (0, 1):
        a = tang[:, t]
        f = frac[r, a]
        for side, dist in ((-1, f), (1, 1 - f)):
            c = cell.copy()
            c[r, a] += side
            c = np.clip(c, 0, lim)
            o = occ[c[:, 0], c[:, 1], c[:, 2]]
            shade.append(np.where(o, np.clip(1 - dist / 0.5, 0, 1), 0.0))
    s = np.sum(shade, axis=0)
    return 1 - 0.22 * np.clip(s, 0, 2)


def _point_lights(b: BlockBuild, occ: np.ndarray, origin: Tuple[int, int, int], pt: np.ndarray,
                  nrm: np.ndarray, eye: np.ndarray, reach: float = 12.0, radius: float = 7.0) -> np.ndarray:
    """Direct light at each hit from the emitting blocks near the eye, with
    hard shadows from opaque blocks (sampled every 0.25 m)."""
    src = [(np.array([x + .5, y + .5, z + .5]), _emission(blk) / 15.0) for (x, y, z), blk in b.blocks.items()
           if _emission(blk) and np.linalg.norm(np.array([x + .5, y + .5, z + .5]) - eye) < reach]
    total = np.zeros(len(pt))
    lim = np.array(occ.shape) - 1
    off = np.array(origin) - 1
    start = pt + nrm * 0.3
    for pos, power in src:
        vec = pos - start
        dist = np.linalg.norm(vec, axis=1)
        near = dist < radius
        if not near.any():
            continue
        lam = np.clip(np.einsum("ij,ij->i", vec, nrm) / np.maximum(dist, 1e-6), 0, 1)
        fall = power * (0.35 + 0.65 * lam) / (1 + (dist / 1.6) ** 2) * np.clip(1 - dist / radius, 0, 1)
        lit = near & (fall > 0.005)
        idx = np.nonzero(lit)[0]
        steps = int(radius / 0.25) + 1
        blocked = np.zeros(len(idx), bool)
        for k in range(steps):
            t = k * 0.25 + 0.0 * dist[idx]
            go = t < dist[idx] - 0.45
            if not go.any():
                break
            q = start[idx] + vec[idx] / dist[idx, None] * t[:, None]
            c = np.clip(np.floor(q).astype(np.int64) - off, 0, lim)
            blocked |= go & occ[c[:, 0], c[:, 1], c[:, 2]]
        total[idx] += np.where(blocked, 0.0, fall[idx])
    return total


def _sun_visible(occ: np.ndarray, origin: Tuple[int, int, int], start: np.ndarray, sd: np.ndarray,
                 want: np.ndarray, step: float = 0.4) -> np.ndarray:
    """1 where the sun is not blocked by an opaque block, marching from each
    start point towards the sun until it leaves the build's box."""
    out = np.zeros(len(start))
    idx = np.nonzero(want)[0]
    if not len(idx):
        return out
    off = np.array(origin) - 1
    shp = np.array(occ.shape)
    blocked = np.zeros(len(idx), bool)
    live = np.ones(len(idx), bool)
    pts = start[idx].copy()
    for _ in range(int(np.linalg.norm(shp) / step) + 2):
        if not live.any():
            break
        pts[live] += sd * step
        c = np.floor(pts).astype(np.int64) - off
        inside = np.all((c >= 0) & (c < shp), axis=1)
        live &= inside
        cc = np.clip(c, 0, shp - 1)
        hit = live & occ[cc[:, 0], cc[:, 1], cc[:, 2]]
        blocked |= hit
        live &= ~hit
    out[idx] = np.where(blocked, 0.0, 1.0)
    return out


def render_view(b: BlockBuild, eye: Sequence[float], look_at: Sequence[float], *, fov: float = 70.0,
                width: int = 1280, height: int = 720, sky: bool = True, sun: float = 0.85,
                warm: Tuple[float, float, float] = (1.0, 0.82, 0.55), exposure: float = 1.15,
                glow: float = 0.0, sun_dir: Optional[Sequence[float]] = None,
                ground_y: float = 0.0) -> Image.Image:
    """What a player with eyes at ``eye`` sees looking at ``look_at`` (world
    metres; block (x, y, z) fills x..x+1). ``fov`` is the vertical field of
    view in degrees, as in Minecraft's settings."""
    g = b.grid(SUB, visual=True)
    labels = g.labels
    nx, ny, nz = labels.shape
    p = g.pitch
    origin = np.array(g.origin, float)
    names = [""] * (int(labels.max()) + 1)
    rgb_of = np.zeros((len(names) + 1, 3))
    for e in g.palette:
        names[e["label"]] = e["minecraft"] + ("|d" if any(
            other["minecraft"] == e["minecraft"] and other["label"] < e["label"] for other in g.palette) else "")
        rgb_of[e["label"]] = np.array(e["rgb"], float) / 255.0
    names_list = names[1:]
    glass = np.zeros(len(names) + 1, bool)
    thin = np.zeros(len(names) + 1, bool)
    lantern = np.zeros(len(names) + 1, bool)
    emissive = np.zeros(len(names) + 1, float)
    for e in g.palette:
        nm, props = parse(e["minecraft"])
        if nm.endswith("glass_pane") or nm.endswith("glass"):
            glass[e["label"]] = True
        if kind(e["minecraft"]) in ("carpet", "plate"):
            thin[e["label"]] = True
        bright = (e["rgb"][0] > 200 and e["rgb"][1] > 120 and e["rgb"][2] < 120)
        if bright and _emission(e["minecraft"]) or (bright and nm in ("furnace", "smoker")
                                                     and props.get("lit") == "true"):
            emissive[e["label"]] = 1.0

    # Block light and sky light on a block grid aligned with the visual grid.
    bshape = (nx // SUB, ny // SUB, nz // SUB)
    borigin = (int(round(origin[0])), int(round(origin[1])), int(round(origin[2])))
    blight, slight = light_levels(b, bshape, borigin, sky=sky)
    # Pad by one block so interpolation at the edges reads open air.
    bl = np.pad(blight, 1, constant_values=0).astype(float)
    occ = np.pad(_opaque_grid(b, bshape, borigin), 1, constant_values=False)
    sl = np.pad(slight, 1, constant_values=15 if sky else 0).astype(float)

    # Camera.
    eye = np.asarray(eye, float)
    fwd = np.asarray(look_at, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    th = math.tan(math.radians(fov) / 2)
    aspect = width / height
    jj, ii = np.meshgrid(np.arange(width), np.arange(height))
    sx = ((jj + .5) / width * 2 - 1) * th * aspect
    sy = (1 - (ii + .5) / height * 2) * th
    dirs = fwd[None, None] + sx[..., None] * right[None, None] + sy[..., None] * up[None, None]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    d = dirs.reshape(-1, 3)
    n = len(d)

    # DDA in grid cells.
    pos = (eye - origin) / p
    cell = np.floor(np.broadcast_to(pos, (n, 3))).astype(np.int64)
    step = np.where(d > 0, 1, -1).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(d != 0, 1.0 / d, np.inf)
    nxt = np.where(d > 0, cell + 1, cell) - pos
    tmax = np.abs(nxt * inv) * 1.0
    tmax = np.where(d != 0, nxt * inv, np.inf)
    tdelta = np.abs(inv)
    hit_lab = np.zeros(n, np.int64)
    hit_axis = np.full(n, -1, np.int64)
    hit_t = np.full(n, np.inf)
    through_glass = np.zeros(n, bool)
    active = np.arange(n)
    last_axis = np.full(n, 1, np.int64)
    limit = int(3 * (nx + ny + nz))
    shp = np.array([nx, ny, nz])
    for _ in range(limit):
        if not len(active):
            break
        c = cell[active]
        inside = np.all((c >= 0) & (c < shp), axis=1)
        lab = np.zeros(len(active), np.int64)
        lab[inside] = labels[c[inside, 0], c[inside, 1], c[inside, 2]]
        gl = glass[lab]
        through_glass[active[gl]] = True
        hit = (lab > 0) & ~gl
        # Carpets and pressure plates are a sixteenth of a block thick, not
        # a whole picture cell: hit their top face or pass over them.
        th = thin[lab] & hit
        if th.any():
            ia = active[th]
            plane = c[th, 1] + 0.25
            dy = d[ia, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                tp = (plane - pos[1]) / dy
            q = pos[None] + d[ia] * tp[:, None]
            inside_cell = (dy < 0) & np.all(np.abs(q[:, [0, 2]] - (c[th][:, [0, 2]] + 0.5)) <= 0.5, axis=1)
            ok = inside_cell
            hit_t[ia[ok]] = tp[ok]
            hit_axis[ia[ok]] = 1
            hit_lab[ia[ok]] = lab[th][ok]
            thin_hit = np.zeros(len(active), bool)
            thin_hit[np.nonzero(th)[0][ok]] = True
            gone_thin = np.zeros(len(active), bool)
            gone_thin[np.nonzero(th)[0]] = True
            hit = hit & ~gone_thin
            done_thin = thin_hit
        else:
            done_thin = np.zeros(len(active), bool)
        if hit.any():
            h = active[hit]
            hit_lab[h] = lab[hit]
            hit_axis[h] = last_axis[h]
            ax = last_axis[h]
            prev = tmax[h, ax] - tdelta[h, ax]
            hit_t[h] = np.where(np.isfinite(prev), prev, 0.0)
        # Leave rays that have hit or left the grid heading away.
        gone = ~inside & np.any(((c < 0) & (step[active] < 0)) | ((c >= shp) & (step[active] > 0)), axis=1)
        keep = ~hit & ~gone & ~done_thin
        active = active[keep]
        if not len(active):
            break
        tm = tmax[active]
        ax = np.argmin(tm, axis=1)
        r = np.arange(len(active))
        cell[active, ax] += step[active, ax]
        tmax[active, ax] += tdelta[active, ax]
        last_axis[active] = ax

    # Shade the hits.
    out = np.zeros((n, 3))
    got = hit_lab > 0
    t = hit_t[got] * p
    pt = eye[None] + d[got] * t[:, None]
    ax = hit_axis[got]
    nrm = np.zeros((len(ax), 3))
    nrm[np.arange(len(ax)), ax] = -step[got][np.arange(len(ax)), ax]
    # Where on its block the ray landed: u, v across the face.
    rel = pt - np.floor(pt + nrm * -1e-4) * 1.0
    blk = np.floor(pt - nrm * 1e-4).astype(np.int64)
    fx = pt - np.floor(pt - nrm * 1e-4)
    u = np.where(ax == 0, fx[:, 2], fx[:, 0])
    v = np.where(ax == 1, fx[:, 2], fx[:, 1])
    lab = hit_lab[got]
    base = rgb_of[lab]
    log_axis = np.array([{"x": 0, "y": 1, "z": 2}.get(parse(nm.split("|")[0])[1].get("axis", "y"), 1)
                         if nm else 1 for nm in names_list] + [1])
    tex = _texture(names_list, lab - 1, u, v, blk[:, 0], blk[:, 1], blk[:, 2], ax, log_axis)
    face = np.select([ax == 1, ax == 0], [np.where(nrm[:, 1] > 0, 1.0, 0.5), 0.62], 0.8)
    # Smooth light: trilinear between block centres, half a block out from the face.
    q = pt + nrm * 0.5 - np.array(borigin, float) + 1.0 - 0.5   # in padded block-centre coords
    i0 = np.floor(q).astype(np.int64)
    fr = q - i0
    lim = np.array(bl.shape) - 2
    i0 = np.clip(i0, 0, lim)
    acc_b = np.zeros(len(q))
    acc_s = np.zeros(len(q))
    wsum = np.zeros(len(q))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                wgt = (np.where(dx, fr[:, 0], 1 - fr[:, 0]) * np.where(dy, fr[:, 1], 1 - fr[:, 1])
                       * np.where(dz, fr[:, 2], 1 - fr[:, 2]))
                ii_, jj_, kk_ = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                lb, ls = bl[ii_, jj_, kk_], sl[ii_, jj_, kk_]
                acc_b += wgt * lb
                acc_s += wgt * ls
                wsum += wgt
    lb_ = acc_b / np.maximum(wsum, 1e-6)
    ls_ = acc_s / np.maximum(wsum, 1e-6)
    sky_b = _brightness(ls_) * sun
    if sun_dir is not None:
        sd = np.asarray(sun_dir, float)
        sd = sd / np.linalg.norm(sd)
        lam = np.clip(nrm @ sd, 0, 1)
        lit = _sun_visible(occ, borigin, pt + nrm * 0.05, sd, lam > 0)
        # Ambient sky stays (softer); the sun adds a warm key light.
        sky_b = sky_b * (0.55 + 0.75 * lam * lit)
    light = (_brightness(lb_)[:, None] * np.array(warm)[None]
             + sky_b[:, None] * np.array([1.0, 0.97, 0.92])[None])
    if glow > 0:
        light = light + glow * _point_lights(b, occ, borigin, pt, nrm, eye)[:, None] * np.array(warm)[None]
    light = np.minimum(light, 1.35)
    col = base * tex[:, None] * face[:, None] * light
    # Ambient occlusion: Minecraft darkens a face towards each edge or corner
    # where an opaque block sits beside the air the face looks into.
    col = col * _occlusion(occ, pt, nrm, ax, borigin)[:, None]
    em = emissive[lab]
    glow = base * 1.15 * np.array([1.08, 1.0, 0.9])[None]
    # Lanterns: iron bars across the glass, at the visual grid's quarter-metre cells.
    fq = (pt * SUB) % 1.0
    fu = np.where(ax == 0, fq[:, 2], fq[:, 0])
    fv = np.where(ax == 1, fq[:, 2], fq[:, 1])
    bars = (np.minimum(fu, 1 - fu) < 0.14) | (np.minimum(fv, 1 - fv) < 0.14)
    is_lantern = np.array([parse(nm.split("|")[0])[0] == "lantern" if nm else False for nm in names_list] + [False])
    iron = np.array([0.20, 0.20, 0.23])[None] * light
    glow = np.where((is_lantern[lab - 1] & bars)[:, None], iron, glow)
    col = np.where(em[:, None] > 0, glow, col)
    out[got] = col

    # Missed: sky above the horizon, a lit grass plain below it.
    miss = ~got
    dm = d[miss]
    dusk = np.array([1.0, 0.62, 0.42]) * (1 - sun) + _SKY * sun
    up_ = np.clip(dm[:, 1], 0, 1)[:, None]
    zenith = dusk * np.array([0.55, 0.75, 1.05])
    sky_col = (dusk[None] * (1 - up_) + zenith[None] * up_) * (0.35 + 0.65 * sun)
    # Soft clouds: blotches of white in the upper sky.
    cu = dm[:, 0] / np.maximum(dm[:, 1], 0.05)
    cv = dm[:, 2] / np.maximum(dm[:, 1], 0.05)
    cloud = (_hash(np.floor(cu * 3).astype(np.int64), np.floor(cv * 6).astype(np.int64)) > 0.88) & (dm[:, 1] > 0.08)
    sky_col = np.where(cloud[:, None], 0.92 * (0.35 + 0.65 * sun) + 0 * sky_col, sky_col)
    tg = np.where(dm[:, 1] < -1e-3, -(eye[1] - ground_y) / dm[:, 1], np.inf)
    ground = np.isfinite(tg) & (tg > 0)
    gp = eye[None] + dm * np.where(ground, tg, 0)[:, None]
    gnoise = _hash(np.floor(gp[:, 0] * 16).astype(np.int64), np.floor(gp[:, 2] * 16).astype(np.int64))
    gcol = _GRASS[None] * (0.85 + 0.25 * gnoise)[:, None] * sun
    out[miss] = np.where(ground[:, None], gcol, sky_col)
    out[through_glass] = out[through_glass] * np.array([0.9, 0.95, 1.0])[None] + 0.03

    img = out.reshape(height, width, 3) * exposure
    # A soft vignette, as screenshots have.
    yy, xx = np.mgrid[0:height, 0:width]
    r2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
    img *= (1 - 0.18 * np.clip(r2, 0, 1.4))[..., None]
    img = np.clip(img, 0, 1) ** (1 / 1.1)
    return Image.fromarray((img * 255).astype(np.uint8))


def room_camera(b: BlockBuild, floor_y: int = 1, eye_height: float = 1.62,
                area: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Tuple[float, float, float],
                                                                                   Tuple[float, float, float]]:
    """A player's eye and look-at point for a view across the room whose
    floor is at ``floor_y``: standing in the open floor cell farthest from
    the room's far side, looking at that far side a little below eye level,
    as a screenshot taken on walking in would be. ``area`` (x0, z0, x1, z1)
    limits the search to one room."""
    def clear(x: int, z: int) -> bool:
        if (x, floor_y - 1, z) not in b.blocks or kind(b.blocks[(x, floor_y - 1, z)]) != "full":
            return False
        for dy in (0, 1):
            blk = b.blocks.get((x, floor_y + dy, z))
            if blk is not None and kind(blk) not in ("air", "carpet", "plate", "plant", "wall_banner", "torch", "vine"):
                return False
        return True

    (lx, _, lz), (hx, _, hz) = b.bounds()
    if area is not None:
        lx, lz, hx, hz = area
    cells = [(x, z) for x in range(lx, hx + 1) for z in range(lz, hz + 1) if clear(x, z)]
    # Keep cells with a roof over them: inside the building.
    roofed = [c for c in cells if any((c[0], y, c[1]) in b.blocks for y in range(floor_y + 2, floor_y + 12))]
    cells = roofed or cells
    if not cells:
        raise ValueError("no open floor at that height")
    cs = set(cells)
    open_ = [c for c in cells if sum((c[0] + dx, c[1] + dz) in cs for dx in (-1, 0, 1) for dz in (-1, 0, 1)) >= 6]
    eyes = open_ or cells
    def blocked_near(a: Tuple[int, int], c: Tuple[int, int]) -> bool:
        # Anything solid at eye height in the first three metres of the view
        # fills the picture; a screenshot would not be taken standing there.
        dx, dz = c[0] - a[0], c[1] - a[1]
        n = math.hypot(dx, dz) or 1.0
        for k in range(1, 7):
            t = 0.5 * k
            x, z = int(math.floor(a[0] + 0.5 + dx / n * t)), int(math.floor(a[1] + 0.5 + dz / n * t))
            blk = b.blocks.get((x, floor_y + 1, z))
            if blk is not None and kind(blk) not in ("air", "carpet", "plate", "plant", "wall_banner", "torch",
                                                     "lantern", "chain", "vine"):
                return True
        return False

    def score(ac: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        return (ac[0][0] - ac[1][0]) ** 2 + (ac[0][1] - ac[1][1]) ** 2

    pairs = sorted(((a, c) for a in eyes for c in cells), key=score, reverse=True)
    best = next((ac for ac in pairs[:4000] if not blocked_near(*ac)), pairs[0])
    (ax, az), (cx, cz) = best
    dx, dz = cx - ax, cz - az
    norm = math.hypot(dx, dz) or 1.0
    eye = (ax + 0.5 - 0.3 * dx / norm, floor_y + eye_height, az + 0.5 - 0.3 * dz / norm)
    look = (cx + 0.5, floor_y + 1.55, cz + 0.5)
    return eye, look


__all__ = ["render_view", "light_levels", "room_camera"]
