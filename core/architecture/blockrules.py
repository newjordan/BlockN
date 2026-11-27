"""Score a block build against the construction rulebook in ``lore``.

Every check is measured from the blocks alone, so it scores any build: ours,
a sub-architect's, or a rival's world loaded from a save. A check returns its
measured value, the range it was held to, and whether it passed.

Building checks: load path, roof pitch for its covering, roof oversailing the
walls, chimney, and on each facade depth planes, longest flat run, window
ratio, stacked openings, material families and a base course.

Town checks: street enclosure (street width to fronting eave height), street
frontage, and the market square's size and enclosure.
"""
from __future__ import annotations

import math
from collections import Counter, deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import lore
from .blockmode import BlockBuild, kind, parse

Pos = Tuple[int, int, int]

STRUCT = {"full", "slab", "stairs", "wall", "fence", "pane", "door", "trapdoor", "gate"}
OPENING = {"pane", "door"}
_WOOD = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry", "bamboo", "crimson",
         "warped", "log", "wood", "planks")
_MASONRY = ("brick", "cobble", "stone", "deepslate", "blackstone", "andesite", "diorite", "granite", "tuff",
            "calcite", "sandstone", "mud", "terracotta", "concrete", "quartz")
_FAMILY_STRIP = ("mossy_", "cracked_", "chiseled_", "polished_", "smooth_", "cut_", "stripped_", "infested_",
                 "weathered_", "exposed_", "oxidized_", "waxed_")
_FAMILY_SUFFIX = ("_stairs", "_slab", "_wall", "_fence_gate", "_fence", "_trapdoor", "_door", "_pane", "_planks",
                  "_log", "_wood", "_bricks", "_brick", "_tiles", "_tile", "s")


_ORGANIC = ("_leaves", "hay_block", "moss_block", "_wool", "melon", "pumpkin", "bee_nest", "cake")


def structural(block: str) -> bool:
    """A block that carries or is carried: not decoration, plants, cloth or
    leaves, which are hung, potted or grown."""
    n = parse(block)[0]
    return kind(block) in STRUCT and not any(n.endswith(o) or o in n for o in _ORGANIC if o != "hay_block")


def name_of(block: str) -> str:
    return parse(block)[0].split(":")[-1]


def family(block: str) -> str:
    """A material family: mossy, cracked and chiseled stone bricks are one
    family; a stair or slab belongs to its block's family."""
    n = name_of(block)
    changed = True
    while changed:
        changed = False
        for p in _FAMILY_STRIP:
            if n.startswith(p):
                n, changed = n[len(p):], True
        for s in _FAMILY_SUFFIX:
            if n.endswith(s) and len(n) > len(s) + 2:
                n, changed = n[: -len(s)], True
                break
    if "glass" in n:
        return "glass"
    return n


def is_wood(block: str) -> bool:
    """Timber, or a roof covering, which lies on timber rafters."""
    n = name_of(block)
    if kind(block) in ("stairs", "slab") and lore.covering_of(block):
        return True
    return any(w in n for w in _WOOD) and not any(m in n for m in ("stone", "brick"))


def _check(name: str, value: Any, passed: bool, want: Any, note: str = "") -> Dict[str, Any]:
    return {"rule": name, "value": value, "want": want, "pass": bool(passed), "note": note}


# ------------------------------------------------------------- load path
def load_path(b: BlockBuild, span_wood: int = 9, span_stone: int = 5) -> Dict[str, Any]:
    """Which structural blocks stand on something. A block is carried when
    the block below it is carried; when it reaches out no further than its
    material's cantilever (two blocks for timber, one for stone) from a
    carried block in its layer or diagonally below; or when it lies on a
    straight run between two carried blocks no longer than a beam or lintel
    of its material can span."""
    hung = set(getattr(b, "mechanisms", ()) or ())
    solid: Dict[Pos, str] = {p: blk for p, blk in b.blocks.items() if structural(blk) and p not in hung}
    if not solid:
        return {"structural": 0, "floating": 0, "share": 1.0, "examples": []}
    ys = sorted({p[1] for p in solid})
    y0 = ys[0]
    ok: Set[Pos] = {p for p in solid if p[1] == y0}
    layers: Dict[int, List[Pos]] = {}
    for p in solid:
        layers.setdefault(p[1], []).append(p)
    nb4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for y in ys[1:]:
        layer = set(layers[y])
        reach: Dict[Pos, int] = {}
        q: deque = deque()
        for (x, _, z) in layer:
            if (x, y - 1, z) in ok:
                reach[(x, y, z)] = 0
                q.append((x, y, z))
        for (x, _, z) in layer:
            if (x, y, z) not in reach and any((x + dx, y - 1, z + dz) in ok for dx, dz in nb4):
                reach[(x, y, z)] = 1
                q.append((x, y, z))
        # Beams and lintels: a straight run between two carried blocks.
        anchored = set(reach)
        for (x, _, z) in layer:
            if (x, y, z) in reach:
                continue
            span = span_wood if is_wood(solid[(x, y, z)]) else span_stone
            for dx, dz in ((1, 0), (0, 1)):
                ends = 0
                length = 1
                for sgn in (1, -1):
                    k = 1
                    while k <= span:
                        c = (x + sgn * dx * k, y, z + sgn * dz * k)
                        if c not in layer:
                            break
                        if c in anchored:
                            ends += 1
                            break
                        k += 1
                    length += k
                if ends == 2 and length - 1 <= span + 1:
                    reach[(x, y, z)] = 0
                    q.append((x, y, z))
                    break
        while q:
            c = q.popleft()
            r = reach[c]
            for dx, dz in nb4:
                n = (c[0] + dx, y, c[2] + dz)
                if n in layer:
                    lim = 2 if is_wood(solid[n]) else 1
                    if r + 1 <= lim and reach.get(n, 99) > r + 1:
                        reach[n] = r + 1
                        q.append(n)
        for c, r in reach.items():
            if r <= (2 if is_wood(solid[c]) else 1):
                ok.add(c)
    floating = [p for p in solid if p not in ok]
    share = 1 - len(floating) / len(solid)
    return {"structural": len(solid), "floating": len(floating), "share": round(share, 4),
            "examples": sorted(floating)[:8]}


# ------------------------------------------------------------------ roof
def roof(b: BlockBuild) -> Dict[str, Any]:
    """The top block of every column: which covering it reads as, the
    roof's pitch from the slope of those tops, and where the ridge is."""
    tops: Dict[Tuple[int, int], Tuple[int, str]] = {}
    for (x, y, z), blk in b.blocks.items():
        if kind(blk) not in ("full", "slab", "stairs"):
            continue
        t = tops.get((x, z))
        if t is None or y > t[0]:
            tops[(x, z)] = (y, blk)
    cells = {c: t for c, t in tops.items() if lore.covering_of(t[1])}
    if not cells:
        return {"covering": None, "pitch": None, "cells": 0}
    cov = Counter(lore.covering_of(t[1]) for t in cells.values()).most_common(1)[0][0]
    if lore.COVERINGS.get(cov, {}).get("parapet"):
        # A flat roof's parapet is a thin upstand round its edge: a cell one
        # block or more above the roof on one side, with the roof falling
        # away or ending on the other. It is a wall, not a slope of the roof.
        def _upstand(x: int, z: int, y: int) -> bool:
            for dx, dz in ((1, 0), (0, 1)):
                a, c = cells.get((x - dx, z - dz)), cells.get((x + dx, z + dz))
                for i, o in ((a, c), (c, a)):
                    if i is not None and i[0] < y and (o is None or o[0] < y):
                        return True
            return False
        cells = {c: t for c, t in cells.items() if not _upstand(c[0], c[1], t[0])} or cells
    slopes = []
    for (x, z), (y, blk) in cells.items():
        h = y + (0.5 if kind(blk) in ("slab", "stairs") else 1.0)
        best = 0.0
        for dx, dz in ((1, 0), (0, 1)):
            a, c = cells.get((x - dx, z - dz)), cells.get((x + dx, z + dz))
            if a and c:
                ha = a[0] + (0.5 if kind(a[1]) in ("slab", "stairs") else 1.0)
                hc = c[0] + (0.5 if kind(c[1]) in ("slab", "stairs") else 1.0)
                best = max(best, abs(hc - ha) / 2)
        # A jump of more than two blocks a block is a wall (a terrace's
        # setback, a parapet, a chimney), not a roof slope.
        if best <= 2.0:
            slopes.append(best)
    if not slopes:
        slopes = [0.0]
    slopes.sort()
    med = slopes[len(slopes) // 2]
    # Ridge cells read flat; take the median over sloped cells as the pitch.
    sloped = [s for s in slopes if s > 0.1]
    pitch = math.degrees(math.atan(sloped[len(sloped) // 2])) if len(sloped) > len(slopes) * 0.3 else \
        math.degrees(math.atan(med))
    ridge = max(t[0] for t in cells.values())
    return {"covering": cov, "pitch": round(pitch, 1), "cells": len(cells), "ridge": ridge,
            "tops": cells}


# --------------------------------------------------------------- facades
def _walls_footprint(b: BlockBuild, y0: int) -> Set[Tuple[int, int]]:
    cells = set()
    for (x, y, z), blk in b.blocks.items():
        if y0 + 1 <= y <= y0 + 2 and kind(blk) in STRUCT:
            cells.add((x, z))
    return cells


def _outside(foot: Set[Tuple[int, int]], tops: Dict[Tuple[int, int], Any]) -> Set[Tuple[int, int]]:
    """Cells reachable from beyond the building without crossing a wall:
    the outside, as against a courtyard or the rooms inside."""
    cells = set(foot) | set(tops)
    if not cells:
        return set()
    xa, xb = min(c[0] for c in cells) - 1, max(c[0] for c in cells) + 1
    za, zb = min(c[1] for c in cells) - 1, max(c[1] for c in cells) + 1
    seen = {(xa, za)}
    todo = [(xa, za)]
    while todo:
        x, z = todo.pop()
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = (x + dx, z + dz)
            if xa <= c[0] <= xb and za <= c[1] <= zb and c not in seen and c not in foot:
                seen.add(c)
                todo.append(c)
    return seen


def facades(b: BlockBuild, storey: Optional[int] = None) -> Dict[str, Any]:
    """March in from each of the four sides, over every column of the wall
    zone (one block above the lowest layer up to the eaves), to the first
    block that is not air or decoration. That surface is the facade seen
    from outside: its depth planes, longest flat run, window ratio, stacked
    openings, materials and base course. An open trapdoor beside a window
    is a shutter lying against the wall, not the wall itself."""
    solid = {p: blk for p, blk in b.blocks.items() if kind(blk) in STRUCT
             and not (kind(blk) == "trapdoor" and parse(blk)[1].get("open") == "true")}
    if not solid:
        return {}
    (x0, y0, z0), (x1, y1, z1) = (min(p[0] for p in solid), min(p[1] for p in solid), min(p[2] for p in solid)), \
        (max(p[0] for p in solid), max(p[1] for p in solid), max(p[2] for p in solid))
    S = storey or getattr(getattr(b, "sizes", None), "storey", 4) or 4
    r = roof(b)
    tops = r.get("tops") or {}
    # The eave: the lowest roof-covering top over the perimeter of the walls.
    foot = _walls_footprint(b, y0)
    eave = y1
    if tops:
        # Only the outside counts: a courtyard's own roofs and stairs are
        # inside the walls, not the eave the street sees.
        outside = _outside(foot, tops)
        edge = [t[0] for c, t in tops.items() if c in outside or (c in foot and any(
            (c[0] + dx, c[1] + dz) in outside for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))))]
        if edge:
            eave = min(edge)
    out: Dict[str, Any] = {}
    for side, (axis, sgn) in {"north": ("z", -1), "south": ("z", 1), "west": ("x", -1), "east": ("x", 1)}.items():
        hits: Dict[Tuple[int, int], Tuple[int, str]] = {}
        us = range(x0, x1 + 1) if axis == "z" else range(z0, z1 + 1)
        depth_span = (z0, z1) if axis == "z" else (x0, x1)
        for u in us:
            for y in range(y0, max(y0 + 2, eave)):
                rng = range(depth_span[0], depth_span[1] + 1) if sgn < 0 else range(depth_span[1], depth_span[0] - 1, -1)
                for d, w in enumerate(rng):
                    p = (u, y, w) if axis == "z" else (w, y, u)
                    blk = solid.get(p)
                    if blk is not None:
                        hits[(u, y)] = (d, blk)
                        break
        if len(hits) < 6:
            continue
        n = len(hits)
        depths = Counter(d for d, _ in hits.values())
        planes = sorted(d for d, k in depths.items() if k >= max(2, 0.05 * n))
        # The wall plane is the deepest plane with a big share; reliefs stand out of it.
        relief = (planes[-1] - planes[0]) if planes else 0
        openings = sum(1 for _, blk in hits.values() if kind(blk) in OPENING or "glass" in name_of(blk))
        fams = Counter(family(blk) for _, blk in hits.values() if kind(blk) not in OPENING)
        nf = sum(fams.values()) or 1
        families = sorted(f for f, k in fams.items() if k >= 0.05 * nf)
        # Largest blank panel: the biggest rectangle of facade at one depth,
        # in one material family, with no opening in it. A timber frame is a
        # lattice, not a panel, so its posts and rails never add up to one.
        flat = 0
        flat_at = None
        groups: Dict[Tuple[int, str], Set[Tuple[int, int]]] = {}
        for c, (d, blk) in hits.items():
            if kind(blk) not in OPENING and "glass" not in name_of(blk):
                groups.setdefault((d, family(blk)), set()).add(c)
        ylo = min(y for _, y in hits)
        yhi = max(y for _, y in hits)
        for (d, fam), cells in groups.items():
            if len(cells) <= flat:
                continue
            heights = {u: 0 for u in us}
            for y in range(ylo, yhi + 1):
                for u in us:
                    heights[u] = heights[u] + 1 if (u, y) in cells else 0
                stack: List[Tuple[int, int]] = []
                ulist = list(us) + [None]
                for i, u in enumerate(ulist):
                    h = heights[u] if u is not None else 0
                    start = i
                    while stack and stack[-1][1] >= h:
                        si, sh = stack.pop()
                        # A strip one or two blocks wide (a post, a chimney, a pier
                        # between windows) is not a panel.
                        area = sh * (i - si) if sh >= 3 and i - si >= 3 else 0
                        if area > flat:
                            flat, flat_at = area, {"cell": (ulist[si], y), "depth": d, "family": fam,
                                                   "size": [i - si, sh]}
                        start = si
                    stack.append((start, h))
        # Stacked openings: each window column of a storey above the ground
        # has an opening in the same column in the storey below.
        cols: Dict[int, Set[int]] = {}
        for (u, y), (d, blk) in hits.items():
            if kind(blk) in OPENING or "glass" in name_of(blk):
                cols.setdefault((y - y0) // S, set()).add(u)
        aligned = total = 0
        for k in sorted(cols):
            if k - 1 in cols:
                for u in cols[k]:
                    total += 1
                    aligned += any(abs(u - v) <= 0 for v in cols[k - 1])
        # Base course: the bottom wall row in another family or plane than the row above it.
        low = [hits[(u, y0)] for u in us if (u, y0) in hits]
        mid = [hits[(u, y0 + 1 + S // 2)] for u in us if (u, y0 + 1 + S // 2) in hits]
        lowf = Counter(family(bk) for _, bk in low).most_common(1)
        midf = Counter(family(bk) for _, bk in mid).most_common(1)
        base = bool(lowf and midf and (lowf[0][0] != midf[0][0] or
                                        Counter(d for d, _ in low).most_common(1)[0][0] !=
                                        Counter(d for d, _ in mid).most_common(1)[0][0]))
        out[side] = {"cells": n, "planes": len(planes), "relief": relief, "flat_run": flat, "flat_at": flat_at,
                     "window_ratio": round(openings / n, 3), "stacked": (round(aligned / total, 2) if total else None),
                     "families": families, "base": base}
    return {"eave": eave, "storey": S, "sides": out}


# ------------------------------------------------------------- the score
def score_building(b: BlockBuild, tradition: Optional[str] = None, purpose: Optional[str] = None) -> Dict[str, Any]:
    """Every building rule, measured. Returns the checks and the share passed."""
    t = lore.TRADITIONS.get(tradition) if tradition else None
    checks: List[Dict[str, Any]] = []
    lp = load_path(b)
    checks.append(_check("load-path", lp["share"], lp["floating"] == 0, "every block carried",
                         f"{lp['floating']} of {lp['structural']} blocks float"))
    r = roof(b)
    if r["covering"]:
        lo, hi = lore.COVERINGS[r["covering"]]["pitch"]
        checks.append(_check("roof-pitch", r["pitch"], lo - 3 <= r["pitch"] <= hi + 3, f"{lo} to {hi} for {r['covering']}"))
    else:
        checks.append(_check("roof-pitch", None, False, "a roof covering", "no roof covering found"))
    if t and r["covering"]:
        checks.append(_check("covering-fits", r["covering"], r["covering"] in t.coverings,
                             " or ".join(t.coverings), f"what {t.name} roofs were covered with"))
    y0 = min(p[1] for p in b.blocks) if b.blocks else 0
    foot = _walls_footprint(b, y0)
    tops = r.get("tops") or {}
    over = sum(1 for c in tops if c not in foot)
    checks.append(_check("roof-covers", over, over > 0 or r.get("covering") == "flat-earth",
                         "roof cells beyond the walls"))
    ridge = r.get("ridge")
    chim = ridge is not None and any(
        y >= ridge and lore.covering_of(blk) is None and kind(blk) == "full" and
        any(m in name_of(blk) for m in ("brick", "cobble", "stone", "campfire"))
        for (x, y, z), blk in b.blocks.items()) or any(name_of(blk) == "campfire" for blk in b.blocks.values())
    needs_chimney = (lore.PURPOSES.get(purpose or "", {}).get("chimney", True)
                     and (t is None or t.climate in ("temperate", "cold")) and (t is None or t.key != "norse"))
    if needs_chimney:
        checks.append(_check("chimney", chim, chim, "a chimney that clears the ridge"))
    f = facades(b)
    sides = f.get("sides", {})
    pr = lore.PURPOSES.get(purpose or "")
    if pr and foot:
        xs = [c[0] for c in foot]
        zs = [c[1] for c in foot]
        a, c = max(xs) - min(xs) + 1, max(zs) - min(zs) + 1
        front, deep = sorted((a, c))
        (flo, fhi), (dlo, dhi) = pr["footprint"]
        flo, fhi = sorted((flo, dlo))[0], max(fhi, dhi)
        ok_fp = flo - 1 <= front <= fhi + 1 and flo - 1 <= deep <= fhi + 1
        checks.append(_check("footprint", [front, deep], ok_fp, f"{pr['footprint']} for a {purpose}"))
        # A flat roof has no attic: its storeys run to the highest roof, as a
        # terraced block steps up behind a one-storey front.
        top = (r.get("ridge", y0) + 1) if r.get("covering") == "flat-earth" else f.get("eave", y0)
        storeys = max(1, round((top - y0) / max(1, f.get("storey", 4))))
        slo, shi = pr["storeys"]
        checks.append(_check("storeys", storeys, slo <= storeys <= shi + 1, f"{slo} to {shi} (+ attic)"))
    wr_lo, wr_hi = (t.window_ratio() if t else lore.COMPOSITION["window-ratio"]["range"])
    if purpose and "window_ratio" in lore.PURPOSES.get(purpose, {}):
        wr_lo, wr_hi = lore.PURPOSES[purpose]["window_ratio"]
    # A party wall (shared with the next house in a row) or a wall built
    # against rock is not a facade: nobody sees it, so it is not composed.
    party = set(getattr(b, "party_walls", ()) or ())
    for side, s in sorted(sides.items()):
        if side in party:
            continue
        checks.append(_check(f"{side}:facade-depth", s["planes"], s["planes"] >= 2 and s["relief"] >= 1,
                             ">= 2 planes", f"relief {s['relief']}"))
        fa = s.get("flat_at") or {}
        checks.append(_check(f"{side}:flat-run", s["flat_run"], s["flat_run"] <= lore.COMPOSITION["facade-depth"]["flat_run"],
                             f"<= {lore.COMPOSITION['facade-depth']['flat_run']} cells",
                             f"{fa.get('family')} at {fa.get('cell')}" if fa else ""))
        checks.append(_check(f"{side}:window-ratio", s["window_ratio"], wr_lo <= s["window_ratio"] <= wr_hi,
                             f"{wr_lo} to {wr_hi}"))
        if s["stacked"] is not None:
            checks.append(_check(f"{side}:stacked-openings", s["stacked"],
                                 s["stacked"] >= lore.COMPOSITION["stacked-openings"]["min"], ">= 0.6"))
        lo, hi = lore.COMPOSITION["materials"]["range"]
        checks.append(_check(f"{side}:materials", len(s["families"]), lo <= len(s["families"]) <= hi,
                             f"{lo} to {hi}", ", ".join(s["families"])))
        checks.append(_check(f"{side}:base-course", s["base"], s["base"], "a base course"))
    passed = sum(c["pass"] for c in checks)
    return {"product": b.product, "tradition": tradition, "purpose": purpose, "passed": passed,
            "total": len(checks), "score": round(passed / len(checks), 3) if checks else 0.0,
            "pitch": r.get("pitch"), "covering": r.get("covering"), "checks": checks}


def by_rule(results: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """Pass counts per rule (facade sides pooled) over many buildings."""
    tally: Dict[str, List[int]] = {}
    for res in results:
        for c in res["checks"]:
            key = c["rule"].split(":")[-1]
            t = tally.setdefault(key, [0, 0])
            t[0] += c["pass"]
            t[1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(tally.items())}


# ------------------------------------------------------------------ towns
def score_town(b: BlockBuild, era: str = "medieval") -> Dict[str, Any]:
    """Street enclosure, frontage and the square, from a town build."""
    info = getattr(b, "town", None)
    if not info or "use" not in info:
        return {"checks": [], "note": "not a town build"}
    import numpy as np
    from .blocktown import CASTLE, HOUSE, PAD, PLAZA, STREET

    use = info["use"]
    G = info["heights"]
    n, m = use.shape
    # Height to the eaves: walls, not the roof over them (sloped roof blocks
    # are left out), as street and square proportions are measured.
    top = np.full((n, m), -999, dtype=int)
    for (x, y, z), blk in b.blocks.items():
        if 0 <= x < n and 0 <= z < m and kind(blk) in STRUCT and y > top[x, z]:
            if kind(blk) in ("stairs", "slab") and lore.covering_of(name_of(blk)) is not None:
                continue
            top[x, z] = y
    built = (use == HOUSE) | (use == PAD) | (use == CASTLE)
    eave = np.where(built, top - G, 0)
    street = (use == STREET)

    def width_at(x: int, z: int) -> int:
        best = 99
        for dx, dz in ((1, 0), (0, 1)):
            k = 1
            a = 0
            while 0 <= x - dx * k < n and 0 <= z - dz * k < m and street[x - dx * k, z - dz * k]:
                a += 1
                k += 1
            k = 1
            c = 0
            while 0 <= x + dx * k < n and 0 <= z + dz * k < m and street[x + dx * k, z + dz * k]:
                c += 1
                k += 1
            best = min(best, a + c + 1)
        return best

    ratios = []
    edge = fronted = 0
    for x in range(1, n - 1):
        for z in range(1, m - 1):
            if not street[x, z]:
                continue
            if all(street[x + dx, z + dz] for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                continue
            edge += 1
            near = [eave[x + dx, z + dz] for dx in range(-2, 3) for dz in range(-2, 3)
                    if 0 <= x + dx < n and 0 <= z + dz < m and built[x + dx, z + dz] and eave[x + dx, z + dz] > 1]
            if near:
                fronted += 1
                ratios.append(width_at(x, z) / max(near))
    rule = lore.TOWN["street-enclosure"]
    lo, hi = rule.get(era, rule["range"])
    inside = sum(1 for r in ratios if lo <= r <= hi) / len(ratios) if ratios else 0.0
    checks = [_check("street-enclosure", round(inside, 3), inside >= 0.6, f"60% of fronted street edge in {lo}:1 to {hi}:1",
                     f"median {sorted(ratios)[len(ratios) // 2]:.2f}:1" if ratios else ""),
              _check("frontage", round(fronted / edge, 3) if edge else 0.0,
                     edge > 0 and fronted / edge >= lore.TOWN["frontage"]["min"], ">= 0.7 of street edge fronted")]
    sq = np.argwhere(use == PLAZA)
    if len(sq):
        w = int(sq[:, 0].max() - sq[:, 0].min() + 1)
        d = int(sq[:, 1].max() - sq[:, 1].min() + 1)
        short, long_ = min(w, d), max(w, d)
        ring = []
        for x, z in sq:
            for dx in range(-3, 4):
                for dz in range(-3, 4):
                    a, c = x + dx, z + dz
                    if 0 <= a < n and 0 <= c < m and built[a, c] and eave[a, c] > 1:
                        ring.append(int(eave[a, c]))
        hgt = sorted(ring)[len(ring) // 2] if ring else 0
        ratio = short / hgt if hgt else None
        lo, hi = lore.TOWN["square-enclosure"]["range"]
        checks.append(_check("square-enclosure", round(ratio, 2) if ratio else None,
                             ratio is not None and lo <= ratio <= hi, f"{lo}:1 to {hi}:1",
                             f"{short} by {long_}, surrounding eaves {hgt}"))
        checks.append(_check("square-size", [short, long_], short <= lore.TOWN["square-size"]["short_max"] and
                             long_ <= lore.TOWN["square-size"]["long_max"], "short <= 45, long <= 90"))
        rim = 0
        fronted_rim = 0
        for x, z in sq:
            if any(0 <= x + dx < n and 0 <= z + dz < m and use[x + dx, z + dz] != PLAZA
                   for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                rim += 1
                fronted_rim += any(0 <= x + dx < n and 0 <= z + dz < m and built[x + dx, z + dz]
                                   for dx in range(-2, 3) for dz in range(-2, 3))
        checks.append(_check("square-frontage", round(fronted_rim / rim, 3) if rim else 0.0,
                             rim > 0 and fronted_rim / rim >= 0.6, ">= 0.6 of the rim fronted"))
    passed = sum(c["pass"] for c in checks)
    return {"passed": passed, "total": len(checks), "score": round(passed / len(checks), 3), "checks": checks}
