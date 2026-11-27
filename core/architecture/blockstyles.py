"""Style sub-architects: a building tradition and a purpose, built by the
primary house architect inside the rules of ``lore``.

A sub-architect never invents structure. It takes its tradition's system,
roof covering, palette and window ratio from ``lore.TRADITIONS``, its plan
from ``lore.PURPOSES``, asks ``blockcraft.house`` for the body (rooms,
stairs, furniture, every floor walked), and then dresses the street front in
its tradition's own way: jetties and close studding for a Tudor town house,
pilaster strips and a stepped gable for a Hanseatic one.

Town houses stand gable to the street on narrow, deep plots. The house
architect lays its stair along the long side, so a sub-architect builds the
house long along x, opens a street door in the east gable, dresses that gable,
and turns the house so the gable faces north, onto z = 0 like every other
block-mode building. The house's own long-side door stays as the side door
onto the passage between plots.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import lore
from .blockcraft import EXTRA_STYLES, house, paste
from .blockmode import BlockBuild, kind, parse, state

# Palettes for the house architect. They are taken by name and never picked
# at random for an ordinary house.
EXTRA_STYLES.update({
    "tudor": {"base": (("bricks", 4), ("stone_bricks", 1)), "post": "dark_oak_log",
              "band": "stripped_dark_oak_log", "lower": (("calcite", 6), ("white_terracotta", 1)),
              "upper": (("calcite", 6), ("white_terracotta", 1)), "roof": "brick", "gable": "calcite",
              "door": "dark_oak", "floor": "spruce_planks", "panel": "spruce_planks",
              "shutter": "dark_oak_trapdoor", "chimney": (("bricks", 1),)},
    "hanseatic": {"base": (("stone_bricks", 3), ("andesite", 1)), "post": "stripped_spruce_log",
                  "band": "stripped_spruce_log", "lower": (("bricks", 1),), "upper": (("bricks", 1),),
                  "roof": "granite", "gable": "bricks", "door": "spruce", "floor": "oak_planks",
                  "panel": "oak_planks", "shutter": "spruce_trapdoor", "chimney": (("bricks", 1),)},
})

QUARTER_EAST_TO_NORTH = 3   # quarter-turns that bring the east gable round to face north


# ------------------------------------------------------ facade repair
_AXES = {"north": ("z", -1), "south": ("z", 1), "west": ("x", -1), "east": ("x", 1)}


def window_rhythm(b: BlockBuild, spacing: int = 3, skip_ground: bool = False,
                  sides: Optional[Tuple[str, ...]] = None) -> int:
    """Windows in a steady rhythm: on every storey of every wall that is not a
    party wall, a window two blocks high in every ``spacing``-th column, where
    the wall is a plain block with open room behind it, stacked storey over
    storey. Returns how many it made."""
    from . import blockrules as R
    party = set(getattr(b, "party_walls", ()) or ())
    S = b.sizes.storey
    solid = {p for p, blk in b.blocks.items() if kind(blk) in R.STRUCT}
    if not solid:
        return 0
    (x0, y0, z0) = (min(p[0] for p in solid), min(p[1] for p in solid), min(p[2] for p in solid))
    (x1, y1, z1) = (max(p[0] for p in solid), max(p[1] for p in solid), max(p[2] for p in solid))
    eave = R.facades(b).get("eave", y1)
    made = 0
    for side, (axis, sgn) in _AXES.items():
        if side in party or (sides is not None and side not in sides):
            continue
        us = range(x0 + 1, x1) if axis == "z" else range(z0 + 1, z1)
        march = list(range(z0, z1 + 1) if axis == "z" else range(x0, x1 + 1))
        if sgn > 0:
            march.reverse()
        for u in us:
            if (u - us.start) % spacing != spacing // 2:
                continue
            for L in range(0 if not skip_ground else 1, max(1, (eave - y0) // S)):
                rows = (y0 + L * S + 2, y0 + L * S + 3)
                if rows[1] >= eave:
                    continue

                def at(wv: int, y: int):
                    return (u, y, wv) if axis == "z" else (wv, y, u)

                fw = next((wv for wv in march if at(wv, rows[0]) in solid), None)
                if fw is None or at(fw, rows[1]) not in solid:
                    continue
                # Not beside a door or another opening already there.
                if any(kind(b.blocks.get(at(fw, y)) or "air") != "full" for y in rows):
                    continue
                if any(kind(b.blocks.get(at(fw - sgn, y)) or "air") in R.STRUCT for y in rows):
                    continue
                nb = [(u + d, rows[0], fw) if axis == "z" else (fw, rows[0], u + d) for d in (-1, 1)]
                if any(kind(b.blocks.get(p) or "air") in ("door", "pane") for p in nb):
                    continue
                for y in rows:
                    b.blocks[at(fw, y)] = state("glass_pane")
                made += 1
    return made


def relieve_flat_walls(b: BlockBuild, accent: str = "stripped_spruce_log", limit: int = 12,
                       rounds: int = 60) -> int:
    """Break every blank panel the checker finds on a facade, largest first:
    a window, two blocks high, in the middle of the panel where a room lies
    behind it; where the wall is backed by something solid (a stair, a
    chimney, rock), a post of ``accent`` up the panel instead. Party walls
    are left alone. Returns how many openings and posts it made."""
    from . import blockrules as R
    party = set(getattr(b, "party_walls", ()) or ())
    made = 0
    tried: set = set()
    for _ in range(rounds):
        sides = R.facades(b).get("sides", {})
        worst = [(s["flat_run"], name) for name, s in sides.items()
                 if name not in party and s["flat_run"] > limit and s.get("flat_at")]
        worst = [w for w in sorted(worst, reverse=True)
                 if (w[1], tuple(sides[w[1]]["flat_at"]["cell"])) not in tried]
        if not worst:
            break
        _, side = worst[0]
        fa = sides[side]["flat_at"]
        tried.add((side, tuple(fa["cell"])))
        (u0, ytop), (w, h) = fa["cell"], fa["size"]
        ylo = ytop - h + 1
        axis, sgn = _AXES[side]
        solid = [p for p, blk in b.blocks.items() if kind(blk) in R.STRUCT]
        lo = min(p[2 if axis == "z" else 0] for p in solid)
        hi = max(p[2 if axis == "z" else 0] for p in solid)
        march = list(range(lo, hi + 1) if sgn < 0 else range(hi, lo - 1, -1))

        def at(u: int, wv: int, y: int):
            return (u, y, wv) if axis == "z" else (wv, y, u)

        def face_of(u: int, y: int):
            for wv in march:
                blk = b.blocks.get(at(u, wv, y))
                if blk is not None and kind(blk) in R.STRUCT:
                    return wv
            return None

        # Window first: the column nearest the middle, the rows nearest the
        # middle, where the wall is a plain block with open room behind it.
        done = False
        cols = sorted(range(u0, u0 + w), key=lambda u: abs(u - (u0 + w // 2)))
        S = b.sizes.storey
        # Window rows sit inside a storey, never across a floor line.
        pairs = sorted(((y, y + 1) for y in range(ylo, ytop) if y % S in (1, 2) and (y + 1) % S != 0),
                       key=lambda r: (r[0] % S != 2, abs(r[0] - (ylo + ytop) / 2)))
        for u in cols:
            for rows in pairs:
                fw = face_of(u, rows[0])
                if fw is None or face_of(u, rows[1]) != fw:
                    continue
                if all(kind(b.blocks.get(at(u, fw, y)) or "air") == "full" and
                       kind(b.blocks.get(at(u, fw - sgn, y)) or "air") not in R.STRUCT for y in rows):
                    for y in rows:
                        b.blocks[at(u, fw, y)] = state("glass_pane")
                    done = True
                    break
            if done:
                break
        if not done:
            # No room behind: a post of another material up the panel.
            post = accent if R.family(state(accent)) != fa["family"] else "dark_oak_log"
            uc = u0 + w // 2
            for y in range(ylo, ytop + 1):
                fw = face_of(uc, y)
                if fw is not None and kind(b.blocks.get(at(uc, fw, y)) or "air") == "full":
                    b.blocks[at(uc, fw, y)] = state(post, axis="y")
        made += 1
    return made


class _Front:
    """The street gable of a long house before it is turned: the wall plane
    is x = X, the frontage runs z = 0 .. Z, storeys are S blocks."""

    def __init__(self, b: BlockBuild, storeys: int):
        self.b = b
        walls = [p for p, blk in b.blocks.items() if p[1] == 1 and kind(blk) == "full"]
        self.X = max(p[0] for p in walls if p[0] >= 0)
        self.Z = max(p[2] for p in walls if p[2] >= 0)
        self.S = b.sizes.storey
        self.storeys = storeys
        self.top = storeys * self.S
        self.c = self.Z // 2

    def roof_at(self, z: int) -> int:
        """The roof's outer surface layer over row z of the gable (45 degrees,
        one block of overhang)."""
        return self.top + min(z + 1, self.Z + 1 - z)


def _strip_front(f: _Front, wall: Callable[[int, int], str]) -> None:
    """Clear the house architect's windows and shutters out of the street
    gable, back to plain wall, before it is dressed."""
    b = f.b
    for y in range(1, f.top):
        for z in range(0, f.Z + 1):
            blk = b.get(f.X, y, z)
            if blk is not None and (kind(blk) in ("pane", "trapdoor") or "glass" in blk):
                b.set(f.X, y, z, wall(y, z))
        for z in range(-1, f.Z + 2):
            for x in range(f.X + 1, f.X + 3):
                blk = b.get(x, y, z)
                if blk is not None and kind(blk) in ("trapdoor", "pot", "slab", "fence", "lantern", "torch"):
                    b.set(x, y, z, None)


def _extend_roof(f: _Front, k: int) -> None:
    """Carry the roof k blocks further over the street, so it covers jetties
    below it: the gable moves out to x = X + k and the verge goes with it."""
    if k <= 0:
        return
    b = f.b
    X = f.X
    gable = {(y, z): blk for (x, y, z), blk in list(b.blocks.items()) if x == X and y >= f.top}
    verge = {(y, z): blk for (x, y, z), blk in list(b.blocks.items()) if x == X + 1 and y >= f.top - 1}
    for (y, z) in verge:
        b.set(X + 1, y, z, None)
    roof_kinds = ("stairs", "slab")
    for i in range(1, k + 1):
        for (y, z), blk in gable.items():
            if kind(blk) in roof_kinds or i == k:
                b.set(X + i, y, z, blk)
    # The old gable stays as a wall inside the roof; open it so the attic runs through.
    for (y, z), blk in gable.items():
        if kind(blk) not in ("stairs", "slab") and 0 < z < f.Z:
            b.set(X, y, z, None)
    for (y, z), blk in verge.items():
        b.set(X + k + 1, y, z, blk)


def _street_door(f: _Front, z: int, wood: str, name: str, x: Optional[int] = None) -> Dict[str, Any]:
    """A door in the street gable at row z, with the way in kept clear."""
    b = f.b
    x = f.X if x is None else x
    for xx in range(f.X + 1, x + 1):
        for y in (1, 2):
            b.set(xx, y, z, None)
    port = b.door(x, 1, z, "west", wood=wood, along="south", name=name)
    # Clear the threshold inside of anything the furnisher put there, and the step outside.
    for xx in (f.X - 1, f.X - 2):
        for y in (1, 2, 3):
            blk = b.get(xx, y, z)
            if blk is not None and kind(blk) not in ("stairs",) and y <= 2:
                b.set(xx, y, z, None)
    for xx in range(f.X, x):
        b.set(xx, 0, z, b.get(f.X - 1, 0, z) or state("spruce_planks"))
    b.set(x + 1, 0, z, state("stone_brick_slab", type="top"))
    for y in (1, 2):
        b.set(x + 1, y, z, None)
    return port


def _widen_windows(b: BlockBuild, X: int, Z: int, top: int, S: int) -> None:
    """Town-house windows run wide: each window on the long sides takes the
    panel beside it too, and its shutter moves out of the way; on the back
    gable the upper windows widen the same way, clear of the chimney."""
    for (fixed, out) in ((0, -1), (Z, Z + 1)):
        for x in range(2, X - 1, 3):
            for y in range(1, top):
                if b.get(x, y, fixed) is not None and "glass" in b.get(x, y, fixed):
                    b.set(x - 1, y, fixed, state("glass_pane"))
                    for xs in (x - 1, x + 1):
                        if b.get(xs, y, out) is not None and kind(b.get(xs, y, out)) == "trapdoor":
                            b.set(xs, y, out, None)
    cz = Z // 2
    for y in range(S + 1, top):
        for z in range(1, Z):
            if b.get(0, y, z) is not None and "glass" in b.get(0, y, z):
                for zz in (z - 1, z + 1):
                    if 0 < zz < Z and zz != cz and b.get(0, y, zz) is not None and kind(b.get(0, y, zz)) == "full":
                        b.set(0, y, zz, state("glass_pane"))
                        break


def _turn(b: BlockBuild) -> BlockBuild:
    """Turn the house so its street gable faces north, onto z = 0."""
    tmp = BlockBuild(b.product, b.player, b.seed)
    paste(tmp, b, 0, 0, QUARTER_EAST_TO_NORTH)
    (lx, _, lz), _hi = tmp.bounds()
    out = BlockBuild(b.product, b.player, b.seed)
    paste(out, tmp, -lx, -lz)
    out.notes = list(b.notes)
    for key in ("tradition", "purpose", "kind", "style_notes"):
        if hasattr(b, key):
            setattr(out, key, getattr(b, key))
    out.ports = [p for p in out.ports if "street" in p["id"]] + [p for p in out.ports if "street" not in p["id"]]
    # The back, onto the yard, gets its windows in a rhythm too.
    window_rhythm(out, spacing=3, sides=("south",))
    return out


# ------------------------------------------------------------------ Tudor
def tudor_merchant_house(player: Any = "blocky", seed: int = 0, name: str = "tudor-merchant") -> BlockBuild:
    """A Tudor merchant's house, gable to the street: a shop on the ground
    floor behind a wide window and a drop-down shutter counter; each storey
    above jettied one block over the street on its joists, with a moulded
    bressumer and brackets at the posts; close studding in dark oak with
    limewashed infill; a mullioned window across the middle of each upper
    storey; the gable jettied over the top storey and studded too; a clay
    tile roof and a brick chimney at the back."""
    rng = random.Random(seed * 7919 + 17)
    t = lore.TRADITIONS["tudor"]
    storeys = 3                            # three storeys and a garret: a square 15 wide stays open
    frontage = rng.choice((7, 8))
    depth = rng.randint(10, 12)
    b = house(player, seed, width=depth, depth=frontage, storeys=storeys, style="tudor", name=name)
    f = _Front(b, storeys)
    X, Z, S, top = f.X, f.Z, f.S, f.top
    frame, infill = "dark_oak_log", "calcite"
    stud = lambda z: state(frame, axis="y") if z % 2 == 0 or z in (0, Z) else state(infill)  # noqa: E731
    _strip_front(f, lambda y, z: stud(z))
    J = storeys - 1
    _widen_windows(b, X, Z, top, S)
    # Ground storey: the shop front on the gable, brick plinth below it.
    for z in range(0, Z + 1):
        b.set(X, 0, z, state("bricks"))
        for y in range(1, S):
            b.set(X, y, z, state(frame, axis="y") if z in (0, Z) else stud(z))
        b.set(X, S - 1, z, state("stripped_dark_oak_log", axis="z"))
    door_z = 1 if rng.random() < 0.5 else Z - 1
    shop = [z for z in range(1, Z) if abs(z - door_z) >= 2]
    for z in shop:
        for y in (2, 3):
            b.set(X, y, z, state("glass_pane"))
        b.set(X, 1, z, state("stripped_dark_oak_log", axis="z"))
        # The shop board, let down by day as a counter.
        b.set(X + 1, 1, z, state("spruce_trapdoor", facing="east", half="top", open="false"))
    port = _street_door(f, door_z, "dark_oak", name + "-street")
    port["id"] = port["id"].replace("port.", "port.street.")
    # Jettied storeys.
    for L in range(1, storeys):
        xL = X + L
        y0, y1 = L * S, L * S + S - 1
        for z in range(0, Z + 1):
            # The bressumer along the jetty front and the floor behind it.
            b.set(xL, y0, z, state("stripped_dark_oak_log", axis="z"))
            for xx in range(X + 1, xL):
                b.set(xx, y0, z, state("spruce_planks"))
            for y in range(y0 + 1, y1 + 1):
                for xx in range(X + 1, xL):
                    # The jetty's ends show on the flanks: limewash there, boards inside.
                    b.set(xx, y, z, state(infill) if z in (0, Z) else state("spruce_planks"))
                b.set(xL, y, z, state(frame, axis="y") if z in (0, Z) else stud(z))
            b.set(xL, y1, z, state("stripped_dark_oak_log", axis="z"))
        # Brackets under the jetty at the posts.
        for z in (0, f.c, Z):
            b.set(xL, y0 - 1, z, state("dark_oak_stairs", facing="west", half="top"))
        # A mullioned window across the middle, through the jetty's depth.
        for z in range(f.c - 1, f.c + 2):
            for y in (y0 + 2, y0 + 3):
                for xx in range(X, xL):
                    b.set(xx, y, z, None)
                b.set(xL, y, z, state("dark_oak_fence") if z == f.c else state("glass_pane"))
            b.set(xL + 1, y0 + 1, z, state("dark_oak_slab", type="top"))
        # The room behind gets the deeper floor too.
        for xx in range(X, xL):
            b.set(xx, y0, f.c, state("spruce_planks"))
    _extend_roof(f, J)
    # The gable over the top jetty: close studded, a small window, a finial.
    xg = X + J
    for (x, y, z), blk in list(b.blocks.items()):
        if x == xg and y >= top and kind(blk) == "full":
            b.set(x, y, z, stud(z))
    gy = top + 1
    for z in (f.c,):
        for y in (gy, gy + 1):
            b.set(xg, y, z, state("glass_pane"))
    b.set(xg, top, f.c - 1, state("stripped_dark_oak_log", axis="z"))
    b.set(xg, top, f.c + 1, state("stripped_dark_oak_log", axis="z"))
    b.set(xg + 1, top - 1, 0, state("dark_oak_stairs", facing="west", half="top"))
    b.set(xg + 1, top - 1, Z, state("dark_oak_stairs", facing="west", half="top"))
    # A hanging lantern under the first jetty, by the door.
    b.set(X + 1, S - 1, door_z + (1 if door_z < f.c else -1), state("lantern", hanging="true"))
    b.tradition, b.purpose, b.kind = "tudor", "merchant-house", "house"  # type: ignore[attr-defined]
    b.notes.append(f"{name}: Tudor merchant house, {frontage} wide, {depth} deep, {storeys} storeys, "
                   f"{J} jetties to the street")
    return _turn(b)


# -------------------------------------------------------------- Hanseatic
def hanseatic_gable_house(player: Any = "blocky", seed: int = 0, name: str = "hanseatic-house") -> BlockBuild:
    """A Hanseatic merchant's house, brick, gable to the street: pilaster
    strips rise full height between the window bays and stand proud of the
    wall; a stone string course at each floor; tall windows set back between
    the strips; a portal with a stone surround; the gable stepped above the
    roof, each step capped in stone, its panels between the strips
    whitewashed as blind niches; a hoist beam and loading door near the apex;
    a red pantile roof."""
    rng = random.Random(seed * 7919 + 29)
    storeys = 3
    frontage = rng.choice((9, 9, 7))       # three-block bays and room for three steps a side
    depth = rng.randint(10, 12)
    b = house(player, seed, width=depth, depth=frontage, storeys=storeys, style="hanseatic", name=name)
    f = _Front(b, storeys)
    X, Z, S, top, c = f.X, f.Z, f.S, f.top, f.c
    brick = state("bricks")
    _strip_front(f, lambda y, z: brick)
    _widen_windows(b, X, Z, top, S)
    # Strips at the corners and between bays; bays are the gaps between them.
    # Bays two or three blocks wide: strips at the corners and one or two between.
    strips = [0, Z // 2, Z] if Z <= 8 else [0, Z // 3, Z - Z // 3, Z]
    bays = [z for z in range(1, Z) if z not in strips]
    # Remove the verge: a stepped gable is a parapet, with no roof over it.
    for (x, y, z), blk in list(b.blocks.items()):
        if x >= X + 1 and y >= top - 1:
            b.set(x, y, z, None)
    # The gable wall, stepped: two rows per step, a block above the roof.
    def step_top(z: int) -> int:
        d = min(z, Z - z)
        return top + 2 * (d // 2) + 3

    for z in range(0, Z + 1):
        for y in range(top, step_top(z) + 1):
            b.set(X, y, z, brick)
        b.set(X, step_top(z) + 1, z, state("stone_brick_slab", type="bottom"))
    # Whitewashed blind niches in the gable between the strips, pointed with stairs.
    for z in bays:
        for y in range(top + 1, step_top(z) - 1):
            b.set(X, y, z, state("white_terracotta"))
    # Ground storey wall and plinth.
    for z in range(0, Z + 1):
        b.set(X, 0, z, state("stone_bricks"))
        for y in range(1, top):
            if b.get(X, y, z) is None or kind(b.get(X, y, z)) == "full":
                b.set(X, y, z, brick)
    door_z = bays[0] if Z <= 8 else c
    # Windows: tall, set back between the strips, in every bay row of each storey.
    for L in range(storeys):
        for z in bays:
            if L == 0 and abs(z - door_z) <= 0:
                continue
            for y in range(L * S + 1 + (1 if L == 0 else 0), L * S + S - 1):
                b.set(X, y, z, state("glass_pane"))
    # Side lights near the street end of each flank: a merchant's front room
    # (the shop below, the best room above) took light from the alley too.
    for L in range(storeys):
        for zf in (0, Z):
            for y in ((2, 3) if L == 0 else (L * S + 1, L * S + 2)):
                if kind(b.get(X - 2, y, zf) or "air") == "full":
                    b.set(X - 2, y, zf, state("glass_pane"))
    # Pilaster strips proud of the wall, up to the steps, with stone caps.
    for z in strips:
        hz = step_top(z)
        for y in range(1, hz + 1):
            b.set(X + 1, y, z, brick)
        b.set(X + 1, hz + 1, z, state("stone_brick_slab", type="bottom"))
    # String courses at each floor line, across the bays.
    for L in range(1, storeys + 1):
        for z in bays:
            b.set(X + 1, L * S, z, state("stone_brick_slab", type="top"))
    # The portal: door in the middle bay under a stone surround.
    port = _street_door(f, door_z, "spruce", name + "-street")
    port["id"] = port["id"].replace("port.", "port.street.")
    for y in (1, 2):
        for dz in (-1, 1):
            if 0 <= door_z + dz <= Z and (door_z + dz) not in strips:
                b.set(X, y, door_z + dz, state("stone_bricks"))
    b.set(X, 3, door_z, state("chiseled_stone_bricks"))
    b.set(X + 1, 3, door_z, state("stone_brick_stairs", facing="west", half="top"))
    # Loading door and hoist beam near the apex.
    ly = step_top(c) - 3
    if ly > top:
        b.set(X, ly, c, state("spruce_trapdoor", facing="east", half="bottom", open="false"))
        b.set(X, ly + 1, c, state("spruce_trapdoor", facing="east", half="bottom", open="false"))
        for x in (X + 1, X + 2):
            b.set(x, ly + 2, c, state("stripped_spruce_log", axis="x"))
        b.set(X + 2, ly + 1, c, state("chain", axis="y"))
    b.tradition, b.purpose, b.kind = "hanseatic", "merchant-house", "house"  # type: ignore[attr-defined]
    b.notes.append(f"{name}: Hanseatic gable house, {frontage} wide, {depth} deep, {storeys} storeys, "
                   f"stepped gable, {len(strips)} pilaster strips")
    return _turn(b)


# ------------------------------------------------------------------ Norse
def norse_longhouse(player: Any = "blocky", seed: int = 0, name: str = "norse-longhouse") -> BlockBuild:
    """A Norse longhouse, long side to the street: bow-sided walls of
    horizontal spruce logs on a fieldstone footing, the long walls bellying
    out a block in the middle as Trelleborg and Fyrkat houses did; posts
    outside the walls carry the eaves; a turf roof at about 34 degrees that
    comes down onto the walls and follows the bow, so the ridge dips over
    the end bays; crossed boards at both gable ends; a door in each long
    wall at the same bay (the cross passage); inside, two rows of posts
    carry the roof over a long hearth under a smoke hole, with sleeping
    benches along the walls. No chimney and no glass."""
    rng = random.Random(seed * 7919 + 41)
    b = BlockBuild(name, player, seed)
    L = rng.randint(17, 23)                     # x = 0 .. L-1
    end = 3 + (L > 20)                          # the narrower end bays
    W = 8                                       # walls z = 0 .. 8 in the middle bays
    wall_h = 4                                  # walls y = 1 .. 4, the turf comes down onto them
    log_x, log_z = state("spruce_log", axis="x"), state("spruce_log", axis="z")
    stone = b.mix(("stone", 4), ("andesite", 2), ("cobblestone", 1))
    post = state("spruce_log", axis="y")
    mid = W // 2

    def zn(x: int) -> int:                      # the north wall's line at x
        return 1 if x < end or x > L - 1 - end else 0

    def zs(x: int) -> int:
        return W - zn(x)

    # Footing, earth floor and walls.
    for x in range(L):
        for z in range(zn(x), zs(x) + 1):
            wall = x in (0, L - 1) or z in (zn(x), zs(x))
            b.set(x, 0, z, stone() if wall else state("coarse_dirt"))
            if not wall:
                continue
            for y in range(1, wall_h + 1):
                b.set(x, y, z, log_z if x in (0, L - 1) else log_x)
        # Where the wall steps out at the end of the bow, close the corner.
        if 0 < x < L - 1 and zn(x) != zn(x - 1):
            for z in (0, 1, W - 1, W):
                b.set(x, 0, z, stone())
                for y in range(1, wall_h + 1):
                    b.set(x, y, z, log_x)
    # Posts outside the long walls carry the eaves, every fourth bay.
    for x in range(2, L - 1, 4):
        for z in (zn(x) - 1, zs(x) + 1):
            for y in range(0, wall_h):
                b.set(x, y, z, post if y else stone())
    # Gable posts proud of the end walls.
    for x in (-1, L):
        for z in (mid - 1, mid + 1):
            for y in range(0, wall_h):
                b.set(x, y, z, post if y else stone())

    # Turf roof, two blocks of rise in three (a slab, then two sods), from
    # eaves a block outside the walls to the ridge. It follows the bow, so
    # the ridge dips over the narrower end bays as on the old houses.
    y0 = wall_h
    sod = state("grass_block")
    top: Dict[Tuple[int, int], int] = {}
    for x in range(-1, L + 1):
        xx = min(max(x, 0), L - 1)
        zlo, zhi = zn(xx) - 1, zs(xx) + 1
        for z in range(zlo, zhi + 1):
            d = min(z - zlo, zhi - z)
            h = 1 + d + (d + 1) // 3              # height in half blocks
            if h % 2:
                y = y0 + (h - 1) // 2
                b.set(x, y, z, state("mossy_cobblestone_slab", type="bottom"))
                if rng.random() < 0.3:
                    b.set(x, y + 1, z, state("moss_carpet"))
            else:
                y = y0 + h // 2 - 1
                b.set(x, y, z, sod)
            if d >= 2:
                b.set(x, y - 1, z, state("spruce_planks"))
            top[(x, z)] = y
    # Gable ends under the roof, boarded.
    for x in (0, L - 1):
        for z in range(zn(x), zs(x) + 1):
            for y in range(wall_h + 1, top[(x, z)]):
                if b.get(x, y, z) is None:
                    b.set(x, y, z, state("spruce_planks"))
    # Crossed gable boards at both ends of the ridge.
    for x in (-1, L):
        for z in (mid - 1, mid + 1):
            b.set(x, top[(x, z)] + 1, z, state("spruce_fence"))
        b.set(x, top[(x, mid)] + 1, mid, state("spruce_fence"))
        b.set(x, top[(x, mid)] + 2, mid, state("spruce_fence"))

    # The cross passage: a door in each long wall, a bay off the middle.
    dx = L // 2 - 2
    port = b.door(dx, 1, zn(dx), "south", wood="spruce", name=name + "-street")
    port["id"] = port["id"].replace("port.", "port.street.")
    b.door(dx, 1, zs(dx), "north", wood="spruce", name=name + "-yard")
    # The long hearth under the smoke hole.
    c = L // 2 + 1
    for x in range(c - 1, c + 2):
        b.set(x, 0, mid, state("campfire", lit="false") if x == c else state("cobblestone"))
    for y in range(top[(c, mid)] - 1, top[(c, mid)] + 1):
        b.set(c, y, mid, None)
    # Two rows of posts inside carry the roof (a three-aisled hall), clear
    # of the cross passage and the hearth; benches along the walls.
    for x in range(2, L - 2, 4):
        if abs(x - dx) <= 1 or abs(x - c) <= 1:
            continue
        for z in (2, W - 2):
            for y in range(1, top[(x, z)]):
                if b.get(x, y, z) is None:
                    b.set(x, y, z, post)
    for x in range(1, L - 1):
        if abs(x - dx) <= 1:
            continue
        for z in (zn(x) + 1, zs(x) - 1):
            if zn(x) == 0 and b.get(x, 1, z) is None:
                b.set(x, 1, z, state("spruce_slab", type="bottom"))
    b.tradition, b.purpose, b.kind = "norse", "longhouse", "house"  # type: ignore[attr-defined]
    b.notes.append(f"{name}: Norse longhouse, {L} long, bow-sided {W + 1} to {W - 1} wide, turf roof, "
                   "cross passage, long hearth under a smoke hole")
    return b


# ----------------------------------------------------------------- Alpine
def alpine_chalet(player: Any = "blocky", seed: int = 0, name: str = "alpine-chalet") -> BlockBuild:
    """An Alpine chalet, its low wide gable to the street (the valley): a
    rubble-stone ground storey with dressed quoins on a stone-brick plinth,
    a log storey above; a shingle roof at about 27 degrees with eaves
    reaching two blocks out on every side; a balcony on carved brackets
    across the gable and down both long sides of the upper storey; windows
    in stacked bays; a stone chimney up the back wall and through the eaves;
    a stair inside up to the log storey."""
    rng = random.Random(seed * 7919 + 53)
    b = BlockBuild(name, player, seed)
    X = rng.choice((8, 8, 10))                  # gable front x = 0 .. X
    Z = rng.choice((8, 9))                      # depth z = 0 .. Z
    S = b.sizes.storey
    rubble = b.mix(("cobblestone", 4), ("stone", 2), ("mossy_cobblestone", 1))
    quoin = state("stone_bricks")
    log_x, log_z = state("spruce_log", axis="x"), state("spruce_log", axis="z")
    top_wall = 2 * S - 1                        # walls y = 1 .. 9; eaves at 2S
    # Plinth and floors.
    b.fill(0, 0, 0, X, 0, Z, state("polished_andesite"))
    b.fill(1, 0, 1, X - 1, 0, Z - 1, state("spruce_planks"))
    b.fill(1, S, 1, X - 1, S, Z - 1, state("spruce_planks"))
    for y in range(1, top_wall + 1):
        for x in range(X + 1):
            for z in range(Z + 1):
                if 0 < x < X and 0 < z < Z:
                    continue
                corner = x in (0, X) and z in (0, Z)
                if y < S:
                    blk = quoin if corner and y % 2 else rubble()
                elif y == S:
                    blk = state("stripped_spruce_log", axis="x" if z in (0, Z) else "z")
                else:
                    blk = state("spruce_log", axis="y") if corner else (log_x if z in (0, Z) else log_z)
                b.set(x, y, z, blk)
    # Windows in stacked bays, the same columns on both storeys; shutters on
    # the log storey.
    cx = X // 2
    front_cols = [c for c in range(2, X - 1, 2)]
    side_cols = [c for c in range(2, Z - 1, 2)]
    for L in range(2):
        ys = (2, 3) if L == 0 else (S + 2, S + 3)
        for x in front_cols:
            for z, out in ((0, -1), (Z, Z + 1)):
                if L == 0 and z == 0 and x == cx:
                    continue
                for y in ys:
                    b.set(x, y, z, state("glass_pane"))
        for z in side_cols:
            for x in (0, X):
                for y in ys:
                    b.set(x, y, z, state("glass_pane"))
    # Upper storey: a balcony across the gable and down both long sides, on
    # brackets, with a fence rail.
    for x in range(-1, X + 2):
        for z in range(-1, Z + 2):
            if 0 <= x <= X and 0 <= z <= Z:
                continue
            if z == Z + 1 and 0 <= x <= X:
                continue                        # no balcony across the back
            b.set(x, S, z, state("spruce_slab", type="top"))
            edge = x in (-1, X + 1) or z == -1
            if edge:
                b.set(x, S + 1, z, state("spruce_fence"))
            if (x + z) % 3 == 0 and (0 <= x <= X or 0 <= z <= Z):
                facing = "south" if z == -1 else ("east" if x == -1 else "west")
                b.set(x, S - 1, z, state("spruce_stairs", facing=facing, half="top"))
    # Posts at the balcony's corners carry the corners of the eaves.
    for x in (-1, X + 1):
        for z in (-1, Z + 1):
            for y in range(S + 1, 2 * S - 1):
                b.set(x, y, z, state("stripped_spruce_log", axis="y"))
    # The door to the street, under the balcony, and the step.
    port = b.door(cx, 1, 0, "south", wood="spruce", name=name + "-street")
    port["id"] = port["id"].replace("port.", "port.street.")
    # A door onto the balcony above it.
    b.door(cx, S + 1, 0, "north", wood="spruce", name=name + "-balcony")
    # Shingle roof, ridge along z, half-pitch, eaves two blocks out.
    y0 = 2 * S - 1
    top: Dict[Tuple[int, int], int] = {}
    xlo, xhi = -2, X + 2
    for x in range(xlo, xhi + 1):
        d = min(x - xlo, xhi - x)
        y = y0 + d // 2
        up = "east" if x - xlo <= xhi - x else "west"
        for z in range(-2, Z + 3):
            if d % 2 == 0:
                b.set(x, y, z, state("spruce_slab", type="bottom"))
            else:
                b.set(x, y, z, state("spruce_stairs", facing=up, half="bottom"))
            if d >= 2 and 0 <= z <= Z and 0 <= x <= X:
                b.set(x, y - 1, z, state("spruce_planks"))
            top[(x, z)] = y
    ridge = max(top.values())
    # Gables between the eaves and the roof: boarded, with a small window.
    for z in (0, Z):
        for x in range(0, X + 1):
            for y in range(2 * S, top[(x, z)]):
                if b.get(x, y, z) is None:
                    b.set(x, y, z, state("spruce_planks"))
        if top[(cx, z)] - 2 * S >= 2:
            b.set(cx, 2 * S, z, state("glass_pane"))
    # Chimney through the back slope, clear of the ridge.
    chx, chz = 1, Z + 1
    for y in range(0, ridge + 2):
        b.set(chx, y, chz, state("stone" if y % 3 else "andesite"))
    b.set(chx, ridge + 2, chz, state("stone_slab", type="bottom"))
    # A stair inside along the back wall to the log storey.
    for i in range(S - 1):
        b.set(2 + i, 1 + i, Z - 1, state("spruce_stairs", facing="east"))
    for x in range(2, 2 + S):
        for y in range(S, S + 2):
            if x < 2 + S - 1:
                b.set(x, S, Z - 1, None)
    b.tradition, b.purpose, b.kind = "alpine", "cottage", "house"  # type: ignore[attr-defined]
    b.notes.append(f"{name}: Alpine chalet, {X + 1} wide gable, {Z + 1} deep, stone and log storeys, "
                   "shingle roof with deep eaves, balconies")
    return b


# ---------------------------------------------------------- Mediterranean
def mediterranean_courtyard_house(player: Any = "blocky", seed: int = 0,
                                  name: str = "mediterranean-courtyard") -> BlockBuild:
    """A Mediterranean courtyard house: four two-storey wings round an open
    court, turned in on themselves. Lime-washed or sandstone walls over a
    stone plinth, with quoins at the corners and a string course at the
    floor line; small windows in stacked bays with painted shutters; one
    entrance, a passage through the street wing; low roofs of terracotta at
    about 27 degrees, sloping both ways over each wing; in the court, an
    outside stair up to the upper floor of the street wing, a cistern and
    pots."""
    rng = random.Random(seed * 7919 + 67)
    b = BlockBuild(name, player, seed)
    N = rng.choice((12, 12, 14))                # outer walls 0 .. N
    S = b.sizes.storey
    wing = 3                                    # inner walls at 3 and N - 3
    render = rng.choice(("calcite", "calcite", "smooth_sandstone"))
    stone = state("stone_bricks")
    shutter = rng.choice(("spruce", "dark_oak", "warped"))
    top_wall = 2 * S - 1
    lo, hi = wing, N - wing                     # the inner walls
    # Plinth, floors and walls.
    for x in range(N + 1):
        for z in range(N + 1):
            outer = x in (0, N) or z in (0, N)
            inner = (x in (lo, hi) and lo <= z <= hi) or (z in (lo, hi) and lo <= x <= hi)
            in_court = lo < x < hi and lo < z < hi
            b.set(x, 0, z, state("stone") if in_court else stone if (outer or inner) else state("spruce_planks"))
            if in_court:
                continue
            if not (outer or inner):
                b.set(x, S, z, state("spruce_planks"))
                continue
            for y in range(1, top_wall + 1):
                b.set(x, y, z, state(render))
    # Plinth proud of the walls, pilasters at the bays, a string course.
    for u in range(-1, N + 2):
        for (x, z) in ((u, -1), (u, N + 1), (-1, u), (N + 1, u)):
            if not (-1 <= x <= N + 1 and -1 <= z <= N + 1):
                continue
            b.set(x, 0, z, stone)
            corner = (x in (-1, N + 1)) and (z in (-1, N + 1))
            along = z if x in (-1, N + 1) else x
            if corner:                  # quoins at the corners only
                for y in range(1, top_wall):
                    b.set(x, y, z, state("stone_bricks" if y % 2 else "polished_andesite"))
            else:
                b.set(x, S, z, state("stone_brick_slab", type="top"))
    # Small windows in stacked bays, shutters on the outer face.
    nb = (N - 4) // 4 + 1                       # bays, centred on the wall
    cols = [(N - 4 * (nb - 1)) // 2 + 4 * i for i in range(nb)]
    for L in range(2):
        ys = (2, 3) if L == 0 else (S + 2, S + 3)
        for c in cols:
            for (x, z, sx, sz, f) in ((c, 0, 1, 0, "north"), (c, N, 1, 0, "south"),
                                      (0, c, 0, 1, "west"), (N, c, 0, 1, "east")):
                ox, oz = {"north": (0, -1), "south": (0, 1), "west": (-1, 0), "east": (1, 0)}[f]
                for y in ys:
                    b.set(x, y, z, state("glass_pane"))
                    for k in (-1, 1):       # painted shutters, open against the wall
                        b.set(x + ox + k * sx, y, z + oz + k * sz,
                              state(shutter + "_trapdoor", facing=f, open="true", half="bottom"))
    # The entrance: a passage through the street wing to the court.
    door_x = cols[len(cols) // 2]
    port = b.door(door_x, 1, 0, "south", wood="spruce", name=name + "-street")
    port["id"] = port["id"].replace("port.", "port.street.")
    for z in range(1, lo + 1):
        for y in (1, 2):
            b.set(door_x, y, z, None)
        b.set(door_x, 0, z, stone)
    b.set(door_x, 3, 0, state("chiseled_stone_bricks"))
    # Doors from the court into the other wings.
    for (x, z, f) in ((N // 2, hi, "south"), (lo, N // 2, "west"), (hi, N // 2, "east")):
        b.door(x, 1, z, f, wood="spruce", name=name + "-room")
    # Outside stair up the court's east side to the street wing's upper floor.
    sx = hi - 1
    steps = S - 1
    for i in range(steps):
        z = lo + 1 + steps - i
        for y in range(1, i + 1):
            b.set(sx, y, z, stone)
        b.set(sx, i + 1, z, state("stone_brick_stairs", facing="north"))
    for y in range(1, S + 1):
        b.set(sx, y, lo + 1, stone)
    b.door(sx, S + 1, lo, "north", wood="spruce", name=name + "-upper")
    # Low terracotta roofs, both ways over each wing, overhanging a block.
    y0 = top_wall
    tiles = b.mix(("terracotta", 1),)
    for x in range(-1, N + 2):
        for z in range(-1, N + 2):
            d_out = min(x + 1, z + 1, N + 1 - x, N + 1 - z)
            d_in = max(lo + 1 - x, x - (hi - 1), lo + 1 - z, z - (hi - 1)) - 1
            if d_in < 0:
                continue
            d = min(d_out, d_in)
            y = y0 + d // 2
            b.set(x, y, z, tiles())
            for yy in range(y0, y):
                if b.get(x, yy, z) is None:
                    b.set(x, yy, z, state("spruce_planks"))
    # The court: a cistern and pots.
    c = N // 2
    b.set(c, 1, c, state("water_cauldron", level="3"))
    for (x, z) in ((lo + 1, lo + 1), (hi - 2, hi - 1), (lo + 1, hi - 1)):
        b.set(x, 1, z, state("potted_azalea_bush" if (x + z) % 2 else "potted_red_tulip"))
    b.tradition, b.purpose, b.kind = "mediterranean", "courtyard-house", "house"  # type: ignore[attr-defined]
    b.notes.append(f"{name}: Mediterranean courtyard house, {N + 1} square, {render} render, court "
                   f"{hi - lo - 1} square with an outside stair")
    return b


# ------------------------------------------------------------------ Pueblo
def pueblo_terrace(player: Any = "blocky", seed: int = 0, name: str = "pueblo-terrace") -> BlockBuild:
    """A Pueblo room block, one house wide in a contiguous row (party walls
    east and west): two or three storeys of packed earth, each stepping back
    from the plaza in front and a little from the back, so every roof is the
    terrace of the rooms above. Flat earth roofs on round beams (vigas) whose
    ends run out through the wall head; a low rounded parapet of mud brick;
    a mud-brick footing; few, small openings; doors onto each terrace, and
    ladders from the plaza to the first roof and from roof to roof."""
    rng = random.Random(seed * 7919 + 71)
    b = BlockBuild(name, player, seed)
    S = b.sizes.storey
    X = 7                                       # x 0 .. X: the house's frontage
    n = rng.choice((2, 3, 3))
    Z = 13 if n == 2 else 15
    mud, brick = state("packed_mud"), state("mud_bricks")
    spans = [(3 * k, Z - 2 * k) for k in range(n)]
    for k, (za, zb) in enumerate(spans):
        yb = k * S
        for x in range(X + 1):
            for z in range(za, zb + 1):
                wall = x in (0, X) or z in (za, zb)
                if k == 0:
                    b.set(x, 0, z, brick if wall else mud)
                if wall:
                    for y in range(yb + 1, yb + S):
                        b.set(x, y, z, mud)
                b.set(x, yb + S, z, mud)          # roof: the terrace above
        # Vigas: round beams through the wall head, ends out front and back.
        for x in range(1, X, 2):
            for z in (za - 1, zb + 1):
                b.set(x, yb + S - 1, z, state("stripped_spruce_log", axis="z"))
        # Inside, the vigas span the short way, wall to wall.
        for z in range(za + 1, zb, 2):
            for x in range(1, X):
                b.set(x, yb + S - 1, z, state("stripped_spruce_log", axis="x"))
    # Parapets round each roof, where no higher wall stands on it.
    for k, (za, zb) in enumerate(spans):
        y = (k + 1) * S + 1
        upper = spans[k + 1] if k + 1 < n else None
        for x in range(X + 1):
            for z in range(za, zb + 1):
                edge = x in (0, X) or z in (za, zb)
                if not edge or (upper and upper[0] <= z <= upper[1]):
                    continue
                b.set(x, y, z, mud)               # a low, unbroken earth parapet
        # Canales: wooden spouts through the parapet foot shed the roof's rain
        # clear of the wall, front and back.
        for x in (2, X - 2):
            for z in (za - 1, zb + 1):
                b.set(x, y, z, state("spruce_trapdoor", half="bottom",
                                      facing="north" if z < za else "south", open="false"))
    # Openings: a street door, one small window either side of the roof
    # ladder on the ground storey, a small back window or two.
    door_x = 3
    port = b.door(door_x, 1, 0, "south", wood="spruce", name=name + "-street")
    port["id"] = port["id"].replace("port.", "port.street.")
    b.set(6, 2, 0, state("glass_pane"))
    for x in (2, 5):
        b.set(x, 2, Z, state("glass_pane"))
    # Upper storeys open onto the terrace below them.
    for k in range(1, n):
        za, zb = spans[k]
        b.door(4, k * S + 1, za, "south", wood="spruce", name=f"{name}-terrace{k}")
        b.set(1, k * S + 2, za, state("glass_pane"))
        b.set(5, k * S + 2, zb, state("glass_pane"))
    # Ladders: from the plaza to the first roof, then roof to roof; their
    # poles stand above the parapet as they do on the real thing.
    for k in range(n):
        za = spans[k][0]
        lx = X - 1 if k % 2 == 0 else 1
        for y in range(k * S + 1, (k + 1) * S + 2):
            b.set(lx, y, za - 1, state("ladder", facing="north"))
        b.set(lx, (k + 1) * S + 1, za, None)          # a gap in the parapet at the ladder's head
    # A beehive oven (horno) on the first terrace.
    if n >= 2:
        hx, hz = 2, 1
        for (dx, dz) in ((0, 0), (1, 0), (0, 1), (1, 1)):
            b.set(hx + dx, S + 1, hz + dz, state("mud_bricks"))
        b.set(hx, S + 2, hz, state("mud_brick_slab", type="bottom"))
    b.party_walls = {"east", "west"}  # type: ignore[attr-defined]
    b.tradition, b.purpose, b.kind = "pueblo", "row-house", "house"  # type: ignore[attr-defined]
    b.notes.append(f"{name}: Pueblo room block, {n} storeys stepping back, {X + 1} by {Z + 1}")
    return b


# -------------------------------------------------------- English shop
def english_shop(player: Any = "blocky", seed: int = 0, name: str = "english-shop", row: bool = False,
                 street_door: bool = False) -> BlockBuild:
    """A seld: the small lock-up shop of a medieval market street, a lean-to
    one bay deep. A timber frame with lime infill on a rubble footing, the
    door and a shop window to the street, a loft under a single-pitch roof
    of shingles that rises from the street eave to a high back wall. The
    back wall is meant to stand against a bank, a terrace wall or another
    building, so it is not a facade. With ``row`` it is built to fill a gap
    in a street front: its side walls are party walls against the houses on
    either hand, with no sill, window or eave out over them. With
    ``street_door`` its front wall stands on the street line itself, the door
    opening straight onto the street and the counter in the shop window, so
    it fits a strip only as deep as the shop (a seld built against the back
    of a house on the next street)."""
    rng = random.Random(seed * 7919 + 73)
    b = BlockBuild(name, player, seed)
    W = rng.choice((3, 4, 4, 5, 6) if row else (5, 6, 6, 7))   # walls at x = 0 and W
    D = rng.choice((2, 3, 4, 5))               # walls at z = 0 (street) and D: a booth to a shop
    Hf = 4                                     # the street eave
    frame, infill = state("dark_oak_log"), state("calcite")
    foot = b.mix(("cobblestone", 3), ("mossy_cobblestone", 1), ("stone", 1))
    for x in range(W + 1):
        for z in range(D + 1):
            wall = x in (0, W) or z in (0, D)
            b.set(x, 0, z, foot() if wall else state("spruce_planks"))
            if not wall:
                continue
            top = Hf + z                       # side walls follow the roof
            for y in range(1, top + 1):
                post = (x in (0, W) and z in (0, D)) or (z == 0 and x in (2, W // 2 + 1)) or \
                       (x in (0, W) and z == D // 2 + 1) or y == Hf
                if z == D:
                    b.set(x, y, z, foot())     # rubble back wall, against the bank
                else:
                    b.set(x, y, z, frame if post else infill)
    # A sill course of rubble proud of the frame, front and sides.
    lo, hi = (0, W) if row else (-1, W + 1)
    for u in range(lo, hi + 1):
        if not street_door:
            b.set(u, 0, -1, foot())
    for z in range(0, D + 1):
        if not row:
            b.set(-1, 0, z, foot())
            b.set(W + 1, 0, z, foot())
    # Door and shop window to the street, a small light each side.
    port = b.door(1, 1, 0, "south", wood="dark_oak", name=name + "-street")
    port["id"] = port["id"].replace("port.", "port.street.")
    for x in range(3, W):
        if b.get(x, 2, 0) == infill:
            b.set(x, 2, 0, state("glass_pane"))
    cx = W // 2 + 2 if W // 2 + 2 < W else W - 1
    if street_door:
        b.set(cx, 1, 0, state("spruce_slab", type="top"))   # the counter, in the window
        # A pentice board over the shop front keeps the rain off the counter,
        # high enough to walk under.
        for x in range(1, W):
            b.set(x, 3, -1, state("spruce_slab", type="top"))
    else:
        b.set(cx, 1, -1, state("spruce_slab", type="top"))  # the counter
    for x in (() if row else (0, W)):
        b.set(x, 2, 1 + (D > 4), state("glass_pane"))
        b.set(x, 3, 1 + (D > 4), state("glass_pane"))
    # Single-pitch shingle roof, a block out at the street and the sides.
    for x in range(lo, hi + 1):
        for z in range(-1, D + 1):
            y = Hf + 1 + z
            b.set(x, y, z, state("dark_oak_stairs", facing="south", half="bottom"))
    # A loft floor and a ladder up to it.
    for x in range(1, W):
        for z in range(1, D):
            if not (x == W - 1 and z == D - 1):
                b.set(x, Hf, z, state("spruce_planks"))
    b.set(W - 1, Hf, D - 1, None)
    for y in range(1, Hf + 1):
        b.set(W - 1, y, D - 1, state("ladder", facing="north"))
    b.party_walls = {"south", "east", "west"} if row else {"south"}  # type: ignore[attr-defined]
    b.street_door = int(street_door)  # type: ignore[attr-defined]
    b.tradition, b.purpose, b.kind = "english-medieval", "shop", "house"  # type: ignore[attr-defined]
    b.notes.append(f"{name}: a seld, {W + 1} by {D + 1}, lean-to shingle roof")
    return b


# --------------------------------------------------------------- registry
SUB_ARCHITECTS: Dict[Tuple[str, str], Callable[..., BlockBuild]] = {
    ("tudor", "merchant-house"): tudor_merchant_house,
    ("hanseatic", "merchant-house"): hanseatic_gable_house,
    ("norse", "longhouse"): norse_longhouse,
    ("alpine", "cottage"): alpine_chalet,
    ("mediterranean", "courtyard-house"): mediterranean_courtyard_house,
    ("pueblo", "row-house"): pueblo_terrace,
    ("english-medieval", "shop"): english_shop,
}


def build(tradition: str, purpose: str, player: Any = "blocky", seed: int = 0) -> BlockBuild:
    key = (tradition, purpose)
    if key not in SUB_ARCHITECTS:
        have = ", ".join(f"{t}/{p}" for t, p in sorted(SUB_ARCHITECTS))
        raise ValueError(f"no sub-architect for {tradition}/{purpose}; there are: {have}")
    return SUB_ARCHITECTS[key](player, seed)


def purpose_of(tradition: str) -> str:
    """The purpose a tradition's sub-architect builds (the first, if several)."""
    for tr, pu in SUB_ARCHITECTS:
        if tr == tradition:
            return pu
    raise KeyError(f"no sub-architect for {tradition}")


def for_climate(climate: str) -> List[str]:
    """The traditions with a sub-architect whose home climate is ``climate``."""
    out = sorted({tr for tr, pu in SUB_ARCHITECTS if pu != "shop"
                  and getattr(lore.TRADITIONS.get(tr), "climate", "temperate") == climate})
    if not out:
        raise KeyError(f"no sub-architect for a {climate} climate")
    return out
