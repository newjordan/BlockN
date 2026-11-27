"""Block-mode building types beyond the plain house: a tower, a workshop with
a lean-to, a windmill and a mine.

The tower, workshop and windmill are houses (``blockcraft.house``) given a
shape of their own, so their floors, stairs and doors are sized to the
player exactly as a house's are and the same walk check covers them. What
they add outside (the lean-to's tools, the mill's sails, the tower's flag)
is decoration: the player need not reach it, and none of it stands on a
floor or in a lane.

The mine is dug, not built. Its blocks include ``cave_air`` for every block
of rock it takes out, so pasting it into a town carves the hill, and a skin
of rock and ore round every carved block, so it is closed wherever it runs.
Its entrance faces north (-z) at z = 0 and it runs south (+z) into the hill.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from .blockcraft import _reach_body, house
from .blockmode import BlockBuild, kind, parse, state

Pos = Tuple[int, int, int]

AIR = "minecraft:cave_air"


def _extent(b: BlockBuild) -> Tuple[int, int, int]:
    """(X, Z, top): the far corner of a house's base and its eave layer."""
    base = [(x, z) for (x, y, z), blk in b.blocks.items() if y == 0 and 0 <= x and 0 <= z]
    X = max(x for x, _ in base)
    Z = max(z for _, z in base)
    return X, Z, 0


def _ridge(b: BlockBuild, x: int, z: int) -> int:
    return max(y for (bx, y, bz) in b.blocks if bx == x and bz == z)


# ------------------------------------------------------------------ tower
def tower(player: Any = "blocky", seed: int = 0, storeys: Optional[int] = None, name: str = "tower") -> BlockBuild:
    """A square stone tower of three or four storeys under a hipped roof,
    with a flag on its top. Inside it is a house: a stair up every storey,
    rooms on each floor."""
    probe = BlockBuild(name, player, seed)
    s = probe.sizes
    storeys = storeys or probe.rng.choice((3, 3, 4))
    w = s.storey + s.lane + 5
    b = house(player, seed, width=w, depth=w, storeys=storeys, style="tower", name=name, roof="hip")
    X, Z, _ = _extent(b)
    cx, cz = X // 2, Z // 2
    top = _ridge(b, cx, cz)
    # The flag: a pole of fence posts on the cap, a banner on its top.
    for k in range(1, 4):
        b.set(cx, top + k, cz, state("spruce_fence"))
    colour = b.rng.choice(("red", "blue", "yellow", "white", "green"))
    b.set(cx, top + 4, cz, state(f"{colour}_banner", rotation=str(b.rng.randint(0, 15))))
    # Corbels under the eaves on every side: a course of upside-down stone
    # stairs, as on a tower's machicolations (outside the walls, decoration).
    eave = storeys * s.storey
    for x in range(0, X + 1, 2):
        b.set(x, eave - 2, -1, state("stone_brick_stairs", facing="south", half="top"), replace=False)
        b.set(x, eave - 2, Z + 1, state("stone_brick_stairs", facing="north", half="top"), replace=False)
    for z in range(0, Z + 1, 2):
        b.set(-1, eave - 2, z, state("stone_brick_stairs", facing="east", half="top"), replace=False)
        b.set(X + 1, eave - 2, z, state("stone_brick_stairs", facing="west", half="top"), replace=False)
    b.kind = "tower"  # type: ignore[attr-defined]
    return b


# --------------------------------------------------------------- workshop
_STATIONS = {
    "smith": ("anvil[facing=east]", "blast_furnace[facing=east,lit=true]", "lava_cauldron", "grindstone[face=floor]",
              "smithing_table", "barrel[facing=up]", "chest[facing=east]"),
    "carpenter": ("crafting_table", "stonecutter[facing=east]", "barrel[facing=up]", "stripped_oak_log[axis=x]",
                  "stripped_spruce_log[axis=x]", "chest[facing=east]", "composter"),
    "mason": ("stonecutter[facing=east]", "stone_bricks", "chiseled_stone_bricks", "cobblestone",
              "crafting_table", "barrel[facing=up]", "chest[facing=east]"),
    "potter": ("furnace[facing=east,lit=true]", "decorated_pot", "decorated_pot", "barrel[facing=up]",
               "clay", "crafting_table", "chest[facing=east]"),
}


def workshop(player: Any = "blocky", seed: int = 0, trade: Optional[str] = None,
             name: str = "workshop") -> BlockBuild:
    """A house with an open lean-to along its east end: a slab roof on posts
    over a stone floor, the trade's stations against the house wall and a
    clear walk in front of them, open to the street end."""
    probe = BlockBuild(name, player, seed)
    rng = probe.rng
    trade = trade or rng.choice(sorted(_STATIONS))
    b = house(player, seed, storeys=rng.choice((1, 2, 2)), name=name, width=1, depth=1)
    s = b.sizes
    X, Z, _ = _extent(b)
    D = s.lane + 3                       # depth of the lean-to: stations, a lane, the posts
    yr = s.door_height + 2               # its roof layer: a door's height and a block over the lane
    x0, x1 = X + 1, X + D
    for x in range(x0, x1 + 1):
        for z in range(0, Z + 1):
            b.set(x, 0, z, state(rng.choice(("stone_bricks", "cobblestone", "andesite", "stone_bricks"))))
    for x in range(x0, x1 + 2):
        for z in range(-1, Z + 2):
            b.set(x, yr, z, state("spruce_slab", type="bottom"), replace=False)
    for z in sorted({0, Z // 2, Z}):
        for y in range(1, yr):
            b.set(x1, y, z, state("spruce_fence"))
    items = list(_STATIONS[trade])
    rng.shuffle(items)
    zs = [z for z in range(1, Z)]
    for z, blk in zip(zs, items):
        b.set(x0, 1, z, f"minecraft:{blk}")
    # A lantern hung from the lean-to's roof over the walk.
    b.set(x0 + 1 + s.lane // 2, yr - 1, Z // 2, state("lantern", hanging="true"))
    from .blockdress import fill_containers
    fill_containers(b, trade)
    b.kind = "workshop"  # type: ignore[attr-defined]
    b.notes[-1] += f"; a {trade}'s lean-to on the east end"
    return b


# ---------------------------------------------------------------- apartment
def apartment(player: Any = "blocky", seed: int = 0, storeys: Optional[int] = None,
              name: str = "apartment") -> BlockBuild:
    """A tall, shallow house of flats, built to stand with its back to a rock
    face: three to five storeys on a stone ground floor, a stair through every
    storey. In a town its back wall goes against a cliff, and a door is cut in
    it where a storey's floor meets the ground on top."""
    import random as _random
    rng = _random.Random(seed * 7919 + 13)
    storeys = storeys or rng.choice((3, 4, 4, 5))
    style = rng.choice(("stone", "brick", "stone", "timber", "oak"))
    from .blockcraft import HOUSE_STYLES
    if style not in HOUSE_STYLES:
        style = sorted(HOUSE_STYLES)[seed % len(HOUSE_STYLES)]
    # Narrow, so a row of them can stand along a face: the least width the
    # stair allows, or two blocks more.
    b = house(player, seed, width=1 + rng.choice((0, 0, 2)), depth=1, storeys=storeys, style=style, name=name,
              roof=("gable", "hip", "cross")[seed % 3])
    b.kind = "apartment"  # type: ignore[attr-defined]
    b.storeys = storeys  # type: ignore[attr-defined]
    # Its back stands against the rock: not a facade. The other three get
    # windows in a steady rhythm up every storey, and any blank panel left
    # is broken by a window or a post.
    b.party_walls = {"south"}  # type: ignore[attr-defined]
    from .blockstyles import relieve_flat_walls, window_rhythm
    window_rhythm(b)
    relieve_flat_walls(b)
    b.notes[-1] += f"; flats, {storeys} storeys, back to the rock"
    return b


# ---------------------------------------------------------------- windmill
def windmill(player: Any = "blocky", seed: int = 0, name: str = "windmill") -> BlockBuild:
    """A three-storey mill house, square, under a hipped roof, with four
    sails on an axle out of its east wall. The sails are decoration: their
    lowest point stays clear over a body's head."""
    probe = BlockBuild(name, player, seed)
    s = probe.sizes
    w = s.storey + s.lane + 4
    b = house(player, seed, width=w, depth=w, storeys=3, style="mill", name=name, roof="hip")
    X, Z, _ = _extent(b)
    top = 3 * s.storey
    hy = top - 2                                   # the hub, high in the top storey
    zc = Z // 2
    xs = X + 3                                     # the plane of the sails, clear of the eaves
    for x in range(X + 1, xs + 1):
        b.set(x, hy, zc, state("stripped_spruce_log", axis="x"))
    b.set(X, hy, zc, state("stripped_spruce_log", axis="x"))
    b.set(xs + 1, hy, zc, state("spruce_planks"))
    clear = s.door_height + 2                      # lowest a sail may come
    L = max(4, int((hy - clear) / math.sqrt(0.5)) - 2)
    r2 = math.sqrt(0.5)
    while True:
        cells: Dict[Tuple[int, int], str] = {}
        R = L + 4
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                for sy, sz in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                    u = (dy * sy + dz * sz) * r2
                    v = (dy * sz - dz * sy) * r2   # across the arm, on its leading side
                    if 0 < u <= L * math.sqrt(2) + .2 and abs(v) <= .5:
                        cells[(dy, dz)] = "spruce_planks"
                    elif 1.8 < u <= L * math.sqrt(2) + .2 and .5 < v <= 2.6 and (dy, dz) not in cells:
                        cells[(dy, dz)] = "white_wool"
        if hy + min(dy for dy, _ in cells) >= clear or L <= 4:
            break
        L -= 1
    for (dy, dz), blk in cells.items():
        b.set(xs, hy + dy, zc + dz, state(blk), replace=False)
    # The sails and the windshaft turn on the shaft's bearing in the cap: the
    # load-path check counts them as a mechanism hung from it, not as blocks
    # that must stand on the ones below.
    b.mechanisms = {(xs, hy + dy, zc + dz) for (dy, dz) in cells} | {  # type: ignore[attr-defined]
        (x, hy, zc) for x in range(X + 1, xs + 2)}
    from .blockstyles import relieve_flat_walls
    relieve_flat_walls(b)
    b.kind = "windmill"  # type: ignore[attr-defined]
    b.notes[-1] = b.notes[-1].replace(f"{name}: ", f"{name}: sails {L} blocks, ", 1)
    return b


# -------------------------------------------------------------------- mine
_ORES_HIGH = (("stone", 30), ("andesite", 6), ("tuff", 4), ("gravel", 2), ("coal_ore", 3), ("copper_ore", 2),
              ("iron_ore", 1.5))
_ORES_DEEP = (("deepslate", 26), ("tuff", 6), ("cobbled_deepslate", 3), ("deepslate_iron_ore", 2.5),
              ("deepslate_coal_ore", 1.5), ("deepslate_gold_ore", .8), ("deepslate_redstone_ore", .8),
              ("deepslate_lapis_ore", .4))
_MINE_FLOOR = (("gravel", 3), ("cobblestone", 3), ("coarse_dirt", 2), ("stone", 2), ("andesite", 1))


class _Dig:
    """A mine being dug: air carved, floors laid, supports set, and the
    floors a body walks recorded for the walk check."""

    def __init__(self, b: BlockBuild):
        self.b = b
        self.air: Set[Pos] = set()
        self.walk: Set[Pos] = set()        # (x, surface y, z) a body should reach
        self.floor_mix = b.mix(*_MINE_FLOOR)

    def carve(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> None:
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for z in range(min(z0, z1), max(z0, z1) + 1):
                    self.air.add((x, y, z))

    def floor(self, x0: int, z0: int, x1: int, z1: int, y: int, walk: bool = True) -> None:
        """Floor blocks at layer ``y``, walked on at ``y + 1``."""
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for z in range(min(z0, z1), max(z0, z1) + 1):
                self.b.set(x, y, z, self.floor_mix())
                if walk:
                    self.walk.add((x, y + 1, z))

    def finish(self, wood: str) -> None:
        """Air into every carved block not otherwise filled; a skin of rock
        and ore round the carving."""
        b = self.b
        deep = b.mix(*_ORES_DEEP)
        high = b.mix(*_ORES_HIGH)
        for p in self.air:
            if p not in b.blocks:
                b.blocks[p] = AIR
        for (x, y, z) in list(self.air):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        q = (x + dx, y + dy, z + dz)
                        if q[2] < 0:
                            continue      # before the hill face: the open apron
                        if q not in b.blocks:
                            b.blocks[q] = deep() if q[1] < -6 else high()


def _supports(d: _Dig, run: List[Tuple[int, int]], xa: int, xb: int, y: int, H: int, wood: str, axis: str,
              every: int = 4) -> None:
    """Timber sets along a passage: a post in each wall and a cap beam
    across the top layer of the carving, every ``every`` cells of ``run``.
    ``run`` lists (along, index) cells; the passage spans xa .. xb across."""
    b = d.b
    for k, (a, _) in enumerate(run):
        if k % every:
            continue
        for yy in range(y + 1, y + H + 1):
            if axis == "z":
                b.set(xa - 1, yy, a, state(f"{wood}_log", axis="y"))
                b.set(xb + 1, yy, a, state(f"{wood}_log", axis="y"))
            else:
                b.set(a, yy, xa - 1, state(f"{wood}_log", axis="y"))
                b.set(a, yy, xb + 1, state(f"{wood}_log", axis="y"))
        for c in range(xa - 1, xb + 2):
            if axis == "z":
                b.set(c, y + H + 1, a, state(f"stripped_{wood}_log", axis="x"))
            else:
                b.set(a, y + H + 1, c, state(f"stripped_{wood}_log", axis="z"))


def _incline(d: _Dig, x0: int, x1: int, z0: int, y_top: int, drop: int, H: int, stairs: bool,
             block: str) -> Tuple[int, int]:
    """A flight down toward +z from the landing at z0 - 1 (walked at
    y_top + 1): stair blocks dropping a block a cell, or for a body that
    does not climb stairs, slabs and blocks dropping half a block a cell.
    Returns (z just past the foot, floor layer at the foot)."""
    b = d.b
    z = z0
    if stairs:
        for j in range(drop):
            y = y_top - j
            for x in range(x0, x1 + 1):
                b.set(x, y, z, state(f"{block}_stairs", facing="north", half="bottom"))
                for yy in range(y - 2, y):
                    b.set(x, yy, z, state("cobblestone"), replace=False)
            d.carve(x0, y + 1, z, x1, y + 1 + H, z)
            z += 1
        return z, y_top - drop
    full = {"stone_brick": "stone_bricks"}.get(block, block)
    for k in range(1, 2 * drop + 1):
        # Half a block down each cell: a bottom slab, then a whole block.
        y = y_top - (k - 1) // 2 if k % 2 else y_top - k // 2
        for x in range(x0, x1 + 1):
            b.set(x, y, z, state(f"{block}_slab", type="bottom") if k % 2 else state(full))
            b.set(x, y - 1, z, state("cobblestone"), replace=False)
        d.carve(x0, y + 1, z, x1, y + 1 + H, z)
        z += 1
    return z, y_top - drop


def mine(player: Any = "blocky", seed: int = 0, name: str = "mine", hill: bool = True) -> BlockBuild:
    """A mine dug into a hill: a timber portal at z = 0, a level adit into
    the rock with a rail down it, a flight down to a working chamber, two
    galleries off it east and west to ore faces, and on some seeds a second
    flight to a deeper chamber.

    Passages are the lane plus two wide (a rail down the middle, a body
    beside it) and a door's height plus one clear under each timber set, a
    block more between sets. A body that climbs stairs goes down stair
    blocks; one that does not goes down half-block steps. With ``hill`` the
    mine sits in a block of rock with a level apron before the portal, so it
    stands and walks on its own; in a town the town's hill takes its place.
    ``b.mine`` records the floors a body should reach."""
    b = BlockBuild(name, player, seed)
    rng = b.rng
    s = b.sizes
    d = _Dig(b)
    wood = rng.choice(("spruce", "oak", "dark_oak"))
    w = max(3, s.lane + 2)
    H = s.door_height + 1                 # clear under a timber set
    xa, xb = 0, w - 1
    rail_x = w // 2
    stair_block = rng.choice(("cobblestone", "stone_brick", "cobbled_deepslate"))

    # The portal: a timber frame in the hill face, lanterns on posts before it.
    L1 = rng.randint(10, 15)
    d.carve(xa, 1, 0, xb, H + 1, L1)
    d.floor(xa, 0, xb, L1, 0)
    for y in range(1, H + 2):
        b.set(xa - 1, y, 0, state(f"{wood}_log", axis="y"))
        b.set(xb + 1, y, 0, state(f"{wood}_log", axis="y"))
    for x in range(xa - 2, xb + 3):
        b.set(x, H + 2, 0, state(f"{wood}_log", axis="x"))
    for x in (xa - 2, xb + 2):
        b.set(x, 1, -1, state(f"{wood}_fence"))
        b.set(x, 2, -1, state("lantern"))
    _supports(d, [(z, 0) for z in range(4, L1 + 1)], xa, xb, 0, H, wood, "z")
    for z in range(1, L1 + 1):
        b.set(rail_x, 1, z, state("rail", shape="north_south"))
    for z in range(4, L1 + 1, 8):
        b.set(xa, H, z, state("lantern", hanging="true"))

    # The flight down to the chamber.
    S = s.storey
    drop = S + 2
    zc, yb = _incline(d, xa, xb, L1 + 1, 0, drop, H, s.climbs_stairs, stair_block)
    # The chamber: four blocks wider than a passage, two higher.
    C = rng.randint(8, 11)
    cx0, cx1 = xa - 2, xb + 2
    cz0, cz1 = zc, zc + C - 1
    Hc = H + 2
    zg = (cz0 + cz1) // 2                 # the galleries leave from the chamber's middle
    ga, gb = zg - w // 2, zg - w // 2 + w - 1
    d.carve(cx0, yb + 1, cz0, cx1, yb + Hc + 1, cz1)
    d.floor(cx0, cz0, cx1, cz1, yb)
    for z in range(cz0 + 1, cz1 + 1, 3):
        for x in range(cx0 - 1, cx1 + 2):
            b.set(x, yb + Hc + 1, z, state(f"stripped_{wood}_log", axis="x"))
        if ga - 1 <= z <= gb + 1:
            continue                      # no post across a gallery's mouth
        for x in (cx0 - 1, cx1 + 1):
            for y in range(yb + 1, yb + Hc + 1):
                b.set(x, y, z, state(f"{wood}_log", axis="y"))
    for z in range(cz0, cz1 + 1):
        b.set(rail_x, yb + 1, z, state("rail", shape="north_south"))
    b.set((cx0 + cx1) // 2 - 1, yb + Hc, cz0 + 1, state("lantern", hanging="true"))
    b.set((cx0 + cx1) // 2 + 1, yb + Hc, cz0 + 4, state("lantern", hanging="true"), replace=False)

    # Galleries east and west from the chamber's middle, to ore faces.
    galleries = []
    for side in (-1, 1):
        L2 = rng.randint(9, 14)
        if side < 0:
            g0, g1 = cx0 - L2, cx0 - 1
            run = list(range(g1, g0 - 1, -1))
        else:
            g0, g1 = cx1 + 1, cx1 + L2
            run = list(range(g0, g1 + 1))
        d.carve(g0, yb + 1, ga, g1, yb + H + 1, gb)
        d.floor(g0, ga, g1, gb, yb)
        _supports(d, [(x, 0) for x in run[2:]], ga, gb, yb, H, wood, "x")
        rz = ga + w // 2
        for x in run:
            b.set(x, yb + 1, rz, state("rail", shape="east_west"))
        b.set(run[2], yb + H, ga, state("lantern", hanging="true"))
        # The ore face at the end: rich, dug at.
        end = run[-1] + side
        ores = ("iron_ore", "coal_ore", "copper_ore", "deepslate_iron_ore", "raw_iron_block", "deepslate_gold_ore")
        for z in range(ga - 1, gb + 2):
            for y in range(yb, yb + H + 2):
                b.set(end, y, z, state(rng.choice(ores)))
        galleries.append({"side": "west" if side < 0 else "east", "length": L2})

    # Chamber stores against its walls, clear of the passage mouths.
    stores = ["chest[facing=east]", "barrel[facing=up]", "barrel[facing=up]", "crafting_table",
              "furnace[facing=east]", "raw_iron_block", "coal_block", "raw_copper_block"]
    rng.shuffle(stores)
    spots = [(x, z) for x in (cx0, cx1) for z in range(cz0 + 1, cz1 + 1) if not (ga - 1 <= z <= gb + 1)]
    rng.shuffle(spots)
    free = {(x, z) for x in range(cx0, cx1 + 1) for z in range(cz0, cz1 + 1)}
    entry = [(x, cz0) for x in range(xa, xb + 1)]
    for (x, z) in spots:
        if not stores:
            break
        # A store goes only where every other cell of the chamber floor is
        # still reached by a body as wide as the lane.
        trial = free - {(x, z)}
        if _reach_body(trial, entry, s.lane) != trial:
            continue
        free = trial
        blk = stores.pop()
        facing = "east" if x == cx0 else "west"
        b.set(x, yb + 1, z, "minecraft:" + blk.replace("facing=east", f"facing={facing}"))
        d.walk.discard((x, yb + 1, z))

    levels = 1
    deeper = None
    if rng.random() < 0.6:
        # A second flight from the chamber's far end to a deeper working.
        z2, y2 = _incline(d, xa, xb, cz1 + 1, yb, S + 1, H, s.climbs_stairs, stair_block)
        C2 = rng.randint(6, 8)
        d.carve(cx0, y2 + 1, z2, cx1, y2 + Hc + 1, z2 + C2 - 1)
        d.floor(cx0, z2, cx1, z2 + C2 - 1, y2)
        for z in range(z2 + 1, z2 + C2, 3):
            for x in range(cx0 - 1, cx1 + 2):
                b.set(x, y2 + Hc + 1, z, state(f"stripped_{wood}_log", axis="x"))
        b.set((cx0 + cx1) // 2, y2 + Hc, z2 + 1, state("lantern", hanging="true"))
        # Glow lichen and amethyst on the far face: deep-rock decoration.
        for x in range(cx0, cx1 + 1):
            for y in range(y2 + 1, y2 + Hc + 1):
                b.set(x, y, z2 + C2, state(rng.choice(("amethyst_block", "deepslate_diamond_ore", "deepslate",
                                                        "budding_amethyst", "deepslate_lapis_ore"))))
        levels = 2
        deeper = {"floor": y2, "z": [z2, z2 + C2 - 1]}

    # Carts on the rails, a chest cart loaded at the end of a gallery; goods
    # in the chamber's stores; a framed pickaxe on the chamber wall.
    from .blockdress import fill_containers, hang
    b.entity("minecart", rail_x, 1, 2)
    end_x = (cx0 - galleries[0]["length"]) if galleries else rail_x
    b.entity("chest_minecart", end_x + 1, yb + 1, ga + w // 2,
             items=[("raw_iron", rng.randint(8, 32)), ("coal", rng.randint(16, 48)), ("raw_copper", rng.randint(8, 40))])
    d.finish(wood)
    fill_containers(b, "mine")
    hang(b, (cx0, cz0, cx1, cz1), yb + 2, "mine", frames=2, paintings=0)
    if hill:
        # A block of rock round it all, its top a hill face over the portal,
        # and a level apron of grass and gravel before the portal.
        (lx, ly, lz), (hx, hy, hz) = b.bounds()
        top = H + 6
        stone = b.mix(*_ORES_HIGH)
        for x in range(lx - 2, hx + 3):
            for z in range(0, hz + 3):
                for y in range(ly - 1, top + 1):
                    if (x, y, z) not in b.blocks:
                        b.blocks[(x, y, z)] = stone() if y < top else state("grass_block")
        for x in range(lx - 2, hx + 3):
            for z in range(-6, 0):
                if (x, 0, z) not in b.blocks:
                    b.set(x, 0, z, state("gravel" if xa - 1 <= x <= xb + 1 else "grass_block"))
                for y in range(ly - 1, 0):
                    b.blocks.setdefault((x, y, z), stone())
    # The entrance as a port, as a house's door is.
    port = {"id": f"port.{name}.000", "source": "declared",
            "position": [(xa + xb + 1) / 2.0, 1.0, 0.0], "facing": [0.0, 0.0, -1.0], "wall_axis": "x",
            "width": float(w), "height": float(H), "blocks": [w, H], "profile": b.player.name}
    b.ports.append(port)
    b.kind = "mine"  # type: ignore[attr-defined]
    b.mine = {"walk": sorted(d.walk), "width": w, "clear": H, "adit": L1, "chamber": [cz0, cz1, yb],  # type: ignore
              "galleries": galleries, "levels": levels, "deeper": deeper,
              "stairs": "stair blocks" if s.climbs_stairs else "half-block steps"}
    b.notes.append(f"{name}: adit {L1} long, {w} wide, {H} clear under the sets; down {drop} to a chamber of {C}, "
                   f"galleries {', '.join(str(g['length']) for g in galleries)} long"
                   + (f", a second flight to a deeper working" if deeper else ""))
    return b


def mine_walk(b: BlockBuild, w=None) -> Dict[str, Any]:
    """Walk a mine (on its own, with its hill) from the apron and say what
    share of its floors the body reaches."""
    import numpy as np
    from .gameplay import walk as walk_grid

    sub = b.walk_sub()
    g = b.grid(sub)
    ox, oy, oz = g.origin
    if w is None:
        seed = (int((0.5 - ox) * sub), int((1 - oy) * sub), int((-3.5 - oz) * sub))
        w = walk_grid(g, b.player, seeds=[seed])
    nodes = w.nodes
    wx = np.floor(nodes[:, 0] / sub + ox).astype(int)
    wy = nodes[:, 1] / sub + oy
    wz = np.floor(nodes[:, 2] / sub + oz).astype(int)
    reached = w.reached
    targets = b.mine["walk"]
    ok = 0
    missing = []
    key = wx * 1000003 + wz
    for (x, y, z) in targets:
        here = (key == x * 1000003 + z) & (np.abs(wy - y) < .3)
        if here.any() and reached[here].any():
            ok += 1
        else:
            missing.append((x, y, z))
    return {"floors": len(targets), "reached": ok, "missing": missing[:12]}


__all__ = ["tower", "workshop", "windmill", "mine", "mine_walk", "AIR"]
