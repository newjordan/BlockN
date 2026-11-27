"""Set dressing for block-mode builds that Minecraft keeps as block entities
and entities: what is in a chest or barrel, the patterns on a banner, item
frames and paintings on the walls, carts on a mine's rails.

None of it changes a block or a floor: block entity data rides on blocks
already placed (``b.nbt``), and entities (``b.entities``) are hung in empty
blocks against a wall, where nothing collides with them, so the walk check
is the same with or without them. The export writes them as SNBT on the
``setblock`` and as ``summon`` commands, in Java Edition 1.21 form.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .blockmode import DIRS, OPPOSITE, PAINTINGS_1X1, BlockBuild, _AnyWall, kind, parse

# What a container holds, by theme: (item, most of it).
_GOODS: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "baker": (("bread", 12), ("wheat", 32), ("sugar", 16), ("egg", 12), ("pumpkin_pie", 4), ("cake", 1)),
    "smith": (("iron_ingot", 24), ("coal", 32), ("iron_sword", 1), ("iron_pickaxe", 1), ("shield", 1),
              ("iron_nugget", 40), ("lava_bucket", 1)),
    "weaver": (("white_wool", 32), ("string", 24), ("shears", 1), ("light_gray_wool", 16), ("cyan_dye", 8),
               ("red_dye", 8)),
    "brewer": (("glass_bottle", 12), ("sugar", 16), ("sweet_berries", 24), ("honey_bottle", 4), ("apple", 8)),
    "scribe": (("paper", 32), ("book", 8), ("writable_book", 1), ("feather", 12), ("ink_sac", 8), ("map", 2)),
    "farmer": (("wheat_seeds", 32), ("carrot", 24), ("potato", 24), ("beetroot", 16), ("bone_meal", 16),
               ("wheat", 32)),
    "carpenter": (("oak_planks", 48), ("stick", 32), ("oak_log", 16), ("iron_axe", 1), ("oak_slab", 24)),
    "mason": (("cobblestone", 64), ("stone_bricks", 48), ("iron_pickaxe", 1), ("chiseled_stone_bricks", 8)),
    "potter": (("clay_ball", 32), ("brick", 24), ("flower_pot", 6), ("decorated_pot", 2)),
    "house": (("bread", 8), ("apple", 6), ("torch", 16), ("coal", 12), ("stick", 16), ("white_wool", 4),
              ("candle", 4), ("baked_potato", 6)),
    "kitchen": (("bread", 10), ("apple", 8), ("carrot", 12), ("potato", 12), ("cooked_cod", 6), ("bowl", 4),
                ("milk_bucket", 1)),
    "mine": (("coal", 48), ("raw_iron", 24), ("raw_copper", 32), ("torch", 32), ("rail", 24), ("iron_pickaxe", 1),
             ("cobblestone", 64), ("raw_gold", 6)),
    "castle": (("arrow", 48), ("bow", 1), ("iron_sword", 1), ("shield", 1), ("bread", 16), ("iron_helmet", 1),
               ("iron_chestplate", 1), ("torch", 32)),
}
# What hangs in a frame, by theme.
_FRAMED: Dict[str, Tuple[str, ...]] = {
    "baker": ("bread", "wheat"), "smith": ("iron_sword", "iron_pickaxe", "shield"), "weaver": ("shears", "white_wool"),
    "brewer": ("glass_bottle", "honey_bottle"), "scribe": ("map", "writable_book", "compass"),
    "farmer": ("wheat", "carrot", "golden_hoe"), "carpenter": ("iron_axe",), "mason": ("iron_pickaxe",),
    "potter": ("flower_pot",), "house": ("clock", "compass", "map"), "mine": ("iron_pickaxe", "diamond_pickaxe"),
    "castle": ("iron_sword", "shield", "bow"),
}
_BANNER_PATTERNS = ("stripe_center", "stripe_middle", "cross", "border", "half_horizontal", "chevron", "bricks",
                    "triangles_bottom", "rhombus", "circle", "flower", "stripe_top", "curly_border")
_DYES = ("white", "red", "blue", "yellow", "black", "green", "light_blue", "orange", "gray", "purple")


def fill_containers(b: BlockBuild, theme: str, rng: Optional[random.Random] = None) -> int:
    """Give every chest and barrel that holds nothing yet a few stacks of the
    theme's goods, in random slots. Returns how many were filled."""
    rng = rng or b.rng
    goods = _GOODS.get(theme, _GOODS["house"])
    n = 0
    for pos, blk in sorted(b.blocks.items()):
        name = parse(blk)[0]
        if name not in ("chest", "barrel", "trapped_chest") or pos in b.nbt:
            continue
        picks = rng.sample(goods, min(len(goods), rng.randint(2, 5)))
        slots = rng.sample(range(27), len(picks))
        items = sorted(((s_, (it, rng.randint(1, most))) for s_, (it, most) in zip(slots, picks)))
        b.nbt[pos] = "{Items:" + _slotted(items) + "}"
        n += 1
    return n


def _slotted(items: Sequence[Tuple[int, Tuple[str, int]]]) -> str:
    parts = []
    for slot, (item, count) in items:
        parts.append(f'{{Slot:{slot}b,id:"minecraft:{item}",count:{int(count)}}}')
    return "[" + ",".join(parts) + "]"


def pattern_banners(b: BlockBuild, rng: Optional[random.Random] = None) -> int:
    """Two or three patterns on every banner that has none yet."""
    rng = rng or b.rng
    n = 0
    for pos, blk in sorted(b.blocks.items()):
        if kind(blk) not in ("banner", "wall_banner") or pos in b.nbt:
            continue
        base = parse(blk)[0].split("_wall_banner")[0].split("_banner")[0]
        pats = []
        for p in rng.sample(_BANNER_PATTERNS, rng.randint(2, 3)):
            c = rng.choice([d for d in _DYES if d != base])
            pats.append(f'{{pattern:"minecraft:{p}",color:"{c}"}}')
        b.nbt[pos] = "{patterns:[" + ",".join(pats) + "]}"
        n += 1
    return n


def _wall_behind(b: BlockBuild, x: int, y: int, z: int, facing: str) -> bool:
    """Is the block behind (x, y, z), seen facing ``facing``, a wall to hang
    on: a whole block, not glass, not furniture?"""
    dx, dz = DIRS[OPPOSITE[facing]]
    blk = b.get(x + dx, y, z + dz)
    if blk is None or kind(blk) != "full":
        return False
    name = parse(blk)[0]
    return name not in _AnyWall._FURNITURE and "glass" not in name and "leaves" not in name


def hang(b: BlockBuild, room: Tuple[int, int, int, int], y: int, theme: str, frames: int = 1, paintings: int = 1,
         rng: Optional[random.Random] = None, avoid: Iterable[Tuple[int, int]] = ()) -> int:
    """Item frames and one-block paintings on the walls of ``room`` (the
    inclusive interior rectangle) at layer ``y``: in empty blocks along its
    edge with a wall behind them, never two side by side. Returns how many."""
    rng = rng or b.rng
    x0, z0, x1, z1 = room
    avoid = set(avoid)
    spots: List[Tuple[int, int, str]] = []
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            here = b.get(x, y, z)
            if (x, z) in avoid or (here is not None and kind(here) != "air"):
                continue
            for facing, edge in (("south", z == z0), ("north", z == z1), ("east", x == x0), ("west", x == x1)):
                if edge and _wall_behind(b, x, y, z, facing):
                    spots.append((x, z, facing))
    rng.shuffle(spots)
    taken: List[Tuple[int, int]] = [tuple(e["pos"][::2]) for e in b.entities if e["pos"][1] == y]  # type: ignore
    placed = 0
    want = ["item_frame"] * frames + ["painting"] * paintings
    for x, z, facing in spots:
        if not want:
            break
        if any(abs(x - tx) + abs(z - tz) <= 1 for tx, tz in taken):
            continue
        what = want.pop(0)
        if what == "item_frame":
            b.entity("item_frame", x, y, z, facing, item=rng.choice(_FRAMED.get(theme, _FRAMED["house"])))
        else:
            b.entity("painting", x, y, z, facing, variant=rng.choice(PAINTINGS_1X1))
        taken.append((x, z))
        placed += 1
    return placed


__all__ = ["fill_containers", "pattern_banners", "hang"]
