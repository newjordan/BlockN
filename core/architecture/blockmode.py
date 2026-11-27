"""Block mode: architects that place Minecraft blocks directly on the 1 m grid.

Mesh mode builds exact geometry and voxelizes it, which at Minecraft's 1 m
block size aliases: a 40 degree roof becomes a ragged staircase, a 0.3 m sill
vanishes or doubles. Block mode designs to the grid instead. Walls, roof
pitches, stairs and doors are whole blocks, slopes use stair and slab blocks,
and every block carries its Minecraft block state, so what is exported is what
was designed and nothing has to be guessed back from a mesh.

A ``BlockBuild`` is a sparse map from integer block positions to block states
(``"minecraft:oak_stairs[facing=east,half=bottom]"``). +X is east, +Z is south
and +Y is up, as in Minecraft and in BlockN. Block (x, y, z) fills the unit
cube from (x, y, z) to (x + 1, y + 1, z + 1); y = 0 is the first layer above
the ground. A separate ground layer (y = -1) holds paths, fields and water.

What comes out:

* ``to_mcfunction()``: setblock and fill commands with full block states, in an
  order Minecraft accepts (solid blocks first, attached blocks last).
* ``grid(sub=1)``: a ``blockn.voxels/v1`` grid at 1 m, one cell per block, its
  palette holding each block state.
* ``grid(sub=4)``: a 0.25 m collision grid in which every block has its true
  shape (a bottom slab is its lower half, a stair its lower half plus the back
  of the upper half, a fence a post and rails 1.5 m tall, a door or a flower
  nothing). The player profile checks (``gameplay.walk``) run on this grid
  unchanged.
* ``grid(sub=4, visual=True)``: the same at 0.25 m with decoration drawn
  (lanterns, panes, doors, plants), for ``render.render``.

Doorways are data: every door an architect places is recorded as a port with
its world position, facing, clear width and height and the profile it was
sized for, in the same form the walk reports measured ports.

Sizes come from the player profile in whole blocks (``BlockSizes``). Block
mode keeps Minecraft's own rule for openings: the smallest whole-block opening
the body fits through, so a 1.8 m player takes a 2-block door, and a 2.6 m
giant a 3 by 2 one. Stair blocks give 0.5 m rises; a body whose step is below
0.5 m climbs them only by jumping, which the walk reports as such.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .voxels import AIR, CHUNK, VOXEL_SCHEMA, VoxelGrid

BLOCKS_SCHEMA = "blockn.blocks/v1"
SUB = 4

Pos = Tuple[int, int, int]

# Direction names and their unit steps in (x, z).
DIRS: Dict[str, Tuple[int, int]] = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
LEFT = {"north": "west", "west": "south", "south": "east", "east": "north"}
RIGHT = {v: k for k, v in LEFT.items()}


def facing_of(dx: int, dz: int) -> str:
    for name, step in DIRS.items():
        if step == (int(np.sign(dx)), int(np.sign(dz))):
            return name
    raise ValueError("facing needs a step along one axis")


def state(block: str, **props: Any) -> str:
    """A block state string: ``state("oak_stairs", facing="east")``."""
    name = block if ":" in block else f"minecraft:{block}"
    if not props:
        return name
    inner = ",".join(f"{k}={str(v).lower()}" for k, v in sorted(props.items()))
    return f"{name}[{inner}]"


def parse(block: str) -> Tuple[str, Dict[str, str]]:
    """``"minecraft:oak_stairs[facing=east]"`` -> ``("oak_stairs", {"facing": "east"})``."""
    name, _, rest = block.partition("[")
    name = name.split(":", 1)[-1]
    props: Dict[str, str] = {}
    if rest:
        for part in rest.rstrip("]").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                props[k.strip()] = v.strip()
    return name, props


# ------------------------------------------------------------------ colours
# Average texture colours, for renders. Keys are material names; block ids map
# to them through ``material``.
_RGB: Dict[str, Tuple[int, int, int]] = {
    "stone_bricks": (122, 121, 122), "mossy_stone_bricks": (112, 122, 100), "cracked_stone_bricks": (116, 115, 116),
    "chiseled_stone_bricks": (120, 119, 120), "cobblestone": (127, 127, 127), "mossy_cobblestone": (108, 120, 92),
    "stone": (125, 125, 125), "smooth_stone": (159, 159, 159), "andesite": (136, 136, 137),
    "polished_andesite": (132, 135, 134), "deepslate_bricks": (70, 70, 72), "deepslate_tiles": (54, 54, 56),
    "polished_deepslate": (72, 72, 74), "cobbled_deepslate": (77, 77, 80), "tuff": (108, 109, 102),
    "oak_planks": (162, 131, 79), "spruce_planks": (115, 85, 49), "dark_oak_planks": (67, 43, 20),
    "birch_planks": (192, 175, 121), "oak_log": (109, 85, 51), "stripped_oak_log": (177, 144, 86),
    "spruce_log": (59, 38, 17), "stripped_spruce_log": (116, 90, 52), "dark_oak_log": (60, 47, 26),
    "stripped_dark_oak_log": (96, 76, 49), "white_terracotta": (210, 178, 161), "terracotta": (152, 94, 68),
    "bricks": (150, 97, 83), "white_concrete": (207, 213, 214), "calcite": (223, 224, 220),
    "glass_pane": (170, 205, 218), "glass": (175, 210, 220), "iron_bars": (105, 105, 108),
    "lantern": (232, 180, 80), "torch": (255, 205, 95), "wall_torch": (255, 205, 95), "chain": (60, 62, 72),
    "red_wool": (160, 39, 34), "blue_wool": (53, 57, 157), "yellow_wool": (240, 175, 21),
    "white_wool": (233, 236, 236), "red_banner": (160, 39, 34), "blue_banner": (53, 57, 157),
    "yellow_banner": (240, 175, 21), "red_wall_banner": (160, 39, 34), "blue_wall_banner": (53, 57, 157),
    "yellow_wall_banner": (240, 175, 21), "red_carpet": (160, 39, 34), "blue_carpet": (53, 57, 157),
    "oak_leaves": (64, 122, 42), "vine": (58, 104, 32), "spruce_leaves": (54, 92, 54), "birch_leaves": (96, 138, 62),
    "dirt_path": (148, 122, 65), "gravel": (131, 127, 126), "farmland": (88, 52, 22), "wheat": (186, 166, 70),
    "carrots": (90, 150, 40), "hay_block": (176, 146, 40), "water": (63, 118, 228), "grass_block": (98, 140, 60),
    "dirt": (134, 96, 67), "poppy": (200, 40, 30), "dandelion": (232, 210, 40), "cornflower": (70, 110, 230),
    "oxeye_daisy": (225, 225, 210), "allium": (170, 110, 210), "short_grass": (90, 140, 55),
    "tall_grass": (90, 140, 55), "fern": (80, 130, 55), "sweet_berry_bush": (70, 110, 50),
    "barrel": (120, 90, 55), "chest": (164, 116, 44), "crafting_table": (130, 100, 60), "furnace": (110, 110, 110),
    "smoker": (95, 85, 75), "bookshelf": (117, 94, 59), "red_bed": (160, 35, 35), "blue_bed": (53, 57, 157),
    "campfire": (130, 95, 50), "composter": (110, 80, 40), "flower_pot": (124, 68, 52),
    "potted_poppy": (150, 60, 40), "oak_door": (140, 110, 62), "spruce_door": (102, 78, 46),
    "dark_oak_door": (75, 50, 24), "spruce_trapdoor": (104, 79, 47), "oak_trapdoor": (140, 112, 64),
    "dark_oak_trapdoor": (75, 50, 24), "oak_pressure_plate": (162, 131, 79),
    "spruce_pressure_plate": (115, 85, 49), "anvil": (68, 68, 68), "cauldron": (70, 70, 72),
    "bell": (220, 180, 60), "grindstone": (140, 140, 140), "loom": (150, 120, 90), "lectern": (160, 125, 80),
    "cartography_table": (90, 70, 50), "smithing_table": (60, 50, 50), "blast_furnace": (100, 100, 100),
    "potatoes": (110, 150, 45), "beetroots": (95, 140, 50), "coarse_dirt": (119, 85, 59), "podzol": (91, 63, 24),
    "azure_bluet": (205, 215, 220), "pumpkin": (198, 118, 24), "melon": (111, 145, 30),
    "orange_wool": (240, 118, 19), "light_blue_wool": (58, 175, 217), "green_wool": (84, 109, 27),
    "cyan_wool": (21, 137, 145), "birch_log": (216, 215, 210), "stripped_birch_log": (196, 176, 118),
    "cobblestone_stairs": (127, 127, 127), "moss_block": (89, 109, 45),
    "candle": (226, 210, 170), "white_candle": (226, 226, 220), "red_candle": (160, 40, 34),
    "potted_cornflower": (124, 68, 52), "potted_red_tulip": (124, 68, 52), "potted_fern": (124, 68, 52),
    "potted_azure_bluet": (124, 68, 52), "potted_oxeye_daisy": (124, 68, 52), "potted_dandelion": (124, 68, 52),
    "white_carpet": (233, 236, 236), "light_gray_carpet": (142, 142, 135), "brown_carpet": (114, 72, 41),
    "cyan_carpet": (21, 137, 145), "green_carpet": (84, 109, 27), "yellow_carpet": (240, 175, 21),
    "light_blue_carpet": (58, 175, 217), "orange_carpet": (240, 118, 19), "purple_carpet": (122, 42, 172),
    "green_bed": (84, 109, 27), "yellow_bed": (240, 175, 21), "white_bed": (233, 236, 236),
    "cyan_bed": (21, 137, 145), "brown_bed": (114, 72, 41), "decorated_pot": (140, 80, 60),
    "stripped_spruce_wood": (116, 90, 52), "stripped_oak_wood": (177, 144, 86), "jukebox": (95, 65, 45),
    "azalea_leaves": (92, 118, 44), "flowering_azalea_leaves": (98, 120, 52),
    "mud_bricks": (137, 104, 79), "packed_mud": (142, 106, 79), "stripped_birch_wood": (196, 176, 118),
    "chiseled_bookshelf": (175, 140, 87), "brewing_stand": (110, 100, 80), "light_gray_wool": (142, 142, 135),
    "dark_oak_leaves": (52, 88, 30), "stripped_dark_oak_wood": (96, 76, 49),
    "cyan_terracotta": (87, 91, 91), "light_gray_terracotta": (135, 107, 98), "brown_terracotta": (77, 51, 36),
}


def material(block: str) -> str:
    """The material key of a block for colouring: stairs, slabs, walls and
    fences take their base block's colour."""
    name, _ = parse(block)
    if name in _RGB:
        return name
    base = name
    for suffix in ("_stairs", "_slab", "_wall", "_fence_gate", "_fence", "_button"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for key in (base, base + "s", base + "_planks", base.replace("brick", "bricks"), base.replace("tile", "tiles")):
        if key in _RGB:
            return key
    return base


def rgb(block: str) -> Tuple[int, int, int]:
    return _RGB.get(material(block), (165, 160, 150))


class _AnyWall:
    """Every full block counts as wall that furniture may stand against,
    except furniture itself."""

    _FURNITURE = ("barrel", "chest", "crafting_table", "furnace", "smoker", "bookshelf", "cauldron", "lectern",
                  "cartography_table", "blast_furnace", "loom", "grindstone", "anvil", "jukebox",
                  "chiseled_bookshelf", "smithing_table", "hay_block", "composter", "brewing_stand",
                  "decorated_pot", "white_wool", "light_gray_wool")

    def __contains__(self, block: str) -> bool:
        return parse(block)[0] not in self._FURNITURE


# ------------------------------------------------------------------ detail
_BOOKS = ((150, 40, 36), (52, 70, 150), (60, 112, 52), (196, 160, 90), (110, 60, 120), (170, 120, 50))
_FLOWER = {"cornflower": (70, 110, 230), "red_tulip": (205, 50, 40), "fern": (70, 125, 50),
           "azure_bluet": (215, 222, 226), "oxeye_daisy": (232, 232, 214), "dandelion": (235, 212, 40),
           "poppy": (200, 40, 30)}


def detail(block: str, x: int, y: int, z: int) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Surface colours that make furniture read in a picture: (mask, rgb)
    pairs over a SUB cube, painted over the block's own colour. Pictures
    only; collision and export never see them."""
    name, props = parse(block)
    n = SUB
    out: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

    def shell() -> np.ndarray:
        m = np.zeros((n, n, n), dtype=bool)
        m[[0, -1], :, :] = True
        m[:, :, [0, -1]] = True
        return m

    if name == "bookshelf":
        sh = shell()
        for row in (1, 2):
            for i in range(n):
                col = _BOOKS[(x * 7 + z * 5 + y * 3 + row * 11 + i * 13) % len(_BOOKS)]
                m = np.zeros((n, n, n), dtype=bool)
                m[i, row, :] = True
                m[:, row, i] = True
                out.append((m & sh, col))
    elif name == "chiseled_bookshelf":
        f = props.get("facing", "north")
        for i in range(6):
            if props.get(f"slot_{i}_occupied") != "true":
                continue
            row, colm = (2 if i < 3 else 1), i % 3
            m = np.zeros((n, n, n), dtype=bool)
            a = slice(colm + (colm > 0), colm + 1 + (colm > 0)) if n == 4 else slice(colm, colm + 1)
            if f == "north":
                m[a, row, 0] = True
            elif f == "south":
                m[a, row, n - 1] = True
            elif f == "west":
                m[0, row, a] = True
            else:
                m[n - 1, row, a] = True
            out.append((m, _BOOKS[(x * 7 + z * 5 + y * 3 + i * 13) % len(_BOOKS)]))
    elif name == "hay_block":
        m = shell()
        m[:, [0, 2, 3], :] = False
        out.append((m, (120, 60, 30)))
    elif name == "decorated_pot":
        m = shell()
        m[:, [0, 3], :] = False
        out.append((m, (190, 120, 80)))
    elif name in ("loom", "smithing_table", "cartography_table", "composter"):
        m = np.zeros((n, n, n), dtype=bool)
        m[:, n - 1, :] = True
        col = {"loom": (205, 190, 160), "smithing_table": (40, 36, 46), "cartography_table": (215, 205, 175),
               "composter": (70, 95, 35)}[name]
        out.append((m, col))
    elif name == "brewing_stand":
        m = np.zeros((n, n, n), dtype=bool)
        m[1:3, 1:, 1:3] = True
        out.append((m, (230, 140, 60)))
    elif name == "chest":
        m = shell()
        m[:, :2, :] = False
        m[:, 3:, :] = False
        out.append((m, (110, 72, 26)))
        f = props.get("facing", "north")
        sx, sz = {"north": (slice(1, 3), 0), "south": (slice(1, 3), n - 1),
                  "west": (0, slice(1, 3)), "east": (n - 1, slice(1, 3))}[f]
        m2 = np.zeros((n, n, n), dtype=bool)
        m2[sx, 2, sz] = True
        out.append((m2, (190, 190, 196)))
    elif name == "barrel":
        m = shell()
        m[:, [0, 2], :] = False
        out.append((m, (82, 60, 36)))
        top = np.zeros((n, n, n), dtype=bool)
        top[1:3, n - 1, 1:3] = True
        out.append((top, (92, 66, 38)))
    elif name in ("furnace", "smoker", "blast_furnace"):
        f = props.get("facing", "north")
        m = np.zeros((n, n, n), dtype=bool)
        if f == "north":
            m[1:3, 0:2, 0] = True
        elif f == "south":
            m[1:3, 0:2, n - 1] = True
        elif f == "west":
            m[0, 0:2, 1:3] = True
        else:
            m[n - 1, 0:2, 1:3] = True
        out.append((m, (40, 36, 34) if props.get("lit") != "true" else (240, 140, 50)))
    elif name == "crafting_table":
        m = np.zeros((n, n, n), dtype=bool)
        m[[0, 2], n - 1, :] = True
        m[:, n - 1, [0, 2]] = True
        out.append((m, (96, 64, 36)))
        tools = shell()
        tools[:, :1, :] = False
        tools[:, 3:, :] = False
        out.append((tools & (np.indices((n, n, n))[0] % 2 == 1), (150, 150, 150)))
    elif name.startswith("potted_"):
        col = _FLOWER.get(name[len("potted_"):], (70, 125, 50))
        m = np.zeros((n, n, n), dtype=bool)
        m[1:3, 2:4, 1:3] = True
        out.append((m, col))
    elif name.endswith("candle"):
        if props.get("lit") == "true":
            m = np.zeros((n, n, n), dtype=bool)
            for (cx, cz) in ((1, 1), (2, 2), (2, 1), (1, 2))[:int(props.get("candles", "1"))]:
                m[cx, 1, cz] = True
            out.append((m, (255, 200, 90)))
    elif name.endswith("lantern"):
        m = np.zeros((n, n, n), dtype=bool)
        if props.get("hanging", "false") == "true":
            m[1, 2:, 1] = True
        else:
            m[1, 1, 1] = True
        out.append((m, (60, 62, 72)))
    elif name == "flowering_azalea_leaves":
        m = (np.indices((n, n, n)).sum(axis=0) % 3 == 0)
        out.append((m, (200, 120, 190)))
    elif name in ("campfire", "soul_campfire"):
        m = np.zeros((n, n, n), dtype=bool)
        m[1:3, 1, 1:3] = True
        out.append((m, (250, 150, 40)))
        if props.get("lit", "true") == "true":
            f = np.zeros((n, n, n), dtype=bool)
            f[1:3, 2, 1:3] = True
            f[1, 3, 2] = True
            out.append((f, (255, 196, 70)))
    elif name.endswith("_bed"):
        if props.get("part") == "head":
            m = np.zeros((n, n, n), dtype=bool)
            m[:, 1, :] = True
            f = props.get("facing", "south")
            # The pillow, at the head end.
            if f == "south":
                m[:, :, :n - 2] = False
            elif f == "north":
                m[:, :, 2:] = False
            elif f == "east":
                m[:n - 2, :, :] = False
            else:
                m[2:, :, :] = False
            out.append((m, (236, 236, 230)))
    return out


# ------------------------------------------------------------------- shapes
_PLANTS = ("poppy", "dandelion", "cornflower", "allium", "oxeye_daisy", "azure_bluet", "short_grass",
           "tall_grass", "fern", "wheat", "carrots", "potatoes", "beetroots", "sweet_berry_bush", "lilac",
           "rose_bush", "peony", "sugar_cane")
_THIN = ("chain",)


def kind(block: str) -> str:
    """What shape family a block belongs to."""
    name, _ = parse(block)
    if name in ("air", "cave_air", "void_air"):
        return "air"
    if name.endswith("_stairs"):
        return "stairs"
    if name.endswith("_slab"):
        return "slab"
    if name.endswith("_wall_banner"):
        return "wall_banner"
    if name.endswith("_banner"):
        return "banner"
    if name.endswith("_fence_gate"):
        return "gate"
    if name.endswith("_fence"):
        return "fence"
    if name.endswith("_wall") and not name.endswith("wall_torch"):
        return "wall"
    if name.endswith("glass_pane") or name == "iron_bars":
        return "pane"
    if name.endswith("_trapdoor"):
        return "trapdoor"
    if name.endswith("_door"):
        return "door"
    if name.endswith("lantern"):
        return "lantern"
    if name.endswith("torch"):
        return "torch"
    if name.endswith("_carpet"):
        return "carpet"
    if name.endswith("_pressure_plate"):
        return "plate"
    if name.endswith("_bed"):
        return "bed"
    if name in ("campfire", "soul_campfire"):
        return "campfire"
    if name == "water":
        return "water"
    if name in _THIN:
        return "chain"
    if name == "flower_pot" or name.startswith("potted_") or name == "candle" or name.endswith("_candle"):
        return "pot"
    if name in _PLANTS:
        return "plant"
    if name in ("vine", "glow_lichen", "ladder"):
        return "vine"                     # a thin climbable layer on a wall face
    if name in ("rail", "powered_rail", "detector_rail", "activator_rail"):
        return "carpet"                   # a rail lies flat on the floor; a body walks over it
    return "full"


# Families that connect fences, walls and panes, and that they stand against.
_SOLID_KINDS = {"full", "slab", "stairs"}
_CONNECT = {"fence": {"fence", "gate"}, "wall": {"wall"}, "pane": {"pane", "wall"}}


def _half(axis_dir: str) -> Tuple[slice, slice]:
    """The (x, z) slices of the half of a block on the ``axis_dir`` side."""
    h = SUB // 2
    return {
        "east": (slice(h, SUB), slice(0, SUB)),
        "west": (slice(0, h), slice(0, SUB)),
        "south": (slice(0, SUB), slice(h, SUB)),
        "north": (slice(0, SUB), slice(0, h)),
    }[axis_dir]


def _edge(axis_dir: str) -> Tuple[slice, slice]:
    """The (x, z) slices of the one-cell plate along the ``axis_dir`` edge."""
    return {
        "east": (slice(SUB - 1, SUB), slice(0, SUB)),
        "west": (slice(0, 1), slice(0, SUB)),
        "south": (slice(0, SUB), slice(SUB - 1, SUB)),
        "north": (slice(0, SUB), slice(0, 1)),
    }[axis_dir]


_SHAPES: Dict[Tuple[str, frozenset, bool], np.ndarray] = {}


def shape(block: str, connections: Iterable[str] = (), visual: bool = False) -> np.ndarray:
    """A SUB x SUB x SUB mask, indexed [x, y, z], of the cells a block fills.

    ``visual`` draws decoration (lanterns, plants, doors, carpets) that the
    player does not collide with; without it the mask is collision only.
    """
    conn = frozenset(connections)
    key = (block, conn, visual)
    if key in _SHAPES:
        return _SHAPES[key]
    k = kind(block)
    _, props = parse(block)
    m = np.zeros((SUB, SUB, SUB), dtype=bool)
    h = SUB // 2
    mid = slice(1, SUB - 1)
    if k == "full":
        m[:] = True
    elif k == "slab":
        t = props.get("type", "bottom")
        if t == "double":
            m[:] = True
        elif t == "top":
            m[:, h:, :] = True
        else:
            m[:, :h, :] = True
    elif k == "stairs":
        top = props.get("half", "bottom") == "top"
        sx, sz = _half(props.get("facing", "north"))
        if top:
            m[:, h:, :] = True
            m[sx, :h, sz] = True
        else:
            m[:, :h, :] = True
            m[sx, h:, sz] = True
    elif k in ("fence", "wall"):
        m[mid, :, mid] = True
        rails = range(SUB) if (k == "wall" or not visual) else (1, 3)
        for d in conn:
            sx, sz = _half(d)
            for j in rails:
                arm = np.zeros((SUB, SUB), dtype=bool)
                arm[sx, sz] = True
                band = np.zeros((SUB, SUB), dtype=bool)
                if d in ("east", "west"):
                    band[:, mid] = True
                else:
                    band[mid, :] = True
                m[:, j, :] |= arm & band
    elif k == "pane":
        dirs = conn or {"north", "south", "east", "west"}
        c = 1
        m[c, :, c] = True
        for d in dirs:
            if d == "east":
                m[c:, :, c] = True
            elif d == "west":
                m[: c + 1, :, c] = True
            elif d == "south":
                m[c, :, c:] = True
            else:
                m[c, :, : c + 1] = True
    elif k == "bed":
        m[:, :h, :] = True
    elif k == "campfire":
        m[:, :h, :] = True
    elif k == "trapdoor":
        if props.get("open", "false") == "true":
            sx, sz = _edge(OPPOSITE[props.get("facing", "north")])
            m[sx, :, sz] = True
        elif props.get("half", "bottom") == "top":
            m[:, SUB - 1, :] = True
        else:
            m[:, 0, :] = True
    elif visual:
        if k == "door":
            # A closed door is a plate on the side away from its facing.
            sx, sz = _edge(OPPOSITE[props.get("facing", "north")])
            m[sx, :, sz] = True
        elif k == "lantern":
            # A squat glass body with an iron top (painted by ``detail``),
            # hung from a chain or standing.
            if props.get("hanging", "false") == "true":
                m[mid, 1, mid] = True
                m[1, 2:, 1] = True
            else:
                m[mid, 0, mid] = True
                m[1, 1, 1] = True
        elif k in ("torch", "chain"):
            m[1, :3, 1] = True
        elif k == "wall_banner":
            sx, sz = _edge(OPPOSITE[props.get("facing", "north")])
            m[sx, :, sz] = True
        elif k == "banner":
            m[1, :, 1] = True
            m[mid, 1:, 1] = True
        elif k in ("carpet", "plate"):
            m[:, 0, :] = True
        elif k == "vine":
            # A sheet on each face it clings to; nothing to collide with.
            for d in ("north", "south", "east", "west"):
                if props.get(d) == "true":
                    sx, sz = _edge(d)
                    m[sx, :, sz] = True
        elif k == "plant":
            for i in range(SUB):
                m[i, :3, i] = True
                m[i, :3, SUB - 1 - i] = True
        elif k == "pot":
            if parse(block)[0].endswith("candle"):
                # Thin candles, one to four, a quarter block tall.
                for (cx, cz) in ((1, 1), (2, 2), (2, 1), (1, 2))[:int(props.get("candles", "1"))]:
                    m[cx, 0, cz] = True
            else:
                m[mid, :2, mid] = True
        elif k == "water":
            m[:, :SUB - 1, :] = True
        elif k == "gate":
            m[0, :, mid] = True
            m[SUB - 1, :, mid] = True
    _SHAPES[key] = m
    return m


# -------------------------------------------------------------------- sizes
@dataclass(frozen=True)
class BlockSizes:
    """A player profile in whole blocks."""

    door_width: int      # clear width of a doorway
    door_height: int     # clear height of a doorway
    ceiling: int         # clear height of a room
    storey: int          # floor to floor: the ceiling plus one block of floor
    lane: int            # clear width of a walk, stair or passage
    headroom: int        # clear blocks over a stair the body needs
    climbs_stairs: bool  # whether stair blocks (0.5 m rises) are walked, not jumped

    @classmethod
    def of(cls, player: Any) -> "BlockSizes":
        from .profile import resolve

        p = resolve(player)
        if p is None:
            p = resolve("blocky")
        door_w = max(1, int(math.ceil(2 * p.radius + 0.1 - 1e-9)))
        door_h = max(2, int(math.ceil(p.height + 0.05 - 1e-9)))
        # Two blocks over the door, and a block over the body's own clearance:
        # rooms a door's height plus one read as cramped from a player's eye.
        ceiling = max(door_h + 2, int(math.ceil(p.clearances().ceiling_height - 1e-9)) + 1)
        return cls(door_w, door_h, ceiling, ceiling + 1, door_w, door_h, p.step_height >= 0.5 - 1e-9)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------- entities
# Java Edition 1.21 data: an item frame's Facing is a direction id (0 down,
# 1 up, 2 north, 3 south, 4 west, 5 east); a painting's is horizontal
# (0 south, 1 west, 2 north, 3 east). Items carry ``count``, as since 1.20.5.
_FACING_3D = {"north": 2, "south": 3, "west": 4, "east": 5}
_FACING_2D = {"south": 0, "west": 1, "north": 2, "east": 3}
# One-block paintings only, so a painting fills exactly the block it hangs in.
PAINTINGS_1X1 = ("kebab", "aztec", "alban", "aztec2", "bomb", "plant", "wasteland", "meditative")


def snbt_items(items: Sequence[Tuple[str, int]]) -> str:
    """``[{Slot:0b,id:"minecraft:bread",count:3},...]`` for a container."""
    parts = []
    for slot, (item, count) in enumerate(items):
        iid = item if ":" in item else f"minecraft:{item}"
        parts.append(f'{{Slot:{slot}b,id:"{iid}",count:{int(count)}}}')
    return "[" + ",".join(parts) + "]"


def entity_command(e: Mapping[str, Any]) -> str:
    """The ``summon`` command for one entity, relative to the build's origin."""
    x, y, z = e["pos"]
    eid = e["id"]
    name = eid.split(":")[-1]
    if name in ("item_frame", "glow_item_frame"):
        item = e.get("item", "minecraft:air")
        item = item if ":" in item else f"minecraft:{item}"
        return (f"summon {eid} ~{x + .5} ~{y + .5} ~{z + .5} "
                f'{{Facing:{_FACING_3D[e["facing"]]}b,Fixed:1b,Item:{{id:"{item}",count:1}}}}')
    if name == "painting":
        v = e.get("variant", "kebab")
        return (f"summon {eid} ~{x} ~{y} ~{z} "
                f'{{facing:{_FACING_2D[e["facing"]]}b,variant:"minecraft:{v}"}}')
    if name == "chest_minecart":
        return f"summon {eid} ~{x + .5} ~{y} ~{z + .5} {{Items:{snbt_items(e.get('items', []))}}}"
    return f"summon {eid} ~{x + .5} ~{y} ~{z + .5}"


# Colours an entity is drawn in, in pictures: a frame's wood round its item,
# a painting's ground and figure.
_ENTITY_RGB = {"item_frame": (139, 99, 59), "glow_item_frame": (120, 150, 120), "painting": (96, 70, 44)}
_PAINT_RGB = {"kebab": ((180, 92, 46), (232, 196, 116)), "aztec": ((150, 60, 40), (210, 170, 90)),
              "alban": ((70, 110, 150), (220, 210, 170)), "aztec2": ((110, 70, 40), (190, 120, 60)),
              "bomb": ((90, 120, 70), (60, 60, 60)), "plant": ((90, 130, 70), (200, 180, 120)),
              "wasteland": ((120, 150, 170), (100, 80, 60)), "meditative": ((150, 120, 170), (230, 220, 200))}
_ITEM_RGB = {"iron": (205, 205, 205), "gold": (240, 200, 60), "diamond": (90, 220, 210), "bread": (190, 140, 60),
             "map": (220, 210, 170), "wheat": (210, 180, 80), "shears": (200, 200, 200), "potion": (190, 60, 150),
             "book": (120, 70, 40), "wool": (235, 235, 235), "pickaxe": (150, 150, 150), "sword": (205, 205, 205)}


def _item_rgb(item: str) -> Tuple[int, int, int]:
    name = item.split(":")[-1]
    for key, col in _ITEM_RGB.items():
        if key in name:
            return col
    return (170, 150, 120)


# -------------------------------------------------------------------- build
_ATTACHED_LAST = ("door", "trapdoor", "lantern", "torch", "wall_banner", "banner", "carpet", "plate", "plant",
                  "vine", "pot", "bed", "chain")


class BlockBuild:
    """A build made of Minecraft blocks on the 1 m grid."""

    def __init__(self, product: str, player: Any = "blocky", seed: int = 0):
        from .profile import resolve

        self.product = product
        self.player = resolve(player) or resolve("blocky")
        self.sizes = BlockSizes.of(self.player)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.blocks: Dict[Pos, str] = {}
        self.ground: Dict[Tuple[int, int], str] = {}
        self.ports: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        # Block entity data (a chest's items, a banner's patterns) as SNBT,
        # by block; and entities (item frames, paintings, minecarts).
        self.nbt: Dict[Pos, str] = {}
        self.entities: List[Dict[str, Any]] = []

    # ------------------------------------------------------------ placing
    @property
    def lining_blocks(self) -> "_AnyWall":
        """Blocks that furniture treats as a wall to stand against."""
        return _AnyWall()

    def set(self, x: int, y: int, z: int, block: Optional[str], replace: bool = True) -> None:
        pos = (int(x), int(y), int(z))
        if block is None or parse(block)[0] == "air":
            self.blocks.pop(pos, None)
            self.nbt.pop(pos, None)
            return
        if not replace and pos in self.blocks:
            return
        self.blocks[pos] = block if ":" in block else f"minecraft:{block}"
        self.nbt.pop(pos, None)

    def entity(self, eid: str, x: int, y: int, z: int, facing: str = "north", **data: Any) -> Dict[str, Any]:
        """An entity in block (x, y, z): an item frame or a painting hung on
        the wall behind it (``facing`` is the way it faces, out of that
        wall), a minecart on a rail. ``data`` is kept for the export (an
        item frame's ``item``, a painting's ``variant``, a chest cart's
        ``items``)."""
        e = {"id": eid if ":" in eid else f"minecraft:{eid}", "pos": [int(x), int(y), int(z)], "facing": facing}
        e.update(data)
        self.entities.append(e)
        return e

    def get(self, x: int, y: int, z: int) -> Optional[str]:
        return self.blocks.get((int(x), int(y), int(z)))

    def fill(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int, block: Any, replace: bool = True) -> None:
        """Fill the box between two corners, inclusive. ``block`` may be a
        callable returning a block per cell (for mixed materials)."""
        for x in range(min(x0, x1), max(x0, x1) + 1):
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for z in range(min(z0, z1), max(z0, z1) + 1):
                    self.set(x, y, z, block() if callable(block) else block, replace)

    def clear(self, x0: int, y0: int, z0: int, x1: int, y1: int, z1: int) -> None:
        self.fill(x0, y0, z0, x1, y1, z1, None)

    def ring(self, x0: int, z0: int, x1: int, z1: int, y0: int, y1: int, block: Any) -> None:
        """The perimeter walls of a rectangle, inclusive."""
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                for z in (z0, z1):
                    self.set(x, y, z, block() if callable(block) else block)
            for z in range(z0 + 1, z1):
                for x in (x0, x1):
                    self.set(x, y, z, block() if callable(block) else block)

    def mix(self, *choices: Tuple[str, float]) -> Callable[[], str]:
        """A picker that returns one of the blocks at random, by weight, from
        the build's own seeded generator."""
        names = [state(c[0]) if ":" not in c[0] and "[" not in c[0] else c[0] for c in choices]
        weights = [c[1] for c in choices]
        return lambda: self.rng.choices(names, weights)[0]

    def path(self, x: int, z: int, block: str = "dirt_path") -> None:
        self.ground[(int(x), int(z))] = state(block) if ":" not in block else block

    # ------------------------------------------------------- composites
    def door(self, x: int, y: int, z: int, facing: str, wood: str = "spruce", width: Optional[int] = None,
             height: Optional[int] = None, along: Optional[str] = None, name: str = "door") -> Dict[str, Any]:
        """A doorway sized to the player, its lowest block at (x, y, z).

        ``facing`` is the way a person walking in faces. The opening is
        ``width`` blocks wide, running toward ``along`` (default: to the
        right of ``facing``), and ``height`` blocks tall. Its bottom two blocks
        of each column hold a door leaf; anything taller stays open above.
        Returns the port, which is also recorded.
        """
        width = width or self.sizes.door_width
        height = height or self.sizes.door_height
        along = along or RIGHT[facing]
        ax, az = DIRS[along]
        cols = [(x + i * ax, z + i * az) for i in range(width)]
        for i, (cx, cz) in enumerate(cols):
            for j in range(height):
                self.set(cx, y + j, cz, None)
            if wood:
                hinge = "left" if (width == 1 or i == 0) else "right"
                if width == 2 and i == 0:
                    hinge = "left"
                self.set(cx, y, cz, state(f"{wood}_door", facing=facing, half="lower", hinge=hinge, open="false"))
                self.set(cx, y + 1, cz, state(f"{wood}_door", facing=facing, half="upper", hinge=hinge,
                                              open="false"))
        cx = (cols[0][0] + cols[-1][0]) / 2.0 + 0.5
        cz = (cols[0][1] + cols[-1][1]) / 2.0 + 0.5
        fx, fz = DIRS[OPPOSITE[facing]]
        port = {
            "id": f"port.{name}.{len(self.ports):03d}",
            "source": "declared",
            "position": [round(cx, 4), float(y), round(cz, 4)],
            # Ports face out of the building, as measured ports do.
            "facing": [float(fx), 0.0, float(fz)],
            "wall_axis": "x" if along in ("east", "west") else "z",
            "width": float(width),
            "height": float(height),
            "blocks": [width, height],
            "profile": self.player.name,
        }
        self.ports.append(port)
        return port

    def stair_flight(self, x: int, y: int, z: int, facing: str, rise: int, block: str, width: int = 1,
                     along: Optional[str] = None, support: Optional[str] = None) -> Pos:
        """A straight flight climbing ``rise`` blocks toward ``facing``.

        Its first stair is at (x, y, z) with its tread top at y + 1; each next
        stair is one block on and one up, so the last tread is level with
        y + rise. ``support`` fills under the flight. Returns the block just
        past the top step, where the landing is.
        """
        along = along or RIGHT[facing]
        ax, az = DIRS[along]
        dx, dz = DIRS[facing]
        for i in range(rise):
            for w in range(width):
                sx, sz = x + i * dx + w * ax, z + i * dz + w * az
                self.set(sx, y + i, sz, state(block, facing=facing, half="bottom"))
                if support:
                    for yy in range(y, y + i):
                        self.set(sx, yy, sz, support)
        return (x + rise * dx, y + rise, z + rise * dz)

    def gable_roof(self, x0: int, z0: int, x1: int, z1: int, y: int, stairs: str, ridge: str, gable: Any,
                   overhang: int = 1, axis: Optional[str] = None) -> int:
        """A 45 degree gable roof over the rectangle (inclusive), its eaves at
        layer ``y``. The ridge runs along ``axis`` ("x" or "z", default the
        longer side). Gable ends are filled with ``gable``. Returns the ridge
        layer."""
        if axis is None:
            axis = "x" if (x1 - x0) >= (z1 - z0) else "z"
        if axis == "x":
            lo, hi, a0, a1 = z0, z1, x0, x1
        else:
            lo, hi, a0, a1 = x0, x1, z0, z1
        span = hi - lo + 1
        layers = (span + 1) // 2
        top = y
        for k in range(layers + 1):
            yy = y + k
            left, right = lo - overhang + k, hi + overhang - k
            if left > right:
                break
            top = yy
            for a in range(a0 - overhang, a1 + overhang + 1):
                end = a < a0 or a > a1
                if left == right:
                    pos = (a, yy, left) if axis == "x" else (left, yy, a)
                    self.set(*pos, state(ridge, type="bottom"))
                    continue
                if left + 1 == right:
                    for c in (left, right):
                        pos = (a, yy, c) if axis == "x" else (c, yy, a)
                        self.set(*pos, state(ridge, type="bottom"))
                    continue
                up_l = "south" if axis == "x" else "east"
                for c, f in ((left, up_l), (right, OPPOSITE[up_l])):
                    pos = (a, yy, c) if axis == "x" else (c, yy, a)
                    self.set(*pos, state(stairs, facing=f, half="bottom"))
                # Underside of the next course and the gable walls.
                if not end:
                    for c in range(left + 1, right):
                        if c < lo or c > hi:
                            continue
                        pos = (a, yy, c) if axis == "x" else (c, yy, a)
                        if a in (a0, a1):
                            self.set(*pos, gable() if callable(gable) else gable)
                        else:
                            self.blocks.pop(pos, None)
        return top

    def hip_roof(self, x0: int, z0: int, x1: int, z1: int, y: int, stairs: str, cap: str, overhang: int = 1,
                 solid: Optional[str] = None) -> int:
        """A 45 degree hipped (pyramid when square) roof, eaves at layer ``y``.
        ``solid`` fills it (a closed roof with no loft to stand in). Returns
        the top layer."""
        k = 0
        top = y
        while True:
            a0, b0, a1, b1 = x0 - overhang + k, z0 - overhang + k, x1 + overhang - k, z1 + overhang - k
            if a0 > a1 or b0 > b1:
                break
            yy = y + k
            top = yy
            if a1 - a0 <= 1 or b1 - b0 <= 1:
                self.fill(a0, yy, b0, a1, yy, b1, state(cap, type="bottom"))
                break
            if solid is not None:
                self.fill(a0 + 1, yy, b0 + 1, a1 - 1, yy, b1 - 1, state(solid))
            for x in range(a0, a1 + 1):
                self.set(x, yy, b0, state(stairs, facing="south", half="bottom"))
                self.set(x, yy, b1, state(stairs, facing="north", half="bottom"))
            for z in range(b0 + 1, b1):
                self.set(a0, yy, z, state(stairs, facing="east", half="bottom"))
                self.set(a1, yy, z, state(stairs, facing="west", half="bottom"))
            for x, z, sh in ((a0, b0, "outer_right"), (a1, b0, "outer_left"), (a0, b1, "outer_left"),
                             (a1, b1, "outer_right")):
                f = "south" if z == b0 else "north"
                self.set(x, yy, z, state(stairs, facing=f, half="bottom", shape=sh))
            k += 1
        return top

    def spire(self, cx: float, cz: float, radius: float, y: int, stairs: str, cap: str,
              finial: Optional[str] = "spruce_fence") -> int:
        """A spire over a round tower: like ``cone_roof`` but each ring is a
        layer taller than the one outside it, so the roof starts at the
        turret's pitch and narrows to a needle, topped with a finial (two
        fence posts and a lightning rod). Returns the top layer."""
        r = radius
        yy = y
        steep = 2
        while r >= 0.5:
            ring = disc(cx, cz, r) - disc(cx, cz, r - 1.0)
            inner = disc(cx, cz, r - 1.0)
            for layer in range(steep):
                for (x, z) in ring:
                    dx, dz = x + 0.5 - cx, z + 0.5 - cz
                    if abs(dx) >= abs(dz):
                        f = "west" if dx > 0 else "east"
                    else:
                        f = "north" if dz > 0 else "south"
                    self.set(x, yy, z, state(stairs, facing=f, half="bottom") if layer == steep - 1 else state(cap))
                for (x, z) in inner:
                    self.set(x, yy, z, state(cap))
                yy += 1
            r -= 1.0
            steep = min(steep + 1, 4)
            if not inner:
                break
        tx, tz = int(math.floor(cx)), int(math.floor(cz))
        self.set(tx, yy, tz, state(cap))
        if finial:
            self.set(tx, yy + 1, tz, state(finial))
            self.set(tx, yy + 2, tz, state(finial))
            self.set(tx, yy + 3, tz, state("lightning_rod", facing="up"))
            yy += 3
        return yy

    def cone_roof(self, cx: float, cz: float, radius: float, y: int, stairs: str, cap: str,
                  steep: int = 2) -> int:
        """A conical roof over a round tower, its eave ring at layer ``y``.
        Each ring is ``steep`` layers tall before stepping in one block, so the
        cone reads steep, like a castle turret's. Returns the top layer."""
        r = radius
        yy = y
        while r >= 0.5:
            ring = disc(cx, cz, r) - disc(cx, cz, r - 1.0)
            inner = disc(cx, cz, r - 1.0)
            for layer in range(steep):
                for (x, z) in ring:
                    dx, dz = x + 0.5 - cx, z + 0.5 - cz
                    if abs(dx) >= abs(dz):
                        f = "west" if dx > 0 else "east"
                    else:
                        f = "north" if dz > 0 else "south"
                    if layer < steep - 1:
                        self.set(x, yy, z, state(cap))
                    else:
                        self.set(x, yy, z, state(stairs, facing=f, half="bottom"))
                # Solid inside: a turret cone has no loft to stand in.
                for (x, z) in inner:
                    self.set(x, yy, z, state(cap))
                yy += 1
            r -= 1.0
            if not inner:
                break
        self.set(int(math.floor(cx)), yy, int(math.floor(cz)), state(cap))
        return yy

    # ----------------------------------------------------------- exports
    def bounds(self) -> Tuple[Pos, Pos]:
        pts = np.array(list(self.blocks)) if self.blocks else np.zeros((1, 3), dtype=int)
        lo, hi = pts.min(0), pts.max(0)
        if self.ground:
            g = np.array(list(self.ground))
            lo[0], lo[2] = min(lo[0], g[:, 0].min()), min(lo[2], g[:, 1].min())
            hi[0], hi[2] = max(hi[0], g[:, 0].max()), max(hi[2], g[:, 1].max())
        return tuple(int(v) for v in lo), tuple(int(v) for v in hi)  # type: ignore[return-value]

    def connections(self, pos: Pos) -> Set[str]:
        k = kind(self.blocks[pos])
        if k not in _CONNECT:
            return set()
        out = set()
        x, y, z = pos
        for name, (dx, dz) in DIRS.items():
            other = self.blocks.get((x + dx, y, z + dz))
            if other is None:
                continue
            ok = kind(other)
            if ok in _CONNECT[k] or ok == "full":
                out.add(name)
        return out

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for block in self.blocks.values():
            name = parse(block)[0]
            out[name] = out.get(name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_mcfunction(self, ground: bool = True) -> str:
        """Commands that place the build relative to where they run: ``~ ~ ~``
        is block (0, 0, 0), the first layer above the ground. Solid blocks go
        first and attached ones (doors, lanterns, banners, plants) last, so
        nothing pops off for want of support. Same-state runs along X become
        one ``fill``."""
        lines = [f"# {BLOCKS_SCHEMA} {self.product} seed {self.seed} for {self.player.name}; "
                 f"run standing at the spot that should be block (0, 0, 0), on the ground"]
        if ground:
            for (x, z), block in sorted(self.ground.items()):
                lines.append(f"setblock ~{x} ~-1 ~{z} {block}")
        first = {p: b for p, b in self.blocks.items() if kind(b) not in _ATTACHED_LAST}
        last = {p: b for p, b in self.blocks.items() if kind(b) in _ATTACHED_LAST}
        for group in (first, last):
            done: Set[Pos] = set()
            for pos in sorted(group, key=lambda p: (p[1], p[2], p[0], 0 if parse(group[p])[1].get("half") != "upper" else 1)):
                if pos in done:
                    continue
                block = group[pos]
                x, y, z = pos
                end = x
                if kind(block) not in ("door", "bed") and pos not in self.nbt:
                    while (group.get((end + 1, y, z)) == block and (end + 1, y, z) not in done
                           and (end + 1, y, z) not in self.nbt):
                        end += 1
                for xx in range(x, end + 1):
                    done.add((xx, y, z))
                if end == x:
                    lines.append(f"setblock ~{x} ~{y} ~{z} {block}{self.nbt.get(pos, '')}")
                else:
                    lines.append(f"fill ~{x} ~{y} ~{z} ~{end} ~{y} ~{z} {block}")
        # Entities last, once the walls they hang on stand.
        for e in self.entities:
            lines.append(entity_command(e))
        return "\n".join(lines) + "\n"

    def climbs(self, sub: int, origin: Sequence[int]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        """Ladders as climbs for ``gameplay.walk``: for each vertical run of
        ladder blocks, a box of grid cells over the ladder's column and the
        wall it hangs on, from the run's foot to a body above its head, so
        the foot and the wall top it leads to are joined."""
        ox, oy, oz = origin
        runs = []
        seen = set()
        for (x, y, z), block in sorted(self.blocks.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1])):
            name, props = parse(block)
            if name != "ladder" or (x, y, z) in seen:
                continue
            top = y
            while parse(self.blocks.get((x, top + 1, z), "air"))[0] == "ladder":
                top += 1
            for yy in range(y, top + 1):
                seen.add((x, yy, z))
            fx, fz = DIRS.get(props.get("facing", "north"), (0, -1))
            bx, bz = x - fx, z - fz                       # the wall behind the ladder
            x0, x1 = min(x, bx), max(x, bx) + 1
            z0, z1 = min(z, bz), max(z, bz) + 1
            lo = ((x0 - ox) * sub, (y - 1 - oy) * sub, (z0 - oz) * sub)   # a foot one step up counts
            hi = ((x1 - ox) * sub, (top + 2 - oy) * sub, (z1 - oz) * sub)
            runs.append((lo, hi))
        return runs

    def walk_sub(self) -> int:
        """Cells per block for the walk: SUB, or twice that when the body,
        rounded out to SUB cells, would not fit its own doorway."""
        k = max(0, int(math.ceil(self.player.radius * SUB - 0.5)))
        return SUB if (2 * k + 1) <= self.sizes.door_width * SUB else 2 * SUB

    def grid(self, sub: int = SUB, visual: bool = False, margin: int = 3, ground_layer: bool = False) -> VoxelGrid:
        """The build as a ``VoxelGrid``.

        ``sub=1`` gives one cell per block (1 m), with each block state in the
        palette. ``sub=SUB`` gives true block shapes at 1/SUB m; ``visual``
        adds decoration. ``ground_layer`` (visual only) adds the y = -1 layer
        of paths and fields below the ground, for pictures.
        """
        if sub not in (1, SUB, 2 * SUB):
            raise ValueError(f"sub must be 1, {SUB} or {2 * SUB}")
        (lx, ly, lz), (hx, hy, hz) = self.bounds()
        ox, oz = lx - margin, lz - margin
        # Layer 0 is the ground; a build that goes below it (a mine) takes
        # its lowest layer as the origin.
        oy = min(ly, -1 if (ground_layer and sub > 1) else 0)
        nbx = hx - ox + 1 + margin
        nby = max(hy, 0) + 1 + 2 - oy
        nbz = hz - oz + 1 + margin

        def pad(n: int) -> int:
            return int(math.ceil(n / CHUNK)) * CHUNK

        size = (pad(nbx * sub), pad(nby * sub), pad(nbz * sub))
        labels = np.zeros(size, dtype=np.uint16)
        palette: List[Dict[str, Any]] = []
        index: Dict[str, int] = {}

        def label_for(block: str, colour: Optional[Tuple[int, int, int]] = None) -> int:
            key = block if colour is None else f"{block}|{colour}"
            if key not in index:
                palette.append({"label": len(palette) + 1, "substance": material(block), "minecraft": block,
                                "rgb": list(colour or rgb(block))})
                index[key] = len(palette)
            return index[key]

        if sub == 1:
            for (x, y, z), block in self.blocks.items():
                if kind(block) != "air":
                    labels[x - ox, y - oy, z - oz] = label_for(block)
        else:
            r = sub // SUB
            for (x, y, z), block in self.blocks.items():
                m = shape(block, self.connections((x, y, z)), visual)
                if r > 1:
                    m = m.repeat(r, 0).repeat(r, 1).repeat(r, 2)
                if not m.any():
                    continue
                bx, by, bz = (x - ox) * sub, (y - oy) * sub, (z - oz) * sub
                view = labels[bx:bx + sub, by:by + sub, bz:bz + sub]
                view[m] = label_for(block)
                if visual:
                    for dm, col in detail(block, x, y, z):
                        if r > 1:
                            dm = dm.repeat(r, 0).repeat(r, 1).repeat(r, 2)
                        dm = dm & m if kind(block) not in ("pot", "campfire") else dm
                        if dm.any():
                            view[dm] = label_for(block, col)
                k = kind(block)
                if not visual and k in ("fence", "wall") and (x, y + 1, z) not in self.blocks:
                    # Fences and walls collide 1.5 blocks high.
                    above = labels[bx:bx + sub, by + sub:by + sub + sub // 2, bz:bz + sub]
                    above[m[:, : sub // 2, :]] = label_for(block)
            if visual:
                # Item frames and paintings: a plate a cell thick on the face
                # of the wall behind them, a frame round its item, a painting
                # in two colours.
                for e in self.entities:
                    name = e["id"].split(":")[-1]
                    if name not in _ENTITY_RGB:
                        continue
                    x, y, z = e["pos"]
                    if not (lx - margin <= x and lz - margin <= z):
                        continue
                    m = np.zeros((SUB, SUB, SUB), dtype=bool)
                    sx, sz = _edge(OPPOSITE[e["facing"]])
                    inner = np.zeros((SUB, SUB, SUB), dtype=bool)
                    if name == "painting":
                        m[sx, :, sz] = True
                        lo_, hi_ = 1, SUB - 1
                        fig = np.zeros((SUB, SUB, SUB), dtype=bool)
                        fig[sx, lo_:hi_, sz] = True
                        if e["facing"] in ("north", "south"):
                            fig[:lo_, :, :] = False
                            fig[hi_:, :, :] = False
                        else:
                            fig[:, :, :lo_] = False
                            fig[:, :, hi_:] = False
                        inner = fig
                        ground_c, fig_c = _PAINT_RGB.get(e.get("variant", "kebab"), ((150, 110, 70), (220, 200, 150)))
                        cols = (ground_c, fig_c)
                    else:
                        m[sx, 1:SUB - 1, sz] = True
                        if e["facing"] in ("north", "south"):
                            m[:1, :, :] = False
                            m[SUB - 1:, :, :] = False
                        else:
                            m[:, :, :1] = False
                            m[:, :, SUB - 1:] = False
                        inner = np.zeros_like(m)
                        c0 = 1 + (SUB - 2) // 2 - 1
                        inner[sx, c0:c0 + 2, sz] = True
                        if e["facing"] in ("north", "south"):
                            inner[:c0, :, :] = False
                            inner[c0 + 2:, :, :] = False
                        else:
                            inner[:, :, :c0] = False
                            inner[:, :, c0 + 2:] = False
                        cols = (_ENTITY_RGB[name], _item_rgb(e.get("item", "")))
                    if r > 1:
                        m = m.repeat(r, 0).repeat(r, 1).repeat(r, 2)
                        inner = inner.repeat(r, 0).repeat(r, 1).repeat(r, 2)
                    bx, by, bz = (x - ox) * sub, (y - oy) * sub, (z - oz) * sub
                    view = labels[bx:bx + sub, by:by + sub, bz:bz + sub]
                    if view.shape != m.shape:
                        continue
                    view[m & ~inner] = label_for(f"minecraft:{name}", cols[0])
                    view[inner] = label_for(f"minecraft:{name}", cols[1])
            if ground_layer and visual:
                for (x, z), block in self.ground.items():
                    bx, bz = (x - ox) * sub, (z - oz) * sub
                    m = shape(block, (), True)
                    if r > 1:
                        m = m.repeat(r, 0).repeat(r, 1).repeat(r, 2)
                    labels[bx:bx + sub, 0:sub, bz:bz + sub][m] = label_for(block)
        pitch = 1.0 / sub
        origin = (float(ox), float(oy), float(oz))
        source = {"product": self.product, "seed": self.seed, "mode": "blocks", "schema": BLOCKS_SCHEMA}
        return VoxelGrid(labels, pitch, origin, palette, [dict(p) for p in self.ports], self.player.to_dict(), source)

    def manifest(self) -> Dict[str, Any]:
        lo, hi = self.bounds()
        return {
            "schema": BLOCKS_SCHEMA,
            "product": self.product,
            "seed": self.seed,
            "profile": self.player.to_dict(),
            "sizes": self.sizes.to_dict(),
            "bounds": [list(lo), list(hi)],
            "block_count": len(self.blocks),
            "ground_count": len(self.ground),
            "blocks_used": self.counts(),
            "ports": self.ports,
            "notes": self.notes,
            "voxels_schema": VOXEL_SCHEMA,
        }


# ----------------------------------------------------------------- helpers
def disc(cx: float, cz: float, r: float) -> Set[Tuple[int, int]]:
    """Blocks whose centres lie within ``r`` of (cx, cz)."""
    out = set()
    if r <= 0:
        return out
    for x in range(int(math.floor(cx - r)) - 1, int(math.ceil(cx + r)) + 1):
        for z in range(int(math.floor(cz - r)) - 1, int(math.ceil(cz + r)) + 1):
            if (x + 0.5 - cx) ** 2 + (z + 0.5 - cz) ** 2 <= r * r:
                out.add((x, z))
    return out


def circle(cx: float, cz: float, r: float, thick: float = 1.0) -> Set[Tuple[int, int]]:
    """A ring of blocks ``thick`` wide whose outer edge is at ``r``."""
    return disc(cx, cz, r) - disc(cx, cz, r - thick)


__all__ = ["BLOCKS_SCHEMA", "SUB", "DIRS", "OPPOSITE", "LEFT", "RIGHT", "BlockBuild", "BlockSizes", "state",
           "parse", "kind", "shape", "material", "rgb", "disc", "circle", "facing_of"]
