"""Block-mode architects: village houses, a village street and a castle.

These design straight onto Minecraft's 1 m grid with ``blockmode.BlockBuild``.
Every size that a body has to pass (doors, stairs, wall-walks, room heights)
comes from the player profile in whole blocks (``BlockSizes``); everything
else is designed to read well in Minecraft: 45 degree stair roofs with
overhangs, timber frames on stone bases, mixed stone, windows with shutters,
lanterns, banners, furnished rooms and gardens.

Each architect builds in its own frame with the front facing north and is
turned into place by ``paste``, which turns block states with it.
"""

from __future__ import annotations

from collections import deque
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .blockmode import (DIRS, LEFT, OPPOSITE, RIGHT, BlockBuild, _AnyWall, circle, disc, kind, parse, state)

Cell = Tuple[int, int]

_TURN = {"north": "east", "east": "south", "south": "west", "west": "north"}


def _turn_state(block: str, quarter: int) -> str:
    """A block state turned ``quarter`` quarter-turns clockwise seen from above."""
    if quarter % 4 == 0:
        return block
    name, props = parse(block)
    if not props:
        return block
    for _ in range(quarter % 4):
        if "facing" in props and props["facing"] in _TURN:
            props["facing"] = _TURN[props["facing"]]
        if "axis" in props and props["axis"] in ("x", "z"):
            props["axis"] = "z" if props["axis"] == "x" else "x"
        sides = {d: props.get(d) for d in ("north", "east", "south", "west") if d in props}
        for d, v in sides.items():
            props[_TURN[d]] = v
    return state(name, **props)


def _turn_xz(x: float, z: float, quarter: int) -> Tuple[float, float]:
    for _ in range(quarter % 4):
        x, z = -z, x
    return x, z


def paste(target: BlockBuild, part: BlockBuild, dx: int, dz: int, quarter: int = 0, dy: int = 0) -> None:
    """Copy ``part`` into ``target``, turned ``quarter`` quarter-turns
    clockwise about block (0, 0) and moved by (dx, dy, dz). Ports and ground
    go with it."""
    for (x, y, z), block in part.blocks.items():
        tx, tz = _turn_xz(x, z, quarter)
        # Turning moves a block's min corner to another corner; keep the
        # block on the cell it covers.
        cx, cz = _turn_xz(x + 0.5, z + 0.5, quarter)
        to = (int(math.floor(cx)) + dx, y + dy, int(math.floor(cz)) + dz)
        target.set(*to, _turn_state(block, quarter))
        if (x, y, z) in part.nbt:
            target.nbt[to] = part.nbt[(x, y, z)]
    for e in part.entities:
        x, y, z = e["pos"]
        cx, cz = _turn_xz(x + 0.5, z + 0.5, quarter)
        f = e["facing"]
        for _ in range(quarter % 4):
            f = _TURN.get(f, f)
        ne = dict(e)
        ne["pos"] = [int(math.floor(cx)) + dx, y + dy, int(math.floor(cz)) + dz]
        ne["facing"] = f
        target.entities.append(ne)
    for (x, z), block in part.ground.items():
        cx, cz = _turn_xz(x + 0.5, z + 0.5, quarter)
        target.ground[(int(math.floor(cx)) + dx, int(math.floor(cz)) + dz)] = block
    for port in part.ports:
        p = dict(port)
        px, pz = _turn_xz(port["position"][0], port["position"][2], quarter)
        p["position"] = [round(px + dx, 4), port["position"][1] + dy, round(pz + dz, 4)]
        fx, fz = _turn_xz(port["facing"][0], port["facing"][2], quarter)
        p["facing"] = [float(round(fx)), 0.0, float(round(fz))]
        if quarter % 2:
            p["wall_axis"] = "z" if port["wall_axis"] == "x" else "x"
        p["id"] = f"{port['id']}@{len(target.ports):03d}"
        target.ports.append(p)
    target.notes.extend(part.notes)


# ------------------------------------------------------------------ styles
HOUSE_STYLES: Dict[str, Dict[str, Any]] = {
    "timber": {"base": (("cobblestone", 6), ("mossy_cobblestone", 2), ("andesite", 1)),
               "post": "dark_oak_log", "band": "stripped_dark_oak_log",
               "lower": (("calcite", 5), ("white_terracotta", 1)), "upper": (("calcite", 5), ("white_terracotta", 1)),
               "roof": "dark_oak", "gable": "dark_oak_planks", "door": "dark_oak", "floor": "spruce_planks",
               "panel": "spruce_planks", "shutter": "dark_oak_trapdoor", "chimney": (("bricks", 5), ("cobblestone", 1))},
    "stone": {"base": (("stone_bricks", 5), ("mossy_stone_bricks", 2), ("cracked_stone_bricks", 1)),
              "post": "spruce_log", "band": "stripped_spruce_log",
              "lower": (("cobblestone", 5), ("mossy_cobblestone", 2), ("andesite", 2)),
              "upper": (("spruce_planks", 1),), "roof": "spruce", "gable": "spruce_planks", "door": "spruce",
              "floor": "oak_planks", "panel": "spruce_planks", "shutter": "spruce_trapdoor", "chimney": (("cobblestone", 3), ("stone_bricks", 1))},
    "brick": {"base": (("stone_bricks", 3), ("andesite", 1)), "post": "stripped_oak_log", "band": "stripped_oak_log",
              "lower": (("bricks", 1),), "upper": (("bricks", 1),), "roof": "deepslate_tile",
              "gable": "bricks", "door": "oak", "floor": "dark_oak_planks", "panel": "oak_planks",
              "shutter": "spruce_trapdoor",
              "chimney": (("bricks", 1),)},
    "oak": {"base": (("cobblestone", 4), ("stone_bricks", 2), ("mossy_cobblestone", 1)),
            "post": "oak_log", "band": "stripped_oak_log", "lower": (("cobblestone", 3), ("andesite", 1)),
            "upper": (("oak_planks", 1),), "roof": "spruce", "gable": "oak_planks", "door": "oak",
            "floor": "spruce_planks", "panel": "oak_planks", "shutter": "oak_trapdoor", "chimney": (("cobblestone", 1),)},
}


def _roof_blocks(roof: str) -> Tuple[str, str, str]:
    """(stairs, slab, full block) for a roof material."""
    full = {"deepslate_tile": "deepslate_tiles", "brick": "bricks", "stone_brick": "stone_bricks", "granite": "granite"}.get(
        roof, f"{roof}_planks")
    return f"{roof}_stairs", f"{roof}_slab", full


# Styles for building types other than the plain house (towers, mills):
# house() takes them by name but never picks one at random.
EXTRA_STYLES: Dict[str, Dict[str, Any]] = {
    "tower": {"base": (("stone_bricks", 4), ("mossy_stone_bricks", 2), ("cracked_stone_bricks", 1)),
              "post": "deepslate", "band": "stripped_spruce_log",
              "lower": (("stone_bricks", 5), ("cobblestone", 2), ("mossy_stone_bricks", 2), ("andesite", 1)),
              "upper": (("stone_bricks", 4), ("cobblestone", 1), ("mossy_stone_bricks", 1)),
              "roof": "deepslate_tile", "gable": "stone_bricks", "door": "spruce", "floor": "spruce_planks",
              "panel": "spruce_planks", "shutter": "spruce_trapdoor",
              "chimney": (("stone_bricks", 3), ("cobblestone", 1))},
    "mill": {"base": (("cobblestone", 4), ("mossy_cobblestone", 2)), "post": "stripped_spruce_log",
             "band": "stripped_spruce_log", "lower": (("cobblestone", 3), ("stone", 1), ("andesite", 1)),
             "upper": (("spruce_planks", 3), ("stripped_spruce_wood", 1)), "roof": "spruce",
             "gable": "spruce_planks", "door": "spruce", "floor": "oak_planks", "panel": "oak_planks",
             "shutter": "spruce_trapdoor", "chimney": (("cobblestone", 1),)},
}


# ---------------------------------------------------------------- furniture
def _floor_cells(b: BlockBuild, x0: int, z0: int, x1: int, z1: int, y: int) -> Set[Cell]:
    """Interior cells (inclusive rectangle) whose standing layer ``y`` and
    the layer above are free of anything solid, over a floor (a stairwell's
    hole is not floor)."""
    out = set()
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if b.get(x, y - 1, z) is None:
                continue
            if all(b.get(x, yy, z) is None or kind(b.get(x, yy, z)) in ("door", "lantern", "carpet", "plant", "vine")
                   for yy in (y, y + 1)):
                out.add((x, z))
    return out


def _reach(cells: Set[Cell], start: Iterable[Cell]) -> Set[Cell]:
    seen = {c for c in start if c in cells}
    queue = deque(seen)
    while queue:
        x, z = queue.popleft()
        for dx, dz in DIRS.values():
            n = (x + dx, z + dz)
            if n in cells and n not in seen:
                seen.add(n)
                queue.append(n)
    return seen


def _reach_body(cells: Set[Cell], start: Iterable[Cell], lane: int) -> Set[Cell]:
    """Cells a body ``lane`` blocks wide can stand on, moving from ``start``
    through free cells: the union of every lane x lane square of free cells
    joined to a square that holds a start cell."""
    if lane <= 1:
        return _reach(cells, start)
    squares = {(x, z) for (x, z) in cells
               if all((x + i, z + j) in cells for i in range(lane) for j in range(lane))}
    starts = set(start)
    seen = {q for q in squares if any((q[0] + i, q[1] + j) in starts for i in range(lane) for j in range(lane))}
    queue = deque(seen)
    while queue:
        x, z = queue.popleft()
        for dx, dz in DIRS.values():
            n = (x + dx, z + dz)
            if n in squares and n not in seen:
                seen.add(n)
                queue.append(n)
    return {(x + i, z + j) for (x, z) in seen for i in range(lane) for j in range(lane)}


def _lane(b: BlockBuild, cells: Set[Cell], start: Cell, goal: Cell) -> Set[Cell]:
    """Cells of a shortest path between two cells, widened to the lane."""
    prev: Dict[Cell, Optional[Cell]] = {start: None}
    queue = deque([start])
    while queue:
        c = queue.popleft()
        if c == goal:
            break
        for dx, dz in DIRS.values():
            n = (c[0] + dx, c[1] + dz)
            if n in cells and n not in prev:
                prev[n] = c
                queue.append(n)
    if goal not in prev:
        return set()
    out, c = set(), goal
    while c is not None:
        for wx in range(b.sizes.lane):
            for wz in range(b.sizes.lane):
                out.add((c[0] + wx, c[1] + wz))
        c = prev[c]
    return out & cells


def _walled(b: BlockBuild, room: Tuple[int, int, int, int], c: Cell, y: int) -> bool:
    """Whether a cell is wall for furniture: outside the room, or a solid
    block (an outer wall, panelling) at the standing layer."""
    x0, z0, x1, z1 = room
    if not (x0 <= c[0] <= x1 and z0 <= c[1] <= z1):
        return True
    blk = b.get(c[0], y, c[1])
    return blk is not None and kind(blk) == "full" and blk in b.lining_blocks


def _inward(b: BlockBuild, room: Tuple[int, int, int, int], c: Cell, y: int) -> Optional[str]:
    """The way into the room from a cell against its wall, if it is one."""
    for d, (dx, dz) in (("south", (0, -1)), ("north", (0, 1)), ("east", (-1, 0)), ("west", (1, 0))):
        if _walled(b, room, (c[0] + dx, c[1] + dz), y):
            return d
    return None


def furnish(b: BlockBuild, room: Tuple[int, int, int, int], y: int, items: Sequence[Any],
            keep: Set[Cell], doors: Sequence[Cell]) -> None:
    """Place furniture along the walls of a room without cutting it apart.

    ``room`` is the inclusive interior rectangle (x0, z0, x1, z1) and ``y``
    the standing layer. ``keep`` are cells to leave clear (lanes, stair feet);
    ``doors`` the cells every free cell must stay reachable from. ``items``
    are placed in order wherever they fit, each one of:

    * a block state: one block against a wall; a block with a ``facing``
      (a chest, a furnace) is turned to face into the room;
    * a list of (dx, dz, dy, block) pieces, anchored against a wall and
      tried in all four turns;
    * a dict ``{"pieces": [...], "where": "wall" | "any", "back": bool}``:
      ``"any"`` lets the anchor stand away from the walls (a table with its
      chairs), and ``back`` puts the anchor's north side against a wall (a
      bed's head, a wardrobe's back).
    """
    x0, z0, x1, z1 = room
    free = _floor_cells(b, x0, z0, x1, z1, y)
    lane = b.sizes.lane
    reach = _reach_body(free, doors, lane)
    keep = set(keep) | set(doors)
    wall_cells = [c for c in sorted(free) if _inward(b, room, c, y)]
    inner = [c for c in sorted(free) if c not in wall_cells]
    rng = b.rng
    rng.shuffle(wall_cells)
    rng.shuffle(inner)
    for item in items:
        if isinstance(item, dict):
            pieces, where, back = item["pieces"], item.get("where", "wall"), item.get("back", False)
        elif isinstance(item, list):
            pieces, where, back = item, "wall", False
        else:
            pieces, where, back = [(0, 0, 0, item)], "wall", False
        single = len(pieces) == 1
        pool = wall_cells if where == "wall" else inner + wall_cells
        turns = [0] if single else rng.sample(range(4), 4)
        placed = False
        for (cx, cz) in list(pool):
            for q in turns:
                if back:
                    ox, oz = _turn_xz(0, -1, q)
                    if not _walled(b, room, (cx + int(ox), cz + int(oz)), y):
                        continue
                turned = []
                for dx, dz, dy, blk in pieces:
                    tx, tz = _turn_xz(dx, dz, q)
                    turned.append((cx + int(tx), cz + int(tz), dy, _turn_state(blk, q)))
                cells = [(tx, tz) for tx, tz, _, _ in turned]
                if any(c not in free or c in keep for c in cells):
                    continue
                blocking = {(tx, tz) for tx, tz, _, blk in turned if kind(blk) not in ("carpet", "plant")}
                left = free - blocking
                now = _reach_body(left, doors, lane)
                if now != reach - blocking:
                    continue
                for tx, tz, dy, blk in turned:
                    if single and dy == 0 and "facing=" in blk and kind(blk) == "full":
                        face = _inward(b, room, (tx, tz), y)
                        if face:
                            name, props = parse(blk)
                            props["facing"] = face
                            blk = state(name, **props)
                    b.set(tx, y + dy, tz, blk)
                free -= blocking
                reach = now
                for c in cells:
                    if c in wall_cells:
                        wall_cells.remove(c)
                    if c in inner:
                        inner.remove(c)
                placed = True
                break
            if placed:
                break


def _table(wood: str) -> List[Tuple[int, int, int, str]]:
    return [(0, 0, 0, state(f"{wood}_fence")), (0, 0, 1, state(f"{wood}_pressure_plate"))]


def _dining(wood: str, chairs: int = 2) -> Dict[str, Any]:
    """A table with chairs either side, stair blocks with their backs out."""
    pieces = _table(wood) + [(1, 0, 0, state(f"{wood}_stairs", facing="east", half="bottom")),
                             (-1, 0, 0, state(f"{wood}_stairs", facing="west", half="bottom"))][:chairs]
    return {"pieces": pieces, "where": "any"}


def _bed(colour: str, side: Optional[str] = None) -> Dict[str, Any]:
    """A bed with its head against a wall and a nightstand with a candle."""
    pieces = [(0, 0, 0, state(f"{colour}_bed", facing="north", part="head")),
              (0, 1, 0, state(f"{colour}_bed", facing="north", part="foot"))]
    if side:
        pieces += [(1, 0, 0, state("barrel", facing="up")),
                   (1, 0, 1, state(side, candles="2", lit="true") if side.endswith("candle")
                    else state(side, hanging="false"))]
    return {"pieces": pieces, "back": True}


def _on_barrel(top: str) -> List[Tuple[int, int, int, str]]:
    return [(0, 0, 0, state("barrel", facing="up")), (0, 0, 1, top)]


def _hutch(base: str, top: str) -> Dict[str, Any]:
    """A cupboard against the wall: a chest, smoker or barrel, a barrel shelf
    over it and a pot or candle on top, reaching the ceiling."""
    return {"pieces": [(0, 0, 0, base), (0, 0, 1, state("barrel", facing="south")), (0, 0, 2, top)],
            "back": True}


def _shelves(height: int = 2) -> Dict[str, Any]:
    return {"pieces": [(0, 0, k, state("bookshelf")) for k in range(height)], "back": True}


_CHISELED = {"facing": "north", **{f"slot_{i}_occupied": "true" for i in range(6)}}
_POTS = ("potted_cornflower", "potted_red_tulip", "potted_fern", "potted_azure_bluet", "potted_oxeye_daisy",
         "potted_dandelion")
_RUGS = (("red", "brown"), ("brown", "red"), ("green", "brown"), ("blue", "light_gray"),
         ("cyan", "brown"), ("brown", "yellow"))


# Interior palettes. Each house picks its own: a two-tone wall (a darker
# wainscot at the foot and a frieze under the ceiling, a lighter field
# between), stripped-log trim, and floors that change room to room with a
# darker border. Rooms by use may take their own field: brick or mud brick
# in a kitchen, dark wood in a study.
_FIELD = {"timber": ("calcite", "white_terracotta", "birch_planks"),
          "stone": ("oak_planks", "stripped_oak_wood", "calcite"),
          "brick": ("white_terracotta", "oak_planks", "packed_mud"),
          "oak": ("birch_planks", "calcite", "stripped_birch_wood")}
_WAINSCOT = {"timber": ("dark_oak_planks", "spruce_planks"), "stone": ("spruce_planks", "dark_oak_planks"),
             "brick": ("dark_oak_planks", "spruce_planks"), "oak": ("spruce_planks", "dark_oak_planks")}
_ROOM_FIELD = {"kitchen": ("bricks", "mud_bricks", "packed_mud", "stone_bricks"),
               "study": ("dark_oak_planks", "spruce_planks", "bookshelf"),
               "bedroom": ("birch_planks", "calcite", "white_terracotta", "stripped_birch_wood")}
_TRIM = ("stripped_dark_oak_log", "stripped_spruce_log", "stripped_oak_log", "dark_oak_log", "spruce_log")
_FLOOR_WOOD = ("oak_planks", "spruce_planks", "dark_oak_planks", "birch_planks")
_KITCHEN_FLOOR = ("polished_andesite", "mud_bricks", "spruce_planks", "oak_planks")
# Floors that are not planks; a town finds a house's floor by its planks or these.
INTERIOR_FLOORS = ("polished_andesite", "mud_bricks", "packed_mud")


def interior_palette(style: str, rng: random.Random, uses: Sequence[str]) -> Dict[str, Any]:
    """The materials one house's rooms are finished in, chosen by seed so no
    two houses read the same."""
    wains = rng.choice(_WAINSCOT.get(style, _WAINSCOT["oak"]))
    field = rng.choice(_FIELD.get(style, _FIELD["oak"]))
    rooms = []
    for use in uses:
        f = field
        if use in _ROOM_FIELD and rng.random() < 0.65:
            f = rng.choice(_ROOM_FIELD[use])
        w = wains if f != wains else ("oak_planks" if wains != "oak_planks" else "spruce_planks")
        if use == "kitchen":
            main = rng.choice(_KITCHEN_FLOOR)
        else:
            main = rng.choice(_FLOOR_WOOD)
        edge = "dark_oak_planks" if main not in ("dark_oak_planks",) else "spruce_planks"
        rooms.append({"use": use, "field": f, "wainscot": w, "floor": main, "border": edge})
    leaf = rng.choice((None, None, "dark_oak_leaves", "azalea_leaves", "spruce_leaves"))
    return {"trim": rng.choice(_TRIM), "rooms": rooms, "trade": rng.choice(sorted(_TRADES)), "leaf_posts": leaf}


# A trade for each house: its workstation and the goods about it, so the
# ground-floor room says who lives there.
_TRADES: Dict[str, Tuple[Any, ...]] = {
    "baker": (("smoker", {"facing": "north", "lit": "true"}), ("hay_block", {}), ("composter", {}),
              ("barrel", {"facing": "up"})),
    "smith": (("blast_furnace", {"facing": "north"}), ("smithing_table", {}), ("grindstone", {"face": "floor"}),
              ("anvil", {"facing": "north"})),
    "weaver": (("loom", {"facing": "north"}), ("white_wool", {}), ("light_gray_wool", {}),
               ("chest", {"facing": "north"})),
    "brewer": (("brewing_stand", {}), ("cauldron", {}), ("barrel", {"facing": "up"}), ("decorated_pot", {})),
    "scribe": (("cartography_table", {}), ("lectern", {"facing": "north"}), ("chiseled_bookshelf", _CHISELED),
               ("chest", {"facing": "north"})),
    "farmer": (("composter", {}), ("hay_block", {}), ("barrel", {"facing": "up"}), ("hay_block", {})),
}


def room_items(use: str, wood: str, rng: random.Random, size: int, tall: int = 2,
               trade: Optional[str] = None) -> List[Any]:
    """Furniture for a room by its use, most important first; ``size`` is the
    floor area in blocks, so small rooms get the essentials only; ``tall`` is
    how high a bookcase may stand (the room's clear height). ``trade`` puts
    a workstation and its goods in the kitchen or cottage; pieces after the
    essentials come in a seeded order with seeded stand-ins (a barrel for a
    chest, a blast furnace for a furnace), so no two rooms read the same."""
    pot = lambda: state(rng.choice(_POTS + ("decorated_pot", "flower_pot")))  # noqa: E731
    shelves = lambda: _shelves(tall)  # noqa: E731
    lantern = state("lantern", hanging="false")
    bed = rng.choice(("red", "blue", "green", "yellow", "white", "cyan", "brown"))
    crate = lambda: state(rng.choice(("barrel", "chest")), facing="north")  # noqa: E731
    goods = [state(n, **pr) for n, pr in _TRADES[trade]] if trade in _TRADES else []
    if use == "kitchen":
        items = [state(rng.choice(("furnace", "smoker")), facing="north"), _dining(wood)] + goods[:2] + \
                [state("crafting_table"), _on_barrel(pot()), shelves(), crate(), _on_barrel(lantern),
                 state("cauldron"), _on_barrel(pot())] + goods[2:]
    elif use == "bedroom":
        items = [_bed(bed, "candle"), state("chest", facing="north"), shelves(), _on_barrel(pot()),
                 state("lectern", facing="north"), shelves(), _on_barrel(lantern)]
        if size >= 30:
            items.insert(1, _bed(rng.choice(("red", "blue", "green", "yellow")), None))
    elif use == "study":
        items = [shelves(), shelves(), state("lectern", facing="north"), state("cartography_table"),
                 shelves(), _on_barrel(lantern), state("chest", facing="north"), _on_barrel(pot()), shelves()]
    else:   # a one-room house: a bit of everything
        items = [_bed(bed, "candle"), state("furnace", facing="north"), _dining(wood, 1)] + goods[:2] + \
                [state("crafting_table"), shelves(), _on_barrel(pot()), crate(), _on_barrel(lantern)]
    if tall >= 3:
        hutches = {"kitchen": [_hutch(state("smoker", facing="south"), pot()),
                               _hutch(state("chest", facing="south"), state("candle", candles="3", lit="true"))],
                   "bedroom": [_hutch(state("chest", facing="south"), pot())],
                   "study": [_hutch(state("chest", facing="south"), state("candle", candles="2", lit="true"))]}
        for k, h in enumerate(hutches.get(use, [])):
            items.insert(2 + 2 * k, h)
    if tall >= 3:
        # Crates stacked in a corner: two barrels on their sides and a pot.
        items.insert(4, {"pieces": [(0, 0, 0, state("barrel", facing="east")),
                                    (0, 0, 1, state("barrel", facing="south")), (0, 0, 2, pot())], "back": True})
    # Then fill the walls further in bigger rooms.
    extra = [_on_barrel(pot()), crate(), shelves(), state("decorated_pot"), crate(),
             _on_barrel(state("candle", candles="3", lit="true")), state("composter"), _on_barrel(pot())]
    rng.shuffle(extra)
    items += extra[: max(0, size // 12 - 2)]
    # The essentials first; the rest in a seeded order.
    head, tail = items[:3], items[3:]
    rng.shuffle(tail)
    return head + tail


def rug(b: BlockBuild, room: Tuple[int, int, int, int], y: int, colours: Tuple[str, str],
        most: Tuple[int, int] = (4, 3)) -> int:
    """A bordered carpet on the most open patch of floor away from the walls,
    at most ``most`` blocks. Carpets are walked over, so they never block.
    Returns blocks laid."""
    x0, z0, x1, z1 = room

    def open_(x: int, z: int) -> bool:
        under = b.get(x, y - 1, z)
        return b.get(x, y, z) is None and under is not None and kind(under) == "full"

    best, best_n = None, 0
    for w, d in ((most[0], most[1]), (most[1], most[0]), (3, 2), (2, 3), (2, 2)):
        for rx in range(x0 + 1, x1 - w + 1):
            for rz in range(z0 + 1, z1 - d + 1):
                n = sum(open_(x, z) for x in range(rx, rx + w) for z in range(rz, rz + d))
                if n > best_n or (n == best_n and best and w * d > best[2] * best[3]):
                    best, best_n = (rx, rz, w, d), n
        if best_n and best and best_n >= 0.8 * best[2] * best[3]:
            break
    if not best:
        return 0
    rx, rz, w, d = best
    laid = 0
    for x in range(rx, rx + w):
        for z in range(rz, rz + d):
            if open_(x, z):
                edge = (w > 2 and d > 2) and (x in (rx, rx + w - 1) or z in (rz, rz + d - 1))
                b.set(x, y, z, state(f"{colours[1] if edge else colours[0]}_carpet"))
                laid += 1
    return laid


def wall_banner(b: BlockBuild, room: Tuple[int, int, int, int], y: int, top: int, rng: random.Random) -> None:
    """A banner hung on a bare stretch of wall above the floor, out of the way:
    banners are not walked into."""
    x0, z0, x1, z1 = room
    spots = []
    for x in range(x0 + 1, x1):
        for z, out_z, face in ((z0, z0 - 1, "south"), (z1, z1 + 1, "north")):
            spots.append(((x, out_z), (x, z), face))
    for z in range(z0 + 1, z1):
        for x, out_x, face in ((x0, x0 - 1, "east"), (x1, x1 + 1, "west")):
            spots.append(((out_x, z), (x, z), face))
    rng.shuffle(spots)
    colour = rng.choice(("red", "blue", "yellow"))
    for (wx, wz), (cx, cz), face in spots:
        wall = [b.get(wx, yy, wz) for yy in (top - 1, top)]
        if any(w is None or kind(w) != "full" for w in wall):
            continue
        if any(b.get(cx, yy, cz) is not None for yy in (top - 1, top)):
            continue
        b.set(cx, top, cz, state(f"{colour}_wall_banner", facing=face))
        return


def ceiling_decor(b: BlockBuild, room: Tuple[int, int, int, int], y: int, top: int, rng: random.Random,
                  avoid: Set[Cell], head_clear: bool) -> None:
    """Under a ceiling: lanterns hung over the room and azalea leaves trailing
    from the corners. ``top`` is the layer just under the ceiling; leaves go
    there only when ``head_clear`` says that layer is above every head."""
    x0, z0, x1, z1 = room
    cells = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)
             if (x, z) not in avoid and b.get(x, top, z) is None and b.get(x, top + 1, z) is not None]
    rng.shuffle(cells)
    lanterns = max(1, (x1 - x0 + 1) * (z1 - z0 + 1) // 14)
    placed: List[Cell] = []
    for c in cells:
        if len(placed) >= lanterns:
            break
        if all(abs(c[0] - p[0]) + abs(c[1] - p[1]) >= 4 for p in placed):
            b.set(c[0], top, c[1], state("lantern", hanging="true"))
            placed.append(c)
    if not head_clear:
        return
    corners = [(x0, z0), (x1, z0), (x0, z1), (x1, z1)]
    rng.shuffle(corners)
    for (cx, cz) in corners[:3]:
        for (x, z) in ((cx, cz), (cx + (1 if cx == x0 else -1), cz), (cx, cz + (1 if cz == z0 else -1))):
            if (x, z) in avoid or b.get(x, top, z) is not None or b.get(x, top + 1, z) is None:
                continue
            if rng.random() < 0.75:
                b.set(x, top, z, state(rng.choice(("azalea_leaves", "flowering_azalea_leaves")),
                                     persistent="true", distance="1"))


_LOW_FURNITURE = ("barrel", "chest", "crafting_table", "furnace", "smoker", "cauldron", "cartography_table", "loom",
                  "hay_block", "composter", "white_wool", "light_gray_wool",
                  "blast_furnace", "fletching_table", "smithing_table")


def wall_shelves(b: BlockBuild, room: Tuple[int, int, int, int], y: int, top: int, wood: str,
                 rng: random.Random, most: int = 3) -> None:
    """Shelves on the wall over block-high furniture: a top slab one layer up
    with a pot, a candle or a lantern standing on it. Nobody walks there (the
    furniture below already fills the cell), so the walk is unchanged."""
    if y + 2 > top:
        return
    x0, z0, x1, z1 = room
    cells = [(x, z) for x in range(x0, x1 + 1) for z in range(z0, z1 + 1)
             if (blk := b.get(x, y, z)) is not None and parse(blk)[0] in _LOW_FURNITURE
             and b.get(x, y + 1, z) is None and b.get(x, y + 2, z) is None and _inward(b, room, (x, z), y + 1)]
    rng.shuffle(cells)
    goods = [lambda: state(rng.choice(_POTS)), lambda: state("candle", candles=str(rng.randint(1, 3)), lit="true"),
             lambda: state("lantern", hanging="false"), lambda: state("flower_pot")]
    for (x, z) in cells[:most]:
        b.set(x, y + 1, z, state(f"{wood}_slab", type="top"))
        b.set(x, y + 2, z, rng.choice(goods)())


def great_hall(b: BlockBuild, room: Tuple[int, int, int, int], y: int, top: int, lane: int, keep: Set[Cell],
               starts: Sequence[Cell], rng: random.Random, wood: str = "dark_oak", throne: bool = True) -> Set[Cell]:
    """A hall: a long table down the middle with chairs either side and lit
    candles on it, a carpet round it, a throne against the far wall under
    banners and torches, and lanterns hung over the table. Pieces that would
    cut floor off from the body are taken out again. Returns the cells used,
    so wall furniture can be kept off the aisles round the table."""
    x0, z0, x1, z1 = room
    cx = (x0 + x1) // 2
    used: Set[Cell] = set()
    placed: List[Tuple[int, int, int]] = []

    def put(x: int, yy: int, z: int, blk: str) -> None:
        b.set(x, yy, z, blk)
        placed.append((x, yy, z))

    def reach_ok() -> bool:
        free = _floor_cells(b, x0, z0, x1, z1, y)
        return _reach_body(free, starts, lane) == free

    # Throne on the wall farther from where people come in.
    sz = sum(c[1] for c in starts) / max(1, len(starts))
    tz, face, back = (z0, "north", -1) if abs(sz - z0) > abs(sz - z1) else (z1, "south", 1)
    t0 = z0 + 2 + lane if tz == z0 else z0 + 1 + lane
    t1 = z1 - 1 - lane if tz == z0 else z1 - 2 - lane
    if throne and (cx, tz) not in keep:
        put(cx, y, tz, state(f"{wood}_stairs", facing=face, half="bottom"))
        if top - y >= 2:
            for dx, colour in ((-1, "red"), (1, "red")):
                if b.get(cx + dx, top, tz) is None:
                    put(cx + dx, top, tz, state(f"{colour}_wall_banner", facing=OPPOSITE[face]))
            for dx in (-2, 2):
                if b.get(cx + dx, top, tz) is None:
                    put(cx + dx, top, tz, state("wall_torch", facing=OPPOSITE[face]))
        if not reach_ok():
            for c in placed:
                b.set(*c, None)
            placed.clear()
        else:
            used.add((cx, tz))
    # The table: top slabs on the axis, candles on every other one, chairs
    # facing it in pairs.
    run = [z for z in range(t0, t1 + 1) if (cx, z) not in keep and b.get(cx, y, z) is None]
    for i, z in enumerate(run):
        mark = len(placed)
        put(cx, y, z, state(f"{wood}_slab", type="top"))
        if i % 2 == 1:
            put(cx, y + 1, z, state("candle", candles=str(rng.randint(2, 4)), lit="true"))
        if i % 2 == 0 and 0 < i < len(run) - 1:
            for dx, f in ((-1, "west"), (1, "east")):
                if (cx + dx, z) not in keep and b.get(cx + dx, y, z) is None:
                    put(cx + dx, y, z, state(f"{wood}_stairs", facing=f, half="bottom"))
        if not reach_ok():
            for c in placed[mark:]:
                b.set(*c, None)
            del placed[mark:]
            break
        used |= {(cx, z), (cx - 1, z), (cx + 1, z)}
    # Carpet round the table, then lanterns over it.
    if run:
        for x in range(cx - 2, cx + 3):
            for z in range(run[0] - 1, run[-1] + 2):
                if x0 <= x <= x1 and z0 <= z <= z1 and (x, z) not in keep and b.get(x, y, z) is None:
                    under = b.get(x, y - 1, z)
                    if under is not None and kind(under) == "full":
                        b.set(x, y, z, state("red_carpet" if abs(x - cx) < 2 else "yellow_carpet"))
                        used.add((x, z))
        drop = max(0, top - y - 3)        # hang to about head height plus a block
        for z in run[1::3]:
            if b.get(cx, top, z) is None and b.get(cx, top + 1, z) is not None:
                for yy in range(top, top - drop, -1):
                    b.set(cx, yy, z, state("chain", axis="y"))
                b.set(cx, top - drop, z, state("lantern", hanging="true"))
    return used


def corner_pots(b: BlockBuild, room: Tuple[int, int, int, int], y: int, rng: random.Random, keep: Set[Cell]) -> None:
    """Potted plants in empty corners; a pot is not walked into, it is
    stepped past, and never stands on a lane."""
    x0, z0, x1, z1 = room
    for c in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
        if c in keep or b.get(c[0], y, c[1]) is not None or b.get(c[0], y + 1, c[1]) is not None:
            continue
        # Only beside open floor: a pot next to furniture would close a nook
        # a wide body can no longer turn into.
        lane = b.sizes.lane
        near = [(sx, sz) for sx in range(c[0] - lane, c[0] + lane + 1)
                for sz in range(c[1] - lane, c[1] + lane + 1) if (sx, sz) != c]
        if any(b.get(sx, y, sz) is not None and (kind(b.get(sx, y, sz)) != "full"
                                                 or parse(b.get(sx, y, sz))[0] in _AnyWall._FURNITURE)
               for sx, sz in near):
            continue
        if rng.random() < 0.7:
            b.set(c[0], y, c[1], state(rng.choice(_POTS)))


# ------------------------------------------------------------------- house
def house(player: Any = "blocky", seed: int = 0, width: Optional[int] = None, depth: Optional[int] = None,
          storeys: Optional[int] = None, style: Optional[str] = None, name: str = "house",
          roof: str = "gable") -> BlockBuild:
    """A village house in its own frame: the front wall on z = 0 facing north,
    the door in it, the house running east along x and south along z.

    Ground floor floor at layer 0 (stood on at y = 1), a stair outside the
    door, timber-framed walls on a stone base, windows with shutters, a 45
    degree gable roof with overhangs, an outside chimney, and furnished rooms
    with a stair to the upper floor sized to the player.
    """
    b = BlockBuild(name, player, seed)
    rng = b.rng
    s = b.sizes
    style = style or rng.choice(sorted(HOUSE_STYLES))
    st = HOUSE_STYLES[style] if style in HOUSE_STYLES else EXTRA_STYLES[style]
    storeys = storeys if storeys is not None else rng.choice((1, 2, 2))
    S = s.storey
    lane = s.lane
    if storeys > 1:
        min_w = S + lane + 4
    else:
        min_w = max(6, 2 * lane + 4)
    width = max(width or rng.randint(min_w + 1, min_w + 4), min_w)
    depth = max(depth or (rng.randint(8, 11) if storeys > 1 else rng.randint(7, 10)), 2 * lane + 5, 7)
    X, Z = width - 1, depth - 1
    base = b.mix(*st["base"])
    lower = b.mix(*st["lower"])
    upper = b.mix(*st["upper"])
    post = st["post"]
    roof_stairs, roof_slab, roof_full = _roof_blocks(st["roof"])
    top = storeys * S                     # eave layer

    # Floor and base course.
    b.fill(0, 0, 0, X, 0, Z, base)
    b.fill(1, 0, 1, X - 1, 0, Z - 1, state(st["floor"]))
    # Walls.
    for level in range(storeys):
        y0 = level * S + 1
        y1 = level * S + S - 1
        fill = lower if level == 0 else upper
        b.ring(0, 0, X, Z, y0, y1, fill)
        if level > 0:
            # A timber band at each upper floor.
            for x in range(0, X + 1):
                b.set(x, level * S, 0, state(st["band"], axis="x"))
                b.set(x, level * S, Z, state(st["band"], axis="x"))
            for z in range(1, Z):
                b.set(0, level * S, z, state(st["band"], axis="z"))
                b.set(X, level * S, z, state(st["band"], axis="z"))
            b.fill(1, level * S, 1, X - 1, level * S, Z - 1, state(st["floor"]))
    for (x, z) in ((0, 0), (X, 0), (0, Z), (X, Z)):
        b.fill(x, 1, z, x, top - 1, z, state(post, axis="y"))
    # Posts along long walls for the timber frame.
    if style in ("timber", "oak", "tudor"):
        for x in range(3, X - 1, 3):
            for z in (0, Z):
                b.fill(x, 1, z, x, top - 1, z, state(post, axis="y"))

    if roof == "hip":
        # A hipped roof, a pyramid on a square plan: every side has eaves.
        ridge = b.hip_roof(0, 0, X, Z, top, roof_stairs, roof_slab, overhang=1)
        for x in range(-1, X + 2):
            b.set(x, top - 1, -1, state(roof_stairs, facing="north", half="top"), replace=False)
            b.set(x, top - 1, Z + 1, state(roof_stairs, facing="south", half="top"), replace=False)
        for z in range(0, Z + 1):
            b.set(-1, top - 1, z, state(roof_stairs, facing="west", half="top"), replace=False)
            b.set(X + 1, top - 1, z, state(roof_stairs, facing="east", half="top"), replace=False)
    else:
        # Roof along x, gables at the east and west ends.
        gable = b.mix(*(((st["gable"], 1),)))
        ridge = b.gable_roof(0, 0, X, Z, top, roof_stairs, roof_slab, gable, overhang=1, axis="x")
        # Upside-down stairs under the eaves give the overhang a lip.
        for x in range(-1, X + 2):
            b.set(x, top - 1, -1, state(roof_stairs, facing="north", half="top"), replace=False)
            b.set(x, top - 1, Z + 1, state(roof_stairs, facing="south", half="top"), replace=False)

    # Door in the front wall, the stair outside it.
    dx = rng.randint(1, max(1, X - lane - (S + 1 if storeys > 1 else 1)))
    dx = min(dx, X - lane)
    if roof == "cross" and X >= 6 and Z >= 6:
        # A cross gable over the door bay facing the street: a second gable
        # roof running back from the front wall to the main ridge, with a
        # window in its gable.
        cx = min(max(dx + lane // 2, 2), X - 2)
        # Laid over the main roof, not cut into it: the main roof stays whole
        # beneath, so no pocket of attic opens under the cross gable.
        cross = BlockBuild("cross", player, seed)
        cross.gable_roof(cx - 2, 0, cx + 2, Z // 2, top, roof_stairs, roof_slab, b.mix(*(((st["gable"], 1),))),
                         overhang=1, axis="z")
        for (x, y, z), blk in cross.blocks.items():
            # Keep what shows: the front gable and its overhang, and what
            # stands on or above the main roof's slope. Eaves that would
            # fall inside the attic are left out.
            main = top + min(z + 1, Z + 1 - z)
            if z <= 0 or y >= main:
                b.set(x, y, z, blk)
        for x in range(cx - 3, cx + 4):
            b.set(x, top - 1, -1, state(roof_stairs, facing="north", half="top"), replace=False)
        b.set(cx, top + 1, 0, state("glass_pane"))
    b.door(dx, 1, 0, "south", wood=st["door"], along="east", name=name)
    for w in range(lane):
        b.set(dx + w, 0, -1, state(st["floor"].replace("_planks", "_stairs") if "planks" in st["floor"]
                                  else "stone_brick_stairs", facing="south", half="bottom"))
        b.set(dx + w, 0, 0, state(st["floor"]))
    b.set(dx - 1, min(s.door_height, S - 1), -1, state("wall_torch", facing="north"))

    # Windows: two blocks tall, every third block, with open shutters outside.
    def windows(level: int) -> None:
        y = level * S + 2
        h = min(2, S - 2)
        for wall, (fixed, run, facing) in {"front": (0, range(2, X - 1, 3), "north"),
                                          "back": (Z, range(2, X - 1, 3), "south")}.items():
            for x in run:
                if level == 0 and wall == "front" and dx - 1 <= x <= dx + lane:
                    continue
                for j in range(h):
                    b.set(x, y + j, fixed, state("glass_pane"))
                    for side in (x - 1, x + 1):
                        if b.get(side, y + j, fixed) and kind(b.get(side, y + j, fixed)) == "full":
                            nz = fixed - 1 if facing == "north" else fixed + 1
                            b.set(side, y + j, nz, state(st["shutter"], facing=facing, half="bottom",
                                                         open="true"))
        for fixed, facing in ((0, "west"), (X, "east")):
            for z in range(2, Z - 1, 3):
                for j in range(h):
                    b.set(fixed, y + j, z, state("glass_pane"))

    for level in range(storeys):
        windows(level)
    wood_ = st["floor"].replace("_planks", "") if st["floor"].endswith("_planks") else "spruce"
    # Flower boxes under the upper windows, front and back: a top slab
    # outside the sill with a potted plant on it.
    for level in range(1, storeys):
        y = level * S + 1
        for fixed, out in ((0, -1), (Z, Z + 1)):
            for x in range(2, X - 1, 3):
                if b.get(x, y + 1, fixed) is not None and "glass" in b.get(x, y + 1, fixed) and rng.random() < 0.7:
                    b.set(x, y, out, state(f"{wood_}_slab", type="top"), replace=False)
                    b.set(x, y + 1, out, state(rng.choice(_POTS)), replace=False)
    # A balcony over the door on taller houses: a slab floor and a fence rail
    # in front of the upper storey, a lantern hung under it by the door.
    if storeys > 1:
        for x in range(dx - 1, dx + lane + 1):
            if 0 <= x <= X:
                b.set(x, S, -1, state(f"{wood_}_slab", type="top"))
                b.set(x, S + 1, -1, state(f"{wood_}_fence"))
        b.set(dx + lane, S - 1, -1, state("lantern", hanging="true"), replace=False)

    # Chimney up the west gable, outside, with a campfire for smoke.
    cz = Z // 2
    chim = b.mix(*st["chimney"])
    b.fill(-1, 0, cz, -1, ridge + 1, cz, chim)
    b.set(-1, ridge + 2, cz, state("campfire", lit="true", signal_fire="false"))

    # Inside: stair to the upper floor, then furniture that leaves the lanes.
    inside = (1, 1, X - 1, Z - 1)
    door_cells = [(dx + w, 1) for w in range(lane)]
    sx, sz = X - S, Z - lane
    # A gallery: in two-storey houses deep enough, the upper floor covers only
    # the back of the house (the stair and a lane beside its head), and the
    # front of the main room is open to the rafters, two storeys high, with a
    # rail along the gallery's edge.
    zg = sz - lane - 1                    # the rail's row; the gallery is zg .. Z - 1
    gallery = storeys == 2 and zg >= 4 and rng.random() < 0.7
    gal_room = (1, zg + 1, X - 1, Z - 1)
    if gallery:
        for x in range(1, X):
            for z in range(1, zg):
                b.set(x, S, z, None)
            b.set(x, S + 1, zg, state(f"{wood_}_fence"))
    stairs_near: Set[Cell] = set()
    if storeys > 1:
        for i in range(-1, S + 1):
            for w in range(-1, lane + 1):
                stairs_near.add((sx + i, sz + w))

    # The rooms' finishes: this house's own palette, a use per storey.
    uses = (["kitchen", "bedroom", "study"][:storeys] if storeys > 1 else ["cottage"])
    if storeys > 2:
        uses = ["kitchen", "study", "bedroom", "bedroom", "study", "bedroom"][:storeys]
    pal = interior_palette(style, rng, uses)
    floors_used = {st["floor"]}
    # Floors: each storey its own, with a darker border just inside the
    # walls (inside the panelling where there is one).
    lined_guess = storeys > 1 and X - 1 >= 6 and Z - 1 >= 6
    r = 2 if lined_guess else 1
    for level in range(storeys):
        room = pal["rooms"][level]
        for x in range(1, X):
            for z in range(1, Z):
                cur = b.get(x, level * S, z)
                if cur is None or parse(cur)[0] != st["floor"]:
                    continue
                edge = r <= x <= X - r and r <= z <= Z - r and (x in (r, X - r) or z in (r, Z - r))
                b.set(x, level * S, z, state(room["border"] if edge else room["floor"]))
        floors_used |= {room["border"], room["floor"]}

    # Panelling: a second skin of planks inside the walls, with stripped-log
    # pilasters, in rooms that have a ceiling and are big enough to give up a
    # block each side. (Under the roof its top would be a ledge nobody
    # reaches.) Windows go through it as a second pane; the door, the hearth
    # and the stairs keep the outer wall.
    lined = storeys > 1 and X - 1 >= 6 and Z - 1 >= 6
    if lined:
        pilaster = pal["trim"]
        ring = [(x, z) for x in range(1, X) for z in range(1, Z) if x in (1, X - 1) or z in (1, Z - 1)]
        for level in range(storeys - 1):
            placed_here: List[Tuple[int, int, int]] = []
            for (x, z) in ring:
                if ((x, z) in door_cells or (x, z - 1) in door_cells or (x, z) == (1, cz)
                        or (x, z) in stairs_near):
                    continue
                outer = [(ox, oz) for ox, oz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1))
                         if ox in (0, X) or oz in (0, Z)]
                corner = len(outer) > 1
                # Under the open rafters the panelling runs up both storeys.
                up_to = 2 * S if gallery and level == 0 and z < zg else level * S + S
                if up_to > level * S + S:
                    # ... and on to the roof, so its top is no ledge.
                    while up_to < ridge + 1 and b.get(x, up_to, z) is None:
                        up_to += 1
                for yy in range(level * S + 1, up_to):
                    if any(b.get(ox, yy, oz) and "glass_pane" in b.get(ox, yy, oz) for ox, oz in outer):
                        # Up in the open rafters a whole glass block, so the
                        # panelling under the window is no sill to stand on.
                        b.set(x, yy, z, state("glass_pane" if yy < level * S + S else "glass"))
                    elif corner or (x + z) % 4 == 0:
                        if pal["leaf_posts"] and not corner:
                            # Some houses let leaves grow up the posts.
                            b.set(x, yy, z, state(pal["leaf_posts"], persistent="true", distance="1"))
                        else:
                            b.set(x, yy, z, state(pilaster, axis="y"))
                    else:
                        # A wainscot at the foot, a frieze under the ceiling,
                        # the room's field between.
                        room = pal["rooms"][level]
                        local = (yy - 1) % S
                        b.set(x, yy, z, state(room["wainscot"] if local in (0, S - 2) else room["field"]))
                    placed_here.append((x, yy, z))
            # Keep it only if a body as wide as the lane still reaches every
            # free cell of the floor from the door or the stair head.
            yl = level * S + 1
            free = _floor_cells(b, *inside, yl)
            starts = door_cells if level == 0 else [(sx + S - 1 - w, sz - 1) for w in range(lane)]
            if _reach_body(free, starts, lane) != free:
                for (x, yy, z) in placed_here:
                    b.set(x, yy, z, None)
    keep: Set[Cell] = set()
    if storeys > 1:
        # The flight runs east along the back wall, rows Z - lane .. Z - 1.
        for level in range(storeys - 1):
            yb = level * S + 1
            b.stair_flight(sx, yb, sz, "east", S, f"{st['floor'].replace('_planks', '')}_stairs", width=lane,
                           along="south")
            hb = s.headroom
            # The hole in the floor above: wherever the body on a tread would
            # meet the floor, plus a block for its width.
            for i in range(max(0, S - 2 - hb), S - 1):
                for w in range(lane):
                    b.set(sx + i, (level + 1) * S, sz + w, None)
            for i in range(S):
                for w in range(lane):
                    keep.add((sx + i, sz + w))
            for w in range(lane):
                keep.add((sx - 1, sz + w))
        free = _floor_cells(b, *inside, 1)
        keep |= _lane(b, free, door_cells[0], (sx - 1, sz))
    else:
        free = _floor_cells(b, *inside, 1)
        keep |= _lane(b, free, door_cells[0], (X // 2, Z // 2))
    wood = st["floor"].replace("_planks", "")
    area = (X - 1) * (Z - 1)

    # A fireplace in the west wall, backed by the chimney, with a hearth.
    hearth = (1, cz)
    b.set(0, 1, cz, state("campfire", lit="true", signal_fire="false", facing="east"))
    for yy in range(2, S):
        b.set(0, yy, cz, chim())
    b.set(1, 0, cz, state("bricks"))
    keep.add(hearth)

    rugs = list(_RUGS)
    rng.shuffle(rugs)
    open_room = (2, 2, X - 2, Z - 2) if lined else inside
    room_of = lambda level: open_room if level < storeys - 1 else (gal_room if gallery else inside)  # noqa: E731
    # A room with a ceiling takes bookcases up to it; a room open to the roof
    # takes nothing over a block high, or its top would be floor nobody reaches.
    # Under a gallery they stop a block short, so no top lines up with its floor.
    # Under the open roof, shelves may stand taller than the player reaches:
    # their tops are decoration, at no floor's height.
    tall_of = lambda level: (S - 2 if gallery else S - 1) if level < storeys - 1 else 2  # noqa: E731
    furnish(b, inside, 1, room_items(uses[0], wood, rng, area, tall_of(0), trade=pal["trade"]), keep, door_cells)
    if storeys > 1:
        wall_shelves(b, inside, 1, S - 1, wood, rng)
    wall_banner(b, open_room, 1, S - 1, rng)
    rug(b, open_room, 1, rugs[0])
    corner_pots(b, open_room, 1, rng, keep)
    if storeys > 1:
        landing = [(X - 1 + 1 - 1, Z - lane + w) for w in range(lane)]
        for level in range(1, storeys):
            ys = level * S + 1
            upkeep: Set[Cell] = set()
            up_room = gal_room if gallery else inside
            free_up = _floor_cells(b, *up_room, ys)
            # The flight ends at the east wall; the landing is beside its top.
            arrive = [(sx + S - 1 - w, sz - 1) for w in range(lane) if (sx + S - 1 - w, sz - 1) in free_up]
            for w in range(lane):
                upkeep.add((sx + S - 1 - w, sz - 1))
                upkeep.add((sx + S - 1 - w, sz - 1 - lane + 1))
            for i in range(S):
                for w in range(lane):
                    upkeep.add((sx + i, sz + w))
            if level < storeys - 1:
                # The next flight starts here: keep its foot and the way to it.
                for w in range(lane):
                    upkeep.add((sx - 1, sz + w))
                upkeep |= _lane(b, free_up, arrive[0], (sx - 1, sz)) if arrive else set()
            up_area = (up_room[2] - up_room[0] + 1) * (up_room[3] - up_room[1] + 1)
            furnish(b, up_room, ys, room_items(uses[level], wood, rng, up_area, tall_of(level)), upkeep, arrive)
            if level < storeys - 1:
                wall_shelves(b, inside, ys, level * S + S - 1, wood, rng)
            wall_banner(b, room_of(level), ys, level * S + S - 1, rng)
            rug(b, room_of(level), ys, rugs[level % len(rugs)])
            corner_pots(b, room_of(level), ys, rng, upkeep)
        if gallery and Z // 2 < zg:
            # Over the open front a lantern hangs from the rafters on a chain.
            lx, lz = (X + 1) // 2, max(1, zg // 2)
            yy = S + 1
            while b.get(lx, yy, lz) is None and yy < ridge:
                b.set(lx, yy, lz, state("chain", axis="y"))
                yy += 1
            b.set(lx, S, lz, state("lantern", hanging="true"))
        else:
            b.set((X + 1) // 2, S - 1, Z // 2, state("lantern", hanging="true"))
        b.set((X + 1) // 2, 2 * S - 1 if storeys > 2 else ridge - 1, Z // 2, state("lantern", hanging="true"),
              replace=False)
    else:
        b.set((X + 1) // 2, ridge - 1, Z // 2, state("chain", axis="y"))
        b.set((X + 1) // 2, ridge - 2, Z // 2, state("lantern", hanging="true"))
        for yy in range(top, ridge - 2):
            b.set((X + 1) // 2, yy, Z // 2, state("chain", axis="y"), replace=False)

    # A storage loft under the roof at the chimney end: a floor of bottom
    # slabs over the top room's west end, at the eaves, stacked with stores.
    # Nobody climbs to it (decoration, half a block off any floor's height),
    # and the room under it keeps its full height to the eaves.
    loft = min(3, (X + 1) // 2 - 1)
    lofted = 0
    if roof == "gable" and loft >= 2 and S - 1 >= s.door_height + 1:
        stores = ["barrel[facing=up]", "hay_block[axis=y]", "chest[facing=east]", "barrel[facing=up]",
                  "hay_block[axis=x]", "white_wool", "decorated_pot", "barrel[facing=east]"]
        for x in range(1, loft + 1):
            for z in range(1, Z):
                if b.get(x, top, z) is None:
                    b.set(x, top, z, state(f"{wood_}_slab", type="bottom"))
        for z in range(1, Z):
            for x in range(1, loft + 1):
                if rng.random() < .45 and b.get(x, top + 1, z) is None:
                    b.set(x, top + 1, z, "minecraft:" + rng.choice(stores))
                    lofted += 1
                    if rng.random() < .3 and b.get(x, top + 2, z) is None and x < loft:
                        b.set(x, top + 2, z, state("barrel", facing="up"))
        # Its edge is a rail.
        for z in range(1, Z):
            if b.get(loft, top + 1, z) is None:
                b.set(loft, top + 1, z, state(f"{wood_}_fence"))

    # Lanterns and trailing leaves under each ceiling.
    for level in range(storeys - 1):
        ceiling_decor(b, open_room, level * S + 1, (level + 1) * S - 1, rng, stairs_near,
                      head_clear=S - 1 >= s.door_height + 1)
    # Beams laid in each ceiling, a stripped log every fourth row across the
    # planks: they show from below without taking any headroom.
    for level in range(storeys - 1):
        yb = (level + 1) * S
        beam = (post if post.startswith("stripped_") else f"stripped_{post}") if post.endswith("_log") \
            else "stripped_spruce_log"
        for x in range(2, X - 1, 4):
            for z in range(1, Z):
                cur = b.get(x, yb, z)
                if cur is not None and parse(cur)[0] in floors_used:
                    b.set(x, yb, z, state(beam, axis="z"))
    # Built-in shelves let into the panelling above the wainscot: book
    # shelves in a study, barrel cubbies in a kitchen, a few to a room. They
    # replace panelling block for block, so nothing a body meets changes.
    if lined:
        into = lambda x, z: ("east" if x == 1 else "west" if x == X - 1 else  # noqa: E731
                             "south" if z == 1 else "north")
        for level in range(storeys - 1):
            room = pal["rooms"][level]
            yy = level * S + 2
            spots = [(x, z) for x in range(1, X) for z in range(1, Z)
                     if (x in (1, X - 1)) != (z in (1, Z - 1)) and b.get(x, yy, z) is not None
                     and parse(b.get(x, yy, z))[0] == room["field"] and b.get(*{
                         "east": (x + 1, yy, z), "west": (x - 1, yy, z),
                         "south": (x, yy, z + 1), "north": (x, yy, z - 1)}[into(x, z)]) is None]
            rng.shuffle(spots)
            for (x, z) in spots[:3]:
                f = into(x, z)
                if room["use"] == "study":
                    blk = state("chiseled_bookshelf", facing=f, **{f"slot_{i}_occupied": rng.choice(("true", "false"))
                                                                    for i in range(6)})
                elif room["use"] == "kitchen":
                    blk = state("barrel", facing=f)
                else:
                    blk = state("bookshelf")
                b.set(x, yy, z, blk)
    # What Minecraft keeps as data: goods in the chests and barrels, patterns
    # on the banners, a framed tool of the trade and a painting on the walls.
    from .blockdress import fill_containers, hang, pattern_banners
    fill_containers(b, pal["trade"])
    pattern_banners(b)
    for level in range(storeys):
        hang(b, room_of(level), level * S + 2, pal["trade"], frames=1,
             paintings=1 if uses[level] in ("bedroom", "study", "cottage") else 0)
    b.notes.append(f"{name}: {style}, {width}x{depth}, {storeys} storey(s), storey {S} blocks"
                   + (", open to the rafters in front of a gallery" if gallery else "")
                   + f", the {pal['trade']}'s house, rooms in "
                   + "; ".join(f"{r_['use']}: {r_['field']} over {r_['wainscot']} on {r_['floor']}"
                               for r_ in pal["rooms"]))
    return b


# -------------------------------------------------------------------- tree
def tree(b: BlockBuild, x: int, z: int, height: Optional[int] = None, wood: str = "oak") -> None:
    rng = b.rng
    h = height or rng.randint(4, 6)
    for y in range(h):
        b.set(x, y, z, state(f"{wood}_log", axis="y"))
    leaves = state(f"{wood}_leaves", persistent="true")
    for (lx, lz) in disc(x + 0.5, z + 0.5, 2.6):
        for y in range(h - 2, h):
            if b.get(lx, y, lz) is None and rng.random() > 0.12:
                b.set(lx, y, lz, leaves)
    for (lx, lz) in disc(x + 0.5, z + 0.5, 1.6):
        for y in range(h, h + 2):
            if b.get(lx, y, lz) is None and (y == h or (lx, lz) == (x, z) or rng.random() > 0.4):
                b.set(lx, y, lz, leaves)


def lamp_post(b: BlockBuild, x: int, z: int, height: int = 3) -> None:
    for y in range(height - 1):
        b.set(x, y, z, state("spruce_fence"))
    b.set(x, height - 1, z, state("lantern", hanging="false"))


def well(b: BlockBuild, x0: int, z0: int) -> None:
    """A 3 by 3 well with a roof, its min corner at (x0, z0)."""
    for x in range(x0, x0 + 3):
        for z in range(z0, z0 + 3):
            b.path(x, z, "cobblestone")
            if (x, z) == (x0 + 1, z0 + 1):
                b.path(x, z, "water")
                continue
            b.set(x, 0, z, state("cobblestone_wall"))
    for x, z in ((x0, z0), (x0 + 2, z0), (x0, z0 + 2), (x0 + 2, z0 + 2)):
        b.set(x, 1, z, state("spruce_fence"))
        b.set(x, 2, z, state("spruce_fence"))
    b.fill(x0, 3, z0, x0 + 2, 3, z0 + 2, state("spruce_slab", type="bottom"))
    b.set(x0 + 1, 2, z0 + 1, state("chain", axis="y"))


# ----------------------------------------------------------------- village
def village(player: Any = "blocky", seed: int = 0, houses: int = 8) -> BlockBuild:
    """A village street: houses either side of a path, front gardens with
    fences and flowers, fields behind, trees, lamp posts and a well."""
    b = BlockBuild("village", player, seed)
    rng = b.rng
    lane = b.sizes.lane
    street = max(5, 2 * lane + 3)
    per_side = (houses + 1) // 2
    garden = 4
    x = 0
    lots: List[Tuple[int, int, BlockBuild]] = []
    for i in range(houses):
        h = house(player, seed * 101 + i, name=f"house{i:02d}")
        lots.append((i % 2, i // 2, h))
    # Lay out each side left to right.
    cursor = [0, 0]
    placed = []
    for side, _, h in lots:
        (lx, _, lz), (hx, _, hz) = h.bounds()
        w = hx - lx + 1
        x0 = cursor[side] + 1 - lx
        cursor[side] += w + rng.randint(2, 4)
        placed.append((side, x0, h, (lx, lz, hx, hz)))
    length = max(cursor) + 2
    # Street along x at z in [0, street).
    for x in range(-2, length + 2):
        for z in range(0, street):
            b.path(x, z, "dirt_path" if rng.random() > 0.15 else "gravel")
    for side, x0, h, (lx, lz, hx, hz) in placed:
        if side == 0:
            # South of the street, front facing north: no turn.
            front = street + garden
            paste(b, h, x0, front)
            door = h.ports[0]
            px = int(math.floor(door["position"][0])) + x0
            for z in range(street, front - 1):
                for w in range(lane):
                    b.path(px + w - (lane // 2 if lane > 1 else 0), z, "dirt_path")
            fence_z = street
            back = front + hz + 2
            field_dir = 1
        else:
            # North of the street, front facing south: half a turn.
            front = -garden - 1
            paste(b, h, x0 + (hx - lx) + 1 + lx * 0, front, quarter=2)
            door = b.ports[-1]
            px = int(math.floor(door["position"][0]))
            for z in range(front + 2, 0):
                for w in range(lane):
                    b.path(px - w + (lane // 2 if lane > 1 else 0), z, "dirt_path")
            fence_z = -1
            back = front - hz - 2
            field_dir = -1
        # Garden fence with a gap at the path, and flowers.
        gx0, gx1 = x0 + lx - 1, x0 + hx + 1 if side == 0 else x0 + (hx - lx) + 1 + 1
        if side == 1:
            gx0 = x0 - 1 - (hx - lx) + (hx - lx)
            gx1 = x0 + (hx - lx) + 1
        for gx in range(min(gx0, gx1), max(gx0, gx1) + 1):
            if any(abs(gx - (px + w * (1 if side == 0 else -1))) < 1 for w in range(lane)):
                continue
            if b.get(gx, 0, fence_z) is None:
                b.set(gx, 0, fence_z, state("oak_fence"))
        flowers = ("poppy", "dandelion", "cornflower", "oxeye_daisy", "allium")
        for gx in range(min(gx0, gx1) + 1, max(gx0, gx1)):
            for gz in (fence_z + field_dir,):
                if b.get(gx, 0, gz) is None and (gx, gz) not in b.ground and rng.random() < 0.6:
                    b.set(gx, 0, gz, state(rng.choice(flowers)))
        # A field behind: farmland rows with a water channel.
        fz0 = back
        for fx in range(min(gx0, gx1), max(gx0, gx1) + 1):
            for k in range(5):
                fz = fz0 + field_dir * k
                if k == 2:
                    b.path(fx, fz, "water")
                else:
                    b.path(fx, fz, "farmland")
                    b.set(fx, 0, fz, state("wheat", age=rng.choice((5, 6, 7, 7))))
    # Trees between some lots, lamp posts along the street, a well midway.
    for side, x0, h, (lx, lz, hx, hz) in placed:
        if rng.random() < 0.7:
            tz = street + garden + 2 if side == 0 else -garden - 3
            tx = x0 - 2 if side == 0 else x0 + (hx - lx) + 3
            if all(b.get(tx + dx, 0, tz + dz) is None for dx in (-1, 0, 1) for dz in (-1, 0, 1)):
                tree(b, tx, tz)
    for x in range(2, length, 9):
        for z in (-1, street):
            if b.get(x, 0, z) is None and all(b.get(x, 0, z + d) is None for d in (-1, 1)):
                lamp_post(b, x, z)
    well(b, length // 2 - 1, street // 2 - 1)
    for x in range(length // 2 - 1, length // 2 + 2):
        for z in range(street // 2 - 1, street // 2 + 2):
            pass
    b.notes.append(f"village: {houses} houses, street {street} wide")
    return b


# ------------------------------------------------------------------ castle
def castle(player: Any = "blocky", seed: int = 0) -> BlockBuild:
    """A castle: a curtain wall with a wall-walk and crenellations, round
    corner towers with conical roofs, a twin-towered gatehouse, stairs from
    the bailey to the wall-walk, a square keep with furnished floors and a
    roof walk, and a bailey with houses, a well and a path.

    The wall-walk, every stair, door and room is sized to the player; the
    wall-walk runs through the towers, so one stair from the bailey reaches
    all of it.
    """
    b = BlockBuild("castle", player, seed)
    rng = b.rng
    s = b.sizes
    S, lane = s.storey, s.lane
    walk = max(2, lane + 1)
    T = walk + 1                          # curtain thickness: walk plus parapet
    Hw = max(8, 2 * S)                    # wall-walk surface height
    keep_w = S + 2 * lane + 11
    W = max(44, keep_w + 2 * 10 + 2 * T)
    D = max(38, keep_w + 24)
    stone = b.mix(("stone_bricks", 14), ("mossy_stone_bricks", 3), ("cracked_stone_bricks", 2), ("andesite", 1))
    base = b.mix(("cobblestone", 3), ("mossy_cobblestone", 1), ("polished_andesite", 1))
    trim = "polished_andesite"
    roof_stairs, roof_slab, roof_full = _roof_blocks(rng.choice(("spruce", "dark_oak", "deepslate_tile")))

    def curtain(x0: int, z0: int, x1: int, z1: int, out: str) -> None:
        """A straight wall from (x0, z0) to (x1, z1) (inclusive, one axis),
        thickness T toward the inside; ``out`` is the outside direction."""
        ox, oz = DIRS[out]
        ix, iz = -ox, -oz
        for t in range(T):
            for x in range(min(x0, x1) + ix * t if ix else min(x0, x1), max(x0, x1) + ix * t + 1 if ix else max(x0, x1) + 1):
                for z in range(min(z0, z1) + iz * t if iz else min(z0, z1),
                               max(z0, z1) + iz * t + 1 if iz else max(z0, z1) + 1):
                    b.set(x, 0, z, base())
                    b.fill(x, 1, z, x, Hw - 1, z, stone)
                    if t == 0:
                        b.set(x, Hw, z, stone())
                        if (x + z) % 2 == 0:
                            b.set(x, Hw + 1, z, stone())
                            b.set(x, Hw + 2, z, state("stone_brick_slab", type="bottom"))

    X, Z = W - 1, D - 1
    curtain(0, 0, X, 0, "north")
    curtain(0, Z, X, Z, "south")
    curtain(0, 0, 0, Z, "west")
    curtain(X, 0, X, Z, "east")
    # A string course of trim under the parapet, outside.
    for x in range(0, X + 1):
        b.set(x, Hw - 1, 0, state(trim))
        b.set(x, Hw - 1, Z, state(trim))
    for z in range(0, Z + 1):
        b.set(0, Hw - 1, z, state(trim))
        b.set(X, Hw - 1, z, state(trim))

    # Round corner towers: a solid drum up to the wall-walk, a room on it that
    # the wall-walk runs through, and a steep conical roof.
    R = max(4.0, T + 2.5)
    room_top = Hw + s.ceiling           # eave layer of the tower roof
    corners = ((0.5 + T / 2, 0.5 + T / 2), (X + 0.5 - T / 2, 0.5 + T / 2), (0.5 + T / 2, Z + 0.5 - T / 2),
               (X + 0.5 - T / 2, Z + 0.5 - T / 2))
    for (cx, cz) in corners:
        drum = disc(cx, cz, R)
        wall_ring = circle(cx, cz, R, 1.0)
        for (x, z) in drum:
            b.set(x, 0, z, base())
            b.fill(x, 1, z, x, Hw - 1, z, stone)
        for (x, z) in wall_ring:
            b.fill(x, Hw, z, x, room_top - 1, z, stone)
        for (x, z) in drum - wall_ring:
            for y in range(Hw, room_top):
                b.set(x, y, z, None)
        # Top: a stone deck with a crenellated parapet, and a slender spire
        # rising from its middle.
        for (x, z) in drum:
            b.set(x, room_top, z, stone())
        for (x, z) in wall_ring:
            b.set(x, room_top + 1, z, stone())
            if (x + z) % 2 == 0:
                b.set(x, room_top + 2, z, stone())
        b.spire(cx, cz, max(1.6, R - 1.6), room_top + 1, roof_stairs, roof_full)
        # Corbels: a ring of upside-down stairs just under the tower room, so
        # its top reads as machicolated (kept below the wall-walk's height so
        # their tops are not mistaken for it).
        for (x, z) in disc(cx, cz, R + 1.0) - drum:
            dx, dz = x + 0.5 - cx, z + 0.5 - cz
            f = ("west" if dx > 0 else "east") if abs(dx) >= abs(dz) else ("north" if dz > 0 else "south")
            if b.get(x, Hw - 2, z) is None:
                b.set(x, Hw - 2, z, state("stone_brick_stairs", facing=f, half="top"))
        # Arrow slits around the drum, at the room.
        for (x, z) in wall_ring:
            ang = math.atan2(z + 0.5 - cz, x + 0.5 - cx)
            if abs(math.sin(4 * ang)) < 0.08:
                b.set(x, Hw + 1, z, state("glass_pane"))
                b.set(x, Hw + 2, z, state("glass_pane"))
        b.set(int(math.floor(cx)), room_top - 1, int(math.floor(cz)), state("lantern", hanging="true"))

    # The wall-walk lanes: rows 1..walk inside the outer face, clear for the
    # body; cut through the tower rings as doorways.
    walk_cells: Set[Cell] = set()
    for x in range(0, X + 1):
        for t in range(1, T):
            walk_cells.add((x, t))
            walk_cells.add((x, Z - t))
    for z in range(0, Z + 1):
        for t in range(1, T):
            walk_cells.add((t, z))
            walk_cells.add((X - t, z))
    for (x, z) in walk_cells:
        for y in range(Hw, Hw + s.door_height):
            blk = b.get(x, y, z)
            if blk is not None and "lantern" not in blk:
                b.set(x, y, z, None)
    # Record the tower doorways where the walks cross the drums.
    for (cx, cz) in corners:
        ring = circle(cx, cz, R, 1.0)
        crossings = sorted(c for c in ring if c in walk_cells)
        for axis in ("x", "z"):
            group = [c for c in crossings if (abs(c[1] - cz) < R - 0.6 if axis == "x" else abs(c[0] - cx) < R - 0.6)]
            if not group:
                continue
        for side in ("east", "west", "north", "south"):
            dx, dz = DIRS[side]
            cells = [c for c in crossings if (c[0] - cx) * dx + (c[1] - cz) * dz > R * 0.6]
            if not cells:
                continue
            xs = sorted({c[0] for c in cells}) if side in ("north", "south") else None
            zs = sorted({c[1] for c in cells}) if side in ("east", "west") else None
            mx = sum(c[0] for c in cells) / len(cells) + 0.5
            mz = sum(c[1] for c in cells) / len(cells) + 0.5
            b.ports.append({"id": f"port.tower.{len(b.ports):03d}", "source": "declared",
                            "position": [round(mx, 4), float(Hw), round(mz, 4)],
                            "facing": [float(dx), 0.0, float(dz)],
                            "wall_axis": "z" if side in ("east", "west") else "x",
                            "width": float(walk - 0), "height": float(s.door_height),
                            "blocks": [walk, s.door_height], "profile": b.player.name})

    # Gatehouse in the south wall: two square towers and a passage.
    gw = max(3, lane + 2)
    gh = s.door_height + 2
    gx0 = W // 2 - gw // 2
    gx1 = gx0 + gw - 1
    tw = 5
    for tx0 in (gx0 - tw, gx1 + 1):
        tz0, tz1 = Z - T - 1, Z + 2
        tx1 = tx0 + tw - 1
        for x in range(tx0, tx1 + 1):
            for z in range(tz0, tz1 + 1):
                b.set(x, 0, z, base())
                b.fill(x, 1, z, x, Hw - 1, z, stone)
        b.ring(tx0, tz0, tx1, tz1, Hw, Hw + s.ceiling - 1, stone)
        b.clear(tx0 + 1, Hw, tz0 + 1, tx1 - 1, Hw + s.ceiling - 1, tz1 - 1)
        for x in range(tx0, tx1 + 1):
            for z in (tz0, tz1):
                b.set(x, Hw - 1, z, state(trim))
        b.hip_roof(tx0, tz0, tx1, tz1, Hw + s.ceiling, roof_stairs, roof_slab, overhang=1, solid=roof_full)
        b.set(tx0 + tw // 2, Hw + s.ceiling - 1, (tz0 + tz1) // 2, state("lantern", hanging="true"))
        # Doors from the wall-walk on either side.
        for side, xx in (("east", tx0), ("west", tx1)):
            pass
        # Banners on the outer face.
        for bx in (tx0 + 1, tx1 - 1):
            b.set(bx, Hw - 2, tz1 + 1, state("red_wall_banner", facing="south"))
            b.set(bx, Hw - 3, tz1 + 1, None)
        # Arrow slits facing out.
        b.set(tx0 + tw // 2, Hw + 1, tz1, state("glass_pane"))
        b.set(tx0 + tw // 2, Hw + 2, tz1, state("glass_pane"))
    # The wall-walk through the gate towers.
    for x in range(gx0 - tw, gx1 + tw + 1):
        for t in range(1, T):
            for y in range(Hw, Hw + s.door_height):
                blk = b.get(x, y, Z - t)
                if blk is not None and kind(blk) == "full":
                    b.set(x, y, Z - t, None)
    for tx0, side in ((gx0 - tw, "east"), (gx1 + 1, "west")):
        for edge, facing in ((tx0, "east"), (tx0 + tw - 1, "west")):
            b.ports.append({"id": f"port.gatehouse.{len(b.ports):03d}", "source": "declared",
                            "position": [edge + 0.5, float(Hw), Z - T / 2 + 0.5],
                            "facing": [float(DIRS[OPPOSITE[facing]][0]), 0.0, 0.0], "wall_axis": "z",
                            "width": float(walk), "height": float(s.door_height),
                            "blocks": [walk, s.door_height], "profile": b.player.name})
    # The passage, with an arch of upside-down stairs and a raised portcullis.
    for x in range(gx0, gx1 + 1):
        for z in range(Z - T - 1, Z + 3):
            b.set(x, 0, z, None)
            b.path(x, z, "gravel")
            for y in range(1, gh):
                b.set(x, y, z, None)
    for z in (Z + 2, Z - T - 1):
        b.set(gx0, gh - 1, z, state("stone_brick_stairs", facing="west", half="top"))
        b.set(gx1, gh - 1, z, state("stone_brick_stairs", facing="east", half="top"))
    for x in range(gx0, gx1 + 1):
        b.set(x, gh, Z + 1, state("iron_bars"))
    b.ports.append({"id": f"port.gate.{len(b.ports):03d}", "source": "declared",
                    "position": [(gx0 + gx1 + 1) / 2.0, 0.0, Z + 3.0], "facing": [0.0, 0.0, 1.0],
                    "wall_axis": "x", "width": float(gw), "height": float(gh - 1),
                    "blocks": [gw, gh - 1], "profile": b.player.name})

    # Stairs from the bailey to the wall-walk, against the inner face of the
    # west and east walls, climbing north.
    for wx, along in ((T, "east"), (X - T - lane + 1, "east")):
        z_top = int(math.ceil(0.5 + T / 2 + R)) + 1     # clear of the corner tower
        z_first = z_top + Hw - 1
        b.stair_flight(wx, 0, z_first, "north", Hw, "stone_brick_stairs", width=lane, along=along,
                       support=None)
        for i in range(Hw):
            for w in range(lane):
                for y in range(0, i):
                    b.set(wx + w, y, z_first - i, stone())
        b.notes.append(f"castle: stair to the wall-walk at x {wx}")

    # The keep, at the back of the bailey.
    kx0 = W // 2 - keep_w // 2
    kz0 = T + 4
    kx1, kz1 = kx0 + keep_w - 1, kz0 + keep_w - 1
    floors = 3
    # The great hall (level 1) stands two blocks taller than a storey.
    rises = [S, S + 2, S]
    fy = [0, S, 2 * S + 2, 3 * S + 2]    # floor layer of each level and the roof deck
    K_top = fy[floors]                   # roof deck floor layer
    b.fill(kx0, 0, kz0, kx1, 0, kz1, base)
    for t in range(2):
        b.ring(kx0 + t, kz0 + t, kx1 - t, kz1 - t, 1, K_top + 1, stone)
    for (x, z) in ((kx0, kz0), (kx1, kz0), (kx0, kz1), (kx1, kz1)):
        b.fill(x, 1, z, x, K_top + 1, z, state("chiseled_stone_bricks"))
    ix0, iz0, ix1, iz1 = kx0 + 2, kz0 + 2, kx1 - 2, kz1 - 2
    for k in range(1, floors + 1):
        b.fill(ix0, fy[k], iz0, ix1, fy[k], iz1, state("spruce_planks") if k < floors else stone())
        for x in range(kx0, kx1 + 1):
            for z in (kz0, kz1):
                b.set(x, fy[k], z, state(trim))
        for z in range(kz0, kz1 + 1):
            for x in (kx0, kx1):
                b.set(x, fy[k], z, state(trim))
    b.fill(ix0, 0, iz0, ix1, 0, iz1, state("polished_andesite"))
    # Crenellations round the roof deck.
    for x in range(kx0, kx1 + 1):
        for z in (kz0, kz1):
            if (x + z) % 2 == 0:
                b.set(x, K_top + 2, z, stone())
    for z in range(kz0, kz1 + 1):
        for x in (kx0, kx1):
            if (x + z) % 2 == 0:
                b.set(x, K_top + 2, z, stone())
    b.fill(kx0 + 1, K_top + 1, kz0 + 1, kx1 - 1, K_top + 1, kz1 - 1, None)
    b.fill(kx0 + 2, K_top, kz0 + 2, kx1 - 2, K_top, kz1 - 2, stone)
    b.fill(kx0 + 1, K_top, kz0 + 1, kx1 - 1, K_top, kz0 + 1, stone)
    # Corner turrets on the roof, with small cones.
    for (cx, cz) in ((kx0 + 1.5, kz0 + 1.5), (kx1 - 0.5, kz0 + 1.5), (kx0 + 1.5, kz1 - 0.5), (kx1 - 0.5, kz1 - 0.5)):
        for (x, z) in disc(cx, cz, 1.6):
            b.fill(x, K_top + 1, z, x, K_top + 3, z, stone)
        b.spire(cx, cz, 2.2, K_top + 4, roof_stairs, roof_full)
    # Door in the south face, with a step.
    kdx = (kx0 + kx1) // 2 - lane // 2
    b.door(kdx, 1, kz1 - 1, "north", wood="dark_oak", along="east", name="keep")
    for w in range(lane):
        b.set(kdx + w, 1, kz1, None)
        b.set(kdx + w, 2, kz1, None)
        if s.door_height > 2:
            for y in range(3, 1 + s.door_height):
                b.set(kdx + w, y, kz1, None)
        b.set(kdx + w, 0, kz1 + 1, state("stone_brick_stairs", facing="north", half="bottom"))
        b.set(kdx + w, 0, kz1, state("polished_andesite"))
        b.set(kdx + w, 0, kz1 - 1, state("polished_andesite"))
    for y in range(1, 1 + s.door_height):
        pass
    b.set(kdx - 1, 1 + s.door_height, kz1 + 1, state("wall_torch", facing="south"))
    b.set(kdx + lane, 1 + s.door_height, kz1 + 1, state("wall_torch", facing="south"))
    # Windows: slits on the ground floor, pairs of panes above, tall ones
    # in the hall.
    for k in range(floors):
        y = fy[k] + 2
        tall = 1 if k == 0 else rises[k] - 2
        for x in range(ix0 + 1, ix1, 3):
            for z, f in ((kz0, "north"), (kz1, "south")):
                if k == 0 and abs(x - kdx) <= lane:
                    continue
                for j in range(tall):
                    b.set(x, y + j, z, state("glass_pane"))
                    inner = z + 1 if f == "north" else z - 1
                    b.set(x, y + j, inner, None)
        for z in range(iz0 + 1, iz1, 3):
            for x in (kx0, kx1):
                for j in range(tall):
                    b.set(x, y + j, z, state("glass_pane"))
                    b.set(x + (1 if x == kx0 else -1), y + j, z, None)
    # Banners on the keep's faces.
    for x in (kx0 + 3, kx1 - 3):
        b.set(x, K_top - 1, kz1 + 1, state("blue_wall_banner", facing="south"))
        b.set(x, K_top - 1, kz0 - 1, state("blue_wall_banner", facing="north"))
    # Stairs: flights along the west and east inner walls, alternating, the
    # last through a hatch to the roof deck.
    keeps: Dict[int, Set[Cell]] = {k: set() for k in range(floors + 1)}
    arrivals: Dict[int, List[Cell]] = {0: [(kdx + w, iz1) for w in range(lane)]}
    for k in range(floors):
        yb = fy[k] + 1
        R = rises[k]
        if k % 2 == 0:
            fx, fz, facing, along = ix0, iz1 - 1, "north", "east"
        else:
            fx, fz, facing, along = ix1 - lane + 1, iz0 + 1 + S - 1, "south", "east"
            fz = iz0 + 1
            facing = "south"
        b.stair_flight(fx, yb, fz, facing, R, "spruce_stairs", width=lane, along=along)
        dz = DIRS[facing][1]
        hb = s.headroom
        for i in range(max(0, R - 2 - hb), R - 1):
            for w in range(lane):
                b.set(fx + w, fy[k + 1], fz + i * dz, None)
        for i in range(-1, R + 1):
            for w in range(lane):
                keeps[k].add((fx + w, fz + i * dz))
        for i in range(R - 2 - hb, R + 1):
            for w in range(lane):
                keeps[k + 1].add((fx + w, fz + i * dz))
        arrivals[k + 1] = [(fx + w, fz + R * dz) for w in range(lane)]
    hall_items: List[Any] = [state("barrel", facing="up"), _on_barrel(state("lantern", hanging="false"))]
    halls: Dict[int, Set[Cell]] = {}
    for k in (0, 1):
        # The guard room below and the great hall above get a long table.
        halls[k] = great_hall(b, (ix0, iz0, ix1, iz1), fy[k] + 1, fy[k + 1] - 1, lane, keeps[k], arrivals[k],
                              rng, throne=(k == 1))
    rooms = {
        0: [state("barrel", facing="up"), state("barrel", facing="up"), state("chest", facing="south"),
            state("hay_block", axis="y"), state("barrel", facing="up"), state("chest", facing="north"),
            state("anvil", facing="east"), state("barrel", facing="up")],
        1: [_table("spruce"), _table("spruce"), state("crafting_table"),
            state("lectern", facing="south"), state("barrel", facing="up")],
        2: [_bed("blue", "candle"),
            state("bookshelf"), state("bookshelf"), state("chest", facing="south"), state("bookshelf"),
            state("cartography_table")],
    }
    for k in range(floors):
        aisles = set()
        for (x, z) in halls.get(k, set()):
            # Keep a lane clear on each side of the table.
            for dx in range(-lane, lane + 1):
                aisles.add((x + dx, z))
        furnish(b, (ix0, iz0, ix1, iz1), fy[k] + 1, rooms[k] + hall_items, keeps[k] | aisles, arrivals[k])
        cx, cz = (ix0 + ix1) // 2, (iz0 + iz1) // 2
        top = fy[k + 1] - 1
        if k not in halls:
            b.set(cx, top, cz, state("chain", axis="y"), replace=False)
            b.set(cx, top - 1, cz, state("lantern", hanging="true"), replace=False)
        for _ in range(2 + 2 * (k == 1)):
            wall_banner(b, (ix0, iz0, ix1, iz1), fy[k] + 1, top, rng)
        ceiling_decor(b, (ix0, iz0, ix1, iz1), fy[k] + 1, top, rng, set().union(*keeps.values()),
                      head_clear=False)
    # Hatch onto the roof deck is the last flight's opening; the deck is
    # stone, its trim the parapet.

    # Lanterns hung from stone brackets along both faces of the curtain.
    def bracket_lantern(x: int, z: int, out: str) -> None:
        dx, dz = DIRS[out]
        bx, bz = x + dx, z + dz
        yb = Hw - 3
        if any(b.get(bx, yy, bz) is not None for yy in range(yb - 3, yb + 1)):
            return
        b.set(bx, yb, bz, state("stone_brick_stairs", facing=OPPOSITE[out], half="top"))
        b.set(bx, yb - 1, bz, state("lantern", hanging="true"))
    for x in range(6, X - 5, 7):
        bracket_lantern(x, 0, "north")
        bracket_lantern(x, Z, "south")
        bracket_lantern(x, T - 1, "south")
        bracket_lantern(x, Z - T + 1, "north")
    for z in range(6, Z - 5, 7):
        bracket_lantern(0, z, "west")
        bracket_lantern(X, z, "east")
        bracket_lantern(T - 1, z, "east")
        bracket_lantern(X - T + 1, z, "west")
    # A balcony on the keep's front over the door, at the hall: a stone
    # slab floor on corbels with a fence rail, lanterns at its corners.
    yb = fy[1]
    bx0, bx1 = kdx - 2, kdx + lane + 1
    for x in range(bx0, bx1 + 1):
        b.set(x, yb, kz1 + 1, state("polished_andesite"))
        b.set(x, yb - 1, kz1 + 1, state("stone_brick_stairs", facing="north", half="top"))
        b.set(x, yb, kz1 + 2, state("stone_brick_slab", type="top"))
        b.set(x, yb + 1, kz1 + 2, state("dark_oak_fence"))
    for x in (bx0 - 1, bx1 + 1):
        b.set(x, yb, kz1 + 1, state("stone_brick_slab", type="top"))
        b.set(x, yb + 1, kz1 + 1, state("dark_oak_fence"))
        b.set(x, yb + 2, kz1 + 1, state("lantern", hanging="false"))
    # The hall opens onto it where nothing in the hall stands in the way.
    if all(b.get(x, y, kz1 - 2) is None for x in range(kdx, kdx + lane)
           for y in range(yb + 1, yb + 1 + s.door_height)):
        for x in range(kdx, kdx + lane):
            for y in range(yb + 1, yb + 1 + s.door_height):
                b.set(x, y, kz1, None)
                b.set(x, y, kz1 - 1, None)
        b.door(kdx, yb + 1, kz1, "south", wood="dark_oak", along="east", name="keep_balcony")

    # Timber work, as castles that grew a town about them have it: the top of
    # the keep is timber-framed, and stretches of the curtain carry roofed
    # timber hoardings over the wall-walk, their framed face where the
    # parapet was, on joists out of the outer face.
    frame_wood = rng.choice(("spruce", "dark_oak", "oak"))
    post_b = f"stripped_{frame_wood}_log" if frame_wood != "oak" else "dark_oak_log"
    panel_b = rng.choice(("white_terracotta", "calcite", "white_terracotta", f"{frame_wood}_planks"))

    def framed(x: int, y: int, z: int, along: int, bottom: int, top_: int) -> None:
        """A block of a timber-framed face: posts every third block along it
        and at its ends, a rail at its foot, plaster or planks between."""
        cur = b.get(x, y, z)
        if cur is not None and kind(cur) == "pane":
            return
        if along % 3 == 0 or y == bottom:
            b.set(x, y, z, state(post_b, axis="y" if along % 3 == 0 else "x"))
        elif (along + y) % 5 == 0 and bottom < y < top_:
            b.set(x, y, z, state(f"{frame_wood}_planks"))
        else:
            b.set(x, y, z, state(panel_b))

    # The keep's top level: its outer skin framed, windows kept.
    for yy in range(fy[2] + 1, fy[3]):
        for x in range(kx0 + 1, kx1):
            for z in (kz0, kz1):
                framed(x, yy, z, x - kx0, fy[2] + 1, fy[3] - 1)
        for z in range(kz0 + 1, kz1):
            for x in (kx0, kx1):
                framed(x, yy, z, z - kz0, fy[2] + 1, fy[3] - 1)
    hd = s.door_height
    hoard_top = Hw + hd + 1                  # eave layer over the wall-walk
    tower_clear = int(math.ceil(T / 2 + R)) + 3
    stretches = []
    span_x = (tower_clear, X - tower_clear)
    span_z = (tower_clear + Hw, Z - T - 4)   # clear of the bailey stairs' heads and the gatehouse
    if span_x[1] - span_x[0] >= 8:
        L = min(13, span_x[1] - span_x[0])
        a0 = (span_x[0] + span_x[1] - L) // 2
        stretches.append(("north", a0, a0 + L - 1))
    if span_z[1] - span_z[0] >= 7:
        L = min(11, span_z[1] - span_z[0])
        a0 = (span_z[0] + span_z[1] - L) // 2
        for side in ("west", "east"):
            if rng.random() < 0.8:
                stretches.append((side, a0, a0 + L - 1))
    # Either side of the gatehouse on the front wall.
    for lo_, hi_ in ((tower_clear - 1, gx0 - tw - 1), (gx1 + tw + 1, X - tower_clear + 1)):
        if hi_ - lo_ >= 4:
            stretches.append(("south", lo_, hi_))
    for side, a0, a1 in stretches:
        for a in range(a0, a1 + 1):
            if side == "north":
                outer, joist, facing = (a, 0), (a, -1), "south"
            elif side == "west":
                outer, joist, facing = (0, a), (-1, a), "east"
            elif side == "south":
                outer, joist, facing = (a, Z), (a, Z + 1), "north"
            else:
                outer, joist, facing = (X, a), (X + 1, a), "west"
            for yy in range(Hw, hoard_top):
                framed(outer[0], yy, outer[1], a - a0, Hw, hoard_top - 1)
            # (below the wall-walk's height, like the towers' corbels, so
            # their tops are not taken for it)
            b.set(joist[0], Hw - 2, joist[1], state(f"{frame_wood}_stairs", facing=facing, half="top"))
        if side == "north":
            b.gable_roof(a0, 0, a1, T - 1, hoard_top, roof_stairs, roof_slab, state(panel_b), overhang=1, axis="x")
        elif side == "south":
            b.gable_roof(a0, Z - T + 1, a1, Z, hoard_top, roof_stairs, roof_slab, state(panel_b), overhang=1,
                         axis="x")
        elif side == "west":
            b.gable_roof(0, a0, T - 1, a1, hoard_top, roof_stairs, roof_slab, state(panel_b), overhang=1, axis="z")
        else:
            b.gable_roof(X - T + 1, a0, X, a1, hoard_top, roof_stairs, roof_slab, state(panel_b), overhang=1,
                         axis="z")
        # A lantern under each hoarding's roof, over the walk.
        mid = (a0 + a1) // 2
        pos = {"north": (mid, 1), "south": (mid, Z - 1), "west": (1, mid), "east": (X - 1, mid)}[side]
        b.set(pos[0], hoard_top - 1, pos[1], state("lantern", hanging="true"))

    # Vines: strands hanging down the outer faces of the curtain, the towers
    # and the keep, each clinging to the stone behind it.
    def hang(cells: Iterable[Tuple[int, int, str]], y_top: int, share: float) -> None:
        for (x, z, cling) in cells:
            if rng.random() >= share:
                continue
            for yy in range(y_top, max(0, y_top - rng.randint(3, 8)), -1):
                wall = b.get(x + DIRS[cling][0], yy, z + DIRS[cling][1])
                if b.get(x, yy, z) is not None or wall is None or kind(wall) != "full":
                    break
                b.set(x, yy, z, state("vine", **{cling: "true"}))
    outside: List[Tuple[int, int, str]] = []
    outside += [(x, -1, "south") for x in range(0, X + 1)] + [(x, Z + 1, "north") for x in range(0, X + 1)]
    outside += [(-1, z, "east") for z in range(0, Z + 1)] + [(X + 1, z, "west") for z in range(0, Z + 1)]
    hang(outside, Hw - 2, 0.16)
    for (cx, cz) in corners:
        drum = disc(cx, cz, R)
        ring_out = []
        for (x, z) in disc(cx, cz, R + 1.0) - drum:
            for d, (ddx, ddz) in DIRS.items():
                if (x + ddx, z + ddz) in drum:
                    ring_out.append((x, z, d))
                    break
        hang(ring_out, Hw - 3, 0.2)
    keep_out = ([(x, kz0 - 1, "south") for x in range(kx0, kx1 + 1)] +
                [(kx0 - 1, z, "east") for z in range(kz0, kz1 + 1)] +
                [(kx1 + 1, z, "west") for z in range(kz0, kz1 + 1)])
    hang(keep_out, fy[2], 0.18)

    # The bailey: a path from the gate to the keep, a well, houses along the
    # walls clear of the stairs, hay and barrels.
    for z in range(kz1 + 1, Z - T):
        for x in range(W // 2 - 1, W // 2 + 2):
            if b.get(x, 0, z) is None:
                b.path(x, z, "dirt_path" if rng.random() > 0.2 else "gravel")
    well(b, kx1 + 4, kz1 + 3)
    hx = T + lane + 2
    placed = 0
    for i, (px, pz, quarter) in enumerate(((T + lane + 1, Z - T - 1, 2), (X - T - lane - 1, Z - T - 1, 2))):
        h = house(player, seed * 31 + i, storeys=1, name=f"bailey{i}")
        (lx, _, lz), (hx2, _, hz2) = h.bounds()
        if quarter == 2:
            ox = px + (hx2 - lx) + 1 if i == 0 else px
            ox = px + (hx2 + 1) if i == 0 else px
            paste(b, h, ox, pz, quarter=2)
        placed += 1
    for (x, z) in ((kx0 - 3, kz1 - 2), (kx1 + 3, kz0 + 2)):
        b.set(x, 0, z, state("hay_block", axis="y"))
        b.set(x, 1, z, state("hay_block", axis="y"))
    for x in (W // 2 - 3, W // 2 + 3):
        lamp_post(b, x, Z - T - 6)
    from .blockdress import fill_containers, pattern_banners
    fill_containers(b, "castle")
    pattern_banners(b)
    b.notes.append(f"castle: {W}x{D}, curtain {T} thick, wall-walk at {Hw}, keep {keep_w} square, "
                   f"{floors} floors of {S}")
    return b


BLOCK_PRODUCTS = {
    "house": lambda player, seed: house(player, seed),
    "village": lambda player, seed: village(player, seed),
    "castle": lambda player, seed: castle(player, seed),
}


def _town(player: Any, seed: int) -> BlockBuild:
    from .blocktown import town
    return town(player, seed)


BLOCK_PRODUCTS["town"] = _town


def _kind(name: str):
    def make(player: Any, seed: int) -> BlockBuild:
        from . import blockkinds
        return getattr(blockkinds, name)(player, seed)
    return make


for _name in ("tower", "workshop", "windmill", "mine"):
    BLOCK_PRODUCTS[_name] = _kind(_name)

# Mesh-mode product names that block mode answers for.
BLOCK_ALIASES = {
    "norman-castle": "castle",
    "english-village": "village",
    "english-village:market-town": "village",
    "village-house": "house",
    "castle-town": "town",
}


def build(product: str, player: Any = "blocky", seed: int = 0) -> BlockBuild:
    """Build ``product`` in block mode."""
    key = BLOCK_ALIASES.get(product, product)
    if key not in BLOCK_PRODUCTS:
        raise ValueError(f"no block-mode architect for '{product}'; block mode has: "
                         f"{', '.join(sorted(set(BLOCK_PRODUCTS) | set(BLOCK_ALIASES)))}")
    return BLOCK_PRODUCTS[key](player, seed)


__all__ = ["paste", "house", "village", "castle", "tree", "well", "lamp_post", "furnish", "build",
           "BLOCK_PRODUCTS", "BLOCK_ALIASES", "HOUSE_STYLES"]
