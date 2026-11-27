"""Block mode: architects that place Minecraft blocks on the 1 m grid."""

import numpy as np

from core.architecture import blockcraft
from core.architecture.blockmode import BlockBuild, BlockSizes, kind, parse, shape, state
from core.architecture.gameplay import walk


def _walk(b):
    return walk(b.grid(b.walk_sub()), b.player)


def test_block_states_round_trip():
    s = state("oak_stairs", facing="east", half="bottom")
    assert s == "minecraft:oak_stairs[facing=east,half=bottom]"
    assert parse(s) == ("oak_stairs", {"facing": "east", "half": "bottom"})
    assert kind(s) == "stairs"
    assert kind("minecraft:red_wall_banner[facing=south]") == "wall_banner"
    assert kind("minecraft:cobblestone_wall") == "wall"
    assert kind("minecraft:wall_torch[facing=north]") == "torch"


def test_shapes_are_true_block_shapes():
    slab = shape(state("oak_slab", type="bottom"))
    assert slab[:, :2, :].all() and not slab[:, 2:, :].any()
    stair = shape(state("oak_stairs", facing="east", half="bottom"))
    # Lower half full, upper half only on the east (facing) side.
    assert stair[:, :2, :].all()
    assert stair[2:, 2:, :].all() and not stair[:2, 2:, :].any()
    assert not shape(state("poppy")).any()
    assert shape(state("poppy"), visual=True).any()
    assert not shape(state("oak_door", facing="south", half="lower")).any()


def test_sizes_are_whole_blocks_from_the_profile():
    blocky = BlockSizes.of("blocky")
    assert (blocky.door_width, blocky.door_height) == (1, 2)
    assert blocky.climbs_stairs
    giant = BlockSizes.of("giant")
    assert (giant.door_width, giant.door_height) == (2, 3)
    assert giant.storey > blocky.storey


def test_house_is_walked_end_to_end_for_blocky_and_giant():
    for profile, seed in (("blocky", 1), ("blocky", 5), ("giant", 3)):
        b = blockcraft.house(profile, seed, storeys=2)
        w = _walk(b)
        summary = w.summary()
        # Every floor is walked from the door, to every cell the furniture
        # leaves free; the tops of furniture are left to jumps.
        g = b.grid(b.walk_sub())
        y = g.origin[1] + w.nodes[:, 1] * g.pitch
        S = b.sizes.storey
        on_floor = np.isin(np.round(y, 3), [1.0, S + 1.0])
        assert (w.reached[on_floor & w.interior]).all(), (profile, seed, summary)
        # Decoration (a storage loft under the roof, the tops of tall
        # shelves) need not be reached: the check is floors and routes.
        assert summary["interior_reached_share"] >= 0.6, (profile, seed, summary)
        door = b.ports[0]
        assert door["blocks"] == [b.sizes.door_width, b.sizes.door_height]
        assert door["profile"] == profile


def test_minecraft_export_keeps_states_and_order():
    b = blockcraft.house("blocky", 1, storeys=2)
    text = b.to_mcfunction()
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    assert any("_stairs[facing=" in l for l in lines)
    first_door = min(i for i, l in enumerate(lines) if "_door[" in l)
    last_full = max(i for i, l in enumerate(lines) if "cobblestone" in l and "wall" not in l)
    assert last_full < first_door
    grid = b.grid(1)
    assert grid.pitch == 1.0
    assert any("[" in e["minecraft"] for e in grid.palette)
    assert grid.to_mcfunction().count("\n") > 10


def test_paste_turns_states_with_the_part():
    part = BlockBuild("part", "blocky")
    part.set(0, 0, 0, state("oak_stairs", facing="north", half="bottom"))
    part.set(0, 0, 1, state("oak_log", axis="x"))
    whole = BlockBuild("whole", "blocky")
    blockcraft.paste(whole, part, 10, 10, quarter=1)
    blocks = sorted(whole.blocks.items())
    names = {parse(b)[0]: parse(b)[1] for _, b in blocks}
    assert names["oak_stairs"]["facing"] == "east"
    assert names["oak_log"]["axis"] == "z"


def test_castle_wall_walk_and_keep_roof_are_reached():
    b = blockcraft.castle("blocky", 1)
    g = b.grid(b.walk_sub())
    w = walk(g, b.player)
    sub = b.walk_sub()
    heights = w.nodes[:, 1] / sub
    hw = max(8, 2 * b.sizes.storey)
    on_walk = np.abs(heights - hw) < 1e-6
    assert on_walk.sum() > 500
    assert w.reached[on_walk].mean() > 0.95
    assert w.summary()["interior_reached_share"] > 0.95


def test_keep_floors_and_great_hall_are_walked():
    b = blockcraft.castle("blocky", 1)
    cols = {}
    for (x, y, z), blk in b.blocks.items():
        if "chiseled_stone_bricks" in blk:
            cols[(x, z)] = cols.get((x, z), 0) + 1
    tall = [c for c, n in cols.items() if n > 10]
    x0, x1 = min(c[0] for c in tall), max(c[0] for c in tall)
    z0, z1 = min(c[1] for c in tall), max(c[1] for c in tall)
    sub = b.walk_sub()
    g = b.grid(sub)
    w = walk(g, b.player)
    ox, oy, oz = g.origin
    wx, wy, wz = (w.nodes[:, 0] / sub + ox, w.nodes[:, 1] / sub + oy, w.nodes[:, 2] / sub + oz)
    inside = (wx > x0 + 2) & (wx < x1 - 1) & (wz > z0 + 2) & (wz < z1 - 1)
    S = b.sizes.storey
    for floor in (1, S + 1, 2 * S + 3):          # guard room, great hall, chamber
        here = inside & (np.abs(wy - floor) < 0.3)
        assert here.sum() > 1000 and w.reached[here].mean() == 1.0
    # The hall has its long table, throne and chandeliers.
    names = [blk for (x, y, z), blk in b.blocks.items() if y == S + 1 and x0 < x < x1 and z0 < z < z1]
    assert sum("dark_oak_slab" in n for n in names) >= 5
    assert any("dark_oak_stairs" in n for n in names)
