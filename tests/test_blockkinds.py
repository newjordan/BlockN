"""Block-mode building types (tower, workshop, windmill), the mine, and the
set dressing the Minecraft export carries as block entities and entities."""

import numpy as np

from core.architecture import blockcraft, blockkinds
from core.architecture.blockmode import kind, parse
from core.architecture.gameplay import walk


def _floors_walked(b):
    w = walk(b.grid(b.walk_sub()), b.player)
    g = b.grid(b.walk_sub())
    y = g.origin[1] + w.nodes[:, 1] * g.pitch
    S = b.sizes.storey
    floors = 0
    for k in range(5):
        on = np.isclose(y, 1 + k * S) & w.interior
        if on.any():
            floors += 1
            # A sub-block walker node squeezed against a wall between two
            # pieces of furniture can be left over; no floor area is.
            assert int((~w.reached[on]).sum()) <= 2, (b.kind, k, int((~w.reached[on]).sum()))
    return floors


def test_tower_workshop_and_windmill_floors_are_walked():
    for profile in ("blocky", "giant"):
        t = blockkinds.tower(profile, 1)
        assert t.kind == "tower" and _floors_walked(t) >= 3
        w = blockkinds.workshop(profile, 2, trade="smith")
        assert w.kind == "workshop" and _floors_walked(w) >= 1
        assert any(parse(v)[0] in ("anvil", "smithing_table", "blast_furnace") for v in w.blocks.values())
        m = blockkinds.windmill(profile, 0)
        assert m.kind == "windmill" and _floors_walked(m) >= 3
        assert sum(parse(v)[0] == "white_wool" for v in m.blocks.values()) > 8


def test_apartments_floors_are_walked():
    for profile in ("blocky", "giant"):
        for seed in (0, 2):
            a = blockkinds.apartment(profile, seed)
            assert a.kind == "apartment" and a.storeys >= 3
            assert _floors_walked(a) == min(a.storeys, 5)


def test_world_save_round_trips_blocks_and_block_entities(tmp_path):
    from core.architecture import blockworld
    b = blockcraft.house("blocky", 1, storeys=2)
    info = blockworld.to_world(b, str(tmp_path / "w"), dy=64)
    assert (tmp_path / "w" / "level.dat").exists() and info["entities"] == len(b.entities)
    chunk = blockworld.read_chunk(str(tmp_path / "w" / "region" / "r.0.0.mca"), 0, 0)
    for (x, y, z), blk in b.blocks.items():
        if 0 <= x < 16 and 0 <= z < 16:
            assert blockworld.block_at(chunk, x, y + 64, z) == "minecraft:" + parse(blk)[0]
    chests = [t for t in chunk.value["block_entities"].value if t.value["id"].value == "minecraft:chest"]
    assert any("Items" in t.value for t in chests)
    assert blockworld.snbt('{a:1b,b:[{c:"x"}],d:2.5f}').value["b"].value[0].value["c"].value == "x"


def test_mine_is_walked_for_every_body():
    for profile, seeds in (("blocky", (0, 1)), ("giant", (0,)), ("human", (0,))):
        for seed in seeds:
            b = blockkinds.mine(profile, seed)
            info = b.mine
            # A giant's passages are wider and taller than a blocky's, and a
            # body that can't climb stairs gets half-block steps.
            assert info["width"] >= b.sizes.lane + 2 and info["clear"] >= b.sizes.door_height + 1
            assert info["stairs"] == ("stair blocks" if b.sizes.climbs_stairs else "half-block steps")
            r = blockkinds.mine_walk(b)
            assert r["floors"] > 50 and r["reached"] == r["floors"], (profile, seed, r["missing"][:4])
            assert b.ports and any(e["id"] == "minecraft:chest_minecart" for e in b.entities)


def test_export_carries_chest_contents_banners_frames_and_paintings():
    b = blockcraft.house("blocky", 1, storeys=2)
    text = b.to_mcfunction()
    assert "{Items:[{Slot:" in text
    assert "summon minecraft:item_frame" in text
    assert any(e["id"] == "minecraft:painting" for e in b.entities)
    # Entities come after every block, so they have a wall to hang on.
    last_block = max(i for i, line in enumerate(text.splitlines()) if line.startswith(("setblock", "fill")))
    first_summon = min(i for i, line in enumerate(text.splitlines()) if line.startswith("summon"))
    assert first_summon > last_block
    c = blockcraft.castle("blocky", 2)
    assert "{patterns:[{pattern:" in c.to_mcfunction()


def test_valley_wall_town_has_building_types_and_a_mine():
    from core.architecture import blocktown
    b = blocktown.town("blocky", 1, terrain="valley-wall")
    plan = b.town
    assert plan["kinds"].get("tower", 0) >= 1 and plan["kinds"].get("windmill", 0) >= 1
    assert plan["mine"] and plan["mine"]["cover_margin"] >= 0
    # Flats stand against the faces that rise behind street fronts, most
    # with a door out of the back onto the ground above.
    assert plan["kinds"].get("apartment", 0) >= 5 and plan["back_doors"] >= 3
    # Street life: a fountain on the square, props and benches off the streets.
    assert plan["counts"]["street_props"] >= 40 and plan["counts"]["benches"] >= 10
    assert any(kind(v) == "water" for (x, y, z), v in b.blocks.items() if y > plan["square"]["T"] + 2)
    w, report = blocktown.town_walk(b)
    floors, reached = (int(x) for x in report["mine_floors_reached"].split("/"))
    assert floors > 50 and reached == floors
    assert report["houses_every_floor_reached"] == report["houses"]
