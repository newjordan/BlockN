"""Block-mode towns fitted to a given height map."""

import numpy as np

from core.architecture import blocktown
from core.architecture.blockcraft import build


def test_sample_heights_are_whole_blocks():
    for kind in ("hill", "slope", "valley", "valley-wall"):
        h = blocktown.sample_heights(kind, 96, 2)
        assert h.shape == (96, 96) and h.dtype.kind == "i" and h.min() >= 2


def test_town_fits_the_ground_and_every_house_is_walked():
    heights = blocktown.sample_heights("slope", 144, 1)
    b = blocktown.town("blocky", 1, heights=heights)
    plan = b.town
    assert plan["counts"]["houses"] >= 30
    # The input map is only changed where something was built on it.
    assert (plan["heights_in"] == heights).all()
    # Streets never step more than one block between cells along their line.
    for street in plan["streets"]:
        assert street["length"] > 0 and street["max_step"] <= 1, street
    w, report = blocktown.town_walk(b)
    assert report["houses_every_floor_reached"] == report["houses"]
    assert report["castle_wall_walk_reached_share"] > 0.95
    assert report["castle_bailey_reached"]


def test_town_climbs_a_valley_wall_in_terraces():
    b = blocktown.town("blocky", 2, terrain="valley-wall")
    plan = b.town
    # The castle sits at the foot, not perched on a platform built up the wall.
    assert plan["castle"]["T"] < 20
    assert plan["terraces"] >= 2 and plan["switchbacks"] >= 1
    terraces = [s for s in plan["streets"] if s["name"].startswith(("terrace", "switchback"))]
    assert all(s["max_step"] <= 1 for s in terraces)
    assert plan["counts"]["houses"] >= 50


def test_house_rooms_differ_by_seed():
    from core.architecture.blockcraft import house
    seen = {house("blocky", seed, storeys=2).notes[-1].split("house, rooms in ")[-1] for seed in range(6)}
    assert len(seen) >= 5


def test_town_is_a_block_product():
    b = build("castle-town", "blocky", 3)
    assert b.product == "town" and b.town["counts"]["houses"] > 20
    assert "fill ~" in b.to_mcfunction()
