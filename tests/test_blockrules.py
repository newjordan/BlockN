"""The construction rulebook and its checker."""
import pytest

from core.architecture import blockrules as R
from core.architecture import lore
from core.architecture.blockcraft import house
from core.architecture.blockkinds import windmill
from core.architecture.blockmode import BlockBuild, state


def _box(floating=False):
    b = BlockBuild("box")
    b.fill(0, 0, 0, 9, 0, 7, "cobblestone")
    b.ring(0, 0, 9, 7, 1, 6, "cobblestone")
    b.fill(0, 7, 0, 9, 7, 7, "cobblestone")
    if floating:
        b.set(4, 10, 3, "cobblestone")
    return b


def test_every_tradition_uses_known_systems_coverings_and_purposes():
    for t in lore.TRADITIONS.values():
        assert t.systems and all(s in lore.SYSTEMS for s in t.systems), t.key
        assert all(c in lore.COVERINGS for c in t.coverings), t.key
        assert all(p in lore.PURPOSES for p in t.purposes), t.key
        lo, hi = t.window_ratio()
        assert 0 <= lo < hi <= 1


def test_every_rule_names_its_basis():
    for table in (lore.SYSTEMS, lore.COVERINGS, lore.TOWN):
        for key, rule in table.items():
            assert rule.get("basis"), key


def test_roof_blocks_read_as_their_covering():
    assert lore.covering_of("minecraft:deepslate_tile_stairs[facing=north]") == "slate"
    assert lore.covering_of("dark_oak_stairs") == "shingle"
    assert lore.covering_of("hay_block") == "thatch"
    assert lore.covering_of("stone") is None


def test_a_bad_box_fails_and_a_floating_block_is_found():
    r = R.score_building(_box(floating=True))
    fails = {c["rule"].split(":")[-1] for c in r["checks"] if not c["pass"]}
    assert {"load-path", "roof-pitch", "facade-depth", "flat-run", "window-ratio", "materials"} <= fails
    assert R.load_path(_box())["floating"] == 0
    assert R.load_path(_box(floating=True))["floating"] == 1


def test_a_cantilever_past_the_limit_floats_and_a_span_does_not():
    b = BlockBuild("beam")
    b.fill(0, 0, 0, 0, 3, 0, "stone_bricks")
    b.fill(6, 0, 0, 6, 3, 0, "stone_bricks")
    b.fill(1, 3, 0, 5, 3, 0, "stone_bricks")         # a 5 block lintel between piers
    assert R.load_path(b)["floating"] == 0
    b.fill(7, 3, 0, 9, 3, 0, "stone_bricks")         # three blocks out past the pier
    assert R.load_path(b)["floating"] == 2
    b.fill(7, 3, 0, 9, 3, 0, "spruce_planks")        # timber reaches two
    assert R.load_path(b)["floating"] == 1


def test_house_roof_is_measured_at_45_degrees_and_windmill_sails_are_hung():
    r = R.roof(house("blocky", 1, style="brick"))
    assert r["covering"] == "slate" and abs(r["pitch"] - 45) < 1
    assert R.load_path(windmill("blocky", 0))["floating"] == 0


def test_score_reports_every_rule_with_value_and_range():
    r = R.score_building(house("blocky", 2, style="timber"), tradition="english-medieval", purpose="cottage")
    rules = {c["rule"].split(":")[-1] for c in r["checks"]}
    assert {"load-path", "roof-pitch", "roof-covers", "chimney", "facade-depth", "flat-run", "window-ratio",
            "materials", "base-course", "footprint", "storeys"} <= rules
    assert 0 < r["score"] <= 1
    assert all("want" in c and "value" in c for c in r["checks"])
