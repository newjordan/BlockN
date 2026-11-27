"""Regional town layouts: a Pueblo laid out as Pueblos are, scored by the
same town rules as the English town and walked from its gate."""
from core.architecture import blockregion, blockrules
from core.architecture import blockstyles
from core.architecture.blockmode import state
from core.architecture.gameplay import walk as walk_grid


def test_pueblo_rings_its_plaza_passes_the_town_rules_and_every_floor_is_walked():
    b = blockregion.pueblo("blocky", 1)
    plan = b.town
    assert plan["counts"]["houses"] >= 55
    r = blockrules.score_town(b)
    assert r["passed"] == r["total"] == 5, [c for c in r["checks"] if not c["pass"]]
    _, rep = blockregion.region_walk(b)
    assert rep["houses_every_floor_reached"] == rep["houses"]
    assert rep["kiva_floor_reached"]


def test_ladders_are_climbed_to_the_roof_they_lead_to():
    b = blockstyles.pueblo_terrace("blocky", 1)
    for x in range(-4, 12):
        for z in range(-6, 20):
            b.blocks.setdefault((x, 0, z), state("grass_block"))
    sub = b.walk_sub()
    g = b.grid(sub)
    ox, oy, oz = g.origin
    seed = (int((3.5 - ox) * sub), int((1 - oy) * sub), int((-4.5 - oz) * sub))
    S = b.sizes.storey
    top = 3 * S + 1          # standing on the third storey's roof
    no = walk_grid(g, b.player, seeds=[seed])
    yes = walk_grid(g, b.player, seeds=[seed], climbs=b.climbs(sub, g.origin))
    def reached(w):
        ys = w.nodes[:, 1] / sub + oy
        return bool(w.reached[abs(ys - top) < .3].any())
    assert not reached(no) and reached(yes)


def test_norse_village_lines_its_street_with_farmsteads_and_every_longhouse_is_walked():
    b = blockregion.norse("blocky", 1)
    plan = b.town
    assert plan["counts"]["street_farms"] >= 16
    assert sum(h["kind"] == "hall" for h in plan["houses"]) == 1
    r = blockrules.score_town(b)
    assert r["passed"] == r["total"] == 5, [c for c in r["checks"] if not c["pass"]]
    _, rep = blockregion.region_walk(b)
    assert rep["houses_every_floor_reached"] == rep["houses"]


def test_a_flat_roofs_parapet_is_a_wall_not_a_slope():
    b = blockstyles.pueblo_terrace("blocky", 1)
    r = blockrules.roof(b)
    assert r["covering"] == "flat-earth" and r["pitch"] <= 8


def test_mediterranean_blocks_of_courtyard_houses_round_a_piazza_pass_and_are_walked():
    b = blockregion.mediterranean("blocky", 1)
    assert b.town["counts"]["houses"] >= 56
    r = blockrules.score_town(b)
    assert r["passed"] == r["total"] == 5, [c for c in r["checks"] if not c["pass"]]
    _, rep = blockregion.region_walk(b)
    assert rep["houses_every_floor_reached"] == rep["houses"]
