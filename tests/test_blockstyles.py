"""Style sub-architects: each builds a period house that faces the street
and passes the construction rulebook for its tradition and purpose."""
import pytest

from core.architecture import blockrules as R
from core.architecture import blockstyles as BS
from core.architecture import lore

KEYS = sorted(BS.SUB_ARCHITECTS)


@pytest.mark.parametrize("key", KEYS)
def test_sub_architect_names_a_known_tradition_and_purpose(key):
    t, p = key
    assert t in lore.TRADITIONS and p in lore.PURPOSES


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sub_architect_house_stands_faces_the_street_and_passes_the_rules(key, seed):
    t, p = key
    b = BS.build(t, p, "blocky", seed)
    assert (b.tradition, b.purpose, b.kind) == (t, p, "house")
    # The street door comes first and opens north, onto the street at z=0.
    assert b.ports and "street" in b.ports[0]["id"]
    assert tuple(b.ports[0]["facing"]) == (0, 0, -1)
    assert R.load_path(b)["floating"] == 0
    r = R.score_building(b, tradition=t, purpose=p)
    assert r["passed"] >= r["total"] - 2, [c for c in r["checks"] if not c["pass"]]
    rules = R.by_rule([r])
    for must in ("load-path", "roof-pitch", "covering-fits", "roof-covers"):
        ok, n = rules[must]
        assert ok == n, must


def test_unknown_sub_architect_says_what_there_is():
    with pytest.raises(ValueError, match="tudor/merchant-house"):
        BS.build("klingon", "merchant-house")


@pytest.mark.parametrize("row", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_seld_with_its_door_on_the_street_line_passes_every_rule(row, seed):
    b = BS.english_shop("blocky", seed, row=row, street_door=True)
    assert b.street_door == 1
    # Nothing stands in front of the street wall at the ground: the door
    # opens straight onto the street.
    assert not any(z < 0 and y <= 2 for (x, y, z) in b.blocks)
    r = R.score_building(b, tradition="english-medieval", purpose="shop")
    assert r["passed"] == r["total"], [c for c in r["checks"] if not c["pass"]]
