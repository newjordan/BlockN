"""One architecture grammar exercised across four very different briefs."""
from copy import deepcopy

import pytest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture.agent import series_recipe
from core.architecture.construction import construction_report


BRIEFS = {
    "brutalist-civic-center": ("forum", "assembly-gate"),
    "brutalist-housing": ("slab", "stepped"),
    "brutalist-arts-center": ("sky-bridges", "stepped-galleries"),
    "brutalist-portal-field": ("procession", "syncopated"),
}
COMMON = {"protected_void", "visible_bearing", "bay_rhythm", "material_expression"}


@pytest.mark.parametrize("product", BRIEFS)
def test_shared_rules_and_distinct_programs(product):
    desc = describe_product(product)
    assert desc["architect_id"] == "brutalist"
    assert desc["variants"] == ["auto", "random", *BRIEFS[product]]
    rules = desc["rules"]["grammar"]
    assert set(rules) == COMMON
    assert all(record["expression"] and record["measures"] for record in rules.values())
    assert len({describe_product(name)["rules"]["grammar"]["protected_void"]["expression"]
                for name in BRIEFS}) == len(BRIEFS)

    fixed_geometry = []
    for variant in BRIEFS[product]:
        request = roll_request(product, 19, variant=variant, detail="concept")
        result = build(request)
        assert result.quality.status == "pass", [c.code for c in result.quality.checks if c.status == "fail"]
        assert {f"brutalist.{code}" for code in COMMON} <= {c.code for c in result.quality.checks if c.status == "pass"}
        doc = result.document
        assert doc.fingerprint() == BuildDocument.from_dict(doc.to_dict()).fingerprint()
        assert doc.fingerprint() == build(request).document.fingerprint()
        assert "brutalist-concrete" in construction_report(doc, result.quality)["traditions"]
        assert all(b.mesh.is_watertight and b.mesh.is_winding_consistent and b.mesh.volume > 0
                   for b in result.scene.blocks)
        matched = build(BuildRequest(product=product, seed=19, variant=variant, detail="concept",
                                     parameters={"variation": 0}))
        assert matched.quality.status == "pass"
        fixed_geometry.append([(p.role, p.position, p.dimensions) for p in matched.document.placements])
    assert fixed_geometry[0] != fixed_geometry[1]


def test_random_series_is_replayable_and_varies_more_than_color():
    product = "brutalist-portal-field"
    rolls = [roll_request(product, seed, detail="concept").to_dict() for seed in range(12)]
    recipe = series_recipe(12, 0, (product,), detail="concept")
    assert rolls == [entry["request"] for entry in recipe["buildings"]]
    assert {r["variant"] for r in rolls} == set(BRIEFS[product])
    assert len({(r["width"], r["height"], r["depth"], r["bays"]) for r in rolls}) >= 10
    # A zero-variation study locks all geometry when the explicit variant and envelope are unchanged.
    def geometry(seed, variation):
        r = build(BuildRequest(product=product, seed=seed, variant="syncopated", detail="concept",
                               parameters={"variation": variation}))
        return [(p.role, p.position, p.dimensions) for p in r.document.placements]
    assert geometry(1, 0) == geometry(2, 0)
    assert geometry(1, 1) != geometry(2, 1)


def test_metadata_free_audit_catches_blockage_missing_bearing_drift_and_imitation():
    result = build(roll_request("brutalist-portal-field", 7, variant="procession", detail="concept"))
    doc = deepcopy(result.document)
    doc.metadata.clear()
    original = next(p for p in doc.placements if p.role == "portal_pier")
    copy = deepcopy(original)
    copy.id += ".obstruction"
    copy.position = (0, original.position[1], original.position[2])
    doc.placements.append(copy)
    doc.placements.remove(original)
    lintel = next(p for p in doc.placements if p.role == "portal_lintel")
    x, y, z = lintel.position
    lintel.position = (x, y, z + 0.25)
    lintel.substance = "limestone"
    failed = {c.code for c in rebuild(doc).quality.checks if c.status == "fail"}
    assert {f"brutalist.{code}" for code in COMMON} <= failed


@pytest.mark.parametrize("variant", BRIEFS["brutalist-arts-center"])
def test_both_stair_towers_reach_both_skybridges(variant):
    request = roll_request("brutalist-arts-center", 24, variant=variant, detail="concept")
    doc = build(request).document
    routes = doc.metadata["brutalist_rules"]["walking_routes"]
    assert set(routes) == {f"{wing}-to-skybridge-{level}" for wing in ("west", "east") for level in (1, 2)}
    for path in routes.values():
        assert path[0]["member"].startswith("brutalist.site_base.")
        assert path[-1]["member"].startswith("brutalist.sky_bridge.")
        assert any(step["member"].startswith("brutalist.stair_tread.") for step in path)
    doc.metadata.clear()
    assert next(c for c in rebuild(doc).quality.checks if c.code == "brutalist.connected_circulation").status == "pass"


@pytest.mark.parametrize("damage", ("missing-tread", "blocked-door", "lost-landing"))
def test_circulation_fails_on_route_damage_with_metadata_removed(damage):
    result = build(BuildRequest(product="brutalist-arts-center", detail="concept"))
    doc = result.document
    path = doc.metadata["brutalist_rules"]["walking_routes"]["east-to-skybridge-1"]
    doc.metadata.clear()
    if damage == "missing-tread":
        member = next(p for p in doc.placements if p.id == next(s["member"] for s in path
                      if ".stair_tread." in s["member"]))
        doc.placements.remove(member)
    elif damage == "lost-landing":
        member = next(p for p in doc.placements if p.id == path[-1]["member"])
        x, y, z = member.position
        member.position = (x, y + .24, z)
    else:
        door = next(s for s in path if s.get("zone") == "gallery-door")
        (x0, z0), (x1, z1) = door["plan"]
        member = deepcopy(next(p for p in doc.placements if p.role == "stair_tread"))
        member.id = "brutalist.door-obstruction"
        member.position = ((x0+x1)/2, door["elevation"]+.55, (z0+z1)/2)
        member.dimensions = (.50, .8, .5)
        doc.placements.append(member)
    check = next(c for c in rebuild(doc).quality.checks if c.code == "brutalist.connected_circulation")
    assert check.status == "fail"
    assert member.id in check.primitive_ids
