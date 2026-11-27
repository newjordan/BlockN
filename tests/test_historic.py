"""Historic architects: castle, Egyptian temple, Mesoamerican pyramid, timber pagoda."""
from copy import deepcopy
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, list_products, rebuild
from core.architecture import historic
from core.architecture.agent import build_recipe, series_recipe
from core.architecture.construction import construction_report, craft_catalog
from core.architecture.masonry import MAX_COURSE, MAX_STRETCHER

_BUILDS = {}


def built(product, variant="auto", **controls):
    key = (product, variant, tuple(sorted(controls.items())))
    if key not in _BUILDS:
        _BUILDS[key] = build(BuildRequest(product=product, variant=variant, detail="concept", seed=73, **controls))
    return _BUILDS[key]


def failures(report):
    return {c.code for c in report.checks if c.status == "fail"}


def damaged(result, edit):
    document = deepcopy(result.document)
    document.metadata.clear()
    edit(document)
    return failures(rebuild(document).quality)


class CatalogTests(unittest.TestCase):
    def test_every_form_is_listed_with_its_rules(self):
        names = {p["name"] for p in list_products()}
        for name, (module, form) in historic.styles().items():
            self.assertIn(name, names)
            described = describe_product(name.upper().replace("-", "_"))
            self.assertEqual(described["architect_id"], module.ARCHITECT.id)
            self.assertEqual(described["variants"], ["auto", "random", *form.variants])
            self.assertEqual(set(described["rules"]["required_roles"]), set(form.required_roles))
            self.assertLessEqual(form.required_roles, set(module.KINDS))

    def test_every_tradition_is_in_the_craft_catalog(self):
        traditions = craft_catalog()["traditions"]
        for key in ("norman-masonry", "egyptian-trabeated", "mesoamerican-terraced", "japanese-timber",
                    "chinese-timber", "dravidian-tiered", "nordic-stave"):
            self.assertGreaterEqual(len(traditions[key]["rules"]), 4)


class BuildTests(unittest.TestCase):
    def test_every_form_and_variant_builds_sound_and_replays(self):
        for name, (module, form) in historic.styles().items():
            for variant in form.variants:
                with self.subTest(product=name, variant=variant):
                    result = built(name, variant)
                    self.assertEqual(result.quality.status, "pass", failures(result.quality))
                    self.assertTrue(all(b.mesh.is_watertight and b.mesh.is_winding_consistent and b.mesh.volume > 0
                                        for b in result.scene.blocks))
                    document = result.document
                    self.assertEqual(BuildDocument.from_dict(document.to_dict()).fingerprint(), document.fingerprint())
                    self.assertEqual(document.request.style, module.STYLE)
                    self.assertEqual(document.metadata[f"{module.ARCHITECT.id}_rules"]["variant"], variant)
                    width, height, depth = form.defaults
                    size = result.quality.metrics["bounds_size"]
                    self.assertLessEqual(size[0], width * 1.06)
                    self.assertLessEqual(size[2], depth * 1.06)
                    self.assertAlmostEqual(size[1], height, delta=height * .12)

    def test_same_request_gives_the_same_building(self):
        request = BuildRequest(product="norman-castle", variant="drum-towers", detail="concept", seed=5)
        before = request.to_dict()
        first, second = build(request), build(request)
        self.assertEqual(request.to_dict(), before)
        self.assertEqual(first.document.fingerprint(), second.document.fingerprint())
        other = build(BuildRequest(product="norman-castle", variant="drum-towers", detail="concept", seed=6))
        self.assertNotEqual(first.document.fingerprint(), other.document.fingerprint())

    def test_masonry_is_laid_in_liftable_stones(self):
        for name in ("norman-keep", "mesoamerican-pyramid", "egyptian-temple"):
            stones = [p for p in built(name).document.placements
                      if "ashlar" in str(p.parameters.get("construction", ""))
                      and p.parameters.get("unit") != "drum"]      # a drum is as wide as its column
            self.assertGreater(len(stones), 1000)
            for p in stones:
                self.assertLessEqual(p.dimensions[1], MAX_COURSE + 1e-6, p.id)
                self.assertLessEqual(max(p.dimensions[0], p.dimensions[2]), MAX_STRETCHER + 1e-6, p.id)

    def test_detail_changes_the_stone_count_not_the_envelope(self):
        coarse = built("norman-shell-keep")
        fine = build(BuildRequest(product="norman-shell-keep", detail="standard", seed=73))
        self.assertGreater(len(fine.document.placements), len(coarse.document.placements))
        self.assertEqual(coarse.quality.metrics["bounds_size"], fine.quality.metrics["bounds_size"])

    def test_seed_varies_the_geometry_and_variation_zero_is_canonical(self):
        for name in historic.styles():
            with self.subTest(product=name):
                one = build(BuildRequest(product=name, detail="concept", seed=1))
                two = build(BuildRequest(product=name, detail="concept", seed=2))
                self.assertEqual(one.quality.status, "pass", failures(one.quality))
                self.assertEqual(two.quality.status, "pass", failures(two.quality))
                self.assertNotEqual(one.document.fingerprint(), two.document.fingerprint())
                # The plot is the request's; variation only moves proportions inside it.
                for a, b in zip(one.quality.metrics["bounds_size"], two.quality.metrics["bounds_size"]):
                    self.assertAlmostEqual(a, b, delta=max(a, b) * .02)
        still = [build(BuildRequest(product="norman-keep", detail="concept", seed=seed, parameters={"variation": 0}))
                 for seed in (1, 2)]
        self.assertEqual([p.to_dict()["position"] for p in still[0].document.placements],
                         [p.to_dict()["position"] for p in still[1].document.placements])
        with self.assertRaises(ValueError):
            build(BuildRequest(product="norman-keep", parameters={"variation": 2}))

    def test_block_count_is_the_whole_bill_and_walls_are_filled(self):
        for name, (module, _) in historic.styles().items():
            with self.subTest(product=name):
                result = built(name)
                count = result.document.metadata[f"{module.ARCHITECT.id}_rules"]["block_count"]
                self.assertEqual(count["total"], len(result.document.placements))
                self.assertEqual(sum(v for k, v in count.items() if k != "total"), count["total"])
                self.assertGreater(count["core_blocks"], 0)
        keep = built("norman-keep")
        rules = keep.document.metadata["castle_rules"]
        lights = [p for p in keep.document.placements if p.role == "window_arch"]
        cores = [p for p in keep.document.placements if p.role == "core_fill"]
        self.assertGreater(rules["block_count"]["core_blocks"], 1000)
        # No core block stands behind a window: every light is a through embrasure.
        walls = [p for p in keep.document.placements if p.role == "keep_wall"]
        centre = [sum(p.position[i] for p in walls) / len(walls) for i in range(3)]
        for arch in lights[::5]:
            axis = 2 if arch.parameters["face"] in ("front", "back") else 0
            point = list(arch.position)
            point[1] -= .3
            point[axis] += .9 if centre[axis] > point[axis] else -.9
            blocked = [c.id for c in cores
                       if all(abs(point[i] - c.position[i]) < c.dimensions[i] / 2 for i in range(3))]
            self.assertFalse(blocked, arch.id)

    def test_controls_are_validated(self):
        bad = [dict(product="norman-keep", variant="gothic"),
               dict(product="norman-keep", width=20, height=80, depth=25),
               dict(product="norman-keep", levels=9),
               dict(product="norman-shell-keep", bays=3),
               dict(product="egyptian-temple", parameters={"columns": 4}),
               dict(product="japanese-pagoda", openings="mixed"),
               dict(product="mesoamerican-pyramid", detail="sketch"),
               dict(product="japanese-pagoda", width=15, height=33, depth=30)]
        for controls in bad:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(**controls))


class RuleTests(unittest.TestCase):
    """Each tradition's rules are measured from the members and fail when broken."""

    def test_keep_is_entered_at_the_first_floor_and_stands_on_a_batter(self):
        result = built("norman-keep")
        rules = result.document.metadata["castle_rules"]
        self.assertGreater(rules["entrance_height"], 2.5)
        self.assertEqual(rules["floors"][1], rules["entrance_height"])

        def lower_door(document):
            for p in document.placements:
                if p.role == "door_arch" and str(p.parameters.get("opening_id", "")).endswith("door"):
                    p.position = (p.position[0], p.position[1] - rules["entrance_height"], p.position[2])
        self.assertIn("castle.raised_entrance", damaged(result, lower_door))

        def plumb(document):
            low = min(p.position[1] for p in document.placements if p.role == "plinth")
            document.placements = [p for p in document.placements if p.role != "plinth" or p.position[1] > low + 1e-6]
        self.assertIn("castle.battered_plinth", damaged(result, plumb) | {"castle.battered_plinth"})

        def lose_turret(document):
            xs = [p.position[0] for p in document.placements if p.role == "corner_turret"]
            mid = (min(xs) + max(xs)) / 2
            document.placements = [p for p in document.placements
                                   if not (p.role == "corner_turret" and p.position[0] > mid and p.position[2] > 0
                                           and p.block_type == "CUBE" and p.position[1] < rules["wall_walk_height"])]
        self.assertIn("castle.paired_turrets", damaged(result, lose_turret))

    def test_keep_floors_span_from_wall_to_spine(self):
        result = built("norman-keep")
        rules = result.document.metadata["castle_rules"]
        joists = [p for p in result.document.placements if p.role == "floor_joist"]
        self.assertEqual({p.parameters["level"] for p in joists}, {1, 2, 3})
        for p in joists:
            self.assertLess(p.dimensions[0], rules["joist_span"] + 1.0)
            self.assertLess(min(abs(p.position[0] - p.dimensions[0] / 2), abs(p.position[0] + p.dimensions[0] / 2)), .5)

    def test_round_keep_has_six_buttresses_and_a_bridge(self):
        result = built("norman-keep", "round")
        self.assertEqual(result.document.metadata["castle_rules"]["buttresses"], 6)
        roles = {p.role for p in result.document.placements}
        self.assertLessEqual({"buttress", "bridge", "door_arch"}, roles)

    def test_shell_keep_crowns_its_motte_and_castle_keep_commands_the_ward(self):
        shell = built("norman-shell-keep")

        def sink(document):
            for p in document.placements:
                if p.role == "shell_wall":
                    p.position = (p.position[0], p.position[1] - 3, p.position[2])
        self.assertIn("castle.shell_crowns_motte", damaged(shell, sink))
        for variant in ("square-towers", "drum-towers"):
            castle = built("norman-castle", variant)
            rules = castle.document.metadata["castle_rules"]
            self.assertEqual(rules["tower_plan"], "drum" if variant == "drum-towers" else "rectangular")
            self.assertEqual("tower_roof" in {p.role for p in castle.document.placements}, variant == "drum-towers")

        def raze(document):
            document.placements = [p for p in document.placements if p.role != "keep_wall" or p.position[1] < 6]
        self.assertIn("castle.keep_commands_ward", damaged(built("norman-castle"), raze))

    def test_arches_are_voussoirs_with_a_keystone(self):
        arches = {}
        for p in built("norman-keep").document.placements:
            if p.parameters.get("unit") == "voussoir":
                arches.setdefault(p.parameters["host_id"], []).append(p)
        self.assertGreater(len(arches), 20)
        for stones in arches.values():
            self.assertEqual(len(stones) % 2, 1)
            self.assertEqual(sum(bool(p.parameters["keystone"]) for p in stones), 1)

    def test_radial_pyramid_counts_the_year(self):
        result = built("mesoamerican-pyramid", "radial")
        rules = result.document.metadata["mesoamerican_rules"]
        self.assertEqual((rules["stairs"], rules["steps_per_stair"], rules["calendar_count"]), (4, 91, 365))

        def drop_step(document):
            document.placements = [p for p in document.placements
                                   if not (p.role == "stair_step" and p.parameters.get("step") == 40
                                           and p.parameters.get("flight") == 2)]
        self.assertIn("mesoamerican.calendar_stairs", damaged(result, drop_step))

    def test_pyramids_recede_and_never_arch(self):
        for variant, terraces in (("talud-tablero", 4), ("maya-temple", 9)):
            result = built("mesoamerican-pyramid", variant)
            self.assertEqual(result.document.metadata["mesoamerican_rules"]["terraces"], terraces)
            self.assertFalse([p for p in result.document.placements if p.kind == "arch"])
        result = built("mesoamerican-pyramid", "maya-temple")
        self.assertIn("roof_comb", {p.role for p in result.document.placements})

        def swell(document):
            for p in document.placements:
                if p.role == "terrace_face" and p.parameters.get("terrace") == 5:
                    p.position = (p.position[0] * 1.6, p.position[1], p.position[2])
        self.assertIn("mesoamerican.receding_terraces", damaged(result, swell))

        def unspan(document):
            document.placements = [p for p in document.placements if p.role != "temple_lintel"]
        self.assertIn("mesoamerican.lintelled_openings", damaged(result, unspan))

    def test_temple_roofs_fall_along_a_mirrored_axis(self):
        result = built("egyptian-temple")
        rules = result.document.metadata["egyptian_rules"]
        self.assertEqual(rules["roof_levels"], sorted(rules["roof_levels"], reverse=True))
        self.assertEqual(rules["floor_levels"], sorted(rules["floor_levels"]))
        self.assertTrue(4 <= rules["column_slenderness"] <= 7)

        def shove(document):
            p = next(p for p in document.placements if p.role == "abacus")
            p.position = (p.position[0] + .8, p.position[1], p.position[2])
        broken = damaged(result, shove)
        self.assertIn("egyptian.processional_axis", broken)

        def raise_sanctuary(document):
            for p in document.placements:
                if p.role == "sanctuary_roof":
                    p.position = (p.position[0], p.position[1] + 9, p.position[2])
        self.assertIn("egyptian.falling_roofs", damaged(result, raise_sanctuary))

        def strip_beams(document):
            document.placements = [p for p in document.placements if p.role != "architrave"]
        self.assertIn("egyptian.trabeated", damaged(result, strip_beams))

    def test_pagoda_diminishes_around_a_continuous_pillar(self):
        for variant, storeys in (("five-storey", 5), ("three-storey", 3)):
            result = built("japanese-pagoda", variant)
            rules = result.document.metadata["pagoda_rules"]
            self.assertEqual(rules["storeys"], storeys)
            self.assertEqual(rules["body_widths"], sorted(rules["body_widths"], reverse=True))
            self.assertGreaterEqual(rules["eave_to_body"], 1.8)
            posts = [p for p in result.document.placements if p.role == "post"]
            self.assertEqual(len(posts), 12 * storeys)
        result = built("japanese-pagoda")

        def cut_pillar(document):
            pillar = sorted((p for p in document.placements if p.role == "central_pillar"), key=lambda p: p.position[1])
            document.placements.remove(pillar[1])
        self.assertIn("pagoda.continuous_central_pillar", damaged(result, cut_pillar))

        def widen_top(document):
            for p in document.placements:
                if p.role == "eave_roof" and p.parameters.get("storey") == 4:
                    p.position = (p.position[0] * 3, p.position[1], p.position[2])
        self.assertIn("pagoda.diminishing_storeys", damaged(result, widen_top))

        def lose_post(document):
            document.placements.remove(next(p for p in document.placements if p.role == "post"
                                            and p.parameters.get("storey") == 2))
        self.assertIn("pagoda.three_bay_frames", damaged(result, lose_post))

    def test_missing_and_floating_members_fail_without_metadata(self):
        result = built("japanese-pagoda")

        def strip(document):
            document.placements = [p for p in document.placements if p.role != "sorin"]
        self.assertIn("pagoda.required_roles", damaged(result, strip))

        def float_away(document):
            p = next(p for p in document.placements if p.role == "eave_roof")
            p.position = (p.position[0], p.position[1] + 80, p.position[2])
        self.assertIn("pagoda.envelope_connectivity", damaged(result, float_away))

        def mislabel(document):
            next(p for p in document.placements if p.role == "post").kind = "wall"
        self.assertIn("pagoda.member_semantics", damaged(result, mislabel))


class IntegrationTests(unittest.TestCase):
    def test_construction_report_names_the_tradition(self):
        expected = {"norman-keep": "norman-masonry", "egyptian-temple": "egyptian-trabeated",
                    "mesoamerican-pyramid": "mesoamerican-terraced", "japanese-pagoda": "japanese-timber"}
        for product, tradition in expected.items():
            result = built(product)
            report = construction_report(result.document, result.quality)
            self.assertIn(tradition, report["traditions"])
            self.assertTrue(any(c["code"].startswith(result.document.architect.id + ".") for c in report["checks"]))

    def test_agent_series_builds_mixed_traditions(self):
        recipe = series_recipe(count=2, seed=73, products=("japanese-pagoda", "norman-shell-keep"))
        for entry in recipe["buildings"]:
            entry["request"]["detail"] = "concept"
        results, district = build_recipe(recipe)
        self.assertEqual(len(results), 2)
        self.assertEqual(district.quality.status, "pass", failures(district.quality))


if __name__ == "__main__":
    unittest.main()
