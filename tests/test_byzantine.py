"""Byzantine architect: the cross-in-square church in its four plans."""
from copy import deepcopy
import random
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild
from core.architecture.byzantine import random_controls
from core.architecture.construction import construction_report, craft_catalog

_BUILDS = {}
PRODUCT = "byzantine-church"
PLANS = ("four-column", "two-column", "five-dome", "four-pier")


def built(variant="auto", **controls):
    key = (variant, tuple(sorted(controls.items())))
    if key not in _BUILDS:
        _BUILDS[key] = build(BuildRequest(product=PRODUCT, variant=variant, detail="concept", seed=73, **controls))
    return _BUILDS[key]


def failures(report):
    return {c.code for c in report.checks if c.status == "fail"}


def damaged(result, edit):
    document = deepcopy(result.document)
    document.metadata.clear()
    edit(document)
    return failures(rebuild(document).quality)


def members(result, role):
    return [p for p in result.document.placements if p.role == role]


def hosts(membership):
    return {p.parameters.get("host_id") for p in membership}


def top(result, role):
    return max(p.position[1] + p.dimensions[1] / 2 for p in members(result, role))


class CatalogTests(unittest.TestCase):
    def test_product_is_described_with_its_plans_and_substances(self):
        product = describe_product(PRODUCT)
        self.assertEqual(product["architect_id"], "byzantine")
        self.assertEqual(product["variants"], ["auto", "random", *PLANS])
        self.assertEqual(product["rules"]["bays"], [4, 20])
        self.assertIn("Middle Byzantine", product["rules"]["period"])
        self.assertLessEqual({"fired-clay-brick", "limestone", "marble", "lead-sheet", "clay-roof-tile",
                              "lime-mortar", "oak", "wrought-iron"}, set(product["substances"]))

    def test_tradition_is_in_the_craft_catalog(self):
        self.assertGreaterEqual(len(craft_catalog()["traditions"]["byzantine-cross-in-square"]["rules"]), 4)
        for variant in PLANS:
            report = construction_report(built(variant).document, built(variant).quality)
            self.assertIn("byzantine-cross-in-square", report["traditions"])


class PlanTests(unittest.TestCase):
    def test_four_columns_carry_four_arches_and_pendentives_carry_the_drum(self):
        church = built("four-column")
        self.assertEqual(church.quality.status, "pass", failures(church.quality))
        self.assertEqual(len(hosts(members(church, "column_shaft"))), 6)     # four in the naos, two in the narthex
        self.assertEqual(len(hosts(members(church, "great_arch"))), 4)
        self.assertEqual({p.parameters["corner"] for p in members(church, "pendentive")}, {"-1-1", "-1+1", "+1-1", "+1+1"})

        def shift_pendantive(document):
            brick = next(p for p in document.placements if p.role == "pendentive")
            brick.position = (brick.position[0] + 1.2, brick.position[1], brick.position[2])
        self.assertIn("byzantine.pendentives_carry_drum", damaged(church, shift_pendantive))

        def level_a_course(document):
            high = [p for p in document.placements if p.role == "pendentive" and p.parameters["corner"] == "+1+1"]
            courses = sorted({p.parameters["course"] for p in high})
            for p in high:
                if p.parameters["course"] == courses[-1]:          # push the top course back out from under the dome
                    p.position = (p.position[0] + .7, p.position[1], p.position[2] + .7)
        self.assertIn("byzantine.pendentives_corbelled", damaged(church, level_a_course))

    def test_two_column_rests_the_sanctuary_arches_on_piers(self):
        church = built("two-column")
        self.assertEqual(church.quality.status, "pass", failures(church.quality))
        axis = sum(p.position[2] for p in members(church, "dome") if not p.parameters.get("lesser")) / len(members(church, "dome"))
        shafts = [p for p in members(church, "column_shaft") if p.dimensions[0] > .5]        # the naos columns are the stout ones
        piers = [p for p in members(church, "pier") if p.dimensions[2] > 1.0]
        self.assertEqual(len({p.parameters["host_id"] for p in shafts}), 2)
        self.assertEqual(len({p.parameters["host_id"] for p in piers}), 2)
        self.assertTrue(all(p.position[2] > axis for p in piers))            # both stand at the sanctuary end
        self.assertTrue(all(p.position[2] < axis for p in shafts))

    def test_five_dome_raises_a_lesser_dome_over_each_corner_bay(self):
        church = built("five-dome")
        self.assertEqual(church.quality.status, "pass", failures(church.quality))
        domes = members(church, "dome")
        lesser = [p for p in domes if p.parameters.get("lesser")]
        main = [p for p in domes if not p.parameters.get("lesser")]
        self.assertEqual(len(hosts(lesser)), 4)
        self.assertLess(max(p.position[1] + p.dimensions[1] / 2 for p in lesser), top(church, "dome") - 1.0)
        rules = church.document.metadata["byzantine_rules"]
        self.assertIn("lesser dome", rules["typology"])
        self.assertTrue(main and lesser)

    def test_four_pier_fills_the_corners_and_keeps_one_apse(self):
        church = built("four-pier")
        self.assertEqual(church.quality.status, "pass", failures(church.quality))
        axis = sum(p.position[2] for p in members(church, "dome")) / len(members(church, "dome"))
        self.assertFalse([p for p in members(church, "column_shaft") if abs(p.position[2] - axis) < 4])
        columns = built("four-column")
        axis = sum(p.position[2] for p in members(columns, "dome")) / len(members(columns, "dome"))
        naos_shafts = [p for p in members(columns, "column_shaft") if abs(p.position[2] - axis) < 4]
        self.assertEqual(len(hosts(naos_shafts)), 4)
        self.assertTrue(members(church, "cell"))
        self.assertEqual(len(hosts(members(church, "apse_wall"))), 1)
        self.assertEqual(len(hosts(members(church, "great_arch"))), 4)

    def test_drum_carries_the_lights_asked_for(self):
        church = built("four-column", bays=8)
        self.assertEqual(church.quality.status, "pass", failures(church.quality))
        lights = {p.parameters["opening_id"] for p in members(church, "window_arch")
                  if str(p.parameters.get("opening_id", "")).startswith("drum.")}
        self.assertEqual(len(lights), 8)

        def put_out_a_light(document):
            document.placements = [p for p in document.placements
                                   if not str(p.parameters.get("opening_id", "")).startswith("drum.0.")]
        self.assertIn("byzantine.drum_lights", damaged(church, put_out_a_light))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", bays=9))

    def test_walls_are_banded_round_the_whole_church(self):
        church = built("four-column")
        walls = [p for p in church.document.placements if p.role in {"naos_wall", "narthex_wall", "partition"}]
        self.assertEqual({p.parameters["band"] for p in walls}, {"brick", "stone"})
        self.assertEqual({p.parameters["band"] for p in church.document.placements if p.role == "apse_wall"},
                         {"brick", "stone"})

        def turn_a_wall_to_stone(document):                        # one wall with no brick in it at all
            walls = [p for p in document.placements if p.role == "naos_wall" and "band" in p.parameters]
            host = walls[0].parameters["host_id"]
            for p in walls:
                if p.parameters["host_id"] == host:
                    p.parameters["band"] = "stone"
        self.assertIn("byzantine.banded_walls", damaged(church, turn_a_wall_to_stone))

    def test_narthex_spans_the_west_front_and_opens_through_three_arches(self):
        church = built("four-column")
        arcade = {p.parameters["opening_id"] for p in members(church, "arcade_arch")}
        self.assertEqual(arcade, {"arcade.-1", "arcade.0", "arcade.+1"})
        narthex = [p.position[2] - p.dimensions[2] / 2 for p in members(church, "narthex_wall")]
        naos = [p.position[2] - p.dimensions[2] / 2 for p in members(church, "naos_wall")]
        self.assertLess(min(narthex), min(naos) - 2.5)

        def close_the_arcade(document):
            document.placements = [p for p in document.placements if not str(p.parameters.get("opening_id", "")).startswith("arcade.+1")]
        self.assertIn("byzantine.narthex_and_arcade", damaged(church, close_the_arcade))

    def test_dome_crowns_the_church_at_the_height_asked_for(self):
        for height in (19, 24):
            church = build(BuildRequest(product=PRODUCT, detail="concept", seed=5, width=16, height=height, depth=24))
            self.assertEqual(church.quality.status, "pass", failures(church.quality))
            self.assertAlmostEqual(church.quality.metrics["bounds_size"][1], height, delta=.01)
            self.assertAlmostEqual(church.quality.metrics["bounds_size"][0], 16, delta=.01)
            rules = church.document.metadata["byzantine_rules"]
            self.assertLess(rules["drum_height"], 2.5 * rules["dome_diameter"] / 2 + .01)
            self.assertGreater(rules["springing"], 3.1)

        def lower_the_dome(document):
            for p in document.placements:
                if p.role in {"dome", "drum"}:
                    p.position = (p.position[0], p.position[1] - 6.0, p.position[2])
        self.assertIn("byzantine.dome_crowns_the_church", damaged(church, lower_the_dome))


class VariationTests(unittest.TestCase):
    def test_seed_changes_the_church_but_not_its_plot(self):
        plots, prints = [], set()
        for seed in (1, 2, 3, 4):
            church = build(BuildRequest(product=PRODUCT, detail="concept", seed=seed))
            self.assertEqual(church.quality.status, "pass", failures(church.quality))
            plots.append(church.quality.metrics["bounds_size"])
            prints.add(church.document.fingerprint())
        self.assertEqual(len(prints), 4)
        for plot in plots[1:]:
            for a, b in zip(plot, plots[0]):
                self.assertAlmostEqual(a, b, delta=.01)

    def test_variation_zero_builds_the_canonical_church_whatever_the_seed(self):
        still = [build(BuildRequest(product=PRODUCT, detail="concept", seed=seed, parameters={"variation": 0}))
                 for seed in (1, 2)]
        self.assertEqual([p.to_dict()["position"] for p in still[0].document.placements],
                         [p.to_dict()["position"] for p in still[1].document.placements])
        canonical = still[0].document.metadata["byzantine_rules"]
        self.assertEqual(canonical["exonarthex"], 0)
        self.assertEqual(canonical["drum_sides"], "round")
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", parameters={"variation": 2}))

    def test_detail_refines_the_stonework_and_not_the_envelope(self):
        coarse = built("four-column")
        fine = build(BuildRequest(product=PRODUCT, detail="presentation", seed=73))
        self.assertEqual(fine.quality.status, "pass", failures(fine.quality))
        self.assertGreater(len(fine.document.placements), len(coarse.document.placements))
        self.assertEqual(coarse.quality.metrics["bounds_size"], fine.quality.metrics["bounds_size"])

    def test_random_controls_pass_and_replay(self):
        for seed in (11, 12, 13):
            controls = random_controls(PRODUCT, random.Random(seed))
            controls["detail"] = "concept"
            with self.subTest(controls=controls):
                church = build(BuildRequest(product=PRODUCT, seed=11, **controls))
                self.assertEqual(church.quality.status, "pass", failures(church.quality))
                self.assertEqual(BuildDocument.from_dict(church.document.to_dict()).fingerprint(),
                                 church.document.fingerprint())
                self.assertEqual(build(church.document.request).document.fingerprint(), church.document.fingerprint())

    def test_controls_are_validated(self):
        bad = [dict(product=PRODUCT, detail="concept", bays=19),
               dict(product=PRODUCT, detail="concept", bays=3),
               dict(product=PRODUCT, detail="concept", width=5),
               dict(product=PRODUCT, detail="concept", width=15, height=32, depth=23),
               dict(product=PRODUCT, detail="concept", width=15, height=20, depth=17),
               dict(product=PRODUCT, detail="concept", levels=2),
               dict(product=PRODUCT, detail="concept", parameters={"dome": 1}),
               dict(product=PRODUCT, detail="concept", variant="athenian"),
               dict(product=PRODUCT, detail="concept", openings="mixed")]
        for controls in bad:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(**controls))


if __name__ == "__main__":
    unittest.main()
