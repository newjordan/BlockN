"""Chinese architect: the timber hall, its terrace, its bracket sets and its concave roofs."""
from copy import deepcopy
import random
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild
from core.architecture import historic
from core.architecture.chinese import random_controls
from core.architecture.construction import construction_report, craft_catalog
from core.architecture.substances import SUBSTANCES

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


def members(result, role):
    return [p for p in result.document.placements if p.role == role]


HALL, PALACE = "chinese-hall", "chinese-palace-hall"


class CatalogTests(unittest.TestCase):
    def test_both_halls_are_described_with_what_they_are_built_of(self):
        for name in (HALL, PALACE):
            described = describe_product(name)
            self.assertEqual(described["architect_id"], "chinese")
            self.assertLessEqual(set(described["substances"]), set(SUBSTANCES))
            for substance in ("painted-timber", "glazed-tile", "marble", "softwood", "earth-fill", "fired-clay-brick"):
                self.assertIn(substance, described["substances"], name)
        self.assertEqual(describe_product(HALL)["variants"], ["auto", "random", "hip-roof", "hip-and-gable"])
        self.assertEqual(describe_product(HALL)["rules"]["bays"], [3, 9])
        self.assertIn("variation", describe_product(PALACE)["parameters"])

    def test_the_tradition_is_in_the_craft_catalog(self):
        self.assertGreaterEqual(len(craft_catalog()["traditions"]["chinese-timber"]["rules"]), 4)
        result = built(HALL)
        self.assertIn("chinese-timber", construction_report(result.document, result.quality)["traditions"])


class HallTests(unittest.TestCase):
    def test_bays_shorten_outward_and_the_roof_steepens_to_the_ridge(self):
        for variant in ("hip-roof", "hip-and-gable"):
            rules = built(HALL, variant).document.metadata["chinese_rules"]
            self.assertEqual((rules["bays"], rules["bracket_jumps"], rules["beasts"]), (7, 2, 7))
            self.assertLessEqual(rules["eave_column_height"], rules["middle_bay"])      # the Song rule
            self.assertEqual(rules["purlin_slopes"], sorted(rules["purlin_slopes"]))    # jujia: each step steeper
            self.assertLess(rules["purlin_slopes"][-2], rules["purlin_slopes"][-1])

        def lose_corner_column(document):
            columns = [p for p in document.placements if p.role == "column"]
            front, end = min(p.position[2] for p in columns), min(p.position[0] for p in columns)
            document.placements = [p for p in document.placements if not (
                p.role == "column" and abs(p.position[2] - front) < 1e-3 and abs(p.position[0] - end) < 1e-3)]
        self.assertIn("chinese.wider_middle_bay", damaged(built(HALL), lose_corner_column))

        def stretch_column(document):
            column = next(p for p in document.placements if p.role == "column")
            column.dimensions = (column.dimensions[0], column.dimensions[1] * 2, column.dimensions[2])
        self.assertIn("chinese.column_within_bay", damaged(built(HALL), stretch_column))

        def flatten_upper_steps(document):
            for p in document.placements:
                if p.role == "roof_tile" and p.parameters["slope"] == "long" and p.parameters["step"] >= 3:
                    p.position = (p.position[0], p.position[1] - 2, p.position[2])
        self.assertIn("chinese.roof_steepens_to_ridge", damaged(built(HALL), flatten_upper_steps))

    def test_the_hall_stands_on_its_terrace_under_its_bracket_sets(self):
        def sink_base(document):
            base = next(p for p in document.placements if p.role == "column_base")
            base.position = (base.position[0], base.position[1] - .5, base.position[2])
        self.assertIn("chinese.hall_on_terrace", damaged(built(HALL), sink_base))

        def strip_bracket_set(document):
            column = next(p for p in document.placements if p.role == "column")
            document.placements = [p for p in document.placements if not (
                p.role == "bracket_block" and p.parameters.get("name") == "ludou"
                and abs(p.position[0] - column.position[0]) < p.dimensions[0]
                and abs(p.position[2] - column.position[2]) < p.dimensions[2])]
        self.assertIn("chinese.brackets_carry_eaves", damaged(built(HALL), strip_bracket_set))

    def test_the_corners_fly_and_the_hips_carry_their_ornaments(self):
        def drop_corner(document):
            front = [p for p in document.placements if p.role == "roof_tile" and p.parameters["course"] == 0
                     and p.parameters["slope"] == "long" and p.position[2] < 0]
            edge = max(p.parameters["column"] for p in front)
            for p in front:
                if p.parameters["column"] in (0, edge):
                    p.position = (p.position[0], p.position[1] - .6, p.position[2])
        self.assertIn("chinese.flying_corners", damaged(built(HALL), drop_corner))

        def lose_owl_tail(document):
            document.placements.remove([p for p in document.placements if p.role == "chiwen"][1])
        self.assertIn("chinese.owl_tail_each_end", damaged(built(HALL), lose_owl_tail))

        def lose_beast(document):
            document.placements.remove(next(p for p in document.placements
                                            if p.role == "beast" and "beast" in p.parameters))
        self.assertIn("chinese.beast_line_each_hip", damaged(built(HALL), lose_beast))

    def test_only_the_gabled_roof_closes_with_a_gable(self):
        hip, gable = built(HALL), built(HALL, "hip-and-gable")
        self.assertFalse(members(hip, "gable_panel"))
        self.assertTrue(members(gable, "bargeboard"))
        for roof in (hip, gable):
            ridge = members(roof, "main_ridge")
            span = max(p.position[0] + p.dimensions[0] / 2 for p in ridge) - min(p.position[0] - p.dimensions[0] / 2 for p in ridge)
            roof.span = span
        self.assertGreater(gable.span, hip.span)            # a hip-and-gable ridge runs further

        def strip_gable(document):
            document.placements = [p for p in document.placements if p.role != "gable_panel"]
        self.assertIn("chinese.gable_closes_hips", damaged(gable, strip_gable))

    def test_the_palace_hall_has_two_eaves_and_nine_step_brackets(self):
        for variant in ("double-hip-roof", "double-hip-and-gable"):
            result = built(PALACE, variant)
            rules = result.document.metadata["chinese_rules"]
            self.assertEqual((rules["bays"], rules["bracket_jumps"]), (9, 3))      # nine bays: the nine-step sets
            purlins = [p for p in result.document.placements
                       if p.role == "purlin" and p.parameters.get("part") == "eave purlin"]
            lower = [p.position[1] for p in purlins if p.parameters["storey"] == "lower"]
            upper = [p.position[1] for p in purlins if p.parameters["storey"] == "upper"]
            self.assertLess(max(lower), min(upper))
            self.assertTrue(members(result, "upper_wall"))

        def swap_eaves(document):
            for p in document.placements:
                if p.role == "purlin" and p.parameters.get("storey") == "lower":
                    p.position = (p.position[0], p.position[1] + 9, p.position[2])
        self.assertIn("chinese.two_eaves", damaged(built(PALACE), swap_eaves))

    def test_every_hall_stays_inside_its_plot_and_is_billowed_as_timber_and_stone(self):
        for name, (module, form) in historic.styles().items():
            if module.ARCHITECT.id == "chinese":
                with self.subTest(product=name):
                    result = built(name)
                    width, _, depth = form.defaults
                    size = result.quality.metrics["bounds_size"]
                    self.assertLessEqual(size[0], width + 1e-3)
                    self.assertLessEqual(size[2], depth + 1e-3)
                    self.assertTrue(all(p.substance in SUBSTANCES for p in result.document.placements))
                    count = result.document.metadata["chinese_rules"]["block_count"]
                    self.assertGreater(count.get("timbers", 0), 1000)
                    self.assertEqual(count["total"], len(result.document.placements))

    def test_variation_zero_is_canonical_and_a_build_replays(self):
        still = [build(BuildRequest(product=HALL, detail="concept", seed=seed, parameters={"variation": 0}))
                 for seed in (1, 2)]
        self.assertEqual([p.to_dict()["position"] for p in still[0].document.placements],
                         [p.to_dict()["position"] for p in still[1].document.placements])
        request = BuildRequest(product=HALL, detail="concept", seed=4)
        self.assertEqual(build(request).document.fingerprint(), build(request).document.fingerprint())

    def test_a_random_hall_is_the_default_one_resized_whole(self):
        for seed in (1, 2):
            controls = random_controls(HALL, random.Random(seed))
            controls["detail"] = "concept"
            with self.subTest(controls=controls):
                result = build(BuildRequest(product=HALL, seed=seed, **controls))
                self.assertEqual(result.quality.status, "pass", failures(result.quality))

    def test_controls_are_validated(self):
        bad = [dict(product=HALL, detail="concept", bays=8), dict(product=HALL, detail="concept", bays=11),
               dict(product=HALL, detail="concept", levels=2),
               dict(product=HALL, detail="concept", width=44, height=40, depth=24),
               dict(product=HALL, detail="concept", width=20, height=18, depth=40),
               dict(product=PALACE, detail="concept", bays=7),
               dict(product=HALL, detail="concept", parameters={"fengshui": 1})]
        for controls in bad:
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(**controls))


if __name__ == "__main__":
    unittest.main()
