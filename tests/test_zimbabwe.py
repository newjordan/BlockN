"""Great Zimbabwe enclosure: variants, lines, and rules measured from the stones."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture import sonnet

PRODUCT = "zimbabwe-enclosure"


def failed(document):
    document.metadata.clear()            # rules must read the geometry, not the claims about it
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


def shift(p, dx=0., dy=0., dz=0.):
    p.position = (p.position[0] + dx, p.position[1] + dy, p.position[2] + dz)


class ZimbabweTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = BuildRequest(product=PRODUCT, variant="great-enclosure", detail="concept", seed=73)
        cls.great = build(cls.request)

    def test_variants_build_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "zimbabwe")
        for seed, variant in enumerate(("great-enclosure", "valley-enclosure"), 31):
            with self.subTest(variant=variant):
                request = roll_request(PRODUCT, seed, variant=variant, detail="concept")
                result = build(request)
                self.assertEqual(result.quality.status, "pass")
                roles = {p.role for p in result.document.placements}
                self.assertEqual("passage_wall" in roles, variant == "great-enclosure")
                size = result.quality.metrics["bounds_size"]
                for got, asked in zip(size, (request.width, request.height, request.depth)):
                    self.assertAlmostEqual(got, asked, places=3)
                self.assertEqual(result.document.fingerprint(), build(request).document.fingerprint())
                self.assertEqual(sonnet.verse(request), result.document.metadata["zimbabwe_rules"]["verse"])

    def test_lines_can_be_pinned(self):
        self.assertEqual(set(sonnet.lines_of(PRODUCT)), {"band", "tower", "entrances", "huts"})
        request = BuildRequest(product=PRODUCT, variant="valley-enclosure", detail="concept", seed=5,
                               parameters={"band": "dentelle", "tower": "twin", "entrances": "two", "huts": "cluster"})
        result = build(request)
        self.assertEqual(result.quality.status, "pass")
        document = result.document
        self.assertEqual(document.metadata["zimbabwe_rules"]["verse"], request.parameters)
        self.assertEqual({p.parameters["tower"] for p in document.placements if p.role == "tower_stone"}, {0, 1})
        self.assertEqual({p.parameters["pattern"] for p in document.placements if p.role == "decorated_band"},
                         {"dentelle"})
        self.assertEqual(len({p.parameters["facet"] for p in document.placements if p.role == "threshold"}), 2)
        self.assertEqual(len({p.parameters["hut"] for p in document.placements if p.role == "hut_roof"}), 4)
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", parameters={"band": "tartan"}))

    def test_wall_is_laid_dry_in_liftable_stones_around_a_core(self):
        document = self.great.document
        stones = [p for p in document.placements if p.role in ("outer_wall", "wall_lining", "passage_wall")]
        self.assertGreater(len(stones), 2000)
        for p in stones:
            self.assertLessEqual(p.dimensions[1], .8 + 1e-6)
            self.assertLessEqual(max(p.dimensions[0], p.dimensions[2]), 1.6 + 1e-6)
        self.assertGreater(document.metadata["zimbabwe_rules"]["block_count"]["core_blocks"], 0)
        monoliths = [p for p in document.placements if p.role == "monolith"]
        self.assertTrue(monoliths)
        for p in monoliths:
            self.assertAlmostEqual(p.position[1] + p.dimensions[1] / 2, 9.0)

    def test_a_stone_off_the_curve_is_caught(self):
        document = deepcopy(self.great.document)
        stone = next(p for p in document.placements if p.role == "outer_wall" and p.position[2] > 10)
        shift(stone, dz=-5)
        self.assertIn("zimbabwe.curved_walls", failed(document))

    def test_a_straightened_wall_is_caught(self):
        document = deepcopy(self.great.document)
        # Pull one flank of the foot course onto a straight line.
        for p in document.placements:
            if p.role == "outer_wall" and p.parameters["course"] == 0 and p.position[0] > 0 and abs(p.position[2]) < 9:
                p.position = (20.5, p.position[1], p.position[2])
        self.assertIn("zimbabwe.curved_walls", failed(document))

    def test_a_wall_thicker_at_the_head_is_caught(self):
        document = deepcopy(self.great.document)
        top = max(p.parameters["course"] for p in document.placements if p.role == "wall_lining")
        for p in document.placements:
            if p.role == "wall_lining" and p.parameters["course"] == top and abs(p.position[0]) < 2 and p.position[2] > 0:
                shift(p, dz=-2)
        self.assertIn("zimbabwe.battered_walls", failed(document))

    def test_a_hollow_or_engaged_tower_is_caught(self):
        hollow = deepcopy(self.great.document)
        hollow.placements = [p for p in hollow.placements if p.role != "tower_core"]
        self.assertIn("zimbabwe.solid_tower", failed(hollow))
        engaged = deepcopy(self.great.document)
        for p in engaged.placements:
            if p.role in ("tower_stone", "tower_core"):
                shift(p, dz=3)
        self.assertIn("zimbabwe.solid_tower", failed(engaged))

    def test_a_band_below_the_wall_head_is_caught(self):
        document = deepcopy(self.great.document)
        shift(next(p for p in document.placements if p.role == "decorated_band"), dy=-2)
        self.assertIn("zimbabwe.crowning_band", failed(document))


if __name__ == "__main__":
    unittest.main()
