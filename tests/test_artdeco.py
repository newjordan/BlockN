"""Art Deco setback tower: variants, sonnet lines, and zoning rules measured from the members."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture import sonnet

PRODUCT = "artdeco-tower"


def request(**controls):
    parameters = {"setbacks": "symmetric", **controls.pop("parameters", {})}
    return BuildRequest(product=PRODUCT, detail="concept", seed=73, parameters=parameters, **controls)


def failed(document, change):
    document = deepcopy(document)
    document.metadata.clear()          # rules must read the geometry, not the claims about it
    change(document.placements)
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


def stretch(p, axis, factor):
    dims = list(p.dimensions)
    dims[axis] *= factor
    p.dimensions = tuple(dims)


class ArtDecoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = build(request())
        cls.document = cls.result.document

    def test_variants_build_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "artdeco")
        for variant in ("spire-crown", "stepped-crown"):
            with self.subTest(variant=variant):
                rolled = roll_request(PRODUCT, 11, variant=variant, detail="concept")
                result = build(rolled)
                self.assertEqual(result.quality.status, "pass")
                self.assertEqual(result.document.fingerprint(), build(rolled).document.fingerprint())
                size = result.quality.metrics["bounds_size"]
                for got, asked in zip(size, (rolled.width, rolled.height, rolled.depth)):
                    self.assertAlmostEqual(got, asked, places=4)
                self.assertTrue(any(p.parameters.get("unit") == "core" for p in result.document.placements))

    def test_envelope_follows_the_zoning_law(self):
        self.assertEqual(self.result.quality.status, "pass")
        rules = self.document.metadata["artdeco_rules"]
        self.assertGreaterEqual(rules["tiers"], 3)
        self.assertLessEqual(rules["tower_share_of_lot"], .25)
        piers = [p for p in self.document.placements if p.role == "pier"]
        tower = max(p.parameters["tier"] for p in piers)
        # One pier per bay line per tier: none is cut at a floor.
        spots = [(p.parameters["tier"], p.position[0], p.position[2]) for p in piers]
        self.assertEqual(len(spots), len(set(spots)))
        tallest = max(p.position[1] + p.dimensions[1] / 2 for p in self.document.placements)
        self.assertAlmostEqual(tallest, 120.0, places=4)
        self.assertEqual({p.substance for p in self.document.placements if p.role == "frame_column"}, {"wrought-iron"})
        self.assertGreater(len([p for p in piers if p.parameters["tier"] == tower]), 4)

    def test_lines_are_seeded_and_can_be_pinned(self):
        rolled = roll_request(PRODUCT, 5, detail="concept")
        document = build(rolled).document
        self.assertEqual(sonnet.verse(rolled), document.metadata["artdeco_rules"]["verse"])
        pinned = request(variant="stepped-crown", parameters={"ornament": "ziggurat", "piers": "paired",
                                                              "portal": "stepped-arch"})
        built = build(pinned)
        self.assertEqual(built.quality.status, "pass")
        verse = built.document.metadata["artdeco_rules"]["verse"]
        self.assertEqual((verse["ornament"], verse["piers"], verse["portal"], verse["setbacks"]),
                         ("ziggurat", "paired", "stepped-arch", "symmetric"))
        roles = {p.role for p in built.document.placements}
        self.assertLessEqual({"mullion", "lantern_post", "lantern_glazing"}, roles)
        self.assertTrue(all(p.parameters["motif"] == "ziggurat" for p in built.document.placements
                            if p.role == "setback_ornament"))
        with self.assertRaises(ValueError):
            build(request(parameters={"ornament": "gargoyle"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, width=44, height=90, depth=44))     # too squat to be a tower

    def test_street_stepped_tower_stands_back_from_the_street(self):
        result = build(request(parameters={"setbacks": "street-stepped"}))
        self.assertEqual(result.quality.status, "pass")
        piers = [p for p in result.document.placements if p.role == "pier"]
        tower = max(p.parameters["tier"] for p in piers)
        centre = sum(p.position[2] for p in piers if p.parameters["tier"] == tower)
        self.assertGreater(centre, 1.0)

    def test_audits_fail_when_the_rules_are_broken(self):
        def tier(placements, k, role="pier"):
            return [p for p in placements if p.role == role and p.parameters.get("tier") == k]
        top = max(p.parameters["tier"] for p in self.document.placements if p.role == "pier")

        def pierce_the_plane(placements):
            pier = tier(placements, 1)[0]
            stretch(pier, 1, 1.6)
            pier.position = (pier.position[0], pier.position[1] + pier.dimensions[1] * .1875, pier.position[2])
        self.assertIn("artdeco.sky_exposure", failed(self.document, pierce_the_plane))

        def fatten_the_tower(placements):
            pier = max(tier(placements, top), key=lambda p: p.position[0] + p.position[2])
            pier.position = (pier.position[0] + 9, pier.position[1], pier.position[2] + 9)
        self.assertIn("artdeco.quarter_lot_tower", failed(self.document, fatten_the_tower))

        def overhang(placements):
            for p in tier(placements, 1):
                p.position = (p.position[0] + 6, p.position[1], p.position[2])
        self.assertIn("artdeco.receding_tiers", failed(self.document, overhang))

        def cut_a_pier(placements):
            pier = tier(placements, 0)[3]
            stretch(pier, 1, .5)
            pier.position = (pier.position[0], pier.position[1] - pier.dimensions[1] / 2, pier.position[2])
        self.assertIn("artdeco.unbroken_piers", failed(self.document, cut_a_pier))

        def shift_a_pier(placements):
            pier = tier(placements, 2)[1]
            pier.position = (pier.position[0] + 1.1, pier.position[1], pier.position[2])
        self.assertIn("artdeco.unbroken_piers", failed(self.document, shift_a_pier))

        def swell_the_crown(placements):
            crown = [p for p in placements if p.role == "crown_tier"]
            last = max(p.parameters["crown_tier"] for p in crown)
            stretch(next(p for p in crown if p.parameters["crown_tier"] == last and p.parameters["face"] == "front"), 0, 3)
        self.assertIn("artdeco.tapering_crown", failed(self.document, swell_the_crown))


if __name__ == "__main__":
    unittest.main()
