"""Muscovite church: both crowns, the lines of its sonnet, and rules measured from the members."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture import sonnet

PRODUCT = "muscovite-church"


def failed(result, damage):
    """Codes that fail after ``damage`` edits a copy whose metadata is gone."""
    document = deepcopy(result.document)
    document.metadata.clear()
    damage(document)
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


def shift(document, chosen, dx=0., dy=0., dz=0.):
    for p in document.placements:
        if chosen(p):
            p.position = (p.position[0] + dx, p.position[1] + dy, p.position[2] + dz)


def drop(document, chosen):
    document.placements = [p for p in document.placements if not chosen(p)]


class MuscoviteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = {variant: roll_request(PRODUCT, 31 + n, variant=variant, detail="concept")
                        for n, variant in enumerate(("five-domed", "tent-roof"))}
        cls.built = {variant: build(request) for variant, request in cls.requests.items()}

    def test_variants_build_fill_the_plot_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "muscovite")
        for variant, result in self.built.items():
            with self.subTest(variant=variant):
                request = self.requests[variant]
                self.assertEqual(result.quality.status, "pass")
                for got, asked in zip(result.quality.metrics["bounds_size"],
                                      (request.width, request.height, request.depth)):
                    self.assertAlmostEqual(got, asked, places=4)
                self.assertEqual(result.document.fingerprint(), build(request).document.fingerprint())
                crowns = {p.parameters["crown"] for p in result.document.placements if p.role == "dome"}
                self.assertEqual(crowns, set(range(5)) if variant == "five-domed" else {0})
                self.assertEqual({p.role for p in result.document.placements if p.role == "tent"},
                                 {"tent"} if variant == "tent-roof" else set())

    def test_the_verse_is_read_without_building_and_lines_can_be_pinned(self):
        for variant, result in self.built.items():
            self.assertEqual(sonnet.verse(self.requests[variant]),
                             result.document.metadata["muscovite_rules"]["verse"])
        request = BuildRequest(product=PRODUCT, variant="five-domed", detail="concept", seed=5,
                               parameters={"domes": "helmet", "apses": "one", "gables": "kokoshniki"})
        result = build(request)
        self.assertEqual(result.quality.status, "pass")
        verse = result.document.metadata["muscovite_rules"]["verse"]
        self.assertEqual((verse["domes"], verse["apses"], verse["gables"]), ("helmet", "one", "kokoshniki"))
        self.assertEqual(verse, sonnet.verse(request))
        self.assertEqual({p.parameters["apse"] for p in result.document.placements if p.role == "apse_wall"}, {0})
        self.assertEqual({p.parameters["tier"] for p in result.document.placements
                          if p.role == "gable" and p.parameters["shape"] == "kokoshniki"}, {0, 1})
        three = build(BuildRequest(product=PRODUCT, detail="concept", seed=5, parameters={"apses": "three"}))
        self.assertEqual({p.parameters["apse"] for p in three.document.placements if p.role == "apse_wall"}, {0, 1, 2})
        # Whatever the lines say, the church fills the same plot to the same height.
        self.assertEqual(result.quality.metrics["bounds_size"], three.quality.metrics["bounds_size"])
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, parameters={"domes": "geodesic"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, width=20, height=20, depth=30))

    def test_five_domed_rules_fail_when_broken(self):
        result = self.built["five-domed"]
        self.assertIn("muscovite.domes_on_drums",
                      failed(result, lambda d: shift(d, lambda p: p.role == "drum" and p.parameters["crown"] == 0, dx=1.5)))
        self.assertIn("muscovite.crown_hierarchy",
                      failed(result, lambda d: drop(d, lambda p: p.parameters.get("crown") == 3)))
        self.assertIn("muscovite.crown_hierarchy",
                      failed(result, lambda d: shift(d, lambda p: p.parameters.get("crown") == 2, dx=.8)))
        self.assertIn("muscovite.eastern_apses",
                      failed(result, lambda d: shift(d, lambda p: p.role == "apse_wall", dz=-3.)))
        self.assertIn("muscovite.eastern_apses",
                      failed(result, lambda d: shift(d, lambda p: p.role in ("apse_wall", "apse_roof"), dy=30.)))
        self.assertIn("muscovite.crosses",
                      failed(result, lambda d: drop(d, lambda p: p.role == "cross" and p.parameters["crown"] == 2)))
        self.assertIn("muscovite.crosses",
                      failed(result, lambda d: shift(d, lambda p: p.role == "cross" and p.parameters["crown"] == 0, dz=.6)))

    def test_tent_rules_fail_when_broken(self):
        result = self.built["tent-roof"]
        self.assertIn("muscovite.crown_hierarchy",
                      failed(result, lambda d: drop(d, lambda p: p.role == "tent" and p.parameters["tier"] > 0)))
        self.assertIn("muscovite.domes_on_drums",
                      failed(result, lambda d: drop(d, lambda p: p.role == "drum")))
        self.assertIn("muscovite.domes_on_drums",
                      failed(result, lambda d: shift(d, lambda p: p.role in ("drum", "dome", "cross"), dx=4.)))


if __name__ == "__main__":
    unittest.main()
