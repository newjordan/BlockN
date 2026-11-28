"""Lalibela rock-hewn churches: variants, lines and audits measured from the members."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture import sonnet

PRODUCT = "lalibela-church"
CHURCH = {"plinth_step", "entry_step", "hewn_band", "blind_panel", "window_frame", "pillar", "colonnade_pier",
          "church_roof", "roof_relief"}


def failed(document):
    document.metadata.clear()       # rules must read the geometry, not the claims about it
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


def moved(p, dx=0., dy=0., dz=0.):
    p.position = (p.position[0] + dx, p.position[1] + dy, p.position[2] + dz)


class LalibelaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {variant: build(roll_request(PRODUCT, 20 + i, variant=variant, detail="concept"))
                       for i, variant in enumerate(("cruciform", "colonnaded"))}

    def test_variants_build_fill_the_plot_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "lalibela")
        for variant, result in self.results.items():
            with self.subTest(variant=variant):
                request = result.document.request
                self.assertEqual(result.quality.status, "pass")
                self.assertEqual(result.document.fingerprint(), build(request).document.fingerprint())
                size = result.quality.metrics["bounds_size"]
                for got, asked in zip(size, (request.width, request.height, request.depth)):
                    self.assertAlmostEqual(got, asked, places=6)
                count = result.document.metadata["lalibela_rules"]["block_count"]
                self.assertGreater(count["core_blocks"], 0)
                self.assertGreater(count["monoliths"], 100)
                roles = {p.role for p in result.document.placements}
                self.assertEqual("colonnade_pier" in roles, variant == "colonnaded")

    def test_seed_changes_the_church_inside_the_same_plot(self):
        one, two = (build(BuildRequest(product=PRODUCT, detail="concept", seed=seed)) for seed in (1, 2))
        self.assertNotEqual(one.document.fingerprint(), two.document.fingerprint())
        self.assertEqual(one.quality.metrics["bounds_size"], two.quality.metrics["bounds_size"])

    def test_lines_are_seeded_pinned_and_validated(self):
        for variant, result in self.results.items():
            self.assertEqual(sonnet.verse(result.document.request),
                             result.document.metadata["lalibela_rules"]["verse"])
        request = roll_request(PRODUCT, 5, variant="cruciform", detail="concept")
        request.parameters.update(approach="tunnel", windows="aksumite", plinth="two-step")
        result = build(request)
        self.assertEqual(result.quality.status, "pass")
        verse = result.document.metadata["lalibela_rules"]["verse"]
        self.assertEqual((verse["approach"], verse["windows"], verse["plinth"]), ("tunnel", "aksumite", "two-step"))
        placements = result.document.placements
        self.assertTrue(any(p.role == "tunnel_lintel" for p in placements))
        self.assertEqual({p.parameters["member"] for p in placements if p.role == "window_frame"}, {"monkey-head"})
        self.assertEqual({p.parameters["step"] for p in placements if p.role == "plinth_step"}, {0, 1})
        request.parameters["approach"] = "open-stair"
        self.assertFalse(any(p.role == "tunnel_lintel" for p in build(request).document.placements))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, parameters={"windows": "gothic"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, width=30, height=20, depth=40))

    def test_church_must_stand_free_in_its_pit_below_the_ground(self):
        document = deepcopy(self.results["cruciform"].document)
        for p in document.placements:
            if p.role in CHURCH:
                moved(p, dx=1.0)
        self.assertIn("lalibela.freestanding_in_pit", failed(document))

        document = deepcopy(self.results["cruciform"].document)
        moved(next(p for p in document.placements if p.role == "roof_relief"), dy=.5)
        self.assertIn("lalibela.roof_at_ground", failed(document))

        document = deepcopy(self.results["colonnaded"].document)
        for p in document.placements:
            if p.role in {"church_roof", "roof_relief"}:
                moved(p, dy=-.5)
        self.assertIn("lalibela.roof_at_ground", failed(document))

    def test_plinth_plan_and_stair_audits_detect_damage(self):
        document = deepcopy(self.results["cruciform"].document)
        top = max(p.parameters["step"] for p in document.placements if p.role == "plinth_step")
        for p in document.placements:
            if p.role == "plinth_step" and p.parameters["step"] == top:
                moved(p, dx=1.5)
        self.assertIn("lalibela.stepped_plinth", failed(document))

        document = deepcopy(self.results["cruciform"].document)
        north = max(p.position[2] for p in document.placements if p.role == "hewn_band")
        document.placements = [p for p in document.placements
                               if not (p.role == "hewn_band" and p.position[2] > north - 1.5)]
        self.assertIn("lalibela.plan", failed(document))

        document = deepcopy(self.results["colonnaded"].document)
        side = sorted((p for p in document.placements
                       if p.role == "colonnade_pier" and p.parameters["side"] == "z+1"), key=lambda p: p.position[2])
        document.placements.remove(side[len(side) // 2])
        self.assertIn("lalibela.plan", failed(document))

        document = deepcopy(self.results["colonnaded"].document)
        document.placements = [p for p in document.placements
                               if not (p.role == "stair_step" and 20 <= p.parameters["step"] < 26)]
        self.assertIn("lalibela.descending_stair", failed(document))


if __name__ == "__main__":
    unittest.main()
