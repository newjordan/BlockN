"""Tibetan fortress-palace: both massings, the sonnet lines, and audits measured from the members."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture import sonnet

PRODUCT = "tibetan-palace"


def top(p):
    return p.position[1] + p.dimensions[1] / 2


def failures(document, damage):
    document = deepcopy(document)
    document.metadata.clear()           # the audits must read the geometry, not the claims about it
    damage(document)
    return {c.code for c in rebuild(document).quality.checks if c.status == "fail"}


class TibetanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {variant: build(roll_request(PRODUCT, 20 + i, variant=variant, detail="concept"))
                       for i, variant in enumerate(("potala", "dzong"))}

    def test_both_massings_build_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "tibetan")
        for variant, result in self.results.items():
            with self.subTest(variant=variant):
                self.assertEqual(result.quality.status, "pass")
                document, request = result.document, result.document.request
                self.assertEqual(document.fingerprint(), build(request).document.fingerprint())
                size = result.quality.metrics["bounds_size"]
                for got, asked in zip(size, (request.width, request.height, request.depth)):
                    self.assertAlmostEqual(got, asked, places=4)
                self.assertEqual(sonnet.verse(request), document.metadata["tibetan_rules"]["verse"])
                self.assertTrue(any(p.parameters.get("unit") == "core" for p in document.placements))
        roofs = {p.role for p in self.results["dzong"].document.placements}
        self.assertIn("range_roof", roofs)
        self.assertNotIn("range_roof", {p.role for p in self.results["potala"].document.placements})

    def test_seed_writes_a_different_palace(self):
        one, two = (build(BuildRequest(product=PRODUCT, detail="concept", seed=seed)) for seed in (1, 2))
        self.assertNotEqual(one.document.fingerprint(), two.document.fingerprint())
        self.assertEqual(one.quality.metrics["bounds_size"], two.quality.metrics["bounds_size"])

    def test_pinned_lines_are_honoured(self):
        pins = {"roofs": "banners", "stair": "axial", "frieze": "double", "wings": "level"}
        request = BuildRequest(product=PRODUCT, variant="potala", detail="concept", seed=5, parameters=dict(pins))
        self.assertEqual(sonnet.verse(request), pins)
        result = build(request)
        self.assertEqual(result.quality.status, "pass")
        placements = result.document.placements
        self.assertEqual(result.document.metadata["tibetan_rules"]["verse"], pins)
        self.assertEqual(sum(p.role == "banner" for p in placements), 5)
        self.assertFalse(any(p.role == "pavilion_roof" for p in placements))
        self.assertEqual({p.parameters["block"] for p in placements if p.role == "wing_wall"},
                         {"left-wing-0", "right-wing-0"})
        self.assertAlmostEqual(max(top(p) for p in placements), 29.0, places=4)
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", parameters={"roofs": "thatch"}))
        with self.assertRaises(ValueError):
            build(BuildRequest(product=PRODUCT, detail="concept", width=80))

    def test_windows_are_wider_at_the_sill_and_larger_toward_the_roof(self):
        frames = [p for p in self.results["potala"].document.placements
                  if p.role == "window_frame" and p.parameters["block"] == "central" and p.parameters["face"] == "f"]
        heads = {p.parameters["row"]: p.dimensions[0] for p in frames if p.parameters["member"] == "head"}
        sills = {p.parameters["row"]: p.dimensions[0] for p in frames if p.parameters["member"] == "sill"}
        self.assertGreater(min(heads), 0)                       # row 0 is blind
        for row in heads:
            self.assertGreater(sills[row], heads[row])
        self.assertGreater(heads[max(heads)], heads[min(heads)])

    def test_audits_fail_when_a_rule_is_broken(self):
        document = self.results["potala"].document

        def bulge(doc):
            stones = [p for p in doc.placements if p.role == "palace_wall"]
            crest = max(map(top, stones))
            for p in stones:
                if top(p) > crest - .9 and p.position[0] > 0:
                    p.position = (p.position[0] + 3.0, *p.position[1:])
        self.assertIn("tibetan.battered_walls", failures(document, bulge))

        def low_window(doc):
            foot = min(p.position[1] for p in doc.placements if p.role == "palace_wall")
            jamb = next(p for p in doc.placements if p.role == "window_frame" and p.parameters["member"] == "jamb"
                        and p.parameters["block"] == "central")
            jamb.position = (jamb.position[0], foot + 1.2, jamb.position[2])
        self.assertIn("tibetan.windows_grow", failures(document, low_window))

        def raise_wing(doc):
            crest = max(top(p) for p in doc.placements if p.role == "palace_wall")
            wing = next(p for p in doc.placements if p.role == "wing_wall")
            wing.position = (wing.position[0], crest, wing.position[2])
        self.assertIn("tibetan.central_block", failures(document, raise_wing))

        def drop_frieze(doc):
            for p in doc.placements:
                if p.role == "frieze":
                    p.position = (p.position[0], p.position[1] - 2.0, p.position[2])
        self.assertIn("tibetan.frieze_band", failures(document, drop_frieze))

        def break_stair(doc):
            steps = sorted((p for p in doc.placements if p.role == "stair_step"), key=top)
            doomed = {p.id for p in steps[len(steps) // 2:len(steps) // 2 + 6]}
            doc.placements = [p for p in doc.placements if p.id not in doomed]
        self.assertIn("tibetan.approach_stair", failures(document, break_stair))

    def test_variations_deal_distinct_verses(self):
        requests = sonnet.variations(PRODUCT, 8, seed=3, detail="concept", hold={"roofs": "banners"})
        self.assertEqual(len({sonnet.stanza(r) for r in requests}), 8)
        self.assertTrue(all(r.parameters["roofs"] == "banners" for r in requests))


if __name__ == "__main__":
    unittest.main()
