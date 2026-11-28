"""Swahili plan variants and audits measured from actual members."""
from copy import deepcopy
import unittest

from core.architecture import BuildRequest, build, describe_product, rebuild, roll_request
from core.architecture.construction import construction_report

PRODUCT = "swahili-courtyard-house"


class SwahiliTests(unittest.TestCase):
    def test_plan_variants_build_and_replay(self):
        self.assertEqual(describe_product(PRODUCT)["architect_id"], "swahili")
        for count, variant in enumerate(("single-court", "double-court"), 1):
            with self.subTest(variant=variant):
                request = roll_request(PRODUCT, 20 + count, variant=variant, detail="concept")
                result = build(request)
                self.assertEqual(result.quality.status, "pass")
                document = result.document
                self.assertEqual({p.parameters["court"] for p in document.placements if p.role == "court_paving"},
                                 set(range(count)))
                self.assertEqual(document.fingerprint(), build(request).document.fingerprint())
                self.assertIn("swahili-stone-house", construction_report(document, result.quality)["traditions"])

    def test_short_plot_has_room_for_two_open_courts(self):
        result = build(BuildRequest(product=PRODUCT, width=10, depth=23, height=6, bays=4,
                                    detail="concept", seed=73, variant="double-court"))
        self.assertEqual(result.quality.status, "pass")

    def test_audits_detect_a_covered_court_and_aligned_entrances(self):
        request = roll_request(PRODUCT, 21, variant="single-court", detail="concept")
        document = deepcopy(build(request).document)
        document.metadata.clear()
        paving = [p for p in document.placements if p.role == "court_paving"]
        roof = next(p for p in document.placements if p.role == "roof_deck")
        roof.position = (roof.position[0], roof.position[1], sum(p.position[2] for p in paving) / len(paving))
        lintels = [p for p in document.placements if p.role == "entry_lintel" and p.position[1] < 2.5]
        street = next(p for p in lintels if p.parameters.get("edge") == "street")
        screen = next(p for p in lintels if p.parameters.get("edge") == "screen")
        screen.position = (street.position[0], *screen.position[1:])
        self.assertLessEqual({"swahili.open_courts", "swahili.screened_entry"},
                             {c.code for c in rebuild(document).quality.checks if c.status == "fail"})
