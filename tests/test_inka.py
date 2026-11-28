"""The Imperial Inka kancha and its measured building rules."""
from copy import deepcopy
from functools import lru_cache
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild
from core.architecture.construction import construction_report, craft_catalog


PRODUCT = "inka-kancha"


@lru_cache(maxsize=None)
def kancha(detail="concept", seed=73):
    return build(BuildRequest(product=PRODUCT, detail=detail, seed=seed))


class InkaTests(unittest.TestCase):
    def test_catalog_identifies_imperial_period_and_andean_tradition(self):
        product = describe_product(PRODUCT)
        self.assertEqual(product["architect_id"], "inka")
        self.assertIn("Andes", product["rules"]["region"])
        self.assertIn("granite", product["substances"])
        self.assertIn("imperial-inka", craft_catalog()["traditions"])

    def test_kancha_builds_with_fitted_stones_and_roundtrips(self):
        result = kancha()
        self.assertEqual(result.quality.status, "pass", result.quality.counts)
        document = result.document
        self.assertEqual(document.request.style, "imperial-inka")
        cuts = [p for p in document.placements if p.parameters.get("unit") == "fitted_stone"]
        self.assertTrue(cuts and all(p.block_type == "MESH" for p in cuts))
        self.assertEqual(len({p.parameters["pavilion"] for p in document.placements if p.role == "thatch"}), 4)
        self.assertIn("imperial-inka", construction_report(document, result.quality)["traditions"])
        self.assertEqual(BuildDocument.from_dict(document.to_dict()).fingerprint(), document.fingerprint())
        self.assertEqual(build(document.request).document.fingerprint(), document.fingerprint())

    def test_geometry_audit_catches_lifted_lintel_and_covered_court(self):
        document = deepcopy(kancha().document)
        document.metadata.clear()
        lintel = next(p for p in document.placements if p.role == "lintel" and p.parameters["pavilion"] == "north")
        x, y, z = lintel.position
        lintel.position = (x, y + .3, z)
        thatch = next(p for p in document.placements if p.role == "thatch")
        thatch.position = (0., thatch.position[1], 0.)
        failures = {c.code for c in rebuild(document).quality.checks if c.status == "fail"}
        self.assertLessEqual({"inka.trapezoidal_openings", "inka.open_court"}, failures)

    def test_detail_refines_stonework_without_enlarging_the_plot(self):
        coarse = kancha()
        fine = kancha("presentation")
        self.assertEqual(fine.quality.status, "pass", fine.quality.counts)
        self.assertGreater(len(fine.document.placements), len(coarse.document.placements))
        self.assertEqual(fine.quality.metrics["bounds_size"], coarse.quality.metrics["bounds_size"])

    def test_rectangular_plot_with_three_doorways_per_hall(self):
        result = build(BuildRequest(product=PRODUCT, width=33, height=7, depth=21, bays=3, detail="concept"))
        self.assertEqual(result.quality.status, "pass", result.quality.counts)
        lintels = [p for p in result.document.placements if p.role == "lintel"]
        self.assertEqual(len(lintels), 13)  # one enclosure gate, plus three per hall
