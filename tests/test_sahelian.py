"""Djenné-inspired Sudano-Sahelian mosque builds and geometry audits."""
from copy import deepcopy
from functools import lru_cache
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild
from core.architecture.construction import construction_report, craft_catalog


PRODUCT = "sahelian-mosque"


@lru_cache(maxsize=None)
def built(detail="concept"):
    return build(BuildRequest(product=PRODUCT, detail=detail, seed=73))


def failures(result):
    return {check.code for check in result.quality.checks if check.status == "fail"}


class SahelianTests(unittest.TestCase):
    def test_catalog_documents_period_region_and_materials(self):
        product = describe_product(PRODUCT)
        self.assertEqual(product["architect_id"], "sahelian")
        self.assertIn("djenne", product["variants"])
        self.assertIn("Mali", product["rules"]["region"])
        self.assertIn("mud-brick", product["substances"])
        self.assertIn("sudano-sahelian", craft_catalog()["traditions"])

    def test_default_mosque_builds_and_replays(self):
        result = built()
        self.assertEqual(result.quality.status, "pass", failures(result))
        document = result.document
        self.assertEqual(document.request.style, "sudano-sahelian")
        self.assertIn("sudano-sahelian", construction_report(document, result.quality)["traditions"])
        self.assertTrue(all(block.mesh.is_watertight and block.mesh.volume > 0 for block in result.scene.blocks))
        restored = BuildDocument.from_dict(document.to_dict())
        self.assertEqual(restored.fingerprint(), document.fingerprint())
        self.assertEqual(build(document.request).document.fingerprint(), document.fingerprint())

    def test_toron_must_project_from_and_embed_in_earthen_hosts(self):
        document = deepcopy(built().document)
        document.metadata.clear()
        toron = next(p for p in document.placements if p.role == "toron")
        x, y, z = toron.position
        toron.position = (x, y, z + toron.dimensions[2] * 2)
        self.assertIn("sahelian.embedded_toron", failures(rebuild(document)))

    def test_roof_must_not_cover_the_courtyard(self):
        document = deepcopy(built().document)
        document.metadata.clear()
        roof = next(p for p in document.placements if p.role == "roof_deck")
        x, y, _ = roof.position
        roof.position = (x, y, -document.request.depth / 4 if document.request.depth else -12)
        self.assertIn("sahelian.open_courtyard", failures(rebuild(document)))

    def test_all_detail_levels_and_seeded_controls(self):
        for detail in ("standard", "presentation"):
            with self.subTest(detail=detail):
                result = built(detail)
                self.assertEqual(result.quality.status, "pass", failures(result))
        result = build(BuildRequest(product=PRODUCT, variant="random", detail="concept", seed=19,
                                    width=32, height=14, depth=34, bays=7))
        self.assertEqual(result.quality.status, "pass", failures(result))
        self.assertNotEqual(result.document.fingerprint(), built().document.fingerprint())
        for controls in ({"bays": 4}, {"levels": 2}, {"height": 65}, {"openings": "mixed"}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(product=PRODUCT, **controls))
