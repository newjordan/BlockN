"""Angkor-period Khmer temple-mountain geometry and craft checks."""
from copy import deepcopy
from functools import lru_cache
import unittest

from core.architecture import BuildDocument, BuildRequest, build, describe_product, rebuild
from core.architecture.construction import construction_report

PRODUCT = "khmer-temple-mountain"


@lru_cache(maxsize=None)
def temple(detail="concept"):
    return build(BuildRequest(product=PRODUCT, detail=detail, seed=73))


def failed(result):
    return {c.code for c in result.quality.checks if c.status == "fail"}


class KhmerTests(unittest.TestCase):
    def test_angkor_style_is_discoverable(self):
        product = describe_product(PRODUCT)
        self.assertEqual(product["architect_id"], "khmer")
        self.assertIn("Cambodia", product["rules"]["region"])
        self.assertIn("angkor-wat", product["variants"])
        self.assertEqual(set(product["substances"]), {"sandstone", "laterite"})

    def test_temple_builds_and_replays(self):
        result = temple()
        self.assertEqual(result.quality.status, "pass", failed(result))
        document = result.document
        self.assertEqual(document.request.style, "angkor-temple-mountain")
        self.assertEqual(len([p for p in document.placements if p.role == "tower_finial"]), 5)
        self.assertTrue(all(block.mesh.is_watertight and block.mesh.volume > 0 for block in result.scene.blocks))
        self.assertEqual(BuildDocument.from_dict(document.to_dict()).fingerprint(), document.fingerprint())
        self.assertEqual(build(document.request).document.fingerprint(), document.fingerprint())
        self.assertIn(document.request.style, construction_report(document, result.quality)["traditions"])

    def test_audits_catch_displaced_shrine_and_broken_corbel_from_placements(self):
        document = deepcopy(temple().document)
        document.metadata.clear()
        peak = next(p for p in document.placements if p.role == "tower_finial" and p.parameters["shrine"] == "center")
        x, y, z = peak.position
        peak.position = (x + 2, y, z)
        gallery = next(p for p in document.placements if p.role == "gallery_corbel")
        x, y, z = gallery.position
        gallery.position = (x, y + .6, z)
        self.assertLessEqual({"khmer.five_tower_quincunx", "khmer.inward_corbel_steps"}, failed(rebuild(document)))

    def test_presentation_keeps_plot_and_custom_bays_are_buildable(self):
        fine = temple("presentation")
        self.assertEqual(fine.quality.status, "pass", failed(fine))
        self.assertGreater(len(fine.document.placements), len(temple().document.placements))
        self.assertEqual(fine.quality.metrics["bounds_size"], temple().quality.metrics["bounds_size"])
        custom = build(BuildRequest(product=PRODUCT, width=50, depth=41, height=27, bays=5, detail="concept"))
        self.assertEqual(custom.quality.status, "pass", failed(custom))
        for controls in ({"levels": 6}, {"bays": 2}, {"height": 11}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                build(BuildRequest(product=PRODUCT, **controls))
