"""Seeded Rumah Gadang plans and measurable construction rules."""
from copy import deepcopy
import unittest

from core.architecture import build, describe_product, rebuild, roll_request
from core.architecture.agent import series_recipe
from core.architecture.construction import construction_report

PRODUCT = "minangkabau-rumah-gadang"


class MinangkabauTests(unittest.TestCase):
    def test_catalog_and_seeded_agent_requests(self):
        product = describe_product(PRODUCT)
        self.assertEqual(product["architect_id"], "minangkabau")
        self.assertEqual(product["variants"], ["auto", "random", "bodi-caniago", "koto-piliang"])
        requests = [roll_request(PRODUCT, seed, detail="concept").to_dict() for seed in range(12)]
        recipe = series_recipe(12, 0, (PRODUCT,), detail="concept")
        self.assertEqual([entry["request"] for entry in recipe["buildings"]], requests)
        self.assertEqual({r["variant"] for r in requests}, {"bodi-caniago", "koto-piliang"})
        self.assertGreater(len({r["bays"] for r in requests}), 1)

    def test_both_plans_build_and_replay(self):
        for variant in ("bodi-caniago", "koto-piliang"):
            with self.subTest(variant=variant):
                request = roll_request(PRODUCT, 24, variant=variant, detail="concept")
                result = build(request)
                self.assertEqual(result.quality.status, "pass",
                                 [c.code for c in result.quality.checks if c.status == "fail"])
                doc = result.document
                roofs = {p.parameters["peak"] for p in doc.placements if p.role == "gonjong"}
                self.assertEqual(roofs, {"west", "east", "front-west", "front-east"})
                platforms = {p.parameters["end"] for p in doc.placements if p.role == "anjuang_floor"}
                self.assertEqual(platforms, {-1, 1} if variant == "koto-piliang" else set())
                self.assertEqual(doc.fingerprint(), build(request).document.fingerprint())
                self.assertIn("minangkabau-timber", construction_report(doc, result.quality)["traditions"])

    def test_intrinsic_audits_reject_a_shifted_peak_and_floating_post(self):
        doc = deepcopy(build(roll_request(PRODUCT, 7, detail="concept")).document)
        doc.metadata.clear()
        for part in doc.placements:
            if part.role == "gonjong" and part.parameters["peak"] == "west":
                x, y, z = part.position
                part.position = (x, y - 1.8, z)
        post = next(p for p in doc.placements if p.role == "floor_post")
        x, y, z = post.position
        post.position = (x, y + .25, z)
        failed = {c.code for c in rebuild(doc).quality.checks if c.status == "fail"}
        self.assertLessEqual({"minangkabau.swept_gonjong", "minangkabau.raised_frame"}, failed)
