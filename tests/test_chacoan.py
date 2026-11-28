"""Chacoan great-house layout, mesh contracts and damage checks."""
from copy import deepcopy
import unittest

from core.architecture import BuildDocument, build, describe_product, rebuild, roll_request
from core.architecture.agent import series_recipe
from core.architecture.construction import construction_report

PRODUCT = "chacoan-great-house"
VARIANTS = ("single-kiva", "paired-kivas")


class ChacoanTests(unittest.TestCase):
    def test_catalog_and_batch_use_the_same_sampler(self):
        descriptor = describe_product(PRODUCT)
        self.assertEqual(descriptor["architect_id"], "chacoan")
        self.assertEqual(descriptor["variants"], ["auto", "random", *VARIANTS])
        rolls = [roll_request(PRODUCT, seed, detail="concept").to_dict() for seed in range(12)]
        recipe = series_recipe(12, 0, (PRODUCT,), detail="concept")
        self.assertEqual(rolls, [entry["request"] for entry in recipe["buildings"]])
        self.assertEqual({r["variant"] for r in rolls}, set(VARIANTS))
        self.assertGreater(len({r["bays"] for r in rolls}), 1)

    def test_both_room_ranges_build_and_roundtrip(self):
        for count, variant in enumerate(VARIANTS, 1):
            with self.subTest(variant=variant):
                request = roll_request(PRODUCT, count, variant=variant, detail="concept")
                result = build(request)
                doc = result.document
                self.assertEqual(result.quality.status, "pass",
                                 [c.code for c in result.quality.checks if c.status == "fail"])
                self.assertEqual({p.parameters["kiva"] for p in doc.placements if p.role == "kiva_floor"},
                                 set(range(count)))
                self.assertEqual({p.parameters["tier"] for p in doc.placements if p.role == "terrace_deck"},
                                 set(range(count + 1)))
                plaza = max(p.position[1] + p.dimensions[1] / 2 for p in doc.placements if p.role == "plaza_surface")
                self.assertTrue(all(p.position[1] + p.dimensions[1] / 2 < plaza - .8
                                    for p in doc.placements if p.role == "kiva_floor"))
                self.assertEqual(doc.fingerprint(), BuildDocument.from_dict(doc.to_dict()).fingerprint())
                self.assertIn("chacoan-sandstone", construction_report(doc, result.quality)["traditions"])

    def test_audit_without_metadata_detects_plugged_kiva_and_displaced_terrace(self):
        result = build(roll_request(PRODUCT, 9, variant="paired-kivas", detail="concept"))
        doc = deepcopy(result.document)
        doc.metadata.clear()
        floor = next(p for p in doc.placements if p.role == "kiva_floor")
        cap = next(p for p in doc.placements if p.role == "plaza_surface" and p.block_type == "CUBE")
        cap.position = (floor.position[0], floor.position[1] + .45, floor.position[2])
        for p in doc.placements:
            if p.role == "terrace_deck" and p.parameters["tier"] == 1:
                x, y, z = p.position
                p.position = (x, y, z - 4)
        stone = next(p for p in doc.placements if p.role == "kiva_wall")
        doc.placements.remove(stone)
        failed = {c.code for c in rebuild(doc).quality.checks if c.status == "fail"}
        self.assertLessEqual({"chacoan.open_kivas", "chacoan.kiva_courses", "chacoan.terraced_range"}, failed)
