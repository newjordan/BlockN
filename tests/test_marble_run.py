"""Closed aqueduct/spline canal and a wrapping marble path."""
import unittest

import numpy as np

from core.architecture import BuildRequest, build, list_products
from core.architecture.marble_run import (
    PRODUCT,
    catmull_rom,
    closed_spline,
    point_on_loop,
    resample_closed,
)


class MarbleRunTests(unittest.TestCase):
    def test_closed_spline_wraps_and_has_positive_length(self):
        controls = [
            (2, 1, 0), (0, 1.2, 2), (-2, 1, 0), (0, 0.8, -2),
        ]
        samples, length = resample_closed(closed_spline(controls, 6), 0.5)
        self.assertGreater(length, 8)
        self.assertGreaterEqual(len(samples), 24)
        start, _ = point_on_loop(samples, length, 0)
        looped, _ = point_on_loop(samples, length, length)
        np.testing.assert_allclose(start, looped, atol=1e-6)
        later, _ = point_on_loop(samples, length, length * 3.25)
        mid, _ = point_on_loop(samples, length, length * 0.25)
        np.testing.assert_allclose(later, mid, atol=1e-6)
        self.assertTrue(np.isfinite(catmull_rom(*[np.array(p) for p in controls], 0.5)).all())

    def test_product_builds_a_looping_canal_with_a_marble_on_the_path(self):
        result = build(BuildRequest(product=PRODUCT, seed=73, bays=6, levels=1, detail="concept"))
        self.assertEqual(result.quality.status, "pass",
                         [c.to_dict() for c in result.quality.checks if c.status == "fail"])
        roles = {p.role for p in result.document.placements}
        self.assertTrue({"foundation", "pier", "arch", "channel_bed", "channel_wall", "marble"} <= roles)
        record = result.document.metadata["marble_run"]
        self.assertTrue(record["closed"] and record["looping"] and record["infinite"])
        self.assertGreater(record["length"], 20)
        self.assertEqual(result.scene.generation_metadata["marble_run"]["samples"], record["samples"])
        marble = next(p for p in result.document.placements if p.role == "marble")
        path = np.asarray(record["path"])
        self.assertLess(np.min(np.linalg.norm(path - marble.position, axis=1)), 1.5)
        start, _ = point_on_loop(path, record["length"], 0)
        wrapped, _ = point_on_loop(path, record["length"], record["length"] * 4)
        np.testing.assert_allclose(start, wrapped, atol=1e-5)
        from core.architecture.masonry import MAX_COURSE, MAX_STRETCHER, placement_is_oversized

        beds = [p for p in result.document.placements if p.role == "channel_bed"]
        walls = [p for p in result.document.placements if p.role == "channel_wall"]
        self.assertGreater(len(beds), 8)
        self.assertEqual(
            [p.id for p in beds if max(p.dimensions[0], p.dimensions[2]) > MAX_STRETCHER + 1e-6],
            [],
        )
        self.assertEqual([p.id for p in beds if placement_is_oversized(p)], [])
        self.assertEqual([p.id for p in walls if placement_is_oversized(p)], [])
        footings = [p for p in result.document.placements if p.role == "foundation"]
        arches = [p for p in result.document.placements if p.role == "arch"]
        self.assertGreater(len(footings), 8)
        self.assertGreater(len(arches), 8)
        self.assertEqual([p.id for p in footings if placement_is_oversized(p)], [])
        self.assertEqual([p.id for p in arches if placement_is_oversized(p)], [])

    def test_presentation_guide_spline_is_segmented(self):
        from core.architecture.masonry import MAX_COURSE, MAX_STRETCHER
        result = build(BuildRequest(product=PRODUCT, seed=73, bays=6, levels=1, detail="presentation"))
        self.assertEqual(result.quality.status, "pass",
                         [c.to_dict() for c in result.quality.checks if c.status == "fail"])
        guides = [p for p in result.document.placements if p.role == "guide_spline"]
        self.assertGreater(len(guides), 8)
        leftover = [
            p.id for p in guides
            if p.dimensions[1] > MAX_COURSE + 1e-6
            or max(p.dimensions[0], p.dimensions[2]) > MAX_STRETCHER + 1e-6
        ]
        self.assertEqual(leftover, [])

    def test_catalog_lists_marble_run(self):
        names = [item["name"] for item in list_products()]
        self.assertIn(PRODUCT, names)
        self.assertEqual(len(names), len(set(names)))


if __name__ == "__main__":
    unittest.main()
