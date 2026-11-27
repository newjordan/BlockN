"""Construction regressions for radial masonry, open rooms and roof bearings."""
import unittest

import numpy as np

from core.architecture import BuildDocument, BuildRequest, build, rebuild


def vertical_hits(mesh, x, z):
    """Intersect a vertical line with real triangles without optional rtree."""
    triangles = np.asarray(mesh.triangles)
    a = triangles[:, 0][:, [0, 2]]
    u = triangles[:, 1][:, [0, 2]] - a
    v = triangles[:, 2][:, [0, 2]] - a
    point = np.array([x, z]) - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(det) > 1e-10
    s = np.zeros(len(det)); t = np.zeros(len(det))
    s[valid] = (point[:, 0] * v[:, 1] - point[:, 1] * v[:, 0])[valid] / det[valid]
    t[valid] = (u[:, 0] * point[:, 1] - u[:, 1] * point[:, 0])[valid] / det[valid]
    valid &= (s >= -1e-8) & (t >= -1e-8) & (s + t <= 1 + 1e-8)
    heights = (triangles[:, 0, 1] + s * (triangles[:, 1, 1] - triangles[:, 0, 1])
               + t * (triangles[:, 2, 1] - triangles[:, 0, 1]))[valid]
    return np.unique(np.round(heights, 8))


class MasonryFormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {form: build(BuildRequest(product="classical-" + form, detail="standard", seed=73))
                       for form in ("theatre", "amphitheatre", "basilica", "rotunda", "baths")}

    def parts(self, form, role):
        return [b for b in self.results[form].scene.blocks if b.parameters["component_role"] == role]

    def test_changed_members_are_positive_watertight_solids_and_replay(self):
        for form, result in self.results.items():
            with self.subTest(form=form):
                self.assertEqual(result.quality.status, "pass",
                                 [c.to_dict() for c in result.quality.checks if c.status == "fail"])
                for block in result.scene.blocks:
                    if block.block_type == "MESH":
                        self.assertTrue(block.mesh.is_watertight, block.parameters["primitive_id"])
                        self.assertTrue(block.mesh.is_winding_consistent)
                        self.assertGreater(block.mesh.volume, 0)
                replay = rebuild(BuildDocument.from_dict(result.document.to_dict()))
                self.assertEqual(result.document.fingerprint(), replay.document.fingerprint())
                for original, restored in zip(result.scene.blocks, replay.scene.blocks):
                    np.testing.assert_allclose(original.mesh.vertices, restored.mesh.vertices)

    def test_seating_uses_curved_treads_and_radial_access_stairs(self):
        for form, width, depth in (("theatre", 26, 20), ("amphitheatre", 28, 22)):
            seats = self.parts(form, "seating_tier")
            self.assertTrue(all(b.block_type == "MESH" for b in seats))
            for block in seats:
                points = block.mesh.vertices[:, [0, 2]] / [width, depth]
                radii = np.linalg.norm(points, axis=1)
                self.assertEqual(len(np.unique(np.round(radii, 8))), 2)
                self.assertGreater(len(np.unique(np.round(points, 8), axis=0)), 8)
            # An access aisle is an actual gap in the cavea, traversed by steps.
            angle = np.pi / 6 if form == "theatre" else 0
            x, z = width * .30 * np.cos(angle), depth * .30 * np.sin(angle)
            self.assertFalse(any(len(vertical_hits(b.mesh, x, z)) for b in seats))
            self.assertTrue(any(len(vertical_hits(b.mesh, x, z))
                                for b in self.parts(form, "aisle_step")))
            self.assertFalse(any(len(vertical_hits(b.mesh, 0, 0)) for b in seats))

    def test_cavea_and_stage_are_human_scale_ashlar(self):
        from core.architecture.masonry import placement_is_oversized

        for form in ("theatre", "amphitheatre"):
            with self.subTest(form=form):
                seats = [p for p in self.results[form].document.placements if p.role == "seating_tier"]
                self.assertGreater(len(seats), 20)
                self.assertGreater(len({p.parameters.get("course") for p in seats}), 1)
                self.assertTrue(all(p.parameters.get("unit") == "radial_ashlar" for p in seats))
                oversized = [p.id for p in seats if placement_is_oversized(p)]
                self.assertEqual(oversized, [])
                steps = [p for p in self.results[form].document.placements if p.role == "aisle_step"]
                self.assertGreater(len(steps), 8)
                self.assertTrue(all(p.parameters.get("unit") == "radial_ashlar" for p in steps))
                self.assertEqual([p.id for p in steps if placement_is_oversized(p)], [])
                self.assertEqual(self.results[form].quality.status, "pass",
                                 [c.to_dict() for c in self.results[form].quality.checks if c.status == "fail"])
        stage = [p for p in self.results["theatre"].document.placements if p.role == "stage"]
        self.assertGreater(len(stage), 1)
        self.assertEqual([p.id for p in stage if placement_is_oversized(p)], [])
        arena = [p for p in self.results["amphitheatre"].document.placements if p.role == "arena"]
        self.assertGreater(len(arena), 8)
        self.assertTrue(all(p.parameters.get("unit") == "paving" for p in arena))
        self.assertEqual([p.id for p in arena if placement_is_oversized(p)], [])

    def test_basilica_apse_has_shared_mitred_joints_and_open_room(self):
        walls = self.parts("basilica", "apse")
        self.assertGreater(len(walls), 11)
        courses = {}
        for block in walls:
            courses.setdefault(block.parameters["course"], []).append(block)
        for blocks in courses.values():
            blocks.sort(key=lambda item: item.parameters.get("facet", 0))
            self.assertGreaterEqual(len(blocks), 2)
            for left, right in zip(blocks, blocks[1:]):
                distance = np.linalg.norm(left.mesh.vertices[:, None] - right.mesh.vertices[None, :], axis=2)
                self.assertGreaterEqual(int(np.sum(np.min(distance, axis=1) < 1e-6)), 4)
        self.assertFalse(any(len(vertical_hits(b.mesh, 0, 26 * .40)) for b in walls))
        self.assertTrue(self.parts("basilica", "apse_roof"))

    def test_rotunda_drum_and_dome_preserve_real_voids(self):
        drums = [b.mesh for b in self.parts("rotunda", "drum")]
        domes = [b.mesh for b in self.parts("rotunda", "dome")]
        self.assertGreater(len(drums), 4)
        self.assertGreater(len(domes), 8)
        self.assertTrue(all(len(vertical_hits(drum, 0, 0)) == 0 for drum in drums))
        self.assertTrue(all(len(vertical_hits(cell, 0, 0)) == 0 for cell in domes))
        self.assertTrue(any(len(vertical_hits(drum, 16 * .36, 0)) == 2 for drum in drums))
        heights = np.unique(np.concatenate(
            [vertical_hits(cell, 16 * .24, 0) for cell in domes] or [np.array([])]))
        self.assertEqual(len(heights), 2)
        self.assertLess(np.ptp(heights), 10 * .07)

    def test_leaf_capitals_carry_the_rotunda_bearing_ring(self):
        for order in ("corinthian", "composite"):
            with self.subTest(order=order):
                result = build(BuildRequest(product="classical-rotunda", variant=order,
                                             detail="presentation", seed=73))
                self.assertEqual(result.quality.status, "pass",
                                 [c.to_dict() for c in result.quality.checks if c.status == "fail"])
                drum = next(b.mesh for b in result.scene.blocks if b.parameters["component_role"] == "drum")
                heads = [b.mesh for b in result.scene.blocks
                         if b.parameters["component_role"] == "column_capital_abacus"]
                self.assertGreater(len(heads), 5)
                for head in heads:
                    self.assertAlmostEqual(head.bounds[1, 1], drum.bounds[0, 1])

    def test_basilica_pitched_roof_is_a_shell_above_an_open_nave(self):
        roofs = self.parts("basilica", "nave_roof")
        self.assertGreater(len(roofs), 4)
        def stacked(x, z):
            hits = [vertical_hits(b.mesh, x, z) for b in roofs]
            hits = [h for h in hits if len(h)]
            self.assertTrue(hits)
            return np.unique(np.concatenate(hits))
        ridge = stacked(.05, 0)
        eave = stacked(18 * .265, 0)
        self.assertEqual(len(ridge), 2)
        self.assertEqual(len(eave), 2)
        self.assertGreater(ridge[0] - eave[0], 1.)
        self.assertLess(float(np.ptp(ridge)), .4)
        self.assertGreater(ridge[0], 9.)

    def test_baths_vault_has_intrados_and_caldarium_bears_on_ring(self):
        roofs = [b.mesh for b in self.parts("baths", "hall_roof")]
        self.assertGreater(len(roofs), 8)
        hits = np.unique(np.concatenate(
            [vertical_hits(cell, .25, -22 * .15 + .10) for cell in roofs] or [np.array([])]))
        self.assertEqual(len(hits), 2)
        self.assertAlmostEqual(float(np.ptp(hits)), .35, places=2)
        domes = [b.mesh for b in self.parts("baths", "dome")]
        drums = [b.mesh for b in self.parts("baths", "drum")]
        self.assertGreater(len(drums), 4)
        self.assertGreater(len(domes), 8)
        self.assertTrue(all(len(vertical_hits(cell, 0, 22 * .285)) == 0 for cell in domes))
        self.assertTrue(all(len(vertical_hits(ring, 0, 22 * .285)) == 0 for ring in drums))
        probe_x = min(26 * .225, 22 * .195) * 0.94
        self.assertTrue(any(len(vertical_hits(ring, probe_x, 22 * .285)) == 2 for ring in drums))
        self.assertAlmostEqual(min(cell.bounds[0, 1] for cell in domes),
                               max(ring.bounds[1, 1] for ring in drums))
        self.assertGreaterEqual(len(self.parts("baths", "hall_pier")), 4)

    def test_baths_vault_ribs_are_cubit_stretchers(self):
        from core.architecture.masonry import placement_is_oversized

        ribs = [p for p in self.results["baths"].document.placements if p.role == "vault_rib"]
        self.assertGreater(len(ribs), 8)
        self.assertTrue(all(p.parameters.get("unit") == "profile_course" for p in ribs))
        self.assertEqual([p.id for p in ribs if placement_is_oversized(p)], [])

    def test_baths_pool_bed_is_coursed_ashlar(self):
        from core.architecture.masonry import placement_is_oversized

        beds = [p for p in self.results["baths"].document.placements if p.role == "pool_bed"]
        self.assertGreater(len(beds), 1)
        self.assertTrue(all(p.block_type == "CUBE" for p in beds))
        self.assertTrue(all(p.parameters.get("unit") == "ashlar" for p in beds))
        self.assertEqual([p.id for p in beds if placement_is_oversized(p)], [])

    def test_masonry_arcades_are_coursed_not_arch_primitives(self):
        from core.architecture.masonry import placement_is_oversized

        for form in ("baths", "theatre", "amphitheatre"):
            with self.subTest(form=form):
                result = self.results[form]
                self.assertFalse(any(p.block_type == "ARCH" for p in result.document.placements))
                arches = [p for p in result.document.placements if p.role == "arch"]
                self.assertEqual([p.id for p in arches if placement_is_oversized(p)], [])
        for product in ("classical-aqueduct", "classical-triumphal-arch"):
            with self.subTest(product=product):
                result = build(BuildRequest(product=product, detail="standard", seed=73))
                self.assertFalse(any(p.block_type == "ARCH" for p in result.document.placements))
                arches = [p for p in result.document.placements if p.role == "arch"]
                self.assertGreater(len(arches), 2)
                self.assertEqual([p.id for p in arches if placement_is_oversized(p)], [])
        arena = [p for p in self.results["amphitheatre"].document.placements if p.role == "arena"]
        self.assertGreater(len(arena), 4)
        self.assertTrue(all(p.block_type == "CUBE" for p in arena))
        self.assertEqual([p.id for p in arena if placement_is_oversized(p)], [])


if __name__ == "__main__":
    unittest.main()
