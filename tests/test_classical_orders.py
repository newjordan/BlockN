"""Check the actual carved solids, order identity, and generator contract."""

import json
import unittest

import numpy as np
import trimesh

from core.architectural_geometry import mesh_parameters
from core.classical_orders import column_meshes, fluted_shaft, fluted_shaft_courses, spiral_volute
from core.generators.classical import ORDER_SPECS, _build_column
from core.primitives import create_primitive_mesh
from core.scene import Scene


def _interior_surface_witness(surface, solid):
    """Prove overlap using solid-angle winding and exact triangle distance.

    A sampled surface point strictly inside the other closed mesh certifies
    intersecting solids. This deliberately does not use placement envelopes.
    """
    points = np.concatenate([surface.vertices, surface.triangles_center])
    triangles = solid.triangles
    for start in range(0, len(points), 48):
        batch = points[start:start + 48]
        vectors = triangles[None] - batch[:, None, None]
        lengths = np.linalg.norm(vectors, axis=3)
        a, b, c = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
        numerator = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        denominator = (np.prod(lengths, axis=2)
            + np.einsum("pfi,pfi->pf", a, b) * lengths[:, :, 2]
            + np.einsum("pfi,pfi->pf", b, c) * lengths[:, :, 0]
            + np.einsum("pfi,pfi->pf", c, a) * lengths[:, :, 1])
        winding = np.abs(np.sum(2 * np.arctan2(numerator, denominator), axis=1)) / (4 * np.pi)
        inside = batch[winding > 0.999]
        if len(inside):
            _, distances, _ = trimesh.proximity.closest_point_naive(solid, inside)
            indices = np.flatnonzero(distances > 1e-8)
            if len(indices):
                index = indices[np.argmax(distances[indices])]
                return inside[index], distances[index]
    return None


class TestClassicalOrders(unittest.TestCase):
    def assert_solid(self, mesh):
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)
        self.assertTrue(np.all(np.isfinite(mesh.vertices)))
        self.assertTrue(np.all(mesh.area_faces > 0))

    def parts(self, name):
        spec = ORDER_SPECS[name]
        return column_meshes(name, 0.5, spec.shaft_height_ratio,
                             spec.base_height_ratio, spec.base_width_ratio,
                             spec.capital_height_ratio, spec.capital_width_ratio)

    def test_every_order_is_solid_and_spans_its_requested_height(self):
        for name, spec in ORDER_SPECS.items():
            with self.subTest(order=name):
                parts = self.parts(name)
                for part in parts:
                    self.assert_solid(part.mesh)
                bounds = np.array([part.mesh.bounds for part in parts])
                self.assertAlmostEqual(bounds[:, 0, 1].min(), 0)
                self.assertAlmostEqual(bounds[:, 1, 1].max(),
                    spec.base_height_ratio + spec.shaft_height_ratio + spec.capital_height_ratio)
                half_width = max(spec.base_width_ratio, spec.capital_width_ratio, 1) / 2
                self.assertGreaterEqual(bounds[:, 0, [0, 2]].min(), -half_width - 1e-9)
                self.assertLessEqual(bounds[:, 1, [0, 2]].max(), half_width + 1e-9)

    def test_flutes_are_twenty_and_twenty_four_concave_hollows(self):
        for count, fillet in [(20, 0), (24, 0.18)]:
            with self.subTest(flutes=count):
                mesh = fluted_shaft(1, 6, flutes=count, fillet=fillet)
                self.assert_solid(mesh)
                section = mesh.vertices[np.isclose(mesh.vertices[:, 1], 0)][:, [0, 2]]
                section = section[np.linalg.norm(section, axis=1) > 0.1]
                order = np.argsort(np.arctan2(section[:, 1], section[:, 0]))
                section = section[order]
                radius = np.linalg.norm(section, axis=1)
                minima = (radius < np.roll(radius, 1) - 1e-9) & (radius < np.roll(radius, -1) - 1e-9)
                self.assertEqual(np.count_nonzero(minima), count)
                edges = np.roll(section, -1, axis=0) - section
                turns = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
                self.assertGreater(np.count_nonzero(turns < -1e-8), count)
                self.assertLess(radius.min(), 0.96)
                self.assertAlmostEqual(radius.max(), 1)

    def test_fluted_shaft_courses_are_cubit_drums(self):
        drums = fluted_shaft_courses(0.5, 5.4, flutes=20, top_ratio=0.83)
        self.assertGreater(len(drums), 4)
        bottoms = [float(drum.bounds[0, 1]) for drum in drums]
        tops = [float(drum.bounds[1, 1]) for drum in drums]
        self.assertAlmostEqual(min(bottoms), 0.0, places=6)
        self.assertAlmostEqual(max(tops), 5.4, places=6)
        for drum in drums:
            self.assert_solid(drum)
            self.assertLessEqual(float(drum.bounds[1, 1] - drum.bounds[0, 1]), 0.80 + 1e-6)

    def test_taper_has_restrained_entasis_and_tuscan_is_smooth(self):
        mesh = fluted_shaft(1, 8, flutes=0, top_ratio=0.86)
        self.assert_solid(mesh)
        levels = np.unique(mesh.vertices[:, 1])
        for y in levels:
            ring = mesh.vertices[np.isclose(mesh.vertices[:, 1], y)]
            radius = np.linalg.norm(ring[:, [0, 2]], axis=1)
            radius = radius[radius > 0.1]
            self.assertLess(np.ptp(radius), 1e-10)
            if 0 < y < 8:
                self.assertGreater(radius.max(), 1 - 0.14 * y / 8)
            self.assertLessEqual(radius.max(), 1 + 1e-10)
        top = mesh.vertices[np.isclose(mesh.vertices[:, 1], 8)]
        self.assertAlmostEqual(np.linalg.norm(top[:, [0, 2]], axis=1).max(), 0.86)

    def test_doric_absorbs_legacy_base_allowance_without_growing(self):
        parts = column_meshes("Doric", 0.5, 5, 0.4, 1.1, 0.6, 1.4)
        self.assertFalse(any(part.element.startswith("base") for part in parts))
        shaft = next(part.mesh for part in parts if part.element == "shaft")
        np.testing.assert_allclose(shaft.bounds[:, 1], [0, 5.4])
        self.assertAlmostEqual(max(part.mesh.bounds[1, 1] for part in parts), 6)
        self.assertEqual(ORDER_SPECS["Doric"].base_height_ratio, 0)

    def test_ionic_has_molded_base_and_opposed_spiral_faces(self):
        parts = self.parts("Ionic")
        elements = [part.element for part in parts]
        self.assertTrue({"base", "base_lower", "base_scotia", "base_upper", "capital_cushion"}.issubset(elements))
        scrolls = [part.mesh for part in parts if part.element == "capital_volute"]
        self.assertEqual(len(scrolls), 4)
        centers = np.array([mesh.bounds.mean(axis=0) for mesh in scrolls])
        self.assertEqual(len(set(np.sign(centers[:, 0]))), 2)
        self.assertEqual(len(set(np.sign(centers[:, 2]))), 2)
        for scroll in scrolls:
            self.assert_solid(scroll)
            self.assertLess(scroll.volume / np.prod(scroll.extents), 0.40)
        abacus = next(part.mesh for part in parts if part.element == "capital_abacus")
        self.assertLess(abacus.extents[1], ORDER_SPECS["Ionic"].capital_height_ratio * 0.15)

    def test_export_payloads_rebuild_actual_shapes_and_are_detached(self):
        for name in ORDER_SPECS:
            for part in self.parts(name):
                with self.subTest(order=name, element=part.element):
                    position, dimensions, parameters = mesh_parameters(part.mesh)
                    parameters = json.loads(json.dumps(parameters, allow_nan=False))
                    mesh = create_primitive_mesh("MESH", np.asarray(position), np.asarray(dimensions), parameters)
                    self.assert_solid(mesh)
                    np.testing.assert_allclose(mesh.vertices, part.mesh.vertices, atol=1e-12)
                    np.testing.assert_array_equal(mesh.faces, part.mesh.faces)
        first = self.parts("Doric")
        original = first[0].mesh.vertices.copy()
        first[0].mesh.vertices[:] += 10
        np.testing.assert_array_equal(self.parts("Doric")[0].mesh.vertices, original)

    def test_builder_preserves_height_metadata_and_world_placement(self):
        for name in ORDER_SPECS:
            with self.subTest(order=name):
                scene = Scene(use_collision=False)
                metadata = {"column_index": 7, "run_face": "front"}
                top = _build_column(scene, ORDER_SPECS[name], 4, -3, 2,
                                    0.5, 5, 0.35, 1.35, 0.65, 1.65, metadata)
                self.assertAlmostEqual(top, 8)
                self.assertEqual(metadata, {"column_index": 7, "run_face": "front"})
                shafts = [block for block in scene.blocks if block.parameters.get("element") == "shaft"]
                self.assertGreater(len(shafts), 1)
                self.assertTrue(all(block.block_type == "MESH" for block in shafts))
                self.assertTrue(all(float(block.dimensions[1]) <= 0.80 + 1e-6 for block in shafts))
                self.assertTrue(all(block.parameters["radius"] == 0.5 for block in shafts))
                self.assertTrue(all(block.parameters["column_index"] == 7 for block in shafts))
                for shaft in shafts:
                    np.testing.assert_allclose(shaft.position[[0, 2]], [4, -3])
                self.assertAlmostEqual(max(block.mesh.bounds[1, 1] for block in scene.blocks), top)
                self.assertAlmostEqual(min(block.mesh.bounds[0, 1] for block in scene.blocks), 2)

    def test_invalid_dimensions_and_profiles_are_rejected(self):
        for value in [0, -1, float("inf"), float("nan"), True, "1"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                column_meshes("Doric", value, 5, 0, 1.1, 0.5, 1.4)
        with self.assertRaises(ValueError):
            column_meshes("Unknown", 0.5, 5, 0, 1.1, 0.5, 1.4)
        with self.assertRaises(ValueError):
            fluted_shaft(0.5, 5, flutes=21)
        with self.assertRaises(ValueError):
            spiral_volute(0.2, 0.1, hand=0)

    def test_ionic_side_colonnade_turns_its_scroll_faces_outward(self):
        scenes = []
        for axis in ("X", "Z"):
            scene = Scene(use_collision=False)
            _build_column(scene, ORDER_SPECS["Ionic"], 0, 0, 0, 0.5, 5,
                          0.5, 1.35, 0.52, 1.58, {"run_axis": axis})
            scenes.append(scene)
        front = [block for block in scenes[0].blocks if block.parameters["element"] == "capital_volute"]
        side = [block for block in scenes[1].blocks if block.parameters["element"] == "capital_volute"]
        for a, b in zip(front, side):
            np.testing.assert_allclose(b.mesh.vertices, a.mesh.vertices[:, [2, 1, 0]] * [1, 1, -1])
            self.assert_solid(b.mesh)

    def test_foliate_bell_has_a_real_bearing_surface_inside_the_abacus(self):
        for name in ("Corinthian", "Composite"):
            with self.subTest(order=name):
                parts = self.parts(name)
                bell = next(part.mesh for part in parts if part.element == "capital_bell")
                abacus = next(part.mesh for part in parts if part.element == "capital_abacus")
                # The complete top cap of the bell enters the slab. Merely
                # overlapping a decorated capital's broad envelope is weaker.
                bearing = bell.vertices[np.isclose(bell.vertices[:, 1], bell.bounds[1, 1])]
                self.assertGreater(len(bearing), 3)
                self.assertTrue(np.all(bearing > abacus.bounds[0] + 1e-9))
                self.assertTrue(np.all(bearing < abacus.bounds[1] - 1e-9))

    def test_ionic_neck_bolster_scrolls_and_abacus_have_actual_mesh_contact(self):
        parts = self.parts("Ionic")
        solids = {part.element: part.mesh for part in parts}
        cushion, neck, abacus = [solids[element] for element in
                                ("capital_cushion", "capital_neck", "capital_abacus")]
        for name, bearing in [("neck", neck), ("abacus", abacus)]:
            with self.subTest(bearing=name):
                self.assertIsNotNone(_interior_surface_witness(cushion, bearing))
        scrolls = [part.mesh for part in parts if part.element == "capital_volute"]
        self.assertEqual(len(scrolls), 4)
        for index, scroll in enumerate(scrolls):
            with self.subTest(scroll=index):
                self.assertIsNotNone(_interior_surface_witness(scroll, cushion))
                # Both faces' scroll heads meet actual triangles of the slab;
                # the cushion now also supplies a positive-volume bearing.
                _, distances, _ = trimesh.proximity.closest_point_naive(abacus, scroll.vertices)
                self.assertLess(distances.min(), 1e-9)


if __name__ == "__main__":
    unittest.main()
