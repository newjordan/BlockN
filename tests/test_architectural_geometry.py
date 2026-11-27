"""Geometry and persistence checks for the cathedral sculpting primitives."""

import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from core.architectural_geometry import extrude_profile, mesh_parameters, tube_path
from core.architecture.compiler import compile_document
from core.architecture.io import export_bundle
from core.architecture.service import load_document
from core.architecture.schema import ArchitectInfo, BuildDocument, BuildRequest, Placement
from core.primitives import MESH, create_primitive_mesh


def _pointed_arch_boundary():
    """A two-center pointed arch with a real open-bottom concave boundary."""
    def arc(half_width, rise):
        center = (rise * rise - half_width * half_width) / (2 * half_width)
        radius = half_width + center
        left_x = np.linspace(-half_width, 0.0, 21)
        left_y = 2.0 + np.sqrt(np.maximum(0.0, radius * radius - (left_x - center) ** 2))
        return np.concatenate([
            np.column_stack([left_x, left_y]),
            np.column_stack([-left_x[-2::-1], left_y[-2::-1]]),
        ])

    return np.concatenate([
        [[-2, 0]], arc(2, 3), [[2, 0], [1.6, 0]],
        arc(1.6, 2.55)[::-1], [[-1.6, 0]],
    ])


def _cap_covers(mesh, point):
    triangles = mesh.triangles
    triangles = triangles[np.all(np.isclose(triangles[:, :, 2], mesh.bounds[1, 2]), axis=1), :, :2]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    def cross2(left, right):
        return left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0]

    point = np.asarray(point)
    return bool(np.any(
        (cross2(b - a, point - a) >= -1e-10)
        & (cross2(c - b, point - b) >= -1e-10)
        & (cross2(a - c, point - c) >= -1e-10)
    ))


class TestProfileExtrusion(unittest.TestCase):
    def assert_solid(self, mesh):
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)
        self.assertTrue(np.all(np.isfinite(mesh.vertices)))
        self.assertTrue(np.all(mesh.area_faces > 0))

    def test_pointed_arch_keeps_its_opening_and_exact_envelope(self):
        profile = _pointed_arch_boundary()
        original = profile.copy()
        mesh = extrude_profile(profile, 0.6)
        self.assert_solid(mesh)
        np.testing.assert_allclose(mesh.bounds, [[-2, 0, -0.3], [2, 5, 0.3]])
        self.assertFalse(_cap_covers(mesh, [0, 1]))
        self.assertFalse(_cap_covers(mesh, [0, 4.3]))
        self.assertTrue(_cap_covers(mesh, [-1.8, 1]))
        self.assertTrue(_cap_covers(mesh, [0, 4.8]))
        np.testing.assert_array_equal(profile, original)

    def test_reversed_winding_produces_the_same_positive_solid(self):
        profile = _pointed_arch_boundary()
        forward = extrude_profile(profile, 0.6)
        backward = extrude_profile(profile[::-1], 0.6)
        self.assert_solid(backward)
        self.assertAlmostEqual(backward.volume, forward.volume)
        np.testing.assert_allclose(backward.bounds, forward.bounds)

    def test_profiled_cornice_and_collinear_samples(self):
        profile = [[0, 0], [1, 0], [2, 0], [2, 0.2], [0.8, 0.2], [0.8, 0.6], [0, 0.6]]
        mesh = extrude_profile(profile, 1.5)
        self.assert_solid(mesh)
        self.assertAlmostEqual(mesh.volume, 1.08)
        self.assertEqual(mesh.euler_number, 2)

    def test_scaled_translated_profiles_keep_their_shape(self):
        profile = np.array([[0, 0], [2, 0], [2, 0.3], [0.8, 0.3], [0.8, 2], [0, 2]])
        for scale, offset in [(1e-6, 0.0), (1e6, 3e7)]:
            with self.subTest(scale=scale):
                mesh = extrude_profile(profile * scale + offset, scale)
                self.assert_solid(mesh)
                np.testing.assert_allclose(mesh.bounds[:, :2], [[offset, offset], [offset + 2 * scale] * 2])
                self.assertAlmostEqual(mesh.volume / scale ** 3, 1.96, places=5)

    def test_invalid_profiles_fail_before_mesh_creation(self):
        profiles = [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 0], [0, 1], [0, 0]],
            [[0, 0], [1, 1], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [2, 0]],
            [[0, 0], [1, 0], [0.5, 0], [1, 1], [0, 1]],
            [[0, 0], [1, 0], [float("nan"), 1]],
            [[0, 0], [1, 0], ["0", "1"]],
            [[0, 0], [2, 0], [2, 2], [1, 0], [0, 2]],
        ]
        for profile in profiles:
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                extrude_profile(profile, 0.5)
        for depth in [0, -1, float("nan"), float("inf"), True, "1"]:
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                extrude_profile([[0, 0], [1, 0], [0, 1]], depth)


class TestTubePath(unittest.TestCase):
    def assert_solid(self, mesh):
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)
        self.assertTrue(np.all(mesh.area_faces > 0))

    def test_straight_ribs_work_along_every_axis(self):
        for axis in range(3):
            with self.subTest(axis=axis):
                start = np.array([3.0, 7.0, -4.0])
                end = start.copy()
                end[axis] += 5
                mesh = tube_path([start, end], 0.4, sections=8)
                self.assert_solid(mesh)
                minimum, maximum = start - 0.4, end + 0.4
                minimum[axis], maximum[axis] = start[axis], end[axis]
                np.testing.assert_allclose(mesh.bounds, [minimum, maximum])
                self.assertAlmostEqual(mesh.volume, 4 * np.sin(np.pi / 4) * 0.4 ** 2 * 5)

    def test_curved_vault_rib_retains_centerline_radius_without_frame_flips(self):
        t = np.linspace(0, np.pi, 65)
        path = np.column_stack([4 * np.cos(t), 6 * np.sin(t), 0.3 * np.sin(2 * t)])
        original = path.copy()
        mesh = tube_path(path, 0.08, sections=12)
        self.assert_solid(mesh)
        rings = np.asarray(mesh.vertices[:-2]).reshape(len(path), 12, 3)
        np.testing.assert_allclose(rings.mean(axis=1), path, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(rings - path[:, None, :], axis=2), 0.08)
        first_rays = (rings[:, 0] - path) / 0.08
        self.assertGreater(np.min(np.sum(first_rays[:-1] * first_rays[1:], axis=1)), 0.97)
        np.testing.assert_array_equal(path, original)

    def test_closed_tracery_ring_has_no_caps_or_seam_crack(self):
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        path = np.column_stack([3 * np.cos(t), 3 * np.sin(t), np.zeros_like(t)])
        mesh = tube_path(path, 0.1, sections=16, closed=True)
        self.assert_solid(mesh)
        self.assertEqual(mesh.euler_number, 0)
        self.assertEqual(len(mesh.vertices), 64 * 16)
        self.assertEqual(len(mesh.faces), 2 * 64 * 16)
        self.assertAlmostEqual(mesh.volume / (2 * np.pi * 3 * np.pi * 0.1 ** 2), 1.0, delta=0.03)
        np.testing.assert_allclose(mesh.bounds, [[-3.1, -3.1, -0.1], [3.1, 3.1, 0.1]])

    def test_nonplanar_closed_rib_distributes_frame_rotation(self):
        t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        path = np.column_stack([3 * np.cos(t), 2 * np.sin(t), 0.7 * np.sin(2 * t)])
        mesh = tube_path(path, 0.045, sections=8, closed=True)
        self.assert_solid(mesh)
        rings = np.asarray(mesh.vertices).reshape(len(path), 8, 3)
        first_rays = (rings[:, 0] - path) / 0.045
        adjacent_alignment = np.sum(first_rays * np.roll(first_rays, -1, axis=0), axis=1)
        self.assertGreater(adjacent_alignment.min(), 0.97)
        np.testing.assert_allclose(np.linalg.norm(rings - path[:, None, :], axis=2), 0.045)

    def test_invalid_paths_and_options_fail(self):
        cases = [
            ([], {}),
            ([[0, 0, 0]], {}),
            ([[0, 0, 0], [0, 0, 0]], {}),
            ([[0, 0, 0], [0, 1, 0], [0, 0, 0]], {}),
            ([[0, 0, 0], [0, 1, 0]], {"closed": True}),
            ([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]], {"closed": True}),
            ([[0, 0, 0], [0, float("inf"), 0]], {}),
            ([[0, 0, 0], [0, 1, 0]], {"sections": 2}),
            ([[0, 0, 0], [0, 1, 0]], {"sections": 8.0}),
            ([[0, 0, 0], [0, 1, 0]], {"closed": "yes"}),
        ]
        for path, options in cases:
            with self.subTest(path=path, options=options), self.assertRaises(ValueError):
                tube_path(path, 0.1, **options)
        for radius in [0, -1, float("nan"), float("inf"), True, "1"]:
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                tube_path([[0, 0, 0], [0, 1, 0]], radius)


class TestMeshPrimitive(unittest.TestCase):
    def setUp(self):
        self.mesh = extrude_profile(_pointed_arch_boundary(), 0.6)
        self.position, self.dimensions, self.parameters = mesh_parameters(self.mesh)

    def test_normalized_parameters_rebuild_exact_mesh_and_detach_all_inputs(self):
        rebuilt = create_primitive_mesh(MESH, self.position, self.dimensions, self.parameters)
        np.testing.assert_allclose(rebuilt.vertices, self.mesh.vertices, atol=1e-12)
        np.testing.assert_array_equal(rebuilt.faces, self.mesh.faces)
        self.assertEqual(json.loads(json.dumps(self.parameters)), self.parameters)
        self.assertTrue(rebuilt.is_watertight)
        self.assertTrue(rebuilt.is_winding_consistent)
        self.assertGreater(rebuilt.volume, 0)
        retained = rebuilt.vertices.copy()
        self.parameters["vertices"][0][0] = 999
        self.parameters["faces"][0][0] = 999
        self.mesh.vertices[0][0] = 999
        np.testing.assert_array_equal(rebuilt.vertices, retained)

    def test_payload_and_placement_arrays_are_not_mutated_by_compilation(self):
        parameters = {key: np.asarray(value) for key, value in self.parameters.items()}
        before = copy.deepcopy(parameters)
        position, dimensions = np.asarray(self.position), np.asarray(self.dimensions)
        position_before, dimensions_before = position.copy(), dimensions.copy()
        mesh = create_primitive_mesh(MESH, position, dimensions, parameters)
        mesh.vertices[0] = 0
        for key in parameters:
            np.testing.assert_array_equal(parameters[key], before[key])
        np.testing.assert_array_equal(position, position_before)
        np.testing.assert_array_equal(dimensions, dimensions_before)

    def test_scaling_translation_and_small_envelope_roundoff(self):
        parameters = copy.deepcopy(self.parameters)
        parameters["vertices"] = (np.asarray(parameters["vertices"]) * (1 - 1e-10)).tolist()
        mesh = create_primitive_mesh(MESH, [11, 20, -7], [4, 9, 2], parameters)
        np.testing.assert_allclose(mesh.bounds, [[9, 15.5, -8], [13, 24.5, -6]], atol=1e-12)

    def test_normalization_rejects_flat_and_malformed_source_meshes(self):
        flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False)
        for mesh in [trimesh.Trimesh(), flat]:
            with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                mesh_parameters(mesh)
        with self.assertRaises(TypeError):
            mesh_parameters([[0, 0, 0]])

    def test_invalid_payloads_are_rejected_instead_of_repaired_silently(self):
        cases = []
        for key in ("vertices", "faces"):
            parameters = copy.deepcopy(self.parameters)
            del parameters[key]
            cases.append(parameters)
        for invalid in [float("nan"), float("inf"), "bad", True, 0.9]:
            parameters = copy.deepcopy(self.parameters)
            parameters["vertices"][0][0] = invalid
            cases.append(parameters)
        for invalid in [-1, len(self.mesh.vertices), 0.5, True]:
            parameters = copy.deepcopy(self.parameters)
            parameters["faces"][0][0] = invalid
            cases.append(parameters)
        parameters = copy.deepcopy(self.parameters)
        parameters["faces"][0][1] = parameters["faces"][0][0]
        cases.append(parameters)
        parameters = copy.deepcopy(self.parameters)
        parameters["faces"].append(parameters["faces"][0][:])
        cases.append(parameters)
        parameters = copy.deepcopy(self.parameters)
        parameters["vertices"].append([0, 0, 0])
        cases.append(parameters)
        parameters = copy.deepcopy(self.parameters)
        parameters["vertices"] = (np.asarray(parameters["vertices"]) * 0.9).tolist()
        cases.append(parameters)
        for index, parameters in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                create_primitive_mesh(MESH, self.position, self.dimensions, parameters)

    def test_component_size_limits_reject_unbounded_payloads(self):
        parameters = dict(self.parameters, vertices=np.zeros((65_537, 3)))
        with self.assertRaisesRegex(ValueError, "vertices"):
            create_primitive_mesh(MESH, self.position, self.dimensions, parameters)
        parameters = dict(self.parameters, faces=np.zeros((131_073, 3), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "faces"):
            create_primitive_mesh(MESH, self.position, self.dimensions, parameters)
        with self.assertRaisesRegex(ValueError, "mesh vertices"):
            tube_path(np.zeros((513, 3)), 0.1, sections=128, closed=True)

    def test_explicit_octagonal_spire_roundtrips_as_an_oriented_solid(self):
        # Eight-sided cone from trimesh supplies an independent mesh topology.
        spire = trimesh.creation.cone(radius=1.4, height=7, sections=8)
        spire.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        spire.apply_translation([6, 20, -3])
        position, dimensions, parameters = mesh_parameters(spire)
        rebuilt = create_primitive_mesh(MESH, position, dimensions, parameters)
        self.assertTrue(rebuilt.is_watertight)
        self.assertTrue(rebuilt.is_winding_consistent)
        self.assertAlmostEqual(rebuilt.volume, spire.volume)
        np.testing.assert_allclose(rebuilt.bounds, spire.bounds)

    def test_architecture_bundle_and_glb_rebuild_preserve_real_opening(self):
        document = BuildDocument(
            architect=ArchitectInfo("sculpture-test", "1.0", "Sculpted arch"),
            request=BuildRequest(product="sculpted-arch", seed=731),
            placements=[Placement(
                id="arch-001", kind="arch", block_type=MESH,
                position=self.position, dimensions=self.dimensions,
                material=(20, 255, 80, 255), role="pointed_arch",
                parameters=self.parameters,
            )],
        )
        result = compile_document(document)
        self.assertEqual(result.quality.status, "pass")
        with tempfile.TemporaryDirectory() as directory:
            paths = export_bundle(result, Path(directory) / "pointed-arch")
            loaded_document = load_document(paths["architecture"])
            self.assertEqual(loaded_document.fingerprint(), document.fingerprint())
            rebuilt = compile_document(loaded_document)
            self.assertEqual(rebuilt.quality.status, "pass")
            mesh = rebuilt.scene.blocks[0].mesh
            self.assertTrue(mesh.is_watertight)
            self.assertFalse(_cap_covers(mesh, [0, 1]))
            exported = trimesh.load(io.BytesIO(paths["glb"].read_bytes()), file_type="glb", force="scene", process=False)
            self.assertEqual(set(exported.geometry), {"arch-001"})
            exported_mesh = exported.geometry["arch-001"]
            self.assertTrue(exported_mesh.is_watertight)
            np.testing.assert_allclose(exported_mesh.bounds, self.mesh.bounds, atol=1e-6)
            self.assertFalse(_cap_covers(exported_mesh, [0, 1]))


if __name__ == "__main__":
    unittest.main()
