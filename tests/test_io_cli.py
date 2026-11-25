import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

from core.io import export_meshes_to_glb, load_scene_from_json, save_scene_to_json
from core.scene import Scene

ROOT = Path(__file__).resolve().parents[1]


class TestSceneJson(unittest.TestCase):
    def test_numpy_parameters_round_trip(self):
        scene = Scene(use_collision=False)
        scene.support_check_mode = "OFF"
        added = scene.add_block(
            position=np.array([0.0, 0.5, 0.0]),
            parameters={"radius": np.float64(0.4), "count": np.int64(3), "flag": True},
        )
        self.assertIsNotNone(added)

        path = ROOT / "exports" / "roundtrip-test.json"
        try:
            save_scene_to_json(scene, str(path))
            payload = json.loads(path.read_text())
            self.assertEqual(payload["blocks"][0]["parameters"]["radius"], 0.4)
            self.assertEqual(payload["blocks"][0]["parameters"]["count"], 3)

            restored = Scene(use_collision=False)
            restored.support_check_mode = "OFF"
            load_scene_from_json(restored, str(path))
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(len(restored.blocks), 1)
        self.assertEqual(restored.blocks[0].parameters["count"], 3)
        self.assertAlmostEqual(restored.blocks[0].position[1], 0.5)


class TestGlbExport(unittest.TestCase):
    def test_exported_glb_loads(self):
        scene = Scene(use_collision=False)
        scene.support_check_mode = "OFF"
        scene.add_block(position=np.array([0.0, 0.5, 0.0]), dimensions=np.array([1.0, 1.0, 1.0]))
        path = ROOT / "exports" / "roundtrip-test.glb"
        try:
            export_meshes_to_glb(scene.get_all_meshes(), str(path))
            loaded = trimesh.load(path)
        finally:
            path.unlink(missing_ok=True)

        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            self.assertGreater(len(meshes), 0)
            vertex_count = sum(len(mesh.vertices) for mesh in meshes)
        else:
            vertex_count = len(loaded.vertices)
        self.assertGreater(vertex_count, 0)


class TestCli(unittest.TestCase):
    def test_generate_writes_glb_and_summary(self):
        output = ROOT / "exports" / "cli-test.glb"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "cli.py"),
                    "generate",
                    "--classical-order",
                    "Doric",
                    "--classical-preset",
                    "compact",
                    "--seed",
                    "7",
                    "--width",
                    "8",
                    "--height",
                    "6",
                    "--depth",
                    "8",
                    "--out",
                    str(output),
                    "--summary",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.exists())
            self.assertIn("Blocks:", completed.stdout)
            self.assertIn("invariants_ok=True", completed.stdout)
            loaded = trimesh.load(output)
        finally:
            output.unlink(missing_ok=True)

        self.assertIsNotNone(loaded)

    def test_non_positive_dimension_is_rejected(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "cli.py"),
                "generate",
                "--width",
                "0",
                "--height",
                "6",
                "--depth",
                "8",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("positive", completed.stderr)


class TestBlenderUnionScript(unittest.TestCase):
    def test_script_unions_meshes_instead_of_exporting_the_first(self):
        source = (ROOT / "core" / "blender_union_script.py").read_text()
        self.assertIn('type="BOOLEAN"', source)
        self.assertIn('modifier.operation = "UNION"', source)
        self.assertNotIn("Exporting the first one", source)
        self.assertNotIn("success_flag", source)


if __name__ == "__main__":
    unittest.main()
