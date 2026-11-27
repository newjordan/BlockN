import json
from pathlib import Path
import shutil
import tempfile
import unittest

import trimesh

from core.architecture import (
    BuildRequest,
    build,
    export_bundle,
    export_document,
    export_manifest,
    export_mesh_artifact,
    export_quality,
    load_document,
)
from core.scene import Scene


class TestArchitectureIO(unittest.TestCase):
    def setUp(self):
        self.result = build(
            BuildRequest(
                product="wall",
                seed=20260709,
                width=8.0,
                height=3.2,
                depth=0.3,
                variant="x",
                openings="mixed",
            )
        )

    def assert_canonical_json(self, path: Path):
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        expected = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        self.assertEqual(raw, expected)
        return payload

    def test_document_quality_and_manifest_exports(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "nested"
            document_path = export_document(
                self.result.document,
                root / "wall.architecture.json",
            )
            quality_path = export_quality(
                self.result.quality,
                root / "wall.quality.json",
            )
            manifest_path = export_manifest(
                self.result,
                root / "wall.manifest.json",
            )

            document_payload = self.assert_canonical_json(document_path)
            quality_payload = self.assert_canonical_json(quality_path)
            manifest_payload = self.assert_canonical_json(manifest_path)

            self.assertEqual(
                document_payload["fingerprint"],
                self.result.document.fingerprint(),
            )
            self.assertEqual(quality_payload, self.result.quality.to_dict())
            self.assertEqual(manifest_payload, self.result.manifest())

            loaded = load_document(document_path)
            self.assertEqual(loaded.to_dict(), self.result.document.to_dict())
            self.assertEqual(loaded.fingerprint(), self.result.document.fingerprint())
            self.assertEqual(list(root.glob(".*.tmp")), [])

    def test_bundle_writes_glb_and_semantic_node_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stem = Path(temporary_directory) / "bundle" / "wall"
            paths = export_bundle(self.result, stem)

            self.assertEqual(
                set(paths),
                {"glb", "architecture", "quality", "construction", "manifest", "physics"},
            )
            for path in paths.values():
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            exported_scene = trimesh.load(paths["glb"], force="scene", process=False)
            self.assertIsInstance(exported_scene, trimesh.Scene)
            expected_names = {placement.id for placement in self.result.document.placements}
            self.assertTrue(expected_names)
            self.assertTrue(expected_names.issubset(set(exported_scene.geometry)))
            self.assertTrue(expected_names.issubset(set(exported_scene.graph.nodes_geometry)))

            document_payload = self.assert_canonical_json(paths["architecture"])
            self.assertEqual(
                document_payload["fingerprint"],
                self.result.document.fingerprint(),
            )
            self.assert_canonical_json(paths["quality"])
            self.assert_canonical_json(paths["manifest"])

    def test_mesh_export_rejects_empty_scene_and_unknown_format(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(ValueError, "no mesh geometry"):
                export_mesh_artifact(Scene(use_collision=False), root / "empty.glb")
            with self.assertRaisesRegex(ValueError, "Unsupported mesh artifact format"):
                export_mesh_artifact(self.result.scene, root / "wall.xyz")
            if shutil.which("blender") is None:
                with self.assertRaisesRegex(ValueError, "FBX export needs Blender"):
                    export_mesh_artifact(self.result.scene, root / "wall.fbx")
                self.assertFalse((root / "wall.fbx").exists())

    def test_bundle_preflight_does_not_leave_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stem = Path(temporary_directory) / "invalid"
            self.result.document.metadata["invalid"] = float("nan")
            with self.assertRaisesRegex(ValueError, "NaN or infinite|finite JSON"):
                export_bundle(self.result, stem)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
