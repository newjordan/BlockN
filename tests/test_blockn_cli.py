import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import trimesh


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class TestBlockNCLI(unittest.TestCase):
    def run_blockn(self, *arguments: object) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        return subprocess.run(
            [sys.executable, "-m", "blockn", *(str(value) for value in arguments)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )

    def json_stdout(self, completed: subprocess.CompletedProcess[str]):
        self.assertEqual(
            completed.returncode,
            0,
            f"command failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            self.fail(
                f"stdout was not valid JSON: {exc}\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

    def test_catalog_json_lists_all_products_and_mesh_primitives(self):
        completed = self.run_blockn("catalog", "--json", "--include-meshes")
        payload = self.json_stdout(completed)

        products = payload["products"]
        product_names = [item["name"] for item in products]
        self.assertEqual(len(products), 67)
        self.assertEqual(len(set(product_names)), len(products))
        self.assertIn("wall", product_names)
        self.assertIn("classical-building", product_names)
        self.assertIn("marble-run", product_names)
        self.assertIn("timber-house", product_names)
        self.assertIn("victorian-terrace", product_names)
        for historic_name in (
            "norman-keep", "norman-shell-keep", "norman-castle",
            "egyptian-temple", "mesoamerican-pyramid", "japanese-pagoda",
            "chinese-hall", "chinese-palace-hall", "dravidian-gopuram", "dravidian-vimana",
            "stave-church", "stave-loft", "sahelian-mosque", "inka-kancha", "khmer-temple-mountain", "swahili-courtyard-house",
            "minangkabau-rumah-gadang", "chacoan-great-house", "brutalist-civic-center",
            "brutalist-housing", "brutalist-arts-center", "brutalist-portal-field", "byzantine-church",
            "mesopotamian-ziggurat", "buddhist-stupa", "muscovite-church", "zimbabwe-enclosure", "lalibela-church",
            "artdeco-tower", "geodesic-dome", "tibetan-palace", "mughal-tomb",
        ):
            self.assertIn(historic_name, product_names)

        primitive_names = {item["name"] for item in payload["mesh_primitives"]}
        self.assertEqual(primitive_names, {"CUBE", "WEDGE", "CYLINDER", "ARCH", "DOME", "MESH"})

        architect_payload = self.json_stdout(self.run_blockn("architects", "--json"))
        architect_ids = {item["id"] for item in architect_payload["architects"]}
        self.assertEqual(
            architect_ids,
            {"primitive", "circulation", "envelope", "site", "plan", "classical", "structure", "spire",
             "cathedral", "marble", "timberwright", "castle", "egyptian", "mesoamerican", "pagoda", "persian",
             "settlement", "victorian", "chinese", "dravidian", "stave", "sahelian", "inka", "khmer", "swahili",
             "minangkabau", "chacoan", "brutalist", "byzantine", "mesopotamian", "stupa", "muscovite", "zimbabwe",
             "lalibela", "artdeco", "geodesic", "tibetan", "mughal"},
        )

    def test_roll_recipe_and_make_randomize_share_the_same_request(self):
        product = "swahili-courtyard-house"
        recipe = self.json_stdout(self.run_blockn("agent", "series", "--products", product, "--count", 1,
                                                   "--seed", 21, "--recipe-only", "--detail", "concept"))
        default_roll = self.json_stdout(self.run_blockn("roll", product, "--seed", 21, "--detail", "concept"))
        self.assertEqual(recipe["buildings"][0]["request"], default_roll)
        options = ("--seed", 21, "--variant", "single-court", "--detail", "concept")
        pinned = self.json_stdout(self.run_blockn("roll", product, *options))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rolled.json"
            self.json_stdout(self.run_blockn("make", product, "--randomize", *options, "--bays", 4,
                                             "--document", path, "--json-summary"))
            self.assertEqual(json.loads(path.read_text())["request"], {**pinned, "bays": 4})

    def test_make_wall_bundle_has_quality_and_semantic_glb_geometry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stem = Path(temporary_directory) / "nested" / "mixed-wall"
            completed = self.run_blockn(
                "make",
                "wall",
                "--width",
                6,
                "--height",
                3,
                "--depth",
                0.3,
                "--openings",
                "mixed",
                "--seed",
                77,
                "--bundle",
                stem,
                "--json-summary",
            )
            summary = self.json_stdout(completed)

            expected_paths = {
                "glb": Path(f"{stem}.glb"),
                "architecture": Path(f"{stem}.architecture.json"),
                "quality": Path(f"{stem}.quality.json"),
                "construction": Path(f"{stem}.construction.json"),
                "manifest": Path(f"{stem}.manifest.json"),
                "physics": Path(f"{stem}.physics.json"),
            }
            self.assertEqual(summary["quality"], "pass")
            self.assertEqual(
                summary["artifacts"],
                {key: str(path) for key, path in sorted(expected_paths.items())},
            )
            for path in expected_paths.values():
                self.assertTrue(path.is_file(), path)
                self.assertGreater(path.stat().st_size, 0, path)

            document = json.loads(expected_paths["architecture"].read_text(encoding="utf-8"))
            quality = json.loads(expected_paths["quality"].read_text(encoding="utf-8"))
            manifest = json.loads(expected_paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(quality["status"], "pass")
            self.assertEqual(manifest["quality"]["status"], "pass")
            self.assertEqual(document["fingerprint"], summary["fingerprint"])
            self.assertEqual(manifest["fingerprint"], summary["fingerprint"])

            semantic_ids = {placement["id"] for placement in document["placements"]}
            exported_scene = trimesh.load(
                expected_paths["glb"],
                force="scene",
                process=False,
            )
            self.assertIsInstance(exported_scene, trimesh.Scene)
            self.assertEqual(len(exported_scene.geometry), len(document["placements"]))
            self.assertEqual(set(exported_scene.geometry), semantic_ids)
            self.assertEqual(set(exported_scene.graph.nodes_geometry), semantic_ids)

    def test_same_command_is_deterministic(self):
        arguments = (
            "make",
            "column",
            "--variant",
            "round",
            "--height",
            2.5,
            "--width",
            0.6,
            "--depth",
            0.6,
            "--seed",
            4242,
            "--json-summary",
        )
        first = self.json_stdout(self.run_blockn(*arguments))
        second = self.json_stdout(self.run_blockn(*arguments))

        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.assertEqual(first, second)
        self.assertEqual(first["quality"], "pass")

    def test_summary_retains_repeated_output_formats(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stem = root / "bundle"
            extra_one = root / "one.glb"
            extra_two = root / "two.glb"
            summary = self.json_stdout(
                self.run_blockn(
                    "make",
                    "slab",
                    "--bundle",
                    stem,
                    "--out",
                    extra_one,
                    "--out",
                    extra_two,
                    "--json-summary",
                )
            )
            self.assertEqual(summary["artifacts"]["glb"], f"{stem}.glb")
            self.assertEqual(summary["artifacts"]["glb_2"], str(extra_one))
            self.assertEqual(summary["artifacts"]["glb_3"], str(extra_two))

    def test_invalid_dimensions_and_unknown_product_are_actionable(self):
        invalid_dimensions = self.run_blockn(
            "make",
            "wall",
            "--width",
            0,
            "--json-summary",
        )
        self.assertEqual(invalid_dimensions.returncode, 2)
        self.assertIn("width must be between", invalid_dimensions.stderr)

        unknown_product = self.run_blockn(
            "make",
            "moon-base",
            "--json-summary",
        )
        self.assertEqual(unknown_product.returncode, 2)
        self.assertIn("unknown product 'moon-base'", unknown_product.stderr)
        self.assertIn("wall", unknown_product.stderr)
        self.assertIn("classical-building", unknown_product.stderr)

        with tempfile.TemporaryDirectory() as temporary_directory:
            stem = Path(temporary_directory) / "nonfinite"
            nonfinite = self.run_blockn(
                "make",
                "wall",
                "--param",
                "unused=NaN",
                "--bundle",
                stem,
            )
            self.assertEqual(nonfinite.returncode, 2)
            self.assertIn("NaN or infinite", nonfinite.stderr)
            self.assertEqual(list(Path(temporary_directory).iterdir()), [])

    def test_build_request_json_then_inspect_exported_document(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            request_path = root / "slab.request.json"
            request_path.write_text(
                json.dumps(
                    {
                        "request": {
                            "product": "slab",
                            "seed": 909,
                            "width": 4.0,
                            "height": 0.25,
                            "depth": 3.0,
                            "variant": "floor",
                        }
                    }
                ),
                encoding="utf-8",
            )
            stem = root / "request-build" / "slab"

            build_completed = self.run_blockn(
                "build",
                request_path,
                "--bundle",
                stem,
                "--json-summary",
            )
            build_summary = self.json_stdout(build_completed)
            self.assertEqual(build_summary["product"], "slab")
            self.assertEqual(build_summary["quality"], "pass")
            # A 4 by 3 floor is coursed into lift-able ashlars, not one slab.
            self.assertEqual(build_summary["placements"], 15)

            document_path = Path(f"{stem}.architecture.json")
            self.assertTrue(document_path.is_file())
            inspect_completed = self.run_blockn("inspect", document_path, "--json")
            quality = self.json_stdout(inspect_completed)
            self.assertEqual(quality["status"], "pass")
            self.assertEqual(quality["metrics"]["placement_count"], 15)
            self.assertEqual(quality["metrics"]["compiled_block_count"], 15)
            self.assertTrue(
                all(check["status"] == "pass" for check in quality["checks"]),
                quality["checks"],
            )


if __name__ == "__main__":
    unittest.main()
