import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

from core.architecture.agent import SCHEMA, BatchQualityError, build_recipe, export_recipe, series_recipe
from core.architecture import rebuild


def kit_recipe(offset=0.6):
    return {"schema_version": SCHEMA, "buildings": [{"id": "courtyard-kit", "parts": [
        {"id": "podium", "request": {"product": "site-pad", "width": 22, "depth": 22, "height": 0.6}},
        {"id": "rotunda", "request": {"product": "classical-rotunda", "variant": "ionic"}, "offset": [0, offset, 0]},
    ]}]}


class AgentBatchTests(unittest.TestCase):
    def test_series_is_repeatable_and_varied_without_mutating_recipe(self):
        recipe = series_recipe(3, 73)
        before = deepcopy(recipe)
        first, district = build_recipe(recipe)
        second, again = build_recipe(series_recipe(3, 73))
        self.assertEqual(recipe, before)
        self.assertEqual(district.document.fingerprint(), again.document.fingerprint())
        self.assertEqual(district.quality.status, "pass")
        self.assertEqual(len({r.document.fingerprint() for _, r in first}), 3)
        self.assertNotEqual(recipe, series_recipe(3, 74))
        self.assertEqual(len(district.document.placements), sum(len(r.document.placements) for _, r in first))
        self.assertEqual([r.document.fingerprint() for _, r in first], [r.document.fingerprint() for _, r in second])

    def test_kitbash_offsets_roundtrip_and_intrinsic_connectivity(self):
        results, _ = build_recipe(kit_recipe())
        doc = results[0][1].document
        self.assertEqual(rebuild(doc).quality.status, "pass")
        self.assertEqual({p.parameters['kitbash_part_id'] for p in doc.placements}, {"podium", "rotunda"})
        doc.metadata.clear()
        for p in doc.placements:
            if p.parameters['kitbash_part_id'] == 'rotunda':
                p.position = (p.position[0] + 100, p.position[1], p.position[2])
        self.assertEqual(rebuild(doc).quality.status, "fail")

    def test_disconnected_kit_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'kit'
            with self.assertRaises(BatchQualityError):
                export_recipe(kit_recipe(100), output)
            self.assertFalse(output.exists())

    def test_bundle_paths_exist_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'kit'
            receipt = export_recipe(kit_recipe(), output, ('glb', 'obj'))
            self.assertEqual(receipt['count'], 1)
            for relative in receipt['buildings'][0]['artifacts'].values():
                self.assertTrue((output / relative).is_file())
            for relative in receipt['series'].values():
                self.assertTrue((output / relative).is_file())
            original = (output / 'batch.manifest.json').read_bytes()
            with self.assertRaises(ValueError):
                export_recipe(kit_recipe(), output)
            self.assertEqual((output / 'batch.manifest.json').read_bytes(), original)

    def test_automatic_kitbash_series_has_three_connected_parts_per_building(self):
        recipe = series_recipe(3, 73, kitbash=True)
        results, district = build_recipe(recipe)
        self.assertEqual(district.quality.status, "pass")
        for _, result in results:
            self.assertEqual(result.document.request.product, "kitbashed-building")
            self.assertEqual(result.quality.status, "pass")
            self.assertEqual({p.parameters['kitbash_part_id'] for p in result.document.placements},
                             {"podium", "building", "approach"})

    def test_bad_ids_unknown_fields_and_limits_rejected(self):
        for change in ({'id': '../escape'}, {'id': 'bad/id'}, {'request': {'product':'wall'}}):
            recipe = kit_recipe()
            recipe['buildings'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_recipe(recipe)
        recipe = kit_recipe()
        recipe['buildings'][0]['parts'][0]['request']['widht'] = 20
        with self.assertRaisesRegex(ValueError, 'widht'):
            build_recipe(recipe)
        for count in (0, 25, True):
            with self.assertRaises(ValueError):
                series_recipe(count)

    def test_agent_cli_outputs_json_and_error_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'series'
            command = [sys.executable, '-m', 'blockn', 'agent', 'series', '--count', '2', '--seed', '73', '--output', str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)['count'], 2)
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout)['status'], 'error')


if __name__ == '__main__':
    unittest.main()
