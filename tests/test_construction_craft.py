import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import trimesh

from core.architecture import BuildRequest, build, rebuild
from core.architecture.construction import craft_catalog, construction_report
from core.architecture.io import export_bundle
from core.architecture.schema import BuildDocument


class ConstructionCraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doric = build(BuildRequest(product="classical-building", variant="doric", seed=73))
        cls.ionic = build(BuildRequest(product="classical-building", variant="ionic", seed=73))

    def test_fluted_shafts_are_cubit_drums(self):
        from core.architecture.masonry import MAX_COURSE, MAX_STRETCHER, placement_is_oversized
        for result in (self.doric, self.ionic):
            shafts = [p for p in result.document.placements if p.role == "column_shaft"]
            self.assertTrue(shafts)
            hosts = {(round(p.position[0], 3), round(p.position[2], 3)) for p in shafts}
            self.assertGreater(len(shafts), len(hosts))
            self.assertTrue(all(p.parameters.get("unit") == "drum" for p in shafts))
            self.assertTrue(all(p.block_type == "MESH" for p in shafts))
            self.assertEqual(
                [p.id for p in shafts if placement_is_oversized(p)
                 or p.dimensions[1] > MAX_COURSE + 1e-6
                 or max(p.dimensions[0], p.dimensions[2]) > MAX_STRETCHER + 1e-6],
                [],
            )
            carved = [p for p in result.document.placements
                      if p.role in {"column_capital_abacus", "column_capital_bell", "column_capital_echinus"}]
            self.assertEqual(
                [p.id for p in carved
                 if p.dimensions[1] > MAX_COURSE + 1e-6
                 or max(p.dimensions[0], p.dimensions[2]) > MAX_STRETCHER + 1e-6],
                [],
            )

    def test_temple_bearings_and_orders_pass_for_real_geometry(self):
        for result in (self.doric, self.ionic):
            self.assertEqual(result.quality.status, "pass")
            checks = {c.code: c.status for c in result.quality.checks}
            for code in ("craft.order_members", "craft.shaft_profiles", "craft.capital_bearings", "classical.temple_members"):
                self.assertEqual(checks[code], "pass")

    def test_removing_an_ionic_scroll_is_caught_without_invariant_metadata(self):
        doc = BuildDocument.from_dict(self.ionic.document.to_dict())
        doc.metadata = {}
        victim = next(p for p in doc.placements if p.role == "column_capital_volute")
        doc.placements.remove(victim)
        checks = {c.code: c.status for c in rebuild(doc).quality.checks}
        self.assertEqual(checks["craft.order_members"], "fail")

    def test_cylinder_substitution_cannot_pass_by_claiming_flutes(self):
        doc = BuildDocument.from_dict(self.doric.document.to_dict())
        shaft = next(p for p in doc.placements if p.role == "column_shaft")
        shaft.block_type = "CYLINDER"
        shaft.parameters = {"element": "shaft", "order": "Doric", "flute_count": 20,
                            "radius": shaft.dimensions[0]/2, "height": shaft.dimensions[1], "sections": 40}
        checks = {c.code: c.status for c in rebuild(doc).quality.checks}
        self.assertEqual(checks["craft.shaft_profiles"], "fail")

    def test_displacing_side_lintels_removes_capital_bearing(self):
        doc = BuildDocument.from_dict(self.doric.document.to_dict())
        for p in doc.placements:
            if p.role == "architrave" and p.parameters.get("face") == "left":
                p.position = (p.position[0]-2, p.position[1], p.position[2])
        checks = {c.code: c.status for c in rebuild(doc).quality.checks}
        self.assertEqual(checks["craft.capital_bearings"], "fail")

    def test_temple_has_open_attic_and_a_real_cella_doorway(self):
        for result in (self.doric, self.ionic):
            roofs = [b for b in result.scene.blocks if b.parameters["component_role"] == "roof_panel"]
            self.assertGreaterEqual(len(roofs), 2)
            self.assertEqual({b.parameters.get("side") for b in roofs}, {-1, 1})
            combined = trimesh.util.concatenate([b.mesh for b in roofs])
            self.assertLess(combined.volume / np.prod(combined.extents), .22)
            lintels = [b for b in result.scene.blocks if b.parameters["component_role"] == "cella_lintel"]
            self.assertTrue(lintels)
            door = np.mean([b.position for b in lintels], axis=0)
            door[1] = min(b.mesh.bounds[0, 1] for b in lintels) - .1
            walls = [b for b in result.scene.blocks if b.parameters["component_role"] == "cella_wall"]
            self.assertFalse(any(np.all(door > b.mesh.bounds[0]) and np.all(door < b.mesh.bounds[1]) for b in walls))

    def test_temple_stairs_ridge_and_raking_cornice_are_coursed_ashlar(self):
        from core.architecture.masonry import placement_is_oversized

        for result in (self.doric, self.ionic):
            for role in ("stair", "ridge", "raking_cornice"):
                parts = [p for p in result.document.placements if p.role == role]
                self.assertGreater(len(parts), 1)
                self.assertTrue(all(p.block_type == "CUBE" for p in parts))
                self.assertTrue(all(p.parameters.get("unit") == "ashlar" for p in parts))
                self.assertEqual([p.id for p in parts if placement_is_oversized(p)], [])
            panels = [p for p in result.document.placements if p.role == "roof_panel"]
            self.assertGreater(len(panels), 2)
            self.assertTrue(all(p.block_type == "MESH" for p in panels))
            self.assertEqual(
                [p.id for p in panels if max(p.dimensions[0], p.dimensions[2]) > 1.60 + 1e-6
                 or p.dimensions[1] > 0.80 + 1e-6],
                [],
            )
            tympana = [p for p in result.document.placements if p.role == "pediment"]
            self.assertGreater(len(tympana), 2)
            self.assertTrue(all(p.block_type == "MESH" for p in tympana))
            self.assertEqual([p.id for p in tympana if placement_is_oversized(p)], [])
            seams = [p for p in result.document.placements if p.role == "roof_seam"]
            self.assertGreater(len(seams), 2)
            self.assertTrue(all(p.block_type == "MESH" for p in seams))
            self.assertEqual(
                [p.id for p in seams if max(p.dimensions[0], p.dimensions[2]) > 1.60 + 1e-6
                 or p.dimensions[1] > 0.80 + 1e-6],
                [],
            )

    def test_construction_export_covers_every_member_once(self):
        report = construction_report(self.ionic.document, self.ionic.quality)
        ids = [pid for stage in report["assembly_stages"] for pid in stage["primitive_ids"]]
        self.assertEqual(sorted(ids), sorted(p.id for p in self.ionic.document.placements))
        self.assertIn("greek-ionic", report["traditions"])
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_bundle(self.ionic, Path(tmp)/"ionic", formats=("glb",))
            self.assertEqual(json.loads(paths["construction"].read_text()), report)

    def test_craft_catalog_is_detached_and_cli_is_json(self):
        one = craft_catalog()
        one["traditions"]["greek-doric"]["rules"].clear()
        self.assertGreater(len(craft_catalog()["traditions"]["greek-doric"]["rules"]), 0)
        completed = subprocess.run([sys.executable, "-m", "blockn", "craft"], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout)["schema_version"], "blockn.craft/v1")

    def test_roman_vault_ribs_do_not_claim_gothic_construction(self):
        for product in ("classical-baths", "classical-rotunda"):
            result = build(BuildRequest(product=product, detail="concept", seed=73))
            traditions = construction_report(result.document, result.quality)["traditions"]
            self.assertIn("roman-masonry", traditions)
            self.assertNotIn("gothic-masonry", traditions)
            # Kitbash namespaces retain provenance even after the outer
            # document no longer requests the original building product.
            doc = BuildDocument.from_dict(result.document.to_dict())
            doc.request = BuildRequest(product="kitbash-series", architect="kitbash")
            for p in doc.placements:
                p.id = "sample." + p.id
            self.assertIn("roman-masonry", construction_report(doc)["traditions"])


if __name__ == "__main__":
    unittest.main()
