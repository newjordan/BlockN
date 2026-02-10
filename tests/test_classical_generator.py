import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.generators.classical import generate_classical_building, get_classical_snapshot_metrics
from core.scene import Scene


class TestClassicalGenerator(unittest.TestCase):
    def _generate(self, order: str, seed: int = 1234, **overrides):
        scene = Scene(use_collision=False)
        flags = {"use_wedge_pediment": False}
        flags.update(overrides.pop("flags", {}))
        generate_classical_building(
            scene,
            width=14,
            depth=10,
            podium_height_actual=3,
            seed=seed,
            flags=flags,
            order_override=order,
            preset_override=overrides.pop("preset_override", "canonical"),
            clear_scene=True,
            **overrides,
        )
        return scene

    def _scene_signature(self, scene: Scene):
        signature = []
        for block in scene.blocks:
            normalized_parameters = []
            for key, value in sorted(block.parameters.items()):
                if isinstance(value, float):
                    normalized_parameters.append((key, round(value, 6)))
                else:
                    normalized_parameters.append((key, value))
            signature.append(
                (
                    block.block_type,
                    tuple(float(v) for v in np.round(block.position, 6)),
                    tuple(float(v) for v in np.round(block.dimensions, 6)),
                    tuple(normalized_parameters),
                )
            )
        return signature

    def _shafts(self, scene: Scene):
        shafts = []
        for block in scene.blocks:
            if block.block_type != "CYLINDER":
                continue
            if block.parameters.get("element") != "shaft":
                continue
            shafts.append(block)
        return shafts

    def _cornice_front_width(self, scene: Scene) -> float:
        widths = [
            float(block.dimensions[0])
            for block in scene.blocks
            if block.parameters.get("element") == "cornice" and block.parameters.get("face") == "front"
        ]
        self.assertGreater(len(widths), 0)
        return max(widths)

    def test_doric_vs_ionic_have_distinct_column_rules(self):
        doric_scene = self._generate("Doric")
        ionic_scene = self._generate("Ionic")

        doric_shafts = self._shafts(doric_scene)
        ionic_shafts = self._shafts(ionic_scene)

        self.assertGreater(len(doric_shafts), 0)
        self.assertGreater(len(ionic_shafts), 0)

        doric_radius = np.mean([shaft.parameters["radius"] for shaft in doric_shafts])
        ionic_radius = np.mean([shaft.parameters["radius"] for shaft in ionic_shafts])
        self.assertGreater(doric_radius, ionic_radius)

        doric_slenderness = np.mean(
            [(shaft.parameters["height"] / (shaft.parameters["radius"] * 2.0)) for shaft in doric_shafts]
        )
        ionic_slenderness = np.mean(
            [(shaft.parameters["height"] / (shaft.parameters["radius"] * 2.0)) for shaft in ionic_shafts]
        )
        self.assertGreater(ionic_slenderness, doric_slenderness)

    def test_style_identity_elements_are_distinct(self):
        doric_scene = self._generate("Doric")
        ionic_scene = self._generate("Ionic")

        doric_triglyph_count = sum(
            1 for block in doric_scene.blocks if block.parameters.get("element") == "triglyph"
        )
        ionic_relief_count = sum(
            1 for block in ionic_scene.blocks if block.parameters.get("element") == "frieze_relief"
        )

        self.assertGreater(doric_triglyph_count, 0)
        self.assertGreater(ionic_relief_count, 0)

    def test_generation_is_deterministic_for_same_seed_and_order(self):
        scene_a = self._generate("Doric", seed=4242)
        scene_b = self._generate("Doric", seed=4242)

        self.assertEqual(self._scene_signature(scene_a), self._scene_signature(scene_b))

        metrics_a = get_classical_snapshot_metrics(scene_a)
        metrics_b = get_classical_snapshot_metrics(scene_b)
        self.assertEqual(metrics_a["block_count"], metrics_b["block_count"])
        self.assertEqual(metrics_a["bounds_min"], metrics_b["bounds_min"])
        self.assertEqual(metrics_a["bounds_max"], metrics_b["bounds_max"])

    def test_entablature_and_pediment_exist_when_enabled(self):
        scene = self._generate("Ionic", seed=2026)
        elements = {block.parameters.get("element") for block in scene.blocks}
        self.assertIn("architrave", elements)
        self.assertIn("frieze", elements)
        self.assertIn("cornice", elements)
        self.assertIn("pediment", elements)

    def test_snapshot_metrics_include_facade_and_column_counts(self):
        scene = self._generate("Doric", seed=1001)
        metrics = get_classical_snapshot_metrics(scene)

        self.assertIn("block_count", metrics)
        self.assertIn("bounds_min", metrics)
        self.assertIn("bounds_max", metrics)
        self.assertIn("facade_span", metrics)
        self.assertIn("front_columns", metrics)
        self.assertIn("side_columns", metrics)

        self.assertGreater(metrics["block_count"], 0)
        self.assertGreater(metrics["facade_span"], 0.0)
        self.assertGreaterEqual(metrics["front_columns"], 4)
        self.assertGreaterEqual(metrics["side_columns"], 4)

    def test_colonnade_runs_are_centered_within_footprint(self):
        scene = Scene(use_collision=False)
        width = 10
        depth = 5
        generate_classical_building(
            scene,
            width=width,
            depth=depth,
            podium_height_actual=2,
            seed=424242,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="canonical",
            clear_scene=True,
        )

        shafts = self._shafts(scene)
        self.assertGreater(len(shafts), 0)

        shaft_x = [shaft.position[0] for shaft in shafts]
        shaft_z = [shaft.position[2] for shaft in shafts]
        shaft_mid_x = (min(shaft_x) + max(shaft_x)) / 2.0
        shaft_mid_z = (min(shaft_z) + max(shaft_z)) / 2.0

        self.assertAlmostEqual(shaft_mid_x, width / 2.0, delta=0.01)
        self.assertAlmostEqual(shaft_mid_z, max(6, depth) / 2.0, delta=0.01)

    def test_placement_invariants_report_centering_footprint_and_doorway_clearance(self):
        scene = Scene(use_collision=False)
        metrics = generate_classical_building(
            scene,
            width=10,
            depth=5,
            podium_height_actual=2,
            seed=1122,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="canonical",
            clear_scene=True,
        )

        self.assertIn("invariants", metrics)
        self.assertTrue(metrics["invariants"]["centered_colonnade"])
        self.assertTrue(metrics["invariants"]["facade_in_footprint"])
        self.assertTrue(metrics["invariants"]["doorway_clearances_valid"])
        self.assertTrue(metrics["invariants"]["ok"])

        doorway = metrics["human_access"]["doorway_clearances"]
        self.assertGreaterEqual(doorway["front_clear_width"], doorway["front_passage_width"])
        self.assertGreaterEqual(doorway["side_clear_width"], doorway["side_passage_width"])
        self.assertGreater(doorway["front_clear_width"], 0.0)
        self.assertGreater(doorway["side_clear_width"], 0.0)

    def test_foundation_apron_is_opt_in_and_adds_support_blocks(self):
        default_scene = Scene(use_collision=False)
        generate_classical_building(
            default_scene,
            width=14,
            depth=10,
            podium_height_actual=3,
            seed=4040,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="canonical",
            clear_scene=True,
        )
        default_aprons = [b for b in default_scene.blocks if b.parameters.get("element") == "foundation_apron"]
        default_thresholds = [b for b in default_scene.blocks if b.parameters.get("element") == "doorway_threshold"]
        self.assertEqual(default_aprons, [])
        self.assertEqual(default_thresholds, [])

        apron_scene = Scene(use_collision=False)
        metrics = generate_classical_building(
            apron_scene,
            width=14,
            depth=10,
            podium_height_actual=3,
            seed=4040,
            flags={"use_wedge_pediment": False, "foundation_apron": True},
            order_override="Doric",
            preset_override="canonical",
            clear_scene=True,
        )
        apron_blocks = [b for b in apron_scene.blocks if b.parameters.get("element") == "foundation_apron"]
        threshold_blocks = [b for b in apron_scene.blocks if b.parameters.get("element") == "doorway_threshold"]

        self.assertGreater(len(apron_blocks), 0)
        self.assertGreater(len(threshold_blocks), 0)
        self.assertTrue(metrics["foundation_apron"]["enabled"])
        self.assertEqual(metrics["foundation_apron"]["blocks"], len(apron_blocks))
        self.assertEqual(metrics["foundation_apron"]["thresholds"], len(threshold_blocks))

    def test_human_access_elements_are_generated(self):
        scene = self._generate("Doric", seed=2024)

        entrance_stairs = [
            block
            for block in scene.blocks
            if block.parameters.get("element") == "stair"
            and block.parameters.get("stair_role") == "entrance"
        ]
        exit_stairs = [
            block
            for block in scene.blocks
            if block.parameters.get("element") == "stair"
            and block.parameters.get("stair_role") == "exit"
        ]

        self.assertGreater(len(entrance_stairs), 0)
        self.assertGreater(len(exit_stairs), 0)

        metrics = get_classical_snapshot_metrics(scene)
        self.assertIn("human_access", metrics)
        self.assertGreaterEqual(metrics["human_access"]["entrances"], 1)
        self.assertGreaterEqual(metrics["human_access"]["exits"], 1)
        self.assertGreater(metrics["human_access"]["stairs"], 0)

    def test_stair_spans_scale_beyond_doorway_widths(self):
        scene = self._generate("Doric", seed=2024)
        metrics = get_classical_snapshot_metrics(scene)
        stair_proportions = metrics["human_access"]["stair_proportions"]
        doorway = metrics["human_access"]["doorway_clearances"]

        self.assertGreater(stair_proportions["front_stair_width"], doorway["front_passage_width"] * 2.0)
        self.assertGreater(stair_proportions["side_stair_span"], doorway["side_passage_width"] * 1.5)

        front_steps = [
            block
            for block in scene.blocks
            if block.parameters.get("element") == "stair"
            and block.parameters.get("stair_role") == "entrance"
            and block.parameters.get("stair_side") == "front"
        ]
        self.assertGreater(len(front_steps), 0)
        self.assertGreater(max(block.dimensions[0] for block in front_steps), doorway["front_passage_width"] * 2.0)

    def test_large_facade_can_enable_wrap_stairs(self):
        scene = Scene(use_collision=False)
        metrics = generate_classical_building(
            scene,
            width=24,
            depth=20,
            podium_height_actual=4,
            seed=99,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="monumental",
            clear_scene=True,
        )

        stair_proportions = metrics["human_access"]["stair_proportions"]
        self.assertTrue(stair_proportions["front_wrap_enabled"])
        self.assertGreater(stair_proportions["front_wrap_span_z"], 0.0)

        wrap_steps = [
            block
            for block in scene.blocks
            if block.parameters.get("element") == "stair"
            and block.parameters.get("stair_layout") == "wrap"
        ]
        self.assertGreater(len(wrap_steps), 0)

    def test_wrap_trigger_is_tuned_by_preset(self):
        def wrap_enabled_for(preset: str, width: int, depth: int):
            scene = Scene(use_collision=False)
            metrics = generate_classical_building(
                scene,
                width=width,
                depth=depth,
                podium_height_actual=4,
                seed=99,
                flags={"use_wedge_pediment": False},
                order_override="Doric",
                preset_override=preset,
                clear_scene=True,
            )
            return bool(metrics["human_access"]["stair_proportions"]["front_wrap_enabled"])

        self.assertFalse(wrap_enabled_for("compact", width=20, depth=14))
        self.assertTrue(wrap_enabled_for("canonical", width=20, depth=14))
        self.assertTrue(wrap_enabled_for("monumental", width=20, depth=14))
        self.assertTrue(wrap_enabled_for("compact", width=24, depth=20))

    def test_wrap_trigger_thresholds_can_be_overridden(self):
        baseline_scene = Scene(use_collision=False)
        baseline = generate_classical_building(
            baseline_scene,
            width=20,
            depth=14,
            podium_height_actual=4,
            seed=99,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="canonical",
            clear_scene=True,
        )
        self.assertTrue(baseline["human_access"]["stair_proportions"]["front_wrap_enabled"])

        override_scene = Scene(use_collision=False)
        overridden = generate_classical_building(
            override_scene,
            width=20,
            depth=14,
            podium_height_actual=4,
            seed=99,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="canonical",
            wrap_min_podium_span_x_override=26.0,
            wrap_min_podium_span_z_override=16.0,
            wrap_front_coverage_ratio_override=0.70,
            clear_scene=True,
        )
        proportions = overridden["human_access"]["stair_proportions"]
        self.assertFalse(proportions["front_wrap_enabled"])
        self.assertAlmostEqual(proportions["wrap_trigger_min_span_x"], 26.0, places=6)
        self.assertAlmostEqual(proportions["wrap_trigger_min_span_z"], 16.0, places=6)
        self.assertAlmostEqual(proportions["wrap_trigger_front_coverage_ratio"], 0.70, places=6)

    def test_human_access_markers_are_opt_in(self):
        default_scene = self._generate("Doric", seed=2025)
        default_markers = [
            block
            for block in default_scene.blocks
            if block.parameters.get("element") in {"entrance_marker", "exit_marker"}
        ]
        self.assertEqual(default_markers, [])

        marked_scene = Scene(use_collision=False)
        generate_classical_building(
            marked_scene,
            width=14,
            depth=10,
            podium_height_actual=3,
            seed=2025,
            flags={"use_wedge_pediment": False, "entrance_markers": True},
            order_override="Doric",
            preset_override="canonical",
            clear_scene=True,
        )
        opt_in_markers = [
            block
            for block in marked_scene.blocks
            if block.parameters.get("element") in {"entrance_marker", "exit_marker"}
        ]
        self.assertGreater(len(opt_in_markers), 0)

    def test_manual_overrides_and_presets_are_applied(self):
        scene = Scene(use_collision=False)
        metrics = generate_classical_building(
            scene,
            width=30,
            depth=30,
            podium_height_actual=3,
            seed=31415,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="monumental",
            front_column_override=8,
            side_column_override=8,
            pediment_slope_multiplier=1.10,
            entablature_depth_multiplier=1.20,
            clear_scene=True,
        )

        self.assertEqual(metrics["preset"], "monumental")
        self.assertEqual(metrics["front_columns"], 8)
        self.assertEqual(metrics["side_columns"], 8)

    def test_side_roof_overhang_multiplier_widens_side_eaves(self):
        base_scene = Scene(use_collision=False)
        generate_classical_building(
            base_scene,
            width=22,
            depth=14,
            podium_height_actual=3,
            seed=5151,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="canonical",
            side_roof_overhang_multiplier=1.0,
            clear_scene=True,
        )

        widened_scene = Scene(use_collision=False)
        generate_classical_building(
            widened_scene,
            width=22,
            depth=14,
            podium_height_actual=3,
            seed=5151,
            flags={"use_wedge_pediment": False},
            order_override="Doric",
            preset_override="canonical",
            side_roof_overhang_multiplier=1.4,
            clear_scene=True,
        )

        base_front_cornice = self._cornice_front_width(base_scene)
        widened_front_cornice = self._cornice_front_width(widened_scene)
        self.assertGreater(widened_front_cornice, base_front_cornice)

    def test_regression_matrix_keeps_classical_population_invariants(self):
        cases = [
            (10, 5, 1001, "Doric"),
            (10, 5, 1002, "Ionic"),
            (12, 8, 2001, "Doric"),
            (14, 10, 3001, "Ionic"),
            (18, 12, 4001, "Doric"),
        ]
        for width, depth, seed, order in cases:
            with self.subTest(width=width, depth=depth, seed=seed, order=order):
                scene = Scene(use_collision=False)
                metrics = generate_classical_building(
                    scene,
                    width=width,
                    depth=depth,
                    podium_height_actual=3,
                    seed=seed,
                    flags={"use_wedge_pediment": False, "foundation_apron": True},
                    order_override=order,
                    preset_override="canonical",
                    clear_scene=True,
                )
                self.assertTrue(metrics["invariants"]["ok"])
                self.assertTrue(metrics["invariants"]["centered_colonnade"])
                self.assertTrue(metrics["invariants"]["facade_in_footprint"])
                self.assertTrue(metrics["invariants"]["doorway_clearances_valid"])
                self.assertGreater(metrics["block_count"], 0)

    def test_random_and_typo_alias_resolve_to_supported_orders(self):
        random_scene = self._generate("Random", seed=77)
        random_orders = {
            block.parameters.get("order")
            for block in random_scene.blocks
            if block.parameters.get("element") == "shaft"
        }
        self.assertTrue(random_orders.issubset({"Doric", "Ionic"}))

        typo_scene = self._generate("ioinic", seed=77)
        typo_orders = {
            block.parameters.get("order")
            for block in typo_scene.blocks
            if block.parameters.get("element") == "shaft"
        }
        self.assertEqual(typo_orders, {"Ionic"})


if __name__ == '__main__':
    unittest.main()
