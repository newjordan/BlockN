"""Headless GUI checks, skipped when optional desktop dependencies are absent."""
import importlib.util
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
GUI_AVAILABLE = all(importlib.util.find_spec(name) for name in ("PyQt5", "pyvista", "pyvistaqt"))


@unittest.skipUnless(GUI_AVAILABLE, "optional GUI dependencies unavailable")
class ClassicalGUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_form_selection_loads_defaults_and_relevant_controls(self):
        from gui.panels import GenerationPanel
        from core.architecture.classical_forms import FORMS, MASONRY_FORMS
        panel = GenerationPanel()
        self.assertEqual((panel.width_spinbox.value(), panel.height_spinbox.value(), panel.depth_spinbox.value()), (14, 9, 20))
        self.assertEqual(panel.age_spinbox.value(), 0.35)
        self.assertEqual((panel.age_spinbox.minimum(), panel.age_spinbox.maximum()), (0.0, 1.0))
        self.assertEqual(panel.classical_order_combo.count(), 6)
        self.assertEqual(panel.classical_form_combo.count(), 31)
        for product, (_, size, _) in FORMS.items():
            panel.classical_form_combo.setCurrentIndex(panel.classical_form_combo.findData(product))
            self.assertEqual((panel.width_spinbox.value(), panel.height_spinbox.value(), panel.depth_spinbox.value()), size)
            self.assertTrue(panel.classical_tuning_group.isHidden())
            self.assertEqual(panel.classical_order_row.isHidden(), product in MASONRY_FORMS)
            self.assertEqual(panel.form_levels.isEnabled(), product == "classical-aqueduct")
        panel.classical_form_combo.setCurrentIndex(panel.classical_form_combo.findData("marble-run"))
        self.assertEqual((panel.width_spinbox.value(), panel.height_spinbox.value(), panel.depth_spinbox.value()), (36, 10, 28))
        self.assertTrue(panel.classical_order_row.isHidden())
        self.assertTrue(panel.form_levels.isEnabled())
        panel.classical_form_combo.setCurrentIndex(0)
        self.assertFalse(panel.classical_tuning_group.isHidden())
        panel.close()

    def test_timberwright_controls_generate_a_house_not_a_classical_order(self):
        from gui.main_window import GenerationThread
        from gui.panels import GenerationPanel

        panel = GenerationPanel()
        combo = panel.classical_form_combo
        combo.setCurrentIndex(combo.findData('timber-house'))
        self.assertFalse(panel.timber_controls.isHidden())
        self.assertTrue(panel.classical_order_row.isHidden())
        self.assertEqual((panel.form_levels.minimum(), panel.form_levels.maximum()), (1, 2))
        panel.timber_variant.setCurrentIndex(panel.timber_variant.findData('twin-gable'))
        panel.timber_finish.setCurrentIndex(panel.timber_finish.findData('rose-shale'))
        panel._on_randomize_controls()
        self.assertEqual(panel.timber_variant.currentData(), 'twin-gable')
        panel.age_spinbox.setValue(0.0)
        worker = GenerationThread({'classical_form': combo.currentData(), 'width': panel.width_spinbox.value(),
            'depth': panel.depth_spinbox.value(), 'height': panel.height_spinbox.value(), 'seed': 73,
            'form_levels': panel.form_levels.value(), 'form_bays': panel.form_bays.value(), 'form_detail': 'concept',
            'timber_variant': panel.timber_variant.currentData(), 'timber_parameters': panel.timber_parameters(),
            'age': panel.age_spinbox.value()})
        emitted = []
        worker.generation_finished.connect(lambda scene, error, metadata: emitted.append((scene, error, metadata)))
        worker.run()
        scene, error, metadata = emitted[0]
        self.assertFalse(error, error)
        self.assertEqual(scene.export_metadata['age'], 0.0)
        self.assertEqual(metadata['order'], 'twin-gable')
        self.assertEqual(metadata['quality'], 'pass')
        self.assertTrue(any(b.parameters.get('component_role') == 'entry_lintel' for b in scene.blocks))
        combo.setCurrentIndex(0)
        self.assertTrue(panel.timber_controls.isHidden())
        panel.close()

    def test_cathedral_controls_offer_real_design_parameters(self):
        from gui.panels import GenerationPanel
        panel=GenerationPanel()
        panel.classical_form_combo.setCurrentIndex(panel.classical_form_combo.findData("gothic-cathedral"))
        self.assertFalse(panel.cathedral_controls.isHidden())
        self.assertTrue(panel.classical_order_row.isHidden())
        self.assertTrue(panel.form_levels.isHidden())
        self.assertEqual((panel.form_bays.minimum(),panel.form_bays.maximum()),(4,10))
        panel.cathedral_choir.setValue(4)
        panel.cathedral_apse.setCurrentText("9")
        self.assertEqual(panel.cathedral_parameters(),{"choir_bays":4,"apse_sides":9,"spire_ratio":.30})
        panel.cathedral_style.setCurrentIndex(panel.cathedral_style.findData("early-english"))
        self.assertNotIn("spire_ratio",panel.cathedral_parameters())
        panel.close()

    def test_cathedral_worker_uses_style_and_choir_controls(self):
        from gui.main_window import GenerationThread
        outputs=[]
        worker=GenerationThread({"classical_form":"gothic-cathedral","width":32,"height":50,"depth":64,
            "seed":73,"cathedral_style":"early-english","form_detail":"concept","form_bays":8,
            "cathedral_parameters":{"choir_bays":4,"apse_sides":9}})
        worker.generation_finished.connect(lambda *args: outputs.append(args))
        worker.run()
        scene,error,metadata=outputs[0]
        self.assertEqual(error,"")
        self.assertEqual(metadata["order"],"early-english")
        self.assertEqual(metadata["bays"],8)
        self.assertTrue(any(b.parameters["component_role"]=="vault_rib" for b in scene.blocks))

    def test_worker_builds_semantic_scene_and_reports_invalid_input(self):
        from gui.main_window import GenerationThread
        for width, depth, expect_error in ((16, 16, False), (20, 10, True)):
            outputs = []
            worker = GenerationThread({"classical_form": "classical-rotunda", "width": width,
                "height": 10, "depth": depth, "seed": 7, "classical_order": "Composite"})
            worker.generation_finished.connect(
                lambda scene, error, metadata: outputs.append((scene, error, metadata))
            )
            worker.run()
            self.assertEqual(len(outputs), 1)
            scene, error, metadata = outputs[0]
            self.assertEqual(bool(error), expect_error)
            if expect_error:
                self.assertIsNone(scene)
                self.assertIsNone(metadata)
            else:
                self.assertGreater(len(scene.blocks), 10)
                self.assertTrue(all("primitive_id" in b.parameters for b in scene.blocks))
                self.assertEqual(metadata["order"], "Composite")
                self.assertEqual(metadata["typology"], "rotunda")
                self.assertEqual(metadata["quality"], "pass")
                self.assertGreater(metadata["checks"]["pass"], 0)

    def test_temple_worker_uses_audited_semantic_geometry(self):
        from gui.main_window import GenerationThread
        for order in ("Doric", "Ionic"):
            outputs = []
            worker = GenerationThread({"classical_form": "classical-building", "width": 14,
                "height": 9, "depth": 20, "seed": 73, "classical_order": order,
                "form_detail": "presentation"})
            worker.generation_finished.connect(lambda *args: outputs.append(args))
            worker.run()
            scene, error, metadata = outputs[0]
            self.assertEqual(error, "")
            self.assertEqual(metadata["quality"], "pass")
            self.assertEqual(metadata["order"].lower(), order.lower())
            self.assertTrue(all("primitive_id" in b.parameters for b in scene.blocks))
            self.assertTrue(any(b.parameters.get("component_role") == "roof_panel" for b in scene.blocks))
            self.assertTrue(any(b.parameters.get("component_role") == "cella_wall" for b in scene.blocks))

    def test_worker_reports_masonry_forms_as_having_no_order(self):
        from gui.main_window import GenerationThread
        outputs = []
        worker = GenerationThread({"classical_form": "classical-aqueduct", "width": 28, "height": 12,
            "depth": 3, "seed": 3, "classical_order": "Doric", "form_bays": 4, "form_levels": 3})
        worker.generation_finished.connect(
            lambda scene, error, metadata: outputs.append((scene, error, metadata))
        )
        worker.run()
        scene, error, metadata = outputs[0]
        self.assertEqual(error, "")
        self.assertIsNone(metadata["order"])
        self.assertEqual(metadata["typology"], "aqueduct")

    def test_seed_field_rejects_partial_input_and_generate_never_raises(self):
        from PyQt5.QtGui import QValidator
        from gui.panels import GenerationPanel
        panel = GenerationPanel()
        validator = panel.seed_input.validator()
        self.assertEqual(validator.validate("-", 1)[0], QValidator.Invalid)
        emitted = []
        panel.generate_triggered.connect(lambda *values: emitted.append(values))
        for text in ("-", "+", ""):
            panel.seed_input.setText(text)
            panel._on_generate_clicked()
        self.assertEqual(len(emitted), 3)
        self.assertTrue(all(isinstance(values[3], int) for values in emitted))
        panel.close()

    def test_marble_rider_trace_is_hotter_and_thicker_than_phosphor(self):
        from gui.viewport import MARBLE_LINE_WIDTH, MARBLE_RADIUS_SCALE, MARBLE_TRACE, PHOSPHOR

        def rgb(color):
            digits = color.lstrip("#")
            return tuple(int(digits[i:i + 2], 16) for i in (0, 2, 4))

        marble = rgb(MARBLE_TRACE)
        phosphor = rgb(PHOSPHOR)
        self.assertNotEqual(MARBLE_TRACE.lower(), PHOSPHOR.lower())
        self.assertEqual(max(marble), marble[1])
        self.assertGreater(sum(marble), sum(phosphor))
        self.assertGreaterEqual(MARBLE_LINE_WIDTH, 4)
        self.assertGreater(MARBLE_RADIUS_SCALE, 1.0)
        self.assertLess(MARBLE_RADIUS_SCALE, 2.0)

    def test_randomize_varies_typology_geometry_without_changing_form_or_order(self):
        from gui.panels import GenerationPanel
        from core.architecture import BuildRequest, build
        from core.architecture.classical_forms import FORMS, MASONRY_FORMS
        panel = GenerationPanel()
        for product in ("classical-basilica", "classical-aqueduct"):
            panel.classical_form_combo.setCurrentIndex(panel.classical_form_combo.findData(product))
            panel.classical_order_combo.setCurrentText("Ionic")
            seen = set()
            for _ in range(5):
                panel._on_randomize_controls()
                self.assertEqual(panel.classical_form_combo.currentData(), product)
                self.assertEqual(panel.classical_order_combo.currentText(), "Ionic")
                self.assertTrue(panel.seed_input.text().isdigit())
                controls = (panel.width_spinbox.value(), panel.height_spinbox.value(),
                            panel.depth_spinbox.value(), panel.form_bays.value(),
                            panel.form_detail.currentText())
                seen.add(controls)
                result = build(BuildRequest(
                    product=product, width=controls[0], height=controls[1], depth=controls[2],
                    bays=controls[3], detail=controls[4],
                    variant="auto" if product in MASONRY_FORMS else "ionic",
                    levels=panel.form_levels.value() if product == "classical-aqueduct" else None,
                ))
                self.assertEqual(result.quality.status, "pass")
            self.assertGreater(len(seen), 1, product)
            self.assertNotEqual(seen, {FORMS[product][1] + (5, "standard")})
        panel.close()


if __name__ == "__main__":
    unittest.main()
