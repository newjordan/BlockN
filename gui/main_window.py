from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import trimesh
from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.geometry import optimize_blocks
from core.architecture.io import export_mesh_artifact
from core.io import save_scene_to_json
from core.scene import ColorTuple, Scene
from .panels import (
    ColorPalettePanel,
    GenerationPanel,
    OutlinerPanel,
    SceneExportPanel,
)
from .viewport import ViewportWidget


from .theme import PHOSPHOR_STYLESHEET


class GenerationThread(QThread):
    """Runs generation in the background to keep UI responsive."""

    generation_finished = pyqtSignal(object, str, object)

    def __init__(self, params: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.params = params

    def _apply_scene_runtime_settings(self, scene: Scene):
        support_enabled = bool(self.params.get("support_enabled", True))
        overlap_enabled = bool(self.params.get("strict_overlap", True))

        # KISS baseline: support toggle uses GRID check only.
        scene.support_check_mode = "GRID" if support_enabled else "OFF"

        if scene.collision_manager is None:
            scene.enable_strict_overlap_prevention = False
        else:
            scene.enable_strict_overlap_prevention = overlap_enabled

    def run(self):
        new_scene = Scene()
        self._apply_scene_runtime_settings(new_scene)
        error_message = ""

        try:
            width = int(self.params["width"])
            height = int(self.params["height"])
            depth = int(self.params["depth"])
            seed = self.params["seed"]
            age = float(self.params.get("age", 0.35))

            product = self.params.get("classical_form", "classical-building")
            if product != "classical-building":
                from core.architecture import BuildRequest, build
                from core.architecture.classical_forms import MASONRY_FORMS
                from core.architecture.cathedral import PRODUCT as CATHEDRAL_PRODUCT
                from core.architecture.marble_run import PRODUCT as MARBLE_PRODUCT
                from core.architecture.timberwright import PRODUCT as TIMBER_PRODUCT
                from core.architecture.settlement import PRODUCT_NAMES as SETTLEMENT_PRODUCTS
                from core.architecture import historic
                if product in SETTLEMENT_PRODUCTS:
                    result = build(BuildRequest(
                        product=product, width=width, depth=depth, seed=seed or 0,
                        variant=self.params.get("settlement_variant") or "auto",
                        detail=self.params.get("form_detail", "concept"),
                        age=age,
                    ))
                    if result.quality.status == "fail":
                        raise ValueError("; ".join(
                            c.message for c in result.quality.checks if c.status == "fail"))
                    plan = result.document.metadata["settlement"]
                    self.generation_finished.emit(result.scene, "", {
                        "order": plan["variant"],
                        "variant_label": "plan",
                        "typology": plan["typology"],
                        "bays": None,
                        "detail": result.document.request.detail,
                        "quality": result.quality.status,
                        "checks": dict(result.quality.counts),
                    })
                    return
                found = historic.style_for(product)
                from core.architecture.persian import PRODUCT as PERSIAN_PRODUCT
                if product == TIMBER_PRODUCT:
                    variant = self.params.get("timber_variant", "auto")
                    design_params = self.params.get("timber_parameters", {})
                elif product == PERSIAN_PRODUCT:
                    variant = self.params.get("persian_variant", "auto")
                    design_params = self.params.get("persian_parameters", {})
                elif product == CATHEDRAL_PRODUCT:
                    variant = self.params.get("cathedral_style", "high-gothic")
                    design_params = self.params.get("cathedral_parameters", {})
                else:
                    variant = ("auto" if product in MASONRY_FORMS or product == MARBLE_PRODUCT
                               else self.params.get("classical_order", "Random").lower())
                    design_params = {}
                result = build(BuildRequest(
                    product=product, width=width, height=height, depth=depth, seed=seed or 0,
                    variant=self.params.get("historic_style") or "auto",
                    detail=self.params.get("form_detail", "standard"),
                    bays=self.params.get("form_bays") if "bays" in found[1].controls else None,
                    levels=self.params.get("form_levels") if "levels" in found[1].controls else None,
                    age=age,
                )) if found else build(BuildRequest(
                    product=product, width=width, height=height, depth=depth,
                    seed=seed or 0,
                    style="timber" if product == TIMBER_PRODUCT else "persian" if product == PERSIAN_PRODUCT else "classical",
                    variant=variant,
                    detail=self.params.get("form_detail", "standard"),
                    bays=self.params.get("form_bays", 5),
                    levels=self.params.get("form_levels", 2) if product in {"classical-aqueduct", MARBLE_PRODUCT, TIMBER_PRODUCT} else None,
                    parameters=design_params,
                    age=age,
                ))
                if result.quality.status == "fail":
                    raise ValueError("; ".join(c.message for c in result.quality.checks if c.status == "fail"))
                rules = result.document.metadata.get("classical_rules") or {}
                if product == CATHEDRAL_PRODUCT:
                    rules = {"order": result.document.metadata["cathedral_rules"]["variant"],
                             "typology": "Gothic cathedral", "bays": result.document.metadata["cathedral_rules"]["nave_bays"]}
                if product == MARBLE_PRODUCT:
                    rules = {"order": None, "typology": "marble-run", "bays": result.document.request.bays}
                if product == TIMBER_PRODUCT:
                    house = result.document.request
                    rules = {"order": house.variant, "typology": f"Half-timbered house / {house.parameters['finish']}", "bays": house.bays}
                if product == PERSIAN_PRODUCT:
                    court = result.document.request
                    rules = {"order": court.variant, "typology": f"Iwan court / {court.parameters['finish']}", "bays": None}
                if found:
                    rules = {"order": result.document.request.variant,
                             "typology": result.document.metadata[f"{found[0].ARCHITECT.id}_rules"]["typology"],
                             "bays": result.document.request.bays}
                self.generation_finished.emit(result.scene, "", {
                    "order": rules.get("order"),
                    "variant_label": ("plan" if product in {TIMBER_PRODUCT, PERSIAN_PRODUCT} else
                                      "style" if product == CATHEDRAL_PRODUCT or found else "order"),
                    "typology": rules.get("typology"),
                    "bays": rules.get("bays"),
                    "detail": result.document.request.detail,
                    "quality": result.quality.status,
                    "checks": dict(result.quality.counts),
                    "marble_run": (result.document.metadata or {}).get("marble_run"),
                })
                return

            from core.architecture import BuildRequest, build
            temple_parameters = {key: value for key, value in self.params.items()
                if key.startswith("classical_") and key not in {"classical_form", "classical_order"}
                and value is not None}
            result = build(BuildRequest(product="classical-building", width=width,
                height=height, depth=depth, seed=seed or 0,
                variant=self.params.get("classical_order", "Random").lower(),
                detail=self.params.get("form_detail", "standard"), parameters=temple_parameters,
                age=age))
            if result.quality.status != "pass":
                raise ValueError("; ".join(c.message for c in result.quality.checks if c.status == "fail"))
            metrics = result.document.metadata["design_metrics"]
            self.generation_finished.emit(result.scene, "", {"order": metrics["order"],
                "typology": "temple", "detail": result.document.request.detail,
                "quality": result.quality.status, "checks": dict(result.quality.counts)})
            return

        except Exception as exc:
            error_message = str(exc)
            new_scene = None

        self.generation_finished.emit(new_scene, error_message, None)


class MainWindow(QMainWindow):
    """Main application window for the KISS GUI baseline."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("BlockN — Classical Workbench")
        self.setGeometry(100, 100, 1280, 820)
        self.setObjectName("MainWindow")

        self.scene = Scene()
        self._selected_color: Optional[ColorTuple] = None
        self._last_optimized_meshes: Optional[List[trimesh.Trimesh]] = None
        self._last_optimize_report: Optional[Dict[str, Any]] = None
        self._generation_thread: Optional[GenerationThread] = None
        self._panel_animations: List[QPropertyAnimation] = []
        self._entry_animation_started = False
        self._dock_widgets: List[QDockWidget] = []
        self._last_requested_seed: Optional[int] = None

        self._support_enabled = True
        self._strict_overlap_enabled = True

        central = QWidget()
        central.setObjectName("CentralCanvas")
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.viewport = ViewportWidget()
        self.viewport.block_picked.connect(self.handle_block_picked)
        layout.addWidget(self.viewport)

        self._create_panels()
        self._apply_visual_theme()

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready", 2000)

        self._bootstrap_scene_defaults()
        self.update_viewport()

        QTimer.singleShot(140, self._run_entry_animation)

    def _create_scroll_host(self, panel: QWidget) -> QScrollArea:
        scroll_area = QScrollArea(self)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(panel)
        return scroll_area

    def _create_panels(self):
        self.generation_panel = GenerationPanel()
        self.generation_panel.setMinimumWidth(360)
        self.generation_panel.generate_triggered.connect(self.handle_generate_request)
        self.generation_panel.support_toggled.connect(self.handle_support_toggled)
        self.generation_panel.overlap_toggled.connect(self.handle_overlap_toggled)
        self.generation_panel.cathedral_cutaway.toggled.connect(self.update_viewport)

        generation_dock = QDockWidget("<> Generation //", self)
        generation_dock.setObjectName("GenerationDock")
        generation_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        generation_dock.setMinimumWidth(390)
        generation_dock.setWidget(self._create_scroll_host(self.generation_panel))
        generation_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, generation_dock)
        self._dock_widgets.append(generation_dock)

        self.scene_export_panel = SceneExportPanel()
        self.scene_export_panel.setMinimumWidth(260)
        self.scene_export_panel.optimize_triggered.connect(self.handle_optimize_request)
        self.scene_export_panel.export_glb_triggered.connect(self.handle_export_glb_request)
        self.scene_export_panel.export_json_triggered.connect(self.handle_export_json_request)
        self.scene_export_panel.export_snapshot_triggered.connect(self.handle_export_snapshot_request)

        scene_dock = QDockWidget("[] Scene / Export <>", self)
        scene_dock.setObjectName("SceneDock")
        scene_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        scene_dock.setWidget(self._create_scroll_host(self.scene_export_panel))
        scene_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, scene_dock)
        self.splitDockWidget(generation_dock, scene_dock, Qt.Vertical)
        self._dock_widgets.append(scene_dock)

        self.color_palette_panel = ColorPalettePanel()
        self.color_palette_panel.setMinimumWidth(260)
        self.color_palette_panel.color_selected.connect(self.handle_color_selected)

        color_dock = QDockWidget("++ Color Palette []", self)
        color_dock.setObjectName("ColorDock")
        color_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        color_dock.setWidget(self._create_scroll_host(self.color_palette_panel))
        color_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, color_dock)
        self.splitDockWidget(scene_dock, color_dock, Qt.Vertical)
        self._dock_widgets.append(color_dock)

        self.outliner_panel = OutlinerPanel()
        self.outliner_panel.setMinimumWidth(240)
        outliner_dock = QDockWidget("// Outliner []", self)
        outliner_dock.setObjectName("OutlinerDock")
        outliner_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        outliner_dock.setWidget(self._create_scroll_host(self.outliner_panel))
        outliner_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, outliner_dock)
        self._dock_widgets.append(outliner_dock)

        self.resizeDocks([generation_dock], [420], Qt.Horizontal)
        self.resizeDocks([generation_dock, scene_dock, color_dock], [520, 150, 150], Qt.Vertical)

    def _apply_visual_theme(self):
        self.setStyleSheet(PHOSPHOR_STYLESHEET)
        self.setDockNestingEnabled(True)
        self.setTabPosition(Qt.LeftDockWidgetArea, QTabWidget.North)
        self.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)

    def _run_entry_animation(self):
        if self._entry_animation_started:
            return
        self._entry_animation_started = True
        self._panel_animations.clear()

        for index, dock in enumerate(self._dock_widgets):
            target = dock.widget() if dock.widget() is not None else dock
            effect = QGraphicsOpacityEffect(target)
            effect.setOpacity(0.0)
            target.setGraphicsEffect(effect)

            animation = QPropertyAnimation(effect, b"opacity", self)
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
            animation.setDuration(280 + (index * 80))
            animation.setEasingCurve(QEasingCurve.OutCubic)
            self._panel_animations.append(animation)
            QTimer.singleShot(index * 90, animation.start)

    def _bootstrap_scene_defaults(self):
        collision_enabled = self.scene.collision_manager is not None
        self.generation_panel.apply_collision_defaults(collision_enabled)

        self._support_enabled = self.generation_panel.support_check_cb.isChecked()
        self._strict_overlap_enabled = self.generation_panel.overlap_check_cb.isChecked()

        if not collision_enabled:
            self.statusBar().showMessage(
                "Collision backend unavailable. Using collision-free mode (support defaults OFF).",
                5000,
            )

        self._sync_scene_runtime_settings()

    def _sync_scene_runtime_settings(self):
        self.scene.support_check_mode = "GRID" if self._support_enabled else "OFF"
        if self.scene.collision_manager is None:
            self.scene.enable_strict_overlap_prevention = False
        else:
            self.scene.enable_strict_overlap_prevention = self._strict_overlap_enabled

    def handle_color_selected(self, color: ColorTuple):
        self._selected_color = color
        self.statusBar().showMessage(f"Color selected: {color}", 2000)

    def handle_block_picked(self, block_id: Optional[int]):
        if block_id is None:
            self.statusBar().showMessage("Selection cleared.", 1500)
            return

        if self._selected_color is None:
            self.statusBar().showMessage(f"Block {block_id} selected.", 1500)
            return

        block = self.scene.get_block_by_id(block_id)
        if block is None:
            self.statusBar().showMessage(f"Block {block_id} not found.", 3000)
            return

        block.update_color(self._selected_color)
        self.viewport.update_block_visuals(block)
        self.statusBar().showMessage(f"Applied color to block {block_id}.", 2000)

    def handle_support_toggled(self, checked: bool):
        self._support_enabled = checked
        self._sync_scene_runtime_settings()
        status = "ON" if checked else "OFF"
        self.statusBar().showMessage(f"Support check: {status}", 2000)

    def handle_overlap_toggled(self, checked: bool):
        if checked and self.scene.collision_manager is None:
            self.generation_panel.overlap_check_cb.blockSignals(True)
            self.generation_panel.overlap_check_cb.setChecked(False)
            self.generation_panel.overlap_check_cb.blockSignals(False)
            self._strict_overlap_enabled = False
            self.statusBar().showMessage(
                "Strict overlap check unavailable without collision backend.",
                4000,
            )
            return

        self._strict_overlap_enabled = checked
        self._sync_scene_runtime_settings()
        status = "ON" if checked else "OFF"
        self.statusBar().showMessage(f"Strict overlap: {status}", 2000)

    def handle_generate_request(
        self,
        width: int,
        height: int,
        depth: int,
        seed: Optional[int],
        classical_order: str,
        classical_preset: str,
        classical_front_columns: Optional[int],
        classical_side_columns: Optional[int],
        classical_pediment_slope: float,
        classical_entablature_depth: float,
        classical_side_roof_overhang: float,
        classical_foundation_apron: bool,
        classical_wrap_min_front_span: Optional[float],
        classical_wrap_min_side_span: Optional[float],
        classical_wrap_front_coverage: Optional[float],
        classical_add_arch_portal: bool,
        classical_add_dome_cap: bool,
    ):
        if self._generation_thread and self._generation_thread.isRunning():
            self.statusBar().showMessage("Generation already running.", 2000)
            return

        self._last_optimized_meshes = None
        self._last_optimize_report = None

        params = {
            "width": width,
            "height": height,
            "depth": depth,
            "seed": seed,
            "classical_order": classical_order,
            "classical_form": self.generation_panel.classical_form_combo.currentData(),
            "form_bays": self.generation_panel.form_bays.value(),
            "form_levels": self.generation_panel.form_levels.value(),
            "form_detail": self.generation_panel.form_detail.currentText(),
            "cathedral_style": self.generation_panel.cathedral_style.currentData(),
            "cathedral_parameters": self.generation_panel.cathedral_parameters(),
            "historic_style": self.generation_panel.historic_style.currentData(),
            "timber_variant": self.generation_panel.timber_variant.currentData(),
            "timber_parameters": self.generation_panel.timber_parameters(),
            "persian_variant": self.generation_panel.persian_variant.currentData(),
            "persian_parameters": self.generation_panel.persian_parameters(),
            "settlement_variant": self.generation_panel.settlement_variant.currentData(),
            "classical_preset": classical_preset,
            "classical_front_columns": classical_front_columns,
            "classical_side_columns": classical_side_columns,
            "classical_pediment_slope": classical_pediment_slope,
            "classical_entablature_depth": classical_entablature_depth,
            "classical_side_roof_overhang": classical_side_roof_overhang,
            "classical_foundation_apron": classical_foundation_apron,
            "classical_wrap_min_front_span": classical_wrap_min_front_span,
            "classical_wrap_min_side_span": classical_wrap_min_side_span,
            "classical_wrap_front_coverage": classical_wrap_front_coverage,
            "classical_add_arch_portal": classical_add_arch_portal,
            "classical_add_dome_cap": classical_add_dome_cap,
            "age": float(self.generation_panel.age_spinbox.value()),
            "support_enabled": self._support_enabled,
            "strict_overlap": self._strict_overlap_enabled,
        }
        self._last_requested_seed = seed
        self.generation_panel.set_generation_pending(
            seed=seed,
            order_hint=classical_order,
            preset=classical_preset,
        )

        self._set_generation_controls_enabled(False)

        self._generation_thread = GenerationThread(params)
        self._generation_thread.generation_finished.connect(self.handle_generation_finished)
        self._generation_thread.finished.connect(self._on_generation_thread_finished)
        self._generation_thread.start()

        self.statusBar().showMessage(
            (
                f"Generating Classical: order={classical_order} preset={classical_preset} "
                f"size={width}x{height}x{depth} (seed={seed})"
                f" apron={'on' if classical_foundation_apron else 'off'}"
                f" arch={'on' if classical_add_arch_portal else 'off'}"
                f" dome={'on' if classical_add_dome_cap else 'off'}"
            ),
            3500,
        )

    def handle_generation_finished(
        self,
        new_scene: Optional[Scene],
        error_message: str,
        build_metadata: Optional[Dict[str, Any]] = None,
    ):
        if error_message or new_scene is None:
            self.statusBar().showMessage(f"Generation failed: {error_message}", 5000)
            return

        self.scene = new_scene
        self._sync_scene_runtime_settings()
        self.update_viewport()
        self.outliner_panel.update_list(self.scene.blocks)
        metrics = getattr(self.scene, "last_classical_metrics", None)
        if isinstance(metrics, dict):
            self.generation_panel.set_generation_metadata(
                seed=self._last_requested_seed,
                order=metrics.get("order"),
                preset=metrics.get("preset"),
                front_columns=metrics.get("front_columns"),
                side_columns=metrics.get("side_columns"),
            )
            self.statusBar().showMessage(
                (
                    "Generation complete:"
                    f" {len(self.scene.blocks)} blocks | order={metrics.get('order')}"
                    f" preset={metrics.get('preset')}"
                    f" columns={metrics.get('front_columns')}x{metrics.get('side_columns')}"
                    f" arch={metrics.get('feature_shapes', {}).get('arch_count', 0)}"
                    f" dome={metrics.get('feature_shapes', {}).get('dome_count', 0)}"
                ),
                4500,
            )
        elif isinstance(build_metadata, dict):
            self.generation_panel.set_typology_metadata(self._last_requested_seed, build_metadata)
            tally = " ".join(
                f"{name}={count}" for name, count in sorted((build_metadata.get("checks") or {}).items())
            )
            self.statusBar().showMessage(
                (
                    f"Generation complete: {len(self.scene.blocks)} blocks"
                    f" | typology={build_metadata.get('typology') or 'n/a'}"
                    f" {build_metadata.get('variant_label', 'order')}={build_metadata.get('order') or 'masonry (no order)'}"
                    + (f" bays={build_metadata.get('bays')}" if build_metadata.get('bays') is not None else "")
                    + f" detail={build_metadata.get('detail')}"
                    f" quality={build_metadata.get('quality')} ({tally})"
                ),
                4500,
            )
            ride = build_metadata.get("marble_run") or (getattr(self.scene, "generation_metadata", {}) or {}).get("marble_run")
            if ride:
                self.viewport.start_marble_run(ride)
                self.statusBar().showMessage(
                    f"Marble run looping {ride.get('length', 0):.1f} units of canal. Chase camera is live.",
                    5000,
                )
        else:
            self.generation_panel.set_generation_metadata(
                seed=self._last_requested_seed, order=None, preset=None,
                front_columns=None, side_columns=None,
            )
            self.statusBar().showMessage(f"Generation complete: {len(self.scene.blocks)} blocks; quality passed.", 3000)

    def _on_generation_thread_finished(self):
        self._generation_thread = None
        self._set_generation_controls_enabled(True)

    def _set_generation_controls_enabled(self, enabled: bool):
        self.generation_panel.generate_button.setEnabled(enabled)
        self.scene_export_panel.optimize_button.setEnabled(enabled)

    def update_viewport(self):
        blocks = self.scene.blocks if self.scene else []
        if self.generation_panel.cathedral_cutaway.isChecked():
            removable = {"nave_roof", "aisle_roof", "transept_roof", "apse_roof", "roof_seam", "ridge", "glazing"}
            opened = {"castle", "egyptian", "mesoamerican", "pagoda", "chinese", "dravidian", "stave"}
            blocks = [b for b in blocks if not (
                (b.parameters.get("architect") == "cathedral" and b.parameters.get("component_role") in removable)
                or (b.parameters.get("architect") in opened and b.parameters.get("kind") == "roof"))]
        self.viewport.display_meshes(blocks)

    def handle_optimize_request(self):
        if not self.scene.blocks:
            self.statusBar().showMessage("Scene is empty. Nothing to optimize.", 3000)
            return

        try:
            unioned_meshes, optimize_report = optimize_blocks(self.scene.blocks, return_report=True)
        except Exception as exc:
            QMessageBox.critical(self, "Optimization Error", str(exc))
            self.statusBar().showMessage(f"Optimization failed: {exc}", 5000)
            return

        if not unioned_meshes:
            self.statusBar().showMessage("Optimization produced no meshes.", 3000)
            return

        self._last_optimized_meshes = unioned_meshes
        self._last_optimize_report = optimize_report
        self.viewport.display_optimized_result(unioned_meshes)
        backend_status = optimize_report.get("backend_status", {}) if isinstance(optimize_report, dict) else {}
        backend_engines = backend_status.get("engines", [])
        backend_text = ",".join(backend_engines) if backend_engines else "none"
        attempted = int(optimize_report.get("groups_union_attempted", 0))
        succeeded = int(optimize_report.get("groups_union_succeeded", 0))
        fallback = int(optimize_report.get("groups_fallback", 0))
        fallback_reasons = optimize_report.get("fallback_reasons", {})
        fallback_text = f" reasons={fallback_reasons}" if fallback else ""
        self.statusBar().showMessage(
            (
                f"Optimization complete: {len(unioned_meshes)} meshes | backend={backend_text} "
                f"| union={succeeded}/{attempted} | fallback_groups={fallback}{fallback_text}"
            ),
            5000 if fallback else 3500,
        )

    def _get_meshes_for_export(self) -> List[trimesh.Trimesh]:
        if self._last_optimized_meshes:
            return self._last_optimized_meshes
        return self.scene.get_all_meshes()

    def handle_export_glb_request(self):
        meshes = self._get_meshes_for_export()
        if not meshes:
            self.statusBar().showMessage("No meshes available to export.", 4000)
            return

        default_path = str(Path("builds") / "scene.glb")
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export GLB",
            default_path,
            "GLB Files (*.glb);;All Files (*)",
        )
        if not file_name:
            self.statusBar().showMessage("Export cancelled.", 1500)
            return

        try:
            if not file_name.lower().endswith(".glb"):
                file_name += ".glb"
            source = trimesh.Scene(meshes) if self._last_optimized_meshes else self.scene
            export_mesh_artifact(source, file_name, glb_materials=self.scene_export_panel.glb_materials.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            self.statusBar().showMessage(f"GLB export failed: {exc}", 5000)
            return

        self.statusBar().showMessage(f"GLB exported: {file_name}", 3000)

    def handle_export_json_request(self):
        default_path = str(Path("builds") / "scene.json")
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_name:
            self.statusBar().showMessage("Export cancelled.", 1500)
            return

        try:
            save_scene_to_json(self.scene, file_name)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            self.statusBar().showMessage(f"JSON export failed: {exc}", 5000)
            return

        self.statusBar().showMessage(f"JSON exported: {file_name}", 3000)

    def handle_export_snapshot_request(self):
        snapshots_dir = Path("snapshots")
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = str(snapshots_dir / f"snapshot_{timestamp}.png")
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Snapshot",
            default_path,
            "PNG Files (*.png);;All Files (*)",
        )
        if not file_name:
            self.statusBar().showMessage("Snapshot export cancelled.", 1500)
            return

        try:
            self.viewport.save_snapshot(file_name)
        except Exception as exc:
            QMessageBox.critical(self, "Snapshot Error", str(exc))
            self.statusBar().showMessage(f"Snapshot export failed: {exc}", 5000)
            return

        self.statusBar().showMessage(f"Snapshot exported: {file_name}", 3000)

    def closeEvent(self, event):
        if self._generation_thread and self._generation_thread.isRunning():
            self._generation_thread.quit()
            if not self._generation_thread.wait(3000):
                self._generation_thread.terminate()
                self._generation_thread.wait()

        self.viewport.close()
        super().closeEvent(event)
