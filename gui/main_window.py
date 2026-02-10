from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import trimesh
from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QGraphicsOpacityEffect,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.generators.classical import generate_classical_building
from core.geometry import optimize_blocks
from core.io import export_meshes_to_glb, save_scene_to_json
from core.scene import ColorTuple, Scene
from .panels import (
    ColorPalettePanel,
    GenerationPanel,
    OutlinerPanel,
    SceneExportPanel,
)
from .viewport import ViewportWidget


MEMPHIS_SUNSET_STYLESHEET = """
QMainWindow {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #ffd07a,
        stop: 0.20 #f39b63,
        stop: 0.48 #e56f5c,
        stop: 0.70 #5a5d89,
        stop: 1 #2d4466
    );
    color: #fff4dc;
}

QWidget {
    color: #fff4dc;
    font-family: "Avenir Next", "Gill Sans MT", "Trebuchet MS", sans-serif;
    font-size: 12px;
}

QDockWidget {
    border: 2px solid rgba(255, 244, 220, 0.25);
    border-radius: 12px;
    margin-top: 22px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 rgba(20, 33, 50, 0.95),
        stop: 0.28 rgba(26, 44, 67, 0.95),
        stop: 0.29 rgba(37, 63, 86, 0.92),
        stop: 0.33 rgba(24, 39, 57, 0.95),
        stop: 0.62 rgba(19, 33, 51, 0.95),
        stop: 0.63 rgba(45, 71, 91, 0.90),
        stop: 0.67 rgba(21, 36, 54, 0.95),
        stop: 1 rgba(16, 27, 41, 0.95)
    );
}

QDockWidget::title {
    text-align: left;
    background: rgba(255, 129, 84, 0.88);
    color: #171f30;
    padding-left: 12px;
    padding-right: 12px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-family: "Futura", "Avenir Next", "Gill Sans MT", sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}

QDockWidget > QWidget {
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
    border: 1px solid rgba(255, 244, 220, 0.16);
}

QDockWidget#GenerationDock > QWidget {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 rgba(23, 38, 56, 0.95),
        stop: 0.18 rgba(35, 55, 76, 0.93),
        stop: 0.19 rgba(24, 39, 58, 0.95),
        stop: 0.39 rgba(24, 39, 58, 0.95),
        stop: 0.40 rgba(45, 67, 88, 0.92),
        stop: 0.44 rgba(22, 37, 56, 0.95),
        stop: 1 rgba(19, 32, 49, 0.95)
    );
}

QDockWidget#SceneDock > QWidget {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 rgba(20, 34, 52, 0.95),
        stop: 0.22 rgba(28, 52, 74, 0.92),
        stop: 0.23 rgba(21, 35, 53, 0.95),
        stop: 0.46 rgba(21, 35, 53, 0.95),
        stop: 0.47 rgba(42, 67, 88, 0.90),
        stop: 0.52 rgba(22, 36, 54, 0.95),
        stop: 1 rgba(17, 29, 45, 0.95)
    );
}

QDockWidget#ColorDock > QWidget {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 rgba(23, 38, 56, 0.95),
        stop: 0.30 rgba(22, 37, 56, 0.95),
        stop: 0.31 rgba(45, 69, 91, 0.92),
        stop: 0.36 rgba(22, 37, 56, 0.95),
        stop: 0.68 rgba(20, 34, 52, 0.95),
        stop: 0.69 rgba(40, 63, 83, 0.90),
        stop: 0.73 rgba(21, 35, 53, 0.95),
        stop: 1 rgba(17, 29, 44, 0.95)
    );
}

QDockWidget#OutlinerDock > QWidget {
    background: qlineargradient(
        x1: 1, y1: 0, x2: 0, y2: 1,
        stop: 0 rgba(22, 37, 55, 0.95),
        stop: 0.24 rgba(32, 52, 73, 0.92),
        stop: 0.25 rgba(21, 35, 53, 0.95),
        stop: 0.50 rgba(21, 35, 53, 0.95),
        stop: 0.51 rgba(43, 67, 88, 0.90),
        stop: 0.56 rgba(21, 35, 53, 0.95),
        stop: 1 rgba(16, 28, 43, 0.95)
    );
}

QGroupBox {
    background: rgba(16, 25, 40, 0.74);
    border: 1px solid rgba(255, 244, 220, 0.30);
    border-radius: 12px;
    margin-top: 14px;
    padding: 16px 10px 10px 10px;
    font-family: "Futura", "Avenir Next", "Gill Sans MT", sans-serif;
    font-size: 11px;
    font-weight: 600;
    color: #ffe7bc;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 4px;
    padding: 1px 8px;
    background: rgba(47, 125, 128, 0.85);
    border-radius: 8px;
    color: #fff6e3;
}

QLabel#PanelSubtitle {
    font-family: "Palatino Linotype", "Book Antiqua", serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #ffd67f;
    padding-bottom: 2px;
}

QPushButton {
    border: 1px solid rgba(255, 244, 220, 0.55);
    border-radius: 10px;
    padding: 7px 10px;
    background: rgba(215, 90, 67, 0.86);
    color: #fff8e8;
    font-family: "Futura", "Avenir Next", "Gill Sans MT", sans-serif;
    font-size: 11px;
    font-weight: 600;
}

QPushButton:hover {
    background: rgba(242, 132, 76, 0.95);
}

QPushButton:pressed {
    background: rgba(163, 62, 48, 0.95);
}

QPushButton#PrimaryActionButton {
    min-height: 36px;
    border-width: 2px;
    border-color: rgba(255, 244, 220, 0.88);
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #f56b52,
        stop: 1 #f2b053
    );
    color: #172034;
    font-size: 13px;
    font-weight: 700;
}

QPushButton#PrimaryActionButton:hover {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #ff8868,
        stop: 1 #ffd177
    );
}

QPushButton#SwatchButton {
    min-width: 32px;
    min-height: 32px;
    max-width: 32px;
    max-height: 32px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
    background: rgba(12, 21, 34, 0.80);
    border: 1px solid rgba(255, 244, 220, 0.38);
    border-radius: 8px;
    padding: 5px 6px;
    color: #fff6e5;
    selection-background-color: rgba(255, 179, 92, 0.95);
    selection-color: #1f2838;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus {
    border: 1px solid rgba(255, 196, 110, 0.92);
    background: rgba(16, 28, 43, 0.94);
}

QComboBox::drop-down {
    width: 22px;
    border: 0;
    border-left: 1px solid rgba(255, 244, 220, 0.30);
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 rgba(47, 125, 128, 0.75),
        stop: 1 rgba(28, 78, 90, 0.78)
    );
}

QComboBox::down-arrow {
    width: 0px;
    height: 0px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 7px solid rgba(255, 230, 173, 0.96);
    margin-right: 3px;
}

QComboBox:on {
    background: rgba(18, 30, 45, 0.96);
}

QComboBox QAbstractItemView {
    background: rgba(14, 24, 38, 0.97);
    border: 1px solid rgba(255, 209, 130, 0.55);
    border-radius: 8px;
    color: #fff6e5;
    selection-background-color: rgba(246, 144, 83, 0.95);
    selection-color: #1d2636;
    padding: 4px;
    outline: 0;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid rgba(255, 244, 220, 0.30);
    border-bottom: 1px solid rgba(255, 244, 220, 0.20);
    background: rgba(39, 92, 98, 0.72);
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid rgba(255, 244, 220, 0.30);
    background: rgba(33, 74, 90, 0.75);
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 0px;
    height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 6px solid rgba(255, 230, 173, 0.96);
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 0px;
    height: 0px;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid rgba(255, 230, 173, 0.96);
}

QListWidget::item {
    padding: 5px 6px;
    border-radius: 6px;
}

QListWidget::item:selected {
    background: rgba(242, 160, 86, 0.92);
    color: #152136;
}

QListWidget::item:hover {
    background: rgba(64, 112, 124, 0.56);
}

QTabBar::tab {
    background: rgba(18, 30, 46, 0.92);
    color: #ffe8bf;
    border: 1px solid rgba(255, 214, 148, 0.36);
    border-bottom: 0;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 5px 10px;
    margin-right: 4px;
    font-family: "Futura", "Avenir Next", "Gill Sans MT", sans-serif;
    font-size: 10px;
    letter-spacing: 1px;
}

QTabBar::tab:selected {
    background: rgba(245, 145, 84, 0.94);
    color: #172034;
    border-color: rgba(255, 230, 173, 0.92);
}

QTabBar::tab:hover:!selected {
    background: rgba(40, 74, 97, 0.95);
}

QDockWidget::close-button, QDockWidget::float-button {
    border: 1px solid rgba(255, 244, 220, 0.40);
    border-radius: 6px;
    background: rgba(24, 39, 57, 0.82);
    padding: 1px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background: rgba(244, 140, 84, 0.92);
}

QScrollBar:vertical {
    width: 12px;
    background: rgba(9, 17, 30, 0.72);
    margin: 2px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    min-height: 24px;
    border-radius: 6px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 rgba(70, 128, 132, 0.86),
        stop: 1 rgba(241, 141, 88, 0.90)
    );
}

QScrollBar:horizontal {
    height: 12px;
    background: rgba(9, 17, 30, 0.72);
    margin: 2px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    min-width: 24px;
    border-radius: 6px;
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 rgba(70, 128, 132, 0.86),
        stop: 1 rgba(241, 141, 88, 0.90)
    );
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    border: none;
    background: transparent;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 4px;
    border: 1px solid rgba(255, 244, 220, 0.65);
    background: rgba(10, 19, 31, 0.85);
}

QCheckBox::indicator:checked {
    background: rgba(250, 180, 88, 0.98);
}

QCheckBox::indicator:hover {
    border-color: rgba(255, 208, 135, 0.95);
}

QPushButton:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QListWidget:disabled {
    color: rgba(255, 240, 209, 0.50);
    border-color: rgba(255, 244, 220, 0.22);
    background: rgba(9, 16, 27, 0.62);
}

QStatusBar {
    background: rgba(9, 16, 28, 0.90);
    color: #ffe7bc;
    border-top: 1px solid rgba(255, 244, 220, 0.25);
}

QStatusBar::item {
    border: none;
}

QToolTip {
    background: rgba(12, 22, 36, 0.95);
    color: #fff2d2;
    border: 1px solid rgba(255, 207, 126, 0.85);
    padding: 4px 6px;
}
"""


class GenerationThread(QThread):
    """Runs generation in the background to keep UI responsive."""

    generation_finished = pyqtSignal(object, str)

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

            generate_classical_building(
                new_scene,
                width=width,
                depth=depth,
                podium_height_actual=max(1, height // 3),
                seed=seed,
                flags={"foundation_apron": bool(self.params.get("classical_foundation_apron", False))},
                order_override=self.params.get("classical_order", "Random"),
                preset_override=self.params.get("classical_preset", "canonical"),
                front_column_override=self.params.get("classical_front_columns"),
                side_column_override=self.params.get("classical_side_columns"),
                pediment_slope_multiplier=float(self.params.get("classical_pediment_slope", 1.0)),
                entablature_depth_multiplier=float(
                    self.params.get("classical_entablature_depth", 1.0)
                ),
                side_roof_overhang_multiplier=float(
                    self.params.get("classical_side_roof_overhang", 1.0)
                ),
                wrap_min_podium_span_x_override=self.params.get("classical_wrap_min_front_span"),
                wrap_min_podium_span_z_override=self.params.get("classical_wrap_min_side_span"),
                wrap_front_coverage_ratio_override=self.params.get("classical_wrap_front_coverage"),
                clear_scene=True,
            )
        except Exception as exc:
            error_message = str(exc)
            new_scene = None

        self.generation_finished.emit(new_scene, error_message)


class MainWindow(QMainWindow):
    """Main application window for the KISS GUI baseline."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("LegoGen")
        self.setGeometry(100, 100, 1280, 820)
        self.setObjectName("MainWindow")

        self.scene = Scene()
        self._selected_color: Optional[ColorTuple] = None
        self._last_optimized_meshes: Optional[List[trimesh.Trimesh]] = None
        self._generation_thread: Optional[GenerationThread] = None
        self._panel_animations: List[QPropertyAnimation] = []
        self._entry_animation_started = False
        self._dock_widgets: List[QDockWidget] = []

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

    def _create_panels(self):
        self.generation_panel = GenerationPanel()
        self.generation_panel.generate_triggered.connect(self.handle_generate_request)
        self.generation_panel.support_toggled.connect(self.handle_support_toggled)
        self.generation_panel.overlap_toggled.connect(self.handle_overlap_toggled)

        generation_dock = QDockWidget("<> Generation //", self)
        generation_dock.setObjectName("GenerationDock")
        generation_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        generation_dock.setWidget(self.generation_panel)
        generation_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, generation_dock)
        self._dock_widgets.append(generation_dock)

        self.scene_export_panel = SceneExportPanel()
        self.scene_export_panel.optimize_triggered.connect(self.handle_optimize_request)
        self.scene_export_panel.export_glb_triggered.connect(self.handle_export_glb_request)
        self.scene_export_panel.export_json_triggered.connect(self.handle_export_json_request)
        self.scene_export_panel.export_snapshot_triggered.connect(self.handle_export_snapshot_request)

        scene_dock = QDockWidget("[] Scene / Export <>", self)
        scene_dock.setObjectName("SceneDock")
        scene_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        scene_dock.setWidget(self.scene_export_panel)
        scene_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, scene_dock)
        self.splitDockWidget(generation_dock, scene_dock, Qt.Vertical)
        self._dock_widgets.append(scene_dock)

        self.color_palette_panel = ColorPalettePanel()
        self.color_palette_panel.color_selected.connect(self.handle_color_selected)

        color_dock = QDockWidget("++ Color Palette []", self)
        color_dock.setObjectName("ColorDock")
        color_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        color_dock.setWidget(self.color_palette_panel)
        color_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, color_dock)
        self.splitDockWidget(scene_dock, color_dock, Qt.Vertical)
        self._dock_widgets.append(color_dock)

        self.outliner_panel = OutlinerPanel()
        outliner_dock = QDockWidget("// Outliner []", self)
        outliner_dock.setObjectName("OutlinerDock")
        outliner_dock.setFeatures(
            QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        outliner_dock.setWidget(self.outliner_panel)
        outliner_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, outliner_dock)
        self._dock_widgets.append(outliner_dock)

    def _apply_visual_theme(self):
        self.setStyleSheet(MEMPHIS_SUNSET_STYLESHEET)
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
    ):
        if self._generation_thread and self._generation_thread.isRunning():
            self.statusBar().showMessage("Generation already running.", 2000)
            return

        self._last_optimized_meshes = None

        params = {
            "width": width,
            "height": height,
            "depth": depth,
            "seed": seed,
            "classical_order": classical_order,
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
            "support_enabled": self._support_enabled,
            "strict_overlap": self._strict_overlap_enabled,
        }

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
            ),
            3500,
        )

    def handle_generation_finished(self, new_scene: Optional[Scene], error_message: str):
        if error_message or new_scene is None:
            self.statusBar().showMessage(f"Generation failed: {error_message}", 5000)
            return

        self.scene = new_scene
        self._sync_scene_runtime_settings()
        self.update_viewport()
        self.outliner_panel.update_list(self.scene.blocks)
        metrics = getattr(self.scene, "last_classical_metrics", None)
        if isinstance(metrics, dict):
            self.statusBar().showMessage(
                (
                    "Generation complete:"
                    f" {len(self.scene.blocks)} blocks | order={metrics.get('order')}"
                    f" preset={metrics.get('preset')}"
                    f" columns={metrics.get('front_columns')}x{metrics.get('side_columns')}"
                ),
                4500,
            )
        else:
            self.statusBar().showMessage(f"Generation complete: {len(self.scene.blocks)} blocks.", 3000)

    def _on_generation_thread_finished(self):
        self._generation_thread = None
        self._set_generation_controls_enabled(True)

    def _set_generation_controls_enabled(self, enabled: bool):
        self.generation_panel.generate_button.setEnabled(enabled)
        self.scene_export_panel.optimize_button.setEnabled(enabled)

    def update_viewport(self):
        blocks = self.scene.blocks if self.scene else []
        self.viewport.display_meshes(blocks)

    def handle_optimize_request(self):
        if not self.scene.blocks:
            self.statusBar().showMessage("Scene is empty. Nothing to optimize.", 3000)
            return

        try:
            unioned_meshes = optimize_blocks(self.scene.blocks)
        except Exception as exc:
            QMessageBox.critical(self, "Optimization Error", str(exc))
            self.statusBar().showMessage(f"Optimization failed: {exc}", 5000)
            return

        if not unioned_meshes:
            self.statusBar().showMessage("Optimization produced no meshes.", 3000)
            return

        self._last_optimized_meshes = unioned_meshes
        self.viewport.display_optimized_result(unioned_meshes)
        self.statusBar().showMessage(f"Optimization complete: {len(unioned_meshes)} meshes.", 3000)

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
            export_meshes_to_glb(meshes, file_name)
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
