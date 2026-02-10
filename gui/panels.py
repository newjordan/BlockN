from __future__ import annotations

import functools
import random
from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.materials import DEFAULT_COLORS, ColorTuple
from core.scene import Block


class GenerationPanel(QWidget):
    """Minimal generation controls for the KISS GUI."""

    generate_triggered = pyqtSignal(
        int,
        int,
        int,
        object,
        str,
        str,
        object,
        object,
        float,
        float,
        float,
        bool,
        object,
        object,
        object,
    )
    support_toggled = pyqtSignal(bool)
    overlap_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("GenerationPanel")

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setSpacing(10)

        gen_group = QGroupBox("Generation")
        gen_group.setObjectName("GenerationGroup")
        gen_layout = QVBoxLayout(gen_group)
        gen_layout.setSpacing(8)

        style_label = QLabel("Temple Composer")
        style_label.setObjectName("PanelSubtitle")
        gen_layout.addWidget(style_label)

        dim_layout = QHBoxLayout()
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 200)
        self.width_spinbox.setValue(5)

        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 200)
        self.height_spinbox.setValue(5)

        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setRange(1, 200)
        self.depth_spinbox.setValue(5)

        dim_layout.addWidget(QLabel("Width:"))
        dim_layout.addWidget(self.width_spinbox)
        dim_layout.addWidget(QLabel("Height:"))
        dim_layout.addWidget(self.height_spinbox)
        dim_layout.addWidget(QLabel("Depth:"))
        dim_layout.addWidget(self.depth_spinbox)
        gen_layout.addLayout(dim_layout)

        seed_layout = QHBoxLayout()
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Optional seed")
        self.seed_input.setValidator(QIntValidator())
        self.randomize_seed_button = QPushButton("Randomize")
        self.randomize_seed_button.clicked.connect(self._on_randomize_seed)
        seed_layout.addWidget(QLabel("Seed:"))
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.randomize_seed_button)
        gen_layout.addLayout(seed_layout)

        self.classical_order_row = QWidget()
        classical_order_layout = QHBoxLayout(self.classical_order_row)
        classical_order_layout.setContentsMargins(0, 0, 0, 0)
        self.classical_order_combo = QComboBox()
        self.classical_order_combo.addItems(["Random", "Doric", "Ionic"])
        classical_order_layout.addWidget(QLabel("Order:"))
        classical_order_layout.addWidget(self.classical_order_combo)
        gen_layout.addWidget(self.classical_order_row)

        self.classical_tuning_group = QGroupBox("Classical Tuning")
        self.classical_tuning_group.setObjectName("ClassicalTuningGroup")
        tuning_layout = QGridLayout(self.classical_tuning_group)

        self.classical_preset_combo = QComboBox()
        self.classical_preset_combo.addItems(["canonical", "compact", "monumental"])
        self.classical_preset_combo.setCurrentText("canonical")
        tuning_layout.addWidget(QLabel("Preset:"), 0, 0)
        tuning_layout.addWidget(self.classical_preset_combo, 0, 1, 1, 3)

        self.classical_front_columns_spinbox = QSpinBox()
        self.classical_front_columns_spinbox.setRange(0, 24)
        self.classical_front_columns_spinbox.setSingleStep(2)
        self.classical_front_columns_spinbox.setSpecialValueText("Auto")
        self.classical_front_columns_spinbox.setToolTip("Auto (0) or an even count >= 4")
        self.classical_front_columns_spinbox.valueChanged.connect(
            functools.partial(self._normalize_column_override_value, self.classical_front_columns_spinbox)
        )
        tuning_layout.addWidget(QLabel("Front Columns:"), 1, 0)
        tuning_layout.addWidget(self.classical_front_columns_spinbox, 1, 1)

        self.classical_side_columns_spinbox = QSpinBox()
        self.classical_side_columns_spinbox.setRange(0, 24)
        self.classical_side_columns_spinbox.setSingleStep(2)
        self.classical_side_columns_spinbox.setSpecialValueText("Auto")
        self.classical_side_columns_spinbox.setToolTip("Auto (0) or an even count >= 4")
        self.classical_side_columns_spinbox.valueChanged.connect(
            functools.partial(self._normalize_column_override_value, self.classical_side_columns_spinbox)
        )
        tuning_layout.addWidget(QLabel("Side Columns:"), 1, 2)
        tuning_layout.addWidget(self.classical_side_columns_spinbox, 1, 3)

        self.classical_pediment_slope_spinbox = QDoubleSpinBox()
        self.classical_pediment_slope_spinbox.setRange(0.35, 1.80)
        self.classical_pediment_slope_spinbox.setSingleStep(0.05)
        self.classical_pediment_slope_spinbox.setDecimals(2)
        self.classical_pediment_slope_spinbox.setValue(1.00)
        tuning_layout.addWidget(QLabel("Pediment Slope:"), 2, 0)
        tuning_layout.addWidget(self.classical_pediment_slope_spinbox, 2, 1)

        self.classical_entablature_depth_spinbox = QDoubleSpinBox()
        self.classical_entablature_depth_spinbox.setRange(0.65, 1.80)
        self.classical_entablature_depth_spinbox.setSingleStep(0.05)
        self.classical_entablature_depth_spinbox.setDecimals(2)
        self.classical_entablature_depth_spinbox.setValue(1.00)
        tuning_layout.addWidget(QLabel("Entablature Depth:"), 2, 2)
        tuning_layout.addWidget(self.classical_entablature_depth_spinbox, 2, 3)

        self.classical_side_roof_overhang_spinbox = QDoubleSpinBox()
        self.classical_side_roof_overhang_spinbox.setRange(0.65, 1.80)
        self.classical_side_roof_overhang_spinbox.setSingleStep(0.05)
        self.classical_side_roof_overhang_spinbox.setDecimals(2)
        self.classical_side_roof_overhang_spinbox.setValue(1.00)
        tuning_layout.addWidget(QLabel("Side Roof Overhang:"), 5, 2)
        tuning_layout.addWidget(self.classical_side_roof_overhang_spinbox, 5, 3)

        self.classical_foundation_apron_cb = QCheckBox("Foundation Apron")
        self.classical_foundation_apron_cb.setChecked(False)
        tuning_layout.addWidget(self.classical_foundation_apron_cb, 3, 0, 1, 4)

        self.classical_wrap_min_front_span_spinbox = QDoubleSpinBox()
        self.classical_wrap_min_front_span_spinbox.setRange(0.0, 60.0)
        self.classical_wrap_min_front_span_spinbox.setSingleStep(0.5)
        self.classical_wrap_min_front_span_spinbox.setDecimals(1)
        self.classical_wrap_min_front_span_spinbox.setSpecialValueText("Auto")
        self.classical_wrap_min_front_span_spinbox.setValue(0.0)
        tuning_layout.addWidget(QLabel("Wrap Min Front Span:"), 4, 0)
        tuning_layout.addWidget(self.classical_wrap_min_front_span_spinbox, 4, 1)

        self.classical_wrap_min_side_span_spinbox = QDoubleSpinBox()
        self.classical_wrap_min_side_span_spinbox.setRange(0.0, 60.0)
        self.classical_wrap_min_side_span_spinbox.setSingleStep(0.5)
        self.classical_wrap_min_side_span_spinbox.setDecimals(1)
        self.classical_wrap_min_side_span_spinbox.setSpecialValueText("Auto")
        self.classical_wrap_min_side_span_spinbox.setValue(0.0)
        tuning_layout.addWidget(QLabel("Wrap Min Side Span:"), 4, 2)
        tuning_layout.addWidget(self.classical_wrap_min_side_span_spinbox, 4, 3)

        self.classical_wrap_front_coverage_spinbox = QDoubleSpinBox()
        self.classical_wrap_front_coverage_spinbox.setRange(0.0, 1.20)
        self.classical_wrap_front_coverage_spinbox.setSingleStep(0.02)
        self.classical_wrap_front_coverage_spinbox.setDecimals(2)
        self.classical_wrap_front_coverage_spinbox.setSpecialValueText("Auto")
        self.classical_wrap_front_coverage_spinbox.setValue(0.0)
        tuning_layout.addWidget(QLabel("Wrap Coverage Ratio:"), 5, 0)
        tuning_layout.addWidget(self.classical_wrap_front_coverage_spinbox, 5, 1)

        gen_layout.addWidget(self.classical_tuning_group)

        checks_layout = QVBoxLayout()
        self.support_check_cb = QCheckBox("Enable Support Check")
        self.support_check_cb.setChecked(True)
        self.support_check_cb.toggled.connect(self.support_toggled.emit)
        checks_layout.addWidget(self.support_check_cb)

        self.overlap_check_cb = QCheckBox("Strict Overlap Check")
        self.overlap_check_cb.setChecked(True)
        self.overlap_check_cb.toggled.connect(self.overlap_toggled.emit)
        checks_layout.addWidget(self.overlap_check_cb)
        gen_layout.addLayout(checks_layout)

        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("PrimaryActionButton")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        gen_layout.addWidget(self.generate_button)

        main_layout.addWidget(gen_group)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _on_generate_clicked(self):
        seed_value: Optional[int]
        seed_text = self.seed_input.text().strip()
        if seed_text:
            seed_value = int(seed_text)
        else:
            seed_value = random.randint(0, 2**31 - 1)

        classical_order = self.classical_order_combo.currentText()
        classical_preset = self.classical_preset_combo.currentText()
        front_columns = self.classical_front_columns_spinbox.value()
        side_columns = self.classical_side_columns_spinbox.value()
        front_override = int(front_columns) if front_columns > 0 else None
        side_override = int(side_columns) if side_columns > 0 else None
        pediment_slope = float(self.classical_pediment_slope_spinbox.value())
        entablature_depth = float(self.classical_entablature_depth_spinbox.value())
        side_roof_overhang = float(self.classical_side_roof_overhang_spinbox.value())
        foundation_apron = bool(self.classical_foundation_apron_cb.isChecked())
        wrap_min_front_span = (
            float(self.classical_wrap_min_front_span_spinbox.value())
            if self.classical_wrap_min_front_span_spinbox.value() > 0.0
            else None
        )
        wrap_min_side_span = (
            float(self.classical_wrap_min_side_span_spinbox.value())
            if self.classical_wrap_min_side_span_spinbox.value() > 0.0
            else None
        )
        wrap_front_coverage = (
            float(self.classical_wrap_front_coverage_spinbox.value())
            if self.classical_wrap_front_coverage_spinbox.value() > 0.0
            else None
        )

        self.generate_triggered.emit(
            self.width_spinbox.value(),
            self.height_spinbox.value(),
            self.depth_spinbox.value(),
            seed_value,
            classical_order,
            classical_preset,
            front_override,
            side_override,
            pediment_slope,
            entablature_depth,
            side_roof_overhang,
            foundation_apron,
            wrap_min_front_span,
            wrap_min_side_span,
            wrap_front_coverage,
        )

    def _on_randomize_seed(self):
        self.seed_input.setText(str(random.randint(0, 2**31 - 1)))

    def _normalize_column_override_value(self, spinbox: QSpinBox):
        value = spinbox.value()
        if value <= 0:
            return

        adjusted = value
        if adjusted < 4:
            adjusted = 4
        if adjusted % 2 == 1:
            adjusted = max(4, adjusted - 1)

        if adjusted == value:
            return

        old_state = spinbox.blockSignals(True)
        spinbox.setValue(adjusted)
        spinbox.blockSignals(old_state)

    def apply_collision_defaults(self, collision_enabled: bool):
        """Sync UI defaults when collision backend is unavailable."""
        if collision_enabled:
            self.overlap_check_cb.setEnabled(True)
            return

        support_signal_state = self.support_check_cb.blockSignals(True)
        overlap_signal_state = self.overlap_check_cb.blockSignals(True)
        self.support_check_cb.setChecked(False)
        self.overlap_check_cb.setChecked(False)
        self.support_check_cb.blockSignals(support_signal_state)
        self.overlap_check_cb.blockSignals(overlap_signal_state)
        self.overlap_check_cb.setEnabled(False)


class SceneExportPanel(QWidget):
    """Minimal scene actions: optimize and exports."""

    optimize_triggered = pyqtSignal()
    export_glb_triggered = pyqtSignal()
    export_json_triggered = pyqtSignal()
    export_snapshot_triggered = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SceneExportPanel")

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        group = QGroupBox("Scene / Export")
        group.setObjectName("SceneExportGroup")
        layout = QVBoxLayout(group)

        self.optimize_button = QPushButton("Optimize (Union)")
        self.optimize_button.clicked.connect(self.optimize_triggered.emit)
        layout.addWidget(self.optimize_button)

        self.export_glb_button = QPushButton("Export GLB")
        self.export_glb_button.clicked.connect(self.export_glb_triggered.emit)
        layout.addWidget(self.export_glb_button)

        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.clicked.connect(self.export_json_triggered.emit)
        layout.addWidget(self.export_json_button)

        self.export_snapshot_button = QPushButton("Export Snapshot (PNG)")
        self.export_snapshot_button.clicked.connect(self.export_snapshot_triggered.emit)
        layout.addWidget(self.export_snapshot_button)

        main_layout.addWidget(group)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))


class ColorPalettePanel(QWidget):
    """Simple color picker."""

    color_selected = pyqtSignal(tuple)

    def __init__(self, colors: List[ColorTuple] = DEFAULT_COLORS, parent=None):
        super().__init__(parent)
        self.setObjectName("ColorPalettePanel")

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        palette_group = QGroupBox("Color Palette")
        palette_group.setObjectName("ColorPaletteGroup")
        palette_layout = QGridLayout(palette_group)

        cols = 3
        for index, color_tuple in enumerate(colors):
            button = QPushButton()
            button.setObjectName("SwatchButton")
            button.setFixedSize(32, 32)
            button.setToolTip(f"RGBA {color_tuple}")
            button.setStyleSheet(
                "QPushButton {"
                f"background-color: rgba({color_tuple[0]}, {color_tuple[1]}, {color_tuple[2]}, {color_tuple[3]});"
                "border: 2px solid rgba(255, 244, 220, 0.7);"
                "border-radius: 10px;"
                "}"
                "QPushButton:hover { border-color: rgba(23, 31, 48, 0.95); }"
            )
            button.clicked.connect(functools.partial(self._on_color_button_clicked, color_tuple))
            palette_layout.addWidget(button, index // cols, index % cols)

        main_layout.addWidget(palette_group)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

    def _on_color_button_clicked(self, color: ColorTuple):
        self.color_selected.emit(color)


class OutlinerPanel(QWidget):
    """Minimal scene outliner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OutlinerPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

    def update_list(self, blocks: List[Block]):
        self.list_widget.clear()
        if not blocks:
            self.list_widget.addItem("Scene is empty")
            self.list_widget.setEnabled(False)
            return

        self.list_widget.setEnabled(True)
        for block in blocks:
            item_text = (
                f"Block {block.block_id} "
                f"[{block.position[0]:.2f}, {block.position[1]:.2f}, {block.position[2]:.2f}]"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, block.block_id)
            self.list_widget.addItem(item)
