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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import config
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
    notice = pyqtSignal(str)

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
        self.width_spinbox.setValue(config.DEFAULT_WIDTH)
        self.width_spinbox.setToolTip("Temple width in block units. Values below 6 are built as 6.")

        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 200)
        self.height_spinbox.setValue(config.DEFAULT_HEIGHT)
        self.height_spinbox.setToolTip("Podium height is one third of this value, and at least 1.")

        self.depth_spinbox = QSpinBox()
        self.depth_spinbox.setRange(1, 200)
        self.depth_spinbox.setValue(config.DEFAULT_DEPTH)
        self.depth_spinbox.setToolTip("Temple depth in block units. Values below 6 are built as 6.")

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

        hint = QLabel(
            "Builds a Doric or Ionic temple. Width and depth below 6 are raised to 6. "
            "Podium height is the integer third of Height."
        )
        hint.setObjectName("PanelHint")
        hint.setWordWrap(True)
        gen_layout.addWidget(hint)

        self.generate_button = QPushButton("Generate Temple")
        self.generate_button.setObjectName("PrimaryActionButton")
        self.generate_button.clicked.connect(self._on_generate_clicked)
        gen_layout.addWidget(self.generate_button)

        main_layout.addWidget(gen_group)

    def _parse_seed(self) -> Optional[int]:
        seed_text = self.seed_input.text().strip()
        if not seed_text:
            seed_value = random.randint(0, 2**31 - 1)
            self.seed_input.setText(str(seed_value))
            return seed_value
        try:
            return int(seed_text)
        except ValueError:
            self.notice.emit("Seed must be an integer.")
            return None

    def _on_generate_clicked(self):
        seed_value = self._parse_seed()
        if seed_value is None:
            return

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

        self._swatches = []
        cols = 3
        for index, color_tuple in enumerate(colors):
            button = QPushButton()
            button.setObjectName("SwatchButton")
            button.setFixedSize(32, 32)
            button.setToolTip(f"RGBA {color_tuple}")
            button.setProperty("swatch", color_tuple)
            self._apply_swatch_style(button, color_tuple, selected=False)
            button.clicked.connect(functools.partial(self._on_color_button_clicked, button, color_tuple))
            palette_layout.addWidget(button, index // cols, index % cols)
            self._swatches.append(button)

        main_layout.addWidget(palette_group)

    def _apply_swatch_style(self, button: QPushButton, color: ColorTuple, selected: bool):
        border = "3px solid #ffd36a" if selected else "2px solid rgba(255, 244, 220, 0.7)"
        button.setStyleSheet(
            "QPushButton {"
            f"background-color: rgba({color[0]}, {color[1]}, {color[2]}, {color[3]});"
            f"border: {border};"
            "border-radius: 10px;"
            "}"
            "QPushButton:hover { border-color: #fff6e3; }"
        )

    def _on_color_button_clicked(self, button: QPushButton, color: ColorTuple):
        for swatch in self._swatches:
            swatch_color = swatch.property("swatch")
            self._apply_swatch_style(swatch, swatch_color, selected=swatch is button)
        self.color_selected.emit(color)


class OutlinerPanel(QWidget):
    """Scene member list. Selecting a row highlights that block in the viewport."""

    block_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("OutlinerPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.summary_label = QLabel("No blocks")
        self.summary_label.setObjectName("PanelSubtitle")
        layout.addWidget(self.summary_label)
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def update_list(self, blocks: List[Block]):
        self.list_widget.clear()
        if not blocks:
            self.summary_label.setText("No blocks")
            self.list_widget.addItem("Scene is empty")
            self.list_widget.setEnabled(False)
            return

        count = len(blocks)
        noun = "block" if count == 1 else "blocks"
        self.summary_label.setText(f"{count} {noun}")
        self.list_widget.setEnabled(True)
        for block in blocks:
            element = block.parameters.get("element") if block.parameters else None
            kind = str(element or block.block_type)
            item_text = (
                f"{block.block_id:>4}  {kind:<16} "
                f"({block.position[0]:.2f}, {block.position[1]:.2f}, {block.position[2]:.2f})"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, block.block_id)
            self.list_widget.addItem(item)

    def _on_item_clicked(self, item: QListWidgetItem):
        block_id = item.data(Qt.UserRole)
        if block_id is None:
            return
        self.block_selected.emit(int(block_id))
