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

from core.materials import ColorTuple
from .theme import PHOSPHOR_COLORS
from core.scene import Block
from core.architecture.classical_forms import FORMS, MASONRY_FORMS, random_controls
from core.architecture import cathedral, historic, marble_run, persian, settlement, timberwright


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
        bool,
        bool,
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

        self.style_label = QLabel("Classical Architecture Composer")
        self.style_label.setObjectName("PanelSubtitle")
        gen_layout.addWidget(self.style_label)
        self.classical_form_combo = QComboBox()
        self.classical_form_combo.addItem("Classical Temple", "classical-building")
        for name, (description, _, _) in FORMS.items():
            self.classical_form_combo.addItem(name.removeprefix("classical-").replace("-", " ").title(), name)
            self.classical_form_combo.setItemData(self.classical_form_combo.count() - 1, description, Qt.ToolTipRole)
        self.classical_form_combo.addItem("Gothic Cathedral", cathedral.PRODUCT)
        for name, (_, form) in historic.styles().items():
            self.classical_form_combo.addItem(name.replace("-", " ").title(), name)
            self.classical_form_combo.setItemData(self.classical_form_combo.count() - 1, form.description, Qt.ToolTipRole)
        self.classical_form_combo.addItem("Marble Run", marble_run.PRODUCT)
        self.classical_form_combo.setItemData(
            self.classical_form_combo.count() - 1,
            "Closed aqueduct-and-spline canal; marble loops forever",
            Qt.ToolTipRole,
        )
        self.classical_form_combo.addItem("Half-Timbered House", timberwright.PRODUCT)
        self.classical_form_combo.addItem("Iwan Court (Persian)", persian.PRODUCT)
        self.classical_form_combo.addItem("English Village", settlement.ENGLISH)
        self.classical_form_combo.setItemData(
            self.classical_form_combo.count() - 1,
            "Medieval English village: burgage tofts, a church, and lanes",
            Qt.ToolTipRole,
        )
        self.classical_form_combo.addItem("Roman Town", settlement.ROMAN)
        self.classical_form_combo.setItemData(
            self.classical_form_combo.count() - 1,
            "Roman town quarter: cardo, decumanus, insulae and a forum",
            Qt.ToolTipRole,
        )
        gen_layout.addWidget(self.classical_form_combo)
        self.settlement_variant = QComboBox()
        self.settlement_variant.setToolTip("Settlement plan")
        self.settlement_variant.hide()
        self.settlement_variant.currentIndexChanged.connect(self._on_settlement_variant)
        gen_layout.addWidget(self.settlement_variant)
        self.timber_controls = QWidget()
        timber_layout = QGridLayout(self.timber_controls)
        timber_layout.setContentsMargins(0, 0, 0, 0)
        self.timber_variant = QComboBox()
        self.timber_finish = QComboBox()
        for row, (label, combo, choices) in enumerate((
                ("Roof plan:", self.timber_variant, timberwright.VARIANTS),
                ("Finish:", self.timber_finish, timberwright.FINISHES))):
            combo.addItem("From seed", "auto")
            for value in choices:
                combo.addItem(value.replace("-", " ").title(), value)
            timber_layout.addWidget(QLabel(label), row, 0)
            timber_layout.addWidget(combo, row, 1)
        gen_layout.addWidget(self.timber_controls)
        self.historic_style = QComboBox()
        self.historic_style.setToolTip("Building tradition within this architect")
        self.historic_style.hide()
        gen_layout.addWidget(self.historic_style)
        self.persian_controls = QWidget()
        persian_layout = QGridLayout(self.persian_controls)
        persian_layout.setContentsMargins(0, 0, 0, 0)
        self.persian_variant = QComboBox()
        self.persian_finish = QComboBox()
        for row, (label, combo, choices) in enumerate((
                ("Court plan:", self.persian_variant, persian.VARIANTS),
                ("Finish:", self.persian_finish, persian.FINISHES))):
            combo.addItem("From seed", "auto")
            for value in choices:
                combo.addItem(value.replace("-", " ").title(), value)
            persian_layout.addWidget(QLabel(label), row, 0)
            persian_layout.addWidget(combo, row, 1)
        gen_layout.addWidget(self.persian_controls)
        self.cathedral_style = QComboBox()
        for variant in cathedral.VARIANTS:
            self.cathedral_style.addItem(variant.replace("-", " ").title(), variant)
        self.cathedral_style.hide()
        gen_layout.addWidget(self.cathedral_style)
        self.cathedral_cutaway = QCheckBox("Open roof / inspect vaults")
        self.cathedral_cutaway.hide()
        gen_layout.addWidget(self.cathedral_cutaway)
        self.cathedral_controls = QWidget()
        cathedral_layout = QGridLayout(self.cathedral_controls)
        cathedral_layout.setContentsMargins(0,0,0,0)
        self.cathedral_choir = QSpinBox()
        self.cathedral_choir.setRange(2,4)
        self.cathedral_choir.setValue(3)
        self.cathedral_apse = QComboBox()
        self.cathedral_apse.addItems(["5","7","9"])
        self.cathedral_apse.setCurrentText("7")
        self.cathedral_spire = QDoubleSpinBox()
        self.cathedral_spire.setRange(.22,.36)
        self.cathedral_spire.setSingleStep(.01)
        self.cathedral_spire.setValue(.30)
        self.cathedral_spire.setToolTip("West-spire share of total height")
        cathedral_layout.addWidget(QLabel("Choir bays:"),0,0)
        cathedral_layout.addWidget(self.cathedral_choir,0,1)
        cathedral_layout.addWidget(QLabel("Apse sides:"),0,2)
        cathedral_layout.addWidget(self.cathedral_apse,0,3)
        cathedral_layout.addWidget(QLabel("West spires:"),1,0)
        cathedral_layout.addWidget(self.cathedral_spire,1,1)
        self.cathedral_style.currentIndexChanged.connect(
            lambda: self.cathedral_spire.setEnabled(self.cathedral_style.currentData()!="early-english"))
        self.cathedral_controls.hide()
        gen_layout.addWidget(self.cathedral_controls)
        self.form_controls = QWidget()
        form_layout = QHBoxLayout(self.form_controls)
        form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_bays = QSpinBox()
        self.form_bays.setRange(2, 12)
        self.form_bays.setValue(5)
        self.form_levels = QSpinBox()
        self.form_levels.setRange(1, 3)
        self.form_levels.setValue(2)
        self.form_detail = QComboBox()
        self.form_detail.addItems(["concept", "standard", "presentation"])
        self.form_detail.setCurrentText("standard")
        self.form_bays_label=QLabel("Bays:")
        form_layout.addWidget(self.form_bays_label)
        form_layout.addWidget(self.form_bays)
        self.form_levels_label=QLabel("Tiers:")
        form_layout.addWidget(self.form_levels_label)
        form_layout.addWidget(self.form_levels)
        form_layout.addWidget(self.form_detail)
        gen_layout.addWidget(self.form_controls)
        self.form_controls.hide()

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
        self.seed_input.setValidator(QIntValidator(0, 2 ** 31 - 1))
        self.randomize_seed_button = QPushButton("Randomize")
        self.randomize_seed_button.clicked.connect(self._on_randomize_controls)
        seed_layout.addWidget(QLabel("Seed:"))
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.randomize_seed_button)
        gen_layout.addLayout(seed_layout)

        age_layout = QHBoxLayout()
        self.age_spinbox = QDoubleSpinBox()
        self.age_spinbox.setRange(0.0, 1.0)
        self.age_spinbox.setSingleStep(0.05)
        self.age_spinbox.setDecimals(2)
        self.age_spinbox.setValue(0.35)
        self.age_spinbox.setToolTip("Surface age from 0 (newly worked) to 1 (centuries)")
        age_layout.addWidget(QLabel("Age:"))
        age_layout.addWidget(self.age_spinbox)
        gen_layout.addLayout(age_layout)

        self.feature_shapes_group = QGroupBox("Feature Shapes")
        self.feature_shapes_group.setObjectName("FeatureShapesGroup")
        feature_shapes_layout = QVBoxLayout(self.feature_shapes_group)
        self.classical_add_arch_portal_cb = QCheckBox("Arch Portal")
        self.classical_add_arch_portal_cb.setChecked(False)
        self.classical_add_arch_portal_cb.setToolTip("Add a generated arch entry shape to the facade")
        feature_shapes_layout.addWidget(self.classical_add_arch_portal_cb)

        self.classical_add_dome_cap_cb = QCheckBox("Dome Cap")
        self.classical_add_dome_cap_cb.setChecked(False)
        self.classical_add_dome_cap_cb.setToolTip("Add a generated dome shape on the roof")
        feature_shapes_layout.addWidget(self.classical_add_dome_cap_cb)
        gen_layout.addWidget(self.feature_shapes_group)

        self.classical_order_row = QWidget()
        classical_order_layout = QHBoxLayout(self.classical_order_row)
        classical_order_layout.setContentsMargins(0, 0, 0, 0)
        self.classical_order_combo = QComboBox()
        self.classical_order_combo.addItems(["Random", "Doric", "Ionic", "Tuscan", "Corinthian", "Composite"])
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

        self.seed_metadata_label = QLabel("Last build: n/a")
        self.seed_metadata_label.setObjectName("GenerationMetadataLabel")
        self.seed_metadata_label.setWordWrap(True)
        self.seed_metadata_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        gen_layout.addWidget(self.seed_metadata_label)

        main_layout.addWidget(gen_group)
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.classical_form_combo.currentIndexChanged.connect(self._on_form_changed)
        self._on_form_changed()

    def _on_form_changed(self):
        product = self.classical_form_combo.currentData()
        if product in settlement.PRODUCT_NAMES:
            self._apply_settlement_form(product)
            return
        self._leave_settlement_form()
        legacy = product == "classical-building"
        gothic = product == cathedral.PRODUCT
        marble = product == marble_run.PRODUCT
        timber = product == timberwright.PRODUCT
        found = historic.style_for(product)
        is_persian = product == persian.PRODUCT
        self.persian_controls.setVisible(is_persian)
        self.timber_controls.setVisible(timber)
        self.historic_style.setVisible(bool(found))
        if found:
            module, form = found
            self.historic_style.clear()
            for variant in form.variants:
                self.historic_style.addItem(variant.replace("-", " ").title(), variant)
            self.style_label.setText(module.ARCHITECT.title)
        elif timber:
            self.style_label.setText(timberwright.ARCHITECT.title)
        elif gothic:
            self.style_label.setText("Cathedral Architect")
        elif marble:
            self.style_label.setText("Marble Run Architect")
        elif is_persian:
            self.style_label.setText(persian.ARCHITECT.title)
        else:
            self.style_label.setText("Classical Architecture Composer")
        self.cathedral_style.setVisible(gothic)
        self.cathedral_cutaway.setVisible(gothic or bool(found))
        self.cathedral_controls.setVisible(gothic)
        self.form_bays_label.setText("Nave bays:" if gothic else "Bays:")
        self.form_levels_label.setVisible(not gothic)
        self.form_levels.setVisible(not gothic)
        self.form_controls.setVisible(not legacy)
        self.classical_tuning_group.setVisible(legacy)
        self.feature_shapes_group.setVisible(legacy)
        self.classical_order_row.setVisible(product not in MASONRY_FORMS and not (gothic or marble or timber or found or is_persian))
        self.form_levels.setRange(1, 2 if timber else 3)
        self.form_levels.setEnabled(product in {"classical-aqueduct", marble_run.PRODUCT, timberwright.PRODUCT})
        self.form_bays.setRange(4 if gothic else 2, 7 if timber else 10 if gothic else 12)
        self.form_bays.setEnabled(True)
        if found:
            # Each form says which of bays and levels it understands, and their range.
            form = found[1]
            for name, widget in (("bays", self.form_bays), ("levels", self.form_levels)):
                widget.setEnabled(name in form.controls)
                if name in form.controls:
                    widget.setRange(*form.rules[name])
                    widget.setValue(form.controls[name])
        if legacy:
            for widget, value in zip((self.width_spinbox, self.height_spinbox, self.depth_spinbox), (14, 9, 20)):
                widget.setValue(value)
        else:
            if found:
                size = found[1].defaults
            elif gothic:
                size = cathedral.DEFAULTS
            elif marble:
                size = marble_run.DEFAULTS
            elif timber:
                size = tuple(timberwright.DEFAULTS[k] for k in ("width", "height", "depth"))
            elif is_persian:
                size = tuple(persian.DEFAULTS[k] for k in ("width", "height", "depth"))
            else:
                size = FORMS[product][1]
            for widget, value in zip((self.width_spinbox, self.height_spinbox, self.depth_spinbox), size):
                widget.setValue(int(value))
            if gothic:
                self.form_bays.setValue(7)
            elif marble:
                self.form_bays.setValue(8)
            elif timber:
                self.form_bays.setValue(timberwright.DEFAULTS['bays'])
                self.form_levels.setValue(timberwright.DEFAULTS['levels'])

    def _apply_settlement_form(self, product):
        """Site size and plan. Height, bays and order belong to a single building."""
        self.persian_controls.hide()
        self.timber_controls.hide()
        self.historic_style.hide()
        self.cathedral_style.hide()
        self.cathedral_cutaway.hide()
        self.cathedral_controls.hide()
        self.classical_tuning_group.hide()
        self.feature_shapes_group.hide()
        self.classical_order_row.hide()
        self.form_controls.show()
        self.form_bays.hide()
        self.form_bays_label.hide()
        self.form_levels.hide()
        self.form_levels_label.hide()
        self.form_detail.show()
        self.form_detail.setCurrentText("concept")
        self.height_spinbox.setEnabled(False)
        self.height_spinbox.setValue(1)
        self.style_label.setText(settlement.ARCHITECT.title)
        variants = settlement.ENGLISH_VARIANTS if product == settlement.ENGLISH else settlement.ROMAN_VARIANTS
        self.settlement_variant.blockSignals(True)
        self.settlement_variant.clear()
        for variant in variants:
            self.settlement_variant.addItem(variant.replace("-", " ").title(), variant)
        self.settlement_variant.blockSignals(False)
        self.settlement_variant.show()
        self._apply_settlement_site()

    def _leave_settlement_form(self):
        self.settlement_variant.hide()
        self.height_spinbox.setEnabled(True)
        self.width_spinbox.setMinimum(1)
        self.depth_spinbox.setMinimum(1)
        self.form_bays.show()
        self.form_bays_label.show()

    def _on_settlement_variant(self):
        if self.classical_form_combo.currentData() in settlement.PRODUCT_NAMES:
            self._apply_settlement_site()

    def _apply_settlement_site(self):
        variant = self.settlement_variant.currentData()
        if variant not in settlement.DEFAULT_SITES:
            return
        width, depth = settlement.DEFAULT_SITES[variant]
        self.width_spinbox.setMinimum(int(width))
        self.depth_spinbox.setMinimum(int(depth))
        self.width_spinbox.setValue(int(width))
        self.depth_spinbox.setValue(int(depth))

    def timber_parameters(self):
        finish = self.timber_finish.currentData()
        return {} if finish == 'auto' else {'finish': finish}

    def cathedral_parameters(self):
        parameters={"choir_bays":self.cathedral_choir.value(),
                    "apse_sides":int(self.cathedral_apse.currentText())}
        if self.cathedral_spire.isEnabled():
            parameters["spire_ratio"]=self.cathedral_spire.value()
        return parameters

    def persian_parameters(self):
        finish = self.persian_finish.currentData()
        return {} if finish == 'auto' else {'finish': finish}

    def _on_generate_clicked(self):
        seed_value: Optional[int]
        seed_text = self.seed_input.text().strip()
        try:
            seed_value = int(seed_text) if seed_text else random.randint(0, 2**31 - 1)
        except ValueError:
            # A partially typed seed must never escape as an exception from a Qt
            # slot, which PyQt turns into a fatal abort.
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
        add_arch_portal = bool(self.classical_add_arch_portal_cb.isChecked())
        add_dome_cap = bool(self.classical_add_dome_cap_cb.isChecked())

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
            add_arch_portal,
            add_dome_cap,
        )

    def _on_randomize_controls(self):
        self.seed_input.setText(str(random.randint(0, 2**31 - 1)))
        product = self.classical_form_combo.currentData()
        if product in settlement.PRODUCT_NAMES:
            controls = settlement.random_controls(product, variant=self.settlement_variant.currentData())
            self.width_spinbox.setValue(int(controls["width"]))
            self.depth_spinbox.setValue(int(controls["depth"]))
            self.form_detail.setCurrentText(str(controls["detail"]))
            return
        if product != "classical-building":
            # The typology owns its buildable envelope; the selected form and
            # order stay exactly as the user set them.
            if historic.style_for(product):
                controls = historic.style_for(product)[0].random_controls(product, random)
                controls["bays"] = controls["bays"] or self.form_bays.value()
            elif product == cathedral.PRODUCT:
                controls = cathedral.random_controls()
            elif product == marble_run.PRODUCT:
                controls = marble_run.random_controls()
            elif product == timberwright.PRODUCT:
                controls = timberwright.random_controls(variant=self.timber_variant.currentData())
            else:
                controls = random_controls(product)
            self.width_spinbox.setValue(int(controls["width"]))
            self.height_spinbox.setValue(int(controls["height"]))
            self.depth_spinbox.setValue(int(controls["depth"]))
            self.form_bays.setValue(int(controls["bays"]))
            self.form_detail.setCurrentText(str(controls["detail"]))
            if controls["levels"] is not None:
                self.form_levels.setValue(int(controls["levels"]))
            return
        # Keep order unchanged; randomize all other generation-facing values.
        def _rand_int_in(spinbox: QSpinBox, low: int, high: int) -> int:
            floor = max(low, spinbox.minimum())
            ceil = min(high, spinbox.maximum())
            if floor > ceil:
                floor = spinbox.minimum()
                ceil = spinbox.maximum()
            return random.randint(floor, ceil)

        self.width_spinbox.setValue(_rand_int_in(self.width_spinbox, 8, 32))
        self.height_spinbox.setValue(_rand_int_in(self.height_spinbox, 6, 24))
        self.depth_spinbox.setValue(_rand_int_in(self.depth_spinbox, 8, 32))

        preset_count = self.classical_preset_combo.count()
        if preset_count > 0:
            self.classical_preset_combo.setCurrentIndex(random.randrange(preset_count))

        front_auto = random.random() < 0.35
        side_auto = random.random() < 0.35
        self.classical_front_columns_spinbox.setValue(
            0 if front_auto else random.choice([4, 6, 8, 10, 12])
        )
        self.classical_side_columns_spinbox.setValue(
            0 if side_auto else random.choice([4, 6, 8, 10, 12])
        )

        self.classical_pediment_slope_spinbox.setValue(round(random.uniform(0.75, 1.35), 2))
        self.classical_entablature_depth_spinbox.setValue(round(random.uniform(0.80, 1.30), 2))
        self.classical_side_roof_overhang_spinbox.setValue(round(random.uniform(0.80, 1.40), 2))
        self.classical_foundation_apron_cb.setChecked(random.random() < 0.5)

        if random.random() < 0.5:
            self.classical_wrap_min_front_span_spinbox.setValue(0.0)
        else:
            self.classical_wrap_min_front_span_spinbox.setValue(round(random.uniform(12.0, 28.0), 1))

        if random.random() < 0.5:
            self.classical_wrap_min_side_span_spinbox.setValue(0.0)
        else:
            self.classical_wrap_min_side_span_spinbox.setValue(round(random.uniform(8.0, 20.0), 1))

        if random.random() < 0.5:
            self.classical_wrap_front_coverage_spinbox.setValue(0.0)
        else:
            self.classical_wrap_front_coverage_spinbox.setValue(round(random.uniform(0.35, 0.85), 2))

        self.classical_add_arch_portal_cb.setChecked(random.random() < 0.35)
        self.classical_add_dome_cap_cb.setChecked(random.random() < 0.25)

    def set_generation_pending(self, seed: Optional[int], order_hint: str, preset: str):
        seed_text = str(seed) if seed is not None else "n/a"
        self.seed_metadata_label.setText(
            f"Generating: seed={seed_text} | order={order_hint} | preset={preset}"
        )

    def set_typology_metadata(self, seed: Optional[int], metadata: dict):
        seed_text = str(seed) if seed is not None else "n/a"
        counts = metadata.get("checks") or {}
        tally = " ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "no checks"
        self.seed_metadata_label.setText(
            f"Last build: seed={seed_text}"
            f" | typology={metadata.get('typology') or 'n/a'}"
            f" | {metadata.get('variant_label', 'order')}={metadata.get('order') or 'masonry (no order)'}"
            f" | bays={metadata.get('bays') if metadata.get('bays') is not None else 'n/a'}"
            f" | detail={metadata.get('detail') or 'n/a'}"
            f" | quality={metadata.get('quality') or 'n/a'} ({tally})"
        )

    def set_generation_metadata(
        self,
        seed: Optional[int],
        order: Optional[str],
        preset: Optional[str],
        front_columns: Optional[int],
        side_columns: Optional[int],
    ):
        seed_text = str(seed) if seed is not None else "n/a"
        order_text = str(order) if order else "n/a"
        preset_text = str(preset) if preset else "n/a"
        front_text = str(front_columns) if isinstance(front_columns, int) else "n/a"
        side_text = str(side_columns) if isinstance(side_columns, int) else "n/a"
        self.seed_metadata_label.setText(
            f"Last build: seed={seed_text} | order={order_text} | preset={preset_text} | cols={front_text}x{side_text}"
        )

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

        self.glb_materials = QCheckBox("Include materials in GLB")
        self.glb_materials.setChecked(True)
        self.glb_materials.setToolTip("Export the authored finishes. Turn off for neutral geometry; the viewport stays neon.")
        layout.addWidget(self.glb_materials)
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

    def __init__(self, colors: List[ColorTuple] = PHOSPHOR_COLORS, parent=None):
        super().__init__(parent)
        self.setObjectName("ColorPalettePanel")

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        palette_group = QGroupBox("Phosphor Intensity")
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
                "border: 1px solid #008822;"
                "border-radius: 0px;"
                "}"
                "QPushButton:hover { border-color: #00ff41; }"
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
            parameters = block.parameters or {}
            role = parameters.get("component_role") or parameters.get("element") or block.block_type.lower()
            item = QListWidgetItem(f"{block.block_id:04d} {role}")
            item.setToolTip(
                f"{parameters.get('primitive_id', role)}\n"
                f"Position: {tuple(round(float(v), 3) for v in block.position)}"
            )
            item.setData(Qt.UserRole, block.block_id)
            self.list_widget.addItem(item)
