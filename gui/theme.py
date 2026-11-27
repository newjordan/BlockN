"""Green phosphor terminal styling for the classical workbench."""
PHOSPHOR_COLORS = [(0, level, level // 4, 255) for level in (64, 88, 112, 136, 160, 184, 208, 232, 255)]
PHOSPHOR_STYLESHEET = """
QWidget {
    background: #000000; color: #00dd38;
    font-family: "Berkeley Mono", "DejaVu Sans Mono", monospace;
    font-size: 12px; selection-background-color: #00ff41; selection-color: #000000;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #000000; }
QDockWidget { color: #00ff41; border: 1px solid #0d4f21; }
QDockWidget::title { background: #001c07; padding: 6px; border-bottom: 1px solid #00aa2b; }
QGroupBox { border: 1px solid #0d4f21; margin-top: 16px; padding: 12px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #00ff41; }
QLabel#PanelSubtitle { color: #00ff41; font-size: 13px; font-weight: bold; padding: 8px 0; }
QLabel#GenerationMetadataLabel { color: #00aa2b; padding-top: 8px; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #000000; color: #00ff41; border: 1px solid #0d4f21; padding: 6px;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #00ff41; }
QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { background: #00300c; border: 0; width: 16px; }
QComboBox::down-arrow, QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid #00dd38;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none; width: 0; height: 0;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid #00dd38;
}
QComboBox QAbstractItemView { border: 1px solid #00ff41; padding: 4px; }
QPushButton { border: 1px solid #008822; background: #000000; color: #00ff41; padding: 7px 10px; }
QPushButton:hover { background: #00300c; border-color: #00ff41; }
QPushButton:pressed { background: #00ff41; color: #000000; }
QPushButton#PrimaryActionButton { border: 2px solid #00ff41; font-weight: bold; padding: 12px; }
QWidget:disabled { color: #075419; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #008822; background: #000000; }
QCheckBox::indicator:checked { background: #00cc33; }
QListWidget { border: 1px solid #0d4f21; background: #000000; }
QListWidget::item { padding: 4px; }
QListWidget::item:selected { background: #00300c; color: #00ff41; border-left: 2px solid #00ff41; }
QStatusBar { border-top: 1px solid #0d4f21; color: #00aa2b; }
QScrollBar:vertical { background: #000000; width: 9px; }
QScrollBar:horizontal { background: #000000; height: 9px; }
QScrollBar::handle { background: #075419; min-height: 24px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QToolTip { background: #000000; color: #00ff41; border: 1px solid #008822; }
"""
