import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt")

from PyQt5.QtCore import Qt

from core.scene import Block
from gui.panels import ColorPalettePanel, GenerationPanel, OutlinerPanel
import numpy as np


def test_empty_seed_is_written_back(qtbot):
    panel = GenerationPanel()
    qtbot.addWidget(panel)
    panel.seed_input.clear()
    with qtbot.waitSignal(panel.generate_triggered, timeout=1000) as blocker:
        panel.generate_button.click()
    shown = panel.seed_input.text()
    assert shown.isdigit()
    assert blocker.args[3] == int(shown)


def test_invalid_seed_does_not_generate(qtbot):
    panel = GenerationPanel()
    qtbot.addWidget(panel)
    panel.seed_input.setText("nope")
    with qtbot.waitSignal(panel.notice, timeout=1000) as notice:
        with qtbot.assertNotEmitted(panel.generate_triggered):
            panel._on_generate_clicked()
    assert "integer" in notice.args[0]


def test_outliner_names_members_and_emits_selection(qtbot):
    panel = OutlinerPanel()
    qtbot.addWidget(panel)
    block = Block(
        position=np.array([1.0, 2.0, 3.0]),
        block_type="CYLINDER",
        parameters={"element": "shaft", "radius": 0.4, "height": 1.0},
        block_id=12,
    )
    panel.update_list([block])
    assert panel.summary_label.text() == "1 block"
    item = panel.list_widget.item(0)
    assert "shaft" in item.text()
    with qtbot.waitSignal(panel.block_selected, timeout=1000) as selected:
        panel.list_widget.itemClicked.emit(item)
    assert selected.args == [12]


def test_empty_outliner_does_not_emit_a_selection(qtbot):
    panel = OutlinerPanel()
    qtbot.addWidget(panel)
    panel.update_list([])
    assert panel.summary_label.text() == "No blocks"
    item = panel.list_widget.item(0)
    with qtbot.assertNotEmitted(panel.block_selected):
        panel.list_widget.itemClicked.emit(item)


def test_color_swatch_marks_the_selection(qtbot):
    panel = ColorPalettePanel()
    qtbot.addWidget(panel)
    panel.show()
    button = panel._swatches[0]
    with qtbot.waitSignal(panel.color_selected, timeout=1000):
        qtbot.mouseClick(button, Qt.LeftButton)
    assert "#ffd36a" in button.styleSheet()
    assert "#ffd36a" not in panel._swatches[1].styleSheet()
