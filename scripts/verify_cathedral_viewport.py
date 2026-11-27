"""Exercise the live Qt/VTK cathedral: picking, cutaway, append and orbit cost.

Run on a working display with the GUI dependencies. This is an integration
probe, separate from headless unit tests. Its FPS is a local diagnostic only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
from PyQt5.QtWidgets import QApplication
from core.scene import Block
from gui.main_window import MainWindow
from gui.camera import frame_cathedral


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "builds/cathedral-upgrade/validation")
    parser.add_argument("--frames", type=int, default=105)
    parser.add_argument("--no-fxaa", action="store_true", help="Measure native line rendering without post-process antialiasing")
    args = parser.parse_args()
    if args.frames < 30:
        parser.error("at least 30 frames are needed")
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.resize(1682, 935)
    win.show()
    p = win.generation_panel
    p.classical_form_combo.setCurrentIndex(p.classical_form_combo.findData("gothic-cathedral"))
    p.form_detail.setCurrentText("presentation")
    p.seed_input.setText("73")
    started = time.perf_counter()
    p.generate_button.click()
    while win._generation_thread is not None:
        app.processEvents()
        time.sleep(.01)
    assert len(win.scene.blocks) > 1000, win.statusBar().currentMessage()
    generation = time.perf_counter() - started
    v = win.viewport
    if args.no_fxaa:
        v.plotter.renderer.SetUseFXAA(False)
    assert len(v._batch_cell_ids) == sum(len(b.mesh.faces) for b in win.scene.blocks)
    assert not p.cathedral_style.isHidden() and p.classical_order_row.isHidden()
    capabilities = v.plotter.render_window.ReportCapabilities()
    (args.output / "renderer.txt").write_text(capabilities)
    chosen = []
    v.block_picked.connect(chosen.append)
    renderer = v.plotter.renderer
    for point in ([0, .3, -28], [-8, 14, -27], [-8, 1, -28]):
        renderer.SetWorldPoint(*point, 1)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        picker = v.plotter.iren.picker
        picker.Pick(x, y, 0, renderer)
        if picker.GetActor() == v._batch_actor:
            v._handle_pick(picker.GetActor())
            break
    ids = {b.block_id for b in win.scene.blocks}
    assert chosen and chosen[-1] in ids, chosen
    v._clear_highlight()
    v.plotter.render()
    times = []
    for i in range(args.frames):
        t = time.perf_counter()
        v.plotter.camera.Azimuth(.18)
        v.plotter.render()
        if i >= 15:
            times.append(time.perf_counter() - t)
    v.apply_stable_camera()
    v.plotter.render()
    v.plotter.screenshot(str(args.output / "cathedral.png"))
    p.cathedral_cutaway.setChecked(True)
    app.processEvents()
    assert {b.block_id for b in win.scene.blocks} == ids
    assert len(v._batch_blocks) < len(ids)
    hidden = {"nave_roof", "aisle_roof", "transept_roof", "apse_roof", "roof_seam", "ridge", "glazing"}
    assert not any(b.parameters["component_role"] in hidden for b in v._batch_blocks.values())
    assert any(b.parameters["component_role"] == "vault_rib" for b in v._batch_blocks.values())
    frame_cathedral(v.plotter, np.vstack([b.mesh.vertices for b in v._batch_blocks.values()]), (-1.3, 2.6, -.6))
    v.plotter.render()
    app.processEvents()
    v.plotter.render_window.Render()
    v.plotter.screenshot(str(args.output / "cutaway.png"))
    p.cathedral_cutaway.setChecked(False)
    app.processEvents()
    assert len(v._batch_blocks) == len(ids)
    # An appended member must not invalidate the old batch's picking IDs.
    extra = Block(position=np.array([40., .5, 0.]), block_id=max(ids) + 1)
    v.display_meshes([extra], clear_existing=False)
    assert set(v._batch_blocks) == ids | {extra.block_id}
    assert set(v._batch_cell_ids) == ids | {extra.block_id}
    assert len([actor for actor, value in v._actors.items() if value == -1]) == 1
    report = {
        "status": "pass", "placements": len(ids), "triangles": sum(len(b.mesh.faces) for b in win.scene.blocks),
        "generation_and_display_seconds": generation,
        "median_orbit_fps": 1 / float(np.median(times)), "p95_frame_ms": float(np.quantile(times, .95)) * 1000,
        "vtk_viewport": v.plotter.window_size, "qt_window": [win.width(), win.height()],
        "LP_NUM_THREADS": os.environ.get("LP_NUM_THREADS", "renderer default"),
        "MSAA": v.plotter.render_window.GetMultiSamples(), "FXAA": bool(v.plotter.renderer.GetUseFXAA()),
        "checks": ["exact batch-cell picking", "cutaway preserves export scene", "all coverings hidden",
                   "vault ribs retained", "restore all members", "append retains old and new picking IDs"],
        "architect_sha256": hashlib.sha256((ROOT / "core/architecture/cathedral.py").read_bytes()).hexdigest(),
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    win.close()
    app.processEvents()


if __name__ == "__main__":
    main()
