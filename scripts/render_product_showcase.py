"""Record a product introduction using the real Blocks workbench and assets.

Run after exporting examples/through-the-ages.json to the adjacent asset folder.
The recorded frame rate is an offline video setting, not a runtime benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from render_architect_showcase import (
    ROOT, W, H, FPS, GREEN, DIM, Image, ImageDraw, np,
    QApplication, MainWindow, ViewportWidget, BuildRequest, build,
    load_document, rebuild, frame_cathedral, Writer, caption,
    font, capture_window, model_clip,
)
from render_peaceful_score import render_score, mux_score
from core.generators.classical import ORDER_SPECS, _build_column
from core.scene import Scene


def workbench(app, out, product, order, seconds, name, title, chapter, cutaway=False):
    win = MainWindow()
    win.resize(1840, 846)
    win.show()
    panel = win.generation_panel
    panel.classical_form_combo.setCurrentIndex(panel.classical_form_combo.findData(product))
    panel.classical_order_combo.setCurrentText(order)
    panel.form_detail.setCurrentText("presentation")
    panel.seed_input.setText("73")
    panel.generate_button.click()
    while win._generation_thread is not None:
        app.processEvents()
        time.sleep(.01)
    assert len(win.scene.blocks) > 100, win.statusBar().currentMessage()
    original = {b.block_id for b in win.scene.blocks}
    app.processEvents()
    points = np.vstack([b.mesh.vertices for b in win.scene.blocks])
    frame_cathedral(win.viewport.plotter, points, (-1.8, .9, -2.2), margin=1.08)
    writer = Writer(out, name)
    for i in range(seconds * FPS):
        if cutaway and i == 3 * FPS:
            panel.cathedral_cutaway.setChecked(True)
            app.processEvents()
            points = np.vstack([b.mesh.vertices for b in win.viewport._batch_blocks.values()])
            frame_cathedral(win.viewport.plotter, points, (-1.3, 2.2, -.7), margin=1.08)
        win.viewport.plotter.camera.Azimuth(.07)
        app.processEvents()
        image = caption(capture_window(win),
            "OPEN THE ROOF. INSPECT THE VAULTS." if cutaway and i >= 3 * FPS else title,
            "THE ACTUAL APP / CHOOSE A STYLE, SET PROPORTIONS, GENERATE AND INSPECT",
            "ROOF CUTAWAY PRESERVES THE COMPLETE EXPORT" if cutaway else "A DESKTOP WORKBENCH + AN AGENT-READY BUILD SYSTEM",
            chapter)
        if i == 2 * FPS:
            image.save(out / (name + ".png"))
            if not cutaway:
                image.save(out / "workbench.png")
        if cutaway and i == 6 * FPS:
            image.save(out / "workbench-cutaway.png")
        writer.write(image)
    assert original == {b.block_id for b in win.scene.blocks}
    path = writer.close()
    win.close()
    app.processEvents()
    return path


def capital_clip(app, out):
    scene = Scene(use_collision=False)
    scene.support_check_mode = "OFF"
    spec = ORDER_SPECS["Ionic"]
    top = _build_column(scene, spec, 0, 0, 0, .5, spec.shaft_height_ratio,
        spec.base_height_ratio, spec.base_width_ratio,
        spec.capital_height_ratio, spec.capital_width_ratio)
    view = ViewportWidget()
    view.resize(1840, 846)
    view.show()
    app.processEvents()
    view.display_meshes(scene.blocks)
    app.processEvents()
    points = np.vstack([b.mesh.vertices for b in scene.blocks])
    focus = points[points[:, 1] > top - spec.capital_height_ratio - .48]
    frame_cathedral(view.plotter, focus, (-.55, .2, -2.4), margin=1.15)
    target = np.array(view.plotter.camera.focal_point)
    vector = np.array(view.plotter.camera.position) - target
    writer = Writer(out, "04-ionic-craft")
    for i in range(5 * FPS):
        angle = math.radians((i / (5 * FPS - 1) - .5) * 12)
        rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        view.plotter.camera.position = target + rotation @ vector
        view.plotter.render()
        image = caption(view.plotter.screenshot(return_img=True),
            "IONIC / SCULPTED INTO THE MESH.",
            "SPIRAL VOLUTES / MOULDED CAPITAL / 24 FLUTES WITH FILLETS",
            "REAL PARTS, WITH NAMED ROLES AND CHECKED CONNECTIONS", "04 / 13")
        if i == 2 * FPS:
            image.save(out / "04-ionic-craft.png")
        writer.write(image)
    path = writer.close()
    view.close()
    app.processEvents()
    return path


def terminal(out, receipt):
    assert receipt["status"] == "pass" and receipt["count"] == 6
    for record in receipt["buildings"]:
        assert "construction" in record["artifacts"]
    lines = [
        "> python -m blockn agent build examples/through-the-ages.json",
        "    --output builds/through-the-ages --format glb --format obj", "",
        "STATUS       " + receipt["status"].upper(),
        "BUILDINGS    " + str(receipt["count"]), "",
        "DORIC TEMPLE / IONIC STOA / CORINTHIAN ROTUNDA",
        "ROMAN BATHS / AMPHITHEATRE / GOTHIC CATHEDRAL", "",
        "buildings/              individual GLB + OBJ assets",
        "series.glb              the combined kitbash scene",
        "recipe.json             replayable assembly request",
        "*.construction.json     parts, joints and assembly stages",
        "*.quality.json          machine-readable geometry checks", "",
        "> python -m blockn craft       # discover construction rules",
    ]
    writer = Writer(out, "11-agent-export")
    for i in range(7 * FPS):
        image = Image.new("RGB", (1840, 846), "black")
        draw = ImageDraw.Draw(image)
        visible = min(len(lines), int(i / FPS * 4) + 1)
        for row, line in enumerate(lines[:visible]):
            draw.text((35, 55 + row * 43), line, font=font(24), fill=GREEN if row in (0, 3, 4, 15) else DIM)
        image = caption(image, "ASK YOUR AGENT FOR A BUILDING KIT.",
            "ACTUAL EXPORT RECEIPT / PRESENTED AS A TERMINAL SEQUENCE",
            "PORTABLE SKILL + PLUGIN / NO DESKTOP NEEDED FOR EXPORT", "11 / 13")
        if i == 6 * FPS:
            image.save(out / "agent-export.png")
        writer.write(image)
    return writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "builds/masonry-upgrade/showcase")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assets = out.parent / "through-the-ages"
    receipt = json.loads((assets / "batch.manifest.json").read_text())
    assert receipt["status"] == "pass"
    results = {}
    for record in receipt["buildings"]:
        result = rebuild(load_document(assets / record["artifacts"]["architecture"]))
        assert result.quality.status == "pass", record["id"]
        results[record["id"]] = result
    district = rebuild(load_document(assets / receipt["series"]["architecture"]))
    assert district.quality.status == "pass"
    basilica = build(BuildRequest(product="classical-basilica", variant="ionic", detail="presentation", seed=73))
    assert basilica.quality.status == "pass"
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    clips = []
    def scene(name, result, seconds, title, sub, footer, number, **kwargs):
        clips.append(model_clip(app, out, name, result.scene, seconds, title, sub, footer, number, **kwargs))
    scene("01-product-intro", results["doric-temple"], 6,
        "MEET BLOCKS.", "A PROCEDURAL 3D ARCHITECTURE WORKBENCH.",
        "MAKE BUILDING ASSETS FOR GAME WORLDS, SCENES AND KITBASH LIBRARIES.", "01 / 13", direction=(-1.8, .8, -2.2))
    clips.append(workbench(app, out, "classical-building", "Ionic", 6, "02-workbench",
        "CHOOSE A STYLE. BUILD YOUR SCENE.", "02 / 13"))
    scene("03-doric-craft", results["doric-temple"], 5,
        "DORIC / THE CONSTRUCTION MATTERS.",
        "BASELESS FLUTED SHAFTS / SHAPED ECHINUS / GROOVED TRIGLYPHS",
        "STEPPED STONE PLATFORM / BEAMS THAT BEAR ON THEIR COLUMNS", "03 / 13", direction=(-1.5, .7, -2.3), angle=12)
    clips.append(capital_clip(app, out))
    for name, key, title, sub, chapter, direction in [
        ("05-rotunda", "roman-rotunda", "ROTUNDA / AN OPEN SKY ABOVE.", "HOLLOW DOME / OPEN OCULUS / ANNULAR BEARING", "05 / 13", (-1.4, 1.8, -1.7)),
        ("06-baths", "roman-baths", "BATHS / ROOMS UNDER VAULTS.", "GROIN-VAULTED HALL / DOMED CALDARIUM / SUPPORTED WINGS", "06 / 13", (-1.5, 1.4, -1.9)),
        ("07-amphitheatre", "amphitheatre", "AMPHITHEATRE / FOLLOW THE CURVE.", "ELLIPTICAL SEATING COURSES / RADIAL ACCESS STAIRS", "07 / 13", (-1.4, 2.1, -1.7)),
    ]:
        scene(name, results[key], 4, title, sub, "ARCHITECTURAL FORMS, AS REUSABLE 3D PARTS", chapter, direction=direction)
    scene("08-basilica", basilica, 4, "BASILICA / A CONNECTED ASSEMBLY.",
        "MITRED APSE WALLS / PITCHED NAVE / CLOSED ROOF SHELLS",
        "GREEK ORDERS TO ROMAN MASONRY AND LATER REVIVALS", "08 / 13", direction=(-1.5, 1.1, 2))
    scene("09-gothic", results["gothic-cathedral"], 6,
        "GOTHIC / A NEW SCALE OF DETAIL.", "PIERCED OPENINGS / ROSE TRACERY / RIBBED VAULTS / FLYING BUTTRESSES",
        "HIGH GOTHIC / EARLY ENGLISH / RAYONNANT", "09 / 13")
    clips.append(workbench(app, out, "gothic-cathedral", "Doric", 8, "10-inspection",
        "DESIGN OUTSIDE. INSPECT INSIDE.", "10 / 13", cutaway=True))
    clips.append(terminal(out, receipt))
    scene("12-kitbash", district, 7, "SIX BUILDINGS. ONE REPLAYABLE KIT.",
        "INDIVIDUAL ASSETS + COMBINED SCENE + RECIPE + CONSTRUCTION REPORTS",
        "THE GEOMETRY ON SCREEN IS RELOADED FROM THE EXPORTED DOCUMENTS", "12 / 13", direction=(-.3, 1.8, -2.2), angle=8)
    scene("13-end", results["gothic-cathedral"], 4, "FROM STYLE TO SCENE.",
        "BLOCKS / REQUEST. GENERATE. INSPECT. KITBASH. EXPORT.",
        "BLACK SCREEN. GREEN PHOSPHOR. PRACTICAL 3D OUTPUT.", "13 / 13", direction=(-1.9, .7, -1.3), angle=10)
    concat = out / "concat.txt"
    concat.write_text("".join("file '" + p.name + "'\n" for p in clips))
    silent = out / "blocks-introduction-and-craft-silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(silent)], check=True)
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(silent)], text=True))
    score = render_score(out, duration)
    final = out / "blocks-introduction-and-craft-peaceful.mp4"
    mux_score(silent, out / score["file"], final)
    source_files = ("core/classical_orders.py", "core/temple_masonry.py", "core/architecture/classical_forms.py", "core/architecture/cathedral.py", "core/architecture/construction.py", "gui/viewport.py", "scripts/render_product_showcase.py", "scripts/render_peaceful_score.py")
    manifest = {"video": final.name, "sha256": hashlib.sha256(final.read_bytes()).hexdigest(), "seconds": duration,
        "resolution": [W, H], "fps": FPS,
        "capture": "Actual Qt/VTK app and geometry rendered offline; not an interactive FPS measurement",
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "contains_working_tree_changes": True,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
        "audio_master": score,
        "preview_images": ["01-product-intro.png", "04-ionic-craft.png", "workbench.png", "workbench-cutaway.png", "12-kitbash.png"],
        "agent_receipt": receipt}
    (out / "showcase-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "x-post.txt").write_text("Meet Blocks: a green-phosphor 3D architecture workbench.\n\nFrom Doric and Ionic columns to Roman vaults and Gothic cathedrals. Build in the app, or ask an agent for a replayable kit with GLB/OBJ assets and construction reports.\n\nBlack screen. Green wireframe. Practical output.\n")
    print(json.dumps({"video": str(final), "seconds": duration, "sha256": manifest["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
