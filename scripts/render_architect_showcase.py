"""Render real Blocks geometry and Qt controls into an X-ready showcase.

Requires the GUI extra, Pillow, FFmpeg and a working Qt display. Frame rendering
is offline: the finished video's frame rate is not an interactive FPS claim.
Run from the repository root after tests: python3 scripts/render_architect_showcase.py
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QApplication

from core.architecture import BuildRequest, build, load_document, rebuild
from gui.camera import frame_cathedral
from gui.main_window import MainWindow
from gui.viewport import ViewportWidget

W, H, FPS = 1920, 1080, 30
GREEN, DIM, RULE = "#00ff41", "#00a92c", "#004a15"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


@lru_cache(maxsize=32)
def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def caption(image, title, sub, footer, number):
    canvas = Image.new("RGB", (W, H), "black")
    image = Image.fromarray(image).convert("RGB") if isinstance(image, np.ndarray) else image.copy().convert("RGB")
    image.thumbnail((W - 80, H - 234))
    canvas.paste(image, ((W - image.width) // 2, 156 + (H - 234 - image.height) // 2))
    draw = ImageDraw.Draw(canvas)
    draw.text((60, 22), "BLOCKS  /  ARCHITECTURE SYSTEM", font=font(18), fill=DIM)
    draw.text((58, 58), title, font=font(43, True), fill=GREEN)
    draw.text((61, 117), sub, font=font(21), fill=DIM)
    draw.line((60, 151, W - 60, 151), fill=RULE)
    draw.line((60, H - 68, W - 60, H - 68), fill=RULE)
    draw.text((60, H - 46), footer, font=font(20), fill=DIM)
    draw.text((W - 170, H - 46), number, font=font(20), fill=GREEN)
    # Apply the same green-phosphor video grade to native Qt dock glyphs.
    # Geometry and controls remain the actual captured frame.
    pixels = np.asarray(canvas).copy()
    intensity = np.maximum(pixels[:, :, 1], pixels[:, :, 0])
    pixels[:, :, 0] = 0
    pixels[:, :, 1] = intensity
    pixels[:, :, 2] = (intensity.astype(np.float32) * (65 / 255)).astype(np.uint8)
    return Image.fromarray(pixels)


class Writer:
    def __init__(self, out, name):
        self.path = out / (name + ".mp4")
        self.log = (out / (name + ".ffmpeg.log")).open("wb")
        self.process = subprocess.Popen([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-threads", "4", str(self.path),
        ], stdin=subprocess.PIPE, stderr=self.log)
        self.frames = 0

    def write(self, image):
        self.process.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        result = self.process.wait()
        self.log.close()
        if result:
            raise RuntimeError(f"FFmpeg failed for {self.path}; see its log")
        print(f"{self.path.name}: {self.frames} frames", flush=True)
        return self.path


def capture_window(win):
    """Capture Qt widgets and their embedded VTK framebuffer at the same state."""
    win.viewport.plotter.render()
    buffer = win.grab().toImage().convertToFormat(4)
    pointer = buffer.bits()
    pointer.setsize(buffer.byteCount())
    array = np.frombuffer(pointer, dtype=np.uint8).reshape(buffer.height(), buffer.bytesPerLine() // 4, 4)
    image = Image.fromarray(array[:, :buffer.width(), [2, 1, 0]].copy())
    widget = win.viewport.plotter.interactor
    point = widget.mapTo(win, QPoint(0, 0))
    frame = Image.fromarray(win.viewport.plotter.screenshot(return_img=True))
    if frame.size != (widget.width(), widget.height()):
        frame = frame.resize((widget.width(), widget.height()))
    image.paste(frame, (point.x(), point.y()))
    return image


def model_clip(app, out, name, scene, seconds, title, sub, footer, number,
               direction=(-2.1, .65, -1.25), angle=18, cutaway=False, west_detail=False):
    viewport = ViewportWidget()
    viewport.resize(1840, 846)
    viewport.show()
    app.processEvents()
    blocks = scene.blocks
    if cutaway:
        roofs = {"nave_roof", "aisle_roof", "transept_roof", "apse_roof", "roof_seam", "ridge", "glazing"}
        blocks = [b for b in blocks if b.parameters.get("component_role") not in roofs]
    viewport.display_meshes(blocks)
    app.processEvents()
    points = np.vstack([b.mesh.vertices for b in blocks])
    focus = points
    if west_detail:
        lo, hi = points.min(axis=0), points.max(axis=0)
        focus = points[(points[:, 1] < lo[1] + (hi[1] - lo[1]) * .51)
                       & (points[:, 2] < lo[2] + (hi[2] - lo[2]) * .14)]
    frame_cathedral(viewport.plotter, focus, direction, margin=1.10)
    start = viewport.plotter.camera.position
    target = np.array(viewport.plotter.camera.focal_point)
    vector = np.array(start) - target
    writer = Writer(out, name)
    for i in range(seconds * FPS):
        t = i / max(1, seconds * FPS - 1)
        a = math.radians((t - .5) * angle)
        rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        viewport.plotter.camera.position = target + rotation @ vector
        viewport.plotter.render()
        image = caption(viewport.plotter.screenshot(return_img=True), title, sub, footer, number)
        if i == seconds * FPS // 2:
            image.save(out / (name + ".png"))
        writer.write(image)
    path = writer.close()
    viewport.close()
    app.processEvents()
    return path


def app_clip(app, out):
    win = MainWindow()
    win.resize(1840, 846)
    win.show()
    panel = win.generation_panel
    panel.classical_form_combo.setCurrentIndex(panel.classical_form_combo.findData("gothic-cathedral"))
    panel.form_detail.setCurrentText("presentation")
    panel.seed_input.setText("73")
    panel.generate_button.click()
    while win._generation_thread is not None:
        app.processEvents()
        time.sleep(.01)
    assert len(win.scene.blocks) > 1000, win.statusBar().currentMessage()
    original_ids = {b.block_id for b in win.scene.blocks}
    writer = Writer(out, "05-workbench")
    for i in range(10 * FPS):
        if i == 3 * FPS:
            panel.cathedral_cutaway.setChecked(True)
            app.processEvents()
            points = np.vstack([b.mesh.vertices for b in win.viewport._batch_blocks.values()])
            frame_cathedral(win.viewport.plotter, points, (-1.3, 2.2, -.7), margin=1.09)
        win.viewport.plotter.camera.Azimuth(.06)
        app.processEvents()
        image = caption(capture_window(win),
                        "DESIGN THE BUILDING." if i < 3 * FPS else "OPEN THE ROOF. INSPECT THE VAULTS.",
                        "REAL WORKBENCH / STYLE, PROPORTIONS, BAYS AND SEED",
                        "THE COMPLETE BUILDING REMAINS IN THE EXPORT SCENE", "05 / 08")
        if i in (2 * FPS, 6 * FPS):
            image.save(out / ("workbench.png" if i < 3 * FPS else "workbench-cutaway.png"))
        writer.write(image)
    assert {b.block_id for b in win.scene.blocks} == original_ids
    path = writer.close()
    win.close()
    app.processEvents()
    return path


def terminal_clip(out, receipt):
    assert receipt["count"] == 3 and receipt["status"] == "pass", receipt
    command = "python3 -m blockn agent build examples/cathedral-kitbash.json"
    lines = ["> " + command, "    --output builds/cathedral-studies --format glb --format obj", "",
             "STATUS       " + receipt["status"].upper(), "BUILDINGS    " + str(receipt["count"]), "",
             "high-gothic-precinct  / cathedral + podium + approach",
             "english-crossing     / dominant central spire",
             "rayonnant-study      / tracery + nine-sided apse", "",
             "buildings/  -> individual GLB + OBJ assets", "series.glb  -> complete kitbashed scene",
             "recipe.json -> replayable assembly request", "batch.manifest.json -> machine-readable receipt", "",
             "PER ASSET: ARCHITECTURE + GEOMETRY + QUALITY + MANIFEST"]
    writer = Writer(out, "06-agent-export")
    for i in range(8 * FPS):
        image = Image.new("RGB", (1840, 846), "black")
        draw = ImageDraw.Draw(image)
        visible = min(len(lines), int(i / FPS * 4) + 1)
        for row, line in enumerate(lines[:visible]):
            if row == 0:
                line = line[:min(len(line), int(i / FPS * 85) + 3)]
            draw.text((35, 55 + row * 42), line, font=font(24), fill=GREEN if row in (0, 3, 4, 15) else DIM)
        image = caption(image, "ASK FOR A KIT. GET USABLE ASSETS.", "AGENT MODE / ACTUAL EXPORT RECEIPT, PRESENTED AS A TERMINAL SEQUENCE",
                        "GLB / OBJ / STL / PLY  +  CANONICAL JSON", "06 / 08")
        if i == 6 * FPS:
            image.save(out / "agent-export.png")
        writer.write(image)
    return writer.close()


def soundtrack(out, seconds):
    """Original synthesized soundtrack, no third-party recordings or samples."""
    sr = 48000
    audio = np.zeros(int(seconds * sr), np.float32)
    rng = np.random.default_rng(73)
    notes = [130.81, 196.00, 261.63, 293.66, 146.83, 220.00, 293.66, 349.23]
    for beat in range(int(seconds * 2.4)):
        start = int(beat / 2.4 * sr)
        n = min(int(.7 * sr), len(audio) - start)
        t = np.arange(n) / sr
        f = notes[beat % len(notes)]
        env = (1 - np.exp(-t * 90)) * np.exp(-t * 5)
        audio[start:start+n] += .04 * env * (np.sin(2*np.pi*f*t) + .22*np.sin(2*np.pi*f*2*t))
        if beat % 2 == 0:
            audio[start:start+n] += .035 * np.sin(2*np.pi*(43*t + 3*(1-np.exp(-t*20)))) * np.exp(-t*15)
        audio[start:start+n] += rng.normal(0, 1, n) * np.exp(-t*85) * .008
    fade = int(sr * 1.5)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    stereo = np.column_stack((audio, audio*.97))
    with wave.open(str(out / "original-phosphor-score.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((np.clip(stereo, -1, 1)*32767).astype("<i2").tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "builds/cathedral-upgrade/showcase")
    parser.add_argument("--reuse-export", action="store_true", help="Reuse a completed export from this same source revision")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assets = out.parent / "cathedral-studies"
    command = [sys.executable, "-m", "blockn", "agent", "build", "examples/cathedral-kitbash.json", "--output", str(assets), "--format", "glb", "--format", "obj"]
    source_hash = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in (
        "core/architecture/cathedral.py", "core/architectural_geometry.py", "core/architecture/agent.py",
        "core/architecture/compiler.py", "core/architecture/io.py", "examples/cathedral-kitbash.json"))).hexdigest()
    if not args.reuse_export:
        print("Exporting the three-building cathedral kit with GLB and OBJ...", flush=True)
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
        (out / "agent-receipt.json").write_text(completed.stdout)
        (out / "export-source.sha256").write_text(source_hash + "\n")
    elif (out / "export-source.sha256").read_text().strip() != source_hash:
        raise RuntimeError("Export source changed; choose a fresh output directory and regenerate the kit")
    receipt = json.loads((out / "agent-receipt.json").read_text())
    print("Agent kit export:", receipt["status"], flush=True)
    # Film the exported, reloaded documents. This also verifies their practical
    # replay path and keeps the showcased kit identical to the downloadable kit.
    district = rebuild(load_document(assets / receipt["series"]["architecture"]))
    assert district.quality.status == "pass"
    app = QApplication.instance() or QApplication([])
    results = {style: rebuild(load_document(assets / record["artifacts"]["architecture"]))
               for style, record in zip(("high-gothic", "early-english", "rayonnant"), receipt["buildings"])}
    for result in results.values():
        assert result.quality.status == "pass"
    print("Exported documents reloaded and revalidated. Recording footage...", flush=True)
    clips = []
    clips.append(model_clip(app, out, "01-cathedral-hero", results["high-gothic"].scene, 6,
        "BLOCKS HAS A NEW ARCHITECT.", "CATHEDRALS / CLASSICAL FORMS / REUSABLE ARCHITECTURE",
        "BLACK SCREEN. GREEN PHOSPHOR. REAL GEOMETRY.", "01 / 08"))
    clips.append(model_clip(app, out, "02-high-gothic", results["high-gothic"].scene, 5,
        "HIGH GOTHIC / INSIDE THE DETAIL", "CARVED PORTAL PROFILES / PIERCED OPENINGS / ROSE TRACERY",
        "30 ARCHITECTURAL PRODUCTS  /  10 ARCHITECTS", "02 / 08", direction=(-.30, .12, -2.3), angle=9, west_detail=True))
    clips.append(model_clip(app, out, "03-early-english", results["early-english"].scene, 5,
        "EARLY ENGLISH", "LOWER WEST TOWERS / A DOMINANT CROSSING SPIRE",
        "ONE ARCHITECT. THREE DISTINCT GOTHIC COMPOSITIONS.", "03 / 08"))
    clips.append(model_clip(app, out, "04-rayonnant", results["rayonnant"].scene, 5,
        "RAYONNANT", "ROSE TRACERY / MULTI-LIGHT WINDOWS / POLYGONAL APSE",
        "NAMED MEMBERS  /  DETERMINISTIC SEEDS  /  QUALITY REPORTS", "04 / 08", direction=(1.8, 1.1, 1.6)))
    clips.append(app_clip(app, out))
    # A quick, readable tour of the previously expanded classical vocabulary.
    for i, (product, label) in enumerate([
        ("classical-rotunda", "ROTUNDA"), ("classical-aqueduct", "AQUEDUCT"),
        ("classical-amphitheatre", "AMPHITHEATRE"), ("classical-baths", "BATHS")]):
        result = build(BuildRequest(product=product, detail="presentation", seed=73))
        assert result.quality.status == "pass"
        clips.append(model_clip(app, out, f"classical-{i}", result.scene, 2,
            "A LARGER CLASSICAL VOCABULARY.", label + " / TWELVE EXPANDED FORMS / FIVE CLASSICAL ORDERS",
            "COMPOSE FROM ARCHITECTURAL RULES AND REUSABLE PARTS", "05 / 08", direction=(-1.4, 1.2, -1.7)))
    clips.append(terminal_clip(out, receipt))
    clips.append(model_clip(app, out, "07-kitbash", district.scene, 7,
        "THREE BUILDINGS. ONE REPLAYABLE KIT.", "INDIVIDUAL ASSETS + COMBINED SCENE + RECIPE + REPORTS",
        "HIGH GOTHIC PRECINCT / ENGLISH CROSSING / RAYONNANT STUDY", "07 / 08", direction=(-.3, 1.6, -2.2), angle=9))
    clips.append(model_clip(app, out, "08-end", results["high-gothic"].scene, 4,
        "BUILD SOMETHING WORTH EXPLORING.", "BLOCKS / THE PHOSPHOR ARCHITECTURE WORKBENCH",
        "python3 -m blockn describe gothic-cathedral", "08 / 08", direction=(-1.9, .65, -1.3), angle=10))
    concat = out / "concat.txt"
    concat.write_text("".join("file '" + p.name + "'\n" for p in clips))
    silent = out / "blocks-architects-upgrade-silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(silent)], check=True)
    duration = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(silent)], text=True))
    soundtrack(out, duration)
    final = out / "blocks-architects-upgrade.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent), "-i", str(out / "original-phosphor-score.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(final)], check=True)
    manifest = {"video": final.name, "sha256": hashlib.sha256(final.read_bytes()).hexdigest(), "seconds": duration,
                "resolution": [W, H], "fps": FPS, "capture": "Actual Qt/VTK geometry rendered offline; not an interactive FPS measurement",
                "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "architect_sha256": hashlib.sha256((ROOT / "core/architecture/cathedral.py").read_bytes()).hexdigest(),
                "agent_receipt": receipt}
    (out / "showcase-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"video": str(final), "seconds": duration, "sha256": manifest["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
