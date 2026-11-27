"""Render every historic tradition into one phosphor contact sheet.

Each cell is the real workbench viewport drawing a real build: the same batch
path, feature-edge traces and camera framing as the desktop application. A
build that is not a clean ``pass`` is never drawn, so the sheet cannot show a
building that would fail its own quality gate.

Requires the GUI extra (PyQt5, pyvista, vtk) and Pillow. No display is needed:

    python3 scripts/render_historic_sheet.py
    python3 scripts/render_historic_sheet.py --products chinese-hall --detail standard --output /tmp/hall.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import QApplication

from core.architecture import BuildRequest, build
from core.architecture import historic
from gui.camera import frame_cathedral
from gui.viewport import ViewportWidget

GREEN, RULE = "#00ff41", "#004a15"
CELL, COLUMNS, LABEL, GAP = (600, 400), 4, 24, 8
# Looking down on the front (-Z) and the left (-X) face; front is -Z in every architect.
VIEW = (-2.0, 1.25, -2.6)


def label_font(size):
    """A monospace label font from wherever this machine keeps one."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
        Path(__file__).resolve().parent.parent / "public/historic/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size)


def render(viewport, result, direction, margin=1.06):
    """Draw one compiled scene and return it as an image, framed on its own geometry."""
    viewport.display_meshes(result.scene.blocks)
    shown = list(viewport._batch_blocks.values()) or list(result.scene.blocks)
    points = np.vstack([block.mesh.vertices for block in shown if block.mesh is not None and not block.mesh.is_empty])
    frame_cathedral(viewport.plotter, points, direction, margin=margin)
    viewport.plotter.render()
    return Image.fromarray(viewport.plotter.screenshot(return_img=True)).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "public/historic/contact-sheet.png")
    parser.add_argument("--detail", choices=historic.DETAILS, default="concept")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--products", nargs="*", help="Only these products; default is every historic product")
    parser.add_argument("--margin", type=float, default=1.06, help="Camera margin around the building")
    args = parser.parse_args()

    wanted = {name.lower() for name in (args.products or ())}
    cells = []
    for name, (module, form) in historic.styles().items():
        if wanted and name not in wanted:
            continue
        for variant in form.variants:
            cells.append((name, module, variant))
    if not cells:
        parser.error("no historic product matches --products; choose from: "
                     + ", ".join(historic.styles()))

    app = QApplication.instance() or QApplication([])
    viewport = ViewportWidget()
    viewport.resize(*CELL)
    viewport.show()
    app.processEvents()
    font = label_font(13)
    tiles = []
    for name, module, variant in cells:
        result = build(BuildRequest(product=name, variant=variant, detail=args.detail, seed=args.seed))
        status = result.quality.status
        failed = [check.code for check in result.quality.checks if check.status != "pass"]
        if status != "pass":
            viewport.close()
            raise SystemExit(f"{name} / {variant} is {status}: {', '.join(failed)}; refusing to draw it")
        image = render(viewport, result, getattr(module, "VIEW", VIEW), args.margin)
        count = len(result.document.placements)
        tiles.append((f"{name} / {variant} ({count} members, {status})", image))
        print(f"{name} / {variant}: {count} members, {status}", flush=True)
    viewport.close()

    rows = -(-len(tiles) // COLUMNS)
    sheet = Image.new("RGB", (COLUMNS * (CELL[0] + GAP) + GAP, rows * (CELL[1] + LABEL + GAP) + GAP), "black")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(tiles):
        x = GAP + index % COLUMNS * (CELL[0] + GAP)
        y = GAP + index // COLUMNS * (CELL[1] + LABEL + GAP)
        draw.text((x + 2, y), label, font=font, fill=GREEN)
        draw.line((x, y + LABEL - 6, x + CELL[0], y + LABEL - 6), fill=RULE)
        sheet.paste(image, (x, y + LABEL))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, optimize=True)
    print(f"{args.output}: {len(tiles)} builds at {args.detail} detail, seed {args.seed}")


if __name__ == "__main__":
    main()
