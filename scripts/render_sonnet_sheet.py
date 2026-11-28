"""Render variations on one theme, or an anthology of every sonnet form, as a phosphor sheet.

Each cell is a different verse of the same form: the rules are the same in
every cell and no verse is repeated. The cells are dealt by
``core.architecture.sonnet.variations`` and drawn by the same viewport as the
historic contact sheet. A build that is not a clean ``pass`` is never drawn.

Requires the GUI extra (PyQt5, pyvista, vtk) and Pillow. No display is needed:

    python3 scripts/render_sonnet_sheet.py muscovite-church --count 12
    python3 scripts/render_sonnet_sheet.py mughal-tomb --count 8 --variant taj --hold dome=onion
    python3 scripts/render_sonnet_sheet.py --anthology --count 4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw
from PyQt5.QtWidgets import QApplication

from core.architecture import build, historic, sonnet
from gui.viewport import ViewportWidget
from render_historic_sheet import CELL, GAP, GREEN, LABEL, RULE, VIEW, label_font, render

HEAD = 44           # sheet title band
WIDE = 74           # characters of label that fit over one cell


def caption(request, prefix=""):
    """Seed, variant and the verse, shortened to fit over the cell."""
    text = f"{prefix}{request.seed:04d} {request.variant}: " + " / ".join(sonnet.verse(request).values())
    return text if len(text) <= WIDE else text[:WIDE - 1] + "…"


def deal(args):
    """(row title, requests) for each row group of the sheet."""
    hold = dict(item.split("=", 1) for item in args.hold or ())
    if args.anthology:
        names = [name for name, (_, form) in historic.styles().items() if form.rules.get("sonnet")]
        return [(name, sonnet.variations(name, args.count, args.seed, detail=args.detail)) for name in names]
    if not args.product:
        raise SystemExit("name a product, or pass --anthology")
    return [(args.product, sonnet.variations(args.product, args.count, args.seed, variant=args.variant,
                                              detail=args.detail, hold=hold))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("product", nargs="?")
    parser.add_argument("--anthology", action="store_true", help="One row of verses for every sonnet form")
    parser.add_argument("--count", type=int, default=12, help="Verses per form")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant")
    parser.add_argument("--hold", action="append", metavar="LINE=ANSWER")
    parser.add_argument("--detail", choices=historic.DETAILS, default="concept")
    parser.add_argument("--columns", type=int, help="Cells per row (default 4, or --count for an anthology)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--margin", type=float, default=1.06)
    args = parser.parse_args()
    groups = deal(args)
    columns = args.columns or (args.count if args.anthology else 4)
    name = "anthology" if args.anthology else sonnet.canonical_product_name(args.product)
    output = args.output or ROOT / f"public/sonnets/{name}.png"

    app = QApplication.instance() or QApplication([])
    viewport = ViewportWidget()
    viewport.resize(*CELL)
    viewport.show()
    app.processEvents()
    tiles = []
    for product, requests in groups:
        module = historic.style_for(product)[0]
        for request in requests:
            result = build(request)
            if result.quality.status != "pass":
                viewport.close()
                failed = [check.code for check in result.quality.checks if check.status != "pass"]
                raise SystemExit(f"{product} seed {request.seed} is {result.quality.status}: {', '.join(failed)}")
            label = caption(request, f"{product.split('-')[0]} " if args.anthology else "")
            tiles.append((label, render(viewport, result, getattr(module, "VIEW", VIEW), args.margin)))
            print(f"{product} {caption(request)}", flush=True)
    viewport.close()

    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * (CELL[0] + GAP) + GAP, HEAD + rows * (CELL[1] + LABEL + GAP) + GAP), "black")
    draw = ImageDraw.Draw(sheet)
    verses = sum(sonnet.verse_count(product) for product, _ in groups)
    title = (f"ARCHITECTURE SONNETS: {len(groups)} forms, {verses} verses, {len(tiles)} shown" if args.anthology else
             f"{name.upper()}: {len(tiles)} of {verses} verses, one form, no verse repeated")
    draw.text((GAP + 2, 10), title, font=label_font(22), fill=GREEN)
    draw.line((GAP, HEAD - 4, sheet.width - GAP, HEAD - 4), fill=GREEN)
    font = label_font(13)
    for index, (label, image) in enumerate(tiles):
        x = GAP + index % columns * (CELL[0] + GAP)
        y = HEAD + GAP + index // columns * (CELL[1] + LABEL + GAP)
        draw.text((x + 2, y), label, font=font, fill=GREEN)
        draw.line((x, y + LABEL - 6, x + CELL[0], y + LABEL - 6), fill=RULE)
        sheet.paste(image, (x, y + LABEL))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    print(f"{output}: {len(tiles)} builds at {args.detail} detail from seed {args.seed}")


if __name__ == "__main__":
    main()
