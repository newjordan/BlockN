"""Before/after pictures of builds sized to a player.

    python scripts/render_profiles.py norman-keep:square classical-baths --profile giant --out docs/renders

For each product it builds the historic version (no profile) and the version
sized for the profile, walks both with that player, and writes an exterior
view and a cutaway with the walk overlay for each, plus a side-by-side sheet.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from core.architecture import BuildRequest, build  # noqa: E402
from core.architecture.gameplay import walk  # noqa: E402
from core.architecture.profile import resolve  # noqa: E402
from core.architecture.render import legend, render  # noqa: E402
from core.architecture.voxels import profile_pitch, voxelize  # noqa: E402


def one(product, variant, profile, seed, sized, pitch, detail):
    extra = {"variant": variant} if variant else {}
    req = BuildRequest(product=product, seed=seed, detail=detail, **extra, **({"profile": profile} if sized else {}))
    result = build(req)
    player = resolve(profile)
    grid = voxelize(result.document, result.scene, pitch=pitch or profile_pitch(player), profile=player)
    w = walk(grid, player)
    grid.ports = w.ports
    s = w.summary()
    status = {c.code: c.status for c in (result.quality.checks if sized else [])}
    return result, grid, w, player, s, status


def figure_at(grid, w, player):
    if w.ports:
        # The outermost doorway: furthest from the middle of the grid.
        cx = grid.origin[0] + grid.labels.shape[0] * grid.pitch / 2
        cz = grid.origin[2] + grid.labels.shape[2] * grid.pitch / 2
        port = max(w.ports, key=lambda q: (q["position"][0] - cx) ** 2 + (q["position"][2] - cz) ** 2)
        # Stood in the doorway on its floor, so the door reads against the body.
        return {"player": player, "at": tuple(port["position"])}
    o, p = grid.origin, grid.pitch
    return {"player": player, "at": (o[0] + 1.0, o[2] + 1.0)}


def cut_height(grid, w, player):
    """Just above the player's head on the floor they mostly reach inside.

    Falls back to the floor with the most enclosed cells, then to the ground.
    """
    import numpy as np
    for mask in (w.interior & w.reached, w.interior):
        ys = w.nodes[mask, 1]
        if len(ys):
            vals, counts = np.unique(ys, return_counts=True)
            floor = grid.origin[1] + vals[counts.argmax()] * grid.pitch
            return round(floor + player.height + 0.6, 2)
    return round(player.height + 0.6, 2)


def sheet(images, labels, out, cols=2):
    pad, head = 16, 40
    wmax = max(i.width for i in images)
    hmax = max(i.height for i in images)
    rows = math.ceil(len(images) / cols)
    canvas = Image.new("RGB", (cols * (wmax + pad) + pad, rows * (hmax + head + pad) + pad), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    for k, (img, lab) in enumerate(zip(images, labels)):
        r, c = divmod(k, cols)
        x, y = pad + c * (wmax + pad), pad + r * (hmax + head + pad)
        d.text((x, y + 8), lab, fill=(20, 20, 20), font=font)
        canvas.paste(img, (x, y + head))
    canvas.save(out, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("products", nargs="+", help="product or product:variant")
    ap.add_argument("--profile", default="giant")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pitch", type=float)
    ap.add_argument("--detail", default="concept")
    ap.add_argument("--azimuth", type=float, default=35.0)
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--max-cell", type=float, default=0.3,
                    help="draw no finer than this many metres (0 draws every cell)")
    ap.add_argument("--section", action="store_true",
                    help="add a third view per version, cut through the middle front to back, "
                         "so every floor and stair shows")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cell = args.max_cell or None
    index = []
    for spec in args.products:
        product, _, variant = spec.partition(":")
        name = product + (f"-{variant}" if variant else "")
        images, labels, row = [], [], {"product": spec, "profile": args.profile}
        built = {}
        for sized in (True, False):
            t = time.time()
            built[sized] = one(product, variant, args.profile, args.seed, sized, args.pitch, args.detail) + (time.time() - t,)
        # Both versions are cut at the same height: the floor the sized
        # build's player mostly reaches inside.
        _, g1, w1, player, _, _, _ = built[True]
        cut_at = cut_height(g1, w1, player)
        for sized in (False, True):
            result, grid, w, player, s, status, secs = built[sized]
            tag = f"sized for {args.profile}" if sized else "historic (no profile)"
            shut = s["interior_cells"] - s["interior_reached"]
            line = (f"{tag}: interior floor reached {s['interior_reached']}/{s['interior_cells']} cells"
                    f" ({s['interior_reached_walking']} walking)")
            fig = figure_at(grid, w, player)
            ext = render(grid, azimuth=args.azimuth, width=args.width, figure=fig, max_cell=cell)
            cut = render(grid, azimuth=args.azimuth, width=args.width, cut_above=cut_at, walk=w, figure=fig,
                         max_cell=cell)
            images += [ext, cut]
            labels += [f"{tag}, outside", f"{tag}, cut at {cut_at:.1f} m with the {args.profile}'s walk"]
            if args.section:
                mid = grid.origin[2] + grid.labels.shape[2] * grid.pitch / 2
                images.append(render(grid, azimuth=args.azimuth, width=args.width, cut_front_z=mid, walk=w,
                                     max_cell=cell))
                labels.append(f"{tag}, cut through the middle with the walk")
            row["sized" if sized else "historic"] = {"walk": s, "checks": status, "seconds": round(secs, 1),
                                                     "line": line, "unreached_interior_cells": shut, "cut_at": cut_at}
            print(name, line, f"{secs:.0f}s", flush=True)
        path = args.out / f"{name}-{args.profile}.png"
        sheet(images, labels, path, cols=3 if args.section else 2)
        row["image"] = path.name
        index.append(row)
    # Merge into an existing index, one row per product.
    path = args.out / f"index-{args.profile}.json"
    rows = {r["product"]: r for r in (json.loads(path.read_text()) if path.exists() else [])}
    rows.update({r["product"]: r for r in index})
    path.write_text(json.dumps(sorted(rows.values(), key=lambda r: r["product"]), indent=2) + "\n")
    print(legend())


if __name__ == "__main__":
    main()
