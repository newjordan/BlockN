"""Exterior pictures of builds at a chosen detail, for comparing by eye.

    python scripts/render_gallery.py classical-theatre gothic-cathedral --detail presentation --out docs/renders/gallery
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.architecture import BuildRequest, build  # noqa: E402
from core.architecture.render import render  # noqa: E402
from core.architecture.voxels import voxelize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("products", nargs="+", help="product or product:variant")
    ap.add_argument("--detail", default="presentation")
    ap.add_argument("--profile")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pitch", type=float, default=0.2)
    ap.add_argument("--azimuth", type=float, nargs="+", default=[35.0, 215.0])
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.products:
        product, _, variant = spec.partition(":")
        extra = {"variant": variant} if variant else {}
        if args.profile:
            extra["profile"] = args.profile
        t = time.time()
        result = build(BuildRequest(product=product, seed=args.seed, detail=args.detail, **extra))
        grid = voxelize(result.document, result.scene, pitch=args.pitch)
        name = product + (f"-{variant}" if variant else "") + f"-{args.detail}" + (f"-{args.profile}" if args.profile else "")
        for az in args.azimuth:
            render(grid, azimuth=az, width=args.width, max_cell=None,
                   title=f"{spec}, {args.detail} detail, seed {args.seed}" + (f", sized for {args.profile}" if args.profile else "")
                   ).save(args.out / f"{name}-az{int(az)}.png", optimize=True)
        print(name, len(result.document.placements), "placements", f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
