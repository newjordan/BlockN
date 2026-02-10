"""Minimal CLI for generating and exporting LegoGen scenes.

Usage examples:
  python3 cli.py generate --seed 123 --out builds/classical.glb
  python3 cli.py generate --classical-order Doric --classical-preset monumental --classical-front-columns 8 --classical-pediment-slope 1.1 --summary
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np

from core.scene import Scene
from core.generators.classical import generate_classical_building, get_classical_snapshot_metrics
from core.geometry import optimize_blocks
from core.io import export_meshes_to_glb, save_scene_to_json


def _build_scene(
    width: int,
    height: int,
    depth: int,
    seed: int | None,
    classical_order: str,
    classical_preset: str,
    classical_front_columns: int | None,
    classical_side_columns: int | None,
    classical_pediment_slope: float,
    classical_entablature_depth: float,
    classical_side_roof_overhang: float,
    classical_foundation_apron: bool,
    classical_wrap_min_front_span: float | None,
    classical_wrap_min_side_span: float | None,
    classical_wrap_front_coverage: float | None,
):
    scene = Scene(use_collision=False)
    podium_height = max(1, height // 3)
    generate_classical_building(
        scene,
        width=width,
        depth=depth,
        podium_height_actual=podium_height,
        seed=seed,
        flags={"foundation_apron": classical_foundation_apron},
        order_override=classical_order,
        preset_override=classical_preset,
        front_column_override=classical_front_columns,
        side_column_override=classical_side_columns,
        pediment_slope_multiplier=classical_pediment_slope,
        entablature_depth_multiplier=classical_entablature_depth,
        side_roof_overhang_multiplier=classical_side_roof_overhang,
        wrap_min_podium_span_x_override=classical_wrap_min_front_span,
        wrap_min_podium_span_z_override=classical_wrap_min_side_span,
        wrap_front_coverage_ratio_override=classical_wrap_front_coverage,
        clear_scene=True,
    )
    return scene


def _export(scene: Scene, output: Path | None, json_path: Path | None, optimize: bool, summary: bool):
    meshes: List = []
    if optimize:
        meshes = optimize_blocks(scene.blocks)
    else:
        meshes = scene.get_all_meshes()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        export_meshes_to_glb(meshes, str(output))

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        save_scene_to_json(scene, str(json_path))

    if summary:
        all_vertices = []
        for mesh in meshes:
            if mesh is not None and mesh.vertices.size:
                all_vertices.append(mesh.vertices)
        if all_vertices:
            stacked = np.vstack(all_vertices)
            bounds = (stacked.min(axis=0), stacked.max(axis=0))
        else:
            bounds = (np.array([0, 0, 0]), np.array([0, 0, 0]))
        print(f"Blocks: {len(scene.blocks)} | Meshes: {len(meshes)} | Bounds: min{bounds[0].round(3)} max{bounds[1].round(3)}")
        if hasattr(scene, "last_classical_metrics"):
            metrics = get_classical_snapshot_metrics(scene)
            invariant_status = metrics.get("invariants", {}).get("ok")
            apron_blocks = metrics.get("foundation_apron", {}).get("blocks")
            print(
                "Classical metrics:"
                f" order={metrics.get('order')}"
                f" preset={metrics.get('preset')}"
                f" facade_span={metrics.get('facade_span')}"
                f" front_columns={metrics.get('front_columns')}"
                f" side_columns={metrics.get('side_columns')}"
                f" invariants_ok={invariant_status}"
                f" apron_blocks={apron_blocks}"
            )


def main():
    parser = argparse.ArgumentParser(description="LegoGen CLI (KISS baseline)")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate a classical scene and export to GLB/JSON")
    gen.add_argument("--width", type=int, default=5, help="X dimension")
    gen.add_argument("--height", type=int, default=5, help="Y dimension")
    gen.add_argument("--depth", type=int, default=5, help="Z dimension")
    gen.add_argument(
        "--classical-order",
        choices=["Random", "Doric", "Ionic"],
        default="Random",
        help="Classical mode order override",
    )
    gen.add_argument(
        "--classical-preset",
        choices=["compact", "canonical", "monumental"],
        default="canonical",
        help="Classical proportion preset",
    )
    gen.add_argument("--classical-front-columns", type=int, default=None, help="Override front colonnade count (even)")
    gen.add_argument("--classical-side-columns", type=int, default=None, help="Override side colonnade count (even)")
    gen.add_argument(
        "--classical-pediment-slope",
        type=float,
        default=1.0,
        help="Pediment slope multiplier for classical mode",
    )
    gen.add_argument(
        "--classical-entablature-depth",
        type=float,
        default=1.0,
        help="Entablature depth multiplier for classical mode",
    )
    gen.add_argument(
        "--classical-side-roof-overhang",
        type=float,
        default=1.0,
        help="Side roof overhang multiplier for classical mode",
    )
    gen.add_argument(
        "--classical-foundation-apron",
        action="store_true",
        help="Add optional apron and threshold support blocks for classical stairs/doorways",
    )
    gen.add_argument(
        "--classical-wrap-min-front-span",
        type=float,
        default=None,
        help="Override minimum podium width (X) needed to enable front wrap stairs",
    )
    gen.add_argument(
        "--classical-wrap-min-side-span",
        type=float,
        default=None,
        help="Override minimum podium depth (Z) needed to enable front wrap stairs",
    )
    gen.add_argument(
        "--classical-wrap-front-coverage",
        type=float,
        default=None,
        help="Override required front stair width / podium width ratio for wrap stairs",
    )
    gen.add_argument("--seed", type=int, default=None)
    gen.add_argument("--out", type=Path, help="Output GLB path")
    gen.add_argument("--json", type=Path, help="Optional JSON scene save path")
    gen.add_argument("--optimize", action="store_true", help="Union meshes by color before export")
    gen.add_argument("--summary", action="store_true", help="Print counts and bounds")

    args = parser.parse_args()

    if args.command == "generate":
        scene = _build_scene(
            args.width,
            args.height,
            args.depth,
            args.seed,
            args.classical_order,
            args.classical_preset,
            args.classical_front_columns,
            args.classical_side_columns,
            args.classical_pediment_slope,
            args.classical_entablature_depth,
            args.classical_side_roof_overhang,
            args.classical_foundation_apron,
            args.classical_wrap_min_front_span,
            args.classical_wrap_min_side_span,
            args.classical_wrap_front_coverage,
        )
        _export(scene, args.out, args.json, args.optimize, args.summary)


if __name__ == "__main__":
    main()
