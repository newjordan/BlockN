"""BlockN command-line interface.

BlockN has two compatible generation paths:

* ``make`` / ``build`` use the semantic architecture pipeline and emit a
  canonical document, compiled scene, quality report, and named mesh assets.
* ``generate`` preserves the original classical-building command surface.

Examples::

    blockn catalog
    blockn make wall --openings mixed --bundle builds/wall
    blockn make structural-grid --bays 4 --levels 3 --bundle builds/frame
    blockn build request.json --bundle builds/request
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable, List, Mapping

import numpy as np

from core.architecture.compiler import ArchitectureCompileError, BuildResult, compile_document
from core.architecture.io import (
    SUPPORTED_MESH_FORMATS,
    export_bundle,
    export_document,
    export_manifest,
    export_mesh_artifact,
    export_quality,
)
from core.architecture.schema import BuildDocument, BuildRequest
from core.architecture.service import (
    build as architecture_build,
    describe_product as architecture_describe_product,
    list_architects,
    list_products as architecture_products,
)
from core.generators.classical import generate_classical_building, get_classical_snapshot_metrics
from core.geometry import optimize_blocks
from core.io import export_meshes_to_glb, save_scene_to_json
from core.primitives import primitive_catalog
from core.scene import Scene


VERSION = "0.4.0"


def _architecture_products() -> list[dict[str, Any]]:
    return architecture_products()


def _architecture_product(name: str) -> dict[str, Any]:
    return architecture_describe_product(name)


def _build_architecture(request: BuildRequest) -> BuildResult:
    return architecture_build(request)


def _parse_parameter(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("parameters must use KEY=VALUE syntax")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("parameter key cannot be empty")
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed = raw_value
    return key, parsed


def _parameter_dict(values: Iterable[tuple[str, Any]] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values or ():
        result[key] = value
    return result


def _request_from_args(args: argparse.Namespace) -> BuildRequest:
    descriptor = _architecture_product(args.product)
    architect = descriptor.get("architect", {})
    architect_id = descriptor.get("architect_id") or architect.get("id") or "primitive"
    request = BuildRequest(
        architect=str(architect_id),
        product=str(descriptor["name"]),
        seed=args.seed,
        width=args.width,
        height=args.height,
        depth=args.depth,
        style=args.style,
        detail=args.detail,
        variant=args.variant,
        bays=args.bays,
        levels=args.levels,
        openings=args.openings,
        parameters=_parameter_dict(args.param),
    )
    if getattr(args, "age", None) is not None:
        request.age = args.age
    if getattr(args, "profile", None):
        text = args.profile.strip()
        request.profile = json.loads(text) if text.startswith("{") else text
    return request


def _handle_block_voxels(args: argparse.Namespace) -> int:
    from core.architecture.blockcraft import build as build_blocks
    from core.architecture.gameplay import walk

    profile = args.profile or "blocky"
    if isinstance(profile, str) and profile.strip().startswith("{"):
        profile = json.loads(profile)
    town_report = None
    if args.product in ("town", "castle-town"):
        from core.architecture.blocktown import load_heights, town, town_walk

        heights = load_heights(args.heightmap) if args.heightmap else None
        built = town(profile, args.seed, heights=heights, terrain=args.terrain)
    else:
        built = build_blocks(args.product, profile, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = built.grid(1).save(out, args.name)
    written = {key: str(path) for key, path in paths.items()}
    target = out / f"{args.name}.mcfunction"
    target.write_text(built.to_mcfunction())
    written["mcfunction"] = str(target)
    if getattr(built, "town", None) is not None:
        result, town_report = town_walk(built)
    else:
        result = walk(built.grid(built.walk_sub()), built.player)
    manifest = built.manifest()
    manifest["walk"] = result.summary()
    if town_report is not None:
        manifest["town"] = {k: v for k, v in town_report.items() if k != "summary"}
        manifest["town"]["counts"] = built.town["counts"]
        manifest["town"]["streets"] = built.town["streets"]
    manifest["measured_ports"] = result.ports
    target = out / f"{args.name}.blocks.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    written["blocks"] = str(target)
    if args.plan is not None:
        target = out / f"{args.name}.plan.txt"
        target.write_text(built.grid(1).plan_ascii(args.plan))
        written["plan"] = str(target)
    if args.render:
        from core.architecture.render import render

        target = out / f"{args.name}.png"
        render(built.grid(4, visual=True, ground_layer=town_report is None), block=4, ground=town_report is None,
               width=1500 if town_report is not None else 1100,
               title=f"{args.product}, block mode, seed {args.seed}, sized for {built.player.name}").save(target)
        written["render"] = str(target)
    print(json.dumps({
        "schema": manifest["schema"],
        "product": args.product,
        "mode": "blocks",
        "blocks": manifest["block_count"],
        "sizes": manifest["sizes"],
        "walk": manifest["walk"],
        **({"town": manifest["town"]["counts"], "houses_every_floor_reached":
            manifest["town"]["houses_every_floor_reached"]} if town_report is not None else {}),
        "ports": len(built.ports),
        "files": written,
    }, indent=2))
    return 0


def _handle_voxels(args: argparse.Namespace) -> int:
    from core.architecture.gameplay import audit as playability
    from core.architecture.voxels import voxelize

    if getattr(args, "blocks", False):
        return _handle_block_voxels(args)

    request = _request_from_args(args)
    if request.profile is None:
        request.profile = "human"
    result = _build_architecture(request)
    _checks, metrics, grid = playability(result.document, result.scene, pitch=args.pitch)
    if grid is None:
        # A settlement is checked building by building; export the whole of it
        # at a coarser cell unless asked otherwise.
        grid = voxelize(result.document, result.scene, pitch=args.pitch or 0.5)
        grid.ports = metrics["ports"]
    elif args.pitch is not None and abs(grid.pitch - args.pitch) > 1e-9:
        grid = voxelize(result.document, result.scene, pitch=args.pitch)
    paths = grid.save(args.out, args.name)
    written = {key: str(path) for key, path in paths.items()}
    if args.mcfunction:
        target = Path(args.out) / f"{args.name}.mcfunction"
        target.write_text(grid.to_mcfunction())
        written["mcfunction"] = str(target)
    if args.plan is not None:
        target = Path(args.out) / f"{args.name}.plan.txt"
        target.write_text(grid.plan_ascii(args.plan))
        written["plan"] = str(target)
    if args.blockout:
        target = Path(args.out) / f"{args.name}.blockout.json"
        target.write_text(json.dumps({"schema": "blockn.blockout/v1", "boxes": grid.blockout_boxes()}, indent=1))
        written["blockout"] = str(target)
    game = {c.code: c.status for c in result.quality.checks if c.code.startswith("game.")}
    print(json.dumps({
        "schema": grid.manifest()["schema"],
        "product": request.product,
        "pitch": grid.pitch,
        "size": list(grid.shape),
        "solid_cells": int(grid.solid.sum()),
        "quality": result.quality.status,
        "playability": game,
        "walk": metrics.get("walk"),
        "buildings": [{"plot": b["plot"], "product": b["product"], **b["checks"]} for b in metrics.get("buildings", [])],
        "ports": grid.ports,
        "files": written,
    }, indent=2))
    return 1 if result.quality.status == "fail" and not args.allow_quality_fail else 0


def _result_summary(result: BuildResult) -> dict[str, Any]:
    metrics = result.quality.metrics
    physical = metrics.get("physical", {})
    summary = {
        "architect": result.document.architect.id,
        "product": result.document.request.product,
        "seed": result.document.request.seed,
        "age": result.document.request.age,
        "fingerprint": result.document.fingerprint(),
        "quality": result.quality.status,
        "placements": metrics.get("placement_count", len(result.document.placements)),
        "blocks": metrics.get("compiled_block_count", len(result.scene.blocks)),
        "bounds_min": metrics.get("bounds_min"),
        "bounds_max": metrics.get("bounds_max"),
        "roles": metrics.get("role_counts", {}),
        "substances": metrics.get("substance_counts", {}),
        "physical": {
            "total_mass_tonnes": physical.get("total_mass_kg", 0.0) / 1000.0,
            "center_of_mass_m": physical.get("center_of_mass_m"),
            "footprint_area_m2": physical.get("footprint_area_m2"),
            "ground_bearing_pressure_kpa": physical.get("ground_bearing_pressure_kpa"),
            "bill_of_materials": physical.get("bill_of_materials", {}),
            "max_member_utilisation": physical.get("max_member_utilisation"),
            "max_utilisation_member_id": physical.get("max_utilisation_member_id"),
        },
    }
    return summary


def _write_architecture_outputs(result: BuildResult, args: argparse.Namespace) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    def record(key: str, path: Path) -> None:
        candidate = key
        index = 2
        while candidate in paths:
            candidate = f"{key}_{index}"
            index += 1
        paths[candidate] = path

    if args.bundle is not None:
        for key, path in export_bundle(result, args.bundle, formats=args.format or ("glb",), glb_materials=args.glb_materials).items():
            record(key, path)
    for output in args.out or ():
        path = export_mesh_artifact(result.scene, output, glb_materials=args.glb_materials)
        record(path.suffix.lower().lstrip("."), path)
    if args.document is not None:
        record("architecture", export_document(result.document, args.document))
    if args.quality is not None:
        record("quality", export_quality(result.quality, args.quality))
    if args.manifest is not None:
        record("manifest", export_manifest(result, args.manifest))
    if args.scene_json is not None:
        # Legacy scene I/O is intentionally chatty; keep the new command's
        # structured stdout stable for automation.
        with redirect_stdout(io.StringIO()):
            save_scene_to_json(result.scene, str(args.scene_json))
        record("scene", args.scene_json)
    return paths


def _print_build_result(
    result: BuildResult,
    paths: Mapping[str, Path],
    *,
    json_output: bool,
    quiet: bool,
) -> None:
    if quiet:
        return
    summary = _result_summary(result)
    if paths:
        summary["artifacts"] = {key: str(value) for key, value in sorted(paths.items())}
    if json_output or not paths:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(
        f"BlockN {summary['product']} | architect={summary['architect']} "
        f"seed={summary['seed']} | quality={summary['quality']} | "
        f"placements={summary['placements']}"
    )
    print(f"Fingerprint: {summary['fingerprint']}")
    physical = summary.get("physical") or {}
    if physical:
        bill = physical.get("bill_of_materials") or {}
        heaviest = max(bill.items(), key=lambda pair: pair[1]["mass_tonnes"], default=None)
        print(
            f"Mass: {physical.get('total_mass_tonnes', 0.0):.1f} t"
            f" | footprint {physical.get('footprint_area_m2', 0.0):.1f} m2"
            f" | ground bearing {physical.get('ground_bearing_pressure_kpa', 0.0):.1f} kPa"
            f" | max utilisation {physical.get('max_member_utilisation', 0.0):.3f}"
            f" ({physical.get('max_utilisation_member_id') or 'n/a'})"
        )
        if heaviest:
            print(
                f"Bill of materials: {len(bill)} substance(s), "
                f"heaviest {heaviest[1]['mass_tonnes']:.1f} t of {heaviest[0]}"
            )
    for key, path in sorted(paths.items()):
        print(f"{key}: {path}")


def _handle_catalog(args: argparse.Namespace) -> int:
    products = _architecture_products()
    payload: dict[str, Any] = {"products": products}
    if args.include_meshes:
        payload["mesh_primitives"] = list(primitive_catalog())
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("PRODUCT             CATEGORY    ARCHITECT       DEFAULT SIZE")
    for item in products:
        defaults = item.get("defaults", {})
        size = "x".join(str(defaults.get(axis, "-")) for axis in ("width", "height", "depth"))
        architect = item.get("architect", {})
        architect_id = item.get("architect_id") or architect.get("id", "-")
        print(f"{item['name']:<19} {item['category']:<11} {architect_id:<15} {size}")
    if args.include_meshes:
        print("\nInternal mesh vocabulary: " + ", ".join(item["name"] for item in payload["mesh_primitives"]))
    return 0


def _handle_describe(args: argparse.Namespace) -> int:
    descriptor = _architecture_product(args.product)
    print(json.dumps(descriptor, indent=2, sort_keys=True))
    return 0


def _handle_architects(args: argparse.Namespace) -> int:
    architects = list_architects()
    if args.json:
        print(json.dumps({"architects": architects}, indent=2, sort_keys=True))
        return 0
    print("ARCHITECT       VERSION   PRODUCTS")
    for architect in architects:
        product_names = ", ".join(product["name"] for product in architect["products"])
        print(f"{architect['id']:<15} {architect['version']:<9} {product_names}")
    return 0


def _handle_make(args: argparse.Namespace) -> int:
    result = _build_architecture(_request_from_args(args))
    paths = _write_architecture_outputs(result, args)
    _print_build_result(result, paths, json_output=args.json_summary, quiet=args.quiet)
    if result.quality.status == "fail" and not args.allow_quality_fail:
        return 1
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _handle_build(args: argparse.Namespace) -> int:
    payload = _load_json(args.request)
    if "placements" in payload:
        result = compile_document(BuildDocument.from_dict(payload))
    else:
        request_payload = payload.get("request", payload)
        if not isinstance(request_payload, Mapping):
            raise ValueError("request JSON must contain an object")
        result = _build_architecture(BuildRequest.from_dict(request_payload))
    paths = _write_architecture_outputs(result, args)
    _print_build_result(result, paths, json_output=args.json_summary, quiet=args.quiet)
    if result.quality.status == "fail" and not args.allow_quality_fail:
        return 1
    return 0


def _handle_inspect(args: argparse.Namespace) -> int:
    document = BuildDocument.from_dict(_load_json(args.document_path))
    result = compile_document(document)
    payload = result.quality.to_dict() if args.json else _result_summary(result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if result.quality.status != "fail" or args.allow_quality_fail else 1


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
    classical_add_arch_portal: bool = False,
    classical_add_dome_cap: bool = False,
) -> Scene:
    scene = Scene(use_collision=False)
    podium_height = max(1, height // 3)
    generate_classical_building(
        scene,
        width=width,
        depth=depth,
        podium_height_actual=podium_height,
        seed=seed,
        flags={
            "foundation_apron": classical_foundation_apron,
            "add_arch_portal": classical_add_arch_portal,
            "add_central_dome": classical_add_dome_cap,
        },
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
    scene.generation_metadata = {
        "schema_version": "blockn.legacy-classical/v1",
        "architect": "classical",
        "product": "classical-temple",
        "seed": seed,
        "metrics": get_classical_snapshot_metrics(scene),
    }
    return scene


def _export(scene: Scene, output: Path | None, json_path: Path | None, optimize: bool, summary: bool) -> None:
    meshes: List = []
    optimize_report = None
    if optimize:
        meshes, optimize_report = optimize_blocks(scene.blocks, return_report=True)
    else:
        meshes = scene.get_all_meshes()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        export_meshes_to_glb(meshes, str(output))
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        save_scene_to_json(scene, str(json_path))
    if not summary:
        return

    all_vertices = [mesh.vertices for mesh in meshes if mesh is not None and mesh.vertices.size]
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
    if optimize and optimize_report is not None:
        backend = optimize_report.get("backend_status", {})
        print(
            "Optimize report:"
            f" backends_available={backend.get('engines', [])}"
            f" union_attempted={optimize_report.get('groups_union_attempted')}"
            f" union_succeeded={optimize_report.get('groups_union_succeeded')}"
            f" fallback_groups={optimize_report.get('groups_fallback')}"
            f" fallback_reasons={optimize_report.get('fallback_reasons', {})}"
        )


def _handle_generate(args: argparse.Namespace) -> int:
    if args.width <= 0 or args.height <= 0 or args.depth <= 0:
        raise ValueError("width, height, and depth must be positive integers")
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
        args.classical_add_arch_portal,
        args.classical_add_dome_cap,
    )
    # A no-output invocation still reports useful work instead of silently
    # generating and discarding a scene.
    summary = args.summary or (args.out is None and args.json is None)
    _export(scene, args.out, args.json, args.optimize, summary)
    return 0


class _AppendFormats(argparse.Action):
    """Accept repeated flags and comma-separated format lists."""

    def __call__(self, parser, namespace, values, option_string=None):
        current = list(getattr(namespace, self.dest) or [])
        for part in str(values).split(","):
            item = part.strip().lower().lstrip(".")
            if not item:
                continue
            if item not in SUPPORTED_MESH_FORMATS:
                choices = ", ".join(SUPPORTED_MESH_FORMATS)
                raise argparse.ArgumentError(self, f"unknown format '{item}'. Choose from: {choices}.")
            if item not in current:
                current.append(item)
        setattr(namespace, self.dest, current)


def _add_glb_material_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-glb-materials", action="store_false", dest="glb_materials",
                        help="Export neutral GLB geometry without materials or vertex colors (other formats unchanged)")


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    _add_glb_material_option(parser)
    parser.add_argument("--bundle", type=Path, help="Artifact stem; writes mesh, architecture, quality, physics, and manifest files")
    parser.add_argument("--format", action=_AppendFormats, help="Bundle mesh formats, repeatable or comma-separated (default: glb)")
    parser.add_argument("--out", action="append", type=Path, help="Additional mesh output (repeatable; suffix selects the format)")
    parser.add_argument("--document", type=Path, help="Canonical semantic architecture JSON path")
    parser.add_argument("--quality", type=Path, help="Quality report JSON path")
    parser.add_argument("--manifest", type=Path, help="Combined manifest JSON path")
    parser.add_argument("--scene-json", type=Path, help="Compiled legacy-compatible scene JSON path")
    parser.add_argument("--json-summary", action="store_true", help="Print a machine-readable summary")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary")
    parser.add_argument("--allow-quality-fail", action="store_true", help="Exit zero even when a hard quality gate fails")


def _add_make_arguments(parser: argparse.ArgumentParser) -> None:
    _add_request_arguments(parser)
    _add_output_arguments(parser)


def _add_request_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("product", help="Product name from `blockn catalog`")
    parser.add_argument("--width", type=float)
    parser.add_argument("--height", type=float)
    parser.add_argument("--depth", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--age", type=float, default=None, help="Surface age from 0 (new) to 1 (centuries); default 0.35")
    parser.add_argument("--style", choices=("contemporary", "classical", "industrial", "timber"), default="contemporary")
    parser.add_argument("--detail", choices=("concept", "standard", "presentation"), default="standard")
    parser.add_argument("--variant", default="auto", help="Product-specific variant")
    parser.add_argument("--openings", default="auto", help="Product-specific opening mode")
    parser.add_argument("--bays", type=int)
    parser.add_argument("--levels", type=int)
    parser.add_argument("--param", action="append", type=_parse_parameter, metavar="KEY=VALUE", help="Product-specific parameter (JSON values accepted)")
    parser.add_argument("--profile", help="Player profile to build for: a preset (human, small, brute, giant, runner, "
                                          "blocky) or a JSON object; sizes doors and stairs and adds playability checks")


def _handle_agent(args: argparse.Namespace) -> int:
    from core.architecture.agent import (
        SCHEMA, DEFAULT_PRODUCTS, BatchQualityError, export_recipe, series_recipe,
    )
    try:
        if args.agent_command == "schema":
            print(json.dumps({
                "schema_version": "blockn.agent/v1",
                "recipe_schema": SCHEMA,
                "commands": ["craft", "agent schema", "agent series", "agent build"],
                "limits": {"buildings": 24, "parts_per_building": 8, "placements": 30000},
                "example": {"schema_version": SCHEMA, "buildings": [{
                    "id": "rotunda", "request": {"product": "classical-rotunda", "variant": "corinthian"},
                }]},
                "part_fields": {"id": "safe filename token", "request": "BuildRequest object", "offset": "[x,y,z], default [0,0,0]"},
                "recipe_fields": ["schema_version", "buildings", "spacing", "columns"],
                "building_fields": "id and exactly one of request or parts",
                "artifacts": ["per-building bundles", "combined series bundle", "construction reports", "recipe.json", "batch.manifest.json"],
            }, indent=2))
            return 0
        if args.agent_command == "series":
            products = args.products.split(",") if args.products else DEFAULT_PRODUCTS
            recipe = series_recipe(args.count, args.seed, products, kitbash=args.kitbash)
        else:
            recipe = json.load(sys.stdin) if str(args.recipe) == "-" else json.loads(args.recipe.read_text())
        with redirect_stdout(io.StringIO()):
            result = export_recipe(recipe, args.output, args.format or ("glb",), glb_materials=args.glb_materials)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, ArchitectureCompileError) as exc:
        print(json.dumps({"schema_version": "blockn.agent/v1", "status": "error", "error": str(exc)}))
        return 1 if isinstance(exc, BatchQualityError) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blockn",
        description="BlockN semantic architecture and block-model tool",
    )
    parser.add_argument("--version", action="version", version=f"BlockN {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("craft", help="Print historical construction traditions and architectural rules as JSON")

    catalog = sub.add_parser("catalog", aliases=["list"], help="List architecture products and architects")
    catalog.add_argument("--json", action="store_true")
    catalog.add_argument("--include-meshes", action="store_true", help="Also show the internal render primitive vocabulary")

    describe = sub.add_parser("describe", help="Describe one architecture product")
    describe.add_argument("product")

    architects = sub.add_parser("architects", help="List architect strategies and their products")
    architects.add_argument("--json", action="store_true")

    make = sub.add_parser("make", aliases=["primitive"], help="Produce an architecture primitive or assembly")
    _add_make_arguments(make)

    build = sub.add_parser("build", help="Build a request JSON or recompile an architecture document")
    build.add_argument("request", type=Path)
    _add_output_arguments(build)

    voxels = sub.add_parser("voxels", help="Voxelize a product for a player profile and check it is playable")
    _add_request_arguments(voxels)
    voxels.add_argument("--out", type=Path, required=True, help="Output directory")
    voxels.add_argument("--name", default="build", help="File stem (default: build)")
    voxels.add_argument("--pitch", type=float, help="Cell size in metres (default: a third of the player radius; "
                                                    "use 1.0 for Minecraft)")
    voxels.add_argument("--mcfunction", action="store_true", help="Also write a Minecraft .mcfunction (needs --pitch 1.0)")
    voxels.add_argument("--plan", type=float, metavar="HEIGHT", help="Also write an ASCII plan sliced at HEIGHT metres")
    voxels.add_argument("--blockout", action="store_true", help="Also write merged boxes for an engine blockout")
    voxels.add_argument("--allow-quality-fail", action="store_true")
    voxels.add_argument("--blocks", action="store_true",
                        help="Block mode: architects place Minecraft blocks on the 1 m grid instead of building "
                             "meshes (castle, village, house; norman-castle and english-village answer too)")
    voxels.add_argument("--render", action="store_true", help="With --blocks, also write a picture of the build")
    voxels.add_argument("--terrain", default="hill", choices=("hill", "slope", "valley", "valley-wall"),
                        help="With --blocks and the town product: the sample ground when no --heightmap is given")
    voxels.add_argument("--heightmap", default=None,
                        help="With --blocks and the town product: a .npy or .json 2-D array of ground heights in "
                             "blocks, indexed [x][z], to fit the town to")

    inspect = sub.add_parser("inspect", help="Compile and audit an architecture document")
    inspect.add_argument("document_path", type=Path)
    inspect.add_argument("--json", action="store_true", help="Print the full quality report")
    inspect.add_argument("--allow-quality-fail", action="store_true")

    agent = sub.add_parser("agent", help="Machine-readable kitbash series and recipe exports")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_sub.add_parser("schema", help="Print the agent contract and an example recipe")
    series = agent_sub.add_parser("series", help="Export a repeatable series of buildings")
    series.add_argument("--count", type=int, default=6)
    series.add_argument("--kitbash", action="store_true", help="Assemble each building with a fitted podium and approach stair")
    series.add_argument("--seed", type=int, default=0)
    series.add_argument("--products", help="Comma-separated catalog products")
    recipe = agent_sub.add_parser("build", help="Kitbash a JSON recipe; use - for stdin")
    recipe.add_argument("recipe", type=Path)
    for command in (series, recipe):
        _add_glb_material_option(command)
        command.add_argument("--output", type=Path, required=True, help="New output directory (existing outputs are preserved)")
        command.add_argument("--format", action=_AppendFormats, help="Mesh formats, repeatable or comma-separated (default: glb)")

    gen = sub.add_parser("generate", help="Generate a classical scene (legacy-compatible)")
    gen.add_argument("--width", type=int, default=5, help="X dimension")
    gen.add_argument("--height", type=int, default=5, help="Y dimension / podium scale")
    gen.add_argument("--depth", type=int, default=5, help="Z dimension")
    gen.add_argument("--classical-order", choices=["Random", "Doric", "Ionic", "Tuscan", "Corinthian", "Composite"], default="Random")
    gen.add_argument("--classical-preset", choices=["compact", "canonical", "monumental"], default="canonical")
    gen.add_argument("--classical-front-columns", type=int)
    gen.add_argument("--classical-side-columns", type=int)
    gen.add_argument("--classical-pediment-slope", type=float, default=1.0)
    gen.add_argument("--classical-entablature-depth", type=float, default=1.0)
    gen.add_argument("--classical-side-roof-overhang", type=float, default=1.0)
    gen.add_argument("--classical-foundation-apron", action="store_true")
    gen.add_argument("--classical-wrap-min-front-span", type=float)
    gen.add_argument("--classical-wrap-min-side-span", type=float)
    gen.add_argument("--classical-wrap-front-coverage", type=float)
    gen.add_argument("--classical-add-arch-portal", action="store_true")
    gen.add_argument("--classical-add-dome-cap", action="store_true")
    gen.add_argument("--seed", type=int)
    gen.add_argument("--out", type=Path, help="Output GLB path")
    gen.add_argument("--json", type=Path, help="Optional JSON scene path")
    gen.add_argument("--optimize", action="store_true")
    gen.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "craft":
            from core.architecture.construction import craft_catalog
            print(json.dumps(craft_catalog(), indent=2))
            return 0
        if args.command in {"catalog", "list"}:
            return _handle_catalog(args)
        if args.command == "describe":
            return _handle_describe(args)
        if args.command == "architects":
            return _handle_architects(args)
        if args.command in {"make", "primitive"}:
            return _handle_make(args)
        if args.command == "build":
            return _handle_build(args)
        if args.command == "inspect":
            return _handle_inspect(args)
        if args.command == "voxels":
            return _handle_voxels(args)
        if args.command == "agent":
            return _handle_agent(args)
        if args.command == "generate":
            return _handle_generate(args)
    except (ArchitectureCompileError, FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
