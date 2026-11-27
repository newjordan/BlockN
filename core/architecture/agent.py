"""Bounded, deterministic kitbash batches with machine-readable artifacts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile

from .compiler import compile_document
from .io import export_bundle
from .schema import ArchitectInfo, BuildDocument, BuildRequest
from .service import build, describe_product

SCHEMA = "blockn.kitbash/v1"
ARCHITECT = ArchitectInfo("kitbash", "1.0.0", "Kitbash Composer")
DEFAULT_PRODUCTS = (
    "classical-portico", "classical-villa", "classical-rotunda",
    "classical-basilica", "classical-stoa", "classical-peristyle",
)
MAX_BUILDINGS = 24
MAX_PARTS = 8
MAX_PLACEMENTS = 30000


class BatchQualityError(ValueError):
    """No artifacts are published when a requested assembly fails quality."""


def _keys(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label} field(s): {', '.join(sorted(unknown))}")


def _id(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value):
        raise ValueError(f"{label} must be 1–64 letters, digits, hyphens or underscores")
    return value


def _integer(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer from {minimum} to {maximum}")
    return value


def _offset(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("offset must contain three finite coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1000 for v in value):
        raise ValueError("offset coordinates must be finite numbers within -1000..1000")
    return tuple(float(v) for v in value)


def _request(value):
    _keys(value, {field.name for field in fields(BuildRequest)}, "request")
    if not isinstance(value.get("product"), str) or not value["product"].strip():
        raise ValueError("request requires an explicit product")
    for dimension in ("width", "height", "depth"):
        number = value.get(dimension)
        if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not 0 < number <= 200):
            raise ValueError(f"agent {dimension} must be greater than 0 and at most 200")
    parameters = value.get("parameters") or {}
    if not isinstance(parameters, dict):
        raise ValueError("request parameters must be an object")
    for key, number in parameters.items():
        if "column" in key and isinstance(number, (float, int)) and number > 24:
            raise ValueError("agent column counts cannot exceed 24")
    return BuildRequest.from_dict(value)


def _translated(placements, prefix, offset, part_id=None):
    result = deepcopy(placements)
    for p in result:
        p.id = f"{prefix}.{p.id}"
        if p.parent_id is not None:
            p.parent_id = f"{prefix}.{p.parent_id}"
        host = p.parameters.get("host_id")
        if isinstance(host, str) and host:
            p.parameters["host_id"] = f"{prefix}.{host}"
        p.position = tuple(p.position[i] + offset[i] for i in range(3))
        if part_id is not None:
            p.parameters["kitbash_part_id"] = part_id
    return result


def parts_connected(placements):
    """Conservative part-envelope connectivity, not structural certification."""
    bounds = {}
    for p in placements:
        part = p.parameters.get("kitbash_part_id")
        if not isinstance(part, str) or not part:
            return False
        lo = [p.position[i] - p.dimensions[i] / 2 for i in range(3)]
        hi = [p.position[i] + p.dimensions[i] / 2 for i in range(3)]
        if part in bounds:
            old_lo, old_hi = bounds[part]
            lo = [min(lo[i], old_lo[i]) for i in range(3)]
            hi = [max(hi[i], old_hi[i]) for i in range(3)]
        bounds[part] = (lo, hi)
    if not bounds:
        return False
    pending = [next(iter(bounds))]
    seen = set(pending)
    while pending:
        lo, hi = bounds[pending.pop()]
        for key, (other_lo, other_hi) in bounds.items():
            if key not in seen and all(hi[i] + 1e-4 >= other_lo[i] and other_hi[i] + 1e-4 >= lo[i] for i in range(3)):
                seen.add(key)
                pending.append(key)
    return len(seen) == len(bounds)


def _check(result, label):
    if result.quality.status == "fail":
        codes = [c.code for c in result.quality.checks if c.status == "fail"]
        raise BatchQualityError(f"{label} failed quality: {', '.join(codes)}")
    return result


def build_recipe(recipe):
    """Build everything before publishing; callers receive independent results."""
    _keys(recipe, {"schema_version", "buildings", "spacing", "columns"}, "recipe")
    if recipe.get("schema_version") != SCHEMA:
        raise ValueError(f"schema_version must be {SCHEMA}")
    buildings = recipe.get("buildings")
    if not isinstance(buildings, list) or not 1 <= len(buildings) <= MAX_BUILDINGS:
        raise ValueError(f"buildings must contain 1–{MAX_BUILDINGS} entries")
    spacing = recipe.get("spacing", 4)
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not math.isfinite(spacing) or not 0.1 <= spacing <= 100:
        raise ValueError("spacing must be between 0.1 and 100")
    columns = _integer(recipe.get("columns", math.ceil(math.sqrt(len(buildings)))), "columns", 1, MAX_BUILDINGS)
    results, seen, total = [], set(), 0
    for entry in buildings:
        _keys(entry, {"id", "request", "parts"}, "building")
        name = _id(entry.get("id"), "building id")
        if name.casefold() in seen:
            raise ValueError(f"duplicate building id: {name}")
        seen.add(name.casefold())
        if ("request" in entry) == ("parts" in entry):
            raise ValueError(f"{name} must supply exactly one of request or parts")
        if "request" in entry:
            result = _check(build(_request(entry["request"])), name)
        else:
            parts = entry["parts"]
            if not isinstance(parts, list) or not 1 <= len(parts) <= MAX_PARTS:
                raise ValueError(f"{name} must contain 1–{MAX_PARTS} parts")
            placements, sources, part_ids = [], [], set()
            for part in parts:
                _keys(part, {"id", "request", "offset"}, "part")
                part_id = _id(part.get("id"), "part id")
                if part_id.casefold() in part_ids:
                    raise ValueError(f"duplicate part id: {part_id}")
                part_ids.add(part_id.casefold())
                source = _check(build(_request(part.get("request"))), f"{name}/{part_id}")
                offset = _offset(part.get("offset", [0, 0, 0]))
                placements.extend(_translated(source.document.placements, part_id, offset, part_id))
                sources.append({"id": part_id, "request": source.document.request.to_dict(),
                                "fingerprint": source.document.fingerprint(), "offset": list(offset)})
                if len(placements) + total > MAX_PLACEMENTS:
                    raise ValueError("batch exceeds the placement budget")
            document = BuildDocument(ARCHITECT, BuildRequest(architect="kitbash", product="kitbashed-building"), placements,
                                     metadata={"parts": sources})
            result = _check(compile_document(document), name)
        total += len(result.document.placements)
        if total > MAX_PLACEMENTS:
            raise ValueError("batch exceeds the placement budget")
        results.append((name, result))
    # Layout uses measured bounds rather than requested dimensions (which may
    # exclude roofs, aprons or ornaments). Every asset rests on the same ground.
    cell_w = max(r.quality.metrics["bounds_size"][0] for _, r in results) + spacing
    cell_d = max(r.quality.metrics["bounds_size"][2] for _, r in results) + spacing
    placements, members = [], []
    for index, (name, result) in enumerate(results):
        lo, hi = result.quality.metrics["bounds_min"], result.quality.metrics["bounds_max"]
        offset = (index % columns * cell_w - (lo[0] + hi[0]) / 2, -lo[1],
                  index // columns * cell_d - (lo[2] + hi[2]) / 2)
        placements.extend(_translated(result.document.placements, name, offset))
        members.append({"id": name, "offset": list(offset), "fingerprint": result.document.fingerprint()})
    district = _check(compile_document(BuildDocument(ARCHITECT,
        BuildRequest(architect="kitbash", product="kitbash-series"), placements,
        metadata={"members": members})), "series")
    return results, district


def series_recipe(count=6, seed=0, products=DEFAULT_PRODUCTS, kitbash=False):
    """Different products and proportion-preserving sizes, repeatable by seed."""
    _integer(count, "count", 1, MAX_BUILDINGS)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not products:
        raise ValueError("at least one product is required")
    descriptors = [describe_product(p) for p in products]
    rng = random.Random(seed)
    buildings = []
    for i in range(count):
        descriptor = descriptors[i % len(descriptors)]
        scale = rng.uniform(0.88, 1.12)
        defaults = descriptor["defaults"]
        variants = [v for v in descriptor["variants"] if v not in {"auto", "random"}]
        request = {"product": descriptor["name"], "seed": seed + i, "style": "classical",
                   "variant": rng.choice(variants) if variants else "auto",
                   **{key: defaults[key] * scale for key in ("width", "height", "depth")}}
        if descriptor["name"] == "timber-house":
            from .timberwright import random_controls
            request = {"product": descriptor["name"], "seed": seed + i, "style": "timber",
                       **random_controls(rng)}
        entry = {"id": f"building-{i + 1:03d}", "request": request}
        if kitbash:
            measured = _check(build(_request(request)), entry["id"])
            lo, hi = measured.quality.metrics["bounds_min"], measured.quality.metrics["bounds_max"]
            width, height, depth = [hi[j] - lo[j] for j in range(3)]
            podium_h = max(0.25, min(1.0, height * 0.05))
            pad_w, pad_d = width + 2, depth + 2
            stair_d = max(1.0, podium_h * 5)
            entry = {"id": entry["id"], "parts": [
                {"id": "podium", "request": {"product": "site-pad", "width": pad_w, "height": podium_h, "depth": pad_d}},
                {"id": "building", "request": request,
                 "offset": [-(lo[0] + hi[0]) / 2, podium_h - lo[1], -(lo[2] + hi[2]) / 2]},
                {"id": "approach", "request": {"product": "stair", "width": min(4.0, pad_w * 0.22), "height": podium_h, "depth": stair_d},
                 "offset": [0, 0, -(pad_d + stair_d) / 2]},
            ]}
        buildings.append(entry)
    return {"schema_version": SCHEMA, "buildings": buildings}


def export_recipe(recipe, output, formats=("glb",), *, glb_materials=True):
    target = Path(output).resolve()
    if target.exists():
        raise ValueError(f"output already exists: {target}; choose a new directory")
    results, district = build_recipe(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        assets = []
        for name, result in results:
            paths = export_bundle(result, temporary / "buildings" / name, formats, glb_materials=glb_materials)
            assets.append({"id": name, "quality": result.quality.status,
                "fingerprint": result.document.fingerprint(),
                "artifacts": {key: str(path.relative_to(temporary)) for key, path in paths.items()}})
        paths = export_bundle(district, temporary / "series", formats, glb_materials=glb_materials)
        manifest = {"schema_version": "blockn.batch/v1", "status": "warning" if district.quality.status == "warning" or any(a["quality"] == "warning" for a in assets) else "pass", "count": len(assets),
                    "buildings": assets, "series": {key: str(p.relative_to(temporary)) for key, p in paths.items()}}
        (temporary / "recipe.json").write_text(json.dumps(recipe, indent=2, allow_nan=False) + "\n")
        (temporary / "batch.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if target.exists():
            raise ValueError(f"output already exists: {target}")
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {**manifest, "output": str(target), "manifest": str(target / "batch.manifest.json")}
