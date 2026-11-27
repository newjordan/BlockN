"""Semantic adapter for the mature legacy classical building generator."""

from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import io
import math
import re
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from core.generators.classical import generate_classical_building
from core.scene import Scene

from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement


ARCHITECT = ArchitectInfo(
    id="classical",
    version="3.1.0",
    title="Classical Architect",
)


# The physical substance behind each adapted legacy role.  A five-order temple
# reads as marble order and roof members on a limestone podium and cella; a
# Roman concrete dome cap covers the composite variants.
CLASSICAL_SUBSTANCES: Dict[str, str] = {
    "podium": "limestone",
    "foundation_apron": "limestone",
    "doorway_threshold": "limestone",
    "stair": "limestone",
    "entrance": "limestone",
    "exit": "limestone",
    "cella_wall": "limestone",
    "arch_portal": "limestone",
    "base": "marble",
    "base_lower": "marble",
    "base_upper": "marble",
    "base_scotia": "marble",
    "shaft": "marble",
    "shaft_astragal": "marble",
    "capital_neck": "marble",
    "capital_echinus": "marble",
    "capital_abacus": "marble",
    "capital_annuli": "marble",
    "capital_cushion": "marble",
    "capital_bell": "marble",
    "capital_helix": "marble",
    "capital_foliage": "marble",
    "capital_volute": "marble",
    "architrave": "marble",
    "frieze": "marble",
    "cornice": "marble",
    "raking_cornice": "marble",
    "triglyph": "marble",
    "frieze_relief": "marble",
    "cella_lintel": "marble",
    "pediment": "marble",
    "ridge": "marble",
    "roof_panel": "marble",
    "roof_seam": "marble",
}
CLASSICAL_DEFAULT_SUBSTANCE = "limestone"


def _substance_for_role(role: str) -> str:
    return CLASSICAL_SUBSTANCES.get(role, CLASSICAL_DEFAULT_SUBSTANCE)


PRODUCT: Dict[str, Any] = {
    "name": "classical-building",
    "category": "assembly",
    "architect": ARCHITECT.to_dict(),
    "architect_id": ARCHITECT.id,
    "architect_title": ARCHITECT.title,
    "architect_version": ARCHITECT.version,
    "description": "Complete seeded temple composition in the five classical orders.",
    "defaults": {
        "width": 14.0,
        "height": 9.0,
        "depth": 20.0,
        "variant": "auto",
        "detail": "standard",
    },
    "variants": ["auto", "random", "doric", "ionic", "tuscan", "corinthian", "composite"],
    "detail_levels": ["concept", "standard", "presentation"],
    "opening_modes": ["auto"],
    "parameters": [
        "preset",
        "front_columns",
        "side_columns",
        "pediment_slope",
        "entablature_depth",
        "side_roof_overhang",
        "foundation_apron",
        "wrap_min_front_span",
        "wrap_min_side_span",
        "wrap_front_coverage",
        "arch_portal",
        "dome_cap",
    ],
    "substances": sorted(set(CLASSICAL_SUBSTANCES.values()) | {CLASSICAL_DEFAULT_SUBSTANCE}),
}


_DETAIL_TO_PRESET = {
    "concept": "compact",
    "standard": "canonical",
    "presentation": "monumental",
}

_VARIANT_TO_ORDER = {
    "auto": "Random",
    "random": "Random",
    "doric": "Doric",
    "ionic": "Ionic",
    "tuscan": "Tuscan",
    "corinthian": "Corinthian",
    "composite": "Composite",
}

_SUPPORTED_PARAMETERS = {
    "preset", "classical_preset",
    "front_columns", "front_column_override", "classical_front_columns",
    "side_columns", "side_column_override", "classical_side_columns",
    "pediment_slope", "pediment_slope_multiplier", "classical_pediment_slope",
    "entablature_depth", "entablature_depth_multiplier", "classical_entablature_depth",
    "side_roof_overhang", "overhang", "side_roof_overhang_multiplier", "classical_side_roof_overhang",
    "foundation_apron", "apron", "classical_foundation_apron",
    "arch_portal", "add_arch_portal", "classical_add_arch_portal",
    "dome_cap", "central_dome", "add_central_dome", "classical_add_dome_cap",
    "wrap_min_front_span", "wrap_min_podium_span_x", "classical_wrap_min_front_span",
    "wrap_min_side_span", "wrap_min_podium_span_z", "classical_wrap_min_side_span",
    "wrap_front_coverage", "wrap_front_coverage_ratio", "classical_wrap_front_coverage",
}

_ELEMENT_SEMANTICS: Dict[str, Tuple[str, str]] = {
    "podium": ("foundation", "podium"),
    "base": ("column", "column_base"),
    "base_lower": ("column", "column_base_lower"),
    "base_upper": ("column", "column_base_upper"),
    "shaft": ("column", "column_shaft"),
    "capital_neck": ("column", "column_capital_neck"),
    "capital_echinus": ("column", "column_capital_echinus"),
    "capital_abacus": ("column", "column_capital_abacus"),
    "architrave": ("entablature", "architrave"),
    "frieze": ("entablature", "frieze"),
    "cornice": ("entablature", "cornice"),
    "triglyph": ("ornament", "triglyph"),
    "frieze_relief": ("ornament", "frieze_relief"),
    "pediment": ("roof", "pediment"),
    "ridge": ("roof", "ridge"),
    "stair": ("circulation", "stair"),
    "entrance_marker": ("circulation", "entrance"),
    "exit_marker": ("circulation", "exit"),
    "foundation_apron": ("foundation", "foundation_apron"),
    "doorway_threshold": ("opening", "doorway_threshold"),
    "arch_portal": ("opening", "arch_portal"),
    "dome_cap": ("roof", "dome_cap"),
    "base_scotia": ("column", "column_base_scotia"),
    "shaft_astragal": ("column", "column_shaft_astragal"),
    "capital_annuli": ("column", "column_capital_annuli"),
    "capital_cushion": ("column", "column_capital_cushion"),
    "capital_bell": ("column", "column_capital_bell"),
    "capital_helix": ("column", "column_capital_helix"),
    "capital_foliage": ("column", "column_capital_foliage"),
    "capital_volute": ("column", "column_capital_volute"),
    "raking_cornice": ("roof", "raking_cornice"),
    "roof_panel": ("roof", "roof_panel"),
    "roof_seam": ("roof", "roof_seam"),
    "cella_wall": ("wall", "cella_wall"),
    "cella_lintel": ("entablature", "cella_lintel"),
}


def _positive_dimension(name: str, value: Any, default: float) -> float:
    selected = default if value is None else value
    if isinstance(selected, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _parameter(parameters: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in parameters:
            return parameters[name]
    return default


def _optional_positive_float(parameters: Mapping[str, Any], *names: str) -> Optional[float]:
    value = _parameter(parameters, *names)
    if value is None:
        return None
    return _positive_dimension(names[0], value, 1.0)


def _positive_float(
    parameters: Mapping[str, Any],
    *names: str,
    default: float,
) -> float:
    value = _parameter(parameters, *names, default=default)
    return _positive_dimension(names[0], value, default)


def _optional_positive_int(parameters: Mapping[str, Any], *names: str) -> Optional[int]:
    value = _parameter(parameters, *names)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{names[0]} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{names[0]} must be a positive integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{names[0]} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{names[0]} must be a positive integer")
    return result


def _boolean(parameters: Mapping[str, Any], *names: str, default: bool = False) -> bool:
    value = _parameter(parameters, *names, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"{names[0]} must be a boolean")


def _semantic_pair(parameters: Mapping[str, Any]) -> Tuple[str, str]:
    element = str(parameters.get("element") or "podium").strip().lower()
    if element in _ELEMENT_SEMANTICS:
        return _ELEMENT_SEMANTICS[element]
    safe_element = re.sub(r"[^a-z0-9_]+", "_", element).strip("_") or "component"
    return "classical_component", safe_element


def _primitive_id(role: str, counters: Counter[str]) -> str:
    safe_role = re.sub(r"[^a-z0-9_]+", "_", role.lower()).strip("_") or "component"
    index = counters[safe_role]
    counters[safe_role] += 1
    return f"classical.{safe_role}.{index:04d}"


def _placement_from_block(block: Any, counters: Counter[str]) -> Placement:
    parameters = deepcopy(dict(getattr(block, "parameters", {}) or {}))
    kind, role = _semantic_pair(parameters)
    position = tuple(float(value) for value in np.asarray(block.position, dtype=float).tolist())
    dimensions = tuple(float(value) for value in np.asarray(block.dimensions, dtype=float).tolist())

    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError(f"Legacy classical block {block.block_id} has an invalid position")
    if (
        len(dimensions) != 3
        or not all(math.isfinite(value) for value in dimensions)
        or not all(value > 0.0 for value in dimensions)
    ):
        raise ValueError(f"Legacy classical block {block.block_id} has invalid dimensions")

    material = tuple(int(channel) for channel in block.color)
    if len(material) != 4:
        raise ValueError(f"Legacy classical block {block.block_id} has invalid RGBA material")

    return Placement(
        id=_primitive_id(role, counters),
        kind=kind,
        block_type=str(block.block_type),
        position=position,  # type: ignore[arg-type]
        dimensions=dimensions,  # type: ignore[arg-type]
        material=material,  # type: ignore[arg-type]
        role=role,
        parameters=parameters,
        parent_id=None,
        substance=_substance_for_role(role),
    )


def build_classical_document(request: BuildRequest) -> BuildDocument:
    """Generate a legacy classical scene and adapt it to the v1 semantic schema."""
    if not isinstance(request, BuildRequest):
        raise TypeError("request must be a BuildRequest")

    width = _positive_dimension("width", request.width, 14.0)
    height = _positive_dimension("height", request.height, 9.0)
    depth = _positive_dimension("depth", request.depth, 20.0)

    variant = str(request.variant or "auto").strip().lower()
    if variant not in _VARIANT_TO_ORDER:
        choices = ", ".join(_VARIANT_TO_ORDER)
        raise ValueError(f"variant must be one of: {choices}")

    detail = str(request.detail or "standard").strip().lower()
    if detail not in _DETAIL_TO_PRESET:
        choices = ", ".join(_DETAIL_TO_PRESET)
        raise ValueError(f"detail must be one of: {choices}")

    parameters = request.parameters or {}
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    unknown_parameters = sorted(str(key) for key in parameters if key not in _SUPPORTED_PARAMETERS)
    if unknown_parameters:
        raise ValueError(
            "unsupported parameter(s) for classical-building: "
            + ", ".join(unknown_parameters)
        )
    if request.bays is not None:
        raise ValueError("classical-building does not support --bays; use front_columns/side_columns parameters")
    if request.levels is not None:
        raise ValueError("classical-building does not support --levels")
    if str(request.openings or "auto").strip().lower() != "auto":
        raise ValueError("classical-building does not support --openings; use arch_portal=true")

    preset = str(_parameter(parameters, "preset", "classical_preset", default=_DETAIL_TO_PRESET[detail]))
    preset = preset.strip().lower()
    if preset not in {"compact", "canonical", "monumental"}:
        raise ValueError("preset must be compact, canonical, or monumental")

    flags = {
        "foundation_apron": _boolean(
            parameters, "foundation_apron", "apron", "classical_foundation_apron"
        ),
        "add_arch_portal": _boolean(
            parameters, "arch_portal", "add_arch_portal", "classical_add_arch_portal"
        ),
        "add_central_dome": _boolean(
            parameters, "dome_cap", "central_dome", "add_central_dome", "classical_add_dome_cap"
        ),
    }

    scene = Scene(use_collision=False)
    # The legacy generator is intentionally chatty; command-mode architecture
    # builds should be clean and communicate through the resulting manifest.
    with redirect_stdout(io.StringIO()):
        metrics = generate_classical_building(
            scene,
            width=max(1, int(round(width))),
            depth=max(1, int(round(depth))),
            podium_height_actual=max(.3, height * .10),
            seed=int(request.seed),
            flags=flags,
            order_override=_VARIANT_TO_ORDER[variant],
            preset_override=preset,
            front_column_override=_optional_positive_int(
                parameters, "front_columns", "front_column_override", "classical_front_columns"
            ),
            side_column_override=_optional_positive_int(
                parameters, "side_columns", "side_column_override", "classical_side_columns"
            ),
            pediment_slope_multiplier=_positive_float(
                parameters,
                "pediment_slope",
                "pediment_slope_multiplier",
                "classical_pediment_slope",
                default=1.0,
            ),
            entablature_depth_multiplier=_positive_float(
                parameters,
                "entablature_depth",
                "entablature_depth_multiplier",
                "classical_entablature_depth",
                default=1.0,
            ),
            side_roof_overhang_multiplier=_positive_float(
                parameters,
                "side_roof_overhang",
                "overhang",
                "side_roof_overhang_multiplier",
                "classical_side_roof_overhang",
                default=1.0,
            ),
            wrap_min_podium_span_x_override=_optional_positive_float(
                parameters,
                "wrap_min_front_span",
                "wrap_min_podium_span_x",
                "classical_wrap_min_front_span",
            ),
            wrap_min_podium_span_z_override=_optional_positive_float(
                parameters,
                "wrap_min_side_span",
                "wrap_min_podium_span_z",
                "classical_wrap_min_side_span",
            ),
            wrap_front_coverage_ratio_override=_optional_positive_float(
                parameters,
                "wrap_front_coverage",
                "wrap_front_coverage_ratio",
                "classical_wrap_front_coverage",
            ),
            clear_scene=True,
        )

    counters: Counter[str] = Counter()
    placements = [_placement_from_block(block, counters) for block in scene.blocks]
    if not placements:
        raise ValueError("Legacy classical generator produced no placements")

    design_metrics = deepcopy(
        getattr(scene, "last_classical_metrics", None)
        if isinstance(getattr(scene, "last_classical_metrics", None), dict)
        else metrics
    )
    metadata = {
        "design_metrics": design_metrics,
        "legacy_generator": "core.generators.classical.generate_classical_building",
        "invariants": {
            "min_placements": 1,
            "ground_contact": True,
            "parent_links_valid": True,
        },
    }
    return BuildDocument(
        architect=ARCHITECT,
        request=request,
        placements=placements,
        metadata=metadata,
    )


__all__ = ["ARCHITECT", "PRODUCT", "build_classical_document"]
