from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ..generator import generate_podium
from ..materials import ColorTuple
from ..scene import CUBE, CYLINDER, WEDGE, Scene


STACK_EPSILON = 1e-5


@dataclass(frozen=True)
class GreekOrderSpec:
    name: str
    base_diameter: float
    shaft_height_ratio: float
    clear_spacing_ratio: float
    edge_margin_ratio: float
    base_height_ratio: float
    base_width_ratio: float
    capital_height_ratio: float
    capital_width_ratio: float
    entablature_ratio: float
    pediment_rise_over_half_span: float
    shaft_sections: int


@dataclass(frozen=True)
class TempleEnvelope:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    entablature_top_y: float
    cornice_depth: float


@dataclass(frozen=True)
class ClassicalPreset:
    name: str
    diameter_scale: float
    shaft_height_scale: float
    spacing_scale: float
    entablature_depth_scale: float
    pediment_slope_scale: float
    front_max_columns: int
    side_max_columns: int


ORDER_SPECS: Dict[str, GreekOrderSpec] = {
    "Doric": GreekOrderSpec(
        name="Doric",
        base_diameter=1.00,
        shaft_height_ratio=5.4,
        clear_spacing_ratio=1.30,
        edge_margin_ratio=0.66,
        base_height_ratio=0.17,
        base_width_ratio=1.08,
        capital_height_ratio=0.66,
        capital_width_ratio=1.40,
        entablature_ratio=0.27,
        pediment_rise_over_half_span=0.25,
        shaft_sections=18,
    ),
    "Ionic": GreekOrderSpec(
        name="Ionic",
        base_diameter=0.76,
        shaft_height_ratio=9.1,
        clear_spacing_ratio=2.35,
        edge_margin_ratio=0.92,
        base_height_ratio=0.48,
        base_width_ratio=1.34,
        capital_height_ratio=0.80,
        capital_width_ratio=1.58,
        entablature_ratio=0.18,
        pediment_rise_over_half_span=0.21,
        shaft_sections=24,
    ),
}

PRESET_SPECS: Dict[str, ClassicalPreset] = {
    "compact": ClassicalPreset(
        name="compact",
        diameter_scale=0.92,
        shaft_height_scale=0.94,
        spacing_scale=0.88,
        entablature_depth_scale=1.06,
        pediment_slope_scale=0.92,
        front_max_columns=8,
        side_max_columns=10,
    ),
    "canonical": ClassicalPreset(
        name="canonical",
        diameter_scale=1.00,
        shaft_height_scale=1.00,
        spacing_scale=1.00,
        entablature_depth_scale=1.00,
        pediment_slope_scale=1.00,
        front_max_columns=10,
        side_max_columns=14,
    ),
    "monumental": ClassicalPreset(
        name="monumental",
        diameter_scale=1.08,
        shaft_height_scale=1.05,
        spacing_scale=1.10,
        entablature_depth_scale=1.18,
        pediment_slope_scale=1.06,
        front_max_columns=12,
        side_max_columns=16,
    ),
}

DEFAULT_FLAGS: Dict[str, bool] = {
    "podium": True,
    "columns": True,
    "entablature": True,
    "pediment": True,
    "roof": True,
    "apply_symmetry": True,
    "use_wedge_pediment": True,
    "human_access": True,
    "main_stairs": True,
    "egress_stairs": True,
    "entrance_markers": False,
    "foundation_apron": False,
    "validate_invariants": True,
    # The support grid is unit-cell based and rejects valid overhangs/continuous members.
    "disable_support_for_classical": True,
}

COLUMN_COLOR: ColorTuple = (220, 220, 220, 255)
BASE_COLOR: ColorTuple = (200, 200, 200, 255)
ARCHITRAVE_COLOR: ColorTuple = (168, 168, 168, 255)
DORIC_FRIEZE_COLOR: ColorTuple = (205, 205, 205, 255)
IONIC_FRIEZE_COLOR: ColorTuple = (212, 212, 212, 255)
CORNICE_COLOR: ColorTuple = (146, 146, 146, 255)
DORIC_TRIGLYPH_COLOR: ColorTuple = (176, 176, 176, 255)
IONIC_RELIEF_COLOR: ColorTuple = (188, 188, 188, 255)
PEDIMENT_COLOR: ColorTuple = (150, 150, 150, 255)
ROOF_COLOR: ColorTuple = (112, 112, 128, 255)
ACCESS_MARKER_COLOR: ColorTuple = (232, 186, 110, 255)
STAIR_COLOR: ColorTuple = (172, 172, 172, 255)
APRON_COLOR: ColorTuple = (166, 166, 166, 255)
THRESHOLD_COLOR: ColorTuple = (186, 186, 186, 255)


def _merge_flags(flags: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    merged = dict(DEFAULT_FLAGS)
    if flags:
        for key, value in flags.items():
            merged[key] = bool(value)
    return merged


def _resolve_order(order_override: Optional[str], rng: random.Random) -> str:
    aliases = {
        "doric": "Doric",
        "ionic": "Ionic",
        "ioinic": "Ionic",  # common typo
        "random": None,
        "": None,
    }

    if order_override is None:
        return rng.choice(["Doric", "Ionic"])

    normalized = order_override.strip().lower()
    if normalized in aliases:
        resolved = aliases[normalized]
        return resolved if resolved else rng.choice(["Doric", "Ionic"])

    for canonical in ORDER_SPECS.keys():
        if order_override == canonical:
            return canonical

    print(f"Warning: Unsupported order '{order_override}', using random Doric/Ionic.")
    return rng.choice(["Doric", "Ionic"])


def _resolve_preset(preset_override: Optional[str]) -> ClassicalPreset:
    if preset_override is None:
        return PRESET_SPECS["canonical"]

    normalized = preset_override.strip().lower()
    aliases = {
        "": "canonical",
        "default": "canonical",
        "standard": "canonical",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in PRESET_SPECS:
        return PRESET_SPECS[normalized]

    print(f"Warning: Unsupported classical preset '{preset_override}', using canonical.")
    return PRESET_SPECS["canonical"]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _required_span(count: int, member_width: float, clear_spacing: float, margin: float) -> float:
    return count * member_width + (count - 1) * clear_spacing + 2.0 * margin


def _fit_even_column_count(
    span: float,
    member_width: float,
    clear_spacing: float,
    margin: float,
    minimum: int = 4,
    maximum: int = 10,
) -> int:
    minimum = max(2, minimum)
    if minimum % 2 == 1:
        minimum += 1
    maximum = max(minimum, maximum)
    if maximum % 2 == 1:
        maximum -= 1

    for count in range(maximum, minimum - 1, -2):
        if _required_span(count, member_width, clear_spacing, margin) <= span + 1e-8:
            return count
    return minimum


def _column_run_coordinates(
    span: float,
    count: int,
    member_width: float,
    clear_spacing: float,
    margin: float,
) -> List[float]:
    required = _required_span(count, member_width, clear_spacing, margin)
    # Center runs when the fitted count leaves extra footprint width so podium/facade stay aligned.
    centering_offset = max(0.0, (span - required) / 2.0)
    start = centering_offset + margin + member_width / 2.0
    center_spacing = member_width + clear_spacing
    return [round(start + i * center_spacing, 6) for i in range(count)]


def _resolve_column_count(
    span: float,
    member_width: float,
    clear_spacing: float,
    margin: float,
    minimum: int,
    maximum: int,
    override: Optional[int],
) -> int:
    auto_count = _fit_even_column_count(
        span=span,
        member_width=member_width,
        clear_spacing=clear_spacing,
        margin=margin,
        minimum=minimum,
        maximum=maximum,
    )
    if override is None:
        return auto_count

    requested = int(override)
    desired = max(minimum, min(maximum, requested))
    if desired % 2 == 1:
        desired = desired + 1 if desired < maximum else desired - 1

    if desired != requested:
        print(
            f"Warning: Requested column count {override} adjusted to {desired} "
            f"(must be even and within {minimum}..{maximum})."
        )

    if _required_span(desired, member_width, clear_spacing, margin) <= span + 1e-8:
        return desired

    fitted = _fit_even_column_count(
        span=span,
        member_width=member_width,
        clear_spacing=clear_spacing,
        margin=margin,
        minimum=minimum,
        maximum=desired,
    )
    print(
        f"Warning: Requested column count {override} does not fit span {span:.2f}; "
        f"using {fitted} instead."
    )
    return fitted


def _pediment_peak_height(span_x: float, base_ratio: float, slope_multiplier: float) -> float:
    span_norm = _clamp((span_x - 8.0) / 18.0, 0.0, 1.0)
    normalized_ratio = base_ratio * (1.08 - 0.18 * span_norm)
    normalized_ratio = _clamp(normalized_ratio, 0.16, 0.34)
    ratio = normalized_ratio * _clamp(slope_multiplier, 0.35, 1.8)
    proposed_height = (span_x / 2.0) * ratio
    min_height = max(0.75, span_x * 0.06)
    max_height = max(min_height, span_x * 0.22)
    return _clamp(proposed_height, min_height, max_height)


def _scene_bounds(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    meshes = scene.get_all_meshes()
    vertices = [mesh.vertices for mesh in meshes if mesh is not None and mesh.vertices.size]
    if not vertices:
        zero = np.array([0.0, 0.0, 0.0], dtype=float)
        return zero, zero
    stacked = np.vstack(vertices)
    return stacked.min(axis=0), stacked.max(axis=0)


def get_classical_snapshot_metrics(scene: Scene) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    cached = getattr(scene, "last_classical_metrics", None)
    if isinstance(cached, dict):
        metrics.update(cached)

    metrics.setdefault("block_count", len(scene.blocks))

    shaft_blocks = [
        block
        for block in scene.blocks
        if block.block_type == CYLINDER and block.parameters.get("element") == "shaft"
    ]
    if shaft_blocks:
        avg_radius = float(np.mean([block.parameters["radius"] for block in shaft_blocks]))
        avg_slenderness = float(
            np.mean(
                [
                    block.parameters["height"] / (block.parameters["radius"] * 2.0)
                    for block in shaft_blocks
                ]
            )
        )
        metrics.setdefault("avg_column_radius", round(avg_radius, 6))
        metrics.setdefault("avg_column_slenderness", round(avg_slenderness, 6))

    if "bounds_min" not in metrics or "bounds_max" not in metrics:
        bounds_min, bounds_max = _scene_bounds(scene)
        metrics["bounds_min"] = bounds_min.round(6).tolist()
        metrics["bounds_max"] = bounds_max.round(6).tolist()

    return metrics


def _center_y(base_y: float, height: float) -> float:
    return base_y + height / 2.0


def _opening_candidates(run_coords: List[float], member_width: float) -> List[tuple[float, float]]:
    candidates: List[tuple[float, float]] = []
    if len(run_coords) < 2:
        return candidates

    sorted_coords = sorted(run_coords)
    for left, right in zip(sorted_coords[:-1], sorted_coords[1:]):
        clear_width = right - left - member_width
        if clear_width <= 0.0:
            continue
        candidates.append(((left + right) / 2.0, clear_width))
    return candidates


def _pick_opening_center(
    run_coords: List[float],
    member_width: float,
    preferred_center: float,
) -> tuple[float, float]:
    candidates = _opening_candidates(run_coords, member_width)
    if not candidates:
        fallback = preferred_center if run_coords else 0.0
        return float(fallback), max(0.8, member_width * 0.65)

    return min(candidates, key=lambda candidate: (abs(candidate[0] - preferred_center), -candidate[1]))


def _resolve_doorway_width(clear_width: float, *, ratio: float, min_width: float) -> float:
    max_width = max(0.60, clear_width * 0.96)
    target_width = max(min_width, clear_width * ratio)
    return _clamp(target_width, 0.60, max_width)


def _resolve_stair_span(
    run_coords: List[float],
    *,
    member_width: float,
    preferred_center: float,
    podium_min: float,
    podium_max: float,
    doorway_width: float,
    coverage_ratio: float,
    side_shoulder_ratio: float,
    min_columns: int = 2,
) -> float:
    podium_span = max(0.0, podium_max - podium_min)
    doorway_width = max(0.60, float(doorway_width))
    if podium_span <= 0.0:
        return doorway_width

    max_half_span = min(preferred_center - podium_min, podium_max - preferred_center)
    if max_half_span <= 0.0:
        return doorway_width

    sorted_coords = sorted(run_coords)
    count = len(sorted_coords)
    if count < 2:
        return _clamp(doorway_width, doorway_width, max(doorway_width, max_half_span * 2.0))

    target_columns = int(round(count * coverage_ratio))
    target_columns = max(2, min(count, target_columns))
    target_columns = max(min_columns, target_columns)
    target_columns = min(count, target_columns)

    if count % 2 == 0 and target_columns % 2 == 1:
        if target_columns < count:
            target_columns += 1
        else:
            target_columns -= 1
    target_columns = max(2, min(count, target_columns))

    best_left = sorted_coords[0] - member_width / 2.0
    best_right = sorted_coords[-1] + member_width / 2.0
    best_score = float("inf")
    for start_idx in range(0, count - target_columns + 1):
        end_idx = start_idx + target_columns - 1
        left_edge = sorted_coords[start_idx] - member_width / 2.0
        right_edge = sorted_coords[end_idx] + member_width / 2.0
        span_center = (left_edge + right_edge) / 2.0
        score = abs(span_center - preferred_center)
        if score < best_score:
            best_score = score
            best_left = left_edge
            best_right = right_edge

    shoulder = max(0.0, member_width * side_shoulder_ratio)
    target_width = (best_right - best_left) + shoulder * 2.0
    target_width = max(doorway_width, target_width)
    return _clamp(target_width, doorway_width, max(doorway_width, max_half_span * 2.0))


def _wrap_profile_for_preset(preset_name: str) -> Dict[str, float]:
    normalized = (preset_name or "canonical").strip().lower()
    if normalized == "compact":
        return {
            "min_podium_span_x": 22.0,
            "min_podium_span_z": 14.0,
            "front_coverage_ratio": 0.62,
            "wrap_span_min_ratio": 0.20,
            "wrap_span_max_ratio": 0.38,
        }
    if normalized == "monumental":
        return {
            "min_podium_span_x": 16.0,
            "min_podium_span_z": 10.0,
            "front_coverage_ratio": 0.48,
            "wrap_span_min_ratio": 0.26,
            "wrap_span_max_ratio": 0.50,
        }
    return {
        "min_podium_span_x": 18.0,
        "min_podium_span_z": 12.0,
        "front_coverage_ratio": 0.54,
        "wrap_span_min_ratio": 0.24,
        "wrap_span_max_ratio": 0.44,
    }


def _add_stair_run(
    scene: Scene,
    *,
    center_x: float,
    center_z: float,
    width_x: float,
    width_z: float,
    ascend_axis: str,
    direction_sign: float,
    top_edge_coord: float,
    top_y: float,
    step_count: int,
    color: ColorTuple,
    parameters: Dict[str, Any],
):
    if step_count <= 0 or top_y <= 0.0:
        return

    step_rise = top_y / float(step_count)
    step_depth = width_z if ascend_axis == "Z" else width_x

    for idx in range(step_count):
        current_height = step_rise * (idx + 1)
        y_center = current_height / 2.0
        # Keep the highest step adjacent to the podium edge and lowest step outside.
        outward_step_index = (step_count - idx) - 0.5

        if ascend_axis == "Z":
            z_center = top_edge_coord + direction_sign * (step_depth * outward_step_index)
            x_center = center_x
            dims = np.array([width_x, current_height, step_depth], dtype=float)
        else:
            x_center = top_edge_coord + direction_sign * (step_depth * outward_step_index)
            z_center = center_z
            dims = np.array([step_depth, current_height, width_z], dtype=float)

        _add_cube(
            scene,
            x_center,
            y_center,
            z_center,
            dims,
            color,
            parameters={**parameters, "step_index": int(idx)},
        )


def _add_cube(
    scene: Scene,
    x: float,
    y: float,
    z: float,
    dimensions: np.ndarray,
    color: ColorTuple,
    parameters: Optional[Dict[str, Any]] = None,
):
    scene.add_block(
        position=np.array([float(x), float(y), float(z)]),
        block_type=CUBE,
        color=color,
        dimensions=np.array(dimensions, dtype=float),
        parameters=parameters or {},
    )


def _add_cylinder(
    scene: Scene,
    x: float,
    y: float,
    z: float,
    radius: float,
    height: float,
    sections: int,
    color: ColorTuple,
    parameters: Optional[Dict[str, Any]] = None,
):
    if height <= 0.0 or radius <= 0.0:
        return
    param_dict = {
        "radius": float(radius),
        "height": float(height),
        "sections": int(sections),
    }
    if parameters:
        param_dict.update(parameters)

    scene.add_block(
        position=np.array([float(x), float(y), float(z)]),
        block_type=CYLINDER,
        color=color,
        dimensions=np.array([radius * 2.0, height, radius * 2.0], dtype=float),
        parameters=param_dict,
    )


def _add_wedge(
    scene: Scene,
    x: float,
    y: float,
    z: float,
    dimensions: np.ndarray,
    orientation: str,
    color: ColorTuple,
    parameters: Optional[Dict[str, Any]] = None,
):
    param_dict = {"orientation": orientation}
    if parameters:
        param_dict.update(parameters)

    scene.add_block(
        position=np.array([float(x), float(y), float(z)]),
        block_type=WEDGE,
        color=color,
        dimensions=np.array(dimensions, dtype=float),
        parameters=param_dict,
    )


def _build_column(
    scene: Scene,
    order: GreekOrderSpec,
    x: float,
    z: float,
    podium_top: float,
    column_radius: float,
    shaft_height: float,
    base_height: float,
    base_width: float,
    capital_height: float,
    capital_width: float,
    column_metadata: Optional[Dict[str, Any]] = None,
) -> float:
    metadata = dict(column_metadata or {})
    y = podium_top

    if order.name == "Ionic":
        lower_h = base_height * 0.55
        upper_h = base_height - lower_h
        _add_cylinder(
            scene,
            x,
            _center_y(y, lower_h),
            z,
            radius=(base_width / 2.0) * 1.05,
            height=lower_h,
            sections=order.shaft_sections,
            color=BASE_COLOR,
            parameters={"order": order.name, "element": "base_lower", **metadata},
        )
        y += lower_h + STACK_EPSILON
        _add_cylinder(
            scene,
            x,
            _center_y(y, upper_h),
            z,
            radius=(base_width / 2.0) * 0.95,
            height=upper_h,
            sections=order.shaft_sections,
            color=BASE_COLOR,
            parameters={"order": order.name, "element": "base_upper", **metadata},
        )
        y += upper_h + STACK_EPSILON
    else:
        _add_cube(
            scene,
            x,
            _center_y(y, base_height),
            z,
            np.array([base_width, base_height, base_width]),
            BASE_COLOR,
            parameters={"order": order.name, "element": "base", **metadata},
        )
        y += base_height + STACK_EPSILON

    _add_cylinder(
        scene,
        x,
        _center_y(y, shaft_height),
        z,
        radius=column_radius,
        height=shaft_height,
        sections=order.shaft_sections,
        color=COLUMN_COLOR,
        parameters={"order": order.name, "element": "shaft", **metadata},
    )
    y += shaft_height + STACK_EPSILON

    if order.name == "Ionic":
        neck_h = capital_height * 0.45
        abacus_h = capital_height - neck_h
        _add_cylinder(
            scene,
            x,
            _center_y(y, neck_h),
            z,
            radius=max(column_radius * 0.95, (capital_width / 2.0) * 0.84),
            height=neck_h,
            sections=order.shaft_sections,
            color=COLUMN_COLOR,
            parameters={"order": order.name, "element": "capital_neck", **metadata},
        )
        y += neck_h + STACK_EPSILON
        _add_cube(
            scene,
            x,
            _center_y(y, abacus_h),
            z,
            np.array([capital_width * 1.12, abacus_h, capital_width * 1.12]),
            BASE_COLOR,
            parameters={"order": order.name, "element": "capital_abacus", **metadata},
        )
        y += abacus_h
    else:
        echinus_h = capital_height * 0.42
        abacus_h = capital_height - echinus_h
        _add_cylinder(
            scene,
            x,
            _center_y(y, echinus_h),
            z,
            radius=(capital_width / 2.0),
            height=echinus_h,
            sections=order.shaft_sections,
            color=COLUMN_COLOR,
            parameters={"order": order.name, "element": "capital_echinus", **metadata},
        )
        y += echinus_h + STACK_EPSILON
        _add_cube(
            scene,
            x,
            _center_y(y, abacus_h),
            z,
            np.array([capital_width, abacus_h, capital_width]),
            BASE_COLOR,
            parameters={"order": order.name, "element": "capital_abacus", **metadata},
        )
        y += abacus_h

    return y


def _add_human_access_elements(
    scene: Scene,
    *,
    order_name: str,
    preset_name: str,
    wrap_min_podium_span_x_override: Optional[float],
    wrap_min_podium_span_z_override: Optional[float],
    wrap_front_coverage_ratio_override: Optional[float],
    podium_top_y: float,
    podium_x_min: float,
    podium_x_max: float,
    podium_z_min: float,
    podium_z_max: float,
    envelope: TempleEnvelope,
    front_run: List[float],
    side_run_full: List[float],
    member_width: float,
    apply_symmetry: bool,
    add_main_stairs: bool,
    add_egress_stairs: bool,
    add_markers: bool,
) -> Dict[str, Any]:
    empty_metrics = {
        "entrances": 0,
        "exits": 0,
        "stairs": 0,
        "egress_points": [],
        "main_entrance_center": None,
        "entrance_center_x": None,
        "entrance_clear_width": 0.0,
        "entrance_width": 0.0,
        "side_exit_center_z": None,
        "side_exit_clear_width": 0.0,
        "side_exit_width": 0.0,
        "front_stair_width": 0.0,
        "side_stair_span": 0.0,
        "front_wrap_enabled": False,
        "front_wrap_span_z": 0.0,
        "wrap_trigger_min_span_x": 0.0,
        "wrap_trigger_min_span_z": 0.0,
        "wrap_trigger_front_coverage_ratio": 0.0,
        "front_stair_outer_z": None,
        "left_stair_outer_x": None,
        "right_stair_outer_x": None,
        "doorway_clearances_valid": True,
    }
    if podium_top_y <= 0.0:
        return empty_metrics

    building_center_x = (envelope.x_min + envelope.x_max) / 2.0
    building_center_z = (envelope.z_min + envelope.z_max) / 2.0

    entrance_center_x, entrance_clear_width = _pick_opening_center(
        front_run, member_width, building_center_x
    )
    side_exit_center_z, side_exit_clear_width = _pick_opening_center(
        side_run_full, member_width, building_center_z
    )

    entrance_width = _resolve_doorway_width(
        entrance_clear_width,
        ratio=0.86,
        min_width=1.20,
    )
    side_exit_width = _resolve_doorway_width(
        side_exit_clear_width,
        ratio=0.78,
        min_width=1.00,
    )

    entrance_center_x = _clamp(
        entrance_center_x,
        podium_x_min + (entrance_width / 2.0),
        podium_x_max - (entrance_width / 2.0),
    )
    side_exit_center_z = _clamp(
        side_exit_center_z,
        podium_z_min + (side_exit_width / 2.0),
        podium_z_max - (side_exit_width / 2.0),
    )

    front_coverage_ratio = 0.70 if order_name == "Doric" else 0.64
    front_coverage_ratio += _clamp((len(front_run) - 6) * 0.03, 0.0, 0.14)
    side_coverage_ratio = 0.42 if order_name == "Doric" else 0.38
    side_coverage_ratio += _clamp((len(side_run_full) - 8) * 0.025, 0.0, 0.10)

    front_stair_width = _resolve_stair_span(
        front_run,
        member_width=member_width,
        preferred_center=entrance_center_x,
        podium_min=podium_x_min,
        podium_max=podium_x_max,
        doorway_width=entrance_width,
        coverage_ratio=front_coverage_ratio,
        side_shoulder_ratio=0.42,
        min_columns=4 if len(front_run) >= 4 else 2,
    )
    side_stair_span = _resolve_stair_span(
        side_run_full,
        member_width=member_width,
        preferred_center=side_exit_center_z,
        podium_min=podium_z_min,
        podium_max=podium_z_max,
        doorway_width=side_exit_width,
        coverage_ratio=side_coverage_ratio,
        side_shoulder_ratio=0.30,
        min_columns=2,
    )

    max_front_width = max(
        entrance_width,
        2.0 * min(entrance_center_x - podium_x_min, podium_x_max - entrance_center_x),
    )
    front_stair_width = _clamp(
        max(front_stair_width, entrance_width + member_width * 1.10, member_width * 2.20),
        entrance_width,
        max(entrance_width, max_front_width),
    )

    max_side_span = max(
        side_exit_width,
        2.0 * min(side_exit_center_z - podium_z_min, podium_z_max - side_exit_center_z),
    )
    side_stair_span = _clamp(
        max(side_stair_span, side_exit_width + member_width * 0.90, member_width * 1.90),
        side_exit_width,
        max(side_exit_width, max_side_span),
    )

    stair_rise_target = _clamp(member_width * 0.28, 0.24, 0.36)
    stair_count = max(1, int(math.ceil(podium_top_y / stair_rise_target)))
    step_rise = podium_top_y / float(stair_count)
    front_step_depth = _clamp(
        step_rise * 2.35,
        max(0.72, member_width * 0.62),
        member_width * 1.80,
    )
    side_step_depth = _clamp(
        step_rise * 2.15,
        max(0.68, member_width * 0.58),
        member_width * 1.65,
    )

    podium_span_x = max(0.0, podium_x_max - podium_x_min)
    podium_span_z = max(0.0, podium_z_max - podium_z_min)
    wrap_profile = _wrap_profile_for_preset(preset_name)
    wrap_min_podium_x = float(wrap_profile["min_podium_span_x"])
    wrap_min_podium_z = float(wrap_profile["min_podium_span_z"])
    wrap_coverage_threshold = float(wrap_profile["front_coverage_ratio"])
    if wrap_min_podium_span_x_override is not None:
        wrap_min_podium_x = max(6.0, float(wrap_min_podium_span_x_override))
    if wrap_min_podium_span_z_override is not None:
        wrap_min_podium_z = max(6.0, float(wrap_min_podium_span_z_override))
    if wrap_front_coverage_ratio_override is not None:
        wrap_coverage_threshold = _clamp(float(wrap_front_coverage_ratio_override), 0.30, 0.90)

    front_wrap_enabled = (
        add_main_stairs
        and apply_symmetry
        and podium_span_x >= wrap_min_podium_x
        and podium_span_z >= wrap_min_podium_z
        and front_stair_width >= podium_span_x * wrap_coverage_threshold
    )
    front_wrap_span_z = 0.0
    if front_wrap_enabled:
        wrap_min_ratio = float(wrap_profile["wrap_span_min_ratio"])
        wrap_max_ratio = float(wrap_profile["wrap_span_max_ratio"])
        front_wrap_span_z = _clamp(
            max(side_stair_span, podium_span_z * wrap_min_ratio),
            max(side_exit_width, member_width * 1.80),
            max(side_exit_width, podium_span_z * wrap_max_ratio),
        )

    # Front entrance marker inside the front colonnade edge.
    if add_markers:
        marker_height = 0.08
        _add_cube(
            scene,
            entrance_center_x,
            podium_top_y + marker_height / 2.0,
            podium_z_max - (member_width * 0.44),
            np.array([entrance_width, marker_height, max(0.30, member_width * 0.24)], dtype=float),
            ACCESS_MARKER_COLOR,
            parameters={
                "order": order_name,
                "element": "entrance_marker",
                "circulation_role": "entry",
                "location": "front",
            },
        )

    stair_blocks = 0
    front_stair_outer_z: Optional[float] = None
    if add_main_stairs:
        _add_stair_run(
            scene,
            center_x=entrance_center_x,
            center_z=0.0,
            width_x=front_stair_width,
            width_z=front_step_depth,
            ascend_axis="Z",
            direction_sign=1.0,
            top_edge_coord=podium_z_max,
            top_y=podium_top_y,
            step_count=stair_count,
            color=STAIR_COLOR,
            parameters={
                "order": order_name,
                "element": "stair",
                "stair_role": "entrance",
                "stair_side": "front",
            },
        )
        stair_blocks += stair_count
        front_stair_outer_z = podium_z_max + front_step_depth * stair_count

        if front_wrap_enabled and front_wrap_span_z > 0.0:
            wrap_center_z = podium_z_max - front_wrap_span_z / 2.0
            _add_stair_run(
                scene,
                center_x=0.0,
                center_z=wrap_center_z,
                width_x=side_step_depth,
                width_z=front_wrap_span_z,
                ascend_axis="X",
                direction_sign=-1.0,
                top_edge_coord=podium_x_min,
                top_y=podium_top_y,
                step_count=stair_count,
                color=STAIR_COLOR,
                parameters={
                    "order": order_name,
                    "element": "stair",
                    "stair_role": "entrance",
                    "stair_side": "left_front_wrap",
                    "stair_layout": "wrap",
                },
            )
            _add_stair_run(
                scene,
                center_x=0.0,
                center_z=wrap_center_z,
                width_x=side_step_depth,
                width_z=front_wrap_span_z,
                ascend_axis="X",
                direction_sign=1.0,
                top_edge_coord=podium_x_max,
                top_y=podium_top_y,
                step_count=stair_count,
                color=STAIR_COLOR,
                parameters={
                    "order": order_name,
                    "element": "stair",
                    "stair_role": "entrance",
                    "stair_side": "right_front_wrap",
                    "stair_layout": "wrap",
                },
            )
            stair_blocks += stair_count * 2

    egress_points: List[str] = []
    left_stair_outer_x: Optional[float] = None
    right_stair_outer_x: Optional[float] = None
    if add_egress_stairs:
        _add_stair_run(
            scene,
            center_x=0.0,
            center_z=side_exit_center_z,
            width_x=side_step_depth,
            width_z=side_stair_span,
            ascend_axis="X",
            direction_sign=-1.0,
            top_edge_coord=podium_x_min,
            top_y=podium_top_y,
            step_count=stair_count,
            color=STAIR_COLOR,
            parameters={
                "order": order_name,
                "element": "stair",
                "stair_role": "exit",
                "stair_side": "left",
            },
        )
        stair_blocks += stair_count
        egress_points.append("left")
        left_stair_outer_x = podium_x_min - side_step_depth * stair_count

        if apply_symmetry:
            _add_stair_run(
                scene,
                center_x=0.0,
                center_z=side_exit_center_z,
                width_x=side_step_depth,
                width_z=side_stair_span,
                ascend_axis="X",
                direction_sign=1.0,
                top_edge_coord=podium_x_max,
                top_y=podium_top_y,
                step_count=stair_count,
                color=STAIR_COLOR,
                parameters={
                    "order": order_name,
                    "element": "stair",
                    "stair_role": "exit",
                    "stair_side": "right",
                },
            )
            stair_blocks += stair_count
            egress_points.append("right")
            right_stair_outer_x = podium_x_max + side_step_depth * stair_count

    if add_markers:
        marker_height = 0.08
        _add_cube(
            scene,
            podium_x_min + (member_width * 0.44),
            podium_top_y + marker_height / 2.0,
            side_exit_center_z,
            np.array([max(0.30, member_width * 0.24), marker_height, side_exit_width], dtype=float),
            ACCESS_MARKER_COLOR,
            parameters={
                "order": order_name,
                "element": "exit_marker",
                "circulation_role": "exit",
                "location": "left",
            },
        )
        if apply_symmetry:
            _add_cube(
                scene,
                podium_x_max - (member_width * 0.44),
                podium_top_y + marker_height / 2.0,
                side_exit_center_z,
                np.array([max(0.30, member_width * 0.24), marker_height, side_exit_width], dtype=float),
                ACCESS_MARKER_COLOR,
                parameters={
                    "order": order_name,
                    "element": "exit_marker",
                    "circulation_role": "exit",
                    "location": "right",
                },
            )

    doorway_clearances_valid = True
    if add_main_stairs:
        doorway_clearances_valid = doorway_clearances_valid and (
            entrance_clear_width > 0.0 and entrance_width <= entrance_clear_width + 1e-6
        )
    if add_egress_stairs:
        doorway_clearances_valid = doorway_clearances_valid and (
            side_exit_clear_width > 0.0 and side_exit_width <= side_exit_clear_width + 1e-6
        )

    return {
        "entrances": 1 if add_main_stairs else 0,
        "exits": len(egress_points),
        "stairs": stair_blocks,
        "egress_points": egress_points,
        "main_entrance_center": [round(float(entrance_center_x), 6), round(float(podium_z_max), 6)],
        "entrance_center_x": round(float(entrance_center_x), 6),
        "entrance_clear_width": round(float(entrance_clear_width), 6),
        "entrance_width": round(float(entrance_width), 6),
        "side_exit_center_z": round(float(side_exit_center_z), 6),
        "side_exit_clear_width": round(float(side_exit_clear_width), 6),
        "side_exit_width": round(float(side_exit_width), 6),
        "front_stair_width": round(float(front_stair_width), 6),
        "side_stair_span": round(float(side_stair_span), 6),
        "front_wrap_enabled": bool(front_wrap_enabled),
        "front_wrap_span_z": round(float(front_wrap_span_z), 6),
        "wrap_trigger_min_span_x": round(float(wrap_min_podium_x), 6),
        "wrap_trigger_min_span_z": round(float(wrap_min_podium_z), 6),
        "wrap_trigger_front_coverage_ratio": round(float(wrap_coverage_threshold), 6),
        "front_stair_outer_z": round(float(front_stair_outer_z), 6) if front_stair_outer_z is not None else None,
        "left_stair_outer_x": round(float(left_stair_outer_x), 6) if left_stair_outer_x is not None else None,
        "right_stair_outer_x": round(float(right_stair_outer_x), 6) if right_stair_outer_x is not None else None,
        "doorway_clearances_valid": bool(doorway_clearances_valid),
    }


def _add_foundation_apron(
    scene: Scene,
    *,
    order_name: str,
    podium_top_y: float,
    podium_x_min: float,
    podium_x_max: float,
    podium_z_min: float,
    podium_z_max: float,
    member_width: float,
    access_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if podium_top_y <= 0.0:
        return {"enabled": False, "blocks": 0, "thresholds": 0}

    apron_height = _clamp(podium_top_y * 0.20, 0.18, 0.45)
    threshold_height = _clamp(apron_height * 0.50, 0.08, 0.18)
    toe_depth = _clamp(member_width * 0.20, 0.20, 0.60)
    threshold_depth = _clamp(member_width * 0.18, 0.18, 0.45)

    blocks_added = 0
    threshold_blocks = 0

    def add_apron_block(center_x: float, center_z: float, dimensions: np.ndarray, apron_role: str):
        nonlocal blocks_added
        if dimensions[0] <= 0.05 or dimensions[1] <= 0.01 or dimensions[2] <= 0.05:
            return
        _add_cube(
            scene,
            center_x,
            apron_height / 2.0,
            center_z,
            dimensions,
            APRON_COLOR,
            parameters={"order": order_name, "element": "foundation_apron", "apron_role": apron_role},
        )
        blocks_added += 1

    entrance_center_x = access_metrics.get("entrance_center_x")
    entrance_width = float(access_metrics.get("entrance_width", 0.0) or 0.0)
    front_stair_width = float(access_metrics.get("front_stair_width", 0.0) or 0.0)
    front_outer_z = access_metrics.get("front_stair_outer_z")
    if (
        isinstance(entrance_center_x, (int, float))
        and isinstance(front_outer_z, (int, float))
        and entrance_width > 0.0
    ):
        front_stair_width = max(entrance_width, front_stair_width)
        front_span_z = max(0.0, float(front_outer_z) - podium_z_max)
        if front_span_z > 0.05:
            left_gap = max(0.0, float(entrance_center_x) - front_stair_width / 2.0 - podium_x_min)
            right_gap = max(0.0, podium_x_max - (float(entrance_center_x) + front_stair_width / 2.0))

            if left_gap > 0.05:
                add_apron_block(
                    podium_x_min + left_gap / 2.0,
                    podium_z_max + front_span_z / 2.0,
                    np.array([left_gap, apron_height, front_span_z], dtype=float),
                    "front_left_flank",
                )
            if right_gap > 0.05:
                add_apron_block(
                    podium_x_max - right_gap / 2.0,
                    podium_z_max + front_span_z / 2.0,
                    np.array([right_gap, apron_height, front_span_z], dtype=float),
                    "front_right_flank",
                )

            front_toe_width = _clamp(
                front_stair_width + member_width * 0.40,
                max(front_stair_width, 0.60),
                podium_x_max - podium_x_min,
            )
            add_apron_block(
                float(entrance_center_x),
                float(front_outer_z) + toe_depth / 2.0,
                np.array([front_toe_width, apron_height, toe_depth], dtype=float),
                "front_toe",
            )

        _add_cube(
            scene,
            float(entrance_center_x),
            podium_top_y + threshold_height / 2.0,
            podium_z_max - threshold_depth / 2.0,
            np.array([entrance_width, threshold_height, threshold_depth], dtype=float),
            THRESHOLD_COLOR,
            parameters={"order": order_name, "element": "doorway_threshold", "threshold_side": "front"},
        )
        threshold_blocks += 1

    side_exit_center_z = access_metrics.get("side_exit_center_z")
    side_exit_width = float(access_metrics.get("side_exit_width", 0.0) or 0.0)
    side_stair_span = float(access_metrics.get("side_stair_span", 0.0) or 0.0)
    egress_points = set(access_metrics.get("egress_points", []))

    def add_side_apron(side: str, outer_x: Optional[float]):
        nonlocal threshold_blocks
        if (
            not isinstance(side_exit_center_z, (int, float))
            or side_exit_width <= 0.0
            or not isinstance(outer_x, (int, float))
        ):
            return

        if side == "left":
            span_x = max(0.0, podium_x_min - float(outer_x))
            x_center = podium_x_min - span_x / 2.0
            toe_center_x = float(outer_x) - toe_depth / 2.0
            threshold_x = podium_x_min + threshold_depth / 2.0
            threshold_side = "left"
        else:
            span_x = max(0.0, float(outer_x) - podium_x_max)
            x_center = podium_x_max + span_x / 2.0
            toe_center_x = float(outer_x) + toe_depth / 2.0
            threshold_x = podium_x_max - threshold_depth / 2.0
            threshold_side = "right"

        if span_x <= 0.05:
            return

        stair_span = max(side_exit_width, side_stair_span)
        lower_gap = max(0.0, float(side_exit_center_z) - stair_span / 2.0 - podium_z_min)
        upper_gap = max(0.0, podium_z_max - (float(side_exit_center_z) + stair_span / 2.0))
        if lower_gap > 0.05:
            add_apron_block(
                x_center,
                podium_z_min + lower_gap / 2.0,
                np.array([span_x, apron_height, lower_gap], dtype=float),
                f"{side}_lower_flank",
            )
        if upper_gap > 0.05:
            add_apron_block(
                x_center,
                podium_z_max - upper_gap / 2.0,
                np.array([span_x, apron_height, upper_gap], dtype=float),
                f"{side}_upper_flank",
            )

        side_toe_depth = _clamp(
            stair_span + member_width * 0.40,
            max(stair_span, 0.60),
            podium_z_max - podium_z_min,
        )
        add_apron_block(
            toe_center_x,
            float(side_exit_center_z),
            np.array([toe_depth, apron_height, side_toe_depth], dtype=float),
            f"{side}_toe",
        )

        _add_cube(
            scene,
            threshold_x,
            podium_top_y + threshold_height / 2.0,
            float(side_exit_center_z),
            np.array([threshold_depth, threshold_height, side_exit_width], dtype=float),
            THRESHOLD_COLOR,
            parameters={"order": order_name, "element": "doorway_threshold", "threshold_side": threshold_side},
        )
        threshold_blocks += 1

    if "left" in egress_points:
        add_side_apron("left", access_metrics.get("left_stair_outer_x"))
    if "right" in egress_points:
        add_side_apron("right", access_metrics.get("right_stair_outer_x"))

    return {
        "enabled": True,
        "blocks": int(blocks_added),
        "thresholds": int(threshold_blocks),
        "apron_height": round(float(apron_height), 6),
        "threshold_height": round(float(threshold_height), 6),
    }


def _compute_placement_invariants(
    *,
    footprint_width: float,
    footprint_depth: float,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    front_run: List[float],
    side_run_full: List[float],
    access_metrics: Dict[str, Any],
    require_human_access: bool,
) -> Dict[str, Any]:
    footprint_center_x = float(footprint_width) / 2.0
    footprint_center_z = float(footprint_depth) / 2.0

    colonnade_center_x = (front_run[0] + front_run[-1]) / 2.0 if front_run else footprint_center_x
    colonnade_center_z = (side_run_full[0] + side_run_full[-1]) / 2.0 if side_run_full else footprint_center_z
    center_delta_x = abs(colonnade_center_x - footprint_center_x)
    center_delta_z = abs(colonnade_center_z - footprint_center_z)
    centered_colonnade = center_delta_x <= 0.02 and center_delta_z <= 0.02

    tol = 1e-6
    facade_in_footprint = (
        x_min >= -tol
        and z_min >= -tol
        and x_max <= float(footprint_width) + tol
        and z_max <= float(footprint_depth) + tol
    )

    entrances = int(access_metrics.get("entrances", 0))
    exits = int(access_metrics.get("exits", 0))
    entrance_clear = float(access_metrics.get("entrance_clear_width", 0.0) or 0.0)
    entrance_width = float(access_metrics.get("entrance_width", 0.0) or 0.0)
    side_clear = float(access_metrics.get("side_exit_clear_width", 0.0) or 0.0)
    side_width = float(access_metrics.get("side_exit_width", 0.0) or 0.0)

    main_doorway_valid = True
    if require_human_access and entrances > 0:
        main_doorway_valid = entrance_clear > 0.0 and entrance_width <= entrance_clear + 1e-6

    side_doorway_valid = True
    if require_human_access and exits > 0:
        side_doorway_valid = side_clear > 0.0 and side_width <= side_clear + 1e-6

    doorway_clearances_valid = (
        main_doorway_valid
        and side_doorway_valid
        and bool(access_metrics.get("doorway_clearances_valid", True))
    )

    return {
        "ok": bool(centered_colonnade and facade_in_footprint and doorway_clearances_valid),
        "centered_colonnade": bool(centered_colonnade),
        "facade_in_footprint": bool(facade_in_footprint),
        "doorway_clearances_valid": bool(doorway_clearances_valid),
        "center_delta_x": round(float(center_delta_x), 6),
        "center_delta_z": round(float(center_delta_z), 6),
        "doorway_clearances": {
            "front_clear_width": round(float(entrance_clear), 6),
            "front_passage_width": round(float(entrance_width), 6),
            "side_clear_width": round(float(side_clear), 6),
            "side_passage_width": round(float(side_width), 6),
        },
    }


def _place_colonnade(
    scene: Scene,
    order: GreekOrderSpec,
    run_axis: str,
    run_face: str,
    fixed_coord: float,
    run_coords: List[float],
    podium_top: float,
    column_radius: float,
    shaft_height: float,
    base_height: float,
    base_width: float,
    capital_height: float,
    capital_width: float,
) -> float:
    top_y = podium_top
    for index, coord in enumerate(run_coords):
        if run_axis == "X":
            x = coord
            z = fixed_coord
        else:
            x = fixed_coord
            z = coord

        top_y = max(
            top_y,
            _build_column(
                scene,
                order,
                x=x,
                z=z,
                podium_top=podium_top,
                column_radius=column_radius,
                shaft_height=shaft_height,
                base_height=base_height,
                base_width=base_width,
                capital_height=capital_height,
                capital_width=capital_width,
                column_metadata={
                    "run_axis": run_axis,
                    "run_face": run_face,
                    "run_index": int(index),
                },
            ),
        )
    return top_y


def _add_entablature_layers(
    scene: Scene,
    order: GreekOrderSpec,
    column_top_y: float,
    podium_top_y: float,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    member_width: float,
    front_column_centers: List[float],
    apply_symmetry: bool,
    depth_multiplier: float = 1.0,
    side_overhang_multiplier: float = 1.0,
) -> TempleEnvelope:
    span_x = max(0.1, x_max - x_min)
    span_z = max(0.1, z_max - z_min)

    column_total_height = max(0.1, column_top_y - podium_top_y)
    entablature_height = max(0.8, column_total_height * order.entablature_ratio)
    architrave_h = entablature_height * 0.40
    frieze_h = entablature_height * 0.35
    cornice_h = entablature_height - architrave_h - frieze_h

    depth_scale = _clamp(depth_multiplier, 0.65, 1.8)
    side_scale = _clamp(side_overhang_multiplier, 0.65, 1.8)
    architrave_d = member_width * 0.92 * depth_scale
    frieze_d = member_width * 1.05 * depth_scale
    cornice_d = member_width * (1.28 if order.name == "Doric" else 1.24) * depth_scale
    architrave_side = member_width * 0.92 * side_scale
    frieze_side = member_width * 1.05 * side_scale
    cornice_side = member_width * (1.28 if order.name == "Doric" else 1.24) * side_scale

    current_base = column_top_y + STACK_EPSILON

    layers = [
        ("architrave", architrave_h, architrave_d, architrave_side, ARCHITRAVE_COLOR),
        (
            "frieze",
            frieze_h,
            frieze_d,
            frieze_side,
            DORIC_FRIEZE_COLOR if order.name == "Doric" else IONIC_FRIEZE_COLOR,
        ),
        ("cornice", cornice_h, cornice_d, cornice_side, CORNICE_COLOR),
    ]

    for layer_name, layer_h, layer_d, layer_side, color in layers:
        layer_center_y = _center_y(current_base, layer_h)

        front_z = z_max - layer_d / 2.0
        _add_cube(
            scene,
            (x_min + x_max) / 2.0,
            layer_center_y,
            front_z,
            np.array([span_x + (2.0 * layer_side), layer_h, layer_d]),
            color,
            parameters={"order": order.name, "element": layer_name, "face": "front"},
        )

        if apply_symmetry:
            back_z = z_min + layer_d / 2.0
            _add_cube(
                scene,
                (x_min + x_max) / 2.0,
                layer_center_y,
                back_z,
                np.array([span_x + (2.0 * layer_side), layer_h, layer_d]),
                color,
                parameters={"order": order.name, "element": layer_name, "face": "back"},
            )

            side_span = span_z - (2.0 * layer_d)
            if side_span > 0.05:
                left_x = x_min - layer_side / 2.0
                right_x = x_max + layer_side / 2.0
                _add_cube(
                    scene,
                    left_x,
                    layer_center_y,
                    (z_min + z_max) / 2.0,
                    np.array([layer_side, layer_h, side_span]),
                    color,
                    parameters={"order": order.name, "element": layer_name, "face": "left"},
                )
                _add_cube(
                    scene,
                    right_x,
                    layer_center_y,
                    (z_min + z_max) / 2.0,
                    np.array([layer_side, layer_h, side_span]),
                    color,
                    parameters={"order": order.name, "element": layer_name, "face": "right"},
                )

        if layer_name == "frieze":
            if order.name == "Doric":
                triglyph_depth = max(0.08, member_width * 0.24)
                triglyph_width = max(0.16, member_width * 0.34)
                triglyph_height = layer_h * 0.84
                triglyph_center_y = current_base + layer_h * 0.55
                positions = list(front_column_centers)
                positions.extend((a + b) / 2.0 for a, b in zip(front_column_centers[:-1], front_column_centers[1:]))
                positions.extend([x_min + triglyph_width / 2.0, x_max - triglyph_width / 2.0])

                for x_pos in sorted({round(pos, 4) for pos in positions}):
                    _add_cube(
                        scene,
                        float(x_pos),
                        triglyph_center_y,
                        z_max + triglyph_depth / 2.0,
                        np.array([triglyph_width, triglyph_height, triglyph_depth]),
                        DORIC_TRIGLYPH_COLOR,
                        parameters={"order": order.name, "element": "triglyph", "face": "front"},
                    )
                    if apply_symmetry:
                        _add_cube(
                            scene,
                            float(x_pos),
                            triglyph_center_y,
                            z_min - triglyph_depth / 2.0,
                            np.array([triglyph_width, triglyph_height, triglyph_depth]),
                            DORIC_TRIGLYPH_COLOR,
                            parameters={"order": order.name, "element": "triglyph", "face": "back"},
                        )
            else:
                relief_h = layer_h * 0.32
                relief_d = max(0.08, member_width * 0.16)
                relief_y = current_base + layer_h * 0.58
                _add_cube(
                    scene,
                    (x_min + x_max) / 2.0,
                    relief_y,
                    z_max + relief_d / 2.0,
                    np.array([span_x * 0.94, relief_h, relief_d]),
                    IONIC_RELIEF_COLOR,
                    parameters={"order": order.name, "element": "frieze_relief", "face": "front"},
                )
                if apply_symmetry:
                    _add_cube(
                        scene,
                        (x_min + x_max) / 2.0,
                        relief_y,
                        z_min - relief_d / 2.0,
                        np.array([span_x * 0.94, relief_h, relief_d]),
                        IONIC_RELIEF_COLOR,
                        parameters={"order": order.name, "element": "frieze_relief", "face": "back"},
                    )
                    if side_span > 0.05:
                        _add_cube(
                            scene,
                            x_min - relief_d / 2.0,
                            relief_y,
                            (z_min + z_max) / 2.0,
                            np.array([relief_d, relief_h, side_span * 0.94]),
                            IONIC_RELIEF_COLOR,
                            parameters={"order": order.name, "element": "frieze_relief", "face": "left"},
                        )
                        _add_cube(
                            scene,
                            x_max + relief_d / 2.0,
                            relief_y,
                            (z_min + z_max) / 2.0,
                            np.array([relief_d, relief_h, side_span * 0.94]),
                            IONIC_RELIEF_COLOR,
                            parameters={"order": order.name, "element": "frieze_relief", "face": "right"},
                        )

        current_base += layer_h

    return TempleEnvelope(
        x_min=x_min - cornice_side,
        x_max=x_max + cornice_side,
        z_min=z_min,
        z_max=z_max,
        entablature_top_y=current_base,
        cornice_depth=cornice_d,
    )


def _add_pediment(
    scene: Scene,
    center_x: float,
    base_y: float,
    center_z: float,
    width: float,
    depth: float,
    peak_height: float,
    use_wedges: bool,
    order_name: str,
    face: str,
):
    if width <= 0.0 or depth <= 0.0 or peak_height <= 0.0:
        return

    if use_wedges and width > 0.2:
        wedge_dims = np.array([width / 2.0, peak_height, depth], dtype=float)
        pediment_center_y = _center_y(base_y, peak_height)
        _add_wedge(
            scene,
            center_x - (width / 4.0),
            pediment_center_y,
            center_z,
            wedge_dims,
            "+X",
            PEDIMENT_COLOR,
            parameters={"order": order_name, "element": "pediment", "face": face},
        )
        _add_wedge(
            scene,
            center_x + (width / 4.0),
            pediment_center_y,
            center_z,
            wedge_dims,
            "-X",
            PEDIMENT_COLOR,
            parameters={"order": order_name, "element": "pediment", "face": face},
        )
        return

    layer_count = max(1, int(math.ceil(peak_height)))
    for i in range(layer_count):
        y_bottom = base_y + i
        y_center = y_bottom + 0.5
        normalized = (i + 0.5) / max(peak_height, 1e-6)
        half_width = max(0.0, (width / 2.0) * (1.0 - normalized))
        if half_width <= 0.0:
            continue

        x_start = int(math.floor(center_x - half_width))
        x_end = int(math.ceil(center_x + half_width)) - 1
        for x_coord in range(x_start, x_end + 1):
            _add_cube(
                scene,
                float(x_coord) + 0.5,
                y_center,
                center_z,
                np.array([1.0, 1.0, depth]),
                PEDIMENT_COLOR,
                parameters={"order": order_name, "element": "pediment", "face": face},
            )


def generate_classical_building(
    scene: Scene,
    width: int = 10,
    depth: int = 10,
    podium_height_actual: float = 2.0,
    seed: Optional[int] = None,
    flags: Optional[Dict[str, bool]] = None,
    order_override: Optional[str] = None,
    preset_override: Optional[str] = None,
    front_column_override: Optional[int] = None,
    side_column_override: Optional[int] = None,
    pediment_slope_multiplier: float = 1.0,
    entablature_depth_multiplier: float = 1.0,
    side_roof_overhang_multiplier: float = 1.0,
    wrap_min_podium_span_x_override: Optional[float] = None,
    wrap_min_podium_span_z_override: Optional[float] = None,
    wrap_front_coverage_ratio_override: Optional[float] = None,
    clear_scene: bool = True,
) -> Dict[str, Any]:
    """
    Generate a block-based Greek temple using codified Doric/Ionic rules.

    Geometry conventions:
    - all block positions are geometric centers
    - dimensions are width(X), height(Y), depth(Z)

    Order rules (simplified but style-specific):
    - Doric: thicker/shorter columns, tighter spacing, stronger triglyph frieze rhythm
    - Ionic: slimmer/taller columns, wider spacing, layered base/capital and smoother frieze band

    Human-use circulation philosophy:
    - prioritize a clear primary entrance centered on the front facade
    - maintain at least two independent egress paths where symmetry allows
    - size stair risers/treads proportionally so podiums remain traversable by people
    """
    if clear_scene:
        scene.clear()

    rng = random.Random(seed)
    gen_flags = _merge_flags(flags)
    order_name = _resolve_order(order_override, rng)
    order = ORDER_SPECS[order_name]
    preset = _resolve_preset(preset_override)

    footprint_width = max(6, int(width))
    footprint_depth = max(6, int(depth))

    style_scale = _clamp(min(footprint_width, footprint_depth) / 10.0, 0.65, 1.15)
    diameter = order.base_diameter * style_scale * preset.diameter_scale

    base_height = max(0.06, order.base_height_ratio * diameter)
    base_width = order.base_width_ratio * diameter
    capital_height = max(0.10, order.capital_height_ratio * diameter)
    capital_width = order.capital_width_ratio * diameter
    member_width = max(base_width, capital_width)

    clear_spacing = order.clear_spacing_ratio * diameter * preset.spacing_scale
    edge_margin = order.edge_margin_ratio * diameter

    min_footprint_required = _required_span(4, member_width, clear_spacing, edge_margin)
    min_available_span = float(min(footprint_width, footprint_depth))
    if min_footprint_required > min_available_span:
        scale = min_available_span / min_footprint_required
        diameter *= max(0.55, scale)
        base_height = max(0.06, order.base_height_ratio * diameter)
        base_width = order.base_width_ratio * diameter
        capital_height = max(0.10, order.capital_height_ratio * diameter)
        capital_width = order.capital_width_ratio * diameter
        member_width = max(base_width, capital_width)
        clear_spacing = order.clear_spacing_ratio * diameter * preset.spacing_scale
        edge_margin = order.edge_margin_ratio * diameter

    front_columns = _resolve_column_count(
        span=float(footprint_width),
        member_width=member_width,
        clear_spacing=clear_spacing,
        margin=edge_margin,
        minimum=4,
        maximum=preset.front_max_columns,
        override=front_column_override,
    )
    side_columns = _resolve_column_count(
        span=float(footprint_depth),
        member_width=member_width,
        clear_spacing=clear_spacing,
        margin=edge_margin,
        minimum=4,
        maximum=max(6, min(preset.side_max_columns, front_columns + 4)),
        override=side_column_override,
    )

    front_run = _column_run_coordinates(
        span=float(footprint_width),
        count=front_columns,
        member_width=member_width,
        clear_spacing=clear_spacing,
        margin=edge_margin,
    )
    side_run_full = _column_run_coordinates(
        span=float(footprint_depth),
        count=side_columns,
        member_width=member_width,
        clear_spacing=clear_spacing,
        margin=edge_margin,
    )
    side_run_inner = side_run_full[1:-1] if len(side_run_full) > 2 else []

    left_x = front_run[0]
    right_x = front_run[-1]
    back_z = side_run_full[0]
    front_z = side_run_full[-1]

    x_min = left_x - member_width / 2.0
    x_max = right_x + member_width / 2.0
    z_min = back_z - member_width / 2.0
    z_max = front_z + member_width / 2.0

    shaft_height = max(2.4, order.shaft_height_ratio * diameter * preset.shaft_height_scale)
    column_radius = diameter / 2.0

    podium_height = max(0, int(round(podium_height_actual)))
    podium_top_y = float(podium_height if gen_flags["podium"] else 0.0)

    print(
        f"Generating Classical Building ({order_name})"
        f" seed={seed if seed is not None else 'random'}"
        f" preset={preset.name}"
        f" footprint={footprint_width}x{footprint_depth}"
    )

    original_support_mode = scene.support_check_mode
    if gen_flags.get("disable_support_for_classical", True):
        scene.support_check_mode = "OFF"

    pediment_span_x = 0.0
    pediment_peak_h = 0.0
    snapshot_metrics: Dict[str, Any] = {}
    access_metrics: Dict[str, Any] = {
        "entrances": 0,
        "exits": 0,
        "stairs": 0,
        "egress_points": [],
        "main_entrance_center": None,
        "entrance_center_x": None,
        "entrance_clear_width": 0.0,
        "entrance_width": 0.0,
        "side_exit_center_z": None,
        "side_exit_clear_width": 0.0,
        "side_exit_width": 0.0,
        "front_stair_width": 0.0,
        "side_stair_span": 0.0,
        "front_wrap_enabled": False,
        "front_wrap_span_z": 0.0,
        "wrap_trigger_min_span_x": 0.0,
        "wrap_trigger_min_span_z": 0.0,
        "wrap_trigger_front_coverage_ratio": 0.0,
        "front_stair_outer_z": None,
        "left_stair_outer_x": None,
        "right_stair_outer_x": None,
        "doorway_clearances_valid": True,
    }
    apron_metrics: Dict[str, Any] = {"enabled": False, "blocks": 0, "thresholds": 0}
    placement_invariants: Dict[str, Any] = {
        "ok": True,
        "centered_colonnade": True,
        "facade_in_footprint": True,
        "doorway_clearances_valid": True,
        "center_delta_x": 0.0,
        "center_delta_z": 0.0,
        "doorway_clearances": {
            "front_clear_width": 0.0,
            "front_passage_width": 0.0,
            "side_clear_width": 0.0,
            "side_passage_width": 0.0,
        },
    }

    try:
        if gen_flags["podium"] and podium_height > 0:
            generate_podium(
                scene,
                width=footprint_width,
                depth=footprint_depth,
                height=podium_height,
                clear_scene=False,
                base_position=np.array([0.0, 0.0, 0.0]),
            )

        highest_column_top = podium_top_y
        if gen_flags["columns"]:
            highest_column_top = max(
                highest_column_top,
                _place_colonnade(
                    scene,
                    order=order,
                    run_axis="X",
                    run_face="front",
                    fixed_coord=front_z,
                    run_coords=front_run,
                    podium_top=podium_top_y,
                    column_radius=column_radius,
                    shaft_height=shaft_height,
                    base_height=base_height,
                    base_width=base_width,
                    capital_height=capital_height,
                    capital_width=capital_width,
                ),
            )

            if gen_flags["apply_symmetry"]:
                highest_column_top = max(
                    highest_column_top,
                    _place_colonnade(
                        scene,
                        order=order,
                        run_axis="X",
                        run_face="back",
                        fixed_coord=back_z,
                        run_coords=front_run,
                        podium_top=podium_top_y,
                        column_radius=column_radius,
                        shaft_height=shaft_height,
                        base_height=base_height,
                        base_width=base_width,
                        capital_height=capital_height,
                        capital_width=capital_width,
                    ),
                )

                if side_run_inner:
                    highest_column_top = max(
                        highest_column_top,
                        _place_colonnade(
                            scene,
                            order=order,
                            run_axis="Z",
                            run_face="left",
                            fixed_coord=left_x,
                            run_coords=side_run_inner,
                            podium_top=podium_top_y,
                            column_radius=column_radius,
                            shaft_height=shaft_height,
                            base_height=base_height,
                            base_width=base_width,
                            capital_height=capital_height,
                            capital_width=capital_width,
                        ),
                    )
                    highest_column_top = max(
                        highest_column_top,
                        _place_colonnade(
                            scene,
                            order=order,
                            run_axis="Z",
                            run_face="right",
                            fixed_coord=right_x,
                            run_coords=side_run_inner,
                            podium_top=podium_top_y,
                            column_radius=column_radius,
                            shaft_height=shaft_height,
                            base_height=base_height,
                            base_width=base_width,
                            capital_height=capital_height,
                            capital_width=capital_width,
                        ),
                    )

        envelope = TempleEnvelope(
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            entablature_top_y=highest_column_top,
            cornice_depth=member_width,
        )

        if gen_flags["columns"] and gen_flags["entablature"]:
            envelope = _add_entablature_layers(
                scene,
                order=order,
                column_top_y=highest_column_top,
                podium_top_y=podium_top_y,
                x_min=x_min,
                x_max=x_max,
                z_min=z_min,
                z_max=z_max,
                member_width=member_width,
                front_column_centers=front_run,
                apply_symmetry=gen_flags["apply_symmetry"],
                depth_multiplier=preset.entablature_depth_scale * entablature_depth_multiplier,
                side_overhang_multiplier=side_roof_overhang_multiplier,
            )

        if gen_flags["columns"] and gen_flags["pediment"]:
            pediment_span_x = max(0.1, envelope.x_max - envelope.x_min)
            pediment_peak_h = _pediment_peak_height(
                pediment_span_x,
                order.pediment_rise_over_half_span,
                preset.pediment_slope_scale * pediment_slope_multiplier,
            )
            pediment_depth = max(0.12, envelope.cornice_depth * 0.95)
            pediment_base_y = envelope.entablature_top_y
            center_x = (envelope.x_min + envelope.x_max) / 2.0

            _add_pediment(
                scene,
                center_x=center_x,
                base_y=pediment_base_y,
                center_z=envelope.z_max + pediment_depth / 2.0,
                width=pediment_span_x,
                depth=pediment_depth,
                peak_height=pediment_peak_h,
                use_wedges=gen_flags["use_wedge_pediment"],
                order_name=order.name,
                face="front",
            )

            if gen_flags["apply_symmetry"]:
                _add_pediment(
                    scene,
                    center_x=center_x,
                    base_y=pediment_base_y,
                    center_z=envelope.z_min - pediment_depth / 2.0,
                    width=pediment_span_x,
                    depth=pediment_depth,
                    peak_height=pediment_peak_h,
                    use_wedges=gen_flags["use_wedge_pediment"],
                    order_name=order.name,
                    face="back",
                )

                if gen_flags["roof"]:
                    roof_depth = max(0.1, envelope.z_max - envelope.z_min)
                    slice_count = max(1, int(math.ceil(roof_depth)))
                    slice_depth = roof_depth / slice_count
                    for idx in range(slice_count):
                        z_center = envelope.z_min + slice_depth * (idx + 0.5)
                        _add_pediment(
                            scene,
                            center_x=center_x,
                            base_y=pediment_base_y,
                            center_z=z_center,
                            width=pediment_span_x,
                            depth=slice_depth,
                            peak_height=pediment_peak_h,
                            use_wedges=gen_flags["use_wedge_pediment"],
                            order_name=order.name,
                            face="roof",
                        )

                    ridge_h = max(0.10, envelope.cornice_depth * 0.20)
                    _add_cube(
                        scene,
                        center_x,
                        _center_y(pediment_base_y + pediment_peak_h, ridge_h),
                        (envelope.z_min + envelope.z_max) / 2.0,
                        np.array([pediment_span_x * 0.12, ridge_h, roof_depth]),
                        ROOF_COLOR,
                        parameters={"order": order.name, "element": "ridge"},
                    )

        if gen_flags.get("human_access", True):
            access_metrics = _add_human_access_elements(
                scene,
                order_name=order.name,
                preset_name=preset.name,
                wrap_min_podium_span_x_override=wrap_min_podium_span_x_override,
                wrap_min_podium_span_z_override=wrap_min_podium_span_z_override,
                wrap_front_coverage_ratio_override=wrap_front_coverage_ratio_override,
                podium_top_y=podium_top_y,
                podium_x_min=0.0,
                podium_x_max=float(footprint_width),
                podium_z_min=0.0,
                podium_z_max=float(footprint_depth),
                envelope=envelope,
                front_run=front_run,
                side_run_full=side_run_full,
                member_width=member_width,
                apply_symmetry=gen_flags["apply_symmetry"],
                add_main_stairs=gen_flags.get("main_stairs", True),
                add_egress_stairs=gen_flags.get("egress_stairs", True),
                add_markers=gen_flags.get("entrance_markers", True),
            )

        if gen_flags.get("foundation_apron", False):
            apron_metrics = _add_foundation_apron(
                scene,
                order_name=order.name,
                podium_top_y=podium_top_y,
                podium_x_min=0.0,
                podium_x_max=float(footprint_width),
                podium_z_min=0.0,
                podium_z_max=float(footprint_depth),
                member_width=member_width,
                access_metrics=access_metrics,
            )

        placement_invariants = _compute_placement_invariants(
            footprint_width=float(footprint_width),
            footprint_depth=float(footprint_depth),
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            front_run=front_run,
            side_run_full=side_run_full,
            access_metrics=access_metrics,
            require_human_access=bool(gen_flags.get("human_access", True)),
        )
        if gen_flags.get("validate_invariants", True) and not placement_invariants["ok"]:
            print(f"Warning: Classical placement invariant failure: {placement_invariants}")

        bounds_min, bounds_max = _scene_bounds(scene)
        snapshot_metrics = {
            "order": order.name,
            "preset": preset.name,
            "front_columns": int(front_columns),
            "side_columns": int(side_columns),
            "facade_span": round(float(max(0.0, x_max - x_min)), 6),
            "pediment_peak_height": round(float(pediment_peak_h), 6),
            "block_count": int(len(scene.blocks)),
            "bounds_min": bounds_min.round(6).tolist(),
            "bounds_max": bounds_max.round(6).tolist(),
            "avg_column_radius": round(float(column_radius), 6),
            "avg_column_slenderness": round(float(shaft_height / max(1e-8, 2.0 * column_radius)), 6),
            "side_roof_overhang_multiplier": round(float(side_roof_overhang_multiplier), 6),
            "human_access": {
                "philosophy": "entry-centered circulation, multi-egress safety, proportional stairs",
                "entrances": int(access_metrics["entrances"]),
                "exits": int(access_metrics["exits"]),
                "stairs": int(access_metrics["stairs"]),
                "egress_points": list(access_metrics["egress_points"]),
                "main_entrance_center": access_metrics["main_entrance_center"],
                "stair_proportions": {
                    "front_stair_width": float(access_metrics["front_stair_width"]),
                    "side_stair_span": float(access_metrics["side_stair_span"]),
                    "front_wrap_enabled": bool(access_metrics["front_wrap_enabled"]),
                    "front_wrap_span_z": float(access_metrics["front_wrap_span_z"]),
                    "wrap_trigger_min_span_x": float(access_metrics["wrap_trigger_min_span_x"]),
                    "wrap_trigger_min_span_z": float(access_metrics["wrap_trigger_min_span_z"]),
                    "wrap_trigger_front_coverage_ratio": float(
                        access_metrics["wrap_trigger_front_coverage_ratio"]
                    ),
                },
                "doorway_clearances": {
                    "front_clear_width": float(access_metrics["entrance_clear_width"]),
                    "front_passage_width": float(access_metrics["entrance_width"]),
                    "side_clear_width": float(access_metrics["side_exit_clear_width"]),
                    "side_passage_width": float(access_metrics["side_exit_width"]),
                },
            },
            "foundation_apron": apron_metrics,
            "invariants": placement_invariants,
        }

    finally:
        scene.support_check_mode = original_support_mode

    scene.last_classical_metrics = snapshot_metrics
    return snapshot_metrics
