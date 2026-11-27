"""Deterministic quality gates for semantic architecture builds."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import BuildDocument, canonical_product_name
from .substances import SUBSTANCES
from .physics import PhysicalReport, physical_report


PASS = "pass"
WARNING = "warning"
FAIL = "fail"


@dataclass
class QualityCheck:
    code: str
    status: str
    severity: str
    message: str
    primitive_ids: List[str] = field(default_factory=list)
    observed: Any = None
    expected: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": str(self.code),
            "status": str(self.status),
            "severity": str(self.severity),
            "message": str(self.message),
            "primitive_ids": sorted(str(item) for item in self.primitive_ids),
            "observed": _json_value(self.observed),
            "expected": _json_value(self.expected),
        }


@dataclass
class QualityReport:
    status: str
    counts: Dict[str, int]
    checks: List[QualityCheck]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": str(self.status),
            "counts": {
                str(key): int(value)
                for key, value in sorted(self.counts.items())
            },
            "checks": [check.to_dict() for check in self.checks],
            "metrics": _json_value(self.metrics),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check(
    code: str,
    passed: bool,
    message_pass: str,
    message_fail: str,
    *,
    primitive_ids: Iterable[str] = (),
    observed: Any = None,
    expected: Any = None,
    failure_status: str = FAIL,
    failure_severity: str = "error",
) -> QualityCheck:
    return QualityCheck(
        code=code,
        status=PASS if passed else failure_status,
        severity="info" if passed else failure_severity,
        message=message_pass if passed else message_fail,
        primitive_ids=sorted(str(item) for item in primitive_ids),
        observed=observed,
        expected=expected,
    )


def _finite_vector(values: Sequence[Any], length: int) -> bool:
    if len(values) != length:
        return False
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _scene_mesh_records(scene: Any) -> List[Tuple[str, Any, Any]]:
    records: List[Tuple[str, Any, Any]] = []
    for index, block in enumerate(getattr(scene, "blocks", ())):
        parameters = getattr(block, "parameters", {}) or {}
        primitive_id = str(parameters.get("primitive_id", f"scene-block-{index}"))
        records.append((primitive_id, block, getattr(block, "mesh", None)))
    return records


def _counter_dict(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _material_key(material: Any) -> str:
    try:
        return ",".join(str(channel) for channel in material)
    except TypeError:
        return "<invalid>"


def _placement_bounds(placement: Any) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(placement.position, dtype=float)
    half = np.asarray(placement.dimensions, dtype=float) / 2.0
    return center - half, center + half


def _placements_touch(left: Any, right: Any, tolerance: float = 1e-6) -> bool:
    left_min, left_max = _placement_bounds(left)
    right_min, right_max = _placement_bounds(right)
    return bool(
        np.all(left_max + tolerance >= right_min)
        and np.all(right_max + tolerance >= left_min)
    )


def _touch_graph_is_connected(placements: Sequence[Any], tolerance: float = 1e-4) -> bool:
    if not placements:
        return False
    visited = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for candidate in range(len(placements)):
            if candidate in visited:
                continue
            if _placements_touch(placements[current], placements[candidate], tolerance):
                visited.add(candidate)
                frontier.append(candidate)
    return len(visited) == len(placements)


def _spire_domain_checks(document: BuildDocument) -> Tuple[List[QualityCheck], Dict[str, Any]]:
    """Verify that a spire cluster's claimed routes connect real envelopes."""

    placements = list(document.placements)
    platforms: Dict[str, List[Any]] = {}
    for placement in placements:
        if placement.role != "circulation.spire_platform":
            continue
        key = str((placement.parameters or {}).get("host_id") or placement.id)
        platforms.setdefault(key, []).append(placement)
    route_groups: Dict[int, List[Any]] = {}
    for placement in placements:
        if placement.role != "circulation.spire_bridge":
            continue
        try:
            route_index = int(placement.parameters["route_index"])
        except (KeyError, TypeError, ValueError):
            route_index = -1
        route_groups.setdefault(route_index, []).append(placement)

    invalid_route_ids: List[str] = []
    connected_platform_edges: List[Tuple[str, str]] = []
    for route_index, route in sorted(route_groups.items()):
        route.sort(key=lambda item: int(item.parameters.get("route_segment", 0)))
        first = route[0]
        from_id = str(first.parameters.get("from_platform_id", ""))
        to_id = str(first.parameters.get("to_platform_id", ""))
        start = platforms.get(from_id) or []
        end = platforms.get(to_id) or []
        route_valid = (
            route_index >= 0
            and bool(start)
            and bool(end)
            and all(str(item.parameters.get("from_platform_id", "")) == from_id for item in route)
            and all(str(item.parameters.get("to_platform_id", "")) == to_id for item in route)
            and _touch_graph_is_connected([*start, *route, *end])
        )
        if route_valid:
            connected_platform_edges.append((from_id, to_id))
        else:
            invalid_route_ids.extend(str(item.id) for item in route)

    platform_adjacency: Dict[str, set[str]] = {platform_id: set() for platform_id in platforms}
    for left, right in connected_platform_edges:
        platform_adjacency[left].add(right)
        platform_adjacency[right].add(left)
    visited_platforms: set[str] = set()
    if platform_adjacency:
        frontier = [sorted(platform_adjacency)[0]]
        while frontier:
            current = frontier.pop()
            if current in visited_platforms:
                continue
            visited_platforms.add(current)
            frontier.extend(sorted(platform_adjacency[current] - visited_platforms))
    bridge_graph_connected = (
        len(platforms) >= 2
        and not invalid_route_ids
        and len(visited_platforms) == len(platforms)
    )
    bridge_check = _check(
        "domain.spire_bridge_connectivity",
        bridge_graph_connected,
        "Bridge envelopes form a connected platform graph.",
        "Bridge envelopes do not connect every declared platform endpoint.",
        primitive_ids=sorted(set(invalid_route_ids) | (set(platforms) - visited_platforms)),
        observed={
            "platforms": len(platforms),
            "connected_platforms": len(visited_platforms),
            "valid_routes": len(connected_platform_edges),
        },
        expected={
            "connected_platforms": len(platforms),
            "minimum_routes": max(0, len(platforms) - 1),
        },
    )

    site_stones = [item for item in placements if item.role == "site.spire_pad"]
    access_steps = sorted(
        (
            item
            for item in placements
            if item.role == "circulation.stair_step"
            and item.parameters.get("connects_platform_id")
        ),
        key=lambda item: int(item.parameters.get("step_index", 0)),
    )
    target_id = str(access_steps[0].parameters.get("connects_platform_id", "")) if access_steps else ""
    target = platforms.get(target_id) or []
    access_connected = (
        bool(site_stones)
        and bool(target)
        and bool(access_steps)
        and _touch_graph_is_connected([*site_stones, *access_steps, *target])
    )
    access_ids = [str(item.id) for item in access_steps]
    access_ids.extend(str(item.id) for item in site_stones)
    access_ids.extend(str(item.id) for item in target)
    access_check = _check(
        "domain.spire_ground_access",
        access_connected,
        "A continuous stair envelope connects the site pad to the platform graph.",
        "The access stair does not continuously connect ground and a spire platform.",
        primitive_ids=[] if access_connected else access_ids,
        observed={"steps": len(access_steps), "target_platform": target_id or None},
        expected={"steps": ">=1", "target_platform": "connected"},
    )
    return [bridge_check, access_check], {
        "spire_platform_count": len(platforms),
        "spire_bridge_route_count": len(route_groups),
        "spire_connected_platform_count": len(visited_platforms),
        "spire_access_step_count": len(access_steps),
    }


def _metric(metrics: Mapping[str, Any], path: str) -> Any:
    current: Any = metrics
    for segment in str(path).split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None
        current = current[segment]
    return current


def _compare(observed: Any, operator: str, expected: Any) -> bool:
    try:
        if operator in {"==", "eq"}:
            return observed == expected
        if operator in {"!=", "ne"}:
            return observed != expected
        if operator in {">", "gt"}:
            return observed > expected
        if operator in {">=", "gte", "min"}:
            return observed >= expected
        if operator in {"<", "lt"}:
            return observed < expected
        if operator in {"<=", "lte", "max"}:
            return observed <= expected
        if operator in {"contains", "includes"}:
            return expected in observed
        if operator in {"contains_all", "includes_all"}:
            return set(expected).issubset(set(observed))
    except (TypeError, ValueError):
        return False
    return False


def _declared_invariant_checks(
    declarations: Any,
    metrics: Mapping[str, Any],
    document: BuildDocument,
) -> List[QualityCheck]:
    if not declarations:
        return []

    if isinstance(declarations, Mapping):
        items = [(str(key), value) for key, value in sorted(declarations.items())]
    elif isinstance(declarations, list):
        items = []
        for index, value in enumerate(declarations):
            if isinstance(value, Mapping):
                code = str(value.get("code", f"declared_{index:03d}"))
            else:
                code = f"declared_{index:03d}"
            items.append((code, value))
    else:
        items = [("declaration", declarations)]

    role_names = set(metrics.get("role_counts", {}))
    kind_names = set(metrics.get("kind_counts", {}))
    block_type_names = set(metrics.get("block_type_counts", {}))
    checks: List[QualityCheck] = []

    known_metrics = {
        "placement_count": "placement_count",
        "compiled_block_count": "compiled_block_count",
        "mesh_count": "mesh_count",
        "watertight_mesh_count": "watertight_mesh_count",
    }

    for name, declaration in items:
        code = f"invariant.{name}"
        observed: Any = None
        expected: Any = declaration
        passed = False
        supported = True

        if isinstance(declaration, bool):
            observed = bool(declaration)
            expected = True
            passed = observed is True
        elif name in {"min_placements", "minimum_placements"}:
            observed = metrics.get("placement_count")
            passed = _compare(observed, ">=", declaration)
        elif name in {"max_placements", "maximum_placements"}:
            observed = metrics.get("placement_count")
            passed = _compare(observed, "<=", declaration)
        elif name in {"placement_count", "compiled_block_count", "mesh_count"}:
            observed = metrics.get(known_metrics[name])
            passed = _compare(observed, "==", declaration)
        elif name in {"require_ground_contact", "ground_contact"}:
            observed = bool(metrics.get("ground_contact", False))
            expected = bool(declaration)
            passed = observed == expected
        elif name in {"require_roles", "required_roles"}:
            observed = sorted(role_names)
            expected = sorted(str(item) for item in declaration)
            passed = set(expected).issubset(role_names)
        elif name in {"require_kinds", "required_kinds"}:
            observed = sorted(kind_names)
            expected = sorted(str(item) for item in declaration)
            passed = set(expected).issubset(kind_names)
        elif name in {"require_block_types", "required_block_types"}:
            observed = sorted(block_type_names)
            expected = sorted(str(item) for item in declaration)
            passed = set(expected).issubset(block_type_names)
        elif name in {"parent_links_valid", "parents_valid"}:
            observed = bool(metrics.get("parent_links_valid", False))
            expected = bool(declaration)
            passed = observed == expected
        elif isinstance(declaration, Mapping):
            metric_name = str(declaration.get("metric", name))
            operator = str(declaration.get("operator", declaration.get("op", "=="))).lower()
            expected = declaration.get("expected", declaration.get("value"))
            observed = _metric(metrics, metric_name)
            if expected is None and "expected" not in declaration and "value" not in declaration:
                supported = False
            else:
                passed = _compare(observed, operator, expected)
        elif name in metrics:
            observed = metrics[name]
            passed = _compare(observed, "==", declaration)
        else:
            supported = False

        if not supported:
            checks.append(
                QualityCheck(
                    code=code,
                    status=FAIL,
                    severity="error",
                    message=f"Declared invariant '{name}' could not be evaluated.",
                    primitive_ids=[],
                    observed=None,
                    expected=declaration,
                )
            )
            continue

        checks.append(
            _check(
                code,
                passed,
                f"Declared invariant '{name}' passed.",
                f"Declared invariant '{name}' failed.",
                observed=observed,
                expected=expected,
            )
        )

    return checks


def classical_form_checks(document: BuildDocument, product_name: str) -> List[QualityCheck]:
    """Intrinsic checks survive removal of user-editable invariant metadata.

    Envelope connectivity is a conservative geometric diagnostic: it does not
    prove mesh contact inside arches or certify a physical load path.
    """
    from .classical_forms import FORMS

    placements = document.placements
    required = {"foundation", *FORMS[product_name][2]}
    observed = {p.role for p in placements}
    checks = [_check("classical.required_roles", required <= observed,
                     "Typology-defining components are present.",
                     "The classical form is missing required components.",
                     observed=sorted(observed), expected=sorted(required))]
    checks.extend(_envelope_connectivity_checks(document, "classical"))
    return checks


def _aabb_overlap_indices(lo, hi, index, tolerance, candidates=None):
    """Indices whose AABBs touch lo/hi[index] within *tolerance*."""
    if candidates is None:
        return np.flatnonzero(
            np.all(hi + tolerance >= lo[index], axis=1)
            & np.all(lo - tolerance <= hi[index], axis=1)
        )
    if not candidates:
        return np.empty(0, dtype=np.intp)
    picked = np.fromiter(candidates, dtype=np.intp, count=len(candidates))
    return picked[
        np.all(hi[picked] + tolerance >= lo[index], axis=1)
        & np.all(lo[picked] - tolerance <= hi[index], axis=1)
    ]


def _aabb_touch_neighbors(lo, hi, tolerance):
    """AABB-touch adjacency: uniform grid of expanded boxes; large boxes vs all."""
    count = int(lo.shape[0])
    neighbors: List[List[int]] = [[] for _ in range(count)]
    if count == 0:
        return neighbors
    tol = float(tolerance)
    max_extent = np.maximum(np.max(hi - lo, axis=1), 0.0)
    cell = max(float(np.median(max_extent)), 10.0 * tol, 1e-6)
    q0 = np.floor((lo - tol) / cell).astype(np.int64)
    q1 = np.floor((hi + tol) / cell).astype(np.int64)
    span = (q1 - q0 + 1).astype(np.float64)
    large_mask = (span[:, 0] * span[:, 1] * span[:, 2]) > 256.0
    large_ids = np.flatnonzero(large_mask)

    if large_ids.size:
        chunk = 64
        for start in range(0, int(large_ids.size), chunk):
            ids = large_ids[start:start + chunk]
            touch = (
                np.all(hi[None, :, :] + tol >= lo[ids, None, :], axis=2)
                & np.all(lo[None, :, :] - tol <= hi[ids, None, :], axis=2)
            )
            for local, index in enumerate(ids):
                index = int(index)
                for other in np.flatnonzero(touch[local]):
                    other = int(other)
                    if other == index:
                        continue
                    neighbors[index].append(other)
                    if not large_mask[other]:
                        neighbors[other].append(index)

    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for index in range(count):
        if large_mask[index]:
            continue
        x0, y0, z0 = int(q0[index, 0]), int(q0[index, 1]), int(q0[index, 2])
        x1, y1, z1 = int(q1[index, 0]), int(q1[index, 1]), int(q1[index, 2])
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    buckets.setdefault((x, y, z), []).append(index)

    for index in range(count):
        if large_mask[index]:
            continue
        candidates = set()
        x0, y0, z0 = int(q0[index, 0]), int(q0[index, 1]), int(q0[index, 2])
        x1, y1, z1 = int(q1[index, 0]), int(q1[index, 1]), int(q1[index, 2])
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    bucket = buckets.get((x, y, z))
                    if bucket:
                        candidates.update(other for other in bucket if other > index)
        if not candidates:
            continue
        for other in _aabb_overlap_indices(lo, hi, index, tol, candidates):
            other = int(other)
            neighbors[index].append(other)
            neighbors[other].append(index)
    return neighbors


def _envelope_connectivity_checks(document, domain):
    placements = document.placements
    if not placements or any(not _finite_vector(p.position, 3) or
                             not _finite_vector(p.dimensions, 3) for p in placements):
        return []
    bounds = [_placement_bounds(p) for p in placements]
    lo = np.array([item[0] for item in bounds])
    hi = np.array([item[1] for item in bounds])
    # Legacy column stacks include intentional 1e-5 separation between layers.
    tolerance = 1e-4
    reached = set(np.flatnonzero((lo[:, 1] <= tolerance) & (hi[:, 1] >= -tolerance)).tolist())
    pending = list(reached)
    neighbors = _aabb_touch_neighbors(lo, hi, tolerance)
    while pending:
        current = pending.pop()
        for candidate in neighbors[current]:
            if candidate not in reached:
                reached.add(candidate)
                pending.append(candidate)
    detached = [p.id for i, p in enumerate(placements) if i not in reached]
    return [_check(f"{domain}.envelope_connectivity", not detached,
                         "Every component envelope connects to ground.",
                         "Some component envelopes are detached from ground.",
                         primitive_ids=detached, observed=len(detached), expected=0)]


def cathedral_checks(document):
    """Intrinsic composition checks remain active after metadata is removed."""
    from .cathedral import REQUIRED_ROLES, KINDS
    placements = document.placements
    observed = {p.role for p in placements}
    checks = [_check("cathedral.required_roles", REQUIRED_ROLES <= observed,
        "Cathedral-defining components are present.", "The cathedral is missing defining components.",
        observed=sorted(observed), expected=sorted(REQUIRED_ROLES))]
    checks.extend(_envelope_connectivity_checks(document, "cathedral"))
    invalid = [p.id for p in placements if p.role not in KINDS or p.kind != KINDS.get(p.role)]
    checks.append(_check("cathedral.member_semantics", not invalid,
        "Every cathedral member has a recognized role and matching coarse kind.",
        "Cathedral member semantics are inconsistent.", primitive_ids=invalid))
    # Mirrored primary structural members must retain their positions and sizes.
    primary = {"nave_pier", "buttress_pier", "flying_buttress", "west_tower"}
    signatures = {(p.role, tuple(round(float(v),4) for v in p.position),
                   tuple(round(float(v),4) for v in p.dimensions)) for p in placements if p.role in primary}
    asymmetric = [p.id for p in placements if p.role in primary and
        (p.role, tuple(round(float(v),4) for v in (-p.position[0],p.position[1],p.position[2])),
         tuple(round(float(v),4) for v in p.dimensions)) not in signatures]
    checks.append(_check("cathedral.bilateral_composition", not asymmetric,
        "Primary cathedral members balance about the nave axis.",
        "Primary cathedral members lost their paired composition.", primitive_ids=asymmetric))
    return checks


def audit_build(document: BuildDocument, scene: Any, physical: Optional[PhysicalReport] = None) -> QualityReport:
    """Audit a compiled build in a stable, hard-gated check order."""
    checks: List[QualityCheck] = []
    placements = list(document.placements)
    placement_ids = [str(placement.id) for placement in placements]
    id_counts = Counter(placement_ids)
    duplicate_ids = sorted(item for item, count in id_counts.items() if count > 1)

    checks.append(
        _check(
            "document.unique_ids",
            not duplicate_ids and all(placement_ids),
            "All primitive ids are non-empty and unique.",
            "Primitive ids must be non-empty and unique.",
            primitive_ids=duplicate_ids,
            observed={"duplicates": duplicate_ids, "empty_ids": placement_ids.count("")},
            expected={"duplicates": [], "empty_ids": 0},
        )
    )

    invalid_positions: List[str] = []
    invalid_dimensions: List[str] = []
    for placement in placements:
        if not _finite_vector(placement.position, 3):
            invalid_positions.append(str(placement.id))
        if not _finite_vector(placement.dimensions, 3) or not all(
            float(value) > 0.0 for value in placement.dimensions
        ):
            invalid_dimensions.append(str(placement.id))

    invalid_transforms = sorted(set(invalid_positions + invalid_dimensions))
    checks.append(
        _check(
            "document.transforms",
            not invalid_transforms,
            "All placement transforms are finite with positive dimensions.",
            "Placements contain non-finite coordinates or non-positive dimensions.",
            primitive_ids=invalid_transforms,
            observed={
                "invalid_positions": sorted(invalid_positions),
                "invalid_dimensions": sorted(invalid_dimensions),
            },
            expected={"invalid_positions": [], "invalid_dimensions": []},
        )
    )

    invalid_materials: List[str] = []
    for placement in placements:
        try:
            material = tuple(placement.material)
            valid_material = (
                len(material) == 4
                and all(
                    not isinstance(channel, bool) and isinstance(channel, Integral)
                    for channel in material
                )
                and all(0 <= int(channel) <= 255 for channel in material)
            )
        except TypeError:
            valid_material = False
        if not valid_material:
            invalid_materials.append(str(placement.id))
    checks.append(
        _check(
            "document.materials",
            not invalid_materials,
            "All placement materials are valid RGBA colors.",
            "Materials must contain four integer RGBA channels in the range 0..255.",
            primitive_ids=invalid_materials,
            observed={"invalid_material_count": len(invalid_materials)},
            expected={"invalid_material_count": 0},
        )
    )

    missing_substances: List[str] = []
    unknown_substances: List[str] = []
    for placement in placements:
        substance_id = getattr(placement, "substance", None)
        if not isinstance(substance_id, str) or not substance_id:
            missing_substances.append(str(placement.id))
        elif substance_id not in SUBSTANCES:
            unknown_substances.append(str(placement.id))
    # A document that carries no substances at all predates them and stays
    # valid; a partly assigned one is an authoring error.
    predates_substances = len(missing_substances) == len(placements)
    checks.append(
        _check(
            "document.substances",
            not unknown_substances and (not missing_substances or predates_substances),
            "Every placement carries a registered substance, or the document predates substances.",
            "Every placement must carry a substance id from the substance registry.",
            primitive_ids=sorted(set(unknown_substances if predates_substances
                                     else missing_substances + unknown_substances)),
            observed={
                "missing": len(missing_substances),
                "unknown": len(unknown_substances),
                "predates_substances": predates_substances,
            },
            expected={"missing": 0, "unknown": 0},
        )
    )

    placement_signatures: Dict[Tuple[Any, ...], str] = {}
    duplicate_geometry: List[str] = []
    for placement in placements:
        signature = (
            str(placement.block_type),
            *(round(float(value), 6) for value in placement.position),
            *(round(float(value), 6) for value in placement.dimensions),
        )
        if signature in placement_signatures:
            duplicate_geometry.extend([placement_signatures[signature], str(placement.id)])
        else:
            placement_signatures[signature] = str(placement.id)
    duplicate_geometry = sorted(set(duplicate_geometry))
    checks.append(
        _check(
            "document.unique_geometry",
            not duplicate_geometry,
            "No placements duplicate the same primitive envelope.",
            "Two or more placements duplicate the same primitive envelope.",
            primitive_ids=duplicate_geometry,
            observed={"duplicate_primitive_count": len(duplicate_geometry)},
            expected={"duplicate_primitive_count": 0},
        )
    )

    records = _scene_mesh_records(scene)
    compiled_ids = [record[0] for record in records]
    placement_counter = Counter(placement_ids)
    compiled_counter = Counter(compiled_ids)
    parity_ok = placement_counter == compiled_counter
    parity_ids = sorted((placement_counter - compiled_counter) + (compiled_counter - placement_counter))
    checks.append(
        _check(
            "compile.parity",
            parity_ok,
            "Every placement compiled to exactly one scene block.",
            "Compiled scene blocks do not match document placements.",
            primitive_ids=parity_ids,
            observed={"document": dict(sorted(placement_counter.items())), "scene": dict(sorted(compiled_counter.items()))},
            expected=dict(sorted(placement_counter.items())),
        )
    )

    required_tags = ("primitive_id", "kind", "architect", "component_role", "parent_id")
    semantic_failures: List[str] = []
    placement_by_id = {str(placement.id): placement for placement in placements}
    for primitive_id, block, _mesh in records:
        parameters = getattr(block, "parameters", {}) or {}
        placement = placement_by_id.get(primitive_id)
        missing = [key for key in required_tags if key not in parameters]
        mismatch = (
            placement is not None
            and (
                parameters.get("kind") != placement.kind
                or parameters.get("architect") != document.architect.id
                or parameters.get("component_role") != placement.role
                or parameters.get("parent_id") != placement.parent_id
            )
        )
        if missing or mismatch:
            semantic_failures.append(primitive_id)
    checks.append(
        _check(
            "compile.semantic_tags",
            not semantic_failures,
            "Compiled blocks retain all semantic architecture tags.",
            "Compiled blocks are missing or contradict semantic architecture tags.",
            primitive_ids=semantic_failures,
            observed={"invalid_count": len(semantic_failures)},
            expected={"invalid_count": 0},
        )
    )

    empty_meshes: List[str] = []
    nonfinite_meshes: List[str] = []
    nonwatertight_meshes: List[str] = []
    inconsistent_winding_meshes: List[str] = []
    nonpositive_volume_meshes: List[str] = []
    finite_vertices: List[np.ndarray] = []
    vertex_count = 0
    face_count = 0
    total_mesh_volume = 0.0
    for primitive_id, _block, mesh in records:
        if mesh is None or bool(getattr(mesh, "is_empty", True)):
            empty_meshes.append(primitive_id)
            continue
        vertices = np.asarray(getattr(mesh, "vertices", np.empty((0, 3))), dtype=float)
        faces = np.asarray(getattr(mesh, "faces", np.empty((0, 3))))
        if vertices.size == 0 or faces.size == 0:
            empty_meshes.append(primitive_id)
            continue
        vertex_count += int(len(vertices))
        face_count += int(len(faces))
        winding_consistent = bool(getattr(mesh, "is_winding_consistent", False))
        if not winding_consistent:
            inconsistent_winding_meshes.append(primitive_id)
        else:
            try:
                volume = float(mesh.volume)
            except (TypeError, ValueError):
                volume = float("nan")
            if not math.isfinite(volume) or volume <= 1e-12:
                nonpositive_volume_meshes.append(primitive_id)
            else:
                total_mesh_volume += volume
        if not bool(np.isfinite(vertices).all()):
            nonfinite_meshes.append(primitive_id)
        else:
            finite_vertices.append(vertices)
        if not bool(getattr(mesh, "is_watertight", False)):
            nonwatertight_meshes.append(primitive_id)

    checks.append(
        _check(
            "mesh.nonempty",
            not empty_meshes and len(records) == len(placements),
            "Every compiled primitive has non-empty mesh geometry.",
            "One or more compiled primitives have empty mesh geometry.",
            primitive_ids=empty_meshes,
            observed={"empty": len(empty_meshes), "mesh_records": len(records)},
            expected={"empty": 0, "mesh_records": len(placements)},
        )
    )
    checks.append(
        _check(
            "mesh.finite",
            not nonfinite_meshes,
            "All mesh vertices are finite.",
            "One or more meshes contain non-finite vertices.",
            primitive_ids=nonfinite_meshes,
            observed=len(nonfinite_meshes),
            expected=0,
        )
    )
    checks.append(
        _check(
            "mesh.watertight",
            not nonwatertight_meshes and bool(records),
            "All compiled meshes are watertight.",
            "One or more compiled meshes are not watertight.",
            primitive_ids=nonwatertight_meshes,
            observed={"watertight": len(records) - len(nonwatertight_meshes), "total": len(records)},
            expected={"watertight": len(records), "total": len(records)},
        )
    )
    checks.append(
        _check(
            "mesh.winding",
            not inconsistent_winding_meshes and bool(records),
            "All compiled meshes have consistent face winding.",
            "One or more compiled meshes have inconsistent face winding.",
            primitive_ids=inconsistent_winding_meshes,
            observed={
                "consistent": len(records) - len(inconsistent_winding_meshes),
                "total": len(records),
            },
            expected={"consistent": len(records), "total": len(records)},
        )
    )
    checks.append(
        _check(
            "mesh.positive_volume",
            not nonpositive_volume_meshes and bool(records),
            "All consistently wound meshes enclose positive volume.",
            "One or more meshes have zero, negative, or non-finite volume.",
            primitive_ids=nonpositive_volume_meshes,
            observed={"invalid_volume_count": len(nonpositive_volume_meshes)},
            expected={"invalid_volume_count": 0},
        )
    )

    # Physical properties and the vertical-load diagnostic.  Reported, not a
    # hard gate: an over-utilised member warns without failing the build.
    if physical is None:
        physical = physical_report(document, scene)
    max_utilisation = physical.max_utilisation
    max_member = physical.max_utilisation_member_id
    checks.append(
        _check(
            "structure.vertical_load",
            not physical.members_over_utilisation,
            "Every member with a resolved bed bears its accumulated vertical "
            f"load within a factor of safety (max utilisation {max_utilisation:.3f}"
            f"{' at ' + str(max_member) if max_member else ''}"
            f"; {physical.unresolved_bearing_count} member(s) without a resolved bed excluded).",
            "Bearing stress exceeds the factored compressive strength of "
            f"{len(physical.members_over_utilisation)} member(s); max utilisation "
            f"{max_utilisation:.3f}"
            f"{' at ' + str(max_member) if max_member else ''}.",
            primitive_ids=physical.members_over_utilisation,
            observed={
                "max_member_utilisation": round(max_utilisation, 6),
                "max_utilisation_member_id": max_member,
                "members_over_utilisation": list(physical.members_over_utilisation),
                "members_without_resolved_bearing": int(physical.unresolved_bearing_count),
                "total_mass_kg": round(physical.total_mass_kg, 3),
            },
            expected={"max_member_utilisation": 1.0},
            failure_status=WARNING,
            failure_severity="warning",
        )
    )

    bounds_min: Optional[np.ndarray] = None
    bounds_max: Optional[np.ndarray] = None
    if finite_vertices:
        stacked = np.vstack(finite_vertices)
        bounds_min = stacked.min(axis=0)
        bounds_max = stacked.max(axis=0)

    precision = int((document.coordinate_system or {}).get("precision", 6))
    ground_tolerance = max(1e-6, 10.0 ** (-max(1, precision)))
    min_y = None if bounds_min is None else float(bounds_min[1])
    ground_contact = min_y is not None and abs(min_y) <= ground_tolerance
    checks.append(
        _check(
            "build.ground_contact",
            ground_contact,
            "Build geometry contacts the Y=0 ground plane.",
            "Build geometry must contact the Y=0 ground plane.",
            observed=min_y,
            expected={"ground_y": 0.0, "tolerance": ground_tolerance},
        )
    )

    known_ids = set(placement_ids)
    bad_parent_ids = sorted(
        str(placement.id)
        for placement in placements
        if placement.parent_id is not None and str(placement.parent_id) not in known_ids
    )
    parent_links_valid = not bad_parent_ids
    checks.append(
        _check(
            "document.parent_links",
            parent_links_valid,
            "All declared parent ids reference document primitives.",
            "One or more parent ids do not reference a document primitive.",
            primitive_ids=bad_parent_ids,
            observed=len(bad_parent_ids),
            expected=0,
        )
    )

    parent_map = {
        str(placement.id): None if placement.parent_id is None else str(placement.parent_id)
        for placement in placements
    }
    cyclic_parent_ids: set[str] = set()
    for start_id in sorted(parent_map):
        path: List[str] = []
        path_index: Dict[str, int] = {}
        current: Optional[str] = start_id
        while current is not None and current in parent_map:
            if current in path_index:
                cyclic_parent_ids.update(path[path_index[current]:])
                break
            path_index[current] = len(path)
            path.append(current)
            current = parent_map[current]
    checks.append(
        _check(
            "document.parent_acyclic",
            not cyclic_parent_ids,
            "Placement parent relationships are acyclic.",
            "Placement parent relationships contain a cycle.",
            primitive_ids=sorted(cyclic_parent_ids),
            observed={"cyclic_primitive_count": len(cyclic_parent_ids)},
            expected={"cyclic_primitive_count": 0},
        )
    )

    from .classical_forms import FORMS
    product_name = canonical_product_name(document.request.product)
    if product_name in FORMS:
        checks.extend(classical_form_checks(document, product_name))
    if product_name == "gothic-cathedral":
        checks.extend(cathedral_checks(document))
    from .historic import historic_checks
    checks.extend(historic_checks(document))
    if product_name == "marble-run":
        from .marble_run import marble_run_checks
        checks.extend(marble_run_checks(document))
    if product_name == "timber-house":
        from .timberwright import timber_checks
        checks.extend(timber_checks(document, scene))
    if product_name == "iwan-court":
        from .persian import persian_checks
        checks.extend(persian_checks(document, scene))
    if product_name in {"english-village", "roman-town"}:
        from .settlement import settlement_checks
        checks.extend(settlement_checks(document, scene))
    if product_name == "victorian-terrace":
        from .victorian import victorian_checks
        checks.extend(victorian_checks(document, scene))
    if product_name in FORMS or product_name in {"classical-building", "classical-temple", "kitbashed-building", "kitbash-series"}:
        from .construction import order_checks
        checks.extend(order_checks(document, scene))
    from .masonry import ashlar_checks
    checks.extend(ashlar_checks(document))
    if product_name in {"classical-building", "classical-temple"}:
        required = {"podium", "column_shaft", "column_capital_abacus", "architrave", "frieze", "cornice", "pediment", "roof_panel"}
        observed = {p.role for p in placements}
        checks.append(_check("classical.temple_members", required <= observed,
            "The temple has its foundation, order and roof members.",
            "The temple is missing a defining member.", expected=sorted(required), observed=sorted(observed)))
        checks.extend(_envelope_connectivity_checks(document, "classical"))

    domain_metrics: Dict[str, Any] = {}
    if product_name == "kitbashed-building":
        from .agent import parts_connected
        checks.append(_check(
            "kitbash.part_connectivity", parts_connected(placements),
            "Kitbash part envelopes form one connected assembly.",
            "Kitbash parts are disconnected or missing part IDs.",
        ))
    if product_name == "spire-cluster":
        domain_checks, domain_metrics = _spire_domain_checks(document)
        checks.extend(domain_checks)

    metrics: Dict[str, Any] = {
        "document_fingerprint": document.fingerprint(),
        "placement_count": len(placements),
        "unique_placement_id_count": len(set(placement_ids)),
        "compiled_block_count": len(records),
        "mesh_count": len(records) - len(empty_meshes),
        "finite_mesh_count": len(records) - len(empty_meshes) - len(nonfinite_meshes),
        "watertight_mesh_count": len(records) - len(empty_meshes) - len(nonwatertight_meshes),
        "consistent_winding_mesh_count": len(records) - len(empty_meshes) - len(inconsistent_winding_meshes),
        "positive_volume_mesh_count": len(records) - len(empty_meshes) - len(inconsistent_winding_meshes) - len(nonpositive_volume_meshes),
        "ground_contact": bool(ground_contact),
        "ground_min_y": min_y,
        "parent_links_valid": parent_links_valid,
        "parent_links_acyclic": not cyclic_parent_ids,
        "duplicate_geometry_count": len(duplicate_geometry),
        "vertex_count": vertex_count,
        "triangle_count": face_count,
        "total_mesh_volume": round(total_mesh_volume, precision),
        "kind_counts": _counter_dict(placement.kind for placement in placements),
        "block_type_counts": _counter_dict(placement.block_type for placement in placements),
        "role_counts": _counter_dict(placement.role for placement in placements),
        "material_counts": _counter_dict(
            _material_key(placement.material) for placement in placements
        ),
        "substance_counts": _counter_dict(
            str(getattr(placement, "substance", "") or "<unassigned>")
            for placement in placements
        ),
        "physical": physical.to_dict(),
        "bounds_min": None if bounds_min is None else bounds_min.round(precision).tolist(),
        "bounds_max": None if bounds_max is None else bounds_max.round(precision).tolist(),
        "bounds_size": None if bounds_min is None else (bounds_max - bounds_min).round(precision).tolist(),
    }
    metrics.update(domain_metrics)

    checks.extend(
        _declared_invariant_checks(
            document.metadata.get("invariants"),
            metrics,
            document,
        )
    )

    # Playability for the request's player profile. Requests without a profile
    # skip this entirely, so their reports are exactly what they were.
    if getattr(document.request, "profile", None) is not None and scene is not None:
        from .gameplay import audit as _playability

        try:
            game_checks, game_metrics, _grid = _playability(document, scene)
        except Exception as exc:  # report, never crash the audit
            game_checks = [QualityCheck("game.voxelize", WARNING, "warning",
                                        f"Playability could not be measured: {exc}")]
            game_metrics = {"error": str(exc)}
        checks.extend(game_checks)
        metrics["playability"] = game_metrics

    status_counts = Counter(check.status for check in checks)
    report_status = FAIL if status_counts[FAIL] else WARNING if status_counts[WARNING] else PASS
    counts = {
        "total": len(checks),
        PASS: int(status_counts[PASS]),
        WARNING: int(status_counts[WARNING]),
        FAIL: int(status_counts[FAIL]),
    }
    return QualityReport(
        status=report_status,
        counts=counts,
        checks=checks,
        metrics=metrics,
    )


__all__ = [
    "PASS",
    "WARNING",
    "FAIL",
    "QualityCheck",
    "QualityReport",
    "audit_build",
    "classical_form_checks",
]
