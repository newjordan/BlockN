"""Settlement planner: streets, plots, and whole buildings from existing architects.

A settlement is a BuildDocument. It does not invent building meshes. It lays
out streets and plots, calls the normal authoring path for each plot, and
places that document into the town. Rotation is a quarter-turn about Y so
every member stays axis-aligned.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement, canonical_product_name


ARCHITECT = ArchitectInfo("settlement", "1.0.0", "Settlement Planner")
ENGLISH = "english-village"
ROMAN = "roman-town"
PRODUCT_NAMES = frozenset({ENGLISH, ROMAN})
ENGLISH_VARIANTS = ("street-village", "green-village", "market-town")
ROMAN_VARIANTS = ("forum-quarter", "residential-quarter", "colonia-grid")
DETAILS = ("concept", "standard", "presentation")
_ALLOWED_PARAMETERS = frozenset({"plot_overrides", "manor"})
_OVERRIDE_KEYS = frozenset({
    "seed", "variant", "width", "height", "depth", "bays", "levels", "detail", "parameters", "product",
})
# Ground, boundaries and pavements are the town, not a second building.
# Roles must not contain "wall": masonry treats that substring as ashlar.
_INFRA_ROLES = frozenset({
    "street", "kerb", "stepping_stone", "square", "plot", "garden",
    "pavement", "podium", "fence", "enclosure",
})
_YAW = {"south": 0, "east": 90, "north": 180, "west": 270}
_CS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
_WEDGE = ("+X", "+Z", "-X", "-Z")
_FRONT_PROBE = 0.9
_OVERLAP_AREA = 0.02

# Quarter-turn of a normalised mesh vertex. Compile scales vertices by the
# member dimensions, so the vertex swap and the dimension swap are one rotation.
_VERTEX = (
    lambda x, y, z: (x, y, z),
    lambda x, y, z: (-z, y, x),
    lambda x, y, z: (-x, y, -z),
    lambda x, y, z: (z, y, -x),
)

# The default is the smallest site on which that variant still reads as a town.
# Larger sites grow the rows; they do not open the green's short side.
DEFAULT_SITES = {
    "street-village": (100.0, 72.0),
    "green-village": (120.0, 96.0),
    "market-town": (120.0, 68.0),
    "forum-quarter": (90.0, 90.0),
    "residential-quarter": (90.0, 90.0),
    "colonia-grid": (128.0, 128.0),
}

_EARTH = (138, 116, 78, 255)
_PAVING = (186, 178, 160, 255)
_KERB = (150, 144, 132, 255)
_STEP = (170, 164, 150, 255)
_YARD = (118, 108, 78, 255)
_GRASS = (108, 124, 80, 255)
_OAK = (120, 78, 46, 255)
_WALL = (168, 162, 148, 255)
_PODIUM_COLOR = (176, 170, 154, 255)
_LANE = 4.0
_MINOR = 4.5
_MAJOR = 8.0
_KERB_W = 0.22
_PAVE_W = 1.15
_PAVING_T = 0.16
_EARTH_T = 0.12
_KERB_H = 0.14
_PAVE_H = 0.12
_YARD_T = 0.10
_SETBACK = 0.40
_YARD_MIN = 1.4
_PLOT_NARROW = 10.4
_PLOT_WIDE = 11.7
_FRONT_REACH = 4.0
_COVERAGE_MIN = 0.995
_FRONTAGE_MIN = 0.65
_INSULA_MIN = 0.45
_STOA_D = 6.8

_CACHE: Dict[str, BuildDocument] = {}


def list_products() -> List[Dict[str, Any]]:
    """Catalog records for the two settlement products."""
    from .castle import DEFAULT_SUBSTANCE as CASTLE_DEFAULT
    from .castle import SUBSTANCES as CASTLE_SUBSTANCES
    from .cathedral import CATHEDRAL_DEFAULT_SUBSTANCE, CATHEDRAL_SUBSTANCES
    from .classical_forms import FORM_DEFAULT_SUBSTANCE, FORM_SUBSTANCES
    from .timberwright import MATERIAL_SUBSTANCES, ROOF_SUBSTANCES

    english = sorted(
        set(MATERIAL_SUBSTANCES.values()) | set(ROOF_SUBSTANCES.values())
        | set(CATHEDRAL_SUBSTANCES.values()) | {CATHEDRAL_DEFAULT_SUBSTANCE}
        | set(CASTLE_SUBSTANCES.values()) | {CASTLE_DEFAULT, "earth-fill", "limestone"}
    )
    roman = sorted(set(FORM_SUBSTANCES.values()) | {FORM_DEFAULT_SUBSTANCE, "earth-fill", "limestone"})
    return [
        _descriptor(
            ENGLISH,
            "Medieval English nucleated village: continuous burgage frontage, a churchyard and lanes",
            DEFAULT_SITES["street-village"],
            ENGLISH_VARIANTS,
            english,
            {"frontage_m": [6, 12], "toft_depth_m": [30, 60], "lane_width_m": [3, 5]},
        ),
        _descriptor(
            ROMAN,
            "Roman provincial town quarter: cardo, decumanus, built-out insulae and a forum",
            DEFAULT_SITES["forum-quarter"],
            ROMAN_VARIANTS,
            roman,
            {"insula_m": [35, 75], "major_street_m": [6, 10], "side_street_m": [4, 6]},
        ),
    ]


def _descriptor(name, description, size, variants, substances, rules):
    width, depth = size
    return {
        "name": name,
        "category": "assembly",
        "architect": ARCHITECT.to_dict(),
        "architect_id": ARCHITECT.id,
        "architect_title": ARCHITECT.title,
        "architect_version": ARCHITECT.version,
        "description": description,
        "defaults": {"width": width, "depth": depth, "variant": variants[0], "detail": "standard"},
        "variants": ["auto", *variants],
        "detail_levels": list(DETAILS),
        "opening_modes": ["auto"],
        "parameters": ["manor", "plot_overrides"],
        "substances": substances,
        "rules": {
            "width": [90, 200],
            "depth": [68, 200],
            **rules,
            "scope": "starter settlement for a building game; not a surveyed town plan",
        },
    }


def random_controls(product: str, variant: Optional[str] = None, rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """A buildable site for the GUI randomize button. The chosen variant stays."""
    rng = rng or random.Random()
    product = canonical_product_name(product)
    options = ENGLISH_VARIANTS if product == ENGLISH else ROMAN_VARIANTS
    chosen = options[0] if variant in (None, "", "auto") else variant
    if chosen not in options:
        chosen = options[0]
    width, depth = DEFAULT_SITES[chosen]
    # Deltas stay non-negative: the default site is already the small legal one.
    return {
        "width": int(width + rng.choice((0, 6, 12))),
        "depth": int(depth + rng.choice((0, 4, 8))),
        "height": 1,
        "detail": "concept",
        "variant": chosen,
        "bays": None,
        "levels": None,
    }


def build_document(request: BuildRequest) -> BuildDocument:
    """Lay out one settlement and place each plot's authored building."""
    product = canonical_product_name(request.product)
    if product not in PRODUCT_NAMES:
        raise ValueError(f"unknown settlement '{request.product}'")
    if request.detail not in DETAILS:
        raise ValueError("detail must be concept, standard, or presentation")
    unknown = set(request.parameters or {}) - _ALLOWED_PARAMETERS
    if unknown:
        raise ValueError(
            f"unknown settlement parameters: {sorted(unknown)}; expected manor, plot_overrides"
        )
    overrides = request.parameters.get("plot_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("plot_overrides must be an object keyed by plot id")
    for plot_id, override in overrides.items():
        if not isinstance(override, dict):
            raise ValueError(f"plot override '{plot_id}' must be an object")
        extra = set(override) - _OVERRIDE_KEYS
        if extra:
            raise ValueError(f"plot override '{plot_id}' has unknown fields {sorted(extra)}")
    manor = request.parameters.get("manor", None)
    if manor is not None and not isinstance(manor, bool):
        raise ValueError("manor must be a boolean")

    variants = ENGLISH_VARIANTS if product == ENGLISH else ROMAN_VARIANTS
    variant = variants[0] if request.variant in ("auto", "", None) else str(request.variant)
    if variant not in variants:
        raise ValueError(f"{product} variant must be auto or one of {', '.join(variants)}")
    width = _span("width", request.width, DEFAULT_SITES[variant][0])
    depth = _span("depth", request.depth, DEFAULT_SITES[variant][1])
    if product == ENGLISH:
        plan = _english(variant, width, depth, manor, int(request.seed or 0))
    else:
        plan = _roman(variant, width, depth)

    placements: List[Placement] = []
    plot_records = []
    for plot in plan["plots"]:
        override = overrides.get(plot["id"]) if isinstance(overrides.get(plot["id"]), dict) else {}
        plot_seed = _plot_seed(int(request.seed or 0), plot["id"], override)
        # An override is the request that gets built. Otherwise a timber plot
        # rolls its own variant and finish from the plot seed.
        child = _child_request(plot, override, plot_seed, request.detail)
        if child is not None and request.profile is not None and child.profile is None:
            # Every building in the settlement is made for the same player.
            child.profile = request.profile
        _yard_id, members, bounds = _realize_plot(plot, child)
        placements.extend(members)
        record = {
            "id": plot["id"],
            "product": None if child is None else child.product,
            "frontage_m": round(float(plot["frontage"]), 3),
            "depth_m": round(float(plot["depth"]), 3),
            "facing": plot["facing"],
            "yaw": _YAW[plot["facing"]],
            "plot_seed": plot_seed,
            "kind": plot["kind"],
        }
        if plot.get("insula") is not None:
            record["insula_m"] = round(float(plot["insula"]), 3)
        if plot.get("insula_id"):
            record["insula_id"] = plot["insula_id"]
        plot_records.append(record)
        if bounds is not None:
            _require_inside(plot["id"], bounds, plot["rect"])

    placements.extend(_boundary_members(plan["plots"]))
    for street in plan["streets"]:
        placements.extend(_street_members(street))
    for square in plan["squares"]:
        placements.append(_square_member(square))

    resolved = BuildRequest(
        architect=ARCHITECT.id,
        product=product,
        seed=int(request.seed or 0),
        width=width,
        height=None,
        depth=depth,
        style="contemporary",
        detail=request.detail,
        variant=variant,
        bays=None,
        levels=None,
        openings="auto",
        parameters=dict(request.parameters or {}),
        age=request.age,
        profile=request.profile,
    )
    metadata = {
        "settlement": {
            "tradition": product,
            "variant": variant,
            "typology": plan["typology"],
            "site": {"width": width, "depth": depth},
            "plots": plot_records,
            "streets": plan["street_records"],
            "frontage_edges": plan.get("frontage_edges") or [],
            "insulae": plan.get("insulae") or [],
            "frontage_rule": (
                "Each called architect's entrance is its local -Z side. "
                "After a quarter-turn about Y, that side must have a street or square within a metre."
            ),
        },
        "invariants": {"required_roles": ["plot", "street"], "require_ground_contact": True},
    }
    return BuildDocument(architect=ARCHITECT, request=resolved, placements=placements, metadata=metadata)


def turn_members(
    placements: Sequence[Placement],
    yaw: int,
    anchor: Sequence[float],
) -> List[Placement]:
    """Rotate placements by a quarter-turn about Y. Positions, sizes, meshes, and stone axes move together."""
    steps = _steps(yaw)
    origin = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
    turned = []
    for placement in placements:
        position, dimensions, parameters = _rotate_member(
            placement.position, placement.dimensions, dict(placement.parameters or {}),
            placement.block_type, steps, origin,
        )
        turned.append(Placement(
            id=placement.id,
            kind=placement.kind,
            block_type=placement.block_type,
            position=position,
            dimensions=dimensions,
            material=placement.material,
            role=placement.role,
            parameters=parameters,
            parent_id=placement.parent_id,
            substance=placement.substance,
        ))
    return turned


def unturn_members(
    placements: Sequence[Placement],
    yaw: int,
    anchor: Sequence[float],
) -> List[Placement]:
    """Inverse of :func:`turn_members` for the same yaw and anchor."""
    steps = _steps(yaw)
    inverse = (4 - steps) % 4
    origin = (float(anchor[0]), float(anchor[1]), float(anchor[2]))
    restored = []
    for placement in placements:
        dimensions, parameters = _rotate_payload(
            placement.dimensions, dict(placement.parameters or {}), placement.block_type, inverse,
        )
        position = _inverse_position(placement.position, steps, origin)
        restored.append(Placement(
            id=placement.id,
            kind=placement.kind,
            block_type=placement.block_type,
            position=position,
            dimensions=dimensions,
            material=placement.material,
            role=placement.role,
            parameters=parameters,
            parent_id=placement.parent_id,
            substance=placement.substance,
        ))
    return restored


def settlement_checks(document: BuildDocument, scene: Any) -> list:
    """Overlap, frontage, streets, density, and each plot's own architect gates."""
    from .quality import _check

    product = canonical_product_name(document.request.product)
    if product not in PRODUCT_NAMES:
        return []
    plots = [placement for placement in document.placements if placement.role == "plot"]
    streets = [placement for placement in document.placements if placement.role == "street"]
    squares = [placement for placement in document.placements if placement.role == "square"]
    by_plot: Dict[str, List[Placement]] = {}
    for placement in document.placements:
        plot_id = (placement.parameters or {}).get("plot_id")
        if plot_id and placement.role not in _INFRA_ROLES:
            by_plot.setdefault(str(plot_id), []).append(placement)
    width = float(document.request.width or 0)
    depth = float(document.request.depth or 0)
    checks = [
        _overlap_check(_check, by_plot, streets, squares),
        _frontage_check(_check, plots, by_plot, streets, squares),
        _network_check(_check, streets, width, depth),
        _building_quality_check(_check, plots, by_plot, scene),
        _ground_coverage_check(_check, document, by_plot, width, depth),
        _street_frontage_check(_check, document, by_plot),
        _insula_coverage_check(_check, document, plots, by_plot),
    ]
    return checks


# ---------------------------------------------------------------- rotation


def _steps(yaw: int) -> int:
    quarter = int(yaw)
    if quarter not in (0, 90, 180, 270) or quarter % 90:
        raise ValueError("settlement rotation must be 0, 90, 180, or 270 degrees about Y")
    return quarter // 90


def _rotate_member(position, dimensions, parameters, block_type, steps, anchor):
    dimensions, parameters = _rotate_payload(dimensions, parameters, block_type, steps)
    x, y, z = (float(v) for v in position)
    c, s = _CS[steps]
    world = (
        anchor[0] + x * c - z * s,
        anchor[1] + y,
        anchor[2] + x * s + z * c,
    )
    return world, dimensions, parameters


def _inverse_position(position, steps, anchor) -> Tuple[float, float, float]:
    c, s = _CS[steps]
    dx = float(position[0]) - anchor[0]
    dz = float(position[2]) - anchor[2]
    return (
        dx * c + dz * s,
        float(position[1]) - anchor[1],
        -dx * s + dz * c,
    )


def _rotate_payload(dimensions, parameters, block_type, steps):
    width, height, depth = (float(v) for v in dimensions)
    if steps % 2:
        width, depth = depth, width
    if steps:
        if block_type == "MESH" and "vertices" in parameters:
            turn = _VERTEX[steps]
            parameters["vertices"] = [list(turn(float(x), float(y), float(z))) for x, y, z in parameters["vertices"]]
        axis = parameters.get("axis")
        if steps % 2 and axis in ("x", "z"):
            parameters["axis"] = "z" if axis == "x" else "x"
        orientation = parameters.get("orientation")
        if block_type == "WEDGE" and orientation in _WEDGE:
            parameters["orientation"] = _WEDGE[(_WEDGE.index(orientation) + steps) % 4]
    return (width, height, depth), parameters


# ---------------------------------------------------------------- plans


def _span(name: str, value: Any, default: float) -> float:
    number = default if value is None else value
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
        raise ValueError(f"settlement {name} must be a finite length in metres")
    number = float(number)
    if not 36 <= number <= 200:
        raise ValueError(f"settlement {name} must be between 36 and 200 metres")
    return number


def _plot_seed(settlement_seed: int, plot_id: str, override: Dict[str, Any]) -> int:
    if override and "seed" in override:
        seed = override["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"plot override '{plot_id}' seed must be an integer")
        return seed
    digest = hashlib.sha256(f"{settlement_seed}:{plot_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _retarget(request: BuildRequest, plot_seed: int, detail: str) -> BuildRequest:
    """Detail follows the settlement. Seed-insensitive civic buildings keep a fixed seed so they cache."""
    data = request.to_dict()
    data["detail"] = detail
    data["age"] = 0
    if data["product"] == "timber-house":
        data["seed"] = plot_seed
    return BuildRequest.from_dict(data)


def _apply_override(request: BuildRequest, override: Dict[str, Any]) -> BuildRequest:
    if not override:
        return request
    data = request.to_dict()
    for key in ("seed", "variant", "width", "height", "depth", "bays", "levels", "detail", "product"):
        if key in override:
            data[key] = override[key]
    if "parameters" in override:
        merged = dict(data.get("parameters") or {})
        merged.update(override["parameters"])
        data["parameters"] = merged
    data["age"] = 0
    return BuildRequest.from_dict(data)


def _child_request(plot: Dict[str, Any], override: Dict[str, Any], plot_seed: int, detail: str) -> Optional[BuildRequest]:
    """The request actually sent to the architect. Overrides win over the rolled house."""
    template = plot.get("request")
    if template is None:
        return None
    if override:
        child = _apply_override(template, override)
        data = child.to_dict()
        data["age"] = 0
        if "detail" not in override:
            data["detail"] = detail
        if "seed" not in override and data["product"] == "timber-house":
            data["seed"] = plot_seed
        return BuildRequest.from_dict(data)
    if template.product == "timber-house" and plot["kind"] in {"house", "hall"}:
        return _vary_house(template, plot_seed, detail, float(plot["frontage"]))
    return _retarget(template, plot_seed, detail)


def _require_site(variant: str, width: float, depth: float) -> None:
    need_w, need_d = DEFAULT_SITES[variant]
    if width < need_w - 1e-6 or depth < need_d - 1e-6:
        raise ValueError(
            f"{variant} needs a site of at least {need_w:.0f} by {need_d:.0f} m "
            f"(got {width:.0f} by {depth:.0f})"
        )


def _english(variant: str, width: float, depth: float, manor: Optional[bool], seed: int) -> Dict[str, Any]:
    _require_site(variant, width, depth)
    if variant == "green-village":
        return _english_green(width, depth, True if manor is None else manor, seed)
    if variant == "market-town":
        return _english_market(width, depth, seed)
    return _english_street(width, depth, seed)


def _house_choice(plot_seed: int) -> Dict[str, Any]:
    """One timber frame for this plot. The same seed picks the same frame at layout and at build."""
    rng = random.Random(plot_seed)
    two_storey = rng.random() < 0.34
    variant = rng.choice(("longhouse", "gable-front", "twin-gable"))
    finish = rng.choice(("chalk-slate", "honey-clay", "rose-shale"))
    porch = rng.random() < 0.70
    chimney = rng.random() < 0.85
    if two_storey:
        spec = {"plot_w": _PLOT_WIDE, "width": 9, "depth": 9, "height": 11, "levels": 2}
    else:
        spec = {"plot_w": _PLOT_NARROW, "width": 8, "depth": 8, "height": 7, "levels": 1}
    spec.update(variant=variant, finish=finish, porch=porch, chimney=chimney)
    return spec


def _house_request(spec: Dict[str, Any]) -> BuildRequest:
    return BuildRequest(
        product="timber-house", seed=0,
        width=spec["width"], depth=spec["depth"], height=spec["height"],
        levels=spec["levels"], bays=2, variant=spec["variant"], detail="concept", age=0,
        parameters={"finish": spec["finish"], "porch": spec["porch"], "chimney": spec["chimney"]},
    )


def _pack_houses(seed, origin, limit, cross0, cross1, facing, index, min_remainder):
    """Burgage plots along one street edge. `origin`/`limit` run along the street."""
    along_x = facing in ("south", "north")
    plots = []
    cursor = origin
    stop = limit - min_remainder
    while index < 80:
        plot_id = f"plot-house-{index:02d}"
        spec = _house_choice(_plot_seed(seed, plot_id, {}))
        span = spec["plot_w"]
        if cursor + span > stop + 1e-6:
            break
        if along_x:
            rect = (cursor, cross0, cursor + span, cross1)
        else:
            rect = (cross0, cursor, cross1, cursor + span)
        plots.append(_plot(
            plot_id, rect, facing, span, abs(cross1 - cross0), "house",
            _house_request(spec), boundary="fence",
        ))
        cursor += span
        index += 1
    return plots, cursor, index


def _close_plot(plot_id, rect, facing) -> Optional[Dict[str, Any]]:
    x0, z0, x1, z1 = rect
    if x1 - x0 < 0.05 or z1 - z0 < 0.05:
        return None
    if facing in ("south", "north"):
        frontage, depth = x1 - x0, z1 - z0
    else:
        frontage, depth = z1 - z0, x1 - x0
    return _plot(plot_id, rect, facing, frontage, depth, "close", None, boundary="all")


def _edge(ident, axis, at, side, start, end, exclude) -> Dict[str, Any]:
    return {
        "id": ident,
        "axis": axis,
        "at": round(float(at), 3),
        "side": int(side),
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "exclude": [[round(float(a), 3), round(float(b), 3)] for a, b in exclude if b - a > 0.05],
    }


def _english_street(width: float, depth: float, seed: int) -> Dict[str, Any]:
    """One lane, tofts facing each other, church closing the north-east and a green the south-east."""
    south_toft = min(60.0, (depth - _LANE) / 2)
    north_toft = min(60.0, depth - _LANE - south_toft)
    lane_z0 = south_toft
    lane_z1 = south_toft + _LANE
    north_z1 = lane_z1 + north_toft
    plots, south_end, index = _pack_houses(
        seed, 0.0, width, 0.0, lane_z0, "north", 1, 14.0,
    )
    north, north_end, index = _pack_houses(
        seed, 0.0, width, lane_z1, north_z1, "south", index, 24.0,
    )
    plots.extend(north)
    if len(plots) < 10:
        raise ValueError("street-village cannot place 10 houses on this site")
    church = _plot(
        "plot-church", (north_end, lane_z1, width, north_z1), "south",
        width - north_end, north_toft, "church", _church(), boundary="wall", gateway=3.2,
    )
    plots.append(church)
    green = (south_end, 0.0, width, lane_z0)
    field = _close_plot("plot-field", (0.0, north_z1, width, depth), "south")
    if field is not None:
        plots.append(field)
    lane = _lane("street-lane", 0.0, lane_z0, width, lane_z1, "lane")
    edges = [
        _edge("lane-south", "x", lane_z0, -1, 0.0, width, [(south_end, width)]),
        _edge("lane-north", "x", lane_z1, 1, 0.0, width, [(north_end, width)]),
    ]
    return _english_plan(
        "Medieval English street village", plots, [lane],
        [_square("square-green", green, "street-lane", "green")], edges,
    )


def _english_market(width: float, depth: float, seed: int) -> Dict[str, Any]:
    """Houses on both sides of the street, a limestone market, a hall and a church."""
    south_toft = min(60.0, (depth - _LANE) / 2)
    north_toft = min(60.0, depth - _LANE - south_toft)
    lane_z0 = south_toft
    lane_z1 = south_toft + _LANE
    north_z1 = lane_z1 + north_toft
    church_w, market_w, hall_min = 24.0, 16.0, 13.0
    reserved = church_w + market_w + hall_min
    south, south_end, index = _pack_houses(
        seed, 0.0, width, 0.0, lane_z0, "north", 1, 0.0,
    )
    north, north_end, index = _pack_houses(
        seed, 0.0, width - reserved, lane_z1, north_z1, "south", index, 0.0,
    )
    plots = [*south, *north]
    if len(plots) < 14:
        raise ValueError("market-town cannot place 14 houses on this site")
    hall_x1 = width - church_w - market_w
    church_x0 = width - church_w
    market = (hall_x1, lane_z1, church_x0, north_z1)
    plots.append(_plot(
        "plot-hall", (north_end, lane_z1, hall_x1, north_z1), "east",
        north_z1 - lane_z1, hall_x1 - north_end, "hall", _house(12.0),
        boundary="fence", along="low",
    ))
    south_gap = _close_plot("plot-close-south", (south_end, 0.0, width, lane_z0), "north")
    if south_gap is not None:
        plots.append(south_gap)
    plots.append(_plot(
        "plot-church", (church_x0, lane_z1, width, north_z1), "west",
        north_toft, church_w, "church", _church(), boundary="wall", gateway=3.2,
    ))
    field = _close_plot("plot-field", (0.0, north_z1, width, depth), "south")
    if field is not None:
        plots.append(field)
    lane = _lane("street-lane", 0.0, lane_z0, width, lane_z1, "lane")
    edges = [
        _edge("lane-south", "x", lane_z0, -1, 0.0, width, []),
        _edge("lane-north", "x", lane_z1, 1, 0.0, width, [(hall_x1, church_x0), (church_x0, width)]),
    ]
    return _english_plan(
        "Medieval English market town", plots, [lane],
        [_square("square-market", market, "street-lane", "stone")], edges,
    )


def _english_green(width: float, depth: float, manor: bool, seed: int) -> Dict[str, Any]:
    """Houses on all four sides of a green, with the church on the north side and a manor to the north-east."""
    green_d = 24.0
    side = 32.0
    lane_n = depth - _LANE
    # South toft + green + north toft fill the space between the lanes. Extra depth
    # lengthens those tofts up to 60 m and leaves the rest as a field beyond the north lane.
    pair = (lane_n - _LANE) - green_d
    if pair < 60.0 - 1e-6:
        raise ValueError("green-village cannot keep a 30 m toft around the green on this site")
    toft = min(60.0, pair / 2.0)
    green_z0 = _LANE + toft
    green_z1 = green_z0 + green_d
    row_z1 = green_z1 + toft
    lane_z0 = row_z1
    lane_z1 = row_z1 + _LANE
    green_x0 = _LANE + side
    green_x1 = width - side
    if green_x1 - green_x0 < 52.0 - 1e-6 or lane_z1 > depth + 1e-6:
        raise ValueError("green-village cannot keep a 30 m toft around the green on this site")
    plots = []
    south, south_end, index = _pack_houses(
        seed, green_x0, green_x1, _LANE, green_z0, "north", 1, 0.0,
    )
    west, west_end, index = _pack_houses(
        seed, green_z0, green_z1, _LANE, green_x0, "east", index, 0.0,
    )
    church_room = 16.0
    north, north_end, index = _pack_houses(
        seed, green_x0, green_x1, green_z1, row_z1, "south", index, church_room,
    )
    east, east_end, index = _pack_houses(
        seed, green_z0, green_z1, green_x1, width, "west", index, 0.0,
    )
    plots.extend(south)
    plots.extend(west)
    plots.extend(north)
    plots.extend(east)
    if len(plots) < 10:
        raise ValueError("green-village cannot place 10 houses around the green on this site")
    plots.append(_plot(
        "plot-church", (north_end, green_z1, green_x1, row_z1), "south",
        green_x1 - north_end, toft, "church", _church(),
        boundary="wall", gateway=3.2,
    ))
    plots.append(_plot(
        "plot-manor", (green_x1, green_z1, width, row_z1), "north",
        side, toft, "manor", _keep() if manor else None,
        boundary="fence",
    ))
    closes = (
        ("plot-close-sw", (_LANE, _LANE, green_x0, green_z0), "north"),
        ("plot-close-nw", (_LANE, green_z1, green_x0, row_z1), "south"),
        ("plot-close-se", (green_x1, _LANE, width, green_z0), "north"),
        ("plot-close-s", (south_end, _LANE, green_x1, green_z0), "north"),
        ("plot-close-w", (_LANE, west_end, green_x0, green_z1), "east"),
        ("plot-close-e", (green_x1, east_end, width, green_z1), "west"),
    )
    for plot_id, rect, facing in closes:
        close = _close_plot(plot_id, rect, facing)
        if close is not None:
            plots.append(close)
    field = _close_plot("plot-field", (_LANE, lane_z1, width, depth), "south")
    if field is not None:
        plots.append(field)
    west_lane = _lane("street-west", 0.0, 0.0, _LANE, depth, "lane")
    south_lane = _lane("street-south", _LANE, 0.0, width, _LANE, "lane")
    north_lane = _lane("street-north", _LANE, lane_z0, width, lane_z1, "lane")
    green = (green_x0, green_z0, green_x1, green_z1)
    edges = [
        _edge("green-south", "x", green_z0, -1, green_x0, green_x1, []),
        _edge("green-north", "x", green_z1, 1, green_x0, green_x1, [(north_end, green_x1)]),
        _edge("green-west", "z", green_x0, -1, green_z0, green_z1, []),
        _edge("green-east", "z", green_x1, 1, green_z0, green_z1, []),
    ]
    return _english_plan(
        "Medieval English green village", plots, [west_lane, south_lane, north_lane],
        [_square("square-green", green, "street-west", "green")], edges,
    )


def _english_plan(typology, plots, streets, squares, edges) -> Dict[str, Any]:
    return {
        "typology": typology,
        "plots": plots,
        "streets": streets,
        "squares": squares,
        "street_records": [_street_record(street) for street in streets],
        "frontage_edges": edges,
        "insulae": [],
    }


def _plot(plot_id, rect, facing, frontage, depth, kind, request, cross=None, insula=None,
         boundary=None, along=None, podium=0.0, gateway=0.0, insula_id=None) -> Dict[str, Any]:
    return {
        "id": plot_id,
        "rect": tuple(round(float(v), 4) for v in rect),
        "facing": facing,
        "frontage": float(frontage),
        "depth": float(depth),
        "kind": kind,
        "request": request,
        "cross": cross,
        "insula": insula,
        "boundary": boundary,
        "along": along,
        "podium": float(podium or 0.0),
        "gateway": float(gateway or 0.0),
        "insula_id": insula_id,
    }


def _house(frontage: float) -> BuildRequest:
    width = 8 if frontage < 11 else 9
    return BuildRequest(
        product="timber-house", seed=0, width=width, depth=8, height=7,
        levels=1, bays=2, variant="gable-front", detail="concept", age=0,
        parameters={"finish": "chalk-slate", "porch": True, "chimney": True},
    )


def _church() -> BuildRequest:
    # 12 x 15 x 20 is the smallest integer triple that clears the cathedral
    # ratio floors. 19.2 / 12 and 14.4 / 12 fall just short in floating point.
    return BuildRequest(
        product="gothic-cathedral", seed=0, width=12, height=15, depth=20,
        variant="early-english", bays=4, detail="concept", age=0,
        parameters={"choir_bays": 2, "apse_sides": 5},
    )


def _keep() -> BuildRequest:
    return BuildRequest(
        product="norman-keep", seed=0, width=12, height=16, depth=14,
        variant="square", bays=2, levels=3, detail="concept", age=0,
        parameters={"variation": 0.0},
    )


def _roman(variant: str, width: float, depth: float) -> Dict[str, Any]:
    count = _roman_count(variant, width, depth)
    x_streets, x_blocks, insula_x = _bands(width, count, _MINOR, _MAJOR)
    z_streets, z_blocks, insula_z = _bands(depth, count, _MINOR, _MAJOR)
    _require_insula(variant, width, depth, insula_x, insula_z, count)
    major = count // 2
    streets = []
    for index, (z0, z1, street_class) in enumerate(z_streets):
        ident = "street-decumanus" if index == major else f"street-side-z-{index}"
        streets.append(_roman_run(ident, "z", z0, z1, street_class, width, x_streets, z_streets))
    for index, (x0, x1, street_class) in enumerate(x_streets):
        ident = "street-cardo" if index == major else f"street-side-x-{index}"
        streets.append(_roman_run(ident, "x", x0, x1, street_class, depth, z_streets, x_streets))

    plots = []
    squares = []
    insulae = []
    insula = min(insula_x, insula_z)
    for col, (x0, x1) in enumerate(x_blocks):
        for row, (z0, z1) in enumerate(z_blocks):
            rect = (x0, z0, x1, z1)
            role = _insula_role(variant, col, row, count, major)
            use = "forum" if role == "forum" else "residential"
            insula_id = f"insula-{col}-{row}"
            block = _insula_plots(role, rect, col, row, major, insula, insula_id, count)
            plots.extend(block)
            insulae.append({
                "id": insula_id,
                "col": col,
                "row": row,
                "rect": [round(v, 3) for v in rect],
                "use": use,
                "buildings": sum(1 for plot in block if plot.get("request") is not None),
            })
            if role == "forum":
                squares.append(_forum_square(rect))
    cardo = x_streets[major]
    decumanus = z_streets[major]
    horizontal = [(item[0], item[1]) for item in z_streets]
    vertical = [(item[0], item[1]) for item in x_streets]
    edges = [
        _edge("cardo-west", "z", cardo[0], -1, 0.0, depth, horizontal),
        _edge("cardo-east", "z", cardo[1], 1, 0.0, depth, horizontal),
        _edge("decumanus-south", "x", decumanus[0], -1, 0.0, width, vertical),
        _edge("decumanus-north", "x", decumanus[1], 1, 0.0, width, vertical),
    ]
    return {
        "typology": f"Roman {variant.replace('-', ' ')}",
        "plots": plots,
        "streets": streets,
        "squares": squares,
        "street_records": [_street_record(street) for street in streets],
        "frontage_edges": edges,
        "insulae": insulae,
    }


def _roman_count(variant: str, width: float, depth: float) -> int:
    if variant != "colonia-grid":
        return 2
    for count in (4, 3):
        if _insula_ok(width, count) and _insula_ok(depth, count):
            return count
    raise ValueError("colonia-grid needs a site of at least 128 m by 128 m so each insula is at least 35 m")


def _insula_ok(span: float, count: int) -> bool:
    insula = (span - (_MINOR * count + _MAJOR)) / count
    return 35.0 - 1e-6 <= insula <= 75.0 + 1e-6


def _bands(span: float, count: int, minor: float, major: float):
    insula = (span - (minor * count + major)) / count
    major_index = count // 2
    cursor = 0.0
    streets = []
    blocks = []
    for index in range(count + 1):
        street_w = major if index == major_index else minor
        kind = "major" if index == major_index else "side"
        streets.append((cursor, cursor + street_w, kind))
        cursor += street_w
        if index < count:
            blocks.append((cursor, cursor + insula))
            cursor += insula
    return streets, blocks, insula


def _require_insula(variant, width, depth, insula_x, insula_z, count):
    if 35.0 - 1e-6 <= insula_x <= 75.0 + 1e-6 and 35.0 - 1e-6 <= insula_z <= 75.0 + 1e-6:
        return
    if variant == "colonia-grid":
        minimum = 166 if count == 4 else 128
        raise ValueError(
            f"colonia-grid on this site gives insulae of {insula_x:.1f} by {insula_z:.1f} m; "
            f"use at least {minimum} m on a side so each insula stays between 35 and 75 m"
        )
    raise ValueError(
        f"{variant} needs a site between 87 and 167 m on a side so a 2 by 2 insula stays between 35 and 75 m "
        f"(this site gives {insula_x:.1f} by {insula_z:.1f} m)"
    )


def _insula_role(variant: str, col: int, row: int, count: int, major: int) -> str:
    if variant != "residential-quarter" and (col, row) == (major, major):
        return "forum"
    if variant != "residential-quarter" and (col, row) == (max(0, major - 1), max(0, major - 1)):
        return "baths"
    return "residential"


def _insula_plots(role, rect, col, row, major, insula, insula_id, count) -> List[Dict[str, Any]]:
    if role == "forum":
        return _forum_plots(rect, insula, insula_id)
    return _residential_plots(role, rect, col, row, major, insula, insula_id, count)


def _major_edges(col: int, row: int, major: int) -> List[str]:
    edges = []
    if row == major:
        edges.append("south")
    if row + 1 == major:
        edges.append("north")
    if col == major:
        edges.append("west")
    if col + 1 == major:
        edges.append("east")
    return edges


def _cut_strip(rect, facing, depth):
    x0, z0, x1, z1 = rect
    if facing == "south":
        return (x0, z0, x1, min(z1, z0 + depth)), (x0, min(z1, z0 + depth), x1, z1)
    if facing == "north":
        return (x0, max(z0, z1 - depth), x1, z1), (x0, z0, x1, max(z0, z1 - depth))
    if facing == "west":
        return (x0, z0, min(x1, x0 + depth), z1), (min(x1, x0 + depth), z0, x1, z1)
    return (max(x0, x1 - depth), z0, x1, z1), (x0, z0, max(x0, x1 - depth), z1)


def _residential_plots(role, rect, col, row, major, insula, insula_id, count) -> List[Dict[str, Any]]:
    """Stoae on the cardo and decumanus, courtyard houses on the remaining street edges."""
    edges = _major_edges(col, row, major)
    plots = []
    inner = rect
    for facing in edges:
        strip, inner = _cut_strip(inner, facing, _STOA_D)
        plots.append(_building_plot(
            f"plot-stoa-{col}-{row}-{facing}", strip, facing, "stoa",
            _stoa_request(strip, facing), insula, insula_id,
        ))
    taken = list(edges)
    if col == 0 and row == count - 1 and "north" not in edges:
        # No classical form is a row of shops. The aqueduct is the narrow street-front block.
        strip, inner = _cut_strip(inner, "north", 3.6)
        plots.append(_building_plot(
            f"plot-aqueduct-{col}-{row}", strip, "north", "aqueduct", _aqueduct(), insula, insula_id,
        ))
        taken.append("north")
    plots.extend(_inner_houses(role, inner, taken, col, row, insula, insula_id))
    if sum(1 for plot in plots if plot.get("request") is not None) < 3:
        raise ValueError(f"insula {col},{row} could not hold three buildings")
    return plots


def _inner_houses(role, rect, cut, col, row, insula, insula_id) -> List[Dict[str, Any]]:
    x0, z0, x1, z1 = rect
    if x1 - x0 < 8 or z1 - z0 < 8:
        return []
    remaining = [facing for facing in ("south", "west", "north", "east") if facing not in cut]
    if not cut:
        return _corner_houses(rect, col, row, insula, insula_id, role == "baths")
    plots = []
    if len(remaining) >= 2:
        first, second = remaining[0], remaining[1]
        band = min(16.5, (z1 - z0) * 0.55 if first in ("south", "north") else (x1 - x0) * 0.55)
        strip, rest = _cut_strip(rect, first, max(14.5, band))
        kind = "baths" if role == "baths" else "villa"
        plots.append(_sized_house(
            f"plot-{kind}-{col}-{row}", strip, first, kind, insula, insula_id, baths=role == "baths",
        ))
        if rest[2] - rest[0] > 10 and rest[3] - rest[1] > 10:
            plots.append(_sized_house(
                f"plot-villa-{col}-{row}-b", rest, second, "villa", insula, insula_id,
            ))
        return [plot for plot in plots if plot is not None]
    facing = remaining[0]
    mid = (x0 + x1) / 2 if facing in ("south", "north") else (z0 + z1) / 2
    if facing in ("south", "north"):
        left, right = (x0, z0, mid, z1), (mid, z0, x1, z1)
    else:
        left, right = (x0, z0, x1, mid), (x0, mid, x1, z1)
    plots.append(_sized_house(f"plot-villa-{col}-{row}-a", left, facing, "villa", insula, insula_id))
    plots.append(_sized_house(
        f"plot-villa-{col}-{row}-b", right, facing, "baths" if role == "baths" else "villa",
        insula, insula_id, baths=role == "baths",
    ))
    return [plot for plot in plots if plot is not None]


def _corner_houses(rect, col, row, insula, insula_id, baths) -> List[Dict[str, Any]]:
    x0, z0, x1, z1 = rect
    mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
    corners = (
        ("sw", (x0, z0, mx, mz), "south", baths),
        ("se", (mx, z0, x1, mz), "south", False),
        ("nw", (x0, mz, mx, z1), "north", False),
        ("ne", (mx, mz, x1, z1), "north", False),
    )
    plots = []
    for name, plot_rect, facing, is_baths in corners:
        kind = "baths" if is_baths else "villa"
        plots.append(_sized_house(
            f"plot-{kind}-{col}-{row}-{name}", plot_rect, facing, kind, insula, insula_id, baths=is_baths,
        ))
    return [plot for plot in plots if plot is not None]


def _sized_house(plot_id, rect, facing, kind, insula, insula_id, baths=False):
    x0, z0, x1, z1 = rect
    room_w = (x1 - x0) if facing in ("south", "north") else (z1 - z0)
    room_d = (z1 - z0) if facing in ("south", "north") else (x1 - x0)
    if baths:
        width, depth = 16.0, 13.0
        if room_w < width + 0.5 or room_d < depth + _SETBACK + 0.4:
            baths = False
            kind = "villa"
    if not baths:
        width = min(16.0, max(8.0, room_w - 0.9))
        depth = min(14.0, max(8.0, room_d - _SETBACK - 0.4))
        if width > room_w - 0.3 or depth > room_d - _SETBACK - 0.05:
            return None
    request = _baths() if baths else _villa_request(width, depth)
    return _building_plot(plot_id, rect, facing, kind, request, insula, insula_id)


def _face_major(col: int, row: int, major: int) -> str:
    if col + 1 == major:
        return "east"
    if col == major:
        return "west"
    if row + 1 == major:
        return "north"
    if row == major:
        return "south"
    return "south"


def _forum_plots(rect, insula, insula_id) -> List[Dict[str, Any]]:
    """Open court. A stoa and the basilica face each other on the long sides; the temple is raised at the north end."""
    x0, z0, x1, z1 = rect
    span_x, span_z = x1 - x0, z1 - z0
    stoa_d, basilica_d, end_d = _STOA_D, 12.0, 8.0
    if span_x < stoa_d + basilica_d + 16.6 or span_z < stoa_d + end_d + 10:
        raise ValueError("forum insula is too small for stoae, a basilica and an open court")
    south, rest = _cut_strip(rect, "south", stoa_d)
    east, rest = _cut_strip(rest, "east", stoa_d)
    west, court = _cut_strip(rest, "west", basilica_d)
    cx0, cz0, cx1, cz1 = court
    arch_w = 8.4
    arch = (cx0, cz1 - end_d, cx0 + arch_w, cz1)
    temple = (cx0 + arch_w, cz1 - end_d, cx1, cz1)
    plots = [
        _building_plot("plot-stoa-south", south, "north", "stoa", _stoa_request(south, "north"), insula, insula_id),
        _building_plot("plot-stoa-east", east, "west", "stoa", _stoa_request(east, "west"), insula, insula_id),
        _building_plot("plot-basilica", west, "east", "basilica", _basilica_request(west), insula, insula_id),
        _building_plot("plot-arch", arch, "south", "arch", _arch(), insula, insula_id),
        _building_plot("plot-temple", temple, "south", "temple", _portico(), insula, insula_id, podium=0.55),
    ]
    return plots


def _forum_square(rect) -> Dict[str, Any]:
    x0, z0, x1, z1 = rect
    stoa_d, basilica_d, end_d = _STOA_D, 12.0, 8.0
    square = (
        x0 + basilica_d, z0 + stoa_d,
        x1 - stoa_d, z1 - end_d,
    )
    return _square("square-forum", square, "street-decumanus", "stone")


def _building_plot(plot_id, rect, facing, kind, request, insula, insula_id=None, podium=0.0) -> Dict[str, Any]:
    x0, z0, x1, z1 = rect
    frontage = (x1 - x0) if facing in ("south", "north") else (z1 - z0)
    depth = (z1 - z0) if facing in ("south", "north") else (x1 - x0)
    return _plot(
        plot_id, rect, facing, frontage, depth, kind, request,
        insula=insula, insula_id=insula_id, podium=podium,
    )


def _stoa_request(rect, facing) -> BuildRequest:
    x0, z0, x1, z1 = rect
    length = (x1 - x0) if facing in ("south", "north") else (z1 - z0)
    depth = (z1 - z0) if facing in ("south", "north") else (x1 - x0)
    width = max(8.0, round(length - 1.0, 2))
    body = max(5.0, round(min(6.2, depth - _SETBACK - 0.05), 2))
    bays = max(2, min(12, int(width // 5.5)))
    return BuildRequest(
        product="classical-stoa", seed=0, width=width, height=6, depth=body,
        bays=bays, variant="doric", detail="concept", age=0,
    )


def _basilica_request(rect) -> BuildRequest:
    x0, z0, x1, z1 = rect
    # The basilica faces east, so its length runs north-south.
    length = max(12.0, round(min(z1 - z0 - 1.0, 28.0), 2))
    body = max(8.0, round(min(11.0, (x1 - x0) - _SETBACK - 0.4), 2))
    bays = max(2, min(12, int(length // 5.0)))
    return BuildRequest(
        product="classical-basilica", seed=0, width=length, height=8, depth=body,
        bays=bays, variant="doric", detail="concept", age=0,
    )


def _villa_request(width: float, depth: float) -> BuildRequest:
    width = round(max(8.0, min(18.0, width)), 2)
    depth = round(max(8.0, min(16.0, depth)), 2)
    return BuildRequest(
        product="classical-villa", seed=0, width=width, height=6, depth=depth,
        bays=2, variant="doric", detail="concept", age=0,
    )


def _villa(col: int, row: int) -> BuildRequest:
    wide = (col + row) % 2 == 0
    return BuildRequest(
        product="classical-villa", seed=0,
        width=14 if wide else 12, height=6, depth=12 if wide else 10,
        bays=2, variant="doric" if wide else "tuscan", detail="concept", age=0,
    )


def _basilica() -> BuildRequest:
    return BuildRequest(
        product="classical-basilica", seed=0, width=12, height=7, depth=14,
        bays=3, variant="doric", detail="concept", age=0,
    )


def _portico() -> BuildRequest:
    return BuildRequest(
        product="classical-portico", seed=0, width=8, height=6, depth=5,
        bays=3, variant="tuscan", detail="concept", age=0,
    )


def _baths() -> BuildRequest:
    return BuildRequest(
        product="classical-baths", seed=0, width=16, height=8, depth=13,
        bays=3, variant="auto", detail="concept", age=0,
    )


def _arch() -> BuildRequest:
    return BuildRequest(
        product="classical-triumphal-arch", seed=0, width=8, height=8, depth=3.2,
        bays=2, variant="auto", detail="concept", age=0,
    )


def _aqueduct() -> BuildRequest:
    return BuildRequest(
        product="classical-aqueduct", seed=0, width=12, height=6, depth=2.4,
        bays=3, levels=1, variant="auto", detail="concept", age=0,
    )


def _lane(ident, x0, z0, x1, z1, street_class) -> Dict[str, Any]:
    return {
        "id": ident,
        "rects": [(x0, z0, x1, z1)],
        "class": street_class,
        "substance": "earth-fill",
        "kerbs": [],
        "width_m": _LANE,
    }


def _street_record(street: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": street["id"],
        "class": street["class"],
        "width_m": street["width_m"],
        "substance": street["substance"],
    }


def _square(ident, rect, parent, surface) -> Dict[str, Any]:
    return {"id": ident, "rect": tuple(rect), "parent": parent, "surface": surface}


def _roman_run(ident, axis, a0, a1, street_class, span, crossing, own):
    """One street. The slab is the whole band; kerbs and pavements sit on it beside the insulae."""
    rects = []
    kerbs = []
    pavements = []
    major = street_class == "major"
    if axis == "z":
        rects.append((0.0, a0, span, a1))
        if major:
            for x0, x1 in _block_intervals(crossing):
                kerbs.append((x0, a0, x1, a0 + _KERB_W))
                kerbs.append((x0, a1 - _KERB_W, x1, a1))
                pavements.append((x0, a0 + _KERB_W, x1, a0 + _KERB_W + _PAVE_W))
                pavements.append((x0, a1 - _KERB_W - _PAVE_W, x1, a1 - _KERB_W))
    else:
        for z0, z1 in _vertical_intervals(crossing):
            rects.append((a0, z0, a1, z1))
            if major:
                kerbs.append((a0, z0, a0 + _KERB_W, z1))
                kerbs.append((a1 - _KERB_W, z0, a1, z1))
                pavements.append((a0 + _KERB_W, z0, a0 + _KERB_W + _PAVE_W, z1))
                pavements.append((a1 - _KERB_W - _PAVE_W, z0, a1 - _KERB_W, z1))
    return {
        "id": ident,
        "rects": rects,
        "class": "cardo" if axis == "x" and major else "decumanus" if major else "side",
        "substance": "limestone",
        "kerbs": kerbs,
        "pavements": pavements,
        "width_m": _MAJOR if major else _MINOR,
        "stepping": major,
    }


def _block_intervals(streets) -> List[Tuple[float, float]]:
    """Ranges between street bands: the insulae, where a kerb can run without closing a junction."""
    return [(streets[index][1], streets[index + 1][0]) for index in range(len(streets) - 1)]


def _vertical_intervals(horizontal) -> List[Tuple[float, float]]:
    """Vertical bands fill the gap between horizontal bands, so the two together tile the streets."""
    return [(horizontal[index][1], horizontal[index + 1][0]) for index in range(len(horizontal) - 1)]


# ---------------------------------------------------------------- placement


def _vary_house(request: BuildRequest, plot_seed: int, detail: str, frontage: float) -> BuildRequest:
    """The plot seed chooses the frame. Layout used the same choice, so the plot already fits it."""
    if request.product != "timber-house":
        return _retarget(request, plot_seed, detail)
    spec = _house_choice(plot_seed)
    # A plot override may have replaced the frame. Keep that request's size and variant,
    # and only fill the seed and the detail. Layout does not read overrides.
    if request.variant != spec["variant"] or int(request.levels or 1) != spec["levels"]:
        varied = _retarget(request, plot_seed, detail)
        return varied
    return BuildRequest(
        product="timber-house",
        seed=plot_seed,
        width=spec["width"],
        depth=spec["depth"],
        height=spec["height"],
        levels=spec["levels"],
        bays=2,
        variant=spec["variant"],
        detail=detail,
        age=0,
        parameters={
            "finish": spec["finish"],
            "porch": spec["porch"],
            "chimney": spec["chimney"],
        },
    )


def _authored(request: BuildRequest) -> BuildDocument:
    payload = request.to_dict()
    payload["age"] = 0
    if payload["product"].startswith("classical-") or payload["product"] in {"gothic-cathedral", "norman-keep"}:
        if payload["variant"] != "random":
            payload["seed"] = 0
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cached = _CACHE.get(key)
    if cached is None:
        from .service import author_document

        cached = author_document(BuildRequest.from_dict(payload))
        _CACHE[key] = cached
    return cached


def _anchor(local_lo, local_hi, yaw, rect, facing, cross, along=None, podium=0.0) -> Tuple[float, float, float]:
    turned_lo, turned_hi = _turn_aabb(local_lo, local_hi, yaw)
    x0, z0, x1, z1 = rect
    mid_x = (turned_lo[0] + turned_hi[0]) / 2
    mid_z = (turned_lo[2] + turned_hi[2]) / 2
    cx = (x0 + x1) / 2
    cz = (z0 + z1) / 2
    if facing == "south":
        ax, az = cx - mid_x, z0 + _SETBACK - turned_lo[2]
    elif facing == "north":
        ax, az = cx - mid_x, z1 - _SETBACK - turned_hi[2]
    elif facing == "west":
        ax, az = x0 + _SETBACK - turned_lo[0], cz - mid_z
    else:
        ax, az = x1 - _SETBACK - turned_hi[0], cz - mid_z
    if cross is not None:
        if facing in ("south", "north"):
            ax = float(cross) - mid_x
        else:
            az = float(cross) - mid_z
    if along == "low":
        if facing in ("south", "north"):
            ax = x0 + 0.5 - turned_lo[0]
        else:
            az = z0 + 0.5 - turned_lo[2]
    elif along == "high":
        if facing in ("south", "north"):
            ax = x1 - 0.5 - turned_hi[0]
        else:
            az = z1 - 0.5 - turned_hi[2]
    return (ax, float(podium or 0.0), az)


def _turn_aabb(lo, hi, yaw):
    xs, zs = [], []
    for x in (lo[0], hi[0]):
        for z in (lo[2], hi[2]):
            c, s = _CS[_steps(yaw)]
            xs.append(x * c - z * s)
            zs.append(x * s + z * c)
    return (min(xs), lo[1], min(zs)), (max(xs), hi[1], max(zs))


def _place_members(members, plot_id, product, yaw, anchor, yard_id) -> List[Placement]:
    mapping = {placement.id: f"{plot_id}.{placement.id}" for placement in members}
    steps = _steps(yaw)
    placed = []
    for placement in members:
        parameters = _prefix_hosts(_map_refs(dict(placement.parameters or {}), mapping), plot_id)
        position, dimensions, parameters = _rotate_member(
            placement.position, placement.dimensions, parameters, placement.block_type, steps, anchor,
        )
        parameters["plot_id"] = plot_id
        parameters["building_product"] = product
        parameters["source_id"] = placement.id
        parameters["source_parent"] = placement.parent_id
        parent = mapping[placement.parent_id] if placement.parent_id else yard_id
        placed.append(Placement(
            id=mapping[placement.id],
            kind=placement.kind,
            block_type=placement.block_type,
            position=position,
            dimensions=dimensions,
            material=placement.material,
            role=placement.role,
            parameters=parameters,
            parent_id=parent,
            substance=placement.substance,
        ))
    return placed


def _map_refs(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_map_refs(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _map_refs(item, mapping) for key, item in value.items()}
    return value


def _prefix_hosts(value, plot_id):
    """Ashlar host ids are reused by every call of an architect. A plot prefix keeps each bond its own."""
    prefix = plot_id + "."
    if isinstance(value, dict):
        prefixed = {}
        for key, item in value.items():
            if key == "host_id" and isinstance(item, str) and not item.startswith(prefix):
                prefixed[key] = prefix + item
            else:
                prefixed[key] = _prefix_hosts(item, plot_id)
        return prefixed
    if isinstance(value, list):
        return [_prefix_hosts(item, plot_id) for item in value]
    return value


def _plot_parameters(plot: Dict[str, Any]) -> Dict[str, Any]:
    sent = plot.get("_sent")
    anchor = plot.get("_anchor")
    parameters = {
        "plot_id": plot["id"],
        "building_product": None if not sent else sent.get("product"),
        "frontage_m": round(float(plot["frontage"]), 3),
        "depth_m": round(float(plot["depth"]), 3),
        "facing": plot["facing"],
        "yaw": _YAW[plot["facing"]],
        "anchor": None if anchor is None else [float(v) for v in anchor],
        "building_request": sent,
        "kind_name": plot["kind"],
    }
    if plot.get("insula") is not None:
        parameters["insula_m"] = round(float(plot["insula"]), 3)
    if plot.get("insula_id"):
        parameters["insula_id"] = plot["insula_id"]
    return parameters


def _grounds(plot: Dict[str, Any], bounds, members: Sequence[Placement]) -> List[Placement]:
    """Yard and garden fill the plot up to the building's plan. The largest piece is the plot root.

    The hole is the union of the placed members, not their bounding box. A
    cruciform church and a porch leave ground in the corners of that box.
    A podium keeps the box, because the slab under the temple is that box.
    """
    if plot.get("podium") and bounds is not None:
        lo, hi = bounds
        holes = [(lo[0], lo[2], hi[0], hi[2])]
    else:
        holes = _member_plan(plot["rect"], members)
    pieces = _subtract_many(plot["rect"], holes)
    if not pieces:
        x0, z0, x1, z1 = plot["rect"]
        pieces = [(x0, z0, x1, z0 + min(_YARD_MIN, z1 - z0))]
    pieces.sort(key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]), reverse=True)
    parameters = _plot_parameters(plot)
    members = []
    for index, rect in enumerate(pieces):
        x0, z0, x1, z1 = rect
        if x1 - x0 < 0.02 or z1 - z0 < 0.02:
            continue
        role = "plot" if index == 0 else "garden"
        color = _YARD if role == "plot" else _GRASS
        members.append(_slab(
            plot["id"] if role == "plot" else f"{plot['id']}.garden.{index:02d}",
            role, "site", (x0 + x1) / 2, _YARD_T / 2, (z0 + z1) / 2,
            x1 - x0, _YARD_T, z1 - z0, color, "earth-fill",
            None if role == "plot" else plot["id"],
            dict(parameters),
        ))
    if not members:
        raise ValueError(f"{plot['id']} has no ground")
    if plot.get("podium") and bounds is not None:
        lo, hi = bounds
        hx0, hz0, hx1, hz1 = _clip(plot["rect"], (lo[0], lo[2], hi[0], hi[2]))
        height = float(plot["podium"])
        if hx1 - hx0 > 0.2 and hz1 - hz0 > 0.2 and height > 0:
            members.append(_slab(
                f"{plot['id']}.podium", "podium", "site",
                (hx0 + hx1) / 2, height / 2, (hz0 + hz1) / 2,
                hx1 - hx0, height, hz1 - hz0, _PODIUM_COLOR, "limestone", plot["id"],
                {"plot_id": plot["id"], "building_product": parameters.get("building_product")},
            ))
    return members


def _member_plan(rect, members) -> List[Tuple[float, float, float, float]]:
    """Axis-aligned plan of each placed member, clipped to the plot."""
    holes = []
    for member in members or ():
        cx, _, cz = member.position
        width, _, depth = member.dimensions
        hx0, hz0, hx1, hz1 = _clip(
            rect, (cx - width / 2.0, cz - depth / 2.0, cx + width / 2.0, cz + depth / 2.0),
        )
        if hx1 - hx0 > 1e-4 and hz1 - hz0 > 1e-4:
            holes.append((hx0, hz0, hx1, hz1))
    return holes


def _subtract_many(rect, holes) -> List[Tuple[float, float, float, float]]:
    """Complement of axis-aligned holes. The pieces meet the holes and the plot edge."""
    x0, z0, x1, z1 = rect
    clipped = []
    xs = {x0, x1}
    for hx0, hz0, hx1, hz1 in holes:
        if hx1 - hx0 <= 1e-6 or hz1 - hz0 <= 1e-6:
            continue
        clipped.append((hx0, hz0, hx1, hz1))
        xs.add(hx0)
        xs.add(hx1)
    if not clipped:
        return [(x0, z0, x1, z1)]
    ordered = sorted(xs)
    by_start = sorted(clipped, key=lambda item: (item[0], item[2]))
    active: List[Tuple[float, float, float, float]] = []
    cursor = 0
    raw = []
    for left, right in zip(ordered, ordered[1:]):
        while cursor < len(by_start) and by_start[cursor][0] <= left + 1e-9:
            active.append(by_start[cursor])
            cursor += 1
        if right - left <= 1e-6:
            continue
        active = [item for item in active if item[2] > left + 1e-9]
        z = z0
        for a, b in _merge_intervals((item[1], item[3]) for item in active):
            if a - z > 0.02:
                raw.append((left, z, right, a))
            if b > z:
                z = b
        if z1 - z > 0.02:
            raw.append((left, z, right, z1))
    return _merge_slabs(raw)


def _merge_intervals(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1] + 1e-6:
            if b > merged[-1][1]:
                merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def _merge_slabs(raw) -> List[Tuple[float, float, float, float]]:
    """Join slabs that share a z range and meet along x."""
    groups: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
    for left, z0, right, z1 in raw:
        groups.setdefault((round(z0, 5), round(z1, 5)), []).append((left, right))
    pieces = []
    for (z0, z1), spans in groups.items():
        spans.sort()
        sx0, sx1 = spans[0]
        for a, b in spans[1:]:
            if a <= sx1 + 1e-4:
                if b > sx1:
                    sx1 = b
            else:
                pieces.append((sx0, z0, sx1, z1))
                sx0, sx1 = a, b
        pieces.append((sx0, z0, sx1, z1))
    return pieces


def _clip(rect, hole) -> Tuple[float, float, float, float]:
    return (
        max(rect[0], hole[0]), max(rect[1], hole[1]),
        min(rect[2], hole[2]), min(rect[3], hole[3]),
    )


def _require_inside(plot_id, bounds, rect):
    lo, hi = bounds
    x0, z0, x1, z1 = rect
    if lo[0] < x0 - 0.05 or hi[0] > x1 + 0.05 or lo[2] < z0 - 0.05 or hi[2] > z1 + 0.05:
        raise ValueError(
            f"{plot_id} building extents {(lo[0], lo[2], hi[0], hi[2])} do not fit plot {(x0, z0, x1, z1)}"
        )


def _aabb(placements):
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for placement in placements:
        center = placement.position
        half = [float(v) / 2 for v in placement.dimensions]
        for axis in range(3):
            lo[axis] = min(lo[axis], float(center[axis]) - half[axis])
            hi[axis] = max(hi[axis], float(center[axis]) + half[axis])
    return tuple(lo), tuple(hi)


def _street_members(street: Dict[str, Any]) -> List[Placement]:
    substance = street["substance"]
    color = _EARTH if substance == "earth-fill" else _PAVING
    thickness = _EARTH_T if substance == "earth-fill" else _PAVING_T
    members = []
    root = None
    for index, rect in enumerate(street["rects"]):
        ident = street["id"] if index == 0 else f"{street['id']}.span.{index:02d}"
        x0, z0, x1, z1 = rect
        member = _slab(
            ident, "street", "circulation", (x0 + x1) / 2, thickness / 2, (z0 + z1) / 2,
            max(x1 - x0, 0.05), thickness, max(z1 - z0, 0.05), color, substance, root,
            _ground_parameters(street, None),
        )
        if root is None:
            root = ident
        members.append(member)
    for index, rect in enumerate(street.get("kerbs") or []):
        x0, z0, x1, z1 = rect
        if x1 - x0 < 0.05 or z1 - z0 < 0.05:
            continue
        members.append(_slab(
            f"{street['id']}.kerb.{index:02d}", "kerb", "circulation",
            (x0 + x1) / 2, thickness + _KERB_H / 2, (z0 + z1) / 2,
            x1 - x0, _KERB_H, z1 - z0, _KERB, "limestone", root,
            _ground_parameters(street, None),
        ))
    for index, rect in enumerate(street.get("pavements") or []):
        x0, z0, x1, z1 = rect
        if x1 - x0 < 0.05 or z1 - z0 < 0.05:
            continue
        members.append(_slab(
            f"{street['id']}.pavement.{index:02d}", "pavement", "circulation",
            (x0 + x1) / 2, thickness + _PAVE_H / 2, (z0 + z1) / 2,
            x1 - x0, _PAVE_H, z1 - z0, _PAVING, "limestone", root,
            _ground_parameters(street, None),
        ))
    if street.get("stepping"):
        members.extend(_stepping_stones(street, root, thickness))
    return members


def _stepping_stones(street, root, paving_top) -> List[Placement]:
    stones = []
    index = 0
    for rect in street["rects"]:
        x0, z0, x1, z1 = rect
        length_x, length_z = x1 - x0, z1 - z0
        if length_x >= length_z:
            count = max(1, int(length_x // 9))
            for step in range(count):
                x = x0 + length_x * (step + 0.5) / count
                z = (z0 + z1) / 2
                stones.append(_stone(street, root, index, x, z, paving_top))
                index += 1
        else:
            count = max(1, int(length_z // 9))
            for step in range(count):
                z = z0 + length_z * (step + 0.5) / count
                x = (x0 + x1) / 2
                stones.append(_stone(street, root, index, x, z, paving_top))
                index += 1
    return stones


def _stone(street, root, index, x, z, paving_top):
    return _slab(
        f"{street['id']}.step.{index:02d}", "stepping_stone", "circulation",
        x, paving_top + 0.11, z, 0.7, 0.22, 0.7, _STEP, "limestone", root,
        _ground_parameters(street, None),
    )


def _square_member(square: Dict[str, Any]) -> Placement:
    x0, z0, x1, z1 = square["rect"]
    if x1 - x0 < 1 or z1 - z0 < 1:
        raise ValueError(f"{square['id']} collapsed to {(x0, z0, x1, z1)}")
    stone = square["surface"] == "stone"
    thickness = _PAVING_T if stone else _EARTH_T
    color = _PAVING if stone else _GRASS if square["surface"] == "green" else _EARTH
    return _slab(
        square["id"], "square", "circulation", (x0 + x1) / 2, thickness / 2, (z0 + z1) / 2,
        x1 - x0, thickness, z1 - z0, color,
        "limestone" if stone else "earth-fill",
        square["parent"],
        {"plot_id": None, "building_product": None, "street_id": square["parent"], "surface_kind": square["surface"]},
    )


def _ground_parameters(street, product):
    return {
        "plot_id": None,
        "building_product": product,
        "street_id": street["id"],
        "street_class": street["class"],
        "width_m": street["width_m"],
    }


def _slab(ident, role, kind, x, y, z, w, h, d, color, substance, parent, parameters) -> Placement:
    def r(value):
        number = round(float(value), 6)
        return 0.0 if number == 0 else number
    return Placement(
        id=ident,
        kind=kind,
        block_type="CUBE",
        position=(r(x), r(y), r(z)),
        dimensions=(r(w), r(h), r(d)),
        material=color,
        role=role,
        parameters=parameters,
        parent_id=parent,
        substance=substance,
    )


def _round_anchor(anchor) -> Tuple[float, float, float]:
    """The same rounded anchor is stored and used, so turning back is exact."""
    rounded = []
    for value in anchor:
        number = round(float(value), 6)
        rounded.append(0.0 if number == 0.0 else number)
    return (rounded[0], rounded[1], rounded[2])


def _realize_plot(plot: Dict[str, Any], request: Optional[BuildRequest]):
    """Place one authored building and fill the rest of the plot with ground."""
    document = None if request is None else _authored(request)
    members = [] if document is None else list(document.placements)
    bounds = None
    anchor = None
    placed = []
    if members:
        local_lo, local_hi = _aabb(members)
        anchor = _round_anchor(_anchor(
            local_lo, local_hi, _YAW[plot["facing"]], plot["rect"], plot["facing"],
            plot.get("cross"), plot.get("along"), plot.get("podium") or 0.0,
        ))
        placed = _place_members(members, plot["id"], request.product, _YAW[plot["facing"]], anchor, plot["id"])
        bounds = _aabb(placed)
    plot["_anchor"] = anchor
    plot["_sent"] = None if request is None else request.to_dict()
    ground = _grounds(plot, bounds, placed)
    return ground[0].id, [*ground, *placed], bounds


def _boundary_members(plots: Sequence[Dict[str, Any]]) -> List[Placement]:
    """Fences and the churchyard wall. A shared edge is drawn once, on the plot with the lower id."""
    owned = []
    keys = {}
    for plot in plots:
        if plot.get("boundary") not in {"fence", "wall", "all"}:
            continue
        for name, edge in _plot_edges(plot).items():
            key = _edge_key(*edge)
            keys.setdefault(key, []).append(plot["id"])
    members = []
    seen_posts = set()
    for plot in plots:
        mode = plot.get("boundary")
        if mode not in {"fence", "wall", "all"}:
            continue
        skip = None if mode in {"wall", "all"} else plot["facing"]
        for name, edge in _plot_edges(plot).items():
            if name == skip:
                continue
            key = _edge_key(*edge)
            owners = keys.get(key) or [plot["id"]]
            if min(owners) != plot["id"]:
                continue
            if mode == "wall":
                members.extend(_wall_run(plot, name, edge))
            else:
                members.extend(_fence_run(plot, name, edge, seen_posts))
    return members


def _plot_edges(plot):
    x0, z0, x1, z1 = plot["rect"]
    return {
        "south": (x0, z0, x1, z0),
        "north": (x0, z1, x1, z1),
        "west": (x0, z0, x0, z1),
        "east": (x1, z0, x1, z1),
    }


def _edge_key(x0, z0, x1, z1):
    a = (round(x0, 3), round(z0, 3))
    b = (round(x1, 3), round(z1, 3))
    return (a, b) if a <= b else (b, a)


def _inward(name):
    return {"south": (0.0, 1.0), "north": (0.0, -1.0), "west": (1.0, 0.0), "east": (-1.0, 0.0)}[name]


def _fence_run(plot, name, edge, seen) -> List[Placement]:
    product = None if not plot.get("request") else plot["request"].product
    parameters = {"plot_id": plot["id"], "building_product": product}
    x0, z0, x1, z1 = edge
    dx, dz = x1 - x0, z1 - z0
    length = math.hypot(dx, dz)
    if length < 0.8:
        return []
    ix, iz = _inward(name)
    inset = 0.16
    ox, oz = ix * inset, iz * inset
    posts = max(2, int(length // 2.2) + 1)
    members = []
    points = []
    for step in range(posts):
        t = step / (posts - 1)
        px = x0 + dx * t + ox
        pz = z0 + dz * t + oz
        key = (round(px, 3), round(pz, 3))
        points.append((px, pz))
        if key in seen:
            continue
        seen.add(key)
        members.append(_slab(
            f"{plot['id']}.fence.{name}.{step:02d}", "fence", "site",
            px, _YARD_T + 0.55, pz, 0.14, 1.10, 0.14, _OAK, "oak", plot["id"], parameters,
        ))
    along_x = abs(dx) >= abs(dz)
    for step in range(len(points) - 1):
        ax, az = points[step]
        bx, bz = points[step + 1]
        span = math.hypot(bx - ax, bz - az) - 0.16
        if span < 0.4:
            continue
        members.append(_slab(
            f"{plot['id']}.rail.{name}.{step:02d}", "fence", "site",
            (ax + bx) / 2, _YARD_T + 0.62, (az + bz) / 2,
            span if along_x else 0.08, 0.10, 0.08 if along_x else span,
            _OAK, "oak", plot["id"], parameters,
        ))
    return members


def _wall_run(plot, name, edge) -> List[Placement]:
    product = None if not plot.get("request") else plot["request"].product
    parameters = {"plot_id": plot["id"], "building_product": product}
    x0, z0, x1, z1 = edge
    dx, dz = x1 - x0, z1 - z0
    length = math.hypot(dx, dz)
    if length < 0.8:
        return []
    ix, iz = _inward(name)
    # Centre the wall in the setback so it stays inside the plot and clear of the front at 0.40 m.
    inset = 0.18
    ox, oz = ix * inset, iz * inset
    gate = float(plot.get("gateway") or 0.0) if name == plot.get("facing") else 0.0
    gate = min(gate, length - 1.0)
    members = []
    cursor = 0.0
    index = 0
    block = 1.15
    while cursor < length - 0.3:
        span = min(block, length - cursor)
        mid = cursor + span / 2
        if gate and abs(mid - length / 2) < gate / 2:
            cursor += span
            continue
        t = mid / length
        px = x0 + dx * t + ox
        pz = z0 + dz * t + oz
        along_x = abs(dx) >= abs(dz)
        for course in range(2):
            members.append(_slab(
                f"{plot['id']}.enclosure.{name}.{index:02d}.{course}", "enclosure", "site",
                px, _YARD_T + 0.22 + course * 0.42, pz,
                span if along_x else 0.28, 0.40, 0.28 if along_x else span,
                _WALL, "limestone", plot["id"], parameters,
            ))
        cursor += span
        index += 1
    return members


# ---------------------------------------------------------------- quality



def _overlap_check(_check, by_plot, streets, squares):
    boxes = {plot_id: _union_rect(members) for plot_id, members in by_plot.items() if members}
    ground = [*(_rect(item) for item in streets), *(_rect(item) for item in squares)]
    collisions = []
    names = sorted(boxes)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if _area(boxes[left], boxes[right]) > _OVERLAP_AREA:
                collisions.append(f"{left}|{right}")
        for rect in ground:
            if _area(boxes[left], rect) > _OVERLAP_AREA:
                collisions.append(left)
                break
    collisions = sorted(set(collisions))
    return _check(
        "settlement.overlap",
        not collisions,
        "Buildings stay clear of each other and of the streets.",
        "A building overlaps another building or crosses a street or square.",
        primitive_ids=collisions,
        observed=len(collisions),
        expected=0,
    )


def _frontage_check(_check, plots, by_plot, streets, squares):
    targets = [_rect(item) for item in (*streets, *squares)]
    missed = []
    for plot in plots:
        request = (plot.parameters or {}).get("building_request")
        members = by_plot.get(plot.id) or []
        if not request or not members:
            continue
        yaw = int((plot.parameters or {}).get("yaw", 0))
        probe = _front_probe(members, yaw)
        if probe is None or not any(_contains(probe, rect) for rect in targets):
            missed.append(plot.id)
    return _check(
        "settlement.frontage",
        not missed,
        "Every building entrance faces a street or a square. The entrance is the architect's local -Z side.",
        "A building entrance does not face a street or a square.",
        primitive_ids=missed,
        observed=missed,
        expected=[],
    )


def _network_check(_check, streets, width, depth):
    rects = [_rect(item) for item in streets]
    if not rects:
        return _check(
            "settlement.street_network", False,
            "Streets form one network.", "The settlement has no street.",
            observed=0, expected="connected to the site edge",
        )
    seen = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for index, rect in enumerate(rects):
            if index in seen:
                continue
            if _touches(rects[current], rect):
                seen.add(index)
                pending.append(index)
    connected = len(seen) == len(rects)
    reaches = any(_reaches_edge(rect, width, depth) for index, rect in enumerate(rects) if index in seen)
    return _check(
        "settlement.street_network",
        connected and reaches,
        "Streets form one network and reach the site edge.",
        "Streets are disconnected or stop short of the site edge.",
        observed={"components": _components(rects), "reaches_edge": reaches},
        expected={"components": 1, "reaches_edge": True},
    )


def _building_quality_check(_check, plots, by_plot, scene):
    failures = []
    for plot in plots:
        payload = (plot.parameters or {}).get("building_request")
        members = by_plot.get(plot.id) or []
        if not payload or not members:
            continue
        try:
            local = _restore_plot(plot, members, scene)
            failed = [check for check in _child_checks(local) if check.status != "pass"]
        except Exception as exc:
            failures.append(f"{plot.id}: {exc}")
            continue
        if failed:
            failures.append(plot.id + ": " + ", ".join(check.code for check in failed[:6]))
    return _check(
        "settlement.building_quality",
        not failures,
        "Every building passes its own architect's quality gates.",
        "A placed building no longer passes its architect's quality gates.",
        primitive_ids=failures,
        observed=failures,
        expected=[],
    )


def _ground_coverage_check(_check, document, by_plot, width, depth):
    roles = {"street", "kerb", "stepping_stone", "square", "plot", "garden", "pavement", "podium"}
    rects = [_rect(item) for item in document.placements if item.role in roles]
    for members in by_plot.values():
        if members:
            rects.append(_union_rect(members))
    ratio = _raster_coverage(rects, width, depth)
    return _check(
        "settlement.ground_coverage",
        ratio >= _COVERAGE_MIN,
        "Ground and building footprints cover the site.",
        "Part of the site has no ground and no building.",
        observed=round(ratio, 4),
        expected={"minimum": _COVERAGE_MIN},
    )


def _street_frontage_check(_check, document, by_plot):
    edges = ((document.metadata or {}).get("settlement") or {}).get("frontage_edges") or []
    footprints = [_union_rect(members) for members in by_plot.values() if members]
    ratios = []
    failed = []
    for edge in edges:
        ratio = _edge_ratio(edge, footprints)
        if ratio is None:
            continue
        ratios.append({"id": edge.get("id"), "ratio": round(ratio, 4)})
        if ratio < _FRONTAGE_MIN - 1e-6:
            failed.append(edge.get("id") or "edge")
    minimum = None if not ratios else min(item["ratio"] for item in ratios)
    return _check(
        "settlement.street_frontage",
        not failed,
        "Built frontage covers each side of the main street, green or market.",
        "A side of the main street, green or market is less than 65 percent building.",
        primitive_ids=failed,
        observed={"minimum": minimum, "sides": ratios},
        expected={"minimum": _FRONTAGE_MIN},
    )


def _insula_coverage_check(_check, document, plots, by_plot):
    insulae = ((document.metadata or {}).get("settlement") or {}).get("insulae") or []
    residential = [item for item in insulae if item.get("use") == "residential"]
    ratios = []
    failed = []
    for insula in residential:
        rect = tuple(insula["rect"])
        area = max(1e-6, (rect[2] - rect[0]) * (rect[3] - rect[1]))
        covered = 0.0
        for plot in plots:
            if (plot.parameters or {}).get("insula_id") != insula["id"]:
                continue
            if not (plot.parameters or {}).get("building_request"):
                continue
            members = by_plot.get(plot.id) or []
            if not members:
                continue
            covered += _clipped_area(_union_rect(members), rect)
        ratio = covered / area
        ratios.append({"id": insula["id"], "ratio": round(ratio, 4)})
        if ratio < _INSULA_MIN - 1e-6:
            failed.append(insula["id"])
    minimum = None if not ratios else min(item["ratio"] for item in ratios)
    return _check(
        "settlement.insula_coverage",
        not failed,
        "Residential insulae are built out.",
        "A residential insula is less than 45 percent building.",
        primitive_ids=failed,
        observed={"minimum": minimum, "insulae": ratios},
        expected={"minimum": _INSULA_MIN},
    )


def _raster_coverage(rects, width, depth, step=0.25) -> float:
    if width <= 0 or depth <= 0:
        return 0.0
    nx = max(1, int(round(width / step)))
    nz = max(1, int(round(depth / step)))
    cover = np.zeros((nx, nz), dtype=np.uint8)
    for rect in rects:
        x0, z0, x1, z1 = rect
        ix0 = max(0, int(math.ceil(x0 / step - 0.5 - 1e-9)))
        ix1 = min(nx, int(math.floor(x1 / step - 0.5 + 1e-9)) + 1)
        iz0 = max(0, int(math.ceil(z0 / step - 0.5 - 1e-9)))
        iz1 = min(nz, int(math.floor(z1 / step - 0.5 + 1e-9)) + 1)
        if ix1 > ix0 and iz1 > iz0:
            cover[ix0:ix1, iz0:iz1] = 1
    return float(cover.sum()) / float(nx * nz)


def _edge_ratio(edge, footprints) -> Optional[float]:
    start, end = float(edge["start"]), float(edge["end"])
    included = _subtract_intervals([(start, end)], edge.get("exclude") or [])
    denom = sum(b - a for a, b in included)
    if denom < 0.5:
        return None
    along_x = edge.get("axis") == "x"
    at = float(edge["at"])
    side = int(edge.get("side") or 1)
    built = []
    for rect in footprints:
        if along_x:
            gap = (rect[1] - at) if side > 0 else (at - rect[3])
            span = (rect[0], rect[2])
        else:
            gap = (rect[0] - at) if side > 0 else (at - rect[2])
            span = (rect[1], rect[3])
        if -0.3 <= gap <= _FRONT_REACH:
            built.append(span)
    return _overlap_length(included, built) / denom


def _subtract_intervals(base, cuts):
    pieces = list(base)
    for cut in cuts:
        a, b = float(cut[0]), float(cut[1])
        if b < a:
            a, b = b, a
        nxt = []
        for left, right in pieces:
            if b <= left or a >= right:
                nxt.append((left, right))
                continue
            if a > left:
                nxt.append((left, a))
            if b < right:
                nxt.append((b, right))
        pieces = [(left, right) for left, right in nxt if right - left > 1e-6]
    return pieces


def _overlap_length(included, built) -> float:
    total = 0.0
    for left, right in included:
        pieces = _subtract_intervals([(left, right)], [])
        covered = []
        for a, b in built:
            lo, hi = (a, b) if a <= b else (b, a)
            if hi <= left or lo >= right:
                continue
            covered.append((max(left, lo), min(right, hi)))
        total += _union_length(covered)
    return total


def _union_length(spans) -> float:
    if not spans:
        return 0.0
    ordered = sorted(spans)
    total = 0.0
    start, end = ordered[0]
    for left, right in ordered[1:]:
        if left > end:
            total += end - start
            start, end = left, right
        else:
            end = max(end, right)
    return total + end - start


def _clipped_area(foot, insula) -> float:
    x0 = max(foot[0], insula[0])
    z0 = max(foot[1], insula[1])
    x1 = min(foot[2], insula[2])
    z1 = min(foot[3], insula[3])
    if x1 <= x0 or z1 <= z0:
        return 0.0
    return (x1 - x0) * (z1 - z0)


def _restore_plot(plot: Placement, members: Sequence[Placement], scene: Any) -> Tuple[BuildDocument, Any]:
    yaw = int(plot.parameters.get("yaw", 0))
    anchor = tuple(plot.parameters.get("anchor") or (0.0, 0.0, 0.0))
    prefix = plot.id + "."
    steps = _steps(yaw)
    inverse = (4 - steps) % 4
    restored = []
    world_ids = []
    for member in members:
        parameters = dict(member.parameters or {})
        source_id = parameters.pop("source_id")
        source_parent = parameters.pop("source_parent", None)
        parameters.pop("plot_id", None)
        parameters.pop("building_product", None)
        parameters = _strip_prefix(parameters, prefix)
        dimensions, parameters = _rotate_payload(member.dimensions, parameters, member.block_type, inverse)
        position = _inverse_position(member.position, steps, anchor)
        restored.append(Placement(
            id=source_id,
            kind=member.kind,
            block_type=member.block_type,
            position=position,
            dimensions=dimensions,
            material=member.material,
            role=member.role,
            parameters=parameters,
            parent_id=source_parent,
            substance=member.substance,
        ))
        world_ids.append((member.id, source_id, member.block_type))
    request = BuildRequest.from_dict(plot.parameters["building_request"])
    document = BuildDocument(
        architect=ArchitectInfo("settlement", "1.0.0", "Settlement Planner"),
        request=request,
        placements=restored,
    )
    local_scene = _local_scene(scene, world_ids, yaw, anchor) if _needs_scene(request.product) else None
    return document, local_scene


def _strip_prefix(value, prefix):
    if isinstance(value, str):
        return value[len(prefix):] if value.startswith(prefix) else value
    if isinstance(value, list):
        return [_strip_prefix(item, prefix) for item in value]
    if isinstance(value, dict):
        return {key: _strip_prefix(item, prefix) for key, item in value.items()}
    return value


def _needs_scene(product: str) -> bool:
    return product in {"timber-house", "classical-portico", "classical-basilica", "classical-villa",
                       "classical-stoa", "classical-peristyle", "classical-rotunda", "classical-forum"}


def _local_scene(scene, world_ids, yaw, anchor):
    by_id = {}
    for block in getattr(scene, "blocks", ()) or ():
        parameters = getattr(block, "parameters", {}) or {}
        primitive = parameters.get("primitive_id")
        if primitive is not None:
            by_id[str(primitive)] = block
    blocks = []
    steps = _steps(yaw)
    for world_id, source_id, block_type in world_ids:
        block = by_id.get(world_id)
        mesh = getattr(block, "mesh", None) if block is not None else None
        if mesh is None:
            continue
        copied = mesh.copy()
        vertices = np.array(copied.vertices, dtype=float, copy=True)
        dx = vertices[:, 0] - anchor[0]
        dz = vertices[:, 2] - anchor[2]
        c, s = _CS[steps]
        vertices[:, 0] = dx * c + dz * s
        vertices[:, 2] = -dx * s + dz * c
        vertices[:, 1] = vertices[:, 1] - anchor[1]
        copied.vertices = vertices
        blocks.append(_FakeBlock(source_id, copied))
    return _FakeScene(blocks)


class _FakeBlock:
    def __init__(self, primitive_id, mesh):
        self.parameters = {"primitive_id": primitive_id}
        self.mesh = mesh


class _FakeScene:
    def __init__(self, blocks):
        self.blocks = blocks


def _child_checks(local):
    document, scene = local
    product = canonical_product_name(document.request.product)
    checks = []
    if product == "gothic-cathedral":
        from .quality import cathedral_checks
        checks.extend(cathedral_checks(document))
    elif product == "timber-house":
        from .timberwright import timber_checks
        checks.extend(timber_checks(document, scene))
    elif product == "iwan-court":
        from .persian import persian_checks
        checks.extend(persian_checks(document, scene))
    else:
        from .classical_forms import FORMS
        if product in FORMS:
            from .quality import classical_form_checks
            checks.extend(classical_form_checks(document, product))
            if any(placement.role == "column_shaft" for placement in document.placements):
                from .construction import order_checks
                checks.extend(order_checks(document, scene))
        from .historic import historic_checks
        checks.extend(historic_checks(document))
    from .masonry import ashlar_checks
    checks.extend(ashlar_checks(document))
    return checks


def _rect(placement: Placement) -> Tuple[float, float, float, float]:
    x, _, z = (float(v) for v in placement.position)
    w, _, d = (float(v) for v in placement.dimensions)
    return (x - w / 2, z - d / 2, x + w / 2, z + d / 2)


def _union_rect(members: Sequence[Placement]) -> Tuple[float, float, float, float]:
    rects = [_rect(member) for member in members]
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def _area(left, right) -> float:
    dx = min(left[2], right[2]) - max(left[0], right[0])
    dz = min(left[3], right[3]) - max(left[1], right[1])
    if dx <= 0.02 or dz <= 0.02:
        return 0.0
    return dx * dz


def _contains(point, rect) -> bool:
    x, z = point
    return rect[0] - 0.05 <= x <= rect[2] + 0.05 and rect[1] - 0.05 <= z <= rect[3] + 0.05


def _front_probe(members, yaw):
    rect = _union_rect(members)
    x0, z0, x1, z1 = rect
    if yaw == 0:
        face = ((x0 + x1) / 2, z0)
        normal = (0.0, -1.0)
    elif yaw == 90:
        face = (x1, (z0 + z1) / 2)
        normal = (1.0, 0.0)
    elif yaw == 180:
        face = ((x0 + x1) / 2, z1)
        normal = (0.0, 1.0)
    elif yaw == 270:
        face = (x0, (z0 + z1) / 2)
        normal = (-1.0, 0.0)
    else:
        return None
    return (face[0] + normal[0] * _FRONT_PROBE, face[1] + normal[1] * _FRONT_PROBE)


def _touches(left, right, gap=0.05) -> bool:
    dx = min(left[2], right[2]) - max(left[0], right[0])
    dz = min(left[3], right[3]) - max(left[1], right[1])
    return dx >= -gap and dz >= -gap


def _reaches_edge(rect, width, depth, tolerance=0.05) -> bool:
    return rect[0] <= tolerance or rect[2] >= width - tolerance or rect[1] <= tolerance or rect[3] >= depth - tolerance


def _components(rects) -> int:
    seen = set()
    components = 0
    for start in range(len(rects)):
        if start in seen:
            continue
        components += 1
        pending = [start]
        seen.add(start)
        while pending:
            current = pending.pop()
            for index, rect in enumerate(rects):
                if index in seen:
                    continue
                if _touches(rects[current], rect):
                    seen.add(index)
                    pending.append(index)
    return components


__all__ = [
    "ARCHITECT",
    "ENGLISH",
    "ROMAN",
    "PRODUCT_NAMES",
    "DEFAULT_SITES",
    "build_document",
    "list_products",
    "random_controls",
    "settlement_checks",
    "turn_members",
    "unturn_members",
]
