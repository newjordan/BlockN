"""Built-in semantic architects for BlockN.

The builders in this module deliberately emit semantic :class:`Placement`
records instead of mutating ``core.scene.Scene``.  A later compiler can turn
the placements into the existing mesh/block types while retaining stable IDs,
roles, parent relationships, and deterministic ordering.

All geometry follows the repository convention: X is width, Y is up, Z is
depth; positions are geometric centers; dimensions are full extents.
"""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import trimesh

from core.architectural_geometry import mesh_parameters
from core.primitives import MESH
from core.scene import CUBE, CYLINDER, DOME, WEDGE

from .masonry import (
    MAX_COURSE,
    MAX_STRETCHER,
    MIN_STONE,
    course_profile_tiles,
    iter_ashlar,
    iter_masonry_solid,
    iter_stack,
    radial_ashlar_mesh,
    radial_divisions,
    stretcher_target,
)
from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement


VERSION = "1.1.0"
SCHEMA_VERSION = "blockn.architecture/v1"
EPSILON = 1e-6
MAX_PLACEMENTS = 30_000

RGBA = Tuple[int, int, int, int]


# Palettes are keyed by semantic material role.  Keeping color choice here,
# rather than inside individual builders, makes style changes coherent across
# products and gives exporters a dependable material signal.
PALETTES: Dict[str, Dict[str, RGBA]] = {
    "contemporary": {
        "primary": (210, 215, 218, 255),
        "secondary": (144, 153, 160, 255),
        "structure": (104, 113, 120, 255),
        "accent": (72, 143, 171, 255),
        "roof": (62, 70, 78, 255),
        "site": (126, 132, 128, 255),
        "glass": (92, 166, 194, 190),
    },
    "classical": {
        "primary": (220, 214, 198, 255),
        "secondary": (187, 178, 158, 255),
        "structure": (163, 153, 135, 255),
        "accent": (181, 137, 78, 255),
        "roof": (114, 103, 105, 255),
        "site": (151, 145, 132, 255),
        "glass": (113, 155, 170, 190),
    },
    "industrial": {
        "primary": (151, 157, 159, 255),
        "secondary": (91, 98, 101, 255),
        "structure": (58, 65, 68, 255),
        "accent": (202, 112, 46, 255),
        "roof": (67, 72, 73, 255),
        "site": (112, 109, 102, 255),
        "glass": (83, 130, 143, 190),
    },
    "timber": {
        "primary": (178, 139, 91, 255),
        "secondary": (128, 91, 58, 255),
        "structure": (91, 62, 42, 255),
        "accent": (202, 177, 123, 255),
        "roof": (83, 74, 67, 255),
        "site": (126, 119, 95, 255),
        "glass": (104, 153, 166, 190),
    },
}

DETAIL_LEVELS = ("concept", "standard", "presentation")


# Real substances behind each palette's material roles, keyed by style then by
# the material role passed to _BuildContext.add.  Keeping the physical choice
# beside the colour palette means a style stays coherent in both channels.
STYLE_SUBSTANCES: Dict[str, Dict[str, str]] = {
    "contemporary": {
        "primary": "limestone",
        "secondary": "sandstone",
        "structure": "engineering-brick",
        "accent": "glazed-tile",
        "roof": "slate",
        "site": "tuff",
        "glass": "window-glass",
    },
    "classical": {
        "primary": "marble",
        "secondary": "limestone",
        "structure": "limestone",
        "accent": "terracotta",
        "roof": "clay-roof-tile",
        "site": "limestone",
        "glass": "window-glass",
    },
    "industrial": {
        "primary": "fired-clay-brick",
        "secondary": "engineering-brick",
        "structure": "wrought-iron",
        "accent": "wrought-iron",
        "roof": "lead-sheet",
        "site": "basalt",
        "glass": "window-glass",
    },
    "timber": {
        "primary": "oak",
        "secondary": "oak",
        "structure": "softwood",
        "accent": "cedar",
        "roof": "clay-roof-tile",
        "site": "limestone",
        "glass": "window-glass",
    },
}

BUILTIN_SUBSTANCES = sorted(
    {
        substance_id
        for role_map in STYLE_SUBSTANCES.values()
        for substance_id in role_map.values()
    }
)


def _catalog_item(
    name: str,
    category: str,
    architect_id: str,
    title: str,
    description: str,
    defaults: Mapping[str, Any],
    variants: Sequence[str],
    opening_modes: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "architect": {"id": architect_id, "title": title, "version": VERSION},
        # Flat aliases keep CLI/catalog consumers simple while preserving the
        # grouped architect record as the authoritative shape.
        "architect_id": architect_id,
        "architect_title": title,
        "architect_version": VERSION,
        "description": description,
        "defaults": dict(defaults),
        "variants": list(variants),
        "opening_modes": list(opening_modes),
        "substances": list(BUILTIN_SUBSTANCES),
    }


_PRODUCTS: List[Dict[str, Any]] = [
    _catalog_item(
        "wall", "primitive", "primitive", "Primitive Architect",
        "Coursed ashlar wall of human-scale stones with optional door, window, mixed, or arch openings.",
        {"width": 8.0, "height": 3.2, "depth": 0.3, "variant": "x", "openings": "none"},
        ("x", "z"), ("none", "door", "window", "mixed", "arch"),
    ),
    _catalog_item(
        "slab", "primitive", "primitive", "Primitive Architect",
        "Grounded floor or roof slab with optional expressed perimeter edge.",
        {"width": 8.0, "height": 0.3, "depth": 6.0, "variant": "floor", "openings": "none"},
        ("floor", "roof", "foundation"), (),
    ),
    _catalog_item(
        "column", "primitive", "primitive", "Primitive Architect",
        "Square, round, or proportioned classical vertical support.",
        {"width": 0.8, "height": 4.0, "depth": 0.8, "variant": "round", "openings": "none"},
        ("square", "round", "classical"), (),
    ),
    _catalog_item(
        "beam", "primitive", "primitive", "Primitive Architect",
        "Axis-aligned structural beam with optional bearing blocks.",
        {"width": 8.0, "height": 0.5, "depth": 0.4, "variant": "x", "openings": "none"},
        ("x", "z"), (),
    ),
    _catalog_item(
        "stair", "primitive", "circulation", "Circulation Architect",
        "Grounded straight stair assembled from deterministic cumulative step solids.",
        {"width": 2.0, "height": 3.0, "depth": 5.0, "variant": "+z", "openings": "none", "steps": 12},
        ("+z", "-z", "+x", "-x"), (),
    ),
    _catalog_item(
        "ramp", "primitive", "circulation", "Circulation Architect",
        "Straight accessible or service ramp with optional edge curbs.",
        {"width": 2.0, "height": 1.0, "depth": 8.0, "variant": "+z", "openings": "none"},
        ("+z", "-z", "+x", "-x"), (),
    ),
    _catalog_item(
        "roof", "primitive", "envelope", "Envelope Architect",
        "Flat, gable, or dome roof primitive grounded at its bearing plane.",
        {"width": 8.0, "height": 2.0, "depth": 6.0, "variant": "gable", "openings": "none"},
        ("flat", "gable", "dome"), (),
    ),
    _catalog_item(
        "site-pad", "primitive", "site", "Site Architect",
        "Grounded site plinth with optional terrace and approach path articulation.",
        {"width": 14.0, "height": 0.5, "depth": 10.0, "variant": "rectangular", "openings": "none"},
        ("rectangular", "terraced"), (),
    ),
    _catalog_item(
        "platform", "primitive", "circulation", "Circulation Architect",
        "Supported, floating, or cantilevered platform with optional guard edges.",
        {"width": 5.0, "height": 2.5, "depth": 4.0, "variant": "supported", "openings": "none"},
        ("supported", "floating", "cantilevered"), (),
    ),
    _catalog_item(
        "bridge", "primitive", "circulation", "Circulation Architect",
        "Ground-supported deck bridge with optional rails or expressed truss rhythm.",
        {"width": 8.0, "height": 2.5, "depth": 1.8, "variant": "deck", "openings": "none"},
        ("deck", "railed", "truss"), (),
    ),
    _catalog_item(
        "room-shell", "assembly", "plan", "Plan Architect",
        "Floor slab and four coursed ashlar walls forming a reusable single-room shell.",
        {"width": 8.0, "height": 3.2, "depth": 6.0, "variant": "rectangular", "openings": "mixed"},
        ("rectangular", "pavilion"), ("none", "door", "window", "mixed", "arch"),
    ),
    _catalog_item(
        "arcade", "assembly", "classical", "Classical Architect",
        "Repeated pier-and-arch bays with coherent rhythm, base, and cornice.",
        {"width": 12.0, "height": 5.0, "depth": 1.0, "variant": "roman", "openings": "arch", "bays": 4},
        ("roman", "light", "monumental"), ("arch",),
    ),
    _catalog_item(
        "structural-grid", "assembly", "structure", "Structural Architect",
        "Foundations, columns, beams, and optional floor plates on a regular multi-level grid.",
        {"width": 12.0, "height": 4.0, "depth": 9.0, "variant": "frame", "openings": "none", "bays": 3, "levels": 1},
        ("frame", "slab-frame", "heavy"), (),
    ),
    _catalog_item(
        "facade", "assembly", "envelope", "Envelope Architect",
        "Bayed ashlar facade with hosted openings and style-aware pilasters, frames, and cornice.",
        {"width": 12.0, "height": 5.0, "depth": 0.35, "variant": "bayed", "openings": "mixed", "bays": 4},
        ("bayed", "storefront", "monumental"), ("none", "door", "window", "mixed", "arch"),
    ),
    _catalog_item(
        "spire-cluster", "assembly", "spire", "Spire Architect",
        "Seeded cluster of towers, platforms, bridges, and a grounded access stair.",
        {"width": 16.0, "height": 12.0, "depth": 14.0, "variant": "connected", "openings": "none", "bays": 3},
        ("connected", "citadel", "needle"), (),
    ),
]

_PRODUCT_BY_NAME = {item["name"]: item for item in _PRODUCTS}
_ALIASES = {name.replace("-", "_"): name for name in _PRODUCT_BY_NAME}

_EXTRA_PARAMETERS: Dict[str, set[str]] = {
    "stair": {"steps"},
    "ramp": {"max_slope"},
    "platform": {"deck_thickness"},
    "bridge": {"deck_thickness"},
    "structural-grid": {"bays_x", "bays_z"},
    "spire-cluster": {"spire_count"},
}

for _product_descriptor in _PRODUCTS:
    _product_descriptor["parameters"] = sorted(
        _EXTRA_PARAMETERS.get(_product_descriptor["name"], set())
    )


def list_products() -> list[dict]:
    """Return the deterministic built-in product catalog."""
    return deepcopy(_PRODUCTS)


def describe_product(name: str) -> dict:
    """Return one catalog entry or raise ``ValueError`` for an unknown name."""
    normalized = _normalize_product_name(name)
    return deepcopy(_PRODUCT_BY_NAME[normalized])


def _normalize_product_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("product must be a non-empty string")
    normalized = name.strip().lower().replace("_", "-")
    if normalized not in _PRODUCT_BY_NAME:
        choices = ", ".join(_PRODUCT_BY_NAME)
        raise ValueError(f"unknown product '{name}'; expected one of: {choices}")
    return normalized


def _as_float(name: str, value: Any, *, minimum: float = EPSILON, maximum: float = 10_000.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


def _as_int(name: str, value: Any, *, minimum: int = 1, maximum: int = 100) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _rounded(value: float) -> float:
    result = round(float(value), 6)
    return 0.0 if result == -0.0 else result


def _tuple3(values: Iterable[float]) -> Tuple[float, float, float]:
    result = tuple(_rounded(value) for value in values)
    if len(result) != 3:
        raise ValueError("expected a three-component vector")
    return result  # type: ignore[return-value]


def _request_parameters(request: BuildRequest, item: Mapping[str, Any]) -> Dict[str, Any]:
    params = dict(item["defaults"])
    supplied = getattr(request, "parameters", {}) or {}
    if not isinstance(supplied, Mapping):
        raise ValueError("parameters must be a mapping")
    allowed_parameters = set(item["defaults"]) | _EXTRA_PARAMETERS.get(item["name"], set())
    unknown_parameters = sorted(str(key) for key in supplied if key not in allowed_parameters)
    if unknown_parameters:
        raise ValueError(
            f"unsupported parameter(s) for {item['name']}: {', '.join(unknown_parameters)}"
        )
    params.update(dict(supplied))

    for field in ("width", "height", "depth", "bays", "levels"):
        value = getattr(request, field, None)
        if value is not None:
            if field in {"bays", "levels"} and field not in item["defaults"]:
                raise ValueError(f"{item['name']} does not support --{field}")
            params[field] = value

    requested_variant = str(getattr(request, "variant", "auto") or "auto").strip().lower()
    if requested_variant != "auto":
        params["variant"] = requested_variant

    requested_openings = getattr(request, "openings", "auto")
    if requested_openings not in (None, "", "auto"):
        params["openings"] = requested_openings

    params["width"] = _as_float("width", params["width"])
    params["height"] = _as_float("height", params["height"])
    params["depth"] = _as_float("depth", params["depth"])
    if "bays" in params:
        params["bays"] = _as_int("bays", params["bays"], maximum=24)
    if "levels" in params:
        params["levels"] = _as_int("levels", params["levels"], maximum=12)

    variant = str(params.get("variant", item["variants"][0])).strip().lower()
    if variant not in item["variants"]:
        allowed = ", ".join(item["variants"])
        raise ValueError(f"variant for {item['name']} must be one of: {allowed}")
    params["variant"] = variant

    openings = params.get("openings", "none")
    if isinstance(openings, str):
        openings = openings.strip().lower()
        allowed_openings = item["opening_modes"]
        if allowed_openings and openings not in allowed_openings:
            allowed = ", ".join(allowed_openings)
            raise ValueError(f"openings for {item['name']} must be one of: {allowed}")
        if not allowed_openings and openings not in ("none", "auto"):
            raise ValueError(f"{item['name']} does not support openings")
        params["openings"] = "none" if openings == "auto" else openings
    elif not isinstance(openings, (list, tuple)):
        raise ValueError("openings must be a catalog mode or a list of opening specifications")

    return params


class _BuildContext:
    def __init__(
        self,
        product: str,
        style: str,
        detail: str,
        seed: int,
        params: Dict[str, Any],
    ) -> None:
        self.product = product
        self.style = style
        self.detail = detail
        self.seed = seed
        self.params = params
        self.rng = random.Random(seed)
        self.palette = PALETTES[style]
        self.placements: List[Placement] = []
        # The player the build is for (``profile.PlayerProfile``) and its
        # derived clearances; both stay None for a request without a profile.
        self.player = None
        self.clear = None
        # Doorways the architect cut, recorded when a profile is set.
        self.ports: List[Dict[str, Any]] = []
        # Parameter names the request set itself, as opposed to defaults.
        self.explicit: set = set()

    def add(
        self,
        placement_id: str,
        kind: str,
        block_type: str,
        position: Sequence[float],
        dimensions: Sequence[float],
        material_role: str,
        role: str,
        *,
        parameters: Optional[Mapping[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> Placement:
        if len(self.placements) >= MAX_PLACEMENTS:
            raise ValueError(f"generation exceeds the {MAX_PLACEMENTS} placement safety limit")
        if any(existing.id == placement_id for existing in self.placements):
            raise ValueError(f"duplicate placement id '{placement_id}'")
        pos = _tuple3(position)
        dims = _tuple3(dimensions)
        if any((not math.isfinite(value) or value <= 0.0) for value in dims):
            raise ValueError(f"placement '{placement_id}' has invalid dimensions {dims}")
        if pos[1] - dims[1] / 2.0 < -EPSILON:
            raise ValueError(f"placement '{placement_id}' extends below Y=0")
        if material_role not in self.palette:
            raise ValueError(f"unknown material role '{material_role}'")
        semantic_parameters = {
            "product": self.product,
            "semantic_role": role,
            "material_role": material_role,
            "style": self.style,
            "detail": self.detail,
        }
        if parameters:
            semantic_parameters.update(dict(parameters))
        placement = Placement(
            id=placement_id,
            kind=kind,
            block_type=block_type,
            position=pos,
            dimensions=dims,
            material=self.palette[material_role],
            role=role,
            parameters=semantic_parameters,
            parent_id=parent_id,
            substance=STYLE_SUBSTANCES[self.style][material_role],
        )
        self.placements.append(placement)
        return placement


def _opening_specs(
    mode_or_specs: Any,
    length: float,
    wall_height: float,
    thickness: float,
    *,
    player: Any = None,
) -> List[Dict[str, Any]]:
    if isinstance(mode_or_specs, str):
        mode = mode_or_specs.lower()
        door_width = min(max(1.0, length * 0.16), max(1.0, length * 0.30))
        door_height = min(max(2.0, wall_height * 0.68), wall_height * 0.88)
        if player is not None:
            # A profile sets the smallest door its player fits through. Named
            # opening modes grow to it; explicit opening specs are left as
            # written and the playability checks report them instead.
            clear = player.clearances()
            door_width = max(door_width, clear.door_width)
            door_height = max(door_height, clear.door_height)
            if door_height >= wall_height - 0.1 - EPSILON:
                raise ValueError(
                    f"wall height {wall_height:.2f} is too low for a '{player.name}' door "
                    f"({door_height:.2f} clear plus a header); raise the height"
                )
        window_width = min(max(1.2, length * 0.18), max(1.2, length * 0.28))
        window_height = min(max(0.9, wall_height * 0.36), wall_height * 0.55)
        window_sill = min(max(0.65, wall_height * 0.24), wall_height - window_height - 0.25)
        if mode in ("none", "auto"):
            raw: List[Dict[str, Any]] = []
        elif mode == "door":
            raw = [{"id": "door.000", "kind": "door", "center": 0.0, "width": door_width, "height": door_height, "sill": 0.0}]
        elif mode == "window":
            raw = [{"id": "window.000", "kind": "window", "center": 0.0, "width": window_width, "height": window_height, "sill": window_sill}]
        elif mode == "arch":
            raw = [{"id": "arch.000", "kind": "arch", "center": 0.0, "width": door_width * 1.18, "height": door_height, "sill": 0.0}]
        elif mode == "mixed":
            offset = length * 0.25
            raw = [
                {"id": "door.000", "kind": "door", "center": -offset, "width": door_width, "height": door_height, "sill": 0.0},
                {"id": "window.000", "kind": "window", "center": offset, "width": window_width, "height": window_height, "sill": window_sill},
            ]
        else:
            raise ValueError(f"unsupported opening mode '{mode}'")
    else:
        raw = [dict(spec) for spec in mode_or_specs]

    result: List[Dict[str, Any]] = []
    for index, spec in enumerate(raw):
        kind = str(spec.get("kind", "window")).strip().lower()
        if kind not in ("door", "window", "arch"):
            raise ValueError(f"opening {index} kind must be door, window, or arch")
        width = _as_float(f"opening {index} width", spec.get("width"))
        height = _as_float(f"opening {index} height", spec.get("height"))
        sill = _as_float(f"opening {index} sill", spec.get("sill", 0.0), minimum=0.0)
        try:
            center = float(spec.get("center", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"opening {index} center must be a number") from exc
        if not math.isfinite(center):
            raise ValueError(f"opening {index} center must be finite")
        if sill + height >= wall_height - EPSILON:
            raise ValueError(f"opening {index} must leave a positive wall header")
        edge_clearance = max(thickness, 0.12)
        if center - width / 2.0 < -length / 2.0 + edge_clearance - EPSILON:
            raise ValueError(f"opening {index} exceeds the wall start clearance")
        if center + width / 2.0 > length / 2.0 - edge_clearance + EPSILON:
            raise ValueError(f"opening {index} exceeds the wall end clearance")
        result.append({
            "id": str(spec.get("id", f"opening.{index:03d}")),
            "kind": kind,
            "center": center,
            "width": width,
            "height": height,
            "sill": sill,
        })

    result.sort(key=lambda spec: (spec["center"], spec["id"]))
    for left, right in zip(result[:-1], result[1:]):
        left_edge = left["center"] + left["width"] / 2.0
        right_edge = right["center"] - right["width"] / 2.0
        if left_edge > right_edge - max(thickness * 0.5, 0.08):
            raise ValueError(f"openings '{left['id']}' and '{right['id']}' overlap or leave no jamb")
    return result


def _axis_position(axis: str, along: float, y: float, cross: float = 0.0) -> Tuple[float, float, float]:
    return (along, y, cross) if axis == "x" else (cross, y, along)


def _axis_dimensions(axis: str, length: float, height: float, thickness: float) -> Tuple[float, float, float]:
    return (length, height, thickness) if axis == "x" else (thickness, height, length)


def _ashlar_parameters(stone, host_id: str, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {
        "axis": stone.axis,
        "course": stone.course,
        "unit": stone.unit,
        "bond": "running",
        "construction": "ashlar running bond",
        "host_id": host_id,
        "stone_index": stone.index,
    }
    if stone.opening_id:
        parameters["opening_id"] = stone.opening_id
        parameters["opening_index"] = stone.opening_index
    if extra:
        parameters.update(dict(extra))
    return parameters


def _emit_masonry(
    ctx: _BuildContext,
    root_id: str,
    kind: str,
    stones,
    material_role: str,
    role: str,
    *,
    parent_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    role_map: Optional[Mapping[str, str]] = None,
) -> None:
    mapping = dict(role_map or {})
    for stone in stones:
        unit_role = mapping.get(stone.unit, role)
        placement_id = f"{root_id}.{stone.unit}.c{stone.course:03d}.{stone.index:04d}"
        if stone.unit == "lintel":
            material = "secondary" if material_role == "primary" else material_role
        else:
            material = material_role
        ctx.add(
            placement_id, kind, CUBE, stone.position, stone.dimensions, material, unit_role,
            parameters=_ashlar_parameters(stone, root_id, extra),
            parent_id=parent_id or root_id,
        )


def _box_is_human_scale(dimensions: Sequence[float]) -> bool:
    width, height, depth = (float(value) for value in dimensions)
    return height <= MAX_COURSE + EPSILON and max(width, depth) <= MAX_STRETCHER + EPSILON


def _solid_parameters(root_id: str, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(extra or {})
    payload.setdefault("host_id", root_id)
    return payload


def _emit_solid(
    ctx: _BuildContext,
    root_id: str,
    kind: str,
    position: Sequence[float],
    dimensions: Sequence[float],
    material_role: str,
    role: str,
    *,
    parent_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit one CUBE, or running-bond ashlars when the box exceeds human scale."""
    payload = _solid_parameters(root_id, extra)
    if _box_is_human_scale(dimensions):
        ctx.add(
            root_id, kind, CUBE, position, dimensions, material_role, role,
            parameters=payload, parent_id=parent_id,
        )
        return
    remaining = max(1, MAX_PLACEMENTS - len(ctx.placements))
    stones = iter_masonry_solid(position, dimensions, ctx.detail, max_stones=remaining)
    _emit_masonry(
        ctx, root_id, kind, stones, material_role, role,
        parent_id=parent_id, extra=payload,
    )


def _emit_mesh(
    ctx: _BuildContext,
    root_id: str,
    kind: str,
    mesh,
    material_role: str,
    role: str,
    *,
    parent_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    unit: str = "profile_course",
    course: int = 0,
    index: int = 0,
) -> None:
    position, dimensions, payload = mesh_parameters(mesh)
    bottom = position[1] - dimensions[1] / 2.0
    if bottom < -1e-4:
        raise ValueError(f"placement '{root_id}' extends below Y=0")
    if bottom < 0.0:
        position = (position[0], position[1] - bottom, position[2])
    parameters = dict(extra or {})
    parameters.update(payload)
    parameters.update({
        "construction": str(parameters.get("construction") or "ashlar running bond"),
        "unit": unit,
        "course": int(course),
        "stone_index": int(index),
    })
    parameters.setdefault("host_id", root_id)
    ctx.add(
        f"{root_id}.{unit}.c{int(course):03d}.{int(index):04d}",
        kind, MESH, position, dimensions, material_role, role,
        parameters=parameters, parent_id=parent_id or root_id,
    )


def _axis_aligned_rect(
    polygon: Sequence[Sequence[float]],
    depth: float,
    origin: Sequence[float],
    axis: str,
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    points = [(float(point[0]), float(point[1])) for point in polygon]
    if len(points) < 4:
        return None
    for index, point in enumerate(points):
        other = points[(index + 1) % len(points)]
        if abs(point[0] - other[0]) > 1e-8 and abs(point[1] - other[1]) > 1e-8:
            return None
    xs = sorted({round(point[0], 6) for point in points})
    ys = sorted({round(point[1], 6) for point in points})
    if len(xs) != 2 or len(ys) != 2:
        return None
    x0, x1 = xs[0], xs[1]
    y0, y1 = ys[0], ys[1]
    if min(x1 - x0, y1 - y0, float(depth)) < MIN_STONE * 0.5:
        return None
    along = (x0 + x1) / 2.0
    height = y1 - y0
    length = x1 - x0
    center_y = (y0 + y1) / 2.0
    if axis == "x":
        return (
            (origin[0] + along, center_y, origin[2]),
            (length, height, float(depth)),
        )
    return (
        (origin[0], center_y, origin[2] + along),
        (float(depth), height, length),
    )


def _place_profile_mesh(mesh, origin: Sequence[float], axis: str):
    if axis == "z":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2.0, [0.0, 1.0, 0.0]))
    mesh.apply_translation((float(origin[0]), 0.0, float(origin[2])))
    return mesh


def _emit_plate(
    ctx: _BuildContext,
    root_id: str,
    kind: str,
    polygon: Sequence[Sequence[float]],
    depth: float,
    origin: Sequence[float],
    axis: str,
    material_role: str,
    role: str,
    *,
    parent_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Course an elevation plate: ashlars for a rectangle, MESH tiles otherwise."""
    if axis not in ("x", "z"):
        raise ValueError("plate axis must be 'x' or 'z'")
    rect = _axis_aligned_rect(polygon, depth, origin, axis)
    if rect is not None:
        _emit_solid(
            ctx, root_id, kind, rect[0], rect[1], material_role, role,
            parent_id=parent_id, extra=extra,
        )
        return
    payload = _solid_parameters(root_id, extra)
    for mesh, course, tile in course_profile_tiles(polygon, depth, ctx.detail):
        _place_profile_mesh(mesh, origin, axis)
        _emit_mesh(
            ctx, root_id, kind, mesh, material_role, role,
            parent_id=parent_id, extra=payload, unit="profile_course",
            course=course, index=tile,
        )


def _arc_points(
    radius: float,
    rise: float,
    y0: float,
    start: float,
    end: float,
    segments: int,
) -> List[Tuple[float, float]]:
    count = max(2, int(segments))
    return [
        (radius * math.cos(start + (end - start) * index / count),
         y0 + rise * math.sin(start + (end - start) * index / count))
        for index in range(count + 1)
    ]


def _emit_arch(
    ctx: _BuildContext,
    root_id: str,
    position: Sequence[float],
    dimensions: Sequence[float],
    axis: str,
    material_role: str,
    role: str,
    *,
    parent_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    kind: str = "arch",
) -> None:
    """Jambs plus two half-heads so the opening stays a void when Y-clipped."""
    if axis not in ("x", "z"):
        raise ValueError("arch axis must be 'x' or 'z'")
    cx, cy, cz = (float(value) for value in position)
    dx, dy, dz = (float(value) for value in dimensions)
    along, thick = (dx, dz) if axis == "x" else (dz, dx)
    origin = (cx, 0.0, cz)
    bottom = cy - dy / 2.0
    radius = along / 2.0
    thickness = min(along * 0.16, radius * 0.42)
    thickness = max(MIN_STONE, min(thickness, radius * 0.70))
    rise = min(dy * 0.55, radius)
    spring = dy - rise
    if spring < MIN_STONE:
        spring = 0.0
        rise = dy
    inner_radius = max(radius - thickness, MIN_STONE)
    inner_rise = max(rise - thickness, MIN_STONE)
    segments = {"concept": 8, "standard": 12, "presentation": 16}[ctx.detail]
    payload = dict(extra or {})
    payload.pop("host_id", None)
    if spring > MIN_STONE:
        y1 = bottom + spring
        _emit_plate(
            ctx, f"{root_id}.jamb.left", kind,
            [(-radius, bottom), (-radius + thickness, bottom),
             (-radius + thickness, y1), (-radius, y1)],
            thick, origin, axis, material_role, role,
            parent_id=parent_id, extra=payload,
        )
        _emit_plate(
            ctx, f"{root_id}.jamb.right", kind,
            [(radius - thickness, bottom), (radius, bottom),
             (radius, y1), (radius - thickness, y1)],
            thick, origin, axis, material_role, role,
            parent_id=parent_id, extra=payload,
        )
    y_spring = bottom + spring
    outer_left = _arc_points(radius, rise, y_spring, math.pi, math.pi / 2.0, segments)
    inner_left = _arc_points(inner_radius, inner_rise, y_spring, math.pi / 2.0, math.pi, segments)
    outer_right = _arc_points(radius, rise, y_spring, math.pi / 2.0, 0.0, segments)
    inner_right = _arc_points(inner_radius, inner_rise, y_spring, 0.0, math.pi / 2.0, segments)
    _emit_plate(
        ctx, f"{root_id}.crown.left", kind, outer_left + inner_left[1:],
        thick, origin, axis, material_role, role, parent_id=parent_id, extra=payload,
    )
    _emit_plate(
        ctx, f"{root_id}.crown.right", kind, outer_right + inner_right[1:],
        thick, origin, axis, material_role, role, parent_id=parent_id, extra=payload,
    )


def _emit_dome(
    ctx: _BuildContext,
    root_id: str,
    position: Sequence[float],
    dimensions: Sequence[float],
    material_role: str,
    role: str,
    *,
    parent_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    kind: str = "roof",
) -> None:
    """Annular radial ashlars forming a hollow dome shell with an oculus."""
    cx, cy, cz = (float(value) for value in position)
    width, height, depth = (float(value) for value in dimensions)
    rx, rz = width / 2.0, depth / 2.0
    bottom = cy - height / 2.0
    radius = max(rx, rz)
    thickness = max(MIN_STONE, min(min(rx, rz, height) * 0.12, min(rx, rz) * 0.35))
    module = min(stretcher_target(ctx.detail), 0.85)
    payload = _solid_parameters(root_id, extra)
    index = 0
    oculus = 0.12
    for center_y, course_h, course in iter_stack(bottom, height, ctx.detail):
        y0 = center_y - course_h / 2.0
        y1 = center_y + course_h / 2.0
        t_mid = 0.0 if height <= EPSILON else min(1.0, max(0.0, (center_y - bottom) / height))
        outer = math.sqrt(max(0.0, 1.0 - t_mid * t_mid))
        if outer <= oculus:
            continue
        inner_radius = max(0.0, outer * radius - thickness)
        inner = inner_radius / radius if radius > EPSILON else 0.0
        if inner >= outer - EPSILON:
            continue
        span = (outer - inner) * radius
        bands = max(1, int(math.ceil(span / module - EPSILON)))
        step = (outer - inner) / bands
        for band in range(bands):
            lo = inner + band * step
            hi = lo + step
            facets = max(4, int(math.ceil(math.tau * hi * radius / module - EPSILON)))
            for a0, a1, by0, by1, _course, _facet in radial_divisions(
                0.0, math.tau, y0, y1, hi * radius, ctx.detail, facets,
            ):
                mesh = radial_ashlar_mesh(rx, rz, lo, hi, a0, a1, by0, by1, segments=2)
                mesh.apply_translation((cx, 0.0, cz))
                _emit_mesh(
                    ctx, root_id, kind, mesh, material_role, role,
                    parent_id=parent_id, extra=payload, unit="radial_ashlar",
                    course=course, index=index,
                )
                index += 1
    if index == 0:
        raise ValueError(f"{root_id} dome produced no masonry courses")


def _emit_gable(
    ctx: _BuildContext,
    root_id: str,
    width: float,
    height: float,
    depth: float,
    material_role: str,
    *,
    parent_id: Optional[str] = None,
) -> None:
    thickness = min(max(0.12, min(width, depth, height) * 0.08), 0.35)
    if thickness >= height - MIN_STONE:
        _emit_solid(
            ctx, root_id, "roof", (0.0, height / 2.0, 0.0),
            (width, height, depth), material_role, "roof.gable",
            parent_id=parent_id,
        )
        return
    left = [
        (-width / 2.0, 0.0),
        (0.0, height - thickness),
        (0.0, height),
        (-width / 2.0, thickness),
    ]
    right = [
        (0.0, height - thickness),
        (width / 2.0, 0.0),
        (width / 2.0, thickness),
        (0.0, height),
    ]
    _emit_plate(
        ctx, f"{root_id}.left", "roof_plane", left, depth, (0.0, 0.0, 0.0), "x",
        material_role, "roof.gable_plane", parent_id=parent_id, extra={"side": "left"},
    )
    _emit_plate(
        ctx, f"{root_id}.right", "roof_plane", right, depth, (0.0, 0.0, 0.0), "x",
        material_role, "roof.gable_plane", parent_id=parent_id, extra={"side": "right"},
    )


def _emit_wedge(
    ctx: _BuildContext,
    root_id: str,
    kind: str,
    position: Sequence[float],
    dimensions: Sequence[float],
    orientation: str,
    material_role: str,
    role: str,
    *,
    parent_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = _solid_parameters(root_id, extra)
    payload["orientation"] = orientation
    if _box_is_human_scale(dimensions):
        ctx.add(
            root_id, kind, WEDGE, position, dimensions, material_role, role,
            parameters=payload, parent_id=parent_id,
        )
        return
    cx, cy, cz = (float(value) for value in position)
    dx, dy, dz = (float(value) for value in dimensions)
    bottom = cy - dy / 2.0
    if orientation in ("+X", "-X"):
        half = dx / 2.0
        peak_x = half if orientation == "+X" else -half
        triangle = [(-half, bottom), (half, bottom), (peak_x, bottom + dy)]
        _emit_plate(
            ctx, root_id, kind, triangle, dz, (cx, 0.0, cz), "x",
            material_role, role, parent_id=parent_id, extra=payload,
        )
        return
    if orientation not in ("+Z", "-Z"):
        raise ValueError(f"unsupported wedge orientation '{orientation}'")
    half = dz / 2.0
    peak = half if orientation == "+Z" else -half
    triangle = [(-half, bottom), (half, bottom), (peak, bottom + dy)]
    _emit_plate(
        ctx, root_id, kind, triangle, dx, (cx, 0.0, cz), "z",
        material_role, role, parent_id=parent_id, extra=payload,
    )


def _add_wall_run(
    ctx: _BuildContext,
    *,
    root_id: str,
    length: float,
    height: float,
    thickness: float,
    axis: str,
    base_y: float = 0.0,
    cross: float = 0.0,
    along_offset: float = 0.0,
    openings: Any = "none",
    role: str = "wall",
    parent_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if axis not in ("x", "z"):
        raise ValueError("wall axis must be 'x' or 'z'")
    specs = _opening_specs(openings, length, height, thickness, player=ctx.player)
    stones = iter_ashlar(
        length=length, height=height, thickness=thickness, axis=axis,
        base_y=base_y, cross=cross, along_offset=along_offset,
        openings=specs, detail=ctx.detail,
    )
    _emit_masonry(
        ctx, root_id, "wall_segment", stones, "primary", f"{role}.ashlar",
        parent_id=parent_id, extra={"segment_host": root_id},
        role_map={"sill": f"{role}.sill", "lintel": f"{role}.header",
                  "ashlar": f"{role}.ashlar", "jamb": f"{role}.ashlar"},
    )
    if ctx.player is not None:
        for spec in specs:
            if spec["kind"] in ("door", "arch") and spec["sill"] <= EPSILON:
                ctx.ports.append({
                    "id": f"{root_id}.{spec['id']}",
                    "kind": spec["kind"],
                    "source": "declared",
                    "host": root_id,
                    "position": [round(v, 6) for v in _axis_position(axis, along_offset + spec["center"], base_y, cross)],
                    "wall_axis": axis,
                    "normal_axis": "z" if axis == "x" else "x",
                    "width": round(spec["width"], 6),
                    "height": round(spec["height"], 6),
                })
    for index, spec in enumerate(specs):
        if spec["kind"] == "arch" and ctx.detail != "concept":
            arch_depth = thickness * (1.08 if ctx.detail == "presentation" else 1.02)
            _emit_arch(
                ctx, f"{root_id}.arch_trim.{index:03d}",
                _axis_position(axis, along_offset + spec["center"], base_y + spec["height"] / 2.0, cross),
                _axis_dimensions(axis, spec["width"] * 1.06, spec["height"], arch_depth),
                axis, "accent", f"{role}.arch_surround",
                parent_id=parent_id or root_id,
                extra={"axis": axis, "opening_id": spec["id"]},
                kind="opening_surround",
            )
    return specs


def _build_wall(ctx: _BuildContext) -> None:
    p = ctx.params
    _add_wall_run(
        ctx, root_id="wall.000", length=p["width"], height=p["height"], thickness=p["depth"],
        axis=p["variant"], openings=p.get("openings", "none"), role="envelope.wall",
    )


def _build_slab(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "slab.000"
    material = "structure" if p["variant"] == "foundation" else "primary"
    position = (0.0, p["height"] / 2.0, 0.0)
    dimensions = (p["width"], p["height"], p["depth"])
    _emit_solid(ctx, root, "slab", position, dimensions, material, f"horizontal.{p['variant']}")
    if ctx.detail == "presentation":
        edge_h = min(p["height"] * 0.28, 0.12)
        _emit_solid(
            ctx, f"{root}.edge.front", "slab_edge",
            (0.0, p["height"] + edge_h / 2.0, p["depth"] / 2.0 - edge_h / 2.0),
            (p["width"], edge_h, edge_h), "accent", "horizontal.edge", parent_id=root,
        )
        _emit_solid(
            ctx, f"{root}.edge.back", "slab_edge",
            (0.0, p["height"] + edge_h / 2.0, -p["depth"] / 2.0 + edge_h / 2.0),
            (p["width"], edge_h, edge_h), "accent", "horizontal.edge", parent_id=root,
        )


def _add_column(
    ctx: _BuildContext,
    root: str,
    x: float,
    z: float,
    width: float,
    height: float,
    depth: float,
    variant: str,
    *,
    base_y: float = 0.0,
    parent_id: Optional[str] = None,
) -> None:
    if variant == "square":
        _emit_masonry(
            ctx, root, "column",
            iter_masonry_solid((x, base_y + height / 2.0, z), (width, height, depth), ctx.detail),
            "structure", "structure.column", parent_id=parent_id,
        )
        return

    if variant == "round":
        diameter = min(width, depth)
        sections = {"concept": 12, "standard": 18, "presentation": 28}[ctx.detail]
        for center_y, drum_h, course in iter_stack(base_y, height, ctx.detail):
            ctx.add(
                f"{root}.drum.{course:03d}", "column", CYLINDER,
                (x, center_y, z), (diameter, drum_h, diameter),
                "structure", "structure.column",
                parameters={"radius": diameter / 2.0, "height": drum_h, "sections": sections,
                            "unit": "drum", "course": course, "construction": "ashlar stack", "host_id": root},
                parent_id=parent_id,
            )
        return

    base_h = min(height * 0.10, width * 0.30)
    cap_h = min(height * 0.12, width * 0.36)
    shaft_h = height - base_h - cap_h
    if shaft_h <= EPSILON:
        raise ValueError("classical column is too short for its base and capital")
    _emit_solid(
        ctx, f"{root}.base", "column_base", (x, base_y + base_h / 2.0, z),
        (width * 1.18, base_h, depth * 1.18), "secondary", "structure.column_base",
        parent_id=parent_id or root,
    )
    sections = {"concept": 12, "standard": 20, "presentation": 32}[ctx.detail]
    diameter = min(width, depth) * 0.76
    for center_y, drum_h, course in iter_stack(base_y + base_h, shaft_h, ctx.detail):
        ctx.add(
            f"{root}.shaft.{course:03d}", "column_shaft", CYLINDER,
            (x, center_y, z), (diameter, drum_h, diameter),
            "primary", "structure.column_shaft",
            parameters={"radius": diameter / 2.0, "height": drum_h, "sections": sections,
                        "unit": "drum", "course": course, "construction": "ashlar stack", "host_id": root},
            parent_id=parent_id or root,
        )
    _emit_solid(
        ctx, f"{root}.capital", "column_capital",
        (x, base_y + height - cap_h / 2.0, z), (width * 1.35, cap_h, depth * 1.35),
        "secondary", "structure.column_capital", parent_id=parent_id or root,
    )


def _build_column(ctx: _BuildContext) -> None:
    p = ctx.params
    _add_column(ctx, "column.000", 0.0, 0.0, p["width"], p["height"], p["depth"], p["variant"])


def _build_beam(ctx: _BuildContext) -> None:
    p = ctx.params
    dims = _axis_dimensions(p["variant"], p["width"], p["height"], p["depth"])
    root = "beam.000"
    _emit_solid(
        ctx, root, "beam", (0.0, p["height"] / 2.0, 0.0), dims,
        "structure", "structure.beam", extra={"axis": p["variant"]},
    )
    if ctx.detail == "presentation":
        bearing = max(min(p["width"] * 0.06, 0.5), p["depth"])
        for index, sign in enumerate((-1.0, 1.0)):
            along = sign * (p["width"] / 2.0 - bearing / 2.0)
            _emit_solid(
                ctx, f"{root}.bearing.{index:03d}", "bearing",
                _axis_position(p["variant"], along, p["height"] + p["height"] * 0.12, 0.0),
                _axis_dimensions(p["variant"], bearing, p["height"] * 0.24, p["depth"] * 1.25),
                "secondary", "structure.bearing", parent_id=root,
            )


def _stair_axis(variant: str) -> Tuple[str, float]:
    if variant not in ("+z", "-z", "+x", "-x"):
        raise ValueError("stair/ramp direction must be +z, -z, +x, or -x")
    return variant[-1], 1.0 if variant[0] == "+" else -1.0


def _max_riser(ctx: _BuildContext, default: float) -> float:
    """The tallest riser to build: the architect's default, or less for a player
    who can't step that high."""
    if ctx.player is None:
        return default
    return min(default, ctx.player.step_height)


def _add_stair(
    ctx: _BuildContext,
    root: str,
    width: float,
    rise: float,
    run: float,
    variant: str,
    steps: int,
    *,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    parent_id: Optional[str] = None,
) -> None:
    axis, sign = _stair_axis(variant)
    step_depth = run / steps
    step_rise = rise / steps
    if step_rise > 0.45:
        raise ValueError("stair riser exceeds 0.45 blocks; increase steps")
    for index in range(steps):
        current_h = step_rise * (index + 1)
        along = sign * (-run / 2.0 + step_depth * (index + 0.5))
        x = origin[0] + (along if axis == "x" else 0.0)
        z = origin[2] + (along if axis == "z" else 0.0)
        dims = (step_depth, current_h, width) if axis == "x" else (width, current_h, step_depth)
        _emit_solid(
            ctx, f"{root}.step.{index:03d}", "stair_step",
            (x, origin[1] + current_h / 2.0, z), dims,
            "structure", "circulation.stair_step",
            extra={"step_index": index, "step_count": steps, "riser": step_rise,
                   "tread": step_depth, "direction": variant},
            parent_id=parent_id or root,
        )
    if ctx.detail == "presentation":
        rail_h = min(1.0, max(0.5, width * 0.38))
        rail_t = min(0.10, max(0.05, width * 0.04))
        if axis == "z":
            for label, x in (("left", origin[0] - width / 2.0 + rail_t / 2.0), ("right", origin[0] + width / 2.0 - rail_t / 2.0)):
                _emit_solid(
                    ctx, f"{root}.guard.{label}", "guard",
                    (x, origin[1] + rise / 2.0 + rail_h / 2.0, origin[2]),
                    (rail_t, rise + rail_h, run), "accent", "circulation.guard",
                    parent_id=parent_id or root,
                )
        else:
            for label, z in (("left", origin[2] - width / 2.0 + rail_t / 2.0), ("right", origin[2] + width / 2.0 - rail_t / 2.0)):
                _emit_solid(
                    ctx, f"{root}.guard.{label}", "guard",
                    (origin[0], origin[1] + rise / 2.0 + rail_h / 2.0, z),
                    (run, rise + rail_h, rail_t), "accent", "circulation.guard",
                    parent_id=parent_id or root,
                )


def _build_stair(ctx: _BuildContext) -> None:
    p = ctx.params
    steps = _as_int("steps", p.get("steps", max(2, int(math.ceil(p["height"] / _max_riser(ctx, 0.24))))), maximum=200)
    if ctx.player is not None and "steps" not in ctx.explicit:
        # The catalogue's default step count is for people; a profile adds
        # steps until every riser is one the player can walk up.
        steps = max(steps, int(math.ceil(p["height"] / ctx.player.step_height - 1e-9)))
    _add_stair(ctx, "stair.000", p["width"], p["height"], p["depth"], p["variant"], steps)


def _build_ramp(ctx: _BuildContext) -> None:
    p = ctx.params
    axis, _ = _stair_axis(p["variant"])
    slope = p["height"] / p["depth"]
    max_slope = float(p.get("max_slope", 0.35))
    if slope > max_slope + EPSILON:
        raise ValueError(f"ramp slope {slope:.3f} exceeds max_slope {max_slope:.3f}")
    orientation = {"+x": "+X", "-x": "-X", "+z": "+Z", "-z": "-Z"}[p["variant"]]
    dims = (p["depth"], p["height"], p["width"]) if axis == "x" else (p["width"], p["height"], p["depth"])
    root = "ramp.000"
    _emit_wedge(
        ctx, root, "ramp", (0.0, p["height"] / 2.0, 0.0), dims, orientation,
        "structure", "circulation.ramp", extra={"slope": slope},
    )
    if ctx.detail != "concept":
        curb_t = min(0.12, p["width"] * 0.05)
        curb_h = min(0.18, p["height"] * 0.20)
        if axis == "z":
            for label, x in (("left", -p["width"] / 2.0 + curb_t / 2.0), ("right", p["width"] / 2.0 - curb_t / 2.0)):
                _emit_solid(
                    ctx, f"{root}.curb.{label}", "curb", (x, curb_h / 2.0, 0.0),
                    (curb_t, curb_h, p["depth"]), "accent", "circulation.curb", parent_id=root,
                )
        else:
            for label, z in (("left", -p["width"] / 2.0 + curb_t / 2.0), ("right", p["width"] / 2.0 - curb_t / 2.0)):
                _emit_solid(
                    ctx, f"{root}.curb.{label}", "curb", (0.0, curb_h / 2.0, z),
                    (p["depth"], curb_h, curb_t), "accent", "circulation.curb", parent_id=root,
                )


def _build_roof(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "roof.000"
    if p["variant"] == "flat":
        thickness = min(p["height"], max(0.18, p["height"] * 0.20))
        _emit_solid(
            ctx, root, "roof", (0.0, thickness / 2.0, 0.0),
            (p["width"], thickness, p["depth"]), "roof", "roof.flat",
        )
        if ctx.detail != "concept":
            parapet_h = min(max(0.25, thickness), p["height"])
            parapet_t = min(max(0.12, thickness * 0.60), min(p["width"], p["depth"]) * 0.08)
            for label, position, dims in (
                ("front", (0.0, thickness + parapet_h / 2.0, p["depth"] / 2.0 - parapet_t / 2.0), (p["width"], parapet_h, parapet_t)),
                ("back", (0.0, thickness + parapet_h / 2.0, -p["depth"] / 2.0 + parapet_t / 2.0), (p["width"], parapet_h, parapet_t)),
            ):
                _emit_solid(
                    ctx, f"{root}.parapet.{label}", "parapet", position, dims,
                    "secondary", "roof.parapet", parent_id=root,
                )
    elif p["variant"] == "dome":
        _emit_dome(
            ctx, root, (0.0, p["height"] / 2.0, 0.0),
            (p["width"], p["height"], p["depth"]), "roof", "roof.dome",
        )
    else:
        _emit_gable(ctx, root, p["width"], p["height"], p["depth"], "roof")
        if ctx.detail == "presentation":
            ridge_w = min(max(0.10, p["width"] * 0.025), 0.30)
            _emit_solid(
                ctx, f"{root}.ridge", "ridge",
                (0.0, p["height"] + ridge_w / 2.0, 0.0), (ridge_w, ridge_w, p["depth"]),
                "accent", "roof.ridge", parent_id=root,
            )


def _build_site_pad(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "site-pad.000"
    _emit_solid(
        ctx, root, "site_pad", (0.0, p["height"] / 2.0, 0.0),
        (p["width"], p["height"], p["depth"]), "site", "site.pad",
    )
    if p["variant"] == "terraced" or ctx.detail == "presentation":
        terrace_w = p["width"] * 0.72
        terrace_d = p["depth"] * 0.72
        terrace_h = min(max(0.16, p["height"] * 0.45), p["height"])
        _emit_solid(
            ctx, f"{root}.terrace", "terrace",
            (0.0, p["height"] + terrace_h / 2.0, 0.0), (terrace_w, terrace_h, terrace_d),
            "secondary", "site.terrace", parent_id=root,
        )
        path_w = min(max(1.2, p["width"] * 0.12), p["width"] * 0.35)
        path_h = min(0.12, p["height"] * 0.25)
        path_d = max(EPSILON * 2, (p["depth"] - terrace_d) / 2.0)
        _emit_solid(
            ctx, f"{root}.approach", "walkway",
            (0.0, p["height"] + path_h / 2.0, terrace_d / 2.0 + path_d / 2.0),
            (path_w, path_h, path_d), "accent", "site.approach", parent_id=root,
        )


def _build_platform(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "platform.000"
    deck_t = _as_float("deck_thickness", p.get("deck_thickness", min(0.35, max(0.18, p["height"] * 0.10))))
    deck_y = max(deck_t / 2.0, p["height"] - deck_t / 2.0)
    if p["variant"] == "floating":
        # The platform remains visually unsupported, while a thin site datum
        # makes the standalone architectural product explicitly Y=0 grounded.
        datum_h = min(0.08, max(0.03, deck_t * 0.20))
        _emit_solid(
            ctx, f"{root}.datum", "site_datum", (0.0, datum_h / 2.0, 0.0),
            (p["width"] * 1.12, datum_h, p["depth"] * 1.12),
            "site", "site.platform_datum", parent_id=root,
        )
    _emit_solid(
        ctx, f"{root}.deck", "platform_deck", (0.0, deck_y, 0.0),
        (p["width"], deck_t, p["depth"]), "primary", "circulation.platform", parent_id=root,
    )
    if p["variant"] != "floating":
        support_h = max(EPSILON * 2, deck_y - deck_t / 2.0)
        support_w = min(max(0.25, min(p["width"], p["depth"]) * 0.10), 0.8)
        locations = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if p["variant"] == "supported" else [(-1, -1), (-1, 1)]
        for index, (sx, sz) in enumerate(locations):
            x = sx * (p["width"] / 2.0 - support_w)
            z = sz * (p["depth"] / 2.0 - support_w)
            _emit_solid(
                ctx, f"{root}.support.{index:03d}", "column",
                (x, support_h / 2.0, z), (support_w, support_h, support_w),
                "structure", "structure.platform_support", parent_id=root,
            )
    if ctx.detail == "presentation":
        guard_t = min(0.10, p["depth"] * 0.04)
        guard_h = min(1.0, max(0.5, p["height"] * 0.25))
        for label, z in (("front", p["depth"] / 2.0 - guard_t / 2.0), ("back", -p["depth"] / 2.0 + guard_t / 2.0)):
            _emit_solid(
                ctx, f"{root}.guard.{label}", "guard",
                (0.0, deck_y + deck_t / 2.0 + guard_h / 2.0, z), (p["width"], guard_h, guard_t),
                "accent", "circulation.guard", parent_id=root,
            )


def _build_bridge(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "bridge.000"
    deck_t = _as_float("deck_thickness", p.get("deck_thickness", 0.28))
    support_height = max(deck_t, p["height"] - deck_t)
    _emit_solid(
        ctx, f"{root}.deck", "bridge_deck",
        (0.0, support_height + deck_t / 2.0, 0.0), (p["width"], deck_t, p["depth"]),
        "primary", "circulation.bridge_deck", parent_id=root,
    )
    pier_w = min(max(0.25, p["depth"] * 0.22), 0.65)
    for index, x in enumerate((-p["width"] * 0.32, p["width"] * 0.32)):
        _emit_masonry(
            ctx, f"{root}.pier.{index:03d}", "bridge_pier",
            iter_masonry_solid(
                (x, support_height / 2.0, 0.0),
                (pier_w, support_height, p["depth"] * 0.72), ctx.detail,
            ),
            "structure", "structure.bridge_pier", parent_id=root,
        )
    add_rails = p["variant"] in ("railed", "truss") or ctx.detail == "presentation"
    if add_rails:
        rail_t = min(0.10, p["depth"] * 0.05)
        rail_h = min(1.1, max(0.55, p["height"] * 0.25))
        for label, z in (("front", p["depth"] / 2.0 - rail_t / 2.0), ("back", -p["depth"] / 2.0 + rail_t / 2.0)):
            _emit_solid(
                ctx, f"{root}.rail.{label}", "bridge_rail",
                (0.0, support_height + deck_t + rail_h / 2.0, z), (p["width"], rail_h, rail_t),
                "accent", "circulation.bridge_rail", parent_id=root,
            )
    if p["variant"] == "truss":
        post_count = max(3, min(9, int(round(p["width"] / 2.0))))
        spacing = p["width"] / (post_count - 1)
        for index in range(post_count):
            x = -p["width"] / 2.0 + spacing * index
            _emit_solid(
                ctx, f"{root}.truss_post.{index:03d}", "truss_post",
                (x, support_height + deck_t + 0.75, 0.0), (rail_t, 1.5, p["depth"]),
                "structure", "structure.bridge_truss", parent_id=root,
            )


def _build_room_shell(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "room-shell.000"
    slab_t = min(0.25, max(0.12, p["height"] * 0.05))
    wall_t = min(max(0.18, min(p["width"], p["depth"]) * 0.035), 0.45)
    _emit_solid(
        ctx, f"{root}.floor", "slab", (0.0, slab_t / 2.0, 0.0),
        (p["width"], slab_t, p["depth"]), "structure", "space.floor", parent_id=root,
    )
    base_y = slab_t
    front_mode = p.get("openings", "mixed")
    _add_wall_run(ctx, root_id=f"{root}.wall.front", length=p["width"], height=p["height"], thickness=wall_t,
                  axis="x", base_y=base_y, cross=p["depth"] / 2.0 - wall_t / 2.0, openings=front_mode,
                  role="space.boundary.front", parent_id=root)
    _add_wall_run(ctx, root_id=f"{root}.wall.back", length=p["width"], height=p["height"], thickness=wall_t,
                  axis="x", base_y=base_y, cross=-p["depth"] / 2.0 + wall_t / 2.0,
                  openings="window" if front_mode != "none" else "none", role="space.boundary.back", parent_id=root)
    _add_wall_run(ctx, root_id=f"{root}.wall.left", length=p["depth"] - 2.0 * wall_t, height=p["height"], thickness=wall_t,
                  axis="z", base_y=base_y, cross=-p["width"] / 2.0 + wall_t / 2.0, openings="none",
                  role="space.boundary.left", parent_id=root)
    _add_wall_run(ctx, root_id=f"{root}.wall.right", length=p["depth"] - 2.0 * wall_t, height=p["height"], thickness=wall_t,
                  axis="z", base_y=base_y, cross=p["width"] / 2.0 - wall_t / 2.0, openings="none",
                  role="space.boundary.right", parent_id=root)
    if p["variant"] == "pavilion" or ctx.detail == "presentation":
        roof_t = min(0.28, wall_t)
        _emit_solid(
            ctx, f"{root}.canopy", "roof",
            (0.0, slab_t + p["height"] + roof_t / 2.0, 0.0),
            (p["width"] + wall_t * 2.0, roof_t, p["depth"] + wall_t * 2.0),
            "roof", "roof.canopy", parent_id=root,
        )


def _build_arcade(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "arcade.000"
    bays = p["bays"]
    pier_w = min(max(0.28, p["width"] / (bays * 8.0)), p["width"] / (bays * 3.0))
    bay_span = (p["width"] - pier_w * (bays + 1)) / bays
    if bay_span <= pier_w:
        raise ValueError("arcade width is too small for the requested bay count")
    base_h = min(0.35, p["height"] * 0.07)
    arch_h = p["height"] - base_h - (0.25 if ctx.detail != "concept" else 0.10)
    for index in range(bays + 1):
        x = -p["width"] / 2.0 + pier_w / 2.0 + index * (bay_span + pier_w)
        _emit_masonry(
            ctx, f"{root}.pier.{index:03d}", "pier",
            iter_masonry_solid((x, base_h + arch_h / 2.0, 0.0), (pier_w, arch_h, p["depth"]), ctx.detail),
            "structure", "facade.arcade_pier", parent_id=root, extra={"bay_index": index},
        )
    for index in range(bays):
        x = -p["width"] / 2.0 + pier_w + bay_span / 2.0 + index * (bay_span + pier_w)
        _emit_arch(
            ctx, f"{root}.arch.{index:03d}",
            (x, base_h + arch_h / 2.0, 0.0), (bay_span, arch_h, p["depth"]),
            "x", "primary", "facade.arcade_arch",
            parent_id=root, extra={"bay_index": index},
        )
    _emit_solid(
        ctx, f"{root}.base", "plinth", (0.0, base_h / 2.0, 0.0),
        (p["width"], base_h, p["depth"] * 1.08), "secondary", "facade.arcade_base",
        parent_id=root,
    )
    if ctx.detail != "concept":
        cornice_h = min(0.35, p["height"] * 0.07)
        _emit_solid(
            ctx, f"{root}.cornice", "cornice",
            (0.0, base_h + arch_h + cornice_h / 2.0, 0.0),
            (p["width"] * 1.03, cornice_h, p["depth"] * 1.18), "accent", "facade.cornice",
            parent_id=root,
        )


def _grid_coordinates(span: float, bays: int) -> List[float]:
    return [_rounded(-span / 2.0 + span * index / bays) for index in range(bays + 1)]


def _build_structural_grid(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "structural-grid.000"
    bays_x = _as_int("bays_x", p.get("bays_x", p["bays"]), maximum=16)
    bays_z = _as_int("bays_z", p.get("bays_z", max(1, p["bays"] - 1)), maximum=16)
    levels = p["levels"]
    xs = _grid_coordinates(p["width"], bays_x)
    zs = _grid_coordinates(p["depth"], bays_z)
    level_h = p["height"]
    column_w = min(max(0.24, min(p["width"] / bays_x, p["depth"] / bays_z) * 0.10), 0.8)
    footing_w = column_w * (1.8 if p["variant"] == "heavy" else 1.5)
    footing_h = min(0.35, level_h * 0.08)
    beam_h = min(max(0.25, level_h * 0.10), 0.65)
    beam_d = column_w * 0.72

    for xi, x in enumerate(xs):
        for zi, z in enumerate(zs):
            _emit_solid(
                ctx, f"{root}.footing.{xi:02d}.{zi:02d}", "foundation",
                (x, footing_h / 2.0, z), (footing_w, footing_h, footing_w),
                "secondary", "structure.foundation", parent_id=root,
            )
            total_column_h = level_h * levels
            _emit_masonry(
                ctx, f"{root}.column.{xi:02d}.{zi:02d}", "column",
                iter_masonry_solid(
                    (x, footing_h + total_column_h / 2.0, z),
                    (column_w, total_column_h, column_w), ctx.detail,
                ),
                "structure", "structure.column", parent_id=root,
                extra={"grid_x": xi, "grid_z": zi},
            )

    x_span = p["width"] / bays_x
    z_span = p["depth"] / bays_z
    for level in range(1, levels + 1):
        beam_y = footing_h + level * level_h - beam_h / 2.0
        for zi, z in enumerate(zs):
            for bay in range(bays_x):
                x = (xs[bay] + xs[bay + 1]) / 2.0
                _emit_solid(
                    ctx, f"{root}.beam_x.{level:02d}.{zi:02d}.{bay:02d}", "beam",
                    (x, beam_y, z), (x_span, beam_h, beam_d),
                    "structure", "structure.beam_x", extra={"level": level}, parent_id=root,
                )
        for xi, x in enumerate(xs):
            for bay in range(bays_z):
                z = (zs[bay] + zs[bay + 1]) / 2.0
                _emit_solid(
                    ctx, f"{root}.beam_z.{level:02d}.{xi:02d}.{bay:02d}", "beam",
                    (x, beam_y, z), (beam_d, beam_h, z_span),
                    "structure", "structure.beam_z", extra={"level": level}, parent_id=root,
                )
        if p["variant"] == "slab-frame" or ctx.detail == "presentation":
            slab_h = min(0.22, beam_h * 0.55)
            _emit_solid(
                ctx, f"{root}.slab.{level:02d}", "slab",
                (0.0, footing_h + level * level_h + slab_h / 2.0, 0.0),
                (p["width"] + column_w, slab_h, p["depth"] + column_w),
                "primary", "structure.floor_slab", extra={"level": level}, parent_id=root,
            )


def _facade_openings(mode: Any, bays: int, width: float, height: float) -> List[Dict[str, Any]]:
    if not isinstance(mode, str):
        return [dict(item) for item in mode]
    if mode == "none":
        return []
    bay_width = width / bays
    result: List[Dict[str, Any]] = []
    for index in range(bays):
        center = -width / 2.0 + bay_width * (index + 0.5)
        kind = "window"
        sill = height * 0.26
        opening_h = height * 0.44
        if mode in ("door", "arch") or (mode == "mixed" and index == bays // 2):
            kind = "arch" if mode == "arch" else "door"
            sill = 0.0
            opening_h = height * 0.72
        result.append({
            "id": f"{kind}.{index:03d}", "kind": kind, "center": center,
            "width": bay_width * 0.56, "height": opening_h, "sill": sill,
        })
    return result


def _build_facade(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "facade.000"
    openings = _facade_openings(p.get("openings", "mixed"), p["bays"], p["width"], p["height"])
    _add_wall_run(ctx, root_id=f"{root}.wall", length=p["width"], height=p["height"], thickness=p["depth"],
                  axis="x", openings=openings, role="facade.wall", parent_id=root)
    if ctx.detail != "concept" or p["variant"] == "monumental":
        bay_w = p["width"] / p["bays"]
        pilaster_w = min(max(0.16, bay_w * 0.10), 0.45)
        pilaster_d = p["depth"] * 1.12
        for index in range(p["bays"] + 1):
            x = -p["width"] / 2.0 + bay_w * index
            _emit_masonry(
                ctx, f"{root}.pilaster.{index:03d}", "pilaster",
                iter_masonry_solid(
                    (x, p["height"] / 2.0, p["depth"] * 0.04),
                    (pilaster_w, p["height"], pilaster_d), ctx.detail,
                ),
                "secondary", "facade.pilaster", parent_id=root, extra={"bay_index": index},
            )
        cornice_h = min(0.4, p["height"] * 0.08)
        _emit_solid(
            ctx, f"{root}.cornice", "cornice",
            (0.0, p["height"] + cornice_h / 2.0, 0.0),
            (p["width"] * 1.03, cornice_h, p["depth"] * 1.25), "accent", "facade.cornice",
            parent_id=root,
        )


def _build_spire_cluster(ctx: _BuildContext) -> None:
    p = ctx.params
    root = "spire-cluster.000"
    count = _as_int("spire_count", p.get("spire_count", p.get("bays", 3)), minimum=2, maximum=5)
    pad_h = min(0.35, max(0.15, p["height"] * 0.025))
    _emit_solid(
        ctx, f"{root}.site", "site_pad", (0.0, pad_h / 2.0, 0.0),
        (p["width"], pad_h, p["depth"]), "site", "site.spire_pad", parent_id=root,
    )

    candidates = [
        (-p["width"] * 0.28, -p["depth"] * 0.22),
        (p["width"] * 0.25, -p["depth"] * 0.18),
        (0.0, p["depth"] * 0.25),
        (-p["width"] * 0.25, p["depth"] * 0.20),
        (p["width"] * 0.28, p["depth"] * 0.24),
    ]
    ctx.rng.shuffle(candidates)
    positions = sorted(candidates[:count], key=lambda point: (point[0], point[1]))
    spires: List[Dict[str, float]] = []
    base_span = min(p["width"], p["depth"])
    # One shared datum lets the orthogonal bridge vocabulary make true
    # platform-to-platform connections without inventing unsupported diagonal
    # transforms or leaving vertical gaps at the higher endpoint.
    shared_platform_y = pad_h + p["height"] * ctx.rng.uniform(0.42, 0.50)
    for index, (x, z) in enumerate(positions):
        width = base_span * ctx.rng.uniform(0.10, 0.16)
        depth = width * ctx.rng.uniform(0.82, 1.15)
        height_scale = ctx.rng.uniform(0.58, 0.96)
        if p["variant"] == "needle":
            width *= 0.72
            depth *= 0.72
            height_scale = min(1.0, height_scale * 1.15)
        height = max(width * 2.5, p["height"] * height_scale)
        _emit_solid(
            ctx, f"{root}.spire.{index:03d}", "spire",
            (x, pad_h + height / 2.0, z), (width, height, depth),
            "primary", "spire.tower", extra={"spire_index": index}, parent_id=root,
        )
        cap_h = min(width * 0.9, height * 0.18)
        cap_variant = "dome" if (ctx.style == "classical" and index % 2 == 0) else "wedge"
        cap_dims = (width * 1.08, cap_h, depth * 1.08)
        cap_pos = (x, pad_h + height + cap_h / 2.0, z)
        if cap_variant == "dome":
            if _box_is_human_scale(cap_dims):
                ctx.add(
                    f"{root}.spire.{index:03d}.cap", "spire_cap", DOME, cap_pos, cap_dims,
                    "roof", "spire.dome_cap",
                    parameters=_solid_parameters(f"{root}.spire.{index:03d}.cap",
                                                {"layers": 8, "sections": 20, "spire_index": index}),
                    parent_id=f"{root}.spire.{index:03d}",
                )
            else:
                _emit_dome(
                    ctx, f"{root}.spire.{index:03d}.cap", cap_pos, cap_dims,
                    "roof", "spire.dome_cap", parent_id=f"{root}.spire.{index:03d}",
                    extra={"spire_index": index}, kind="spire_cap",
                )
        else:
            _emit_wedge(
                ctx, f"{root}.spire.{index:03d}.cap", "spire_cap", cap_pos, cap_dims,
                "+X" if index % 2 == 0 else "-X", "roof", "spire.roof_cap",
                parent_id=f"{root}.spire.{index:03d}", extra={"spire_index": index},
            )
        platform_y = shared_platform_y
        platform_t = min(0.25, width * 0.14)
        _emit_solid(
            ctx, f"{root}.platform.{index:03d}", "platform",
            (x, platform_y, z), (width * 1.55, platform_t, depth * 1.55),
            "secondary", "circulation.spire_platform", extra={"spire_index": index}, parent_id=root,
        )
        spires.append({"x": x, "z": z, "width": width, "depth": depth, "height": height, "platform_y": platform_y})

    # A deterministic nearest-neighbour chain guarantees a connected cluster.
    remaining = list(range(1, len(spires)))
    chain = [0]
    while remaining:
        current = chain[-1]
        next_index = min(
            remaining,
            key=lambda idx: (
                (spires[idx]["x"] - spires[current]["x"]) ** 2 + (spires[idx]["z"] - spires[current]["z"]) ** 2,
                idx,
            ),
        )
        remaining.remove(next_index)
        chain.append(next_index)

    for edge_index, (a_idx, b_idx) in enumerate(zip(chain[:-1], chain[1:])):
        a = spires[a_idx]
        b = spires[b_idx]
        dx = b["x"] - a["x"]
        dz = b["z"] - a["z"]
        bridge_y = a["platform_y"]
        deck_t = min(0.22, base_span * 0.015)
        deck_width = min(a["width"], a["depth"], b["width"], b["depth"]) * 0.48
        route_segments: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]] = []
        if abs(dx) > EPSILON:
            route_segments.append((
                "x",
                ((a["x"] + b["x"]) / 2.0, bridge_y, a["z"]),
                (abs(dx) + deck_width, deck_t, deck_width),
            ))
        if abs(dz) > EPSILON:
            route_segments.append((
                "z",
                (b["x"], bridge_y, (a["z"] + b["z"]) / 2.0),
                (deck_width, deck_t, abs(dz) + deck_width),
            ))
        if not route_segments:
            raise ValueError("spire bridge endpoints must be distinct")

        for segment_index, (axis, center, dims) in enumerate(route_segments):
            _emit_solid(
                ctx, f"{root}.bridge.{edge_index:03d}.{axis}",
                "bridge", center, dims, "accent", "circulation.spire_bridge",
                extra={
                    "from_spire": a_idx,
                    "to_spire": b_idx,
                    "from_platform_id": f"{root}.platform.{a_idx:03d}",
                    "to_platform_id": f"{root}.platform.{b_idx:03d}",
                    "axis": axis,
                    "route_index": edge_index,
                    "route_segment": segment_index,
                    "route_segment_count": len(route_segments),
                },
                parent_id=root,
            )

    first = spires[chain[0]]
    stair_rise = first["platform_y"] - pad_h
    stair_run = min(max(stair_rise * 1.8, 3.0), p["depth"] * 0.42)
    steps = max(4, int(math.ceil(stair_rise / _max_riser(ctx, 0.35))))
    if first["z"] >= 0.0:
        stair_direction = "+z"
        stair_center_z = first["z"] - first["depth"] / 2.0 - stair_run / 2.0
    else:
        stair_direction = "-z"
        stair_center_z = first["z"] + first["depth"] / 2.0 + stair_run / 2.0
    _add_stair(ctx, f"{root}.access", min(1.8, first["width"]), stair_rise, stair_run,
               stair_direction, steps, origin=(first["x"], pad_h, stair_center_z), parent_id=root)
    for placement in ctx.placements:
        if placement.id.startswith(f"{root}.access.step."):
            placement.parameters["connects_platform_id"] = f"{root}.platform.{chain[0]:03d}"


_BUILDERS: Dict[str, Callable[[_BuildContext], None]] = {
    "wall": _build_wall,
    "slab": _build_slab,
    "column": _build_column,
    "beam": _build_beam,
    "stair": _build_stair,
    "ramp": _build_ramp,
    "roof": _build_roof,
    "site-pad": _build_site_pad,
    "platform": _build_platform,
    "bridge": _build_bridge,
    "room-shell": _build_room_shell,
    "arcade": _build_arcade,
    "structural-grid": _build_structural_grid,
    "facade": _build_facade,
    "spire-cluster": _build_spire_cluster,
}


def _placement_bounds(placements: Sequence[Placement]) -> Tuple[List[float], List[float]]:
    if not placements:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    bounds_min = [float("inf"), float("inf"), float("inf")]
    bounds_max = [float("-inf"), float("-inf"), float("-inf")]
    for placement in placements:
        for axis in range(3):
            low = placement.position[axis] - placement.dimensions[axis] / 2.0
            high = placement.position[axis] + placement.dimensions[axis] / 2.0
            bounds_min[axis] = min(bounds_min[axis], low)
            bounds_max[axis] = max(bounds_max[axis], high)
    return ([_rounded(value) for value in bounds_min], [_rounded(value) for value in bounds_max])


def _resolve_parent_links(placements: Sequence[Placement]) -> None:
    """Resolve semantic group parents to concrete placement anchors.

    Builders use readable group IDs such as ``room-shell.000`` while they are
    assembling components.  The canonical quality contract requires every
    non-null parent to reference an emitted placement, so the first placement
    beneath each abstract group becomes its stable anchor.  The original group
    remains in parameters for consumers that want the semantic hierarchy.
    """
    known_ids = {placement.id for placement in placements}
    abstract_parents = sorted({
        placement.parent_id
        for placement in placements
        if placement.parent_id is not None and placement.parent_id not in known_ids
    })
    for abstract_parent in abstract_parents:
        assert abstract_parent is not None
        children = [
            placement
            for placement in placements
            if placement.parent_id == abstract_parent
        ]
        if not children:
            continue
        prefix_children = [
            placement
            for placement in children
            if placement.id.startswith(f"{abstract_parent}.")
        ]
        anchor = (prefix_children or children)[0]
        for placement in children:
            placement.parameters = dict(placement.parameters or {})
            placement.parameters.setdefault("semantic_parent_group", abstract_parent)
            placement.parent_id = None if placement.id == anchor.id else anchor.id


def build_document(request: BuildRequest) -> BuildDocument:
    """Build a validated, deterministic semantic document for ``request``.

    The function uses a call-local ``random.Random(seed)`` only.  Invalid
    products, styles, detail levels, variants, dimensions, openings, or
    product-specific constraints raise ``ValueError`` before a document is
    returned.
    """
    if request is None:
        raise ValueError("request is required")
    product = _normalize_product_name(getattr(request, "product", None))
    item = _PRODUCT_BY_NAME[product]

    style = str(getattr(request, "style", "contemporary") or "contemporary").strip().lower()
    if style not in PALETTES:
        raise ValueError(f"style must be one of: {', '.join(PALETTES)}")
    detail = str(getattr(request, "detail", "standard") or "standard").strip().lower()
    if detail not in DETAIL_LEVELS:
        raise ValueError(f"detail must be one of: {', '.join(DETAIL_LEVELS)}")
    seed_value = getattr(request, "seed", 0)
    if isinstance(seed_value, bool):
        raise ValueError("seed must be an integer")
    try:
        seed = int(seed_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be an integer") from exc

    params = _request_parameters(request, item)
    context = _BuildContext(product, style, detail, seed, params)
    player_of = getattr(request, "player_profile", None)
    context.player = player_of() if callable(player_of) else None
    context.clear = None if context.player is None else context.player.clearances()
    context.explicit = set((getattr(request, "parameters", None) or {}).keys())
    _BUILDERS[product](context)
    if not context.placements:
        raise ValueError(f"{product} produced no placements")
    _resolve_parent_links(context.placements)

    bounds_min, bounds_max = _placement_bounds(context.placements)
    architect_record = item["architect"]
    architect = ArchitectInfo(
        id=architect_record["id"],
        version=architect_record["version"],
        title=architect_record["title"],
    )
    metadata = {
        "product": product,
        "category": item["category"],
        "root_id": f"{product}.000",
        "seed": seed,
        "style": style,
        "detail": detail,
        "variant": params["variant"],
        "openings": params.get("openings", "none"),
        "resolved_parameters": deepcopy(params),
        "placement_count": len(context.placements),
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "palette": {key: list(value) for key, value in context.palette.items()},
        "invariants": {
            "min_placements": 1,
            "placement_count": len(context.placements),
            "ground_contact": True,
            "parent_links_valid": True,
        },
        "design_metrics": {
            "semantic_role_count": len({placement.role for placement in context.placements}),
            "kind_count": len({placement.kind for placement in context.placements}),
            "block_type_count": len({placement.block_type for placement in context.placements}),
        },
    }
    if context.player is not None:
        # Only profiled builds carry these, so other documents keep their
        # fingerprints.
        metadata["player"] = {"profile": context.player.name, "clearances": context.clear.to_dict()}
        metadata["ports"] = context.ports
    return BuildDocument(
        architect=architect,
        request=request,
        placements=context.placements,
        metadata=metadata,
        schema_version=SCHEMA_VERSION,
        units="block",
    )


__all__ = ["build_document", "describe_product", "list_products"]
