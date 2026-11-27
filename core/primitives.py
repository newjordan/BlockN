"""Authoritative registry for BlockN's render-level mesh primitives.

Architects should emit semantic architecture placements.  The compiler lowers
those placements to this small, stable vocabulary of render primitives.
Keeping dispatch here prevents block creation, scene insertion, and loading
from drifting apart as new mesh types are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping

import numpy as np
import trimesh

from .geometry import (
    create_arch_mesh,
    create_block_mesh,
    create_cylinder_mesh,
    create_dome_mesh,
    create_wedge_mesh,
)
from .architectural_geometry import _normalized_mesh

CUBE = "CUBE"
WEDGE = "WEDGE"
CYLINDER = "CYLINDER"
ARCH = "ARCH"
DOME = "DOME"
MESH = "MESH"

MeshFactory = Callable[[np.ndarray, np.ndarray, Mapping[str, Any]], trimesh.Trimesh]


@dataclass(frozen=True)
class PrimitiveSpec:
    """Description and factory for one internal mesh primitive."""

    name: str
    description: str
    factory: MeshFactory
    parameter_defaults: Mapping[str, Any]
    tags: tuple[str, ...] = ()

    def create(
        self,
        position: np.ndarray,
        dimensions: np.ndarray,
        parameters: Mapping[str, Any] | None = None,
    ) -> trimesh.Trimesh:
        position_array = np.asarray(position, dtype=float)
        dimensions_array = np.asarray(dimensions, dtype=float)
        if position_array.shape != (3,):
            raise ValueError(f"{self.name} position must contain exactly three values")
        if dimensions_array.shape != (3,):
            raise ValueError(f"{self.name} dimensions must contain exactly three values")
        if not np.all(np.isfinite(position_array)):
            raise ValueError(f"{self.name} position must contain only finite values")
        if not np.all(np.isfinite(dimensions_array)) or np.any(dimensions_array <= 0):
            raise ValueError(f"{self.name} dimensions must be finite and positive")

        resolved_parameters: Dict[str, Any] = dict(self.parameter_defaults)
        resolved_parameters.update(dict(parameters or {}))
        mesh = self.factory(position_array, dimensions_array, resolved_parameters)
        if mesh is None or mesh.is_empty:
            raise ValueError(f"{self.name} factory returned an empty mesh")
        return mesh


def _cube_factory(position: np.ndarray, dimensions: np.ndarray, _: Mapping[str, Any]) -> trimesh.Trimesh:
    return create_block_mesh(position, dimensions=dimensions)


def _wedge_factory(
    position: np.ndarray,
    dimensions: np.ndarray,
    parameters: Mapping[str, Any],
) -> trimesh.Trimesh:
    return create_wedge_mesh(
        position,
        dimensions=dimensions,
        orientation=str(parameters.get("orientation", "+X")),
    )


def _cylinder_factory(
    position: np.ndarray,
    dimensions: np.ndarray,
    parameters: Mapping[str, Any],
) -> trimesh.Trimesh:
    radius = float(parameters.get("radius", dimensions[0] / 2.0))
    height = float(parameters.get("height", dimensions[1]))
    if radius <= 0 or height <= 0:
        raise ValueError("CYLINDER radius and height must be positive")
    return create_cylinder_mesh(
        position,
        radius=radius,
        height=height,
        parameters=dict(parameters),
    )


def _arch_factory(
    position: np.ndarray,
    dimensions: np.ndarray,
    parameters: Mapping[str, Any],
) -> trimesh.Trimesh:
    return create_arch_mesh(position, dimensions=dimensions, parameters=dict(parameters))


def _dome_factory(
    position: np.ndarray,
    dimensions: np.ndarray,
    parameters: Mapping[str, Any],
) -> trimesh.Trimesh:
    return create_dome_mesh(position, dimensions=dimensions, parameters=dict(parameters))


PRIMITIVE_REGISTRY: Dict[str, PrimitiveSpec] = {
    CUBE: PrimitiveSpec(
        CUBE,
        "Axis-aligned rectangular solid",
        _cube_factory,
        {},
        ("solid", "orthogonal"),
    ),
    WEDGE: PrimitiveSpec(
        WEDGE,
        "Axis-aligned triangular prism",
        _wedge_factory,
        {"orientation": "+X"},
        ("solid", "slope"),
    ),
    CYLINDER: PrimitiveSpec(
        CYLINDER,
        "Y-axis cylinder",
        _cylinder_factory,
        {"sections": 16},
        ("solid", "round"),
    ),
    ARCH: PrimitiveSpec(
        ARCH,
        "Segmented arch assembly normalized to a requested envelope",
        _arch_factory,
        {"segments": 14},
        ("architectural", "opening"),
    ),
    DOME: PrimitiveSpec(
        DOME,
        "Stepped hemispherical cap normalized to a requested envelope",
        _dome_factory,
        {"layers": 12, "sections": 24},
        ("architectural", "roof"),
    ),
    MESH: PrimitiveSpec(
        MESH,
        "Explicit triangular architectural mesh normalized to a placement envelope",
        _normalized_mesh,
        {},
        ("architectural", "sculpted"),
    ),
}


def get_primitive_spec(block_type: str) -> PrimitiveSpec:
    """Return a registered primitive or raise an actionable error."""

    try:
        return PRIMITIVE_REGISTRY[str(block_type).upper()]
    except KeyError as exc:
        choices = ", ".join(PRIMITIVE_REGISTRY)
        raise ValueError(f"Unknown block type '{block_type}'. Available: {choices}") from exc


def create_primitive_mesh(
    block_type: str,
    position: np.ndarray,
    dimensions: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> trimesh.Trimesh:
    """Compile one registry primitive to a mesh."""

    return get_primitive_spec(block_type).create(position, dimensions, parameters)


def normalize_primitive_dimensions(
    block_type: str,
    dimensions: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Resolve type-specific dimensions before persistence and support checks."""

    resolved = np.asarray(dimensions, dtype=float).copy()
    params = parameters or {}
    if str(block_type).upper() == CYLINDER:
        radius = float(params.get("radius", resolved[0] / 2.0))
        height = float(params.get("height", resolved[1]))
        resolved = np.array([radius * 2.0, height, radius * 2.0], dtype=float)
    return resolved


def primitive_names() -> tuple[str, ...]:
    return tuple(PRIMITIVE_REGISTRY)


def primitive_catalog() -> Iterable[dict[str, Any]]:
    """Return JSON-ready registry metadata for discovery commands."""

    for spec in PRIMITIVE_REGISTRY.values():
        yield {
            "name": spec.name,
            "description": spec.description,
            "parameter_defaults": dict(spec.parameter_defaults),
            "tags": list(spec.tags),
        }
