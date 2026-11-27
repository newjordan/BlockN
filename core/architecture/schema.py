"""Canonical semantic document types for BlockN architecture builds.

The schema deliberately contains no mesh or GUI dependencies.  Architects emit
``BuildDocument`` instances, the compiler turns placements into scene blocks,
and exporters can persist the same canonical representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Tuple


SCHEMA_VERSION = "blockn.architecture/v1"


def canonical_product_name(value: Any) -> str:
    """Return the catalog spelling used to route and audit a product name."""
    return str(value).strip().lower().replace("_", "-")


def _default_coordinate_system() -> Dict[str, Any]:
    return {
        "axes": {"x": "width", "y": "height", "z": "depth"},
        "handedness": "right",
        "precision": 6,
    }


def _canonical_value(value: Any) -> Any:
    """Return a JSON-compatible value with deterministic container ordering."""
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        normalized = [_canonical_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=True),
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Architecture documents cannot contain NaN or infinite values")
        if value == 0.0:
            return 0.0
        return value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _canonical_value(value.item())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Value of type {type(value).__name__} is not JSON-compatible")


# ``age`` is a surface parameter. It changes weathering, not geometry, substances,
# or placement identity, so documents keep the fingerprint they had before the
# control existed.
_FINGERPRINT_EXCLUDED_KEYS = {
    "age",
    "created_at",
    "generated_at",
    "quality",
    "quality_report",
    "timestamp",
    "timestamps",
    "updated_at",
}

# 0 is newly worked; 1 is centuries of weather. The default is a maintained building.
DEFAULT_SURFACE_AGE = 0.35


def _surface_age(value: Any) -> float:
    """Return a surface age in 0..1. Missing values use the default."""
    if value is None:
        return DEFAULT_SURFACE_AGE
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("age must be a number between 0 and 1") from exc
    if isinstance(value, bool) or not math.isfinite(age) or age < 0.0 or age > 1.0:
        raise ValueError("age must be a number between 0 and 1")
    return round(age, 4)


def _fingerprint_payload(value: Any) -> Any:
    """Remove volatile report/timestamp fields from a canonical document."""
    if isinstance(value, Mapping):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key.lower() in _FINGERPRINT_EXCLUDED_KEYS:
                continue
            cleaned[normalized_key] = _fingerprint_payload(item)
        return cleaned
    if isinstance(value, list):
        return [_fingerprint_payload(item) for item in value]
    return value


def _material_channels(values: Any) -> Tuple[int, int, int, int]:
    try:
        material = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "Placement material must be four integer RGBA channels in the range 0..255"
        ) from exc
    if (
        len(material) != 4
        or any(isinstance(value, bool) or not isinstance(value, Integral) for value in material)
        or any(int(value) < 0 or int(value) > 255 for value in material)
    ):
        raise ValueError("Placement material must be four integer RGBA channels in the range 0..255")
    return tuple(int(value) for value in material)  # type: ignore[return-value]


def _optional_finite_float(name: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _optional_integer(name: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return result


@dataclass
class ArchitectInfo:
    id: str
    version: str
    title: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "version": str(self.version),
            "title": str(self.title),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchitectInfo":
        return cls(
            id=str(data.get("id", "")),
            version=str(data.get("version", "")),
            title=str(data.get("title", "")),
        )


@dataclass
class BuildRequest:
    architect: str = "primitive"
    product: str = "wall"
    seed: int = 0
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None
    style: str = "contemporary"
    detail: str = "standard"
    variant: str = "auto"
    bays: Optional[int] = None
    levels: Optional[int] = None
    openings: str = "auto"
    parameters: Dict[str, Any] = field(default_factory=dict)
    age: float = DEFAULT_SURFACE_AGE
    # The player the build is made for: a preset name, a profile object, or
    # None. See ``profile.py``. Serialized only when set, so requests and
    # documents made before profiles existed keep their exact fingerprints.
    profile: Optional[Any] = None

    def player_profile(self):
        """The resolved ``PlayerProfile``, or None when the request has none."""
        from .profile import resolve

        return resolve(self.profile)

    def normalized_dict(self) -> Dict[str, Any]:
        """Return the stable, JSON-compatible request representation."""
        seed = _optional_integer("seed", self.seed)
        if seed is None:
            seed = 0
        payload = {
            "architect": str(self.architect).strip().lower(),
            "product": canonical_product_name(self.product),
            "seed": seed,
            "width": _optional_finite_float("width", self.width),
            "height": _optional_finite_float("height", self.height),
            "depth": _optional_finite_float("depth", self.depth),
            "style": str(self.style).strip().lower(),
            "detail": str(self.detail).strip().lower(),
            "variant": str(self.variant).strip().lower(),
            "bays": _optional_integer("bays", self.bays),
            "levels": _optional_integer("levels", self.levels),
            "openings": str(self.openings).strip().lower(),
            "parameters": _canonical_value(self.parameters),
            "age": _surface_age(self.age),
        }
        player = self.player_profile()
        if player is not None:
            payload["profile"] = player.to_dict()
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return self.normalized_dict()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BuildRequest":
        return cls(
            architect=str(data.get("architect", "primitive")),
            product=str(data.get("product", "wall")),
            seed=_optional_integer("seed", data.get("seed", 0)) or 0,
            width=_optional_finite_float("width", data.get("width")),
            height=_optional_finite_float("height", data.get("height")),
            depth=_optional_finite_float("depth", data.get("depth")),
            style=str(data.get("style", "contemporary")),
            detail=str(data.get("detail", "standard")),
            variant=str(data.get("variant", "auto")),
            bays=_optional_integer("bays", data.get("bays")),
            levels=_optional_integer("levels", data.get("levels")),
            openings=str(data.get("openings", "auto")),
            parameters=dict(data.get("parameters") or {}),
            age=_surface_age(data.get("age", DEFAULT_SURFACE_AGE)),
            profile=_profile_value(data.get("profile")),
        )


def _profile_value(value: Any) -> Optional[Any]:
    if value is None:
        return None
    from .profile import resolve

    return resolve(value).to_dict()


@dataclass
class Placement:
    id: str
    kind: str
    block_type: str
    position: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    material: Tuple[int, int, int, int]
    role: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    substance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        material = _material_channels(self.material)
        payload = {
            "id": str(self.id),
            "kind": str(self.kind),
            "block_type": str(self.block_type),
            "position": [float(value) for value in self.position],
            "dimensions": [float(value) for value in self.dimensions],
            "material": list(material),
            "role": str(self.role),
            "parameters": _canonical_value(self.parameters),
            "parent_id": None if self.parent_id is None else str(self.parent_id),
        }
        # The substance is serialized only when set, so documents authored
        # before it existed keep their exact canonical form and fingerprints.
        if self.substance is not None:
            payload["substance"] = str(self.substance)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Placement":
        position = tuple(float(value) for value in data.get("position", ()))
        dimensions = tuple(float(value) for value in data.get("dimensions", ()))
        material = _material_channels(data.get("material", ()))
        substance = data.get("substance")
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "")),
            block_type=str(data.get("block_type", "")),
            position=position,  # type: ignore[arg-type]
            dimensions=dimensions,  # type: ignore[arg-type]
            material=material,  # type: ignore[arg-type]
            role=str(data.get("role", "")),
            parameters=dict(data.get("parameters") or {}),
            parent_id=None if data.get("parent_id") is None else str(data["parent_id"]),
            substance=None if substance is None else str(substance),
        )


@dataclass
class BuildDocument:
    architect: ArchitectInfo
    request: BuildRequest
    placements: List[Placement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    units: str = "block"
    coordinate_system: Dict[str, Any] = field(default_factory=_default_coordinate_system)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported architecture schema_version '{self.schema_version}'; "
                f"expected '{SCHEMA_VERSION}'"
            )
        if isinstance(self.architect, Mapping):
            self.architect = ArchitectInfo.from_dict(self.architect)
        if isinstance(self.request, Mapping):
            self.request = BuildRequest.from_dict(self.request)
        self.placements = [
            Placement.from_dict(item) if isinstance(item, Mapping) else item
            for item in self.placements
        ]
        self.metadata = dict(self.metadata or {})
        self.coordinate_system = dict(self.coordinate_system or _default_coordinate_system())
        # Validate free-form payloads at the schema boundary even when a raw
        # document does not carry an exported fingerprint.
        self.request.normalized_dict()
        _canonical_value(self.metadata)
        _canonical_value(self.coordinate_system)
        for placement in self.placements:
            _canonical_value(placement.parameters)

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical architecture document representation."""
        return {
            "schema_version": str(self.schema_version),
            "units": str(self.units),
            "coordinate_system": _canonical_value(self.coordinate_system),
            "architect": self.architect.to_dict(),
            "request": self.request.normalized_dict(),
            "placements": [placement.to_dict() for placement in self.placements],
            "metadata": _canonical_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BuildDocument":
        document = cls(
            architect=ArchitectInfo.from_dict(data.get("architect") or {}),
            request=BuildRequest.from_dict(data.get("request") or {}),
            placements=[
                Placement.from_dict(item)
                for item in (data.get("placements") or [])
            ],
            metadata=dict(data.get("metadata") or {}),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            units=str(data.get("units", "block")),
            coordinate_system=dict(data.get("coordinate_system") or _default_coordinate_system()),
        )
        expected_fingerprint = data.get("fingerprint")
        if expected_fingerprint is not None and str(expected_fingerprint) != document.fingerprint():
            raise ValueError("Architecture document fingerprint does not match its contents")
        return document

    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 content identity for build geometry.

        Volatile quality results and timestamps are intentionally excluded so
        auditing or exporting the same build does not change its identity.
        """
        payload = _fingerprint_payload(self.to_dict())
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_SURFACE_AGE",
    "canonical_product_name",
    "ArchitectInfo",
    "BuildRequest",
    "Placement",
    "BuildDocument",
]
