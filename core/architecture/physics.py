"""Physical properties for compiled builds and a first vertical-load check.

The quality contract states plainly that its geometry checks are not structural
engineering.  With real substances assigned to every member we can do better
than that disclaimer without pretending to be a solver: after compilation this
module measures each member from its real mesh (volume, centroid), gives it a
mass from its substance density, and accumulates vertical load downwards
through contact.  The result is a *reported* diagnostic:

* a member carries its own weight plus the share of every member resting on
  it, shared by the overlap of their ground-plane footprints;
* bearing stress at a member's base is that load divided by its contact area;
* utilisation compares that stress against the substance's compressive
  strength through a factor of safety (10 for masonry-like materials, 4 for
  timber and metal).

Known limits, deliberately: only vertical gravity load is considered; contact
is measured on axis-aligned footprints, so a tapered or curved member's real
bearing may differ from the estimate, and members whose detected contact
covers less than a quarter of their own bed are reported as unresolved and
left out of the ranking and the accumulation rather than made to carry load
through a sliver; thrust, wind, bending, tension, connections and soil
capacity are not evaluated at all.

Geometry is authored in metres, so one block unit is treated as one metre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .substances import SUBSTANCES, Substance


GRAVITY = 9.80665  # m/s2, standard gravity
# Members are authored to stack exactly; legacy column stacks carry an
# intentional 1e-5 gap, so the contact tolerance must exceed it.
CONTACT_TOLERANCE = 1e-3  # m
MIN_CONTACT_AREA = 1e-6  # m2
GROUND_TOLERANCE = 1e-6  # m, matches the build.ground_contact gate

# Bearing resolution.  A contact smaller than this fraction of a member's own
# bed, or thinner than the absolute floor, is not treated as a resolved bed:
# the coarse axis-aligned model cannot see how such a member really bears
# (an edge kiss at a bond joint, a spandrel wedge against a curved arch ring,
# a sloping plate).  Those members keep their mass and report an as-if-bedded
# stress, are counted as unresolved, and drop out of the utilisation ranking
# and of the downward accumulation rather than funnelling load through a
# sliver of nothing.
BEARING_FRACTION = 0.25
MIN_RESOLVED_BEARING = 0.01  # m2

# Factors of safety by substance category.  Brittle masonry-like materials are
# held to 10, as specified for the structural check; timber and ductile metal
# to 4.  Textile is included so banners have a finite, non-binding value.
FACTOR_OF_SAFETY: Dict[str, float] = {
    "stone": 10.0,
    "earth": 10.0,
    "soil": 3.0,
    "binder": 10.0,
    "roofing": 10.0,
    "glass": 10.0,
    "timber": 4.0,
    "metal": 4.0,
    "textile": 4.0,
}
DEFAULT_FACTOR_OF_SAFETY = 10.0


def factor_of_safety(substance: Substance) -> float:
    return FACTOR_OF_SAFETY.get(substance.category, DEFAULT_FACTOR_OF_SAFETY)


@dataclass(frozen=True)
class MemberPhysical:
    """One member's measured physical state, in document order."""

    primitive_id: str
    substance: str
    volume_m3: float
    mass_kg: float
    centroid_m: Tuple[float, float, float]
    supported_load_kg: float      # own mass plus everything resting on it
    contact_area_m2: float       # bearing area at the member's base
    bearing_stress_mpa: float
    utilisation: float
    resolved_bearing: bool       # False when no real bed contact could be found

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primitive_id": str(self.primitive_id),
            "substance": str(self.substance),
            "volume_m3": round(self.volume_m3, 6),
            "mass_kg": round(self.mass_kg, 3),
            "centroid_m": [round(value, 6) for value in self.centroid_m],
            "supported_load_kg": round(self.supported_load_kg, 3),
            "contact_area_m2": round(self.contact_area_m2, 6),
            "bearing_stress_mpa": round(self.bearing_stress_mpa, 6),
            "utilisation": None if math.isnan(self.utilisation) else round(self.utilisation, 6),
            "resolved_bearing": bool(self.resolved_bearing),
        }


@dataclass
class PhysicalReport:
    """Per-member physics plus document aggregates.

    ``members`` follows document order.  Per-member records live here, not on
    the placements, so a build's semantic fingerprint never changes when its
    physical measurements are recomputed.
    """

    members: List[MemberPhysical] = field(default_factory=list)
    total_mass_kg: float = 0.0
    center_of_mass_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    footprint_area_m2: float = 0.0
    ground_bearing_pressure_kpa: float = 0.0
    bill_of_materials: Dict[str, Dict[str, float]] = field(default_factory=dict)
    max_utilisation: float = 0.0
    max_utilisation_member_id: Optional[str] = None
    members_over_utilisation: List[str] = field(default_factory=list)
    unresolved_bearing_count: int = 0
    members_without_substance: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Aggregates only; the per-member table stays out of JSON exports."""
        return {
            "total_mass_kg": round(self.total_mass_kg, 3),
            "center_of_mass_m": [round(value, 3) for value in self.center_of_mass_m],
            "footprint_area_m2": round(self.footprint_area_m2, 3),
            "ground_bearing_pressure_kpa": round(self.ground_bearing_pressure_kpa, 3),
            "bill_of_materials": {
                substance_id: {
                    "volume_m3": round(values["volume_m3"], 3),
                    "mass_tonnes": round(values["mass_kg"] / 1000.0, 3),
                    "members": int(values["members"]),
                }
                for substance_id, values in sorted(self.bill_of_materials.items())
            },
            "max_member_utilisation": round(self.max_utilisation, 6),
            "max_utilisation_member_id": self.max_utilisation_member_id,
            "members_over_utilisation": list(self.members_over_utilisation),
            "members_without_resolved_bearing": int(self.unresolved_bearing_count),
            "members_without_substance": int(self.members_without_substance),
        }


def _scene_records(scene: Any) -> List[Tuple[str, Any]]:
    """(primitive_id, mesh) pairs in compiled block order."""
    records = []
    for index, block in enumerate(getattr(scene, "blocks", ())):
        parameters = getattr(block, "parameters", {}) or {}
        primitive_id = str(parameters.get("primitive_id", f"scene-block-{index}"))
        records.append((primitive_id, getattr(block, "mesh", None)))
    return records


def _mesh_volume(mesh: Any, dimensions: Sequence[float]) -> float:
    """Volume from the real mesh; the extent box is the fallback for primitives."""
    fallback = float(dimensions[0]) * float(dimensions[1]) * float(dimensions[2])
    if mesh is None or bool(getattr(mesh, "is_empty", True)):
        return fallback
    try:
        volume = float(mesh.volume)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(volume) or volume <= 1e-12:
        return fallback
    return volume


def _mesh_centroid(mesh: Any, position: Sequence[float]) -> Tuple[float, float, float]:
    """Volume centroid of the real mesh; the envelope center is the fallback."""
    if mesh is not None and bool(getattr(mesh, "is_watertight", False)):
        try:
            centroid = np.asarray(mesh.center_mass, dtype=float).reshape(3)
        except (TypeError, ValueError):
            centroid = None
        if centroid is not None and bool(np.isfinite(centroid).all()):
            return (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    return (float(position[0]), float(position[1]), float(position[2]))


def _placement_bounds(placement: Any) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(placement.position, dtype=float)
    half = np.asarray(placement.dimensions, dtype=float) / 2.0
    return center - half, center + half


def _rectangle_union_area(rectangles: Sequence[Tuple[float, float, float, float]]) -> float:
    """Area of the union of (x0, z0, x1, z1) rectangles, without overlap bias."""
    if not rectangles:
        return 0.0
    xs = sorted({value for rect in rectangles for value in (rect[0], rect[2])})
    zs = sorted({value for rect in rectangles for value in (rect[1], rect[3])})
    if len(xs) < 2 or len(zs) < 2:
        return 0.0
    x_index = {value: index for index, value in enumerate(xs)}
    z_index = {value: index for index, value in enumerate(zs)}
    covered = np.zeros((len(xs) - 1, len(zs) - 1), dtype=bool)
    for x0, z0, x1, z1 in rectangles:
        if not (x1 > x0 and z1 > z0):
            continue
        i0, i1 = x_index[x0], x_index[x1]
        j0, j1 = z_index[z0], z_index[z1]
        covered[i0:i1, j0:j1] = True
    widths = np.diff(np.asarray(xs, dtype=float))
    depths = np.diff(np.asarray(zs, dtype=float))
    cell = widths[:, None] * depths[None, :]
    return float(cell[covered].sum())


def _resting_overlaps(
    lo: np.ndarray,
    hi: np.ndarray,
) -> List[List[Tuple[int, float]]]:
    """For each member j, the members it rests on: (index, footprint overlap).

    A member rests on another when the base of the upper member sits at the
    top of the lower one within ``CONTACT_TOLERANCE`` and their ground-plane
    footprints overlap.  Candidate pairs are located by binary search over
    sorted member tops, then filtered by footprint overlap.
    """
    count = len(lo)
    supports: List[List[Tuple[int, float]]] = [[] for _ in range(count)]
    if count == 0:
        return supports

    bases = lo[:, 1]
    tops = hi[:, 1]
    order = np.argsort(tops, kind="stable")
    sorted_tops = tops[order]
    window_start = np.searchsorted(sorted_tops, bases - CONTACT_TOLERANCE, side="left")
    window_end = np.searchsorted(sorted_tops, bases + CONTACT_TOLERANCE, side="right")

    for j in range(count):
        start, end = int(window_start[j]), int(window_end[j])
        if start >= end:
            continue
        candidates = order[start:end]
        overlap_x = np.minimum(hi[candidates, 0], hi[j, 0]) - np.maximum(lo[candidates, 0], lo[j, 0])
        overlap_z = np.minimum(hi[candidates, 2], hi[j, 2]) - np.maximum(lo[candidates, 2], lo[j, 2])
        areas = overlap_x * overlap_z
        areas = np.where((overlap_x > 0.0) & (overlap_z > 0.0), areas, 0.0)
        hit = np.flatnonzero(areas > MIN_CONTACT_AREA)
        if hit.size:
            supports[j] = [
                (int(candidates[k]), float(areas[k]))
                for k in hit
                if int(candidates[k]) != j
            ]
    return supports


def _accumulate_loads(
    masses: np.ndarray,
    lo: np.ndarray,
    supports: List[List[Tuple[int, float]]],
    resolved: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return each member's supported load (own mass plus what it carries).

    Members without a resolved bearing do not forward their load: the coarse
    contact model could not find the bed it would flow through, and pushing
    it through an edge sliver would report fiction.
    """
    if resolved is None:
        resolved = np.ones(len(masses), dtype=bool)
    received = np.zeros(len(masses), dtype=float)
    # Highest bases first: everything resting on a member is emitted before it.
    order = sorted(range(len(masses)), key=lambda index: (-float(lo[index, 1]), index))
    for j in order:
        load = float(masses[j]) + float(received[j])
        if not resolved[j] or not supports[j]:
            continue
        total = sum(area for _index, area in supports[j])
        if total <= 0.0:
            continue
        for i, area in supports[j]:
            received[i] += load * (area / total)
    return masses + received


def physical_report(document: Any, scene: Any) -> PhysicalReport:
    """Measure every member of a compiled build and check vertical bearing."""
    placements = list(document.placements)
    records = _scene_records(scene)
    meshes = {primitive_id: mesh for primitive_id, mesh in records}

    count = len(placements)
    report = PhysicalReport()
    if count == 0:
        return report

    lo = np.empty((count, 3), dtype=float)
    hi = np.empty((count, 3), dtype=float)
    masses = np.zeros(count, dtype=float)
    volumes = np.zeros(count, dtype=float)
    centroids = np.zeros((count, 3), dtype=float)
    substance_ids: List[str] = []
    bof: Dict[str, Dict[str, float]] = {}

    for index, placement in enumerate(placements):
        bounds_lo, bounds_hi = _placement_bounds(placement)
        lo[index] = bounds_lo
        hi[index] = bounds_hi
        substance_id = str(getattr(placement, "substance", "") or "")
        substance_ids.append(substance_id)
        substance = SUBSTANCES.get(substance_id)
        density = substance.density if substance is not None else 0.0
        mesh = meshes.get(str(placement.id))
        volume = _mesh_volume(mesh, placement.dimensions)
        volumes[index] = volume
        masses[index] = density * volume
        centroids[index] = _mesh_centroid(mesh, placement.position)
        entry = bof.setdefault(substance_id, {"volume_m3": 0.0, "mass_kg": 0.0, "members": 0})
        entry["volume_m3"] += volume
        entry["mass_kg"] += masses[index]
        entry["members"] += 1

    # Load path: what rests on what, then top-down accumulation.
    supports = _resting_overlaps(lo, hi)
    bed_areas = np.maximum(
        (hi[:, 0] - lo[:, 0]) * (hi[:, 2] - lo[:, 2]), MIN_CONTACT_AREA
    )
    contact_areas = np.zeros(count, dtype=float)
    resolved = np.ones(count, dtype=bool)
    for j in range(count):
        contact = sum(area for _i, area in supports[j])
        if lo[j, 1] <= GROUND_TOLERANCE:
            # Grounded: the member beds on the soil across its own footprint.
            contact = float(bed_areas[j])
        elif contact < max(MIN_RESOLVED_BEARING, BEARING_FRACTION * bed_areas[j]):
            # No real bed could be resolved; see BEARING_FRACTION above.
            contact = float(bed_areas[j])
            resolved[j] = False
        contact_areas[j] = contact
    supported_load = _accumulate_loads(masses, lo, supports, resolved)

    members: List[MemberPhysical] = []
    over_utilisation: List[str] = []
    unresolved_count = 0
    max_utilisation = 0.0
    max_member: Optional[str] = None
    unassessed_count = 0
    for index, placement in enumerate(placements):
        substance_id = substance_ids[index]
        substance = SUBSTANCES.get(substance_id)
        stress_mpa = (float(supported_load[index]) * GRAVITY / contact_areas[index]) / 1.0e6
        if substance is None:
            # Without a substance there is no strength to compare against.
            # document.substances reports the missing or unknown id; here the
            # member is measured but left out of the load check.
            utilisation = float("nan")
        else:
            strength = substance.compressive_strength
            safety = factor_of_safety(substance)
            utilisation = float("inf") if strength <= 0.0 else stress_mpa * safety / strength
        if substance is not None and not math.isfinite(utilisation):
            utilisation = float("inf")
        member = MemberPhysical(
            primitive_id=str(placement.id),
            substance=substance_id,
            volume_m3=float(volumes[index]),
            mass_kg=float(masses[index]),
            centroid_m=(float(centroids[index, 0]), float(centroids[index, 1]), float(centroids[index, 2])),
            supported_load_kg=float(supported_load[index]),
            contact_area_m2=float(contact_areas[index]),
            bearing_stress_mpa=float(stress_mpa),
            utilisation=utilisation,
            resolved_bearing=bool(resolved[index]),
        )
        members.append(member)
        if substance is None:
            unassessed_count += 1
            continue
        if not resolved[index]:
            unresolved_count += 1
            continue
        if utilisation > max_utilisation and math.isfinite(utilisation):
            max_utilisation = utilisation
            max_member = member.primitive_id
        if utilisation > 1.0:
            over_utilisation.append(member.primitive_id)

    report.members = members
    report.bill_of_materials = bof
    report.max_utilisation = max_utilisation
    report.max_utilisation_member_id = max_member
    report.members_over_utilisation = sorted(over_utilisation)
    report.unresolved_bearing_count = unresolved_count
    report.members_without_substance = unassessed_count

    total_mass = float(masses.sum())
    report.total_mass_kg = total_mass
    if total_mass > 0.0:
        weighted = centroids.T @ masses
        report.center_of_mass_m = (
            float(weighted[0] / total_mass),
            float(weighted[1] / total_mass),
            float(weighted[2] / total_mass),
        )
    else:
        report.center_of_mass_m = (0.0, 0.0, 0.0)

    ground_rectangles = [
        (float(lo[index, 0]), float(lo[index, 2]), float(hi[index, 0]), float(hi[index, 2]))
        for index in range(count)
        if lo[index, 1] <= GROUND_TOLERANCE
    ]
    report.footprint_area_m2 = _rectangle_union_area(ground_rectangles)
    if report.footprint_area_m2 > 0.0:
        report.ground_bearing_pressure_kpa = total_mass * GRAVITY / report.footprint_area_m2 / 1000.0
    return report


__all__ = [
    "GRAVITY",
    "FACTOR_OF_SAFETY",
    "CONTACT_TOLERANCE",
    "MemberPhysical",
    "PhysicalReport",
    "factor_of_safety",
    "physical_report",
]
