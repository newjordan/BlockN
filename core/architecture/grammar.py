"""Transferable geometry predicates for architectural rules.

These checks use axis-aligned member envelopes in the Y-up build document.
For boxes they measure the solid exactly; rotated/concave members need a
different predicate. Contact is not a calculation of structural capacity.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

from .schema import Placement

TOL = 1e-5


@dataclass(frozen=True)
class Bounds:
    low: tuple[float, float, float]
    high: tuple[float, float, float]

    @classmethod
    def of(cls, member: Placement) -> Bounds:
        return cls(tuple(x - d / 2 for x, d in zip(member.position, member.dimensions)),
                   tuple(x + d / 2 for x, d in zip(member.position, member.dimensions)))

    def overlap(self, other: Bounds, axis: int) -> float:
        return max(0., min(self.high[axis], other.high[axis]) - max(self.low[axis], other.low[axis]))

    def span(self, axis: int) -> float:
        return self.high[axis] - self.low[axis]

    def intersects(self, other: Bounds) -> bool:
        return all(self.overlap(other, axis) > TOL for axis in range(3))


def obstructing_members(void: Bounds, members: Iterable[Placement]) -> set[str]:
    """Boundary contact is allowed; positive-volume intrusion is not."""
    return {p.id for p in members if void.intersects(Bounds.of(p))}


@dataclass(frozen=True)
class Bearing:
    """An upper block seats on one lower block, with contact on its top face."""
    upper: str
    lower: str
    minimum_fraction: float = .65

    def holds(self, members: Mapping[str, Placement]) -> bool:
        if self.upper not in members or self.lower not in members:
            return False
        upper, lower = (Bounds.of(members[k]) for k in (self.upper, self.lower))
        return (abs(upper.low[1] - lower.high[1]) <= TOL
                and all(upper.overlap(lower, axis) > TOL
                        and upper.overlap(lower, axis) + TOL >= self.minimum_fraction
                        * min(upper.span(axis), lower.span(axis)) for axis in (0, 2)))


@dataclass(frozen=True)
class Datum:
    """Compare a run of labeled members to explicit positions on one axis.

    Repetition can be even, alternating or stepped: it is the relation, not a
    fixed gap, that transfers. Scale the tolerance from the program, in metres.
    """
    axis: int
    stations: tuple[tuple[str, float], ...]
    tolerance: float = .02

    def departures(self, members: Mapping[str, Placement]) -> set[str]:
        return {slot for slot, at in self.stations if slot not in members or not
                math.isclose(members[slot].position[self.axis], at, rel_tol=0, abs_tol=self.tolerance)}


@dataclass(frozen=True)
class WalkSurface:
    """A rectangular patch of a member's actual top face, not an abstract graph edge."""

    member: str
    plan: tuple[tuple[float, float], tuple[float, float]]  # near/far corners in XZ
    elevation: float
    zone: str = "walk"

    def clearance(self, headroom: float) -> Bounds:
        (x0, z0), (x1, z1) = self.plan
        return Bounds((x0, self.elevation, z0), (x1, self.elevation + headroom, z1))

    def traversable(self, solid: Bounds | None) -> bool:
        floor = self.clearance(0.)
        return (solid is not None and abs(solid.high[1] - self.elevation) <= TOL
                and all(solid.low[a] <= floor.low[a] + TOL
                        and solid.high[a] >= floor.high[a] - TOL for a in (0, 2)))

    def edge_width(self, other: WalkSurface) -> float:
        a, b = self.clearance(0.), other.clearance(0.)
        return max((a.overlap(b, along) for cross, along in ((0, 2), (2, 0))
                    if min(abs(a.high[cross] - b.low[cross]), abs(b.high[cross] - a.low[cross])) <= TOL),
                   default=0.)


@dataclass(frozen=True)
class WalkingRoute:
    surfaces: tuple[WalkSurface, ...]
    width: float = 1.2
    headroom: float = 2.1
    max_riser: float = .18
    min_going: float = .25

    def departures(self, members: Mapping[str, Bounds]) -> set[str]:
        """Diagnose missing floors, discontinuities and solids in the clear walking volume."""
        bad = set()
        for surface in self.surfaces:
            floor = surface.clearance(self.headroom)
            short = min(floor.span(0), floor.span(2))
            long = max(floor.span(0), floor.span(2))
            if (not surface.traversable(members.get(surface.member)) or long + TOL < self.width
                    or short <= TOL or (surface.zone == "stair" and short + TOL < self.min_going)):
                bad.add(surface.member)
            bad.update(member for member, solid in members.items()
                       if member != surface.member and floor.intersects(solid))
        for a, b in zip(self.surfaces, self.surfaces[1:]):
            max_riser = self.max_riser if "stair" in (a.zone, b.zone) else 0.
            if a.edge_width(b) + TOL < self.width or not -TOL <= b.elevation - a.elevation <= max_riser + TOL:
                bad.update((a.member, b.member))
        return bad
