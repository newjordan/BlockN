"""Masonry laid on one set of course lines: the shared base of the coursed architects.

An architect whose walls are banded, like a Byzantine church, builds on this base so that a band
of brick or a moulding carries round a corner without a step. Every wall of the building is cut
on the same course lines. A light is cut from the courses it crosses and closed with a round
arch, or a horseshoe arch; a curved wall or a dome is laid in radial courses; and any part may be
composed square to the axes and then turned into place.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

from core.architectural_geometry import mesh_parameters
from .historic import Composer, TAU, Voids, box_mesh, clamp, hull, oriented_profile, sector_mesh
from .masonry import MAX_STRETCHER, course_target, stretcher_target


def arc_x(radius, rise):
    """Half-width of a circle ``rise`` above its centre; nothing above the crown."""
    return math.sqrt(max(0.0, radius * radius - rise * rise)) if rise < radius else 0.0

def clean(polygon, eps=1e-6):
    """Drop coincident corners from a stone's outline; an outline of no area is nothing."""
    points = []
    for point in polygon:
        if not points or math.dist(point, points[-1]) > eps:
            points.append(point)
    while len(points) > 1 and math.dist(points[0], points[-1]) <= eps:
        points.pop()
    if len(points) < 3:
        return []
    twice = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))
    return points if abs(twice) > 6e-3 else []

def clip(polygon, inside, meet):
    """Cut a convex outline where ``inside`` changes, meeting the edge along the way."""
    out = []
    for p, q in zip(polygon, polygon[1:] + polygon[:1]):
        if inside(p):
            out.append(p)
        if inside(p) != inside(q):
            out.append(meet(p, q))
    return out

def solid(points):
    """The convex hull of a brick's corners, or None where they do not enclose a volume."""
    try:
        mesh = hull(points)
    except Exception:                      # a flat set of corners has no hull
        return None
    return mesh if len(mesh.vertices) >= 4 and mesh.is_watertight and mesh.volume > 1e-5 else None

def extent(a0, a1, inner, outer):
    """Plan extent of an annular sector, over the angles its extreme points can take."""
    angles = [a0, a1] + [q for q in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2) if a0 < q < a1]
    xs = [r * math.cos(q) for q in angles for r in (inner, outer)]
    zs = [r * math.sin(q) for q in angles for r in (inner, outer)]
    return max(max(xs) - min(xs), max(zs) - min(zs))

def angular_cuts(a0, a1, inner, outer, limit=1.45):
    """Angles that divide an annular sector into stones whose envelope stays within a stretcher."""
    if extent(a0, a1, inner, outer) <= limit or a1 - a0 < .02:
        return [a0, a1]
    mid = (a0 + a1) / 2
    return angular_cuts(a0, mid, inner, outer, limit)[:-1] + angular_cuts(mid, a1, inner, outer, limit)


class Lights(Voids):
    """Openings of one wall run. ``arch`` closes a light with a round arch, or a horseshoe arch
    whose centre stands ``lift`` above its springing line, so that the arch is widest above its feet."""

    def __init__(self, base):
        super().__init__(base)
        self.lifts = {}

    def arch(self, center, half, sill, spring, ring, opening_id, role=None, lift=0.0):
        foot = half * math.cos(math.asin(lift / half)) if lift else half
        self.specs.append({"id": opening_id, "center": center, "width": 2 * foot,
                           "sill": sill - self.base, "height": spring - sill})
        self.specs.append({"id": opening_id + ".head", "center": center, "width": 2 * (half + ring),
                           "sill": spring - self.base, "height": lift + half + ring})
        self.arches.append((center, half, spring, ring, opening_id, role))
        self.lifts[opening_id] = lift


class Coursed(Composer):
    """A composer that cuts every wall on one set of course lines and dyes the courses in bands.

    A subclass sets ``y_f`` (its floor level), ``pitch`` and ``stretch``, and ``T`` where it
    builds round walls, before it builds. ``BANDED`` names the roles whose courses take the
    bands; ``brick_courses`` and ``stone_courses`` give their rhythm.
    """
    BANDED = frozenset()
    brick_courses, stone_courses = 4, 2
    BRICK, BRICK_DARK = (174, 106, 78, 255), (152, 90, 66, 255)          # fired clay, two tones
    ASHLAR, ASHLAR_DARK = (210, 200, 180, 255), (194, 184, 164, 255)    # dressed limestone, two tones

    def lines(self, y0, y1):
        """Edges of the courses between two heights, on lines the whole church shares, and the first course's index."""
        first = math.floor((y0 - self.y_f) / self.pitch + 1e-9) + 1
        marks = [self.y_f + k * self.pitch for k in range(first, first + int((y1 - y0) / self.pitch) + 2)]
        return [y0, *[y for y in marks if y0 + 1e-5 < y < y1 - 1e-5], y1], first - 1

    def line(self, y, up=False):
        """The course line at or below a height, or at or above it: an opening starts and stops on a bed joint."""
        k = math.ceil((y - self.y_f) / self.pitch - 1e-9) if up else math.floor((y - self.y_f) / self.pitch + 1e-9)
        return self.y_f + k * self.pitch

    @staticmethod
    def whole(pieces, least=.07):
        """Close a sliver stone into its neighbour that keeps the same beds, so never across a light.

        ``pieces`` are (start, end, beds): a stone with no beds is an opening, and is dropped.
        """
        out = []
        for piece in (piece for piece in pieces if piece[2]):
            if (out and out[-1][2] == piece[2] and abs(out[-1][1] - piece[0]) < 1e-9
                    and min(piece[1] - piece[0], out[-1][1] - out[-1][0]) < least):
                out[-1] = (out[-1][0], piece[1], out[-1][2])
            else:
                out.append(piece)
        return [piece for piece in out if piece[1] - piece[0] > .02]

    def wall(self, role, axis, along, cross, length, y0, top, voids=None, **params):
        """A wall of the full thickness, every stone set on the course lines the whole church shares.

        Alternate courses break joint by half a stone. A light is cut from the courses it
        crosses, so its sill and the lintel over it need not fall on a bed joint, and a round
        arch of voussoirs closes its head.
        """
        host = self.host(role)
        edges, first = self.lines(y0, top)
        count = max(1, round(length / self.stretch))
        module, origin = length / count, along - length / 2
        # A light's sill is stored from its own wall's foot, which need not be the building's floor.
        openings = [(along + s["center"] - s["width"] / 2, along + s["center"] + s["width"] / 2,
                     voids.base + s["sill"], voids.base + s["sill"] + s["height"]) for s in (voids.specs if voids else ())]
        for course, (ya, yb) in enumerate(zip(edges, edges[1:])):
            shift = .5 if (first + course) % 2 else 0.0
            joints = [origin + module * (k + shift) for k in range(0 if shift else 1, count + 1)]
            live = [(lo, hi, sill, head) for lo, hi, sill, head in openings if sill < yb - 1e-6 and head > ya + 1e-6]
            jambs = [round(edge, 6) for lo, hi, _, _ in live for edge in (lo, hi)]
            marks = sorted({origin, origin + length, *[j for j in joints if origin < j < origin + length],
                            *[edge for lo, hi, _, _ in live for edge in (lo, hi)]})
            pieces = []
            for p, q in zip(marks, marks[1:]):
                beds = [(ya, yb)]
                for lo, hi, sill, head in live:
                    if lo - 1e-6 <= p and q <= hi + 1e-6:     # the stone lies in the light: keep only what the light leaves
                        beds = [(a, b) for a, b in ((ya, min(yb, sill)), (max(ya, head), yb)) if b - a > .03]
                pieces.append((p, q, tuple(beds)))
            for index, (p, q, beds) in enumerate(self.whole(pieces)):
                if beds != ((ya, yb),):
                    unit = "sill" if beds[0][0] == ya else "lintel"
                else:
                    unit = "jamb" if round(p, 6) in jambs or round(q, 6) in jambs else "ashlar"
                for y_lo, y_hi in beds:
                    middle = (y_lo + y_hi) / 2
                    if axis == "x":
                        position, dims = ((p + q) / 2, middle, cross), (q - p, y_hi - y_lo, self.T)
                    else:
                        position, dims = (cross, middle, (p + q) / 2), (self.T, y_hi - y_lo, q - p)
                    self.add(role, position, dims, construction="ashlar running bond", unit=unit,
                             course=first + course, stone_index=index, host_id=host, axis=axis, **params)
        for centre, half, spring, ring, opening_id, arch_role in (voids.arches if voids else ()):
            self.arch(arch_role, axis, along + centre, cross, half, spring, ring, opening_id,
                      lift=getattr(voids, "lifts", {}).get(opening_id, 0.0))

    def arch(self, role, axis, along, cross, half, spring, ring, opening_id, thick=None, lift=0.0):
        """The arch of brick voussoirs that closes a light, with the spandrels over it cut in courses.

        A round arch springs from its widest points. A horseshoe arch has its centre ``lift``
        above its springing line, so it is widest above its feet, which stand ``lift`` apart
        inside them; its ring is cut off at the springing line, and its spandrels are filled
        both over the ring and, below the centre, between the ring and the edge of the light.
        """
        outer, host, thick = half + ring, self.host(role), thick or self.T
        tilt = math.asin(lift / half) if lift else 0.0
        centre, start, span = spring + lift, -tilt, math.pi + 2 * tilt
        count = max(7, int(math.ceil(span * half / .3)) | 1)
        for index in range(count):
            q0, qm, q1 = (start + span * (index + step) / count for step in (0, .5, 1))
            polygon = [(outer * math.cos(q), centre + outer * math.sin(q)) for q in (q0, qm, q1)]
            polygon += [(half * math.cos(q), centre + half * math.sin(q)) for q in (q1, qm, q0)]
            if lift:                                   # the ring stands on the springing line, not below it
                polygon = clip(polygon, lambda p: p[1] >= spring - 1e-9,
                               lambda p, q: (p[0] + (q[0] - p[0]) * (spring - p[1]) / (q[1] - p[1]), spring))
            profile = self.plate(polygon, thick, axis, along, cross)
            if profile is not None:
                self.mesh(role, profile, unit="voussoir", construction="ashlar voussoir arch", host_id=host,
                          course=index, keystone=index == count // 2, opening_id=opening_id)
        for side in (-1, 1):
            lo, hi = sorted((0.0, side * outer))
            self.coursed(role, axis, along, cross, thick, lo, hi, centre, centre + outer,
                         lower=lambda u: centre + arc_x(outer, u), opening_id=opening_id, register="spandrel")
            if lift:
                reach = math.sqrt(outer * outer - lift * lift)   # where the ring's extrados meets the springing line
                lo, hi = sorted((side * reach, side * outer))
                self.coursed(role, axis, along, cross, thick, lo, hi, spring, centre,
                             lower=lambda u: spring, upper=lambda u: centre - arc_x(outer, u),
                             opening_id=opening_id, register="spandrel")

    def band_for_course(self, course):
        return "brick" if course % (self.brick_courses + self.stone_courses) < self.brick_courses else "stone"

    def bands(self, start):
        """Dye the masonry in layers, four courses of brick to two or three of stone, as Constantinopolitan walls were."""
        for p in self.placements[start:]:
            unit = p.parameters.get("unit")
            if p.role not in self.BANDED or p.kind in self.TIMBER_KINDS or unit in ("core", "voussoir", "drum"):
                continue
            course = p.parameters.get("course")
            if course is None or p.parameters.get("register") == "spandrel":
                course = round((p.position[1] - self.y_f) / self.pitch)   # a spandrel takes the band it sits in
            band = self.band_for_course(course)
            brick = band == "brick"
            p.material = (self.BRICK, self.BRICK_DARK)[course % 2] if brick else (self.ASHLAR, self.ASHLAR_DARK)[course % 2]
            p.substance = "fired-clay-brick" if brick else "limestone"
            if brick and unit in ("ashlar", "sill", "lintel", "jamb"):
                p.parameters["unit"] = "brick"
            p.parameters["band"] = band

    def coursed(self, role, axis, along, cross, thickness, u0, u1, ya, yb, lower=None, upper=None, **params):
        """Fill a band between two boundaries with horizontal courses of cut stones.

        ``lower`` and ``upper`` give the height of the band's foot and head at any point along it,
        so the same courses close a gable under a roof or the spandrel over an arch. Long courses
        are cut into stones no longer than a stretcher and alternate courses break joint.
        """
        foot, head = lower or (lambda u: ya), upper or (lambda u: yb)
        edges, first = self.lines(ya, yb)
        host = self.host(role)
        count = max(1, round((u1 - u0) / self.stretch))
        for course, (y0, y1) in enumerate(zip(edges, edges[1:])):
            shift = .5 if (first + course) % 2 else 0.0
            marks = [u0 + (u1 - u0) * (k + shift) / count for k in range(count + 1)]
            cuts = sorted({u0, u1, *[u for u in marks if u0 < u < u1]})
            for ua, ub in zip(cuts, cuts[1:]):
                self.trapezoid(role, axis, along, cross, thickness, ua, ub, y0, y1, foot, head, host,
                               first + course, **params)
        return host

    def trapezoid(self, role, axis, along, cross, thickness, ua, ub, y0, y1, foot, head, host, course, **params):
        """One course between two cuts, following a curved boundary closely enough to leave no gap."""
        um = (ua + ub) / 2
        if ub - ua > .12 and any(abs(edge(um) - (edge(ua) + edge(ub)) / 2) > .025 for edge in (foot, head)):
            self.trapezoid(role, axis, along, cross, thickness, ua, um, y0, y1, foot, head, host, course, **params)
            self.trapezoid(role, axis, along, cross, thickness, um, ub, y0, y1, foot, head, host, course, **params)
            return
        polygon = clean([(ua, clamp(foot(ua), y0, y1)), (ub, clamp(foot(ub), y0, y1)),
                          (ub, clamp(head(ub), y0, y1)), (ua, clamp(head(ua), y0, y1))])
        if polygon:
            self.stone(role, oriented_profile(polygon, thickness, axis, along, cross), host, course, **params)

    def plate(self, polygon, depth, axis, along, cross):
        """A cleaned outline cut from a course and set through a wall, or None where nothing is left."""
        outline = clean(polygon)
        return oriented_profile(outline, depth, axis, along, cross) if outline else None

    def turn_by(self, start, angle, cx=0.0, cz=0.0):
        """Turn everything placed since ``start`` through any angle about a vertical axis.

        A part is composed square to the axes and then set on the circle: a light in a drum.
        """
        c, s = math.cos(angle), math.sin(angle)
        for p in self.placements[start:]:
            if p.block_type == "MESH":
                vertices = np.asarray(p.parameters["vertices"], dtype=float) * p.dimensions + p.position
                faces = p.parameters["faces"]
            else:
                mesh = box_mesh(p.position, p.dimensions)
                vertices, faces = np.asarray(mesh.vertices), mesh.faces
            x, z = vertices[:, 0] - cx, vertices[:, 2] - cz
            vertices = np.column_stack((cx + x * c - z * s, vertices[:, 1], cz + x * s + z * c))
            p.position, p.dimensions, data = mesh_parameters(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
            p.block_type = "MESH"
            p.parameters.pop("axis", None)
            if p.parameters.get("unit") in ("ashlar", "sill", "lintel", "jamb"):
                p.parameters["unit"] = "profile_course"
            p.parameters.update(data)

    def curved(self, role, centre, radius, y0, y1, lights=(), arc=(0.0, TAU), faces=0, thick=None, **params):
        """A round wall of radial brick courses, a drum or an apse, pierced by arched lights."""
        thick = thick or self.T
        middle = radius - thick / 2
        gaps = []
        for theta, sill, half, spring, ring, opening_id in lights:
            gaps.append((theta, (half + ring) / middle, sill, spring + half + ring))
            self.head(theta, centre, radius, sill, half, spring, ring, opening_id, thick)
        self.radial(role, centre, radius - thick, radius, y0, y1, gaps, arc, faces, **params)

    def radial(self, role, centre, inner, outer, y0, y1, gaps=(), arc=(0.0, TAU), faces=0, **params):
        """A round wall of radial courses between two radii, with arched lights cut through it.

        With ``faces`` the courses are cut at the corners of a polygon. Stones stay within a
        stretcher measured across their envelope, and alternate courses of a closed ring break joint.
        """
        cx, cz = centre
        edges, first = self.lines(y0, y1)
        host = self.host(role)
        start, end = arc
        closed = end - start >= TAU - 1e-9
        step = TAU / faces if faces else 0.0
        turn = 0.0 if faces else .5 * self.stretch / max(inner, .1)
        for course, (ya, yb) in enumerate(zip(edges, edges[1:])):
            base = start + (turn if closed and course % 2 else 0.0)
            spans = ([(start + (k + .5) * step, start + (k + 1.5) * step) for k in range(faces)] if faces
                     else [(base, base + end - start)])     # a face is centred on an axis
            if any(g2 <= ya + 1e-6 and g3 >= yb - 1e-6 for _, _, g2, g3 in gaps):
                for angle, half, _, _ in gaps:
                    for shift in ((-TAU, 0.0, TAU) if closed else (0.0,)):
                        lo, hi = angle + shift - half, angle + shift + half
                        spans = [piece for a, b in spans for piece in ((a, min(b, lo)), (max(a, hi), b))
                                 if piece[1] - piece[0] > 1e-9]
            for a, b in spans:
                cuts = angular_cuts(a, b, inner, outer)
                for c0, c1 in zip(cuts, cuts[1:]):
                    self.mesh(role, sector_mesh(cx, cz, inner, outer, c0, c1, ya, yb), unit="radial_ashlar",
                              course=first + course, host_id=host, **{"construction": "ashlar running bond", **params})
        return host

    def head(self, theta, centre, radius, sill, half, spring, ring, opening_id, thick):
        """The brick arch over a light in a round wall: voussoirs and spandrels, set square and turned into place."""
        cx, cz = centre
        start = len(self.placements)
        self.arch("window_arch", "x", cx, cz - (radius - thick / 2), half, spring, ring, opening_id, thick)
        self.turn_by(start, theta + math.pi / 2, cx, cz)

    def dome(self, role, centre, base, radius, thickness, arc=(0.0, TAU), **params):
        """A hemispherical shell in horizontal courses, each a ring of stones; a half-dome when ``arc`` is open."""
        if not (0 < thickness < MAX_STRETCHER and radius > thickness + .18):
            raise ValueError("dome needs a positive shell thickness and room for its crown opening")
        cx, cz = centre
        limit = math.acos((thickness + .18) / radius)
        courses = max(2, round(radius * math.sin(limit) / (.8 * self.pitch)))
        span, host = arc[1] - arc[0], self.host(role)
        # The top of a tapered sector reaches further inward than its bed.
        # Bound the whole plan footprint regardless of where the sector turns.

        while True:
            latitudes = [limit * k / courses for k in range(courses + 1)]
            radii = [radius * math.cos(p) for p in latitudes]
            if all(r0 - r1 + thickness < MAX_STRETCHER for r0, r1 in zip(radii, radii[1:])):
                break
            courses *= 2
        # For a sector of sweep θ, the longest plan diagonal is either the
        # outer chord or the line from the outer bed to the inner top edge.
        angles = []
        for outer, top in zip(radii, radii[1:]):
            inner = top - thickness
            cross = math.sqrt((MAX_STRETCHER**2 - (outer-inner)**2) / (4*outer*inner))
            angles.append(2 * math.asin(min(1., cross, MAX_STRETCHER / (2*outer))))
        count = max(12, round(span * radius / self.stretch), math.ceil(span / min(angles)))
        cuts = [arc[0] + span * i / count for i in range(count + 1)]  # align the joints up the shell
        for k, (r0, r1) in enumerate(zip(radii, radii[1:])):
            y0, y1 = (base + radius * math.sin(p) for p in latitudes[k:k+2])
            for a0, a1 in zip(cuts, cuts[1:]):
                self.mesh(role, sector_mesh(cx, cz, r0 - thickness, r0, a0, a1, y0, y1, r1 - thickness, r1),
                          unit="radial_ashlar", course=k, host_id=host, **{"construction": "ashlar running bond", **params})
        return base + radius * math.sin(limit), radius * math.cos(limit)
