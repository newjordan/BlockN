"""Geodesic architect: Buckminster Fuller's lattice dome.

A geodesic dome has one rule and applies it everywhere. Take an icosahedron,
divide each edge of each face into the same number of parts (the frequency),
join the division points into small triangles, and push every point out onto
one sphere. Every member is then a straight strut between two points on that
sphere, every opening is a triangle, and a handful of strut lengths builds the
whole shell. The sphere is cut off by a level plane, and every hub left on the
cut edge is carried to the ground.

``single-layer`` is the plain lattice. ``double-layer`` follows the space frame
of the Montreal Biosphere in principle: a second, concentric lattice inside the
first, with web struts between the two.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import math
import sys

import numpy as np

from .historic import clamp, hull
from .schema import ArchitectInfo
from .sonnet import Poem, controls_for, form

ARCHITECT = ArchitectInfo("geodesic", "1.0.0", "Geodesic Dome Architect")
STYLE = "geodesic-lattice"
PRODUCT = "geodesic-dome"
REGION = "International; first built in the United States and Canada"
PERIOD = "Mid-20th century, from Fuller's 1954 patent to Expo 67"
VARIANTS = ("single-layer", "double-layer")
KINDS = {
    "floor_core": "core", "kerb": "foundation", "floor": "slab", "footing": "foundation",
    "hub": "bracket", "strut": "beam", "web_strut": "beam", "leg": "column",
    "glass_panel": "glazing", "skin_panel": "cladding",
    "portal_post": "column", "portal_beam": "beam", "canopy": "roof",
}
REQUIRED = {"floor_core", "kerb", "floor", "footing", "hub", "strut", "leg"}
FAMILIES = {2: 2, 3: 3, 4: 6}          # strut lengths of a class I lattice, by frequency
FORMS = {PRODUCT: form(
    PRODUCT, "Geodesic lattice dome: an icosahedron subdivided onto one sphere, in struts, hubs and triangular panels",
    (40, 28, 40), VARIANTS, REQUIRED,
    lines={"frequency": ("3v", "2v", "4v"), "skin": ("glazed", "open-frame", "alternating"),
           "entry": ("portal", "lifted-skirt"), "base": ("legs", "v-legs")},
    size={"width": (28, 60), "height": (14, 44), "depth": (28, 60)},
    ratios={"depth/width": (1.0, 1.0), "height/width": (.45, .76)},
    region=REGION, period=PERIOD, plan="circular; depth equals width",
    strut_families=dict((f"{v}v", n) for v, n in FAMILIES.items()),
    sphere="the centre stands at or above the floor, so the shell is a hemisphere or more")}
DEFAULT_SUBSTANCE = "wrought-iron"
SUBSTANCES = {"floor_core": "earth-fill", "kerb": "limestone", "floor": "cast-concrete", "footing": "cast-concrete",
              "glass_panel": "window-glass", "skin_panel": "lead-sheet", "canopy": "lead-sheet"}
PLINTH, DECK, PAD = .5, .2, .25
TOL = 1e-4


@lru_cache(maxsize=None)
def lattice(frequency):
    """Unit-sphere nodes, edges and faces of a class I icosahedral lattice with one node at the zenith.

    Each face also says whether its small triangle points with or against its parent face.
    """
    ring = 2 / math.sqrt(5)
    corners = [(0., 1., 0.)]
    corners += [(ring * math.cos(a), ring / 2, ring * math.sin(a))
                for a in (-math.pi / 2 + k * math.tau / 5 for k in range(5))]
    corners += [(ring * math.cos(a), -ring / 2, ring * math.sin(a))
                for a in (-math.pi / 2 + (k + .5) * math.tau / 5 for k in range(5))]
    corners.append((0., -1., 0.))
    corners = np.asarray(corners)
    parents = []
    for k in range(5):
        u0, u1, l0, l1 = 1 + k, 1 + (k + 1) % 5, 6 + k, 6 + (k + 1) % 5
        parents += [(0, u0, u1), (u0, l0, u1), (l0, l1, u1), (11, l1, l0)]
    index, nodes, faces, edges = {}, [], [], set()

    def node(a, b, c, i, j):
        point = (corners[a] * (frequency - i - j) + corners[b] * i + corners[c] * j) / frequency
        point = point / np.linalg.norm(point)
        key = tuple(np.round(point, 6) + 0.)
        if key not in index:
            index[key] = len(nodes)
            nodes.append(point)
        return index[key]

    for a, b, c in parents:
        for i in range(frequency):
            for j in range(frequency - i):
                faces.append((node(a, b, c, i, j), node(a, b, c, i + 1, j), node(a, b, c, i, j + 1), True))
                if i + j < frequency - 1:
                    faces.append((node(a, b, c, i + 1, j), node(a, b, c, i + 1, j + 1), node(a, b, c, i, j + 1), False))
    for a, b, c, _ in faces:
        edges.update((tuple(sorted(pair)) for pair in ((a, b), (b, c), (a, c))))
    return np.asarray(nodes), tuple(sorted(edges)), tuple(faces)


class Dome(Poem):
    PALETTE = {"floor_core": (132, 112, 92, 255), "kerb": (196, 190, 176, 255), "floor": (176, 176, 172, 255),
               "footing": (150, 150, 148, 255), "hub": (58, 62, 68, 255), "strut": (92, 98, 106, 255),
               "web_strut": (120, 126, 134, 255), "leg": (70, 74, 82, 255),
               "glass_panel": (150, 200, 214, 120), "skin_panel": (206, 208, 204, 255),
               "portal_post": (70, 74, 82, 255), "portal_beam": (70, 74, 82, 255), "canopy": (206, 208, 204, 255)}

    def compose(self):
        w, h = self.w, self.h
        self.frequency = int(self.line("frequency")[0])
        double = self.variant == "double-layer"
        self.radius = radius = min(w / 2 - 1.2, h - PLINTH - .45)
        self.gauge = clamp(.006 * radius, .07, .16) * self.vary("gauge", .15)
        self.hub = hub = clamp(2.6 * self.gauge, .2, .42)
        self.radius = radius = min(w / 2 - 1.2, h - PLINTH - hub)
        self.centre = centre = h - hub - radius                 # the zenith hub tops out at h
        skirt = clamp(.05 * radius, .7, 1.4) * self.vary("skirt", .25)
        depth = clamp(.07 * radius, 1.0, 2.2) * self.vary("depth", .12)
        self.plinth()
        units, edges, faces = lattice(self.frequency)
        chord = float(np.mean([np.linalg.norm(units[a] - units[b]) for a, b in edges])) * radius
        world = units * radius + (0., centre, 0.)
        # The door: the skirt is lifted where the entrance axis meets the cut edge.
        low = PLINTH + skirt
        reach = math.sqrt(max(0., radius ** 2 - (low - centre) ** 2))
        door = np.array((0., low, -reach))
        keep = {i for i, point in enumerate(world) if point[1] >= low - 1e-9}
        gone = {i for i in keep if np.linalg.norm(world[i] - door) < .72 * chord and world[i][1] < low + .8 * chord}
        if not gone:
            gone = {min(keep, key=lambda i: (round(float(np.linalg.norm(world[i] - door)), 6), i))}
        keep -= gone
        neighbours = defaultdict(set)
        for a, b in edges:
            neighbours[a].add(b)
            neighbours[b].add(a)
        while True:                                             # nothing hangs by a single strut
            loose = {i for i in keep if len(neighbours[i] & keep) < 2}
            if not loose:
                break
            keep -= loose
        base = sorted(i for i in keep if neighbours[i] - keep)
        edges = [(a, b) for a, b in edges if a in keep and b in keep]
        faces = [face for face in faces if set(face[:3]) <= keep]
        layers = [("outer", radius)] + ([("inner", radius - depth)] if double else [])
        points = {}
        for layer, r in layers:
            points[layer] = {i: units[i] * r + (0., centre, 0.) for i in sorted(keep)}
            for i, point in points[layer].items():
                self.node(point, layer=layer, node=i)
            for a, b in edges:
                self.strut("strut", points[layer][a], points[layer][b], self.gauge, layer=layer,
                           frequency=self.frequency)
            for i in base:
                self.stand(points[layer][i], layer)
        if double:
            for i in sorted(keep):
                self.strut("web_strut", points["outer"][i], points["inner"][i], self.gauge * .8, web="radial")
            for a, b in edges:
                self.strut("web_strut", points["outer"][a], points["inner"][b], self.gauge * .8, web="diagonal")
        skin = points[layers[-1][0]]
        panels = self.skin(skin, faces, door, chord)
        if self.line("entry") == "portal":
            self.portal(reach)
        return self.document({
            "region": REGION, "period": PERIOD, "frequency": self.frequency, "layers": len(layers),
            "sphere_radius": round(radius, 3), "sphere_centre_height": round(centre, 3),
            "sphere_fraction": round((radius + centre - low) / (2 * radius), 3),
            "hubs": len(keep) * len(layers), "base_hubs": len(base) * len(layers),
            "lattice_struts": len(edges) * len(layers), "panels": panels,
            "strut_length_families": FAMILIES[self.frequency], "layer_depth": round(depth, 3) if double else None,
            "construction": "straight struts pinned at hubs; class I icosahedral subdivision"})

    def plinth(self):
        top, r = PLINTH - DECK, self.w / 2
        self.ring("kerb", 0, 0, r, .4, 0, top)
        self.drum("floor_core", 0, 0, 0, top, r - .4, sides=48, construction="rammed fill", unit="core")
        self.drum("floor", 0, 0, top, PLINTH, r, sides=48, construction="cast concrete")

    def node(self, point, **params):
        g = self.hub
        corners = [tuple(point + np.eye(3)[axis] * sign * g) for axis in range(3) for sign in (-1, 1)]
        self.mesh("hub", hull(corners), construction="forged hub", **params)

    def strut(self, role, a, b, gauge, **params):
        """A slim three-sided prism from one hub centre to another."""
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        axis = (b - a) / np.linalg.norm(b - a)
        side = np.cross(axis, (0., 1., 0.) if abs(axis[1]) < .9 else (1., 0., 0.))
        side /= np.linalg.norm(side)
        other = np.cross(axis, side)
        ring = [gauge * (math.cos(t) * side + math.sin(t) * other) for t in (math.pi / 2, 7 * math.pi / 6, 11 * math.pi / 6)]
        self.mesh(role, hull([tuple(end + offset) for end in (a, b) for offset in ring]),
                  construction="pinned strut", **params)

    def stand(self, point, layer):
        """Carry one base hub to the ground: one plumb leg, or two raking ones, each on its own pad."""
        x, y, z = (float(v) for v in point)
        plan = math.hypot(x, z)
        feet = [(x, z)]
        if self.line("base") == "v-legs" and plan > 1e-6:
            spread = min(.45 * (y - PLINTH - PAD), 1.1)
            tx, tz = -z / plan * spread, x / plan * spread
            feet = [(x + tx, z + tz), (x - tx, z - tz)]
        for fx, fz in feet:
            self.member("footing", fx - .3, fx + .3, PLINTH, PLINTH + PAD, fz - .3, fz + .3,
                        construction="cast concrete", layer=layer)
            self.strut("leg", (x, y, z), (fx, PLINTH + PAD, fz), self.gauge * 1.25, layer=layer)

    def skin(self, points, faces, door, chord):
        """Triangular panels set just inside the lattice; none where the door is."""
        style, count = self.line("skin"), 0
        if style == "open-frame":
            return count
        for a, b, c, upright in faces:
            corners = np.array([points[a], points[b], points[c]])
            middle = corners.mean(axis=0)
            if np.linalg.norm(middle - door) < 1.15 * chord and middle[1] < door[1] + 1.2 * chord:
                continue
            normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            normal /= np.linalg.norm(normal)
            if normal @ (middle - (0., self.centre, 0.)) > 0:
                normal = -normal                                 # inward
            inset = middle + (corners - middle) * .9 + normal * (self.gauge + .03)
            role = "skin_panel" if style == "alternating" and upright else "glass_panel"
            self.mesh(role, hull([tuple(p + normal * t) for p in inset for t in (0., .05)]),
                      construction="framed panel")
            count += 1
        return count

    def portal(self, reach):
        """A framed porch standing in the lifted skirt, on the entrance axis."""
        half, head, z0, z1 = 1.5, PLINTH + 3.0, -reach - .9, -reach + .5
        for x in (-half, half):
            for z in (z0 + .12, z1 - .12):
                self.member("portal_post", x - .12, x + .12, PLINTH, head, z - .12, z + .12)
        for x in (-half, half):
            self.member("portal_beam", x - .12, x + .12, head, head + .22, z0, z1, run="side")
        for z in (z0 + .12, z1 - .12):
            self.member("portal_beam", -half + .12, half - .12, head, head + .22, z - .12, z + .12, run="cross")
        self.member("canopy", -half - .3, half + .3, head + .22, head + .3, z0, z1)


Dome.module = sys.modules[__name__]
random_controls = controls_for(sys.modules[__name__])


def build_document(request):
    return Dome(request).compose()


# ---------------------------------------------------------------- rules
def _world(placement):
    vertices = np.asarray(placement.parameters.get("vertices", ()), dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return np.empty((0, 3))
    return vertices * np.asarray(placement.dimensions, dtype=float) + np.asarray(placement.position, dtype=float)


def strut_ends(placement):
    """The two ends of a strut, measured from its mesh: the centres of its end faces."""
    points = _world(placement)
    if len(points) < 4:
        return None
    gaps = np.linalg.norm(points[:, None] - points[None], axis=2)
    i, j = np.unravel_index(np.argmax(gaps), gaps.shape)
    along = (points - points[i]) @ (points[j] - points[i]) / gaps[i, j] ** 2
    return points[along < .5].mean(axis=0), points[along >= .5].mean(axis=0)


def fit_sphere(points):
    """Least-squares centre and radius of the sphere through a set of points."""
    points = np.asarray(points, dtype=float)
    solution, *_ = np.linalg.lstsq(np.c_[2 * points, np.ones(len(points))], (points ** 2).sum(axis=1), rcond=None)
    centre = solution[:3]
    return centre, math.sqrt(max(0., solution[3] + centre @ centre))


def _families(lengths):
    lengths = sorted(lengths)
    return 1 + sum(b - a > .001 * lengths[-1] for a, b in zip(lengths, lengths[1:])) if lengths else 0


def _open_side(hub, centre, others, limit=100.):
    """True where the struts round a hub leave a gap: the hub is on the cut edge, not inside it."""
    out = hub - centre
    out = out / max(np.linalg.norm(out), 1e-9)
    east = np.cross(out, (0., 1., 0.) if abs(out[1]) < .9 else (1., 0., 0.))
    east /= np.linalg.norm(east)
    north = np.cross(out, east)
    bearings = sorted(math.atan2(float((p - hub) @ north), float((p - hub) @ east)) for p in others)
    if len(bearings) < 3:
        return True
    gaps = [b - a for a, b in zip(bearings, bearings[1:])] + [bearings[0] + math.tau - bearings[-1]]
    return max(gaps) > math.radians(limit)


def rule_checks(document, form):
    placements = document.placements
    hubs = [p for p in placements if p.role == "hub"]
    centres = np.array([p.position for p in hubs], dtype=float).reshape(-1, 3)
    layers = defaultdict(list)
    for i, p in enumerate(hubs):
        layers[p.parameters.get("layer", "outer")].append(i)

    # 1. One sphere for each layer, and the layers concentric.
    off, spheres = [], {}
    for layer, members in layers.items():
        if len(members) < 8:
            off += [hubs[i].id for i in members]
            continue
        centre, radius = spheres[layer] = fit_sphere(centres[members])
        off += [hubs[i].id for i in members
                if abs(np.linalg.norm(centres[i] - centre) - radius) > .004 * radius]
    origin = spheres.get("outer", (np.zeros(3), 0.))[0]
    concentric = all(np.linalg.norm(centre - origin) < .01 * radius for centre, radius in spheres.values())
    yield ("spherical", bool(spheres) and not off and concentric, "Every hub of a layer lies on one sphere.",
           "A hub is off its sphere, or the layers are not concentric.", off)

    def hub_at(point):
        if not len(centres):
            return None
        gaps = np.linalg.norm(centres - point, axis=1)
        nearest = int(np.argmin(gaps))
        return nearest if gaps[nearest] <= max(hubs[nearest].dimensions) / 2 else None

    # 2. Struts run hub to hub and the lattice is triangulated.
    stray, degree, lengths, carried, unfooted = [], defaultdict(int), defaultdict(list), set(), []
    joined = defaultdict(list)
    footings = [p for p in placements if p.role == "footing"]
    floors = [p for p in placements if p.role == "floor"]
    floor_top = max((p.position[1] + p.dimensions[1] / 2 for p in floors), default=None)
    grounded = bool(floors) and min((p.position[1] - p.dimensions[1] / 2 for p in placements), default=1.) <= TOL \
        and any(p.role == "floor_core" and p.position[1] - p.dimensions[1] / 2 <= TOL for p in placements)
    for p in placements:
        if p.role not in ("strut", "web_strut", "leg"):
            continue
        ends = strut_ends(p)
        if ends is None:
            stray.append(p.id)
            continue
        found = [hub_at(end) for end in ends]
        if p.role == "strut":
            lengths[p.parameters.get("layer", "outer"), p.parameters.get("frequency")].append(
                float(np.linalg.norm(ends[1] - ends[0])))
        if p.role == "leg":
            top, foot = (found[0], ends[1]) if ends[0][1] > ends[1][1] else (found[1], ends[0])
            seated = floor_top is not None and any(
                all(abs(foot[a] - f.position[a]) <= f.dimensions[a] / 2 + 1e-3 for a in range(3))
                and abs(f.position[1] - f.dimensions[1] / 2 - floor_top) <= 1e-3 for f in footings)
            if top is None or not seated:
                unfooted.append(p.id)
            else:
                carried.add(top)
        elif None in found or found[0] == found[1]:
            stray.append(p.id)
        elif p.role == "strut":
            same = hubs[found[0]].parameters.get("layer", "outer") == hubs[found[1]].parameters.get("layer", "outer")
            if not same:
                stray.append(p.id)
                continue
            for end, other in (found, found[::-1]):
                degree[end] += 1
                joined[end].append(other)
    interior = [i for i in range(len(hubs)) if i not in carried]
    wrong = [hubs[i].id for i in interior if degree[i] not in (5, 6)]
    wrong += [hubs[i].id for i in carried if degree[i] < 2]
    for layer, members in layers.items():
        fives = [i for i in members if i in interior and degree[i] == 5]
        centre = spheres.get(layer, (np.zeros(3), 1.))[0]
        rays = [(centres[i] - centre) / max(np.linalg.norm(centres[i] - centre), 1e-9) for i in fives]
        close = any(float(a @ b) > math.cos(math.radians(60)) for n, a in enumerate(rays) for b in rays[n + 1:])
        if len(fives) > 11 or close:                # the five-strut hubs are the icosahedron's own corners
            wrong += [hubs[i].id for i in fives]
    yield ("triangulated", bool(degree) and not stray and not wrong,
           "Every strut joins two hubs, and every hub inside the cut edge gathers five or six struts.",
           "A strut misses a hub, or a hub inside the cut edge does not gather five or six struts.", stray + wrong)

    # 3. A few strut lengths build the whole shell.
    many = []
    for (layer, frequency), found in lengths.items():
        allowed = FAMILIES.get(frequency, max(FAMILIES.values()))
        if _families(found) > allowed or max(found) > 1.35 * min(found):
            many.append(f"{layer}:{_families(found)}")
    yield ("strut_families", bool(lengths) and not many,
           "The lattice struts of each layer fall into the few lengths the frequency allows.",
           "The lattice struts come in more lengths than the frequency allows.", [])

    # 4. Every hub on the cut edge is carried to the ground.
    hanging = []
    for i, hub in enumerate(hubs):
        centre = spheres.get(hub.parameters.get("layer", "outer"), (np.zeros(3), 1.))[0]
        if i not in carried and _open_side(centres[i], centre, centres[joined[i]]):
            hanging.append(hub.id)
    yield ("grounded_base", grounded and bool(carried) and not unfooted and not hanging,
           "Every hub on the cut edge stands on a leg and a footing that reaches the floor.",
           "A hub on the cut edge is not carried, or a leg has no footing on the floor.", unfooted + hanging)
