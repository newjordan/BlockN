"""Minangkabau Rumah Gadang: raised timber frame and swept gonjong roofs."""
from __future__ import annotations

from collections import defaultdict
import itertools
import math
import random
import sys

import numpy as np
import trimesh

from .grammar import Bearing
from .historic import Composer, Form, extent, number, oriented_profile
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("minangkabau", "1.0.0", "Minangkabau Architect")
STYLE = "minangkabau-timber"
PRODUCT = "minangkabau-rumah-gadang"
REGION = "Minangkabau Highlands, West Sumatra, Indonesia"
PERIOD = "Traditional Rumah Gadang domestic architecture"
VARIANTS = ("bodi-caniago", "koto-piliang")
PEAKS = ("west", "east", "front-west", "front-east")
KINDS = {
    "padstone": "foundation", "floor_post": "column", "wall_post": "column", "underfloor_brace": "beam",
    "floor_girder": "beam", "floor_joist": "beam", "living_floor": "slab", "anjuang_floor": "slab",
    "anjuang_riser": "wall", "entry_tread": "circulation", "entry_stringer": "beam",
    "wall_board": "wall", "wall_rail": "beam", "window_lattice": "infill", "frieze": "ornament",
    "roof_tie": "beam", "roof_rafter": "beam", "gonjong": "roof",
}
OPTIONAL = {"anjuang_floor", "anjuang_riser"}
FORMS = {PRODUCT: Form(
    PRODUCT, "Raised West Sumatran timber communal house with swept gonjong peaks and an optional pair of raised end platforms",
    (26, 13, 15), VARIANTS, frozenset(KINDS.keys() - OPTIONAL), {"bays": 5},
    rules={"region": REGION, "period": PERIOD, "bays": [3, 7], "bays_step": 2,
           "bays_meaning": "lengthwise bays across the front, always odd",
           "width": [20, 36], "height": [10, 19], "depth": [12, 22],
           "depth/width": [.46, .72], "height/width": [.42, .65],
           "raised_ends": {"bodi-caniago": False, "koto-piliang": True}},
)}
DEFAULT_SUBSTANCE = "hardwood"
SUBSTANCES = {"padstone": "granite", "gonjong": "thatch"}
TOL = 1e-5
# Identical triangulation on both faces preserves a closed, thickened patch
# even when the four surface samples do not lie in a plane.
PATCH_FACES = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
               (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
               (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0))


def roof_patch(points, thickness):
    top = np.asarray(points)
    mesh = trimesh.Trimesh(vertices=np.vstack((top, top - (0., thickness, 0.))),
                          faces=PATCH_FACES, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


class Gadang(Composer):
    PALETTE = {
        "padstone": (140, 138, 128, 255), "column": (106, 71, 47, 255),
        "beam": (124, 89, 57, 255), "wall": (139, 69, 39, 255),
        "window_lattice": (191, 137, 63, 255), "frieze": (214, 152, 64, 255),
        "gonjong": (58, 54, 49, 255), "slab": (146, 108, 68, 255),
        "entry_tread": (151, 106, 64, 255),
    }
    TIMBER_KINDS = frozenset({"column", "beam"})

    def compose(self):
        for name, value in zip(("width", "height", "depth"), (self.w, self.h, self.d)):
            number(value, f"{PRODUCT} {name}", *self.form.rules[name])
        self.ratio(self.d / self.w, "depth/width", *self.form.rules["depth/width"])
        self.ratio(self.h / self.w, "height/width", *self.form.rules["height/width"])
        bays = self.control("bays", 3, 7)
        if bays % 2 != 1:
            raise ValueError("Rumah Gadang requires an odd bay count for its central entry")
        self.x, self.z = .397 * self.w, .34 * self.d
        self.floor = self.h * .16 * self.vary("floor-lift", .085)
        self.eave = self.h * .50 * self.vary("eave-height", .045)
        self.sweep = self.vary("sweep", .18)
        self.lines = np.linspace(-self.x, self.x, bays + 1)
        self.frame()
        self.deck()
        self.enclose()
        self.stair()
        self.roof()
        return self.document({"region": REGION, "period": PERIOD, "lengthwise_bays": bays,
                              "raised_ends": self.variant == "koto-piliang", "gonjong_peaks": 4})

    def frame(self):
        xs = self.lines
        for i, x in enumerate(xs):
            for row, z in enumerate((-self.z, 0., self.z)):
                half = .34
                self.member("padstone", x - half, x + half, 0, .29, z - half, z + half, bent=i, row=row)
                self.member("floor_post", x - .14, x + .14, .29, self.floor - .36,
                            z - .14, z + .14, bent=i, row=row)
                head = self.roof_y(x, z) - .14
                self.member("wall_post", x - .14, x + .14, self.floor - .13, head + .025,
                            z - .14, z + .14, bent=i, row=row)
            self.member("floor_joist", x - .115, x + .115, self.floor - .21, self.floor - .085,
                        -self.z - .15, self.z + .15, bent=i)
            self.member("roof_tie", x - .14, x + .14, self.eave - .14, self.eave + .06,
                        -self.z - .14, self.z + .14, bent=i)
            for side in (-1, 1):
                # Transverse bracing in the undercroft, seated against the posts.
                t = side * (self.z - .11)
                s = side * .12
                profile = [(s, .47), (t, self.floor - .57), (t, self.floor - .38), (s, .66)]
                self.mesh("underfloor_brace", oriented_profile(profile, .16, "z", 0., x), bent=i)
        for row, z in enumerate((-self.z, 0., self.z)):
            self.member("floor_girder", -self.x - .15, self.x + .15, self.floor - .36, self.floor - .21,
                        z - .125, z + .125, row=row)

    def deck(self):
        x, z, top = self.x + .16, self.z + .13, self.floor
        self.slab("living_floor", -x, x, -z, z, top - .085, top, module=1.32)
        if self.variant == "koto-piliang":
            lift = .42 * self.vary("anjuang-lift", .16)
            for side in (-1, 1):
                a, b = sorted((side * x, side * (self.x - .78 * (self.lines[1] - self.lines[0]))))
                self.slab("anjuang_floor", a, b, -z, z, top + lift - .08, top + lift, module=1.15, end=side)
                for c in (a, b):
                    c = min(c, b - .12)
                    self.member("anjuang_riser", c, c + .12, top, top + lift - .08, -z, z, end=side)

    def enclosure_panel(self, a, b, cross, axis, base, top, **tags):
        """Taper a timber board to meet the underside of the roof envelope."""
        def head(u):
            x, z = (u, cross) if axis == "x" else (cross, u)
            return min(top, self.roof_y(x, z) - .13)
        ya, yb = head(a), head(b)
        if min(ya, yb) - base <= .02:
            return
        profile = [(a, base), (b, base), (b, yb), (a, ya)]
        self.mesh("wall_board", oriented_profile(profile, .095, axis, 0., cross), **tags)

    def enclose(self):
        pitch = {"concept": .55, "standard": .38, "presentation": .25}[self.detail]
        # The street facade has one central stair and a symmetric window rhythm.
        for i, (a, b) in enumerate(zip(self.lines, self.lines[1:])):
            front = -self.z
            mid = (a + b) / 2
            entry = i == (len(self.lines) - 2) // 2
            half = .65 if entry else min(.31 * (b - a), .92)
            sill = self.floor if entry else self.floor + .78
            head = self.floor + (2.07 if entry else 1.68)
            breaks = (a, mid - half, mid + half, b)
            for n, (lo, hi) in enumerate(zip(breaks, breaks[1:])):
                ticks = np.linspace(lo, hi, math.ceil((hi - lo) / pitch) + 1)
                for u, v in zip(ticks, ticks[1:]):
                    if n == 1:
                        self.enclosure_panel(u, v, front, "x", self.floor, sill, bay=i)
                        self.enclosure_panel(u, v, front, "x", head, math.inf, bay=i)
                    else:
                        self.enclosure_panel(u, v, front, "x", self.floor, math.inf, bay=i)
            if not entry:
                self.member("wall_rail", mid - half - .10, mid + half + .10, sill - .11, sill + .05,
                            front - .075, front + .075, bay=i)
                for x in np.linspace(mid - half + .05, mid + half - .05, 7):
                    self.member("window_lattice", x - .025, x + .025, sill, head + .045,
                                front - .037, front + .036, bay=i)
            # Raised, patterned fascia below the roof over the lengthwise bays.
            y = self.floor + 2.20
            self.member("frieze", a, b, y, y + .23, front - .072, front - .045, bay=i)
            for x in np.linspace(a + .28, b - .28, 6):
                profile = [(x - .09, y + .115), (x, y + .04), (x + .09, y + .115), (x, y + .19)]
                self.mesh("frieze", oriented_profile(profile, .045, "x", 0., front - .081), bay=i,
                          construction="stylized carved timber relief")
                self.placements[-1].material = (112, 52, 35, 255)
        for axis, cross, start, end in (("x", self.z, -self.x, self.x), ("z", -self.x, -self.z, self.z),
                                         ("z", self.x, -self.z, self.z)):
            ticks = np.linspace(start, end, math.ceil((end - start) / pitch) + 1)
            for a, b in zip(ticks, ticks[1:]):
                self.enclosure_panel(a, b, cross, axis, self.floor, math.inf)

    def stair(self):
        count = math.ceil(self.floor / .24)
        run = .15 * self.d
        start, end = -self.z - run, -self.z
        for k in range(count):
            lo, hi = start + k * run / count, start + (k + 1) * run / count
            y = self.floor * (k + 1) / count
            self.member("entry_tread", -.68, .68, max(0., y - .105), y, lo, hi, step=k)
        points = [(start, 0.), (end, self.floor - .31), (end, self.floor - .10), (start, self.floor / count - .095)]
        for x in (-.49, .49):
            self.mesh("entry_stringer", oriented_profile(points, .18, "z", 0., x))

    def main_point(self, t, v, sign):
        ridge = self.h * (.755 + .245 * abs(t) ** (3.5 / self.sweep))
        edge = self.eave + self.h * .13 * abs(t) ** 2.6
        return (self.w * (.5 - .047 * v) * t, ridge + (edge - ridge) * v ** .74,
                sign * self.d * .475 * v)

    def roof_y(self, x, z):
        v = abs(z) / (.475 * self.d)
        t = x / (self.w * (.5 - .047 * v))
        return self.main_point(t, v, 1)[1]

    def front_point(self, t, v, sign, center):
        half = self.w * (.14 - .067 * t ** 1.5)
        x = center + sign * half * v
        z = -self.d * (.11 + (.39 - .044 * v) * t)
        # Follow the main slope at the valley instead of burying a flat
        # cross-gable in it. The rear edge tucks under the covering; only the
        # swept front tip projects past the main eave.
        bed = self.roof_y(x, max(z, -.475 * self.d))
        lift = self.h * (.39 * t ** (3.25 / self.sweep) * (1 - v ** .74) - .016)
        return x, bed + lift, z

    def roof(self):
        # Coverings use edge-matched patches, not flat bounding-box roofs.
        steps, slopes = {"concept": (8, 5), "standard": (12, 8), "presentation": (18, 11)}[self.detail]
        for i, peak in enumerate(PEAKS):
            end = -1 if i % 2 == 0 else 1
            center = end * self.w * .227
            if i < 2:
                def sample(t, v, s): return self.main_point(end * t, v, s)
            else:
                def sample(t, v, s): return self.front_point(t, v, s, center)
            for side, j, k in itertools.product((-1, 1), range(steps), range(slopes)):
                u, b = j / steps, (j + 1) / steps
                v, w = k / slopes, (k + 1) / slopes
                p = [sample(t, f, side) for t, f in ((u, v), (b, v), (b, w), (u, w))]
                self.mesh("gonjong", roof_patch(p, .14), peak=peak, side=side, station=j, course=k,
                          construction="layered dark palm-fiber covering")
                shade = (51, 54, 56, 53)[k % 4]
                self.placements[-1].material = (shade + 6, shade + 3, shade - 1, 255)
                if j % max(2, steps // 5) == 0:
                    # Interpolate inside exactly the same roof quad to seat the rib.
                    a, c, d, e = np.asarray(p)
                    f = .16
                    q = np.array([a, a + f * (c - a), e + f * (d - e), e]) - (0., .135, 0.)
                    self.mesh("roof_rafter", roof_patch(q, .14), peak=peak, side=side, station=j, course=k)


Gadang.module = sys.modules[__name__]


def build_document(request):
    return Gadang(request).compose()


def random_controls(product, rng):
    w = rng.randint(22, 32)
    return {"width": w, "height": round(w * rng.uniform(.47, .55)), "depth": round(w * rng.uniform(.53, .64)),
            "bays": rng.choice((3, 5, 7)), "detail": rng.choice(("concept", "standard"))}


def bound(part, axis, end):
    return part.position[axis] + end * part.dimensions[axis] / 2


def ridge_edge(part):
    """The two covering vertices on the ridge, in root-to-tip order."""
    vertices = np.asarray(part.parameters.get("vertices", []), dtype=float)
    if vertices.shape != (8, 3):
        return None
    # roof() authors the upper ridge endpoints first (roof_patch keeps the
    # vertex order on both flanks, even when it reverses face winding).
    return vertices[:2] * part.dimensions + part.position


def swept_ridges(panels):
    """Trace and compare both joined flanks of each mirrored ridge pair."""
    peaks = defaultdict(dict)
    for part in panels:
        name = part.parameters.get("peak")
        key = (part.parameters.get("station"), part.parameters.get("side"))
        edge = ridge_edge(part)
        if key in peaks[name] or edge is None:
            return False
        peaks[name][key] = edge
    if set(peaks) != set(PEAKS):
        return False

    def matches(a, b):
        return np.allclose(a, b, atol=TOL, rtol=0)

    ridges = {}
    for name, flank in peaks.items():
        count = len(flank) // 2
        if count < 4 or flank.keys() != set(itertools.product(range(count), (-1, 1))):
            return False
        edges = [flank[i, 1] for i in range(count)]
        if (not all(matches(edge, flank[i, -1]) for i, edge in enumerate(edges))
                or not all(matches(a[1], b[0]) for a, b in zip(edges, edges[1:]))):
            return False
        ridge = np.vstack((edges[0][0], [edge[1] for edge in edges]))
        heights = ridge[:, 1]
        if not (heights[-1] - min(heights[:count // 2 + 1]) > .14 * heights[-1]
                and heights[-1] - heights[-2] > heights[-2] - heights[-3] > 0):
            return False
        ridges[name] = ridge
    return all(ridges[west].shape == ridges[east].shape and matches(ridges[west] * (-1, 1, 1), ridges[east])
               for west, east in (("west", "east"), ("front-west", "front-east")))


def rule_checks(document, form):
    roles = defaultdict(list)
    for part in document.placements:
        roles[part.role].append(part)
    pads = {(p.parameters.get("bent"), p.parameters.get("row")): p for p in roles["padstone"]}
    girders = {p.parameters.get("row"): p for p in roles["floor_girder"]}
    posts = roles["floor_post"]
    members = {p.id: p for p in (*posts, *pads.values(), *girders.values())}
    bad = []
    for p in posts:
        key = p.parameters.get("bent"), p.parameters.get("row")
        foot, head = pads.get(key), girders.get(key[1])
        if (not foot or not head or not Bearing(p.id, foot.id, 1.).holds(members)
                or not Bearing(head.id, p.id, 1.).holds(members) or abs(bound(foot, 1, -1)) > TOL
                or any(abs(p.position[a] - foot.position[a]) > TOL or p.dimensions[a] > foot.dimensions[a] + TOL
                       for a in (0, 2))
                or abs(p.position[2] - head.position[2]) > TOL
                or bound(p, 0, -1) < bound(head, 0, -1) - TOL or bound(p, 0, 1) > bound(head, 0, 1) + TOL):
            bad.append(p.id)
    bays = document.request.bays if document.request.bays is not None else form.controls["bays"]
    grid = set(itertools.product(range(bays + 1), range(3)))
    actual = {(p.parameters.get("bent"), p.parameters.get("row")) for p in posts}
    yield ("raised_frame", actual == grid == set(pads) and len(posts) == len(roles["padstone"]) == len(grid) and not bad,
           "The raised post grid seats on padstones and carries the floor girders.",
           "A timber post is missing, unseated or does not carry its girder.", bad)

    ridge_panels = [p for p in roles["gonjong"] if p.parameters.get("course") == 0]
    sweep_ok = swept_ridges(ridge_panels)
    yield ("swept_gonjong", sweep_ok, "Two matched pairs of ridges sweep to rising gonjong tips.",
           "A gonjong ridge is incomplete, flat or displaced relative to its pair.",
           [] if sweep_ok else [p.id for p in ridge_panels])

    deck, ends = roles["living_floor"], roles["anjuang_floor"]
    single_level = bool(deck) and max(bound(p, 1, 1) for p in deck) - min(bound(p, 1, 1) for p in deck) < TOL
    variant = document.request.variant
    if variant in {"auto", "random"}:
        variant = (form.variants[0] if variant == "auto" else
                   random.Random(document.request.seed).choice(form.variants))
    ok = single_level and not (ends or roles["anjuang_riser"])
    if variant == "koto-piliang" and single_level:
        middle, upper = extent(deck)[1], defaultdict(list)
        for p in ends:
            upper[p.parameters.get("end")].append(p)
        ok = set(upper) == {-1, 1} and bool(roles["anjuang_riser"])
        if ok:
            west, east = extent(upper[-1]), extent(upper[1])
            lift = west[1][1] - middle[1]
            ok = (.30 < lift < .54 and np.allclose(west[0][[0, 1, 2]],
                  (-east[1][0], east[0][1], east[0][2]), atol=TOL, rtol=0)
                  and abs(west[1][0] + east[0][0]) < TOL and abs(west[1][1] - east[1][1]) < TOL
                  and abs(east[1][0] - middle[0]) < TOL and west[1][0] < 0 < east[0][0]
                  and all(abs(bound(p, 1, 1) - (middle[1] + lift)) < TOL for p in ends))
    yield ("floor_tradition", ok, "The shared floor is level, with paired anjuang only in the raised-end variant.",
           "The modeled floor levels do not match the selected layout.", [] if ok else [p.id for p in ends])
