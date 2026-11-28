"""Imperial Inka kancha: inward-facing halls around an enclosed open court.

The layout and masonry draw on late pre-Hispanic Inka compounds of the
Cusco region. The fitted, irregular stone faces, inclined door jambs and
steep reconstructed roofs are stylized, not a survey of one site.
"""
from __future__ import annotations

from collections import defaultdict
import math
import sys

import numpy as np

from .historic import Composer, Form, number, oriented_profile
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("inka", "1.0.0", "Imperial Inka Architect")
STYLE = "imperial-inka"
REGION = "Cusco region, central Andes (Peru)"
PERIOD = "Imperial Inka, 15th to early 16th century"
PRODUCT = "inka-kancha"
KINDS = {
    "foundation": "foundation", "court_floor": "slab", "opening_step": "circulation",
    "court_wall": "wall", "hall_front": "wall", "hall_back": "wall", "gable": "wall",
    "lintel": "beam", "tie_beam": "beam", "rafter": "beam", "thatch": "roof",
}
FORMS = {PRODUCT: Form(
    PRODUCT, "Imperial Inka courtyard compound with fitted stone walls, trapezoidal doors and thatched gables",
    (28, 8, 28), ("cusco-court",), frozenset(KINDS), {"bays": 2},
    rules={"region": REGION, "period": PERIOD, "bays": [1, 3],
           "bays_meaning": "trapezoidal doorways per hall facing the court",
           "width": [20, 38], "height": [6, 10], "depth": [20, 38],
           "plan/height": [2.9, 4.8]},
)}
DEFAULT_SUBSTANCE = "granite"
SUBSTANCES = {"foundation": "earth-fill", "tie_beam": "softwood", "rafter": "softwood", "thatch": "thatch"}
STONE = ((148, 139, 126, 255), (156, 148, 137, 255), (137, 131, 120, 255), (163, 155, 142, 255))
PAVILIONS = ("north", "south", "east", "west")
TOL = 1e-5


class Kancha(Composer):
    PALETTE = {"wall": STONE[0], "lintel": STONE[1], "foundation": (121, 101, 78, 255),
               "court_floor": (164, 146, 120, 255), "opening_step": STONE[0],
               "tie_beam": (95, 66, 40, 255), "rafter": (95, 66, 40, 255), "thatch": (159, 120, 61, 255)}
    TIMBER_KINDS = frozenset({"beam"})
    BASE, WALL, LINTEL = .18, .48, .24

    def compose(self):
        for name, value in zip(("width", "height", "depth"), (self.w, self.h, self.d)):
            number(value, f"{PRODUCT} {name}", *self.form.rules[name])
        ratios = self.form.rules["plan/height"]
        self.ratio(self.w / self.h, "width/height", *ratios)
        self.ratio(self.d / self.h, "depth/height", *ratios)
        self.doors = self.control("bays", *self.form.rules["bays"])
        self.eaves = self.h * .48 * self.vary("eaves", .035)
        self.door_top = self.BASE + min(2.05, .28 * self.h)
        self.depth = self.h * .74
        self.course, self.stretcher, self.roof_module = {
            "concept": (.68, 1.34, 2.4), "standard": (.53, 1.09, 1.75), "presentation": (.40, .86, 1.2),
        }[self.detail]
        x, z = self.w / 2 - .48, self.d / 2 - .48
        ix, iz = x - self.depth, z - self.depth
        lx = min(.34 * self.w, x - 2.45)
        lz = iz - 1.55  # leave a walkable path from the offset gate around the south hall
        self.slab("foundation", -self.w / 2, self.w / 2, -self.d / 2, self.d / 2,
                  0, self.BASE, module=2.8, unit="core", construction="compacted earth fill")
        self.slab("court_floor", -ix + .42, ix - .42, -iz + .42, iz - .42,
                  self.BASE, self.BASE + .06, module=1.45)

        self.hall("north", "x", lx, iz, z)
        self.hall("south", "x", lx, -iz, -z)
        self.hall("east", "z", lz, ix, x)
        self.hall("west", "z", lz, -ix, -x)
        # Infill extends each outside wall to the corner; the gate is offset
        # so its approach bypasses the south hall instead of running into it.
        for sign in (-1, 1):
            for a, b in ((-x - self.WALL / 2, -lx), (lx, x + self.WALL / 2)):
                gate = sign == -1 and a > 0
                center = (a + b) / 2
                self.wall("court_wall", "x", a, b, sign * z, self.door_top + .63,
                          "enclosure", (center,) if gate else ())
                if gate:
                    self.member("opening_step", center - .64, center + .64, self.BASE, self.BASE + .06,
                                -z - self.WALL / 2, -z + self.WALL / 2)
            for a, b in ((-z + self.WALL / 2, -lz), (lz, z - self.WALL / 2)):
                self.wall("court_wall", "z", a, b, sign * x, self.door_top + .63, "enclosure")
        return self.document({"region": REGION, "period": PERIOD, "doors_per_hall": self.doors,
                              "pavilions": list(PAVILIONS)})

    def fitted(self, role, axis, cross, pavilion, y0, y1, edges, course, opening=None):
        """Tile a band with mating polygonal joints shared by adjacent stones."""
        low, high = edges
        pitch = max(high[1] - high[0], low[1] - low[0])
        count = max(1, math.ceil(pitch / self.stretcher))
        x0, x1 = np.linspace(low[0], low[1], count + 1), np.linspace(high[0], high[1], count + 1)
        # Stagger the joints while holding both boundary seams fixed.
        if count > 1:
            shift = (.21 if course % 2 else -.21) / count
            x0[1:-1] += shift * (low[1] - low[0])
            x1[1:-1] += shift * (high[1] - high[0])
        mid_y = (y0 + y1) / 2
        middle = (x0 + x1) / 2
        for i in range(1, count):
            middle[i] += .32 * pitch / count * (self.vary(f"joint:{role}:{axis}:{cross}:{y0}:{middle[i]}", .4) - 1)
            middle[i] += (.07 if (i + course) % 2 else -.07) * pitch / count
        for i in range(count):
            profile = [(x0[i], y0), (x0[i + 1], y0)]
            if i + 1 < count:
                profile.append((middle[i + 1], mid_y))
            if x1[i + 1] - x1[i] > TOL:
                profile.append((x1[i + 1], y1))
            profile.append((x1[i], y1))
            if i:
                profile.append((middle[i], mid_y))
            self.mesh(role, oriented_profile(profile, self.WALL, axis, 0., cross),
                      pavilion=pavilion, course=course, unit="fitted_stone", construction="dry-fitted stone",
                      **({"opening_id": opening} if opening else {}))
            placement = self.placements[-1]
            placement.material = self.pick(f"stone:{placement.id}", STONE)

    def wall(self, role, axis, a, b, cross, top, pavilion, centers=()):
        openings = [(center, f"{pavilion}.door-{i}") for i, center in enumerate(centers, 1)]
        marks = [self.BASE, self.door_top, self.door_top + self.LINTEL, top] if openings else [self.BASE, top]
        course = 0
        for bottom, top in zip(marks, marks[1:]):
            bands = np.linspace(bottom, top, math.ceil((top - bottom) / self.course) + 1)
            for y0, y1 in zip(bands, bands[1:]):
                if openings and y0 < self.door_top + self.LINTEL - TOL:
                    foot, head = self.door_half(y0), self.door_half(y1)
                    if y0 >= self.door_top - TOL:
                        foot = head = self.door_half(self.door_top) + .23
                    self.jamb_band(role, axis, a, b, cross, pavilion, y0, y1,
                                   course, openings, foot, head)
                else:
                    self.fitted(role, axis, cross, pavilion, y0, y1, ((a, b), (a, b)), course)
                course += 1
        for center, opening in openings:
            half = self.door_half(self.door_top) + .23
            profile = [(center - half, self.door_top), (center + half, self.door_top),
                       (center + half, self.door_top + self.LINTEL), (center - half, self.door_top + self.LINTEL)]
            self.mesh("lintel", oriented_profile(profile, self.WALL, axis, 0., cross),
                      opening_id=opening, pavilion=pavilion)

    def door_half(self, y):
        return .59 - .17 * min(1., (y - self.BASE) / (self.door_top - self.BASE))

    def jamb_band(self, role, axis, a, b, cross, pavilion, y0, y1, course, doors, foot, head):
        # Split a panel between doors into left and right reveal groups so
        # the audit can recover both apertures from the actual stone edges.
        right_edges = [(center - foot, center - head) for center, _ in doors] + [(b, b)]
        for i in range(len(doors) + 1):
            if i:
                center, opening = doors[i - 1]
                left = (center + foot, center + head)
            else:
                left, opening = (a, a), doors[0][1]
            right = right_edges[i]
            if 0 < i < len(doors):
                mid = tuple((x + y) / 2 for x, y in zip(left, right))
                pieces = [(left, mid, opening), (mid, right, doors[i][1])]
            else:
                pieces = [(left, right, opening)]
            for start, stop, tag in pieces:
                self.fitted(role, axis, cross, pavilion, y0, y1, tuple(zip(start, stop)), course, tag)

    def hall(self, name, axis, half, front, back):
        centers = np.linspace(-half, half, self.doors + 2)[1:-1]
        self.wall("hall_front", axis, -half, half, front, self.eaves, name, tuple(centers))
        self.wall("hall_back", axis, -half, half, back, self.eaves, name)
        cross_axis = "z" if axis == "x" else "x"
        lo, hi = sorted((front, back))
        ridge = (lo + hi) / 2
        for end in (-half + self.WALL / 2, half - self.WALL / 2):
            self.wall("gable", cross_axis, lo + self.WALL / 2, hi - self.WALL / 2, end, self.eaves, name)
            peak = self.h - .32
            y_marks = np.linspace(self.eaves, peak, math.ceil((peak - self.eaves) / self.course) + 1)
            for course, (y0, y1) in enumerate(zip(y_marks, y_marks[1:])):
                span = [(hi - lo) / 2 * (peak - y) / (peak - self.eaves) for y in (y0, y1)]
                self.fitted("gable", cross_axis, end, name, y0, y1,
                            tuple((ridge - value, ridge + value) for value in span), course)

        def roof_panel(role, a, b, left, right, under, above):
            slope = (self.h - .19 - self.eaves) / (ridge - lo)
            yl = self.h - .19 - slope * abs(left - ridge)
            yr = self.h - .19 - slope * abs(right - ridge)
            profile = [(left, yl + under), (right, yr + under), (right, yr + above), (left, yl + above)]
            self.mesh(role, oriented_profile(profile, b - a, cross_axis, 0., (a + b) / 2), pavilion=name)

        frame_count = max(3, math.ceil(half * 2 / 2.3))
        for mark in np.linspace(-half + .28, half - .28, frame_count):
            beam = [(lo - .14, self.eaves - .18), (hi + .14, self.eaves - .18),
                    (hi + .14, self.eaves), (lo - .14, self.eaves)]
            self.mesh("tie_beam", oriented_profile(beam, .16, cross_axis, 0., mark), pavilion=name)
            for left, right in ((lo - .22, ridge), (ridge, hi + .22)):
                roof_panel("rafter", mark - .08, mark + .08, left, right, -.16, 0.)
        modules = np.linspace(-half - .19, half + .19, math.ceil(2 * (half + .19) / self.roof_module) + 1)
        for a, b in zip(modules, modules[1:]):
            for left, right in ((lo - .22, ridge), (ridge, hi + .22)):
                bands = np.linspace(left, right, 2 if self.detail == "concept" else 4)
                for low, high in zip(bands, bands[1:]):
                    roof_panel("thatch", a, b, low, high, 0., .19)

    def block_count(self):
        # Stone lintels are not timbers even though they span an opening.
        bill = super().block_count()
        cuts = sum(p.parameters.get("unit") == "fitted_stone" for p in self.placements)
        bill["cut_stones"] = cuts + self.counts["lintel"]
        bill["timbers"] -= self.counts["lintel"]
        bill["other_members"] -= cuts
        return bill


Kancha.module = sys.modules[__name__]


def build_document(request):
    return Kancha(request).compose()


def random_controls(product, rng):
    h = rng.randint(7, 9)
    return {"variant": FORMS[product].variants[0], "width": round(h * rng.uniform(3.25, 4.12)),
            "height": h, "depth": round(h * rng.uniform(3.3, 4.1)), "bays": rng.randint(1, 3),
            "levels": None, "detail": rng.choice(("concept", "standard", "presentation"))}


def _bounds(p):
    center, radius = np.asarray(p.position), np.asarray(p.dimensions) / 2
    return center - radius, center + radius


def _vertices(p):
    return np.asarray(p.parameters["vertices"]) * p.dimensions + p.position


def rule_checks(document, form):
    by_role, jambs = defaultdict(list), defaultdict(list)
    for member in document.placements:
        by_role[member.role].append(member)
        opening = member.parameters.get("opening_id")
        if member.role in {"hall_front", "court_wall"} and opening and member.block_type == "MESH":
            jambs[opening].append(member)

    bad_doors = []
    valid_doors = defaultdict(list)
    for lintel in by_role["lintel"]:
        reveals = jambs.get(lintel.parameters.get("opening_id"), ())
        lo, hi = _bounds(lintel)
        axis = 0 if lintel.dimensions[0] > lintel.dimensions[2] else 2
        cross = 2 - axis
        center = lintel.position[axis]
        groups = [[], []]
        for p in reveals:
            if p.position[1] < lo[1] and abs(p.position[cross] - lintel.position[cross]) < TOL:
                groups[int(p.position[axis] > center)].append(_vertices(p))
        if all(groups):
            left, right = (np.concatenate(g) for g in groups)
            base = min(left[:, 1].min(), right[:, 1].min())
            a = left[np.abs(left[:, 1] - base) < TOL, axis]
            b = right[np.abs(right[:, 1] - base) < TOL, axis]
            c = left[np.abs(left[:, 1] - lo[1]) < TOL, axis]
            d = right[np.abs(right[:, 1] - lo[1]) < TOL, axis]
            if all(v.size for v in (a, b, c, d)) and (
                b.min() - a.max() > d.min() - c.max() + .15
                and d.min() - c.max() >= .62
                and lo[axis] < c.max() < center < d.min() < hi[axis]
            ):
                valid_doors[lintel.parameters.get("pavilion")].append(lintel)
                continue
        bad_doors.append(lintel.id)
    door_count = document.request.bays if document.request.bays is not None else form.controls["bays"]
    yield ("trapezoidal_openings", not bad_doors and len(valid_doors["enclosure"]) == 1
           and all(len(valid_doors[name]) == door_count for name in PAVILIONS),
           "Each stone lintel rests over a doorway narrower at the head than at the sill.",
           "A lintel is missing or no longer spans the tapering jambs.", bad_doors)

    floors = [_bounds(p) for p in by_role["court_floor"]]
    bad_covers, outward = [], []
    if floors:
        lo, hi = np.min([lo for lo, _ in floors], axis=0), np.max([hi for _, hi in floors], axis=0)
        court = (lo + hi) / 2
        for p in by_role["thatch"] + by_role["rafter"] + by_role["tie_beam"]:
            a, b = _bounds(p)
            if b[1] > hi[1] + TOL and all(min(hi[i], b[i]) - max(lo[i], a[i]) > TOL for i in (0, 2)):
                bad_covers.append(p.id)
        for name in PAVILIONS:
            backs = [p for p in by_role["hall_back"] if p.parameters.get("pavilion") == name]
            axis = 2 if name in {"north", "south"} else 0
            for p in valid_doors[name]:
                door, rear = p.position[axis] - court[axis], [q.position[axis] - court[axis] for q in backs]
                if not rear or any(door * r <= 0 or abs(door) >= abs(r) - TOL for r in rear):
                    outward.append(p.id)
    yield ("open_court", bool(floors and by_role["thatch"]) and not bad_covers,
           "Roofing and frames stay off the open central court.", "The central court is absent or roofed.", bad_covers)
    yield ("inward_halls", bool(floors) and not outward and all(valid_doors[name] for name in PAVILIONS),
           "Four rectangular halls open inward, toward the court.", "A hall is missing or does not face the court.", outward)

    bad_roof = []
    paired = defaultdict(set)
    for r in by_role["rafter"]:
        pavilion = r.parameters.get("pavilion")
        axis = 2 if pavilion in {"north", "south"} else 0
        along = 2 - axis
        pts = _vertices(r) if r.block_type == "MESH" else None
        if pts is not None:
            at_left = pts[np.abs(pts[:, axis] - pts[:, axis].min()) < TOL, 1].max()
            at_right = pts[np.abs(pts[:, axis] - pts[:, axis].max()) < TOL, 1].max()
            pitch = (at_right - at_left) / r.dimensions[axis]
            if .85 <= abs(pitch) <= 1.85:
                paired[pavilion, round(r.position[along], 5)].add(pitch > 0)
                continue
        bad_roof.append(r.id)
    yield ("steep_gables", not bad_roof and all(len(signs) == 2 for signs in paired.values())
           and {name for name, _ in paired} == set(PAVILIONS),
           "Every hall carries steep paired rafters beneath its thatch.",
           "A gable is absent, shallow, or missing one of its paired slopes.", bad_roof)
