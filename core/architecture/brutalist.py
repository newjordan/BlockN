"""A concrete Brutalist grammar, applied to three programs and one spatial study."""
from __future__ import annotations

from dataclasses import asdict
import math
import sys

from .grammar import Bearing, Bounds, Datum, TOL, WalkSurface, WalkingRoute, obstructing_members
from .historic import Composer, Form, number
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("brutalist", "1.1.0", "Brutalism Studio")
STYLE = "brutalist-concrete"
CIVIC, HOUSING, ARTS, FIELD = (f"brutalist-{brief}" for brief in (
    "civic-center", "housing", "arts-center", "portal-field"))
DEFAULT_SUBSTANCE = "cast-concrete"
SUBSTANCES = {"infill": "window-glass"}
KINDS = {
    "site_base": "foundation", "podium": "slab", "approach_step": "circulation",
    "gate_pier": "column", "transfer_girder": "beam", "council_roof": "roof",
    "pilotis": "column", "living_deck": "slab", "party_wall": "wall",
    "gallery_deck": "slab", "bridge_pier": "column", "bridge_seat": "beam",
    "sky_bridge": "circulation", "parapet": "wall", "shear_wall": "wall",
    "facade_rib": "column", "infill": "window", "cast_relief": "trim",
    "stair_spine": "wall", "stair_tread": "circulation", "stair_landing": "circulation", "stair_guard": "wall",
    "portal_pier": "column", "portal_lintel": "beam", "processional_bench": "seating",
}

RULES = {
    "protected_void": ("Reserve space before placing solids", ["minimum clear width", "minimum clear headroom", "no positive-volume intrusion"]),
    "visible_bearing": ("Make the geometric path of major members legible", ["face contact", "minimum contact fraction in X and Z"]),
    "bay_rhythm": ("Repeat an ordered set of datums, not necessarily equally spaced", ["station count and position", "vertical alignment of repeats"]),
    "material_expression": ("Distinguish exposed construction from infill", ["substrate by role", "cast member vs recessed glazing"]),
}


def _form(product, description, defaults, variants, required, controls, sizes, expressions):
    return Form(product, description, defaults, variants, frozenset({"site_base", *required.split()}), controls,
                rules={**dict(zip(("width", "height", "depth"), map(list, sizes))),
                       "bays": [3, 8], "bays_meaning": "Facade bays per range, or portal frames in the field",
                       **({"levels": [4, 8], "minimum_storey_rise": 2.55} if "levels" in controls else {}),
                       "grammar": {key: {"intent": intent, "measures": measures, "expression": expression}
                                   for (key, (intent, measures)), expression in zip(RULES.items(), expressions)}})


FORMS = {
    CIVIC: _form(CIVIC, "Civic forum and elevated council chamber on exposed transfer members", (46, 21, 38),
                 ("forum", "assembly-gate"), "podium approach_step gate_pier transfer_girder council_roof facade_rib infill",
                 {"bays": 4}, ((36, 65), (16, 31), (29, 52)), (
                     "A processional forecourt and tall gateway under the council chamber",
                     "Twin gate piers seat a deep transfer girder; the chamber starts on its upper face",
                     "Two glazed wings use repeating blade bays around a wider ceremonial center",
                     "Cast-concrete supports and council mass contrast with recessed civic glazing")),
    HOUSING: _form(HOUSING, "Collective housing lifted on pilotis, with deep loggias and repeated dwelling bays",
                    (45, 25, 23), ("slab", "stepped"), "pilotis living_deck party_wall facade_rib infill parapet",
                    {"bays": 6, "levels": 6}, ((34, 66), (18, 38), (19, 31)), (
                        "A through-promenade runs lengthwise under the dwellings between the pilotis rows",
                        "The first dwelling slab seats on paired rows of pilotis; later slabs seat on party walls",
                        "Dwelling partitions stack by bay even when the loggia edge steps back",
                        "Concrete slab edges, party walls and balcony parapets frame recessed glass")),
    ARTS: _form(ARTS, "Cultural complex with differentiated gallery blocks and exposed bridges over a public passage",
                 (44, 23, 36), ("sky-bridges", "stepped-galleries"), "gallery_deck bridge_pier bridge_seat sky_bridge facade_rib infill parapet "
                 "stair_spine stair_tread stair_landing stair_guard",
                 {"bays": 4}, ((35, 60), (17, 32), (28, 50)), (
                     "The public passage connects to both bridge decks by rear switchbacks and clear gallery doors",
                     "Bridge ends meet distinct concrete seats on uprights at the gallery edges",
                     "Independent galleries repeat blade bays; walkway heights connect to their floor datums",
                     "Deep cast-concrete gallery and bridge members stand apart from recessed glazing")),
    FIELD: _form(FIELD, "Abstract study of concrete portal, interval, compression and release", (26, 11, 42),
                  ("procession", "syncopated"), "portal_pier portal_lintel processional_bench",
                  {"bays": 5}, ((20, 38), (8, 15), (29, 63)), (
                      "An open axial corridor passes through a sequence of frames",
                      "Each lintel seats directly on two separate monolithic piers",
                      "Matched supports share stations; stations form an even or alternating cadence",
                      "Monolithic cast concrete remains the construction; no imitation stone or decorative cladding")),
}


def random_controls(product, rng):
    form = FORMS[product]
    w, h, d = [rng.randint(math.ceil(lo + (hi - lo) * .10), math.floor(hi - (hi - lo) * .10))
               for lo, hi in (form.rules[k] for k in ("width", "height", "depth"))]
    result = {"width": w, "height": h, "depth": d, "bays": rng.randint(3, 8)}
    if product == HOUSING:
        result["levels"] = rng.randint(4, min(8, math.floor((h - Studio.PLINTH) / 2.55 - 1.25)))
    return result


class Studio(Composer):
    module = sys.modules[__name__]
    TIMBER_KINDS = frozenset()
    PLINTH = .36
    PALETTE = {"site_base": (139, 133, 122, 255), "podium": (175, 168, 152, 255),
               "infill": (46, 61, 63, 255), "facade_rib": (188, 179, 161, 255),
               "wall": (163, 154, 139, 255), "beam": (182, 172, 153, 255),
               "column": (168, 160, 146, 255), "slab": (194, 184, 167, 255),
               "roof": (186, 175, 155, 255), "circulation": (178, 164, 143, 255),
               "trim": (160, 151, 137, 255), "seating": (205, 187, 159, 255)}

    def __init__(self, request):
        super().__init__(request)
        for name, size in zip(("width", "height", "depth"), (self.w, self.h, self.d)):
            number(size, f"{self.form.name} {name}", *self.form.rules[name])
        self.bays = self.control("bays", *self.form.rules["bays"])
        self.voids: dict[str, Bounds] = {}
        self.bearings: list[Bearing] = []
        self.datums: list[Datum] = []
        self.recesses: list[tuple[str, tuple[str, ...], bool]] = []
        self.connections: list[tuple[str, str]] = []
        self.walking_routes: dict[str, WalkingRoute] = {}

    def reserve(self, name, x0, x1, y0, y1, z0, z1, *, cross=0, width, headroom):
        space = Bounds((x0, y0, z0), (x1, y1, z1))
        if (name in self.voids or space.span(cross) + TOL < width
                or space.span(1) + TOL < headroom or space.span(2 - cross) <= TOL):
            raise ValueError(f"{self.form.name} cannot reserve {name}: need {width:g} m clear width "
                             f"and {headroom:g} m headroom")
        self.voids[name] = space

    def box(self, role, x0, x1, y0, y1, z0, z1, **params):
        if not (x1 > x0 and y1 > y0 and z1 > z0):
            raise ValueError(f"{self.form.name} has a collapsed {role}")
        return self.member(role, x0, x1, y0, y1, z0, z1,
                           construction="recessed glazing" if role == "infill" else "cast concrete",
                           finish="glazed" if role == "infill" else "exposed as-cast", **params)

    def seat(self, upper, *lowers):
        self.bearings.extend(Bearing(upper, lower) for lower in lowers)

    def station(self, axis, *slots):
        self.datums.append(Datum(axis, tuple(slots)))

    def compose(self):
        self.site_id = self.box("site_base", -self.w/2, self.w/2, 0, self.PLINTH, -self.d/2, self.d/2)
        getattr(self, {CIVIC: "civic", HOUSING: "housing", ARTS: "arts", FIELD: "field"}[self.form.name])()
        routes = {name: [asdict(p) for p in route.surfaces] for name, route in self.walking_routes.items()}
        return self.document({"grammar": self.form.rules["grammar"], "bays": self.bays,
                              "protected_spaces": {name: {"low": b.low, "high": b.high} for name, b in self.voids.items()},
                              "specified_bearings": len(self.bearings), "datum_runs": len(self.datums),
                              "connected_decks": len(self.connections),
                              **({"walking_routes": routes} if routes else {})})

    def relief(self, x0, x1, y0, y1, z):
        """Shallow horizontal casting lifts, modeled as relief, not as painted lines."""
        count = {"concept": 0, "standard": 3, "presentation": 6}[self.detail]
        for i in range(1, count + 1):
            y = y0 + (y1 - y0) * i / (count + 1)
            self.box("cast_relief", x0, x1, y, y + .055, z - .035, z)

    def facade(self, x0, x1, y0, y1, z, recess, *, count=None, back=False):
        """A solid blade bay with an inset glazed plane. Returns the bearing ribs."""
        n = self.bays if count is None else count
        pitch = (x1 - x0) / n
        rib = pitch * .15
        sign = -1 if back else 1
        def skin(offset, thickness):
            return sorted((z + sign * offset, z + sign * (offset + thickness)))
        ribs, run = [], []
        for i in range(n + 1):
            x = x0 + i * pitch
            a, b = max(x0, x - rib / 2), min(x1, x + rib / 2)
            pid = self.box("facade_rib", a, b, y0, y1, *skin(0, recess + .21), bay=i)
            ribs.append(pid)
            run.append((pid, (a + b) / 2))
            if i:
                pane = self.box("infill", x - pitch + rib / 2, a, y0, y1, *skin(recess, .14), bay=i-1)

                self.recesses.append((pane, (ribs[i-1], pid), back))
        self.station(0, *run)
        return ribs

    def civic(self):
        w, h, d, y = self.w, self.h, self.d, self.PLINTH
        assembly = self.variant == "assembly-gate"
        mouth = w * (.116 if assembly else .180) * self.vary("forum-width", .09)
        flank = mouth + .075 * w
        gate_z, back_z = -.215*d, .105*d
        dais, spring = y + .48, h * (.66 if assembly else .51) * self.vary("gateway-height", .064)
        lintel_top = spring + .134*h
        self.reserve("processional-axis", -mouth, mouth, dais, spring, -.46*d, back_z, width=7.5, headroom=6.5)
        self.box("podium", -flank, flank, y, dais, -.313*d, back_z)
        for i in range(4):
            self.box("approach_step", -flank, flank, y, y + .12*(i + 1),
                     (-.391 + .0195*i)*d, (-.391 + .0195*(i + 1))*d, tread=i)
        left = self.box("gate_pier", -flank, -mouth, dais, spring, gate_z, back_z)
        right = self.box("gate_pier", mouth, flank, dais, spring, gate_z, back_z)
        girder = self.box("transfer_girder", -flank, flank, spring, lintel_top, gate_z, back_z)
        self.seat(girder, left, right)
        self.relief(-flank, -mouth, dais, spring, gate_z)
        self.relief(mouth, flank, dais, spring, gate_z)
        self.relief(-flank, flank, spring, lintel_top, gate_z)
        # Chamber sits on the girder, with its front either recessed or nearly flush.
        chz = gate_z + d * (.086 if assembly else .035)
        ch = .91 * flank
        ribs = self.facade(-ch, ch, lintel_top, h-.055*h, chz, .48, count=3)
        back = self.box("shear_wall", -ch, ch, lintel_top, h-.055*h, back_z-.52, back_z)
        cap = self.box("council_roof", -ch, ch, h-.055*h, h, chz, back_z)
        self.seat(cap, back, *ribs)
        self.seat(back, girder)
        for rib in ribs:
            self.seat(rib, girder)
        for a, b in ((-.46*w, -flank), (flank, .46*w)):
            wing_top = spring * (.66 if assembly else .87)
            front = gate_z - (.102 if assembly else .023)*d
            frame = self.facade(a, b, y, wing_top, front, .68)
            side = self.box("shear_wall", a, b, y, wing_top, back_z-.53, back_z)
            roof = self.box("gallery_deck", a, b, wing_top, wing_top+.56, front, back_z)
            self.seat(roof, side, *frame)
            # End returns close the side wings without filling the center.
            outer = a if b < 0 else b - .52
            self.box("shear_wall", outer, outer+.52, y, wing_top, front+.89, back_z-.53)

    def housing(self):
        w, h, d, y = self.w, self.h, self.d, self.PLINTH
        levels = self.control("levels", *self.form.rules["levels"])
        rise = (h-y) / (levels + 1.25)
        if rise < 2.55:
            raise ValueError(f"{HOUSING} needs a storey rise of at least 2.55 m; increase height or reduce levels")
        a, b, rear = -.453*w, .453*w, .328*d
        f0 = -.349*d
        terrace = self.variant == "stepped"
        stepback = .31*d*self.vary("terrace-setback", .10)/(levels-1) if terrace else 0
        deck_t = min(.52, .156*rise)
        lift = 1.25 * rise
        self.reserve("undercroft-promenade", -w/2, w/2, y, y+2.8, -.104*d, .104*d,
                     cross=2, width=3.9, headroom=2.8)
        pitch = (b-a)/self.bays
        xs = [a + (i+.5)*pitch for i in range(self.bays)]
        support = []
        for z in (f0+.077*d, rear-.063*d):
            width = min(1.23, pitch*.26) * self.vary("pier-width", .18)
            run = []
            for i, x in enumerate(xs):
                pid = self.box("pilotis", x-width/2, x+width/2, y, y+lift, z-width*.68, z+width*.68, bay=i)
                support.append(pid)
                run.append((pid, x))
            self.station(0, *run)
        for k in range(levels):
            front, floor = f0 + k*stepback, y+lift+k*rise
            slab_front = f0 + max(0, k-1)*stepback
            deck = self.box("living_deck", a, b, floor, floor+deck_t, slab_front, rear, storey=k)
            self.seat(deck, *support)
            head = floor+rise if k < levels-1 else h-deck_t
            # Paired party walls mark unit boundaries and carry the slabs above.
            support = []
            run = []
            for i in range(self.bays+1):
                x = a+i*pitch
                low, high = max(a, x-.18), min(b, x+.18)
                pid = self.box("party_wall", low, high, floor+deck_t, head, front+1.56, rear-.66, bay=i, storey=k)
                run.append((pid, (low+high)/2))
                support.append(pid)
            self.station(0, *run)
            self.facade(a, b, floor+deck_t, head, front+.74, .61)
            self.facade(a, b, floor+deck_t, head, rear, .45, back=True)
            self.box("parapet", a, b, floor+deck_t, floor+deck_t+.73, slab_front, slab_front+.17, storey=k)
        roof = self.box("living_deck", a, b, h-deck_t, h, f0+(levels-1)*stepback, rear, storey=levels)
        self.seat(roof, *support)

    def arts(self):
        w, h, d, y = self.w, self.h, self.d, self.PLINTH
        stepped = self.variant == "stepped-galleries"
        gap = (.126 if stepped else .164)*w*self.vary("gallery-separation", .10)
        first = h*(.31 if stepped else .38)*self.vary("bridge-rise", .09)
        second = h*(.60 if stepped else .69)*self.vary("upper-bridge-rise", .07)
        zslots = (d*(-.19 if stepped else -.13), d*(.038 if stepped else .145))
        ys = (first, second)
        self.reserve("gallery-passage", -gap+1.22, gap-1.22, y, y+3.4, -d/2, d/2, width=5.4, headroom=3.4)
        deck_t = .57
        gallery_decks = [[], []]
        east_height = max(h*(.83 if stepped else .91), second + 2*deck_t + 2.40)
        wings = ((-.455*w, -gap, -.36*d, .34*d, h), (gap, .455*w, -.29*d, .36*d, east_height))
        for j, (a, b, f, rear, height) in enumerate(wings):
            base = y
            for k, top in enumerate((*ys, height-deck_t)):
                # In the stepped variant the upper gallery withdraws from the street.
                front = f + (.112*d if stepped and k == 2 else 0)
                frame = self.facade(a, b, base, top, front, 1.0 if k == 0 else .69)
                if k:
                    left, right = self.stair_x(j, gap, *self.STAIR_LANDING)
                    back = [self.box("shear_wall", x0, x1, base, top, rear-.62, rear)
                            for x0, x1 in ((a, left), (right, b))]
                    back.append(self.box("shear_wall", left, right, base+2.18, top, rear-.62, rear))
                else:
                    back = [self.box("shear_wall", a, b, base, top, rear-.62, rear)]
                x0 = a if j == 0 else b-.62
                self.box("shear_wall", x0, x0+.62, base, top, front+1.21, rear-.62)
                floor = self.box("gallery_deck", a, b, top, top+deck_t, front, rear, storey=k, wing=j)
                self.seat(floor, *back, *frame)
                if k < 2:
                    gallery_decks[k].append(floor)
                base = top+deck_t
        bridges = []
        for i, (elevation, z) in enumerate(zip(ys, zslots)):
            half = d * (.052 if i else .065) * self.vary(f"skyway-depth-{i}", .14)
            z0, z1 = z-half, z+half
            supports = []
            for a, b in ((-gap, -gap+1.16), (gap-1.16, gap)):
                post = self.box("bridge_pier", a, b, y, elevation-.63, z0+.15, z1-.15, crossing=i)
                seat = self.box("bridge_seat", a, b, elevation-.63, elevation, z0, z1, crossing=i)
                self.seat(seat, post)
                supports.append(seat)
            span = self.box("sky_bridge", -gap, gap, elevation, elevation+deck_t, z0, z1, crossing=i)
            bridges.append(WalkSurface(span, ((-gap, z-.66), (gap, z+.66)), elevation+deck_t, "bridge"))
            self.seat(span, *supports)
            self.connections.extend((span, landing) for landing in gallery_decks[i])
            self.station(1, *((p, elevation+deck_t/2) for p in (*gallery_decks[i], span)))
            self.reserve(f"skywalk-{i+1}", -gap, gap, elevation+deck_t, elevation+deck_t+2.12,
                         z0+.22, z1-.22, cross=2, width=2., headroom=2.12)
            for zz in (z0, z1-.21):
                self.box("parapet", -gap, gap, elevation+deck_t, elevation+deck_t+.96, zz, zz+.21, crossing=i)

        for wing, (a, b, _, rear, _) in enumerate(wings):
            self.stair_tower(wing, gap, b-a, rear, bridges, [decks[wing] for decks in gallery_decks])

    STAIR_LANDING = (.18, 1.58)

    @staticmethod
    def stair_x(wing, gap, low, high):
        return (gap+low, gap+high) if wing else (-gap-high, -gap-low)

    def stair_tower(self, wing, gap, width, rear, bridges, decks):
        """Cast open switchbacks around a spine; add a flight pair when the bay is narrow."""
        near, run_start = self.STAIR_LANDING
        room = width-.18-1.40-run_start
        base = self.PLINTH
        apron, edge = rear+.18, rear+3.14
        lanes = ((apron, rear+1.50), (rear+1.82, edge))
        name = "east" if wing else "west"

        def box(role, a, b, y0, y1, z0, z1):
            return self.box(role, *self.stair_x(wing, gap, a, b), y0, y1, z0, z1, stair_tower=name)

        def walk(member, a, b, elevation, z0, z1, zone="walk"):
            x0, x1 = self.stair_x(wing, gap, a, b)
            return WalkSurface(member, ((x0, z0), (x1, z1)), elevation, zone)

        path = [walk(self.site_id, -2.66, -1.34, base, -self.d/2, apron),
                walk(self.site_id, -2.66, near, base, apron, lanes[0][1]),
                walk(self.site_id, near, run_start, base, rear, edge)]
        for storey, (bridge, deck) in enumerate(zip(bridges, decks), 1):
            height = bridge.elevation-base
            flights = 2 if (math.ceil(height/(2*.175))-1)*.26 <= room else 4
            risers = math.ceil(height/(flights*.175))
            rise = height/(flights*risers)
            going = min(.30, room/(risers-1))
            run_end = run_start+going*(risers-1)
            # Full-height concrete between the two flights supports the tread
            # ends and acts as their inner guard; both flights stay open above.
            box("stair_spine", run_start, run_end, base, bridge.elevation+.95, lanes[0][1], lanes[1][0])
            for flight in range(flights):
                outward = flight % 2 == 0
                z0, z1 = lanes[flight % 2]
                for i in range(risers-1):
                    a = run_start+i*going if outward else run_end-(i+1)*going
                    top = base+(flight*risers+i+1)*rise
                    tread = box("stair_tread", a, a+going, top-.23, top, z0, z1)
                    path.append(walk(tread, a, a+going, top, z0, z1, "stair"))
                    guard0, guard1 = (z0-.13, z0) if outward else (z1, z1+.13)
                    box("stair_guard", a, a+going, top-.12, top+.96, guard0, guard1)
                a, b = (run_end, run_end+1.40) if outward else (near, run_start)
                top = base+(flight+1)*risers*rise
                at_gallery = flight == flights-1
                z0 = rear if at_gallery else apron
                landing = box("stair_landing", a, b, top-.24, top, z0, edge)
                path.append(walk(landing, a, b, top, z0, edge, "landing"))
                guard0, guard1 = (b, b+.13) if outward else (a-.13, a)
                box("stair_guard", guard0, guard1, top-.12, top+.96, z0, edge)
                box("stair_guard", a, b, top-.12, top+.96, edge, edge+.13)
                if not at_gallery:
                    box("stair_guard", a, b, top-.12, top+.96, z0-.13, z0)
            (_, z0), (_, z1) = bridge.plan
            branch = (walk(deck, near, run_start, bridge.elevation, rear-.62, rear, "gallery-door"),
                      walk(deck, near, run_start, bridge.elevation, z1, rear-.62),
                      walk(deck, near, run_start, bridge.elevation, z0, z1),
                      walk(deck, 0, near, bridge.elevation, z0, z1), bridge)
            self.walking_routes[f"{name}-to-skybridge-{storey}"] = WalkingRoute((*path, *branch))
            base = bridge.elevation

    def field(self):
        w, h, d, y = self.w, self.h, self.d, self.PLINTH
        sync = self.variant == "syncopated"
        gap = (.139 if sync else .166)*w*self.vary("axial-gap", .12)
        self.reserve("axis", -gap, gap, y, h*.44, -d/2, d/2, width=4.8, headroom=3.1)
        ratio = .52*self.vary("syncopation", .14) if sync else 0
        weights = [1 + (-ratio if i % 2 == 0 else ratio) for i in range(self.bays-1)]
        weighted_pitch = .738*d/sum(weights)
        zs = [-.369*d]
        for weight in weights:
            zs.append(zs[-1]+weight*weighted_pitch)
        run, left, right = [], [], []
        for i, z in enumerate(zs):
            tall = i == self.bays-1
            level = h if tall else h*(.86 if not sync else (.94 if i % 2 else .74))*self.vary(f"height-{i}", .064)
            slab = h*(.145 if not sync else (.18 if i % 2 else .126))
            spring = level-slab
            thick = .29*min(weights)*weighted_pitch*self.vary(f"depth-{i}", .19)
            outer = gap + (.114 if sync and i % 2 == 0 else .09)*w*self.vary(f"blade-{i}", .15)
            a, b = z-thick/2, z+thick/2
            lp = self.box("portal_pier", -outer, -gap, y, spring, a, b, bay=i, side="left")
            rp = self.box("portal_pier", gap, outer, y, spring, a, b, bay=i, side="right")
            upper = self.box("portal_lintel", -outer-.45, outer+.45, spring, level, a-.10, b+.10, bay=i)
            self.seat(upper, lp, rp)
            run.append((upper, z))
            left.append((lp, z))
            right.append((rp, z))
            self.relief(-outer, -gap, y, spring, a)
            self.relief(gap, outer, y, spring, a)
            self.relief(-outer-.45, outer+.45, spring, level, a-.10)
        bench_z = .24*d
        for a, b in ((-.385*w, -.326*w), (.326*w, .385*w)):
            self.box("processional_bench", a, b, y, y+.52, -bench_z, bench_z)
        for series in (run, left, right):
            self.station(2, *series)


def build_document(request):
    return Studio(request).compose()


def rule_checks(document, form):
    # Reconstruct the brief from normalized request controls. Geometry and
    # substances are read from the edited document, never its cached metadata.
    studio = Studio(document.request)
    studio.compose()
    members = {p.id: p for p in document.placements}
    obstructed = set().union(*(obstructing_members(space, members.values()) for space in studio.voids.values()))
    yield ("protected_void", not obstructed, "Required public clearances are open.",
           "A solid projects into required public clearance.", sorted(obstructed))
    broken = {b.upper for b in studio.bearings if not b.holds(members)}
    yield ("visible_bearing", not broken, "Major members seat on their specified supports.",
           "A required bearing is missing, displaced or too small.", sorted(broken))
    off_grid = set().union(*(datum.departures(members) for datum in studio.datums))
    for first, second in studio.connections:
        if first not in members or second not in members:
            off_grid.update((first, second))
            continue
        a, b = Bounds.of(members[first]), Bounds.of(members[second])
        if (abs(a.high[1]-b.high[1]) > .02 or a.overlap(b, 2) < 1.2
                or min(abs(a.high[0]-b.low[0]), abs(b.high[0]-a.low[0])) > .02):
            off_grid.update((first, second))
    yield ("bay_rhythm", not off_grid, "Repeats and deck connections meet their program datums.",
           "A member has drifted off its bay or a deck connection has separated.", sorted(off_grid))
    imitation = {p.id for p in members.values() if p.substance != SUBSTANCES.get(p.role, DEFAULT_SUBSTANCE)}
    for window, ribs, back in studio.recesses:
        if window not in members or any(rib not in members for rib in ribs):
            imitation.add(window)
            continue
        pane = Bounds.of(members[window])
        for rib in ribs:
            blade = Bounds.of(members[rib])
            reveal = blade.high[2]-pane.high[2] if back else pane.low[2]-blade.low[2]
            adjacent = min(abs(blade.high[0]-pane.low[0]), abs(pane.high[0]-blade.low[0])) <= TOL
            if reveal < .35 or blade.overlap(pane, 2) < .12 or blade.overlap(pane, 1) <= TOL or not adjacent:
                imitation.add(window)
    yield ("material_expression", not imitation, "Exposed concrete and recessed glazing express different jobs.",
           "A substrate has been imitated or the glazing no longer sits in a deep reveal.", sorted(imitation))
    if studio.walking_routes:
        solids = {p.id: Bounds.of(p) for p in members.values()}
        failed = {name: ids for name, route in studio.walking_routes.items() if (ids := route.departures(solids))}
        yield ("connected_circulation", not failed,
               "Both stair towers reach both skybridges through clear landings and gallery doors.",
               f"Missing, obstructed or discontinuous route: {', '.join(failed)}.",
               sorted(set().union(*failed.values())))
