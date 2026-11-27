"""Timberwright: bay-framed houses assembled around open rooms, Y up.

Dimensions describe the frame and ridge; eaves, steps and the porch project
beyond the frame. Only the design phase draws randomness. Changing detail
never relocates a door, a roof ridge, or the opening in the upper floor.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from core.architectural_geometry import extrude_profile, mesh_parameters

from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement


ARCHITECT = ArchitectInfo("timberwright", "1.0.0", "Timberwright Architect")
PRODUCT = "timber-house"
VARIANTS = ("longhouse", "gable-front", "twin-gable")
DETAILS = ("concept", "standard", "presentation")
CONTROLS = ("finish", "porch", "chimney")
DEFAULTS = {"width": 12, "height": 10, "depth": 10, "bays": 4, "levels": 2,
            "variant": "auto", "detail": "standard"}
FINISHES = {
    # plaster, frame, joinery, glazing, roof and weathered tile accents
    "chalk-slate": ("f2dfb9", "52392f", "956f4c", "77a8af", "445c70", "617d8b"),
    "honey-clay": ("ecc690", "5e4030", "a47746", "749995", "a0533e", "c77c54"),
    "rose-shale": ("dfae9b", "473433", "866043", "82acb6", "525564", "777483"),
}
# A finish names the roof covering as well as the palette: slate districts and
# clay-tile districts laid their roofs differently, and the house records it.
ROOF_SUBSTANCES = {
    "chalk-slate": "slate",
    "honey-clay": "clay-roof-tile",
    "rose-shale": "slate",
}
# The physical substance behind each palette key: oak frame and joinery,
# lime-plaster infill and gables, stone plinths, brick and terracotta chimney
# work, lead flashings and glazing.
MATERIAL_SUBSTANCES = {
    "plaster": "lime-plaster",
    "oak": "oak",
    "joinery": "oak",
    "glass": "window-glass",
    "stone": "limestone",
    "brick": "fired-clay-brick",
    "clay": "terracotta",
    "lead": "lead-sheet",
}
REQUIRED = {"plinth", "frame_post", "top_plate", "lime_infill", "knee_brace",
            "entry_lintel", "entry_step", "window_glass", "gable_field", "tie_beam", "principal_rafter",
            "king_post", "roof_plane", "ridge_cap"}
MEMBER_KINDS = {
    'plinth': 'foundation', 'upper_floor': 'slab', 'stair_tread': 'circulation', 'entry_step': 'circulation',
    'frame_post': 'column', 'porch_post': 'column', 'king_post': 'column',
    **dict.fromkeys(('top_plate', 'jetty_joist', 'entry_lintel', 'knee_brace', 'porch_beam',
                     'gable_tie', 'gable_strut', 'tie_beam', 'principal_rafter'), 'beam'),
    **dict.fromkeys(('ridge_cap', 'bargeboard', 'roof_plane', 'roof_tile', 'porch_roof', 'valley_flashing'), 'roof'),
}
BASE, DECK, CAP = .36, .14, .10


def list_products():
    return [{"name": PRODUCT, "category": "assembly", "architect": ARCHITECT.to_dict(),
             "architect_id": ARCHITECT.id, "architect_title": ARCHITECT.title,
             "architect_version": ARCHITECT.version,
             "description": "Half-timbered house with braced limewash bays, a walk-in entry and pitched tiled roofs",
             "defaults": dict(DEFAULTS), "variants": ["auto", "random", *VARIANTS],
             "detail_levels": list(DETAILS), "opening_modes": ["auto"], "parameters": list(CONTROLS),
             "substances": sorted(set(MATERIAL_SUBSTANCES.values()) | set(ROOF_SUBSTANCES.values())),
             "rules": {"width": [8, 24], "depth": [6, 18], "height": [5, 18], "levels": [1, 2],
                       "bays": [2, 7], "minimum_clear_bay": 1.6, "storey_height": [2.4, 4.8],
                       "roof_rise_to_run": [.45, 1.4], "finish": list(FINISHES),
                       "porch": "boolean", "chimney": "boolean", "required_roles": sorted(REQUIRED)}}]


def _number(name, value, default, low, high, *, integral=False):
    value = default if value is None else value
    if (isinstance(value, bool) or not isinstance(value, int if integral else (float, int))
            or not low <= value <= high):
        raise ValueError(f"timber-house {name} must be {'an integer ' if integral else ''}between {low} and {high}")
    return value


@dataclass(frozen=True)
class Design:
    width: float
    depth: float
    height: float
    bays: int
    levels: int
    variant: str
    detail: str
    finish: str
    porch: bool
    chimney: bool
    rise: float
    jetty: float
    door_bay: int
    cadence: int

    @property
    def eave(self):
        return self.height - CAP - DECK - self.rise

    @property
    def storey(self):
        return (self.eave - BASE) / self.levels

    @classmethod
    def resolve(cls, request, player=None):
        """``player``, when given, sets a lowest storey that fits its door.

        A frame too low for that storey is raised just enough, and the new
        height is what the resolved request records.
        """
        rng = random.Random(request.seed)
        automatic = request.variant in ("auto", "random")
        if (not automatic and request.variant not in VARIANTS) or request.detail not in DETAILS:
            raise ValueError("unknown timber-house variant or detail level")
        if request.openings != "auto":
            raise ValueError("timber-house uses automatic framed openings")
        params = request.parameters
        if set(params) - set(CONTROLS):
            raise ValueError(f"unknown timber-house parameters: {sorted(set(params) - set(CONTROLS))}")
        finish = params.get("finish", rng.choice(tuple(FINISHES)))
        if not isinstance(finish, str) or finish not in FINISHES:
            raise ValueError(f"timber-house finish must be one of {', '.join(FINISHES)}")
        porch = params.get("porch", rng.random() < .70)
        chimney = params.get("chimney", rng.random() < .85)
        if not isinstance(porch, bool) or not isinstance(chimney, bool):
            raise ValueError("timber-house porch and chimney must be booleans")
        w = _number("width", request.width, DEFAULTS['width'], 8, 24)
        d = _number("depth", request.depth, DEFAULTS['depth'], 6, 18)
        levels = _number("levels", request.levels, DEFAULTS['levels'], 1, 2, integral=True)
        bays = _number("bays", request.bays, DEFAULTS['bays'], 2, 7, integral=True)
        h = _number("height", request.height, DEFAULTS['height'] if levels == 2 else 7, 5, 18)
        preferred_jetty = round(rng.uniform(.20, .34), 3) if levels == 2 else 0.0
        min_jetty = .20 if levels == 2 else 0.0
        width_limit = (w - (1.6 + .20)*bays)/2
        if width_limit < min_jetty - 1e-9:
            raise ValueError("timber-house needs at least 1.6 clear width between bay posts")
        min_storey = 2.4
        if player is not None:
            # Door clear = storey - deck (two levels) - sole rail - lintel margin.
            need = player.clearances().door_height + .16 + .13 + (DECK if levels == 2 else 0)
            min_storey = max(min_storey, math.ceil(need * 100) / 100)
        while True:
            feasible = cls._feasible(automatic, request.variant, w, d, h, levels, min_storey,
                                     preferred_jetty, width_limit, min_jetty)
            if feasible or min_storey <= 2.4 or h >= 18:
                break
            h = min(18, round(h + .1, 2))
        if not feasible and min_storey > 2.4:
            raise ValueError(f"timber-house cannot fit a door for a '{player.name}' (a {min_storey:.2f} m storey) "
                             "under 18 m; use fewer levels or a smaller roof")
        if not feasible:
            raise ValueError("timber-house dimensions cannot fit both habitable storeys and a pitched roof")
        # One draw even for a pinned plan: resolved requests replay like their Auto originals.
        variant, jetty, run, low, high = feasible[int(rng.random()*len(feasible))]
        rise = max(low, min(high, run * rng.uniform(.73, 1.12)))
        door = rng.choice(((bays - 1) // 2, bays // 2))
        return cls(w, d, h, bays, levels, variant, request.detail, finish, porch, chimney,
                   rise, jetty, door, rng.randrange(6))

    @staticmethod
    def _feasible(automatic, requested, w, d, h, levels, min_storey, preferred_jetty, width_limit, min_jetty):
        available = h - BASE - DECK - CAP
        feasible = []
        for variant in VARIANTS if automatic else (requested,):
            run = {'longhouse': d/2, 'gable-front': w/2, 'twin-gable': w/4}[variant]
            # A seed may choose less projection, but must never consume the
            # clearance between posts or either landing of the straight stair.
            jetty, max_storey = 0.0, 4.8
            if levels == 2:
                shortest_storey = max(min_storey, (available - 1.4*run)/levels)
                depth_limit = (d - 1.9 - 1.24*shortest_storey)/2
                jetty = min(preferred_jetty, width_limit, depth_limit)
                if jetty < min_jetty - 1e-9:
                    continue
                max_storey = min(max_storey, (d - 2*jetty - 1.9)/1.24)
            low = max(.45 * run, available - max_storey * levels)
            high = min(1.4 * run, available - min_storey * levels)
            if low <= high + 1e-9:
                feasible.append((variant, jetty, run, min(low, high), high))
        return feasible


def random_controls(rng=None, variant='auto'):
    """Buildable GUI/series dimensions. Honor an explicit plan; never retry random bad boxes."""
    rng = rng if rng is not None else random.Random()
    roof = rng.choice(VARIANTS) if variant in ('auto', 'random') else variant
    bays = rng.randint(2, 7)
    width = rng.randint(max(9, 2*bays+1), min(22, 4*bays+1))
    depth = rng.randint(8, 14)
    levels = rng.randint(1, 2)
    run = {'longhouse': depth/2, 'gable-front': width/2, 'twin-gable': width/4}[roof]
    height = min(18, round(BASE+DECK+CAP + levels*rng.uniform(2.9, 3.7) + run*rng.uniform(.67, 1.0)))
    return dict(width=width, depth=depth, height=height, levels=levels, bays=bays,
                detail=rng.choice(DETAILS), variant=roof)


def _palette(finish):
    colors = dict(zip(('plaster', 'oak', 'joinery', 'glass', 'roof', 'weathering'), FINISHES[finish]))
    colors.update(stone='928579', brick='9b604a', clay='bc7959', lead='4c5458')
    palette = {name: (*bytes.fromhex(value), 255) for name, value in colors.items()}
    for i in range(5):
        palette[f'tile{i}'] = tuple(round(a + (b - a) * i / 7)
                                    for a, b in zip(palette['roof'], palette['weathering']))
    return palette


class House:
    def __init__(self, design, player=None):
        self.d = design
        # The player the house is for; None keeps the historic door sizes.
        self.player = player
        self.palette = _palette(design.finish)
        self.roof_substance = ROOF_SUBSTANCES[design.finish]
        self.placements = []
        self.counts = Counter()
        self.root = None

    def substance_for(self, material):
        if material == "roof" or material == "weathering" or material.startswith("tile"):
            return self.roof_substance
        return MATERIAL_SUBSTANCES[material]

    def add(self, role, material, position, size, payload=None):
        index = self.counts[role]
        self.counts[role] += 1
        pid = f'{PRODUCT}.{role}.{index:03d}'
        self.placements.append(Placement(
            id=pid, kind=MEMBER_KINDS.get(role, role), role=role, block_type='CUBE' if payload is None else 'MESH',
            position=tuple(round(v, 8) for v in position), dimensions=tuple(round(v, 8) for v in size),
            material=self.palette[material], parent_id=self.root,
            parameters={'product': PRODUCT, 'material_role': material, **(payload or {})},
            substance=self.substance_for(material)))
        return pid

    def box(self, role, material, x0, x1, y0, y1, z0, z1):
        return self.add(role, material, ((x0+x1)/2, (y0+y1)/2, (z0+z1)/2), (x1-x0, y1-y0, z1-z0))

    def panel(self, role, material, side, a, b, lo, hi, normal, thick):
        """Rectangle in an elevation. Side elevations run along Z, others along X."""
        near, far = normal - thick / 2, normal + thick / 2
        if side:
            return self.box(role, material, near, far, lo, hi, a, b)
        return self.box(role, material, a, b, lo, hi, near, far)

    def profile(self, role, material, side, points, normal, thick):
        mesh = extrude_profile(points, thick)
        if side:   # Rotate the local XY section onto ZY, preserving face winding.
            mesh.apply_transform([[0, 0, -1, normal], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        else:
            mesh.apply_translation((0, 0, normal))
        return self.add(role, material, *mesh_parameters(mesh))

    def slope(self, role, material, side, a, ya, b, yb, normal, depth, gauge):
        return self.profile(role, material, side, [(a, ya), (b, yb), (b, yb+gauge), (a, ya+gauge)], normal, depth)

    def compose(self):
        d = self.d
        w0, d0 = d.width - 2*d.jetty, d.depth - 2*d.jetty
        self.root = self.box('plinth', 'stone', -w0/2-.16, w0/2+.16, 0, BASE, -d0/2-.16, d0/2+.16)
        for level in range(d.levels):
            w, depth = (w0, d0) if level == 0 else (d.width, d.depth)
            bottom = BASE + level*d.storey
            top = bottom + d.storey - (DECK if level < d.levels-1 else 0)
            # The front and back frames own the corner posts; side plates butt into them.
            for side in (False, True):
                span, across = (depth, w) if side else (w, depth)
                bays = max(2, round(depth/2.9)) if side else d.bays
                for sign in (-1, 1):
                    self.elevation(side, sign, span, sign*across/2, bays, level, bottom, top)
        if d.levels == 2:
            self.floor_and_stair(w0, d0)
        door_x = -w0/2 + (d.door_bay+.5)*w0/d.bays
        self.entrance(door_x, -d0/2, w0/d.bays)
        side = d.variant == 'longhouse'
        span, length = (d.depth, d.width) if side else (d.width, d.depth)
        roofs = 2 if d.variant == 'twin-gable' else 1
        # Each interior bent aligns with the perimeter posts. Both roofs of a
        # twin-gable house bear on a common tie below the valley.
        bents = d.bays if side else max(2, round(length/2.9))
        stations = [-length/2 + i*length/bents for i in range(1, bents)]
        for station in stations:
            self.panel('tie_beam', 'oak', side, -span/2, span/2, d.eave-.20, d.eave, station, .20)
        for index in range(roofs):
            run = span / (2*roofs)
            center = -span/2 + (2*index+1)*run
            self.roof(side, center, run, length, stations, index == 0, index == roofs-1)
        if roofs == 2:
            base = d.eave + DECK
            lip = base + .18*d.rise/run
            self.profile('valley_flashing', 'lead', side, [(-.18, lip-.01), (0, base-.01), (.18, lip-.01),
                         (.18, lip+.035), (0, base+.035), (-.18, lip+.035)], 0, length+.92)
        if d.chimney:
            self.chimney(side, span/(2*roofs), center)

    def elevation(self, side, sign, span, n, bays, level, bottom, top):
        d = self.d
        post, rail = .20, .16
        pitch = span/bays
        inset = post/2 if side else -post/2
        self.panel('top_plate', 'oak', side, -span/2+inset, span/2-inset, top-rail, top, n, post)
        for i in range(int(side), bays + 1 - int(side)):
            u = -span/2 + i*pitch
            self.panel('frame_post', 'oak', side, u-post/2, u+post/2, bottom, top-rail, n, post)
        for i in range(bays):
            a = -span/2 + i*pitch + post/2
            b = a + pitch - post
            c = (a+b)/2
            crown = top - rail
            outer = n + sign*.06
            door = not side and sign == -1 and level == 0 and i == d.door_bay
            blind = bays > 2 and (i + level + int(side) + sign + d.cadence) % 4 == 0
            if door or not blind:
                width = min(1.45 if door else 1.85, (b-a)*(.65 if door else .64))
                head_room = 2.27
                if door and self.player is not None:
                    # Grow the door to the player's clearance, as far as the
                    # bay and storey allow; the playability checks report
                    # a frame too small to fit it.
                    clear = self.player.clearances()
                    width = max(width, min(clear.door_width, (b-a) - .2))
                    head_room = max(head_room, clear.door_height)
                left, right = c-width/2, c+width/2
                if door:
                    self.door_width = width
                sill = bottom if door else bottom+(top-bottom)*.34
                head = min(bottom+head_room, crown-.13) if door else bottom+(top-bottom)*.77
                for lo, hi in ((a, left-.09), (right+.09, b)):
                    self.panel('sole_plate', 'oak', side, lo, hi, bottom, bottom+rail, n, post)
                    self.panel('lime_infill', 'plaster', side, lo, hi, bottom+rail, crown, n, .12)
                for lo, hi in ((left-.09, left), (right, right+.09)):
                    self.panel('opening_jamb', 'joinery', side, lo, hi, bottom, crown, n, post)
                self.panel('entry_lintel' if door else 'window_head', 'oak', side, left, right, head, head+.12, n, .23)
                self.panel('lime_infill', 'plaster', side, left, right, head+.12, crown, n, .12)
                if door:
                    continue
                self.panel('window_glass', 'glass', side, left, right, sill, head, n, .045)
                self.panel('window_sill', 'joinery', side, left, right, sill-.12, sill, outer, .26)
                self.panel('sole_plate', 'oak', side, left, right, bottom, bottom+rail, n, post)
                self.panel('lime_infill', 'plaster', side, left, right, bottom+rail, sill-.12, n, .12)
                self.panel('mullion', 'joinery', side, c-.035, c+.035, sill, head, outer, .09)
                if d.detail != 'concept':
                    bar = sill + .64*(head-sill)
                    self.panel('transom', 'joinery', side, left, right, bar, bar+.055, outer, .095)
                low, high, a, b = bottom+rail, sill-.23, left, right
            else:
                self.panel('sole_plate', 'oak', side, a, b, bottom, bottom+rail, n, post)
                self.panel('lime_infill', 'plaster', side, a, b, bottom+rail, crown, n, .12)
                low, high = bottom+rail, crown-.11
            # Matched knee braces in the spandrels. Reverse the motif on alternating bays.
            ya, yb = (low, high) if (i+level+d.cadence) % 2 else (high, low)
            self.slope('knee_brace', 'oak', side, a, ya, c, yb, outer, .11, .11)
            self.slope('knee_brace', 'oak', side, c, yb, b, ya, outer, .11, .11)

    def floor_and_stair(self, w0, d0):
        d = self.d
        upper = BASE + d.storey
        x0, x1 = -w0/2+.46, -w0/2+1.56
        if self.player is not None:
            # Widen the flight and the hole above it to the player's door.
            x1 = max(x1, x0 + self.player.clearances().door_width)
        z0, z1 = -.62*d.storey, .62*d.storey
        # Four non-overlapping floor pieces leave an actual hole above the flight.
        for a, b, c, e in ((-d.width/2-.1, x0, -d.depth/2-.1, d.depth/2+.1),
                           (x1, d.width/2+.1, -d.depth/2-.1, d.depth/2+.1),
                           (x0, x1, -d.depth/2-.1, z0), (x0, x1, z1, d.depth/2+.1)):
            self.box('upper_floor', 'joinery', a, b, upper-DECK, upper, c, e)
        steps = math.ceil(d.storey/.195)
        for i in range(steps):
            self.box('stair_tread', 'joinery', x0, x1, BASE, BASE+(i+1)*d.storey/steps,
                     z0+i*(z1-z0)/steps, z0+(i+1)*(z1-z0)/steps)
        for x in (x0-.05, x1+.05):
            for z in (z0, 0, z1):
                self.box('guard_post', 'oak', x-.04, x+.04, upper, upper+.82, z-.04, z+.04)
            self.box('guard_rail', 'joinery', x-.05, x+.05, upper+.82, upper+.90, z0-.05, z1+.05)
        self.box('guard_rail', 'joinery', x0, x1, upper+.82, upper+.90, z0-.05, z0+.05)
        for sign in (-1, 1):
            for i in range(d.bays+1):
                x = -w0/2 + i*w0/d.bays
                n = sign*(d0/2 + d.jetty/2)
                self.panel('jetty_joist', 'oak', False, x-.085, x+.085, upper-DECK-.16, upper-DECK,
                           n, d.jetty+.27)

    def entrance(self, x, front, bay_width):
        width = min(1.45, (bay_width-.2)*.65)
        if self.player is not None:
            width = max(width, getattr(self, 'door_width', width))
        half = width/2 + .18
        for i in range(2):
            self.box('entry_step', 'stone', x-half, x+half, 0, BASE*(i+1)/2,
                     front-.16-(2-i)*.34, front-.16-(1-i)*.34)
        if not self.d.porch:
            return
        top = BASE + self.d.storey - (DECK if self.d.levels == 2 else 0)
        head = min(BASE+2.27, top-.29)
        if self.player is not None:
            # The porch beam sits over open ground; keep the player's door
            # height under it.
            head = min(max(head, self.player.clearances().door_height + .04), top-.29)
        edge, wall = front-1.38, front-.06
        eave = head+.06
        for u in (x-half-.08, x+half+.08):
            self.box('porch_foot', 'stone', u-.13, u+.13, 0, BASE, edge, edge+.26)
            self.box('porch_post', 'oak', u-.08, u+.08, BASE, eave-.10, edge+.05, edge+.21)
        self.box('porch_beam', 'joinery', x-half-.20, x+half+.20, eave-.10, eave+.06, edge, edge+.26)
        self.slope('porch_roof', 'roof', True, edge-.12, eave, wall, eave+.31, x, 2*half+.58, .10)

    def roof(self, side, c, run, length, stations, first, last):
        d = self.d
        slope = d.rise/run
        peak = d.eave + d.rise
        full = length + .84
        for station in stations:
            self.panel('king_post', 'oak', side, c-.09, c+.09, d.eave, peak-.17, station, .18)
        for sign, outside in ((-1, first), (1, last)):
            reach = run + (.40 if outside else 0)
            edge = c + sign*reach
            eave = peak - slope*reach
            self.slope('roof_plane', 'roof', side, edge, eave, c, peak, 0, full, DECK)
            for station in stations:
                self.slope('principal_rafter', 'oak', side, c+sign*run, d.eave-.17, c, peak-.17,
                           station, .18, .17)
            for end in (-1, 1):
                self.slope('bargeboard', 'joinery', side, edge, eave-.19, c, peak-.19,
                           end*(full/2-.055), .11, .19)
            if d.detail == 'presentation':
                self.tiles(side, c, peak, sign, slope, reach if outside else reach-.16, full)
        cap = peak + DECK
        self.profile('ridge_cap', 'weathering', side,
                     [(c-.13, cap-.13*slope), (c, cap), (c+.13, cap-.13*slope),
                      (c+.13, cap-.13*slope+.06), (c, cap+CAP), (c-.13, cap-.13*slope+.06)], 0, full+.06)
        for end in (-1, 1):
            n = end*length/2
            self.profile('gable_field', 'plaster', side, [(c-run, d.eave), (c+run, d.eave), (c, peak)], n, .14)
            n += end*.075
            self.panel('gable_tie', 'oak', side, c-run, c+run, d.eave, d.eave+.15, n, .19)
            for fraction in ((0,) if d.detail == 'concept' else (-.55, 0, .55)):
                u = c+fraction*run
                self.panel('gable_stud', 'oak', side, u-.075, u+.075, d.eave+.15,
                           peak-(abs(u-c)+.075)*slope, n, .15)
            for sign in (-1, 1):
                self.slope('gable_strut', 'oak', side, c+sign*.08, d.eave+.15,
                           c+sign*.7*run, d.eave+.30*d.rise-.12, n, .15, .12)

    def tiles(self, side, c, peak, sign, slope, reach, length):
        # Stagger the joints; end tiles are cut to the verge, not stretched.
        rows = math.ceil(reach/.78)
        columns = math.ceil(length/.95)
        for row in range(rows):
            r0, r1 = row*reach/rows, (row+1)*reach/rows-.012
            cuts = [-length/2, *(-length/2+(i+.5*(row % 2))*length/columns for i in range(columns)
                                  if 0 < i+.5*(row % 2) < columns), length/2]
            for col, (a, b) in enumerate(zip(cuts, cuts[1:])):
                shade = (row*7 + col*col + sign*col + self.d.cadence) % 5
                self.slope('roof_tile', f'tile{shade}', side, c+sign*r0, peak-slope*r0+DECK,
                           c+sign*r1, peak-slope*r1+DECK, (a+b)/2, b-a-.014, .035)

    def chimney(self, side, run, center):
        d = self.d
        x, z = .36*d.width, .27*d.depth
        nearest = max(0, abs((z if side else x)-center)-.34)
        top = d.eave + d.rise*(1-nearest/run) + DECK + .67
        self.box('chimney_stack', 'brick', x-.34, x+.34, BASE, top, z-.34, z+.34)
        for drop in (.32, .11):
            self.box('chimney_course', 'clay', x-.39, x+.39, top-drop, top-drop+.07, z-.39, z+.39)
        self.box('chimney_coping', 'stone', x-.43, x+.43, top, top+.09, z-.43, z+.43)
        self.box('chimney_pot', 'clay', x-.19, x+.19, top+.09, top+.36, z-.19, z+.19)
        self.box('flue', 'lead', x-.12, x+.12, top+.36, top+.37, z-.12, z+.12)


def build_document(request: BuildRequest) -> BuildDocument:
    player_of = getattr(request, "player_profile", None)
    player = player_of() if callable(player_of) else None
    d = Design.resolve(request, player)
    house = House(d, player)
    house.compose()
    resolved = replace(request, architect=ARCHITECT.id, product=PRODUCT, style='timber', variant=d.variant,
                       width=d.width, height=d.height, depth=d.depth, bays=d.bays, levels=d.levels,
                       parameters={'finish': d.finish, 'porch': d.porch, 'chimney': d.chimney})
    required = REQUIRED | ({'upper_floor', 'stair_tread', 'jetty_joist'} if d.levels == 2 else set())
    return BuildDocument(architect=ARCHITECT, request=resolved, placements=house.placements, metadata={
        'timber_rules': {'storey_height': round(d.storey, 6), 'roof_rise': round(d.rise, 6),
                         'jetty': d.jetty, 'entry_bay': d.door_bay+1, 'brace_cadence': d.cadence},
        'invariants': {'require_ground_contact': True, 'required_roles': sorted(required)},
    })


def _profile_overlaps_box(mesh, center, half):
    """Narrow the AABB test for pitched rafters, eaves, etc.

    All authored profiles are extruded along X or Z. In the other two axes,
    a convex profile and rectangle can also separate on a profile edge normal.
    Project the *whole* mesh, so this stays conservative for a concave profile.
    """
    if mesh is None or mesh.is_empty:
        return True  # Mesh-health checks diagnose missing geometry separately.
    normals = mesh.face_normals
    projection = (mesh.vertices - center) @ normals.T
    radius = np.abs(normals) @ half
    return not np.any((projection.min(axis=0) >= radius-1e-5) | (projection.max(axis=0) <= -radius+1e-5))


def timber_checks(document, scene):
    """Check authored geometry on inspection, independently of user-editable design metadata."""
    from .quality import _check, _envelope_connectivity_checks, _placement_bounds

    placements = document.placements
    roles = {p.role for p in placements}
    required = REQUIRED | ({'upper_floor', 'stair_tread', 'jetty_joist'} if document.request.levels == 2 else set())
    checks = [_check('timber.required_roles', required <= roles,
                     'House has its defining frame, envelope and access members.',
                     'House is missing defining members.', expected=sorted(required), observed=sorted(roles))]
    envelopes = [(p, *_placement_bounds(p)) for p in placements]
    feet = [hi[1] for p, lo, hi in envelopes if p.role == 'plinth']
    lintels = [(lo, hi) for p, lo, hi in envelopes if p.role == 'entry_lintel']
    blocked = []
    doorway = len(feet) == len(lintels) == 1
    eps = 1e-5
    if doorway:
        lo, hi = lintels[0]
        doorway = lo[1] - feet[0] >= 1.95 and hi[0] - lo[0] >= .9
        lower, upper = np.array((lo[0], feet[0], lo[2])), np.array((hi[0], lo[1], hi[2]))
        center, half = (lower+upper)/2, (upper-lower)/2
        meshes = {b.parameters.get('primitive_id'): b.mesh for b in scene.blocks}
        blocked = [p.id for p, a, b in envelopes if min(b[0], hi[0]) - max(a[0], lo[0]) > eps
                   and min(b[1], lo[1]) - max(a[1], feet[0]) > eps
                   and min(b[2], hi[2]) - max(a[2], lo[2]) > eps
                   and (p.block_type != 'MESH' or _profile_overlaps_box(meshes.get(p.id), center, half))]
    checks.append(_check('timber.entry_clear', doorway and not blocked,
                         'The front entrance has a clear passage beneath its lintel.',
                         'The entrance is missing, obstructed or undersized.', primitive_ids=blocked))
    if document.request.levels == 2:
        steps = sorted(((lo, hi) for p, lo, hi in envelopes if p.role == 'stair_tread'), key=lambda s: s[1][1])
        floors = [(lo, hi) for p, lo, hi in envelopes if p.role == 'upper_floor']
        clear = bool(steps and floors and feet)
        if clear:
            first, last = steps[0][1], steps[-1][1]
            clear = (0 < first[1]-feet[0] <= .20 and
                     all(0 < b[1][1]-a[1][1] <= .20 and abs(b[0][2]-a[1][2]) < eps
                         for a, b in zip(steps, steps[1:])) and
                     any(abs(lo[2]-last[2]) < eps and abs(hi[1]-last[1]) < eps
                         and lo[0] <= steps[-1][0][0]+eps and hi[0] >= last[0]-eps for lo, hi in floors) and
                     not any(min(b[0], hi[0])-max(a[0], lo[0]) > eps and
                             min(b[2], hi[2])-max(a[2], lo[2]) > eps for lo, hi in steps for a, b in floors))
        checks.append(_check('timber.stair_access', clear, 'The stair ascends through a clear well onto the upper floor.',
                             'The stair does not reach a clear upper landing.'))
    checks.extend(_envelope_connectivity_checks(document, 'timber'))
    return checks
