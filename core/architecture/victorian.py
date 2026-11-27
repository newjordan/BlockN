"""Victorian Architect: brick terraces, bay fronts and Gothic Revival villas.

British domestic building from about 1840 to 1900, composed from the
tradition's own rules: one-brick load-bearing walls laid in bond under Welsh
slate pitched between 30 and 45 degrees on the terraces and 50 and 60 on the
villa, timber joist floors spanning front to back between the party walls,
sash windows seated in reveals under gauged-brick arches, and a chimney
breast and stack on every party wall. Party walls and gables are carried up
through the roof as raked, coped parapets. Only the design phase draws
randomness: changing ``detail`` never moves a party wall, a door, a joist or
a stack, only the coursing of the bricks between them.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
import trimesh

from core.architectural_geometry import extrude_profile, mesh_parameters

from .masonry import course_profile_tiles, fill_box, iter_stack
from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement


ARCHITECT = ArchitectInfo("victorian", "1.0.0", "Victorian Architect")
PRODUCT = "victorian-terrace"
VARIANTS = ("two-up-two-down", "bay-fronted", "gothic-villa")
DETAILS = ("concept", "standard", "presentation")
CONTROLS = ("finish",)
DEFAULTS = {"width": 32, "height": 9.2, "depth": 7.5, "bays": 6, "levels": 2,
            "variant": "auto", "detail": "standard"}
FINISHES = {
    # brick, band, dressings, slate, sash, glass, door, rail, tile, pot
    "london-stock": ("c7ae7e", "9b5a45", "bcb09a", "5b6570", "ece5d3",
                     "9fc9d2", "4f5d52", "3a3e40", "a35c42", "8a4a34"),
    "red-engineering": ("9c4a34", "59616e", "b8ad9a", "5b6570", "ece5d3",
                        "9fc9d2", "513f38", "2e3234", "a35c42", "8a4a34"),
    "polychrome": ("a34f38", "d9b96a", "b0a28c", "565e6a", "f2ead8",
                   "9fc9d2", "4a3f5c", "3a3e40", "8a6f56", "7c4630"),
}
# The physical brick behind each finish: London stock and polychrome work are
# common fired clay; the red-engineering finish is Class B engineering brick
# with blue-brick banding. Dressings are limestone or sandstone per finish.
FINISH_BRICK = {
    "london-stock": "fired-clay-brick",
    "red-engineering": "engineering-brick",
    "polychrome": "fired-clay-brick",
}
FINISH_DRESSING = {
    "london-stock": "limestone",
    "red-engineering": "sandstone",
    "polychrome": "sandstone",
}
MATERIAL_SUBSTANCES = {
    "slate": "slate", "sash": "softwood", "joist": "softwood", "board": "softwood",
    "rafter": "softwood", "plate": "softwood", "bargeboard": "softwood",
    "glass": "window-glass", "door": "softwood", "rail": "wrought-iron",
    "mortar": "lime-mortar", "pot": "terracotta", "tile": "glazed-tile",
    "step": "limestone", "coping": "limestone",
}

# Fixed construction modules (detail changes brick coursing, never these).
BRICK_T = 0.22          # one-brick party, front and rear walls
BRICK_BASE_T = 0.33     # one-and-a-half brick at the base of three-storey fronts
PLINTH_H = 0.45         # coursed plinth under every street-facing wall
JOIST_W = 0.10          # floor joist width
JOIST_H = 0.22          # floor joist depth, pocketed into the walls
JOIST_PITCH = 0.45      # target joist spacing along the frontage
JOIST_POCKET = 0.15     # how deep a joist end beds into its wall pocket
JOIST_SETBACK = 0.48    # joist zone setback from a party face (the hearth strip)
BOARD_T = 0.024         # floorboard thickness
BOARD_W = 0.135         # floorboard width, laid across the joists
PLATE_T = 0.10          # wall plate depth on the wall heads
BED_T = 0.02            # lime-mortar bedding under plates and copings
SLATE_T = 0.04          # slate covering thickness
SLATE_OVERHANG = 0.12   # slate courses project past the wall faces
RAFTER_T = 0.09         # common rafter depth
RAFTER_PITCH = 0.75     # target rafter spacing along the run
RING_T = 0.11           # gauged-brick arch ring, one brick on edge
PARAPET_UPSTAND = 0.38  # raked parapet above the slates; the London Acts' 15 in
PARAPET_COPING_T = 0.09 # brick-on-edge coping on the raked parapet
PARAPET_COPING_W = 0.32 # coping projects past both wall faces
VERGE_BOARD_T = 0.08    # bargeboard thickness along the ridge axis
VERGE_LAP = 0.16          # bargeboard laps this far below the slate top
VERGE_BOARD_DEPTH = (PARAPET_UPSTAND - PARAPET_COPING_T) + VERGE_LAP
STACK_W = 0.84          # chimney stack plan, two flues
STACK_H = 1.35          # stack shaft above the fire-break parapet
POT_H = 0.35            # terracotta pot height
POT_R = 0.10            # terracotta pot radius
COPING_T = 0.09         # stone stack coping
DOOR_W = 0.84           # door leaf width
DOOR_H = 2.05           # door leaf height under its arch
SASH_HALF = 0.575       # half the sash light width
REVEAL = 0.09           # sash frame setback from the outer face
OUTSHUT_D = 2.1         # rear outshut depth
OUTSHUT_DROP = 0.9      # outshut eave below the ground-storey top
GARDEN_D = 3.2          # front garden between street and front wall
BAY_PROJ = 0.9          # bay window projection from the front wall
BAY_FACE = 1.6          # bay front face width
EPS = 1e-6

# Measured period bands, published in the catalog and enforced from geometry.
FRONTAGE_BAND = {
    "two-up-two-down": (4.0, 5.15),
    "bay-fronted": (5.1, 7.15),
    "gothic-villa": (6.8, 9.4),
}
HOUSE_RANGE = {
    "two-up-two-down": (4, 10),
    "bay-fronted": (3, 8),
    "gothic-villa": (1, 1),
}
WIDTH_BAND = {
    "two-up-two-down": (16.4, 51.5),
    "bay-fronted": (15.3, 57.2),
    "gothic-villa": (6.8, 9.4),
}
DEPTH_BAND = {
    "two-up-two-down": (5.8, 8.2),
    "bay-fronted": (6.0, 9.0),
    "gothic-villa": (6.8, 9.5),
}
PITCH_BAND = (30.0, 45.0)
VILLA_PITCH_BAND = (50.0, 60.0)
ARCH_RISE_RATIO = 0.14  # segmental gauged arch rise to outer span

REQUIRED = {
    "front_wall", "back_wall", "party_wall", "plinth", "joist", "floorboard",
    "plate", "rafter", "slate_course", "ridge_cap", "gauged_arch",
    "sash_stile", "sash_glass", "sill", "door_leaf", "entry_step",
    "chimney_breast", "stack", "stack_coping", "chimney_pot",
}
VARIANT_REQUIRED = {
    "two-up-two-down": REQUIRED | {"outshut_wall", "parapet_coping"},
    "bay-fronted": REQUIRED | {"bay_wall", "bay_plinth", "garden_wall",
                                "railing_bar", "path_tile", "parapet_coping"},
    "gothic-villa": (REQUIRED | {"house_wall", "bargeboard", "porch_pier",
                                 "porch_arch", "pointed_window", "parapet_coping"})
                   - {"front_wall", "back_wall", "party_wall"},
}
KINDS = {
    "front_wall": "wall", "back_wall": "wall", "party_wall": "wall",
    "outshut_wall": "wall", "bay_wall": "wall", "bay_flank": "wall",
    "bay_plinth": "foundation", "house_wall": "wall",
    "garden_wall": "wall", "garden_pier": "wall", "spandrel": "wall",
    "chimney_breast": "wall", "plinth": "foundation",
    "gauged_arch": "arch", "porch_arch": "arch", "pointed_window": "window",
    "porch_pier": "wall", "sill": "sill", "sash_stile": "window", "sash_rail": "window",
    "sash_glass": "window", "door_leaf": "door", "door_jamb": "door",
    "entry_step": "circulation", "joist": "beam", "floorboard": "slab",
    "plate": "beam", "mortar_bed": "bedding", "rafter": "roof",
    "slate_course": "roof", "ridge_cap": "roof", "bargeboard": "roof",
    "parapet_coping": "coping",
    "stack": "chimney", "stack_coping": "chimney", "chimney_pot": "chimney",
    "railing_bar": "ironwork", "railing_rail": "ironwork", "path_tile": "slab",
}
SLATE_ROW = {"concept": 1.0, "standard": 0.75, "presentation": 0.5}
BAND_MODULE = 0.9       # polychrome banding pitch on the villa


def list_products():
    finishes = [(name, *roles) for name, roles in FINISHES.items()]
    substances = set()
    for finish in FINISHES:
        substances.add(FINISH_BRICK[finish])
        substances.add(FINISH_DRESSING[finish])
        substances.update(MATERIAL_SUBSTANCES.values())
    return [{"name": PRODUCT, "category": "assembly", "architect": ARCHITECT.to_dict(),
             "architect_id": ARCHITECT.id, "architect_title": ARCHITECT.title,
             "architect_version": ARCHITECT.version,
             "description": "Victorian brick terraces and Gothic Revival villas: "
                            "bonded load-bearing walls, joist floors, slated roofs, "
                            "sash windows and party-wall stacks",
             "defaults": dict(DEFAULTS), "variants": ["auto", "random", *VARIANTS],
             "detail_levels": list(DETAILS), "opening_modes": ["auto"],
             "parameters": list(CONTROLS), "finishes": finishes,
             "substances": sorted(substances),
             "rules": {"width": [16.4, 57.2], "depth": [5.8, 9.5],
                       "levels": [2, 3],
                       "houses": {name: list(HOUSE_RANGE[name]) for name in VARIANTS},
                       "frontage": {name: list(FRONTAGE_BAND[name]) for name in VARIANTS},
                       "roof_pitch_deg": list(PITCH_BAND),
                       "villa_pitch_deg": list(VILLA_PITCH_BAND),
                       "wall_thickness": [BRICK_T, BRICK_BASE_T],
                       "arch_rise_to_span": [0.10, 0.20],
                       "finish": list(FINISHES),
                       "required_roles": sorted(set().union(*VARIANT_REQUIRED.values())),
                       "scope": "architectural visualization and modular asset generation"}}]


def _palette(finish):
    colors = dict(zip(("brick", "band", "dressings", "slate", "sash", "glass",
                       "door", "rail", "tile", "pot"), FINISHES[finish]))
    # Shared members outside the finish's palette: lime bedding, stone steps
    # and copings, and the raw softwood of the structural floors and roofs.
    colors.update(mortar="d9d4c5", step=colors["dressings"], coping=colors["dressings"],
                  joist="b08a5f", board="c8a87e", rafter="a9825c",
                  plate="b08a5f", bargeboard=colors["sash"])
    return {name: (*bytes.fromhex(value), 255) for name, value in colors.items()}


def _substances(finish):
    out = dict(MATERIAL_SUBSTANCES)
    out["brick"] = FINISH_BRICK[finish]
    out["band"] = FINISH_BRICK[finish]
    out["dressings"] = FINISH_DRESSING[finish]
    return out


def _number(name, value, default, low, high, *, integral=False):
    if value is None:
        return default
    if integral:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
        if not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")
        return value
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number from {low} to {high}") from None
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be a finite number from {low} to {high}")
    return result


def _y_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0],
                     [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


def segmental_geometry(half, rise, thickness):
    """A segmental ring's centres and radii; the centre is below the springing."""
    half, rise = float(half), float(rise)
    cy = (rise * rise - half * half) / (2.0 * rise)
    radius = rise - cy
    inner_r = radius - thickness
    inner_half = math.sqrt(max(inner_r * inner_r - cy * cy, 0.0))
    return {"cy": cy, "r": radius, "ri": inner_r, "wi": inner_half,
            "theta0": math.atan2(-cy, half),
            "theta_i": math.atan2(-cy, max(inner_half, 1e-4)),
            "crown": math.pi / 2.0}


def pointed_geometry(half, rise, thickness):
    """A pointed ring's centres and radii; the centres sit beyond the axis."""
    half, rise = float(half), float(rise)
    k = (rise * rise - half * half) / (2.0 * half)
    radius = half + k
    inner_r = radius - thickness
    inner_rise = math.sqrt(max(inner_r * inner_r - k * k, 0.0))
    return {"k": k, "r": radius, "ri": inner_r, "wi": -k + inner_r,
            "rise_i": inner_rise, "theta0": 0.0,
            "crown": math.atan2(rise, k)}


@dataclass(frozen=True)
class Design:
    variant: str
    finish: str
    detail: str
    levels: int
    houses: int
    frontage: float
    depth: float
    storeys: Tuple[float, ...]
    pitch_deg: float
    bay_full: bool
    wing_side: int

    @property
    def eave(self):
        return round(PLINTH_H + sum(self.storeys), 6)

    @property
    def storey_tops(self):
        height = PLINTH_H
        tops = []
        for storey in self.storeys:
            height += storey
            tops.append(round(height, 6))
        return tops

    @property
    def floors(self):
        lines = [PLINTH_H]
        height = PLINTH_H
        for storey in self.storeys[:-1]:
            height += storey
            lines.append(round(height, 6))
        return lines

    @property
    def rise(self):
        return math.tan(math.radians(self.pitch_deg)) * self.depth / 2.0

    @property
    def ridge(self):
        return self.eave + self.rise

    @property
    def total_width(self):
        return self.houses * self.frontage

    @classmethod
    def resolve(cls, request):
        rng = random.Random(request.seed)
        automatic = request.variant in ("auto", "random")
        if (not automatic and request.variant not in VARIANTS) or request.detail not in DETAILS:
            raise ValueError("unknown victorian-terrace variant or detail level")
        if request.openings != "auto":
            raise ValueError("victorian-terrace authors its own openings")
        params = request.parameters
        if set(params) - set(CONTROLS):
            raise ValueError(f"unknown victorian-terrace parameters: "
                             f"{sorted(set(params) - set(CONTROLS))}")
        finish = params.get("finish", rng.choice(tuple(FINISHES)))
        if not isinstance(finish, str) or finish not in FINISHES:
            raise ValueError(f"victorian-terrace finish must be one of {', '.join(FINISHES)}")
        variant = request.variant
        if automatic:
            variant = rng.choice(VARIANTS)
        levels = _number("levels", request.levels, 2, 2, 3, integral=True)
        if variant == "gothic-villa":
            # One house; the villa frontage is its plan width. A wide request
            # is clamped into the villa band rather than rejected, so a
            # catalogue default still resolves under variant auto.
            raw = _number("width", request.width, 7.8, 4.0, 60.0)
            frontage = min(WIDTH_BAND[variant][1], max(WIDTH_BAND[variant][0], raw))
            houses = 1
        else:
            lo, hi = HOUSE_RANGE[variant]
            band = FRONTAGE_BAND[variant]
            typical = rng.uniform(*band)
            if request.bays is not None:
                houses = _number("bays", request.bays, 6, lo, hi, integral=True)
                frontage = _number("width", request.width, houses * typical,
                                   *WIDTH_BAND[variant]) / houses
            else:
                requested = _number("width", request.width, 30.0, *WIDTH_BAND[variant])
                houses = max(lo, min(hi, int(round(requested / typical))))
                frontage = requested / houses
            if not band[0] - 1e-9 <= frontage <= band[1] + 1e-9:
                for step in (-1, 1):
                    if request.width and lo <= houses + step <= hi:
                        trial = request.width / (houses + step)
                        if band[0] - 1e-9 <= trial <= band[1] + 1e-9:
                            houses += step
                            frontage = trial
                            break
                frontage = max(band[0], min(band[1], frontage))
        depth = _number("depth", request.depth, DEPTH_BAND[variant][0] + 0.7,
                        *DEPTH_BAND[variant])
        storeys = [rng.uniform(2.85, 3.05)]
        storeys += [rng.uniform(2.6, 2.85) for _ in range(levels - 1)]
        storeys = tuple(round(s, 4) for s in storeys)
        pitch_band = VILLA_PITCH_BAND if variant == "gothic-villa" else PITCH_BAND
        pitch_deg = round(rng.uniform(pitch_band[0] + 1.0, pitch_band[1] - 1.0), 2)
        if request.height is not None:
            height = _number("height", request.height, 9.2, 4.0, 30.0)
            eave = PLINTH_H + sum(storeys)
            run = frontage / 2.0 if variant == "gothic-villa" else depth / 2.0
            rise = height - eave
            rise = max(math.tan(math.radians(pitch_band[0])) * run,
                       min(math.tan(math.radians(pitch_band[1])) * run, rise))
            pitch_deg = round(math.degrees(math.atan2(rise, run)), 2)
        return cls(variant=variant, finish=finish, detail=request.detail, levels=levels,
                   houses=houses, frontage=round(frontage, 6), depth=depth,
                   storeys=storeys, pitch_deg=pitch_deg,
                   bay_full=rng.random() < 0.4, wing_side=rng.choice((-1, 1)))


def random_controls(rng=None, variant="auto"):
    """Buildable GUI/series dimensions; never retries a random bad box."""
    rng = rng if rng is not None else random.Random()
    roof = rng.choice(VARIANTS) if variant in ("auto", "random") else variant
    if roof == "two-up-two-down":
        houses = rng.randint(4, 10)
        width = round(houses * rng.uniform(4.1, 5.0), 2)
        depth = round(rng.uniform(5.9, 8.1), 2)
    elif roof == "bay-fronted":
        houses = rng.randint(3, 8)
        width = round(houses * rng.uniform(5.2, 7.0), 2)
        depth = round(rng.uniform(6.1, 8.9), 2)
    else:
        houses = 1
        width = round(rng.uniform(6.9, 9.3), 2)
        depth = round(rng.uniform(6.9, 9.4), 2)
    return {"width": width, "depth": depth,
            "bays": None if roof == "gothic-villa" else houses,
            "levels": rng.randint(2, 3), "detail": rng.choice(DETAILS),
            "variant": roof}


class VictorianBuilder:
    def __init__(self, design):
        self.d = design
        self.detail = design.detail
        self.palette = _palette(design.finish)
        self.substances = _substances(design.finish)
        self.placements = []
        self.counts = Counter()
        self.ridge_y = design.ridge

    # ------------------------------------------------------------ members
    def add(self, role, position, size, payload=None, material="brick", block_type="CUBE"):
        index = self.counts[role]
        self.counts[role] += 1
        pid = f"{PRODUCT}.{role}.{index:03d}"
        self.placements.append(Placement(
            id=pid, kind=KINDS.get(role, role), role=role, block_type=block_type,
            position=tuple(round(v, 8) for v in position),
            dimensions=tuple(round(v, 8) for v in size),
            material=self.palette[material], parent_id=None,
            parameters={"product": PRODUCT, "material_role": material, **(payload or {})},
            substance=self.substances[material]))
        return pid

    def box(self, role, x0, x1, y0, y1, z0, z1, material="brick", **payload):
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        z0, z1 = sorted((z0, z1))
        return self.add(role, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                        (x1 - x0, y1 - y0, z1 - z0), payload, material)

    def coursed(self, role, x0, x1, y0, y1, z0, z1, material="brick", **payload):
        """A load-bearing member as human-scale brick panels in running bond."""
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        z0, z1 = sorted((z0, z1))
        host = f"{PRODUCT}.{role}.{self.counts[role]:04d}"
        for stone in fill_box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                              (x1 - x0, y1 - y0, z1 - z0), self.detail):
            self.add(role, stone.position, stone.dimensions,
                     {"construction": "ashlar running bond", "unit": stone.unit,
                      "course": stone.course, "host_id": host, "axis": stone.axis,
                      **payload}, material)
        return host

    def wall_run(self, role, axis, a0, a1, y0, y1, cross_lo, cross_hi,
                 openings=(), material="brick", **payload):
        """Course one straight wall band; openings stay true voids.

        ``openings`` are (opening_id, center, half, sill, spring, rise_outer)
        in world heights along the run. Each void covers the light and the
        gauged arch head above it, the way a bricklayer leaves the gap; a
        void wider than this run is clipped to it, so a pair of doors under
        one arch can be left open by both adjoining houses.
        """
        host = f"{PRODUCT}.{role}.{self.counts[role]:04d}"
        run_mid = (a0 + a1) / 2.0
        specs = []
        for opening_id, center, half, sill, spring, rise_o in openings:
            # iter_ashlar takes each centre relative to the run's midpoint.
            center = center - run_mid
            sill = max(sill, y0)
            if sill >= y1 - 0.05:
                continue
            light_top = min(spring, y1)
            if light_top - sill > 0.02:
                specs.append({"id": opening_id, "center": center, "width": 2 * half,
                              "sill": sill - y0, "height": light_top - sill})
            head_top = min(spring + rise_o, y1)
            if spring < y1 - 0.05 and head_top > spring + 0.02:
                specs.append({"id": opening_id + ".head", "center": center,
                              "width": 2 * (half + RING_T), "sill": spring - y0,
                              "height": head_top - spring})
        if axis == "x":
            center = ((a0 + a1) / 2, (y0 + y1) / 2, (cross_lo + cross_hi) / 2)
            dims = (a1 - a0, y1 - y0, cross_hi - cross_lo)
        else:
            center = ((cross_lo + cross_hi) / 2, (y0 + y1) / 2, (a0 + a1) / 2)
            dims = (cross_hi - cross_lo, y1 - y0, a1 - a0)
        if min(dims) <= 0.02:
            return host
        for stone in fill_box(center, dims, self.detail, specs, axis):
            extra = {"opening_id": stone.opening_id} if stone.opening_id else {}
            self.add(role, stone.position, stone.dimensions,
                     {"construction": "ashlar running bond", "unit": stone.unit,
                      "course": stone.course, "host_id": host, "axis": stone.axis,
                      **extra, **payload}, material)
        return host

    def mesh(self, role, polygon, extrusion, x=0.0, y=0.0, z=0.0, material="brick",
             angle=0.0, along="x", **payload):
        """Add an extruded profile member; ``polygon`` is (along, y)."""
        pts = np.asarray(polygon, dtype=float)
        dedup = [pts[0].tolist()]
        for row in pts[1:]:
            if any(abs(row[k] - dedup[-1][k]) > 1e-9 for k in range(2)):
                dedup.append(row.tolist())
        if len(dedup) > 1 and all(abs(dedup[0][k] - dedup[-1][k]) <= 1e-9 for k in range(2)):
            dedup.pop()
        mesh = extrude_profile(np.asarray(dedup, dtype=float), extrusion)
        if along == "z":
            mesh.apply_transform(np.array([[0., 0., -1., 0.], [0., 1., 0., 0.],
                                           [1., 0., 0., 0.], [0., 0., 0., 1.]]))
        if angle:
            mesh.apply_transform(_y_rotation(angle))
        mesh.apply_translation((x, y, z))
        position, size, data = mesh_parameters(mesh)
        return self.add(role, position, size, {**data, **payload}, material, "MESH")

    def rotated_box(self, role, center, dims, angle, material="brick", **payload):
        """A coursed brick turned in plan, for canted bay flanks."""
        mesh = trimesh.creation.box(extents=[float(v) for v in dims])
        mesh.apply_transform(_y_rotation(angle))
        mesh.apply_translation([float(v) for v in center])
        position, size, data = mesh_parameters(mesh)
        return self.add(role, position, size,
                        {"construction": "ashlar flank bond", **data, **payload},
                        material, "MESH")

    def gable_field(self, role, polygon, depth, cross, material="brick", along="x",
                    **payload):
        """A triangular or raked field coursed into human-scale tiles.

        ``polygon`` is an elevation profile in (along, y). With ``along="x"``
        the profile is extruded across the depth at ``z = cross``; with
        ``along="z"`` it is extruded across the width at ``x = cross``, which
        is how a party wall's raked parapet is laid.
        """
        host = f"{PRODUCT}.{role}.{self.counts[role]:04d}"
        for mesh, course, tile in course_profile_tiles(polygon, depth, self.detail):
            if along == "z":
                mesh.apply_transform(np.array(
                    [[0., 0., -1., 0.], [0., 1., 0., 0.],
                     [1., 0., 0., 0.], [0., 0., 0., 1.]]))
                mesh.apply_translation((cross, 0.0, 0.0))
            else:
                mesh.apply_translation((0.0, 0.0, cross))
            position, size, data = mesh_parameters(mesh)
            self.add(role, position, size,
                     {"construction": "ashlar profile course", "unit": "profile_course",
                      "course": course, "tile": tile, "host_id": host, **data, **payload},
                     material, "MESH")
        return host

    # ------------------------------------------------------------- arches
    def gauged_arch(self, role, axis, center, cross, thickness, half, spring,
                    rise, opening_id, material="brick", arch="segmental",
                    lights=None, sill=0.0, **payload):
        """A gauged-brick arch of voussoirs bearing on the jamb masonry.

        The ring springs past the light's edges, so each end embeds into the
        jambs. ``lights`` lists [center, half] pairs beneath this arch so the
        audit can prove each light is a true void.
        """
        wo = half + RING_T
        if arch == "segmental":
            g = segmental_geometry(wo, rise, RING_T)
            wedge = True
        else:
            g = pointed_geometry(wo, rise, RING_T)
            wedge = False
        body = {"construction": "gauged arch course", "unit": "voussoir",
                "arch": arch, "span": round(2 * half, 6), "half": round(half, 6),
                "ring": RING_T, "spring_y": round(spring, 6), "sill": round(sill, 6),
                "rise": round(rise, 6), "axis": axis,
                "center": round(center, 6), "span_half": round(wo, 6),
                "opening_id": opening_id,
                "lights": [[round(c, 6), round(h, 6)]
                           for c, h in (lights or [[center, half]])],
                **payload}
        if arch == "segmental":
            arc_len = g["r"] * (g["crown"] - g["theta0"])
        else:
            arc_len = g["r"] * g["crown"]
        count = max(5, int(math.ceil(2.0 * arc_len / 0.42)) | 1)
        half_count = max(2, count // 2)
        a_lo = g["theta_i"] if wedge else g["theta0"]

        def oy(y_rel):
            return spring + y_rel

        for sign in (1, -1):
            if wedge:
                tm = (g["theta0"] + g["theta_i"]) / 2.0
                poly = [(sign * g["r"] * math.cos(t), oy(g["cy"] + g["r"] * math.sin(t)))
                        for t in (g["theta0"], tm, g["theta_i"])]
                poly.append((sign * g["wi"], spring))
                self._arch_piece(role, poly, axis, center, cross, thickness,
                                 material, **dict(body, voussoir=0))
            for i in range(half_count):
                a0 = a_lo + (g["crown"] - a_lo) * i / half_count
                a1 = a_lo + (g["crown"] - a_lo) * (i + 1) / half_count
                am = (a0 + a1) / 2.0
                if arch == "segmental":
                    poly = [(sign * g["r"] * math.cos(t), oy(g["cy"] + g["r"] * math.sin(t)))
                            for t in (a0, am, a1)]
                    poly += [(sign * g["ri"] * math.cos(t), oy(g["cy"] + g["ri"] * math.sin(t)))
                             for t in (a1, am, a0)]
                else:
                    poly = [(sign * (-g["k"] + g["r"] * math.cos(t)), oy(g["r"] * math.sin(t)))
                            for t in (a0, am, a1)]
                    poly += [(sign * (-g["k"] + g["ri"] * math.cos(t)), oy(g["ri"] * math.sin(t)))
                             for t in (a1, am, a0)]
                self._arch_piece(role, poly, axis, center, cross, thickness,
                                 material, **dict(body, voussoir=i + 1,
                                                  keystone=(i == half_count - 1)))
        return opening_id

    def _arch_piece(self, role, poly, run_axis, run_center, cross, thickness, material, **payload):
        if run_axis == "x":
            self.mesh(role, poly, thickness, run_center, 0.0, cross, material,
                      along="x", **payload)
        else:
            self.mesh(role, poly, thickness, cross, 0.0, run_center, material,
                      along="z", **payload)

    def arch_spandrels(self, axis, center, cross_lo, cross_hi, half, spring,
                       rise, top, material="brick", arch="segmental"):
        """Fill the head zone each side of the ring up to the band top."""
        wo = half + RING_T
        if arch == "segmental":
            g = segmental_geometry(wo, rise, RING_T)
            k, cy, radius = 0.0, g["cy"], g["r"]
        else:
            g = pointed_geometry(wo, rise, RING_T)
            k, cy, radius = g["k"], 0.0, g["r"]

        def arc_y(x_abs):
            """The outer curve's height above the springing line."""
            if arch == "segmental":
                return cy + math.sqrt(max(radius * radius - x_abs * x_abs, 0.0))
            return math.sqrt(max(radius * radius - (x_abs + k) ** 2, 0.0))

        for sign in (-1, 1):
            edge = abs(wo)
            while edge > 0.14 and top - (spring + arc_y(edge)) > 0.05:
                inner = max(0.14, edge - 0.42)
                y0 = min(max(spring + arc_y(inner), spring), top - 0.04)
                if top - y0 > 0.04:
                    a0 = center + min(sign * edge, sign * inner)
                    a1 = center + max(sign * edge, sign * inner)
                    if axis == "x":
                        self.box("spandrel", a0, a1, y0, top,
                                 cross_lo, cross_hi, material,
                                 construction="ashlar spandrel fill")
                    else:
                        self.box("spandrel", cross_lo, cross_hi, y0, top,
                                 a0, a1, material,
                                 construction="ashlar spandrel fill")
                edge = inner

    # ------------------------------------------------------------ joinery
    def sash_window(self, axis, center, face, inward, wall_t, sill, head, half,
                    opening_id, **payload):
        """A sash window seated in a reveal: case, meeting rail, two panes."""
        frame_a = face + inward * (REVEAL + 0.06)
        frame_b = face + inward * REVEAL
        glass_a = face + inward * (REVEAL + 0.045)
        glass_b = glass_a - inward * 0.02
        meeting = sill + 0.46 * (head - sill)
        body = {"opening_id": opening_id, "reveal": REVEAL, **payload}
        for sign in (-1, 1):
            edge = center + sign * half
            if axis == "x":
                self.box("sash_stile", edge - 0.055, edge, sill, head,
                         min(frame_a, frame_b), max(frame_a, frame_b), "sash",
                         construction="sash case", **body)
            else:
                self.box("sash_stile", min(frame_a, frame_b), max(frame_a, frame_b),
                         sill, head, edge - 0.055, edge, "sash",
                         construction="sash case", **body)
        if axis == "x":
            self.box("sash_rail", center - half, center + half, meeting, meeting + 0.075,
                     min(frame_a, frame_b), max(frame_a, frame_b), "sash",
                     construction="sash meeting rail", **body)
            self.box("sash_glass", center - half + 0.055, center + half - 0.055,
                     sill + 0.03, meeting, min(glass_a, glass_b), max(glass_a, glass_b),
                     "glass", construction="sash pane", pane="lower", **body)
            self.box("sash_glass", center - half + 0.055, center + half - 0.055,
                     meeting + 0.075, head - 0.03, min(glass_a, glass_b),
                     max(glass_a, glass_b), "glass", construction="sash pane",
                     pane="upper", **body)
        else:
            self.box("sash_rail", min(frame_a, frame_b), max(frame_a, frame_b),
                     meeting, meeting + 0.075, center - half, center + half, "sash",
                     construction="sash meeting rail", **body)
            self.box("sash_glass", min(glass_a, glass_b), max(glass_a, glass_b),
                     sill + 0.03, meeting, center - half + 0.055, center + half - 0.055,
                     "glass", construction="sash pane", pane="lower", **body)
            self.box("sash_glass", min(glass_a, glass_b), max(glass_a, glass_b),
                     meeting + 0.075, head - 0.03, center - half + 0.055,
                     center + half - 0.055, "glass", construction="sash pane",
                     pane="upper", **body)
        for pane in self.placements[-2:]:
            pane.material = (pane.material[0], pane.material[1], pane.material[2], 190)
        # Stone sill bedded on the coursing below the light, projecting outward.
        if axis == "x":
            self.box("sill", center - half - 0.09, center + half + 0.09, sill - 0.075,
                     sill, face - inward * 0.045, face + inward * (wall_t - 0.02),
                     "dressings", construction="stone sill", opening_id=opening_id)
        else:
            self.box("sill", face - inward * 0.045, face + inward * (wall_t - 0.02),
                     sill - 0.075, sill, center - half - 0.09, center + half + 0.09,
                     "dressings", construction="stone sill", opening_id=opening_id)

    def door(self, axis, center, face, inward, half, opening_id, steps=True, **payload):
        """A four-panel door in a deep reveal, stone steps up to the threshold."""
        leaf_a = face + inward * 0.30
        leaf_b = leaf_a - inward * 0.05
        body = {"opening_id": opening_id, **payload}
        for sign in (-1, 1):
            edge = center + sign * half
            if axis == "x":
                self.box("door_jamb", edge - 0.08, edge, 0.0, DOOR_H,
                         min(leaf_a, leaf_b), max(leaf_a, leaf_b), "door",
                         construction="door frame", **body)
            else:
                self.box("door_jamb", min(leaf_a, leaf_b), max(leaf_a, leaf_b),
                         0.0, DOOR_H, edge - 0.08, edge, "door",
                         construction="door frame", **body)
        if axis == "x":
            self.box("door_leaf", center - half + 0.08, center + half - 0.08, 0.02,
                     DOOR_H - 0.03, min(leaf_a, leaf_b), max(leaf_a, leaf_b), "door",
                     construction="four-panel door", **body)
        else:
            self.box("door_leaf", min(leaf_a, leaf_b), max(leaf_a, leaf_b), 0.02,
                     DOOR_H - 0.03, center - half + 0.08, center + half - 0.08, "door",
                     construction="four-panel door", **body)
        if steps:
            outward = -inward
            for i, height in enumerate((0.16, 0.32)):
                z_a = face + outward * (0.62 - i * 0.26)
                z_b = face + outward * 0.10
                if axis == "x":
                    self.box("entry_step", center - half - 0.05, center + half + 0.05,
                             0.0, height, min(z_a, z_b), max(z_a, z_b), "step",
                             construction="stone step", step=i, opening_id=opening_id)
                else:
                    self.box("entry_step", min(z_a, z_b), max(z_a, z_b), 0.0, height,
                             center - half - 0.05, center + half + 0.05, "step",
                             construction="stone step", step=i, opening_id=opening_id)

    # ---------------------------------------------------------- chimneys
    def stack(self, x, z, base, party=None, material="brick"):
        """A stack on its fire-break parapet: drums, mortar bed, coping, pots."""
        host = f"{PRODUCT}.stack.{self.counts['stack']:04d}"
        for center_y, course_h, index in iter_stack(base, STACK_H, self.detail):
            self.add("stack", (x, center_y, z), (STACK_W, course_h, STACK_W),
                     {"construction": "ashlar stack", "unit": "drum", "course": index,
                      "host_id": host, "party": party, "base_y": round(base, 6),
                      "top_y": round(base + STACK_H, 6)}, material)
        top = base + STACK_H
        self.box("mortar_bed", x - STACK_W / 2, x + STACK_W / 2, top, top + BED_T,
                 z - STACK_W / 2, z + STACK_W / 2, "mortar",
                 construction="lime mortar bedding", party=party)
        self.box("stack_coping", x - STACK_W / 2 - 0.07, x + STACK_W / 2 + 0.07,
                 top + BED_T, top + BED_T + COPING_T, z - STACK_W / 2 - 0.07,
                 z + STACK_W / 2 + 0.07, "coping",
                 construction="stone stack coping", party=party)
        cap = top + BED_T + COPING_T
        for sign in (-1, 1):
            self.add("chimney_pot", (x + sign * 0.19, cap + POT_H / 2, z),
                     (2 * POT_R, POT_H, 2 * POT_R),
                     {"construction": "chimney pot", "party": party}, "pot", "CYLINDER")

    def chimney_breast(self, x_party, side, z_lo, z_hi, top, party=None, material="brick"):
        """A breast projecting from the party wall, grounded on its own footing."""
        x_lo = min(x_party + side * (BRICK_T / 2), x_party + side * (BRICK_T / 2 + 0.33))
        x_hi = max(x_party + side * (BRICK_T / 2), x_party + side * (BRICK_T / 2 + 0.33))
        self.coursed("chimney_breast", x_lo, x_hi, 0.0, top, z_lo, z_hi, material,
                     construction="ashlar running bond", party=party)

    # ------------------------------------------------------------- floors
    def floor(self, x_lo, x_hi, z_lo, z_hi, y, house, avoid=()):
        """Joists spanning front to back between the party walls, boards over.

        Joist lines that would cross a chimney breast are framed around it,
        and the boards keep clear of the hearth zone the same way.
        """
        count = max(2, int(round((x_hi - x_lo) / JOIST_PITCH)))
        pitch = (x_hi - x_lo) / count
        for i in range(count + 1):
            x = min(x_lo + i * pitch, x_hi - JOIST_W)
            blocked = any(ax_lo - JOIST_W < x < ax_hi and az_lo < z_hi + JOIST_POCKET
                          and az_hi > z_lo - JOIST_POCKET for ax_lo, ax_hi, az_lo, az_hi in avoid)
            if blocked:
                continue
            self.box("joist", x, x + JOIST_W, y, y + JOIST_H,
                     z_lo - JOIST_POCKET, z_hi + JOIST_POCKET, "joist",
                     construction="joist span", house=house, floor_y=round(y, 6))
        rows = max(2, int(round((z_hi - z_lo) / BOARD_W)))
        board_pitch = (z_hi - z_lo) / rows
        for j in range(rows):
            z = z_lo + j * board_pitch
            blocked = any(az_lo - 0.05 < z + board_pitch and az_hi > z
                          for _, _, az_lo, az_hi in avoid)
            if blocked:
                continue
            self.box("floorboard", x_lo, x_hi, y + JOIST_H, y + JOIST_H + BOARD_T,
                     z, z + board_pitch - 0.008, "board",
                     construction="floorboard", house=house, floor_y=round(y, 6), row=j)

    # -------------------------------------------------------------- roofs
    def roof_slope(self, profile_axis, along_lo, along_hi, eave_line, y_bottom,
                   run, slope_sign, material="slate", **payload):
        """Common rafters and slate courses over one pitch of a roof.

        The rafters spring from ``y_bottom`` at the eaves line and rise
        ``run * tan(pitch)`` to the ridge; they end exactly at the eaves line
        so the pitch stays measurable, and the slate courses ride one
        centimetre above them, overhanging the eaves.
        """
        d = self.d
        slope = math.tan(math.radians(d.pitch_deg))
        ridge_line = eave_line + slope_sign * run
        y_ridge = y_bottom + slope * run
        y_slate = y_bottom + RAFTER_T + 0.01
        count = max(2, int(round((along_hi - along_lo) / RAFTER_PITCH)))
        pitch = (along_hi - along_lo) / count
        for i in range(count + 1):
            along = min(along_lo + i * pitch, along_hi - RAFTER_T)
            poly = [(eave_line, y_bottom), (ridge_line, y_ridge),
                    (ridge_line, y_ridge + RAFTER_T), (eave_line, y_bottom + RAFTER_T)]
            if profile_axis == "z":
                self.mesh("rafter", poly, RAFTER_T, along, 0.0, 0.0, "rafter",
                          along="z", construction="common rafter",
                          pitch_deg=d.pitch_deg, **payload)
            else:
                self.mesh("rafter", poly, RAFTER_T, 0.0, 0.0, along, "rafter",
                          along="x", construction="common rafter",
                          pitch_deg=d.pitch_deg, **payload)
        length = math.hypot(run, y_ridge - y_bottom)
        rows = max(2, int(round(length / SLATE_ROW[self.detail])))
        row_pitch = length / rows
        over = SLATE_OVERHANG / math.hypot(1.0, slope)
        for r in range(rows):
            d0 = -over + r * (length + over) / rows
            d1 = min(-over + (r + 1) * (length + over) / rows, length - 0.02)
            if d1 <= d0 + 0.01:
                continue
            l0 = eave_line + slope_sign * (d0 / math.hypot(1.0, slope))
            l1 = eave_line + slope_sign * (d1 / math.hypot(1.0, slope))
            y0 = y_slate + slope * d0 / math.hypot(1.0, slope)
            y1 = y_slate + slope * d1 / math.hypot(1.0, slope)
            poly = [(l0, y0), (l1, y1), (l1, y1 + SLATE_T), (l0, y0 + SLATE_T)]
            if profile_axis == "z":
                self.mesh("slate_course", poly, along_hi - along_lo,
                          (along_lo + along_hi) / 2, 0.0, 0.0, material, along="z",
                          construction="slate course", row=r, rows=rows, **payload)
            else:
                self.mesh("slate_course", poly, along_hi - along_lo, 0.0, 0.0,
                          (along_lo + along_hi) / 2, material, along="x",
                          construction="slate course", row=r, rows=rows, **payload)

    def ridge_cap(self, profile_axis, along_lo, along_hi, ridge_line, y_cap,
                  eave_y, material="slate", **payload):
        poly = [(ridge_line - 0.14, y_cap), (ridge_line, y_cap + 0.17),
                (ridge_line + 0.14, y_cap), (ridge_line + 0.14, y_cap + 0.09),
                (ridge_line, y_cap + 0.26), (ridge_line - 0.14, y_cap + 0.09)]
        if profile_axis == "z":
            self.mesh("ridge_cap", poly, along_hi - along_lo,
                      (along_lo + along_hi) / 2, 0.0, 0.0, material, along="z",
                      construction="ridge cap", ridge_y=round(y_cap, 6),
                      eave_y=round(eave_y, 6), **payload)
        else:
            self.mesh("ridge_cap", poly, along_hi - along_lo, 0.0, 0.0,
                      (along_lo + along_hi) / 2, material, along="x",
                      construction="ridge cap", ridge_y=round(y_cap, 6),
                      eave_y=round(eave_y, 6), **payload)

    def terrace_roof_top(self, z):
        """Height of the slate covering over a terrace's main roof at depth z."""
        d = self.d
        slope = math.tan(math.radians(d.pitch_deg))
        y_slate = d.eave + BED_T + PLATE_T + RAFTER_T + 0.01 + SLATE_T
        return y_slate + slope * (d.depth / 2.0 - abs(z))

    def raked_parapet(self, axis, at, lo, hi, roof_top, role="party_wall",
                      eave=None, **payload):
        """A wall carried through the roof as a raked parapet.

        The wall head follows the roof pitch on both sides of the ridge,
        standing a fire-break's height above the slates, and is finished with
        its own stone coping. The raking top course is cut to the rake while
        every course below it keeps a level bed, so the coursing steps up the
        slope. ``axis`` is the direction the head runs; ``at`` is the fixed
        coordinate across the wall.
        """
        top_y = self.d.eave if eave is None else eave
        mid = (lo + hi) / 2.0
        base = [roof_top(s) + PARAPET_UPSTAND - PARAPET_COPING_T
                for s in (lo, mid, hi)]
        poly = [(lo, top_y), (lo, base[0]), (mid, base[1]),
                (hi, base[2]), (hi, top_y)]
        self.gable_field(role, poly, BRICK_T, at, along=axis,
                         construction="ashlar raked parapet course",
                         parapet=True, parapet_axis=axis,
                         parapet_at=round(float(at), 6), **payload)
        # Stone raking coping, laid in lift-able lengths cut to the rake.
        for a, b in ((lo, mid), (mid, hi)):
            pieces = max(2, int(math.ceil((b - a) / 0.6)))
            for k in range(pieces):
                p0 = a + (b - a) * k / pieces
                p1 = a + (b - a) * (k + 1) / pieces
                y0 = roof_top(p0) + PARAPET_UPSTAND
                y1 = roof_top(p1) + PARAPET_UPSTAND
                quad = [(p0, y0 - PARAPET_COPING_T), (p1, y1 - PARAPET_COPING_T),
                        (p1, y1), (p0, y0)]
                if axis == "z":
                    self.mesh("parapet_coping", quad, PARAPET_COPING_W, x=at,
                              material="coping", along="z",
                              construction="stone raking coping", parapet=True,
                              parapet_axis=axis, parapet_at=round(float(at), 6),
                              **payload)
                else:
                    self.mesh("parapet_coping", quad, PARAPET_COPING_W, z=at,
                              material="coping", along="x",
                              construction="stone raking coping", parapet=True,
                              parapet_axis=axis, parapet_at=round(float(at), 6),
                              **payload)

    def mono_pitch(self, profile_axis, along_lo, along_hi, hi_line, lo_line,
                   y_hi, y_lo, material="slate", **payload):
        """A lean-to covering from a high wall down to a low one.

        The rafters record the pitch of this slope. An outshut or a bay is
        shallower than the main roof, so that pitch is not the terrace band.
        """
        run = abs(hi_line - lo_line)
        slope = (y_hi - y_lo) / run
        pitch_deg = round(math.degrees(math.atan2(abs(y_hi - y_lo), run)), 2)
        length = math.hypot(run, y_hi - y_lo)
        sign = 1.0 if hi_line > lo_line else -1.0
        y_slate = y_lo + RAFTER_T + 0.01
        count = max(2, int(round((along_hi - along_lo) / RAFTER_PITCH)))
        pitch = (along_hi - along_lo) / count
        for i in range(count + 1):
            along = min(along_lo + i * pitch, along_hi - RAFTER_T)
            poly = [(lo_line, y_lo), (hi_line, y_hi), (hi_line, y_hi + RAFTER_T),
                    (lo_line, y_lo + RAFTER_T)]
            if profile_axis == "z":
                self.mesh("rafter", poly, RAFTER_T, along, 0.0, 0.0, "rafter",
                          along="z", construction="lean-to rafter",
                          pitch_deg=pitch_deg, **payload)
            else:
                self.mesh("rafter", poly, RAFTER_T, 0.0, 0.0, along, "rafter",
                          along="x", construction="lean-to rafter",
                          pitch_deg=pitch_deg, **payload)
        rows = max(2, int(round(length / SLATE_ROW[self.detail])))
        row_pitch = length / rows
        for r in range(rows):
            d0 = r * row_pitch
            d1 = min((r + 1) * row_pitch, length - 0.02)
            l0 = lo_line + sign * (d0 / math.hypot(1.0, slope))
            l1 = lo_line + sign * (d1 / math.hypot(1.0, slope))
            y0 = y_slate + slope * d0 / math.hypot(1.0, slope)
            y1 = y_slate + slope * d1 / math.hypot(1.0, slope)
            poly = [(l0, y0), (l1, y1), (l1, y1 + SLATE_T), (l0, y0 + SLATE_T)]
            if profile_axis == "z":
                self.mesh("slate_course", poly, along_hi - along_lo,
                          (along_lo + along_hi) / 2, 0.0, 0.0, material, along="z",
                          construction="slate course", row=r, rows=rows, **payload)
            else:
                self.mesh("slate_course", poly, along_hi - along_lo, 0.0, 0.0,
                          (along_lo + along_hi) / 2, material, along="x",
                          construction="slate course", row=r, rows=rows, **payload)

    def wall_plate(self, axis, a0, a1, cross, y, **payload):
        """A softwood wall plate bedded in lime mortar on the wall head."""
        if axis == "x":
            self.box("mortar_bed", a0, a1, y, y + BED_T, cross - 0.05, cross + 0.05,
                     "mortar", construction="lime mortar bedding", **payload)
            self.box("plate", a0, a1, y + BED_T, y + BED_T + PLATE_T,
                     cross - 0.06, cross + 0.06, "plate",
                     construction="wall plate", **payload)
        else:
            self.box("mortar_bed", cross - 0.05, cross + 0.05, y, y + BED_T, a0, a1,
                     "mortar", construction="lime mortar bedding", **payload)
            self.box("plate", cross - 0.06, cross + 0.06, y + BED_T,
                     y + BED_T + PLATE_T, a0, a1, "plate",
                     construction="wall plate", **payload)

    # ---------------------------------------------------------- openings
    def window_opening(self, axis, center, face, inward, wall_t, sill, spring,
                       half, opening_id, material="brick", **payload):
        """One sash window with its gauged arch and spandrel fill."""
        rise = ARCH_RISE_RATIO * 2 * (half + RING_T)
        self.sash_window(axis, center, face, inward, wall_t, sill, spring, half,
                         opening_id, **payload)
        cross = face + inward * wall_t / 2.0
        self.gauged_arch("gauged_arch", axis, center, cross, wall_t, half, spring,
                         rise, opening_id, material, sill=sill, **payload)
        self.arch_spandrels(axis, center, min(face, face + inward * wall_t),
                            max(face, face + inward * wall_t), half, spring,
                            rise, spring + rise + RING_T + 0.02, material)

    def door_opening(self, axis, center, face, inward, wall_t, half, opening_id,
                     sill=0.0, steps=True, material="brick", **payload):
        """One door with its own gauged arch and spandrel fill."""
        rise = ARCH_RISE_RATIO * 2 * (half + RING_T)
        self.door(axis, center, face, inward, half, opening_id, steps, **payload)
        cross = face + inward * wall_t / 2.0
        spring = sill + DOOR_H
        self.gauged_arch("gauged_arch", axis, center, cross, wall_t, half, spring,
                         rise, opening_id, material, sill=sill, **payload)
        self.arch_spandrels(axis, center, min(face, face + inward * wall_t),
                            max(face, face + inward * wall_t), half, spring,
                            rise, spring + rise + RING_T + 0.02, material)

    def pair_arch(self, axis, center, face, inward, wall_t, opening_id,
                  material="brick", **payload):
        """The shared gauged arch over a pair of doors at a party wall."""
        half = 0.95
        rise = ARCH_RISE_RATIO * 2 * (half + RING_T)
        cross = face + inward * wall_t / 2.0
        lights = [[round(center + s * (BRICK_T / 2 + DOOR_W / 2), 6), DOOR_W / 2]
                  for s in (-1, 1)]
        self.gauged_arch("gauged_arch", axis, center, cross, wall_t, half, DOOR_H,
                         rise, opening_id, material, lights=lights, **payload)
        self.arch_spandrels(axis, center, min(face, face + inward * wall_t),
                            max(face, face + inward * wall_t), half, DOOR_H,
                            rise, DOOR_H + rise + RING_T + 0.02, material)

    def single_arch(self, axis, center, face, inward, wall_t, opening_id,
                    material="brick", **payload):
        """The arch over one door beside a terrace end."""
        half = DOOR_W / 2
        rise = ARCH_RISE_RATIO * 2 * (half + RING_T)
        cross = face + inward * wall_t / 2.0
        self.gauged_arch("gauged_arch", axis, center, cross, wall_t, half, DOOR_H,
                         rise, opening_id, material, **payload)
        self.arch_spandrels(axis, center, min(face, face + inward * wall_t),
                            max(face, face + inward * wall_t), half, DOOR_H,
                            rise, DOOR_H + rise + RING_T + 0.02, material)

    # ------------------------------------------------------- compositions
    def compose(self):
        if self.d.variant == "two-up-two-down":
            self.compose_terrace(outshut=True, garden=False)
        elif self.d.variant == "bay-fronted":
            self.compose_terrace(outshut=False, garden=True)
        else:
            self.compose_villa()

    def terrace_bands(self, front):
        """Wall bands: plinth, then one per storey. A three-storey front
        stands on a one-and-a-half brick base projecting forward of the face
        above it."""
        d = self.d
        base_t = BRICK_BASE_T if d.levels == 3 else BRICK_T
        bands = [(0.0, PLINTH_H, base_t, front - (base_t - BRICK_T))]
        y = PLINTH_H
        for top in d.storey_tops:
            bands.append((y, top, BRICK_T, front))
            y = top
        return bands

    def compose_terrace(self, outshut, garden):
        d = self.d
        front = -d.depth / 2.0
        back = d.depth / 2.0
        x_start = -d.total_width / 2.0
        parties = [x_start + i * d.frontage for i in range(d.houses + 1)]
        tops = d.storey_tops
        eave = d.eave
        g_top = tops[0]
        bands = self.terrace_bands(front)
        back_bands = [(0.0, PLINTH_H, BRICK_T, back)]
        y = PLINTH_H
        for top in tops:
            back_bands.append((y, top, BRICK_T, back))
            y = top
        outshut_eave = g_top - OUTSHUT_DROP if outshut else 0.0
        bay_full = d.bay_full
        door_rise = ARCH_RISE_RATIO * 2 * (DOOR_W / 2 + RING_T)
        bay_head = (eave - 0.45) if bay_full else g_top
        pair_records = {}

        for h in range(d.houses):
            x_lo = parties[h] + BRICK_T / 2
            x_hi = parties[h + 1] - BRICK_T / 2
            house_c = (x_lo + x_hi) / 2
            if d.variant == "two-up-two-down":
                pair_x = parties[h] if h % 2 == 0 else parties[h + 1]
                door_c = pair_x + (1 if h % 2 == 0 else -1) * (BRICK_T / 2 + DOOR_W / 2)
                pair_id = f"door.pair.{h:02d}"
                pair_records[round(pair_x, 4)] = (pair_id, door_c)
                pair_rise = ARCH_RISE_RATIO * 2 * (0.95 + RING_T)
                pair_spec = (pair_id, pair_x, 0.95, 0.0, DOOR_H, pair_rise)
                if h % 2 == 0:
                    w_lo = pair_x + 0.95 + RING_T + 0.18
                    w_hi = x_hi - (SASH_HALF + RING_T) - 0.10
                else:
                    w_lo = x_lo + (SASH_HALF + RING_T) + 0.10
                    w_hi = pair_x - 0.95 - RING_T - 0.18
                win_c = (w_lo + w_hi) / 2
                bay_c = None
                upper = [win_c]
            else:
                door_c = x_lo + 0.64
                door_id = f"door.{h:02d}"
                door_spec = (door_id, door_c, DOOR_W / 2, 0.0, DOOR_H, door_rise)
                bay_c = (x_lo + 1.32 + x_hi) / 2
                win_c = house_c
                # Upper sashes keep clear of a through bay opening; above a
                # ground bay the storey is solid and any spacing serves.
                if bay_full:
                    zones = [(x_lo, bay_c - 1.7 - 0.10), (bay_c + 1.7 + 0.10, x_hi)]
                    upper = [(a + b) / 2 for a, b in zones if b - a >= 1.45]
                else:
                    upper = ([win_c - 0.85, win_c + 0.85]
                             if d.frontage - BRICK_T >= 3.9 else [win_c])
            for bi, (y0, y1, t, face) in enumerate(bands):
                openings = []
                if bi <= 1:
                    openings.append(pair_spec if d.variant == "two-up-two-down"
                                    else door_spec)
                if bi == 1:
                    if d.variant == "two-up-two-down":
                        openings.append((f"window.{h:02d}.g", win_c, SASH_HALF,
                                         PLINTH_H + 0.95, g_top - 0.30,
                                         ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                    else:
                        openings.append((f"bay.{h:02d}", bay_c, 1.7, PLINTH_H,
                                         bay_head, 0.0))
                elif bi >= 2:
                    if d.variant == "two-up-two-down" or len(upper) == 1:
                        openings.append((f"window.{h:02d}.u{bi - 1}", upper[0],
                                         SASH_HALF, y0 + 0.90, y1 - 0.30,
                                         ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                    else:
                        for wi, c in enumerate(upper):
                            openings.append((f"window.{h:02d}.u{bi - 1}.{wi}", c,
                                             SASH_HALF, y0 + 0.90, y1 - 0.30,
                                             ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                self.wall_run("plinth" if bi == 0 else "front_wall", "x", x_lo, x_hi,
                              y0, y1, face, face + t, openings, house=h, band=bi)
            if d.variant == "two-up-two-down":
                self.door("x", door_c, front, 1, DOOR_W / 2, f"door.{h:02d}", house=h)
                self.window_opening("x", win_c, front, 1, BRICK_T, PLINTH_H + 0.95,
                                    g_top - 0.30, SASH_HALF, f"window.{h:02d}.g",
                                    house=h, storey=0)
                for s in range(d.levels - 1):
                    y0, y1 = bands[s + 2][0], bands[s + 2][1]
                    self.window_opening("x", upper[0], front, 1, BRICK_T, y0 + 0.90,
                                        y1 - 0.30, SASH_HALF, f"window.{h:02d}.u{s}",
                                        house=h, storey=s + 1)
            else:
                self.door("x", door_c, front, 1, DOOR_W / 2, door_id, house=h)
                # The ground storey belongs to the door and the bay; sashes
                # rise over them from the first upper storey.
                for s in range(d.levels - 1):
                    y0, y1 = bands[s + 2][0], bands[s + 2][1]
                    for wi, c in enumerate(upper):
                        oid = (f"window.{h:02d}.u{s}.{wi}" if len(upper) > 1
                               else f"window.{h:02d}.u{s}")
                        self.window_opening("x", c, front, 1, BRICK_T, y0 + 0.90,
                                            y1 - 0.30, SASH_HALF, oid, house=h,
                                            storey=s + 1)
            # Back wall: service door or kitchen window at ground, sashes above.
            for bi, (y0, y1, t, face) in enumerate(back_bands):
                openings = []
                if bi == 1:
                    if outshut:
                        openings.append((f"backdoor.{h:02d}", house_c, DOOR_W / 2,
                                         PLINTH_H, PLINTH_H + DOOR_H,
                                         ARCH_RISE_RATIO * 2 * (DOOR_W / 2 + RING_T)))
                    else:
                        openings.append((f"kwindow.{h:02d}", house_c, SASH_HALF,
                                         PLINTH_H + 0.95, g_top - 0.30,
                                         ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                elif bi >= 2:
                    openings.append((f"bwindow.{h:02d}.u{bi - 1}", house_c, SASH_HALF,
                                     y0 + 0.90, y1 - 0.30,
                                     ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                self.wall_run("plinth" if bi == 0 else "back_wall", "x", x_lo, x_hi,
                              y0, y1, face - t, face, openings, house=h, band=bi)
            if outshut:
                self.door_opening("x", house_c, back, -1, BRICK_T, DOOR_W / 2,
                                  f"backdoor.{h:02d}", sill=PLINTH_H, steps=False,
                                  house=h)
            else:
                self.window_opening("x", house_c, back, -1, BRICK_T, PLINTH_H + 0.95,
                                    g_top - 0.30, SASH_HALF, f"kwindow.{h:02d}",
                                    house=h, storey=0)
            for s in range(d.levels - 1):
                y0, y1 = back_bands[s + 2][0], back_bands[s + 2][1]
                self.window_opening("x", house_c, back, -1, BRICK_T, y0 + 0.90,
                                    y1 - 0.30, SASH_HALF, f"bwindow.{h:02d}.u{s}",
                                    house=h, storey=s + 1)

        # One shared arch over each door pair; a lone door at a terrace end
        # keeps its own arch.
        end_lo, end_hi = parties[0], parties[-1]
        for pair_x, (pair_id, door_c) in pair_records.items():
            if abs(pair_x - end_lo) < 0.01 or abs(pair_x - end_hi) < 0.01:
                self.single_arch("x", door_c, front, 1, BRICK_T, pair_id)
            else:
                self.pair_arch("x", pair_x, front, 1, BRICK_T, pair_id)

        # Party walls: full depth, carried through the roof as raked parapets.
        # The coursing breaks at the door-arch head so the pier between a
        # door pair finishes its top course exactly under the shared arch.
        pair_rise = ARCH_RISE_RATIO * 2 * (0.95 + RING_T)
        door_head = DOOR_H + pair_rise
        for i, px in enumerate(parties):
            self.coursed("party_wall", px - BRICK_T / 2, px + BRICK_T / 2, 0.0, door_head,
                         front, back, party=i)
            self.coursed("party_wall", px - BRICK_T / 2, px + BRICK_T / 2, door_head, eave,
                         front, back, party=i)
            self.raked_parapet("z", px, front, back, self.terrace_roof_top,
                               role="party_wall", party=i)
            side = 1 if i == 0 else (-1 if i == len(parties) - 1 else (1 if i % 2 else -1))
            self.chimney_breast(px, side, back - BRICK_T - 1.6, back - BRICK_T - 0.3,
                                eave, party=i)
        parapet_top = self.terrace_roof_top(0.0) + PARAPET_UPSTAND
        for i, px in enumerate(parties):
            self.stack(px, 0.0, parapet_top, party=i)

        # Floors: joists front to back between the party walls, boards over.
        for h in range(d.houses):
            x_lo = parties[h] + BRICK_T / 2 + JOIST_SETBACK
            x_hi = parties[h + 1] - BRICK_T / 2 - JOIST_SETBACK
            for f, y in enumerate(d.floors):
                self.floor(x_lo, x_hi, front + BRICK_T, back - BRICK_T, y, h)

        # The slated roof: plates on the wall heads, rafters, courses, ridge.
        plate_top = eave + BED_T + PLATE_T
        for h in range(d.houses):
            a0 = parties[h] + 0.13
            a1 = parties[h + 1] - 0.13
            self.wall_plate("x", a0, a1, front + BRICK_T / 2, eave, house=h)
            self.wall_plate("x", a0, a1, back - BRICK_T / 2, eave, house=h)
            self.roof_slope("z", a0, a1, front, plate_top, d.depth / 2.0, 1, house=h)
            self.roof_slope("z", a0, a1, back, plate_top, d.depth / 2.0, -1, house=h)
            cap_y = plate_top + d.rise + RAFTER_T + 0.01 - 0.06
            self.ridge_cap("z", a0, a1, 0.0, cap_y, eave, house=h)

        # The rear outshut: party extensions, back wall, lean-to roof.
        if outshut:
            z_out_back = back + OUTSHUT_D
            for i, px in enumerate(parties):
                self.coursed("outshut_wall", px - BRICK_T / 2, px + BRICK_T / 2,
                             0.0, outshut_eave, back, z_out_back, party=i)
            for h in range(d.houses):
                x_lo = parties[h] + BRICK_T / 2
                x_hi = parties[h + 1] - BRICK_T / 2
                door_c = x_lo + 0.64
                win_c = (x_lo + 1.32 + x_hi - 0.685) / 2
                openings = [(f"odoor.{h:02d}", door_c, DOOR_W / 2, 0.0, DOOR_H,
                             door_rise)]
                has_window = win_c - SASH_HALF * 0.7 - RING_T > door_c + 0.53 + 0.12
                if has_window:
                    openings.append((f"owindow.{h:02d}", win_c, SASH_HALF * 0.7,
                                     1.0, outshut_eave - 0.50,
                                     ARCH_RISE_RATIO * 2 * (SASH_HALF * 0.7 + RING_T)))
                self.wall_run("outshut_wall", "x", x_lo, x_hi, 0.0, outshut_eave,
                              z_out_back - BRICK_T, z_out_back, openings, house=h)
                self.door_opening("x", door_c, z_out_back, -1, BRICK_T, DOOR_W / 2,
                                  f"odoor.{h:02d}", house=h)
                if has_window:
                    self.window_opening("x", win_c, z_out_back, -1, BRICK_T, 1.0,
                                        outshut_eave - 0.50, SASH_HALF * 0.7,
                                        f"owindow.{h:02d}", house=h)
                self.mono_pitch("z", parties[h] + 0.05, parties[h + 1] - 0.05,
                                back, z_out_back, g_top, outshut_eave, house=h)

        # The bay windows and the front garden.
        if d.variant == "bay-fronted":
            self.compose_bays(parties, front, g_top, eave, bay_full, bay_head)
            if garden:
                self.compose_garden(parties, front)

    def compose_bays(self, parties, front, g_top, eave, bay_full, bay_head):
        d = self.d
        bay_top = bay_head
        bay_face_hi = front - BAY_PROJ
        bay_face_lo = bay_face_hi - BRICK_T
        for h in range(d.houses):
            x_lo = parties[h] + BRICK_T / 2
            x_hi = parties[h + 1] - BRICK_T / 2
            bay_c = (x_lo + 1.32 + x_hi) / 2
            bay_x0, bay_x1 = bay_c - BAY_FACE / 2, bay_c + BAY_FACE / 2
            # Bay plinth and front wall with its sash lights.
            self.coursed("bay_plinth", bay_x0, bay_x1, 0.0, 0.25, bay_face_lo,
                         bay_face_hi, "dressings", house=h)
            ground_spring = (g_top - 0.35) if bay_full else (bay_top - 0.30)
            openings = [(f"baywin.{h:02d}.g", bay_c, SASH_HALF, 0.90, ground_spring,
                         ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T))]
            if bay_full:
                openings.append((f"baywin.{h:02d}.u", bay_c, SASH_HALF, g_top + 0.90,
                                 bay_top - 0.30,
                                 ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
            self.wall_run("bay_wall", "x", bay_x0, bay_x1, 0.25, bay_top,
                          bay_face_lo, bay_face_hi, openings, house=h)
            self.window_opening("x", bay_c, bay_face_hi, -1, BRICK_T, 0.90,
                                ground_spring, SASH_HALF, f"baywin.{h:02d}.g",
                                house=h, storey=0)
            if bay_full:
                self.window_opening("x", bay_c, bay_face_hi, -1, BRICK_T, g_top + 0.90,
                                    bay_top - 0.30, SASH_HALF, f"baywin.{h:02d}.u",
                                    house=h, storey=1)
            # Canted flanks: coursed bricks turned to the bay's rake, with
            # their own grounded plinth course.
            for sign in (-1, 1):
                p_main = (bay_c + sign * (BAY_FACE / 2 + BAY_PROJ), front)
                p_bay = (bay_c + sign * BAY_FACE / 2, bay_face_lo)
                dx, dz = p_bay[0] - p_main[0], p_bay[1] - p_main[1]
                run_len = math.hypot(dx, dz)
                angle = math.atan2(-dz, dx)
                for center_y, course_h, index in iter_stack(0.0, 0.25, self.detail):
                    for k, (s0, s1) in enumerate(self._flank_cuts(run_len, 0.5)):
                        mid = (s0 + s1) / 2
                        self.rotated_box("bay_flank",
                                         (p_main[0] + dx * mid / run_len, center_y,
                                          p_main[1] + dz * mid / run_len),
                                         (s1 - s0, course_h, BRICK_T), angle,
                                         house=h, course=index, plinth=True)
                y = 0.25
                for center_y, course_h, index in iter_stack(0.25, bay_top - 0.25,
                                                           self.detail):
                    for k, (s0, s1) in enumerate(self._flank_cuts(run_len, 0.45)):
                        mid = (s0 + s1) / 2
                        self.rotated_box("bay_flank",
                                         (p_main[0] + dx * mid / run_len, center_y,
                                          p_main[1] + dz * mid / run_len),
                                         (s1 - s0, course_h, BRICK_T), angle,
                                         house=h, course=100 + index)
            # The bay roof: a lean-to from the main wall down to the bay face,
            # overhanging the canted flanks at the front corners.
            attach = bay_top + 0.7
            self.mono_pitch("z", bay_x0 - 0.42, bay_x1 + 0.42, front - 0.02,
                            bay_face_hi, attach, bay_top, house=h, bay=True)

    @staticmethod
    def _flank_cuts(run_len, target):
        pieces = max(1, int(math.ceil(run_len / target)))
        out = []
        for k in range(pieces):
            s0 = k * run_len / pieces
            s1 = (k + 1) * run_len / pieces
            if s1 - s0 >= 0.12:
                out.append((s0, s1))
        return out

    def compose_garden(self, parties, front):
        d = self.d
        street = front - GARDEN_D
        for i, px in enumerate(parties):
            self.coursed("garden_pier", px - 0.13, px + 0.13, 0.0, 0.75, street,
                         street + 0.26, "dressings", party=i)
        for h in range(d.houses):
            x_lo = parties[h] + BRICK_T / 2
            x_hi = parties[h + 1] - BRICK_T / 2
            door_c = x_lo + 0.64
            gate_lo, gate_hi = door_c - 0.60, door_c + 0.60
            for s, (a0, a1) in enumerate([(x_lo + 0.02, gate_lo - 0.02),
                                         (gate_hi + 0.02, x_hi - 0.13)]):
                if a1 - a0 < 0.3:
                    continue
                self.coursed("garden_wall", a0, a1, 0.0, 0.55, street, street + 0.26,
                             house=h, span=s)
                count = max(2, int(round((a1 - a0) / 0.18)))
                pitch = (a1 - a0) / count
                for k in range(count + 1):
                    x = min(a0 + k * pitch, a1 - 0.025)
                    self.box("railing_bar", x, x + 0.025, 0.55, 1.10, street + 0.09,
                             street + 0.17, "rail", house=h, span=s,
                             construction="wrought-iron railing bar")
                self.box("railing_rail", a0, a1, 1.10, 1.15, street + 0.06, street + 0.20,
                         "rail", house=h, span=s, construction="wrought-iron handrail")
            tiles = max(2, int(round(GARDEN_D / 0.45)))
            pitch = GARDEN_D / tiles
            for j in range(tiles):
                z = street + j * pitch
                self.box("path_tile", door_c - 0.45, door_c + 0.45, 0.0, 0.05,
                         z, z + pitch - 0.012, "tile", house=h,
                         construction="encaustic path tile", row=j)

    # ------------------------------------------------------------- villa
    def compose_villa(self):
        d = self.d
        half_w = d.frontage / 2.0
        half_dp = d.depth / 2.0
        front, back = -half_dp, half_dp
        side = d.wing_side
        wing_w = min(3.4, 0.45 * d.frontage)
        wing_d = 1.9
        eave = d.eave
        tops = d.storey_tops
        g_top = tops[0]
        run = half_w
        rise = math.tan(math.radians(d.pitch_deg)) * run
        ridge = eave + rise
        self.ridge_y = ridge
        wing_run = wing_w / 2.0
        wing_lo = -half_w if side < 0 else half_w - wing_w
        wing_hi = wing_lo + wing_w
        wing_face = front - wing_d
        wing_face_outer = wing_face - BRICK_T
        porch_c = (wing_lo + wing_hi) / 2
        porch_face = wing_face - 1.1
        plate_top = eave + BED_T + PLATE_T
        slope = math.tan(math.radians(d.pitch_deg))
        porch_plate = 3.35
        porch_apex = porch_plate + 0.95
        porch_half = 1.1
        bands = self.terrace_bands(front)
        door_rise = ARCH_RISE_RATIO * 2 * (DOOR_W / 2 + RING_T)

        def main_roof_top(x):
            """Height of the main roof's slate surface at width ``x``."""
            return (plate_top + RAFTER_T + 0.01 + SLATE_T
                    + slope * (half_w - abs(x)))

        def wing_roof_top(x):
            return (plate_top + RAFTER_T + 0.01 + SLATE_T
                    + slope * (wing_run - abs(x - porch_c)))

        def porch_roof_top(x):
            return (porch_plate + 0.09 + (porch_apex - porch_plate)
                    * (porch_half - abs(x - porch_c)) / porch_half)

        def verge_boards(kind, z_board, x_lo, x_hi, roof_top):
            """Close each verge: a board under the coping, lapped over the slate.

            The foot drops below the slate top by ``VERGE_LAP``, so the end
            of the covering — and a rafter brought to the face — is masked
            in the gable elevation. The head sits in the bed of the coping.
            """
            apex = (x_lo + x_hi) / 2.0
            for s in (-1, 1):
                edge = x_hi if s > 0 else x_lo
                y_edge = roof_top(edge) + PARAPET_UPSTAND - PARAPET_COPING_T
                y_apex = roof_top(apex) + PARAPET_UPSTAND - PARAPET_COPING_T
                poly = [(edge, y_edge - VERGE_BOARD_DEPTH),
                        (apex, y_apex - VERGE_BOARD_DEPTH),
                        (apex, y_apex), (edge, y_edge)]
                self.mesh("bargeboard", poly, VERGE_BOARD_T, 0.0, 0.0, z_board,
                          "bargeboard", along="x",
                          construction="decorative bargeboard", gable=kind)

        def banded(y0, y1):
            """Split a storey band into alternating polychrome courses."""
            out = []
            count = max(1, int(round((y1 - y0) / BAND_MODULE)))
            pitch = (y1 - y0) / count
            for i in range(count):
                out.append((y0 + i * pitch, y0 + (i + 1) * pitch,
                            "band" if i % 3 == 1 else "brick"))
            return out

        # Main front wall: sashes on the half clear of the wing.
        win_c = -side * d.frontage * 0.30
        for bi, (y0, y1, t, face) in enumerate(bands):
            first = True
            for sy0, sy1, mat in banded(y0, y1):
                openings = []
                if bi >= 1:
                    openings.append((f"vwindow.{bi}", win_c, SASH_HALF, y0 + 0.90,
                                     y1 - 0.30,
                                     ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                self.wall_run("plinth" if bi == 0 else "house_wall", "x", -half_w,
                              half_w, sy0, sy1, face, face + t, openings, material=mat,
                              band=bi, wall_face="front")
                if first and bi >= 1:
                    self.window_opening("x", win_c, front, 1, BRICK_T, y0 + 0.90,
                                        y1 - 0.30, SASH_HALF, f"vwindow.{bi}",
                                        storey=bi - 1)
                first = False

        # Back wall: service door and sashes.
        for bi, (y0, y1, _t, _f) in enumerate(bands):
            first = True
            for sy0, sy1, mat in banded(y0, y1):
                openings = []
                if bi == 0:
                    openings.append(("vbackdoor", -side * half_w * 0.5, DOOR_W / 2,
                                     0.0, DOOR_H, door_rise))
                elif bi == 1:
                    openings.append(("vbackdoor", -side * half_w * 0.5, DOOR_W / 2,
                                     0.0, DOOR_H, door_rise))
                    openings.append(("vbwindow.g", side * half_w * 0.5, SASH_HALF,
                                     y0 + 0.90, y1 - 0.30,
                                     ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                elif bi >= 2:
                    for s in (-1, 1):
                        openings.append((f"vbwindow.u{bi - 1}.{s}", s * half_w * 0.5,
                                         SASH_HALF, y0 + 0.90, y1 - 0.30,
                                         ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                self.wall_run("plinth" if bi == 0 else "house_wall", "x", -half_w,
                              half_w, sy0, sy1, back - BRICK_T, back, openings,
                              material=mat, band=bi, wall_face="back")
                if first:
                    if bi == 1:
                        self.door_opening("x", -side * half_w * 0.5, back, -1, BRICK_T,
                                           DOOR_W / 2, "vbackdoor", storey=0)
                        self.window_opening("x", side * half_w * 0.5, back, -1, BRICK_T,
                                            y0 + 0.90, y1 - 0.30, SASH_HALF,
                                            "vbwindow.g", storey=0)
                    elif bi >= 2:
                        for s in (-1, 1):
                            self.window_opening("x", s * half_w * 0.5, back, -1,
                                                BRICK_T, y0 + 0.90, y1 - 0.30,
                                                SASH_HALF, f"vbwindow.u{bi - 1}.{s}",
                                                storey=bi - 1)
                first = False

        # Side walls: plain banded runs, with a sash on one flank.
        for sx, inward in ((-half_w, 1), (half_w, -1)):
            face = sx if inward > 0 else sx - BRICK_T
            for bi, (y0, y1, _t, _f) in enumerate(bands):
                first = True
                for sy0, sy1, mat in banded(y0, y1):
                    openings = []
                    if bi >= 1 and inward > 0:
                        openings.append((f"vswindow.{bi}", -half_dp * 0.4, SASH_HALF,
                                         y0 + 0.90, y1 - 0.30,
                                         ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                    self.wall_run("plinth" if bi == 0 else "house_wall", "z", front,
                                  back, sy0, sy1, face, face + BRICK_T, openings,
                                  material=mat, band=bi, wall_face="side")
                    if first and bi >= 1 and inward > 0:
                        self.window_opening("z", -half_dp * 0.4, sx, inward, BRICK_T,
                                            y0 + 0.90, y1 - 0.30, SASH_HALF,
                                            f"vswindow.{bi}", storey=bi - 1)
                    first = False

        # The projecting gabled wing: front wall with the porch door, flank
        # walls returned to the main front, and its own gable.
        for bi, (y0, y1, _t, _f) in enumerate(bands):
            first = True
            for sy0, sy1, mat in banded(y0, y1):
                openings = []
                if bi == 0:
                    openings.append(("vwingdoor", porch_c, DOOR_W / 2, 0.0, DOOR_H,
                                     door_rise))
                elif bi == 1:
                    openings.append(("vwingdoor", porch_c, DOOR_W / 2, 0.0, DOOR_H,
                                     door_rise))
                elif bi >= 2:
                    openings.append((f"vwingwin.{bi}", porch_c, SASH_HALF, y0 + 0.90,
                                     y1 - 0.30,
                                     ARCH_RISE_RATIO * 2 * (SASH_HALF + RING_T)))
                self.wall_run("plinth" if bi == 0 else "house_wall", "x", wing_lo,
                              wing_hi, sy0, sy1, wing_face_outer, wing_face, openings,
                              material=mat, band=bi, wall_face="wing")
                if first:
                    if bi == 1:
                        self.door_opening("x", porch_c, wing_face_outer, 1, BRICK_T,
                                           DOOR_W / 2, "vwingdoor", steps=False)
                    elif bi >= 2:
                        self.window_opening("x", porch_c, wing_face_outer, 1, BRICK_T,
                                            y0 + 0.90, y1 - 0.30, SASH_HALF,
                                            f"vwingwin.{bi}", storey=bi - 1)
                first = False
        for wx, inward in ((wing_lo, 1), (wing_hi, -1)):
            face = wx if inward > 0 else wx - BRICK_T
            for bi, (y0, y1, _t, _f) in enumerate(bands):
                for sy0, sy1, mat in banded(y0, y1):
                    self.wall_run("plinth" if bi == 0 else "house_wall", "z",
                                  wing_face_outer, front, sy0, sy1,
                                  face, face + BRICK_T, material=mat, band=bi,
                                  wall_face="wing")
        self.raked_parapet("x", (wing_face_outer + wing_face) / 2, wing_lo, wing_hi,
                           wing_roof_top, role="house_wall", gable="wing")
        # A pointed light in the wing gable.
        light_half, light_rise, light_spring = 0.30, 0.52, eave + 0.35
        self.gauged_arch("pointed_window", "x", porch_c,
                         (wing_face_outer + wing_face) / 2, BRICK_T, light_half,
                         light_spring, light_rise, "vlight", arch="pointed",
                         sill=light_spring)
        glass_lo = wing_face_outer + 0.04
        self.box("sash_glass", porch_c - light_half + 0.03, porch_c + light_half - 0.03,
                 light_spring + 0.04, light_spring + light_rise - 0.05,
                 glass_lo, glass_lo + 0.02, "glass", construction="pointed light",
                 opening_id="vlight")
        pane = self.placements[-1]
        pane.material = (pane.material[0], pane.material[1], pane.material[2], 190)

        # The projecting pointed-arch porch on the wing face.
        for s in (-1, 1):
            px = porch_c + s * (0.66 + 0.22)
            self.coursed("porch_pier", px - 0.22, px + 0.22, 0.0, 2.25, porch_face,
                         wing_face, porch=s)
        self.gauged_arch("porch_arch", "x", porch_c, porch_face + BRICK_T / 2, BRICK_T,
                         0.66, 2.25, 0.85, "vporch", arch="pointed")
        self.arch_spandrels("x", porch_c, porch_face, porch_face + BRICK_T, 0.66,
                            2.25, 0.85, 3.25, arch="pointed")
        self.door("x", porch_c, wing_face, 1, DOOR_W / 2, "vporchdoor", steps=False)
        for i, height in enumerate((0.16, 0.32)):
            self.box("entry_step", porch_c - 0.65, porch_c + 0.65, 0.0, height,
                     porch_face - 0.30 + i * 0.14, porch_face - 0.02, "step",
                     construction="stone step", step=i, opening_id="vporch")
        # Porch gable, raked and coped, closed by a bargeboard on each verge.
        self.raked_parapet("x", porch_face + BRICK_T / 2, porch_c - porch_half,
                           porch_c + porch_half, porch_roof_top,
                           role="house_wall", eave=3.25, gable="porch")
        verge_boards("porch", porch_face - VERGE_BOARD_T / 2,
                     porch_c - porch_half, porch_c + porch_half, porch_roof_top)

        # Main gables front and back: raked parapets, standing a fire break
        # clear of the slates, coped in stone and dressed with bargeboards.
        for gz, inner in ((front, 1), (back, -1)):
            self.raked_parapet("x", gz + inner * BRICK_T / 2, -half_w, half_w,
                               main_roof_top, role="house_wall", gable="main")
            verge_boards("main", gz - inner * VERGE_BOARD_T / 2,
                         -half_w, half_w, main_roof_top)
        verge_boards("wing", wing_face_outer - VERGE_BOARD_T / 2,
                     wing_lo, wing_hi, wing_roof_top)

        # The slated roof: plates on the side walls, two slopes, the ridge,
        # and the wing roof dying into the main front gable.
        self.wall_plate("z", front + 0.22, back - 0.22, -half_w + BRICK_T / 2, eave)
        self.wall_plate("z", front + 0.22, back - 0.22, half_w - BRICK_T / 2, eave)
        self.roof_slope("x", front + 0.22, back - 0.22, -half_w, plate_top,
                        run, 1)
        self.roof_slope("x", front + 0.22, back - 0.22, half_w, plate_top,
                        run, -1)
        cap_y = plate_top + rise + RAFTER_T + 0.01 - 0.06
        self.ridge_cap("x", front + 0.22, back - 0.22, 0.0, cap_y, eave)
        wing_plate = plate_top
        self.roof_slope("x", wing_face + 0.22, front, wing_lo, wing_plate,
                        wing_run, 1, gable="wing")
        self.roof_slope("x", wing_face + 0.22, front, wing_hi, wing_plate,
                        wing_run, -1, gable="wing")

        # Porch roof: a small pitch behind the porch gable.
        for s in (-1, 1):
            poly = [(porch_c + s * porch_half, porch_plate),
                    (porch_c, porch_apex), (porch_c, porch_apex + 0.09),
                    (porch_c + s * porch_half, porch_plate + 0.09)]
            self.mesh("slate_course", poly, wing_face - porch_face, 0.0, 0.0,
                      (porch_face + wing_face) / 2, "slate", along="x",
                      construction="slate course", gable="porch")
        poly = [(porch_c - 0.10, porch_apex), (porch_c, porch_apex + 0.17),
                (porch_c + 0.10, porch_apex), (porch_c + 0.10, porch_apex + 0.09),
                (porch_c, porch_apex + 0.26), (porch_c - 0.10, porch_apex + 0.09)]
        self.mesh("ridge_cap", poly, wing_face - porch_face, 0.0, 0.0,
                  (porch_face + wing_face) / 2, "slate", along="x",
                  construction="ridge cap", gable="porch")

        # Chimney breasts and stacks on the flank wall.
        for zc in (-half_dp * 0.45, half_dp * 0.45):
            self.chimney_breast(half_w, -1, zc - 0.65, zc + 0.65, eave)
        for zc in (-half_dp * 0.45, half_dp * 0.45):
            self.stack(half_w + 0.10, zc, eave)

        # Floors: joists front to back between the side walls, boards over.
        avoid = [(half_w - 0.46, half_w - 0.11, zc - 0.67, zc + 0.67)
                 for zc in (-half_dp * 0.45, half_dp * 0.45)]
        for f, y in enumerate(d.floors):
            self.floor(-half_w + BRICK_T / 2 + 0.16, half_w - BRICK_T / 2 - 0.16,
                       front + BRICK_T, back - BRICK_T, y, 0, avoid)


def build_document(request: BuildRequest) -> BuildDocument:
    d = Design.resolve(request)
    builder = VictorianBuilder(d)
    builder.compose()
    resolved = replace(request, architect=ARCHITECT.id, product=PRODUCT, style="victorian",
                      variant=d.variant, width=round(d.total_width, 6),
                      height=round(builder.ridge_y, 6), depth=round(d.depth, 6),
                      bays=(None if d.variant == "gothic-villa" else d.houses),
                      levels=d.levels, parameters={"finish": d.finish})
    required = VARIANT_REQUIRED[d.variant]
    return BuildDocument(architect=ARCHITECT, request=resolved, placements=builder.placements,
                         metadata={"victorian_rules": {
                             "variant": d.variant, "finish": d.finish,
                             "houses": d.houses, "frontage": round(d.frontage, 6),
                             "levels": d.levels, "storeys": list(d.storeys),
                             "pitch_deg": d.pitch_deg, "eave": round(d.eave, 6),
                             "ridge": round(builder.ridge_y, 6),
                             "bay_full": d.bay_full, "wing_side": d.wing_side,
                             "scope": "visual architecture; geometric checks are not "
                                      "structural engineering"},
                             "invariants": {"require_ground_contact": True,
                                            "required_roles": sorted(required)}})


# ---------------------------------------------------------------- audit
WALL_ROLES = ("front_wall", "back_wall", "party_wall", "outshut_wall", "bay_wall",
              "house_wall", "garden_wall", "garden_pier", "chimney_breast", "plinth")


def _bounds(placement):
    center = np.asarray(placement.position, dtype=float)
    half = np.asarray(placement.dimensions, dtype=float) / 2.0
    return center - half, center + half


def _distinct_lines(values, tol=0.02):
    """Cluster near-equal wall lines, so floating noise cannot split one.

    A coursed wall's stones share one axis. Rounding each stone to the
    centimetre can put two of them on opposite sides of a half-centimetre
    boundary and invent a second wall; the tolerance keeps one wall one line.
    """
    out = []
    for value in sorted(values):
        if not out or value - out[-1] > tol:
            out.append(value)
    return out


def _party_axes(lo, hi, indices, tol=0.12):
    """The centre line of each party wall, one line however the wall is coursed."""
    if len(indices) == 0:
        return []
    centers = (lo[indices, 0] + hi[indices, 0]) / 2.0
    return _distinct_lines((float(v) for v in centers), tol)


def _world_vertices(placement):
    """Reconstruct a mesh member's world vertices from its placement payload."""
    params = placement.parameters or {}
    verts = params.get("vertices")
    if not verts:
        return None
    return (np.asarray(verts, dtype=float) * np.asarray(placement.dimensions, dtype=float)
            + np.asarray(placement.position, dtype=float))


def _silhouette(placements, indices, samples, along, top=True):
    """Envelope of a group of meshes over one horizontal axis.

    Each solid is projected onto the (along, y) plane and its edges are
    interpolated, so a raking top course reports its true cut at every
    sample rather than the corner of its bounding box.
    """
    segments = []
    for i in indices:
        placement = placements[i]
        verts = _world_vertices(placement)
        faces = (placement.parameters or {}).get("faces")
        if verts is None or not faces:
            continue
        tri = np.asarray(faces, dtype=int)
        edges = np.concatenate((tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        a, b = verts[edges[:, 0]], verts[edges[:, 1]]
        segments.append(np.column_stack((a[:, along], a[:, 1], b[:, along], b[:, 1])))
    if not segments:
        return None
    edge = np.concatenate(segments)
    if not len(edge):
        return None
    low = np.minimum(edge[:, 0], edge[:, 2])
    high = np.maximum(edge[:, 0], edge[:, 2])
    dz = edge[:, 2] - edge[:, 0]
    flat = np.abs(dz) <= 1e-9
    out = np.full(len(samples), np.nan)
    for k, sample in enumerate(samples):
        mask = (low - 1e-6 <= sample) & (high + 1e-6 >= sample)
        if not mask.any():
            continue
        y0, y1 = edge[mask, 1], edge[mask, 3]
        count = int(mask.sum())
        frac = np.divide(sample - edge[mask, 0], dz[mask],
                         out=np.full(count, 0.5), where=~flat[mask])
        y = np.where(flat[mask], np.maximum(y0, y1),
                     y0 + (y1 - y0) * np.clip(frac, 0.0, 1.0))
        out[k] = y.max() if top else y.min()
    return out


def _raked_parapet_checks(placements, role, lo, hi):
    """Party walls and gables rake above both roof pitches and are coped.

    Each parapet is sampled from its finished courses, along the direction
    its head runs. The top of its own coping is compared with the slate
    surface on either side, so a flat-topped wall, a wall that stands the
    wrong height, or one that lost its coping all fail. A stack rising out of
    a party wall is the one place the measurement is excused.
    """
    from .quality import _check

    raked = np.array([bool((p.parameters or {}).get("parapet")) for p in placements])
    wall_idx = np.flatnonzero(raked & np.isin(role, ("party_wall", "house_wall")))
    cope_idx = np.flatnonzero(raked & (role == "parapet_coping"))
    slate_idx = np.flatnonzero(role == "slate_course")
    gable = np.array([str((p.parameters or {}).get("gable", "")) for p in placements])
    axes = np.array([str((p.parameters or {}).get("parapet_axis", "")) for p in placements])
    stations = np.array([float((p.parameters or {}).get("parapet_at", np.nan))
                         for p in placements])

    groups = {}
    for i in wall_idx:
        groups.setdefault((axes[i], round(float(stations[i]), 3),
                           str((placements[i].parameters or {}).get("gable", ""))),
                          []).append(i)
    upstand_bad, coping_bad, upstands = [], [], []
    for (axis, at, kind), group in sorted(groups.items()):
        wall = np.asarray(group)
        cope = cope_idx[(axes[cope_idx] == axis)
                        & (np.abs(stations[cope_idx] - at) <= 0.01)]
        along = 2 if axis == "z" else 0
        across = 0 if axis == "z" else 2
        p_lo = float(lo[wall, along].min())
        p_hi = float(hi[wall, along].max())
        samples = np.linspace(p_lo + 0.02, p_hi - 0.02, 97)
        top = _silhouette(placements, wall, samples, along, top=True)
        cap = _silhouette(placements, cope, samples, along, top=True) if len(cope) else None
        label = f"gable:{kind}" if kind else f"party:{at:g}"
        if top is None or np.isnan(top).all():
            upstand_bad.append(label)
            continue
        if cap is None or np.isnan(cap).any():
            coping_bad.append(label)
            continue
        lift = cap - top
        if np.nanmax(lift) > 0.16 or np.nanmin(lift) < 0.02:
            coping_bad.append(label)
        if axis == "z":
            near_sets = []
            for side in (-1, 1):
                if side > 0:
                    near_sets.append(slate_idx[(lo[slate_idx, across] >= at - 0.02)
                                               & (lo[slate_idx, across] <= at + 0.24)])
                else:
                    near_sets.append(slate_idx[(hi[slate_idx, across] <= at + 0.02)
                                               & (hi[slate_idx, across] >= at - 0.24)])
            keep = np.abs(samples) > STACK_W / 2.0 + 0.10
            sides = ((f"{label}:+d", near_sets[1]), (f"{label}:-d", near_sets[0]))
        else:
            near = slate_idx[gable[slate_idx] == ("" if kind == "main" else kind)]
            keep = np.ones(len(samples), dtype=bool)
            sides = ((label, near),)
        for name, near in sides:
            roof = _silhouette(placements, near, samples, along, top=True)
            if roof is None:
                # An exposed terrace end has roof on one side only.
                continue
            up = cap - roof
            probe = keep & ~np.isnan(up)
            if not probe.any():
                continue
            upstands.extend(up[probe].tolist())
            if float(up[probe].min()) < 0.25 or float(up[probe].max()) > 0.50:
                upstand_bad.append(name)
    return [_check(
        "victorian.raked_parapets",
        bool(groups) and not upstand_bad and not coping_bad,
        "Party walls and gables are raked to both roof pitches, stand 0.25 to 0.5 m clear of the slates, and carry their own coping.",
        "A parapet is flat-topped, stands clear of the slates by the wrong amount, or lacks its coping.",
        primitive_ids=sorted(set(upstand_bad + coping_bad)),
        observed={"upstand_m": [round(min(upstands), 3), round(max(upstands), 3)]
                  if upstands else None},
        expected={"upstand_m": [0.25, 0.50], "coping": True})]


def _gable_verge_checks(placements, role, lo, hi):
    """Every villa gable verge is closed by a bargeboard under its coping."""
    from .quality import _check

    gable = np.array([str((p.parameters or {}).get("gable", "")) for p in placements])
    raked = np.array([bool((p.parameters or {}).get("parapet")) for p in placements])
    board_idx = np.flatnonzero(role == "bargeboard")
    field = np.flatnonzero((role == "house_wall") & raked)
    bad, observed = [], {}
    for kind in ("main", "wing", "porch"):
        group = field[gable[field] == kind]
        if not len(group):
            bad.append(f"gable:{kind}")
            continue
        x_lo = float(lo[group, 0].min())
        x_hi = float(hi[group, 0].max())
        apex = (x_lo + x_hi) / 2.0
        samples = np.linspace(x_lo + 0.02, x_hi - 0.02, 65)
        wall_top = _silhouette(placements, group, samples, 0, top=True)
        if wall_top is None or np.isnan(wall_top).all():
            bad.append(f"gable:{kind}")
            continue
        wall_edge = float(np.nanmin(wall_top))
        wall_apex = float(np.nanmax(wall_top))
        boards = board_idx[gable[board_idx] == kind]
        observed[kind] = int(len(boards))
        for side in (-1, 1):
            edge = x_hi if side > 0 else x_lo
            rake = [j for j in boards
                    if side * (float((lo[j, 0] + hi[j, 0]) / 2.0) - apex) > 0]
            closes = any(
                float(hi[j, 1]) >= wall_apex - 0.16
                and float(lo[j, 1]) <= wall_edge - VERGE_BOARD_DEPTH + 0.09
                and (float(hi[j, 0]) >= edge - 0.05 if side > 0
                     else float(lo[j, 0]) <= edge + 0.05)
                and (float(lo[j, 0]) <= apex + 0.05 if side > 0
                     else float(hi[j, 0]) >= apex - 0.05)
                for j in rake)
            if not closes:
                bad.append(f"gable:{kind}:verge{side:+d}")
        # The board's foot must lap the slate along the rake. A board that
        # stops short of the covering leaves the verge open.
        slate = np.flatnonzero(role == "slate_course")
        slate = slate[gable[slate] == ("" if kind == "main" else kind)]
        samples = np.linspace(x_lo + 0.04, x_hi - 0.04, 41)
        foot = _silhouette(placements, boards, samples, 0, top=False) if len(boards) else None
        roof = _silhouette(placements, slate, samples, 0, top=True) if len(slate) else None
        if foot is None or roof is None:
            bad.append(f"gable:{kind}:lap")
        else:
            both = ~np.isnan(foot) & ~np.isnan(roof)
            if not both.any() or float(np.max(foot[both] - roof[both])) > 0.01:
                bad.append(f"gable:{kind}:lap")
    return [_check(
        "victorian.gable_verges", not bad,
        "Every gable verge is closed by a decorative bargeboard hung under the coping, so no rafter end shows.",
        "A gable lost the bargeboard that closes its verge.",
        primitive_ids=sorted(set(bad)), observed=observed,
        expected={"bargeboards_per_gable": 2})]


def victorian_checks(document, scene):
    """Check the composed geometry from the placements themselves.

    Every rule is measured from finished envelopes or from parameters the
    builder recorded while laying the member, so the audit survives a
    document round-trip with its mutable metadata stripped.
    """
    from .quality import _check, _envelope_connectivity_checks

    placements = list(document.placements)
    roles = {p.role for p in placements}
    variant = str(document.request.variant)
    required = VARIANT_REQUIRED.get(variant, set())
    checks = [_check("victorian.required_roles", required <= roles,
                     "The composition has its defining members.",
                     "The composition is missing defining members.",
                     observed=sorted(roles), expected=sorted(required))]
    checks.extend(_envelope_connectivity_checks(document, "victorian"))

    count = len(placements)
    lo = np.empty((count, 3))
    hi = np.empty((count, 3))
    for i, p in enumerate(placements):
        c = np.asarray(p.position, dtype=float)
        h = np.asarray(p.dimensions, dtype=float) / 2.0
        lo[i] = c - h
        hi[i] = c + h
    role = np.array([p.role for p in placements])
    kind = np.array([p.kind for p in placements])
    flank = np.array([(p.parameters or {}).get("construction") == "ashlar flank bond"
                      for p in placements])
    cube = np.array([p.block_type == "CUBE" for p in placements])
    wall_mask = cube & np.isin(role, WALL_ROLES)
    # Masonry for bearing and void probes: coursed walls and gable tiles,
    # with the plinth courses that joists pocket into.
    bearing_mask = (np.isin(kind, ("wall", "foundation"))) & ~flank

    # Bonded one-brick walls, with the thicker base allowed at street level.
    # A jamb bat cut narrower than a brick is honest bond, so the thickness
    # is measured across the wall, on the axis the stone was laid.
    bad = []
    for i in np.flatnonzero(wall_mask):
        axis = str((placements[i].parameters or {}).get("axis", ""))
        if axis == "z":
            t = hi[i, 0] - lo[i, 0]
        elif axis == "x":
            t = hi[i, 2] - lo[i, 2]
        else:
            t = min(hi[i, 0] - lo[i, 0], hi[i, 2] - lo[i, 2])
        if not (0.205 <= t <= 0.35):
            bad.append(placements[i].id)
    checks.append(_check(
        "victorian.one_brick_walls", bool(wall_mask.any()) and not bad,
        "Load-bearing walls are one bonded brick thick, one and a half at the base of three-storey fronts.",
        "A load-bearing wall left the one-brick band.",
        primitive_ids=bad, observed=len(bad), expected=0))

    caps = [p for p in placements
            if p.role == "ridge_cap" and "ridge_y" in (p.parameters or {})]
    ridge_y = max((float(p.parameters["ridge_y"]) for p in caps), default=None)
    eave_y = max((float(p.parameters["eave_y"]) for p in caps), default=None)
    if eave_y is None and wall_mask.any():
        eave_y = float(hi[wall_mask, 1].max())

    # Party walls carried up through the roof as fire breaks.
    if variant in ("two-up-two-down", "bay-fronted"):
        party_idx = np.flatnonzero(role == "party_wall")
        lines = _party_axes(lo, hi, party_idx)
        parapet_ok = bool(lines) and ridge_y is not None and len(lines) >= 2
        short = []
        for line in lines:
            near = party_idx[np.abs((lo[party_idx, 0] + hi[party_idx, 0]) / 2.0 - line) <= 0.16]
            if not len(near) or float(hi[near, 1].max()) < ridge_y + 0.25:
                short.append(round(line, 3))
        checks.append(_check(
            "victorian.party_walls_through_roof",
            parapet_ok and not short,
            "Every party wall is carried up through the roof as a fire break above the ridge.",
            "A party wall stops short of the ridge.",
            primitive_ids=[f"party:{s}" for s in short],
            observed={"party_lines": len(lines), "short": len(short)},
            expected={"party_lines": ">= 2", "short": 0}))

    # Party walls and every gable are raked to both roof pitches and finished
    # with a coping, standing a fire break above the slates.
    checks.extend(_raked_parapet_checks(placements, role, lo, hi))

    # Openings: every one carries a bearing arch, or a banded lintel course
    # spanning it on the jambs; and each light is a true void.
    records = [p for p in placements
               if p.role in ("gauged_arch", "porch_arch", "pointed_window")]
    unsupported, unpierced = [], []
    for record in records:
        pr = record.parameters
        axis = str(pr.get("axis", "x"))
        along_i = 0 if axis == "x" else 2
        cross_i = 2 - along_i
        center = float(pr["center"])
        wo = float(pr["span_half"])
        half = float(pr["half"])
        spring = float(pr["spring_y"])
        sill = float(pr.get("sill", 0.0))
        cross_c = float(record.position[cross_i])
        for s in (-1, 1):
            strip_a = center + s * half - 0.05
            strip_b = center + s * (wo + 0.08)
            jamb = bearing_mask & (lo[:, along_i] < max(strip_a, strip_b)) \
                & (hi[:, along_i] > min(strip_a, strip_b)) \
                & (hi[:, 1] >= spring - 0.02) & (lo[:, 1] < spring) \
                & (np.abs((lo[:, cross_i] + hi[:, cross_i]) / 2.0 - cross_c) <= 0.6)
            if not jamb.any():
                unsupported.append(record.id)
                break
        for light_c, light_h in (pr.get("lights") or [[center, half]]):
            light_c, light_h = float(light_c), float(light_h)
            # Flank bricks lie diagonally outside the wall plane; their AABBs
            # overstate their reach, so they cannot block a light on probe.
            probe = bearing_mask & ~flank \
                & (lo[:, along_i] > light_c - light_h + 0.08) \
                & (hi[:, along_i] < light_c + light_h - 0.08) \
                & (lo[:, 1] < spring - 0.1) & (hi[:, 1] > sill + 0.1) \
                & (np.abs((lo[:, cross_i] + hi[:, cross_i]) / 2.0 - cross_c) <= 0.6)
            if probe.any():
                unpierced.append(record.id)

    # The bay windows open through the main front wall under a banded
    # lintel: the course above the void spans it and bears on both jambs.
    if variant == "bay-fronted":
        bay_groups = {}
        for i, p in enumerate(placements):
            if p.role == "bay_plinth":
                bay_groups.setdefault(int((p.parameters or {}).get("house", 0)), []).append(i)
        for house, group in bay_groups.items():
            array = np.asarray(group)
            bay_c = float((lo[array, 0].min() + hi[array, 0].max()) / 2.0)
            front_band = np.flatnonzero(cube & (role == "front_wall"))
            covering = front_band[(lo[front_band, 0] <= bay_c) & (hi[front_band, 0] >= bay_c)]
            heads = [float(lo[j, 1]) for j in covering if lo[j, 1] > PLINTH_H + 0.5]
            if not heads:
                unsupported.append(f"bay:{house}")
                continue
            head = min(heads)
            course = front_band[np.abs(lo[front_band, 1] - head) <= 0.04]
            edges = sorted((float(lo[j, 0]), float(hi[j, 0])) for j in course)
            merged = []
            for a, b in edges:
                # Stones reconstruct with sub-micron joint drift; a tenth of
                # a millimetre is still one bonded course.
                if merged and a <= merged[-1][1] + 1e-4:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b))
                else:
                    merged.append((a, b))
            span = [bay_c - 1.55, bay_c + 1.55]
            covered = any(a <= span[0] and b >= span[1] for a, b in merged)
            if not covered:
                unsupported.append(f"bay:{house}")
            for s in (-1, 1):
                edge = bay_c + s * 1.7
                strip = (edge - 0.25, edge + 0.05) if s < 0 else (edge - 0.05, edge + 0.25)
                bearing = bearing_mask & (lo[:, 0] < strip[1]) & (hi[:, 0] > strip[0]) \
                    & (np.abs(hi[:, 1] - head) <= 0.04) & (lo[:, 1] < head)
                if not bearing.any():
                    unsupported.append(f"bay:{house}")
                    break
    checks.append(_check(
        "victorian.openings_bear", bool(records) and not unsupported and not unpierced,
        "Every opening carries a gauged arch or a banded lintel bearing on its jambs, and every light is pierced.",
        "An opening lost its arch, its bearing, or its void.",
        primitive_ids=sorted(set(unsupported + unpierced))))

    # Joists span front to back between the party walls and bear on both.
    joist_idx = np.flatnonzero(role == "joist")
    if variant in ("two-up-two-down", "bay-fronted"):
        boundaries = _party_axes(lo, hi, np.flatnonzero(role == "party_wall"))
        front_band = cube & (role == "front_wall")
        back_band = cube & (role == "back_wall")
        front_inner = float(hi[front_band, 2].max()) if front_band.any() else None
        back_inner = float(lo[back_band, 2].min()) if back_band.any() else None
    else:
        face = np.array([str((p.parameters or {}).get("wall_face", "")) for p in placements])
        side_band = cube & (role == "house_wall") & (face == "side")
        front_band = cube & (role == "house_wall") & (face == "front")
        back_band = cube & (role == "house_wall") & (face == "back")
        boundaries = []
        if side_band.any():
            boundaries = [float(lo[side_band, 0].min()) + BRICK_T,
                         float(hi[side_band, 0].max()) - BRICK_T]
        front_inner = float(hi[front_band, 2].max()) if front_band.any() else None
        back_inner = float(lo[back_band, 2].min()) if back_band.any() else None
    joist_bad = []
    for i in joist_idx:
        ok = front_inner is not None and back_inner is not None
        if ok and (lo[i, 2] > front_inner + 0.02 or hi[i, 2] < back_inner - 0.02):
            ok = False
        if ok and variant in ("two-up-two-down", "bay-fronted"):
            ok = any(boundaries[k] + 0.08 <= lo[i, 0] and hi[i, 0] <= boundaries[k + 1] - 0.08
                     for k in range(len(boundaries) - 1))
        elif ok:
            ok = lo[i, 0] >= boundaries[0] - 0.02 and hi[i, 0] <= boundaries[1] + 0.02
        if ok:
            for z_a, z_b in ((lo[i, 2], lo[i, 2] + 0.2), (hi[i, 2] - 0.2, hi[i, 2])):
                bearing = bearing_mask & (lo[:, 2] < z_b) & (hi[:, 2] > z_a) \
                    & (np.abs(hi[:, 1] - lo[i, 1]) <= 0.008) & (lo[:, 1] < lo[i, 1])
                if not bearing.any():
                    ok = False
        if not ok:
            joist_bad.append(placements[i].id)
    checks.append(_check(
        "victorian.joists_span_between_party_walls", bool(joist_idx.any()) and not joist_bad,
        "Joists span front to back inside one house bay and pocket into the walls at both ends.",
        "A joist left its bay, fell short of a wall, or lost its bearing.",
        primitive_ids=joist_bad))

    # The roof: slate over rafters. Every rafter records the pitch of its
    # own envelope. The period band applies to the common rafters of the
    # main roof and the villa wing; a lean-to outshut or bay is shallower
    # and is not held to that band.
    rafter_idx = np.flatnonzero(role == "rafter")
    slate_idx = np.flatnonzero(role == "slate_course")
    band = VILLA_PITCH_BAND if variant == "gothic-villa" else PITCH_BAND
    pitch_bad, pitch_missing = [], []
    pitches = []
    for i in rafter_idx:
        recorded = (placements[i].parameters or {}).get("pitch_deg")
        run_span = max(hi[i, 0] - lo[i, 0], hi[i, 2] - lo[i, 2])
        rise_span = (hi[i, 1] - lo[i, 1]) - RAFTER_T
        if recorded is None or run_span <= 0.01 or float(recorded) <= 1.0:
            pitch_missing.append(placements[i].id)
            continue
        pitch = math.degrees(math.atan2(rise_span, run_span))
        if abs(pitch - float(recorded)) > 0.75:
            pitch_missing.append(placements[i].id)
            continue
        if (placements[i].parameters or {}).get("construction") != "common rafter":
            continue
        pitches.append(round(pitch, 2))
        if not band[0] - 0.6 <= pitch <= band[1] + 0.6:
            pitch_bad.append(placements[i].id)
    slate_ok = bool(slate_idx.any()) and bool(pitches)
    if slate_ok:
        slate_hi_y = float(hi[slate_idx, 1].max())
        slate_lo_y = float(hi[slate_idx, 1].min())
        slate_ok = slate_hi_y >= ridge_y - 0.30 and slate_lo_y <= eave_y + 0.55
    checks.append(_check(
        "victorian.slated_pitch_in_band", slate_ok and not pitch_bad and not pitch_missing,
        f"The slated roof is pitched between {band[0]:g} and {band[1]:g} degrees over common rafters, courses from eaves to ridge.",
        "The roof pitch left the band, a rafter recorded no pitch, or the slate covering failed to reach eaves and ridge.",
        primitive_ids=pitch_bad + pitch_missing,
        observed=sorted(set(pitches))[:6], expected=[band[0], band[1]]))

    # Chimneys: a breast and stack for every party wall, pots and coping real.
    stack_idx = np.flatnonzero(role == "stack")
    breast_idx = np.flatnonzero(role == "chimney_breast")
    coping_idx = np.flatnonzero(role == "stack_coping")
    pot_idx = np.flatnonzero(role == "chimney_pot")
    stacks = {}
    for i in stack_idx:
        key = (round(float(lo[i, 0] + hi[i, 0]) / 2.0, 2), round(float(lo[i, 2] + hi[i, 2]) / 2.0, 2))
        entry = stacks.setdefault(key, {"base": lo[i, 1], "top": hi[i, 1]})
        entry["base"] = min(entry["base"], lo[i, 1])
        entry["top"] = max(entry["top"], hi[i, 1])
    stack_bad = []
    for key, entry in stacks.items():
        cap = [j for j in coping_idx
               if abs(lo[j, 1] - (entry["top"] + BED_T)) <= 0.04
               and lo[j, 0] <= key[0] + 0.45 and hi[j, 0] >= key[0] - 0.45
               and lo[j, 2] <= key[1] + 0.45 and hi[j, 2] >= key[1] - 0.45]
        if not cap:
            stack_bad.append(f"stack:{key[0]},{key[1]}")
            continue
        cap_top = float(hi[cap[0], 1])
        pots = [j for j in pot_idx
                if abs(lo[j, 1] - cap_top) <= 0.04
                and lo[j, 0] <= key[0] + 0.45 and hi[j, 0] >= key[0] - 0.45]
        if len(pots) < 2:
            stack_bad.append(f"stack:{key[0]},{key[1]}")
    flue_ok = bool(stacks) and bool(breast_idx.any()) and not stack_bad
    if variant in ("two-up-two-down", "bay-fronted"):
        for line in boundaries:
            near = [k for k in stacks if abs(k[0] - line) <= 0.35]
            if not near or max(stacks[k]["base"] for k in near) < ridge_y + 0.2:
                flue_ok = False
        flue_ok = flue_ok and len(stacks) >= len(boundaries)
    checks.append(_check(
        "victorian.stacks_on_party_walls", flue_ok,
        "Every party wall carries a grounded breast and a stack above the ridge, capped and potted.",
        "A stack is missing, unpotted, or does not rise on its party wall.",
        primitive_ids=stack_bad))

    # Frontage inside the period band for the variant.
    facade = None
    if variant in ("two-up-two-down", "bay-fronted"):
        front_idx = np.flatnonzero(role == "front_wall")
        if front_idx.any():
            span = float(hi[front_idx, 0].max() - lo[front_idx, 0].min()) + BRICK_T
            facade = span / len(boundaries[:-1] or [1])
        houses = len(boundaries) - 1
        band_lo, band_hi = FRONTAGE_BAND[variant]
        frontage_ok = facade is not None and band_lo - 1e-4 <= facade <= band_hi + 1e-4
        frontage_ok = frontage_ok and HOUSE_RANGE[variant][0] <= houses <= HOUSE_RANGE[variant][1]
    else:
        wall_lo = float(lo[wall_mask, 0].min()) if wall_mask.any() else None
        wall_hi = float(hi[wall_mask, 0].max()) if wall_mask.any() else None
        facade = (wall_hi - wall_lo) if wall_lo is not None else None
        band_lo, band_hi = FRONTAGE_BAND[variant]
        frontage_ok = facade is not None and band_lo - 0.15 <= facade <= band_hi + 0.15
    checks.append(_check(
        "victorian.frontage_band", frontage_ok,
        f"Each house keeps its frontage inside the {band_lo:g} to {band_hi:g} metre band.",
        "A frontage left the period band for its variant.",
        observed=None if facade is None else round(facade, 3),
        expected=[band_lo, band_hi]))

    # Variant character.
    if variant == "two-up-two-down":
        out_idx = np.flatnonzero(role == "outshut_wall")
        ok = bool(out_idx.any()) and back_inner is not None
        if ok:
            ok = float(lo[out_idx, 1].min()) <= 0.01 and \
                float(hi[out_idx, 2].max()) >= back_inner + OUTSHUT_D - 0.45
        checks.append(_check(
            "victorian.outshut", ok,
            "Each house carries a grounded rear outshut behind the back wall.",
            "The rear outshut is missing or does not reach the ground.",
            primitive_ids=[] if ok else ["outshut"]))
    if variant == "bay-fronted":
        garden_idx = np.flatnonzero(role == "garden_wall")
        rail_idx = np.flatnonzero(role == "railing_bar")
        path_idx = np.flatnonzero(role == "path_tile")
        bay_idx = np.flatnonzero(role == "bay_wall")
        ok = bool(garden_idx.any()) and bool(bay_idx.any())
        if ok:
            rail_len = float(hi[garden_idx, 0].max() - lo[garden_idx, 0].min())
            ok = len(rail_idx) >= max(4, rail_len / 0.3) and bool(path_idx.any())
        doors = [p for p in placements if p.role == "door_leaf"
                 and "door." in str(p.parameters.get("opening_id", ""))]
        street = float(lo[garden_idx, 2].min()) if len(garden_idx) else None
        for door in doors:
            dlo, dhi = _bounds(door)
            door_x = (dlo[0] + dhi[0]) / 2.0
            near = path_idx[(lo[path_idx, 0] < door_x + 0.5) & (hi[path_idx, 0] > door_x - 0.5)]
            if not len(near) or street is None \
                    or float(lo[near, 2].min()) > street + 0.4 \
                    or float(hi[near, 2].max()) < front_inner - BRICK_T - 0.1:
                ok = False
        checks.append(_check(
            "victorian.garden_bay", ok,
            "Canted bays stand on their own plinths, and a railed garden wall with a tiled path fronts each door.",
            "A bay, the garden wall, its railings, or the tiled path is missing.",
            primitive_ids=[] if ok else ["garden"]))
    if variant == "gothic-villa":
        barge = np.flatnonzero(role == "bargeboard")
        porch = [p for p in placements if p.role == "porch_arch"]
        pointed = [p for p in placements if p.role == "porch_pier"]
        light = [p for p in placements if p.role == "pointed_window"]
        bands_used = {p.parameters.get("material_role")
                      for p in placements if p.role == "house_wall"}
        ok = (len(barge) >= 6 and len(porch) >= 1 and bool(pointed) and bool(light)
              and {"brick", "band"} <= bands_used)
        pier_grounded = bool(pointed) and float(lo[np.flatnonzero(role == "porch_pier"), 1].min()) <= 0.01
        ok = ok and pier_grounded
        checks.append(_check(
            "victorian.gothic_villa", ok,
            "The villa keeps its steep gables with bargeboards, a pointed porch on grounded piers, and polychrome banding.",
            "The villa lost its bargeboards, its pointed porch, or its banding.",
            primitive_ids=[] if ok else ["villa"]))
        checks.extend(_gable_verge_checks(placements, role, lo, hi))
    return checks
