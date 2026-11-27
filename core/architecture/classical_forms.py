"""Composable Greco-Roman block-model typologies with bounded design rules.

Orders govern member proportions; typologies govern plans. These are reusable
visual assemblies, not archaeological reconstructions or engineering designs.
"""
from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import io
import math
import random
from typing import Dict

import numpy as np
import trimesh

from core.architectural_geometry import extrude_profile, mesh_parameters, tube_path
from core.generators.classical import ORDER_SPECS, _build_column
from core.scene import Scene
from .classical import ARCHITECT, _placement_from_block
from .masonry import (
    MAX_COURSE, MAX_STRETCHER, MIN_STONE, _envelope6, fill_box, iter_shell_wedges,
    radial_ashlar_mesh, radial_divisions, shell_cell_meshes, shell_wedge_mesh,
    stretcher_target, wall_axis,
)
from .schema import BuildDocument, BuildRequest, Placement, canonical_product_name

# Required roles are evaluated against the compiled document, not declared true.
FORMS = {
    "classical-portico": ("Roofed entrance colonnade", (14, 7, 5), ["column_shaft", "entablature", "roof"]),
    "classical-stoa": ("Long colonnaded hall with a rear wall", (24, 7, 8), ["column_shaft", "rear_wall", "roof"]),
    "classical-peristyle": ("Four colonnaded walks around an open court", (18, 7, 16), ["column_shaft", "court_walk", "roof"]),
    "classical-basilica": ("Raised nave, side aisles and an apsidal end", (18, 10, 26), ["column_shaft", "nave_roof", "aisle_roof", "apse"]),
    "classical-rotunda": ("Circular colonnade, annular bearing and hollow oculus dome", (16, 10, 16), ["column_shaft", "dome"]),
    "classical-triumphal-arch": ("Single or triple arched monument with attic", (14, 10, 4), ["arch", "pier", "attic"]),
    "classical-aqueduct": ("Aligned tiers of arcades carrying an open channel", (28, 12, 3), ["arch", "pier", "channel_bed", "channel_wall"]),
    "classical-theatre": ("Semicircular stepped cavea and stage building", (26, 7, 20), ["seating_tier", "stage", "stage_wall"]),
    "classical-amphitheatre": ("Elliptical stepped seating around an open arena", (28, 8, 22), ["seating_tier", "arena"]),
    "classical-forum": ("Open civic court with colonnades and a rear civic hall", (28, 9, 24), ["court_walk", "column_shaft", "civic_hall"]),
    "classical-villa": ("Courtyard house with surrounding room wings", (22, 7, 20), ["court_walk", "column_shaft", "room_wall"]),
    "classical-baths": ("Groin-vaulted bathing hall, domed caldarium and side chambers", (26, 10, 22), ["dome", "hall_wall", "pool_bed"]),
}
ORDERS = tuple(name.lower() for name in ORDER_SPECS)
MASONRY_FORMS = {"classical-triumphal-arch", "classical-aqueduct", "classical-theatre",
                 "classical-amphitheatre", "classical-baths"}
DETAIL_LEVELS = ("concept", "standard", "presentation")

# Coarse kinds stay a small shared vocabulary so ``required_kinds`` invariants
# and kind_counts metrics remain comparable across the whole catalog; the fine
# role keeps the typology-specific meaning.
ROLE_KINDS = {
    "foundation": "foundation",
    "column_shaft": "column",
    "entablature": "entablature",
    "roof": "roof", "nave_roof": "roof", "aisle_roof": "roof", "room_roof": "roof",
    "hall_roof": "roof", "chamber_roof": "roof", "attic": "roof", "dome": "roof", "drum": "roof",
    "apse_roof": "roof", "dome_rib": "roof", "vault_rib": "roof", "bearing_ring": "entablature",
    "rear_wall": "wall", "room_wall": "wall", "aisle_wall": "wall", "hall_wall": "wall",
    "stage_wall": "wall", "clerestory": "wall", "apse": "wall", "civic_hall": "wall",
    "channel_wall": "wall", "pool_rim": "wall",
    "pier": "pier", "hall_pier": "pier", "arch": "arch",
    "court_walk": "circulation", "aisle_step": "circulation",
    "arena": "slab", "channel_bed": "slab", "pool_bed": "slab",
    "stage": "platform", "seating_tier": "seating",
}


def role_kind(role):
    """Return the coarse catalog kind that owns a fine typology role."""
    try:
        return ROLE_KINDS[role]
    except KeyError:
        raise ValueError(f"role '{role}' has no coarse kind in the classical vocabulary") from None


# The physical substance behind each typology role.  Greek and Roman work
# reads as marble order members on limestone walls and foundations, terracotta
# tiled roofs over the colonnaded forms, and Roman concrete vaults, domes,
# channels and pools in the masonry forms.
FORM_SUBSTANCES: Dict[str, str] = {
    "foundation": "limestone",
    "bearing_ring": "limestone",
    "column_base": "marble",
    "column_base_lower": "marble",
    "column_base_upper": "marble",
    "column_base_scotia": "marble",
    "column_shaft": "marble",
    "column_shaft_astragal": "marble",
    "column_capital_neck": "marble",
    "column_capital_echinus": "marble",
    "column_capital_abacus": "marble",
    "column_capital_annuli": "marble",
    "column_capital_cushion": "marble",
    "column_capital_bell": "marble",
    "column_capital_helix": "marble",
    "column_capital_foliage": "marble",
    "column_capital_volute": "marble",
    "entablature": "marble",
    "roof": "clay-roof-tile",
    "nave_roof": "clay-roof-tile",
    "aisle_roof": "clay-roof-tile",
    "room_roof": "clay-roof-tile",
    "hall_roof": "clay-roof-tile",
    "chamber_roof": "clay-roof-tile",
    "apse_roof": "clay-roof-tile",
    "attic": "limestone",
    "dome": "roman-concrete",
    "drum": "limestone",
    "dome_rib": "roman-concrete",
    "vault_rib": "roman-concrete",
    "rear_wall": "limestone",
    "room_wall": "limestone",
    "aisle_wall": "limestone",
    "hall_wall": "fired-clay-brick",
    "stage_wall": "limestone",
    "clerestory": "limestone",
    "apse": "limestone",
    "civic_hall": "limestone",
    "channel_wall": "limestone",
    "pool_rim": "marble",
    "pool_bed": "roman-concrete",
    "channel_bed": "roman-concrete",
    "pier": "limestone",
    "hall_pier": "fired-clay-brick",
    "arch": "limestone",
    "court_walk": "limestone",
    "aisle_step": "limestone",
    "arena": "limestone",
    "stage": "limestone",
    "seating_tier": "limestone",
}
FORM_DEFAULT_SUBSTANCE = "limestone"


def list_products():
    return [{
        "name": name, "category": "assembly", "architect": ARCHITECT.to_dict(),
        "architect_id": ARCHITECT.id, "architect_title": ARCHITECT.title,
        "architect_version": ARCHITECT.version, "description": description,
        "defaults": {"width": size[0], "height": size[1], "depth": size[2], "variant": "auto", "detail": "standard", "bays": 5},
        "variants": ["auto"] if name in MASONRY_FORMS else ["auto", "random", *ORDERS],
        "detail_levels": list(DETAIL_LEVELS),
        "opening_modes": ["auto"], "parameters": [],
        "substances": sorted(set(FORM_SUBSTANCES.values()) | {FORM_DEFAULT_SUBSTANCE}),
        "rules": {"required_roles": list(roles),
                  "required_kinds": sorted({role_kind(role) for role in ("foundation", *roles)}),
                  "bays": [2, 12],
                  "levels": [1, 3] if name == "classical-aqueduct" else None,
                  "scope": "stylized architectural block model"},
    } for name, (description, size, roles) in FORMS.items()]


def _dimension(name, value, default):
    result = default if value is None else value
    if isinstance(result, bool) or not math.isfinite(float(result)) or not 1 <= float(result) <= 200:
        raise ValueError(f"{name} must be between 1 and 200")
    return float(result)


def _radial_solid(profile, sections=48, z_scale=1.0):
    """Revolve a closed radius/height section without filling its central void."""
    vertices = [(r * math.cos(math.tau * i / sections), y,
                 r * z_scale * math.sin(math.tau * i / sections))
                for r, y in profile for i in range(sections)]
    faces = []
    for row in range(len(profile)):
        following = (row + 1) % len(profile)
        for i in range(sections):
            j = (i + 1) % sections
            a, b = row * sections + i, row * sections + j
            c, d = following * sections + j, following * sections + i
            faces.extend(((a, b, c), (a, c, d)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _sector_solid(rx, rz, inner, outer, start, end, bottom, top, segments=12):
    """A curved annular wedge: true radial faces, level tread and vertical riser."""
    angles = np.linspace(start, end, segments + 1)
    profile = [(rx * outer * math.cos(t), rz * outer * math.sin(t)) for t in angles]
    profile += [(rx * inner * math.cos(t), rz * inner * math.sin(t)) for t in reversed(angles)]
    mesh = extrude_profile(profile, top - bottom)
    transform = np.array([[1., 0., 0., 0.], [0., 0., -1., (top + bottom) / 2],
                          [0., 1., 0., 0.], [0., 0., 0., 1.]])
    mesh.apply_transform(transform)
    return mesh


def _dome_shell(radius, rise, thickness, bottom, sections=48, layers=12, oculus=.11):
    """Elliptical meridian with separate intrados, extrados and open oculus."""
    cells = _dome_shell_cells(radius, rise, thickness, bottom, sections, layers, oculus)
    return trimesh.util.concatenate(cells)


def _dome_shell_cells(radius, rise, thickness, bottom, sections=48, layers=12, oculus=.11):
    """One watertight shell cell per meridian band so a dome is ashlars, not a cap."""
    sections = max(8, int(sections))
    layers = max(4, int(layers))
    elev = np.linspace(0.0, math.acos(oculus), layers + 1)
    inner_r, inner_rise = radius - thickness, max(rise - thickness, rise * 0.5)
    meshes = []
    for j in range(layers):
        a0, a1 = float(elev[j]), float(elev[j + 1])
        for i in range(sections):
            t0, t1 = math.tau * i / sections, math.tau * (i + 1) / sections
            outer = [
                (radius * math.cos(a0) * math.cos(t0), bottom + rise * math.sin(a0),
                 radius * math.cos(a0) * math.sin(t0)),
                (radius * math.cos(a0) * math.cos(t1), bottom + rise * math.sin(a0),
                 radius * math.cos(a0) * math.sin(t1)),
                (radius * math.cos(a1) * math.cos(t1), bottom + rise * math.sin(a1),
                 radius * math.cos(a1) * math.sin(t1)),
                (radius * math.cos(a1) * math.cos(t0), bottom + rise * math.sin(a1),
                 radius * math.cos(a1) * math.sin(t0)),
            ]
            inner = [
                (inner_r * math.cos(a1) * math.cos(t0), bottom + inner_rise * math.sin(a1),
                 inner_r * math.cos(a1) * math.sin(t0)),
                (inner_r * math.cos(a1) * math.cos(t1), bottom + inner_rise * math.sin(a1),
                 inner_r * math.cos(a1) * math.sin(t1)),
                (inner_r * math.cos(a0) * math.cos(t1), bottom + inner_rise * math.sin(a0),
                 inner_r * math.cos(a0) * math.sin(t1)),
                (inner_r * math.cos(a0) * math.cos(t0), bottom + inner_rise * math.sin(a0),
                 inner_r * math.cos(a0) * math.sin(t0)),
            ]
            mesh = trimesh.Trimesh(
                vertices=outer + inner,
                faces=[(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                       (0, 7, 6), (0, 6, 1), (1, 6, 5), (1, 5, 2),
                       (2, 5, 4), (2, 4, 3), (3, 4, 7), (3, 7, 0)],
                process=False)
            if mesh.volume < 0:
                mesh.invert()
            mesh.fix_normals(multibody=False)
            meshes.append(mesh)
    return meshes


def _groin_point(ix, iz, n, half_width, half_depth, spring, rise, offset):
    x = 2 * ix / n - 1
    z = 2 * iz / n - 1
    y = spring + rise * math.sqrt(max(0.0, 1.0 - min(x * x, z * z)))
    return (x * half_width, y + offset, z * half_depth)


def _groin_vault_cells(half_width, half_depth, spring, rise, thickness, subdivisions=16):
    """Thick groin cells. Each patch is a cubit-scale shell, the hall stays open."""
    n = max(4, int(subdivisions))
    meshes = []

    def point(xn, zn, offset):
        y = spring + rise * math.sqrt(max(0.0, 1.0 - min(xn * xn, zn * zn)))
        return (xn * half_width, y + offset, zn * half_depth)

    def emit(x0, z0, x1, z1):
        outer = [point(x0, z0, 0.0), point(x1, z0, 0.0), point(x1, z1, 0.0), point(x0, z1, 0.0)]
        inner = [point(x0, z1, -thickness), point(x1, z1, -thickness),
                 point(x1, z0, -thickness), point(x0, z0, -thickness)]
        mesh = trimesh.Trimesh(
            vertices=outer + inner,
            faces=[(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7),
                   (0, 7, 6), (0, 6, 1), (1, 6, 5), (1, 5, 2),
                   (2, 5, 4), (2, 4, 3), (3, 4, 7), (3, 7, 0)],
            process=False)
        if mesh.volume < 0:
            mesh.invert()
        mesh.fix_normals(multibody=False)
        meshes.append(mesh)

    def rec(x0, z0, x1, z1, depth=0):
        pts = [point(x0, z0, 0.0), point(x1, z0, 0.0), point(x1, z1, 0.0), point(x0, z1, 0.0),
               point(x0, z0, -thickness), point(x1, z0, -thickness),
               point(x1, z1, -thickness), point(x0, z1, -thickness)]
        xs, ys, zs = zip(*pts)
        dy = max(ys) - min(ys)
        dx = max(xs) - min(xs)
        dz = max(zs) - min(zs)
        if (dy <= MAX_COURSE + 1e-6 and dx <= MAX_STRETCHER + 1e-6
                and dz <= MAX_STRETCHER + 1e-6):
            emit(x0, z0, x1, z1)
            return
        # Depth 8 stops a steep springing cell on a long vault while the lift
        # is still taller than one course. Keep the same halving until it fits.
        if depth >= 24:
            emit(x0, z0, x1, z1)
            return
        if dy >= dx and dy >= dz:
            # Split the steeper plan direction so a springing cell is not one 2 m lift.
            if dx >= dz:
                xm = (x0 + x1) / 2.0
                rec(x0, z0, xm, z1, depth + 1)
                rec(xm, z0, x1, z1, depth + 1)
            else:
                zm = (z0 + z1) / 2.0
                rec(x0, z0, x1, zm, depth + 1)
                rec(x0, zm, x1, z1, depth + 1)
        elif dx >= dz:
            xm = (x0 + x1) / 2.0
            rec(x0, z0, xm, z1, depth + 1)
            rec(xm, z0, x1, z1, depth + 1)
        else:
            zm = (z0 + z1) / 2.0
            rec(x0, z0, x1, zm, depth + 1)
            rec(x0, zm, x1, z1, depth + 1)

    step = 2.0 / n
    for iz in range(n):
        z0 = -1.0 + iz * step
        for ix in range(n):
            x0 = -1.0 + ix * step
            rec(x0, z0, x0 + step, z0 + step)
    return meshes


def _groin_vault(half_width, half_depth, spring, rise, thickness, subdivisions=16):
    """Intersecting barrel surfaces, closed as a shell with four open arch ends."""
    return trimesh.util.concatenate(
        _groin_vault_cells(half_width, half_depth, spring, rise, thickness, subdivisions))


def _shell_cell_midpoint(left, right):
    return tuple((left[axis] + right[axis]) / 2.0 for axis in range(3))


def _bisect_shell_cell(cell):
    """Halve a thick shell cell on its longest edge into two smaller cells.

    ``iter_shell_wedges`` only promises distinct envelopes between the two
    children of one split; it knows nothing about a neighbouring facet. This
    keeps that promise when a cell must be divided again to clear a neighbour.
    """
    apex, p, q, apex_lo, p_lo, q_lo = cell
    p_q, p_q_lo = _shell_cell_midpoint(p, q), _shell_cell_midpoint(p_lo, q_lo)
    apex_p, apex_p_lo = _shell_cell_midpoint(apex, p), _shell_cell_midpoint(apex_lo, p_lo)
    apex_q, apex_q_lo = _shell_cell_midpoint(apex, q), _shell_cell_midpoint(apex_lo, q_lo)
    splits = [
        (math.dist(p, q), (apex, p, p_q, apex_lo, p_lo, p_q_lo),
         (apex, p_q, q, apex_lo, p_q_lo, q_lo)),
        (math.dist(apex, p), (apex, apex_p, q, apex_lo, apex_p_lo, q_lo),
         (apex_p, p, q, apex_p_lo, p_lo, q_lo)),
        (math.dist(apex, q), (apex, p, apex_q, apex_lo, p_lo, apex_q_lo),
         (apex_q, p, q, apex_q_lo, p_lo, q_lo)),
    ]
    splits.sort(key=lambda item: -item[0])
    for _length, first, second in splits:
        if _envelope6(first) != _envelope6(second):
            return [first, second]
    return []


_SHELL_REFINEMENT_LIMIT = 24


def _facet_shell_cells(cells, facet, claimed, depth=0):
    """Return one facet's shell cells, none repeating another facet's envelope.

    ``shell_cell_meshes`` welds cells that share an envelope inside one facet.
    Facets are divided one at a time, though, and two neighbours meet on a
    shared meridian. A cell hugging that meridian from either side can end up
    with the same axis-aligned envelope, which the audit reads as a duplicate
    primitive. The later facet halves such a cell until its envelope is its
    own, so both stones stay whole and every primitive is distinct.
    """
    unique = []
    for cell in cells:
        envelope = _envelope6(cell)
        owner = claimed.get(envelope)
        if owner is None or owner == facet:
            claimed.setdefault(envelope, facet)
            unique.append(cell)
            continue
        if shell_wedge_mesh(*cell).volume <= 0:
            # A flat cell compiles to nothing, so it cannot duplicate a stone.
            continue
        children = _bisect_shell_cell(cell) if depth < _SHELL_REFINEMENT_LIMIT else []
        if not children:
            unique.append(cell)
            continue
        unique.extend(_facet_shell_cells(children, facet, claimed, depth + 1))
    return unique


class _Composer:
    def __init__(self, request):
        self.request = request
        self.name = canonical_product_name(request.product)
        _, defaults, required = FORMS[self.name]
        self.required = list(required)
        self.w, self.h, self.d = [_dimension(key, getattr(request, key), default)
                                for key, default in zip(("width", "height", "depth"), defaults)]
        if request.parameters:
            raise ValueError(f"{self.name} does not accept parameters")
        if request.openings != "auto":
            raise ValueError(f"{self.name} authors its own openings; use openings=auto")
        if request.detail not in set(DETAIL_LEVELS):
            raise ValueError("detail must be concept, standard, or presentation")
        self.bays = 5 if request.bays is None else request.bays
        if isinstance(self.bays, bool) or not isinstance(self.bays, int) or not 2 <= self.bays <= 12:
            raise ValueError("bays must be an integer between 2 and 12")
        self.levels = 2 if request.levels is None else request.levels
        if self.name != "classical-aqueduct" and request.levels is not None:
            raise ValueError(f"{self.name} does not support levels")
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or not 1 <= self.levels <= 3:
            raise ValueError("levels must be an integer between 1 and 3")
        variant = request.variant
        if self.name in MASONRY_FORMS and variant != "auto":
            raise ValueError(f"{self.name} uses masonry piers, not a column order; use variant=auto")
        if variant == "auto":
            variant = "doric"
        elif variant == "random":
            variant = random.Random(request.seed).choice(ORDERS)
        if variant not in ORDERS:
            raise ValueError(f"variant must be auto, random, or one of {', '.join(ORDERS)}")
        self.order = ORDER_SPECS[variant.title()]
        # Masonry forms carry no column order, so no placement claims one.
        self.order_parameters = {} if self.name in MASONRY_FORMS else {"order": self.order.name}
        # Prevent needle buildings and excessively compressed member systems.
        if not 0.25 <= self.w / self.d <= 16 or self.h > min(self.w, self.d) * 6:
            raise ValueError("dimensions cannot accommodate this classical composition")
        self.placements = []
        self.counts = Counter()
        self.base = self.h * 0.04
        self.root = self.masonry("foundation", (0, self.base / 2, 0), (self.w, self.base, self.d), parent=None)
        self.column_cache = {}
        # The player the building is for; None keeps the historic plan.
        player_of = getattr(request, "player_profile", None)
        self.player = player_of() if callable(player_of) else None

    def add(self, role, position, dimensions, mesh="CUBE", parent="root", kind=None, **parameters):
        index = self.counts[role]
        self.counts[role] += 1
        pid = f"{self.name}.{role}.{index:04d}"
        color = (222, 213, 191, 255) if "roof" not in role and role != "dome" else (160, 112, 87, 255)
        self.placements.append(Placement(id=pid, kind=kind or role_kind(role), role=role, block_type=mesh,
            position=tuple(position), dimensions=tuple(dimensions), material=color,
            parent_id=self.root if parent == "root" else parent,
            parameters={**self.order_parameters, **parameters},
            substance=FORM_SUBSTANCES.get(role, FORM_DEFAULT_SUBSTANCE)))
        return pid

    def sculpt(self, role, mesh, **parameters):
        position, dimensions, payload = mesh_parameters(mesh)
        return self.add(role, position, dimensions, mesh="MESH", **payload, **parameters)

    def tile_profile(self, role, polygon, depth, translation=(0, 0, 0), **parameters):
        """Split a long extrusion into cubit stretchers, then course the profile."""
        from .cathedral import iter_fitted_profile_pieces
        run = float(depth)
        count = max(1, int(math.ceil(abs(run) / MAX_STRETCHER - 1e-9)))
        chunk = run / count
        host = f"{self.name}.{role}.{self.counts[role]:04d}"
        last = None
        for index in range(count):
            zc = translation[2] - run / 2.0 + (index + 0.5) * chunk
            for piece, course in iter_fitted_profile_pieces(
                    polygon, chunk, 0, translation[0], translation[1], zc):
                mesh = extrude_profile(piece, chunk)
                mesh.apply_translation((translation[0], translation[1], zc))
                last = self.sculpt(
                    role, mesh, **{
                        "construction": "ashlar running bond", **parameters,
                        "unit": "profile_course", "course": course, "host_id": host,
                    })
        return last

    def masonry(self, role, position, dimensions, openings=None, **parameters):
        host = f"{self.name}.{role}.{self.counts[role]:04d}"
        stones = fill_box(position, dimensions, self.request.detail, openings or (), wall_axis(dimensions))
        last = None
        for stone in stones:
            last = self.add(role, stone.position, stone.dimensions,
                            construction="ashlar running bond", unit=stone.unit,
                            course=stone.course, bond="running", host_id=host,
                            axis=stone.axis, **parameters)
        return last

    def _pave(self, role, position, dimensions):
        """Course a paving box. Never emit one plan-sized slab."""
        width, height, depth = (float(value) for value in dimensions)
        if height <= MAX_COURSE + 1e-6 and max(width, depth) <= MAX_STRETCHER + 1e-6:
            return self.add(role, position, dimensions,
                            construction="ashlar running bond", unit="paving")
        return self.masonry(role, position, dimensions)

    def course_sector(self, role, rx, rz, inner, outer, start, end, bottom, top,
                      translation=(0, 0, 0), facets=None, segments=1, **parameters):
        host = f"{self.name}.{role}.{self.counts[role]:04d}"
        radius = max(rx, rz) * outer
        if facets is None:
            facets = self._arc_facets(start, end, rx, rz, outer)
        last = None
        for a0, a1, y0, y1, course, index in radial_divisions(
                start, end, bottom, top, radius, self.request.detail, facets):
            last = self._lay_radial_ashlar(
                role, rx, rz, inner, outer, a0, a1, y0, y1, segments,
                translation, host, course, index, parameters)
        return last

    def _lay_radial_ashlar(self, role, rx, rz, inner, outer, a0, a1, y0, y1,
                           segments, translation, host, course, facet, parameters, depth=0):
        """Emit a radial ashlar, halving a wedge that is still one slab.

        ``radial_divisions`` drops gaps shorter than half a stone using a metre
        threshold on angles in radians. On a large cavea every facet of a
        cuneus is under that angle, so they weld back into one member. Wedges
        that already fit are sculpted as they were.
        """
        mesh = radial_ashlar_mesh(rx, rz, inner, outer, a0, a1, y0, y1, segments)
        if translation != (0, 0, 0):
            mesh.apply_translation(translation)
        if depth < 24 and _radial_mesh_exceeds_ashlar(mesh):
            laid = self._halve_radial_ashlar(
                role, rx, rz, inner, outer, a0, a1, y0, y1, segments,
                translation, host, course, facet, parameters, depth, mesh)
            if laid is not None:
                return laid
        meta = {"construction": "ashlar running bond", **parameters,
                "unit": "radial_ashlar", "course": course, "facet": facet, "host_id": host}
        return self.sculpt(role, mesh, **meta)

    def _halve_radial_ashlar(self, role, rx, rz, inner, outer, a0, a1, y0, y1,
                             segments, translation, host, course, facet, parameters, depth, mesh):
        span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
        height = float(span[1])
        along = max(float(span[0]), float(span[2]))
        too_tall = height > MAX_COURSE + 1e-6
        too_long = along > MAX_STRETCHER + 1e-6
        radial_span = abs(outer - inner) * max(abs(rx), abs(rz))
        arc_span = abs(a1 - a0) * max(abs(rx), abs(rz)) * max(abs(inner), abs(outer))

        def lay(**changes):
            args = dict(
                role=role, rx=rx, rz=rz, inner=inner, outer=outer, a0=a0, a1=a1,
                y0=y0, y1=y1, segments=segments, translation=translation, host=host,
                course=course, facet=facet, parameters=parameters, depth=depth + 1)
            args.update(changes)
            return self._lay_radial_ashlar(**args)

        if too_tall and (y1 - y0) > 1e-4 and (not too_long or height >= along):
            mid = (y0 + y1) / 2.0
            lay(y1=mid)
            return lay(y0=mid, course=course + 1)
        if too_long and (outer - inner) > 1e-6 and (radial_span >= arc_span or (a1 - a0) <= 1e-6):
            mid = (inner + outer) / 2.0
            lay(outer=mid)
            return lay(inner=mid)
        if too_long and (a1 - a0) > 1e-6:
            mid = (a0 + a1) / 2.0
            lay(a1=mid)
            return lay(a0=mid)
        if too_tall and (y1 - y0) > 1e-4:
            mid = (y0 + y1) / 2.0
            lay(y1=mid)
            return lay(y0=mid, course=course + 1)
        return None

    def _radial_bands(self, inner, outer, rx, rz):
        span = (outer - inner) * max(abs(rx), abs(rz))
        # Keep AABB under MAX_STRETCHER even when a wedge sits on the diagonal.
        module = min(stretcher_target(self.request.detail), 0.85)
        count = max(1, int(math.ceil(span / module - 1e-9)))
        step = (outer - inner) / count
        return [(inner + i * step, inner + (i + 1) * step) for i in range(count)]

    def _arc_facets(self, start, end, rx, rz, outer):
        arc = abs(end - start) * max(abs(rx), abs(rz)) * outer
        module = min(stretcher_target(self.request.detail), 0.85)
        return max(1, int(math.ceil(arc / module - 1e-9)))

    def column(self, x, z, bottom, height, clearance):
        spec = self.order
        diameter = height / (spec.shaft_height_ratio + spec.base_height_ratio + spec.capital_height_ratio)
        if diameter * spec.capital_width_ratio * 1.3 >= clearance:
            raise ValueError("columns would collide: increase span, reduce height, or reduce bays")
        key = (height,)
        if key not in self.column_cache:
            scene = Scene(use_collision=False)
            scene.support_check_mode = "OFF"
            scene.enable_strict_overlap_prevention = False
            with redirect_stdout(io.StringIO()):
                _build_column(scene, spec, 0, 0, 0, diameter / 2,
                    diameter * spec.shaft_height_ratio, diameter * spec.base_height_ratio,
                    diameter * spec.base_width_ratio, diameter * spec.capital_height_ratio,
                    diameter * spec.capital_width_ratio)
            counters = Counter()
            self.column_cache[key] = [_placement_from_block(block, counters) for block in scene.blocks]
        for part in self.column_cache[key]:
            params = deepcopy(part.parameters)
            # Cylinder parameters describe local sizes, so translation is safe.
            self.add(part.role, (part.position[0] + x, part.position[1] + bottom, part.position[2] + z),
                     part.dimensions, part.block_type, kind="column",
                     **{k: v for k, v in params.items() if k != "order"})

    def row(self, axis, fixed, span, bottom, height, count=None):
        count = count or self.bays + 1
        spacing = span / (count - 1)
        for i in range(count):
            along = -span / 2 + i * spacing
            self.column(along if axis == "x" else fixed, fixed if axis == "x" else along,
                        bottom, height, spacing)
        thickness = height * 0.12
        dims = (span + thickness, thickness, thickness * 1.7) if axis == "x" else (thickness * 1.7, thickness, span + thickness)
        self.masonry("entablature", (0 if axis == "x" else fixed, bottom + height + thickness / 2,
                                    fixed if axis == "x" else 0), dims)

    def open_arch(self, role, x, bottom, z, width, height, depth, thickness=None):
        """Jambs, bearing and two half-heads. The opening stays a true void."""
        half = float(width) / 2.0
        t = float(thickness or min(width * 0.16, max(depth * 0.35, 0.28)))
        inner = max(half - t, width * 0.12)
        bearing = min(max(t, height * 0.08), height * 0.22)
        spring = min(max(height * 0.22, 0.16), height - bearing - 0.2)
        inner_rise = max(height - spring - bearing, 0.2)
        if spring > MIN_STONE * 0.5:
            self.masonry(role, (x - (half + inner) / 2.0, bottom + spring / 2.0, z),
                         (half - inner, spring, depth))
            self.masonry(role, (x + (half + inner) / 2.0, bottom + spring / 2.0, z),
                         (half - inner, spring, depth))
        if bearing > MIN_STONE * 0.5:
            self.masonry(role, (x, bottom + height - bearing / 2.0, z), (width, bearing, depth))
        n = 8 if self.request.detail == "concept" else 12
        left = [(-half, spring), (-half, height - bearing), (0.0, height - bearing)]
        left.extend((inner * math.cos(th), spring + inner_rise * math.sin(th))
                    for th in np.linspace(math.pi / 2.0, math.pi, n + 1)[1:])
        right = [(half, spring), (half, height - bearing), (0.0, height - bearing)]
        right.extend((inner * math.cos(th), spring + inner_rise * math.sin(th))
                     for th in np.linspace(math.pi / 2.0, 0.0, n + 1)[1:])
        self.tile_profile(role, left, depth, translation=(x, bottom, z),
                          construction="ashlar running bond")
        self.tile_profile(role, right, depth, translation=(x, bottom, z),
                          construction="ashlar running bond")

    def course_rib(self, role, points, radius, closed=False, **parameters):
        """Segment a tube so a vault rib is stretchers, not one 8 m MESH."""
        from .cathedral import _polyline_samples
        spacing = max(MIN_STONE * 2, min(MAX_STRETCHER, MAX_COURSE) - 2.0 * radius - 0.05)
        samples = _polyline_samples(points, spacing, closed=closed)
        host = f"{self.name}.{role}.{self.counts[role]:04d}"
        last = None
        for index, (a, b) in enumerate(zip(samples[:-1], samples[1:])):
            if float(np.linalg.norm(np.asarray(b) - np.asarray(a))) < MIN_STONE * 0.5:
                continue
            last = self.sculpt(
                role, tube_path([a, b], radius, sections=6, closed=False),
                construction="ashlar stack", unit="profile_course",
                course=index, host_id=host, **parameters)
        return last

    def lunette(self, role, along, plane, centre, half, spring, rise, depth, window=.62, **parameters):
        """Wall the arch under a vault's edge, from its springing to its crown,
        round a thermal window: a half-ellipse of the same shape, split in
        three by two mullions. ``along`` is the axis the wall runs on and
        ``plane`` its place on the other axis."""
        from .cathedral import _profile_mesh, iter_fitted_profile_pieces
        angle = 0.0 if along == "x" else math.pi / 2
        # Rotating a quarter turn sends the profile's x to -z; the lunette is
        # symmetric, so only its place matters.
        x, z = (centre, plane) if along == "x" else (plane, centre)
        host = f"{self.name}.{role}.{self.counts[role]:04d}"
        pieces = []
        wedges = 10 if self.request.detail == "concept" else 14
        edges = np.linspace(0.0, math.pi, wedges + 1)
        for a0, a1 in zip(edges[:-1], edges[1:]):
            arc = [(math.cos(a), math.sin(a)) for a in (a0, a1)]
            pieces.append([(half * c, spring + rise * s_) for c, s_ in arc]
                          + [(window * half * c, spring + window * rise * s_) for c, s_ in reversed(arc)])
        bar = max(.22, half * .035)
        for u in (-window * half / 3, window * half / 3):
            top = spring + window * rise * math.sqrt(max(0.0, 1 - (u / (window * half)) ** 2))
            pieces.append([(u - bar / 2, spring), (u + bar / 2, spring), (u + bar / 2, top), (u - bar / 2, top)])
        course = 0
        for polygon in pieces:
            for piece, _ in iter_fitted_profile_pieces(polygon, depth, angle, x, 0, z):
                self.sculpt(role, _profile_mesh(piece, depth, angle, x, 0, z),
                            **{"construction": "ashlar running bond", **parameters,
                               "unit": "profile_course", "course": course, "host_id": host})
                course += 1

    def portico(self, stoa=False):
        w, h, d, b = self.w, self.h, self.d, self.base
        ch = h * 0.70
        self.row("x", -d * 0.32, w * 0.86, b, ch)
        if stoa:
            self.masonry("rear_wall", (0, b + ch / 2, d * 0.36), (w * 0.94, ch, d * 0.08))
        else:
            self.row("x", d * 0.32, w * 0.86, b, ch)
        self.masonry("roof", (0, b + ch * 1.12 + h * 0.06, 0), (w * 0.97, h * 0.12, d * 0.92))

    def court(self, mode):
        w, h, d, b = self.w, self.h, self.d, self.base
        ch, band = h * 0.64, min(w, d) * 0.16
        for z in (-d * 0.34, d * 0.34):
            self.row("x", z, w * 0.68, b, ch)
            self._pave("court_walk", (0, b + h * 0.01, z), (w * 0.9, h * 0.02, band))
            self.masonry("roof", (0, b + ch * 1.12 + h * 0.03, z), (w * 0.9, h * 0.06, band))
        # Side rows exclude corners, preserving unique supports.
        for x in (-w * 0.34, w * 0.34):
            spacing = d * 0.68 / self.bays
            for i in range(1, self.bays):
                self.column(x, -d * 0.34 + spacing * i, b, ch, spacing)
            self.masonry("entablature", (x, b + ch * 1.06, 0), (ch * 0.204, ch * 0.12, d * 0.68))
            self._pave("court_walk", (x, b + h * 0.01, 0), (band, h * 0.02, d * 0.68 - band))
            self.masonry("roof", (x, b + ch * 1.12 + h * 0.03, 0), (band, h * 0.06, d * 0.68 - band))
        if mode == "forum":
            # The hall wall rises to roof level so its roof rests on it directly,
            # instead of depending on reaching the court roofs across the plan.
            self.masonry("civic_hall", (0, b + ch * 0.56, d * 0.46), (w * 0.9, ch * 1.12, d * 0.05))
            self.masonry("hall_roof", (0, b + ch * 1.12 + h * 0.03, d * 0.43), (w * 0.9, h * 0.06, d * 0.12))
        if mode == "villa":
            for x in (-w * 0.46, w * 0.46):
                self.masonry("room_wall", (x, b + ch / 2, 0), (w * 0.025, ch, d * 0.92))
                for i in range(self.bays + 1):
                    z = -d * 0.44 + d * 0.88 * i / self.bays
                    self.masonry("room_wall", (x * 0.9, b + ch / 2, z), (w * 0.10, ch, d * 0.025))
                self.masonry("room_roof", (x * 0.9, b + ch * 1.12 + h * 0.03, 0), (w * 0.12, h * 0.06, d * 0.92))

    def basilica(self):
        w, h, d, b = self.w, self.h, self.d, self.base
        ch = h * 0.60
        eaves, ridge, roof_t = b + ch * 1.12 + h * .12, h * .965, h * .035
        for x in (-w * 0.24, w * 0.24):
            self.row("z", x, d * 0.70, b, ch)
            self.masonry("clerestory", (x, b + ch * 1.12 + h * 0.06, 0), (w * 0.03, h * 0.12, d * 0.74))
        for x in (-w * 0.46, w * 0.46):
            self.masonry("aisle_wall", (x, b + ch * 0.38, 0), (w * 0.04, ch * 0.76, d * 0.78))
            sign = -1 if x < 0 else 1
            inner, outer = sign * w * .235, sign * w * .49
            profile = [(inner, b + ch * 1.12), (outer, b + ch * .76),
                       (outer, b + ch * .76 + roof_t), (inner, b + ch * 1.12 + roof_t)]
            self.tile_profile("aisle_roof", profile, d * .78,
                              construction="raking aisle roof; inner and outer wall bearings")
        # Two rakes; a chevron shell double-backs when Y-clipped.
        left = [(-w * .27, eaves), (0, ridge), (0, ridge + roof_t), (-w * .27, eaves + roof_t)]
        right = [(0, ridge), (w * .27, eaves), (w * .27, eaves + roof_t), (0, ridge + roof_t)]
        self.tile_profile("nave_roof", left, d * .74, translation=(0, 0, -d * .02),
                          construction="pitched open-bottom roof shell")
        self.tile_profile("nave_roof", right, d * .74, translation=(0, 0, -d * .02),
                          construction="pitched open-bottom roof shell")
        # Every polygon wall shares its complete mitred face with its neighbour.
        # The centre remains a real open room rather than a collection of cubes.
        facets = {"concept": 7, "standard": 11, "presentation": 15}[self.request.detail]
        self.course_sector("apse", w * .255, d * .13, .88, 1.,
                           0, math.pi, b, eaves, translation=(0, 0, d * .35),
                           segments=1, masonry="mitred polygon wall")
        # An open-front semi-conical shell bears continuously on the apse wall.
        # Each facet is a single closed wedge, not a solid cap filling the room.
        # Neighbouring facets share a meridian, so their cells are claimed
        # against one envelope register and no two stones repeat an envelope.
        angles = np.linspace(0, math.pi, facets + 1)
        claimed = {}
        for i in range(facets):
            a, c = angles[i:i + 2]
            outer = [(w * .255 * math.cos(t), eaves, d * .35 + d * .13 * math.sin(t)) for t in (a, c)]
            top = (0, ridge, d * .35)
            top_hi = (top[0], top[1] + roof_t, top[2])
            outer_hi = [(x, y + roof_t, z) for x, y, z in outer]
            host = f"{self.name}.apse_roof.{self.counts['apse_roof']:04d}"
            cells = _facet_shell_cells(iter_shell_wedges(
                top, outer[0], outer[1], top_hi, outer_hi[0], outer_hi[1]), i, claimed)
            for course, roof in enumerate(shell_cell_meshes(cells)):
                self.sculpt("apse_roof", roof, facet=i, construction="ashlar running bond",
                            unit="profile_course", course=course, host_id=host)

    def rotunda(self):
        w, h, d, b = self.w, self.h, self.d, self.base
        if abs(w - d) > min(w, d) * 0.05:
            raise ValueError("rotunda requires width and depth within 5% of each other")
        radius, ch, count = min(w, d) * 0.36, h * 0.56, max(8, self.bays * 2)
        spacing = 2 * radius * math.sin(math.pi / count)
        for i in range(count):
            angle = math.tau * i / count
            self.column(radius * math.cos(angle), radius * math.sin(angle), b, ch, spacing)
        drum_h = h * 0.08
        sections = {"concept": 32, "standard": 48, "presentation": 64}[self.request.detail]
        inner, outer = radius * .81, min(w, d) * .44
        for lo, hi in self._radial_bands(inner, outer, 1.0, 1.0):
            self.course_sector("drum", 1.0, 1.0, lo, hi, 0, math.tau,
                               b + ch, b + ch + drum_h, segments=2,
                               construction="annular bearing; open interior")
        host = f"{self.name}.dome.{self.counts['dome']:04d}"
        for course, cell in enumerate(_dome_shell_cells(
                outer, h * .30, min(w, h) * .028, b + ch + drum_h, sections, 14)):
            self.sculpt("dome", cell, construction="ashlar running bond",
                        unit="profile_course", course=course, host_id=host)
        if self.request.detail != "concept":
            for i in range(12):
                azimuth = math.tau * i / 12
                points = [(outer * math.cos(t) * math.cos(azimuth),
                           b + ch + drum_h + h * .30 * math.sin(t),
                           outer * math.cos(t) * math.sin(azimuth))
                          for t in np.linspace(0, math.acos(.11), 16)]
                self.course_rib("dome_rib", points, min(w, h) * .005)

    def arcade(self, aqueduct):
        w, h, d, b = self.w, self.h, self.d, self.base
        levels = self.levels if aqueduct else 1
        bays = self.bays if aqueduct else (1 if self.bays < 3 else 3)
        tier = (h - b - h * 0.16) / levels
        pier = w / (bays + 1) * 0.25
        span = (w - (bays + 1) * pier) / bays
        for level in range(levels):
            bottom = b + level * tier
            arch_h = tier * 0.84
            for i in range(bays + 1):
                x = -w / 2 + pier / 2 + i * (span + pier)
                self.masonry("pier", (x, bottom + arch_h / 2, 0), (pier, arch_h, d * 0.8))
            for i in range(bays):
                x = -w / 2 + pier + span / 2 + i * (span + pier)
                self.open_arch("arch", x, bottom, 0, span, arch_h, d * 0.8)
            self.masonry("entablature", (0, bottom + arch_h + tier * 0.08, 0), (w, tier * 0.16, d * 0.9))
        top = b + levels * tier
        if aqueduct:
            self.masonry("channel_bed", (0, top + h * 0.025, 0), (w, h * 0.05, d))
            for z in (-d * 0.43, d * 0.43):
                self.masonry("channel_wall", (0, top + h * 0.105, z), (w, h * 0.11, d * 0.14))
        else:
            self.masonry("attic", (0, top + h * 0.08, 0), (w, h * 0.16, d))

    def theatre(self, full):
        w, h, d, b = self.w, self.h, self.d, self.base
        tiers = {"concept": 3, "standard": 6, "presentation": 9}[self.request.detail]
        sectors = max(8 if full else 6, self.bays * 2 if full else self.bays)
        sweep = math.tau if full else math.pi
        aisle_angle = min(.035, sweep / sectors * .10)

        def half_gap(r):
            """Half the angle an aisle takes at radius fraction ``r``.

            Historic aisles are a fixed angle. For a player they keep the
            player's door width from the front row to the back, as real scalae
            do, so they don't fan out into wide wedges at the top of the cavea;
            they never take more than a third of a wedge of seats.
            """
            if self.player is None:
                return aisle_angle / 2
            half = self.player.clearances().door_width / 2
            return max(aisle_angle / 2, min(sweep / sectors / 6,
                                            math.asin(min(1.0, half / (min(w, d) * r)))))
        # An amphitheatre built for a player gets its two axial gates.
        gates = 0.0
        if full and self.player is not None and sectors % 2 == 0:
            gates = self.player.clearances().door_width / 2 + .15
        for tier in range(tiers):
            inner = .23 + tier * .23 / tiers
            outer = .23 + (tier + 1) * .23 / tiers
            step = (h - b) / tiers
            y1 = b + (tier + 1) * step
            y0 = b
            # One aisle width per tier, shared by the seats and the steps, so
            # the steps fill the break between wedges exactly with no slots.
            g = half_gap(inner)
            # For a player each tier is cut into rows of seats, as a real cavea
            # is, two aisle steps to a row, so each row meets the aisle at a
            # step's height and the stairs read as scalae between fine rows.
            rows = 1
            if self.player is not None:
                rows = max(1, int(math.ceil(step / (2 * self.player.step_height) - 1e-9)))
            bands = []
            for row in range(rows):
                top = y1 if rows == 1 else b + (tier + (row + 1) / rows) * step
                r0, r1 = inner + (outer - inner) * row / rows, inner + (outer - inner) * (row + 1) / rows
                bands += [(lo, hi, top) for lo, hi in self._radial_bands(r0, r1, w, d)]
            for lo, hi, y1 in bands:
                if gates:
                    # The cavea parts along the long axis for a gate as wide
                    # as the player's door, at every ring of seating.
                    gap = math.asin(min(1.0, gates / (d * lo)))
                for i in range(sectors):
                    a0 = sweep * i / sectors + g
                    a1 = sweep * (i + 1) / sectors - g
                    if gates:
                        if i in (0, sectors // 2):
                            a0 = max(a0, sweep * i / sectors + gap)
                        if i + 1 in (sectors // 2, sectors):
                            a1 = min(a1, sweep * (i + 1) / sectors - gap)
                    self.course_sector(
                        "seating_tier", w, d, lo, hi, a0, a1, y0, y1,
                        facets=self._arc_facets(a0, a1, w, d, hi), segments=6,
                        tier=tier, cuneus=i, construction="radial masonry cavea")
            # Half-riser steps make the radial breaks usable access routes; for a
            # player, as many steps as keep every riser within their step.
            risers = 2 * rows
            for i in range(1, sectors if not full else sectors + 1):
                angle = sweep * i / sectors
                if gates and i in (sectors // 2, sectors):
                    continue
                for half in range(risers):
                    lo = inner + (outer - inner) * half / risers
                    hi = inner + (outer - inner) * (half + 1) / risers
                    top = b + (tier + (half + 1) / risers) * (h - b) / tiers
                    self.course_sector(
                        "aisle_step", w, d, lo, hi,
                        angle - g, angle + g,
                        b, top, segments=1, tier=tier, aisle=i, half=half,
                        construction="radial aisle steps")
        if full:
            rx, rz = w * 0.22, d * 0.22
            y = b + h * 0.005
            pad_h = max(h * 0.01, MIN_STONE)
            module = min(MAX_STRETCHER, 1.5)
            nx = max(1, int(math.ceil(2 * rx / module - 1e-9)))
            nz = max(1, int(math.ceil(2 * rz / module - 1e-9)))
            dx, dz = 2 * rx / nx, 2 * rz / nz
            host = f"{self.name}.arena.{self.counts['arena']:04d}"
            for i in range(nx):
                for j in range(nz):
                    cx = -rx + (i + 0.5) * dx
                    cz = -rz + (j + 0.5) * dz
                    if (cx / max(rx, MIN_STONE)) ** 2 + (cz / max(rz, MIN_STONE)) ** 2 > 1.0:
                        continue
                    self.add("arena", (cx, y, cz), (dx, pad_h, dz),
                             construction="ashlar running bond", unit="paving",
                             course=0, host_id=host)
        else:
            if self.player is None:
                self.masonry("stage", (0, b + h * 0.10, -d * 0.12), (w * 0.72, h * 0.20, d * 0.18))
            else:
                # Pull the stage front back to leave a passage, the parodos, as
                # wide as the player's door between the stage and the cavea.
                front = max(min(-d * 0.03, -self.player.clearances().door_width - .1), -d * 0.21 + 1.0)
                back = -d * 0.21
                self.masonry("stage", (0, b + h * 0.10, (front + back) / 2), (w * 0.72, h * 0.20, front - back))
            self.masonry("stage_wall", (0, b + h * 0.36, -d * 0.23), (w * 0.8, h * 0.72, d * 0.04))

    def baths(self):
        w, h, d, b = self.w, self.h, self.d, self.base
        spring, half_w, half_d, centre_z = b + h * .54, w * .25, d * .265, -d * .15
        wall_t = min(w, d) * .035
        rise = h * .30
        for x in (-half_w, half_w):
            # The side walls rise to the springing, and a thermal window in the
            # lunette above each lights the hall over the chamber roofs.
            self.masonry("hall_wall", (x, b + h * .27, centre_z), (wall_t, h * .54, half_d * 2))
            self.lunette("hall_wall", "z", x, centre_z, half_d, spring, rise, wall_t)
            for z in (centre_z - half_d, centre_z + half_d):
                self.masonry("hall_pier", (x, b + h * .27, z), (wall_t * 1.8, h * .54, wall_t * 1.8))
        # The ends close under their arches too, each with a great doorway:
        # in from the court at the front, on to the hot rooms at the back.
        door_w, door_h = half_w * .68, (spring - b) * .62
        if self.player is not None:
            clear = self.player.clearances()
            door_w = max(door_w, clear.door_width + .4)
            door_h = min(max(door_h, clear.door_height + .3), spring - b - .4)
        span = 2 * (half_w - wall_t * .9)
        for z, end in ((centre_z - half_d, "front"), (centre_z + half_d, "back")):
            self.masonry("hall_wall", (0, b + (spring - b) / 2, z), (span, spring - b, wall_t),
                         [{"center": 0.0, "width": door_w, "sill": 0.0, "height": door_h, "id": f"hall.{end}"}])
            self.lunette("hall_wall", "x", z, 0.0, half_w - wall_t * .9, spring, rise, wall_t)
        host = f"{self.name}.hall_roof.{self.counts['hall_roof']:04d}"
        for course, cell in enumerate(_groin_vault_cells(
                half_w, half_d, spring, rise, h * .035,
                12 if self.request.detail == "concept" else 20)):
            cell = cell.copy()
            cell.apply_translation((0, 0, centre_z))
            self.sculpt("hall_roof", cell, construction="ashlar running bond",
                        unit="profile_course", course=course, host_id=host)
        for sign_x in (-1, 1):
            for sign_z in (-1, 1):
                points = [(half_w * t * sign_x, spring + h * .30 * math.sqrt(max(0., 1 - t * t)),
                           centre_z + half_d * t * sign_z) for t in np.linspace(0, 1, 14)]
                self.course_rib("vault_rib", points, h * .012)
        # Circular caldarium: a real bearing wall and ring carry a hollow dome.
        radius, dome_z, wall_h = min(w * .225, d * .195), d * .285, h * .52
        self.course_sector("hall_wall", radius, radius, .88, 1.,
                           -math.pi / 2 + .34, math.pi * 1.5 - .34, b, b + wall_h,
                           translation=(0, 0, dome_z), segments=2,
                           construction="coursed caldarium wall; front entry")
        entry_z = dome_z - radius * math.cos(.34)
        doorway = radius * math.sin(.34) * 2
        self.open_arch("arch", 0, b, entry_z, doorway, wall_h, radius * .18)
        for lo, hi in self._radial_bands(.88, 1., radius, radius):
            self.course_sector("drum", radius, radius, lo, hi, 0, math.tau,
                               b + wall_h, b + h * .60, translation=(0, 0, dome_z),
                               segments=2, construction="annular dome bearing")
        host = f"{self.name}.dome.{self.counts['dome']:04d}"
        for course, cell in enumerate(_dome_shell_cells(
                radius, h * .30, min(h, radius) * .055, b + h * .60, 48, 14, oculus=.10)):
            cell = cell.copy()
            cell.apply_translation((0, 0, dome_z))
            self.sculpt("dome", cell, construction="ashlar running bond",
                        unit="profile_course", course=course, host_id=host)
        self.masonry("pool_bed", (0, b + h * 0.005, 0), (w * 0.28, h * 0.01, d * 0.17))
        for x in (-w * 0.145, w * 0.145):
            self.masonry("pool_rim", (x, b + h * 0.025, 0), (w * 0.01, h * 0.05, d * 0.19))
        for z in (-d * 0.09, d * 0.09):
            self.masonry("pool_rim", (0, b + h * 0.025, z), (w * 0.28, h * 0.05, d * 0.01))
        for x in (-w * 0.38, w * 0.38):
            self.masonry("chamber_roof", (x, b + h * 0.53, 0), (w * 0.25, h * 0.06, d * 0.90))
            # A low tiled gable over each range of chambers, not a flat slab.
            eave, ridge = b + h * .56, b + h * .56 + w * .25 * .22
            self.tile_profile("chamber_roof", [(x - w * .125, eave), (x + w * .125, eave), (x, ridge)], d * .90,
                              construction="tiled gable on the chamber vaults")
            # Round-headed windows along the outer wall, one to each chamber bay.
            bays = max(2, int(round(d * .90 / 4.5)))
            pitch = d * .90 / bays
            windows = [{"center": (i + .5) * pitch - d * .45, "width": min(1.4, pitch * .3),
                        "sill": h * .22, "height": h * .16, "id": f"window.{'lr'[x > 0]}.{i}"}
                       for i in range(bays)]
            self.masonry("hall_wall", (x * 1.25, b + h * 0.25, 0), (w * 0.03, h * 0.50, d * 0.90), windows)
            # Close the ends of the range under its gables, each with a door.
            inner, outer = abs(x) / .38 * .265 + w * .015, abs(x) * 1.25 - w * .015
            door_w, door_h = min(1.6, (outer - inner) * .3), h * .30
            if self.player is not None:
                clear = self.player.clearances()
                door_w = min(max(door_w, clear.door_width + .2), (outer - inner) - 1.0)
                door_h = min(max(door_h, clear.door_height + .2), h * .5 - .4)
            for z, end in ((-d * .45, "front"), (d * .45, "back")):
                self.masonry("hall_wall", (math.copysign((inner + outer) / 2, x), b + h * .25, z),
                             (outer - inner, h * .50, w * .02),
                             [{"center": 0.0, "width": door_w, "sill": 0.0, "height": door_h,
                               "id": f"range.{'lr'[x > 0]}.{end}"}])
            # Both longitudinal edges and the chamber divisions carry the slab;
            # the rear half no longer hangs beside the detached circular hall.
            self.masonry("hall_wall", (x / .38 * .265, b + h * .25, 0),
                         (w * .03, h * .50, d * .90))
            for z in (-d * .15, d * .15, d * .44):
                if self.player is None or z > d * .2:
                    self.masonry("hall_wall", (x, b + h * .25, z), (w * .23, h * .50, d * .02))
                    continue
                # The chambers open one into the next through a doorway sized to
                # the player, so the back rooms are not sealed cells.
                clear = self.player.clearances()
                door = {"center": 0.0, "width": min(clear.door_width, w * .23 - .6), "sill": 0.0,
                        "height": min(clear.door_height, h * .50 - .4),
                        "id": f"chamber.{'lr'[x > 0]}.{'ab'[z > 0]}"}
                self.masonry("hall_wall", (x, b + h * .25, z), (w * .23, h * .50, d * .02), [door])


def _radial_mesh_exceeds_ashlar(mesh):
    """True when a radial wedge's world AABB is larger than one ashlar."""
    span = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    return (float(span[1]) > MAX_COURSE + 1e-6
            or max(float(span[0]), float(span[2])) > MAX_STRETCHER + 1e-6)


def _scaled_dimension(value, rng):
    return max(1, min(200, int(round(value * rng.uniform(0.7, 1.5)))))


def _controls_are_buildable(name, controls, variants):
    from .quality import PASS, classical_form_checks

    for variant in variants:
        try:
            document = build_document(BuildRequest(product=name, variant=variant, **controls))
        except ValueError:
            return False
        if any(check.status != PASS for check in classical_form_checks(document, name)):
            return False
    return True


def random_controls(product, rng=None):
    """Return randomized controls that every order of ``product`` can build.

    Candidates are rejected unless the composer accepts them and the intrinsic
    classical checks pass for every order the form offers.  Those are the two
    gates a dimension choice can fail; the remaining audit checks are properties
    of the placement generator rather than of the controls.  Falls back to the
    descriptor defaults if no candidate survives.
    """
    name = canonical_product_name(product)
    _, defaults, _ = FORMS[name]
    rng = random.Random() if rng is None else rng
    variants = ("auto",) if name in MASONRY_FORMS else ORDERS
    for _ in range(64):
        controls = {
            "width": _scaled_dimension(defaults[0], rng),
            "height": _scaled_dimension(defaults[1], rng),
            "depth": _scaled_dimension(defaults[2], rng),
            "bays": rng.randint(2, 12),
            "detail": rng.choice(DETAIL_LEVELS),
            "levels": rng.randint(1, 3) if name == "classical-aqueduct" else None,
        }
        if name == "classical-rotunda":
            controls["depth"] = controls["width"]
        if _controls_are_buildable(name, controls, variants):
            return controls
    return {"width": defaults[0], "height": defaults[1], "depth": defaults[2], "bays": 5,
            "detail": "standard", "levels": 2 if name == "classical-aqueduct" else None}


def build_document(request: BuildRequest) -> BuildDocument:
    c = _Composer(request)
    form = c.name.removeprefix("classical-")
    if form in {"portico", "stoa"}:
        c.portico(form == "stoa")
    elif form in {"peristyle", "forum", "villa"}:
        c.court(form)
    elif form == "basilica":
        c.basilica()
    elif form == "rotunda":
        c.rotunda()
    elif form in {"triumphal-arch", "aqueduct"}:
        c.arcade(form == "aqueduct")
    elif form in {"theatre", "amphitheatre"}:
        c.theatre(form == "amphitheatre")
    elif form == "baths":
        c.baths()
    return BuildDocument(architect=ARCHITECT, request=request, placements=c.placements, metadata={
        "classical_rules": {"order": None if c.name in MASONRY_FORMS else c.order.name, "typology": form, "bays": c.bays,
                            "scope": "stylized architectural block model; no structural certification"},
        "invariants": {"required_roles": ["foundation", *c.required],
                       "required_kinds": sorted({role_kind(role) for role in ("foundation", *c.required)}),
                       "min_placements": 3},
    })
