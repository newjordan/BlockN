"""Persian Architect: iwan courts, squinch domes and muqarnas, masonry only.

Dimensions follow the four-centred tradition. Only the design phase draws
randomness: changing ``detail`` never moves an iwan, the dome or a minaret.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.architectural_geometry import extrude_profile, mesh_parameters

from .masonry import fill_box, iter_stack
from .schema import ArchitectInfo, BuildDocument, BuildRequest, Placement


ARCHITECT = ArchitectInfo("persian", "1.0.0", "Persian Architect")
PRODUCT = "iwan-court"
VARIANTS = ("four-iwan", "chahar-taq", "caravanserai")
DETAILS = ("concept", "standard", "presentation")
CONTROLS = ("finish",)
DEFAULTS = {"width": 36, "height": 14, "depth": 46, "variant": "auto", "detail": "standard"}
FINISHES = {
    # brick, tile, render, stone, vault, accent
    "buff-turquoise": ("c9a06a", "2ea6a6", "e2cfa8", "b98d5f", "caa571", "1f7f8f"),
    "sandstone-lapis": ("c2a173", "2b5c9e", "dcc9a4", "a8895e", "c4a375", "1e3f6e"),
    "rose-azure": ("c99b83", "4f7fb5", "e4cfb2", "ab7f62", "c69c7d", "2f5f8f"),
}
# The physical substance behind each palette key. The finish names the
# masonry: buff and rose finishes are fired brick laid in lime mortar, and the
# sandstone finish is cut sandstone. Vaults and domes follow the masonry,
# turquoise, lapis and azure work is glazed tile, and render is lime plaster.
FINISH_MASONRY = {
    "buff-turquoise": ("fired-clay-brick", "limestone"),
    "sandstone-lapis": ("sandstone", "sandstone"),
    "rose-azure": ("fired-clay-brick", "limestone"),
}
MATERIAL_SUBSTANCES = {"tile": "glazed-tile", "accent": "glazed-tile", "render": "lime-plaster"}


def _substances(finish):
    masonry, stone = FINISH_MASONRY[finish]
    return {**MATERIAL_SUBSTANCES, "brick": masonry, "vault": masonry, "stone": stone}


REQUIRED = {"court_paving", "pishtaq", "muqarnas_cell", "entrance_arch"}
VARIANT_REQUIRED = {
    "four-iwan": {"court_paving", "riwaq_pier", "riwaq_bay", "riwaq_gallery",
                  "riwaq_roof", "enclosure_wall", "pishtaq", "iwan", "iwan_wall",
                  "entrance_arch", "sanctuary_wall", "squinch", "transition_band",
                  "drum", "dome_course", "muqarnas_cell", "minaret", "minaret_drum",
                  "minaret_balcony", "minaret_cap"},
    "chahar-taq": {"court_paving", "chamber_wall", "chamber_arch", "squinch",
                   "transition_band", "drum", "dome_course", "muqarnas_cell"},
    "caravanserai": {"court_paving", "cell_wall", "cell_roof", "cell_front",
                     "cell_arch", "corner_tower", "riwaq_pier", "riwaq_bay",
                     "riwaq_roof", "iwan", "iwan_wall", "pishtaq",
                     "entrance_arch", "muqarnas_cell"},
}
KINDS = {
    "court_paving": "slab", "riwaq_pier": "pier", "riwaq_bay": "arch",
    "riwaq_gallery": "slab", "riwaq_roof": "slab", "pishtaq": "wall",
    "iwan": "vault", "iwan_wall": "wall", "entrance_arch": "arch",
    "sanctuary_wall": "wall", "enclosure_wall": "wall", "cell_front": "wall",
    "cell_arch": "arch", "chamber_wall": "wall", "chamber_arch": "arch",
    "squinch": "vault", "transition_band": "wall", "drum": "drum",
    "dome_course": "dome", "dome_knob": "finial", "muqarnas_cell": "corbel",
    "minaret": "tower", "minaret_drum": "tower", "minaret_balcony": "corbel",
    "minaret_cap": "finial", "corner_tower": "tower", "cell_wall": "wall",
    "cell_roof": "slab",
}

# Fixed construction modules (detail changes stone coursing, never these).
WALL_T = 0.7          # load-bearing wall thickness
FRAME_T = 0.8         # pishtaq frame thickness
ARCH_T = 0.5          # arch ring thickness
PIER_W = 0.7          # arcade pier width
PLINTH_H = 0.45       # springing plinth for ground-level arches
IMPOST = 0.02         # skewback lip: the springing course bites into the ring
SLAB_T = 0.28         # gallery floor and riwaq roof thickness
SHAFT_STEP = 0.01     # pier shafts set back a finger behind the ring's face
FRONT_T = 0.45        # hujra front wall thickness in the caravanserai
ARCH_RING_MAX = 0.35  # arch ring depth before it splits into ring courses
VAULT_CHUNK = 1.4     # vault ring pitch along the hall
MUQ_CELL = 0.42       # muqarnas cell plan width
MUQ_DEPTH = 0.22      # muqarnas cell height and course thickness
MUQ_STEP = 0.10       # hood recession per tier, at most half a cell depth
BAND_H = 0.8          # sixteen-sided band height
BAND_T = 0.6          # band radial thickness
BAND_IN = 0.15        # band projection inside the chamber face
DRUM_H = 1.0          # drum height
EPS = 1e-6


def list_products():
    finishes = [(name, *roles) for name, roles in FINISHES.items()]
    return [{"name": PRODUCT, "category": "assembly", "architect": ARCHITECT.to_dict(),
             "architect_id": ARCHITECT.id, "architect_title": ARCHITECT.title,
             "architect_version": ARCHITECT.version,
             "description": "Persian court architecture: four-centred arches, squinch domes, muqarnas portals",
             "defaults": dict(DEFAULTS), "variants": ["auto", "random", *VARIANTS],
             "detail_levels": list(DETAILS), "opening_modes": ["auto"],
             "parameters": list(CONTROLS), "finishes": finishes,
             "substances": sorted({sub for finish in FINISHES for sub in _substances(finish).values()}),
             "rules": {"width": [18, 60], "depth": [20, 70], "height": [8, 22],
                       "arch_rise_to_span": [0.45, 0.75], "dome_apex_to_diameter": [0.55, 0.8],
                       "iwan_depth_to_width": [0.8, 1.6], "muqarnas_tiers": [3, 6],
                       "drum_diameter_to_chamber": [0.55, 1.0],
                       "minaret_above_dome_apex": [0.3, 4.0],
                       "scope": "architectural visualization and modular asset generation"}}]


def _palette(finish):
    colors = dict(zip(("brick", "tile", "render", "stone", "vault", "accent"), FINISHES[finish]))
    return {name: (*bytes.fromhex(value), 255) for name, value in colors.items()}


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


def four_centred_arc(half: float, rise: float, segments: int = 12) -> np.ndarray:
    """One half of a Persian four-centred pointed arch, springing to apex.

    Lower arcs of large radius rise from the springing with a steep tangent;
    small crown arcs meet on the axis at the pointed apex. Returns points
    from (half, 0) to (0, rise): monotone in x and y.
    """
    half, rise = float(half), float(rise)
    if half <= EPS or rise <= EPS:
        raise ValueError("four-centred arch needs positive span and rise")
    # Join point splits the flank from the crown. The flank circle's centre
    # sits on the springing line inside the span; monotone rise requires the
    # join above the 45-degree line off the springing.
    join_x = (1.0 - 0.18) * half
    join_y = 0.5 * rise
    flank_dx = half - join_x
    flank_r = (flank_dx * flank_dx + join_y * join_y) / (2.0 * flank_dx)
    flank_cx = half - flank_r
    # Crown circle: centre on the far side of the axis, well below springing.
    # Passing through the join and the apex fixes the offset d; d > 0 across
    # the whole 0.45..0.75 rise-to-span band, so the mirrored crown arcs meet
    # at a true point instead of a rounded crown.
    crown_cy = -3.0 * rise
    apex_d = ((rise - crown_cy) ** 2 - (join_y - crown_cy) ** 2 - join_x ** 2) / (2.0 * join_x)
    crown_r = math.hypot(apex_d, rise - crown_cy)
    a_flank = math.atan2(join_y, join_x - flank_cx)
    a_crown_join = math.atan2(join_y - crown_cy, join_x + apex_d)
    a_apex = math.atan2(rise - crown_cy, apex_d)
    n_flank = max(2, segments // 2)
    n_crown = max(2, segments - segments // 2)
    pts = []
    for i in range(n_flank + 1):
        a = a_flank * i / n_flank
        pts.append((flank_cx + flank_r * math.cos(a), flank_r * math.sin(a)))
    for i in range(1, n_crown + 1):
        a = a_crown_join + (a_apex - a_crown_join) * i / n_crown
        pts.append((-apex_d + crown_r * math.cos(a), crown_cy + crown_r * math.sin(a)))
    return np.asarray(pts, dtype=float)


def _arc_x_at(arc: np.ndarray, y: float) -> float:
    """The arch curve's half-span at height ``y`` (arc is monotone in y)."""
    ys = arc[:, 1]
    if y <= ys[0]:
        return float(arc[0, 0])
    if y >= ys[-1]:
        return 0.0
    i = int(np.searchsorted(ys, y) - 1)
    y0, y1 = ys[i], ys[i + 1]
    t = 0.0 if y1 <= y0 else (y - y0) / (y1 - y0)
    return float(arc[i, 0] + (arc[i + 1, 0] - arc[i, 0]) * t)


def _arch_ring(half: float, rise: float, thickness: float, segments: int = 16) -> np.ndarray:
    """The closed four-centred arch ring: outer curve, inner curve, no crossings."""
    inner = four_centred_arc(half - thickness, rise - thickness, segments)
    outer = four_centred_arc(half, rise, segments)
    pts = [tuple(p) for p in outer]                          # (half,0) .. (0,rise)
    pts += [(-p[0], p[1]) for p in outer[:-1][::-1]]          # .. (-half,0)
    pts += [(-p[0], p[1]) for p in inner]                    # (-(half-t),0) .. (0,rise-t)
    pts += [tuple(p) for p in inner[:-1][::-1]]              # .. (half-t,0)
    return np.asarray(pts, dtype=float)


def _y_rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s, 0.0], [0.0, 1.0, 0.0, 0.0],
                     [-s, 0.0, c, 0.0], [0.0, 0.0, 0.0, 1.0]])


class PersianBuilder:
    def __init__(self, design):
        self.d = design
        self.detail = design.detail
        self.palette = _palette(design.finish)
        self.substances = _substances(design.finish)
        self.placements = []
        self.counts = Counter()
        self.anchors = {"dome": [], "iwans": [], "minarets": []}

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
        """A load-bearing member as human-scale ashlar in running bond."""
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        z0, z1 = sorted((z0, z1))
        host = f"{PRODUCT}.{role}.{self.counts[role]:04d}"
        last = None
        for stone in fill_box(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                              (x1 - x0, y1 - y0, z1 - z0), self.detail):
            last = self.add(role, stone.position, stone.dimensions,
                            {"construction": "ashlar running bond", "unit": stone.unit,
                             "course": stone.course, "host_id": host, "axis": stone.axis,
                             **payload}, material)
        return last

    def coursed_axis(self, role, cross0, cross1, y0, y1, along0, along1, axis,
                    material="brick", **payload):
        """Course a wall given (cross, height, along) spans for either axis."""
        if axis == "z":
            return self.coursed(role, cross0, cross1, y0, y1, along0, along1,
                                material, **payload)
        return self.coursed(role, along0, along1, y0, y1, cross0, cross1,
                            material, **payload)

    def mesh(self, role, polygon, extrusion, x=0.0, y=0.0, z=0.0, material="brick",
             angle=0.0, along="x", **payload):
        pts = np.asarray(polygon, dtype=float)
        dedup = [pts[0].tolist()]
        for row in pts[1:]:
            if any(abs(row[k] - dedup[-1][k]) > 1e-9 for k in range(2)):
                dedup.append(row.tolist())
        if len(dedup) > 1 and all(abs(dedup[0][k] - dedup[-1][k]) <= 1e-9 for k in range(2)):
            dedup.pop()
        mesh = extrude_profile(np.asarray(dedup, dtype=float), extrusion)
        if along == "z":
            # Profile x spans world z; the extrusion runs along world x.
            mesh.apply_transform(np.array([[0., 0., -1., 0.], [0., 1., 0., 0.],
                                            [1., 0., 0., 0.], [0., 0., 0., 1.]]))
        if angle:
            mesh.apply_transform(_y_rotation(angle))
        mesh.apply_translation((x, y, z))
        position, size, data = mesh_parameters(mesh)
        return self.add(role, position, size, {**data, **payload}, material, block_type="MESH")

    def paving(self, x0, x1, z0, z1, y=0.0, role="court_paving", material="stone"):
        """Flagstones in running bond: alternate courses break joint by half a stone."""
        host = f"{PRODUCT}.{role}.{self.counts[role]:04d}"
        tiles_x = max(1, int(round((x1 - x0) / 1.2)))
        tiles_z = max(1, int(round((z1 - z0) / 1.2)))
        dx, dz = (x1 - x0) / tiles_x, (z1 - z0) / tiles_z
        for iz in range(tiles_z):
            za, zb = z0 + iz * dz, z0 + (iz + 1) * dz
            shift = dx / 2 if iz % 2 and tiles_x > 1 else 0.0
            joints = [x0 + shift + k * dx for k in range(tiles_x + 1)]
            edges = [x0] + [x for x in joints if x0 + 0.01 < x < x1 - 0.01] + [x1]
            for ix, (xa, xb) in enumerate(zip(edges[:-1], edges[1:])):
                self.add(role, ((xa + xb) / 2, y + 0.04, (za + zb) / 2),
                         (xb - xa - 0.02, 0.08, zb - za - 0.02),
                         {"construction": "paving", "unit": "paving", "course": iz,
                          "host_id": host, "axis": "x", "stone_index": ix}, material)

    def arch(self, role, x, z, span, spring_y, rise, depth, along="x",
             thickness=ARCH_T, material="brick", bay=None, **payload):
        """A four-centred arch laid as ring courses between its springings.

        Each ring is one stone course following the arch curve through the
        wall, so the ring's bedding face is the curve itself.
        """
        ring = _arch_ring(span / 2.0, rise, thickness)
        rings = max(3, int(math.ceil(depth / ARCH_RING_MAX)))
        chunk = depth / rings
        bay_id = bay or f"{role}.{self.counts[role]:04d}"
        body = {"construction": "arch course", "unit": "arch_ring", "arch": "four-centred",
                "span": round(span, 6), "rise": round(rise, 6), "spring_y": round(spring_y, 6),
                "ratio": round(rise / span, 4), "thickness": round(thickness, 6),
                "bay_id": bay_id, **payload}
        for i in range(rings):
            offset = -depth / 2.0 + (i + 0.5) * chunk
            piece = dict(body, ring=i, rings=rings)
            if along == "x":
                self.mesh(role, ring, chunk, x, spring_y, z + offset, material,
                          along="x", **piece)
            else:
                self.mesh(role, ring, chunk, x + offset, spring_y, z, material,
                          along="z", **piece)
        return bay_id

    def vault(self, role, span, spring_y, rise, length, axis, cross_c, along_c,
              material="vault", **payload):
        """A pointed barrel vault: four-centred rings laid along the hall."""
        ring = _arch_ring(span / 2.0, rise, 0.45)
        runs = max(1, int(math.ceil(length / VAULT_CHUNK)))
        chunk = length / runs
        body = {"construction": "vault course", "unit": "vault_ring", "vault": "pointed barrel",
                "span": round(span, 6), "rise": round(rise, 6), "spring_y": round(spring_y, 6),
                **payload}
        for i in range(runs):
            offset = -length / 2.0 + (i + 0.5) * chunk
            piece = dict(body, ring=i, rings=runs)
            if axis == "z":
                self.mesh(role, ring, chunk, cross_c, spring_y, along_c + offset,
                          material, along="x", **piece)
            else:
                self.mesh(role, ring, chunk, along_c + offset, spring_y, cross_c,
                          material, along="z", **piece)

    def iwan(self, side, axis, cx, cz, width, height, face, back, frame_top,
             material="brick", hood_tiers=0, back_wall=True):
        """A vaulted hall walled on three sides, open to the court on the fourth.

        ``axis`` is the hall's depth direction; ``face`` is the opening plane
        coordinate along it and ``back`` the far plane. ``cx`` is the hall's
        center along that axis and ``cz`` its center across the span. The
        vault rings carry the hall's own anchors: side, width, depth, height.
        """
        wall_h = 0.62 * height
        vault_rise = 0.38 * height
        depth = abs(back - face)
        cross_c = cz if axis == "x" else cx
        along_c = cx if axis == "x" else cz
        s_court = -1.0 if back > face else 1.0
        a0, a1 = sorted((face, back))
        hall = {"side": side, "width": round(width, 6), "depth": round(depth, 6),
                "height": round(height, 6), "frame_top": round(frame_top, 6)}
        # Flank walls and back wall; the mouth stays open.
        for sign in (-1, 1):
            edge = cross_c + sign * (width / 2 - WALL_T / 2)
            self.coursed_axis("iwan_wall", edge - WALL_T / 2, edge + WALL_T / 2,
                              0.0, wall_h, a0, a1, axis, material)
        if back_wall:
            b0, b1 = sorted((back, back - s_court * WALL_T))
            self.coursed_axis("iwan_wall", cross_c - (width / 2 - WALL_T),
                              cross_c + (width / 2 - WALL_T), 0.0, wall_h, b0, b1,
                              axis, material)
        # The mouth is an opening, so it carries a four-centred arch whose
        # springing plinths are its own jamb stones.
        face_span = 0.78 * width
        face_rise = self.d.arch_ratio * face_span
        for sign in (-1, 1):
            jx = face_span / 2 + 0.3 * sign
            lo, hi = sorted((face - 0.3, face + 0.3))
            if axis == "z":
                self.box("iwan_wall", cross_c + jx - 0.3, cross_c + jx + 0.3, 0.0,
                         PLINTH_H + IMPOST, lo, hi, material, construction="ashlar stack",
                         unit="jamb", side=side)
            else:
                self.box("iwan_wall", lo, hi, 0.0, PLINTH_H + IMPOST, cross_c + jx - 0.3,
                         cross_c + jx + 0.3, material, construction="ashlar stack",
                         unit="jamb", side=side)
        if axis == "z":
            self.arch("entrance_arch", cross_c, face, face_span, PLINTH_H, face_rise,
                      0.5, along="x", material=material, bay=f"iwan_face.{side}",
                      side=side)
        else:
            self.arch("entrance_arch", face, cross_c, face_span, PLINTH_H, face_rise,
                      0.5, along="z", material=material, bay=f"iwan_face.{side}",
                      side=side)
        # Vault rings: the hall's own arched roof. They start one metre clear
        # of the face plane so the muqarnas hood owns the mouth's head zone.
        v0 = face - s_court * 1.0
        self.vault("iwan", width, wall_h, vault_rise, abs(back - v0), axis, cross_c,
                   (v0 + back) / 2, material="vault", **hall)
        # Pishtaq frame standing proud of the face plane, above the arcade.
        band_a0, band_a1 = sorted((face, face + s_court * FRAME_T))
        for sign in (-1, 1):
            edge = cross_c + sign * (width / 2 + FRAME_T / 2)
            self.coursed_axis("pishtaq", edge - FRAME_T / 2, edge + FRAME_T / 2,
                              0.0, frame_top, band_a0, band_a1, axis, material)
        self.coursed_axis("pishtaq", cross_c - (width / 2 + FRAME_T),
                          cross_c + (width / 2 + FRAME_T), frame_top - 0.7, frame_top,
                          band_a0, band_a1, axis, material)
        if hood_tiers:
            self.muqarnas_hood(axis, cross_c, face, -s_court, PLINTH_H + face_rise,
                               face_span, hood_tiers, material)
        self.anchors["iwans"].append(dict(hall, x=round(cx, 6), z=round(cz, 6)))

    def muqarnas_hood(self, axis, cross_c, face, s_back, crown_y, span, tiers,
                      material="brick", crown_role="pishtaq"):
        """A corbelled hood over a portal arch head.

        ``s_back`` points from the face plane into the recess behind it. Tier
        rows of cells step back by at most half a cell depth per tier; every
        row stands on a masonry course spanning the arch head, so each cell's
        bearing is checked from the geometry, not asserted.
        """
        d = MUQ_DEPTH
        step = MUQ_STEP
        r0 = max(0.8, span / 2 - 0.30)
        fw = span + 2 * FRAME_T
        hood = f"{PRODUCT}.muqarnas_cell.{self.counts['muqarnas_cell']:04d}"
        lo, hi = sorted((face - 0.3 * s_back, face + 0.8 * s_back))
        # Head ledge: the masonry course the first tier rests on. It bears on
        # the arch ring's crown and the flanking frame walls.
        self.coursed_axis("pishtaq", cross_c - fw / 2, cross_c + fw / 2,
                          crown_y - d, crown_y, lo, hi, axis, material,
                          construction="ashlar running bond", unit="hood ledge",
                          hood_id=hood)
        for tier in range(tiers):
            r = r0 - tier * step
            y0 = crown_y + tier * 2 * d
            n = max(4, int(round(2 * r / MUQ_CELL)))
            n += n % 2
            w = 2 * r / n
            for j in range(n):
                off = -r + (j + 0.5) * w
                body = {"construction": "muqarnas corbel", "unit": "muqarnas_cell",
                        "tier": tier, "radius": round(r, 6), "row": j,
                        "step": step, "span": round(span, 6), "hood_id": hood,
                        "height": d}
                if axis == "z":
                    self.add("muqarnas_cell", (cross_c + off, y0 + d / 2,
                                               face + 0.25 * s_back),
                             (w, d, 0.55), body, "tile")
                else:
                    self.add("muqarnas_cell", (face + 0.25 * s_back, y0 + d / 2,
                                               cross_c + off),
                             (0.55, d, w), body, "tile")
            if tier + 1 < tiers:
                r_next = r0 - (tier + 1) * step
                self.coursed_axis("pishtaq", cross_c - (r_next + 0.3),
                                  cross_c + (r_next + 0.3),
                                  y0 + d, y0 + 2 * d, lo, hi, axis, material,
                                  construction="ashlar running bond", unit="hood course",
                                  hood_id=hood, tier=tier + 1)
        y_top = crown_y + tiers * 2 * d - d
        r_last = r0 - (tiers - 1) * step
        # The crown course closes the hood; in a domed kiosk it is the wall's
        # own head course, so the transition above bears on wall masonry.
        self.coursed_axis(crown_role, cross_c - (r_last + 0.3), cross_c + (r_last + 0.3),
                          y_top, y_top + 0.7, lo, hi, axis, material,
                          construction="ashlar running bond", unit="hood crown",
                          hood_id=hood)
        return y_top + 0.7

    def squinch(self, cx, cz, sx, sz, base_y, span, rise, material="vault"):
        """A small four-centred arch bridging a chamber corner over the wall."""
        ring = _arch_ring(span / 2.0, rise, 0.35)
        angle = math.pi / 4 if sx == sz else -math.pi / 4
        body = {"construction": "squinch arch", "unit": "squinch_ring", "arch": "four-centred",
                "span": round(span, 6), "rise": round(rise, 6), "spring_y": round(base_y, 6),
                "corner": (sx, sz)}
        self.mesh("squinch", ring, 0.45, cx, base_y, cz, material, angle=angle,
                  along="x", **body)

    def transition(self, cx, cz, half, wall_top, wall_role, material="brick"):
        """Square to octagon to sixteen-sided band to drum, each stage on the last.

        Four corner squinches spring from the wall top; the walls between
        their feet rise to the octagon level; a sixteen-sided band spans
        squinches and shoulders; the drum lands inside the band's inner face;
        the dome springs from the drum.
        """
        d = self.d
        cut = max(0.55, 0.42 * half)
        span = cut * math.sqrt(2)
        rise = d.arch_ratio * span
        oct_y = wall_top + rise
        host = f"{PRODUCT}.transition_band.{self.counts['transition_band']:04d}"
        # Corner squinches: feet on the two walls that meet at each corner.
        for sx, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            mid_x = cx + sx * (half - cut / 2)
            mid_z = cz + sz * (half - cut / 2)
            self.squinch(mid_x, mid_z, sx, sz, wall_top, span, rise)
        # Shoulders: the wall heads between squinch feet, up to octagon level.
        w = WALL_T
        for sx, sz in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if sz:
                self.coursed(wall_role, cx - (half - cut), cx + (half - cut),
                             wall_top, oct_y, cz + sz * half, cz + sz * (half + w),
                             material)
            else:
                self.coursed(wall_role, cx + sx * half, cx + sx * (half + w),
                             wall_top, oct_y, cz - (half - cut), cz + (half - cut),
                             material)
        # Sixteen-sided band inside the octagon's footprint.
        band_top = oct_y + BAND_H
        r_in = half - BAND_IN
        for k in range(16):
            alpha = k * math.pi / 8
            theta = 3 * math.pi / 2 - alpha
            chord = 2 * r_in * math.tan(math.pi / 16)
            pieces = max(1, int(math.ceil(chord / 1.5)))
            mid_r = r_in + BAND_T / 2
            for piece in range(pieces):
                l0 = -chord / 2 + piece * chord / pieces
                l1 = -chord / 2 + (piece + 1) * chord / pieces
                poly = [(l0, 0.0), (l1, 0.0), (l1, BAND_H), (l0, BAND_H)]
                px = cx + mid_r * math.cos(alpha)
                pz = cz + mid_r * math.sin(alpha)
                self.mesh("transition_band", np.asarray(poly), BAND_T, px, oct_y, pz,
                          material, angle=theta, along="x",
                          construction="band course", unit="band_stone", segment=k,
                          piece=piece, band_sides=16, host_id=host,
                          r_inner=round(r_in, 6), r_outer=round(r_in + BAND_T, 6),
                          y_lo=round(oct_y, 6), y_hi=round(band_top, 6))
        # Drum: an ashlar ring inside the band, its diameter the chamber side.
        drum_top = band_top + DRUM_H
        n = max(28, int(math.ceil(2 * math.pi * half / 0.55)))
        for course in range(2):
            y0 = band_top + course * DRUM_H / 2
            y1 = y0 + DRUM_H / 2
            for j in range(n):
                a0 = j * 2 * math.pi / n
                a1 = (j + 1) * 2 * math.pi / n
                mid = (a0 + a1) / 2
                chord = 2 * half * math.sin(math.pi / n) * 1.04
                rad = 0.55
                # Tangential spread is chord*|sin|, radial spread rad*|cos|.
                dx = chord * abs(math.sin(mid)) + rad * abs(math.cos(mid))
                dz = chord * abs(math.cos(mid)) + rad * abs(math.sin(mid))
                self.add("drum", (cx + half * math.cos(mid), (y0 + y1) / 2,
                                  cz + half * math.sin(mid)),
                         (dx, DRUM_H / 2, dz),
                         {"construction": "drum stack", "unit": "drum", "course": course,
                          "ring_index": j, "drum_radius": round(half, 6),
                          "chamber_side": round(2 * half, 6), "r_inner": round(half - rad / 2, 6),
                          "r_outer": round(half + rad / 2, 6)}, material)
        apex = self.dome(cx, cz, drum_top, half, d.dome_apex_ratio, material)
        return {"octagon_y": oct_y, "band_top": band_top, "drum_top": drum_top,
                "apex": apex, "chamber_half": half, "drum_radius": half}

    def dome(self, cx, cz, base_y, radius, apex_ratio, material="brick"):
        """A pointed dome laid in horizontal courses of named stones.

        The meridian is the four-centred arch curve rotated about the axis, so
        the dome springs vertically from the drum and closes at a true point.
        """
        height = apex_ratio * 2.0 * radius
        arc = four_centred_arc(radius, height, 64)
        courses = max(8, int(math.ceil(height / 0.75)))
        host = f"{PRODUCT}.dome_course.{self.counts['dome_course']:04d}"
        apex = base_y + height
        radii = [_arc_x_at(arc, height * i / courses) for i in range(courses + 1)]
        for course in range(courses):
            y_lo = base_y + height * course / courses
            y_hi = base_y + height * (course + 1) / courses
            r_lo, r_hi = radii[course], radii[course + 1]
            if r_hi < 0.16:
                break
            self._dome_course_ring(cx, cz, course, y_lo, y_hi, r_lo, r_hi, host, material)
        # Stepped cap: the last ring closes the point, then the keystone.
        r_cap = radii[course]
        cap_lo = base_y + height * course / courses
        cap_hi = min(cap_lo + 0.35, apex - 0.02)
        if r_cap > 0.14 and cap_hi > cap_lo + 0.05:
            self._dome_course_ring(cx, cz, course, cap_lo, cap_hi, r_cap, 0.12,
                                   host, material)
            course += 1
        key_lo = min(cap_hi, apex - 0.3)
        key_hi = apex
        if key_hi <= key_lo + 0.02:
            key_lo = key_hi - 0.3
        self.add("dome_course", (cx, (key_lo + key_hi) / 2, cz),
                 (0.45, key_hi - key_lo, 0.45),
                 {"construction": "dome course", "unit": "dome_stone", "course": course,
                  "ring_index": 0, "host_id": host, "keystone": True,
                  "y_lo": round(key_lo, 6), "y_hi": round(key_hi, 6),
                  "r_lo": 0.12, "r_hi": 0.0}, material)
        self.add("dome_knob", (cx, apex - 0.15, cz), (0.5, 0.5, 0.5),
                 {"construction": "finial", "unit": "finial", "role_kind": "dome_cap"},
                 "tile")
        self.anchors["dome"].append({"x": round(cx, 6), "z": round(cz, 6),
                                     "base_y": round(base_y, 6),
                                     "diameter": round(2 * radius, 6),
                                     "apex": round(apex, 6),
                                     "apex_ratio": round(apex_ratio, 6),
                                     "courses": courses})
        return apex

    def _dome_course_ring(self, cx, cz, course, y_lo, y_hi, r_lo, r_hi, host, material):
        r_mid = (r_lo + r_hi) / 2
        rad = max(r_lo - r_hi, 0.08)
        n = max(6, int(math.ceil(2 * math.pi * max(r_mid, 0.2) / 1.1)))
        for j in range(n):
            a0 = j * 2 * math.pi / n
            a1 = (j + 1) * 2 * math.pi / n
            mid = (a0 + a1) / 2
            chord = 2 * r_mid * math.sin(math.pi / n) * 1.06
            # Tangential spread is chord*|sin|, radial spread rad*|cos|.
            dx = chord * abs(math.sin(mid)) + rad * abs(math.cos(mid))
            dz = chord * abs(math.cos(mid)) + rad * abs(math.sin(mid))
            self.add("dome_course", (cx + r_mid * math.cos(mid), (y_lo + y_hi) / 2,
                                      cz + r_mid * math.sin(mid)),
                     (max(dx, 0.08), y_hi - y_lo, max(dz, 0.08)),
                     {"construction": "dome course", "unit": "dome_stone", "course": course,
                      "ring_index": j, "host_id": host,
                      "y_lo": round(y_lo, 6), "y_hi": round(y_hi, 6),
                      "r_lo": round(r_lo, 6), "r_hi": round(r_hi, 6)}, material)

    def minaret(self, x, z, tip_y, material="brick"):
        """A cylindrical tapering shaft with a corbelled balcony and a cap."""
        base_r = 0.62
        plinth = self.add("minaret", (x, PLINTH_H / 2, z), (2 * base_r, PLINTH_H, 2 * base_r),
                          {"construction": "plinth course", "unit": "ashlar",
                           "tip_y": round(tip_y, 6)}, material)
        host = plinth
        balcony_frac = 0.62
        bal_y = PLINTH_H + balcony_frac * (tip_y - PLINTH_H)
        lower = bal_y - PLINTH_H
        for center_y, course_h, index in iter_stack(PLINTH_H, lower, self.detail):
            frac = (center_y - PLINTH_H) / lower
            r = base_r * (1.0 - 0.18 * frac)
            self.add("minaret_drum", (x, center_y, z), (2 * r, course_h, 2 * r),
                     {"construction": "drum stack", "unit": "drum", "course": index,
                      "host": host, "taper": round(frac, 4)}, material, "CYLINDER")
        r_b = base_r * (1.0 - 0.18) + 0.02
        y = bal_y
        for step, (out, thick) in enumerate(((0.05, 0.16), (0.06, 0.15), (0.06, 0.14))):
            r = r_b + out
            self.add("minaret_balcony", (x, y + thick / 2, z), (2 * r, thick, 2 * r),
                     {"construction": "corbel stack", "unit": "corbel", "course": step,
                      "host": host, "overhang": round(out, 4)}, material, "CYLINDER")
            y += thick
        upper = tip_y - 0.44 - y
        r_top_start = r_b + 0.17
        for center_y, course_h, index in iter_stack(y, upper, self.detail):
            frac = (center_y - y) / upper
            r = r_top_start * (1.0 - 0.30 * frac)
            self.add("minaret_drum", (x, center_y, z), (2 * r, course_h, 2 * r),
                     {"construction": "drum stack", "unit": "drum",
                      "course": 100 + index, "host": host, "taper": round(frac, 4)},
                     material, "CYLINDER")
        self.add("minaret_cap", (x, tip_y - 0.22, z), (0.52, 0.44, 0.52),
                 {"construction": "finial", "unit": "finial", "host": host,
                  "role_kind": "minaret_cap"}, "tile")
        self.anchors["minarets"].append({"x": round(x, 6), "z": round(z, 6),
                                         "tip": round(tip_y, 6)})

    def corner_tower(self, x, z, height, material="stone"):
        host = f"{PRODUCT}.corner_tower.{self.counts['corner_tower']:04d}"
        for center_y, course_h, index in iter_stack(0.0, height, self.detail):
            self.add("corner_tower", (x, center_y, z), (1.5, course_h, 1.5),
                     {"construction": "drum stack", "unit": "drum", "course": index,
                      "host_id": host}, material)

    def arcade_run(self, axis, line, a0, a1, base_y, spring, material="brick"):
        """One arcade segment: coursed piers to the springing, bay arches over.

        Every bay keeps the design's rise-to-span ratio, whatever its pitch.
        Returns the pier centers and the bay pitch, or ``([], 0.0)`` when the
        segment was too short to place anything.
        """
        if a1 - a0 < PIER_W + 1.1:
            return [], 0.0
        c0, c1 = a0 + PIER_W / 2, a1 - PIER_W / 2
        bays = max(1, int(round((c1 - c0) / 3.4)))
        pitch = (c1 - c0) / bays
        rise = self.d.arch_ratio * pitch
        centers = []
        for i in range(bays + 1):
            center = c0 + i * pitch
            centers.append(center)
            if axis == "z":
                self.coursed("riwaq_pier", center - PIER_W / 2, center + PIER_W / 2,
                             base_y, spring + IMPOST, line - PIER_W / 2, line + PIER_W / 2, material)
            else:
                self.coursed("riwaq_pier", line - PIER_W / 2, line + PIER_W / 2,
                             base_y, spring + IMPOST, center - PIER_W / 2, center + PIER_W / 2, material)
        for i in range(bays):
            center = c0 + (i + 0.5) * pitch
            if axis == "z":
                self.arch("riwaq_bay", center, line, pitch, spring, rise, PIER_W,
                          along="x", material=material, bay=f"riwaq.{axis}.{i:02d}")
            else:
                self.arch("riwaq_bay", line, center, pitch, spring, rise, PIER_W,
                          along="z", material=material, bay=f"riwaq.{axis}.{i:02d}")
        return centers, pitch

    def pier_shaft(self, axis, line, centers, y0, y1, material="brick"):
        """The pier masonry that rises beside the bay arches to carry the
        slab above them.

        Each shaft stands on the pier below and steps a finger back from the
        ring's faces, so the arch courses read as bedding into the pier while
        their springing stones stay proud of its faces.
        """
        if y1 - y0 <= 0.05:
            return
        lo = line - PIER_W / 2 + SHAFT_STEP
        hi = line + PIER_W / 2 - SHAFT_STEP
        for c in centers:
            if axis == "z":
                self.coursed("riwaq_pier", c - PIER_W / 2, c + PIER_W / 2,
                             y0, y1, lo, hi, material)
            else:
                self.coursed("riwaq_pier", lo, hi, y0, y1,
                             c - PIER_W / 2, c + PIER_W / 2, material)

    def level_courses(self, role, x0, x1, z0, z1, gallery0, roof_top,
                      material="brick", **payload):
        """Course a wall whose joints meet the slabs bonding into it.

        Course tops land at the gallery and roof soffits, and the band
        between them fills the joint the slabs bite into, so the floors and
        the roof are laid into the wall the way ashlar bonds at a junction.
        """
        for y0, y1 in ((0.0, gallery0 + IMPOST),
                       (gallery0 + IMPOST, gallery0 + SLAB_T),
                       (gallery0 + SLAB_T, roof_top)):
            if y1 - y0 > 0.04:
                self.coursed(role, x0, x1, y0, y1, z0, z1, material, **payload)

    # -- compositions ------------------------------------------------------

    def compose(self):
        if self.d.variant == "four-iwan":
            self.compose_four_iwan()
        elif self.d.variant == "chahar-taq":
            self.compose_chahar_taq()
        else:
            self.compose_caravanserai()

    def compose_four_iwan(self):
        d = self.d
        qibla_w = min(9.2, max(4.4, d.width * 0.25))
        qibla_d = 1.15 * qibla_w
        side_w = max(2.6, 0.62 * qibla_w)
        side_d = 1.15 * side_w
        court_w = max(7.5, d.width - 2 * side_d)
        sanct_half = max(3.0, 0.42 * qibla_w)
        north_d = side_d
        z_outer = -d.depth / 2
        z_wall = z_outer + WALL_T                 # north enclosure inner face
        z_cn = z_outer + north_d
        court_d = max(6.0, d.depth - north_d - qibla_d - 2 * WALL_T - 2 * sanct_half)
        z_cs = z_cn + court_d
        z_swall = z_cs + qibla_d                  # south enclosure inner face
        z_front = z_swall + WALL_T
        cx = 0.0
        x_e = court_w / 2 + side_d                # east enclosure outer face
        x_wall = x_e - WALL_T
        mid_z = (z_cn + z_cs) / 2

        qibla_h = 0.64 * d.height
        side_h = 0.40 * d.height

        # The riwaq: a two-storey arcade ring on the court edge, broken at
        # each iwan mouth by that iwan's pishtaq, and carried back to the
        # outer wall. Storeys scale with the requested height, and the levels
        # follow the tallest bay so no crown rises past the slab above it.
        gap_side = side_w / 2 + FRAME_T
        gap_qibla = qibla_w / 2 + FRAME_T
        spring1 = max(1.9, 0.185 * d.height)
        storey = spring1
        north_segs = [(gap_side, court_w / 2 - PIER_W),
                      (-(court_w / 2 - PIER_W), -gap_side)]
        south_segs = [(gap_qibla, court_w / 2 - PIER_W),
                     (-(court_w / 2 - PIER_W), -gap_qibla)]
        side_segs = [(z_cn + PIER_W, mid_z - gap_side),
                     (mid_z + gap_side, z_cs - PIER_W)]
        sides = [
            dict(name="north", axis="z", line=z_cn, segs=north_segs,
                 gap=gap_side, mouth=side_w, band=(z_wall + 0.05, z_cn + 0.35),
                 edge=(z_wall, z_cn)),
            dict(name="south", axis="z", line=z_cs, segs=south_segs,
                 gap=gap_qibla, mouth=qibla_w, band=(z_cs - 0.35, z_swall + 0.05),
                 edge=(z_cs, z_swall)),
            dict(name="west", axis="x", line=-court_w / 2, segs=side_segs,
                 gap=gap_side, mouth=side_w, band=(-x_wall - 0.05, -court_w / 2 + 0.35),
                 edge=(-x_wall, -court_w / 2)),
            dict(name="east", axis="x", line=court_w / 2, segs=side_segs,
                 gap=gap_side, mouth=side_w, band=(court_w / 2 - 0.35, x_wall + 0.05),
                 edge=(court_w / 2, x_wall)),
        ]
        ground = []
        for side in sides:
            for lo, hi in side["segs"]:
                centers, pitch = self.arcade_run(side["axis"], side["line"],
                                                 lo, hi, 0.0, spring1)
                ground.append((side, lo, hi, centers, pitch))
        pitched = [run for run in ground if run[3]]
        max_pitch = max((run[4] for run in pitched), default=3.4)
        bay_rise = d.arch_ratio * max_pitch
        gallery0 = spring1 + bay_rise
        spring2 = gallery0 + SLAB_T + storey
        roof0 = spring2 + bay_rise
        roof_top = roof0 + SLAB_T

        # Pier shafts rise beside both storeys of arches; corner piers carry
        # the slabs around each court corner. On a side too short for a bay
        # the band closes with wall mass and its corners need no piers.
        bayed = {run[0]["name"]: True for run in pitched}
        z_corner_n = z_cn + (0.35 if bayed.get("north") else 0.0)
        z_corner_s = z_cs - (0.35 if bayed.get("south") else 0.0)
        for side in sides:
            runs = [run for run in pitched if run[0] is side]
            if not runs:
                continue
            centers = [c for run in runs for c in run[3]]
            self.pier_shaft(side["axis"], side["line"], centers,
                            spring1 + IMPOST, gallery0 + IMPOST)
        for z_edge, live in ((z_cn, bayed.get("north")), (z_cs, bayed.get("south"))):
            if not live:
                continue
            for sx in (-1, 1):
                self.coursed("riwaq_pier",
                             sx * (court_w / 2 - PIER_W / 2) - PIER_W / 2,
                             sx * (court_w / 2 - PIER_W / 2) + PIER_W / 2,
                             0.0, gallery0 + IMPOST,
                             z_edge - PIER_W / 2, z_edge + PIER_W / 2)
                self.coursed("riwaq_pier",
                             sx * (court_w / 2 - PIER_W / 2) - PIER_W / 2,
                             sx * (court_w / 2 - PIER_W / 2) + PIER_W / 2,
                             gallery0 + SLAB_T, roof0 + IMPOST,
                             z_edge - PIER_W / 2, z_edge + PIER_W / 2)
        # The qibla band runs deeper than one aisle when its iwans are deep,
        # so a middle row of back piers divides it into two covered aisles.
        two_aisle = qibla_d > 7.0
        back_centers = []
        if two_aisle:
            z_b = z_cs + qibla_d / 2
            for run in pitched:
                if run[0]["name"] == "south":
                    back_centers.extend(run[3])
            for c in back_centers:
                self.coursed("riwaq_pier", c - PIER_W / 2, c + PIER_W / 2,
                             0.0, gallery0 + IMPOST, z_b - PIER_W / 2, z_b + PIER_W / 2)
                self.coursed("riwaq_pier", c - PIER_W / 2, c + PIER_W / 2,
                             gallery0 + SLAB_T, roof0 + IMPOST, z_b - PIER_W / 2, z_b + PIER_W / 2)

        # The upper storey: the same piers and bays standing on the gallery.
        for run in ground:
            if run[3]:
                self.arcade_run(run[0]["axis"], run[0]["line"], run[1], run[2],
                                gallery0 + SLAB_T, spring2)
        for side in sides:
            runs = [run for run in pitched if run[0] is side]
            if not runs:
                continue
            centers = [c for run in runs for c in run[3]]
            self.pier_shaft(side["axis"], side["line"], centers,
                            spring2 + IMPOST, roof0 + IMPOST)

        # The gallery floor and the roof: each slab spans its side of the
        # band, court face to outer wall, and bonds into the piers, the back
        # piers and the wall at its bearing ends.
        span_sets = {
            "north": [(gap_side, x_wall + 0.05), (-(x_wall + 0.05), -gap_side)],
            "south": [(max(gap_qibla, sanct_half + WALL_T), x_wall + 0.05),
                      (-(x_wall + 0.05), -max(gap_qibla, sanct_half + WALL_T))],
            "west": [(z_corner_n, mid_z - gap_side), (mid_z + gap_side, z_corner_s)],
            "east": [(z_corner_n, mid_z - gap_side), (mid_z + gap_side, z_corner_s)],
        }
        south_band = [(z_cs - 0.35, z_b - 0.30), (z_b + 0.30, z_swall + 0.05)] \
            if two_aisle else [(z_cs - 0.35, z_swall + 0.05)]
        band_sets = {"north": [sides[0]["band"]], "south": south_band,
                     "west": [sides[2]["band"]], "east": [sides[3]["band"]]}
        for name in ("north", "south", "west", "east"):
            if name not in bayed:
                continue
            side = next(s for s in sides if s["name"] == name)
            for along0, along1 in span_sets[name]:
                for b0, b1 in band_sets[name]:
                    if side["axis"] == "z":
                        self.coursed("riwaq_gallery", along0, along1,
                                    gallery0, gallery0 + SLAB_T, b0, b1)
                        self.coursed("riwaq_roof", along0, along1,
                                    roof0, roof_top, b0, b1)
                    else:
                        self.coursed("riwaq_gallery", b0, b1,
                                    gallery0, gallery0 + SLAB_T, along0, along1)
                        self.coursed("riwaq_roof", b0, b1,
                                    roof0, roof_top, along0, along1)

        frame_side = roof_top + 0.9
        frame_qibla = max(qibla_h + 1.1, roof_top + 1.6)

        # Iwans recessed into the band: each hall opens onto the court
        # through its pishtaq and the outer enclosure closes its back.
        self.iwan("south", "z", cx, z_cs + qibla_d / 2, qibla_w, qibla_h,
                  z_cs, z_swall, frame_qibla,
                  hood_tiers=d.muqarnas_tiers, back_wall=False)
        for side, x_face, sign in (("east", court_w / 2, 1), ("west", -court_w / 2, -1)):
            self.iwan(side, "x", x_face + sign * (side_d - WALL_T) / 2, mid_z,
                      side_w, side_h, x_face, x_face + sign * (side_d - WALL_T),
                      frame_side, back_wall=False)
        self.iwan("north", "z", cx, z_cn - (north_d - WALL_T) / 2, side_w, side_h,
                  z_cn, z_wall, frame_side, back_wall=False)

        # The domed sanctuary behind the qibla iwan. The wall top keeps room
        # for the passage arch even when a short request squeezes the dome.
        dome_rise = d.dome_apex_ratio * 2 * sanct_half
        cut = max(0.55, 0.42 * sanct_half)
        squinch_rise = d.arch_ratio * cut * math.sqrt(2)
        sanct_wall_top = max(3.4, d.height * 1.02 - dome_rise - DRUM_H - BAND_H
                             - squinch_rise)
        self.sanctuary(cx, z_front, sanct_half, sanct_wall_top)
        dome_apex = sanct_wall_top + squinch_rise + BAND_H + DRUM_H + dome_rise

        # Outer enclosure: one load-bearing wall at the outer face of the
        # iwan band, rising to the riwaq roof, continuous but for the
        # entrance portal. The qibla side meets the sanctuary block, whose
        # front wall closes the line between its own jamb walls.
        self.level_courses("enclosure_wall", sanct_half + WALL_T, x_wall,
                           z_swall, z_front, gallery0, roof_top)
        self.level_courses("enclosure_wall", -x_wall, -(sanct_half + WALL_T),
                           z_swall, z_front, gallery0, roof_top)
        self.level_courses("enclosure_wall", x_wall, x_e, z_wall, z_front,
                           gallery0, roof_top)
        self.level_courses("enclosure_wall", -x_e, -x_wall, z_wall, z_front,
                           gallery0, roof_top)

        # Mouth walls: bond mass in the band beside each recessed iwan, from
        # the hall's flank wall to the first arcade bay, so the gallery and
        # its roof meet masonry at the recess and the court elevation reads
        # as one wall. A side too short for any bay closes its whole band
        # with the same mass, its wall face standing on the court edge.
        portal_span = min(max(3.0, side_w - 1.2), 0.78 * side_w - 0.1)
        fw_half = portal_span / 2 + FRAME_T
        for side in sides:
            name, axis = side["name"], side["axis"]
            mouth, gap = side["mouth"], side["gap"]
            e0, e1 = side["edge"]
            inner = fw_half if name == "north" else mouth / 2
            if name in bayed:
                spans = ((inner, gap), (-gap, -inner))
                if axis == "x":
                    spans = ((mid_z + inner, mid_z + gap),
                             (mid_z - gap, mid_z - inner))
            elif axis == "z":
                spans = ((inner, x_wall + 0.05), (-(x_wall + 0.05), -inner))
            else:
                spans = ((z_corner_n, mid_z - inner), (mid_z + inner, z_corner_s))
            for a0, a1 in spans:
                if a1 - a0 < 0.12:
                    continue
                if axis == "z":
                    self.level_courses("iwan_wall", a0, a1, e0, e1,
                                       gallery0, roof_top)
                else:
                    self.level_courses("iwan_wall", e0, e1, a0, a1,
                                       gallery0, roof_top)

        # Entrance pishtaq portal on the north face, flanked by minarets.
        portal_rise = d.arch_ratio * portal_span
        self.level_courses("enclosure_wall", fw_half, x_e, z_outer, z_wall,
                          gallery0, roof_top)
        self.level_courses("enclosure_wall", -x_e, -fw_half, z_outer, z_wall,
                          gallery0, roof_top)
        portal_top = self.portal("z", cx, z_outer, portal_span, 3.6, portal_rise,
                                 d.muqarnas_tiers)
        # The frame's flanks bond the portal slab into the wall opening: they
        # carry the muqarnas ledge, and courses above the head band close the
        # wall to its roof line around the proud pishtaq.
        portal_crown = 3.6 + portal_rise
        for sx in (-1, 1):
            self.coursed("pishtaq", sx * portal_span / 2, sx * fw_half, 0.0,
                         portal_crown - MUQ_DEPTH, z_outer, z_wall)
            if roof_top - portal_top > 0.05:
                self.coursed("pishtaq", sx * portal_span / 2, sx * fw_half,
                             portal_top, roof_top, z_outer, z_wall)
        for sx in (-1, 1):
            self.minaret(sx * (portal_span / 2 + FRAME_T + 0.67), z_outer - 0.45,
                         dome_apex + 1.2)

        # Paved sahn and hall floors.
        self.paving(-court_w / 2 + 0.4, court_w / 2 - 0.4, z_cn + 0.4, z_cs - 0.4)
        self.paving(cx - qibla_w / 2, cx + qibla_w / 2, z_cs, z_swall)
        self.paving(cx - sanct_half, cx + sanct_half, z_front, z_front + 2 * sanct_half)

    def portal(self, axis, cross_c, face, span, spring, rise, tiers, material="brick"):
        """A grand entrance frame: jambs, arch, muqarnas hood, head band.

        The frame recesses one frame depth behind the face plane, so the hood
        hangs inside the opening it crowns.
        """
        fw = span + 2 * FRAME_T
        # Jamb piers are embedded under the ring's ends, so the springing
        # bears on their finished top course rather than kissing its edge.
        for sign in (-1, 1):
            j0 = cross_c + sign * (span / 2 - 0.6)
            j1 = cross_c + sign * span / 2
            lo, hi = sorted((face, face + FRAME_T))
            if axis == "z":
                self.coursed("pishtaq", min(j0, j1), max(j0, j1),
                             0.0, spring + IMPOST, lo, hi, material)
            else:
                self.coursed("pishtaq", lo, hi, 0.0, spring + IMPOST,
                             min(j0, j1), max(j0, j1), material)
        if axis == "z":
            self.arch("entrance_arch", cross_c, face + FRAME_T / 2, span, spring, rise,
                      0.5, along="x", material=material, bay="portal")
        else:
            self.arch("entrance_arch", face + FRAME_T / 2, cross_c, span, spring, rise,
                      0.5, along="z", material=material, bay="portal")
        top = self.muqarnas_hood(axis, cross_c, face, 1.0, spring + rise, span, tiers, material)
        lo, hi = sorted((face, face + FRAME_T))
        if axis == "z":
            self.coursed("pishtaq", cross_c - fw / 2, cross_c + fw / 2, top - 0.7, top,
                         lo, hi, material)
        else:
            self.coursed("pishtaq", lo, hi, top - 0.7, top,
                         cross_c - fw / 2, cross_c + fw / 2, material)
        return top

    def sanctuary(self, cx, z_front, half, wall_top, material="brick"):
        """A square domed chamber; its front wall opens into the qibla iwan."""
        d = self.d
        w = WALL_T
        arch_half = min(0.5 * half, 0.3 * d.width)
        span = 2 * arch_half
        rise = d.arch_ratio * span
        # The connecting arch closes into a solid head course, so the wall
        # top carries the transition with masonry across its whole width.
        head_h = 0.7
        spring = wall_top - rise - head_h
        crown = spring + rise
        z_back = z_front + 2 * half
        # Side and back walls; the top course ends exactly at the wall top.
        for sx in (-1, 1):
            self.coursed("sanctuary_wall", cx + sx * half, cx + sx * (half + w),
                         0.0, wall_top, z_front, z_back)
        self.coursed("sanctuary_wall", cx - half, cx + half, 0.0, wall_top,
                     z_back, z_back + w)
        # Front wall: full-height jamb walls beside the connecting arch,
        # split at the springing so the arch bears on a finished course.
        for sx in (-1, 1):
            x_in = cx + sx * (arch_half - 0.3)
            x_out = cx + sx * (half + w)
            lo, hi = sorted((x_in, x_out))
            self.coursed("sanctuary_wall", lo, hi, 0.0, spring + IMPOST, z_front - w, z_front)
            self.coursed("sanctuary_wall", lo, hi, spring, wall_top, z_front - w, z_front)
        self.arch("entrance_arch", cx, z_front - w / 2, span, spring, rise, w,
                  along="x", material=material, bay="qibla_passage")
        # The head course over the arch crown: wall masonry across the top.
        self.coursed("sanctuary_wall", cx - (half + w), cx + (half + w),
                     crown, wall_top, z_front - w, z_front)
        info = self.transition(cx, (z_front + z_back) / 2, half, wall_top,
                               "sanctuary_wall", material)
        # The squinch cuts the chamber corners, so the front face would open
        # beside the passage between the shoulder and the side walls. That
        # face is the qibla enclosure: close it with coursed wall up to the
        # octagon line. The passage below stays the arch.
        cut = max(0.55, 0.42 * half)
        if info["octagon_y"] - wall_top > 0.04:
            for sx in (-1, 1):
                x0 = cx + sx * (half - cut)
                x1 = cx + sx * (half + w)
                lo, hi = sorted((x0, x1))
                self.coursed("sanctuary_wall", lo, hi, wall_top, info["octagon_y"],
                             z_front - w, z_front)

    def compose_chahar_taq(self):
        d = self.d
        # Solve the interior half-side so the apex lands near the request
        # height: wall top, squinch, band, drum and dome all scale with it.
        coef = 2.094 * d.arch_ratio + 2 * d.dome_apex_ratio
        target = d.height * 1.02 - 3.5 - 0.88 * d.muqarnas_tiers
        half = min(max(target / coef, 2.4), 9.0,
                   min(d.width, d.depth) / 2 - WALL_T)
        span = 1.5 * half
        rise = d.arch_ratio * span
        spring = 1.2
        crown = spring + rise
        wall_top = crown + d.muqarnas_tiers * 2 * MUQ_DEPTH - MUQ_DEPTH + 0.7
        w = WALL_T
        self.paving(-half - w - 2.2, half + w + 2.2, -half - w - 2.2, half + w + 2.2)
        self.paving(-half, half, -half, half)
        # Four massive walls, each pierced by one tall four-centred arch whose
        # hood fills the head zone. North and south walls own the corners.
        for fixed in (half, -half):
            inward = -1.0 if fixed > 0 else 1.0
            lo, hi = sorted((fixed, fixed + inward * w))
            for s0, s1 in ((-(half + w), -(span / 2 - 0.3)), ((span / 2 - 0.3), half + w)):
                self.coursed("chamber_wall", s0, s1, 0.0, spring + IMPOST, lo, hi)
                self.coursed("chamber_wall", s0, s1, spring, wall_top, lo, hi)
            self.arch("chamber_arch", 0.0, fixed + inward * w / 2, span, spring, rise,
                      w, along="x",
                      bay=f"chahar-taq.{'north' if fixed > 0 else 'south'}")
            self.muqarnas_hood("z", 0.0, fixed + inward * w / 2, inward,
                               crown, span, d.muqarnas_tiers, crown_role="chamber_wall")
        for fixed in (half, -half):
            inward = -1.0 if fixed > 0 else 1.0
            lo, hi = sorted((fixed, fixed + inward * w))
            for s0, s1 in ((-half, -(span / 2 - 0.3)), ((span / 2 - 0.3), half)):
                self.coursed("chamber_wall", lo, hi, 0.0, spring + IMPOST, s0, s1)
                self.coursed("chamber_wall", lo, hi, spring, wall_top, s0, s1)
            self.arch("chamber_arch", fixed + inward * w / 2, 0.0, span, spring, rise,
                      w, along="z",
                      bay=f"chahar-taq.{'east' if fixed > 0 else 'west'}")
            self.muqarnas_hood("x", 0.0, fixed + inward * w / 2, inward,
                               crown, span, d.muqarnas_tiers, crown_role="chamber_wall")
        self.transition(0.0, 0.0, half, wall_top, "chamber_wall")

    def compose_caravanserai(self):
        d = self.d
        w = 0.9
        # The cell band and the iwans scale with the request so a small court
        # still fits its arcade between the iwan mouths.
        band_d = min(4.4, max(2.8, 0.16 * min(d.width, d.depth)))
        z_out_n, z_out_s = -d.depth / 2, d.depth / 2
        x_out_w, x_out_e = -d.width / 2, d.width / 2
        court_x0, court_x1 = x_out_w + w + band_d, x_out_e - w - band_d
        court_z0, court_z1 = z_out_n + w + band_d, z_out_s - w - band_d
        mid_z = (court_z0 + court_z1) / 2
        # The iwans span the cell band, so their width follows the band depth
        # and the 0.8..1.6 depth-to-width band holds at every request size.
        iwan_w = min(4.6, max(3.0, band_d / 1.05))
        iwan_gap = iwan_w / 2 + FRAME_T
        # A single pishtaq entrance pierces the north wall; its head band
        # sets the wall height so the portal stands proud of the ramparts.
        portal_span = max(3.2, min(4.4, 0.24 * d.width))
        portal_rise = d.arch_ratio * portal_span
        portal_top = (1.0 + portal_rise + d.muqarnas_tiers * 2 * MUQ_DEPTH
                      - MUQ_DEPTH + 0.7)
        # The court arcade: piers and four-centred bays on every side, broken
        # at each iwan mouth. The portico roof follows the tallest bay.
        spring = 2.7
        segs_x = [(court_x0 + PIER_W, -iwan_gap), (iwan_gap, court_x1 - PIER_W)]
        segs_z = [(court_z0 + PIER_W, mid_z - iwan_gap),
                  (mid_z + iwan_gap, court_z1 - PIER_W)]
        sides = [
            dict(name="north", axis="z", line=court_z0, segs=segs_x),
            dict(name="south", axis="z", line=court_z1, segs=segs_x),
            dict(name="west", axis="x", line=court_x0, segs=segs_z),
            dict(name="east", axis="x", line=court_x1, segs=segs_z),
        ]
        runs = {side["name"]: [] for side in sides}
        for side in sides:
            for lo, hi in side["segs"]:
                centers, pitch = self.arcade_run(side["axis"], side["line"],
                                                  lo, hi, 0.0, spring, "stone")
                if centers:
                    runs[side["name"]].append((centers, pitch))
        max_pitch = max((p for rs in runs.values() for _, p in rs), default=3.2)
        roof0 = spring + d.arch_ratio * max_pitch
        outer_h = max(4.0, portal_top, roof0 + 0.25 + 0.5)
        # The portico: the roofed walkway between the arcade and the hujra
        # fronts, carried by the arcade piers and the cell front walls.
        pd = max(1.3, min(2.0, 0.42 * band_d))
        f_n, f_s = court_z0 - pd - FRONT_T, court_z1 + pd + FRONT_T
        f_w, f_e = court_x0 - pd - FRONT_T, court_x1 + pd + FRONT_T
        self.paving(court_x0 + 0.3, court_x1 - 0.3, court_z0 + 0.3, court_z1 - 0.3)
        # Fortified outer wall in two segments per side, leaving the portal
        # mouth open; east and west walls run between the corner towers.
        mouth_half = portal_span / 2 + 0.3
        self.coursed("cell_wall", x_out_w, -mouth_half, 0.0, outer_h, z_out_n, z_out_n + w)
        self.coursed("cell_wall", mouth_half, x_out_e, 0.0, outer_h, z_out_n, z_out_n + w)
        self.coursed("cell_wall", x_out_w, x_out_e, 0.0, outer_h, z_out_s - w, z_out_s)
        for x_edge in (x_out_w, x_out_e):
            sign = 1 if x_edge == x_out_w else -1
            self.coursed("cell_wall", x_edge, x_edge + sign * w, 0.0, outer_h,
                         z_out_n + w, z_out_s - w)
        for sx in (-1, 1):
            for sz in (-1, 1):
                self.corner_tower(x_out_w + sx * (w / 2 + 0.35),
                                  z_out_n + sz * (w / 2 + 0.35), outer_h + 1.5)
        # Hujra cells stand against the rampart: partitions bound each cell
        # and its roof runs from the rampart to the front wall facing the
        # portico.
        for z_lo, z_hi in ((z_out_n + w, f_n), (f_s, z_out_s - w)):
            x = x_out_w + w + 1.5
            while x < x_out_e - w - 1.5:
                if abs(x) >= iwan_gap:
                    self.coursed("cell_wall", x - 0.15, x + 0.15, 0.0, 3.0, z_lo, z_hi)
                x += 3.0
        for x_lo, x_hi in ((x_out_w + w, f_w), (f_e, x_out_e - w)):
            z = z_out_n + w + 1.5
            while z < z_out_s - w - 1.5:
                if abs(z - mid_z) >= iwan_gap:
                    self.coursed("cell_wall", x_lo, x_hi, 0.0, 3.0, z - 0.15, z + 0.15)
                z += 3.0
        for z_lo, z_hi in ((z_out_n + w, f_n + 0.05), (f_s + 0.05, z_out_s - w)):
            for s0, s1 in ((x_out_w + w, -iwan_gap), (iwan_gap, x_out_e - w)):
                self.coursed("cell_roof", s0, s1, 3.0, 3.25, z_lo, z_hi)
        for x_lo, x_hi in ((x_out_w + w, f_w + 0.05), (f_e + 0.05, x_out_e - w)):
            for s0, s1 in ((z_out_n + w, mid_z - iwan_gap), (mid_z + iwan_gap, z_out_s - w)):
                self.coursed("cell_roof", x_lo, x_hi, 3.0, 3.25, s0, s1)
        # Front walls: each cell closes on the portico through a four-centred
        # door, and the wall tops carry the portico roofs back to the bays.
        for z_wall in (f_n, f_s):
            for s0, s1 in ((x_out_w + w, -iwan_w / 2), (iwan_w / 2, x_out_e - w)):
                self.hujra_front("x", s0, s1, z_wall, z_wall + FRONT_T,
                                 roof0, (x_out_w + w, x_out_e - w))
        for x_wall in (f_w, f_e):
            for s0, s1 in ((z_out_n + w, mid_z - iwan_w / 2),
                           (mid_z + iwan_w / 2, z_out_s - w)):
                self.hujra_front("z", s0, s1, x_wall, x_wall + FRONT_T,
                                 roof0, (z_out_n + w, z_out_s - w))
        # Pier shafts behind the bays and the portico roofs: every bay becomes
        # a covered portico in front of its cell, roofed from the arcade to
        # the cell fronts on sides that carry an arcade. The north and south
        # roofs wrap the corner squares; the east and west roofs stop against
        # them, and every roof stops at an iwan's pishtaq frame.
        for side in sides:
            if not runs[side["name"]]:
                continue
            centers = [c for cs, _ in runs[side["name"]] for c in cs]
            self.pier_shaft(side["axis"], side["line"], centers,
                            spring + IMPOST, roof0 + IMPOST, "stone")
            if side["axis"] == "z":
                line = side["line"]
                if side["name"] == "north":
                    # f_n is the outer face; the front thickens toward the court.
                    depth = (f_n + 0.05, court_z0 + 0.35)
                    spans = ((x_out_w + w - 0.05, -iwan_gap),
                             (iwan_gap, x_out_e - w + 0.05))
                else:
                    # f_s is the court face; the front thickens away from the court.
                    depth = (court_z1 - 0.35, f_s + FRONT_T - 0.05)
                    spans = ((x_out_w + w - 0.05, -iwan_gap),
                             (iwan_gap, x_out_e - w + 0.05))
                for a0, a1 in spans:
                    self.coursed("riwaq_roof", a0, a1, roof0, roof0 + 0.25,
                                 depth[0], depth[1], "stone")
            else:
                line = side["line"]
                if side["name"] == "west":
                    depth = (f_w + 0.05, court_x0 + 0.35)
                else:
                    depth = (court_x1 - 0.35, f_e + FRONT_T - 0.05)
                for a0, a1 in ((court_z0 + 0.35, mid_z - iwan_gap),
                               (mid_z + iwan_gap, court_z1 - 0.35)):
                    self.coursed("riwaq_roof", depth[0], depth[1], roof0, roof0 + 0.25,
                                 a0, a1, "stone")
        # Four iwans on the court axes, opening inward; the outer wall closes
        # each hall's back, so no separate back wall is doubled into it.
        iwan_h = max(4.6, 0.36 * d.height)
        frame_top = roof0 + 0.25 + 0.9
        self.iwan("south", "z", 0.0, (court_z1 + z_out_s - w) / 2, iwan_w, iwan_h,
                  court_z1, z_out_s - w, frame_top, "stone", back_wall=False)
        self.iwan("north", "z", 0.0, (court_z0 + z_out_n + w) / 2, iwan_w, iwan_h,
                  court_z0, z_out_n + w, frame_top, "stone", back_wall=False)
        for side, x_face, sign in (("east", court_x1, 1), ("west", court_x0, -1)):
            self.iwan(side, "x", x_face + sign * band_d / 2, mid_z, iwan_w, iwan_h,
                      x_face, x_face + sign * band_d, frame_top, "stone", back_wall=False)
        # The single pishtaq entrance on the north face.
        self.portal("z", 0.0, z_out_n, portal_span, 1.0, portal_rise,
                    d.muqarnas_tiers, "stone")

    def hujra_front(self, axis, a0, a1, c0, c1, roof0, part_range, material="stone"):
        """One run of hujra front walls between partition faces.

        Each stretch carries a four-centred doorway where it is wide enough
        for one: coursed jambs embedded under the ring, the ring itself, and
        a head course rising to the portico-roof bearing. Narrow stretches
        and the ends beside the iwans close solid.
        """
        door_spring = 1.85
        for s0, s1 in self._wall_stretches(a0, a1, part_range):
            span_w = s1 - s0
            center = (s0 + s1) / 2
            if span_w < 2.0:
                if axis == "x":
                    self.coursed("cell_front", s0, s1, 0.0, roof0 + IMPOST,
                                 c0, c1, material)
                else:
                    self.coursed("cell_front", c0, c1, 0.0, roof0 + IMPOST,
                                 s0, s1, material)
                continue
            door_w = min(1.6, max(1.0, span_w - 1.4))
            rise = self.d.arch_ratio * door_w
            crown = door_spring + rise
            j0, j1 = center - door_w / 2 + 0.3, center + door_w / 2 - 0.3
            if axis == "x":
                self.coursed("cell_front", s0, j0, 0.0, door_spring + IMPOST,
                             c0, c1, material)
                self.coursed("cell_front", j1, s1, 0.0, door_spring + IMPOST,
                             c0, c1, material)
                self.arch("cell_arch", center, (c0 + c1) / 2, door_w, door_spring,
                          rise, FRONT_T, along="x", material=material,
                          bay=f"hujra.{center:05.2f}")
                self.coursed("cell_front", s0, s1, crown, roof0 + IMPOST,
                             c0, c1, material)
            else:
                self.coursed("cell_front", c0, c1, 0.0, door_spring + IMPOST,
                             s0, j0, material)
                self.coursed("cell_front", c0, c1, 0.0, door_spring + IMPOST,
                             j1, s1, material)
                self.arch("cell_arch", (c0 + c1) / 2, center, door_w, door_spring,
                          rise, FRONT_T, along="z", material=material,
                          bay=f"hujra.{center:05.2f}")
                self.coursed("cell_front", c0, c1, crown, roof0 + IMPOST,
                             s0, s1, material)

    def _wall_stretches(self, a0, a1, part_range):
        """Wall stretches between partition faces over one front run."""
        edges, start = [], a0
        pos = part_range[0] + 1.5
        while pos < part_range[1] - 1.5:
            if a0 + 0.18 < pos < a1 - 0.18:
                edges.append((start, pos - 0.15))
                start = pos + 0.15
            pos += 3.0
        edges.append((start, a1))
        return [(s, e) for s, e in edges if e - s > 0.12]


@dataclass(frozen=True)
class Design:
    variant: str
    finish: str
    detail: str
    width: float
    depth: float
    height: float
    arch_ratio: float
    dome_apex_ratio: float
    muqarnas_tiers: int

    @classmethod
    def resolve(cls, request):
        rng = random.Random(request.seed)
        automatic = request.variant in ("auto", "random")
        if (not automatic and request.variant not in VARIANTS) or request.detail not in DETAILS:
            raise ValueError("unknown iwan-court variant or detail level")
        if request.openings != "auto":
            raise ValueError("iwan-court uses automatic four-centred openings")
        params = request.parameters
        if set(params) - set(CONTROLS):
            raise ValueError(f"unknown iwan-court parameters: {sorted(set(params) - set(CONTROLS))}")
        finish = params.get("finish", rng.choice(tuple(FINISHES)))
        if not isinstance(finish, str) or finish not in FINISHES:
            raise ValueError(f"iwan-court finish must be one of {', '.join(FINISHES)}")
        variant = request.variant
        if automatic:
            variant = rng.choice(VARIANTS)
        w = _number("width", request.width, DEFAULTS["width"], 18, 60)
        d = _number("depth", request.depth, DEFAULTS["depth"], 20, 70)
        h = _number("height", request.height, DEFAULTS["height"], 8, 22)
        arch_ratio = round(rng.uniform(0.48, 0.68), 3)
        dome_apex_ratio = round(rng.uniform(0.58, 0.78), 3)
        muqarnas_tiers = rng.randint(3, 6)
        return cls(variant=variant, finish=finish, detail=request.detail, width=w, depth=d,
                   height=h, arch_ratio=arch_ratio, dome_apex_ratio=dome_apex_ratio,
                   muqarnas_tiers=muqarnas_tiers)


def build_document(request: BuildRequest) -> BuildDocument:
    d = Design.resolve(request)
    builder = PersianBuilder(d)
    builder.compose()
    resolved = replace(request, architect=ARCHITECT.id, product=PRODUCT, style="persian",
                       variant=d.variant, width=d.width, height=d.height, depth=d.depth,
                       parameters={"finish": d.finish})
    required = VARIANT_REQUIRED.get(d.variant, REQUIRED)
    return BuildDocument(architect=ARCHITECT, request=resolved, placements=builder.placements,
                         metadata={"anchors": builder.anchors,
                                   "persian_rules": {
                                       "arch_rise_to_span": d.arch_ratio,
                                       "dome_apex_ratio": d.dome_apex_ratio,
                                       "muqarnas_tiers": d.muqarnas_tiers},
                                   "invariants": {"require_ground_contact": True,
                                                  "required_roles": sorted(required)}})


def _bounds(placement):
    import numpy as _np
    center = _np.asarray(placement.position, dtype=float)
    half = _np.asarray(placement.dimensions, dtype=float) / 2.0
    return center - half, center + half


def _plan_overlap(lo_a, hi_a, lo_b, hi_b):
    return (min(hi_a[0], hi_b[0]) - max(lo_a[0], lo_b[0]) > 1e-6 and
            min(hi_a[2], hi_b[2]) - max(lo_a[2], lo_b[2]) > 1e-6)


def persian_checks(document, scene):
    """Check authored geometry from the placements themselves.

    Every rule is measured from finished envelopes or from parameters the
    builder recorded while laying the member, so the audit survives a
    document round-trip with its mutable metadata stripped.
    """
    from .quality import _check, _envelope_connectivity_checks

    placements = list(document.placements)
    roles = {p.role for p in placements}
    variant = document.request.variant
    required = VARIANT_REQUIRED.get(variant, REQUIRED)
    checks = [_check("persian.required_roles", required <= roles,
                     "The court has its defining arcade, portal, iwan and masonry members.",
                     "The court is missing defining members.",
                     expected=sorted(required), observed=sorted(roles))]
    checks.extend(_envelope_connectivity_checks(document, "persian"))

    # Masonry only: nothing is timber and no span is a lintel.
    timber = [p.id for p in placements
              if "timber" in str(p.parameters.get("construction", "")).lower()
              or "timber" in str(p.material).lower()]
    lintels = [p.id for p in placements
               if p.parameters.get("spanning") == "lintel"
               or p.parameters.get("unit") == "lintel"]
    checks.append(_check("persian.masonry_only", not timber and not lintels,
                         "Every spanning member is arch, vault or dome masonry.",
                         "A timber member or lintel appeared in a masonry-only tradition.",
                         primitive_ids=sorted(set(timber + lintels))))

    # Four-centred proportion band, measured from the arch records.
    arches = [p for p in placements if p.parameters.get("construction") == "arch course"]
    ratios = sorted({round(float(p.parameters["rise"]) / float(p.parameters["span"]), 4)
                     for p in arches})
    in_band = bool(arches) and all(0.45 <= float(p.parameters["rise"]) / float(p.parameters["span"]) <= 0.75
                                   for p in arches)
    checks.append(_check("persian.four_centred_band", in_band,
                         "Every opening and arcade bay is a four-centred arch with rise-to-span between 0.45 and 0.75.",
                         "An arch leaves the 0.45..0.75 rise-to-span band.",
                         observed=ratios[:8], expected=[0.45, 0.75]))

    # Every springing bears on a pier or wall below its own footprint.
    spans_only = [p for p in placements
                  if p.parameters.get("construction") in ("arch course", "vault course")]
    solids = [p for p in placements
              if p.parameters.get("construction") not in ("arch course", "vault course")
              and p.role != "muqarnas_cell"]
    unsupported = []
    for p in spans_only:
        lo, hi = _bounds(p)
        spring = lo[1]
        foot = [q for q in solids
                if _plan_overlap(lo, hi, *_bounds(q))
                and lo[1] - 0.05 <= q.position[1] + q.dimensions[1] / 2 <= spring + 0.4]
        if not foot:
            unsupported.append(p.id)
    checks.append(_check("persian.springing_support", not unsupported,
                         "Every arch and vault springing rests on a pier or wall.",
                         "A springing hangs over open air.",
                         primitive_ids=unsupported))

    domed = variant in ("four-iwan", "chahar-taq")
    squinches = [p for p in placements if p.role == "squinch"]
    bands = [p for p in placements if p.role == "transition_band"]
    drums = [p for p in placements if p.role == "drum"]
    wall_roles = ("sanctuary_wall", "chamber_wall", "iwan_wall")
    if domed:
        walls = [p for p in placements if p.role in wall_roles]
        # The zone of transition: squinches bridge the corners on the walls.
        floating = []
        for p in squinches:
            lo, hi = _bounds(p)
            under = [q for q in walls
                     if _plan_overlap(lo, hi, *_bounds(q))
                     and abs(q.position[1] + q.dimensions[1] / 2 - (lo[1] - 0.05)) < 0.35]
            if not under:
                floating.append(p.id)
        checks.append(_check("persian.squinch_bearing", len(squinches) == 4 and not floating,
                             "Four corner squinches bridge the chamber on its walls.",
                             "The squinch zone is missing or does not bear on the walls.",
                             primitive_ids=floating, observed=len(squinches), expected=4))
        # Each band panel sits on the stage below with no real overhang.
        supports = walls + squinches + bands
        hanging = []
        for p in bands:
            lo, hi = _bounds(p)
            depth = min(p.dimensions[0], p.dimensions[2])
            best = None
            for q in supports:
                if q is p:
                    continue
                qlo, qhi = _bounds(q)
                if qhi[1] > lo[1] + 1e-6:
                    continue
                if not _plan_overlap(lo, hi, qlo, qhi):
                    continue
                gap = lo[1] - qhi[1]
                if best is None or gap < best:
                    best = gap
            if best is None or best > depth / 3 + 0.05:
                hanging.append(p.id)
        checks.append(_check("persian.band_stack", len(bands) >= 16 and not hanging,
                             "The sixteen-sided band sits on squinches and walls with no member overhanging its support by a third of its depth.",
                             "A band panel hangs or overhangs its support.",
                             primitive_ids=hanging, observed=len(bands), expected=">=16"))
        # The drum lands inside the band and its diameter is at most S.
        drum_ok = bool(drums) and all(
            float(p.parameters["drum_radius"]) * 2 <= float(p.parameters["chamber_side"]) + 1e-6
            for p in drums)
        if drums:
            r_inner = min(float(p.parameters["r_inner"]) for p in bands)
            r_outer = max(float(p.parameters["r_outer"]) for p in bands)
            drum_rad = float(drums[0].parameters["drum_radius"])
            stone_half = 0.275
            drum_ok = (drum_ok
                       and drum_rad + stone_half <= r_outer + 1e-6
                       and drum_rad - stone_half >= r_inner - 2 / 3 * 0.55 - 1e-6)
        checks.append(_check("persian.drum_diameter", drum_ok,
                             "The circular drum lands inside the band with diameter at most the chamber side.",
                             "The drum overhangs the band or exceeds the chamber side."))

    # The dome: pointed, coursed, and springing from the drum.
    dome_anchors = [a for a in (document.metadata.get("anchors") or {}).get("dome", [])
                    if "apex" in a]
    dome_stones = [p for p in placements if p.role == "dome_course"
                   and not p.parameters.get("keystone")]
    dome_ok = bool(dome_anchors) if domed else not dome_stones
    if domed and dome_anchors:
        anchor = dome_anchors[0]
        courses = {int(p.parameters["course"]) for p in dome_stones}
        apex_ok = 0.55 <= float(anchor["apex_ratio"]) <= 0.8
        coursed_ok = len(courses) >= 8
        by_course = {}
        for p in dome_stones:
            by_course.setdefault(int(p.parameters["course"]), []).append(p)
        ordered = sorted(by_course)
        monotone = all(
            max(float(q.parameters["r_lo"]) for q in by_course[c]) >=
            max(float(q.parameters["r_hi"]) for q in by_course[c + 1]) - 1e-6
            for c in ordered[:-1])
        pointed = True
        if len(ordered) >= 3:
            top = ordered[-1]
            stones = by_course[top]
            dr = max(float(s.parameters["r_lo"]) for s in stones)
            dy = max(float(s.parameters["y_hi"]) - float(s.parameters["y_lo"]) for s in stones)
            pointed = dr / max(dy, 1e-6) <= 4.0
        spring_ok = any(
            abs(float(s.parameters["y_lo"]) - float(anchor["base_y"])) <= 0.01
            and abs(float(s.parameters["r_lo"]) - float(anchor["diameter"]) / 2) <= 0.05
            for s in dome_stones)
        dome_ok = apex_ok and coursed_ok and monotone and pointed and spring_ok
        checks.append(_check("persian.dome_rule", dome_ok,
                             "The pointed dome rises 0.55 to 0.8 of its diameter in named courses from the drum.",
                             "The dome is missing, out of proportion, uncoursed, or springs off its drum.",
                             observed={"apex_ratio": anchor["apex_ratio"],
                                       "courses": len(courses)}))
    checks.append(_check("persian.dome_present", dome_ok,
                         "Domed variants carry their dome and domed kiosks stay empty where no dome belongs.",
                         "The dome is missing."))

    # Iwans: vaulted halls in the 0.8..1.6 depth band, framed above the arcade.
    iwans = [p for p in placements if p.role == "iwan"]
    if "iwan" in required:
        pier_tops = [p.position[1] + p.dimensions[1] / 2 for p in placements
                     if p.role == "riwaq_pier"]
        arcade_roof = max(pier_tops) if pier_tops else 0.0
        proportions = [float(p.parameters["depth"]) / float(p.parameters["width"])
                      for p in iwans]
        prop_ok = bool(iwans) and all(0.8 <= r <= 1.6 for r in proportions)
        frames_ok = all(float(p.parameters["frame_top"]) > arcade_roof + 0.4
                        for p in iwans)
        walls_ok = True
        for p in iwans:
            lo, hi = _bounds(p)
            flank = [q for q in placements if q.role in wall_roles
                     and _plan_overlap(lo, hi, *_bounds(q))]
            if len(flank) < 3:
                walls_ok = False
        checks.append(_check("persian.iwan_rule", prop_ok and frames_ok and walls_ok,
                             "Iwans are vaulted halls walled on three sides, framed by pishtaqs above the arcade roof.",
                             "An iwan is out of proportion, unframed, or missing its flank walls.",
                             observed=sorted(set(round(r, 3) for r in proportions))[:6]))

    # Muqarnas: corbelled tiers whose every cell bears on masonry.
    cells = [p for p in placements if p.role == "muqarnas_cell"]
    tiers = sorted({int(p.parameters["tier"]) for p in cells})
    hood_ok = bool(cells) and 3 <= len(tiers) <= 6 and tiers[0] == 0
    if hood_ok:
        radii = {}
        for t in tiers:
            radii[t] = max(float(p.parameters["radius"]) for p in cells
                           if int(p.parameters["tier"]) == t)
        steps = [radii[t] - radii[t + 1] for t in tiers[:-1]]
        depth = float(cells[0].dimensions[1])
        hood_ok = (all(-1e-6 <= s <= depth / 2 + 1e-6 for s in steps))
    cell_solids = [p for p in placements if p.role != "muqarnas_cell"]
    resting = []
    for p in cells:
        lo, hi = _bounds(p)
        base = lo[1]
        ok = False
        for q in cell_solids:
            qlo, qhi = _bounds(q)
            if qhi[1] > base + 1e-6:
                continue
            if _plan_overlap(lo, hi, qlo, qhi) and base - qhi[1] < 0.15:
                ok = True
                break
        if not ok:
            for q in cells:
                if q is p or q.parameters.get("tier") != p.parameters.get("tier"):
                    continue
                if _plan_overlap(lo, hi, *_bounds(q)):
                    ok = True
                    break
        if not ok:
            resting.append(p.id)
    checks.append(_check("persian.muqarnas_corbel", hood_ok and not resting,
                         "Portal hoods are muqarnas of 3 to 6 corbelled tiers, each stepping back at most half a cell depth, every cell resting on masonry.",
                         "The muqarnas hood is missing, too shallow, overhangs a tier, or a cell rests on air.",
                         primitive_ids=resting, observed=tiers))

    if variant == "four-iwan":
        # Symmetry about the qibla axis, and the qibla strictly largest.
        by_side = {}
        for p in iwans:
            by_side.setdefault(p.parameters["side"], []).append(p)
        mirrored = set(by_side) == {"north", "south", "east", "west"}
        signatures = {(round(p.position[0], 4), round(p.position[1], 4), round(p.position[2], 4),
                      tuple(round(v, 4) for v in p.dimensions)) for p in iwans}
        for p in iwans:
            if abs(p.position[0]) < 1e-6:
                continue
            mirror = (round(-p.position[0], 4), round(p.position[1], 4), round(p.position[2], 4),
                     tuple(round(v, 4) for v in p.dimensions))
            if mirror not in signatures:
                mirrored = False
        qibla = by_side.get("south", [None])[0]
        largest = bool(qibla) and all(
            float(qibla.parameters["width"]) > float(p.parameters["width"])
            and float(qibla.parameters["height"]) > float(p.parameters["height"])
            for side, group in by_side.items() if side != "south" for p in group)
        axis_ok = mirrored and largest
        if dome_anchors:
            axis_ok = axis_ok and abs(float(dome_anchors[0]["x"]) - float(qibla.position[0])) < 1e-6
        minarets = [p for p in placements if p.role == "minaret"]
        if len(minarets) == 2:
            axis_ok = axis_ok and abs(minarets[0].position[0] + minarets[1].position[0]
                                      - 2 * float(qibla.position[0])) < 1e-6
        checks.append(_check("persian.qibla_axis", axis_ok,
                             "Court, iwans, sanctuary and portal are symmetric about the qibla axis, and the qibla iwan is strictly the largest.",
                             "The composition lost its axis symmetry or the qibla dominance."))

        # Minarets: cylindrical taper, corbelled balcony, tip above the dome.
        caps = [p for p in placements if p.role == "minaret_cap"]
        drums = [p for p in placements if p.role == "minaret_drum"]
        balconies = [p for p in placements if p.role == "minaret_balcony"]
        taper_ok = bool(minarets)
        for m in minarets:
            shaft = [p for p in drums if p.parameters.get("host") == m.id]
            shaft.sort(key=lambda p: p.position[1])
            if len(shaft) < 2 or shaft[-1].dimensions[0] >= shaft[0].dimensions[0]:
                taper_ok = False
        corbel_ok = all(float(b.parameters["overhang"]) <= float(b.dimensions[1]) / 2 + 1e-6
                       for b in balconies)
        tip_ok = (bool(caps) and bool(dome_anchors)
                  and all(c.position[1] > float(dome_anchors[0]["apex"]) for c in caps))
        checks.append(_check("persian.minaret_rule",
                             taper_ok and corbel_ok and tip_ok and len(balconies) >= 2,
                             "Minarets are cylindrical and tapering, with corbelled balconies, their tips above the dome apex.",
                             "A minaret is missing, untapered, corbels too far, or fails to clear the dome."))
    return checks
