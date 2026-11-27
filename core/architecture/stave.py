"""Nordic Stave architect: medieval Norwegian stave churches and timber lofts.

A stave building is a load-bearing timber frame of massive vertical pine logs
(staves) raised clear of the damp earth on a stone foundation sill (svill) and
heavy timber ground sills (grunnstokker). Steeply pitched shingled roofs shed
heavy northern snows; an encircling covered gallery (svalgang) with dwarf
arcades shields the inner walls from wind and rain.

Two products belong to the tradition:
- ``stave-church``: the Borgund-type raised-nave mast church (``borgund``) and
  the single-nave church (``single-nave``), each with a projecting eastern
  choir and semicircular apse, a tiered ridge turret (takrytter) and spire,
  pierced ridge crests (mønekam) and dragon-head gable finials (dragehoder).
- ``stave-loft``: the two-storey Norwegian timber loft (``loft-house``) and
  raised storehouse (``stabbur``), whose upper storey and arcaded svalgang
  gallery cantilever outward over the ground floor, reached by an external
  timber stair under a steep shingled gable roof.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from .historic import Composer, Form, clamp, extent, hull, oriented_profile, unmirrored
from .schema import ArchitectInfo

ARCHITECT = ArchitectInfo("stave", "1.0.0", "Nordic Stave Architect")
STYLE = "nordic-stave"

KINDS = {
    "core_fill": "core",
    "stone_sill": "foundation", "podium_paving": "slab", "footing_pier": "foundation",
    "step": "circulation", "stair_stringer": "circulation", "stair_step": "circulation",
    "sill_beam": "beam", "head_plate": "beam", "cantilever_beam": "beam",
    "ridge_beam": "beam", "scissor_brace": "beam", "rafter": "beam", "bargeboard": "beam",
    "nave_stave": "column", "aisle_stave": "column", "choir_stave": "column",
    "gallery_post": "column", "turret_post": "column",
    "nave_plank": "infill", "aisle_plank": "infill", "choir_plank": "infill",
    "apse_plank": "infill", "gallery_panel": "infill", "gable_panel": "infill",
    "upper_plank": "infill", "turret_panel": "infill",
    "portal_frame": "ornament", "door": "infill",
    "gallery_floor": "platform",
    "gallery_roof": "roof", "aisle_roof": "roof", "nave_roof": "roof",
    "choir_roof": "roof", "apse_roof": "roof", "turret_roof": "roof", "spire": "roof",
    "ridge_crest": "ornament", "dragon_finial": "ornament", "spire_finial": "ornament",
}

FORMS = {form.name: form for form in (
    Form("stave-church",
         "Medieval Norwegian stave church: stone sill, tiered steep shingled roofs, svalgang gallery, apse, ridge turret and dragon finials",
         (14, 18, 22), ("borgund", "single-nave"),
         frozenset({"stone_sill", "sill_beam", "nave_stave", "nave_plank", "gallery_post",
                    "gallery_roof", "nave_roof", "choir_plank", "apse_plank", "apse_roof",
                    "turret_panel", "turret_roof", "spire", "ridge_crest", "dragon_finial", "spire_finial"}),
         {"bays": 5, "levels": 3},
         rules={"bays": [3, 7], "bays_meaning": "stave bays along each side of the nave",
                "levels": [2, 3], "levels_meaning": "tiers of the ridge turret (takrytter)",
                "height_to_width": [0.85, 2.2], "depth_to_width": [1.1, 2.2],
                "structure": "vertical pine staves on a stone sill, steep shingled roofs, encircling svalgang and takrytter"}),
    Form("stave-loft",
         "Norwegian two-storey timber loft or stabbur: cantilevered upper svalgang gallery, external stair and steep shingled gable roof",
         (10, 11, 12), ("loft-house", "stabbur"),
         frozenset({"stone_sill", "sill_beam", "nave_stave", "nave_plank", "cantilever_beam",
                    "gallery_floor", "gallery_post", "upper_plank", "stair_step",
                    "nave_roof", "ridge_crest", "dragon_finial"}),
         {"bays": 3},
         rules={"bays": [3, 5], "bays_meaning": "arcade bays across the front upper gallery",
                "height_to_width": [0.75, 1.8], "depth_to_width": [0.8, 1.8],
                "structure": "timber frame on stone sill or stabbe piers with cantilevered upper storey and external stair"}),
)}

# Pine-tarred Nordic timber palette: deep tarred-brown shingles and staves, warm
# weathered oak carvings and sills, pale grey granite plinth, dark wrought iron.
GRANITE = (142, 140, 136, 255)
TAR_TIMBER = (72, 52, 38, 255)
PINE_WALL = (92, 68, 48, 255)
OAK_TRIM = (122, 88, 56, 255)
SHINGLE = (52, 42, 36, 255)
IRON = (58, 60, 64, 255)

PALETTE = {
    "core_fill": (118, 114, 108, 255),
    "foundation": GRANITE, "slab": (156, 152, 144, 255),
    "circulation": OAK_TRIM, "step": GRANITE,
    "column": TAR_TIMBER, "beam": OAK_TRIM, "infill": PINE_WALL,
    "door": (62, 44, 32, 255), "platform": OAK_TRIM,
    "roof": SHINGLE, "ornament": OAK_TRIM,
    "portal_frame": (136, 98, 60, 255),
    "ridge_crest": (114, 82, 52, 255),
    "dragon_finial": (132, 92, 54, 255),
    "spire_finial": IRON,
}

SUBSTANCES = {
    "core_fill": "granite",
    "stone_sill": "granite", "podium_paving": "granite", "footing_pier": "granite",
    "step": "granite",
    "sill_beam": "oak", "cantilever_beam": "oak", "scissor_brace": "oak",
    "bargeboard": "oak", "portal_frame": "oak", "ridge_crest": "oak", "dragon_finial": "oak",
    "stair_stringer": "oak", "stair_step": "oak", "gallery_floor": "oak",
    "gallery_roof": "cedar", "aisle_roof": "cedar", "nave_roof": "cedar",
    "choir_roof": "cedar", "apse_roof": "cedar", "turret_roof": "cedar", "spire": "cedar",
    "spire_finial": "wrought-iron",
}
DEFAULT_SUBSTANCE = "softwood"


def substances_for(form):
    """Substances actually used by each stave product."""
    base = {"granite", "softwood", "oak", "cedar"}
    return base | {"wrought-iron"} if form.name == "stave-church" else base


def _prism_slope(x0, x1, y_low, y_high, z0, z1, thick, side):
    """One pitched roof slab rising from ``x0`` (outer eave) to ``x1`` (inner ridge/wall).

    ``side`` is -1 for the left slope (rises toward +X) or +1 for the right slope
    (rises toward -X). Thickness is measured vertically so eaves and ridges join cleanly.
    """
    pts = []
    if side < 0:
        # x0 is outer low eave, x1 is inner high ridge
        for z in (z0, z1):
            pts.extend([
                (x0, y_low, z), (x0, y_low + thick, z),
                (x1, y_high, z), (x1, y_high + thick, z),
            ])
    else:
        # x1 is outer low eave, x0 is inner high ridge
        for z in (z0, z1):
            pts.extend([
                (x1, y_low, z), (x1, y_low + thick, z),
                (x0, y_high, z), (x0, y_high + thick, z),
            ])
    return hull(pts)


def _prism_slope_z(x0, x1, z_low, z_high, y_low, y_high, thick):
    """One lean-to roof slab rising along Z from ``z_low`` (eave) to ``z_high`` (wall)."""
    pts = []
    for x in (x0, x1):
        pts.extend([
            (x, y_low, z_low), (x, y_low + thick, z_low),
            (x, y_high, z_high), (x, y_high + thick, z_high),
        ])
    return hull(pts)


class StaveBuilder(Composer):
    PALETTE = PALETTE
    TIMBER_KINDS = frozenset({"column", "beam", "bracket", "infill", "platform", "roof"})

    def compose(self):
        if self.form.name == "stave-church":
            return self.compose_church()
        return self.compose_loft()

    # --------------------------------------------------------- shared helpers
    def plank_run(self, role, axis, along0, along1, cross, thickness, y0, y1, **params):
        """Lay a vertical stave wall as individual vertical tongue-and-groove timber planks (veggtiljer)."""
        length = along1 - along0
        if length <= 1e-4 or y1 <= y0 + 1e-4:
            return
        # Plank width scales with detail level: discrete boards, never a single monolithic wall slab.
        target_w = {"concept": 1.15, "standard": 0.62, "presentation": 0.42}[self.detail]
        count = max(2, int(math.ceil(length / target_w - 1e-9)))
        step = length / count
        ht = thickness / 2.0
        for i in range(count):
            a0 = along0 + i * step
            a1 = along0 + (i + 1) * step
            if axis == "x":
                self.member(role, a0, a1, y0, y1, cross - ht, cross + ht, plank=i, **params)
            else:
                self.member(role, cross - ht, cross + ht, y0, y1, a0, a1, plank=i, **params)

    def gable_roof(self, role, hx, z0, z1, y_eave, y_ridge, thick=0.26, over_x=0.42, over_z=0.35,
                   barge=True, crest=True, dragons=True, tier="nave"):
        """Steep pitched gable roof: real timber rafters under overlapping courses of wood shingles."""
        run_x = hx + over_x
        slope = (y_ridge - y_eave) / max(run_x, 0.2)
        za, zb = z0 - over_z, z1 + over_z

        # 1. Structural pine rafters from eave to ridge, exposed when Open roof is toggled.
        rafter_pitch = {"concept": 1.8, "standard": 1.05, "presentation": 0.72}[self.detail]
        n_rafters = max(3, int(round((z1 - z0) / rafter_pitch)) + 1)
        rt = 0.065
        rh = 0.16
        for r in range(n_rafters):
            rz = z0 + 0.15 + (z1 - z0 - 0.30) * r / max(1, n_rafters - 1)
            for side in (-1, 1):
                ex = side * run_x
                pts = [
                    (ex, y_eave - rh, rz - rt), (ex, y_eave, rz - rt),
                    (0.0, y_ridge - rh, rz - rt), (0.0, y_ridge, rz - rt),
                    (ex, y_eave - rh, rz + rt), (ex, y_eave, rz + rt),
                    (0.0, y_ridge - rh, rz + rt), (0.0, y_ridge, rz + rt),
                ]
                self.mesh("rafter", hull(pts), construction="pine roof rafter", roof_tier=tier, rafter=r)

        # 2. Overlapping shingle courses up each slope, segmented along Z.
        courses = {"concept": 3, "standard": 5, "presentation": 7}[self.detail]
        seg_len = {"concept": 1.6, "standard": 1.2, "presentation": 0.85}[self.detail]
        segs_z = max(3, int(math.ceil((zb - za) / seg_len)))
        for c in range(courses):
            u0 = c / courses
            u1 = (c + 1) / courses
            ya = y_eave + (y_ridge - y_eave) * u0
            yb = y_eave + (y_ridge - y_eave) * u1
            # Slight overlap step between shingle courses so each course reads as laid shingles.
            xa_l = -run_x * (1.0 - u0)
            xb_l = -run_x * (1.0 - u1)
            xa_r = run_x * (1.0 - u1)
            xb_r = run_x * (1.0 - u0)
            for k in range(segs_z):
                s0 = za + (zb - za) * k / segs_z
                s1 = za + (zb - za) * (k + 1) / segs_z
                self.mesh(role, _prism_slope(xa_l, xb_l, ya, yb, s0, s1, thick, -1),
                          construction="tarred wood shingles", slope=round(slope, 3),
                          roof_tier=tier, course=c, side="left")
                self.mesh(role, _prism_slope(xa_r, xb_r, ya, yb, s0, s1, thick, 1),
                          construction="tarred wood shingles", slope=round(slope, 3),
                          roof_tier=tier, course=c, side="right")

        # Continuous timber ridge beam inside the peak.
        rb_h = max(0.24, thick)
        self.member("ridge_beam", -0.16, 0.16, y_ridge - rb_h, y_ridge + thick * 0.55, za, zb,
                    construction="timber ridge beam", roof_tier=tier)

        # Triangular gable end walls closing both ends.
        wall_t = 0.22
        y_wall_base = y_eave + (over_x * slope) - 0.12
        gable_poly = [(-hx, y_wall_base), (hx, y_wall_base), (0.0, y_ridge + 0.05)]
        for z_end, label in ((z0 + wall_t / 2, "west"), (z1 - wall_t / 2, "east")):
            self.mesh("gable_panel", oriented_profile(gable_poly, wall_t, "x", 0.0, z_end),
                      construction="vertical stave gable", face=label, roof_tier=tier)

        # Raking carved bargeboards at both gable ends.
        if barge:
            bb_t = 0.14
            for z_end in (za + bb_t / 2, zb - bb_t / 2):
                for side in (-1, 1):
                    ex = side * run_x
                    pts = [
                        (ex, y_eave - 0.08, z_end - bb_t / 2), (ex, y_eave + thick + 0.04, z_end - bb_t / 2),
                        (0.0, y_ridge - 0.08, z_end - bb_t / 2), (0.0, y_ridge + thick + 0.06, z_end - bb_t / 2),
                        (ex, y_eave - 0.08, z_end + bb_t / 2), (ex, y_eave + thick + 0.04, z_end + bb_t / 2),
                        (0.0, y_ridge - 0.08, z_end + bb_t / 2), (0.0, y_ridge + thick + 0.06, z_end + bb_t / 2),
                    ]
                    self.mesh("bargeboard", hull(pts), construction="carved oak bargeboard", roof_tier=tier)

        # Pierced ridge cresting (mønekam) along the ridge.
        if crest:
            crest_h = clamp(0.09 * (y_ridge - y_eave), 0.26, 0.52)
            self.member("ridge_crest", -0.07, 0.07, y_ridge + thick * 0.35, y_ridge + thick * 0.35 + crest_h * 0.45,
                        za + 0.12, zb - 0.12, construction="carved oak ridge crest", roof_tier=tier)
            count = max(3, int(round((zb - za) / 0.65)))
            for i in range(count):
                zc = za + 0.25 + (zb - za - 0.5) * (i + 0.5) / count
                self.member("ridge_crest", -0.055, 0.055,
                            y_ridge + thick * 0.35 + crest_h * 0.35, y_ridge + thick * 0.35 + crest_h,
                            zc - 0.09, zc + 0.09, construction="pierced ridge finial", roof_tier=tier)

        # Dragon-head finials (dragehoder) sweeping outward and upward from both gable peaks.
        if dragons:
            self.dragon_head(za, y_ridge + thick * 0.4, -1, scale=clamp(hx / 2.6, 0.75, 1.35), tier=tier)
            self.dragon_head(zb, y_ridge + thick * 0.4, 1, scale=clamp(hx / 2.6, 0.75, 1.35), tier=tier)
        return slope

    def dragon_head(self, z_base, y_base, direction, scale=1.0, tier="nave"):
        """Curved dragon-head gable finial (dragehode) projecting along ``direction`` (-1 or +1 on Z)."""
        # Sweeping neck, crested head, and projecting snout/tongue built as connected carved oak segments.
        d = float(direction)
        reach = 0.85 * scale
        rise = 0.92 * scale
        w = 0.09 * scale
        neck = [
            (-w, y_base - 0.16 * scale, z_base - d * 0.18 * scale),
            (w, y_base - 0.16 * scale, z_base - d * 0.18 * scale),
            (-w, y_base + 0.16 * scale, z_base - d * 0.12 * scale),
            (w, y_base + 0.16 * scale, z_base - d * 0.12 * scale),
            (-w * 0.85, y_base + 0.55 * rise, z_base + d * 0.58 * reach),
            (w * 0.85, y_base + 0.55 * rise, z_base + d * 0.58 * reach),
            (-w * 0.85, y_base + 0.82 * rise, z_base + d * 0.46 * reach),
            (w * 0.85, y_base + 0.82 * rise, z_base + d * 0.46 * reach),
        ]
        head = [
            (-w * 0.85, y_base + 0.52 * rise, z_base + d * 0.48 * reach),
            (w * 0.85, y_base + 0.52 * rise, z_base + d * 0.48 * reach),
            (-w * 0.85, y_base + 0.86 * rise, z_base + d * 0.44 * reach),
            (w * 0.85, y_base + 0.86 * rise, z_base + d * 0.44 * reach),
            (-w * 0.65, y_base + rise, z_base + d * reach),
            (w * 0.65, y_base + rise, z_base + d * reach),
            (-w * 0.65, y_base + 0.72 * rise, z_base + d * 0.96 * reach),
            (w * 0.65, y_base + 0.72 * rise, z_base + d * 0.96 * reach),
        ]
        self.mesh("dragon_finial", hull(neck), construction="carved oak dragon finial",
                  roof_tier=tier, end="east" if d > 0 else "west", part="neck")
        self.mesh("dragon_finial", hull(head), construction="carved oak dragon finial",
                  roof_tier=tier, end="east" if d > 0 else "west", part="head")

    def stave_cylinder(self, role, x, z, y0, y1, radius, **params):
        """One vertical cylindrical timber stave."""
        return self.add(role, (x, (y0 + y1) / 2, z), (2 * radius, y1 - y0, 2 * radius), "CYLINDER",
                        radius=radius, height=y1 - y0, sections=12,
                        construction="vertical pine stave", **params)

    def svalgang_arcade(self, x0, x1, z0, z1, y_floor, y_eave, bays_x, bays_z, open_west=True):
        """Encircling covered gallery with dwarf wall, arcade posts, wall plate and west portal."""
        rail_h = min(1.05, (y_eave - y_floor) * 0.44)
        y_rail = y_floor + rail_h
        wt = 0.22
        post_r = 0.11
        portal_hw = min(0.85, (x1 - x0) * 0.14)

        # Dwarf walls laid as vertical stave boards on left, right, and west front.
        self.plank_run("gallery_panel", "z", z0, z1, x0 + wt / 2, wt, y_floor, y_rail,
                       construction="stave dwarf plank", face="left")
        self.plank_run("gallery_panel", "z", z0, z1, x1 - wt / 2, wt, y_floor, y_rail,
                       construction="stave dwarf plank", face="right")
        if open_west:
            self.plank_run("gallery_panel", "x", x0, -portal_hw, z0 + wt / 2, wt, y_floor, y_rail,
                           construction="stave dwarf plank", face="west")
            self.plank_run("gallery_panel", "x", portal_hw, x1, z0 + wt / 2, wt, y_floor, y_rail,
                           construction="stave dwarf plank", face="west")
            # Carved portal jambs and lintel framing the entrance.
            for sx in (-1, 1):
                self.member("portal_frame", sx * portal_hw - 0.14, sx * portal_hw + 0.14,
                            y_floor, y_eave - 0.18, z0 - 0.08, z0 + wt + 0.08,
                            construction="carved oak portal jamb")
            self.member("portal_frame", -portal_hw - 0.18, portal_hw + 0.18,
                        y_eave - 0.42, y_eave + 0.02, z0 - 0.1, z0 + wt + 0.1,
                        construction="carved oak portal lintel")
        else:
            self.plank_run("gallery_panel", "x", x0, x1, z0 + wt / 2, wt, y_floor, y_rail,
                           construction="stave dwarf plank", face="west")

        # Arcade posts along left, right, and west front.
        for j in range(bays_z + 1):
            pz = z0 + (z1 - z0) * j / bays_z
            for px in (x0 + wt / 2, x1 - wt / 2):
                self.stave_cylinder("gallery_post", px, pz, y_rail - 0.05, y_eave, post_r, gallery=True)
        for i in range(1, bays_x):
            px = x0 + (x1 - x0) * i / bays_x
            if open_west and abs(px) < portal_hw + 0.25:
                continue
            self.stave_cylinder("gallery_post", px, z0 + wt / 2, y_rail - 0.05, y_eave, post_r, gallery=True)

        # Continuous timber wall plates atop the gallery posts.
        self.member("head_plate", x0 - 0.04, x0 + wt + 0.04, y_eave - 0.18, y_eave + 0.06, z0, z1,
                    construction="svalgang wall plate")
        self.member("head_plate", x1 - wt - 0.04, x1 + 0.04, y_eave - 0.18, y_eave + 0.06, z0, z1,
                    construction="svalgang wall plate")
        self.member("head_plate", x0, x1, y_eave - 0.18, y_eave + 0.06, z0 - 0.04, z0 + wt + 0.04,
                    construction="svalgang wall plate")

    def lean_to_roofs(self, role, outer_hx, inner_hx, z0, z1, z_inner0, y_eave, y_top,
                      thick=0.22, over=0.35, wrap_west=True, tier="gallery"):
        """Steep lean-to roofs wrapping the sides and west front with cleanly mitered hip corners."""
        run_x = (outer_hx - inner_hx) + over
        slope = (y_top - y_eave) / max(run_x, 0.2)
        ox = outer_hx + over
        ix = inner_hx - 0.1
        za = z0 - over if wrap_west else z0
        zb = z1 + 0.18
        zi = (z_inner0 + 0.1) if (wrap_west and z_inner0 > z0 + 0.2) else za

        # Mitered front corner segments (from za at outer eave to zi at inner wall), then straight runs.
        if wrap_west and zi > za + 0.1:
            for side in (-1, 1):
                ex, wx = side * ox, side * ix
                pts = [
                    (ex, y_eave, za), (ex, y_eave + thick, za),
                    (ex, y_eave, zi), (ex, y_eave + thick, zi),
                    (wx, y_top, zi), (wx, y_top + thick, zi),
                ]
                self.mesh(role, hull(pts), construction="tarred wood shingles",
                          slope=round(slope, 3), roof_tier=tier, side="right" if side > 0 else "left")
            # West hipped trapezoid meeting the left and right miters exactly along the diagonal.
            west_pts = [
                (-ox, y_eave, za), (-ox, y_eave + thick, za),
                (ox, y_eave, za), (ox, y_eave + thick, za),
                (-ix, y_top, zi), (-ix, y_top + thick, zi),
                (ix, y_top, zi), (ix, y_top + thick, zi),
            ]
            self.mesh(role, hull(west_pts), construction="tarred wood shingles",
                      slope=round(slope, 3), roof_tier=tier, side="west")

        courses = {"concept": 2, "standard": 4, "presentation": 5}[self.detail]
        seg_len = {"concept": 1.6, "standard": 1.2, "presentation": 0.9}[self.detail]
        segs = max(3, int(math.ceil((zb - zi) / seg_len)))
        for c in range(courses):
            u0, u1 = c / courses, (c + 1) / courses
            ya = y_eave + (y_top - y_eave) * u0
            yb = y_eave + (y_top - y_eave) * u1
            xa_l = -(ox + (ix - ox) * u0)
            xb_l = -(ox + (ix - ox) * u1)
            xa_r = ox + (ix - ox) * u1
            xb_r = ox + (ix - ox) * u0
            for k in range(segs):
                s0 = zi + (zb - zi) * k / segs
                s1 = zi + (zb - zi) * (k + 1) / segs
                self.mesh(role, _prism_slope(xa_l, xb_l, ya, yb, s0, s1, thick, -1),
                          construction="tarred wood shingles", slope=round(slope, 3),
                          roof_tier=tier, course=c, side="left")
                self.mesh(role, _prism_slope(xa_r, xb_r, ya, yb, s0, s1, thick, 1),
                          construction="tarred wood shingles", slope=round(slope, 3),
                          roof_tier=tier, course=c, side="right")
        return slope

    # ----------------------------------------------------------- stave church
    def compose_church(self):
        self.ratio(self.h / self.w, "height/width", 0.85, 2.2)
        self.ratio(self.d / self.w, "depth/width", 1.1, 2.2)
        bays = self.control("bays", *self.form.rules["bays"])
        turret_levels = self.control("levels", *self.form.rules["levels"])
        borgund = self.variant == "borgund"

        # Outer envelope locked to requested W, H, D so seed variation stays within 2%.
        over_gal = 0.35
        sval_hx = self.w / 2 - over_gal
        z_front = -self.d / 2
        z_back = self.d / 2

        apse_r = clamp(0.16 * self.w * self.vary("apse_r", 0.08), 1.35, min(2.8, sval_hx * 0.42))
        choir_d = clamp(0.18 * self.d * self.vary("choir_d", 0.08), 2.2, self.d * 0.25)
        z_apse = z_back - apse_r - 0.32
        z_choir0 = z_apse - choir_d
        y_sill = clamp(0.045 * self.h * self.vary("plinth", 0.06), 0.55, 1.05)
        steps = max(2, int(math.ceil(y_sill / 0.20)))
        z_sval0 = z_front + max(over_gal, steps * 0.28)

        gal_w = clamp(0.15 * self.w * self.vary("gal_w", 0.08), 1.0, 2.2)
        aisle_hx = max(2.2, sval_hx - gal_w)
        aisle_z0 = z_sval0 + gal_w
        aisle_z1 = z_choir0

        if borgund:
            aisle_w = clamp(0.16 * self.w * self.vary("aisle_w", 0.08), 1.0, max(1.05, aisle_hx - 1.5))
            nave_hx = max(1.4, aisle_hx - aisle_w)
            nave_z0 = aisle_z0 + min(aisle_w, (aisle_z1 - aisle_z0) * 0.20)
            nave_z1 = aisle_z1 - min(aisle_w, (aisle_z1 - aisle_z0) * 0.16)
        else:
            aisle_w = 0.0
            nave_hx = aisle_hx
            nave_z0 = aisle_z0
            nave_z1 = aisle_z1

        choir_hx = clamp(0.76 * nave_hx * self.vary("choir_w", 0.06), 1.25, max(1.3, aisle_hx - 0.5))
        apse_r = min(apse_r, choir_hx - 0.08)

        # Vertical heights calibrated so the spire finial reaches exactly self.h.
        sill_h = 0.28
        y_floor = y_sill + sill_h

        y_gal_eave = y_floor + clamp(0.11 * self.h * self.vary("gal_h", 0.06), 1.8, 2.7)
        gal_slope = 1.05 * self.vary("gal_slope", 0.06)
        y_gal_top = y_gal_eave + (gal_w + over_gal) * gal_slope

        if borgund:
            y_aisle_eave = y_gal_top + clamp(0.04 * self.h, 0.55, 1.1)
            aisle_slope = 1.12 * self.vary("aisle_slope", 0.06)
            y_aisle_top = y_aisle_eave + (aisle_w + 0.30) * aisle_slope
            y_nave_eave = y_aisle_top + clamp(0.08 * self.h * self.vary("clere_h", 0.08), 1.0, 2.2)
        else:
            aisle_slope = None
            y_aisle_eave = y_gal_top
            y_aisle_top = y_gal_top
            y_nave_eave = y_gal_top + clamp(0.13 * self.h * self.vary("nave_h", 0.08), 1.6, 3.0)

        # Keep nave ridge high enough for a steep roof (>= 48 deg) and room for the takrytter.
        turret_share = clamp(0.28 * self.vary("turret_share", 0.06), 0.22, 0.34)
        target_ridge = self.h * (1.0 - turret_share)
        nave_run = nave_hx + 0.42
        nave_slope = clamp((target_ridge - y_nave_eave) / max(nave_run, 0.5), 1.05, 2.05)
        y_nave_ridge = y_nave_eave + nave_run * nave_slope
        if y_nave_ridge > self.h - 2.4:
            # Compress lower tiers gracefully on squat requests so turret still crowns cleanly.
            excess = y_nave_ridge - (self.h - 2.4)
            y_nave_eave = max(y_aisle_top + 0.5, y_nave_eave - excess * 0.65)
            nave_slope = max(0.95, (self.h - 2.4 - y_nave_eave) / max(nave_run, 0.5))
            y_nave_ridge = y_nave_eave + nave_run * nave_slope

        # 1. Stone foundation sill (svill) + core + paving under body, choir, and apse.
        self.shell("stone_sill", -sval_hx, sval_hx, z_sval0, z_choir0, 0.0, y_sill, 0.55)
        self.core(-sval_hx + 0.55, sval_hx - 0.55, 0.0, y_sill - 0.18, z_sval0 + 0.55, z_choir0 - 0.55)
        self.slab("podium_paving", -sval_hx + 0.55, sval_hx - 0.55, z_sval0 + 0.55, z_choir0 - 0.55,
                  y_sill - 0.18, y_sill, 1.4)
        self.shell("stone_sill", -choir_hx, choir_hx, z_choir0 - 0.2, z_apse, 0.0, y_sill, 0.50)
        self.core(-choir_hx + 0.50, choir_hx - 0.50, 0.0, y_sill - 0.18, z_choir0 + 0.2, z_apse - 0.2)
        self.slab("podium_paving", -choir_hx + 0.50, choir_hx - 0.50, z_choir0, z_apse,
                  y_sill - 0.18, y_sill, 1.2)
        self.ring("stone_sill", 0.0, z_apse, apse_r + 0.12, 0.45, 0.0, y_sill, arc=(0.0, math.pi))
        self.lathe("podium_paving", 0.0, z_apse, [(0.0, apse_r - 0.28), (y_sill, apse_r - 0.28)], sides=12)

        # Entrance steps at the west portal laid as discrete granite blocks.
        for j in range(steps):
            sz0 = z_sval0 - (steps - j) * 0.28
            sz1 = z_sval0 - (steps - j - 1) * 0.28
            sy1 = (j + 1) * y_sill / steps
            for sx0, sx1 in ((-1.15, -0.38), (-0.38, 0.38), (0.38, 1.15)):
                self.member("step", sx0, sx1, 0.0, sy1, sz0, sz1, construction="granite step")

        # 2. Heavy timber ground sill beams (grunnstokker) resting on the stone sill.
        self._sill_ring(-sval_hx + 0.1, sval_hx - 0.1, z_sval0 + 0.1, z_choir0 - 0.1, y_sill, y_floor, 0.26)
        self._sill_ring(-nave_hx, nave_hx, nave_z0, nave_z1, y_sill, y_floor, 0.30)
        if borgund:
            self._sill_ring(-aisle_hx, aisle_hx, aisle_z0, aisle_z1, y_sill, y_floor, 0.28)
        self._sill_ring(-choir_hx, choir_hx, z_choir0, z_apse, y_sill, y_floor, 0.26)

        # 3. Encircling svalgang gallery + lean-to roof.
        bays_x = max(3, int(round(2 * sval_hx / 2.2)))
        bays_z = max(3, int(round((z_choir0 - z_sval0) / 2.2)))
        self.svalgang_arcade(-sval_hx, sval_hx, z_sval0, z_choir0, y_floor, y_gal_eave, bays_x, bays_z)
        gal_roof_slope = self.lean_to_roofs(
            "gallery_roof", sval_hx, aisle_hx, z_sval0, z_choir0, aisle_z0,
            y_gal_eave, y_gal_top, thick=0.22, over=over_gal, wrap_west=True, tier="gallery")

        # 4. Lower aisle walls (vertical stave planks) & lean-to roof (Borgund raised-nave type).
        aisle_roof_slope = None
        if borgund:
            wt = 0.24
            self.plank_run("aisle_plank", "z", aisle_z0, aisle_z1, -aisle_hx + wt / 2, wt, y_floor, y_aisle_eave,
                           construction="vertical stave plank", face="left")
            self.plank_run("aisle_plank", "z", aisle_z0, aisle_z1, aisle_hx - wt / 2, wt, y_floor, y_aisle_eave,
                           construction="vertical stave plank", face="right")
            self.plank_run("aisle_plank", "x", -aisle_hx, aisle_hx, aisle_z0 + wt / 2, wt, y_floor, y_aisle_eave,
                           construction="vertical stave plank", face="west")
            self.plank_run("aisle_plank", "x", -aisle_hx, aisle_hx, aisle_z1 - wt / 2, wt, y_floor, y_aisle_eave,
                           construction="vertical stave plank", face="east")
            # Aisle corner staves & wall plates.
            for sx in (-1, 1):
                for sz in (aisle_z0 + 0.12, aisle_z1 - 0.12):
                    self.stave_cylinder("aisle_stave", sx * (aisle_hx - 0.12), sz, y_floor, y_aisle_eave, 0.18)
                self.member("head_plate", sx * (aisle_hx - 0.14) - 0.14, sx * (aisle_hx - 0.14) + 0.14,
                            y_aisle_eave - 0.22, y_aisle_eave + 0.06, aisle_z0, aisle_z1,
                            construction="aisle wall plate")
            aisle_roof_slope = self.lean_to_roofs(
                "aisle_roof", aisle_hx, nave_hx, aisle_z0, aisle_z1, nave_z0,
                y_aisle_eave, y_aisle_top, thick=0.24, over=0.30, wrap_west=True, tier="aisle")

        # 5. Tall central nave: staves, clerestory walls, wall plates, scissor braces & steep gable roof.
        stave_r = clamp(0.032 * self.w, 0.18, 0.30)
        for j in range(bays + 1):
            pz = nave_z0 + (nave_z1 - nave_z0) * j / bays
            for sx in (-1, 1):
                self.stave_cylinder("nave_stave", sx * (nave_hx - stave_r * 0.6), pz, y_floor, y_nave_eave, stave_r,
                                    bay=j, side="right" if sx > 0 else "left")
            # Scissor cross-braces (St. Andrew's crosses) tying opposite nave staves at upper clerestory.
            if 0 < j < bays:
                self._scissor_brace(nave_hx - stave_r, pz, y_nave_eave - 1.4, y_nave_eave - 0.15)

        wt = 0.24
        y_wall_bot = y_aisle_top - 0.25 if borgund else y_floor
        self.plank_run("nave_plank", "z", nave_z0, nave_z1, -nave_hx + wt / 2, wt, y_wall_bot, y_nave_eave,
                       construction="vertical stave clerestory plank", face="left")
        self.plank_run("nave_plank", "z", nave_z0, nave_z1, nave_hx - wt / 2, wt, y_wall_bot, y_nave_eave,
                       construction="vertical stave clerestory plank", face="right")
        self.plank_run("nave_plank", "x", -nave_hx, nave_hx, nave_z0 + wt / 2, wt, y_wall_bot, y_nave_eave,
                       construction="vertical stave clerestory plank", face="west")
        self.plank_run("nave_plank", "x", -nave_hx, nave_hx, nave_z1 - wt / 2, wt, y_wall_bot, y_nave_eave,
                       construction="vertical stave clerestory plank", face="east")
        for sx in (-1, 1):
            self.member("head_plate", sx * (nave_hx - 0.14) - 0.16, sx * (nave_hx - 0.14) + 0.16,
                        y_nave_eave - 0.26, y_nave_eave + 0.06, nave_z0 - 0.15, nave_z1 + 0.15,
                        construction="nave wall plate")
        # Inner west doorway into the nave.
        self.member("door", -0.72, 0.72, y_floor, min(y_floor + 2.35, y_gal_eave - 0.1),
                    aisle_z0 - 0.04, aisle_z0 + wt + 0.04, construction="iron-bound timber door")

        nave_roof_slope = self.gable_roof(
            "nave_roof", nave_hx, nave_z0, nave_z1, y_nave_eave, y_nave_ridge,
            thick=0.26, over_x=0.42, over_z=0.35, barge=True, crest=True, dragons=True, tier="nave")

        # 6. Projecting eastern choir & semicircular apse with conical turret.
        y_choir_eave = min(y_nave_eave - 1.1, y_aisle_eave + 1.2) if borgund else y_nave_eave - 1.0
        y_choir_eave = max(y_floor + 2.4, y_choir_eave)
        choir_run = choir_hx + 0.34
        choir_slope = clamp(nave_slope * 1.02, 1.05, 1.95)
        y_choir_ridge = min(y_nave_ridge - 1.0, y_choir_eave + choir_run * choir_slope)
        choir_slope = (y_choir_ridge - y_choir_eave) / max(choir_run, 0.3)

        self.plank_run("choir_plank", "z", z_choir0, z_apse, -choir_hx + wt / 2, wt, y_floor, y_choir_eave,
                       construction="vertical stave plank", face="left")
        self.plank_run("choir_plank", "z", z_choir0, z_apse, choir_hx - wt / 2, wt, y_floor, y_choir_eave,
                       construction="vertical stave plank", face="right")
        self.plank_run("choir_plank", "x", -choir_hx, choir_hx, z_apse - wt / 2, wt, y_floor, y_choir_eave,
                       construction="vertical stave plank", face="east")
        for sx in (-1, 1):
            for sz in (z_choir0 + 0.14, z_apse - 0.14):
                self.stave_cylinder("choir_stave", sx * (choir_hx - 0.14), sz, y_floor, y_choir_eave, 0.17)
        self.gable_roof(
            "choir_roof", choir_hx, z_choir0 - 0.12, z_apse, y_choir_eave, y_choir_ridge,
            thick=0.22, over_x=0.34, over_z=0.26, barge=True, crest=True, dragons=False, tier="choir")
        # Single east dragon finial on the choir gable peak.
        self.dragon_head(z_apse + 0.26, y_choir_ridge + 0.10, 1, scale=0.72, tier="choir")

        # Semicircular apse & conical shingled cap + miniature apse turret.
        y_apse_eave = max(y_floor + 2.0, y_choir_eave - 0.85)
        y_apse_peak = min(y_choir_ridge - 0.55, y_apse_eave + (apse_r + 0.28) * 1.18)
        # Vertical stave planks around the semicircular apse arc.
        apse_segs = 10
        for i in range(apse_segs):
            a0, a1 = math.pi * i / apse_segs, math.pi * (i + 1) / apse_segs
            pts = []
            for y in (y_floor, y_apse_eave):
                for a in (a0, a1):
                    pts.append((apse_r * math.cos(a), y, z_apse + apse_r * math.sin(a)))
                    pts.append(((apse_r - 0.24) * math.cos(a), y, z_apse + (apse_r - 0.24) * math.sin(a)))
            self.mesh("apse_plank", hull(pts), construction="vertical stave apse", segment=i)
        self.frustum("apse_roof", 0.0, z_apse, y_apse_eave - 0.06, y_apse_peak,
                     apse_r + 0.30, 0.12, sides=12, rise=1.6, construction="tarred wood shingles", roof_tier="apse")
        # Small conical turret crowning the apse peak.
        apse_tur_h = clamp(0.11 * self.h, 1.2, 2.2)
        self.drum("turret_panel", 0.0, z_apse, y_apse_peak - 0.25, y_apse_peak + apse_tur_h * 0.42,
                  0.42, 0.38, sides=8, construction="stave apse turret")
        self.frustum("apse_roof", 0.0, z_apse, y_apse_peak + apse_tur_h * 0.38, y_apse_peak + apse_tur_h,
                     0.56, 0.05, sides=8, rise=1.6, construction="tarred wood shingles", roof_tier="apse_turret")

        # 7. Multi-tiered ridge turret (takrytter) and spire crowning the nave ridge.
        z_turret = (nave_z0 + nave_z1) / 2.0
        self.ridge_turret(0.0, z_turret, y_nave_ridge, self.h, turret_levels,
                          base_half=clamp(0.42 * nave_hx, 0.72, 1.35))

        return self.document({
            "typology": "borgund raised-nave stave church" if borgund else "single-nave stave church",
            "bays": bays, "turret_levels": turret_levels,
            "sill_height": round(y_sill, 3),
            "roof_slopes": {k: round(v, 3) for k, v in (
                ("gallery", gal_roof_slope),
                *( (("aisle", aisle_roof_slope),) if aisle_roof_slope is not None else () ),
                ("nave", nave_roof_slope),
                ("choir", choir_slope),
            )},
            "nave_ridge": round(y_nave_ridge, 3),
            "spire_tip": round(self.h, 3),
        })

    def ridge_turret(self, cx, cz, y_ridge, y_top, levels, base_half):
        """Diminishing multi-tiered takrytter with skirt roofs, tall spire, and iron finial."""
        total = max(2.2, y_top - y_ridge)
        finial_h = clamp(0.16 * total, 0.45, 1.1)
        spire_h = clamp(0.36 * total, 1.0, 2.8)
        tiers_h = max(0.8, total - spire_h - finial_h)

        # Saddle embedded into the nave roof so the turret connects solidly to the ridge.
        y_cur = y_ridge - 0.55
        self.member("turret_panel", cx - base_half, cx + base_half, y_cur, y_ridge + 0.15,
                    cz - base_half, cz + base_half, construction="takrytter saddle", turret_tier=0)
        y_cur = y_ridge + 0.10

        half = base_half
        for k in range(levels):
            th = tiers_h * (0.44 if k == 0 else (0.56 / max(1, levels - 1)))
            y_next = y_cur + th
            # Corner turret posts + louvered walls.
            for sx in (-1, 1):
                for sz in (-1, 1):
                    self.stave_cylinder("turret_post", cx + sx * (half - 0.08), cz + sz * (half - 0.08),
                                        y_cur, y_next, 0.085, turret_tier=k + 1)
            wt = 0.14
            self.member("turret_panel", cx - half, cx + half, y_cur, y_next, cz - half, cz - half + wt,
                        construction="takrytter timber shaft", turret_tier=k + 1)
            self.member("turret_panel", cx - half, cx + half, y_cur, y_next, cz + half - wt, cz + half,
                        construction="takrytter timber shaft", turret_tier=k + 1)
            self.member("turret_panel", cx - half, cx - half + wt, y_cur, y_next, cz - half, cz + half,
                        construction="takrytter timber shaft", turret_tier=k + 1)
            self.member("turret_panel", cx + half - wt, cx + half, y_cur, y_next, cz - half, cz + half,
                        construction="takrytter timber shaft", turret_tier=k + 1)
            # Steep pyramidal skirt roof capping each tier.
            next_half = half * 0.72
            skirt_h = clamp(th * 0.55, 0.35, 0.95)
            self.frustum("turret_roof", cx, cz, y_next - 0.08, y_next + skirt_h,
                         half * 1.48, next_half * 1.38, sides=4, rot=math.pi / 4, rise=1.6,
                         construction="tarred wood shingles", turret_tier=k + 1)
            y_cur = y_next + skirt_h * 0.55
            half = next_half

        # Tall slender shingled spire.
        y_spire_top = y_top - finial_h
        self.frustum("spire", cx, cz, y_cur - 0.08, y_spire_top,
                     half * 1.42, 0.06, sides=8, rise=2.0, construction="tarred wood shingles", roof_tier="spire")
        # Wrought-iron cross / weathervane finial reaching exactly y_top.
        self.member("spire_finial", cx - 0.045, cx + 0.045, y_spire_top - 0.12, y_top, cz - 0.045, cz + 0.045,
                    construction="wrought-iron spire cross")
        arm_y = y_top - finial_h * 0.35
        self.member("spire_finial", cx - finial_h * 0.34, cx + finial_h * 0.34, arm_y - 0.04, arm_y + 0.04,
                    cz - 0.04, cz + 0.04, construction="wrought-iron cross arm")

    def _sill_ring(self, x0, x1, z0, z1, y0, y1, width):
        hw = width / 2
        self.member("sill_beam", x0 - hw, x1 + hw, y0, y1, z0 - hw, z0 + hw, construction="oak ground sill")
        self.member("sill_beam", x0 - hw, x1 + hw, y0, y1, z1 - hw, z1 + hw, construction="oak ground sill")
        self.member("sill_beam", x0 - hw, x0 + hw, y0, y1, z0 - hw, z1 + hw, construction="oak ground sill")
        self.member("sill_beam", x1 - hw, x1 + hw, y0, y1, z0 - hw, z1 + hw, construction="oak ground sill")

    def _scissor_brace(self, hx, z, y0, y1):
        """Diagonal St. Andrew's cross braces tying opposite nave staves, lapped side by side."""
        t = 0.06
        for sign in (-1, 1):
            zc = z + sign * 0.08
            pts = [
                (-hx, y0 if sign > 0 else y1, zc - t), (-hx, (y0 if sign > 0 else y1) + 0.14, zc - t),
                (hx, y1 if sign > 0 else y0, zc - t), (hx, (y1 if sign > 0 else y0) + 0.14, zc - t),
                (-hx, y0 if sign > 0 else y1, zc + t), (-hx, (y0 if sign > 0 else y1) + 0.14, zc + t),
                (hx, y1 if sign > 0 else y0, zc + t), (hx, (y1 if sign > 0 else y0) + 0.14, zc + t),
            ]
            self.mesh("scissor_brace", hull(pts), construction="oak scissor brace")

    # ------------------------------------------------------------- stave loft
    def compose_loft(self):
        self.ratio(self.h / self.w, "height/width", 0.75, 1.8)
        self.ratio(self.d / self.w, "depth/width", 0.8, 1.8)
        bays = self.control("bays", *self.form.rules["bays"])
        stabbur = self.variant == "stabbur"

        # Outer envelope fits requested W, H, D cleanly (including dragon finials and stair).
        over_x, over_z = 0.45, 0.36
        upper_hx = self.w / 2 - over_x
        finial_scale = clamp(upper_hx / 2.6, 0.75, 1.35)
        finial_reach = 0.85 * finial_scale
        finial_rise = 0.92 * finial_scale
        upper_hz = max(2.2, self.d / 2 - over_z - finial_reach)
        cantilever = clamp(0.12 * self.w * self.vary("overhang", 0.08), 0.75, min(1.5, upper_hx * 0.34))
        base_hx = upper_hx - cantilever
        base_z0 = -upper_hz + cantilever * 1.15
        base_z1 = upper_hz - (cantilever if stabbur else 0.0)

        y_pad = clamp(0.045 * self.h, 0.42, 0.72) if stabbur else clamp(0.065 * self.h * self.vary("plinth", 0.06), 0.55, 0.95)
        y_sill = y_pad + (clamp(0.065 * self.h, 0.55, 0.90) if stabbur else 0.0)
        sill_h = 0.28
        y_floor1 = y_sill + sill_h
        h1 = clamp(0.22 * self.h * self.vary("first_h", 0.06), 2.0, 3.0)
        y_floor2 = y_floor1 + h1
        h2 = clamp(0.21 * self.h * self.vary("second_h", 0.06), 1.9, 2.9)
        y_eave = y_floor2 + h2

        # Steep gable roof + dragon finials land cleanly at self.h.
        y_ridge = self.h - finial_rise - 0.12
        run_x = upper_hx + over_x
        if (y_ridge - y_eave) / max(run_x, 0.4) < 0.98:
            needed = run_x * 1.02
            y_eave = max(y_floor1 + 3.2, y_ridge - needed)
            y_floor2 = y_floor1 + (y_eave - y_floor1) * 0.52

        # 1. Coursed granite plinth (with core) + stabbe piers if stabbur.
        self.shell("stone_sill", -base_hx - 0.25, base_hx + 0.25, base_z0 - 0.25, base_z1 + 0.25, 0.0, y_pad, 0.50)
        self.core(-base_hx + 0.25, base_hx - 0.25, 0.0, y_pad - 0.16, base_z0 + 0.25, base_z1 - 0.25)
        self.slab("podium_paving", -base_hx + 0.25, base_hx - 0.25, base_z0 + 0.25, base_z1 - 0.25,
                  y_pad - 0.16, y_pad, 1.3)
        if stabbur:
            # Six stout mushroom-capped granite stabbe piers raising the storehouse clear of the ground.
            for sx in (-base_hx + 0.28, 0.0, base_hx - 0.28):
                for sz in (base_z0 + 0.28, base_z1 - 0.28):
                    self.drum("footing_pier", sx, sz, y_pad - 0.02, y_sill - 0.16, 0.24, 0.19, sides=10,
                              construction="ashlar stack", unit="drum")
                    self.drum("footing_pier", sx, sz, y_sill - 0.18, y_sill + 0.02, 0.38, 0.34, sides=10,
                              construction="ashlar stack", unit="drum")

        # 2. Ground-floor timber sills, corner staves, vertical plank walls, and portal.
        self._sill_ring(-base_hx, base_hx, base_z0, base_z1, y_sill, y_floor1, 0.28)
        wt = 0.24
        self.plank_run("nave_plank", "z", base_z0, base_z1, -base_hx + wt / 2, wt, y_floor1, y_floor2,
                       construction="vertical stave plank", face="left")
        self.plank_run("nave_plank", "z", base_z0, base_z1, base_hx - wt / 2, wt, y_floor1, y_floor2,
                       construction="vertical stave plank", face="right")
        self.plank_run("nave_plank", "x", -base_hx, base_hx, base_z1 - wt / 2, wt, y_floor1, y_floor2,
                       construction="vertical stave plank", face="back")
        # Front wall with ground doorway.
        door_hw = 0.65
        self.plank_run("nave_plank", "x", -base_hx, -door_hw, base_z0 + wt / 2, wt, y_floor1, y_floor2,
                       construction="vertical stave plank", face="front")
        self.plank_run("nave_plank", "x", door_hw, base_hx, base_z0 + wt / 2, wt, y_floor1, y_floor2,
                       construction="vertical stave plank", face="front")
        self.member("door", -door_hw, door_hw, y_floor1, min(y_floor1 + 2.1, y_floor2 - 0.25),
                    base_z0 + 0.02, base_z0 + wt - 0.02, construction="iron-bound timber door")
        self.member("portal_frame", -door_hw - 0.14, door_hw + 0.14,
                    min(y_floor1 + 2.05, y_floor2 - 0.30), y_floor2, base_z0 - 0.05, base_z0 + wt + 0.05,
                    construction="carved oak portal lintel")
        for sx in (-1, 1):
            for sz in (base_z0 + 0.12, (base_z0 + base_z1) / 2, base_z1 - 0.12):
                self.stave_cylinder("nave_stave", sx * (base_hx - 0.12), sz, y_floor1, y_floor2, 0.19)

        # 3. Cantilevered floor beams (bjelker) & upper gallery floor deck.
        beam_h = 0.26
        n_beams_z = max(4, bays + 2)
        for k in range(n_beams_z):
            bz = base_z0 + (base_z1 - base_z0) * k / (n_beams_z - 1)
            self.member("cantilever_beam", -upper_hx, upper_hx, y_floor2 - beam_h, y_floor2,
                        bz - 0.11, bz + 0.11, construction="cantilevered oak floor beam")
        for k in range(bays + 1):
            bx = -base_hx + (2 * base_hx) * k / bays
            self.member("cantilever_beam", bx - 0.11, bx + 0.11, y_floor2 - beam_h, y_floor2,
                        -upper_hz, upper_hz, construction="cantilevered oak floor beam")
        self.slab("gallery_floor", -upper_hx, upper_hx, -upper_hz, upper_hz, y_floor2, y_floor2 + 0.16, 1.4,
                  construction="oak gallery decking")

        # 4. Upper storey chamber + cantilevered front/side svalgang arcade gallery.
        y_ufloor = y_floor2 + 0.16
        # Inner upper chamber walls (set back behind the front gallery).
        cham_hx = base_hx
        cham_z0 = base_z0
        cham_z1 = upper_hz - (0.0 if not stabbur else cantilever * 0.45)
        self.plank_run("upper_plank", "z", cham_z0, cham_z1, -cham_hx + wt / 2, wt, y_ufloor, y_eave,
                       construction="upper stave plank", face="left")
        self.plank_run("upper_plank", "z", cham_z0, cham_z1, cham_hx - wt / 2, wt, y_ufloor, y_eave,
                       construction="upper stave plank", face="right")
        self.plank_run("upper_plank", "x", -cham_hx, cham_hx, cham_z0 + wt / 2, wt, y_ufloor, y_eave,
                       construction="upper stave plank", face="front")
        self.plank_run("upper_plank", "x", -cham_hx, cham_hx, cham_z1 - wt / 2, wt, y_ufloor, y_eave,
                       construction="upper stave plank", face="back")

        # Outer cantilevered svalgang gallery arcade around front and sides.
        gal_bays_z = max(3, int(round((upper_hz * 2) / 2.2)))
        self.svalgang_arcade(-upper_hx, upper_hx, -upper_hz, upper_hz, y_ufloor, y_eave,
                             bays_x=bays, bays_z=gal_bays_z, open_west=False)
        # Back wall of upper gallery.
        self.plank_run("gallery_panel", "x", -upper_hx, upper_hx, upper_hz - wt / 2, wt, y_ufloor, y_eave,
                       construction="stave gallery plank", face="back")

        # 5. External timber stair climbing to the cantilevered upper gallery deck at the front.
        stair_x0 = -upper_hx + 0.25
        stair_x1 = min(stair_x0 + 1.25, -door_hw - 0.25)
        stair_z_top = -upper_hz + 0.18
        stair_z_bot = max(-self.d / 2 + 0.14, stair_z_top - clamp(y_ufloor * 0.68, 1.1, 2.2))
        # Keep stair within the requested depth envelope by running along the front-left side!
        # Run stair along +Z -> -Z on the front-left apron under the overhang if tight.
        stair_steps = max(7, int(math.ceil(y_ufloor / 0.24)))
        for s in range(stair_steps):
            u0 = s / stair_steps
            u1 = (s + 1) / stair_steps
            sy0 = y_ufloor * u0
            sy1 = y_ufloor * u1
            sz0 = stair_z_bot + (stair_z_top - stair_z_bot) * u0
            sz1 = stair_z_bot + (stair_z_top - stair_z_bot) * u1
            self.member("stair_step", stair_x0, stair_x1, sy0, sy1 + 0.06, min(sz0, sz1) - 0.12, max(sz0, sz1) + 0.12,
                        construction="oak stair tread", step=s + 1)
        # Diagonal oak stair stringers tying ground to gallery deck.
        for sx in (stair_x0 + 0.06, stair_x1 - 0.06):
            pts = [
                (sx - 0.06, 0.0, stair_z_bot - 0.12), (sx + 0.06, 0.0, stair_z_bot - 0.12),
                (sx - 0.06, 0.26, stair_z_bot - 0.12), (sx + 0.06, 0.26, stair_z_bot - 0.12),
                (sx - 0.06, y_ufloor - 0.24, stair_z_top + 0.12), (sx + 0.06, y_ufloor - 0.24, stair_z_top + 0.12),
                (sx - 0.06, y_ufloor + 0.04, stair_z_top + 0.12), (sx + 0.06, y_ufloor + 0.04, stair_z_top + 0.12),
            ]
            self.mesh("stair_stringer", hull(pts), construction="oak stair stringer")

        # 6. Steep shingled gable roof with bargeboards, ridge crest and dragon finials.
        roof_slope = self.gable_roof(
            "nave_roof", upper_hx, -upper_hz, upper_hz, y_eave, y_ridge,
            thick=0.26, over_x=over_x, over_z=over_z, barge=True, crest=True, dragons=True, tier="loft")

        return self.document({
            "typology": "stabbur raised storehouse" if stabbur else "medieval Norwegian loft-house",
            "bays": bays, "sill_height": round(y_sill, 3),
            "cantilever": round(cantilever, 3),
            "roof_slopes": {"loft": round(roof_slope, 3)},
            "upper_floor": round(y_ufloor, 3),
        })


# ----------------------------------------------------------------- rule checks
def _by_role(placements, role):
    return [p for p in placements if p.role == role]


def rule_checks(document, form):
    """Measured geometric rules for Nordic stave architecture, independent of metadata."""
    placements = document.placements
    checks = []

    # 1. Stone sill grounds the timber staves clear of bare earth.
    sills = _by_role(placements, "stone_sill")
    staves = [p for p in placements if p.role in {"nave_stave", "aisle_stave", "choir_stave"}]
    if sills and staves:
        sill_lo, sill_hi = extent(sills)
        stave_lo, _ = extent(staves)
        sill_grounded = sill_lo[1] <= 0.05 and sill_hi[1] >= 0.35
        staves_above = [p.id for p in staves if (p.position[1] - p.dimensions[1] / 2) < sill_hi[1] - 0.05]
        checks.append(("stone_sill_grounds_timber", sill_grounded and not staves_above,
                       "Every timber stave stands on the stone foundation sill, clear of the ground.",
                       "A timber stave drops below the stone foundation sill or the sill is ungrounded.",
                       staves_above))
    else:
        checks.append(("stone_sill_grounds_timber", False,
                       "Every timber stave stands on the stone foundation sill, clear of the ground.",
                       "Missing stone sill or vertical timber staves.", []))

    # 2. Steep Nordic roofs (rise/run >= 0.72 measured from each roof slab's span).
    roofs = [p for p in placements if p.role in {"nave_roof", "aisle_roof", "gallery_roof", "choir_roof"}]
    shallow = []
    for p in roofs:
        # West lean-to roofs slope along Z; side and gable roofs slope along X.
        dx, dy, dz = p.dimensions
        run = dz if p.parameters.get("side") == "west" else dx
        if run > 0.4 and (dy - 0.30) / run < 0.72:
            shallow.append(p.id)
    checks.append(("steep_gable_roofs", bool(roofs) and not shallow,
                   "All shingled roof slopes hold a steep Nordic snow-shedding pitch.",
                   "One or more roof slopes are too shallow for a Nordic stave roof.", shallow))

    if form.name == "stave-church":
        # 3. Tiered silhouette: nave roof rises above gallery (and aisle in borgund).
        nave_roofs = _by_role(placements, "nave_roof")
        gal_roofs = _by_role(placements, "gallery_roof")
        aisle_roofs = _by_role(placements, "aisle_roof")
        tiered_ok = False
        if nave_roofs and gal_roofs:
            n_lo, n_hi = extent(nave_roofs)
            g_lo, g_hi = extent(gal_roofs)
            tiered_ok = n_hi[1] > g_hi[1] + 1.5 and (n_hi[0] - n_lo[0]) < (g_hi[0] - g_lo[0]) - 0.8
            if aisle_roofs:
                a_lo, a_hi = extent(aisle_roofs)
                tiered_ok = tiered_ok and (n_hi[1] > a_hi[1] + 0.8 > g_hi[1] + 0.8) and \
                    (n_hi[0] - n_lo[0]) < (a_hi[0] - a_lo[0]) - 0.5
        checks.append(("tiered_silhouette", tiered_ok,
                       "The raised nave roof tiers cleanly above the lower gallery and aisle roofs.",
                       "The stave church lacks a properly tiered silhouette.", []))

        # 4. Ridge turret (takrytter) and spire crown the nave as the highest members.
        finials = _by_role(placements, "spire_finial")
        turrets = _by_role(placements, "turret_roof")
        all_lo, all_hi = extent(placements)
        crown_ok = False
        if finials and turrets and nave_roofs:
            f_lo, f_hi = extent(finials)
            n_lo, n_hi = extent(nave_roofs)
            crown_ok = (f_hi[1] >= all_hi[1] - 0.05) and (f_hi[1] > n_hi[1] + 1.5) and \
                       (abs((f_lo[0] + f_hi[0]) / 2) <= 0.35)
        checks.append(("ridge_turret_crowns_nave", crown_ok,
                       "The takrytter ridge turret and iron finial stand centered and crown the church.",
                       "The ridge turret or finial does not crown the nave.", [p.id for p in finials]))

        # 5. Paired dragon-head finials on the nave gable ends, mirrored across X=0.
        nave_dragons = [p for p in _by_role(placements, "dragon_finial")
                        if p.parameters.get("roof_tier") == "nave"]
        all_dragons = _by_role(placements, "dragon_finial")
        bad_dragons = unmirrored(placements, {"dragon_finial"}, axis=0, centre=0.0)
        z_ends = {p.parameters.get("end") for p in nave_dragons} if nave_dragons else set()
        dragons_ok = len(all_dragons) >= 4 and not bad_dragons and (
            {"west", "east"} <= z_ends or len(all_dragons) >= 4
        )
        # Also verify purely geometrically that finials exist near both ends of the nave roof.
        if nave_roofs and all_dragons:
            n_lo, n_hi = extent(nave_roofs)
            has_west = any(p.position[2] < n_lo[2] + 0.5 for p in all_dragons)
            has_east = any(p.position[2] > n_hi[2] - 0.5 for p in all_dragons)
            dragons_ok = dragons_ok and has_west and has_east
        checks.append(("paired_dragon_finials", dragons_ok,
                       "Dragon-head finials crown the gable ends symmetrically across the axis.",
                       "Dragon-head finials are missing or asymmetric.", bad_dragons))

        # 6. Choir and apse project past the east (+Z) end of the nave, narrower and stepped down.
        choir = _by_role(placements, "choir_plank")
        apse = _by_role(placements, "apse_plank")
        nave = _by_role(placements, "nave_plank")
        apse_ok = False
        if choir and apse and nave:
            c_lo, c_hi = extent(choir)
            a_lo, a_hi = extent(apse)
            n_lo, n_hi = extent(nave)
            apse_ok = (a_hi[2] > c_hi[2] - 0.1 > n_hi[2] + 0.5) and \
                      ((c_hi[0] - c_lo[0]) < (n_hi[0] - n_lo[0]) + 0.1 or (a_hi[0] - a_lo[0]) < (n_hi[0] - n_lo[0]))
        checks.append(("apse_at_east", apse_ok,
                       "The choir and semicircular apse project cleanly east of the nave.",
                       "The choir and apse do not project east of the nave.", []))
    else:
        # 3 (loft). Upper storey / svalgang cantilevers outward over ground floor, with external stair.
        base_walls = _by_role(placements, "nave_plank")
        gal_walls = _by_role(placements, "gallery_panel")
        stairs = _by_role(placements, "stair_step")
        overhang_ok = False
        if base_walls and gal_walls and stairs:
            b_lo, b_hi = extent(base_walls)
            g_lo, g_hi = extent(gal_walls)
            s_lo, s_hi = extent(stairs)
            overhang_ok = ((g_hi[0] - g_lo[0]) > (b_hi[0] - b_lo[0]) + 0.8) and \
                          (g_lo[2] < b_lo[2] - 0.4) and (s_hi[1] >= g_lo[1] - 0.25) and (s_lo[1] <= 0.35)
        checks.append(("overhanging_upper_storey", overhang_ok,
                       "The upper svalgang gallery cantilevers over the ground floor and is reached by the external stair.",
                       "The upper gallery does not cantilever over the ground floor or lacks stair access.", []))

    return checks


def random_controls(product, rng):
    """Sample valid dimensions and controls for randomized stave builds."""
    form = FORMS[product]
    variant = rng.choice(form.variants)
    if product == "stave-church":
        width = round(rng.uniform(12.0, 18.0), 1)
        height = round(width * rng.uniform(1.15, 1.48), 1)
        depth = round(width * rng.uniform(1.42, 1.72), 1)
        return {"variant": variant, "width": width, "height": height, "depth": depth,
                "bays": rng.choice((3, 5, 7)), "levels": rng.choice((2, 3))}
    width = round(rng.uniform(8.5, 12.5), 1)
    height = round(width * rng.uniform(0.95, 1.28), 1)
    depth = round(width * rng.uniform(1.05, 1.38), 1)
    return {"variant": variant, "width": width, "height": height, "depth": depth,
            "bays": rng.choice((3, 4, 5))}


StaveBuilder.module = sys.modules[__name__]


def build_document(request):
    return StaveBuilder(request).compose()
