#!/usr/bin/env python3
"""Render the BlockN vertical launch showcase (1080x1920, 30 fps, 92.0 s).

Every frame is a pure deterministic function of the global frame index. Real BlockN builds
(seed 73, detail=concept, age=0.35) are rendered offscreen via PyVista/VTK EGL in true
perspective in two synchronized passes:
  1. Phosphor CAD pass: black occluding faces, crisp role-colored feature edges, hot/warm
     landing flashes as members appear in the architect's real construction order.
  2. Weathered PBR Studio pass: exact per-face weathered sRGB colors from
     core.architecture.surfaces.face_colors(), 3-point architectural lighting, and SSAO.

Emerald laser scan sweeps materialize phosphor wireframes into weathered architectural
materials, an open-roof cutaway exposes the bare softwood timber truss frame, and 3D
world-projected callout pins track real member centroids across moving perspective cameras.

Narration is generated via ElevenLabs (voice Eric, eleven_multilingual_v2) with word-level
timestamps; bespoke score stems and SFX are generated via ElevenLabs Sound Generation and
aligned by measured attack/peak transients.

    python3 scripts/render_blocks_showcase.py --fetch-audio           # request missing audio
    python3 scripts/render_blocks_showcase.py --stills 120 420 820    # render review stills
    python3 scripts/render_blocks_showcase.py                         # full film + verify
    python3 scripts/render_blocks_showcase.py --remix                 # remix audio only
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _var in ("DISPLAY", "WAYLAND_DISPLAY"):
    os.environ.pop(_var, None)

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from render_two_architects_demo import decode, loudness, mux, scan, schedule, sheet, stages

OUT = ROOT / "builds" / "blocks-showcase"
W, H, FPS = 1080, 1920, 30
LAST = 3224                                                     # frames 0..3224 = 107.500 s
FADE_OUT = 3201
SEED = 73
AGE = 0.35
SR = 48000
REPO_URL = "github.com/newjordan/Blocks"
CLI_CMD = "$ blockn make chinese-hall --age 0.35"
MASS_CLI_CMD = "$ blockn agent series --count 15 --seed 73 --products timber-house"

# Visual palette: obsidian studio + BlockN phosphor emerald + warm paper editorial white
C_GREEN = (0, 255, 85)
C_MINT = (150, 255, 184)
C_SAGE = (156, 188, 166)
C_DIM = (82, 128, 98)
C_RULE = (24, 96, 52)
C_WHITE = (242, 246, 243)
C_AMBER = (242, 186, 68)
HOT_HEX, WARM_HEX = "#e6ffee", "#5eff90"
HOT_FRAMES, WARM_FRAMES = 3, 8
LEFT, RIGHT = 64, 1016                                          # 64 px horizontal safe area

FONT_CANDIDATES = {
    "sans_black": ("/usr/share/fonts/TTF/Roboto-Black.ttf", "/usr/share/fonts/TTF/JetBrainsMono-ExtraBold.ttf"),
    "sans_bold": ("/usr/share/fonts/TTF/Roboto-Bold.ttf", "/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf"),
    "sans_med": ("/usr/share/fonts/TTF/Roboto-Medium.ttf", "/usr/share/fonts/TTF/JetBrainsMono-Medium.ttf"),
    "mono_xbold": ("/usr/share/fonts/TTF/JetBrainsMono-ExtraBold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    "mono_bold": ("/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"),
    "mono_med": ("/usr/share/fonts/TTF/JetBrainsMono-Medium.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
}
TEXT_STYLES = {
    "eyebrow": ("mono_bold", 25),
    "title": ("sans_black", 70),
    "sub": ("mono_med", 26),
    "stat": ("mono_xbold", 54),
    "stat_sm": ("mono_bold", 28),
    "line": ("mono_med", 27),
    "rule": ("mono_med", 25),
    "tag": ("mono_xbold", 23),
    "pin": ("mono_bold", 22),
    "pin_sub": ("mono_med", 19),
    "reel_title": ("sans_black", 56),
    "index": ("mono_bold", 26),
    "mark": ("sans_black", 142),
    "cta_head": ("sans_black", 48),
    "cta_sub": ("sans_med", 31),
    "code": ("mono_bold", 30),
    "code_sm": ("mono_bold", 22),
    "repo": ("mono_bold", 34),
    "footer": ("mono_med", 23),
}

# Models built from BlockN's real pipeline
HERO_MODELS = {
    "hall": ("chinese-hall", "hip-and-gable"),
    "palace": ("chinese-palace-hall", "double-hip-roof"),
    "gate": ("dravidian-gopuram", "rajagopuram"),
    "vimana": ("dravidian-vimana", "shala-vimana"),
}
REEL_ITEMS = [
    dict(key="iwan-court", product="iwan-court", variant="auto",
         title="PERSIAN IWAN COURT", architect="PERSIAN ARCHITECT",
         sub_tag="SANDSTONE  ·  GLAZED TURQUOISE TILE",
         cam0=(-34, 15, 0.88, (0.0, 0.02, 0.0), 34), cam1=(18, 26, 0.78, (0.0, -0.02, 0.0), 34)),
    dict(key="victorian-terrace", product="victorian-terrace", variant="auto",
         title="VICTORIAN TERRACE", architect="VICTORIAN ARCHITECT",
         sub_tag="8 SUBSTANCES  ·  BRICK · SLATE · GLASS",
         cam0=(-48, 14, 1.10, (0.0, -0.04, 0.0), 34), cam1=(-14, 22, 0.98, (0.0, -0.04, 0.0), 34)),
    dict(key="gothic-cathedral", product="gothic-cathedral", variant="auto",
         title="GOTHIC CATHEDRAL", architect="CATHEDRAL ARCHITECT",
         sub_tag="LIMESTONE ASHLAR  ·  FLYING BUTTRESSES",
         cam0=(-42, 14, 1.00, (0.0, -0.14, 0.0), 36), cam1=(-12, 24, 0.90, (0.0, -0.14, 0.0), 36)),
    dict(key="norman-castle", product="norman-castle", variant="drum-towers",
         title="NORMAN CASTLE", architect="CASTLE ARCHITECT",
         sub_tag="LIMESTONE + FLINT CORE  ·  DRUM TOWERS",
         cam0=(-46, 26, 1.02, (0.0, -0.16, 0.0), 35), cam1=(-10, 34, 0.92, (0.0, -0.16, 0.0), 35)),
    dict(key="japanese-pagoda", product="japanese-pagoda", variant="five-storey",
         title="JAPANESE PAGODA", architect="PAGODA ARCHITECT",
         sub_tag="HINOKI CEDAR  ·  FREE CENTRAL PILLAR",
         cam0=(-36, 10, 1.02, (0.0, -0.12, 0.0), 34), cam1=(16, 20, 0.92, (0.0, -0.12, 0.0), 34)),
    dict(key="mesoamerican-pyramid", product="mesoamerican-pyramid", variant="maya-temple",
         title="MAYA TEMPLE PYRAMID", architect="MESOAMERICAN ARCHITECT",
         sub_tag="CUT LIMESTONE  ·  PIERCED ROOF COMB",
         cam0=(-44, 18, 0.94, (0.0, 0.02, 0.0), 35), cam1=(10, 28, 0.84, (0.0, 0.04, 0.0), 35)),
    dict(key="english-village", product="english-village", variant="auto",
         title="MEDIEVAL ENGLISH VILLAGE", architect="SETTLEMENT PLANNER",
         sub_tag="10 SUBSTANCES  ·  BURGAGE STREET + CHURCH",
         cam0=(-40, 28, 0.96, (0.0, -0.02, 0.0), 36), cam1=(8, 36, 0.84, (0.0, -0.02, 0.0), 36)),
    dict(key="roman-town", product="roman-town", variant="auto",
         title="ROMAN TOWN QUARTER", architect="SETTLEMENT PLANNER",
         sub_tag="CARDO · DECUMANUS · FORUM · INSULAE",
         cam0=(-36, 30, 0.96, (0.0, -0.02, 0.0), 36), cam1=(14, 38, 0.84, (0.0, -0.02, 0.0), 36)),
]

# Roof roles hidden during Chinese Hall open-roof structural reveal (S03)
CHINESE_ROOF_ROLES = {
    "roof_tile", "main_ridge", "hip_ridge", "surround_ridge",
    "chiwen", "beast", "gable_panel", "bargeboard",
}

# Exact frame-directed scenes (inclusive global frame ranges, 0..3224 = 107.5 s)
REEL_FIRST = 1905
REEL_LENGTHS = [63, 63, 63, 63, 63, 63, 66, 66]                 # sum = 510 -> 1905..2414
MASS_FIRST = 2415
SEED_MORPH_END = 2644                                           # 2415..2644 (7.67 s): 6 live seed mutations
GRID_FIRST = 2645                                               # 2645..2879 (7.83 s): 15-building grid cascade
OUTRO_FIRST = 2880                                              # 2880..3224 (11.50 s): closing lockup & CTA

# Six high-contrast seed mutations from the 15-house randomized batch (indices into batch)
MORPH_STEPS = [
    (2415, 0),    # SEED 73: gable-front 2F 6b
    (2475, 4),    # SEED 77: twin-gable  1F 3b ("Change a seed")
    (2513, 6),    # SEED 79: longhouse   2F 5b ("proportion, roof plan")
    (2547, 5),    # SEED 78: twin-gable  2F 3b ("weathered finish")
    (2581, 8),    # SEED 81: gable-front 2F 5b ("mutates within")
    (2613, 11),   # SEED 84: twin-gable  1F 7b ("tradition's rules")
]
GRID_DROP_START = 2656                                          # 15 buildings drop every 8 frames: 2656..2768
GRID_DROP_STEP = 8

SCENES = [
    dict(key="s01_open", first=0, last=299, model="hall", mode="wire_grow", grow=(4, 258),
         eyebrow="BLOCKN  ·  PROCEDURAL ARCHITECTURE", title="BUILT FROM RULES",
         sub="Not a static mesh — every beam, bracket & stone compiled live"),
    dict(key="s02_hall_pbr", first=300, last=599, model="hall", mode="scan_diag", scan=(378, 468),
         eyebrow="NEW  ·  CHINESE ARCHITECT", title="TIMBER HALL",
         sub="{bays} bays · {bracket_sets} bracket sets · {tonnes} t weathered PBR",
         materials=[(454, "MARBLE TERRACE"), (492, "VERMILION TIMBER"), (546, "GLAZED TILE")]),
    dict(key="s03_hall_macro", first=600, last=899, model="hall", mode="macro_open_roof",
         peel=(802, 834), eyebrow="CHINESE ARCHITECT  ·  PROPORTION + FRAME", title="DOUGONG + TRUSS",
         sub="Qing doukou brackets · flying eaves · softwood truss inside"),
    dict(key="s04_palace", first=900, last=1169, model="palace", mode="pbr_glint", glint=(900, 932),
         eyebrow="NEW  ·  CHINESE ARCHITECT", title="PALACE HALL",
         sub="{bays} bays · double eaves · {bracket_sets} bracket sets · {tonnes} t"),
    dict(key="s05_gate_grow", first=1170, last=1499, model="gate", mode="wire_grow", grow=(1182, 1458),
         eyebrow="NEW  ·  DRAVIDIAN ARCHITECT", title="GATEWAY TOWER",
         sub="{talas} talas · 46 m granite & brick · open passage through base"),
    dict(key="s06_gate_pbr", first=1500, last=1754, model="gate", mode="scan_up", scan=(1504, 1586),
         eyebrow="DRAVIDIAN ARCHITECT  ·  AUDITED GEOMETRY", title="RAJAGOPURAM",
         sub="{talas} diminishing talas · {tonnes} t · max load {util_pct}%",
         rules_in=1636, rules=("dravidian.diminishing_talas", "dravidian.hara_each_tala", "dravidian.passage_open")),
    dict(key="s06_vimana", first=1755, last=1904, model="vimana", mode="pbr_glint", glint=(1755, 1784),
         eyebrow="NEW  ·  DRAVIDIAN ARCHITECT", title="SHALA VIMANA",
         sub="100% solid granite · {talas} talas over sanctum + pillared hall"),
]
_cur = REEL_FIRST
for _idx, (_item, _dur) in enumerate(zip(REEL_ITEMS, REEL_LENGTHS)):
    SCENES.append(dict(
        key=f"s07_reel_{_idx}", first=_cur, last=_cur + _dur - 1,
        model=_item["key"], mode="reel", reel_index=_idx, reel_item=_item,
    ))
    _cur += _dur
SCENES.append(dict(
    key="s08_seed_morph", first=MASS_FIRST, last=SEED_MORPH_END, model="mass_batch", mode="seed_morph",
    eyebrow="PARAMETRIC RANDOMIZATION  ·  SEED VARIATION", title="INFINITE FORMS",
    sub="Same tradition rules · unique roofs, bays & weathered finishes",
))
SCENES.append(dict(
    key="s08_mass_grid", first=GRID_FIRST, last=OUTRO_FIRST - 1, model="mass_batch", mode="mass_grid",
    eyebrow="AGENT BATCH MODE  ·  MASS GENERATION", title="15 UNIQUE BUILDS",
    sub="One deterministic command · every building quality-audited",
))
SCENES.append(dict(key="s09_outro", first=OUTRO_FIRST, last=LAST, model="palace", mode="outro"))

# ElevenLabs voiceover takes & exact synced entry frames
VOICE_ID = "cjVigY5qzO86Huf0OWal"                               # Eric - Smooth, Trustworthy
VOICE_MODEL = "eleven_multilingual_v2"
VOICE_SETTINGS = {"stability": 0.44, "similarity_boost": 0.78, "style": 0.38, "use_speaker_boost": True, "speed": 1.03}
NARRATION = [
    ("v1", 6, None,
     "This isn't a static 3D mesh. Block-N builds real architecture—member by member, straight from historical rules. And this update is massive."),
    ("v2", 308, None,
     "Meet the new Chinese Architect. Seven thousand individual pieces—white marble terrace, vermilion columns, and weathered glazed tile."),
    ("v3", 608, None,
     "Every joint obeys classical proportion: one hundred twenty-one dougong brackets, upturned flying eaves, and a complete timber frame beneath the roof."),
    ("v4", 908, None,
     "Scale up to a nine-bay Palace Hall—double eaves, two hundred forty-two bracket sets, four thousand tonnes of audited geometry."),
    ("v5", 1182, 9.45,                                          # trim tiny trailing breath artifact at 9.75s
     "Now, the Dravidian Architect. Fifteen thousand blocks rising forty-six meters into a tiered gateway tower—with an open granite passage straight through the base."),
    ("v6", 1506, None,
     "Seven diminishing storeys, miniature shrine parapets, gilded finials—every tradition rule verified on the finished mesh. Plus a solid-granite Vimana over its pillared hall."),
    ("v7", 1910, None,
     "Every block now carries real physical mass, weathered PBR surfaces, and gravity load checks—across forty-five products, twenty architects, and full historic settlements. Ready for GLB, OBJ, USD, and physics."),
    ("v_mass", 2421, None,
     "Need a whole neighborhood? Change a seed and every proportion, roof plan, and weathered finish mutates within the tradition's rules. Run one agent command to mass-generate dozens of unique, quality-checked buildings in a single deterministic batch."),
    ("v8", 2901, None,
     "Want a real benchmark for your favorite AI model? Have it add a new architectural tradition to Block-N—and grow the open-source library for everyone."),
]

# Outro word-synced cue frames (v8 starts at 2901; "Have it add" @ +2.91s=2988, "grow the open-source" @ +6.66s=3101)
OUTRO_IN = {
    "mark": OUTRO_FIRST + 6,
    "stats": OUTRO_FIRST + 16,
    "cli": OUTRO_FIRST + 26,
    "ask": 2901,
    "line1": 2988,
    "line2": 3101,
    "repo": 3113,
}

# Export format pills synced to v7's words (v7 starts at 1910: GLB +12.50s=2285, OBJ +13.22s=2306, USD +13.90s=2327, physics +14.69s=2351)
EXPORT_PILLS = [
    (2282, "GLB"), (2304, "OBJ"), (2325, "USD"), (2348, "PHYSICS.JSON"),
]

LANDINGS = {"s01_open": ("wood_1", "wood_2", "wood_3"), "s05_gate_grow": ("stone_1", "stone_2", "stone_3")}


# ------------------------------------------------------------------ math & easing
def clamp01(v: float) -> float:
    return 0.0 if v <= 0.0 else 1.0 if v >= 1.0 else float(v)


def ease(v: float) -> float:
    u = clamp01(v)
    return u * u * (3.0 - 2.0 * u)


def ease_quint(v: float) -> float:
    u = clamp01(v)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def ease_out(v: float) -> float:
    u = 1.0 - clamp01(v)
    return 1.0 - u * u * u


def ramp(frame: int, start: int, length: int = 12) -> float:
    return clamp01((frame - start) / max(1, length))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def scene_at(frame: int) -> dict:
    for s in SCENES:
        if s["first"] <= frame <= s["last"]:
            return s
    return SCENES[-1]


# ------------------------------------------------------------------ BlockN model preparation
def trace_style(block, cathedral: bool):
    role = block.parameters.get("component_role", block.parameters.get("element", "")) or ""
    if cathedral:
        if role == "apse_roof":
            return "#00882a", 5
        if role == "roof_seam":
            return "#006a20", 26
        if role in {"window_tracery", "facade_gallery", "pier_shaft", "finial"} or block.parameters.get("section") == "fillet":
            return "#00982e", 26
        return "#00de46", 26
    if role in {"column_shaft", "shaft", "column", "inner_column"}:
        return "#00d444", 16
    if "capital" in role or "base" in role or "bracket" in role:
        return "#00f55c", 12
    if role in {"dome", "dome_rib", "groin_vault", "apse_roof", "roof_tile"}:
        return "#00a838", 12
    if role in {"roof_seam", "core_fill", "base_core"}:
        return "#006422", 26
    return "#00df4a", 26


def build_shaded_polydata(pv, blocks):
    """Build a single split-normal PolyData carrying exact weathered per-face sRGB colors."""
    from core.architecture.surfaces import face_colors

    pts_l, tri_l, col_l = [], [], []
    off = 0
    for b in blocks:
        v = np.asarray(b.mesh.vertices, dtype=np.float32)
        f = np.asarray(b.mesh.faces, dtype=np.int64)
        fc = face_colors(b.mesh)
        if len(fc) != len(f):
            col = np.array(b.color[:3], dtype=np.uint8)
            fc = np.repeat(col[None, :], len(f), axis=0)
        pts_l.append(v)
        tri_l.append(f + off)
        col_l.append(fc)
        off += len(v)
    pts = np.vstack(pts_l)
    tris = np.vstack(tri_l)
    cols = np.vstack(col_l)
    cells = np.column_stack((np.full(len(tris), 3, dtype=np.int64), tris)).ravel()
    poly = pv.PolyData(pts, cells)
    poly.cell_data["rgb"] = cols
    return poly.compute_normals(cell_normals=True, point_normals=True, split_vertices=True, feature_angle=35.0)


class Model:
    """Real BlockN build prepared for both member-ordered phosphor CAD and weathered PBR."""

    def __init__(self, key: str, product: str, variant: str, need_wire: bool = False, split_roof: bool = False):
        import pyvista as pv
        from core.architecture import BuildRequest, build

        t0 = time.time()
        self.key, self.product, self.variant = key, product, variant
        self.result = build(BuildRequest(product=product, variant=variant, detail="concept", seed=SEED, age=AGE))
        assert self.result.quality.status == "pass", f"{product}/{variant} failed quality gate"
        blocks = [b for b in self.result.scene.blocks if b.mesh is not None and not b.mesh.is_empty]
        self.blocks = blocks
        self.points = np.vstack([b.mesh.vertices for b in blocks]).astype(np.float32)
        self.lo = self.points.min(axis=0).astype(np.float64)
        self.hi = self.points.max(axis=0).astype(np.float64)
        self.center = (self.lo + self.hi) * 0.5
        self.ext = self.hi - self.lo
        self.sample = self.points[:: max(1, len(self.points) // 25000)]

        # Shaded weathered PBR geometry
        self.pbr_poly = build_shaded_polydata(pv, blocks)
        self.pbr_body_poly = None
        self.pbr_roof_poly = None
        if split_roof:
            body_b = [b for b in blocks if b.parameters.get("component_role") not in CHINESE_ROOF_ROLES]
            roof_b = [b for b in blocks if b.parameters.get("component_role") in CHINESE_ROOF_ROLES]
            self.pbr_body_poly = build_shaded_polydata(pv, body_b)
            self.pbr_roof_poly = build_shaded_polydata(pv, roof_b)

        # Phosphor wireframe prefix arrays (only for models that animate or scan in wireframe)
        self.need_wire = need_wire
        self.cells = None
        self.face_member = None
        self.traces = []
        if need_wire:
            cathedral = any(b.parameters.get("architect") == "cathedral" for b in blocks)
            offsets = np.cumsum([0] + [len(b.mesh.vertices) for b in blocks])
            tris = np.vstack([b.mesh.faces + o for b, o in zip(blocks, offsets)])
            self.face_member = np.concatenate([np.full(len(b.mesh.faces), i) for i, b in enumerate(blocks)])
            self.cells = np.column_stack((np.full(len(tris), 3, dtype=np.int64), tris)).ravel()
            groups = {}
            for i, b in enumerate(blocks):
                groups.setdefault(trace_style(b, cathedral), []).append(i)
            for (colour, angle), members in groups.items():
                local = np.cumsum([0] + [len(blocks[i].mesh.vertices) for i in members])
                m_pts = np.vstack([blocks[i].mesh.vertices for i in members]).astype(np.float32)
                m_tris = np.vstack([blocks[i].mesh.faces + o for i, o in zip(members, local)])
                poly = pv.PolyData(m_pts, np.column_stack((np.full(len(m_tris), 3, dtype=np.int64), m_tris)).ravel())
                poly.cell_data["member"] = np.concatenate([np.full(len(blocks[i].mesh.faces), i) for i in members])
                edges = poly.extract_feature_edges(
                    feature_angle=angle, boundary_edges=True, non_manifold_edges=True, manifold_edges=False
                )
                if not edges.n_lines:
                    continue
                lines = edges.lines.reshape(-1, 3)[:, 1:]
                member = np.asarray(edges.cell_data["member"])
                order = np.argsort(member, kind="stable")
                self.traces.append((
                    colour,
                    np.asarray(edges.points, dtype=np.float32),
                    np.column_stack((np.full(len(order), 2, dtype=np.int64), lines[order])).ravel(),
                    member[order],
                ))

        self.metadata = self._metadata()
        self.anchors = self._anchors()
        self.counts = None
        self.stage_bounds = None
        self.stage_labels = None
        print(
            f"built {product}/{variant}: {len(blocks):,} members, {self.metadata['tonnes']} t, "
            f"{self.result.quality.status}, {time.time() - t0:.1f}s",
            flush=True,
        )

    def _metadata(self) -> dict:
        rules = next((v for k, v in self.result.document.metadata.items() if k.endswith("_rules")), {})
        facts = {k: v for k, v in rules.items() if isinstance(v, (int, float, str))}
        facts.update(rules.get("block_count", {}))
        phys = self.result.quality.metrics.get("physical", {})
        tonnes = int(round(float(phys.get("total_mass_kg", 0.0)) / 1000.0))
        util_pct = int(round(float(phys.get("max_member_utilisation", 0.0)) * 100.0))
        subs = len(self.result.quality.metrics.get("substance_counts", {}))
        out = {k: f"{v:,}" if isinstance(v, int) else v for k, v in facts.items()}
        out["tonnes"] = f"{tonnes:,}"
        out["tonnes_raw"] = tonnes
        out["util_pct"] = str(util_pct)
        out["substances_count"] = str(subs)
        return out

    def _anchors(self) -> dict:
        """Real 3D world-space anchor points measured directly from component meshes."""
        by_role: dict[str, list[np.ndarray]] = {}
        for b in self.blocks:
            r = b.parameters.get("component_role", "")
            if r:
                c = b.mesh.bounds.mean(axis=0)
                by_role.setdefault(r, []).append(c)

        def pick(role: str, score_fn):
            pts = by_role.get(role)
            if not pts:
                return self.center.copy()
            return np.asarray(max(pts, key=score_fn), dtype=np.float64)

        anchors = {}
        if self.key == "hall":
            # Front-left dougong bracket set (z < 0 is front, x near -6..0)
            anchors["bracket"] = pick("bracket_arm", lambda p: -p[2] - 0.12 * abs(p[0] + 6.0))
            # Front-left upturned corner flying eave / hip beast
            anchors["corner"] = pick("beast", lambda p: -p[0] - p[2] + 0.5 * p[1])
            # Internal bare softwood cross-beam / king post under roof
            anchors["truss"] = pick("cross_beam", lambda p: p[1] - 0.1 * abs(p[0]) - 0.08 * abs(p[2]))
            # Carved marble terrace step/balustrade
            anchors["terrace"] = pick("baluster_post", lambda p: -p[2] - 0.08 * abs(p[0] + 8.0))
        elif self.key == "palace":
            # Left front double-eave surround ridge, right front upper bracket set (never cross leaders!)
            anchors["eaves"] = pick("surround_ridge", lambda p: -p[0] - 1.2 * p[2] + 0.2 * p[1])
            anchors["brackets"] = pick("bracket_arm", lambda p: p[0] - 1.2 * p[2] + 0.6 * p[1])
        elif self.key == "gate":
            anchors["passage"] = np.array([self.center[0], self.lo[1] + self.ext[1] * 0.18, self.lo[2] + 0.8])
            anchors["talas"] = pick("tala_wall", lambda p: -p[0] - p[2] - abs(p[1] - self.ext[1] * 0.62) * 1.5)
            anchors["hara"] = pick("shala_roof", lambda p: -p[2] + 0.2 * p[0] - abs(p[1] - self.ext[1] * 0.50) * 1.8)
            anchors["kalasa"] = pick("kalasa", lambda p: p[1] - 0.1 * abs(p[0]))
        return anchors

    def grow(self, first: int, last: int):
        self.counts = schedule(self.blocks, first, last)
        self.stage_bounds = stages(self.blocks)
        self.stage_labels = [
            Counter(self.blocks[i].parameters.get("component_role") or "member" for i in range(a, b))
            .most_common(1)[0][0]
            .replace("_", " ")
            for a, b in self.stage_bounds
        ]

    def laying(self, count: int) -> str:
        ends = [b for _, b in self.stage_bounds]
        idx = min(int(np.searchsorted(ends, count, side="left")), len(ends) - 1)
        return self.stage_labels[idx]

    def check(self, code: str) -> tuple[str, str]:
        c = next(item for item in self.result.quality.checks if item.code == code)
        return c.status, c.message


class MassBatch:
    """Real 15-building randomized BlockN agent series (3x5 grid) for seed-morph + mass-generation showcase."""

    def __init__(self, key: str = "mass_batch"):
        import pyvista as pv
        from core.architecture.agent import build_recipe, series_recipe

        t0 = time.time()
        self.key = key
        self.product = "timber-house-series-15"
        self.variant = "randomized-batch"
        rec = series_recipe(count=15, seed=SEED, products=("timber-house",), kitbash=False)
        rec["columns"] = 3
        rec["spacing"] = 4.2
        results, district = build_recipe(rec)
        assert district.quality.status == "pass", "15-building mass series failed quality gate"
        self.result = district
        self.blocks = [b for b in district.scene.blocks if b.mesh is not None and not b.mesh.is_empty]

        # Measure full 3x5 neighborhood bounds and center it at X/Z = 0, Y = 0
        all_pts = np.vstack([b.mesh.vertices for b in self.blocks]).astype(np.float32)
        raw_lo = all_pts.min(axis=0).astype(np.float64)
        raw_hi = all_pts.max(axis=0).astype(np.float64)
        grid_shift = np.array([(raw_lo[0] + raw_hi[0]) * 0.5, raw_lo[1], (raw_lo[2] + raw_hi[2]) * 0.5], dtype=np.float64)

        self.points = (all_pts - grid_shift.astype(np.float32))
        self.lo = self.points.min(axis=0).astype(np.float64)
        self.hi = self.points.max(axis=0).astype(np.float64)
        self.center = (self.lo + self.hi) * 0.5
        self.ext = self.hi - self.lo
        self.sample = self.points[:: max(1, len(self.points) // 25000)]

        # Per-building solo (origin-centered) + grid-placed PolyData + crisp emerald touchdown edges
        offsets_by_id = {m["id"]: np.asarray(m["offset"], dtype=np.float64) for m in district.document.metadata["members"]}
        self.items = []
        solo_pts_all = []
        for (bid, res), entry in zip(results, rec["buildings"]):
            blks = [b for b in res.scene.blocks if b.mesh is not None and not b.mesh.is_empty]
            req = entry["request"]
            rules = res.document.metadata.get("timber_rules", {})
            phys = res.quality.metrics.get("physical", {})
            tonnes = int(round(float(phys.get("total_mass_kg", 0.0)) / 1000.0))

            solo_poly = build_shaded_polydata(pv, blks)
            s_lo = np.array(solo_poly.bounds[::2], dtype=np.float64)
            s_hi = np.array(solo_poly.bounds[1::2], dtype=np.float64)
            solo_shift = np.array([(s_lo[0] + s_hi[0]) * 0.5, s_lo[1], (s_lo[2] + s_hi[2]) * 0.5], dtype=np.float64)
            solo_poly.points = solo_poly.points - solo_shift

            # Grid-placed copy + emerald touchdown wireframe edges
            lot_off = offsets_by_id[bid] - grid_shift
            grid_poly = build_shaded_polydata(pv, blks)
            grid_poly.points = grid_poly.points + lot_off
            grid_edges = grid_poly.extract_feature_edges(
                feature_angle=30, boundary_edges=True, non_manifold_edges=True, manifold_edges=False
            )

            if len(solo_pts_all) < 6:
                solo_pts_all.append(np.asarray(solo_poly.points, dtype=np.float32))
            subs_map = res.quality.metrics.get("substance_counts", {})
            roof_mat = "SLATE ROOF" if "slate" in subs_map else "CLAY TILE ROOF"
            self.items.append({
                "id": bid,
                "seed": int(req["seed"]),
                "variant": str(req["variant"]).upper(),
                "levels": int(req["levels"]),
                "bays": int(req["bays"]),
                "footprint": f"{int(round(req['width']))}×{int(round(req['depth']))} m",
                "roof_mat": roof_mat,
                "brace": int(rules.get("brace_cadence", 3)),
                "members": len(blks),
                "tonnes": tonnes,
                "solo_poly": solo_poly,
                "grid_poly": grid_poly,
                "grid_edges": grid_edges,
            })

        # Reference bounding box across all 6 morph showcase houses so solo camera stays rock steady
        morph_pts = np.vstack([np.asarray(self.items[idx]["solo_poly"].points, dtype=np.float32) for _, idx in MORPH_STEPS])
        self.solo_lo = morph_pts.min(axis=0).astype(np.float64)
        self.solo_hi = morph_pts.max(axis=0).astype(np.float64)
        self.solo_center = (self.solo_lo + self.solo_hi) * 0.5
        self.solo_ext = self.solo_hi - self.solo_lo
        self.solo_sample = morph_pts[:: max(1, len(morph_pts) // 18000)]

        total_tonnes = sum(it["tonnes"] for it in self.items)
        self.metadata = {"tonnes": f"{total_tonnes:,}", "tonnes_raw": total_tonnes, "count": len(self.items)}
        self.pbr_poly = None
        self.pbr_body_poly = None
        self.pbr_roof_poly = None
        self.need_wire = False
        self.counts = None
        print(
            f"built {self.product}: 15 unique houses, {len(self.blocks):,} total members, "
            f"{total_tonnes:,} t, {district.quality.status}, {time.time() - t0:.1f}s",
            flush=True,
        )


# ------------------------------------------------------------------ Dual-pass GPU Stage
class Stage:
    """Dual-pass offscreen RTX 3090 renderer: Weathered PBR Studio + Crisp Perspective Phosphor CAD."""

    def __init__(self):
        import pyvista as pv

        self.pv = pv
        pv.global_theme.anti_aliasing = None

        # Pass 1: Weathered PBR Studio plotter with 3-point lighting + SSAO
        self.p_pbr = pv.Plotter(off_screen=True, window_size=(W, H), lighting="none")
        self.p_pbr.set_background("#050907", top="#0e1915")
        self.p_pbr.renderer.SetUseFXAA(True)
        self.lights = [
            pv.Light(position=(-60, 60, -50), focal_point=(0, 8, 0), color=(1.00, 0.94, 0.85), intensity=1.24),
            pv.Light(position=(65, 30, -35), focal_point=(0, 8, 0), color=(0.58, 0.79, 0.95), intensity=0.50),
            pv.Light(position=(25, 45, 70), focal_point=(0, 8, 0), color=(0.30, 0.96, 0.56), intensity=0.52),
            pv.Light(light_type="headlight", color=(0.82, 0.86, 0.85), intensity=0.18),
        ]
        for lt in self.lights:
            self.p_pbr.add_light(lt)
        self.pbr_floor = self.p_pbr.add_mesh(
            pv.Plane(center=(0, -0.04, 0), direction=(0, 1, 0), i_size=420, j_size=420, i_resolution=1, j_resolution=1),
            color="#070c0a", ambient=0.22, diffuse=0.45, specular=0.12, specular_power=16,
        )
        self.pbr_grid = self.p_pbr.add_mesh(
            pv.Plane(center=(0, -0.02, 0), direction=(0, 1, 0), i_size=360, j_size=360, i_resolution=60, j_resolution=60),
            style="wireframe", color="#10331e", line_width=1, lighting=False,
        )
        self.p_pbr.enable_ssao(radius=2.0, bias=0.02, kernel_size=64, blur=True)

        # Pass 2: Perspective Phosphor CAD plotter
        self.p_wire = pv.Plotter(off_screen=True, window_size=(W, H), lighting="none")
        self.p_wire.set_background("#020503")
        self.p_wire.renderer.SetUseFXAA(True)
        self.wire_grid = self.p_wire.add_mesh(
            pv.Plane(center=(0, -0.02, 0), direction=(0, 1, 0), i_size=360, j_size=360, i_resolution=60, j_resolution=60),
            style="wireframe", color="#072c14", line_width=1, lighting=False,
        )

        self.pbr_actors: dict[str, dict] = {}
        self.wire_actors: dict[str, tuple] = {}
        self.wire_state: dict[str, tuple] = {}
        self.cur_pbr: str | None = None
        self.cur_wire: str | None = None

    def _ensure_pbr(self, model: Model):
        if model.key in self.pbr_actors:
            return self.pbr_actors[model.key]
        entry = {}
        kw = dict(scalars="rgb", rgb=True, smooth_shading=True, ambient=0.24, diffuse=0.82, specular=0.18, specular_power=26)
        if isinstance(model, MassBatch):
            entry["solo"] = [self.p_pbr.add_mesh(it["solo_poly"], **kw) for it in model.items]
            entry["grid"] = [self.p_pbr.add_mesh(it["grid_poly"], **kw) for it in model.items]
            edges = []
            for it in model.items:
                ea = self.p_pbr.add_mesh(it["grid_edges"], color="#36ff7a", line_width=1.8, lighting=False)
                ea.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                ea.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)
                edges.append(ea)
            entry["grid_edges"] = edges
        elif model.pbr_body_poly is not None:
            entry["full"] = None
            entry["body"] = self.p_pbr.add_mesh(model.pbr_body_poly, **kw)
            entry["roof"] = self.p_pbr.add_mesh(model.pbr_roof_poly, **kw)
        else:
            entry["full"] = self.p_pbr.add_mesh(model.pbr_poly, **kw)
        self.pbr_actors[model.key] = entry
        return entry

    def _ensure_wire(self, model: Model):
        if model.key in self.wire_actors:
            return self.wire_actors[model.key]
        pv = self.pv
        faces = self.p_wire.add_mesh(
            pv.PolyData(model.points, model.cells), color="#020503", lighting=False, backface_culling=True
        )
        faces.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        faces.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(1, 1)
        traces = []
        for colour, points, lines, _ in model.traces:
            cool = self.p_wire.add_mesh(pv.PolyData(points, lines=lines), color=colour, line_width=1.15, lighting=False)
            warm = self.p_wire.add_mesh(pv.PolyData(points, lines=lines[:0]), color=WARM_HEX, line_width=1.4, lighting=False)
            hot = self.p_wire.add_mesh(pv.PolyData(points, lines=lines[:0]), color=HOT_HEX, line_width=2.2, lighting=False)
            for actor in (cool, warm, hot):
                actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
                actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, -2)
            traces.append((cool, warm, hot))
        self.wire_actors[model.key] = (faces, traces)
        return self.wire_actors[model.key]

    def _place_lights_and_floor(self, model: Model):
        c, e, lo = model.center, model.ext, model.lo
        span = max(float(np.max(e)), 18.0)
        self.lights[0].position = (c[0] - span * 1.45, lo[1] + span * 1.45, c[2] - span * 1.25)
        self.lights[0].focal_point = tuple(c)
        self.lights[1].position = (c[0] + span * 1.55, lo[1] + span * 0.85, c[2] - span * 0.90)
        self.lights[1].focal_point = tuple(c)
        self.lights[2].position = (c[0] + span * 0.55, lo[1] + span * 1.15, c[2] + span * 1.65)
        self.lights[2].focal_point = tuple(c)
        self.pbr_floor.SetPosition(c[0], lo[1] - 0.04, c[2])
        self.pbr_grid.SetPosition(c[0], lo[1] - 0.02, c[2])
        self.wire_grid.SetPosition(c[0], lo[1] - 0.02, c[2])

    def set_camera(self, plotter, pos, focal, fov_deg: float):
        cam = plotter.camera
        cam.parallel_projection = False
        cam.view_angle = float(fov_deg)
        cam.position = tuple(float(v) for v in pos)
        cam.focal_point = tuple(float(v) for v in focal)
        cam.up = (0.0, 1.0, 0.0)
        plotter.reset_camera_clipping_range()

    def project(self, plotter, point3d) -> tuple[float, float]:
        ren = plotter.renderer
        ren.SetWorldPoint(float(point3d[0]), float(point3d[1]), float(point3d[2]), 1.0)
        ren.WorldToDisplay()
        dx, dy, _ = ren.GetDisplayPoint()
        return float(dx), float(H - dy)

    def _hide_all_pbr(self):
        for items in self.pbr_actors.values():
            for actor in items.values():
                if isinstance(actor, list):
                    for sub in actor:
                        sub.SetVisibility(False)
                elif actor is not None:
                    actor.SetVisibility(False)

    def show_pbr(self, model: Model, show_roof: bool = True, roof_lift: float = 0.0):
        entry = self._ensure_pbr(model)
        if self.cur_pbr != model.key:
            self._hide_all_pbr()
            for actor in entry.values():
                if actor is not None and not isinstance(actor, list):
                    actor.SetVisibility(True)
            self._place_lights_and_floor(model)
            self.cur_pbr = model.key
        if entry.get("roof") is not None:
            entry["roof"].SetVisibility(show_roof)
            entry["roof"].SetPosition(0.0, float(roof_lift), 0.0)

    def shot_mass_solo(self, batch: MassBatch, item_index: int, pos, focal, fov_deg: float) -> np.ndarray:
        entry = self._ensure_pbr(batch)
        self._hide_all_pbr()
        self.cur_pbr = "mass_solo"
        # Place studio lights tight for single house scale
        c, e, lo = batch.solo_center, batch.solo_ext, batch.solo_lo
        span = max(float(np.max(e)), 18.0)
        self.lights[0].position = (c[0] - span * 1.45, lo[1] + span * 1.45, c[2] - span * 1.25)
        self.lights[0].focal_point = tuple(c)
        self.lights[1].position = (c[0] + span * 1.55, lo[1] + span * 0.85, c[2] - span * 0.90)
        self.lights[1].focal_point = tuple(c)
        self.lights[2].position = (c[0] + span * 0.55, lo[1] + span * 1.15, c[2] + span * 1.65)
        self.lights[2].focal_point = tuple(c)
        self.pbr_floor.SetPosition(c[0], lo[1] - 0.04, c[2])
        self.pbr_grid.SetPosition(c[0], lo[1] - 0.02, c[2])
        entry["solo"][item_index].SetVisibility(True)
        self.set_camera(self.p_pbr, pos, focal, fov_deg)
        self.p_pbr.render()
        return np.asarray(self.p_pbr.screenshot(return_img=True))

    def shot_mass_grid(self, batch: MassBatch, frame: int, pos, focal, fov_deg: float) -> np.ndarray:
        entry = self._ensure_pbr(batch)
        if self.cur_pbr != "mass_grid":
            self._hide_all_pbr()
            self._place_lights_and_floor(batch)
            self.cur_pbr = "mass_grid"
        for i, (ga, ea) in enumerate(zip(entry["grid"], entry["grid_edges"])):
            drop_f = GRID_DROP_START + i * GRID_DROP_STEP
            if frame < drop_f:
                ga.SetVisibility(False)
                ea.SetVisibility(False)
            else:
                age_f = frame - drop_f
                u_drop = ease_out(clamp01(age_f / 9.0))
                lift = (1.0 - u_drop) * 5.8
                ga.SetVisibility(True)
                ga.SetPosition(0.0, float(lift), 0.0)
                # Crisp glowing emerald wireframe touchdown highlight for first 7 frames!
                show_edge = age_f < 7
                ea.SetVisibility(show_edge)
                if show_edge:
                    ea.SetPosition(0.0, float(lift), 0.0)
        self.set_camera(self.p_pbr, pos, focal, fov_deg)
        self.p_pbr.render()
        return np.asarray(self.p_pbr.screenshot(return_img=True))

    def show_wire(self, model: Model, count: int, warm_from: int, hot_from: int):
        pv = self.pv
        faces, traces = self._ensure_wire(model)
        if self.cur_wire != model.key:
            for k, (f_act, t_acts) in self.wire_actors.items():
                vis = k == model.key
                f_act.SetVisibility(vis)
                for triple in t_acts:
                    for a in triple:
                        a.SetVisibility(vis)
            self._place_lights_and_floor(model)
            self.cur_wire = model.key
        st = (count, warm_from, hot_from)
        if self.wire_state.get(model.key) == st:
            return
        self.wire_state[model.key] = st
        face_count = int(np.searchsorted(model.face_member, count))
        faces.mapper.SetInputData(pv.PolyData(model.points, model.cells[: 4 * face_count]))
        for (cool, warm, hot), (_, points, lines, member) in zip(traces, model.traces):
            c_idx = int(np.searchsorted(member, warm_from))
            w_idx = int(np.searchsorted(member, hot_from))
            l_idx = int(np.searchsorted(member, count))
            cool.mapper.SetInputData(pv.PolyData(points, lines=lines[: 3 * c_idx]))
            warm.mapper.SetInputData(pv.PolyData(points, lines=lines[3 * c_idx : 3 * w_idx]))
            hot.mapper.SetInputData(pv.PolyData(points, lines=lines[3 * w_idx : 3 * l_idx]))

    def shot_pbr(self, model: Model, pos, focal, fov_deg: float, show_roof: bool = True, roof_lift: float = 0.0) -> np.ndarray:
        self.show_pbr(model, show_roof=show_roof, roof_lift=roof_lift)
        self.set_camera(self.p_pbr, pos, focal, fov_deg)
        self.p_pbr.render()
        return np.asarray(self.p_pbr.screenshot(return_img=True))

    def shot_wire(self, model: Model, pos, focal, fov_deg: float, count: int, warm_from: int, hot_from: int) -> np.ndarray:
        self.show_wire(model, count, warm_from, hot_from)
        self.set_camera(self.p_wire, pos, focal, fov_deg)
        self.p_wire.render()
        return np.asarray(self.p_wire.screenshot(return_img=True))


# ------------------------------------------------------------------ Perspective camera math
def fit_distance(model: Model, az_deg: float, el_deg: float, fov_deg: float,
                 fill_w: float = 0.88, fill_h: float = 0.54) -> float:
    """Compute exact perspective camera distance so the model fills (fill_w, fill_h) of 1080x1920."""
    az, el = math.radians(az_deg), math.radians(el_deg)
    # Camera looks from dir_vec toward center; azimuth 0 faces -Z side (front of BlockN buildings)
    d = np.array([math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el)], dtype=np.float64)
    d /= np.linalg.norm(d)
    right = np.cross(np.array([0.0, 1.0, 0.0]), d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)

    rel = model.sample - model.center
    half_w = float(np.max(np.abs(rel @ right)))
    half_h = float(np.max(np.abs(rel @ up)))
    depth_pad = float(np.max(rel @ d))

    tan_v = math.tan(math.radians(fov_deg * 0.5))
    tan_h = tan_v * (W / H)
    dist_w = half_w / max(tan_h * fill_w, 1e-4)
    dist_h = half_h / max(tan_v * fill_h, 1e-4)
    return max(dist_w, dist_h) + depth_pad * 0.65


def spherical_camera(center: np.ndarray, az_deg: float, el_deg: float, dist: float,
                     focal_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = np.array([math.sin(az) * math.cos(el), math.sin(el), -math.cos(az) * math.cos(el)], dtype=np.float64)
    focal = np.asarray(center, dtype=np.float64) + np.asarray(focal_offset, dtype=np.float64)
    pos = focal + d * float(dist)
    return pos, focal


# ------------------------------------------------------------------ Typography & compositor helpers
def find_font_file(style_key: str) -> Path:
    for candidate in FONT_CANDIDATES[style_key]:
        p = Path(candidate)
        if p.is_file():
            return p
    raise SystemExit(f"Font not found for {style_key}: {FONT_CANDIDATES[style_key]}")


class TypeEngine:
    def __init__(self):
        self.fonts = {k: ImageFont.truetype(str(find_font_file(f)), sz) for k, (f, sz) in TEXT_STYLES.items()}
        self.cache: dict = {}

    def measure(self, text: str, kind: str) -> tuple[int, int]:
        box = self.fonts[kind].getbbox(text)
        return box[2] - box[0], box[3] - box[1]

    def put(self, layer: Image.Image, text: str, kind: str, colour: tuple[int, int, int],
            x: float, y: float, alpha: float = 1.0, align: str = "left"):
        if alpha <= 0.004 or not text:
            return
        key = (text, kind, colour)
        if key not in self.cache:
            font = self.fonts[kind]
            box = font.getbbox(text)
            w, h = box[2] - box[0] + 6, box[3] - box[1] + 6
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(img).text((3 - box[0], 3 - box[1]), text, font=font, fill=colour + (255,))
            self.cache[key] = (img, box[2] - box[0], box[3] - box[1])
        img, width, _ = self.cache[key]
        if alpha < 0.999:
            img = img.copy()
            img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha)))
        if align == "centre":
            x -= width * 0.5
        elif align == "right":
            x -= width
        layer.alpha_composite(img, (int(round(x)), int(round(y))))


def phosphor_bloom(rgb: np.ndarray, tight_gain: float = 0.48, wide_gain: float = 0.20) -> np.ndarray:
    """Clean dual-radius phosphor bloom that preserves razor-sharp architectural edges."""
    base = Image.fromarray(rgb)
    tight = base.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(2.2)).resize((W, H), Image.BILINEAR)
    wide = base.resize((W // 8, H // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4.8)).resize((W, H), Image.BILINEAR)
    return (
        rgb.astype(np.float32)
        + tight_gain * np.asarray(tight, dtype=np.float32)
        + wide_gain * np.asarray(wide, dtype=np.float32)
    )


def scan_blend(pbr: np.ndarray, wire: np.ndarray, progress: float, mode: str = "diag",
               beam_strength: float = 1.0) -> np.ndarray:
    """Transition smoothly from phosphor CAD to weathered PBR along a glowing emerald laser plane."""
    u = clamp01(progress)
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    if mode == "diag":
        # Sweeps top-left to bottom-right across the hall
        field = 0.62 * xx + 0.38 * yy
        front = lerp(-0.08, 1.08, u)
        revealed = np.clip((front - field) / 0.018, 0.0, 1.0)[..., None]
        dist = field - front
    elif mode == "up":
        # Sweeps bottom-to-top up the 46m Dravidian tower!
        field = 1.0 - yy + 0.08 * (xx - 0.5)
        front = lerp(-0.06, 1.12, u)
        revealed = np.clip((front - field) / 0.018, 0.0, 1.0)[..., None]
        dist = field - front
    else:
        field = xx
        front = lerp(-0.06, 1.06, u)
        revealed = np.clip((front - field) / 0.020, 0.0, 1.0)[..., None]
        dist = field - front

    comp = pbr.astype(np.float32) * revealed + wire.astype(np.float32) * (1.0 - revealed)
    if beam_strength > 0.0 and 0.0 < u < 1.0:
        core = np.exp(-((dist / 0.0055) ** 2))[..., None]
        halo = np.exp(-((dist / 0.024) ** 2))[..., None]
        edge_fade = math.sin(math.pi * u) ** 0.35
        comp += beam_strength * edge_fade * (
            core * np.array([195.0, 255.0, 215.0], dtype=np.float32)
            + halo * np.array([15.0, 195.0, 82.0], dtype=np.float32) * 0.62
        )
    return comp


def emerald_wipe_glint(img: np.ndarray, progress: float) -> np.ndarray:
    """Fast subtle emerald scan glint sweeping across a freshly cut PBR hero frame."""
    u = clamp01(progress)
    if u <= 0.0 or u >= 1.0:
        return img
    xx = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]
    field = 0.78 * xx + 0.22 * yy
    front = lerp(-0.08, 1.12, u)
    dist = field - front
    core = np.exp(-((dist / 0.007) ** 2))[..., None]
    halo = np.exp(-((dist / 0.032) ** 2))[..., None]
    env = math.sin(math.pi * u)
    return img + env * (
        core * np.array([140.0, 255.0, 180.0], dtype=np.float32) * 0.55
        + halo * np.array([0.0, 180.0, 75.0], dtype=np.float32) * 0.32
    )


# ------------------------------------------------------------------ Main Film Controller
class Film:
    def __init__(self):
        self.stage = Stage()
        self.type = TypeEngine()
        self.models: dict[str, Model] = {}
        self.facts = catalogue_facts()
        print("catalogue:", self.facts, flush=True)

        # Precompute subtle cinematic vignette + top/bottom HUD scrim masks
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        rad = np.hypot((xx - W * 0.5) / (W * 0.55), (yy - H * 0.5) / (H * 0.55)) / math.sqrt(2.0)
        self.vignette = (1.0 - 0.26 * np.clip(rad - 0.38, 0.0, 1.0) ** 1.6)[..., None]

        top_a = np.clip((430.0 - yy) / 430.0, 0.0, 1.0) ** 1.45 * 0.78
        bot_a = np.clip((yy - 1310.0) / 610.0, 0.0, 1.0) ** 1.35 * 0.88
        self.scrim_alpha = np.maximum(top_a, bot_a)[..., None]

    def model(self, key: str) -> Model:
        if key not in self.models:
            if key == "mass_batch":
                m = MassBatch(key)
            elif key in HERO_MODELS:
                product, variant = HERO_MODELS[key]
                m = Model(key, product, variant, need_wire=(key in ("hall", "gate", "palace")), split_roof=(key == "hall"))
                for s in SCENES:
                    if s["model"] == key and s.get("grow"):
                        m.grow(*s["grow"])
            else:
                item = next(it for it in REEL_ITEMS if it["key"] == key)
                m = Model(key, item["product"], item["variant"], need_wire=False, split_roof=False)
            self.models[key] = m
        return self.models[key]

    def count_at(self, scene: dict, model: Model, frame: int) -> tuple[int, int, int]:
        if not scene.get("grow"):
            n = len(model.blocks)
            return n, n, n
        counts = model.counts
        def at(back: int) -> int:
            return counts[min(max(frame - back, 0), len(counts) - 1)]
        return at(0), at(WARM_FRAMES), at(HOT_FRAMES)

    # ------------------------------ Camera choreography per scene
    def camera_for(self, scene: dict, model: Model, frame: int) -> tuple[np.ndarray, np.ndarray, float]:
        u = clamp01((frame - scene["first"]) / max(1, scene["last"] - scene["first"]))
        key = scene["key"]
        c, e, lo = model.center, model.ext, model.lo

        if key == "s01_open":
            # Dramatic low-angle corner pullback crane orbit into wide hero establishing shot
            t = ease_quint(u)
            fov = lerp(36.0, 33.0, t)
            az = lerp(-46.0, -24.0, t)
            el = lerp(11.0, 21.0, t)
            d_end = fit_distance(model, -24.0, 21.0, 33.0, fill_w=0.90, fill_h=0.46)
            d_start = d_end * 0.72
            dist = lerp(d_start, d_end, t)
            fo = (lerp(-e[0] * 0.14, 0.0, t), lerp(-e[1] * 0.10, -e[1] * 0.03, t), 0.0)
            pos, focal = spherical_camera(c, az, el, dist, fo)
            return pos, focal, fov

        if key == "s02_hall_pbr":
            # Seamless matched continuation from S01! Orbits around front to +14 deg while dollying in
            t = ease(u)
            fov = 33.0
            az = lerp(-24.0, 14.0, t)
            el = lerp(21.0, 15.5, t)
            d0 = fit_distance(model, -24.0, 21.0, 33.0, fill_w=0.90, fill_h=0.46)
            d1 = fit_distance(model, 14.0, 15.5, 33.0, fill_w=0.96, fill_h=0.48) * 0.88
            dist = lerp(d0, d1, t)
            fo = (0.0, lerp(-e[1] * 0.03, -e[1] * 0.06, t), 0.0)
            pos, focal = spherical_camera(c, az, el, dist, fo)
            return pos, focal, fov

        if key == "s03_hall_macro":
            # Intimate close-up glide along dougong brackets & upturned flying eaves, then crane up as roof opens!
            peel_u = ease_quint(clamp01((frame - 772) / 115.0))
            glide_u = ease(u)
            fov = lerp(31.0, 33.0, peel_u)
            az = lerp(-34.0, -52.0, glide_u)
            el = lerp(11.0, 33.0, peel_u)
            d_tight = fit_distance(model, -38.0, 14.0, 31.0, fill_w=0.95, fill_h=0.52) * 0.56
            d_open = fit_distance(model, -50.0, 33.0, 33.0, fill_w=0.92, fill_h=0.50) * 0.72
            dist = lerp(d_tight, d_open, peel_u)
            # Focus starts near front-left eaves & brackets, lifts gently toward central timber truss
            fo = (
                lerp(-e[0] * 0.22, -e[0] * 0.06, peel_u),
                lerp(e[1] * 0.04, e[1] * 0.06, peel_u),
                lerp(-e[2] * 0.24, -e[2] * 0.04, peel_u),
            )
            pos, focal = spherical_camera(c, az, el, dist, fo)
            return pos, focal, fov

        if key == "s04_palace":
            # Low across marble triple-stairs -> sweeping crane up to reveal double eaves & 9 bays
            t = ease_quint(u)
            fov = 34.0
            az = lerp(-36.0, 18.0, t)
            el = lerp(10.0, 23.0, t)
            d_mid = fit_distance(model, -18.0, 18.0, 34.0, fill_w=0.92, fill_h=0.48)
            dist = lerp(d_mid * 0.78, d_mid * 0.96, t)
            fo = (lerp(-e[0] * 0.08, 0.0, t), lerp(-e[1] * 0.08, 0.02 * e[1], t), 0.0)
            pos, focal = spherical_camera(c, az, el, dist, fo)
            return pos, focal, fov

        if key == "s05_gate_grow":
            # Dramatic LOW upward tilt through the open granite passage -> crane skyward as 7 talas rise!
            t = ease_quint(u)
            fov = lerp(38.0, 35.0, t)
            az = lerp(-18.0, 18.0, t)
            el = lerp(5.5, 16.5, t)
            d_full = fit_distance(model, 16.0, 16.5, 35.0, fill_w=0.80, fill_h=0.45)
            dist = lerp(d_full * 0.72, d_full * 1.02, t)
            # Focal sits slightly lower than tower center so the 46m tower sits cleanly in y 330..1410
            fy = lerp(lo[1] + e[1] * 0.22, lo[1] + e[1] * 0.40, t) - c[1]
            pos, focal = spherical_camera(c, az, el, dist, (0.0, fy, 0.0))
            return pos, focal, fov

        if key == "s06_gate_pbr":
            # Seamless matched orbit from S05 end (+18 deg -> -16 deg), granite base stays well clear of rule cards
            t = ease(u)
            fov = 35.0
            az = lerp(18.0, -16.0, t)
            el = lerp(16.5, 14.0, t)
            d0 = fit_distance(model, 16.0, 16.5, 35.0, fill_w=0.80, fill_h=0.45)
            dist = lerp(d0 * 1.02, d0 * 0.96, t)
            fy = lerp(lo[1] + e[1] * 0.40, lo[1] + e[1] * 0.37, t) - c[1]
            pos, focal = spherical_camera(c, az, el, dist, (0.0, fy, 0.0))
            return pos, focal, fov

        if key == "s06_vimana":
            # Low sweeping arc across the pillared mandapa rising up the solid-granite 5-tala Vimana
            t = ease(u)
            fov = 35.0
            az = lerp(-56.0, -24.0, t)
            el = lerp(13.0, 21.0, t)
            d = fit_distance(model, -40.0, 17.0, 35.0, fill_w=0.84, fill_h=0.44)
            dist = lerp(d * 1.00, d * 0.90, t)
            pos, focal = spherical_camera(c, az, el, dist, (0.0, -e[1] * 0.14, 0.0))
            return pos, focal, fov

        if scene["mode"] == "reel":
            it = scene["reel_item"]
            az0, el0, s0, f0, fov = it["cam0"]
            az1, el1, s1, f1, _ = it["cam1"]
            t = ease(u)
            az = lerp(az0, az1, t)
            el = lerp(el0, el1, t)
            d_ref = fit_distance(model, (az0 + az1) * 0.5, (el0 + el1) * 0.5, fov, fill_w=0.90, fill_h=0.48)
            dist = d_ref * lerp(s0, s1, t)
            fo = (
                lerp(f0[0], f1[0], t) * e[0],
                lerp(f0[1], f1[1], t) * e[1],
                lerp(f0[2], f1[2], t) * e[2],
            )
            pos, focal = spherical_camera(c, az, el, dist, fo)
            return pos, focal, float(fov)

        if scene["mode"] == "seed_morph":
            # Smooth orbital sweep around origin-centered solo houses while seeds mutate rapidly!
            t = ease(u)
            fov = 34.0
            az = lerp(-42.0, 16.0, t)
            el = lerp(15.0, 21.0, t)
            # Fit smoothly against max bounds across all 6 morph showcase houses
            dummy = type("D", (), {"sample": model.solo_sample, "center": model.solo_center})()
            d_ref = fit_distance(dummy, -15.0, 18.0, fov, fill_w=0.84, fill_h=0.42)
            dist = lerp(d_ref * 0.96, d_ref * 0.88, t)
            pos, focal = spherical_camera(model.solo_center, az, el, dist, (0.0, -model.solo_ext[1] * 0.08, 0.0))
            return pos, focal, fov

        if scene["mode"] == "mass_grid":
            # Dramatic sweeping aerial crane pullback as 15 unique randomized houses cascade onto the 3x5 lot!
            t = ease_quint(u)
            fov = 35.0
            az = lerp(-36.0, 18.0, t)
            el = lerp(24.0, 42.0, t)
            d_full = fit_distance(model, 8.0, 38.0, fov, fill_w=0.90, fill_h=0.48)
            dist = lerp(d_full * 0.68, d_full * 1.06, t)
            fo = (
                lerp(-e[0] * 0.14, 0.0, t),
                lerp(-e[1] * 0.10, -e[1] * 0.25, t),
                lerp(-e[2] * 0.22, 0.0, t),
            )
            pos, focal = spherical_camera(c, az, el, dist, fo)
            return pos, focal, fov

        # Outro: slow majestic orbital drift behind crisp static closing lockup
        az = lerp(-28.0, -10.0, u)
        el = 19.0
        dist = fit_distance(model, -20.0, 19.0, 34.0, fill_w=0.92, fill_h=0.46) * 0.92
        pos, focal = spherical_camera(c, az, el, dist, (0.0, -e[1] * 0.14, 0.0))
        return pos, focal, 34.0

    # ------------------------------ Render a single frame
    def render(self, frame: int) -> Image.Image:
        scene = scene_at(frame)
        model = self.model(scene["model"])
        pos, focal, fov = self.camera_for(scene, model, frame)
        mode = scene["mode"]
        count, warm_from, hot_from = self.count_at(scene, model, frame)
        proj_plotter = self.stage.p_pbr

        if mode == "wire_grow":
            raw_wire = self.stage.shot_wire(model, pos, focal, fov, count, warm_from, hot_from)
            img = phosphor_bloom(raw_wire)
            proj_plotter = self.stage.p_wire

        elif mode in ("scan_diag", "scan_up"):
            s0, s1 = scene["scan"]
            if frame < s0:
                raw_wire = self.stage.shot_wire(model, pos, focal, fov, len(model.blocks), len(model.blocks), len(model.blocks))
                img = phosphor_bloom(raw_wire)
                proj_plotter = self.stage.p_wire
            elif frame <= s1:
                raw_wire = self.stage.shot_wire(model, pos, focal, fov, len(model.blocks), len(model.blocks), len(model.blocks))
                wire_glow = np.clip(phosphor_bloom(raw_wire), 0, 255)
                pbr = self.stage.shot_pbr(model, pos, focal, fov)
                scan_u = ease_quint((frame - s0) / max(1, s1 - s0))
                img = scan_blend(pbr, wire_glow, scan_u, mode="diag" if mode == "scan_diag" else "up")
                proj_plotter = self.stage.p_pbr
            else:
                img = self.stage.shot_pbr(model, pos, focal, fov).astype(np.float32)
                proj_plotter = self.stage.p_pbr

        elif mode == "macro_open_roof":
            p0, p1 = scene["peel"]
            if frame < p0:
                img = self.stage.shot_pbr(model, pos, focal, fov, show_roof=True, roof_lift=0.0).astype(np.float32)
            elif frame <= p1:
                pu = ease_quint((frame - p0) / max(1, p1 - p0))
                body_only = self.stage.shot_pbr(model, pos, focal, fov, show_roof=False).astype(np.float32)
                with_roof = self.stage.shot_pbr(model, pos, focal, fov, show_roof=True, roof_lift=pu * 5.2).astype(np.float32)
                img = body_only * pu + with_roof * (1.0 - pu)
                # Subtle emerald rim pulse as roof clears
                pulse = math.sin(math.pi * pu) * 0.14
                img[:, :, 1] = np.clip(img[:, :, 1] + pulse * 140.0, 0, 255)
            else:
                img = self.stage.shot_pbr(model, pos, focal, fov, show_roof=False).astype(np.float32)
            proj_plotter = self.stage.p_pbr

        elif mode == "pbr_glint":
            img = self.stage.shot_pbr(model, pos, focal, fov).astype(np.float32)
            g0, g1 = scene["glint"]
            if g0 <= frame <= g1:
                img = emerald_wipe_glint(img, (frame - g0) / max(1, g1 - g0))
            proj_plotter = self.stage.p_pbr

        elif mode == "reel":
            img = self.stage.shot_pbr(model, pos, focal, fov).astype(np.float32)
            local = frame - scene["first"]
            if local < 10:
                img = emerald_wipe_glint(img, local / 10.0)
            proj_plotter = self.stage.p_pbr

        elif mode == "seed_morph":
            step_start, item_idx = max((s for s in MORPH_STEPS if s[0] <= frame), key=lambda s: s[0])
            img = self.stage.shot_mass_solo(model, item_idx, pos, focal, fov).astype(np.float32)
            local = frame - step_start
            if local < 9:
                img = emerald_wipe_glint(img, local / 9.0)
            proj_plotter = self.stage.p_pbr

        elif mode == "mass_grid":
            img = self.stage.shot_mass_grid(model, frame, pos, focal, fov).astype(np.float32)
            local = frame - scene["first"]
            if local < 10:
                img = emerald_wipe_glint(img, local / 10.0)
            proj_plotter = self.stage.p_pbr

        else:  # outro
            pbr = self.stage.shot_pbr(model, pos, focal, fov).astype(np.float32)
            raw_wire = self.stage.shot_wire(model, pos, focal, fov, len(model.blocks), len(model.blocks), len(model.blocks))
            wire_g = phosphor_bloom(raw_wire, 0.35, 0.15)
            dim_u = ease(ramp(frame, OUTRO_FIRST, 28))
            img = (pbr * 0.16 + wire_g * 0.14) * lerp(1.10, 0.62, dim_u)
            proj_plotter = self.stage.p_pbr

        # Apply subtle cinematic vignette + dark HUD scrims so typography pops cleanly at phone size
        img *= self.vignette
        if mode != "outro":
            img = img * (1.0 - self.scrim_alpha) + np.array([4.0, 8.0, 6.0], dtype=np.float32) * self.scrim_alpha
        else:
            img *= 0.85

        # Composite vector typography, 3D pins, telemetry & rule badges
        canvas = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if mode == "outro":
            self._draw_outro(layer, frame)
        elif mode == "reel":
            self._draw_reel(layer, frame, scene, model)
        elif mode == "seed_morph":
            self._draw_seed_morph(layer, frame, scene, model)
        elif mode == "mass_grid":
            self._draw_mass_grid(layer, frame, scene, model)
        else:
            self._draw_hero_hud(layer, frame, scene, model, count, proj_plotter)
        canvas.alpha_composite(layer)

        out = np.asarray(canvas.convert("RGB"), dtype=np.float32)
        out *= self._envelope(frame)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def _envelope(self, frame: int) -> float:
        level = 1.0
        if frame < 5:
            level = min(level, ease(frame / 5.0))
        # Short crisp dip only at the major act switch into Dravidian (1170) & Outro (2415)
        for cut in (1170, OUTRO_FIRST):
            d = abs(frame - cut + 0.5)
            level = min(level, clamp01((d - 0.5) / 4.0))
        if frame >= FADE_OUT:
            level = min(level, 1.0 - ease((frame - FADE_OUT + 1) / (LAST - FADE_OUT + 1)))
        return level

    # ------------------------------ HUD drawing helpers
    def _draw_header(self, layer: Image.Image, frame: int, start: int, eyebrow: str, title: str, sub: str, leaving: float = 1.0):
        draw = ImageDraw.Draw(layer)
        a0 = ease_out(ramp(frame, start, 10)) * leaving
        a1 = ease_out(ramp(frame, start + 3, 10)) * leaving
        a2 = ease_out(ramp(frame, start + 7, 10)) * leaving
        rise = 20.0 * (1.0 - a0)

        # Glowing emerald accent pill bar beside eyebrow
        if a0 > 0.01:
            draw.rounded_rectangle((LEFT, int(105 + rise), LEFT + 6, int(131 + rise)), radius=3, fill=C_GREEN + (int(255 * a0),))
            self.type.put(layer, eyebrow, "eyebrow", C_GREEN, LEFT + 18, 104 + rise, a0)
        self.type.put(layer, title, "title", C_WHITE, LEFT, 144 + 18.0 * (1.0 - a1), a1)
        self.type.put(layer, sub, "sub", C_SAGE, LEFT, 232 + 12.0 * (1.0 - a2), a2)
        line_w = (RIGHT - LEFT) * ease_out(ramp(frame, start + 4, 16))
        if line_w > 1 and a0 > 0.01:
            draw.line((LEFT, 282, LEFT + line_w, 282), fill=C_RULE + (int(255 * a0),), width=2)
            # Small glowing emerald tip on rule
            draw.line((LEFT, 282, LEFT + min(110.0, line_w), 282), fill=C_GREEN + (int(220 * a0),), width=2)

    def _draw_pill(self, layer: Image.Image, x: float, y: float, text: str, active: float,
                   accent: tuple[int, int, int] = C_GREEN, alpha: float = 1.0) -> float:
        if alpha <= 0.01:
            return x
        tw, th = self.type.measure(text, "tag")
        pw, ph = tw + 28, 42
        draw = ImageDraw.Draw(layer)
        bg_a = int((140 + 75 * active) * alpha)
        bd_col = tuple(int(lerp(c0, c1, active)) for c0, c1 in zip(C_RULE, accent))
        tx_col = tuple(int(lerp(c0, c1, active)) for c0, c1 in zip(C_SAGE, C_WHITE))
        draw.rounded_rectangle((int(x), int(y), int(x + pw), int(y + ph)), radius=8,
                               fill=(8, 18, 12, bg_a), outline=bd_col + (int(255 * alpha),), width=2)
        self.type.put(layer, text, "tag", tx_col, x + 14, y + 9, alpha)
        return x + pw + 14

    def _draw_pin(self, layer: Image.Image, plotter, world_pt, label: str, sub: str,
                  frame: int, start: int, end: int, side: str = "right", y_bias: float = -70.0):
        """3D world-tracked architectural callout pin locked to an actual BlockN member centroid."""
        if frame < start or frame > end + 12:
            return
        a_in = ease_out(ramp(frame, start, 10))
        a_out = 1.0 - ramp(frame, end, 10)
        alpha = a_in * a_out
        if alpha <= 0.01:
            return
        sx, sy = self.stage.project(plotter, world_pt)
        sx = max(90.0, min(W - 90.0, sx))
        sy = max(350.0, min(1330.0, sy))

        w1, _ = self.type.measure(label, "pin")
        w2, _ = self.type.measure(sub, "pin_sub") if sub else (0, 0)
        box_w = max(w1, w2) + 34
        box_h = 62 if sub else 40

        # Place callout card cleanly on chosen side, strictly inside safe vertical & horizontal bounds
        if side == "right":
            bx = min(RIGHT - box_w, max(LEFT, sx + 48))
        elif side == "left":
            bx = max(LEFT, min(RIGHT - box_w, sx - 48 - box_w))
        else:
            bx = max(LEFT, min(RIGHT - box_w, sx - box_w * 0.5))
        by = max(324.0, min(1320.0 - box_h, sy + y_bias))

        draw = ImageDraw.Draw(layer)
        # Elbow leader line from real 3D anchor point to box edge
        attach_x = bx if bx > sx else bx + box_w
        attach_y = by + box_h * 0.5
        mid_x = lerp(sx, attach_x, 0.45)
        draw.line((sx, sy, mid_x, attach_y, attach_x, attach_y), fill=C_GREEN + (int(220 * alpha),), width=2)
        # Glowing anchor ring on geometry
        r_out = 11 + 6 * (1.0 - a_in)
        draw.ellipse((sx - r_out, sy - r_out, sx + r_out, sy + r_out), outline=C_GREEN + (int(230 * alpha),), width=2)
        draw.ellipse((sx - 4, sy - 4, sx + 4, sy + 4), fill=C_MINT + (int(255 * alpha),))
        # Glass tag card with crisp emerald left accent bar
        draw.rounded_rectangle((int(bx), int(by), int(bx + box_w), int(by + box_h)), radius=8,
                               fill=(6, 14, 10, int(218 * alpha)), outline=C_RULE + (int(240 * alpha),), width=1)
        draw.rounded_rectangle((int(bx), int(by + 6), int(bx + 5), int(by + box_h - 6)), radius=2,
                               fill=C_GREEN + (int(255 * alpha),))
        self.type.put(layer, label, "pin", C_WHITE, bx + 16, by + 9, alpha)
        if sub:
            self.type.put(layer, sub, "pin_sub", C_MINT, bx + 16, by + 35, alpha)

    def _draw_hero_hud(self, layer: Image.Image, frame: int, scene: dict, model: Model, count: int, plotter):
        draw = ImageDraw.Draw(layer)
        key = scene["key"]
        leaving = 1.0 - ramp(frame, scene["last"] - 6, 6) if key not in ("s01_open", "s05_gate_grow") else 1.0
        sub = scene["sub"].format_map(model.metadata)
        self._draw_header(layer, frame, scene["first"] + 2, scene["eyebrow"], scene["title"], sub, leaving)

        # 3D world-tracked architectural pins synchronized to exact spoken words (non-overlapping, floating in clear sky)
        if key == "s03_hall_macro":
            self._draw_pin(layer, plotter, model.anchors["bracket"], "121 DOUGONG BRACKETS",
                           "QING DOUKOU PROPORTION", frame, 686, 742, side="right", y_bias=-360)
            self._draw_pin(layer, plotter, model.anchors["corner"], "FLYING CORNERS + HIP BEASTS",
                           "UPTURNED CONCAVE EAVES", frame, 754, 798, side="left", y_bias=-310)
            self._draw_pin(layer, plotter, model.anchors["truss"], "OPEN ROOF · TIMBER FRAME",
                           "BARE SOFTWOOD BEAMS + PURLINS", frame, 818, 892, side="right", y_bias=-310)
        elif key == "s04_palace":
            self._draw_pin(layer, plotter, model.anchors["eaves"], "DOUBLE EAVES + CLERESTORY",
                           "9-BAY PALACE RANK", frame, 992, 1152, side="left", y_bias=-310)
            self._draw_pin(layer, plotter, model.anchors["brackets"], "242 BRACKET SETS",
                           "9-STEP QING DOUGONG", frame, 1028, 1152, side="right", y_bias=-300)
        elif key == "s05_gate_grow":
            self._draw_pin(layer, plotter, model.anchors["passage"], "OPEN GRANITE PASSAGE",
                           "VERIFIED THROUGH-BASE CLEARANCE", frame, 1392, 1492, side="right", y_bias=40)
        elif key == "s06_gate_pbr":
            self._draw_pin(layer, plotter, model.anchors["talas"], "7 DIMINISHING TALAS",
                           "CONCAVE STEPPED PROFILE", frame, 1514, 1598, side="left", y_bias=-340)
            self._draw_pin(layer, plotter, model.anchors["hara"], "KUTA + SHALA PARAPETS",
                           "MINIATURE SHRINE HARAS", frame, 1560, 1630, side="right", y_bias=-260)
            self._draw_pin(layer, plotter, model.anchors["kalasa"], "GILDED COPPER KALASAS",
                           "SUMMIT CROWN FINIALS", frame, 1612, 1742, side="right", y_bias=-95)

        # Bottom telemetry / verification dock (y 1470..1800)
        total = len(model.blocks)
        q = model.result.quality
        q_str = f"QUALITY PASS  ·  {q.counts['pass']}/{q.counts['total']} CHECKS"
        y0 = 1520

        if scene.get("grow"):
            g0, g1 = scene["grow"]
            a = ease_out(ramp(frame, g0, 10))
            done = ease_out(ramp(frame, g1, 10))
            cnt_txt = f"{count:,}"
            cw, _ = self.type.measure(cnt_txt, "stat")
            self.type.put(layer, cnt_txt, "stat", C_WHITE, LEFT, y0, a)
            self.type.put(layer, f"OF {total:,} MEMBERS", "stat_sm", C_GREEN, LEFT + cw + 18, y0 + 20, a)
            # Sleek progress bar
            bar_y = y0 + 74
            draw.rounded_rectangle((LEFT, bar_y, RIGHT, bar_y + 10), radius=5,
                                   fill=(14, 34, 22, int(200 * a)), outline=C_RULE + (int(220 * a),), width=1)
            fill_w = max(10, int((RIGHT - LEFT) * count / max(1, total)))
            draw.rounded_rectangle((LEFT, bar_y, LEFT + fill_w, bar_y + 10), radius=5,
                                   fill=C_GREEN + (int(255 * a),))
            # Stage label transitions into quality pass + tonnage upon completion
            self.type.put(layer, f"ASSEMBLING  ·  {model.laying(count).upper()}", "line", C_SAGE, LEFT, y0 + 104, a * (1.0 - done))
            self.type.put(layer, f"{q_str}  ·  {model.metadata['tonnes']} TONNES", "line", C_MINT, LEFT, y0 + 104, done)

        elif key == "s02_hall_pbr":
            a = ease_out(ramp(frame, 304, 12)) * leaving
            s0, s1 = scene["scan"]
            scan_u = clamp01((frame - s0) / max(1, s1 - s0))
            self.type.put(layer, f"{total:,}", "stat", C_WHITE, LEFT, y0 - 18, a)
            cw, _ = self.type.measure(f"{total:,}", "stat")
            self.type.put(layer, "MEMBERS", "stat_sm", C_SAGE, LEFT + cw + 16, y0 + 2, a)
            # Tonnage badge on right
            self.type.put(layer, f"{model.metadata['tonnes']} TONNES", "stat_sm", C_AMBER, RIGHT, y0 + 2, a * ease_out(ramp(frame, 390, 14)), align="right")
            # Mode status + substance pills synced to Eric's voice
            mode_lbl = "PHOSPHOR CAD" if scan_u < 0.05 else ("MATERIALIZING WEATHERED PBR..." if scan_u < 0.99 else "WEATHERED PBR  ·  AGE 0.35  ·  6 SUBSTANCES")
            self.type.put(layer, mode_lbl, "line", C_MINT if scan_u >= 0.05 else C_GREEN, LEFT, y0 + 56, a)
            px = LEFT
            for trig, lbl in scene["materials"]:
                act = ease_out(ramp(frame, trig, 10))
                px = self._draw_pill(layer, px, y0 + 114, lbl, act, accent=C_AMBER if "GLAZED" in lbl else C_GREEN, alpha=a)

        elif key == "s03_hall_macro":
            a = ease_out(ramp(frame, 606, 12)) * leaving
            peel_a = ease_out(ramp(frame, 802, 14))
            badge = "VIEW: OPEN ROOF CUTAWAY" if peel_a > 0.4 else "VIEW: MACRO PROPORTION AUDIT"
            self._draw_pill(layer, LEFT, y0 - 20, badge, 1.0, accent=C_GREEN, alpha=a)
            self.type.put(layer, "SONG + QING RULES  ·  VERIFIED ON GEOMETRY", "line", C_WHITE, LEFT, y0 + 44, a)
            self.type.put(layer, f"{q_str}  ·  MAX GRAVITY LOAD {model.metadata['util_pct']}%", "line", C_SAGE, LEFT, y0 + 92, a)

        elif key == "s04_palace":
            a = ease_out(ramp(frame, 912, 12)) * leaving
            mass_hl = ease_out(ramp(frame, 1072, 12))
            self.type.put(layer, f"{total:,}", "stat", C_WHITE, LEFT, y0 - 10, a)
            cw, _ = self.type.measure(f"{total:,}", "stat")
            self.type.put(layer, "MEMBERS", "stat_sm", C_SAGE, LEFT + cw + 16, y0 + 10, a)
            self.type.put(layer, f"{model.metadata['tonnes']} TONNES", "stat_sm", C_AMBER if mass_hl > 0.2 else C_SAGE,
                          RIGHT, y0 + 10, a, align="right")
            px = LEFT
            px = self._draw_pill(layer, px, y0 + 68, "9 BAYS", ease_out(ramp(frame, 932, 10)), alpha=a)
            px = self._draw_pill(layer, px, y0 + 68, "242 BRACKET SETS", ease_out(ramp(frame, 1028, 10)), alpha=a)
            self._draw_pill(layer, px, y0 + 68, f"PASS {q.counts['pass']}/{q.counts['total']}", mass_hl, accent=C_AMBER, alpha=a)

        elif key == "s06_gate_pbr":
            r_in = scene["rules_in"]
            pre_a = ease_out(ramp(frame, 1504, 10)) * (1.0 - ramp(frame, r_in - 12, 10))
            if pre_a > 0.01:
                self.type.put(layer, f"{model.metadata['tonnes']} TONNES", "stat", C_AMBER, LEFT, y0 - 14, pre_a)
                self.type.put(layer, f"12,084 t GRANITE  ·  4,850 t BRICK  ·  GILT COPPER", "line", C_WHITE, LEFT, y0 + 56, pre_a)
                self.type.put(layer, f"GRAVITY LOAD CHECK: PASS (MAX {model.metadata['util_pct']}%)", "line", C_MINT, LEFT, y0 + 104, pre_a)
            # Measured rule checks card
            ra = ease_out(ramp(frame, r_in, 10)) * leaving
            if ra > 0.01:
                self.type.put(layer, "MEASURED DIRECTLY ON FINISHED GEOMETRY", "eyebrow", C_GREEN, LEFT, 1466, ra)
                ry = 1512
                for idx, code in enumerate(scene["rules"]):
                    status, msg = model.check(code)
                    app = r_in + 10 + idx * 18
                    ia = ease_out(ramp(frame, app, 10)) * leaving
                    rise = 14.0 * (1.0 - ia)
                    if ia > 0.01:
                        draw.rounded_rectangle((LEFT, int(ry + rise), LEFT + 92, int(ry + rise + 34)), radius=6,
                                               fill=(0, 62, 26, int(230 * ia)), outline=C_GREEN + (int(255 * ia),), width=1)
                        self.type.put(layer, status.upper(), "tag", C_MINT, LEFT + 46, ry + rise + 5, ia, align="centre")
                        for r_i, line in enumerate(textwrap.wrap(msg, 46)[:2]):
                            self.type.put(layer, line, "rule", C_WHITE, LEFT + 112, ry + rise + r_i * 34, ia)
                    ry += 92

        elif key == "s06_vimana":
            a = ease_out(ramp(frame, 1762, 10)) * leaving
            self.type.put(layer, f"{total:,}", "stat", C_WHITE, LEFT, y0 - 10, a)
            cw, _ = self.type.measure(f"{total:,}", "stat")
            self.type.put(layer, "GRANITE BLOCKS", "stat_sm", C_SAGE, LEFT + cw + 16, y0 + 10, a)
            self.type.put(layer, f"{model.metadata['tonnes']} TONNES", "stat_sm", C_AMBER, RIGHT, y0 + 10, a, align="right")
            self.type.put(layer, f"{q_str}  ·  MAX GRAVITY LOAD {model.metadata['util_pct']}%", "line", C_MINT, LEFT, y0 + 72, a)

    def _draw_reel(self, layer: Image.Image, frame: int, scene: dict, model: Model):
        idx = scene["reel_index"]
        it = scene["reel_item"]
        # Persistent top header across the entire 8-product PBR showcase reel
        head_a = ease_out(ramp(frame, REEL_FIRST + 2, 10)) * (1.0 - ramp(frame, MASS_FIRST - 8, 8))
        self._draw_header(
            layer, frame, REEL_FIRST + 2,
            f"BLOCKN PLATFORM  ·  {self.facts['architects']} ARCHITECTS",
            f"{self.facts['products']} PRODUCTS",
            "29 real substances · per-face weathering · gravity audited",
            leaving=1.0 - ramp(frame, MASS_FIRST - 8, 8),
        )

        local_a = ease_out(ramp(frame, scene["first"] + 1, 7))
        rise = 14.0 * (1.0 - local_a)
        y0 = 1468
        self.type.put(layer, it["architect"], "eyebrow", C_GREEN, LEFT, y0 + rise, local_a)
        self.type.put(layer, f"{idx + 1:02d} / {len(REEL_ITEMS):02d}", "index", C_SAGE, RIGHT, y0, head_a, align="right")
        self.type.put(layer, it["title"], "reel_title", C_WHITE, LEFT, y0 + 36 + rise, local_a)
        self.type.put(
            layer,
            f"{len(model.blocks):,} MEMBERS  ·  {model.metadata['tonnes']} t  ·  PASS",
            "line", C_AMBER, LEFT, y0 + 112, ease_out(ramp(frame, scene["first"] + 4, 7)),
        )
        self.type.put(layer, it["sub_tag"], "sub", C_SAGE, LEFT, y0 + 154, ease_out(ramp(frame, scene["first"] + 5, 7)))

        # Export format pills ribbon lights up word-by-word as Eric says "GLB, OBJ, USD, and physics"
        ribbon_a = ease_out(ramp(frame, 2266, 12)) * (1.0 - ramp(frame, MASS_FIRST - 8, 8))
        if ribbon_a > 0.01:
            px = LEFT
            for trig, lbl in EXPORT_PILLS:
                act = ease_out(ramp(frame, trig, 8))
                px = self._draw_pill(layer, px, 1692, lbl, act, accent=C_GREEN, alpha=ribbon_a)

    def _draw_seed_morph(self, layer: Image.Image, frame: int, scene: dict, batch: MassBatch):
        leaving = 1.0 - ramp(frame, scene["last"] - 6, 6)
        self._draw_header(layer, frame, scene["first"] + 2, scene["eyebrow"], scene["title"], scene["sub"], leaving)

        step_idx, (step_start, item_idx) = max(
            ((i, s) for i, s in enumerate(MORPH_STEPS) if s[0] <= frame), key=lambda pair: pair[1][0]
        )
        it = batch.items[item_idx]
        a = ease_out(ramp(frame, scene["first"] + 4, 10)) * leaving
        flash = 1.0 - clamp01((frame - step_start) / 14.0)
        y0 = 1480

        # Active SEED headline + live mutation counter
        seed_str = f"SEED #{it['seed']}"
        sw, _ = self.type.measure(seed_str, "stat")
        seed_col = tuple(int(lerp(c0, c1, flash)) for c0, c1 in zip(C_WHITE, C_MINT))
        self.type.put(layer, seed_str, "stat", seed_col, LEFT, y0, a)
        self.type.put(layer, f"TIMBERWRIGHT  ·  {it['members']:,} MEMBERS", "stat_sm", C_GREEN, LEFT + sw + 20, y0 + 20, a)
        self.type.put(layer, f"{step_idx + 1:02d} / {len(MORPH_STEPS):02d}", "index", C_SAGE, RIGHT, y0 + 20, a, align="right")

        # Live mutating architectural attribute pills
        st_lbl = "1 STOREY" if it["levels"] == 1 else f"{it['levels']} STOREYS"
        px = LEFT
        px = self._draw_pill(layer, px, y0 + 78, it["variant"], 1.0, accent=C_GREEN, alpha=a)
        px = self._draw_pill(layer, px, y0 + 78, f"{st_lbl} · {it['bays']} BAYS", 1.0, accent=C_AMBER, alpha=a)
        self._draw_pill(layer, px, y0 + 78, "QUALITY PASS", 1.0, accent=C_GREEN, alpha=a)

        self.type.put(
            layer,
            f"{it['footprint']}  ·  {it['roof_mat']}  ·  BRACE {it['brace']}  ·  {it['tonnes']:,} TONNES",
            "sub", C_SAGE, LEFT, y0 + 142, a,
        )

    def _draw_mass_grid(self, layer: Image.Image, frame: int, scene: dict, batch: MassBatch):
        draw = ImageDraw.Draw(layer)
        leaving = 1.0 - ramp(frame, scene["last"] - 6, 6)
        self._draw_header(layer, frame, scene["first"] + 2, scene["eyebrow"], scene["title"], scene["sub"], leaving)

        # Count how many of the 15 unique randomized houses have landed
        landed = sum(1 for i in range(len(batch.items)) if frame >= GRID_DROP_START + i * GRID_DROP_STEP)
        a = ease_out(ramp(frame, scene["first"] + 4, 10)) * leaving
        y0 = 1468

        # CLI batch command pill
        cmd_a = ease_out(ramp(frame, 2650, 10)) * leaving
        if cmd_a > 0.01:
            draw.rounded_rectangle((LEFT, y0, RIGHT, y0 + 50), radius=10,
                                   fill=(7, 18, 12, int(225 * cmd_a)), outline=C_GREEN + (int(220 * cmd_a),), width=2)
            self.type.put(layer, MASS_CLI_CMD, "code_sm", C_MINT, LEFT + 18, y0 + 13, cmd_a)

        # Live batch building counter + progress bar
        cy = y0 + 68
        cnt_str = f"{landed:02d}"
        cw, _ = self.type.measure(cnt_str, "stat")
        self.type.put(layer, cnt_str, "stat", C_WHITE, LEFT, cy, a)
        self.type.put(layer, f"OF {len(batch.items)} UNIQUE BUILDINGS", "stat_sm", C_GREEN, LEFT + cw + 18, cy + 20, a)
        self.type.put(layer, f"{landed}/{len(batch.items)} PASS", "stat_sm", C_AMBER, RIGHT, cy + 20, a, align="right")

        bar_y = cy + 74
        draw.rounded_rectangle((LEFT, bar_y, RIGHT, bar_y + 10), radius=5,
                               fill=(14, 34, 22, int(200 * a)), outline=C_RULE + (int(220 * a),), width=1)
        if landed > 0:
            fw = max(12, int((RIGHT - LEFT) * landed / len(batch.items)))
            draw.rounded_rectangle((LEFT, bar_y, LEFT + fw, bar_y + 10), radius=5, fill=C_GREEN + (int(255 * a),))

        # Batch artifact strip
        self.type.put(
            layer,
            "SERIES.GLB  ·  15 INDIVIDUAL GLB  ·  BATCH.MANIFEST.JSON",
            "sub", C_SAGE, LEFT, bar_y + 28, a,
        )

    def _draw_outro(self, layer: Image.Image, frame: int):
        draw = ImageDraw.Draw(layer)
        at = OUTRO_IN
        def r_off(t0: int) -> tuple[float, float]:
            a = ease_out(ramp(frame, t0, 14))
            return a, 22.0 * (1.0 - a)

        # BLOCKN wordmark + emerald horizon bar
        a, dy = r_off(at["mark"])
        self.type.put(layer, "BLOCKN", "mark", C_WHITE, W * 0.5, 404 + dy, a, align="centre")
        bar_w = 240.0 * ease_out(ramp(frame, at["mark"] + 6, 16))
        if bar_w > 1:
            draw.line((W * 0.5 - bar_w, 578, W * 0.5 + bar_w, 578), fill=C_GREEN + (int(255 * a),), width=4)

        a, dy = r_off(at["stats"])
        self.type.put(
            layer,
            f"{self.facts['products']} PRODUCTS  ·  {self.facts['architects']} ARCHITECTS  ·  29 SUBSTANCES",
            "eyebrow", C_MINT, W * 0.5, 610 + dy, a, align="centre",
        )

        # CLI pill
        a, dy = r_off(at["cli"])
        cw, _ = self.type.measure(CLI_CMD, "code")
        half = cw * 0.5 + 32
        cy = 684 + dy
        if a > 0.01:
            draw.rounded_rectangle((int(W * 0.5 - half), int(cy), int(W * 0.5 + half), int(cy + 68)), radius=12,
                                   fill=(7, 18, 12, int(225 * a)), outline=C_RULE + (int(255 * a),), width=2)
            self.type.put(layer, CLI_CMD, "code", C_GREEN, W * 0.5, cy + 17, a, align="centre")

        # Hairline divider before CTA
        div_a = ease_out(ramp(frame, at["ask"] - 6, 14))
        if div_a > 0.01:
            dw = 340.0 * div_a
            draw.line((W * 0.5 - dw, 822, W * 0.5 + dw, 822), fill=C_RULE + (int(255 * div_a),), width=2)

        # AI Model Benchmark CTA synced to v8
        a, dy = r_off(at["ask"])
        self.type.put(layer, "BENCHMARK YOUR FAVORITE AI MODEL", "cta_head", C_WHITE, W * 0.5, 874 + dy, a, align="centre")

        a, dy = r_off(at["line1"])
        self.type.put(layer, "Have it add a new architectural tradition to BlockN,", "cta_sub", C_SAGE,
                      W * 0.5, 966 + dy, a, align="centre")

        a, dy = r_off(at["line2"])
        self.type.put(layer, "and grow the open-source library for everyone.", "cta_sub", C_MINT,
                      W * 0.5, 1020 + dy, a, align="centre")

        # Highlighted GitHub repo badge
        a, dy = r_off(at["repo"])
        rw, _ = self.type.measure(REPO_URL, "repo")
        rhalf = rw * 0.5 + 42
        ry = 1138 + dy
        if a > 0.01:
            draw.rounded_rectangle((int(W * 0.5 - rhalf), int(ry), int(W * 0.5 + rhalf), int(ry + 84)), radius=14,
                                   fill=(0, 44, 20, int(235 * a)), outline=C_GREEN + (int(255 * a),), width=3)
            self.type.put(layer, REPO_URL, "repo", C_WHITE, W * 0.5, ry + 21, a, align="centre")

        self.type.put(layer, "OPEN SOURCE  ·  MIT  ·  DESKTOP WORKBENCH + CLI + AGENTS",
                      "footer", C_DIM, W * 0.5, 1300, a, align="centre")


# ------------------------------------------------------------------ Catalogue & Audio Pipeline
def catalogue_facts() -> dict:
    from cli import _architecture_products
    from core.architecture import list_architects
    return {"products": len(_architecture_products()), "architects": len(list_architects())}


def cue_list() -> list[dict]:
    cues = [
        {"cue": "score-act1", "frame": 0, "take": "score_bed_a", "align": "start"},
        {"cue": "score-act2", "frame": 1500, "take": "score_bed_b", "align": "start"},
        {"cue": "score-outro", "frame": OUTRO_FIRST, "take": "score_bed_c", "align": "start"},
        {"cue": "landings-hall", "first_frame": 4, "last_frame": 258},
        {"cue": "done-hall", "frame": 258, "take": "land_a", "align": "attack"},
        {"cue": "scan-hall", "frame": 378, "take": "scan_sweep", "align": "attack"},
        {"cue": "whoosh-macro", "frame": 600, "take": "whoosh_a", "align": "peak"},
        {"cue": "pin-brackets", "frame": 686, "take": "pin_tick", "align": "attack"},
        {"cue": "pin-corners", "frame": 756, "take": "pin_tick", "align": "attack"},
        {"cue": "open-roof", "frame": 802, "take": "roof_peel_clean", "align": "attack"},
        {"cue": "whoosh-palace", "frame": 900, "take": "whoosh_b", "align": "peak"},
        {"cue": "pin-eaves", "frame": 992, "take": "pin_tick", "align": "attack"},
        {"cue": "pin-palace-brackets", "frame": 1028, "take": "pin_tick", "align": "attack"},
        {"cue": "pass-palace", "frame": 1085, "take": "pass_a", "align": "attack"},
        {"cue": "riser-dravidian", "frame": 1170, "take": "riser_impact", "align": "peak"},
        {"cue": "landings-gate", "first_frame": 1182, "last_frame": 1458},
        {"cue": "pin-passage", "frame": 1392, "take": "pin_tick", "align": "attack"},
        {"cue": "done-gate", "frame": 1458, "take": "land_b", "align": "attack"},
        {"cue": "scan-gate", "frame": 1504, "take": "scan_sweep", "align": "attack"},
        {"cue": "pin-talas", "frame": 1514, "take": "pin_tick", "align": "attack"},
        {"cue": "pin-hara", "frame": 1560, "take": "pin_tick", "align": "attack"},
        {"cue": "pin-kalasa", "frame": 1612, "take": "pin_tick", "align": "attack"},
        {"cue": "pass-gate-0", "frame": 1648, "take": "pass_a", "align": "attack"},
        {"cue": "pass-gate-1", "frame": 1666, "take": "pass_b", "align": "attack"},
        {"cue": "pass-gate-2", "frame": 1684, "take": "pass_a", "align": "attack"},
        {"cue": "whoosh-vimana", "frame": 1755, "take": "whoosh_a", "align": "peak"},
        {"cue": "whoosh-reel", "frame": 1905, "take": "whoosh_b", "align": "peak"},
        {"cue": "whoosh-morph", "frame": MASS_FIRST, "take": "whoosh_a", "align": "peak"},
        {"cue": "whoosh-grid", "frame": GRID_FIRST, "take": "whoosh_b", "align": "peak"},
        {"cue": "pass-mass-done", "frame": GRID_DROP_START + 14 * GRID_DROP_STEP + 6, "take": "pass_a", "align": "attack"},
        {"cue": "logo-hit", "frame": OUTRO_IN["mark"], "take": "logo", "align": "attack"},
    ]
    # Crisp precision cuts across the 8-product PBR showcase reel
    cut_f = REEL_FIRST
    for idx, dur in enumerate(REEL_LENGTHS[:-1], start=1):
        cut_f += dur
        cues.append({"cue": f"cut-{idx}", "frame": cut_f, "take": ("cut_a", "cut_b")[idx % 2], "align": "attack"})
    # Crisp UI/cut clicks on each of the 5 live seed mutations
    for idx, (mf, _) in enumerate(MORPH_STEPS[1:], start=1):
        cues.append({"cue": f"morph-{idx}", "frame": mf, "take": ("cut_a", "cut_b")[idx % 2], "align": "attack"})
    # Tactile wooden touchdown pops as each of the 15 randomized houses drops onto the 3x5 lot
    for idx in range(15):
        cues.append({
            "cue": f"drop-{idx + 1:02d}",
            "frame": GRID_DROP_START + idx * GRID_DROP_STEP,
            "take": ("wood_1", "wood_2", "wood_3")[idx % 3],
            "align": "attack",
        })
    # Voiceover takes
    for tid, frame, _, _ in NARRATION:
        cues.append({"cue": f"voice-{tid}", "frame": frame, "take": tid, "align": "start"})
    for c in cues:
        c["seconds"] = round(c.get("frame", c.get("first_frame")) / FPS, 4)
    return sorted(cues, key=lambda item: item["seconds"])


def _audio_path(name: str) -> Path:
    if name in {tid for tid, _, _, _ in NARRATION}:
        return OUT / "narration_v2" / f"{name}.wav"
    v2 = OUT / "sfx_v2" / f"{name}.wav"
    if v2.exists():
        return v2
    return OUT / "sfx" / f"{name}.wav"


def load_take(name: str) -> tuple[np.ndarray, int, int]:
    path = _audio_path(name)
    with wave.open(str(path)) as handle:
        if handle.getframerate() != SR:
            raise RuntimeError(f"{name}: {handle.getframerate()} Hz, expected {SR}")
        data = np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(np.float64) / 32768.0
        data = data.reshape(-1, handle.getnchannels())
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    env = np.sqrt(np.convolve(data.mean(axis=1) ** 2, np.ones(96) / 96.0, mode="same"))
    attack = int(np.argmax(env > 0.5 * env.max())) if env.max() > 0 else 0
    peak = int(np.argmax(env))
    return data, attack, peak


def place(track: np.ndarray, data: np.ndarray, at: int):
    start, end = max(at, 0), min(at + len(data), len(track))
    if end > start:
        track[start:end] += data[start - at : end - at]


def loop_crossfade(stem: np.ndarray, total_samples: int, xfade_s: float = 1.8) -> np.ndarray:
    """Seamlessly tile a musical stem with equal-power crossfades and zero click/seam."""
    xf = int(xfade_s * SR)
    step = len(stem) - xf
    out = np.zeros((total_samples, 2), dtype=np.float64)
    pos = 0
    first = True
    while pos < total_samples:
        chunk = stem.copy()
        if not first:
            fade_in = np.sin(np.linspace(0.0, math.pi * 0.5, xf))[:, None]
            chunk[:xf] *= fade_in
        if pos + len(chunk) < total_samples:
            fade_out = np.cos(np.linspace(0.0, math.pi * 0.5, xf))[:, None]
            chunk[-xf:] *= fade_out
        place(out, chunk, pos)
        pos += step
        first = False
    return out


def fetch_audio():
    """Fetch any missing ElevenLabs narration takes or SFX stems."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY is required for --fetch-audio")
    nar = OUT / "narration_v2"
    nar.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for idx, (tid, _, _, text) in enumerate(NARRATION):
        wav = nar / f"{tid}.wav"
        if wav.exists():
            continue
        payload = {"text": text, "model_id": VOICE_MODEL, "voice_settings": VOICE_SETTINGS, "seed": SEED + idx}
        if idx > 0:
            payload["previous_text"] = NARRATION[idx - 1][3]
        if idx + 1 < len(NARRATION):
            payload["next_text"] = NARRATION[idx + 1][3]
        req = urllib.request.Request(
            f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/with-timestamps?output_format=mp3_44100_128",
            data=json.dumps(payload).encode(),
            headers={"xi-api-key": key, "Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        mp3 = nar / f"{tid}.mp3"
        mp3.write_bytes(base64.b64decode(resp["audio_base64"]))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3), "-ar", str(SR), "-ac", "1", str(wav)], check=True)
        fetched += 1
    print(f"audio: {fetched} new takes fetched", flush=True)


def mix(cues: list[dict], film: Film):
    """Mix narration, bespoke 3-act architectural score, and frame-aligned SFX/Foley."""
    count = int((LAST + 1) / FPS * SR)
    t = np.arange(count) / SR
    voice = np.zeros((count, 2), dtype=np.float64)
    effects = np.zeros((count, 2), dtype=np.float64)
    transients = np.zeros(count, dtype=np.float64)

    trim_map = {tid: trim_s for tid, _, trim_s, _ in NARRATION}
    peaks = {
        "land_a": 0.44, "land_b": 0.46,
        "pass_a": 0.16, "pass_b": 0.16,
        "pin_tick": 0.16,
        "whoosh_a": 0.24, "whoosh_b": 0.24,
        "cut_a": 0.20, "cut_b": 0.20,
        "scan_sweep": 0.28,
        "roof_peel_clean": 0.30,
        "riser_impact": 0.42,
        "logo": 0.48,
        "wood_1": 0.22, "wood_2": 0.22, "wood_3": 0.22,
    }

    for cue in cues:
        name = cue.get("take")
        if not name or name.startswith("score_bed_"):
            continue
        data, attack, peak = load_take(name)
        if name.startswith("v"):
            trim_s = trim_map.get(name)
            if trim_s is not None:
                n_keep = min(len(data), int(trim_s * SR))
                data = data[:n_keep].copy()
                fade_n = min(int(0.06 * SR), len(data))
                data[-fade_n:] *= np.linspace(1.0, 0.0, fade_n)[:, None]
            active = data[np.abs(data[:, 0]) > 0.01]
            rms = float(np.sqrt(np.mean(active ** 2))) if len(active) else 0.1
            place(voice, data * (10 ** (-18.5 / 20.0) / rms), int(cue["seconds"] * SR))
            continue
        data = data * (peaks[name] / max(float(np.abs(data).max()), 1e-6))
        offset = {"attack": attack, "peak": peak, "start": 0}[cue["align"]]
        place(effects, data, int(cue["seconds"] * SR) - offset)
        if cue["cue"].startswith(("done-", "cut-", "logo-hit")):
            place(transients[:, None], data[:, :1], int(cue["seconds"] * SR) - offset)

    # Tactile wood & stone block landings synced to member counts during build scenes
    rng = np.random.default_rng(SEED)
    for scene in SCENES:
        if scene["key"] not in LANDINGS:
            continue
        shots = []
        for s_name in LANDINGS[scene["key"]]:
            d, att, _ = load_take(s_name)
            d = d[max(att - 240, 0) :][: int(0.20 * SR)]
            d = d * np.minimum(1.0, np.linspace(4.0, 0.0, len(d)))[:, None]
            shots.append(d / max(float(np.abs(d).max()), 1e-6))
        counts = film.model(scene["model"]).counts
        g0, g1 = scene["grow"]
        for f in range(g0 + 1, g1 + 1):
            laid = counts[f] - counts[f - 1]
            hits = min(3, int(laid > 0) + int(laid >= 25) + int(laid >= 95))
            for h_i in range(hits):
                sh = shots[rng.integers(len(shots))]
                rate = rng.uniform(0.90, 1.12)
                idx = np.arange(0, len(sh) - 1, rate)
                sh = np.stack([np.interp(idx, np.arange(len(sh)), sh[:, ch]) for ch in (0, 1)], axis=1)
                pan = rng.uniform(-0.35, 0.35)
                gain = 0.115 * rng.uniform(0.65, 1.0) * np.array([1.0 - pan, 1.0 + pan])
                place(effects, sh * gain, int((f + h_i / 3.0 + rng.uniform(0.0, 0.22)) / FPS * SR))

    # 3-act bespoke architectural score bed with smooth crossfades
    bed_a, _, _ = load_take("score_bed_a")
    bed_b, _, _ = load_take("score_bed_b")
    bed_c, _, _ = load_take("score_bed_c")

    def norm_rms(x: np.ndarray, target_db: float) -> np.ndarray:
        r = float(np.sqrt(np.mean(x ** 2)))
        return x * (10 ** (target_db / 20.0) / max(r, 1e-6))

    s_a = norm_rms(loop_crossfade(bed_a, int(52.5 * SR), 2.0), -25.0)
    s_b = norm_rms(loop_crossfade(bed_b, int(48.5 * SR), 2.0), -24.5)
    s_c = norm_rms(bed_c, -23.5)

    music = np.zeros((count, 2), dtype=np.float64)
    # Act 1: 0.0s .. 51.5s (fades out 49.5..51.5)
    n_a = min(len(s_a), int(51.5 * SR))
    s_a = s_a[:n_a].copy()
    xf = int(2.0 * SR)
    s_a[-xf:] *= np.cos(np.linspace(0.0, math.pi * 0.5, xf))[:, None]
    place(music, s_a, 0)

    # Act 2 (Dravidian PBR + 45-product reel + Seed/Mass-gen): 49.5s .. 97.2s
    outro_s = OUTRO_FIRST / FPS                                 # 96.0s
    xf2 = int(1.8 * SR)
    b_end_s = outro_s + 1.2
    n_b = min(len(s_b), int((b_end_s - 49.5) * SR))
    s_b = s_b[:n_b].copy()
    s_b[:xf] *= np.sin(np.linspace(0.0, math.pi * 0.5, xf))[:, None]
    s_b[-xf2:] *= np.cos(np.linspace(0.0, math.pi * 0.5, xf2))[:, None]
    place(music, s_b, int(49.5 * SR))

    # Act 3 Outro: 95.4s .. 107.5s
    outro_at = int((outro_s - 0.6) * SR)
    n_c = min(len(s_c), count - outro_at)
    s_c = s_c[:n_c].copy()
    s_c[:xf2] *= np.sin(np.linspace(0.0, math.pi * 0.5, xf2))[:, None]
    place(music, s_c, outro_at)

    # Fast O(N) boxcar moving-average voice detector for smooth sidechain ducking
    k = int(0.32 * SR)
    raw_sp = (np.abs(voice[:, 0]) > 0.006).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(np.pad(raw_sp, (k // 2, k - 1 - k // 2)))))
    speech_env = (cs[k:] - cs[:-k]) / float(k)
    duck_music = 1.0 - 0.58 * np.clip(speech_env * 2.8, 0.0, 1.0)
    duck_sfx = 1.0 - 0.30 * np.clip(speech_env * 2.8, 0.0, 1.0)

    audio = voice + effects * duck_sfx[:, None] + music * duck_music[:, None]
    fades = np.clip(t / 0.15, 0.0, 1.0) * np.clip(((LAST + 1) / FPS - t) / 1.1, 0.0, 1.0)
    audio *= fades[:, None]
    gain = 0.62 / max(float(np.max(np.abs(audio))), 1e-6)
    audio, transients = audio * gain, transients * fades * gain

    with wave.open(str(OUT / "mix.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
    return audio, transients


def sync_report(decoded: np.ndarray, stem: np.ndarray, cues: list[dict]) -> list[dict]:
    report = []
    for cue in cues:
        if not cue["cue"].startswith(("done-", "cut-", "logo-hit")):
            continue
        at = int(cue["seconds"] * SR)
        ref = stem[at - 2400 : at + 4800]
        norm = float(np.linalg.norm(ref))
        if norm < 1e-6:
            continue
        ref = ref / norm
        lag = max(
            range(-480, 481),
            key=lambda s: float(np.dot(decoded[at - 2400 + s : at + 4800 + s], ref)),
        )
        report.append({"cue": cue["cue"], "frame": cue["frame"], "lag_ms": round(lag / SR * 1000.0, 2)})
    return report


def render_movie(film: Film, target: str) -> Image.Image:
    started = time.time()
    log = open(OUT / "review" / "ffmpeg.log", "wb")
    writer = subprocess.Popen(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
            "-crf", "17", "-pix_fmt", "yuv420p", "-profile:v", "high", target,
        ],
        stdin=subprocess.PIPE,
        stderr=log,
    )
    cover = None
    for frame in range(LAST + 1):
        img = film.render(frame)
        if frame == 428:                                        # hero cover: mid-scan emerald reveal on Timber Hall!
            cover = img.copy()
        writer.stdin.write(img.tobytes())
        if frame % 150 == 0 or frame == LAST:
            print(f"frame {frame}/{LAST} at {time.time() - started:.1f}s", flush=True)
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("ffmpeg failed; see review/ffmpeg.log")
    log.close()
    print(f"video: {LAST + 1} frames in {time.time() - started:.1f}s", flush=True)
    return cover


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stills", nargs="*", type=int, help="render only these global frames + contact sheet")
    parser.add_argument("--output", default=str(OUT), help="output directory")
    parser.add_argument("--fetch-audio", action="store_true", help="fetch missing ElevenLabs narration takes")
    parser.add_argument("--remix", action="store_true", help="remix sound onto existing rendered video")
    args = parser.parse_args()
    OUT = Path(args.output)
    review = OUT / "review"
    review.mkdir(parents=True, exist_ok=True)
    if args.fetch_audio:
        fetch_audio()
        return

    film = Film()
    if args.stills:
        paths = []
        for f in args.stills:
            p = review / f"still-{f:04d}.png"
            film.render(f).save(p)
            paths.append(str(p))
            print(f"rendered still {f:04d}", flush=True)
        sh = sheet(paths, str(review / "stills-sheet.png"), columns=min(4, len(paths)))
        print("stills sheet:", sh)
        return

    name = "blocks-showcase"
    silent, final = str(OUT / f"{name}-silent.mp4"), str(OUT / f"{name}.mp4")
    if args.remix:
        if os.path.exists(final):
            os.replace(final, silent)
        for s in SCENES:
            if s.get("grow"):
                film.model(s["model"])
    else:
        cover = render_movie(film, silent)
        cover.save(OUT / "cover.png")

    cues = cue_list()
    with open(OUT / "cues.json", "w") as h:
        json.dump(
            {
                "fps": FPS,
                "frames": LAST + 1,
                "duration_s": round((LAST + 1) / FPS, 3),
                "note": "sample position = frame / fps; align names attack, peak, or start",
                "cues": cues,
            },
            h,
            indent=2,
        )
    _, transients = mix(cues, film)
    mux(silent, str(OUT / "mix.wav"), final)

    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-count_frames",
                "-show_entries", "stream=codec_type,codec_name,profile,width,height,pix_fmt,avg_frame_rate,nb_read_frames,sample_rate,channels,duration",
                "-of", "json", final,
            ]
        )
    )
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    cuts = {s["first"] for s in SCENES if s["first"] > 0}
    scan_res = scan(final, cuts)
    blacks, jumps = scan_res["near_black"], scan_res["largest_changes"]

    # Extract encoded inspection stills across every act, scan, cutaway, reel cut, randomization/mass-gen, and ending
    sample_frames = [
        0, 60, 180, 265,
        340, 425, 540,
        695, 768, 845,
        920, 1050, 1150,
        1170, 1260, 1405, 1465,
        1545, 1665, 1730, 1825,
        1935, 1998, 2060, 2124, 2188, 2250, 2315, 2380,
        2445, 2490, 2560, 2630,
        2690, 2740, 2820, 2870,
        2915, 3030, 3155, 3224,
    ]
    select_expr = "+".join(f"eq(n\\,{f})" for f in sample_frames)
    for old in review.glob("encoded-*.png"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", final, "-vf", f"select={select_expr}", "-fps_mode", "passthrough",
         str(review / "encoded-%02d.png")],
        check=True,
    )
    encoded = sorted(str(p) for p in review.glob("encoded-*.png"))
    sheets = [
        Path(sheet(encoded[i : i + 8], str(review / f"encoded-sheet-{i // 8 + 1}.png"), columns=4)).name
        for i in range(0, len(encoded), 8)
    ]

    report = {
        "file": Path(final).name,
        "bytes": os.path.getsize(final),
        "video": {k: video.get(k) for k in ("codec_name", "profile", "width", "height", "pix_fmt", "avg_frame_rate", "duration", "nb_read_frames")},
        "audio": {k: audio.get(k) for k in ("codec_name", "sample_rate", "channels", "duration")},
        "matches_brief": {
            "width": video["width"] == W,
            "height": video["height"] == H,
            "frames": int(video["nb_read_frames"]) == LAST + 1,
            "fps": video["avg_frame_rate"] == f"{FPS}/1",
            "duration_s": abs(float(video["duration"]) - (LAST + 1) / FPS) < 0.05,
        },
        "catalogue": film.facts,
        "builds": {
            m.product: {"members": len(m.blocks), "tonnes": m.metadata["tonnes_raw"], "quality": m.result.quality.status}
            for m in film.models.values()
        },
        "last_frame_is_black": bool(blacks and blacks[-1][1] == LAST),
        "audio_length_matches_video": abs(float(audio["duration"]) - float(video["duration"])) < 0.08,
        "loudness": loudness(final),
        "transient_lag_in_delivered_file": sync_report(decode(final), transients, cues),
        "near_black_frames": blacks,
        "largest_frame_changes": jumps[:8],
        "review_sheets": sheets,
    }
    with open(OUT / "verification.json", "w") as h:
        json.dump(report, h, indent=2)
    if os.path.exists(silent):
        os.remove(silent)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
