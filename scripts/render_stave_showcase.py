#!/usr/bin/env python3
"""Render a widescreen cinematic showcase video of the Nordic Stave Architect.

Combines BlockN's phosphor CAD assembly view with weathered PBR studio lighting,
animated camera choreography (orbits, crane rises, low-angle sweeps), an
open-roof cutaway reveal of the inner stave frame and scissor braces, and an
original synthesized Nordic modal ambient score.

Usage:
    python3 scripts/render_stave_showcase.py
    python3 scripts/render_stave_showcase.py --output builds/stave-showcase
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv

W, H = 1280, 720
FPS = 30
LAST = 734          # 24.5 seconds at 30 fps
FADE_OUT = 708
SR = 48000

BRIGHT = (0, 255, 110)
SOFT = (155, 225, 180)
WARM = (232, 212, 178)
DIM = (115, 165, 135)
RULE = (42, 92, 62)

FONT_DIRS = (
    "/usr/share/fonts/TTF", "/usr/share/fonts/truetype/jetbrains-mono",
    "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/noto", "/usr/share/fonts/liberation",
    "/usr/local/share/fonts", "~/.local/share/fonts",
)
FONTS = {
    "medium": ("JetBrainsMono-Medium.ttf", "DejaVuSansMono.ttf", "NotoSansMono-Regular.ttf", "LiberationMono-Regular.ttf"),
    "bold": ("JetBrainsMono-ExtraBold.ttf", "DejaVuSansMono-Bold.ttf", "NotoSansMono-Bold.ttf", "LiberationMono-Bold.ttf"),
}


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


def ease(v: float) -> float:
    u = clamp01(v)
    return u * u * (3.0 - 2.0 * u)


def ease_out(v: float) -> float:
    u = clamp01(v)
    return 1.0 - (1.0 - u) ** 3


def fade_in(frame: int, start: int, span: int = 12) -> float:
    return clamp01((frame - start) / span)


def find_font(weight: str) -> Path:
    for name in FONTS[weight]:
        for folder in FONT_DIRS:
            p = Path(folder).expanduser() / name
            if p.is_file():
                return p
    raise SystemExit(f"No monospace font found for {weight}")


def trimesh_to_pv(mesh, with_face_rgb: bool = False) -> pv.PolyData:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), faces)).ravel()
    pd = pv.PolyData(verts, cells)
    if with_face_rgb:
        from core.architecture.surfaces import face_colors
        fc = face_colors(mesh)
        if len(fc) == len(faces):
            pd.cell_data["RGB"] = np.asarray(fc, dtype=np.uint8)
    return pd


def build_pbr_plotter() -> pv.Plotter:
    """Create a moody Nordic studio offscreen PyVista plotter."""
    pl = pv.Plotter(off_screen=True, window_size=(W, H))
    pl.set_background("#0c1110", top="#182321")
    pl.enable_anti_aliasing("fxaa")
    pl.remove_all_lights()
    # Warm low northern sun key light
    key = pv.Light(position=(26, 22, -24), focal_point=(0, 6, 0), color="#ffeed6", intensity=1.15)
    # Cool fjord sky fill light
    fill = pv.Light(position=(-24, 18, 22), focal_point=(0, 6, 0), color="#8fb4c9", intensity=0.55)
    # Rim light from behind/above to separate dragon finials and takrytter silhouette
    rim = pv.Light(position=(0, 28, 30), focal_point=(0, 8, 0), color="#d2e8dc", intensity=0.45)
    pl.add_light(key)
    pl.add_light(fill)
    pl.add_light(rim)
    return pl


def build_cad_plotter() -> pv.Plotter:
    """Create a BlockN phosphor CAD offscreen PyVista plotter."""
    pl = pv.Plotter(off_screen=True, window_size=(W, H))
    pl.set_background("#040806", top="#09130e")
    pl.enable_anti_aliasing("fxaa")
    pl.remove_all_lights()
    pl.add_light(pv.Light(position=(18, 26, -22), focal_point=(0, 6, 0), color="#b8ffd4", intensity=1.0))
    pl.add_light(pv.Light(position=(-20, 12, 20), focal_point=(0, 6, 0), color="#3aff88", intensity=0.45))
    return pl


def add_ground_grid(pl: pv.Plotter, mode: str, radius: float = 22.0) -> None:
    """Subtle dark studio plinth / phosphor CAD grid on Y=0."""
    if mode == "cad":
        plane = pv.Plane(center=(0, -0.02, 0), direction=(0, 1, 0), i_size=radius * 2, j_size=radius * 2,
                         i_resolution=22, j_resolution=22)
        pl.add_mesh(plane, style="wireframe", color="#0c3a20", line_width=1.0, lighting=False)
    else:
        disc = pv.Disc(center=(0, -0.02, 0), inner=0.0, outer=radius, normal=(0, 1, 0), r_res=1, c_res=48)
        pl.add_mesh(disc, color="#141a18", roughness=0.92, metallic=0.0, smooth_shading=False)


def render_blocks(pl: pv.Plotter, blocks, mode: str, *, hide_roof: bool = False) -> None:
    """Populate a plotter with a slice of blocks in either 'cad' or 'pbr' mode."""
    from core.architecture.substances import SUBSTANCES
    pl.clear_actors()
    add_ground_grid(pl, mode)
    if not blocks:
        return

    if mode == "cad":
        # Merge into one PolyData for fast, razor-sharp emerald wireframe + dark core rendering.
        meshes = []
        for b in blocks:
            if hide_roof and b.parameters.get("kind") == "roof":
                continue
            meshes.append(trimesh_to_pv(b.mesh))
        if meshes:
            merged = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
            pl.add_mesh(
                merged,
                color="#071e12",
                show_edges=True,
                edge_color="#00ff6e",
                line_width=1.15,
                opacity=0.95,
                ambient=0.35,
                diffuse=0.65,
            )
        return

    # PBR mode: group meshes by substance_id and keep BlockN's real per-face weathered
    # RGB from core/architecture/surfaces.py (8 finish tones, rain-shadow shelter, biofilm,
    # soot, silvering, and base grime) — no flat colour overrides or harsh edge lines.
    groups: dict[str, list] = {}
    for b in blocks:
        if hide_roof and b.parameters.get("kind") == "roof":
            continue
        sub_id = b.parameters.get("substance") or "softwood"
        pd = trimesh_to_pv(b.mesh, with_face_rgb=True)
        if "RGB" not in pd.cell_data:
            col = np.asarray(b.color[:3], dtype=np.uint8)
            pd.cell_data["RGB"] = np.tile(col, (pd.n_cells, 1))
        groups.setdefault(sub_id, []).append(pd)

    for sub_id, parts in groups.items():
        merged = parts[0].merge(parts[1:]) if len(parts) > 1 else parts[0]
        sub = SUBSTANCES.get(sub_id)
        rough = float(sub.roughness) if sub else 0.72
        metal = float(sub.metallic) if sub else 0.0
        pl.add_mesh(
            merged,
            scalars="RGB",
            rgb=True,
            preference="cell",
            pbr=True,
            metallic=metal,
            roughness=rough,
            smooth_shading=False,
            show_edges=False,
        )


def set_camera(pl: pv.Plotter, center, dist: float, yaw_deg: float, pitch_deg: float, fov: float = 34.0) -> None:
    """Position camera spherically around ``center`` with Y up."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cx, cy, cz = center
    x = cx + dist * math.cos(pitch) * math.sin(yaw)
    y = cy + dist * math.sin(pitch)
    z = cz - dist * math.cos(pitch) * math.cos(yaw)
    pl.camera.focal_point = (cx, cy, cz)
    pl.camera.position = (x, max(0.6, y), z)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.view_angle = fov


def synthesize_score(out_wav: Path) -> None:
    """Original synthesized Nordic modal ambient score (D Dorian open fifths + chimes)."""
    n = int((LAST + 1) / FPS * SR)
    t = np.arange(n, dtype=np.float64) / SR
    audio = np.zeros(n, dtype=np.float64)

    # Scene chords (Hz): rich open-fifth Nordic modal harmonies
    sections = [
        (0, 180, (73.42, 110.00, 146.83, 220.00, 293.66)),      # D2 A2 D3 A3 D4 (open fifth)
        (180, 390, (73.42, 110.00, 174.61, 220.00, 329.63)),    # Dm9 warm reveal
        (390, 510, (65.41, 98.00, 146.83, 196.00, 293.66)),     # Csus2 / G fifth
        (510, 660, (82.41, 123.47, 164.81, 246.94, 329.63)),    # E5 Nordic lift
        (660, LAST + 1, (73.42, 110.00, 146.83, 220.00, 329.63)),  # Dsus2 outro
    ]
    for f0, f1, chord in sections:
        s0, s1 = f0 / FPS, f1 / FPS
        env = np.clip((t - s0) / 0.85, 0.0, 1.0) * np.clip((s1 + 0.9 - t) / 1.1, 0.0, 1.0)
        for idx, freq in enumerate(chord):
            detune = 1.0 + 0.0014 * (idx - 2)
            audio += 0.042 * env * (
                np.sin(2 * np.pi * freq * t)
                + 0.48 * np.sin(2 * np.pi * freq * detune * t)
                + 0.14 * np.sin(4 * np.pi * freq * t)
            )

    # Subtle filtered wind/timber swell during the two assembly sequences
    rng = np.random.default_rng(73)
    noise = np.convolve(rng.normal(0, 1, n), np.ones(64) / 64, mode="same")
    for g0, g1 in ((10, 155), (518, 590)):
        t0, t1 = g0 / FPS, g1 / FPS
        swell = np.where((t >= t0) & (t <= t1), np.sin(np.pi * np.clip((t - t0) / (t1 - t0), 0, 1)), 0.0)
        audio += 0.055 * noise * swell
        # Deep timber settlement thump at assembly completion
        tau = t - t1
        thump = np.where(tau >= 0, np.exp(-tau * 7.5) * np.sin(2 * np.pi * (55 * tau + 75 * (1 - np.exp(-tau * 22)) / 22)), 0.0)
        audio += 0.24 * thump

    # Soft crystalline chime on PBR scan reveal (frame 180), open-roof (frame 255), and outro (frame 664)
    for hit_frame, freq in ((180, 587.33), (255, 659.25), (664, 880.0)):
        tau = t - hit_frame / FPS
        chime = np.where(tau >= 0, np.exp(-tau * 2.4) * np.sin(2 * np.pi * freq * tau)
                         + 0.35 * np.exp(-tau * 5.0) * np.sin(2 * np.pi * freq * 2.0 * tau), 0.0)
        audio += 0.11 * chime

    # Master fade in / fade out and normalize
    audio *= np.clip(t / 0.35, 0.0, 1.0) * np.clip(((LAST + 1) / FPS - t) / 0.85, 0.0, 1.0)
    peak = np.max(np.abs(audio)) or 1.0
    audio = audio * (0.55 / peak)
    stereo = np.column_stack((audio, audio * 0.96)).astype(np.float32)
    pcm = (stereo.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


class Overlay:
    def __init__(self):
        self.f_eyebrow = ImageFont.truetype(str(find_font("medium")), 18)
        self.f_title = ImageFont.truetype(str(find_font("bold")), 38)
        self.f_sub = ImageFont.truetype(str(find_font("medium")), 19)
        self.f_stat = ImageFont.truetype(str(find_font("medium")), 17)
        self.f_big = ImageFont.truetype(str(find_font("bold")), 46)
        self.f_code = ImageFont.truetype(str(find_font("medium")), 22)

    def draw_hud(self, img: Image.Image, eyebrow: str, title: str, detail: str, stats: str,
                 alpha: float, badge: str | None = None, accent=BRIGHT) -> Image.Image:
        if alpha <= 0.01:
            return img
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        a = int(255 * alpha)
        # Subtle dark gradient vignette at top-left and bottom-left for crisp legibility
        d.rectangle((0, 0, W, 118), fill=(6, 10, 9, int(155 * alpha)))
        d.rectangle((0, H - 86, W, H), fill=(6, 10, 9, int(165 * alpha)))
        d.line((44, 112, W - 44, 112), fill=RULE + (a,), width=1)
        d.text((44, 24), eyebrow, font=self.f_eyebrow, fill=DIM + (a,))
        d.text((44, 50), title, font=self.f_title, fill=accent + (a,))
        if badge:
            bw = d.textlength(badge, font=self.f_stat) + 28
            bx0 = W - 44 - bw
            d.rounded_rectangle((bx0, 44, W - 44, 80), radius=8,
                                fill=(12, 32, 22, int(210 * alpha)), outline=accent + (a,), width=2)
            d.text((bx0 + 14, 52), badge, font=self.f_stat, fill=accent + (a,))
        d.text((44, H - 64), stats, font=self.f_stat, fill=accent + (a,))
        d.text((44, H - 38), detail, font=self.f_sub, fill=SOFT + (a,))
        out = img.convert("RGBA")
        out.alpha_composite(layer)
        return out.convert("RGB")

    def draw_outro(self, img: Image.Image, frame: int) -> Image.Image:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rectangle((0, 0, W, H), fill=(5, 9, 8, 175))
        lines = [
            (664, self.f_big, BRIGHT, "NORDIC STAVE ARCHITECT", 210),
            (674, self.f_sub, WARM, "STAVE-CHURCH  /  STAVE-LOFT  /  4 VARIANTS", 280),
            (684, self.f_sub, SOFT, "7 MEASURED GEOMETRIC RULES  /  ALL QUALITY GATES PASS", 322),
        ]
        for start, font, col, text, y in lines:
            a = ease(fade_in(frame, start, 14))
            if a > 0:
                tw = d.textlength(text, font=font)
                d.text(((W - tw) / 2, y + 14 * (1 - a)), text, font=font, fill=col + (int(255 * a),))
        ca = ease(fade_in(frame, 692, 14))
        if ca > 0:
            cmd = "blockn make stave-church --variant borgund"
            tw = d.textlength(cmd, font=self.f_code)
            bx0, bx1 = (W - tw) / 2 - 28, (W + tw) / 2 + 28
            d.rounded_rectangle((bx0, 412, bx1, 466), radius=10,
                                fill=(8, 24, 16, int(210 * ca)), outline=BRIGHT + (int(220 * ca),), width=2)
            d.text(((W - tw) / 2, 426), cmd, font=self.f_code, fill=BRIGHT + (int(255 * ca),))
        out = img.convert("RGBA")
        out.alpha_composite(layer)
        return out.convert("RGB")


def scan_wipe(cad_img: np.ndarray, pbr_img: np.ndarray, progress: float) -> np.ndarray:
    """Emerald vertical laser-scan wipe transitioning from phosphor CAD to weathered PBR."""
    u = ease(progress)
    cut = int(round(W * u))
    out = cad_img.copy()
    if cut > 0:
        out[:, :cut] = pbr_img[:, :cut]
    if 2 < cut < W - 2:
        band = 14
        x0, x1 = max(0, cut - band), min(W, cut + band)
        xs = np.arange(x0, x1)
        glow = np.exp(-((xs - cut) / 4.2) ** 2)[:, None]
        stripe = np.array([0.0, 255.0, 120.0])[None, :]
        region = out[:, x0:x1].astype(np.float32)
        out[:, x0:x1] = np.clip(region * (1.0 - 0.55 * glow) + stripe * (0.85 * glow), 0, 255).astype(np.uint8)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Nordic Stave Architect showcase film.")
    parser.add_argument("--output", type=Path, default=ROOT / "builds" / "stave-showcase")
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    from core.architecture import BuildRequest, build

    print("Building Nordic Stave models (standard detail, weathered age=0.45)...", flush=True)
    t0 = time.time()
    models = {
        "borgund": build(BuildRequest(product="stave-church", variant="borgund", detail="standard", seed=73, age=0.45)),
        "single": build(BuildRequest(product="stave-church", variant="single-nave", detail="standard", seed=73, age=0.45)),
        "stabbur": build(BuildRequest(product="stave-loft", variant="stabbur", detail="standard", seed=73, age=0.45)),
    }
    for k, r in models.items():
        assert r.quality.status == "pass", f"{k} failed quality"
        print(f"  {k}: {len(r.scene.blocks)} blocks, bounds={[round(x, 1) for x in r.quality.metrics['bounds_size']]}, PASS", flush=True)

    pl_cad = build_cad_plotter()
    pl_pbr = build_pbr_plotter()
    hud = Overlay()

    # Track loaded state so we only rebuild VTK geometry when block count or cutaway changes!
    cad_state = None
    pbr_state = None

    def grab_cad(key: str, count: int, center, dist, yaw, pitch, fov=34.0, hide_roof=False):
        nonlocal cad_state
        st = (key, count, hide_roof)
        if cad_state != st:
            render_blocks(pl_cad, models[key].scene.blocks[:count], "cad", hide_roof=hide_roof)
            cad_state = st
        set_camera(pl_cad, center, dist, yaw, pitch, fov)
        pl_cad.render()
        return pl_cad.screenshot(return_img=True)

    def grab_pbr(key: str, count: int, center, dist, yaw, pitch, fov=34.0, hide_roof=False):
        nonlocal pbr_state
        st = (key, count, hide_roof)
        if pbr_state != st:
            render_blocks(pl_pbr, models[key].scene.blocks[:count], "pbr", hide_roof=hide_roof)
            pbr_state = st
        set_camera(pl_pbr, center, dist, yaw, pitch, fov)
        pl_pbr.render()
        return pl_pbr.screenshot(return_img=True)

    # Also save two crisp 1280x720 PBR posters up front.
    p1 = grab_pbr("borgund", len(models["borgund"].scene.blocks), (0, 8.2, 0.4), 44.0, -36.0, 18.0, 33.0)
    Image.fromarray(p1).save(out_dir / "poster-stave-church.png")
    p2 = grab_pbr("stabbur", len(models["stabbur"].scene.blocks), (0, 5.4, 0.0), 29.5, -34.0, 14.0, 33.0)
    Image.fromarray(p2).save(out_dir / "poster-stabbur.png")
    print("Saved PBR posters.", flush=True)

    n_borgund = len(models["borgund"].scene.blocks)
    n_single = len(models["single"].scene.blocks)
    n_stabbur = len(models["stabbur"].scene.blocks)

    print(f"Rendering {LAST + 1} frames at {W}x{H} ({FPS} fps)...", flush=True)
    for f in range(LAST + 1):
        if f < 180:
            # Scene 1: Borgund CAD assembly + rising orbit
            u = f / 179.0
            grow_u = ease(clamp01((f - 10) / 145.0))
            count = max(1, int(round(grow_u * n_borgund)))
            yaw = -42.0 + 30.0 * u
            pitch = 13.0 + 9.0 * ease(u)
            dist = 47.0 - 4.0 * ease(u)
            raw = grab_cad("borgund", count, (0, 8.2, 0.4), dist, yaw, pitch)
            alpha = fade_in(f, 4, 12) * (1.0 - fade_in(f, 170, 10))
            img = hud.draw_hud(
                Image.fromarray(raw),
                "BLOCKS / NEW ARCHITECT: NORDIC STAVE",
                "STAVE CHURCH  [BORGUND]",
                "granite svill, vertical pine staves, svalgang arcade, takrytter spire",
                f"{count:,} / {n_borgund:,} MEMBERS  |  PHOSPHOR CAD ASSEMBLY",
                alpha,
                badge="ASSEMBLING" if count < n_borgund else "QUALITY PASS",
            )
        elif f < 390:
            # Scene 2: Borgund emerald scan-wipe into weathered PBR + Open-Roof cutaway + 360-style sweep
            u = (f - 180) / 209.0
            yaw = -12.0 + 165.0 * u
            # Crane up slightly over the roof cutaway, then dip to view the eastern apse
            pitch = 20.0 + 7.0 * math.sin(math.pi * u)
            dist = 44.0 - 4.5 * math.sin(math.pi * u)
            cutaway = 255 <= f <= 328
            pbr_raw = grab_pbr("borgund", n_borgund, (0, 8.2, 0.4), dist, yaw, pitch, hide_roof=cutaway)
            if f < 214:
                cad_raw = grab_cad("borgund", n_borgund, (0, 8.2, 0.4), dist, yaw, pitch, hide_roof=False)
                raw = scan_wipe(cad_raw, pbr_raw, (f - 180) / 34.0)
            else:
                raw = pbr_raw
            alpha = fade_in(f, 184, 10) * (1.0 - fade_in(f, 380, 10))
            badge = "OPEN ROOF CUTAWAY" if cutaway else "WEATHERED PBR"
            detail = (
                "exposed vertical pine staves, head plates and St. Andrew's scissor braces"
                if cutaway else
                "tarred cedar shingles, carved oak dragon finials, semicircular eastern apse"
            )
            img = hud.draw_hud(
                Image.fromarray(raw),
                "BLOCKS / NORDIC STAVE ARCHITECT",
                "STAVE CHURCH  [BORGUND]",
                detail,
                f"{n_borgund:,} MEMBERS  |  7 MEASURED RULES  |  QUALITY PASS",
                alpha,
                badge=badge,
                accent=WARM if not cutaway else BRIGHT,
            )
        elif f < 510:
            # Scene 3: Single-Nave Stave Church low-angle PBR flyby
            u = (f - 390) / 119.0
            yaw = -52.0 + 72.0 * u
            pitch = 12.0 + 8.0 * u
            dist = 45.0 - 3.5 * ease(u)
            raw = grab_pbr("single", n_single, (0, 8.0, 0.4), dist, yaw, pitch)
            alpha = fade_in(f, 394, 10) * (1.0 - fade_in(f, 500, 10))
            img = hud.draw_hud(
                Image.fromarray(raw),
                "BLOCKS / NORDIC STAVE ARCHITECT",
                "STAVE CHURCH  [SINGLE-NAVE]",
                "unbroken steep gable nave, encircling svalgang gallery, ridge turret",
                f"{n_single:,} MEMBERS  |  GRANITE / PINE / OAK / CEDAR / IRON",
                alpha,
                badge="QUALITY PASS",
                accent=WARM,
            )
        elif f < 660:
            # Scene 4: Stabbur assembly from granite stabbe piers + low cantilever sweep in PBR
            u = (f - 510) / 149.0
            yaw = -48.0 + 82.0 * u
            pitch = 10.0 + 10.0 * ease(u)
            dist = 30.5 - 2.5 * math.sin(math.pi * u)
            if f < 588:
                grow_u = ease(clamp01((f - 516) / 68.0))
                count = max(1, int(round(grow_u * n_stabbur)))
                raw = grab_cad("stabbur", count, (0, 5.4, 0.0), dist, yaw, pitch)
                badge = "STABBE PIERS + CANTILEVER"
                stats = f"{count:,} / {n_stabbur:,} MEMBERS  |  PHOSPHOR CAD ASSEMBLY"
                accent = BRIGHT
            elif f < 614:
                cad_raw = grab_cad("stabbur", n_stabbur, (0, 5.4, 0.0), dist, yaw, pitch)
                pbr_raw = grab_pbr("stabbur", n_stabbur, (0, 5.4, 0.0), dist, yaw, pitch)
                raw = scan_wipe(cad_raw, pbr_raw, (f - 588) / 26.0)
                badge = "WEATHERED PBR"
                stats = f"{n_stabbur:,} MEMBERS  |  QUALITY PASS"
                accent = WARM
            else:
                raw = grab_pbr("stabbur", n_stabbur, (0, 5.4, 0.0), dist, yaw, pitch)
                badge = "WEATHERED PBR"
                stats = f"{n_stabbur:,} MEMBERS  |  QUALITY PASS"
                accent = WARM
            alpha = fade_in(f, 514, 10) * (1.0 - fade_in(f, 650, 10))
            img = hud.draw_hud(
                Image.fromarray(raw),
                "BLOCKS / NORDIC STAVE ARCHITECT",
                "STAVE LOFT  [STABBUR]",
                "six granite stabbe piers, cantilevered upper svalgang, external oak stair",
                stats,
                alpha,
                badge=badge,
                accent=accent,
            )
        else:
            # Scene 5: Outro over slow-orbiting Borgund church
            u = (f - 660) / max(1, LAST - 660)
            yaw = -34.0 + 14.0 * u
            raw = grab_pbr("borgund", n_borgund, (0, 8.2, 0.4), 45.0, yaw, 17.0)
            img = hud.draw_outro(Image.fromarray(raw), f)

        if f >= FADE_OUT:
            fade = ease((f - FADE_OUT + 1) / (LAST - FADE_OUT + 1))
            img = Image.blend(img, Image.new("RGB", (W, H), (0, 0, 0)), fade)

        img.save(frames_dir / f"frame_{f:04d}.png", compress_level=1)
        if f % 120 == 0 or f == LAST:
            print(f"  frame {f:4d}/{LAST} ({time.time() - t0:.1f}s)", flush=True)

    pl_cad.close()
    pl_pbr.close()

    wav_path = out_dir / "score.wav"
    print("Synthesizing Nordic ambient score...", flush=True)
    synthesize_score(wav_path)

    mp4_path = out_dir / "stave-showcase.mp4"
    print(f"Encoding {mp4_path}...", flush=True)
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-i", str(wav_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(mp4_path),
    ], check=True)
    print(f"Done! Video saved to {mp4_path} ({time.time() - t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
