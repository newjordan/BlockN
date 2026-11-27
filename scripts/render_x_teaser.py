"""Render the 55-second Blocks teaser for X from actual geometry and the actual app.

Every frame is derived from the global frame index (30 fps). Models are real
Blocks builds drawn in the viewport's phosphor feature-edge language; the
workbench scene drives the real Qt window offscreen; the terminal shows the real
`blockn make ... --bundle` receipt. Score and effects are synthesized here.

    python3 scripts/render_x_teaser.py --output builds/blocks-teaser
    python3 scripts/render_x_teaser.py --output /tmp/t --scenes s1,s2 --step 15   # stills only
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
# The workbench is captured in a child process that keeps the X display for
# the app's own Qt/VTK context; everything else renders headless through EGL.
CAPTURE = "--capture-workbench" in sys.argv
DISPLAY = os.environ.get("DISPLAY")
if not CAPTURE:
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
# Plotters below pass off_screen=True themselves; a global PYVISTA_OFF_SCREEN
# would leave the workbench's Qt viewport without an interactor.

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core.architecture import BuildRequest, build

W, H, FPS, TOTAL = 1920, 1080, 30, 1650
GREEN, DIM, RULE, PALE = (0, 255, 65), (0, 169, 44), (0, 74, 21), (190, 255, 205)
FONT_DIR = Path("/usr/share/fonts/TTF")
SAFE = 96
SCENES = {"s1": (0, 179), "s2": (180, 329), "s3": (330, 569), "s4": (570, 929),
          "s5": (930, 1229), "s6": (1230, 1439), "s7": (1440, 1649)}
CUES = []  # (frame, kind) -> synthesized effects aligned to visual events


@lru_cache(maxsize=None)
def font(size, bold=False):
    name = "JetBrainsMonoNerdFont-Bold.ttf" if bold else "JetBrainsMonoNerdFont-Regular.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.truetype("DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf", size)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)


def ramp(f, a, b):
    return ease((f - a) / max(1, b - a))


def cue(frame, kind):
    CUES.append((int(frame), kind))


# ---------------------------------------------------------------- grading
_ROWS = np.ones((H, 1, 1), np.float32)
_ROWS[2::3] = .86


def grade(img):
    """Phosphor bloom and faint scanlines, applied to every composed frame."""
    small = img.resize((W // 3, H // 3), Image.BILINEAR).filter(ImageFilter.GaussianBlur(4))
    glow = np.asarray(small.resize((W, H), Image.BILINEAR), np.float32)
    arr = np.asarray(img, np.float32) + .7 * glow
    arr *= _ROWS
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def text(draw, xy, s, size, fill=GREEN, bold=False, anchor="la"):
    draw.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def typed(s, f, start, cps=1.0):
    return s[:max(0, min(len(s), int((f - start) * cps)))]


def headline(draw, f, start, title, sub=None, y=SAFE - 10):
    a = ramp(f, start, start + 8)
    if a <= 0:
        return
    col = tuple(int(c * a) for c in GREEN)
    text(draw, (SAFE, y + int((1 - a) * 18)), title, 62, col, True)
    if sub:
        b = ramp(f, start + 6, start + 14)
        text(draw, (SAFE + 2, y + 84), sub, 28, tuple(int(c * b) for c in DIM))


def footer(draw, s, f, start):
    a = ramp(f, start, start + 10)
    draw.line((SAFE, H - 78, W - SAFE, H - 78), fill=tuple(int(c * a) for c in RULE), width=2)
    text(draw, (SAFE, H - 60), s, 24, tuple(int(c * a) for c in DIM))


def cursor(draw, x, y, press=0.0):
    pts = [(0, 0), (0, 30), (8, 23), (14, 36), (20, 33), (14, 21), (24, 21)]
    draw.polygon([(x + px, y + py) for px, py in pts], fill=GREEN, outline=(0, 0, 0))
    if press > 0:
        r = 10 + 34 * press
        a = int(255 * (1 - press))
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, a, int(a * .25)), width=3)


# ---------------------------------------------------------------- geometry
def build_scene(product, **kw):
    t = time.time()
    result = build(BuildRequest(product=product, seed=73, **kw))
    assert result.quality.status == "pass", (product, kw)
    print(f"  built {product} {kw} -> {len(result.scene.blocks)} members in {time.time()-t:.0f}s", flush=True)
    return result


def role(block):
    return block.parameters.get("component_role", "")


def group_actors(plotter, blocks):
    """One black occluding face actor plus green feature edges for a block group."""
    meshes = [b.mesh for b in blocks if b.mesh is not None and not b.mesh.is_empty]
    if not meshes:
        return []
    combined = trimesh.util.concatenate(meshes)
    poly = pv.wrap(combined)
    faces = plotter.add_mesh(poly, color="black", lighting=False, backface_culling=True)
    faces.mapper.SetResolveCoincidentTopologyToPolygonOffset()
    actors = [faces]
    solid = [b.mesh for b in blocks if role(b) != "glazing" and b.mesh is not None and not b.mesh.is_empty]
    if solid:
        edges = pv.wrap(trimesh.util.concatenate(solid)).extract_feature_edges(
            feature_angle=26, boundary_edges=True, non_manifold_edges=True, manifold_edges=False)
        if edges.n_points:
            actors.append(plotter.add_mesh(edges, color="#00ff41", line_width=1, lighting=False))
    return actors


def plotter(size=(W, H)):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("black")
    p.enable_parallel_projection()
    return p


def framing(points, direction, size, margin=1.08, px=.5, py=.5, box=(1., 1.), angles=(0,)):
    """Parallel camera that fits points into a screen box at (px, py), stable over orbit angles."""
    pts = np.asarray(points)[::max(1, len(points) // 40000)]
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2
    aspect = size[0] / size[1]
    # Centre on the projected silhouette (not the bounding box) at the
    # reference angle, then size for the widest silhouette over the orbit.
    ref = 0 if 0 in angles else angles[len(angles) // 2]
    f, right, up = basis(rotate(direction, ref))
    hx, vy = (pts - center) @ right, (pts - center) @ up
    center = center + right * (hx.min() + hx.max()) / 2 + up * (vy.min() + vy.max()) / 2
    scale = 0
    for a in angles:
        f, right, up = basis(rotate(direction, a))
        hx, vy = (pts - center) @ right, (pts - center) @ up
        scale = max(scale, max(abs(vy.min()), vy.max()) / box[1], max(abs(hx.min()), hx.max()) / (aspect * box[0]))
    return {"center": center, "scale": scale * margin, "px": px, "py": py, "aspect": aspect,
            "dist": float(np.ptp(pts, axis=0).max() * 3), "direction": direction}


def rotate(direction, degrees):
    a = math.radians(degrees)
    x, y, z = direction
    return np.array([x * math.cos(a) + z * math.sin(a), y, -x * math.sin(a) + z * math.cos(a)], float)


def basis(direction):
    f = np.asarray(direction, float)
    f /= np.linalg.norm(f)
    right = np.cross([0, 1, 0], f)
    right /= np.linalg.norm(right)
    return f, right, np.cross(f, right)


def place(p, view, angle=0.0, zoom=1.0, pan=(0, 0)):
    f, right, up = basis(rotate(view["direction"], angle))
    scale = view["scale"] / zoom
    c = view["center"]
    focal = (c - right * (view["px"] - .5 + pan[0]) * 2 * scale * view["aspect"]
               + up * (view["py"] - .5 + pan[1]) * 2 * scale)
    p.camera_position = [tuple(focal + f * view["dist"]), tuple(focal), tuple(up)]
    p.camera.parallel_scale = float(scale)
    p.reset_camera_clipping_range()


def shot(p):
    p.render()
    return Image.fromarray(p.screenshot(return_img=True)).convert("RGB")


def to_screen(p, point):
    ren = p.renderer
    ren.SetWorldPoint(*point, 1.0)
    ren.WorldToDisplay()
    x, y, _ = ren.GetDisplayPoint()
    return x, p.window_size[1] - y


# ---------------------------------------------------------------- scenes
class Cathedral:
    """S1 assemble + S2 anatomy + S7 end card share one high-gothic stage."""

    def __init__(self):
        self.result = build_scene("gothic-cathedral", variant="high-gothic", detail="presentation")
        blocks = [b for b in self.result.scene.blocks if b.mesh is not None and not b.mesh.is_empty]
        blocks.sort(key=lambda b: (round(float(b.mesh.bounds[0][1]), 2), float(b.mesh.bounds[0][2])))
        self.p = plotter()
        k = 64
        self.chunks = []
        for i in range(k):
            part = blocks[i * len(blocks) // k:(i + 1) * len(blocks) // k]
            self.chunks.append((len(part), group_actors(self.p, part)))
        self.total = len(self.result.scene.blocks)
        self.points = np.vstack([b.mesh.vertices for b in blocks])
        self.roles = Counter(role(b) for b in self.result.scene.blocks)
        self.view1 = framing(self.points, (-1.4, .85, -1.5), (W, H), 1.04, px=.7, py=.54,
                             box=(.52, .76), angles=(-14, 0, 4))
        self.view2 = framing(self.points, (-1.25, .62, -1.6), (W, H), 1.0, px=.64, py=.57,
                             box=(.58, .72), angles=(-6, 0))
        self.view7 = framing(self.points, (1.35, .8, 1.55), (W, H), 1.04, px=.69, py=.5,
                             box=(.5, .7), angles=(-8, 0, 6))
        self.anchors = self._anchors(blocks)

    def _anchors(self, blocks):
        zs = self.points[:, 2]
        front, depth = zs.min(), np.ptp(zs)

        def centroid(sel):
            v = np.vstack([b.mesh.vertices for b in sel])
            return v.mean(axis=0)

        def pick(name, test):
            sel = [b for b in blocks if role(b) == name and test(b.mesh.bounds)]
            return sel or [b for b in blocks if role(b) == name]

        spire = pick("spire", lambda bb: bb[1][0] < -3)
        tip = max(spire, key=lambda b: b.mesh.bounds[1][1]).mesh.bounds[1][1]
        spire = [b for b in spire if b.mesh.bounds[0][1] > tip - 9]
        rose = pick("rose_tracery", lambda bb: bb[0][2] < front + 3)
        portal = pick("west_portal", lambda bb: abs(bb[0][0] + bb[1][0]) < 2 and bb[0][2] < front + 3)
        mid = front + depth * .32
        flyer = pick("flying_buttress", lambda bb: bb[1][0] < -4 and abs((bb[0][2] + bb[1][2]) / 2 - mid) < 1.5)
        pin = pick("pinnacle", lambda bb: bb[1][0] < -10 and abs((bb[0][2] + bb[1][2]) / 2 - (mid + 6)) < 2)
        return [("spire", centroid(spire)), ("rose_tracery", centroid(rose)),
                ("flying_buttress", centroid(flyer)), ("west_portal", centroid(portal)),
                ("pinnacle", centroid(pin))]

    def reveal(self, fraction):
        shown = int(round(fraction * len(self.chunks)))
        count = 0
        for i, (n, actors) in enumerate(self.chunks):
            for a in actors:
                a.SetVisibility(i < shown)
            count += n if i < shown else 0
        return count

    def s1(self, f):
        start, end = 48, 146
        frac = ease((f - start) / (end - start)) if f >= start else 0.0
        count = self.reveal(frac) if f < end else self.reveal(1.0)
        if f == end:
            cue(f, "boom")
        place(self.p, self.view1, angle=-14 + 18 * ease(f / 329))
        img = shot(self.p)
        d = ImageDraw.Draw(img)
        cmd = "blockn make gothic-cathedral --detail presentation --seed 73"
        s = typed(cmd, f, 6, 1.6)
        if f >= 6 and len(s) < len(cmd) and int((f - 6) * 1.6) != int((f - 7) * 1.6):
            cue(f, "tick")
        if f == 46:
            cue(f, "enter")
        blink = "_" if (f // 8) % 2 == 0 or len(s) < len(cmd) else " "
        text(d, (SAFE, SAFE), "$ " + s + (blink if f < 46 else ""), 30)
        if f >= 46:
            label = "building" + "." * (1 + (f // 6) % 3) if f < end else "quality=pass"
            text(d, (SAFE, SAFE + 48), f"{label:<14}members {count:>6,}", 30, DIM if f < end else GREEN)
        a = ramp(f, 150, 162)
        if a > 0:
            if f == 150:
                cue(f, "title")
            y = int(650 + (1 - a) * 30)
            text(d, (SAFE - 6, y), "BLOCKS", 172, tuple(int(c * a) for c in GREEN), True)
            text(d, (SAFE, y + 196), "procedural architecture,", 36, tuple(int(c * a) for c in PALE))
            text(d, (SAFE, y + 244), "stone by stone.", 36, tuple(int(c * a) for c in PALE))
        return img

    def s2(self, f):
        self.reveal(1.0)
        # Blend from the assembly framing into the anatomy framing.
        t = ramp(f, 180, 214)
        v = dict(self.view2)
        for key in ("scale", "px", "py"):
            v[key] = self.view1[key] + (self.view2[key] - self.view1[key]) * t
        v["center"] = self.view1["center"] + (self.view2["center"] - self.view1["center"]) * t
        v["direction"] = tuple(np.asarray(self.view1["direction"]) * (1 - t) + np.asarray(self.view2["direction"]) * t)
        place(self.p, v, angle=(-14 + 18 * ease(f / 329)) * (1 - t) + (-4 + 6 * ramp(f, 214, 329)) * t)
        img = shot(self.p)
        d = ImageDraw.Draw(img)
        headline(d, f, 186, "EVERY STONE IS A NAMED MEMBER.",
                 f"{self.total:,} members  ·  role, id, course  ·  each stone ≤ 0.8 m")
        for i, (name, point) in enumerate(self.anchors):
            start = 200 + i * 18
            if f == start:
                cue(f, "pop")
            a = ramp(f, start, start + 6)
            if a <= 0:
                continue
            ax, ay = to_screen(self.p, point)
            lx, ly = SAFE, 330 + i * 112
            col = tuple(int(c * a) for c in GREEN)
            text(d, (lx, ly), name, 32, col, True)
            text(d, (lx, ly + 42), f"×{self.roles[name]:,} members", 22, tuple(int(c * a) for c in DIM))
            ex = lx + font(32, True).getlength(name) + 18
            line = ramp(f, start + 3, start + 14)
            mx, my = ex + (ax - ex) * line, ly + 18 + (ay - ly - 18) * line
            d.line((ex, ly + 18, mx, my), fill=col, width=2)
            if line >= 1:
                r = 7 + 3 * math.sin(f / 3)
                d.ellipse((ax - r, ay - r, ax + r, ay + r), outline=PALE, width=3)
        return img

    def s7(self, f):
        self.reveal(1.0)
        if f == 1440:
            cue(f, "boom")
        place(self.p, self.view7, angle=-8 + 14 * ease((f - 1440) / 209), zoom=1.0 + .1 * ease((f - 1440) / 209))
        img = shot(self.p)
        d = ImageDraw.Draw(img)
        a = ramp(f, 1448, 1462)
        text(d, (SAFE - 6, 300 + int((1 - a) * 24)), "BLOCKS", 172, tuple(int(c * a) for c in GREEN), True)
        b = ramp(f, 1466, 1480)
        text(d, (SAFE, 500), "procedural architecture,", 40, tuple(int(c * b) for c in PALE))
        text(d, (SAFE, 552), "stone by stone.", 40, tuple(int(c * b) for c in PALE))
        c = ramp(f, 1494, 1506)
        if f == 1494:
            cue(f, "pop")
        text(d, (SAFE, 664), "github.com/newjordan/BlockN", 38, tuple(int(x * c) for x in GREEN), True)
        e = ramp(f, 1512, 1524)
        text(d, (SAFE, 730), "$ python -m blockn make gothic-cathedral", 28, tuple(int(x * e) for x in DIM))
        footer(d, "open source · MIT  ·  30 products  ·  desktop workbench  ·  CLI + agents", f, 1530)
        return img


class Montage:
    ITEMS = [("classical-building", "doric temple", {"variant": "doric", "detail": "standard"}),
             ("classical-rotunda", "rotunda", {"detail": "standard"}),
             ("classical-basilica", "basilica", {"detail": "standard"}),
             ("classical-aqueduct", "aqueduct", {"detail": "standard"}),
             ("classical-amphitheatre", "amphitheatre", {"detail": "concept"}),
             ("classical-triumphal-arch", "triumphal arch", {"detail": "standard"}),
             ("classical-theatre", "theatre", {"detail": "concept"}),
             ("marble-run", "marble run", {"detail": "concept"})]

    def __init__(self):
        self.stages = []
        for product, label, kw in self.ITEMS:
            result = build_scene(product, **kw)
            p = plotter()
            blocks = [b for b in result.scene.blocks if b.mesh is not None and not b.mesh.is_empty]
            group_actors(p, blocks)
            pts = np.vstack([b.mesh.vertices for b in blocks])
            view = framing(pts, (-1.5, .75, -1.7), (W, H), 1.05, px=.58, py=.56, box=(.7, .68), angles=(-10, 0, 10))
            self.stages.append((product, label, p, view))

    def s3(self, f):
        i = min(len(self.stages) - 1, (f - 330) // 30)
        k = (f - 330) % 30
        if k == 0:
            cue(f, "cut")
        product, label, p, view = self.stages[i]
        place(p, view, angle=-10 + 20 * k / 29, zoom=1.0 + .04 * k / 29)
        img = shot(p)
        d = ImageDraw.Draw(img)
        headline(d, f, 330, "30 PRODUCTS. WALLS TO CATHEDRALS.",
                 "classical orders  ·  masonry  ·  circulation  ·  gothic  ·  play")
        text(d, (SAFE, H - 190), label.upper(), 54, GREEN, True)
        text(d, (SAFE, H - 124), product, 28, DIM)
        text(d, (W - SAFE, H - 124), f"{i + 1:02d} / {len(self.stages):02d}", 28, DIM, anchor="ra")
        return img


class Workbench:
    """S4: the real Qt workbench, driven by an on-screen cursor."""

    def __init__(self, app):
        sys.path.insert(0, str(ROOT / "scripts"))
        from PyQt5.QtCore import QPoint
        from gui.main_window import MainWindow
        self.QPoint = QPoint
        self.app = app
        self.win = MainWindow()
        self.win.resize(1840, 846)
        self.win.show()
        app.processEvents()
        self.panel = self.win.generation_panel
        self.state = 0
        self.seed = ""
        self.last = None
        self.generated_at = None

    def center(self, widget):
        p = widget.mapTo(self.win, self.QPoint(widget.width() // 2, widget.height() // 2))
        return np.array([p.x(), p.y()], float)

    def capture(self):
        from render_architect_showcase import capture_window
        return capture_window(self.win)

    def step(self, f):
        panel, app = self.panel, self.app
        if self.state == 0 and f >= 600:
            panel.classical_form_combo.setCurrentIndex(panel.classical_form_combo.findData("gothic-cathedral"))
            panel.form_detail.setCurrentText("presentation")
            cue(f, "click")
            self.state = 1
        if self.state == 1 and f >= 645:
            panel.seed_input.setFocus()
            panel.seed_input.setText("")
            cue(f, "click")
            self.state = 2
        if self.state == 2 and f >= 650:
            digits = "73"[:min(2, (f - 650) // 4 + 1)]
            if digits != panel.seed_input.text():
                panel.seed_input.setText(digits)
                cue(f, "tick")
            if digits == "73":
                self.state = 3
        if self.state == 3 and f >= 690:
            cue(f, "click")
            panel.generate_button.click()
            app.processEvents()
            self.state = 4
        if self.state == 4 and f >= 702:
            while self.win._generation_thread is not None:
                app.processEvents()
                time.sleep(.01)
            app.processEvents()
            from gui.camera import frame_cathedral
            points = np.vstack([b.mesh.vertices for b in self.win.scene.blocks])
            frame_cathedral(self.win.viewport.plotter, points, (-1.8, .9, -2.2), margin=1.08)
            cue(f, "boom")
            self.generated_at = f
            self.state = 5
        if self.state == 5 and f >= 830:
            panel.cathedral_cutaway.setChecked(True)
            app.processEvents()
            from gui.camera import frame_cathedral
            points = np.vstack([b.mesh.vertices for b in self.win.viewport._batch_blocks.values()])
            frame_cathedral(self.win.viewport.plotter, points, (-1.3, 2.2, -.7), margin=1.09)
            cue(f, "click")
            self.state = 6
        if self.state >= 5:
            self.win.viewport.plotter.camera.Azimuth(.18)
        app.processEvents()

    def target(self, f):
        p = self.panel
        rest = np.array([1500., 560.])
        path = [(570, np.array([1800., 700.])), (598, self.center(p.classical_form_combo)),
                (612, self.center(p.classical_form_combo)), (642, self.center(p.seed_input)),
                (662, self.center(p.seed_input)), (688, self.center(p.generate_button)),
                (712, self.center(p.generate_button)), (760, rest), (800, rest),
                (828, self.center(p.cathedral_cutaway) - [60, 0]), (850, self.center(p.cathedral_cutaway) - [60, 0]),
                (900, rest)]
        for (f0, a), (f1, b) in zip(path, path[1:]):
            if f0 <= f <= f1:
                return a + (b - a) * ease((f - f0) / max(1, f1 - f0))
        return path[-1][1]

    def record(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        meta = {}
        for f in range(570, 930):
            self.step(f)
            self.capture().save(folder / f"app{f:04d}.png")
            meta[f] = {"target": self.target(f).tolist(), "state": self.state,
                       "generated_at": self.generated_at}
        (folder / "meta.json").write_text(json.dumps({"frames": meta, "cues": CUES}))


class WorkbenchFootage:
    """S4 composition from frames the capture process recorded from the real app."""

    def __init__(self, folder):
        self.folder = folder
        data = json.loads((folder / "meta.json").read_text())
        self.meta = {int(k): v for k, v in data["frames"].items()}
        for frame, kind in data["cues"]:
            cue(frame, kind)

    def s4(self, f):
        m = self.meta[f]
        app_img = Image.open(self.folder / f"app{f:04d}.png").convert("RGB")
        aw, ah = app_img.size
        # Push in on the controls while they are used, then pull back out.
        k = ramp(f, 574, 594) * (1 - ramp(f, 694, 716))
        zx0, zy0, zw = 0.0, 40.0, 1025.0
        cw = aw + (zw - aw) * k
        ch = cw * ah / aw
        cx0, cy0 = zx0 * k, zy0 * k
        app_img = app_img.crop((int(cx0), int(cy0), int(cx0 + cw), int(cy0 + ch)))
        scale = 1728 / cw
        app_img = app_img.resize((1728, int(round(ah * 1728 / aw))), Image.LANCZOS)
        img = Image.new("RGB", (W, H), "black")
        ox, oy = SAFE, 226
        img.paste(app_img, (ox, oy))
        d = ImageDraw.Draw(img)
        d.rectangle((ox - 2, oy - 2, ox + app_img.width + 1, oy + app_img.height + 1), outline=RULE, width=2)
        headline(d, f, 570, "1 / USE THE DESKTOP WORKBENCH.",
                 "pick a form  ›  set style + seed  ›  generate  ›  open the roof to inspect")
        tx, ty = (np.asarray(m["target"]) - [cx0, cy0]) * scale + [ox, oy]
        press = 0.0
        for click in (600, 645, 690, 830):
            if click <= f < click + 10:
                press = (f - click) / 10
        if f < 912:
            cursor(d, tx, ty, press)
        if m["state"] == 4 or (m["generated_at"] and f < m["generated_at"] + 18):
            text(d, (ox + app_img.width - 16, oy + app_img.height + 12), "generation time compressed", 22, DIM, anchor="ra")
        return img


class Terminal:
    COMMAND = ["blockn make gothic-cathedral --variant rayonnant \\",
               "    --detail presentation --seed 73 --bundle builds/cathedral-73"]

    def __init__(self, receipt_lines):
        self.lines = receipt_lines
        self.result = build_scene("gothic-cathedral", variant="rayonnant", detail="presentation")
        blocks = [b for b in self.result.scene.blocks if b.mesh is not None and not b.mesh.is_empty]
        blocks.sort(key=lambda b: float(b.mesh.bounds[0][1]))
        self.size = (760, 860)
        self.p = plotter(self.size)
        self.chunks = [group_actors(self.p, blocks[i * len(blocks) // 24:(i + 1) * len(blocks) // 24]) for i in range(24)]
        pts = np.vstack([b.mesh.vertices for b in blocks])
        self.view = framing(pts, (-1.3, .9, -1.5), self.size, 1.04, angles=(-20, 0, 20))

    def s5(self, f):
        img = Image.new("RGB", (W, H), "black")
        d = ImageDraw.Draw(img)
        headline(d, f, 930, "2 / OR TYPE ONE COMMAND.", "the same architect from the CLI — or from an agent")
        x0, y0, x1, y1 = SAFE, 236, 1040, 920
        d.rectangle((x0, y0, x1, y1), outline=RULE, width=2)
        d.rectangle((x0, y0, x1, y0 + 40), fill=(0, 22, 8))
        text(d, (x0 + 18, y0 + 8), "~/blocks", 20, DIM)
        y = y0 + 64
        cps, start = 1.7, 940
        consumed = 0
        for i, line in enumerate(self.COMMAND):
            prefix = "$ " if i == 0 else "  "
            s = typed(line, f, start + consumed / cps, cps)
            consumed += len(line)
            text(d, (x0 + 22, y), prefix + s, 22, GREEN)
            if 0 < len(s) < len(line) and int((f - start) * cps) % 2 == 0:
                cue(f, "tick")
            y += 34
        if f == 1012:
            cue(f, "enter")
        y += 14
        for i, line in enumerate(self.lines):
            appear = 1020 + i * 7
            if f < appear:
                break
            if f == appear and i == 0:
                cue(f, "pop")
            col = GREEN if "quality=pass" in line or line.startswith("glb") else DIM
            for chunk in textwrap.wrap(line, 70, subsequent_indent="  ", break_on_hyphens=False):
                text(d, (x0 + 22, y), chunk, 20, col)
                y += 30
        if f >= 1100:
            a = ramp(f, 1100, 1112)
            text(d, (x0 + 22, y1 - 70), "JSON in  →  GLB + architecture + construction + quality out",
                 22, tuple(int(c * a) for c in PALE))
        # The exported model resolves beside its receipt.
        frac = ramp(f, 1060, 1100)
        for i, actors in enumerate(self.chunks):
            for a in actors:
                a.SetVisibility(i < int(round(frac * len(self.chunks))))
        if f == 1060:
            cue(f, "boom")
        place(self.p, self.view, angle=-20 + 40 * ramp(f, 1060, 1229))
        model = shot(self.p)
        img.paste(model, (W - SAFE - self.size[0], 150))
        if frac > 0:
            text(d, (W - SAFE - self.size[0] // 2, 236), "cathedral-73.glb  ·  rayonnant", 22, DIM, anchor="ma")
        footer(d, "deterministic seeds  ·  replayable documents  ·  GLB / OBJ / STL export", f, 1120)
        return img


class Styles:
    STYLES = [("high-gothic", "HIGH GOTHIC", "paired spires · flèche"),
              ("early-english", "EARLY ENGLISH", "crossing tower · capped west towers"),
              ("rayonnant", "RAYONNANT", "three-light tracery · 12-petal rose")]

    def __init__(self):
        self.p = plotter()
        self.groups = []
        allpts = []
        for i, (variant, label, note) in enumerate(self.STYLES):
            result = build_scene("gothic-cathedral", variant=variant, detail="presentation")
            blocks = [b for b in result.scene.blocks if b.mesh is not None and not b.mesh.is_empty]
            for b in blocks:
                b.mesh = b.mesh.copy()
                b.mesh.apply_translation([(i - 1) * 74.0, 0, 0])
            actors = group_actors(self.p, blocks)
            pts = np.vstack([b.mesh.vertices for b in blocks])
            center = (pts.min(axis=0) + pts.max(axis=0)) / 2
            for a in actors:
                a.SetOrigin(*center)
            self.groups.append((actors, center))
            allpts.append(pts)
        self.view = framing(np.vstack(allpts), (0, .55, -1), (W, H), 1.08, px=.5, py=.5, box=(.95, .56))

    def s6(self, f):
        angle = -28 + 56 * ramp(f, 1230, 1439)
        for actors, center in self.groups:
            for a in actors:
                a.SetOrientation(0, angle, 0)
        place(self.p, self.view)
        img = shot(self.p)
        d = ImageDraw.Draw(img)
        headline(d, f, 1236, "ONE RULE SET. THREE STYLES.", "same request schema  ·  different composition rules")
        for i, (variant, label, note) in enumerate(self.STYLES):
            start = 1250 + i * 12
            if f == start:
                cue(f, "pop")
            a = ramp(f, start, start + 8)
            cx = int(W * (i + .5) / 3)
            text(d, (cx, 900), label, 34, tuple(int(c * a) for c in GREEN), True, "ma")
            text(d, (cx, 946), note, 22, tuple(int(c * a) for c in DIM), anchor="ma")
        return img


# ---------------------------------------------------------------- audio
def soundtrack(path, seconds, cues, sr=48000):
    """Original synthesized score plus effects on the logged event frames."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(73)
    out = np.zeros((n, 2), np.float32)
    bpm = 112
    beat = 60 / bpm
    roots = [45, 41, 48, 43]  # A, F, C, G (MIDI)

    def hz(m):
        return 440 * 2 ** ((m - 69) / 12)

    # Pad: slow detuned chord per bar, opening from the title onward.
    bar = 4 * beat
    for b in range(int(seconds / bar) + 1):
        s = int(b * bar * sr)
        e = min(n, int((b + 1) * bar * sr) + int(.4 * sr))
        if s >= n:
            break
        tt = np.arange(e - s) / sr
        env = np.minimum(1, tt / .6) * np.minimum(1, (e - s - np.arange(e - s)) / sr / .5)
        root = roots[b % 4]
        chord = [root + 12, root + 19, root + 24, root + 28 if b % 4 != 1 else root + 27]
        sig = sum(np.sin(2 * np.pi * hz(m) * tt * (1 + det)) for m in chord for det in (-.002, .002))
        sig += .5 * np.sin(2 * np.pi * hz(root) * tt)
        out[s:e, 0] += .028 * env * sig
        out[s:e, 1] += .028 * env * np.roll(sig, 40)
    # Pulse: eighth-note arpeggio and kick, building through the middle scenes.
    for k in range(int(seconds / (beat / 2))):
        at = k * beat / 2
        s = int(at * sr)
        if s >= n:
            break
        energy = .35 if at < 6 else (1.0 if at < 47 else .4)
        bar_i = int(at / bar) % 4
        m = roots[bar_i] + 24 + [0, 7, 12, 7, 15, 12, 7, 12][k % 8]
        L = min(n - s, int(.22 * sr))
        tt = np.arange(L) / sr
        pluck = np.sign(np.sin(2 * np.pi * hz(m) * tt)) * .3 + np.sin(2 * np.pi * hz(m) * tt)
        out[s:s + L] += (.016 * energy * pluck * np.exp(-tt * 18))[:, None]
        if k % 2 == 0 and 5 < at < 47.5:
            L = min(n - s, int(.35 * sr))
            tt = np.arange(L) / sr
            kick = np.sin(2 * np.pi * (48 * tt + 90 * (1 - np.exp(-tt * 30)) / 30)) * np.exp(-tt * 9)
            out[s:s + L] += (.11 * energy * kick)[:, None]
        if k % 2 == 1 and 9 < at < 47.5:
            L = min(n - s, int(.05 * sr))
            out[s:s + L] += (.012 * rng.normal(0, 1, L) * np.exp(-np.arange(L) / sr * 90))[:, None]

    def add(at, sig, gain):
        s = int(at * sr)
        L = min(len(sig), n - s)
        if L > 0:
            out[s:s + L] += (gain * sig[:L])[:, None]

    for frame, kind in cues:
        at = frame / FPS
        if kind == "tick":
            L = int(.012 * sr)
            add(at, np.diff(rng.normal(0, 1, L + 1)) * np.exp(-np.arange(L) / sr * 400), .05)
        elif kind == "enter":
            tt = np.arange(int(.12 * sr)) / sr
            add(at, np.sin(2 * np.pi * 220 * tt) * np.exp(-tt * 40) + np.diff(rng.normal(0, 1, len(tt) + 1)) * np.exp(-tt * 300), .12)
        elif kind == "click":
            tt = np.arange(int(.06 * sr)) / sr
            add(at, (np.sin(2 * np.pi * 1800 * tt) + .6 * np.sin(2 * np.pi * 900 * tt)) * np.exp(-tt * 90), .08)
        elif kind == "pop":
            tt = np.arange(int(.09 * sr)) / sr
            add(at, np.sin(2 * np.pi * (880 + 600 * tt / .09) * tt) * np.exp(-tt * 45), .07)
        elif kind == "cut":
            tt = np.arange(int(.16 * sr)) / sr
            noise = np.diff(rng.normal(0, 1, len(tt) + 1))
            add(at - .03, noise * np.exp(-tt * 28), .06)
        elif kind in ("boom", "title"):
            tt = np.arange(int(1.4 * sr)) / sr
            add(at, np.sin(2 * np.pi * (38 * tt + 60 * (1 - np.exp(-tt * 12)) / 12)) * np.exp(-tt * 3.2), .32)
    fade = int(1.2 * sr)
    out[-fade:] *= np.linspace(1, 0, fade)[:, None]
    out *= .89 / max(1e-6, np.abs(out).max())
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((out * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------- main
def wipe(prev, cur, k, n=7):
    """Scan-bar transition: the new scene draws in from the top."""
    y = int(H * ease(k / n))
    if y >= H or prev is None:
        return cur
    out = prev.copy()
    out.paste(cur.crop((0, 0, W, y)), (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle((0, max(0, y - 3), W, y + 2), fill=PALE)
    return out


def bundle_receipt(work):
    """Run the real CLI once and keep its stdout, with paths shortened."""
    prefix = work / "cathedral-73"
    cmd = [sys.executable, "-m", "blockn", "make", "gothic-cathedral", "--variant", "rayonnant",
           "--detail", "presentation", "--seed", "73", "--bundle", str(prefix)]
    out = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True).stdout
    lines = [line.replace(str(work) + "/", "builds/") for line in out.strip().splitlines()]
    (work / "cli-receipt.txt").write_text(out)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "builds/blocks-teaser")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--step", type=int, default=1, help=">1 renders review stills only")
    ap.add_argument("--capture-workbench", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.capture_workbench:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        Workbench(app).record(args.capture_workbench)
        return
    out = args.output.resolve()
    review = out / "review"
    review.mkdir(parents=True, exist_ok=True)
    wanted = args.scenes.split(",")
    print("Preparing stages...", flush=True)
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    stages = {}
    if {"s1", "s2", "s7"} & set(wanted):
        stages["cathedral"] = Cathedral()
    if "s3" in wanted:
        stages["montage"] = Montage()
    if "s5" in wanted:
        stages["terminal"] = Terminal(bundle_receipt(out))
    if "s6" in wanted:
        stages["styles"] = Styles()
    if "s4" in wanted:
        footage = out / "workbench-frames"
        if not (footage / "meta.json").exists():
            print("Capturing the real workbench...", flush=True)
            env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
            if DISPLAY:
                env["DISPLAY"] = DISPLAY
            subprocess.run([sys.executable, __file__, "--capture-workbench", str(footage)],
                           cwd=ROOT, env=env, check=True)
        stages["workbench"] = WorkbenchFootage(footage)
    render = {"s1": lambda f: stages["cathedral"].s1(f), "s2": lambda f: stages["cathedral"].s2(f),
              "s3": lambda f: stages["montage"].s3(f), "s4": lambda f: stages["workbench"].s4(f),
              "s5": lambda f: stages["terminal"].s5(f), "s6": lambda f: stages["styles"].s6(f),
              "s7": lambda f: stages["cathedral"].s7(f)}
    full = args.step == 1 and wanted == list(SCENES)
    writer = None
    silent = out / "blocks-teaser-silent.mp4"
    if full:
        writer = subprocess.Popen(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                                   "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                                   "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-profile:v", "high",
                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent)],
                                  stdin=subprocess.PIPE)
    keep = ({SCENES[s][0] + k for s in SCENES for k in (0, 4, 8)} | {SCENES[s][1] for s in SCENES}
            | set(range(0, TOTAL, 45)) | {1560})
    prev = last = None
    started = time.time()
    for name in wanted:
        a, b = SCENES[name]
        print(f"Rendering {name} {a}-{b}...", flush=True)
        for f in range(a, b + 1):
            if args.step > 1 and (f - a) % args.step and f != b:
                continue
            if f == a:
                prev = last
                if f > 0:
                    cue(f, "cut")
            frame = grade(render[name](f))
            if f - a < 8 and f > 0:
                frame = wipe(prev, frame, f - a)
            if writer:
                writer.stdin.write(frame.tobytes())
            if not full or f in keep:
                frame.save(review / f"f{f:04d}.png")
            last = frame
        print(f"  done at {time.time() - started:.0f}s", flush=True)
    cues = sorted(set(CUES))
    (out / "cues.json").write_text(json.dumps([{"frame": f, "seconds": round(f / FPS, 3), "kind": k} for f, k in cues], indent=1))
    if not full:
        return
    writer.stdin.close()
    writer.wait()
    wav = out / "blocks-teaser-score.wav"
    soundtrack(wav, TOTAL / FPS, cues)
    final = out / "blocks-teaser-x.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(silent), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
                    "-movflags", "+faststart", str(final)], check=True)
    Image.open(review / "f1560.png").save(out / "cover.png") if (review / "f1560.png").exists() else None
    manifest = {"video": final.name, "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                "frames": TOTAL, "fps": FPS, "resolution": [W, H], "scenes": SCENES,
                "source": "actual Blocks geometry, actual Qt workbench (offscreen), actual CLI receipt",
                "architect_sha256": hashlib.sha256((ROOT / "core/architecture/cathedral.py").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"video": str(final), "seconds": TOTAL / FPS}), flush=True)


if __name__ == "__main__":
    main()
