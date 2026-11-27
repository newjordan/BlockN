#!/usr/bin/env python3
"""Render the vertical demo of the Chinese and Dravidian architects from real builds.

Every animated value is a function of the global frame index. The models are built by the
product's own pipeline and drawn by its own viewport, offscreen. Member counts and the
quality status in the captions are read from each build, never typed in. Each build is
paced by the stages its architect laid its members in, not by their number.

Requires the GUI extra (PyQt5, pyvista, vtk), Pillow, and FFmpeg with libx264 and aac. No
display is needed. Frames are rendered offline; the film's frame rate is not a measure of
interactive performance.

    python3 scripts/render_two_architects_demo.py                        # the film, with its score
    python3 scripts/render_two_architects_demo.py --posters              # 1280x720 stills for a README
    python3 scripts/render_two_architects_demo.py --stills 60 540        # single frames, for review
    python3 scripts/render_two_architects_demo.py --full --resume        # score a finished silent render
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import wave

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "builds" / "two-architects-demo"       # --output moves it

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920                       # the canvas
VW, VH, VY = 1080, 1000, 400            # the viewport window, and where it sits on the canvas
FPS, LAST = 30, 1109
FADE_OUT = 1085                         # the film goes black over its last 25 frames
BRIGHT, SOFT, DIM, RULE = (0, 255, 65), (0, 190, 90), (0, 150, 70), (0, 70, 30)
# Every monospace face below has a 0.6 em advance, which keeps each caption on its measured line.
FONT_DIRECTORIES = ("/usr/share/fonts/TTF", "/usr/share/fonts/truetype/jetbrains-mono", "/usr/share/fonts/truetype/dejavu",
                    "/usr/share/fonts/truetype/noto", "/usr/share/fonts/noto", "/usr/share/fonts/liberation",
                    "/usr/local/share/fonts", "~/.local/share/fonts", "~/.fonts", "~/Library/Fonts", "/Library/Fonts",
                    "C:/Windows/Fonts")
FONTS = {"medium": ("JetBrainsMono-Medium.ttf", "DejaVuSansMono.ttf", "NotoSansMono-Regular.ttf", "LiberationMono-Regular.ttf"),
         "bold": ("JetBrainsMono-ExtraBold.ttf", "DejaVuSansMono-Bold.ttf", "NotoSansMono-Bold.ttf", "LiberationMono-Bold.ttf")}
TEXT = {"eyebrow": ("medium", 36), "head": ("bold", 64), "stat": ("medium", 40), "line": ("medium", 34),
        "big": ("bold", 72), "mid": ("medium", 44), "code": ("medium", 40), "repo": ("medium", 34),
        "poster": ("bold", 34), "note": ("medium", 19)}
# Product, variant, title, view direction, and how far to stand back and slide the model clear of the captions.
POSTERS = [("chinese-hall", "hip-and-gable", "Chinese hall", (-2.0, 1.15, -2.6), 1.3, 140),
           ("chinese-palace-hall", "double-hip-roof", "Chinese palace hall", (-2.0, 1.35, -2.6), 1.3, 140),
           ("dravidian-gopuram", "rajagopuram", "Dravidian gateway tower", (-2.0, .95, -2.6), 1.2, 170),
           ("dravidian-vimana", "shala-vimana", "Dravidian vimana", (-2.5, .9, -1.1), 1.22, 150)]
PW, PH = 1280, 720                         # a poster's window
REPO_URL = "github.com/newjordan/Blocks"

SCENES = [
    dict(key="hall", first=0, last=299, model=("chinese-hall", "hip-and-gable"), grow=(8, 200),
         view=(-2.0, 1.15, -2.6), orbit=10.0, eyebrow="BLOCKS / NEW: CHINESE ARCHITECT", title="TIMBER HALL",
         detail="hip-and-gable roof, bracket sets, terrace", caption_in=12, stats_in=214, caption_out=292),
    dict(key="palace", first=300, last=461, model=("chinese-palace-hall", "double-hip-roof"), grow=None,
         view=(-2.0, 1.35, -2.6), orbit=12.0, eyebrow="BLOCKS / CHINESE ARCHITECT", title="PALACE HALL",
         detail="nine bays, nine-step brackets, two eaves", caption_in=300, stats_in=300, caption_out=454),
    dict(key="gate", first=462, last=799, model=("dravidian-gopuram", "rajagopuram"), grow=(472, 692),
         view=(-2.0, .95, -2.6), orbit=10.0, eyebrow="BLOCKS / NEW: DRAVIDIAN ARCHITECT", title="GATEWAY TOWER",
         detail="7 talas, kutas and shalas, open passage", caption_in=462, stats_in=706, caption_out=792),
    dict(key="vimana", first=800, last=929, model=("dravidian-vimana", "shala-vimana"), grow=None,
         view=(-2.5, .9, -1.1), orbit=10.0, eyebrow="BLOCKS / DRAVIDIAN ARCHITECT", title="VIMANA",
         detail="granite throughout, with a pillared hall", caption_in=800, stats_in=800, caption_out=922),
    dict(key="outro", first=930, last=LAST, model=("chinese-hall", "hip-and-gable"), model_key="hall",
         backdrop=True,                                             # the hall again, dimmed
         view=(-2.0, 1.0, -2.6), orbit=6.0),
]
OUTRO_LINES = [(934, "big", BRIGHT, "{architects} ARCHITECTS", 560), (948, "mid", BRIGHT, "{products} PRODUCTS / {variants} VARIANTS", 680),
               (962, "mid", BRIGHT, "{rules} MEASURED RULES", 752), (972, "line", SOFT, "every build passes its quality gate", 824)]
BELL = OUTRO_LINES[0][0]                                               # the bell rings as the first line lands
SR = 48000
CHORDS = {"hall": (110.0, 164.81, 261.63, 329.63), "palace": (87.31, 130.81, 220.0, 261.63),
          "gate": (146.83, 220.0, 349.23, 440.0), "vimana": (98.0, 146.83, 246.94, 293.66),
          "outro": (110.0, 164.81, 220.0, 277.18)}


def clamp01(value):
    return 0.0 if value < 0 else 1.0 if value > 1 else float(value)


def find_font(weight):
    """The first installed file of a weight: JetBrains Mono where it exists, then the usual fallbacks."""
    for name in FONTS[weight]:
        for folder in FONT_DIRECTORIES:
            path = Path(folder).expanduser() / name
            if path.is_file():
                return path
    raise SystemExit(f"no monospace font found for {weight}; looked for {', '.join(FONTS[weight])}")


def ease(value):
    u = clamp01(value)
    return u * u * (3 - 2 * u)


def fade_in(frame, start, frames=12):
    return clamp01((frame - start) / frames)


def scene_at(frame):
    return next(scene for scene in SCENES if scene["first"] <= frame <= scene["last"])


LEVELS = ("tala", "storey", "tier", "level")


def stages(blocks, recall=400):
    """The build's stages, in the order the architect laid them. A stage continues while the same component
    roles keep coming back (a column's base alternates with its shaft). It ends when a role has been unseen
    for `recall` members, or when the architect moves on to another storey."""
    keys = [key for key in LEVELS if sum(key in block.parameters for block in blocks) >= .1 * len(blocks)]
    marks, seen, level = [], {}, None
    for index, block in enumerate(blocks):
        role = block.parameters.get("component_role")
        tier = block.parameters.get(keys[0]) if keys else None
        if (tier is not None and level is not None and tier != level) or index - seen.get(role, -recall - 1) > recall:
            marks.append(index)
        if tier is not None:
            level = tier
        seen[role] = index
    return list(zip(marks, marks[1:] + [len(blocks)]))


def pace(sizes, frames):
    """Schedule frames for each stage: three fifths of the growth shared equally among the stages and two
    fifths by the square root of their size. By count alone a terrace's thousand paving stones take a second
    and every column arrives in one frame, and a tower's plain base outlasts all its talas."""
    weights = [size ** .5 for size in sizes]
    share = [(3 * frames / len(sizes) + 2 * frames * weight / sum(weights)) / 5 for weight in weights]
    lengths = [max(1, round(value)) for value in share]
    lengths[int(np.argmax(share))] += frames - sum(lengths)     # rounding: on the longest stage
    return lengths


def schedule(blocks, first, last):
    """Members on screen at each frame of a growth that starts at `first` and is complete at `last`. The
    stages' schedule is read through a curve that is quick off the mark and slows to a finish, so that the
    first stone does not jump and the ridge, or the crown, is given its moment: 6u^2 - 8u^3 + 3u^4."""
    bounds = stages(blocks)
    lengths = pace([end - start for start, end in bounds], last - first)
    laid = [0]
    for (start, end), length in zip(bounds, lengths):
        laid += [int(round(start + (end - start) * step / length)) for step in range(1, length + 1)]
    total = last - first
    counts = [0] * first
    for frame in range(total + 1):
        u = frame / total
        at = total * u * u * (6 - 8 * u + 3 * u * u)
        low = int(at)
        counts.append(laid[-1] if low == total else laid[low] + (laid[low + 1] - laid[low]) * (at - low))
    return [int(round(count)) for count in counts]


class Sprites:
    """Each caption is rasterised once with the real font, then faded and moved per frame."""

    def __init__(self):
        self.fonts = {key: ImageFont.truetype(str(find_font(weight)), size) for key, (weight, size) in TEXT.items()}
        self.cache = {}

    def get(self, text, kind, colour):
        key = (text, kind, colour)
        if key not in self.cache:
            font = self.fonts[kind]
            box = font.getbbox(text)
            image = Image.new("RGBA", (box[2] - box[0] + 4, box[3] - box[1] + 4), (0, 0, 0, 0))
            ImageDraw.Draw(image).text((2 - box[0], 2 - box[1]), text, font=font, fill=colour + (255,))
            self.cache[key] = (image, box[2] - box[0])
        return self.cache[key]

    def put(self, layer, text, kind, colour, x, y, alpha, centre=False):
        if alpha <= 0:
            return
        image, width = self.get(text, kind, colour)
        image = image.copy()
        image.putalpha(image.getchannel("A").point(lambda value: int(value * alpha)))
        layer.alpha_composite(image, (int(x - width / 2) if centre else int(x), int(y)))


class Film:
    def __init__(self):
        from PyQt5.QtWidgets import QApplication
        from gui.viewport import ViewportWidget
        self.app = QApplication.instance() or QApplication([])
        self.viewport = ViewportWidget()
        self.viewport.resize(VW, VH)
        self.viewport.show()
        self.app.processEvents()
        self.models, self.blocks, self.corners, self.counts, self.shown = {}, {}, {}, {}, None
        self.sprites = Sprites()
        self.canvas = Image.new("RGB", (W, H), "black")
        ramp = np.ones((VH, 1, 1), dtype=np.float32)
        for row in range(70):                           # the viewport fades to black at its top and bottom
            ramp[row] = row / 70
            ramp[VH - 1 - row] = row / 70
        self.ramp = ramp
        self.facts = outro_facts()
        print("outro:", self.facts, flush=True)

    def model(self, scene):
        key = scene.get("model_key", scene["key"])
        if key not in self.models:
            from core.architecture import BuildRequest, build
            product, variant = scene["model"]
            started = time.time()
            self.models[key] = build(BuildRequest(product=product, variant=variant, detail="concept", seed=73))
            self.blocks[key] = list(self.models[key].scene.blocks)
            if scene.get("grow"):
                self.counts[key] = schedule(self.blocks[key], *scene["grow"])
            metrics = self.models[key].quality.metrics
            lo, hi = metrics["bounds_min"], metrics["bounds_max"]
            self.corners[key] = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            print(f"built {product} / {variant}: {len(self.blocks[key])} members, {self.models[key].quality.status}, "
                  f"{time.time() - started:.1f}s", flush=True)
        return self.models[key]

    def shoot(self, frame, scene):
        """The product's viewport at this frame: the model whole, or as far as the build has got."""
        from gui.camera import frame_cathedral
        self.model(scene)
        key = scene.get("model_key", scene["key"])
        blocks = self.blocks[key]
        count = len(blocks)
        counts = self.counts.get(key)
        if counts:
            count = counts[min(frame, len(counts) - 1)]      # as far along its stages as this frame is
        state = (key, count)
        if self.shown != state:
            self.viewport.display_meshes(blocks[:count])
            self.shown = state
        plotter = self.viewport.plotter
        frame_cathedral(plotter, self.corners[key], scene["view"], margin=1.04)
        camera = plotter.camera
        target = np.array(camera.focal_point)
        vector = np.array(camera.position) - target
        angle = math.radians(scene["orbit"] * (frame - scene["first"]) / (scene["last"] - scene["first"]))
        turn = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        camera.position = target + turn @ vector
        plotter.render()
        return plotter.screenshot(return_img=True)

    def compose(self, frame, scene, shot):
        image = shot.astype(np.float32) * self.ramp
        if scene.get("backdrop"):
            image *= .22
        canvas = self.canvas.copy()
        canvas.paste(Image.fromarray(image.clip(0, 255).astype(np.uint8)), (0, VY))
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sprites = self.sprites
        if scene.get("backdrop"):
            for start, kind, colour, text, y in OUTRO_LINES:
                alpha = fade_in(frame, start, 14)
                sprites.put(layer, text.format(**self.facts), kind, colour, W / 2, y + 20 * (1 - ease(alpha)), alpha, centre=True)
            alpha = fade_in(frame, 990, 14)
            draw = ImageDraw.Draw(layer)
            box = (154, 956, 926, 1044)
            draw.rounded_rectangle(box, radius=14, outline=DIM + (int(255 * alpha),), width=2)
            sprites.put(layer, "blockn make chinese-hall", "code", BRIGHT, W / 2, 970, alpha, centre=True)
            sprites.put(layer, REPO_URL, "repo", SOFT, W / 2, 1130, fade_in(frame, 1020, 14), centre=True)
        else:
            leaving = 1 - fade_in(frame, scene["caption_out"], 8)
            alpha = fade_in(frame, scene["caption_in"], 12) * leaving
            rise = 24 * (1 - ease(fade_in(frame, scene["caption_in"], 12)))
            sprites.put(layer, scene["eyebrow"], "eyebrow", DIM, 72, 250 + rise, alpha)
            sprites.put(layer, scene["title"], "head", BRIGHT, 72, 292 + rise, alpha)
            ImageDraw.Draw(layer).line((72, 380, 1008, 380), fill=RULE + (int(255 * alpha),), width=1)
            stats = fade_in(frame, scene["stats_in"], 14) * leaving
            key = scene.get("model_key", scene["key"])
            result = self.models[key]
            report = f"{len(self.blocks[key]):,} MEMBERS / QUALITY {result.quality.status.upper()}"
            sprites.put(layer, report, "stat", BRIGHT, 72, 1424 + 12 * (1 - ease(stats)), stats)
            sprites.put(layer, scene["detail"], "line", SOFT, 72, 1472 + 12 * (1 - ease(stats)), stats)
        out = canvas.convert("RGBA")
        out.alpha_composite(layer)
        out = out.convert("RGB")
        if frame >= FADE_OUT:
            out = Image.blend(out, Image.new("RGB", out.size, "black"), ease((frame - FADE_OUT + 1) / (LAST - FADE_OUT + 1)))
        return out


def cue_list():
    cues = []
    for scene in SCENES:
        if scene.get("grow"):
            g0, g1 = scene["grow"]
            cues += [{"cue": f"swell-{scene['key']}", "first_frame": g0, "last_frame": g1},
                     {"cue": f"done-{scene['key']}", "frame": g1}]
    cues += [{"cue": f"pad-{scene['key']}", "frame": scene["first"]} for scene in SCENES]    # a chord on each cut
    cues += [{"cue": "bell", "frame": BELL}]
    for cue in cues:
        cue["seconds"] = round(cue.get("frame", cue.get("first_frame")) / FPS, 4)
    return cues


def score():
    """An original synthesized score. Nothing here is a recording."""
    count = int((LAST + 1) / FPS * SR)
    t = np.arange(count) / SR
    audio, transients = np.zeros(count), np.zeros(count)    # the thumps and the bell, kept apart for the sync check
    order = [scene["key"] for scene in SCENES]                    # a chord for each scene
    starts = {scene["key"]: scene["first"] for scene in SCENES}
    for index, key in enumerate(order):
        start = starts[key] / FPS
        end = starts[order[index + 1]] / FPS if index + 1 < len(order) else t[-1]
        envelope = np.clip((t - start) / .9, 0, 1) * np.clip((end + .9 - t) / 1.2, 0, 1)
        for note, pitch in enumerate(CHORDS[key]):
            beat = 1 + .0012 * (note + 1)
            audio += .055 * envelope * (np.sin(2 * np.pi * pitch * t) + .6 * np.sin(2 * np.pi * pitch * beat * t)
                                       + .15 * np.sin(4 * np.pi * pitch * t))
    rng = np.random.default_rng(73)
    noise = np.convolve(rng.normal(0, 1, count), np.ones(48) / 48, mode="same")
    for scene in SCENES:
        if scene.get("grow"):
            g0, g1 = (value / FPS for value in scene["grow"])
            span = g1 - g0
            swell = np.where((t >= g0) & (t <= g1), np.sin(np.pi * np.clip((t - g0) / span, 0, 1)), 0.0)
            audio += .07 * noise * swell
            tau = t - g1
            thump = np.where(tau >= 0, np.exp(-tau * 8) * np.sin(2 * np.pi * (52 * tau + 90 * (1 - np.exp(-tau * 25)) / 25)), 0.0)
            transients += .3 * thump
    tau = t - BELL / FPS
    bell = np.where(tau >= 0, np.exp(-tau * 2.2) * np.sin(2 * np.pi * 880 * tau)
                    + .4 * np.exp(-tau * 5) * np.sin(2 * np.pi * 880 * 2.76 * tau), 0.0)
    transients += .16 * bell
    audio += transients
    audio *= np.clip(t / .3, 0, 1) * np.clip(((FADE_OUT + 1) / FPS - t) / .6, 0, 1)
    gain = .5 / np.max(np.abs(audio))
    audio, transients = audio * gain, transients * gain
    stereo = np.column_stack((audio, audio * .97)).astype(np.float32)
    with wave.open(os.path.join(OUT, "score.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes((stereo * 32767).astype("<i2").tobytes())
    return audio, transients


def outro_facts():
    """What the outro claims, counted from the architects: the registry for products and variants,
    and each module's own `rule_checks` for the rule codes it can report."""
    from core.architecture import chinese, dravidian, historic
    forms = [form for module, form in historic.styles().values() if module in (chinese, dravidian)]
    rules = set()
    for module in (chinese, dravidian):
        tree = ast.parse(inspect.getsource(module))
        check = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rule_checks")
        for node in ast.walk(check):
            first = node.args[0].elts[0] if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Tuple) \
                and node.args[0].elts else None
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                rules.add(first.value)
    return {"architects": 2, "products": len(forms), "variants": sum(len(form.variants) for form in forms), "rules": len(rules)}


def mux(video, audio_path, target):
    """Two-pass loudness normalisation: measure the score, then apply one linear gain."""
    goal = "loudnorm=I=-16:TP=-1.5:LRA=11"
    measured = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", audio_path, "-af", goal + ":print_format=json",
                               "-f", "null", "-"], capture_output=True, text=True, check=True).stderr
    values = json.loads(measured[measured.rindex("{"):measured.rindex("}") + 1])
    normalise = goal + ":linear=true:" + ":".join(
        f"{key}={values[source]}" for key, source in (("measured_I", "input_i"), ("measured_TP", "input_tp"),
                                                       ("measured_LRA", "input_lra"), ("measured_thresh", "input_thresh"),
                                                       ("offset", "target_offset"))) + ",aresample=48000"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", audio_path, "-filter_complex",
                    f"[1:a]{normalise}[a]", "-map", "0:v:0", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", target], check=True)


def loudness(path):
    """Integrated loudness and true peak of a finished file, as ffmpeg's meter reads it."""
    text = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                          capture_output=True, text=True, check=True).stderr
    summary = text[text.rindex("Summary:"):]
    return {"integrated_lufs": float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1)),
            "true_peak_dbtp": float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1))}


def decode(path):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"])
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def sync_report(decoded, stem, cues):
    """For each transient, how far the delivered audio sits behind the score. Positive is late."""
    report = []
    for cue in cues:
        if not cue["cue"].startswith(("done-", "bell")):
            continue
        at = int(cue["seconds"] * SR)
        reference = stem[at: at + 4800]
        reference = reference / np.linalg.norm(reference)
        lag = max(range(-480, 481), key=lambda shift: float(np.dot(decoded[at + shift: at + shift + 4800], reference)))
        report.append({"cue": cue["cue"], "frame": cue["frame"], "lag_ms": round(lag / SR * 1000, 2)})
    return report


def scan(path, cuts):
    """Every frame of the delivered file, at low resolution: which frames are black, and where the picture
    changes abruptly. Abrupt changes belong at the cuts and nowhere else."""
    width, height = 216, 384
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={width}:{height}", "-f", "rawvideo",
                                   "-pix_fmt", "gray", "-"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width).astype(np.float32)
    dark = np.flatnonzero(np.diff(np.r_[0, frames.mean(axis=(1, 2)) < 1.0, 0].astype(np.int8)))
    change = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))          # change[i]: from frame i into frame i + 1
    ranked = np.argsort(change)[::-1][:6]
    return {"frames": int(len(frames)),
            "near_black": [[int(a), int(b) - 1] for a, b in zip(dark[::2], dark[1::2])],
            "largest_changes": [{"into_frame": int(i + 1), "change": round(float(change[i]), 2), "is_cut": i + 1 in cuts}
                                for i in sorted(ranked)]}


def sheet(paths, target, columns=6, thumb=(270, 480)):
    rows = -(-len(paths) // columns)
    board = Image.new("RGB", (columns * (thumb[0] + 6) + 6, rows * (thumb[1] + 6) + 6), (40, 0, 0))
    for index, path in enumerate(paths):
        board.paste(Image.open(path).convert("RGB").resize(thumb),
                    (6 + index % columns * (thumb[0] + 6), 6 + index // columns * (thumb[1] + 6)))
    board.save(target)
    return target


def posters(film, folder):
    """One 1280x720 still of each product, in the workbench's colours, for a README. Each build is
    shown whole, with its member count and quality result read from the build itself."""
    from gui.camera import frame_cathedral
    film.viewport.resize(PW, PH)
    film.app.processEvents()
    written = []
    for product, variant, title, view, margin, slide in POSTERS:
        scene = dict(key=product, model=(product, variant), view=view)
        film.model(scene)
        film.viewport.display_meshes(film.blocks[product])
        frame_cathedral(film.viewport.plotter, film.corners[product], view, margin=margin)
        pan(film.viewport.plotter.camera, slide, PH)          # the captions have the left column to themselves
        film.viewport.plotter.render()
        shot = np.asarray(film.viewport.plotter.screenshot(return_img=True))
        assert shot.shape == (PH, PW, 3), shot.shape                  # the window followed the widget
        image = Image.fromarray(shot.astype(np.uint8)).convert("RGB")
        note = f"{len(film.blocks[product]):,} members / quality {film.models[product].quality.status}"
        plaque(image, 48, 44, "Two new architects",
               f"{film.facts['architects']} architects / {film.facts['products']} products / "
               f"{film.facts['variants']} variants / every member named", film.sprites.fonts)
        plaque(image, 48, PH - 136, "> " + title, note, film.sprites.fonts)
        path = os.path.join(folder, product + ".png")
        image.save(path)
        written.append(path)
    return written


def pan(camera, pixels, height):
    """Slide a parallel-projection camera sideways so the model moves `pixels` to the right
    in a window `height` pixels tall."""
    matrix = camera.GetModelViewTransformObject().GetMatrix()
    right = np.array([matrix.GetElement(0, axis) for axis in range(3)])
    step = right * (pixels * 2 * camera.GetParallelScale() / height)
    camera.SetFocalPoint(np.array(camera.GetFocalPoint()) - step)
    camera.SetPosition(np.array(camera.GetPosition()) - step)


def plaque(image, x, y, title, note, fonts):
    """A caption box: a thin outline, a bold title and a line beneath it."""
    draw = ImageDraw.Draw(image)
    width = math.ceil(max(draw.textlength(title, font=fonts["poster"]), draw.textlength(note, font=fonts["note"])))
    draw.rectangle((x, y, x + width + 32, y + 92), fill=(0, 0, 0), outline=(0, 110, 40), width=2)
    draw.text((x + 16, y + 12), title, font=fonts["poster"], fill=BRIGHT)
    draw.text((x + 16, y + 58), note, font=fonts["note"], fill=(0, 190, 90))


def render(film, target):
    """Every frame in order, piped to ffmpeg. The cover is frame 240, kept before compression."""
    started = time.time()
    log = open(os.path.join(OUT, "review", "ffmpeg.log"), "wb")
    writer = subprocess.Popen([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-profile:v", "high", target], stdin=subprocess.PIPE, stderr=log)
    cover = None
    for frame in range(LAST + 1):
        scene = scene_at(frame)
        image = film.compose(frame, scene, film.shoot(frame, scene))
        if frame == 240:
            cover = image.copy()
        writer.stdin.write(image.tobytes())
        if frame % 150 == 0:
            print(f"frame {frame}/{LAST} at {time.time() - started:.0f}s", flush=True)
    writer.stdin.close()
    if writer.wait():
        raise RuntimeError("ffmpeg failed; see review/ffmpeg.log")
    log.close()
    print(f"video: {LAST + 1} frames in {time.time() - started:.0f}s", flush=True)
    return cover


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stills", nargs="*", type=int, help="render only these global frames, for review")
    parser.add_argument("--full", action="store_true", help="render the film, score it, mux and probe it")
    parser.add_argument("--posters", action="store_true", help="one 1280x720 still of each product, for a README")
    parser.add_argument("--resume", action="store_true",
                        help="with --full: the silent render is already complete; score, mux and probe that")
    parser.add_argument("--output", default=str(OUT), help="where to write the film and its review")
    args = parser.parse_args()
    args.full = args.full or args.resume
    if not any((args.posters, args.stills is not None, args.full)):
        args.full = True                            # no mode given: make the film
    if sum((args.posters, args.stills is not None, args.full)) > 1:
        parser.error("choose one of --posters, --stills or --full")
    OUT = Path(args.output)
    review = os.path.join(OUT, "review")
    os.makedirs(review, exist_ok=True)
    film = Film()
    if args.posters:
        print("posters:", *posters(film, str(OUT)), sep="\n  ")
        return
    if args.stills:
        paths = []
        for frame in args.stills:
            scene = scene_at(frame)
            image = film.compose(frame, scene, film.shoot(frame, scene))
            path = os.path.join(review, f"still-{frame:04d}.png")
            image.save(path)
            paths.append(path)
        paths = sorted(os.path.join(review, name) for name in os.listdir(review) if name.startswith("still-"))
        print("stills:", *paths, sep="\n  ")
        print("sheet:", sheet(paths, os.path.join(review, "stills-sheet.png")))
        return

    name = "blocks-two-architects"
    silent = os.path.join(OUT, name + "-silent.mp4")
    final = os.path.join(OUT, name + ".mp4")
    if args.resume:
        if not os.path.exists(silent):
            raise RuntimeError(f"nothing to resume from: {os.path.basename(silent)} does not exist; render without --resume")
        counted = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
            "stream=nb_read_frames", "-of", "json", silent]))["streams"][0]["nb_read_frames"]
        if int(counted) != LAST + 1:
            raise RuntimeError(f"{os.path.basename(silent)} has {counted} frames, not {LAST + 1}; render again without --resume")
        cover = film.compose(240, scene_at(240), film.shoot(240, scene_at(240)))     # the same frame as in the film
    else:
        cover = render(film, silent)
    cover.save(os.path.join(OUT, "cover.png"))

    cues = cue_list()
    with open(os.path.join(OUT, "cues.json"), "w") as handle:
        json.dump({"fps": FPS, "frames": LAST + 1, "note": "sample position = frame / fps; every transient starts on a zero crossing at its frame",
                   "cues": cues}, handle, indent=2)
    _, transients = score()
    mux(silent, os.path.join(OUT, "score.wav"), final)
    os.remove(silent)

    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration", "-of", "json", final]))
    sound = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
        "stream=codec_name,sample_rate,channels,duration", "-of", "json", final]))
    samples = [0, 20, 45, 60, 90, 120, 160, 200, 240, 299, 300, 380, 461, 462, 480, 530, 600, 650, 692, 750, 799, 800, 860,
               929, 930, 980, 1010, 1070, 1100, LAST]
    for stale in os.listdir(review):
        if stale.startswith("encoded-"):
            os.remove(os.path.join(review, stale))
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", final, "-vf",
                    "select='" + "+".join(f"eq(n\\,{frame})" for frame in samples) + "'", "-fps_mode", "vfr",
                    os.path.join(review, "encoded-%02d.png")], check=True)
    encoded = sorted(path for path in os.listdir(review) if path.startswith("encoded-"))
    for number, start in enumerate(range(0, len(encoded), 12), 1):
        sheet([os.path.join(review, path) for path in encoded[start:start + 12]],
              os.path.join(review, f"encoded-sheet-{number}.png"), columns=4)
    decoded = decode(final)
    motion = scan(final, [scene["first"] for scene in SCENES[1:]])
    last = np.asarray(Image.open(os.path.join(review, encoded[-1])).convert("RGB"))
    stream = probe["streams"][0]
    verification = {
        "file": os.path.basename(final), "bytes": os.path.getsize(final),
        "video": stream, "audio": sound["streams"][0],
        "matches_brief": {"width": stream["width"] == W, "height": stream["height"] == H,
                          "frames": int(stream["nb_read_frames"]) == LAST + 1,
                          "fps": stream["avg_frame_rate"] == "30/1", "duration_s": abs(float(stream["duration"]) - (LAST + 1) / FPS) < .05},
        "last_frame_max_channel": int(last.max()), "last_frame_is_black": bool(last.max() < 4),
        "audio_length_matches_video": abs(float(sound["streams"][0]["duration"]) - float(stream["duration"])) < .1,
        "loudness": loudness(final),
        "transient_lag_in_delivered_file": sync_report(decoded, transients, cues),
        "frames_scanned": motion["frames"],
        "near_black_frames": motion["near_black"],
        "largest_frame_changes": motion["largest_changes"],
        "rms_dbfs_per_second": [round(float(20 * np.log10(np.sqrt(np.mean(decoded[i:i + SR] ** 2)) + 1e-9)), 1)
                                for i in range(0, len(decoded) - SR + 1, SR)],
        "review_sheets": sorted(name for name in os.listdir(review) if name.startswith("encoded-sheet")),
    }
    with open(os.path.join(OUT, "verification.json"), "w") as handle:
        json.dump(verification, handle, indent=2)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
