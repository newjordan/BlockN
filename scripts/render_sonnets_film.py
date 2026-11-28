#!/usr/bin/env python3
"""Render the vertical film of the seventeen-architect update from real builds.

The film leads with one idea: a sonnet form has fixed rules and open lines, and the seed writes a
different building each time. One church is raised stage by stage. The same form is then cut through
twelve verses on one plot, with the lines that changed lit as they change. Seven more sonnet forms
follow, two verses each, then the eight other architects new in the update, and a tomb is raised to
close.

It is made the same way as the two-architects film and shares its pipeline: every animated value
is a function of the global frame index, the models come from the product's own build and are drawn
by its own viewport, and every count and status on screen is read from a build, never typed in.

Requires the GUI extra (PyQt5, pyvista, pyvistaqt, vtk), Pillow, and FFmpeg with libx264 and aac.

    python3 scripts/render_sonnets_film.py                       # the film, with its score
    python3 scripts/render_sonnets_film.py --stills 120 400      # single frames, for review
    python3 scripts/render_sonnets_film.py --full --resume       # score a finished silent render
"""
from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from PIL import Image, ImageDraw

from render_two_architects_demo import (BRIGHT, DIM, FPS, H, RULE, SOFT, SR, VH, VW, VY, W, Sprites, decode,
                                        ease, fade_in, loudness, mux, scan, schedule, sheet, sync_report)

REPO_URL = "github.com/newjordan/BlockN"            # the public repository

OUT = ROOT / "builds" / "sonnets-film"
SEED = 73
VIEW = (-2.0, 1.25, -2.6)
HERO = ("muscovite-church", "five-domed")
FINALE = ("mughal-tomb", "taj")
VERSES, VERSE_FRAMES = 12, 26
REEL_FRAMES = 84                                    # two verses of each form, 42 frames apiece
MORE_FRAMES = 66                                    # the other new architects: two builds each, 33 frames apiece
# product, architect as the eyebrow names it, title, the rule in one line, view direction
REEL = [("mesopotamian-ziggurat", "MESOPOTAMIAN", "ZIGGURAT", "stages recede, every face leans inward", (-2.0, 1.15, -2.6)),
        ("buddhist-stupa", "BUDDHIST STUPA", "STUPA", "a clear path circles the solid mound", (-2.0, 1.3, -2.6)),
        ("zimbabwe-enclosure", "GREAT ZIMBABWE", "DRY-STONE ENCLOSURE", "granite laid dry, no straight run or corner",
         (-2.0, 1.7, -2.6)),
        ("lalibela-church", "LALIBELA", "ROCK-HEWN CHURCH", "cut down from the rock, roof at ground level",
         (-2.0, 2.1, -2.6)),
        ("artdeco-tower", "ART DECO", "SETBACK TOWER", "massed under the 1916 sky-exposure plane", (-2.0, .8, -2.6)),
        ("geodesic-dome", "GEODESIC", "GEODESIC DOME", "every hub on one sphere, all faces triangles", (-2.0, .95, -2.6)),
        ("tibetan-palace", "TIBETAN", "FORTRESS-PALACE", "walls lean inward, windows grow upward", (-2.0, 1.1, -2.6))]
# The other architects new in this update. They are not sonnet forms: they vary by variant and proportion.
MORE = [("byzantine-church", "BLOCKS / NEW: BYZANTINE ARCHITECT", "CROSS-IN-SQUARE CHURCH", "four pendentives carry the drum and dome"),
        ("sahelian-mosque", "BLOCKS / NEW: SUDANO-SAHELIAN ARCHITECT", "EARTHEN MOSQUE", "three battered towers face the qibla"),
        ("inka-kancha", "BLOCKS / NEW: IMPERIAL INKA ARCHITECT", "KANCHA", "fitted stone, jambs narrow toward lintels"),
        ("khmer-temple-mountain", "BLOCKS / NEW: ANGKOR KHMER ARCHITECT", "TEMPLE-MOUNTAIN", "five towers over three receding terraces"),
        ("swahili-courtyard-house", "BLOCKS / NEW: SWAHILI COAST ARCHITECT", "COURTYARD HOUSE", "open courts behind a screened entrance"),
        ("minangkabau-rumah-gadang", "BLOCKS / NEW: MINANGKABAU ARCHITECT", "RUMAH GADANG", "raised timber frame, swept gonjong roof"),
        ("chacoan-great-house", "BLOCKS / NEW: CHACOAN ARCHITECT", "GREAT HOUSE", "terraced rooms around recessed round kivas"),
        ("brutalist-civic-center", "BLOCKS / NEW: BRUTALISM STUDIO", "CIVIC CENTER", "cast members seated on expressed supports")]

A_LAST = 299
B_FIRST, B_LAST = A_LAST + 1, A_LAST + VERSES * VERSE_FRAMES
C_FIRST, C_LAST = B_LAST + 1, B_LAST + len(REEL) * REEL_FRAMES
M_FIRST, M_LAST = C_LAST + 1, C_LAST + len(MORE) * MORE_FRAMES
D_FIRST, D_LAST = M_LAST + 1, M_LAST + 318
E_FIRST, LAST = D_LAST + 1, D_LAST + 198
FADE_OUT = LAST - 24
GROWS = {"form": (8, 200), "finale": (D_FIRST + 2, D_FIRST + 210)}
BELL = E_FIRST + 4
OUTRO_LINES = [(BELL, "big", BRIGHT, "{architects} NEW ARCHITECTS", 540),
               (BELL + 14, "mid", BRIGHT, "{products} PRODUCTS / {verses} SONNET VERSES", 660),
               (BELL + 28, "line", SOFT, "one form, many buildings, no verse repeated", 744),
               (BELL + 38, "line", SOFT, "every build passes its quality gate", 796)]
COMMAND = "blockn sonnet mughal-tomb --count 12"
CHORDS = [(110.0, 164.81, 261.63, 329.63), (87.31, 130.81, 220.0, 261.63), (146.83, 220.0, 349.23, 440.0),
          (98.0, 146.83, 246.94, 293.66), (130.81, 196.0, 261.63, 392.0), (110.0, 164.81, 220.0, 277.18)]


def scenes():
    """The timeline. Cells are the buildings a scene cuts through, each (first frame, model key)."""
    reel = []
    for index, (product, architect, title, rule, view) in enumerate(REEL):
        first = C_FIRST + index * REEL_FRAMES
        reel.append(dict(key=f"reel-{product}", first=first, last=first + REEL_FRAMES - 1, view=view, orbit=9.0,
                         eyebrow=f"BLOCKS / NEW: {architect} ARCHITECT", title=title, detail=rule,
                         cells=[(first, f"{product}:a"), (first + REEL_FRAMES // 2, f"{product}:b")]))
    more = []
    for index, (product, eyebrow, title, rule) in enumerate(MORE):
        first = M_FIRST + index * MORE_FRAMES
        more.append(dict(key=f"more-{product}", first=first, last=first + MORE_FRAMES - 1, orbit=7.0,
                         eyebrow=eyebrow, title=title, detail=rule, product=product,
                         cells=[(first, f"{product}:a"), (first + MORE_FRAMES // 2, f"{product}:b")]))
    return [
        dict(key="form", first=0, last=A_LAST, view=VIEW, orbit=10.0, grow=GROWS["form"],
             eyebrow="BLOCKS / NEW: ARCHITECTURE SONNETS", title="ONE FORM", detail="muscovite church: the rules are fixed",
             cells=[(0, "hero")], stats_in=GROWS["form"][1] + 12),
        dict(key="verses", first=B_FIRST, last=B_LAST, view=VIEW, orbit=14.0, eyebrow="BLOCKS / MUSCOVITE CHURCH",
             title="MANY VERSES", detail="the seed answers each open line", counted=True,
             cells=[(B_FIRST + index * VERSE_FRAMES, f"verse-{index}") for index in range(VERSES)]),
        *reel,
        *more,
        dict(key="finale", first=D_FIRST, last=D_LAST, view=(-2.0, 1.1, -2.6), orbit=10.0, grow=GROWS["finale"],
             eyebrow="BLOCKS / NEW: MUGHAL ARCHITECT", title="GARDEN TOMB", detail="one face composed, then turned four times",
             cells=[(D_FIRST, "finale")], stats_in=GROWS["finale"][1] + 12),
        dict(key="outro", first=E_FIRST, last=LAST, view=(-2.0, 1.0, -2.6), orbit=6.0, backdrop=True,
             cells=[(E_FIRST, "finale")]),
    ]


SCENES = scenes()


def scene_at(frame):
    return next(scene for scene in SCENES if scene["first"] <= frame <= scene["last"])


def cell_at(frame, scene):
    """(index, first frame, model key) of the building on screen."""
    index = max(i for i, (first, _) in enumerate(scene["cells"]) if first <= frame)
    return (index, *scene["cells"][index])


def requests():
    """Every building in the film, on its form's default plot so that only the verse changes."""
    from core.architecture import BuildRequest, sonnet
    from core.architecture.historic import style_for

    def on_default_plot(request):
        return BuildRequest(product=request.product, variant=request.variant, detail="concept", seed=request.seed,
                            parameters=dict(request.parameters))
    wanted = {"hero": on_default_plot(sonnet.variations(HERO[0], 1, SEED, variant=HERO[1])[0]),
              "finale": on_default_plot(sonnet.variations(FINALE[0], 1, SEED, variant=FINALE[1])[0])}
    for index, request in enumerate(sonnet.variations(HERO[0], VERSES, 0)):
        wanted[f"verse-{index}"] = on_default_plot(request)
    for product, *_ in REEL + MORE:
        variants = style_for(product)[1].variants
        for letter, variant, seed in (("a", variants[0], SEED), ("b", variants[-1], SEED + 1)):
            wanted[f"{product}:{letter}"] = on_default_plot(sonnet.variations(product, 1, seed, variant=variant)[0])
    return wanted


class Film:
    def __init__(self):
        from PyQt5.QtWidgets import QApplication
        from gui.viewport import ViewportWidget
        self.app = QApplication.instance() or QApplication([])
        self.viewport = ViewportWidget()
        self.viewport.resize(VW, VH)
        self.viewport.show()
        self.app.processEvents()
        self.requests = requests()
        self.models, self.blocks, self.corners, self.counts, self.verses, self.shown = {}, {}, {}, {}, {}, None
        self.sprites = Sprites()
        self.canvas = Image.new("RGB", (W, H), "black")
        ramp = np.ones((VH, 1, 1), dtype=np.float32)
        for row in range(70):                           # the viewport fades to black at its top and bottom
            ramp[row] = row / 70
            ramp[VH - 1 - row] = row / 70
        self.ramp = ramp

    def model(self, key):
        if key not in self.models:
            from core.architecture import build, sonnet
            request = self.requests[key]
            started = time.time()
            result = build(request)
            if result.quality.status != "pass":
                raise SystemExit(f"{key} ({request.product}) is {result.quality.status}; the film only shows clean passes")
            self.models[key] = result
            self.blocks[key] = list(result.scene.blocks)
            metrics = result.quality.metrics
            lo, hi = metrics["bounds_min"], metrics["bounds_max"]
            self.corners[key] = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            self.verses[key] = sonnet.verse(request)
            print(f"built {key}: {request.product} / {request.variant}, {len(self.blocks[key])} members, "
                  f"{time.time() - started:.1f}s", flush=True)
        return self.models[key]

    def view(self, scene, key):
        """A scene's own view direction, or the one its architect asks for."""
        from core.architecture.historic import style_for
        return scene.get("view") or getattr(style_for(self.requests[key].product)[0], "VIEW", VIEW)

    def drop(self, text, kind="line"):
        """How far below the cap line a word's ink starts. Sprites are cropped to their ink, so a word
        with no ascender would otherwise ride up in a column of rows."""
        font = self.sprites.fonts[kind]
        return font.getbbox(text)[1] - font.getbbox("H")[1]

    def facts(self):
        """What the outro claims, counted from the registry: the architects whose buildings are in this
        film, every product of those architects, and the verses of the ones that are sonnet forms."""
        from core.architecture import sonnet
        from core.architecture.historic import style_for, styles
        architects = {style_for(request.product)[0] for request in self.requests.values()}
        products = [(name, form) for name, (module, form) in styles().items() if module in architects]
        return {"architects": len(architects), "products": len(products),
                "verses": sum(sonnet.verse_count(name) for name, form in products if form.rules.get("sonnet"))}

    def shoot(self, frame, scene):
        """The product's viewport at this frame: the building whole, or as far as the build has got."""
        from gui.camera import frame_cathedral
        _, _, key = cell_at(frame, scene)
        self.model(key)
        blocks = self.blocks[key]
        count = len(blocks)
        if scene.get("grow"):
            if key not in self.counts:
                self.counts[key] = {}
            if scene["key"] not in self.counts[key]:
                self.counts[key][scene["key"]] = schedule(blocks, *scene["grow"])
            counts = self.counts[key][scene["key"]]
            count = counts[min(frame, len(counts) - 1)]
        state = (key, count)
        if self.shown != state:
            self.viewport.display_meshes(blocks[:count])
            self.shown = state
        plotter = self.viewport.plotter
        frame_cathedral(plotter, self.corners[key], self.view(scene, key), margin=1.04)
        camera = plotter.camera
        target = np.array(camera.focal_point)
        vector = np.array(camera.position) - target
        angle = math.radians(scene["orbit"] * (frame - scene["first"]) / (scene["last"] - scene["first"]))
        turn = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        camera.position = target + turn @ vector
        plotter.render()
        shot = plotter.screenshot(return_img=True)
        if shot.shape != (VH, VW, 3):                   # the offscreen window has drifted from its size
            self.viewport.resize(VW, VH)
            plotter.window_size = [VW, VH]
            self.app.processEvents()
            plotter.render()
            shot = plotter.screenshot(return_img=True)
            if shot.shape != (VH, VW, 3):
                raise RuntimeError(f"viewport is {shot.shape[1]}x{shot.shape[0]}, not {VW}x{VH}")
        return shot

    def compose(self, frame, scene, shot):
        image = shot.astype(np.float32) * self.ramp
        if scene.get("backdrop"):
            image *= .22
        canvas = self.canvas.copy()
        canvas.paste(Image.fromarray(image.clip(0, 255).astype(np.uint8)), (0, VY))
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sprites, draw = self.sprites, ImageDraw.Draw(layer)
        if scene.get("backdrop"):
            facts = self.facts()
            for start, kind, colour, text, y in OUTRO_LINES:
                alpha = fade_in(frame, start, 14)
                sprites.put(layer, text.format(**facts), kind, colour, W / 2, y + 20 * (1 - ease(alpha)), alpha, centre=True)
            alpha = fade_in(frame, BELL + 56, 14)
            draw.rounded_rectangle((84, 956, 996, 1044), radius=14, outline=DIM + (int(255 * alpha),), width=2)
            sprites.put(layer, COMMAND, "code", BRIGHT, W / 2, 970, alpha, centre=True)
            sprites.put(layer, REPO_URL, "repo", SOFT, W / 2, 1130, fade_in(frame, BELL + 86, 14), centre=True)
        else:
            index, cell_first, key = cell_at(frame, scene)
            result = self.models[key]
            leaving = 1 - fade_in(frame, scene["last"] - 7, 8)
            arriving = fade_in(frame, scene["first"] + (12 if scene["first"] == 0 else 0), 12)
            alpha = arriving * leaving
            rise = 24 * (1 - ease(arriving))
            sprites.put(layer, scene["eyebrow"], "eyebrow", DIM, 72, 250 + rise, alpha)
            sprites.put(layer, scene["title"], "head", BRIGHT, 72, 292 + rise, alpha)
            draw.line((72, 380, 1008, 380), fill=RULE + (int(255 * alpha),), width=1)
            stats = fade_in(frame, scene.get("stats_in", scene["first"]), 14) * leaving
            settle = 12 * (1 - ease(fade_in(frame, scene.get("stats_in", scene["first"]), 14)))
            status = result.quality.status.upper()
            if scene.get("counted"):
                from core.architecture import sonnet
                report = f"VERSE {index + 1:02d} / {sonnet.verse_count(HERO[0])} POSSIBLE / QUALITY {status}"
            else:
                report = f"{len(self.blocks[key]):,} MEMBERS / QUALITY {status}"
            sprites.put(layer, report, "stat", BRIGHT, 72, 1424 + settle, stats)
            sprites.put(layer, scene["detail"], "line", SOFT, 72, 1472 + settle, stats)
            # The verse of the building on screen. A line whose answer changed at this cut is lit, then settles.
            before = {}
            if index:
                self.model(scene["cells"][index - 1][1])
                before = self.verses[scene["cells"][index - 1][1]]
            draw.line((72, 1536, 1008, 1536), fill=RULE + (int(255 * stats),), width=1)
            rows = [("VARIANT", self.requests[key].variant, False)]
            rows += [(name, answer, bool(before) and before.get(name) != answer)
                     for name, answer in self.verses[key].items()]
            for row, (name, answer, changed) in enumerate(rows):
                lit = 1 - fade_in(frame, cell_first + 6, 14) if changed else 0.0
                y = 1556 + 44 * row + settle
                sprites.put(layer, name, "line", DIM, 72, y + self.drop(name), stats)
                sprites.put(layer, answer, "line", SOFT, 298, y + self.drop(answer), stats * (1 - lit))
                sprites.put(layer, answer, "line", BRIGHT, 298, y + self.drop(answer), stats * lit)
        out = canvas.convert("RGBA")
        out.alpha_composite(layer)
        out = out.convert("RGB")
        if frame >= FADE_OUT:
            out = Image.blend(out, Image.new("RGB", out.size, "black"), ease((frame - FADE_OUT + 1) / (LAST - FADE_OUT + 1)))
        return out


def cue_list():
    cues = []
    for scene in SCENES:
        cues.append({"cue": f"pad-{scene['key']}", "frame": scene["first"]})
        if scene.get("grow"):
            g0, g1 = scene["grow"]
            cues += [{"cue": f"swell-{scene['key']}", "first_frame": g0, "last_frame": g1},
                     {"cue": f"done-{scene['key']}", "frame": g1}]
        if len(scene["cells"]) > 1:
            cues += [{"cue": f"tick-{scene['key']}-{index}", "frame": first}
                     for index, (first, _) in enumerate(scene["cells"]) if index]
    cues.append({"cue": "bell", "frame": BELL})
    for cue in cues:
        cue["seconds"] = round(cue.get("frame", cue.get("first_frame")) / FPS, 4)
    return cues


def score(cues):
    """An original synthesized score. Nothing here is a recording.

    A chord on every scene, a swell of filtered noise under each build and a low thump as it completes,
    a short pluck on every change of verse, and a bell as the outro's first line lands.
    """
    count = int((LAST + 1) / FPS * SR)
    t = np.arange(count) / SR
    audio, transients = np.zeros(count), np.zeros(count)
    starts = [scene["first"] / FPS for scene in SCENES]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else t[-1]
        envelope = np.clip((t - start) / .9, 0, 1) * np.clip((end + .9 - t) / 1.2, 0, 1)
        for note, pitch in enumerate(CHORDS[index % len(CHORDS)]):
            beat = 1 + .0012 * (note + 1)
            audio += .055 * envelope * (np.sin(2 * np.pi * pitch * t) + .6 * np.sin(2 * np.pi * pitch * beat * t)
                                       + .15 * np.sin(4 * np.pi * pitch * t))
    noise = np.convolve(np.random.default_rng(SEED).normal(0, 1, count), np.ones(48) / 48, mode="same")
    for cue in cues:
        name = cue["cue"]
        if name.startswith("swell-"):
            g0, g1 = cue["first_frame"] / FPS, cue["last_frame"] / FPS
            audio += .07 * noise * np.where((t >= g0) & (t <= g1), np.sin(np.pi * np.clip((t - g0) / (g1 - g0), 0, 1)), 0.0)
            continue
        tau = t - cue["frame"] / FPS
        if name.startswith("done-"):
            transients += .3 * np.where(tau >= 0, np.exp(-tau * 8) * np.sin(
                2 * np.pi * (52 * tau + 90 * (1 - np.exp(-tau * 25)) / 25)), 0.0)
        elif name.startswith("tick-"):
            pitch = 660.0 if "verses" in name else 440.0
            audio += .06 * np.where(tau >= 0, np.exp(-tau * 26) * np.sin(2 * np.pi * pitch * tau), 0.0)
        elif name == "bell":
            transients += .16 * np.where(tau >= 0, np.exp(-tau * 2.2) * np.sin(2 * np.pi * 880 * tau)
                                         + .4 * np.exp(-tau * 5) * np.sin(2 * np.pi * 880 * 2.76 * tau), 0.0)
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
    return transients


def render(film, target, cover_frame):
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
        if frame == cover_frame:
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
    parser.add_argument("--resume", action="store_true", help="the silent render is complete; score, mux and probe that")
    parser.add_argument("--output", default=str(OUT), help="where to write the film and its review")
    args = parser.parse_args()
    OUT = Path(args.output)
    review = os.path.join(OUT, "review")
    os.makedirs(review, exist_ok=True)
    film = Film()
    if args.stills:
        paths = []
        for frame in args.stills:
            scene = scene_at(frame)
            image = film.compose(frame, scene, film.shoot(frame, scene))
            paths.append(os.path.join(review, f"still-{frame:04d}.png"))
            image.save(paths[-1])
        print("sheet:", sheet(paths, os.path.join(review, "stills-sheet.png")))
        return

    name = "blocks-architecture-sonnets"
    silent, final = os.path.join(OUT, name + "-silent.mp4"), os.path.join(OUT, name + ".mp4")
    cover_frame = GROWS["form"][1] + 40
    if args.resume:
        scene = scene_at(cover_frame)
        cover = film.compose(cover_frame, scene, film.shoot(cover_frame, scene))
    else:
        cover = render(film, silent, cover_frame)
    cover.save(os.path.join(OUT, "cover.png"))

    cues = cue_list()
    with open(os.path.join(OUT, "cues.json"), "w") as handle:
        json.dump({"fps": FPS, "frames": LAST + 1, "note": "sample position = frame / fps", "cues": cues}, handle, indent=2)
    transients = score(cues)
    mux(silent, os.path.join(OUT, "score.wav"), final)
    os.remove(silent)

    def probe(stream, entries):
        return json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", stream,
                                                   "-show_entries", "stream=" + entries, "-of", "json", final]))["streams"][0]
    video = probe("v:0", "codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration")
    sound = probe("a:0", "codec_name,sample_rate,channels,duration")
    cuts = sorted({first for scene in SCENES for first, _ in scene["cells"]} - {0})
    samples = sorted({0, 30, 110, 240, A_LAST, D_LAST - 30, LAST - 30, LAST, BELL + 60, D_FIRST + 120, D_LAST}
                     | {first + 12 for scene in SCENES for first, _ in scene["cells"]})
    for stale in os.listdir(review):
        if stale.startswith("encoded-"):
            os.remove(os.path.join(review, stale))
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", final, "-vf",
                    "select='" + "+".join(f"eq(n\\,{frame})" for frame in samples) + "'", "-fps_mode", "vfr",
                    os.path.join(review, "encoded-%02d.png")], check=True)
    encoded = sorted(path for path in os.listdir(review) if path.startswith("encoded-") and "sheet" not in path)
    for number, start in enumerate(range(0, len(encoded), 12), 1):
        sheet([os.path.join(review, path) for path in encoded[start:start + 12]],
              os.path.join(review, f"encoded-sheet-{number}.png"), columns=4)
    decoded = decode(final)
    motion = scan(final, cuts)
    last = np.asarray(Image.open(os.path.join(review, encoded[-1])).convert("RGB"))
    verification = {
        "file": os.path.basename(final), "bytes": os.path.getsize(final), "video": video, "audio": sound,
        "matches_brief": {"width": video["width"] == W, "height": video["height"] == H,
                          "frames": int(video["nb_read_frames"]) == LAST + 1, "fps": video["avg_frame_rate"] == "30/1",
                          "duration_s": abs(float(video["duration"]) - (LAST + 1) / FPS) < .05},
        "outro_facts": film.facts(),
        "buildings_shown": len(film.models), "all_quality_pass": all(r.quality.status == "pass" for r in film.models.values()),
        "last_frame_is_black": bool(last.max() < 4),
        "audio_length_matches_video": abs(float(sound["duration"]) - float(video["duration"])) < .1,
        "loudness": loudness(final),
        "transient_lag_in_delivered_file": sync_report(decoded, transients, cues),
        "frames_scanned": motion["frames"], "near_black_frames": motion["near_black"],
        "largest_frame_changes": motion["largest_changes"],
        "review_sheets": sorted(name for name in os.listdir(review) if name.startswith("encoded-sheet")),
    }
    with open(os.path.join(OUT, "verification.json"), "w") as handle:
        json.dump(verification, handle, indent=2)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
