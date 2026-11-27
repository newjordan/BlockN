"""Render Quiet Courtyard, an original, slow score for the Blocks introduction.

The melody, accompaniment and plucked-string synthesis are defined here. No
recordings, samples, MIDI files or existing compositions are imported. Rendering
and replacing the soundtrack require only NumPy and FFmpeg, without the GUI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave

import numpy as np

SAMPLE_RATE = 48000
TITLE = "Quiet Courtyard"

# Each original phrase spans four bars. The upper voice takes time to breathe;
# the accompaniment is a soft, unhurried broken chord throughout.
# (bass, three inner voices, melody as (MIDI pitch or rest, quarter-note beats))
BARS = [
    (50, [57, 62, 66], [(69, 1.5), (66, .5), (64, .75), (66, .75), (69, .5)]),
    (49, [57, 61, 64], [(66, 2), (64, 1), (66, .75), (None, .25)]),
    (47, [54, 57, 62], [(62, 1), (66, 1), (69, 1.75), (None, .25)]),
    (43, [55, 59, 62], [(71, 1.5), (69, .5), (66, 1.75), (None, .25)]),
    (52, [55, 59, 62], [(67, 1), (66, .5), (64, .5), (71, 1.75), (None, .25)]),
    (45, [52, 57, 62], [(69, 2), (67, .75), (64, 1), (None, .25)]),
    (42, [57, 62, 66], [(66, .75), (69, .75), (74, 1.25), (73, .75), (69, .5)]),
    (45, [52, 57, 61], [(73, 1.5), (71, .5), (69, 1.75), (None, .25)]),
    (47, [54, 57, 62], [(66, 1), (69, 1), (71, 1.75), (None, .25)]),
    (45, [54, 57, 61], [(69, 1.5), (66, .5), (73, 1.75), (None, .25)]),
    (43, [55, 59, 62], [(71, 1), (69, .5), (67, .5), (66, 1.75), (None, .25)]),
    (42, [57, 62, 66], [(64, 1), (66, 1), (69, 1.75), (None, .25)]),
    (52, [55, 59, 62], [(67, 1.5), (71, .5), (66, 1.5), (64, .5)]),
    (45, [52, 55, 61], [(62, 1), (64, .75), (67, .75), (64, 1.25), (None, .25)]),
    (43, [55, 59, 62], [(74, 1.5), (71, .5), (69, 1), (66, .75), (None, .25)]),
    (38, [54, 57, 62], [(64, 1), (62, 3)]),
]


def score_events(seconds):
    """Keep all sixteen bars and the final release when fitting the video."""
    if not math.isfinite(seconds) or seconds < 30:
        raise ValueError("This complete score needs at least 30 seconds")
    weights = np.array([1.02, 1, .98, 1.05, .98, 1.02, .98, 1.08,
                        .98, .99, 1, 1.04, 1, 1.04, 1.08, 1.28])
    tail = min(4.5, seconds * .065)
    lengths = weights / weights.sum() * (seconds - tail - .25)
    rng = np.random.default_rng(7309)
    events = []
    beginning = .25
    for bar, ((bass, chord, melody), length) in enumerate(zip(BARS, lengths)):
        beat = length / 4
        phrase = 1 + .07 * math.sin(bar / 15 * math.pi)
        if bar >= 14:
            phrase *= .9 if bar == 14 else .78

        def add(part, midi, at, held, gain, pan):
            jitter = rng.uniform(-.007, .007) if at else 0
            events.append({"part": part, "bar": bar + 1, "midi": midi,
                           "start": float(beginning + at * beat + jitter),
                           "hold": float(held * beat),
                           "gain": float(gain * phrase * rng.uniform(.96, 1.04)),
                           "pan": pan})

        add("bass", bass, 0, 3.7 if bar < 15 else 5, .20, -.08)
        if bar < 15:
            add("bass", bass + 7, 2, 1.85, .095, -.08)
        pattern = [(0, .5), (1, 1), (2, 1.5), (0, 2.5), (2, 3), (1, 3.5)]
        for index, at in pattern:
            if bar == 15 and at > 1.5:
                continue
            pitch = 61 if bar == 5 and at >= 3 and index == 2 else chord[index]
            add("inner", pitch, at, 1.3 if bar < 15 else 4.2,
                .095 if at % 1 else .083, -.24 + index * .045)
        at = 0
        for midi, held in melody:
            if midi is not None:
                add("melody", midi, at + .024, held * .94,
                    .255 if at == 0 else .23, .14)
            at += held
        if abs(at - 4) > 1e-9:
            raise ValueError(f"Bar {bar + 1} must contain four beats")
        beginning += length
    return sorted(events, key=lambda event: event["start"])


def pluck(midi, hold, part):
    """A rounded digital string: damped partials with a soft finger attack."""
    lead = part == "melody"
    release = 1.1 if lead else .75
    t = np.arange(round((hold + release) * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    frequency = 440 * 2 ** ((midi - 69) / 12)
    # A tiny delayed vibrato lets long melody notes breathe without pitch scoops.
    vibrato = (.00075 if lead else .00015) * (1 - np.exp(-t * 1.1))
    phase = 2 * np.pi * frequency * t - frequency * vibrato / 4.7 * np.cos(2 * np.pi * 4.7 * t)
    tone = np.zeros(len(t), np.float64)
    pick = .23 if lead else .19
    decay = 3.9 if lead else (3.6 if part == "bass" else 2.5)
    amplitudes = []
    for harmonic in range(1, 15):
        if frequency * harmonic > 6500:
            break
        amplitude = math.sin(math.pi * harmonic * pick) / harmonic ** 1.45
        amplitude *= math.exp(-harmonic * .085)
        amplitudes.append(abs(amplitude))
        damping = np.exp(-t * harmonic ** .65 / decay)
        tone += amplitude * damping * np.sin(phase * harmonic * (1 + .000009 * harmonic ** 2))
    tone /= sum(amplitudes)
    attack = .013 if lead else .009
    envelope = np.sin(np.minimum(t / attack, 1) * np.pi / 2) ** 2
    # Smooth finger damping at release avoids both hard cuts and a pad-like wash.
    off = np.clip((t - hold) / release, 0, 1)
    envelope *= np.cos(off * np.pi / 2) ** 2
    return (tone * envelope).astype(np.float32)


def render_score(out, seconds):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames = round(seconds * SAMPLE_RATE)
    audio = np.zeros((frames, 2), dtype=np.float32)
    events = score_events(seconds)
    for event in events:
        mono = pluck(event["midi"], event["hold"], event["part"]) * event["gain"]
        start = round(event["start"] * SAMPLE_RATE)
        count = min(len(mono), frames - start)
        pan = (event["pan"] + 1) * np.pi / 4
        audio[start:start + count, 0] += mono[:count] * math.cos(pan)
        audio[start:start + count, 1] += mono[:count] * math.sin(pan)

    # Quiet early reflections and a diffuse, decaying room tail. There is no
    # metronome, percussion or repeated synth pulse underneath the melody.
    dry = audio.copy()
    for index, delay in enumerate((.031, .043, .071, .113, .173, .251, .367, .503, .691, .887, 1.117, 1.409)):
        offset = round(delay * SAMPLE_RATE)
        reflected = dry[:-offset, ::-1] if index % 2 else dry[:-offset]
        audio[offset:] += reflected * (.11 * math.exp(-delay / .49))
    fade_in = round(.09 * SAMPLE_RATE)
    fade_out = round(2.5 * SAMPLE_RATE)
    audio[:fade_in] *= np.linspace(0, 1, fade_in)[:, None]
    audio[-fade_out:] *= np.cos(np.linspace(0, np.pi / 2, fade_out))[:, None] ** 2
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(audio).all() or peak <= 0:
        raise RuntimeError("Invalid rendered samples")
    audio *= .78 / peak
    raw = out / "quiet-courtyard-mix.wav"
    with wave.open(str(raw), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(np.round(audio * 32767).astype("<i2").tobytes())

    target = "loudnorm=I=-19:TP=-2:LRA=9"
    measured = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(raw),
                               "-af", target + ":print_format=json", "-f", "null", "-"],
                              text=True, capture_output=True, check=True).stderr
    values = json.loads(measured[measured.rfind("{"):])
    normalize = target + (f":measured_I={values['input_i']}:measured_TP={values['input_tp']}"
                          f":measured_LRA={values['input_lra']}:measured_thresh={values['input_thresh']}"
                          f":offset={values['target_offset']}:linear=true")
    master = out / "quiet-courtyard.wav"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
                    "-af", normalize, "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(master)], check=True)
    metadata = {"title": TITLE, "file": master.name, "seconds": seconds,
                "description": "Original slow melody over soft digital plucked strings, with a gentle final cadence",
                "composition": "Original melody and accompaniment, authored for this Blocks video",
                "sound_source": "Locally synthesized; no imported recordings, samples or existing score",
                "key": "D major", "meter": "4/4", "bars": len(BARS),
                "quarter_bpm_average": round(60 * 64 / (seconds - min(4.5, seconds * .065) - .25), 2),
                "phrasing": "Rubato, four four-bar phrases; final bar slows and rings out",
                "sample_rate": SAMPLE_RATE, "channels": 2, "notes": len(events),
                "mastering_filter": normalize,
                "sha256": hashlib.sha256(master.read_bytes()).hexdigest(),
                "renderer": "scripts/render_peaceful_score.py",
                "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "quiet-courtyard-score.json").write_text(json.dumps({**metadata, "events": events}, indent=2) + "\n")
    return metadata


def mux_score(video, soundtrack, destination):
    if Path(video).resolve() == Path(destination).resolve():
        raise ValueError("Use a distinct destination to preserve the source video")
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(video), "-i", str(soundtrack), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE),
                    "-shortest", "-movflags", "+faststart", "-metadata", "comment=Original score: Quiet Courtyard",
                    str(destination)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=70)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if bool(args.video) != bool(args.destination):
        parser.error("--video and --destination must be provided together")
    metadata = render_score(args.output, args.seconds)
    if args.video:
        mux_score(args.video, args.output / metadata["file"], args.destination)
        metadata["video"] = str(args.destination)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
