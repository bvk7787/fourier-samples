"""A synthetic sample library: every file generated from a seed, no downloads, no licenses.

It stands in for a real library wherever Fourier must run without one: the CI golden
build (tests/golden), the demo (`fourier demo`, later), and the DSP tests. Every file is a
known case with the metadata a classifier would give it, so a build over it exercises the
whole pipeline: routing, clustering, naming, the export DSP, kits, slices and verify.

    files = generate(root)            # writes WAVs under root, returns [SynthFile]

Each SynthFile carries its library-relative path and the metadata the fake Sononym rows
use (categories, classes, bpm, base note, descriptors). The defect cases (a phase-flipped
stereo file, a note 40 cents flat, a hot start, a long tail) are there to be fixed by the
export DSP.

The output depends only on the seed and this code: the same seed gives byte-identical
WAVs on every machine.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 44100


@dataclass
class SynthFile:
    rel: str                         # library-relative path
    categories: list                 # Sononym-style category labels
    classes: list                    # ["OneShot"] or ["Loop"]
    bpm: float | None = None
    note: float | None = None        # MIDI base note
    brightness: float = 0.5
    noisiness: float = 0.3
    harmonicity: float = 0.5
    ableton_tags: list = field(default_factory=list)
    duration_s: float = 0.0
    channels: int = 1


def _rng(*key) -> np.random.Generator:
    h = hashlib.sha256(repr(key).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _env(n, attack=0.002, decay=0.2, sr=SR):
    t = np.arange(n) / sr
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / max(decay, 1e-3))


def _midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _norm(y, peak=0.8):
    m = float(np.max(np.abs(y))) or 1.0
    return (y / m * peak).astype(np.float32)


# --- one-shots -------------------------------------------------------------

def kick(r, v):
    n = int(SR * (0.35 + 0.1 * v))
    t = np.arange(n) / SR
    f = 45 + 110 * np.exp(-t * (25 + 10 * v))
    return _norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.18 + 0.04 * v)
                 + 0.02 * r.standard_normal(n) * _env(n, 0.0005, 0.01))


def snare(r, v):
    n = int(SR * 0.3)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * (180 + 20 * v) * t) * _env(n, 0.001, 0.05)
    return _norm(tone + (0.6 + 0.1 * v) * r.standard_normal(n) * _env(n, 0.001, 0.09 + 0.02 * v))


def clap(r, v):
    n = int(SR * 0.3)
    y = np.zeros(n)
    for k in range(3):
        o = int(SR * 0.011 * k)
        y[o:] += r.standard_normal(n - o) * _env(n - o, 0.0005, 0.02 if k < 2 else 0.12)
    return _norm(y)


def hat(r, v, open_=False):
    n = int(SR * (0.6 if open_ else 0.09))
    x = r.standard_normal(n)
    x = x - np.concatenate([[0], x[:-1]]) * (0.7 + 0.05 * v)       # bright
    return _norm(x * _env(n, 0.0005, 0.25 if open_ else 0.025))


def tom(r, v):
    n = int(SR * 0.45)
    t = np.arange(n) / SR
    f = (90 + 30 * v) * (1 + 0.3 * np.exp(-t * 20))
    return _norm(np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.15))


def perc(r, v):
    n = int(SR * 0.2)
    t = np.arange(n) / SR
    return _norm(np.sin(2 * np.pi * (600 + 150 * v) * t) * _env(n, 0.0005, 0.03)
                 + 0.2 * r.standard_normal(n) * _env(n, 0.0005, 0.004))


def cymbal(r, v, ride=False):
    n = int(SR * (1.5 if ride else 2.2))
    x = r.standard_normal(n)
    x = x - 0.8 * np.concatenate([[0], x[:-1]])
    return _norm(x * _env(n, 0.001, 0.5 if ride else 0.9))


def tone(r, note, secs, shape="saw", decay=0.6, attack=0.005, detune=0.0):
    n = int(SR * secs)
    t = np.arange(n) / SR
    f = _midi_hz(note) * 2 ** (detune / 1200)
    ph = (f * t) % 1.0
    y = {"saw": 2 * ph - 1, "sine": np.sin(2 * np.pi * ph),
         "square": np.sign(np.sin(2 * np.pi * ph))}[shape]
    return _norm(y * _env(n, attack, decay))


def pad(r, note, v):
    n = int(SR * 3.0)
    t = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * _midi_hz(note + iv) * (1 + d) * t)
            for iv in (0, 4, 7) for d in (-0.003, 0.003))
    return _norm(y * np.clip(t / 0.6, 0, 1) * np.exp(-t / 2.5))


def stab(r, note, v):
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    y = sum(2 * ((_midi_hz(note + iv) * t) % 1) - 1 for iv in (0, 3, 7))
    return _norm(y * _env(n, 0.002, 0.12))


def keys(r, note, v):
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    y = sum(np.sin(2 * np.pi * _midi_hz(note) * k * t) / k ** 1.5 * np.exp(-t * k * 1.2)
            for k in range(1, 7))
    return _norm(y * np.clip(t / 0.003, 0, 1))


def vox(r, note, v):
    n = int(SR * 1.2)
    t = np.arange(n) / SR
    f0 = _midi_hz(note) * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    src = 2 * ((np.cumsum(f0) / SR) % 1) - 1
    y = sum(np.convolve(src, np.sin(2 * np.pi * fm * np.arange(64) / SR) * np.hanning(64), "same")
            for fm in (700 + 100 * v, 1200, 2500))
    return _norm(y * np.clip(t / 0.05, 0, 1) * np.clip((1.2 - t) / 0.1, 0, 1))


def riser(r, v):
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    f = 200 * 2 ** (t * (2 + v))
    return _norm((np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.3 * r.standard_normal(n)) * (t / t[-1]))


def blip(r, v):
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    return _norm(np.sin(2 * np.pi * (1500 + 400 * v) * t * (1 + 3 * t)) * _env(n, 0.0005, 0.04))


def cycle(r, v):
    n = 1024
    k = np.arange(1, 12)
    ph = np.arange(n) / n
    amps = r.random(len(k)) / k ** (0.5 + 0.2 * v)
    return _norm(sum(a * np.sin(2 * np.pi * kk * ph) for a, kk in zip(amps, k)))


def acoustic(r, note, v):
    n = int(SR * 2.0)
    t = np.arange(n) / SR
    f = _midi_hz(note) * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))
    ph = np.cumsum(f) / SR
    y = sum(np.sin(2 * np.pi * k * ph) / k for k in range(1, 9))
    return _norm(y * np.clip(t / 0.08, 0, 1) * np.exp(-t / 1.6))


# --- loops -----------------------------------------------------------------

def drum_loop(r, bpm, bars, v, lead_in_s=0.0):
    beat = 60.0 / bpm
    n = int(SR * (lead_in_s + beat * 4 * bars))
    y = np.zeros(n)
    k, s, h = kick(r, v), snare(r, v), hat(r, v)
    for b in range(4 * bars):
        o = int(SR * (lead_in_s + b * beat))
        for hit, when in ((k, b % 4 in (0, 2) or (v > 1 and b % 4 == 3)), (s, b % 4 in (1, 3))):
            if when:
                m = min(len(hit), n - o)
                y[o:o + m] += hit[:m]
        for e in (0, 0.5):
            oo = o + int(SR * beat * e)
            m = min(len(h), n - oo)
            if m > 0:
                y[oo:oo + m] += 0.4 * h[:m]
    return _norm(y)


def phrase(r, bpm, note, v):
    beat = 60.0 / bpm
    n = int(SR * beat * 8)
    y = np.zeros(n)
    for i, iv in enumerate((0, 0, 7, 5, 0, 3, 7, 10)):
        o = int(SR * i * beat)
        tt = tone(r, note + iv, beat * 0.9, "saw", decay=0.15)
        m = min(len(tt), n - o)
        y[o:o + m] += tt[:m]
    return _norm(y)


# --- the library -----------------------------------------------------------

def _plan():
    """(rel path, SynthFile fields, builder) for every file."""
    out = []

    counts = {}

    def add(rel, builder, **meta):
        # four packs per kind, in turn: a real library has several vendors per kind,
        # and curation caps any one vendor's share of a folder
        kind = re.sub(r"\d.*$", "", rel)            # "…/Hats/Open Hat 03.wav" -> "…/Hats/Open Hat "
        i = counts[kind] = counts.get(kind, -1) + 1
        top, rest = rel.split("/", 1)
        out.append((f"{top} {'ABCD'[i % 4]}/{rest}", builder, meta))

    for i in range(12):
        add(f"Synth Drums/Kit/Kicks/Kick {i + 1:02d}.wav", lambda r, i=i: kick(r, i % 4),
            categories=["Perc Kicks"], classes=["OneShot"], brightness=0.2, noisiness=0.1)
        add(f"Synth Drums/Kit/Snares/Snare {i + 1:02d}.wav", lambda r, i=i: snare(r, i % 4),
            categories=["Perc Snares"], classes=["OneShot"], brightness=0.6, noisiness=0.7)
        add(f"Synth Drums/Kit/Claps/Clap {i + 1:02d}.wav", lambda r, i=i: clap(r, i % 4),
            categories=["Perc Claps"], classes=["OneShot"], brightness=0.6, noisiness=0.8)
        add(f"Synth Drums/Kit/Hats/Closed Hat {i + 1:02d}.wav", lambda r, i=i: hat(r, i % 4),
            categories=["Perc Hats & Shakers"], classes=["OneShot"], brightness=0.9, noisiness=0.9)
        add(f"Synth Drums/Kit/Hats/Open Hat {i + 1:02d}.wav", lambda r, i=i: hat(r, i % 4, True),
            categories=["Perc Hats & Shakers"], classes=["OneShot"], brightness=0.9, noisiness=0.9)
        add(f"Synth Drums/Kit/Toms/Tom {i + 1:02d}.wav", lambda r, i=i: tom(r, i),
            categories=["Perc Toms"], classes=["OneShot"], brightness=0.3, noisiness=0.2)
        add(f"Synth Drums/Kit/Percussion/Wood Hit {i + 1:02d}.wav", lambda r, i=i: perc(r, i % 4),
            categories=["Perc Wood Hits"], classes=["OneShot"], brightness=0.6, noisiness=0.3)
        add(f"Synth Drums/Kit/Cymbals/Crash {i + 1:02d}.wav", lambda r, i=i: cymbal(r, i % 4),
            categories=["Perc Cymbal Crashes"], classes=["OneShot"], brightness=0.9, noisiness=0.9)
        add(f"Synth Drums/Kit/Cymbals/Ride {i + 1:02d}.wav", lambda r, i=i: cymbal(r, i % 4, True),
            categories=["Perc Cymbal Rides"], classes=["OneShot"], brightness=0.8, noisiness=0.8)
        note = 36 + (i * 5) % 12
        add(f"Synth Tones/Bass/Sub Bass {i + 1:02d} C.wav", lambda r, n=note: tone(r, n, 1.0, "sine", 0.5),
            categories=["Tone Bass & LowKeys"], classes=["OneShot"], note=note, brightness=0.1,
            noisiness=0.05, harmonicity=0.9)
        n2 = 60 + (i * 7) % 12
        add(f"Synth Tones/Leads/Saw Lead {i + 1:02d}.wav", lambda r, n=n2, i=i: tone(r, n, 0.8, "saw", 0.4),
            categories=["Tone Leads & MidHiKeys"], classes=["OneShot"], note=n2, brightness=0.7,
            harmonicity=0.8)
        add(f"Synth Tones/Pads/Warm Pad {i + 1:02d}.wav", lambda r, n=n2, i=i: pad(r, n - 12, i),
            categories=["Tone Pads & Textures"], classes=["OneShot"], note=n2 - 12, brightness=0.4,
            harmonicity=0.8)
        add(f"Synth Tones/Stabs/Chord Stab {i + 1:02d}.wav", lambda r, n=n2, i=i: stab(r, n, i),
            categories=["Tone Stabs & Orch. Hits"], classes=["OneShot"], note=n2, brightness=0.6,
            harmonicity=0.7)
        add(f"Synth Tones/Blips/Blip {i + 1:02d}.wav", lambda r, i=i: blip(r, i),
            categories=["Tone Blips & HighKeys"], classes=["OneShot"], brightness=0.9, harmonicity=0.6)
        add(f"Synth Tones/Vocals/Vowel {i + 1:02d}.wav", lambda r, n=48 + i % 12, i=i: vox(r, n, i % 4),
            categories=["Tone Voice & Acapella"], classes=["OneShot"], note=48 + i % 12, harmonicity=0.7)
        add(f"Synth FX/Risers/Riser {i + 1:02d}.wav", lambda r, i=i: riser(r, i % 4),
            categories=["XFX Sweeps & Lasers"], classes=["OneShot"], brightness=0.7, noisiness=0.5)
        add(f"Synth Tones/Wavetables/Single Cycle {i + 1:02d}.wav", lambda r, i=i: cycle(r, i % 4),
            categories=["Tone Leads & MidHiKeys"], classes=["OneShot"])
        n3 = 50 + i
        add(f"Synth Keys/Grand Piano/Piano {i + 1:02d}.wav", lambda r, n=n3: keys(r, n, 0),
            categories=["Tone Leads & MidHiKeys"], classes=["OneShot"], note=n3, harmonicity=0.9,
            ableton_tags=["Piano"])
        add(f"Synth Strings/Violin/Violin Sustain {i + 1:02d}.wav", lambda r, n=n3 + 12: acoustic(r, n, 0),
            categories=["Tone Leads & MidHiKeys"], classes=["OneShot"], note=n3 + 12, harmonicity=0.9,
            ableton_tags=["Violin"])
    for bpm in (90, 128, 174):
        for i in range(8):
            add(f"Synth Loops/Drum Loops/Break {bpm} {i + 1:02d}.wav",
                lambda r, b=bpm, i=i: drum_loop(r, b, 2, i % 4, lead_in_s=0.05 if i == 0 else 0.0),
                categories=["Perc Kicks"], classes=["Loop"], bpm=float(bpm),
                ableton_tags=["Drum Loop"])
            add(f"Synth Loops/Bass Loops/Bassline {bpm} {i + 1:02d} C.wav",
                lambda r, b=bpm, i=i: phrase(r, b, 36 + i % 5, i),
                categories=["Tone Bass & LowKeys"], classes=["Loop"], bpm=float(bpm), note=36.0,
                harmonicity=0.8)
    # defect cases the export DSP must fix
    add("Synth Tones/Leads/Flat Lead C.wav", lambda r: tone(r, 60, 0.8, "saw", 0.4, detune=-40),
        categories=["Tone Leads & MidHiKeys"], classes=["OneShot"], note=59.6, harmonicity=0.8)
    add("Synth Drums/Kit/Kicks/Hot Start Kick.wav", lambda r: np.concatenate([[0.9], kick(r, 1)[1:]]).astype(np.float32),
        categories=["Perc Kicks"], classes=["OneShot"])
    add("Synth Tones/Pads/Long Tail Pad.wav",
        lambda r: np.concatenate([pad(r, 48, 0), 1e-4 * r.standard_normal(SR * 4)]).astype(np.float32),
        categories=["Tone Pads & Textures"], classes=["OneShot"], note=48.0)
    add("Synth Tones/Pads/Phase Flipped Pad.wav", lambda r: np.stack([pad(r, 50, 1), -pad(r, 50, 1)], 1),
        categories=["Tone Pads & Textures"], classes=["OneShot"], note=50.0)
    return out


def generate(root, seed: int = 0) -> list[SynthFile]:
    """Write the library under root (16-bit WAVs) and return its files, in a stable order."""
    import soundfile as sf
    root = Path(root)
    files = []
    for rel, builder, meta in _plan():
        y = builder(_rng(seed, rel))
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, y, SR, subtype="PCM_16")
        files.append(SynthFile(rel=rel, duration_s=len(y) / SR, channels=1 if y.ndim == 1 else y.shape[1],
                               **meta))
    return files
