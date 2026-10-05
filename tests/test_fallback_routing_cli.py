"""The routing fixes of tests/test_fallback_routing.py through the real CLI (one process per
command, the sandbox and stand-in CLAP of tests/test_other_libraries_cli.py: CLAP hears each
file as the sound it is): keys named by their folder land in PIANO, a flat folder's sustained
pads in PADS and its loops at the tempo their length implies, lo-fi drum loops named as
phrases in DRUMLOOPS, 808s named for the note they play once retuned, and shakers in a
percussion folder in PERC; verify passes and `why` says each reason."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_first_run import SR, Sandbox, _flat, _sounds  # noqa: E402
from test_other_libraries_cli import RUNNER  # noqa: E402

# one sandbox, built once, that the tests below change in turn: one worker runs them all, in
# order (pytest -n: --dist loadgroup, conftest.py)
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("fallback_routing_cli")]

ROOT = Path(__file__).resolve().parents[1]
PC = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _hz(note, octave):
    return 440.0 * 2 ** ((PC[note] + 12 * (octave + 1) - 69) / 12)


def make_library(lib: Path) -> dict:
    kick, snare, hat, clap, _loop, _bass, _pad = _sounds(np.random.default_rng(5))
    hears = {}

    def w(rel, y, sound):
        p = lib / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (0.8 * y / np.max(np.abs(y))).astype("float32"), SR, subtype="PCM_16")
        hears[rel] = f"{sound} {rel.rsplit('/', 1)[-1]}"

    def drums(bpm, bars, i, lofi=False):
        beat = 60.0 / bpm * SR
        n = int(round(beat * 4 * bars))
        y = 0.01 * np.random.default_rng(i).standard_normal(n)
        for b in range(4 * bars):
            at = int(round(b * beat))
            hit = (kick(i) if b % 2 == 0 else snare(i))[:int(beat // 2)]
            y[at:at + len(hit)] += hit[:max(0, n - at)]
            h = hat(i)[:int(beat // 4)]
            at2 = at + int(beat // 2)
            y[at2:at2 + len(h)] += 0.5 * h[:max(0, n - at2)]
        if lofi:                     # filtered drums under a chord: they read tonal
            b_, a_ = butter(2, 900 / (SR / 2))
            t = np.arange(n) / SR
            y = lfilter(b_, a_, y) + 0.27 * sum(np.sin(2 * np.pi * f * t) for f in (220, 261.6, 329.6))
        return y

    def keys(f0):
        t = np.arange(int(2.5 * SR)) / SR
        y = sum(np.sin(2 * np.pi * f0 * m * t) * (0.6 ** k) for k, m in enumerate((1, 2, 3, 4)))
        return (y + sum(np.sin(2 * np.pi * f0 * r * t) for r in (1.189, 1.498))) * np.exp(-t / 0.8)

    def pad(i):
        seg = int(1.5 * SR)
        tt = np.arange(seg) / SR
        return np.concatenate([sum(np.sin(2 * np.pi * f * (1 + 0.01 * i) * tt) for f in ch)
                               * np.minimum(1, tt / 0.3 + 0.2)
                               for ch in ((220, 277, 330), (196, 247, 294), (175, 220, 262), (247, 311, 370))])

    def sub(f0):
        t = np.arange(int(1.2 * SR)) / SR
        f = f0 * (1 + 0.5 * np.exp(-t / 0.02))
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5)

    for i, (name, note) in enumerate((("Tine C", "C"), ("Tine F", "F"), ("Bright G", "G"),
                                      ("Mellow A", "A"), ("Warm D", "D"), ("Soft E", "E"))):
        w(f"Keys/{name}.wav", keys(_hz(note, 3)), "rhodes electric piano note")
    def swell_pad(i):
        """A pad changing chords every 0.75 s with a noisy swell at each change: held (no
        silence), slow onsets, tonal, but harmonic enough only for a loop by its onsets."""
        seg = int(0.75 * SR)
        tt = np.arange(seg) / SR
        rng = np.random.default_rng(i)
        out = []
        for k in range(8):
            ch = ((220, 277, 330), (196, 247, 294), (175, 220, 262), (247, 311, 370))[k % 4]
            y = sum(np.sin(2 * np.pi * f * (1 + 0.01 * i) * tt) for f in ch) * np.minimum(1, tt / 0.08 + 0.3)
            c = int(0.08 * SR)
            y[:c] += 8 * rng.standard_normal(c) * np.exp(-np.arange(c) / (0.08 / 3 * SR))
            out.append(y)
        return np.concatenate(out)

    for i in range(6):
        w(f"Dump/{101 + i}.wav", pad(i), "warm pad texture")
        w(f"Dump/{201 + i}.wav", drums(88 if i % 2 else 90, 4, 40 + i), "funk drum break")
        w(f"Dump/{301 + i}.wav", swell_pad(i), "warm pad texture")
    for v in ("Acme", "Northwind", "Vendor C"):
        for i in range(2):
            w(f"{v} Dusty/Loops/{v} Dust {i + 1} 85 BPM.wav", drums(85, 4, 60 + i + 3 * len(v), lofi=True),
              "boom bap break drum loop")
    for name, note in (("Sub_C", "C"), ("Sub_A#", "A#"), ("Sub_F", "F"), ("Sub_G", "G"), ("Sub_D", "D"),
                       ("Sub_Eb", "D#")):
        w(f"Bass/808s/{name}.wav", sub(_hz(note, 1)), "808 sub bass")
    # 6: lo-fi drums in a folder named only by a tempo; keys grooves in a drum-loop folder
    for i in range(4):
        w(f"Loops/85/loop_{i + 17}.wav", drums(85, 4, 80 + i, lofi=True), "boom bap break drum loop")
        beat = 60.0 / 90 * SR
        t = np.arange(int(round(beat * 16))) / SR
        stabs = np.zeros_like(t)
        for b in range(16):
            at = int(round(b * beat))
            stabs[at:at + int(beat * 0.8)] = np.exp(-np.arange(min(int(beat * 0.8), len(t) - at)) / (0.2 * SR))
        ch = sum(np.sin(2 * np.pi * f * (1 + 0.01 * i) * t) for f in (220, 277, 330, 415))
        w(f"Drum Loops/Rhodes Vamp 90 BPM {i + 1:02d}.wav", ch * stabs, "rhodes chord progression")
    # 8: recognized, in a category that is off (the build's config): placed nowhere
    for i in range(2):
        t = np.arange(int(0.15 * SR)) / SR
        w(f"Blips/Blip {i + 1:02d}.wav", np.sign(np.sin(2 * np.pi * (900 + 200 * i) * t)) * np.exp(-t / 0.05),
          "blip beep")
    for i in range(6):
        y = hat(i + 7)
        w(f"Perc/Shaker {i + 11}.wav", y * np.linspace(1, 0.2, len(y)), "shaker percussion")
        v = ("Acme", "Northwind", "Vendor C")[i % 3]
        w(f"Drums/{v}/Kicks/Kick {i + 1:02d}.wav", kick(i), "kick drum")
        w(f"Drums/{v}/Snares/Snare {i + 1:02d}.wav", snare(i), "snare drum")
        w(f"Drums/{v}/Hats/Closed Hat {i + 1:02d}.wav", hat(i), "closed hi-hat")
        w(f"Drums/{v}/Claps/Clap {i + 1:02d}.wav", clap(i), "hand clap")
    return hears


class Box(Sandbox):
    def __init__(self, tmp: Path):
        holder = {}

        def make(lib):
            holder.update(make_library(lib))
            return len(holder)
        super().__init__(tmp, make=make)
        (tmp / "hears.json").write_text(json.dumps(holder))
        self.runner.write_text(RUNNER.format(src=str(ROOT / "src"), golden=str(ROOT / "tests" / "golden"),
                                             concepts=str(tmp / "hears.json")))

    def manifest(self):
        return json.loads((self.master / "manifest.json").read_text())


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return Box(tmp_path_factory.mktemp("fallback-routing"))


def _per(man):
    return {c: {e["src"].split("SampleLibrary/")[1]: e for e in v.get("entries") or ()}
            for c, v in man["categories"].items() if v.get("entries")}


def test_a_library_without_sononym_routes_and_names_what_it_has(box):
    b = box
    b.run("setup", "--yes", "--no-clap", "--library", str(b.lib), "--device", "digitakt_2",
          "--preset", "balanced")
    cfg = b.user / ".config" / "fourier" / "fourier.toml"
    cfg.write_text(cfg.read_text() + 'categories = { BLIPS = "off" }\n')
    out = _flat(b.run("build", "--all", "-j", "1", "--no-describe"))
    assert "verify: PASS" in out, out[-2000:]
    per = _per(b.manifest())
    # 8: keys named by their folder are PIANO's
    assert sum(k.startswith("Keys/") for k in per.get("PIANO", {})) >= 4, per.get("PIANO")
    # 9: a flat folder's sustained pads
    assert sum(k.startswith("Dump/1") for k in per.get("PADS", {})) >= 4
    # ...and the ones only their onsets call loops, with no tempo they state: held, tonal
    assert sum(k.startswith("Dump/3") for k in per.get("PADS", {})) >= 2, per.get("PADS")
    assert not any(k.startswith("Dump/3") for c, es in per.items() if c != "PADS" for k in es)
    # 7: a flat folder's loops at the tempo their length implies
    loops = per["DRUMLOOPS"]
    dump = {k: e for k, e in loops.items() if k.startswith("Dump/2")}
    assert len(dump) >= 4 and all(e["bpm_src"] == "length" for e in dump.values()), dump
    # 6: lo-fi drum loops named as phrases are the drum loops'
    lofi = {k: e for k, e in loops.items() if "Dusty/" in k}
    assert len(lofi) >= 4 and all(e.get("rehomed") for e in lofi.values()), list(loops)
    # 12j: drum loops aren't "musical"
    assert not any("musical" in e["family"] for k, e in loops.items() if k.startswith("Dump/"))
    # 10: 808s named for the note they play; a second C says the note it was
    subs = {k.rsplit("/", 1)[-1]: e["out"].rsplit("/", 1)[-1] for k, e in per["SUB"].items()}
    assert subs["Sub_C.wav"] == "Sub_C.wav", subs
    assert subs["Sub_F.wav"] == "Sub_C_from-f.wav" and subs["Sub_A#.wav"] == "Sub_C_from-as.wav", subs
    assert not any(o.endswith("_2.wav") for o in subs.values())
    # 6: a tempo folder's lo-fi drums are the drum loops'; keys grooves in a drum-loop folder
    # that sound like a phrase fall back to PHRASES
    anon = {k: e for k, e in loops.items() if k.startswith("Loops/85/")}
    assert len(anon) >= 3, list(loops)
    ph = {k: e for k, e in per.get("PHRASES", {}).items() if k.startswith("Drum Loops/")}
    assert len(ph) >= 3 and all(e.get("rehomed") for e in ph.values()), (per.get("PHRASES"), list(loops))
    assert not any(k.startswith("Drum Loops/") for k in loops)
    # 12i: shakers in a percussion folder
    assert sum(k.startswith("Perc/") for k in per.get("PERC", {})) >= 4
    assert not any(k.startswith("Perc/") for k in per.get("HATS", {}))


def test_why_says_each_reason(box):
    b = box
    if not (b.master / "manifest.json").exists():
        pytest.skip("needs the build above")
    out = _flat(b.run("why", "--detail", "Keys/Tine C"))
    assert "decided by: keys -> PIANO" in out and "named as keys" in out
    out = _flat(b.run("why", "--detail", "Dump/201"))
    assert "tempo from length (4 bars" in out
    held = next(k for k in _per(b.manifest()).get("PADS", {}) if k.startswith("Dump/3"))
    out = _flat(b.run("why", "--detail", held))
    assert "decided by: clap -> PADS" in out and "labels: path: -; audio: class.loop" in out, out
    # shaken percussion: its path label and its votes say PERC, as its folder does
    shaker = next(k for k in _per(b.manifest())["PERC"] if k.startswith("Perc/"))
    out = _flat(b.run("why", "--detail", shaker))
    assert "path: perc.hand" in out and "PERC" in out and "HATS" not in out.split("votes:")[1], out
    out = _flat(b.run("why", "--detail", "Acme Dust 1"))
    assert "fell back to DRUMLOOPS by sound" in out
    out = _flat(b.run("why", "--detail", "Rhodes Vamp 90 BPM 01"))
    assert "fell back to PHRASES by sound" in out
    # 8: recognized, but no category took it
    out = _flat(b.run("why", "--detail", "--unrecognized"))
    assert "Blips/Blip 01.wav (recognized as blip, but no category took it)" in out, out
    # 8: the dry run's PIANO is what PIANO's own rules take, not a share of the leftovers
    out = b.run("build", "--dry-run")
    row = next(line for line in out.splitlines() if line.strip().startswith("PIANO"))
    assert len(_per(b.manifest()).get("PIANO", {})) <= int(row.split()[1]), row
