"""A first run end to end, the way a newcomer runs it: no Sononym, no Ableton Live, a plain
folder of WAVs. The real CLI, one process per command (as from a shell), in a sandboxed home,
config, master, publish root and device locks:

    fourier setup (--yes), doctor, tools scan, build --all (--dry-run; --no-scan; one that
    scans, analyzes and fails before any category, one that fails part way and --resume),
    verify, why, render, sync, publish (--dry-run), releases, render --release (--dry-run)

CLAP is the synthetic golden's stand-in (tests/golden/synthetic_build.py): the files' and
prompts' words point the same way, so no torch or model is needed. The library folder must be
byte-identical (path, size, mtime, sha256) after every command."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "golden"))
from synthetic_build import fingerprint_changes, library_fingerprint  # noqa: E402

SR = 44100
VENDORS = ("Vendor A", "Vendor B", "Vendor C")      # three, so no vendor cap empties a folder
FILLED = ("KICKS", "SNARES", "HATS", "CLAPS", "DRUMLOOPS", "SUB", "PADS")

# Runs one fourier command with the stand-in CLAP; FAIL_CATEGORIES=KICKS,PADS (or ALL) makes
# those categories' builds fail, as a crash part way through would.
RUNNER = """
import os, sys
sys.path[:0] = [{src!r}, {golden!r}]
import numpy as np
import synthetic_build as SB
import fourier.analysis.clap_features as CF


def embed_audio_file(path):
    rel = str(path).split("SampleLibrary/")[-1]
    v = SB.fake_embed_text(rel) + 0.3 * SB.unit("file", rel)
    # what CLAP hears that the words don't say: a four-on-the-floor beat a little like a
    # drum break (under the gate's minimum), a house loop's drums much like one
    for words, w in (("Four Floor", 0.2), ("House 120 Loop", 1.5)):
        if words in rel:
            v = v + w * SB.fake_embed_text("funk drum break")
    return (v / np.linalg.norm(v)).astype("float32")


CF.embed_text = SB.fake_embed_text
CF.embed_audio_file = embed_audio_file
fail = set(filter(None, os.environ.get("FAIL_CATEGORIES", "").split(",")))
if fail:
    from fourier.packs import curate
    real = curate.build_taxonomy

    def build_taxonomy(session, category, *a, **kw):
        if "ALL" in fail or category in fail:
            raise RuntimeError("stopped for the test")
        return real(session, category, *a, **kw)
    curate.build_taxonomy = build_taxonomy
from fourier.cli import main
if __name__ == "__main__":      # macOS spawns worker processes, which import this file
    main(sys.argv[1:], prog_name="fourier")
"""


def _env(n, decay):
    return np.exp(-np.arange(n) / SR / decay)


def _sounds(rng):
    def kick(i):
        n = int(0.35 * SR)
        f = 50 + 110 * np.exp(-np.arange(n) / SR / 0.03) * (1 + 0.08 * i)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.12)

    def snare(i):
        n = int(0.25 * SR)
        return (0.7 * rng.standard_normal(n) + np.sin(2 * np.pi * (180 + 12 * i) * np.arange(n) / SR)) * _env(n, 0.07)

    def hat(i):
        n = int(0.12 * SR)
        return rng.standard_normal(n) * _env(n, 0.02 + 0.004 * i)

    def clap(i):
        n = int(0.25 * SR)
        y = rng.standard_normal(n) * _env(n, 0.05)
        y[int(0.01 * SR):int(0.02 * SR)] *= 0.3 + 0.05 * i
        return y

    def loop(bpm, i):
        beat = int(60 / bpm * SR)
        y = np.zeros(beat * 4)
        for b in range(4):
            hit = (kick(i) if b % 2 == 0 else snare(i))[:beat // 2]
            y[b * beat:b * beat + len(hit)] += hit
            h = hat(i)[:beat // 4]
            y[b * beat + beat // 2:b * beat + beat // 2 + len(h)] += 0.5 * h
        return y

    def bass(i):
        t = np.arange(int(0.8 * SR)) / SR
        return np.sign(np.sin(2 * np.pi * 55 * (1 + 0.06 * i) * t)) * _env(len(t), 0.3)

    def pad(i):
        t = np.arange(int(1.5 * SR)) / SR
        return sum(np.sin(2 * np.pi * f * t) for f in (220 * (1 + 0.05 * i), 277, 330)) * np.minimum(1, t / 0.4)

    return kick, snare, hat, clap, loop, bass, pad


def make_library(root: Path) -> int:
    """Kicks, snares, hats, claps, drum loops at two tempos, basses and pads: six of each
    kind (twelve loops) across three vendors."""
    kick, snare, hat, clap, loop, bass, pad = _sounds(np.random.default_rng(7))
    n = 0

    def w(rel, y):
        nonlocal n
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (0.8 * y / np.max(np.abs(y))).astype("float32"), SR, subtype="PCM_16")
        n += 1

    for i in range(6):
        v = VENDORS[i % 3]
        w(f"{v}/Drums/Kicks/Kick {i + 1:02d}.wav", kick(i))
        w(f"{v}/Drums/Snares/Snare {i + 1:02d}.wav", snare(i))
        w(f"{v}/Drums/Hats/Closed Hat {i + 1:02d}.wav", hat(i))
        w(f"{v}/Drums/Claps/Clap {i + 1:02d}.wav", clap(i))
        w(f"{v}/Loops/Drum Loops/Break {i + 1:02d} 120bpm.wav", loop(120, i))
        w(f"{v}/Loops/Drum Loops/Break {i + 7:02d} 174bpm.wav", loop(174, i))
        w(f"{v}/Bass/Bass {i + 1:02d} C.wav", bass(i))
        w(f"{v}/Pads/Pad {i + 1:02d} A.wav", pad(i))
    return n


def _sustained(rng):
    """Sounds with onsets and no silent gap, as a groove has: a riser that steps up in
    pitch, a pad that changes chords, a legato bassline."""
    def riser(i):
        t = np.arange(int(3.0 * SR)) / SR
        f = 150 * (1.12 + 0.02 * i) ** np.floor(t / 0.25)
        y = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.5 + 0.3 * rng.standard_normal(len(t))
        return y * (0.2 + t / 3)

    def pad(i):
        seg = int(0.6 * SR)
        tt = np.arange(seg) / SR
        chords = [(220, 277, 330), (196, 247, 294), (175, 220, 262), (247, 311, 370),
                  (220, 262, 330), (196, 247, 294)]
        return np.concatenate([sum(np.sin(2 * np.pi * f * (1 + 0.01 * i) * tt) for f in ch)
                               * np.minimum(1, tt / 0.08 + 0.3) for ch in chords])

    def bassline(i):
        seg = int(0.25 * SR)
        f = np.repeat(np.array([55, 55, 65.4, 73.4, 55, 82.4, 73.4, 65.4] * 2) * (1 + 0.03 * i), seg)
        ph = 2 * np.pi * np.cumsum(f) / SR
        return ((ph / np.pi) % 2 - 1) * np.tile(0.5 + 0.5 * np.exp(-np.arange(seg) / SR / 0.08), 16)

    return riser, pad, bassline


def make_small_library(root: Path) -> int:
    """A small library from three vendors, laid out the way packs are, whose pack names say
    the opposite of what some files are: a "Breaks and Hits" pack's kicks and snares, a
    "Jungle Breaks" pack's risers, pads and basslines beside its loops (all with onsets and
    no silence between them, as its grooves have), and a "Drum Hits" pack's loops, among
    them four-on-the-floor beats that sound less like breaks than the CLAP gate wants, and
    house loops named only by their tempo, a chord under the drums (too harmonic for the
    drum loops' harmonicity gate)."""
    kick, snare, hat, clap, loop, bass, _pad = _sounds(np.random.default_rng(3))
    riser, pad, bassline = _sustained(np.random.default_rng(11))
    n = 0

    def w(rel, y):
        nonlocal n
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (0.8 * y / np.max(np.abs(y))).astype("float32"), SR, subtype="PCM_16")
        n += 1

    def groove(bpm, i, bars):
        beat = int(60 / bpm * SR)
        y = 0.05 * np.random.default_rng(i).standard_normal(beat * 4 * bars)    # never silent
        for b in range(4 * bars):
            hit = (kick(i) if b % 2 == 0 else snare(i))[:beat]
            y[b * beat:b * beat + len(hit)] += hit
        return y

    def four_floor(i):
        beat = int(60 / 124 * SR)
        y = 0.03 * np.random.default_rng(i).standard_normal(beat * 16)
        for b in range(16):
            parts = [(0, kick(i)[:beat], 1.0), (beat // 2, hat(i)[:beat // 2], 0.6)]
            parts += [(0, clap(i)[:beat // 2], 0.7)] if b % 2 else []
            for at, hit, g in parts:
                y[b * beat + at:b * beat + at + len(hit)] += g * hit
        return y

    def house(i):
        beat = int(60 / 120 * SR)
        t = np.arange(beat * 16) / SR
        y = 0.02 * np.random.default_rng(i).standard_normal(len(t))
        for b in range(16):
            for at, hit in ((0, kick(i)[:beat]), (beat // 2, 0.4 * hat(i)[:beat // 4])):
                y[b * beat + at:b * beat + at + len(hit)] += hit
        return y + 0.35 * sum(np.sin(2 * np.pi * f * (1 + 0.01 * i) * t) for f in (110, 220, 277, 330))

    a, nw, c = "Acme Audio/Acme Breaks and Hits", "Northwind Loops/Jungle Breaks 174", "Vendor C/Drum Hits"
    for i in range(6):
        w(f"{a}/Kicks/Kick {i + 1:02d}.wav", kick(i))
        w(f"{a}/Snares/Snare {i + 1:02d}.wav", snare(i))
        w(f"{a}/Hats/Hat {i + 1:02d}.wav", hat(i))
        w(f"{c}/Claps/Clap {i + 1:02d}.wav", clap(i))
    t = np.arange(int(2.0 * SR)) / SR
    for i in range(3):
        w(f"{nw}/Loops/Jungle {i + 1:02d} 174bpm.wav", groove(174, i, 4))
        w(f"{nw}/FX/Riser {i + 1:02d}.wav", np.sin(2 * np.pi * (200 + (800 + 50 * i) * t) * t) * t / 2)
        w(f"{c}/Beats/Four Floor {i + 1:02d} 124bpm.wav", four_floor(i + 20))
        w(f"{c}/Loops/WAV/House 120 Loop {i + 1:02d}.wav", house(i + 30))
    for i in range(2):
        w(f"{nw}/FX/Riser {i + 4:02d}.wav", riser(i))
        w(f"{nw}/Pads/Pad {i + 1:02d}.wav", pad(i))
        w(f"{nw}/Bass/Bass {i + 5:02d}.wav", bassline(i))
        w(f"{c}/Loops/Groove {i + 1:02d} 120bpm.wav", groove(120, i + 10, 3))
    for i in range(4):
        w(f"{c}/Bass/Bass {i + 1:02d} C.wav", bass(i))
    return n


class Sandbox:
    def __init__(self, tmp: Path, make=None):
        self.tmp = tmp
        self.lib = tmp / "lib" / "SampleLibrary"
        self.home = tmp / "home"                       # $FOURIER_HOME
        self.user = tmp / "user"                       # $HOME: ~/.config/fourier, ~/Music/Fourier
        self.master = tmp / "out" / "FourierCurated"
        self.renders = tmp / "out" / "FourierRenders"
        self.locks = tmp / "locks"
        self.user.mkdir(parents=True)
        self.runner = tmp / "run_fourier.py"
        self.runner.write_text(RUNNER.format(src=str(ROOT / "src"), golden=str(ROOT / "tests" / "golden")))
        self.n_files = (make or make_library)(self.lib)
        self.print = library_fingerprint(self.lib)
        self.log = []

    def env(self, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("FOURIER_")}
        env.update(HOME=str(self.user), FOURIER_HOME=str(self.home),
                   FOURIER_CURATED_DIR=str(self.master), FOURIER_LOCK_DIR=str(self.locks),
                   FOURIER_PITCH_CACHE=str(self.tmp / "pitch"), FOURIER_NO_AUDIO_CACHE="1",
                   FOURIER_NO_RENDER_CACHE="1", **extra)
        return env

    def run(self, *args, ok=True, **extra):
        r = subprocess.run([sys.executable, str(self.runner), *args], env=self.env(**extra),
                           cwd=self.tmp, capture_output=True, text=True, timeout=900)
        out = r.stdout + r.stderr
        self.log.append((args, r.returncode, out))
        assert "Traceback" not in out, f"fourier {' '.join(args)}:\n{out[-3000:]}"
        if ok is not None:
            assert (r.returncode == 0) == ok, f"fourier {' '.join(args)} -> {r.returncode}:\n{out[-3000:]}"
        changed = fingerprint_changes(self.print, library_fingerprint(self.lib))
        assert not changed, f"fourier {' '.join(args)} changed the library: {changed[:5]}"
        return out

    def db(self, sql):
        import duckdb
        con = duckdb.connect(str(self.home / "library.duckdb"), read_only=True)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    return Sandbox(tmp_path_factory.mktemp("first-run"))


def _without_tmp_path(out, tmp_path):
    """Remove a fixture's path, including terminal-wrapped spellings."""
    import re
    pattern = r"\s*".join(re.escape(c) for c in str(tmp_path))
    return re.sub(pattern, "", out)


def _flat(out, tmp_path=None):
    if tmp_path is not None:
        out = _without_tmp_path(out, tmp_path)
    return " ".join(out.split())                   # rich wraps long lines


def _has(out, text):
    """Whether out says text, wherever a terminal or Rich wrapped either (a long temporary
    path, as macOS's /private/var/folders/..., is folded mid-path at the console's width)."""
    return "".join(text.split()) in "".join(out.split())


@pytest.mark.slow
def test_a_first_run_without_sononym_or_live(box):
    b = box
    assert b.n_files == 48
    # before anything: help and doctor read nothing and say what's missing
    b.run("--help")
    out = _flat(b.run("doctor", ok=False), b.tmp)
    assert "FAIL" in out and "fourier setup" in out
    # the CLAP stand-in counts as installed; the model download is skipped (--no-clap)
    out = _flat(b.run("setup", "--yes", "--library", str(b.lib), "--device", "digitakt_2",
                      "--preset", "balanced", "--no-clap"))
    assert (b.user / ".config" / "fourier" / "fourier.toml").exists()
    assert f"Found {b.n_files} audio files" in out and "Ready to build" in out, out
    assert f"analyzes {b.n_files} new samples" in out and "Next: fourier build (it can stop" in out

    # nothing scanned: one line saying what to run, no traceback
    out = _flat(b.run("tools", "analyze", ok=False), b.tmp)
    assert "nothing scanned yet: run `fourier tools scan`" in out
    out = _flat(b.run("build", "--all", "--no-scan", ok=False), b.tmp)
    assert "nothing scanned yet: run `fourier build` without --no-scan" in out
    # what the first build does itself is NEXT, not a failure: a fresh, correct install passes
    out = _flat(b.run("doctor"), b.tmp)
    assert "NEXT samples in the database" in out and "NEXT CLAP index" in out and "FAIL" not in out
    assert "OK build time: about 2 min for the first" in out and "can't estimate" not in out
    # a small library's master scales down to it (scale = "library", the default)
    assert f"OK master size: Your library has {b.n_files} audio files: the master will hold up to about" in out

    out = _flat(b.run("tools", "scan"), b.tmp)
    assert f"{b.n_files} new" in out
    # the walk records what each file's header says, and its path under the library
    rows = b.db("SELECT rel_path, duration_s, sample_rate, channels, file_format, file_hash FROM samples")
    assert len(rows) == b.n_files
    assert all(r[0] and not r[0].startswith("/") and r[1] > 0 and r[2] == SR and r[3] == 1
               and r[4] == "wav" and r[5] for r in rows)
    out = _flat(b.run("build", "--all", "--no-scan", ok=False), b.tmp)
    assert "no CLAP embeddings yet: run `fourier build` without --no-scan" in out
    out = _flat(b.run("build", "--all", "--dry-run"), b.tmp)
    assert "can't estimate before the library is analyzed" in out
    assert b.db("SELECT COUNT(*) FROM sample_features")[0][0] == 0      # a dry run reads only

    # a build scans and analyzes first; one that fails before any category is done leaves no
    # <master>.next behind
    nxt = b.master.with_name(b.master.name + ".next")
    out = _flat(b.run("build", "--all", ok=False, FAIL_CATEGORIES="ALL"), b.tmp)
    assert "1/4 Scan the library" in out and "0 new" in out and "2/4 Analyze what's new" in out
    assert f"{b.n_files} new samples to analyze" in out and "3/4 Build" in out
    assert not nxt.exists() and not b.master.exists(), out
    n = b.db("SELECT COUNT(*), COUNT(clap_embedding), COUNT(mfcc_mean), COUNT(derived_computed_at), "
             "COUNT(events_computed_at) FROM sample_features")[0]
    assert n == (b.n_files,) * 5, n
    assert (b.home / "clap_index.npz").exists()
    # one that fails part way keeps it for --resume, and says so; its analysis is incremental
    out = _flat(b.run("build", "--all", ok=False, FAIL_CATEGORIES="PADS"), b.tmp)
    assert "All samples already have CLAP embeddings" in out and "new samples to analyze" not in out
    # the labels are current (their marker reads back as written): nothing to relabel
    assert "metadata:" not in out
    assert nxt.is_dir() and "`fourier build --resume` continues it, or delete it" in out
    raw = b.run("doctor")
    out = _flat(raw, b.tmp)
    assert "FAIL" not in out and "path and audio" in out
    assert str(b.master) in raw and str(b.lib) in raw, raw   # check lines aren't hard-wrapped
    out = _flat(b.run("build", "--all", "--dry-run"), b.tmp)
    assert f"Your library has {b.n_files} usable samples: the master will hold up to about" in out
    assert "Budgets, scaled to this library" in out and "can't estimate" not in out
    out = _flat(b.run("build", "--all", "--resume"), b.tmp)
    assert "resuming:" in out and not nxt.exists() and "4/4 Verify" in out
    man = json.loads((b.master / "manifest.json").read_text())
    per = {c: len(cd.get("entries") or ()) for c, cd in man["categories"].items()}
    assert all(per.get(c, 0) >= 6 for c in FILLED), per
    assert set(man["providers"]) == {"path", "audio"}, man["providers"]
    assert 0 < man["scale"]["factor"] < 1 and man["scale"]["samples"] == b.n_files
    assert "left empty" in out and "TOMS" in out                 # what the library can't fill

    b.run("verify")
    out = _flat(b.run("why", "--detail", "Kick 01"), b.tmp)
    assert "in the master: KICKS/" in out and "path + audio" in out and "sononym" not in out.lower()

    raw = b.run("render", "digitakt_2")
    out = _flat(raw)
    render = b.renders / "digitakt_2"
    assert render.is_dir() and "audio files" in out
    assert f"-> {render}" in raw, raw                  # a path is never broken mid-path
    out = _flat(b.run("sync", "digitakt_2", str(b.tmp / "card"), ok=False))
    assert "Elektron Transfer" in out and _has(out, str(render)) and "[red]" not in out

    out = _flat(b.run("publish", "--dry-run"))
    assert "No release yet" in out and not (b.user / "Music" / "Fourier").exists()   # wrote nothing
    out = _flat(b.run("publish", "--notes", "first"))
    counts = f"{sum(1 for v in per.values() if v)} categories, {sum(per.values())} audio files"
    assert f"Released v1: {counts}" in out, out
    out = _flat(b.run("publish", "--dry-run"))                  # the master is v1: only adds
    assert "plan: master vs v1" in out and "ADDITIVE" in out
    out = _flat(b.run("releases"))
    assert f"v1 *LATEST {counts}" in out and "None" not in out, out
    out = _flat(b.run("render", "digitakt_2", "--release", "v1"))
    assert (b.locks / "digitakt_2.lock.json").exists(), out
    out = _flat(b.run("render", "digitakt_2", "--dry-run"))
    assert "locked paths" in out and "SAFE: every path on the device keeps its audio" in out


def test_sync_before_any_render_says_to_render(tmp_path):
    b = Sandbox(tmp_path)
    b.run("setup", "--yes", "--library", str(b.lib), "--device", "m8_tracker", "--preset", "balanced",
          "--no-clap")
    out = _flat(b.run("sync", "m8_tracker", str(tmp_path / "card"), ok=False))
    assert "no render yet: run `fourier render m8_tracker`" in out


def test_config_show_is_readable(tmp_path):
    b = Sandbox(tmp_path)
    toml = b.user / ".config" / "fourier" / "fourier.toml"
    toml.parent.mkdir(parents=True)
    toml.write_text(f'library = ["{b.lib}"]\ndevices = ["digitakt_2"]\ntempo = "80-170"\n')
    out = b.run("config", "show")
    assert "$dict" not in out and "$tuple" not in out
    flat = _flat(out)
    assert "preset: balanced (the default:" in flat           # a config naming none gets balanced
    assert "tempo = 80-170 (your fourier.toml)" in flat
    assert "TEMPO_BANDS = [" in flat and "your fourier.toml (tempo)" in flat
    out = b.run("config", "show", "--all")
    assert "$dict" not in out and "$tuple" not in out and "curate_config.BUDGETS = {" in _flat(out)


def test_help_and_doctor_create_no_database(tmp_path):
    b = Sandbox(tmp_path)
    b.run("--help")
    b.run("doctor", "--help")
    b.run("doctor", ok=False)
    assert not b.home.exists() or not any(b.home.glob("library.*")), list(b.home.iterdir())


@pytest.mark.slow
def test_a_small_three_vendor_library_fills_its_drums_and_loops(tmp_path):
    """A small library a newcomer might try first: three vendors, each kind of drum from one
    of them, pack names that say the opposite of some files. The drum categories and the
    loops fill (no per-vendor cap on a pool from one or two vendors), the pack names decide
    nothing, sustained sounds in a breaks pack stay out of the drum loops, the CLAP gate's
    floor fills the drum loops to their minimum, what the library can't fill is named with
    its reasons, and the audio libraries' warnings stay out of the output."""
    b = Sandbox(tmp_path, make=make_small_library)
    assert b.n_files == 48
    out = _flat(b.run("setup", "--yes", "--library", str(b.lib), "--device", "m8_tracker",
                      "--preset", "balanced", "--no-clap"))
    assert "WARN library size: a small library (48 audio files in 3 pack folders)" in out
    # the style's budgets and minimum (6), not scaled to the library: the floors that fill a
    # category to its minimum are what this test is about (the next test scales it)
    toml = b.user / ".config" / "fourier" / "fourier.toml"
    toml.write_text(toml.read_text().replace('preset = "balanced"', 'preset = "balanced"\nscale = "off"', 1))
    assert 'scale = "off"' in toml.read_text()
    assert "for the first build to do" in out and "nothing that would stop a build" in out
    assert "WARN build time" not in out and "can't estimate" not in out
    # a dry run (every category, without --all) writes nothing, not even the resolved config
    import shutil
    shutil.rmtree(b.home / "run", ignore_errors=True)
    out = _flat(b.run("build", "--dry-run"))
    assert "Nothing was written" in out and not (b.home / "run").exists(), list((b.home / "run").iterdir())

    # before any scan, why says so (not "every sample is recognized")
    out = _flat(b.run("why", "--detail", "--unrecognized"))
    assert "No samples in the database yet" in out, out
    out = b.run("build")                               # no CATEGORY: the whole build, as --all
    flat = _flat(out)
    assert "UserWarning" not in out and "warnings.warn" not in out and "n_fft" not in out
    assert f"metadata: updating the path labels ({b.n_files} of {b.n_files} samples new or changed)" in flat
    # a first build: no previous master kept, no changelog, and it says neither
    assert "New master in place at" in flat and "previous kept" not in flat and "CHANGELOG" not in flat
    # a dry run on a built library opens the database read-only: not even its time changes;
    # what it can fill counts no category under its minimum
    db = b.home / "library.duckdb"
    before = db.stat().st_mtime_ns
    out = _flat(b.run("build", "--all", "--dry-run"))
    assert db.stat().st_mtime_ns == before and "This library: up to about 41 files" in out, out
    man = json.loads((b.master / "manifest.json").read_text())
    where = {e["src"].split("SampleLibrary/")[-1]: cat for cat, cd in man["categories"].items()
             for e in cd.get("entries") or ()}
    per = {c: sum(1 for x in where.values() if x == c) for c in set(where.values())}
    assert all(per.get(c, 0) >= 6 for c in ("KICKS", "SNARES", "HATS", "CLAPS", "DRUMLOOPS")), per
    # pack names decide nothing: a "Breaks" pack's kick is a kick, a "Jungle Breaks" riser no
    # loop, a "Drum Hits" pack's grooves are loops (and so are grooves with no silent gap)
    assert where["Acme Audio/Acme Breaks and Hits/Kicks/Kick 01.wav"] == "KICKS"
    assert where["Acme Audio/Acme Breaks and Hits/Snares/Snare 01.wav"] == "SNARES"
    assert "Northwind Loops/Jungle Breaks 174/FX/Riser 01.wav" not in where
    assert where["Vendor C/Drum Hits/Loops/Groove 01 120bpm.wav"] == "DRUMLOOPS"
    # a "Drum Hits" pack's house loops, named only by a tempo, are drum loops, not phrases
    assert where["Vendor C/Drum Hits/Loops/WAV/House 120 Loop 01.wav"] == "DRUMLOOPS"
    assert "PHRASES" not in per and "PHRASES: none found" in flat
    assert where["Northwind Loops/Jungle Breaks 174/Loops/Jungle 01 174bpm.wav"] == "DRUMLOOPS"
    # a breaks pack's riser, pad and bassline (onsets, no silent gap) are what their names
    # say, not drum loops: the basslines fill SUB, the risers and pads are FX and PADS candidates
    nw = "Northwind Loops/Jungle Breaks 174/"
    assert not [p for p, c in where.items() if c == "DRUMLOOPS" and "/Loops/" not in p
                and "/Beats/" not in p], where
    assert where[nw + "Bass/Bass 05.wav"] == "SUB" and where[nw + "Bass/Bass 06.wav"] == "SUB"
    # five loops pass every gate; the house loops pass the CLAP gate but are too harmonic, the
    # four-on-the-floor beats fail it: the floor readmits the best-scoring house loop to reach
    # DRUMLOOPS' minimum of 6, and no more
    beats = sorted(p for p, c in where.items() if c == "DRUMLOOPS" and "/Beats/" in p)
    house = sorted(p for p, c in where.items() if c == "DRUMLOOPS" and "/House " in p)
    assert per["DRUMLOOPS"] == 6 and not beats and len(house) == 1, where
    assert "3 loops gated out, 2 too harmonic" in flat, flat
    assert "1 too-harmonic loops readmitted to reach the minimum of 6" in flat
    assert "gated loops readmitted" not in flat
    # what's left empty says why, once, with its counts
    assert "FX: 5 found; 0 near-duplicates, 0 over the per-vendor share; need 6: left empty" in flat
    assert "PADS: 2 found" in flat
    assert "FX: FX:" not in flat and "TOMS: none found; need 6" in flat
    out = _flat(b.run("why", "--detail", "Riser 01"))
    assert "left out of FX: FX was left empty: 5 found" in out and "path: fx.sweep" in out
    out = _flat(b.run("why", "--detail", "Riser 04"))
    assert "votes: path + audio: FX" in out and "labels: path: fx.sweep; audio: class.oneshot" in out
    assert "DRUMLOOPS" not in out
    out = _flat(b.run("why", "--detail", "Pad 01"))
    assert "votes: path + audio: PADS" in out and "left out of PADS: PADS was left empty: 2 found" in out
    # why shows each loop's gates: passed, readmitted past the harmonicity gate (its value and
    # the threshold), failed one (with its score or value)
    out = _flat(b.run("why", "--detail", house[0].rsplit("/", 1)[-1]))
    assert "decided by: votes -> DRUMLOOPS" in out and "phrase" not in out
    assert "DRUMLOOPS: passed its CLAP gate (score" in out
    assert "DRUMLOOPS: readmitted past DRUMLOOPS's harmonicity gate (harmonicity 0.8" in out
    assert "> 0.68: too tonal) to reach the minimum of 6: a loop by its own name" in out
    other = next(f"House 120 Loop {i:02d}" for i in (1, 2, 3) if f"Loop {i:02d}" not in house[0])
    out = _flat(b.run("why", "--detail", other))
    assert "left out of DRUMLOOPS: passed its CLAP gate (score" in out
    assert "left out because failed DRUMLOOPS's harmonicity gate (harmonicity 0.8" in out
    out = _flat(b.run("why", "--detail", "Four Floor 01"))
    assert "left out of DRUMLOOPS: failed its CLAP gate (score 0." in out and "< 0.30)" in out
    out = _flat(b.run("why", "--detail", "Groove 01"))
    assert "DRUMLOOPS: passed its CLAP gate (score" in out
    out = _flat(b.run("why", "--detail", "Kick 01"))
    assert "in the master: KICKS/" in out and "labels: path: kick; audio: class.oneshot" in out
    # search tells same-named files apart by folder, and labels them without Sononym
    out = b.run("search", "--min-dur", "0.3", "--max-dur", "0.36", "--top", "3")
    assert "Kick 01.wav" in out and "Acme Breaks and Hits/Kicks" in out and " kick " in out, out
    # a tempo only where it means something: loops, not one-shots
    out = b.run("search", "--min-dur", "0.24", "--max-dur", "0.26", "--top", "12")
    assert "Snare 0" in out and " 117 " not in out, out
    out = b.run("search", "--min-dur", "5.5", "--max-dur", "5.6", "--top", "2")
    assert "Jungle 0" in out and "—" not in out, out             # each loop shows its tempo
    # the tempo a build resolves (the name's, whole bars), loops only: the 120 BPM grooves the
    # analysis reads at 60 show and filter at 120, and the basslines never match
    out = b.run("search", "--bpm-min", "115", "--bpm-max", "125", "--top", "20")
    assert "Groove 01" in out and "House 120" in out and "Four Floor" in out, out
    assert "Bass 0" not in out and "Jungle" not in out and " 60 " not in out
    assert "8 candidates found" in _flat(out)
    out = b.run("search", "--bpm-min", "55", "--bpm-max", "65", "--top", "20")
    assert "No samples found" in out, out
    out = b.run("search", "--bpm-min", "85", "--bpm-max", "90", "--bpm-octave", "--top", "20")
    assert "Jungle 01" in out and "Groove" not in out, out
    # a second build keeps the first as <master>.prev and writes what changed
    out = _flat(b.run("build", "--all", "--no-scan"))
    assert "previous kept at" in out and "CHANGELOG.md says what changed" in out
    # a one-category build of a category the library can't fill: one clean line, no traceback
    out = _flat(b.run("build", "FX", "--out", str(tmp_path / "fx"), ok=False))
    assert "FX: 5 found" in out and "Nothing built" in out
    assert "Errno" not in out and "manifest.json" not in out and "FX: FX:" not in out


@pytest.mark.slow
def test_a_small_library_gets_a_master_its_size(tmp_path):
    """The same small library with the master scaled to it (scale = "library", the default):
    every category with a few usable samples builds, each keeps what it has (no floor needs to
    readmit a gated loop), folders are few and small, verify holds the build to its recorded
    budgets, and doctor, setup, the dry run and the build say how big the master will be."""
    b = Sandbox(tmp_path, make=make_small_library)
    out = _flat(b.run("setup", "--yes", "--library", str(b.lib), "--device", "m8_tracker",
                      "--preset", "balanced", "--no-clap"))
    assert "the master scales down to it (scale = \"library\"), and a category builds from 2 usable" in out
    assert "OK master size: Your library has 48 audio files: the master will hold up to about" in out
    assert "scale: library (the master follows the library's size" in _flat(b.run("config", "show"))
    out = _flat(b.run("build", "--all"))
    assert "Your library has 48 usable samples: the master will hold up to about" in out
    assert "KICKS: budget 6 (the style's" in out and "minimum 2, 1 folder" in out
    assert "readmitted" not in out and "PHRASES: none found; need 2" in out
    assert "4/4 Verify" in out and "New master in place at" in out
    man = json.loads((b.master / "manifest.json").read_text())
    assert 0 < man["scale"]["factor"] < 0.01 and man["scale"]["min_files"] == 2
    where = {e["src"].split("SampleLibrary/")[-1]: cat for cat, cd in man["categories"].items()
             for e in cd.get("entries") or ()}
    per = {c: sum(1 for x in where.values() if x == c) for c in set(where.values())}
    # what the style's minimum of 6 left empty builds now: FX's five risers, PADS' two pads
    assert per["FX"] == 5 and per["PADS"] == 2, per
    assert set(per) == {"KICKS", "SNARES", "HATS", "CLAPS", "SUB", "FX", "PADS", "DRUMLOOPS"}, per
    # the loops that pass every gate, and no gated one: the minimum is met without the floor
    loops = sorted(p for p, c in where.items() if c == "DRUMLOOPS")
    assert len(loops) == 5 and not [p for p in loops if "/House " in p or "/Beats/" in p], loops
    for c, cd in man["categories"].items():
        fams = {e["family"] for e in cd["entries"]}
        assert len(fams) == 1 and len(cd["entries"]) <= cd["budget"] <= 24, (c, fams, cd["budget"])
    assert sum(per.values()) == len(where) < b.n_files
    out = _flat(b.run("why", "--detail", "Riser 01"))
    assert "in the master: FX/" in out
    out = _flat(b.run("build", "--all", "--dry-run"))
    assert "Your library has 48 usable samples: the master will hold up to about" in out and "in up to 8 categories" in out
    b.run("verify")
    out = _flat(b.run("doctor"))
    assert "OK master size: Your library has 48 usable samples: the master will hold up to about" in out
