"""`fourier demo`: the whole pipeline on a synthetic library, in a sandbox folder.

    fourier demo [--dir PATH]          # default: ./fourier-demo

It writes the synthetic library (fourier.synthlib), fills a database with stand-in analysis,
builds the master and renders it for the Digitakt 2 and the M8, all inside PATH:

    PATH/
      .fourier-demo     the marker: a later `fourier demo --dir PATH` may replace what's here
      fourier.toml      the demo's own config (library, devices, preset, [output], budgets)
      demo.env          `. PATH/demo.env` points fourier at the demo in this shell
      demo.log          everything the build and renders printed
      library/          the synthetic library (280 generated WAVs)
      home/             the demo's Fourier home: database, CLAP index, caches, build archive
      master/           the curated master
      renders/<device>  one render per device
      publish/          releases and device path locks, if you publish from the demo

Nothing outside PATH is read or written: the demo sets its own home, config, library and
output folders for the run, so ~/.fourier, ~/Music and the user's fourier.toml stay as they
were. It needs no model and no download: the analysis the build reads is stand-in data, the
same the CI golden build uses (tests/golden/synthetic_build.py), and CLAP is a deterministic
stand-in that points a file and the words that describe it the same way.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from contextlib import contextmanager
from pathlib import Path

MARKER = ".fourier-demo"
DEVICES = ("digitakt_2", "m8_tracker")
DEFAULT_DIR = "fourier-demo"
PRESET = "breaks-acid"
SCALE = "off"          # the budgets below as they are, not scaled to the small library

# budgets sized to the synthetic library (12 to 24 files per category); the same as
# tests/golden/synthetic.toml
BUDGETS = {c: 24 for c in ("DRUMLOOPS", "PHRASES", "KICKS", "SNARES", "HATS", "PERC", "SUB", "TOMS",
                           "CLAPS", "CYMBALS", "FX", "SYNTH", "PADS", "STABS", "BLIPS", "VOX",
                           "PIANO", "WAVES", "ACOUSTIC")}

# what a demo folder holds: a rerun replaces these and leaves anything else alone
CONTENTS = (MARKER, "fourier.toml", "demo.env", "demo.log", "library", "home", "master",
            "master.next", "master.prev", "renders", "publish")

# the variables that keep their value in the demo: switches, not places
KEEP_ENV = ("FOURIER_NO_AUDIO_CACHE", "FOURIER_NO_RENDER_CACHE", "FOURIER_NO_PITCH_CHECK")


class DemoError(RuntimeError):
    pass


# --- the stand-in analysis (shared with tests/golden/synthetic_build.py) -------------------

DIM = 512


def unit(*key):
    """A deterministic unit vector for key (a stand-in CLAP embedding)."""
    import numpy as np
    seed = int.from_bytes(hashlib.sha256(repr(key).encode()).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(DIM).astype("float32")
    return v / np.linalg.norm(v)


# Concepts: a stand-in CLAP space where a file and the words that describe it point the same
# way, so the CLAP gates and names behave (a break reads as a break, a pad as a pad).
CONCEPTS = [
    ("kick", r"\bkick|bass drum|\bbd\b"), ("snare", r"snare|rimshot"), ("clap", r"\bclap"),
    ("openhat", r"open|sizzl|washy"), ("hat", r"hat|shaker|hi-?hat"), ("tom", r"\btom"), ("perc", r"conga|bongo|wood|percussion|perc\b"),
    ("cymbal", r"cymbal|crash|\bride"), ("bass", r"\bbass|\bsub\b|808"), ("lead", r"lead|synth"),
    ("pad", r"\bpad|texture|ambient"), ("stab", r"stab|chord|orchestra hit"),
    ("blip", r"blip|zap|beep"), ("vox", r"vocal|voice|vowel|choir|sing"),
    ("riser", r"riser|sweep|uplift|whoosh|build"), ("chirp", r"chirp|song|nature"),
    ("break", r"break|drum loop|beat\b|groove"), ("piano", r"piano|keys|rhodes|keyboard"),
    ("strings", r"violin|string|cello|viola|orchestral"), ("wave", r"wave|cycle|oscillator"),
]
KIND_CONCEPT = {"Kicks": "kick", "Snares": "snare", "Claps": "clap", "Hats": "hat", "Toms": "tom",
                "Percussion": "perc", "Cymbals": "cymbal", "Bass": "bass", "Leads": "lead",
                "Pads": "pad", "Stabs": "stab", "Blips": "blip", "Vocals": "vox", "Risers": "riser",
                "Drum Loops": "break", "Bass Loops": "bass", "Grand Piano": "piano",
                "Violin": "strings", "Wavetables": "wave"}


def audio_embedding(f):
    """Files of one kind sit near each other and near the words for that kind; each file
    is still distinct (no near-dups)."""
    import numpy as np
    kind = f.rel.split("/")[-2]
    v = unit("concept", KIND_CONCEPT[kind]) + 0.35 * unit("file", f.rel)
    if "Open Hat" in f.rel:
        v = v + unit("concept", "openhat")
    if "Loops" in f.rel:
        v = v + 0.5 * unit("concept", "loop")
    return (v / np.linalg.norm(v)).astype("float32")


def fake_embed_text(text):
    """The stand-in CLAP text encoder: the concepts text names, plus a little of its own."""
    import re

    import numpy as np
    v = 0.35 * unit("text", text)
    for name, pat in CONCEPTS:
        if re.search(pat, text, re.IGNORECASE):
            v = v + unit("concept", name)
    if re.search(r"\bloop", text, re.IGNORECASE):
        v = v + 0.5 * unit("concept", "loop")
    return (v / np.linalg.norm(v)).astype("float32")


def fill_db(db_path, lib_root, files, without=()) -> int:
    """A library database for files (fourier.synthlib.generate's) under lib_root: a Sample, a
    Sononym-style row and a stand-in CLAP embedding per file, and the CLAP index built from
    them. `without` leaves out providers' data ("sononym", "ableton"). Returns the file count."""
    from .analysis.clap_features import build_index
    from .db import session as S
    S.init_db(db_path)
    with S.session_scope() as s:
        add_rows(s, lib_root, files, without)
    with S.session_scope() as s:
        build_index(s)
    return len(files)


def add_rows(s, lib_root, files, without=()) -> None:
    """fill_db's rows, added to session s (its caller commits)."""
    import soundfile as sf

    from .analysis.clap_features import embedding_to_bytes
    from .db.models import Sample, SampleFeatures, SononymMeta
    lib_root = Path(lib_root)
    for i, f in enumerate(files, 1):
        p = lib_root / f.rel
        info = sf.info(p)
        h = hashlib.sha256(p.read_bytes()[:8192]).hexdigest()
        smp = Sample(id=i, path=str(p), rel_path=f.rel, filename=p.name,
                     file_size_bytes=p.stat().st_size, file_hash=h, modified_at=0,
                     duration_s=round(info.duration, 6), sample_rate=info.samplerate,
                     channels=info.channels, bit_depth=16, file_format="wav",
                     ableton_tags=(list(f.ableton_tags) or None) if "ableton" not in without else None)
        s.add(smp)
        if "sononym" not in without:
            s.add(SononymMeta(sample_id=i, classes=list(f.classes), class_strengths=[0.9],
                              categories=list(f.categories), category_strengths=[0.8],
                              base_note=f.note, base_note_confidence=0.8 if f.note is not None else None,
                              pitch_class=None, peak_db=-1.0, rms_db=-14.0, crest_factor=10.0,
                              bpm=f.bpm, bpm_confidence=0.9 if f.bpm else None,
                              brightness=f.brightness, noisiness=f.noisiness,
                              harmonicity=f.harmonicity))
        s.add(SampleFeatures(sample_id=i, clap_embedding=embedding_to_bytes(audio_embedding(f)),
                             clap_model="stand-in", tempo_bpm=f.bpm,
                             harmonic_percussive_ratio=f.harmonicity,
                             spectral_centroid_mean=1000 + 4000 * f.brightness,
                             spectral_flatness_mean=f.noisiness,
                             chroma_concentration=round(1.0 + 1.5 * f.harmonicity, 3),
                             onset_rate_hz=(f.bpm / 30.0) if f.bpm else 1.0,
                             # the event count enrich measures (the audio provider reads it);
                             # only without Sononym, so the golden's DB stays as it was
                             n_events=(None if "sononym" not in without else
                                       max(4, round(info.duration * (f.bpm or 120) / 60))
                                       if "Loop" in f.classes else 1)))


@contextmanager
def standin_clap():
    """The stand-in CLAP text encoder in place of the model, for as long as this lasts."""
    from .analysis import clap_features as CF
    real = CF.embed_text
    CF.embed_text = fake_embed_text
    try:
        yield
    finally:
        CF.embed_text = real


# --- the sandbox -----------------------------------------------------------------------------

def active() -> Path | None:
    """The demo folder this process is pointed at (its config or its Fourier home is one of
    a demo's, `. demo.env`), or None."""
    import os
    for env in ("FOURIER_CONFIG", "FOURIER_HOME"):
        v = os.environ.get(env)
        if v and v.lower() != "none":
            p = Path(v).expanduser()
            folder = p.parent
            if (folder / MARKER).is_file():
                return folder
    return None


REAL_ONLY = ("The demo's analysis and CLAP are stand-ins (no model, no download), so {what} "
             "needs your own library: `fourier setup` in a new Terminal window.")


def prepare(root) -> Path:
    """root, ready for a demo: a new or empty folder, or an earlier demo's (its contents are
    replaced). Anything else is refused: the demo never writes into a folder of the user's."""
    root = Path(root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise DemoError(f"{root} is a file: give the demo a folder (--dir)")
    if root.is_dir() and any(root.iterdir()):
        if not (root / MARKER).is_file():
            raise DemoError(f"{root} isn't empty and isn't an earlier demo: give the demo a new or "
                            f"empty folder (--dir)")
        for name in CONTENTS:
            p = root / name
            if p.is_symlink() or p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("A Fourier Samples demo (fourier demo). `fourier demo --dir` may "
                               "replace this folder's contents.\n")
    return root


def layout(root: Path) -> dict[str, Path]:
    return {"library": root / "library", "home": root / "home", "master": root / "master",
            "renders": root / "renders", "publish": root / "publish",
            "config": root / "fourier.toml", "env": root / "demo.env", "log": root / "demo.log",
            "db": root / "home" / "library.db"}


def config_text(where: dict[str, Path]) -> str:
    q = lambda p: json.dumps(str(p))          # a TOML basic string
    budgets = ", ".join(f"{c} = {n}" for c, n in BUDGETS.items())
    return ("# The config `fourier demo` wrote for this demo; your own is elsewhere (fourier setup).\n"
            f"library = [{q(where['library'])}]\n"
            f"devices = [{', '.join(q(d) for d in DEVICES)}]\n"
            f"preset = {q(PRESET)}        # the style the CI golden build uses\n"
            f"scale = {q(SCALE)}           # the budgets below, not scaled to the library\n\n"
            "[output]\n"
            f"master = {q(where['master'])}\n"
            f"renders = {q(where['renders'])}\n"
            f"publish = {q(where['publish'])}\n\n"
            "# budgets sized to the synthetic library\n"
            "[advanced]\n"
            f"BUDGETS = {{{budgets}}}\n")


def env_text(where: dict[str, Path]) -> str:
    return ("# Point fourier at this demo in the current shell:  . " + shlex.quote(str(where["env"]))
            + "\n# (a new shell is back to your own home and config)\n"
            f"export FOURIER_HOME={shlex.quote(str(where['home']))}\n"
            f"export FOURIER_CONFIG={shlex.quote(str(where['config']))}\n")


@contextmanager
def sandbox_env(where: dict[str, Path]):
    """The demo's home and config in the environment (its config names its library and
    output folders) and every other Fourier variable unset but the switches in KEEP_ENV, for
    as long as this lasts; then the environment as it was."""
    saved = dict(os.environ)
    try:
        for k in [k for k in os.environ if k.startswith("FOURIER_") and k not in KEEP_ENV]:
            del os.environ[k]
        # what demo.env sets: the rest (library, master, renders, publish) is in the config
        os.environ.update(FOURIER_HOME=str(where["home"]), FOURIER_CONFIG=str(where["config"]))
        from . import places
        places.reset()
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)
        from . import places
        places.reset()


# --- the run -----------------------------------------------------------------------------------

def _fourier(args: list[str], log) -> None:
    """Run a fourier command in this process, its output into log; raise DemoError if it fails."""
    from .cli._app import console
    from .cli import main
    code = 0
    with console.capture() as cap:
        try:
            main.main(args, prog_name="fourier", standalone_mode=False)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except Exception as e:  # noqa: BLE001 (any failure: reported with the log)
            console.print(f"{type(e).__name__}: {e}", markup=False)
            code = 1
    out = cap.get()
    log.write(f"$ fourier {' '.join(args)}\n{out}\n")
    log.flush()
    if code:
        tail = "\n".join(out.rstrip().splitlines()[-15:])
        raise DemoError(f"`fourier {' '.join(args)}` failed:\n{tail}")


def _count_wavs(d: Path) -> int:
    return sum(1 for p in d.rglob("*.wav") if not p.name.startswith("."))


def run(root=DEFAULT_DIR, say=print) -> dict:
    """The demo in root. say(line) reports progress. Returns what the summary reports:
    {"root", "master", "renders": {device: (path, files)}, "categories": {cat: files},
     "example", "files"}."""
    from . import synthlib
    root = prepare(root)
    where = layout(root)
    for k in ("library", "home", "renders", "publish"):
        where[k].mkdir(parents=True, exist_ok=True)
    where["config"].write_text(config_text(where))
    where["env"].write_text(env_text(where))
    db = str(where["db"])
    with sandbox_env(where), standin_clap(), open(where["log"], "w") as log:
        say("1/4  Writing the synthetic library (generated tones, drums and loops)...")
        files = synthlib.generate(where["library"])
        say(f"     {len(files)} files in {where['library']}")
        say("2/4  Filling the demo database with stand-in analysis (no model, no download)...")
        fill_db(where["db"], where["library"], files)
        say("3/4  Building the master: routing, families, names, the export DSP, kits, slices,")
        say("     then verify (about a minute)...")
        _fourier(["--db", db, "build", "--all", "--no-scan", "--no-describe"], log)
        say(f"4/4  Rendering for {' and '.join(DEVICES)}...")
        for d in DEVICES:
            _fourier(["--db", db, "render", d], log)
        from .packs.curate_config import CATEGORY_ORDER     # the play order, as the build had it
        order = list(CATEGORY_ORDER)
    from .packs import manifests
    manifest = manifests.read(where["master"])
    built = manifest.get("categories") or {}
    cats = {c: len(built[c].get("entries") or ()) for c in sorted(
        built, key=lambda c: order.index(c) if c in order else len(order))}
    kick = next((e for e in (manifest["categories"].get("KICKS") or {}).get("entries") or ()), None)
    example = Path(kick["src"]).stem if kick else "Kick 01"
    return {"root": root, "master": where["master"], "env": where["env"], "log": where["log"],
            "renders": {d: (where["renders"] / d, _count_wavs(where["renders"] / d)) for d in DEVICES},
            "categories": cats, "example": example, "files": len(files),
            "sets": {s: _count_wavs(where["master"] / s) for s in ("KITS", "SLICE")
                     if (where["master"] / s).is_dir()}}


def summary(r: dict) -> list[str]:
    """The lines the demo ends with: where things are, what was built, what to try next."""
    cats = r["categories"]
    width = max(map(len, cats), default=0)
    rows = [f"  {c:<{width}}  {n:>3}" for c, n in cats.items()]
    half = (len(rows) + 1) // 2
    cols = [f"{a}      {b}".rstrip() for a, b in zip(rows[:half], rows[half:] + [""])]
    env = shlex.quote(str(r["env"]))
    lines = [
        "",
        f"Done. The demo is in {r['root']}",
        "",
        f"Master (built, verify passed): {r['master']}",
        f"  {sum(cats.values())} files from the {r['files']} in the library:",
        *cols,
    ]
    if r.get("sets"):
        lines.append("  plus " + " and ".join(f"{s} ({n} files)" for s, n in r["sets"].items())
                     + ", drawn from the categories")
    lines += ["", "Renders:"]
    lines += [f"  {d}: {p}  ({n} files)" for d, (p, n) in r["renders"].items()]
    lines += [
        "",
        "The analysis is stand-in data and CLAP is a stand-in (no model, no download), so",
        "the families and names are a sketch of what a real library gets.",
        "",
        "Try next. The first line points fourier at the demo in this shell:",
        f"  . {env}",
        f"  {'fourier why ' + json.dumps(r['example']):<28}  # where a sample landed, and the rule that put it there",
        f"  {'fourier open':<28}  # the demo's master in your file browser",
        f"  {'fourier review rate ' + json.dumps(r['example']) + ' drop':<28}  # rate a file, then",
        f"  {'fourier build':<28}  # rebuild: the rating takes effect",
        "",
        "On your own library, in a new Terminal window:",
        *(f"  {cmd:<38}  # {what}" for cmd, what in (
            ("fourier setup", "your samples, devices and style; the CLAP model"),
            ("fourier build", "scan, analyze, build and verify (resumable)"),
            ("fourier publish", "save it as release v1, which never changes"),
            (f"fourier render {DEVICES[0]} --release v1", "the folders to load with Elektron Transfer"),
            (f"fourier sync {DEVICES[1]} <card>", "or onto the M8's SD card (after its render)"))),
    ]
    return lines
