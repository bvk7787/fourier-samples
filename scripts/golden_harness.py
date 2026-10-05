"""The sealed golden harness: rebuild the master from frozen inputs and compare it to a
reference build (normally a release), without touching the real Fourier home or master.

    python scripts/golden_harness.py --reference <manifest.json | master dir>
        [--work DIR]            fresh folder for this run (default ~/.fourier-golden/<stamp>)
        [--home DIR]            the Fourier home to freeze (default $FOURIER_HOME or ~/.fourier)
        [--lock-dir DIR]        device path locks to freeze (default: <publish root>/devices)
        [--audio-cache clone|cold]   clone the processed-audio cache (default) or build cold
        [--no-sticky]           build fresh picks (FOURIER_NO_STICKY=1) instead of sticking
                                to the reference
        [--config TOML]         build with these config layers (e.g. a fourier.toml with
                                your overlay); default: none, the code's defaults
        [--jobs N]              category workers (default 8)
        [--setup-only]          freeze the inputs, print the command, don't build

What it does, in order:
1. Refuses a work folder that isn't empty, so every run starts fresh.
2. Checks that the home's ratings store hashes to the reference's ratings_hash: a build
   with other ratings can't reproduce it.
3. Freezes the inputs into <work>/home: the library DB, the CLAP index and its fast copies,
   ratings.json, the clap_text, pitch and (optionally) audio caches. On macOS these are
   APFS clones (cp -c): instant, and they take no space until something changes. The
   reference manifest is the only archived build, so a sticky build sticks to it. The
   device locks are copied to <work>/locks.
4. Builds with FOURIER_HOME=<work>/home, FOURIER_CURATED_DIR=<work>/out/FourierCurated,
   FOURIER_LOCK_DIR=<work>/locks, FOURIER_CONFIG=none (no config layers) or the --config
   file, every narrower override and any resolved config unset, and --no-describe.
   Nothing is read from or written to the real home, master, releases or locks.
5. Compares the new manifest to the reference (fourier.packs.golden) and checks that the
   build saw the reference's ratings and used exactly the overrides --config resolves to
   (none without it). Writes <work>/golden_report.json.
6. Checks the real home's ratings, archived builds and master manifest are unchanged.

Exit codes: 0 identical, 1 different, 2 bad arguments or work folder, 3 ratings don't
match the reference, 4 the build failed, 5 the real state changed during the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fourier.packs.golden import compare_manifests, load
from fourier.paths import fourier_home

# overrides that would pull a piece of state out of the frozen home
NARROW_OVERRIDES = ("FOURIER_RATINGS", "FOURIER_AUDIO_CACHE", "FOURIER_NO_AUDIO_CACHE",
                    "FOURIER_RENDER_CACHE", "FOURIER_CLAP_TEXT_CACHE", "FOURIER_CLAP_INDEX",
                    "FOURIER_PITCH_CACHE", "FOURIER_DESCRIBE_CACHE", "FOURIER_STAGING_DIR",
                    "FOURIER_NO_STICKY", "FOURIER_RESOLVED_CONFIG",
                    "FOURIER_CONFIG")
CLAP_FILES = ("clap_index.npz", "clap_index.ids.npy", "clap_index.emb.npy")


def sha16(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def clone(src: Path, dst: Path) -> None:
    """Copy a file or folder, as an APFS clone on macOS (instant, no space used)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        subprocess.run(["cp", "-cpR", str(src), str(dst)], check=True)
    elif src.is_dir():
        shutil.copytree(src, dst, copy_function=shutil.copy2)
    else:
        shutil.copy2(src, dst)


def db_file(home: Path) -> Path:
    """The DB fourier would open in this home (same rule as db.session.get_db_path)."""
    duck, lite = home / "library.duckdb", home / "library.db"
    return lite if lite.exists() and not duck.exists() else duck


def real_state(home: Path) -> dict:
    """What must not change while the harness runs."""
    builds = sorted(p.name for p in (home / "builds").glob("*.json")) if (home / "builds").exists() else []
    from fourier.packs.ratings import live_master_dir
    master = os.environ.get("FOURIER_CURATED_DIR") or live_master_dir()
    mf = Path(master) / "manifest.json" if master else None
    return {"ratings": sha16(home / "ratings.json"), "builds": builds,
            "master_manifest": sha16(mf) if mf and mf.exists() else None}


def freeze(home: Path, work: Path, ref: dict, ref_path: Path, lock_dir: Path | None,
           audio_cache: str, log=print) -> Path:
    """Clone the build's inputs into <work>/home and return it."""
    fh = work / "home"
    fh.mkdir(parents=True)
    db = db_file(home)
    if not db.exists():
        raise FileNotFoundError(f"no library DB in {home}")
    # a SQLite DB in WAL mode keeps recent writes in its -wal file until a checkpoint
    wal = [db.with_name(db.name + s) for s in ("-wal", "-shm") if db.with_name(db.name + s).exists()]
    parts = [db, *wal, home / "ratings.json"] + [home / f for f in CLAP_FILES if (home / f).exists()]
    parts += [home / "cache" / c for c in ("clap_text", "pitch")
              + (("audio",) if audio_cache == "clone" else ()) if (home / "cache" / c).exists()]
    for p in parts:
        t0 = time.time()
        clone(p, fh / p.relative_to(home))
        log(f"froze {p.relative_to(home)} ({time.time() - t0:.1f}s)")
    stamp = (ref.get("generated") or "reference").replace(":", "").replace("-", "")
    (fh / "builds").mkdir()
    shutil.copy2(ref_path, fh / "builds" / f"{stamp}.json")
    (work / "locks").mkdir()
    if lock_dir and lock_dir.exists():
        for lk in lock_dir.glob("*.lock.json"):
            shutil.copy2(lk, work / "locks" / lk.name)
    (work / "out").mkdir()
    return fh


def build_env(work: Path, no_sticky: bool, audio_cache: str, config: Path | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in NARROW_OVERRIDES}
    env.update(FOURIER_HOME=str(work / "home"),
               FOURIER_CURATED_DIR=str(work / "out" / "FourierCurated"),
               FOURIER_LOCK_DIR=str(work / "locks"),
               # the given config layers, or none: no fourier.toml, preset or overlay
               FOURIER_CONFIG=str(config) if config else "none")
    if no_sticky:
        env["FOURIER_NO_STICKY"] = "1"
    if audio_cache == "cold":
        env["FOURIER_NO_AUDIO_CACHE"] = "1"
    return env


def expected_overrides(config: Path | None) -> dict:
    """The overrides a build with this config must record, encoded as manifests do."""
    if config is None:
        return {}
    from fourier import layers, settings
    r = layers.resolve(config=str(config))
    return {k: settings.encode(v) for k, v in sorted(r.values.items())}


def build_cmd(work: Path, jobs: int) -> list[str]:
    exe = Path(sys.executable).with_name("fourier")
    exe = str(exe) if exe.exists() else (shutil.which("fourier") or "fourier")
    # --no-scan: the frozen database as it is, never scanned or analyzed
    return [exe, "--db", str(db_file(work / "home")), "build", "--all", "--no-scan",
            "-j", str(jobs), "--out", str(work / "out" / "FourierCurated"), "--no-describe"]


def default_lock_dir(config: Path | None) -> Path:
    """The lock folder the build's config names (fourier/places.py), read as the build will."""
    from fourier import places
    saved = {k: os.environ.pop(k, None) for k in ("FOURIER_CONFIG", "FOURIER_LOCK_DIR")}
    os.environ["FOURIER_CONFIG"] = str(config) if config else "none"
    try:
        return places.lock_dir()
    finally:
        os.environ.pop("FOURIER_CONFIG", None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--work")
    ap.add_argument("--home")
    ap.add_argument("--lock-dir", help="the real device locks to freeze (default: the config's, "
                                       "<[output] publish>/devices)")
    ap.add_argument("--audio-cache", choices=("clone", "cold"), default="clone")
    ap.add_argument("--no-sticky", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--setup-only", action="store_true")
    ap.add_argument("--build-cmd", help=argparse.SUPPRESS)   # tests: a stand-in build
    a = ap.parse_args(argv)

    home = Path(a.home).expanduser() if a.home else fourier_home()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    work = Path(a.work).expanduser() if a.work else Path("~/.fourier-golden").expanduser() / stamp
    if work.exists() and any(work.iterdir()):
        print(f"golden harness: {work} isn't empty; every run needs a fresh folder", file=sys.stderr)
        return 2
    try:
        ref_path = Path(a.reference).expanduser()
        ref_path = ref_path / "manifest.json" if ref_path.is_dir() else ref_path
        ref = load(ref_path)
    except (OSError, ValueError) as e:
        print(f"golden harness: can't read the reference: {e}", file=sys.stderr)
        return 2

    config = Path(a.config).expanduser().resolve() if a.config else None
    try:
        expected = expected_overrides(config)
    except Exception as e:   # ConfigError, a missing file
        print(f"golden harness: --config {a.config}: {e}", file=sys.stderr)
        return 2

    have = sha16(home / "ratings.json")
    if have != ref.get("ratings_hash"):
        print(f"golden harness: {home / 'ratings.json'} hashes to {have}, the reference was built "
              f"with {ref.get('ratings_hash')}; its ratings can't be reproduced from this home",
              file=sys.stderr)
        return 3

    before = real_state(home)
    work.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(a.lock_dir).expanduser() if a.lock_dir else default_lock_dir(config)
    freeze(home, work, ref, ref_path, lock_dir, a.audio_cache, log=lambda m: print(m, file=sys.stderr))
    env = build_env(work, a.no_sticky, a.audio_cache, config)
    cmd = shlex.split(a.build_cmd) if a.build_cmd else build_cmd(work, a.jobs)
    run = {"reference": str(ref_path), "home": str(home), "work": str(work), "cmd": cmd,
           "no_sticky": a.no_sticky, "audio_cache": a.audio_cache,
           "config": str(config) if config else None, "expected_overrides": sorted(expected),
           "env": {k: env[k] for k in sorted(env) if k.startswith("FOURIER_")}}
    (work / "run.json").write_text(json.dumps(run, indent=2))
    if a.setup_only:
        print(json.dumps(run, indent=2))
        return 0

    t0 = time.time()
    with open(work / "build.log", "w") as logf:
        rc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False).returncode
    secs = round(time.time() - t0)
    out = work / "out" / "FourierCurated"
    cand = out / "manifest.json"
    if not cand.exists() and (out.parent / "FourierCurated.next" / "manifest.json").exists():
        cand = out.parent / "FourierCurated.next" / "manifest.json"
    after = real_state(home)
    if after != before:
        print(f"golden harness: the real state changed during the run: {before} -> {after}",
              file=sys.stderr)
        return 5
    if not cand.exists():
        print(f"golden harness: the build wrote no manifest (exit {rc}, {secs}s); see {work / 'build.log'}",
              file=sys.stderr)
        return 4
    new = load(cand)
    rep = compare_manifests(ref, new)
    ratings_ok = new.get("ratings_hash") == ref.get("ratings_hash")
    overrides = new.get("overrides") or {}          # must be exactly what --config resolves to
    overrides_ok = overrides == expected
    report = dict(rep.to_json(), build_exit=rc, build_seconds=secs, candidate=str(cand),
                  ratings_hash_ok=ratings_ok, overrides=overrides, overrides_ok=overrides_ok,
                  tunables_hash=new.get("tunables_hash"), **run)
    (work / "golden_report.json").write_text(json.dumps(report, indent=2, default=list))
    print("\n".join(rep.lines()))
    print(f"build: exit {rc}, {secs}s; ratings hash {'matches' if ratings_ok else 'DIFFERS'}; "
          f"report: {work / 'golden_report.json'}")
    if rc != 0:
        return 4
    if not overrides_ok:
        extra = sorted(set(overrides) - set(expected))
        missing = sorted(set(expected) - set(overrides))
        changed = sorted(k for k in set(overrides) & set(expected) if overrides[k] != expected[k])
        print(f"golden harness: the build's overrides aren't the config's: extra {extra}, "
              f"missing {missing}, different {changed}", file=sys.stderr)
    return 0 if rep.ok and ratings_ok and overrides_ok else 1


if __name__ == "__main__":
    sys.exit(main())
