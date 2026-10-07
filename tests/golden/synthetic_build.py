"""Build the synthetic library end to end: the CI golden (tests/test_synthetic_golden.py).

    python tests/golden/synthetic_build.py <work dir> [--jobs 1]

1. Writes the synthetic library (fourier.synthlib) under <work>/lib/SampleLibrary.
2. Fills a fresh SQLite library DB in <work>/home with a Sample, a Sononym-style row and a
   stand-in CLAP embedding per file, and builds the CLAP index from it. With
   --without sononym and/or ableton those providers' data is left out, and the build must
   still fill every category (file names and audio stand in). With --preset NAME the
   build uses that style preset under the synthetic budgets, and must pass verify. With
   --scale it uses the style's own budgets and scales them to the library (scale =
   "library"): every category must build, within its scaled budget, and pass verify.
3. Replaces the CLAP text encoder with a deterministic stand-in (no torch, no model).
4. Runs `fourier build --all --no-scan` with small budgets from
   tests/golden/synthetic.toml (scale = "off"), into <work>/out/FourierCurated.

--sound trains a sound model (fourier/metadata/train.py) on the filled database with the
stand-in CLAP, and labels every sample with it (`fourier tools analyze --only sound`) before
the build: with Sononym it routes nothing (--check: the golden still), without it
(--without sononym) it places what the names leave open.

--schema-v1 fills the database as the code before schema 2 made it (tests/data/schema_v1:
that code's tables, at version 1), so the build migrates it first; --duckdb uses a DuckDB
database; --own then reads Fourier's own analysis (`fourier tools analyze --only own`) and
builds again. With --check every build must equal the golden: neither the migration nor
Fourier's own analysis moves a pick.

Everything is derived from the synthetic seed, so the manifest is the same on every run
of the same code on the same OS.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

# the stand-in analysis and CLAP text encoder `fourier demo` uses too
from fourier.demo import fake_embed_text, fill_db, unit  # noqa: E402,F401


def library_fingerprint(root) -> dict:
    """{relative path: (size, mtime_ns, sha256)} for every file under root: a build reads the
    library and never changes it, so this is the same before and after."""
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            st = p.stat()
            out[p.relative_to(root).as_posix()] = (st.st_size, st.st_mtime_ns,
                                                   hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def fingerprint_changes(before: dict, after: dict) -> list[str]:
    """The files added, removed or changed between two library_fingerprint()s."""
    return sorted({k for k in before.keys() | after.keys() if before.get(k) != after.get(k)})


AUDIO_FIELDS = ("out_md5", "level_db", "gain_db", "rotate_ms", "phase_fix", "root_midi", "retune",
                "root_src", "bars", "bpm_bars", "swing", "slice_clean", "slice_hits", "slice_ready")


def deps() -> dict:
    """The libraries the processed audio depends on: a golden's audio fields hold only for these."""
    import importlib
    out = {}
    for m in ("numpy", "scipy", "soundfile", "librosa", "soxr", "sklearn"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:  # noqa: BLE001 (any import or version failure: unknown)
            out[m] = None
    try:
        import soundfile
        out["libsndfile"] = soundfile.__libsndfile_version__
    except Exception:  # noqa: BLE001
        out["libsndfile"] = None
    return out


def fill_db_v1(db_path, lib_root, files, without=()) -> int:
    """fill_db's database as the code before schema 2 made it: the tables tests/data/schema_v1
    holds (that code's DDL, for SQLite and DuckDB), at version 1, then the same rows and the
    CLAP index."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from fourier.analysis.clap_features import build_index
    from fourier.demo import add_rows
    kind = "duckdb" if str(db_path).endswith(".duckdb") else "sqlite"
    eng = create_engine(f"{kind}:///{db_path}")
    ddl = (ROOT / "tests" / "data" / "schema_v1" / f"{kind}.sql").read_text()
    with eng.begin() as c:
        for stmt in (x.strip() for x in ddl.split(";\n")):
            if stmt:
                c.execute(text(stmt.rstrip(";")))
        c.execute(text("INSERT INTO schema_version (version, applied_at) VALUES (1, '2000-01-01T00:00:00')"))
    class Rows(list):                        # add_rows' objects, inserted below
        add = list.append
    got = Rows()
    add_rows(got, lib_root, files, without)
    with eng.begin() as c:                   # only the columns each row sets: the old tables
        for obj in got:                      # lack the newer ones
            c.execute(type(obj).__table__.insert(),
                      [{k: v for k, v in vars(obj).items() if not k.startswith("_")}])
    with Session(eng) as s:
        build_index(s)
    eng.dispose()
    return len(files)


def sound_model(work: Path, db: Path) -> Path:
    """Train a sound model on the database (the trainer reads it only) and use it:
    $FOURIER_SOUND_MODEL names it for the rest of the run."""
    from fourier.db import session as DS
    from fourier.metadata import train
    if DS._engine is not None:             # the fill's connection: closed, so a DuckDB file
        DS._engine.dispose()               # opens read only beside it
    DS.use_db(db)
    out = work / "sound_model.npz"
    train.main(["--db", str(db), "--ratings", "none", "--out", str(out),
              "--report", str(work / "sound_model_report.md")], log=lambda m: None)
    os.environ["FOURIER_SOUND_MODEL"] = str(out)
    return out


def golden_path() -> Path:
    return HERE / f"synthetic-{sys.platform}.json"


# what a golden records instead of the run's own stamps (the compare ignores these keys:
# fourier.packs.golden META_KEYS, CATEGORY_META_KEYS)
FIXED_TIME = "2000-01-01T00:00:00Z"


def normalized(manifest: dict, lib: Path) -> dict:
    """The manifest with this run's library path cut down to .../SampleLibrary/<relative>, and
    its timestamps and git sha replaced by fixed placeholders."""
    import json
    out = json.loads(json.dumps(manifest).replace(str(lib.parent), "<lib>"))
    if "generated" in out:
        out["generated"] = FIXED_TIME
    if "git_sha" in out:
        out["git_sha"] = ""
    for cat in (out.get("categories") or {}).values():
        if isinstance(cat, dict) and "built" in cat:
            cat["built"] = FIXED_TIME
    return out


def check(manifest: dict, lib: Path) -> int:
    import json

    from fourier.packs.golden import compare_manifests
    gp = golden_path()
    if not gp.exists():
        print(f"synthetic golden: no {gp.name} yet; make it with --update", file=sys.stderr)
        return 3
    golden = json.loads(gp.read_text())
    same_deps = golden["deps"] == deps()
    ignore = () if same_deps else AUDIO_FIELDS
    rep = compare_manifests(golden["manifest"], normalized(manifest, lib), ignore=ignore)
    print("\n".join(rep.lines()))
    if not same_deps:
        print(f"synthetic golden: audio libraries differ from the golden's ({golden['deps']} vs "
              f"{deps()}): compared structure only. Refresh with --update.")
    return 0 if rep.ok else 1


def check_fallback(manifest: dict, without) -> int:
    """A build without these providers used the ones left, and filled every category."""
    from fourier.packs.curate_config import CATEGORY_ORDER
    used = manifest.get("providers")
    if used is None or any(p in used for p in without):
        print(f"synthetic --without {','.join(without)}: the build used {used}")
        return 1
    per = {c: len((manifest["categories"].get(c) or {}).get("entries") or ()) for c in CATEGORY_ORDER}
    empty = sorted(c for c in CATEGORY_ORDER if not per[c])
    print(f"synthetic --without {','.join(without)}: providers {used}; "
          f"{sum(1 for n in per.values() if n)} of {len(per)} categories filled, {sum(per.values())} files")
    print("  " + ", ".join(f"{c} {n}" for c, n in per.items()))
    if empty:
        print(f"synthetic --without {','.join(without)}: empty: {', '.join(empty)}")
    return 1 if empty else 0


def check_scaled(manifest: dict) -> int:
    """A library-scaled build (--scale): it recorded its scale, filled every category with a
    budget no larger than the style's, and kept its folders small."""
    from fourier.packs.curate_config import BUDGETS, CATEGORY_ORDER
    sc = manifest.get("scale") or {}
    cats = manifest["categories"]
    per = {c: len((cats.get(c) or {}).get("entries") or ()) for c in CATEGORY_ORDER}
    fams = {c: len({e["family"] for e in (cats.get(c) or {}).get("entries") or ()}) for c in CATEGORY_ORDER}
    print(f"synthetic --scale: factor {sc.get('factor')} for {sc.get('samples')} samples; "
          f"{sum(1 for n in per.values() if n)} of {len(per)} categories, {sum(per.values())} files, "
          f"{sum(fams.values())} folders, verify passed")
    print("  " + ", ".join(f"{c} {per[c]}/{(cats.get(c) or {}).get('budget')} in {fams[c]}" for c in per))
    bad = [c for c in CATEGORY_ORDER if not per[c] or "budget" not in cats[c]
           or not per[c] <= cats[c]["budget"] <= BUDGETS[c]]
    if not 0 < sc.get("factor", 1) < 1 or bad:
        print(f"synthetic --scale: not scaled, or these categories missed: {', '.join(bad)}")
        return 1
    return 0


def main(argv=None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="compare with the golden for this OS")
    ap.add_argument("--update", action="store_true", help="write the golden for this OS")
    ap.add_argument("--preset", help="build with this style preset (config/presets/) over the "
                                     "synthetic budgets; it must build and pass verify; no golden")
    ap.add_argument("--without", default="", metavar="sononym,ableton",
                    help="leave these providers' data out of the DB and check the build still "
                         "fills its categories (the fallback providers); no golden")
    ap.add_argument("--scale", action="store_true",
                    help="the style's own budgets, scaled to the synthetic library (scale = \"library\"); "
                         "every category must build and pass verify; no golden")
    ap.add_argument("--interrupt", metavar="CAT,CAT", help="fail these categories in a first build, "
                    "then resume it (serial: -j 1); compare as usual")
    ap.add_argument("--schema-v1", action="store_true",
                    help="fill the database as the code before schema 2 made it; the build migrates it")
    ap.add_argument("--duckdb", action="store_true", help="a DuckDB database (else SQLite)")
    ap.add_argument("--sound", action="store_true",
                    help="train a sound model on the filled database and label the samples with it "
                         "before the build; with --check the build must still equal the golden")
    ap.add_argument("--own", action="store_true",
                    help="then read Fourier's own analysis (tools analyze --only own) and build again; "
                         "with --check both builds must equal the golden")
    a = ap.parse_args(argv)
    without = tuple(x for x in a.without.split(",") if x)
    if (without or a.preset or a.scale) and (a.check or a.update):
        ap.error("--without, --preset and --scale builds have no golden; drop --check/--update")
    work = Path(a.work).resolve()
    home, lib, out = work / "home", work / "lib" / "SampleLibrary", work / "out" / "FourierCurated"
    for d in (home, lib, out.parent, work / "locks"):
        d.mkdir(parents=True, exist_ok=True)
    config = HERE / "synthetic.toml"
    if a.scale:                   # the style's budgets, scaled to the library
        config = work / "fourier.toml"
        config.write_text(f'preset = "{a.preset or "breaks-acid"}"\nscale = "library"\n')
    elif a.preset:                # the preset, then the synthetic budgets over it
        config = work / "fourier.toml"
        budgets = [ln for ln in (HERE / "synthetic.toml").read_text().splitlines()
                   if not ln.startswith("preset =")]
        config.write_text(f'preset = "{a.preset}"\n' + "\n".join(budgets) + "\n")
    os.environ.update(FOURIER_HOME=str(home), FOURIER_CURATED_DIR=str(out),
                      FOURIER_LOCK_DIR=str(work / "locks"), FOURIER_CONFIG=str(config),
                      FOURIER_LIBRARY=str(lib))
    for k in ("FOURIER_RATINGS", "FOURIER_RESOLVED_CONFIG", "FOURIER_NO_STICKY", "FOURIER_SOUND_MODEL"):
        os.environ.pop(k, None)
    os.environ["FOURIER_NO_STICKY"] = "1"          # nothing to stick to: a fresh build
    import fourier.analysis.clap_features as CF
    from fourier import synthlib
    from fourier.packs import manifests
    CF.embed_text = fake_embed_text               # the stand-in text encoder
    files = synthlib.generate(lib)
    lib_before = library_fingerprint(lib)
    db = home / ("library.duckdb" if a.duckdb else "library.db")
    (fill_db_v1 if a.schema_v1 else fill_db)(db, lib, files, without)
    from fourier.cli import main as cli
    if a.sound:
        sound_model(work, db)
        cli(["--db", str(db), "tools", "analyze", "--only", "sound"], standalone_mode=False)
    build = ["--db", str(db), "build", "--all", "--no-scan", "-j", str(a.jobs),
             "--out", str(out), "--no-describe"]
    if a.interrupt:
        # stop a first build after some categories (the rest fail), then resume it: what the
        # resumed build makes must be what an uninterrupted one makes (the golden, with --check)
        # resolve the config first (a harmless command), so curate is imported with it
        cli(["--db", str(db), "config", "show"], standalone_mode=False)
        from fourier.packs import curate
        real, stop = curate.build_taxonomy, set(a.interrupt.split(","))

        def failing(session, category, *args, **kw):
            if category in stop:
                raise RuntimeError("stopped for the resume check")
            return real(session, category, *args, **kw)
        curate.build_taxonomy = failing
        try:
            cli(build, standalone_mode=False)
        except SystemExit as e:
            assert e.code, "the interrupted build should fail"
        finally:
            curate.build_taxonomy = real
        build.append("--resume")
    try:
        cli(build, standalone_mode=False)
    except SystemExit as e:
        partial = out.with_name(out.name + ".next") / "manifest.json"
        if e.code and not (without and partial.exists()):
            return int(e.code)
        if e.code:          # categories failed: check what the partial build filled
            return check_fallback(manifests.read(partial), without)
    if a.own:
        # Fourier's own analysis on top (with Sononym it moves no pick), then the build again;
        # the first build's manifest is checked here, the second's below
        if a.check:
            first = check(manifests.read(out / "manifest.json"), lib)
            if first:
                print("synthetic --own: the build before Fourier's own analysis differs from the golden")
                return first
        try:
            cli(["--db", str(db), "tools", "analyze", "--only", "own", "--workers", "2"],
                standalone_mode=False)
            cli(build, standalone_mode=False)
        except SystemExit as e:
            if e.code:
                return int(e.code)
    # the other readers of the sample rows run too: a crash in any of them fails the check
    # (tools audit maps the master to the library by file name, which renaming breaks here)
    for args in (["why", "Synth Drums", "--master", str(out)], ["tools", "db-stats", "--metadata"],
                 ["tools", "db-stats", "--disagreements"]):
        try:
            cli(["--db", str(db), *args], standalone_mode=False)
        except SystemExit as e:
            if e.code:
                print(f"synthetic: `fourier {' '.join(args[:2])}` failed ({e.code})")
                return int(e.code)
    changed = fingerprint_changes(lib_before, library_fingerprint(lib))
    if changed:                   # the library is read, never written
        print(f"synthetic: the build changed {len(changed)} library file(s), e.g. {changed[0]}")
        return 1
    manifest = manifests.read(out / "manifest.json")
    if a.scale:                   # it built, so verify passed
        return check_scaled(manifest)
    if a.preset:                  # it built, so verify passed
        print(f"synthetic --preset {a.preset}: {len(manifest['categories'])} categories, "
              f"{sum(len(v.get('entries') or ()) for v in manifest['categories'].values())} files, verify passed")
        return 0
    if without:
        return check_fallback(manifest, without)
    if a.update:
        golden_path().write_text(json.dumps({"deps": deps(), "manifest": normalized(manifest, lib)},
                                            indent=1, sort_keys=True) + "\n")
        print(f"synthetic golden: wrote {golden_path().name}")
    if a.check:
        return check(manifest, lib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
