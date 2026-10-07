"""A library that moves under a built master, through the real CLI with the stand-in CLAP
(as tests/test_first_run.py runs it): a library folder that is missing or empty (an
unmounted drive) stops a build before anything changes; so do sources of the master that
are gone from disk while the database would still pick them; a pick that can't be exported
stops the build before the swap instead of leaving a short master; and a renamed folder,
once scanned, builds a whole master from the files where they are now."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fourier.packs import manifests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_first_run import Sandbox, _flat, _has  # noqa: E402

# one sandbox, built once, that the tests below change in turn: one worker runs them all, in
# order (pytest -n: --dist loadgroup, conftest.py)
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("missing_library")]


def _run(b: Sandbox, *args, ok=True):
    r = subprocess.run([sys.executable, str(b.runner), *args], env=b.env(), cwd=b.tmp,
                       capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, f"fourier {' '.join(args)}:\n{out[-3000:]}"
    if ok is not None:
        assert (r.returncode == 0) == ok, f"fourier {' '.join(args)} -> {r.returncode}:\n{out[-3000:]}"
    return _flat(out)


def _manifest(b):
    return (b.master / "manifest.json").read_bytes()


def _sources(b):
    man = manifests.read(b.master / "manifest.json")             # sources absolute
    return sorted(e["src"] for cd in man["categories"].values() for e in cd.get("entries") or ())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    b = Sandbox(tmp_path_factory.mktemp("missing"))
    _run(b, "setup", "--yes", "--library", str(b.lib), "--device", "digitakt_2", "--preset",
         "balanced", "--no-clap")
    _run(b, "build", "--all")
    assert (b.master / "manifest.json").exists()
    return b


def test_a_missing_or_empty_library_folder_stops_the_build(built):
    b = built
    before = _manifest(b)
    away = b.tmp / "unplugged"
    os.rename(b.lib, away)
    try:
        out = _run(b, "build", "--all", "--no-scan", ok=False)
        assert _has(out, f"The library folder {b.lib} is missing.") and "unmounted drive" in out
        out = _run(b, "build", "--all", ok=False)                 # before the scan, too
        assert "is missing" in out and "1/4 Scan the library" not in out
        b.lib.mkdir()                                             # the mount point, nothing mounted
        out = _run(b, "build", "--all", "--no-scan", ok=False)
        assert "is empty" in out and "nothing in" in out
        out = _run(b, "tools", "scan", ok=False)
        assert "is empty" in out and "no sample was marked missing" in out
        b.lib.rmdir()
    finally:
        os.rename(away, b.lib)
    assert _manifest(b) == before
    assert b.db("SELECT COUNT(*) FROM missing_files")[0][0] == 0


def test_sources_gone_from_disk_stop_the_build_until_scanned(built):
    b = built
    before = _manifest(b)
    srcs = _sources(b)
    gone = [p for p in srcs if "/Vendor B/Drums/Hats/" in p or "/Vendor B/Drums/Kicks/" in p]
    assert len(gone) >= 3, srcs
    stash = b.tmp / "stash"
    stash.mkdir()
    for p in gone:
        shutil.move(p, stash / Path(p).name)
    try:
        out = _run(b, "build", "--all", "--no-scan", ok=False)
        assert _has(out, f"{len(gone)} of the {len(srcs)} files the master at {b.master} was built from are gone")
        assert "Build stopped" in out and os.path.basename(gone[0]) in out
        assert _manifest(b) == before and not b.master.with_name(b.master.name + ".next").exists()
        # a walk marks them missing: the build then leaves them out and replaces the master
        out = _run(b, "tools", "scan")
        assert f"{len(gone)} sample(s) in the database weren't found" in out
        out = _run(b, "build", "--all", "--no-scan")
        assert "New master in place" in out
        assert not set(gone) & set(_sources(b)) and all(os.path.exists(p) for p in _sources(b))
    finally:
        for p in gone:
            shutil.move(stash / Path(p).name, p)
    _run(b, "tools", "scan")                                          # back again
    assert b.db("SELECT COUNT(*) FROM missing_files")[0][0] == 0


def test_a_pick_that_cant_be_exported_stops_before_the_swap(built):
    b = built
    _run(b, "build", "--all", "--no-scan")
    before = _manifest(b)
    victim = next(p for p in _sources(b) if "/Drum Loops/" in p)
    away = b.tmp / "away.wav"
    shutil.move(victim, away)            # one file gone: too few to stop before the build
    try:
        out = _run(b, "build", "--all", "--no-scan", ok=False)
        assert "1 picked file couldn't be exported (DRUMLOOPS 1)" in out, out
        assert _has(out, f"The master at {b.master} is unchanged") and "2/2 Verify" not in out
        assert _manifest(b) == before
    finally:
        shutil.move(away, victim)
    out = _run(b, "build", "--all", "--no-scan", "--resume")
    assert "New master in place" in out and "couldn't be exported" not in out


def test_a_renamed_folder_builds_from_where_the_files_are_now(built):
    b = built
    n_before = len(_sources(b))
    old, new = b.lib / "Vendor C", b.lib / "Vendor C Renamed"
    os.rename(old, new)
    try:
        out = _run(b, "build", "--all")
        # the same files at new paths: moved, with their analysis (nothing new to analyze)
        assert "16 file(s) moved or renamed in the library" in out and "16 new" in out
        assert "weren't found" not in out and "new samples to analyze" not in out, out
        assert "New master in place" in out
        srcs = _sources(b)
        assert all(os.path.exists(p) for p in srcs) and not [p for p in srcs if "/Vendor C/" in p]
        assert [p for p in srcs if "/Vendor C Renamed/" in p] and len(srcs) == n_before
        _run(b, "verify")
    finally:
        os.rename(new, old)


def _store(b):
    p = b.home / "ratings.json"
    return json.loads(p.read_text()).get("ratings", {}) if p.exists() else {}


def test_a_renamed_folder_takes_its_keep_ratings_along(built):
    """The walk sees the same content at a new path with the old path gone: a move. Its
    analysis and its ratings follow it, so the Keep stays a Keep of the file where it is now,
    and no later harvest brings the old path back."""
    b = built
    _run(b, "build", "--all")                          # (the folder renamed above is back)
    src = next(p for p in _sources(b) if "/Vendor B/Drums/Kicks/" in p)
    _run(b, "review", "rate", src, "keep")
    old, new = b.lib / "Vendor B" / "Drums", b.lib / "Vendor B" / "Drum Hits"
    moved = src.replace("/Vendor B/Drums/", "/Vendor B/Drum Hits/")
    os.rename(old, new)
    try:
        out = _run(b, "build", "--all")
        assert "moved or renamed in the library" in out and "New master in place" in out
        assert moved in _store(b) and src not in _store(b)
        assert moved in _sources(b)
        log = (b.master / "CHANGELOG.md").read_text()
        assert "moved in the library" in log, log
        out = _run(b, "build", "--all")                  # the old master's tag came back once
        assert src not in _store(b) and "verify: PASS" in out
    finally:
        os.rename(new, old)
        _run(b, "tools", "scan")
    assert src in _store(b) and moved not in _store(b)   # and back with it


def test_a_keep_whose_file_is_gone_warns_and_can_be_cleared(built):
    b = built
    gone = str(b.lib / "Vendor A" / "Drums" / "Kicks" / "Old Kick 99.wav")
    p = b.home / "ratings.json"
    doc = json.loads(p.read_text()) if p.exists() else {"version": 1}
    doc.setdefault("ratings", {})[gone] = dict(verdict="keep", category="KICKS", family="kick",
                                               out="KICKS/kick/Old_Kick_99.wav", master="manual",
                                               rated_at="2026-01-01T00:00:00+00:00", source="cli")
    p.write_text(json.dumps(doc))
    out = _run(b, "build", "--all", "--no-scan")
    i = out.find("at files that are gone")
    assert "1 Keep rating points at files that are gone (moved or deleted): Old Kick 99.wav" in out, \
        out[max(0, i - 200):i + 300]
    assert "New master in place" in out
    out = _run(b, "review", "rate", "Old Kick 99", "clear")
    assert "Keep rating cleared" in out and gone not in _store(b)


def test_an_unplugged_linked_pack_stops_the_build(built):
    """A folder symlink whose target isn't there (a drive not plugged in) is reported like a
    missing library folder: the build stops before anything changes, unless --allow-missing."""
    import numpy as np
    import soundfile as sf
    b = built
    ext = b.tmp / "drive" / "Linked Pack"
    for i in range(3):
        f = ext / "Kicks" / f"Linked Kick {i + 1}.wav"
        f.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(0.3 * 44100)) / 44100
        sf.write(f, (0.8 * np.sin(2 * np.pi * (60 + 7 * i) * t) * np.exp(-t / 0.1)).astype("float32"), 44100)
    link = b.lib / "Linked Pack"
    os.symlink(ext, link)
    away = ext.with_name("Linked Pack (unplugged)")
    try:
        _run(b, "build", "--all")
        assert "FAIL linked folder" not in " ".join(_run(b, "doctor", ok=None).split())
        os.rename(ext, away)
        before = _manifest(b)
        # doctor says what the build stops on, as the scan says it
        out = " ".join(_run(b, "doctor", ok=False).split())
        assert f"FAIL linked folder: A linked folder is unavailable: {link} -> " in out, out
        assert "--allow-missing" in out
        out = _run(b, "build", "--all", ok=False)
        assert _has(out, f"A linked folder is unavailable: {link}"), out
        assert _has(out, f"Build stopped: nothing in {b.master} changed") and _manifest(b) == before
        out = _run(b, "build", "--all", "--allow-missing")
        assert "New master in place" in out
        assert not [s for s in _sources(b) if "/Linked Pack/" in s]
    finally:
        if away.exists():
            os.rename(away, ext)
        os.unlink(link)
        _run(b, "tools", "scan")
