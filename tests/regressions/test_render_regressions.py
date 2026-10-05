"""Regressions: audio conversion + device rendering + path locks.

Each test reproduces a defect the code must not bring back.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from fourier.devices import exporter as E
from fourier.devices.loader import DeviceProfile
from fourier.packs import device_lock as L
from fourier.packs import releases
from fourier.packs import render as R


def _dev(card_dir="/Samples/Fourier", root="", max_path_length=127):
    """M8-like: 44.1k/16/mono, no device root (so the render root is out_dir itself)."""
    return DeviceProfile(device_id="m8_bh", name="M8 bh", sample_rate=44100, bit_depth=16,
                         channels="mono", folder_depth=2, max_path_length=max_path_length,
                         card_dir=card_dir, root=root)


def _master(root: Path, files: dict, manifest=True, n=4410):
    """files: {"CAT/family/name.wav": (src, freq)}"""
    cats: dict = {}
    for rel, (src, freq) in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(n) / 44100
        sf.write(str(p), (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), 44100)
        cat, fam, name = rel.split("/")
        cats.setdefault(cat, []).append(dict(family=fam, out=f"{fam}/{name}", src=src,
                                             out_md5=f"md5-{src}-{freq}"))
    if manifest:
        (root / "manifest.json").write_text(json.dumps(
            {"categories": {c: {"entries": e} for c, e in cats.items()}}))
    return root


_quiet = lambda m: None


# --- render.py ------------------------------------------------------------------------------

def test_render_out_dir_that_contains_master_does_not_delete_master(tmp_path):
    """render_master_to_device clears out_dir (device root "" as on M8/Digitakt), so it must
    refuse an out_dir that contains the master (a render into the master's parent folder would
    delete the master)."""
    music = tmp_path / "Music"
    master = _master(music / "FourierCurated", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    try:
        R.render_master_to_device(master, _dev(), music, log=_quiet)
    except Exception:
        pass
    assert (master / "KICKS/dark/BD_01.wav").exists(), "render deleted the master it renders from"


def test_write_lock_does_not_lock_files_the_budget_dropped(tmp_path, monkeypatch):
    """write_lock records every a.new path even when --max-mb skipped the file, so the lock claims
    it is on the device; next render treats it as locked and exempts it from the budget."""
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    v1 = _master(rel_root / "v1", {f"KICKS/dark/BD_{i:02d}.wav": (f"src/bd{i}", 60 + i)
                                   for i in range(6)}, n=44100)
    out = tmp_path / "r1"
    r = R.render_master_to_device(v1, _dev(), out, max_mb=0.2, log=_quiet, release="v1",
                                  write_lock=True, lock_dir=tmp_path / "locks")
    assert r["skipped"] > 0
    lock = L.load_lock("m8_bh", tmp_path / "locks")
    on_disk = {p.relative_to(out).as_posix() for p in out.rglob("*.wav")}
    locked = {e["path"] for e in lock.files.values()}
    assert locked <= on_disk, f"locked but never rendered: {sorted(locked - on_disk)}"


def test_plan_unsafe_when_device_root_changes(tmp_path, monkeypatch):
    """Lock paths are relative to the device root (paths.root). If the profile gains a root,
    every on-card path moves, but only card_dir was checked."""
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    v1 = _master(rel_root / "v1", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    dev1 = _dev()  # root ""
    R.render_master_to_device(v1, dev1, tmp_path / "r1", log=_quiet, release="v1",
                              write_lock=True, lock_dir=tmp_path / "locks")
    lock = L.load_lock("m8_bh", tmp_path / "locks")
    before = R.card_paths(v1, dev1, lock)
    dev2 = _dev(root="SAMPLES")
    after = R.card_paths(v1, dev2, lock)
    assert before != after  # precondition: the on-card path really changes
    plan = R.device_plan(v1, dev2, lock, log=_quiet)
    assert not plan["safe"], f"plan says SAFE although {before} -> {after}"


def test_card_dir_trailing_slash_is_the_same_card_location(tmp_path):
    """_card_prefix strips the trailing slash, so '/Samples/Fourier/' and '/Samples/Fourier'
    produce identical paths, yet the lock check compares raw strings and refuses/flags NOT SAFE."""
    work = _master(tmp_path / "m", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    lock = L.Lock("m8_bh", "/Samples/Fourier", files={
        "KICKS|src/bd1": dict(path="KICKS/dark/BD_01.wav", master="KICKS/dark/BD_01.wav",
                              md5="md5-src/bd1-60", release="v1")})
    dev = _dev(card_dir="/Samples/Fourier/")
    assert R.card_paths(work, dev, lock) == R.card_paths(work, _dev(), lock)
    assert R.device_plan(work, dev, lock, log=_quiet)["safe"]


def test_locked_file_without_manifest_md5_keeps_its_audio(tmp_path, monkeypatch):
    """Change detection relies solely on the manifest's out_md5. A master file with no manifest
    entry (key 'master:...', md5 None) can have its audio replaced and the render overwrites the
    locked device path with the new audio while device_plan says SAFE."""
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    v1 = _master(rel_root / "v1", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)}, manifest=False)
    R.render_master_to_device(v1, _dev(), tmp_path / "r1", log=_quiet, release="v1",
                              write_lock=True, lock_dir=tmp_path / "locks")
    lock = L.load_lock("m8_bh", tmp_path / "locks")
    original = (tmp_path / "r1/01_KICKS/dark/BD_01.wav").read_bytes()

    work = _master(tmp_path / "m", {"KICKS/dark/BD_01.wav": ("src/bd1", 900)}, manifest=False)
    plan = R.device_plan(work, _dev(), lock, log=_quiet)
    R.render_master_to_device(work, _dev(), tmp_path / "r2", log=_quiet, lock=lock)
    rendered = (tmp_path / "r2/01_KICKS/dark/BD_01.wav").read_bytes()
    assert rendered == original or plan["changed"] == 1, \
        "locked path got new audio and the plan did not flag it"


def test_render_reports_a_failed_conversion(tmp_path, monkeypatch):
    """A conversion failure is swallowed (failed += 1; continue) and never logged; for a locked
    path this means the image silently lacks a file the device's projects reference."""
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    v1 = _master(rel_root / "v1", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    R.render_master_to_device(v1, _dev(), tmp_path / "r1", log=_quiet, release="v1",
                              write_lock=True, lock_dir=tmp_path / "locks")
    lock = L.load_lock("m8_bh", tmp_path / "locks")
    work = _master(tmp_path / "m", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    (work / "KICKS/dark/BD_01.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")  # unreadable
    msgs = []
    r = R.render_master_to_device(work, _dev(), tmp_path / "r2", log=msgs.append, lock=lock)
    assert r["failed"] == 1  # precondition
    msgs = [m.replace(str(tmp_path), "<tmp>") for m in msgs]
    assert any("BD_01" in m or "fail" in m.lower() for m in msgs), msgs


def test_same_source_twice_in_category_is_not_double_counted(tmp_path):
    """assign() gives one path to a key ('first wins') but render still queues a job per master
    file, so both write the same dest; the summary counts 2 files for 1 on disk (and the last
    file's audio lands under the first file's name)."""
    work = _master(tmp_path / "m", {"KICKS/dark/BD_01.wav": ("src/bd1", 60),
                                    "KICKS/punchy/BD_01b.wav": ("src/bd1", 61)})
    out = tmp_path / "out"
    r = R.render_master_to_device(work, _dev(), out, log=_quiet)
    assert r["files"] == len(list(out.rglob("*.wav")))


# --- exporter.py: _convert_and_copy -----------------------------------------------------------

def test_convert_honours_dither_flag(tmp_path):
    """`dither` is accepted but never used: 24->16-bit output is byte-identical with dither on or
    off (plain libsndfile truncation), although every device YAML sets dither: true."""
    src = tmp_path / "quiet24.wav"
    t = np.arange(44100) / 44100
    sf.write(str(src), (2e-5 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 44100,
             subtype="PCM_24")
    a, b = tmp_path / "d.wav", tmp_path / "nd.wav"
    E._convert_and_copy(src, a, 44100, 16, True, dither=True)
    E._convert_and_copy(src, b, 44100, 16, True, dither=False)
    assert a.read_bytes() != b.read_bytes(), "dither=True had no effect on a 24->16-bit reduction"


def test_resample_preserves_exact_length(tmp_path):
    """librosa computes ceil(n * 48000/44100) in floating point; for loops cut to an exact
    multiple of 441 samples (e.g. 4 s @120 BPM, 1.92 s @125 BPM) that yields one extra sample."""
    src = tmp_path / "loop.wav"
    n = 176400  # 4.0 s at 44.1k -> exactly 192000 at 48k
    sf.write(str(src), (0.3 * np.sin(np.arange(n) / 10.0)).astype("float32"), 44100)
    dst = tmp_path / "out.wav"
    E._convert_and_copy(src, dst, 48000, 16, True, dither=False)
    assert sf.info(str(dst)).frames == 192000
