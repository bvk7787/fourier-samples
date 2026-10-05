"""Device path locks: paths already on a device never change across rebuilds."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from fourier.devices.loader import DeviceProfile
from fourier.packs import device_lock as L
from fourier.packs import releases
from fourier.packs import render as R


def _device(card_dir="/Samples/Fourier", max_path_length=127):
    return DeviceProfile(device_id="m8_test", name="M8 test", sample_rate=44100, bit_depth=16,
                         channels="mono", folder_depth=2, max_path_length=max_path_length,
                         card_dir=card_dir, dither=False)


def _master(root: Path, files: dict[str, tuple[str, float]]):
    """files: {"CAT/family/name.wav": (src_path, freq)}; writes wavs + manifest.json."""
    cats: dict[str, list] = {}
    for rel, (src, freq) in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(4410) / 44100
        sf.write(str(p), (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), 44100)
        cat, fam, name = rel.split("/")
        cats.setdefault(cat, []).append(dict(family=fam, out=f"{fam}/{name}", src=src,
                                             out_md5=f"md5-{src}-{freq}"))
    (root / "manifest.json").write_text(json.dumps(
        {"categories": {c: {"entries": e} for c, e in cats.items()}}))
    return root


def _fresh(f):
    return f"{f.category}/{f.family}/{f.stem}.wav"


def _mf(key, master, md5=None):
    cat, fam, name = master.split("/")
    return L.MasterFile(key, master, md5, cat, fam, name[:-4])


def test_assign_keeps_locked_paths_and_avoids_them_case_insensitively():
    lock = L.Lock("d", "/S", files={
        "src/kick.wav": dict(path="KICKS/fam/Kick.wav", master="KICKS/fam/Kick.wav", md5="1"),
        "src/retired.wav": dict(path="KICKS/fam/Retired.wav", master="KICKS/fam/Retired.wav", md5="2")})
    files = [_mf("src/kick.wav", "KICKS/fam2/Kick.wav", "1"),         # moved in the master
             _mf("src/new.wav", "KICKS/fam/KICK.wav", "3"),           # collides (case) with locked
             _mf("src/new2.wav", "KICKS/fam/retired.wav", "4")]          # collides with a retired path
    a = L.assign(files, lock, _fresh)
    assert a.paths["src/kick.wav"] == "KICKS/fam/Kick.wav"
    assert a.paths["src/new.wav"] == "KICKS/fam/KICK_2.wav"
    assert a.paths["src/new2.wav"] == "KICKS/fam/retired_2.wav"
    assert a.retired == ["src/retired.wav"] and a.kept == ["src/kick.wav"] and a.changed == []


def test_assign_is_order_independent_and_flags_changed_audio():
    lock = L.Lock("d", "", files={"k": dict(path="A/f/x.wav", master="A/f/x.wav", md5="old")})
    files = [_mf("k", "A/f/x.wav", "new"), _mf("n1", "A/f/y.wav"), _mf("n2", "A/g/y.wav")]
    a1 = L.assign(files, lock, _fresh)
    a2 = L.assign(list(reversed(files)), lock, _fresh)
    assert a1.paths == a2.paths and a1.changed == ["k"]


def test_suffix_respects_path_limit():
    taken = {"a/b/" + "x" * 20 + ".wav"}
    out = L._unique("a/b/" + "x" * 20 + ".wav", taken, limit_len=30, prefix_len=0)
    assert len(out) <= 30 and out.endswith("_2.wav") and out.lower() not in taken


def test_release_render_locks_and_rebuilds_never_move_device_paths(tmp_path, monkeypatch):
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    lock_dir = tmp_path / "locks"
    dev = _device()

    v1 = _master(rel_root / "v1", {
        "KICKS/dark/BD_01.wav": ("src/bd1", 60), "KICKS/dark/BD_02.wav": ("src/bd2", 70),
        "BASS/sub/Sub_A.wav": ("src/suba", 40)})
    out1 = tmp_path / "r1"
    r1 = R.render_master_to_device(v1, dev, out1, log=lambda m: None, lock=None, release="v1",
                                   write_lock=True, lock_dir=lock_dir)
    lock = L.load_lock("m8_test", lock_dir)
    assert r1["new"] == 3 and len(lock.files) == 3 and lock.card_dir == "/Samples/Fourier"
    before = {p.relative_to(out1).as_posix(): p.read_bytes() for p in out1.rglob("*.wav")}

    # a rebuild: bd1 re-homed and re-transcoded, bd2 dropped, a new file whose name collides
    work = _master(tmp_path / "master", {
        "KICKS/punchy/BD_01.wav": ("src/bd1", 65),
        "KICKS/dark/bd_02.wav": ("src/bd_new", 80),
        "BASS/sub/Sub_A.wav": ("src/suba", 40)})
    plan = R.device_plan(work, dev, lock, log=lambda m: None)
    assert plan["safe"] and plan["changed"] == 1 and plan["retired"] == 1 and plan["new"] == 1

    out2 = tmp_path / "r2"
    r2 = R.render_master_to_device(work, dev, out2, log=lambda m: None, lock=lock)
    after = {p.relative_to(out2).as_posix(): p.read_bytes() for p in out2.rglob("*.wav")}
    # every path already on the device is still there with identical audio
    assert all(after.get(k) == v for k, v in before.items())
    assert "01_KICKS/dark/bd_02_2.wav" in after and len(after) == 4
    assert r2["retired"] == 1 and r2["missing"] == 0

    # the card_dir is part of the contract
    with pytest.raises(RuntimeError, match="card_dir"):
        R.render_master_to_device(work, _device(card_dir="/Samples/Other"), tmp_path / "r3",
                                  log=lambda m: None, lock=lock)


def test_plan_is_unsafe_when_original_audio_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(tmp_path / "no_releases"))
    lock = L.Lock("m8_test", "/Samples/Fourier", files={
        "src/bd2": dict(path="KICKS/dark/BD_02.wav", master="KICKS/dark/BD_02.wav", md5="x",
                        release="v1")})
    work = _master(tmp_path / "master", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    plan = R.device_plan(work, _device(), lock, log=lambda m: None)
    assert not plan["safe"] and plan["at_risk"] == 1


def test_write_lock_requires_a_release(tmp_path):
    work = _master(tmp_path / "master", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    with pytest.raises(RuntimeError, match="release"):
        R.render_master_to_device(work, _device(), tmp_path / "out", log=lambda m: None,
                                  write_lock=True, lock_dir=tmp_path / "locks")


def test_uppercase_wav_and_cross_category_moves(tmp_path, monkeypatch):
    """.WAV files render (some sources use an upper-case extension); a move to another category keeps the old
    device file and adds a new one; the same source in two categories stays two files."""
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    dev = _device()
    v1 = _master(rel_root / "v1", {"SYNTH/keys/EP_C3.WAV": ("src/ep", 130),
                                   "PIANO/grand/EP_C3.wav": ("src/ep", 130)})
    lock_dir = tmp_path / "locks"
    r1 = R.render_master_to_device(v1, dev, tmp_path / "r1", log=lambda m: None, release="v1",
                                   write_lock=True, lock_dir=lock_dir)
    assert r1["files"] == 2 and r1["new"] == 2
    lock = L.load_lock("m8_test", lock_dir)
    assert set(lock.files) == {"SYNTH|src/ep", "PIANO|src/ep"}
    work = _master(tmp_path / "master", {"PIANO/grand/EP_C3.wav": ("src/ep", 130),
                                         "ACOUSTIC/str/Cello.wav": ("src/cello", 220)})
    plan = R.device_plan(work, dev, lock, log=lambda m: None)
    assert plan["safe"] and plan["retired"] == 1 and plan["new"] == 1 and plan["kept"] == 1
    r2 = R.render_master_to_device(work, dev, tmp_path / "r2", log=lambda m: None, lock=lock)
    assert r2["retired"] == 1 and r2["new"] == 1
    names = sorted(p.relative_to(tmp_path / "r2").as_posix() for p in (tmp_path / "r2").rglob("*.wav"))
    assert names == ["11_SYNTH/keys/EP_C3.wav", "14_PIANO/grand/EP_C3.wav", "15_ACOUSTIC/str/Cello.wav"]


def test_lock_pins_the_audio_format(tmp_path, monkeypatch):
    """Changing the profile's rate/depth/mono would silently change every locked file's audio."""
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    v1 = _master(rel_root / "v1", {"KICKS/dark/BD_01.wav": ("src/bd1", 60)})
    R.render_master_to_device(v1, _device(), tmp_path / "r1", log=lambda m: None, release="v1",
                              write_lock=True, lock_dir=tmp_path / "locks")
    lock = L.load_lock("m8_test", tmp_path / "locks")
    assert lock.fmt == dict(sample_rate=44100, bit_depth=16, mono=True)
    dev = _device()
    dev.sample_rate = 48000
    assert not R.device_plan(v1, dev, lock, log=lambda m: None)["safe"]
    with pytest.raises(RuntimeError, match="audio format"):
        R.render_master_to_device(v1, dev, tmp_path / "r2", log=lambda m: None, lock=lock)
