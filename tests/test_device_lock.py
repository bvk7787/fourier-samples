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
    for rel, (src, freq, *md5) in files.items():       # an optional third item: the out_md5
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(4410) / 44100
        sf.write(str(p), (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), 44100)
        cat, fam, name = rel.split("/")
        cats.setdefault(cat, []).append(dict(family=fam, out=f"{fam}/{name}", src=src,
                                             out_md5=md5[0] if md5 else f"md5-{src}-{freq}"))
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


# A library that moved: the lock's keys name each source by its absolute path
OLD_LIB = "/Users/q/Music/SampleLibrary"          # conftest: FOURIER_LIBRARY=SampleLibrary


def _moved(files: dict, old: str, new: str) -> dict:
    return {rel: (src.replace(old, new, 1), *rest) for rel, (src, *rest) in files.items()}


def test_library_root_change_keeps_every_locked_path(tmp_path, monkeypatch):
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    lock_dir, dev = tmp_path / "locks", _device()
    files = {f"KICKS/dark/BD_{i:02d}.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qk {i}.wav", 50 + i, f"m{i}")
             for i in range(1, 6)}
    files["BASS/sub/Sub_A.wav"] = (f"{OLD_LIB}/Northwind/QX4d/Qb A.wav", 40, "ms")
    R.render_master_to_device(_master(rel_root / "v1", files), dev, tmp_path / "r1",
                              log=lambda m: None, release="v1", write_lock=True, lock_dir=lock_dir)
    lock = L.load_lock("m8_test", lock_dir)
    before = {e["path"] for e in lock.files.values()}

    # the same library, now on another volume: every source path changed, nothing else did
    v2 = _master(rel_root / "v2", _moved(files, "/Users/q/Music", "/Volumes/Qzv"))
    plan = R.device_plan(v2, dev, lock, log=lambda m: None)
    assert plan["safe"] and plan["kept"] == 6 and plan["new"] == plan["retired"] == 0
    r2 = R.render_master_to_device(v2, dev, tmp_path / "r2", log=lambda m: None, lock=lock,
                                   release="v2", write_lock=True, lock_dir=lock_dir)
    assert r2["kept"] == 6 and r2["new"] == 0 and r2["files"] == 6
    assert not [p for p in (tmp_path / "r2").rglob("*_2.wav")]
    lock = L.load_lock("m8_test", lock_dir)       # re-keyed to the new sources, paths as they were
    assert {e["path"] for e in lock.files.values()} == before
    assert all(k.split("|", 1)[1].startswith("/Volumes/Qzv/SampleLibrary/") for k in lock.files)
    assert {e["release"] for e in lock.files.values()} == {"v1"}
    a = L.assign(L.master_files(v2), lock, _fresh)
    assert len(a.kept) == 6 and not a.rekeyed and not a.new and not a.retired


def _lock_of(files: dict) -> L.Lock:
    return L.Lock("d", "", files={f"{m.split('/')[0]}|{src}": dict(path=f"01_{m}", master=m, md5=md5)
                                  for m, (src, md5) in files.items()})


def _files(files: dict) -> list:
    return [_mf(f"{m.split('/')[0]}|{src}", m, md5) for m, (src, md5) in files.items()]


def test_renamed_pack_folder_keeps_its_paths_and_new_files_get_new_ones():
    lock = _lock_of({"KICKS/dark/BD_1.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qk 1.wav", "a"),
                     "SNARES/dry/SD_1.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qs 1.wav", "b")})
    now = {"KICKS/dark/BD_1.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit Zq/Qk 1.wav", "a"),     # pack renamed
           "SNARES/dry/SD_1.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit Zq/Qs 1.wav", "b"),
           "KICKS/dark/BD_2.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit Zq/Qk 2.wav", "c")}     # new file
    a = L.assign(_files(now), lock, _fresh)
    assert len(a.kept) == 2 and not a.retired and not a.changed
    assert a.paths[f"KICKS|{OLD_LIB}/Acme/Qxvwz Kit Zq/Qk 1.wav"] == "01_KICKS/dark/BD_1.wav"
    assert a.new == [f"KICKS|{OLD_LIB}/Acme/Qxvwz Kit Zq/Qk 2.wav"]
    assert a.paths[a.new[0]] == "KICKS/dark/BD_2.wav"
    rec = L.record(lock, "d", "", _files(now), a, "v2")
    assert sorted(e["path"] for e in rec.files.values()) == [
        "01_KICKS/dark/BD_1.wav", "01_SNARES/dry/SD_1.wav", "KICKS/dark/BD_2.wav"]
    assert all("Qxvwz Kit Zq/" in k for k in rec.files)


def test_renamed_library_folder_follows_the_library_path(monkeypatch):
    """The library folder itself renamed (and fourier.toml with it), the audio re-processed:
    the same vendor/pack/file keeps its path, flagged changed, as when nothing moved."""
    monkeypatch.setenv("FOURIER_LIBRARY", "/Volumes/Qzv/Samples")
    lock = _lock_of({"KICKS/dark/BD_1.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qk 1.wav", "a"),
                     "KICKS/dark/BD_9.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qk 9.wav", "z")})
    now = {"KICKS/punchy/BD_1.wav": ("/Volumes/Qzv/Samples/Acme/Qxvwz Kit/Qk 1.wav", "a2")}
    a = L.assign(_files(now), lock, _fresh)
    k = "KICKS|/Volumes/Qzv/Samples/Acme/Qxvwz Kit/Qk 1.wav"
    assert a.kept == [k] and a.changed == [k] and a.paths[k] == "01_KICKS/dark/BD_1.wav"
    assert a.locked_key(k) == f"KICKS|{OLD_LIB}/Acme/Qxvwz Kit/Qk 1.wav"
    assert a.retired == [f"KICKS|{OLD_LIB}/Acme/Qxvwz Kit/Qk 9.wav"]


def test_changed_audio_under_a_renamed_pack_is_a_new_file_as_before():
    lock = _lock_of({"KICKS/dark/BD_1.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qk 1.wav", "a")})
    a = L.assign(_files({"KICKS/dark/BD_1.wav": (f"{OLD_LIB}/Acme/Vqx/Qk 1.wav", "a2")}), lock, _fresh)
    assert not a.kept and len(a.new) == 1 and len(a.retired) == 1 and not a.rekeyed
    # unmoved, the same source with new audio keeps its path and is flagged changed
    a = L.assign(_files({"KICKS/dark/BD_1.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qk 1.wav", "a2")}),
                 lock, _fresh)
    assert len(a.kept) == 1 and len(a.changed) == 1 and not a.rekeyed


def test_a_library_path_two_locked_sources_share_is_not_guessed():
    lock = _lock_of({"KICKS/dark/BD_1.wav": ("/Volumes/Qa/SampleLibrary/Acme/Kit/Qk 1.wav", "a"),
                     "KICKS/deep/BD_1.wav": ("/Volumes/Qb/SampleLibrary/Acme/Kit/Qk 1.wav", "b")})
    a = L.assign(_files({"KICKS/dark/BD_1x.wav": ("/Volumes/Qc/SampleLibrary/Acme/Kit/Qk 1.wav", "c")}),
                 lock, _fresh)
    assert not a.kept and len(a.new) == 1 and len(a.retired) == 2


def test_plan_is_unsafe_when_a_new_file_repeats_locked_audio(tmp_path, monkeypatch):
    """A source the lock can't follow (pack renamed and re-homed to another family at once)
    would go on the card twice: render --dry-run says so instead of SAFE."""
    rel_root = tmp_path / "releases"
    monkeypatch.setattr(releases, "RELEASES_ROOT", str(rel_root))
    v1 = _master(rel_root / "v1", {"KICKS/dark/BD_01.wav": (f"{OLD_LIB}/Acme/Qxvwz Kit/Qk 1.wav", 60, "m1")})
    R.render_master_to_device(v1, _device(), tmp_path / "r1", log=lambda m: None, release="v1",
                              write_lock=True, lock_dir=tmp_path / "locks")
    lock = L.load_lock("m8_test", tmp_path / "locks")
    work = _master(tmp_path / "master",
                   {"KICKS/punchy/BD_01.wav": (f"{OLD_LIB}/Acme/Vqx/Qk 1.wav", 60, "m1")})
    out: list[str] = []
    plan = R.device_plan(work, _device(), lock, log=out.append)
    assert not plan["safe"] and plan["duplicates"] == 1 and plan["at_risk"] == 0
    assert any("DUPLICATE" in m for m in out)
