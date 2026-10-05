"""A render follows the profile fields it used to ignore (formats, 8-bit, max_duration_s,
max_slices, max_name_length, ascii_names, files_per_folder), and only when a profile sets
them: a profile that doesn't renders exactly as before."""
import json
import unicodedata

import numpy as np
import soundfile as sf

from fourier.devices.loader import DeviceLoader, DeviceProfile
from fourier.packs import render as R
from fourier.packs.device_lock import load_lock, record, save_lock

SR = 44100


def _dev(**kw):
    d = dict(sample_rate=44100, bit_depth=16, channels="stereo", folder_depth=2)
    d.update(kw)
    return DeviceProfile(device_id="zz_dev", name="Zz Device", **d)


def _tone(path, seconds=0.5, channels=2):
    t = np.arange(int(SR * seconds)) / SR
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(str(path), np.tile(y, (channels, 1)).T if channels == 2 else y, SR, subtype="PCM_24")


def _master(root, families):
    """families: {(CATEGORY, family): [(file stem, seconds)]}"""
    for (cat, fam), files in families.items():
        d = root / cat / fam
        d.mkdir(parents=True, exist_ok=True)
        for stem, secs in files:
            _tone(d / f"{stem}.wav", secs)
    return root


def _rendered(out):
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()
                  and not p.name.startswith("."))


def test_the_package_profiles_get_the_settings_they_always_had():
    """No new key in a package profile's conversion settings (the render cache keys stay
    put), the path lock format unchanged, no name rule and no slice filter."""
    for did in ("digitakt_2", "m8_tracker"):
        p = DeviceLoader(user_dirs=[]).load(did)
        for cat in ("KICKS", "WAVES"):
            assert R._convert_opts(p, cat) == dict(
                target_sr=p.sample_rate, target_bit_depth=p.bit_depth, convert_to_mono=p.mono,
                dither=p.dither, collapse_dual_mono=not p.mono, preserve_length=cat == "WAVES")
        assert R._render_fmt(p) == dict(sample_rate=p.sample_rate, bit_depth=p.bit_depth, mono=p.mono)
        assert not R._names_apply(p) and R._ext(p) == ".wav" and R.expected_subtype(p) == "PCM_16"
        assert R._rel_dest("KICKS", "f", "A b", 2, p) == R._rel_dest("KICKS", "f", "A b", 2)


def test_aiff_and_8_bit(tmp_path):
    master = _master(tmp_path / "m", {("KICKS", "tone"): [("Tone_A", 0.3), ("Tone_B", 0.31)]})
    out = tmp_path / "aif"
    dev = _dev(formats=["aiff"], bit_depth=8)
    R.render_master_to_device(master, dev, out, log=lambda m: None)
    files = sorted(out.rglob("*.aif"))
    assert [f.name for f in files] == ["Tone_A.aif", "Tone_B.aif"] and not list(out.rglob("*.wav"))
    for f in files:
        info = sf.info(str(f))
        assert (info.format, info.subtype, info.samplerate) == ("AIFF", "PCM_S8", 44100)
        assert info.subtype == R.expected_subtype(dev)
        assert 0.3 < float(np.max(np.abs(sf.read(str(f))[0]))) < 0.6
    out8 = tmp_path / "wav8"
    R.render_master_to_device(master, _dev(bit_depth=8), out8, log=lambda m: None)
    assert {sf.info(str(f)).subtype for f in out8.rglob("*.wav")} == {"PCM_U8"}
    out24 = tmp_path / "aif24"
    R.render_master_to_device(master, _dev(formats=["aif"], bit_depth=24), out24, log=lambda m: None)
    a = sf.read(str(next(out24.rglob("Tone_A.aif"))), dtype="int32")[0]
    w = tmp_path / "w24"
    R.render_master_to_device(master, _dev(bit_depth=24), w, log=lambda m: None)
    assert np.array_equal(a, sf.read(str(next(w.rglob("Tone_A.wav"))), dtype="int32")[0])  # same samples


def test_max_duration_cuts_with_a_fade_but_never_a_cycle(tmp_path):
    master = _master(tmp_path / "m", {("PADS", "long"): [("Long_A", 3.0), ("Short_A", 0.5)],
                                      ("WAVES", "cycle"): [("Cycle_A", 2.0)]})
    out = tmp_path / "o"
    R.render_master_to_device(master, _dev(sample_rate=48000, max_duration_s=1.25), out, log=lambda m: None)
    long_ = sf.read(str(next(out.rglob("Long_A.wav"))))[0]
    assert len(long_) == int(1.25 * 48000)
    assert np.max(np.abs(long_[-3:])) < 0.01                       # faded, no click
    assert sf.info(str(next(out.rglob("Short_A.wav")))).frames == 24000
    assert sf.info(str(next(out.rglob("Cycle_A.wav")))).frames == int(2.0 * SR)   # relabelled, whole


def test_max_name_length_cuts_names_and_keeps_them_apart(tmp_path):
    master = _master(tmp_path / "m", {("KICKS", "a-very-long-family-name"): [
        ("Tone_Alpha_Long_Name_One", 0.1), ("Tone_Alpha_Long_Name_One_B", 0.1), ("Tone_B", 0.1)]})
    out = tmp_path / "o"
    R.render_master_to_device(master, _dev(max_name_length=16), out, log=lambda m: None)
    got = _rendered(out)
    assert len(got) == 3 and len(set(p.lower() for p in got)) == 3
    for p in got:
        assert all(len(part) <= 16 for part in p.split("/")), p
    assert any(p.endswith("/Tone_B.wav") for p in got)


def test_ascii_names(tmp_path):
    nfd = unicodedata.normalize("NFD", "Café_Straße_Å")
    master = _master(tmp_path / "m", {("PADS", "família"): [(nfd, 0.1)]})
    out = tmp_path / "ascii"
    R.render_master_to_device(master, _dev(ascii_names=True), out, log=lambda m: None)
    assert [p for p in _rendered(out)] == ["13_PADS/familia/Cafe_Strasse_A.wav"]
    out = tmp_path / "nfc"
    R.render_master_to_device(master, _dev(ascii_names=False), out, log=lambda m: None)
    [p] = [p for p in out.rglob("*.wav")]
    assert p.name == unicodedata.normalize("NFC", nfd) + ".wav" and p.name != nfd + ".wav"
    out = tmp_path / "unset"
    R.render_master_to_device(master, _dev(), out, log=lambda m: None)
    assert [p.name for p in out.rglob("*.wav")] == [nfd + ".wav"]           # as the master has it
    assert R._ascii("æøß_中_x") == "aeoss_x"


def test_files_per_folder_splits_and_a_lock_keeps_its_paths(tmp_path, monkeypatch):
    fam = ("KICKS", "tone")
    master = _master(tmp_path / "m", {fam: [(f"Tone_{c}", 0.05) for c in "BCDEF"]})
    dev = _dev(files_per_folder=2)
    files, a = R._assignment(master, dev)
    assert sorted(a.paths.values()) == [
        "01_KICKS/tone-2/Tone_D.wav", "01_KICKS/tone-2/Tone_E.wav", "01_KICKS/tone-3/Tone_F.wav",
        "01_KICKS/tone/Tone_B.wav", "01_KICKS/tone/Tone_C.wav"]
    assert R._assignment(master, _dev(files_per_folder=5))[1].paths == \
        R._assignment(master, _dev(files_per_folder=None))[1].paths          # within the limit: as before
    # a release locks those paths; a new file that sorts first fills the room left (tone-3)
    lock_dir = tmp_path / "locks"
    save_lock(record(None, dev.device_id, "", files, a, "v1", fmt=R._render_fmt(dev)), lock_dir)
    _tone(master / "KICKS" / "tone" / "Tone_A.wav", 0.05)
    _files, b = R._assignment(master, dev, load_lock(dev.device_id, lock_dir))
    assert {k: b.paths[k] for k in a.paths} == a.paths
    [new] = b.new
    assert b.paths[new] == "01_KICKS/tone-3/Tone_A.wav"
    out = tmp_path / "o"
    R.render_master_to_device(master, dev, out, log=lambda m: None, lock=load_lock(dev.device_id, lock_dir))
    assert max(len(list(d.glob("*.wav"))) for d in (out / "01_KICKS").iterdir()) == 2


def _slice_master(root, bars):
    m = _master(root, {("DRUMLOOPS", "loops"): [("Loop_A", 0.2)], ("SLICE", "loops"): [("Loop_A", 0.2)]})
    (root / "manifest.json").write_text(json.dumps({
        "categories": {"DRUMLOOPS": {"entries": [{"family": "loops", "out": "loops/Loop_A.wav",
                                                  "src": "/lib/SampleLibrary/X/Loop A.wav", "bars": bars}]}},
        "sets": {"SLICE": {"entries": [{"family": "loops", "out": "loops/Loop_A.wav",
                                        "src": "/lib/SampleLibrary/X/Loop A.wav",
                                        "came_from": "DRUMLOOPS/loops/Loop_A.wav"}]}}}))
    return m


def test_max_slices_leaves_loops_with_a_finer_grid_out_of_the_slice_set(tmp_path):
    m = _slice_master(tmp_path / "m", bars=2.0)          # 32 sixteenths
    def slice_files(dev):
        return [v for v in R._assignment(m, dev)[1].paths.values() if v.startswith("00_SLICE")]
    assert slice_files(_dev(max_slices=64)) == slice_files(_dev()) != []
    assert slice_files(_dev(max_slices=32)) != []
    assert slice_files(_dev(max_slices=16)) == []
    assert R._slice_overflow(m, _dev(max_slices=16)) == {"SLICE/loops/Loop_A.wav"}
    assert any(v.startswith("08_DRUMLOOPS") for v in R._assignment(m, _dev(max_slices=16))[1].paths.values())
