"""Tests for the device render step (fourier.packs.render)."""

import numpy as np
import soundfile as sf

from fourier.devices.loader import DeviceProfile
from fourier.packs import render as R


def _fake_device(**kw):
    d = dict(sample_rate=48000, bit_depth=16, channels="mono", folder_depth=2, root="SAMPLES")
    d.update(kw)
    return DeviceProfile(device_id="test_dev", name="Test Device", **d)


def test_device_root_from_profile():
    assert R._device_root(_fake_device()) == "SAMPLES"
    assert R._device_root(_fake_device(root="/SAMPLES/")) == "SAMPLES"


def _write_master(tmp_path):
    """A tiny master: one category, one family, two stereo 44.1k wavs."""
    fam = tmp_path / "KICKS" / "909-house-dark"
    fam.mkdir(parents=True)
    sr = 44100
    for name in ("BD_one.wav", "BD_two.wav"):
        stereo = np.tile(np.sin(np.linspace(0, 6.28, sr // 2)).astype("float32"), (2, 1)).T
        sf.write(str(fam / name), stereo, sr, subtype="PCM_24")
    (tmp_path / "_manifest_dir").mkdir()  # an underscore dir the render must ignore
    return tmp_path


def test_render_converts_format_and_layout(tmp_path):
    master = _write_master(tmp_path / "master")
    out = tmp_path / "out"
    dev = _fake_device()
    summary = R.render_master_to_device(str(master), dev, str(out), log=lambda m: None)

    assert summary["files"] == 2
    base = out / "SAMPLES"
    # depth-2 device: family folded into the filename under SAMPLES/<CATEGORY>/
    rendered = sorted((base / "01_KICKS").glob("*.wav"))   # numbered in play order
    assert len(rendered) == 2
    assert rendered[0].name.startswith("909-house-dark__")
    # underscore-prefixed dirs are skipped, so no stray folders
    assert not (base / "_manifest_dir").exists()

    info = sf.info(str(rendered[0]))
    assert info.samplerate == 48000
    assert info.channels == 1
    assert info.subtype == "PCM_16"


def test_render_deeper_layout_keeps_family_folder(tmp_path):
    master = _write_master(tmp_path / "master")
    out = tmp_path / "out"
    dev = _fake_device(folder_depth=3)  # room for CATEGORY/family both as folders
    R.render_master_to_device(str(master), dev, str(out), log=lambda m: None)
    assert (out / "SAMPLES" / "01_KICKS" / "909-house-dark").is_dir()


def test_card_paths_match_render_layout_and_flag_long_paths(tmp_path):
    master = _write_master(tmp_path / "master")
    dev = _fake_device(folder_depth=3, max_path_length=40)
    dev.card_dir = "/Samples/Fourier/"
    paths = R.card_paths(master, dev)
    assert paths == ["/Samples/Fourier/SAMPLES/01_KICKS/909-house-dark/BD_one.wav",
                     "/Samples/Fourier/SAMPLES/01_KICKS/909-house-dark/BD_two.wav"]
    bad = R.path_violations(master, dev)
    assert [n for _, n in bad] == [len(paths[0])] * 2
    dev.max_path_length = 128
    assert R.path_violations(master, dev) == []
    # rendered files land where card_paths says (minus the card_dir)
    out = tmp_path / "out"
    R.render_master_to_device(master, dev, out, log=lambda m: None)
    rel = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav"))
    assert ["/Samples/Fourier/" + r for r in rel] == paths


def test_render_keeps_master_names_within_the_limit_and_cuts_only_what_is_over(tmp_path):
    """A path within the device's limit keeps the master's names (one name per file on every
    device the master was sized for); a device it wasn't sized for gets its family folder and
    file name cut in the middle until the path fits, and a path that can't fit stops the
    render, which writes nothing."""
    import pytest
    master = _write_master(tmp_path / "master")
    dev = _fake_device(folder_depth=3, max_path_length=61)
    dev.card_dir = "/Samples/Fourier"
    # /Samples/Fourier/SAMPLES/01_KICKS/909-house-dark/BD_one.wav is 59 chars: fits, unchanged
    assert R.path_violations(master, dev) == []
    assert R.card_paths(master, dev)[0].endswith("/909-house-dark/BD_one.wav")
    dev.max_path_length = 53                  # 6 over: the family folder is cut
    paths = R.card_paths(master, dev)
    assert R.path_violations(master, dev) == [] and all(len(p) <= 53 for p in paths)
    assert all("/01_KICKS/909~ark/BD_" in p for p in paths), paths
    long = tmp_path / "long"
    fam = long / "KICKS" / "909-house-dark"
    fam.mkdir(parents=True)
    name = "BD_" + "x" * 40
    sf.write(str(fam / (name + ".wav")), np.zeros(4410, "float32"), 44100)
    dev.max_path_length = 70
    (p,) = R.card_paths(long, dev)
    assert 64 <= len(p) <= 70 and "~" in p and p.endswith(".wav") and not R.path_violations(long, dev)
    dev.max_path_length = 50                  # can't fit even cut: reported, and not rendered
    assert len(R.path_violations(long, dev)) == 1
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="would exceed its 50-character limit"):
        R.render_master_to_device(long, dev, out, log=lambda m: None)
    assert not list(out.rglob("*.wav"))


def test_a_device_the_master_wasnt_sized_for_gets_one_name_per_folder(tmp_path):
    """Files of one family folder whose paths a device's limit cuts by different amounts land
    in one folder (its name cut once, sized from the longest path), each file cut only as much
    as it still needs; `render --check` (card_paths) says what render writes, and both say
    the names were cut."""
    master = tmp_path / "master"
    fam = master / "PADS" / "chord-bright-grand-piano-noisy"
    fam.mkdir(parents=True)
    for name in ("Pad 01.wav", "Pad Bright Grand Long Take 02.wav", "Pad Bright Grand Longer Take Number 03.wav"):
        sf.write(str(fam / name), np.zeros(4410, "float32"), 44100)
    dev = _fake_device(folder_depth=3, max_path_length=70)
    dev.card_dir = "/Samples"
    files, a = R._assignment(master, dev)
    folders = {p.rsplit("/", 1)[0] for p in a.paths.values()}
    assert len(folders) == 1, folders                       # one folder, not one per cut
    paths = R.card_paths(master, dev)
    assert all(len(p) <= 70 for p in paths) and R.path_violations(master, dev) == []
    assert any(p.endswith("/Pad_01.wav") for p in paths)    # a short name keeps itself
    folders_cut, names_cut = a.shortened
    assert folders_cut == 1 and names_cut >= 1
    note = R.shortened_note(dev, a.shortened)
    assert "1 family folder" in note and "devices in fourier.toml" in note
    logs = []
    out = tmp_path / "out"
    R.render_master_to_device(master, dev, out, log=logs.append)
    rel = sorted("/Samples/" + p.relative_to(out).as_posix() for p in out.rglob("*.wav"))
    assert rel == paths                                     # check predicts render exactly
    assert any("cut to fit test_dev's 70-character path limit" in m for m in logs)
    # within the limit: nothing cut, nothing said
    dev.max_path_length = 200
    _f, a = R._assignment(master, dev)
    assert a.shortened == (0, 0) and R.shortened_note(dev, a.shortened) is None


def test_names_match_across_devices():
    from fourier.devices.exporter import canonical_stem
    from fourier.packs.curate_config import STEM_MAX, FAMILY_NAME_MAX
    st = canonical_stem("Acme Kick 909 Tune Max Decay Min Attack Max Take 12", STEM_MAX)
    assert len(st) <= STEM_MAX and canonical_stem(st, STEM_MAX) == st          # idempotent
    fam = "f" * FAMILY_NAME_MAX
    deep = R._rel_dest("DRUMLOOPS", fam, st, 2).as_posix()
    flat = R._rel_dest("DRUMLOOPS", fam, st, 1).as_posix()
    assert deep.endswith("/" + st + ".wav") and flat.endswith("__" + st + ".wav")
    # the budget: the M8's card prefix + the longest category + the longest family + a "_2"
    assert len("/Samples/Fourier/" + deep) + 2 <= 127


def test_depth_limited_render_folds_family_into_name(tmp_path):
    master = _write_master(tmp_path / "master")
    dev = _fake_device(folder_depth=2, max_path_length=None)
    assert R.card_paths(master, dev)[0] == "SAMPLES/01_KICKS/909-house-dark__BD_one.wav"
    assert R.path_violations(master, dev) == []


def test_stereo_device_keeps_stereo_and_collapses_dual_mono(tmp_path):
    """Digitakt 2 keeps each file's channels; a stereo file with identical channels
    carries no width and is written mono."""
    from fourier.devices.exporter import _convert_and_copy
    t = np.arange(4410) / 44100
    left = 0.3 * np.sin(2 * np.pi * 220 * t)
    wide, dual = tmp_path / "wide.wav", tmp_path / "dual.wav"
    sf.write(str(wide), np.stack([left, 0.5 * left], 1).astype("float32"), 44100, subtype="PCM_24")
    sf.write(str(dual), np.stack([left, left], 1).astype("float32"), 44100, subtype="PCM_24")
    for src in (wide, dual):
        _convert_and_copy(src, tmp_path / f"o_{src.name}", 48000, 16, convert_to_mono=False,
                          dither=True, collapse_dual_mono=True)
    assert sf.info(str(tmp_path / "o_wide.wav")).channels == 2
    assert sf.info(str(tmp_path / "o_dual.wav")).channels == 1


def test_digitakt_profile_keeps_stereo():
    from fourier.devices.loader import DeviceLoader
    from fourier.packs.render import _render_fmt
    assert _render_fmt(DeviceLoader().load("digitakt_2"))["mono"] is False


def test_card_sync_copies_clean(tmp_path, monkeypatch):
    import os
    import shutil
    import pytest
    from click.testing import CliRunner
    from fourier.cli import main
    if not shutil.which("rsync"):
        pytest.skip("rsync not installed")
    import fourier.platforms as plat
    monkeypatch.setattr(plat, "mount_points", lambda: [tmp_path / "CARD"])     # the card, mounted
    render = tmp_path / "FourierRenders" / "m8_tracker"
    (render / "01_KICKS" / "punchy").mkdir(parents=True)
    (render / "01_KICKS" / "punchy" / "a.wav").write_bytes(b"RIFFxxxxWAVE")
    (render / "01_KICKS" / "punchy" / "._a.wav").write_bytes(b"junk")
    (render / ".fourier-render").write_text("m8_tracker")
    card = tmp_path / "CARD"
    (card / "Samples" / "Fourier" / "01_KICKS").mkdir(parents=True)
    (card / "Samples" / "Fourier" / "01_KICKS" / "._old.wav").write_bytes(b"junk")
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "lib.db"), "sync", "m8_tracker",
                                  str(card), "--from", str(render)])
    assert r.exit_code == 0, r.output
    dest = card / "Samples" / "Fourier"
    names = {os.path.relpath(os.path.join(dp, f), dest) for dp, _, fs in os.walk(dest) for f in fs}
    assert names == {"01_KICKS/punchy/a.wav", ".fourier-card"}   # the card id marker
    # a path that now holds a different sample (same size, newer render) is replaced on the card
    import time
    time.sleep(1.1)
    (render / "01_KICKS" / "punchy" / "a.wav").unlink()
    (render / "01_KICKS" / "punchy" / "a.wav").write_bytes(b"RIFFyyyyWAVE")
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "lib.db"), "sync", "m8_tracker",
                                  str(card), "--from", str(render)])
    assert r.exit_code == 0, r.output
    assert (dest / "01_KICKS" / "punchy" / "a.wav").read_bytes() == b"RIFFyyyyWAVE"


def test_render_cache_reuses_identical_conversions(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_RENDER_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("FOURIER_NO_RENDER_CACHE", raising=False)
    master = _write_master(tmp_path / "m")
    dev = _fake_device()
    logs = []
    R.render_master_to_device(str(master), dev, str(tmp_path / "o1"), log=logs.append)
    first = {p.relative_to(tmp_path / "o1"): p.read_bytes() for p in (tmp_path / "o1").rglob("*.wav")}
    calls = []
    real = R._convert_and_copy
    monkeypatch.setattr(R, "_convert_and_copy", lambda *a, **k: (calls.append(a), real(*a, **k)))
    R.render_master_to_device(str(master), dev, str(tmp_path / "o2"), log=logs.append)
    second = {p.relative_to(tmp_path / "o2"): p.read_bytes() for p in (tmp_path / "o2").rglob("*.wav")}
    assert first == second and len(first) == 2 and calls == []
    assert any("2 reused, 0 converted" in m for m in logs)
    # a changed source converts again
    f = next((master / "KICKS").rglob("BD_one.wav"))
    sf.write(str(f), np.zeros((1000, 2), "float32"), 44100, subtype="PCM_24")
    R.render_master_to_device(str(master), dev, str(tmp_path / "o3"), log=logs.append)
    assert len(calls) == 1


def test_render_pool_matches_in_process(tmp_path, monkeypatch):
    master = _write_master(tmp_path / "m")
    dev = _fake_device()
    R.render_master_to_device(str(master), dev, str(tmp_path / "a"), log=lambda m: None)
    monkeypatch.setattr(R, "RENDER_POOL_MIN", 1)
    monkeypatch.setattr(R, "RENDER_JOBS", 2)
    R.render_master_to_device(str(master), dev, str(tmp_path / "b"), log=lambda m: None)
    a = {p.relative_to(tmp_path / "a"): p.read_bytes() for p in (tmp_path / "a").rglob("*.wav")}
    b = {p.relative_to(tmp_path / "b"): p.read_bytes() for p in (tmp_path / "b").rglob("*.wav")}
    assert a == b and len(a) == 2


def test_resampling_keeps_faded_edges_quiet(tmp_path):
    """A master file faded in from zero stays quiet at its first sample after the 48 kHz
    resample (the resampling filter can lift a faded start off zero)."""
    import numpy as np
    import soundfile as sf
    from fourier.devices.exporter import _convert_and_copy
    sr = 44100
    t = np.arange(sr // 4) / sr
    y = 0.9 * np.cos(2 * np.pi * 3000 * t) * np.exp(-10 * t)
    y[0] = 0.0                                  # quiet first sample, the hit right after it
    y[-1] = 0.0
    src = tmp_path / "hit.wav"
    sf.write(str(src), y, sr, subtype="PCM_16")
    out = tmp_path / "o.wav"
    _convert_and_copy(src, out, 48000, 16, convert_to_mono=False, dither=False)
    z, _ = sf.read(str(out))
    assert abs(z[0]) <= 0.01 * np.abs(z).max() and abs(z[-1]) <= 0.01 * np.abs(z).max() + 1e-4
