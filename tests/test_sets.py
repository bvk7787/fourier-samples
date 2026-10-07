"""Derived sets (packs/sets.py): slice metrics, kits, SLICE copies, verify."""
import pytest
import json
import os

import numpy as np
import soundfile as sf

from fourier.packs import sets as S
from fourier.packs import manifests

SR = 44100


def _clicks(times, dur, sr=SR):
    y = np.zeros(int(dur * sr))
    for t in times:
        i = int(t * sr)
        n = min(len(y) - i, 2000)
        y[i:i + n] += np.exp(-np.arange(n) / 200.0) * np.sin(np.arange(n) * 0.3)
    return y * 0.5


def _loop(path, bpm=120, bars=2, swing=50.0):
    six = 60.0 / bpm / 4
    n = bars * 16
    times = [(k + ((swing - 50) / 50.0 if k % 2 else 0)) * six for k in range(n)]
    sf.write(str(path), _clicks(times, n * six), SR)


def test_straight_loop_slices_clean(tmp_path):
    p = tmp_path / "a.wav"
    _loop(p)
    m = S.slice_metrics(p, 120)
    assert m["bars"] == 2
    assert m["slice_clean"] >= S.SLICE_READY
    assert m["swing"] is not None and abs(m["swing"] - 50) < 3


def test_onsets_over_a_ringing_ride_land_on_the_attacks():
    """Hits over a long ringing tail (a 909 ride loop) are found at their attacks, not
    25-35 ms early where the tail already passes 30% of the window's peak."""
    bpm, sr = 120, SR
    six = 60.0 / bpm / 4
    n = int(32 * six * sr)
    y = np.zeros(n)
    times = [k * 2 * six for k in range(16)]                         # 8ths
    for t in times:
        i = int(t * sr)
        m = n - i
        y[i:] += np.exp(-np.arange(m) / (0.4 * sr)) * np.sin(np.arange(m) * 0.9) * 0.4
    on = S._onsets(y, sr)
    dev = on[1:] - np.round(on[1:] / six) * six
    assert len(on) >= 12
    assert np.max(np.abs(dev)) < 0.002


def test_sparse_loop_is_not_slice_ready(tmp_path):
    """A crash loop with one hit a bar has every slice "clean" (empty) but nothing to chop:
    it stays out of 00_SLICE."""
    p = tmp_path / "crash.wav"
    six = 60.0 / 120 / 4
    sf.write(str(p), _clicks([0.0, 16 * six], 32 * six), SR)
    m = S.slice_metrics(p, 120)
    assert m["slice_clean"] == 1.0 and m["slice_hits"] < S.SLICE_MIN_HITS
    assert not S.slice_ready(dict(m, swing=None))
    q = tmp_path / "busy.wav"
    _loop(q)
    assert S.slice_ready(S.slice_metrics(q, 120))


def test_swung_loop_flagged(tmp_path):
    p = tmp_path / "b.wav"
    _loop(p, swing=64)
    m = S.slice_metrics(p, 120)
    assert m["swing"] >= S.SWING_FLAG
    assert m["slice_clean"] < 1.0


def test_no_tempo_no_metrics(tmp_path):
    p = tmp_path / "c.wav"
    _loop(p)
    assert S.slice_metrics(p, None) == {}


def _master(tmp_path):
    root = tmp_path / "M"
    cats = {}
    pack = "/x/SampleLibrary/Vendor/Pack909/"
    def add(cat, fam, name, band=None, **kw):
        d = root / cat / fam
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / name), np.random.default_rng(len(name)).normal(0, .1, 2000), SR)
        e = dict(family=fam, out=f"{fam}/{name}", src=pack + name, band=band,
                 out_md5=S._md5(d / name), **kw)
        cats.setdefault(cat, {"entries": []})["entries"].append(e)
    for r, (cat, band) in {"k": ("KICKS", None), "s": ("SNARES", None), "c": ("CLAPS", None),
                           "hc": ("HATS", "closed"), "ho": ("HATS", "open"), "cy": ("CYMBALS", None),
                           "t": ("TOMS", None), "p": ("PERC", None)}.items():
        for i in range(3):
            add(cat, "fam", f"909 {r}{i}.wav", band)
    d = root / "DRUMLOOPS" / "120bpm"
    d.mkdir(parents=True)
    _loop(d / "straight.wav"); _loop(d / "swung.wav", swing=64)
    cats["DRUMLOOPS"] = {"entries": [
        dict(family="120bpm", out=f"120bpm/{n}", src=pack + n, bpm=120, out_md5=S._md5(d / n),
             **S.slice_metrics(d / n, 120))
        for n in ("straight.wav", "swung.wav")]}
    (root / "manifest.json").write_text(json.dumps({"categories": cats}))
    return root


def test_build_sets(tmp_path):
    root = _master(tmp_path)
    s = S.build_sets(root, log=lambda *a: None)
    kits = {e["family"] for e in s["KITS"]["entries"]}
    assert len(kits) == 1
    kit = next(iter(kits))
    assert kit.startswith("kit-")
    assert len(s["KITS"]["entries"]) == 10          # 6 single roles + 2 toms + 2 percs
    assert [e["out"] for e in s["SLICE"]["entries"]] == ["120bpm/straight.wav"]
    assert (root / "SLICE" / "120bpm" / "straight.wav").read_bytes() == \
        (root / "DRUMLOOPS" / "120bpm" / "straight.wav").read_bytes()
    rows = (root / "loops.csv").read_text().splitlines()
    assert rows[0].startswith("folder,file,bpm") and len(rows) == 3
    assert "swung" in [r for r in rows if "swung.wav" in r][0]
    man = manifests.read(root / "manifest.json")
    assert "sets" in man and "KITS" not in man["categories"]
    # rebuilding replaces, not accumulates
    S.build_sets(root, log=lambda *a: None)
    assert len(list((root / "KITS").rglob("*.wav"))) == 10


def test_kit_needs_required_roles(tmp_path):
    root = _master(tmp_path)
    man = manifests.read(root / "manifest.json")
    man["categories"]["HATS"]["entries"] = [e for e in man["categories"]["HATS"]["entries"]
                                            if e["band"] != "open"]
    assert S._kits(man) == []


def test_verify_checks_sets(tmp_path):
    from fourier.packs.verify import _Ctx, check_manifest, check_sets, FAIL, PASS
    root = _master(tmp_path)
    S.build_sets(root, log=lambda *a: None)
    ctx = _Ctx(root, store_path=str(tmp_path / "none.json"))
    res = list(check_manifest(ctx)) + list(check_sets(ctx))
    assert all(r.level == PASS for r in res), [(r.check, r.detail) for r in res if r.level != PASS]
    # a copy that drifted from its source fails
    f = next((root / "KITS").rglob("*.wav"))
    f.unlink(); sf.write(str(f), np.zeros(100), SR)
    ctx = _Ctx(root, store_path=str(tmp_path / "none.json"))
    assert any(r.level == FAIL for r in check_sets(ctx))


def test_derived_dirs_render_first():
    from fourier.packs.curate_config import category_dir
    assert category_dir("KITS") == "00_KITS" and category_dir("SLICE") == "00_SLICE"
    assert category_dir("KICKS") == "01_KICKS"


def test_master_files_key_sets(tmp_path):
    from fourier.packs.device_lock import master_files
    root = _master(tmp_path)
    S.build_sets(root, log=lambda *a: None)
    fs = [f for f in master_files(root) if f.category == "KITS"]
    assert fs and all(f.key.startswith("KITS|/x/SampleLibrary/") for f in fs)


@pytest.mark.usefixtures("umbrella_vendor")
def test_kit_names_no_repeats():
    assert S.kit_name("Acme/909 Kit", "909") == "kit-909"
    assert S.kit_name("Acme/XR1 Kit", "drumbox") == "kit-drumbox-xr1"
    assert S.kit_name("Vendor B/PRESETS", None) == "kit-vendor-b"
    assert S.kit_name("Vendor C/Drum Machine Collection Volume 2", "909") == "kit-909-vendor-c"
    assert S.kit_name("Acme/Basic Drums", None) == "kit-basic-drums"


def test_machine_kits_first(monkeypatch):
    man = {"categories": {}}
    def add(cat, pack, name, band=None):
        man["categories"].setdefault(cat, {"entries": []})["entries"].append(
            dict(family="f", out=f"f/{name}", src=f"/x/SampleLibrary/{pack}/{name}", band=band))
    roles = [("KICKS", None), ("SNARES", None), ("HATS", "closed"), ("HATS", "open"), ("CLAPS", None), ("PERC", None)]
    for i in range(30):                    # 30 big generic kits
        for cat, band in roles + [("TOMS", None), ("CYMBALS", None)]:
            for k in range(3):
                add(cat, f"V{i}/Pack{i}", f"{cat}{band}{k}.wav", band)
    for cat, band in roles:                 # one small 909 kit
        add(cat, "V/Nine", f"909 {cat}{band}.wav", band)
    names = [n for n, _ in S._kits(man)]
    assert len(names) == S.KIT_COUNT and names[0].startswith("kit-909")


@pytest.mark.usefixtures("umbrella_vendor")
def test_kit_names_drop_machine_spellings():
    assert S.kit_name("Acme/Drumbux Kit", "drumbox") == "kit-drumbox"   # a spelling of the machine
    assert S.kit_name("Vendor D/Machine Pack", "606") == "kit-606-machine"


def test_late_loop_is_not_swung(tmp_path):
    p = tmp_path / "late.wav"
    six = 60.0 / 120 / 4
    times = [k * six + 0.025 for k in range(32)]
    sf.write(str(p), _clicks(times, 32 * six + 0.05), SR)
    assert abs(S.slice_metrics(p, 120)["swing"] - 50) < 3


def test_kit_slots_follow_names():
    C = lambda n: ("KICKS", dict(src=f"/x/{n}.wav", out=f"f/{n}.wav"))
    assert [os.path.basename(ce[1]["src"]) for ce in S._slot_candidates(
        "kick", [C("LT 5 Low Tune Decay"), C("Kick 909 Long")])] == ["Kick 909 Long.wav"]
    H = lambda n: ("HATS", dict(src=f"/x/{n}.wav", out=f"f/{n}.wav"))
    assert [os.path.basename(ce[1]["src"]) for ce in S._slot_candidates(
        "hat_closed", [H("Tamb 707 2"), H("Hat 707 Closed")])] == ["Hat 707 Closed.wav"]
    # nothing fits: any file of the role
    assert len(S._slot_candidates("clap", [C("Swipe")])) == 1


def test_additive_build_keeps_the_release_sets(tmp_path):
    root = _master(tmp_path)
    S.build_sets(root, log=lambda *a: None)
    man = manifests.read(root / "manifest.json")
    kits_before = sorted(e["out"] for e in man["sets"]["KITS"]["entries"])
    # an additive build: one more kick, one new loop
    man["base"] = "v1"
    d = root / "KICKS" / "fam"
    sf.write(str(d / "909 kZ.wav"), np.random.default_rng(9).normal(0, .1, 2000), SR)
    man["categories"]["KICKS"]["entries"].append(dict(family="fam", out="fam/909 kZ.wav",
                                                      src="/x/SampleLibrary/Vendor/Pack909/909 kZ.wav",
                                                      added_over="v1"))
    _loop(root / "DRUMLOOPS" / "120bpm" / "new.wav")
    man["categories"]["DRUMLOOPS"]["entries"].append(dict(
        family="120bpm", out="120bpm/new.wav", src="/x/new.wav", bpm=120, added_over="v1",
        **S.slice_metrics(root / "DRUMLOOPS" / "120bpm" / "new.wav", 120)))
    (root / "manifest.json").write_text(json.dumps(man))
    s = S.build_sets(root, log=lambda *a: None)
    assert sorted(e["out"] for e in s["KITS"]["entries"]) == kits_before
    assert sorted(e["out"] for e in s["SLICE"]["entries"]) == ["120bpm/new.wav", "120bpm/straight.wav"]


def test_slice_ready_needs_whole_short_bars():
    ok = dict(slice_clean=0.9, swing=50.0, bars=4, bar_err=0.02)
    assert S.slice_ready(ok)
    assert not S.slice_ready(dict(ok, bars=8))            # 128 slices: past the 64 grid
    assert not S.slice_ready(dict(ok, bars=0.25))         # a quarter-bar fill
    assert not S.slice_ready(dict(ok, bar_err=0.3))       # not whole bars
    assert not S.slice_ready(dict(ok, swing=60.0))


def test_kit_slots_turned_down_to_their_role_level(tmp_path):
    from fourier.packs.curate import _short_term_db
    from fourier.packs.verify import _Ctx, check_sets, FAIL, PASS
    root = _master(tmp_path)
    man = manifests.read(root / "manifest.json")
    loud = 0.89 * np.sin(np.arange(8000) * 0.02) * np.exp(-np.arange(8000) / 4000.0)  # a hot kick
    for e in man["categories"]["KICKS"]["entries"]:
        p = root / "KICKS" / e["out"]
        sf.write(str(p), loud, SR, subtype="PCM_16")
        e["out_md5"] = S._md5(p)
    (root / "manifest.json").write_text(json.dumps(man))
    s = S.build_sets(root, log=lambda *a: None)
    kick = next(e for e in s["KITS"]["entries"] if e["came_from"].startswith("KICKS/"))
    assert kick["gain_db"] < -1
    y, sr = sf.read(str(root / "KITS" / kick["out"]), always_2d=True)
    assert abs(_short_term_db(y, sr) - S.KIT_LEVEL_DB["kick"]) < 0.2
    assert sf.info(str(root / "KITS" / kick["out"])).subtype == "PCM_16"
    # quiet slots stay byte-identical copies
    hat = next(e for e in s["KITS"]["entries"] if e["came_from"].startswith("HATS/"))
    assert "gain_db" not in hat
    ctx = _Ctx(root, store_path=str(tmp_path / "none.json"))
    res = list(check_sets(ctx))
    assert all(r.level == PASS for r in res), [(r.check, r.detail) for r in res if r.level != PASS]
    # a kit file that doesn't match its recorded gain fails
    man = manifests.read(root / "manifest.json")
    for e in man["sets"]["KITS"]["entries"]:
        if e.get("gain_db"):
            e["gain_db"] += 2.0
    (root / "manifest.json").write_text(json.dumps(man))
    assert any(r.level == FAIL for r in check_sets(_Ctx(root, store_path=str(tmp_path / "none.json"))))


def test_wide_kit_slot_levelled_as_the_m8_plays_it(tmp_path):
    """A wide stereo hat is levelled by what the M8's level-keeping mono downmix plays, not
    by the plain L+R average, which reads it up to 3 dB quiet."""
    from fourier.devices.exporter import _mono_downmix
    from fourier.packs.curate import _short_term_db
    n = 4000
    rng = np.random.default_rng(1)
    env = np.exp(-np.arange(n) / 4000.0)
    l, r = (np.clip(rng.normal(0, 0.3, n), -0.99, 0.99) * env for _ in range(2))   # uncorrelated: wide
    y = np.stack([l, r], axis=1)
    assert S.kit_level_db(y, SR) - _short_term_db(y, SR) > 1.0
    assert _short_term_db(y, SR) < S.KIT_LEVEL_DB["hat_closed"] < S.kit_level_db(y, SR)   # the old read let it through
    src, dst = tmp_path / "hat.wav", tmp_path / "kit" / "hat.wav"
    sf.write(str(src), y, SR, subtype="PCM_24")
    g = S._write_kit_file(src, dst, "hat_closed")
    out, _ = sf.read(str(dst), always_2d=True)
    m8 = _mono_downmix(out.T, keep_level=True)
    assert g < 0 and _short_term_db(m8, SR) <= S.KIT_LEVEL_DB["hat_closed"] + 0.1


def test_acoustic_kit_names():
    assert S.kit_name("Vendor A/70s-room-kit", None) == "kit-70s-room"
    assert S.kit_name("Vendor B/STUDIO drum sample pack 2", None) == "kit-studio-drum-2"
    assert S.kit_name("Vendor B/XR1 Sample Pack", None) == "kit-xr1"
    assert S.kit_name("Vendor C/Studio Drums Collection 2 by Vendor D", None) == "kit-studio-drums-collection"


def test_loops_csv_bars_at_folder_tempo(tmp_path):
    """loops.csv doesn't double a bar count that was already taken at the folder tempo."""
    root = _master(tmp_path)
    man = manifests.read(root / "manifest.json")
    for e in man["categories"]["DRUMLOOPS"]["entries"]:
        e.update(bpm=60.0, bpm_fold=120.0, bpm_bars=120.0)
    (root / "manifest.json").write_text(json.dumps(man))
    S.build_sets(root, log=lambda *a: None)
    rows = (root / "loops.csv").read_text().splitlines()[1:]
    for r in rows:
        f = r.split(",")
        assert f[3] == f[5] or float(f[3]) == float(f[5])       # bars == bars_at_folder_bpm
