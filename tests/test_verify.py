"""`fourier verify` (packs/verify.py): a clean master passes, and each rule catches
the violation it exists for."""
import hashlib
import json
import os

import numpy as np
import soundfile as sf

from fourier.packs.verify import FAIL, verify_master

SR = 44100
LIB = "/lib/SampleLibrary"


def _hit(seconds=0.4, dc=0.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.8 * np.exp(-12 * t) * np.sin(2 * np.pi * 60 * t) + dc).astype("float32")


def _master(tmp_path, files, ratings=None):
    """files: [(category, family, filename, audio, src_rel)] -> (master_dir, store_path)."""
    root = tmp_path / "master"
    cats, fams = {}, {}
    for cat, fam, fn, y, src_rel in files:
        d = root / cat / fam
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / fn, y, SR, subtype="PCM_24")
        md5 = hashlib.md5((d / fn).read_bytes()).hexdigest()
        cats.setdefault(cat, {"families": 0, "files": 0, "source_samples": 0, "entries": []})
        cats[cat]["entries"].append({"family": fam, "out": f"{fam}/{fn}", "src": f"{LIB}/{src_rel}",
                                     "out_md5": md5})
        fams.setdefault(cat, {}).setdefault(fam, 0)
        fams[cat][fam] += 1
    for cat, fl in fams.items():
        cats[cat]["families"] = len(fl)
        cats[cat]["files"] = len(cats[cat]["entries"])
        (root / cat / "_manifest.json").write_text(json.dumps(
            [{"family": f, "copied": n, "n": n * 3, "clap": None, "traits": []} for f, n in fl.items()]))
    (root / "manifest.json").write_text(json.dumps({"categories": cats}))
    store = tmp_path / "ratings.json"
    store.write_text(json.dumps({"version": 1, "ratings": ratings or {}}))
    return str(root), str(store)


def _clean():
    return [
        ("KICKS", "punchy-sub-heavy", "k1.wav", _hit(), "V1/Pack/k1.wav"),
        ("KICKS", "punchy-sub-heavy", "k2.wav", _hit(0.3), "V2/Pack/k2.wav"),
        ("PADS", "warm-analog-dark", "p1.wav", _hit(0.5), "V3/Pack/p1.wav"),   # decays to -50 dB by its end
    ]


def _fails(tmp_path, files, ratings=None):
    root, store = _master(tmp_path, files, ratings)
    ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    return ok, {r.check for r in res if r.level == FAIL}


def test_clean_master_passes(tmp_path):
    ok, fails = _fails(tmp_path, _clean())
    assert ok and not fails, fails


def test_drop_in_master_fails(tmp_path):
    ok, fails = _fails(tmp_path, _clean(), {f"{LIB}/V1/Pack/k1.wav": {"category": "KICKS", "verdict": "drop"}})
    assert not ok and "no Drop in the master" in fails


def test_lost_keep_and_misfile_back_fail(tmp_path):
    here = tmp_path / "lib" / "Kept.wav"            # a Keep whose source is there, not in the master
    here.parent.mkdir()
    sf.write(here, _hit(), SR)
    r = {str(here): {"category": "KICKS", "verdict": "keep"},
         f"{LIB}/V3/Pack/p1.wav": {"category": "PADS", "verdict": "misfiled"}}
    ok, fails = _fails(tmp_path, _clean(), r)
    assert "no Misfiled file back in its folder" in fails
    assert any(f.startswith("every Keep") for f in fails)


def test_a_keep_whose_source_is_gone_warns_and_passes(tmp_path):
    """A Keep whose file is gone (moved or deleted) can't be placed: verify warns, as the
    build does, rather than failing every build until the rating is cleared."""
    root, store = _master(tmp_path, _clean(), {f"{LIB}/Gone/x.wav": {"category": "KICKS", "verdict": "keep"}})
    ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    assert ok, [r.check for r in res if r.level == FAIL]
    warn = [r for r in res if r.check.startswith("Keeps whose source is gone")]
    assert warn and warn[0].level == "WARN" and "x.wav" in warn[0].detail


def test_length_cap_and_bpm_phrase_fail_unless_kept(tmp_path):
    files = _clean() + [("PADS", "warm-analog-dark", "long.wav", _hit(22.0), "V4/Pack/long.wav"),
                        ("PADS", "warm-analog-dark", "Rhodes 120bpm.wav", _hit(1.0), "V5/Pack/x 120bpm.wav")]
    ok, fails = _fails(tmp_path, files)
    assert {"length caps hold", "no bpm-named phrases in one-shot folders"} <= fails
    keep = {f"{LIB}/V4/Pack/long.wav": {"category": "PADS", "verdict": "keep"},
            f"{LIB}/V5/Pack/x 120bpm.wav": {"category": "PADS", "verdict": "keep"}}
    ok, fails = _fails(tmp_path / "k", files, keep)
    assert not ({"length caps hold", "no bpm-named phrases in one-shot folders"} & fails)


def test_naming_rules(tmp_path):
    files = [("KICKS", "punchy-long", "k1.wav", _hit(), "V1/Pack/k1.wav"),
             ("PADS", "c3-gated", "p1.wav", _hit(2.0), "V3/Pack/p1.wav"),
             ("SYNTH", "c4-grand-piano-note-short", "s.wav", _hit(), "V2/Pack/s.wav")]
    ok, fails = _fails(tmp_path, files)
    assert {"traits added whole (no half traits)", "tonal folders use tonal words",
            "no keys names in SYNTH/SUB/FX"} <= fails


def test_manifest_disk_and_md5(tmp_path):
    root, store = _master(tmp_path, _clean())
    sf.write(os.path.join(root, "KICKS", "punchy-sub-heavy", "k1.wav"), _hit(0.5), SR, subtype="PCM_24")
    sf.write(os.path.join(root, "KICKS", "punchy-sub-heavy", "stray.wav"), _hit(), SR)
    ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    fails = {r.check for r in res if r.level == FAIL}
    assert {"manifest matches disk", "out_md5 matches every file"} <= fails


def test_identical_audio_in_two_categories(tmp_path):
    y = _hit()
    files = _clean() + [("PERC", "cajon-hit-soft", "a.wav", y, "V7/Pack/a.wav"),
                        ("SNARES", "fat-soft", "b.wav", y, "V8/Pack/b.wav")]
    ok, fails = _fails(tmp_path, files)
    assert "no identical audio in two categories" in fails


def test_dc_and_waves(tmp_path):
    lots_dc = [("PERC", "cajon-hit-soft", f"d{i}.wav", _hit(0.3 + i / 100, dc=0.2), f"V9/Pack/d{i}.wav")
               for i in range(12)]
    t = np.arange(2048) / 2048
    wave = (0.8 * np.sin(2 * np.pi * t) + 0.1).astype("float32")
    src = tmp_path / "lib" / "WAVETABLES" / "Synth A" / "SAW.wav"
    src.parent.mkdir(parents=True)
    sf.write(src, np.tile(wave, 2), SR)                   # source has 4096 samples, export 2048
    files = _clean() + lots_dc + [("WAVES", "cycle-saw-bright", "SAW.wav", wave, "WAVETABLES/Synth A/SAW.wav")]
    root, store = _master(tmp_path, files)
    m = json.loads(open(os.path.join(root, "manifest.json")).read())
    for e in m["categories"]["WAVES"]["entries"]:
        e["src"] = str(src)
    open(os.path.join(root, "manifest.json"), "w").write(json.dumps(m))
    ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    fails = {r.check for r in res if r.level == FAIL}
    assert {"DC offset under 2% of peak", "WAVES carry no DC (loop cleanly)",
            "WAVES exported sample-exact"} <= fails


def _render_master(tmp_path):
    t = np.arange(int(SR * 0.3)) / SR
    s = (0.7 * np.exp(-6 * t) * np.sin(2 * np.pi * 300 * t)).astype("float32")
    anti = np.stack([s, -0.97 * s], axis=1)
    cyc = (0.8 * np.sin(2 * np.pi * np.arange(2048) / 2048)).astype("float32")
    return _master(tmp_path, _clean() + [
        ("CLAPS", "gated-snappy", "Clap Flip.wav", anti, "V1/Pack/Clap Flip.wav"),
        ("WAVES", "cycle-saw-bright", "SAW-C3.wav", cyc, "Vendor A/WAVETABLES/Synth A/SAW-C3.wav"),
    ])


def test_render_verify_m8_and_digitakt(tmp_path):
    from fourier.packs.verify import render_and_verify
    root, _ = _render_master(tmp_path)
    ok, res = render_and_verify(root, ["m8_tracker", "digitakt_2"], log=lambda *a: None)
    assert ok, [(r.check, r.detail) for r in res if r.level == FAIL]
    checks = {r.check for r in res}
    assert "digitakt_2: WAVES keep every sample" in checks
    # the M8 keeps stereo like the Digitakt
    assert "m8_tracker: phase-inverted stereo keeps its level in mono" not in checks


def test_m8_render_keeps_stereo(tmp_path):
    from fourier.devices.loader import DeviceLoader
    from fourier.packs.render import render_master_to_device
    root, _ = _render_master(tmp_path)
    out = tmp_path / "r"
    render_master_to_device(root, DeviceLoader().load("m8_tracker"), out, log=lambda *a: None, lock=None)
    f = next(out.rglob("*Flip*"))
    assert sf.info(str(f)).channels == 2 and sf.info(str(f)).samplerate == 44100


def test_render_verify_catches_resampled_waves_and_mono_cancel(tmp_path, monkeypatch):
    from fourier.packs import render as R
    from fourier.devices import exporter as E
    from fourier.packs.verify import render_and_verify
    from fourier.devices.loader import DeviceLoader
    root, _ = _render_master(tmp_path)
    _load = DeviceLoader.load

    def mono_m8(self, did):                    # a mono render (the M8's old behaviour)
        d = _load(self, did)
        if did == "m8_tracker":
            d.channels = "mono"
        return d
    monkeypatch.setattr(DeviceLoader, "load", mono_m8)
    monkeypatch.setattr(R, "CYCLE_CATEGORIES", set())                      # waves resampled
    monkeypatch.setattr(E, "_mono_downmix", lambda a, keep_level=False: a.mean(axis=0) if a.ndim == 2 else a)
    ok, res = render_and_verify(root, ["m8_tracker", "digitakt_2"], log=lambda *a: None)
    fails = {r.check for r in res if r.level == FAIL}
    assert not ok
    assert "m8_tracker: phase-inverted stereo keeps its level in mono" in fails
    assert "digitakt_2: WAVES keep every sample" in fails                  # 48 kHz resample


def test_publish_refuses_a_master_that_fails_verify(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from fourier.cli import main
    monkeypatch.setenv("HOME", str(tmp_path))                     # scratch DB + ratings store
    root, _ = _master(tmp_path, _clean())
    sf.write(os.path.join(root, "KICKS", "punchy-sub-heavy", "stray.wav"), _hit(), SR)
    dest = tmp_path / "publish"
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "lib.db"), "publish",
                                  "--from", root, "--to", str(dest)])
    assert r.exit_code == 1, r.output
    assert "not publishing" in r.output
    assert not (dest / "releases" / "v1").exists()


def test_piano_chord_and_note_folders_kept_apart(tmp_path):
    good = _clean() + [("PIANO", "chord-rhodes-clean", "Keys Chord Ebmaj9.wav", _hit(), "V1/P/Keys Chord Ebmaj9.wav"),
                       ("PIANO", "c4-grand-piano-note-airy", "Grand C4.wav", _hit(), "V1/P/Grand C4.wav")]
    ok, fails = _fails(tmp_path, good)
    assert "PIANO chord and note folders kept apart" not in fails
    bad = _clean() + [("PIANO", "c4-grand-piano-note-airy", "EP Chord Am7.wav", _hit(), "V1/P/EP Chord Am7.wav")]
    ok, fails = _fails(tmp_path / "b", bad)
    assert "PIANO chord and note folders kept apart" in fails


def test_verify_flags_contradicting_and_twin_names():
    from types import SimpleNamespace
    from fourier.packs.verify import check_naming, FAIL, WARN
    ctx = SimpleNamespace(fams={"PADS": [{"family": "gritty-bright-dark", "copied": 20},
                                         {"family": "short-plucky", "copied": 20},
                                         {"family": "short-plucky-dark", "copied": 20}]}, entries=[], cats={}, pins={})
    res = {r.check: r.level for r in check_naming(ctx) if "trait" in r.check or "sibling" in r.check}
    assert res["no folder name holds both sides of a trait"] == FAIL
    assert res["sibling folder names differ by more than order or one missing word"] == WARN


def test_verify_twin_names_keep_number_words():
    from types import SimpleNamespace
    from fourier.packs.verify import check_naming, PASS
    ctx = SimpleNamespace(fams={"SUB": [{"family": "303-plucky-short", "copied": 20},
                                        {"family": "dark-plucky-short", "copied": 20}]},
                          entries=[], cats={}, pins={})
    res = {r.check: r.level for r in check_naming(ctx) if "sibling" in r.check}
    assert res["sibling folder names differ by more than order or one missing word"] == PASS


def test_verify_flags_hard_starts_and_tiny_cycles(tmp_path):
    t = np.arange(int(SR * 0.3)) / SR
    hard = (0.9 * np.cos(2 * np.pi * 60 * t) * np.exp(-12 * t)).astype("float32")
    tiny = (0.8 * np.sin(2 * np.pi * np.arange(64) / 64)).astype("float32")
    ok, fails = _fails(tmp_path, _clean() + [
        ("KICKS", "punchy-sub-heavy", "hard.wav", hard, "V1/Pack/hard.wav"),
        ("WAVES", "cycle-saw-bright", "SINE-C6.wav", tiny, "Vendor A/WAVETABLES/Synth A/SINE-C6.wav")])
    assert "one-shots start quietly (faded in, no click)" in fails
    assert "single cycles at least 256 samples" in fails


def test_verify_fails_a_flipped_stereo_file(tmp_path):
    t = np.arange(int(SR * 0.3)) / SR
    s = (0.7 * np.exp(-12 * t) * np.sin(2 * np.pi * 300 * t)).astype("float32")
    s[:22] *= np.linspace(0, 1, 22, endpoint=False).astype("float32")
    ok, fails = _fails(tmp_path, _clean() + [
        ("CLAPS", "gated-snappy", "flip.wav", np.stack([s, -s], axis=1), "V1/Pack/flip.wav")])
    assert "no stereo file cancels in mono (one channel flipped)" in fails


def test_tail_check_counts_a_blip_the_trim_cut_at():
    # the trim keeps a tail up to its last 20 ms run in the source; after the cut, the fade and
    # the envelope window at the edge shorten that run under 20 ms, and verify read everything
    # before it as quiet
    from fourier.packs.verify import _tail_sound
    sr = 44100
    env = np.full(sr, 1e-4)
    env[: sr // 10] = 1.0                        # the hit
    env[int(0.5 * sr): int(0.51 * sr)] = 0.02    # a 10 ms blip mid-tail: not sound
    env[-int(0.012 * sr):] = 0.02                # 12 ms at the very end, where the trim cut
    nz = _tail_sound(env, 0.01, sr)
    assert nz[-1] == len(env) - 1
    env[-int(0.012 * sr):] = 1e-4                # ...but a quiet end is still quiet
    nz = _tail_sound(env, 0.01, sr)
    assert nz[-1] < sr // 10


def test_stab_length_is_judged_as_the_build_judged_it():
    # verify judges a stab's length by the stored duration, as the build did, not by its frames:
    # a stored float a hair over the limit is too long for a stab in both
    from types import SimpleNamespace
    from fourier.packs import curate as C
    from fourier.packs.verify import _build_duration
    ctx = SimpleNamespace(dur={"/lib/fx_stab_x.wav": 4.000000476837158})
    d = _build_duration(ctx, "/lib/fx_stab_x.wav")
    assert d == 4.000000476837158 and not C._is_named_stab("fx_stab_x.wav", d, 1)
