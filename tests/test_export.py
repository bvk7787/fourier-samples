"""Unit tests for the trim/loudness export primitives (pure DSP, no files)."""
import numpy as np

from fourier.packs.curate import LOOP_RMS_DB, _trim_edges, _normalize, _export_policy, _amp_from_db


def _oneshot(sr=48000):
    return np.concatenate([np.zeros(4800), np.ones(4800) * 0.8, np.zeros(4800)]).astype("float64")


def test_trim_removes_silence_for_oneshots():
    sr = 48000
    y = _oneshot(sr)
    out = _trim_edges(y, sr, True, True)
    assert len(out) < len(y)
    assert float(np.abs(out).max()) > 0.5   # the body survives


def test_trim_never_touches_loops():
    sr = 48000
    y = np.concatenate([np.zeros(4800), np.ones(4800) * 0.8]).astype("float64")
    assert len(_trim_edges(y, sr, False, False)) == len(y)


def test_trim_keeps_lead_pad():
    sr = 48000
    y = np.concatenate([np.zeros(4800), np.ones(100) * 0.8]).astype("float64")
    out = _trim_edges(y, sr, True, False)
    # trims the long lead-in but keeps a few-ms pad, not a hard cut on the onset
    assert 100 <= len(out) < len(y)


def test_trim_all_silent_is_noop():
    sr = 48000
    y = np.zeros(2000, dtype="float64")
    assert len(_trim_edges(y, sr, True, True)) == len(y)


def test_normalize_peak_hits_ceiling():
    y = (np.ones(1000) * 0.5).astype("float64")
    out = _normalize(y, "peak")
    assert abs(float(np.abs(out).max()) - _amp_from_db(-1.0)) < 1e-3


def test_normalize_caps_gain_on_quiet():
    y = (np.ones(1000) * 1e-4).astype("float64")   # near-silent
    out = _normalize(y, "rms")
    # amplification capped at +18 dB, so it stays quiet instead of exploding
    assert float(np.abs(out).max()) <= 1e-4 * _amp_from_db(18.0) + 1e-9


def test_normalize_rms_never_clips():
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(2000) * 0.05).astype("float64")
    out = _normalize(y, "rms")
    assert float(np.abs(out).max()) <= _amp_from_db(-1.0) + 1e-6



def _rms_db(y):
    return 20 * np.log10(np.sqrt(np.mean(np.square(y))))


def test_loops_land_on_minus_20_dbfs_rms():
    assert LOOP_RMS_DB == -20.0
    t = np.arange(48000) / 48000
    groove = 0.05 * np.sin(2 * np.pi * 110 * t)            # -29 dBFS RMS, 3 dB crest: room to rise
    assert abs(_rms_db(_normalize(groove, "rms")) - LOOP_RMS_DB) < 0.01
    clicks = np.zeros(48000)
    clicks[::12000] = 0.5                                   # peaky: the -1 dBFS ceiling holds it
    out = _normalize(clicks, "rms")
    assert _rms_db(out) < LOOP_RMS_DB - 6 and abs(np.abs(out).max() - _amp_from_db(-1.0)) < 1e-9

def test_normalize_none_is_noop():
    y = np.ones(10) * 0.3
    assert np.allclose(_normalize(y, None), y)


def test_normalize_silent_is_noop():
    y = np.zeros(10)
    assert np.allclose(_normalize(y, "peak"), y)


def test_export_policy_per_kind():
    assert _export_policy("loop") == ("rms", (False, False))
    assert _export_policy("oneshot") == ("peak", (True, True))
    assert _export_policy("gated") == ("peak", (True, True))
    assert _export_policy("instrument") == ("peak", (True, True))   # tail only below -60 dB
    assert _export_policy("waves") == ("peak", (False, False))


def test_render_resample_never_clips(tmp_path):
    import numpy as np
    import soundfile as sf
    from fourier.devices.exporter import _convert_and_copy
    sr = 44100
    t = np.arange(sr) / sr
    # a near-Nyquist tone sampled off its peaks: upsampling reveals inter-sample overs
    y = np.sin(2 * np.pi * 11025 * t + np.pi / 4) * 0.891       # -1 dBFS sample peak
    y = y / np.abs(y).max() * 0.891
    src = tmp_path / "a.wav"
    sf.write(str(src), y, sr, subtype="PCM_16")
    dst = tmp_path / "b.wav"
    _convert_and_copy(src, dst, target_sr=48000, target_bit_depth=16, convert_to_mono=False, dither=False)
    z, _ = sf.read(str(dst))
    assert np.abs(z).max() <= 0.8913 + 1e-3


def test_mono_downmix_keeps_level_of_wide_files():
    import numpy as np
    from fourier.devices.exporter import _mono_downmix
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.1, (2, 48000))                  # uncorrelated: the plain mean loses 3 dB
    m0 = _mono_downmix(a)
    m1 = _mono_downmix(a, keep_level=True)
    ch = np.sqrt(np.mean(a ** 2))
    assert np.sqrt(np.mean(m0 ** 2)) < ch * 0.75
    assert abs(np.sqrt(np.mean(m1 ** 2)) - ch) / ch < 0.02 or np.abs(m1).max() >= np.abs(a).max() - 1e-9


def test_short_hit_gets_an_end_fade():
    import numpy as np
    from fourier.packs.curate import _end_fade
    sr = 44100
    y = np.sin(np.arange(1240) * 0.3)[:, None] * np.array([[0.6, 0.5]])   # 28 ms, ends hot
    z = _end_fade(y, sr)
    assert np.abs(z[-1]).max() < 1e-9 and np.abs(z[:100] - y[:100]).max() == 0


def test_tiny_click_ends_below_the_verify_threshold():
    import numpy as np
    from fourier.packs.curate import _end_fade
    from fourier.packs.curate_config import END_HOT_DB
    y = 0.6 * np.cos(np.arange(36) * 0.9) * np.exp(-np.arange(36) / 12.0)   # a 36-sample click
    y[-6:] = [0.2, -0.1, 0.06, -0.05, 0.04, -0.03]                           # cut off, not decayed
    z = _end_fade(y, 44100)
    assert np.abs(z[-4:]).max() <= np.abs(z).max() * 10 ** (END_HOT_DB / 20)


def test_canonical_stem_pairs_brackets():
    from fourier.devices.exporter import canonical_stem
    assert canonical_stem("Gbmaj Soft Guitar Chord [108 BPM", 46) == "Gbmaj_Soft_Guitar_Chord_108_BPM"
    assert canonical_stem("Drum_Loop_2_[112_BPM]", 46) == "Drum_Loop_2_[112_BPM]"
    assert canonical_stem("Loop_(Chord)_mix)", 46) == "Loop_(Chord)_mix"
    long = "Strings_[Pizzicato_Very_Long_Articulation_Name]_C3_V01"
    out = canonical_stem(long, 46)
    assert len(out) <= 46 and out.count("[") == out.count("]")
    assert canonical_stem(out, 46) == out                     # idempotent


# ---------------------------------------------------------------------------
# more than two channels: folded to stereo (or mono) for the master and every device
# ---------------------------------------------------------------------------
def test_fold_to_stereo_weights_by_channel_count():
    from fourier.devices.exporter import _fold_weights, fold_to_stereo
    g = 0.7071067811865476
    assert _fold_weights(4) == [(1.0, 0.0), (0.0, 1.0), (g, 0.0), (0.0, g)]           # quad
    assert _fold_weights(6) == [(1.0, 0.0), (0.0, 1.0), (g, g), (g, g), (g, 0.0), (0.0, g)]  # 5.1
    assert _fold_weights(8)[6:] == [(g, 0.0), (0.0, g)]                             # 7.1's sides
    assert [w for w, _ in _fold_weights(10)][8:] == [g, 0.0]                        # past 8: pairs
    for n in (4, 6, 8):
        x = np.zeros((n, 100))
        x[0], x[1] = 0.5, -0.25                     # left and right
        x[2:] = 0.1
        y = fold_to_stereo(x)
        assert y.shape == (2, 100)
        assert float(np.abs(y).max()) <= 0.5 + 1e-12                    # never past the source's peak
        assert y[0, 0] > 0.5 * 0.9 and y[1, 0] > -0.25                  # the front pair leads
        t = fold_to_stereo(x.T.copy(), channels_first=False)
        assert t.shape == (100, 2) and np.allclose(t, y.T)
    st = np.ones((2, 10))
    assert fold_to_stereo(st) is st and fold_to_stereo(np.ones(10)).ndim == 1


def test_fold_to_stereo_caps_integer_input_before_it_casts():
    from fourier.devices.exporter import fold_to_stereo
    x = np.full((4, 3), 30000, dtype=np.int16)          # the fold's sum is past int16's range
    y = fold_to_stereo(x)
    assert y.dtype == np.int16 and y.shape == (2, 3)
    assert (y > 0).all() and int(np.abs(y).max()) <= 30000          # capped, never wrapped
    assert int(y.max()) == 30000                                     # scaled to the source's peak


def _wide_master(tmp_path, sf, counts=(4, 6, 8)):
    master = tmp_path / "master"
    for n in counts:
        p = master / "KICKS" / "wide" / f"Kick {n}ch.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(22050) / 44100
        y = np.stack([0.3 * np.sin(2 * np.pi * (60 + 20 * k) * t) * np.exp(-6 * t) for k in range(n)], 1)
        sf.write(str(p), y, 44100, subtype="PCM_24")
    return master


def test_a_render_folds_more_than_two_channels(tmp_path, monkeypatch):
    import soundfile as sf
    from fourier.devices.loader import DeviceLoader
    from fourier.packs import render as R
    from fourier.packs.verify import verify_render
    monkeypatch.setenv("FOURIER_NO_RENDER_CACHE", "1")
    master = _wide_master(tmp_path, sf)
    for did, want in (("m8_tracker", 2), ("digitakt_2", 2)):
        dev = DeviceLoader().load(did)
        out = tmp_path / did
        R.render_master_to_device(str(master), dev, str(out), log=lambda m: None)
        got = sorted(sf.info(str(p)).channels for p in out.rglob("*.wav"))
        assert got == [want] * 3, (did, got)
        ok, res = verify_render(master, out, dev, log=lambda m: None)
        assert all(r.level != "FAIL" for r in res if "device format" in r.check), res
    dev = DeviceLoader().load("m8_tracker")
    dev.channels = "mono"
    R.render_master_to_device(str(master), dev, str(tmp_path / "mono"), log=lambda m: None)
    assert {sf.info(str(p)).channels for p in (tmp_path / "mono").rglob("*.wav")} == {1}


def test_verify_render_fails_a_file_with_more_than_two_channels(tmp_path, monkeypatch):
    import soundfile as sf
    from fourier.devices.loader import DeviceLoader
    from fourier.packs import render as R
    from fourier.packs.verify import verify_render
    monkeypatch.setenv("FOURIER_NO_RENDER_CACHE", "1")
    master = _wide_master(tmp_path, sf, counts=(4,))
    dev = DeviceLoader().load("m8_tracker")
    out = tmp_path / "out"
    R.render_master_to_device(str(master), dev, str(out), log=lambda m: None)
    (p,) = list(out.rglob("*.wav"))
    sf.write(str(p), np.zeros((100, 4)), dev.sample_rate, subtype="PCM_16")      # as an older render
    _ok, res = verify_render(master, out, dev, log=lambda m: None)
    assert any(r.level == "FAIL" and "device format" in r.check for r in res)


def test_the_master_export_folds_more_than_two_channels(tmp_path, monkeypatch):
    import pytest
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("sklearn")
    from fourier.packs import curate
    monkeypatch.setenv("FOURIER_AUDIO_CACHE", str(tmp_path / "acache"))
    n, dim = 6, 16
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((n, dim)).astype("float32")
    rec = []
    for i, ch in enumerate((4, 6, 8, 4, 6, 8)):
        p = tmp_path / "SampleLibrary" / f"Acme{i}" / "Pack" / f"Kick {i:02d}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(8820) / 44100
        y = np.stack([0.5 * np.sin(2 * np.pi * (50 + 10 * k) * t) * np.exp(-20 * t) for k in range(ch)], 1)
        sf.write(str(p), y, 44100, subtype="PCM_24")
        rec.append(dict(id=i, row=i, path=str(p), pack=f"Acme{i}", clip=0, br=0.5, atk=10.0,
                        dec=300.0, sub=0.1, noi=0.3, har=0.7, cr=4.0, tune=None, bpm=None,
                        bpm_src=None, bpm_fold=None, sup=1, qual=1.0, fav=0, ab=[], vendor=f"Acme{i}"))
    monkeypatch.setattr(curate, "index_stale", lambda s: (False, n, n))
    monkeypatch.setattr(curate, "load_index", lambda: (np.arange(n), emb))
    monkeypatch.setattr(curate, "_fetch_rows", lambda *a, **k: [])
    monkeypatch.setattr(curate, "_select_records", lambda *a, **k: (list(rec), set(), dict(twins=0)))
    monkeypatch.setattr(curate, "embed_text",
                        lambda t: np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(dim)
                        .astype("float32"))
    for loudness in (True, False):                 # processed, and copied as they are
        out = tmp_path / f"master-{loudness}"
        _summary, entries = curate.build_taxonomy(None, "KICKS", str(out), homes={}, support={},
                                                  votes={}, describe=False, loudness=loudness,
                                                  return_entries=True, log=lambda m: None)
        files = list(out.rglob("*.wav"))
        assert files and all(sf.info(str(p)).channels <= 2 for p in files), \
            (loudness, [sf.info(str(p)).channels for p in files])
