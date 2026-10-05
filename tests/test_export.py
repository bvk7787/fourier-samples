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
