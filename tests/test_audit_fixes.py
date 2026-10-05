"""Export and routing rules: mono phase, DC, instrument tails, waves,
length caps, bpm-named phrases, identical audio across categories."""
import json

import numpy as np
import soundfile as sf

from fourier.devices.exporter import _convert_and_copy, _mono_downmix
from fourier.packs import curate as C

SR = 44100


def test_mono_downmix_keeps_a_channel_when_the_sum_cancels():
    t = np.arange(SR // 10) / SR
    s = np.sin(2 * np.pi * 440 * t)
    anti = np.stack([s, -0.95 * s])
    m = _mono_downmix(anti)
    assert np.allclose(m, s)                                  # louder channel, whole
    wide = np.stack([s, 0.5 * s])
    assert np.allclose(_mono_downmix(wide), 0.75 * s)         # ordinary file: plain average


def test_waves_render_sample_exact(tmp_path):
    t = np.arange(2048) / 2048
    cyc = (0.8 * np.sin(2 * np.pi * t)).astype("float32")
    src = tmp_path / "cyc.wav"; sf.write(src, cyc, SR, subtype="PCM_24")
    out = tmp_path / "out.wav"
    _convert_and_copy(src, out, target_sr=48000, target_bit_depth=16, convert_to_mono=True,
                      dither=True, preserve_length=True)
    y, sr = sf.read(out)
    assert sr == 48000 and len(y) == 2048                     # never 2229.1 samples
    assert np.max(np.abs(y - cyc)) < 1e-4
    out2 = tmp_path / "out2.wav"
    _convert_and_copy(src, out2, target_sr=48000, target_bit_depth=16, convert_to_mono=True,
                      dither=False)
    assert len(sf.read(out2)[0]) == 2229                      # what a normal render does


def _hit(dc=0.0):
    t = np.arange(SR // 2) / SR
    return np.exp(-8 * t) * np.sin(2 * np.pi * 60 * t) + dc


def test_dc_removed_without_a_step():
    y = _hit(dc=0.2)
    out = C._remove_dc(y, SR)
    assert abs(out.mean()) < 0.01
    assert abs(out[0]) < 0.05 and abs(out[-1]) < 0.05         # no step at either end
    clean = _hit()
    assert C._remove_dc(clean, SR) is clean                  # below DC_FIX: untouched


def test_dc_removed_from_a_short_hit_and_relative_to_peak():
    t = np.arange(6349) / 48000
    hat = np.exp(-40 * t) * np.random.default_rng(0).standard_normal(len(t)) * 0.3 - 0.037
    out = C._remove_dc(hat, 48000)
    assert abs(out.mean()) < 0.005 and abs(out[0]) < 1e-9 and abs(out[-1]) < 1e-9
    quiet = _hit() * 0.1 + 0.01                      # 0.01 absolute, but 10% of the peak
    assert abs(C._remove_dc(quiet, SR).mean()) < 0.002


def test_dc_exact_for_waves():
    cyc = np.sin(2 * np.pi * np.arange(256) / 256) + 0.3
    out = C._remove_dc(cyc, SR, exact=True)
    assert len(out) == 256 and abs(out.mean()) < 1e-9
    small = np.sin(2 * np.pi * np.arange(256) / 256) - 0.014    # below any threshold: still removed
    assert abs(C._remove_dc(small, SR, exact=True).mean()) < 1e-9


def test_instrument_tail_trimmed_below_minus_60_with_fade():
    y = np.concatenate([_hit(), np.zeros(SR * 3)])
    out = C._trim_edges(y, SR, True, True, tail_floor_db=-60.0, fade_ms=20.0)
    assert len(out) < len(y) - SR * 2
    assert abs(out[-1]) < 1e-3


def test_waves_detection_and_labels():
    assert C._is_wave("Vendor A/Synth Pack/WAVETABLES/Synth Waves/Saw C3.wav", 0.004)
    assert C._is_wave("Vendor B/Wavetable Pack 1/x/Table 1 (2048x32)/Table_31.wav", 0.046)
    assert not C._is_wave("Vendor C/Drum Pack/Samples/Waveforms/Drums/Snare/Snare_07.wav", 0.2)
    assert not C._is_wave("Vendor A/Synth Pack/WAVETABLES/Synth Waves/Long Drone.wav", 4.0)
    assert not C._is_wave("Vendor C/Drum Machine Pack/WAV/Hat Closed 27.wav", 0.09)
    assert C._wave_source("Vendor A/Synth Pack/WAVETABLES/Soft Synth - Set 2/Soft_WT_71.wav") == "soft-synth"
    assert C._wave_source("Vendor B/Wavetable Pack 1/General/Table_31.wav") == "vendor-b"
    assert C._wave_band(2048 / 44100, 44100) == "cycle"
    assert C._wave_band(32768 / 44100, 44100) == "table"


def test_twin_homes_unified():
    homes = {1: "PERC", 2: "SNARES", 3: "KICKS"}
    moved = C._unify_twin_homes(homes, {1: "h", 2: "h", 3: "other"})
    assert homes == {1: "PERC", 2: "PERC", 3: "KICKS"} and moved == 1


def test_validate_flags_identical_audio_in_two_categories(tmp_path):
    from fourier.packs.validate import validate_master
    man = {"categories": {
        "PERC": {"families": 1, "files": 1, "entries": [
            {"family": "a", "out": "a/x.wav", "src": "/p/x.wav", "out_md5": "m1"}]},
        "SNARES": {"families": 1, "files": 1, "entries": [
            {"family": "b", "out": "b/y.wav", "src": "/q/y.wav", "out_md5": "m1"}]}}}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    ok, res = validate_master(str(tmp_path), budgets={}, log=lambda *a: None)
    assert any(c == "dup" and "identical audio" in d for _, c, d in res)


def test_a_stray_click_does_not_hold_the_tail():
    y = np.concatenate([_hit(), np.zeros(SR)])
    y[len(_hit()) + SR // 2] = 0.01                      # one sample at -38 dB, 0.5 s into silence
    out = C._trim_edges(y, SR, True, True, tail_floor_db=-60.0, fade_ms=20.0)
    assert len(out) < len(_hit()) + SR // 4


def test_a_short_blip_after_the_decay_is_cut():
    blip = 0.04 * np.sin(2 * np.pi * 900 * np.arange(int(SR * 0.005)) / SR)   # 5 ms click
    y = np.concatenate([_hit(), np.zeros(SR // 2), blip, np.zeros(SR // 4)])
    out = C._trim_edges(y, SR, True, True, tail_floor_db=-60.0, fade_ms=5.0)
    assert len(out) < len(_hit()) + SR // 10
    assert list(C._sustained_above(np.array([0, 1, 1, 1, 0, 0, 1, 0]), 0.5, 2)) == [1, 2, 3]


def test_dc_is_judged_after_the_trim():
    # a thump with a small DC offset over the whole source (under the fix line) that grows
    # once its long quiet tail is trimmed
    sr = SR
    t = np.arange(int(sr * 0.15)) / sr
    hit = 0.9 * np.exp(-20 * t) * np.sin(2 * np.pi * 90 * t) + 0.08 * np.exp(-8 * t)
    y = np.concatenate([hit, np.zeros(int(sr * 0.85))])
    assert abs(y.mean()) / np.abs(y).max() < C.DC_FIX                  # passes before the trim
    out = C._process_audio(y, sr, "oneshot", True, True, "peak")
    assert len(out) < len(y)                                            # the tail was trimmed
    assert abs(out.mean()) / np.abs(out).max() < 0.01


def test_long_hit_falls_back_to_the_mean():
    # a very short burst in a long file: the high-pass leaves a residual mean it can't
    # see as DC (0.75% here), so the mean is subtracted after it
    sr = SR
    t = np.arange(int(sr * 1.5)) / sr
    y = 0.9 * np.exp(-300 * t) * (np.sin(2 * np.pi * 60 * t) + 0.6)
    out = C._remove_dc(y, sr, thr=0.001)
    assert abs(out.mean()) / np.abs(out).max() < 0.001                 # the high-pass alone leaves 0.75%
    assert out[0] == 0 and out[-1] == 0                                   # faded ends


def test_dc_fixed_on_a_click_a_few_ms_long():
    # a short click with a large DC offset. The fades cover most of a click this
    # short, so the offset is weighted by them and the result has no mean left
    sr = SR
    t = np.arange(364) / sr
    y = 0.4 * np.sin(2 * np.pi * 900 * t) * np.exp(-400 * t) + 0.3
    for out in (C._remove_dc(y, sr), C._process_audio(y, sr, "oneshot", True, True, "peak")):
        assert abs(out.mean()) / np.abs(out).max() < 0.005
        assert out[0] == 0 and out[-1] == 0


def test_low_square_keeps_no_dc_after_the_end_fade():
    # a C1 square cut off mid-cycle: the last end fade moves its mean, which must not stay
    # behind as DC
    sr = 44100
    t = (np.arange(2697) * 32.7 / sr + 0.35) % 1
    y = np.where(t < 0.5, 0.8, -0.8)
    out = C._process_audio(y, sr, "oneshot", True, True, "peak")
    pk = np.abs(out).max()
    assert abs(out.mean()) <= 0.02 * pk
    assert np.abs(out[-4:]).max() <= pk * 10 ** (-40 / 20)
