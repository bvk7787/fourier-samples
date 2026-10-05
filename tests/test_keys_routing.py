"""Keys routing: piano notes and chords out of SYNTH/SUB/FX."""
import numpy as np

from fourier.packs import curate as C


def _anchors():
    pa = np.zeros(8, "float32"); pa[0] = 1
    pn = np.zeros(8, "float32"); pn[1] = 1
    return pa, pn


def _vec(a, n):
    v = np.zeros(8, "float32"); v[0], v[1] = a, n
    return v


def test_keys_by_name():
    for n in ("Piano Soft 71.wav", "Piano_Loop_A_Minor_90bpm.wav", "E-Piano Soft C.wav",
              "Rhodes Bass C#1.wav", "Wurli C3.wav", "Clavinet D2.wav"):
        assert C._is_keys(n, None, None, None), n
    assert not C._is_keys("Saw Lead C3.wav", None, None, None)


def test_keys_by_clap_needs_margin_and_pitch():
    an = _anchors()
    assert C._is_keys("Tone Long 38.wav", _vec(0.40, 0.25), 0.8, an)
    assert not C._is_keys("Tone Long 38.wav", _vec(0.40, 0.35), 0.8, an)   # margin 0.05
    assert not C._is_keys("Paper_Noise_04.wav", _vec(0.40, 0.24), 0.2, an)  # not pitched


def test_chord_by_name_or_chroma():
    for n in ("Piano Chords 07.wav", "Piano Chord Warm.wav", "Keys Cmaj7.wav", "EP Am.wav",
              "Rhodes F#m7 soft.wav", "Upright Dsus4.wav", "Keys Soft Cmin.wav",
              "Piano_Loop_A_Minor_90bpm.wav"):
        assert C._is_chord(n, 3.0, 1.6), n
    for n in ("Grand Piano MF C3.wav", "Piano Soft 71.wav", "Rhodes E2.wav",
              "Amen Piano.wav", "Emotive Piano C4.wav"):
        assert not C._is_chord(n, 3.0, 1.6), n
    assert C._is_chord("Piano-83.wav", 1.3, 1.6)          # spread chroma = several pitches


def test_keys_candidates():
    assert C._keys_candidates({"SYNTH", "PADS"}, True, False, True) == {"PADS"}
    assert C._keys_candidates({"SYNTH"}, True, True, True) == set()        # chords are PIANO's
    assert C._keys_candidates({"STABS", "PADS"}, True, True, True) == {"PADS"}
    assert C._keys_candidates({"SUB"}, True, True, False) == set()   # a chord phrase: nowhere
    assert C._keys_candidates({"SYNTH"}, False, True, True) == {"SYNTH"}


def test_stab_is_short_and_unmetered():
    assert C._is_stab("Piano_Chord_short_31.wav", 1.2, True)
    assert not C._is_stab("Pad Progression Am 88bpm.wav", 3.0, True)
    assert not C._is_stab("Piano_174_Amin Pt3.wav", 3.0, True)
    assert not C._is_stab("Rhodes Chord Pad.wav", 9.0, True)
    assert not C._is_stab("Piano Chords 07.wav", 1.0, False)


def test_unpitched_piano_noise_is_not_a_chord():
    assert not C._is_chord("piano knock 5.wav", 1.2, 1.6, harmonicity=0.2)
    assert C._is_chord("Piano-83.wav", 1.3, 1.6, harmonicity=0.8)


def test_keys_families_take_keys_phrases():
    phrases = ["pizzicato stab", "choir stab", "house piano stab", "electric piano stab"]
    m = C._keys_phrase_mask([0.9, 0.2], phrases)
    assert m[0].tolist() == [True, True, False, False]
    assert not m[1].any()
    assert not C._keys_phrase_mask([0.9], ["kick", "snare"]).any()   # no keys phrase: no mask
