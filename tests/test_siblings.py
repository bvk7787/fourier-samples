"""Multisample sets: the sibling key reads every way a name states its note (glued, below octave
0, before a pack ID or a sampler's hex code), retunes from a detected root need a steady voiced
note, and FX keeps two notes of a set a folder. Invented names only."""
import os

from fourier.packs import curate as C
from fourier.packs import curate_config as CC

PACK = "/x/Acme/Qxvwz Synths"


def k(name):
    return C._sibling_key(os.path.join(PACK, name))


# --- 1. the sibling key -------------------------------------------------------------------------

def test_a_pack_id_after_the_note_is_not_the_name():
    assert k("66_Glimmox_ZQ101_F#3-QZ1X.wav") == k("85_Glimmox_ZQ101_C#5-2ZQV.wav") is not None
    assert k("Vqx Chord Wob-C3-127-86QZ.wav") == k("Vqx Chord Wob-E3-127-7QZX.wav") is not None
    assert k("Vqx Chord Wob-C3-127-86QZ.wav") == k("Vqx Chord Wob-A1-127-QZXV.wav")


def test_a_note_below_octave_0_and_a_glued_note():
    assert k("Qzv Bass A-1.wav") == k("Qzv Bass C2.wav") is not None
    assert k("21_Frobnik_ZQ5_A-1.wav") == k("48_Frobnik_ZQ5_C2.wav")
    assert k("Qzv GlorbF#2-QZ12.wav") == k("Qzv GlorbC3-ZXQV.wav") == k("Qzv Glorb C6.wav") is not None
    assert k("GlimmoxC3-QZ12.wav") == k("GlimmoxD#5-1ZQX.wav") is not None
    assert k("Qzv Buzz (mod)F#3-6QZX.wav") == k("Qzv Buzz (mod)D#4-QZ1V.wav")


def test_a_note_with_a_hex_code_after_it():
    assert k("Frobnik_f#34b00ff.wav") == k("Frobnik_d#44b00ff.wav") == k("Frobnik_a14b00ff.wav") is not None
    assert k("Acme Lead_c44b00ff.wav") != k("Frobnik_c44b00ff.wav")


def test_different_patches_stay_apart():
    assert k("GlimmoxC3-QZ12.wav") != k("FrobnikC3-QZ12.wav")
    # a MIDI number, then a four-letter patch name: the name stays, the ID after the note goes
    assert k("99_Qxvw_ZQ101_D#6-QZ12.wav") != k("40_Vqxa_ZQ101_E1-QZ34.wav")
    assert k("99_Qxvw_ZQ101_D#6-QZ12.wav") == k("56_Qxvw_ZQ101_G#2-ZQ9X.wav")
    # a lowercase word after the note is a word, not an ID
    assert k("Glimmox_C3_soft.wav") != k("Glimmox_C3_hard.wav")
    # a model number isn't a note
    assert k("Qzv Pad D50 01.wav") != k("Qzv Pad JX8 01.wav")


def test_numbered_one_shots_and_chords_as_before():
    assert k("Kick_01.wav") is None and k("Kick_02.wav") is None
    assert k("106_QZVX.wav") is None                    # a number and an ID-like word: no set
    assert k("Qzv Pad 127 X1Y2.wav") == k("Qzv Pad 127 Z9W8.wav")
    assert k("Acme Stab Cmaj7.wav") != k("Acme Stab Abm7.wav")
    assert k("Acme Chord Dm9 02.wav") != k("Acme Chord Am9 02.wav")
    assert k("Qzv Saw Lead 02.wav") == k("Qzv Saw Lead 05.wav")      # as before (a patch number)


# --- 2. one note per C, from the keys ----------------------------------------------------------

def _rec(name, category="SUB", tune=None, **kw):
    path = os.path.join(PACK, name)
    rt, src = C._retune_shift(category, name, tune)
    return dict(path=path, tune=tune, retune=rt, root_src=src, name_pc=C._name_root_pc(name), **kw)


def test_glued_notes_on_one_c_keep_one():
    rec = [_rec("Qzv GlorbF#2-QZ12.wav"), _rec("Qzv GlorbC3-ZXQV.wav"), _rec("Qzv GlorbD#3-1QZV.wav")]
    assert [r["retune"] for r in rec] == [6, 0, -3]
    groups, n = C._one_per_retune_target([[0], [1, 2]], rec)
    assert n == 2 and groups == [[1]]


def test_a_detected_root_counts_when_the_name_states_the_note():
    rec = [_rec("Frobnik_f#34b00ff.wav", "SYNTH", tune=54.0), _rec("Frobnik_d#44b00ff.wav", "SYNTH", tune=63.0)]
    # the name's own note is read now (before a sampler's hex code): the retune is the name's
    assert [(r["retune"], r["root_src"]) for r in rec] == [(6, "name"), (-3, "name")]
    groups, n = C._one_per_retune_target([[0, 1]], rec)
    assert n == 1 and groups == [[1]]                       # -3 over +6, both on C4
    # numbered takes with a measured root may be other patches: untouched (as before)
    rec = [_rec("Qzv Saw Lead 02.wav", "SYNTH", tune=54.0), _rec("Qzv Saw Lead 05.wav", "SYNTH", tune=63.0)]
    assert C._one_per_retune_target([[0, 1]], rec) == ([[0, 1]], 0)


def test_a_pyin_root_never_retunes_onto_a_c_its_set_plays(monkeypatch):
    monkeypatch.delenv("FOURIER_NO_PITCH_CHECK", raising=False)
    pitch = {"Frobnik_c34b00ff.wav": 47.99, "Frobnik_f#24b00ff.wav": 42.01, "Frobnik_a44b00ff.wav": 69.0,
             "Qzv Saw Lead 02.wav": 42.0, "Qzv Saw Lead 05.wav": 48.0}
    monkeypatch.setattr(C, "_detect_pitch", lambda p: pitch[os.path.basename(p)])
    rec = [dict(path=os.path.join(PACK, n), tune=None, retune=None, root_src=None, name_pc=None) for n in pitch]
    assert C._fill_roots("SUB", list(range(5)), rec) == 4
    # c3 (+0.01) takes C3 first; f#2 (+5.99) would play it too and keeps its pitch; a4 goes to C5
    assert [r["retune"] for r in rec] == [0.01, None, 3.0, 6.0, 0.0]


# --- 3. a detected root needs a steady voiced note ------------------------------------------------

def test_a_detected_root_retunes_only_a_steady_voiced_note(monkeypatch):
    monkeypatch.delenv("FOURIER_NO_PITCH_CHECK", raising=False)
    heard = {"Vqx Yell Back.wav": None, "Vqx Ooh 03.wav": 57.02, "Vqx Phrase.wav": None}
    monkeypatch.setattr(C, "_detect_pitch", lambda p: heard[os.path.basename(p)])
    rec = [dict(path=os.path.join(PACK, "Vqx Yell Back.wav"), retune=2.83, root_src="detect"),
           dict(path=os.path.join(PACK, "Vqx Ooh 03.wav"), retune=3.0, root_src="detect"),
           dict(path=os.path.join(PACK, "Vqx Phrase.wav"), retune=-1.0, root_src="name")]
    assert C._check_detected_roots([0, 1, 2], rec) == 1
    assert [(r["retune"], r["root_src"]) for r in rec] == [(None, None), (3.0, "detect"), (-1.0, "name")]
    monkeypatch.setenv("FOURIER_NO_PITCH_CHECK", "1")                # no pitch checks: as before
    shout = [dict(path=os.path.join(PACK, "Vqx Yell Back.wav"), retune=2.83, root_src="detect")]
    assert C._check_detected_roots([0], shout) == 0 and shout[0]["retune"] == 2.83


# --- 4. FX keeps two notes of a set a folder ------------------------------------------------------

def test_fx_caps_sets_at_two():
    assert "FX" in CC.SIBLING_CAP_CATS and CC.sibling_cap("FX") == 2 and CC.sibling_cap("SYNTH") == 3
    rec = [dict(path=os.path.join(PACK, f"Qzv Laser Zap {n}.wav"), tune=t, pin=False)
           for n, t in (("C2", 36), ("C3", 48), ("C4", 60), ("C5", 72))]
    assert C._cap_siblings(list(range(4)), rec, cap=CC.sibling_cap("FX")) == [0, 3]
