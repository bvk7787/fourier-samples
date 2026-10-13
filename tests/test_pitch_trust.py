"""Pitch trust: a name's note is read only where it is one (a word boundary, never a pack or
device code leading the name, lowercase notes ending the name or before a sampler's hex
code); a retune to C needs two of the name, Sononym's root and pYIN's steady pitch to agree
within half a semitone, takes its cents from pYIN, and a name the audio contradicts says
the note the file plays; a set's spread stays in a playable register; the note window
reads pYIN where Sononym isn't sure. Invented names only."""
import os
from types import SimpleNamespace

import pytest

from fourier.packs import curate as C

PACK = "/x/Acme/Qxvwz Synths"


@pytest.fixture(autouse=True)
def _pitch_checks_on(monkeypatch):
    monkeypatch.delenv("FOURIER_NO_PITCH_CHECK", raising=False)


def _heard(monkeypatch, pitches):
    monkeypatch.setattr(C, "_detect_pitch", lambda p: pitches.get(os.path.basename(p)))


def _rec(name, category="SYNTH", tune=None):
    rt, src = C._retune_shift(category, name, tune)
    return dict(path=os.path.join(PACK, name), tune=tune, retune=rt, root_src=src,
                name_pc=C._name_root_pc(name))


# --- 1. reading a name's note ------------------------------------------------------------------

def test_a_note_needs_a_word_boundary():
    assert C._name_root_midi("G5zork_03.wav") is None             # a code glued to a word
    assert C._name_root_midi("G5zorkQX_09.wav") is None
    assert C._name_root_midi("Qzv Pad A4.wav") == 69 and C._name_root_midi("warmpadF#3.wav") == 54
    assert C._name_root_midi("Qzv Pad D50 01.wav") is None        # a model number, as before


def test_a_leading_device_code_is_never_the_note():
    assert C._name_root_midi("E7_qz1_052_e3.wav") == 52            # the note ends the name
    assert C._name_root_midi("G5_Zorbs_08.wav") is None            # a code and no note
    assert C._name_root_pc("G5_Zorbs_08.wav") is None
    assert C._name_root_midi("C#3_Qzv_Pad.wav") == 49               # an accidental: a note
    assert C._retuned_stem("E7_qz1_052_e3", -4.0) == "E7_qz1_052_c3"


def test_lowercase_notes_ending_the_name_and_before_a_hex_code():
    assert C._name_root_midi("Qxvwz_g#36b00ff.wav") == 56
    assert C._name_root_midi("Qxvwz Lead_c46b00ff.wav") == 60
    assert C._name_root_midi("Qzv_Wub_05 a1.wav") == 33 and C._name_root_midi("Qzv_Pad_e3.wav") == 52
    assert C._name_root_midi("Qzv_b1_Pad.wav") is None             # not the last token
    # the name says the C it plays, in its own case, the hex code kept
    assert C._retuned_stem("Qxvwz_g#36b00ff", 4.0) == "Qxvwz_c46b00ff"
    assert C._retuned_stem("Qzv_Wub_05 a1", 3.0) == "Qzv_Wub_05 c2"
    # a set of such names is still one set (the hex code isn't the name)
    k = lambda n: C._sibling_key(os.path.join(PACK, n))         # noqa: E731
    assert k("Qxvwz_g#36b00ff.wav") == k("Qxvwz_c46b00ff.wav") is not None


# --- 2. two readings agree ---------------------------------------------------------------------

def test_readings_agree_within_half_a_semitone_octaves_aside():
    assert C._pitches_agree(57.4, 45) and C._pitches_agree(44.6, 45) and not C._pitches_agree(45.6, 45)
    assert C._pitch_conflicts(9, 70.0) and C._pitch_conflicts(9, 64.0)     # a semitone, a fifth
    assert not C._pitch_conflicts(9, 81.2)


def test_a_name_backed_by_pyin_takes_pyins_cents(monkeypatch):
    _heard(monkeypatch, {"Qzv Lead A2.wav": 57.3, "Qzv Lead F#2.wav": 53.8, "Qzv Lead B2.wav": 47.02})
    rec = [_rec("Qzv Lead A2.wav"), _rec("Qzv Lead F#2.wav"), _rec("Qzv Lead B2.wav")]
    assert C._check_name_pitches([0, 1, 2], rec, "SYNTH") == 0
    assert rec[0]["retune"] == 2.7                         # A +0.3 to C
    assert rec[1]["retune"] == -5.8                        # a flat F# goes down: F# -0.2 to C
    assert rec[2]["retune"] == 1.0                         # 2 cents: pYIN's resolution, left


def test_name_and_sononym_agree_against_a_semitone_off_pyin(monkeypatch):
    _heard(monkeypatch, {"Qzv Pad D#4.wav": 64.1})
    rec = [_rec("Qzv Pad D#4.wav", tune=63.2)]
    assert C._check_name_pitches([0], rec, "SYNTH") == 0 and rec[0]["retune"] == -3.2


def test_sononym_and_pyin_agree_on_another_note(monkeypatch):
    """A pack labelled G that plays G#: retuned by pYIN to C and named for that C."""
    _heard(monkeypatch, {"Qzv Sub G3.wav": 44.2})
    rec = [_rec("Qzv Sub G3.wav", "SUB", tune=44.0)]
    assert rec[0]["retune"] == 5.0
    assert C._check_name_pitches([0], rec, "SUB") == 1
    r = rec[0]
    assert (r["retune"], r["root_src"], r["root_midi"], r["renote"]) == (3.8, "pyin", 44.2, 60)
    assert "pitch_conflict" not in r
    assert C._renoted_stem("Qzv Sub G3", r["renote"]) == "Qzv Sub C4"


def test_pyin_alone_against_the_name_keeps_the_pitch_and_names_its_note(monkeypatch):
    _heard(monkeypatch, {"Qzv Pad D#4.wav": 64.1, "Qzv Bell_c3.wav": 61.0})
    rec = [_rec("Qzv Pad D#4.wav"), _rec("Qzv Bell_c3.wav", tune=55.0)]
    assert C._check_name_pitches([0, 1], rec, "SYNTH") == 2
    assert [(r["retune"], r["root_src"], r["renote"]) for r in rec] == [(None, None, 64), (None, None, 49)]
    assert rec[0]["pitch_conflict"] == 64.1
    assert C._renoted_stem("Qzv Pad D#4", 64) == "Qzv Pad E4"
    assert C._renoted_stem("Qzv Bell_c3", 49) == "Qzv Bell_c#3"


def test_a_fifth_over_the_name_keeps_pitch_and_name(monkeypatch):
    """pYIN on the named note's 3rd harmonic: even with Sononym beside it, nothing moves."""
    _heard(monkeypatch, {"QzvC3-QZ1X.wav": 55.1})
    rec = [_rec("QzvC3-QZ1X.wav", tune=55.0)]
    assert C._check_name_pitches([0], rec, "SYNTH") == 1
    assert rec[0]["retune"] is None and "renote" not in rec[0] and rec[0]["pitch_conflict"] == 55.1


def test_sononym_alone_against_the_name_keeps_pitch_and_name(monkeypatch):
    _heard(monkeypatch, {})
    rec = [_rec("Qzv Pad A2.wav", tune=50.0), _rec("Qzv Pad B2.wav")]
    assert C._check_name_pitches([0, 1], rec, "SYNTH") == 1
    assert rec[0]["retune"] is None and rec[0]["pitch_conflict"] == 50.0 and "renote" not in rec[0]
    assert rec[1]["retune"] == 1.0                       # a name nothing contradicts stands


def test_a_capped_category_names_the_note_it_keeps(monkeypatch):
    """VOX moves at most 3 semitones: Sononym and pYIN agree on an F, 5 from C, so the file
    keeps its pitch and is named F."""
    _heard(monkeypatch, {"Vqx Ooh A3.wav": 65.0})
    rec = [_rec("Vqx Ooh A3.wav", "VOX", tune=65.1)]
    assert C._check_name_pitches([0], rec, "VOX") == 1
    assert rec[0]["retune"] is None and rec[0]["renote"] == 53


def test_a_detected_root_needs_pyin_on_its_note(monkeypatch):
    _heard(monkeypatch, {"Qzv Bell 43.wav": 60.03, "Qzv Bell 44.wav": 62.4, "Qzv Bell 45.wav": None})
    rec = [dict(path=os.path.join(PACK, f"Qzv Bell {n}.wav"), tune=t, retune=rt, root_src="detect")
           for n, t, rt in ((43, 77.03, -5.03), (44, 62.2, -2.2), (45, 60.5, -0.5))]
    assert C._check_detected_roots([0, 1, 2], rec) == 2
    assert (rec[0]["retune"], rec[0]["pitch_conflict"]) == (None, 60.03)   # a C Sononym read as F
    assert rec[1]["retune"] == -2.4                                     # pYIN's cents
    assert rec[2]["retune"] is None and "pitch_conflict" not in rec[2]  # no steady pitch
    assert C._fill_roots("SYNTH", [0], rec) == 0                        # and nothing retunes it


# --- 3. a set's spread stays playable; the window reads pYIN -----------------------------------

def test_a_sets_spread_stays_in_the_playable_register():
    names = ("Qzv Bass C-2.wav", "Qzv Bass C0.wav", "Qzv Bass C2.wav", "Qzv Bass C4.wav",
             "Qzv Bass C6.wav", "Qzv Bass C8.wav")
    rec = [dict(path=os.path.join(PACK, n), tune=None, retune=0.0, pin=False, name_pc=0) for n in names]
    rng = (24, 96)
    assert [C._outside_spread(r, rng) for r in rec] == [True, False, False, False, False, True]
    assert C._cap_siblings(list(range(6)), rec, cap=3, spread=rng) == [1, 3, 4]
    assert C._cap_siblings(list(range(6)), rec, cap=3) == [0, 2, 5]              # as before
    assert C._cap_siblings([0, 1, 5], rec, cap=3, spread=rng) == [0, 1, 5]       # room for all
    assert C._cap_siblings([0, 1, 5], rec, cap=2, spread=rng) == [0, 1]          # one far note fills
    # Sononym's sure root decides over the name's octave; pYIN's counts where it isn't sure
    assert not C._outside_spread(dict(rec[0], tune=36.0), rng)
    assert C._outside_spread(dict(rec[2], own_root=20.0), rng)


def test_the_note_window_reads_pyin_when_sononym_is_unsure():
    r = SimpleNamespace(base_note=100.0, base_note_confidence=0.2, own_root_midi=70.0)
    assert C._window_note(r) == 70.0
    assert C._window_note(SimpleNamespace(base_note=100.0, base_note_confidence=0.9, own_root_midi=70.0)) == 100.0
    assert C._window_note(SimpleNamespace(base_note=None, base_note_confidence=None)) is None
