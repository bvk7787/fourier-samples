"""Retune to C: the shorter way round, one note per C a multisample set lands on, and folder
notes from what the files play once exported. Invented names only."""
import os

import numpy as np

from fourier.packs import curate as C

PACK = "/x/Acme/Qxvwz Keys"


def _rec(name, category="STABS", tune=None, **kw):
    path = os.path.join(PACK, name)
    rt, src = C._retune_shift(category, name, tune)
    return dict(path=path, tune=tune, retune=rt, root_src=src, name_pc=C._name_root_pc(name), **kw)


# --- 1. the shorter direction ----------------------------------------------------------------

def test_flat_fs_goes_down_sharp_or_exact_fs_goes_up():
    assert C._retune_shift("SYNTH", "Lead.wav", 65.51) == (-5.51, "detect")       # not +6.49
    assert C._retune_shift("SYNTH", "Lead.wav", 66.3) == (5.7, "detect")
    assert C._retune_shift("SYNTH", "Lead.wav", 66.0) == (6, "detect")             # exactly F#: up, as before
    assert C._retune_shift("SUB", "Qzv Bass F#1.wav", 29.99) == (-5.99, "name")    # name agrees, a hair flat
    assert C._retune_shift("SUB", "Qzv Bass F#1.wav", None) == (6, "name")
    assert C._retune_shift("SYNTH", "warmpadF#3.wav", None) == (6.0, "name")
    # the capped categories still keep a far F# as it is, either way
    assert C._retune_shift("VOX", "Vqx F#2 Ah.wav", 41.6) == (None, None)
    assert C._retune_shift("VOX", "Vqx F#2 Ah.wav", 42.4) == (None, None)


def test_every_shift_takes_the_nearer_way_and_other_notes_unchanged():
    for midi in np.arange(36.0, 48.0, 0.05):
        rt, _ = C._retune_shift("SYNTH", "Lead.wav", float(midi))
        assert abs(rt) <= 6.0 + 1e-9
        assert abs(((midi + rt) % 12 + 6) % 12 - 6) < 1e-6           # lands on a C
        if int(round(midi)) % 12 != 6:
            pc = int(round(midi)) % 12
            d = (-pc) % 12
            d = d - 12 if d > 6 else d
            assert rt == round(d - (midi - round(midi)), 3)          # as before


# --- 2. one note per C a set lands on --------------------------------------------------------

def test_set_notes_retuned_onto_one_c_keep_one():
    names = ["Qxvwz Chord F#2.wav", "Qxvwz Chord A#2.wav", "Qxvwz Chord C3.wav",
             "Qxvwz Chord D#3.wav", "Qxvwz Chord E3.wav"]
    rec = [_rec(n) for n in names]
    groups, n = C._one_per_retune_target([[0, 1, 2], [3, 4]], rec)
    assert n == 4 and groups == [[2]]                          # C3 needs no shift: it stays


def test_one_per_c_prefers_smallest_shift_and_keeps_other_cs():
    names = ["Qxvwz Chord A#2.wav", "Qxvwz Chord D3.wav", "Qxvwz Chord G3.wav", "Qxvwz Chord A3.wav"]
    rec = [_rec(n) for n in names]                             # A#2 +2, D3 -2 -> C3; G3 +5, A3 +3 -> C4
    groups, n = C._one_per_retune_target([[0, 1, 2, 3]], rec)
    assert n == 2 and groups == [[0, 3]]                       # tie at 2: the lower note (set order)


def test_one_per_c_keeps_layers_of_the_kept_note_and_keeps():
    rec = [_rec("Qxvwz Chord C3 v1.wav"), _rec("Qxvwz Chord C3 v2.wav"),
           _rec("Qxvwz Chord B2 v2.wav", pin=True), _rec("Qxvwz Chord D3 v2.wav")]
    groups, n = C._one_per_retune_target([[0, 1, 2, 3]], rec)
    # a Keep (B2, +1) is the set's note on C3: the other notes there go, layers and all
    assert n == 3 and groups == [[2]]
    rec[2]["pin"] = False
    groups, n = C._one_per_retune_target([[0, 1, 2, 3]], rec)
    assert n == 2 and groups == [[0, 1]]                       # both layers of C3 (the sibling cap picks)


def test_one_per_c_leaves_other_sets_and_unretuned_files_alone():
    rec = [_rec("Qxvwz Chord F#2.wav"), _rec("Qxvwz Chord A#2.wav"),
           dict(path=os.path.join(PACK, "Northwind Stab F#2.wav"), tune=None, retune=None),
           dict(path=os.path.join(PACK, "Northwind Stab A#2.wav"), tune=None, retune=None),
           _rec("Vqx Pluck A#2.wav", category="STABS")]
    groups, n = C._one_per_retune_target([[0, 1, 2, 3, 4]], rec)
    assert n == 1 and groups == [[1, 2, 3, 4]]                 # A#2 (+2) over F#2 (+6)


def test_one_per_c_drops_an_emptied_folder():
    rec = [_rec("Qxvwz Chord C3.wav"), _rec("Qxvwz Chord A#2.wav"), _rec("Qxvwz Chord D3.wav")]
    groups, n = C._one_per_retune_target([[0], [1, 2]], rec)
    assert n == 2 and groups == [[0]]


# --- 3. a folder's note is what its files play ------------------------------------------------

def test_folder_note_follows_the_retuned_files():
    names = ["Qxvwz Mallet G2.wav", "Qxvwz Mallet A2.wav", "Qxvwz Mallet A#2.wav",
             "Qxvwz Mallet B2.wav", "Qxvwz Mallet C3.wav", "Qxvwz Mallet D3.wav"]
    recs = [_rec(n, category="ACOUSTIC") for n in names]
    assert C._folder_played_note(recs) == 48.0                 # all play C3 (sources: median A#2)
    # mostly kept at their own pitch (too far for the cap): the median of what they play
    far = [dict(path=os.path.join(PACK, f"Qzv Horn {n}.wav"), tune=None, retune=None)
           for n in ("F#2", "G2", "G2", "F2", "F#2")]
    assert C._folder_played_note(far + recs[:2]) == 43.0      # G2: two retuned to C3 don't decide it
    assert C._folder_played_note(recs[:4]) is None             # under 5 known notes


def test_numbered_takes_with_a_measured_root_all_stay():
    rec = [dict(path=os.path.join(PACK, f"Qzv Saw Lead {n:02d}.wav"), tune=t, retune=rt, root_src="detect")
           for n, t, rt in ((2, 43.0, 5.0), (5, 45.0, 3.0), (7, 50.0, -2.0))]
    groups, n = C._one_per_retune_target([[0, 1, 2]], rec)
    assert n == 0 and groups == [[0, 1, 2]]
