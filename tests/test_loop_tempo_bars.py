"""A loop's tempo agrees with its length: a tempo its name states is taken only when the loop
is then whole beats (packs/curate._resolve_tempo), folding keeps it so, rotation needs it, and
verify checks it. Then the keys phrases.csv reads from bare note letters (packs/keys.py)."""
import json

import numpy as np
import pytest
import soundfile as sf

from fourier.packs import curate as C
from fourier.packs import keys as K
from fourier.packs.verify import FAIL, verify_master


def _beats(n, bpm, extra_s=0.0):
    return n * 60.0 / bpm + extra_s


# --- the name's tempo only where the length agrees ----------------------------------------------

@pytest.mark.parametrize("name, dur, son, lib, want", [
    ("Qxvwz Beat 120bpm.wav", _beats(16, 120), None, None, (120.0, "name")),
    ("Qzv Fill 96bpm.wav", _beats(2, 96), None, None, (96.0, "name")),            # a 2-beat fill
    ("Qzv Waltz 90bpm.wav", _beats(12, 90), None, None, (90.0, "name")),          # 3/4: whole beats
    ("Qxvwz Groove 100bpm.wav", _beats(16, 110), 110.0, None, (110.0, "sononym")),   # mislabelled
    ("Qxvwz Tops 127bpm.wav", _beats(8, 120), None, 120.0, (120.0, "librosa")),
    ("Qxvwz_Loop_072.wav", _beats(8, 128), 128.0, None, (128.0, "sononym")),    # a take number
    ("Qxvwz_Funk_128_Vqx.wav", _beats(8, 130), 130.0, None, (130.0, "sononym")),  # 0.12 beat off
    ("Qxvwz Odd 101bpm.wav", 7.2, None, 136.0, (None, None)),                      # 3 bars at 100: no fit
])
def test_a_name_tempo_needs_the_length_to_agree(name, dur, son, lib, want):
    assert C._resolve_tempo(name, dur, son, lib) == want


def test_a_break_cut_a_little_long_takes_the_tempo_its_length_gives():
    # a "96bpm" break that is one bar at 95.07: 24 ms over a bar at 96
    dur = _beats(4, 95.07)
    assert not C._name_tempo_fits(dur, 96.0, True)
    assert C._resolve_tempo("Qxvwz Break 96bpm.wav", dur, None, None) == (95.07, "name")
    # ... ahead of an analyzer that reads it at double time
    assert C._resolve_tempo("Qxvwz Perc 89bpm.wav", _beats(16, 88.74), 177.0, None) == (88.74, "name")
    # a bare number: after the analyzers (it may be a take number), then snapped the same way
    assert C._resolve_tempo("Qzv_Bass_137_Vqx.wav", _beats(32, 136.88), 91.5, None) == (136.88, "name")
    assert C._resolve_tempo("Qzv_Bass_137_Vqx.wav", _beats(32, 136.88), 136.9, None)[1] == "sononym"


def _clicks(path, bpm, beats, lead_s=0.0, tail_s=0.0, floor_db=None, sr=44100):
    """A loop of `beats` clicks at bpm with silence before the first and after the last beat
    (or a noise floor at floor_db dBFS throughout, as a recorded break has)."""
    y = np.zeros(int(round(sr * (lead_s + beats * 60.0 / bpm + tail_s))), dtype="float32")
    for b in range(beats):
        o = int(round(sr * (lead_s + b * 60.0 / bpm)))
        n = min(int(sr * 60.0 / bpm), len(y) - o)
        y[o:o + n] = 0.5 * np.exp(-np.arange(n) / (sr * 0.01)) * np.sin(np.arange(n) * 0.3)
    if floor_db is not None:
        y += (10 ** (floor_db / 20) * np.random.default_rng(0).standard_normal(len(y))).astype("float32")
    sf.write(path, y, sr, subtype="PCM_24")
    return len(y) / sr


def test_silence_before_the_downbeat_or_after_the_loop_keeps_the_name_tempo(tmp_path):
    lead = tmp_path / "Qxvwz Break 128bpm.wav"
    dur = _clicks(lead, 128, 8, lead_s=0.05)                     # 2 bars and 50 ms of silence first
    assert C._resolve_tempo(lead.name, dur, 128.0, None) != (128.0, "name")       # the length alone
    assert C._resolve_tempo(lead.name, dur, 128.0, None, path=str(lead)) == (128.0, "name")
    pad = tmp_path / "Qxvwz Break 174bpm.wav"
    dur = _clicks(pad, 174, 8, tail_s=0.06)                       # a silent pad after the last beat
    assert C._resolve_tempo(pad.name, dur, None, None, path=str(pad)) == (174.0, "name")
    # sound to the end: the length decides (a 96bpm break cut 1% long is still 95.07)
    full = tmp_path / "Qxvwz Break 96bpm.wav"
    dur = _clicks(full, 95.07, 4, floor_db=-70)
    assert C._resolve_tempo(full.name, dur, None, None, path=str(full)) == (95.07, "name")
    # a quiet decay isn't a pad: only digital silence at the end is
    assert C._quiet_ends(str(full))[1] < 0.001 and C._quiet_ends(str(pad))[1] > 0.06
    # a file exactly whole bars at the analyzers' tempo keeps its silence (a take number's loop
    # ending on a rest): 8 beats at 128, its last 60 ms silent, named 65
    take = tmp_path / "Qxvwz_Loop_065.wav"
    dur = _clicks(take, 128, 8)
    assert C._name_tempo_fits(dur, 65.0, False, str(take))
    assert C._resolve_tempo(take.name, dur, 128.0, None, path=str(take)) == (128.0, "sononym")
    assert C._quiet_ends(str(tmp_path / "gone.wav")) == (0.0, 0.0)


def test_the_tolerance_is_in_milliseconds_and_beats():
    assert C._name_tempo_fits(_beats(16, 120, 0.009), 120.0, True)
    assert not C._name_tempo_fits(_beats(16, 120, 0.012), 120.0, True)        # 12 ms over
    assert not C._name_tempo_fits(_beats(4, 65, 0.0577), 65.0, False)       # 0.06 beat, 58 ms
    assert C._name_tempo_fits(_beats(16, 120), 120.0, False)
    assert not C._name_tempo_fits(_beats(6, 120), 120.0, False)             # a bare number: whole bars
    assert C._name_tempo_fits(_beats(6, 120), 120.0, True)
    assert not C._whole_beats(_beats(0.5, 120), 120.0)                        # under one beat


def test_a_stated_bpm_word_without_sononym_needs_the_length_too():
    from collections import namedtuple
    Row = namedtuple("Row", "filename rel_path path duration_s acid_bpm acid_beats")
    ok = Row("Beat bpm124.wav", "Acme/Beat bpm124.wav", "/l/Acme/Beat bpm124.wav", _beats(8, 124), None, None)
    off = Row("Beat bpm124.wav", "Acme/Beat bpm124.wav", "/l/Acme/Beat bpm124.wav", 5.0, None, None)
    assert C._resolve_tempo(ok.filename, ok.duration_s, None, None, C._fallback_tempos(ok)) == (124.0, "name")
    assert C._resolve_tempo(off.filename, off.duration_s, None, None, C._fallback_tempos(off)) == (None, None)


def test_folding_never_leaves_half_a_beat():
    assert C._fold_keeps_beats(_beats(4, 180), 180.0, 90.0)          # one bar halved: 2 beats
    assert not C._fold_keeps_beats(_beats(3, 180), 180.0, 90.0)      # 1.5 beats
    assert C._fold_keeps_beats(_beats(3, 80), 80.0, 160.0)           # doubling always
    assert C._fold_keeps_beats(_beats(3, 180), 180.0, 180.0)


def test_rotation_skips_a_loop_that_isnt_whole_beats(tmp_path, monkeypatch):
    p = tmp_path / "loop.wav"
    y = (np.random.default_rng(1).standard_normal(int(44100 * 1.1)) * 0.1).astype("float32")   # 2.2 beats
    sf.write(p, y, 44100, subtype="PCM_24")
    before = p.read_bytes()
    monkeypatch.setattr(C, "_grid_offset", lambda *a, **k: 0.02)
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 1.0, "bar_err": 0.3, "swing": None})
    assert "rotate_ms" not in C._rotate_and_slice(p, 120.0, "/l/Acme/Qxvwz Beat.wav")
    assert p.read_bytes() == before
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 1.0, "bar_err": 0.0, "swing": None})
    assert C._rotate_and_slice(p, 120.0, "/l/Acme/Qxvwz Beat.wav").get("rotate_ms") == 20.0
    # a silent lead-in isn't off the grid: such a loop still rotates
    q = tmp_path / "late.wav"
    _clicks(q, 174, 8, lead_s=0.04)
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 2.0, "bar_err": 0.116, "swing": None})
    assert C._rotate_and_slice(q, 174.0, "/l/Acme/Qxvwz Beat.wav").get("rotate_ms") == 20.0


# --- verify ---------------------------------------------------------------------------------------

SR = 44100
LIB = "/lib/SampleLibrary"
CHECK = "loops are whole beats at their tempo"


def _loop_master(tmp_path, loops):
    """loops: [(category, filename, seconds, entry fields)] -> (master, store)."""
    root = tmp_path / "master"
    cats = {}
    for cat, fn, secs, extra in loops:
        d = root / cat / "fam"
        d.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(round(SR * secs))) / SR
        sf.write(d / fn, (0.3 * np.sin(2 * np.pi * 110 * t)).astype("float32"), SR, subtype="PCM_24")
        cats.setdefault(cat, {"families": 1, "files": 0, "source_samples": 0, "entries": []})
        cats[cat]["entries"].append({"family": "fam", "out": f"fam/{fn}", "src": f"{LIB}/Acme/{fn}", **extra})
        cats[cat]["files"] += 1
    for cat in cats:
        (root / cat / "_manifest.json").write_text(json.dumps(
            [{"family": "fam", "copied": cats[cat]["files"], "n": 9, "clap": None, "traits": []}]))
    (root / "manifest.json").write_text(json.dumps({"categories": cats}))
    store = tmp_path / "ratings.json"
    store.write_text(json.dumps({"version": 1, "ratings": {}}))
    return str(root), str(store)


def _verify(tmp_path, loops):
    root, store = _loop_master(tmp_path, loops)
    _ok, res = verify_master(root, session=None, store_path=store, audio=False, log=lambda *a: None)
    return {r.check: r for r in res}


def test_verify_checks_each_loops_bar_math(tmp_path):
    good = [("DRUMLOOPS", "Qxvwz Beat 120bpm.wav", _beats(16, 120), {"bpm": 120.0, "bpm_src": "name"}),
            ("DRUMLOOPS", "Qzv Break.wav", _beats(8, 87.3, 0.02), {"bpm": 87.3, "bpm_src": "sononym",
                                                                     "bpm_fold": 174.6}),
            ("PHRASES", "Qzv Fill 96bpm.wav", _beats(2, 96), {"bpm": 96.0, "bpm_src": "name"}),
            ("DRUMLOOPS", "Vqx Beat.wav", 5.0, {"bpm": 97.0, "bpm_src": "acid"})]     # as stated
    got = _verify(tmp_path / "a", good)
    assert got[CHECK].level != FAIL, got[CHECK].detail
    bad = [("DRUMLOOPS", "Qxvwz Groove 100bpm.wav", _beats(16, 110), {"bpm": 100.0, "bpm_src": "name"}),
           ("PHRASES", "Qzv Lead 96bpm.wav", _beats(4, 95.07), {"bpm": 96.0, "bpm_src": "name"}),
           ("DRUMLOOPS", "Qzv Roll.wav", _beats(3, 180), {"bpm": 180.0, "bpm_src": "sononym", "bpm_fold": 90.0})]
    got = _verify(tmp_path / "b", bad)
    assert got[CHECK].level == FAIL and got[CHECK].detail.startswith("3: ")
    assert all(fn in got[CHECK].detail for _c, fn, _s, _e in bad), got[CHECK].detail
    assert "16+0.00 beats at 100" not in got[CHECK].detail and "beats at 100" in got[CHECK].detail


# --- keys: a bare note letter where packs write the key ---------------------------------------------

@pytest.mark.parametrize("name, key", [
    ("Qxvwz_126_3c_A_Lead", (9, None)),          # after the tempo, a numbered variant between
    ("QX_124_E_Lead", (4, None)),
    ("Qzv Bass 134bpm F", (5, None)),             # the last token after the tempo
    ("Qzv Lead 72 Vqx C 02", (0, None)),          # ...but a take number
    ("Qzv_03_bpm_104_G", (7, None)),
    ("Qzv Loop [F Major 122 BPM]", (5, "major")),
    ("Qzv Cello GMaj 118bpm 7", (7, "major")),
    ("Qx_127_B_F#_Keys", (6, None)),              # a written key wins over a bare letter
    ("A_Loop_124", None),                         # before the tempo: a word
    ("Qzv_A_Loop_124", None),
    ("Qzv Bass C_106bpm", None),
    ("QX_124_E-Piano_Chords", None),              # "E-Piano" is no note
    ("C-07_Qzv", None),
    ("Qzv_Takes_B_21_E", None),                   # 20 is no tempo
    ("Qx_124_A_Loop_B", None),                    # two letters, two keys
    ("Qzv Loop A 01", None),
])
def test_bare_note_letters(name, key):
    assert K.name_key(name) == key


def _files(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"")
    K.forget()


def test_never_a_neighbours_key_over_the_files_own_letter(tmp_path):
    d = tmp_path / "Acme" / "Qxvwz Kit"
    _files(d, "Qxvwz_126_G#m_Bass.wav", "Qxvwz_126_G#_Pad.wav", "Qxvwz_126_3c_A_Lead.wav",
           "Qxvwz_Lead_B_v.wav", "Qxvwz_Drums.wav")
    rel = "Acme/Qxvwz Kit/"
    assert K.stated_key(str(d / "Qxvwz_126_3c_A_Lead.wav"), rel + "Qxvwz_126_3c_A_Lead.wav") == ((9, None), "name")
    # a letter it doesn't place (no tempo): no key, never the neighbours' G#
    assert K.stated_key(str(d / "Qxvwz_Lead_B_v.wav"), rel + "Qxvwz_Lead_B_v.wav") == (None, "")
    # no letter at all: the neighbours', where those that state one agree (G# and A disagree here)
    assert K.stated_key(str(d / "Qxvwz_Drums.wav"), rel + "Qxvwz_Drums.wav") == (None, "")
    e = tmp_path / "Acme" / "Qzv Kit"
    _files(e, "Qzv_126_G#m_Bass.wav", "Qzv_126_G#_Pad.wav", "Qzv_Drums.wav")
    assert K.stated_key(str(e / "Qzv_Drums.wav"), "Acme/Qzv Kit/Qzv_Drums.wav") == ((8, "minor"), "siblings")
