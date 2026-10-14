"""A stated tempo wins at whole beats, whatever the bar count: a name's ("126.0", a bare
number an analyzer agrees with), a folder's ("126_BPM", "Loops (126BPM)"); names that mark a
phrase by a key then a tempo or a leading tempo; and a late start on silence moved to the
front, for phrases too (packs/curate._resolve_tempo, BPM_NAME_RE, _rotate_and_slice)."""
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from fourier.packs import curate as C
from fourier.packs import why as W
from fourier.packs.curate_config import BPM_NAME_RE

SR = 44100


def _beats(n, bpm):
    return n * 60.0 / bpm


# --- odd bar counts --------------------------------------------------------------------------

def test_a_bare_name_tempo_at_an_odd_bar_count_wins_when_an_analyzer_agrees():
    # 5 bars at the name's 126: Sononym's 100.8 makes it 4 bars, librosa hears 125
    dur = _beats(20, 126)
    assert C._resolve_tempo("Qxvwz_126_Fm_Vqx.wav", dur, 100.8, 125.1) == (126.0, "name")
    # ...an analyzer at double or half time agrees too
    assert C._resolve_tempo("Qxvwz_126_Fm_Vqx.wav", dur, 100.8, 251.0) == (126.0, "name")
    # a take number: 5 beats at 75 of a 2-bar loop at 120 that both analyzers hear
    assert C._resolve_tempo("Qxvwz_Groove_075.wav", _beats(8, 120), 120.0, 119.5) == (120.0, "sononym")
    # whole named bars need no analyzer, as before
    assert C._resolve_tempo("Qxvwz_126_Fm_Vqx.wav", _beats(16, 126), 100.8, None) == (126.0, "name")


def test_an_explicit_name_tempo_wins_at_any_whole_beats():
    assert C._resolve_tempo("Qzv Keys 126bpm.wav", _beats(20, 126), 100.8, None) == (126.0, "name")
    assert C._resolve_tempo("Qzv Arp 104bpm.wav", _beats(6, 104), 138.7, 141.0) == (104.0, "name")


def test_a_decimal_name_tempo_is_explicit():
    assert C._filename_tempos("Qzv_Lead_09_126.0_A.wav") == [(126.0, True)]
    # half a bar at 126, which no analyzer reads
    assert C._resolve_tempo("Qzv_Lead_09_126.0_A.wav", _beats(2, 126), None, 161.5) == (126.0, "name")
    assert C._filename_tempos("Qzv_Lead_2.5_Vqx.wav") == []


# --- folder tempos ---------------------------------------------------------------------------

@pytest.mark.parametrize("folder, want", [
    ("Loops (126BPM)/WAV/Arps", [126.0]),
    ("126_BPM", [126.0]),
    ("BPM 96", [96.0]),
    ("Acme 128 bpm/Loops/90 BPM", [90.0, 128.0]),          # the nearest first
    ("Drum Loops/70 - 95 bpm", []),                         # a range
    ("Loops/120-130 BPM", []),
    ("Loops/174", []),                                       # number-only: the fallback's
    ("Qx 300 BPM", []),
])
def test_folder_tempos(folder, want):
    assert C._folder_tempos(f"/l/SampleLibrary/Acme/{folder}/Qx_31_F.wav") == want


def test_a_folder_tempo_wins_at_whole_beats():
    p = "/l/SampleLibrary/Acme/Qx Pack/Loops (126BPM)/WAV/Qx_31_F.wav"
    # 10 bars at the folder's 126, read at 141 and 100.8
    assert C._resolve_tempo("Qx_31_F.wav", _beats(40, 126), 141.0, 100.8, path=p) == (126.0, "folder")
    # 4 beats at 126, which Sononym reads at 168.5
    q = "/l/SampleLibrary/Acme/Qx Pack/126_BPM/512_QX.wav"
    assert C._resolve_tempo("512_QX.wav", _beats(4, 126), 168.5, 147.0, path=q) == (126.0, "folder")
    # the name's explicit tempo comes first; a length the folder's tempo misses falls through
    assert C._resolve_tempo("Qx 100bpm.wav", _beats(8, 100), None, None, path=p) == (100.0, "name")
    assert C._resolve_tempo("Qx_31_F.wav", _beats(16, 100), 100.0, None, path=p) == (100.0, "sononym")
    from fourier.metadata.resolve import FOLDER, tempo_source
    assert tempo_source("folder", False) == FOLDER


def test_a_number_only_folder_without_sononym_takes_whole_beats_too():
    Row = namedtuple("Row", "filename rel_path path duration_s acid_bpm acid_beats tempo_bpm")
    r = Row("Qx Seq.wav", "Acme/Loops/120/Qx Seq.wav", "/l/SampleLibrary/Acme/Loops/120/Qx Seq.wav",
            _beats(20, 120), None, None, None)
    assert C._resolve_tempo(r.filename, r.duration_s, None, None, C._fallback_tempos(r)) == (120.0, "folder")


def test_why_says_when_the_bars_are_odd(tmp_path):
    path = "/lib/SampleLibrary/V/P/Qxvwz_126_Fm_Vqx.wav"
    r = SimpleNamespace(id=1, rel_path="V/P/Qxvwz_126_Fm_Vqx.wav", path=path, filename="Qxvwz_126_Fm_Vqx.wav",
                        categories="[]", classes='["Loop"]', ableton_tags="[]", harmonicity=0.8,
                        chroma_concentration=None, n_events=20, duration_s=_beats(20, 126), file_hash="h",
                        bpm=100.8, tempo_bpm=125.1)
    w = W.explain_row(r, store_path=str(tmp_path / "none.json"))
    assert w.tempo[0] == 126.0
    assert "126 BPM (from its name; 20 beats, 5 bars)" in "\n".join(W.format_why(w))
    r.duration_s = _beats(16, 126)
    assert "beats" not in "\n".join(W.format_why(W.explain_row(r, store_path=str(tmp_path / "none.json"))))


# --- names that mark a phrase ----------------------------------------------------------------

@pytest.mark.parametrize("name, marked", [
    ("qxvwz_Fm_126.wav", True),             # a key then a tempo
    ("vqx_Abm_122.wav", True),
    ("Qzv_C#m_128_Lead.wav", True),
    ("Qzv Pad Gmaj 90.wav", True),
    ("110 - Qxvwz Vqx.wav", True),          # a leading tempo, then a spaced dash
    ("Lead_174_Amin.wav", True),            # (a tempo then a key, as before)
    ("Qx FM 77 Kick.wav", False),           # FM synthesis, not F minor
    ("DM_88_Qzv.wav", False),               # a prefix, not D minor
    ("60 Warm Pad C3.wav", False),          # a MIDI note number
    ("Kick 01 - Hard.wav", False),
    ("120-Qx.wav", False),
    ("Qx_Gm_1300.wav", False),
])
def test_a_key_then_a_tempo_or_a_leading_tempo_marks_a_phrase(name, marked):
    assert bool(BPM_NAME_RE.search(name)) is marked


# --- late starts on silence -----------------------------------------------------------------

def _phrase(path, lead_s, beats=8, bpm=120, head=None):
    """A chord phrase whose first note starts `lead_s` in, after digital silence (or `head`:
    a quiet sound there); `beats` beats long overall."""
    n = int(round(SR * _beats(beats, bpm)))
    y = np.zeros(n)
    i0 = int(round(SR * lead_s))
    t = np.arange(n - i0) / SR
    y[i0:] = 0.4 * np.exp(-((t * bpm / 60) % 2) * 1.5) * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t))
    if head is not None:
        y[:i0] = head * np.sin(2 * np.pi * 330 * np.arange(i0) / SR)
    sf.write(str(path), y, SR, subtype="PCM_24")
    return n


def test_a_phrase_late_on_silence_rotates_to_its_first_sound(tmp_path):
    p = tmp_path / "Qzv Chords 120bpm.wav"
    n = _phrase(p, 0.036)
    out = C._rotate_and_slice(p, 120.0, "/l/Acme/Qzv Chords 120bpm.wav", phrase=True)
    assert set(out) == {"rotate_ms"} and 34 <= out["rotate_ms"] <= 36       # 1 ms kept ahead of it
    y, _sr = sf.read(str(p))
    assert len(y) == n and np.abs(y[:int(0.0009 * SR)]).max() < 1e-4 and np.abs(y[:int(0.003 * SR)]).max() > 0.01
    # a floor far under LOOP_QUIET_DB (hum, dither) ahead of it is silence too
    q = tmp_path / "Qzv Keys 120bpm.wav"
    _phrase(q, 0.030, head=10 ** (-80 / 20))
    assert 28 <= C._rotate_and_slice(q, 120.0, "/l/Acme/Qzv Keys 120bpm.wav", phrase=True)["rotate_ms"] <= 30


@pytest.mark.parametrize("lead, head", [
    (0.036, 10 ** (-50 / 20)),     # a quiet pickup or swell isn't silence: its own lead-in
    (0.006, None),                 # under ROTATE_MIN_MS
    (0.120, None),                 # a 16th or more: a phrase that starts off the beat
])
def test_a_phrases_musical_lead_in_stays(tmp_path, lead, head):
    p = tmp_path / "Qzv Lead 120bpm.wav"
    _phrase(p, lead, head=head)
    before = p.read_bytes()
    assert C._rotate_and_slice(p, 120.0, "/l/Acme/Qzv Lead 120bpm.wav", phrase=True) == {}
    assert p.read_bytes() == before


def test_a_phrase_whole_beats_only_without_its_silent_lead_in_is_cut(tmp_path):
    p = tmp_path / "Qzv Bass 120bpm.wav"
    n = _phrase(p, 0.020, beats=8)
    y, _sr = sf.read(str(p))
    sf.write(str(p), np.concatenate([np.zeros(int(0.020 * SR)), y[int(0.020 * SR):], np.zeros(int(0.020 * SR))]),
             SR, subtype="PCM_24")
    out = C._rotate_and_slice(p, 120.0, "/l/Acme/Qzv Bass 120bpm.wav", phrase=True)
    assert 18 <= out["trim_ms"] <= 20 and sf.info(str(p)).frames < n + int(0.020 * SR)


def test_a_swung_loop_behind_a_faint_floor_still_rotates(tmp_path, monkeypatch):
    """Digital silence, then a floor 72 dB down from 4 ms, then soft 5 ms attacks from 15.5 ms:
    the onset detector's steepest rise is the floor's start, which read as an on-time
    downbeat; the first sound over LOOP_QUIET_DB stands for the downbeat instead."""
    rng = np.random.default_rng(1)
    bpm, beats = 97.0, 16
    n = int(round(_beats(beats, bpm) * SR))
    y = np.zeros(n)
    six = 60 / bpm / 4

    def hit(t, amp, dec, f):
        i, m = int(t * SR), int(0.3 * SR)
        tt = np.arange(m) / SR
        env = np.minimum(1, tt / 0.005) * np.exp(-np.maximum(0, tt - 0.005) / dec)
        s = amp * env * (np.sin(2 * np.pi * f * tt) + 0.3 * rng.standard_normal(m))
        y[i:i + m] += s[:max(0, min(m, n - i))]
    for b in range(beats):
        hit(b * 60 / bpm + 0.0155, 0.8, 0.08, 60)
        for k in range(4):
            hit(b * 60 / bpm + 0.0155 + k * six + (0 if k % 2 == 0 else 0.24 * six), 0.2, 0.02, 3000)
    y[int(0.004 * SR):] += 10 ** (-73 / 20) * rng.standard_normal(n - int(0.004 * SR))
    y[:int(0.0155 * SR)] = np.where(np.arange(int(0.0155 * SR)) < int(0.004 * SR), 0.0, y[:int(0.0155 * SR)])
    p = tmp_path / "swung.wav"
    sf.write(str(p), y / np.abs(y).max() * 0.89, SR, subtype="PCM_24")
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 4.0, "bar_err": 0.0, "swing": 62.0})
    out = C._rotate_and_slice(p, bpm, "/l/Acme/Qxvwz Vqx 97bpm.wav")
    assert 12 <= out["rotate_ms"] <= 15.5 and sf.info(str(p)).frames == n
