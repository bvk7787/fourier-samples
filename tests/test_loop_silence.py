"""Loops that are mostly silence (an ending take, a fill that stops, an empty last bar) are left
out of the loop categories (curate._silent_share, _mostly_silent), and a fill's or a swung
loop's late start behind silence is rotated (or cut) onto its downbeat."""
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from fourier.packs import curate as C
from fourier.packs import why_log

SR = 44100


def _hit(n):
    return 0.6 * np.exp(-np.arange(n) / (SR * 0.03)) * np.sin(np.arange(n) * 0.3)


def _loop(path, bpm, beats, hits, lead_s=0.0, floor_db=None):
    """`beats` beats at bpm with a hit at each beat position in `hits` (in beats), after
    `lead_s` of silence (or a noise floor at floor_db dBFS throughout)."""
    b = 60.0 / bpm
    y = np.zeros(int(round(SR * (lead_s + beats * b))))
    for t in hits:
        o = int(round(SR * (lead_s + t * b)))
        n = min(int(SR * 0.25), len(y) - o)
        y[o:o + n] += _hit(n)
    if floor_db is not None:
        y += 10 ** (floor_db / 20) * np.random.default_rng(0).standard_normal(len(y))
    sf.write(str(path), y, SR, subtype="PCM_24")
    return path


def test_mostly_silent_loops_measure_over_the_limit(tmp_path):
    eighths = [k / 2 for k in range(32)]
    # a fill that stops after one bar of a two-bar loop, over a quiet noise floor
    fill = _loop(tmp_path / "fill.wav", 120, 8, [k / 4 for k in range(16)], floor_db=-70)
    # one note, then three bars of nothing
    one = _loop(tmp_path / "one.wav", 120, 16, [0])
    # a 4-bar loop whose middle two bars are empty (a gap inside, not at its ends)
    hole = _loop(tmp_path / "hole.wav", 120, 16, [t for t in eighths if t < 4 or t >= 12])
    for p in (fill, one, hole):
        s = C._silent_share(str(p))
        assert s > C.LOOP_SILENT_SHARE_MAX, (p.name, s)
    assert C._silent_share(str(one)) > 0.85


def test_sparse_grooves_stay_under_the_limit(tmp_path):
    # offbeat hats: an eighth of silence before the first hit, then a hit every beat
    tops = _loop(tmp_path / "tops.wav", 125, 16, [k + 0.5 for k in range(16)])
    # a half-time break: kick on 1, snare on 3, a ghost note before it, two bars
    half = _loop(tmp_path / "half.wav", 172, 8, [0, 1.75, 2, 4, 5.5, 6])
    # a 4-bar loop with one empty beat at its end (a rest, not an empty bar)
    rest = _loop(tmp_path / "rest.wav", 120, 16, [k / 2 for k in range(30)])
    for p in (tops, half, rest):
        s = C._silent_share(str(p))
        assert s < C.LOOP_SILENT_SHARE_MAX, (p.name, s)


def test_unreadable_and_all_zero_files(tmp_path):
    assert C._silent_share(str(tmp_path / "missing.wav")) == 0.0       # can't read: no verdict
    z = tmp_path / "zero.wav"
    sf.write(str(z), np.zeros(SR), SR, subtype="PCM_16")
    assert C._silent_share(str(z)) == 1.0


def test_only_loop_categories_are_held_to_it(tmp_path):
    one = _loop(tmp_path / "one.wav", 120, 16, [0])
    r = SimpleNamespace(path=str(one))
    got = C._mostly_silent(r, {"kind": "loop"})
    assert got is not None and got[1] == C.LOOP_SILENT_SHARE_MAX and got[0] > 0.85
    assert C._mostly_silent(r, {"kind": "oneshot"}) is None
    assert "mostly silence" in why_log.describe("mostly_silent", got, {"category": "PHRASES"})
    assert "mostly_silent" in why_log.FILTER_WORDS


def test_selection_leaves_a_mostly_silent_loop_out(tmp_path, monkeypatch):
    import fourier.packs.ratings as ratings
    monkeypatch.setattr(ratings, "keep_pins", lambda *a, **k: {})
    monkeypatch.setattr(ratings, "misfiled_map", lambda *a, **k: {})
    monkeypatch.setattr(C, "_misfiled_index", lambda m, e, i, pid=None: _MisfiledIndex())
    ok = _loop(tmp_path / "Qxvwz Groove 120bpm.wav", 120, 16, [k / 2 for k in range(32)])
    end = _loop(tmp_path / "Qxvwz Ending 120bpm.wav", 120, 16, [k / 2 for k in range(8)])
    rows = [SimpleNamespace(id=i, rel_path=f"Acme/Acme Loops/{p.name}", path=str(p), filename=p.name,
                            file_hash=None, file_format="wav", duration_s=8.0, ableton_tags=None,
                            brightness=0.5, noisiness=0.3, harmonicity=0.7, crest_factor=4.0,
                            base_note=None, base_note_confidence=None, bpm=120.0, classes='["Loop"]',
                            attack_time_ms=10.0, decay_time_ms=300.0, sub_weight=0.1, is_clipped=0,
                            tempo_bpm=None, bpm_reliable=1, onset_rate_hz=4.0, chroma_concentration=1.0,
                            spectral_flatness_mean=0.3, rms_mean=0.1, dc_offset_ratio=0.0)
            for i, p in enumerate((ok, end))]
    cfg = {k: v for k, v in C.CATEGORIES["PHRASES"].items() if k not in ("anti", "dir_cap")}
    emb = np.eye(2, 16, dtype="float32")
    why: dict = {}
    rec, _fm, stats = C._select_records(rows, cfg, emb, {0: 0, 1: 1}, why=why)
    assert [d["path"] for d in rec] == [str(ok)]
    assert stats["mostly_silent"] == 1 and list(why["mostly_silent"]) == [1]


class _MisfiledIndex:
    def categories(self, path, vec=None):
        return set()


# --- late starts behind silence ----------------------------------------------------------------

def _late(path, bpm, beats, lead_s, extra_s=0.0):
    """A 16th-note groove `lead_s` late behind digital silence; the file is `beats` beats long
    plus `extra_s` (the lead-in on top of whole beats when extra_s == lead_s)."""
    six = 60.0 / bpm / 4
    n = int(round(SR * (beats * 4 * six + extra_s)))
    y = np.zeros(n)
    for k in range(beats * 4):
        i = int(round(SR * (k * six + lead_s)))
        m = min(n - i, 1500)
        if m > 0:
            y[i:i + m] += np.exp(-np.arange(m) / 150.0) * np.sin(np.arange(m) * 0.4)
    sf.write(str(path), y * 0.5, SR, subtype="PCM_16")
    return n


def test_a_swung_or_fill_loop_behind_silence_rotates(tmp_path, monkeypatch):
    p = tmp_path / "late.wav"
    n = _late(p, 120, 8, 0.022)
    # swung: its hits off the grid are feel, but the silence ahead of the downbeat is a cut
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 2.0, "bar_err": 0.0, "swing": 62.0})
    out = C._rotate_and_slice(p, 120.0, "/l/Acme/Qxvwz Beat 120.wav")
    assert 20 <= out["rotate_ms"] <= 22 and sf.info(str(p)).frames == n
    assert np.abs(sf.read(str(p))[0][:int(0.001 * SR)]).max() < 0.05      # starts at its downbeat
    q = tmp_path / "fill.wav"
    _late(q, 120, 8, 0.022)
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 2.0, "bar_err": 0.0, "swing": None})
    assert 20 <= C._rotate_and_slice(q, 120.0, "/l/Acme/Qxvwz Vqx Fill 120.wav")["rotate_ms"] <= 22


def test_a_swung_loops_offset_hits_are_still_left_alone(tmp_path, monkeypatch):
    """Only the quiet head counts for a swung loop: hits late with no silence ahead (the first
    one wrapped round to the start) are its feel."""
    p = tmp_path / "feel.wav"
    six = 60.0 / 120 / 4
    n = int(round(SR * 32 * six))
    y = np.zeros(n)
    for k in range(32):
        i = int(round(SR * (k * six + 0.022)))
        idx = (i + np.arange(1500)) % n
        y[idx] += np.exp(-np.arange(1500) / 150.0) * np.sin(np.arange(1500) * 0.4)
    y[:int(0.022 * SR)] += 0.2 * np.sin(np.arange(int(0.022 * SR)) * 0.05)
    sf.write(str(p), y * 0.5, SR, subtype="PCM_16")
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 2.0, "bar_err": 0.0, "swing": 62.0})
    before = p.read_bytes()
    assert "rotate_ms" not in C._rotate_and_slice(p, 120.0, "/l/Acme/Qxvwz Beat 120.wav")
    assert p.read_bytes() == before


def test_a_lead_in_on_top_of_whole_beats_is_cut(tmp_path, monkeypatch):
    """8 beats and 18 ms of silence ahead: rotating keeps it 18 ms long each time round; the
    lead-in is cut, so it is whole beats from its downbeat."""
    p = tmp_path / "long.wav"
    n = _late(p, 117, 8, 0.018, extra_s=0.018)
    monkeypatch.setattr(C, "_slice_fields", lambda path, bpm: {"bars": 2.0, "bar_err": 0.036, "swing": 57.0})
    out = C._rotate_and_slice(p, 117.0, "/l/Acme/Qxvwz Beat 117.wav")
    assert "rotate_ms" not in out and 16 <= out["trim_ms"] <= 18
    frames = sf.info(str(p)).frames
    assert frames < n and C._whole_beats(frames / SR, 117.0, C.NAME_TEMPO_TOL_MS)
    assert np.abs(sf.read(str(p))[0][:int(0.0008 * SR)]).max() < 0.05
