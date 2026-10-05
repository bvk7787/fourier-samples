"""The resolver (metadata/resolve.py): with Sononym it chooses exactly what curation chose
before it existed, it names each value's source, and it sets Sononym's readings beside
Fourier's own."""
import random
from types import SimpleNamespace

import pytest

from fourier.metadata import resolve as R
from fourier.metadata import rows
from fourier.metadata.providers import Active


def _row(**kw):
    base = dict(id=1, filename="Loop 01.wav", rel_path="Acme/Loops/Loop 01.wav", path="/l/Loop 01.wav",
                duration_s=8.0, bpm=None, tempo_bpm=None, bpm_reliable=None, base_note=None,
                base_note_confidence=None, root_note=None, acid_bpm=None, acid_beats=None,
                classes=["Loop"])
    base.update(kw)
    return SimpleNamespace(**base)


# --- the build's choices, as curation made them before the resolver (packs/curate._mkrec) ---

def _old_tune(r, fallback):
    tune = (r.base_note if (r.base_note is not None and r.base_note_confidence is not None
            and r.base_note_confidence > 0.4 and 12 <= r.base_note <= 84) else None)
    chunk = (fallback and tune is None and getattr(r, "root_note", None) is not None
             and 12 <= r.root_note <= 108)
    if chunk:
        tune = float(r.root_note)
    return tune, chunk


def _old_oneshot_bpm(r):
    return r.tempo_bpm if (r.bpm_reliable and r.tempo_bpm) else None


@pytest.mark.parametrize("fallback", [False, True])
def test_the_root_and_a_one_shots_tempo_are_what_curation_chose(fallback):
    rng = random.Random(7)
    pick = lambda *xs: rng.choice(xs)        # noqa: E731
    for _ in range(3000):
        r = _row(base_note=pick(None, 11.0, 12.0, 48.3, 84.0, 85.0),
                 base_note_confidence=pick(None, 0.39, 0.4, 0.41, 0.9),
                 root_note=pick(None, 11, 12, 60, 108, 109),
                 tempo_bpm=pick(None, 0.0, 87.0, 174.0), bpm_reliable=pick(None, 0, 1, True, False))
        tune, chunk = _old_tune(r, fallback)
        got = R.root(r, fallback)
        assert got.value == tune and (got.source == R.CHUNK) == chunk
        assert (got.source is None) == (tune is None)
        assert R.oneshot_tempo(r).value == _old_oneshot_bpm(r)
        assert bool(R.tempo_reliable(r).value) == bool(r.bpm_reliable)


def test_a_loops_tempo_is_curations_chain_with_its_source():
    from fourier.packs import curate as C
    cases = [
        (_row(filename="Break 174bpm.wav"), False, (174.0, "name", R.NAME)),
        (_row(bpm=87.0, duration_s=11.034), False, None),           # Sononym's, doubled to whole bars
        (_row(tempo_bpm=120.0, duration_s=8.0), False, (120.0, "librosa", R.AUDIO)),
        (_row(bpm=120.0, duration_s=8.0), True, (120.0, "sononym", R.NAME)),   # the path's, without Sononym
        (_row(acid_bpm=96.0, duration_s=10.0), True, (96.0, "acid", R.ACID)),
    ]
    for r, fb, want in cases:
        got, chain = R.loop_tempo(r, fb)
        old = C._resolve_tempo(r.filename, r.duration_s, r.bpm, r.tempo_bpm,
                               C._fallback_tempos(r) if fb else None)
        assert (got.value, chain) == old
        if want:
            assert (got.value, chain, got.source) == want
        else:
            assert got.source == R.SONONYM and chain == "sononym"


def test_a_stated_acid_tempo_beats_the_audio_estimate_without_sononym():
    """Without Sononym the WAV's ACID tempo (stated) comes before Fourier's own estimate, after
    a tempo the name or path states; with Sononym (no fallback tempos) the chain is as it was."""
    r = _row(acid_bpm=96.0, tempo_bpm=120.0, duration_s=8.0)      # 120 makes it 4 bars, 96 doesn't
    got, chain = R.loop_tempo(r, True)
    assert (got.value, chain, got.source) == (96.0, "acid", R.ACID)
    got, chain = R.loop_tempo(r, False)                             # with Sononym: never read
    assert (got.value, chain, got.source) == (120.0, "librosa", R.AUDIO)
    named = _row(filename="Loop 01 100bpm.wav", acid_bpm=96.0, tempo_bpm=120.0, duration_s=9.6)
    assert R.loop_tempo(named, True)[0].value == 100.0              # the name's comes first
    worded = _row(filename="Loop bpm 90.wav", acid_bpm=96.0, tempo_bpm=120.0, duration_s=8.0)
    assert R.loop_tempo(worded, True)[0].value == 90.0              # "bpm 90" is stated too


def test_own_tonal_is_what_the_key_step_reads():
    assert R.own_tonal(0.7, 1.6) and not R.own_tonal(0.7, 1.4) and not R.own_tonal(0.5, 2.5)
    assert not R.own_tonal(None, 2.0)
    assert f">= {R.OWN_TONAL_HPR}" in R.KEY_CANDIDATE_SQL and f">= {R.OWN_TONAL_CHROMA}" in R.KEY_CANDIDATE_SQL


def test_tempo_sources():
    assert R.tempo_source("sononym", False) == R.SONONYM
    assert R.tempo_source("sononym", True) == R.NAME       # without Sononym: the path's tempo
    assert [R.tempo_source(s, False) for s in ("name", "librosa", "acid", "folder", "length")] == \
        [R.NAME, R.AUDIO, R.ACID, R.FOLDER, R.LENGTH]
    assert R.tempo_source(None, False) is None


def test_the_why_log_records_where_a_picks_tempo_and_root_came_from():
    assert R.recorded({"bpm": 174.0, "bpm_src": "sononym"}, False) == {"tempo": R.SONONYM}
    assert R.recorded({"bpm": 120.0, "bpm_src": None}, False) == {"tempo": R.AUDIO}
    assert R.recorded({"bpm": None, "retune": 2, "root_src": "detect"}, False) == {"root": R.SONONYM}
    assert R.recorded({"retune": 0, "root_src": "pyin"}, False) == {"root": R.AUDIO}
    assert R.recorded({"retune": 1, "root_src": "chunk"}, True) == {"root": R.CHUNK}
    assert R.recorded({"retune": None, "root_src": "name"}, False) == {}


# --- Fourier's own readings ---

def test_own_tempo_confidence_is_how_close_the_loop_is_to_whole_bars():
    assert R.own_tempo_confidence(8.0, 120.0) == 1.0          # 4 bars
    assert R.own_tempo_confidence(8.0, 60.0) == 1.0           # 2 bars, or 4 at the double
    assert R.own_tempo_confidence(8.0, 123.0) == 0.6          # 16.4 beats: 0.4 off at x1
    assert R.own_tempo_confidence(8.0, 105.0) == 0.0          # 14 beats: 7, 14 and 28 fit no bars
    assert R.own_tempo_confidence(None, 120.0) is None and R.own_tempo_confidence(8.0, None) is None


def test_own_pitched_trusts_pyin_when_it_ran():
    assert R.own_pitched(0.9, 3.0) is True
    assert R.own_pitched(0.3, 3.0) is False and R.own_pitched(0.9, 1.2) is False
    assert R.own_pitched(None, 3.0) is None
    assert R.own_pitched(0.9, 3.0, root_midi=None, root_at="2026-01-01") is False
    assert R.own_pitched(0.1, 1.0, root_midi=60.0, root_at="2026-01-01") is True


def test_sononyms_pitched_is_the_derived_rule():
    from fourier.analysis.derived import compute_derived
    for har in (None, 0.5, 0.65, 0.75):
        for conf in (None, 0.4, 0.6):
            meta = SimpleNamespace(harmonicity=har, pitch_confidence=conf, brightness=None,
                                   noisiness=None, crest_factor=None, classes=None,
                                   class_signature=None, bpm=None, bpm_confidence=None,
                                   categories=None)
            want = compute_derived(meta)["is_pitched"]
            got = R.sononym_pitched(har, conf)
            assert (None if want is None else bool(want)) == got


def test_tempos_agree_across_an_octave():
    assert R.tempos_agree(87.0, 174.0) and R.tempos_agree(174.0, 87.0) and R.tempos_agree(175.0, 174.0)
    assert not R.tempos_agree(130.0, 174.0)


def _readings(son=None, own=None):
    s = dict(shape=None, tempo=None, pitched=None, root=None, key=None)
    o = dict(shape=None, tempo=None, tempo_confidence=None, pitched=None, root=None, key=None)
    s.update(son or {})
    o.update(own or {})
    return R.Readings(1, "Acme/Loops/Loop 01.wav", s, o)


def test_differences_show_both_readings():
    rd = _readings({"shape": "loop", "tempo": 174.0}, {"shape": "loop", "tempo": 130.0})
    assert R.differences(rd) == [("tempo", "174 (Sononym) / 130 (Fourier's own analysis)")]
    assert R.differences(_readings({"shape": "loop", "tempo": 174.0},
                                   {"shape": "loop", "tempo": 87.0})) == []    # an octave apart
    assert R.differences(_readings({"shape": "oneshot", "tempo": 174.0},
                                   {"shape": "oneshot", "tempo": 120.0})) == []   # a one-shot's tempo
    got = dict(R.differences(_readings({"shape": "loop", "pitched": True, "root": 60.0},
                                       {"shape": "oneshot", "pitched": False, "root": 72.4})))
    assert got == {"shape": "loop (Sononym) / one-shot (Fourier's own analysis)",
                   "pitched": "pitched (Sononym) / unpitched (Fourier's own analysis)"}  # C4 vs C5: same note
    assert dict(R.differences(_readings({"root": 60.0}, {"root": 62.0}))) == {
        "root": "C4 (Sononym) / D4 (Fourier's own analysis)"}
    assert R.differences(_readings({"shape": "loop"}, {})) == []      # one reading: nothing to compare


# --- every field the build reads has one source ---

SONONYM_LIVE = Active(("sononym", "ableton"), ("sononym",), ("ableton",), "auto")
FALLBACK = Active(("path", "audio"), ("path", "audio"), (), "auto")


@pytest.mark.parametrize("act", [SONONYM_LIVE, FALLBACK], ids=["sononym", "fallback"])
def test_every_field_names_its_source(act):
    for f in rows.FIELDS:
        src = rows.field_source(f, act)
        assert src is None or src in ("file", "ableton") or src.split("+")[0] in R.SOURCES, (f, src)


def test_field_sources():
    fs = rows.field_source
    assert fs("bpm", SONONYM_LIVE) == R.SONONYM and fs("bpm", FALLBACK) == R.NAME
    assert fs("harmonicity", SONONYM_LIVE) == R.SONONYM and fs("harmonicity", FALLBACK) == R.AUDIO
    assert fs("brightness", FALLBACK) is None                  # no fallback provider measures it
    assert fs("bpm_reliable", FALLBACK) == R.SONONYM == fs("sub_weight", SONONYM_LIVE)
    assert fs("tempo_bpm", SONONYM_LIVE) == R.AUDIO == fs("own_root_midi", FALLBACK)
    assert fs("root_note", FALLBACK) == R.CHUNK and fs("acid_bpm", FALLBACK) == R.ACID
    assert fs("sononym_bpm", FALLBACK) is None and fs("sononym_bpm", SONONYM_LIVE) == R.SONONYM
    assert fs("ableton_tags", FALLBACK) is None and fs("ableton_tags", SONONYM_LIVE) == "ableton"
    assert fs("duration_s", FALLBACK) == "file"
    with pytest.raises(KeyError):
        fs("nonsense", FALLBACK)
