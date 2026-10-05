"""Regressions for curate.py pure logic (synthetic data, no real DB).

Each test reproduces a defect the code must not bring back.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from fourier.packs import curate
from fourier.packs.curate import (
    _resolve_tempo, _fold_tempo, _tempo_band_groups, _name_override, _MisfiledIndex,
)
from fourier.packs.curate_config import TEMPO_FOLD, TEMPO_BANDS


# ---------------------------------------------------------------------------
# Guards against: only the FIRST in-range filename number is ever tried
# ---------------------------------------------------------------------------
def test_resolve_tempo_tries_later_filename_numbers():
    # a 4-bar loop at 120 BPM is exactly 8.0 s. "78" (the CR-78 drum machine) is the
    # first in-range token; it fails the bar check, and the real tempo (120) that
    # follows is never considered -> with no analyzer BPM the loop goes "freetempo".
    assert _resolve_tempo("CR-78 Beat 120.wav", 8.0, None, None) == (120.0, "name")


# ---------------------------------------------------------------------------
# Guards against: _fold_tempo rounds AFTER folding, so it can return the excluded upper edge
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bpm", [179.996, 89.999])
def test_fold_tempo_stays_in_half_open_window(bpm):
    lo, hi = TEMPO_FOLD
    t = _fold_tempo(bpm, TEMPO_FOLD)
    assert lo <= t < hi, t


def test_folded_filename_tempo_is_not_banded_as_unknown():
    # "179.996bpm" is accepted verbatim by _filename_tempos (source "name" is not rounded),
    # folds to 180.0, and then falls outside every [lo, hi) band -> lands in the
    # unknown/freetempo group (whose folder then gets labeled "180bpm").
    bpm, src = _resolve_tempo("Loop 179.996bpm.wav", 4 * 4 * 60 / 179.996, None, None)
    assert src == "name"
    folds = [_fold_tempo(176.0, TEMPO_FOLD), _fold_tempo(bpm, TEMPO_FOLD)]
    # 179.996 ~ 180 == 90 an octave down: it must land in a real band, never "unknown"
    # (before the fix it folded to 180.0, outside every [lo, hi) band). Edges must cover the window.
    assert TEMPO_BANDS[0] <= TEMPO_FOLD[0] and TEMPO_BANDS[-1] >= TEMPO_FOLD[1]
    lo, hi = TEMPO_FOLD
    assert all(lo <= t < hi for t in folds)
    groups = _tempo_band_groups(folds, TEMPO_BANDS, 1)
    assert sorted(i for g in groups for i in g) == [0, 1] and len(groups) == 2


# ---------------------------------------------------------------------------
# Guards against: NAME_OVERRIDES substring "the clav" hijacks claves (percussion) into PIANO
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["The Clave 01.wav", "Hit The Clave.wav"])
def test_clave_is_not_forced_to_piano(name):
    assert _name_override(name) is None


# ---------------------------------------------------------------------------
# _select_records harness (ratings + misfiled index faked; no DB)
# ---------------------------------------------------------------------------
_ROW_DEFAULTS = dict(
    file_hash=None, file_format="wav", duration_s=2.0, ableton_tags=None,
    brightness=0.5, noisiness=0.3, harmonicity=0.7, crest_factor=4.0,
    base_note=None, base_note_confidence=None, bpm=None, classes='["OneShot"]',
    attack_time_ms=10.0, decay_time_ms=300.0, sub_weight=0.1, is_clipped=0,
    tempo_bpm=None, bpm_reliable=0, onset_rate_hz=0.5, chroma_concentration=1.0,
    spectral_flatness_mean=0.3, rms_mean=0.1, dc_offset_ratio=0.0,
)


def _row(i, vendor, fname):
    rel = f"{vendor}/{vendor} Pack/{fname}"
    return SimpleNamespace(id=i, rel_path=rel, path=f"/lib/SampleLibrary/{rel}",
                           filename=fname, **_ROW_DEFAULTS)


def _run_select(monkeypatch, rows, category, homes, keeps, misfiled, emb):
    import fourier.packs.ratings as ratings
    monkeypatch.setattr(ratings, "keep_pins", lambda *a, **k: dict(keeps))
    monkeypatch.setattr(ratings, "misfiled_map", lambda *a, **k: dict(misfiled))
    # the real _misfiled_index looks sample ids up in the DB; skip only that glue
    # (vec_of -> None means the name alone decides sibling-ness)
    monkeypatch.setattr(curate, "_misfiled_index",
                        lambda m, e, i, pid=None: _MisfiledIndex(m))
    id2row = {r.id: j for j, r in enumerate(rows)}
    rec, _, _ = curate._select_records(rows, curate.CATEGORIES[category], emb, id2row,
                                       homes=homes, support={}, category=category)
    return {d["path"] for d in rec}


def _pool(n_vendors=5, per=4, dim=16):
    rows, i = [], 0
    for v in range(n_vendors):
        for k in range(per):
            rows.append(_row(i, f"V{v}", f"Texture {v}-{k} Unique{i}.wav"))
            i += 1
    return rows


def _emb(n, dim=16, seed=0):
    e = np.random.default_rng(seed).standard_normal((n, dim)).astype("float32")
    return e / np.linalg.norm(e, axis=1, keepdims=True)


# Guards against: an explicit Keep is dropped because a multisample SIBLING was rated Misfiled
def test_sibling_misfile_does_not_drop_explicit_keep(monkeypatch):
    rows = _pool()
    a = _row(100, "V0", "Warm Pad C3.wav")      # rated Keep in PADS
    b = _row(101, "V0", "Warm Pad D3.wav")      # sibling rated Misfiled in PADS
    rows += [a, b]
    homes = {r.id: "PADS" for r in rows}
    got = _run_select(monkeypatch, rows, "PADS", homes,
                      keeps={a.path: "PADS"}, misfiled={b.path: "PADS"},
                      emb=_emb(len(rows)))
    assert b.path not in got          # the misfile itself is honored
    assert a.path in got              # ...but the human Keep must survive (it is lost now)


# Guards against: _MisfiledIndex keeps one category per file; a set flagged Misfiled in two
# categories is only excluded from one of them
def test_set_misfiled_in_two_categories_excluded_from_both(monkeypatch):
    rows = _pool()
    a = _row(100, "V0", "Bright Chord C3.wav")   # rated Misfiled in FX, re-homed to PADS
    b = _row(101, "V0", "Bright Chord D3.wav")   # sibling rated Misfiled in PADS
    rows += [a]
    homes = {r.id: "PADS" for r in rows}
    got = _run_select(monkeypatch, rows, "PADS", homes, keeps={},
                      misfiled={a.path: "FX", b.path: "PADS"}, emb=_emb(len(rows)))
    # the set was ruled not-a-pad (via b); a is its sibling -> must not land in PADS
    assert a.path not in got


# ---------------------------------------------------------------------------
# Guards against: export_one's collision handling only goes one deep ("_2"); a third file
# with the same basename in a family silently overwrites the second
# ---------------------------------------------------------------------------
def test_three_same_named_files_in_one_family_all_exported(monkeypatch, tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("sklearn")
    dim, n = 16, 10
    rng = np.random.default_rng(1)
    base = rng.standard_normal(dim)
    emb = rng.standard_normal((n, dim)).astype("float32")
    emb[:3] = base          # the three same-named kicks are acoustically identical
    ids = np.arange(n)

    src_root = tmp_path / "SampleLibrary"
    rec = []
    for i in range(n):
        vendor = f"V{i}"
        name = "Kick 01.wav" if i < 3 else f"Kick {i:02d} x.wav"
        p = src_root / vendor / "Pack" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.full(64, 0.01 * (i + 1)), 44100, subtype="PCM_16")
        rec.append(dict(id=i, row=i, path=str(p), pack=vendor, clip=0, br=0.5, atk=10.0,
                        dec=300.0, sub=0.1, noi=0.3, har=0.7, cr=4.0, tune=None,
                        bpm=None, bpm_src=None, bpm_fold=None, sup=1, qual=1.0, fav=0,
                        ab=[], vendor=vendor))

    monkeypatch.setattr(curate, "index_stale", lambda s: (False, n, n))
    monkeypatch.setattr(curate, "load_index", lambda: (ids, emb))
    monkeypatch.setattr(curate, "_fetch_rows", lambda *a, **k: [])
    monkeypatch.setattr(curate, "_select_records",
                        lambda *a, **k: (list(rec), set(), dict(twins=0)))
    monkeypatch.setattr(curate, "embed_text",
                        lambda t: np.random.default_rng(abs(hash(t)) % 2**32)
                        .standard_normal(dim).astype("float32"))

    out = tmp_path / "master"
    summary, entries = curate.build_taxonomy(
        None, "KICKS", str(out), homes={}, support={}, votes={}, describe=False,
        loudness=False, return_entries=True, log=lambda m: None)

    kick01 = [e for e in entries if e["src"].endswith("Kick 01.wav")]
    assert len(kick01) == 3                               # all three were selected
    outs = [e["out"] for e in entries]
    assert len(set(outs)) == len(outs), "two manifest entries point at one file"
    on_disk = [p for p in (out / "KICKS").rglob("*.wav")]
    assert len(on_disk) == summary["files"]


@pytest.mark.parametrize("name,cat", [("SoftKalimba.wav", "ACOUSTIC"), ("kalimba_C#.wav", "ACOUSTIC"),
                                      ("Clavinette Soft.wav", "PIANO"),
                                      ("The Clav 01.wav", "PIANO")])
def test_override_keeps_compound_name_hits(name, cat):
    """The clave veto must not cost real hits: word-boundary matching would lose compound
    names, so matching stays substring-based."""
    assert _name_override(name) == cat


# "one-shots" that are velocity/note ladders (many snare hits in one file).
def test_chain_guard_drops_sample_chains_but_not_keeps(monkeypatch):
    rows = _pool()
    ladder = _row(100, "V0", "Snare-Ladder-Hits.wav")
    ladder.n_events, ladder.event_regularity, ladder.event_echo = 48, 0.1, 0
    kept = _row(101, "V1", "Tom-Stick-Hits.wav")
    kept.n_events, kept.event_regularity, kept.event_echo = 32, 0.01, 0
    echo = _row(102, "V2", "Snare Echo 2.wav")
    echo.n_events, echo.event_regularity, echo.event_echo = 4, 0.0, 1
    rows += [ladder, kept, echo]
    homes = {r.id: "SNARES" for r in rows}
    got = _run_select(monkeypatch, rows, "SNARES", homes, keeps={kept.path: "SNARES"},
                      misfiled={}, emb=_emb(len(rows)))
    assert ladder.path not in got
    assert kept.path in got          # a human Keep wins over the detector
    assert echo.path in got          # a decaying echo is one sound


# Keys routing: chords routed into STABS from other buckets are capped at
# ROUTED_MAX_SHARE of the pool (so one large chord pack can't take over STABS).
def test_routed_rows_are_capped(monkeypatch):
    import fourier.packs.ratings as ratings
    monkeypatch.setattr(ratings, "keep_pins", lambda *a, **k: {})
    monkeypatch.setattr(ratings, "misfiled_map", lambda *a, **k: {})
    monkeypatch.setattr(curate, "_misfiled_index", lambda m, e, i, pid=None: _MisfiledIndex(m))
    rows = _pool(n_vendors=5, per=4)                              # 20 native
    routed = [_row(200 + i, f"K{i % 5}", f"Piano Chord {i}.wav") for i in range(30)]
    rows += routed
    homes = {r.id: "STABS" for r in rows}
    id2row = {r.id: j for j, r in enumerate(rows)}
    # (the routed-share mechanism; STABS' own named-stabs gate is tested in test_taxonomy)
    cfg = dict(curate.CATEGORIES["STABS"], stab_names=False)
    rec, _, stats = curate._select_records(rows, cfg, _emb(len(rows)),
                                           id2row, homes=homes, support={}, category="STABS",
                                           routed_ids={r.id for r in routed})
    got = {d["id"] for d in rec}
    n_routed = len(got & {r.id for r in routed})
    pool = len(rec) + stats["routed_capped"]
    assert n_routed == int(np.ceil(curate.ROUTED_MAX_SHARE * pool))
    assert n_routed < 30 and len(got - {r.id for r in routed}) == 20


# Waves live only in WAVES; length caps; bpm-named phrases; a Keep wins.
def test_waves_caps_and_bpm_guards(monkeypatch):
    rows = _pool()
    wave = _row(100, "Vendor B", "Table_31.wav"); wave.duration_s = 0.046
    wave.rel_path = "Vendor B/Wavetable Pack 1/General/Table_31.wav"
    long_ = _row(101, "V1", "Bass 7 C.wav"); long_.duration_s = 26.0
    phrase = _row(102, "V2", "Bass Riff A 95bpm.wav")
    kept = _row(103, "V3", "Low Sub Drone.wav"); kept.duration_s = 30.0
    rows += [wave, long_, phrase, kept]
    homes = {r.id: "SUB" for r in rows}
    got = _run_select(monkeypatch, rows, "SUB", homes, keeps={kept.path: "SUB"}, misfiled={},
                      emb=_emb(len(rows)))
    assert wave.path not in got and long_.path not in got and phrase.path not in got
    assert kept.path in got


# ---------------------------------------------------------------------------
# Guards against: a name override skipping the category's length cap. It holds,
# judged on the exported length (a cowbell's quiet tail trims away)
# ---------------------------------------------------------------------------
def test_a_name_override_holds_the_length_cap_on_the_exported_length(monkeypatch, tmp_path):
    import soundfile as sf
    sr = 44100
    rng = np.random.default_rng(0)
    long_take = 0.3 * rng.standard_normal(int(15.1 * sr))             # 15 s of cowbells ringing
    t = np.arange(int(6.0 * sr)) / sr
    short_hit = np.sin(2 * np.pi * 800 * t) * np.exp(-t / 0.05)        # a hit, then 5.5 s of nothing
    rows = _pool()
    for i, (name, y) in enumerate((("Cowbells Ringing.wav", long_take), ("Cowbell Hit 1.wav", short_hit))):
        p = tmp_path / name
        sf.write(str(p), y, sr)
        r = _row(100 + i, "V9", name)
        r.path, r.duration_s = str(p), len(y) / sr
        rows.append(r)
    assert curate._row_override(rows[-1])[1] == "PERC" and curate._row_override(rows[-2])[1] == "PERC"
    got = _run_select(monkeypatch, rows, "PERC", {r.id: "PERC" for r in rows}, keeps={},
                      misfiled={}, emb=_emb(len(rows)))
    assert rows[-2].path not in got                    # 15.1 s exported: over PERC's 4 s cap
    assert rows[-1].path in got                        # 6 s source, well under 4 s exported
