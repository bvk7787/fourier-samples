"""Every rule that leaves a file out writes its reason to the why log, the changelog lists
every file that left or joined with its reason, and verify warns on a leaver with none.
Invented names only."""
import csv
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from fourier.packs import builddiff as B
from fourier.packs import curate as C
from fourier.packs import why_log
from fourier.packs.curate_config import CATEGORIES


def _row(i, name, folder="Acme/Kicks", dur=0.4, **kw):
    base = dict(id=i, rel_path=f"{folder}/{name}", path=f"/l/{folder}/{name}", filename=name,
                file_hash=f"h{i}", file_format="wav", duration_s=dur, brightness=0.5,
                noisiness=0.3, harmonicity=0.7, crest_factor=4.3, base_note=None,
                base_note_confidence=None, attack_time_ms=20.0, decay_time_ms=350.0,
                sub_weight=0.0, is_clipped=0, tempo_bpm=None, bpm_reliable=0,
                onset_rate_hz=None, chroma_concentration=None, spectral_flatness_mean=0.5,
                rms_mean=0.2, dc_offset_ratio=0.0, ableton_tags=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _emb(n):
    E = np.zeros((n, 8), dtype="float32")
    E[:, 0] = 1.0
    for i in range(n):
        E[i, 1 + i % 7] = 0.5 + 0.05 * i                     # far apart: no near-duplicates
    return E / np.linalg.norm(E, axis=1, keepdims=True)


@pytest.fixture
def ratings(monkeypatch):
    """Drops and Misfiled ratings as the test sets them (vectors ignored: siblings count)."""
    got = dict(drops=set(), misfiled={})
    from fourier.packs import ratings as R
    monkeypatch.setattr(R, "drop_set", lambda *a, **k: got["drops"])
    monkeypatch.setattr(R, "drop_categories", lambda *a, **k: {p: None for p in got["drops"]})
    monkeypatch.setattr(R, "rating_scopes", lambda *a, **k: {})
    monkeypatch.setattr(R, "misfiled_map", lambda *a, **k: got["misfiled"])
    C._REACH.clear()
    monkeypatch.setattr(C, "_misfiled_index", lambda m, *a, **k: C._MisfiledIndex(m))
    return got


# --- the selection's rules ------------------------------------------------------------------

def test_ratings_wavetables_and_routing_each_write_their_reason(ratings):
    rows = [_row(1, "Kick 01.wav"),
            _row(2, "Kick 02.wav"),                                       # rated Drop
            _row(3, "Kick 03.wav"),                                       # rated Misfiled here
            _row(4, "Qxvwz Kick C2.wav"),                                 # its set's C1 rated Misfiled
            _row(5, "Wt 01.wav", folder="Acme/Wavetables", dur=2.0),      # a long wavetable
            _row(6, "Kick 06.wav"),                                       # routed to SNARES (previous pick)
            _row(7, "Kick 07.wav")]                                       # routed to SNARES (new)
    ratings["drops"] = {rows[1].path}
    ratings["misfiled"] = {rows[2].path: "KICKS", "/l/Acme/Kicks/Qxvwz Kick C1.wav": "KICKS"}
    homes = {r.id: "KICKS" for r in rows} | {6: "SNARES", 7: "SNARES"}
    why = {}
    rec, _fm, _st = C._select_records(rows, CATEGORIES["KICKS"], _emb(7), {i + 1: i for i in range(7)},
                                      homes=homes, category="KICKS", why=why,
                                      prev={rows[5].path, rows[0].path})
    assert [r["id"] for r in rec] == [1]
    assert why["rated_drop"] == {2: None}
    assert why["rated_misfiled"] == {3: None, 4: "Qxvwz Kick C1.wav"}
    assert why["long_wavetable"] == {5: [2.0, C.WAVE_MAX_DUR]}
    # routing is recorded for the previous build's files (what a changelog explains) only
    assert why["homed_elsewhere"] == {6: "SNARES"}
    doc = {"category": "KICKS"}
    assert why_log.describe("rated_drop", None, doc) == "you rated it Drop: out of every category"
    assert "Qxvwz Kick C1.wav, of the same multisample set, Misfiled in KICKS" in \
        why_log.describe("rated_misfiled", "Qxvwz Kick C1.wav", doc)
    assert "2 s, over WAVE_MAX_DUR 1 s" in why_log.describe("long_wavetable", [2.0, 1.0], doc)
    assert why_log.describe("homed_elsewhere", "SNARES", doc).startswith("routed to SNARES")


def test_an_fx_folders_sound_says_it_never_goes_to_pads(ratings):
    rows = [_row(1, "Qzlaser 01.wav", folder="Acme/FX/Lasers", canonical=["class.oneshot", "fx.sci_fi"]),
            _row(2, "Warm Pad.wav", folder="Acme/Pads", canonical=["class.oneshot", "synth.pad"])]
    why = {}
    C._select_records(rows, CATEGORIES["PADS"], _emb(2), {1: 0, 2: 1}, homes={1: "FX", 2: "SYNTH"},
                      category="PADS", why=why)
    assert why == {"fx_folder": {1: "FX"}}                  # the pad just went elsewhere: not recorded
    assert "never PADS (FX_FOLDER_NOT); it went to FX" in why_log.describe("fx_folder", "FX", {"category": "PADS"})


def test_a_content_twin_is_recorded():
    rows = [_row(1, "BD 808.wav"), _row(2, "BD 808.aif")]
    why = {}
    C._select_records(rows, CATEGORIES["KICKS"], _emb(2), {1: 0, 2: 1}, category="KICKS", why=why)
    assert why == {"twins": {2: None}}


# --- the set rules after clustering ---------------------------------------------------------

def test_the_set_cap_names_the_notes_it_kept():
    rec = [dict(id=10 + k, path=f"/l/Acme/P/Qxvwz Pad {n}.wav", tune=t, pin=False)
           for k, (n, t) in enumerate((("C2", 36), ("C3", 48), ("C4", 60), ("C5", 72), ("C6", 84)))]
    rec.append(dict(id=99, path="/l/Acme/P/Kick.wav", tune=None, pin=False))
    why = {}
    assert C._cap_siblings(list(range(6)), rec, cap=3, why=why) == [0, 2, 4, 5]
    kept = ["Qxvwz Pad C2.wav", "Qxvwz Pad C4.wav", "Qxvwz Pad C6.wav"]
    assert why == {11: [3, kept], 13: [3, kept]}
    line = why_log.describe("sibling_capped", why[11], {"category": "PADS"})
    assert line.startswith("over its multisample set's cap: at most 3 notes of one set in a PADS folder")
    assert "the folder keeps Qxvwz Pad C2.wav" in line
    assert C._cap_siblings(list(range(6)), rec, cap=3) == [0, 2, 4, 5]      # without why: as before


def test_one_note_per_c_names_the_note_it_keeps_and_the_c():
    names = ["Qxvwz Chord A#2.wav", "Qxvwz Chord C3.wav", "Qxvwz Chord D3.wav"]
    rec = []
    for k, n in enumerate(names):
        rt, src = C._retune_shift("STABS", n, None)
        rec.append(dict(id=k + 1, path=f"/x/Acme/Keys/{n}", tune=None, retune=rt, root_src=src,
                        name_pc=C._name_root_pc(n)))
    why = {}
    groups, n = C._one_per_retune_target([[0, 1, 2]], rec, why=why)
    assert n == 2 and groups == [[1]]
    assert set(why) == {1, 3} and why[1][0] == "Qxvwz Chord C3.wav" and why[1][1].startswith("C")
    line = why_log.describe("one_note_per_c", why[1], {"category": "STABS"})
    assert "already plays in STABS as Qxvwz Chord C3.wav" in line


def _pads(monkeypatch, tmp_path, n=8):
    """One PADS build over n notes of one multisample set (the selection stubbed)."""
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("sklearn")
    dim = 16
    emb = np.random.default_rng(3).standard_normal((n, dim)).astype("float32")
    rec = []
    for i in range(n):
        p = tmp_path / "SampleLibrary" / "Acme" / "Pack" / f"Qxvwz Pad C{i + 1}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.full(64, 0.01 * (i + 1)), 44100, subtype="PCM_16")
        rec.append(dict(id=i, row=i, path=str(p), pack="Acme", clip=0, br=0.5, atk=10.0,
                        dec=300.0, sub=0.1, noi=0.3, har=0.7, cr=4.0, tune=None, bpm=None,
                        bpm_src=None, bpm_fold=None, sup=1, qual=1.0, fav=0, ab=[], vendor="Acme"))
    monkeypatch.setattr(C, "index_stale", lambda s: (False, n, n))
    monkeypatch.setattr(C, "load_index", lambda: (np.arange(n), emb))
    monkeypatch.setattr(C, "_fetch_rows", lambda *a, **k: [])
    monkeypatch.setattr(C, "_select_records", lambda *a, **k: (list(rec), set(), dict(twins=0)))
    monkeypatch.setattr(C, "embed_text", lambda t: np.random.default_rng(abs(hash(t)) % 2**32)
                        .standard_normal(dim).astype("float32"))
    from .test_cloud_only import _cloud
    _cloud(monkeypatch, [])
    return rec


def test_a_build_logs_and_counts_the_set_cap_apart_from_the_budget(monkeypatch, tmp_path):
    _pads(monkeypatch, tmp_path)
    msgs, out = [], tmp_path / "master"
    C.build_taxonomy(None, "PADS", str(out), homes={}, support={}, votes={}, describe=False,
                     loudness=False, return_entries=True, budget_override=50, log=msgs.append)
    doc = why_log.read(out, "PADS")
    capped = doc["out"].get("sibling_capped") or {}
    assert len(capped) == 8 - doc["files"] > 0                  # the cap, not the budget (50)
    assert "over_budget" not in doc["out"] and doc["counts"]["sibling_capped"] == len(capped)
    assert f"PADS: {len(capped)} extra notes of multisample sets left out (at most 3 a set a folder)" in msgs


# --- the previous build's files -------------------------------------------------------------

def test_previous_files_the_category_never_saw_get_a_reason(monkeypatch):
    monkeypatch.setattr(C, "_sample_ids", lambda srcs: {"/l/a.wav": 1, "/l/b.wav": 2, "/l/c.wav": 3})
    monkeypatch.setattr(C, "_missing_ids", lambda s, ids: {3})
    monkeypatch.setattr("fourier.packs.ratings.keep_pins", lambda *a, **k: {})
    prev = {"/l/old/a.wav": "/l/a.wav", "/l/b.wav": "/l/b.wav", "/l/c.wav": "/l/c.wav",
            "/l/d.wav": "/l/d.wav", "/l/e.wav": "/l/e.wav", "/l/f.wav": "/l/f.wav"}
    rows = [SimpleNamespace(id=5, path="/l/e.wav"), SimpleNamespace(id=6, path="/l/f.wav")]
    why = {"over_budget": {5: None}}
    got = C._why_prev(None, "KICKS", prev, rows, why, homes={1: "SNARES", 2: "KICKS"},
                      picked={"/l/f.wav"})
    assert got == {"/l/old/a.wav": 1, "/l/b.wav": 2, "/l/c.wav": 3, "/l/d.wav": None,
                   "/l/e.wav": 5, "/l/f.wav": 6}
    assert why["homed_elsewhere"] == {1: "SNARES"} and why["not_candidate"] == {2: None}
    assert why["missing"] == {3: None} and why["over_budget"] == {5: None}
    doc = {"category": "KICKS", "prev": {k: v for k, v in got.items()},
           "out": {k: {str(i): d for i, d in v.items()} for k, v in why.items()}}
    assert why_log.prev_reason(doc, "/l/d.wav") == ("missing", None)       # unknown to the database
    assert why_log.prev_reason(doc, "/l/old/a.wav") == ("homed_elsewhere", "SNARES")
    assert why_log.prev_reason(doc, "/l/f.wav") is None                    # still picked
    assert why_log.prev_reason(dict(doc, status="too_few", need=6), "/l/f.wav") == ("category_empty", 6)


def test_every_new_reason_reads_as_words():
    doc = {"category": "KICKS"}
    for key in ("sibling_capped", "one_note_per_c", "long_wavetable", "fx_folder", "homed_elsewhere",
                "rated_drop", "rated_misfiled", "not_instrument", "not_phrase", "not_wave", "not_scratch_named",
                "noise_named", "not_analyzed", "ir_preview", "hygiene", "missing", "not_candidate"):
        assert key in why_log.FILTER_WORDS
        assert not why_log.describe(key, None, doc).startswith("left out by ")


# --- the changelog and verify ---------------------------------------------------------------

def _e(fam, name, cat_dir="V"):
    return {"family": fam, "out": f"{fam}/{name}", "src": f"/l/SampleLibrary/{cat_dir}/{name}"}


OLD = {"generated": "2000-01-25T10:00:00", "categories": {
    "PADS": {"entries": [_e("warm", "Qxvwz Pad C1.wav"), _e("warm", "Qxvwz Pad C2.wav"),
                         _e("warm", "Pad 3.wav"), _e("warm", "Qpad 4.wav")]},
    "FX": {"entries": [_e("hit", "Hit 1.wav"), _e("hit", "Qhit 2.wav")]}}}
NEW = {"generated": "2000-01-26T10:00:00", "categories": {
    "PADS": {"entries": [_e("warm", "Qxvwz Pad C1.wav"), _e("warm", "Pad 9.wav"), _e("warm", "Qhit 2.wav"),
                         _e("warm", "Pad 8.wav")]},
    "FX": {"entries": [_e("hit", "Hit 1.wav"), _e("hit", "Qpad 4.wav")]}}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_RATINGS", str(tmp_path / "home" / "ratings.json"))
    (tmp_path / "home").mkdir()
    pin = "/l/SampleLibrary/V/Pad 8.wav"
    (tmp_path / "home" / "ratings.json").write_text(json.dumps(
        {"ratings": {pin: {"verdict": "keep", "category": "PADS"}}}))
    return tmp_path


def _why_logs(master, bare=False):
    src = "/l/SampleLibrary/V/"
    pads = {"category": "PADS", "status": "built",
            "prev": {src + "Qxvwz Pad C1.wav": 1, src + "Qxvwz Pad C2.wav": 2, src + "Pad 3.wav": 3,
                     src + "Qpad 4.wav": 4},
            "out": {"sibling_capped": {"2": [3, ["Qxvwz Pad C1.wav"]]}, "homed_elsewhere": {"4": "FX"}}}
    if not bare:
        pads["out"]["rated_drop"] = {"3": None}
    fx = {"category": "FX", "status": "built", "prev": {src + "Hit 1.wav": 5, src + "Qhit 2.wav": 6},
          "out": {"homed_elsewhere": {"6": "PADS"}}}
    why_log.write(master, "PADS", pads)
    why_log.write(master, "FX", fx)


def _master(root, doc):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(doc))
    return root


def test_the_changelog_lists_every_file_that_left_or_joined_with_its_reason(home):
    B.archive_build(_master(home / "m1", OLD))
    new = _master(home / "m2", NEW)
    _why_logs(new)
    ch = B.write_changelog(new, log=lambda *a: None)
    rows = list(csv.DictReader(open(B.changes_csv(NEW))))
    assert B.changes_csv(NEW).parent == B.builds_dir() and not (new / "changes.csv").exists()
    got = {(r["change"], r["category"], os.path.basename(r["src"])): r["reason"] for r in rows}
    assert got == {("left", "PADS", "Qxvwz Pad C2.wav"): "sibling_capped",
                   ("left", "PADS", "Pad 3.wav"): "rated_drop",
                   ("left", "PADS", "Qpad 4.wav"): "homed_elsewhere",      # moved to FX
                   ("left", "FX", "Qhit 2.wav"): "homed_elsewhere",        # moved to PADS
                   ("joined", "PADS", "Pad 9.wav"): "refill",
                   ("joined", "PADS", "Pad 8.wav"): "rated_in",
                   ("joined", "PADS", "Qhit 2.wav"): "moved_in",
                   ("joined", "FX", "Qpad 4.wav"): "moved_in"}
    capped = next(r for r in rows if r["reason"] == "sibling_capped")
    assert "the folder keeps Qxvwz Pad C1.wav" in capped["detail"] and capped["file"] == "warm/Qxvwz Pad C2.wav"
    assert next(r for r in rows if r["src"].endswith("Qpad 4.wav") and r["change"] == "left")["detail"] \
        .startswith("now in FX")
    text = ch.read_text()
    assert "## Why files left or joined" in text and str(B.changes_csv(NEW)) in text
    assert "2 left the master, 4 joined (2 of them moved between categories)." in text
    assert "- 1 sibling_capped: over a multisample set's cap in a folder" in text
    assert "- 1 refill" in text and "- 2 moved_in" in text


def test_routing_to_a_category_that_left_it_out_says_both(monkeypatch):
    old = {"categories": {"PADS": {"entries": [_e("warm", "Qpad 4.wav")]}}}
    new = {"categories": {"PADS": {"entries": []}, "FX": {"entries": []}}}
    docs = {"PADS": {"category": "PADS", "prev": {"/l/SampleLibrary/V/Qpad 4.wav": 4},
                     "out": {"homed_elsewhere": {"4": "FX"}}},
            "FX": {"category": "FX", "out": {"vendor_capped": {"4": "Acme"}}}}
    monkeypatch.setattr(why_log, "read", lambda m, c: docs.get(c))
    rows = B.change_rows(old, new, "/nowhere", pins={})
    assert rows[0]["reason"] == "homed_elsewhere"
    assert "routed to FX" in rows[0]["detail"] and "FX: over the per-vendor share: Acme" in rows[0]["detail"]


def test_verify_warns_on_a_leaver_without_a_reason(home):
    from fourier.packs.verify import WARN, PASS, _Ctx, check_leavers
    m2 = _master(home / "m2", NEW)
    assert list(check_leavers(_Ctx(m2))) == []                 # no previous build: nothing said
    B.archive_build(_master(home / "m1", OLD))
    B.archive_build(m2)                                        # the master's own archive isn't "previous"
    assert list(check_leavers(_Ctx(m2))) == []                 # no why logs for this master: quiet
    _why_logs(m2, bare=True)
    (res,) = list(check_leavers(_Ctx(m2)))
    assert res.level == WARN and "Pad 3.wav" in res.detail and "(2 left)" in res.check
    _why_logs(m2)
    (res,) = list(check_leavers(_Ctx(m2)))
    assert res.level == PASS
