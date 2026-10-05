"""Regressions: ratings harvest / write-back / scorecard / pin + misfile consumption.

Each test reproduces a defect the code must not bring back."""
import json
import os

import numpy as np

from fourier.packs.ratings import (
    LIVE_USER_XMP, _render_xmp, apply_tags, harvest, load_store, save_store, scorecard,
)
from fourier.packs.review import REVIEW_DIR, REVIEW_INDEX

QUIET = lambda *_: None  # noqa: E731

SD_SRC = "/lib/SampleLibrary/V/P/SD 1.wav"
SD_OUT = "crack/SD 1.wav"
QNAME = "SNARES - crack - SD 1.wav"


def _write_manifest(root, stamp, entries):
    """entries: {cat: [(family, filename, src)]}"""
    os.makedirs(root, exist_ok=True)
    cats = {c: {"entries": [{"family": f, "out": f"{f}/{n}", "src": s} for f, n, s in es]}
            for c, es in entries.items()}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump({"fourier_manifest": 2, "generated": stamp, "categories": cats}, fh)
    return str(root)


def _sidecar(folder, items):
    info = os.path.join(folder, "Ableton Folder Info")
    os.makedirs(info, exist_ok=True)
    p = os.path.join(info, LIVE_USER_XMP)
    with open(p, "w") as fh:
        fh.write(_render_xmp(items, "Live"))
    return p


def _queue(root, qid, items):
    q = os.path.join(root, REVIEW_DIR)
    os.makedirs(q, exist_ok=True)
    with open(os.path.join(q, REVIEW_INDEX), "w") as fh:
        json.dump({"id": qid, "items": items}, fh)
    return q


SD_QITEM = {QNAME: {"src": SD_SRC, "category": "SNARES", "family": "crack",
                    "out": SD_OUT, "bucket": "random"}}


# ---------------------------------------------------------------------------
# Guards against: an unreadable sidecar is treated as "everything untagged" -> ratings cleared
# ---------------------------------------------------------------------------
def test_unreadable_sidecar_does_not_clear_ratings(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _write_manifest(tmp_path / "m", "t1", {"SNARES": [("crack", "SD 1.wav", SD_SRC)]})
    p = _sidecar(os.path.join(m, "SNARES", "crack"), {"SD 1.wav": ["Fourier|Keep"]})
    harvest(m, store, log=QUIET)
    assert load_store(store)["ratings"][SD_SRC]["verdict"] == "keep"
    # Live (or a sync client) leaves the sidecar truncated / mid-write
    with open(p) as fh:
        txt = fh.read()
    with open(p, "w") as fh:
        fh.write(txt[: len(txt) // 2])
    s = harvest(m, store, log=QUIET)
    assert s["cleared"] == 0
    assert load_store(store)["ratings"].get(SD_SRC, {}).get("verdict") == "keep"


# ---------------------------------------------------------------------------
# Guards against: a stale review-queue tag overrides a newer, conflicting master tag
# ---------------------------------------------------------------------------
def test_new_master_drop_not_overridden_by_stale_queue_keep(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _write_manifest(tmp_path / "m", "t1", {"SNARES": [("crack", "SD 1.wav", SD_SRC)]})
    q = _queue(m, "review-1", SD_QITEM)
    _sidecar(q, {QNAME: ["Fourier|Keep"]})          # rated Keep in the queue
    harvest(m, store, log=QUIET)
    apply_tags(m, store, log=QUIET)                  # write-back puts Keep on the master copy
    assert load_store(store)["ratings"][SD_SRC]["verdict"] == "keep"
    # later, browsing the master, the owner changes their mind: Drop (queue copy untouched)
    _sidecar(os.path.join(m, "SNARES", "crack"), {"SD 1.wav": ["Fourier|Drop"]})
    harvest(m, store, log=QUIET)
    # the new Drop must not be silently replaced by the old queue Keep (and pinned)
    assert load_store(store)["ratings"][SD_SRC]["verdict"] == "drop"


# ---------------------------------------------------------------------------
# Guards against: scorecard counts a Drop rated in the current master's review queue as "back"
# ---------------------------------------------------------------------------
def test_scorecard_review_drop_on_current_master_is_not_back(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _write_manifest(tmp_path / "m", "t1", {"SNARES": [("crack", "SD 1.wav", SD_SRC)]})
    q = _queue(m, "review-1", SD_QITEM)
    _sidecar(q, {QNAME: ["Fourier|Drop"]})
    harvest(m, store, log=QUIET)
    sc = scorecard(m, store)                 # no rebuild happened
    assert sc["categories"]["SNARES"]["drop"] == 1
    assert sc["categories"]["SNARES"]["drops_back"] == 0


# ---------------------------------------------------------------------------
# Guards against: harvesting a rebuilt master re-stamps written-back Drops -> "back" vanishes
# ---------------------------------------------------------------------------
def test_scorecard_drop_brought_back_stays_back_after_harvest(tmp_path):
    store = str(tmp_path / "ratings.json")
    ent = {"SNARES": [("crack", "SD 1.wav", SD_SRC)]}
    m1 = _write_manifest(tmp_path / "m1", "t1", ent)
    _sidecar(os.path.join(m1, "SNARES", "crack"), {"SD 1.wav": ["Fourier|Drop"]})
    harvest(m1, store, log=QUIET)
    # rebuild keeps the dropped snare; post-build write-back re-tags it Drop
    m2 = _write_manifest(tmp_path / "m2", "t2", ent)
    apply_tags(m2, store, log=QUIET)
    assert scorecard(m2, store)["categories"]["SNARES"]["drops_back"] == 1
    # `fourier review score` / the next build harvests the live master (m2) first
    harvest(m2, store, log=QUIET)
    assert scorecard(m2, store)["categories"]["SNARES"]["drops_back"] == 1


# ---------------------------------------------------------------------------
# _select_records harness (synthetic rows, no DB)
# ---------------------------------------------------------------------------
class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):          # every other column: NULL
        return None


def _run_select(tmp_path, monkeypatch, rows, store_data, category="KICKS"):
    import fourier.db.session as dbs
    from fourier.packs import curate

    store = str(tmp_path / "ratings.json")
    s = load_store(store)
    for k, v in store_data.items():
        s[k].update(v)
    save_store(s, store)
    monkeypatch.setenv("FOURIER_RATINGS", store)

    class _S:
        def execute(self, *_a, **_k):
            return []

    class _CM:
        def __enter__(self):
            return _S()

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(dbs, "session_scope", lambda: _CM())
    rng = np.random.default_rng(0)
    E = rng.normal(size=(len(rows), 16)).astype("float32")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    id2row = {r.id: i for i, r in enumerate(rows)}
    cfg = {"noise": r"^\b$", "kind": "oneshot", "loop_guard": False, "no_vendor_cap": True,
           "near_dup_cos": 1.0}
    rec, _fm, _st = curate._select_records(rows, cfg, E, id2row, homes=None, support=None,
                                           category=category)
    return [d["path"] for d in rec]


# ---------------------------------------------------------------------------
# Guards against: an auto-misfile call on an UNRATED sibling evicts a human Keep
# ---------------------------------------------------------------------------
def test_auto_misfile_on_sibling_does_not_evict_human_keep(tmp_path, monkeypatch):
    base = "/lib/SampleLibrary/V/Toy Piano/"
    a = base + "Toy Piano A#3 v1.wav"           # rated Keep in KICKS by the human
    b = base + "Toy Piano D#4 v2.wav"           # its sibling, flagged by the detector only
    rows = [_Row(id=1, path=a, filename=os.path.basename(a), rel_path="V/Toy Piano/x"),
            _Row(id=2, path=b, filename=os.path.basename(b), rel_path="V/Toy Piano/y")]
    got = _run_select(tmp_path, monkeypatch, rows, {
        "ratings": {a: {"verdict": "keep", "category": "KICKS"}},
        "auto_misfiled": {b: {"category": "KICKS", "other": "PERC", "margin": 0.5}}})
    assert b not in got           # the detector call itself is honored
    assert a in got               # ...but a human Keep always wins over a detector call


# ---------------------------------------------------------------------------
# Guards against: a pinned Keep is lost when an earlier, unpinned byte-identical twin wins dedup
# ---------------------------------------------------------------------------
def test_pinned_keep_survives_hash_twin_dedup(tmp_path, monkeypatch):
    twin = "/lib/SampleLibrary/V/PackA/Kick Big.wav"
    kept = "/lib/SampleLibrary/W/PackB/Kick Big Copy.wav"   # the file rated Keep
    rows = [_Row(id=1, path=twin, filename="Kick Big.wav", rel_path="V/PackA/Kick Big.wav",
                 file_hash="h1", duration_s=0.5),
            _Row(id=2, path=kept, filename="Kick Big Copy.wav",
                 rel_path="W/PackB/Kick Big Copy.wav", file_hash="h1", duration_s=0.5)]
    got = _run_select(tmp_path, monkeypatch, rows,
                      {"ratings": {kept: {"verdict": "keep", "category": "KICKS"}}})
    assert kept in got


# ---------------------------------------------------------------------------
# Guards against: build --out harvests a DIFFERENT master than the one it replaces
# ---------------------------------------------------------------------------
def test_build_taxonomy_harvests_the_master_it_replaces(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import shutil
    from fourier import cli
    from fourier.packs import curate

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("FOURIER_CURATED_DIR", raising=False)
    store = str(tmp_path / "ratings.json")
    monkeypatch.setenv("FOURIER_RATINGS", store)
    out = _write_manifest(tmp_path / "master", "t1",
                          {"KICKS": [("808", "BD.wav", "/lib/SampleLibrary/V/P/BD.wav")]})
    _sidecar(os.path.join(out, "KICKS", "808"), {"BD.wav": ["Fourier|Keep"]})

    def fake_build(session, cat, out_dir, **kw):   # a rebuild replaces the category folder
        shutil.rmtree(os.path.join(out_dir, cat), ignore_errors=True)
        _write_manifest(out_dir, "t2", {"KICKS": [("808", "BD.wav", "/lib/SampleLibrary/V/P/BD.wav")]})
    monkeypatch.setattr(curate, "build_taxonomy", fake_build)
    r = CliRunner().invoke(cli.main, ["--db", str(tmp_path / "lib.db"), "build",
                                      "KICKS", "--no-scan", "--out", out])
    assert r.exit_code == 0, r.output
    assert load_store(store)["ratings"].get("/lib/SampleLibrary/V/P/BD.wav", {}).get("verdict") == "keep"


def test_drop_excludes_the_file_everywhere_and_only_that_file(tmp_path, monkeypatch):
    """A Drop keeps the file out of every category (not only out of the measurements);
    unrated files and Keeps are unaffected."""
    rows = [_Row(id=1, path="/l/P/Kick A.wav", filename="Kick A.wav", rel_path="P/Kick A.wav"),
            _Row(id=2, path="/l/P/Kick B.wav", filename="Kick B.wav", rel_path="P/Kick B.wav"),
            _Row(id=3, path="/l/P/Kick C.wav", filename="Kick C.wav", rel_path="P/Kick C.wav")]
    store = {"ratings": {
        "/l/P/Kick A.wav": dict(verdict="drop", category="SNARES"),   # dropped while in SNARES
        "/l/P/Kick C.wav": dict(verdict="keep", category="KICKS")}}
    got = _run_select(tmp_path, monkeypatch, rows, store, category="KICKS")
    assert "/l/P/Kick A.wav" not in got
    assert {"/l/P/Kick B.wav", "/l/P/Kick C.wav"} <= set(got)
