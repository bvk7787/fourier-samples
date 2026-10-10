"""`fourier tools audit` (packs/audit.py) and `fourier tools resolve`
(analysis/resolution.py), which no test reached, plus the edges of manifests.py and
vendors.py the other tests leave out. The CLAP index, the homes and the local LLM are
stand-ins."""
import json

import numpy as np
import pytest
from sqlalchemy import text

from fourier.db import session as S
from fourier.db.models import Sample, SononymMeta
from fourier.packs import curate as C
from fourier.packs import manifests
from fourier.packs import vendors as V


@pytest.fixture
def db(tmp_path):
    from fourier.metadata.rows import forget_current
    old = (S._engine, S._SessionLocal)
    S.init_db(tmp_path / "t.db")
    forget_current()
    yield
    forget_current()
    S._engine, S._SessionLocal = old


def _samples(n, cats=("Perc Kicks", "Tone Pads & Textures", "Perc Snares")):
    with S.session_scope() as s:
        for i in range(1, n + 1):
            pack = f"Vendor {i % 3}/Pack {i % 5}"
            s.add(Sample(id=i, path=f"/lib/SampleLibrary/{pack}/Qxvwz {i:03d}.wav",
                         rel_path=f"{pack}/Qxvwz {i:03d}.wav", filename=f"Qxvwz {i:03d}.wav",
                         duration_s=0.5))
            s.add(SononymMeta(sample_id=i, classes=["OneShot"], categories=[cats[i % len(cats)]]))


def test_resolve_writes_one_row_per_home_with_conflicts(db, monkeypatch):
    from fourier.analysis.resolution import persist_resolution
    _samples(3)
    monkeypatch.setattr(C, "load_index", lambda: (np.array([1, 2, 3]), np.eye(3, dtype="float32")))
    votes = {1: {"son_cats": ["KICKS"], "ab_cats": ["KICKS"], "son_labels": ["kick"]},
             2: {"son_cats": ["SNARES"], "ab_cats": ["CLAPS"]},
             3: {"son_cats": ["PADS"]}}
    monkeypatch.setattr(C, "compute_homes", lambda session, emb_n, id2row, log=print: (
        {1: "KICKS", 2: "SNARES", 3: "PADS"}, {(1, "KICKS"): 2}, votes))
    with S.session_scope() as s:
        assert persist_resolution(s, log=lambda m: None) == 3
        rows = {r[0]: r[1:] for r in s.execute(text(
            "SELECT sample_id, home, support, conflict, son_labels FROM sample_resolution"))}
    assert rows[1] == ("KICKS", 2, 0, json.dumps(["kick"]))
    assert rows[2][:3] == ("SNARES", 1, 1)               # the two classifiers disagree
    assert rows[3][:3] == ("PADS", 1, 0)                 # one classifier only: no conflict


def test_audit_reports_coverage_and_names_the_blindspots(db, monkeypatch, tmp_path):
    from fourier.packs import audit as A
    n = 300
    _samples(n)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, 16)).astype("float32")
    monkeypatch.setattr(A, "load_index", lambda: (np.arange(1, n + 1), emb))
    master = tmp_path / "master"
    for i in range(1, 21):                               # the master holds the first 20
        f = master / "KICKS" / "fam" / f"Qxvwz {i:03d}.wav"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    asked = []
    monkeypatch.setattr(A, "_llm_label", lambda profiles, model: asked.append(profiles) or {"0": "invented label"})
    out = tmp_path / "audit.md"
    with S.session_scope() as s:
        got = A.run_audit(s, master, out_path=out, log=lambda m: None)
    assert got["curated_mapped"] == 20 and got["poorly_covered"] > 50
    assert got["blindspot_clusters"] and asked and len(asked[0]) == len(got["blindspot_clusters"])
    report = out.read_text()
    assert "## 1. Taxonomy coverage" in report and "## 3. Provenance coverage" in report
    assert "**invented label**" in report and "cluster 1" in report      # unlabeled: by number


def test_audit_needs_a_master_it_can_map(db, monkeypatch, tmp_path):
    from fourier.packs import audit as A
    _samples(5)
    monkeypatch.setattr(A, "load_index", lambda: (np.arange(1, 6), np.eye(5, dtype="float32")))
    with S.session_scope() as s, pytest.raises(RuntimeError, match="no curated files"):
        A.run_audit(s, tmp_path / "empty", log=lambda m: None)


def test_manifests_pass_odd_documents_through():
    assert manifests.resolve(None) is None and manifests.resolve([1]) == [1]
    odd = {"categories": {"KICKS": "not a dict", "SNARES": {"entries": "nope"},
                          "HATS": {"entries": ["a string", {"out": "x.wav"}]}}}
    assert manifests.stored(odd) == odd                  # nothing to make relative, nothing lost


def test_vendor_layout_edges(monkeypatch):
    assert V.detect([]) == {"layout": V.PACKS, "umbrellas": [], "files": 0}
    assert V.describe({}) == V.describe({"_key": 1}) == "no samples scanned yet"

    class Broken:
        def get_bind(self):
            raise RuntimeError("no database")
    monkeypatch.setattr(V, "detect_folders", lambda roots: {"r": "from the files"})
    from fourier.packs import curate_config as cc
    monkeypatch.setattr(cc, "VENDORS", "auto")
    assert V.layouts(Broken(), ["r"]) == {"r": "from the files"}
