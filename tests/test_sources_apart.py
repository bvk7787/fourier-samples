"""No column holds values from two sources (schema 2, db/migrations.py): an older database
migrates without a value changing, the analysis steps write Sononym's readings to Sononym's
places and Fourier's own to its own, Fourier's own analysis runs with or without Sononym,
`fourier why` shows both readings where they differ, and a build on the synthetic library
picks the same before and after the migration and after Fourier's own analysis."""
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from sqlalchemy import create_engine, text

from fourier.db import migrations as M
from fourier.db import session as S

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "tests" / "data" / "schema_v1"
NEW = {c for c, _t in M.V2_COLUMNS}


def _v1_database(path):
    """A database as the code before schema 2 made it (its tables, at version 1), with rows
    as that code's analysis left them: bpm_corrected from Sononym's tempo or, without one,
    librosa's; a stand-in 0 for bpm_reliable without a Sononym row."""
    kind = "duckdb" if str(path).endswith(".duckdb") else "sqlite"
    eng = create_engine(f"{kind}:///{path}")
    with eng.begin() as c:
        for stmt in (x.strip() for x in (V1 / f"{kind}.sql").read_text().split(";\n")):
            if stmt:
                c.execute(text(stmt.rstrip(";")))
        c.execute(text("INSERT INTO schema_version (version, applied_at) VALUES (1, 'then')"))
        for i, (son_bpm, tempo, corrected, reliable) in enumerate(
                [(87.0, 174.0, 174.0, 1), (230.0, None, 115.0, 1), (None, 96.0, 96.0, 0),
                 (25.0, None, 100.0, 0), (None, None, None, 0)], 1):
            c.execute(text("INSERT INTO samples (id, path, rel_path, filename, duration_s, is_favorite, "
                           "is_hidden) VALUES (:i, :p, :r, :f, 2.0, false, false)"),
                      {"i": i, "p": f"/l/SampleLibrary/Acme/Loops/Loop {i}.wav",
                       "r": f"Acme/Loops/Loop {i}.wav", "f": f"Loop {i}.wav"})
            if i < 5:
                c.execute(text("INSERT INTO sononym_meta (id, sample_id, classes, bpm, bpm_confidence) "
                               "VALUES (:i, :i, '[\"Loop\"]', :b, 0.9)"), {"i": i, "b": son_bpm})
            c.execute(text("INSERT INTO sample_features (id, sample_id, tempo_bpm, bpm_corrected, "
                           "bpm_reliable, is_pitched) VALUES (:i, :i, :t, :c, :r, 0)"),
                      {"i": i, "t": tempo, "c": corrected, "r": reliable})
    eng.dispose()


def _dump(path, tables=("samples", "sononym_meta", "sample_features", "labels", "descriptors")):
    kind = "duckdb" if str(path).endswith(".duckdb") else "sqlite"
    eng = create_engine(f"{kind}:///{path}")
    out = {}
    with eng.connect() as c:
        for t in tables:
            res = c.execute(text(f"SELECT * FROM {t} ORDER BY 1, 2"))
            keys = list(res.keys())
            out[t] = [dict(zip(keys, r)) for r in res]
        version = c.execute(text("SELECT MAX(version) FROM schema_version")).scalar()
    eng.dispose()
    return out, version


@pytest.fixture
def fresh_engine():
    old = (S._engine, S._SessionLocal)
    S._engine = S._SessionLocal = None
    yield
    if S._engine is not None:
        S._engine.dispose()
    S._engine, S._SessionLocal = old


@pytest.mark.parametrize("name", ["lib.db", "lib.duckdb"])
def test_an_older_database_migrates_and_keeps_every_value(tmp_path, fresh_engine, name):
    db = tmp_path / name
    _v1_database(db)
    before, v = _dump(db)
    assert v == 1 and not NEW & set(before["sample_features"][0])
    S.init_db(db)
    S._engine.dispose()
    after, v = _dump(db)
    assert v == M.LATEST == 2
    for table, rows in before.items():                 # every value that was there, unchanged
        assert [{k: r[k] for k in old} for r, old in zip(after[table], rows)] == rows, table
    sf_after = after["sample_features"]
    assert NEW <= set(sf_after[0])
    # Sononym's tempo folded into 60-200 (none where Sononym has none); Fourier's own root
    # waits for `fourier tools analyze --only own`
    assert [r["sononym_bpm_folded"] for r in sf_after] == [87.0, 115.0, None, 100.0, None]
    assert all(r["own_root_midi"] is None and r["own_root_at"] is None for r in sf_after)
    assert (tmp_path / f"{name}.before-v2").exists() == name.endswith(".db")
    S.init_db(db)                                      # once only
    S._engine.dispose()
    assert _dump(db) == (after, 2)


# --- the analysis steps write each source's values to its own place ---

@pytest.fixture
def db(tmp_path, fresh_engine):
    from fourier.metadata.rows import forget_current
    path = tmp_path / "t.db"
    S.init_db(path)
    forget_current()
    yield path
    forget_current()


def _cli(db, *args):
    from fourier.cli import main
    r = CliRunner().invoke(main, ["--db", str(db), *args])
    assert r.exit_code == 0, r.output
    return r.output


def test_derived_writes_sononyms_values_only(db):
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    with S.session_scope() as s:
        for i in (1, 2):
            s.add(Sample(id=i, path=f"/l/SampleLibrary/V/P/Kick {i}.wav", rel_path=f"V/P/Kick {i}.wav",
                         filename=f"Kick {i}.wav", duration_s=0.4))
        s.add(SononymMeta(sample_id=1, classes=["OneShot"], categories=["Perc Kicks"], brightness=0.2,
                          harmonicity=0.8, noisiness=0.3, crest_factor=6.0, bpm=None, pitch_confidence=0.9))
    _cli(db, "tools", "analyze", "--only", "derived")
    with S.session_scope() as s:
        f = {x.sample_id: x for x in s.query(SampleFeatures)}
        assert f[1].is_pitched == 1 and f[1].bpm_reliable == 0 and f[1].sub_weight is not None
        assert f[1].computed_at is None            # librosa's stamp, and librosa hasn't run
    # without Sononym (no rows at all) a sample gets no stand-in for Sononym's reading
    with S.session_scope() as s:
        s.query(SampleFeatures).delete()
        s.query(SononymMeta).delete()
    from fourier.metadata.rows import forget_current
    forget_current()
    _cli(db, "tools", "analyze", "--only", "derived")
    with S.session_scope() as s:
        for x in s.query(SampleFeatures):
            assert x.derived_computed_at is not None
            assert x.bpm_reliable is None and x.is_pitched is None and x.loop_confidence is None


def test_bpm_fix_folds_sononyms_tempo_only(db):
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    with S.session_scope() as s:
        for i, (son, tempo) in enumerate([(230.0, 115.0), (None, 96.0)], 1):
            s.add(Sample(id=i, path=f"/l/SampleLibrary/V/Loops/L{i}.wav", rel_path=f"V/Loops/L{i}.wav",
                         filename=f"L{i}.wav", duration_s=4.0))
            s.add(SononymMeta(sample_id=i, classes=["Loop"], bpm=son))
            s.add(SampleFeatures(sample_id=i, tempo_bpm=tempo))
    _cli(db, "tools", "analyze", "--only", "bpm-fix")
    with S.session_scope() as s:
        f = {x.sample_id: x for x in s.query(SampleFeatures)}
        assert f[1].sononym_bpm_folded == 115.0 and f[2].sononym_bpm_folded is None
        assert f[1].bpm_corrected is None and f[2].bpm_corrected is None      # legacy: not written
    assert "Every Sononym tempo is folded already" in _cli(db, "tools", "analyze", "--only", "bpm-fix")
    with S.session_scope() as s:
        s.query(SononymMeta).delete()
    assert "No Sononym tempos to fold" in _cli(db, "tools", "analyze", "--only", "bpm-fix")


def test_the_audio_provider_has_no_tempo_of_sononyms(db):
    """The audio provider's descriptors are Fourier's own: no librosa tempo gated by
    Sononym's confidence in its own tempo."""
    from fourier.db.models import Sample, SampleFeatures
    from fourier.metadata import shadow
    with S.session_scope() as s:
        s.add(Sample(id=1, path="/l/SampleLibrary/V/Loops/L.wav", rel_path="V/Loops/L.wav",
                     filename="L.wav", duration_s=4.0))
        s.add(SampleFeatures(sample_id=1, tempo_bpm=120.0, bpm_reliable=1, n_events=8,
                             harmonic_percussive_ratio=0.3))
    with S.session_scope() as s:
        shadow.rebuild(s, shadow.AUDIO)
        names = {n for (n,) in s.execute(text("SELECT name FROM descriptors WHERE provider = 'audio'"))}
    assert names == {"analysed", "duration_s", "harmonicity"}


# --- Fourier's own analysis, with Sononym there ---

def _wav(path, y, sr=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y.astype("float32"), sr)


def test_own_analysis_runs_with_sononym_and_only_once(tmp_path, db):
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    sr = 22050
    t = np.arange(int(1.2 * sr)) / sr
    lib = tmp_path / "SampleLibrary" / "Acme"
    _wav(lib / "Tones" / "Tone A.wav", 0.5 * np.sin(2 * math.pi * 440.0 * t) * np.exp(-t))
    _wav(lib / "Hits" / "Noise.wav", 0.5 * np.random.default_rng(1).standard_normal(len(t)) * np.exp(-8 * t))
    rows = [("Tones/Tone A.wav", 0.95, 3.5), ("Hits/Noise.wav", 0.05, 1.05)]
    with S.session_scope() as s:
        for i, (rel, hpr, chroma) in enumerate(rows, 1):
            p = lib / rel
            s.add(Sample(id=i, path=str(p), rel_path=f"Acme/{rel}", filename=p.name, duration_s=1.2))
            # Sononym calls neither pitched: Fourier's own analysis doesn't ask it
            s.add(SononymMeta(sample_id=i, classes=["OneShot"], harmonicity=0.1, pitch_confidence=0.1))
            s.add(SampleFeatures(sample_id=i, harmonic_percussive_ratio=hpr, chroma_concentration=chroma,
                                 n_events=1, is_pitched=0))
    out = _cli(db, "tools", "analyze", "--only", "own", "--workers", "1")
    assert "Own roots (pYIN): 1 samples" in out and "1 with one clear pitch" in out
    with S.session_scope() as s:
        f = {x.sample_id: x for x in s.query(SampleFeatures)}
        assert abs(f[1].own_root_midi - 69.0) < 0.5 and f[1].own_root_at is not None
        assert f[2].own_root_at is None                     # a noise hit: not a candidate
        assert f[1].key_confidence is not None and f[2].key_confidence is None
    out = _cli(db, "tools", "analyze", "--only", "own", "--workers", "1")
    assert "Every tonal one-shot already has Fourier's own root" in out and "Key detection" not in out
    assert "Root notes" in _cli(db, "tools", "analyze", "--status")


# --- both readings: fourier why, db-stats --disagreements ---

def _two_readings(db):
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    from fourier.metadata import store
    with S.session_scope() as s:
        s.add(Sample(id=1, path="/l/SampleLibrary/Acme/Loops/Groove 01.wav", rel_path="Acme/Loops/Groove 01.wav",
                     filename="Groove 01.wav", duration_s=11.0344828))
        s.add(SononymMeta(sample_id=1, classes=["Loop"], categories=[], bpm=174.0, bpm_confidence=0.9,
                          harmonicity=0.2))
        s.add(SampleFeatures(sample_id=1, tempo_bpm=130.0, n_events=12, onset_rate_hz=6.0,
                             harmonic_percussive_ratio=0.2, chroma_concentration=1.2))
        s.add(Sample(id=2, path="/l/SampleLibrary/Acme/Kicks/Kick 01.wav", rel_path="Acme/Kicks/Kick 01.wav",
                     filename="Kick 01.wav", duration_s=0.4))
        s.add(SononymMeta(sample_id=2, classes=["OneShot"], categories=["Perc Kicks"], bpm=None,
                          harmonicity=0.2))
        s.add(SampleFeatures(sample_id=2, n_events=1, harmonic_percussive_ratio=0.2, chroma_concentration=1.1))
    with S.session_scope() as s:
        store.rebuild(s, store.SONONYM)


def test_why_shows_both_readings_where_they_differ(db):
    _two_readings(db)
    out = " ".join(_cli(db, "why", "--detail", "Groove 01").split())
    assert "readings differ: tempo 174 (Sononym) / 130 (Fourier's own analysis)" in out
    assert "tempo: 174 BPM (from Sononym's analysis)" in out
    assert "readings differ" not in _cli(db, "why", "--detail", "Kick 01")


def test_disagreements_are_a_report(db):
    _two_readings(db)
    before = _dump(db)
    out = " ".join(_cli(db, "tools", "db-stats", "--disagreements").split())
    assert "Sononym vs Fourier's own analysis, over 2 samples" in out
    assert "tempo (beyond an octave): 1 of 1 compared disagree (100.0%)" in out
    assert "Acme/Loops/Groove 01.wav: 174 (Sononym) / 130 (Fourier's own analysis)" in out
    assert "one-shot or loop: 0 of 2 compared disagree" in out
    assert _dump(db) == before                             # nothing changed


def test_why_names_the_source_of_a_picks_tempo(tmp_path):
    from types import SimpleNamespace

    from fourier.packs import why as W
    from fourier.packs.curate import left_out_doc
    p = "/lib/SampleLibrary/V/Loops/Break 01.wav"
    doc = left_out_doc("DRUMLOOPS", "built", 6, 6, {"total": 6}, {}, sources={"1": {"tempo": "sononym"}})
    r = SimpleNamespace(id=1, rel_path="V/Loops/Break 01.wav", path=p, filename="Break 01.wav",
                        categories="[]", classes='["Loop"]', ableton_tags="[]", harmonicity=0.2,
                        chroma_concentration=None, n_events=8, duration_s=8.0, file_hash="h",
                        bpm=120.0, tempo_bpm=120.0)
    man = {"categories": {"DRUMLOOPS": {"entries": [{"src": p, "out": "x/Break 01.wav"}]}}}
    w = W.explain_row(r, manifest=man, store_path=str(tmp_path / "n.json"),
                      why_docs=lambda cat: doc if cat == "DRUMLOOPS" else None)
    assert "the last build took: tempo from Sononym" in "\n".join(W.format_why(w))


# --- the synthetic library: the same picks before and after ---

def _synthetic(tmp_path, *args):
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "golden" / "synthetic_build.py"),
                        str(tmp_path / "work"), *args], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.slow
def test_a_migrated_database_builds_the_golden_before_and_after_own_analysis(tmp_path):
    """The golden's database (Sononym simulated) as the code before schema 2 made it: the build
    migrates it and picks exactly the golden; Fourier's own analysis then changes no pick."""
    out = _synthetic(tmp_path, "--schema-v1", "--own", "--check")
    assert out.count("golden: identical") == 2 and "Own roots (pYIN)" in out


@pytest.mark.skipif(importlib.util.find_spec("duckdb_engine") is None, reason="duckdb-engine not installed")
@pytest.mark.slow
def test_a_migrated_duckdb_database_builds_the_golden(tmp_path):
    assert "golden: identical" in _synthetic(tmp_path, "--schema-v1", "--duckdb", "--check")


def test_one_worker_runs_in_this_process(monkeypatch):
    """With one worker the pYIN and event steps map in this process: no pool is started, so no
    worker can die under them (it did once on a CI runner)."""
    import concurrent.futures as cf

    from fourier.cli import enrich

    def no_pool(*a, **k):
        raise AssertionError("a process pool was started for one worker")
    monkeypatch.setattr(cf, "ProcessPoolExecutor", no_pool)
    with enrich._jobs(1) as jobs:
        assert list(jobs(lambda x: x * 2, [1, 2, 3], chunksize=16)) == [2, 4, 6]
    with enrich._jobs(0) as jobs:
        assert list(jobs(str, [7])) == ["7"]
