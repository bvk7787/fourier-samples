"""Tests for db.session and db.models — SQLite backward-compat and DuckDB path."""

import os
import tempfile
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(suffix: str):
    """Create a fresh engine backed by a temp file."""
    tmp = tempfile.mktemp(suffix=suffix)
    from fourier.db import session as sess
    # Reset module-level singletons so each test gets a clean engine
    sess._engine = None
    sess._SessionLocal = None
    engine = sess.init_db(tmp)
    yield engine, tmp
    sess._engine = None
    sess._SessionLocal = None
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# _is_sqlite
# ---------------------------------------------------------------------------

def test_is_sqlite_dot_db():
    from fourier.db.session import _is_sqlite
    assert _is_sqlite(Path("library.db")) is True


def test_is_sqlite_dot_duckdb():
    from fourier.db.session import _is_sqlite
    assert _is_sqlite(Path("library.duckdb")) is False


def test_is_sqlite_full_path():
    from fourier.db.session import _is_sqlite
    assert _is_sqlite(Path("/home/user/.fourier/library.db")) is True
    assert _is_sqlite(Path("/home/user/.fourier/library.duckdb")) is False


# ---------------------------------------------------------------------------
# get_db_path backward-compat logic
# ---------------------------------------------------------------------------

def test_get_db_path_prefers_duckdb(tmp_path, monkeypatch):
    from fourier.db import session as sess
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FOURIER_HOME", raising=False)
    p = sess.get_db_path()
    assert p.suffix == ".duckdb"


def test_get_db_path_falls_back_to_sqlite(tmp_path, monkeypatch):
    from fourier.db import session as sess
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FOURIER_HOME", raising=False)
    # Create the SQLite DB but not the DuckDB one
    sqlite_path = tmp_path / ".fourier" / "library.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_path.touch()
    p = sess.get_db_path()
    assert p == sqlite_path


def test_get_db_path_duckdb_wins_when_both_exist(tmp_path, monkeypatch):
    from fourier.db import session as sess
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("FOURIER_HOME", raising=False)
    d = tmp_path / ".fourier"
    d.mkdir(parents=True, exist_ok=True)
    (d / "library.db").touch()
    (d / "library.duckdb").touch()
    p = sess.get_db_path()
    assert p.suffix == ".duckdb"


# ---------------------------------------------------------------------------
# SQLite engine smoke test
# ---------------------------------------------------------------------------

def test_sqlite_engine_creates_tables():
    for engine, tmp in _make_engine(".db"):
        from sqlalchemy import inspect as sa_inspect
        tables = sa_inspect(engine).get_table_names()
        for expected in ("samples", "sononym_meta", "sample_features", "packs", "pack_items"):
            assert expected in tables, f"Missing table: {expected}"


def test_sqlite_insert_and_autoincrement():
    for engine, tmp in _make_engine(".db"):
        from fourier.db.models import Sample
        with Session(engine) as s:
            s.add(Sample(path="/tmp/a.wav", filename="a.wav", is_favorite=False, is_hidden=False))
            s.add(Sample(path="/tmp/b.wav", filename="b.wav", is_favorite=False, is_hidden=False))
            s.commit()
        with Session(engine) as s:
            rows = s.query(Sample).order_by(Sample.id).all()
            assert len(rows) == 2
            assert rows[0].id == 1
            assert rows[1].id == 2


# ---------------------------------------------------------------------------
# DuckDB engine smoke test
# ---------------------------------------------------------------------------

def test_duckdb_engine_creates_tables():
    for engine, tmp in _make_engine(".duckdb"):
        from sqlalchemy import inspect as sa_inspect
        tables = sa_inspect(engine).get_table_names()
        for expected in ("samples", "sononym_meta", "sample_features", "packs", "pack_items"):
            assert expected in tables, f"Missing table: {expected}"


def test_duckdb_insert_and_autoincrement():
    for engine, tmp in _make_engine(".duckdb"):
        from fourier.db.models import Sample
        with Session(engine) as s:
            s.add(Sample(path="/tmp/a.wav", filename="a.wav", is_favorite=False, is_hidden=False))
            s.add(Sample(path="/tmp/b.wav", filename="b.wav", is_favorite=False, is_hidden=False))
            s.commit()
        with Session(engine) as s:
            rows = s.query(Sample).order_by(Sample.id).all()
            assert len(rows) == 2
            assert rows[0].id == 1
            assert rows[1].id == 2


# ---------------------------------------------------------------------------
# JSONText round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".db", ".duckdb"])
def test_jsontext_list_roundtrip(suffix):
    for engine, tmp in _make_engine(suffix):
        from fourier.db.models import Sample, SononymMeta
        with Session(engine) as s:
            sample = Sample(path="/tmp/x.wav", filename="x.wav", is_favorite=False, is_hidden=False)
            s.add(sample)
            s.flush()
            meta = SononymMeta(
                sample_id=sample.id,
                classes=["OneShot"],
                categories=["Perc Kicks"],
                class_strengths=[0.95],
                category_strengths=[0.88],
            )
            s.add(meta)
            s.commit()

        with Session(engine) as s:
            meta = s.query(SononymMeta).first()
            assert meta.classes == ["OneShot"]
            assert meta.categories == ["Perc Kicks"]
            assert abs(meta.class_strengths[0] - 0.95) < 1e-6
            assert abs(meta.category_strengths[0] - 0.88) < 1e-6


@pytest.mark.parametrize("suffix", [".db", ".duckdb"])
def test_jsontext_none_roundtrip(suffix):
    for engine, tmp in _make_engine(suffix):
        from fourier.db.models import Sample, SononymMeta
        with Session(engine) as s:
            sample = Sample(path="/tmp/y.wav", filename="y.wav", is_favorite=False, is_hidden=False)
            s.add(sample)
            s.flush()
            meta = SononymMeta(sample_id=sample.id)
            s.add(meta)
            s.commit()
        with Session(engine) as s:
            meta = s.query(SononymMeta).first()
            assert meta.classes is None
            assert meta.categories is None


# ---------------------------------------------------------------------------
# _table_cols helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".db", ".duckdb"])
def test_table_cols(suffix):
    from fourier.db import session as sess
    tmp = tempfile.mktemp(suffix=suffix)
    sess._engine = None
    sess._SessionLocal = None
    sess.init_db(tmp)
    from fourier.cli import _table_cols
    cols = _table_cols("samples")
    assert "id" in cols
    assert "path" in cols
    assert "filename" in cols
    assert "is_favorite" in cols
    sess._engine = None
    sess._SessionLocal = None
    os.unlink(tmp)


# ---------------------------------------------------------------------------
# get_engine lazy-init
# ---------------------------------------------------------------------------

def test_get_engine_lazy_init(tmp_path, monkeypatch):
    from fourier.db import session as sess
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    sess._engine = None
    sess._SessionLocal = None
    engine = sess.get_engine()
    assert engine is not None
    assert engine.dialect.name in ("sqlite", "duckdb")
    sess._engine = None
    sess._SessionLocal = None


def test_a_read_only_open_changes_nothing(tmp_path):
    """build --dry-run opens the database read-only: no write, not even its modification
    time; a database that's new or behind this code's schema opens as usual."""
    import os
    import time

    from sqlalchemy import text

    from fourier.db import session as S
    for name in ("lib.duckdb", "lib.db"):
        db = tmp_path / name
        S.use_db(db)
        assert not S.read_only()                      # not there yet: opens as usual
        from fourier.db.models import Sample
        with S.session_scope() as s:
            s.add(Sample(path="/l/a.wav", filename="a.wav"))
        S.get_engine().dispose()
        S.use_db(db)
        before = os.stat(db).st_mtime_ns
        time.sleep(0.05)
        assert S.read_only() and S.peek("SELECT COUNT(*) FROM samples") == 1
        with S.session_scope() as s:
            assert s.execute(text("SELECT COUNT(*) FROM samples")).scalar() == 1 and S.is_read_only()
        S.get_engine().dispose()
        assert os.stat(db).st_mtime_ns == before
        S.use_db(None)
        assert not S.is_read_only()
