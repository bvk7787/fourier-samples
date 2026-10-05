"""Numbered database migrations (fourier/db/migrations.py)."""
from sqlalchemy import create_engine, text

from fourier.db import migrations as M


def test_a_new_database_is_at_the_latest_version(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path / 'a.db'}")
    assert M.current(e) == 0
    assert M.migrate(e) == [m.version for m in M.MIGRATIONS]
    assert M.current(e) == M.LATEST and M.migrate(e) == []


def test_a_migration_runs_once_after_a_backup(tmp_path, monkeypatch):
    db = tmp_path / "lib.db"
    e = create_engine(f"sqlite:///{db}")
    M.migrate(e)
    ran = []

    def add_table(conn):
        ran.append(1)
        conn.execute(text("CREATE TABLE extra (x INTEGER)"))
    monkeypatch.setattr(M, "MIGRATIONS", M.MIGRATIONS + (M.Migration(M.LATEST + 1, "extra", add_table),))
    assert M.migrate(e) == [M.LATEST + 1] and ran == [1]
    assert (tmp_path / f"lib.db.before-v{M.LATEST + 1}").exists()
    assert M.migrate(e) == [] and ran == [1]
    with e.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM extra")).scalar() == 0
