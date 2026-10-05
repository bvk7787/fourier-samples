"""`fourier tools db-stats --missing --prune` removes the samples whose files are gone with
every row that refers to them, on DuckDB (which checks foreign keys against what is
committed) and on SQLite."""
from __future__ import annotations

import pytest
from click.testing import CliRunner


def _db_with_two_samples(db, tmp_path):
    from fourier.db import session as sess
    from fourier.db.models import Descriptor, Label, MissingFile, Sample, SampleFeatures, SononymMeta
    sess._engine = sess._SessionLocal = None
    sess.init_db(db)
    here = tmp_path / "Kick 01.wav"
    here.write_bytes(b"RIFF")
    s = sess.get_session()
    kept = Sample(path=str(here), filename=here.name)
    gone = Sample(path=str(tmp_path / "gone" / "Kick 02.wav"), filename="Kick 02.wav")
    s.add_all([kept, gone])
    s.flush()
    for sm in (kept, gone):
        s.add(SampleFeatures(sample_id=sm.id))
        s.add(SononymMeta(sample_id=sm.id, sononym_asset_id=sm.id))
        s.add(Label(sample_id=sm.id, provider="path", kind="canonical", label="kick", rank=0))
        s.add(Descriptor(sample_id=sm.id, provider="path", name="analysed", value=1.0))
    s.add(MissingFile(sample_id=gone.id, path=gone.path))
    s.commit()
    ids = kept.id, gone.id
    s.close()
    sess._engine = sess._SessionLocal = None
    return ids


@pytest.mark.parametrize("name", ["library.duckdb", "library.db"])
def test_prune_removes_missing_samples_and_their_rows(tmp_path, name):
    from sqlalchemy import text

    from fourier.cli import main
    from fourier.db import session as sess
    db = tmp_path / name
    kept, gone = _db_with_two_samples(db, tmp_path)
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "db-stats", "--missing", "--prune"])
    assert r.exit_code == 0, r.output
    assert "Removed 1 samples" in r.output
    sess._engine = sess._SessionLocal = None
    sess.init_db(db)
    s = sess.get_session()
    try:
        for table, col in (("samples", "id"), ("sample_features", "sample_id"),
                           ("sononym_meta", "sample_id"), ("labels", "sample_id"),
                           ("descriptors", "sample_id"), ("missing_files", "sample_id")):
            got = {i for (i,) in s.execute(text(f"SELECT {col} FROM {table}"))}
            assert gone not in got, table
            assert table == "missing_files" or kept in got, table
    finally:
        s.close()
        sess._engine = sess._SessionLocal = None

