"""Numbered database migrations, and the schema version a database is at.

A database records its version in `schema_version`. init_db creates the models' tables and
adds any nullable column a model gained (session._add_missing_columns); a change that needs
more than that (a new table from raw SQL, a data move, a column rename) is a numbered
migration here, applied once, in order, inside a transaction. Before a migration that changes
anything runs on an existing SQLite database, the database file is copied next to itself
(<name>.before-v<N>). Migrations only add: tables, columns, rows.

Version 1 is the schema the first numbered migration found (the models plus the
release, resolution, labels and descriptors tables their modules create), so it changes nothing.

Version 2 separates the analysis by source (no column holds values from two sources): it adds
sample_features.sononym_bpm_folded (Sononym's tempo folded into 60-200, filled here from
sononym_meta.bpm), own_root_midi and own_root_at (Fourier's own pYIN root, filled by
`fourier tools analyze --only own`). Every existing column keeps its values, so a build on a
migrated database picks exactly what it picked before.

Version 3 tells files apart by all their bytes: it adds samples.quick_hash (SHA256 of the first
8 KB, what file_hash held so far, copied from it) and samples.file_mtime_ns (the time the walk
read a file at, filled by the next scan). file_hash keeps its values (cut to the 16 hex digits
every scan wrote; an older demo wrote all 64); from here a scan stores the hash of the whole
file there for every file it reads in full (a new or changed file, or one whose first 8 KB
another file shares: ingest/importer.settle_hashes). On DuckDB it drops the index on
file_hash (see _v3_content_identity); SQLite indexes quick_hash. An older version opens a
version-3 database as before: it reads and writes neither new column.

After the migrations, every read-write open of a version-3 database (every_open) drops the
file_hash index on DuckDB again (rolling back to older code may make it again), and settles
rows an older version wrote into it (settle_older_rows): such a row has file_hash and no
quick_hash. Its file_hash, the quick hash, moves to quick_hash and file_hash is cleared, so
the next scan reads that file in full (its analysis stays unless its bytes changed). Rows this
version writes always have both, so on a home only this version wrote the open only reads.
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from sqlalchemy import text
from sqlalchemy.engine import Engine

log = logging.getLogger("fourier")


@dataclass(frozen=True)
class Migration:
    version: int
    what: str
    run: Callable | None = None          # run(connection); None: nothing to change


def _columns(conn, table: str) -> set[str] | None:
    try:
        return set(conn.execute(text(f"SELECT * FROM {table} LIMIT 0")).keys())
    except Exception:
        return None


# (column, type) migration 2 adds to sample_features (the model has them too, so init_db's
# _add_missing_columns may have added them already)
V2_COLUMNS = (("sononym_bpm_folded", "FLOAT"), ("own_root_midi", "FLOAT"), ("own_root_at", "TIMESTAMP"))
FOLD_LO, FOLD_HI = 60.0, 200.0
_FOLD_PASSES = 12                     # octaves either way; a tempo further out stays as it is


def _v2_sources_apart(conn) -> None:
    """Add Sononym's folded tempo and Fourier's own root columns; fill the folded tempo from
    Sononym's (the bpm-fix step's fold: doubled while under 60, halved while over 200, to two
    decimals). Touches no existing column."""
    have = _columns(conn, "sample_features")
    if have is None:                  # a database without the table (create_all makes it)
        return
    for col, typ in V2_COLUMNS:
        if col not in have:
            if typ == "TIMESTAMP" and conn.dialect.name == "sqlite":
                typ = "DATETIME"                  # as SQLAlchemy's DateTime is on SQLite
            conn.execute(text(f"ALTER TABLE sample_features ADD COLUMN {col} {typ}"))
    if _columns(conn, "sononym_meta") is None:
        return
    conn.execute(text(
        "UPDATE sample_features SET sononym_bpm_folded = (SELECT m.bpm FROM sononym_meta m "
        "WHERE m.sample_id = sample_features.sample_id) WHERE sononym_bpm_folded IS NULL AND "
        "sample_id IN (SELECT sample_id FROM sononym_meta WHERE bpm IS NOT NULL AND bpm > 0)"))
    for _ in range(_FOLD_PASSES):
        conn.execute(text(f"UPDATE sample_features SET sononym_bpm_folded = sononym_bpm_folded * 2 "
                          f"WHERE sononym_bpm_folded > 0 AND sononym_bpm_folded < {FOLD_LO}"))
        conn.execute(text(f"UPDATE sample_features SET sononym_bpm_folded = sononym_bpm_folded / 2 "
                          f"WHERE sononym_bpm_folded > {FOLD_HI}"))
    conn.execute(text("UPDATE sample_features SET sononym_bpm_folded = ROUND(sononym_bpm_folded, 2) "
                      "WHERE sononym_bpm_folded IS NOT NULL"))


# (column, type) migration 3 adds to samples (the model has them too)
V3_COLUMNS = (("quick_hash", "VARCHAR"), ("file_mtime_ns", "BIGINT"))


def _v3_content_identity(conn) -> None:
    """Add the quick hash and the walk's file time; the quick hash is what file_hash held. On
    DuckDB the index on file_hash goes: DuckDB can't update an indexed column of a row another
    table references (a sample with its analysis), and a scan now rewrites file_hash. SQLite
    keeps it and indexes the quick hash too."""
    have = _columns(conn, "samples")
    if have is None:
        return
    for col, typ in V3_COLUMNS:
        if col not in have:
            conn.execute(text(f"ALTER TABLE samples ADD COLUMN {col} {typ}"))
    if conn.dialect.name == "duckdb":
        conn.execute(text("DROP INDEX IF EXISTS ix_samples_file_hash"))
    else:
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_samples_quick_hash ON samples (quick_hash)"))
    conn.execute(text("UPDATE samples SET file_hash = SUBSTR(file_hash, 1, 16) WHERE LENGTH(file_hash) > 16"))
    conn.execute(text("UPDATE samples SET quick_hash = file_hash WHERE quick_hash IS NULL AND file_hash IS NOT NULL"))


def settle_older_rows(conn) -> None:
    """Rows an older version wrote into a version-3 database have file_hash (the quick hash
    then) and no quick_hash: the quick hash moves to quick_hash, and file_hash is cleared, so
    the next scan reads the file in full (importer.settle_hashes, _check_known) and keeps its
    analysis unless its bytes changed. Idempotent: rows this version writes always have both."""
    have = _columns(conn, "samples")
    if have is None or "quick_hash" not in have:
        return
    older = "quick_hash IS NULL AND file_hash IS NOT NULL"
    if conn.execute(text(f"SELECT 1 FROM samples WHERE {older} LIMIT 1")).first() is None:
        return                            # a read takes no write lock (SQLite)
    conn.execute(text(f"UPDATE samples SET quick_hash = SUBSTR(file_hash, 1, 16), file_hash = NULL "
                      f"WHERE {older}"))


def every_open(engine: Engine) -> None:
    """What every read-write open of a version-3 database does: on DuckDB, drop the file_hash
    index if it's there (_v3_content_identity says why; the rollback to older code may make it
    again), then settle_older_rows."""
    if engine.dialect.name == "duckdb":
        with engine.begin() as conn:
            conn.execute(text("DROP INDEX IF EXISTS ix_samples_file_hash"))
    with engine.begin() as conn:
        settle_older_rows(conn)


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline: the schema before numbered migrations"),
    Migration(2, "analysis sources apart: Sononym's folded tempo, Fourier's own root", _v2_sources_apart),
    Migration(3, "files told apart by all their bytes: the quick hash and the walk's file time",
              _v3_content_identity),
)
LATEST = MIGRATIONS[-1].version


def current(engine: Engine) -> int:
    """The version this database is at: 0 when it has never been migrated."""
    try:
        with engine.connect() as conn:
            v = conn.execute(text("SELECT MAX(version) FROM schema_version")).scalar()
        return int(v or 0)
    except Exception:
        return 0


def _backup(engine: Engine, version: int) -> Path | None:
    path = engine.url.database
    if engine.dialect.name != "sqlite" or not path or not Path(path).exists():
        return None
    dst = Path(f"{path}.before-v{version}")
    if not dst.exists():
        with engine.connect() as conn:          # fold the WAL in, so the copy is whole
            conn.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        shutil.copy2(path, dst)
    return dst


def migrate(engine: Engine) -> list[int]:
    """Apply the migrations this database hasn't had. Returns the versions applied."""
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS schema_version "
                          "(version INTEGER NOT NULL, applied_at TEXT NOT NULL)"))
    have = current(engine)
    done = []
    for m in MIGRATIONS:
        if m.version <= have:
            continue
        if m.run is not None and have > 0:
            b = _backup(engine, m.version)
            if b:
                log.warning("database: backed up to %s before migration %d", b, m.version)
        with engine.begin() as conn:
            if m.run is not None:
                m.run(conn)
            conn.execute(text("INSERT INTO schema_version (version, applied_at) VALUES (:v, :t)"),
                         {"v": m.version, "t": time.strftime("%Y-%m-%dT%H:%M:%S")})
        done.append(m.version)
    if max(have, *done, 0) >= 3:
        every_open(engine)
    return done
