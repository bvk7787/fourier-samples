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


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "baseline: the schema before numbered migrations"),
    Migration(2, "analysis sources apart: Sononym's folded tempo, Fourier's own root", _v2_sources_apart),
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
    return done
