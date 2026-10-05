"""Database session management."""

from __future__ import annotations

import re
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None
_PATH: Path | str | None = None      # use_db(): the database the first use opens
_READ_ONLY: Path | None = None        # read_only(): this database opens read-only
_RO_ENGINE: Engine | None = None      # ...and the engine that did


def default_db_path() -> Path:
    """Where the default DB is, without creating anything: <FOURIER_HOME>/library.duckdb
    (FOURIER_HOME defaults to ~/.fourier), or <FOURIER_HOME>/library.db when only that
    exists (an older home's SQLite database)."""
    from ..paths import fourier_home
    db_dir = fourier_home()
    duckdb_path = db_dir / "library.duckdb"
    sqlite_path = db_dir / "library.db"
    if sqlite_path.exists() and not duckdb_path.exists():
        return sqlite_path
    return duckdb_path


def get_db_path() -> Path:
    """
    Default DB location (default_db_path), its folder created.

    Prefers <FOURIER_HOME>/library.duckdb (DuckDB format; FOURIER_HOME defaults to
    ~/.fourier). Falls back to <FOURIER_HOME>/library.db if it exists (SQLite backward compat).
    """
    path = default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def use_db(db_path: Path | str | None = None) -> None:
    """Point this process at a database without opening it: the first get_engine() or
    get_session() opens it (creating it when it's new), so a command that never reads the
    database (--help, doctor on a new machine) leaves no file behind."""
    global _engine, _SessionLocal, _PATH, _READ_ONLY
    _PATH, _engine, _SessionLocal, _READ_ONLY = db_path, None, None, None


def read_only() -> bool:
    """Open this process's database read-only from now on (build --dry-run): nothing is
    created, migrated or written, so even its modification time stays. False, and nothing
    changes, when there's no database yet or its schema is behind this code's (it opens
    as usual then, and is brought up to date)."""
    global _engine, _SessionLocal, _READ_ONLY
    path = db_path_in_use()
    if not path.exists():
        return False
    if _engine is not None:          # one configuration per database file in a process
        _engine.dispose()
        _engine = _SessionLocal = None
    from .migrations import LATEST, current
    try:
        eng = _open(path, True)
        ok = current(eng) == LATEST and not _missing_columns(eng)
        eng.dispose()
    except Exception:
        ok = False
    if not ok:
        return False
    _READ_ONLY = path
    return True


def peek(sql: str):
    """One value from the database, read through a read-only connection that is closed again
    (a check before a command opens it; a dry run then leaves the file as it was). None when
    there's no database, or it can't be read that way."""
    path = db_path_in_use()
    if _engine is not None or not path.exists():
        if _engine is None:
            return None
        with _engine.connect() as conn:
            return conn.execute(text(sql)).scalar()
    try:
        eng = _open(path, True)
        try:
            with eng.connect() as conn:
                return conn.execute(text(sql)).scalar()
        finally:
            eng.dispose()
    except Exception:
        return None


def is_read_only() -> bool:
    """Whether this process's database is open read-only (read_only())."""
    return _engine is not None and _engine is _RO_ENGINE


def db_path_in_use() -> Path:
    """The database this process uses or will open (use_db's, else the default)."""
    if _engine is not None and _engine.url.database:
        return Path(_engine.url.database)
    return Path(_PATH) if _PATH else default_db_path()


def db_exists() -> bool:
    """Whether the database this process uses is there yet (nothing is created)."""
    return _engine is not None or db_path_in_use().exists()


def _is_sqlite(path: Path) -> bool:
    """Return True if path points to a SQLite database."""
    name = str(path)
    return name.endswith(".db") and not name.endswith(".duckdb")


def get_engine() -> Engine:
    """Return the current engine, initializing with defaults if needed."""
    if _engine is None:
        init_db(_PATH)
    return _engine


def _open(path: Path, read_only: bool) -> Engine:
    """An engine on an existing database, opened read-only (nothing created or migrated)."""
    if _is_sqlite(path):
        return create_engine(f"sqlite:///file:{path}?mode=ro&uri=true", echo=False)
    return create_engine(f"duckdb:///{path}", echo=False, connect_args={"read_only": read_only})


def init_db(db_path: Path | str | None = None) -> Engine:
    """Initialize the database, creating tables if needed (read_only(): open it as it is)."""
    global _engine, _SessionLocal

    global _RO_ENGINE
    if _READ_ONLY is not None and Path(db_path or default_db_path()) == _READ_ONLY:
        _engine = _RO_ENGINE = _open(_READ_ONLY, True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
        return _engine
    path = Path(db_path) if db_path else get_db_path()

    if _is_sqlite(path):
        url = f"sqlite:///{path}"
        _engine = create_engine(url, echo=False)

        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.close()
    else:
        url = f"duckdb:///{path}"
        try:
            _engine = create_engine(url, echo=False)
        except Exception as e:
            from .. import REINSTALL
            raise RuntimeError(
                f"the database at {path} didn't open ({e}). If duckdb or duckdb-engine is "
                f"missing, {REINSTALL}"
            ) from e

        # DuckDB 1.5+ dropped SERIAL. Create one sequence per autoincrement PK
        # up front, then rewrite SERIAL -> DEFAULT nextval() in DDL.
        with _engine.begin() as _conn:
            for tbl in Base.metadata.tables.values():
                for col in tbl.columns:
                    if col.primary_key and col.name == "id":
                        _conn.execute(
                            text(f"CREATE SEQUENCE IF NOT EXISTS {tbl.name}_id_seq")
                        )

        @event.listens_for(_engine, "before_cursor_execute", retval=True)
        def _fix_serial_ddl(conn, cursor, statement, parameters, context, executemany):
            if "SERIAL" not in statement:
                return statement, parameters
            m = re.search(r"CREATE TABLE\s+\"?(\w+)\"?", statement, re.IGNORECASE)
            if m:
                table_name = m.group(1)
                statement = statement.replace(
                    "SERIAL NOT NULL",
                    f"INTEGER DEFAULT nextval('{table_name}_id_seq') NOT NULL",
                )
            statement = statement.replace("SERIAL", "INTEGER")
            return statement, parameters

    Base.metadata.create_all(_engine)
    _add_missing_columns(_engine)
    from .migrations import migrate
    migrate(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def table_columns(engine: Engine, table: str) -> set[str] | None:
    """Column names of a table, or None if it doesn't exist. Reads a zero-row SELECT
    rather than dialect reflection, which some drivers (duckdb-engine) break on."""
    try:
        with engine.connect() as conn:
            return set(conn.execute(text(f"SELECT * FROM {table} LIMIT 0")).keys())
    except Exception:
        return None


def _missing_columns(engine: Engine) -> list[str]:
    """'table.column' a model has and an existing table lacks (what _add_missing_columns adds)."""
    out = []
    for tbl in Base.metadata.sorted_tables:
        have = table_columns(engine, tbl.name)
        if have is None:
            out.append(tbl.name)
            continue
        out += [f"{tbl.name}.{c.name}" for c in tbl.columns if c.name not in have]
    return out


def _add_missing_columns(engine: Engine) -> list[str]:
    """create_all() never alters an existing table, so a nullable column added to a
    model later would make every SELECT of that model fail on an older database.
    Add any such column (NULL for existing rows). Returns 'table.column' added."""
    added = []
    for tbl in Base.metadata.sorted_tables:
        have = table_columns(engine, tbl.name)
        if have is None:
            continue
        for col in tbl.columns:
            if col.name in have or col.primary_key or not col.nullable:
                continue
            ddl = col.type.compile(dialect=engine.dialect)
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {tbl.name} ADD COLUMN {col.name} {ddl}"))
                added.append(f"{tbl.name}.{col.name}")
            except Exception as e:   # leave the DB as it was; the SELECT will say why
                import logging
                logging.getLogger("fourier").warning(
                    "could not add column %s.%s: %s", tbl.name, col.name, e)
    return added


def get_session() -> Session:
    """Get a new session. Caller is responsible for closing."""
    if _SessionLocal is None:
        init_db(_PATH)
    return _SessionLocal()  # type: ignore[misc]


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for a transactional session."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Tables that hold a row per sample (by sample_id), deleted with it: the ones with a foreign
# key first, then the derived ones (labels, descriptors, the walk's marks, the resolution).
# release_file keeps its rows: a release records what it held, whatever the database holds now.
_SAMPLE_TABLES = ("pack_items", "sononym_meta", "sample_features", "labels", "descriptors",
                  "missing_files", "sample_resolution")


def delete_samples(session: Session, ids) -> int:
    """Remove these samples and every row that refers to them. SQLite does it in one
    transaction. DuckDB checks a foreign key against what is committed, so a parent row can't
    go in the transaction that removed its children: there the dependent rows are removed and
    committed first, then the samples (a stop between the two leaves samples without their
    analysis, which the next `fourier tools analyze` redoes). Returns the samples removed."""
    ids = sorted({int(i) for i in ids})
    if not ids:
        return 0
    bind = session.get_bind()
    have = {t for t in _SAMPLE_TABLES if table_columns(bind, t) is not None}
    duck = bind.dialect.name == "duckdb"

    def chunks():
        for i in range(0, len(ids), 500):
            yield ", ".join(str(x) for x in ids[i:i + 500])

    for table in _SAMPLE_TABLES:
        if table in have:
            for chunk in chunks():
                session.execute(text(f"DELETE FROM {table} WHERE sample_id IN ({chunk})"))
    if duck:
        session.commit()
    n = 0
    for chunk in chunks():          # counted first: DuckDB reports no rowcount for a DELETE
        n += session.execute(text(f"SELECT COUNT(*) FROM samples WHERE id IN ({chunk})")).scalar() or 0
        session.execute(text(f"DELETE FROM samples WHERE id IN ({chunk})"))
    session.commit()
    return n

