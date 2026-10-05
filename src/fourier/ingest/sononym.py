"""
Sononym database reader.

Reads directly from Sononym's library DB and imports samples into the
fourier database, preserving all analyzed metadata.

Supports both:
  - SQLite format (Sononym <= 1.5.x)
  - DuckDB format (Sononym 1.6+, identified by DUCK magic bytes)

Sononym DB locations:
  - Library: sononym.db in the library folder (fourier.toml `sononym_db` names another)
  - File Browser: ~/Library/Application Support/Sononym/<version>/File Browser/*.db
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

def default_sononym_db() -> Path | None:
    """fourier.toml's sononym_db, else sononym.db in the first library folder that has one."""
    from ..places import sononym_db
    return sononym_db()


def default_library_root() -> Path | None:
    """Where Sononym's relative paths resolve: the library folder its DB sits in, else the
    first library folder."""
    from ..places import library_roots
    roots = library_roots()
    db = default_sononym_db()
    for r in roots:
        if db is not None and db.parent == r:
            return r
    return roots[0] if roots else None

_SQLITE_MAGIC = b"SQLite format 3"


def _detect_db_format(db_path: Path) -> str:
    """Return 'duckdb' or 'sqlite'."""
    try:
        with open(db_path, "rb") as f:
            header = f.read(16)
    except OSError as e:
        raise FileNotFoundError(f"Cannot read DB: {db_path}") from e
    if b"DUCK" in header[:16]:
        return "duckdb"
    if header[:15] == _SQLITE_MAGIC:
        return "sqlite"
    return "sqlite"  # fallback


def _load_tags_taxonomy() -> dict[str, dict]:
    """
    Load a Sononym tags taxonomy (UUID -> {name, path, is_group}) from
    fourier/data/sononym_tags_taxonomy.json, if a user made one (none ships with Fourier).
    Returns an empty dict if the taxonomy file is not found.
    """
    try:
        from importlib import resources
        pkg_data = resources.files("fourier.data")
        with (pkg_data / "sononym_tags_taxonomy.json").open("r") as f:
            data = json.load(f)
            return data.get("tags", {})
    except Exception:
        taxonomy_path = Path(__file__).parent.parent / "data" / "sononym_tags_taxonomy.json"
        if taxonomy_path.exists():
            with open(taxonomy_path) as f:
                data = json.load(f)
                return data.get("tags", {})
        log.debug("Sononym tags taxonomy not found; tags will not be resolved")
        return {}


@dataclass
class SononymAsset:
    """A single asset row from Sononym's database, fully parsed."""

    sononym_id: int
    rel_path: str          # relative to library root
    abs_path: Path         # resolved absolute path
    basename: str
    modtime: int
    status: str

    # Format
    file_type: str | None
    file_size: int | None
    duration_s: float | None
    sample_rate: int | None
    channels: int | None
    bit_depth: int | None

    # Classification
    classes: list[str]
    class_strengths: list[float]
    categories: list[str]
    category_strengths: list[float]

    # Tonal
    pitch_class: str | None
    base_note: float | None
    base_note_confidence: float | None

    # Dynamics
    peak_db: float | None
    rms_db: float | None
    crest_factor: float | None

    # Rhythm
    bpm: float | None
    bpm_confidence: float | None

    # Timbral
    brightness: float | None
    noisiness: float | None
    harmonicity: float | None

    # User
    is_favorite: bool
    is_hidden: bool

    # Extended (from assets_extended)
    class_signature: list[float] | None = None
    category_signature: list[float] | None = None
    pitch_confidence: float | None = None

    # Sononym 1.6+: resolved tag path strings
    # e.g. ["Instruments > Drums & Percussion > Kick"]
    sononym_tags: list[str] = field(default_factory=list)

    @property
    def is_oneshot(self) -> bool:
        return "OneShot" in self.classes

    @property
    def is_loop(self) -> bool:
        return "Loop" in self.classes

    @property
    def primary_category(self) -> str | None:
        return self.categories[0] if self.categories else None


def _parse_json_list(value: str | list | None, fallback: list | None = None) -> list:
    """Parse a JSON array string or pass through a native list (DuckDB)."""
    if value is None:
        return fallback or []
    if isinstance(value, list):
        return value
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else (fallback or [])
    except (json.JSONDecodeError, TypeError):
        return fallback or []


def _parse_float_blob(blob: bytes | list | None) -> list[float] | None:
    """
    Parse Sononym's VR (vector real) blob or native array.

    - DuckDB returns native Python lists -> pass through.
    - SQLite returns JSON text or raw float32 binary blobs.
    """
    if blob is None:
        return None
    if isinstance(blob, list):
        return blob
    if isinstance(blob, (bytes, bytearray)):
        try:
            text = blob.decode("utf-8")
            return json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        try:
            n = len(blob) // 4
            return list(struct.unpack(f"<{n}f", blob[:n * 4]))
        except struct.error:
            return None
    if isinstance(blob, str):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None
    return None


class SononymReader:
    """
    Reads and streams assets from a Sononym library database.

    Auto-detects whether the DB is SQLite (Sononym <= 1.5) or DuckDB (Sononym 1.6+)
    and uses the appropriate backend. The public API is identical in both cases.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        library_root: Path | str | None = None,
    ):
        db_path = db_path or default_sononym_db()
        if db_path is None:
            raise FileNotFoundError("no Sononym DB: set sononym_db (or library) in fourier.toml")
        self.db_path = Path(db_path)
        library_root = library_root or default_library_root() or self.db_path.parent
        self.library_root = Path(library_root)

        if not self.db_path.exists():
            raise FileNotFoundError(f"Sononym DB not found: {self.db_path}")

        self.db_format = _detect_db_format(self.db_path)
        log.info(f"Sononym DB: {self.db_path} (format: {self.db_format})")
        log.info(f"Library root: {self.library_root}")

        self._tags_taxonomy: dict[str, dict] = _load_tags_taxonomy()

    # -------------------------------------------------------------------------
    # Connection helpers
    # -------------------------------------------------------------------------

    def _connect_sqlite(self):
        import sqlite3
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        return conn

    def _connect_duckdb(self):
        try:
            import duckdb
        except ImportError as e:
            from .. import REINSTALL
            raise ImportError(f"reading a Sononym 1.6+ database needs duckdb: {REINSTALL}") from e
        return duckdb.connect(str(self.db_path), read_only=True)

    def _connect(self):
        """Backward-compat alias used by importer.py."""
        return self._connect_sqlite()

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def count(self) -> int:
        """Total number of assets in the Sononym DB."""
        if self.db_format == "duckdb":
            conn = self._connect_duckdb()
            try:
                return conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
            finally:
                conn.close()
        else:
            with self._connect_sqlite() as conn:
                return conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]

    def get_categories(self) -> list[str]:
        """Return all known Sononym OneShot categories."""
        if self.db_format == "duckdb":
            conn = self._connect_duckdb()
            try:
                row = conn.execute(
                    "SELECT classes FROM classes WHERE classifier = 'OneShot-Categories'"
                ).fetchone()
                if row:
                    val = row[0]
                    return val if isinstance(val, list) else json.loads(val)
                return []
            finally:
                conn.close()
        else:
            with self._connect_sqlite() as conn:
                row = conn.execute(
                    "SELECT classes FROM classes WHERE classifier = 'OneShot-Categories'"
                ).fetchone()
                if row:
                    return json.loads(row["classes"])
                return []

    def stream_assets(
        self,
        category_filter: list[str] | None = None,
        class_filter: list[str] | None = None,
        file_types: list[str] | None = None,
        favorites_only: bool = False,
        min_duration_s: float | None = None,
        max_duration_s: float | None = None,
        batch_size: int = 500,
        include_extended: bool = True,
        only_existing: bool = True,
        include_tags: bool = True,
    ) -> Iterator[SononymAsset]:
        """
        Stream SononymAsset objects from the DB.

        Args:
            category_filter: e.g. ["Perc Kicks", "Perc Snares"]
            class_filter: ["OneShot"] or ["Loop"]
            file_types: ["wav", "aif"]
            favorites_only: only is_favorite = 1
            min_duration_s / max_duration_s: duration range
            batch_size: rows per DB fetch
            include_extended: join assets_extended for signatures
            only_existing: skip assets whose abs_path doesn't exist on disk
            include_tags: (DuckDB only) resolve assets_tags to tag name paths
        """
        if self.db_format == "duckdb":
            yield from self._stream_duckdb(
                category_filter=category_filter,
                class_filter=class_filter,
                file_types=file_types,
                favorites_only=favorites_only,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
                batch_size=batch_size,
                include_extended=include_extended,
                only_existing=only_existing,
                include_tags=include_tags,
            )
        else:
            yield from self._stream_sqlite(
                category_filter=category_filter,
                class_filter=class_filter,
                file_types=file_types,
                favorites_only=favorites_only,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
                batch_size=batch_size,
                include_extended=include_extended,
                only_existing=only_existing,
            )

    def get_all_modtimes(self) -> dict[str, int]:
        """
        Lightweight snapshot: {rel_path: modtime} for all succeeded assets.

        Used by sync to identify new/changed/removed assets without a full
        stream. Runs a single query (milliseconds, even for a large library).
        """
        if self.db_format == "duckdb":
            conn = self._connect_duckdb()
            try:
                rows = conn.execute(
                    "SELECT filename, modtime FROM assets WHERE status = 'succeeded'"
                ).fetchall()
                return {row[0]: (row[1] or 0) for row in rows}
            finally:
                conn.close()
        else:
            with self._connect_sqlite() as conn:
                rows = conn.execute(
                    "SELECT filename, modtime FROM assets WHERE status = 'succeeded'"
                ).fetchall()
            return {row["filename"]: (row["modtime"] or 0) for row in rows}

    def get_asset_by_rel_path(self, rel_path: str) -> SononymAsset | None:
        """Look up a single asset by its relative path."""
        if self.db_format == "duckdb":
            return self._get_asset_duckdb(rel_path)
        else:
            return self._get_asset_sqlite(rel_path)

    # -------------------------------------------------------------------------
    # DuckDB implementation
    # -------------------------------------------------------------------------

    _ASSET_SELECT_COLS = """
        a.id, a.filename, a.basename, a.modtime, a.status,
        a.file_type_S, a.file_size_R, a.file_length_R,
        a.file_sample_rate_R, a.file_channel_count_R, a.file_bit_depth_R,
        a.classes_VS, a.class_strengths_VR,
        a.categories_VS, a.category_strengths_VR,
        a.pitch_class_S, a.base_note_R, a.base_note_confidence_R,
        a.peak_db_R, a.rms_db_R, a.crest_factor_R,
        a.bpm_R, a.bpm_confidence_R,
        a.brightness_R, a.noisiness_R, a.harmonicity_R,
        a.is_favorite, a.is_hidden
    """

    def _build_duckdb_query(
        self,
        category_filter: list[str] | None,
        class_filter: list[str] | None,
        file_types: list[str] | None,
        favorites_only: bool,
        min_duration_s: float | None,
        max_duration_s: float | None,
        include_extended: bool,
    ) -> tuple[str, list]:
        where_clauses = ["a.status = 'succeeded'"]
        params: list[Any] = []

        if file_types:
            placeholders = ", ".join("?" * len(file_types))
            where_clauses.append(f"a.file_type_S IN ({placeholders})")
            params.extend(file_types)

        if favorites_only:
            where_clauses.append("a.is_favorite = 1")

        if min_duration_s is not None:
            where_clauses.append("a.file_length_R >= ?")
            params.append(min_duration_s)

        if max_duration_s is not None:
            where_clauses.append("a.file_length_R <= ?")
            params.append(max_duration_s)

        if category_filter:
            # categories_VS is a native VARCHAR[] in DuckDB
            cat_clauses = ["list_contains(a.categories_VS, ?)" for _ in category_filter]
            where_clauses.append(f"({' OR '.join(cat_clauses)})")
            params.extend(category_filter)

        if class_filter:
            cls_clauses = ["list_contains(a.classes_VS, ?)" for _ in class_filter]
            where_clauses.append(f"({' OR '.join(cls_clauses)})")
            params.extend(class_filter)

        where_sql = " AND ".join(where_clauses)

        ext_cols = (
            "ae.class_signature_VR, ae.category_signature_VR, ae.pitch_confidence_R"
            if include_extended
            else "NULL as class_signature_VR, NULL as category_signature_VR, NULL as pitch_confidence_R"
        )
        join = "LEFT JOIN assets_extended ae ON ae.asset_id = a.id" if include_extended else ""

        query = f"""
            SELECT {self._ASSET_SELECT_COLS}, {ext_cols}
            FROM assets a
            {join}
            WHERE {where_sql}
            ORDER BY a.id
        """
        return query, params

    def _row_to_asset_duckdb(self, row: tuple, tags: list[str] | None = None) -> SononymAsset:
        (
            asset_id, filename, basename, modtime, status,
            file_type, file_size, file_length,
            sample_rate, channels, bit_depth,
            classes_vs, class_strengths_vr,
            categories_vs, category_strengths_vr,
            pitch_class, base_note, base_note_confidence,
            peak_db, rms_db, crest_factor,
            bpm, bpm_confidence,
            brightness, noisiness, harmonicity,
            is_favorite, is_hidden,
            class_sig_vr, cat_sig_vr, pitch_conf,
        ) = row

        return SononymAsset(
            sononym_id=asset_id,
            rel_path=filename,
            abs_path=self.library_root / filename,
            basename=basename,
            modtime=modtime or 0,
            status=status,
            file_type=file_type,
            file_size=file_size,
            duration_s=file_length,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            classes=_parse_json_list(classes_vs),
            class_strengths=_parse_json_list(class_strengths_vr),
            categories=_parse_json_list(categories_vs),
            category_strengths=_parse_json_list(category_strengths_vr),
            pitch_class=pitch_class,
            base_note=base_note,
            base_note_confidence=base_note_confidence,
            peak_db=peak_db,
            rms_db=rms_db,
            crest_factor=crest_factor,
            bpm=bpm,
            bpm_confidence=bpm_confidence,
            brightness=brightness,
            noisiness=noisiness,
            harmonicity=harmonicity,
            is_favorite=bool(is_favorite),
            is_hidden=bool(is_hidden),
            class_signature=_parse_float_blob(class_sig_vr),
            category_signature=_parse_float_blob(cat_sig_vr),
            pitch_confidence=pitch_conf,
            sononym_tags=tags or [],
        )

    def _fetch_tags_for_ids(self, conn, asset_ids: list[int]) -> dict[int, list[str]]:
        """Fetch resolved tag paths for a batch of asset IDs (DuckDB only)."""
        if not self._tags_taxonomy or not asset_ids:
            return {}

        placeholders = ", ".join("?" * len(asset_ids))
        rows = conn.execute(
            f"SELECT asset_id, tag_uuid FROM assets_tags "
            f"WHERE asset_id IN ({placeholders}) AND is_deleted = false",
            asset_ids,
        ).fetchall()

        result: dict[int, list[str]] = {}
        for asset_id, tag_uuid in rows:
            tag_info = self._tags_taxonomy.get(str(tag_uuid))
            if tag_info:
                result.setdefault(asset_id, []).append(tag_info["path"])
        return result

    def _stream_duckdb(
        self,
        category_filter: list[str] | None,
        class_filter: list[str] | None,
        file_types: list[str] | None,
        favorites_only: bool,
        min_duration_s: float | None,
        max_duration_s: float | None,
        batch_size: int,
        include_extended: bool,
        only_existing: bool,
        include_tags: bool,
    ) -> Iterator[SononymAsset]:
        query, params = self._build_duckdb_query(
            category_filter=category_filter,
            class_filter=class_filter,
            file_types=file_types,
            favorites_only=favorites_only,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            include_extended=include_extended,
        )

        conn = self._connect_duckdb()
        has_tags_table = (
            include_tags
            and bool(self._tags_taxonomy)
            and conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name='assets_tags'"
            ).fetchone()[0] > 0
        )

        try:
            offset = 0
            while True:
                batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                rows = conn.execute(batch_query, params).fetchall()
                if not rows:
                    break

                tags_map: dict[int, list[str]] = {}
                if has_tags_table:
                    batch_ids = [r[0] for r in rows]
                    tags_map = self._fetch_tags_for_ids(conn, batch_ids)

                for row in rows:
                    asset = self._row_to_asset_duckdb(row, tags=tags_map.get(row[0], []))
                    if only_existing and not asset.abs_path.exists():
                        continue
                    yield asset

                offset += batch_size
        finally:
            conn.close()

    def _get_asset_duckdb(self, rel_path: str) -> SononymAsset | None:
        conn = self._connect_duckdb()
        try:
            row = conn.execute(
                f"""
                SELECT {self._ASSET_SELECT_COLS},
                    ae.class_signature_VR, ae.category_signature_VR, ae.pitch_confidence_R
                FROM assets a
                LEFT JOIN assets_extended ae ON ae.asset_id = a.id
                WHERE a.filename = ?
                """,
                (rel_path,),
            ).fetchone()
            if not row:
                return None
            asset_id = row[0]
            tags_map = self._fetch_tags_for_ids(conn, [asset_id])
            return self._row_to_asset_duckdb(row, tags=tags_map.get(asset_id, []))
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # SQLite implementation (backward-compatible with Sononym <= 1.5)
    # -------------------------------------------------------------------------

    def _stream_sqlite(
        self,
        category_filter: list[str] | None,
        class_filter: list[str] | None,
        file_types: list[str] | None,
        favorites_only: bool,
        min_duration_s: float | None,
        max_duration_s: float | None,
        batch_size: int,
        include_extended: bool,
        only_existing: bool,
    ) -> Iterator[SononymAsset]:
        where_clauses = ["a.status = 'succeeded'"]
        params: list[Any] = []

        if file_types:
            placeholders = ",".join("?" * len(file_types))
            where_clauses.append(f"a.file_type_S IN ({placeholders})")
            params.extend(file_types)

        if favorites_only:
            where_clauses.append("a.is_favorite = 1")

        if min_duration_s is not None:
            where_clauses.append("a.file_length_R >= ?")
            params.append(min_duration_s)

        if max_duration_s is not None:
            where_clauses.append("a.file_length_R <= ?")
            params.append(max_duration_s)

        if category_filter:
            cat_clauses = [
                "a.categories_VS LIKE '%' || ? || '%'" for _ in category_filter
            ]
            where_clauses.append(f"({' OR '.join(cat_clauses)})")
            params.extend(category_filter)

        if class_filter:
            cls_clauses = [
                "a.classes_VS LIKE '%' || ? || '%'" for _ in class_filter
            ]
            where_clauses.append(f"({' OR '.join(cls_clauses)})")
            params.extend(class_filter)

        where_sql = " AND ".join(where_clauses)

        if include_extended:
            select_sql = """
                SELECT
                    a.id, a.filename, a.basename, a.modtime, a.status,
                    a.file_type_S, a.file_size_R, a.file_length_R,
                    a.file_sample_rate_R, a.file_channel_count_R, a.file_bit_depth_R,
                    a.classes_VS, a.class_strengths_VR,
                    a.categories_VS, a.category_strengths_VR,
                    a.pitch_class_S, a.base_note_R, a.base_note_confidence_R,
                    a.peak_db_R, a.rms_db_R, a.crest_factor_R,
                    a.bpm_R, a.bpm_confidence_R,
                    a.brightness_R, a.noisiness_R, a.harmonicity_R,
                    a.is_favorite, a.is_hidden,
                    ae.class_signature_VR, ae.category_signature_VR,
                    ae.pitch_confidence_R
                FROM assets a
                LEFT JOIN assets_extended ae ON ae.asset_id = a.id
            """
        else:
            select_sql = """
                SELECT
                    a.id, a.filename, a.basename, a.modtime, a.status,
                    a.file_type_S, a.file_size_R, a.file_length_R,
                    a.file_sample_rate_R, a.file_channel_count_R, a.file_bit_depth_R,
                    a.classes_VS, a.class_strengths_VR,
                    a.categories_VS, a.category_strengths_VR,
                    a.pitch_class_S, a.base_note_R, a.base_note_confidence_R,
                    a.peak_db_R, a.rms_db_R, a.crest_factor_R,
                    a.bpm_R, a.bpm_confidence_R,
                    a.brightness_R, a.noisiness_R, a.harmonicity_R,
                    a.is_favorite, a.is_hidden,
                    NULL as class_signature_VR, NULL as category_signature_VR,
                    NULL as pitch_confidence_R
                FROM assets a
            """

        query = f"{select_sql} WHERE {where_sql} ORDER BY a.id"

        conn = self._connect_sqlite()
        try:
            offset = 0
            while True:
                batch_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
                rows = conn.execute(batch_query, params).fetchall()
                if not rows:
                    break

                for row in rows:
                    asset = self._row_to_asset(row)
                    if only_existing and not asset.abs_path.exists():
                        continue
                    yield asset

                offset += batch_size
        finally:
            conn.close()

    def _row_to_asset(self, row) -> SononymAsset:
        """Convert a sqlite3.Row to a SononymAsset."""
        rel_path = row["filename"]
        abs_path = self.library_root / rel_path

        return SononymAsset(
            sononym_id=row["id"],
            rel_path=rel_path,
            abs_path=abs_path,
            basename=row["basename"],
            modtime=row["modtime"] or 0,
            status=row["status"],
            file_type=row["file_type_S"],
            file_size=row["file_size_R"],
            duration_s=row["file_length_R"],
            sample_rate=row["file_sample_rate_R"],
            channels=row["file_channel_count_R"],
            bit_depth=row["file_bit_depth_R"],
            classes=_parse_json_list(row["classes_VS"]),
            class_strengths=_parse_json_list(row["class_strengths_VR"]),
            categories=_parse_json_list(row["categories_VS"]),
            category_strengths=_parse_json_list(row["category_strengths_VR"]),
            pitch_class=row["pitch_class_S"],
            base_note=row["base_note_R"],
            base_note_confidence=row["base_note_confidence_R"],
            peak_db=row["peak_db_R"],
            rms_db=row["rms_db_R"],
            crest_factor=row["crest_factor_R"],
            bpm=row["bpm_R"],
            bpm_confidence=row["bpm_confidence_R"],
            brightness=row["brightness_R"],
            noisiness=row["noisiness_R"],
            harmonicity=row["harmonicity_R"],
            is_favorite=bool(row["is_favorite"]),
            is_hidden=bool(row["is_hidden"]),
            class_signature=_parse_float_blob(row["class_signature_VR"]),
            category_signature=_parse_float_blob(row["category_signature_VR"]),
            pitch_confidence=row["pitch_confidence_R"],
        )

    def _get_asset_sqlite(self, rel_path: str) -> SononymAsset | None:
        conn = self._connect_sqlite()
        try:
            row = conn.execute(
                """
                SELECT
                    a.id, a.filename, a.basename, a.modtime, a.status,
                    a.file_type_S, a.file_size_R, a.file_length_R,
                    a.file_sample_rate_R, a.file_channel_count_R, a.file_bit_depth_R,
                    a.classes_VS, a.class_strengths_VR,
                    a.categories_VS, a.category_strengths_VR,
                    a.pitch_class_S, a.base_note_R, a.base_note_confidence_R,
                    a.peak_db_R, a.rms_db_R, a.crest_factor_R,
                    a.bpm_R, a.bpm_confidence_R,
                    a.brightness_R, a.noisiness_R, a.harmonicity_R,
                    a.is_favorite, a.is_hidden,
                    ae.class_signature_VR, ae.category_signature_VR,
                    ae.pitch_confidence_R
                FROM assets a
                LEFT JOIN assets_extended ae ON ae.asset_id = a.id
                WHERE a.filename = ?
                """,
                (rel_path,),
            ).fetchone()
            return self._row_to_asset(row) if row else None
        finally:
            conn.close()
