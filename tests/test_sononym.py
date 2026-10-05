"""Tests for the Sononym reader — DuckDB format detection and streaming."""

import itertools
import os
import struct
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# _detect_db_format
# ---------------------------------------------------------------------------

def test_detect_duckdb_magic_at_offset_8(tmp_path):
    """DuckDB magic DUCK sits at bytes 8-11, not 0-7."""
    from fourier.ingest.sononym import _detect_db_format
    db = tmp_path / "test.db"
    # Simulate real DuckDB header: 8 junk bytes then DUCK
    db.write_bytes(b"\xeb\x82\x12\xcd\x86\xed\xa2+" + b"DUCK" + b"\x00" * 52)
    assert _detect_db_format(db) == "duckdb"


def test_detect_sqlite_magic(tmp_path):
    from fourier.ingest.sononym import _detect_db_format
    db = tmp_path / "test.db"
    db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 48)
    assert _detect_db_format(db) == "sqlite"


def test_detect_unknown_defaults_to_sqlite(tmp_path):
    from fourier.ingest.sononym import _detect_db_format
    db = tmp_path / "test.db"
    db.write_bytes(b"\x00" * 64)
    assert _detect_db_format(db) == "sqlite"


def test_detect_raises_on_missing_file():
    from fourier.ingest.sononym import _detect_db_format
    with pytest.raises(FileNotFoundError):
        _detect_db_format(Path("/nonexistent/path/to/sononym.db"))


# ---------------------------------------------------------------------------
# _parse_json_list
# ---------------------------------------------------------------------------

def test_parse_json_list_string():
    from fourier.ingest.sononym import _parse_json_list
    assert _parse_json_list('["OneShot", "Loop"]') == ["OneShot", "Loop"]


def test_parse_json_list_native_list():
    """DuckDB returns native lists — pass through without JSON decode."""
    from fourier.ingest.sononym import _parse_json_list
    assert _parse_json_list(["OneShot"]) == ["OneShot"]


def test_parse_json_list_none():
    from fourier.ingest.sononym import _parse_json_list
    assert _parse_json_list(None) == []
    assert _parse_json_list(None, fallback=["x"]) == ["x"]


def test_parse_json_list_invalid():
    from fourier.ingest.sononym import _parse_json_list
    assert _parse_json_list("not json") == []


def test_parse_json_list_non_list_json():
    from fourier.ingest.sononym import _parse_json_list
    assert _parse_json_list('{"key": "val"}') == []


# ---------------------------------------------------------------------------
# _parse_float_blob
# ---------------------------------------------------------------------------

def test_parse_float_blob_native_list():
    """DuckDB returns Python list — pass straight through."""
    from fourier.ingest.sononym import _parse_float_blob
    result = _parse_float_blob([0.1, 0.9])
    assert result == [0.1, 0.9]


def test_parse_float_blob_json_text():
    from fourier.ingest.sononym import _parse_float_blob
    result = _parse_float_blob(b"[0.5, 0.5]")
    assert result is not None
    assert abs(result[0] - 0.5) < 1e-6


def test_parse_float_blob_raw_binary():
    from fourier.ingest.sononym import _parse_float_blob
    raw = struct.pack("<2f", 0.25, 0.75)
    result = _parse_float_blob(raw)
    assert result is not None
    assert abs(result[0] - 0.25) < 1e-5
    assert abs(result[1] - 0.75) < 1e-5


def test_parse_float_blob_none():
    from fourier.ingest.sononym import _parse_float_blob
    assert _parse_float_blob(None) is None


def test_parse_float_blob_string():
    from fourier.ingest.sononym import _parse_float_blob
    result = _parse_float_blob("[1.0, 2.0]")
    assert result == [1.0, 2.0]


# ---------------------------------------------------------------------------
# Integration: a real Sononym library, at $FOURIER_TEST_SONONYM_DB (skipped without it)
# ---------------------------------------------------------------------------

SONONYM_DB = Path(os.environ.get("FOURIER_TEST_SONONYM_DB") or "/nonexistent/sononym.db")


@pytest.fixture(scope="module")
def sononym_reader():
    if not SONONYM_DB.exists():
        pytest.skip("no Sononym library (set FOURIER_TEST_SONONYM_DB)")
    import subprocess
    try:
        if subprocess.run(["pgrep", "-x", "Sononym"], capture_output=True).returncode == 0:
            pytest.skip("Sononym.app is running; its DuckDB may be locked")
    except Exception:
        pass
    from fourier.ingest.sononym import SononymReader
    return SononymReader(SONONYM_DB)


def test_real_db_detected_as_duckdb(sononym_reader):
    assert sononym_reader.db_format == "duckdb"


def test_real_asset_count(sononym_reader):
    count = sononym_reader.count()
    assert count > 0, f"Expected assets in the database, got {count}"


def test_real_get_categories(sononym_reader):
    cats = sononym_reader.get_categories()
    assert isinstance(cats, list)
    assert len(cats) > 0
    assert any("Kick" in c for c in cats), f"No kick category in {cats}"


def test_real_stream_kicks(sononym_reader):
    kicks = list(itertools.islice(sononym_reader.stream_assets(
        category_filter=["Perc Kicks"],
        class_filter=["OneShot"],
        only_existing=False,
        batch_size=10,
    ), 10))
    assert len(kicks) > 0
    for a in kicks:
        assert a.sononym_id > 0
        assert isinstance(a.classes, list)
        assert isinstance(a.categories, list)
        assert "Perc Kicks" in a.categories or any("Kick" in c for c in a.categories)


def test_real_stream_loops(sononym_reader):
    loops = list(itertools.islice(sononym_reader.stream_assets(
        class_filter=["Loop"],
        only_existing=False,
        batch_size=5,
    ), 5))
    assert len(loops) > 0
    for a in loops:
        assert "Loop" in a.classes


def test_real_extended_signatures(sononym_reader):
    """class_signature and category_signature should parse as float lists."""
    assets = list(itertools.islice(sononym_reader.stream_assets(
        only_existing=False,
        include_extended=True,
        batch_size=20,
    ), 20))
    with_sig = [a for a in assets if a.class_signature is not None]
    assert len(with_sig) > 0
    assert isinstance(with_sig[0].class_signature, list)
    assert all(isinstance(v, float) for v in with_sig[0].class_signature)


def test_real_get_asset_by_rel_path(sononym_reader):
    first = next(sononym_reader.stream_assets(only_existing=False, batch_size=1))
    looked_up = sononym_reader.get_asset_by_rel_path(first.rel_path)
    assert looked_up is not None
    assert looked_up.sononym_id == first.sononym_id
    assert looked_up.rel_path == first.rel_path


def test_real_get_all_modtimes(sononym_reader):
    modtimes = sononym_reader.get_all_modtimes()
    assert isinstance(modtimes, dict)
    assert 0 < len(modtimes) <= sononym_reader.count()   # succeeded assets only
    # Keys are relative paths (strings), values are ints
    sample_key = next(iter(modtimes))
    assert isinstance(sample_key, str)
    assert isinstance(modtimes[sample_key], int)


def test_real_duration_filter(sononym_reader):
    short = list(itertools.islice(sononym_reader.stream_assets(
        max_duration_s=0.5,
        only_existing=False,
        batch_size=10,
    ), 10))
    for a in short:
        assert a.duration_s is None or a.duration_s <= 0.5


def test_real_favorites_filter(sononym_reader):
    """favorites_only filter should not crash (may return 0 results if none starred)."""
    favs = list(sononym_reader.stream_assets(
        favorites_only=True,
        only_existing=False,
        batch_size=10,
    ))
    assert isinstance(favs, list)
