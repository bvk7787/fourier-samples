"""Hints that lead somewhere: each next step a message suggests can do what it says."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from fourier import places
from fourier.db import session as S
from fourier.metadata import rows as R


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def lib(tmp_path, monkeypatch):
    root = tmp_path / "SampleLibrary"
    p = root / "Acme" / "Kick 01.wav"
    p.parent.mkdir(parents=True)
    sf.write(str(p), np.zeros(4410, dtype="float32"), 44100)
    monkeypatch.setenv("FOURIER_LIBRARY", str(root))
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "out" / "FourierCurated"))
    monkeypatch.delenv("FOURIER_CONFIG", raising=False)
    places.reset()
    S._engine = S._SessionLocal = None
    R.forget_current()
    yield tmp_path
    S._engine = S._SessionLocal = None
    places.reset()


def _run(tmp_path, *args, **kw):
    from fourier.cli import main
    S._engine = S._SessionLocal = None
    return CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), *args], **kw)


def test_without_clap_a_build_offers_no_scan_only_with_something_to_build_from(lib, monkeypatch):
    import fourier.cli.enrich as E
    monkeypatch.setattr(E, "clap_extra_missing", lambda: True)
    out = _flat(_run(lib, "build").output)
    assert "The CLAP model a build needs isn't installed: run `fourier setup`." in out, out
    assert "--no-scan" not in out, out              # it answers "nothing scanned yet"
    from fourier.db.models import Sample, SampleFeatures
    S._engine = S._SessionLocal = None
    S.init_db(lib / "t.db")
    with S.session_scope() as s:
        row = Sample(path=str(lib / "SampleLibrary" / "Acme" / "Kick 01.wav"), filename="Kick 01.wav")
        s.add(row)
        s.flush()
        s.add(SampleFeatures(sample_id=row.id, clap_embedding=np.ones(512, "float32").tobytes()))
    out = _flat(_run(lib, "build").output)
    assert "(or build from the database as it is with --no-scan)" in out, out


def test_a_sononym_filter_on_a_library_without_sononym_says_why_nothing_matched(lib):
    from fourier.ingest import importer as IMP
    S._engine = S._SessionLocal = None
    S.init_db(lib / "t.db")
    IMP.scan_filesystem(lib / "SampleLibrary")
    out = _flat(_run(lib, "search", "--class", "Loop", "--harmonicity-min", "0.5").output)
    assert ("No samples found: --class, --harmonicity-min read Sononym's analysis, and this "
            "library has none.") in out, out
    out = _flat(_run(lib, "search", "--min-dur", "100").output)
    assert "Try relaxing your filters" in out and "Sononym" not in out, out


def test_setup_without_clap_says_nothing_of_what_a_build_does_with_it(lib, monkeypatch):
    import fourier.cli.enrich as E
    monkeypatch.setattr(E, "clap_extra_missing", lambda: True)
    target = lib / "fourier.toml"
    r = _run(lib, "setup", "--to", str(target), "--yes", "--no-clap", "--no-llm", "--no-build",
             "--library", str(lib / "SampleLibrary"), "--device", "digitakt_2",
             "--preset", "balanced")
    out = _flat(r.output)
    assert "Skipped. A build needs it" in out, out
    assert "trains a sound model" not in out, out
