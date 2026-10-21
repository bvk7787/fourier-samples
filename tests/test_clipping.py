"""Clipping is read on the file as it is (each channel, at its own rate): a loop limited just
under full scale isn't clipped however its resampled copy rings, a flat-topped one is, and a
database measured before is measured again. Invented names only."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from fourier.analysis import librosa_features as LF
from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures

SR = 44100


def _square(path, peak, seconds=1.0, freq=110.0):
    """A square wave at `peak`: flat tops hundreds of samples long, which a 22,050 Hz copy
    rings over by about a tenth."""
    t = np.arange(int(seconds * SR)) / SR
    y = peak * np.sign(np.sin(2 * np.pi * freq * t))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([y, y], axis=1).astype("float32"), SR, subtype="PCM_16")
    return path


def test_full_scale_share_counts_flat_tops_only():
    y = np.zeros((10_000, 2), dtype="float32")
    assert LF.full_scale_share(y) == 0.0
    y[100:102, 0] = 1.0                       # a peak that touches full scale: two samples
    y[500:502, 1] = -1.0
    assert LF.full_scale_share(y) == 0.0
    y[1000:1030, 1] = -1.0                    # a flat top in one channel: 30 of 20,000 samples
    assert LF.full_scale_share(y) == pytest.approx(30 / 20_000)
    assert LF.full_scale_share(y[:, 1]) == pytest.approx(30 / 10_000)  # mono


@pytest.fixture
def db(tmp_path):
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "t.db")
    yield tmp_path / "t.db"
    if S._engine is not None:
        S._engine.dispose()
    S._engine = S._SessionLocal = None


def _analyze(db, *only):
    from fourier.cli import main
    args = ["--db", str(db), "tools", "analyze"] + [x for o in only for x in ("--only", o)]
    r = CliRunner().invoke(main, args)
    assert r.exit_code == 0, r.output
    return r


def _clipped(db):
    con = sqlite3.connect(db)
    try:
        return dict(con.execute("SELECT s.filename, f.is_clipped FROM samples s JOIN sample_features f "
                                "ON f.sample_id = s.id ORDER BY s.id").fetchall())
    finally:
        con.close()


def test_a_loop_limited_under_full_scale_isnt_clipped(db, tmp_path):
    lib = tmp_path / "SampleLibrary" / "Acme" / "Loops"
    with S.session_scope() as s:
        for name, peak in (("Qzv Loop 01.wav", 0.985), ("Qzv Loop 02.wav", 1.0)):
            p = _square(lib / name, peak)
            s.add(Sample(path=str(p), filename=name, duration_s=1.0))
    _analyze(db, "librosa", "quality")
    assert _clipped(db) == {"Qzv Loop 01.wav": 0, "Qzv Loop 02.wav": 1}


def test_a_database_measured_before_is_measured_again(db, tmp_path):
    """Rows the older reading flagged (or cleared) have no clip_share: the quality step
    measures them again."""
    p = _square(tmp_path / "SampleLibrary" / "Acme" / "Loops" / "Qzv Loop 03.wav", 0.985)
    with S.session_scope() as s:
        smp = Sample(path=str(p), filename=p.name, duration_s=1.0)
        s.add(smp)
        s.flush()
        s.add(SampleFeatures(sample_id=smp.id, is_clipped=1, dc_offset_ratio=0.0))
    r = _analyze(db, "quality")
    assert _clipped(db) == {"Qzv Loop 03.wav": 0}
    assert "Measuring clipping again for 1 sample an earlier version" in r.output
    assert "Measuring clipping again" not in _analyze(db, "quality").output


def test_the_measure_again_line_leaves_out_files_it_cant_read(db, tmp_path):
    """A row measured before whose file is gone (or only in the cloud) is never measured, so
    the "once, after an update" line doesn't count it: it would repeat on every build."""
    p = _square(tmp_path / "SampleLibrary" / "Acme" / "Loops" / "Qzv Loop 04.wav", 0.985)
    with S.session_scope() as s:
        smp = Sample(path=str(p), filename=p.name, duration_s=1.0)
        s.add(smp)
        s.flush()
        s.add(SampleFeatures(sample_id=smp.id, is_clipped=0, dc_offset_ratio=0.0))
    p.unlink()
    for _ in range(2):
        assert "Measuring clipping again" not in _analyze(db, "quality").output
