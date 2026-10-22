"""Clipping is read on the file as it is (each channel, at its own rate) as flat tops against
the file's own peak: a loop limited just under full scale isn't clipped however its resampled
copy rings, a flat-topped one is wherever it peaks, a database measured by an older rule is
measured again, and a file that can't be read is read once. Invented names only."""
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


def _wave(path, peak, seconds=1.0, freq=110.0, clip=None, subtype="PCM_16"):
    """A sine at `peak`; with `clip`, hard-clipped at that share of its amplitude first and
    then scaled to `peak` (flat tops just under full scale when peak < 1)."""
    t = np.arange(int(seconds * SR)) / SR
    y = np.sin(2 * np.pi * freq * t)
    if clip is not None:
        y = np.clip(y, -clip, clip) / clip
    y = peak * y
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.stack([y, y], axis=1).astype("float32"), SR, subtype=subtype)
    return path


def test_flat_top_share_counts_flat_tops_only():
    t = np.arange(10_000) / SR
    y = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype("float32")
    assert LF.flat_top_share(y) == 0.0
    y[100:102, 0] = 1.0                       # a peak that touches full scale: two samples
    y[500:502, 1] = -1.0
    assert LF.flat_top_share(y) == 0.0
    y[998:1032, 1] = [-0.7, -0.9] + [-1.0] * 30 + [-0.9, -0.7]   # a flat top in one channel: 30 of 20,000
    assert LF.flat_top_share(y) == pytest.approx(30 / 20_000)
    assert LF.flat_top_share(y[:, 1]) == pytest.approx(30 / 10_000)  # mono
    assert LF.flat_top_share(np.zeros((100, 2), dtype="float32")) == 0.0


def test_a_flat_top_is_measured_against_the_files_own_peak():
    """Clipped and then normalized under full scale is clipped; a float file running over 1.0
    without flat tops isn't; a square wave's flat tops are its shape; a slow tone at full scale
    rounds its peaks to a few equal samples in a 16-bit file and isn't clipped."""
    t = np.arange(SR) / SR
    s = np.sin(2 * np.pi * 110 * t)
    q16 = lambda y: (np.round(y * 32767) / 32767).astype("float32")      # noqa: E731
    normalized = q16(0.9 * np.clip(s, -0.6, 0.6) / 0.6)
    assert LF.flat_top_share(normalized) > LF.CLIP_SHARE
    assert LF.flat_top_share((1.4 * s).astype("float32")) == 0.0
    assert LF.flat_top_share(q16(0.9 * np.sign(s))) == 0.0
    assert LF.flat_top_share(q16(np.sin(2 * np.pi * 40 * t))) <= LF.CLIP_SHARE
    assert LF.flat_top_share(q16(np.clip(1.3 * s, -1, 1))) > LF.CLIP_SHARE      # clipped at full scale


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
        for name, peak, clip in (("Qzv Loop 01.wav", 0.985, None), ("Qzv Loop 02.wav", 1.0, 0.7),
                                 ("Qzv Loop 05.wav", 0.89, 0.7)):
            p = _wave(lib / name, peak, clip=clip)
            s.add(Sample(path=str(p), filename=name, duration_s=1.0))
    _analyze(db, "librosa", "quality")
    assert _clipped(db) == {"Qzv Loop 01.wav": 0, "Qzv Loop 02.wav": 1, "Qzv Loop 05.wav": 1}


def test_a_database_measured_before_is_measured_again(db, tmp_path):
    """Rows an older rule measured (no clip_rule, or an older one) are measured again, once,
    and the build's analysis count says so."""
    lib = tmp_path / "SampleLibrary" / "Acme" / "Loops"
    p = _wave(lib / "Qzv Loop 03.wav", 0.985)
    p2 = _wave(lib / "Qzv Loop 06.wav", 0.89, clip=0.7)
    with S.session_scope() as s:
        for path, old in ((p, 1), (p2, 0)):
            smp = Sample(path=str(path), filename=path.name, duration_s=1.0)
            s.add(smp)
            s.flush()
            s.add(SampleFeatures(sample_id=smp.id, is_clipped=old, clip_share=0.0, dc_offset_ratio=0.0))
    from fourier.cli.setup import _counted
    assert _counted(False)["quality"] == 2
    r = _analyze(db, "quality")
    assert _clipped(db) == {"Qzv Loop 03.wav": 0, "Qzv Loop 06.wav": 1}
    assert "Measuring clipping again for 2 samples an earlier version" in r.output
    assert "Measuring clipping again" not in _analyze(db, "quality").output
    assert _counted(False)["quality"] == 0


def test_the_measure_again_line_leaves_out_files_it_cant_read(db, tmp_path):
    """A row measured before whose file is gone (or only in the cloud) is never measured, so
    the "once, after an update" line doesn't count it: it would repeat on every build; nor
    does doctor's count, once a walk marked the file missing."""
    p = _wave(tmp_path / "SampleLibrary" / "Acme" / "Loops" / "Qzv Loop 04.wav", 0.985)
    with S.session_scope() as s:
        smp = Sample(path=str(p), filename=p.name, duration_s=1.0)
        s.add(smp)
        s.flush()
        s.add(SampleFeatures(sample_id=smp.id, is_clipped=0, dc_offset_ratio=0.0))
    p.unlink()
    for _ in range(2):
        assert "Measuring clipping again" not in _analyze(db, "quality").output
    # once a walk marked it missing, doctor's count leaves it out too
    from fourier.cli.setup import _counted
    from fourier.db.models import MissingFile
    with S.session_scope() as s:
        s.add(MissingFile(sample_id=smp.id, path=str(p)))
    assert _counted(False)["quality"] == 0


def test_a_file_that_cant_be_read_is_read_once(db, tmp_path, monkeypatch):
    """A file that is there and can't be read is marked measured with no reading: the next
    quality step doesn't read it again, and the build counts nothing left to measure."""
    bad = tmp_path / "SampleLibrary" / "Acme" / "Loops" / "Qzv Loop 07.wav"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    with S.session_scope() as s:
        s.add(Sample(path=str(bad), filename=bad.name, duration_s=1.0))
    r = _analyze(db, "quality")
    assert "1 unreadable" in r.output
    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT clip_rule, clip_share FROM sample_features").fetchall() == [(LF.CLIP_RULE, None)]
    finally:
        con.close()
    from fourier.cli.setup import _counted
    assert _counted(False)["quality"] == 0
    assert "All samples already have quality data" in _analyze(db, "quality").output
