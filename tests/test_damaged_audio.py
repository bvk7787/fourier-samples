"""A floating-point file holding NaN or infinite values (a damaged file): unreadable for
analysis, left out like a corrupt file, and said once with its path."""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from fourier import audioio, places
from fourier.db import session as S
from fourier.ingest import importer as IMP
from fourier.metadata import rows as R

SR = 44100
WHY = "holds NaN or infinite sample values: a damaged floating-point file"


def _wav(p, bad=None, subtype="FLOAT"):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(0.3 * SR)) / SR
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    if bad is not None:
        y[1000:1005] = bad
    sf.write(str(p), y, SR, subtype=subtype)
    return p


@pytest.fixture
def lib(tmp_path, monkeypatch):
    root = tmp_path / "SampleLibrary"
    root.mkdir()
    monkeypatch.setenv("FOURIER_LIBRARY", str(root))
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "out" / "FourierCurated"))
    monkeypatch.delenv("FOURIER_CONFIG", raising=False)
    places.reset()
    S._engine = S._SessionLocal = None
    R.forget_current()
    S.init_db(tmp_path / "t.db")
    yield root
    S._engine = S._SessionLocal = None
    places.reset()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_reading_audio_that_isnt_finite_says_so(tmp_path, bad):
    p = _wav(tmp_path / "Snare 01.wav", bad)
    for kw in ({"sr": None}, {"sr": 48000, "mono": True}):     # as read, and resampled
        with pytest.raises(audioio.NotFinite, match="NaN or infinite"):
            audioio.load(p, **kw)
    assert audioio.not_finite(p)
    assert not audioio.not_finite(_wav(tmp_path / "Snare 02.wav"))
    assert not audioio.not_finite(_wav(tmp_path / "Snare 03.wav", subtype="PCM_16"))


def test_the_scan_leaves_a_damaged_file_out_and_names_why(lib):
    _wav(lib / "Acme" / "Snare 01.wav")
    _wav(lib / "Acme" / "Snare 02.wav", np.nan)
    c = IMP.scan_filesystem(lib)
    assert c["created"] == 1 and c["errors"] == 1, c
    assert c["error_files"] == [f"{lib / 'Acme' / 'Snare 02.wav'} ({WHY})"]


def _stand_in(path):
    """A CLAP stand-in that reads the audio as CLAP does: zeros when it can't."""
    try:
        y, _ = audioio.load(path, sr=48000, mono=True, duration=10)
    except Exception:
        return np.zeros(512, dtype=np.float32)
    v = np.ones(512, dtype=np.float32) + float(np.abs(y).mean())
    return v / np.linalg.norm(v)


def test_a_damaged_file_scanned_before_is_left_out_once_and_holds_no_note(lib, tmp_path, monkeypatch):
    """A database from a version that didn't check: the CLAP step can't embed the file, so it
    used to be retried and leave a note on every command ("1 sample(s) aren't in the CLAP index
    yet") that a build never cleared."""
    from fourier.analysis import clap_features as CF
    from fourier.cli import main
    good = _wav(lib / "Acme" / "Snare 01.wav")
    bad = _wav(lib / "Acme" / "Snare 02.wav", np.nan)
    with monkeypatch.context() as mp:
        mp.setattr(IMP, "not_finite", lambda path: False, raising=False)   # an older version's scan
        assert IMP.scan_filesystem(lib)["created"] == 2
    monkeypatch.setattr(CF, "embed_audio_file", _stand_in)
    db = str(tmp_path / "t.db")
    S._engine = S._SessionLocal = None
    run = lambda *a: CliRunner().invoke(main, ["--db", db, *a])
    res = run("tools", "analyze", "--only", "clap")
    out = " ".join(res.output.split())
    assert res.exit_code == 0, out
    assert f"1 file(s) left out: each {WHY}" in out and str(bad) in out, out
    assert "(0 skipped/errored)" in out, out
    assert IMP.scan_errors([lib]) == {str(bad): WHY}
    res = run("tools", "analyze", "--status")
    assert "aren't in the CLAP index yet" not in res.output, res.output
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "t.db")
    read = []
    monkeypatch.setattr(IMP, "not_finite", lambda path: read.append(path) or False, raising=False)
    c = IMP.scan_filesystem(lib)                               # left out, and not read again
    assert c["errors"] == 1 and str(bad) in c["error_files"][0] and str(bad) not in read
    con = sqlite3.connect(tmp_path / "t.db")
    try:
        assert con.execute("SELECT s.path FROM missing_files m JOIN samples s ON s.id = "
                           "m.sample_id").fetchall() == [(str(bad),)]
    finally:
        con.close()
    assert str(good) not in read                               # known and unchanged: not read


def test_a_known_file_that_failed_one_read_is_read_again(lib, monkeypatch):
    """Only a damaged file is skipped while unchanged: a file the last walk couldn't read for
    another reason (mid-copy, a moment's I/O error) is read again and comes back."""
    _wav(lib / "Acme" / "Kick 01.wav", subtype="PCM_16")
    assert IMP.scan_filesystem(lib)["created"] == 1
    with S.session_scope() as s:                       # no header yet: the walk reads it
        s.execute(sqlite3_text("UPDATE samples SET duration_s = NULL"))
    real, n = IMP.file_info, [0]

    def flaky(p):
        n[0] += 1
        return None if n[0] == 1 else real(p)
    monkeypatch.setattr(IMP, "file_info", flaky)
    assert IMP.scan_filesystem(lib)["errors"] == 1     # failed once
    c = IMP.scan_filesystem(lib)
    assert c["errors"] == 0 and n[0] == 2, c           # read again, and back
    with S.session_scope() as s:
        assert s.execute(sqlite3_text("SELECT COUNT(*) FROM missing_files")).scalar() == 0


def test_a_damaged_file_stays_out_under_a_linked_library_folder(tmp_path, monkeypatch):
    """With the library folder reached through a symlink, a damaged file another step left
    out (recorded at its real path) stays out at the next walk."""
    real = tmp_path / "Drive" / "Samples"
    _wav(real / "Acme" / "Snare 01.wav")
    bad = _wav(real / "Acme" / "Snare 02.wav", np.nan)
    link = tmp_path / "SampleLibrary"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("FOURIER_LIBRARY", str(link))
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "home"))
    places.reset()
    S._engine = S._SessionLocal = None
    R.forget_current()
    S.init_db(tmp_path / "t.db")
    try:
        with monkeypatch.context() as mp:
            mp.setattr(IMP, "not_finite", lambda path: False, raising=False)   # an older scan
            assert IMP.scan_filesystem(link)["created"] == 2
        with S.session_scope() as s:
            sid = s.execute(sqlite3_text("SELECT id FROM samples WHERE path LIKE '%Snare 02.wav'")).scalar()
            assert IMP.leave_out_unreadable(s, [(sid, str(bad))], WHY) == 1
        c = IMP.scan_filesystem(link)
        assert c["errors"] == 1 and "Snare 02.wav" in c["error_files"][0], c
        with S.session_scope() as s:
            assert s.execute(sqlite3_text("SELECT COUNT(*) FROM missing_files")).scalar() == 1
    finally:
        S._engine = S._SessionLocal = None
        places.reset()


def test_librosa_leaves_out_a_file_damaged_past_claps_window(lib, tmp_path, monkeypatch):
    """A damaged file an older version scanned, whose NaNs sit past the 10 s CLAP reads: the
    librosa step leaves it out once instead of failing on it at every build."""
    from fourier.cli import main
    p = lib / "Acme" / "Pad 01.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    y = (0.2 * np.sin(2 * np.pi * 110 * np.arange(12 * SR) / SR)).astype("float32")
    y[11 * SR] = np.nan
    sf.write(str(p), y, SR, subtype="FLOAT")
    with monkeypatch.context() as mp:
        mp.setattr(IMP, "not_finite", lambda path: False, raising=False)
        assert IMP.scan_filesystem(lib)["created"] == 1
    db = str(tmp_path / "t.db")
    S._engine = S._SessionLocal = None
    res = CliRunner().invoke(main, ["--db", db, "tools", "analyze", "--only", "librosa"])
    out = " ".join(res.output.split())
    assert res.exit_code == 0, out
    assert f"1 file(s) left out: each {WHY}" in out, out
    assert IMP.scan_errors([lib]) == {str(p): WHY}


def sqlite3_text(sql):
    from sqlalchemy import text
    return text(sql)
