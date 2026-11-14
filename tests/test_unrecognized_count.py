"""Without Sononym: one count of the samples no rule recognized (the build's note, its quiet
summary and `fourier why --unrecognized` say the same), and a sample whose name says what it
is counts as recognized, even when no category takes it."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from sqlalchemy import text

from fourier import places
from fourier.db import session as S
from fourier.ingest import importer as IMP
from fourier.metadata import rows as R


def _flat(text: str) -> str:
    return " ".join(text.split())


@pytest.fixture
def lib(tmp_path, monkeypatch):
    root = tmp_path / "SampleLibrary"
    monkeypatch.setenv("FOURIER_LIBRARY", str(root))
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "out" / "FourierCurated"))
    monkeypatch.delenv("FOURIER_CONFIG", raising=False)
    for rel, secs in (("Bass/Lines/Boos A-01.wav", 4.0), ("Zzq/Qxvwz 01.wav", 0.5),
                      ("Zzq/Qxvwz 02.wav", 0.5), ("Vvk/Zorbl 01.wav", 0.5)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(secs * 44100)) / 44100
        sf.write(str(p), (0.5 * np.sin(2 * np.pi * 110 * t)).astype("float32"), 44100)
    places.reset()
    S._engine = S._SessionLocal = None
    R.forget_current()
    S.init_db(tmp_path / "t.db")
    IMP.scan_filesystem(root)
    with S.session_scope() as s:
        R.ensure_current(s)
    S._engine = S._SessionLocal = None
    yield tmp_path
    S._engine = S._SessionLocal = None
    R.forget_current()
    places.reset()


def _run(tmp_path, *args):
    S._engine = S._SessionLocal = None
    from fourier.cli import main
    return _flat(CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), *args]).output)


def test_a_sample_named_for_its_sound_is_recognized(lib):
    out = _run(lib, "why", "Boos A-01")
    assert "Not in the master: its name or folders say bass, but no category takes it as it " \
           "is." in out, out
    assert "nothing in its name or folders said what it is" not in out, out
    out = _run(lib, "why", "--detail", "Boos A-01")
    assert "labels: path: bass" in out and "recognized as bass, but no category takes it" in out
    assert "no category word in its name" not in out, out
    out = _run(lib, "why", "Qxvwz 01")
    assert "nothing in its name or folders said what it is" in out, out
    out = _run(lib, "why", "--unrecognized")
    assert out.startswith("3 samples no rule recognized (top folders: Zzq (2), Vvk (1))."), out
    assert "Boos A-01" not in out, out


def test_the_build_and_why_count_the_unrecognized_alike(lib, monkeypatch):
    import fourier.cli.build as B
    from fourier.cli import quiet as Q
    S._engine = S._SessionLocal = None
    S.init_db(lib / "t.db")
    with S.session_scope() as s:
        plorp = s.execute(text("SELECT id FROM samples WHERE filename = 'Zorbl 01.wav'")).scalar()
    monkeypatch.setattr(B, "_clap_placed", lambda master_dir: {plorp: "FX"})   # placed by sound
    note = B._unrecognized_note(None)
    said = _run(lib, "why", "--unrecognized")
    first = ("3 samples no rule recognized: the sound model or the CLAP fallback placed 1 by "
             "sound, and 2 are in no category (top folders: Zzq (2)).")
    assert note[0] == first and said.startswith(first), (note, said)
    log = lib / "build.log"
    log.write_text(note[0] + "\nverify: PASS -- 1 pass, 0 warn, 0 fail\n")
    q = Q.Quiet(log)
    q.out_dir = str(lib / "out")
    summary = q.summary()
    assert ("3 samples no rule recognized: the sound model or the CLAP fallback placed 1 by sound, "
            "and 2 are in no category. `fourier why --unrecognized` lists them.") in summary, summary
