"""Sononym indexes everything under its library folder, Fourier's own output too when the
publish root (or the master, or the renders) sits inside it: the scan's Sononym import and
sync skip those folders as the walk does, so Fourier never takes its own exports back in."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from fourier import places
from fourier.db import session as S
from fourier.db.models import Sample
from fourier.ingest import importer as I


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setattr(places, "master_dir", lambda: str(tmp_path / "Master"))
    monkeypatch.setattr(places, "renders_dir", lambda device_id=None: str(tmp_path / "Renders"))
    monkeypatch.setattr(places, "publish_root", lambda: str(lib / "Fourier"))
    return lib


def test_own_folders_inside_the_library_are_named(outputs, tmp_path, monkeypatch):
    assert I.own_rel_prefixes(outputs) == ("Fourier/",)
    monkeypatch.setattr(places, "publish_root", lambda: str(tmp_path / "Elsewhere"))
    assert I.own_rel_prefixes(outputs) == ()
    assert I._is_own("Fourier/releases/v1/KICKS/kick/Kick_01.wav", ("Fourier/",))
    assert not I._is_own("Fourier Pack/Kick 01.wav", ("Fourier/",))      # a pack that merely starts so


def _sononym_db(path, rel_paths):
    duckdb = pytest.importorskip("duckdb")
    c = duckdb.connect(str(path))
    c.execute("""CREATE TABLE assets (id INTEGER, filename VARCHAR, modtime INTEGER, status VARCHAR,
        basename VARCHAR, file_type_S VARCHAR, file_size_R INTEGER, file_length_R FLOAT,
        file_sample_rate_R INTEGER, file_channel_count_R INTEGER, file_bit_depth_R INTEGER,
        classes_VS VARCHAR[], class_strengths_VR FLOAT[], categories_VS VARCHAR[],
        category_strengths_VR FLOAT[], pitch_class_S VARCHAR, base_note_R FLOAT,
        base_note_confidence_R FLOAT, peak_db_R FLOAT, rms_db_R FLOAT, crest_factor_R FLOAT,
        bpm_R FLOAT, bpm_confidence_R FLOAT, brightness_R FLOAT, noisiness_R FLOAT,
        harmonicity_R FLOAT, is_favorite INTEGER, is_hidden INTEGER)""")
    c.execute("""CREATE TABLE assets_extended (asset_id INTEGER, class_signature_VR FLOAT[],
        category_signature_VR FLOAT[], pitch_confidence_R FLOAT)""")
    for i, rel in enumerate(rel_paths, 1):
        c.execute("INSERT INTO assets VALUES (?, ?, 1, 'succeeded', ?, 'wav', 100, 0.5, 44100, 1, 16, "
                  "['OneShot'], [0.1, 0.9], ['Perc Kicks'], [0.9], NULL, NULL, NULL, -1.0, -12.0, 4.0, "
                  "NULL, NULL, 0.3, 0.2, 0.4, 0, 0)", [i, rel, rel.rsplit("/", 1)[-1]])
    c.close()


@pytest.mark.parametrize("how", ["sync", "import"])
def test_the_sononym_scan_skips_fouriers_own_folders(outputs, tmp_path, monkeypatch, how):
    rels = ["Acme/Kicks/Kick 01.wav", "Fourier/releases/v1/KICKS/kick/Kick_01.wav"]
    for rel in rels:
        (outputs / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(outputs / rel, np.zeros(22050, dtype="float32"), 44100)
    _sononym_db(outputs / "sononym.db", rels)
    old = (S._engine, S._SessionLocal)
    S._engine = S._SessionLocal = None
    try:
        S.init_db(tmp_path / "fourier.db")
        if how == "sync":
            I.sync_from_sononym(sononym_db=outputs / "sononym.db", library_root=outputs)
        else:
            I.import_from_sononym(sononym_db=outputs / "sononym.db", library_root=outputs)
        with S.session_scope() as s:
            got = sorted(p for (p,) in s.query(Sample.path))
        assert got == [str(outputs / rels[0])]
    finally:
        if S._engine is not None:
            S._engine.dispose()
        S._engine, S._SessionLocal = old


def test_doctor_notes_an_output_folder_inside_the_library(outputs, tmp_path, monkeypatch):
    from fourier.cli import setup as D
    monkeypatch.setattr(places, "library_roots", lambda: [outputs])
    rows = D.check_output()
    notes = [r for r in rows if r[0] == D.INFO]
    assert [r[1] for r in notes] == ["output: publish root"] and "the scan skips it" in notes[0][2]
    assert not any(r[0] == D.FAIL for r in rows)
