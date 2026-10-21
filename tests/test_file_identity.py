"""A file's identity is every byte of it (ingest/importer.py): files that start alike are two
files, a rating covers only its own file's bytes, a file whose bytes change at its path is
read and analyzed again (one only touched isn't), and a database from before keeps its
analysis and is told apart where its first 8 KB collide. Invented names only."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine, text

from fourier import places
from fourier.db import migrations as M
from fourier.db import session as S
from fourier.db.models import Descriptor, Label, Sample, SampleFeatures
from fourier.ingest import importer as IMP
from fourier.metadata import rows as R

SR = 44100


def _hit(p: Path, freq: float, seconds=1.0, lead=0.25, tail=None):
    """lead seconds of digital silence, then a tone: two of these share their first 8 KB.
    tail: (seconds from the end, another frequency there)."""
    p.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    if tail:
        k = int(tail[0] * SR)
        y[-k:] = 0.5 * np.sin(2 * np.pi * tail[1] * t[-k:])
    y[:int(lead * SR)] = 0.0
    sf.write(str(p), y.astype("float32"), SR, subtype="PCM_16")
    return p


@pytest.fixture(params=["t.db", "t.duckdb"])
def lib(tmp_path, monkeypatch, request):
    """A library folder and a fresh database of each kind (DuckDB updates an indexed column
    of a row other tables reference only as a delete and an insert)."""
    root = tmp_path / "SampleLibrary"
    root.mkdir()
    monkeypatch.setenv("FOURIER_LIBRARY", str(root))
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "out" / "FourierCurated"))
    places.reset()
    S._engine = S._SessionLocal = None
    R.forget_current()
    S.init_db(tmp_path / request.param)
    yield root
    if S._engine is not None:
        S._engine.dispose()
    S._engine = S._SessionLocal = None
    places.reset()


def _facts(tmp_path):
    with S.session_scope() as s:
        res = s.execute(text("SELECT * FROM samples"))
        keys = list(res.keys())
        rows = [dict(zip(keys, r)) for r in res]
    return {Path(r["path"]).name: dict(id=r["id"], hash=r["file_hash"], size=r["file_size_bytes"], dur=r["duration_s"],
                                       quick=r.get("quick_hash"), mtime=r.get("file_mtime_ns")) for r in rows}


def _start_alike(lib):
    a = _hit(lib / "Acme" / "Claps" / "Qzv Clap A.wav", 1200.0)
    _hit(lib / "Acme" / "Claps" / "Qzv Clap B.wav", 1700.0)                      # same length, same lead
    (lib / "Acme" / "Copies").mkdir()
    shutil.copyfile(a, lib / "Acme" / "Copies" / "Qzv Clap A.wav")
    b = a.read_bytes()
    (lib / "Acme" / "Cut").mkdir()
    (lib / "Acme" / "Cut" / "Qzv Clap A half.wav").write_bytes(b[:len(b) // 2])  # a cut-off download
    _hit(lib / "Acme" / "Loops" / "Qzv Loop.wav", 300.0, seconds=2.0, lead=0.0)
    _hit(lib / "Acme" / "Loops" / "Qzv Loop Fill.wav", 300.0, seconds=2.0, lead=0.0, tail=(0.5, 450.0))


def test_files_that_start_alike_are_two_files(lib, tmp_path):
    _start_alike(lib)
    IMP.scan_filesystem(lib)
    f = _facts(tmp_path)
    a, b, half = f["Qzv Clap A.wav"], f["Qzv Clap B.wav"], f["Qzv Clap A half.wav"]
    assert len({a["hash"], b["hash"], half["hash"]}) == 3              # three files
    assert a["quick"] == b["quick"] == half["quick"]                   # the first 8 KB alike
    assert f["Qzv Loop.wav"]["hash"] != f["Qzv Loop Fill.wav"]["hash"]
    with S.session_scope() as s:
        copies = s.execute(text("SELECT file_hash FROM samples WHERE filename = 'Qzv Clap A.wav'")).all()
    assert len(copies) == 2 and copies[0] == copies[1]                 # a byte copy is the same file


def test_a_rating_covers_only_its_own_files_bytes(lib, tmp_path):
    """A Drop reaches the same recording's byte copy (Reach._same, by hash), never a file
    that only starts like it."""
    from fourier.packs.identity import Reach
    _start_alike(lib)
    IMP.scan_filesystem(lib)
    with S.session_scope() as s:
        rows = {r.path: r for r in s.query(Sample).all()}
    a = str(lib / "Acme" / "Claps" / "Qzv Clap A.wav")
    reach = Reach(drops={a}, facts={a: (rows[a].file_hash, rows[a].duration_s)})
    b, copy = str(lib / "Acme" / "Claps" / "Qzv Clap B.wav"), str(lib / "Acme" / "Copies" / "Qzv Clap A.wav")
    assert reach.dropped(b, rows[b].file_hash, rows[b].duration_s) is None
    assert reach.dropped(copy, rows[copy].file_hash, rows[copy].duration_s) == a


def _analyzed(sid):
    """Stand-ins for what the analysis wrote of a sample: its features, an audio label."""
    with S.session_scope() as s:
        s.add(SampleFeatures(sample_id=sid, tempo_bpm=120.0, clap_model="m"))
        s.add(Label(sample_id=sid, provider="audio", kind="canonical", label="class.oneshot", rank=0))
        s.add(Descriptor(sample_id=sid, provider="audio", name="duration_s", value=1.0))
        s.add(Label(sample_id=sid, provider="path", kind="canonical", label="clap", rank=0))


def _analysis(sid):
    with S.session_scope() as s:
        return (s.query(SampleFeatures).filter_by(sample_id=sid).count(),
                sorted(p for (p,) in s.execute(text("SELECT provider FROM labels WHERE sample_id = :i"), {"i": sid})))


def test_a_file_changed_at_its_path_is_analyzed_again(lib, tmp_path):
    p = _hit(lib / "Acme" / "Kicks" / "Qzv Kick 03.wav", 60.0, seconds=0.5)
    IMP.scan_filesystem(lib)
    sid = _facts(tmp_path)["Qzv Kick 03.wav"]["id"]
    _analyzed(sid)
    old = _facts(tmp_path)["Qzv Kick 03.wav"]
    _hit(p, 440.0, seconds=3.0, lead=0.0)                               # a pad now, at the same path
    got = IMP.scan_filesystem(lib)
    new = _facts(tmp_path)["Qzv Kick 03.wav"]
    assert got.get("changed") == 1 and new["id"] == sid
    assert new["dur"] == pytest.approx(3.0) and new["hash"] != old["hash"] and new["size"] != old["size"]
    assert _analysis(sid) == (0, ["path"])                             # redone; the path's labels stay


def test_a_same_size_edit_after_the_first_8_kb_is_seen(lib, tmp_path):
    p = _hit(lib / "Acme" / "Loops" / "Qzv Loop.wav", 300.0, seconds=2.0)
    IMP.scan_filesystem(lib)
    sid = _facts(tmp_path)["Qzv Loop.wav"]["id"]
    _analyzed(sid)
    _hit(p, 300.0, seconds=2.0, tail=(0.5, 450.0))                     # its last half second edited
    os.utime(p, ns=(os.stat(p).st_atime_ns, os.stat(p).st_mtime_ns + 2_000_000_000))
    assert IMP.scan_filesystem(lib).get("changed") == 1 and _analysis(sid)[0] == 0


def test_a_file_only_touched_keeps_its_analysis(lib, tmp_path):
    p = _hit(lib / "Acme" / "Kicks" / "Qzv Kick 04.wav", 60.0, seconds=0.5)
    IMP.scan_filesystem(lib)
    sid = _facts(tmp_path)["Qzv Kick 04.wav"]["id"]
    _analyzed(sid)
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # a copy that kept no times
    got = IMP.scan_filesystem(lib)
    assert not got.get("changed") and _analysis(sid) == (1, ["audio", "path"])
    assert _facts(tmp_path)["Qzv Kick 04.wav"]["mtime"] == st.st_mtime_ns + 5_000_000_000


def test_a_database_from_before_keeps_its_analysis_and_tells_files_apart(lib, tmp_path):
    """Rows scanned before schema 3 hold the quick hash and no file time: the next scan reads
    no file for them but those whose first 8 KB another file shares, and redoes no analysis."""
    _start_alike(lib)
    _hit(lib / "Acme" / "Pads" / "Qzv Pad.wav", 500.0, lead=0.0)          # no other starts like it
    IMP.scan_filesystem(lib)
    with S.session_scope() as s:                                          # as migration 3 leaves them
        s.execute(text("UPDATE samples SET file_hash = quick_hash, file_mtime_ns = NULL"))
    for f in _facts(tmp_path).values():
        _analyzed(f["id"])
    got = IMP.scan_filesystem(lib)
    f = _facts(tmp_path)
    assert not got.get("changed") and got.get("read_in_full") == 6      # the claps and the loops, not the pad
    assert f["Qzv Clap A.wav"]["hash"] != f["Qzv Clap B.wav"]["hash"] != f["Qzv Clap A half.wav"]["hash"]
    assert all(x["mtime"] is not None for x in f.values())
    assert all(_analysis(x["id"])[0] == 1 for x in f.values())


def test_migration_3_keeps_the_quick_hash_it_had(tmp_path):
    e = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with e.begin() as c:
        c.execute(text("CREATE TABLE samples (id INTEGER PRIMARY KEY, path TEXT, file_hash TEXT)"))
        c.execute(text("INSERT INTO samples VALUES (1, '/l/a.wav', 'abc'), (2, '/l/b.wav', NULL)"))
        c.execute(text("CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL)"))
        c.execute(text("INSERT INTO schema_version VALUES (2, 'then')"))
    assert M.migrate(e) == [3]
    with e.connect() as c:
        assert c.execute(text("SELECT id, file_hash, quick_hash, file_mtime_ns FROM samples ORDER BY id")).all() \
            == [(1, "abc", "abc", None), (2, None, None, None)]


# --- rows from before schema 3, rows an older version writes, and Sononym mode ---------------

def _from_before(names=None):
    """Rows as migration 3 leaves a database scanned before it: file_hash is the quick hash,
    no file time."""
    where = "" if names is None else " WHERE filename IN (%s)" % ", ".join(f"'{n}'" for n in names)
    with S.session_scope() as s:
        s.execute(text("UPDATE samples SET file_hash = quick_hash, file_mtime_ns = NULL" + where))


def _bump(p: Path, seconds=5):
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def _edit_tail(p: Path):
    """Its last 0.3 s silenced in place: the same size and the same first 8 KB."""
    b = bytearray(p.read_bytes())
    k = int(0.3 * SR) * 2
    b[-k:] = bytes(k)
    p.write_bytes(bytes(b))


def test_a_same_size_edit_on_a_row_from_before_is_seen(lib, tmp_path):
    """A row from before has no content hash to compare: after its first scan takes its time,
    a moved time with the same size and first 8 KB counts as a change (an edit past the first
    8 KB can't be told from a touch), so the analysis is redone."""
    p = _hit(lib / "Acme" / "Kicks" / "Qzv Kick 06.wav", 60.0, seconds=1.0, lead=0.0)
    q = _hit(lib / "Acme" / "Kicks" / "Qzv Kick 07.wav", 70.0, seconds=1.0, lead=0.0)
    IMP.scan_filesystem(lib)
    _from_before()
    ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
    for sid in ids.values():
        _analyzed(sid)
    got = IMP.scan_filesystem(lib)                                      # the first scan after the upgrade
    assert not got.get("changed") and not got.get("read_in_full")
    _edit_tail(p)
    _bump(p)
    _bump(q)                                                            # only touched
    got = IMP.scan_filesystem(lib)
    f = _facts(tmp_path)
    assert got.get("changed") == 2
    assert _analysis(ids["Qzv Kick 06.wav"])[0] == 0 and _analysis(ids["Qzv Kick 07.wav"])[0] == 0
    assert f["Qzv Kick 06.wav"]["hash"] != f["Qzv Kick 06.wav"]["quick"]  # exact from here
    _bump(q)                                                            # touched again: now it can tell
    with S.session_scope() as s:
        s.add(SampleFeatures(sample_id=ids["Qzv Kick 07.wav"], tempo_bpm=120.0))
    assert not IMP.scan_filesystem(lib).get("changed") and _analysis(ids["Qzv Kick 07.wav"])[0] == 1


def test_a_stored_size_that_isnt_the_files_keeps_the_analysis(lib, tmp_path):
    """A row from before with no size, or another source's size, and the same bytes keeps its
    analysis (the header says the same length) and takes the file's size; a file whose length
    really changed is analyzed again."""
    a = _hit(lib / "Acme" / "Hats" / "Qzv Hat Null.wav", 5000.0, lead=0.0)
    b = _hit(lib / "Acme" / "Hats" / "Qzv Hat Wrong.wav", 6000.0, lead=0.0)
    c = _hit(lib / "Acme" / "Hats" / "Qzv Hat Cut.wav", 7000.0, seconds=2.0, lead=0.0)
    whole = c.read_bytes()
    c.write_bytes(whole[:len(whole) // 2])                              # a cut-off download
    IMP.scan_filesystem(lib)
    _from_before()
    with S.session_scope() as s:
        s.execute(text("UPDATE samples SET file_size_bytes = NULL WHERE filename = 'Qzv Hat Null.wav'"))
        s.execute(text("UPDATE samples SET file_size_bytes = file_size_bytes + 1 "
                       "WHERE filename = 'Qzv Hat Wrong.wav'"))
    ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
    for sid in ids.values():
        _analyzed(sid)
    c.write_bytes(whole)                                                # ... downloaded in full since
    got = IMP.scan_filesystem(lib)
    f = _facts(tmp_path)
    assert got.get("changed") == 1 and _analysis(ids["Qzv Hat Cut.wav"])[0] == 0
    assert f["Qzv Hat Cut.wav"]["dur"] == pytest.approx(2.0)
    for n, p in (("Qzv Hat Null.wav", a), ("Qzv Hat Wrong.wav", b)):
        assert _analysis(ids[n])[0] == 1
        assert f[n]["size"] == p.stat().st_size and f[n]["mtime"] is not None


def test_rows_an_older_version_writes_are_read_in_full_and_keep_their_analysis(lib, tmp_path):
    """An older version (or a revert) writing into a version-3 database stores the quick hash
    in file_hash and no quick_hash: the next open moves it to quick_hash, the next scan reads
    those files in full, and a later touch keeps their analysis."""
    _start_alike(lib)
    IMP.scan_filesystem(lib)
    older = ("Qzv Clap B.wav", "Qzv Loop Fill.wav")
    with S.session_scope() as s:                                        # as an older version writes them
        s.execute(text("UPDATE samples SET file_hash = quick_hash, quick_hash = NULL, file_mtime_ns = NULL "
                       "WHERE filename IN ('Qzv Clap B.wav', 'Qzv Loop Fill.wav')"))
    ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
    for n in older:
        _analyzed(ids[n])
    db = S._engine.url.database
    S._engine.dispose()
    S._engine = S._SessionLocal = None
    S.init_db(db)                                                       # the next open
    f = _facts(tmp_path)
    assert all(f[n]["hash"] is None and f[n]["quick"] for n in older)
    got = IMP.scan_filesystem(lib)
    f = _facts(tmp_path)
    assert not got.get("changed")
    assert f["Qzv Clap A.wav"]["hash"] != f["Qzv Clap B.wav"]["hash"]
    assert f["Qzv Loop.wav"]["hash"] != f["Qzv Loop Fill.wav"]["hash"]
    for n in older:
        _bump(lib / "Acme" / ("Claps" if "Clap" in n else "Loops") / n)
    assert not IMP.scan_filesystem(lib).get("changed")
    assert all(_analysis(ids[n])[0] == 1 for n in older)


def test_older_rows_are_settled_at_every_open(tmp_path):
    """settle_older_rows: a row with file_hash and no quick_hash (an older version's) gets
    its quick hash from file_hash and file_hash cleared; a 64-digit hash is cut to 16."""
    e = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with e.begin() as c:
        c.execute(text("CREATE TABLE samples (id INTEGER PRIMARY KEY, path TEXT, file_hash TEXT)"))
        c.execute(text("INSERT INTO samples VALUES (1, '/l/a.wav', :h)"), {"h": "ab" * 32})
        c.execute(text("CREATE TABLE schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL)"))
        c.execute(text("INSERT INTO schema_version VALUES (2, 'then')"))
    assert M.migrate(e) == [3]
    with e.begin() as c:
        c.execute(text("INSERT INTO samples (id, path, file_hash) VALUES (2, '/l/b.wav', 'cdcdcdcdcdcdcdcd')"))
    assert M.migrate(e) == [] and M.migrate(e) == []
    with e.connect() as c:
        assert c.execute(text("SELECT id, file_hash, quick_hash FROM samples ORDER BY id")).all() \
            == [(1, "ab" * 8, "ab" * 8), (2, None, "cdcdcdcdcdcdcdcd")]


def _sononym(path, lib, rows):
    """A Sononym database: rows [(rel path, modtime, size)]."""
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
    for i, (rel, modtime, size) in enumerate(rows, 1):
        dur = sf.info(str(lib / rel)).duration
        c.execute("INSERT INTO assets VALUES (?, ?, ?, 'succeeded', ?, 'wav', ?, ?, 44100, 1, 16, "
                  "['OneShot'], [0.1, 0.9], ['Perc Claps'], [0.9], NULL, NULL, NULL, -1.0, -12.0, 4.0, "
                  "NULL, NULL, 0.3, 0.2, 0.4, 0, 0)", [i, rel, modtime, rel.rsplit("/", 1)[-1], size, dur])
    c.close()


def _sync(tmp_path, lib, rows):
    db = tmp_path / "sononym.db"
    if db.exists():
        db.unlink()
    _sononym(db, lib, rows)
    return IMP.sync_from_sononym(sononym_db=db, library_root=lib)


def test_a_sononym_home_is_settled_on_its_first_sync_after_the_upgrade(lib, tmp_path):
    """Sononym mode (no walk): the first sync after the upgrade reads in full the files whose
    first 8 KB another shares, though nothing is new in Sononym, and keeps their analysis; a
    size Sononym stores that isn't the file's doesn't stop it."""
    rels = ["Acme/Claps/Qzv Clap A.wav", "Acme/Claps/Qzv Clap B.wav", "Acme/Pads/Qzv Pad.wav"]
    _hit(lib / rels[0], 1200.0)
    _hit(lib / rels[1], 1700.0)
    _hit(lib / rels[2], 500.0, lead=0.0)
    sizes = [os.path.getsize(lib / r) for r in rels]
    rows = [(rels[0], 1, sizes[0]), (rels[1], 1, sizes[1] + 1), (rels[2], 1, sizes[2])]
    _sync(tmp_path, lib, rows)
    _from_before()
    with S.session_scope() as s:
        s.execute(text("UPDATE samples SET file_size_bytes = file_size_bytes + 1 WHERE filename = 'Qzv Clap B.wav'"))
    ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
    for sid in ids.values():
        _analyzed(sid)
    got = _sync(tmp_path, lib, rows)                                    # nothing new in Sononym
    f = _facts(tmp_path)
    assert f["Qzv Clap A.wav"]["hash"] != f["Qzv Clap B.wav"]["hash"]
    assert got["created"] == 0 and got["updated"] == 0 and not got.get("changed")
    assert got.get("read_in_full") == 2                                 # the claps, not the pad
    assert all(_analysis(sid)[0] == 1 for sid in ids.values())
    assert _sync(tmp_path, lib, rows).get("read_in_full") == 0          # settled: nothing more to read


def test_sononym_sees_a_same_size_edit_on_a_row_from_before(lib, tmp_path):
    """Sononym reports a file's time moved: a row from before with the same size and first
    8 KB is analyzed again; a row with the content hash only when its bytes changed."""
    rels = ["Acme/Loops/Qzv Loop.wav", "Acme/Loops/Qzv Loop 2.wav", "Acme/Kicks/Qzv Kick.wav"]
    for r, hz in zip(rels, (300.0, 330.0, 60.0)):
        _hit(lib / r, hz, seconds=1.0, lead=0.0)
    sizes = [os.path.getsize(lib / r) for r in rels]
    _sync(tmp_path, lib, [(r, 1, n) for r, n in zip(rels, sizes)])
    _from_before(["Qzv Loop.wav", "Qzv Loop 2.wav"])                    # the kick has its content hash
    ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
    for sid in ids.values():
        _analyzed(sid)
    _edit_tail(lib / rels[0])
    _bump(lib / rels[0])
    _bump(lib / rels[2])                                                # touched only
    got = _sync(tmp_path, lib, [(rels[0], 2, sizes[0]), (rels[1], 1, sizes[1]), (rels[2], 2, sizes[2])])
    assert got.get("changed") == 1
    assert _analysis(ids["Qzv Loop.wav"])[0] == 0
    assert _analysis(ids["Qzv Loop 2.wav"])[0] == 1 and _analysis(ids["Qzv Kick.wav"])[0] == 1


def test_a_rating_stored_with_a_quick_hash_follows_only_a_file_of_its_name(tmp_path):
    """A rating whose file is gone, stored with the hash of its first 8 KB, doesn't move to
    another file that only starts like it; it moves to one of the same name."""
    from fourier.packs.ratings import load_store, rekey_by_hash, save_store

    class Sess:
        def __init__(self, rows):
            self.rows = rows

        def execute(self, q):
            rows = self.rows

            class R:
                def all(self):
                    return rows
            return R()

    other = tmp_path / "Acme" / "Qzv Clap B.wav"
    moved = tmp_path / "Moved" / "Qzv Clap A.wav"
    for p in (other, moved):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    store = str(tmp_path / "ratings.json")
    rating = {"verdict": "drop", "category": "CLAPS", "hash": "q1"}
    save_store({"version": 1, "ratings": {"/gone/Qzv Clap A.wav": dict(rating)},
                "colors": {}, "auto_misfiled": {}}, store)
    s = rekey_by_hash(Sess([(str(other), "h2", "q1")]), store, log=lambda x: None)
    assert s["moved"] == 0 and "/gone/Qzv Clap A.wav" in load_store(store)["ratings"]
    s = rekey_by_hash(Sess([(str(other), "h2", "q1"), (str(moved), "h1", "q1")]), store, log=lambda x: None)
    assert s["moved"] == 1 and str(moved) in load_store(store)["ratings"]


def test_the_clap_index_is_stale_when_a_samples_analysis_was_dropped(lib, tmp_path, monkeypatch):
    """forget_analysis drops a sample's CLAP embedding; the index still holds its vector, so
    curation rebuilds it (index_stale compares the samples, not only their count)."""
    from fourier.analysis import clap_features as CF
    from fourier.packs.curate import index_stale
    monkeypatch.setenv("FOURIER_CLAP_INDEX", str(tmp_path / "clap_index.npz"))
    _hit(lib / "Acme" / "Kicks" / "Qzv Kick 1.wav", 60.0)
    _hit(lib / "Acme" / "Kicks" / "Qzv Kick 2.wav", 70.0)
    IMP.scan_filesystem(lib)
    ids = sorted(f["id"] for f in _facts(tmp_path).values())
    with S.session_scope() as s:
        for i in ids:
            s.add(SampleFeatures(sample_id=i, clap_embedding=CF.embedding_to_bytes(np.ones(512) * i)))
    with S.session_scope() as s:
        CF.build_index(s)
        assert index_stale(s)[0] is False
        IMP.forget_analysis(s, ids[0])
        s.add(SampleFeatures(sample_id=ids[0]))                         # analyzed again, CLAP not yet
    with S.session_scope() as s:
        assert index_stale(s)[0] is True


def test_a_row_an_older_version_synced_keeps_its_analysis_in_a_sononym_home(lib, tmp_path):
    """Sononym mode: a row an older version wrote is read in full by the next sync, and a
    later touch Sononym reports keeps its analysis."""
    rels = ["Acme/Claps/Qzv Clap A.wav", "Acme/Claps/Qzv Clap B.wav"]
    _hit(lib / rels[0], 1200.0)
    _hit(lib / rels[1], 1700.0)
    rows = [(r, 1, os.path.getsize(lib / r)) for r in rels]
    _sync(tmp_path, lib, rows)
    with S.session_scope() as s:                                        # as an older version writes it
        s.execute(text("UPDATE samples SET file_hash = quick_hash, quick_hash = NULL, file_mtime_ns = NULL "
                       "WHERE filename = 'Qzv Clap B.wav'"))
    sid = _facts(tmp_path)["Qzv Clap B.wav"]["id"]
    _analyzed(sid)
    db = S._engine.url.database
    S._engine.dispose()
    S._engine = S._SessionLocal = None
    S.init_db(db)
    got = _sync(tmp_path, lib, rows)
    f = _facts(tmp_path)
    assert got.get("read_in_full") == 1 and not got.get("changed")
    assert f["Qzv Clap A.wav"]["hash"] != f["Qzv Clap B.wav"]["hash"] and f["Qzv Clap B.wav"]["mtime"]
    _bump(lib / rels[1])
    got = _sync(tmp_path, lib, [rows[0], (rels[1], 2, rows[1][2])])
    assert not got.get("changed") and _analysis(sid)[0] == 1


# --- times that moved without the bytes: a copy or restore, a clock shift ---------------------

def _shift(p: Path, ns: int):
    st = os.stat(p)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + ns))


def _upgraded(lib, tmp_path, n):
    """n rows from before (the quick hash, then the time the first scan after the upgrade
    took), each analyzed."""
    paths = [_hit(lib / "Acme" / "Kicks" / f"Qzv Kick {i:02}.wav", 50.0 + i, seconds=0.3, lead=0.0)
             for i in range(n)]
    for p in paths:                                                     # times with a fraction of a second
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, (st.st_mtime_ns // 10**9) * 10**9 + 123_456_789))
    IMP.scan_filesystem(lib)
    _from_before()
    ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
    for sid in ids.values():
        _analyzed(sid)
    got = IMP.scan_filesystem(lib)                                      # the first scan after the upgrade
    assert not got.get("changed") and not got.get("read_in_full")
    return paths, ids


def test_a_library_copied_without_its_times_keeps_its_analysis(lib, tmp_path):
    """Many rows from before whose times moved with the same size, first 8 KB and header at
    once are a copy or a restore that didn't keep the times: read in full, analysis kept, and
    counted (the scan says so in one line)."""
    from fourier.cli.ingest import scan_report
    paths, ids = _upgraded(lib, tmp_path, IMP.MANY_TIMES_MOVED + 1)
    for p in paths:
        _shift(p, 37 * 10**9 + 1)
    got = IMP.scan_filesystem(lib)
    assert not got.get("changed") and got.get("times_moved") == len(paths)
    assert all(_analysis(sid)[0] == 1 for sid in ids.values())
    f = _facts(tmp_path)
    assert all(x["hash"] != x["quick"] for x in f.values())            # exact from here
    lines = " ".join(t for t, _style in scan_report(got))
    assert f"{len(paths)} file(s)" in lines and "analysis kept" in lines and "--force" in lines, lines
    _shift(paths[0], 5 * 10**9)                                         # touched again: the content hash decides
    assert not IMP.scan_filesystem(lib).get("changed")


@pytest.mark.parametrize("shift", ["hour", "dst-on-fat", "seconds-cut"])
def test_a_clock_shift_on_a_row_from_before_keeps_its_analysis(lib, tmp_path, shift):
    """A time moved by whole hours (a drive's time zone or daylight saving time), or cut to
    whole seconds by a copy, is the same file when the size, first 8 KB and header are: even
    for a few files."""
    paths, ids = _upgraded(lib, tmp_path, 2)
    for p in paths:
        st = os.stat(p)
        t = st.st_mtime_ns
        new = {"hour": t + 3600 * 10**9, "seconds-cut": t // 10**9 * 10**9,
               "dst-on-fat": (t // (2 * 10**9)) * 2 * 10**9 - 3600 * 10**9}[shift]
        os.utime(p, ns=(st.st_atime_ns, new))
    got = IMP.scan_filesystem(lib)
    assert not got.get("changed") and not got.get("times_moved")
    assert all(_analysis(sid)[0] == 1 for sid in ids.values())


def test_a_few_moved_times_on_rows_from_before_are_taken_as_edits(lib, tmp_path):
    """A few rows from before with a moved time and the same size, first 8 KB and header
    can't be told from an edit past the first 8 KB: they're analyzed again."""
    paths, ids = _upgraded(lib, tmp_path, 3)
    for p in paths:
        _shift(p, 37 * 10**9 + 1)
    got = IMP.scan_filesystem(lib)
    assert got.get("changed") == 3 and not got.get("times_moved")
    assert all(_analysis(sid)[0] == 0 for sid in ids.values())


def test_sononym_keeps_the_analysis_of_a_library_copied_without_its_times(lib, tmp_path):
    rels = [f"Acme/Kicks/Qzv Kick {i:02}.wav" for i in range(IMP.MANY_TIMES_MOVED + 1)]
    for i, r in enumerate(rels):
        _hit(lib / r, 50.0 + i, seconds=0.3, lead=0.0)
    rows = [(r, int(os.stat(lib / r).st_mtime), os.path.getsize(lib / r)) for r in rels]
    _sync(tmp_path, lib, rows)
    _from_before()
    ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
    for sid in ids.values():
        _analyzed(sid)
    for r in rels:
        _shift(lib / r, 37 * 10**9 + 1)
    got = _sync(tmp_path, lib, [(r, int(os.stat(lib / r).st_mtime), n) for r, _t, n in rows])
    assert not got.get("changed") and got.get("times_moved") == len(rels)
    assert all(_analysis(sid)[0] == 1 for sid in ids.values())


# --- the documented rollback on DuckDB, and the open-time repair ------------------------------

def test_the_documented_rollback_and_upgrading_again_work_on_duckdb(tmp_path, monkeypatch):
    """Rollback step 2 on DuckDB (8 KB hashes back in file_hash, the file_hash index made for
    older code), older code adding an analyzed file, then this version again: the open drops
    the index, the open and the scan work, and no analysis is lost."""
    pytest.importorskip("duckdb")
    root = tmp_path / "SampleLibrary"
    root.mkdir()
    monkeypatch.setenv("FOURIER_LIBRARY", str(root))
    places.reset()
    S._engine = S._SessionLocal = None
    R.forget_current()
    db = tmp_path / "t.duckdb"
    S.init_db(db)
    try:
        _start_alike(root)
        _hit(root / "Acme" / "Claps" / "Qzv Clap C.wav", 2200.0)
        IMP.scan_filesystem(root)
        with S.session_scope() as s:                                    # older code's row: the quick hash only
            s.execute(text("UPDATE samples SET file_hash = quick_hash, quick_hash = NULL, file_mtime_ns = NULL "
                           "WHERE filename = 'Qzv Clap C.wav'"))
            s.execute(text("UPDATE samples SET file_hash = quick_hash WHERE quick_hash IS NOT NULL"))
        with S.session_scope() as s:
            s.execute(text("CREATE INDEX ix_samples_file_hash ON samples (file_hash)"))
        ids = {n: f["id"] for n, f in _facts(tmp_path).items()}
        for sid in ids.values():
            _analyzed(sid)
        S._engine.dispose()
        S._engine = S._SessionLocal = None
        S.init_db(db)                                                   # this version again
        with S.session_scope() as s:
            assert not s.execute(text("SELECT index_name FROM duckdb_indexes() "
                                      "WHERE index_name = 'ix_samples_file_hash'")).all()
        got = IMP.scan_filesystem(root)
        f = _facts(tmp_path)
        assert not got.get("changed") and got.get("read_in_full")
        assert len({f[n]["hash"] for n in ("Qzv Clap A.wav", "Qzv Clap B.wav", "Qzv Clap C.wav")}) == 3
        assert all(_analysis(sid)[0] == 1 for sid in ids.values())
    finally:
        S._engine.dispose()
        S._engine = S._SessionLocal = None
        places.reset()


def test_an_open_takes_no_write_lock_on_a_settled_home(tmp_path):
    """The open-time repair only reads a home with nothing to repair: an open waits for no
    other process's write."""
    import sqlite3
    path = tmp_path / "t.db"
    e = create_engine(f"sqlite:///{path}", connect_args={"timeout": 0.2})
    with e.begin() as c:
        c.execute(text("CREATE TABLE samples (id INTEGER PRIMARY KEY, path TEXT, file_hash TEXT)"))
        c.execute(text("INSERT INTO samples VALUES (1, '/l/a.wav', 'abababababababab')"))
    M.migrate(e)
    other = sqlite3.connect(path)
    try:
        other.execute("BEGIN IMMEDIATE")                                # another process writing
        assert M.migrate(e) == []
    finally:
        other.rollback()
        other.close()
        e.dispose()
