"""The library walk (fourier/ingest/walk.py, importer.scan_filesystem, `fourier tools scan`):
every format soundfile reads, what it skips said by extension, unreadable files as errors,
folder symlinks followed out of the library with a loop guard, samples it no longer finds
marked missing (and left out of curation), a missing or empty library folder not walked,
unchanged folders not listed again with the same result, and the scan's order of steps with
Sononym and Live."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from fourier import places
from fourier.db import session as S
from fourier.ingest import formats
from fourier.ingest import importer as IMP
from fourier.ingest import walk as W
from fourier.metadata import rows as R

SR = 44100


def _wav(p: Path, seconds=0.2, sr=SR, **kw):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    sf.write(str(p), (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr, **kw)
    return p


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A library folder (FOURIER_LIBRARY) and a fresh SQLite database."""
    root = tmp_path / "SampleLibrary"
    root.mkdir()
    monkeypatch.setenv("FOURIER_LIBRARY", str(root))
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "out" / "FourierCurated"))
    places.reset()
    S._engine = S._SessionLocal = None
    R.forget_current()
    S.init_db(tmp_path / "t.db")
    yield root
    S._engine = S._SessionLocal = None
    places.reset()


def _rows(tmp_path, sql="SELECT path, rel_path, filename, file_size_bytes, file_format, "
                        "duration_s, sample_rate, channels, file_hash FROM samples ORDER BY id"):
    con = sqlite3.connect(tmp_path / "t.db")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _missing(tmp_path):
    return sorted(p for (p,) in _rows(tmp_path, "SELECT path FROM missing_files"))


# --- formats ------------------------------------------------------------------------------

def test_one_list_of_audio_formats_by_what_soundfile_reads():
    readable = formats.readable_exts()
    for e in (".wav", ".aif", ".aiff", ".aifc", ".flac", ".ogg"):
        assert e in readable
    assert ".m4a" in formats.unreadable_exts()             # no AAC in libsndfile
    assert set(readable) | set(formats.unreadable_exts()) == set(formats.AUDIO_EXTS)
    # import-folder copies the lossless ones and decodes the rest it can read
    assert formats.COPY_EXTS <= set(readable) | {".aifc"}
    assert ".mp3" in formats.convert_exts() or ".mp3" not in readable
    # Live's tag step keeps its own list while Sononym indexes the library
    assert formats.LIVE_TAG_EXTS == (".aif", ".aiff", ".wav", ".ogg", ".flac", ".mp3")


def test_the_walk_reads_every_format_and_says_what_it_skipped(lib, tmp_path):
    _wav(lib / "Acme" / "Kicks" / "Kick 01.wav")
    _wav(lib / "Acme" / "Kicks" / "Kick 02.flac")
    _wav(lib / "Acme" / "Kicks" / "Kick 03.aifc", format="AIFF")
    _wav(lib / "Acme" / "Kicks" / "Kick 04.ogg")
    if ".caf" in formats.readable_exts():
        _wav(lib / "Acme" / "Kicks" / "Kick 05.caf", format="CAF")
    (lib / "Acme" / "Kicks" / "Kick 06.m4a").write_bytes(b"\0" * 64)   # AAC: not readable here
    (lib / "Acme" / "readme.txt").write_text("x")
    (lib / "Acme" / "Kicks" / "._Kick 01.wav").write_bytes(b"\0" * 64)  # a dot file: not audio
    (lib / ".Trash").mkdir()
    _wav(lib / ".Trash" / "Old.wav")
    c = IMP.scan_filesystem(lib)
    names = sorted(r[2] for r in _rows(tmp_path))
    want = ["Kick 01.wav", "Kick 02.flac", "Kick 03.aifc", "Kick 04.ogg"]
    if ".caf" in formats.readable_exts():
        want.append("Kick 05.caf")
    assert names == sorted(want)
    assert c["created"] == len(want) and c["errors"] == 0
    assert c["skipped_ext"] == {".m4a": 1, ".txt": 1}
    # every row has its header facts and a library path with "/"
    assert all(r[1].startswith("Acme/Kicks/") and r[5] > 0 and r[6] == SR and r[8] for r in _rows(tmp_path))


def test_corrupt_and_empty_files_are_scan_errors_named(lib, tmp_path):
    _wav(lib / "Acme" / "Snare 01.wav")
    (lib / "Acme" / "Snare 02.wav").write_bytes(b"")
    (lib / "Acme" / "Snare 03.wav").write_bytes(b"RIFF\0\0\0\0garbage" * 4)
    c = IMP.scan_filesystem(lib)
    assert c["created"] == 1 and c["errors"] == 2
    assert any("Snare 02.wav (empty file)" in f for f in c["error_files"])
    assert any("Snare 03.wav (can't be read)" in f for f in c["error_files"])
    from fourier.cli.ingest import scan_report
    text = "\n".join(t for t, _s in scan_report(c))
    assert "2 file(s) couldn't be read" in text and "Snare 03.wav" in text


def test_errors_are_listed_ten_at_most(lib):
    for i in range(14):
        (lib / "Acme" / f"Bad {i:02d}.wav").parent.mkdir(parents=True, exist_ok=True)
        (lib / "Acme" / f"Bad {i:02d}.wav").write_bytes(b"")
    c = IMP.scan_filesystem(lib)
    assert c["errors"] == 14 and len(c["error_files"]) == IMP.ERRORS_LISTED
    from fourier.cli.ingest import scan_report
    assert any("and 4 more" in t for t, _s in scan_report(c))


# --- symlinks -----------------------------------------------------------------------------

def test_folder_symlinks_out_of_the_library_are_followed_once(lib, tmp_path):
    outside = tmp_path / "Other Drive" / "Northwind Loops"
    _wav(outside / "Loops" / "Loop 01.wav")
    os.symlink(outside / "Loops", outside / "Loops" / "Again")         # a loop back up
    os.symlink(outside, lib / "Northwind Loops")                      # the pack kept elsewhere
    _wav(lib / "Acme" / "Hats" / "Hat 01.wav")
    os.symlink(lib / "Acme", lib / "Acme Copy")                       # inside: walked at its target
    c = IMP.scan_filesystem(lib)
    rows = _rows(tmp_path)
    assert sorted(r[1] for r in rows) == ["Acme/Hats/Hat 01.wav", "Northwind Loops/Loops/Loop 01.wav"]
    loop = next(r for r in rows if r[2] == "Loop 01.wav")
    # stored under the library folder, where the link puts it: its pack is its library path
    real = os.path.realpath(lib)
    assert loop[0] == os.path.join(real, "Northwind Loops", "Loops", "Loop 01.wav")
    assert places.library_rel(loop[0]) == "Northwind Loops/Loops/Loop 01.wav"
    assert c["links_followed"] == 1 and c["links_inside"] == 1 and c["loops"] == 1


def test_a_symlinked_library_folder_gives_the_same_library_paths(tmp_path, monkeypatch):
    real = tmp_path / "Drive" / "SampleLibrary"
    _wav(real / "Acme" / "Kick 01.wav")
    link = tmp_path / "SampleLibraryLink"
    os.symlink(real, link)
    monkeypatch.setenv("FOURIER_LIBRARY", str(link))
    places.reset()
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "t.db")
    try:
        IMP.scan_filesystem(link)
        (path, rel, *_r), = _rows(tmp_path)
        assert path == str(real / "Acme" / "Kick 01.wav") and rel == "Acme/Kick 01.wav"
        assert places.library_rel(path) == "Acme/Kick 01.wav"        # the root's real path
        assert places.library_root_of(path) == str(link)
    finally:
        S._engine = S._SessionLocal = None
        places.reset()


# --- missing files ------------------------------------------------------------------------

def _analysed(tmp_path):
    """Give every sample the fallback classifier's mark and a CLAP embedding, as analysis
    would, so sample_select and the scale count see them."""
    with S.session_scope() as s:
        from sqlalchemy import text
        for (sid,) in s.execute(text("SELECT id FROM samples")).all():
            s.execute(text("INSERT OR IGNORE INTO sample_features (sample_id, clap_embedding) "
                           "VALUES (:i, :e)"), {"i": sid, "e": b"\0" * 8})


def test_a_renamed_folder_marks_the_old_rows_missing_and_curation_leaves_them_out(lib, tmp_path):
    for i in range(3):
        _wav(lib / "Acme" / "Kicks" / f"Kick {i + 1:02d}.wav", seconds=0.1 + 0.05 * i)
    _wav(lib / "Northwind" / "Pads" / "Pad 01.wav")
    IMP.scan_filesystem(lib)
    _analysed(tmp_path)
    with S.session_scope() as s:
        assert not R.missing_marked(s) and R.usable_count(s) == 4
    (lib / "Acme").rename(lib / "Acme Audio")
    c = IMP.scan_filesystem(lib)
    # the same files at a new path: moved, not missing (their old rows record where they went)
    assert c["created"] == 3 and c["moved"] == 3 and c["missing"] == 0 and c["newly_missing"] == 0
    assert [os.path.basename(p) for p in _missing(tmp_path)] == ["Kick 01.wav", "Kick 02.wav", "Kick 03.wav"]
    _analysed(tmp_path)
    with S.session_scope() as s:
        assert R.missing_marked(s)
        assert R.usable_count(s) == 4                       # 7 rows, 3 of them missing
        rows = R.fetch(s, R.sample_select("id", "rel_path", session=s))
    assert sorted(r.rel_path for r in rows) == ["Acme Audio/Kicks/Kick 01.wav", "Acme Audio/Kicks/Kick 02.wav",
                                                "Acme Audio/Kicks/Kick 03.wav", "Northwind/Pads/Pad 01.wav"]
    # renamed back: found again
    (lib / "Acme Audio").rename(lib / "Acme")
    c = IMP.scan_filesystem(lib)
    assert c["found_again"] == 3 and c["created"] == 0
    assert [os.path.basename(p) for p in _missing(tmp_path)] == ["Kick 01.wav", "Kick 02.wav", "Kick 03.wav"]
    assert all("Acme Audio" in p for p in _missing(tmp_path))


def test_a_missing_or_empty_library_folder_marks_nothing(lib, tmp_path):
    _wav(lib / "Acme" / "Kick 01.wav")
    IMP.scan_filesystem(lib)
    gone = tmp_path / "Unmounted"
    os.rename(lib, gone)
    c = IMP.scan_filesystem(lib)
    assert c["root"] == "missing" and _missing(tmp_path) == []
    lib.mkdir()                                 # a mount point with nothing mounted
    c = IMP.scan_filesystem(lib)
    assert c["root"] == "empty" and _missing(tmp_path) == []
    from fourier.cli import main
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), "tools", "scan", "--only", "files"])
    assert r.exit_code == 2 and "is empty" in r.output and "no sample was marked missing" in " ".join(r.output.split())


def test_an_unlistable_folder_marks_nothing_under_it(lib, tmp_path, monkeypatch):
    _wav(lib / "Acme" / "Kick 01.wav")
    _wav(lib / "Northwind" / "Pad 01.wav")
    IMP.scan_filesystem(lib)
    real = W._list

    def flaky(here, w):
        return None if os.path.basename(here) == "Acme" else real(here, w)
    monkeypatch.setattr(W, "_list", flaky)
    c = IMP.scan_filesystem(lib)
    assert c["unlistable"] == 1 and _missing(tmp_path) == []


def test_fourier_s_own_folders_inside_the_library_are_not_walked(lib, tmp_path, monkeypatch):
    master = lib / "Fourier Master"
    _wav(master / "KICKS" / "deep" / "Kick 01.wav")
    _wav(lib / "Acme" / "Kick 01.wav")
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(master))
    c = IMP.scan_filesystem(lib)
    assert [r[1] for r in _rows(tmp_path)] == ["Acme/Kick 01.wav"] and c["created"] == 1


# --- speed: one lookup, unchanged folders not listed again -------------------------------

def _library(root: Path):
    for v in ("Acme", "Northwind"):
        for kind in ("Kicks", "Snares"):
            for i in range(3):
                _wav(root / v / kind / f"{kind[:-1]} {i + 1:02d}.wav", seconds=0.05 + 0.01 * i)


def test_unchanged_folders_are_reused_with_the_same_result(lib, tmp_path, monkeypatch):
    _library(lib)
    monkeypatch.setattr(W, "_SETTLE_NS", -10 ** 12)       # trust folders changed just now
    monkeypatch.setattr("fourier.platforms.dir_mtimes_reliable", lambda p: True)
    c1 = IMP.scan_filesystem(lib)
    assert c1["created"] == 12 and c1["reused"] == 0 and c1["listed"] == 7
    listed = []
    real = W._list
    monkeypatch.setattr(W, "_list", lambda here, w: listed.append(here) or real(here, w))
    c2 = IMP.scan_filesystem(lib)
    assert listed == [] and c2["reused"] == 7 and c2["created"] == 0 and c2["skipped"] == 12
    # a new file changes its folder's time: that folder alone is listed again
    _wav(lib / "Acme" / "Kicks" / "Kick 04.wav")
    c3 = IMP.scan_filesystem(lib)
    assert listed == [str(lib / "Acme" / "Kicks")] and c3["created"] == 1
    # the same rows as a walk that lists every folder, into a fresh database
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "fresh.db")
    monkeypatch.setattr("fourier.platforms.dir_mtimes_reliable", lambda p: False)
    IMP.scan_filesystem(lib)
    a = _rows(tmp_path)
    con = sqlite3.connect(tmp_path / "fresh.db")
    b = con.execute("SELECT path, rel_path, filename, file_size_bytes, file_format, duration_s, "
                    "sample_rate, channels, file_hash FROM samples ORDER BY id").fetchall()
    con.close()
    assert sorted(a) == sorted(b) and len(b) == 13


def test_listings_are_not_trusted_on_a_file_system_without_reliable_folder_times(lib, monkeypatch):
    _library(lib)
    monkeypatch.setattr(W, "_SETTLE_NS", -10 ** 12)
    monkeypatch.setattr("fourier.platforms.dir_mtimes_reliable", lambda p: False)
    IMP.scan_filesystem(lib)
    c = IMP.scan_filesystem(lib)
    assert c["reused"] == 0 and c["listed"] == 7


def test_the_walk_looks_files_up_without_a_query_each(lib, monkeypatch):
    _library(lib)
    IMP.scan_filesystem(lib)
    from sqlalchemy import event
    stmts = []
    eng = S.get_engine()
    listen = lambda conn, cur, stmt, *a: stmts.append(stmt)
    event.listen(eng, "before_cursor_execute", listen)
    try:
        IMP.scan_filesystem(lib)
    finally:
        event.remove(eng, "before_cursor_execute", listen)
    assert len([s for s in stmts if "FROM samples" in s]) <= 3, stmts


def test_new_samples_get_ids_in_path_order(lib, tmp_path):
    for rel in ("b/Kick.wav", "a b/Kick.wav", "a/Kick.wav", "a/b/Kick.wav"):
        _wav(lib / rel)
    IMP.scan_filesystem(lib)
    # pathlib's order (by parts), as the walk always gave ids
    assert [r[1] for r in _rows(tmp_path)] == sorted(
        ["b/Kick.wav", "a b/Kick.wav", "a/Kick.wav", "a/b/Kick.wav"], key=lambda r: Path(r).parts)


# --- Windows paths ------------------------------------------------------------------------

def test_windows_paths_give_posix_library_paths():
    from pathlib import PureWindowsPath
    root = str(PureWindowsPath("C:/Users/me/SampleLibrary"))
    path = str(PureWindowsPath("C:/Users/me/SampleLibrary/Acme/Kicks/Kick 01.wav"))
    assert places._rel(path, (root,), ()) == "Acme/Kicks/Kick 01.wav"
    # Windows folds case, and a path may come back spelled differently
    assert places._rel(path.replace("SampleLibrary", "samplelibrary"), (root,), ()) == "Acme/Kicks/Kick 01.wav"
    assert places._rel(path, (), ("SampleLibrary",)) == "Acme/Kicks/Kick 01.wav"
    unc = str(PureWindowsPath("//nas/share/SampleLibrary/Acme/Kick 01.wav"))
    assert places._rel(unc, (str(PureWindowsPath("//nas/share/SampleLibrary")),), ()) == "Acme/Kick 01.wav"
    assert places._rel(str(PureWindowsPath("D:/Other/Kick.wav")), (root,), ()) is None
    # a POSIX path with a backslash in a name, under its library folder, as before
    assert places._rel("/lib/SampleLibrary/Acme/A\\B.wav", ("/lib/SampleLibrary",), ()) == "Acme/A\\B.wav"


def test_the_walk_stores_slash_library_paths(lib, tmp_path):
    _wav(lib / "Acme" / "Kicks" / "Kick 01.wav")
    w = W.walk(lib, formats.readable_exts())
    (f,) = w.files
    assert f.rel == "Acme/Kicks/Kick 01.wav" and "\\" not in f.rel
    assert f.key == os.path.join(os.path.realpath(lib), "Acme", "Kicks", "Kick 01.wav")
    assert W._sort_key("a/b") == ("a", "b")


# --- the order of the scan's steps --------------------------------------------------------

def _live_db(home: Path, files: dict) -> Path:
    """A minimal Live file index at ~/Library/.../Live-files-12.db: {absolute path: [auto-tags]}."""
    d = home / "Library" / "Application Support" / "Ableton" / "Live Database"
    d.mkdir(parents=True)
    db = d / "Live-files-12.db"
    con = sqlite3.connect(db)
    con.execute("create table files (file_id integer, name text, parent_id integer)")
    con.execute("create table keywords (file_id integer, keyw_id integer, is_auto integer)")
    ids, nxt = {}, [1]

    def node(path):
        if path in ids:
            return ids[path]
        parent = node(os.path.dirname(path)) if os.path.dirname(path) != path else None
        ids[path] = nxt[0]
        nxt[0] += 1
        con.execute("insert into files values (?, ?, ?)", (ids[path], os.path.basename(path) or "", parent))
        return ids[path]
    for path, tags in files.items():
        fid = node(path)
        for t in tags:
            tid = nxt[0]
            nxt[0] += 1
            con.execute("insert into files values (?, ?, ?)", (tid, t, None))
            con.execute("insert into keywords values (?, ?, 1)", (fid, tid))
    con.commit()
    con.close()
    return db


def test_with_live_and_no_sononym_the_scan_walks_then_tags(lib, tmp_path, monkeypatch):
    kick = _wav(lib / "Acme" / "Kicks" / "Kick 01.wav")
    _wav(lib / "Acme" / "Pads" / "Pad 01.flac")
    user = tmp_path / "user"
    _live_db(user, {os.path.realpath(kick): ["Kick", "Drums"],
                    os.path.realpath(lib / "Acme" / "Pads" / "Pad 01.flac"): ["Pad"]})
    monkeypatch.setenv("HOME", str(user))
    from fourier.cli import main
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), "tools", "scan"])
    assert r.exit_code == 0, r.output
    out = r.output
    assert out.index("Files under") < out.index("Live's auto-tags")
    tags = dict(_rows(tmp_path, "SELECT filename, ableton_tags FROM samples"))
    assert tags["Kick 01.wav"] and "Kick" in tags["Kick 01.wav"]
    assert tags["Pad 01.flac"] and "Pad" in tags["Pad 01.flac"]


def test_with_sononym_the_scan_walks_only_when_asked(lib, tmp_path, monkeypatch):
    from fourier.cli import ingest as I
    from fourier.cli import main
    (lib / "sononym.db").write_bytes(b"")
    ran = []
    monkeypatch.setattr(I, "_sononym_sync", lambda prune=False: ran.append("sononym"))
    monkeypatch.setattr(I, "_walk", lambda r: ran.append("walk") or True)
    monkeypatch.setattr(I, "_ableton_tags", lambda *a, **k: ran.append("ableton"))
    monkeypatch.setattr("fourier.ingest.ableton_tags.latest_live_db", lambda: "/x/Live-files-12.db")
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), "tools", "scan"])
    assert r.exit_code == 0 and ran == ["sononym", "ableton"], r.output
    ran.clear()
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), "tools", "scan", "--walk"])
    assert r.exit_code == 0 and ran == ["sononym", "walk", "ableton"], r.output


def test_live_tags_keep_their_list_when_sononym_indexes_the_library(lib, tmp_path, monkeypatch):
    from fourier.cli import ingest as I
    seen = []
    monkeypatch.setattr("fourier.ingest.ableton_tags.import_ableton_tags",
                        lambda sess, db_path=None, log=print, exts=None: seen.append(exts))
    I._ableton_tags()
    assert seen[-1] == formats.readable_exts()
    from fourier.db.models import Sample, SononymMeta
    with S.session_scope() as s:
        s.add(Sample(id=1, path="/x/a.wav", filename="a.wav"))
        s.add(SononymMeta(sample_id=1, classes=["OneShot"]))
    I._ableton_tags()
    assert seen[-1] == formats.LIVE_TAG_EXTS


# --- import-folder -----------------------------------------------------------------------

def test_import_folder_decodes_the_formats_it_reads_but_does_not_copy(tmp_path, monkeypatch):
    from fourier.ingest.folder import CONVERT_EXTS, ingest_folder
    if ".caf" not in formats.readable_exts():
        pytest.skip("this soundfile doesn't read CAF")
    assert ".caf" in CONVERT_EXTS and ".flac" not in CONVERT_EXTS
    monkeypatch.setenv("FOURIER_LIBRARY", str(tmp_path / "lib"))
    places.reset()
    src = tmp_path / "new-pack"
    _wav(src / "Kick 01.caf", format="CAF")
    _wav(src / "Kick 02.flac")
    st = ingest_folder(src, tmp_path / "lib" / "new-pack", prepare=False, log=lambda m: None)
    places.reset()
    assert st["converted"] == 1 and st["copied"] == 1
    assert (tmp_path / "lib" / "new-pack" / "Kick 01.wav").exists()


# --- very long files: what the analysis reads ---------------------------------------------

def test_clap_hears_the_start_of_a_file_longer_than_any_category_takes(tmp_path, monkeypatch):
    import sys
    import types

    from fourier.analysis import clap_features as CF
    short = _wav(tmp_path / "Pad 01.wav", seconds=2.0, sr=8000, subtype="PCM_16")
    long = _wav(tmp_path / "Field Recording 01.wav", seconds=CF.LONG_FILE_S + 1, sr=8000, subtype="PCM_16")
    assert CF.read_seconds(short) is None                       # read whole, as always
    assert CF.read_seconds(long) == CF.LONG_FILE_READ_S
    assert CF.read_seconds(tmp_path / "nope.wav") is None
    asked = []

    def load(path, sr=None, mono=True, duration=None, **kw):
        asked.append((os.path.basename(path), duration))
        raise RuntimeError("stop before the model")
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    import librosa
    monkeypatch.setattr(librosa, "load", load, raising=False)
    assert not CF.embed_audio_file(short).any() and not CF.embed_audio_file(long).any()
    # CLAP decodes its window of any file (the quality check's cap is read_seconds above)
    assert asked == [("Pad 01.wav", CF.CLAP_WINDOW_S), ("Field Recording 01.wav", CF.CLAP_WINDOW_S)]


def test_clap_hears_the_first_ten_seconds_the_same_every_time(tmp_path, monkeypatch):
    """CLAP's feature extractor crops a longer clip at a random offset: Fourier hands it the
    first CLAP_WINDOW_S itself, so a long file embeds the same on every run."""
    import contextlib
    import sys
    import types

    from fourier.analysis import clap_features as CF
    sr = CF.CLAP_SAMPLE_RATE
    t = np.arange(int(15 * sr)) / sr
    y = (0.5 * np.sin(2 * np.pi * (220 + 20 * t) * t)).astype("float32")   # every 10 s differs
    long = tmp_path / "Loop 01 15s.wav"
    sf.write(str(long), y, sr, subtype="FLOAT")
    short = _wav(tmp_path / "Hit 01.wav", seconds=2.0, sr=sr, subtype="FLOAT")
    heard = []

    class Out:
        def cpu(self): return self
        def float(self): return self
        def numpy(self): return np.ones(CF.CLAP_EMBEDDING_DIM, dtype=np.float32)

    class Processor:
        def __call__(self, audio=None, sampling_rate=None, return_tensors=None):
            heard.append(np.asarray(audio[0]).copy())
            return self

        def to(self, device):
            return {}

    class Model:
        def get_audio_features(self, **kw):
            return types.SimpleNamespace(pooler_output=[Out()])

    import librosa
    librosa.load(str(short), sr=sr, duration=CF.CLAP_WINDOW_S)  # its loaders, before the stand-in torch
    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(CF, "_load_model_and_processor", lambda: (Model(), Processor(), "cpu"))
    for path in (long, long, short):
        assert CF.embed_audio_file(path).any()
    first, again, hit = heard
    assert len(first) == CF.CLAP_WINDOW and np.array_equal(first, again)
    assert np.allclose(first, y[:CF.CLAP_WINDOW], atol=1e-6)         # the start, not a random 10 s
    assert len(hit) == 2 * sr                                        # a short file: all of it


def test_the_quality_check_reads_the_start_of_a_very_long_file(tmp_path, monkeypatch):
    from fourier.analysis import clap_features as CF
    from fourier.cli import main
    from fourier.db.models import Sample
    short = _wav(tmp_path / "SampleLibrary" / "Pad 01.wav", seconds=2.0, sr=8000, subtype="PCM_16")
    long = _wav(tmp_path / "SampleLibrary" / "Field Recording 01.wav", seconds=CF.LONG_FILE_S + 1,
                sr=8000, subtype="PCM_16")
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "t.db")
    with S.session_scope() as s:
        for p in (short, long):
            s.add(Sample(path=str(p), filename=p.name))
    S._engine = S._SessionLocal = None
    read = []
    real = sf.read

    def spy(path, *a, **kw):
        y, sr = real(path, *a, **kw)
        read.append((os.path.basename(path), len(y) / sr))
        return y, sr
    monkeypatch.setattr(sf, "read", spy)
    try:
        r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), "tools", "analyze", "--only", "quality"])
    finally:
        S._engine = S._SessionLocal = None
    assert r.exit_code == 0, r.output
    assert sorted(read) == [("Field Recording 01.wav", CF.LONG_FILE_READ_S), ("Pad 01.wav", 2.0)]


@pytest.mark.parametrize("name, folder", [
    ("Linked Pack", True), ("Pack Vol. 2", True), ("Drums 1.5", True), ("Kits.Extended Edition", True),
    ("Kick 01.wav", False), ("Read Me.pdf", False), ("Patch.nki", False),
])
def test_a_dangling_link_named_like_a_folder_is_an_unavailable_folder(name, folder):
    assert W._folder_named(name) is folder


def test_a_dangling_folder_link_is_unavailable_and_a_dangling_file_link_is_not(lib, tmp_path):
    gone = tmp_path / "unplugged"
    os.symlink(gone / "Pack", lib / "Linked Pack")
    os.symlink(gone / "notes.pdf", lib / "notes.pdf")
    w = W.walk(str(lib), formats.AUDIO_EXTS)
    assert [os.path.basename(link) for _k, link, _t in w.unavailable] == ["Linked Pack"]
