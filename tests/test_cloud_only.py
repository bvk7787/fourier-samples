"""Cloud-only source files (a cloud drive's placeholders: fourier/platforms.py cloud_only,
materialize; fourier/safety.py ensure_local): a build downloads its picks before the first
read and stops before writing anything when one won't download; analyze skips them unless
--download; doctor counts them."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from fourier import places
from fourier import platforms as P
from fourier.safety import CloudOnlySources, UnsafePath, ensure_local


# --- the flag logic ----------------------------------------------------------------------
def test_macos_dataless_flag():
    assert P.stat_cloud_only(SimpleNamespace(st_flags=0x40000000), "darwin")
    assert P.stat_cloud_only(SimpleNamespace(st_flags=0x40000000 | 0x8000), "darwin")
    assert not P.stat_cloud_only(SimpleNamespace(st_flags=0x8000), "darwin")       # SF_* others
    assert not P.stat_cloud_only(SimpleNamespace(st_flags=0), "darwin")
    assert not P.stat_cloud_only(SimpleNamespace(), "darwin")                      # no st_flags


@pytest.mark.parametrize("attr", [0x00400000, 0x00040000, 0x1000])
def test_windows_placeholder_attributes(attr):
    assert P.stat_cloud_only(SimpleNamespace(st_file_attributes=attr | 0x20), "win32")


def test_windows_ordinary_file_and_other_systems():
    assert not P.stat_cloud_only(SimpleNamespace(st_file_attributes=0x20), "win32")  # ARCHIVE
    assert not P.stat_cloud_only(SimpleNamespace(st_file_attributes=0x400), "win32")  # a reparse point alone
    # the same bits mean nothing on Linux
    assert not P.stat_cloud_only(SimpleNamespace(st_flags=0x40000000, st_file_attributes=0x1000),
                                 "linux")


def test_cloud_only_never_raises(tmp_path, monkeypatch):
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    assert P.cloud_only(f) is False
    assert P.cloud_only(tmp_path / "missing.wav") is False
    assert P.cloud_only("bad\0path") is False
    monkeypatch.setattr(P.sys, "platform", "darwin")
    monkeypatch.setattr(P.os, "stat", lambda p, follow_symlinks=True: SimpleNamespace(st_flags=0x40000000))
    assert P.cloud_only(f) is True


# --- materialize / ensure_local ------------------------------------------------------------
def _cloud(monkeypatch, cloud, stuck=(), unreadable=()):
    """platforms.cloud_only answers from `cloud`; reading a file through "downloads" it (takes
    it out of `cloud`) unless it's stuck; an unreadable one raises. Returns the reads."""
    cloud = {str(p) for p in cloud}
    reads = []
    monkeypatch.setattr(P, "cloud_only", lambda p: str(p) in cloud)

    def read(p, chunk=1 << 20):
        reads.append(str(p))
        if str(p) in {str(u) for u in unreadable}:
            raise OSError("Resource deadlock avoided")
        if str(p) not in {str(s) for s in stuck}:
            cloud.discard(str(p))
    monkeypatch.setattr(P, "_read_through", read)
    return reads


def test_materialize_reads_only_cloud_only_files(monkeypatch):
    reads = _cloud(monkeypatch, ["a", "b", "c"], stuck=["b"], unreadable=["c"])
    msgs = []
    assert P.materialize(["local", "a", "b", "c"], msgs.append) == ["b", "c"]
    assert reads == ["a", "b", "c"]                 # the local file isn't read
    assert any("can't read c" in m for m in msgs)


def test_materialize_reads_a_real_file_through(tmp_path, monkeypatch):
    f = tmp_path / "a.wav"
    f.write_bytes(b"0" * (3 << 20))
    calls = iter([True, False])                    # cloud-only, then downloaded
    monkeypatch.setattr(P, "cloud_only", lambda p: next(calls))
    assert P.materialize([f]) == []


def test_ensure_local_costs_nothing_without_cloud_only_files(monkeypatch):
    reads = _cloud(monkeypatch, [])
    msgs = []
    assert ensure_local(["a", "b", None, "a"], msgs.append) == 0
    assert reads == [] and msgs == []


def test_ensure_local_downloads_then_stops_on_what_stays_online(monkeypatch):
    _cloud(monkeypatch, ["a", "b"])
    msgs = []
    assert ensure_local(["x", "a", "b"], msgs.append) == 2
    assert msgs[0] == "2 source file(s) are cloud-only; downloading them first"
    stuck = [f"/lib/s{i:02d}.wav" for i in range(12)]
    _cloud(monkeypatch, stuck, stuck=stuck)
    with pytest.raises(CloudOnlySources) as e:
        ensure_local(stuck)
    text = str(e.value)
    assert "12 source file(s) are cloud-only" in text and "/lib/s09.wav" in text
    assert "/lib/s10.wav" not in text and "and 2 more" in text            # up to 10 listed
    assert '"Keep Downloaded"' in text and '"Always keep on this device"' in text
    assert isinstance(e.value, UnsafePath)
    back = pickle.loads(pickle.dumps(e.value))     # crosses the process pool whole
    assert isinstance(back, CloudOnlySources) and back.paths == stuck


# --- the build ------------------------------------------------------------------------------
def _kicks(monkeypatch, tmp_path, n=8):
    """One KICKS build over n generated files (the selection stubbed, as in
    tests/regressions/test_curate_regressions.py)."""
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("sklearn")
    from fourier.packs import curate
    dim = 16
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((n, dim)).astype("float32")
    rec, paths = [], []
    for i in range(n):
        p = tmp_path / "SampleLibrary" / f"V{i}" / "Pack" / f"Kick {i:02d}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), np.full(64, 0.01 * (i + 1)), 44100, subtype="PCM_16")
        paths.append(str(p))
        rec.append(dict(id=i, row=i, path=str(p), pack=f"V{i}", clip=0, br=0.5, atk=10.0,
                        dec=300.0, sub=0.1, noi=0.3, har=0.7, cr=4.0, tune=None, bpm=None,
                        bpm_src=None, bpm_fold=None, sup=1, qual=1.0, fav=0, ab=[], vendor=f"V{i}"))
    monkeypatch.setattr(curate, "index_stale", lambda s: (False, n, n))
    monkeypatch.setattr(curate, "load_index", lambda: (np.arange(n), emb))
    monkeypatch.setattr(curate, "_fetch_rows", lambda *a, **k: [])
    monkeypatch.setattr(curate, "_select_records", lambda *a, **k: (list(rec), set(), dict(twins=0)))
    monkeypatch.setattr(curate, "embed_text",
                        lambda t: np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(dim)
                        .astype("float32"))

    def build(out, msgs):
        return curate.build_taxonomy(None, "KICKS", str(out), homes={}, support={}, votes={},
                                     describe=False, loudness=False, return_entries=True,
                                     log=msgs.append)
    return paths, build


def test_a_build_downloads_its_cloud_only_picks_first(monkeypatch, tmp_path):
    paths, build = _kicks(monkeypatch, tmp_path)
    reads = _cloud(monkeypatch, paths[:2])
    msgs = []
    summary, entries = build(tmp_path / "master", msgs)
    assert "KICKS: 2 source file(s) are cloud-only; downloading them first" in msgs
    assert sorted(reads) == sorted(paths[:2]) and summary["cloud_downloaded"] == 2
    assert {e["src"] for e in entries} >= set(paths[:2])
    assert summary["files"] == len(entries) > 0


def test_no_cloud_only_picks_leave_the_summary_as_it_was(monkeypatch, tmp_path):
    _paths, build = _kicks(monkeypatch, tmp_path)
    _cloud(monkeypatch, [])
    msgs = []
    summary, _entries = build(tmp_path / "master", msgs)
    assert "cloud_downloaded" not in summary and not any("cloud-only" in m for m in msgs)


def test_a_pick_that_wont_download_stops_the_build_before_it_writes(monkeypatch, tmp_path):
    paths, build = _kicks(monkeypatch, tmp_path)
    master = tmp_path / "master"
    old = master / "KICKS" / "old-family" / "Kick Old.wav"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"previous build")
    _cloud(monkeypatch, paths[:3], stuck=[paths[1]])
    from fourier.packs import curate
    exported = []
    monkeypatch.setattr(curate.shutil, "copy2", lambda *a, **k: exported.append(a))
    with pytest.raises(CloudOnlySources) as e:
        build(master, [])
    assert e.value.paths == [paths[1]]
    assert old.read_bytes() == b"previous build"                     # the master is untouched
    assert sorted(str(p.relative_to(master)) for p in master.rglob("*") if p.is_file()) == \
        ["KICKS/old-family/Kick Old.wav"]
    assert not exported


def test_a_worker_hands_the_stop_to_the_parent(monkeypatch):
    from fourier.packs import curate
    from fourier.db import session as S

    class Scope:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(S, "session_scope", lambda: Scope())
    monkeypatch.setitem(curate._WORKER, "homes", {})
    monkeypatch.setitem(curate._WORKER, "support", {})
    monkeypatch.setitem(curate._WORKER, "votes", {})
    monkeypatch.setitem(curate._WORKER, "opts", {})

    def stuck(*a, **k):
        raise CloudOnlySources(["/lib/a.wav"])
    monkeypatch.setattr(curate, "build_taxonomy", stuck)
    cat, summary, entries, err = curate._worker_build("KICKS", "/tmp/x")
    assert isinstance(err, CloudOnlySources) and summary is None

    class Ex:
        cancelled = False

        def shutdown(self, wait=True, cancel_futures=False):
            Ex.cancelled = cancel_futures
    curate._stop_on_unsafe(Ex(), "some other failure")             # an ordinary failure: no stop
    assert not Ex.cancelled
    with pytest.raises(CloudOnlySources):
        curate._stop_on_unsafe(Ex(), err)
    assert Ex.cancelled


# --- analyze --------------------------------------------------------------------------------
@pytest.fixture
def library_db(tmp_path):
    sf = pytest.importorskip("soundfile")
    from fourier.db import session as S
    from fourier.db.models import Sample
    db = tmp_path / "t.db"
    S._engine = S._SessionLocal = None
    S.init_db(db)
    paths = []
    with S.session_scope() as s:
        for i in range(3):
            p = tmp_path / "SampleLibrary" / "Acme" / f"Hit {i}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), np.full(256, 0.1 * (i + 1)), 44100, subtype="PCM_16")
            s.add(Sample(path=str(p), filename=p.name))
            paths.append(str(p))
    yield db, paths
    S._engine = S._SessionLocal = None


def _quality_done(db):
    import sqlite3
    q = ("SELECT s.path FROM samples s JOIN sample_features f ON f.sample_id = s.id "
         "WHERE f.is_clipped IS NOT NULL")
    return sorted(r[0] for r in sqlite3.connect(db).execute(q))


def _analyze(db, *args):
    from fourier.cli import main
    return CliRunner().invoke(main, ["--db", str(db), "tools", "analyze", "--only", "quality", *args])


def test_analyze_skips_cloud_only_files_and_counts_them(library_db, monkeypatch):
    db, paths = library_db
    reads = _cloud(monkeypatch, paths[:1])
    res = _analyze(db)
    assert res.exit_code == 0, res.output
    assert "1 file(s) skipped: they are cloud-only" in res.output and "--download" in res.output
    assert reads == []                                              # nothing downloaded
    assert _quality_done(db) == sorted(paths[1:])


def test_analyze_download_fetches_them_first(library_db, monkeypatch):
    db, paths = library_db
    reads = _cloud(monkeypatch, paths[:2], stuck=[paths[1]])
    res = _analyze(db, "--download")
    assert res.exit_code == 0, res.output
    assert "2 cloud-only file(s): downloading them first" in res.output
    assert "1 file(s) skipped: they couldn't be downloaded" in res.output
    assert sorted(reads) == sorted(paths[:2])
    assert _quality_done(db) == sorted([paths[0], paths[2]])


# --- doctor ---------------------------------------------------------------------------------
def test_doctor_warns_about_cloud_only_library_files(tmp_path, monkeypatch):
    from fourier.cli import main
    from fourier.cli.setup import check_cloud_only
    lib = tmp_path / "Samples"
    for rel in ("A/kick.wav", "A/snare.aif", "B/C/pad.wav", "B/notes.txt", ".hidden/x.wav"):
        (lib / rel).parent.mkdir(parents=True, exist_ok=True)
        (lib / rel).write_bytes(b"x")
    _cloud(monkeypatch, [])
    assert check_cloud_only([lib]) == [("OK", "cloud-only files", "none of 3 audio files")]
    _cloud(monkeypatch, [lib / "A/kick.wav", lib / "B/C/pad.wav", lib / "B/notes.txt"])
    (level, what, detail), = check_cloud_only([lib])
    assert level == "WARN" and detail.startswith("2 of 3 audio files aren't on this machine")
    assert '"Keep Downloaded"' in detail
    (_l, _w, detail), = check_cloud_only([lib], cap=2)
    assert "(the first 2 checked)" in detail
    assert check_cloud_only([tmp_path / "gone"]) == []
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(f'library = ["{lib}"]\ndevices = ["generic_48k"]\n')
    monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    monkeypatch.delenv("FOURIER_LIBRARY", raising=False)
    places.reset()
    try:
        res = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "doctor"])
    finally:
        places.reset()
    assert "WARN  cloud-only files: 2 of 3 audio files" in res.output, res.output
