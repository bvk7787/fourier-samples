"""The analysis steps (cli/enrich.py) when they can't finish: Ctrl-C stops a step at once and
keeps what it finished, a worker process that dies ends the step with one line, a file that
isn't there or can't be read is never stored as read, and the worker counts follow the
machine or -j."""
import os
import signal
import time

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import text

from fourier.cli import enrich
from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures


@pytest.fixture
def db(tmp_path):
    from fourier.metadata.rows import forget_current
    old = (S._engine, S._SessionLocal)
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "t.db")
    forget_current()
    yield tmp_path
    forget_current()
    if S._engine is not None:
        S._engine.dispose()
    S._engine, S._SessionLocal = old


def _files(tmp_path):
    """A readable tone, a file that isn't audio, and a path with no file."""
    lib = tmp_path / "SampleLibrary" / "Acme"
    lib.mkdir(parents=True)
    tone = lib / "Tone C.wav"
    sr = 22050
    t = np.arange(int(sr * 0.6)) / sr
    sf.write(tone, 0.5 * np.sin(2 * np.pi * 261.63 * t), sr)
    bad = lib / "Broken.wav"
    bad.write_bytes(b"RIFF\x00\x00\x00\x00not audio at all" * 4)
    return {1: str(tone), 2: str(bad), 3: str(lib / "Gone.wav")}


def _samples(paths, features=True):
    with S.session_scope() as s:
        for i, p in paths.items():
            s.add(Sample(id=i, path=p, rel_path=os.path.relpath(p, os.path.dirname(os.path.dirname(p))),
                         filename=os.path.basename(p), duration_s=0.6))
            if features:
                s.add(SampleFeatures(sample_id=i))


def _col(col):
    with S.session_scope() as s:
        return dict(s.execute(text(f"SELECT sample_id, {col} FROM sample_features")).all())


@pytest.fixture
def sigint():
    """Python's own Ctrl-C handler while the test runs (a test worker may ignore SIGINT)."""
    old = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, old)


def test_ctrl_c_stops_a_step_at_once_and_keeps_what_finished(db, monkeypatch, sigint, capsys):
    paths = {i: f"/l/SampleLibrary/Acme/Kick {i:02d}.wav" for i in range(1, 41)}
    _samples(paths, features=False)
    calls = []

    def slow(path):
        calls.append(path)
        if len(calls) == 8:                       # Ctrl-C while the step runs
            os.kill(os.getpid(), signal.SIGINT)
        time.sleep(0.2)
        return {"mfcc_mean": [0.0] * 13, "tempo_bpm": 120.0}
    monkeypatch.setattr("fourier.analysis.librosa_features.compute_features", slow)
    with pytest.raises(KeyboardInterrupt):
        enrich.enrich_librosa.callback(limit=0, workers=2, force=False, category_filter=None)
    time.sleep(0.3)                               # what was running when it stopped
    assert len(calls) < 20                        # the queued files never started
    kept = [i for i, v in _col("mfcc_mean").items() if v is not None]
    assert 1 <= len(kept) <= len(calls)           # and what finished is in the database
    out = " ".join(capsys.readouterr().out.split())
    assert "finished sample(s) saved" in out and "next `fourier build` carries on" in out
    assert "--resume" not in out


def test_ctrl_c_during_a_batchs_commit_still_saves_it(db, capsys):
    """Ctrl-C that lands inside a batch's commit: the session is rolled back and the batch
    stored again, with the usual line."""
    from sqlalchemy import event

    from fourier.db.session import get_session
    _samples({1: "/l/SampleLibrary/Acme/Kick 01.wav"}, features=False)
    session = get_session()
    cut: list = []

    def once(_session, _ctx):
        if not cut:
            cut.append(1)
            raise KeyboardInterrupt
    event.listen(session, "after_flush", once)

    def store(rows):
        session.add_all(SampleFeatures(sample_id=sid) for sid in rows)
        session.commit()
    batch = enrich._Batch(store, 1, session)
    try:
        with pytest.raises(KeyboardInterrupt):
            with enrich._keep_on_stop(batch.save, "own"):
                batch.add(1)
    finally:
        session.close()
    assert cut and 1 in _col("sample_id")
    assert "own: stopped; 1 finished sample(s) saved" in " ".join(capsys.readouterr().out.split())


def _sleep_then(x):
    time.sleep(60 if x >= 3 else 0)               # the first three come back at once
    return x


def _die_on_3(x):
    if x == 3:
        os._exit(1)                               # killed, as the system does when out of memory
    time.sleep(0.05)
    return x


def test_a_stop_ends_the_worker_processes():
    import multiprocessing
    t0 = time.monotonic()
    got = []
    with pytest.raises(KeyboardInterrupt):
        for x in enrich._results(_sleep_then, list(range(40)), 2, processes=True):
            got.append(x)
            if len(got) == 3:
                raise KeyboardInterrupt
    deadline = time.monotonic() + 5
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not multiprocessing.active_children()   # ended, not left to finish their minute
    assert time.monotonic() - t0 < 30


def test_a_worker_that_dies_ends_the_step_with_one_line(capsys):
    saved = []
    with pytest.raises(SystemExit) as e:
        with enrich._keep_on_stop(lambda: saved.append(1) or 2, "events"):
            for _x in enrich._results(_die_on_3, list(range(8)), 2, processes=True):
                pass
    assert e.value.code == 1 and saved
    out = capsys.readouterr().out
    assert "events: a worker process stopped" in out and "2 finished sample(s) saved" in out
    assert "--workers 1" in out and "Traceback" not in out


def test_an_error_keeps_what_finished_too():
    saved = []
    with pytest.raises(ValueError):
        with enrich._keep_on_stop(lambda: saved.append(1) or 1, "quality"):
            raise ValueError("a file the step couldn't handle")
    assert saved


def test_one_worker_runs_in_this_process(monkeypatch):
    """One worker maps in this process: no pool, so no worker can die under it."""
    import concurrent.futures as cf

    def no_pool(*a, **k):
        raise AssertionError("a pool was started for one worker")
    monkeypatch.setattr(cf, "ProcessPoolExecutor", no_pool)
    monkeypatch.setattr(cf, "ThreadPoolExecutor", no_pool)
    assert list(enrich._results(lambda x: x * 2, [1, 2, 3], 1, processes=True)) == [2, 4, 6]
    assert list(enrich._results(str, [7], 0)) == ["7"]


def test_events_store_nothing_for_a_file_that_isnt_there(db, capsys):
    paths = _files(db)
    _samples(paths)
    enrich.enrich_events.callback(workers=1, force=False, limit=0, max_dur=60.0)
    done = _col("events_computed_at")
    assert done[1] is not None and _col("n_events")[1] is not None     # read
    assert done[2] is not None and _col("n_events")[2] is None         # unreadable: done, no profile
    assert done[3] is None                                             # not there: read next time
    assert "1 file(s) weren't there" in capsys.readouterr().out


def test_keys_store_nothing_for_a_file_that_cant_be_read(db, capsys):
    paths = _files(db)
    _samples(paths)
    enrich._read_keys(sorted(paths.items()), 1)
    conf = _col("key_confidence")
    assert conf[1] is not None                     # read (a key, or unpitched)
    assert conf[2] is None and conf[3] is None     # never "unpitched": read again next time
    out = capsys.readouterr().out
    assert "1 unreadable" in out and "1 file(s) weren't there" in out


def test_loop_trims_store_nothing_for_a_file_that_cant_be_read(db):
    paths = _files(db)
    _samples(paths)
    enrich.enrich_loop_trim.callback(workers=1, force=False, limit=0, threshold_db=-60.0,
                                     min_silence_s=0.05, all_classes=True)
    done = _col("trim_computed_at")
    assert done[1] is not None and done[2] is None and done[3] is None


def test_own_roots_store_nothing_for_a_file_that_cant_be_read(db):
    paths = _files(db)
    assert enrich._root_job((1, paths[1]))[2] is True
    assert enrich._root_job((2, paths[2]))[2] is False
    assert enrich._root_job((3, paths[3]))[2] is False


def test_worker_counts_follow_the_machine(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert enrich._workers(None, 6) == 2 and enrich._workers(None, 4) == 2
    assert enrich._workers(3, 6) == 3                               # --workers as given
    assert enrich._workers(None, 6, jobs=1) == 2                    # -j never lowers it
    assert enrich._workers(None, 6, jobs=3) == 3 and enrich._workers(None, 4, jobs=8) == 4
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert enrich._workers(None, enrich.WORKER_CAPS["events"]) == 6
    assert enrich._workers(None, 6, jobs=15) == 6 and enrich._workers(32, 6, jobs=15) == 32
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert enrich._workers(None, 4) == 1


def test_the_process_steps_start_one_worker_per_cpu(db, monkeypatch):
    from click.testing import CliRunner

    from fourier.cli import main
    paths = _files(db)
    _samples(paths)
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    r = CliRunner().invoke(main, ["--db", str(db / "t.db"), "tools", "analyze", "--only", "events"])
    assert r.exit_code == 0, r.output
    assert "Event profiles: 3 samples (1 workers)" in r.output


def _no_counts(monkeypatch):
    """No database behind the analysis's sample counts."""
    import fourier.cli.setup as setup
    for name in ("pending_analysis", "pending_estimate", "analysis_counts"):
        monkeypatch.setattr(setup, name, lambda *a, **k: {}, raising=False)


class _Ctx:
    """Records the workers each analysis step is invoked with."""
    def __init__(self):
        self.meta: dict = {}
        self.got: dict = {}

    def invoke(self, cmd, **kw):
        self.got[cmd.name] = kw.get("workers")


CAPPED = ("librosa", "events", "quality", "loop-trim", "own", "key")


@pytest.mark.parametrize("cpus, jobs, want", [
    (16, 15, {"librosa": 4, "quality": 4, "loop-trim": 4, "key": 4, "events": 6, "own": 6}),
    (2, 1, dict.fromkeys(CAPPED, 2)),
    (2, 3, {"librosa": 3, "quality": 3, "loop-trim": 3, "key": 3, "events": 3, "own": 3}),
])
def test_setups_build_runs_the_analysis_on_capped_workers(monkeypatch, cpus, jobs, want):
    """Setup starts the first build with -j <CPUs - 1>, a category worker count. The analysis
    steps keep one worker per CPU (two on two CPUs) and never more than their own caps (six
    pYIN processes on sixteen CPUs); the CLAP step keeps its own."""
    import fourier.cli.ingest as ingest
    from fourier.cli import build
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(enrich, "clap_extra_missing", lambda: False)
    monkeypatch.setattr(ingest, "run_scan", lambda **kw: None)
    monkeypatch.setattr(ingest, "UNAVAILABLE", [])
    _no_counts(monkeypatch)
    ctx = _Ctx()
    build._scan_and_analyze(ctx, None, False, jobs)
    assert ctx.got["clap"] is None
    assert {s: ctx.got[s] for s in CAPPED} == want


def test_tools_analyze_workers_reach_every_step(monkeypatch):
    _no_counts(monkeypatch)
    ctx = _Ctx()
    enrich.run_analysis(ctx, workers=3)
    assert set(ctx.got.values()) - {None} == {3} and ctx.got["clap"] == 3


@pytest.mark.parametrize("args, want", [([], 1), (["-j", "3"], 3)])
def test_a_build_hands_its_jobs_to_the_analysis(tmp_path, monkeypatch, args, want):
    from click.testing import CliRunner

    from fourier.cli import build, main
    lib = tmp_path / "SampleLibrary" / "Acme"
    lib.mkdir(parents=True)
    (lib / "Kick 01.wav").write_bytes(b"")
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FOURIER_LIBRARY", str(tmp_path / "SampleLibrary"))
    got = []

    def stop(ctx, out_dir=None, allow_missing=False, jobs=None):
        got.append(jobs)
        raise SystemExit(0)
    monkeypatch.setattr(build, "_scan_and_analyze", stop)
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), "build", "--all",
                                  "--out", str(tmp_path / "out"), *args])
    assert got == [want], r.output
