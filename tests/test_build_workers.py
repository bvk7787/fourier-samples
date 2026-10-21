"""Build workers (packs/curate.py build_all_parallel): the database shared read-only, the CLAP
text prompts embedded once in the parent, every start method."""
import ast
import multiprocessing
from pathlib import Path

import numpy as np
import pytest

from fourier.analysis import clap_features as CF
from fourier.db import session as S
from fourier.demo import fake_embed_text
from fourier.packs import curate

ROOT = Path(__file__).resolve().parents[1]
METHODS = [m for m in ("fork", "spawn", "forkserver") if m in multiprocessing.get_all_start_methods()]


def _not_in_a_worker(text):
    """The stand-in text encoder, which a worker process must never reach (a forked worker
    inherits this; a spawned one has the real encoder, which the test keeps from loading)."""
    if multiprocessing.parent_process() is not None:
        raise RuntimeError("the CLAP text encoder ran in a worker process")
    return fake_embed_text(text)


def test_a_worker_reads_the_vectors_its_parent_embedded(monkeypatch):
    vec = np.arange(4, dtype=np.float32)
    monkeypatch.setattr(curate, "_TEXT_VECS", {"acme kick": vec})
    monkeypatch.setattr(curate, "_TEXT_MISSES", set())
    monkeypatch.setattr(CF, "embed_text", lambda t: np.zeros(4, dtype=np.float32))
    got = curate.embed_text("acme kick")
    assert np.array_equal(got, vec) and got is not vec
    assert not curate.embed_text("qxvwz pad").any()            # not prepared: the encoder, noted
    assert curate._TEXT_MISSES == {"qxvwz pad"}


def test_the_parent_embeds_every_prompt_a_category_build_uses(monkeypatch):
    """Each embed_text call in curation embeds prompts text_prompts() names, so a worker
    finds them all. A new call site fails here until text_prompts() covers it."""
    known = {"ph", "cfg['phrases']", "anti", "ps", "FX_BAND_PHRASES[b]", "cfg['anti']",
             "cfg['exclude_phrases']", "cfg['exclude_anti']", "cfg['clap_anti']", "phrases"}
    tree = ast.parse((ROOT / "src" / "fourier" / "packs" / "curate.py").read_text())
    sources = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)) and isinstance(node.elt, ast.Call) \
                and getattr(node.elt.func, "id", None) == "embed_text":
            sources.add(ast.unparse(node.generators[0].iter))
    assert sources == known
    prompts = set(curate.text_prompts())
    from fourier.packs.curate_config import CATEGORIES, FX_BAND_PHRASES, NAME_PHRASES, PIANO_ANCHOR, PIANO_ANTI
    for cfg in CATEGORIES.values():
        for key in ("phrases", "anti", "exclude_phrases", "exclude_anti", "clap_anti"):
            assert set(cfg.get(key) or ()) <= prompts
    for group in (*NAME_PHRASES.values(), *FX_BAND_PHRASES.values(), PIANO_ANCHOR, PIANO_ANTI):
        assert set(group) <= prompts
    monkeypatch.setattr(CF, "embed_text", fake_embed_text)
    assert set(curate._prompt_vectors()) == prompts


def _library(tmp_path, monkeypatch, suffix):
    from fourier import synthlib
    from fourier.demo import fill_db
    lib = tmp_path / "lib" / "SampleLibrary"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("FOURIER_HOME", str(home))
    monkeypatch.setenv("FOURIER_LIBRARY", str(lib))
    files = synthlib.generate(lib)
    S._engine = S._SessionLocal = None
    db = home / f"library{suffix}"
    fill_db(db, lib, files)
    return db


def _count_samples(_):
    from sqlalchemy import text
    with S.session_scope() as s:
        return s.execute(text("SELECT count(*) FROM samples")).scalar()


@pytest.mark.parametrize("suffix", [".db", ".duckdb"])
def test_a_worker_that_starts_fresh_reads_the_database_beside_its_parent(tmp_path, suffix):
    """A spawned worker opens the file itself. On DuckDB that fails while another process has
    it open to write, so the parent reads it read-only while the workers run."""
    from sqlalchemy import text

    from fourier.db.models import Sample
    from fourier.platforms import process_pool
    db = tmp_path / f"t{suffix}"
    S._engine = S._SessionLocal = None
    S.init_db(db)
    try:
        with S.session_scope() as s:
            s.add(Sample(id=1, path="/l/Acme/Kick 1.wav", rel_path="Acme/Kick 1.wav", filename="Kick 1.wav"))
        was = multiprocessing.get_start_method(allow_none=True)
        multiprocessing.set_start_method("spawn", force=True)
        try:
            with S.shared_with_workers(db), process_pool(1, S.open_in_worker, (str(db),)) as ex:
                assert list(ex.map(_count_samples, [0])) == [1]
                assert _count_samples(0) == 1                 # the parent reads too
        finally:
            multiprocessing.set_start_method(was, force=True)
        with S.session_scope() as s:                          # and writes again afterwards
            s.execute(text("UPDATE samples SET is_favorite = true WHERE id = 1"))
        assert not S.is_read_only()
    finally:
        if S._engine is not None:
            S._engine.dispose()
        S._engine = S._SessionLocal = None


@pytest.mark.slow
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("suffix", [".db", ".duckdb"])
def test_workers_build_under_every_start_method(tmp_path, monkeypatch, method, suffix):
    """Two categories on two workers: the workers read the database beside the parent (on
    DuckDB a second process can't open a file another one writes) and never run the CLAP
    text encoder; afterwards the parent writes again."""
    db = _library(tmp_path, monkeypatch, suffix)
    monkeypatch.setattr(CF, "embed_text", _not_in_a_worker)
    # a worker that starts fresh has the real encoder: make sure it can't load a model either
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("FOURIER_CLAP_TEXT_CACHE", str(tmp_path / "clap_text"))
    was = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method(method, force=True)
    out = tmp_path / "out.next"
    out.mkdir()
    lines: list[str] = []
    try:
        res = curate.build_all_parallel(
            str(db), ["KICKS", "SNARES"], str(out), 2,
            dict(per_family=25, clap_z=1.0, transcode=False, describe=False, loudness=False),
            log=lambda m: lines.append(str(m)))
    finally:
        multiprocessing.set_start_method(was, force=True)
        if method == "forkserver":        # its server keeps this test's environment: stop it
            from multiprocessing import forkserver
            getattr(forkserver._forkserver, "_stop", lambda: None)()
    log = "\n".join(lines)
    assert sorted(r[0] for r in res) == ["KICKS", "SNARES"], log
    assert "FAILED" not in log and "embedded in a worker" not in log, log
    from sqlalchemy import text
    with S.session_scope() as s:                 # read-write again
        s.execute(text("UPDATE samples SET is_favorite = true WHERE id = 1"))
    S._engine = S._SessionLocal = None


def _a_minute(x):
    import time
    time.sleep(60 if x else 0)
    return x


def test_a_pool_left_by_an_exception_ends_its_workers():
    """Ctrl-C while categories build: the workers are ended, not waited for."""
    import time

    from fourier.platforms import process_pool
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        with process_pool(2) as ex:
            futs = [ex.submit(_a_minute, x) for x in range(6)]
            assert futs[0].result() == 0
            raise KeyboardInterrupt
    deadline = time.monotonic() + 10
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not multiprocessing.active_children() and time.monotonic() - t0 < 30


def _unsafe_or_a_minute(x):
    import time

    from fourier.safety import UnsafePath
    if x == 0:
        return UnsafePath("Acme: cloud-only sources")
    time.sleep(60)
    return None


def _ended(procs, within=10):
    import time
    deadline = time.monotonic() + within
    while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not any(p.is_alive() for p in procs)


def test_an_unsafe_category_ends_the_running_ones():
    """A category stopped by cloud-only sources stops the build: the categories running
    beside it are ended, not waited for (each would run a minute)."""
    from concurrent.futures import as_completed

    from fourier.platforms import process_pool
    from fourier.safety import UnsafePath
    procs = []
    try:
        with pytest.raises(UnsafePath):
            with process_pool(2) as ex:
                futs = [ex.submit(_unsafe_or_a_minute, x) for x in range(4)]
                for fut in as_completed(futs):
                    procs = list(ex._processes.values())
                    curate._stop_on_unsafe(ex, fut.result())
        assert procs and _ended(procs)
    finally:
        for p in procs:
            p.kill()


def test_a_pool_shut_down_early_still_ends_its_workers():
    """A caller that shut the pool down itself before leaving it by an exception: the
    workers are still ended."""
    from fourier.platforms import process_pool
    procs = []
    try:
        with pytest.raises(KeyboardInterrupt):
            with process_pool(2) as ex:
                futs = [ex.submit(_a_minute, x) for x in range(4)]
                assert futs[0].result() == 0
                procs = list(ex._processes.values())
                ex.shutdown(wait=False, cancel_futures=True)
                raise KeyboardInterrupt
        assert procs and _ended(procs)
    finally:
        for p in procs:
            p.kill()


def test_a_database_the_workers_cant_read_stops_with_one_line(tmp_path, monkeypatch):
    """When the database won't open read-only for the workers, the parent says so before any
    worker starts (each would fail in its initializer)."""
    db = tmp_path / "t.duckdb"
    S._engine = S._SessionLocal = None
    S.init_db(db)
    monkeypatch.setattr(S, "read_only", lambda: False)
    try:
        with pytest.raises(RuntimeError, match="run with -j 1"):
            with S.shared_with_workers(db):
                pass
    finally:
        if S._engine is not None:
            S._engine.dispose()
        S._engine = S._SessionLocal = None


def test_a_mac_never_forks_its_workers(monkeypatch):
    """A forked worker on macOS can deadlock on a lock the parent's threads held (an open DuckDB
    database, system frameworks): the pools spawn there even when fork is the default."""
    from fourier import platforms
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("no fork here")
    was = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("fork", force=True)
    try:
        monkeypatch.setattr(platforms.sys, "platform", "darwin")
        assert platforms.pool_context().get_start_method() == "spawn"
        monkeypatch.setattr(platforms.sys, "platform", "linux")
        assert platforms.pool_context().get_start_method() == "fork"
    finally:
        multiprocessing.set_start_method(was, force=True)
