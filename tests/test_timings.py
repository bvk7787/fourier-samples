"""Measured times and estimates (fourier/timings.py)."""
import pytest

from fourier import timings as T


def test_estimates_from_defaults_then_from_this_machine(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path))
    secs, basis = T.estimate_build({"KICKS": 100, "PADS": 100}, jobs=2)
    assert "rough defaults" in basis
    # a small library's first build: the start-up any build spends, however few its samples
    assert T.estimate_build({"KICKS": 6}, jobs=4, samples=45)[0] >= T.DEFAULTS["build_fixed_min_s"]
    r = T.DEFAULTS["build_s_per_file"]
    work = [100 * r["KICKS"], 100 * r["PADS"]]
    assert secs == max(max(work), sum(work) / 2) + T.DEFAULTS["build_fixed_s"]
    res = [("KICKS", {"seconds": 10.0, "files": 100, "cache_hits": 90}, []),
           ("PADS", {"seconds": 40.0, "files": 100, "cache_hits": 0}, [])]
    T.record_build(res, jobs=2, wall_s=60.0)
    secs, basis = T.estimate_build({"KICKS": 200, "PADS": 100}, jobs=2)
    assert "last build" in basis
    fixed = 60.0 - max(40.0, 50.0 / 2)                     # beyond the longest category
    assert secs == max(40.0, (20.0 + 40.0) / 2) + fixed
    T.record_analyze("librosa", 100.0, 1000, workers=4)
    secs, basis = T.estimate_analyze({"librosa": 500}, workers=4)
    assert secs == T.DEFAULTS["analyze_fixed_s"] + 500 * 0.4 / 4 and "this machine" in basis
    assert T.estimate_analyze({"librosa": 0}, workers=4)[0] == 0          # nothing to do: no start-up
    assert T.human(120) == "about 2 min" and T.human(3 * 3600) == "about 3.0 h"


def test_clap_on_the_cpu_has_its_own_rate(tmp_path, monkeypatch):
    """Where PyTorch has only the CPU, CLAP is several times slower a file than on a graphics
    chip, and the estimate says which rates it used."""
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path))
    monkeypatch.setattr(T, "clap_on_cpu", lambda: False)
    gpu, basis = T.estimate_analyze({"clap": 1000}, workers=1)
    assert gpu == T.DEFAULTS["analyze_fixed_s"] + 1000 * T.DEFAULTS["analyze_s_per_sample"]["clap"]
    assert basis == T.DEFAULTS_SOURCE
    monkeypatch.setattr(T, "clap_on_cpu", lambda: True)
    cpu, basis = T.estimate_analyze({"clap": 1000}, workers=1)
    assert cpu == T.DEFAULTS["analyze_fixed_s"] + 1000 * T.DEFAULTS["clap_cpu_s_per_sample"]
    assert T.DEFAULTS["clap_cpu_s_per_sample"] >= 0.2 and "CPU" in basis


def test_the_device_is_read_without_loading_pytorch(monkeypatch):
    import sys
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(T.sys, "platform", "darwin")
    import platform
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert T.clap_on_cpu() is False                        # Apple Silicon: MPS
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert T.clap_on_cpu() is True                         # an Intel Mac
    monkeypatch.setattr(T.sys, "platform", "linux")
    monkeypatch.setattr(T.shutil, "which", lambda name: None)
    assert T.clap_on_cpu() is True                         # no NVIDIA GPU
    assert "torch" not in sys.modules


def test_workers_count_up_to_the_cpus(tmp_path, monkeypatch):
    """A step's workers past the computer's CPUs don't make it faster; CLAP reads one file at
    a time whatever its threads. A rate measured with more workers than CPUs reads back the
    same."""
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path))
    monkeypatch.setattr(T, "clap_on_cpu", lambda: False)
    monkeypatch.setattr(T.os, "cpu_count", lambda: 2)
    rate = T.DEFAULTS["analyze_s_per_sample"]
    secs, _ = T.estimate_analyze({"events": 600, "clap": 600}, workers={"events": 6, "clap": 4})
    assert secs == T.DEFAULTS["analyze_fixed_s"] + 600 * rate["events"] / 2 + 600 * rate["clap"]
    T.record_analyze("events", 100.0, 400, workers=6)      # 6 workers on 2 CPUs
    secs, basis = T.estimate_analyze({"events": 400}, workers={"events": 6})
    assert secs == T.DEFAULTS["analyze_fixed_s"] + 100.0 and "this machine" in basis


def test_a_first_analysis_counts_every_step(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path))
    steps = T.first_analysis(1000)
    assert set(steps) >= {"derived", "librosa", "clap", "events", "quality", "loop-trim", "own", "key"}
    assert steps["events"] == steps["quality"] == steps["clap"] == 1000
    assert steps["own"] == steps["key"] == steps["loop-trim"] == 300


def test_setup_estimate_on_a_cpu_only_computer(tmp_path, monkeypatch):
    """10,000 new files where CLAP runs on the CPU: at least the CLAP time alone, with every
    other step on top."""
    from fourier.cli import setup as S
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path))
    monkeypatch.setattr(S, "_unreadable_files", lambda: {})
    monkeypatch.setattr(T, "clap_on_cpu", lambda: True, raising=False)
    secs, new = S.setup_estimate(10_000)
    assert new == 10_000
    assert secs > 10_000 * 0.25 + 10_000 * T.DEFAULTS["analyze_s_per_sample"]["events"] / 6


def test_a_build_estimates_every_step_it_will_run(tmp_path, monkeypatch):
    """New samples count for every step a build runs: events and quality read each one, and
    own, key and loop-trim their share of the ones librosa hasn't read yet."""
    from datetime import datetime

    from fourier.cli import setup as S
    from fourier.db import session as D
    from fourier.db.models import Sample, SampleFeatures
    from fourier.metadata import rows as R
    old = (D._engine, D._SessionLocal)
    D._engine = D._SessionLocal = None
    R.forget_current()
    D.init_db(tmp_path / "t.db")
    try:
        with D.session_scope() as s:
            for i in range(1, 21):
                s.add(Sample(id=i, path=f"/l/Acme/{i}.wav", rel_path=f"Acme/{i}.wav",
                             filename=f"{i}.wav", duration_s=1.0))
            for i in range(1, 11):                        # half analyzed by every step already
                s.add(SampleFeatures(sample_id=i, mfcc_mean=[0.0], derived_computed_at=datetime.now(),
                                     clap_embedding=b"x", events_computed_at=datetime.now(),
                                     is_clipped=0, trim_computed_at=datetime.now()))
        with D.session_scope() as s:
            counted = S.pending_analysis(s)
            assert counted["librosa"] == counted["clap"] == counted["events"] == counted["quality"] == 10
            est = S.pending_estimate(s)
            assert est["events"] == est["quality"] == 10
            assert est["own"] == est["key"] == est["loop-trim"] == 3
            assert S.analysis_counts(s)["loop-trim"] == 10
    finally:
        D._engine, D._SessionLocal = old
        R.forget_current()


@pytest.mark.parametrize("cpus, jobs", [(2, None), (2, 1), (8, 7), (16, 15), (2, 5)])
def test_the_estimate_assumes_the_workers_the_steps_run_with(monkeypatch, cpus, jobs):
    """step_workers(jobs) gives each step the workers run_analysis starts it with for a build
    at -j jobs, so the estimate and the run agree (on two CPUs, two workers a step)."""
    import os

    import fourier.cli.setup as setup
    from fourier.cli import enrich as E
    monkeypatch.setattr(os, "cpu_count", lambda: cpus)
    monkeypatch.setattr(setup, "analysis_counts", lambda: {})
    ran: dict = {}

    class Ctx:
        meta: dict = {}

        def invoke(self, cmd, **kw):
            ran[cmd.name] = kw.get("workers")
    E.run_analysis(Ctx(), jobs=jobs)
    planned = E.step_workers(jobs)
    assert {s: ran[s] for s in E.WORKER_CAPS} == {s: planned[s] for s in E.WORKER_CAPS}
    assert all(planned[s] == min(cap, max(cpus, jobs or 0)) for s, cap in E.WORKER_CAPS.items())
    clap = next(p for p in E.steps.commands["clap"].params if p.name == "workers")
    assert ran["clap"] is None and planned["clap"] == clap.default


def test_a_builds_estimate_uses_its_jobs(monkeypatch):
    """`fourier build -j N` estimates the analysis with the workers -j N runs."""
    import os

    import fourier.cli.ingest as ingest
    import fourier.cli.setup as setup
    from fourier.cli import build
    from fourier.cli import enrich as E
    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    monkeypatch.setattr(E, "clap_extra_missing", lambda: False)
    monkeypatch.setattr(E, "run_analysis", lambda ctx, **kw: None)
    monkeypatch.setattr(ingest, "run_scan", lambda **kw: None)
    monkeypatch.setattr(ingest, "UNAVAILABLE", [])
    monkeypatch.setattr(setup, "pending_estimate", lambda: {"librosa": 5000, "events": 5000})
    seen = []
    monkeypatch.setattr(T, "estimate_analyze", lambda pending, workers: seen.append(workers) or (60.0, "x"))
    build._scan_and_analyze(object(), None, False, 5)
    assert seen == [E.step_workers(5)] and seen[0]["librosa"] == 4 and seen[0]["events"] == 5


def test_setup_estimates_the_build_it_starts(tmp_path, monkeypatch):
    """Setup's estimate is for the command it runs: `fourier build --all -j <CPUs - 1>`."""
    import os

    from fourier.cli import enrich as E
    from fourier.cli import setup as S
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path))
    monkeypatch.setattr(S, "_unreadable_files", lambda: {})
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert S.setup_jobs() == 7
    seen: dict = {}
    monkeypatch.setattr(T, "estimate_build", lambda files, jobs, samples=0: seen.update(build=jobs) or (0.0, ""))
    monkeypatch.setattr(T, "estimate_analyze", lambda p, workers: seen.update(analyze=workers) or (0.0, ""))
    S.setup_estimate(10_000, jobs=S.setup_jobs())
    assert seen == {"build": 7, "analyze": E.step_workers(7)}
