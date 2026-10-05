"""Measured times and estimates (fourier/timings.py)."""
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
