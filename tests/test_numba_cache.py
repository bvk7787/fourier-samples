"""The suite's numba cache (conftest.py): one per test run, so two runs on one machine never
share one (processes compiling the same function into one cache can crash librosa)."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _configure(**option):
    """conftest.pytest_configure as a run's first process calls it (no pytest-xdist worker)."""
    config = SimpleNamespace(option=SimpleNamespace(**{"dist": "no", **option}))
    sys.modules["conftest"].pytest_configure(config)
    return os.environ["NUMBA_CACHE_DIR"]


def test_two_runs_never_share_a_numba_cache(monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    first = _configure()
    monkeypatch.delenv("NUMBA_CACHE_DIR")
    second = _configure()
    assert first != second and Path(first).parent != Path(second).parent
    assert Path(first).parent.is_dir() and Path(second).parent.is_dir()


def test_a_runs_workers_share_its_folder_each_in_a_subfolder(monkeypatch):
    import conftest
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    run = _configure(dist="loadgroup")                  # the controller: the run's folder
    worker = SimpleNamespace(option=SimpleNamespace(dist="loadgroup"),
                             workerinput={"fourier_dist": "loadgroup"})
    conftest.pytest_configure(worker)                   # a worker inherits it
    assert os.environ["NUMBA_CACHE_DIR"] == str(Path(run) / conftest.WORKER)


def test_a_numba_cache_dir_you_set_is_kept(monkeypatch, tmp_path):
    import conftest
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "mine"))
    assert _configure() == str(tmp_path / "mine" / conftest.WORKER)
