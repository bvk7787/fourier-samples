"""A Ctrl-C ends with what it left and the command that carries on, never a bare "Aborted!"."""
import pytest
from click.testing import CliRunner

from fourier.cli import enrich as E
from fourier.cli import ingest as I
from fourier.cli import main


def _stop(*_a, **_k):
    raise KeyboardInterrupt


@pytest.fixture
def library(monkeypatch, tmp_path):
    from fourier import places
    lib = tmp_path / "Samples"
    (lib / "Acme").mkdir(parents=True)
    monkeypatch.setenv("FOURIER_LIBRARY", str(lib))
    monkeypatch.setattr(E, "clap_extra_missing", lambda: False)
    places.reset()
    yield lib
    places.reset()


def test_a_stop_during_the_analysis_says_the_master_is_unchanged(library, monkeypatch, tmp_path):
    monkeypatch.setattr(I, "run_scan", lambda *a, **k: None)
    monkeypatch.setattr(E, "run_analysis", _stop)
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "build", "--all",
                                  "--out", str(tmp_path / "m")])
    out = " ".join(r.output.split())
    assert r.exit_code == 130, r.output
    assert "Aborted!" not in out
    assert f"Stopped (Ctrl-C). Nothing in {tmp_path / 'm'} changed" in out
    assert "`fourier build` carries on from there" in out


def test_a_stop_during_a_render_says_the_previous_render_stays(monkeypatch, tmp_path):
    from fourier.cli import releases as R
    from fourier.packs import render as PR
    monkeypatch.setattr(PR, "render_device", _stop)
    monkeypatch.setattr(R, "_stop_on_pending_swap", lambda d: None)
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "render", "m8_tracker",
                                  "--from", str(tmp_path / "master"), "--out", str(tmp_path / "r")])
    out = " ".join(r.output.split())
    assert r.exit_code == 130 and "Aborted!" not in out, r.output
    assert "The previous render in" in out and "`fourier render m8_tracker` renders again" in out


def test_a_stop_during_a_scan_says_to_scan_again(monkeypatch, tmp_path):
    monkeypatch.setattr(I, "run_scan", _stop)
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "tools", "scan"])
    out = " ".join(r.output.split())
    assert r.exit_code == 130 and "Aborted!" not in out, r.output
    assert "Stopped (Ctrl-C). The scan stopped part way" in out
