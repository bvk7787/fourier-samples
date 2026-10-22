"""$FOURIER_HOME naming a folder that isn't there: a new home is said once, and a home on a
volume that isn't mounted stops before anything is written."""
from pathlib import Path

import pytest
from click.testing import CliRunner

from fourier import platforms
from fourier.cli import ingest as I
from fourier.cli import main


@pytest.fixture
def volumes(monkeypatch, tmp_path):
    """tmp_path/Volumes stands in for the folder drives are mounted under."""
    root = tmp_path / "Volumes"
    root.mkdir()
    monkeypatch.setattr(platforms, "_volume_roots", lambda: [root], raising=False)
    return root


def _scan(monkeypatch, tmp_path, home: Path):
    ran = []
    monkeypatch.setattr(I, "run_scan", lambda *a, **k: ran.append(1))
    monkeypatch.setenv("FOURIER_HOME", str(home))
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "tools", "scan"])
    return r, ran


def test_a_home_on_a_drive_that_isnt_plugged_in_stops(volumes, monkeypatch, tmp_path):
    home = volumes / "Drive" / "fourier"
    r, ran = _scan(monkeypatch, tmp_path, home)
    out = " ".join(r.output.split())
    assert r.exit_code == 1, r.output
    assert not ran and not (volumes / "Drive").exists()          # nothing written
    assert f"FOURIER_HOME is {home}, on {volumes / 'Drive'}, which isn't mounted" in out, out


def test_a_mount_point_left_behind_stops_too(volumes, monkeypatch, tmp_path):
    (volumes / "Drive").mkdir()                                  # empty, and no mount point
    r, ran = _scan(monkeypatch, tmp_path, volumes / "Drive" / "fourier")
    assert r.exit_code == 1 and not ran, r.output
    assert list((volumes / "Drive").iterdir()) == []


def test_a_drive_that_is_there_takes_a_new_home(volumes, monkeypatch, tmp_path):
    (volumes / "Drive" / "Samples").mkdir(parents=True)          # a drive with files on it
    home = volumes / "Drive" / "fourier"
    r, ran = _scan(monkeypatch, tmp_path, home)
    assert r.exit_code == 0 and ran, r.output
    assert "this is a new, empty Fourier home" in " ".join(r.output.split())


def test_a_new_home_is_said_once(monkeypatch, tmp_path):
    home = tmp_path / "elsewhere" / "fourier"
    r, ran = _scan(monkeypatch, tmp_path, home)
    out = " ".join(r.output.split())
    assert r.exit_code == 0 and ran, r.output
    assert out.count(f"Note: FOURIER_HOME is {home}, which doesn't exist yet") == 1, out
    home.mkdir(parents=True, exist_ok=True)
    r, _ran = _scan(monkeypatch, tmp_path, home)
    assert "doesn't exist yet" not in r.output


def test_the_default_home_says_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("FOURIER_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    monkeypatch.setattr(I, "run_scan", lambda *a, **k: None)
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "tools", "scan"])
    assert "FOURIER_HOME" not in r.output, r.output


@pytest.mark.parametrize("make, missing", [(None, True), ("empty", True), ("files", False),
                                           ("mount", False)])
def test_unmounted_volume(volumes, monkeypatch, make, missing):
    vol = volumes / "Drive"
    if make:
        vol.mkdir()
    if make == "files":
        (vol / "a.wav").write_bytes(b"")
    if make == "mount":
        real = platforms.os.path.ismount
        monkeypatch.setattr(platforms.os.path, "ismount", lambda p: str(p) == str(vol) or real(p))
    assert platforms.unmounted_volume(vol / "fourier") == (vol if missing else None)
    assert platforms.unmounted_volume(volumes.parent / "fourier") is None   # on no volume
