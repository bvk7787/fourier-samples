"""A build's quiet mode (fourier/cli/quiet.py): what the Build and Verify stages say goes to
a log, worker processes' output too, and the terminal gets a summary, or the log's last lines
when the build stops."""
from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from fourier.cli import quiet as Q


def test_quiet_only_on_a_terminal_unless_asked(monkeypatch):
    assert Q.wants_quiet(True) is False and Q.wants_quiet(False) is True
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert Q.wants_quiet(None) is False                  # a pipe or a file: everything
    tty = io.StringIO()
    tty.isatty = lambda: True
    monkeypatch.setattr(sys, "stdout", tty)
    assert Q.wants_quiet(None) is True
    monkeypatch.setenv("FOURIER_VERBOSE", "1")
    assert Q.wants_quiet(None) is False


def _master(tmp_path):
    m = tmp_path / "master"
    (m / "KITS" / "k1").mkdir(parents=True)
    (m / "KITS" / "k1" / "a.wav").write_bytes(b"x")
    (m / "manifest.json").write_text(json.dumps({"categories": {
        "SNARES": {"entries": [{}] * 3}, "KICKS": {"entries": [{}] * 5}}}))
    return m


def test_the_detail_goes_to_the_log_and_the_summary_reads_it(tmp_path, monkeypatch):
    seen = io.StringIO()
    monkeypatch.setattr(sys, "stdout", seen)
    m = _master(tmp_path)
    q = Q.Quiet(tmp_path / "build.log")
    q.start(str(m), 2)
    print("Homes: 12 samples routed")                    # what a build says
    print("1 category left empty: too few samples the rules recognize for them (WAVES).")
    print("40 samples no rule recognized (top folders: Acme (40)).")
    for i in range(7):
        print(f"WARN  check {i}  -- detail")
    print("verify: PASS -- 100 pass, 7 warn, 0 fail")
    print(f"New master in place at {m}.")
    subprocess.run([sys.executable, "-c", "print('from a worker')"], check=True)
    q.stop()
    assert seen.getvalue() == ""                         # nothing reached the terminal
    log = (tmp_path / "build.log").read_text()
    assert "Homes: 12 samples routed" in log and "from a worker" in log
    out = q.summary()
    assert out[0] == "Built 2 categories, 8 files:"
    assert out[1].split() == ["KICKS", "5", "SNARES", "3"]
    assert "plus KITS (1 files), drawn from the categories" in out[2]
    assert "1 category left empty: too few samples the rules recognize for them (WAVES)." in out
    assert "40 samples no rule recognized: `fourier why --unrecognized` lists them." in out
    assert "verify: PASS -- 100 pass, 7 warn, 0 fail" in out
    assert sum(1 for ln in out if ln.startswith("  WARN")) == Q.WARN_SHOWN
    assert any("2 more warnings in the log" in ln for ln in out)
    assert out[-3] == f"New master in place at {m}." and "fourier open report" in out[-2]
    assert str(tmp_path / "build.log") in out[-1]
    assert "Homes:" not in "\n".join(out)


def test_a_stopped_build_shows_the_logs_last_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    q = Q.Quiet(tmp_path / "build.log")
    q.start(str(tmp_path / "master"), 1)
    for i in range(40):
        print(f"line {i}")
    print("build stopped: the source is gone")
    q.stop()
    out = q.failure()
    assert "line 39" in out and "line 10" not in out
    assert "build stopped: the source is gone" in out
    assert out[-1] == f"The build stopped. Everything it said: {tmp_path / 'build.log'}"


def test_old_logs_are_trimmed(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path))
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(Q.KEEP_LOGS + 3):
        (logs / f"build-2026010{i:02d}.log").write_text("x")
    p = Q.log_path()
    assert len(list(logs.glob("build-*.log"))) == Q.KEEP_LOGS - 1 and not p.exists()


@pytest.mark.parametrize("flag", ["--quiet", "--verbose"])
def test_build_takes_the_flags(flag):
    from click.testing import CliRunner

    from fourier.cli import main
    r = CliRunner().invoke(main, ["build", "--help"])
    assert flag.lstrip("-") in r.output


def test_the_lines_the_summary_reads_are_still_printed():
    """The summary finds its lines in the log by how they start: a build that reworded one
    would drop it from the summary silently."""
    from pathlib import Path

    import fourier.cli.build as B
    import fourier.packs.verify as V
    src = Path(B.__file__).read_text()
    for marker in ("New master in place at ", " no rule recognized (top folders: ",
                   " left empty: ", "verify: PASS", "verify: FAIL"):
        assert marker in src, marker
    assert 'log(f"{r.level}  {r.check}"' in Path(V.__file__).read_text()   # "WARN  <check>"
