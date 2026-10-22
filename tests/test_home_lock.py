"""One writing run at a time per home: a second build, scan, analyze, render, sync or publish
stops at once and names the run that holds the home; read-only commands and dry runs don't
wait, and a lock left by a process that's gone is taken over."""
import json
import os
import subprocess
import sys

import pytest
from click.testing import CliRunner

from fourier import homelock
from fourier.cli import ingest as I
from fourier.cli import main
from fourier.paths import fourier_home

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# a real `fourier tools scan` in another process, held inside its scan until stdin closes
_CHILD = """
import sys
from fourier.cli import ingest as I
from fourier.cli import main
def held(*a, **k):
    print("scanning", flush=True)
    sys.stdin.read()
I.run_scan = held
main(["tools", "scan"], standalone_mode=False)
"""


def _env() -> dict:
    """This checkout's fourier for the child, ahead of whatever it has on its path."""
    return dict(os.environ, PYTHONPATH=os.pathsep.join(
        [SRC, *filter(None, [os.environ.get("PYTHONPATH")])]))


@pytest.fixture
def other_scan():
    """A `fourier tools scan` running in another process on this home."""
    p = subprocess.Popen([sys.executable, "-c", _CHILD], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=_env())
    try:
        line = p.stdout.readline()
        assert "scanning" in line, line + p.stdout.read()
        yield p
    finally:
        p.stdin.close()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _scan(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(I, "run_scan", lambda *a, **k: ran.append(1))
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "tools", "scan"])
    return r, ran


def test_a_second_writer_stops_and_names_the_first(other_scan, monkeypatch, tmp_path):
    r, ran = _scan(monkeypatch, tmp_path)
    out = " ".join(r.output.split())
    assert r.exit_code == 1, r.output
    assert not ran                                    # it never started scanning
    assert "`fourier tools scan` (pid " in out and f"pid {other_scan.pid}" in out, out
    assert "started " in out and str(fourier_home()) in out
    assert "run this again once it finishes" in out


def test_a_build_stops_too(other_scan, monkeypatch, tmp_path):
    from fourier.cli import enrich as E
    monkeypatch.setattr(E, "run_analysis", lambda *a, **k: pytest.fail("analysis ran"))
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "build", "--all",
                                  "--out", str(tmp_path / "m")])
    assert r.exit_code == 1, r.output
    assert f"fourier build stopped: `fourier tools scan` (pid {other_scan.pid}" in \
        " ".join(r.output.split())


def test_read_only_commands_run_meanwhile(other_scan, tmp_path):
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "tools", "analyze",
                                  "--status"])
    assert "already writing" not in r.output, r.output


def test_the_lock_goes_with_its_process(monkeypatch, tmp_path):
    """Once the other run has ended (here: killed), its lock file is taken over."""
    p = subprocess.Popen([sys.executable, "-c", _CHILD], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, env=_env())
    assert "scanning" in p.stdout.readline()
    p.kill()
    p.wait()
    assert json.loads(homelock.lock_path().read_text())["pid"] == p.pid    # left behind
    r, ran = _scan(monkeypatch, tmp_path)
    assert r.exit_code == 0 and ran, r.output
    assert homelock.holder() is None                  # and let go when the command ended


_FORKS = """
import os, sys, time
from fourier import homelock
homelock.hold("fourier build")
pid = os.fork()                     # a pool worker, forked while the run holds the lock
if pid == 0:
    time.sleep(60)
    os._exit(0)
print(pid, flush=True)
sys.stdin.read()
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork")
def test_a_worker_left_by_a_killed_run_doesnt_keep_the_lock(monkeypatch, tmp_path):
    """A run killed outright (kill -9, the OOM killer) can leave forked pool workers behind;
    they don't hold its lock, so the next run goes ahead."""
    p = subprocess.Popen([sys.executable, "-c", _FORKS], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True, env=_env())
    worker = int(p.stdout.readline())
    try:
        p.kill()
        p.wait()
        assert homelock.pid_alive(worker)
        r, ran = _scan(monkeypatch, tmp_path)
        assert r.exit_code == 0 and ran, r.output
    finally:
        os.kill(worker, 9)


def test_a_holder_thats_gone_is_named_as_gone(tmp_path):
    """When the lock's recorded process has ended (a worker it left holds the lock), the stop
    line says so and how to find the leftover; another machine's holder: how to clear it."""
    gone = homelock.describe(tmp_path, {"pid": 2 ** 22 + 7, "command": "fourier build"})
    assert "has ended" in gone and "pgrep -fl fourier" in gone
    other = homelock.describe(tmp_path, {"pid": 5, "host": "zorbl-host", "command": "fourier build"})
    assert "on zorbl-host" in other and "fourier.lock" in other


def test_one_run_after_another(monkeypatch, tmp_path):
    for _ in range(2):
        r, ran = _scan(monkeypatch, tmp_path)
        assert r.exit_code == 0 and ran, r.output


def _no_os_locks(monkeypatch):
    def fail(fd):
        raise OSError(37, "No locks available")
    monkeypatch.setattr(homelock, "_os_lock", fail)


def test_without_os_locks_a_dead_pid_is_taken_over(monkeypatch, tmp_path):
    _no_os_locks(monkeypatch)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    home = tmp_path / "home"
    home.mkdir()
    (home / homelock.LOCK_NAME).write_text(json.dumps(
        {"pid": p.pid, "host": __import__("socket").gethostname(), "command": "fourier build"}))
    homelock.hold("fourier build", home)
    try:
        assert json.loads((home / homelock.LOCK_NAME).read_text())["pid"] == os.getpid()
    finally:
        homelock.release(home)


def test_without_os_locks_a_live_pid_holds(monkeypatch, tmp_path):
    _no_os_locks(monkeypatch)
    home = tmp_path / "home"
    home.mkdir()
    (home / homelock.LOCK_NAME).write_text(json.dumps(
        {"pid": os.getppid(), "host": __import__("socket").gethostname(),
         "command": "fourier publish", "started": "2026-01-02T03:04:05"}))
    with pytest.raises(homelock.HomeBusy) as e:
        homelock.hold("fourier build", home)
    assert "`fourier publish` (pid" in str(e.value) and "started 2026-01-02 03:04:05" in str(e.value)


@pytest.mark.parametrize("args", [["sync", "m8_tracker", "{tmp}/card"],
                                  ["publish", "--from", "{tmp}/master", "--to", "{tmp}/pub"]])
def test_sync_and_publish_stop_too(other_scan, tmp_path, args):
    (tmp_path / "master").mkdir()
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"),
                                  *[a.format(tmp=tmp_path) for a in args]])
    assert r.exit_code == 1, r.output
    assert f"fourier {args[0]} stopped: `fourier tools scan` (pid {other_scan.pid}" in \
        " ".join(r.output.split())
    assert not (tmp_path / "pub").exists()


def test_their_dry_runs_dont_wait(other_scan, tmp_path):
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), "sync", "m8_tracker",
                                  str(tmp_path / "card"), "--dry-run"])
    assert "already writing" not in r.output, r.output
