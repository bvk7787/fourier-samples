"""One writing run at a time per Fourier home.

The commands that write the home and the folders it feeds (build, tools scan, tools analyze,
render, sync, publish) hold <home>/fourier.lock while they run. A second one started meanwhile
stops at once, naming the run that holds it (its command, pid and start time). Read-only
commands and dry runs never take it.

The lock is an OS file lock (flock on macOS and Linux, msvcrt on Windows), so the system
drops it when its process ends, killed or crashed included: a lock file left by a process
that's gone is simply taken over. Where the file system has no locks (some network shares),
the pid in the file decides: a pid on this machine that no longer runs is taken over.
"""
from __future__ import annotations

import errno
import json
import os
import socket
from datetime import datetime
from pathlib import Path

from .paths import fourier_home

LOCK_NAME = "fourier.lock"

# lock path -> open file descriptor, for the locks this process holds
_HELD: dict[str, int] = {}


def _forget_in_child() -> None:
    """A forked child (a pool worker) closes its copies of the held locks: the lock belongs to
    the run, and a worker left behind by a run that was killed mustn't keep it."""
    for fd in _HELD.values():
        try:
            os.close(fd)
        except OSError:
            pass
    _HELD.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_in_child)


class HomeBusy(Exception):
    """Another run holds the home's lock. holder: what it wrote in the lock file."""

    def __init__(self, home: Path, holder: dict):
        self.home, self.holder = home, holder
        super().__init__(describe(home, holder))


def lock_path(home: Path | None = None) -> Path:
    return Path(home or fourier_home()) / LOCK_NAME


def describe(home: Path, holder: dict) -> str:
    """The line a run that finds the home busy stops with."""
    cmd = holder.get("command") or "another fourier command"
    bits = []
    if holder.get("pid"):
        bits.append(f"pid {holder['pid']}")
    if holder.get("host") and holder["host"] != socket.gethostname():
        bits.append(f"on {holder['host']}")
    started = _when(holder.get("started"))
    if started:
        bits.append(f"started {started}")
    who = f"`{cmd}`" + (f" ({', '.join(bits)})" if bits else "")
    line = (f"{who} is already writing to {home}. Two runs at once on one home trip over each "
            f"other's files: run this again once it finishes, or stop that one first.")
    here = not holder.get("host") or holder["host"] == socket.gethostname()
    if holder.get("pid") and here and not pid_alive(holder["pid"]):
        line += (" That process has ended, but one it started still holds the lock: find it "
                 "with `pgrep -fl fourier` and stop it.")
    elif not here:
        line += (f" If that run stopped without finishing, empty {lock_path(home)} to clear it.")
    return line


def _when(iso) -> str:
    try:
        return datetime.fromisoformat(str(iso)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return ""


def _os_lock(fd: int) -> bool:
    """Take the OS lock on fd without waiting: True when taken, False when another process
    holds it. OSError when the file system can't lock."""
    try:
        import fcntl
    except ImportError:                       # Windows
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        except OSError as e:
            if e.errno in (errno.EACCES, errno.EDEADLK):
                return False
            raise
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EACCES):
            return False
        raise
    return True


def _os_unlock(fd: int) -> None:
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_UN)
    except ImportError:
        import msvcrt
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        except OSError:
            pass
    except OSError:
        pass


def _read(fd: int) -> dict:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        data = os.read(fd, 64 * 1024)
        got = json.loads(data.decode("utf-8")) if data.strip() else {}
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def pid_alive(pid) -> bool:
    """Whether a process with this pid runs on this machine."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                   # someone else's process: it runs
        return True
    except OSError:
        return False
    return True


def _stale(holder: dict) -> bool:
    """For a file system without locks: the holder named in the file is gone (a pid on this
    machine that no longer runs), or the file names none."""
    if not holder.get("pid"):
        return True
    if holder.get("host") and holder["host"] != socket.gethostname():
        return False                          # can't tell: another machine's process
    return not pid_alive(holder["pid"])


def hold(command: str, home: Path | None = None) -> None:
    """Hold the home's lock for the rest of this process, or until release(). Raises HomeBusy
    when another run holds it. Holding it already is fine (a command run inside another)."""
    home = Path(home or fourier_home())
    path = lock_path(home)
    if str(path) in _HELD:
        return
    home.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            taken = _os_lock(fd)
        except OSError:                       # no locks on this file system: the pid decides
            holder = _read(fd)
            taken = _stale(holder)
        if not taken:
            raise HomeBusy(home, _read(fd))
        me = {"pid": os.getpid(), "host": socket.gethostname(), "command": command,
              "started": datetime.now().isoformat(timespec="seconds")}
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, (json.dumps(me) + "\n").encode("utf-8"))
    except BaseException:
        os.close(fd)
        raise
    _HELD[str(path)] = fd


def release(home: Path | None = None) -> None:
    """Let go of the home's lock (every one this process holds when home is None). The file
    stays, emptied: the next run takes it over."""
    paths = [str(lock_path(home))] if home else list(_HELD)
    for p in paths:
        fd = _HELD.pop(p, None)
        if fd is None:
            continue
        try:
            os.ftruncate(fd, 0)
        except OSError:
            pass
        _os_unlock(fd)
        os.close(fd)


def holder(home: Path | None = None) -> dict | None:
    """Who holds the home's lock now, or None when nobody does (read only: takes nothing)."""
    path = lock_path(home)
    if str(path) in _HELD:
        return _read(_HELD[str(path)])
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return None
    try:
        try:
            free = _os_lock(fd)
        except OSError:
            info = _read(fd)
            return None if _stale(info) else info
        if free:
            _os_unlock(fd)
            return None
        return _read(fd)
    finally:
        os.close(fd)
