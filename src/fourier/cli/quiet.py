"""A build's quiet mode: the Build and Verify stages' detail (metadata, routing, every
category's families, every check) goes to a log file, and the terminal shows a spinner
with how many categories are done, then a short summary: what was built, what was left
empty, verify's result and its warnings, and where the master and the log are.

`fourier build` is quiet on a terminal and prints everything when its output goes to a file
or a pipe (scripts, CI, the golden harness), or with --verbose. A build that stops shows the
log's last lines. The scan and the analysis aren't quieted: on a first run they take a while,
and their progress bars are the progress.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

KEEP_LOGS = 10                    # build logs kept in <home>/logs
TAIL = 25                         # a stopped build shows this many of the log's last lines
WARN_SHOWN = 5
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def wants_quiet(verbose) -> bool:
    """--verbose / --quiet as given, else quiet only when the output is a terminal."""
    if verbose is not None:
        return not verbose
    if os.environ.get("FOURIER_VERBOSE", "").strip() not in ("", "0"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def log_path() -> Path:
    """A new build log in <home>/logs, the oldest beyond KEEP_LOGS removed."""
    from ..paths import fourier_home
    d = fourier_home() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    old = sorted(d.glob("build-*.log"))
    for p in old[:max(0, len(old) - KEEP_LOGS + 1)]:
        p.unlink(missing_ok=True)
    return d / f"build-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"


class Quiet:
    """Started at the Build stage (start()); stopped by `fourier build` when the build ends,
    which then prints summary() or failure()."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.out_dir = ""
        self.total = 0
        self.started = False
        self._spin: threading.Thread | None = None

    # -- capturing ------------------------------------------------------------------------
    def start(self, out_dir: str, total: int) -> None:
        if self.started:
            return
        self.out_dir, self.total = str(out_dir), total
        self.path = self.path or log_path()
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:
                pass
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
        self.saved = (sys.stdout, sys.stderr)
        real = _fd(sys.stdout) == 1
        # what the user sees: the terminal itself (a copy of fd 1, before fd 1 goes to the
        # log), or the stream a caller handed us (a test's)
        self.term = os.fdopen(os.dup(1), "w", buffering=1) if real else sys.stdout
        self._own_term = real
        self.fds = (os.dup(1), os.dup(2))
        os.dup2(self.f.fileno(), 1)          # worker processes write there too
        os.dup2(self.f.fileno(), 2)
        sys.stdout = sys.stderr = self.f
        from ._app import console
        self._width = console._width
        console.width = 1000                 # a log's lines whole (a file is 80 columns to rich)
        self.started = True
        self.t0 = time.monotonic()
        if real and self.term.isatty():
            self._spin = threading.Thread(target=self._spinner, daemon=True)
            self._stop = threading.Event()
            self._spin.start()

    def stop(self) -> None:
        if not self.started:
            return
        if self._spin is not None:
            self._stop.set()
            self._spin.join(timeout=2)
            self.term.write("\r\033[K")
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:
                pass
        sys.stdout, sys.stderr = self.saved
        from ._app import console
        console._width = self._width
        os.dup2(self.fds[0], 1)
        os.dup2(self.fds[1], 2)
        for fd in self.fds:
            os.close(fd)
        self.f.close()
        if self._own_term:
            self.term.flush()
            self.term.close()
        self.started = False

    def _spinner(self) -> None:
        frames = "|/-\\"
        prog = Path(self.out_dir.rstrip("/") + ".next") / ".progress"
        i = 0
        while not self._stop.wait(0.25):
            try:
                done = sum(1 for _ in prog.glob("*.json"))
            except OSError:
                done = 0
            secs = int(time.monotonic() - self.t0)
            what = (f"{done} of {self.total} categories" if self.total > 1 and done < self.total
                    else "checking and finishing")
            self.term.write(f"\r\033[K{frames[i % 4]} Building the master: {what} "
                            f"({secs // 60}:{secs % 60:02d})")
            self.term.flush()
            i += 1

    # -- what the user reads --------------------------------------------------------------
    def lines(self) -> list[str]:
        try:
            return [_ANSI.sub("", ln.rstrip("\n")) for ln in open(self.path, encoding="utf-8",  # type: ignore[arg-type]
                                                                   errors="replace")]
        except OSError:
            return []

    def summary(self) -> list[str]:
        """After a build that finished: what it built, what it left out, verify, where."""
        log = self.lines()
        out = []
        try:
            from ..packs import manifests
            man = manifests.read(Path(self.out_dir) / "manifest.json")
        except (OSError, ValueError):
            man = {}
        cats = {c: len((d or {}).get("entries") or ()) for c, d in (man.get("categories") or {}).items()}
        if cats:
            from ..packs.curate_config import CATEGORY_ORDER
            order = list(CATEGORY_ORDER)
            cats = dict(sorted(cats.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))
            out.append(f"Built {len(cats)} categor{'y' if len(cats) == 1 else 'ies'}, "
                       f"{sum(cats.values()):,} files:")
            width = max(map(len, cats))
            rows = [f"  {c:<{width}}  {n:>5,}" for c, n in cats.items()]
            half = (len(rows) + 1) // 2
            out += [f"{a}      {b}".rstrip() for a, b in zip(rows[:half], rows[half:] + [""])]
            sets = {s: _count_wavs(Path(self.out_dir) / s) for s in ("KITS", "SLICE")}
            sets = {s: n for s, n in sets.items() if n}
            if sets:
                out.append("  plus " + " and ".join(f"{s} ({n:,} files)" for s, n in sets.items())
                           + ", drawn from the categories")
        out += [ln for ln in log if re.search(r"categor(y|ies) left empty", ln)]
        unrec = next((ln for ln in log if " no rule recognized (top folders" in ln), None)
        if unrec:
            out.append(unrec.split(" (top folders")[0] + ": `fourier why --unrecognized` lists them.")
        ver = next((ln for ln in reversed(log) if ln.startswith("verify: ")), None)
        warns = [ln for ln in log if ln.startswith("WARN ")]
        if ver:
            out.append(ver)
            out += ["  " + w for w in warns[:WARN_SHOWN]]
            if len(warns) > WARN_SHOWN:
                out.append(f"  ... {len(warns) - WARN_SHOWN} more warnings in the log "
                           f"(`fourier verify` lists them all)")
        placed = [ln for ln in log if ln.startswith("New master in place at ")]
        out += placed
        if placed:
            out.append("Listen to it, see why each sound is there and rate it: `fourier open report`.")
        out.append(f"Everything the build said: {self.path}")
        return out

    def failure(self) -> list[str]:
        """After a build that stopped: the log's last lines, and where the rest is."""
        tail = [ln for ln in self.lines() if ln.strip()][-TAIL:]
        return ["", *tail, "", f"The build stopped. Everything it said: {self.path}"]


def _fd(stream) -> int | None:
    try:
        return stream.fileno()
    except (AttributeError, ValueError, OSError):
        return None


def _count_wavs(folder: Path) -> int:
    try:
        return sum(1 for p in folder.rglob("*") if p.suffix.lower() in (".wav", ".aif", ".aiff"))
    except OSError:
        return 0
