"""Processed-audio cache.

A whole build re-reads and re-processes every source file it exports (trim, fades, DC, loudness,
mono, retune) even when only routing or naming changed. The cache keeps each processed
result under ~/.fourier/cache/audio, keyed by the source file (path, size, mtime), every
processing argument, and a fingerprint of the DSP code itself (the source of
curate._process_audio and every function and constant it reaches), so any DSP change
misses the cache on its own. A hit is hardlinked into the build (same volume: no copy, no
extra space); a "too quiet" verdict is cached too.

Pruning drops entries no master links to any more (link count 1) that weren't used (access
time) for PRUNE_DAYS. FOURIER_NO_AUDIO_CACHE=1 turns the cache off.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import shutil
import time
import types
from pathlib import Path

PRUNE_DAYS = 14
# bump when curate's export_one changes what it does after _process_audio (quiet gate,
# RMS clamp, output subtype): those steps aren't in dsp_fingerprint
EXPORT_VERSION = 1
_FP = None


def cache_dir() -> Path:
    from .ratings import default_store
    return Path(os.environ.get("FOURIER_AUDIO_CACHE")
                or os.path.join(os.path.dirname(default_store()), "cache", "audio"))


def enabled() -> bool:
    return os.environ.get("FOURIER_NO_AUDIO_CACHE", "") not in ("1", "true", "yes")


def _reach(fn, mod, seen, parts):
    """Source of fn plus every module-level function and plain constant it names (in mod),
    transitively."""
    if fn in seen:
        return
    seen.add(fn)
    try:
        parts.append(inspect.getsource(fn))
    except (OSError, TypeError):
        parts.append(repr(fn))
    # default argument values (fade_ms=END_FADE_MS) are bound at def time, not named in
    # the code: hash their values too
    parts.append(repr(fn.__defaults__))
    parts.append(repr(sorted((fn.__kwdefaults__ or {}).items())))
    names = set()
    stack = [fn.__code__]
    while stack:
        co = stack.pop()
        names |= set(co.co_names)
        stack += [c for c in co.co_consts if isinstance(c, types.CodeType)]
    for n in sorted(names):
        v = getattr(mod, n, None)
        if isinstance(v, types.FunctionType):
            _reach(v, inspect.getmodule(v) or mod, seen, parts)
        elif isinstance(v, (int, float, str, bytes, tuple, frozenset, bool)) or (
                isinstance(v, dict) and all(isinstance(x, (int, float, str, type(None))) for x in v.values())):
            parts.append(f"{n}={v!r}")


def dsp_fingerprint() -> str:
    """Hash of the DSP code a processed file depends on (cached per process)."""
    global _FP
    if _FP is None:
        from . import curate as C
        parts = []
        seen = set()
        for f in (C._process_audio, C._clamp_rms, C._rms_db):
            _reach(f, C, seen, parts)
        import numpy as np
        import scipy
        import soundfile as sf
        parts.append(f"numpy={np.__version__} scipy={scipy.__version__} sf={sf.__version__}")
        # (only what the DSP reaches: an unrelated edit to curate_config, a routing rule or a
        # comment, no longer makes the next build cold; the other export settings are in the key)
        _FP = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    return _FP


def key(src, **args) -> str | None:
    """Cache key for processing src with args, or None when the source can't be stat'ed."""
    try:
        st = os.stat(src)
    except OSError:
        return None
    blob = repr((os.path.abspath(src), st.st_size, st.st_mtime_ns, sorted(args.items()), dsp_fingerprint()))
    return hashlib.sha256(blob.encode()).hexdigest()


def _path(k):
    return cache_dir() / k[:2] / k


def lookup(k, out: Path):
    """'hit' (out now holds the cached audio), 'quiet' (cached as too quiet), or None."""
    if not k or not enabled():
        return None
    p = _path(k)
    if (p.parent / (k + ".quiet")).exists():
        _touch(p.parent / (k + ".quiet"))
        return "quiet"
    wav = p.with_suffix(".wav")
    if not wav.exists():
        return None
    try:
        os.link(wav, out)
    except OSError:
        shutil.copy2(wav, out)
    _touch(wav)
    return "hit"


def store(k, out: Path | None):
    """Remember out (a finished processed file) for k, or a quiet verdict when out is None."""
    if not k or not enabled():
        return
    p = _path(k)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        if out is None:
            (p.parent / (k + ".quiet")).touch()
            return
        wav = p.with_suffix(".wav")
        if wav.exists():
            return
        tmp = p.parent / f".{k}.{os.getpid()}.tmp"
        try:
            os.link(out, tmp)
        except OSError:
            shutil.copy2(out, tmp)
        os.replace(tmp, wav)
    except OSError:
        pass


def _touch(p):
    """Mark a cache entry used: its access time only. Its mtime is the master file's (they're
    hardlinks), and bumping it made Live and Spotlight re-scan every file on each build."""
    try:
        st = os.stat(p)
        os.utime(p, ns=(time.time_ns(), st.st_mtime_ns))
    except OSError:
        pass


def prune(days=PRUNE_DAYS, log=print) -> int:
    """Remove cached files no master links to that weren't used in `days` days."""
    d = cache_dir()
    if not d.exists():
        return 0
    cut = time.time() - days * 86400
    n = 0
    for p in d.glob("*/*"):
        try:
            st = p.stat()
        except OSError:
            continue
        if max(st.st_atime, st.st_mtime) < cut and (p.suffix == ".quiet" or st.st_nlink == 1):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    if n:
        log(f"audio cache: pruned {n} unused entries")
    return n
