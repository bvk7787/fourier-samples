"""Preparers: plug-ins that add an audio codec soundfile doesn't have.

Fourier reads open formats (WAV, AIFF, FLAC, OGG, MP3) through soundfile. A preparer is a
separately installed package for another open or openly documented format (WavPack, say) with its
own file extensions. `fourier tools import-folder` offers it the files with those extensions and
writes what it loads as WAV in the library. A preparer can't claim an extension soundfile
already reads, and Fourier takes no code that circumvents copy protection, here or in a
plug-in.

A preparer is an object with
    name              short id, used to enable it
    extensions        the file extensions it reads, lowercase with the dot (".wv", ...)
    claims(path)      True if it reads this file (a header sniff), False if not,
                      None if the file can't be inspected
    load(path)        (PCM as a float array [frames, channels], sample rate, soundfile subtype)
registered by its package under the entry-point group "fourier.preparers", e.g.

    [project.entry-points."fourier.preparers"]
    wavpack = "my_package.module:PREPARER"

and enabled by name in fourier.toml (so installing one never changes a build by itself):

    preparers = ["wavpack"]
"""
from __future__ import annotations

from typing import Protocol

GROUP = "fourier.preparers"
# what soundfile reads itself: never a preparer's (plus every format libsndfile lists, open_exts())
OPEN_EXTS = frozenset({".wav", ".wave", ".bwf", ".aif", ".aiff", ".aifc", ".flac", ".ogg", ".oga",
                       ".opus", ".mp3", ".caf", ".w64", ".rf64", ".au", ".snd", ".sd2", ".xi",
                       ".voc", ".paf", ".svx", ".8svx", ".iff", ".nist", ".sph", ".ircam", ".sf",
                       ".mat", ".mat4", ".mat5", ".pvf", ".htk", ".sds", ".avr", ".wve", ".mpc",
                       ".raw"})


def open_exts() -> frozenset:
    """OPEN_EXTS and every extension the installed libsndfile names."""
    try:
        import soundfile as sf
        return OPEN_EXTS | {"." + k.lower() for k in sf.available_formats()}
    except Exception:
        return OPEN_EXTS


class Preparer(Protocol):
    name: str
    extensions: tuple[str, ...]

    def claims(self, path) -> bool | None: ...

    def load(self, path): ...


class PreparerError(RuntimeError):
    pass


def installed() -> list[str]:
    """The names of the installed preparers (their entry points; nothing is loaded)."""
    from importlib.metadata import entry_points
    return sorted(ep.name for ep in entry_points(group=GROUP))


def _load(name):
    from importlib.metadata import entry_points
    for ep in entry_points(group=GROUP):
        if ep.name == name:
            try:
                obj = ep.load()
                return obj() if isinstance(obj, type) else obj      # a class: one instance
            except Exception as e:
                raise PreparerError(f"preparer {name!r} failed to load: {e}") from None
    return None


def available() -> dict:
    """{name: preparer} for every installed preparer (loads them all)."""
    return {n: _load(n) for n in installed()}


def configured() -> list[str]:
    """The names fourier.toml enables (`preparers = [...]`); none without a config."""
    from .layers import _read, find_config
    path = find_config()
    names = (_read(path).get("preparers") if path else None) or []
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise PreparerError(f"{path}: preparers must be a list of names")
    return names


def enabled(names: list[str] | None = None) -> list:
    """The enabled preparers, in the order named. A name that isn't installed is an error."""
    names = configured() if names is None else names
    if not names:
        return []
    have = installed()
    missing = [n for n in names if n not in have]
    if missing:
        raise PreparerError(f"preparers not installed: {', '.join(missing)} "
                            f"(installed: {', '.join(have) or 'none'})")
    out = [_load(n) for n in names]           # only the enabled ones are imported
    for p in out:
        check(p)
    return out


def extensions(p) -> set[str]:
    return {str(e).lower() for e in (getattr(p, "extensions", None) or ())}


def check(p) -> None:
    """A preparer names its own extensions, none of them one soundfile reads."""
    exts = extensions(p)
    name = getattr(p, "name", "?")
    if not exts:
        raise PreparerError(f"preparer {name!r} names no file extensions")
    if not all(e.startswith(".") for e in exts):
        raise PreparerError(f"preparer {name!r}: extensions start with a dot (\".wv\")")
    taken = exts & open_exts()
    if taken:
        raise PreparerError(f"preparer {name!r} can't claim {', '.join(sorted(taken))}: "
                            f"soundfile reads those itself")


def claim(path, preparers) -> object | None:
    """The first preparer that claims this file (by extension, then its header), else None."""
    from pathlib import Path
    ext = Path(path).suffix.lower()
    if ext in open_exts():
        return None
    for p in preparers:
        if ext not in extensions(p):
            continue
        try:
            if p.claims(path):
                return p
        except Exception:
            continue
    return None
