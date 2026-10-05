"""Where things are: the library, the master, renders, releases and device path locks.

From fourier.toml (the file layers.find_config names), each with an environment variable that
wins over it and a default:

    library = ["~/Samples"]              # $FOURIER_LIBRARY (os.pathsep-separated)
    devices = ["digitakt_2"]             # render targets (config/devices/<id>.yaml)
    sononym_db = "~/Samples/sononym.db"  # default: sononym.db in the first library folder
    [output]
    master = "~/Music/FourierCurated"    # $FOURIER_CURATED_DIR
    renders = "~/Music/FourierRenders"   # default: FourierRenders beside the master
    publish = "~/Music/Fourier"          # releases/ and devices/ (path locks) go under it;
                                         # $FOURIER_LOCK_DIR moves the locks alone

A library entry is a folder (absolute or ~) or a bare folder name: "Samples" matches the last
folder of that name in a path, wherever the library sits (a library that moves between
machines). A sample's library path, the one curation reads (vendor/pack/...), is its path
under the library folder; a path under none of them is used whole.

Nothing here creates folders; callers do that when they write.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

ENV_LIBRARY = "FOURIER_LIBRARY"
DEFAULT_MASTER = "~/Music/FourierCurated"
DEFAULT_PUBLISH = "~/Music/Fourier"


class PlacesError(ValueError):
    pass


def _config() -> tuple[str | None, dict]:
    from .layers import _read, find_config
    path = find_config()
    return (str(path), _read(path)) if path else (None, {})


def _expand(p: str) -> str:
    return os.path.normpath(os.path.expanduser(str(p)))


def _entries(where: str, value, key: str) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise PlacesError(f"{where}: {key} must be a folder or a list of folders")
    return value


@lru_cache(maxsize=8)
def _library(env: str | None, config: str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(folders, folder names) for this environment value and config file."""
    if env:
        entries = [e for e in env.split(os.pathsep) if e]
    else:
        where, doc = config, (_config()[1] if config else {})
        entries = _entries(where, doc.get("library", []), "library")
    seps = "/" + os.sep
    roots = sorted({_expand(e).rstrip(seps) for e in entries if _is_folder(e)}, key=len, reverse=True)
    names = tuple(e.strip(seps) for e in entries if not _is_folder(e.strip(seps)))
    return tuple(roots), names


def _is_folder(entry: str) -> bool:
    """A library entry that names a folder (a path, or ~), not a bare folder name. On Windows
    a path uses backslashes and may start with a drive."""
    return "/" in entry or os.sep in entry or entry.startswith("~") or bool(os.path.splitdrive(entry)[0])


_FOUND: dict = {}


def library() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(folders, folder names). Looked up once per environment and working folder: curation
    asks for every sample."""
    from .layers import ENV_CONFIG, find_config
    env = os.environ.get(ENV_LIBRARY)
    cfg = os.environ.get(ENV_CONFIG)
    key = (env, cfg, None if env or cfg else os.getcwd())
    if key not in _FOUND:
        path = None if env else find_config()
        _FOUND[key] = _library(env, str(path) if path else None)
    return _FOUND[key]


def reset() -> None:
    """Forget the looked-up library (a test that rewrites fourier.toml)."""
    _FOUND.clear()
    _library.cache_clear()
    _real_roots.cache_clear()


def library_roots() -> list[Path]:
    """The library folders (not the bare folder names)."""
    return [Path(r) for r in library()[0]]


def _rel(path: str, roots: tuple, names: tuple) -> str | None:
    for r in roots:
        if path.startswith(r + "/"):
            return path[len(r) + 1:]
    for n in names:
        i = path.rfind(f"/{n}/")
        if i >= 0:
            return path[i + len(n) + 2:]
        if path.startswith(f"{n}/"):
            return path[len(n) + 1:]
    if _windows_shaped(path):
        return _rel_windows(path, roots, names)
    for r, real in _real_roots(roots):          # the root's own folder, its symlinks resolved
        if path.startswith(real + "/"):
            return path[len(real) + 1:]
    return None


_DRIVE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _windows_shaped(path: str) -> bool:
    """A Windows path: a drive ("C:\\") or a UNC share, or backslashes as separators. Only
    asked once the "/" spellings found no library folder, so a POSIX path with a backslash
    in a file name that sits under a library folder is matched as before."""
    return "\\" in path or bool(_DRIVE.match(path))


def _rel_windows(path: str, roots: tuple, names: tuple) -> str | None:
    """_rel for a Windows path: "/" for every separator, and case ignored (Windows folds
    it), so the library path is "Vendor/Pack/file.wav" whichever way it was spelled."""
    p = path.replace("\\", "/")
    fold = _fold if len(_fold(p)) == len(p) else (lambda x: x)    # keep indexes aligned
    low = fold(p)
    for r in roots:
        rr = r.replace("\\", "/").rstrip("/")
        if low.startswith(fold(rr) + "/"):
            return p[len(rr) + 1:]
    for n in names:
        i = low.rfind(f"/{fold(n)}/")
        if i >= 0:
            return p[i + len(n) + 2:]
        if low.startswith(f"{fold(n)}/"):
            return p[len(n) + 1:]
    return None


def _fold(s: str) -> str:
    return s.lower()


@lru_cache(maxsize=8)
def _real_roots(roots: tuple) -> tuple:
    """(root, its real path) for each root whose real path differs (a symlink in it): the
    walk stores a file under the root's real path."""
    out = []
    for r in roots:
        try:
            real = os.path.realpath(r)
        except (OSError, ValueError):
            continue
        if real != r:
            out.append((r, real))
    return tuple(out)


def library_rel(path: str) -> str:
    """The path under its library folder (vendor/pack/.../file), or the path itself."""
    roots, names = library()
    got = _rel(path or "", roots, names)
    return path if got is None else got


def in_library(path: str) -> bool:
    roots, names = library()
    return _rel(path or "", roots, names) is not None


def library_root_of(path: str) -> str | None:
    """The library folder a path sits in (for a bare folder name, the path up to it)."""
    roots, names = library()
    p = path or ""
    for r in roots:
        if p.startswith(r + "/"):
            return r
    for n in names:
        i = p.rfind(f"/{n}/")
        if i >= 0:
            return p[:i + len(n) + 1]
    if _windows_shaped(p):
        low = _fold(p.replace("\\", "/"))
        for r in roots:
            if low.startswith(_fold(r.replace("\\", "/").rstrip("/")) + "/"):
                return r
    for r, real in _real_roots(roots):
        if p.startswith(real + "/"):
            return r
    # the same folder spelled another way (case on APFS/FAT, a symlink, backslashes): compare
    # what the folders are on disk
    if roots and p:
        from .safety import _contains
        for r in roots:
            if _contains(r, p) and not _contains(p, r):
                return r
    return None


def library_record() -> dict:
    """What a build's manifest records of the library it read (`library`): the configured
    folders and bare folder names. A later build into the same master compares it
    (master_library_problem)."""
    roots, names = library()
    return {"folders": list(roots), "names": list(names)}


def _same_library(recorded: dict, roots, names) -> bool:
    """Whether a recorded library and the configured one share a folder (as spelled, or the
    same folder on disk) or a folder name: one library that gained or lost a folder is the
    same library; one that shares nothing is another."""
    rf = {_expand(f).rstrip("/") for f in recorded.get("folders") or () if isinstance(f, str)}
    rn = {n for n in recorded.get("names") or () if isinstance(n, str)}
    if not rf and not rn:
        return True                             # nothing recorded to compare
    cf, cn = set(roots), set(names)
    if rf & cf or rn & cn:
        return True
    real = lambda fs: {os.path.realpath(f) for f in fs}
    if real(rf) & real(cf):
        return True
    base = lambda fs: {os.path.basename(f) for f in fs}
    return bool(base(rf) & cn or base(cf) & rn)


def master_library_problem(master) -> tuple[str, str] | None:
    """("fail", why) when the master at this folder was built from another library than the
    configured one (its manifest's `library` shares no folder with it): a build would replace
    another library's master. ("old", why) for a master whose manifest predates the record.
    None when there's no master, or it's this library's."""
    import json
    mp = Path(master) / "manifest.json"
    try:
        doc = json.loads(mp.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or not doc.get("categories"):
        return None
    rec = doc.get("library")
    if not isinstance(rec, dict):
        return ("old", f"the master at {master} was built before Fourier recorded which library a "
                       f"master holds; the next build records it")
    try:
        roots, names = library()
    except PlacesError:
        return None                            # the config's own error says so
    if not roots and not names or _same_library(rec, roots, names):
        return None
    was = ", ".join([*(rec.get("folders") or ()), *(rec.get("names") or ())]) or "?"
    now = ", ".join([*roots, *names])
    return ("fail", f"the master at {master} holds another library's build (from {was}; this "
                    f"config's library is {now}): a build would replace it. Give this config its "
                    f"own master ([output] master in fourier.toml, or --out), or move that master "
                    f"aside if it should go")


def _output(key: str) -> str | None:
    where, doc = _config()
    out = doc.get("output") or {}
    if not isinstance(out, dict):
        raise PlacesError(f"{where}: [output] must be a table")
    extra = set(out) - {"master", "renders", "publish"}
    if extra:
        raise PlacesError(f"{where}: [output]: unknown keys {sorted(extra)} (master, renders, publish)")
    return out.get(key)


def master_dir() -> str:
    """The curated master: $FOURIER_CURATED_DIR, [output] master, or ~/Music/FourierCurated."""
    return _expand(os.environ.get("FOURIER_CURATED_DIR") or _output("master") or DEFAULT_MASTER)


def renders_dir(device_id: str | None = None) -> str:
    """Where device renders go: [output] renders, else FourierRenders beside the master."""
    base = _output("renders")
    base = _expand(base) if base else os.path.join(os.path.dirname(master_dir()), "FourierRenders")
    return os.path.join(base, device_id) if device_id else base


def publish_root() -> str:
    """[output] publish, or ~/Music/Fourier: releases/ and devices/ go under it."""
    return _expand(_output("publish") or DEFAULT_PUBLISH)


def releases_root() -> str:
    return os.path.join(publish_root(), "releases")


def lock_dir() -> Path:
    env = os.environ.get("FOURIER_LOCK_DIR")
    return Path(env) if env else Path(publish_root()) / "devices"


def devices() -> list[str]:
    """The devices fourier.toml renders for (`devices = [...]`), or []."""
    where, doc = _config()
    return _entries(where, doc.get("devices", []), "devices")


def sononym_db() -> Path | None:
    """sononym_db in fourier.toml, else sononym.db in the first library folder that has one."""
    where, doc = _config()
    if doc.get("sononym_db"):
        return Path(_expand(doc["sononym_db"]))
    for r in library_roots():
        if (r / "sononym.db").exists():
            return r / "sononym.db"
    return None
