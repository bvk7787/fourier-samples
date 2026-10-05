"""The library walk: the audio files under a library folder, for `fourier tools scan` when
Sononym doesn't index the library (or with `--walk`, beside it).

- Every file with a readable extension (ingest/formats.py) under the folder; dot files and
  dot folders are left out ("._" copies, .Trash), and so are Fourier's own folders (the
  master, renders, releases) when they sit inside it. A file with another extension is
  counted by extension, so the scan can say what it skipped.
- A folder symlink is followed when it points outside the library folder (a pack kept on
  another drive) and its files are stored under the library folder, where the link puts
  them, so their vendor/pack path is the one the library shows. One whose target isn't
  there (a drive that isn't plugged in) is recorded in `unavailable`: its samples are
  marked missing and the scan says so, as for a missing library folder. One that points inside it
  is left to the walk of its target, so a file is one sample. Every real folder is listed
  once (by device and inode), so a link back up the tree can't loop. A file symlink is
  stored at its target, as a walk always stored it.
- A folder whose modification time is what it was at the last walk isn't listed again: its
  listing is reused (table walked_dirs), on file systems that change a folder's time
  whenever an entry is added, removed or renamed (platforms.dir_mtimes_reliable). A folder
  changed in the last few seconds is listed again next time, since its time may not have
  moved yet. The result is the same as listing every folder.

A file is stored at the library folder's real path joined with its path under it (for a
file reached through no symlink, the same as resolving it), and its path under the folder
is "/"-separated on every system (the library path curation reads).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .formats import AUDIO_EXTS

# a listing older than this is trusted: a folder's time may not have moved yet for a
# change made within the file system's time resolution
_SETTLE_NS = 2_000_000_000
# what a stored listing depends on besides the folder: the extensions it records
_LISTING_VERSION = "2:" + ",".join(AUDIO_EXTS)


@dataclass
class Found:
    key: str        # the path the database stores
    rel: str        # the path under the library folder, "/"-separated
    path: str       # where to read it


@dataclass
class Walk:
    root: str
    real_root: str
    files: list[Found] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)      # extension -> files not read
    links_followed: int = 0         # folder symlinks followed out of the library folder
    links_inside: int = 0           # ...pointing inside it (walked at their target)
    loops: int = 0                  # folders reached a second time (a link back up the tree)
    broken_links: int = 0
    # folder symlinks whose target isn't there (a drive that isn't plugged in): (the key
    # prefix of its samples, the link, its target). Reported like a missing library folder.
    unavailable: list = field(default_factory=list)
    unlistable: list[str] = field(default_factory=list)    # key prefixes of folders it couldn't list
    own_skipped: list[str] = field(default_factory=list)   # key prefixes of Fourier's own folders
    listed: int = 0
    reused: int = 0
    cache_put: dict = field(default_factory=dict)          # path -> (mtime_ns, listing json)
    cache_seen: set = field(default_factory=set)


def _ext(name: str) -> str:
    e = os.path.splitext(name)[1].lower()
    return e if e else "(none)"


_FILE_EXT_RE = re.compile(r"^\.[a-z0-9]{1,5}$")


def _folder_named(name: str) -> bool:
    """Named like a folder, not a file: no extension of a file's shape (".wav", ".pdf",
    ".nki"); "Pack Vol. 2" or "Drums 1.5" are folders."""
    e = os.path.splitext(name)[1].lower()
    return not e or not _FILE_EXT_RE.match(e) or e[1:].isdigit()


def _list(here: str, walk: Walk) -> dict | None:
    """What a folder holds: sub folders ("d"), folder symlinks ("l"), audio files ("f"),
    audio file symlinks ("s"), the other files' extensions ("x"), symlinks whose target is
    gone ("u": named like a folder, without an audio extension; "b": the rest, counted)."""
    out = {"d": [], "l": [], "f": [], "s": [], "u": [], "x": {}, "b": 0}
    try:
        with os.scandir(here) as it:
            entries = list(it)
    except OSError:
        return None
    for e in entries:
        if e.name.startswith("."):
            continue
        try:
            if e.is_symlink():
                if e.is_dir():
                    out["l"].append(e.name)
                elif e.is_file():
                    if _ext(e.name) in AUDIO_EXTS:
                        out["s"].append(e.name)
                    else:
                        out["x"][_ext(e.name)] = out["x"].get(_ext(e.name), 0) + 1
                elif _folder_named(e.name):     # a linked folder that isn't there
                    out["u"].append(e.name)
                else:                    # a link to a file that's gone
                    out["b"] += 1
            elif e.is_dir(follow_symlinks=False):
                out["d"].append(e.name)
            elif e.is_file(follow_symlinks=False):
                if _ext(e.name) in AUDIO_EXTS:
                    out["f"].append(e.name)
                else:
                    out["x"][_ext(e.name)] = out["x"].get(_ext(e.name), 0) + 1
        except OSError:
            continue
    for k in ("d", "l", "f", "s", "u"):
        out[k].sort()
    return out


def _join(base: str, rel: str) -> str:
    return os.path.join(base, *rel.split("/")) if rel else base


def walk(root, exts, *, cache: dict | None = None, own: set | None = None,
         reliable=None, now_ns: int | None = None) -> Walk:
    """The audio files under root with one of exts (lowercase, with the dot).
    cache: {folder path: (mtime_ns, listing json)} from earlier walks; own: (st_dev, st_ino)
    of Fourier's own folders, left out; reliable(path) -> bool: whether a folder's time
    can be trusted there (default platforms.dir_mtimes_reliable)."""
    from .. import platforms
    reliable = reliable or platforms.dir_mtimes_reliable
    cache = cache if cache is not None else {}
    own = own or set()
    now_ns = now_ns if now_ns is not None else time.time_ns()
    root_s = os.path.normpath(os.path.expanduser(str(root)))
    real_root = os.path.realpath(root_s)
    w = Walk(root=root_s, real_root=real_root)
    exts = {e.lower() for e in exts}
    visited: set = set()
    trust_root = reliable(real_root)
    stack = [(root_s, "", trust_root, False)]
    while stack:
        here, rel, trust, via_link = stack.pop()
        key_prefix = _join(real_root, rel)
        try:
            st = os.stat(here)
        except OSError:
            w.unlistable.append(key_prefix)
            continue
        ident = (st.st_dev, st.st_ino)
        if ident in own:
            w.own_skipped.append(key_prefix)
            continue
        if ident in visited and (st.st_dev or st.st_ino):
            w.loops += 1
            continue
        visited.add(ident)
        w.links_followed += via_link
        hit = cache.get(here) if trust else None
        listing = None
        if hit is not None and hit[0] == st.st_mtime_ns:
            try:
                doc = json.loads(hit[1])
                if doc.get("v") == _LISTING_VERSION:
                    listing = doc
            except (TypeError, ValueError):
                listing = None
        if listing is not None:
            w.reused += 1
        else:
            listing = _list(here, w)
            if listing is None:
                w.unlistable.append(key_prefix)
                continue
            w.listed += 1
            listing["v"] = _LISTING_VERSION
            if trust and st.st_mtime_ns < now_ns - _SETTLE_NS:
                w.cache_put[here] = (st.st_mtime_ns, json.dumps(listing, sort_keys=True))
        w.cache_seen.add(here)
        w.broken_links += listing.get("b", 0)
        for ext, n in listing.get("x", {}).items():
            w.skipped[ext] += n
        sub = rel + "/" if rel else ""
        for name in listing.get("f", ()):
            ext = _ext(name)
            if ext not in exts:
                w.skipped[ext] += 1
                continue
            r = sub + name
            w.files.append(Found(key=_join(real_root, r), rel=r, path=os.path.join(here, name)))
        for name in listing.get("s", ()):
            ext = _ext(name)
            if ext not in exts:
                w.skipped[ext] += 1
                continue
            p = os.path.join(here, name)
            w.files.append(Found(key=os.path.realpath(p), rel=sub + name, path=p))
        for name in reversed(listing.get("d", ())):
            stack.append((os.path.join(here, name), sub + name, trust, False))
        # folder links, and links that weren't there when the folder was listed (a drive
        # plugged back in since: a folder's time doesn't change when a link's target does)
        for name in reversed(sorted(set(listing.get("l", ())) | set(listing.get("u", ())))):
            p = os.path.join(here, name)
            try:
                target = os.path.realpath(p)
            except (OSError, ValueError):
                w.broken_links += 1
                continue
            if not os.path.isdir(target):        # gone since the listing (a drive unplugged)
                w.unavailable.append((_join(real_root, sub + name), p, target))
                continue
            if target == real_root or target.startswith(real_root.rstrip(os.sep) + os.sep):
                w.links_inside += 1
                continue
            stack.append((p, sub + name, reliable(target), True))
    # one sample per stored path (a file symlink beside its target), in path order: new
    # samples get their ids in this order, whatever order the file system lists them in
    seen, files = set(), []
    for f in sorted(w.files, key=lambda f: _sort_key(f.rel)):
        if f.key not in seen:
            seen.add(f.key)
            files.append(f)
    w.files = files
    return w


def _sort_key(rel: str) -> tuple:
    """Path order as pathlib sorts paths under one folder: by parts (case folded on
    Windows, as PureWindowsPath compares)."""
    parts = rel.split("/")
    return tuple(p.lower() for p in parts) if os.name == "nt" else tuple(parts)


def own_folders() -> set:
    """(st_dev, st_ino) of Fourier's own output folders that exist: the master and its
    .next / .prev, the renders and the publish root (releases, device locks)."""
    from .. import places
    out = set()
    try:
        master = places.master_dir()
        cands = [master, master + ".next", master + ".prev", places.renders_dir(),
                 places.publish_root()]
    except Exception:
        return out
    for c in cands:
        try:
            st = os.stat(c)
        except OSError:
            continue
        out.add((st.st_dev, st.st_ino))
    return out


def under(path: str, prefixes) -> bool:
    """Whether path is one of prefixes or inside one."""
    for p in prefixes:
        if path == p or path.startswith(p.rstrip(os.sep) + os.sep):
            return True
    return False


def root_state(root) -> str | None:
    """Why a library folder can't be walked: "missing" (not there: an unmounted drive, a
    renamed folder), "not a folder", "empty" (nothing in it: a drive's mount point with the
    drive not mounted) or "unreadable"; None when it's fine."""
    p = Path(os.path.expanduser(str(root)))
    if not p.exists():
        return "missing"
    if not p.is_dir():
        return "not a folder"
    try:
        with os.scandir(p) as it:
            if not any(not e.name.startswith(".") for e in it):
                return "empty"
    except OSError:
        return "unreadable"
    return None
