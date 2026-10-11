"""Device path locks: once a file is on a device, its on-card path never changes.

M8 songs and Digitakt II projects reference samples by path (M8 manual p.46: bundle a song
before moving samples; Digitakt II p.29: a sample deleted from the +Drive drops out of every
preset and pattern). A render is regenerable, but the paths it produces are not: after a
release goes to the device, those names are a contract.

The lock (``<publish>/devices/<device_id>.lock.json``, next to the releases) maps
each (category, source sample) to the path it got on the device, plus the card_dir it was
copied to. A family rename or re-homing within a category keeps the device path; a move to
another category adds a new device file and leaves the old one in place.
Renders assign names in two passes: locked sources get their locked path back, verbatim;
new sources get fresh names that avoid every locked path (case-insensitively, since the
cards are FAT32/exFAT), including retired ones, so a name is never reused for other audio.
New names are assigned in sorted order, so the result doesn't depend on scan order.
A profile's files_per_folder (folder_limit) sends new files past a full folder into its
numbered siblings ("punchy-2", "punchy-3"), counting the files the render holds there
(locked paths kept, new ones placed so far; retired files don't count); locked paths stay
where they are, so a release's paths never move.
A key names the source by its absolute path, so a library that moved (another mount, a renamed
library, vendor or pack folder) would find none of its keys. A source the lock doesn't know
takes over the entry of a locked source gone from the master when they are the same master
file (same master path, same audio) or the same library path (vendor/pack/...), one to one,
and recording the render re-keys that entry to the new source (_follow_moves).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..places import lock_dir

LOCK_DIR = lock_dir()    # $FOURIER_LOCK_DIR, else <[output] publish>/devices (fourier/places.py)
LOCK_VERSION = 1


def norm_card_dir(card_dir: str) -> str:
    """'/Samples/Fourier/' and '/Samples/Fourier' are the same place on the card."""
    return (card_dir or "").rstrip("/")


@dataclass
class Lock:
    device: str
    card_dir: str
    files: dict[str, dict] = field(default_factory=dict)  # key -> {path, master, md5, release, locked}
    created: str = ""
    updated: str = ""
    root: str = ""   # device root folder the paths sit under (the profile's paths.root)
    fmt: dict = field(default_factory=dict)  # sample_rate / bit_depth / mono the audio was rendered at

    def taken(self) -> set[str]:
        return {v["path"].lower() for v in self.files.values()}


def lock_path(device_id: str, lock_dir: Path | None = None) -> Path:
    return (lock_dir or LOCK_DIR) / f"{device_id}.lock.json"


def load_lock(device_id: str, lock_dir: Path | None = None) -> Lock | None:
    p = lock_path(device_id, lock_dir)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return Lock(device=d["device"], card_dir=norm_card_dir(d.get("card_dir", "")),
                files=d.get("files", {}), created=d.get("created", ""),
                updated=d.get("updated", ""), root=d.get("root", ""), fmt=d.get("fmt", {}))


def save_lock(lock: Lock, lock_dir: Path | None = None) -> Path:
    p = lock_path(lock.device, lock_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    lock.created = lock.created or now
    lock.updated = now
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(fourier_device_lock=LOCK_VERSION, device=lock.device,
                                   card_dir=norm_card_dir(lock.card_dir), root=lock.root,
                                   fmt=lock.fmt,
                                   created=lock.created,
                                   updated=lock.updated, files=lock.files),
                              indent=1, sort_keys=True))
    os.replace(tmp, p)
    return p


@dataclass(frozen=True)
class MasterFile:
    key: str           # "<CATEGORY>|<source sample path>", else "master:<CAT/family/file>"
    master: str        # CAT/family/file.wav inside the master
    md5: str | None    # master file content hash (manifest out_md5)
    category: str
    family: str
    stem: str


def _file_md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def master_files(master_dir) -> list[MasterFile]:
    """Every WAV in the master (CATEGORY/family/file), keyed by source sample via manifest.json."""
    master = Path(master_dir)
    by_out = {}
    mf = master / "manifest.json"
    if mf.exists():
        from . import manifests
        man = manifests.read(mf)
        for cat, cd in man.get("categories", {}).items():
            for e in cd.get("entries", []):
                by_out[f"{cat}/{e['out']}"] = e
        for sname, sd in (man.get("sets") or {}).items():
            for e in sd.get("entries", []):
                by_out[f"{sname}/{e['out']}"] = e
    out = []
    for cat in sorted(d for d in master.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))):
        for fam in sorted(d for d in cat.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))):
            for wav in sorted(p for p in fam.iterdir() if p.suffix.lower() == ".wav"):
                rel = f"{cat.name}/{fam.name}/{wav.name}"
                e = by_out.get(rel, {})
                md5 = e.get("out_md5") or _file_md5(wav)  # no manifest entry: hash the audio
                # keyed per category: re-homing to another category is a new device file (the
                # old path stays on the card); the same source in two categories stays two files
                src = e.get("src")
                out.append(MasterFile(key=f"{cat.name}|{src}" if src else f"master:{rel}", master=rel,
                                      md5=md5, category=cat.name, family=fam.name,
                                      stem=wav.stem))
    return out


def _unique(rel: str, taken: set[str], limit_len: int | None, prefix_len: int,
            name_len: int | None = None) -> str:
    """rel, or rel with _2, _3 ... (trimming the stem to stay within the path limit, and
    within name_len for the file name when given)."""
    if rel.lower() not in taken:
        return rel
    head, name = rel.rsplit("/", 1) if "/" in rel else ("", rel)
    stem, ext = os.path.splitext(name)
    for k in range(2, 10_000):
        suffix = f"_{k}"
        s = stem
        if limit_len:
            room = limit_len - prefix_len - len(head) - (1 if head else 0) - len(ext) - len(suffix)
            s = stem[:max(1, room)]
        if name_len:
            s = s[:max(1, name_len - len(ext) - len(suffix))]
        cand = f"{head}/{s}{suffix}{ext}" if head else f"{s}{suffix}{ext}"
        if cand.lower() not in taken:
            return cand
    raise RuntimeError(f"could not find a free name for {rel}")


@dataclass
class Assignment:
    paths: dict[str, str]          # key -> device-relative path (under the device root)
    new: list[str]                 # keys given fresh names this time
    kept: list[str]                # keys whose locked path was reused
    changed: list[str]             # kept keys whose master audio differs from the locked md5
    retired: list[str]             # locked keys absent from this master (paths stay reserved)
    shortened: tuple = (0, 0)      # (family folders, file names) a render cut to fit its path limit
    rekeyed: dict[str, str] = field(default_factory=dict)  # kept key -> the lock key it took over

    def locked_key(self, key: str) -> str:
        """The lock's key for a kept key (another one when its source moved)."""
        return self.rekeyed.get(key, key)


def _folder_with_room(rel: str, held: dict[str, int], limit: int, name_len: int | None = None) -> str:
    """rel in its folder, or in the first numbered sibling of it ("fam-2", "fam-3", ...)
    holding fewer than `limit` files (held: lowercase folder -> files). A sibling's name
    stays within name_len."""
    if "/" not in rel:
        return rel
    head, name = rel.rsplit("/", 1)
    parent, last = head.rsplit("/", 1) if "/" in head else ("", head)
    folder, k = head, 1
    while held.get(folder.lower(), 0) >= limit:
        k += 1
        sfx = f"-{k}"
        stem = last[:max(1, name_len - len(sfx))] if name_len else last
        folder = f"{parent}/{stem}{sfx}" if parent else f"{stem}{sfx}"
    return f"{folder}/{name}"


def _src(key: str) -> str | None:
    return key.split("|", 1)[1] if "|" in key and not key.startswith("master:") else None


def _same_library_path(old_src: str, rel: str) -> bool:
    """Whether a locked source sits at `rel` (vendor/pack/...) under a library folder: the one
    it is in now when it's under a configured one, else any (a library that moved)."""
    from ..places import library_rel
    now = library_rel(old_src)
    return now == rel if now != old_src else old_src.endswith("/" + rel)


def _follow_moves(files: list[MasterFile], locked: dict, unmatched: list[str]) -> dict[str, str]:
    """{new key: lock key} for sources the lock doesn't know that are a locked source gone from
    the master, moved: the same master file (master path and audio, as builddiff's library
    moves), else the same category and library path; each one to one, else left new."""
    from ..places import library_rel
    out: dict[str, str] = {}
    left = [f for f in sorted(files, key=lambda f: f.master) if f.key not in locked]
    by_file: dict[tuple, list[str]] = {}
    for k in unmatched:
        e = locked[k]
        if e.get("master") and e.get("md5"):
            by_file.setdefault((e["master"], e["md5"]), []).append(k)
    for f in left:
        olds = by_file.get((f.master, f.md5))
        if f.key not in out and olds and len(olds) == 1:
            out[f.key] = olds.pop()
    taken = set(out.values())
    by_name: dict[tuple, list[str]] = {}
    for k in unmatched:
        src = _src(k)
        if src and k not in taken:
            by_name.setdefault((k.split("|", 1)[0], src.rsplit("/", 1)[-1]), []).append(k)
    for f in left:
        src = _src(f.key)
        if not src or f.key in out:
            continue
        rel = library_rel(src)
        if rel == src or "/" not in rel:      # outside the library, or no folder to tell it by
            continue
        olds = [k for k in by_name.get((f.category, rel.rsplit("/", 1)[-1]), ())
                if k not in taken and _same_library_path(_src(k) or "", rel)]
        if len(olds) == 1:
            out[f.key] = olds[0]
            taken.add(olds[0])
    return out


def assign(files: list[MasterFile], lock: Lock | None, fresh_name, limit_len: int | None = None,
           prefix_len: int = 0, name_len: int | None = None,
           folder_limit: int | None = None) -> Assignment:
    """Map master files to device paths. `fresh_name(f)` proposes a path for a new file.
    name_len: a "_2" that makes a name unique keeps it within this length. folder_limit: a
    new file goes to the first of its folder and its numbered siblings with room."""
    locked = lock.files if lock else {}
    taken = lock.taken() if lock else set()
    present = {f.key for f in files}
    rekeyed = _follow_moves(files, locked, [k for k in locked if k not in present]) if locked else {}
    paths, new, kept, changed = {}, [], [], []
    for f in files:
        k = rekeyed.get(f.key, f.key)
        if k in locked:
            paths[f.key] = locked[k]["path"]
            kept.append(f.key)
            if f.md5 and locked[k].get("md5") and f.md5 != locked[k]["md5"]:
                changed.append(f.key)
    held: dict[str, int] = {}
    if folder_limit:
        for k in set(kept):
            d = paths[k].rsplit("/", 1)[0].lower() if "/" in paths[k] else ""
            held[d] = held.get(d, 0) + 1
    for f in sorted((f for f in files if f.key not in locked and f.key not in rekeyed),
                    key=lambda f: f.master):
        if f.key in paths:  # same source twice in one master: first wins
            continue
        rel = fresh_name(f)
        if folder_limit:
            rel = _folder_with_room(rel, held, folder_limit, name_len)
        rel = _unique(rel, taken, limit_len, prefix_len, name_len)
        if folder_limit and "/" in rel:
            d = rel.rsplit("/", 1)[0].lower()
            held[d] = held.get(d, 0) + 1
        taken.add(rel.lower())
        paths[f.key] = rel
        new.append(f.key)
    moved = set(rekeyed.values())
    retired = sorted(k for k in locked if k not in present and k not in moved)
    return Assignment(paths, new, kept, changed, retired, rekeyed=rekeyed)


def record(lock: Lock | None, device_id: str, card_dir: str, files: list[MasterFile],
           a: Assignment, release: str | None, written: set[str] | None = None,
           root: str = "", fmt: dict | None = None) -> Lock:
    """Add this render's new assignments to the lock (existing entries are never rewritten; a
    moved source's entry moves to its new key as it was). Only keys in `written` (files
    actually in the image) are locked when given."""
    lock = lock or Lock(device=device_id, card_dir=norm_card_dir(card_dir), root=root,
                        fmt=dict(fmt or {}))
    if not lock.fmt and fmt:
        lock.fmt = dict(fmt)
    now = time.strftime("%Y-%m-%d")
    by_key = {f.key: f for f in files}
    for k, old in a.rekeyed.items():
        if old in lock.files and k not in lock.files:
            lock.files[k] = lock.files.pop(old)
    for k in a.new:
        if written is not None and k not in written:
            continue
        f = by_key[k]
        lock.files[k] = dict(path=a.paths[k], master=f.master, md5=f.md5, release=release, locked=now)
    return lock
