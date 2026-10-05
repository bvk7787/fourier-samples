"""Guards for the commands that delete or replace files: a folder Fourier may wipe or
overwrite has to be one it made, and never a library, a release, the home folder, a volume
root or another of Fourier's own places.

Places are compared by what they are on disk, not by how they're spelled: two paths are the
same folder when their nearest existing folders have the same (st_dev, st_ino) and the rest
of the path matches (ignoring case where the disk does, as APFS and FAT do by default). A
place that can't be looked up (a broken fourier.toml, say) stops the command: the guards
fail closed.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path, PurePath


class UnsafePath(ValueError):
    pass


OFFLINE_HINT = ('make the library folder available offline (Finder: "Keep Downloaded"; '
                'OneDrive: "Always keep on this device")')


class CloudOnlySources(UnsafePath):
    """Source files a build needs whose content a cloud drive hasn't put on this machine."""

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        shown = "\n  ".join(self.paths[:10])
        more = f"\n  ... and {len(self.paths) - 10:,} more" if len(self.paths) > 10 else ""
        super().__init__(f"{len(self.paths):,} source file(s) are cloud-only and couldn't be "
                         f"downloaded:\n  {shown}{more}\nTo build, {OFFLINE_HINT}.")

    def __reduce__(self):                 # crosses the build's process pool whole
        return (CloudOnlySources, (self.paths,))


def ensure_local(paths, log=None) -> int:
    """Before a build reads its sources: download the cloud-only ones among paths (one lstat
    each otherwise), and raise CloudOnlySources for any still not on this machine. Returns
    how many were cloud-only."""
    from . import platforms
    cloud = [p for p in dict.fromkeys(paths) if p and platforms.cloud_only(p)]
    if not cloud:
        return 0
    (log or (lambda m: None))(f"{len(cloud)} source file(s) are cloud-only; downloading them first")
    left = platforms.materialize(cloud, log)
    if left:
        raise CloudOnlySources(left)
    return len(cloud)


def _resolve(p) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


# ---------------------------------------------------------------------------
# identity: the same folder under another spelling (case, symlinks)
# ---------------------------------------------------------------------------
def _stat(p) -> tuple[int, int] | None:
    """(st_dev, st_ino) of an existing path, else None (also when the file system reports no
    inode numbers, so the comparison falls back to names)."""
    try:
        st = os.stat(p)
    except (OSError, ValueError):
        return None
    return (st.st_dev, st.st_ino) if st.st_ino else None


def _split(p: Path) -> tuple[Path, tuple[str, ...]]:
    """(nearest existing folder, the names below it that don't exist yet)."""
    tail: list[str] = []
    cur = p
    while _stat(cur) is None and cur.parent != cur:
        tail.append(cur.name)
        cur = cur.parent
    return cur, tuple(reversed(tail))


def _case_insensitive(d: Path) -> bool:
    """Whether names in folder d ignore case. Unknown (no name with letters to try): True,
    which makes the overlap checks refuse more, never less."""
    probes = [d]
    try:
        probes += [d / n for n in sorted(os.listdir(d))[:50]]
    except OSError:
        pass
    for x in probes:
        if x.name and x.name != x.name.swapcase():
            a, b = _stat(x), _stat(x.parent / x.name.swapcase())
            if a is not None:
                return a == b
    return True


def _norm(name: str, fold: bool) -> str:
    n = unicodedata.normalize("NFC", name)
    return n.casefold() if fold else n


def _contains(outer, inner) -> bool:
    """inner is outer, or inside it."""
    o0, to = _split(_resolve(outer))
    i0, ti = _split(_resolve(inner))
    oid = _stat(o0)
    if not to:                                    # outer exists: is it one of inner's folders?
        return oid is not None and oid in {_stat(x) for x in (i0, *i0.parents)}
    if oid is None or oid != _stat(i0) or len(ti) < len(to):
        return False
    fold = _case_insensitive(o0)
    return all(_norm(a, fold) == _norm(b, fold) for a, b in zip(to, ti))


def _overlaps(a, b) -> bool:
    return _contains(a, b) or _contains(b, a)


def _root_refused(r: PurePath, allow_drive_root: bool = False) -> bool:
    """r is a file-system root. A drive root (E:\\ on Windows) may be allowed as a card
    volume; "/" never is."""
    if r != type(r)(r.anchor):
        return False
    return not (allow_drive_root and r.drive)


# ---------------------------------------------------------------------------
# Fourier's own places
# ---------------------------------------------------------------------------
def published_record() -> Path:
    """Where `fourier publish` records each release it cut (any publish root, --to included)."""
    from .paths import home_path
    return home_path("releases.json")


def published_releases() -> list[Path]:
    f = published_record()
    if not f.exists():
        return []
    try:
        return [Path(e["path"]) for e in json.loads(f.read_text()).get("releases", [])]
    except Exception as e:
        raise UnsafePath(f"can't read {f} ({e}), which lists the releases Fourier published: "
                         f"fix or restore it first") from None


def record_release_path(path, version: str) -> None:
    """Remember a published release's folder, so no build or sync ever writes into it."""
    f = published_record()
    try:
        doc = json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        doc = {}
    p = str(_resolve(path))
    rels = [e for e in doc.get("releases", []) if e.get("path") != p]
    rels.append({"version": version, "path": p})
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"releases": rels}, indent=1))
    os.replace(tmp, f)


def _places(master: bool = False) -> list[tuple[str, Path]]:
    """(label, folder) for the library roots and Fourier's own folders, as configured: the
    Fourier home, the releases (the configured ones and every one publish recorded), the
    device locks, the renders (and the master, if asked). The library comes first, so a folder
    holding it and one of Fourier's own is refused for the library. Any lookup that fails
    raises UnsafePath: a guard that can't see a place doesn't pass."""
    try:
        from . import places as P
        from .paths import fourier_home
        out = [("a library folder", _resolve(r)) for r in P.library_roots()]
        out += [("the releases", _resolve(P.releases_root())), ("the device locks", _resolve(P.lock_dir())),
                ("the renders", _resolve(P.renders_dir()))]
        if master:
            out.append(("the master", _resolve(P.master_dir())))
        out.append(("the Fourier home", _resolve(fourier_home())))
    except UnsafePath:
        raise
    except Exception as e:
        raise UnsafePath(f"can't tell where Fourier's own folders are ({e}), so nothing is "
                         f"replaced or deleted: fix fourier.toml and run again") from None
    out += [("a release", _resolve(p)) for p in published_releases()]
    return out


def _library_names() -> tuple[str, ...]:
    try:
        from .places import library
        return library()[1]
    except Exception as e:
        raise UnsafePath(f"can't read the library setting ({e}): fix fourier.toml and run again") from None


def is_library_path(p) -> bool:
    """p is a library folder, inside one or holds one (bare library names included: p is or
    sits in a folder of that name)."""
    r = _resolve(p)
    if any(_overlaps(r, lib) for label, lib in _places() if label == "a library folder"):
        return True
    names = {_norm(n.strip("/\\"), True) for n in _library_names()}
    return any(_norm(part, True) in names for part in r.parts)


def check_not_personal(p, what: str, allow_drive_root: bool = False) -> Path:
    """Refuse the filesystem root, the home folder and anything holding it."""
    r = _resolve(p)
    if _root_refused(r, allow_drive_root) or _contains(r, _resolve("~")):
        raise UnsafePath(f"{what} can't be {r}: pick a folder of its own")
    return r


_RELEASE_NAME = re.compile(r"v\d+(\.partial|\.replaced)?( \d+)?")


def _manifest(d: Path) -> dict | None:
    m = d / "manifest.json"
    if not m.is_file():
        return None
    try:
        doc = json.loads(m.read_text())
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


def release_at(r: Path) -> Path | None:
    """The release r is, sits in or holds (one level down, or a publish root's releases/):
    a folder whose manifest says "release", or a vN folder beside a LATEST.txt."""
    for x in (r, *r.parents):
        if not x.exists():
            continue
        if "release" in (_manifest(x) or {}):
            return x
        if _RELEASE_NAME.fullmatch(x.name) and (x.parent / "LATEST.txt").exists():
            return x
    for x in (r, r / "releases"):
        if (x / "LATEST.txt").exists():
            return x
    return None


def check_clear(p, what: str, extra: tuple = (), master: bool = False) -> Path:
    """p is clear of the home folder, a root, Fourier's own places, the library and every
    release."""
    r = check_not_personal(p, what)
    for label, place in _places(master) + [(lbl, _resolve(x)) for lbl, x in extra if x]:
        if _overlaps(r, place):
            raise UnsafePath(f"{what} ({r}) can't be, hold or sit in {label} ({place})")
    if is_library_path(r):
        raise UnsafePath(f"{what} ({r}) can't be, hold or sit in a library folder")
    rel = release_at(r)
    if rel is not None:
        raise UnsafePath(f"{what} ({r}) can't be, hold or sit in a release ({rel}): releases "
                         f"never change. Build into the working master instead")
    return r


def check_master_dir(out_dir, *, extra: tuple = ()) -> Path:
    """A folder a build may replace: new, empty, or a previous Fourier master (manifest.json
    with fourier_manifest); never a library, a release, the Fourier home, the home folder or
    a root."""
    r = check_clear(out_dir, "the master", extra)
    if r.exists() and not r.is_dir():
        raise UnsafePath(f"the master ({r}) is a file, not a folder: pick a folder")
    if r.exists() and any(r.iterdir()):
        if "fourier_manifest" not in (_manifest(r) or {}):
            raise UnsafePath(f"{r} isn't empty and isn't a Fourier master (no manifest.json): "
                             f"a build replaces its master's contents, so pick a new folder")
    return r


# ---------------------------------------------------------------------------
# what a build may remove from a master
# ---------------------------------------------------------------------------
LIVE_INFO = "Ableton Folder Info"       # Live's tag sidecars (packs/ratings.py writes them too)
REVIEW_DIR = "_REVIEW"                  # packs/review.py
REVIEW_INDEX = "_review.json"


def master_known_files(root) -> set[str]:
    """Every file Fourier made in a master (relative, "/"-separated): the manifest's entries
    and sets, the files a build writes beside them, and the review queue."""
    root = Path(root)
    man = _manifest(root)
    if man is None:
        raise UnsafePath(f"can't read {root / 'manifest.json'}, so there's no telling Fourier's "
                         f"files from yours: restore it, or move the folder aside")
    known = {"manifest.json", "CHANGELOG.md", "loops.csv"}
    for sect in ("categories", "sets"):
        for cat, cd in (man.get(sect) or {}).items():
            known.add(f"{cat}/_manifest.json")
            for e in (cd or {}).get("entries", []):
                if e.get("out"):
                    known.add(f"{cat}/{e['out']}")
    try:
        idx = json.loads((root / REVIEW_DIR / REVIEW_INDEX).read_text())
        known.add(f"{REVIEW_DIR}/{REVIEW_INDEX}")
        known |= {f"{REVIEW_DIR}/{n}" for n in idx.get("items", {})}
    except (OSError, ValueError, AttributeError):
        pass
    return known


def unknown_master_files(root) -> list[str]:
    """Files in a master that Fourier didn't make (a build would delete them). Live's own
    files are Fourier's to replace: "Ableton Folder Info" sidecars and a known file's .asd
    analysis; so are dotfiles (.DS_Store, AppleDouble) and the review queue being built."""
    root = Path(root)
    if not root.is_dir():
        return []
    fold = _case_insensitive(root)
    known = {_norm(k, fold) for k in master_known_files(root)}
    out = []
    for dp, dns, fns in os.walk(root):
        rel_dir = Path(dp).relative_to(root)
        dns[:] = [d for d in dns if not d.startswith(".") and d != LIVE_INFO
                  and not (rel_dir == Path(".") and d == REVIEW_DIR + ".new")]
        for f in fns:
            if f.startswith("."):
                continue
            rel = (rel_dir / f).as_posix()
            k = _norm(rel, fold)
            if k in known or (k.endswith(".asd") and k[:-4] in known):
                continue
            out.append(rel)
    return sorted(out)


def check_master_contents(root, show: int = 10) -> None:
    """Refuse to rebuild a master holding files Fourier didn't make: a build replaces the
    master's contents, and .prev (its only other copy) is replaced by the next build."""
    strange = unknown_master_files(root)
    if strange:
        more = f"\n  ... and {len(strange) - show} more" if len(strange) > show else ""
        raise UnsafePath(
            f"{root} holds {len(strange)} file(s) Fourier didn't make, and a build would delete "
            f"them:\n  " + "\n  ".join(strange[:show]) + more +
            "\nMove them out of the master (keep your own work in a folder of its own), then "
            "build again.")


# ---------------------------------------------------------------------------
# card sync
# ---------------------------------------------------------------------------
CARD_MARK = ".fourier-card"     # in the card folder: this card's id (sync skips dotfiles)


def card_record(device_id: str) -> Path:
    """What earlier `fourier sync`s of a device copied, per card, in the Fourier home."""
    from .paths import home_path
    return home_path("cards", f"{device_id}.json")


def card_id(dest, create: bool = False) -> str | None:
    """The id Fourier wrote on this card (in <card folder>/.fourier-card); with create, one
    is written when there's none."""
    f = Path(dest) / CARD_MARK
    try:
        v = f.read_text().strip()
        if v:
            return v
    except OSError:
        pass
    if not create:
        return None
    import uuid
    v = uuid.uuid4().hex
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(v + "\n")
    return v


def _card_dir(dest, volume) -> str:
    try:
        return "/" + _resolve(dest).relative_to(_resolve(volume)).as_posix()
    except ValueError:
        return str(_resolve(dest))


def _load_cards(device_id: str) -> dict:
    try:
        doc = json.loads(card_record(device_id).read_text())
    except Exception:
        return {}
    cards = doc.get("cards") if isinstance(doc, dict) else None
    return cards if isinstance(cards, dict) else {}     # the old one-card format isn't trusted


def synced_files(device_id: str, dest=None, volume=None) -> set[str]:
    """The files earlier syncs copied into this card folder: trusted only for the same card
    (its id) and the same card folder."""
    if not device_id or dest is None or volume is None:
        return set()
    cid = card_id(dest)
    e = _load_cards(device_id).get(cid) if cid else None
    if not e or e.get("card_dir") != _card_dir(dest, volume):
        return set()
    return set(e.get("files", []))


def render_files(render_dir) -> set[str]:
    render_dir = Path(render_dir)
    return {p.relative_to(render_dir).as_posix() for p in render_dir.rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(render_dir).parts)}


def record_sync(device_id: str, render_dir, dest, volume, merge: bool = True) -> None:
    """Record the render's files as Fourier's on this card. merge keeps what earlier syncs
    recorded (a sync without --delete leaves those files on the card); a finished --delete
    sync passes merge=False. Called before copying too, so a copy that stops half way
    leaves every file it wrote recorded."""
    cid = card_id(dest, create=True)
    files = render_files(render_dir) | (synced_files(device_id, dest, volume) if merge else set())
    cards = _load_cards(device_id)
    try:
        st_dev = os.stat(volume).st_dev
    except OSError:
        st_dev = None
    cards[cid] = {"dest": str(_resolve(dest)), "card_dir": _card_dir(dest, volume),
                  "volume": str(volume), "st_dev": st_dev, "render": str(render_dir),
                  "files": sorted(files)}
    f = card_record(device_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fourier_card_record": 2, "cards": cards}, indent=0))
    os.replace(tmp, f)


def _no_card_folder(vol) -> UnsafePath:
    return UnsafePath("the device profile has no card folder (card_dir), so sync would copy "
                      f"into the volume root {vol}; sync is for devices with a card folder")


def _card_layout(d: PurePath, vol: PurePath) -> None:
    """The card folder is strictly inside the volume (by name; check_card_dest also compares
    them on disk)."""
    if d == vol:
        raise _no_card_folder(vol)
    if vol not in d.parents:
        raise UnsafePath(f"the card folder ({d}) isn't inside the card volume ({vol})")


def check_card_dest(dest, volume, render_dir, delete: bool, device_id: str | None = None) -> Path:
    """A card folder `fourier sync` may copy into (and prune with --delete): inside a volume
    that isn't the file-system root or the home folder (a drive root like E:\\ is fine), clear
    of Fourier's own places, the master, the releases and the library, and, for --delete,
    holding no file that isn't in this render or recorded by an earlier sync to this same
    card (card_record): anything else is someone's own and is never deleted."""
    vol = check_not_personal(volume, "the card volume", allow_drive_root=True)
    d = _resolve(dest)
    _card_layout(d, vol)
    if _contains(d, vol):
        raise _no_card_folder(vol)
    check_clear(d, "the card folder", master=True)
    if _overlaps(d, _resolve(render_dir)):
        raise UnsafePath("the card folder and the render overlap")
    if delete and d.exists():
        render = Path(render_dir)
        known = synced_files(device_id, d, vol)
        strange = []
        for p in d.rglob("*"):
            rel = p.relative_to(d)
            if p.is_dir() or any(part.startswith(".") for part in rel.parts):
                continue
            if not (render / rel).exists() and rel.as_posix() not in known:
                strange.append(rel.as_posix())
        if strange:
            strange.sort()
            raise UnsafePath(f"--delete would remove files neither this render nor an earlier sync "
                             f"to this card made, in {d}: {', '.join(strange[:10])}"
                             f"{' ...' if len(strange) > 10 else ''}. "
                             f"Move them off the card folder first, or sync without --delete")
    return d
