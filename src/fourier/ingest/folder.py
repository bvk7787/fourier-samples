"""Copy a folder of audio into the library: `fourier tools import-folder`.

Walks a source folder (never modified) and writes into <dest>/, keeping its layout:

  - .wav / .aif / .aiff / .aifc / .flac   copied verbatim when soundfile can read them
  - .ogg / .oga / .opus / .mp3 / .caf     decoded to WAV (those soundfile reads here;
                                          ingest/formats.py, the list the walk uses)
  - a file whose extension an enabled preparer names (fourier/preparers.py) and that the
    preparer claims is loaded by it and written as WAV (as "<name>.<ext>.wav" when the
    folder also has a "<name>.wav" or similar)

A file in an open format that soundfile can't read is counted as unreadable and left out;
anything else (documents, presets, images) is ignored. Re-runnable: incremental by default
(skips files already present and not newer at the source); clean=True wipes and reloads a
destination this importer made (it carries a .fourier-import marker), nothing else. The
source and destination may not overlap, and the destination may not be a library root.
max_mb caps the size of a source file.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .formats import COPY_EXTS, convert_exts

CONVERT_EXTS = convert_exts()
MARKER = ".fourier-import"


def check_paths(src, dest) -> tuple[Path, Path]:
    """Resolved (src, dest), refusing overlapping folders, Fourier's own folders, releases and a
    library root as the destination. Folders are compared on disk (fourier/safety.py), so a
    differently cased or symlinked spelling of the same folder is caught."""
    from ..safety import UnsafePath, _contains, _norm, _overlaps, _places, check_not_personal, release_at
    s = Path(src).expanduser().resolve()
    d = Path(dest).expanduser().resolve()
    if _overlaps(s, d):
        raise ValueError(f"the source ({s}) and destination ({d}) can't overlap")
    check_not_personal(d, "the destination")
    roots = []
    for label, place in _places(master=True):          # never into Fourier's own folders
        if label == "a library folder":
            roots.append(place)
        elif _overlaps(d, place):
            raise ValueError(f"the destination ({d}) can't be, hold or sit in {label} ({place})")
    if release_at(d) is not None:
        raise ValueError(f"the destination ({d}) is or sits in a release: releases never change")
    try:
        from ..places import library
        names = {_norm(n, True) for n in library()[1]}
    except Exception as e:
        raise UnsafePath(f"can't read the library setting ({e}): fix fourier.toml and run again") from None
    if any(_contains(d, r) for r in roots) or _norm(d.name, True) in names:
        raise ValueError(f"{d} is a library folder (or holds one): import into a folder inside it")
    return s, d


def prepare_dest(dest: Path, clean: bool, adopt: bool = False) -> None:
    """Make the destination, marked as an import's. A folder that already holds files and
    no marker is someone's own: refused, unless adopt (the user says it's an import folder,
    so a later clean may empty it). With clean, empty a folder an import made."""
    marked = (dest / MARKER).exists()
    if dest.exists() and not marked and any(dest.iterdir()):
        if not adopt:
            raise ValueError(f"{dest} already holds files and isn't an import folder: pick a new "
                             f"folder, or pass --adopt if it only holds earlier imports (a later "
                             f"--clean would then empty it)")
        if clean:
            raise ValueError("--clean with --adopt would empty a folder that wasn't an import's; "
                             "adopt it first without --clean")
    if clean and dest.exists() and marked:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MARKER).touch()


def _readable(path) -> bool:
    import soundfile as sf
    try:
        sf.info(str(path))
        return True
    except Exception:
        return False


def new_stats() -> dict:
    return dict(copied=0, converted=0, prepared=0, up_to_date=0, unreadable=0, too_big=0, failed=0)


def import_file(src: Path, out_base: Path, rel: Path, st: dict, *, preps=(), convert_ogg=True,
                clean=False, cap=0) -> bool:
    """One source file into out_base/rel (as .wav when it's converted or prepared). True if
    it was written."""
    import soundfile as sf

    from ..preparers import claim
    ext = src.suffix.lower()
    prep = claim(src, preps) if preps else None
    if ext not in COPY_EXTS and ext not in CONVERT_EXTS and prep is None:
        return False
    if cap and src.stat().st_size > cap:
        st["too_big"] += 1
        return False
    to_wav = prep is not None or (ext in CONVERT_EXTS and convert_ogg)
    out_rel = rel.with_suffix(".wav") if to_wav else rel
    if prep is not None and any(src.with_suffix(e).exists() for e in COPY_EXTS | CONVERT_EXTS):
        out_rel = rel.with_name(rel.name + ".wav")          # hit.toy beside hit.wav: hit.toy.wav
    out = out_base / out_rel
    if not clean and out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        st["up_to_date"] += 1
        return False
    if prep is None and not _readable(src):
        st["unreadable"] += 1
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if prep is not None:
            sig, sr, subtype = prep.load(str(src))
            sf.write(str(out), sig, sr, subtype=subtype)
            st["prepared"] += 1
        elif ext in CONVERT_EXTS and convert_ogg:
            data, sr = sf.read(str(src))
            sf.write(str(out), data, sr, subtype="PCM_24")
            st["converted"] += 1
        else:
            shutil.copy2(src, out)
            st["copied"] += 1
        return True
    except Exception:
        st["failed"] += 1
        return False


def summary(st: dict) -> str:
    return (f"{st['copied']} copied, {st['converted']} ogg->wav, {st['prepared']} prepared->wav, "
            f"{st['up_to_date']} up-to-date, {st['unreadable']} unreadable, "
            f"{st['too_big']} over-cap, {st['failed']} failed")


def ingest_folder(src_root, dest_root, convert_ogg=True, clean=False, max_mb=0, prepare=True,
                  preparers=None, adopt=False, log=print):
    """Copy the audio under src_root into dest_root. Returns a summary dict.
    preparers: the preparers to offer files to (default: the ones fourier.toml enables)."""
    from ..preparers import check, enabled
    preps = (enabled() if preparers is None else list(preparers)) if prepare else []
    for p in preps:
        check(p)
    if not Path(src_root).expanduser().is_dir():
        raise FileNotFoundError(f"{src_root} is not a folder")
    src, dest = check_paths(src_root, dest_root)
    prepare_dest(dest, clean, adopt)
    cap = int(max_mb * 1_000_000) if max_mb else 0
    st = new_stats()
    for f in sorted(src.rglob("*")):
        if f.is_file() and not f.name.startswith(".") and not f.is_symlink():
            import_file(f, dest, f.relative_to(src), st, preps=preps, convert_ogg=convert_ogg,
                        clean=clean, cap=cap)
    log(f"DONE folder import: {src} -> {dest}\n  {summary(st)}")
    return st
