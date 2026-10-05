"""
Importer: sync Sononym DB into the fourier SQLite DB.

Handles both:
  - Sononym import (preferred path - rich metadata already computed)
  - Filesystem scan fallback (for samples Sononym hasn't indexed)
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
import logging
import os
import time
from pathlib import Path
from typing import Callable

from sqlalchemy import text
from sqlalchemy.orm import Session

from .. import platforms
from ..db.models import Sample, SononymMeta
from ..db.session import session_scope
from .sononym import SononymAsset, SononymReader

log = logging.getLogger(__name__)


def _quick_hash(path: Path, nbytes: int = 8192) -> str:
    """SHA256 of first nbytes of file - fast identity check."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def _upsert_sample_from_sononym(
    session: Session,
    asset: SononymAsset,
    library_root: Path,
    counts: dict | None = None,
) -> tuple[Sample, bool]:
    """
    Insert or update a Sample + SononymMeta from a SononymAsset.
    Returns (sample, created) where created=True if new row inserted. A cloud-only file
    (platforms.cloud_only) isn't hashed, since reading it would download it: counted in
    counts["cloud_only"], it's hashed by a later scan once it's on this machine.
    """
    abs_path = str(asset.abs_path)

    existing = session.query(Sample).filter_by(path=abs_path).first()
    created = existing is None

    if existing is None:
        sample = Sample(path=abs_path)
        session.add(sample)
    else:
        sample = existing

    # Always refresh these in case file changed
    sample.rel_path = asset.rel_path
    sample.filename = asset.basename
    sample.file_size_bytes = asset.file_size
    sample.modified_at = asset.modtime

    # Populate content hash on first ingest (or if missing) — used for dedup
    if sample.file_hash is None:
        try:
            p = asset.abs_path
            if p.exists():
                if platforms.cloud_only(p):
                    if counts is not None:
                        counts["cloud_only"] = counts.get("cloud_only", 0) + 1
                else:
                    sample.file_hash = _quick_hash(p)
        except Exception:
            pass
    sample.duration_s = asset.duration_s
    sample.sample_rate = asset.sample_rate
    sample.channels = asset.channels
    sample.bit_depth = asset.bit_depth
    sample.file_format = asset.file_type
    sample.is_favorite = asset.is_favorite
    sample.is_hidden = asset.is_hidden

    # Upsert SononymMeta
    if sample.sononym is None:
        meta = SononymMeta(sample=sample)
        session.add(meta)
    else:
        meta = sample.sononym

    meta.sononym_asset_id = asset.sononym_id
    meta.classes = asset.classes
    meta.class_strengths = asset.class_strengths
    meta.categories = asset.categories
    meta.category_strengths = asset.category_strengths
    meta.pitch_class = asset.pitch_class
    meta.base_note = asset.base_note
    meta.base_note_confidence = asset.base_note_confidence
    meta.peak_db = asset.peak_db
    meta.rms_db = asset.rms_db
    meta.crest_factor = asset.crest_factor
    meta.bpm = asset.bpm
    meta.bpm_confidence = asset.bpm_confidence
    meta.brightness = asset.brightness
    meta.noisiness = asset.noisiness
    meta.harmonicity = asset.harmonicity
    meta.class_signature = asset.class_signature
    meta.category_signature = asset.category_signature
    meta.pitch_confidence = asset.pitch_confidence

    return sample, created


def _rebuild_generic_metadata(counts: dict) -> None:
    """Keep the generic labels/descriptors tables in step with sononym_meta
    (fourier/metadata/store.py): rebuilt whole after an import that changed anything, or
    when they're still empty (a database from before they existed)."""
    from sqlalchemy import text

    from ..metadata.store import SONONYM, rebuild_sononym
    with session_scope() as session:
        changed = any(counts.get(k) for k in ("created", "updated", "pruned", "removed"))
        empty = not session.execute(text("SELECT 1 FROM descriptors WHERE provider = :p LIMIT 1"),
                                    {"p": SONONYM}).first()
        if changed or empty:
            counts["metadata"] = rebuild_sononym(session, log=log.info)


def _unmark_missing(paths: set) -> None:
    """Samples a walk marked missing that Sononym has just synced from disk are there again."""
    with session_scope() as session:
        try:
            rows = session.execute(text("SELECT sample_id, path FROM missing_files")).all()
        except Exception:
            return
        back = [sid for sid, p in rows if p in paths]
        for i in range(0, len(back), 500):
            ids = ", ".join(str(int(x)) for x in back[i:i + 500])
            session.execute(text(f"DELETE FROM missing_files WHERE sample_id IN ({ids})"))


def import_from_sononym(
    sononym_db: Path | None = None,
    library_root: Path | None = None,
    category_filter: list[str] | None = None,
    class_filter: list[str] | None = None,
    file_types: list[str] | None = None,
    only_existing: bool = True,
    progress_cb: Callable[[int, int, str], None] | None = None,
    batch_size: int = 200,
    limit: int | None = None,
) -> dict:
    """
    Import samples from Sononym's DB into fourier.

    Args:
        sononym_db: path to Sononym's sononym.db (default: places.sononym_db())
        library_root: root path for resolving relative paths
        category_filter: restrict to specific Sononym categories
        class_filter: ["OneShot"] or ["Loop"] or None for all
        file_types: ["wav", "aif"] etc.
        only_existing: skip samples whose file doesn't exist on disk
        progress_cb: callback(current, total_estimate, current_path)
        batch_size: rows per DB transaction

    Returns:
        dict with counts: created, updated, skipped, errors
    """
    reader = SononymReader(db_path=sononym_db, library_root=library_root)
    total_estimate = reader.count()
    log.info(f"Sononym DB has ~{total_estimate:,} assets")

    counts = {"created": 0, "updated": 0, "skipped": 0, "errors": 0}
    processed = 0
    batch: list[SononymAsset] = []

    def flush_batch(session: Session) -> None:
        for asset in batch:
            try:
                _, created = _upsert_sample_from_sononym(session, asset, reader.library_root, counts)
                if created:
                    counts["created"] += 1
                else:
                    counts["updated"] += 1
            except Exception as e:
                log.warning(f"Error importing {asset.rel_path}: {e}")
                counts["errors"] += 1

    stream = reader.stream_assets(
        category_filter=category_filter,
        class_filter=class_filter,
        file_types=file_types,
        only_existing=only_existing,
        include_extended=True,
    )

    with session_scope() as session:
        for asset in stream:
            batch.append(asset)
            processed += 1

            if progress_cb:
                progress_cb(processed, total_estimate, asset.rel_path)

            if limit and processed >= limit:
                log.info(f"Reached import limit of {limit}")
                flush_batch(session)
                batch.clear()
                break

            if len(batch) >= batch_size:
                flush_batch(session)
                batch.clear()
                session.commit()
                log.debug(
                    f"Progress: {processed:,}/{total_estimate:,} "
                    f"(+{counts['created']} new, {counts['updated']} updated)"
                )

        # Final batch
        if batch:
            flush_batch(session)

    _rebuild_generic_metadata(counts)
    log.info(
        f"Import complete: {counts['created']} created, {counts['updated']} updated, "
        f"{counts['skipped']} skipped, {counts['errors']} errors"
    )
    return counts


def sync_from_sononym(
    sononym_db: Path | None = None,
    library_root: Path | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    batch_size: int = 200,
    prune: bool = False,
) -> dict:
    """
    Incremental sync: import only new or changed assets from Sononym.

    Compares Sononym's modtime against fourier's modified_at to detect changes.
    Much faster than a full reimport when most samples are unchanged.

    Args:
        sononym_db:  path to Sononym's sononym.db (default: places.sononym_db())
        library_root: root path for resolving relative paths
        progress_cb: callback(current, total_to_process, current_path)
        batch_size:  rows per DB transaction
        prune:       if True, delete fourier rows for samples no longer in Sononym

    Returns:
        dict with counts: created, updated, unchanged, removed, errors
    """
    reader = SononymReader(db_path=sononym_db, library_root=library_root)

    # --- Step 1: lightweight modtime snapshot from both sides ---------------
    log.info("Loading Sononym modtimes (lightweight scan)...")
    sononym_modtimes: dict[str, int] = reader.get_all_modtimes()

    log.info("Loading fourier modtimes...")
    with session_scope() as session:
        # Sononym's own rows only: a file a walk of the library added (rel_path, no Sononym
        # row) is new to Sononym's side, matched by its path and given its row
        rows = session.execute(
            text("SELECT s.rel_path, s.modified_at, s.file_hash FROM samples s "
                 "JOIN sononym_meta m ON m.sample_id = s.id WHERE s.rel_path IS NOT NULL")
        ).fetchall()
        fourier_modtimes: dict[str, int] = {r[0]: (r[1] or 0) for r in rows}
        # samples not hashed yet (cloud-only when they were scanned) get another try
        unhashed = {r[0] for r in rows if r[2] is None}

    # --- Step 2: classify ---------------------------------------------------
    sononym_paths = set(sononym_modtimes)
    fourier_paths = set(fourier_modtimes)

    new_paths = sononym_paths - fourier_paths
    removed_paths = fourier_paths - sononym_paths
    changed_paths = {
        p for p in (sononym_paths & fourier_paths)
        if sononym_modtimes[p] != fourier_modtimes[p]
    }
    retry_hash = {p for p in (sononym_paths & fourier_paths) - changed_paths
                  if p in unhashed and not platforms.cloud_only(reader.library_root / p)}
    unchanged_count = len(sononym_paths & fourier_paths) - len(changed_paths) - len(retry_hash)
    to_process = new_paths | changed_paths | retry_hash

    log.info(
        f"Sync plan: {len(new_paths):,} new, {len(changed_paths):,} changed, "
        f"{unchanged_count:,} unchanged, {len(removed_paths):,} removed from Sononym"
    )

    counts: dict = {
        "created": 0,
        "updated": 0,
        "unchanged": unchanged_count,
        "removed": len(removed_paths),
        "errors": 0,
    }

    # --- Step 3: stream + process only new / changed ------------------------
    if to_process:
        batch: list[SononymAsset] = []

        def flush_batch(session: Session) -> None:
            for asset in batch:
                try:
                    _, created = _upsert_sample_from_sononym(session, asset, reader.library_root, counts)
                    if created:
                        counts["created"] += 1
                    else:
                        counts["updated"] += 1
                except Exception as e:
                    log.warning(f"Error syncing {asset.rel_path}: {e}")
                    counts["errors"] += 1

        processed = 0
        stream = reader.stream_assets(include_extended=True, only_existing=True)

        with session_scope() as session:
            for asset in stream:
                if asset.rel_path not in to_process:
                    continue  # skip unchanged

                batch.append(asset)
                processed += 1

                if progress_cb:
                    progress_cb(processed, len(to_process), asset.rel_path)

                if len(batch) >= batch_size:
                    flush_batch(session)
                    batch.clear()
                    session.commit()
                    log.debug(
                        f"Sync progress: {processed:,}/{len(to_process):,} "
                        f"(+{counts['created']} new, {counts['updated']} updated)"
                    )

            if batch:
                flush_batch(session)

    if to_process:
        _unmark_missing({str(reader.library_root / p) for p in to_process})

    # --- Step 4: optional prune ---------------------------------------------
    if prune and removed_paths:
        pruned = 0
        from ..db.session import delete_samples
        with session_scope() as session:
            ids = []
            for rel_path in removed_paths:
                ids += [i for (i,) in session.query(Sample.id).filter_by(rel_path=rel_path)]
            pruned = delete_samples(session, ids)
        log.info(f"Pruned {pruned} samples no longer in Sononym")
        counts["pruned"] = pruned

    _rebuild_generic_metadata(counts)
    log.info(
        f"Sync complete: {counts['created']} created, {counts['updated']} updated, "
        f"{counts['unchanged']:,} unchanged, {counts['removed']} removed"
        + (f", {counts.get('pruned', 0)} pruned" if prune else "")
    )
    return counts


_SUBTYPE_BITS = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
                 "FLOAT": 32, "DOUBLE": 64}


def file_info(path) -> dict | None:
    """What a file's header says (duration_s, sample_rate, channels, bit_depth,
    file_format), read with soundfile.info without decoding the audio; None when it can't be
    read, or when the file is cloud-only (reading it would download it)."""
    try:
        if platforms.cloud_only(path):
            return None
        import soundfile as sf
        info = sf.info(str(path))
    except Exception:
        return None
    if not info.samplerate or info.frames < 0:
        return None
    return {"duration_s": round(info.frames / info.samplerate, 6), "sample_rate": int(info.samplerate),
            "channels": int(info.channels), "bit_depth": _SUBTYPE_BITS.get(info.subtype),
            "file_format": Path(path).suffix.lower().lstrip(".") or None}


def _fill_scanned(sample: Sample, path: Path, rel: str, info: dict | None = None) -> None:
    """A walked file's library path, header facts, quick hash and a WAV's own chunks (none of
    them for a cloud-only file, which reading would download). info: file_info(path), when
    the caller has read it already."""
    sample.rel_path = rel
    for k, v in (info if info is not None else (file_info(path) or {})).items():
        if v is not None and getattr(sample, k) is None:
            setattr(sample, k, v)
    if sample.file_hash is None and not platforms.cloud_only(path):
        try:
            sample.file_hash = _quick_hash(path)
        except OSError:
            pass
    _fill_chunks(sample, path)


def _fill_chunks(sample: Sample, path) -> bool:
    """A WAV's acid and smpl chunks (ingest/chunks.py) into the sample, once (chunks_read);
    not for a cloud-only file. Returns whether it read them."""
    if sample.chunks_read or not str(path).lower().endswith((".wav", ".wave")) \
            or platforms.cloud_only(path):
        return False
    from .chunks import read_wav_chunks
    for k, v in read_wav_chunks(path).items():
        setattr(sample, k, v)
    sample.chunks_read = 1
    return True


ERRORS_LISTED = 10       # files a scan names among its errors (the rest are counted)


def _known_rows(session) -> dict:
    """{path: (id, rel_path is None, duration_s is None)} for every sample: one query, so the
    walk looks a file up in memory rather than asking the database for each."""
    rows = session.execute(text("SELECT id, path, rel_path, duration_s FROM samples")).all()
    return {p: (i, rel is None, dur is None) for i, p, rel, dur in rows}


def _unchunked(session) -> set:
    """Ids of the WAV samples whose chunks no walk has read yet (a database from before
    chunks_read): the walk reads them once."""
    try:
        return {i for (i,) in session.execute(text(
            "SELECT id FROM samples WHERE chunks_read IS NULL AND lower(path) LIKE '%.wav'"))}
    except Exception:
        return set()


def _walk_cache(session, root_s: str, real_root: str) -> dict:
    """The stored folder listings under a library folder: {path: (mtime_ns, listing)}."""
    from ..db.models import WalkedDir
    out = {}
    for path, mt, listing in session.query(WalkedDir.path, WalkedDir.mtime_ns, WalkedDir.listing):
        if path == root_s or path.startswith(root_s.rstrip("/\\") + os.sep) \
                or path == real_root or path.startswith(real_root.rstrip("/\\") + os.sep):
            out[path] = (mt, listing)
    return out


def unavailable_links(roots, session=None) -> list[tuple[str, str]]:
    """[(link, target)]: the folder symlinks in these library folders whose target isn't
    there (a drive that isn't plugged in), as the scan's walk finds them (a build stops on
    them unless --allow-missing). Reads the stored folder listings when a database is open
    (only changed folders are listed again) and writes nothing."""
    from .formats import readable_exts
    from .walk import own_folders, root_state, walk
    exts = tuple(e.lower() for e in readable_exts())
    out = []
    for r in roots:
        root_s = os.path.normpath(os.path.expanduser(str(r)))
        if root_state(root_s):
            continue
        cache = {}
        if session is not None:
            try:
                cache = _walk_cache(session, root_s, os.path.realpath(root_s))
            except Exception:           # a database from before the listings
                cache = {}
        w = walk(root_s, exts, cache=cache, own=own_folders())
        out += [(link, target) for _key, link, target in w.unavailable]
    return out


def _save_walk_cache(session, w, old: dict) -> None:
    """Store the folder listings this walk made, and forget folders it no longer found."""
    from ..db.models import WalkedDir
    gone = [p for p in old if p not in w.cache_seen]
    for i in range(0, len(gone), 500):
        session.query(WalkedDir).filter(WalkedDir.path.in_(gone[i:i + 500])).delete(
            synchronize_session=False)
    for path, (mt, listing) in w.cache_put.items():
        if path in old:
            session.query(WalkedDir).filter(WalkedDir.path == path).update(
                {"mtime_ns": mt, "listing": listing}, synchronize_session=False)
        else:
            session.add(WalkedDir(path=path, mtime_ns=mt, listing=listing))


def _under_walk(w, exts, known: dict) -> dict:
    """{sample id: path} of the known samples the walk of w's folder covers: under it, with an
    extension it reads, outside folders it couldn't list or left out."""
    from .walk import under
    prefixes = {w.real_root, w.root}
    skip = list(w.unlistable) + list(w.own_skipped)
    out = {}
    for path, (sid, _r, _d) in known.items():
        if not under(path, prefixes) or path in (w.real_root, w.root):
            continue
        if os.path.splitext(path)[1].lower() not in exts or (skip and under(path, skip)):
            continue
        out[sid] = path
    return out


# what a moved file carries to its new row besides its analysis (sample_features)
_MOVED_SAMPLE_FIELDS = ("is_favorite", "is_hidden", "user_tags")


def _match_moves(session, w, exts, seen_ids: set, known: dict, created: list) -> dict:
    """Files the walk found moved: a sample under the folder it didn't find, and a file new
    to it with the same content (quick hash and size; the same file name first, and only
    where the match is one to one), or a file marked missing that is back where it was. A new
    row gets the old one's analysis (sample_features), so a renamed folder isn't analyzed
    again; the old row is marked missing as any file the walk didn't find. Returns {old path:
    new path}."""
    from ..db.models import SampleFeatures
    new = [s for s in created if s.file_hash and s.id is not None]
    # ...and files marked missing that are back where they were (a folder renamed back):
    # their rows have their analysis, a rating that followed them away comes back
    try:
        flagged = {i for (i,) in session.execute(text("SELECT sample_id FROM missing_files"))}
    except Exception:
        flagged = set()
    back = [session.get(Sample, sid) for sid in sorted(flagged & seen_ids)]
    new += [s for s in back if s is not None and s.file_hash]
    if not new:
        return {}
    gone = {sid: p for sid, p in _under_walk(w, exts, known).items() if sid not in seen_ids}
    if not gone:
        return {}
    fresh = {s.id for s in created}
    by_key = defaultdict(list)
    ids = sorted(gone)
    for i in range(0, len(ids), 500):
        chunk = ", ".join(str(x) for x in ids[i:i + 500])
        for sid, h, size in session.execute(text(
                f"SELECT id, file_hash, file_size_bytes FROM samples WHERE id IN ({chunk})")):
            if h:
                by_key[(h, size)].append(sid)
    cands = defaultdict(list)            # old id -> new samples with its content
    for s in new:
        for sid in by_key.get((s.file_hash, s.file_size_bytes), ()):
            cands[sid].append(s)
    taken, moves = set(), {}
    for sid in sorted(cands):
        olds = cands[sid]
        name = os.path.basename(gone[sid]).lower()
        same = [s for s in olds if (s.filename or "").lower() == name]
        pick = same if same else olds
        rivals = [o for o, ss in cands.items() if o != sid and any(x in ss for x in pick)]
        if len(pick) != 1 or pick[0].id in taken or rivals:
            continue                      # not one to one: a copy, not a move
        dest = pick[0]
        taken.add(dest.id)
        moves[gone[sid]] = dest.path
        if dest.id not in fresh:
            continue                      # a row of its own, analysis and all
        old = session.get(Sample, sid)
        for k in _MOVED_SAMPLE_FIELDS:
            if getattr(old, k, None) is not None:
                setattr(dest, k, getattr(old, k))
        feat = session.query(SampleFeatures).filter_by(sample_id=sid).first()
        if feat is not None and session.query(SampleFeatures).filter_by(sample_id=dest.id).first() is None:
            cols = [c.key for c in SampleFeatures.__table__.columns if c.key not in ("id", "sample_id")]
            session.add(SampleFeatures(sample_id=dest.id, **{c: getattr(feat, c) for c in cols}))
    if moves:
        session.flush()
    return moves


def _record_missing(session, w, exts, seen_ids: set, known: dict, counts: dict,
                    moves: dict | None = None) -> None:
    """Mark the samples under the walked folder that it didn't find (with an extension it
    reads, outside folders it couldn't list or left out) missing; unmark the ones it found.
    The old rows of files this walk found moved (moves: {old path: new path}) record where
    they went (moved_to): counts["missing"] and ["newly_missing"] leave them out, as they
    leave out the old rows of earlier moves."""
    from ..db.models import MissingFile
    moves = moves or {}
    under_root = _under_walk(w, exts, known)
    moved_before = {sid for (sid,) in session.execute(text(
        "SELECT sample_id FROM missing_files WHERE moved_to IS NOT NULL")).all()}
    flagged = dict(session.execute(text("SELECT sample_id, path FROM missing_files")).all())
    found_again = [sid for sid in under_root if sid in seen_ids and sid in flagged]
    for i in range(0, len(found_again), 500):
        session.query(MissingFile).filter(MissingFile.sample_id.in_(found_again[i:i + 500])).delete(
            synchronize_session=False)
    now = int(time.time())
    new = 0
    moved = set()
    for sid, path in under_root.items():
        if sid in seen_ids:
            continue
        if path in moves:
            moved.add(sid)
        if flagged.get(sid) == path:
            if path in moves:
                session.query(MissingFile).filter(MissingFile.sample_id == sid).update(
                    {"moved_to": moves[path]}, synchronize_session=False)
            continue
        if sid in flagged:                       # an id reused for another path
            session.query(MissingFile).filter(MissingFile.sample_id == sid).delete(
                synchronize_session=False)
            moved_before.discard(sid)
        session.add(MissingFile(sample_id=sid, path=path, since=now, moved_to=moves.get(path)))
        if path not in moves:
            new += 1
    # rows of samples that are gone from the database
    session.execute(text("DELETE FROM missing_files WHERE sample_id NOT IN (SELECT id FROM samples)"))
    counts["missing"] = sum(1 for sid in under_root
                            if sid not in seen_ids and sid not in moved and sid not in moved_before)
    counts["newly_missing"] = new
    counts["found_again"] = len(found_again)


def scan_filesystem(
    root: Path,
    extensions=None,
    progress_cb: Callable[[int, str], None] | None = None,
) -> dict:
    """
    Walk a library folder (ingest/walk.py) and import the audio files the DB doesn't have yet.
    Used when Sononym doesn't index the library (or beside it, `tools scan --walk`). Each new
    sample gets its path under root (rel_path, as Sononym's rows have), what its header says
    (file_info) and a quick content hash, so the built-in providers can classify it and later
    scans can match it. A sample an earlier walk added without them gets them now; a sample
    with a rel_path (a Sononym row, or one already filled) is left as it is.

    A file soundfile can't read (corrupt, or empty) is an error, named in
    counts["error_files"] (the first ERRORS_LISTED), not a sample. Samples under root the walk
    didn't find are marked missing (curation leaves them out) and unmarked when it finds them
    again. A root that is missing or empty is not walked and nothing is marked
    (counts["root"] says why).

    Returns counts: created, skipped (already known), filled, errors, error_files,
    skipped_ext {extension: files}, links_followed, links_inside, loops, missing,
    newly_missing, found_again, listed and reused (folders), moved (files found at a new path,
    _match_moves) and unavailable_links [(link, target)]: folder symlinks whose target isn't
    there, whose samples are marked missing.
    """
    from .formats import readable_exts
    from .walk import own_folders, root_state, walk
    exts = tuple(e.lower() for e in (extensions or readable_exts()))
    counts: dict = {"created": 0, "skipped": 0, "errors": 0, "error_files": []}
    why = root_state(root)
    if why:
        counts["root"] = why
        return counts

    errors: dict = {}

    def error(path, what):
        counts["errors"] += 1
        errors[str(path)] = what
        if len(counts["error_files"]) < ERRORS_LISTED:
            counts["error_files"].append(f"{path} ({what})")

    root_s = os.path.normpath(os.path.expanduser(str(root)))
    bad_before = _scan_error_doc().get(root_s) or {}     # the last walk's unreadable files
    with session_scope() as session:
        cache = _walk_cache(session, root_s, os.path.realpath(root_s))
        w = walk(root_s, exts, cache=cache, own=own_folders())
        log.info(f"Found {len(w.files):,} audio files under {root}")
        known = _known_rows(session)
        unchunked = _unchunked(session)
        seen_ids: set = set()
        created: list = []
        for i, f in enumerate(w.files):
            if progress_cb:
                progress_cb(i, f.path)
            try:
                hit = known.get(f.key)
                if hit is None and f.key != _logical(w.root, f.rel):
                    hit = known.get(_logical(w.root, f.rel))   # a row stored at the root as spelled
                if hit is not None:
                    sid, no_rel, no_dur = hit
                    counts["skipped"] += 1
                    info = None
                    if no_dur and not platforms.cloud_only(f.path):
                        info = file_info(f.path)
                        if info is None:                        # left out: marked missing below
                            error(f.path, "can't be read")
                            continue
                    if no_rel:                                  # an earlier walk's row: fill it in
                        _fill_scanned(session.get(Sample, sid), Path(f.path), f.rel, info)
                        counts["filled"] = counts.get("filled", 0) + 1
                    elif sid in unchunked and _fill_chunks(session.get(Sample, sid), f.path):
                        counts["chunks"] = counts.get("chunks", 0) + 1
                    seen_ids.add(sid)
                    continue
                st = os.stat(f.path)
                was = bad_before.get(f.path)
                if was and was.get("stat") == [st.st_size, st.st_mtime_ns]:
                    error(f.path, was.get("why") or "can't be read")   # unchanged since: not read again
                    continue
                cloud = platforms.stat_cloud_only(os.stat(f.path, follow_symlinks=False))
                info = None
                if not cloud:
                    if st.st_size == 0:
                        error(f.path, "empty file")
                        continue
                    info = file_info(f.path)
                    if info is None:
                        error(f.path, "can't be read")
                        continue
                sample = Sample(
                    path=f.key,
                    filename=os.path.basename(f.rel),
                    file_size_bytes=st.st_size,
                    modified_at=int(st.st_mtime),
                    file_format=os.path.splitext(f.rel)[1].lower().lstrip("."),
                )
                _fill_scanned(sample, Path(f.path), f.rel, info or {})
                session.add(sample)
                created.append(sample)
                counts["created"] += 1
                if counts["created"] % 500 == 0:
                    session.commit()
            except Exception as e:
                log.warning(f"Error scanning {f.path}: {e}")
                error(f.path, str(e))
        session.flush()
        moves = _match_moves(session, w, set(exts), seen_ids, known, created)
        counts["moved"] = len(moves)
        _record_missing(session, w, set(exts), seen_ids, known, counts, moves)
        _save_walk_cache(session, w, cache)
    if moves:                     # ratings follow the files (packs/ratings.py)
        from ..packs.ratings import rekey_paths
        try:
            counts["ratings_moved"] = rekey_paths(moves)
        except (OSError, ValueError) as e:
            log.warning(f"ratings not moved with their files: {e}")
    _save_scan_errors(root_s, errors)
    counts.update(skipped_ext=dict(w.skipped), links_followed=w.links_followed,
                  links_inside=w.links_inside, loops=w.loops, listed=w.listed, reused=w.reused,
                  unlistable=len(w.unlistable),
                  unavailable_links=[(link, target) for _key, link, target in w.unavailable])
    log.info(f"Scan complete: {counts}")
    return counts


SCAN_ERRORS = "scan_errors.json"     # in the home: {library folder: {file: why}} of the last walks


def _scan_error_doc() -> dict:
    """{library folder: {file: {"why", "stat": [size, mtime_ns]}}} of the last walks (an older
    file's {file: why} reads as without a stat)."""
    import json
    from ..paths import home_path
    p = home_path(SCAN_ERRORS)
    try:
        doc = json.loads(p.read_text()) if p.exists() else {}
    except (OSError, ValueError):
        return {}
    return {root: {f: (v if isinstance(v, dict) else {"why": v}) for f, v in (errs or {}).items()}
            for root, errs in doc.items() if isinstance(errs, dict)}


def _save_scan_errors(root_s: str, errors: dict) -> None:
    """Remember the files the last walk of a library folder couldn't read (corrupt or empty)
    with their size and time, so doctor and setup count them as errors, not as files waiting
    for analysis, and the next walk doesn't read them again until they change."""
    import json
    from ..paths import home_path
    p = home_path(SCAN_ERRORS)
    doc = _scan_error_doc()
    if errors:
        entry = {}
        for f, why in sorted(errors.items()):
            try:
                st = os.stat(f)
                entry[f] = {"why": why, "stat": [st.st_size, st.st_mtime_ns]}
            except OSError:
                entry[f] = {"why": why}
        doc[root_s] = entry
    else:
        doc.pop(root_s, None)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
        tmp.replace(p)
    except OSError:
        pass


def scan_errors(roots=None) -> dict:
    """{file: why} the last walk of each library folder (these roots, or every one recorded)
    couldn't read."""
    doc = _scan_error_doc()
    want = None if roots is None else {os.path.normpath(os.path.expanduser(str(r))) for r in roots}
    out = {}
    for root, errs in doc.items():
        if want is None or root in want:
            out.update({f: v.get("why") or "can't be read" for f, v in errs.items()})
    return out


def _logical(root_s: str, rel: str) -> str:
    return os.path.join(root_s, *rel.split("/"))
