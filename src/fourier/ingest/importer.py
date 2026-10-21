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


QUICK_BYTES = 8192          # what the quick hash reads
_BLOCK = 1 << 20


def _quick_hash(path: Path, nbytes: int = QUICK_BYTES) -> str:
    """SHA256 of the first nbytes of a file (16 hex digits): a cheap first look, which files
    that start alike share (a loop and its fill, takes after the same silence). Never an
    identity on its own."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def _hashes(path) -> tuple[str, str]:
    """(quick hash, content hash) of a file, read once. The content hash (SHA256 of every byte,
    16 hex digits) is the file's identity; for a file of QUICK_BYTES or fewer it equals the
    quick hash."""
    q, h = hashlib.sha256(), hashlib.sha256()
    with open(path, "rb") as f:
        head = f.read(QUICK_BYTES)
        q.update(head)
        h.update(head)
        for block in iter(lambda: f.read(_BLOCK), b""):
            h.update(block)
    return q.hexdigest()[:16], h.hexdigest()[:16]


def _set_hashes(sample: Sample, path, st=None) -> None:
    """Hash a file in full into its row (quick_hash, file_hash) and record the size and time
    it was read at."""
    st = st if st is not None else os.stat(path)
    sample.quick_hash, sample.file_hash = _hashes(path)
    sample.file_size_bytes = st.st_size
    sample.file_mtime_ns = st.st_mtime_ns


def _full_kind(size, quick, full) -> bool:
    """Whether a row's file_hash is of the whole file: it differs from the quick hash, or the
    file is no longer than the quick hash reads (the two are one). A row without a quick hash
    holds only that in file_hash."""
    if full is None:
        return False
    small = size is not None and size <= QUICK_BYTES
    return small if quick is None else (full != quick or small)


SAME_LENGTH_S = 0.01        # two header lengths this close are one length


def _header_differs(facts, path) -> bool:
    """Whether the file's header (length, sample rate, channels) differs from its row's facts
    (duration_s, sample_rate, channels); a fact the row lacks doesn't count. An unreadable
    header differs."""
    info = file_info(path)
    if info is None:
        return True
    dur, sr, ch = facts
    if dur is not None and info.get("duration_s") is not None and abs(dur - info["duration_s"]) > SAME_LENGTH_S:
        return True
    return any(a is not None and b is not None and int(a) != int(b)
               for a, b in ((sr, info.get("sample_rate")), (ch, info.get("channels"))))


_NS, _HOUR_NS = 10**9, 3600 * 10**9


def _clock_shift(old_ns, new_ns) -> bool:
    """Whether a file's time moved the way a clock or a copy moves it, with no edit: by whole
    hours (a drive's time zone or daylight saving time, up to a day), or cut to whole seconds
    by a copy (within 2 s, a FAT drive's step), or both."""
    if old_ns is None:
        return False
    d = new_ns - old_ns
    k = round(d / _HOUR_NS)
    rest = d - k * _HOUR_NS
    return abs(k) <= 24 and (rest == 0 or (abs(rest) < 2 * _NS and new_ns % _NS == 0 and old_ns % _NS != 0))


def _content_changed(path, st, row, moved: bool = True,
                     old_ns: int | None = None) -> tuple[bool | None, tuple[str, str]]:
    """(whether the file at path holds other audio than its row was measured on, its (quick,
    content) hashes). row: (size, quick hash, file hash, duration_s, sample_rate, channels).
    - A row with the content hash: the content hash decides.
    - A row with only the quick hash (scanned before schema 3): another quick hash is a
      change. Another size, or none stored, is a change only when the header says another
      length, rate or channel count too (a stored size can be another source's). With the
      same size and quick hash and a time that moved since the row's (moved; old_ns, the
      row's time): another header is a change, a clock shift (_clock_shift) isn't, and
      otherwise it can't be told (None): an edit past the first 8 KB or a copy that didn't
      keep the time. The caller counts those (_settle_moved_times). At first sight (moved
      False) the file is taken as the one analyzed."""
    size, quick, full, *facts = row
    q, h = _hashes(path)
    if _full_kind(size, quick, full):
        return h != full, (q, h)
    if q != (quick or full):
        return True, (q, h)
    if size is None or st.st_size != size:
        return _header_differs(facts, path), (q, h)
    if not moved:
        return False, (q, h)
    if _header_differs(facts, path):
        return True, (q, h)
    return (False if _clock_shift(old_ns, st.st_mtime_ns) else None), (q, h)


# more rows from before than this whose times moved in one scan with the same size, first
# 8 KB and header are a library copied or restored without its times (that many edits past
# the first 8 KB that keep each file's size and header are unlikely)
MANY_TIMES_MOVED = 20


def _settle_moved_times(doubt: list, counts: dict) -> None:
    """The rows from before whose time moved with the same size, first 8 KB and header that
    one scan found (_content_changed's None): [(sample, st, hashes, redo)]. Each is hashed in
    full now. A few are taken as edits past the first 8 KB and analyzed again (redo(), counted
    in counts["changed"]); more than MANY_TIMES_MOVED at once are a copy or a restore that
    didn't keep the times: their analysis is kept (counts["times_moved"])."""
    if not doubt:
        return
    many = len(doubt) > MANY_TIMES_MOVED
    for sample, st, hashes, redo in doubt:
        sample.quick_hash, sample.file_hash = hashes
        sample.file_size_bytes, sample.file_mtime_ns = st.st_size, st.st_mtime_ns
        if not many:
            redo()
    key = "times_moved" if many else "changed"
    counts[key] = counts.get(key, 0) + len(doubt)


def _row_facts(sample: Sample) -> tuple:
    """What _content_changed compares a file against, from its row."""
    return (sample.file_size_bytes, sample.quick_hash, sample.file_hash, sample.duration_s,
            sample.sample_rate, sample.channels)


def forget_analysis(session: Session, sample_id: int) -> None:
    """Drop what Fourier measured and read of a sample's audio (sample_features, the audio
    provider's and the sound model's labels), so the analysis steps redo it. Sononym's and
    Live's rows, the path labels and the ratings stay. (The CLAP index drops the sample's
    vector when it's next rebuilt: curate.index_stale compares the samples it holds.)"""
    from ..db.models import SampleFeatures
    from ..metadata.shadow import AUDIO
    from ..metadata.sound import PROVIDER as SOUND
    session.query(SampleFeatures).filter_by(sample_id=sample_id).delete(synchronize_session=False)
    for table in ("labels", "descriptors"):
        session.execute(text(f"DELETE FROM {table} WHERE sample_id = :s AND provider IN (:a, :b)"),
                        {"s": sample_id, "a": AUDIO, "b": SOUND})


def _refresh_changed(session: Session, sample: Sample, path, st, hashes, info: dict) -> None:
    """A file whose bytes changed at its path: its row takes the new size, time, hashes,
    header and chunks, and its analysis is redone (forget_analysis)."""
    sample.quick_hash, sample.file_hash = hashes
    sample.file_size_bytes, sample.file_mtime_ns = st.st_size, st.st_mtime_ns
    sample.modified_at = int(st.st_mtime)
    for k, v in info.items():
        setattr(sample, k, v)
    sample.acid_bpm = sample.acid_beats = sample.root_note = sample.chunks_read = None
    _fill_chunks(sample, path)
    forget_analysis(session, sample.id)


# a row whose file_hash is only its quick hash (scanned before schema 3), of a file longer
# than the quick hash reads
_QUICK_ONLY = (f"file_hash IS NOT NULL AND file_hash = quick_hash "
               f"AND (file_size_bytes IS NULL OR file_size_bytes > {QUICK_BYTES})")


def settle_hashes(session: Session, log_cb=None, counts: dict | None = None, ids=None) -> int:
    """Give exact identity to the rows that lack it, reading only what that needs:
    - rows whose file_hash an older version wrote into this database (db/migrations.py moves
      it to quick_hash and clears file_hash): read in full;
    - rows scanned before schema 3 whose first 8 KB another row shares: read in full, so
      file_hash tells them apart and equal hashes mean equal bytes everywhere (twins,
      ratings, moves).
    A file whose bytes changed since its row was stored (_content_changed at first sight)
    gets its new facts and is analyzed again (counts["changed"]).
    Runs after every scan, whichever source ran: it finds nothing to do once every such row
    is settled. ids: the candidate rows when the caller has found them (the walk does, from
    the facts it already read: _unsettled), else one query finds them. A file that isn't
    there, is cloud-only, or changed since its row's time (the next change check reads it)
    keeps its row as it is. Returns how many files it read."""
    cols = ("SELECT id, path, file_size_bytes, quick_hash, file_hash, file_mtime_ns, duration_s, "
            "sample_rate, channels FROM samples WHERE ")
    older = "(file_hash IS NULL AND quick_hash IS NOT NULL)"
    if ids is None:
        rows = list(session.execute(text(
            cols + f"{older} OR ({_QUICK_ONLY} AND quick_hash IN (SELECT quick_hash FROM samples "
            "WHERE quick_hash IS NOT NULL GROUP BY quick_hash HAVING COUNT(*) > 1))")))
    else:
        rows, ids = [], sorted(ids)
        for i in range(0, len(ids), 500):
            chunk = ", ".join(str(int(x)) for x in ids[i:i + 500])
            rows += list(session.execute(text(cols + f"id IN ({chunk}) AND ({older} OR ({_QUICK_ONLY}))")))
    n, settled = 0, []
    for sid, path, size, quick, full, mtime_ns, dur, sr, ch in rows:
        if not path or platforms.cloud_only(path):
            continue
        try:
            st = os.stat(path)
            if mtime_ns is not None and st.st_mtime_ns != mtime_ns:
                continue                  # changed since its scan: the change check reads it
            changed, hashes = _content_changed(path, st, (size, quick, full, dur, sr, ch), moved=False)
        except OSError:
            continue
        n += 1
        if changed:
            sample = session.get(Sample, sid)
            info = file_info(path)
            if sample is None or info is None:
                continue                  # a file that can't be read: its row as it was
            _refresh_changed(session, sample, path, st, hashes, info)
            if counts is not None:
                counts["changed"] = counts.get("changed", 0) + 1
            continue
        settled.append({"i": sid, "q": hashes[0], "h": hashes[1], "s": st.st_size, "t": st.st_mtime_ns})
    _update_rows(session, settled)
    if n and log_cb:
        log_cb(f"read {n:,} files in full to tell them apart")
    return n


def _upsert_sample_from_sononym(
    session: Session,
    asset: SononymAsset,
    library_root: Path,
    counts: dict | None = None,
    doubt: list | None = None,
) -> tuple[Sample, bool]:
    """
    Insert or update a Sample + SononymMeta from a SononymAsset.
    Returns (sample, created) where created=True if new row inserted. A cloud-only file
    (platforms.cloud_only) isn't hashed, since reading it would download it: counted in
    counts["cloud_only"], it's hashed by a later scan once it's on this machine. doubt, a
    list, gets the rows from before whose time moved with the same bytes as far as they can
    be told (_settle_moved_times decides at the end of the sync); without it they're changed.
    """
    abs_path = str(asset.abs_path)

    existing = session.query(Sample).filter_by(path=abs_path).first()
    created = existing is None

    if existing is None:
        sample = Sample(path=abs_path)
        session.add(sample)
    else:
        sample = existing

    # a known file whose time Sononym saw move: when its bytes changed too, Fourier's own
    # analysis of it is redone (_content_changed). The file's own time says whether it moved
    # since this database read it: the walk's or this sync's (file_mtime_ns), else the
    # whole seconds a walk stored in modified_at. The row takes the size it has on disk.
    p = asset.abs_path
    size_on_disk = None
    if not created and (sample.file_hash or sample.quick_hash) and sample.modified_at != asset.modtime:
        try:
            if p.exists() and not platforms.cloud_only(p):
                st = os.stat(p)
                size_on_disk = st.st_size
                old_ns: int | None = None
                if sample.file_mtime_ns is not None:
                    moved = (st.st_mtime_ns, st.st_size) != (sample.file_mtime_ns, sample.file_size_bytes)
                    old_ns = sample.file_mtime_ns
                else:
                    moved = sample.modified_at != int(st.st_mtime)
                    if sample.modified_at is not None:
                        old_ns = int(sample.modified_at) * 1_000_000_000
                changed: bool | None = False
                if moved or sample.file_mtime_ns is None:
                    changed, hashes = _content_changed(p, st, _row_facts(sample), moved=moved, old_ns=old_ns)
                    sample.quick_hash, sample.file_hash = hashes
                    sample.file_mtime_ns = st.st_mtime_ns

                def redo(sample=sample, p=p) -> None:
                    sample.acid_bpm = sample.acid_beats = sample.root_note = sample.chunks_read = None
                    _fill_chunks(sample, p)
                    forget_analysis(session, sample.id)
                if changed is None and doubt is not None:
                    doubt.append((sample, st, hashes, redo))
                elif changed is not False:
                    redo()
                    if counts is not None:
                        counts["changed"] = counts.get("changed", 0) + 1
        except OSError:
            pass

    # Always refresh these in case file changed
    sample.rel_path = asset.rel_path
    sample.filename = asset.basename
    sample.file_size_bytes = size_on_disk if size_on_disk is not None else asset.file_size
    sample.modified_at = asset.modtime

    # The content hash on first ingest (or if missing): the file's identity (dedup, ratings).
    # (A row an older version wrote has its quick hash only: settle_hashes reads it after the
    # sync and compares.)
    if sample.file_hash is None and sample.quick_hash is None:
        try:
            if p.exists():
                if platforms.cloud_only(p):
                    if counts is not None:
                        counts["cloud_only"] = counts.get("cloud_only", 0) + 1
                else:
                    _set_hashes(sample, p)        # (and the size on disk)
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
        # (whichever rows the sync touched: rows from before schema 3 are settled on the first
        # sync after it, and rows an older version wrote whenever it has)
        counts["read_in_full"] = settle_hashes(session, log_cb=log.info, counts=counts)
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
    doubt: list = []

    def flush_batch(session: Session) -> None:
        for asset in batch:
            try:
                _, created = _upsert_sample_from_sononym(session, asset, reader.library_root, counts, doubt)
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

    own = own_rel_prefixes(reader.library_root)
    with session_scope() as session:
        for asset in stream:
            if _is_own(asset.rel_path, own):
                counts["skipped"] += 1
                continue            # Fourier's own output (own_rel_prefixes)
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
        _settle_moved_times(doubt, counts)

    _rebuild_generic_metadata(counts)
    log.info(
        f"Import complete: {counts['created']} created, {counts['updated']} updated, "
        f"{counts['skipped']} skipped, {counts['errors']} errors"
    )
    return counts


def own_rel_prefixes(library_root: Path) -> tuple[str, ...]:
    """Fourier's own output folders (the master and its .next / .prev, the renders and the
    publish root: releases, device locks) that sit inside this library folder, as paths
    relative to it with a trailing "/". Sononym indexes whatever is under its library, these
    folders too; the scan skips them as the walk does (walk.own_folders), so Fourier never
    takes its own exports back in as library samples."""
    from .. import places
    try:
        master = places.master_dir()
        cands = [master, master + ".next", master + ".prev", places.renders_dir(), places.publish_root()]
    except Exception:
        return ()
    root = os.path.realpath(os.path.expanduser(str(library_root)))
    out = []
    for c in cands:
        real = os.path.realpath(os.path.expanduser(str(c)))
        rel = os.path.relpath(real, root)
        if rel != "." and not rel.startswith(".."):
            out.append(rel.replace(os.sep, "/").rstrip("/") + "/")
    return tuple(out)


def _is_own(rel_path: str, own: tuple[str, ...]) -> bool:
    return bool(own) and (rel_path or "").replace(os.sep, "/").startswith(own)


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
    own = own_rel_prefixes(reader.library_root)
    if own:
        n_own = len(sononym_modtimes)
        sononym_modtimes = {p: m for p, m in sononym_modtimes.items() if not _is_own(p, own)}
        n_own -= len(sononym_modtimes)
        if n_own:
            log.info(f"Skipping {n_own:,} Sononym assets in Fourier's own folders ({', '.join(own)})")

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
        doubt: list = []

        def flush_batch(session: Session) -> None:
            for asset in batch:
                try:
                    _, created = _upsert_sample_from_sononym(session, asset, reader.library_root, counts,
                                                             doubt)
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
            _settle_moved_times(doubt, counts)

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
    """A walked file's library path, header facts, hashes (_set_hashes) and a WAV's own chunks
    (none of them for a cloud-only file, which reading would download). info: file_info(path),
    when the caller has read it already."""
    sample.rel_path = rel
    for k, v in (info if info is not None else (file_info(path) or {})).items():
        if v is not None and getattr(sample, k) is None:
            setattr(sample, k, v)
    if sample.file_hash is None and not platforms.cloud_only(path):
        try:
            _set_hashes(sample, path)
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


def _known_rows(session, facts: dict | None = None) -> dict:
    """{path: (id, rel_path is None, duration_s is None)} for every sample: one query, so the
    walk looks a file up in memory rather than asking the database for each. facts, a dict,
    gets {path: (size, mtime_ns, quick hash, file hash)}: what a scan compares a known file
    against (_check_known)."""
    rows = session.execute(text("SELECT id, path, rel_path, duration_s, file_size_bytes, file_mtime_ns, "
                                "quick_hash, file_hash FROM samples")).all()
    if facts is not None:
        facts.update({p: (size, mt, q, h) for _i, p, _r, _d, size, mt, q, h in rows})
    return {p: (i, rel is None, dur is None) for i, p, rel, dur, *_ in rows}


def _check_known(session, sid: int, path: str, fact, baseline: list, doubt: list) -> str | None:
    """A known file the walk found again: "changed" when its bytes changed (its row refreshed,
    its analysis to be redone: _refresh_changed), "unreadable" when its new bytes can't be
    read, else None. Only a size or time that moved since the last scan reads the file. A row
    with no time yet (scanned before schema 3) takes this scan's as its baseline when its size
    is the same, with no read (baseline, a list, gets (id, mtime_ns) for the caller to store
    at once); with another size or none stored, the file is read and its header decides
    (_content_changed at first sight). A row from before whose time moved with the same size,
    first 8 KB and header goes to doubt, for _settle_moved_times to decide after the walk."""
    size, mtime_ns, quick, full = fact or (None, None, None, None)
    try:
        st = os.stat(path)
    except OSError:
        return None
    if (st.st_size, st.st_mtime_ns) == (size, mtime_ns) or platforms.cloud_only(path):
        return None
    if full is not None and mtime_ns is None and st.st_size == size:
        baseline.append({"i": sid, "t": st.st_mtime_ns})
        return None
    sample = session.get(Sample, sid)
    if sample is None:
        return None
    if full is None and quick is None:     # never hashed (it was cloud-only): hash it now
        try:
            _set_hashes(sample, path, st)
        except OSError:
            pass
        return None
    try:
        changed, hashes = _content_changed(path, st, _row_facts(sample), moved=mtime_ns is not None,
                                           old_ns=mtime_ns)
    except OSError:
        return None
    if changed is None:
        doubt.append((sample, st, hashes,
                      lambda: _refresh_changed(session, sample, path, st, hashes, file_info(path) or {})))
        return None
    if not changed:                        # the same audio (touched only, or a stored size that
                                           # wasn't the file's): hashed in full now
        sample.quick_hash, sample.file_hash = hashes
        sample.file_size_bytes, sample.file_mtime_ns = st.st_size, st.st_mtime_ns
        return None
    info = file_info(path)
    if info is None:
        return "unreadable"
    _refresh_changed(session, sample, path, st, hashes, info)
    return "changed"


def _unsettled(facts: dict, known: dict, created: list) -> list[int]:
    """Ids of the rows settle_hashes reads, found in the facts the walk read (_known_rows) and
    the rows it created, with no query: rows an older version wrote (a quick hash and no
    file_hash), and rows with only the quick hash that another row (or a new file) shares."""
    seen: set = set()
    shared: set = set()
    for _s, _m, q, _h in facts.values():
        if q in seen:
            shared.add(q)
        seen.add(q)
    shared.update(s.quick_hash for s in created if s.quick_hash in seen)
    shared.discard(None)
    return [known[p][0] for p, (size, _m, q, h) in facts.items()
            if q is not None and p in known
            and ((h is None) or (q in shared and not _full_kind(size, q, h)))]


def _update_rows(session, rows: list) -> None:
    """Set these rows' columns ([{"i": id, "t": file_mtime_ns, and "q", "h", "s": quick_hash,
    file_hash, file_size_bytes when given}]; every row the same keys), a few statements in
    all: one UPDATE per row is slow on DuckDB, which takes a list of values in one."""
    if not rows:
        return
    names = {"t": "file_mtime_ns", "q": "quick_hash", "h": "file_hash", "s": "file_size_bytes"}
    keys = [k for k in ("t", "q", "h", "s") if k in rows[0]]

    def lit(k, v):
        return f"'{str(v)}'" if k in ("q", "h") else str(int(v))
    for k in range(0, len(rows), 5000):
        part = rows[k:k + 5000]
        if session.get_bind().dialect.name == "duckdb":
            values = ", ".join("(" + ", ".join([str(int(r["i"]))] + [lit(c, r[c]) for c in keys]) + ")"
                               for r in part)
            sets = ", ".join(f"{names[c]} = v.{c}" for c in keys)
            session.execute(text(f"UPDATE samples SET {sets} FROM (VALUES {values}) AS v(i, {', '.join(keys)}) "
                                 f"WHERE samples.id = v.i"))
        else:
            sets = ", ".join(f"{names[c]} = :{c}" for c in keys)
            session.execute(text(f"UPDATE samples SET {sets} WHERE id = :i"), part)


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
    to it with the same content (quick hash and size, and the content hash where both rows
    have one; the same file name first, and only where the match is one to one), or a file marked missing that is back where it was. A new
    row gets the old one's analysis (sample_features), so a renamed folder isn't analyzed
    again; the old row is marked missing as any file the walk didn't find. Returns {old path:
    new path}."""
    from ..db.models import SampleFeatures
    new = [s for s in created if (s.quick_hash or s.file_hash) and s.id is not None]
    # ...and files marked missing that are back where they were (a folder renamed back):
    # their rows have their analysis, a rating that followed them away comes back
    try:
        flagged = {i for (i,) in session.execute(text("SELECT sample_id FROM missing_files"))}
    except Exception:
        flagged = set()
    back = [session.get(Sample, sid) for sid in sorted(flagged & seen_ids)]
    new += [s for s in back if s is not None and (s.quick_hash or s.file_hash)]
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
        for sid, q, h, size in session.execute(text(
                f"SELECT id, quick_hash, file_hash, file_size_bytes FROM samples WHERE id IN ({chunk})")):
            if q or h:
                by_key[(q or h, size)].append((sid, h if _full_kind(size, q, h) else None))
    cands = defaultdict(list)            # old id -> new samples with its content
    for s in new:
        for sid, h in by_key.get((s.quick_hash or s.file_hash, s.file_size_bytes), ()):
            # (the first 8 KB and the size, and every byte where both rows hashed them all)
            if h is None or not _full_kind(s.file_size_bytes, s.quick_hash, s.file_hash) or h == s.file_hash:
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

    A known file whose size or time moved since the last scan is read again: when its bytes
    changed, its row takes the new facts and its analysis is redone (_check_known; a file only
    touched keeps it when its row has the content hash; a row scanned before schema 3 has only
    the quick hash: a moved time on it with the same size, first 8 KB and header is a change
    for a few files and a copy without times for many, _settle_moved_times). Files whose first 8 KB another file shares are read in full (settle_hashes), so
    file_hash tells every two files apart.

    Returns counts: created, skipped (already known), changed (known, bytes changed),
    times_moved (rows from before whose times moved with their bytes kept, _settle_moved_times),
    read_in_full (settle_hashes), filled, errors, error_files,
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
        facts: dict = {}
        baseline: list = []
        doubt: list = []
        known = _known_rows(session, facts)
        unchunked = _unchunked(session)
        seen_ids: set = set()
        created: list = []
        for i, f in enumerate(w.files):
            if progress_cb:
                progress_cb(i, f.path)
            try:
                at = f.key
                hit = known.get(f.key)
                if hit is None and f.key != _logical(w.root, f.rel):
                    at = _logical(w.root, f.rel)
                    hit = known.get(at)                        # a row stored at the root as spelled
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
                        _fill_scanned(session.get(Sample, sid), Path(f.path), f.rel, info)  # type: ignore[arg-type]
                        counts["filled"] = counts.get("filled", 0) + 1
                    else:
                        # (its bytes changed at the same path: read again, analyzed again)
                        got = _check_known(session, sid, f.path, facts.get(at), baseline, doubt)
                        if got == "unreadable":               # left out: marked missing below
                            error(f.path, "can't be read")
                            continue
                        if got == "changed":
                            counts["changed"] = counts.get("changed", 0) + 1
                        elif sid in unchunked and _fill_chunks(session.get(Sample, sid), f.path):  # type: ignore[arg-type]
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
        _settle_moved_times(doubt, counts)
        _update_rows(session, baseline)
        session.flush()
        # (a new or changed row may share its first 8 KB with a row scanned before schema 3)
        todo = _unsettled(facts, known, created)
        counts["read_in_full"] = settle_hashes(session, counts=counts, ids=todo) if todo else 0
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
