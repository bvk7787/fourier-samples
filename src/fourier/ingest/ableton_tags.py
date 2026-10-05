"""Read Ableton Live's automated sample tags as a second-opinion signal.

Ableton Live keeps a SQLite index at
~/Library/Application Support/Ableton/Live Database/Live-files-<ver>.db. Each
sample has keyword rows (keywords.is_auto=1 marks Live's ML auto-tags); the
keyword string is another files row (keywords.keyw_id -> files.name).

Live analyses the WHOLE library tree it can see, not just Ableton's own packs,
and stores full absolute POSIX paths. So we match every library sample against
the Live index by absolute path (with (parent-dir, filename) and unambiguous-stem
fallbacks) and store the tags on Sample.ableton_tags for curation to use
alongside Sononym - to corroborate, and to rescue samples Sononym left
uncategorised or dumped as noise.
"""
from __future__ import annotations

import glob
import os
import re
import sqlite3
from collections import defaultdict

from .formats import LIVE_TAG_EXTS as AUDIO_EXT     # the default: Live's list as it was


def latest_live_db():
    base = os.path.expanduser("~/Library/Application Support/Ableton/Live Database")
    dbs = [d for d in glob.glob(os.path.join(base, "Live-files-*.db")) if d.endswith(".db")]
    if not dbs:
        return None
    return max(dbs, key=lambda p: int(re.search(r"Live-files-(\d+)\.db$", p).group(1)))


def _norm(p):
    if not p:
        return None
    p = re.sub(r"/+", "/", p)
    return p.rstrip("/")


def read_ableton_tags(db_path=None, auto_only=True, exts=AUDIO_EXT):
    """Return (by_abspath, by_pn, stem_map) from the Ableton index, for the files with one
    of exts (lowercase, with the dot):
    - by_abspath: {normalized absolute path: [tags]}
    - by_pn:      {(parent-dir name, filename): [tags]}
    - stem_map:   {filename stem: [tags]}  (only when unambiguous across the index)
    """
    db_path = db_path or latest_live_db()
    if not db_path or not os.path.exists(db_path):
        return {}, {}, {}
    con = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
    c = con.cursor()
    files = {}
    for fid, name, parent in c.execute("select file_id, name, parent_id from files"):
        files[fid] = (name, parent)

    cache = {}
    def fullpath(fid):
        if fid in cache:
            return cache[fid]
        parts, cur, seen = [], fid, set()
        while cur in files and cur not in seen:
            seen.add(cur)
            nm, par = files[cur]
            parts.append(nm)
            cur = par
        p = "/".join(reversed(parts))
        if not p.startswith("/"):
            p = "/" + p
        p = re.sub(r"/+", "/", p)
        cache[fid] = p
        return p

    tags_of = defaultdict(list)
    q = "select file_id, keyw_id, is_auto from keywords"
    if auto_only:
        q += " where is_auto=1"
    for fid, kid, _auto in c.execute(q):
        kn = files.get(kid, (None,))[0]
        if kn:
            tags_of[fid].append(kn)

    by_abspath, by_pn, by_stem = {}, {}, defaultdict(set)
    for fid, tags in tags_of.items():
        nm = files.get(fid, (None,))[0]
        if not nm or not nm.lower().endswith(tuple(exts)):
            continue
        ap = _norm(fullpath(fid))
        tset = tuple(sorted(set(tags)))
        by_abspath[ap] = list(tset)
        by_pn[(os.path.basename(os.path.dirname(ap)), nm)] = list(tset)
        by_stem[os.path.splitext(nm)[0]].add(tset)
    con.close()
    stem_map = {st: list(next(iter(s))) for st, s in by_stem.items() if len(s) == 1}
    return by_abspath, by_pn, stem_map


def _ensure_column(session):
    from sqlalchemy import text
    try:
        session.execute(text("ALTER TABLE samples ADD COLUMN ableton_tags TEXT"))
        session.commit()
    except Exception:
        session.rollback()  # already exists


def import_ableton_tags(session, db_path=None, log=print, exts=AUDIO_EXT):
    """Populate Sample.ableton_tags for EVERY library sample Live has analysed,
    matched by absolute path (with (parent-dir, filename) and unambiguous-stem
    fallbacks), among Live's files with one of exts. Returns stats."""
    import json
    from sqlalchemy import select, text
    from ..db.models import Sample

    _ensure_column(session)
    by_abspath, by_pn, stem_map = read_ableton_tags(db_path, exts=exts)
    if not by_abspath:
        log("No Ableton tags found (Live Database missing?)")
        return dict(matched=0, total=0)

    rows = session.execute(select(Sample.id, Sample.path, Sample.filename)).all()
    matched = m_abs = m_pn = m_stem = 0
    for sid, pth, fn in rows:
        tags = None
        how = None
        npth = _norm(pth)
        if npth and npth in by_abspath:
            tags, how = by_abspath[npth], "abs"
        if tags is None and pth and fn:
            hit = by_pn.get((os.path.basename(os.path.dirname(pth)), fn))
            if hit:
                tags, how = hit, "pn"
        if tags is None and fn:
            hit = stem_map.get(os.path.splitext(fn)[0])
            if hit:
                tags, how = hit, "stem"
        if tags:
            session.execute(
                text("UPDATE samples SET ableton_tags = :t WHERE id = :i"),
                {"t": json.dumps(tags), "i": sid})
            matched += 1
            if how == "abs":
                m_abs += 1
            elif how == "pn":
                m_pn += 1
            else:
                m_stem += 1
    session.commit()
    from ..metadata.store import ABLETON, rebuild     # the generic labels curation reads
    rebuild(session, ABLETON, log=log)
    session.commit()
    log(f"Ableton tags: matched {matched:,} of {len(rows):,} library samples "
        f"({100*matched/max(len(rows),1):.0f}%) [abs={m_abs:,} pn={m_pn:,} stem={m_stem:,}]; "
        f"{len(by_abspath):,} tagged files in the Live index")
    return dict(matched=matched, total=len(rows), abs=m_abs, pn=m_pn, stem=m_stem)
