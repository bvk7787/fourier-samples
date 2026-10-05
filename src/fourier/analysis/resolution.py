"""Persist the two-classifier (Sononym x Ableton) resolution per sample.

Curation resolves each sample to a single home category from both classifiers
(union membership, agreement-weighted, CLAP breaks conflicts). That resolution is
useful beyond a single build: this materialises it into a `sample_resolution`
table so the second opinion is queryable across the whole library, e.g.

  -- where the two classifiers disagree
  SELECT s.filename, r.son_cats, r.ab_cats, r.home
  FROM sample_resolution r JOIN samples s ON s.id = r.sample_id
  WHERE r.conflict = 1;
"""
from __future__ import annotations

import json

from sqlalchemy import text

DDL = """
CREATE TABLE IF NOT EXISTS sample_resolution (
    sample_id  INTEGER PRIMARY KEY,
    home       TEXT,
    support    INTEGER,
    conflict   INTEGER,
    son_cats   TEXT,
    ab_cats    TEXT,
    son_labels TEXT,
    ab_tags    TEXT
)
"""


def persist_resolution(session, log=print):
    """Compute the two-classifier resolution and write it to sample_resolution.
    Returns the number of rows written."""
    import numpy as np
    from ..packs.curate import load_index, compute_homes

    ids, emb = load_index()
    emb = emb.astype("float32")
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    id2row = {int(i): r for r, i in enumerate(ids)}

    homes, support, votes = compute_homes(session, emb_n, id2row, log=log)

    session.execute(text(DDL))
    session.execute(text("DELETE FROM sample_resolution"))
    n = 0
    for sid, home in homes.items():
        v = votes.get(sid, {})
        son = v.get("son_cats", [])
        ab = v.get("ab_cats", [])
        conflict = int(bool(son) and bool(ab) and set(son) != set(ab))
        session.execute(
            text("INSERT INTO sample_resolution"
                 "(sample_id, home, support, conflict, son_cats, ab_cats, son_labels, ab_tags) "
                 "VALUES (:i,:h,:s,:c,:sc,:ac,:sl,:at)"),
            dict(i=sid, h=home, s=int(support.get((sid, home), 1)), c=conflict,
                 sc=json.dumps(son), ac=json.dumps(ab),
                 sl=json.dumps(v.get("son_labels", [])), at=json.dumps(v.get("ab_tags", []))))
        n += 1
        if n % 5000 == 0:
            session.commit()
    session.commit()

    tot = session.execute(text("SELECT COUNT(*) FROM sample_resolution")).scalar()
    conf = session.execute(text("SELECT COUNT(*) FROM sample_resolution WHERE conflict=1")).scalar()
    agree = session.execute(text("SELECT COUNT(*) FROM sample_resolution WHERE support>=2")).scalar()
    log(f"Resolution persisted: {tot:,} samples, {agree:,} corroborated, {conf:,} conflicts")
    return n
