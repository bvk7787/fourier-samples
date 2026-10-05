"""Provider metadata in the generic tables: labels(sample_id, provider, kind, label, rank,
confidence) and descriptors(sample_id, provider, name, value).

Curation reads no one tool's schema (docs/design-history.md, "How it became configurable"). Each provider
below is copied from where its importer writes it into the generic tables, and
`parity()` proves the copy carries exactly the same information. Curation reads only the
generic tables (metadata/rows.py). The rows are derived: `rebuild()` replaces a provider's
rows from its source; `fourier tools scan` (and so `fourier build`) runs it,
and rows.ensure_current() runs it when the counts drift.

  sononym   source sononym_meta (Sononym's analysis)
            labels       kind "class"    classes, in order ("OneShot", "Loop")
                         kind "category" categories, in order; confidence None, since
                                         Sononym's strength vectors aren't aligned with the lists
            descriptors  the numeric columns, in their own units (SONONYM_DESCRIPTORS)
            labels       kind "canonical": classes and categories in the canonical
                         vocabulary (config/providers/sononym.yaml, metadata/vocab.py),
                         which is what the taxonomy's predicates read
  ableton   source samples.ableton_tags (Live's auto-tags, `fourier tools scan --only ableton`)
            labels       kind "tag", in order

Every sample a provider covers gets the descriptor "analysed" = 1, so a sample it analysed
but gave no labels (many Sononym samples have no category) still counts as covered.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import insert, text

from ..db.models import Label

SONONYM = "sononym"
ABLETON = "ableton"
ANALYSED = "analysed"
SONONYM_DESCRIPTORS = ("brightness", "noisiness", "harmonicity", "crest_factor", "base_note",
                       "base_note_confidence", "bpm", "bpm_confidence", "peak_db", "rms_db",
                       "pitch_confidence")
_CHUNK = 20000


@dataclass(frozen=True)
class Source:
    table: str                  # where the provider's importer writes
    id_col: str                 # that table's sample id column
    covered: str                # SQL condition: the provider analysed this sample
    lists: tuple                # ((labels.kind, JSON list column), ...)
    descriptors: tuple = ()     # numeric columns copied as descriptors


SOURCES = {
    SONONYM: Source("sononym_meta", "sample_id", "1 = 1",
                    (("class", "classes"), ("category", "categories")), SONONYM_DESCRIPTORS),
    ABLETON: Source("samples", "id", "ableton_tags IS NOT NULL", (("tag", "ableton_tags"),)),
}
SONONYM_LISTS = SOURCES[SONONYM].lists


def _dialect(session) -> str:
    return session.get_bind().dialect.name


def covered_count(session, provider: str) -> tuple[int, int]:
    """(samples the source covers, samples with the provider's "analysed" marker)."""
    s = SOURCES[provider]
    need = session.execute(text(f"SELECT count(*) FROM {s.table} WHERE {s.covered}")).scalar()
    have = session.execute(text("SELECT count(*) FROM descriptors WHERE provider = :p AND name = :n"),
                           {"p": provider, "n": ANALYSED}).scalar()
    return need, have


def rebuild(session, provider: str, log=lambda m: None) -> dict:
    """Replace one provider's labels and descriptors from its source. Returns
    {"labels": n, "descriptors": n}."""
    s = SOURCES[provider]
    p = {"p": provider}
    session.execute(text("DELETE FROM labels WHERE provider = :p"), p)
    session.execute(text("DELETE FROM descriptors WHERE provider = :p"), p)
    session.execute(text(
        f"INSERT INTO descriptors (sample_id, provider, name, value) "
        f"SELECT {s.id_col}, :p, :n, 1.0 FROM {s.table} WHERE {s.covered}"), {**p, "n": ANALYSED})
    for name in s.descriptors:
        session.execute(text(
            f"INSERT INTO descriptors (sample_id, provider, name, value) "
            f"SELECT {s.id_col}, :p, :n, {name} FROM {s.table} WHERE {name} IS NOT NULL"),
            {**p, "n": name})
    for kind, col in s.lists:
        if _dialect(session) == "sqlite":
            session.execute(text(
                f"INSERT INTO labels (sample_id, provider, kind, label, rank, confidence) "
                f"SELECT m.{s.id_col}, :p, :k, j.value, j.key, NULL "
                f"FROM {s.table} m, json_each(m.{col}) j WHERE m.{col} IS NOT NULL"),
                {**p, "k": kind})
        else:                                  # portable: decode in Python, insert in chunks
            batch = []
            for sid, raw in session.execute(text(
                    f"SELECT {s.id_col}, {col} FROM {s.table} WHERE {col} IS NOT NULL")):
                for i, lab in enumerate(json.loads(raw) or []):
                    batch.append(dict(sample_id=sid, provider=provider, kind=kind, label=lab,
                                      rank=i, confidence=None))
                if len(batch) >= _CHUNK:
                    session.execute(insert(Label), batch)
                    batch = []
            if batch:
                session.execute(insert(Label), batch)
    _write_canonical(session, provider)
    n_lab = session.execute(text("SELECT count(*) FROM labels WHERE provider = :p"), p).scalar()
    n_des = session.execute(text("SELECT count(*) FROM descriptors WHERE provider = :p"), p).scalar()
    log(f"{provider} metadata: {n_lab:,} labels, {n_des:,} descriptors")
    return {"labels": n_lab, "descriptors": n_des}


CANONICAL_KIND = "canonical"


def _write_canonical(session, provider: str) -> None:
    """The provider's raw labels mapped into the canonical vocabulary (kind "canonical").
    Several raw labels mapping to one canonical label give one row, at the first rank."""
    from .vocab import mapping
    by_canon = defaultdict(list)
    for (kind, raw), canon in mapping(provider).items():
        by_canon[canon].append((kind, raw))
    for i, (canon, raws) in enumerate(sorted(by_canon.items())):
        cond = " OR ".join(f"(kind = :k{i}_{j} AND label = :r{i}_{j})" for j in range(len(raws)))
        params = {"p": provider, "c": canon, "ck": CANONICAL_KIND}
        for j, (kind, raw) in enumerate(raws):
            params[f"k{i}_{j}"], params[f"r{i}_{j}"] = kind, raw
        session.execute(text(
            f"INSERT INTO labels (sample_id, provider, kind, label, rank, confidence) "
            f"SELECT sample_id, :p, :ck, :c, min(rank), NULL FROM labels "
            f"WHERE provider = :p AND ({cond}) GROUP BY sample_id"), params)


def unmapped(session, provider: str) -> dict:
    """{(kind, raw label): samples} for the provider's raw labels with no canonical mapping."""
    from .vocab import KINDS, mapping
    m = mapping(provider)
    if not m:
        return {}
    kinds = sorted(set(KINDS.values()))
    rows = session.execute(text(
        f"SELECT kind, label, count(*) FROM labels WHERE provider = :p AND kind IN "
        f"({', '.join(repr(k) for k in kinds)}) GROUP BY kind, label"), {"p": provider}).all()
    return {(k, lab): n for k, lab, n in rows if (k, lab) not in m}


def rebuild_sononym(session, log=lambda m: None) -> dict:
    return rebuild(session, SONONYM, log)


def parity(session, provider: str, examples: int = 10) -> dict:
    """Compare a provider's generic rows with its source, field by field, for every sample.
    Returns {"samples": n, "differences": {field: count}, "examples": [(sample_id, field,
    in the source, rebuilt)]}; no differences means they say exactly the same."""
    s = SOURCES[provider]
    diffs, ex = defaultdict(int), []

    def note(sid, field, old, new):
        diffs[field] += 1
        if len(ex) < examples:
            ex.append((sid, field, old, new))

    marked = {sid for (sid,) in session.execute(text(
        "SELECT sample_id FROM descriptors WHERE provider = :p AND name = :n"),
        {"p": provider, "n": ANALYSED})}
    got = defaultdict(lambda: defaultdict(list))
    for sid, kind, lab in session.execute(text(
            "SELECT sample_id, kind, label FROM labels WHERE provider = :p "
            "ORDER BY sample_id, kind, rank"), {"p": provider}):
        got[sid][kind].append(lab)

    # lists, rebuilt as the JSON text the source stores (json.dumps); a covered sample with
    # none reads '[]', and a NULL list reads the same as '[]' everywhere curation looks
    cols = ", ".join(c for _, c in s.lists)
    n, seen = 0, set()
    for row in session.execute(text(f"SELECT {s.id_col}, {cols} FROM {s.table} WHERE {s.covered}")):
        sid, raws = row[0], row[1:]
        n += 1
        seen.add(sid)
        if sid not in marked:
            note(sid, ANALYSED, "covered", "no marker")
        for (kind, _), raw in zip(s.lists, raws):
            rebuilt = json.dumps(got[sid][kind])
            if (raw or "[]") != rebuilt:
                note(sid, kind, raw, rebuilt)
    for sid in (set(got) | marked) - seen:
        note(sid, "orphan rows", None, ANALYSED)

    # descriptors: null-safe equality per column
    for name in s.descriptors:
        bad = session.execute(text(
            f"SELECT m.{s.id_col}, m.{name}, d.value FROM {s.table} m "
            f"LEFT JOIN descriptors d ON d.sample_id = m.{s.id_col} AND d.provider = :p AND d.name = :n "
            f"WHERE CASE WHEN m.{name} IS NULL AND d.value IS NULL THEN 0 "
            f"WHEN m.{name} IS NULL OR d.value IS NULL THEN 1 "
            f"WHEN m.{name} = d.value THEN 0 ELSE 1 END = 1"),
            {"p": provider, "n": name}).all()
        for sid, old, new in bad:
            note(sid, name, old, new)
    for (kind, raw), cnt in unmapped(session, provider).items():
        note(None, "unmapped", f"{kind}:{raw}", cnt)
    return {"samples": n, "differences": dict(diffs), "examples": ex}


def sononym_parity(session, examples: int = 10) -> dict:
    return parity(session, SONONYM, examples)
