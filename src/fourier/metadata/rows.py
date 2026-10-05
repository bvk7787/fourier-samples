"""SampleRow: the one place that knows which tables a sample's metadata lives in.

Curation reads a sample as a row of named fields (id, rel_path, classes, categories,
brightness, tempo_bpm, ...). Everything that selects samples builds its query with
`sample_select(*fields)`, narrows it with `like_any(...)`, `.where()`, `.order_by()`, and
runs it with `fetch(session, query)`, which returns rows with exactly those fields.

Where the fields come from (docs/curation.md, "Sources and resolution"):
  samples            the file: id, path, rel_path, filename, duration, format, hash
  labels             the classifier's classes and categories, and Live's auto-tags
                     (ableton_tags), in the provider's order, returned as lists as the
                     JSON columns were (["Perc Kicks"]); ableton_tags is None for a sample
                     Live never tagged
  descriptors        the classifier's measurements in their own units (brightness, bpm,
                     base_note, ...); the sononym_* fields read Sononym's whichever classifies
  sample_features    Fourier's own measurements (fourier tools analyze), and the legacy
                     columns computed from Sononym's (bpm_reliable, sub_weight)

Every field has one source for a given set of providers: field_source() names it, in
metadata/resolve.py's source names, and the resolver (resolve.py) says which source each
value the build uses came from.

A sample the last walk of its library folder marked missing (moved, renamed, deleted or
unreadable: table missing_files) is no candidate: sample_select leaves it out.

The classifier is Sononym when the build uses it, else the fallback providers, path and
audio (metadata/providers.py decides; pass the session to sample_select and like_any so
they follow it). A sample is a candidate only if the classifier looked at it (its
"analysed" descriptor): with Sononym an `import scan` file nobody analysed never is; with
the fallback every file is. Features are optional. Without Sononym, "classes" is the
fallback's shape ("OneShot" / "Loop") and "categories" (Sononym's own names) is empty.

labels and descriptors are rebuilt by the importers (metadata/store.py, shadow.py).
`ensure_current()` rebuilds a provider in use when its sample count doesn't match its
source (a database from before they existed, a test that wrote a source directly); a
worker process never rebuilds, it stops and says so.
"""
from __future__ import annotations

import os
from collections import defaultdict, namedtuple
from typing import Any, Protocol

from sqlalchemy import Select, and_, func, literal, select, text
from sqlalchemy.orm import aliased

from ..db.models import Descriptor, MissingFile, Sample, SampleFeatures
from . import shadow
from .providers import Active, active
from .store import ABLETON, ANALYSED, CANONICAL_KIND, SONONYM, SOURCES, covered_count, rebuild
from . import resolve as _res

CLASSIFIER = "classifier"   # LABEL_FIELDS role: whichever providers classify (providers.py)
_SONONYM_ONLY = Active((SONONYM, ABLETON), (SONONYM,), (ABLETON,), "default")
_SHAPE = {"class.oneshot": "OneShot", "class.loop": "Loop"}    # canonical class -> "classes"

_SAMPLE = {
    "id": Sample.id, "rel_path": Sample.rel_path, "path": Sample.path, "filename": Sample.filename,
    "file_hash": Sample.file_hash, "file_format": Sample.file_format, "duration_s": Sample.duration_s,
    "sample_rate": Sample.sample_rate, "acid_bpm": Sample.acid_bpm, "acid_beats": Sample.acid_beats,
    "root_note": Sample.root_note,
}
# field -> (provider or CLASSIFIER, labels.kind)
LABEL_FIELDS = {"classes": (CLASSIFIER, "class"), "categories": (CLASSIFIER, "category"),
                "canonical": (CLASSIFIER, CANONICAL_KIND),   # the classifier's, canonical vocabulary
                "ableton_tags": (ABLETON, "tag"),
                "sononym_classes": (SONONYM, "class")}     # Sononym's, whichever classifies
_OPTIONAL = {ABLETON}       # providers that cover only part of the library: None when not
DESCRIPTOR_FIELDS = ("brightness", "noisiness", "harmonicity", "crest_factor", "base_note",
                     "base_note_confidence", "bpm", "peak_db")
_FEATURES = {
    "attack_time_ms": SampleFeatures.attack_time_ms, "decay_time_ms": SampleFeatures.decay_time_ms,
    "sub_weight": SampleFeatures.sub_weight, "is_clipped": SampleFeatures.is_clipped,
    "tempo_bpm": SampleFeatures.tempo_bpm, "bpm_reliable": SampleFeatures.bpm_reliable,
    "onset_rate_hz": SampleFeatures.onset_rate_hz,
    "chroma_concentration": SampleFeatures.chroma_concentration,
    "spectral_flatness_mean": SampleFeatures.spectral_flatness_mean, "rms_mean": SampleFeatures.rms_mean,
    "dc_offset_ratio": SampleFeatures.dc_offset_ratio, "n_events": SampleFeatures.n_events,
    "event_regularity": SampleFeatures.event_regularity, "event_echo": SampleFeatures.event_echo,
    "harmonic_percussive_ratio": SampleFeatures.harmonic_percussive_ratio,
    "own_root_midi": SampleFeatures.own_root_midi, "own_root_at": SampleFeatures.own_root_at,
    "detected_key": SampleFeatures.detected_key, "key_confidence": SampleFeatures.key_confidence,
}
# a descriptor of one provider, whichever classifies (None when the build doesn't use it)
PINNED_DESCRIPTORS = {
    "sononym_bpm": (SONONYM, "bpm"), "sononym_harmonicity": (SONONYM, "harmonicity"),
    "sononym_pitch_confidence": (SONONYM, "pitch_confidence"),
    "sononym_base_note": (SONONYM, "base_note"),
    "sononym_base_note_confidence": (SONONYM, "base_note_confidence"),
}
FIELDS = (*_SAMPLE, *LABEL_FIELDS, *DESCRIPTOR_FIELDS, *_FEATURES, *PINNED_DESCRIPTORS)

# the source of each field that isn't the classifier's (field_source)
_FILE = "file"                      # the file itself: its path, header and hash
_SAMPLE_SOURCE = {"acid_bpm": _res.ACID, "acid_beats": _res.ACID, "root_note": _res.CHUNK}
_SONONYM_FEATURES = {"sub_weight", "bpm_reliable"}     # legacy, computed from Sononym's
# without Sononym, the fallback providers' descriptors: harmonicity is the audio provider's
# (HPSS), bpm the path provider's (a tempo written in the path); the rest none gives
_FALLBACK_DESCRIPTOR = {"harmonicity": _res.AUDIO, "bpm": _res.NAME}


def field_source(f: str, act: Active | None = None) -> str | None:
    """Where a field's value comes from, for these providers (None: Sononym and Live): one
    source per field ("file", "ableton", or a resolve.py source; "fourier:name+audio" for
    the fallback's labels, the name's where it has one, else the audio's). None: no
    provider in use gives it."""
    act = act or _SONONYM_ONLY
    if f in _SAMPLE:
        return _SAMPLE_SOURCE.get(f, _FILE)
    if f in LABEL_FIELDS:
        prov, _kind = LABEL_FIELDS[f]
        if prov == CLASSIFIER:
            return _res.SONONYM if not act.fallback else f"{_res.NAME}+audio"
        if prov not in act.names:
            return None
        return _res.SONONYM if prov == SONONYM else prov
    if f in DESCRIPTOR_FIELDS:
        return _res.SONONYM if not act.fallback else _FALLBACK_DESCRIPTOR.get(f)
    if f in PINNED_DESCRIPTORS:
        return _res.SONONYM if PINNED_DESCRIPTORS[f][0] in act.names else None
    if f in _FEATURES:
        return _res.SONONYM if f in _SONONYM_FEATURES else _res.AUDIO
    raise KeyError(f)

# The field sets curation reads.
HOME_FIELDS = ("id", "rel_path", "path", "filename", "categories", "classes", "ableton_tags",
               "harmonicity", "chroma_concentration", "n_events", "duration_s", "file_hash",
               "bpm", "tempo_bpm", "onset_rate_hz", "canonical",
               "acid_bpm", "acid_beats")     # routing: compute_homes, fourier why
CANDIDATE_FIELDS = (
    "id", "rel_path", "path", "filename", "file_hash", "file_format", "duration_s", "ableton_tags",
    "sample_rate", "brightness", "noisiness", "harmonicity", "crest_factor", "base_note",
    "base_note_confidence", "bpm", "classes", "peak_db", "categories",
    "attack_time_ms", "decay_time_ms", "sub_weight", "is_clipped", "tempo_bpm", "bpm_reliable",
    "onset_rate_hz", "chroma_concentration", "spectral_flatness_mean", "rms_mean",
    "dc_offset_ratio", "n_events", "event_regularity", "event_echo", "canonical",
    "acid_bpm", "acid_beats", "root_note",
)                                                     # selection: a category's candidates

_SID = "_sid"               # the sample id fetch() needs, when the caller didn't ask for it
_HAS = "_has_"              # + provider: the sample's "analysed" marker, for optional providers


class SampleRow(Protocol):
    """What fetch() returns: a named tuple with one attribute per requested field (None when
    the source has no value), in the order the fields were asked."""
    id: int
    rel_path: str | None
    path: str
    filename: str | None
    classes: list              # e.g. ["OneShot"]
    categories: list           # classifier categories ([] for none)
    ableton_tags: list | None  # Live's auto-tags (None when never tagged)


def _descriptor(name):
    return aliased(Descriptor, name=f"d_{name}")


def _active(session) -> Active:
    return active(session) if session is not None else _SONONYM_ONLY


def sample_select(*fields: str, session=None) -> Select:
    """A select of these fields over every sample the classifier looked at; run it with
    fetch(). With a session it follows the providers that database and config use; without
    one it assumes Sononym and Live."""
    unknown = [f for f in fields if f not in FIELDS]
    if unknown:
        raise KeyError(f"unknown sample fields: {', '.join(unknown)}")
    act = _active(session)
    marker = _descriptor(ANALYSED)
    cols: list[Any] = []
    joins: list[Any] = []
    for f in fields:
        if f in _SAMPLE:
            cols.append(_SAMPLE[f].label(f))
        elif f in _FEATURES:
            cols.append(_FEATURES[f].label(f))
        elif f in LABEL_FIELDS:
            cols.append(literal(None).label(f))          # filled in by fetch()
        elif f in PINNED_DESCRIPTORS:
            prov, name = PINNED_DESCRIPTORS[f]
            if prov not in act.names:
                cols.append(literal(None).label(f))
                continue
            d = aliased(Descriptor, name=f"p_{f}")
            joins.append((d, prov, name))
            cols.append(d.value.label(f))
        else:                                            # a measurement: the first classifier's that has it
            ds = [(aliased(Descriptor, name=f"d_{f}_{prov}"), prov) for prov in act.classifiers]
            joins += [(d, prov, f) for d, prov in ds]
            cols.append((ds[0][0].value if len(ds) == 1 else func.coalesce(*(d.value for d, _ in ds))).label(f))
    if "id" not in fields and any(f in LABEL_FIELDS for f in fields):
        cols.append(Sample.id.label(_SID))
    optional = sorted({LABEL_FIELDS[f][0] for f in fields if f in LABEL_FIELDS} & _OPTIONAL & set(act.names))
    has = {prov: aliased(Descriptor, name=f"has_{prov}") for prov in optional}
    cols += [has[prov].value.label(_HAS + prov) for prov in optional]
    q = (select(*cols).select_from(Sample)
         .join(marker, and_(marker.sample_id == Sample.id, marker.provider == act.classifiers[0],
                            marker.name == ANALYSED)))
    for prov, h in has.items():
        q = q.outerjoin(h, and_(h.sample_id == Sample.id, h.provider == prov, h.name == ANALYSED))
    for d, prov, name in joins:
        q = q.outerjoin(d, and_(d.sample_id == Sample.id, d.provider == prov, d.name == name))
    if any(f in _FEATURES for f in fields):
        q = q.outerjoin(SampleFeatures, SampleFeatures.sample_id == Sample.id)
    if session is not None and missing_marked(session):
        q = q.where(~select(MissingFile.sample_id).where(
            MissingFile.sample_id == Sample.id, MissingFile.path == Sample.path).exists())
    if session is not None:
        other = outside_library(session)
        if other:
            q = q.where(text(f"samples.id NOT IN ({_id_list(other)})"))
    return q


def _id_list(ids) -> str:
    return ", ".join(str(int(i)) for i in sorted(ids))


# ---------------------------------------------------------------------------
# Other libraries in the same home
# ---------------------------------------------------------------------------

_OUTSIDE: dict = {}


def outside_library(session) -> frozenset:
    """Ids of the samples under none of the configured library folders: another library
    whose config (`fourier --config`) shares this home's database. Curation, the library
    scale, doctor, `build --dry-run`, `why --unrecognized` and search leave them out. Empty
    (nothing left out) unless the database holds samples under a configured folder and
    others under none: a database whose every sample is under the configured folders is
    read exactly as before, and so is one whose library isn't configured (no `library`).
    When none of its samples is under the configured folders, the library isn't scanned
    yet if one of those folders is on disk (library_unscanned: every sample is another
    library's, all left out); a configured folder that isn't there (an unmounted drive, a
    library that moved, a bare folder name) leaves nothing out, and the build's own checks
    say so.

    A sample is under a folder as places.library_rel reads it (the folder, its real path, a
    bare folder name anywhere in the path, a Windows spelling), or, failing that, the same
    folder on disk spelled another way (places.library_root_of). Worked out once per
    database, library and sample count in a process."""
    from .. import places
    try:
        roots, names = places.library()
    except Exception:
        return frozenset()
    if not roots and not names:
        return frozenset()
    try:
        total, top = session.execute(text("SELECT COUNT(*), MAX(id) FROM samples")).one()
    except Exception:
        return frozenset()
    if not total:
        return frozenset()
    key = (str(session.get_bind().url), roots, names, total, top)
    if key in _OUTSIDE:
        return _OUTSIDE[key]
    # the folders' spellings as LIKE patterns: any sample one matches is inside (LIKE's
    # wildcards and SQLite's case folding only ever count more samples in); the rest are
    # checked one by one
    pats = []
    for r in roots:
        pats.append(r.rstrip("/") + "/%")
    for _r, real in places._real_roots(roots):
        pats.append(real.rstrip("/") + "/%")
    for n in names:
        pats += [f"%/{n}/%", f"{n}/%"]
    params = {f"p{i}": v for i, v in enumerate(pats)}
    cond = " OR ".join(f"path LIKE :{k}" for k in params) or "1 = 0"
    rows = session.execute(text(f"SELECT id, path FROM samples WHERE NOT ({cond})"), params).all()
    out = frozenset()
    if rows and (len(rows) < total or _roots_on_disk(roots)):
        under: dict = {}

        def inside(path):
            if places.in_library(path):
                return True
            d = os.path.dirname(path or "")
            if d not in under:
                under[d] = places.library_root_of(path) is not None
            return under[d]
        out = frozenset(i for i, p in rows if not inside(p or ""))
        if len(out) >= total and not _roots_on_disk(roots):
            out = frozenset()                 # none inside, and no folder to scan: as before
    _OUTSIDE.clear()
    _OUTSIDE[key] = out
    return out


def _roots_on_disk(roots) -> bool:
    """Whether a configured library folder is on disk (a bare folder name can't be checked)."""
    return any(os.path.isdir(r) for r in roots)


def library_unscanned(session) -> bool:
    """Whether this config's library isn't scanned yet: its folders are on disk, the database
    holds samples, and none of them is under those folders (they're another config's,
    outside_library leaves them all out). The first scan (`fourier tools scan`, or a build)
    reads it."""
    other = outside_library(session)
    if not other:
        return False
    try:
        total = session.execute(text("SELECT COUNT(*) FROM samples")).scalar() or 0
    except Exception:
        return False
    return len(other) >= total


def outside_note(session) -> str | None:
    """The line a build, doctor and `build --dry-run` print when samples of other libraries
    are left out, else None."""
    n = len(outside_library(session))
    if not n:
        return None
    if library_unscanned(session):
        return (f"This config's library isn't scanned yet: the {n:,} sample{'' if n == 1 else 's'} in "
                f"this home's database {'is' if n == 1 else 'are'} another library's (under none of "
                f"this config's library folders), left out; the first scan reads this one")
    return (f"{n:,} sample{'' if n == 1 else 's'} from other libraries left out (in this home's "
            f"database but under none of this config's library folders)")


def usable_count(session) -> int:
    """Samples with a CLAP embedding (the ones a build can place), less the ones a walk
    marked missing and those of other libraries (outside_library): what the library scale
    counts (packs/scale.py)."""
    if session is None:
        return 0
    n = session.execute(text("SELECT COUNT(*) FROM sample_features "
                             "WHERE clap_embedding IS NOT NULL")).scalar() or 0
    other = outside_library(session) if n else ()
    if other:
        n -= session.execute(text(
            f"SELECT COUNT(*) FROM sample_features WHERE clap_embedding IS NOT NULL "
            f"AND sample_id IN ({_id_list(other)})")).scalar() or 0
    if n and missing_marked(session):
        n -= session.execute(text(
            "SELECT COUNT(*) FROM sample_features f JOIN missing_files m ON m.sample_id = f.sample_id "
            "JOIN samples s ON s.id = m.sample_id AND s.path = m.path "
            "WHERE f.clap_embedding IS NOT NULL"
            + (f" AND f.sample_id NOT IN ({_id_list(other)})" if other else ""))).scalar() or 0
    return n


def missing_marked(session) -> bool:
    """Whether a library walk marked any sample missing (ingest/importer.py: a file it didn't
    find, or couldn't read). Only then does sample_select leave those samples out: a database
    no walk marked (Sononym's, without `tools scan --walk`) is queried exactly as before."""
    try:
        return session.execute(text("SELECT 1 FROM missing_files LIMIT 1")).first() is not None
    except Exception:            # a database from before the table (read as it is)
        return False


_ROWTYPES: dict = {}


def fetch(session, query: Select) -> list:
    """Run a sample_select() query: rows as named tuples, label fields filled in."""
    ensure_current(session)
    result = session.execute(query)
    keys = list(result.keys())
    raw = result.all()
    want = [k for k in keys if k != _SID and not k.startswith(_HAS)]
    rowtype = _ROWTYPES.setdefault(tuple(want), namedtuple("SampleRow", want))
    lab = [k for k in want if k in LABEL_FIELDS]
    if not lab:
        return [rowtype(*r) for r in raw]
    act = active(session)
    pk = set()
    for k in lab:
        prov, kind = LABEL_FIELDS[k]
        if prov != CLASSIFIER:
            pk.add((prov, kind))
        elif not act.fallback:
            pk.add((SONONYM, kind))
        else:                                 # fallback: everything comes from canonical labels
            pk |= {(c, CANONICAL_KIND) for c in act.classifiers}
    sid_at = keys.index("id") if "id" in keys else keys.index(_SID)
    got = _labels(session, [r[sid_at] for r in raw], sorted(pk))
    out = []
    for r in raw:
        d = dict(zip(keys, r))
        mine = got.get(r[sid_at], {})
        canon = _fallback_canonical(mine, act.classifiers) if act.fallback else None
        for k in lab:
            prov, kind = LABEL_FIELDS[k]
            if prov == CLASSIFIER and act.fallback:
                d[k] = (canon if kind == CANONICAL_KIND
                        else [_SHAPE[c] for c in canon if c in _SHAPE] if kind == "class" else [])  # type: ignore[union-attr]
            elif prov == CLASSIFIER:
                d[k] = list(mine.get((SONONYM, kind), ()))
            elif prov not in act.names:
                d[k] = None                           # a provider this build doesn't use
            elif prov in _OPTIONAL and d.get(_HAS + prov) is None:
                d[k] = None                           # the provider never saw this sample
            else:
                d[k] = list(mine.get((prov, kind), ()))
        out.append(rowtype(*(d[k] for k in want)))
    return out


def _fallback_canonical(mine, classifiers) -> list:
    """The fallback classifiers' labels merged: every category label any of them gives, and
    the shape (one-shot / loop) from the first that has one."""
    labels, shape = set(), None
    for prov in classifiers:
        got = mine.get((prov, CANONICAL_KIND), ())
        labels |= {x for x in got if x not in _SHAPE}
        if shape is None:
            shape = [x for x in got if x in _SHAPE] or None
    return sorted(labels | set(shape or ()))


def _labels(session, ids, pk) -> dict:
    """{sample_id: {(provider, kind): [label, ...in rank order]}} for these samples."""
    got = defaultdict(lambda: defaultdict(list))
    cond = " OR ".join(f"(provider = '{p}' AND kind = '{k}')" for p, k in sorted(set(pk)))
    base = f"SELECT sample_id, provider, kind, label FROM labels WHERE ({cond})"
    ids = list(dict.fromkeys(ids))
    if len(ids) > 20000:                       # most of the library: one pass
        rows = session.execute(text(base + " ORDER BY sample_id, provider, kind, rank, label"))
        want = set(ids)
        for sid, prov, kind, lab in rows:
            if sid in want:
                got[sid][(prov, kind)].append(lab)
        return got
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        rows = session.execute(text(
            base + f" AND sample_id IN ({', '.join(str(int(s)) for s in chunk)})"
                   " ORDER BY sample_id, provider, kind, rank, label"))
        for sid, prov, kind, lab in rows:
            got[sid][(prov, kind)].append(lab)
    return got


# Filters on the list-valued fields, as text for q.where(text(sql).bindparams(**params)).
# Each returns (sql, params); prefix keeps the bind names of several filters apart.

def like_any(field: str, values, prefix: str, quoted: bool = False, session=None) -> tuple[str, dict]:
    """The sample has a value of this list field matching any of values: a substring
    match ("%Kicks%"), or with quoted=True a whole value ("Kick", not "Kick Bass"). Both
    ignore ASCII case, as LIKE on the JSON text did. With a session it follows the
    providers in use (a provider the build doesn't use matches nothing)."""
    act = _active(session)
    values = list(values)
    prov, kind = LABEL_FIELDS[field]
    provs = [prov]
    if prov == CLASSIFIER:
        provs = [SONONYM] if not act.fallback else list(act.classifiers)
        if act.fallback:                     # the fallback has only canonical labels
            if kind == "category":
                return "1 = 0", {}
            if kind == "class":
                back = {v: k for k, v in _SHAPE.items()}
                values = [back.get(v, v) for v in values]
            kind = CANONICAL_KIND
    elif prov not in act.names:
        return "1 = 0", {}
    params = {f"{prefix}{i}": (v if quoted else f"%{v}%") for i, v in enumerate(values)}
    match = " OR ".join(f"l.label LIKE :{k}" for k in params)
    ins = ", ".join(f"'{p}'" for p in provs)
    sql = (f"EXISTS (SELECT 1 FROM labels l WHERE l.sample_id = samples.id "
           f"AND l.provider IN ({ins}) AND l.kind = '{kind}' AND ({match}))")
    return sql, params


# ---------------------------------------------------------------------------
# Keeping the generic tables current
# ---------------------------------------------------------------------------

_CURRENT: set = set()


def ensure_current(session, log=lambda m: None) -> bool:
    """Rebuild a provider's labels and descriptors if they don't cover its source (sample
    counts differ). Returns True when it rebuilt. Checked once per database per process."""
    key = str(session.get_bind().url)
    if key in _CURRENT:
        return False
    import multiprocessing as mp
    act = active(session)
    rebuilt = False
    todo = [(p, covered_count, rebuild) for p in SOURCES if p in act.names]
    if act.fallback:
        todo += [(p, shadow.covered_count, shadow.rebuild) for p in act.classifiers]
    from ..db.session import is_read_only
    for prov, count, build in todo:
        need, have = count(session, prov)
        if have == need:
            continue
        if is_read_only():           # build --dry-run: estimates from the labels as they are
            log(f"metadata: the {prov} labels cover {have:,} of {need:,} samples; the next build "
                f"rebuilds them")
            continue
        if mp.parent_process() is not None:
            raise RuntimeError(
                f"the {prov} labels/descriptors cover {have:,} samples but its source has "
                f"{need:,}; run `fourier tools db-stats --metadata --rebuild`")
        log(f"metadata: updating the {prov} labels ({need - have:,} of {need:,} samples new or "
            f"changed)" if have < need else
            f"metadata: updating the {prov} labels ({have - need:,} samples gone)")
        build(session, prov)
        session.commit()
        rebuilt = True
    _CURRENT.add(key)
    return rebuilt


def forget_current() -> None:
    """Tests: check again on the next ensure_current(), and work out the providers again."""
    from .providers import forget
    _CURRENT.clear()
    forget()
