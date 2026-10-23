"""The built-in providers, path and audio: they label samples from what any library has
(folder and file names, Fourier's own audio measurements), without Sononym or Live. When
the build uses Sononym (metadata/providers.py) their labels are only recorded and scored
against its own (the agreement report) and route nothing; without Sononym they are the
classifier, the fallback that lets a stranger's library be curated.

  path    canonical labels from the library-relative path (config/providers/path.yaml), and
          a tempo written in the name ("174bpm") as the descriptor "bpm". One-shot / loop
          words count only in the file's name and its folder (shape_labels), never in a
          vendor's or pack's name ("Breaks and Hits/Kicks/Kick 01")
  audio   the shape of a sample from Fourier's measurements (sample_features): a loop when
          it's long with several events, or long with several onsets (a groove with no
          silence between its hits) unless it is sustained (mostly harmonic, slow onsets: a
          pad); a one-shot when it's a single event (however long).
          A loop guess yields to the file's own name or folder when they name an
          instrument or FX sound and no loop (audio_label: "FX/Riser 04" with onsets is a
          riser); descriptors "duration_s" and "harmonicity" (the harmonic share of HPSS
          energy, 0 to 1, as Sononym's is). No "bpm": Fourier's own tempo is
          sample_features.tempo_bpm, and its reliability Fourier's own (metadata/resolve.py),
          never Sononym's flag on Sononym's tempo

Each writes the generic tables (labels kind "canonical", descriptors) like the other
providers, with the "analysed" marker on every sample it looked at.
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

import yaml
from sqlalchemy import insert, text

from ..db.models import Descriptor, Label
from .store import ANALYSED, CANONICAL_KIND, SONONYM
from .vocab import CANONICAL, PROVIDERS_DIR

PATH, AUDIO = "path", "audio"
SHADOW = (PATH, AUDIO)
CLASSES = ("class.oneshot", "class.loop")
_BPM = re.compile(r"(?<!\d)(\d{2,3}) ?bpm(?![a-z])", re.I)
_CHUNK = 20000

# the audio provider's shape rules: events are counted between silences, onsets by librosa
# (sample_features.onset_rate_hz); a file longer than ONSET_LOOP_MIN_S with at least
# LOOP_MIN_EVENTS onsets, at ONSET_LOOP_MIN_HZ or more, is a loop even with no silent gap
LOOP_MIN_S, LOOP_MIN_EVENTS, ONESHOT_MAX_S = 1.0, 4, 1.5
ONSET_LOOP_MIN_S, ONSET_LOOP_MIN_HZ = 1.5, 1.0
# ...unless it is sustained: mostly harmonic (HPSS, SUSTAINED_HPR or more), its onsets slower
# than SUSTAINED_MAX_HZ and no silence between them (a pad that changes chords, a swell): a
# one-shot, however long, as a single long event is
SUSTAINED_HPR, SUSTAINED_MAX_HZ = 0.8, 2.0

# A folder that says nothing about its files (a format, "Samples"): the one above it speaks
# for them when a file's shape is read (shape_scope)
_GENERIC_WORDS = frozenset({"wav", "wavs", "wave", "aif", "aiff", "flac", "mp3", "ogg", "audio",
                            "sample", "samples", "file", "files", "sound", "sounds", "bit", "khz",
                            "k", "format", "formats"})
# the drums a file's own name can name (path.yaml's labels): that name beats a folder's
# loop word ("Breaks/Kick 01" is a kick)
DRUM_HIT_LABELS = frozenset({"kick", "snare", "clap", "hat"})
# the instrument and FX sounds a file's own name or folder can name (path.yaml's labels, and
# every fx.*): that name beats a loop guess from the audio alone (audio_label)
SOUND_LABELS = frozenset({"bass", "lead", "keys", "pad", "stab", "bell", "blip", "zap", "scratch", "vocal"})


STRONG, WEAK, NEAR = "strong", "weak", "near"
_SECTIONS = {"weak": WEAK, "near_only": NEAR}


def path_words() -> tuple:
    """The `words` knob's words (curate_config.PATH_WORDS: {CATEGORY: [word, ...]}) as
    ((canonical label, (word, ...)), ...): each category's first label, in a stable order."""
    from ..packs.curate_config import CATEGORIES, PATH_WORDS
    out = []
    for cat, words in sorted((PATH_WORDS or {}).items()):
        labels = (CATEGORIES.get(cat) or {}).get("labels") or ()
        if labels and words:
            out.append((labels[0], tuple(sorted({str(w).strip().lower() for w in words if str(w).strip()}))))
    return tuple(out)


def _word_pattern(word: str) -> str:
    """A `words` knob entry as a whole-word pattern ("kck" also matches "kcks"); a word of
    several parts matches with any separator between them."""
    parts = [re.escape(p) for p in re.split(r"[\s_\-.]+", word) if p]
    return r"\b" + r" ?".join(parts) + r"s?\b"


def path_rules() -> tuple:
    """((canonical label, compiled regex, kind), ...) from config/providers/path.yaml and
    the `words` knob (path_words, strong): kind "strong" (the label's own list), "weak"
    (under `weak:`, a word that describes as often as it names: it gives its label only when
    no strong word in the same place gives one) or "near" (under `near_only:`, a weak word
    that counts in the file's name and folder but not in the folders above, where it's
    usually a pack name)."""
    return _path_rules(path_words())


@lru_cache(maxsize=None)
def _path_rules(words: tuple) -> tuple:
    doc = yaml.safe_load((PROVIDERS_DIR / "path.yaml").read_text()) or {}
    out = []

    def add(label, pats, kind):
        if label not in CANONICAL:
            raise ValueError(f"path.yaml: {label!r} is not a canonical label")
        out.append((label, re.compile("|".join(f"(?:{p})" for p in pats), re.I), kind))

    for key, val in doc.items():
        if key in _SECTIONS:
            for label, pats in (val or {}).items():
                add(label, pats, _SECTIONS[key])
        elif key != "not_words":
            add(key, val, STRONG)
    for label, ws in words:
        add(label, [_word_pattern(w) for w in ws], STRONG)
    return tuple(out)


@lru_cache(maxsize=None)
def _not_words():
    """Case-sensitive words dropped before matching (path.yaml `not_words`): a name that
    isn't the thing it spells, like an all-caps TOM, a drum machine."""
    doc = yaml.safe_load((PROVIDERS_DIR / "path.yaml").read_text()) or {}
    pats = doc.get("not_words") or []
    return re.compile("|".join(f"(?:{p})" for p in pats)) if pats else None


# a name's words: separators, then letters from digits ("kick01", "BD01", "808Kick"; a
# plural "s" after a number stays on it: "808s"), then camelCase ("HiHat" -> "Hi Hat")
_SEPARATORS = re.compile(r"[/_\-.()\[\]]+")
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])(?![sS](?![A-Za-z]))")
_CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z])")


def words_of(text: str) -> str:
    """The text the path rules match: separators as spaces, letters split from digits and
    camelCase split, after the case-sensitive not_words are dropped."""
    words = _LETTER_DIGIT.sub(" ", _SEPARATORS.sub(" ", text or ""))
    nw = _not_words()
    words = nw.sub(" ", words) if nw else words
    return _CAMEL.sub(" ", words)


def _matches(text: str, near: bool) -> set:
    words = words_of(text)
    got = {kind: set() for kind in (STRONG, WEAK, NEAR)}
    for label, rx, kind in path_rules():
        if (near or kind != NEAR) and rx.search(words):
            got[kind].add(label)
    strong, weak = got[STRONG], got[WEAK] | got[NEAR]
    shape = {x for x in strong | weak if x in CLASSES}
    return shape | ((strong - shape) or (weak - shape))


def path_labels(rel_path: str) -> set:
    """Canonical labels a library-relative path suggests. The file's own name and folder
    speak first: the folders above them (pack and vendor names, "Synth Drums") give a
    category only when those two give none. One-shot / loop come from the file's name and
    its folder only (shape_labels)."""
    parts = (rel_path or "").split("/")
    near, far = _matches("/".join(parts[-2:]), True), _matches("/".join(parts[:-2]), False)
    return shape_labels(rel_path) | ((near - set(CLASSES)) or (far - set(CLASSES)))


def _generic_folder(name: str) -> bool:
    """A folder named for a format or as "Samples" ("WAV", "24-bit WAV Files", "Samples")."""
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", name.lower())
    return bool(words) and any(w in _GENERIC_WORDS for w in words) and all(
        w in _GENERIC_WORDS or w[0].isdigit() for w in words)


def shape_scope(rel_path: str) -> tuple[str, str]:
    """(the file's name, its folder): the text a file's shape and Live-tag words are read
    from. A generic folder ("WAV", "Samples") hands over to the one above it; the folders
    above that are vendor and pack names, which describe a whole pack, not this file."""
    parts = [p for p in (rel_path or "").split("/") if p]
    if not parts:
        return "", ""
    name, folders = parts[-1], parts[:-1]
    while folders and _generic_folder(folders[-1]) and len(folders) > 1:
        folders = folders[:-1]
    return name, (folders[-1] if folders else "")


def drum_hit_named(filename: str) -> bool:
    """The file's own name names a drum hit (a kick, snare, clap or hat)."""
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    return bool(_matches(stem, True) & DRUM_HIT_LABELS)


def shape_labels(rel_path: str) -> set:
    """One-shot / loop from the file's name, else its folder (shape_scope); a file whose
    own name names a drum hit takes no loop from its folder ("Breaks/Kick 01": the audio
    provider says)."""
    name, folder = shape_scope(rel_path)
    own = _matches(os.path.splitext(name)[0], True) & set(CLASSES)
    if own:
        return own
    got = _matches(folder, True) & set(CLASSES) if folder else set()
    return got - {"class.loop"} if drum_hit_named(name) else got


def near_labels(rel_path: str) -> set:
    """The category labels (no shape) the file's own name and folder give (shape_scope)."""
    name, folder = shape_scope(rel_path)
    return _matches(f"{os.path.splitext(name)[0]} {folder}", True) - set(CLASSES)


def names_a_sound(rel_path: str) -> bool:
    """The file's own name or folder names an instrument or FX sound (SOUND_LABELS, fx.*)."""
    return any(lab in SOUND_LABELS or lab.startswith("fx.") for lab in near_labels(rel_path))


def path_bpm(rel_path: str) -> float | None:
    m = _BPM.search(rel_path or "")
    bpm = float(m.group(1)) if m else None
    return bpm if bpm and 50 <= bpm <= 250 else None


def audio_class(duration_s, n_events, onset_rate_hz=None, harmonicity=None) -> str | None:
    """A loop: long, with several events, or (events are counted between silences, and a
    groove may have none) longer than ONSET_LOOP_MIN_S with several onsets at a steady
    rate, unless it is sustained (harmonicity, the harmonic share of its energy, at least
    SUSTAINED_HPR and onsets slower than SUSTAINED_MAX_HZ: a pad changing chords). A
    one-shot: a single event however long (a pad, a crash), or short when the events weren't
    counted. Anything else: no call."""
    if duration_s is None:
        return None
    if n_events is None:
        return "class.oneshot" if duration_s < ONESHOT_MAX_S else None
    if duration_s >= LOOP_MIN_S and n_events >= LOOP_MIN_EVENTS:
        return "class.loop"
    sustained = (harmonicity is not None and harmonicity >= SUSTAINED_HPR
                 and onset_rate_hz is not None and onset_rate_hz < SUSTAINED_MAX_HZ)
    if onset_rate_hz is not None and duration_s > ONSET_LOOP_MIN_S and not sustained \
            and onset_rate_hz >= ONSET_LOOP_MIN_HZ and onset_rate_hz * duration_s >= LOOP_MIN_EVENTS:
        return "class.loop"
    if n_events <= 1:
        return "class.oneshot"
    return None


def audio_label(rel_path: str, duration_s, n_events, onset_rate_hz=None, harmonicity=None) -> str | None:
    """The audio provider's shape: audio_class, except that a loop guess from the audio alone
    yields to the path: a file whose own name or folder names an instrument or FX sound
    (names_a_sound) and has no loop word (shape_labels) is that sound, a one-shot. A riser,
    a pad or a bassline has onsets and no silent gap, as a groove has. A loop cut to a tempo
    its path states (on_path_grid: "Pack 125bpm/Bass/Lines/Riff A.wav") stays a loop."""
    cls = audio_class(duration_s, n_events, onset_rate_hz, harmonicity)
    if cls == "class.loop" and not shape_labels(rel_path) and names_a_sound(rel_path) \
            and not on_path_grid(rel_path, duration_s):
        return "class.oneshot"
    return cls


GRID_TOL_S = 0.010       # how far off a whole beat a file cut to its tempo's grid may end
GRID_MIN_BEATS = 4


_SHOTS_DIR = re.compile(r"(?<![a-z])shots?(?![a-z])", re.I)
# a file's own name that calls it one sound ("Synth Stab 03", "Brass Hit 2", "Pad Chord C",
# "Bass Note 4"): a held note bounced at whole bars stays a one-shot, whatever tremolo or a
# filter sweep made the audio hear. Only the singular word counts: a name with a plural
# ("Chord Stabs Am", "House Hits", "Notes", "Chords") names a phrase of them, and a loop
# word in the name ("Stab Loop") doesn't count either.
_ONESHOT_NAME = re.compile(r"(?<![a-z])(?:stab|hit|shot|one ?shot|note|chord)(?![a-z])", re.I)
_ONESHOTS_NAME = re.compile(r"(?<![a-z])(?:stabs|hits|shots|notes|chords)(?![a-z])", re.I)


def oneshot_named(rel_path: str) -> bool:
    """The file's own name calls it one sound (_ONESHOT_NAME) and has no plural of one
    (_ONESHOTS_NAME) and no loop or phrase word."""
    from ..packs.curate_config import PHRASE_LOOP_DIR_RE
    stem = re.sub(r"[_-]+", " ", os.path.splitext(os.path.basename(rel_path or ""))[0])
    stem = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    return (bool(_ONESHOT_NAME.search(stem)) and not _ONESHOTS_NAME.search(stem)
            and not PHRASE_LOOP_DIR_RE.search(stem))


def oneshot_folder(rel_path: str) -> bool:
    """A folder of the path names single sounds ("One Shots", "Hits", "Shots", "Chords",
    curate_config.PHRASE_ONESHOT_DIR_RE) and no loops ("Chord Loops" names loops)."""
    from ..packs.curate_config import PHRASE_LOOP_DIR_RE, PHRASE_ONESHOT_DIR_RE
    return any((PHRASE_ONESHOT_DIR_RE.search(f) or _SHOTS_DIR.search(f)) and not PHRASE_LOOP_DIR_RE.search(f)
               for f in (rel_path or "").split("/")[:-1])


def on_path_grid(rel_path: str, duration_s) -> bool:
    """The file lasts a whole number of beats, GRID_MIN_BEATS or more (within GRID_TOL_S), at
    a tempo its path states ("Pack 125bpm/Synth Q1/Riff A-01.wav", 1.920 s), its name states
    no note and names no drum hit, and no folder or word of its path calls it a one-shot
    (oneshot_folder, shape_labels, oneshot_named): a loop cut to its pack's grid. A note
    bounced at exactly a bar names its note ("Bass E3"), calls itself one sound ("Synth Stab
    03") or sits in a one-shot folder ("Pack 124bpm/One Shots/Synths"), so it stays a note."""
    bpm = path_bpm(rel_path)
    if not bpm or not duration_s:
        return False
    if oneshot_folder(rel_path) or "class.oneshot" in shape_labels(rel_path) or drum_hit_named(rel_path) \
            or oneshot_named(rel_path):
        return False
    beats = duration_s * bpm / 60.0
    if round(beats) < GRID_MIN_BEATS or abs(beats - round(beats)) * 60.0 / bpm > GRID_TOL_S:
        return False
    from ..packs.curate import _note_hits          # (the names' note reader)
    stem = os.path.splitext(os.path.basename(rel_path))[0].replace("_", " ")
    return not _note_hits(stem)


# Bump when a built-in provider's rules change in code (path.yaml's text and the audio
# constants are in the versions too), so the labels are rebuilt.
_PATH_RULES = 3
_AUDIO_RULES = 9         # 7: no "bpm" descriptor (it was librosa's tempo where Sononym's was reliable);
                         # 8: a loop on its path's tempo grid stays a loop (on_path_grid);
                         # 9: ...unless its name calls it one sound (oneshot_named)


def rules_version(provider: str) -> float:
    """A number for the rules a built-in provider labels with: path.yaml's text for path,
    the shape rules for audio. The "analysed" marker carries it, so a change of rules
    rebuilds the labels (ensure_current), not only a change in the sample count."""
    import hashlib
    from ..packs.curate_config import PHRASE_LOOP_DIR_RE, PHRASE_ONESHOT_DIR_RE
    # the settable rules the shape labels read (a user's [advanced] change relabels)
    shape = tuple((rx.pattern, rx.flags) for rx in (PHRASE_LOOP_DIR_RE, PHRASE_ONESHOT_DIR_RE))
    if provider == PATH:
        src = (PROVIDERS_DIR / "path.yaml").read_bytes() + repr(
            (_PATH_RULES, sorted(_GENERIC_WORDS), sorted(DRUM_HIT_LABELS), path_words(), shape)).encode()
    elif provider == AUDIO:
        src = repr((_AUDIO_RULES, LOOP_MIN_S, LOOP_MIN_EVENTS, ONESHOT_MAX_S, ONSET_LOOP_MIN_S,
                    ONSET_LOOP_MIN_HZ, SUSTAINED_HPR, SUSTAINED_MAX_HZ, sorted(SOUND_LABELS),
                    path_words(), shape, GRID_TOL_S, GRID_MIN_BEATS)).encode() + (
            PROVIDERS_DIR / "path.yaml").read_bytes()
    else:
        raise KeyError(provider)
    # six hex digits: under 2**24, exact in a 32-bit float (DuckDB's FLOAT column, where seven
    # were rounded, so the marker never matched and every build relabelled every sample)
    return float(int(hashlib.sha1(src).hexdigest()[:6], 16))


def _sql_value(v) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"CAST('{v!r}' AS DOUBLE)"          # the same double a bound parameter is
    return "'" + str(v).replace("'", "''") + "'"


def _insert(session, model, rows: list) -> None:
    """rows into model's table. On DuckDB as multi-row VALUES statements: its driver binds an
    executemany one row at a time, many times slower for the tens of thousands of rows a
    first labelling writes."""
    if session.get_bind().dialect.name != "duckdb":
        session.execute(insert(model), rows)
        return
    cols = list(rows[0])
    conn = session.connection()
    for k in range(0, len(rows), 2000):
        values = ",".join("(" + ",".join(_sql_value(r[c]) for c in cols) + ")" for r in rows[k:k + 2000])
        conn.exec_driver_sql(f"INSERT INTO {model.__tablename__} ({', '.join(cols)}) VALUES {values}")


def _flush(session, labels, descs):
    if labels:
        _insert(session, Label, labels)
        labels.clear()
    if descs:
        _insert(session, Descriptor, descs)
        descs.clear()


# Each labelled sample's "inputs" descriptor stamps the rules version and what its labels
# were read from (stamp): a sample is labelled again when its stamp changes (a new file, a
# file analyzed again, a path filled in, new rules), the others keep theirs.
INPUTS = "inputs"


def _source_sql(provider: str) -> str:
    if provider == PATH:
        return "SELECT id, rel_path, path FROM samples"
    if provider == AUDIO:
        return ("SELECT s.id, s.duration_s, f.n_events, f.harmonic_percussive_ratio, f.onset_rate_hz, "
                "s.rel_path, s.path FROM samples s JOIN sample_features f ON f.sample_id = s.id")
    raise KeyError(provider)


def _label(provider: str, row) -> tuple[set, dict]:
    """(labels, descriptor values) for one source row."""
    if provider == PATH:
        _sid, rel, path = row
        rel = rel or path or ""
        return path_labels(rel), {"bpm": path_bpm(rel)}
    _sid, dur, n_ev, hpr, onsets, rel, path = row
    cls = audio_label(rel or path or "", dur, n_ev, onsets, hpr)
    return ({cls} if cls else set()), {"duration_s": dur, "harmonicity": hpr}


def stamp(version: float, row) -> float:
    """A number for the rules version and a source row's values (all but its id): six hex
    digits, exact in a 32-bit float as rules_version is."""
    import hashlib
    return float(int(hashlib.blake2b(repr((version, tuple(row[1:]))).encode(),
                                      digest_size=3).hexdigest(), 16))


def _add_rows(session, provider: str, rows, version: float) -> None:
    """Write the labels, the "analysed" marker, the stamp and the descriptors of these source
    rows, in chunks."""
    labels: list = []
    descs: list = []
    for row in rows:
        sid = row[0]
        labs, values = _label(provider, row)
        descs.append(dict(sample_id=sid, provider=provider, name=ANALYSED, value=version))
        descs.append(dict(sample_id=sid, provider=provider, name=INPUTS, value=stamp(version, row)))
        for i, lab in enumerate(sorted(labs)):
            labels.append(dict(sample_id=sid, provider=provider, kind=CANONICAL_KIND, label=lab,
                               rank=i, confidence=None))
        for name, v in values.items():
            if v is not None:
                descs.append(dict(sample_id=sid, provider=provider, name=name, value=float(v)))
        if len(descs) >= _CHUNK:
            _flush(session, labels, descs)
    _flush(session, labels, descs)


def _counts(session, provider: str, log) -> dict:
    p = {"p": provider}
    n_lab = session.execute(text("SELECT count(*) FROM labels WHERE provider = :p"), p).scalar()
    n_des = session.execute(text("SELECT count(*) FROM descriptors WHERE provider = :p"), p).scalar()
    log(f"{provider} metadata: {n_lab:,} labels, {n_des:,} descriptors")
    return {"labels": n_lab, "descriptors": n_des}


def rebuild(session, provider: str, log=lambda m: None) -> dict:
    """Replace one built-in provider's labels and descriptors."""
    p = {"p": provider}
    # Each SELECT is read whole before the writes: on DuckDB a write ends a result that is
    # still being read.
    rows = session.execute(text(_source_sql(provider))).all()
    session.execute(text("DELETE FROM labels WHERE provider = :p"), p)
    session.execute(text("DELETE FROM descriptors WHERE provider = :p"), p)
    _add_rows(session, provider, rows, rules_version(provider))
    return _counts(session, provider, log)


class Plan:
    """What update() does for a provider: the source rows to label (new or changed: no stamp,
    or another), and the samples labelled before that its source no longer has."""

    def __init__(self, provider: str, need: int, todo: list, gone: list):
        self.provider, self.need, self.todo, self.gone = provider, need, todo, gone

    def __bool__(self) -> bool:
        return bool(self.todo or self.gone)

    def said(self) -> str:
        """The build's line for it."""
        parts = []
        if self.todo:
            parts.append(f"{len(self.todo):,} of {self.need:,} samples new or changed")
        if self.gone:
            parts.append(f"{len(self.gone):,} sample{'s' if len(self.gone) != 1 else ''} gone")
        return f"metadata: updating the {self.provider} labels ({'; '.join(parts)})"


def plan(session, provider: str) -> Plan:
    """The samples whose labels are missing or out of date (stamp), and those gone."""
    version = rules_version(provider)
    rows = session.execute(text(_source_sql(provider))).all()
    p = {"p": provider}
    stamps = {int(i): v for i, v in session.execute(text(
        "SELECT sample_id, value FROM descriptors WHERE provider = :p AND name = :n"),
        {**p, "n": INPUTS})}
    marked = {int(i) for (i,) in session.execute(text(
        "SELECT sample_id FROM descriptors WHERE provider = :p AND name = :n AND value = :v"),
        {**p, "n": ANALYSED, "v": version})}
    todo = [r for r in rows if int(r[0]) not in marked or stamps.get(int(r[0])) != stamp(version, r)]
    have = {int(r[0]) for r in rows}
    held = {int(i) for (i,) in session.execute(text(
        "SELECT DISTINCT sample_id FROM descriptors WHERE provider = :p "
        "UNION SELECT DISTINCT sample_id FROM labels WHERE provider = :p"), p)}
    return Plan(provider, len(rows), todo, sorted(held - have))


def update(session, provider: str, the_plan: Plan | None = None, log=lambda m: None) -> dict:
    """Label the samples plan() finds new or changed, and drop the rows of those gone; the
    rest keep theirs. Labelling every sample is a rebuild()."""
    pl = the_plan if the_plan is not None else plan(session, provider)
    if len(pl.todo) == pl.need:
        p = {"p": provider}
        session.execute(text("DELETE FROM labels WHERE provider = :p"), p)
        session.execute(text("DELETE FROM descriptors WHERE provider = :p"), p)
    else:
        ids = sorted({int(r[0]) for r in pl.todo} | set(pl.gone))
        for k in range(0, len(ids), 1000):
            chunk = ",".join(str(i) for i in ids[k:k + 1000])
            for table in ("labels", "descriptors"):
                session.execute(text(f"DELETE FROM {table} WHERE provider = :p AND sample_id IN ({chunk})"),
                                {"p": provider})
    _add_rows(session, provider, pl.todo, rules_version(provider))
    return _counts(session, provider, log)


def covered_count(session, provider: str) -> tuple[int, int]:
    """(samples the provider should cover, samples it has marked with the current rules)."""
    need_sql = {PATH: "SELECT count(*) FROM samples",
                AUDIO: "SELECT count(*) FROM samples s JOIN sample_features f ON f.sample_id = s.id"}
    need = session.execute(text(need_sql[provider])).scalar()
    have = session.execute(text("SELECT count(*) FROM descriptors WHERE provider = :p AND name = :n "
                                "AND value = :v"),
                           {"p": provider, "n": ANALYSED, "v": rules_version(provider)}).scalar()
    return need, have


def ensure_current(session, log=lambda m: None) -> list[str]:
    """Label again the samples of the built-in providers that are new or changed, or whose
    rules changed (plan). Returns the providers it updated."""
    out = []
    for prov in SHADOW:
        pl = plan(session, prov)
        if pl:
            log(pl.said())
            update(session, prov, pl, log=log)
            session.commit()
            out.append(prov)
    return out


def agreement(session, provider: str, reference: str = SONONYM, top: int = 8) -> dict:
    """How a shadow provider's canonical labels compare with the reference provider's, over
    the samples both label: class agreement (one-shot vs loop), label agreement (at least one
    non-class label in common) and the most common disagreements."""
    got = defaultdict(lambda: defaultdict(set))
    for sid, prov, lab in session.execute(text(
            "SELECT sample_id, provider, label FROM labels WHERE kind = :k AND provider IN (:a, :b)"),
            {"k": CANONICAL_KIND, "a": provider, "b": reference}):
        got[sid][prov].add(lab)
    n_ref = n_cov = cls_both = cls_same = lab_both = lab_same = 0
    confused = Counter()
    for sid, d in got.items():
        ref, mine = d.get(reference, set()), d.get(provider, set())
        if not ref:
            continue
        n_ref += 1
        if mine:
            n_cov += 1
        rc, mc = ref & set(CLASSES), mine & set(CLASSES)
        if rc and mc:
            cls_both += 1
            cls_same += bool(rc & mc)
        rl, ml = ref - set(CLASSES), mine - set(CLASSES)
        if rl and ml:
            lab_both += 1
            if rl & ml:
                lab_same += 1
            else:
                confused[(",".join(sorted(rl)), ",".join(sorted(ml)))] += 1
    return {"provider": provider, "reference": reference, "samples": n_ref, "labelled": n_cov,
            "class_compared": cls_both, "class_agree": cls_same,
            "label_compared": lab_both, "label_agree": lab_same,
            "confused": confused.most_common(top)}


def format_agreement(a: dict) -> str:
    def pct(x, n):
        return f"{100 * x / n:.1f}%" if n else "-"
    return (f"{a['provider']} vs {a['reference']}: labels {a['labelled']:,} of {a['samples']:,} "
            f"({pct(a['labelled'], a['samples'])}); class agrees {pct(a['class_agree'], a['class_compared'])} "
            f"of {a['class_compared']:,}; label agrees {pct(a['label_agree'], a['label_compared'])} "
            f"of {a['label_compared']:,}")
