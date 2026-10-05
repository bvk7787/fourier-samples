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
    a pad or a bassline has onsets and no silent gap, as a groove has."""
    cls = audio_class(duration_s, n_events, onset_rate_hz, harmonicity)
    if cls == "class.loop" and not shape_labels(rel_path) and names_a_sound(rel_path):
        return "class.oneshot"
    return cls


# Bump when a built-in provider's rules change in code (path.yaml's text and the audio
# constants are in the versions too), so the labels are rebuilt.
_PATH_RULES = 3
_AUDIO_RULES = 7         # 7: no "bpm" descriptor (it was librosa's tempo where Sononym's was reliable)


def rules_version(provider: str) -> float:
    """A number for the rules a built-in provider labels with: path.yaml's text for path,
    the shape rules for audio. The "analysed" marker carries it, so a change of rules
    rebuilds the labels (ensure_current), not only a change in the sample count."""
    import hashlib
    if provider == PATH:
        src = (PROVIDERS_DIR / "path.yaml").read_bytes() + repr(
            (_PATH_RULES, sorted(_GENERIC_WORDS), sorted(DRUM_HIT_LABELS), path_words())).encode()
    elif provider == AUDIO:
        src = repr((_AUDIO_RULES, LOOP_MIN_S, LOOP_MIN_EVENTS, ONESHOT_MAX_S, ONSET_LOOP_MIN_S,
                    ONSET_LOOP_MIN_HZ, SUSTAINED_HPR, SUSTAINED_MAX_HZ, sorted(SOUND_LABELS),
                    path_words())).encode() + (
            PROVIDERS_DIR / "path.yaml").read_bytes()
    else:
        raise KeyError(provider)
    # six hex digits: under 2**24, exact in a 32-bit float (DuckDB's FLOAT column, where seven
    # were rounded, so the marker never matched and every build relabelled every sample)
    return float(int(hashlib.sha1(src).hexdigest()[:6], 16))


def _flush(session, labels, descs):
    if labels:
        session.execute(insert(Label), labels)
        labels.clear()
    if descs:
        session.execute(insert(Descriptor), descs)
        descs.clear()


def rebuild(session, provider: str, log=lambda m: None) -> dict:
    """Replace one built-in provider's labels and descriptors."""
    p = {"p": provider}
    session.execute(text("DELETE FROM labels WHERE provider = :p"), p)
    session.execute(text("DELETE FROM descriptors WHERE provider = :p"), p)
    labels, descs = [], []
    version = rules_version(provider)

    def add(sid, labs, **values):
        descs.append(dict(sample_id=sid, provider=provider, name=ANALYSED, value=version))
        for i, lab in enumerate(sorted(labs)):
            labels.append(dict(sample_id=sid, provider=provider, kind=CANONICAL_KIND, label=lab,
                               rank=i, confidence=None))
        for name, v in values.items():
            if v is not None:
                descs.append(dict(sample_id=sid, provider=provider, name=name, value=float(v)))
        if len(descs) >= _CHUNK:
            _flush(session, labels, descs)

    if provider == PATH:
        for sid, rel, path in session.execute(text("SELECT id, rel_path, path FROM samples")):
            rel = rel or path or ""
            add(sid, path_labels(rel), bpm=path_bpm(rel))
    elif provider == AUDIO:
        for sid, dur, n_ev, hpr, onsets, rel, path in session.execute(text(
                "SELECT s.id, s.duration_s, f.n_events, "
                "f.harmonic_percussive_ratio, f.onset_rate_hz, s.rel_path, s.path FROM samples s "
                "JOIN sample_features f ON f.sample_id = s.id")):
            cls = audio_label(rel or path or "", dur, n_ev, onsets, hpr)
            add(sid, {cls} if cls else set(), duration_s=dur, harmonicity=hpr)
    else:
        raise KeyError(provider)
    _flush(session, labels, descs)
    n_lab = session.execute(text("SELECT count(*) FROM labels WHERE provider = :p"), p).scalar()
    n_des = session.execute(text("SELECT count(*) FROM descriptors WHERE provider = :p"), p).scalar()
    log(f"{provider} metadata: {n_lab:,} labels, {n_des:,} descriptors")
    return {"labels": n_lab, "descriptors": n_des}


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
    """Rebuild the built-in providers whose sample counts drifted or whose rules changed.
    Returns the rebuilt ones."""
    out = []
    for prov in SHADOW:
        need, have = covered_count(session, prov)
        if need != have:
            rebuild(session, prov, log=log)
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
