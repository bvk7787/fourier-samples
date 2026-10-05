"""Name, tag and path rules: what a file's name, its folders, Live's tags and the classifier
say about it. The routing predicates the build (curate.compute_homes, _select_records),
verify and why share; split out of curate.py, which re-exports them."""
from __future__ import annotations

import json
import os
import re

from ..places import library_rel
from .curate_config import (
    ACOUSTIC_ABLETON_TAGS, ACOUSTIC_MALLET_PACKS, ACOUSTIC_BAND_TAGS, ACOUSTIC_NAME_RE, CLASSIC_BREAK_PATH_RE,
    CLASSIC_BREAK_RE, CYMBAL_CRASH_RE, CYMBAL_RIDE_RE, DRUM_NAME_RULES, FX_AB_BANDS,
    FX_BANDS, FX_DRUM_SON, FX_FALL_NAME_RE, FX_RISE_NAME_RE, FX_SYNTH_SON,
    FX_TRANSITION_NAME_RE, FX_TRANSITION_NOT_RE, GENERIC_ACOUSTIC_TAGS, HAT_CLOSED_RE,
    HAT_OPEN_MIN_S, HAT_OPEN_RE, IR_PATH_RE, KEYBOARD_NAME_RE, KEYS_CHORD_HOME,
    KEYS_CHORD_MAX_DUR, KEYS_CHORD_NAME_RE, KEYS_CLAP_MARGIN, KEYS_EXCLUDE,
    KEYS_MIN_HARMONICITY, KEYS_PHRASE_NAME_RE, LOOP_FULL_RE, LOOP_KIT_TAGS, LOOP_TOPS_RE,
    MALLET_INSTRUMENTS, MALLET_NAME_RE, NAME_RESERVED, ORCH_PACKS, ORCH_PATH_INSTRUMENT_RE,
    ORCH_PATH_RE, ORGAN_NAME_RE, PERC_SOURCE_NOT_RE, PERC_SOURCE_RE, PHRASES_CATEGORY,
    PLUCKED_NAME_RE, PREVIEW_PATH_RE, SCRATCH_NAME_EXCLUDE, SCRATCH_NAME_RE,
    STAB_CHORD_NAME_RE, STAB_MOVE_EXCLUDE, STAB_NAME_EXCLUDE, STAB_NAME_RE, STRING_NAME_RE,
    WAVE_MAX_DUR, WAVE_NOT_PATH_RE, WAVE_PATH_RE, WAVE_SINGLE_MAX_SAMPLES, WIND_NAME_RE,
    WIND_RESERVE_NOT_RE, WIND_RESERVE_NOT_TAGS,
)


def _has_any_tag(ableton_tags, tagset):
    return any(t in tagset for t in ableton_tags)


def _pack_of(rel_path: str | None, path: str) -> str:
    """The pack a file came from: its first folder, or its second under an umbrella folder
    (UMBRELLA_VENDORS, INSTRUMENT_ROOTS), minus the vendor's suffix (PACK_SUFFIX_RE)."""
    from .curate_config import INSTRUMENT_ROOTS, PACK_SUFFIX_RE, UMBRELLA_VENDORS
    rel = rel_path or library_rel(path)
    segs = [s for s in rel.split("/") if s]
    umbrella = segs and (segs[0] in INSTRUMENT_ROOTS or segs[0] in UMBRELLA_VENDORS) and len(segs) > 1
    src = segs[1] if umbrella else (segs[0] if segs else "?")
    return re.sub(r"\s+(?:" + PACK_SUFFIX_RE.pattern + r")\b", "", src, flags=re.I).strip()


def _is_nameable_vendor(rel_path: str | None, path: str) -> bool:
    """A file from an umbrella vendor whose packs may lead a folder name (NAMEABLE_VENDORS)."""
    from .curate_config import NAMEABLE_VENDORS
    rel = rel_path or library_rel(path)
    return rel.split("/", 1)[0] in NAMEABLE_VENDORS


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            j = json.loads(v)
            return j if isinstance(j, list) else [v]
        except Exception:
            return [v]
    return list(v)


def _labels_for(cfg):
    """The canonical labels (metadata/vocab.py) a category accepts from the classifier."""
    return list(cfg.get("labels") or ())


def _canonical_of(r):
    """A row's classifier labels in the canonical vocabulary: its "canonical" field when the
    row came from metadata/rows.py, else its classes and categories mapped here."""
    canon = getattr(r, "canonical", None)
    if canon is not None:
        return set(_as_list(canon))
    from ..metadata.vocab import mapping
    m = mapping("sononym")
    return ({m[("class", c)] for c in _as_list(getattr(r, "classes", None)) if ("class", c) in m}
            | {m[("category", c)] for c in _as_list(getattr(r, "categories", None)) if ("category", c) in m})


def _classifier_votes(part, canon):
    """Categories a sample's canonical classifier labels put it in: a one-shot category
    for a one-shot with one of its labels, a gated category for any sample with one, and
    every loop category but PHRASES (routed on its own) for a loop."""
    is_oneshot, is_loop = "class.oneshot" in canon, "class.loop" in canon
    out = set()
    for cat, cfg in part.items():
        k = cfg["kind"]
        if (k == "oneshot" and is_oneshot) or k == "gated":
            if canon & set(_labels_for(cfg)):
                out.add(cat)
        elif k == "loop" and is_loop and cat != PHRASES_CATEGORY:
            out.add(cat)
    return out


def _is_wave(rel_path, duration):
    """A single-cycle waveform or wavetable frame: under a wavetable / single-cycle path
    and no longer than WAVE_MAX_DUR. Waves belong to WAVES and nowhere else."""
    d = os.path.dirname(rel_path or "")
    return (duration is not None and duration <= WAVE_MAX_DUR
            and bool(WAVE_PATH_RE.search(d)) and not WAVE_NOT_PATH_RE.search(d))


def _is_fx_transition(name):
    """Named as a riser / downlifter / sweep (FX_TRANSITION_NAME_RE), not a sweep pad."""
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    return bool(FX_TRANSITION_NAME_RE.search(stem)) and not FX_TRANSITION_NOT_RE.search(stem)


def _wave_source(rel_path):
    """Short source label for a wave: the folder under a WAVETABLES directory
    ("Vendor A - Bass Patches" -> "vendor-a"), else the pack ("Wave Pack A")."""
    segs = [x for x in (rel_path or "").split("/") if x][:-1]
    lab = None
    for i, sg in enumerate(segs):
        if sg.lower() in ("wavetables", "wave tables", "single cycles", "single cycle") and i + 1 < len(segs):
            lab = segs[i + 1]
            break
    if lab is None:
        lab = _pack_of(rel_path, rel_path or "")
    words = [w for w in re.split(r"[^a-z0-9]+", lab.lower()) if w]
    return "-".join(words[:2]) or None


def _wave_band(duration, sample_rate):
    """'cycle' for one frame (<= WAVE_SINGLE_MAX_SAMPLES samples), 'table' for several."""
    if duration is None or not sample_rate:
        return "cycle"
    return "cycle" if round(duration * sample_rate) <= WAVE_SINGLE_MAX_SAMPLES else "table"


def _cymbal_band(filename, son_labels=(), ab_tags=()):
    """'ride', 'crash' or 'cymbal': the name, then Sononym, then Ableton (CYMBAL_*_RE)."""
    stem = re.sub(r"[_\-.]+", " ", os.path.splitext(os.path.basename(filename or ""))[0])
    ride, crash = bool(CYMBAL_RIDE_RE.search(stem)), bool(CYMBAL_CRASH_RE.search(stem))
    if ride != crash:
        return "ride" if ride else "crash"
    son = set(son_labels or ())
    r2, c2 = "Perc Cymbal Rides" in son, "Perc Cymbal Crashes" in son
    if r2 != c2:
        return "ride" if r2 else "crash"
    tags = set(ab_tags or ())
    if ("Ride" in tags) != ("Crash" in tags):
        return "ride" if "Ride" in tags else "crash"
    return "cymbal"


def _hat_band(filename, duration):
    """'open' or 'closed': the name decides when it says so, else the length."""
    stem = os.path.splitext(filename or "")[0]
    if HAT_OPEN_RE.search(stem):
        return "open"
    if HAT_CLOSED_RE.search(stem):
        return "closed"
    return "open" if (duration or 0.0) >= HAT_OPEN_MIN_S else "closed"


def _fx_band(son_labels, ab_tags, filename=None, rel_path=None):
    """'scratch' for a file named as a turntable scratch (SCRATCHES folded into FX), else
    the source library's FX type (first Sononym XFX label, else an Ableton FX tag). With
    neither: 'synth' when Sononym hears it as pitched (a lead / bass / blip / zap), else
    'misc', which _resolve_misc_fx places by CLAP."""
    if filename and _scratch_named(filename, rel_path):
        return "scratch"
    if filename:
        stem = re.sub(r"[_\-.]+", " ", os.path.splitext(os.path.basename(filename))[0])
        up, down = bool(FX_RISE_NAME_RE.search(stem)), bool(FX_FALL_NAME_RE.search(stem))
        if up != down:
            return "riser" if up else "downlifter"      # the name says which way it goes
    for lab in son_labels or ():
        for key, band in FX_BANDS:
            if key in lab:
                return band
    for t in ab_tags or ():
        if t in FX_AB_BANDS:
            return FX_AB_BANDS[t]
    if any(lab.startswith(FX_SYNTH_SON) for lab in son_labels or ()):
        return "synth"
    return "misc"


def _fx_drum_leak(son_labels, ab_tags):
    """A kit drum (by Sononym) whose only FX evidence is Ableton's catch-all "Sound FX"."""
    son = list(son_labels or ())
    return (bool(set(son) & set(FX_DRUM_SON)) and not any("XFX" in lab for lab in son)
            and not any(t in FX_AB_BANDS for t in ab_tags or ()))


def _mallet_label(name, ab_tags=()):
    """Tuned-percussion instrument a file names ("Xylophone A4" -> "xylophone"); a
    "Synth Mallets"-tagged file is a synth-mallet. None if it names none."""
    if "Synth Mallets" in (ab_tags or ()):
        return "synth-mallet"
    for rx, lab in MALLET_INSTRUMENTS:
        if rx.search(name or ""):
            return lab
    return None


def _acoustic_tagged(ab_tags, name, pack, tags=None):
    """Ableton places the file with an acoustic instrument: a specific tag (Cello, Harp,
    Acoustic Guitar, ...) or a catch-all one (Misc Plucked, ...) backed by the name or pack."""
    tags = ACOUSTIC_ABLETON_TAGS if tags is None else tags
    hit = [t for t in ab_tags or () if t in tags]
    if any(t not in GENERIC_ACOUSTIC_TAGS for t in hit):
        return True
    return bool(hit) and bool(ACOUSTIC_NAME_RE.search(name or "") or ORCH_PACKS.search(pack or ""))


def _loop_band(filename, ab_tags, rel=None):
    """'classic' for the classic breaks (CLASSIC_BREAK_*), 'tops' for hat / percussion loops
    (layer them over a break), else 'full'."""
    stem = os.path.splitext(filename or "")[0]
    if CLASSIC_BREAK_RE.search(stem) or (rel and CLASSIC_BREAK_PATH_RE.search(os.path.dirname(rel))):
        return "classic"
    if LOOP_TOPS_RE.search(stem) and not LOOP_FULL_RE.search(stem):
        return "tops"
    if LOOP_FULL_RE.search(stem):
        return "full"
    tags = set(ab_tags or ())
    if tags & set(LOOP_KIT_TAGS):
        return "full"
    if tags & {"Closed Hihat", "Open Hihat", "Shaker", "Tambourine", "Conga", "Bongo", "Ride",
               "Cabasa", "Timbale", "Cowbell", "Wood", "Woodblock", "Guiro"}:
        return "tops"
    return "full"


def _acoustic_band(filename, ab_tags=(), pack=None, rel=None):
    """ACOUSTIC's band: 'mallet' (tuned percussion, timpani, gong), 'plucked', 'wind'
    (woodwind, brass, reeds) or 'string' (bowed and orchestral strings). The name
    decides first, then an orchestra instrument folder (ORCH_PATH_RE), then Ableton's
    instrument tag, then the pack."""
    stem = os.path.splitext(filename or "")[0]
    tags = set(ab_tags or ())
    if _is_mallet_named(stem) or re.search(r"(?<![a-z])(timpani|gong)(?![a-z])", stem, re.I):
        return "mallet"
    if _is_plucked(stem):
        return "plucked"
    op = _orch_path(rel)
    if op:
        return "plucked" if op[1] == "harp" else ("string" if op[0] == "strings" else "wind")
    if WIND_NAME_RE.search(stem):
        return "wind"
    if STRING_NAME_RE.search(stem):
        return "string"
    for band in ("mallet", "plucked", "wind", "string"):
        if tags & ACOUSTIC_BAND_TAGS[band]:
            return band
    if ACOUSTIC_MALLET_PACKS.search(pack or ""):
        return "mallet"
    pk = (pack or "").lower()
    if "brass" in pk or "woodwind" in pk:
        return "wind"
    return "string"


# ---------------------------------------------------------------------------
# Keys (piano / e-piano / clav) routing -- see curate_config KEYS_*
# ---------------------------------------------------------------------------
def _is_plucked(name):
    """A plucked-string one-shot by name (guitar, zither, lyre, ...), not a synth patch."""
    return bool(PLUCKED_NAME_RE.search(name or "")) and "synth" not in (name or "").lower()


def _is_synth_tagged(ab_tags):
    """Ableton calls it a synth voice ("Synth Mallets", "Synth Keys", ...)."""
    return any(t.startswith("Synth ") for t in ab_tags or ())


def _is_mallet_named(name):
    """Tuned percussion by name (xylophone, kalimba, church bell, ...), not a synth / FM patch."""
    nl = (name or "").lower()
    return bool(MALLET_NAME_RE.search(nl)) and "synth" not in nl and not re.search(r"(?<![a-z])fm(?![a-z])", nl)


def _is_acoustic_named(name, ab_tags=()):
    """A one-shot whose name makes it ACOUSTIC's: a plucked string or tuned percussion,
    unless Ableton calls it a synth voice."""
    return (_is_plucked(name) or _is_mallet_named(name)) and not _is_synth_tagged(ab_tags)


def _is_organ_named(name):
    """An organ by name (ORGAN_NAME_RE), not an organ stab (those are STABS')."""
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    return bool(ORGAN_NAME_RE.search(stem)) and not _stab_named(stem)


def _is_wind_named(name, ab_tags=()):
    """A one-shot named as a wind or brass instrument (WIND_NAME_RE): not an air horn,
    whistle or FX (WIND_RESERVE_NOT_RE), a stab, or a synth patch."""
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    tags = set(ab_tags or ())
    return (bool(WIND_NAME_RE.search(stem)) and not WIND_RESERVE_NOT_RE.search(stem)
            and not _stab_named(stem) and not _is_synth_tagged(ab_tags)
            and not tags & set(WIND_RESERVE_NOT_TAGS))


def _perc_sourced(rel_path, name):
    """A file from a drum / percussion folder or an acoustic-percussion pack (PERC_SOURCE_RE)
    whose name doesn't say synth: a SYNTH candidate so sourced is PERC's."""
    rel = "/" + library_rel(rel_path or "")
    return bool(PERC_SOURCE_RE.search(rel)) and not PERC_SOURCE_NOT_RE.search(os.path.basename(name or ""))


def _orch_path(rel_path):
    """(section, instrument) for a file under an orchestra library's Strings / Brass /
    Woodwinds folders (ORCH_PATH_RE), e.g. ("brass", "trombone"); None elsewhere."""
    m = ORCH_PATH_RE.search("/" + library_rel(rel_path or ""))
    if not m:
        return None
    inst = ORCH_PATH_INSTRUMENT_RE.search(m.group(2))
    return m.group(1).lower(), (inst.group(0).lower() if inst else None)


def _acoustic_reserved(name, ab_tags=(), rel_path=None):
    """A one-shot that is ACOUSTIC's whatever else votes for it: a named plucked string or
    tuned percussion (_is_acoustic_named), a named wind / brass instrument, or an
    orchestral instrument by folder (ORCH_PATH_RE)."""
    return (_is_acoustic_named(name, ab_tags) or _is_wind_named(name, ab_tags)
            or _orch_path(rel_path) is not None)


def _plucked_label(name):
    """Instrument word for a plucked-string file ("Cigar Box C2" -> "cigar-box")."""
    m = PLUCKED_NAME_RE.search(name or "")
    return re.sub(r"[^a-z0-9]+", "-", m.group(0).lower()).strip("-") if m else None


def _is_keys(name, vec, harmonicity, anchors):
    """A piano / e-piano / clav sample: named as keys, or CLAP puts it with PIANO's
    anchors by KEYS_CLAP_MARGIN and it is clearly pitched (so a paper scrunch or a ride
    that CLAP half-hears as piano stays put)."""
    if KEYBOARD_NAME_RE.search(name or ""):
        return True
    if vec is None or anchors is None:
        return False
    pa, pn = anchors
    a, n = float(vec @ pa), float(vec @ pn)
    return a - n >= KEYS_CLAP_MARGIN and (harmonicity or 0.0) >= KEYS_MIN_HARMONICITY


def _is_chord(name, chroma, min_chroma, harmonicity=1.0):
    """A chord rather than a single note: named as one, or pitched but with chroma too
    spread for one pitch (below the single-note threshold PIANO uses). Unpitched
    sounds (a piano thud or rattle) also have spread chroma, so that test needs a
    clearly pitched sample."""
    if KEYS_CHORD_NAME_RE.search(os.path.splitext(name or "")[0]):
        return True
    return (chroma is not None and chroma < min_chroma
            and (harmonicity or 0.0) >= KEYS_MIN_HARMONICITY)


def _is_stab(name, duration, single_event):
    """A chord one-shot short and unmetered enough to be a stab (not a progression,
    a bpm-named phrase or a long keys pad)."""
    return (single_event and (duration is None or duration <= KEYS_CHORD_MAX_DUR)
            and not KEYS_PHRASE_NAME_RE.search(name or ""))


def _stab_named(name):
    """Named as a stab: a chord (STAB_CHORD_NAME_RE) or a stab / tonal hit (STAB_NAME_RE),
    with no drum word (STAB_NAME_EXCLUDE). What STABS may hold."""
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    return bool((STAB_CHORD_NAME_RE.search(stem) or STAB_NAME_RE.search(stem))
                and not STAB_NAME_EXCLUDE.search(stem))


def _is_named_stab(name, duration, n_events, oneshot=True):
    """A short single-hit one-shot that belongs in STABS by its name: a chord, or a stab /
    tonal hit that isn't a bass or vocal stab (SUB's and VOX's)."""
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    if not (oneshot and _stab_named(stem) and _is_stab(stem, duration, (n_events or 1) <= 1)):
        return False
    return bool(STAB_CHORD_NAME_RE.search(stem)) or not STAB_MOVE_EXCLUDE.search(stem)


def _scratch_named(name, rel_path=None):
    """Named as a turntable scratch (the file, or the folder it sits in), not a scratchy
    texture or a drum (SCRATCH_NAME_RE / SCRATCH_NAME_EXCLUDE)."""
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    folder = os.path.basename(os.path.dirname(rel_path or ""))
    if SCRATCH_NAME_EXCLUDE.search(stem):
        return False
    return bool(SCRATCH_NAME_RE.search(stem) or (folder and SCRATCH_NAME_RE.search(folder)))


def _drum_named(name):
    """The drum category a filename names, when it names exactly one kind (DRUM_NAME_RULES);
    None for no drum word or two kinds ("Kick Snare")."""
    stem = os.path.splitext(os.path.basename(name or ""))[0].replace("_", " ")
    hits = {cat for rx, cat in DRUM_NAME_RULES if rx.search(stem)}
    if hits == {"HATS", "CYMBALS"}:
        return "HATS" if re.search(r"hi-?hat|hihat|(?<![a-z])(?:hh|oh|ch|hat)(?![a-z])", stem, re.I) else "CYMBALS"
    return next(iter(hits)) if len(hits) == 1 else None


_SHAKEN_RE = re.compile(r"(?<![a-z])(?:shaker|tambourine|tamb|cabasa|maraca)s?(?![a-z])", re.I)
_HAT_WORD_RE = re.compile(r"hi-?hats?|hihats?|(?<![a-z])(?:hh|oh|ch|hat|hats)(?![a-z])", re.I)
_PERC_FOLDER_RE = re.compile(r"(?<![a-z])(?:percs?|percussions?)(?![a-z])", re.I)


def _drum_named_in(name, rel=None, fallback=False):
    """_drum_named, and without Sononym (fallback) the file's folder too: a shaker or
    tambourine (HATS by its name) in a percussion folder ("Perc/...") is PERC's, as its
    folder says, unless its name also says hi-hat."""
    dn = _drum_named(name)
    if not (fallback and dn == "HATS" and rel):
        return dn
    from ..metadata.shadow import shape_scope
    from .curate_config import CATEGORIES
    stem = os.path.splitext(os.path.basename(name or ""))[0].replace("_", " ")
    _n, folder = shape_scope(rel)
    if _SHAKEN_RE.search(stem) and not _HAT_WORD_RE.search(stem) and _PERC_FOLDER_RE.search(folder) \
            and "PERC" in CATEGORIES:
        return "PERC"
    return dn


def _keys_candidates(cand, keys, chord, single_event):
    """Candidate homes for a keys sample: never KEYS_EXCLUDE; a chord stab may also
    home in KEYS_CHORD_HOME."""
    if not keys:
        return cand
    out = set(cand) - KEYS_EXCLUDE
    if KEYS_CHORD_HOME and chord and single_event:
        out.add(KEYS_CHORD_HOME)
    return out


def _acoustic_label(d, tags):
    """Instrument word for an ACOUSTIC record: a specific Ableton tag (Cello, Harp, ...),
    else the plucked string its name says, else a catch-all tag (Misc Plucked)."""
    op = _orch_path(d.get("path"))
    if op and op[1]:
        return op[1]                 # the orchestra folder beats Ableton's guess ("Piano" on a trombone)
    ab = [t for t in d.get("ab", []) if t in tags]
    spec = [t for t in ab if t not in GENERIC_ACOUSTIC_TAGS]
    return spec[0] if spec else (d.get("plk") or (ab[0] if ab else None))


def _is_ir(rel_path, path=None):
    """A reverb impulse response, by the folder it sits in (IR_PATH_RE)."""
    rel = rel_path or library_rel(path or "")
    return bool(IR_PATH_RE.search(rel))


def _is_preview(rel_path, path=None):
    """A preset or kit preview render (PREVIEW_PATH_RE)."""
    rel = rel_path or library_rel(path or "")
    return bool(PREVIEW_PATH_RE.search(rel))


def _name_reserved(name):
    """Category a filename is reserved to (NAME_RESERVED), else None."""
    nl = (name or "").lower()
    for subs, cat in NAME_RESERVED:
        if any(s in nl for s in subs):
            return cat
    return None


def _is_loop_row(r):
    """True if Sononym or Ableton classes the sample as a loop."""
    return any("Loop" in c for c in _as_list(getattr(r, "classes", None))) or \
        "Loop" in _as_list(getattr(r, "ableton_tags", None))
