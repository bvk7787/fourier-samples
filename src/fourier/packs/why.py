"""`fourier why <file>`: where a sample landed and which rule put it there.

PRECEDENCE is the order the routing rules apply in (curate.compute_homes for the home,
curate._select_records for what a category accepts); the first rule that matches a file
decides it, and a later rule can't undo it. `explain` walks the table for one sample
and reports every rule that matched, the one that decided, the classifier votes behind
the rest, and where the current master holds the file.

It replays the name / tag / path rules exactly; the CLAP steps (the keys detector, the
tie-break between two voted homes, a category's CLAP gate) and selection (budget, near-
duplicate pruning, quality gates) are reported, not re-run: a file with a home but no place
in the master says what the last build recorded for it (packs/why_log.py: the per-vendor
share, a near-duplicate of which file, a gate, the budget, or a category left empty).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from ..places import library_rel

# (rule id, what it does, where it applies). Order = precedence.
PRECEDENCE = [
    ("drop", "rated Drop: out of every category", "selection"),
    ("pin", "rated Keep in X, or Move-X: in X, whatever the rules below say", "selection"),
    ("mirror", "a copy in a mirror folder (a pack reorganised into another folder): left out, the original counts", "homes"),
    ("ir_preview", "a reverb impulse response or a preset preview: left out", "homes"),
    ("pack_home", "a pack with a fixed home (PACK_HOME, set by a library overlay): that category only", "selection"),
    ("inst_pack", "an instrument-routed pack (INST_ROUTED_PACKS): its instrument category only", "homes"),
    ("phrase", "a musical phrase (producer-marked, whole bars): PHRASES", "homes"),
    ("acoustic", "an acoustic instrument by tag, name or orchestra folder (ORCH_PATH_RE): ACOUSTIC", "homes"),
    ("organ", "a named organ one-shot (not a stab): PIANO", "homes"),
    ("override", "a NAME_OVERRIDES word (kalimba, cowbell, ...): that category", "selection"),
    ("sound", "the sound model (without Sononym; trained on pack makers' names and your ratings): a "
              "file no name rule recognized, or one only its pack's name or a weak word describes, "
              "where it is confident", "homes"),
    ("keys", "keys (by name here; CLAP in the build): never SYNTH / SUB / FX / STABS or a drum category", "homes"),
    ("scratch", "a named scratch: FX", "homes"),
    ("fx_transition", "a named riser / downlifter / sweep: FX", "homes"),
    ("drum_name", "named as exactly one kind of drum (DRUM_NAME_RULES): that drum category", "homes"),
    ("perc_source", "a SYNTH candidate from a drum / percussion folder or pack (PERC_SOURCE_RE): PERC", "homes"),
    ("stab", "named as a stab or chord, short: STABS; STABS holds named stabs only", "homes"),
    ("misfiled", "rated Misfiled in X (or auto-detected): never X", "homes + selection"),
    ("reserved", "a NAME_RESERVED word: that category only", "homes + selection"),
    ("votes", "the classifier (Sononym, else the path and audio providers) and Live's tags (else "
              "path words) vote; most votes wins, CLAP breaks ties", "homes"),
    ("clap", "no name rule matched (without Sononym): placed by sound, at the category whose CLAP "
             "prompts it sits closest to, when it clears that category's gate by a margin", "homes"),
]
RULE_IDS = [r[0] for r in PRECEDENCE]


@dataclass
class Why:
    path: str
    matched: list = field(default_factory=list)     # [(rule id, detail)]
    decided: tuple | None = None                     # (rule id, category or None)
    votes: dict = field(default_factory=dict)       # {provider: [categories]}, e.g. {"sononym": [...], "ableton": [...]}
    labels: dict = field(default_factory=dict)      # without Sononym: {"path": [...], "audio": [...]} canonical labels
    rating: dict | None = None
    in_master: list = field(default_factory=list)   # [(category, out)]
    left_out: list = field(default_factory=list)    # [(category, why the last build left it out)]
    gate: list = field(default_factory=list)        # [(category, its CLAP gate result)] for a file in the master
    recorded: bool = False                           # the last build recorded its reasons
    tempo: tuple | None = None                       # a loop's (bpm or None, source, measured, bars)
    unrecognized: bool = False                       # no rule recognized it, nor did the CLAP fallback place it
    differ: list = field(default_factory=list)       # [(what, "174 (Sononym) / 87 (Fourier's own analysis)")]
    sources: dict = field(default_factory=dict)      # {what: source} the last build took (why log)
    twin: tuple | None = None                        # (its byte-identical copy, where the master holds it)
    sound: tuple | None = None                       # the sound model's (category, p, loop p), routing or not
    sound_routes: bool = False                       # ...and whether it routes (without Sononym)


def _clap_homed(sid, why_docs, key="clap_homed"):
    """The category the last build's CLAP fallback homed a sample in (its why log), or None;
    key="clap_rehomed": the loop category a loop fell back to by sound (a phrase-marked loop
    in a drum-loop category, a drum loop in PHRASES)."""
    if why_docs is None:
        return None
    from .curate_config import CATEGORIES
    for cat in CATEGORIES:
        doc = why_docs(cat) or {}
        if int(sid) in (doc.get(key) or ()):
            return cat
    return None


def _sound_homed(sid, why_docs):
    """(category, probability) the last build's sound model homed a sample in (its why log),
    or None."""
    if why_docs is None:
        return None
    from .curate_config import CATEGORIES
    for cat in CATEGORIES:
        p = ((why_docs(cat) or {}).get("sound_homed") or {}).get(str(int(sid)))
        if p is not None:
            return cat, float(p)
    return None


def _rows(session, query, limit=5):
    from sqlalchemy import or_
    from ..db.models import Sample
    from ..metadata.rows import HOME_FIELDS, fetch, sample_select
    q = sample_select(*HOME_FIELDS, session=session)
    from .curate import _with_name_tags          # without Live, path words stand in for its tags
    exact = fetch(session, q.where(Sample.path == query))
    if exact:
        return _with_name_tags(session, exact)
    like = f"%{query}%"
    return _with_name_tags(session, fetch(
        session, q.where(or_(Sample.filename.ilike(like), Sample.rel_path.ilike(like))).order_by(Sample.id).limit(limit)))


# where a loop's tempo came from (metadata/resolve.py's source of curate._resolve_tempo's
# step), in words
_TEMPO_FROM = {"sononym": "Sononym's analysis", "fourier:audio": "Fourier's own analysis",
               "fourier:name": "its name", "acid": "the WAV's ACID chunk",
               "fourier:folder": "its folder's name", "fourier:length": "its length"}

# where a stored rating came from (ratings.json's "source"), in words
_RATED_BY = {"tag": "a Live tag", "favorite": "a Live Favorite", "cli": "fourier review rate",
             "csv": "fourier review import"}


def _rated_when(r: dict) -> str:
    """" (on 2026-01-02, by fourier review rate)" for a stored rating: when and how it was given."""
    day = (r.get("rated_at") or "")[:10]
    by = _RATED_BY.get(r.get("source") or "", r.get("source"))
    parts = ([f"on {day}"] if day else []) + ([f"by {by}"] if by else [])
    return f" ({', '.join(parts)})" if parts else ""


def explain_row(r, manifest=None, store_path=None, mirrors=frozenset(), act=None, why_docs=None,
                name_of=str, place_of=None, readings=None, twin_of=None, sound=None):
    """Why for one DB row (the columns compute_homes reads). act: the providers in use
    (metadata/providers.py Active), which name the votes; None: Sononym and Live. why_docs:
    category -> the last build's left-out reasons (why_log), for a file not in the master.
    readings: the sample's readings by source (metadata/resolve.readings), whose
    disagreements are shown. twin_of: row -> (name, place) of a byte-identical copy the
    master holds, or None. sound: the sound model's (category, probability, loop
    probability) for it (metadata/sound.current), shown whether or not it routes."""
    from . import curate as C
    from .curate_config import (CATEGORIES, DRUM_NAME_FROM, FX_TRANSITION_FROM, INST_ROUTED_PACKS,
                                KEYBOARD_NAME_RE, KEYS_EXCLUDE, ORCH_PACK_EXCLUDE, PACK_HOME,
                                PHRASES_CATEGORY, SCRATCH_FROM, SCRATCH_HOME, STAB_CHORD_FROM)
    from .curate_config import ABLETON_TAGS_BY_CATEGORY
    from .ratings import load_store
    w = Why(path=r.path, sound=sound, sound_routes=act is not None and act.fallback)
    fn = os.path.basename(r.path or "")
    pack = C._pack_of(r.rel_path, r.path)
    classes, ab = C._as_list(r.classes), C._as_list(r.ableton_tags)
    is_oneshot = any("OneShot" in c for c in classes)
    is_loop = any("Loop" in c for c in classes)
    oneshot = is_oneshot and not is_loop

    try:
        R = load_store(store_path)["ratings"]
    except (OSError, ValueError):
        R = {}
    w.rating = R.get(r.path)
    v = (w.rating or {}).get("verdict")
    if C._is_loop_row(r):
        from ..metadata.resolve import loop_tempo
        got, _chain = loop_tempo(r, act is not None and act.fallback)
        bpm = got.value
        bars = int(round(bpm * r.duration_s / 240.0)) if bpm and r.duration_s else None
        w.tempo = (bpm, got.source, r.tempo_bpm or getattr(r, "bpm", None), bars)
    if readings is not None:
        from ..metadata.resolve import differences
        w.differ = differences(readings)

    def hit(rule, detail, cat=None, decides=True):
        w.matched.append((rule, detail))
        if decides and w.decided is None:
            w.decided = (rule, cat)

    if v == "drop":
        hit("drop", f"rated Drop in {w.rating.get('category')}{_rated_when(w.rating)}")
    if v == "keep":
        hit("pin", f"rated Keep in {w.rating['category']}", w.rating["category"])
    elif v == "misfiled" and w.rating.get("target"):
        hit("pin", f"rated Move-{w.rating['target']} (from {w.rating['category']})", w.rating["target"])
    if r.path in mirrors:
        hit("mirror", "a mirror-folder copy")
    if C._is_ir(r.rel_path, r.path) or C._is_preview(r.rel_path, r.path):
        hit("ir_preview", "impulse response or preview")
    if pack in PACK_HOME:
        hit("pack_home", f"pack {pack}", PACK_HOME[pack])
    if INST_ROUTED_PACKS.search(pack):
        hit("inst_pack", f"pack {pack}")
    _rh = _clap_homed(r.id, why_docs, "clap_rehomed") if act is not None and act.fallback else None
    if PHRASES_CATEGORY in CATEGORIES and C._is_phrase(r, act is not None and act.fallback):
        if _rh:
            hit("phrase", f"producer-marked phrase, whole bars, but it sounds like a drum loop: "
                          f"fell back to {_rh} by sound (CLAP)", _rh)
        else:
            hit("phrase", "producer-marked phrase, whole bars", PHRASES_CATEGORY)
    elif _rh == PHRASES_CATEGORY:
        hit("phrase", f"a drum loop by its words or audio, but it sounds like a musical phrase: fell "
                      f"back to {PHRASES_CATEGORY} by sound (CLAP)", PHRASES_CATEGORY)
    if not ORCH_PACK_EXCLUDE.search(pack) and (
            C._acoustic_tagged(ab, fn, pack) or (oneshot and C._is_acoustic_named(fn, ab))
            or (not is_loop and C._is_wind_named(fn, ab)) or C._orch_path(r.rel_path) is not None):
        hit("acoustic", "acoustic instrument by tag / name / folder", "ACOUSTIC")
    if not is_loop and C._is_organ_named(fn):
        hit("organ", "organ-named", "PIANO")
    elif act is not None and act.fallback and not is_loop and C._keys_labeled(r) and "PIANO" in CATEGORIES:
        hit("keys", "named as keys (rhodes, wurli, e-piano, organ, piano: the path label \"keys\"): "
                    "PIANO takes it by that name", "PIANO")
    oi, ocat = C._row_override(r)
    if ocat:
        hit("override", f"NAME_OVERRIDES rule {oi}", ocat)

    # the classifier votes (as compute_homes counts them)
    tag2cats = {}
    for cat in CATEGORIES:
        for t in ABLETON_TAGS_BY_CATEGORY.get(cat, []):
            tag2cats.setdefault(t, set()).add(cat)
    son_cats, ab_cats = C._classifier_votes(CATEGORIES, C._canonical_of(r)), set()
    if act is not None and act.fallback:
        son_cats = C._fallback_loop_votes(son_cats, r, ab)
    for t in ab:
        ab_cats |= tag2cats.get(t, set())
    w.votes, w.labels = _credit(r, act, son_cats, ab_cats)
    cand = son_cats | ab_cats
    snd = _sound_homed(r.id, why_docs) if act is not None and act.fallback else None
    if snd and w.decided is None:
        hit("sound", f"sound model: {snd[0]} {snd[1]:.2f}, "
                     + ("no name rule matched" if not cand else
                        "only its pack's name or a weak word named a category"), snd[0])

    if KEYBOARD_NAME_RE.search(fn) and cand & (KEYS_EXCLUDE | C.KIT_CATS):
        hit("keys", "keys-named: kept out of " + "/".join(sorted(cand & (KEYS_EXCLUDE | C.KIT_CATS))),
            decides=False)
    scr = oneshot and C._scratch_named(fn, r.rel_path)
    if scr and cand & SCRATCH_FROM:
        hit("scratch", "scratch-named", SCRATCH_HOME)
    if not is_loop and not scr and cand & FX_TRANSITION_FROM and C._is_fx_transition(fn):
        hit("fx_transition", "riser / downlifter / sweep by name", "FX")
    dn = C._drum_named_in(fn, r.rel_path or r.path, act is not None and act.fallback) \
        if (oneshot and not scr) else None
    if dn and "VOX" not in cand and cand & DRUM_NAME_FROM:
        hit("drum_name", f"named as {dn}", dn)
    if not is_loop and not scr and "SYNTH" in cand and C._perc_sourced(r.rel_path, fn):
        hit("perc_source", "percussion folder or pack", "PERC")
    if C._is_named_stab(fn, r.duration_s, r.n_events, oneshot) and cand & STAB_CHORD_FROM \
            and not (C._is_fx_transition(fn) and not C._is_loop_row(r)):
        hit("stab", "stab / chord-named, short", "STABS")
    if v == "misfiled":
        hit("misfiled", f"rated Misfiled in {w.rating['category']}", decides=False)
    rv = C._name_reserved(fn)
    if rv:
        hit("reserved", "NAME_RESERVED", rv)
    if cand:
        hit("votes", "votes: " + ", ".join(
            f"{c} {(c in son_cats) + (c in ab_cats)}" for c in sorted(cand)), None)
    elif act is not None and act.fallback and w.decided is None:
        clap = _clap_homed(r.id, why_docs)
        if clap:
            hit("clap", "placed by sound (CLAP), no name rule matched", clap)
        else:
            w.unrecognized = True
    for cat, cd in ((manifest or {}).get("categories") or {}).items():
        for e in cd.get("entries", []):
            if e.get("src") == r.path:
                w.in_master.append((cat, e["out"]))
    for sname, sd in ((manifest or {}).get("sets") or {}).items():
        for e in sd.get("entries", []):
            if e.get("src") == r.path:
                w.in_master.append((sname, e["out"]))
    if not w.in_master and why_docs is not None:
        _left_out(w, r.id, [w.decided[1]] if w.decided and w.decided[1] else sorted(cand),
                  why_docs, name_of, place_of)
    if not w.in_master and twin_of is not None and not any("byte-identical" in t for _c, t in w.left_out):
        w.twin = twin_of(r)
    elif why_docs is not None:
        from .why_log import gate_result, readmitted_result
        for cat, _o in w.in_master:
            doc = (why_docs(cat) or {}) if cat in CATEGORIES else {}
            for g in (gate_result(doc, r.id), readmitted_result(doc, r.id)):
                if g:
                    w.gate.append((cat, g))
            w.sources = w.sources or dict((doc.get("sources") or {}).get(str(r.id)) or {})
    return w


def _left_out(w, sid, cats, why_docs, name_of, place_of=None):
    """What the last build recorded for a sample it didn't place, in each category it was
    a candidate for (why_log): the step that left it out, with a gate's value and threshold,
    and for a category left empty, its counts and this file's own reason."""
    from .curate import too_few_line
    from .why_log import describe, gate_result, reason_for
    for cat in cats:
        doc = why_docs(cat)
        if not doc:
            continue
        w.recorded = True
        g = gate_result(doc, sid)
        got = reason_for(doc, sid)
        # the CLAP gate's own failure is the gate result (with its scores)
        own = None if (got is None or (got[0] == "gated_out" and got[1] == "clap" and g)) else \
            describe(*got, doc, name_of=name_of, place_of=place_of)
        if doc.get("status") == "too_few":
            w.left_out.append((cat, f"{cat} was left empty: {too_few_line(doc)}"
                                    + (f"; this file {g}" if g else "")
                                    + ((f", then {own}" if g else f"; this file: {own}") if own else "")))
            continue
        if not got:
            continue
        if own is None:
            w.left_out.append((cat, g))                  # failed the gate, with its scores
        elif g:
            w.left_out.append((cat, f"{g}; left out because {own}"))
        else:
            w.left_out.append((cat, own))


_SOURCE = {"sononym": "sononym", "ableton": "Live tags", "path+audio": "path + audio",
           "path words": "path words (no Live)"}


def _credit(r, act, son_cats, ab_cats):
    """The votes under the names of the providers that cast them, and without Sononym the
    labels each built-in provider gave: ({source: [categories]}, {provider: [labels]})."""
    if act is None or not act.fallback:
        cls_name, labels = "sononym", {}
    else:
        from ..metadata.shadow import audio_label, path_labels
        cls_name = "+".join(act.classifiers)
        a = audio_label(r.rel_path or r.path or "", getattr(r, "duration_s", None),
                        getattr(r, "n_events", None), getattr(r, "onset_rate_hz", None),
                        getattr(r, "harmonicity", None))
        labels = {"path": sorted(path_labels(r.rel_path or r.path or "")), "audio": [a] if a else []}
        labels = {p: labs for p, labs in labels.items() if p in act.classifiers}
    if act is None or "ableton" in act.names:
        tag_name = "ableton"
    else:
        tag_name = "path words"          # without Live, path words stand in for its tags
    return {cls_name: sorted(son_cats), tag_name: sorted(ab_cats)}, labels


# each rule in plain words, for the one-line answer `fourier why` starts with
PLAIN = {
    "drop": "you rated it Drop",
    "pin": "you rated it Keep there (or moved it there)",
    "mirror": "it's a copy of a file in another folder, and the original counts",
    "ir_preview": "it's a reverb impulse response or a preset preview",
    "pack_home": "its pack always goes there (your overlay or the sources setting)",
    "inst_pack": "its pack is sorted by instrument",
    "phrase": "it's a musical phrase: a loop marked as one, in whole bars",
    "acoustic": "it's an acoustic instrument",
    "organ": "it's an organ",
    "override": "its name has a word that always goes there",
    "sound": "the sound model recognized it",
    "keys": "it's keys",
    "scratch": "it's a scratch",
    "fx_transition": "it's a riser, downlifter or sweep",
    "drum_name": "its name says which drum it is",
    "perc_source": "it comes from a drum or percussion folder",
    "stab": "it's named as a stab or chord",
    "misfiled": "you rated it misfiled in another category",
    "reserved": "its name has a word kept for that category",
    "votes": "its name, folders and sound point there",
    "clap": "it sounds most like that category (no name said what it is)",
}


def _plain_reason(why: str) -> str:
    """A left-out reason without the scores."""
    import re
    why = re.sub(r"^passed its CLAP gate \([^)]*\); (left out because )?", "", why)
    m = re.match(r"failed its CLAP gate \(score [^)]*\)", why)
    if m:
        return "it doesn't sound enough like the category (CLAP)" + why[m.end():]
    return why


def headline(w: Why) -> str:
    """Where it is and why, or why it isn't there, in one sentence."""
    if w.in_master:
        c, o = w.in_master[0]
        fam = os.path.dirname(o)
        where = f"{c}/{fam}" if fam else c
        rule = w.decided[0] if w.decided else None
        reason = PLAIN.get(rule)
        if w.rating and w.rating.get("verdict") in ("keep", "Keep") and rule != "pin":
            reason = reason or "you rated it Keep"
        more = f" (and {len(w.in_master) - 1} more place{'s' if len(w.in_master) > 2 else ''})" \
            if len(w.in_master) > 1 else ""
        return f"In {where}{more}" + (f": {reason}." if reason else ".")
    if w.twin:
        return f"Not in the master: it's a byte-identical copy of {w.twin[0]}, which the master holds."
    if any(rule == "drop" for rule, _d in w.matched):
        return "Not in the master: you rated it Drop."
    if any(rule in ("mirror", "ir_preview") for rule, _d in w.matched):
        rule = next(r for r, _d in w.matched if r in ("mirror", "ir_preview"))
        return f"Not in the master: {PLAIN[rule]}."
    if w.left_out:
        cat, why = w.left_out[0]
        return f"Not in the master: left out of {cat}, {_plain_reason(why)}."
    if w.unrecognized:
        return "Not in the master: nothing in its name or folders said what it is, and it didn't sound enough like any category."
    return "Not in the master."


def format_short(w: Why) -> list[str]:
    """`fourier why` without --detail: the file, the one-line answer, and what to do."""
    out = [library_rel(w.path or ""), "  " + headline(w)]
    if w.rating:
        r = w.rating
        out.append(f"  your rating: {r.get('verdict')}" + (f" (to {r['target']})" if r.get("target") else ""))
    if w.tempo and w.tempo[0] is not None:
        out.append(f"  tempo: {w.tempo[0]:g} BPM")
    for cat, why in w.left_out[1:]:
        out.append(f"  also left out of {cat}: {_plain_reason(why)}")
    if w.unrecognized and not w.in_master and not w.twin:
        out.append("  To place files like it, add your packs' word for the sound to fourier.toml "
                   "(`fourier config edit`): words = { KICKS = [\"bombo\"] }.")
    return out


def format_why(w: Why) -> list[str]:
    rel = library_rel(w.path or "")
    out = [rel]
    if w.in_master:
        out.append("  in the master: " + "; ".join(f"{c}/{o}" for c, o in w.in_master))
    else:
        out.append("  not in the master")
    if w.rating:
        r = w.rating
        out.append(f"  rating: {r.get('verdict')} in {r.get('category')}"
                   + (f" -> {r['target']}" if r.get("target") else "") + _rated_when(r))
    if w.tempo:
        bpm, src, measured = w.tempo[:3]
        if bpm is None:
            out.append("  tempo: none that makes it whole bars (a drum loop needs one to sync)")
        elif src == "fourier:length":
            bars = f"{w.tempo[3]} bar{'' if w.tempo[3] == 1 else 's'}" if len(w.tempo) > 3 and w.tempo[3] \
                else "whole bars"
            out.append(f"  tempo: {bpm:g} BPM, tempo from length ({bars}; the analysis read "
                       f"{measured:g})")
        else:
            out.append(f"  tempo: {bpm:g} BPM (from {_TEMPO_FROM.get(src, src)})")
    if w.sources:
        from ..metadata.resolve import SAID_BY
        out.append("  the last build took: " + ", ".join(
            f"{what} from {SAID_BY.get(src, src)}" for what, src in sorted(w.sources.items())))
    if w.differ:
        out.append("  readings differ: " + "; ".join(f"{what} {txt}" for what, txt in w.differ))
    if w.decided:
        rule, cat = w.decided
        if rule == "votes" and cat is None and w.in_master:
            cat = w.in_master[0][0] + " (by support, then CLAP)"
        desc = dict((a, b) for a, b, _ in PRECEDENCE)[rule]
        out.append(f"  decided by: {rule}" + (f" -> {cat}" if cat else "") + f"  ({desc})")
    out.append("  votes: " + " | ".join(f"{_SOURCE.get(p, p)}: {', '.join(cats) or '-'}"
                                         for p, cats in w.votes.items()))
    if w.labels:
        out.append("  labels: " + "; ".join(f"{p}: {', '.join(labs) or '-'}" for p, labs in w.labels.items()))
    if w.sound:
        from ..metadata.sound import describe_label
        out.append(f"  sound model: {describe_label(w.sound)}"
                   + ("" if w.sound_routes else " (a report: Sononym routes)"))
    for rule, detail in w.matched:
        out.append(f"    {RULE_IDS.index(rule) + 1:2d}. {rule:<14} {detail}")
    for cat, why in w.left_out:
        out.append(f"  left out of {cat}: {why}")
    for cat, g in w.gate:
        out.append(f"  {cat}: {g}")
    home = w.decided[1] if w.decided else None
    if w.twin and not w.in_master:
        out.append(f"  a byte-identical copy of {w.twin[0]}, which the master holds ({w.twin[1]})")
    if w.unrecognized and not w.in_master and not w.twin:
        out.append("  no rule recognized it: no category word in its name or folders (path labels "
                   "above), and the CLAP fallback didn't place it by sound. Add your word with the "
                   "`words` knob in fourier.toml (words = { KICKS = [\"bombo\"] }), rename the "
                   "folder, or set a pack's home in a library overlay.")
    # a reason already given above (a Drop rating, a mirror copy, an impulse response or
    # preview, a byte-identical copy): no list of possible ones after it
    specific = w.twin or any(rule in ("drop", "mirror", "ir_preview") for rule, _d in w.matched)
    if not w.in_master and not w.left_out and not specific:
        if w.recorded:
            out.append(f"  ({'home ' + home + ', ' if home else ''}not among the candidates the last "
                       f"build's gates saw: a duplicate copy, clipped, too long, or rated Drop)")
        elif home:
            out.append(f"  (home {home} but not in the master: the next `fourier build` records why)")
    return out


def explain(session, query, master_dir=None, store_path=None, limit=5):
    """[Why] for the samples a path or name fragment matches (at most `limit`)."""
    man = None
    if master_dir and os.path.exists(os.path.join(master_dir, "manifest.json")):
        with open(os.path.join(master_dir, "manifest.json")) as f:
            man = json.load(f)
    rows = _rows(session, query, limit)
    try:
        from ..metadata.providers import active
        act = active(session)
    except Exception:
        act = None
    mirrors = frozenset()
    if rows:
        try:
            from .curate import _mirror_dups
            mirrors = frozenset(_mirror_dups(session))
        except Exception:
            pass
    docs = {}

    def why_docs(cat):
        if master_dir is None:
            return None
        if cat not in docs:
            from .why_log import read
            docs[cat] = read(master_dir, cat)
        return docs[cat]

    def _row(sid):
        from sqlalchemy import text
        return session.execute(text("SELECT rel_path, path FROM samples WHERE id = :i"),
                               {"i": int(sid)}).first()

    def name_of(sid):
        got = _row(sid)
        return library_rel((got[0] or got[1]) if got else str(sid))

    placed = {}
    for cat, cd in ((man or {}).get("categories") or {}).items():
        for e in cd.get("entries", []):
            placed.setdefault(e.get("src"), f"{cat}/{e.get('out')}")

    def place_of(sid):
        """Where the master holds a sample ("KICKS/punchy/Kick 01.wav"), else None."""
        got = _row(sid)
        return placed.get(got[1]) if got else None
    def twin_of(r):
        """(name, place) of a byte-identical copy of r the master holds, or None."""
        h = getattr(r, "file_hash", None)
        if not h or not placed:
            return None
        from sqlalchemy import text
        for sid, rel, path in session.execute(text(
                "SELECT id, rel_path, path FROM samples WHERE file_hash = :h AND id != :i ORDER BY id"),
                {"h": h, "i": int(r.id)}):
            if path in placed:
                return library_rel(rel or path), placed[path]
        return None
    try:
        from ..metadata.resolve import readings
        rd = {x.sample_id: x for x in readings(session, [r.id for r in rows])}
    except Exception:                 # a database the readings can't be read from: none shown
        rd = {}
    try:
        from ..metadata.sound import current
        snd = current(session, ids=[r.id for r in rows]) if rows else {}
    except Exception:                 # no weights, or ones this code can't read: none shown
        snd = {}
    return [explain_row(r, man, store_path, mirrors, act, why_docs, name_of,
                        place_of if man is not None else None, rd.get(r.id), twin_of, snd.get(r.id))
            for r in rows]


def precedence_table() -> list[str]:
    return [f"{i + 1:2d}. {rid:<14} {desc}  [{where}]" for i, (rid, desc, where) in enumerate(PRECEDENCE)]
