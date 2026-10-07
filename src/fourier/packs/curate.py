"""Taxonomy curation: turn the analysed library into CATEGORY/family/file packs.

The validated recipe (see docs/curation.md):

  1. Candidate membership is the UNION of two independent classifiers over the
     same audio: Sononym (OneShot/Loop class + category) and Ableton Live's auto
     tags (mapped to Fourier categories). A sample is a candidate wherever
     EITHER source places it.
  2. Both opinions are recorded. Where they agree, the sample is corroborated and
     preferred as a family representative. Where they conflict, the sample is
     homed to the single category its CLAP embedding sits closest to (agreement
     outranks a lone vote; CLAP breaks ties), and the losing label is kept in the
     manifest as the second opinion. The tuned niches (gated categories, DRUMLOOPS, FX) keep
     their CLAP gates as guardrails.
  3. Dedup: byte-hash, then content-twins by (pack, filename stem, duration), then
     acoustic near-duplicates (CLAP cosine), keeping the higher-quality sample.
  4. Cluster the CLAP embeddings: PCA(full, seeded) -> KMeans -> merge clusters
     whose centroids cos > MERGE_COS.
  5. Name each family: z-scored measured feature traits plus the most DISTINCTIVE
     CLAP text-anchored phrase.
  6. Select PER_FAMILY samples by medoid sub-clustering (spread, not dupes),
     nudged toward corroborated and higher-quality samples.
  7. Export: trim/loudness pass (per kind), WAV written PCM; aif/mp3 transcoded.

Builds are deterministic given the same library, config, CLAP embeddings, and
CURATION_SEED, and each build writes a manifest.json (seed, git sha, config hash,
per-file source/output/hash + both classifiers' opinions). A --all build can run
its categories concurrently (build_all_parallel); each category is independently
seeded, so parallel output is byte-identical to a serial build.

Everything is driven by fourier.packs.curate_config.CATEGORIES.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import func, select, text

from ..db.models import Sample
from .curate_config import taxonomy_view as _tax   # config/taxonomy.yaml + an overlay's added categories
from ..places import in_library, library_rel
from ..metadata.rows import CANDIDATE_FIELDS, HOME_FIELDS, fetch, like_any, sample_select
from ..metadata import resolve as _resolve      # which source each value the build uses came from
from ..devices.exporter import canonical_stem
from ..settings import for_module as _for_module
from . import manifests
from .curate_config import NAMES, STEM_MAX, FAMILY_NAME_MAX  # noqa: F401
from .curate_config import NAME_PHRASES, NAME_EXCLUDE_WORDS, NAME_SUPPORT_FULL, PACK_HOME, FOLDER_MAX
from .curate_config import INSTRUMENT_ROOTS
from .curate_config import VOX_CHOIR_RE, VOX_CHOIR_SHARE
from .curate_config import SET_TOGETHER_SKIP, LEVEL_CATS, DRUM_LEVEL_CATS, DRUM_LEVEL_OVER_DB, LOOP_LIMIT_DB
from .curate_config import RETUNE_CATS, RETUNE_DETECT_CATS, RETUNE_MAX_SEMIS, NOTE_LEAD_CATS
from .curate_config import MONO_CATS, NEAR_MONO_SIDE_DB, FOLD_ANYWAY_RE, TEMPO_FOLD_RANGE, TEMPO_FOLDING
from .curate_config import (FX_BAND_PHRASES, TOPS_PHRASES)
from .curate_config import (GENERIC_ACOUSTIC_TAGS, ACOUSTIC_NAME_RE, ORCH_PACKS, ACOUSTIC_ABLETON_TAGS, WIND_NAME_RE, STRING_NAME_RE)
from .curate_config import (WAVE_PATH_RE, WAVE_MAX_DUR, WAVE_MIN_SAMPLES, DUR_CAP,
                            BPM_NAME_RE, BPM_NAME_GUARD)
from .curate_config import (PHRASES_CATEGORY, PHRASE_LOOP_DIR_RE, PHRASE_ONESHOT_DIR_RE,
                            PHRASE_DIR_STRONG_RE, PHRASE_NOT_RE, PHRASE_GENRE_RE, PHRASE_DUR,
                            PHRASE_ACID_RE, PHRASE_BASS_RE, PHRASE_BASS_TAGS, PHRASE_CHORDS_RE, PHRASE_CHORDS_TAGS,
                            PHRASE_ROLES, PHRASE_ROLE_PHRASES, PHRASE_INSTRUMENTS)
from .curate_config import (KEYBOARD_NAME_RE, KEYS_EXCLUDE, KEYS_CHORD_HOME, PIANO_ANCHOR, PIANO_ANTI,
                            ROUTED_MAX_SHARE)
from .naming import (  # noqa: F401  (re-exported: tests and callers use the curate names)
    NAME_CAP, CLAP_Z, ONESHOT_DIMS, DIM_WEIGHT, LOOP_DIMS, MELODIC_CATS, CATEGORY_NOUN,
    note_name as _note_name, tempo_lead as _tempo_lead, rank_traits as _rank_traits,
    naming_dims, name_families,
)
from .curate_config import CATEGORIES, NAMEABLE_EXTRA, INSTRUMENT_PACKS, INSTRUMENT_CAP
from .curate_config import (FOLDER_TARGET_FILES, FOLDER_MIN, NAME_TOPK, NAME_SUPPORT_MIN,
                            GENRE_SUPPORT_MIN, GENRE_WORDS, INSTRUMENT_WORDS, ACOUSTIC_SOURCE_MIN,
                            SIBLING_CAP, SIBLING_CAP_CATS,
                            LOOP_DIR_ONESHOT_CATS, LOOP_DIR_ONESHOT_MAX_S,
                            STAB_CHORD_FROM, SCRATCH_FROM,
                            SCRATCH_HOME, FX_SCRATCH_PHRASES,
                            DRUM_NAME_FROM, QUIET_RMS_DB, ONESHOT_RMS_CEIL_DB, FOLDER_MIN_FILES,
                            END_HOT_DB, END_FADE_MS, EDGE_HOT_DB, EDGE_FADE_MS, LIMIT_MS, ONESHOT_TAIL_RMS, SOFT_LAYER_PEAK_DB, SOFT_LAYER_GAP_DB,
                            MIRROR_ROOT_RE, FX_TRANSITION_FROM,
                            MACHINE_SHARE, DRUM_MACHINES, DRUM_MACHINE_CATS, SYNTH_MACHINES,
                            SYNTH_MACHINE_CATS)
from .curate_config import DEMO_PATH, INST_ROUTED_PACKS, CURATION_SEED
from . import why_log
from . import vendors as _vendors
from .curate_config import POOL_SURPLUS, VENDOR_MAX_SHARE, BUDGETS, FAVORED_SOURCES, NAME_OVERRIDES, NAME_OVERRIDE_SKIP, NAME_OVERRIDE_VETO, OVERRIDE_MAX_SHARE, INSTRUMENT_PACK_MAX_SHARE, ORCH_PACK_EXCLUDE, TEMPO_BAND_MIN
from .rules import (  # noqa: F401  (re-exported: verify, why, tests and callers use the curate names)
    _has_any_tag, _pack_of, _is_nameable_vendor, _as_list, _labels_for, _canonical_of,
    _classifier_votes, _is_wave, _is_fx_transition, _wave_source, _wave_band, _cymbal_band,
    _hat_band, _fx_band, _fx_drum_leak, _mallet_label, _acoustic_tagged, _loop_band,
    _acoustic_band, _is_plucked, _is_synth_tagged, _is_mallet_named, _is_acoustic_named, _is_organ_named,
    _is_wind_named, _perc_sourced, _orch_path, _acoustic_reserved, _plucked_label, _is_keys,
    _is_chord, _is_stab, _stab_named, _is_named_stab, _scratch_named, _drum_named, _drum_named_in,
    _keys_candidates, _acoustic_label, _is_ir, _is_preview, _name_reserved, _is_loop_row,
)


def embed_text(text):
    """Lazy proxy to the CLAP text encoder. Deferring the import keeps this module
    (and the pure primitives tests import from it) free of torch/transformers, so
    unit tests and non-CLAP commands don't pull the heavy deps at import time."""
    from ..analysis.clap_features import embed_text as _impl
    return _impl(text)


def load_index():
    """The cached CLAP search index (ids, embeddings) from ~/.fourier/clap_index.npz, via its
    memory-mapped .npy copies (clap_features.load_index_fast)."""
    from ..analysis.clap_features import load_index_fast
    try:
        return load_index_fast()
    except FileNotFoundError as e:
        raise RuntimeError(str(e)) from None


def index_stale(session):
    """Return (stale, indexed_count, db_count): True when the DB has more CLAP embeddings than the index."""
    try:
        from ..analysis.clap_features import load_index_fast
        idx_n = len(load_index_fast(ids_only=True)[0])
    except Exception:
        idx_n = 0
    db_n = session.execute(
        text("SELECT COUNT(*) FROM sample_features WHERE clap_embedding IS NOT NULL")
    ).scalar() or 0
    return (db_n > idx_n, idx_n, db_n)

_tunable = _for_module("curate")   # overridable: fourier/settings.py, config/tunables.yaml

MERGE_COS = _tunable("MERGE_COS", 0.80)
NAME_SHARE = _tunable("NAME_SHARE", 0.40)
DEFAULT_PER_FAMILY = _tunable("DEFAULT_PER_FAMILY", 25)
# Budget model: a category's file BUDGET (curate_config.BUDGETS) is distributed
# across its families proportional to cluster size, each folder clamped to
# [MIN_PER_FAMILY, MAX_PER_FAMILY]. So fewer families => fuller folders and the
# category total tracks the budget regardless of how the clustering shook out.
MIN_PER_FAMILY = _tunable("MIN_PER_FAMILY", 6)
MAX_PER_FAMILY = _tunable("MAX_PER_FAMILY", 120)      # with FOLDER_MAX: few, full folders
                          # (under the Digitakt II 128-files-a-folder browsing convention)
# A category gets at least ceil(FAMILY_HEADROOM x budget / ceiling) families (up to its
# kmax). With fewer, every folder sits at the ceiling and cluster size stops mattering:
# a small family and one many times its size would get the same number of kicks. More,
# smaller families keep folders browsable on the hardware and let the budget follow cluster size.
FAMILY_HEADROOM = _tunable("FAMILY_HEADROOM", 1.5)


def _family_kmin(budget, ceil, kmin, kmax, n=None, floor=MIN_PER_FAMILY,
                 headroom=FAMILY_HEADROOM):
    """The minimum cluster count for a category (see FAMILY_HEADROOM). The extra
    families are only asked for while each could still fill its floor (n // floor)."""
    need = int(np.ceil(headroom * budget / max(ceil, 1)))
    if n is not None:
        need = min(need, n // max(floor, 1))
    return int(min(kmax, max(kmin, need)))
DEFAULT_MODEL = _tunable("DEFAULT_MODEL", "qwen3.5:9b")

# --- Selection / export tuning (deterministic; per-category overridable via cfg) ---
QUAL_WEIGHT = _tunable("QUAL_WEIGHT", 0.06)        # medoid nudge toward higher measured quality (breaks near-ties)
FAVOR_WEIGHT = _tunable("FAVOR_WEIGHT", 0.10)       # medoid nudge toward favored sources (curate_config.FAVORED_SOURCES)
NEAR_DUP_COS = _tunable("NEAR_DUP_COS", 0.985)      # CLAP cosine above which two samples are treated as near-identical
TRIM_FLOOR_DB = _tunable("TRIM_FLOOR_DB", -50.0)     # silence floor (dBFS below peak) for edge trimming
TRIM_PAD_MS = _tunable("TRIM_PAD_MS", 3.0)         # lead-in kept before the first non-silent sample (avoids clicks)
PEAK_CEILING_DB = _tunable("PEAK_CEILING_DB", -1.0)    # one-shot / instrument peak-normalization target (dBFS)
LOOP_RMS_DB = _tunable("LOOP_RMS_DB", -20.0)       # loop RMS target (dBFS), plain gain under the -1 dBFS peak ceiling.
                          # A target most loops can reach under the peak cap, so swapping
                          # breaks doesn't drop the level (LOOP_LIMIT_DB helps the rest)
MAX_GAIN_DB = _tunable("MAX_GAIN_DB", 18.0)        # never amplify a quiet sample by more than this
DC_FIX = _tunable("DC_FIX", 0.015)            # remove DC offset when |mean| exceeds this share of the peak
                          # (under verify's 2% so a file can't hover at the line)
DC_HPF_HZ = _tunable("DC_HPF_HZ", 5.0)           # ...with a zero-phase high-pass this low (keeps an 808's sub)
INSTR_TAIL_FLOOR_DB = _tunable("INSTR_TAIL_FLOOR_DB", -60.0)   # instrument release tails are trimmed only below this (of the envelope's max)
TAIL_PEAK_FLOOR_DB = _tunable("TAIL_PEAK_FLOOR_DB", -50.0)    # ...and a tail is never kept under this, relative to the sample peak
TAIL_FADE_MS = _tunable("TAIL_FADE_MS", 5.0)            # ...with a short fade (the cut is at -60 dB: 5 ms is click-free)

# Pitch windows (MIDI note) for pitched categories: a confidently-detected note
# outside the window is dropped. Very high notes read as piercing ~kHz whistles
# (a B7 = 107 triangle is ~3950 Hz), and sub-bass shouldn't sit in lead territory.
# Applies to MELODIC_CATS + instrument kinds; per-category overridable via cfg
# note_min / note_max. Only fires when base_note confidence > NOTE_CONF.
NOTE_CONF = _tunable("NOTE_CONF", 0.4)
NOTE_WINDOW_DEFAULT = _tunable("NOTE_WINDOW_DEFAULT", (24, 96))    # C1..C7
NOTE_WINDOW = _tunable("NOTE_WINDOW", {
    "SUB": (12, 64),      # C0..E4 -- it's bass, not a lead
    "SYNTH": (24, 96),    # C1..C7
    "STABS": (24, 96),
    "PADS": (24, 93),     # C1..A6
    "VOX": (36, 96),      # C2..C7
    "PIANO": (21, 100),   # A0..E7
    "ACOUSTIC": (24, 100),
})

# Mislabeled-loop guard: a loop Sononym mis-classed as OneShot leaks into a
# percussion hit folder and breaks slot-locked play (p-lock/LFO per step). Ableton
# tags the structural axis directly (Loop / Drum Loop), and onset count corroborates.
# Tight-transient drums drop on either signal; hand percussion (resonant single-hit
# articulations that Ableton sometimes mis-tags Loop) require BOTH. Per-category
# overridable via cfg loop_guard ("tag_or_onset" | "tag_and_onset" | False).
ABLETON_LOOP_TAGS = _tunable("ABLETON_LOOP_TAGS", ("Loop", "Drum Loop"))
LOOP_GUARD_ONSET = _tunable("LOOP_GUARD_ONSET", 7)
LOOP_GUARD_TAG_OR_ONSET = _tunable("LOOP_GUARD_TAG_OR_ONSET", {"KICKS", "SNARES", "HATS", "CLAPS"})
LOOP_GUARD_TAG_AND_ONSET = _tunable("LOOP_GUARD_TAG_AND_ONSET", {"TOMS", "PERC", "CYMBALS"})

# A DRUMLOOPS member must show Ableton evidence of being a DRUM loop. Sononym's
# generic "Loop" class also covers bassline/lead/vocal/FX loops, and har_max + the
# CLAP gate miss the low-harmonicity ones (a sweep or a repeating synth riff), so a
# loop that carries no drum tag is dropped when cfg sets drum_tag_required.
DRUM_LOOP_TAGS = _tunable("DRUM_LOOP_TAGS", ("Drum Loop", "Kick", "Snare Hit", "Closed Hihat", "Open Hihat", "Clap",
                  "Rim", "Ride", "Crash", "Low Tom", "Mid Tom", "High Tom", "Conga",
                  "Cowbell", "Shaker", "Tambourine", "Bongo", "Timbale", "Pedal Hihat",
                  "Splash", "Snare Articulation", "Misc Percussion"))


def _has_drum_tag(ableton_tags):
    return any(t in DRUM_LOOP_TAGS for t in ableton_tags)


# the drums a path label names (path.yaml): a file whose own name or folder names one of
# these is drums, for a drum-loop category's vote (_drum_loop_evidence)
_DRUM_PATH_LABELS = ("kick", "snare", "clap", "snap", "hat", "tom", "perc.", "cymbal.")


def _drum_loop_evidence(r, ab_tags, cfg, audio=True) -> bool:
    """Without Sononym (the path and audio providers classify): whether a loop shows drums, so
    a drum-loop category (drum_tag_required) may take its vote. A drum tag (Live's, or the
    words standing in for it), a drum, break, beat or groove word in the file's own name or
    folder (_name_tags, path.yaml's drum labels), a drum word in its pack's name for a loop
    by its own name or folder (_pack_drum_named), or (`audio`) percussive audio: harmonicity
    at most the category's har_max with onsets at the loop rate (shadow.ONSET_LOOP_MIN_HZ).
    A loop by its audio alone (a sustained riser, pad or bassline has onsets and no silent
    gap) shows none."""
    from ..metadata.shadow import ONSET_LOOP_MIN_HZ, near_labels
    rel = r.rel_path or r.path or ""
    if _has_drum_tag(ab_tags) or _has_drum_tag(_name_tags(rel)):
        return True
    if any(lab.startswith(_DRUM_PATH_LABELS) for lab in near_labels(rel)):
        return True
    if _pack_drum_named(rel):
        return True
    har, ons = getattr(r, "harmonicity", None), getattr(r, "onset_rate_hz", None)
    if audio and har is not None and ons is not None and _anonymous_loop(rel) \
            and har <= ANON_LOOP_HAR_MAX and ons >= ANON_LOOP_ONSET_MIN_HZ:
        return True             # a bare "Loops/85/loop_17" with busy, mostly percussive audio
    return (audio and har is not None and cfg.get("har_max") is not None and har <= cfg["har_max"]
            and (ons is None or ons >= ONSET_LOOP_MIN_HZ))


# Without Sononym: a loop whose name and folders say nothing about its sound ("Loops/85/loop_17",
# "Loops/loop 3 90bpm") is a drum loop by its audio when its onsets are busy (at least
# ANON_LOOP_ONSET_MIN_HZ) and its harmonicity at most ANON_LOOP_HAR_MAX: lo-fi drums (filtered,
# under a chord) read more tonal than a drum-loop category's har_max, and nothing else in the
# path speaks for them. A musical loop it takes by mistake fails the drum loops' CLAP gate and
# falls back to PHRASES by sound (_clap_rehome).
ANON_LOOP_HAR_MAX = _tunable("ANON_LOOP_HAR_MAX", 0.80)
ANON_LOOP_ONSET_MIN_HZ = _tunable("ANON_LOOP_ONSET_MIN_HZ", 2.0)
_ANON_STEM_RE = re.compile(r"\d+|bpm|loops?|takes?|[\s_\-.()#]+", re.I)
_ANON_DIR_RE = re.compile(r"^\s*(?:\d{2,3}(?:\s*bpm)?|loops?|wavs?|samples?|audio)\s*$", re.I)


def _anonymous_loop(rel) -> bool:
    """A loop whose file name holds nothing but a number, a tempo and a loop word, in a folder
    named only by a tempo or a loop word, with no word anywhere in its path naming a sound
    (path.yaml)."""
    from ..metadata.shadow import path_labels
    parts = [p for p in (rel or "").split("/") if p]
    if len(parts) < 2 or not _ANON_DIR_RE.match(parts[-2]):
        return False
    if _ANON_STEM_RE.sub("", os.path.splitext(parts[-1])[0]):
        return False
    return not {lab for lab in path_labels(rel) if not lab.startswith("class.")}


# drum words in a pack's name ("Drum Hits", "Percussion Loops", "Breaks"), genre names removed
# ("Drum & Bass"): what kind of loop a pack's loops are (_pack_drum_named)
_PACK_DRUM_RE = re.compile(r"(?<![a-z])(?:drums?|percussion|breaks?|breakbeats?|beats?)(?![a-z])", re.I)


def _pack_drum_named(rel) -> bool:
    """A loop by its own name or folder (shadow.shape_labels: "Loops/House 120 01") whose
    vendor or pack folders name drums ("Drum Hits"): a pack's name can say what kind of loop
    its loops are, never that a file is a loop (as _name_tags), so a riser or bassline in a
    "Breaks" pack shows nothing by it. Without Sononym only."""
    from ..metadata.shadow import shape_labels, shape_scope
    rel = rel or ""
    if "class.loop" not in shape_labels(rel):
        return False
    _name, folder = shape_scope(rel)
    folders = [p for p in rel.split("/") if p][:-1]
    above = folders[:folders.index(folder)] if folder in folders else []
    return bool(above) and bool(_PACK_DRUM_RE.search(PHRASE_GENRE_RE.sub(" ", " ".join(above))))


def _drum_loop_categories():
    return [c for c, cfg in CATEGORIES.items() if cfg["kind"] == "loop" and cfg.get("drum_tag_required")]


def _providers_drum_loop(r, ab_tags) -> bool:
    """Without Sononym: whether the built-in providers call this file a drum loop, so the
    phrase rule leaves it to the drum loops (_is_phrase). The classifier calls it a loop and
    it shows drums (_drum_loop_evidence: a drum tag or word in its own name or folder, a drum
    word in its pack's name, or percussive audio, unless its own name or folder names an
    instrument or FX sound: a "Bass Loops/Bassline 120bpm" with plucky onsets is a phrase)."""
    from ..metadata.shadow import names_a_sound
    if not _row_is_loop(r):
        return False
    audio = not names_a_sound(r.rel_path or r.path or "")
    return any(_drum_loop_evidence(r, ab_tags, CATEGORIES[c], audio=audio)
               for c in _drum_loop_categories())


_KEYS_LABEL = "keys"          # the path provider's label for keys named as such (path.yaml)


def _keys_labeled(r) -> bool:
    """Without Sononym: the built-in providers name this one-shot keys (rhodes, wurli,
    e-piano, organ, piano; path.yaml's "keys") and no other sound: PIANO's (an instrument
    category, which takes it by that name), not a vote for any other category."""
    canon = _canonical_of(r)
    return _KEYS_LABEL in canon and not {c for c in canon if c != _KEYS_LABEL and not c.startswith("class.")}


def _fallback_loop_votes(cats, r, ab_tags) -> set:
    """The classifier's votes without Sononym: a loop's vote for a drum-loop category
    (kind loop, drum_tag_required) stands only with evidence of drums (_drum_loop_evidence):
    class.loop alone isn't a drum loop."""
    return {c for c in cats if not (CATEGORIES[c]["kind"] == "loop" and CATEGORIES[c].get("drum_tag_required"))
            or _drum_loop_evidence(r, ab_tags, CATEGORIES[c])}


# ---------------------------------------------------------------------------
# Cross-family home veto: trust Ableton's front-end classification. A file whose
# Ableton tags point only to one sonic family should not be homed into a category
# of a different family just because CLAP finds it acoustically close (e.g. a
# "Synth Bass" landing in KICKS because a bass reads as kick-ish). The veto only
# runs for categories where Ableton's tag vocabulary is reliable; the coarse ones
# (gated categories, SCRATCHES, MALLETS, BLIPS -- which Ableton lumps under Solo Voice / Field &
# Foley / Sound FX) are exempt so their finer Fourier home is never overridden.
_AB_FAMILY = {}
for _t in ("Kick", "Snare Hit", "Closed Hihat", "Open Hihat", "Pedal Hihat", "Clap", "Rim",
           "Ride", "Crash", "Splash", "Low Tom", "Mid Tom", "High Tom", "Conga", "Bongo",
           "Cowbell", "Shaker", "Tambourine", "Timbale", "Wood", "Woodblock", "Cabasa",
           "Guiro", "Djembe", "Tabla", "Gong", "Misc Percussion", "Snare Articulation",
           "Misc Cymbal", "Drum Loop"):
    _AB_FAMILY[_t] = "drum"
for _t in ("Synth Bass", "808 Bass", "Electric Bass", "Upright Bass", "Acoustic Bass",
           "Double Bass", "Lead", "Synth Keys", "Synth Plucked", "Misc Plucked", "Pad",
           "Atmosphere", "Bell", "Bell Chromatic", "Chime", "Piano", "Electric Piano",
           "Organ", "Clav", "Synth Strings", "Synth Brass", "Synth Mallets", "Synth Woodwind",
           "Cello", "Violin", "Viola", "Flute", "Sax", "Trumpet", "Trombone", "Tuba",
           "French Horn", "Clarinet", "Oboe", "Bassoon", "Xylophone", "Vibraphone", "Marimba",
           "Glockenspiel", "Harp", "Sitar", "Electric Guitar", "Acoustic Guitar",
           "Classical Guitar", "Misc Keys", "Misc Mallets", "Misc Strings", "Misc Woodwind",
           "String Ensemble", "Brass Ensemble"):
    _AB_FAMILY[_t] = "tonal"
for _t in ("Solo Voice", "Synth Voice", "Choir", "Speech"):
    _AB_FAMILY[_t] = "vocal"
for _t in ("Sound FX", "Sweep", "Impact", "Noise", "Field & Foley"):
    _AB_FAMILY[_t] = "fx"

# Fourier categories whose Ableton family is trustworthy enough to veto a
# conflicting home. Where the two disagree on the family, Ableton is usually the one
# that's right (Sononym can file water-slaps/hammers/synth-FX as snares/subs, 808
# bass hits as kicks, and cymbal crashes under "Tone Stabs & Orch. Hits" ->
# STABS), so a clear foreign Ableton family overrides the CLAP tiebreak. STABS and
# BLIPS are guarded as tonal (keeping drum and vocal hits out).
# Exempt (never vetoed): the categories Ableton's vocabulary handles coarsely --
# gated categories, SCRATCHES, MALLETS (lumped as Solo Voice / Field & Foley / Sound FX) -- plus FX
# itself, the intended catch-all sink.
CATEGORY_AB_FAMILY = _tunable("CATEGORY_AB_FAMILY", {
    "KICKS": "drum", "SNARES": "drum", "HATS": "drum", "CLAPS": "drum",
    "TOMS": "drum", "PERC": "drum", "CYMBALS": "drum",
    "SUB": "tonal", "SYNTH": "tonal", "PADS": "tonal", "STABS": "tonal", "BLIPS": "tonal",
    "VOX": "vocal",
})


def _family_veto(cand, ableton_tags):
    """Drop candidate categories whose (trusted) family conflicts with what Ableton
    says the file is. Never returns empty -- if the veto would remove every
    candidate, the original set is kept (Ableton is a tiebreaker, not a nuke)."""
    fams = {_AB_FAMILY[t] for t in ableton_tags if t in _AB_FAMILY}
    if not fams:
        return cand
    kept = {c for c in cand if CATEGORY_AB_FAMILY.get(c) is None or CATEGORY_AB_FAMILY[c] in fams}
    return kept or cand


_EDITION_RE = re.compile(r"[\s_-]*(?:\(?\d{2}\s?-?bits?\)?|\(?\d{2}(?:\.\d)?\s?khz\)?|mono|stereo|wav|aiff?)(?![a-z])",
                         re.I)


def _content_key(pack: str, filename: str, dur):
    """(pack minus edition words, name stem, duration): two editions of one pack
    (a pack and its "- 16bit" copy, or its stereo and mono WAV editions) are one
    sound, and one of them is enough."""
    stem = re.sub(r"[ _-]+$", "", os.path.splitext(filename or "")[0].lower())
    return (_EDITION_RE.sub("", pack or "").strip(" -_()").lower(), stem,
            round(dur, 2) if dur is not None else None)


def _g(v, dv):
    return dv if v is None else v


def _limit_threads():
    """Pin BLAS/OpenMP to one thread so KMeans and PCA are reproducible on-machine
    (their multi-threaded float reductions are otherwise run-to-run nondeterministic
    even with a fixed random_state)."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except Exception:
        return nullcontext()


def _note_window(category, cfg):
    """(note_min, note_max) MIDI window for this category, or (None, None) if it's
    not pitch-gated. Explicit cfg note_min/note_max win; otherwise pitched categories
    (MELODIC_CATS + instrument kind) get NOTE_WINDOW[cat] or NOTE_WINDOW_DEFAULT."""
    if cfg.get("note_min") is not None or cfg.get("note_max") is not None:
        return cfg.get("note_min"), cfg.get("note_max")
    if category in MELODIC_CATS or cfg.get("kind") == "instrument":
        return NOTE_WINDOW.get(category, NOTE_WINDOW_DEFAULT)
    return None, None


# Sample-chain guard: a "one-shot" that is really several events laid end to end (a
# velocity ladder, a note ladder, round robins) -- see analysis/events.py and
# `fourier tools analyze --only events`. strict: two or more loud, silence-separated events that are
# not a decaying echo. regular: three or more on a near-fixed grid (spacing CV <
# CHAIN_REGULAR_CV), so a performed phrase (a rap verse, an FX sequence) survives but a
# chain doesn't. Categories where multi-event content IS the sound (loops, field recordings,
# scratches, blips/morse) are off. Per-category overridable via cfg chain_guard.
CHAIN_GUARD_STRICT = _tunable("CHAIN_GUARD_STRICT", {"KICKS", "SNARES", "HATS", "CLAPS", "TOMS", "PERC", "CYMBALS",
                      "SUB", "SYNTH", "STABS", "PADS", "PIANO", "ACOUSTIC"})
CHAIN_GUARD_REGULAR = _tunable("CHAIN_GUARD_REGULAR", {"VOX", "FX"})
CHAIN_REGULAR_CV = _tunable("CHAIN_REGULAR_CV", 0.2)


def _chain_guard_mode(category, cfg):
    """"strict" | "regular" | None for this category (cfg chain_guard overrides)."""
    m = cfg.get("chain_guard", "default")
    if m != "default":
        return m or None
    if category in CHAIN_GUARD_STRICT:
        return "strict"
    if category in CHAIN_GUARD_REGULAR:
        return "regular"
    return None


def _is_sample_chain(mode, n_events, regularity, echo):
    """True if the event profile says this file is several sounds, not one. Files not
    yet profiled (n_events None) pass."""
    if not mode or n_events is None or echo:
        return False
    if mode == "strict":
        return n_events >= 2
    if mode == "regular":
        return n_events >= 3 and regularity is not None and regularity < CHAIN_REGULAR_CV
    return False


def _loop_guard_mode(category, cfg):
    """Return the mislabeled-loop guard mode for this category, or None if off.
    cfg loop_guard overrides: a mode string, or False/None to disable."""
    m = cfg.get("loop_guard", "default")
    if m != "default":
        return m or None
    if category in LOOP_GUARD_TAG_OR_ONSET:
        return "tag_or_onset"
    if category in LOOP_GUARD_TAG_AND_ONSET:
        return "tag_and_onset"
    return None


def _is_mislabeled_loop(mode, ableton_tags, onset_count, bpm_reliable):
    """True if this hit-folder candidate is really a loop. tag_or_onset: an Ableton
    Loop/Drum Loop tag, OR many onsets with a reliable tempo. tag_and_onset: an
    Ableton loop tag AND many onsets (so resonant single-hit perc articulations that
    Ableton mis-tags Loop survive)."""
    if not mode:
        return False
    is_loop_tag = any(t in ABLETON_LOOP_TAGS for t in ableton_tags)
    hot = onset_count >= LOOP_GUARD_ONSET
    if mode == "tag_or_onset":
        return is_loop_tag or (hot and bool(bpm_reliable))
    if mode == "tag_and_onset":
        return is_loop_tag and hot
    return False


def _allocate_budget(sizes, budget, floor, ceil):
    """Distribute a category's file `budget` across families of the given cluster
    `sizes`, proportional to size and clamped to [floor, min(ceil, size)]. Returns a
    per-family count. If too few families to spend the budget (all hit the ceiling),
    the total lands under budget rather than padding folders with near-dupes."""
    k = len(sizes)
    if k == 0:
        return []
    tot = sum(sizes) or 1
    hi = [min(ceil, s) for s in sizes]
    lo = [min(floor, s) for s in sizes]
    alloc = [int(min(hi[i], max(lo[i], round(budget * sizes[i] / tot)))) for i in range(k)]
    guard = 0
    while sum(alloc) < budget and guard < 100000:
        cand = [i for i in range(k) if alloc[i] < hi[i]]
        if not cand:
            break
        i = max(cand, key=lambda i: (sizes[i] - alloc[i], sizes[i], -i))
        alloc[i] += 1
        guard += 1
    while sum(alloc) > budget and guard < 200000:
        cand = [i for i in range(k) if alloc[i] > lo[i]]
        if not cand:
            break
        i = max(cand, key=lambda i: (alloc[i], sizes[i], -i))
        alloc[i] -= 1
        guard += 1
    return alloc


def _allocate_banded(sizes, keys, shares, budget, floor, ceil, prefer=None):
    """_allocate_budget per band: each band gets its fixed share of the budget (shares
    renormalized over the bands present), spread over its families by size. What a band
    can't hold (its pool is small) goes to families with room anywhere: first to families
    under `prefer` (the previous build's files in them), then largest gap first. Without
    `prefer` a one-file change in a pool's size tipped a near-tie between two bands, and
    ACOUSTIC swapped a file every build."""
    present = sorted({k for k in keys}, key=str)
    tot = sum(shares.get(k, 0.0) for k in present) or 1.0
    alloc = [0] * len(sizes)
    for k in present:
        idx = [i for i, kk in enumerate(keys) if kk == k]
        sub = _allocate_budget([sizes[i] for i in idx], int(round(budget * shares.get(k, 0.0) / tot)),
                               floor, ceil)
        for i, a in zip(idx, sub):
            alloc[i] = a
    hi = [min(ceil, s) for s in sizes]
    guard = 0
    while sum(alloc) < budget and guard < 100000:
        cand = [i for i in range(len(sizes)) if alloc[i] < hi[i]]
        if not cand:
            break
        i = max(cand, key=lambda i: (bool(prefer) and alloc[i] < prefer[i], sizes[i] - alloc[i], sizes[i], -i))
        alloc[i] += 1
        guard += 1
    lo = [min(floor, s) for s in sizes]
    while sum(alloc) > budget and guard < 200000:
        cand = [i for i in range(len(sizes)) if alloc[i] > lo[i]]
        if not cand:
            break
        i = max(cand, key=lambda i: (bool(prefer) and alloc[i] > prefer[i], alloc[i], sizes[i], -i))
        alloc[i] -= 1
        guard += 1
    return alloc


def _quality(r):
    """A defect-avoidance score in [0, 1] from measured features: clipping, an
    audible DC offset, and near-silence pull it down. Used only to break near-ties
    in selection and to keep the better of two near-duplicates -- it never overrides
    CLAP spread."""
    q = 1.0
    if r.get("clip"):
        q -= 0.5
    dc = r.get("dc") or 0.0
    if dc > 0.02:
        q -= min(0.35, (dc - 0.02) * 12.0)
    rms = r.get("rms")
    if rms is not None and rms < 0.004:   # ~ -48 dBFS mean: near-silent / dropout
        q -= 0.35
    return max(0.0, q)


# manifest.json format (manifests.FORMAT): 2 added each entry's canonical classifier labels
# ("labels"), 3 the library-relative source paths
_MANIFEST_VERSION = manifests.FORMAT


# ---------------------------------------------------------------------------
# WAVES: single-cycle waveforms / wavetable frames -- see curate_config WAVE_*
# ---------------------------------------------------------------------------
WAVES_CATEGORY = _tunable("WAVES_CATEGORY", _tax.home("waves"))


def _not_here_as_wave(r, rel, category, wave_hashes=None):
    """True if the row can't be in this category as far as waves go: a wave anywhere but
    WAVES, a non-wave in WAVES, or (outside WAVES) a byte-identical copy of a wave that
    sits outside a wave folder, which would put the same audio in two categories."""
    wave = _is_wave(rel, r.duration_s)
    if category == WAVES_CATEGORY:
        return not wave
    return wave or bool(wave_hashes and r.file_hash and r.file_hash in wave_hashes)


_WAVE_HASHES = {}


def _wave_hashes(session):
    """file_hash of every wave in the library (cached per session)."""
    if session is None:
        return set()
    key = id(session)
    if key not in _WAVE_HASHES:
        from .curate_config import WAVE_PATH_WORDS
        like = ["%wavetable%", "%wave table%", "%single cycle%", "%single-cycle%", "%single_cycle%",
                "%singlecycle%"] + [f"%{w.lower()}%" for w in WAVE_PATH_WORDS]   # a superset; _is_wave decides
        cond = " OR ".join(f"lower(samples.rel_path) LIKE :w{i}" for i in range(len(like)))
        q = (select(Sample.rel_path, Sample.path, Sample.duration_s, Sample.file_hash)
             .where(Sample.file_hash.isnot(None))
             .where(text(f"({cond})").bindparams(**{f"w{i}": v for i, v in enumerate(like)})))
        q = _not_missing(session, q)
        _WAVE_HASHES.clear()
        _WAVE_HASHES[key] = {h for rel, pth, dur, h in session.execute(q).all()
                             if _is_wave(rel or library_rel(pth or ""), dur)}
    return _WAVE_HASHES[key]


_MIRRORS = {}
KIT_CATS = _tunable("KIT_CATS", _tax.with_role("kit"))


def _not_missing(session, q):
    """q without the samples a library walk marked missing (metadata/rows.py: only when any
    are marked, so a database no walk marked is queried as before), and without other
    libraries' samples (rows.outside_library: only when the database holds any)."""
    from ..metadata.rows import _id_list, missing_marked, outside_library
    other = outside_library(session)
    if other:
        q = q.where(text(f"samples.id NOT IN ({_id_list(other)})"))
    if not missing_marked(session):
        return q
    from ..db.models import MissingFile
    return q.where(~select(MissingFile.sample_id).where(
        MissingFile.sample_id == Sample.id, MissingFile.path == Sample.path).exists())


def _mirror_dups(session):
    """Paths under a mirror folder (MIRROR_ROOT_RE: a top-level folder of copies) whose bytes also
    live outside the mirrors: copies, left out of curation. Cached per session."""
    if session is None:
        return set()
    key = id(session)
    if key not in _MIRRORS:
        rows = session.execute(_not_missing(session, select(Sample.rel_path, Sample.path, Sample.file_hash)
                                            .where(Sample.file_hash.isnot(None)))).all()
        mir, outside = [], set()
        for rel, pth, h in rows:
            rel = rel or library_rel(pth or "")
            if MIRROR_ROOT_RE.search(rel):
                mir.append((pth, h))
            else:
                outside.add(h)
        _MIRRORS.clear()
        _MIRRORS[key] = {pth for pth, h in mir if h in outside}
    return _MIRRORS[key]


# ---------------------------------------------------------------------------
# Bands (curate_config: HAT_*, FX_*, LOOP_*): sub-groups clustered on their own
# ---------------------------------------------------------------------------
BAND_ORDER = _tunable("BAND_ORDER", {**_tax.bands(), "phrase": PHRASE_ROLES})


def _band_of(btype, filename, duration=None, sample_rate=None, son_labels=(), ab_tags=(),
             chroma=None, harmonicity=None, min_chroma=1.6, pack=None, rel=None):
    """Band label for a sample under a category's band rule (None if it has none)."""
    if btype == "wave":
        return _wave_band(duration, sample_rate)
    if btype == "chord":
        return "chord" if _is_chord(filename, chroma, min_chroma, harmonicity) else "note"
    if btype == "hat":
        return _hat_band(filename, duration)
    if btype == "cymbal":
        return _cymbal_band(filename, son_labels, ab_tags)
    if btype == "fx":
        return _fx_band(son_labels, ab_tags, filename, rel)
    if btype == "loop":
        return _loop_band(filename, ab_tags, rel)
    if btype == "acoustic":
        return _acoustic_band(filename, ab_tags, pack, rel)
    if btype == "phrase":
        return _phrase_role(filename, rel, ab_tags, son_labels)
    return None


_HAT_OPEN_WORDS = {"open", "sizzly", "washy", "half", "long"}
_HAT_CLOSED_WORDS = {"closed", "pedal", "tight", "short", "ticky"}


def _band_phrase_ok(btype, band, phrase):
    """May a family in this band be named by this phrase? (a closed-hat folder isn't
    "open hi-hat", a chord folder isn't a "note", an impact isn't "ocean waves")"""
    w = set(re.split(r"[^a-z0-9]+", phrase.lower()))
    if btype == "chord":
        return not (w & ({"note"} if band == "chord" else {"chord", "chords"}))
    if btype == "hat":
        return not (w & (_HAT_CLOSED_WORDS if band == "open" else _HAT_OPEN_WORDS))
    if btype == "cymbal":                   # a ride folder isn't "crash cymbal", and back
        other = {"ride": {"crash", "splash", "china", "wash"}, "crash": {"ride", "bell"},
                 "cymbal": {"ride", "crash", "splash", "china"}}[band] if band in ("ride", "crash", "cymbal") else set()
        return not (w & other)
    if btype == "fx" and band == "scratch":
        return phrase in FX_SCRATCH_PHRASES
    if btype == "fx" and band in FX_BAND_PHRASES:
        return phrase in FX_BAND_PHRASES[band]
    if btype == "loop":                     # tops named only by tops phrases, breaks never
        return band != "classic" and (phrase in TOPS_PHRASES) == (band == "tops")
    if btype == "phrase" and band in PHRASE_ROLE_PHRASES:   # a bassline folder isn't "string swell"
        return phrase in PHRASE_ROLE_PHRASES[band]
    return True


_NAME_WORD = {"pizz": "pizzicato", "sax": "sax", "horns": "horn", "vibes": "vibraphone"}


def _acoustic_name_label(name):
    """Instrument word from an ACOUSTIC file's name: plucked string, tuned percussion,
    then a wind or string word ("Bowed Glass" -> "bowed")."""
    if _is_plucked(name):
        return _plucked_label(name)
    if _is_mallet_named(name):
        return _mallet_label(name)
    for rx in (WIND_NAME_RE, STRING_NAME_RE):
        m = rx.search(name or "")
        if m:
            w = re.sub(r"[^a-z0-9]+", "-", m.group(0).lower()).strip("-")
            return _NAME_WORD.get(w, w)
    return None


def _keys_anchors():
    """(piano anchor, anti anchor): the same CLAP test PIANO gates with."""
    out = []
    for ph in (PIANO_ANCHOR, PIANO_ANTI):
        T = np.stack([embed_text(p) for p in ph])
        a = T.mean(0)
        out.append(a / (np.linalg.norm(a) + 1e-9))
    return tuple(out)


KEYS_PHRASE_WORDS = _tunable("KEYS_PHRASE_WORDS", {"piano", "rhodes", "keys", "key", "wurlitzer", "clavinet", "clav"})
KEYS_FAMILY_SHARE = _tunable("KEYS_FAMILY_SHARE", 0.5)


def _keys_phrase_cols(phrases):
    """Boolean per phrase: does it name a keyboard instrument?"""
    return np.array([bool(set(re.split(r"[^a-z0-9]+", p.lower())) & KEYS_PHRASE_WORDS)
                     for p in phrases], dtype=bool)


def _keys_phrase_mask(keys_shares, phrases):
    """Boolean clusters x phrases mask of phrases a family may NOT take: a family whose
    files are mostly keys (by name) is named by a keys phrase when the vocabulary has
    one ("pizzicato stab" on a folder of piano chords is the failure this prevents)."""
    kp = _keys_phrase_cols(phrases)
    mask = np.zeros((len(keys_shares), len(phrases)), bool)
    if kp.any():
        for j, sh in enumerate(keys_shares):
            if sh >= KEYS_FAMILY_SHARE:
                mask[j] = ~kp
    return mask


_MACHINE_RX = [(lab, re.compile(p, re.I)) for lab, p in DRUM_MACHINES]
_SYNTH_MACHINE_RX = [(lab, re.compile(p, re.I)) for lab, p in SYNTH_MACHINES]


def _machine_label(path):
    """The drum machine a sample's path names (DRUM_MACHINES), else None."""
    rel = library_rel(path or "")
    for text_ in (os.path.basename(rel), rel):      # the filename names the machine first
        for lab, rx in _MACHINE_RX:
            if rx.search(text_):
                return lab
    return None


def _unify_twin_homes(homes, hash_of):
    """Give byte-identical samples (same file_hash) the home of the lowest id among
    them -- PHRASES if any copy is a phrase, so a phrase's twin can't sit in a one-shot
    folder. Mutates homes; returns how many homes moved."""
    by_hash = defaultdict(list)
    for sid in homes:
        h = hash_of.get(sid)
        if h:
            by_hash[h].append(sid)
    moved = 0
    for ids in by_hash.values():
        if len(ids) > 1:
            home = homes[min(ids)]
            if any(homes[i] == PHRASES_CATEGORY for i in ids):
                home = PHRASES_CATEGORY
            for i in ids:
                if homes[i] != home:
                    homes[i] = home
                    moved += 1
    return moved


# ---------------------------------------------------------------------------
# Two-classifier home resolution: union candidacy, agreement-weighted, CLAP tie-break
# ---------------------------------------------------------------------------
_HOMES_CACHE = {}


def compute_homes(session, emb_n, id2row, log=print):
    """Resolve each sample to a single home category from BOTH classifiers.

    A sample is a candidate wherever Sononym (class + category) OR Ableton's auto
    tags place it. Support = 2 when both sources agree on a category, 1 for a lone
    vote. The home is the highest-support candidate; ties are broken by which
    category's CLAP phrase-anchor the sample sits closest to. Instrument packs are
    routed separately and excluded here.

    Returns (homes, support, votes):
      homes[sid]           -> chosen category
      support[(sid, cat)]  -> 1 or 2
      votes[sid]           -> {son_labels, ab_tags, son_cats, ab_cats}
    """
    from .curate_config import ABLETON_TAGS_BY_CATEGORY

    part = {cat: cfg for cat, cfg in CATEGORIES.items()
            if cfg["kind"] in ("oneshot", "loop", "gated")}
    anchors = {}
    for cat, cfg in part.items():
        T = np.stack([embed_text(p) for p in cfg["phrases"]])
        a = T.mean(0)
        anchors[cat] = a / (np.linalg.norm(a) + 1e-9)

    tag2cats = defaultdict(set)
    for cat in part:
        for t in ABLETON_TAGS_BY_CATEGORY.get(cat, []):
            tag2cats[t].add(cat)

    rows = _with_name_tags(session, fetch(session, sample_select(*HOME_FIELDS, session=session)))
    from ..metadata.providers import active
    fallback = active(session).fallback          # no Sononym: path and audio classify
    snd = _sound_labels(session)                 # without Sononym: the sound model's labels
    keys_anchors = _keys_anchors()
    piano_min_chroma = CATEGORIES.get("PIANO", {}).get("min_chroma", 1.6)
    n_keys = n_keys_chords = 0

    homes, support, votes = {}, {}, {}
    n_multi = n_conflict = n_phrases = n_chord_stabs = n_named_drums = 0
    n_unrec = n_clap = n_rehomed = n_sound = n_sound_over = 0
    phrases_on = PHRASES_CATEGORY in CATEGORIES
    from .ratings import misfiled_map
    misfiled = _misfiled_index(misfiled_map(), emb_n, id2row, pid={r.path: r.id for r in rows})
    mirrors = _mirror_dups(session)
    for r in rows:
        if r.id not in id2row or r.path in mirrors:
            continue
        pack = _pack_of(r.rel_path, r.path)
        if INST_ROUTED_PACKS.search(pack) or _is_ir(r.rel_path, r.path) or _is_preview(r.rel_path, r.path):
            continue
        # a musical phrase homes in PHRASES ahead of every one-shot vote (and of the acoustic
        # reservation: a guitar riff is a phrase), unless it was rated Misfiled there
        if phrases_on and _is_phrase(r, fallback) and PHRASES_CATEGORY not in misfiled.categories(
                r.path, emb_n[id2row[r.id]]) and _name_reserved(r.filename) in (None, PHRASES_CATEGORY):
            # without Sononym: a loop that PHRASES' CLAP gate would turn away (it sounds like a
            # drum break) and a drum-loop category's gate clearly takes is that one's
            _rh = _clap_rehome(emb_n[id2row[r.id]], anchors, misfiled.categories(
                r.path, emb_n[id2row[r.id]])) if fallback and _row_is_loop(r) else None
            if _rh:
                homes[r.id] = _rh
                support[(r.id, _rh)] = 1
                votes[r.id] = dict(son_labels=_as_list(r.categories), ab_tags=_as_list(r.ableton_tags),
                                   son_cats=[], ab_cats=[], rehomed=_rh)
                n_rehomed += 1
                continue
            homes[r.id] = PHRASES_CATEGORY
            support[(r.id, PHRASES_CATEGORY)] = 2
            votes[r.id] = dict(son_labels=_as_list(r.categories), ab_tags=_as_list(r.ableton_tags),
                               son_cats=[PHRASES_CATEGORY], ab_cats=[PHRASES_CATEGORY])
            n_phrases += 1
            continue
        son_labels = _as_list(r.categories)
        classes = _as_list(r.classes)
        ab_tags = _as_list(r.ableton_tags)
        # acoustic orchestral instruments (strings/winds/brass/classical guitar) are
        # reserved for ACOUSTIC; keep them out of the drum/synth/pad homing entirely
        _fn = os.path.basename(r.path or "")
        if _acoustic_tagged(ab_tags, _fn, pack) and not ORCH_PACK_EXCLUDE.search(pack):
            continue
        is_oneshot = any("OneShot" in c for c in classes)
        is_loop = any("Loop" in c for c in classes)
        # plucked strings named as such are ACOUSTIC's (like the tagged ones above)
        # (and tuned percussion: xylophone, kalimba, church bell, ...)
        # (and a named wind, or anything in an orchestra folder (ORCH_PATH_RE), loop class or not:
        # a sustained tremolo reads as a loop to Sononym)
        if ((is_oneshot and not is_loop and _is_acoustic_named(_fn, ab_tags))
                or (not is_loop and _is_wind_named(_fn, ab_tags))
                or _orch_path(r.rel_path) is not None) and not ORCH_PACK_EXCLUDE.search(pack):
            continue
        # a named organ one-shot (not a stab) is PIANO's
        if not is_loop and _is_organ_named(_fn):
            continue
        # ...and so is one the built-in providers name keys and nothing else (without Sononym)
        if fallback and not is_loop and _keys_labeled(r):
            continue

        son_cats, ab_cats = _classifier_votes(part, _canonical_of(r)), set()
        if fallback:             # without Sononym, a loop votes a drum-loop category only with drums
            son_cats = _fallback_loop_votes(son_cats, r, ab_tags)
        for t in ab_tags:
            ab_cats |= tag2cats.get(t, set())

        cand = son_cats | ab_cats
        # without Sononym: where only a pack's or vendor's name (or a weak word) speaks, a
        # confident sound model decides (SOUND_MODEL_OVERRIDE_MIN) when it says otherwise or
        # picks between several candidates
        if snd and cand and not ab_cats and r.id in snd and _weak_path(r):
            _s = _sound_home(snd[r.id], is_oneshot and not is_loop, is_loop and not is_oneshot,
                             misfiled.categories(r.path, emb_n[id2row[r.id]]), SOUND_MODEL_OVERRIDE_MIN)
            if _s and (_s not in cand or len(cand) > 1) and _name_reserved(_fn) in (None, _s):
                homes[r.id] = _s
                support[(r.id, _s)] = 1
                votes[r.id] = dict(son_labels=son_labels, ab_tags=ab_tags, son_cats=sorted(son_cats),
                                   ab_cats=[], sound=[_s, round(float(snd[r.id][1]), 4)])
                n_sound_over += 1
                continue
        if not cand:
            if fallback:         # no name rule matched: the CLAP fallback may place it by sound
                n_unrec += 1
                # the sound model first, when it's confident (SOUND_MODEL_MIN); a held, tonal
                # "loop" no stated tempo fits is offered the one-shot categories (_held_loop)
                _hl = is_loop and not is_oneshot and bool(snd) and _held_loop(r)
                _s = _sound_home(snd.get(r.id), (is_oneshot and not is_loop) or _hl,
                                 is_loop and not is_oneshot and not _hl,
                                 misfiled.categories(r.path, emb_n[id2row[r.id]])) if snd else None
                if _s and _name_reserved(_fn) in (None, _s):
                    homes[r.id] = _s
                    support[(r.id, _s)] = 1
                    votes[r.id] = dict(son_labels=son_labels, ab_tags=ab_tags, son_cats=[], ab_cats=[],
                                       sound=[_s, round(float(snd[r.id][1]), 4)])
                    n_sound += 1
                    continue
                # (no shape from the name, folder or audio, but one event or a held, tonal
                # sound: the one-shot categories are offered, as for a one-shot; and so they
                # are for a held, tonal "loop" no whole-bar tempo fits, _held_loop)
                _hl = is_loop and not is_oneshot and _held_loop(r)
                _c = _clap_home(emb_n[id2row[r.id]], anchors,
                                (is_oneshot and not is_loop) or (not is_oneshot and not is_loop and _held(r))
                                or _hl,
                                is_loop and not is_oneshot and not _hl,
                                misfiled.categories(r.path, emb_n[id2row[r.id]]))
                if _c:
                    homes[r.id] = _c
                    support[(r.id, _c)] = 1
                    votes[r.id] = dict(son_labels=son_labels, ab_tags=ab_tags, son_cats=[], ab_cats=[],
                                       clap=_c)
                    n_clap += 1
            continue
        # trust Ableton's front-end classification: drop cross-family homes
        cand = _family_veto(cand, ab_tags)
        cand = _name_filter_veto(cand, _fn)
        if "FX" in cand and _fx_drum_leak(son_labels, ab_tags):
            cand = cand - {"FX"}
            if not cand:
                continue
        # keys never home in SYNTH/SUB/FX; a keys chord one-shot may home in STABS
        _keyz = False
        if cand & KEYS_EXCLUDE:
            if _is_keys(_fn, emb_n[id2row[r.id]], r.harmonicity, keys_anchors):
                _keyz = True
                n_keys += 1
                _chord = is_oneshot and not is_loop and _is_chord(
                    _fn, r.chroma_concentration, piano_min_chroma, r.harmonicity)
                cand = _keys_candidates(cand, True, _chord, _is_stab(
                    _fn, r.duration_s, (r.n_events or 1) <= 1))
                if KEYS_CHORD_HOME and KEYS_CHORD_HOME in cand and KEYS_CHORD_HOME not in son_cats | ab_cats:
                    ab_cats = ab_cats | {KEYS_CHORD_HOME}      # counts as one vote
                    n_keys_chords += 1
                if not cand:
                    continue
        _oneshot = is_oneshot and not is_loop
        # a named scratch homes in FX (its scratch band) from any one-shot category
        # (curate_config SCRATCH_NAME_RE; SCRATCHES was folded into FX)
        _scr = _oneshot and _scratch_named(_fn, r.rel_path)
        if _scr and cand & SCRATCH_FROM and SCRATCH_HOME not in misfiled.categories(
                r.path, emb_n[id2row[r.id]]):
            cand = {SCRATCH_HOME}
            ab_cats = ab_cats | {SCRATCH_HOME}
            n_named_drums += 1
        # a named riser / downlifter / sweep homes in FX (not PADS or SYNTH); a
        # file no classifier gave a shape counts as a one-shot here, as verify has it
        if not is_loop and not _scr and cand & FX_TRANSITION_FROM and _is_fx_transition(_fn) \
                and "FX" not in misfiled.categories(r.path, emb_n[id2row[r.id]]):
            cand = {"FX"}
            ab_cats = ab_cats | {"FX"}
        # a file named as exactly one kind of drum homes in that drum category
        _dn = _drum_named_in(_fn, r.rel_path or r.path, fallback) if (_oneshot and not _scr) else None
        if _dn and "VOX" not in cand and cand & DRUM_NAME_FROM and _dn not in misfiled.categories(
                r.path, emb_n[id2row[r.id]]):
            if cand != {_dn}:
                n_named_drums += 1
            cand = {_dn}
            ab_cats = ab_cats | {_dn}
        # acoustic percussion isn't a synth: a SYNTH candidate from a drum / percussion
        # folder or an acoustic-percussion pack homes in PERC; a file no
        # classifier gave a shape counts as a one-shot here, as verify has it
        if not is_loop and not _scr and "SYNTH" in cand and _perc_sourced(r.rel_path, _fn) \
                and "PERC" not in misfiled.categories(r.path, emb_n[id2row[r.id]]):
            cand = {"PERC"}
            ab_cats = ab_cats | {"PERC"}
        # a file named as keys isn't a drum (a "Piano Thud" is no kick, a "Piano Low 02" no tom)
        if not _dn and KEYBOARD_NAME_RE.search(_fn) and cand & KIT_CATS:
            cand = cand - KIT_CATS
            if not cand:
                continue
        # STABS holds named stabs only: Sononym's stab label alone isn't enough
        if "STABS" in cand and not _stab_named(_fn):
            cand = cand - {"STABS"}
            if not cand:
                continue
        # a short one-shot named as a chord or a stab is a stab, not a synth / bass / FX /
        # blip / pad
        # (a named riser / downlifter / sweep stays FX's, "Riser Chord", unless a
        # classifier or Live calls it a loop: verify holds neither rule on those)
        if not _keyz and not _dn and _is_named_stab(_fn, r.duration_s, r.n_events, is_oneshot and not is_loop) \
                and cand & STAB_CHORD_FROM and not (_is_fx_transition(_fn) and not _is_loop_row(r)) \
                and "STABS" not in misfiled.categories(r.path, emb_n[id2row[r.id]]):
            cand = (cand - STAB_CHORD_FROM) | {"STABS"}
            ab_cats = ab_cats | {"STABS"}                    # counts as one vote
            n_chord_stabs += 1
        # rated Misfiled in a category: re-home to the next-best candidate
        _mc = misfiled.categories(r.path, emb_n[id2row[r.id]])
        if _mc:
            cand = cand - _mc
            if not cand:
                continue
        _rv = _name_reserved(os.path.basename(r.path or ""))
        if _rv:
            cand = cand & {_rv}
            if not cand:
                continue
        sup = {c: (1 if c in son_cats else 0) + (1 if c in ab_cats else 0) for c in cand}
        votes[r.id] = dict(son_labels=son_labels, ab_tags=ab_tags,
                           son_cats=sorted(son_cats), ab_cats=sorted(ab_cats))
        for c in cand:
            support[(r.id, c)] = sup[c]
        if len(cand) > 1:
            n_multi += 1
            if son_cats and ab_cats and son_cats != ab_cats:
                n_conflict += 1
        v = emb_n[id2row[r.id]]
        homes[r.id] = max(cand, key=lambda c: (sup[c], float(v @ anchors[c]), c))
        # without Sononym: a loop homed in a drum-loop category by its words or its audio
        # that fails that category's gates (CLAP, or too tonal for it) and clearly passes
        # PHRASES' CLAP gate is a phrase
        if fallback and phrases_on and _row_is_loop(r) and homes[r.id] in _drum_loop_categories() \
                and PHRASES_CATEGORY not in _mc and _phrase_shaped(r):
            _hm = CATEGORIES[homes[r.id]].get("har_max")
            _rh = _clap_rehome(v, anchors, _mc, frm=homes[r.id],
                               failed=_hm is not None and r.harmonicity is not None and r.harmonicity > _hm)
            if _rh == PHRASES_CATEGORY:
                homes[r.id] = _rh
                support[(r.id, _rh)] = 1
                votes[r.id]["rehomed"] = _rh
                n_rehomed += 1

    # byte-identical copies (the same sample shipped in two packs) share one home, so the
    # same audio can't land in two categories (in-category hash dedup then keeps one)
    n_unified = _unify_twin_homes(homes, {r.id: r.file_hash for r in rows})
    log(f"Homes: {n_unified:,} duplicate copies moved to their twin's home")
    if phrases_on:
        log(f"Homes: {n_phrases:,} musical phrases routed to {PHRASES_CATEGORY}")
    if n_rehomed:
        log(f"Homes: {n_rehomed:,} phrase-marked loops that sound like drum loops (CLAP) routed to "
            f"the drum loops")
    log(f"Homes: {n_chord_stabs:,} chord- or stab-named one-shots routed to STABS")
    log(f"Homes: {n_named_drums:,} drum- or scratch-named one-shots routed by their name")
    if fallback and n_unrec:
        log(f"Homes: {n_unrec:,} samples no name rule recognized; the CLAP fallback placed "
            f"{n_clap:,} of them by sound (`fourier why --unrecognized` lists them)")
    if n_sound or n_sound_over:
        log(f"Homes: the sound model placed {n_sound:,} samples no name rule recognized, and "
            f"{n_sound_over:,} only a pack's name or a weak word described")
    log(f"Homes: {len(homes):,} samples routed; {n_multi:,} multi-candidate, "
        f"{n_conflict:,} source-conflicts resolved by support+CLAP; "
        f"{n_keys:,} keys kept out of {'/'.join(sorted(KEYS_EXCLUDE))} "
        + (f" ({n_keys_chords:,} chords offered to {KEYS_CHORD_HOME})" if KEYS_CHORD_HOME else ""))
    return homes, support, votes


def _roots_sql() -> str:
    """SQL: the sample sits under one of the instrument collections' top folders
    (INSTRUMENT_ROOTS); false when there are none."""
    if not INSTRUMENT_ROOTS:
        return "(1 = 0)"
    return "(" + " OR ".join(f"samples.rel_path LIKE :iroot{i}" for i in range(len(INSTRUMENT_ROOTS))) + ")"


def _roots_params() -> dict:
    return {f"iroot{i}": f"{root}/%" for i, root in enumerate(INSTRUMENT_ROOTS)}


# The CLAP fallback (without Sononym only): a sample no name rule recognized (no category
# from its path's words, Live's tags or the words standing in for them) but with a shape
# (one-shot or loop) is homed at the category whose CLAP prompts it sits closest to, when it
# clears that category's gate by CLAP_FALLBACK_MARGIN (the category's break_min, else
# CLAP_FALLBACK_MIN, and its anti-prompts) and beats the runner-up by as much. One-shots
# go to the one-shot categories (not STABS, which holds named stabs only), loops to the
# drum-loop categories' gates. `fourier why` says "placed by sound (CLAP)".
CLAP_FALLBACK_MIN = _tunable("CLAP_FALLBACK_MIN", 0.30)
CLAP_FALLBACK_MARGIN = _tunable("CLAP_FALLBACK_MARGIN", 0.02)
# {nearest: preferred}: a near tie (within CLAP_FALLBACK_PREFER_MARGIN) between these goes
# to the preferred category (claps and hats are both short noise bursts to CLAP)
CLAP_FALLBACK_PREFER = _tunable("CLAP_FALLBACK_PREFER", {"HATS": "CLAPS"})
CLAP_FALLBACK_PREFER_MARGIN = _tunable("CLAP_FALLBACK_PREFER_MARGIN", 0.05)
# The sound model (metadata/sound.py, provider fourier:sound; without Sononym only, and only
# when there are weights trained on the library, metadata/train.py): a sample no name rule recognized is homed at the sound
# model's category, ahead of the CLAP fallback, when its probability is at least
# SOUND_MODEL_MIN; a sample whose own name and folder hold no strong category word (only
# the folders above, a pack's or vendor's name, or a weak word gave it one) is homed there
# when it is at least SOUND_MODEL_OVERRIDE_MIN. One-shot and drum-loop categories only (the
# instruments, waves and phrases go by name), never STABS (named stabs only), and the shape must fit:
# the sample's own (names, else audio), else the model's. `fourier why` says "sound model".
SOUND_MODEL_MIN = _tunable("SOUND_MODEL_MIN", 0.70)
SOUND_MODEL_OVERRIDE_MIN = _tunable("SOUND_MODEL_OVERRIDE_MIN", 0.90)
_CLAP_ANTI = {}


def _clap_anti(cat):
    """The mean of a category's anti-prompts (normalized), or None when it has none."""
    if cat not in _CLAP_ANTI:
        anti = CATEGORIES[cat].get("anti")
        if anti:
            a = np.stack([embed_text(p) for p in anti]).mean(0)
            _CLAP_ANTI[cat] = a / (np.linalg.norm(a) + 1e-9)
        else:
            _CLAP_ANTI[cat] = None
    return _CLAP_ANTI[cat]


# Without Sononym, a loop the phrase rule homes in PHRASES whose CLAP scores fail PHRASES' gate
# (closer to its anti-prompts, the drum breaks) is homed in the drum-loop category whose gate
# it clears by CLAP_REHOME_MARGIN over the gate's minimum and its anti-prompts; there its
# drum evidence is the CLAP gate, and its harmonicity (lo-fi drums under a chord read tonal)
# is not held against it (_select_records: rehomed). The other way round too: a loop homed in
# a drum-loop category that fails its gate and clears PHRASES' by the margin is a phrase.
CLAP_REHOME_MARGIN = _tunable("CLAP_REHOME_MARGIN", 0.10)


def _clap_rehome(v, anchors, misfiled=frozenset(), frm=None, failed=False):
    """The loop category a loop homed in `frm` (PHRASES by default) is re-homed in (see
    CLAP_REHOME_MARGIN), or None: frm's gate (its prompts above its anti-prompts, at least
    its break_min) fails, or failed (another of its gates did: too tonal for a drum-loop
    category), and the other side's (a drum-loop category for a phrase, PHRASES for a drum
    loop) passes with the margin."""
    from .curate_config import CATEGORIES_OFF
    frm = frm or PHRASES_CATEGORY
    fa, fanti = _clap_anchor(frm, anchors), _clap_anti(frm)
    if fa is None or fanti is None:
        return None
    sf_, sa = float(v @ fa), float(v @ fanti)
    if not failed and sf_ > sa and sf_ >= float(CATEGORIES[frm].get("break_min", 0.30)):
        return None                      # its own gate takes it: it stays
    targets = _drum_loop_categories() if frm == PHRASES_CATEGORY else [PHRASES_CATEGORY]
    best = None
    for c in targets:
        if c == frm or c not in CATEGORIES or c in CATEGORIES_OFF or c in misfiled:
            continue
        a, anti = _clap_anchor(c, anchors), _clap_anti(c)
        if a is None:
            continue
        sb = float(v @ a)
        sx = float(v @ anti) if anti is not None else -1.0
        if sb >= float(CATEGORIES[c].get("break_min", 0.30)) + CLAP_REHOME_MARGIN \
                and sb - sx >= CLAP_REHOME_MARGIN and (best is None or sb > best[0]):
            best = (sb, c)
    return best[1] if best else None


def _phrase_shaped(r) -> bool:
    """A loop its producer marks as a phrase (_phrase_marked), PHRASE_DUR long, with a
    whole-bar tempo (without Sononym's tempo sources too): one PHRASES could hold."""
    path = getattr(r, "path", None) or ""
    rel = getattr(r, "rel_path", None) or library_rel(path)
    fn = getattr(r, "filename", None) or os.path.basename(path)
    dur = getattr(r, "duration_s", None)
    if not dur or not (PHRASE_DUR[0] <= dur <= PHRASE_DUR[1]) or _phrase_marked(rel, fn, True) is None:
        return False
    return _resolve_tempo(fn, dur, getattr(r, "bpm", None), getattr(r, "tempo_bpm", None),
                          _fallback_tempos(r))[0] is not None


def _clap_anchor(cat, anchors):
    """A category's prompt anchor (compute_homes' anchors, else the mean of its prompts)."""
    if cat in anchors:
        return anchors[cat]
    ps = CATEGORIES.get(cat, {}).get("phrases")
    if not ps:
        return None
    a = np.stack([embed_text(p) for p in ps]).mean(0)
    anchors[cat] = a / (np.linalg.norm(a) + 1e-9)
    return anchors[cat]


def _held(r) -> bool:
    """A sample with no shape label that sounds like one sound: a single event between
    silences, or a held, mostly harmonic one with slow onsets (shadow.SUSTAINED_HPR,
    SUSTAINED_MAX_HZ: a pad's chord changes). The CLAP fallback offers it the one-shot
    categories."""
    from ..metadata.shadow import SUSTAINED_HPR, SUSTAINED_MAX_HZ
    n_ev, har, ons = getattr(r, "n_events", None), getattr(r, "harmonicity", None), getattr(r, "onset_rate_hz", None)
    if n_ev is not None and n_ev <= 1:
        return True
    return har is not None and har >= SUSTAINED_HPR and ons is not None and ons < SUSTAINED_MAX_HZ


def _held_loop(r) -> bool:
    """Without Sononym: a file the audio alone calls a loop (its onsets) that is sustained (one
    event: no silence in it), slow in onsets (under shadow.SUSTAINED_MAX_HZ: chord changes,
    not a beat) and tonal (Fourier's own tonal call, resolve.own_tonal), with no whole-bar
    tempo anything states (its name, its folder, its ACID chunk): a pad changing chords,
    which no loop category can sync. A tempo only estimated from its audio or its length
    doesn't count: slow onsets carry no beat to estimate. The CLAP fallback offers it the
    one-shot categories (PADS among them), as for a held sound with no shape."""
    from ..metadata.shadow import SUSTAINED_MAX_HZ, shape_labels
    if not _row_is_loop(r):
        return False
    n_ev, ons = getattr(r, "n_events", None), getattr(r, "onset_rate_hz", None)
    if n_ev is None or n_ev > 1 or ons is None or ons >= SUSTAINED_MAX_HZ:
        return False
    if not _resolve.own_tonal(getattr(r, "harmonicity", None), getattr(r, "chroma_concentration", None)):
        return False
    if "class.loop" in shape_labels(getattr(r, "rel_path", None) or getattr(r, "path", None) or ""):
        return False                  # a loop by its own name or folder stays one
    got = _resolve.loop_tempo(r, True)[0]
    return got.value is None or got.source in (_resolve.AUDIO, _resolve.LENGTH)


def _clap_home(v, anchors, oneshot, loop, misfiled=frozenset()):
    """The category the CLAP fallback homes an unrecognized sample in, or None: the nearest
    anchor of its shape's categories (one-shot: kind oneshot, named-only STABS left out;
    loop: the drum-loop categories), when the score clears that category's gate (break_min,
    else CLAP_FALLBACK_MIN, plus CLAP_FALLBACK_MARGIN; above its anti-prompts by the margin)
    and the runner-up by the margin. Categories switched off or rated Misfiled for it aren't
    offered."""
    from .curate_config import CATEGORIES_OFF
    if oneshot:
        cats = [c for c, cfg in CATEGORIES.items() if cfg["kind"] == "oneshot" and c in anchors
                and not cfg.get("stab_names") and not cfg.get("scratch_names")]
    elif loop:
        cats = [c for c in _drum_loop_categories() if c in anchors]
    else:
        return None
    cats = [c for c in cats if c not in CATEGORIES_OFF and c not in misfiled]
    if not cats:
        return None
    scores = sorted(((float(v @ anchors[c]), c) for c in cats), reverse=True)
    s1, c1 = scores[0]
    s2 = scores[1][0] if len(scores) > 1 else -1.0
    # a near tie the CLAP model is known to call the wrong way (a clap heard as a hat): the
    # preferred category wins when it is the runner-up within the margin and clears its gate
    pref = CLAP_FALLBACK_PREFER.get(c1)
    if pref and len(scores) > 1 and scores[1][1] == pref and s1 - s2 < CLAP_FALLBACK_PREFER_MARGIN \
            and s2 >= float(CATEGORIES[pref].get("break_min", CLAP_FALLBACK_MIN)) + CLAP_FALLBACK_MARGIN:
        return pref
    gate = float(CATEGORIES[c1].get("break_min", CLAP_FALLBACK_MIN))
    anti = _clap_anti(c1)
    if s1 < gate + CLAP_FALLBACK_MARGIN or s1 - s2 < CLAP_FALLBACK_MARGIN:
        return None
    if anti is not None and s1 - float(v @ anti) < CLAP_FALLBACK_MARGIN:
        return None
    return c1


def _sound_labels(session) -> dict:
    """Without Sononym: {sample id: (category, probability, loop probability)} from the sound
    model in use (metadata/sound.current); {} with Sononym or without weights."""
    if not _fallback(session):
        return {}
    from ..metadata import sound
    return sound.current(session)


def _sound_home(got, oneshot, loop, misfiled=frozenset(), minimum=None):
    """The category the sound model homes a sample in (SOUND_MODEL_MIN, or `minimum`), or
    None: its category clears the threshold, is a one-shot category or a drum-loop one
    (PHRASES holds producer-marked phrases only) that is on, not STABS (named stabs only),
    not rated Misfiled for it, and fits the sample's shape (oneshot / loop: its own;
    neither: the model's loop probability)."""
    from .curate_config import CATEGORIES_OFF
    if not got:
        return None
    cat, p, p_loop = got
    if p is None or p < (SOUND_MODEL_MIN if minimum is None else minimum):
        return None
    cfg = CATEGORIES.get(cat)
    if cfg is None or cat in CATEGORIES_OFF or cat in misfiled or cfg["kind"] not in ("oneshot", "loop") \
            or cfg.get("stab_names") or cfg.get("scratch_names"):
        return None
    if oneshot == loop:                  # no shape of its own (or both): the model's
        if p_loop is None:
            return None
        oneshot, loop = p_loop < 0.5, p_loop >= 0.5
    if cfg["kind"] == "loop" and not (loop and cat in _drum_loop_categories()):
        return None                      # (PHRASES holds producer-marked phrases only)
    if cfg["kind"] == "oneshot" and not oneshot:
        return None
    return cat


def _weak_path(r) -> bool:
    """No strong category word in the file's own name or folder (shadow.path_rules): the
    path provider's category came from the folders above (a pack's or vendor's name) or from
    a weak word (path.yaml's `weak` section)."""
    from ..metadata.shadow import STRONG, path_rules, shape_scope, words_of
    rel = r.rel_path or r.path or ""
    name, folder = shape_scope(rel)
    words = words_of(f"{os.path.splitext(name)[0]} {folder}")
    return not any(kind == STRONG and not label.startswith("class.") and rx.search(words)
                   for label, rx, kind in path_rules())


def unrecognized(session) -> list:
    """Without Sononym: the samples no name rule recognized, as compute_homes meets them: no
    category from their path's words (the path provider), Live's tags or the words standing
    in for them, and none of the rules that place a file by its name (a phrase, a wave, an
    acoustic instrument, an organ or keys, an impulse response or preview). The CLAP fallback
    may still place them by sound. With Sononym, none (it classifies every sample)."""
    if not _fallback(session):
        return []
    from .curate_config import ABLETON_TAGS_BY_CATEGORY
    part = {cat: cfg for cat, cfg in CATEGORIES.items() if cfg["kind"] in ("oneshot", "loop", "gated")}
    tag2cats = defaultdict(set)
    for cat in part:
        for t in ABLETON_TAGS_BY_CATEGORY.get(cat, []):
            tag2cats[t].add(cat)
    out = []
    for r in _with_name_tags(session, fetch(session, sample_select(*HOME_FIELDS, session=session))):
        rel = r.rel_path or library_rel(r.path or "")
        fn = os.path.basename(r.path or "")
        pack = _pack_of(r.rel_path, r.path)
        ab = _as_list(r.ableton_tags)
        if (_is_wave(rel, r.duration_s) or _is_ir(r.rel_path, r.path) or _is_preview(r.rel_path, r.path)
                or INST_ROUTED_PACKS.search(pack) or _is_phrase(r, True)):
            continue
        if (_acoustic_tagged(ab, fn, pack) or _is_acoustic_named(fn, ab) or _is_wind_named(fn, ab)
                or _orch_path(r.rel_path) is not None or _is_organ_named(fn) or KEYBOARD_NAME_RE.search(fn)
                or _keys_labeled(r)):
            continue
        son = _fallback_loop_votes(_classifier_votes(part, _canonical_of(r)), r, ab)
        if son or any(tag2cats.get(t) for t in ab):
            continue
        out.append(r)
    return out


def placed_nowhere(session, master_dir) -> list | None:
    """Without Sononym: [(row, labels)] of the samples the built-in providers recognized
    (labels other than one-shot / loop) that the last build of master_dir placed nowhere: not
    in the master, and no category's why log names them (left out by a step, held by a
    category too small to build, placed by sound). A rule took them out of every category
    they voted for (keys kept out of SYNTH), or the category they'd go to is off. None
    without a build of master_dir to read; [] with Sononym."""
    if not _fallback(session):
        return []
    from .ratings import drop_set
    man_p = os.path.join(master_dir or "", "manifest.json")
    if not master_dir or not os.path.exists(man_p):
        return None
    try:
        man = manifests.read(man_p)
    except (OSError, ValueError):
        return None
    srcs = {e.get("src") for c in (man.get("categories") or {}).values() for e in (c.get("entries") or ())}
    named = set()
    for cat in CATEGORIES:
        doc = why_log.read(master_dir, cat) or {}
        for ids in (doc.get("out") or {}).values():
            named |= {int(i) for i in ids}
        named |= {int(i) for i in (doc.get("readmitted") or {})}
        for key in ("clap_homed", "clap_rehomed", "held", "sound_homed"):
            named |= {int(i) for i in (doc.get(key) or ())}
    unrec = {r.id for r in unrecognized(session)}
    drops, mirrors = drop_set(), _mirror_dups(session)
    out = []
    rows = _with_name_tags(session, fetch(session, sample_select(*HOME_FIELDS, session=session)))
    held = {r.file_hash for r in rows if r.path in srcs and r.file_hash}     # a copy is in the master
    for r in rows:
        if r.id in unrec or r.id in named or r.path in srcs or r.path in drops or r.path in mirrors \
                or (r.file_hash and r.file_hash in held):
            continue
        rel = r.rel_path or library_rel(r.path or "")
        if (_is_wave(rel, r.duration_s) or _is_ir(r.rel_path, r.path) or _is_preview(r.rel_path, r.path)
                or INST_ROUTED_PACKS.search(_pack_of(r.rel_path, r.path))):
            continue
        labels = sorted(c for c in _canonical_of(r) if not c.startswith("class."))
        if labels:
            out.append((r, labels))
    return out


def top_folders(rels, n=5) -> list:
    """[(folder, files)]: the library folders (first two levels) holding most of these paths."""
    c = Counter("/".join(x.split("/")[:-1][:2]) or "(the library folder itself)" for x in rels)
    return c.most_common(n)


_NOISE_RE = {c: re.compile(cfg["noise"], re.I) for c, cfg in CATEGORIES.items()
             if cfg.get("noise") and cfg["noise"] != r"(?!x)x"}


def _name_filter_veto(cand, filename):
    """Drop candidate homes whose filename filter (a category's `noise`) rejects the file,
    so it homes in its next candidate instead of vanishing at selection (a cymbal
    that voted MALLETS and PERC goes to PERC). Never empties the set."""
    kept = {c for c in cand if not (c in _NOISE_RE and _NOISE_RE[c].search(filename or ""))}
    return kept or cand


_FX_BAND_ANCHORS = {}


def _resolve_misc_fx(rec, emb_n):
    """Place each 'misc' FX record (no FX type from either library, not pitched) in the
    FX type whose phrases CLAP puts it closest to."""
    miss = [d for d in rec if d.get("band") == "misc"]
    if not miss:
        return
    bands = [b for b in FX_BAND_PHRASES if b not in ("synth", "riser", "downlifter")]   # named only
    if not _FX_BAND_ANCHORS:
        for b in FX_BAND_PHRASES:
            T = np.stack([embed_text(p) for p in FX_BAND_PHRASES[b]])
            a = T.mean(0)
            _FX_BAND_ANCHORS[b] = a / (np.linalg.norm(a) + 1e-9)
    A = np.stack([_FX_BAND_ANCHORS[b] for b in bands])
    for d in miss:
        d["band"] = bands[int(np.argmax(A @ emb_n[d["row"]]))]


def _get_homes(session, emb_n, id2row, log):
    key = (id(session), len(id2row))
    if key not in _HOMES_CACHE:
        _HOMES_CACHE.clear()
        _HOMES_CACHE[key] = compute_homes(session, emb_n, id2row, log)
    return _HOMES_CACHE[key]


def _row_select(session=None):
    return sample_select(*CANDIDATE_FIELDS, session=session)


def _fallback(session) -> bool:
    """No Sononym: the built-in path and audio providers classify (metadata/providers.py).
    Without a session, Sononym and Live are assumed (as metadata/rows.py does)."""
    from ..metadata.providers import active
    return session is not None and active(session).fallback


def _live(session) -> bool:
    """Live's tags are in use for this build (metadata/providers.py)."""
    from ..metadata.providers import ABLETON, active
    return ABLETON in active(session).names


def _words_text(text) -> str:
    """Lowercase words, separators and camelCase split, padded: " drum loop 01 "."""
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text or "")
    return " " + " ".join(re.findall(r"[a-z0-9]+", t.lower())) + " "


def _has_word(text, word) -> bool:
    """`word` (or its plural) as whole words of `text` (a _words_text)."""
    w = _words_text(word).strip()
    return bool(w) and re.search(r"(?<= )" + re.escape(w) + r"(?:e?s)?(?= )", text) is not None


def _name_tags(rel, loop=False) -> list:
    """The Live tags a path's words stand in for (NAME_TAGS), for a build without Live: whole
    words of the file's name or its folder (shadow.shape_scope; a "WAV" folder hands over to
    the one above), not of the vendor's or pack's name. A file whose own name names a drum
    hit ("Kick 01") takes no tag from its folder ("Breaks"). A loop (`loop`: the classifier
    says so) whose own name or folder has a loop word (shadow.shape_labels: "Loops/Loop 05")
    may take a loop tag (a NAME_TAGS key naming a loop, "Drum Loop") from the folders above
    too: a pack's name can say what kind of loop its loops are, never that a file is a loop,
    and a loop by its audio alone (a riser with onsets) takes nothing from it."""
    from ..metadata.shadow import drum_hit_named, shape_labels, shape_scope
    from .curate_config import NAME_TAGS
    name, folder = shape_scope(rel)
    drum_hit = drum_hit_named(name)
    own = _words_text(os.path.splitext(name)[0])
    near = own + _words_text(folder) if folder and not drum_hit else own
    tags = [t for t, words in NAME_TAGS.items() if any(_has_word(near, w) for w in words)]
    if loop and not drum_hit and "class.loop" in shape_labels(rel):
        folders = [p for p in (rel or "").split("/") if p][:-1]
        above = folders[:folders.index(folder)] if folder in folders else []
        far = _words_text(" ".join(above))
        tags += [t for t, words in NAME_TAGS.items() if t not in tags and "loop" in t.lower()
                 and any(_has_word(far, w) for w in words)]
    return tags


def _with_name_tags(session, rows):
    """Without Live, each row's ableton_tags are the ones its path stands in for (a loop by
    the classifier's shape may take a loop tag from its pack's name, _name_tags)."""
    if _live(session):
        return rows
    return [r._replace(ableton_tags=_name_tags(r.rel_path or r.path, _row_is_loop(r))) for r in rows]


def _row_is_loop(r) -> bool:
    classes = _as_list(getattr(r, "classes", None))
    return any("Loop" in c for c in classes) and not any("OneShot" in c for c in classes)


def _name_tag_sql(tags, prefix):
    """(sql, params): the path has a word standing in for one of these tags (NAME_TAGS)."""
    from .curate_config import NAME_TAGS
    words = sorted({w for t in tags for w in NAME_TAGS.get(t, ())})
    if not words:
        return "1 = 0", {}
    params = {f"{prefix}{i}": f"%{w}%" for i, w in enumerate(words)}
    return "(" + " OR ".join(f"lower(samples.rel_path) LIKE :{k}" for k in params) + ")", params


def _fetch_rows(session, cfg, category=None):
    """Return sample rows (metadata/rows.py, CANDIDATE_FIELDS) that are candidates
    for this category from EITHER classifier: Sononym (class + category) OR an
    Ableton auto-tag mapped to this category. Ordered by id for determinism.
    Without Live, path words stand in for the tags a category needs (NAME_TAGS)."""
    return _with_name_tags(session, _fetch_rows_raw(session, cfg, category))


def _fetch_rows_raw(session, cfg, category=None):
    q = _row_select(session)
    q0 = q
    params, where = {}, []
    kind = cfg["kind"]
    from .curate_config import ABLETON_TAGS_BY_CATEGORY
    if kind in ("oneshot", "loop"):
        son, sp = like_any("canonical", ["class.oneshot" if kind == "oneshot" else "class.loop"],
                           "cls", quoted=True, session=session)
        params.update(sp)
        cats = _labels_for(cfg)
        if cats:
            cc, cp = like_any("canonical", cats, "c", quoted=True, session=session)
            params.update(cp)
            son = f"({son} AND ({cc}))"
        tags = ABLETON_TAGS_BY_CATEGORY.get(category, []) if category else []
        if tags:
            tc, tp = like_any("ableton_tags", tags, "t", quoted=True, session=session)
            params.update(tp)
            where.append(f"({son} OR ({tc}))")
        else:
            where.append(f"({son})")
    elif kind == "gated":
        cc, cp = like_any("canonical", _labels_for(cfg), "g", quoted=True, session=session)
        params.update(cp)
        where.append(f"({cc})")
    if where:
        q = q.where(text(" AND ".join(where)).bindparams(**params))
    if kind == "instrument":
        if cfg.get("clap_source") and cfg.get("pool_labels"):
            # broaden beyond the instrument packs: single-note keys from anywhere
            # in the library (a OneShot in a keys category) OR any Ableton sample;
            # _select_records then CLAP-gates to this instrument + guards single-note.
            # (without Sononym, also the one-shots the path words name keys: _KEYS_LABEL)
            _pool = list(cfg["pool_labels"]) + ([_KEYS_LABEL] if _fallback(session) else [])
            cc, pp = like_any("canonical", _pool, "p", quoted=True, session=session)
            os_sql, os_p = like_any("canonical", ["class.oneshot"], "os", quoted=True, session=session)
            pp.update(os_p)
            # ...and named organs from anywhere (they home in PIANO)
            q = q.where(text(
                "((" + os_sql + " AND (" + cc + ")) "
                "OR " + _roots_sql() + " "
                "OR lower(samples.filename) LIKE '%organ%' OR lower(samples.filename) LIKE '%hammond%' "
                "OR lower(samples.filename) LIKE '%drawbar%' OR lower(samples.filename) LIKE '%farfisa%')"
            ).bindparams(**_roots_params(), **pp))
        elif cfg.get("ableton_any"):
            # pack files OR anything the given Ableton instrument tags cover (pulls
            # acoustic strings/winds/brass/guitar from the whole library, not one pack)
            tc, ap = like_any("ableton_tags", cfg["ableton_any"], "a", quoted=True, session=session)
            if not _live(session):           # path words stand in for the instrument tags
                tc, ap = _name_tag_sql(cfg["ableton_any"], "a")
            if cfg.get("name_any"):
                # plus plucked strings named in the filename (checked exactly in Python)
                subs = ["guitar", "zither", "lyre", "harp", "banjo", "sitar", "koto", "mandolin",
                        "ukulele", "cigar", "bouzouki", "oud", "balalaika", "dulcimer", "lute",
                        # tuned percussion
                        "xylo", "marimba", "arimba", "vibraphone", "vibes", "glock", "kalimba",
                        "thumb piano", "mbira", "music box", "toy piano", "steel drum", "steel pan",
                        "steelpan", "celest", "gamelan", "chime", "tubular", "church bell",
                        "hand bell", "handbell", "singing bowl", "tibetan", "bell tree", "handpan",
                        # named winds / brass (reserved for ACOUSTIC by _is_wind_named, so
                        # the prefilter must fetch them, or they'd be left out of every
                        # category)
                        "flute", "clarinet", "oboe", "bassoon", "sax", "recorder", "harmonica",
                        "accordion", "melodica", "ocarina", "trumpet", "trombone", "tuba", "horn",
                        "brass", "woodwind", "kazoo", "pan pipe", "panpipe"]
                tc += " OR " + " OR ".join(f"lower(samples.filename) LIKE :n{i}" for i in range(len(subs)))
                ap.update({f"n{i}": f"%{w}%" for i, w in enumerate(subs)})
            # (and the orchestra folders, ORCH_ROOTS, reserved for ACOUSTIC by path)
            from .curate_config import ORCH_ROOTS
            ap.update({f"orch{i}": f"{p}%" for i, p in enumerate(ORCH_ROOTS)})
            orch = "".join(f"OR samples.rel_path LIKE :orch{i} " for i in range(len(ORCH_ROOTS)))
            q = q.where(text("(" + _roots_sql() + " " + orch + "OR " + tc + ")").bindparams(
                **_roots_params(), **ap))
        else:
            q = q.where(text(_roots_sql()).bindparams(**_roots_params()))
    if kind == "waves":
        from .curate_config import WAVE_PATH_WORDS
        pats = ["%wavetable%", "%wave table%", "%cycle%"] + [f"%{w.lower()}%" for w in WAVE_PATH_WORDS]
        cc = " OR ".join(f"lower(samples.rel_path) LIKE :w{i}" for i in range(len(pats)))
        q = q.where(text(f"({cc})").bindparams(**{f"w{i}": p for i, p in enumerate(pats)}))
        q = q.where(Sample.duration_s <= WAVE_MAX_DUR)
        # a cycle too short to play as a tone (a high sine can be a few dozen samples)
        q = q.where(Sample.duration_s * func.coalesce(Sample.sample_rate, 44100) >= WAVE_MIN_SAMPLES - 0.5)
    q = q.order_by(Sample.id)
    rows = fetch(session, q)
    _ov_subs = _override_subs_for(category)
    if _ov_subs:
        _oc = " OR ".join(f"lower(samples.filename) LIKE :o{i}" for i in range(len(_ov_subs)))
        _op = {f"o{i}": f"%{sub}%" for i, sub in enumerate(_ov_subs)}
        _have = {r.id for r in rows}
        for _rr in fetch(session, q0.where(text(_oc).bindparams(**_op)).order_by(Sample.id)):
            if _rr.id not in _have:
                rows.append(_rr)
                _have.add(_rr.id)
    # files rated Keep here, or Move-<this category> elsewhere (ratings.keep_pins): pinned
    # here, whatever the classifier says (a build with other providers may not offer them)
    try:
        from .ratings import keep_pins
        _mv = sorted(p for p, t in keep_pins().items() if t == category)
    except Exception:
        _mv = []
    if _mv:
        _have = {r.id for r in rows}
        for i in range(0, len(_mv), 900):
            for _rr in fetch(session, q0.where(Sample.path.in_(_mv[i:i + 900])).order_by(Sample.id)):
                if _rr.id not in _have:
                    rows.append(_rr)
                    _have.add(_rr.id)
    return _sort_rows(rows)


def _sort_rows(rows):
    # WAV first so a wav twin wins over its aif/mp3 sibling in content-dedup; id for stable order
    return sorted(rows, key=lambda r: (
        0 if (r.filename or "").lower().endswith(".wav")
        else (1 if (r.filename or "").lower().endswith((".aif", ".aiff")) else 2), r.id))


def _fetch_rows_by_id(session, ids):
    """Candidate rows (same columns as _fetch_rows) for the given sample ids."""
    out = []
    ids = list(ids)
    for i in range(0, len(ids), 900):
        out += fetch(session, _row_select(session).where(Sample.id.in_(ids[i:i + 900])))
    return _with_name_tags(session, out)


def _cap_instrument_packs(rec, emb_n, pattern=INSTRUMENT_PACKS, cap=INSTRUMENT_CAP):
    """Cap acoustic/orchestral multisample libraries to `cap` CLAP-spread samples
    per pack (KMeans medoids), so chromatic multisamples don't flood a category."""
    from collections import defaultdict
    by_pack = defaultdict(list)
    for r in rec:
        by_pack[r["pack"]].append(r)
    out = []
    for pack, recs in by_pack.items():
        if pattern.search(pack) and len(recs) > cap:
            P = np.stack([emb_n[r["row"]] for r in recs])
            out.extend(recs[p] for p in sorted(_spread_keep(P, cap, [bool(r.get("incumbent")) for r in recs])))
        else:
            out.extend(recs)
    return out


def _instrument_label(pack):
    """A folder label for an instrument pack: the first instrument word its name holds
    (the pianos, then a library overlay's own pack words, INSTRUMENT_LABELS_EXTRA, then the
    orchestral sections), else the pack name."""
    from .curate_config import INSTRUMENT_LABELS_EXTRA
    p = pack.lower()
    for key, lab in (("grand piano", "grand-piano"), ("upright piano", "upright-piano"),
                     *(tuple(x) for x in INSTRUMENT_LABELS_EXTRA),
                     ("woodwind", "woodwinds"), ("string", "strings"), ("brass", "brass"),
                     ("mallet", "mallets")):
        if key in p:
            return lab
    return re.sub(r"[^a-z0-9]+", "-", p).strip("-")


def _cap_vendor_share(rec, emb_n, max_share, min_vendors=3, keep=0):
    """Cap ANY single vendor (source folder, e.g. Ableton or a third-party vendor) to at
    most max_share of the pool, so no one vendor's aesthetic dominates a category. At
    most one vendor can exceed 50%, so per-vendor targeting against the rest is exact
    at max_share=0.5. Deterministic subsample -- the final per-family medoid pass
    restores CLAP spread, so a cheap seeded draw here is enough (and KMeans with k in
    the thousands would be infeasible).

    A pool from fewer than `min_vendors` vendors isn't capped: one vendor's share of it is
    all of it, and two can't both stay under max_share, so the cap would only empty the
    category. And the cap never takes a pool below `keep` (the category's minimum): the
    best of what it removed (the previous build's files, then quality) come back up to it."""
    if any(r.get("vendor", "?") is None for r in rec):
        # samples with no vendor (packs/vendors.py: a library by sound type, or flat) aren't
        # capped; the rest are capped among themselves
        free = [r for r in rec if r.get("vendor", "?") is None]
        kept = {id(r) for r in _cap_vendor_share([r for r in rec if r.get("vendor", "?") is not None],
                                                 emb_n, max_share, min_vendors, max(0, keep - len(free)))}
        return [r for r in rec if r.get("vendor", "?") is None or id(r) in kept]
    n = len(rec)
    groups = _by_vendor(rec)
    if n == 0 or len(groups) < min_vendors or all(len(g) <= max_share * n for g in groups.values()):
        return rec
    rng = np.random.default_rng(CURATION_SEED)
    out = []
    for v, rs in groups.items():
        target = int(round(max_share / (1.0 - max_share) * (n - len(rs))))
        if len(rs) <= max(1, target):
            out.extend(rs)
        else:
            # the previous build's files first, then a seeded draw
            perm = rng.permutation(len(rs))
            order = sorted(range(len(rs)), key=lambda q: (not rs[perm[q]].get("incumbent"), q))
            keep_i = sorted(perm[q] for q in order[:max(1, target)])
            out.extend(rs[i] for i in keep_i)
    if len(out) < keep:
        return _keep_best(rec, out, keep)
    return out


def _keep_best(rec, kept, keep):
    """`kept` (a subset of rec) topped up to `keep` records with the best of the rest (the
    previous build's files first, then quality, then id), in rec's order."""
    have = {id(r) for r in kept}
    rest = sorted((r for r in rec if id(r) not in have),
                  key=lambda r: (not r.get("incumbent"), -r["qual"], r["id"]))
    have |= {id(r) for r in rest[:keep - len(kept)]}
    return [r for r in rec if id(r) in have]


def _by_vendor(rec):
    g = defaultdict(list)
    for r in rec:
        g[r.get("vendor", "?")].append(r)
    return g


def _tempo_apart(a, b) -> bool:
    """Two loops filed at tempos further apart than one tempo folder holds (naming's
    SAME_TEMPO_BPM): the same groove at another tempo is no duplicate (without Sononym)."""
    ta, tb = a.get("bpm_fold") or a.get("bpm"), b.get("bpm_fold") or b.get("bpm")
    from .naming import SAME_TEMPO_BPM
    return bool(ta and tb and abs(float(ta) - float(tb)) > SAME_TEMPO_BPM)


def _prune_near_dups(rec, emb_n, thr, keep=0, partners=None, apart=None):
    """Drop CLAP near-identical samples, keeping the higher-quality one of each
    group. Deterministic: candidates are visited in (quality desc, id asc) order and
    a sample is dropped when its CLAP cosine to any already-kept sample exceeds thr.
    Survivors keep their original (id) ordering for downstream determinism.

    The prune never takes a pool below `keep` (the category's minimum): the dropped samples
    least like the kept ones come back, one at a time, up to it. `partners`, a dict, gets
    {dropped id: id of the kept sample it duplicates}. `apart(a, b)` (records), when given:
    two samples it calls apart are never each other's duplicate (loops at different tempos)."""
    if len(rec) < 2 or thr >= 1.0:
        return rec, 0
    order = sorted(range(len(rec)), key=lambda i: (not rec[i].get("incumbent"), -rec[i]["qual"], rec[i]["id"]))
    keep_idx, kept_vecs, dropped = [], [], []
    for i in order:
        v = emb_n[rec[i]["row"]]
        if kept_vecs:
            sims = np.asarray(kept_vecs) @ v
            if apart is not None:
                sims = np.where([apart(rec[i], rec[j]) for j in keep_idx], -1.0, sims)
            if float(np.max(sims)) > thr:
                dropped.append((i, keep_idx[int(np.argmax(sims))]))
                continue
        keep_idx.append(i)
        kept_vecs.append(v)
    if dropped and len(keep_idx) < keep:
        left = [i for i, _j in dropped]
        while left and len(keep_idx) < keep:
            K = np.asarray(kept_vecs)
            far = min(left, key=lambda i: (float(np.max(K @ emb_n[rec[i]["row"]])), i))
            left.remove(far)
            keep_idx.append(far)
            kept_vecs.append(emb_n[rec[far]["row"]])
    keep_set = set(keep_idx)
    if partners is not None:
        partners.update({rec[i]["id"]: rec[j]["id"] for i, j in dropped if i not in keep_set})
    pruned = len(rec) - len(keep_set)
    return [r for j, r in enumerate(rec) if j in keep_set], pruned


_NAME_OVERRIDES = [
    (tuple(s.lower() for s in ((subs,) if isinstance(subs, str) else subs)), cat)
    for subs, cat in NAME_OVERRIDES
]
_NAME_OVERRIDE_SKIP = tuple(x.lower() for x in NAME_OVERRIDE_SKIP)


def _name_override_rule(name):
    """(rule index, category) for a filename hitting a NAME_OVERRIDES rule, else
    (None, None). A 'Synth <x>' keeps its synth home."""
    if not name:
        return None, None
    nl = name.lower()
    if any(sk in nl for sk in _NAME_OVERRIDE_SKIP) or NAME_OVERRIDE_VETO.search(nl):
        return None, None
    for i, (subs, cat) in enumerate(_NAME_OVERRIDES):
        if any(s in nl for s in subs):
            return i, cat
    return None, None


def _name_override(name):
    """Filename-keyed home for a few instruments the two analyzers mis-slot
    (a music box tagged as an orchestral mallet -> ACOUSTIC when it is really a
    MALLETS tine sound). Category or None."""
    return _name_override_rule(name)[1]


def _override_subs_for(category):
    """Filename substrings whose override target is this category (fetch pull)."""
    if not category:
        return []
    return [s for subs, cat in _NAME_OVERRIDES if cat == category for s in subs]


def _pack_key2(path):
    """vendor/pack (first two folders under the library folder). Finer than _pack_of, which
    lumps a whole non-umbrella vendor (every pack in its folder) into one pack."""
    seg = [s for s in library_rel(path or "").split("/") if s]
    return "/".join(seg[:2]) or "?"


def _cap_group_spread(rec, emb_n, keyfn, cap):
    """Keep at most `cap` CLAP-spread records (KMeans medoids) per group key.
    Records whose key is None are never capped. Preserves input order."""
    groups = defaultdict(list)
    for j, r in enumerate(rec):
        k = keyfn(r)
        if k is not None:
            groups[k].append(j)
    drop = set()
    for idx in groups.values():
        if len(idx) <= cap:
            continue
        P = np.stack([emb_n[rec[j]["row"]] for j in idx])
        keep = {idx[p] for p in _spread_keep(P, cap, [bool(rec[j].get("incumbent")) for j in idx])}
        drop.update(j for j in idx if j not in keep)
    return [r for j, r in enumerate(rec) if j not in drop]


# ---------------------------------------------------------------------------
# Loop tempo resolution
# ---------------------------------------------------------------------------
TEMPO_MIN, TEMPO_MAX = 60.0, 200.0
BAR_COUNTS = _tunable("BAR_COUNTS", (1, 2, 4, 8, 16))
BAR_TOL_BEATS = _tunable("BAR_TOL_BEATS", 0.15)            # a loop "fits" a tempo if it lands within this of whole bars
_BPM_WORD = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*bpm", re.I)
_BPM_TOKEN = re.compile(r"(?:^|[_\s(\-])(\d{2,3})(?=[_\s)\-]|$)")


def _filename_bpms(filename):
    """Tempos stated in the filename, 60-200, most explicit first: '75bpm' forms, then bare
    numbers ('_100_', '(100)') in order. A model number can come first ('X-78 Beat 120'),
    so callers try each rather than trusting the first."""
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    out = []
    for g in [m.group(1) for m in _BPM_WORD.finditer(stem)] + _BPM_TOKEN.findall(stem):
        v = float(g)
        if TEMPO_MIN <= v <= TEMPO_MAX and v not in out:
            out.append(v)
    return out


def _filename_tempos(filename):
    """[(tempo, explicit)] stated in the filename, 60-200: explicit '75bpm' forms first,
    then bare numbers ('_100_', '(100)'), which may be take numbers ("Loop 072")."""
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    out, seen = [], set()
    for explicit, gs in ((True, [m.group(1) for m in _BPM_WORD.finditer(stem)]),
                         (False, _BPM_TOKEN.findall(stem))):
        for g in gs:
            v = float(g)
            if TEMPO_MIN <= v <= TEMPO_MAX and v not in seen:
                seen.add(v)
                out.append((v, explicit))
    return out


NAMED_BAR_COUNTS = _tunable("NAMED_BAR_COUNTS", (1, 2, 4, 6, 8, 12, 16, 32))


def _whole_bars(dur, bpm):
    """True if a loop is NAMED_BAR_COUNTS bars at bpm: a bare number in a name counts
    with 6 or 12 bars too ("LOOP_120_6x4"), not 3 or 5 (a take number fits those by chance)."""
    if not dur or not bpm:
        return False
    beats = dur * bpm / 60.0
    return any(abs(beats - 4 * n) <= BAR_TOL_BEATS for n in NAMED_BAR_COUNTS)


def _bar_fit(dur, bpm):
    """True if a loop of `dur` seconds is a whole number of 4/4 bars at `bpm`."""
    if not dur or not bpm:
        return False
    beats = dur * bpm / 60.0
    return any(abs(beats - 4 * n) <= BAR_TOL_BEATS for n in BAR_COUNTS)


def _resolve_tempo(filename, dur, sononym_bpm, librosa_bpm, extra=None):
    """(bpm, source) for a loop, or (None, None). The filename is the producer's intent
    and wins as written: an explicit '92bpm'
    always, a bare number when the loop is then a whole number of bars (any count), and
    never snapped to what the length implies (a reverb tail makes a 92 read as 90.06).
    Otherwise Sononym, then librosa, each also tried at x2 / x0.5 (octave errors), accepted
    only when the loop is then 1-16 whole bars, snapped to the length. `extra` (without
    Sononym only: _fallback_tempos): [(bpm, source, explicit)] tried when all of that finds
    nothing, an explicit one as stated, the others when the loop is then whole bars; but the
    tempo the WAV's ACID chunk states comes before librosa's estimate (a stated tempo beats
    an estimate), after the name's and the path's."""
    stated = [(bpm, src) for bpm, src, explicit in (extra or ()) if src in ("name", "acid") and explicit]
    if librosa_bpm and any(src == "acid" for _b, src in stated):
        got = _resolve_tempo_chain(filename, dur, sononym_bpm, None)
        if got[0] is None:
            return stated[0]          # the name's "bpm120" first, else the ACID chunk's
    got = _resolve_tempo_chain(filename, dur, sononym_bpm, librosa_bpm)
    if got[0] is None and extra:
        for bpm, src, explicit in extra:
            if explicit or _whole_bars(dur, bpm):
                return bpm, src
    return got


# tempos a name states in other ways than "120bpm" / "_120_" (without Sononym only): "bpm120"
# and "BPM 120" (explicit), a number glued to a word ("Groove96", "T116": whole bars)
_BPM_AFTER = re.compile(r"bpm[ _-]?(\d{2,3}(?:\.\d+)?)(?!\d)", re.I)
_BPM_GLUED = re.compile(r"(?<=[a-z])(\d{2,3})(?![0-9])", re.I)
_FOLDER_BPM = re.compile(r"^\s*(\d{2,3})\s*$")


def _fallback_tempos(r) -> list:
    """[(bpm, source, explicit)] a loop's own metadata states beyond what _resolve_tempo
    reads (without Sononym only; tried when the usual chain finds nothing): the name's
    "bpm120" (explicit) and "Groove96" / "T116" (whole bars), the WAV's ACID chunk (its tempo
    as stated, else beats over the length), and a folder named only by a number ("Loops/174/",
    whole bars). Each 60-200 BPM."""
    out = []
    fn = getattr(r, "filename", None) or os.path.basename(getattr(r, "path", None) or "")
    stem = os.path.splitext(os.path.basename(fn))[0]

    def add(v, src, explicit):
        v = round(float(v), 2)
        if TEMPO_MIN <= v <= TEMPO_MAX and all(v != o[0] for o in out):
            out.append((v, src, explicit))
    for m in _BPM_AFTER.finditer(stem):
        add(m.group(1), "name", True)
    acid, beats, dur = getattr(r, "acid_bpm", None), getattr(r, "acid_beats", None), getattr(r, "duration_s", None)
    if acid:
        add(acid, "acid", True)
    elif beats and dur:
        add(beats * 60.0 / dur, "acid", True)
    for m in _BPM_GLUED.finditer(stem):
        add(m.group(1), "name", False)
    rel = getattr(r, "rel_path", None) or library_rel(getattr(r, "path", None) or "")
    for seg in reversed([x for x in rel.split("/")[:-1] if x]):
        m = _FOLDER_BPM.match(seg)  # type: ignore[assignment]
        if m:
            add(m.group(1), "folder", False)
    got = _length_tempo(dur, getattr(r, "tempo_bpm", None) or getattr(r, "bpm", None))
    if got:
        add(got, "length", True)
    return out


# Without Sononym, last: the tempo a loop's length implies, when the analysis's tempo is near
# it. A loop of DERIVED_BAR_COUNTS whole bars at T BPM lasts bars x 240 / T seconds; an
# analyzer reads a little off (89.1 for an 8-bar loop at 88), which _bar_fit's tolerance
# rejects. When the analyzer's tempo (or its half or double) is within TEMPO_DERIVE_TOL of
# such a whole-bar tempo, that tempo is the loop's (source "length").
DERIVED_BAR_COUNTS = _tunable("DERIVED_BAR_COUNTS", (1, 2, 4, 8, 16))
TEMPO_DERIVE_TOL = _tunable("TEMPO_DERIVE_TOL", 0.03)


def _length_tempo(dur, measured):
    """The whole-bar tempo a loop's length implies near the measured tempo (see
    DERIVED_BAR_COUNTS), or None: the measured tempo itself first, then its double and
    half, the nearest when several bar counts fit."""
    if not dur or not measured or dur <= 0 or measured <= 0:
        return None
    best = None
    for k, mul in enumerate((1, 2, 0.5)):
        m = measured * mul
        for bars in DERIVED_BAR_COUNTS:
            t = bars * 240.0 / dur
            if not (TEMPO_MIN <= t <= TEMPO_MAX):
                continue
            d = abs(m - t) / t
            if d <= TEMPO_DERIVE_TOL and (best is None or (k, d) < best[0]):
                best = ((k, d), round(t, 2))
    return best[1] if best else None


def _resolve_tempo_chain(filename, dur, sononym_bpm, librosa_bpm):
    named = _filename_tempos(filename)
    for nb, explicit in named:
        if explicit:
            return nb, "name"
    for nb, _ in named:
        if _whole_bars(dur, nb):
            return nb, "name"
    for src, v in (("sononym", sononym_bpm), ("librosa", librosa_bpm)):
        if not v:
            continue
        for mul in (1, 2, 0.5):
            t = v * mul
            if TEMPO_MIN <= t <= TEMPO_MAX and _bar_fit(dur, t):
                return _snap_tempo(dur, round(t, 2)), src
    return None, None


def _snap_tempo(dur, bpm, tol=0.03):
    """The tempo the loop's length says, when within tol of bpm: a 13.714 s loop that
    Sononym calls 140.5 is 32 beats at 140.0 (an analyzer's tempo is often a little off).
    Whole bars, so the snapped tempo keeps the loop a whole number of bars."""
    if not dur or not bpm:
        return bpm
    bars = round(dur * bpm / 240.0)
    if bars <= 0:
        return bpm
    t = bars * 240.0 / dur
    return round(t, 2) if abs(t - bpm) <= tol * bpm else bpm


def _fold_tempo(bpm, window):
    """Fold a tempo by octaves into [lo, hi) (half/double time are equivalent)."""
    if not bpm or not window:
        return bpm
    lo, hi = window
    t = round(float(bpm), 2)  # round before folding so the result stays inside [lo, hi)
    if t <= 0:
        return bpm
    while t < lo:
        t *= 2
    while t >= hi:
        t /= 2
    return t


def _fold_in_range(bpm, folded):
    """The folded tempo, or the loop's own when the fold would leave the style's tempo range
    (TEMPO_FOLD_RANGE, set by the tempo knob for a range narrower than its top octave; None:
    every fold stands)."""
    if TEMPO_FOLD_RANGE and bpm and folded and not (TEMPO_FOLD_RANGE[0] <= folded <= TEMPO_FOLD_RANGE[1]):
        return bpm
    return folded


# ---------------------------------------------------------------------------
# PHRASES: musical loops (curate_config PHRASES)
# ---------------------------------------------------------------------------
def _phrase_marked(rel_path, filename, is_loop):
    """How its producer marks a file as a phrase: 'name' (a tempo in the filename), 'folder'
    (a loop classed as one, in a loop folder not narrowed below by a one-shot folder:
    "Loops/Chords/..." is not, "Loops/Music Loops/..." is), else None."""
    if BPM_NAME_RE.search(filename or ""):
        return "name"
    if not is_loop:
        return None
    ev = False
    for seg in os.path.dirname(rel_path or "").split("/"):
        if PHRASE_ONESHOT_DIR_RE.search(seg) and not PHRASE_DIR_STRONG_RE.search(seg):
            ev = False
        elif PHRASE_LOOP_DIR_RE.search(seg):
            ev = True
    return "folder" if ev else None


def _in_loop_folder(rel_path):
    """True if a file sits in a loop folder (the folder test of _phrase_marked)."""
    return _phrase_marked(rel_path, "", True) == "folder"


_GENRE_RX = {w: (re.compile(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", re.I), re.compile(p, re.I))
             for w, p in GENRE_WORDS.items()}


_INST_RX = {w: (re.compile(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", re.I), re.compile(p, re.I))
            for w, p in INSTRUMENT_WORDS.items()}


def _inst_backers(phrase):
    """For each instrument / source noun in a naming phrase (INSTRUMENT_WORDS), the pattern
    a file's path or Ableton tags must match to back it ("djembe hit" -> [djembe])."""
    return [rx for w, (word_rx, rx) in _INST_RX.items() if word_rx.search(phrase)]


def genre_evidence(category, rel, bpm=None, band=None):
    """What backs a folder's genre words for one file: its path, plus JUNGLE_TEMPO_TOKEN for
    a DRUMLOOPS break (not tops) at JUNGLE_BPM, which backs "jungle" only."""
    from .curate_config import JUNGLE_BPM, JUNGLE_TEMPO_TOKEN
    if category == "DRUMLOOPS" and band not in ("tops",) and bpm and JUNGLE_BPM[0] <= bpm <= JUNGLE_BPM[1]:
        return f"{rel} {JUNGLE_TEMPO_TOKEN}"
    return rel


def _genre_backers(phrase):
    """For each genre / drum-machine word in a naming phrase, the path pattern that must
    back it ("909 house kick" -> [909, house])."""
    return [path_rx for w, (word_rx, path_rx) in _GENRE_RX.items() if word_rx.search(phrase)]


def _phrase_support(V, T, allowed, topk=NAME_TOPK):
    """Share of files (rows of V) for which each allowed phrase (columns of T) is among
    their own top-k phrases. Returns {column: share}."""
    cols = np.where(allowed)[0]
    if not len(cols) or not len(V):
        return {}
    S = V @ T[cols].T
    k = min(topk, len(cols))
    top = np.argsort(-S, axis=1)[:, :k]
    share = np.bincount(top.ravel(), minlength=len(cols)) / len(V)
    return {int(q): float(share[j]) for j, q in enumerate(cols)}


def _folder_kmax(kmax, budget, n):
    """A small category gets about one folder per FOLDER_TARGET_FILES files (at least
    FOLDER_MIN); never more folders than kmax."""
    return min(kmax, max(FOLDER_MIN, int(round(min(budget, n) / FOLDER_TARGET_FILES))))


_VEL_WORDS = {"ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8,
              "soft": 2, "quiet": 2, "med": 5, "medium": 5, "hard": 7, "loud": 7}


def _velocity_rank(path):
    """How hard a multisample layer was played, from its name ("_v3_", "_ff_", "Soft"):
    higher is louder; None when the name doesn't say."""
    stem = os.path.splitext(os.path.basename(path or ""))[0].lower()
    for t in re.split(r"[\s_\-]+", stem):
        m = re.fullmatch(r"v(\d{1,2})", t)
        if m:
            return int(m.group(1))
        if t in _VEL_WORDS:
            return _VEL_WORDS[t]
    return None


def _prefer_loud_layers(rest, rec, room):
    """For a set whose names carry velocity layers, the loudest layer of each note first
    (a soft pizzicato or ppp hit is dull and quiet once exported): one per note spread
    by pitch, then the next-loudest layers if room is left. Sets without velocity words
    are returned unchanged. rest is sorted by pitch."""
    vel = {i: _velocity_rank(rec[i]["path"]) for i in rest}
    if all(v is None for v in vel.values()):
        return None
    notes = defaultdict(list)
    for i in rest:
        t = rec[i]["tune"]
        notes[("t", int(round(t))) if t is not None else ("i", i)].append(i)
    best = [max(g, key=lambda i: (vel[i] if vel[i] is not None else -1, rec[i]["path"])) for g in notes.values()]
    best.sort(key=lambda i: (rec[i]["tune"] is None, rec[i]["tune"] or 0, rec[i]["path"]))
    if room < len(best):
        pos = sorted({int(round(x)) for x in np.linspace(0, len(best) - 1, room)})
        return [best[p] for p in pos]
    more = sorted((i for i in rest if i not in set(best)),
                  key=lambda i: (-(vel[i] if vel[i] is not None else -1), rec[i]["path"]))
    return best + more[:room - len(best)]


def _spread_pick(rest, rec, room):
    """`room` of a set's files: the loudest layer of each note (_prefer_loud_layers), else
    spread across the register. rest is sorted by pitch."""
    if room >= len(rest):
        return list(rest)
    pick = _prefer_loud_layers(rest, rec, room)
    if pick is None:
        pos = sorted({int(round(x)) for x in np.linspace(0, len(rest) - 1, room)})
        pick = [rest[p] for p in pos]
    return pick


def _cap_siblings_fill(rest, rec, room):
    """`room` more of a set's files (rest sorted by pitch): C notes first, then spread."""
    cs = [i for i in rest if rec[i].get("name_pc") == 0]
    pick = _spread_pick(cs, rec, min(room, len(cs))) if cs else []
    if len(pick) < room:
        pick += _spread_pick([i for i in rest if i not in set(pick)], rec, room - len(pick))
    return pick


def _cap_siblings(idxs, rec, cap=SIBLING_CAP):
    """At most `cap` files of one multisample set (_sibling_key) in a folder, spread
    across the set's register (lowest, highest, then between), loudest velocity layer
    first (_prefer_loud_layers). A Keep always stays; files that belong to no set are
    untouched."""
    sets, keep = defaultdict(list), []
    for i in idxs:
        k = _sibling_key(rec[i]["path"])
        (sets[k] if k else keep).append(i)
    for g in sets.values():
        pins = [i for i in g if rec[i].get("pin")]
        rest = sorted((i for i in g if not rec[i].get("pin")),
                      key=lambda i: (rec[i]["tune"] is None, rec[i]["tune"] or 0, rec[i]["path"]))
        room = max(0, cap - len(pins))
        # the previous build's picks of the set first (stable picks), then as below
        inc = [i for i in rest if rec[i].get("incumbent")]
        if inc and 0 < room < len(rest):
            ipick = _spread_pick(inc, rec, min(room, len(inc)))
            left = room - len(ipick)
            rest2 = [i for i in rest if i not in set(ipick)]
            keep += pins + ipick + (_cap_siblings_fill(rest2, rec, left) if left > 0 else [])
            continue
        # a set's C notes first (retuned to C anyway: they need no shift)
        cs = [i for i in rest if rec[i].get("name_pc") == 0]
        if cs and 0 < room < len(rest):
            pick = _spread_pick(cs, rec, min(room, len(cs)))
            left = room - len(pick)
            if left > 0:
                pick += _spread_pick([i for i in rest if i not in set(pick)], rec, left)
        elif room >= len(rest):
            pick = rest
        elif room == 0:
            pick = []
        else:
            pick = _spread_pick(rest, rec, room)
        keep += pins + pick
    return sorted(keep)


def _phrase_near_text(rel_path, filename):
    """What says what a loop holds: its filename stem, the folder it sits in and any loop
    folder above ("Drum Loops", "vox loops"), with genre names ("Drum & Bass") removed.
    Other folders are pack and genre names ("Pack A", "Funk")."""
    segs = [x for x in os.path.dirname(rel_path or "").split("/") if x]
    near = segs[-1:] + [x for x in segs[:-1] if PHRASE_LOOP_DIR_RE.search(x)]
    near.append(os.path.splitext(os.path.basename(filename or ""))[0])
    return PHRASE_GENRE_RE.sub(" ", "/".join(near))


def _is_phrase(r, fallback=False):
    """True if a row (rel_path, path, filename, duration_s, ableton_tags, classes, categories,
    bpm, tempo_bpm; harmonicity and onset_rate_hz with `fallback`) is a musical phrase: marked
    by its producer, a whole number of bars at its tempo, PHRASE_DUR long, and musical (not
    drums, FX, a voice or a sample chain). `fallback` (no Sononym: the built-in providers
    classify): a file they call a drum loop isn't one (_providers_drum_loop)."""
    path = getattr(r, "path", None) or ""
    rel = getattr(r, "rel_path", None) or library_rel(path)
    fn = getattr(r, "filename", None) or os.path.basename(path)
    dur = getattr(r, "duration_s", None)
    if not dur or not (PHRASE_DUR[0] <= dur <= PHRASE_DUR[1]):
        return False
    if _phrase_marked(rel, fn, _is_loop_row(r)) is None:
        return False
    if PHRASE_NOT_RE.search(_phrase_near_text(rel, fn)):
        return False
    if (_is_ir(rel) or _is_preview(rel) or DEMO_PATH.search(rel)
            or WAVE_PATH_RE.search(os.path.dirname(rel)) or _pack_of(rel, path) in PACK_HOME):
        return False
    ab = _as_list(getattr(r, "ableton_tags", None))
    fams = {_AB_FAMILY[t] for t in ab if t in _AB_FAMILY}
    if _has_drum_tag(ab) or fams & {"drum", "vocal"} or fams == {"fx"}:
        return False
    if fallback and _providers_drum_loop(r, ab):
        return False            # path and audio call it a drum loop (without Sononym only)
    son = _as_list(getattr(r, "categories", None))
    if "tonal" not in fams and son and (not any(s.startswith("Tone") for s in son)
                                        or "Tone Voice & Acapella" in son):
        return False            # Sononym hears drums, FX or a voice, and Ableton no instrument
    bpm, _src = _resolve_tempo(fn, dur, getattr(r, "bpm", None), getattr(r, "tempo_bpm", None),
                               _fallback_tempos(r) if fallback else None)
    return bpm is not None


def _phrase_inst_label(name, ab_tags=()):
    """The instrument a phrase's filename names ("Riff_120_A_Guitar" -> "guitar"), else None."""
    stem = os.path.splitext(os.path.basename(name or ""))[0]
    for rx, lab in PHRASE_INSTRUMENTS:
        if rx.search(stem):
            return lab
    return None


def _phrase_role(filename, rel_path=None, ab_tags=(), son_labels=()):
    """PHRASES band: 'acid' (303 lines), 'bass', 'live' (played instruments: guitar, strings, brass, winds),
    'chords' (keys, pads, chord progressions) or 'lead'. The filename decides first, then
    the folder it sits in, then Ableton's tag, then Sononym's label."""
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    folder = os.path.basename(os.path.dirname(rel_path or ""))
    for text_ in (stem, folder):
        if PHRASE_ACID_RE.search(text_):
            return "acid"
        if PHRASE_BASS_RE.search(text_) and not re.search(r"bassoon", text_, re.I):
            return "bass"
        if ACOUSTIC_NAME_RE.search(text_) and not re.search(r"synth", text_, re.I):
            return "live"
        if PHRASE_CHORDS_RE.search(text_):
            return "chords"
    tags = set(ab_tags or ())
    if tags & set(PHRASE_BASS_TAGS):
        return "bass"
    # (a "SynthRiff" Ableton hears as a guitar is still a synth)
    if any(t in ACOUSTIC_ABLETON_TAGS and t not in GENERIC_ACOUSTIC_TAGS for t in tags) \
            and not re.search(r"synth", stem + "/" + folder, re.I):
        return "live"
    if tags & set(PHRASE_CHORDS_TAGS):
        return "chords"
    son = list(son_labels or ())
    if any(s.startswith("Tone Bass") for s in son):
        return "bass"
    if any(s.startswith("Tone Pads") for s in son):
        return "chords"
    return "lead"


def _tempo_band_groups(bpms, edges, min_n):
    """Record indices grouped into tempo bands [lo, hi) in tempo order, unknown tempo
    last. A band with fewer than min_n members merges into the neighbor whose median
    tempo is nearer (a thin 130-150 tops band joins 125-130 techno, not 150+ dnb)."""
    bands = [[lo, hi, []] for lo, hi in zip(edges[:-1], edges[1:])]
    unknown = []
    for i, t in enumerate(bpms):
        for b in bands:
            if t is not None and b[0] <= t < b[1]:
                b[2].append(i)
                break
        else:
            unknown.append(i)
    bands = [b for b in bands if b[2]]
    while len(bands) > 1:
        small = [j for j, b in enumerate(bands) if len(b[2]) < min_n]
        if not small:
            break
        j = min(small, key=lambda x: (len(bands[x][2]), x))
        med = lambda x: float(np.median([bpms[i] for i in bands[x][2]]))
        k = min((x for x in (j - 1, j + 1) if 0 <= x < len(bands)),
                key=lambda x: (abs(med(x) - med(j)), len(bands[x][2]), x))
        a, b = sorted((j, k))
        bands[a] = [bands[a][0], bands[b][1], bands[a][2] + bands[b][2]]
        del bands[b]
    out = [b[2] for b in bands]
    if unknown:
        out.append(unknown)
    return out


def _cap_folders(groups, keys, P, cap, min_per_key=1):
    """Merge the smallest folder into the nearest (CLAP centroid) folder with the same
    key until at most `cap` remain. A None key is never merged (tempo folders keep their
    BPM label true); a key down to `min_per_key` folders is kept (a band never
    disappears, and keeps its guaranteed folders)."""
    groups = [list(g) for g in groups]
    keys = list(keys)
    while len(groups) > cap:
        cents = [P[g].mean(0) for g in groups]
        cents = [c / (np.linalg.norm(c) + 1e-9) for c in cents]
        order = sorted(range(len(groups)), key=lambda j: (len(groups[j]), j))
        done = False
        for i in order:
            mates = [j for j in range(len(groups)) if j != i and keys[j] is not None and keys[j] == keys[i]]
            if not mates or len(mates) + 1 <= min_per_key:
                continue
            j = max(mates, key=lambda m: (float(cents[i] @ cents[m]), -m))
            groups[j] = sorted(groups[j] + groups[i])
            del groups[i]; del keys[i]
            done = True
            break
        if not done:
            break
    return groups


def _merge_small_groups(groups, bands, P, min_n, keep_per_band=1, measure=None):
    """Merge each group under min_n records into the nearest group (CLAP centroid) of the
    same band, smallest first, keeping at least keep_per_band groups a band. `measure`
    (groups, bands) -> sizes judges a group by something other than its record count
    (the files the budget will give it)."""
    groups, bands = [list(g) for g in groups], list(bands)
    stuck = set()
    while True:
        size = measure(groups, bands) if measure else [len(g) for g in groups]
        small = [j for j, g in enumerate(groups) if size[j] < min_n and j not in stuck]
        if not small:
            return groups
        j = min(small, key=lambda x: (size[x], x))
        mates = [m for m in range(len(groups)) if m != j and bands[m] == bands[j]]
        if not mates or len(mates) + 1 <= keep_per_band:
            stuck.add(j)
            continue
        cj = P[groups[j]].mean(0); cj = cj / (np.linalg.norm(cj) + 1e-9)

        def _cos(m):
            cm = P[groups[m]].mean(0)
            return float(cj @ (cm / (np.linalg.norm(cm) + 1e-9)))
        m = max(mates, key=lambda x: (_cos(x), -x))
        groups[m] = sorted(groups[m] + groups[j])
        del groups[j]; del bands[j]
        stuck = {s if s < j else s - 1 for s in stuck if s != j}


_PREV_FAMS: dict = {}
STICKY_SHARE = _tunable("STICKY_SHARE", 0.5)


def _previous_families(category):
    """{family: set(source path)} for a category in this library's latest archived build
    (builddiff.latest_archived), or {} with none: each source where it is now when a library
    walk found it moved since (_moved_paths), so a renamed library folder keeps the same
    picks. Cached per process."""
    if "doc" not in _PREV_FAMS:
        doc = None
        try:
            from .builddiff import latest_archived
            last = None if os.environ.get("FOURIER_NO_STICKY") else latest_archived()
            if last is not None:
                doc = manifests.read(last)
        except Exception:
            doc = None
        _PREV_FAMS["doc"] = doc
        _PREV_FAMS["moves"] = _moved_paths() if doc else {}
    doc = _PREV_FAMS["doc"] or {}
    now = _PREV_FAMS.get("moves") or {}
    out = defaultdict(set)
    for e in ((doc.get("categories") or {}).get(category) or {}).get("entries", []):
        out[e["family"]].add(now.get(e["src"], e["src"]))
    return out


def _moved_paths() -> dict:
    """{old path: where it is now} for the samples a library walk found moved (the
    missing_files rows' moved_to, importer._match_moves), a chain of moves followed to its
    end. Empty when no walk recorded a move (a library Sononym scans: never walked)."""
    try:
        from ..db.session import session_scope
        with session_scope() as s:
            pairs = dict(s.execute(text(  # type: ignore[arg-type]
                "SELECT path, moved_to FROM missing_files WHERE moved_to IS NOT NULL")).all())
    except Exception:
        return {}
    out = {}
    for old in pairs:
        cur, seen = old, {old}
        while cur in pairs and pairs[cur] not in seen:
            cur = pairs[cur]
            seen.add(cur)
        if cur != old:
            out[old] = cur
    return out


def _previous_outnames(category):
    """{source path: file name} for a category in the latest archived build (stable picks),
    by where each source is now (_previous_families)."""
    _previous_families(category)                     # loads the archived build once
    doc = _PREV_FAMS.get("doc") or {}
    now = _PREV_FAMS.get("moves") or {}
    return {now.get(e["src"], e["src"]): os.path.basename(e["out"])
            for e in ((doc.get("categories") or {}).get(category) or {}).get("entries", [])}


_BPM_IN_NAME = re.compile(r"(?<!\d)(?:[6-9]\d|1\d\d|2[0-4]\d)(?!\d)")


def _kept_stem(stem, limit, bpm=None):
    """names = "keep": the source name as it is (FAT-safe, cut in the middle to limit), with
    a loop's BPM added when the name has no tempo-like number."""
    from ..devices.exporter import _balance_brackets, _sanitize_filename
    if bpm and not _BPM_IN_NAME.search(stem):
        stem = f"{stem} {int(round(float(bpm)))}bpm"
    return _balance_brackets(_sanitize_filename(stem, limit))


def _namer(bpm=None):
    """How a file's name is made from its source stem (the names knob, NAMES)."""
    if NAMES == "keep":
        return lambda stem, limit: _kept_stem(stem, limit, bpm)
    return canonical_stem


def _own_names(stem, limit, namer=canonical_stem, alt=None):
    """The names a file of this stem may carry: its stem as named, or it with _2, _3 ...
    alt (a note name, lowercase, "as" for A#): a retuned file whose name now ends in the
    note it plays, where "808_C_2" would read as C2: _from-as, then _from-as_2, _from-as_3 ...
    (lowercase, so no name rule reads it as a note)."""
    base = namer(stem, limit)
    if alt:
        sfx = f"_from-{alt}"

        def kth(k):
            tail = sfx if k == 2 else f"{sfx}_{k - 1}"
            return f"{namer(stem, limit - len(tail))}{tail}"
        return base, kth
    return base, lambda k: f"{namer(stem, limit - len(str(k)) - 1)}_{k}"


def _choose_outname(stem, taken, prev_name=None, limit=None, namer=canonical_stem, alt=None):
    """A file's name in its folder (`taken`: the names already there, lowercased; APFS and
    FAT32 cards fold case). Its previous build's name when that is still one of its own names
    and free, else the canonical stem, else stem_2, stem_3 ... (alt: stem_from-<note>, ...).
    So a name is never given to a different sample while its owner is still in the folder."""
    limit = limit or STEM_MAX
    base, kth = _own_names(stem, limit, namer, alt)
    if prev_name and prev_name.lower() not in taken:
        pstem = os.path.splitext(prev_name)[0]
        if alt:
            if pstem == base or any(pstem == kth(k) for k in range(2, 12)):
                return prev_name
        else:
            m = re.search(r"_(\d+)$", pstem)
            if pstem == base or (m and int(m.group(1)) >= 2 and pstem == kth(int(m.group(1)))):
                return prev_name
    out, k = f"{base}.wav", 2
    while out.lower() in taken:
        out, k = f"{kth(k)}.wav", k + 1
    return out


def _note_word(stem, bare=False) -> str | None:
    """The note a file's own name says ("A#" in "808_A#", "F" in "808_F" with bare), as a
    word no name rule reads as a note: lowercase, a sharp as "s" ("as", "f", "eb")."""
    probe = (stem or "").replace("_", " ")
    for rx in (_NOTE_TOKEN, _CHORD_TOKEN, _KEY_TOKEN, *((_BARE_NOTE,) if bare else ())):
        m = list(rx.finditer(probe))
        if not m:
            continue
        g = m[-1]
        if rx is _NOTE_TOKEN and not g.group(1):
            letter, acc = g.group(4).upper(), g.group(5)
        else:
            letter, acc = g.group(1), g.group(2)
        return letter.lower() + ("s" if acc in ("#", "s") else "b" if acc == "b" else "")
    return None


def _name_still_true(name, c, rels, texts, category):
    """May a folder keep an older name: every envelope word still holds (TRAIT_GATES), every
    genre / machine / instrument word is backed by its files, no excluded word, short enough."""
    from .naming import TRAIT_GATES, trait_ok
    from .curate_config import NAME_EXCLUDE_WORDS
    if len(name) > FAMILY_NAME_MAX:
        return False
    toks = name.lower().split("-")
    joined = " ".join(toks)
    for w in TRAIT_GATES:
        if w in name.lower() and w in (joined.replace(" ", "-") if "-" in w else toks) and not trait_ok(w, c):
            return False
    if set(toks) & NAME_EXCLUDE_WORDS.get(category, set()):
        return False
    from .naming import NOUN_DROP
    if set(toks) & NOUN_DROP.get(category, set()):
        return False                   # an older name still carrying the category noun
    for w, (word_rx, path_rx) in _GENRE_RX.items():
        if word_rx.search(joined) and np.mean([bool(path_rx.search(r)) for r in rels]) < GENRE_SUPPORT_MIN:
            return False
    for w, (word_rx, rx) in _INST_RX.items():
        if word_rx.search(joined) and np.mean([bool(rx.search(t)) for t in texts]) < GENRE_SUPPORT_MIN:
            return False
    return True


_LEAD_RE = re.compile(r"^(?:(?:[a-z]+-)?(?:\d{3}(?:-\d{3})?bpm|freetempo)(?:-tops)?|classic-breaks|[a-z]+-[a-g]s?\d(?=-|$))")


def _name_lead(name):
    """A folder name's lead: role and tempo ("live-120-135bpm", "125-130bpm-tops"), band
    and note ("mallet-c5"), else its first word ("ride", "closed")."""
    m = _LEAD_RE.match(name or "")
    return m.group(0) if m else (name or "").split("-")[0]


def _apply_sticky_names(clusters, category, rec, cfg):
    """Keep a folder's name from the previous build when at least STICKY_SHARE of its files
    came from that folder, the lead (tempo / band / note) is unchanged and the name is
    still true (_name_still_true) and unique: names stay put across rebuilds (dropping one
    folder can change the traits of many of the others)."""
    from .naming import name_collides, twin_names
    prev = _previous_families(category)
    if not prev:
        return 0
    led = bool(cfg.get("band") or cfg.get("tempo_bands") or category in NOTE_LEAD_CATS
               or cfg.get("kind") == "instrument")
    cand = []
    for j, c in enumerate(clusters):
        srcs = {rec[i]["path"] for i in (c.get("_pick") or c["idxs"])}
        for fam, old in prev.items():
            sh = len(srcs & old) / max(1, len(srcs))
            if sh >= STICKY_SHARE and fam != c["name"]:
                cand.append((sh, j, fam))
    n = 0
    taken_old = set()
    for sh, j, fam in sorted(cand, key=lambda t: (-t[0], t[1], t[2])):
        c = clusters[j]
        if c.get("sticky") or fam in taken_old:
            continue
        if led and _name_lead(fam) != _name_lead(c["name"]):
            continue                   # a different band, tempo or note: the name would lie
        others = [clusters[k]["name"].lower() for k in range(len(clusters)) if k != j]
        if name_collides(fam, others) or any(twin_names(fam, o) for o in others):
            continue                   # (a kept name one word off a sibling's reads as it)
        pk = c.get("_pick") or c["idxs"]
        paths = [library_rel(rec[i]["path"] or "") for i in pk]
        rels = [genre_evidence(category, r_, rec[i].get("bpm_fold") or rec[i].get("bpm"), rec[i].get("band"))
                for r_, i in zip(paths, pk)]
        texts = [r_ + " | " + " ".join(rec[i].get("ab") or []) for r_, i in zip(paths, pk)]
        if not _name_still_true(fam, c, rels, texts, category):
            continue
        c["name"], c["sticky"] = fam, True
        taken_old.add(fam)
        n += 1
    return n


def _split_capped_groups(groups, P, planned, ceil, fmax, rec, max_splits=6):
    """Split (KMeans, 2) the largest group the budget would fill to `ceil` while its pool
    holds at least twice that, until none is left or fmax folders; both halves must be
    at least FOLDER_MIN_FILES. Bands stay whole (a half keeps its group's records)."""
    from sklearn.cluster import KMeans
    groups = [list(g) for g in groups]
    # a previous build's folder, kept whole (stable picks), isn't split again
    stuck = set()
    for g in groups:
        pf = Counter(rec[i].get("prev_family") for i in g if rec[i].get("prev_family"))
        if pf and pf.most_common(1)[0][1] >= max(FOLDER_MIN_FILES, 0.8 * sum(pf.values())):
            stuck.add(tuple(g))
    for _ in range(max_splits):
        if len(groups) >= fmax:
            break
        bands = [Counter(rec[i].get("band") for i in g).most_common(1)[0][0] if g else None for g in groups]
        plan = planned(groups, bands)
        big = [j for j, a in enumerate(plan) if a >= ceil and len(groups[j]) >= 2 * ceil
               and tuple(groups[j]) not in stuck]
        if not big:
            break
        j = max(big, key=lambda j: (len(groups[j]), -j))
        g = groups[j]
        with _limit_threads():
            lab = KMeans(n_clusters=2, n_init=4, random_state=CURATION_SEED).fit(P[g]).labels_
        a = [g[k] for k in range(len(g)) if lab[k] == 0]
        b = [g[k] for k in range(len(g)) if lab[k] == 1]
        if min(len(a), len(b)) < FOLDER_MIN_FILES:
            stuck.add(tuple(g))
            continue
        groups[j:j + 1] = [sorted(a), sorted(b)]
    return groups


def _cluster_by_tempo(P, bpms, edges, floor, kmax=30, kmin=8, kdiv=800, files=None, per_folder=None,
                      seedfn=None, band_min=None):
    """Sound clusters WITHIN tempo bands. The total family count follows the usual
    _cluster rule and is split across bands by size; or, given `files` (the files these
    records will yield) and `per_folder`, a band gets one folder per `per_folder` files it
    will hold (narrow tempo bands: one folder each unless it's big). A tempo band keeps its
    own folder from band_min candidates (TEMPO_BAND_MIN; less in a library-scaled build)."""
    n = len(P)
    k0 = int(min(kmax, max(kmin, round(n / kdiv))))
    groups = []
    for idx in _tempo_band_groups(bpms, edges, TEMPO_BAND_MIN if band_min is None else band_min):
        nb = len(idx)
        if files and per_folder:
            kb = max(1, min(int(np.ceil(files * nb / n / per_folder)), nb // max(floor, 1)))
        else:
            kb = max(1, min(int(round(k0 * nb / n)), nb // max(floor, 1)))
        if kb <= 1:
            groups.append(list(idx))
            continue
        init = seedfn(list(idx)) if seedfn else None
        if init is not None and len(init[0]) != kb:
            init = None
        sub, _ = _cluster(P[idx], kmax=kb, kmin=kb, kdiv=1, init=init)
        groups.extend([[idx[i] for i in g] for g in sub])
    return groups, k0


# ---------------------------------------------------------------------------
# Multisample siblings
# ---------------------------------------------------------------------------
# Notes / velocity layers / numbered takes of one multisample set are interchangeable
# for routing: a verdict on one covers the set. Siblings share a pack and a name once
# note, velocity and number tokens are removed, and sound at least loosely alike
# (SIBLING_COS: real siblings span a wide range of CLAP similarity, so the name carries
# the decision and CLAP only stops an unrelated sound).
_SIB_NOTE = re.compile(r"^[a-g](?:#|b|s)?-?\d$")
_SIB_VAR = re.compile(r"^(?:v|vel|rr|take|layer)?\d+$|^(?:ppp|pp|p|mp|mf|f|ff|fff)$")
_SIB_GENERIC = set("""kick kicks snare snares hihat hihats clap claps perc percs cymbal crash ride shaker
conga bongo snap drum drums beat break glitch zap blip click impact riser sweep whoosh texture drone
atmos ambience foley noise sample samples sound sounds loop loops bass pads lead leads synth stab stabs
chord chords pluck keys piano organ bell bells voice vocal choir string strings brass horn tone note
oneshot shot wave cycle table pulse square sine tri triangle arp hits""".split())
SIBLING_COS = _tunable("SIBLING_COS", 0.75)


def _gather_sets(clusters, rec, ceil, min_left=None, set_cap=None):
    """Move a sample set's stragglers (curate._sibling_key) into the folder holding most of
    the set, within one band, while it has room (ceil), holds fewer than set_cap of the set
    (SIBLING_CAP where that applies) and the folder left keeps min_left files
    (FOLDER_MIN_FILES). Updates each cluster's _pick and idxs; returns files moved."""
    from .curate_config import FOLDER_MIN_FILES
    min_left = FOLDER_MIN_FILES if min_left is None else min_left
    where = defaultdict(lambda: defaultdict(list))        # (band, key) -> {cluster j: [i]}
    for j, c in enumerate(clusters):
        for i in c.get("_pick") or []:
            k = _sibling_key(rec[i]["path"])
            if k is not None:
                where[(c.get("band"), k)][j].append(i)
    moved = 0
    for (_, _), by_c in sorted(where.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if len(by_c) < 2:
            continue
        home = max(by_c, key=lambda j: (len(by_c[j]), len(clusters[j]["_pick"]), -j))
        dst = clusters[home]
        for j in sorted(by_c):
            if j == home:
                continue
            src = clusters[j]
            for i in by_c[j]:
                if len(dst["_pick"]) >= ceil or len(src["_pick"]) - 1 < min_left \
                        or (set_cap is not None and len(by_c[home]) >= set_cap):
                    break
                by_c[home].append(i)
                src["_pick"].remove(i)
                dst["_pick"].append(i)
                if i in src["idxs"]:
                    src["idxs"] = [x for x in src["idxs"] if x != i]
                if i not in dst["idxs"]:
                    dst["idxs"] = list(dst["idxs"]) + [i]
                moved += 1
    return moved


def _sibling_key(path):
    """(pack, name minus note/velocity/number tokens), or None if the remaining name
    is too generic to identify a set ("Kick 01" -> "kick")."""
    stem = os.path.splitext(os.path.basename(path or ""))[0].lower()
    # a MIDI note number glued to the name ("60Pad_Chord_C3") isn't the name
    toks = [re.sub(r"^\d+(?=[a-z])", "", t) for t in re.split(r"[\s_\-]+", stem) if t]
    # nor is a trailing ID token after a velocity
    toks = [t for i, t in enumerate(toks)
            if not (i and re.fullmatch(r"[a-z0-9]{4}", t) and re.fullmatch(r"\d{2,3}", toks[i - 1]))]
    kept = [t for t in toks if not _SIB_NOTE.match(t) and not _SIB_VAR.match(t)]
    had_note = any(_SIB_NOTE.match(t) for t in toks)
    if not kept:
        return None
    if len(kept) < 2 and not had_note:
        # one word and a number is a set when the word is a patch name ("Frobnik 47",
        # "Glimmox_12"), not a generic word ("Kick 01", "Pad 3")
        if not (any(_SIB_VAR.match(t) for t in toks) and len(kept[0]) >= 4
                and kept[0] not in _SIB_GENERIC):
            return None
    # scope = the pack (vendor/pack), so stereo/mono twin folders and multisamples split
    # across subfolders are one set; outside the library fall back to the folder
    scope = _pack_key2(path) if in_library(path) else os.path.dirname(path or "")
    return (scope, " ".join(kept))


class _MisfiledIndex:
    """Misfiled ratings expanded to multisample siblings. categories(path, vec) returns
    every category the file or a sibling of it was rated Misfiled in (a scattered set can
    be misfiled in two folders at once); category() returns one of them, else None."""

    def __init__(self, misfiled, vec_of=lambda src: None):
        self.exact = dict(misfiled)
        self.by_key = defaultdict(list)
        for src, cat in misfiled.items():
            k = _sibling_key(src)
            if k is not None:
                self.by_key[k].append((cat, vec_of(src)))

    def categories(self, path, vec=None):
        out = set()
        c = self.exact.get(path)
        if c is not None:
            out.add(c)
        for cat, v in self.by_key.get(_sibling_key(path), ()):
            if v is None or vec is None or float(v @ vec) >= SIBLING_COS:
                out.add(cat)
        return out

    def category(self, path, vec=None):
        c = self.exact.get(path)
        if c is not None:
            return c
        cats = self.categories(path, vec)
        return min(cats) if cats else None


def _misfiled_index(misfiled, emb_n, id2row, pid=None):
    """Build a _MisfiledIndex; `pid` ({path: sample id}) avoids a DB lookup."""
    if not misfiled:
        return _MisfiledIndex({})
    if pid is None:
        from ..db.session import session_scope
        with session_scope() as s:
            pid = {p: i for i, p in s.execute(
                select(Sample.id, Sample.path).where(Sample.path.in_(list(misfiled))))}

    def vec_of(src):
        i = pid.get(src)
        return emb_n[id2row[i]] if i is not None and i in id2row else None
    return _MisfiledIndex(misfiled, vec_of)


def _row_override(r):
    """NAME_OVERRIDES rule for a candidate row. Overrides target single instrument
    sounds, so loops are exempt (a 'kalimba' drum loop is not a tine), and so are synth
    patches named for an instrument: a synth / drum-machine pack's (an FM "Marimba"
    patch) or one Ableton tags as a synth voice ("Synth Mallets")."""
    if _is_loop_row(r) or _is_synth_tagged(_as_list(getattr(r, "ableton_tags", None))) \
            or ORCH_PACK_EXCLUDE.search(_pack_of(getattr(r, "rel_path", None), getattr(r, "path", None) or "")):
        return None, None
    return _name_override_rule(r.filename or "")


_DUR_CAP_SLACK_S = 0.05          # verify's allowance over DUR_CAP (rounding, fades)


def _exported_seconds(path, category):
    """How long a file comes out of the export DSP for `category` (its trims), in seconds;
    the source's length when it can't be read."""
    import soundfile as sf
    from .curate_config import CATEGORIES as _CATS
    cfg = _CATS[category]
    from ..safety import ensure_local
    ensure_local([path])                 # a cloud-only file is downloaded before it's read
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception:
        try:
            return sf.info(path).duration
        except Exception:
            return 0.0
    norm_mode, (trim_lead, trim_tail) = _export_policy(cfg["kind"])
    y = _process_audio(np.asarray(data, dtype="float64"), sr, cfg["kind"], trim_lead, trim_tail,
                       norm_mode, lead_floor_db=cfg.get("trim_lead_db"), mono=category in MONO_CATS)
    return len(y) / sr


def _with_pins(idxs, want, pinned, pick):
    """Pinned indices are always selected; `pick(rest, n)` fills the remaining slots."""
    pins = [i for i in idxs if i in pinned]
    rest = [i for i in idxs if i not in pinned]
    need = want - len(pins)
    return pins + (pick(rest, need) if rest and need > 0 else [])


def _spread_keep(P, want, inc=None, cost=None):
    """Positions (into P) of `want` CLAP-spread picks: KMeans into `want` groups, each group's
    point closest to its centre (or lowest `cost(p, centre)`). With incumbents (`inc`, a bool
    per point: files the previous build kept; without them a pool change of a few files
    can re-pick a large share of the library, because KMeans groups shift with the pool):
    more incumbents than slots, the most spread-out of them; fewer, all of them, and the other slots go to the
    groups no incumbent covers, largest first, so newcomers fill gaps."""
    from sklearn.cluster import KMeans
    n = len(P)
    if want <= 0:
        return []
    if want >= n:
        return list(range(n))
    cost = cost or (lambda p, centre: float(np.linalg.norm(P[p] - centre)))

    def fit(ps, k):
        with _limit_threads():
            return KMeans(n_clusters=k, random_state=CURATION_SEED, n_init=3).fit(P[ps])

    def medoids(ps, k):
        if k >= len(ps):
            return list(ps)
        km = fit(ps, k)
        out = []
        for sc in range(k):
            mem = [ps[q] for q in range(len(ps)) if km.labels_[q] == sc]
            if mem:
                out.append(min(mem, key=lambda p: cost(p, km.cluster_centers_[sc])))
        return out

    kept = [p for p in range(n) if inc is not None and inc[p]]
    if not kept:
        return medoids(list(range(n)), want)
    if len(kept) >= want:
        return medoids(kept, want)
    rest = [p for p in range(n) if not inc[p]]
    need = want - len(kept)
    km = fit(list(range(n)), want)
    covered = {int(km.labels_[p]) for p in kept}
    size = Counter(int(km.labels_[p]) for p in rest)
    out = []
    for sc in sorted((sc for sc in range(want) if sc not in covered), key=lambda sc: (-size[sc], sc)):
        if len(out) >= need:
            break
        mem = [p for p in rest if km.labels_[p] == sc]
        if mem:
            out.append(min(mem, key=lambda p: cost(p, km.cluster_centers_[sc])))
    if len(out) < need:
        taken = set(out)
        out += medoids([p for p in rest if p not in taken], need - len(out))
    return kept + out


def _allocs_for_incumbents(allocs, clusters, rec, sizes, floor, ceil, keep_bands=()):
    """Move folder slots toward the previous build's files: a folder holding more
    incumbents than its allocation gets up to that many (within `ceil` and its pool), paid
    for by newcomer slots of folders in the same band (never below their incumbents or
    `floor`), so the category and each band keep their totals (budgets, band shares)."""
    allocs = list(allocs)
    inc = [sum(1 for i in c["idxs"] if rec[i].get("incumbent") and rec[i]["clip"] == 0) for c in clusters]
    groups = defaultdict(list)
    for j, c in enumerate(clusters):
        if c.get("band") not in keep_bands:
            groups[c.get("band")].append(j)
    for js in groups.values():
        for j in sorted(js, key=lambda j: (-(inc[j] - allocs[j]), j)):
            want = min(inc[j], ceil, sizes[j])
            while allocs[j] < want:
                donors = [d for d in js if d != j and allocs[d] > max(inc[d], floor)]
                if not donors:
                    break
                d = max(donors, key=lambda d: (allocs[d] - max(inc[d], floor), -d))
                allocs[d] -= 1
                allocs[j] += 1
    return allocs


def _medoid_cost(i, centre, rec, P):
    return (float(np.linalg.norm(P[i] - centre)) - 0.03 * (rec[i]["sup"] - 1)
            - QUAL_WEIGHT * rec[i]["qual"] - FAVOR_WEIGHT * rec[i]["fav"])


def _medoids(idxs, want, rec, P):
    """`want` files spread over idxs (clipped files only when nothing else is left)."""
    return _stable_pick(idxs, want, rec, P, stable=False)


def _stable_pick(idxs, want, rec, P, stable=True):
    """`want` files of idxs by _spread_keep, nudged toward corroborated, higher-quality and
    favored sources; the previous build's files ("incumbent") first when `stable`."""
    pool = [i for i in idxs if rec[i]["clip"] == 0] or list(idxs)
    if want <= 0:
        return []
    Q = P[pool]
    inc = [bool(rec[i].get("incumbent")) for i in pool] if stable else None
    pos = _spread_keep(Q, want, inc, cost=lambda p, centre: _medoid_cost(pool[p], centre, rec, P))
    return [pool[p] for p in pos]


def _kit_source_pins(rec, category):
    """Pin one file a kit role (two toms, two percs) from each acoustic kit pack
    (sets.KIT_ACOUSTIC_PACKS) into this category, so the kit can be built from the curated files.
    Chosen as sets._kits chooses: files whose names say the role, spread by name."""
    from .sets import KIT_ACOUSTIC_PACKS, KIT_PER_ROLE, _role, _slot_candidates
    by = defaultdict(list)
    for j, d in enumerate(rec):
        pk = _pack_key2(d["path"])
        r = _role(category, d) if KIT_ACOUSTIC_PACKS.search(pk) else None
        if r:
            by[(pk, r)].append(j)
    n = 0
    for (pk, r), js in sorted(by.items()):
        k = KIT_PER_ROLE.get(r, 1) - sum(1 for j in js if rec[j].get("pin"))
        if k <= 0:
            continue
        c = _slot_candidates(r, [(category, dict(src=rec[j]["path"], j=j)) for j in js
                                 if not rec[j].get("pin")])
        c = sorted(c, key=lambda ce: (os.path.basename(ce[1]["src"]).lower(), ce[1]["src"]))
        step = max(1, len(c) // (k + 1))
        for i in range(min(k, len(c))):
            rec[c[min(len(c) - 1, step * (i + 1))][1]["j"]]["pin"] = "kit"   # a selection pin only; export gates still apply
            n += 1
    return n


def _restore_pins(rec, pinned):
    """Re-add pinned records that a cap or dedup pass removed (a Keep is a human call)."""
    have = {d["id"] for d in rec}
    return rec + [d for d in pinned if d["id"] not in have]


def _mkrec(r, pack, rel, category, support, id2row, fallback=False):
    # the root note, tempo and their sources come from the resolver (metadata/resolve.py):
    # Sononym's base note when sure; without Sononym, a WAV's own root note (its ACID or smpl
    # chunk) stands for the note Sononym would have heard: a stated note, used as a name's is
    # (any tonal category)
    _root = _resolve.root(r, fallback)
    tune = _root.value
    _chunk_root = _root.source == _resolve.CHUNK
    if CATEGORIES.get(category, {}).get("kind") == "loop":
        _bpm, _bsrc = _resolve.loop_tempo(r, fallback)
        _bpm = _bpm.value
        _fold = _fold_tempo(_bpm, CATEGORIES[category].get("tempo_fold")) if TEMPO_FOLDING else _bpm
        _nf = CATEGORIES[category].get("no_fold")
        if _nf and _bpm and _fold and _fold > _bpm and _nf.search(rel) and not FOLD_ANYWAY_RE.search(rel):
            _fold = _bpm                  # a slow groove from a non-breakbeat style stays slow
        if _bsrc == "name" and _bpm and _fold and _fold != _bpm and not (
                _fold > _bpm and FOLD_ANYWAY_RE.search(rel)):
            # a tempo the filename states is where the loop goes: a "182bpm" dnb loop isn't
            # halved to 91, a "65bpm" groove isn't doubled to 130; only a half-time break
            # from a jungle / dnb / breaks path still doubles (86 -> 172)
            _fold = _bpm
        _fold = _fold_in_range(_bpm, _fold)
    else:
        _fold = None
        _bpm, _bsrc = _resolve.oneshot_tempo(r).value, None
    d = dict(
        id=r.id, path=r.path, pack=pack, fmt=(r.file_format or "").lower(), clip=int(r.is_clipped or 0),
        br=_g(r.brightness, .5), atk=_g(r.attack_time_ms, 20.), dec=_g(r.decay_time_ms, 350.),
        sub=_g(r.sub_weight, 0.), noi=_g(r.noisiness, .3), har=_g(r.harmonicity, .7),
        cr=_g(r.crest_factor, 4.3), tune=tune, dur=r.duration_s,
        dc=_g(r.dc_offset_ratio, 0.), rms=r.rms_mean, flat=_g(r.spectral_flatness_mean, .5),
        bpm=_bpm, bpm_src=_bsrc, bpm_fold=_fold,
        is_ab=any(rel.startswith(f"{root}/") for root in INSTRUMENT_ROOTS),
        vendor=_vendors.vendor_of(rel, r.path),        # (first-folder: rel's first folder)
        sup=(support.get((r.id, category), 1) if support else 1),
        ab=_as_list(r.ableton_tags),
        labels=sorted(_canonical_of(r)),
        fav=(1 if FAVORED_SOURCES.search(r.path or "") else 0),
        row=id2row[r.id],
    )
    if category in RETUNE_CATS:
        # without Sononym a capital letter ending the name is its note too ("808_F")
        d["name_pc"] = _name_root_pc(r.filename, bare=fallback)
        d["retune"], d["root_src"] = _retune_shift(category, r.filename, tune, detect=_chunk_root,
                                                   bare=fallback)
        if _chunk_root and d["root_src"] == "detect":
            d["root_src"] = "chunk"
    if _is_loop_row(r):
        d["loop_row"] = True        # exempt from name overrides (_row_override); recorded for verify
    if fallback:                    # Fourier's own pitch focus: whether a loop folder is "musical"
        d["chroma"] = getattr(r, "chroma_concentration", None)
    _cfg = CATEGORIES.get(category, {})
    if _cfg.get("kind") == "waves":
        d["wave_src"] = _wave_source(rel)
    if _cfg.get("band"):
        d["band"] = _band_of(_cfg["band"], r.filename, r.duration_s, getattr(r, "sample_rate", None),
                             _as_list(getattr(r, "categories", None)), _as_list(r.ableton_tags),
                             getattr(r, "chroma_concentration", None), getattr(r, "harmonicity", None),
                             _cfg.get("min_chroma", 1.6), pack=pack, rel=rel)
    if CATEGORIES.get(category, {}).get("name_any") and not _is_synth_tagged(d["ab"]):
        # the name's instrument word (plucked, tuned percussion, wind, string) names a
        # folder ahead of Ableton's catch-all tags (_acoustic_label)
        d["plk"] = _acoustic_name_label(r.filename)

    d["qual"] = _quality(d)
    return d


def _after_gate(r, cfg, fallback=False, har=True):
    """A gated category's checks after its CLAP gate: (reason, detail) for the first that
    fails, None when all pass: "too_harmonic" ([harmonicity, har_max]; not with har=False, a
    loop CLAP re-homed here), "tempo_range" ([lo, hi, tempo]) or "no_tempo" (no whole-bar
    tempo)."""
    _h = _g(r.harmonicity, .7)
    if har and cfg.get("har_max") is not None and _h > cfg["har_max"]:
        return "too_harmonic", [round(float(_h), 3), cfg["har_max"]]
    _t = _resolve.loop_tempo(r, fallback)[0].value
    if cfg.get("bpm_min") and _t and not (cfg["bpm_min"] <= _t <= cfg["bpm_max"]):
        return "tempo_range", [cfg["bpm_min"], cfg["bpm_max"], _t]
    if cfg.get("require_tempo") and not _t:
        return "no_tempo", None              # no whole-bar tempo: can't sync
    return None


# the checks after a CLAP gate whose failures the minimum floor may readmit (_readmit_gated):
# harmonicity, a measurement that reads some real drum loops as tonal. Never a loop with no
# whole-bar tempo (it can't sync) or outside the tempo range.
_READMIT_AFTER_GATE = ("too_harmonic",)


def _clap_sane(sb, sa) -> bool:
    """A CLAP-gated candidate the floor may readmit: it scores at least 0 for the category's
    prompts and more than for its anti-prompts (it fails only the gate's minimum). One that
    sounds more like the anti-prompts (a drum loop's score for PHRASES) never comes back."""
    return sb >= 0.0 and sb > sa


def _gate_belongs(r, cfg) -> bool:
    """A loop category's gated candidate that its own name or folder (shadow.shape_labels)
    or its audio (shadow.audio_class) calls a loop: one the floor may readmit."""
    from ..metadata.shadow import audio_class, shape_labels
    if cfg["kind"] != "loop":
        return False
    return ("class.loop" in shape_labels(r.rel_path or r.path or "")
            or audio_class(r.duration_s, getattr(r, "n_events", None),
                           getattr(r, "onset_rate_hz", None)) == "class.loop")


class _Seen(dict):
    """The byte hashes a category's selection has taken: {file hash: the sample id that took
    it}, so a byte-identical copy met later names the file it repeats (_twin)."""

    def add(self, h, sid=None):
        self.setdefault(h, sid)


def _twin(r, seen_hash, stats, out) -> None:
    """A byte-identical copy of a file the category already took: counted with the twins and
    recorded for `fourier why` (the why log's "twins", the id of the file it repeats)."""
    stats["twins"] += 1
    if seen_hash.get(r.file_hash) is not None:
        out(r, "twins", seen_hash[r.file_hash])


def _readmit_gated(rec, held, floor, category, support, id2row, seen_hash, seen_ck, fm, stats,
                   why, gate, fallback=False):
    """The gates' floor: a category its CLAP gate (or a check after it, _READMIT_AFTER_GATE)
    would leave below its minimum (TooFewSamples, so a library with enough of these sounds
    never gets here) takes back its best-scoring gated loops (score less anti score), up to
    the minimum, deduplicated as the rest. Each failed one gate only (held only when it
    passes every other check, _after_gate), and a CLAP-gated one sounds more like the
    category than its anti-prompts (_clap_sane)."""
    for _m, r, pack, rel, reason in sorted(held, key=lambda x: (-x[0], x[1].id)):
        if len(rec) >= floor:
            break
        if r.file_hash and r.file_hash in seen_hash:
            continue
        ck = _content_key(pack, r.filename, r.duration_s)
        if ck in seen_ck:
            continue
        seen_ck.add(ck)
        if r.file_hash:
            seen_hash.add(r.file_hash, r.id)
        if _is_nameable_vendor(r.rel_path, r.path):
            fm.add(pack)
        rec.append(_mkrec(r, pack, rel, category, support, id2row, fallback))
        stats[reason] -= 1
        _key = "clap_readmitted" if reason == "gated_out" else f"{reason}_readmitted"
        stats[_key] = stats.get(_key, 0) + 1
        if why is not None:
            got = why.get(reason, {}).pop(r.id, None)
            if reason != "gated_out":       # for `fourier why`: what it was readmitted past
                why.setdefault(why_log.READMITTED, {})[r.id] = [reason, got]
        if gate is not None and r.id in gate and reason == "gated_out":
            gate[r.id] = gate[r.id][:2] + [1]


def _select_records(rows, cfg, emb_n, id2row, homes=None, support=None, category=None,
                    exclude_ids=None, routed_ids=None, wave_hashes=None, mirrors=None,
                    floor=0, why=None, gate=None, fallback=False, clap_homed=None, rehomed=None):
    """Home filter, noise filter, byte-hash + content-twin dedup, CLAP gates,
    acoustic near-duplicate prune.

    When `homes` is provided (the two-classifier resolution), oneshot/loop
    candidates not homed to `category` are dropped; without it, behaviour is the
    pre-union single-classifier selection. `exclude_ids` drops samples already in a
    base release (additive builds). `floor`, the category's minimum: the vendor cap and the
    near-duplicate prune never take the pool below it, and when the CLAP gate alone would
    leave it below (TooFewSamples), the best-scoring gated loops are readmitted up to it
    (_readmit_gated; not in an additive build). `why`, a dict, gets the reason each of this
    category's candidates was left out: {reason: {sample id: detail}}. `gate`, a dict, gets
    each CLAP-gated candidate's scores: {sample id: [score, anti score(, 1 if readmitted)]}.
    `clap_homed`: the ids the CLAP fallback homed here (without Sononym): a loop among them
    cleared the drum loops' CLAP gate by a margin, which stands in for drum evidence."""
    noise = re.compile(cfg["noise"], re.I)

    def _out(r, reason, detail=None):
        if why is not None:
            why.setdefault(reason, {})[r.id] = detail

    stats = dict(twins=0, gated_out=0, total=0, near_dup=0, note_gated=0, loop_gated=0,
                 nondrum_gated=0)
    use_home = cfg["kind"] in ("oneshot", "loop") and homes is not None
    nmin, nmax = _note_window(category, cfg)
    guard_mode = _loop_guard_mode(category, cfg)
    chain_mode = _chain_guard_mode(category, cfg)

    gated = bool(cfg.get("anti"))
    if gated:
        Tb = np.stack([embed_text(p) for p in cfg["phrases"]])
        Ta = np.stack([embed_text(p) for p in cfg["anti"]])
        break_anchor = Tb.mean(0); break_anchor /= np.linalg.norm(break_anchor) + 1e-9
        anti_anchor = Ta.mean(0); anti_anchor /= np.linalg.norm(anti_anchor) + 1e-9

    excl = bool(cfg.get("exclude_phrases"))
    if excl:
        Te = np.stack([embed_text(p) for p in cfg["exclude_phrases"]])
        Tx = np.stack([embed_text(p) for p in cfg["exclude_anti"]])
        ex_anchor = Te.mean(0); ex_anchor /= np.linalg.norm(ex_anchor) + 1e-9
        ex_anti = Tx.mean(0); ex_anti /= np.linalg.norm(ex_anti) + 1e-9

    clsrc = bool(cfg.get("clap_source"))
    if clsrc:
        Tp = np.stack([embed_text(p) for p in cfg["phrases"]])
        Tpa = np.stack([embed_text(p) for p in cfg["clap_anti"]])
        pia_anchor = Tp.mean(0); pia_anchor /= np.linalg.norm(pia_anchor) + 1e-9
        pia_anti = Tpa.mean(0); pia_anti /= np.linalg.norm(pia_anti) + 1e-9

    from .ratings import drop_set, keep_pins, misfiled_map
    pins = keep_pins()
    drops = drop_set()
    misfiled = _misfiled_index(misfiled_map(), emb_n, id2row)
    rec, seen_hash, seen_ck, fm = [], _Seen(), set(), set()
    held = []                    # (score margin, row, pack, rel): gated by CLAP alone (the floor)
    phrases_on = PHRASES_CATEGORY in CATEGORIES
    # Keeps pinned here go first, so an unpinned byte-identical or content twin can't
    # claim the dedup slot and knock the pinned file out (stable: order otherwise kept)
    if category and pins:
        rows = sorted(rows, key=lambda r: pins.get(r.path) != category)
    # the loudest source peak of each multisample set, for the soft-layer rule
    _soft = cfg["kind"] in ("oneshot", "gated", "instrument")
    set_peak = {}
    if _soft:
        for r in rows:
            pk = getattr(r, "peak_db", None)
            k = _sibling_key(r.path) if pk is not None else None
            if k is not None and pk > set_peak.get(k, -1e9):
                set_peak[k] = pk
    for r in rows:
        if r.id not in id2row or not r.path or (noise.search(r.filename or "")
                                                and pins.get(r.path) != category):
            continue
        if mirrors and r.path in mirrors:
            stats["mirror_copies"] = stats.get("mirror_copies", 0) + 1
            continue                     # a copy in a mirror folder
        if cfg.get("path_exclude") and cfg["path_exclude"].search(r.rel_path or r.path):
            stats["style_excluded"] = stats.get("style_excluded", 0) + 1
            _out(r, "style_excluded", cfg["path_exclude"].search(r.rel_path or r.path).group(0))
            continue
        _pk = getattr(r, "peak_db", None)
        if _soft and _pk is not None and _pk < SOFT_LAYER_PEAK_DB and pins.get(r.path) != category:
            _k = _sibling_key(r.path)
            if _k is not None and set_peak.get(_k, -1e9) >= max(SOFT_LAYER_PEAK_DB, _pk + SOFT_LAYER_GAP_DB):
                stats["soft_layers"] = stats.get("soft_layers", 0) + 1
                _out(r, "soft_layers", round(float(_pk), 1))
                continue                 # too quiet to normalize, and a louder take exists
        if _is_ir(r.rel_path, r.path):
            stats["impulse_responses"] = stats.get("impulse_responses", 0) + 1
            continue                     # a reverb impulse response: never a sample to play
        if _is_preview(r.rel_path, r.path):
            stats["previews"] = stats.get("previews", 0) + 1
            continue                     # a preset / kit preview render: a demo, not a sample
        # hygiene: drop clipped samples and corrupt multi-minute files
        if r.is_clipped or (r.duration_s and r.duration_s > 60):
            continue
        if r.path in drops:
            stats["dropped"] = stats.get("dropped", 0) + 1
            continue                     # rated Drop: out of every category
        if exclude_ids and r.id in exclude_ids:
            # already in the base release: it still claims its byte-hash / content-twin slot,
            # so an identical copy from another pack isn't added as "new"
            if r.file_hash:
                seen_hash.add(r.file_hash, r.id)
            seen_ck.add(_content_key(_pack_of(r.rel_path, r.path), r.filename, r.duration_s))
            continue
        # rated Misfiled here (itself or a sibling, by you or the detector): never back in
        # this folder. An explicit Keep on the file itself wins over a sibling's verdict.
        if category and pins.get(r.path) != category \
                and category in misfiled.categories(r.path, emb_n[id2row[r.id]]):
            stats["misfiled_out"] = stats.get("misfiled_out", 0) + 1
            continue
        _rv = _name_reserved(r.filename or "")
        if category and _rv and _rv != category and pins.get(r.path) != category:
            continue                     # reserved to another category by name
        # phrases live in PHRASES only, and PHRASES holds phrases only (a Keep wins; a phrase
        # rated Misfiled in PHRASES may go back to a one-shot home)
        if category and phrases_on and pins.get(r.path) != category:
            # (a loop CLAP re-homed here, without Sononym: what this category is)
            _ph = (category == PHRASES_CATEGORY) if (rehomed and r.id in rehomed) else (
                _is_phrase(r, fallback) and PHRASES_CATEGORY not in misfiled.categories(
                    r.path, emb_n[id2row[r.id]]))
            if _ph != (category == PHRASES_CATEGORY):
                stats["phrase_out" if _ph else "not_phrase"] = \
                    stats.get("phrase_out" if _ph else "not_phrase", 0) + 1
                continue
        _ovi, _ov = _row_override(r)
        # a rated Keep pins the file to the category it was rated in (wins over overrides)
        _pin = pins.get(r.path)
        _home = _pin or _ov
        if _home is not None and _home != category:
            continue
        if _home == category:
            rel = r.rel_path or (r.path or "")
            pack = _pack_of(r.rel_path, r.path)
            # a name override still can't put a wave outside WAVES (a "gamelan" wavetable
            # frame); only a Keep overrides that
            if not _pin and _not_here_as_wave(r, rel, category, wave_hashes):
                stats["wave_out"] = stats.get("wave_out", 0) + 1
                _out(r, "wave_out")
                continue
            # ...nor past the category's length cap, judged on the length it exports at (a
            # cowbell's long quiet tail is trimmed; a 15 s field recording of cowbells isn't one)
            _cap = DUR_CAP.get(category)
            if not _pin and _cap and r.duration_s and r.duration_s > _cap + _DUR_CAP_SLACK_S \
                    and _exported_seconds(r.path, category) > _cap + _DUR_CAP_SLACK_S:
                stats["too_long"] = stats.get("too_long", 0) + 1
                _out(r, "too_long", _cap)
                continue
            if r.file_hash and r.file_hash in seen_hash:
                _twin(r, seen_hash, stats, _out)
                continue
            ck = _content_key(pack, r.filename, r.duration_s)
            if ck in seen_ck:
                stats["twins"] += 1
                continue
            seen_ck.add(ck)
            if r.file_hash:
                seen_hash.add(r.file_hash, r.id)
            if _is_nameable_vendor(r.rel_path, r.path):
                fm.add(pack)
            stats["total"] += 1
            _d = _mkrec(r, pack, rel, category, support, id2row, fallback)
            _d["ov"] = None if _pin else _ovi
            _d["pin"] = bool(_pin)
            rec.append(_d)
            continue
        if use_home and homes.get(r.id) != category:
            continue
        # pitch window: drop a confidently-detected note outside [nmin, nmax] for
        # pitched categories (keeps out piercing top-octave notes / sub-sonic mislabels)
        if (nmin is not None or nmax is not None) and r.base_note is not None \
                and r.base_note_confidence is not None and r.base_note_confidence > NOTE_CONF:
            if (nmin is not None and r.base_note < nmin) or \
               (nmax is not None and r.base_note > nmax):
                stats["note_gated"] += 1
                _out(r, "note_gated", [nmin, nmax])
                continue
        # mislabeled-loop guard: a loop that leaked into a percussion hit folder.
        # Real loops are Sononym Loop class and route to DRUMLOOPS, so never reach here.
        if guard_mode:
            oc = round((r.onset_rate_hz or 0.0) * (r.duration_s or 0.0))
            if _is_mislabeled_loop(guard_mode, _as_list(r.ableton_tags), oc,
                                   _resolve.tempo_reliable(r).value):
                stats["loop_gated"] += 1
                _out(r, "loop_gated")
                continue
        # sample-chain guard: several sounds laid end to end in one "one-shot"
        if chain_mode and _is_sample_chain(chain_mode, getattr(r, "n_events", None),
                                           getattr(r, "event_regularity", None),
                                           getattr(r, "event_echo", None)):
            stats["chain_gated"] = stats.get("chain_gated", 0) + 1
            _out(r, "chain_gated")
            continue
        # drum-loop categories: require Ableton drum evidence (drops bassline/lead/
        # vocal/FX loops that Sononym only calls a generic "Loop")
        # (without Sononym, the drums its loop vote needed: _drum_loop_evidence)
        if cfg.get("drum_tag_required") and not _has_drum_tag(_as_list(r.ableton_tags)) and not (
                fallback and (_drum_loop_evidence(r, _as_list(r.ableton_tags), cfg)
                              or (clap_homed and r.id in clap_homed))):
            stats["nondrum_gated"] += 1
            _out(r, "nondrum_gated")
            continue
        rel = r.rel_path or (r.path or "")
        if cfg["kind"] != "instrument" and DEMO_PATH.search(rel):
            continue
        pack = _pack_of(r.rel_path, r.path)
        if PACK_HOME.get(pack, category) != category:
            stats["pack_homed"] = stats.get("pack_homed", 0) + 1
            _out(r, "pack_homed", PACK_HOME.get(pack))
            continue
        # waves live only in WAVES (and everything in WAVES must be a wave)
        if _not_here_as_wave(r, rel, category, wave_hashes):
            if category != WAVES_CATEGORY:
                stats["wave_out"] = stats.get("wave_out", 0) + 1
                _out(r, "wave_out")
            continue
        # a long file from a loop folder isn't a one-shot pad / bass / synth / stab / vox
        # (unless it's a phrase you rated Misfiled in PHRASES: that one may come back)
        if category in LOOP_DIR_ONESHOT_CATS and (r.duration_s or 0) > LOOP_DIR_ONESHOT_MAX_S \
                and _in_loop_folder(rel) and PHRASES_CATEGORY not in misfiled.categories(
                    r.path, emb_n[id2row[r.id]]):
            stats["loop_folder"] = stats.get("loop_folder", 0) + 1
            _out(r, "loop_folder")
            continue
        # length cap and bpm-named phrases (a Keep above still wins over both)
        _cap = DUR_CAP.get(category)
        if _cap and r.duration_s and r.duration_s > _cap:
            stats["too_long"] = stats.get("too_long", 0) + 1
            _out(r, "too_long", _cap)
            continue
        if category in BPM_NAME_GUARD and BPM_NAME_RE.search(r.filename or ""):
            stats["bpm_named"] = stats.get("bpm_named", 0) + 1
            _out(r, "bpm_named")
            continue
        if cfg.get("stab_names") and not _stab_named(r.filename):
            stats["not_stab_named"] = stats.get("not_stab_named", 0) + 1
            _out(r, "not_stab_named")
            continue
        if cfg.get("scratch_names") and not _scratch_named(r.filename, rel):
            stats["not_scratch_named"] = stats.get("not_scratch_named", 0) + 1
            continue
        # named as another drum, or as a scratch: it belongs there (unless rated Misfiled there)
        if category in DRUM_NAME_FROM and cfg["kind"] == "oneshot":
            _dn = SCRATCH_HOME if _scratch_named(r.filename, rel) else _drum_named_in(r.filename, rel, fallback)
            if _dn and _dn != category and _dn not in misfiled.categories(r.path, emb_n[id2row[r.id]]):
                stats["named_elsewhere"] = stats.get("named_elsewhere", 0) + 1
                _out(r, "named_elsewhere", _dn)
                continue
        if cfg["kind"] == "instrument":
            dur = r.duration_s
            # named as keys by the built-in providers (without Sononym): one event between
            # silences is one note or chord, whatever its onsets (a tremolo, a chord's beating)
            _kl = fallback and clsrc and _keys_labeled(r)
            # single-note guard: a single note (short or long-sustained) has ~1
            # onset; phrases/arps/loops have many. Drops phrases, keeps notes.
            onc = (r.onset_rate_hz or 0.0) * (dur or 0.0)
            if cfg.get("max_onsets") is not None and onc > cfg["max_onsets"] and not (
                    _kl and (getattr(r, "n_events", None) or 1) <= 1):
                stats["many_onsets"] = stats.get("many_onsets", 0) + 1
                _out(r, "many_onsets", [round(onc, 1), cfg["max_onsets"]])
                continue
            if cfg.get("chord_band") and _is_chord(r.filename, r.chroma_concentration,
                                                   cfg.get("min_chroma", 1.6), r.harmonicity):
                # a chord: welcome if it's a stab (one hit, short, not a bpm phrase)
                if not _is_stab(r.filename, dur, (getattr(r, "n_events", None) or 1) <= 1):
                    stats["chord_not_stab"] = stats.get("chord_not_stab", 0) + 1
                    _out(r, "chord_not_stab")
                    continue
            elif cfg.get("min_chroma") is not None and (r.chroma_concentration or 0.0) < cfg["min_chroma"]:
                stats["not_one_pitch"] = stats.get("not_one_pitch", 0) + 1
                _out(r, "not_one_pitch")
                continue
            if cfg.get("dur_max") and dur and dur > cfg["dur_max"]:
                stats["too_long"] = stats.get("too_long", 0) + 1
                _out(r, "too_long", cfg["dur_max"])
                continue
            _abt = _as_list(r.ableton_tags)
            if cfg.get("pack_exclude") and cfg["pack_exclude"].search(pack):
                continue
            if cfg.get("name_exclude") and cfg["name_exclude"].search(r.filename or ""):
                continue
            # a named wind / brass one-shot or an orchestra-folder file is ACOUSTIC's, as
            # compute_homes has it for the other categories (without Live's tags to say
            # otherwise, a horn patch could reach PIANO)
            if category != "ACOUSTIC" and not _pin and not ORCH_PACK_EXCLUDE.search(pack) and (
                    _is_wind_named(r.filename, _abt) or _orch_path(r.rel_path) is not None):
                continue
            # keep foreign instruments out (e.g. classical guitar / strings out of PIANO),
            # unless the filename names this instrument (an auto-tag can misname a piano)
            if cfg.get("ableton_exclude") and _has_any_tag(_abt, cfg["ableton_exclude"]) and not (
                    cfg.get("ableton_exclude_unless_name")
                    and (cfg["ableton_exclude_unless_name"].search(r.filename or "")
                         or (cfg.get("chord_band") and _is_organ_named(r.filename)))):
                continue
            in_pack = bool(cfg["pack_re"].search(pack)) or (
                bool(cfg.get("ableton_any")) and _acoustic_tagged(_abt, r.filename, pack, cfg["ableton_any"])) or (
                bool(cfg.get("name_any")) and _acoustic_reserved(r.filename or "", _abt, rel))
            if clsrc:
                # named as keys, from a keys pack, or placed with the instrument by CLAP
                ok = in_pack or _kl or bool(cfg.get("chord_band") and (KEYBOARD_NAME_RE.search(r.filename or "")
                                                                       or _is_organ_named(r.filename)))
                if not ok:
                    v = emb_n[id2row[r.id]]
                    if float(v @ pia_anchor) > float(v @ pia_anti) and \
                       float(v @ pia_anchor) >= cfg.get("clap_min", 0.28):
                        ok = True
                if not ok:
                    continue
            elif not in_pack:
                continue
        elif INST_ROUTED_PACKS.search(pack):
            continue
        stats["total"] += 1
        if excl:
            _v = emb_n[id2row[r.id]]
            if float(_v @ ex_anchor) > float(_v @ ex_anti) and float(_v @ ex_anchor) >= cfg.get("exclude_min", 0.30):
                stats["excluded"] = stats.get("excluded", 0) + 1
                _out(r, "excluded")
                continue
        if gated:
            dur = r.duration_s
            if cfg.get("dur_min") is not None and dur is not None and (dur < cfg["dur_min"] or dur > cfg["dur_max"]):
                stats["duration"] = stats.get("duration", 0) + 1
                _out(r, "duration", [cfg["dur_min"], cfg["dur_max"]])
                continue
            v = emb_n[id2row[r.id]]
            sb = float(v @ break_anchor); sa = float(v @ anti_anchor)
            if gate is not None:
                gate[r.id] = [round(sb, 4), round(sa, 4)]
            _af = _after_gate(r, cfg, fallback, har=not (rehomed and r.id in rehomed))
            if not (sb > sa and sb >= cfg.get("break_min", 0.30)):
                stats["gated_out"] += 1
                _out(r, "gated_out", "clap")
                # held for the floor: a loop by its own name, folder or audio that only the
                # CLAP gate keeps out, and that sounds more like the category than its anti-
                # prompts (_clap_sane; _readmit_gated)
                if exclude_ids is None and floor and _af is None and _clap_sane(sb, sa) \
                        and _gate_belongs(r, cfg):
                    held.append((sb - sa, r, pack, rel, "gated_out"))
                continue
            if _af is not None:
                reason, detail = _af
                stats[reason] = stats.get(reason, 0) + 1
                _out(r, reason, detail)
                # held for the floor too: one that passed the CLAP gate and only its
                # harmonicity keeps out (_READMIT_AFTER_GATE)
                if exclude_ids is None and floor and reason in _READMIT_AFTER_GATE \
                        and _gate_belongs(r, cfg):
                    held.append((sb - sa, r, pack, rel, reason))
                continue
        if r.file_hash and r.file_hash in seen_hash:
            _twin(r, seen_hash, stats, _out)
            continue
        ck = _content_key(pack, r.filename, r.duration_s)
        if ck in seen_ck:
            stats["twins"] += 1
            continue
        seen_ck.add(ck)
        if r.file_hash:
            seen_hash.add(r.file_hash, r.id)
        if _is_nameable_vendor(r.rel_path, r.path):
            fm.add(pack)
        rec.append(_mkrec(r, pack, rel, category, support, id2row, fallback))
    if held and len(rec) < floor:
        _readmit_gated(rec, held, floor, category, support, id2row, seen_hash, seen_ck, fm,
                       stats, why, gate, fallback)
    if category in KIT_CATS:
        stats["kit_pins"] = _kit_source_pins(rec, category)
    _pinned = [d for d in rec if d.get("pin")]
    stats["pinned"] = len(_pinned)
    # the previous build's files win the spread caps and near-dup ties (_spread_keep)
    _prev = _previous_families(category) if category else {}
    _prev_fam = {src: fam for fam, srcs in _prev.items() for src in srcs}
    for d in rec:
        d["incumbent"] = d["path"] in _prev_fam
        d["prev_family"] = _prev_fam.get(d["path"])
    # overrides fix routing, never grant priority: each NAME_OVERRIDES rule may make
    # up at most OVERRIDE_MAX_SHARE of the candidate pool, and in pack/name-selected
    # instrument categories (no vendor cap) each vendor/pack at most
    # INSTRUMENT_PACK_MAX_SHARE. Pool share tracks output share (like the vendor cap).
    def _gone(before, after, reason, detail=None):
        if why is not None:
            kept = {d["id"] for d in after}
            for d in before:
                if d["id"] not in kept:
                    why.setdefault(reason, {})[d["id"]] = detail(d) if callable(detail) else detail
        return after

    _pool = len(rec)
    rec = _gone(rec, _cap_group_spread(rec, emb_n, lambda d: d.get("ov"),
                                       max(1, int(np.ceil(OVERRIDE_MAX_SHARE * _pool)))),
                "override_capped")
    stats["override_capped"] = _pool - len(rec)
    if category == "VOX":
        # choirs and synth voices at most VOX_CHOIR_SHARE of the budget (CLAP-spread)
        _bc = len(rec)
        _vb = int(cfg.get("budget") or BUDGETS.get(category, 200))
        rec = _gone(rec, _cap_group_spread(rec, emb_n, lambda d: "choir" if VOX_CHOIR_RE.search(
            library_rel(d["path"] or "")) else None, max(1, int(np.ceil(VOX_CHOIR_SHARE * _vb)))),
            "choir_capped")
        stats["choir_capped"] = _bc - len(rec)
    if cfg.get("dir_cap"):
        _bd = len(rec)
        rec = _gone(rec, _cap_group_spread(rec, emb_n, lambda d: os.path.dirname(d["path"] or ""),
                                           int(cfg["dir_cap"])), "dir_capped", int(cfg["dir_cap"]))
        stats["dir_capped"] = _bd - len(rec)
    if cfg["kind"] == "instrument":
        _b0 = len(rec)
        rec = _gone(rec, _cap_group_spread(rec, emb_n, lambda d: _vendors.pack_key(d["path"]),
                                           max(1, int(np.ceil(INSTRUMENT_PACK_MAX_SHARE * _pool)))),
                    "pack_capped", lambda d: _vendors.pack_key(d["path"]))
        stats["pack_capped"] = _b0 - len(rec)
    if cfg["kind"] != "instrument":
        _before = len(rec)
        rec = _gone(rec, _cap_instrument_packs(rec, emb_n), "instr_capped", lambda d: d["pack"])
        stats["instr_capped"] = _before - len(rec)
        _b2 = len(rec)
        # an additive build (exclude_ids: a release's samples are out) caps as it always has:
        # its picks join a release, and a later one must add what an earlier one would have
        _additive = exclude_ids is not None
        _floor = 0 if _additive else floor
        if not cfg.get("no_vendor_cap"):
            rec = _gone(rec, _cap_vendor_share(rec, emb_n, VENDOR_MAX_SHARE, keep=_floor,
                                               min_vendors=1 if _additive else 3),
                        "vendor_capped", lambda d: d["vendor"])
        stats["vendor_capped"] = _b2 - len(rec)
        _partners = {} if why is not None else None
        rec, pruned = _prune_near_dups(rec, emb_n, cfg.get("near_dup_cos", NEAR_DUP_COS), keep=_floor,
                                       partners=_partners,
                                       apart=_tempo_apart if (fallback and cfg["kind"] == "loop") else None)
        if why is not None and _partners:
            why.setdefault("near_dup", {}).update(_partners)
        stats["near_dup"] = pruned
    if routed_ids:
        # routed in from another classifier bucket (keys chords): capped share of the pool
        _b3 = len(rec)
        rec = _gone(rec, _cap_group_spread(rec, emb_n, lambda d: "routed" if d["id"] in routed_ids else None,
                                           max(1, int(np.ceil(ROUTED_MAX_SHARE * _b3)))), "routed_capped")
        stats["routed_capped"] = _b3 - len(rec)
    rec = _restore_pins(rec, _pinned)
    if why is not None:            # a restored Keep isn't out
        for d in _pinned:
            for got in why.values():
                got.pop(d["id"], None)
    return rec, fm, stats


def _cluster(P, kmax=30, kmin=8, kdiv=800, init=None):
    """KMeans families, near-identical centroids merged. `init`: the previous build's folders
    as (centroids, folder index per point or -1); when their count fits [kmin, kmax], each
    previous file stays with its folder and a newcomer joins the nearest centroid, so folders
    stay put when the pool barely changed (stable picks)."""
    from sklearn.cluster import KMeans
    n = len(P)
    k0 = int(min(kmax, max(kmin, round(n / kdiv))))
    if init is not None and not isinstance(init, np.ndarray):
        init, fixed = init                       # (centroids, previous folder per point or -1)
    else:
        fixed = None
    if init is not None and max(1, kmin) <= len(init) <= kmax and len(init) <= n:
        # the previous folders: their files stay together, newcomers join the nearest
        k0 = len(init)
        lab = np.argmin(((P[:, None, :] - init[None, :, :]) ** 2).sum(-1), axis=1)
        if fixed is not None:
            lab = np.where(np.asarray(fixed) >= 0, np.asarray(fixed), lab)
        k0 = int(lab.max()) + 1
    else:
        with _limit_threads():
            lab = KMeans(n_clusters=k0, random_state=CURATION_SEED, n_init=5).fit_predict(P)
    raw = {c: [i for i in range(n) if lab[i] == c] for c in range(k0)}
    cen = {c: P[raw[c]].mean(0) for c in raw}
    Cn = np.array([cen[c] / (np.linalg.norm(cen[c]) + 1e-9) for c in range(k0)])
    par = list(range(k0))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    for a in range(k0):
        for b in range(a + 1, k0):
            if float(Cn[a] @ Cn[b]) > MERGE_COS:
                par[find(a)] = find(b)
    grp = defaultdict(list)
    for c in range(k0):
        grp[find(c)].append(c)
    return [[i for c in g for i in raw[c]] for g in grp.values()], k0


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_meta():
    """Reproducibility stamp: seed, git sha, config hash, CLAP model."""
    import subprocess
    from . import curate_config as cc
    repo = str(Path(__file__).resolve().parents[3])
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip() or None
    except Exception:
        sha = None
    try:
        cfg_repr = json.dumps(cc.CATEGORIES, sort_keys=True,
                              default=lambda o: getattr(o, "pattern", str(o)))
        cfg_hash = hashlib.sha256(cfg_repr.encode()).hexdigest()[:16]
    except Exception:
        cfg_hash = None
    # everything else a build depends on: BUDGETS and the other constants, the code and
    # the ratings, not CATEGORIES alone
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for f in ("curate_config.py", "curate.py", "naming.py"):
        try:
            h.update((here / f).read_bytes())
        except OSError:
            pass
    code_hash = h.hexdigest()[:16]
    try:
        from .ratings import default_store
        ratings_hash = hashlib.sha256(Path(default_store()).read_bytes()).hexdigest()[:16]
    except OSError:
        ratings_hash = None
    # every tunable's effective value (defaults plus any resolved config), and the overrides
    # themselves, so a manifest says exactly which settings built it (fourier/settings.py)
    try:
        from .. import settings
        tunables_hash = settings.tunables_hash()
        overrides = {k: settings.encode(v) for k, v in sorted(settings.overrides().items())}
    except Exception:
        tunables_hash, overrides = None, None
    try:        # the metadata providers the build used (metadata/providers.py)
        from ..db.session import session_scope
        from ..metadata.providers import active
        with session_scope() as s:
            providers = list(active(s).names)
    except Exception:
        providers = None
    try:        # an installed copy has no git sha: its version says what built it
        from importlib.metadata import version
        fourier_version = version("fourier-samples")
    except Exception:
        fourier_version = None
    try:        # the library it read: a build into this master from another library stops
        from ..places import library_record
        library = library_record()
    except Exception:
        library = None
    return dict(seed=cc.CURATION_SEED, git_sha=sha, fourier_version=fourier_version,
                config_hash=cfg_hash, code_hash=code_hash,
                ratings_hash=ratings_hash, clap_model="laion/clap-htsat-unfused",
                tunables_hash=tunables_hash, overrides=overrides, providers=providers,
                library=library)


def _short_term_db(y, sr, win_ms=None):
    """Loudness as the loudest LEVEL_WINDOW_MS RMS window (dBFS), hop a third of it."""
    from .curate_config import LEVEL_WINDOW_MS
    m = y if y.ndim == 1 else y.mean(axis=1)
    w = max(1, int(sr * (win_ms or LEVEL_WINDOW_MS) / 1000.0))
    if len(m) <= w:
        return _rms_db(m)
    sq = np.concatenate([[0.0], np.cumsum(m.astype("float64") ** 2)])
    hop = max(1, w // 3)
    starts = np.arange(0, len(m) - w + 1, hop)
    ms = (sq[starts + w] - sq[starts]) / w
    top = float(ms.max())
    return 10 * np.log10(top) if top > 0 else float("-inf")


def phase_stats(y):
    """(L/R correlation, dB lost summed to mono) of a stereo array, or (None, 0.0)."""
    if y.ndim != 2 or y.shape[1] != 2 or not y.size:
        return None, 0.0
    l, r = y[:, 0].astype("float64"), y[:, 1].astype("float64")
    den = float(np.sqrt(np.sum(l * l) * np.sum(r * r)))
    corr = float(np.sum(l * r) / den) if den > 0 else None
    ch = float(np.mean((l * l + r * r) / 2))
    m = float(np.mean(((l + r) / 2) ** 2))
    loss = -10 * np.log10(m / ch) if ch > 0 and m > 0 else (99.0 if ch > 0 else 0.0)
    return corr, loss


def _fix_phase(kroot, entries, kind="oneshot", category=None):
    """A stereo file whose channels nearly cancel (correlation under PHASE_FIX_CORR, or
    PHASE_FIX_LOSS_DB lost summed to mono; drums and drum loops under the stricter
    PHASE_FIX_DRUM_CORR / PHASE_FIX_DRUM_LOSS_DB) keeps its louder channel, as a forced mono
    file does: a clap that vanishes in mono isn't a width choice.
    Written anew; the entry records `phase_fix`. Returns files."""
    import soundfile as sf
    from .curate_config import phase_fix_limits
    max_corr, max_loss = phase_fix_limits(category)
    n = 0
    for e in entries:
        p = Path(kroot) / e["out"]
        try:
            if sf.info(str(p)).channels != 2:
                continue
            y, sr = sf.read(str(p), always_2d=True)
        except Exception:
            continue
        corr, loss = phase_stats(y)
        if not ((corr is not None and corr < max_corr) or loss > max_loss):
            continue
        k = int(np.argmax(np.mean(np.square(y), axis=0)))
        y1 = y[:, k]
        if kind not in ("loop", "waves"):
            # the tail was trimmed on both channels: judge it again on the one kept
            # (it can hold on to a quiet stretch under its new peak)
            y1 = _trim_edges(y1, sr, False, True,
                             tail_floor_db=INSTR_TAIL_FLOOR_DB if kind == "instrument" else TRIM_FLOOR_DB,
                             fade_ms=TAIL_FADE_MS)
        info = sf.info(str(p))
        tmp = p.with_name(p.name + ".tmp.wav")
        sf.write(str(tmp), y1, sr, subtype=info.subtype if info.subtype.startswith("PCM") else "PCM_24")
        os.replace(tmp, p)
        e["phase_fix"] = "left" if k == 0 else "right"
        e["out_md5"] = _md5(p)
        n += 1
    return n


def _level_folder(kroot, entries, tol_db=None, rms_ceil_db=None, up=True):
    """Bring a melodic folder's files within tol_db (LEVEL_TOL_DB) of its median
    short-term loudness: down freely, up only to the -1 dBFS peak ceiling (not at all with
    up=False: drum folders, DRUM_LEVEL_CATS, only turn their loudest hits down). A changed file
    is written anew (never in place: it may be a hardlink into the audio cache); its entry
    gets level_db and a fresh out_md5. Returns files changed."""
    import soundfile as sf
    from .curate_config import LEVEL_TOL_DB
    tol = LEVEL_TOL_DB if tol_db is None else tol_db
    loud = {}
    for e in entries:
        if e.get("dsp_fallback"):
            continue
        try:
            y, sr = sf.read(str(Path(kroot) / e["out"]), always_2d=False)
        except Exception:
            continue
        v = _short_term_db(np.asarray(y, dtype="float64"), sr)
        if np.isfinite(v):
            loud[e["out"]] = v
    if len(loud) < 3:
        return 0
    med = float(np.median(list(loud.values())))
    ceiling = _amp_from_db(PEAK_CEILING_DB)
    n = 0
    for e in entries:
        v = loud.get(e["out"])
        if v is None or abs(v - med) <= tol or (not up and v < med):
            continue
        p = Path(kroot) / e["out"]
        info = sf.info(str(p))
        y, sr = sf.read(str(p), always_2d=False)
        y = np.asarray(y, dtype="float64")
        gain = (med - tol) - v if v < med else (med + tol) - v
        peak = float(np.abs(y).max()) if y.size else 0.0
        if gain > 0 and peak > 0:
            from .curate_config import LEVEL_MAX_UP_DB
            gain = min(gain, 20 * np.log10(ceiling / peak), LEVEL_MAX_UP_DB)
            if rms_ceil_db is not None:          # and no hotter than the category allows
                gain = min(gain, rms_ceil_db - _rms_db(y))
        if abs(gain) < 0.1:
            continue
        y = np.clip(y * _amp_from_db(gain), -1.0, 1.0)
        tmp = p.with_name(p.name + ".tmp.wav")
        sf.write(str(tmp), y, sr, subtype=info.subtype if info.subtype.startswith("PCM") else "PCM_24")
        os.replace(tmp, p)
        e["level_db"] = round(float(gain), 1)
        e["out_md5"] = _md5(p)
        n += 1
    return n


def _grid_offset(y, sr, bpm):
    """Seconds a loop's hits sit off its 16th grid (+ late, - early), when they agree
    (ROTATE_AGREE of them within ROTATE_TOL_MS) and it's ROTATE_MIN_MS-ROTATE_MAX_MS;
    else 0.0. Only the even 16ths count, so swing can't read as an offset."""
    from .curate_config import (ROTATE_AGREE, ROTATE_MAX_MS, ROTATE_MAX_SIXTEENTH, ROTATE_MIN_MS,
                                ROTATE_ON_GRID_MAX, ROTATE_PREROLL_MS, ROTATE_TOL_MS)
    from .sets import _onsets
    if not bpm:
        return 0.0
    m = y if y.ndim == 1 else y.mean(axis=1)
    on = _onsets(m, sr)
    six = 60.0 / bpm / 4
    if len(on) < 6:
        return 0.0
    idx = np.round(on / six).astype(int)
    dev = on - idx * six
    ev = dev[idx % 2 == 0]
    if len(ev) < 4:
        return 0.0
    tol = ROTATE_TOL_MS / 1000
    if on[0] > tol:
        # silence, then the downbeat: the loop was cut early, so it rotates to the attack
        # whatever the other hits do (a long silent lead-in is over the 16th cap but
        # still a cut, not feel)
        a = _attack_start(m, sr, on[0])
        lead = a - ROTATE_PREROLL_MS / 1000
        if ROTATE_MIN_MS / 1000 <= lead <= ROTATE_MAX_MS / 1000 and _quiet_head(m, sr, a):
            return lead
    # a loop that opens on a hit is cut on its downbeat: hits off the grid are its feel
    # (rotating a break played ahead of the beat would move its downbeat too), and so is
    # a loop where many hits sit on the grid and many don't
    if on[0] <= tol or np.mean(np.abs(ev) <= tol) >= ROTATE_ON_GRID_MAX:
        return 0.0
    off = float(np.median(ev))
    if not (ROTATE_MIN_MS / 1000 <= abs(off) <= min(ROTATE_MAX_MS / 1000, ROTATE_MAX_SIXTEENTH * six)):
        return 0.0
    if np.mean(np.abs(ev - off) <= ROTATE_TOL_MS / 1000) < ROTATE_AGREE:
        return 0.0
    if off > 0:
        # a late start is rotated to the downbeat's attack and never past it: the median of
        # the hits can overshoot the first one and move the front of the downbeat to the end
        off = min(off, _attack_start(m, sr, on[0]) - ROTATE_PREROLL_MS / 1000)
        if off < ROTATE_MIN_MS / 1000:
            return 0.0
    return off


def _quiet_head(m, sr, a):
    """True when everything before the attack at `a` (s) is ROTATE_QUIET_HEAD_DB under the
    hit's first 10 ms: a lead-in of silence or room tone, not a pickup or a ringing tail."""
    from .curate_config import ROTATE_QUIET_HEAD_DB
    i = int(a * sr)
    if i < 2:
        return False
    head = float(np.sqrt(np.mean(np.square(m[:i]))))
    hit = float(np.sqrt(np.mean(np.square(m[i:i + int(0.01 * sr)])))) if i < len(m) else 0.0
    return hit > 0 and (head == 0 or 20 * np.log10(hit / head) >= ROTATE_QUIET_HEAD_DB)


def _attack_start(m, sr, t0):
    """Seconds where the hit detected at t0 starts to rise: walking back from it, the first
    point where its 0.5 ms envelope is under ATTACK_START_FRAC of the hit's peak. Walking
    back keeps room tone or vinyl noise ahead of the hit from counting as its attack."""
    from .curate_config import ATTACK_START_FRAC
    w = max(1, int(sr * 0.0005))
    i0 = min(int(t0 * sr), len(m) - 1)
    seg = m[:i0 + int(0.02 * sr)]
    env = np.sqrt(np.convolve(seg * seg, np.ones(w) / w, "same"))
    pk = float(env[i0:].max()) if len(env) > i0 else 0.0
    if pk <= 0:
        return t0
    below = np.nonzero(env[:i0 + 1] < ATTACK_START_FRAC * pk)[0]
    return (int(below[-1]) + 1) / sr if len(below) else 0.0


def _loop_tempo(r):
    """The tempo a drum loop is filed, sliced and gridded at: its folder tempo."""
    return r.get("bpm_fold") or r.get("bpm")


def _rotate_and_slice(path, bpm, src):
    """Rotate an off-grid drum loop onto its grid (_grid_offset; not fills, rolls or swung
    loops), then its slice metrics. A rotated file is written anew (it may be a cache
    hardlink) and keeps its length. Returns the manifest fields."""
    import soundfile as sf
    from .curate_config import ROTATE_SKIP_RE
    from .sets import SWING_FLAG
    out = {}
    try:
        m0 = _slice_fields(path, bpm)
        if bpm and not ROTATE_SKIP_RE.search(os.path.basename(src or "")) \
                and (m0.get("swing") or 50.0) < SWING_FLAG:
            info = sf.info(str(path))
            y, sr = sf.read(str(path), always_2d=False)
            off = _grid_offset(np.asarray(y, dtype="float64"), sr, bpm)
            if off:
                from .curate_config import ROTATE_SEAM_MS
                sh = int(round(off * sr))
                y2 = np.roll(y, -sh, axis=0)
                # the file's old start/end join now sits inside the loop: dip through zero
                # there so a hard cut can't click
                seam = (len(y) - sh) % len(y)
                k = max(1, int(sr * ROTATE_SEAM_MS / 1000))
                ramp = np.linspace(1.0, 0.0, k)
                if seam - k >= 0 and seam + k <= len(y):
                    fo = ramp if y2.ndim == 1 else ramp[:, None]
                    y2 = y2.copy()
                    y2[seam - k:seam] *= fo
                    y2[seam:seam + k] *= fo[::-1]
                y2 = _edge_fades(y2, sr, loop=True)     # the new start is a cut mid-waveform
                tmp = Path(str(path) + ".tmp.wav")
                sf.write(str(tmp), y2, sr, subtype=info.subtype if info.subtype.startswith("PCM") else "PCM_24")
                os.replace(tmp, path)
                out["rotate_ms"] = round(off * 1000, 1)
                m0 = _slice_fields(path, bpm)
        out.update(m0)
    except Exception:
        return _slice_fields(path, bpm)
    return out


def _slice_fields(path, bpm):
    """bars / slice_clean / swing for an exported drum loop (packs/sets.py); {} on failure."""
    try:
        from .sets import slice_metrics
        return slice_metrics(path, bpm)
    except Exception:
        return {}


def update_build_manifest(out_dir, category, summary, file_entries):
    """Read-modify-write the build's top-level manifest.json, replacing this
    category's entry and refreshing the reproducibility stamp. Preserves other
    categories so both single-category and --all builds stay coherent."""
    mpath = Path(out_dir) / "manifest.json"
    try:
        doc = manifests.read(mpath) if mpath.exists() else {}
    except Exception:
        doc = {}
    doc["fourier_manifest"] = _MANIFEST_VERSION
    doc.update(_build_meta())
    doc["generated"] = _now_iso()
    cats = doc.setdefault("categories", {})
    cats[category] = dict(
        families=summary["families"], files=summary["files"],
        source_samples=summary["source_samples"],
        note_gated=summary.get("note_gated", 0), loop_gated=summary.get("loop_gated", 0),
        chain_gated=summary.get("chain_gated", 0),
        **({"budget": summary["budget"]} if "budget" in summary else {}),
        entries=file_entries,
    )
    if summary.get("scale"):
        doc["scale"] = summary["scale"]
    manifests.write(mpath, doc)


def merge_manifest(out_dir, results):
    """Write the top-level manifest for a batch of category builds in one shot.
    Parallel builds each return their file entries; the parent merges here to avoid
    a write race on the shared manifest. Categories already present from an earlier
    (or serial) build are preserved."""
    mpath = Path(out_dir) / "manifest.json"
    try:
        doc = manifests.read(mpath) if mpath.exists() else {}
    except Exception:
        doc = {}
    doc["fourier_manifest"] = _MANIFEST_VERSION
    doc.update(_build_meta())
    doc["generated"] = _now_iso()
    cats = doc.setdefault("categories", {})
    for category, summary, entries in results:
        if summary is None:
            continue
        cats[category] = dict(
            built=_now_iso(), families=summary["families"], files=summary["files"],
            source_samples=summary["source_samples"],
            note_gated=summary.get("note_gated", 0), loop_gated=summary.get("loop_gated", 0),
            chain_gated=summary.get("chain_gated", 0),
            **({"budget": summary["budget"]} if "budget" in summary else {}),
            entries=entries,
        )
    # a library-scaled build records its scale (verify reads the minimum and folder sizes)
    scaled = [s["scale"] for _c, s, _e in results if s is not None and s.get("scale")]
    if scaled:
        doc["scale"] = scaled[0]
    else:
        doc.pop("scale", None)
    manifests.write(mpath, doc)


# ---------------------------------------------------------------------------
# Loudness / trim export pass (per-kind, deterministic)
# ---------------------------------------------------------------------------
def _amp_from_db(db):
    return float(10.0 ** (db / 20.0))


def _trim_edges(y, sr, trim_lead, trim_tail, floor_db=TRIM_FLOOR_DB, pad_ms=TRIM_PAD_MS,
                tail_floor_db=None, fade_ms=0.0, lead_floor_db=None):
    """Trim leading/trailing silence below floor_db (relative to peak), keeping a
    small pad before the first non-silent sample. Loops are never trimmed (the
    caller passes trim_lead=trim_tail=False) so bar-length is preserved.
    tail_floor_db (default floor_db) sets a separate, lower floor for the tail, and
    fade_ms fades out the last stretch of a trimmed tail."""
    if not (trim_lead or trim_tail):
        return y
    mono = np.abs(y if y.ndim == 1 else np.abs(y).mean(axis=1))
    if mono.size == 0:
        return y
    peak = float(mono.max())
    if peak <= 0:
        return y
    nz = np.where(mono > peak * _amp_from_db(floor_db))[0]
    if nz.size == 0:
        return y
    if tail_floor_db is None:
        nzt = nz
    else:
        # a release tail is judged on a 10 ms RMS envelope, so a stray click near the end
        # of the file can't hold on to the silence before it
        env = _rms_env(mono, sr)
        # ...and never kept below TAIL_PEAK_FLOOR_DB of the sample peak: a decay's last
        # seconds under -50 dB are inaudible and cost card streaming (a long decay can spend
        # most of its length there)
        # (the sample peak across channels, as verify reads it, not the channel-mean's)
        rel = max(_amp_from_db(tail_floor_db),
                  float(np.abs(y).max()) * _amp_from_db(TAIL_PEAK_FLOOR_DB) / max(float(env.max()), 1e-12))
        nzt = _sustained_above(env, rel, int(sr * 0.02))
        if nzt.size == 0:
            nzt = nz
    pad = int(sr * pad_ms / 1000.0)
    lead = int(nz[0])
    if lead_floor_db is not None:
        # a field recording's room sits above floor_db: find the lead on a 10 ms RMS
        # envelope against a higher floor, so a stray noise spike can't hold it
        env = _rms_env(mono, sr)
        nzl = np.where(env > float(env.max()) * _amp_from_db(lead_floor_db))[0]
        if nzl.size:
            lead = max(lead, int(nzl[0]))
    a = max(0, lead - pad) if trim_lead else 0
    b = min(len(mono) - 1, int(nzt[-1]) + pad) if trim_tail else len(mono) - 1
    out = y[a:b + 1]
    if trim_tail and fade_ms and b < len(mono) - 1:
        nf = min(len(out), int(sr * fade_ms / 1000.0))
        if nf > 1:
            ramp = np.linspace(1.0, 0.0, nf)
            out = out.copy()
            out[-nf:] = out[-nf:] * (ramp if out.ndim == 1 else ramp[:, None])
    return out


def _sustained_above(env, rel, min_run):
    """Indices of env above rel x its max, ending at the last run at least min_run long:
    a short blip after the decay (a click, a release thump) doesn't extend the tail."""
    above = env > float(env.max()) * rel
    idx = np.where(above)[0]
    if idx.size == 0:
        return idx
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    long_ = np.where(ends - starts + 1 >= min_run)[0]
    last_end = ends[long_[-1]] if long_.size else ends[-1]
    return idx[idx <= last_end]


def _rms_env(mono, sr, ms=10.0):
    """Centered moving RMS of a mono magnitude signal (window ms)."""
    w = max(1, int(sr * ms / 1000.0))
    if w <= 1 or mono.size < w:
        return np.asarray(mono, dtype="float64")
    k = np.ones(w) / w
    return np.sqrt(np.convolve(np.square(np.asarray(mono, dtype="float64")), k, mode="same"))


def _remove_dc(y, sr, exact=False, thr=DC_FIX, hz=DC_HPF_HZ):
    """Remove a DC offset from each channel when it exceeds thr of the file's peak
    (relative, because peak normalization scales DC up with everything else).

    exact=True (a single-cycle waveform: it loops, has no ends) always subtracts the
    mean. A long file gets a zero-phase high-pass at hz, which adds no step at the
    start or end of a decaying hit and keeps an 808's sub. A file too short for that
    filter to settle (under 3 periods of hz) has its mean subtracted, with 2 ms / 5 ms
    fades so the shifted ends can't click."""
    if y.size == 0:
        return y
    yy = y if y.ndim == 2 else y[:, None]
    means = yy.mean(axis=0)
    peak = float(np.abs(yy).max())
    if exact:
        out = yy - means
    elif peak <= 0 or not np.any(np.abs(means) > thr * peak):
        return y
    elif len(yy) < 3 * sr / hz:
        out = _subtract_mean(yy, sr)
    else:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, hz, btype="highpass", fs=sr, output="sos")
        out = sosfiltfilt(sos, yy, axis=0)
        # a hit whose energy sits in its first moments keeps a whole-file mean the
        # filter can't see as DC: fall back to the mean
        pk = float(np.abs(out).max())
        if pk > 0 and np.any(np.abs(out.mean(axis=0)) > thr * pk):
            out = _subtract_mean(out, sr)
    return out if y.ndim == 2 else out[:, 0]


def _subtract_mean(yy, sr):
    """Per-channel offset removed, with 2 ms / 5 ms fades so the shifted ends can't click.
    The offset is the fade-weighted mean, so the faded result itself has zero mean: on a
    click a few ms long the fades cover most of the file, and subtracting the plain mean
    before fading would leave an offset behind."""
    n = len(yy)
    w = np.ones(n)
    fi, fo = min(n // 2, int(sr * 0.002)), min(n // 2, int(sr * 0.005))
    if fi > 1:
        w[:fi] = np.linspace(0.0, 1.0, fi)
    if fo > 1:
        w[-fo:] = np.minimum(w[-fo:], np.linspace(1.0, 0.0, fo))
    c = (w[:, None] * yy).sum(axis=0) / max(float(w.sum()), 1e-12)
    return (yy - c) * w[:, None]


def _rms_db(y):
    """RMS level in dBFS (-inf for silence)."""
    if y.size == 0:
        return float("-inf")
    r = float(np.sqrt(np.mean(np.square(y))))
    return 20 * np.log10(r) if r > 0 else float("-inf")


def _clamp_rms(y, ceil_db):
    """Turn a file down so its RMS is at most ceil_db (ONESHOT_RMS_CEIL_DB); never up."""
    if ceil_db is None or y.size == 0:
        return y
    r = _rms_db(y)
    if not np.isfinite(r) or r <= ceil_db:
        return y
    return y * _amp_from_db(ceil_db - r)


_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# a note with its octave: on its own ("C#3", "Cs4") or glued to a lowercase patch name
# ("warmpadF#3"; not after a capital, so a model code isn't a note), or lowercase with a
# sharp ("g#3"). Not inside a vendor code either: never after a digit or before a capital,
# so a code's letters don't read as the root
_NOTE_TOKEN = re.compile(r"(?<![A-Z0-9])(?<![A-Za-z][A-Z])([A-G])(#|b|s)?(-?\d)(?![0-9A-Z])"
                         r"|(?<![A-Za-z0-9])([a-g])(#)(\d)(?![0-9A-Z])")
# a bare key with an accidental as its own token ("Stab C#", "Chord F# 01")
_KEY_TOKEN = re.compile(r"(?<![A-Za-z0-9#])([A-G])(#|b)(?![A-Za-z0-9#])")
_CHORD_TOKEN = re.compile(r"(?<![A-Za-z])([A-G])(#|b)?(?:maj|min|m(?![a-z])|dim|aug|sus\d?|add\d+|M7|6|7|9|11|13)(?![a-z])")
# without Sononym (bare=True), a capital note letter as the name's own last token, after a
# separator and before nothing but a take number ("808_F", "Bass G", "Sub_Eb_02"): in a tonal
# category the note it plays. With Sononym the root comes from its analysis instead.
_BARE_NOTE = re.compile(r"(?<=\s)([A-G])(#|b)?(?=(?:\s+\d{1,3})?\s*$)")


def _name_multi_note(name):
    """True when a filename names more than one note ("Vocal B2-D3-E2 Amin", a sung
    line): no single root, so it isn't retuned."""
    stem = os.path.splitext(os.path.basename(name or ""))[0].replace("_", " ")
    return len({g.group(0).upper() for g in _NOTE_TOKEN.finditer(stem)}) > 1


def _name_root_midi(name):
    """MIDI note of a filename's note with an octave ("A2" -> 45), else None (and None for
    a name with several notes)."""
    probe = os.path.splitext(os.path.basename(name or ""))[0].replace("_", " ")
    m = list(_NOTE_TOKEN.finditer(probe))
    if not m or _name_multi_note(name):
        return None
    g = m[-1]
    letter, acc, octv = (g.group(1), g.group(2), g.group(3)) if g.group(1) else (g.group(4).upper(), g.group(5), g.group(6))
    return _PC[letter] + (1 if acc in ("#", "s") else -1 if acc == "b" else 0) + (int(octv) + 1) * 12


def _retuned_stem(stem, retune, bare=False):
    """A file name that says the note the file now plays: after retuning by `retune`
    semitones to C, its named note ("Pad_A2", "warmpadF#3", "Stab C#", "Dbmaj7", and with
    bare=True "808_F") reads C ("Pad_C3", "warmpadC4", "Stab C", "Cmaj7", "808_C"), so a
    retuned file never names a note it no longer plays. Names without a note are kept."""
    if retune is None or not stem:
        return stem
    probe = stem.replace("_", " ")
    m = list(_NOTE_TOKEN.finditer(probe))
    if m:
        g = m[-1]
        if g.group(1):
            letter, acc, octv, a, b = g.group(1), g.group(2), g.group(3), g.start(1), g.end(3)
        else:
            letter, acc, octv, a, b = g.group(4).upper(), g.group(5), g.group(6), g.start(4), g.end(6)
        pc = (_PC[letter] + (1 if acc in ("#", "s") else -1 if acc == "b" else 0))
        midi = pc + (int(octv) + 1) * 12 + int(round(retune))
        return stem[:a] + f"C{midi // 12 - 1}" + stem[b:]
    m = list(_CHORD_TOKEN.finditer(probe)) or list(_KEY_TOKEN.finditer(probe)) or (
        list(_BARE_NOTE.finditer(probe)) if bare else [])
    if m:
        g = m[-1]
        return stem[:g.start(1)] + "C" + stem[g.end(2) if g.group(2) else g.end(1):]
    return stem


_LOOSE_NOTE = re.compile(r"(?<![A-Za-z])([A-G])(#|b|s)?(-?\d)(?![0-9])")


def _renamed_detected_note(stem, retune, root_midi):
    """A file retuned by its detected pitch whose name still carries that note where the
    name parser doesn't look (after a digit: "Saw2D1") says the C it now plays ("Saw2C1").
    Only a token whose note is the detected pitch class is
    rewritten, so "mod4" or a patch code stays."""
    pc = int(round(root_midi)) % 12
    hits = [g for g in _LOOSE_NOTE.finditer(stem)
            if (_PC[g.group(1)] + (1 if g.group(2) in ("#", "s") else -1 if g.group(2) == "b" else 0)) % 12 == pc]
    if not hits:
        return stem
    g = hits[-1]
    midi = pc + (int(g.group(3)) + 1) * 12 + int(round(retune))
    return stem[:g.start(1)] + f"C{midi // 12 - 1}" + stem[g.end(3):]


def _name_root_pc(name, bare=False):
    """Pitch class (0 = C) a filename names: its last note with an octave ("C#3", "Cs4",
    "Bb1"), else a chord root ("Dbmaj7", "Em"); None when it names none. Letters must be
    capitals with an octave, accidental or chord quality, so a bare letter ("Kit B", a
    pack's "C" prefix) doesn't count; with bare=True (without Sononym, in a tonal category)
    a capital letter as the name's last token does ("808_F", _BARE_NOTE)."""
    stem = os.path.splitext(os.path.basename(name or ""))[0].replace("_", " ")
    if _name_multi_note(name):
        return None
    m = list(_NOTE_TOKEN.finditer(stem))
    if m:
        g = m[-1].groups()
        letter, acc = (g[0], g[1]) if g[0] else (g[3].upper(), g[4])
    else:
        m = list(_CHORD_TOKEN.finditer(stem)) or list(_KEY_TOKEN.finditer(stem)) or (
            list(_BARE_NOTE.finditer(stem)) if bare else [])
        if not m:
            return None
        letter, acc = m[-1].group(1), m[-1].group(2)
    return (_PC[letter] + (1 if acc in ("#", "s") else -1 if acc == "b" else 0)) % 12


def _retune_shift(category, filename, tune, detect=False, bare=False):
    """(semitones, source) that take a tonal one-shot's root to the nearest C, or (None,
    None). The filename's note first, the detected pitch in RETUNE_DETECT_CATS (any tonal
    category with detect=True: a pYIN pitch, _fill_roots); a detected pitch that agrees
    with the name adds its fine tuning."""
    if category not in RETUNE_CATS or _name_multi_note(filename):
        return None, None
    pc, src = _name_root_pc(filename, bare), "name"
    if pc is None:
        if (category not in RETUNE_DETECT_CATS and not detect) or tune is None:
            return None, None
        pc, src = int(round(tune)) % 12, "detect"
    d = (-pc) % 12
    if d > 6:
        d -= 12
    frac = (tune - round(tune)) if (tune is not None and int(round(tune)) % 12 == pc) else 0.0
    cap = RETUNE_MAX_SEMIS.get(category)
    if cap is not None and abs(d) > cap:              # whole semitones: fine tuning never tips it
        return None, None            # a voice or acoustic this far from C keeps its pitch and note
    return round(d - frac, 3), src


def _repitch(y, sr, semitones):
    """Resample so the file plays `semitones` higher (tape-style: length changes too)."""
    if not semitones or abs(semitones) < 0.01 or y.size == 0:
        return y
    from fractions import Fraction
    from scipy.signal import resample_poly
    f = Fraction(2 ** (-semitones / 12)).limit_denominator(1000)
    return resample_poly(y, f.numerator, f.denominator, axis=0)


def _to_mono(y, force=False, side_db=NEAR_MONO_SIDE_DB, judge=None):
    """Stereo to mono when forced (MONO_CATS) or when the sides are under side_db of the
    middle. A forced file whose channels cancel (phase-inverted) keeps its left channel
    instead of summing to near silence. Mono and waves pass through. judge: the signal
    the width is measured on (the DC-free one), when not y itself."""
    if y.ndim != 2 or y.shape[1] != 2:
        return y
    mid = (y[:, 0] + y[:, 1]) / 2
    side = (y[:, 0] - y[:, 1]) / 2
    rm, rs = float(np.sqrt(np.mean(mid ** 2))), float(np.sqrt(np.mean(side ** 2)))
    if rm <= 0 and rs <= 0:
        return y[:, 0]
    jm, js = rm, rs
    if judge is not None:
        jm = float(np.sqrt(np.mean(((judge[:, 0] + judge[:, 1]) / 2) ** 2)))
        js = float(np.sqrt(np.mean(((judge[:, 0] - judge[:, 1]) / 2) ** 2)))
    if not force and js > jm * _amp_from_db(side_db):
        return y                                     # real stereo: keep it
    if rm < rs * _amp_from_db(-6.0):
        return y[:, 0]                               # the channels cancel: keep one
    return mid


def _process_audio(y, sr, kind, trim_lead, trim_tail, norm_mode, lead_floor_db=None, mono=False,
                   retune=None, dur_cap=None):
    """Export DSP: mono or stereo -> DC -> trim -> DC again -> normalize. The second DC pass judges the
    file's final extent: trimming a long quiet tail concentrates the mean, so a source
    just under the threshold can end over it once trimmed."""
    exact = kind == "waves"
    if retune and not exact:
        y = _repitch(y, sr, retune)          # root to C (RETUNE_CATS)
        if dur_cap and len(y) > int(dur_cap * sr):
            # retuning down lengthens a file (tape-style): hold the category's length cap
            # with a fade, so a pad that fit before still fits (DUR_CAP)
            y = y[:int(dur_cap * sr)].copy()
            nf = min(len(y) // 10, int(0.25 * sr))
            if nf > 1:
                ramp = np.linspace(1.0, 0.0, nf)
                y[-nf:] = y[-nf:] * (ramp if y.ndim == 1 else ramp[:, None])
    if not exact:
        # judged on the DC-free signal: a channel-to-channel DC difference reads as stereo
        # width, so a file whose only "width" was DC would keep two channels and come out
        # near-mono (the mix itself is unchanged: mono first, then the DC pass)
        judge = _remove_dc(y, sr) if (y.ndim == 2 and y.shape[1] == 2) else None
        y = _to_mono(y, force=mono, judge=judge)
    y = _remove_dc(y, sr, exact=exact)
    if kind == "instrument":
        y = _trim_edges(y, sr, trim_lead, trim_tail, tail_floor_db=INSTR_TAIL_FLOOR_DB,
                        fade_ms=TAIL_FADE_MS)
    else:
        # the tail on a 10 ms RMS envelope (ONESHOT_TAIL_RMS): a noise floor's spikes
        # can't hold on to near-silence
        y = _trim_edges(y, sr, trim_lead, trim_tail, lead_floor_db=lead_floor_db,
                        tail_floor_db=TRIM_FLOOR_DB if ONESHOT_TAIL_RMS else None,
                        fade_ms=TAIL_FADE_MS if ONESHOT_TAIL_RMS else 0.0)
    if trim_tail:
        # before the DC pass: fading a short kick cut mid-cycle shifts its mean, which
        # the DC pass then removes (after it, the shift would stay as DC)
        y = _end_fade(y, sr)
    if not exact:
        y = _remove_dc(y, sr)
    if lead_floor_db is not None and trim_lead:
        # judge a field recording's lead again once the DC / rumble pass has run: the
        # high-pass can lower a room's level below the floor that held the first trim
        y = _trim_edges(y, sr, True, False, lead_floor_db=lead_floor_db)
    if trim_tail:
        # ...and after it: the high-pass lifts a faded end off zero when the file carried
        # DC
        yf = _end_fade(y, sr)
        if yf is not y and not exact:
            # the fade can move the mean of a low, square-ish tone: take it back with the
            # fade-weighted mean, which keeps the end at zero
            yy = yf if yf.ndim == 2 else yf[:, None]
            pk = float(np.abs(yy).max())
            if pk > 0 and np.any(np.abs(yy.mean(axis=0)) > DC_FIX * pk):
                yy = _subtract_mean(yy, sr)
                yf = yy if yf.ndim == 2 else yy[:, 0]
        y = yf
    y = _normalize(y, norm_mode, sr=sr)
    # last: a limited loop's peak comes down, so ends that were quiet beside it may not be
    return y if exact else _edge_fades(y, sr, loop=kind == "loop")


def _edge_fades(y, sr, loop=False):
    """A file that starts on an audible sample clicks when triggered: fade its first
    EDGE_FADE_MS in (half a millisecond keeps the transient). A loop's end is its loop
    point, so it fades out the same way when either end is hot, and FWDLOOP joins through
    zero (a hard start, or a jump at the loop point, clicks)."""
    if y.size == 0 or len(y) < 8:
        return y
    a = np.abs(y) if y.ndim == 1 else np.abs(y).max(axis=1)
    peak = float(a.max())
    if peak <= 0:
        return y
    hot = peak * _amp_from_db(EDGE_HOT_DB)
    nf = max(2, min(len(y) // 4, int(round(sr * EDGE_FADE_MS / 1000.0))))
    head, tail = a[0] > hot, a[-1] > hot
    if not (head or (loop and tail)):
        return y
    out = y.astype("float64", copy=True)
    ramp = np.linspace(0.0, 1.0, nf, endpoint=False)
    r = ramp if out.ndim == 1 else ramp[:, None]
    if head or loop:
        out[:nf] *= r
    if loop:
        out[-nf:] *= r[::-1]
    return out


def _end_fade(y, sr, hot_db=END_HOT_DB, fade_ms=END_FADE_MS):
    """Fade out the last fade_ms of a one-shot whose last millisecond still rises above
    hot_db (relative to peak): a sound cut off while audible clicks."""
    if y.size == 0:
        return y
    mono = np.abs(y if y.ndim == 1 else np.abs(y).max(axis=1))
    peak = float(mono.max())
    # the last millisecond, or the last eighth of a very short hit (short files click
    # too)
    n1 = max(1, min(int(sr * 0.001), len(mono) // 8))
    if peak <= 0 or len(mono) < 16:
        return y
    if float(mono[-n1:].max()) <= peak * _amp_from_db(hot_db):
        return y
    # a squared ramp reaches silence faster at the end; very short hits fade over their last
    # half (a fade of a few samples still ends audibly)
    nf = min(len(y) // 2, max(2, int(sr * fade_ms / 1000.0)))
    ramp = np.linspace(1.0, 0.0, nf) ** 2
    out = y.copy()
    out[-nf:] = out[-nf:] * (ramp if out.ndim == 1 else ramp[:, None])
    return out


def _normalize(y, mode, max_gain_db=MAX_GAIN_DB, sr=44100):
    """Peak- or RMS-normalize, capping amplification (so near-silent junk isn't
    blown up) and holding the true peak at PEAK_CEILING_DB for RMS mode."""
    if mode is None or y.size == 0:
        return y
    peak = float(np.abs(y).max())
    if peak <= 0:
        return y
    ceiling = _amp_from_db(PEAK_CEILING_DB)
    if mode == "peak":
        gain = ceiling / peak
    else:  # rms
        rms = float(np.sqrt(np.mean(np.square(y))))
        if rms <= 0:
            return y
        # a loop whose peak stops it short of LOOP_RMS_DB may have up to LOOP_LIMIT_DB of its
        # peaks limited (a peaky loop would otherwise sit well under the target)
        gain = min(_amp_from_db(LOOP_RMS_DB) / rms, ceiling / peak * _amp_from_db(LOOP_LIMIT_DB))
        gain = min(gain, _amp_from_db(max_gain_db))
        if gain * peak > ceiling:
            return np.clip(_limit(y * gain, ceiling, sr), -1.0, 1.0)
    gain = min(gain, _amp_from_db(max_gain_db))
    return np.clip(y * gain, -1.0, 1.0)


def _limit(y, ceiling, sr, ms=LIMIT_MS):
    """Hold every sample under `ceiling` with a smooth gain: the needed gain (ceiling / |x|,
    at most 1) min-filtered over 2*ms around each sample, then averaged over the same
    width. Every sample of the average at t comes from a window holding t, so the gain at
    t never exceeds what t needs; attack and release are each about `ms`."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    a = np.abs(y) if y.ndim == 1 else np.abs(y).max(axis=1)
    need = np.minimum(1.0, ceiling / np.maximum(a, 1e-12))
    w = max(1, int(sr * ms / 1000.0)) * 2 + 1
    g = uniform_filter1d(minimum_filter1d(need, w, mode="nearest"), w, mode="nearest")
    g = np.minimum(g, need)                      # (float rounding at the edges)
    return y * (g if y.ndim == 1 else g[:, None])


def _export_policy(kind):
    """(norm_mode, (trim_lead, trim_tail)) per category kind."""
    if kind == "loop":
        return "rms", (False, False)        # preserve loop length; even out level
    if kind == "instrument":
        return "peak", (True, True)          # align the onset; tail trimmed only below -60 dB
    if kind == "waves":
        return "peak", (False, False)        # a single cycle loops: never cut a sample
    return "peak", (True, True)              # oneshot / gated drum hits


def _scale_doc(scale) -> dict:
    """What a library-scaled build records in its manifest ("scale"): the factor, the library's
    usable samples and the scaled sizes verify holds it to."""
    from .curate_config import (LIBRARY_PER_MASTER, SCALED_FOLDER_FILES, SCALED_FOLDER_MIN_FILES,
                                SCALED_KEEP_ALL, SCALED_MIN_FILES)
    return dict(factor=round(float(scale["factor"]), 6), samples=int(scale.get("samples") or 0),
                per_master=LIBRARY_PER_MASTER, keep_all=SCALED_KEEP_ALL, min_files=SCALED_MIN_FILES,
                folder_files=SCALED_FOLDER_FILES, folder_min_files=SCALED_FOLDER_MIN_FILES)


class TooFewSamples(RuntimeError):
    """A category whose candidates can't fill one folder (fewer than its folder floor after
    the filters): a whole build leaves it out, as a library without those sounds would."""


def left_out_doc(category, status, n, floor, stats, why, **extra) -> dict:
    """What `fourier why` reads for one category (packs/why_log.py): how many candidates it
    found and kept, its minimum, and why each candidate it left out was left out."""
    readmitted = why.get(why_log.READMITTED) or {}
    return dict(category=category, status=status, kept=n, need=floor,
                found=int(stats.get("total", 0)),
                counts={k: int(v) for k, v in stats.items() if isinstance(v, int) and v},
                out={reason: {str(k): v for k, v in ids.items()} for reason, ids in why.items()
                     if ids and reason != why_log.READMITTED},
                **({"readmitted": {str(k): v for k, v in readmitted.items()}} if readmitted else {}),
                **extra)


def _clap_doc(ids, rehomed=(), sound=None) -> dict:
    """The why log's record of the samples the CLAP fallback homed in a category (none with
    Sononym): {"clap_homed": [ids], "clap_rehomed": [ids]}, and those the sound model homed
    there, with its probability: {"sound_homed": {id: p}}; or nothing."""
    out = {"clap_homed": sorted(int(i) for i in ids)} if ids else {}
    if rehomed:
        out["clap_rehomed"] = sorted(int(i) for i in rehomed)
    if sound:
        out["sound_homed"] = {str(int(i)): float(p) for i, p in sorted(sound.items())}  # type: ignore[assignment]
    return out


def too_few_line(doc) -> str:
    """'6 found; 5 near-duplicates, 0 over the per-vendor share; need 6': what took a
    category's candidates below its minimum."""
    c = doc.get("counts") or {}
    if not doc.get("found"):
        return f"none found; need {doc.get('need')}"
    parts = [f"{c.get('near_dup', 0)} near-duplicates", f"{c.get('vendor_capped', 0)} over the per-vendor share"]
    parts += [f"{c[k]} {what.format(cat=doc.get('category', '?'))}" for k, what in why_log.FILTER_WORDS.items()
              if k not in ("near_dup", "vendor_capped") and c.get(k)]
    return f"{doc.get('found', 0)} found; {', '.join(parts)}; need {doc.get('need')}"


def build_taxonomy(session, category, out_dir, per_family=DEFAULT_PER_FAMILY,
                   clap_z=CLAP_Z, transcode=True, describe=True, model=DEFAULT_MODEL,
                   rebuild_index=False, homes=None, support=None, votes=None,
                   exclude_ids=None, loudness=True, return_entries=False, budget_override=None,
                   scale=None, log=print):
    """Build one CATEGORY/family/file tree under out_dir.

    scale: {"factor": f, "samples": n} for a library-scaled build (packs/scale.py), only ever
    given when f < 1; without it (f = 1, scale = "off", an additive build) nothing below
    changes. Returns a summary dict, or (summary, file_entries) when return_entries is True
    (parallel builds collect entries so the parent writes the manifest once)."""
    import soundfile as sf
    from sklearn.decomposition import PCA

    import time as _time
    _t0 = _time.perf_counter()
    category = category.upper()
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; known: {', '.join(CATEGORIES)}")
    cfg = CATEGORIES[category]
    budget = int(cfg.get("budget") or BUDGETS.get(category, 200))
    if budget_override is not None:          # an additive build's allowance
        budget = int(budget_override)
    floor = int(cfg.get("min_per_family", MIN_PER_FAMILY))
    # a folder is allocated at least FOLDER_MIN_FILES when its pool has them (a band's
    # folder minimum can keep a small folder apart)
    alloc_floor = floor if cfg.get("tempo_bands") else max(floor, FOLDER_MIN_FILES)
    ceil = int(cfg.get("max_per_family", MAX_PER_FAMILY))
    # a library-scaled build (packs/scale.py): a smaller minimum (the gates' and caps' floors
    # follow it), smaller folders and fewer of them; the budget follows the pool, below
    _f = float(scale["factor"]) if scale and float(scale.get("factor", 1)) < 1 else None
    _fmin, _bmf = FOLDER_MIN_FILES, int(cfg.get("band_min_folders", 1))
    if _f is not None:
        from .curate_config import SCALED_FOLDER_MIN_FILES, SCALED_MIN_FILES
        _preset_budget = budget
        floor = min(floor, SCALED_MIN_FILES)
        _fmin, _bmf = SCALED_FOLDER_MIN_FILES, 1
        alloc_floor = floor if cfg.get("tempo_bands") else max(floor, _fmin)

    # Auto-refresh a stale CLAP index (embeddings added since it was built), or force
    # a rebuild when asked. Parallel --all refreshes once in the parent, so workers
    # arrive here with a fresh index and this is a no-op.
    stale, idx_n, db_n = index_stale(session)
    if stale or rebuild_index:
        from ..analysis.clap_features import build_index
        log(f"CLAP index {'stale' if stale else 'rebuild forced'} "
            f"({idx_n} indexed vs {db_n} in DB); rebuilding index...")
        build_index(session)

    ids, emb = load_index()
    emb = emb.astype("float32")
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    id2row = {int(i): r for r, i in enumerate(ids)}

    if homes is None or support is None:
        homes, support, votes = _get_homes(session, emb_n, id2row, log)
    votes = votes or {}
    _vendors.prepare(session)            # each library folder's layout (vendors = "auto" only)

    rows = _fetch_rows(session, cfg, category)
    _why, _gate = {}, {}
    if cfg["kind"] in ("oneshot", "loop") and homes:
        # samples routed here from another classifier bucket (keys chords -> STABS)
        _have = {r.id for r in rows}
        _extra = sorted(sid for sid, h in homes.items() if h == category and sid not in _have)
        if _extra:
            rows = _sort_rows(rows + _fetch_rows_by_id(session, _extra))
    else:
        _extra = []
    _xs = set(_extra)
    # samples the CLAP fallback homed here (without Sononym: no name rule matched them)
    _clap_ids = {sid for sid in _xs if (votes.get(sid) or {}).get("clap") == category}
    # samples the sound model homed here (without Sononym): {id: its probability}
    _sound = {sid: vt["sound"][1] for sid, vt in votes.items() if (vt or {}).get("sound")
              and vt["sound"][0] == category and homes.get(sid) == category}
    # loops CLAP re-homed here (without Sononym: _clap_rehome, either way between the loop categories)
    _rehomed = {sid for sid, vt in votes.items() if (vt or {}).get("rehomed") == category
                and homes.get(sid) == category}
    rec, fm, stats = _select_records(rows, cfg, emb_n, id2row, homes, support, category,
                                     # (chord stabs are STABS' own, not a routed share; nor
                                     # are the CLAP fallback's homes)
                                     routed_ids={r.id for r in rows if r.id in _xs and r.id not in _clap_ids
                                                 and r.id not in _sound
                                                 and not (
                                         (category == "STABS" and _stab_named(r.filename))
                                         or (category == SCRATCH_HOME and _scratch_named(r.filename, r.rel_path))
                                         or _drum_named_in(r.filename, r.rel_path or r.path,
                                                           _fallback(session)) == category)},
                                     clap_homed=_clap_ids | _rehomed | set(_sound), rehomed=_rehomed,
                                     exclude_ids=exclude_ids,
                                     wave_hashes=_wave_hashes(session),
                                     mirrors=_mirror_dups(session),
                                     floor=floor, why=_why, gate=_gate,
                                     fallback=_fallback(session))
    n = len(rec)
    # each CLAP-gated candidate's scores, for `fourier why` (a gated category only)
    _gate_doc = dict(gate={str(k): v for k, v in _gate.items()},
                     gate_min=cfg.get("break_min", 0.30)) if _gate else {}
    if n < floor:
        _why_doc = left_out_doc(category, "too_few", n, floor, stats, _why, **_gate_doc,
                                **_clap_doc(_clap_ids, _rehomed, _sound),
                                # (without Sononym: the candidates it had, which `fourier why
                                # --unrecognized` doesn't count as placed nowhere)
                                **({"held": sorted(int(x["id"]) for x in rec)} if _fallback(session) else {}))
        why_log.write(out_dir, category, _why_doc)
        raise TooFewSamples(f"{category}: {too_few_line(_why_doc)}")
    if _f is not None:
        from .scale import category_budget
        budget = category_budget(budget, _f, n)
    # files the previous build kept here stay while they're still in the pool (_stable_pick)
    _prev_srcs = set().union(*_previous_families(category).values()) if _previous_families(category) else set()
    for r in rec:
        r["incumbent"] = r["path"] in _prev_srcs
    log(f"{category}: {n} samples ({stats['twins']} twins collapsed, "
        f"{stats.get('near_dup', 0)} near-dups pruned"
        + (f", {stats['note_gated']} out-of-range notes dropped" if stats.get('note_gated') else "")
        + (f", {stats['loop_gated']} mislabeled loops dropped" if stats.get('loop_gated') else "")
        + (f", {stats['chain_gated']} sample chains dropped" if stats.get('chain_gated') else "")
        + (f", {stats['nondrum_gated']} non-drum loops dropped" if stats.get('nondrum_gated') else "")
        + (f", {stats['phrase_out']} phrases left to {PHRASES_CATEGORY}" if stats.get('phrase_out') else "")
        + (f", {stats['not_phrase']} non-phrases dropped" if stats.get('not_phrase') else "")
        + (f", {stats['pinned']} pinned Keeps" if stats.get('pinned') else "")
        + (f", {stats['mirror_copies']} mirror copies" if stats.get('mirror_copies') else "")
        + (f", {stats['style_excluded']} off-style loops" if stats.get('style_excluded') else "")
        + (f", {stats['soft_layers']} soft layers" if stats.get('soft_layers') else "")
        + (f", {stats['gated_out']} loops gated out" if cfg["kind"] == "loop" else "")
        + (f", {stats['too_harmonic']} too harmonic" if stats.get("too_harmonic") else "")
        + (f", {stats['clap_readmitted']} gated loops readmitted to reach the minimum of {floor}"
           if stats.get("clap_readmitted") else "")
        + (f", {stats['too_harmonic_readmitted']} too-harmonic loops readmitted to reach the minimum "
           f"of {floor}" if stats.get("too_harmonic_readmitted") else "") + ")")

    if cfg.get("tempo_bands"):
        _ts = Counter(r.get("bpm_src") or "unknown" for r in rec)
        log(f"{category} tempo: " + ", ".join(f"{k} {v}" for k, v in _ts.most_common()))
    R = np.array([r["row"] for r in rec])
    ncomp = min(50, max(2, len(R) - 1), emb.shape[1])
    with _limit_threads():
        P = PCA(n_components=ncomp, svd_solver="full", random_state=CURATION_SEED).fit_transform(emb[R])
    fmax = int(cfg.get("folder_max", FOLDER_MAX))
    kmax = min(int(cfg.get("kmax", 30)), fmax)
    if _f is None:
        kmax = _folder_kmax(kmax, budget, len(rec))
        kmin = _family_kmin(budget, ceil, int(cfg.get("kmin", 8)), kmax, n=len(rec), floor=floor)
    else:
        from .scale import family_count, tempo_band_min
        kmax = kmin = fmax = family_count(budget, len(rec), kmax)
        log(f"{category}: budget {budget} (the style's {_preset_budget} at library scale "
            f"{_f:.2g}), minimum {floor}, {kmax} folder{'s' if kmax != 1 else ''}")
    if cfg.get("band") == "fx":
        _resolve_misc_fx(rec, emb_n)
    _pos_of = {r["path"]: i for i, r in enumerate(rec)}
    _prev_members = [[_pos_of[s_] for s_ in srcs if s_ in _pos_of]
                     for _, srcs in sorted(_previous_families(category).items())]

    def _seeds(idx):
        """(centroids, folder per position of idx or -1) of the previous build's folders that
        sit (mostly) within idx, or None."""
        pos = {i: q for q, i in enumerate(idx)}
        cs, fixed = [], np.full(len(idx), -1)
        for mem in _prev_members:
            m = [i for i in mem if i in pos]
            if len(m) >= 3 and len(m) >= 0.5 * len(mem):
                for i in m:
                    fixed[pos[i]] = len(cs)
                cs.append(P[m].mean(0))
        return (np.array(cs), fixed) if cs else None
    if cfg.get("band"):
        # bands used differently on the device (closed vs open hats, impacts vs foley,
        # full breaks vs tops, notes vs chords, single cycles vs tables) are clustered
        # on their own, so a folder is one or the other: CLAP alone groups by timbre
        groups, gkeys, k0 = [], [], 0
        for band in BAND_ORDER[cfg["band"]]:
            idx = [i for i, r in enumerate(rec) if r.get("band") == band]
            if not idx:
                continue
            share = len(idx) / len(rec)
            if cfg.get("band_share"):
                # fixed shares: folders (and files, below) follow what the band is for,
                # not how big its pool happens to be
                share = cfg["band_share"].get(band, share)
            if cfg.get("tempo_bands"):
                sub, k = _cluster_by_tempo(P[idx], [rec[i].get("bpm_fold") or rec[i]["bpm"] for i in idx],
                                           cfg.get("band_tempo", {}).get(band, cfg["tempo_bands"]), floor,
                                           kmax=max(1, int(round(kmax * share))),
                                           kmin=max(1, int(round(kmin * share))),
                                           kdiv=cfg.get("kdiv", 800),
                                           files=budget * share if cfg.get("tempo_folder_files") else None,
                                           per_folder=cfg.get("tempo_folder_files"),
                                           seedfn=lambda loc, idx=idx: _seeds([idx[q] for q in loc]),
                                           band_min=None if _f is None else
                                           tempo_band_min(len(idx), budget * share))
                groups.extend([[idx[i] for i in g] for g in sub]); k0 += k
                gkeys.extend([None] * len(sub))       # tempo folders: never merged across tempos
                continue
            kb = max(1, min(max(int(round(kmax * share)), _bmf),
                            len(idx) // max(floor, 1)))
            if kb <= 1:
                groups.append(idx); gkeys.append(band); k0 += 1
                continue
            sub, k = _cluster(P[idx], kmax=kb, kmin=min(kb, max(1, int(round(kmin * share)))),
                              kdiv=cfg.get("kdiv", 800), init=_seeds(idx))
            groups.extend([[idx[i] for i in g] for g in sub]); gkeys.extend([band] * len(sub)); k0 += k
        # per-band rounding (at least one folder a band) can overshoot the cap
        groups = _cap_folders(groups, gkeys, P, fmax, _bmf)
    elif cfg.get("tempo_bands"):
        groups, k0 = _cluster_by_tempo(P, [r.get("bpm_fold") or r["bpm"] for r in rec],
                                       cfg["tempo_bands"], floor,
                                       kmax=kmax, kmin=kmin, kdiv=cfg.get("kdiv", 800), seedfn=_seeds,
                                       band_min=None if _f is None else tempo_band_min(len(rec), budget))
    else:
        groups, k0 = _cluster(P, kmax=kmax, kmin=kmin, kdiv=cfg.get("kdiv", 800),
                              init=_seeds(list(range(len(rec)))))

    dims = naming_dims(category, cfg)
    clusters = []
    if category in SIBLING_CAP_CATS:
        groups = [_cap_siblings(g, rec) for g in groups]
    if not cfg.get("tempo_bands"):
        # a folder under FOLDER_MIN_FILES merges into the nearest folder of its band
        _gb = [Counter(rec[i].get("band") for i in g).most_common(1)[0][0] if g else None for g in groups]
        _ng = len(groups)
        groups = _merge_small_groups(groups, _gb, P, _fmin, _bmf)

        # ...and so does one the budget would give fewer files than that (its pool may
        # clear the bar while its allocation doesn't)
        def _planned(gs, bs):
            sz = [len([i for i in g if rec[i]["clip"] == 0]) or len(g) for g in gs]
            if cfg.get("band_share"):
                return _allocate_banded(sz, bs, cfg["band_share"], budget, alloc_floor, ceil)
            return _allocate_budget(sz, budget, alloc_floor, ceil)
        _gb = [Counter(rec[i].get("band") for i in g).most_common(1)[0][0] if g else None for g in groups]
        groups = _merge_small_groups(groups, _gb, P, _fmin, _bmf, measure=_planned)
        # a folder the budget fills to the ceiling from a pool far bigger splits in two
        # (rather than sitting at the cap next to much smaller siblings)
        groups = _split_capped_groups(groups, P, _planned, ceil, fmax, rec)
        if len(groups) < _ng:
            log(f"{category}: {_ng - len(groups)} folder(s) under {_fmin} files merged into a neighbor")
    n_sib = 0
    for idxs in groups:
        if category in SIBLING_CAP_CATS:
            _before = len(idxs)
            idxs = _cap_siblings(idxs, rec)
            n_sib += _before - len(idxs)
        c = dict(idxs=idxs)
        for key in ["br", "atk", "dec", "sub", "noi", "har", "cr"]:
            c[key] = float(np.mean([rec[i][key] for i in idxs]))
        tv = [rec[i]["tune"] for i in idxs if rec[i]["tune"] is not None]
        c["tune"] = float(np.median(tv)) if len(tv) >= 5 else None
        if category in RETUNE_CATS:
            # retuned to C: the folder's register is the C its files now sit on
            tr = []
            for i in idxs:
                if rec[i].get("retune") is None:
                    continue
                src_note = rec[i]["tune"] if rec[i]["tune"] is not None else \
                    _name_root_midi(os.path.basename(rec[i]["path"] or ""))
                if src_note is not None:
                    tr.append(src_note + rec[i]["retune"])
            if len(tr) >= 5 and len(tr) * 2 >= len(idxs):
                c["tune"] = float(12 * round(np.median(tr) / 12))
        # medians for the envelope words' absolute gates (naming.TRAIT_GATES)
        _dv = [rec[i].get("dur") for i in idxs if rec[i].get("dur") is not None]
        c["med"] = dict(atk=float(np.median([rec[i]["atk"] for i in idxs])),
                        dur=float(np.median(_dv)) if _dv else None)
        if cfg["kind"] == "loop" and _fallback(session):     # "musical" in absolute terms too
            c["med"]["har"] = float(np.median([rec[i]["har"] for i in idxs]))
            _ch = [rec[i]["chroma"] for i in idxs if rec[i].get("chroma") is not None]
            c["med"]["chroma"] = float(np.median(_ch)) if _ch else None
        bpms = [rec[i].get("bpm_fold") or rec[i]["bpm"] for i in idxs if rec[i]["bpm"] is not None]
        c["bpm"] = float(np.median(bpms)) if len(bpms) >= (1 if cfg.get("tempo_bands") else 4) else None
        c["bpm_range"] = [int(round(min(bpms))), int(round(max(bpms)))] if bpms else None
        c["free_tempo"] = bool(cfg.get("tempo_bands")) and not bpms
        c["band"] = None
        cnt = Counter(rec[i]["pack"] for i in idxs); topp, tv2 = cnt.most_common(1)[0]
        c["share"] = tv2 / len(idxs)
        c["source"] = topp if (c["share"] >= NAME_SHARE and (topp in fm or topp in NAMEABLE_EXTRA)) else None
        if cfg["kind"] == "instrument" and not cfg.get("clap_source"):
            if cfg.get("ableton_any"):
                # name each family by its dominant orchestral instrument (violin, cello,
                # classical-guitar, ...) rather than the pack it came from
                tc = Counter(_acoustic_label(rec[i], cfg["ableton_any"]) for i in c["idxs"])
                tc.pop(None, None)
                if tc and tc.most_common(1)[0][1] / len(c["idxs"]) >= ACOUSTIC_SOURCE_MIN:
                    c["source"] = re.sub(r"[^a-z0-9]+", "-", tc.most_common(1)[0][0].lower()).strip("-")
                elif tc:
                    c["source"] = None          # no instrument is most of the folder
                else:
                    c["source"] = _instrument_label(topp)
            else:
                c["source"] = _instrument_label(topp)
        if cfg.get("name_instruments"):
            _lab = _phrase_inst_label if cfg["name_instruments"] == "phrase" else _mallet_label
            ic = Counter(_lab(os.path.basename(rec[i]["path"] or ""), rec[i].get("ab"))
                         for i in idxs)
            lab, ln = max(((k, v) for k, v in ic.items() if k), key=lambda kv: (kv[1], kv[0]),
                          default=(None, 0))
            c["inst"] = lab if ln / len(idxs) >= 0.5 else None
        if cfg.get("band"):
            c["band"] = Counter(rec[i].get("band") for i in idxs).most_common(1)[0][0]
            c["band_type"] = cfg["band"]
        if cfg["kind"] == "waves":
            srcs = Counter(rec[i].get("wave_src") for i in idxs)
            s0, sn = srcs.most_common(1)[0]
            c["source"] = s0 if s0 and sn / len(idxs) >= 0.5 else None
        if cfg.get("clap_source"):
            c["source"] = None
        clusters.append(c)
    if n_sib:
        log(f"{category}: {n_sib} extra notes of multisample sets left out (at most {SIBLING_CAP} a set a folder)")
    _rank_traits(clusters, dims, cfg["kind"])

    # CLAP distinctive-phrase anchoring (z-scored across clusters). The category's
    # gating/anchor phrases plus naming-only ones (NAME_PHRASES never affect selection).
    primary = list(dict.fromkeys(cfg["phrases"]))
    phrases = primary + [p for p in NAME_PHRASES.get(category, []) if p not in primary]
    T = np.stack([embed_text(p) for p in phrases])
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    cents = np.array([emb_n[[rec[i]["row"] for i in c["idxs"]]].mean(0) for c in clusters])
    cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    S = cents @ T.T
    Zp = (S - S.mean(0)) / (S.std(0) + 1e-9)
    _ks = [np.mean([bool(KEYBOARD_NAME_RE.search(os.path.basename(rec[i]["path"] or "")))
                    for i in c["idxs"]]) for c in clusters]
    Zp[_keys_phrase_mask(_ks, phrases)] = -np.inf
    if cfg.get("band"):
        # a band's folders are only named by phrases that fit the band
        for j, c in enumerate(clusters):
            Zp[j, [not _band_phrase_ok(cfg["band"], c.get("band"), p) for p in phrases]] = -np.inf
    if category in KEYS_EXCLUDE:
        # keys can't live here, so a keys phrase would mislabel (a piano phrase on a
        # folder of basic waveforms)
        Zp[:, _keys_phrase_cols(phrases)] = -np.inf
    if NAME_EXCLUDE_WORDS.get(category):
        _xw = NAME_EXCLUDE_WORDS[category]
        Zp[:, [bool(set(re.split(r"[^a-z0-9]+", p.lower())) & _xw) for p in phrases]] = -np.inf

    # export (the category's folder is cleared once its picks are known to be on this
    # machine, below)
    kroot = Path(out_dir) / category

    def medoid_pick(idxs, want):
        _pz = {i for i in idxs if rec[i].get("pin")}
        if _pz:
            return _with_pins(list(idxs), want, _pz, medoid_pick)
        return _stable_pick(idxs, want, rec, P)

    # budget distribution: split the category budget across families by cluster size
    _sizes = [len([i for i in c["idxs"] if rec[i]["clip"] == 0]) or len(c["idxs"]) for c in clusters]
    if cfg.get("band_share"):
        _inc = ([sum(1 for i in c["idxs"] if rec[i].get("incumbent") and rec[i]["clip"] == 0) for c in clusters]
                if _prev_srcs else None)
        _allocs = _allocate_banded(_sizes, [c.get("band") for c in clusters], cfg["band_share"],
                                   budget, alloc_floor, ceil, prefer=_inc)
    else:
        _allocs = _allocate_budget(_sizes, budget, alloc_floor, ceil)
    if cfg.get("keep_all_bands"):
        # a band kept whole (DRUMLOOPS' classic breaks): every file of it, the rest of the
        # budget shrinking to match, largest folders first
        keep_b = set(cfg["keep_all_bands"])
        for j, c in enumerate(clusters):
            if c.get("band") in keep_b:
                _allocs[j] = min(_sizes[j], ceil)
        others = [j for j, c in enumerate(clusters) if c.get("band") not in keep_b]
        while sum(_allocs) > budget and others:
            j = max(others, key=lambda j: (_allocs[j], -j))
            if _allocs[j] <= floor:
                break
            _allocs[j] -= 1
    if _f is not None:
        # a scaled budget can be smaller than its folders' floors add up to (a band keeps its
        # folder): the largest folders give way, never below a file
        while sum(_allocs) > budget:
            j = max(range(len(_allocs)), key=lambda j: (_allocs[j], -j))
            if _allocs[j] <= 1:
                break
            _allocs[j] -= 1

    if _prev_srcs:
        _allocs = _allocs_for_incumbents(_allocs, clusters, rec, _sizes, alloc_floor, ceil,
                                         keep_bands=set(cfg.get("keep_all_bands") or ()))
    # the files each folder will hold; its name is judged on these (what you'll browse)
    for c, want in zip(clusters, _allocs):
        c["_pick"] = medoid_pick(c["idxs"], want)
    if _prev_srcs:
        _kept = sum(1 for c in clusters for i in c["_pick"] if rec[i]["incumbent"])
        _avail = sum(1 for r in rec if r["incumbent"])
        _infold = sum(1 for c in clusters for i in c["idxs"] if rec[i]["incumbent"])
        _slots = sum(min(a, sum(1 for i in c["idxs"] if rec[i]["incumbent"])) for c, a in zip(clusters, _allocs))
        log(f"{category}: kept {_kept} of the previous build's {len(_prev_srcs)} files "
            f"({_avail} still in the pool, {_infold} in folders, {_slots} with a slot)")
    if category not in SET_TOGETHER_SKIP:
        _n_set = _gather_sets(clusters, rec, ceil, min_left=_fmin,
                              set_cap=SIBLING_CAP if category in SIBLING_CAP_CATS else None)
        if _n_set:
            log(f"{category}: {_n_set} set member(s) moved to their set's folder")
    # a drum folder is named for its drum machine when most of its files come from one
    if category in DRUM_MACHINE_CATS:
        for c in clusters:
            _pk = c["_pick"] or c["idxs"]
            _ml = Counter(_machine_label(rec[i]["path"]) for i in _pk)
            _ml.pop(None, None)
            if _ml:
                lab, n_ = max(_ml.items(), key=lambda kv: (kv[1], kv[0]))
                if n_ / len(_pk) >= MACHINE_SHARE:
                    c["inst"] = f"{lab} {CATEGORY_NOUN.get(category, category.lower())}"
    if category in SYNTH_MACHINE_CATS:
        for c in clusters:
            _pk = c["_pick"] or c["idxs"]
            for lab, _rx in _SYNTH_MACHINE_RX:
                if sum(bool(_rx.search(rec[i]["path"] or "")) for i in _pk) / len(_pk) >= MACHINE_SHARE:
                    c["inst"] = lab
                    break
    # a phrase names a folder only if it describes most of its files, and a genre or
    # drum-machine word only if the files' paths say so too (curate_config NAME_SUPPORT_MIN)
    for j, c in enumerate(clusters):
        V = emb_n[[rec[i]["row"] for i in c["_pick"] or c["idxs"]]]
        sup = _phrase_support(V, T, np.ones(len(phrases), bool) if category in NAME_SUPPORT_FULL
                              else np.isfinite(Zp[j]))
        c["_support"] = sup
        rels = [genre_evidence(category, library_rel(rec[i]["path"] or ""),
                               rec[i].get("bpm_fold") or rec[i].get("bpm"), rec[i].get("band"))
                for i in c["_pick"] or c["idxs"]]
        texts = [library_rel(rec[i]["path"] or "") + " | " + " ".join(rec[i].get("ab") or [])
                 for i in c["_pick"] or c["idxs"]]
        for q, share in sup.items():
            if share < NAME_SUPPORT_MIN or any(
                    np.mean([bool(rx.search(r)) for r in rels]) < GENRE_SUPPORT_MIN
                    for rx in _genre_backers(phrases[q])) or any(
                    np.mean([bool(rx.search(t)) for t in texts]) < GENRE_SUPPORT_MIN
                    for rx in _inst_backers(phrases[q])):
                Zp[j, q] = -np.inf
    name_families(clusters, category, cfg, Zp, phrases, clap_z=clap_z, dims=dims,
                  n_primary=len(primary))
    n_sticky = _apply_sticky_names(clusters, category, rec, cfg)
    if n_sticky:
        log(f"{category}: {n_sticky} folder name(s) kept from the previous build")
    for c in clusters:
        _q = phrases.index(c["clap_phrase"]) if c.get("clap_phrase") else None
        c["clap_support"] = round(c["_support"].get(_q, 0.0), 2) if _q is not None else None

    descriptions = _describe(clusters, category, model) if describe else {}

    norm_mode, (trim_lead, trim_tail) = _export_policy(cfg["kind"])

    _cache_hits = [0]                           # processed-audio cache hits (timings.py)
    _fb = _fallback(session)                    # without Sononym: names read with bare notes

    def export_one(src, dest, pinned=False, retune=None, root_src=None, prev_name=None, root_midi=None,
                   bpm=None):
        b = os.path.basename(src); stem, ext = os.path.splitext(b)
        _was = _note_word(stem, bare=_fb) if (_fb and root_src is not None and retune is not None) else None
        if root_src == "name":
            stem = _retuned_stem(stem, retune, bare=_fb)   # the name says the note it now plays
        elif retune is not None and root_midi is not None:
            stem = _renamed_detected_note(stem, retune, root_midi)
        # the one name this file carries in the build and on every device (without Sononym, a
        # retuned file whose name now says C, beside another C, says the note it was: "_from-as",
        # not a number that reads as an octave)
        taken = {p.name.lower() for p in dest.iterdir()} if dest.exists() else set()
        out = dest / _choose_outname(stem, taken, prev_name, namer=_namer(bpm), alt=_was)

        def _verbatim():
            if ext.lower() == ".wav" or not transcode:
                shutil.copy2(src, out)
            else:
                data, sr = sf.read(src)
                sf.write(str(out), data, sr, subtype="PCM_16")

        if not loudness:
            _verbatim()
            return out
        from . import audiocache
        ck = audiocache.key(src, v=audiocache.EXPORT_VERSION, kind=cfg["kind"], trim=(trim_lead, trim_tail),
                            norm=norm_mode, lead=cfg.get("trim_lead_db"), mono=category in MONO_CATS,
                            retune=retune, pinned=pinned, quiet=QUIET_RMS_DB,
                            ceil=ONESHOT_RMS_CEIL_DB.get(category),
                            dur_cap=DUR_CAP.get(category) if retune else None)
        got = audiocache.lookup(ck, out)
        if got == "hit":
            _cache_hits[0] += 1
            return out
        if got == "quiet":
            return None
        try:
            data, sr = sf.read(src, always_2d=False)
            try:
                sub = sf.SoundFile(src).subtype
            except Exception:
                sub = None
            y = _process_audio(np.asarray(data, dtype="float64"), sr, cfg["kind"],
                               trim_lead, trim_tail, norm_mode, lead_floor_db=cfg.get("trim_lead_db"),
                               mono=category in MONO_CATS, retune=retune,
                               dur_cap=DUR_CAP.get(category) if retune else None)
            if norm_mode == "peak" and cfg["kind"] != "waves" and not pinned and _rms_db(y) < QUIET_RMS_DB:
                audiocache.store(ck, None)
                return None                  # mostly silence: not worth a slot
            if norm_mode == "peak":
                y = _clamp_rms(y, ONESHOT_RMS_CEIL_DB.get(category))   # no outlier hotter than its neighbors
            # Preserve integer PCM depth; render FLOAT/DOUBLE (and anything else) to
            # PCM_24. This keeps the build device-appropriate (Digitakt/M8 want PCM)
            # AND deterministic: libsndfile stamps a float WAV's PEAK chunk with a
            # creation time, so preserving FLOAT would make byte-identical audio hash
            # differently run-to-run. PCM carries no PEAK chunk.
            subtype = sub if sub in ("PCM_16", "PCM_24", "PCM_32") else "PCM_24"
            sf.write(str(out), y, sr, subtype=subtype)
            audiocache.store(ck, out)
        except Exception:
            # never fail a file on a DSP/read hiccup -- fall back to verbatim/transcode, and
            # record it (verify fails a build with unprocessed files)
            if out.exists():
                try:
                    out.unlink()
                except Exception:
                    pass
            _verbatim()
            dsp_fallback.add(str(out))
        return out

    # every pick is on this machine before the first read of a source (pitch checks,
    # export): a read that starts a cloud drive's download can fail and fall back to a
    # verbatim copy or skip a check, so two builds of one library could differ. A pick
    # that won't download stops the build before anything is written.
    from ..safety import ensure_local
    _n_cloud = ensure_local([rec[i]["path"] for c in clusters for i in c["_pick"]],
                            log=lambda m: log(f"{category}: {m}"))
    if kroot.is_dir():
        shutil.rmtree(kroot)
    if category in RETUNE_CATS:
        _nc = _check_name_pitches([i for c in clusters for i in c["_pick"]], rec)
        if _nc:
            log(f"{category}: {_nc} file(s) whose named note the audio contradicts keep their pitch and name")
        _nf = _fill_roots(category, [i for c in clusters for i in c["_pick"]], rec)
        if _nf:
            log(f"{category}: {_nf} file(s) with no named or Sononym root retuned by their pYIN pitch")
    manifest, file_entries, total, conv, failed, n_quiet = [], [], 0, 0, 0, 0
    _quiet_ids = set()           # picks skipped at export as near-silent (`fourier why`: too quiet)
    dsp_fallback = set()
    _prev_names = _previous_outnames(category)
    for c, want in zip(clusters, _allocs):
        dest = kroot / c["name"]; dest.mkdir(parents=True, exist_ok=True)
        cop = 0
        # files the previous build named claim their names first, so a newcomer with the same
        # stem gets the "_2" and a path never starts holding a different sample
        # (without Sononym, then a file already at C ahead of one retuned to C: the plain
        # name goes to the one that always played it)
        for i in sorted(c["_pick"], key=lambda i: (rec[i]["path"] not in _prev_names,
                                                   _fb and abs(rec[i].get("retune") or 0.0) >= 0.5)):
            src = rec[i]["path"]
            try:
                outp = export_one(src, dest, pinned=rec[i].get("pin") is True, retune=rec[i].get("retune"),
                                  root_src=rec[i].get("root_src"), prev_name=_prev_names.get(src),
                                  root_midi=rec[i].get("root_midi") if rec[i].get("root_midi") is not None
                                  else rec[i].get("tune"),
                                  bpm=rec[i].get("bpm") if cfg["kind"] == "loop" else None)
                if outp is None:
                    n_quiet += 1
                    _quiet_ids.add(rec[i]["id"])
                    continue
                cop += 1
                if not src.lower().endswith(".wav"):
                    conv += 1
                vt = votes.get(rec[i]["id"], {})
                file_entries.append(dict(
                    family=c["name"], out=str(outp.relative_to(kroot)), src=src,
                    out_md5=_md5(outp), support=rec[i]["sup"], band=rec[i].get("band"),
                    **({"loop_row": True} if rec[i].get("loop_row") else {}),
                    # a phrase-marked loop CLAP heard as a drum loop (without Sononym: _clap_rehome)
                    **({"rehomed": True} if vt.get("rehomed") == category else {}),
                    sononym=vt.get("son_labels", []), ableton=vt.get("ab_tags") or rec[i].get("ab", []),
                    son_cats=vt.get("son_cats", []), ab_cats=vt.get("ab_cats", []),
                    labels=rec[i].get("labels", []),       # canonical (manifest v2)
                    **({"bpm": rec[i]["bpm"], "bpm_src": rec[i].get("bpm_src"),
                        "bpm_fold": rec[i].get("bpm_fold")}
                       if cfg["kind"] == "loop" else {}),
                    # bars, slices and the grid at the folder's tempo: a break an analyzer read at
                    # half time is counted at the doubled tempo it is filed at
                    **({"bpm_bars": _loop_tempo(rec[i])} if category == "DRUMLOOPS" and _loop_tempo(rec[i]) else {}),
                    **(_rotate_and_slice(outp, _loop_tempo(rec[i]), src) if category == "DRUMLOOPS" and loudness
                       else _slice_fields(outp, _loop_tempo(rec[i])) if category == "DRUMLOOPS" else {}),
                    **({"dsp_fallback": True} if str(outp) in dsp_fallback else {}),
                    **({"retune": rec[i]["retune"], "root_src": rec[i]["root_src"]}
                       if rec[i].get("retune") is not None else {}),
                    **({"root_midi": round(float(rec[i].get("root_midi") if rec[i].get("root_midi") is not None
                                                 else rec[i]["tune"]), 2)}
                       if rec[i].get("retune") is not None and rec[i].get("root_src") in ("pyin", "detect", "chunk")
                       and (rec[i].get("root_midi") is not None or rec[i].get("tune") is not None) else {}),
                    **({"pitch_conflict": rec[i]["pitch_conflict"]} if rec[i].get("pitch_conflict") else {}),
                    # a mirror-folder file here has no twin elsewhere (its twins were left out)
                    **({"mirror_only": True} if MIRROR_ROOT_RE.search(library_rel(src)) else {}),
                ))
                if file_entries[-1].get("rotate_ms"):
                    file_entries[-1]["out_md5"] = _md5(outp)      # rotated after its hash
            except Exception:
                failed += 1
        if category != WAVES_CATEGORY:
            _fix_phase(kroot, [e for e in file_entries if e["family"] == c["name"]], kind=cfg["kind"],
                       category=category)
        if category in LEVEL_CATS and loudness:
            _level_folder(kroot, [e for e in file_entries if e["family"] == c["name"]],
                          rms_ceil_db=ONESHOT_RMS_CEIL_DB.get(category))
        elif category in DRUM_LEVEL_CATS and loudness:
            # a drum folder's loudest hits come down to its median + DRUM_LEVEL_OVER_DB, so
            # browsing it doesn't jump in level
            _level_folder(kroot, [e for e in file_entries if e["family"] == c["name"]],
                          tol_db=DRUM_LEVEL_OVER_DB, up=False)
        total += cop
        manifest.append(dict(
            family=c["name"], desc=descriptions.get(c["name"], ""), source=c["source"],
            share=round(c["share"], 2), clap=c["clap_phrase"], clap_z=c.get("clap_z"),
            clap_support=c.get("clap_support"),
            band=c.get("band"),
            traits=c["traits"][:4],
            bpm=(round(c["bpm"]) if c["bpm"] else None), bpm_range=c.get("bpm_range"),
            base_note=(round(c["tune"]) if c["tune"] is not None else None),
            n=len(c["idxs"]), copied=cop,
        ))
        log(f"  {c['name']:<30} src:{c['source'] or 'mixed':<8} {int(c['share'] * 100):2d}%  "
            f"clap:{c['clap_phrase'] or '(feature-only)':<24} {', '.join(c['traits'][:4])}")

    (kroot / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = dict(category=category, families=len(clusters), from_clusters=k0, files=total,
                   transcoded=conv, failed=failed, source_samples=n,
                   twins_collapsed=stats["twins"], near_dups_pruned=stats.get("near_dup", 0),
                   note_gated=stats.get("note_gated", 0), loop_gated=stats.get("loop_gated", 0),
                   chain_gated=stats.get("chain_gated", 0), mirror_copies=stats.get("mirror_copies", 0),
                   style_excluded=stats.get("style_excluded", 0), soft_layers=stats.get("soft_layers", 0),
                   out=str(kroot), seconds=round(_time.perf_counter() - _t0, 1),
                   cache_hits=_cache_hits[0])
    if _n_cloud:
        summary["cloud_downloaded"] = _n_cloud
    if _f is not None:          # what verify holds a scaled category to (its budget, folders)
        summary["budget"] = int(budget)
        summary["scale"] = _scale_doc(scale)
    log(f"DONE {category}: {len(clusters)} families (from {k0}), {total} files "
        f"({conv} transcoded, {failed} failed, {n_quiet} near-silent skipped), {n} sources, {stats['twins']} twins collapsed, "
        f"{stats.get('near_dup', 0)} near-dups pruned, {stats.get('loop_gated', 0)} loops dropped")
    # why each candidate that isn't in the master was left out, beside the build (not in it)
    _picked = {e["src"] for e in file_entries}
    _why["over_budget"] = {r["id"]: None for r in rec if r["path"] not in _picked and r["id"] not in _quiet_ids}
    if _quiet_ids:
        _why["too_quiet"] = {i: QUIET_RMS_DB for i in sorted(_quiet_ids)}
    # the source of each picked file's tempo and root (metadata/resolve.py), for `fourier why`
    _fb = _fallback(session)
    _src = {str(r["id"]): got for r in rec if r["path"] in _picked
            for got in [_resolve.recorded(r, _fb)] if got}
    why_log.write(out_dir, category, left_out_doc(category, "built", n, floor, stats, _why,
                                                  files=total, budget=budget, **_gate_doc,
                                                  **_clap_doc(_clap_ids, _rehomed, _sound),
                                                  **({"sources": _src} if _src else {})))
    if return_entries:
        return summary, file_entries
    try:
        update_build_manifest(out_dir, category, summary, file_entries)
    except Exception as e:
        log(f"  (manifest update skipped: {e})")
    return summary


# ---------------------------------------------------------------------------
# Parallel --all build: one process per category, homes computed once, manifest
# merged in the parent (avoids a shared-manifest write race). Per-category output
# is byte-identical to a serial build since each category is independently seeded.
# ---------------------------------------------------------------------------
_WORKER: dict = {}


def _worker_init(db_path, homes, support, votes, opts):
    from ..db.session import init_db
    init_db(db_path)
    _WORKER.update(homes=homes, support=support, votes=votes, opts=opts)


def _worker_build(category, out_dir, budget=None):
    from ..db.session import session_scope
    from ..safety import UnsafePath
    try:
        with session_scope() as session:
            summary, entries = build_taxonomy(
                session, category, out_dir,
                homes=_WORKER["homes"], support=_WORKER["support"], votes=_WORKER["votes"],
                rebuild_index=False, return_entries=True, log=lambda m: None,
                budget_override=budget, **_WORKER["opts"])
        if budget is not None:
            summary["budget"] = int(budget)
        return (category, summary, entries, None)
    except (UnsafePath, TooFewSamples) as e:     # stops the build / leaves the category out
        return (category, None, None, e)
    except Exception as e:
        return (category, None, None, str(e))


def _stop_on_unsafe(ex, err):
    """A worker's UnsafePath (cloud-only sources that won't download) stops the whole build:
    the categories not started yet are cancelled and the error is raised."""
    from ..safety import UnsafePath
    if isinstance(err, UnsafePath):
        ex.shutdown(wait=False, cancel_futures=True)
        raise err


def surplus_budgets(results, budgets=None) -> dict:
    """{category: raised budget} for one surplus round (POOL_SURPLUS): the files the
    categories that fell short of their budget didn't fill, shared among the categories
    that filled theirs in proportion to their budgets (largest remainders). {} when no
    category fell short or none filled its budget."""
    budgets = BUDGETS if budgets is None else budgets
    got = {c: s["files"] for c, s, _e in results if s is not None and c in budgets}
    short = sum(budgets[c] - n for c, n in got.items() if n < budgets[c])
    full = {c: budgets[c] for c, n in got.items() if n >= budgets[c]}
    if short <= 0 or not full:
        return {}
    total = sum(full.values())
    raw = {c: b * short / total for c, b in full.items()}
    add = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: (raw[c] - add[c], c), reverse=True)[:short - sum(add.values())]:
        add[c] += 1
    return {c: full[c] + n for c, n in add.items() if n}


def build_all_parallel(db_path, categories, out_dir, jobs, opts, log=print, progress=None, done=None,
                       empty=None):
    """Build several categories concurrently. Homes are computed once here (and a
    stale index refreshed once) then shared to workers via the pool initializer, so
    workers neither recompute homes nor race on the index file. Manifests are merged
    in the parent. A category too few samples could fill (TooFewSamples) is left out of
    the results and, when given the list empty, named in it."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from ..db.session import session_scope

    with session_scope() as session:
        stale, idx_n, db_n = index_stale(session)
        if stale:
            from ..analysis.clap_features import build_index
            log(f"CLAP index stale ({idx_n} vs {db_n}); rebuilding once before fan-out...")
            build_index(session)
        ids, emb = load_index()
        emb = emb.astype("float32")
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        id2row = {int(i): r for r, i in enumerate(ids)}
        homes, support, votes = compute_homes(session, emb_n, id2row, log)
    del emb, emb_n  # free ~0.5 GB in the parent before forking workers

    # a resumed build (packs/progress.py) keeps the categories it already finished
    results = [(c, *done[c]) for c in categories if done and c in done]
    with ProcessPoolExecutor(max_workers=jobs,
                             initializer=_worker_init,
                             initargs=(db_path, homes, support, votes, opts)) as ex:
        # biggest budgets first, so a long category doesn't start last (results are sorted below)
        order = sorted((c for c in categories if not (done and c in done)),
                       key=lambda c: (-BUDGETS.get(c, 0), c))
        futs = {ex.submit(_worker_build, c, out_dir): c for c in order}
        for fut in as_completed(futs):
            category, summary, entries, err = fut.result()
            _stop_on_unsafe(ex, err)
            if isinstance(err, TooFewSamples):
                log(f"[{category}] left empty: {str(err).removeprefix(category + ': ')}")
                if empty is not None:
                    empty.append(category)
            elif err:
                log(f"[{category}] FAILED: {err}")
            else:
                log(f"[{category}] {summary['families']} families, {summary['files']} files"
                    + (f" (budget {summary['budget']})" if summary.get("scale") else "")
                    + f", {summary.get('near_dups_pruned', 0)} near-dups pruned"
                    + (f", {summary['note_gated']} out-of-range notes dropped"
                       if summary.get('note_gated') else "")
                    + (f", {summary['loop_gated']} mislabeled loops dropped"
                       if summary.get('loop_gated') else "")
                    + (f", {summary['chain_gated']} sample chains dropped"
                       if summary.get('chain_gated') else "")
                    + "".join(f", {summary[k]} {lab}" for k, lab in (
                        ("mirror_copies", "mirror copies"), ("style_excluded", "off-style loops"),
                        ("soft_layers", "soft layers"),
                        ("cloud_downloaded", "cloud-only sources downloaded first")) if summary.get(k)))
                results.append((category, summary, entries))
                if progress is not None:
                    progress.done(category, summary, entries)
        # (no surplus round in a library-scaled build: its budgets already follow the pools)
        raised = surplus_budgets(results) if POOL_SURPLUS and not opts.get("scale") else {}
        if raised:
            log(f"surplus: {sum(BUDGETS[c] - s['files'] for c, s, _e in results if s and s['files'] < BUDGETS.get(c, 0))} "
                f"files the short categories couldn't fill go to {len(raised)} others; rebuilding them")
            again = {c: ex.submit(_worker_build, c, out_dir, b) for c, b in sorted(raised.items())}
            redo = {}
            for c, fut in again.items():
                category, summary, entries, err = fut.result()
                _stop_on_unsafe(ex, err)
                if err:
                    log(f"[{category}] surplus rebuild FAILED: {err}")
                    results = [r for r in results if r[0] != category]
                else:
                    log(f"[{category}] {summary['files']} files (budget {summary['budget']})")
                    redo[category] = (category, summary, entries)
            results = [redo.get(r[0], r) for r in results]
    results.sort(key=lambda r: r[0])   # stable order -> deterministic manifest
    n = _dedupe_across(out_dir, results)
    if n:
        log(f"{n} file(s) chosen by two categories kept in one (_dup_winner)")
    merge_manifest(out_dir, results)
    return results


def _dup_winner(src, cats, pins=None):
    """The one category a source file chosen by several categories stays in: keys-named
    -> PIANO, stab / chord-named -> STABS, an acoustic instrument -> ACOUSTIC, else the
    first in CATEGORY_ORDER (else a file can be exported twice, typically a chord both
    PIANO and STABS want)."""
    from .curate_config import CATEGORY_ORDER
    if pins and pins.get(src) in cats:
        return pins[src]                  # a Keep stays where it was rated
    fn = os.path.basename(src or "")
    rel = library_rel(src or "")
    for c, ok in (("PIANO", bool(KEYBOARD_NAME_RE.search(fn))),
                  ("STABS", _stab_named(fn)),
                  ("ACOUSTIC", bool(ORCH_PACKS.search(_pack_of(rel, src)) or _acoustic_reserved(fn, (), rel)))):
        if ok and c in cats:
            return c
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return min(cats, key=lambda c: (order.get(c, 99), c))


def _dedupe_across(out_dir, results):
    """Keep a source file in one category only (_dup_winner): drop the other exported
    copies, their manifest entries and their families' counts. Returns files dropped."""
    from .ratings import keep_pins
    pins = keep_pins()
    where = defaultdict(list)
    for category, summary, entries in results:
        for e in entries or []:
            where[e["src"]].append(category)
    drop = {}
    for src, cats in where.items():
        if len(set(cats)) > 1:
            win = _dup_winner(src, sorted(set(cats)), pins)
            for c in set(cats) - {win}:
                drop.setdefault(c, set()).add(src)
    n = 0
    for category, summary, entries in results:
        gone = drop.get(category)
        if not gone:
            continue
        keep, lost = [], Counter()
        for e in entries:
            if e["src"] in gone:
                try:
                    (Path(out_dir) / category / e["out"]).unlink()
                except FileNotFoundError:
                    pass
                lost[e["family"]] += 1
                n += 1
            else:
                keep.append(e)
        entries[:] = keep
        summary["files"] = len(keep)
        fp = Path(out_dir) / category / "_manifest.json"
        try:
            fams = json.loads(fp.read_text())
            for f in fams:
                if f.get("family") in lost:
                    f["copied"] = f.get("copied", 0) - lost[f["family"]]
            fp.write_text(json.dumps(fams, indent=2))
        except (OSError, ValueError):
            pass
    return n


def _describe(clusters, category, model):
    prof = "\n".join(
        "%s: source=%s(%d%%), clap=%s, traits=%s" % (
            c["name"], c["source"] or "mixed", int(c["share"] * 100),
            c["clap_phrase"] or "-", ", ".join(c["traits"][:4]))
        for c in clusters)
    prompt = (f"Write a 6-10 word description for each {category} folder, using ONLY its "
              f"clap phrase and traits. Invent nothing.\n{prof}\n\n"
              'Reply ONLY JSON: {"d": {"folder": "desc", ...}}')
    # cached by (model, prompt): an unchanged category costs no LLM call (a local model call
    # per category is slow), and a seed keeps the text the same across runs
    key = hashlib.sha256(f"{model}\n{prompt}".encode()).hexdigest()
    cpath = _describe_cache_dir() / f"{key}.json"
    try:
        return json.loads(cpath.read_text())
    except (OSError, ValueError):
        pass
    try:
        payload = {"model": model, "prompt": prompt, "stream": False, "think": False,
                   "format": "json", "options": {"temperature": 0.2, "seed": CURATION_SEED}}
        req = urllib.request.Request(
            "http://localhost:11434/api/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        from ..net import local_opener                  # localhost: never through a proxy
        raw = json.loads(local_opener().open(req, timeout=300).read()).get("response", "{}")
        out = json.loads(raw).get("d", {})
    except Exception:
        return {}
    if out:
        try:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = cpath.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(out))
            os.replace(tmp, cpath)
        except OSError:
            pass
    return out


def _pitch_cache_dir():
    from .ratings import default_store
    return Path(os.environ.get("FOURIER_PITCH_CACHE")
                or os.path.join(os.path.dirname(default_store()), "cache", "pitch"))


def _detect_pitch(path):
    """MIDI pitch (float) of a tonal one-shot's steady part by pYIN, or None when it isn't
    clearly pitched (PITCH_VOICED_MIN of the frames voiced, spread under PITCH_SPREAD_MAX
    semitones). Cached by path, size and mtime under ~/.fourier/cache/pitch."""
    import hashlib
    import json as _json
    import soundfile as sf
    from .curate_config import PITCH_SPREAD_MAX, PITCH_VOICED_MIN
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = hashlib.md5(f"{path}|{st.st_size}|{st.st_mtime_ns}|pyin1".encode()).hexdigest()
    cf = _pitch_cache_dir() / key[:2] / f"{key}.json"
    try:
        return _json.loads(cf.read_text())["midi"]
    except Exception:
        pass
    midi = None
    try:
        import librosa
        info = sf.info(path)
        y, sr = sf.read(path, frames=int(info.samplerate * 3), always_2d=True)
        y = y.mean(axis=1)
        if sr != 22050:
            y = librosa.resample(y, orig_sr=sr, target_sr=22050)
            sr = 22050
        a = np.abs(y)
        if a.size and a.max() > 0:
            y = y[int(np.argmax(a > 0.1 * a.max())) + int(0.05 * sr):]      # past the attack
        if len(y) >= int(0.15 * sr):
            f0, v, p = librosa.pyin(y, fmin=30, fmax=2000, sr=sr, frame_length=2048)
            ok = v & (p > 0.5)
            if ok.mean() >= PITCH_VOICED_MIN:
                mid = librosa.hz_to_midi(f0[ok])
                q75, q25 = np.percentile(mid, [75, 25])
                if q75 - q25 <= PITCH_SPREAD_MAX:
                    midi = round(float(np.median(mid)), 2)
    except Exception:
        midi = None
    try:
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(_json.dumps({"midi": midi}))
    except OSError:
        pass
    return midi


def _pitch_conflicts(name_pc, midi):
    """True when a clearly detected pitch contradicts a filename's note: 2 or more semitones
    apart, not a fourth / fifth (a harmonic the detector can lock onto), octaves ignored."""
    if name_pc is None or midi is None:
        return False
    d = (int(round(midi)) - name_pc) % 12
    d = min(d, 12 - d)
    return d >= 2 and d != 5


def _check_name_pitches(idxs, rec):
    """A picked file retuned by its filename's note keeps its pitch and name when the audio
    clearly plays another note (a patch name can look like a note, and a pack can be
    mislabeled). Records
    `pitch_conflict` (the detected MIDI pitch). Returns how many."""
    if os.environ.get("FOURIER_NO_PITCH_CHECK"):
        return 0
    n = 0
    for i in idxs:
        r = rec[i]
        if r.get("root_src") != "name" or r.get("retune") is None or not r.get("path"):
            continue
        if _name_root_midi(os.path.basename(r["path"])) is None:
            continue                  # a chord or key name ("Am", "Dmin"): pYIN reads one of its notes
        midi = _detect_pitch(r["path"])
        if _pitch_conflicts(r.get("name_pc"), midi):
            r["retune"], r["root_src"], r["pitch_conflict"] = None, None, midi
            n += 1
    return n


def _fill_roots(category, idxs, rec):
    """A picked tonal one-shot with no root yet (no note in its name, nothing Sononym was
    sure of) is retuned by a clear pYIN pitch, within the category's cap. Returns how
    many."""
    if os.environ.get("FOURIER_NO_PITCH_CHECK") or category not in RETUNE_CATS:
        return 0
    n = 0
    for i in idxs:
        r = rec[i]
        if r.get("retune") is not None or r.get("pitch_conflict") or r.get("name_pc") is not None \
                or r.get("loop_row") or not r.get("path"):
            continue
        name = os.path.basename(r["path"])
        if _name_multi_note(name):
            continue
        midi = _detect_pitch(r["path"])
        if midi is None:
            continue
        rt, _src = _retune_shift(category, name, midi, detect=True)
        if rt is not None:
            r["retune"], r["root_src"], r["root_midi"] = rt, "pyin", midi
            n += 1
    return n


def _describe_cache_dir():
    from .ratings import default_store
    return Path(os.environ.get("FOURIER_DESCRIBE_CACHE")
                or os.path.join(os.path.dirname(default_store()), "cache", "describe"))
