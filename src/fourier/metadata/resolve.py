"""What the build uses for each sample, and where each value came from.

Every stored value has one source (db/models.py, docs/curation.md "Sources and resolution").
This module chooses among them the way curation always has, and names the source of what it
chose, so `fourier why` and the why log can say it. Curation calls it on the rows it reads
through metadata/rows.py (`sample_select`), never on a provider's tables.

Sources:

  sononym          Sononym's analysis (sononym_meta; provider "sononym"), and the legacy
                   sample_features columns computed from it (bpm_reliable, sub_weight, ...)
  fourier:audio    Fourier's own analysis of the audio (sample_features: librosa's tempo,
                   onsets and harmonic share, the events, the pYIN root, the key; the audio
                   provider's one-shot / loop)
  fourier:name     Fourier's reading of the file's name and folders (a tempo written in it;
                   the path provider's labels)
  fourier:folder   a folder named only by a tempo ("Loops/174/")
  fourier:length   the whole-bar tempo a loop's length implies near a measured one
  acid             the WAV's ACID chunk (its tempo)
  chunk            the WAV's own root note (its ACID root, else its smpl chunk's unity note)

The build's choices (with Sononym: exactly what curation chose before this module existed;
without it: the fallback chain, unchanged):

  tempo, a loop's  curation's chain (packs/curate._resolve_tempo): a tempo the filename states;
                   else the classifier's tempo (Sononym's; without Sononym the tempo written
                   in the path), else Fourier's own, each tried at x1, x2 and x0.5 and taken
                   when the loop is then whole bars; without Sononym then the WAV's ACID
                   tempo, the name's other tempo forms, a tempo folder and the loop's length
  tempo_reliable   Sononym's flag (bpm_confidence over 0.6); never set without Sononym
  tempo, a one-shot's  Fourier's own tempo, where Sononym's tempo is reliable
  shape            Sononym's class; without Sononym the path provider's (the file's own name
                   or folder), else the audio provider's
  root             Sononym's base note when its confidence is over 0.4 (MIDI 12-84); without
                   Sononym the WAV's own root note (12-108). Curation's retune then takes a
                   note in the name, or pYIN at build time, and records which (root_src)

Fourier's own readings (own_*) are computed whether or not Sononym is there, and reported
(`fourier why`, `fourier tools db-stats --disagreements`), not used to pick: a build with
Sononym picks as it always has.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SONONYM = "sononym"
AUDIO = "fourier:audio"
NAME = "fourier:name"
FOLDER = "fourier:folder"
LENGTH = "fourier:length"
ACID = "acid"
CHUNK = "chunk"
SOURCES = (SONONYM, AUDIO, NAME, FOLDER, LENGTH, ACID, CHUNK)

# how `fourier why` names a source
SAID_BY = {SONONYM: "Sononym", AUDIO: "Fourier's own analysis", NAME: "its name",
           FOLDER: "its folder's name", LENGTH: "its length", ACID: "the WAV's ACID chunk",
           CHUNK: "the WAV's root note"}

# curation's tempo chain names its steps as the manifest's bpm_src does; their sources
_CHAIN = {"name": NAME, "librosa": AUDIO, "acid": ACID, "folder": FOLDER, "length": LENGTH}
# curation's retune names where a root came from (root_src); their sources
_ROOT_SRC = {"name": NAME, "pyin": AUDIO, "chunk": CHUNK}

# the build's root note: Sononym's base note above this confidence, in this MIDI range; the
# WAV's own root note (without Sononym) in the wider range
SONONYM_ROOT_CONF = 0.4
SONONYM_ROOT_RANGE = (12, 84)
CHUNK_ROOT_RANGE = (12, 108)

# Fourier's own readings. Pitched: pYIN found one clear pitch (when it ran), else mostly
# harmonic (HPSS share) and focused on one pitch class (chroma concentration)
OWN_PITCHED_HPR, OWN_PITCHED_CHROMA = 0.6, 2.0
# Own tempo confidence: 1 when the loop is exactly whole 4/4 bars at the tempo (or its double
# or half), falling to 0 half a beat off; reliable from OWN_TEMPO_RELIABLE, for a loop
OWN_TEMPO_BARS = (1, 2, 4, 8, 16)
OWN_TEMPO_OFF_BEATS = 0.5
OWN_TEMPO_RELIABLE = 0.7
# two tempos agree when one is within this share of the other at x1, x2 or x0.5
TEMPO_AGREE = 0.03
# Sononym's pitched flag, as the derived step computes it (analysis/derived.py)
SONONYM_PITCHED_HAR, SONONYM_PITCHED_CONF, SONONYM_PITCHED_HAR_ALONE = 0.6, 0.5, 0.7

# which samples Fourier's own pYIN root is read for (`fourier tools analyze --only own`):
# one event (or not counted), partly harmonic with some pitch focus (no noise hits), a
# one-shot's length; f is sample_features, s samples. pYIN takes about half a second a file
ROOT_CANDIDATE_SQL = ("f.harmonic_percussive_ratio >= 0.4 AND f.chroma_concentration >= 1.5 "
                      "AND s.duration_s >= 0.15 AND s.duration_s <= 10 "
                      "AND (f.n_events IS NULL OR f.n_events <= 1)")
# Fourier's own tonal call (never Sononym's): mostly harmonic (HPSS share) with a pitch focus
# (chroma concentration; a chord reads about 1.5 to 2.5, noise under 1.3); the key step reads
# the samples it calls tonal, or with a pYIN root
OWN_TONAL_HPR, OWN_TONAL_CHROMA = 0.6, 1.5
KEY_CANDIDATE_SQL = (f"(f.own_root_midi IS NOT NULL OR (f.harmonic_percussive_ratio >= {OWN_TONAL_HPR} "
                     f"AND f.chroma_concentration >= {OWN_TONAL_CHROMA}))")


@dataclass(frozen=True)
class Reading:
    """A value and the source it came from (None, None: no value)."""
    value: Any = None
    source: str | None = None


def measured_by(fallback: bool) -> str:
    """Whose measurements the classifier's descriptor fields (brightness, harmonicity, bpm,
    ...) hold: Sononym's, else Fourier's own (rows.field_source has each field's)."""
    return AUDIO if fallback else SONONYM


# ---------------------------------------------------------------------------
# What the build uses
# ---------------------------------------------------------------------------

def tempo_source(chain_src: str | None, fallback: bool) -> str | None:
    """The source of curation's tempo chain's step (the manifest's bpm_src): its "sononym"
    step reads the classifier's tempo, which without Sononym is the one written in the path."""
    if chain_src is None:
        return None
    if chain_src == "sononym":
        return NAME if fallback else SONONYM
    return _CHAIN.get(chain_src, chain_src)


def loop_tempo(r, fallback: bool) -> tuple[Reading, str | None]:
    """(the tempo a loop is filed at, as Reading, and the chain's own step name, which the
    manifest records as bpm_src): curation's chain, unchanged."""
    from ..packs import curate as C
    bpm, src = C._resolve_tempo(getattr(r, "filename", None), getattr(r, "duration_s", None),
                                getattr(r, "bpm", None), getattr(r, "tempo_bpm", None),
                                C._fallback_tempos(r) if fallback else None)
    return Reading(bpm, tempo_source(src, fallback) if bpm is not None else None), src


def tempo_reliable(r) -> Reading:
    """Sononym's flag that its tempo is reliable (the legacy bpm_reliable column)."""
    ok = bool(getattr(r, "bpm_reliable", None))
    return Reading(ok, SONONYM if ok else None)


def oneshot_tempo(r) -> Reading:
    """A one-shot's tempo: Fourier's own, only where Sononym's tempo is reliable."""
    t = getattr(r, "tempo_bpm", None)
    if tempo_reliable(r).value and t:
        return Reading(t, AUDIO)
    return Reading()


def root(r, fallback: bool) -> Reading:
    """The root note curation retunes from before its name and pYIN steps: Sononym's base
    note when sure enough, else (without Sononym) the WAV's own root note."""
    bn, conf = getattr(r, "base_note", None), getattr(r, "base_note_confidence", None)
    if (bn is not None and conf is not None and conf > SONONYM_ROOT_CONF
            and SONONYM_ROOT_RANGE[0] <= bn <= SONONYM_ROOT_RANGE[1]):
        return Reading(bn, measured_by(fallback))
    rn = getattr(r, "root_note", None)
    if fallback and rn is not None and CHUNK_ROOT_RANGE[0] <= rn <= CHUNK_ROOT_RANGE[1]:
        return Reading(float(rn), CHUNK)
    return Reading()


def shape(r, fallback: bool) -> Reading:
    """One-shot or loop ("oneshot", "loop", None) as the classifier calls it."""
    classes = getattr(r, "classes", None) or []
    got = "loop" if "Loop" in classes else "oneshot" if "OneShot" in classes else None
    if got is None:
        return Reading()
    if not fallback:
        return Reading(got, SONONYM)
    from .shadow import shape_labels
    named = shape_labels(getattr(r, "rel_path", None) or getattr(r, "path", None) or "")
    return Reading(got, NAME if named else AUDIO)


def recorded(rec: dict, fallback: bool) -> dict:
    """What the why log keeps for a picked file (curation's record of it): the source of its
    tempo and its root, when it has them."""
    out = {}
    if rec.get("bpm") is not None:
        out["tempo"] = (tempo_source(rec.get("bpm_src"), fallback) if rec.get("bpm_src")
                        else AUDIO)          # a one-shot's tempo (oneshot_tempo)
    rs = rec.get("root_src")
    if rs is not None and rec.get("retune") is not None:
        out["root"] = measured_by(fallback) if rs == "detect" else _ROOT_SRC.get(rs, rs)
    return out


# ---------------------------------------------------------------------------
# Fourier's own readings, and Sononym's beside them
# ---------------------------------------------------------------------------

def own_shape(rel_path, duration_s, n_events, onset_rate_hz=None, hpr=None) -> str | None:
    """Fourier's own one-shot / loop: the audio provider's (metadata/shadow.audio_label)."""
    from .shadow import audio_label
    lab = audio_label(rel_path or "", duration_s, n_events, onset_rate_hz, hpr)
    return {"class.loop": "loop", "class.oneshot": "oneshot"}.get(lab)  # type: ignore[arg-type]


def own_tempo_confidence(duration_s, tempo_bpm) -> float | None:
    """How well Fourier's own tempo makes the file whole 4/4 bars (OWN_TEMPO_BARS), at the
    tempo or its double or half: 1.0 exactly, 0 at OWN_TEMPO_OFF_BEATS off. None without both."""
    if not duration_s or not tempo_bpm or duration_s <= 0 or tempo_bpm <= 0:
        return None
    off = min(abs(duration_s * tempo_bpm * mul / 60.0 - 4 * bars)
              for mul in (1, 2, 0.5) for bars in OWN_TEMPO_BARS)
    return round(max(0.0, 1.0 - off / OWN_TEMPO_OFF_BEATS), 3)


def own_pitched(hpr, chroma, root_midi=None, root_at=None) -> bool | None:
    """Fourier's own pitched call: pYIN found one clear pitch, when it ran; else mostly
    harmonic and focused on one pitch class. None without the measurements."""
    if root_at is not None:
        return root_midi is not None
    if hpr is None or chroma is None:
        return None
    return hpr >= OWN_PITCHED_HPR and chroma >= OWN_PITCHED_CHROMA


def own_tonal(hpr, chroma) -> bool:
    """Fourier's own tonal call (OWN_TONAL_HPR, OWN_TONAL_CHROMA): what the key step reads."""
    return hpr is not None and chroma is not None and hpr >= OWN_TONAL_HPR and chroma >= OWN_TONAL_CHROMA


def sononym_pitched(harmonicity, pitch_confidence) -> bool | None:
    """Sononym's pitched call (the derived step's is_pitched rule)."""
    if harmonicity is None:
        return None
    if pitch_confidence is None:
        return harmonicity > SONONYM_PITCHED_HAR_ALONE
    return harmonicity > SONONYM_PITCHED_HAR and pitch_confidence > SONONYM_PITCHED_CONF


def tempos_agree(a, b, tol: float = TEMPO_AGREE) -> bool:
    """Two tempos agree when one is within tol of the other, at x1, x2 or x0.5."""
    return any(abs(a * m - b) <= tol * b for m in (1, 2, 0.5))


# the sample fields readings() reads (metadata/rows.py)
READING_FIELDS = ("id", "rel_path", "path", "duration_s", "sononym_classes", "sononym_bpm",
                  "sononym_harmonicity", "sononym_pitch_confidence", "sononym_base_note",
                  "sononym_base_note_confidence", "tempo_bpm", "n_events", "onset_rate_hz",
                  "harmonic_percussive_ratio", "chroma_concentration", "own_root_midi",
                  "own_root_at", "detected_key", "key_confidence")


@dataclass
class Readings:
    """One sample's readings by source: {what: value} for Sononym and for Fourier's own
    analysis (None: that source has no reading). what: shape, tempo, pitched, root (MIDI),
    key; own also tempo_confidence."""
    sample_id: int
    rel_path: str
    sononym: dict = field(default_factory=dict)
    own: dict = field(default_factory=dict)


def readings_of(r) -> Readings:
    """Both readings of a row with READING_FIELDS."""
    from ..places import library_rel
    rel = getattr(r, "rel_path", None) or library_rel(getattr(r, "path", None) or "")
    sc = getattr(r, "sononym_classes", None)
    son_shape = None if not sc else "loop" if "Loop" in sc else "oneshot" if "OneShot" in sc else None
    bn, bnc = r.sononym_base_note, r.sononym_base_note_confidence
    son = {"shape": son_shape, "tempo": r.sononym_bpm,
           "pitched": sononym_pitched(r.sononym_harmonicity, r.sononym_pitch_confidence),
           "root": bn if (bn is not None and bnc is not None and bnc > SONONYM_ROOT_CONF) else None,
           "key": None}
    own = {"shape": own_shape(rel, r.duration_s, r.n_events, r.onset_rate_hz, r.harmonic_percussive_ratio),
           "tempo": r.tempo_bpm, "tempo_confidence": own_tempo_confidence(r.duration_s, r.tempo_bpm),
           "pitched": own_pitched(r.harmonic_percussive_ratio, r.chroma_concentration,
                                  r.own_root_midi, r.own_root_at),
           "root": r.own_root_midi, "key": r.detected_key}
    return Readings(r.id, rel, son, own)


_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _note(midi) -> str:
    m = int(round(float(midi)))
    return f"{_NOTES[m % 12]}{m // 12 - 1}"


def _text(what, v) -> str:
    if v is None:
        return "-"
    if what == "tempo":
        return f"{float(v):g}"
    if what == "shape":
        return "one-shot" if v == "oneshot" else str(v)
    if what == "pitched":
        return "pitched" if v else "unpitched"
    if what == "root":
        return _note(v)
    return str(v)


def differences(rd: Readings) -> list[tuple[str, str]]:
    """[(what, "174 (Sononym) / 87 (Fourier's own analysis)")] where both sources have a
    reading and they disagree: shape; tempo beyond an octave (only for a file either calls a
    loop); pitched; root by a semitone or more, octaves ignored."""
    s, o = rd.sononym, rd.own
    out = []

    def add(what):
        out.append((what, f"{_text(what, s[what])} ({SAID_BY[SONONYM]}) / "
                          f"{_text(what, o[what])} ({SAID_BY[AUDIO]})"))
    if s["shape"] and o["shape"] and s["shape"] != o["shape"]:
        add("shape")
    if (s["tempo"] and o["tempo"] and "loop" in (s["shape"], o["shape"])
            and not tempos_agree(float(o["tempo"]), float(s["tempo"]))):
        add("tempo")
    if s["pitched"] is not None and o["pitched"] is not None and s["pitched"] != o["pitched"]:
        add("pitched")
    if s["root"] is not None and o["root"] is not None:
        d = abs(float(s["root"]) - float(o["root"])) % 12
        if min(d, 12 - d) >= 1.0:
            add("root")
    return out


def readings(session, ids=None) -> list[Readings]:
    """Both readings of these samples (None: every sample the classifier looked at), in id
    order."""
    from sqlalchemy import text as _text_sql

    from ..db.models import Sample
    from .rows import fetch, sample_select
    q = sample_select(*READING_FIELDS, session=session)
    if ids is not None:
        ids = sorted({int(i) for i in ids})
        if not ids:
            return []
        q = q.where(Sample.id.in_(ids))
    return [readings_of(r) for r in fetch(session, q.order_by(_text_sql("samples.id")))]


DISAGREEMENT_KINDS = ("shape", "tempo", "pitched", "root")


def disagreements(session) -> dict:
    """Where Sononym and Fourier's own analysis disagree, over every sample the classifier
    looked at: {"samples": n, "compared": {what: n}, "found": {what: [(rel_path, text)]}}.
    Reporting only: nothing the build reads changes."""
    compared = dict.fromkeys(DISAGREEMENT_KINDS, 0)
    found: dict = {k: [] for k in DISAGREEMENT_KINDS}
    rows = readings(session)
    for rd in rows:
        s, o = rd.sononym, rd.own
        compared["shape"] += bool(s["shape"] and o["shape"])
        compared["tempo"] += bool(s["tempo"] and o["tempo"] and "loop" in (s["shape"], o["shape"]))
        compared["pitched"] += s["pitched"] is not None and o["pitched"] is not None
        compared["root"] += s["root"] is not None and o["root"] is not None
        for what, txt in differences(rd):
            found[what].append((rd.rel_path, txt))
    return {"samples": len(rows), "compared": compared, "found": found}
