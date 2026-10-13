"""What counts as "the same sound", in four tiers with one meaning wherever the build asks
(docs/curation.md, "Identity"):

- **recording**: the same audio. A file's bytes (its hash), or curate._same_recording's
  correlation over the start of the sound. Never asked of synthetic single cycles (a waves
  category, `comparable`): a sine and a triangle cycle correlate at 0.99.
- **edition**: one recording released more than once (`editions`): a .wav and an .aif of one
  name, mono and stereo or 16- and 24-bit twin folders, a raw take and its processed edition
  (a last name word EDITION_SUFFIX_RE matches: "_Cleanup", "_Dirty", "-Clean", "-Tube"), and
  a compilation's re-release under another pack of its vendor. The names decide, with lengths
  within EDITION_DUR_TOL.
- **set**: a multisample set, one instrument or patch in one source folder whose names differ
  only by note, velocity or round robin (`set_key`). A bare number is a note only when the
  numbers track the measured pitch (`number_sets`, NUMBER_NOTE_TOL).
- **series**: a numbered batch ("Qzk 01".."Qzk 24", "106_QZ", "Hinge07"), never a set.

The set cap outside the drum categories also binds one named patch's numbered takes and
variant letters ("Drop A 01".."Drop C 09", `patch_key`); in the drum categories
(SIBLING_SETS_ONLY) only sets. Twins of one note and velocity that differ only by their round
robin are one sound (`rr_key`), and so are a kit's two takes of one drum (`take_name`).
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

from .curate_config import EDITION_DUR_TOL, EDITION_SUFFIX_RE, NUMBER_NOTE_TOL, SIBLING_SETS_ONLY

_VEL = re.compile(r"^(?:v|vel|velocity|layer)\d+$|^(?:ppp|pp|p|mp|mf|f|ff|fff)$")
_DYN = re.compile(r"^(?:ppp|pp|p|mp|mf|f|ff|fff)$")
_RR = re.compile(r"^rr\d+$")
_TAKE = re.compile(r"^take\d+$")
_NUM = re.compile(r"^\d+$")
_LETTER = re.compile(r"^[a-z]$")
_MIC = re.compile(r"^\+")                  # a mic perspective after the drum ("+OH", "+Room")
_SPLIT = re.compile(r"[\s_\-]+")


def _C():
    from . import curate
    return curate


def _clean(seg):
    """A folder or pack name without its edition words ("Pack (WAV mono)" -> "pack")."""
    return _C()._EDITION_RE.sub("", seg or "").strip(" -_()").lower()


def _where(path):
    """(vendor, pack, folder) of a source, each without edition words: under the library its
    first folder, first two folders and whole folder; elsewhere the folder for all three."""
    from ..places import in_library, library_rel
    if not in_library(path):
        f = "/".join(s for s in map(_clean, os.path.dirname(path or "").split("/")) if s)
        return f, f, f
    segs = [s for s in map(_clean, library_rel(path).split("/")[:-1]) if s] or [""]
    return segs[0], "/".join(segs[:2]), "/".join(segs)


def _generic(words):
    """True when the words left of a name can't name a patch on their own (one generic or
    short word: "Kick", "Pad", "Dim"), as curate._sibling_key has it."""
    return len(words) < 2 and (not words or len(words[0]) < 4 or words[0] in _C()._SIB_GENERIC)


def _parts(path):
    """curate._sib_tokens's tokens of a name (notes marked) less a last processing word, and
    which of them state a velocity (a dynamics word after the first: "MP_Kick" is a prefix)."""
    toks, note = _C()._sib_tokens(path)
    if len(toks) > 1 and EDITION_SUFFIX_RE.match(toks[-1]):
        toks, note = toks[:-1], note[:-1]
    vel = [bool(_VEL.match(t)) and not (i == 0 and _DYN.match(t)) for i, t in enumerate(toks)]
    return toks, note, vel


# --- recording -------------------------------------------------------------------------------

def comparable(kind):
    """Whether a category of this kind is compared as audio for copies: never synthetic
    single cycles (two different waveforms correlate at 0.99)."""
    return kind != "waves"


# --- edition ---------------------------------------------------------------------------------

def _edition_tokens(path):
    """(name tokens without edition words, the processing word that ended the name or "")."""
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    toks = [t for t in re.split(r"[^a-z0-9#+]+", _C()._EDITION_RE.sub(" ", stem).lower()) if t]
    if len(toks) > 1 and EDITION_SUFFIX_RE.match(toks[-1]):
        return toks[:-1], toks[-1]
    return toks, ""


def edition_name(path):
    """A file's name without its extension, edition words or a last processing word
    ("Hit 87 Cleanup.wav", "Hit 87.aif" -> "hit 87")."""
    return " ".join(_edition_tokens(path)[0])


def edition_key(path):
    """(vendor, edition_name): what two editions of one recording share (`editions` decides)."""
    return _where(path)[0], edition_name(path)


def _named(toks):
    """A name that tells one recording from another: a number in it, or a word that names
    something ("Hit 87", "Slap"; never "Kick" alone)."""
    words = [t for t in toks if not t.isdigit()]
    return len(words) < len(toks) or not _generic(words)


def editions(a, b, dur_a=None, dur_b=None):
    """True when sources a and b are two editions of one recording by their names, in one
    pack: a processed edition (names that differ by a last processing word: "Hit 87" and
    "Hit 87 Cleanup", trimmed differently), or one name in folders that differ only by edition
    words (a .wav and an .aif side by side, mono/stereo or 16/24-bit twin folders) with
    lengths within EDITION_DUR_TOL (when both are known); across two packs of one vendor, one
    distinctive name of one length (a compilation's re-release). One name in two folders of a
    pack ("808/Kick 01", "909/Kick 01") is two recordings."""
    if a == b:
        return False
    (ta, sa), (tb, sb) = _edition_tokens(a), _edition_tokens(b)
    if not ta or ta != tb or not _named(ta):
        return False
    (va, pa, fa), (vb, pb, fb) = _where(a), _where(b)
    if va != vb:
        return False
    if pa == pb and sa != sb:
        return True
    if dur_a and dur_b and abs(dur_a - dur_b) > EDITION_DUR_TOL * max(dur_a, dur_b):
        return False
    if pa == pb:
        return fa == fb
    return len([t for t in ta if not t.isdigit()]) >= 2


def edition_pairs(paths, durs=None):
    """[(i, j)] index pairs (i < j) of paths that are editions of one recording (`editions`);
    durs: their lengths in seconds (None where unknown)."""
    durs = durs or [None] * len(paths)
    by = defaultdict(list)
    for i, p in enumerate(paths):
        k = edition_key(p)
        if k[1]:
            by[k].append(i)
    out = []
    for idx in by.values():
        for x, i in enumerate(idx):
            for j in idx[x + 1:]:
                if editions(paths[i], paths[j], durs[i], durs[j]):
                    out.append((min(i, j), max(i, j)))
    return sorted(out)


# --- set and series --------------------------------------------------------------------------

def set_key(path):
    """(source folder, name) of a multisample set whose names state a note, a velocity or a
    round robin ("Qzv Pad C3 v2", "Qzv Pad E3 v1"); numbers, takes and a processing word aside.
    None for a name that states none of them (a series, or a set `number_sets` may find)."""
    toks, note, vel = _parts(path)
    if not any(n or v or _RR.match(t) for t, n, v in zip(toks, note, vel)):
        return None
    words = [t for t, n, v in zip(toks, note, vel)
             if not (n or v or _RR.match(t) or _TAKE.match(t) or _NUM.match(t))]
    return (_where(path)[2], " ".join(words)) if words else None


def _number_key(path):
    """((source folder, name), numbers) of a patch named with bare numbers and no note
    ("Qzv Lead 059", "Qzv-Lead-40-90"), else None."""
    toks, note, vel = _parts(path)
    if any(note) or any(vel) or any(_RR.match(t) for t in toks):
        return None
    nums = tuple(int(t) for t in toks if _NUM.match(t))
    words = [t for t in toks if not _NUM.match(t) and not _TAKE.match(t)]
    if not nums or _generic(words):
        return None
    return (_where(path)[2], " ".join(words)), nums


def number_sets(items):
    """{path: set key} of the files named by a patch and bare numbers whose numbers are notes:
    in one source folder, for one of the name's numbers, each file's measured root less that
    number agrees within NUMBER_NOTE_TOL semitones across at least three numbers (0-127)
    ("Qzv Lead 059" / "061" / "062" rooted at 58.8, 60.8, 61.8). Numbers that don't track the
    pitch are takes, a series ("Qzv Lead 01" / "02" both at C3). items: [(path, root midi or
    None)]."""
    groups = defaultdict(list)
    for path, root in items:
        nk = _number_key(path)
        if nk is not None:
            groups[(nk[0], len(nk[1]))].append((path, nk[1], root))
    out = {}
    for (key, n), members in groups.items():
        rooted = [(nums, float(r)) for _p, nums, r in members if r is not None]
        for pos in range(n):
            vals = {nums[pos] for nums, _r in rooted}
            if len(vals) < 3 or max(vals) > 127:
                continue        # (two numbers may agree with two roots by chance)
            off = [r - nums[pos] for nums, r in rooted]
            if max(off) - min(off) <= NUMBER_NOTE_TOL:
                out.update({p: key for p, _n, _r in members})
                break
    return out


def set_keys(items):
    """{path: set key} for these (path, root midi or None): a name's note, velocity or round
    robin (set_key), else numbers that track the pitch (number_sets); None for the rest."""
    items = list(items)
    nums = number_sets(items)
    return {p: set_key(p) or nums.get(p) for p, _r in items}


def patch_key(path):
    """(source folder, patch name) of a set, or of one named patch's numbered takes and variant
    letters ("Qzv Drop A 01" .. "Qzv Drop C 09": "qzv drop"); None for a series of generic hits
    ("Kick 01", "106_QZ") and names glued to a number ("Hinge07": each its own patch)."""
    k = set_key(path)
    if k is not None:
        return k
    toks, _note, vel = _parts(path)
    words = []
    for i, t in enumerate(toks):
        if _NUM.match(t) or vel[i] or _TAKE.match(t) or _RR.match(t):
            continue
        if _LETTER.match(t) and i + 1 < len(toks) and _NUM.match(toks[i + 1]) and len(words) >= 2:
            continue      # a variant letter after a described patch, before its take ("Qzv Drop B 03";
            #               "Qzv B 57" is a bank and a preset)
        words.append(t)
    # one word and a number is a patch when the word names one ("Frobnik 47"), not a generic
    # word ("Kick 01", "Pad 3")
    if not words or len(words) < 2 and (_generic(words) or not any(_NUM.match(t) for t in toks)):
        return None
    return _where(path)[2], " ".join(words)


def cap_key(path, category, numbered=None):
    """The group the set cap counts a file in: a set (set_key, or numbered's: a number_sets
    result) in the drum categories (SIBLING_SETS_ONLY), where a pack's numbered hits are a
    series; elsewhere also one named patch's takes (patch_key)."""
    if category in SIBLING_SETS_ONLY:
        return set_key(path) or (numbered or {}).get(path)
    return patch_key(path)


# --- round robins and kit takes ------------------------------------------------------------------

def rr_key(path):
    """(source folder, name less its round robin) of a file whose name numbers a round robin
    ("Qzv_C2_v3_rr2"): its twins of the same note and velocity share it. None otherwise."""
    stem = os.path.splitext(os.path.basename(path or ""))[0].lower()
    toks = [t for t in _SPLIT.split(stem) if t]
    rest = [t for t in toks if not _RR.match(t)]
    if len(rest) == len(toks) or not rest:
        return None
    return _where(path)[2], " ".join(rest)


def rr_number(path):
    """A name's round robin number (0 when it has none)."""
    stem = os.path.splitext(os.path.basename(path or ""))[0].lower()
    return next((int(t[2:]) for t in _SPLIT.split(stem) if _RR.match(t)), 0)


def take_name(stem):
    """The drum a kit candidate's name (its take number already off) is a take of: without a
    mic perspective ("+OH"), a processing word, and a variant letter with the articulation
    before it ("Qzv Tom Brush A +OH" and "Qzv Tom Stick A" are one tom)."""
    toks = [t for t in _SPLIT.split(stem.lower()) if t and not _MIC.match(t)]
    if len(toks) > 1 and EDITION_SUFFIX_RE.match(toks[-1]):
        toks = toks[:-1]
    if len(toks) > 2 and _LETTER.match(toks[-1]):
        toks = toks[:-2]
    return " ".join(toks)
