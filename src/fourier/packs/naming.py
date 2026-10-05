"""Family (folder) naming for curated categories — pure functions, no DB or audio.

A family name is  lead + body + traits:

  lead    tempo for loops ("166bpm", "110-120bpm", "freetempo"), note for tonal
          categories ("cs2"), a band word where a category has bands ("closed",
          "string"; PHRASES: role then tempo, "chords-120-130bpm"); drums have none.
  body    what the folder IS: the CLAP phrase that best separates this cluster from
          its category-mates, else the category noun.
  traits  measured character, in the category's own vocabulary (tonal sounds are not
          "gated" or "clicky"), relative to the category — and, when several families
          share a body, first by what separates those siblings. A trait never repeats
          or contradicts the body ("bright-thin-kick" gets no brightness or weight word).

Names are built from whole units: a hyphenated trait ("long-tail") is added whole or
not at all, and the tempo lead is one unit.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

import numpy as np

from ..settings import for_module as _for_module

from .curate_config import taxonomy_view as _tax   # config/taxonomy.yaml + an overlay's added categories
_tunable = _for_module("naming")   # overridable: fourier/settings.py, config/tunables.yaml

NAME_CAP = _tunable("NAME_CAP", 5)            # max units in a name (lead, each body word, each whole trait)
NAME_MAX_CHARS = _tunable("NAME_MAX_CHARS", 40)     # a trait is added only while the name stays within this
CLAP_Z = _tunable("CLAP_Z", 1.0)            # a phrase names a cluster when its z across clusters is >= this
CLAP_Z_FALLBACK = _tunable("CLAP_Z_FALLBACK", 0.5)   # weaker match, used only for otherwise feature-only clusters and
                        # only with a phrase no other family in the category carries
TRAIT_MIN_Z = _tunable("TRAIT_MIN_Z", 0.3)       # a measured dimension becomes a trait past this |z|
# Envelope words must also be true in absolute terms, on the folder's medians (attack in ms,
# source length in s): relative to the category they mislead on a small screen (a "soft"
# folder of fast attacks, a "short" pad that runs for seconds, a "long-tail" folder of
# short hits).
TRAIT_GATES = _tunable("TRAIT_GATES", {
    "soft": ("atk", ">=", 30.0), "snappy": ("atk", "<=", 15.0),
    "plucky": ("atk", "<=", 50.0), "swelling": ("atk", ">=", 100.0),
    "short": ("dur", "<=", 1.5), "sustained": ("dur", ">=", 1.0),
    "long-tail": ("dur", ">=", 0.5), "gated": ("dur", "<=", 1.0),
})


# a loop folder is "musical" only when its loops are harmonic in absolute terms, not merely
# more than their siblings (a folder of drums alone isn't): its median harmonicity, which a
# build without Sononym records (c["med"]["har"], the harmonic share of the audio's energy),
# and, when recorded, its median pitch focus (c["med"]["chroma"]) at Fourier's own pitched
# call (metadata/resolve.OWN_PITCHED_CHROMA): drums alone, filtered, can read half harmonic,
# but they focus on no pitch class as a chord or a line does
MUSICAL_HAR_MIN = _tunable("MUSICAL_HAR_MIN", 0.5)


def trait_ok(word, c):
    """Does a folder's median (c["med"]) bear out an envelope word (TRAIT_GATES), or
    "musical" (MUSICAL_HAR_MIN, and Fourier's own pitched call on the median pitch focus,
    when they're recorded)? Words without a gate, or folders without medians, pass."""
    med = c.get("med") or {}
    if word == "musical" and med.get("har") is not None:
        from ..metadata.resolve import OWN_PITCHED_CHROMA
        return med["har"] >= MUSICAL_HAR_MIN and (med.get("chroma") is None
                                                  or med["chroma"] >= OWN_PITCHED_CHROMA)
    g = TRAIT_GATES.get(word)
    if not g or med.get(g[0]) is None:
        return True
    v = med[g[0]]
    return v >= g[2] if g[1] == ">=" else v <= g[2]

# measured feature dimensions (key -> (low word, high word)); None = say nothing that side
ONESHOT_DIMS = _tunable("ONESHOT_DIMS", [
    ("tune", ("low-tuned", "high-tuned")), ("br", ("dark", "bright")), ("noi", ("clean", "gritty")),
    ("har", ("clicky", "tonal")), ("atk", ("snappy", "soft")), ("dec", ("gated", "long-tail")),
    ("sub", ("thin", "sub-heavy")), ("cr", ("squashed", "punchy")),
])
DIM_WEIGHT = _tunable("DIM_WEIGHT", {"cr": 0.4})
# drum loops: low harmonicity is just "drums" (the whole category) — only say "musical"
LOOP_DIMS = _tunable("LOOP_DIMS", [("br", ("dark", "bright")), ("noi", ("clean", "gritty")), ("har", (None, "musical"))])
# pitched one-shots: the note lead carries register, so no tune/sub words; envelope words
# are the synth ones, not drum ones
TONAL_DIMS = _tunable("TONAL_DIMS", [
    ("br", ("dark", "bright")), ("noi", ("clean", "gritty")),
    ("atk", ("plucky", "swelling")), ("dec", ("short", "sustained")),
])

MELODIC_CATS = _tunable("MELODIC_CATS", _tax.with_role("melodic"))
# the note lead ("c4-") only where a folder is one instrument across its range (in the
# timbre-clustered categories it sorted by name, not pitch, and said little); ACOUSTIC's
# band lead carries its own note. Mirrors curate_config.NOTE_LEAD_CATS.
NOTE_LEAD_CATS = _tunable("NOTE_LEAD_CATS", _tax.with_role("note_lead"))
TONAL_CATS = _tunable("TONAL_CATS", _tax.with_role("tonal"))   # the one-shot melodic ones
# weight word per drum/fx category; categories absent here get no weight trait (a
# "sub-heavy" hat or cymbal is a measurement artefact, not a character)
SUB_WORD = _tunable("SUB_WORD", {"KICKS": "sub-heavy", "FX": "sub-heavy", "SNARES": "full", "TOMS": "full",
            "PERC": "full", "BLIPS": "full"})
# noise-based sounds: a detected "pitch" is unreliable and not how anyone picks them
NO_TUNE_CATS = _tunable("NO_TUNE_CATS", {"HATS", "CYMBALS", "CLAPS"})

CATEGORY_NOUN = _tunable("CATEGORY_NOUN", _tax.nouns())

# the category's own noun, left out of its folder names:
# KICKS/tight-punchy-gritty, not tight-punchy-kick-gritty. The category folder says
# it already. PIANO / ACOUSTIC / WAVES keep theirs: there the word tells piano from organ,
# or names the instrument or wave type.
NOUN_DROP = _tunable("NOUN_DROP", {
    "KICKS": {"kick", "kicks"}, "SNARES": {"snare", "snares"}, "CLAPS": {"clap", "claps"},
    "HATS": {"hat", "hats", "hihat", "hihats", "hi"}, "CYMBALS": {"cymbal", "cymbals"},
    "TOMS": {"tom", "toms"}, "PERC": {"perc", "percussion"}, "SUB": {"bass"},
    "SYNTH": {"synth"}, "STABS": {"stab", "stabs"}, "PADS": {"pad", "pads"},
    "FX": {"fx"}, "BLIPS": {"blip", "blips"}, "VOX": {"vox"},
    "DRUMLOOPS": {"loop", "loops"}, "PHRASES": {"loop", "loops"},
})
NOUN_DROP = _tax.add_values(NOUN_DROP, "folder_words", set)   # ...and an added category's own


def drop_noun(body, category):
    """body phrases without the category's own noun (NOUN_DROP); empty ones dropped."""
    drop = NOUN_DROP.get(category)
    if not drop:
        return list(body)
    out = []
    for b in body:
        ws = [w for w in words(b) if w not in drop]
        if ws:
            out.append(" ".join(ws))
    return out


# body words that already speak to a dimension: a trait on that dimension would repeat
# it ("soft-round-kick-soft") or contradict it ("bright-thin-kick" + "sub-heavy")
DIM_SIGNALS = _tunable("DIM_SIGNALS", {
    "br": {"dark", "bright", "dull"},
    "noi": {"clean", "gritty", "noisy", "noise", "distorted", "crunchy", "dirty", "lo", "dusty",
            "bitcrush", "airy"},
    "har": {"tonal", "clicky", "click", "pure", "musical", "percussive", "noisy", "chirpy", "chirp"},
    "atk": {"snappy", "soft", "plucky", "pluck", "plucked", "swelling", "swell", "reverse",
            "reversed"},
    "dec": {"short", "long", "gated", "sustained", "staccato", "tail"},
    "tune": {"high", "low", "deep", "tuned", "pitched"},
    "sub": {"thin", "sub", "full", "fat", "boomy", "rumble", "heavy"},
    "cr": {"punchy", "squashed", "tight"},
})

# body words that rule out one side of a dimension: a "stab" or "hit" is by definition
# not slow to start, a "staccato" sound not long
TRAIT_CONFLICTS = _tunable("TRAIT_CONFLICTS", {
    "stab": {"swelling"}, "hit": {"swelling"}, "pluck": {"swelling"}, "plucked": {"swelling"},
    "staccato": {"swelling", "sustained", "long-tail"}, "shot": {"swelling"},
    "drone": {"plucky", "short", "gated"}, "wash": {"plucky", "snappy"},
})

_NOTE = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]


def words(tok) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", str(tok).lower()) if w]


def slug(tok) -> str:
    return "-".join(words(tok))


def note_name(m):
    """MIDI note -> filename-safe pitch label (sharps as 's': C#4 -> Cs4)."""
    if m is None:
        return None
    m = int(round(m))
    return f"{_NOTE[m % 12]}{m // 12 - 1}"


def tempo_lead(c, spread_max=2, step=5):
    """Folder lead for a loop family: its tempo when its files share one, else its range
    widened to multiples of `step`, so every file's tempo is inside the label (a 5-BPM band
    reads "090-095bpm")."""
    lo, hi = c.get("bpm_range") or (None, None)
    if lo is not None and hi - lo > spread_max:
        lo = int(math.floor(lo / step) * step)
        hi = int(math.ceil(hi / step) * step)
        return f"{lo:03d}-{hi:03d}bpm"
    return f"{round(c['bpm']):03d}bpm"


def naming_dims(category, cfg, loop_dims=None):
    """The trait vocabulary for a category."""
    kind = cfg["kind"]
    if kind != "oneshot":
        return cfg.get("dims", loop_dims or LOOP_DIMS)
    if category in TONAL_CATS:
        return TONAL_DIMS
    out = []
    for key, pair in ONESHOT_DIMS:
        if key == "tune" and category in NO_TUNE_CATS:
            continue
        if key == "sub":
            w = SUB_WORD.get(category)
            if not w:
                continue
            pair = (pair[0], w)
        out.append((key, pair))
    return out


def rank_traits(clusters, dims, kind=None):
    """Per cluster: z-score each dimension across the category (c["z"]) and list the
    trait words past TRAIT_MIN_Z, strongest first (c["traits"]; may be empty)."""
    keys = [k for k, _ in dims]
    tmean = None
    if "tune" in keys:
        tv = [c["tune"] for c in clusters if c.get("tune") is not None]
        tmean = float(np.mean(tv)) if tv else 60.0
    Z = {}
    for key in keys:
        vals = np.array([(c.get(key) if c.get(key) is not None else
                          (tmean if key == "tune" else 0.0)) for c in clusters], float)
        Z[key] = (vals - vals.mean()) / (vals.std() + 1e-9)
    for j, c in enumerate(clusters):
        c["z"] = {key: float(Z[key][j]) for key in keys}
        scored = []
        for key, pair in dims:
            z = Z[key][j]
            side = pair[1] if z > 0 else pair[0]
            if side and abs(z) > TRAIT_MIN_Z and trait_ok(side, c):
                scored.append((side, abs(z) * DIM_WEIGHT.get(key, 1.0)))
        scored.sort(key=lambda t: -t[1])
        c["traits"] = [w for w, _ in scored]


def choose_phrases(Zp, phrases, clap_z=CLAP_Z, fallback_z=CLAP_Z_FALLBACK, n_primary=None):
    """Phrase per cluster (or None). Zp: clusters x phrases, z-scored per phrase.

    Phrases come in two tiers: the first n_primary (the category's own vocabulary,
    mostly what a sound IS) and the rest (naming-only genre words, a looser claim).
    A cluster takes its top primary phrase when that z >= clap_z; otherwise its top
    secondary one at that bar. Clusters still unnamed may then take a weaker
    (>= fallback_z) phrase that no other family carries -- primary tier first, strongest
    match first -- so a folder is described by a word rather than only by numbers."""
    Zp = np.asarray(Zp, float)
    n, m = Zp.shape
    n_primary = m if n_primary is None else n_primary
    tiers = [range(0, n_primary), range(n_primary, m)]
    out = [None] * n
    for tier in tiers:
        if not len(tier):
            continue
        for j in range(n):
            if out[j] is not None:
                continue
            q = tier[int(np.argmax(Zp[j, tier.start:tier.stop]))]
            if Zp[j, q] >= clap_z:
                out[j] = phrases[q]
    if fallback_z is None:
        return out
    taken = {p for p in out if p}
    for tier in tiers:
        cand = sorted(((float(Zp[j, q]), j, q) for j in range(n) if out[j] is None
                       for q in tier if Zp[j, q] >= fallback_z),
                      key=lambda t: (-t[0], t[1], t[2]))
        for _z, j, q in cand:
            if out[j] is None and phrases[q] not in taken:
                out[j] = phrases[q]
                taken.add(phrases[q])
    return out


def conflicting(body_words):
    """Trait words the body rules out (TRAIT_CONFLICTS)."""
    return {t for w in body_words for t in TRAIT_CONFLICTS.get(w, ())}


def _signals(dims):
    return {key: DIM_SIGNALS.get(key, set()) | {w for side in pair if side for w in words(side)}
            for key, pair in dims}


def blocked_dims(body_words, dims):
    """Dimensions the body already speaks to (their traits would repeat or contradict)."""
    bw = set(body_words)
    return {key for key, sig in _signals(dims).items() if sig & bw}


def sibling_traits(clusters, bodies, dims, blocked, banned=None):
    """For families sharing a body, the one trait that best separates each from its
    siblings (deviation from the siblings' mean, in category z units). {j: word}."""
    groups = defaultdict(list)
    for j, b in enumerate(bodies):
        groups[tuple(b)].append(j)
    out = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        for j in members:
            best = None
            for key, pair in dims:
                if key in blocked[j]:
                    continue
                mu = float(np.mean([clusters[i]["z"][key] for i in members]))
                dev = (clusters[j]["z"][key] - mu) * DIM_WEIGHT.get(key, 1.0)
                side = pair[1] if dev > 0 else pair[0]
                if banned and side in banned[j]:
                    continue
                if not trait_ok(side, clusters[j]):
                    continue
                if side and abs(dev) > 1e-6 and (best is None or abs(dev) > best[0]):
                    best = (abs(dev), side)
            if best:
                out[j] = best[1]
    return out


def assemble_name(lead, body, traits, ntr, cap=NAME_CAP, max_chars=NAME_MAX_CHARS):
    """lead + body + up to ntr whole traits -> (name, words used)."""
    parts, seen = [], set()
    for tok in lead:
        s = slug(tok)
        if s:
            parts.append(s)
            seen.update(words(tok))
    for w in body:
        for x in words(w):
            if x not in seen:
                parts.append(x)
                seen.add(x)
    parts = parts[:cap]
    added = 0
    for t in traits:
        if added >= ntr or len(parts) >= cap:
            break
        ws = words(t)
        if not ws or any(x in seen for x in ws):
            continue
        if len("-".join(parts + ["-".join(ws)])) > max_chars:
            continue
        parts.append("-".join(ws))
        seen.update(ws)
        added += 1
    return "-".join(parts), seen


_BPM_TOK = re.compile(r"^(\d{2,3})bpm$")
SAME_TEMPO_BPM = _tunable("SAME_TEMPO_BPM", 2)      # folder tempos this close read as the same folder ("126bpm" / "127bpm")


def _name_key(nm):
    """(word set, tempo) of a folder name: word order doesn't make a different folder."""
    ws, bpm = [], None
    for w in nm.lower().split("-"):
        m = _BPM_TOK.match(w)
        if m and bpm is None:
            bpm = int(m.group(1))
        elif w:
            ws.append(w)
    return frozenset(ws), bpm


def colliding_name(nm, used):
    """The used name nm reads as: the same words in any order ("pad-gritty-bright" /
    "pad-bright-gritty"), a tempo within SAME_TEMPO_BPM counting as the same tempo. None
    if it reads as new."""
    k, b = _name_key(nm)
    for u in used:
        ku, bu = _name_key(u)
        if k == ku and (b == bu or (b is not None and bu is not None and abs(b - bu) <= SAME_TEMPO_BPM)):
            return u
    return None


def twin_names(a, b):
    """Two sibling folder names that only reorder or drop a word of each other
    ("bright-sustained" / "bright-swelling-sustained"): hard to tell apart on a device.
    Tempo-led names are exempt (their ranges tell them apart)."""
    wa = frozenset(re.sub(r"-\d{1,2}$", "", a.lower()).split("-"))   # "-2" isn't a word; "303" is
    wb = frozenset(re.sub(r"-\d{1,2}$", "", b.lower()).split("-"))
    return wa == wb or ((wa < wb or wb < wa) and not re.search(r"\d{3}bpm", a + b))


def name_collides(nm, used):
    """True if nm reads as a name already used (colliding_name)."""
    return colliding_name(nm, used) is not None


def pairwise_traits(c, other, dims, blocked=(), banned=(), taken=()):
    """Trait words that tell c from the folder it collides with, largest measured
    difference first, pointing c's way (so "-gritty" goes on the grittier of the two)."""
    out = []
    for key, pair in dims:
        if key in blocked or key not in (c.get("z") or {}) or key not in (other.get("z") or {}):
            continue
        d = (c["z"][key] - other["z"][key]) * DIM_WEIGHT.get(key, 1.0)
        side = pair[1] if d > 0 else pair[0]
        if side and abs(d) > 1e-6 and side not in banned and not set(words(side)) & set(taken) \
                and trait_ok(side, c):
            out.append((abs(d), side))
    return [w for _, w in sorted(out, key=lambda t: -t[0])]


def unique_name(nm, seen, traits, used):
    """Disambiguate against names already used in the category (case-insensitive, word
    order ignored, name_collides): first by one more whole trait, then by -2, -3, ..."""
    if not name_collides(nm, used):
        return nm
    for t in traits:
        ws = words(t)
        if ws and not any(x in seen for x in ws):
            cand = f"{nm}-{'-'.join(ws)}"
            if not name_collides(cand, used):
                return cand
    i = 2
    while f"{nm}-{i}".lower() in used:
        i += 1
    return f"{nm}-{i}"


BAND_DEFAULT = _tunable("BAND_DEFAULT", {"wave": "cycle", "hat": "closed", "fx": "misc", "cymbal": "cymbal"})


def _body_and_lead(c, category, cfg, phrase, noun):
    kind = cfg["kind"]
    if kind == "instrument":
        if cfg.get("clap_source") and phrase:
            body = phrase
        elif c.get("source"):
            body = [c["source"]]
        elif phrase:
            body = phrase
        else:
            body = [noun]
    elif kind == "waves":
        body = [c["source"]] if c.get("source") else (phrase or [noun])
    elif c.get("inst"):
        # the instrument most files name (name_instruments): a phrase naming it says more ("church
        # bell"), any other phrase would contradict the files ("music box" on xylophones)
        inst_w = set(words(c["inst"]))
        body = phrase if (phrase and inst_w <= set(phrase)) else [c["inst"]]
    else:
        body = phrase if phrase else [noun]
    note_tok = note_name(c["tune"]) if c.get("tune") is not None else None
    bt, band = c.get("band_type"), c.get("band")
    if bt in ("fx", "cymbal") and not phrase and body == [noun]:
        body = []                       # "foley-bright", "ride-dark": the band word says it
    tops = ["tops"] if (bt == "loop" and band == "tops") else []
    if bt == "phrase":
        # role first, then tempo: bass-120-130bpm-acid-bassline sorts basslines together,
        # slowest first
        lead = [band or "lead"] + ([tempo_lead(c)] if c.get("bpm") else
                                   (["freetempo"] if c.get("free_tempo") else []))
    elif bt == "loop" and band == "classic":
        lead, body = ["classic"], ["breaks"]    # one folder, every tempo: "classic-breaks"
    elif kind == "loop" and c.get("bpm"):
        lead = [tempo_lead(c)] + tops          # zero-padded so folders sort by tempo
    elif kind == "loop" and c.get("free_tempo"):
        lead = ["freetempo"] + tops            # no confirmable tempo: say so, don't guess
    elif tops:
        lead = tops
    elif bt in ("wave", "hat", "fx", "cymbal"):
        lead = [band or BAND_DEFAULT[bt]]           # cycle|table, closed|open, impact|sweep|...
    elif bt == "acoustic":
        lead = [band or "string"] + ([note_tok] if note_tok else [])   # mallet-c5-xylophone
    elif bt == "chord" and band == "chord":
        lead = ["chord"]                            # chord stabs sort together, no single note
    elif category in NOTE_LEAD_CATS and note_tok:
        lead = [note_tok]
    else:
        lead = []
    return list(body), lead


def name_families(clusters, category, cfg, Zp, phrases, clap_z=CLAP_Z,
                  fallback_z=CLAP_Z_FALLBACK, dims=None, n_primary=None):
    """Name every cluster in a category. Needs rank_traits() to have run (c["z"],
    c["traits"]). Sets c["clap_phrase"], c["clap_z"] (that phrase's z, None when
    unnamed), c["traits"] (final, in name order) and c["name"]; returns the names."""
    dims = dims or naming_dims(category, cfg)
    word_key = {w: key for key, pair in dims for w in pair if w}
    noun = CATEGORY_NOUN.get(category, category.lower())
    cap = family_cap()
    room = NAME_MAX_CHARS + 4 if cap is None else min(NAME_MAX_CHARS + 4, cap)
    chosen = choose_phrases(Zp, phrases, clap_z, fallback_z, n_primary)
    Zp = np.asarray(Zp, float)

    bodies, leads, blocked, banned = [], [], [], []
    for j, c in enumerate(clusters):
        c["clap_phrase"] = chosen[j]
        c["clap_z"] = round(float(Zp[j, phrases.index(chosen[j])]), 2) if chosen[j] else None
        phrase = words(chosen[j]) if chosen[j] else []
        body, lead = _body_and_lead(c, category, cfg, phrase, noun)
        body = drop_noun(body, category)
        bodies.append(body)
        leads.append(lead)
        bw = [x for b in body for x in words(b)]
        blocked.append(blocked_dims(bw, dims))
        banned.append(conflicting(bw))
    sib = sibling_traits(clusters, bodies, dims, blocked, banned)

    # traits read in one fixed order (the dimension order), one word per dimension, so
    # siblings never differ only by word order ("sustained-swelling" / "swelling-sustained")
    order = {side: i for i, (key, pair) in enumerate(dims) for side in pair if side}
    parts = {}                    # j -> (prefix tokens, chosen trait words, seen words)

    def _name(j):
        pre, tw, _ = parts[j]
        return "-".join(pre + ["-".join(words(t)) for t in sorted(tw, key=lambda t: order.get(t, 99))]) or noun

    def _fits(j, t):
        pre, tw, seen = parts[j]
        ws = words(t)
        return (ws and not any(x in seen for x in ws) and word_key.get(t) not in
                {word_key.get(x) for x in tw} and len(_name(j)) + 1 + len("-".join(ws)) <= room)

    def _add(j, t):
        pre, tw, seen = parts[j]
        parts[j] = (pre, tw + [t], seen | set(words(t)))

    used = set()
    owner = {}
    for j, c in enumerate(clusters):
        name_words = {x for t in leads[j] + bodies[j] for x in words(t)}
        traits = [t for t in c.get("traits", [])
                  if word_key.get(t) not in blocked[j] and t not in banned[j]
                  and not set(words(t)) & name_words]
        if j in sib:
            k = word_key.get(sib[j])
            traits = [sib[j]] + [t for t in traits if word_key.get(t) != k]
        ntr = 1 if (leads[j] or len(bodies[j]) >= 2) else 2
        full, seen = (assemble_name(leads[j], bodies[j], traits, ntr) if cap is None else
                      assemble_name(leads[j], bodies[j], traits, ntr, max_chars=min(NAME_MAX_CHARS, cap)))
        pre, _ = assemble_name(leads[j], bodies[j], [], 0)
        pre = pre.split("-") if pre else []
        chosen = full.split("-")[len(pre):] if full else []
        tw = [t for t in traits if set(words(t)) <= set(chosen)][:ntr]
        parts[j] = (pre, tw, set(seen))
        c["traits"] = sorted(tw, key=lambda t: order.get(t, 99)) + [t for t in traits if t not in tw]
        # a clash is settled by a real difference: the family's other traits, then its
        # weaker measured leanings (below TRAIT_MIN_Z), before a bare number
        weak = sorted(((pair[1] if c["z"][key] > 0 else pair[0], abs(c["z"][key]) * DIM_WEIGHT.get(key, 1.0))
                       for key, pair in dims if key in c.get("z", {}) and key not in blocked[j]),
                      key=lambda t: -t[1])
        weak = [w for w, _ in weak if w and w not in traits and w not in banned[j]
                and not set(words(w)) & name_words and trait_ok(w, c)]
        nm = _name(j)
        other = colliding_name(nm, used)
        if other is not None:
            # what tells the two apart (pairwise), then the rest; one word per dimension
            pair = pairwise_traits(c, clusters[owner[other]], dims, blocked[j], banned[j], parts[j][2]) \
                if other in owner else []
            for t in pair + traits + weak:
                if _fits(j, t):
                    _add(j, t)
                    if not name_collides(_name(j), used):
                        break
                    pre_, tw_, seen_ = parts[j]
                    parts[j] = (pre_, tw_[:-1], seen_ - set(words(t)))
            nm = _name(j)
            if name_collides(nm, used):
                i = 2
                while f"{nm}-{i}".lower() in used:
                    i += 1
                nm = f"{nm}-{i}"
        owner[nm.lower()] = j
        used.add(nm.lower())
        c["name"] = nm

    # a folder whose words are all inside a sibling's ("short-plucky" beside
    # "short-plucky-dark") says the other side of what sets that sibling apart, when it
    # measures that way ("short-plucky-bright")
    for _ in range(3):
        changed = False
        for a in range(len(clusters)):
            for b in range(len(clusters)):
                if a == b or re.search(r"-\d+$", clusters[a]["name"]):
                    continue
                wa, wb = set(clusters[a]["name"].split("-")), set(clusters[b]["name"].split("-"))
                if not wa < wb:
                    continue
                for t in parts[b][1]:
                    k = word_key.get(t)
                    if not k or not set(words(t)) <= (wb - wa):
                        continue
                    pair = dict(dims)[k]
                    opp = pair[0] if t == pair[1] else pair[1]
                    za, zb = clusters[a].get("z", {}).get(k), clusters[b].get("z", {}).get(k)
                    if not opp or za is None or zb is None or k in blocked[a] or opp in banned[a]:
                        continue
                    if (za - zb > 0) != (opp == pair[1]) or not trait_ok(opp, clusters[a]) or not _fits(a, opp):
                        continue
                    _add(a, opp)
                    cand = _name(a)
                    others = {clusters[x]["name"].lower() for x in range(len(clusters)) if x != a}
                    if name_collides(cand, others):
                        pre_, tw_, seen_ = parts[a]
                        parts[a] = (pre_, tw_[:-1], seen_ - set(words(opp)))
                        continue
                    clusters[a]["name"] = cand
                    changed = True
                    break
        if not changed:
            break
    # ...and when that side doesn't measure or fit, any trait that tells it from that
    # sibling (pairwise), so no two siblings differ by one missing word ("bright-sustained"
    # beside "bright-swelling-sustained")
    for a in range(len(clusters)):
        for b in range(len(clusters)):
            if a == b or re.search(r"-\d+$", clusters[a]["name"]):
                continue
            if not twin_names(clusters[a]["name"], clusters[b]["name"]):
                continue
            wa, wb = set(clusters[a]["name"].split("-")), set(clusters[b]["name"].split("-"))
            if not wa < wb:
                continue
            others = {clusters[x]["name"].lower() for x in range(len(clusters)) if x != a}
            for t in pairwise_traits(clusters[a], clusters[b], dims, blocked[a], banned[a], parts[a][2]):
                if set(words(t)) & wb or not _fits(a, t):
                    continue
                _add(a, t)
                cand = _name(a)
                if name_collides(cand, others) or any(twin_names(cand, o) for o in others):
                    pre_, tw_, seen_ = parts[a]
                    parts[a] = (pre_, tw_[:-1], seen_ - set(words(t)))
                    continue
                clusters[a]["name"] = cand
                break
    # a name that ends in half of a hyphenated trait ("open-long" from the phrase "long hat",
    # with "long-tail" a trait here) says the whole trait instead, as verify requires
    names = {c["name"].lower() for c in clusters}
    for c in clusters:
        whole_nm = whole_trait_ending(c["name"], dims)
        if whole_nm != c["name"] and whole_nm.lower() not in names:
            names.discard(c["name"].lower())
            names.add(whole_nm.lower())
            c["name"] = whole_nm
    if cap is not None:            # a device's path limit: every name within FAMILY_NAME_MAX
        names = set()
        for c in clusters:
            c["name"] = fit_name(c["name"], cap, names, dims)
            names.add(c["name"].lower())
    return [c["name"] for c in clusters]


def family_cap() -> int | None:
    """FAMILY_NAME_MAX when a device's path limit (the devices knob) set it under what a
    name reaches on its own (NAME_MAX_CHARS + 4: a trait added to tell siblings apart), else
    None, and names are built exactly as without a device."""
    from .curate_config import FAMILY_NAME_MAX
    return FAMILY_NAME_MAX if FAMILY_NAME_MAX < NAME_MAX_CHARS + 4 else None


def fit_name(nm: str, cap: int, used: set, dims=()) -> str:
    """A family name within cap characters and not in `used` (lowercase): whole units
    dropped from the end (the lead and the first word stay; never a half of a hyphenated
    trait left at the end), cut in the middle if one unit is still too long, then -2, -3 ...
    within the cap for a name its siblings already have."""
    toks = [t for t in nm.split("-") if t]
    while len("-".join(toks)) > cap and len(toks) > 1:
        toks.pop()
        while len(toks) > 1 and whole_trait_ending("-".join(toks), dims) != "-".join(toks):
            toks.pop()                     # "open-long" of "long-tail": drop the half too
    out = "-".join(toks) or nm
    if len(out) > cap:
        from ..devices.exporter import _truncate_middle
        out = re.sub("-+", "-", _truncate_middle(out, cap).replace("~", "-")).strip("-")
    if out.lower() not in used:
        return out
    for i in range(2, 1000):
        sfx = f"-{i}"
        cand = out[:cap - len(sfx)].rstrip("-") + sfx
        if cand.lower() not in used:
            return cand
    return out


def whole_trait_ending(nm, dims):
    """`nm` with a trailing fragment of a hyphenated trait ("long" of "long-tail") made the
    whole trait; unchanged when it ends in a whole trait or the fragment is ambiguous."""
    whole = {w for _, pair in dims for w in pair if w}
    toks = nm.split("-")
    last = toks[-1]
    if last in whole or any(nm.endswith(w) for w in whole):
        return nm
    owners = [w for w in whole if "-" in w and last in w.split("-")]
    if len(owners) != 1:
        return nm
    head = [t for t in toks[:-1] if t not in owners[0].split("-")]
    return "-".join(head + owners[0].split("-"))
