"""Per-category configuration for `fourier build`.

Each category is either a one-shot type (filtered by Sononym OneShot class +
category substring) or the loop-derived BREAKS bucket (Loop class gated to
breakbeat character by a contrastive CLAP anchor). The CLAP phrase vocabularies
are the text anchors used to give each cluster an evocative, distinctive name.
"""
from __future__ import annotations

import re as _re

from ..settings import for_module as _for_module

from .. import taxonomy as _taxonomy
_tunable = _for_module("curate_config")   # overridable: fourier/settings.py, config/tunables.yaml

# Categories a library overlay adds, {NAME: entry} (class library; usually an overlay's
# add_categories table, fourier/layers.py; docs/curation.md, "Adding a category"). The
# curation modules read the taxonomy with them merged in: taxonomy_view.
ADDED_CATEGORIES = _tunable("ADDED_CATEGORIES", {})   # library
taxonomy_view = _tax = _taxonomy.with_added(ADDED_CATEGORIES)

# Library rules (class "library" in config/tunables.yaml) match pack names and folders
# of one person's library, so the code ships them empty: _NEVER matches nothing, and
# the tuples, sets and dicts are empty. A user's overlay sets them (config layers,
# fourier/layers.py).
_NEVER = r"(?!)"
# Other names a library's packs use for a machine the code knows ({label: [regex, ...]}), added
# to the machine's patterns (DRUM_MACHINES, SYNTH_MACHINES, the acid phrase role and the
# genre word backing "303") after its own name and before its model codes.
MACHINE_ALIASES = _tunable("MACHINE_ALIASES", {})   # library

# Acoustic/orchestral multisample instrument libraries. Each such pack is
# capped to INSTRUMENT_CAP CLAP-spread samples during selection, so chromatic
# note-by-note multisamples (pianos, strings, winds and brass) don't flood the
# melodic categories. Matched on the provenance pack name (_pack_of). Drum and
# breakbeat packs never match, so they pass through untouched.
INSTRUMENT_PACKS = _tunable("INSTRUMENT_PACKS", _re.compile(_NEVER, _re.I))   # library
INSTRUMENT_CAP = _tunable("INSTRUMENT_CAP", 20)
# An instrument pack's folder label comes from the instrument its name says (curate.
# _instrument_label: pianos, orchestral sections); an overlay adds its own pack words.
INSTRUMENT_LABELS_EXTRA = _tunable("INSTRUMENT_LABELS_EXTRA", [])   # library: [(pack substring, label)]

# Pack demo/preview/jam files, identified by path. Kept out of every category but the
# instrument ones (they are full/busy, not clean hits).
DEMO_PATH = _tunable("DEMO_PATH", _re.compile(
    r"(ableton folder info/previews|/demos?/|[ _-]demo[s]?[ _.)\-]|\bpreview\b|"
    r"full[ _-]?(mix|track|song)|construction[ _-]?kit)", _re.I))

# Cap any single source folder (a vendor's pack folder) to this share of a
# category, so no one vendor's aesthetic dominates.
VENDOR_MAX_SHARE = _tunable("VENDOR_MAX_SHARE", 0.40)  # per-vendor cap per category
# Which folder of a library path is a sample's vendor for that cap (packs/vendors.py, the
# `vendors` knob): "first-folder" (a vendor/pack/... library) or "auto" (each library
# folder's layout decides: vendor/pack folders, folders by sound type, an umbrella folder
# of packs, or a flat folder of files).
VENDORS = _tunable("VENDORS", "first-folder")
# Favored sources get a selection nudge (curate.py FAVOR_WEIGHT); matched on the path.
FAVORED_SOURCES = _tunable("FAVORED_SOURCES", _re.compile(_NEVER, _re.I))   # library

# Instrument multisample libraries routed into their own PIANO / ACOUSTIC
# categories and OUT of the general categories, so the two never mix.
PIANO_PACKS = _tunable("PIANO_PACKS", _re.compile(_NEVER, _re.I))   # library
ORCH_PACKS = _tunable("ORCH_PACKS", _re.compile(_NEVER, _re.I))     # library
# ...and the acoustic packs of struck objects, whose files take ACOUSTIC's mallet band when
# nothing else in their name or tags gives one (rules._acoustic_band)
ACOUSTIC_MALLET_PACKS = _tunable("ACOUSTIC_MALLET_PACKS", _re.compile(_NEVER, _re.I))   # library
# Ableton's catch-all instrument tags. Alone they admit a file to ACOUSTIC only when its
# name says acoustic instrument (ACOUSTIC_NAME_RE) or it comes from an instrument pack:
# "Misc Plucked" also covers synth patches, dub chord chops and resonator hits. They
# also lose to the filename when a folder is named.
GENERIC_ACOUSTIC_TAGS = _tunable("GENERIC_ACOUSTIC_TAGS", ("Misc Plucked", "Misc Strings", "Misc Woodwind", "Misc Mallets",
                         "Bell Chromatic", "Chime"))
ACOUSTIC_NAME_RE = _tunable("ACOUSTIC_NAME_RE", _re.compile(
    r"guitar|zither|(?<![a-z])lyre|(?<![a-z])harp(?!si)|banjo|sitar|koto|mandolin|ukulele|cigar ?box"
    r"|bouzouki|(?<![a-z])oud(?![a-z])|balalaika|dulcimer|(?<![a-z])lute(?![a-z])|violin|viola|cello"
    r"|contrabass|double ?bass|upright ?bass|acoustic ?bass|string|pizz|bowed|(?<![a-z])arco(?![a-z])"
    r"|flute|clarinet|oboe|bassoon|(?<![a-z])sax|recorder|harmonica|accordion|melodica|ocarina|kazoo"
    r"|pan ?pipe|trumpet|trombone|tuba|(?<![a-z])horns?(?![a-z])|brass|woodwind"
    r"|(?<![a-z])timpani(?![a-z])|(?<![a-z])gong(?![a-z])|erhu|(?<![a-z])kora(?![a-z])"
    r"|xylo|marimba|vibraphone|(?<![a-z])vibes(?![a-z])|glock|kalimba|thumb ?piano|mbira|music ?box"
    r"|toy ?piano|steel ?(?:drum|pan)|celest|gamelan|chimes?(?![a-z])|tubular|church ?bell|hand ?bell"
    r"|singing ?bowl|tibetan|bell ?tree|handpan|hang ?drum", _re.I))
# Tuned percussion named in the filename (MALLETS was folded into ACOUSTIC): like a
# plucked string, a one-shot so named is ACOUSTIC's. A plain "bell" doesn't count ("FM
# Bell", "Bell Lead" are synth patches); a named bell does (church, hand, tubular).
MALLET_NAME_RE = _tunable("MALLET_NAME_RE", _re.compile(
    r"xylo|marimba|(?<![a-z])arimba|vibraphone|(?<![a-z])vibes(?![a-z])|glock|kalimba|thumb ?piano"
    r"|mbira|music ?box|toy ?piano|steel ?(?:drum|pan)|celest|gamelan|chimes?(?![a-z])|tubular"
    r"|church ?bell|hand ?bell|singing ?bowl|tibetan|bell ?tree|handpan|hang ?drum", _re.I))
# Synth and drum-machine packs (emulations included) whose patches carry instrument names
# that auto-tags take for real acoustic instruments: kept out of ACOUSTIC (no synth
# imitations) and homed normally instead; packs of sampled recordings stay eligible.
SYNTH_EMULATION_PACKS = _tunable("SYNTH_EMULATION_PACKS", ())   # library: synth and drum-machine pack names
ORCH_PACK_EXCLUDE = _tunable("ORCH_PACK_EXCLUDE", _re.compile(      # library
    r"^(?:" + "|".join(_re.escape(p) for p in SYNTH_EMULATION_PACKS) + r")$" if SYNTH_EMULATION_PACKS
    else _NEVER, _re.I))
# A keyboard named in the filename beats a conflicting Ableton instrument tag: an
# auto-tag can call a piano sample a guitar, which would push it out of PIANO and into
# ACOUSTIC.
KEYBOARD_NAME_RE = _tunable("KEYBOARD_NAME_RE", _re.compile(r"piano|rhodes|wurli|clavinet|\bclav\b|harpsichord", _re.I))
# Plucked strings named in the filename (guitars, zithers, a lyre): the classifiers can
# call them synth plucks or mallets. A one-shot so named is ACOUSTIC's and homes nowhere
# else; a bass guitar stays a bass.
PLUCKED_NAME_RE = _tunable("PLUCKED_NAME_RE", _re.compile(
    r"(?<!bass )(?<!bass)guitar|zither|(?<![a-z])lyre|(?<![a-z])harp(?!si)|banjo|sitar|koto"
    r"|mandolin|ukulele|cigar ?box|bouzouki|(?<![a-z])oud(?![a-z])|balalaika|dulcimer|(?<![a-z])lute(?![a-z])"
    # a nylon / steel string is a guitar string, not a bowed or bass string
    r"|nylon[ _-]?(?:string|gtr)|(?:string|gtr)[ _-]?nylon|steel[ _-]?string",
    _re.I))

# Loop tempo: DRUMLOOPS families are clustered by sound WITHIN tempo bands [lo, hi),
# so a folder's BPM label is true for everything in it. Tempos come from
# curate._resolve_tempo (filename > Sononym > librosa, each accepted only if the loop is
# then a whole number of bars). A band with fewer than TEMPO_BAND_MIN candidates merges
# into its smaller neighbor. DRUMLOOPS leaves out loops with no confirmable tempo
# (require_tempo); only a pinned or name-overridden one still lands in a "freetempo"
# folder (PHRASES has no require_tempo, so its tempo-less loops do too).
# Half and double time are interchangeable in performance, so loop tempos are folded
# into one octave, TEMPO_FOLD = [90, 180), before banding and labeling: an 82 BPM amen
# groups with 164 BPM jungle, 86 half-time dnb with 172. The window keeps house /
# techno / breaks (120-140) and dnb (170-176) at their native tempos.
TEMPO_FOLD = _tunable("TEMPO_FOLD", (90.0, 180.0))
# The tempo knob folds into its range's top octave and, for a range narrower than two
# octaves' worth (a low edge above half the high one: house 118-140, hip hop 70-100), never
# out of the range: a fold that would land outside (lo, hi) leaves the loop at its own tempo
# (fourier/knobs.fold_window). None: no such limit (a range that holds its whole top octave,
# as 85-180 does). TEMPO_FOLDING off (the fold knob): no loop is folded at all.
TEMPO_FOLD_RANGE = _tunable("TEMPO_FOLD_RANGE", None)
TEMPO_FOLDING = _tunable("TEMPO_FOLDING", True)
TEMPO_BANDS = _tunable("TEMPO_BANDS", (60, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140,
               145, 150, 155, 160, 165, 170, 175, 180, 190, 201))
# (180-200: a tempo the filename states isn't folded, so 180-193 dnb loops stay fast)
# (5-BPM bands throughout, so a folder's tempo label fits every loop in it, and 60-90 for
# slow loops that aren't folded up, below.)
# A slow loop from a non-breakbeat style isn't folded to double time: a 78 BPM hip-hop or
# reggae groove played at 156 is a half-time feel, not a break, and would crowd the jungle /
# dnb folders. A breakbeat word in the path still folds it.
NO_FOLD_RE = _tunable("NO_FOLD_RE", _re.compile(
    r"hip ?-?hop|trip ?-?hop|down ?-?tempo|lo ?-?fi|boom ?-?bap|nu ?-?jazz|chill ?-?out|reggae|(?<![a-z])dub(?!step)", _re.I))
FOLD_ANYWAY_RE = _tunable("FOLD_ANYWAY_RE", _re.compile(
    r"jungle|dnb|drum ?(?:&|and|n|'n') ?bass|break|rave|amen|hardcore", _re.I))
# Classic breaks: the loops a user marks as classic by name (CLASSIC_BREAK_RE) or folder
# (CLASSIC_BREAK_PATH_RE) get their own DRUMLOOPS band and folder, whatever the tempo, and
# all of them are kept. The code knows only the generic words; a user's overlay names the
# rest (class library). A name says nothing about where a loop came from or its license.
CLASSIC_BREAK_RE = _tunable("CLASSIC_BREAK_RE", _re.compile(
    r"(?<![a-z])amen(?![a-z])|classic ?breaks?", _re.I))   # library: the loops a user calls classic
CLASSIC_BREAK_PATH_RE = _tunable("CLASSIC_BREAK_PATH_RE", _re.compile(_NEVER, _re.I))   # library: folders of classic breaks
TEMPO_BAND_MIN = _tunable("TEMPO_BAND_MIN", 40)      # candidates, so a band's folder isn't a sliver
# Tops (hat / percussion loops) are usually a small pool next to the full breaks, so the
# break bands would leave their folders near empty. They get wide bands that
# match how they're played: under 120 (hip-hop, downtempo), 120-124 house, 125-129
# techno, 130-149 faster techno / trap, 150+ jungle and dnb.
TOPS_TEMPO_BANDS = _tunable("TOPS_TEMPO_BANDS", (60, 90, 120, 125, 130, 150, 201))
INST_ROUTED_PACKS = _tunable("INST_ROUTED_PACKS", _re.compile(_NEVER, _re.I))   # library
INSTR_PHRASES = _tunable("INSTR_PHRASES", _tax.prompts("ACOUSTIC"))
INSTR_DIMS = _tunable("INSTR_DIMS", [("br", ("dark", "bright")), ("noi", ("clean", "airy")), ("har", ("noisy", "tonal"))])  # (was "percussive": it measured noise)

# PIANO is sourced library-wide (piano packs often hold phrases, not loose
# per-note samples): single-note, pitched samples whose CLAP embedding is
# close to these piano/e-piano anchors and far from the anti set. PIANO_POOL_LABELS
# bounds the candidate pool to keys-ish one-shots (plus all Ableton).
PIANO_ANCHOR = _tunable("PIANO_ANCHOR", _tax.prompts("PIANO"))
PIANO_ANTI = _tunable("PIANO_ANTI", _tax.anti("PIANO"))
PIANO_POOL_LABELS = _tunable("PIANO_POOL_LABELS", _tax.piano_pool_labels())

# --- CLAP phrase vocabularies (evocative anchors, z-scored across clusters) ---

KICK_PHRASES = _tunable("KICK_PHRASES", _tax.prompts("KICKS"))

SNARE_PHRASES = _tunable("SNARE_PHRASES", _tax.prompts("SNARES"))

HAT_PHRASES = _tunable("HAT_PHRASES", _tax.prompts("HATS"))

SUB_PHRASES = _tunable("SUB_PHRASES", _tax.prompts("SUB"))

SYNTH_PHRASES = _tunable("SYNTH_PHRASES", _tax.prompts("SYNTH"))

CLAP_PHRASES = _tunable("CLAP_PHRASES", _tax.prompts("CLAPS"))

TOM_PHRASES = _tunable("TOM_PHRASES", _tax.prompts("TOMS"))

PERC_PHRASES = _tunable("PERC_PHRASES", _tax.prompts("PERC"))

CYMBAL_PHRASES = _tunable("CYMBAL_PHRASES", _tax.prompts("CYMBALS"))

# a library's own FX prompts (class library), put where the taxonomy's FX prompts hold a
# {library: FX_PROMPTS_EXTRA} entry
FX_PROMPTS_EXTRA = _tunable("FX_PROMPTS_EXTRA", [])   # library
FX_PHRASES = _tunable("FX_PHRASES", _tax.prompts("FX", {"FX_PROMPTS_EXTRA": FX_PROMPTS_EXTRA}))

VOX_PHRASES = _tunable("VOX_PHRASES", _tax.prompts("VOX"))

# a library's own classic-break prompts (class library), put where the taxonomy's
# DRUMLOOPS prompts hold a {library: CLASSIC_BREAK_PROMPTS} entry
CLASSIC_BREAK_PROMPTS = _tunable("CLASSIC_BREAK_PROMPTS", [])   # library
BREAK_PHRASES = _tunable("BREAK_PHRASES", _tax.prompts("DRUMLOOPS", {"CLASSIC_BREAK_PROMPTS": CLASSIC_BREAK_PROMPTS}))

# anti-anchor: loops that are NOT drum breaks (melodic/bass/fx), used contrastively
BREAK_ANTI_PHRASES = _tunable("BREAK_ANTI_PHRASES", _tax.anti("DRUMLOOPS"))

# DRUMLOOPS tops folders are named only from these (naming-only), full breaks never
TOPS_PHRASES = _tunable("TOPS_PHRASES", ["hi-hat loop", "shaker loop", "ride cymbal loop", "conga and bongo loop",
                "tambourine loop", "percussion top loop", "909 hi-hat loop", "open hat groove"])

# FX types by share of the FX budget: transitions first. A band
# whose pool runs short spills to the others (curate._allocate_banded).
FX_BAND_SHARE = _tunable("FX_BAND_SHARE", {"riser": 0.15, "downlifter": 0.06, "sweep": 0.08, "impact": 0.19, "noise": 0.14,
                 "synth": 0.14, "foley": 0.10, "ambience": 0.05, "whoosh": 0.05, "scratch": 0.04})
# Risers and downlifters by name get their own FX bands, split from the sweeps, so each
# transition has a folder of its own (a riser is the transition a set reaches for most).
# Riser 0.15, downlifter 0.06, sweep 0.08 (FX_BAND_SHARE). A name saying both, or neither,
# leaves the library's FX type.
FX_RISE_NAME_RE = _tunable("FX_RISE_NAME_RE", _re.compile(
    r"(?<![a-z])(?:up ?lift(?:er)?s?|risers?|rise|rising|fx ?up|build ?ups?|sweep ?up|up ?sweep)(?![a-z])", _re.I))
FX_FALL_NAME_RE = _tunable("FX_FALL_NAME_RE", _re.compile(
    r"(?<![a-z])(?:down ?lift(?:er)?s?|falls?|falling|fx ?down|sweep ?down|down ?sweep)(?![a-z])", _re.I))
# a scratch-band folder is named from these (naming only; the band comes from the name)
FX_SCRATCH_PHRASES = _tunable("FX_SCRATCH_PHRASES", ["turntable scratch", "baby scratch", "chirp scratch", "transformer scratch",
                      "record stop", "vinyl rewind", "vinyl spinback", "echo scratch", "scribble scratch",
                      "hip hop scratch"])

# Naming-only vocabulary: extra CLAP phrases a family may be NAMED by, never used for
# gating or anchoring (the per-category `phrases` lists above also anchor selection, so
# they must not grow for naming's sake). A second tier: a family is named by the
# category's own phrases when one fits (what a sound IS beats a genre guess), by these
# otherwise. The default (preset breaks-acid) leans toward breaks, jungle/dnb, acid,
# house, techno, rave.
# Words a category's folders may not be named with, though its anchor phrases use them:
# STABS holds the named stabs, so a SYNTH folder called "...-stab-..." would point to the
# wrong place.
# DRUMLOOPS / PHRASES: "fast" / "slow" say what the tempo lead already does, or contradict
# it (a "slow" folder in a wide tempo range)
NAME_EXCLUDE_WORDS = _tunable("NAME_EXCLUDE_WORDS", {"SYNTH": {"stab", "stabs"}, "DRUMLOOPS": {"fast"}, "PHRASES": {"slow", "fast"}})
NAME_PHRASES = _tunable("NAME_PHRASES", {
    "KICKS": ["techno rumble kick", "tight 909 kick", "breakbeat kick", "hardcore rave kick"],
    "SNARES": ["breakbeat snare", "rave snare", "garage snare"],
    "HATS": ["909 open hat", "808 open hat", "garage shuffle hat"],
    "CYMBALS": ["909 ride", "909 crash"],
    "SUB": ["303 acid bass", "jungle sub bass", "rave bass stab"],
    "SYNTH": ["rave hoover lead", "detroit techno lead", "acid arp", "rave piano key", "m1 organ key",
              "fm bell tone", "fm marimba mallet", "glassy digital bell"],
    "STABS": ["rave piano chord stab", "m1 organ house stab", "detroit techno chord stab",
              "dub techno chord stab"],
    "PADS": ["dub techno pad", "detroit strings pad", "rave pad"],
    "FX": ["air horn", "dub siren", "rewind spinback", "impact hit"] + FX_SCRATCH_PHRASES,
    "VOX": ["diva vocal chop", "ragga mc shout", "rewind selector shout", "rave vocal stab"],
    "DRUMLOOPS": ["rave breakbeat", "hardcore breakbeat", "garage shuffle loop"] + TOPS_PHRASES,
    # PIANO's own phrases name single notes; its chord folders need chord words
    "PIANO": ["grand piano chord", "house piano chord", "rhodes chord stab", "electric piano chord",
              "wurlitzer chord", "clavinet chord stab", "organ note", "drawbar organ", "organ chord"],
})

# WAVES: single-cycle waveforms and wavetable frames, gathered from wherever they sit
# (otherwise they land in the drum and pad categories as 20 ms "clicks"). Looped, they
# are oscillators: the M8 sampler's OSC / FWDLOOP play modes (manual p. 61) and the
# Digitakt II's FORWARD LOOP (manual p. 70). A wave is a file under a wavetable /
# single-cycle folder (or one named for a single-cycle waveform collection,
# WAVE_PATH_WORDS) no longer than WAVE_MAX_DUR; it may live nowhere else.
# Matched against the FOLDERS only (curate._is_wave), not the filename: a bare "cycle"
# in a pack title doesn't make its drums waves, and a one-shot named "... Wavetable.wav"
# isn't one. Wave sources sit in wave folders.
WAVE_PATH_RE = _tunable("WAVE_PATH_RE", _re.compile(r"wave ?tables?|single[ _-]?cycles?", _re.I))
# ...and folders named for a particular collection of single-cycle waves (whole words)
WAVE_PATH_WORDS = _tunable("WAVE_PATH_WORDS", ())   # library
if WAVE_PATH_WORDS:
    WAVE_PATH_RE = _re.compile(WAVE_PATH_RE.pattern + "".join(
        r"|(?<![a-z])" + _re.escape(w.lower()) + r"(?![a-z])" for w in WAVE_PATH_WORDS), WAVE_PATH_RE.flags)
# A drum kit shipped inside a wavetable pack isn't waves: a "Drumkit/" folder holds drum
# hits, which home like any other drum.
WAVE_NOT_PATH_RE = _tunable("WAVE_NOT_PATH_RE", _re.compile(r"(?:^|/)drum ?kits?(?:/|$)", _re.I))
WAVE_MAX_DUR = _tunable("WAVE_MAX_DUR", 1.0)
WAVE_MIN_SAMPLES = _tunable("WAVE_MIN_SAMPLES", 256)          # shorter cycles are too coarse to use
WAVE_SINGLE_MAX_SAMPLES = _tunable("WAVE_SINGLE_MAX_SAMPLES", 4097)     # up to one 4096-sample frame: "cycle"; longer: "table"
WAVE_PHRASES = _tunable("WAVE_PHRASES", _tax.prompts("WAVES"))
WAVE_DIMS = _tunable("WAVE_DIMS", [("br", ("dark", "bright")), ("noi", ("smooth", "gritty")), ("har", ("inharmonic", "pure"))])

# Length caps (s) per category: a very long "bass", "pad" or ride costs sample RAM on
# the device (the Digitakt II has 400 MB per project) and isn't the one-shot the folder
# promises. PIANO/ACOUSTIC keep their dur_max (30 s).
DUR_CAP = _tunable("DUR_CAP", {
    "KICKS": 4.0, "SNARES": 3.0, "CLAPS": 3.0, "HATS": 4.0, "TOMS": 5.0, "PERC": 4.0,
    "CYMBALS": 12.0, "SUB": 8.0, "SYNTH": 10.0, "STABS": 4.0, "PADS": 20.0,
    "VOX": 20.0, "BLIPS": 3.0, "FX": 30.0,
})
# A bpm in the name marks a loop or phrase; keep those out of one-shot folders. FX is
# exempt (tempo-named risers and sweeps are one-shots meant to line up with a bar).
BPM_NAME_RE = _tunable("BPM_NAME_RE", _re.compile(
    r"\d{2,3}\s*[_-]?\s*bpm|bpm\s*[_-]?\s*\d{2,3}"
    # a tempo then a key ("Lead_174_Amin"): a phrase named like a loop
    r"|(?<![0-9a-z])(?:[6-9]\d|1\d\d|200)[ _-]+[a-g](?:#|b)?(?:maj|min|m)(?![a-z])"
    # ...but not a multisampled synth note: a MIDI note number, a patch name, a note ("60
    # Warm Pad C3", "72 Soft Lead G#2_0001") is a one-shot
    r"(?!.*[ _-][a-g](?:#|b|s)?-?\d(?:[ _-]\d+)?(?:\.\w+)?$)", _re.I))
BPM_NAME_GUARD = _tunable("BPM_NAME_GUARD", {"KICKS", "SNARES", "CLAPS", "HATS", "TOMS", "PERC", "CYMBALS", "SUB",
                  "SYNTH", "STABS", "PADS", "VOX", "BLIPS", "PIANO", "ACOUSTIC"})

# --- PHRASES: musical loops, kept apart from the one-shots ---
# A phrase is a bassline, chord progression, riff, arpeggio or string swell, ready to play
# at a tempo: what a producer made as a loop. A file is one (curate._is_phrase) when:
#   - the producer marks it: a tempo in the name (BPM_NAME_RE), or a loop folder
#     (PHRASE_LOOP_DIR_RE: "Bass Loops", "Music Loops", "LPS", "Grooves")
#     not narrowed below by a one-shot folder ("Loops/Chords/..."), and Sononym
#     or Ableton class it as a loop (a tempo in the name is enough on its own);
#   - it is a whole number of bars at that tempo (curate._resolve_tempo), 1-20 s long;
#   - it is musical: no drum or vocal tag from Ableton, not FX alone, not a Sononym
#     drum / FX label alone, and no drum, FX, vocal or sample-chain word in its name, the
#     folder it sits in or a loop folder above it ("vox loops/100 bpm/..."; PHRASE_NOT_RE;
#     genre names like "Drum & Bass" don't count, nor does a pack named for grooves). Vocal phrases stay VOX's, drum loops
#     DRUMLOOPS', risers FX's; a chain of notes (a "Combined" multisample) is never one.
# A phrase homes in PHRASES ahead of every one-shot vote and is kept out of every other
# category (a Keep still wins), so one-shot folders hold one-shots only. Sample chains
# are not phrases: slicing them is work Fourier doesn't ask for.
PHRASES_CATEGORY = _tunable("PHRASES_CATEGORY", _tax.home("phrases"))
PHRASE_LOOP_DIR_RE = _tunable("PHRASE_LOOP_DIR_RE", _re.compile(
    r"(?<![a-z])(?:loops?|lps|phrases?|riffs?|basslines?|grooves?|patterns?|progressions?|sequences?)(?![a-z])",
    _re.I))
# a folder of single sounds below (or beside) a loop folder: "Loops/Chords/..."
PHRASE_ONESHOT_DIR_RE = _tunable("PHRASE_ONESHOT_DIR_RE", _re.compile(
    r"one[ _-]?shots?|(?<![a-z])(?:hits?|stabs?|chords?|notes?|single|multi[ _-]?samples?|kits?)(?![a-z])",
    _re.I))
# ...unless the folder names phrases itself ("Chord Progressions")
PHRASE_DIR_STRONG_RE = _tunable("PHRASE_DIR_STRONG_RE", _re.compile(r"(?<![a-z])(?:progressions?|phrases?|riffs?|basslines?)(?![a-z])", _re.I))
PHRASE_NOT_RE = _tunable("PHRASE_NOT_RE", _re.compile(
    # drums (perc loops named "Perc", "Percussion", "Hats" and the like)
    r"drum|perc(?:ussion|s)?(?![a-z])|(?<![a-z])(?:beats?|breaks?|breakbeats?|tops?|hats?|hi-?hats?|kicks?"
    r"|snares?|claps?|rides?|shakers?|congas?|bongos?|toms?|cymbals?|fills?)(?![a-z])"
    # FX, textures and field recordings (PADS and FX hold those)
    r"|(?<![a-z])(?:s?fx|risers?|uplifters?|downlifters?|sweeps?|impacts?|noises?|foley|whoosh\w*|atmos\w*"
    r"|ambien\w*|textures?|drones?|soundscapes?|crackle\w*)(?![a-z])"
    # voices (VOX)
    r"|(?<![a-z])(?:vox|vocals?|voices?|acapellas?|a ?cappella|chants?|choirs?|speech|spoken|rap|shouts?"
    r"|adlibs?|sung|singing)(?![a-z])"
    # sample chains and multisample sets
    r"|combined|multi[ _-]?samples?|(?<![a-z])chains?(?![a-z])|velocity|round ?robin",
    _re.I))
# genre names in folders aren't drums ("Drum & Bass/Bass Loops"): removed before PHRASE_NOT_RE
PHRASE_GENRE_RE = _tunable("PHRASE_GENRE_RE", _re.compile(r"drum ?(?:&|and|n|'n'|\u2019n\u2019)? ?bass|(?<![a-z])d ?& ?b(?![a-z])", _re.I))
PHRASE_DUR = _tunable("PHRASE_DUR", (1.0, 20.0))          # s; 20 s is 8 bars at 96 BPM, RAM-friendly on the Digitakt II
# folded into TEMPO_FOLD, then banded: 90-119 (hip-hop, downtempo, breaks), 120-139 (house,
# techno, acid), 140-179 (dubstep, jungle, dnb, and 85-89 hip-hop at double time)
PHRASE_TEMPO_BANDS = _tunable("PHRASE_TEMPO_BANDS", (60, 90, 120, 140, 201))
# roles (the folder's lead word), in the order the name decides them: a bass guitar is a
# bassline, a string loop is played live, a piano riff is chords, the rest is a lead line
PHRASE_BASS_RE = _tunable("PHRASE_BASS_RE", _re.compile(
    r"bass|(?<![a-z])sub(?![a-z])|303|acid|(?<![a-z])reese|wobble", _re.I))
PHRASE_BASS_TAGS = _tunable("PHRASE_BASS_TAGS", ("Synth Bass", "808 Bass", "Electric Bass", "Upright Bass", "Acoustic Bass",
                    "Double Bass"))
PHRASE_CHORDS_RE = _tunable("PHRASE_CHORDS_RE", _re.compile(
    r"chords?|(?<![a-z])keys?(?![a-z])|piano|rhodes|organ|wurli|clav|e-?piano|(?<![a-z])pads?(?![a-z])"
    r"|progressions?|(?<![a-z])prog(?![a-z])|stabs?|(?<![a-z])m1(?![a-z])", _re.I))
PHRASE_CHORDS_TAGS = _tunable("PHRASE_CHORDS_TAGS", ("Piano", "Electric Piano", "Organ", "Clav", "Synth Keys", "Pad", "Atmosphere",
                      "Synth Strings", "Misc Keys"))
# a 303 / acid line is its own role, not a bassline; it's checked before bass
# (case-aware so "SynthAcid" counts and "Placid" doesn't)
PHRASE_ACID_RE = _tunable("PHRASE_ACID_RE", _re.compile("|".join([
    r"303", r"(?<![A-Za-z])(?i:acid)", r"(?<=[a-z])Acid",
    *(f"(?i:{a})" for a in MACHINE_ALIASES.get("303", ())), r"(?<![A-Za-z])(?i:tb-?3)"])))
PHRASE_ROLES = _tunable("PHRASE_ROLES", _tax.phrase_roles())
PHRASE_ROLE_SHARE = _tunable("PHRASE_ROLE_SHARE", {"acid": 0.10, "bass": 0.15, "chords": 0.30, "lead": 0.25, "live": 0.20})
# CLAP vocabulary per role: the union anchors the gate (musical phrase vs drum break), and
# a folder is named only from its own role's phrases (the role word already leads)
PHRASE_ROLE_PHRASES = _tunable("PHRASE_ROLE_PHRASES", {
    "acid": ["303 bassline", "squelchy resonant riff", "rolling sixteenth sequence", "303 lead riff"],
    "bass": ["reese bass riff", "deep house bassline", "funk bass groove",
             "dub reggae bassline", "wobble bass riff", "rolling sub bassline", "electro bassline",
             "slap bass groove", "techno bass sequence"],
    "chords": ["deep house chords", "dub techno chords", "rave piano riff", "rhodes chord progression",
               "organ groove", "jazzy keys loop", "detroit chord progression", "lush pad progression",
               "soul piano loop", "chord stab sequence"],
    "lead": ["synth arpeggio", "acid lead riff", "rave hoover riff", "melodic synth lead",
             "plucked synth sequence", "chiptune melody", "detroit synth melody", "trance lead riff",
             "bleep sequence", "modular sequence"],
    "live": ["funk guitar riff", "string swell", "pizzicato strings", "brass section riff",
             "flute melody", "reggae guitar skank", "blues guitar riff", "orchestral string phrase",
             "sitar melody", "jazz horn riff"],
})
PHRASE_PHRASES = [p for ps in PHRASE_ROLE_PHRASES.values() for p in ps]
# the instrument a phrase's filename names; when most of a folder names one, it names the
# folder ("live-090-120bpm-guitar"), as ACOUSTIC's folders are named for their instrument
PHRASE_INSTRUMENTS = _tunable("PHRASE_INSTRUMENTS", [(_re.compile(p, _re.I), lab) for p, lab in (
    (r"guitar|gtr|(?<![a-z])git(?![a-z])", "guitar"), (r"cello", "cello"), (r"pizz", "pizzicato"),
    (r"violin|viola|string", "strings"), (r"brass|horns?(?![a-z])|trumpet|trombone", "brass"),
    (r"(?<![a-z])sax", "sax"), (r"flute", "flute"), (r"(?<![a-z])harp(?!si)", "harp"), (r"sitar", "sitar"),
    (r"rhodes|e-?piano|electric piano|wurli", "rhodes"), (r"piano", "piano"), (r"organ", "organ"),
    (r"(?<![a-z])clav", "clav"), (r"(?<![a-z])pads?(?![a-z])", "pad"),
    (r"(?<![a-z])arp(?:eggio|s)?(?![a-z])|(?-i:Arp)|arpeggiat", "arp"), (r"acid|303", "acid"), (r"reese", "reese"),
    (r"slap", "slap-bass"), (r"pluck", "pluck"), (r"bells?(?![a-z])", "bell"))])
PHRASE_DIMS = _tunable("PHRASE_DIMS", [("br", ("dark", "bright")), ("noi", ("clean", "gritty")), ("atk", ("plucky", "swelling"))])

# --- bands: sub-groups clustered on their own, so a folder is one or the other ---
# A category's `band` names its rule; the band word leads
# the folder name, and a folder never mixes bands. CLAP alone groups by timbre, which
# put open and closed hats, or impacts and foley, in one folder.
#   wave  WAVES: cycle (one frame) / table (several)
#   chord PIANO: note / chord
#   hat   HATS: closed / open (name first, then length)
#   fx    FX: the source library's own FX types
#   loop  DRUMLOOPS: full kits / tops (hat and percussion loops, for layering)
HAT_OPEN_MIN_S = _tunable("HAT_OPEN_MIN_S", 0.35)
HAT_OPEN_RE = _tunable("HAT_OPEN_RE", _re.compile(r"open|(?<![a-z])o\.?h(?:h)?(?![a-z])|sizzl|washy", _re.I))
HAT_CLOSED_RE = _tunable("HAT_CLOSED_RE", _re.compile(r"closed|(?<![a-z])c\.?h(?:h)?(?![a-z])|pedal|(?<![a-z])p\.?h(?![a-z])|tight", _re.I))
FX_BANDS = _tunable("FX_BANDS", [("Explosions & Shots", "impact"), ("Whooshes & Whips", "whoosh"),
            ("Sweeps & Lasers", "sweep"), ("Noise & Distortion", "noise"),
            ("Nature & Athmospheric", "ambience"), ("Cracks & Rustle", "foley")])
FX_AB_BANDS = _tunable("FX_AB_BANDS", {"Impact": "impact", "Sweep": "sweep", "Noise": "noise", "Field & Foley": "foley"})
# FX with no library FX type: pitched ones (Sononym says lead / bass / blip / stab / zap)
# are synth FX, alarms, zaps and beeps you can play chromatically, rather than a "misc"
# catch-all. They get their own `synth` band; the rest join the FX type whose phrases
# CLAP puts them closest to.
FX_SYNTH_SON = _tunable("FX_SYNTH_SON", ("Tone Leads", "Tone Bass", "Tone Blips", "Tone Stabs", "Perc Zaps"))
# A file Sononym calls a kit drum, with no FX label from either library beyond Ableton's
# catch-all "Sound FX", doesn't home in FX (else cymbals, snares and hats leak in).
FX_DRUM_SON = _tunable("FX_DRUM_SON", ("Perc Kicks", "Perc Snares", "Perc Claps", "Perc Hats & Shakers",
               "Perc Cymbal Crashes", "Perc Cymbal Rides", "Perc Toms"))
# phrases a family in each FX band may be named by (the band word already leads)
FX_BAND_PHRASES = _tunable("FX_BAND_PHRASES", {
    "impact": ["boom explosion", "sub drop impact", "gunshot", "distortion blast", "door slam",
               "glass shatter", "metal hit", "cinematic hit"],
    "whoosh": ["whoosh swipe", "whip crack", "air burst", "fast pass-by", "reverse swell"],
    "sweep": ["white noise sweep", "laser zap", "sci-fi beam", "tape stop", "reverse swell"],
    "riser": ["riser build-up", "white noise sweep", "reverse swell", "sci-fi beam"],
    "downlifter": ["downlifter fall", "white noise sweep", "tape stop"],
    "noise": ["radio static", "bitcrush noise", "glitch stutter", "vinyl crackle",
              "digital error beep", "electric zap", "distortion blast"],
    "ambience": ["rain and thunder", "wind gust", "ocean waves", "fire crackle", "water droplet",
                 "insect ambience", "dark drone", "engine rumble", "granular texture"],
    "foley": ["footstep foley", "metallic scrape", "chain rattle", "paper rustle", "wood crack",
              "mechanical whir", "door slam", "glass shatter"],
    "synth": ["alarm siren", "sci-fi beam", "laser zap", "digital error beep", "robot bleep",
              "arcade game sound", "ring modulated tone", "modem chirp", "synth drop", "police siren"],
})
LOOP_TOPS_RE = _tunable("LOOP_TOPS_RE", _re.compile(r"(?<![a-z])tops?(?![a-z])|hi-?hats?|(?<![a-z])hats?(?![a-z])|shaker"
                           r"|perc(ussion)?[ _-]*loop|conga|bongo|tambourine|(?<![a-z])ride(?![a-z])", _re.I))
LOOP_FULL_RE = _tunable("LOOP_FULL_RE", _re.compile(r"break|full|(?<![a-z])beat|drum ?loop|kit", _re.I))
LOOP_KIT_TAGS = _tunable("LOOP_KIT_TAGS", ("Kick", "Snare Hit"))

# --- category registry ---
# kind: "oneshot" | "loop"
# cat:  Sononym category substring (oneshot only)
# noise: filename regex of samples to reject (mislabeled neighbours)
# phrases: CLAP naming vocabulary

PAD_PHRASES = _tunable("PAD_PHRASES", _tax.prompts("PADS"))

STAB_PHRASES = _tunable("STAB_PHRASES", _tax.prompts("STABS"))

BLIP_PHRASES = _tunable("BLIP_PHRASES", _tax.prompts("BLIPS"))

ORCH_ABLETON_TAGS = _tunable("ORCH_ABLETON_TAGS", ["Violin", "Viola", "Cello", "Double Bass", "Upright Bass", "Acoustic Bass",
                     "String Ensemble", "Misc Strings", "Harp", "Sitar", "Misc Plucked",
                     "Flute", "Clarinet", "Oboe", "Bassoon", "English Horn", "Woodwind Ensemble",
                     "Sax", "Misc Woodwind",
                     "Trumpet", "Trombone", "Tuba", "French Horn", "Brass Ensemble",
                     "Timpani", "Gong", "Classical Guitar", "Acoustic Guitar", "Electric Guitar"])
# Tuned-percussion instrument words, for naming ACOUSTIC's mallet folders by instrument
# (curate._mallet_label), so a folder is named for what it holds. First match wins.
MALLET_INSTRUMENTS = _tunable("MALLET_INSTRUMENTS", [
    (r"toy ?piano", "toy-piano"), (r"music ?box", "music-box"), (r"xylo", "xylophone"),
    (r"marimba|(?<![a-z])arimba", "marimba"), (r"vibraphone|(?<![a-z])vibes?(?![a-z])|(?<![a-z])vibra(?![a-z])", "vibraphone"),
    (r"glock", "glockenspiel"), (r"kalimba|thumb ?piano|mbira", "kalimba"),
    (r"steel ?(drum|pan)", "steel-drum"), (r"celest", "celesta"), (r"gamelan", "gamelan"),
    (r"singing ?bowl|tibetan", "singing-bowl"), (r"handpan|hang ?drum", "handpan"),
    (r"chime", "chimes"), (r"triangle", "triangle"), (r"bell", "bell"), (r"mallet", "mallet"),
])
MALLET_INSTRUMENTS = [(_re.compile(p, _re.I), lab) for p, lab in MALLET_INSTRUMENTS]
# ACOUSTIC keeps keyboards out (PIANO's) and files whose name says synth, whatever
# Ableton's instrument tag says (an "FM_Bass_2" patch tagged as a string bass).
ACOUSTIC_NAME_EXCLUDE = _tunable("ACOUSTIC_NAME_EXCLUDE", _re.compile(
    KEYBOARD_NAME_RE.pattern + r"|(?<![a-z])(?:fm|synth\w*|saw(?:tooth)?|square|pwm|303|moog|juno)(?![a-z])",
    _re.I))
# tuned percussion: ACOUSTIC's mallet band (MALLETS was folded into ACOUSTIC)
MALLET_ABLETON_TAGS = _tunable("MALLET_ABLETON_TAGS", ["Xylophone", "Vibraphone", "Marimba", "Glockenspiel", "Misc Mallets",
                       "Bell Chromatic", "Chime"])
ACOUSTIC_ABLETON_TAGS = _tunable("ACOUSTIC_ABLETON_TAGS", ORCH_ABLETON_TAGS + MALLET_ABLETON_TAGS)
# ACOUSTIC bands (curate._acoustic_band): name first, then Ableton tag, then pack
ACOUSTIC_BAND_TAGS = _tunable("ACOUSTIC_BAND_TAGS", {
    "mallet": set(MALLET_ABLETON_TAGS) | {"Timpani", "Gong"},
    "plucked": {"Harp", "Sitar", "Misc Plucked", "Classical Guitar", "Acoustic Guitar", "Electric Guitar"},
    "wind": {"Flute", "Clarinet", "Oboe", "Bassoon", "English Horn", "Woodwind Ensemble", "Sax",
             "Misc Woodwind", "Trumpet", "Trombone", "Tuba", "French Horn", "Brass Ensemble"},
    "string": {"Violin", "Viola", "Cello", "Double Bass", "Upright Bass", "Acoustic Bass",
               "String Ensemble", "Misc Strings"},
})
WIND_NAME_RE = _tunable("WIND_NAME_RE", _re.compile(r"flute|clarinet|oboe|bassoon|(?<![a-z])sax|recorder|harmonica|accordion|kazoo"
                           r"|melodica|ocarina|pan ?pipe|trumpet|trombone|tuba|(?<![a-z])horns?(?![a-z])"
                           r"|brass|woodwind|whistle", _re.I))
# A one-shot named as a wind or brass instrument is ACOUSTIC's, like a named plucked
# string: short wind notes (a sax blip, a kazoo) can read as drums or synths to the
# classifiers. An air horn, a whistle or siren, a stab, or a synth patch (Ableton: Lead,
# Pad, Synth ...) isn't.
WIND_RESERVE_NOT_RE = _tunable("WIND_RESERVE_NOT_RE", _re.compile(
    r"synth|air ?horn|fog ?horn|car ?horn|whistle|siren|(?<![a-z])fx(?![a-z])|riser|sweep", _re.I))
WIND_RESERVE_NOT_TAGS = _tunable("WIND_RESERVE_NOT_TAGS", ("Lead", "Pad"))
# Acoustic percussion isn't a synth, though Sononym's "Tone Triangles & Bells" and
# Ableton's Bell / Chime tags can say so (hand drums, triangles, metal struck objects).
# A SYNTH candidate from a drum or percussion folder, or from an acoustic-percussion or
# found-object pack, homes in PERC unless its name says synth.
PERC_SOURCE_RE = _tunable("PERC_SOURCE_RE", _re.compile(    # library: generic folder words; an overlay adds packs
    r"/(?:perc|percs|percussion|drums?|foley|metal)/", _re.I))
PERC_SOURCE_NOT_RE = _tunable("PERC_SOURCE_NOT_RE", _re.compile(r"synth", _re.I))
# An orchestra library may file its instruments under section folders with
# abbreviated file names that Ableton guesses at (Piano, Ride), which would scatter them
# over unrelated categories. The folder decides: they are ACOUSTIC's, banded and labeled
# by the instrument folder.
ORCH_PATH_RE = _tunable("ORCH_PATH_RE", _re.compile(_NEVER, _re.I))   # library: groups (section, instrument folder)
# the library-relative folders ORCH_PATH_RE covers, as prefixes ("Orchestra/Strings/"): the
# ACOUSTIC pool reads every file under them
ORCH_ROOTS = _tunable("ORCH_ROOTS", ())                               # library
ORCH_PATH_INSTRUMENT_RE = _tunable("ORCH_PATH_INSTRUMENT_RE", _re.compile(
    r"violin|viola|cello|contrabass|harp|horn|trombone|trumpet|tuba|bassoon|clarinet|flute|oboe"
    r"|piccolo", _re.I))
STRING_NAME_RE = _tunable("STRING_NAME_RE", _re.compile(r"violin|viola|cello|contrabass|double ?bass|upright ?bass|acoustic ?bass"
                             r"|string|pizz|bowed|(?<![a-z])arco(?![a-z])|erhu", _re.I))

# Loops from styles the default taste isn't for, by path word (dubstep and trap loops
# would crowd the 140-150 BPM breaks and tops). A style preset for those styles names only
# the folders below (config/presets/: balanced, trap, ...).
DRUMLOOP_STYLE_EXCLUDE = _tunable("DRUMLOOP_STYLE_EXCLUDE", _re.compile(
    r"(?<![a-z])(?:fx ?loops?|impact ?fx|synth ?bass|brass ?stab)(?![a-z])", _re.I))
# (folders that hold no drum loops: "FX Loops", "Impact FX", "Synth Bass"; preset
# breaks-acid also leaves out dubstep, trap and future bass folders)
# A drum loop's longest length and its CLAP gate's minimum: 16 s holds 8 bars at 120 BPM
# and 4 bars down to 60; a style with slow 8-bar loops (hip hop at 85 BPM: 22.6 s) raises it.
DRUMLOOP_DUR_MAX = _tunable("DRUMLOOP_DUR_MAX", 16.0)
DRUMLOOP_BREAK_MIN = _tunable("DRUMLOOP_BREAK_MIN", 0.30)
# the drum loops' harmonicity gate: a loop more tonal than this is a phrase, not a drum loop
# (lo-fi drums under a chord, or filtered, read more tonal: hiphop-lofi raises it)
DRUMLOOP_HAR_MAX = _tunable("DRUMLOOP_HAR_MAX", 0.68)
CATEGORIES = _tunable("CATEGORIES", _tax.with_taxonomy({   # kind and labels: config/taxonomy.yaml
    "KICKS":  dict(phrases=KICK_PHRASES,
                   noise=r"(^sd[ _-]|snare|snr|clap|\bhat\b|^(?:lt|mt|ht)[._ -]|\btoms?\b|low tune)"),
    "SNARES": dict(phrases=SNARE_PHRASES,
                   noise=r"(^bd[ _-]|kick|\bhat\b|hi.?hat|cymbal|\bride\b|^oh[ _]|^ch[ _])"),
    "HATS":   dict(phrases=HAT_PHRASES, band="hat",
                   noise=r"(^bd[ _-]|kick|snare|\bsnr\b|\bclap\b|\btom\b|cymbal|crash)"),
    "SUB":    dict(phrases=SUB_PHRASES,
                   noise=r"(?!x)x", kmax=40,),
    "SYNTH":  dict(phrases=SYNTH_PHRASES,
                   noise=r"(?!x)x", kmax=48,),
    "CLAPS":  dict(phrases=CLAP_PHRASES,
                   noise=r"(kick|snare|\bhat\b|\btom\b)"),
    "TOMS":   dict(phrases=TOM_PHRASES,
                   noise=r"(kick|snare|\bhat\b|\bclap\b|cymbal)"),
    "PERC":   dict(phrases=PERC_PHRASES, noise=r"(?!x)x"),
    "CYMBALS": dict(phrases=CYMBAL_PHRASES, noise=r"(?!x)x", band="cymbal"),
    "FX":     dict(phrases=FX_PHRASES + [p for ps in FX_BAND_PHRASES.values() for p in ps
                                         if p not in FX_PHRASES],
                   band="fx", kmax=40,
                   # fixed type shares, weighted to transitions for breaks / dnb / techno
                   # (foley and synth blips usually outnumber risers and impacts in a pool);
                   # scratches (SCRATCHES folded in) are named files only
                   band_share=FX_BAND_SHARE,
                   # drum hits that leak in from the FX classifiers (a clap, a ride, an open hat)
                   noise=r"clap|snare|(?<![a-z])ride(?![a-z])|hi-?hat|(?<![a-z])(oh|ch|hh|bd|sd)(?![a-z])"
                         r"|kick ?drum|(?<![a-z])kick(?![a-z])"),
                   # (an added category may carve its sounds out of FX: exclude_phrases,
                   # exclude_anti, exclude_min; taxonomy.Taxonomy.with_taxonomy)
    "VOX":    dict(phrases=VOX_PHRASES,
                   noise=r"(?!x)x"),
    "PADS":   dict(phrases=PAD_PHRASES, noise=r"(?!x)x",),
    "STABS":  dict(phrases=STAB_PHRASES, noise=r"(?!x)x",
                   stab_names=True,),   # named stabs only (STAB_NAME_RE, STAB_CHORD_NAME_RE)
    "BLIPS":  dict(phrases=BLIP_PHRASES, noise=r"(?!x)x",),
    "DRUMLOOPS": dict(phrases=BREAK_PHRASES, band="loop",
                   anti=BREAK_ANTI_PHRASES, noise=r"(?!x)x",  # no filename noise filter for loops
                   break_min=DRUMLOOP_BREAK_MIN, har_max=DRUMLOOP_HAR_MAX, dur_min=1.0, dur_max=DRUMLOOP_DUR_MAX,
                   bpm_min=60, bpm_max=200,
                   kdiv=180, kmax=40, kmin=12, drum_tag_required=True, tempo_bands=TEMPO_BANDS, tempo_fold=TEMPO_FOLD,
                   # a loop with no whole-bar tempo can't sync to a sequencer, and a
                   # "freetempo" folder would be a catch-all
                   require_tempo=True, path_exclude=DRUMLOOP_STYLE_EXCLUDE,
                   band_tempo={"tops": TOPS_TEMPO_BANDS, "classic": (0, 1000)}, folder_max=32,
                   no_fold=NO_FOLD_RE, keep_all_bands=("classic",),
                   tempo_folder_files=110,),   # a 5-BPM band is one folder unless it will hold 110+ files  # drum loops (breaks + electronic)
    # musical loops (basslines, chord progressions, riffs, string swells): see PHRASES above.
    # Homed ahead of the one-shots (curate.compute_homes), gated musical-over-drum-break by
    # CLAP, and folded by tempo into role x tempo-range folders ("chords-120-135bpm-...").
    "PHRASES": dict(phrases=PHRASE_PHRASES, anti=BREAK_PHRASES,
                   # the gate is contrastive only: closer to the musical phrases than to the
                   # drum breaks (an absolute floor drops real phrases: acid lines, sequences,
                   # bowed strings)
                   noise=r"(?!x)x", dims=PHRASE_DIMS, break_min=0.0,
                   dur_min=PHRASE_DUR[0], dur_max=PHRASE_DUR[1], bpm_min=60, bpm_max=200,
                   band="phrase", band_share=PHRASE_ROLE_SHARE, folder_max=20,   # 5 roles x 4 tempo ranges
                   tempo_bands=PHRASE_TEMPO_BANDS, tempo_fold=TEMPO_FOLD,
                   # one folder per role and tempo range (12): kmax 8 splits into ~2-3 a role,
                   # and each tempo range keeps its own folder
                   kdiv=100000, kmax=8, kmin=8, chain_guard=None, loop_guard=None,
                   name_instruments="phrase",
                   # one folder of one pack (many phrases of one instrument, say)
                   # supplies at most this many CLAP-spread candidates
                   dir_cap=30,),
    "PIANO":  dict(pack_re=PIANO_PACKS, phrases=PIANO_ANCHOR, dims=INSTR_DIMS,
                   noise=r"(?!x)x", clap_source=True, chord_band=True, band="chord", clap_anti=PIANO_ANTI, clap_min=0.30,
                   pool_labels=PIANO_POOL_LABELS, max_onsets=2.0, min_chroma=1.6, dur_max=30.0,
                   ableton_exclude=ORCH_ABLETON_TAGS + MALLET_ABLETON_TAGS,
                   ableton_exclude_unless_name=KEYBOARD_NAME_RE,
                   kdiv=40, kmax=14, kmin=6,),
    "WAVES":  dict(phrases=WAVE_PHRASES, dims=WAVE_DIMS, noise=r"(?!x)x", band="wave",
                   near_dup_cos=1.0, kmin=6, kmax=16, kdiv=60),
    "ACOUSTIC": dict(pack_re=ORCH_PACKS, phrases=INSTR_PHRASES, dims=INSTR_DIMS,
                   name_any=PLUCKED_NAME_RE, band="acoustic",
                   # fixed band shares, weighted to what jungle / dnb / house use (strings,
                   # then winds) over pool size (so no band takes the category because its
                   # pool is large); every band gets at least two folders
                   band_share={"string": 0.30, "wind": 0.20, "plucked": 0.25, "mallet": 0.25},
                   band_min_folders=2,
                   noise=r"(?!x)x", max_onsets=2.5, dur_max=30.0, ableton_any=ACOUSTIC_ABLETON_TAGS,
                   pack_exclude=ORCH_PACK_EXCLUDE, name_exclude=ACOUSTIC_NAME_EXCLUDE,
                   kdiv=20, kmax=30, kmin=12,),
}))
# Drum hits start at the hit: a lead of low noise above the -50 dB floor would keep
# near-silence before the attack. Their lead is judged on the
# 10 ms RMS envelope at this floor (a category's trim_lead_db).
HIT_LEAD_FLOOR_DB = _tunable("HIT_LEAD_FLOOR_DB", -40.0)
for _c in ("KICKS", "SNARES", "CLAPS", "TOMS"):
    CATEGORIES[_c].setdefault("trim_lead_db", HIT_LEAD_FLOOR_DB)
# A stereo file whose channels nearly cancel (one flipped) keeps its louder channel
# (summed to mono it can lose tens of dB); a partly flipped one warns.
PHASE_FIX_CORR = _tunable("PHASE_FIX_CORR", -0.8)
PHASE_FIX_LOSS_DB = _tunable("PHASE_FIX_LOSS_DB", 10.0)
PHASE_WARN_CORR = _tunable("PHASE_WARN_CORR", -0.3)

# Keys routing: Sononym files keys under "Tone Leads & MidHiKeys", so piano / e-piano /
# clav notes and chords would home in SYNTH, SUB and FX. A keys sample is one
# named as keys (KEYBOARD_NAME_RE) or one CLAP places with PIANO's anchors (margin over
# the anti set >= KEYS_CLAP_MARGIN) that is also clearly pitched. It never homes in
# KEYS_EXCLUDE: every piano file is PIANO's, single notes and chord stabs alike, and
# PIANO splits them into note and chord folders (chord_band).
KEYS_EXCLUDE = _tunable("KEYS_EXCLUDE", {"SYNTH", "SUB", "FX", "STABS"})
KEYS_CLAP_MARGIN = _tunable("KEYS_CLAP_MARGIN", 0.10)
KEYS_MIN_HARMONICITY = _tunable("KEYS_MIN_HARMONICITY", 0.5)
KEYS_CHORD_HOME = _tunable("KEYS_CHORD_HOME", _tax.home("key_chords"))   # chords are PIANO's too
# a chord STAB is short and unmetered; a progression, a bpm-named phrase or a long
# keys pad is not, and PIANO leaves it out
KEYS_CHORD_MAX_DUR = _tunable("KEYS_CHORD_MAX_DUR", 4.0)
KEYS_PHRASE_NAME_RE = _tunable("KEYS_PHRASE_NAME_RE", _re.compile(r"\d+\s*bpm|progression|\bprog\b|\bloop|\bpt\s*\d", _re.I))
# samples routed into a category from outside its own classifier bucket (keys chords
# into STABS) may make up at most this share of its pool, so a big piano-chord pack
# can't turn STABS into a piano folder
ROUTED_MAX_SHARE = _tunable("ROUTED_MAX_SHARE", 0.35)
KEYS_CHORD_NAME_RE = _tunable("KEYS_CHORD_NAME_RE", _re.compile(
    r"chord|(?<![a-z])(?:maj|min|major|minor|m7|maj7|min7|sus[24]?|dim|aug|add9)(?![a-z])"
    # root + quality ("Am", "F#m7", "Cmaj7"): case-sensitive, so an upper-case prefix isn't one
    r"|(?-i:(?<![A-Za-z])[A-G](?:#|s|b)?(?:m|maj|min|dim|aug|sus)\d*(?![A-Za-z]))", _re.I))

# Packs whose every sample has exactly one home: a pack that belongs to one category puts
# its files there or nowhere, where a category's CLAP gate alone would let a few slip into
# a neighbouring category.
PACK_HOME = _tunable("PACK_HOME", {})   # library: pack name -> its only category

# Categories renamed after ratings were stored against them. The ratings
# store maps old names on load.
# MALLETS was folded into ACOUSTIC (acoustic tuned percussion) and SYNTH (FM / synth
# mallets); ratings.load_store sends a synth-pack Keep to SYNTH.
RENAMED_CATEGORIES = _tunable("RENAMED_CATEGORIES", _tax.renamed())

# Where packs sit in the library. A pack is the first folder under the library root,
# except under an umbrella folder (UMBRELLA_VENDORS, INSTRUMENT_ROOTS): there it's the second.
UMBRELLA_VENDORS = _tunable("UMBRELLA_VENDORS", ())                   # library
# top folders of instrument collections (a folder per instrument pack): umbrella folders
# whose files are all in the instrument categories' pools (PIANO, ACOUSTIC)
INSTRUMENT_ROOTS = _tunable("INSTRUMENT_ROOTS", ())                   # library
# words a vendor adds to every pack name, dropped from the pack ("Deep Kit Acme" -> "Deep Kit")
PACK_SUFFIX_RE = _tunable("PACK_SUFFIX_RE", _re.compile(_NEVER, _re.I))   # library
# umbrella vendors whose packs may lead a folder name (their pack names say what they hold)
NAMEABLE_VENDORS = _tunable("NAMEABLE_VENDORS", ())                   # library
# ...and other packs allowed to lead a folder name
NAMEABLE_EXTRA = _tunable("NAMEABLE_EXTRA", set())   # library


# ---------------------------------------------------------------------------
# Deterministic build seed. Fixed so a rebuild reproduces the same clustering,
# family names, and file selection given the same library + config + CLAP
# embeddings. Bump only to intentionally reshuffle. Determinism is same-machine /
# same-env and is recorded in each build's manifest.json.
CURATION_SEED = _tunable("CURATION_SEED", 0)

# Ableton Live auto-tag -> Fourier category. Live analyses the whole library
# tree, so these make Ableton a co-equal classifier with Sononym: a sample is a
# candidate wherever EITHER source places it (compute_homes; the scan's Live step, ingest/ableton_tags.py).
# Type tags only, not the One Shot / Loop class tags.
ABLETON_TAG_CATEGORY = _tunable("ABLETON_TAG_CATEGORY", {
    "Kick": "KICKS",
    "Snare Hit": "SNARES", "Rim": "SNARES",
    "Closed Hihat": "HATS", "Open Hihat": "HATS", "Shaker": "HATS", "Tambourine": "HATS",
    "Clap": "CLAPS",
    "High Tom": "TOMS", "Mid Tom": "TOMS", "Low Tom": "TOMS",
    "Conga": "PERC", "Bongo": "PERC", "Wood": "PERC", "Cowbell": "PERC",
    "Woodblock": "PERC", "Timbale": "PERC", "Cabasa": "PERC", "Guiro": "PERC",
    "Ride": "CYMBALS", "Crash": "CYMBALS",
    "Synth Bass": "SUB",
    "Lead": "SYNTH", "Synth Keys": "SYNTH",
    "Pad": "PADS", "Atmosphere": "PADS",
    # tuned-percussion tags backed by a name or pack are reserved for ACOUSTIC before
    # homing; what's left under these is a synth bell or mallet
    "Bell": "SYNTH", "Misc Mallets": "SYNTH", "Bell Chromatic": "SYNTH", "Chime": "SYNTH",
    "Synth Mallets": "SYNTH",
    "Solo Voice": "VOX", "Synth Voice": "VOX",
    "Choir": "VOX",
    "Sound FX": "FX", "Sweep": "FX", "Impact": "FX", "Noise": "FX", "Field & Foley": "FX",
    "Drum Loop": "DRUMLOOPS",
})
# Without Live (metadata/providers.py), the few Live tags a category needs as evidence come
# from the file's name and its folder instead (curate._name_tags; never a vendor's or pack's
# name): a word here, as whole words (a plural too), stands in for the tag. DRUMLOOPS keeps
# a loop only with a drum tag; the ACOUSTIC pool takes bowed strings by their instrument tag.
NAME_TAGS = _tunable("NAME_TAGS", {
    "Drum Loop": ["drum loop", "drumloop", "drum_loop", "break", "breakbeat", "beat", "groove"],
    "Violin": ["violin", "fiddle"], "Viola": ["viola"], "Cello": ["cello"],
    "Double Bass": ["contrabass", "double bass", "upright bass"],
    "String Ensemble": ["string ensemble", "string section", "string quartet"],
})
# Words a library names its sounds with that the path rules don't know (the `words` knob,
# fourier/knobs.py): {CATEGORY: [word, ...]}, each a strong path word for the category's first
# canonical label (metadata/shadow.path_words): "bombo" for KICKS, "nappe" for PADS. The path
# provider classifies only without Sononym, so with Sononym these route nothing.
PATH_WORDS = _tunable("PATH_WORDS", {})
ABLETON_TAGS_BY_CATEGORY = {}
for _t, _c in ABLETON_TAG_CATEGORY.items():
    ABLETON_TAGS_BY_CATEGORY.setdefault(_c, []).append(_t)


# Per-category file BUDGET: total curated files aimed for, distributed across a
# category's families by cluster size (per folder: floor FOLDER_MIN_FILES, or
# curate.MIN_PER_FAMILY for tempo-banded categories; ceiling curate.MAX_PER_FAMILY, 120), so fewer
# families => fuller folders and the total tracks the budget. The default (preset
# breaks-acid) is weighted toward breakbeat and acid material, and toward sounds a
# sampler can't make itself (pads and chords, poly stabs, washy drones and atmospheres).
# Tonal categories and FX get room in proportion to their large pools, and folders are
# few and full (FOLDER_MAX 12, up to 120 files a folder): fewer folders to read on the
# device, more to choose from inside each.
BUDGETS = _tunable("BUDGETS", {     # the balanced style's (every preset scales these)
    "DRUMLOOPS": 1140, "PHRASES": 720, "KICKS": 660, "SNARES": 600, "HATS": 600, "PERC": 600,
    "SUB": 630, "TOMS": 340, "CLAPS": 360, "CYMBALS": 300, "FX": 660,
    # (scratches are FX's "scratch" band; FX's types have fixed shares, FX_BAND_SHARE)
    "SYNTH": 800, "PADS": 720, "STABS": 390, "BLIPS": 270, "VOX": 488,
    # (PIANO and STABS: what their pools can usually fill; PIANO holds the organs too)
    "PIANO": 455, "WAVES": 350, "ACOUSTIC": 455,
})
BUDGETS = _tax.add_values(BUDGETS, "budget")   # ...and an added category's own
# What one exported file weighs on a card, on average, per category (MB; 16-bit WAV; rough
# defaults: loops and pads heavy, drum hits light), and the derived sets' extra share
# (00_KITS and 00_SLICE copies, as a share of the categories). The size knob turns a
# card size into budgets with these (fourier/knobs.py).
AVG_FILE_MB = _tunable("AVG_FILE_MB", {
    "KICKS": 0.06, "SNARES": 0.05, "CLAPS": 0.05, "HATS": 0.04, "CYMBALS": 0.29, "TOMS": 0.10,
    "PERC": 0.05, "DRUMLOOPS": 1.01, "PHRASES": 1.89, "SUB": 0.37, "SYNTH": 0.47, "STABS": 0.26,
    "PADS": 1.53, "PIANO": 0.64, "ACOUSTIC": 0.44, "WAVES": 0.03, "FX": 0.72, "BLIPS": 0.05,
    "VOX": 0.46,
})
AVG_FILE_MB = _tax.add_values(AVG_FILE_MB, "avg_file_mb")
SETS_SHARE = _tunable("SETS_SHARE", 0.12)
# size = "auto" (fourier/knobs.py): budgets scaled down (never up) so the curated build with
# its sets fits this share of the smallest configured device's storage, and one surplus round:
# the files a category's pool couldn't fill go to the categories that filled theirs, which
# are rebuilt with the larger budget (curate.surplus_budgets). Off by default (the budgets
# as the preset sets them).
STORAGE_SHARE = _tunable("STORAGE_SHARE", 0.5)
POOL_SURPLUS = _tunable("POOL_SURPLUS", True)
# scale = "library" (fourier/knobs.py, packs/scale.py): the master's size follows the analyzed
# library. A library with fewer than LIBRARY_PER_MASTER usable samples for each file the
# budgets add up to gets a master scaled by f = samples / (LIBRARY_PER_MASTER x budgets), with
# a smaller minimum, smaller folders and fewer of them; at f = 1 (a library at least that
# large) nothing changes. In a scaled build a category keeps every usable file up to
# SCALED_KEEP_ALL and one in LIBRARY_PER_MASTER beyond that, or its scaled budget if that's
# more (never more than its budget or its pool); it builds from SCALED_MIN_FILES usable
# files (one file is no choice, and alone in a category of a small library it is more often
# a stray than a sound to play); it gets about one folder per SCALED_FOLDER_FILES files, and
# a folder under SCALED_FOLDER_MIN_FILES merges into its nearest neighbour of the same band.
LIBRARY_SCALE = _tunable("LIBRARY_SCALE", True)
LIBRARY_PER_MASTER = _tunable("LIBRARY_PER_MASTER", 8)
SCALED_KEEP_ALL = _tunable("SCALED_KEEP_ALL", 24)
SCALED_MIN_FILES = _tunable("SCALED_MIN_FILES", 2)
SCALED_FOLDER_FILES = _tunable("SCALED_FOLDER_FILES", 24)
SCALED_FOLDER_MIN_FILES = _tunable("SCALED_FOLDER_MIN_FILES", 3)

# Device renders number the category folders so they sort in the order they're reached
# for, drums first (01_KICKS, 02_SNARES, ...), rather than alphabetically (ACOUSTIC and
# BLIPS on top). Both devices sort folders by name and jump to either end of a list (M8
# [LEFT]/[RIGHT], manual p. 16). The build keeps plain category names; only the device
# path changes (render._rel_dest). PHRASES sits right after DRUMLOOPS, so the loops are
# together (09_PHRASES; added before any device path lock). A category an overlay adds is
# numbered after all of these (taxonomy.with_added), so it never renumbers them.
CATEGORY_ORDER = _tunable("CATEGORY_ORDER", _tax.order())
# categories left out of a build (the categories knob: WAVES = "off"); their folder numbers
# stay free, so the others keep theirs
CATEGORIES_OFF = _tunable("CATEGORIES_OFF", set())


# Derived sets (packs/sets.py): copies of curated files grouped for playing,
# numbered 00 so they sit above the categories on both devices.
DERIVED_DIRS = _tunable("DERIVED_DIRS", _tax.sets())


def category_dir(category: str) -> str:
    """Device folder for a category: '01_KICKS'; a category not in the order keeps its name."""
    if category in DERIVED_DIRS:
        return DERIVED_DIRS[category]
    try:
        return f"{CATEGORY_ORDER.index(category) + 1:02d}_{category}"
    except ValueError:
        return category


# ---------------------------------------------------------------------------
# Name-based home overrides
# ---------------------------------------------------------------------------
# A few instruments the two analyzers reliably mis-slot: Ableton tags a music
# box / kalimba / thumb piano as an orchestral mallet tag so they route to
# ACOUSTIC's mallet band when they are really tine sounds; a church bell and a
# clavinet land by name in the wrong keys folder; an "ah vox" solo vowel lands
# in ACOUSTIC instead of VOX. A filename containing one of these substrings is
# forced to the paired category and kept out of every other one. A "Synth <x>"
# is exempt (NAME_OVERRIDE_SKIP) so synth versions keep their synth home.
# Case-insensitive substring match on the filename.
# Each rule is (substrings, category); substrings naming the same instrument are
# grouped into one rule so they share one quota (clavinet + "the clav").
NAME_OVERRIDES = _tunable("NAME_OVERRIDES", [
    # tuned percussion is ACOUSTIC's (its mallet band; MALLETS was folded in),
    # cowbells and agogos PERC's. A synth-pack patch or a "Synth ..."-tagged file keeps
    # its own home (curate._row_override).
    (("music box",), "ACOUSTIC"),
    (("thumb piano", "mbira"), "ACOUSTIC"),
    (("kalimba",), "ACOUSTIC"),
    (("church bell",), "ACOUSTIC"),
    (("marimba",), "ACOUSTIC"),
    (("xylophone",), "ACOUSTIC"),
    (("vibraphone",), "ACOUSTIC"),
    (("glockenspiel",), "ACOUSTIC"),
    (("steel drum", "steelpan", "steel pan"), "ACOUSTIC"),
    (("celesta", "celeste"), "ACOUSTIC"),
    (("gamelan",), "ACOUSTIC"),
    (("toy piano",), "ACOUSTIC"),
    (("cowbell", "agogo"), "PERC"),
    (("cello harmonics",), "ACOUSTIC"),
    (("clavinet", "the clav"), "PIANO"),
    (("ah vox",), "VOX"),
])
NAME_OVERRIDE_SKIP = _tunable("NAME_OVERRIDE_SKIP", ("synth",))
# Words that veto every override despite a substring hit: "The Clave" (percussion) contains
# "the clav". Matching stays substring-based on purpose: compound names like "SoftKalimba",
# "kalimba_C#" and "Clavinette" are real hits that word boundaries would lose.
NAME_OVERRIDE_VETO = _tunable("NAME_OVERRIDE_VETO", _re.compile(r"(?<![a-z])claves?(?![a-z])"))

# Folders per category (dozens in one category is a lot to scroll on a small device).
# Clustering asks for at most FOLDER_MAX families (a category's kmax is capped by it) and each folder may hold up to curate.MAX_PER_FAMILY (120) files, so the
# budget fits. Banded categories merge a band's nearest folders when rounding overshoots.
# DRUMLOOPS keeps its tempo ranges (one or more folders each), so it gets more.
FOLDER_MAX = _tunable("FOLDER_MAX", 12)
# ...and a small category gets about one folder per FOLDER_TARGET_FILES files (at least
# FOLDER_MIN), so a small category isn't split 8 ways. Only lowers a category's folder
# count, never raises it.
FOLDER_TARGET_FILES = _tunable("FOLDER_TARGET_FILES", 45)
FOLDER_MIN = _tunable("FOLDER_MIN", 3)

# --- naming must describe the folder ---
# The CLAP phrase that best separates a folder from its siblings need not describe what
# it holds (a phrase for one kind of voice on another). So a
# phrase may name a folder only when it is among the top NAME_TOPK phrases (of those the
# folder may use) for at least NAME_SUPPORT_MIN of its files. A genre or drum-machine word
# ("909", "house", "footwork") also needs GENRE_SUPPORT_MIN of the files' paths to say
# so. Otherwise the folder is named by its category noun and traits ("kick-punchy-dark").
NAME_TOPK = _tunable("NAME_TOPK", 3)
NAME_SUPPORT_MIN = _tunable("NAME_SUPPORT_MIN", 0.40)
# Categories whose naming phrase must rank among a file's top NAME_TOPK over ALL the
# category's phrases, not just its band's: an FX band has few phrases, so one can top the
# band's list while fitting a small share of the folder.
# Such a folder falls back to its band and traits ("foley-bright").
NAME_SUPPORT_FULL = _tunable("NAME_SUPPORT_FULL", {"FX"})
GENRE_SUPPORT_MIN = _tunable("GENRE_SUPPORT_MIN", 0.30)
# a break (not a tops loop) at jungle tempo backs "jungle" like a path saying so: a break
# at 155-180 BPM is a jungle break, whatever its folder says
JUNGLE_BPM = _tunable("JUNGLE_BPM", (155.0, 180.0))
JUNGLE_TEMPO_TOKEN = _tunable("JUNGLE_TEMPO_TOKEN", "@jt@")     # (no "jungle" in it: it backs "jungle" only, not dnb or ragga)
GENRE_WORDS = _tunable("GENRE_WORDS", {   # phrase word -> what a file path must say to back it
    "909": r"909", "808": r"808", "606": r"606", "707": r"707|727", "303": "|".join(["303", "acid", *MACHINE_ALIASES.get("303", ())]),
    # drum machines a folder may be named for (DRUM_MACHINES)
    "cr78": r"cr-?78", "linndrum": r"linn|lindrum|(?<![a-z])lm-?1(?![0-9])", "dmx": r"(?<![a-z])dmx",
    "drumulator": r"drumulator", "drumtraks": r"drumtra[xk]", "sp1200": r"sp-?1200",
    "mpc60": r"mpc ?60(?![0-9])", "mpc3000": r"mpc ?3000|mpc3k", "simmons": r"simmons|sds ?v|sdsv|sds ?800",
    "synare": r"synare", "rytm": r"rytm",
    "acid": r"acid|303", "house": r"house", "techno": r"techno", "footwork": r"footwork|juke",
    "reggae": r"reggae|dub(?!step)", "dub": r"dub(?!step)", "dubstep": r"dubstep", "jungle": r"jungle|@jt@",
    "dnb": r"dnb|drum ?(?:&|and|n|'n') ?bass|(?<![a-z])d ?& ?b(?![a-z])|jungle",
    "drum and bass": r"dnb|drum ?(?:&|and|n|'n') ?bass|(?<![a-z])d ?& ?b(?![a-z])|jungle",
    "trance": r"trance", "rave": r"rave|hardcore", "hardcore": r"hardcore|rave",
    "hip hop": r"hip ?-?hop", "boom bap": r"boom ?-?bap", "disco": r"disco", "gospel": r"gospel",
    "funk": r"funk", "soul": r"soul", "jazz": r"jazz", "blues": r"blues", "detroit": r"detroit",
    "amen": r"amen", "garage": r"garage|2 ?-?step|ukg", "trap": r"(?<![a-z])trap",
    "tribal": r"tribal", "electro": r"electro", "m1": r"(?<![a-z])m1(?![a-z])|korg",
    "ragga": r"ragga|reggae|dancehall|jungle", "dancehall": r"dancehall|ragga|reggae",
})
# ...plus the words of a library's own naming prompts (CLASSIC_BREAK_PROMPTS), added after them
GENRE_WORDS_EXTRA = _tunable("GENRE_WORDS_EXTRA", {})   # library: phrase word -> path regex
GENRE_WORDS = {**GENRE_WORDS, **GENRE_WORDS_EXTRA}
# Instrument and source nouns a naming phrase may carry only when the files back them:
# at least GENRE_SUPPORT_MIN of the folder's files say so in their path or Ableton tags,
# like genre words. CLAP alone hears the family resemblance, not the instrument, so a
# "djembe" or "china" phrase can fit a folder that holds none.
INSTRUMENT_WORDS = _tunable("INSTRUMENT_WORDS", {
    # hand and kit percussion
    "agogo": r"agogo", "bongo": r"bongo", "cabasa": r"cabasa", "cajon": r"cajon", "castanet": r"castanet",
    "china": r"china", "clave": r"clave", "conga": r"conga|tumba|quinto", "cowbell": r"cow ?bell|(?<![a-z])cb(?![a-z])",
    "djembe": r"djembe", "guiro": r"guiro", "pedal": r"pedal|(?<![a-z])ph(?![a-z])|foot",
    "rimshot": r"rim|(?<![a-z])rs(?![a-z])|side ?stick", "rim": r"rim|(?<![a-z])rs(?![a-z])|side ?stick",
    "ride": r"ride", "rototom": r"roto", "tabla": r"tabla", "taiko": r"taiko", "tambourine": r"tamb",
    "timbale": r"timbal", "triangle": r"triang", "udu": r"(?<![a-z])udu", "woodblock": r"wood ?block|(?<![a-z])wb(?![a-z])",
    "anvil": r"anvil", "shaker": r"shak|maraca|cabasa",
    # keys and instruments
    "clav": r"clav", "clavinet": r"clav", "rhodes": r"rhodes|fender|e-?piano|electric piano|(?<![a-z])ep(?![a-z])|suitcase",
    "wurlitzer": r"wurli", "piano": r"piano|grand|upright|(?<![a-z])keys?(?![a-z])", "organ": r"organ|hammond|b-?3|drawbar",
    "flute": r"flute|piccolo", "piccolo": r"piccolo", "sitar": r"sitar", "guitar": r"guitar|gtr",
    "marimba": r"marimba", "brass": r"brass|horn|trumpet|trombone|tuba|sax", "horn": r"horn",
    "choir": r"choir|choral|(?<![a-z])(?:aah|ooh)(?![a-z])", "female": r"female|(?<![a-z])fem|girl|woman|lady|diva|soprano",
    "male": r"(?<![a-z])(?:male|man|guy|baritone)(?![a-z])", "diva": r"diva|female|soul",
    "reese": r"reese", "hoover": r"hoover", "supersaw": r"super ?saw|jp-?8000", "donk": r"donk",
    # sources and techniques a CLAP phrase may claim (a sound can resemble a talkbox or a
    # modular patch without being one)
    "talkbox": r"talk ?box|vocoder", "vocoder": r"vocoder", "beatbox": r"beat ?box|human ?beat",
    "cardboard": r"cardboard|box", "modular": r"modular|eurorack|buchla|serge",
    "slap": r"slap", "vinyl": r"vinyl|record|dusty|crackle", "tape": r"tape|cassette",
})
# ...plus an added category's own words (its name_words, after these)
INSTRUMENT_WORDS = _tax.add_words(INSTRUMENT_WORDS, "name_words")
# ACOUSTIC names a folder for its instrument only when at least this share of its files
# are that instrument
ACOUSTIC_SOURCE_MIN = _tunable("ACOUSTIC_SOURCE_MIN", 0.4)

# --- one note of an instrument is enough ---
# On the Digitakt II and M8 a pitched sample is played chromatically, so twenty notes of
# one patch fill twenty slots with one sound. In these categories a folder keeps at most
# SIBLING_CAP files of one
# multisample set (curate._sibling_key: same pack and name once note, velocity and
# number tokens are removed), spread across its register. A Keep is always kept.
SIBLING_CAP = _tunable("SIBLING_CAP", 3)
SIBLING_CAP_CATS = _tunable("SIBLING_CAP_CATS", {"SUB", "SYNTH", "PADS", "STABS", "VOX", "BLIPS", "PIANO", "ACOUSTIC"})

# Note leads ("d2-", "as1-") sorted by name, not pitch, and a folder's median note says
# little about a folder clustered by sound: only PIANO keeps one (ACOUSTIC's band lead
# carries its note).
NOTE_LEAD_CATS = _tunable("NOTE_LEAD_CATS", _tax.with_role("note_lead"))

# Long files from a loop folder that didn't qualify as phrases (no whole-bar tempo, not
# classed a loop) stay out of the tonal one-shot folders: a long "..._Loop_01" is not a
# pad. Short ones (a chopped hit from a "Drum Loops/" folder) are one-shots.
LOOP_DIR_ONESHOT_CATS = _tunable("LOOP_DIR_ONESHOT_CATS", {"SUB", "SYNTH", "PADS", "STABS", "VOX"})

# Chord stabs live in STABS. A short one-shot named as a chord ("Pad (Chord) 01",
# "Chords C1", "Chord Gm", "Keys Dbmaj7") would otherwise home
# in SYNTH, SUB, FX, BLIPS or PADS: a baked chord
# plays only in parallel, so it's used like a stab, not a lead or a bassline. By name
# only: the chroma test for "chord" also fired on cymbals and toms. A bare key ("Bass Fm")
# labels a single note, so it doesn't count; vocal chords stay in VOX and keys chords in
# PIANO. Short and single-hit as for keys (curate._is_stab: <= KEYS_CHORD_MAX_DUR s).
STAB_CHORD_NAME_RE = _tunable("STAB_CHORD_NAME_RE", _re.compile(
    r"(?<![a-z])chords?(?![a-z])"
    r"|(?<![a-z])(?:maj7|maj9|min7|min9|m7|m9|7th|sus[24]?|dim7?|aug|add9)(?![a-z])"
    r"|(?-i:(?<![A-Za-z])[A-G](?:#|b)?(?:maj7|maj9|m7|m9|min7|dim|aug|sus[24]?|add9)(?![A-Za-z]))", _re.I))
STAB_CHORD_FROM = _tunable("STAB_CHORD_FROM", {"SYNTH", "SUB", "FX", "BLIPS", "PADS", "PERC"})   # PERC: a chord in a percussion folder is still a chord
# STABS holds only named stabs: a chord (above) or a file its maker calls a
# stab or a hit of a tonal source ("Synth Stab 1", "Brass Stab 02", "Orchestra
# Hit", "Hoover Hit"). Sononym's "Tone Stabs & Orch. Hits" also takes zaps, glitch
# percussion, kicks and synth notes, so a
# file it alone calls a stab doesn't home in STABS. A stab-named file homes in STABS
# from STAB_CHORD_FROM like a chord does, except a bass stab (SUB's) or a vocal one (VOX's).
# Drum words veto both ("Perc Stab", "Kick Hit").
STAB_NAME_RE = _tunable("STAB_NAME_RE", _re.compile(
    r"(?<![a-z])(?:stabs?|stb)(?![a-z])|orch(?:estra(?:l)?)?[ _-]?hits?|orchhit"
    r"|(?<![a-z])(?:brass|string|horn|organ|choir|synth|chord|rave)[ _-]?hits?(?![a-z])|hoover", _re.I))
STAB_NAME_EXCLUDE = _tunable("STAB_NAME_EXCLUDE", _re.compile(
    r"kick|snare|percussion|(?<![a-z])(?:hh|hats?|toms?|percs?|claps?|cymbals?|rims?|zaps?|bd|sd|cy)(?![a-z])", _re.I))
STAB_MOVE_EXCLUDE = _tunable("STAB_MOVE_EXCLUDE", _re.compile(r"bass|(?<![a-z])(?:vox|vocals?|voices?)(?![a-z])", _re.I))

# --- scratches ---
# Scratches are named files only: Sononym's "Perc Vinyl Scratches" label also takes toms,
# open hats and zaps. A library may hold only a few dozen one-shot scratches, so they are
# FX's: a file named as a scratch (file or its folder) homes in FX, in its
# "scratch" band, from any one-shot category.
# "Piano scratch" / "Perc Scratch" textures and drum-named files don't count.
SCRATCH_NAME_RE = _tunable("SCRATCH_NAME_RE", _re.compile(
    r"(?<![a-z])(?:scratch(?:es|ed)?|scrtch|scr)(?![a-z])|transformer|turntabl|scribble"
    r"|(?:chirp|crab|flare|baby|tear|stab|orbit|drag|forward|backward)[ _-]?scratch"
    r"|record ?stop|spin ?back|back ?spin|(?<![a-z])rewind", _re.I))
SCRATCH_NAME_EXCLUDE = _tunable("SCRATCH_NAME_EXCLUDE", _re.compile(
    r"piano|perc|kick|snare|(?<![a-z])(?:hh|hats?|hihat|toms?|claps?|cymbals?)(?![a-z])|noise|foley|texture"
    # hand percussion scraped (a "Shaker Scratch") is a hit, not a turntable
    r"|shaker|guiro|cabasa|washboard|scrape", _re.I))
SCRATCH_HOME = _tunable("SCRATCH_HOME", _tax.home("scratches"))
SCRATCH_FROM = _tunable("SCRATCH_FROM", {"KICKS", "SNARES", "CLAPS", "HATS", "CYMBALS", "TOMS", "PERC", "SUB", "SYNTH",
                "FX", "BLIPS", "PADS", "STABS"})

# A file named as exactly one kind of drum homes in that drum category, whatever the
# classifiers say (a "BD" they call a tom or a bass, a "Clap" they call a snare). A name
# that names two kinds ("Kick Snare") decides nothing. "TOM" in capitals is a drum
# machine's name, not a tom; "zap" is a blip.
# "sd" takes no plural, so a machine name like "SDS800" isn't a snare; a kit that names its
# kicks "Bass" before the machine name ("Bass SDS 800") is a kick.
# CamelCase drum words count ("LowKick_1", "hiCymb_1"), an abbreviated cymbal counts
# ("Cymb 01"), and "RS" opening a name is a rimshot ("RS 01", as drum machines abbreviate it).
# Rims live in SNARES.
DRUM_NAME_RULES = _tunable("DRUM_NAME_RULES", [
    (_re.compile(r"(?<![a-z])(?:kicks?|kiks?|bds?|bass ?drums?)(?![a-z])|(?<![a-z])bass(?= ?sds ?800)"
                 r"|(?-i:(?<=[a-z])Kick)", _re.I), "KICKS"),
    (_re.compile(r"(?<![a-z])(?:snares?|snrs?|sd|rims?|rimshots?|sidesticks?|side ?stick)(?![a-z])|^rs(?= )"
                 r"|(?-i:(?<=[a-z])Snare)", _re.I), "SNARES"),
    (_re.compile(r"(?<![a-z])(?:clap|claps|handclap|snaps?|finger ?snaps?)(?![a-z])", _re.I), "CLAPS"),
    (_re.compile(r"hi-?hats?|hihats?|(?<![a-z])(?:hh|oh|ch|hat|hats|shakers?|tambourines?)(?![a-z])", _re.I), "HATS"),
    (_re.compile(r"(?<![A-Za-z])(?:Tom|tom|Toms|toms)(?![a-z])"), "TOMS"),
    (_re.compile(r"cymbal|(?<![a-z])(?:crash|ride|cy|cymbs?|splash|china)(?![a-z])|(?-i:(?<=[a-z])Cymb)",
                 _re.I), "CYMBALS"),
    (_re.compile(r"(?<![a-z])(?:conga|bongo|cowbell|clave|woodblock|timbale|agogo)s?(?![a-z])", _re.I),
     "PERC"),
    (_re.compile(r"(?<![a-z])zaps?(?![a-z])", _re.I), "BLIPS"),
])
DRUM_NAME_FROM = _tunable("DRUM_NAME_FROM", {"KICKS", "SNARES", "CLAPS", "HATS", "CYMBALS", "TOMS", "PERC", "SUB", "SYNTH",
                  "FX", "BLIPS", "PADS"})

# Near-silent one-shots: a peak-normalized one-shot whose RMS stays under this is mostly
# silence (an empty file, a faint brushes sweep). Not exported.
QUIET_RMS_DB = _tunable("QUIET_RMS_DB", -38.0)
# Tonal one-shots are retuned to C at export, so any sample plays in key from the same
# note on the Digitakt / M8 / Live. The root comes from the filename's note ("C#3", "Dbmaj7"); the detected pitch is trusted only
# in the categories where it tends to agree with named notes (RETUNE_DETECT_CATS; bass and
# stabs fool a pitch tracker). The shift is the smallest to reach a C (at most 6 semitones), by resampling.
# A multisample set gives its C notes first (they need no shift).
# A name with several notes (a sung line) isn't retuned. Voices and acoustic instruments move
# their formants and body resonance with the pitch (tape-style), so a large shift makes a
# voice a chipmunk and a clavinet a synth: VOX, PIANO and
# ACOUSTIC move at most RETUNE_MAX_SEMIS, and a file further from C keeps its own pitch and
# note name (a multisample set usually has its C notes anyway).
RETUNE_CATS = _tunable("RETUNE_CATS", {"SUB", "SYNTH", "STABS", "PADS", "PIANO", "ACOUSTIC", "VOX"})
RETUNE_DETECT_CATS = _tunable("RETUNE_DETECT_CATS", {"SYNTH", "PIANO", "VOX"})
PITCH_VOICED_MIN = _tunable("PITCH_VOICED_MIN", 0.5)      # a named note is checked against pYIN when half the frames are voiced
PITCH_SPREAD_MAX = _tunable("PITCH_SPREAD_MAX", 0.5)      # ...and their pitch sits within this many semitones (interquartile)
RETUNE_MAX_SEMIS = _tunable("RETUNE_MAX_SEMIS", {"VOX": 3.0, "PIANO": 3.0, "ACOUSTIC": 3.0})
# Stereo only where it's real: kicks and bass are mono (stereo lows are
# mud), and a stereo file whose sides are under NEAR_MONO_SIDE_DB of its middle is mono in
# all but name. Everything else keeps its width.
MONO_CATS = _tunable("MONO_CATS", {"KICKS", "SUB"})
NEAR_MONO_SIDE_DB = _tunable("NEAR_MONO_SIDE_DB", -30.0)   # sides under 3% of the middle: no audible width, and it doubles Digitakt RAM
# VOX is for vocal shots: choirs and synth voices (string-machine voices, aah / ooh pads)
# could crowd them out, so they're capped at VOX_CHOIR_SHARE of VOX's budget. A library's
# own names for such voices go in VOX_CHOIR_EXTRA (regexes).
VOX_CHOIR_EXTRA = _tunable("VOX_CHOIR_EXTRA", [])   # library
VOX_CHOIR_RE = _tunable("VOX_CHOIR_RE", _re.compile(
    "|".join([r"choir|choral|(?<![a-z])(?:aah|ahh|ooh|oohs)(?![a-z])|vp-?330", *VOX_CHOIR_EXTRA,
              r"(?:vox|voice|vocal) ?pad|vocoder ?choir"]),
    _re.I))
VOX_CHOIR_SHARE = _tunable("VOX_CHOIR_SHARE", 0.25)
# Organs live in PIANO (the keys category), not in SYNTH, PADS or SUB where the
# classifiers may put them. A named organ one-shot homes nowhere else; an organ STAB
# ("Organ Stab 3", the M1 house stab) stays a stab.
ORGAN_NAME_RE = _tunable("ORGAN_NAME_RE", _re.compile(r"(?<![a-z])organs?(?![a-z])|hammond|drawbar|farfisa|(?<![a-z])leslie(?![a-z])", _re.I))
# CYMBALS bands, so rides sit together and apart from crashes. The
# name decides (ride; crash / splash / china), then Sononym's ride / crash label, then
# Ableton's tag; anything else is "cymbal" (bells, gongs, washes).
CYMBAL_RIDE_RE = _tunable("CYMBAL_RIDE_RE", _re.compile(r"(?<![a-z])rides?(?![a-z])|(?<![a-z])rd(?![a-z])", _re.I))
CYMBAL_CRASH_RE = _tunable("CYMBAL_CRASH_RE", _re.compile(r"(?<![a-z])(?:crash(?:es)?|splash|china)(?![a-z])", _re.I))
# One name per file, in the build and on every device (a name cut only for one device's
# path limit would differ between devices). The build's file
# names are cut (in the middle, boilerplate stripped) to STEM_MAX characters, sized for
# the tightest path: the M8's 127 (manual p.74) = "/Samples/Fourier/" (17) + "08_DRUMLOOPS/"
# (13) + a family name of up to FAMILY_NAME_MAX + "/" + STEM_MAX + ".wav" + 2 for a "_2".
STEM_MAX = _tunable("STEM_MAX", 46)
# names (fourier/knobs.py): "canonical", a file's source name minus vendor boilerplate, cut in
# the middle to STEM_MAX (exporter.canonical_stem); "keep", the source name as it is (only
# made FAT-safe and cut to STEM_MAX), with a loop's BPM added when the name doesn't say it.
NAMES = _tunable("NAMES", "canonical")
# Folder descriptions written by a local LLM (Ollama, curate.DEFAULT_MODEL) into the manifest:
# off unless asked (fourier build --describe, or DESCRIBE = true in [advanced]).
DESCRIBE = _tunable("DESCRIBE", False)
# The sound model trained on this library (metadata/train.py): without Sononym, the analysis's
# sound step trains one when there's none yet and the library's own names label at least
# SOUND_TRAIN_MIN_LABELS samples across SOUND_TRAIN_MIN_PACKS packs, and trains it again when
# the labelled samples have grown by SOUND_RETRAIN_GROWTH (and by 500 more). Off: SOUND_TRAIN =
# false in [advanced] (fourier setup --no-sound-model), or $FOURIER_SOUND_MODEL set.
SOUND_TRAIN = _tunable("SOUND_TRAIN", True)
SOUND_TRAIN_MIN_LABELS = _tunable("SOUND_TRAIN_MIN_LABELS", 1000)
SOUND_TRAIN_MIN_PACKS = _tunable("SOUND_TRAIN_MIN_PACKS", 8)
SOUND_RETRAIN_GROWTH = _tunable("SOUND_RETRAIN_GROWTH", 0.25)
# ...and keeps it only when it's reliable on the packs it didn't learn from: at least
# SOUND_KEEP_MIN_ACCURACY of its calls at SOUND_MODEL_MIN or more agree with the names, those
# calls are at least SOUND_KEEP_MIN_COVERAGE of the held-out samples, and at least
# SOUND_KEEP_MIN_HELD were held out. Otherwise it's discarded (the report says why) and tried
# again when the library has grown as much as a retrain needs.
SOUND_KEEP_MIN_ACCURACY = _tunable("SOUND_KEEP_MIN_ACCURACY", 0.80)
SOUND_KEEP_MIN_COVERAGE = _tunable("SOUND_KEEP_MIN_COVERAGE", 0.10)
SOUND_KEEP_MIN_HELD = _tunable("SOUND_KEEP_MIN_HELD", 200)
# Pack boilerplate stripped from file names, in order (exporter._strip_boilerplate): regexes,
# case-insensitive. Vendor names and codes go in a library overlay.
NAME_BOILERPLATE = _tunable("NAME_BOILERPLATE", [r"[ _-]*sample[ _-]?pack(?=[ _-]|$)"])
FAMILY_NAME_MAX = _tunable("FAMILY_NAME_MAX", 44)
# A one-shot whose end is still loud (last 10 ms above this, dB below peak) gets an
# END_FADE_MS fade-out: a source cut off while still sounding clicks (pads, synths and
# basses most often).
END_HOT_DB = _tunable("END_HOT_DB", -40.0)
LOOP_LIMIT_DB = _tunable("LOOP_LIMIT_DB", 3.0)     # a loop's peaks may be limited by up to this to reach LOOP_RMS_DB
LIMIT_MS = _tunable("LIMIT_MS", 5.0)          # ...with a smooth gain: about this attack and release
DRUM_LEVEL_CATS = _tunable("DRUM_LEVEL_CATS", {"KICKS", "SNARES", "CLAPS", "HATS", "CYMBALS", "TOMS", "PERC"})
DRUM_LEVEL_OVER_DB = _tunable("DRUM_LEVEL_OVER_DB", 3.0)   # a drum hit more than this over its folder's median comes down to it
EDGE_HOT_DB = _tunable("EDGE_HOT_DB", -40.0)     # a file starting (or a loop ending) above this, relative to peak, gets an edge fade
EDGE_FADE_MS = _tunable("EDGE_FADE_MS", 0.5)      # ...this long: 22 samples at 44.1 kHz, under a transient's rise
END_FADE_MS = _tunable("END_FADE_MS", 10.0)
# A one-shot's tail is judged on a 10 ms RMS envelope (like instrument tails): judged
# sample by sample, a noise floor's spikes would hold on to long near-silence.
ONESHOT_TAIL_RMS = _tunable("ONESHOT_TAIL_RMS", True)
# A soft layer the 18 dB gain cap can't bring up (source peak under this) is left out
# when a sibling of the same sound is at least SOFT_LAYER_GAP_DB louder: a soft velocity
# layer or round robin stays quiet even at the cap.
SOFT_LAYER_PEAK_DB = _tunable("SOFT_LAYER_PEAK_DB", -19.0)
SOFT_LAYER_GAP_DB = _tunable("SOFT_LAYER_GAP_DB", 6.0)
# Mirror folders: a top-level folder of copies holds packs that also live
# elsewhere in the library (byte-identical files). A file there with a twin elsewhere is
# left out, and
# ratings on it apply to the twin; a file only there stays.
MIRROR_ROOT_RE = _tunable("MIRROR_ROOT_RE", _re.compile(_NEVER, _re.I))   # library: mirror folders
# Named transitions are FX's, wherever the classifiers put them (a "Sweep Down" or an
# "Uplifter" in a pad pack). A "sweep pad" is a pad.
FX_TRANSITION_NAME_RE = _tunable("FX_TRANSITION_NAME_RE", _re.compile(
    r"(?<![a-z])(?:up ?lift(?:er)?s?|down ?lift(?:er)?s?|risers?|sweep(?:s|ing)?|build ?ups?"
    r"|fx ?up|fx ?down)(?![a-z])", _re.I))
FX_TRANSITION_NOT_RE = _tunable("FX_TRANSITION_NOT_RE", _re.compile(r"(?<![a-z])pads?(?![a-z])", _re.I))
FX_TRANSITION_FROM = _tunable("FX_TRANSITION_FROM", {"PADS", "SYNTH", "BLIPS", "SUB", "VOX", "STABS"})
# Hot one-shots: peak normalization leaves a dense, square-ish hit far louder than its
# neighbors (a synthetic clap or closed hat can sit far above its category), which
# jumps out when you swap samples live. A one-shot's RMS is capped here: about a
# category's typical median RMS + 6 dB. Gain only goes down.
ONESHOT_RMS_CEIL_DB = _tunable("ONESHOT_RMS_CEIL_DB", {
    "KICKS": -6.0, "SNARES": -10.5, "CLAPS": -12.0, "HATS": -12.0, "CYMBALS": -16.0,
    "TOMS": -9.5, "PERC": -11.0, "FX": -10.0, "BLIPS": -9.0, "SUB": -7.0, "SYNTH": -7.0,
    "STABS": -10.0, "PADS": -9.0, "VOX": -8.0, "PIANO": -12.5, "ACOUSTIC": -12.5,
})
ONESHOT_RMS_CEIL_DB = _tax.add_values(ONESHOT_RMS_CEIL_DB, "rms_ceiling_db")
# A folder under this many files merges into the nearest folder of its band, so no folder
# is a handful of files. Tempo-banded categories keep their tempo folders.
FOLDER_MIN_FILES = _tunable("FOLDER_MIN_FILES", 15)
# Sample sets kept in one folder (clustering alone splits numbered / velocity sets
# across folders): after allocation, a set's stragglers join the folder holding most
# of it, within the same band, while that folder has room and the one they leave keeps
# FOLDER_MIN_FILES. Not where folders are led by tempo or register.
# Additive builds (a release built on a base release): a category may grow by ADD_ALLOWANCE of its
# budget over the base release; a new group joins an existing folder when their CLAP
# centroids are this close (and the folder has room). verify allows base + allowance.
ADD_ALLOWANCE = _tunable("ADD_ALLOWANCE", 0.15)
ADD_ROUTE_MIN = _tunable("ADD_ROUTE_MIN", 0.80)
# Levelling in melodic folders: loudness is the
# loudest LEVEL_WINDOW_MS of a file (whole-file RMS let long tails read quiet), and a file
# more than LEVEL_TOL_DB from its folder's median moves to that edge: turned down freely,
# turned up only as far as the -1 dBFS peak ceiling allows. Drums stay peak-normalized.
LEVEL_CATS = _tunable("LEVEL_CATS", {"SUB", "SYNTH", "STABS", "PADS", "VOX", "PIANO", "ACOUSTIC"})
LEVEL_WINDOW_MS = _tunable("LEVEL_WINDOW_MS", 75)
LEVEL_TOL_DB = _tunable("LEVEL_TOL_DB", 3.0)
LEVEL_MAX_UP_DB = _tunable("LEVEL_MAX_UP_DB", 6.0)          # a lift never exceeds this (a quiet take's noise floor comes up too)
# Off-grid drum loops: a loop whose hits all sit
# ROTATE_MIN_MS-ROTATE_MAX_MS off the 16th grid (cut early or late) is rotated by that
# offset. The length stays the same, nothing is quantized; fills, rolls and swung loops
# are left alone.
ROTATE_MIN_MS = _tunable("ROTATE_MIN_MS", 12.0)
ROTATE_MAX_MS = _tunable("ROTATE_MAX_MS", 45.0)
ROTATE_MAX_SIXTEENTH = _tunable("ROTATE_MAX_SIXTEENTH", 0.3)     # ...and at most this share of a 16th (past it, "late" may be "early")
ROTATE_SEAM_MS = _tunable("ROTATE_SEAM_MS", 1.5)           # a dip this long either side of the old start/end join
ROTATE_AGREE = _tunable("ROTATE_AGREE", 0.75)           # share of hits within ROTATE_TOL_MS of the offset
ROTATE_ON_GRID_MAX = _tunable("ROTATE_ON_GRID_MAX", 0.25)     # ...and fewer than this share already on the grid (else it's feel)
ROTATE_TOL_MS = _tunable("ROTATE_TOL_MS", 8.0)
ROTATE_QUIET_HEAD_DB = _tunable("ROTATE_QUIET_HEAD_DB", 24.0)    # a lead-in this far under the downbeat is silence: rotate to the downbeat
ROTATE_PREROLL_MS = _tunable("ROTATE_PREROLL_MS", 1.0)       # a late start keeps this much before the downbeat's attack
ATTACK_START_FRAC = _tunable("ATTACK_START_FRAC", 0.1)       # a hit starts where its envelope, walking back, falls under this share of its peak
ROTATE_SKIP_RE = _tunable("ROTATE_SKIP_RE", _re.compile(r"(?<![a-z])(fill|fills|roll|rolls|intro|outro|reverse|rev)(?![a-z])", _re.I))
SET_TOGETHER_SKIP = _tunable("SET_TOGETHER_SKIP", _tax.with_role("no_sets"))

# Drum machines a drum folder may be named for, when at least MACHINE_SHARE of its files
# come from one (path match): "kick-909-punchy" is how you pick drums for acid and house.
MACHINE_SHARE = _tunable("MACHINE_SHARE", 0.5)
DRUM_MACHINES = _tunable("DRUM_MACHINES", [
    ("909", r"(?<![0-9])909(?![0-9])"), ("808", r"(?<![0-9])808(?![0-9])"),
    ("707", r"(?<![0-9])(?:707|727)(?![0-9])"), ("606", r"(?<![0-9])606(?![0-9])"),
    ("cr78", r"cr-?78"), ("linndrum", r"linn|lindrum|(?<![a-z])lm-?1(?![0-9])"), ("dmx", r"(?<![a-z])dmx"),
    ("drumulator", r"drumulat"), ("drumtraks", r"drumtra[xk]"), ("sp1200", r"sp-?1200"),
    ("mpc60", r"mpc ?60(?![0-9])"), ("mpc3000", r"mpc ?3000|mpc3k"), ("simmons", r"simmons|sds ?v|sdsv|sds ?800"),
    ("synare", r"synare"), ("rytm", "|".join([r"analog ?rytm", *MACHINE_ALIASES.get("rytm", ())])),
    ("505", r"(?<![0-9])505(?![0-9])"), ("626", r"(?<![0-9])626(?![0-9])"),
])
# ...plus machines only a library's own pack names give away (a library overlay)
DRUM_MACHINES_EXTRA = _tunable("DRUM_MACHINES_EXTRA", [])   # library: [(label, regex)]
DRUM_MACHINES = [*DRUM_MACHINES, *(tuple(x) for x in DRUM_MACHINES_EXTRA)]
# (A device's bare name isn't enough where folders named for the device hold other sounds.
# The filename is checked before the folders, so a file named for one machine in a folder
# named for another is the file's machine.)
DRUM_MACHINE_CATS = _tunable("DRUM_MACHINE_CATS", {"KICKS", "SNARES", "CLAPS", "HATS", "CYMBALS", "TOMS", "PERC"})
# ...and a bass or synth folder for its synth, the same way (a folder mostly of 303-style
# lines is a "303" folder)
SYNTH_MACHINES = _tunable("SYNTH_MACHINES", [("303", "|".join([
    r"(?<![0-9])303(?![0-9])", *MACHINE_ALIASES.get("303", ()), r"(?<![a-z])tb-?3", r"(?<![a-z])td-?3", "x0x"]))])
SYNTH_MACHINE_CATS = _tunable("SYNTH_MACHINE_CATS", {"SUB", "SYNTH"})
LOOP_DIR_ONESHOT_MAX_S = _tunable("LOOP_DIR_ONESHOT_MAX_S", 4.0)

# Impulse responses are for loading into a convolution reverb, not for playing: an IR
# collection holds many short files that read as hits to the classifiers. A file under a
# folder named IRs / Impulse Responses / Impulses / Convolution Reverb is kept out of every
# category, Keeps included. Filenames alone
# don't decide it: an "...IR" suffix in a synth patch name and a drum hit named "Impulse"
# are not IRs.
IR_PATH_RE = _tunable("IR_PATH_RE", _re.compile(r"(?:^|/)(?:irs?|impulse ?responses?|impulses|convolution reverb[^/]*)/", _re.I))

# Preset and kit previews are demo renders of an instrument rack or drum kit (a
# "Previews" folder, or a preset file's rendered audio named "<preset>.adg.wav" or
# "<kit>.xpm.mp3"), not samples to play. DEMO_PATH exempts ACOUSTIC and
# PIANO, and a plural "[Previews]" folder isn't its \bpreview\b, so previews get their own
# rule. Blocked everywhere, Keeps included.
PREVIEW_PATH_RE = _tunable("PREVIEW_PATH_RE", _re.compile(
    r"ableton folder info/previews/|(?:^|/)\[?previews?\]?/"
    r"|\.(?:adg|adv|agr|alc|als|xpm|xpj)\.(?:wav|aif|aiff|mp3|flac|ogg)$", _re.I))

# Name-reserved sounds: a filename containing one of these substrings may only appear in
# the paired category (it is rejected everywhere else), but unlike NAME_OVERRIDES it is
# not force-pulled in, so the category's own gate still decides.
NAME_RESERVED = _tunable("NAME_RESERVED", [])   # library: [((substring, ...), category)]

# Overrides fix ROUTING; they never grant priority. A forced file still competes for
# its slot under the budget, and each rule may make up at most OVERRIDE_MAX_SHARE of
# the target's candidate pool (CLAP-spread subset; pool share tracks output share,
# as with the vendor cap), so a keyword matching a big new pack cannot flood its target.
# verify --quick warns on a breach.
OVERRIDE_MAX_SHARE = _tunable("OVERRIDE_MAX_SHARE", 0.10)

# Instrument-kind categories (PIANO, ACOUSTIC) are pack/name-selected and skip the
# vendor cap, so one pack could dominate. Each vendor/pack (two folders under
# the library root, so one vendor's two piano packs count separately) may supply
# at most this share of the candidate pool. verify --quick warns on a breach.
INSTRUMENT_PACK_MAX_SHARE = _tunable("INSTRUMENT_PACK_MAX_SHARE", 0.20)
