"""Why a build left each candidate out, kept beside the build for `fourier why`.

A build writes one small JSON file per category into its home
(`$FOURIER_HOME/why/<master key>/<CATEGORY>.json`, packs/curate.left_out_doc): how many
candidates the category found and kept, its minimum, and for each candidate it left out
the step that did it (the per-vendor share, a near-duplicate and which file it repeats, a
CLAP gate, the budget, ...), by sample id. Nothing of it goes into the master, its manifest
or a release, so a build's files are the same with or without it; a file that can't be
written is skipped, never a failed build.

The key is the master the build is for: `<master>.next` (a whole build on its way in) and
the master itself share one, so `fourier why` finds the latest build's reasons.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# stats key -> how a reason reads (a count: "5 near-duplicates"; a sample: "a near-duplicate");
# {cat} is the category
FILTER_WORDS = {
    "near_dup": "near-duplicates",
    "vendor_capped": "over the per-vendor share",
    "gated_out": "failed the category's CLAP gate",
    "too_harmonic": "too harmonic for {cat}",
    "tempo_range": "outside {cat}'s tempo range",
    "excluded": "sounded like another category",
    "twins": "duplicate copies",
    "too_long": "over the length cap",
    "duration": "outside {cat}'s length window",
    "note_gated": "outside the note range",
    "loop_gated": "loops in a one-shot category",
    "chain_gated": "sample chains",
    "nondrum_gated": "loops without drums",
    "no_tempo": "loops with no tempo",
    "pack_homed": "from packs with another home",
    "wave_out": "single-cycle waves",
    "loop_folder": "long files from loop folders",
    "bpm_named": "tempo-named phrases",
    "not_stab_named": "not named as stabs",
    "named_elsewhere": "named as another drum",
    "override_capped": "over a name rule's share",
    "dir_capped": "over the per-folder cap",
    "pack_capped": "over the per-pack share",
    "instr_capped": "over an instrument pack's cap",
    "choir_capped": "over the choir share",
    "routed_capped": "over the routed share",
    "soft_layers": "soft velocity layers",
    "style_excluded": "off-style loops (DRUMLOOP_STYLE_EXCLUDE)",
    "too_quiet": "too quiet to export",
    "mirror_copies": "mirror-folder copies",
    "clap_readmitted": "gated loops readmitted to reach the minimum",
    "too_harmonic_readmitted": "too-harmonic loops readmitted to reach the minimum",
    "many_onsets": "not single notes (many onsets)",
    "chord_not_stab": "chords too long or metered for a stab",
    "not_one_pitch": "not one clear pitch",
}
# the why dict's key (curate._readmit_gated) for candidates the floor readmitted past a check
# after the CLAP gate: {sample id: [reason, detail]}; a doc's "readmitted"
READMITTED = "_readmitted"


def master_key(out_dir) -> str:
    """One key for a master and its `.next` build folder."""
    p = os.path.abspath(os.path.expanduser(str(out_dir))).rstrip("/")
    if p.endswith(".next"):
        p = p[:-len(".next")]
    return hashlib.sha1(p.encode()).hexdigest()[:16]


def folder(out_dir) -> Path:
    from ..paths import home_path
    return home_path("why", master_key(out_dir))


def write(out_dir, category: str, doc: dict) -> None:
    """Write one category's reasons (atomically); skipped when the home can't be written."""
    try:
        d = folder(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{category}.{os.getpid()}.json"
        tmp.write_text(json.dumps(doc, separators=(",", ":"), default=str))
        os.replace(tmp, d / f"{category}.json")
    except OSError:
        pass


def read(master_dir, category: str) -> dict | None:
    try:
        return json.loads((folder(master_dir) / f"{category}.json").read_text())
    except (OSError, ValueError):
        return None


def reason_for(doc: dict, sample_id: int) -> tuple[str, object] | None:
    """(reason, detail) a category left this sample out for, or None."""
    sid = str(sample_id)
    for reason, ids in (doc.get("out") or {}).items():
        if sid in ids:
            return reason, ids[sid]
    return None


def gate_result(doc: dict, sample_id) -> str | None:
    """How a sample fared at its category's CLAP gate (the build's scores, doc["gate"]):
    passed, failed (score below the minimum, or not above the anti-prompts' score), or
    readmitted below it to reach the category's minimum. None when it never reached it."""
    g = (doc.get("gate") or {}).get(str(sample_id))
    if not g:
        return None
    lo = float(doc.get("gate_min") or 0.0)
    sb, sa = float(g[0]), float(g[1])
    why = (f"score {sb:.2f} < {lo:.2f}" if sb < lo else
           f"score {sb:.2f} not above its anti-prompts' {sa:.2f}" if sb <= sa else
           f"score {sb:.2f} >= {lo:.2f}, anti-prompts {sa:.2f}")
    if len(g) > 2 and g[2]:
        return (f"readmitted below its CLAP gate ({why}) to reach the minimum of "
                f"{doc.get('need', '?')}: a loop by its own name, folder or audio")
    if sb > sa and sb >= lo:
        return f"passed its CLAP gate ({why})"
    return f"failed its CLAP gate ({why})"


def readmitted_result(doc: dict, sample_id) -> str | None:
    """How a sample in the master got past a check after its category's CLAP gate: readmitted
    to reach the minimum (doc["readmitted"]). None when it wasn't."""
    got = (doc.get("readmitted") or {}).get(str(sample_id))
    if not got:
        return None
    reason, detail = got
    return (f"readmitted past {gate_line(reason, detail, doc)} to reach the minimum of "
            f"{doc.get('need', '?')}: a loop by its own name, folder or audio")


def gate_line(reason: str, detail, doc: dict) -> str:
    """A check after the CLAP gate, with the file's value and the threshold."""
    cat = doc.get("category", "?")
    if reason == "too_harmonic" and isinstance(detail, list) and len(detail) == 2:
        return f"{cat}'s harmonicity gate (harmonicity {detail[0]:.2f} > {detail[1]:.2f}: too tonal)"
    if reason in ("too_harmonic", "harmonic"):
        return f"{cat}'s harmonicity gate (too tonal)"
    if reason == "tempo_range" and detail:
        got = f"; its tempo {detail[2]:g}" if len(detail) > 2 and detail[2] else ""
        return f"{cat}'s tempo range ({detail[0]} to {detail[1]} BPM{got})"
    return f"{cat}'s {reason} check"


def describe(reason: str, detail, doc: dict, name_of=lambda sid: str(sid), place_of=None) -> str:
    """One line for a left-out sample: what left it out of doc's category. `place_of`, a sample
    id -> where the master holds it ("KICKS/punchy/Kick 01.wav") or None."""
    cat = doc.get("category", "?")
    if reason == "near_dup":
        where = place_of(detail) if place_of else None
        if place_of is None:
            return f"a near-duplicate of {name_of(detail)} (CLAP), which {cat} kept"
        if where:
            return f"a near-duplicate of {name_of(detail)} (CLAP), which {cat} kept (in the master: {where})"
        return (f"a near-duplicate of {name_of(detail)} (CLAP), which {cat} kept as a candidate; "
                f"`fourier why` on it says where it went")
    if reason == "twins" and detail is not None:
        where = place_of(detail) if place_of else None
        if where:
            return f"a byte-identical copy of {name_of(detail)}, which the master holds ({where})"
        return f"a byte-identical copy of {name_of(detail)}, which {cat} kept as a candidate"
    if reason == "too_harmonic":
        return f"failed {gate_line(reason, detail, doc)}"
    if reason == "tempo_range" and detail:
        return f"its tempo is outside {gate_line(reason, detail, doc)}"
    if reason == "vendor_capped":
        return (f"over the per-vendor share: {detail} has more than its share of {cat}'s "
                f"candidates (VENDOR_MAX_SHARE)")
    if reason == "gated_out":         # ("harmonic": a build from before the harmonicity gate's own reason)
        gate = {"clap": f"{cat}'s CLAP gate (it sounds less like {cat} than its anti-prompts)",
                "harmonic": f"{cat}'s harmonicity gate (too tonal)"}.get(detail, f"{cat}'s gate")
        return f"failed {gate}"
    if reason == "over_budget":
        return (f"over budget: {cat} kept {doc.get('files', '?')} of its {doc.get('kept', '?')} "
                f"candidates (budget {doc.get('budget', '?')}); the folders' spread picks chose others")
    if reason == "too_long" and detail:
        return f"over {cat}'s length cap ({detail} s)"
    if reason == "note_gated" and detail:
        return f"its note is outside {cat}'s range (MIDI {detail[0]} to {detail[1]})"
    if reason == "duration" and detail:
        return f"its length is outside {cat}'s window ({detail[0]} to {detail[1]} s)"
    if reason == "named_elsewhere" and detail:
        return f"named as a {detail} sound"
    if reason == "pack_homed" and detail:
        return f"its pack's home is {detail} (PACK_HOME)"
    if reason == "instr_capped" and detail:
        return f"over the cap for instrument pack {detail}"
    if reason == "many_onsets" and detail:
        return (f"not a single note: {detail[0]:g} onsets, over {cat}'s {detail[1]:g} (a phrase, an arp "
                f"or a loop)")
    if reason == "chord_not_stab":
        return f"a chord, and {cat} takes a chord only as a stab (one hit, at most a few seconds)"
    if reason == "not_one_pitch":
        return f"no single clear pitch ({cat} takes single notes, or chords as stabs)"
    if reason == "style_excluded":
        said = f" (\"{detail}\")" if detail else ""
        return (f"an off-style loop: its path names a style or folder this preset leaves out of {cat}"
                f"{said}, DRUMLOOP_STYLE_EXCLUDE; a preset for that style (trap, balanced) takes it")
    if reason == "too_quiet":
        lvl = f" (RMS under {detail:g} dBFS)" if isinstance(detail, (int, float)) else ""
        return (f"too quiet: near-silent once trimmed{lvl}, so it was skipped at export; a louder "
                f"take or a normalized copy would be picked")
    if reason == "soft_layers":
        lvl = f" (peak {detail:g} dBFS)" if isinstance(detail, (int, float)) else ""
        return f"too quiet: a soft velocity layer{lvl} of a multisample whose louder take is a candidate"
    words = FILTER_WORDS.get(reason)
    return f"left out by {reason}" + (f" ({words})" if words else "")
