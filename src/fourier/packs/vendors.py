"""Which folder of a sample's library path is its vendor: what the per-vendor cap
(curate._cap_vendor_share, VENDOR_MAX_SHARE), the instrument categories' per-pack cap and
verify's vendor and pack checks count by.

The `vendors` knob (curate_config.VENDORS):

  auto           each library folder's layout decides (detect): the code's default, and what
                 `fourier setup` writes for a new config:
                   packs      vendor/pack/... folders: as first-folder
                   types      folders by sound type (Drums/Kicks, Loops, Bass, FX, ...): there is
                              no vendor in the path, so no vendor or pack cap
                   umbrella   one or two folders holding many packs (a sample store's download
                              folder, "Downloads"): the vendor is the folder below them, the pack
                              the one below that
                   flat       files with no folders: no vendor or pack cap
  first-folder   the first folder under the library folder is always the vendor, the first
                 two the pack (a library laid out vendor/pack/...; the default before 0.2)

`fourier doctor` and `fourier setup` print the layout they detect in one line. A sample
with no vendor (None) is never capped, so a type-organized or flat library isn't cut to
VENDOR_MAX_SHARE of one "vendor" that is really its whole library.
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

FIRST_FOLDER, AUTO = "first-folder", "auto"
MODES = (FIRST_FOLDER, AUTO)
PACKS, TYPES, UMBRELLA, FLAT = "packs", "types", "umbrella", "flat"

# a layout's thresholds: flat when this share of the files sit in the library folder itself;
# by sound type when this share of the files in folders sit under sound-type folders; an
# umbrella when at most two top folders hold this share of the files and each of them holds
# at least UMBRELLA_MIN_PACKS pack-like folders (not sound types)
FLAT_SHARE, TYPES_SHARE, UMBRELLA_SHARE, UMBRELLA_MIN_PACKS = 0.5, 0.6, 0.8, 4

# folder words that say what kind of sound a folder holds without naming a sound the path
# rules know (those count too: Kicks, Pads, FX, Loops, ...): a folder named only with these
# and the path rules' words is a sound-type folder
TYPE_WORDS = frozenset({
    "drum", "drums", "sample", "samples", "sound", "sounds", "instrument", "instruments",
    "melodic", "music", "musical", "misc", "miscellaneous", "other", "others", "one", "shot",
    "shots", "oneshot", "oneshots", "multisample", "multisamples", "acoustic", "electronic",
    "synth", "synths", "keys", "key", "guitar", "guitars", "piano", "pianos", "organ", "organs",
    "percussion", "percs", "perc", "vocals", "vocal", "vox", "voices", "foley", "fx", "sfx",
    "textures", "texture", "atmospheres", "drones", "hits", "hit", "and", "&", "wav", "wavs",
    "audio", "bass", "basses", "basslines", "leads", "lead", "pads", "stabs", "chords",
    "loops", "loop", "breaks", "beats", "grooves", "tops", "fills", "kits", "kit", "hi", "open",
    "closed", "cymbals", "effects", "processed", "layered", "construction", "full", "dry", "wet",
})


def mode() -> str:
    from .curate_config import VENDORS
    return VENDORS


def _words(name: str) -> list[str]:
    from ..metadata.shadow import words_of
    return re.findall(r"[a-z0-9&]+", words_of(name).lower())


def is_type_folder(name: str) -> bool:
    """A folder named for a kind of sound ("Kicks", "One Shots", "Drum Loops", "808s", "FX"),
    not a vendor or a pack: every word is a sound-type word, or the path rules give it a
    label."""
    from ..metadata.shadow import _matches
    ws = [w for w in _words(name) if not w.isdigit()]
    return bool(ws) and all(w in TYPE_WORDS or _matches(w, True) for w in ws)


def detect(rels) -> dict:
    """The layout of one library folder from its files' library paths: {"layout": packs |
    types | umbrella | flat, "umbrellas": [top folders] (umbrella), "files": n}."""
    rels = [r for r in rels if r]
    n = len(rels)
    out = {"layout": PACKS, "umbrellas": [], "files": n}
    if not n:
        return out
    parts = [r.split("/") for r in rels]
    top = Counter(p[0] if len(p) > 1 else None for p in parts)
    if top[None] >= FLAT_SHARE * n:
        out["layout"] = FLAT
        return out
    in_folders = n - top[None]
    typed = sum(k for f, k in top.items() if f is not None and is_type_folder(f))
    if typed >= TYPES_SHARE * in_folders:
        out["layout"] = TYPES
        return out
    big = [f for f, _ in top.most_common(2) if f is not None]
    held = sum(top[f] for f in big)
    if held >= UMBRELLA_SHARE * in_folders:
        subs = defaultdict(set)
        for p in parts:
            if len(p) > 2 and p[0] in big:
                subs[p[0]].add(p[1])
        umbrellas = [f for f in big if sum(1 for s in subs[f] if not is_type_folder(s)) >= UMBRELLA_MIN_PACKS]
        if umbrellas and sum(top[f] for f in umbrellas) >= UMBRELLA_SHARE * in_folders:
            out.update(layout=UMBRELLA, umbrellas=sorted(umbrellas))
    return out


_LAYOUTS: dict = {}        # library folder (or "") -> detect()'s result, for this process


def prepare(session) -> dict:
    """Detect each library folder's layout from the database (auto mode only; first-folder
    needs nothing). Called by a build, verify and why before they ask for a vendor."""
    if mode() != AUTO:
        return {}
    if session is None:                      # a caller with no session: the database's own
        return ensure()
    from sqlalchemy import text

    from ..places import library_rel, library_root_of
    key = (str(session.get_bind().url),
           session.execute(text("SELECT COUNT(*), MAX(id) FROM samples")).first()[:])
    if _LAYOUTS.get("_key") == key:
        return _LAYOUTS
    from ..metadata.rows import outside_library
    other = outside_library(session)
    by_root = defaultdict(list)
    for sid, rel, path in session.execute(text("SELECT id, rel_path, path FROM samples")):
        if sid in other:                     # another library's sample (another config's)
            continue
        root = library_root_of(path or "") or ""
        by_root[root].append(rel or library_rel(path or ""))
    _LAYOUTS.clear()
    _LAYOUTS.update({root: detect(rels) for root, rels in by_root.items()})
    _LAYOUTS["_key"] = key
    return _LAYOUTS


def ensure(session=None) -> dict:
    """prepare() once in this process (auto mode only), with this session or one of its own;
    nothing when the database can't be opened (a vendor is then the first folder)."""
    if mode() != AUTO or _LAYOUTS:
        return _LAYOUTS
    if session is not None:
        return prepare(session)
    try:
        from ..db.session import db_exists, session_scope
        if not db_exists():
            return _LAYOUTS
        with session_scope() as s:
            return prepare(s)
    except Exception:
        return _LAYOUTS


def layouts(session, roots) -> dict:
    """Each library folder's layout as a build detects it, for doctor and setup (one answer
    for both): from this library's samples in the database once it holds any (prepare;
    another library's samples left out), else from the files on disk (detect_folders)."""
    if session is not None:
        try:
            forget()
            got = prepare(session)
            if any(k != "_key" for k in got):
                return got
        except Exception:
            pass
    return detect_folders(roots)


def forget() -> None:
    _LAYOUTS.clear()


def _layout_for(path: str | None) -> dict:
    if not _LAYOUTS:
        return {"layout": PACKS, "umbrellas": []}
    from ..places import library_root_of
    root = library_root_of(path or "") or ""
    got = _LAYOUTS.get(root)
    if got is None:
        roots = [v for k, v in _LAYOUTS.items() if k != "_key"]
        got = roots[0] if len(roots) == 1 else {"layout": PACKS, "umbrellas": []}
    return got


def vendor_of(rel: str | None, path: str | None = None):
    """The vendor a sample is capped under, or None (no cap). first-folder: the first folder
    of its library path ("?" for none), as always."""
    if mode() != AUTO:
        return rel.split("/")[0] if rel else "?"
    lay = _layout_for(path)
    segs = [s for s in (rel or "").split("/") if s]
    if lay["layout"] in (TYPES, FLAT) or len(segs) < 2:
        return None
    if lay["layout"] == UMBRELLA and segs[0] in lay["umbrellas"]:
        return "/".join(segs[:2]) if len(segs) > 2 else None
    return segs[0]


def pack_key(path: str | None):
    """The vendor/pack the instrument categories' per-pack cap counts by (curate's
    INSTRUMENT_PACK_MAX_SHARE, verify's pack check), or None (no cap). first-folder: the
    first two folders of the library path, as always."""
    from ..places import library_rel
    segs = [s for s in library_rel(path or "").split("/") if s]
    if mode() != AUTO:
        return "/".join(segs[:2]) or "?"
    lay = _layout_for(path)
    if lay["layout"] in (TYPES, FLAT) or len(segs) < 2:
        return None
    if lay["layout"] == UMBRELLA and segs[0] in lay["umbrellas"]:
        return "/".join(segs[:3]) if len(segs) > 3 else None
    return "/".join(segs[:2]) if len(segs) > 2 else None


def describe(layouts: dict) -> str:
    """One line on what was detected (doctor, setup)."""
    roots = {k: v for k, v in layouts.items() if k != "_key"}
    if not roots:
        return "no samples scanned yet"
    words = {PACKS: "vendor/pack folders: the first folder is the vendor",
             TYPES: "folders by sound type: no vendor in the path, no vendor cap",
             UMBRELLA: "an umbrella folder of packs: the folder below it is the vendor",
             FLAT: "files with no folders: no vendor cap"}
    parts = []
    for root, lay in sorted(roots.items()):
        name = os.path.basename(root.rstrip("/")) or "library"
        extra = f" ({', '.join(lay['umbrellas'])})" if lay.get("umbrellas") else ""
        parts.append(f"{name}: {words[lay['layout']]}{extra}")
    return "; ".join(parts)


def detect_folders(roots, cap: int = 20000) -> dict:
    """{library folder: detect()} from the files on disk (setup, doctor before a scan): the
    first `cap` audio files under each folder, as the walk finds them."""
    from ..cli.setup import survey
    out = {}
    for r in roots:
        rels: list = []
        survey([r], cap, rels=rels)
        out[str(r)] = detect(rels)
    return out
