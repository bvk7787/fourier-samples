"""The key a loop's pack states for it: from its own file name, else the nearest folder of its
library path that names one, else the files beside it when every one of them that names a key
names the same one. Never from the audio: on loops whose names state a key, the detectors
tried (Fourier's own, essentia's, madmom's CNN, and one trained on the library) agreed with
the stated key on at most 56% of them (docs/curation.md, section 13).

Used for phrases.csv (packs/sets.py).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_SPELL = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E",
          "E#": "F", "B#": "C"}
# a key as its own token: a note letter with an accidental and/or a mode word ("Am", "F#",
# "C#min", "Eb major", "Bb_minor"); a bare letter ("Loop A") says nothing
_KEY = re.compile(r"(?<![A-Za-z0-9#])([A-G])(#|b|s)?[ _-]?(major|minor|maj|min|m)?(?![A-Za-z0-9#])")
AUDIO_EXTS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg")
SIBLINGS_MIN = 2           # files beside it that state a key, all the same, before it borrows theirs


def name_key(name: str) -> tuple[int, str | None] | None:
    """(pitch class, "major" | "minor" | None) a name states, or None: no key token, or two
    tokens naming different keys."""
    found = set()
    for m in _KEY.finditer((name or "").replace("_", " ")):
        letter, acc, mode = m.group(1), m.group(2), (m.group(3) or "").lower()
        if not acc and not mode:
            continue
        note = letter + {"#": "#", "s": "#", "b": "b"}.get(acc or "", "")
        note = _SPELL.get(note, note)
        if note not in NOTES:
            continue
        found.add((NOTES.index(note), "minor" if mode in ("m", "min", "minor") else
                   "major" if mode in ("maj", "major") else None))
    if len({pc for pc, _ in found}) != 1:
        return None
    modes = {md for _, md in found if md}
    if len(modes) > 1:
        return None
    return (next(iter(found))[0], modes.pop() if modes else None)


def label(key: tuple[int, str | None] | None) -> str:
    """"A minor", "C# major", or "A" when the mode isn't stated; "" for none."""
    if not key:
        return ""
    pc, mode = key
    return f"{NOTES[pc]} {mode}" if mode else NOTES[pc]


@lru_cache(maxsize=4096)
def _folder_keys(folder: str) -> tuple:
    """The keys the audio files in a folder state in their names (one per file)."""
    try:
        names = os.listdir(folder)
    except OSError:
        return ()
    return tuple(k for n in names if n.lower().endswith(AUDIO_EXTS) and not n.startswith(".")
                 for k in [name_key(os.path.splitext(n)[0])] if k)


def stated_key(src: str, rel: str | None = None) -> tuple[tuple[int, str | None] | None, str]:
    """(key, where it came from: "name" | "folder" | "siblings" | "") for a source file.
    rel: its path under its library folder (the folders searched; the top one, the vendor's,
    never counts)."""
    own = name_key(os.path.splitext(os.path.basename(src))[0])
    if own:
        return own, "name"
    for seg in reversed((rel or "").split("/")[1:-1]):
        k = name_key(seg)
        if k:
            return k, "folder"
    keys = list(_folder_keys(os.path.dirname(src)))
    if len(keys) >= SIBLINGS_MIN and len({pc for pc, _ in keys}) == 1:
        modes = {md for _, md in keys if md}
        if len(modes) <= 1:
            return (keys[0][0], modes.pop() if modes else None), "siblings"
    return None, ""


def forget() -> None:
    """Drop the folder listings read so far (a test that changes a folder)."""
    _folder_keys.cache_clear()
