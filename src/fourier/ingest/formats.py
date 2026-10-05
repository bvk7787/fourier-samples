"""The audio file extensions Fourier reads, in one place: the library walk (`fourier tools
scan`), setup's survey and doctor, Live's tags and `fourier tools import-folder`.

AUDIO_EXTS names every format a sample library ships in. A format counts as readable when the
installed soundfile (libsndfile) reads it, since every later step (the header, the analysis,
the export, which writes WAV) reads through soundfile: `readable_exts()`. The rest are known
audio Fourier can't read here (AAC in .m4a, for one: libsndfile has no AAC decoder); the walk
skips them and says how many by extension, and a preparer (fourier/preparers.py) or a
conversion to WAV brings them in.
"""
from __future__ import annotations

from functools import lru_cache

AUDIO_EXTS = (".wav", ".aif", ".aiff", ".aifc", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".caf")

# the libsndfile major format (and, where one container holds several codecs, the subtype)
# each extension needs; an extension with none here isn't read by soundfile
_SF_FORMAT = {".wav": ("WAV", None), ".aif": ("AIFF", None), ".aiff": ("AIFF", None),
              ".aifc": ("AIFF", None), ".flac": ("FLAC", None), ".mp3": ("MP3", None),
              ".ogg": ("OGG", "VORBIS"), ".opus": ("OGG", "OPUS"), ".caf": ("CAF", None)}

# copied into a library folder as they are by import-folder; the other readable formats are
# decoded to WAV there (".oga" is Ogg audio under another name)
COPY_EXTS = frozenset({".wav", ".aif", ".aiff", ".aifc", ".flac"})
EXTRA_CONVERT_EXTS = frozenset({".oga"})

# Live's tag step matches Live's index to the samples by path, by folder and name, and by
# a file's stem when only one tag set has it. With Sononym it keeps this list: another
# format in Live's index can make a stem ambiguous and move tags between Sononym's samples.
LIVE_TAG_EXTS = (".aif", ".aiff", ".wav", ".ogg", ".flac", ".mp3")


@lru_cache(maxsize=1)
def readable_exts() -> tuple[str, ...]:
    """The AUDIO_EXTS the installed soundfile reads, in AUDIO_EXTS' order."""
    try:
        import soundfile as sf
        formats = set(sf.available_formats())
        subtypes = {f: set(sf.available_subtypes(f)) for f in formats}
    except Exception:
        return ()
    out = []
    for ext in AUDIO_EXTS:
        fmt, sub = _SF_FORMAT.get(ext, (None, None))
        if fmt in formats and (sub is None or sub in subtypes.get(fmt, ())):
            out.append(ext)
    return tuple(out)


def unreadable_exts() -> tuple[str, ...]:
    """The AUDIO_EXTS soundfile doesn't read here: known audio the walk skips and counts."""
    have = set(readable_exts())
    return tuple(e for e in AUDIO_EXTS if e not in have)


def convert_exts() -> frozenset:
    """What import-folder decodes to WAV: the readable formats it doesn't copy as they are."""
    return frozenset(e for e in readable_exts() if e not in COPY_EXTS) | EXTRA_CONVERT_EXTS


def is_audio(name: str) -> bool:
    """Whether a file name has one of the readable extensions."""
    n = name.lower()
    return any(n.endswith(e) for e in readable_exts())
