"""Reading audio: librosa.load through soundfile only.

librosa falls back to audioread when soundfile can't open a file. That fallback is slow, warns
about modules Python 3.13 deprecated (aifc, sunau, audioop) and decodes differently. Here a
file soundfile can't open raises soundfile's error instead, and the caller reports it as
unreadable. Every file soundfile opens decodes exactly as librosa.load(path) does: librosa
reads an open SoundFile through the same soundfile path.
"""
from __future__ import annotations


def load(path, **kw):
    """(audio, sample rate), as librosa.load(path, **kw), read through soundfile only."""
    import librosa
    import soundfile as sf
    with sf.SoundFile(str(path)) as f:
        return librosa.load(f, **kw)
