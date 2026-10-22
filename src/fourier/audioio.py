"""Reading audio: librosa.load through soundfile only.

librosa falls back to audioread when soundfile can't open a file. That fallback is slow, warns
about modules Python 3.13 deprecated (aifc, sunau, audioop) and decodes differently. Here a
file soundfile can't open raises soundfile's error instead, and the caller reports it as
unreadable. Every file soundfile opens decodes exactly as librosa.load(path) does: librosa
reads an open SoundFile through the same soundfile path.

A floating-point file can hold NaN or infinite values (a damaged file, or a crashed render);
no analysis can use them, so reading one raises NotFinite and the caller counts it as
unreadable too.
"""
from __future__ import annotations

NOT_FINITE = "holds NaN or infinite sample values: a damaged floating-point file"
FLOAT_SUBTYPES = ("FLOAT", "DOUBLE")


class NotFinite(ValueError):
    """Audio holding NaN or infinite values: unreadable for analysis."""

    def __init__(self, path=None):
        super().__init__(NOT_FINITE)
        self.path = path


def load(path, **kw):
    """(audio, sample rate), as librosa.load(path, **kw), read through soundfile only. Raises
    NotFinite when the audio read holds NaN or infinite values."""
    import librosa
    import numpy as np
    import soundfile as sf
    try:
        with sf.SoundFile(str(path)) as f:
            y, sr = librosa.load(f, **kw)
    except librosa.util.exceptions.ParameterError as e:
        if "finite" in str(e):                # resampling checks the audio first
            raise NotFinite(path) from None
        raise
    if not np.isfinite(y).all():
        raise NotFinite(path)
    return y, sr


def not_finite(path, block: int = 1 << 18) -> bool:
    """Whether a file holds NaN or infinite values anywhere: a floating-point file is read in
    blocks; other files (whole numbers) can't hold them and aren't read. False when the file
    can't be opened (the caller's own read says why)."""
    import numpy as np
    import soundfile as sf
    try:
        with sf.SoundFile(str(path)) as f:
            if f.subtype not in FLOAT_SUBTYPES:
                return False
            for chunk in f.blocks(blocksize=block, dtype="float64", always_2d=True):
                if not np.isfinite(chunk).all():
                    return True
    except Exception:            # noqa: BLE001 (unreadable is the caller's to report)
        return False
    return False
