"""
Musical key estimation using Krumhansl-Schmuckler tonal hierarchy profile matching.

No new dependencies — uses only librosa (already a core dep) and numpy.

Usage::

    from fourier.analysis.key_detection import estimate_key

    result = estimate_key("/path/to/sample.wav")
    # {"detected_key": "A minor", "key_confidence": 0.82}
    # {"detected_key": None, "key_confidence": 0.31}  ← noisy/unpitched sample
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler tonal hierarchy profiles (1982)
# Root = C; rotate by N semitones to get other roots.
# ---------------------------------------------------------------------------
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                   2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                   2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Minimum Pearson r to report a key (below this → sample is too noisy/unpitched).
# White noise typically scores 0.5-0.65 due to KS profile structure; 0.7 keeps
# only genuinely tonal content.  The CLI further guards with is_pitched=1 filter.
_CONFIDENCE_THRESHOLD = 0.7


def _abbreviate_key(key: str) -> str:
    """'C# minor' → 'C#m',  'F major' → 'F'"""
    note, mode = key.rsplit(" ", 1)
    return note + ("m" if mode == "minor" else "")


def estimate_key(
    path: str | Path,
    sr: int = 22050,
    max_duration_s: float = 30.0,
) -> dict:
    """
    Estimate the musical key of an audio file.

    Uses ``librosa.feature.chroma_cqt`` to compute a 12-bin pitch-class
    distribution, then correlates against Krumhansl-Schmuckler major/minor
    profiles for all 24 keys.  The best Pearson-r match is returned.

    Args:
        path:           audio file path
        sr:             analysis sample rate (22050 is sufficient for chroma)
        max_duration_s: cap analysis to first N seconds (30s default — long enough
                        for pads/loops, fast enough for large batches)

    Returns:
        dict with:
            detected_key   str | None  — e.g. "C# minor" / "F major"; None if unpitched
            key_confidence float       — Pearson r of best match (0.0–1.0)
    """
    import librosa

    try:
        from ..audioio import load as load_audio
        y, _ = load_audio(path, sr=sr, mono=True, duration=max_duration_s)
    except Exception as exc:
        log.warning("estimate_key: failed to load %s: %s", Path(path).name, exc)
        return {"detected_key": None, "key_confidence": 0.0}

    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    except Exception as exc:
        log.warning("estimate_key: chroma failed for %s: %s", Path(path).name, exc)
        return {"detected_key": None, "key_confidence": 0.0}

    mean_chroma = chroma.mean(axis=1)                       # (12,) pitch-class weights
    denom = mean_chroma.sum()
    if denom < 1e-9:
        # Essentially silent
        return {"detected_key": None, "key_confidence": 0.0}
    mean_chroma = mean_chroma / denom

    best_r: float = -1.0
    best_key: str | None = None

    for root in range(12):
        for mode_name, profile in (("major", _MAJOR), ("minor", _MINOR)):
            rotated = np.roll(profile, root)
            r = float(np.corrcoef(mean_chroma, rotated)[0, 1])
            if r > best_r:
                best_r = r
                best_key = f"{_NOTES[root]} {mode_name}"

    if best_r < _CONFIDENCE_THRESHOLD:
        return {"detected_key": None, "key_confidence": round(best_r, 3)}

    return {"detected_key": best_key, "key_confidence": round(best_r, 3)}
