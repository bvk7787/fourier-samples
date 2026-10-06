"""
loop_trim.py — Non-destructive trailing-silence detection for loop samples.

Computes the optimal trim endpoint for a loop, storing it in
SampleFeatures.trim_end_s. The original file is never modified; the trim is
applied on-the-fly in _convert_and_copy() during pack export, and caps a
waveform preview.

Algorithm
---------
1. Load audio at 22 050 Hz (mono).
2. Compute a frame-level RMS envelope (hop = 512 ≈ 23 ms resolution).
3. Find the last RMS frame above threshold_db (default -60 dB).
4. Convert to sample index; bail if trailing silence < min_silence_s.
5. Snap the trim point to the nearest zero-crossing within ±20 ms to avoid
   a click at the loop seam.
6. Score loop-seam smoothness via cross-correlation of the last 20 ms against
   the first 20 ms of the file.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

_SR = 22_050           # analysis sample rate (matches librosa_features.py)
_HOP_LENGTH = 512      # RMS hop — 23 ms frame resolution at 22 050 Hz
_SEAM_WINDOW_S = 0.02  # ±20 ms zero-crossing search + seam confidence window
_DEFAULT_THRESHOLD_DB = -60.0
_DEFAULT_MIN_SILENCE_S = 0.05  # don't bother trimming < 50 ms of trailing silence


# ── internal helpers ─────────────────────────────────────────────────────────

def _find_trim_point(
    y: np.ndarray,
    sr: int,
    threshold_db: float,
    min_silence_s: float,
    hop_length: int = _HOP_LENGTH,
) -> tuple[int | None, float]:
    """
    Locate the sample index of the last non-silent frame and the trailing silence.

    Returns:
        (trim_sample_idx, trailing_silence_s)
        trim_sample_idx is None when no trim is warranted.
    """
    try:
        import librosa
    except ImportError:
        return None, 0.0

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if rms.max() == 0:
        return None, 0.0

    rms_db = librosa.amplitude_to_db(rms, ref=rms.max())
    above = np.where(rms_db > threshold_db)[0]

    if len(above) == 0:
        return None, float(len(y)) / sr

    last_active_frame = int(above[-1])
    # The last active sample is the *end* of that hop frame
    active_sample = min(last_active_frame * hop_length + hop_length, len(y) - 1)

    trailing_silence_s = (len(y) - active_sample) / sr
    if trailing_silence_s < min_silence_s:
        return None, trailing_silence_s

    # Snap to the nearest zero crossing within ±_SEAM_WINDOW_S
    try:
        import librosa as _lib
        window = int(_SEAM_WINDOW_S * sr)
        search_start = max(0, active_sample - window)
        search_end   = min(len(y), active_sample + window)
        segment = y[search_start:search_end]
        zc = _lib.zero_crossings(segment, pad=False)
        zc_idx = np.where(zc)[0]
        if len(zc_idx) > 0:
            best = zc_idx[np.argmin(np.abs(zc_idx - (active_sample - search_start)))]
            trim_sample = search_start + int(best)
        else:
            trim_sample = active_sample
    except Exception:
        trim_sample = active_sample

    return trim_sample, trailing_silence_s


def _compute_seam_confidence(
    y: np.ndarray,
    trim_sample: int,
    sr: int,
    active_sample: int,
) -> float:
    """
    Score how smoothly the loop seam will sound after trimming.

    Composite score in [0, 1]:
      • 50 % — zero-crossing proximity  (1 = trim lands on a ZC)
      • 50 % — cross-correlation of the last 20 ms vs first 20 ms of the loop
                (1 = perfect match, loop seam is phase-coherent)
    """
    window = int(_SEAM_WINDOW_S * sr)

    # Zero-crossing proximity score
    zc_dist = abs(trim_sample - active_sample)
    zc_score = float(np.clip(1.0 - zc_dist / max(window, 1), 0.0, 1.0))

    # Cross-correlation seam score
    tail = y[max(0, trim_sample - window): trim_sample]
    head = y[:min(window, len(y))]

    if len(tail) < 4 or len(head) < 4:
        return zc_score

    try:
        tail_n = (tail - tail.mean()) / (tail.std() + 1e-8)
        head_n = (head - head.mean()) / (head.std() + 1e-8)
        xcorr = np.correlate(tail_n, head_n, mode="full")
        xcorr_peak = float(np.clip(xcorr.max() / (len(tail) + 1e-8), 0.0, 1.0))
    except Exception:
        xcorr_peak = 0.5

    return float(np.clip(0.5 * zc_score + 0.5 * xcorr_peak, 0.0, 1.0))


# ── public API ────────────────────────────────────────────────────────────────

def compute_loop_trim(
    path: str | Path,
    threshold_db: float = _DEFAULT_THRESHOLD_DB,
    min_silence_s: float = _DEFAULT_MIN_SILENCE_S,
    sr: int = _SR,
) -> dict:
    """
    Detect the optimal trim endpoint for a loop sample.

    Original file is never modified.

    Returns a dict with keys:
        trim_end_s          float | None  — seconds to trim to, or None
        confidence          float         — 0.0–1.0 seam quality score
        trailing_silence_s  float         — detected trailing silence (s)
        trim_computed_at    datetime      — always set (UTC)

    None trim_end_s means either:
      • trailing silence was < min_silence_s (not worth trimming), OR
      • file unreadable / librosa not installed.
    """
    _null = {
        "trim_end_s": None,
        "confidence": 0.0,
        "trailing_silence_s": 0.0,
        "trim_computed_at": datetime.now(timezone.utc),
    }

    path = Path(path)
    if not path.exists():
        log.debug("loop_trim: file missing %s", path)
        return _null

    try:
        import librosa
    except ImportError:
        log.warning("loop_trim: librosa not installed — skipping %s", path)
        return _null

    try:
        from ..audioio import load as load_audio
        y, _ = load_audio(path, sr=sr, mono=True)
    except Exception as exc:
        log.warning("loop_trim: cannot load %s: %s", path, exc)
        return _null

    if len(y) == 0:
        return _null

    trim_sample, trailing_silence_s = _find_trim_point(
        y, sr, threshold_db, min_silence_s
    )

    if trim_sample is None:
        return {
            **_null,
            "trailing_silence_s": round(trailing_silence_s, 4),
        }

    # Reconstruct active_sample for confidence calc
    try:
        rms = librosa.feature.rms(y=y, hop_length=_HOP_LENGTH)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=rms.max())
        above = np.where(rms_db > threshold_db)[0]
        last_active_frame = int(above[-1]) if len(above) else 0
        active_sample = min(last_active_frame * _HOP_LENGTH + _HOP_LENGTH, len(y) - 1)
    except Exception:
        active_sample = trim_sample

    confidence = _compute_seam_confidence(y, trim_sample, sr, active_sample)
    trim_end_s = round(float(trim_sample) / sr, 6)

    return {
        "trim_end_s": trim_end_s,
        "confidence": round(confidence, 4),
        "trailing_silence_s": round(trailing_silence_s, 4),
        "trim_computed_at": datetime.now(timezone.utc),
    }
