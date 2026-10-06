"""
Tier 2 librosa feature computation.

Reads audio files and computes features not available from Sononym.
Runtime: the basic features take on the order of 10 ms per file; HPSS, attack/decay
and chroma roughly double it.

Features computed:
    mfcc_mean              13 MFCC coefficients — texture fingerprint for diversity
    onset_rate_hz          Onsets/sec — distinguishes dense breaks from sparse loops
    tempo_bpm              Librosa BPM estimate — second opinion when Sononym BPM is missing
    spectral_flatness_mean 0=tonal, 1=white noise — more precise than noisiness for perc

    # Previously declared but empty — now wired in:
    spectral_centroid_mean Mean spectral centroid in Hz (more interpretable than Sononym brightness)
    spectral_rolloff_mean  85th-percentile rolloff frequency in Hz
    zero_crossing_rate_mean Mean ZCR — tonal vs noisy proxy
    rms_mean               Mean RMS energy per frame

    # added later:
    attack_time_ms         ms from onset to peak RMS (10ms frame resolution)
    decay_time_ms          ms from peak RMS to -40dB below peak
    harmonic_percussive_ratio H/(H+P) from HPSS — pitched vs percussive content
    chroma_concentration   max(chroma_mean)/mean(chroma_mean) — pitch focus metric
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Sample rate used for all librosa analysis.
# 22050 Hz is sufficient for timbral/rhythm features and 2× faster than 44100.
_SR = 22050

# Cap duration to avoid memory issues on very long files.
_MAX_DURATION_S = 30.0

# Minimum duration (seconds) required for HPSS and chroma_cqt.
# Very short samples produce degenerate STFT frames.
_MIN_DURATION_FOR_HPSS = 0.05


def compute_features(path: str | Path, sr: int = _SR) -> dict:
    """
    Compute Tier 2 librosa features for a single audio file.

    Args:
        path: Absolute path to the audio file.
        sr:   Sample rate for analysis (default 22050).

    Returns:
        Dict with keys matching SampleFeatures column names.
        Returns an empty dict on any error (caller decides how to handle).

    Keys returned on success:
        mfcc_mean                list[float], 13 elements
        onset_rate_hz            float
        tempo_bpm                float
        spectral_flatness_mean   float
        spectral_centroid_mean   float (Hz)
        spectral_rolloff_mean    float (Hz, 85th percentile)
        zero_crossing_rate_mean  float
        rms_mean                 float
        attack_time_ms           float
        decay_time_ms            float
        harmonic_percussive_ratio float (0–1)
        chroma_concentration     float (≥1.0)
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        from .. import REINSTALL
        log.error(f"librosa can't be imported: {REINSTALL}")
        return {}

    path = Path(path)
    if not path.exists():
        log.debug(f"File not found, skipping: {path}")
        return {}

    try:
        from ..audioio import load as load_audio
        y, _ = load_audio(path, sr=sr, mono=True, duration=_MAX_DURATION_S)
    except Exception as exc:
        log.debug(f"Failed to load {path.name}: {exc}")
        return {}

    if len(y) == 0:
        return {}

    duration = len(y) / sr
    result: dict = {}

    # ── Shared: RMS frames (10ms windows) ───────────────────────────────────
    # Computed once and reused by attack_time_ms and decay_time_ms to avoid
    # redundant computation.
    _rms_frame_length = int(sr * 0.010)   # 10ms
    _rms_hop_length = max(1, _rms_frame_length // 2)  # 5ms hop
    try:
        _rms_frames = librosa.feature.rms(
            y=y, frame_length=_rms_frame_length, hop_length=_rms_hop_length
        )[0]
    except Exception:
        _rms_frames = None

    # ── MFCCs ────────────────────────────────────────────────────────────────
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        result["mfcc_mean"] = [round(float(v), 4) for v in mfcc.mean(axis=1)]
    except Exception as exc:
        log.debug(f"MFCC failed for {path.name}: {exc}")

    # ── Onset density ────────────────────────────────────────────────────────
    try:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="frames")
        result["onset_rate_hz"] = round(len(onset_frames) / duration, 4) if duration > 0 else 0.0
    except Exception as exc:
        log.debug(f"Onset detect failed for {path.name}: {exc}")

    # ── Tempo ─────────────────────────────────────────────────────────────────
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        # Librosa often detects at half or double the true tempo.
        # Fold to [60, 200] — covers all practical electronic music BPM ranges.
        if bpm > 0:
            while bpm < 60.0:
                bpm *= 2.0
            while bpm > 200.0:
                bpm /= 2.0
        result["tempo_bpm"] = round(bpm, 2)
    except Exception as exc:
        log.debug(f"Tempo detection failed for {path.name}: {exc}")

    # ── Spectral flatness ─────────────────────────────────────────────────────
    try:
        flatness = librosa.feature.spectral_flatness(y=y)
        result["spectral_flatness_mean"] = round(float(flatness.mean()), 6)
    except Exception as exc:
        log.debug(f"Spectral flatness failed for {path.name}: {exc}")

    # ── Previously-empty columns: now computed ─────────────────────

    try:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        result["spectral_centroid_mean"] = round(float(centroid.mean()), 2)
    except Exception as exc:
        log.debug(f"Spectral centroid failed for {path.name}: {exc}")

    try:
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        result["spectral_rolloff_mean"] = round(float(rolloff.mean()), 2)
    except Exception as exc:
        log.debug(f"Spectral rolloff failed for {path.name}: {exc}")

    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        result["zero_crossing_rate_mean"] = round(float(zcr.mean()), 6)
    except Exception as exc:
        log.debug(f"ZCR failed for {path.name}: {exc}")

    try:
        rms_full = librosa.feature.rms(y=y)
        result["rms_mean"] = round(float(rms_full.mean()), 6)
    except Exception as exc:
        log.debug(f"rms_mean failed for {path.name}: {exc}")

    # ── attack_time_ms ──────────────────────────────────────────────
    # Time from onset (10% of peak RMS) to peak RMS, in milliseconds.
    # Calibration targets: snap/click ≈ 0–5ms; drums ≈ 5–25ms; swells ≈ 50–300ms.
    if _rms_frames is not None and len(_rms_frames) > 1:
        try:
            peak_idx = int(np.argmax(_rms_frames))
            threshold = 0.10 * float(_rms_frames[peak_idx])
            onset_idx = 0
            for j, val in enumerate(_rms_frames[:peak_idx + 1]):
                if float(val) >= threshold:
                    onset_idx = j
                    break
            frames_to_peak = max(0, peak_idx - onset_idx)
            result["attack_time_ms"] = round(
                frames_to_peak * _rms_hop_length / sr * 1000.0, 2
            )
        except Exception as exc:
            log.debug(f"attack_time_ms failed for {path.name}: {exc}")

    # ── decay_time_ms ───────────────────────────────────────────────
    # Time from peak RMS to -40dB below peak (1% of peak amplitude), in ms.
    # Calibration targets: tight electronic ≈ 100–400ms; acoustic ≈ 300–1000ms.
    if _rms_frames is not None and len(_rms_frames) > 1:
        try:
            peak_idx = int(np.argmax(_rms_frames))
            peak_val = float(_rms_frames[peak_idx])
            # -40dB threshold = amplitude factor of 0.01
            threshold = peak_val * 0.01
            decay_idx = len(_rms_frames) - 1  # default: end of file
            for j in range(peak_idx, len(_rms_frames)):
                if float(_rms_frames[j]) <= threshold:
                    decay_idx = j
                    break
            frames_in_decay = max(0, decay_idx - peak_idx)
            result["decay_time_ms"] = round(
                frames_in_decay * _rms_hop_length / sr * 1000.0, 2
            )
        except Exception as exc:
            log.debug(f"decay_time_ms failed for {path.name}: {exc}")

    # ── harmonic_percussive_ratio ───────────────────────────────────
    # H/(H+P) energy from HPSS. 1.0 = fully harmonic, 0.0 = fully percussive.
    # Calibration targets: sub kick ≈ 0.60–0.80; snare ≈ 0.20–0.40; pad ≈ 0.75–0.95.
    # Guard: skip if audio is too short for a useful STFT.
    if duration >= _MIN_DURATION_FOR_HPSS:
        try:
            D = librosa.stft(y)
            H, P = librosa.decompose.hpss(D)
            h_energy = float(np.sum(np.abs(H) ** 2))
            p_energy = float(np.sum(np.abs(P) ** 2))
            total = h_energy + p_energy
            if total > 0:
                result["harmonic_percussive_ratio"] = round(h_energy / total, 4)
            else:
                result["harmonic_percussive_ratio"] = 0.0
        except Exception as exc:
            log.debug(f"HPSS failed for {path.name}: {exc}")

    # ── Quality gate: clipping and DC offset ─────────────────────────────────
    # Computed on the raw (non-resampled) signal for accuracy.
    # is_clipped:      1 if >0.1% of samples are at ±0.999 (librosa normalises to ±1).
    # dc_offset_ratio: |mean(y)| — should be ~0 for clean audio; >0.02 is audible.
    try:
        clipped = int(np.sum(np.abs(y) >= 0.999))
        result["is_clipped"] = int(clipped / len(y) > 0.001)
        result["dc_offset_ratio"] = round(float(abs(np.mean(y))), 6)
    except Exception as exc:
        log.debug(f"Quality gate failed for {path.name}: {exc}")

    # ── chroma_concentration ────────────────────────────────────────
    # max(chroma_mean) / mean(chroma_mean). High = focused on one pitch class.
    # Calibration targets: tonal note ≈ 2.5–5.0; chord ≈ 1.5–2.5; noise ≈ 1.0–1.3.
    # Always ≥ 1.0 by construction (max ≥ mean).
    if duration >= _MIN_DURATION_FOR_HPSS:
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = chroma.mean(axis=1)   # shape (12,)
            chroma_max = float(chroma_mean.max())
            chroma_avg = float(chroma_mean.mean())
            if chroma_avg > 1e-6:
                result["chroma_concentration"] = round(chroma_max / chroma_avg, 4)
            else:
                result["chroma_concentration"] = 1.0
        except Exception as exc:
            log.debug(f"chroma_concentration failed for {path.name}: {exc}")

    return result
