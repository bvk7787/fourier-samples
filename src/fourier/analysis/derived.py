"""
Tier 1 derived feature computation.

Computes new scalar features from columns already present in SononymMeta.
No audio file I/O required — runs instantly over the full library.

All functions accept a SononymMeta ORM object and return a dict of feature
values ready to be stored on SampleFeatures.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..db.models import SononymMeta


# Maximum crest_factor used for normalization, calibrated on typical libraries: near a
# typical snare's crest factor, so an average snare scores high and the sharpest hits
# (crest_factor >= 8) get score=1.0 (capped). A much higher cap would squash most snares
# toward zero.
_CREST_MAX = 8.0


def _compute_attack_class(
    crest_factor: float | None,
    duration_s: float | None,
) -> str | None:
    """
    Rule-based attack speed label from crest_factor + duration.

    Returns one of "snap", "punch", "swell", or None if inputs are missing.
    """
    if crest_factor is None or duration_s is None:
        return None
    if duration_s < 0.08 and crest_factor >= 8.0:
        return "snap"    # very short + high crest = click/transient
    if crest_factor < 3.0 and duration_s > 0.5:
        return "swell"   # low crest + long = pad / slow-attack sweep
    if crest_factor >= 3.0 and duration_s <= 0.5:
        return "punch"   # medium transient + short = drum punch
    return "swell"       # long sample with any transient


def _compute_drum_subtype(
    primary_category: str | None,
    brightness: float | None,
    harmonicity: float | None,
    noisiness: float | None,
    crest_factor: float | None,
    duration_s: float | None,
) -> str | None:
    """
    Rule-based within-category drum sub-classification.

    Uses only existing SononymMeta fields — no audio I/O required.

    Returns one of:
      sub_kick | click_kick | acoustic_kick   (Perc Kicks)
      rimshot | clap | snare                  (Perc Snares)
      closed_hat | open_hat | cymbal          (Perc Hats & Shakers)
      None — for non-drum categories

    Thresholds (calibrated on typical libraries):
      sub_kick:    brightness<0.10, harmonicity>0.70, dur>0.25 → dark, tonal, long
      click_kick:  brightness>0.45, crest>6, dur<0.15 → bright, sharp, short
      rimshot:     brightness>0.70, crest>9 → sharp/bright snare top
      clap:        brightness>0.55, dur>0.30, noisiness>0.70 → broad, noisy, sustained
    """
    if primary_category is None:
        return None

    cat = primary_category.lower()

    # ── Kicks ────────────────────────────────────────────────────────────────
    if "kick" in cat:
        if (brightness is not None and brightness < 0.10
                and harmonicity is not None and harmonicity > 0.70
                and duration_s is not None and duration_s > 0.25):
            return "sub_kick"
        if (brightness is not None and brightness > 0.45
                and crest_factor is not None and crest_factor > 6.0
                and duration_s is not None and duration_s < 0.15):
            return "click_kick"
        return "acoustic_kick"

    # ── Snares ───────────────────────────────────────────────────────────────
    if "snare" in cat:
        if (brightness is not None and brightness > 0.70
                and crest_factor is not None and crest_factor > 9.0):
            return "rimshot"
        if (brightness is not None and brightness > 0.55
                and duration_s is not None and duration_s > 0.30
                and noisiness is not None and noisiness > 0.70):
            return "clap"
        return "snare"

    # ── Hats & Shakers ───────────────────────────────────────────────────────
    if "hat" in cat or "shaker" in cat:
        if duration_s is not None and brightness is not None:
            if duration_s > 0.80:
                return "cymbal"
            if duration_s < 0.15 and brightness > 0.60:
                return "closed_hat"
            if duration_s >= 0.15 and brightness > 0.55:
                return "open_hat"
        return "closed_hat"  # safe default for hats

    return None  # non-drum category


def compute_derived(meta: "SononymMeta", duration_s: float | None = None) -> dict:
    """
    Compute all Tier 1 derived features from a SononymMeta record.

    Args:
        meta:       SononymMeta ORM object (all timbral fields).
        duration_s: sample duration in seconds, from the associated Sample row.
                    Required for attack_class and drum_subtype; pass None to skip them.

    Returns a dict with keys matching SampleFeatures column names.
    All values are None if the required source fields are missing.

    Fields computed:
        sub_weight        harmonicity * (1 - brightness)
        transient_score   min(crest_factor / 8.0, 1.0)   — see _CREST_MAX comment
        loop_confidence   class_signature confidence for the primary class
        is_pitched        1 if harmonicity > 0.6 AND pitch_confidence > 0.5
        bpm_reliable      1 if bpm IS NOT NULL AND bpm_confidence > 0.6
        timbral_norm      [brightness, harmonicity, noisiness] as unit-L2 vector
        spectral_balance  (1 - brightness) / (brightness + 0.05)
        pitch_stability   1 if harmonicity > 0.65 AND pitch_confidence > 0.65
        attack_class      "snap"|"punch"|"swell" from crest_factor + duration_s
        drum_subtype      fine-grained drum label (requires duration_s)
    """
    result: dict = {
        "sub_weight": None,
        "transient_score": None,
        "loop_confidence": None,
        "is_pitched": None,
        "bpm_reliable": None,
        "timbral_norm": None,
        # added later
        "spectral_balance": None,
        "pitch_stability": None,
        "attack_class": None,
        "drum_subtype": None,
    }

    # ── sub_weight ──────────────────────────────────────────────────────────
    # Sub-bass proxy: dark + harmonic = sub-heavy.
    # A well-tuned DnB sub kick scores ~0.4–0.7.
    if meta.harmonicity is not None and meta.brightness is not None:
        result["sub_weight"] = round(meta.harmonicity * (1.0 - meta.brightness), 4)

    # ── transient_score ─────────────────────────────────────────────────────
    # Punchiness proxy. High crest_factor = sharp transient (snare, perc, kick click).
    if meta.crest_factor is not None:
        result["transient_score"] = round(min(meta.crest_factor / _CREST_MAX, 1.0), 4)

    # ── loop_confidence ─────────────────────────────────────────────────────
    # Sononym's class_signature is [loop_prob, oneshot_prob].
    # We report the confidence for whichever class the sample was tagged as.
    classes = meta.classes or []
    sig = meta.class_signature or []
    if sig and len(sig) >= 2:
        if "Loop" in classes:
            result["loop_confidence"] = round(float(sig[0]), 4)
        elif "OneShot" in classes:
            result["loop_confidence"] = round(float(sig[1]), 4)
        else:
            # Unknown class — use whichever dimension is higher
            result["loop_confidence"] = round(max(float(sig[0]), float(sig[1])), 4)

    # ── is_pitched ──────────────────────────────────────────────────────────
    # Flag: does this sample have reliable pitch information?
    if meta.harmonicity is not None and meta.pitch_confidence is not None:
        result["is_pitched"] = int(
            meta.harmonicity > 0.6 and meta.pitch_confidence > 0.5
        )
    elif meta.harmonicity is not None:
        # Pitch confidence missing — use harmonicity alone as a weaker signal
        result["is_pitched"] = int(meta.harmonicity > 0.7)

    # ── bpm_reliable ────────────────────────────────────────────────────────
    # Flag: is the BPM reading trustworthy?
    if meta.bpm is not None:
        confidence = meta.bpm_confidence or 0.0
        result["bpm_reliable"] = int(confidence > 0.6)
    else:
        result["bpm_reliable"] = 0

    # ── timbral_norm ────────────────────────────────────────────────────────
    # Unit-L2-normalized [brightness, harmonicity, noisiness].
    # Used to augment the diversity vector in _diversity_select().
    b = meta.brightness
    h = meta.harmonicity
    n = meta.noisiness
    if b is not None and h is not None and n is not None:
        norm = math.sqrt(b * b + h * h + n * n)
        if norm > 0:
            result["timbral_norm"] = [
                round(b / norm, 6),
                round(h / norm, 6),
                round(n / norm, 6),
            ]
        else:
            result["timbral_norm"] = [0.0, 0.0, 0.0]

    # ── spectral_balance ────────────────────────────────────────────
    # Low-to-high frequency energy ratio proxy.
    # (1 - brightness) / (brightness + 0.05) — avoids division by zero.
    # High for kicks (dark), around 1 for snares, lower for hats.
    if meta.brightness is not None:
        result["spectral_balance"] = round(
            (1.0 - meta.brightness) / (meta.brightness + 0.05), 4
        )

    # ── pitch_stability ─────────────────────────────────────────────
    # Stricter pitched-sample flag. Thresholds raised from is_pitched (0.6/0.5)
    # to 0.65/0.65 — signals samples safe to transpose chromatically.
    # Requires pitch_confidence; does NOT fall back to harmonicity-only.
    if meta.harmonicity is not None and meta.pitch_confidence is not None:
        result["pitch_stability"] = int(
            meta.harmonicity > 0.65 and meta.pitch_confidence > 0.65
        )

    # ── attack_class ────────────────────────────────────────────────
    # Human-readable attack speed label for display in fourier query.
    # Requires duration_s (passed by caller from associated Sample row).
    result["attack_class"] = _compute_attack_class(meta.crest_factor, duration_s)

    # ── drum_subtype ────────────────────────────────────────────────
    # Fine-grained within-category drum sub-classification.
    # Requires primary_category from meta.categories and duration_s.
    primary_cat = meta.categories[0] if meta.categories else None
    result["drum_subtype"] = _compute_drum_subtype(
        primary_cat,
        meta.brightness,
        meta.harmonicity,
        meta.noisiness,
        meta.crest_factor,
        duration_s,
    )

    return result
