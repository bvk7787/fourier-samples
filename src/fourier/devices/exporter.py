"""
Audio conversion and file naming shared by the device renders (packs/render.py) and the
curated master: resampling and bit depth with dither (_convert_and_copy), mono downmix,
and the canonical file stem every device shows.
"""

from __future__ import annotations

import logging
import math
import re
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


def _correct_bpm_octave(
    detected: float,
    target: float,
    min_bpm: float = 80.0,
    max_bpm: float = 195.0,
) -> float:
    """Return the power-of-2 multiple of `detected` closest to `target` within [min_bpm, max_bpm]."""
    best_bpm, best_dist = detected, float("inf")
    for exp in range(-2, 3):
        candidate = detected * (2.0 ** exp)
        if min_bpm <= candidate <= max_bpm:
            dist = abs(math.log2(target / candidate))
            if dist < best_dist:
                best_dist, best_bpm = dist, candidate
    return best_bpm


def _timestretch(audio: "np.ndarray", sr: int, rate: float) -> "np.ndarray":
    """Time-stretch by `rate` (>1 = faster). Tries pyrubberband first, falls back to librosa."""
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(audio, sr, rate)
    except (ImportError, RuntimeError):
        pass
    import librosa
    return librosa.effects.time_stretch(audio, rate=rate)


NAME_MAX = 64   # convention: short enough to read on a device screen (FAT32 long names allow 255)


def _truncate_middle(name: str, limit: int) -> str:
    """Cut a long name in the middle ("Kick_909_Tuned_Low~Decay_Long_2"): the
    start says what it is, the end tells siblings apart (numbers, variants)."""
    if len(name) <= limit:
        return name
    keep = limit - 1
    head = (keep + 1) // 2
    tail = keep - head
    return name[:head].rstrip("_-. ") + "~" + name[len(name) - tail:].lstrip("_-. ")


def _sanitize_filename(name: str, limit: int | None = NAME_MAX) -> str:
    """Remove characters that cause issues on FAT32 SD cards; cut an over-long name in the
    middle, not the end (an end cut left "..._Long" and "..._Long_2", losing the part
    that told them apart)."""
    # Replace special chars with underscores, collapse whitespace
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = re.sub(r"\s+", "_", name.strip())
    return _truncate_middle(name, limit) if limit else name


# Boilerplate repeated in every file name of a pack says nothing on a device screen:
# "Vendor A Sample Pack_Kick 01 ..." -> "Kick 01 ..." (on a narrow device screen, names
# that share a long prefix all look alike). The patterns are the
# tunable curate_config.NAME_BOILERPLATE, applied in order; vendor names and codes belong in
# a library overlay.
_COMPILED: dict = {}


def _boilerplate() -> list:
    from ..packs.curate_config import NAME_BOILERPLATE
    key = tuple(NAME_BOILERPLATE)
    if key not in _COMPILED:
        _COMPILED[key] = [re.compile(p, re.I) for p in key]
    return _COMPILED[key]


def _strip_boilerplate(stem: str) -> str:
    """A file name minus pack boilerplate (NAME_BOILERPLATE), unless that leaves too little."""
    out = stem
    for rx in _boilerplate():
        out = rx.sub("", out)
    out = out.strip(" _-")
    return out if len(out) >= 6 else stem


def _balance_brackets(name: str) -> str:
    """Drop brackets that don't pair up ("Loop [120 BPM" -> "Loop 120 BPM"), whether
    the source name lost one or the middle cut took it."""
    for o, c in (("[", "]"), ("(", ")"), ("{", "}")):
        depth, keep = 0, []
        for ch in name:
            if ch == o:
                depth += 1
            elif ch == c:
                if depth == 0:
                    continue                      # a closer with no opener
                depth -= 1
            keep.append(ch)
        out = "".join(keep)
        while depth:                              # openers with no closer: drop the last ones
            i = out.rfind(o)
            out = out[:i] + out[i + 1:]
            depth -= 1
        name = out
    return re.sub(r"_{2,}", "_", name).strip("_")


def canonical_stem(stem: str, limit: int) -> str:
    """The one name a file carries in the master and on every device: boilerplate
    stripped, FAT-safe, cut in the middle to `limit` (curate_config.STEM_MAX, sized for the
    tightest device path), brackets paired. Idempotent, so renders keep it as is."""
    return _balance_brackets(_sanitize_filename(_strip_boilerplate(stem), limit))


_SUBTYPES = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
_SUBTYPE_BITS = {"PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
                 "FLOAT": 32, "DOUBLE": 64}


def _tpdf_dither(audio, bits: int, seed: int):
    """Add +/-1 LSB triangular (TPDF) dither for a reduction to `bits`. Exact zeros stay zero
    (digital silence remains silent); the seed comes from the audio so renders are repeatable."""
    import numpy as np

    rng = np.random.default_rng(seed)
    lsb = 1.0 / (2 ** (bits - 1))
    noise = (rng.random(audio.shape) - rng.random(audio.shape)) * lsb
    return np.where(audio != 0, audio + noise, audio).astype(audio.dtype)


def _convert_and_copy(
    source: Path,
    dest: Path,
    target_sr: int,
    target_bit_depth: int,
    convert_to_mono: bool,
    dither: bool,
    trim_end_s: float | None = None,
    source_bpm: float | None = None,
    target_bpm: float | None = None,
    collapse_dual_mono: bool = False,
    preserve_length: bool = False,
) -> bool:
    """
    Convert audio file and write to dest using soundfile + librosa.

    Processing order: load → time-stretch → resample → trim → dither → write.
    Returns True if time-stretch was applied.

    preserve_length: keep every sample exactly (no stretch, resample, trim or dither) and
    only relabel the sample rate. For single-cycle waveforms: resampling a 2048-sample
    cycle to 48 kHz gives 2229.1 samples, which no loop can repeat without a click, while
    relabelling keeps a perfect cycle (its pitch is set on the device anyway).
    Stereo is downmixed with _mono_downmix: a plain L+R sum of a phase-inverted file can
    all but cancel, so those keep their louder channel instead.
    """
    try:
        import librosa
        import numpy as np
        import soundfile as sf
    except ImportError:
        from .. import REINSTALL
        raise RuntimeError(f"audio conversion needs librosa and soundfile: {REINSTALL}") from None

    # Load audio
    try:
        src_bits = _SUBTYPE_BITS.get(sf.info(str(source)).subtype, 0)
    except Exception:
        src_bits = 0
    audio, sr = librosa.load(str(source), sr=None, mono=False)
    src_peak = float(np.max(np.abs(audio), initial=0.0))
    if convert_to_mono:
        audio = _mono_downmix(audio, keep_level=not preserve_length)
    if preserve_length:
        if audio.ndim == 2:
            audio = audio.T
        subtype = _SUBTYPES.get(target_bit_depth, "PCM_16")
        out_bits = int(subtype.split("_")[1])
        audio = np.clip(audio, -1.0, 1.0 - 1.0 / (2 ** (out_bits - 1)))
        sf.write(str(dest), audio, target_sr, subtype=subtype)
        return False
    # a "stereo" file with identical channels carries no width: keep it mono (half the size)
    if collapse_dual_mono and audio.ndim == 2 and audio.shape[0] == 2 \
            and float(np.max(np.abs(audio[0] - audio[1]), initial=0.0)) <= 1.0 / 32768:
        audio = audio[0]

    # Time-stretch if source and target BPM are both known
    stretched = False
    if (
        source_bpm is not None and target_bpm is not None
        and source_bpm > 0 and target_bpm > 0
    ):
        corrected = _correct_bpm_octave(source_bpm, target_bpm)
        if abs(corrected - target_bpm) / target_bpm > 0.01:
            rate = target_bpm / corrected
            audio = _timestretch(audio, sr, rate)
            stretched = True
            if abs(corrected - source_bpm) > 1:
                log.info(
                    "BPM octave corrected: %.1f → %.1f (stretch ×%.3f to %.0f BPM)",
                    source_bpm, corrected, rate, target_bpm,
                )

    # Resample if needed, keeping the exact length (librosa rounds up in floating point,
    # so a 4.0 s loop at 44.1k would gain a sample at 48k)
    if sr != target_sr:
        n_in = audio.shape[-1]
        _pk = float(np.max(np.abs(audio), initial=0.0))
        _quiet = (lambda v: _pk <= 0 or float(np.max(np.abs(v))) <= _pk * 0.01)
        head_quiet, tail_quiet = _quiet(audio[..., 0]), _quiet(audio[..., -1])
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        n_out = (n_in * target_sr + sr // 2) // sr
        if audio.shape[-1] > n_out:
            audio = audio[..., :n_out]
        elif audio.shape[-1] < n_out:
            pad = [(0, 0)] * (audio.ndim - 1) + [(0, n_out - audio.shape[-1])]
            audio = np.pad(audio, pad)
        sr = target_sr
        # the resampled waveform's peaks can land between the old samples and overshoot:
        # scale back to the source peak rather than clip (a -1 dBFS source can reach full
        # scale once resampled)
        new_peak = float(np.max(np.abs(audio), initial=0.0))
        if src_peak > 0 and new_peak > src_peak:
            audio = audio * (src_peak / new_peak)
        # the resampling filter can lift a faded-in start (or a loop's faded end) off zero:
        # keep the master's quiet edges quiet, with the same half-millisecond ramp
        nf = max(2, min(audio.shape[-1] // 4, int(round(target_sr * 0.0005))))
        ramp = np.linspace(0.0, 1.0, nf, endpoint=False)
        if head_quiet and float(np.max(np.abs(audio[..., 0]))) > new_peak * 0.01:
            audio = audio.copy()
            audio[..., :nf] = audio[..., :nf] * ramp
        if tail_quiet and float(np.max(np.abs(audio[..., -1]))) > new_peak * 0.01:
            audio = audio.copy()
            audio[..., -nf:] = audio[..., -nf:] * ramp[::-1]

    # Apply non-destructive loop trim (metadata from DB; original file untouched)
    if trim_end_s is not None and trim_end_s > 0:
        trim_idx = int(trim_end_s * target_sr)
        if audio.ndim == 1:
            audio = audio[:trim_idx]
        else:
            audio = audio[:, :trim_idx]

    # Transpose stereo: soundfile wants (samples, channels), librosa gives (channels, samples)
    if audio.ndim == 2:
        audio = audio.T

    subtype = _SUBTYPES.get(target_bit_depth, "PCM_16")
    out_bits = int(subtype.split("_")[1])
    # Dither only when reducing bit depth (or when processing produced new fractional values);
    # seeded by the content so re-rendering the same file gives the same bytes.
    if dither and out_bits < 32 and (src_bits == 0 or src_bits > out_bits or sr != target_sr
                                     or stretched):
        seed = zlib.crc32(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
        audio = _tpdf_dither(audio, out_bits, seed)
    # resampling and dither can overshoot full scale; clip rather than let the int conversion wrap
    audio = np.clip(audio, -1.0, 1.0 - 1.0 / (2 ** (out_bits - 1)))

    sf.write(str(dest), audio, target_sr, subtype=subtype)
    return stretched


MONO_PHASE_LOSS_DB = -6.0


def _mono_downmix(audio, keep_level=False):
    """(channels, samples) or (samples,) -> mono. Averages the channels, unless that
    loses more than MONO_PHASE_LOSS_DB of the channels' energy (a phase-inverted or
    wide M/S file: L+R cancels), in which case the loudest channel is kept whole.
    keep_level: a wide file loses up to 3 dB in the average; bring the mono back to the
    channels' level, never past the source's peak (so a wide loop plays as loud on a mono
    device as on a stereo one)."""
    import numpy as np
    if audio.ndim == 1:
        return audio
    m = audio.mean(axis=0)
    ch_pow = float(np.mean(np.square(audio)))
    if ch_pow <= 0:
        return m
    m_pow = float(np.mean(np.square(m)))
    loss_db = 10 * np.log10(m_pow / ch_pow + 1e-12)
    if loss_db < MONO_PHASE_LOSS_DB:
        return audio[int(np.argmax(np.mean(np.square(audio), axis=1)))]
    if keep_level and m_pow > 0 and loss_db < -0.1:
        src_peak = float(np.max(np.abs(audio)))
        m_peak = float(np.max(np.abs(m)))
        g = min(float(np.sqrt(ch_pow / m_pow)), src_peak / m_peak if m_peak > 0 else 1.0)
        if g > 1.0:
            m = m * g
    return m


