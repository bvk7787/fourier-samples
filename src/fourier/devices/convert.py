"""Device output a profile asks for beyond 16/24/32-bit WAV at full length: 8-bit, AIFF and a
longest-file limit (audio.bit_depth 8, audio.formats without wav, audio.max_duration_s).

packs/render.py calls this only for a profile that sets one of those, so every other render
goes through exporter._convert_and_copy exactly as before (and keeps its render-cache keys,
which fingerprint the exporter's source). The steps reuse _convert_and_copy for the
resampling, mono and dual-mono handling:

  max_duration_s  the source is cut to the limit first, with a 5 ms fade so the cut doesn't
                  click (single-cycle waveforms, preserve_length, are never cut)
  8-bit           converted at 32-bit (no dither), then TPDF-dithered to 8 bits with the same
                  content-seeded dither the exporter uses, and written as PCM_U8 (WAV) or
                  PCM_S8 (AIFF); a cycle is quantized without dither, as the exporter does
  AIFF            the converted PCM is rewritten in an AIFF container, sample for sample
                  (an odd-length 8-bit mono AIFF reads back one zero sample longer:
                  libsndfile pads its sound data to an even byte count)
"""
from __future__ import annotations

import shutil
import tempfile
import zlib
from pathlib import Path

from .exporter import _convert_and_copy, _tpdf_dither

FADE_S = 0.005
AIFF_EXT = ".aif"


def out_format(formats) -> str:
    """'wav' when the device takes WAV (or names nothing), else 'aiff' when it takes AIFF."""
    fl = [str(f).lower() for f in (formats or ["wav"])]
    if "wav" in fl:
        return "wav"
    if "aiff" in fl or "aif" in fl:
        return "aiff"
    raise ValueError(f"no format a render can write in {formats} (wav, aiff)")


def extension(fmt: str) -> str:
    return AIFF_EXT if fmt == "aiff" else ".wav"


def subtype(bits: int, fmt: str = "wav") -> str:
    """The soundfile subtype a render writes for this bit depth and format."""
    if bits == 8:
        return "PCM_S8" if fmt == "aiff" else "PCM_U8"
    return {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}[bits]


def _trimmed(source: Path, limit_s: float, target_sr: int, tmp: Path) -> Path:
    """source cut to limit_s seconds with a short fade out, as a float WAV in tmp; or
    source itself when it's no longer than that. The cut is counted in the device's
    samples, so resampling can't round the file past the limit."""
    import numpy as np
    import soundfile as sf

    info = sf.info(str(source))
    n = int(limit_s * target_sr) * info.samplerate // target_sr
    if info.frames <= n:
        return source
    y, sr = sf.read(str(source), always_2d=True, dtype="float64", frames=n)
    nf = max(1, min(len(y) // 4, int(round(sr * FADE_S))))
    y[-nf:] *= np.linspace(1.0, 0.0, nf, endpoint=False)[:, None]
    out = tmp / "cut.wav"
    sf.write(str(out), y if y.shape[1] > 1 else y[:, 0], sr, subtype="FLOAT")
    return out


def convert(source, dest, target_sr: int, target_bit_depth: int, convert_to_mono: bool,
            dither: bool, collapse_dual_mono: bool = False, preserve_length: bool = False,
            out_format: str = "wav", max_duration_s: float | None = None) -> None:
    """Convert one file for a device whose profile asks for 8-bit, AIFF or a length limit."""
    import numpy as np
    import soundfile as sf

    source, dest = Path(source), Path(dest)
    tmp = Path(tempfile.mkdtemp(prefix=".fourier-convert-", dir=dest.parent))
    try:
        src = source
        if max_duration_s and not preserve_length:
            src = _trimmed(source, float(max_duration_s), target_sr, tmp)
        bits = target_bit_depth
        inner = bits if bits in (16, 24, 32) else 32
        mid = dest if (out_format == "wav" and inner == bits) else tmp / "mid.wav"
        _convert_and_copy(src, mid, target_sr=target_sr, target_bit_depth=inner,
                          convert_to_mono=convert_to_mono, dither=dither,
                          collapse_dual_mono=collapse_dual_mono, preserve_length=preserve_length)
        if mid == dest:
            return
        if bits == 8:
            y, sr = sf.read(str(mid), dtype="float64")
            if dither and not preserve_length:
                seed = zlib.crc32(np.ascontiguousarray(y, dtype=np.float32).tobytes())
                y = _tpdf_dither(y, 8, seed)
            q = np.clip(np.round(y * 128.0), -128, 127).astype(np.int16) * 256
            sf.write(str(dest), q, sr, subtype=subtype(8, out_format),
                     format="AIFF" if out_format == "aiff" else "WAV")
        else:
            y, sr = sf.read(str(mid), dtype="int32")
            sf.write(str(dest), y, sr, subtype=subtype(bits, out_format), format="AIFF")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
