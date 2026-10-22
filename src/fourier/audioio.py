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


# Speaker positions a file states for its channels, named as the WAV channel mask's bits are
# (in bit order): front left/right/center, LFE, back left/right, front left/right of center,
# back center, side left/right, top center, top front left/center/right, top back
# left/center/right. Core Audio channel layouts (AIFF/AIFC CHAN, CAF chan) use the same bits
# in their bitmaps and labels 1 to 18 for the same speakers; a few speakers of their own are
# named here too (rear surround, wide, left/right total, a second LFE).
WAV_SPEAKERS = ("FL", "FR", "FC", "LFE", "BL", "BR", "FLC", "FRC", "BC", "SL", "SR", "TC",
                "TFL", "TFC", "TFR", "TBL", "TBC", "TBR")
_CA_LABELS = {i + 1: s for i, s in enumerate(WAV_SPEAKERS)} | {
    33: "RLS", 34: "RRS", 35: "LW", 36: "RW", 37: "LFE2", 38: "LT", 39: "RT", 42: "FC"}
# Core Audio's fixed layouts (kAudioChannelLayoutTag_*), by the tag's upper 16 bits
_L, _R, _C, _LFE, _LS, _RS = "FL", "FR", "FC", "LFE", "BL", "BR"
_CA_TAGS = {
    108: (_L, _R, _LS, _RS),                                        # Quadraphonic
    109: (_L, _R, _LS, _RS, _C),                                    # Pentagonal
    110: (_L, _R, _LS, _RS, _C, "BC"),                              # Hexagonal
    111: (_L, _R, _LS, _RS, _C, "BC", "LW", "RW"),                  # Octagonal
    113: (_L, _R, _C), 114: (_C, _L, _R),                           # MPEG 3.0 A, B
    115: (_L, _R, _C, "BC"), 116: (_C, _L, _R, "BC"),               # MPEG 4.0 A, B
    117: (_L, _R, _C, _LS, _RS), 118: (_L, _R, _LS, _RS, _C),       # MPEG 5.0 A to D
    119: (_L, _C, _R, _LS, _RS), 120: (_C, _L, _R, _LS, _RS),
    121: (_L, _R, _C, _LFE, _LS, _RS), 122: (_L, _R, _LS, _RS, _C, _LFE),     # MPEG 5.1 A to D
    123: (_L, _C, _R, _LS, _RS, _LFE), 124: (_C, _L, _R, _LS, _RS, _LFE),
    125: (_L, _R, _C, _LFE, _LS, _RS, "BC"),                        # MPEG 6.1 A
    126: (_L, _R, _C, _LFE, _LS, _RS, "FLC", "FRC"),                # MPEG 7.1 A
    127: (_C, "FLC", "FRC", _L, _R, _LS, _RS, _LFE),                # MPEG 7.1 B
    128: (_L, _R, _C, _LFE, _LS, _RS, "RLS", "RRS"),                # MPEG 7.1 C
    129: (_L, _R, _LS, _RS, _C, _LFE, "FLC", "FRC"),                # Emagic default 7.1
    130: (_L, _R, _C, _LFE, _LS, _RS, "LT", "RT"),                  # SMPTE DTV
    131: (_L, _R, "BC"), 132: (_L, _R, _LS, _RS),                   # ITU 2.1, 2.2
    133: (_L, _R, _LFE), 134: (_L, _R, _LFE, "BC"),                 # DVD 4, 5, 6, 10, 11, 18
    135: (_L, _R, _LFE, _LS, _RS), 136: (_L, _R, _C, _LFE), 137: (_L, _R, _C, _LFE, "BC"),
    138: (_L, _R, _LS, _RS, _LFE),
    139: (_L, _R, _LS, _RS, _C, "BC"),                              # AudioUnit 6.0
    140: (_L, _R, _LS, _RS, _C, "RLS", "RRS"),                      # AudioUnit 7.0
    148: (_L, _R, _LS, _RS, _C, "FLC", "FRC"),                      # AudioUnit 7.0 front
    141: (_C, _L, _R, _LS, _RS, "BC"),                              # AAC 6.0, 6.1, 7.0
    142: (_C, _L, _R, _LS, _RS, "BC", _LFE), 143: (_C, _L, _R, _LS, _RS, "RLS", "RRS"),
    144: (_C, _L, _R, _LS, _RS, "RLS", "RRS", "BC"),                # AAC octagonal
    150: (_L, _C, _R), 151: (_L, _C, _R, "BC"), 152: (_L, _C, _R, _LFE),   # AC-3
    153: (_L, _R, "BC", _LFE), 154: (_L, _C, _R, "BC", _LFE),
    155: (_L, _C, _R, _LS, _RS, "BC"), 156: (_L, _C, _R, _LS, _RS, "RLS", "RRS"),  # E-AC-3
    157: (_L, _C, _R, _LS, _RS, _LFE, "BC"),
}
_MAX_CHUNKS = 256


def speaker_positions(path, channels: int) -> list[str | None] | None:
    """The speaker position of each of a file's channels as its header states it: a WAV
    channel mask (WAVE_FORMAT_EXTENSIBLE), an AIFF/AIFC CHAN chunk or a CAF chan chunk. A
    channel the layout leaves unassigned is None. None when the file states no layout, states
    one for another channel count, or can't be read."""
    import struct
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if head[:4] in (b"RIFF", b"RF64", b"BW64") and head[8:12] == b"WAVE":
                got = _wav_positions(f, channels)
            elif head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
                got = _chunk_layout(f, 12, ">I", 2, b"CHAN", channels)
            elif head[:4] == b"caff":
                got = _chunk_layout(f, 8, ">q", 1, b"chan", channels)
            else:
                return None
    except (OSError, struct.error, ValueError, OverflowError):
        return None
    return got if got and any(got) else None


def _chunks(f, at: int, size_fmt: str, align: int):
    """(id, size) of each chunk from offset at, the file left at the chunk's data."""
    import struct
    n = struct.calcsize(size_fmt)
    for _ in range(_MAX_CHUNKS):
        f.seek(at)
        hd = f.read(4 + n)
        if len(hd) < 4 + n:
            return
        size = struct.unpack(size_fmt, hd[4:])[0]
        yield hd[:4], size
        if size < 0:                         # a CAF data chunk running to the end
            return
        at += 4 + n + size + (size % align)


def _mask_positions(mask: int, channels: int) -> list[str | None]:
    got: list[str | None] = [s for i, s in enumerate(WAV_SPEAKERS) if mask >> i & 1]
    return got[:channels] + [None] * (channels - len(got))


def _wav_positions(f, channels: int):
    import struct
    for cid, size in _chunks(f, 12, "<I", 2):
        if cid == b"fmt ":
            fmt = f.read(min(size, 40))
            if size >= 24 and struct.unpack("<H", fmt[:2])[0] == 0xFFFE:
                return _mask_positions(struct.unpack("<I", fmt[20:24])[0], channels)
            return None
        if cid == b"data":                    # the format comes before the audio
            return None
    return None


def _chunk_layout(f, at: int, size_fmt: str, align: int, want: bytes, channels: int):
    """The positions in the first `want` chunk (a Core Audio AudioChannelLayout)."""
    import struct
    for cid, size in _chunks(f, at, size_fmt, align):
        if cid != want:
            continue
        body = f.read(min(size, 12 + 20 * 64))
        tag, bitmap, count = struct.unpack(">III", body[:12])
        if tag == 0:                          # channel descriptions, one a channel
            if count != channels or len(body) < 12 + 20 * count:
                return None
            labels = [struct.unpack(">I", body[12 + 20 * i:16 + 20 * i])[0] for i in range(count)]
            return [_CA_LABELS.get(lab) for lab in labels]
        if tag == 1 << 16:                    # a bitmap, bits as in the WAV mask
            return _mask_positions(bitmap, channels)
        fixed = _CA_TAGS.get(tag >> 16)
        if fixed is None or (tag & 0xFFFF) != channels or len(fixed) != channels:
            return None
        return list(fixed)
    return None
