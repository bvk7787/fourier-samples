"""What a WAV file's own metadata chunks say about it, read at scan time without decoding
the audio: the ACID chunk's tempo, beat count and root note (written by loop tools to mark a
loop's tempo) and the sampler chunk's (smpl) MIDI unity note.

    acid   flags (0x01 one-shot, 0x02 root note set), root note, beats, meter, tempo
    smpl   ..., MIDI unity note (the note the file plays at its own pitch), ...

Only the chunk headers are read: each chunk's id and size, then the two small chunks
themselves, seeking past everything else (the audio data included). A file that isn't a
RIFF/RF64 WAVE, or whose chunks don't parse, says nothing. The values are only used without
Sononym, and only where the file's name and its measured tempo say nothing
(curate._fallback_tempos, curate._mkrec's root note).
"""
from __future__ import annotations

import struct

MAX_CHUNKS = 64          # chunks looked at before giving up (a malformed file)
TEMPO_MIN, TEMPO_MAX = 40.0, 300.0


def read_wav_chunks(path) -> dict:
    """{"acid_bpm": float, "acid_beats": int, "root_note": int} for what the file's acid and
    smpl chunks state (keys only for values present and sane); {} for anything else."""
    out: dict = {}
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
                return {}
            for _ in range(MAX_CHUNKS):
                h = f.read(8)
                if len(h) < 8:
                    break
                cid, size = h[:4], struct.unpack("<I", h[4:])[0]
                if cid == b"acid" and 24 <= size <= 1024:
                    out.update(_acid(f.read(size)))
                    size = 0
                elif cid == b"smpl" and 36 <= size <= 1 << 16:
                    data = f.read(min(size, 36))
                    unity = struct.unpack("<I", data[12:16])[0]
                    if 0 <= unity <= 127 and "root_note" not in out:
                        out["smpl_root"] = int(unity)
                    size -= len(data)
                if size == 0xFFFFFFFF:          # RF64's data chunk: its size is elsewhere
                    break
                f.seek(size + (size & 1), 1)
    except (OSError, struct.error, ValueError):
        return {}
    if "smpl_root" in out:
        out.setdefault("root_note", out.pop("smpl_root"))
    return out


def _acid(data: bytes) -> dict:
    flags, root, _u1, _u2, beats, _den, _num, tempo = struct.unpack("<IHHfIHHf", data[:24])
    out: dict = {}
    if not flags & 0x01:                 # a loop (not a one-shot): its tempo and beats
        if TEMPO_MIN <= tempo <= TEMPO_MAX:
            out["acid_bpm"] = round(float(tempo), 3)
        if 0 < beats <= 4096:
            out["acid_beats"] = int(beats)
    if flags & 0x02 and 0 <= root <= 127:
        out["root_note"] = int(root)
    return out


def acid_chunk(bpm: float | None, beats: int = 0, root: int | None = None, one_shot: bool = False) -> bytes:
    """An acid chunk (id, size and body) for these values: tests and tools that write one."""
    flags = (0x01 if one_shot else 0) | (0x02 if root is not None else 0)
    body = struct.pack("<IHHfIHHf", flags, root or 0, 0x8000, 0.0, int(beats), 4, 4, float(bpm or 0.0))
    return b"acid" + struct.pack("<I", len(body)) + body


def smpl_chunk(unity: int) -> bytes:
    """A sampler chunk with this MIDI unity note and no loops."""
    body = struct.pack("<9I", 0, 0, 22676, int(unity), 0, 0, 0, 0, 0)
    return b"smpl" + struct.pack("<I", len(body)) + body


def add_chunks(path, *chunks: bytes) -> None:
    """Append chunks to a RIFF WAVE file and fix its RIFF size (tests' fixtures)."""
    with open(path, "r+b") as f:
        data = f.read()
        if data[:4] != b"RIFF":
            raise ValueError(f"{path}: not a RIFF file")
        extra = b"".join(c + (b"\0" if len(c) & 1 else b"") for c in chunks)
        data = data + extra
        data = data[:4] + struct.pack("<I", len(data) - 8) + data[8:]
        f.seek(0)
        f.write(data)
        f.truncate()
