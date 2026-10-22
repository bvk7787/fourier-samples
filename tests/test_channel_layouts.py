"""More than two channels fold by the speaker positions a file states: a WAV channel mask
(WAVE_FORMAT_EXTENSIBLE), an AIFF/AIFC CHAN chunk or a CAF chan chunk. A file stating none is
folded by channel position, as before."""
import struct

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

G = 0.7071067811865476


def _signal(n, live, frames=2000):
    """n channels, silent except channel `live`."""
    y = np.zeros((frames, n))
    y[:, live] = 0.5 * np.sin(np.arange(frames) / 7.0)
    return y


def _wav_with_mask(path, y, mask):
    sf.write(str(path), y, 44100, format="WAVEX", subtype="PCM_16")
    b = bytearray(path.read_bytes())
    i = b.find(b"fmt ")
    assert struct.unpack("<H", b[i + 8:i + 10])[0] == 0xFFFE
    b[i + 28:i + 32] = struct.pack("<I", mask)
    path.write_bytes(bytes(b))


def _layout(tag, bitmap=0, labels=()):
    return struct.pack(">III", tag, bitmap, len(labels)) + b"".join(
        struct.pack(">IIfff", lab, 0, 0.0, 0.0, 0.0) for lab in labels)


def _aiff_with_chan(path, y, layout, form=b"AIFF"):
    # libsndfile writes float AIFF as AIFC
    sf.write(str(path), y, 44100, format="AIFF", subtype="PCM_16" if form == b"AIFF" else "FLOAT")
    b = bytearray(path.read_bytes())
    assert b[8:12] == form
    chunk = b"CHAN" + struct.pack(">I", len(layout)) + layout + (b"\0" if len(layout) % 2 else b"")
    b += chunk
    b[4:8] = struct.pack(">I", len(b) - 8)
    path.write_bytes(bytes(b))


def _caf_with_chan(path, y, layout):
    sf.write(str(path), y, 44100, format="CAF", subtype="PCM_16")
    b = path.read_bytes()
    assert b[8:12] == b"desc"
    at = 8 + 12 + struct.unpack(">q", b[12:20])[0]
    path.write_bytes(b[:at] + b"chan" + struct.pack(">q", len(layout)) + layout + b[at:])


def _balance(path):
    """(left, right) peak after the fold the file gets."""
    from fourier.devices.exporter import fold_sides, fold_to_stereo
    data, _ = sf.read(str(path), always_2d=True)
    y = fold_to_stereo(data, channels_first=False, sides=fold_sides(path, data.shape[1]))
    return round(float(np.abs(y[:, 0]).max()), 3), round(float(np.abs(y[:, 1]).max()), 3)


def test_a_wav_channel_mask_names_each_channels_speaker(tmp_path):
    from fourier.audioio import speaker_positions
    p = tmp_path / "a.wav"
    _wav_with_mask(p, _signal(4, 2), 0x000F)             # FL FR FC LFE (3.1)
    assert speaker_positions(p, 4) == ["FL", "FR", "FC", "LFE"]
    a, b = _balance(p)
    assert a == b > 0                                    # the center goes to both sides
    _wav_with_mask(p, _signal(4, 3), 0x0007)             # FL FR FC, the 4th channel unassigned
    assert speaker_positions(p, 4) == ["FL", "FR", "FC", None]
    a, b = _balance(p)
    assert a == b > 0                                    # unassigned: both sides at -3 dB


def test_a_wav_without_a_mask_keeps_the_positional_fold(tmp_path):
    from fourier.audioio import speaker_positions
    from fourier.devices.exporter import fold_sides
    p = tmp_path / "plain.wav"
    sf.write(str(p), _signal(4, 2), 44100, subtype="PCM_16")      # plain PCM: no mask
    assert speaker_positions(p, 4) is None and fold_sides(p, 4) is None
    a, b = _balance(p)
    assert a > 0 and b == 0                              # position 3 of 4: back left
    _wav_with_mask(p, _signal(4, 2), 0)                  # a mask of 0 states nothing
    assert speaker_positions(p, 4) is None
    _wav_with_mask(p, _signal(6, 2), 0x003F)             # 5.1 in WAV order: the same fold
    assert speaker_positions(p, 6) == ["FL", "FR", "FC", "LFE", "BL", "BR"]
    assert fold_sides(p, 6) is None


@pytest.mark.parametrize("form", [b"AIFF", b"AIFC"])
def test_an_aiff_chan_chunk_with_channel_descriptions(tmp_path, form):
    from fourier.audioio import speaker_positions
    p = tmp_path / ("a.aif" if form == b"AIFF" else "a.aifc")
    # the AIFF-C spec's 6-channel order: L, left center, C, R, right center, surround
    _aiff_with_chan(p, _signal(6, 3), _layout(0, labels=(1, 7, 3, 2, 8, 9)), form)
    assert speaker_positions(p, 6) == ["FL", "FLC", "FC", "FR", "FRC", "BC"]
    a, b = _balance(p)
    assert a == 0 and b > 0                              # channel 4 is right (by position: LFE)


def test_a_caf_chan_chunk_with_a_layout_tag(tmp_path):
    from fourier.audioio import speaker_positions
    p = tmp_path / "a.caf"
    _caf_with_chan(p, _signal(6, 1), _layout((124 << 16) | 6))      # MPEG 5.1 D: C L R Ls Rs LFE
    assert speaker_positions(p, 6) == ["FC", "FL", "FR", "BL", "BR", "LFE"]
    a, b = _balance(p)
    assert a > 0 and b == 0                              # channel 2 is left (by position: right)
    _caf_with_chan(p, _signal(4, 2), _layout(1 << 16, bitmap=0x0603))  # bitmap: L R Ls Rs (side)
    assert speaker_positions(p, 4) == ["FL", "FR", "SL", "SR"]
    _caf_with_chan(p, _signal(6, 1), _layout((124 << 16) | 5))      # a count that disagrees
    assert speaker_positions(p, 6) is None
    _caf_with_chan(p, _signal(4, 1), _layout(147 << 16))             # discrete, in order
    assert speaker_positions(p, 4) is None


def test_fold_weights_from_stated_sides():
    from fourier.devices.exporter import _fold_weights
    assert _fold_weights(4, "lrCC") == [(1.0, 0.0), (0.0, 1.0), (G, G), (G, G)]
    assert _fold_weights(3, "ClR") == [(G, G), (1.0, 0.0), (0.0, G)]
    assert _fold_weights(4) == _fold_weights(4, "lrLR")


def test_a_damaged_header_states_nothing(tmp_path):
    from fourier.audioio import speaker_positions
    for name, blob in (("a.wav", b"RIFF\x10\0\0\0WAVEfmt \xff\xff\xff\x7f"), ("b.aif", b"FORM\0\0\0\x04AIFF"),
                       ("c.caf", b"caff\0\1\0\0chan" + struct.pack(">q", 4) + b"\0\0"),
                       ("d.wav", b"")):
        (tmp_path / name).write_bytes(blob)
        assert speaker_positions(tmp_path / name, 4) is None
    assert speaker_positions(tmp_path / "missing.wav", 4) is None


def test_a_render_folds_by_the_stated_layout(tmp_path, monkeypatch):
    from fourier.devices.loader import DeviceLoader
    from fourier.packs import render as R
    monkeypatch.setenv("FOURIER_NO_RENDER_CACHE", "1")
    master = tmp_path / "master"
    p = master / "KICKS" / "wide" / "Qxvwz Kick.wav"
    p.parent.mkdir(parents=True)
    _wav_with_mask(p, _signal(4, 2), 0x000F)             # FL FR FC LFE: the hit is in the center
    dev = DeviceLoader().load("m8_tracker")
    R.render_master_to_device(str(master), dev, str(tmp_path / "out"), log=lambda m: None)
    (q,) = list((tmp_path / "out").rglob("*.wav"))
    y, _ = sf.read(str(q), always_2d=True)
    a, b = float(np.abs(y[:, 0]).max()), float(np.abs(y[:, -1]).max())
    # the same on both sides (kept as mono, the render's dual-mono rule); by position it was
    # back left only, a stereo file silent on the right
    assert a > 0.1 and abs(a - b) < 1e-3


def test_the_master_export_folds_by_the_stated_layout(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from fourier.packs import curate
    monkeypatch.setenv("FOURIER_AUDIO_CACHE", str(tmp_path / "acache"))
    n, dim = 6, 16
    rng = np.random.default_rng(3)
    emb = rng.standard_normal((n, dim)).astype("float32")
    rec = []
    for i in range(n):
        p = tmp_path / "SampleLibrary" / f"Acme{i}" / "Pack" / f"Kick {i:02d}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        y = _signal(4, 2, 8820) * np.exp(-20 * np.arange(8820) / 44100)[:, None]
        _wav_with_mask(p, y, 0x000F)
        rec.append(dict(id=i, row=i, path=str(p), pack=f"Acme{i}", clip=0, br=0.5, atk=10.0,
                        dec=300.0, sub=0.1, noi=0.3, har=0.7, cr=4.0, tune=None, bpm=None,
                        bpm_src=None, bpm_fold=None, sup=1, qual=1.0, fav=0, ab=[], vendor=f"Acme{i}"))
    monkeypatch.setattr(curate, "index_stale", lambda s: (False, n, n))
    monkeypatch.setattr(curate, "load_index", lambda: (np.arange(n), emb))
    monkeypatch.setattr(curate, "_fetch_rows", lambda *a, **k: [])
    monkeypatch.setattr(curate, "_select_records", lambda *a, **k: (list(rec), set(), dict(twins=0)))
    monkeypatch.setattr(curate, "embed_text",
                        lambda t: np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(dim)
                        .astype("float32"))
    seen = []
    real = curate.fold_to_stereo

    def spy(audio, channels_first=True, sides=None):
        seen.append(sides)
        return real(audio, channels_first=channels_first, sides=sides)
    monkeypatch.setattr(curate, "fold_to_stereo", spy)
    for loudness in (True, False):                 # processed, and copied as they are
        seen.clear()
        out = tmp_path / f"master-{loudness}"
        curate.build_taxonomy(None, "KICKS", str(out), homes={}, support={}, votes={}, describe=False,
                              loudness=loudness, return_entries=True, log=lambda m: None)
        files = list(out.rglob("*.wav"))
        assert files and seen and set(seen) == {"lrCC"}, (loudness, seen)
        if not loudness:                           # copied as they are: the center on both sides
            for f in files:
                y, _ = sf.read(str(f), always_2d=True)
                assert y.shape[1] == 2 and np.allclose(np.abs(y[:, 0]).max(), np.abs(y[:, 1]).max(), atol=1e-3)
