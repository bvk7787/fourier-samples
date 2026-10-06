"""audioio.load: librosa.load through soundfile only."""
import numpy as np
import pytest
import soundfile as sf


@pytest.mark.parametrize("fmt,subtype", [("WAV", "PCM_24"), ("AIFF", "PCM_16"), ("FLAC", "PCM_16")])
def test_it_decodes_exactly_as_librosa_load(tmp_path, fmt, subtype):
    """Same samples as librosa.load(path), resampled or not, mono or stereo, with a duration."""
    import librosa

    from fourier.audioio import load
    sr = 44100
    t = np.arange(sr) / sr
    y = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 330 * t)], 1)
    p = tmp_path / f"tone.{fmt.lower()}"
    sf.write(str(p), y, sr, format=fmt, subtype=subtype)
    for kw in (dict(sr=None, mono=False), dict(sr=22050, mono=True), dict(sr=48000, mono=True, duration=0.5)):
        a, ra = load(p, **kw)
        b, rb = librosa.load(str(p), **kw)
        assert ra == rb and a.shape == b.shape and np.array_equal(a, b), kw


def test_a_file_soundfile_cant_open_raises_with_no_audioread_fallback(tmp_path, monkeypatch):
    """Not audio (or a format soundfile can't read): soundfile's error, and audioread is never
    tried (it is slow and warns about modules Python 3.13 deprecated)."""
    import librosa.core.audio as lca

    from fourier.audioio import load
    bad = tmp_path / "not-audio.wav"
    bad.write_bytes(b"this is not a wav file")
    monkeypatch.setattr(lca, "__audioread_load", lambda *a, **k: pytest.fail("audioread was tried"),
                        raising=False)
    with pytest.raises(sf.LibsndfileError):
        load(bad, sr=None)
