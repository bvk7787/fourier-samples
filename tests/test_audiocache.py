"""Processed-audio cache (packs/audiocache.py)."""
import os
import time

import pytest

from fourier.packs import audiocache as A


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_AUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("FOURIER_NO_AUDIO_CACHE", raising=False)


def test_key_depends_on_args_and_source(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"1234")
    k1 = A.key(str(src), kind="oneshot", mono=False)
    assert k1 == A.key(str(src), kind="oneshot", mono=False)
    assert k1 != A.key(str(src), kind="oneshot", mono=True)
    os.utime(src, ns=(1, 1))
    assert k1 != A.key(str(src), kind="oneshot", mono=False)
    assert A.key(str(tmp_path / "missing.wav")) is None


def test_fingerprint_is_stable_and_nonempty():
    assert A.dsp_fingerprint() == A.dsp_fingerprint()
    assert len(A.dsp_fingerprint()) == 16


def test_store_then_hit_links(tmp_path):
    src = tmp_path / "a.wav"; src.write_bytes(b"x")
    k = A.key(str(src), kind="oneshot")
    out1 = tmp_path / "b1" / "a.wav"; out1.parent.mkdir(); out1.write_bytes(b"processed")
    assert A.lookup(k, tmp_path / "nowhere.wav") is None
    A.store(k, out1)
    out2 = tmp_path / "b2" / "a.wav"; out2.parent.mkdir()
    assert A.lookup(k, out2) == "hit"
    assert out2.read_bytes() == b"processed"


def test_quiet_verdict_cached(tmp_path):
    src = tmp_path / "q.wav"; src.write_bytes(b"x")
    k = A.key(str(src), kind="oneshot")
    A.store(k, None)
    assert A.lookup(k, tmp_path / "o.wav") == "quiet"
    assert not (tmp_path / "o.wav").exists()


def test_disabled(tmp_path, monkeypatch):
    src = tmp_path / "a.wav"; src.write_bytes(b"x")
    k = A.key(str(src))
    out = tmp_path / "o.wav"; out.write_bytes(b"p")
    A.store(k, out)
    monkeypatch.setenv("FOURIER_NO_AUDIO_CACHE", "1")
    assert A.lookup(k, tmp_path / "o2.wav") is None


def test_prune_keeps_linked_and_recent(tmp_path):
    src = tmp_path / "a.wav"; src.write_bytes(b"x")
    ka, kb = A.key(str(src), n=1), A.key(str(src), n=2)
    a = tmp_path / "a_out.wav"; a.write_bytes(b"a")
    b = tmp_path / "b_out.wav"; b.write_bytes(b"b")
    A.store(ka, a); A.store(kb, b)
    b.unlink()                                   # no master links kb any more
    old = time.time() - 30 * 86400
    for k in (ka, kb):
        p = A._path(k).with_suffix(".wav")
        os.utime(p, (old, old))
    assert A.prune(log=lambda *x: None) == 1
    assert A._path(ka).with_suffix(".wav").exists()
    assert not A._path(kb).with_suffix(".wav").exists()


def test_fingerprint_sees_default_arguments(monkeypatch):
    from fourier.packs import curate as C
    A._FP = None
    a = A.dsp_fingerprint()
    old = C._end_fade.__defaults__
    monkeypatch.setattr(C._end_fade, "__defaults__", tuple(x + 1 if isinstance(x, (int, float)) else x for x in old))
    A._FP = None
    assert A.dsp_fingerprint() != a
    monkeypatch.setattr(C._end_fade, "__defaults__", old)
    A._FP = None


def test_dsp_fingerprint_sees_every_constant_the_dsp_reads():
    """The fingerprint hashes the DSP code and the plain constants it names (not the whole
    config); a set, list, regex or nested dict it reads would change output unseen."""
    import inspect
    import types
    from fourier.packs import audiocache as A, curate as C
    seen, parts, bad = set(), [], []
    for f in (C._process_audio, C._clamp_rms, C._rms_db):
        A._reach(f, C, seen, parts)
    for fn in seen:
        mod = inspect.getmodule(fn) or C
        stack, names = [fn.__code__], set()
        while stack:
            co = stack.pop()
            names |= set(co.co_names)
            stack += [c for c in co.co_consts if isinstance(c, types.CodeType)]
        for n in names:
            v = getattr(mod, n, None)
            if isinstance(v, (set, list)) or hasattr(v, "pattern") or (
                    isinstance(v, dict) and not all(isinstance(x, (int, float, str, type(None))) for x in v.values())):
                bad.append((fn.__name__, n))
    assert not bad, bad
