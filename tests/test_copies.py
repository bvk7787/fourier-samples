"""One recording under two names: the audio check (curate._same_recording), the instrument
categories' copy prune asking it, and one recording picked by two categories (_cross_copies)."""
import json

import numpy as np
import pytest
import soundfile as sf

from fourier.packs import curate as C
from fourier.packs import why_log


def _tone(path, seed, sr=44100, secs=1.0, lead=0, gain=1.0):
    """A decaying mix of five partials (seed picks them): one recording per seed, at any rate."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * secs)) / sr
    y = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(rng.uniform(100, 2000, 5), rng.uniform(0, 6, 5)))
    y = np.concatenate([np.zeros(lead), y * np.exp(-3 * t) / 5 * gain])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y.astype("float32"), sr, subtype="PCM_16")
    return str(path)


@pytest.fixture(autouse=True)
def _fresh_signals():
    C._copy_signal.cache_clear()
    yield
    C._copy_signal.cache_clear()


def test_same_recording(tmp_path):
    a = _tone(tmp_path / "a.wav", 1)
    assert C._same_recording(a, _tone(tmp_path / "b.wav", 1, lead=300, gain=0.4)) is True  # trimmed, leveled
    assert C._same_recording(a, _tone(tmp_path / "d.wav", 1, sr=48000)) is True            # another rate
    assert C._same_recording(a, _tone(tmp_path / "e.wav", 1, gain=-1.0)) is True           # flipped polarity
    assert C._same_recording(a, _tone(tmp_path / "c.wav", 2)) is False                     # another recording
    assert C._same_recording(a, str(tmp_path / "missing.wav")) is None
    assert C._same_recording(a, a) is True


def test_instrument_copy_prune_asks_the_audio(tmp_path):
    same = [_tone(tmp_path / "x1.wav", 1), _tone(tmp_path / "x2.wav", 1, lead=50)]
    other = _tone(tmp_path / "y.wav", 2)
    emb_n = np.ones((3, 4), dtype="float32") / 2.0            # CLAP: all three identical
    rec = [dict(id=i, row=i, qual=1.0, path=p) for i, p in enumerate([same[0], same[1], other])]
    kept, pruned = C._prune_near_dups(rec, emb_n, 0.999, partners=(partners := {}),
                                      apart=C._recordings_apart)
    assert pruned == 1 and partners == {1: 0}
    assert [r["id"] for r in kept] == [0, 2]                  # the other chord stays
    # CLAP's prefilter is loose: a source that can't be read stays, as nothing confirms a copy
    rec[2]["path"] = str(tmp_path / "gone.wav")
    kept, pruned = C._prune_near_dups(rec, emb_n, 0.999, apart=C._recordings_apart)
    assert [r["id"] for r in kept] == [0, 2]


def _note(path, f0, seed, sr=44100, secs=2.0, bright=0.5, noise=0.0, subtype="PCM_16", stereo=None):
    """A decaying harmonic note at f0 (seed picks its partials' phases and its noise), as a
    multisample's note or velocity layer would be. stereo: (left, right) gains of a second,
    unrelated voice (seed + 100), for a wide stereo recording."""
    def voice(sd):
        rng = np.random.default_rng(sd)
        t = np.arange(int(sr * secs)) / sr
        y = sum(bright ** (k - 1) * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 6.3)) * np.exp(-(1 + k) * t)
                for k in range(1, 12))
        return y + noise * rng.standard_normal(len(t)) * np.exp(-3 * t)
    y = voice(seed)
    if stereo:
        w = voice(seed + 100)
        y = np.stack([y + stereo[0] * w, y + stereo[1] * w], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, (0.5 * y / np.max(np.abs(y))).astype("float32"), sr, subtype=subtype)
    return str(path)


def _mono_of(stereo_path, path, side=None, sr=None, subtype="PCM_16", trim=0):
    """A mono edition of a stereo file: its sum, or one side; at another rate, depth or trim."""
    from scipy.signal import resample_poly
    y, s0 = sf.read(stereo_path, dtype="float64")
    m = y.mean(axis=1) if side is None else y[:, side]
    if sr and sr != s0:
        m = resample_poly(m, sr, s0)
    sf.write(path, m[trim:].astype("float32"), sr or s0, subtype=subtype)
    return str(path)


def test_editions_of_one_recording_are_copies(tmp_path):
    """One recording shipped as mono and stereo editions, at 44.1 and 48 kHz, in 16 and 24 bit,
    trimmed: CLAP puts such editions as low as 0.966, the audio at 1.0."""
    st = _note(tmp_path / "Qxvwz" / "Qzv Vqx.wav", 130.81, 1, sr=48000, subtype="PCM_24",
               stereo=(0.8, -0.8))                                     # wide: its sides differ
    summed = _mono_of(st, tmp_path / "s.wav", sr=44100)                # Qxvwz Mono: the sum, 44.1 kHz
    left = _mono_of(st, tmp_path / "l.wav", side=0, sr=44100, trim=90)  # or the left side only, trimmed
    assert C._same_recording(st, summed) is True
    assert C._same_recording(left, st) is True
    rate = _note(tmp_path / "r48.wav", 130.81, 2, sr=48000, subtype="PCM_24")
    assert C._same_recording(rate, _note(tmp_path / "r44.wav", 130.81, 2, sr=44100)) is True   # 48/24 vs 44.1/16
    # a mono file is no edition of a stereo recording it isn't
    assert C._same_recording(_note(tmp_path / "o.wav", 130.81, 3), st) is False


def test_notes_and_takes_of_one_multisample_are_not_copies(tmp_path, monkeypatch):
    """Two notes of one multisample (another pitch) and two velocity layers that are separate
    takes of one note pass CLAP's loose prefilter but cross-correlate well under 0.99: other
    notes under 0.5, takes (even these idealized ones, partials alike but for their phases)
    under 0.97."""
    c3 = _note(tmp_path / "Qxvwzqq_C3.wav", 130.81, 1)
    monkeypatch.setattr(C, "COPY_XCORR", 0.5)
    for i, f0 in enumerate((138.59, 196.0, 261.63)):                   # C#3, G3, C4
        assert C._same_recording(c3, _note(tmp_path / f"n{i}.wav", f0, 1)) is False
    monkeypatch.setattr(C, "COPY_XCORR", 0.97)
    for seed, bright in ((2, 0.7), (3, 0.35), (4, 0.5)):              # harder, softer, alike
        take = _note(tmp_path / f"Qxvwzqq_C3_v{seed}.wav", 130.81, seed, bright=bright, noise=0.02)
        assert C._same_recording(c3, take) is False


def test_only_pairs_over_the_prefilter_are_compared():
    """A pair under the CLAP prefilter (CROSS_COPY_COS) is never compared as audio; one over it
    is, and the audio decides."""
    assert 0.85 < C.CROSS_COPY_COS < 0.89
    asked = []
    v = np.array([1.0, 0.0], dtype="float32")
    w = np.array([0.89, np.sqrt(1 - 0.89 ** 2)], dtype="float32")     # CLAP cosine 0.89
    members = [("STABS", "/l/a.wav"), ("SYNTH", "/l/b.wav")]
    vecs = {"/l/a.wav": (1, v), "/l/b.wav": (2, w)}
    real = C._same_recording
    try:
        C._same_recording = lambda a, b: asked.append((a, b)) or True
        assert C._cross_copy_pairs(members, vecs, cos=0.9) == [] and not asked
        assert C._cross_copy_pairs(members, vecs) == [(0, 1)] and asked
        assert C._cross_copy_pairs(members, vecs, cos=1.0) == []
    finally:
        C._same_recording = real


def _master(root, picks):
    """Exported files and family manifests for [(category, family, out file)]."""
    fams = {}
    for cat, fam, out in picks:
        (root / cat / fam).mkdir(parents=True, exist_ok=True)
        (root / cat / fam / out).write_bytes(b"x")
        fams.setdefault(cat, {}).setdefault(fam, 0)
        fams[cat][fam] += 1
    for cat, fs in fams.items():
        (root / cat / "_manifest.json").write_text(json.dumps(
            [{"family": f, "copied": n, "n": n} for f, n in fs.items()]))


def test_one_recording_in_two_categories_stays_in_one(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    a1 = _tone(lib / "Qxvwz Stb Qzv.wav", 1)                 # stab-named: STABS' by its name
    a2 = _tone(lib / "Qxvwz Qzv.wav", 1, lead=200)           # the same recording, renamed
    b1 = _tone(lib / "Qxvwz Stb Vqx.wav", 3)
    b2 = _tone(lib / "Qxvwz Vqx.wav", 4)                     # CLAP's twin of b1, another recording
    v1, v2 = np.eye(4, dtype="float32")[0], np.eye(4, dtype="float32")[1]
    vecs = {a1: (1, v1), a2: (2, v1), b1: (3, v2), b2: (4, v2)}
    monkeypatch.setattr(C, "_copy_vectors", lambda srcs: {s: vecs[s] for s in srcs if s in vecs})
    out = tmp_path / "master.next"
    _master(out, [("STABS", "qb-x", "a.wav"), ("STABS", "qb-x", "c.wav"),
                  ("SYNTH", "qc-x", "b.wav"), ("PIANO", "qa-x", "d.wav")])
    why_log.write(out, "SYNTH", {"category": "SYNTH", "kept": 1, "files": 1, "out": {}, "counts": {}})
    picks = [("PIANO", {"files": 1}, [{"family": "qa-x", "out": "qa-x/d.wav", "src": b2}]),
             ("STABS", {"files": 2}, [{"family": "qb-x", "out": "qb-x/a.wav", "src": a1},
                                      {"family": "qb-x", "out": "qb-x/c.wav", "src": b1}]),
             ("SYNTH", {"files": 1}, [{"family": "qc-x", "out": "qc-x/b.wav", "src": a2}])]
    said = []
    assert C._dedupe_across(out, picks, log=said.append) == 1
    syn_, stabs_, piano_ = picks[2], picks[1], picks[0]
    assert syn_[2] == [] and syn_[1]["files"] == 0             # SYNTH's copy went
    assert len(stabs_[2]) == 2 and len(piano_[2]) == 1         # STABS kept its own; PIANO's other one stays
    assert not (out / "SYNTH" / "qc-x" / "b.wav").exists()
    assert json.loads((out / "SYNTH" / "_manifest.json").read_text())[0]["copied"] == 0
    assert said and "two names" in said[0]
    doc = why_log.read(out, "SYNTH")
    assert doc["out"]["cross_copy"] == {"2": 1} and doc["counts"]["cross_copy"] == 1
    assert doc["files"] == 0 and doc["kept"] == 1                # kept: the candidate pool
    line = why_log.describe("cross_copy", 1, doc, name_of=lambda i: f"#{i}",
                            place_of=lambda i: "STABS/qb-x/a.wav")
    assert "same recording" in line and "STABS/qb-x/a.wav" in line


def test_copy_winner_follows_keeps_then_names():
    a, b = ("STABS", "/l/SampleLibrary/Acme/Qxvwz Stb Qzv.wav"), ("SYNTH", "/l/SampleLibrary/Acme/Qxvwz Qzv.wav")
    assert C._copy_winner([a, b]) == a                         # its name makes it a stab
    assert C._copy_winner([a, b], pins={b[1]: "SYNTH"}) == b   # a Keep stays where it was rated
    c, d = ("PERC", "/l/SampleLibrary/Acme/Qzv 1.wav"), ("FX", "/l/SampleLibrary/Northwind/Qzv 2.wav")
    from fourier.packs.curate_config import CATEGORY_ORDER
    first = min((c, d), key=lambda m: CATEGORY_ORDER.index(m[0]))
    assert C._copy_winner([c, d]) == first                     # no name speaks: play order


def test_no_database_no_copies(tmp_path):
    picks = [("STABS", {"files": 1}, [{"family": "f", "out": "f/a.wav", "src": "/nowhere/a.wav"}]),
             ("SYNTH", {"files": 1}, [{"family": "g", "out": "g/b.wav", "src": "/nowhere/b.wav"}])]
    assert C._cross_copies(picks) == ({}, {})
