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
    # a source that can't be read stays a copy, as CLAP says
    rec[2]["path"] = str(tmp_path / "gone.wav")
    kept, pruned = C._prune_near_dups(rec, emb_n, 0.999, apart=C._recordings_apart)
    assert [r["id"] for r in kept] == [0]


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
    why_log.write(out, "SYNTH", {"category": "SYNTH", "kept": 1, "out": {}, "counts": {}})
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
    assert doc["out"]["cross_copy"] == {"2": 1} and doc["counts"]["cross_copy"] == 1 and doc["kept"] == 0
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
