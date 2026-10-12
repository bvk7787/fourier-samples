"""Name rules that misread a pack's prefix, an ID token or a mic tag: drum abbreviations glued to
a number and more letters, chord qualities inside a random ID, the "+OH" overhead tag in a hat's
name, a closed hat in a kit's open-hat slot, Live's own render folders, and drum names that an
acoustic or organ reservation took (rules._drum_name_wins: compute_homes, why and verify)."""
import json
import os
import re
from collections import namedtuple

import pytest

from fourier.metadata.rows import HOME_FIELDS
from fourier.packs import curate as C
from fourier.packs import sets as S
from fourier.packs.curate_config import CATEGORIES, DEMO_PATH, STAB_CHORD_NAME_RE

from .test_verify import _clean, _hit, _master


# --- drum abbreviations -----------------------------------------------------------------------

@pytest.mark.parametrize("fn, want", [
    ("BD Qzv 07.wav", "KICKS"), ("BD909.wav", "KICKS"), ("bds_qz.wav", "KICKS"),
    ("SD Qzv.wav", "SNARES"), ("SD12.wav", "SNARES"), ("RS 01.wav", "SNARES"),
    ("CH Qzv 07.wav", "HATS"), ("CH01.wav", "HATS"), ("CH 2.wav", "HATS"), ("OH_3.wav", "HATS"),
    ("HH808.wav", "HATS"), ("Cy 3.wav", "CYMBALS"), ("Cymb47.wav", "CYMBALS"),
    # a pack's prefix: the abbreviation glued to a number and more letters names no drum
    ("CH7z-Vqx_05.wav", None), ("OH2x-Acme_1.wav", None), ("BD3k_Qzv.wav", None),
    ("SD4X_Qzv_2.wav", None), ("HH9q Northwind.wav", None), ("CY2b-Qzv.wav", None),
    # ...and the drum word after it still counts
    ("CH7z-Vqx Kick 05.wav", "KICKS"),
    # "+OH" is an overhead mic, not an open hat
    ("Floor Tom Qzv +OH 1.wav", "TOMS"), ("Qzv Stick +OH 8.wav", None),
    ("Kick Qzv +OH 2.wav", "KICKS"), ("Qzv Hat +OH 2.wav", "HATS"),
])
def test_drum_abbreviations(fn, want):
    assert C._drum_named(fn) == want, fn


def test_the_other_abbreviation_rules_read_a_prefix_the_same_way():
    assert C._stab_named("SD4q Vqx Stab.wav") and not C._stab_named("SD Vqx Stab.wav")
    fx = CATEGORIES["FX"]["noise"]
    assert re.search(fx, "OH 3.wav", re.I) and not re.search(fx, "CH7z-Vqx_05.wav", re.I)


# --- chord qualities --------------------------------------------------------------------------

@pytest.mark.parametrize("stem", ["Cm7", "Am9_2", "Fsus4", "Qzv maj7-11", "Keys Dbmaj7", "Chord Gm", "Pad sus2",
                                  "Bbm7 Stab", "Stab Am9 3", "Synth min7", "sus Pad 3", "C#dim"])
def test_chord_names(stem):
    assert STAB_CHORD_NAME_RE.search(stem), stem


@pytest.mark.parametrize("stem", ["38_Vqx_X101_C1-M93Q", "84_Acme_1_X101_C5-M97G", "74_Qzv_X101_C4-SUS7",
                                  "12_Qzv_X101_D2-SUS9", "Bass Fm"])
def test_an_id_token_is_no_chord(stem):
    assert not STAB_CHORD_NAME_RE.search(stem), stem


# --- hat bands and the kits' open-hat slot ---------------------------------------------------

def test_hat_band_reads_past_the_overhead_tag():
    assert C._hat_band("Qzv Hat Closed +OH 5.wav", 0.8) == "closed"
    assert C._hat_band("Qzv Hat Half Open +OH 9.wav", 0.1) == "open"
    assert C._hat_band("Qzv Hat +OH 2.wav", 0.1) == "closed"          # no word: the length decides
    assert C._hat_band("Qzv Hat +OH 2.wav", 0.8) == "open"
    assert C._hat_band("Acme Hat Clsd 3.wav", 0.8) == "closed"
    assert C._hat_band("Qzv Hat Foot 2.wav", 0.8) == "closed"         # a foot (pedal) hat
    assert C._hat_band("OH 6061.wav", 0.1) == "open" and C._hat_band("CH 6061.wav", 0.8) == "closed"
    assert C._hat_band("CH7z-Vqx_05.wav", 0.8) == "open"              # a pack's prefix says neither


def test_the_open_hat_slot_skips_a_hat_named_closed():
    H = lambda n: ("HATS", dict(src=f"/x/{n}.wav", out=f"f/{n}.wav"))
    names = lambda cs: [os.path.basename(ce[1]["src"]) for ce in cs]
    for closed in ("Hat Closed Qz", "Hat Clsd 3", "Hat CL 4", "CHH Qz", "Hat CH 5"):
        assert names(S._slot_candidates("hat_open", [H(closed), H("Hat Open Qz")])) == ["Hat Open Qz.wav"], closed
    # nothing else: never the other hat (a required role), so the kit has no open hat
    assert names(S._slot_candidates("hat_open", [H("Hat Closed Qz")])) == []


# --- Live's renders and demo folders -----------------------------------------------------------

@pytest.mark.parametrize("rel, demo", [
    ("Acme/Pack/Demo Qzv/Samples/Pad 01.wav", True),
    ("Acme/Pack/Live Set/Samples/Processed/Crop/Kick 01 [2020-01-01 101010].wav", True),
    ("Acme/Pack/Live Set/Samples/Processed/Consolidate/Loop 4 [2020-01-01 101010].aif", True),
    ("Acme/Pack/Live Set/Samples/Processed/Reverse/Pad R.wav", True),
    ("Acme/Pack/Live Set/Samples/Processed/Freeze/Freeze Bass.wav", True),
    ("Acme/Pack/Samples/Kicks/Kick 01.wav", False),
    ("Acme/Pack/Demolition Kicks/Kick 01.wav", False),
    ("Acme/QX_PACK_DEMOS/LOOPS/QX_120_Drums_01.wav", False),     # a pack named "... Demos"
    ("Acme/Pack/Samples/Processed Drums/Kick 01.wav", False),
])
def test_demo_paths(rel, demo):
    assert bool(DEMO_PATH.search(rel)) is demo, rel


# --- drum names ahead of the acoustic and organ reservations ---------------------------------

def test_a_drum_name_a_classifier_agrees_on_wins(monkeypatch):
    from fourier.packs import rules
    monkeypatch.setattr(rules, "ORCH_PATH_RE", re.compile(r"/Orchestra Pack/(Strings)/([^/]+)/", re.I))
    W = C._drum_name_wins
    assert W("Chimes Kick Qv.wav", {"class.oneshot", "kick"}, ["Kick", "One Shot"]) == "KICKS"
    assert W("Kick Flute 2.wav", {"class.oneshot", "hat"}, []) == "KICKS"        # any drum vote
    assert W("Organ Snare Qv.wav", {"class.oneshot"}, ["Snare Hit", "One Shot"]) == "SNARES"
    assert W("Rim Qzv Brass.wav", {"class.oneshot"}, ["Rim"]) == "SNARES"
    # no drum vote, a vocal one, no drum name, or an orchestra library's section folder
    assert W("Tom Qzv Zx.wav", {"class.oneshot"}, ["Electric Guitar", "One Shot"]) is None
    assert W("Chimes Kick Qv.wav", {"class.oneshot", "kick", "vocal"}, []) is None
    assert W("Flute C4.wav", {"class.oneshot", "kick"}, ["Kick"]) is None
    assert W("Viola Snap 1.wav", {"class.oneshot", "clap"}, [],
             "Orchestra Pack/Strings/Viola/Viola Snap 1.wav") is None


_Row = namedtuple("_Row", HOME_FIELDS)


def _row(fn, canon, tags):
    d = dict.fromkeys(HOME_FIELDS)
    d.update(id=1, path=f"/lib/SampleLibrary/Acme/Pack/{fn}", rel_path=f"Acme/Pack/{fn}", filename=fn,
             classes=["OneShot"], canonical=sorted(canon), ableton_tags=list(tags), categories=[],
             duration_s=0.4, n_events=1)
    return _Row(**d)


@pytest.mark.parametrize("fn, canon, tags, want", [
    ("Chimes Kick Qv.wav", {"class.oneshot", "kick"}, ["Kick", "One Shot"], "KICKS"),
    ("Kick Flute 2.wav", {"class.oneshot", "hat"}, ["Kick", "One Shot"], "KICKS"),
    ("Organ Snare Qv.wav", {"class.oneshot", "snare"}, ["Snare Hit", "One Shot"], "SNARES"),
    ("Tom Qzv Zx.wav", {"class.oneshot"}, ["Electric Guitar", "One Shot"], "ACOUSTIC"),
    ("Chimes Kick Qv.wav", {"class.oneshot", "lead"}, ["One Shot"], "ACOUSTIC"),
])
def test_why_puts_the_drum_rule_ahead_of_the_reservations(tmp_path, fn, canon, tags, want):
    from fourier.packs.why import explain_row
    w = explain_row(_row(fn, canon, tags), store_path=str(tmp_path / "none.json"))
    assert w.decided[1] == want, w.matched


def _labeled(tmp_path, files, labels):
    """A master of files whose entries carry the labels and tags {filename: (labels, ableton)}."""
    root, store = _master(tmp_path, files)
    p = os.path.join(root, "manifest.json")
    man = json.load(open(p))
    for cd in man["categories"].values():
        for e in cd["entries"]:
            lab, ab = labels.get(os.path.basename(e["src"]), ((), ()))
            e["labels"], e["ableton"] = sorted(lab), list(ab)
    json.dump(man, open(p, "w"))
    return root, store


def _verify_fails(tmp_path, files, labels):
    from fourier.packs.verify import FAIL, verify_master
    root, store = _labeled(tmp_path, files, labels)
    _ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    return {r.check for r in res if r.level == FAIL}


DRUM_CHECK = "files named as a drum (or a scratch) live in that category"


def test_verify_a_drum_in_acoustic_or_piano(tmp_path):
    labels = {"Chimes Kick Qv.wav": ({"class.oneshot", "kick"}, ["Kick", "One Shot"]),
              "Organ Snare Qv.wav": ({"class.oneshot", "snare"}, ["Snare Hit", "One Shot"]),
              "Tom Qzv Zx.wav": ({"class.oneshot"}, ["Electric Guitar", "One Shot"])}
    bad = _clean() + [("ACOUSTIC", "mallet", "Chimes Kick Qv.wav", _hit(), "V4/P/Chimes Kick Qv.wav")]
    assert DRUM_CHECK in _verify_fails(tmp_path / "a", bad, labels)
    bad = _clean() + [("PIANO", "organ", "Organ Snare Qv.wav", _hit(), "V4/P/Organ Snare Qv.wav")]
    assert DRUM_CHECK in _verify_fails(tmp_path / "b", bad, labels)
    # no classifier files it with the drums: the instrument keeps it
    ok = _clean() + [("ACOUSTIC", "plucked", "Tom Qzv Zx.wav", _hit(), "V4/P/Tom Qzv Zx.wav")]
    assert DRUM_CHECK not in _verify_fails(tmp_path / "c", ok, labels)


def test_verify_a_named_wind_or_organ_drum_in_its_drum_category(tmp_path):
    labels = {"Kick Flute 2.wav": ({"class.oneshot", "kick"}, ["Kick", "One Shot"]),
              "Organ Snare Qv.wav": ({"class.oneshot", "snare"}, ["Snare Hit", "One Shot"])}
    files = _clean() + [("KICKS", "dark", "Kick Flute 2.wav", _hit(), "V4/P/Kick Flute 2.wav"),
                        ("SNARES", "dark", "Organ Snare Qv.wav", _hit(), "V4/P/Organ Snare Qv.wav")]
    fails = _verify_fails(tmp_path, files, labels)
    assert not fails & {"named wind / brass one-shots and orchestra folders only in ACOUSTIC",
                        "named organs only in PIANO (organ stabs in STABS)", DRUM_CHECK}, fails
    # without the classifiers' word, the reservations hold as before
    fails = _verify_fails(tmp_path / "x", files, {})
    assert {"named wind / brass one-shots and orchestra folders only in ACOUSTIC",
            "named organs only in PIANO (organ stabs in STABS)"} <= fails, fails

