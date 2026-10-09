"""The key a pack states for a phrase (packs/keys.py) and phrases.csv (packs/sets.py)."""
import csv
import json

import pytest

from fourier.packs import keys as K
from fourier.packs import manifests
from fourier.packs import sets as S


@pytest.mark.parametrize("name, key", [
    ("Acid Qxvwz 128 Am", (9, "minor")),
    ("Chords_F#min_02", (6, "minor")),
    ("Lead Qzv Eb 124", (3, None)),           # an accidental alone: tonic, mode not stated
    ("Pad Qzv C Cmaj", (0, "major")),         # the bare C says nothing; Cmaj does
    ("Bb_minor_Qxvwz", (10, "minor")),
    ("Keys Qxvwz Gs", (8, None)),             # "s" for sharp
    ("Loop A 01", None),                      # a bare letter
    ("Stab Am to Dm", None),                  # two keys
    ("Bass Am A major", None),                # one tonic, two modes
    ("Bassline Qxvwz 140", None),
])
def test_name_key(name, key):
    assert K.name_key(name) == key


def test_label():
    assert K.label((9, "minor")) == "A minor" and K.label((1, "major")) == "C# major"
    assert K.label((3, None)) == "D#" and K.label(None) == ""


def _files(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"")
    K.forget()


def test_the_name_comes_first_then_a_folder_then_the_files_beside_it(tmp_path):
    kit = tmp_path / "Vendor Fm" / "Kit Qzv Gm 124bpm"
    _files(kit, "Bass Qzv.wav", "Lead Qzv Cm.wav")
    # its own name over the folder's
    assert K.stated_key(str(kit / "Lead Qzv Cm.wav"), "Vendor Fm/Kit Qzv Gm 124bpm/Lead Qzv Cm.wav") \
        == ((0, "minor"), "name")
    # the nearest folder that names a key; never the vendor's top folder
    assert K.stated_key(str(kit / "Bass Qzv.wav"), "Vendor Fm/Kit Qzv Gm 124bpm/Bass Qzv.wav") \
        == ((7, "minor"), "folder")
    loose = tmp_path / "Vendor Fm" / "Loops"
    _files(loose, "Arp Qzv.wav")
    assert K.stated_key(str(loose / "Arp Qzv.wav"), "Vendor Fm/Loops/Arp Qzv.wav") == (None, "")


def test_the_files_beside_it_only_when_they_agree(tmp_path):
    d = tmp_path / "Vendor" / "Construction Qxvwz"
    _files(d, "Drums Qxvwz.wav", "Bass Qxvwz Dm.wav", "Keys Qxvwz Dm.wav", "Lead Qxvwz D.wav", "notes.txt")
    rel = "Vendor/Construction Qxvwz/Drums Qxvwz.wav"
    assert K.stated_key(str(d / "Drums Qxvwz.wav"), rel) == ((2, "minor"), "siblings")
    _files(d, "Pad Qxvwz Em.wav")                                  # one disagrees: nothing
    assert K.stated_key(str(d / "Drums Qxvwz.wav"), rel) == (None, "")
    e = tmp_path / "Vendor" / "Single Qxvwz"
    _files(e, "Drums Qxvwz.wav", "Bass Qxvwz Dm.wav")              # one isn't enough
    assert K.stated_key(str(e / "Drums Qxvwz.wav"), "Vendor/Single Qxvwz/Drums Qxvwz.wav") == (None, "")


def test_a_missing_folder_states_nothing(tmp_path):
    K.forget()
    assert K.stated_key(str(tmp_path / "gone" / "Arp Qzv.wav"), "gone/Arp Qzv.wav") == (None, "")


def test_phrases_csv(tmp_path, monkeypatch):
    from fourier import places
    lib = tmp_path / "lib"
    monkeypatch.setenv("FOURIER_LIBRARY", str(lib))
    places.reset()
    kit = lib / "Vendor" / "Kit Qzv Am"
    _files(kit, "Acid Qzv 128.wav", "Chords Qzv 128 Cmaj.wav")
    root = tmp_path / "master"
    (root / "PHRASES" / "acid").mkdir(parents=True)
    man = {"categories": {"PHRASES": {"entries": [
        {"family": "acid", "out": "acid/Acid Qzv 128.wav", "src": str(kit / "Acid Qzv 128.wav"), "bpm": 128.0},
        {"family": "acid", "out": "acid/Chords Qzv 128 Cmaj.wav", "src": str(kit / "Chords Qzv 128 Cmaj.wav"),
         "bpm": 64.0, "bpm_fold": 128.0}]}}}
    (root / "manifest.json").write_text(json.dumps(man))
    try:
        assert S.write_phrases(root, manifests.read(root)) == 2
    finally:
        places.reset()
    rows = list(csv.DictReader(open(root / "phrases.csv")))
    assert [(r["folder"], r["file"], r["bpm"], r["key"], r["key_source"]) for r in rows] == [
        ("acid", "Acid Qzv 128.wav", "128.0", "A minor", "folder"),
        ("acid", "Chords Qzv 128 Cmaj.wav", "128.0", "C major", "name")]
