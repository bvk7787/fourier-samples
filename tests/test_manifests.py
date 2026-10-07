"""manifest.json's library-relative source paths (fourier/packs/manifests.py): written relative
to the library folder they sit in, read back as the absolute paths the build used."""
import json

import pytest

from fourier import places
from fourier.packs import manifests


@pytest.fixture
def lib(tmp_path, monkeypatch):
    home = tmp_path / "home"
    a, b = home / "Samples", tmp_path / "Other Drive" / "Packs"
    for d in (a, b):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FOURIER_LIBRARY", f"{a}:{b}")
    places.reset()
    yield a, b
    places.reset()


def _doc(a, b, outside):
    return {"fourier_manifest": manifests.FORMAT,
            "categories": {"KICKS": {"files": 3, "entries": [
                {"family": "Deep", "out": "Deep/Kick Qxvwz.wav", "src": f"{a}/Acme/Pack/Kick Qxvwz.wav"},
                {"family": "Deep", "out": "Deep/Kick Zyx.wav", "src": f"{b}/Brand/Kick Zyx.wav"},
                {"family": "Deep", "out": "Deep/Kick Wvq.wav", "src": outside}]}},
            "sets": {"KITS": {"entries": [{"family": "Kit", "out": "Kit/Slot Qxvwz.wav",
                                           "src": f"{a}/Acme/Pack/Kick Qxvwz.wav",
                                           "came_from": "KICKS/Deep/Kick Qxvwz.wav"}]}}}


def test_sources_are_stored_relative_and_read_back_absolute(lib, tmp_path):
    a, b = lib
    outside = str(tmp_path / "elsewhere" / "Kick Wvq.wav")
    doc = _doc(a, b, outside)
    before = json.dumps(doc, sort_keys=True)
    manifests.write(tmp_path, doc)
    assert json.dumps(doc, sort_keys=True) == before                 # the caller's doc is untouched
    raw = json.loads((tmp_path / "manifest.json").read_text())
    ents = raw["categories"]["KICKS"]["entries"]
    # the roots sorted, the home folder as ~; the second root's entries say so
    assert raw["src_roots"] == [str(b), "~/Samples"]
    assert [(e["src"], e.get("src_root")) for e in ents] == [
        ("Acme/Pack/Kick Qxvwz.wav", 1), ("Brand/Kick Zyx.wav", None), (outside, None)]
    assert raw["sets"]["KITS"]["entries"][0]["src"] == "Acme/Pack/Kick Qxvwz.wav"
    assert str(tmp_path / "home") not in (tmp_path / "manifest.json").read_text().replace(outside, "")
    assert manifests.read(tmp_path) == doc                           # exactly what was written


def test_a_stored_manifest_written_again_is_the_same(lib, tmp_path):
    a, b = lib
    manifests.write(tmp_path, _doc(a, b, "/elsewhere/Kick Wvq.wav"))
    first = (tmp_path / "manifest.json").read_text()
    manifests.write(tmp_path, manifests.read(tmp_path))
    assert (tmp_path / "manifest.json").read_text() == first
    raw = json.loads(first)
    assert manifests.stored(raw) == raw                     # stored() of a stored document
    assert raw["src_roots"]


def test_older_manifests_read_unchanged(lib, tmp_path):
    a, b = lib
    old = _doc(a, b, "/elsewhere/Kick Wvq.wav")
    old["fourier_manifest"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(old))
    assert manifests.read(tmp_path) == old


def test_without_a_library_sources_stay_absolute(tmp_path, monkeypatch):
    monkeypatch.delenv("FOURIER_LIBRARY", raising=False)
    places.reset()
    doc = {"categories": {"KICKS": {"entries": [{"out": "a.wav", "src": "/x/Acme/a.wav"}]}}}
    try:
        assert "src_roots" not in manifests.stored(doc)
        assert manifests.stored(doc)["categories"]["KICKS"]["entries"][0]["src"] == "/x/Acme/a.wav"
    finally:
        places.reset()


def test_the_library_record_names_no_home_and_still_matches(lib, tmp_path):
    a, b = lib
    rec = places.library_record()
    assert sorted(rec["folders"]) == sorted([str(b), "~/Samples"])
    (tmp_path / "manifest.json").write_text(json.dumps({"categories": {"KICKS": {}}, "library": rec}))
    assert places.master_library_problem(tmp_path) is None
