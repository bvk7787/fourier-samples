"""The guards on commands that delete or replace files (fourier/safety.py)."""
import json
from pathlib import Path

import pytest

from fourier import safety


@pytest.fixture(autouse=True)
def _own_home(tmp_path, monkeypatch):
    """A Fourier home of this test's own (the sync records and published-releases list)."""
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fourier-home"))


def test_the_home_folder_and_a_root_are_never_a_master(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    for bad in (tmp_path / "home", Path("/")):
        with pytest.raises(safety.UnsafePath):
            safety.check_master_dir(bad)


def test_a_master_must_be_new_empty_or_a_fourier_master(tmp_path):
    safety.check_master_dir(tmp_path / "new")                  # doesn't exist: fine
    (tmp_path / "empty").mkdir()
    safety.check_master_dir(tmp_path / "empty")
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "song.wav").write_bytes(b"x")
    with pytest.raises(safety.UnsafePath, match="isn't a Fourier master"):
        safety.check_master_dir(mine)
    (mine / "manifest.json").write_text(json.dumps({"fourier_manifest": 2, "categories": {}}))
    safety.check_master_dir(mine)


def test_a_master_never_overlaps_the_fourier_home_or_a_library(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fhome"))
    with pytest.raises(safety.UnsafePath, match="the Fourier home"):
        safety.check_master_dir(tmp_path / "fhome" / "master")
    lib = tmp_path / "SampleLibrary"                            # conftest: FOURIER_LIBRARY=SampleLibrary
    (lib / "Vendor A").mkdir(parents=True)
    with pytest.raises(safety.UnsafePath, match="library"):
        safety.check_master_dir(lib / "Vendor A" / "out")


def test_sync_needs_a_card_folder_and_prunes_only_its_own(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    render = tmp_path / "render"
    (render / "01_KICKS").mkdir(parents=True)
    vol = tmp_path / "CARD"
    vol.mkdir()
    with pytest.raises(safety.UnsafePath, match="no card folder"):
        safety.check_card_dest(vol, vol, render, delete=True)
    with pytest.raises(safety.UnsafePath):
        safety.check_card_dest(tmp_path / "home" / "Fourier", tmp_path / "home", render, delete=True)
    card = vol / "Samples" / "Fourier"
    (card / "01_KICKS").mkdir(parents=True)
    safety.check_card_dest(card, vol, render, delete=True)      # only the render's own folders
    (card / "01_KICKS" / "kick.wav").write_bytes(b"x")          # in the render: fine
    (render / "01_KICKS" / "kick.wav").write_bytes(b"x")
    safety.check_card_dest(card, vol, render, delete=True)
    (card / "01_KICKS" / "mine.wav").write_bytes(b"x")          # a user's file, even inside
    with pytest.raises(safety.UnsafePath, match="neither this render nor an earlier sync"):
        safety.check_card_dest(card, vol, render, delete=True)
    safety.check_card_dest(card, vol, render, delete=False)     # without --delete: fine


def test_a_file_an_earlier_sync_to_this_card_wrote_may_go(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fhome"))
    render = tmp_path / "render"
    (render / "01_KICKS").mkdir(parents=True)
    (render / "01_KICKS" / "old.wav").write_bytes(b"x")
    vol = tmp_path / "CARD"
    card = vol / "Samples" / "Fourier"
    (card / "01_KICKS").mkdir(parents=True)
    safety.record_sync("m8_tracker", render, card, vol)        # the last sync copied old.wav
    (render / "01_KICKS" / "old.wav").unlink()                 # this render retired it
    (card / "01_KICKS" / "old.wav").write_bytes(b"x")
    safety.check_card_dest(card, vol, render, delete=True, device_id="m8_tracker")
    with pytest.raises(safety.UnsafePath):                     # another device's record
        safety.check_card_dest(card, vol, render, delete=True, device_id="digitakt_2")
    other = tmp_path / "CARD2"                                 # the same folder on another card
    (other / "Samples" / "Fourier" / "01_KICKS").mkdir(parents=True)
    (other / "Samples" / "Fourier" / "01_KICKS" / "old.wav").write_bytes(b"x")
    with pytest.raises(safety.UnsafePath):
        safety.check_card_dest(other / "Samples" / "Fourier", other, render, delete=True,
                               device_id="m8_tracker")
    moved = vol / "Other"                                      # this card, another card folder
    (moved / "01_KICKS").mkdir(parents=True)
    (moved / "01_KICKS" / "old.wav").write_bytes(b"x")
    (moved / safety.CARD_MARK).write_text(safety.card_id(card))
    with pytest.raises(safety.UnsafePath):
        safety.check_card_dest(moved, vol, render, delete=True, device_id="m8_tracker")


def test_sync_records_merge_until_a_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fhome"))
    render, vol = tmp_path / "render", tmp_path / "CARD"
    card = vol / "Samples" / "Fourier"
    (render / "K").mkdir(parents=True)
    for n in ("a", "b"):
        (render / "K" / f"{n}.wav").write_bytes(b"x")
    safety.record_sync("m8_tracker", render, card, vol)
    (render / "K" / "b.wav").unlink()
    safety.record_sync("m8_tracker", render, card, vol)         # without --delete: b stays known
    assert safety.synced_files("m8_tracker", card, vol) == {"K/a.wav", "K/b.wav"}
    safety.record_sync("m8_tracker", render, card, vol, merge=False)   # after --delete
    assert safety.synced_files("m8_tracker", card, vol) == {"K/a.wav"}
    rec = json.loads(safety.card_record("m8_tracker").read_text())
    (entry,) = rec["cards"].values()
    assert entry["card_dir"] == "/Samples/Fourier" and entry["dest"] == str(card.resolve())
    safety.card_record("m8_tracker").write_text(json.dumps({"render": str(render), "files": ["K/b.wav"]}))
    assert safety.synced_files("m8_tracker", card, vol) == set()        # the old format: not trusted


def test_the_guards_fail_closed(tmp_path, monkeypatch):
    cfg = tmp_path / "fourier.toml"
    cfg.write_text('[output]\nrenderz = "x"\n')                  # a typo in [output]
    monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    from fourier import places
    places.reset()
    with pytest.raises(safety.UnsafePath, match="can't tell where"):
        safety.check_master_dir(tmp_path / "out")
    with pytest.raises(safety.UnsafePath, match="can't tell where"):
        safety.check_card_dest(tmp_path / "CARD" / "F", tmp_path / "CARD", tmp_path / "r", delete=True)
    monkeypatch.setenv("FOURIER_CONFIG", "none")
    places.reset()
    safety.published_record().parent.mkdir(parents=True, exist_ok=True)
    safety.published_record().write_text("{not json")               # the published-releases list
    with pytest.raises(safety.UnsafePath, match="releases Fourier published"):
        safety.check_master_dir(tmp_path / "out")


def test_a_release_is_never_a_master(tmp_path):
    rel = tmp_path / "Somewhere" / "releases" / "v3"
    (rel / "KICKS").mkdir(parents=True)
    (rel / "manifest.json").write_text(json.dumps({"fourier_manifest": 2}))
    (rel.parent / "LATEST.txt").write_text("v3\n")                 # a vN beside LATEST.txt
    for bad in (rel, rel / "KICKS", rel.parent, rel.parent.parent):
        with pytest.raises(safety.UnsafePath, match="release"):
            safety.check_master_dir(bad)
    (rel.parent / "LATEST.txt").unlink()
    safety.check_master_dir(rel)                                   # nothing says it's a release
    (rel / "manifest.json").write_text(json.dumps({"fourier_manifest": 2, "release": "v3"}))
    with pytest.raises(safety.UnsafePath, match="release"):        # its manifest says so
        safety.check_master_dir(rel)
    (rel / "manifest.json").write_text(json.dumps({"fourier_manifest": 2}))
    safety.record_release_path(rel, "v3")                          # publish --to recorded it
    with pytest.raises(safety.UnsafePath, match="release"):
        safety.check_master_dir(rel / "KICKS")


def test_a_file_is_never_a_master(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(safety.UnsafePath, match="is a file"):
        safety.check_master_dir(tmp_path / "notes.txt")


def _master(root, entries=("KICKS/fam/k1.wav",), sets=("KITS/kit/k1.wav",)):
    man = {"fourier_manifest": 2, "categories": {}, "sets": {}}
    for sect, paths in (("categories", entries), ("sets", sets)):
        for p in paths:
            cat, out = p.split("/", 1)
            man[sect].setdefault(cat, {"entries": []})["entries"].append({"out": out})
            (root / p).parent.mkdir(parents=True, exist_ok=True)
            (root / p).write_bytes(b"x")
    (root / "manifest.json").write_text(json.dumps(man))
    return root


def test_a_master_holds_only_what_fourier_made(tmp_path):
    m = _master(tmp_path / "M")
    for f in ("CHANGELOG.md", "loops.csv", "KICKS/_manifest.json", ".DS_Store", "KICKS/fam/._k1.wav",
              "KICKS/fam/k1.wav.asd", "KICKS/fam/Ableton Folder Info/dc66.xmp",
              "_REVIEW/KICKS - fam - k1.wav", "_REVIEW.new/x.wav"):
        (m / f).parent.mkdir(parents=True, exist_ok=True)
        (m / f).write_bytes(b"x")
    (m / "_REVIEW" / "_review.json").write_text(json.dumps({"items": {"KICKS - fam - k1.wav": {}}}))
    assert safety.unknown_master_files(m) == []
    safety.check_master_contents(m)
    for f in ("KICKS/fam/mine.wav", "MY SET/live.wav", "_REVIEW/keep.wav", "KICKS/fam/mine.wav.asd"):
        (m / f).parent.mkdir(parents=True, exist_ok=True)
        (m / f).write_bytes(b"x")
    assert safety.unknown_master_files(m) == ["KICKS/fam/mine.wav", "KICKS/fam/mine.wav.asd",
                                              "MY SET/live.wav", "_REVIEW/keep.wav"]
    with pytest.raises(safety.UnsafePath, match="4 file"):
        safety.check_master_contents(m)
    (m / "manifest.json").write_text("{broken")
    with pytest.raises(safety.UnsafePath, match="can't read"):
        safety.check_master_contents(m)


def test_a_card_volume_may_be_a_drive_root_but_the_card_folder_never_is():
    from pathlib import PurePosixPath, PureWindowsPath
    assert not safety._root_refused(PureWindowsPath("E:/"), allow_drive_root=True)
    assert safety._root_refused(PureWindowsPath("E:/"), allow_drive_root=False)   # a master
    assert safety._root_refused(PureWindowsPath("e:\\"), allow_drive_root=False)
    assert safety._root_refused(PurePosixPath("/"), allow_drive_root=True)        # "/" never
    assert not safety._root_refused(PureWindowsPath("E:/Samples"), allow_drive_root=False)
    safety._card_layout(PureWindowsPath("E:/Samples/Fourier"), PureWindowsPath("E:/"))
    for dest in ("E:/", "e:\\", "F:/Samples/Fourier"):
        with pytest.raises(safety.UnsafePath):
            safety._card_layout(PureWindowsPath(dest), PureWindowsPath("E:/"))


def _ci_stat(p):
    """os.stat as a case-insensitive disk (APFS, FAT) answers it."""
    import os
    p = Path(p)
    cur = Path(p.anchor)
    for part in p.parts[1:]:
        try:
            m = next((n for n in os.listdir(cur) if n.casefold() == part.casefold()), None)
        except OSError:
            return None
        if m is None:
            return None
        cur = cur / m
    st = os.stat(cur)
    return (st.st_dev, st.st_ino)


@pytest.fixture
def case_insensitive(monkeypatch):
    monkeypatch.setattr(safety, "_stat", _ci_stat)
    monkeypatch.setattr(safety, "_case_insensitive", lambda d: True)


def test_places_compare_as_the_disk_does(tmp_path, monkeypatch, case_insensitive):
    home = tmp_path / "home"
    lib = home / "Music" / "Samples"
    lib.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FOURIER_LIBRARY", str(lib))
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fh"))
    from fourier import places
    places.reset()
    for out in ("~/music/samples/Curated", "~/MUSIC/Samples", "~/music/SAMPLES/Acme"):
        with pytest.raises(safety.UnsafePath, match="library"):
            safety.check_master_dir(out)
    with pytest.raises(safety.UnsafePath, match="the Fourier home"):
        safety.check_master_dir(tmp_path / "FH" / "master")
    with pytest.raises(safety.UnsafePath, match="pick a folder of its own"):
        safety.check_master_dir(tmp_path / "HOME")
    assert places.library_root_of(str(home / "music" / "SAMPLES" / "Acme" / "k.wav")) == str(lib)
    safety.check_master_dir("~/Music/FourierCurated")
    places.reset()
