"""Where things are (fourier/places.py): fourier.toml's library, devices and [output]."""
import os

import pytest

from fourier import places


@pytest.fixture
def toml(tmp_path, monkeypatch):
    """Write a fourier.toml and point FOURIER_CONFIG at it (FOURIER_LIBRARY cleared)."""
    def write(text: str):
        p = tmp_path / "fourier.toml"
        p.write_text(text)
        monkeypatch.setenv("FOURIER_CONFIG", str(p))
        monkeypatch.delenv("FOURIER_LIBRARY", raising=False)
        places.reset()
        return p
    yield write
    places.reset()


def test_library_folders_and_paths_under_them(toml, tmp_path):
    lib = tmp_path / "Samples"
    toml(f'library = ["{lib}", "{tmp_path}/Samples/Deep"]\n')
    assert places.library_roots() == [lib / "Deep", lib]           # longest first
    assert places.library_rel(f"{lib}/Acme/Drums/kick.wav") == "Acme/Drums/kick.wav"
    assert places.library_rel(f"{lib}/Deep/Acme/kick.wav") == "Acme/kick.wav"
    assert places.library_rel("/elsewhere/Acme/kick.wav") == "/elsewhere/Acme/kick.wav"
    assert places.library_rel(f"{lib}Extra/kick.wav") == f"{lib}Extra/kick.wav"   # not a prefix match
    assert places.in_library(f"{lib}/Acme/kick.wav") and not places.in_library("/x/kick.wav")
    assert places.library_root_of(f"{lib}/Deep/a.wav") == str(lib / "Deep")


def test_a_bare_folder_name_matches_wherever_it_sits(toml):
    toml('library = ["Samples"]\n')
    assert places.library_roots() == []
    assert places.library_rel("/Volumes/USB/Samples/Acme/kick.wav") == "Acme/kick.wav"
    assert places.library_rel("/a/Samples/b/Samples/Acme/kick.wav") == "Acme/kick.wav"   # the last one
    assert places.library_rel("Samples/Acme/kick.wav") == "Acme/kick.wav"
    assert places.library_rel("/a/MySamples/kick.wav") == "/a/MySamples/kick.wav"
    assert places.library_root_of("/Volumes/USB/Samples/Acme/kick.wav") == "/Volumes/USB/Samples"


def test_the_environment_wins(toml, monkeypatch, tmp_path):
    toml(f'library = ["{tmp_path}/A"]\n')
    monkeypatch.setenv("FOURIER_LIBRARY", os.pathsep.join([f"{tmp_path}/B", "Samples"]))
    assert places.library_rel(f"{tmp_path}/B/x.wav") == "x.wav"
    assert places.library_rel(f"{tmp_path}/A/x.wav") == f"{tmp_path}/A/x.wav"
    assert places.library_rel("/q/Samples/x.wav") == "x.wav"


def test_output_folders(toml, monkeypatch, tmp_path):
    monkeypatch.delenv("FOURIER_CURATED_DIR", raising=False)
    monkeypatch.delenv("FOURIER_LOCK_DIR", raising=False)
    toml("")
    home = os.path.expanduser("~")
    assert places.master_dir() == f"{home}/Music/FourierCurated"
    assert places.renders_dir("m8_tracker") == f"{home}/Music/FourierRenders/m8_tracker"
    assert places.releases_root() == f"{home}/Music/Fourier/releases"
    assert str(places.lock_dir()) == f"{home}/Music/Fourier/devices"
    toml(f'[output]\nmaster = "{tmp_path}/M"\npublish = "{tmp_path}/P"\n')
    assert places.master_dir() == f"{tmp_path}/M"
    assert places.renders_dir() == f"{tmp_path}/FourierRenders"
    assert places.releases_root() == f"{tmp_path}/P/releases"
    monkeypatch.setenv("FOURIER_CURATED_DIR", f"{tmp_path}/E")
    monkeypatch.setenv("FOURIER_LOCK_DIR", f"{tmp_path}/L")
    assert places.master_dir() == f"{tmp_path}/E" and str(places.lock_dir()) == f"{tmp_path}/L"
    toml('[output]\nmystery = "x"\n')
    with pytest.raises(places.PlacesError, match="unknown keys"):
        places.releases_root()


def test_devices_and_the_sononym_db(toml, tmp_path):
    lib = tmp_path / "Samples"
    lib.mkdir()
    toml(f'library = ["{lib}"]\ndevices = ["m8_tracker"]\n')
    assert places.devices() == ["m8_tracker"]
    assert places.sononym_db() is None
    (lib / "sononym.db").write_bytes(b"")
    assert places.sononym_db() == lib / "sononym.db"
    toml(f'library = ["{lib}"]\nsononym_db = "{tmp_path}/other.db"\n')
    assert places.sononym_db() == tmp_path / "other.db"
    toml('devices = "m8_tracker"\n')
    assert places.devices() == ["m8_tracker"]
    toml("devices = [1]\n")
    with pytest.raises(places.PlacesError):
        places.devices()


def test_a_master_records_its_library_and_another_librarys_build_stops(toml, tmp_path):
    """places.master_library_problem: a master whose manifest records library folders that
    share none with this config's is another library's ("fail"); a library that gained or lost
    a folder, or the same folder spelled through a symlink or by its bare name, is the same;
    a manifest from before the record says so once ("old"); no master, nothing."""
    import json
    a, b = tmp_path / "LibA", tmp_path / "LibB"
    a.mkdir()
    b.mkdir()
    master = tmp_path / "master"
    master.mkdir()
    assert places.master_library_problem(master) is None                    # no master yet

    def write(library):
        doc = {"categories": {"KICKS": {"entries": [{"src": f"{a}/k.wav"}]}}}
        if library is not None:
            doc["library"] = library
        (master / "manifest.json").write_text(json.dumps(doc))
    toml(f'library = ["{a}"]\n')
    assert places.library_record() == {"folders": [str(a)], "names": []}
    write(places.library_record())
    assert places.master_library_problem(master) is None                    # this library's
    toml(f'library = ["{a}", "{b}"]\n')
    assert places.master_library_problem(master) is None                    # it gained a folder
    toml(f'library = ["{b}"]\n')
    kind, why = places.master_library_problem(master)
    assert kind == "fail" and str(a) in why and str(b) in why and "[output] master" in why
    toml('library = ["LibA"]\n')
    assert places.master_library_problem(master) is None                    # its bare name
    (tmp_path / "link").symlink_to(a)
    toml(f'library = ["{tmp_path}/link"]\n')
    assert places.master_library_problem(master) is None                    # through a symlink
    toml(f'library = ["{b}"]\n')
    write(None)
    kind, why = places.master_library_problem(master)
    assert kind == "old" and "the next build records it" in why
