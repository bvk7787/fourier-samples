"""The preparer hook (fourier/preparers.py) and `import folder`."""
import numpy as np
import pytest
import soundfile as sf

from fourier import layers, preparers
from fourier.ingest.folder import ingest_folder


class Toy:
    """A made-up codec with its own extension: 4 bytes 'TOY1' then float32 mono PCM at 8 kHz."""
    name = "toy"
    extensions = (".toy",)

    def claims(self, path):
        with open(path, "rb") as f:
            return f.read(4) == b"TOY1"

    def load(self, path):
        with open(path, "rb") as f:
            f.read(4)
            return np.frombuffer(f.read(), np.float32).reshape(-1, 1), 8000, "PCM_16"


def _toy(path, n=800):
    path.write_bytes(b"TOY1" + (0.25 * np.sin(np.arange(n) / 5.0)).astype(np.float32).tobytes())


@pytest.fixture
def folder(tmp_path):
    src = tmp_path / "src" / "Kit"
    src.mkdir(parents=True)
    _toy(src / "hit.toy")
    sf.write(str(src / "plain.wav"), np.zeros(100, np.float32), 44100)
    (src / "notes.txt").write_text("not audio")
    return tmp_path


def test_import_folder_prepares_a_claimed_file_as_wav(folder):
    st = ingest_folder(str(folder / "src"), str(folder / "out"), preparers=[Toy()], log=lambda *_: None)
    assert st["prepared"] == 1 and st["copied"] == 1 and st["unreadable"] == 0
    info = sf.info(str(folder / "out" / "Kit" / "hit.wav"))
    assert info.samplerate == 8000 and info.frames == 800
    assert not (folder / "out" / "Kit" / "notes.txt").exists()


def test_without_a_preparer_its_files_are_ignored(folder):
    st = ingest_folder(str(folder / "src"), str(folder / "out"), preparers=[], log=lambda *_: None)
    assert st["prepared"] == 0 and st["copied"] == 1
    st2 = ingest_folder(str(folder / "src"), str(folder / "out2"), prepare=False, preparers=[Toy()],
                        log=lambda *_: None)
    assert st2["prepared"] == 0


def test_a_preparer_can_only_claim_its_own_extensions(tmp_path):
    class Greedy(Toy):
        name = "greedy"
        extensions = (".aif",)
    with pytest.raises(preparers.PreparerError, match="soundfile reads those itself"):
        preparers.check(Greedy())

    class Nameless(Toy):
        name = "nameless"
        extensions = ()
    with pytest.raises(preparers.PreparerError, match="names no file extensions"):
        preparers.check(Nameless())
    # an open format is never offered, whatever a preparer says
    p = tmp_path / "x.aif"
    _toy(p)
    assert preparers.claim(p, [Toy()]) is None
    q = tmp_path / "x.toy"
    _toy(q)
    assert preparers.claim(q, [Toy()]).name == "toy"


def test_enabled_by_name_in_fourier_toml(tmp_path, monkeypatch):
    monkeypatch.setattr(preparers, "installed", lambda: ["toy"])
    monkeypatch.setattr(preparers, "_load", lambda n: Toy())
    cfg = tmp_path / "fourier.toml"
    cfg.write_text('preparers = ["toy"]\n')
    monkeypatch.setenv(layers.ENV_CONFIG, str(cfg))
    assert [p.name for p in preparers.enabled()] == ["toy"]
    cfg.write_text('preparers = ["missing"]\n')
    with pytest.raises(preparers.PreparerError, match="not installed: missing"):
        preparers.enabled()
    monkeypatch.setenv(layers.ENV_CONFIG, "none")
    assert preparers.enabled() == []                       # nothing enabled without a config


def test_a_preparer_that_fails_to_sniff_is_skipped(tmp_path):
    class Broken:
        name = "broken"
        extensions = (".toy",)

        def claims(self, path):
            raise OSError("boom")
    p = tmp_path / "x.toy"
    _toy(p)
    assert preparers.claim(p, [Broken(), Toy()]).name == "toy"


def test_clean_only_wipes_what_an_import_made_and_folders_cant_overlap(folder):
    out = folder / "out"
    ingest_folder(str(folder / "src"), str(out), preparers=[Toy()], log=lambda *_: None)
    ingest_folder(str(folder / "src"), str(out), clean=True, preparers=[Toy()], log=lambda *_: None)
    assert (out / "Kit" / "plain.wav").exists()
    mine = folder / "mine"
    mine.mkdir()
    (mine / "keep.wav").write_bytes(b"x")
    for clean in (False, True):                  # someone's own folder: never adopted silently
        with pytest.raises(ValueError, match="isn't an import folder"):
            ingest_folder(str(folder / "src"), str(mine), clean=clean, preparers=[], log=lambda *_: None)
    with pytest.raises(ValueError, match="adopt it first without --clean"):
        ingest_folder(str(folder / "src"), str(mine), clean=True, adopt=True, preparers=[],
                      log=lambda *_: None)
    assert (mine / "keep.wav").exists() and not (mine / ".fourier-import").exists()
    ingest_folder(str(folder / "src"), str(mine), adopt=True, preparers=[], log=lambda *_: None)
    assert (mine / "keep.wav").exists() and (mine / ".fourier-import").exists()
    for dest in (folder / "src", folder / "src" / "Kit" / "x", folder):
        with pytest.raises(ValueError, match="can't overlap"):
            ingest_folder(str(folder / "src"), str(dest), clean=True, preparers=[], log=lambda *_: None)
    assert (folder / "src" / "Kit" / "plain.wav").exists()


def test_a_library_root_is_never_the_destination(folder, monkeypatch):
    import fourier.places as P
    monkeypatch.setattr(P, "library_roots", lambda: [folder / "lib"])
    with pytest.raises(ValueError, match="is a library folder"):
        ingest_folder(str(folder / "src"), str(folder / "lib"), preparers=[], log=lambda *_: None)
    st = ingest_folder(str(folder / "src"), str(folder / "lib" / "Kit Import"), preparers=[],
                       log=lambda *_: None)
    assert st["copied"] == 1


def test_a_prepared_file_never_replaces_a_same_named_wav(folder):
    sf.write(str(folder / "src" / "Kit" / "hit.wav"), np.ones(50, np.float32) * 0.1, 44100)
    ingest_folder(str(folder / "src"), str(folder / "out"), preparers=[Toy()], log=lambda *_: None)
    assert sf.info(str(folder / "out" / "Kit" / "hit.wav")).samplerate == 44100
    assert sf.info(str(folder / "out" / "Kit" / "hit.toy.wav")).samplerate == 8000


def test_a_preparer_cant_claim_any_format_libsndfile_reads():
    class Caf(Toy):
        name = "caf"
        extensions = (".CAF",)
    with pytest.raises(preparers.PreparerError, match="soundfile reads those itself"):
        preparers.check(Caf())
