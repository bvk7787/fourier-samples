"""Mistakes in fourier.toml, an overlay or a device profile end in one plain line that names
the file, never a crash with a bug-report file; a key nothing reads is named, with the key
it's closest to."""
import pytest
from click.testing import CliRunner

from fourier import layers, places
from fourier.cli import main
from fourier.devices.loader import DeviceLoader, DeviceProfileError
from fourier.settings import ConfigError, _coerce


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "no-user-config.toml")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _resolve(cfg, text):
    (cfg / "fourier.toml").write_text(text)
    return layers.resolve()


def test_a_misspelled_setting_is_named_with_the_one_it_resembles(cfg):
    r = _resolve(cfg, 'prest = "breaks-acid"\nloudnes = "hot"\ntempo = "85-180"\n')
    said = " ".join(r.warnings)
    assert "'prest'" in said and "did you mean preset?" in said
    assert "'loudnes'" in said and "did you mean loudness?" in said
    assert "tempo" not in said.replace("'prest'", "")       # a known knob is no warning


def test_every_key_fourier_reads_is_known(cfg):
    r = _resolve(cfg, 'preset = "balanced"\nlibrary = ["/x"]\ndevices = ["m8_tracker"]\n'
                      'sononym_db = "/x.db"\nproviders = ["path"]\npreparers = []\n'
                      'loudness = "hot"\n[output]\nmaster = "/m"\n[sources]\nvendor_max = 0.5\n'
                      '[advanced]\nLOOP_LIMIT_DB = 2.0\n')
    assert r.warnings == []


def test_an_overlay_key_nothing_reads_is_named(cfg):
    (cfg / "mine.yaml").write_text("extend: balanced\nadvanced:\n  LOOP_LIMIT_DB: 2.0\n")
    r = _resolve(cfg, 'overlay = "mine.yaml"\n')
    assert any("mine.yaml" in w and "did you mean extends?" in w for w in r.warnings), r.warnings


@pytest.mark.parametrize("text, said", [
    ("preset = 7\n", 'preset = 7: give a name or a path in quotes'),
    ("overlay = [\"a.yaml\"]\n", "overlay = ['a.yaml']: give a name or a path in quotes"),
    ("advanced = 7\n", "advanced = 7: it's a table"),
    ("[advanced]\nBUDGETS = {KICKS = -5}\n", "KICKS = -5: a number of files, 0 or more"),
    ("[advanced]\nBUDGETS = {SNARES = \"many\"}\n", "SNARES = 'many': a number of files"),
    ("[advanced]\nAVG_FILE_MB = {KICKS = \"big\"}\n", "KICKS = 'big': expected a number"),
])
def test_a_value_of_the_wrong_kind_is_a_config_error(cfg, text, said):
    with pytest.raises(ConfigError, match=None) as e:
        r = _resolve(cfg, text)
        for k, v in r.values.items():               # as each module reads it
            _coerce(k, v, layers.defaults()[k])
    assert said in str(e.value)


def test_an_overlay_that_is_a_list_is_a_config_error(cfg):
    (cfg / "mine.yaml").write_text("- KICKS\n- SNARES\n")
    with pytest.raises(ConfigError, match="mine.yaml: a preset or overlay holds name: value lines"):
        _resolve(cfg, 'overlay = "mine.yaml"\n')


def test_a_yaml_mistake_names_its_line_in_plain_words(cfg):
    (cfg / "mine.yaml").write_text("advanced:\n  TEMPO_BANDS: [60, 75\n  X: [\n")
    with pytest.raises(ConfigError) as e:
        _resolve(cfg, 'overlay = "mine.yaml"\n')
    msg = str(e.value)
    assert "mine.yaml: line 2 (TEMPO_BANDS: [60, 75): a list [ ] or table { } that isn't closed" in msg
    assert "unicode string" not in msg and "\n" not in msg


def test_a_places_mistake_ends_in_one_line_and_no_bug_report(cfg, monkeypatch):
    home = cfg / "home"
    monkeypatch.setenv("FOURIER_HOME", str(home))
    (cfg / "fourier.toml").write_text("library = 7\n")
    monkeypatch.setenv(layers.ENV_CONFIG, str(cfg / "fourier.toml"))
    monkeypatch.delenv("FOURIER_LIBRARY", raising=False)
    places.reset()
    try:
        r = CliRunner().invoke(main, ["doctor"])
    finally:
        places.reset()
    out = " ".join(r.output.split())
    assert "library must be a folder or a list of folders" in out, r.output
    assert "bug report" not in out and "`fourier config edit` opens it to fix" in out
    assert not list(home.rglob("error-*.txt"))


def test_a_profile_of_yours_that_isnt_valid_yaml_is_named(tmp_path):
    mine = tmp_path / "devices"
    mine.mkdir()
    (mine / "qxvwz_box.yaml").write_text("id: qxvwz_box\nname: [Qxvwz Box\nload: folder\n")
    loader = DeviceLoader(user_dirs=[mine])
    with pytest.raises(DeviceProfileError) as e:
        loader.load("qxvwz_box")
    msg = str(e.value)
    assert "qxvwz_box.yaml: isn't valid YAML: line 2 (name: [Qxvwz Box)" in msg
    assert "No device profile" not in msg
    assert "isn't valid YAML" in dict((p.name, w) for p, w in loader.problems())["qxvwz_box.yaml"]
