"""Device profiles of your own: where they're read from, which one wins, `extends`, the new
profile fields' checks, and `fourier devices new`."""
import pytest
import yaml
from click.testing import CliRunner

from fourier.cli import main
from fourier.devices.loader import (
    DEFAULT_DEVICES_DIR, DeviceLoader, DeviceProfile, DeviceProfileError, user_devices_dirs,
)

TINY = ("id: {id}\nname: {name}\nstatus: unverified\nload: copy\n"
        "audio:\n  sample_rate: {{value: {rate}, status: unverified}}\n")


def _write(folder, fname, **kw):
    folder.mkdir(parents=True, exist_ok=True)
    kw.setdefault("name", kw["id"])
    kw.setdefault("rate", 32000)
    (folder / fname).write_text(TINY.format(**kw))


def test_user_folders_come_from_the_environment(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("FOURIER_DEVICES", f"{a}:{b}")
    dirs = user_devices_dirs()
    assert dirs[:2] == [a, b] and dirs[-1].as_posix().endswith(".config/fourier/devices")
    monkeypatch.setenv("FOURIER_DEVICES", "none")
    assert user_devices_dirs() == []


def test_a_profile_of_your_own_loads_beside_the_package_ones(tmp_path):
    mine = tmp_path / "mine"
    _write(mine, "box.yaml", id="zz_box", name="Zz Box")
    loader = DeviceLoader(user_dirs=[mine])
    assert "zz_box" in loader.list_devices() and "m8_tracker" in loader.list_devices()
    p = loader.load("zz_box")
    assert (p.origin, p.sample_rate, p.verified) == ("user", 32000, False)
    assert "your own profile" in p.summary()
    assert loader.load("m8_tracker").origin == "package"


def test_a_package_id_is_taken_only_with_override(tmp_path):
    mine = tmp_path / "mine"
    _write(mine, "dt.yaml", id="digitakt_2", rate=22050)
    loader = DeviceLoader(user_dirs=[mine])
    assert loader.load("digitakt_2").sample_rate == 48000          # the package's, untouched
    [(path, why)] = loader.problems()
    assert path.name == "dt.yaml" and "override: true" in why
    (mine / "dt.yaml").write_text("override: true\n" + (mine / "dt.yaml").read_text())
    loader = DeviceLoader(user_dirs=[mine])
    p = loader.load("digitakt_2")
    assert (p.sample_rate, p.origin) == (22050, "user") and loader.problems() == []


def test_the_first_of_your_folders_wins(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "x.yaml", id="zz_twin", rate=22050)
    _write(b, "x.yaml", id="zz_twin", rate=32000)
    loader = DeviceLoader(user_dirs=[a, b])
    assert loader.load("zz_twin").sample_rate == 22050
    assert [p.parent for p, _ in loader.problems()] == [b]


def test_extends_starts_from_another_profile(tmp_path):
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "m8x.yaml").write_text(
        "id: zz_m8_small\nname: Small folders\nextends: m8_tracker\n"
        "paths:\n  files_per_folder: {value: 32, status: unverified}\n")
    p = DeviceLoader(user_dirs=[mine]).load("zz_m8_small")
    m8 = DeviceLoader(user_dirs=[]).load("m8_tracker")
    assert p.files_per_folder == 32 and p.extends == "m8_tracker"
    assert (p.max_path_length, p.card_dir, p.manual) == (m8.max_path_length, m8.card_dir, m8.manual)
    assert p.source("paths.max_path_length") == m8.source("paths.max_path_length")
    # an override that extends the package profile of its own id
    (mine / "dt.yaml").write_text(
        "id: digitakt_2\noverride: true\nextends: digitakt_2\n"
        "paths:\n  files_per_folder: {value: 64, status: convention}\n")
    dt = DeviceLoader(user_dirs=[mine]).load("digitakt_2")
    assert (dt.files_per_folder, dt.sample_rate, dt.origin) == (64, 48000, "user")
    (mine / "bad.yaml").write_text("id: zz_bad\nextends: zz_nowhere\n")
    with pytest.raises(DeviceProfileError, match="zz_nowhere"):
        DeviceLoader(user_dirs=[mine]).load("zz_bad")
    (mine / "c1.yaml").write_text("id: zz_c1\nextends: zz_c2\n")
    (mine / "c2.yaml").write_text("id: zz_c2\nextends: zz_c1\n")
    with pytest.raises(DeviceProfileError, match="circle"):
        DeviceLoader(user_dirs=[mine]).load("zz_c1")
    with pytest.raises(DeviceProfileError, match="extends"):
        DeviceProfile.from_dict({"id": "x", "extends": "m8_tracker"})


@pytest.mark.parametrize("section,key,value,match", [
    ("audio", "bit_depth", 12, "bit_depth"),
    ("audio", "formats", ["mp3"], "formats"),
    ("audio", "max_duration_s", -1, "max_duration_s"),
    ("paths", "ascii_names", "yes", "ascii_names"),
    ("paths", "max_name_length", 4, "max_name_length"),
    ("paths", "files_per_folder", 0, "files_per_folder"),
    ("audio", "sample_rate", 4000, "audio.sample_rate 4000: a whole number of Hz from 8000 to 192000"),
    ("audio", "sample_rate", 44100.5, "audio.sample_rate"),
    ("audio", "sample_rate", "48k", "audio.sample_rate"),
    ("audio", "bit_depth", True, "audio.bit_depth"),
    ("audio", "dither", "yes", "audio.dither"),
    ("paths", "folder_depth", 0, "paths.folder_depth"),
    ("paths", "card_dir", 5, "paths.card_dir"),
])
def test_new_fields_are_checked(section, key, value, match):
    raw = {"id": "x", "status": "unverified", section: {key: {"value": value, "status": "unverified"}}}
    with pytest.raises(DeviceProfileError, match=match):
        DeviceProfile.from_dict(raw)


def test_an_unusual_sample_rate_loads_with_a_warning(tmp_path, monkeypatch):
    """A rate inside 8000 to 192000 Hz that devices rarely use (a typo: 12345) loads, and
    `devices show`, doctor and render say it's unusual; the usual ones say nothing."""
    raw = {"id": "x", "status": "unverified", "audio": {"sample_rate": {"value": 12345, "status": "unverified"}}}
    p = DeviceProfile.from_dict(raw)
    assert p.sample_rate == 12345 and len(p.warnings) == 1 and "12345 Hz is unusual" in p.warnings[0]
    assert not DeviceLoader().load("m8_tracker").warnings and not DeviceLoader().load("digitakt_2").warnings
    mine = tmp_path / "mine"
    _write(mine, "odd.yaml", id="zz_odd", rate=12345)
    monkeypatch.setenv("FOURIER_DEVICES", str(mine))
    out = CliRunner().invoke(main, ["devices", "show", "zz_odd"]).output
    assert "warning: audio.sample_rate 12345 Hz is unusual" in " ".join(out.split())


def test_the_new_generic_profiles_claim_nothing():
    for did in ("generic_folder", "generic_sd_card"):
        p = DeviceLoader(user_dirs=[]).load(did)
        assert not p.verified and p.manual is None and not p.citations
        assert p.max_path_length is None and p.storage_mb is None
        assert {p.source(n) for n in p.sources} == {"unverified"}
    assert DeviceLoader(user_dirs=[]).load("generic_folder").files_per_folder is None
    assert DeviceLoader(user_dirs=[]).load("generic_sd_card").ascii_names is True


def test_the_package_profiles_set_none_of_the_new_rules():
    """digitakt_2 and m8_tracker render as they always have: no name rules, WAV, no length
    limit, a slice limit at or above the master's grid."""
    for did in ("digitakt_2", "m8_tracker"):
        p = DeviceLoader(user_dirs=[]).load(did)
        assert p.ascii_names is None and p.max_name_length is None and p.max_duration_s is None
        assert p.formats == ["wav"] and p.bit_depth == 16 and p.max_slices >= 64


def _new(tmp_path, *args, input=None):
    out = tmp_path / "devs"
    r = CliRunner().invoke(main, ["devices", "new", "--dir", str(out), *args], input=input)
    return r, out


def test_devices_new_from_options(tmp_path):
    r, out = _new(tmp_path, "--name", "Zz Box Two", "--load", "card", "--card-dir", "SAMPLES/",
                  "--sample-rate", "48000", "--bit-depth", "8", "--channels", "mono",
                  "--format", "aiff", "--files-per-folder", "0", "--max-path", "200",
                  "--max-name", "24", "--max-seconds", "20", "--storage-gb", "16", "--yes")
    assert r.exit_code == 0, r.output
    text = (out / "zz_box_two.yaml").read_text()
    assert "status: unverified" in text and "docs/device-profiles.md" in text
    raw = yaml.safe_load(text)
    assert raw["load"] == "card-sync" and raw["paths"]["card_dir"]["value"] == "/SAMPLES"
    p = DeviceLoader(user_dirs=[out]).load("zz_box_two")
    assert (p.sample_rate, p.bit_depth, p.channels, p.formats) == (48000, 8, "mono", ["aiff"])
    assert (p.files_per_folder, p.max_path_length, p.max_name_length) == (None, 200, 24)
    assert (p.max_duration_s, p.storage_mb, p.ascii_names) == (20.0, 16000, True)
    assert all(s == {"status": "unverified"} for s in p.sources.values())
    # the same id again needs --force; a package id never
    r, _ = _new(tmp_path, "--name", "Zz Box Two", "--yes")
    assert r.exit_code == 1 and "--force" in r.output
    r, _ = _new(tmp_path, "--name", "Zz Box Two", "--yes", "--force")
    assert r.exit_code == 0 and DeviceLoader(user_dirs=[out]).load("zz_box_two").bit_depth == 16
    r, _ = _new(tmp_path, "--name", "x", "--id", "m8_tracker", "--yes")
    assert r.exit_code == 1 and "package profile" in r.output


def test_devices_new_asks(tmp_path):
    answers = "\n".join(["Zz Folder Thing", "", "folder", "44100", "24", "stereo", "wav",
                         "2", "0", "0", "0", "0", "0", "n"]) + "\n"
    r, out = _new(tmp_path, input=answers)
    assert r.exit_code == 0, r.output
    p = DeviceLoader(user_dirs=[out]).load("zz_folder_thing")
    assert (p.load, p.bit_depth, p.card_dir, p.files_per_folder, p.ascii_names) == ("copy", 24, "", None, None)
    assert "card_dir" not in (out / "zz_folder_thing.yaml").read_text()


def test_devices_list_names_yours_and_a_skipped_one(tmp_path, monkeypatch):
    mine = tmp_path / "mine"
    _write(mine, "a.yaml", id="zz_listed")
    _write(mine, "b.yaml", id="m8_tracker")
    monkeypatch.setenv("FOURIER_DEVICES", str(mine))
    r = CliRunner().invoke(main, ["devices", "list"], terminal_width=200)
    assert r.exit_code == 0 and "zz_listed" in r.output and "yours" in r.output
    assert "skipped" in r.output and "devices new" in r.output
    r = CliRunner().invoke(main, ["devices", "show", "zz_nowhere"])
    assert r.exit_code == 1 and "No device profile" in r.output


def test_the_package_folder_is_where_the_package_profiles_are():
    assert {p.stem for p in DEFAULT_DEVICES_DIR.glob("*.yaml")} >= {"generic_folder", "generic_sd_card"}


# --- path limits a device's names can't fit -------------------------------------------------
def test_devices_new_refuses_a_path_limit_that_leaves_no_room_for_names(tmp_path):
    from fourier.cli.devices import DEVICE_DOCS_URL, _path_limit_needed
    need = _path_limit_needed("/Samples")
    for limit in (40, need - 1):
        r, out = _new(tmp_path, "--name", "Zz Tight", "--load", "card", "--card-dir", "/Samples",
                      "--max-path", str(limit), "--yes")
        assert r.exit_code == 1 and f"use at least {need}" in r.output, r.output
        assert not (out / "zz_tight.yaml").exists()
    # a shorter card folder leaves more room
    assert _path_limit_needed("") < need
    r, out = _new(tmp_path, "--name", "Zz Tight", "--load", "card", "--card-dir", "/Samples",
                  "--max-path", str(need), "--yes")
    assert r.exit_code == 0, r.output
    assert DEVICE_DOCS_URL in r.output and DEVICE_DOCS_URL.startswith("https://github.com/")
    r, _ = _new(tmp_path, "--name", "Zz Short Names", "--max-name", "10", "--yes")
    assert r.exit_code == 1 and "use at least 14" in r.output


def test_a_device_whose_limit_leaves_no_room_fails_doctor_not_every_command(tmp_path, monkeypatch):
    """A profile whose path limit can't hold the master's names is a doctor FAIL naming the
    device and the fix, and a build stops on it; the other commands still run."""
    mine = tmp_path / "devices"
    mine.mkdir()
    (mine / "zz_tight.yaml").write_text(
        "id: zz_tight\nname: Zz Tight\nstatus: unverified\nload: card-sync\nsample_refs: path\n"
        "paths:\n  card_dir: {value: /Samples, status: unverified}\n"
        "  max_path_length: {value: 40, status: unverified}\n"
        "audio:\n  sample_rate: {value: 44100, status: unverified}\n")
    lib = tmp_path / "SampleLibrary"
    (lib / "Acme" / "Kicks").mkdir(parents=True)
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(f'library = ["{lib}"]\ndevices = ["zz_tight", "m8_tracker"]\npreset = "balanced"\n')
    monkeypatch.setenv("FOURIER_DEVICES", str(mine))
    monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    run = lambda *a: CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), *a], terminal_width=300)
    r = run("doctor")
    out = " ".join(r.output.split())
    assert r.exit_code == 1 and "FAIL device: zz_tight: its path limit of 40 leaves" in out, r.output
    assert "Set paths.max_path_length to at least" in out and "zz_tight.yaml" in out
    assert "OK device: m8_tracker" in out
    for args in (("config", "show"), ("devices", "list"), ("devices", "show", "zz_tight"), ("why", "--rules")):
        r = run(*args)
        assert r.exit_code == 0, (args, r.output)
    r = run("build", "--all", "--no-scan")
    assert r.exit_code == 2 and "device zz_tight: its path limit of 40" in " ".join(r.output.split())
    # the master's names are sized for the devices that fit (the M8's, as without it)
    from fourier import knobs, layers
    got = knobs.devices(["zz_tight", "m8_tracker"], layers.defaults())
    assert got == knobs.devices(["m8_tracker"], layers.defaults())
