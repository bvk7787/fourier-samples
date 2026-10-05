"""Config layers (fourier/layers.py): precedence, conversion, and the CLI hand-off."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fourier import layers
from fourier.settings import ConfigError

ROOT = Path(__file__).resolve().parents[1]
KNOWN = {
    "curate_config.LOOP_LIMIT_DB": 3.0,
    "curate_config.TEMPO_BANDS": (60, 75, 85),
    "curate_config.MONO_CATS": {"KICKS", "SUB"},
    "curate_config.DEMO_PATH": re.compile("demo", re.IGNORECASE),
    "curate_config.NOTE_LEAD_CATS": {"PIANO"},
    "naming.NOTE_LEAD_CATS": {"PIANO"},
    "curate.NOTE_WINDOW": {"SUB": (24, 60)},
    "sets.KIT_PER_ROLE": {"tom": 2},
    "curate_config.DRUM_MACHINES": [("909", r"909")],
}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "no-user-config.toml")   # not the real one
    monkeypatch.chdir(tmp_path)
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "base.yaml").write_text("advanced:\n  LOOP_LIMIT_DB: 1.0\n  TEMPO_BANDS: [60, 90]\n")
    (presets / "style.yaml").write_text("extends: base.yaml\nadvanced:\n  LOOP_LIMIT_DB: 2.0\n")
    (presets / "mine.yaml").write_text("advanced:\n  MONO_CATS: [KICKS]\n  TEMPO_BANDS: [70]\n")
    (presets / "none.yaml").write_text("")
    # a config naming no preset gets DEFAULT_PRESET: here one that sets nothing, so each test
    # sees only its own layers (test_a_config_naming_no_preset_gets_balanced: the real one)
    monkeypatch.setattr(layers, "DEFAULT_PRESET", str(presets / "none.yaml"))
    return tmp_path


def _write(tmp, text):
    p = tmp / "fourier.toml"
    p.write_text(text)
    return p


def test_no_config_resolves_nothing(cfg):
    r = layers.resolve(known=KNOWN)
    assert r.values == {} and r.files == []


def test_precedence_default_preset_overlay_config_set(cfg):
    _write(cfg, 'preset = "presets/style.yaml"\noverlay = "presets/mine.yaml"\n'
                '[advanced]\nMONO_CATS = ["SUB"]\n')
    r = layers.resolve(known=KNOWN)
    assert r.values["curate_config.LOOP_LIMIT_DB"] == 2.0          # style over base
    assert r.sources["curate_config.LOOP_LIMIT_DB"] == "preset:style"
    assert r.values["curate_config.TEMPO_BANDS"] == (70,)          # overlay over preset, a tuple
    assert r.values["curate_config.MONO_CATS"] == {"SUB"}          # fourier.toml over overlay, a set
    assert r.sources["curate_config.MONO_CATS"] == "config:fourier.toml"
    r2 = layers.resolve(sets=("LOOP_LIMIT_DB=4.5", "curate_config.MONO_CATS=['KICKS']"), known=KNOWN)
    assert r2.values["curate_config.LOOP_LIMIT_DB"] == 4.5 and r2.sources["curate_config.LOOP_LIMIT_DB"] == "--set"
    assert r2.values["curate_config.MONO_CATS"] == {"KICKS"}
    assert [Path(f).name for f in r2.files] == ["base.yaml", "style.yaml", "mine.yaml", "fourier.toml"]


def test_conversions_follow_the_default(cfg):
    _write(cfg, '[advanced]\nDEMO_PATH = "(?:preview|demo)"\n"curate.NOTE_WINDOW" = {SUB = [20, 50]}\n'
                'KIT_PER_ROLE = {tom = 3}\nDRUM_MACHINES = [["808", "808"]]\n')
    v = layers.resolve(known=KNOWN).values
    assert v["curate_config.DEMO_PATH"].pattern == "(?:preview|demo)"
    assert v["curate_config.DEMO_PATH"].flags & re.IGNORECASE                 # the default's flags kept
    assert v["curate.NOTE_WINDOW"] == {"SUB": (20, 50)}
    assert v["sets.KIT_PER_ROLE"] == {"tom": 3}
    assert v["curate_config.DRUM_MACHINES"] == [("808", "808")]


@pytest.mark.parametrize("text, message", [
    ("[advanced]\nLOOP_LIMIT = 3.0\n", "unknown tunable"),
    ("[advanced]\nNOTE_LEAD_CATS = []\n", "in 2 modules"),
    ("[advanced]\nLOOP_LIMIT_DB = \"hot\"\n", "expected a float"),
    ("[advanced]\nDEMO_PATH = \"(\"\n", "bad regex"),
    ('preset = "nope"\n', "not a file"),
    ("preset = [1\n", "fourier.toml"),
])
def test_bad_configs_fail_loudly(cfg, text, message):
    _write(cfg, text)
    with pytest.raises(ConfigError, match=re.escape(message)):
        layers.resolve(known=KNOWN)


def test_a_config_naming_no_preset_gets_balanced(cfg, monkeypatch):
    monkeypatch.setattr(layers, "DEFAULT_PRESET", "balanced")
    _write(cfg, "[advanced]\nLOOP_LIMIT_DB = 2.5\n")
    r = layers.resolve()
    assert r.preset == "balanced" and r.preset_default
    # balanced: the code's defaults, nothing more
    assert not any(src.startswith("preset:") for src in r.sources.values()), r.sources
    assert r.values["curate_config.LOOP_LIMIT_DB"] == 2.5
    _write(cfg, 'preset = "breaks-acid"\n[advanced]\nLOOP_LIMIT_DB = 2.5\n')
    r = layers.resolve()
    assert r.preset == "breaks-acid" and not r.preset_default
    assert r.sources["curate_config.BUDGETS"] == "preset:breaks-acid"
    assert r.values["curate_config.BUDGETS"]["DRUMLOOPS"] == 1900
    assert r.values["curate_config.LOOP_LIMIT_DB"] == 2.5
    _write(cfg, "")
    monkeypatch.setenv(layers.ENV_CONFIG, "none")
    r = layers.resolve()                       # no config at all: the code's defaults untouched
    assert r.values == {} and r.preset is None


def test_extends_loop(cfg):
    (cfg / "presets" / "a.yaml").write_text("extends: b.yaml\n")
    (cfg / "presets" / "b.yaml").write_text("extends: a.yaml\n")
    _write(cfg, 'preset = "presets/a.yaml"\n')
    with pytest.raises(ConfigError, match="loop"):
        layers.resolve(known=KNOWN)


def test_config_lookup_order(cfg, monkeypatch, tmp_path):
    assert layers.find_config() is None
    local = _write(cfg, "")
    assert layers.find_config() == Path.cwd() / "fourier.toml"
    other = tmp_path / "other.toml"
    other.write_text("")
    monkeypatch.setenv(layers.ENV_CONFIG, str(other))
    assert layers.find_config() == other
    assert layers.find_config(str(local)) == local                  # --config wins
    monkeypatch.setenv(layers.ENV_CONFIG, "none")
    assert layers.find_config() is None


def test_write_names_the_file_by_hash(cfg):
    r = layers.Resolved(values={"curate_config.LOOP_LIMIT_DB": 2.0})
    p = layers.write(r, cfg / "run")
    assert p.name == f"resolved-{r.hash}.json" and json.loads(p.read_text())["version"] == 1


def _cli(args, cwd, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in ("FOURIER_RESOLVED_CONFIG", "FOURIER_CONFIG")}
    (cwd / "home").mkdir(exist_ok=True)
    env.update(FOURIER_HOME=str(cwd / "home"), PYTHONPATH=str(ROOT / "src"),
               HOME=str(cwd), **(extra_env or {}))            # HOME: not the real ~/.config/fourier
    return subprocess.run([sys.executable, "-m", "fourier.cli", "--db", str(cwd / "home" / "t.duckdb"), *args],
                          cwd=cwd, env=env, capture_output=True, text=True, check=False)


def test_cli_resolves_once_and_workers_see_it(tmp_path):
    """--set reaches this process's modules and a spawned worker, through the env var."""
    (tmp_path / "fourier.toml").write_text("[advanced]\nSLICE_MIN_HITS = 0.25\n")
    r = _cli(["--set", "LOOP_LIMIT_DB=4.5", "config", "show"], tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "LOOP_LIMIT_DB" in r.stdout and "--set" in r.stdout
    assert "SLICE_MIN_HITS" in r.stdout and "your fourier.toml" in r.stdout
    assert not (tmp_path / "home" / "run").exists()      # config only reads: nothing left behind
    r = _cli(["--set", "LOOP_LIMIT_DB=4.5", "devices", "list"], tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    resolved = sorted((tmp_path / "home" / "run").glob("resolved-*.json"))
    assert len(resolved) == 1
    probe = tmp_path / "probe.py"          # spawn workers need a real file for __main__
    probe.write_text(
        "import sys, multiprocessing as mp\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from fourier.packs import curate_config as cc\n"
        "def f(_):\n    from fourier.packs import sets\n    return sets.SLICE_MIN_HITS\n"
        "if __name__ == '__main__':\n"
        "    with mp.get_context('spawn').Pool(1) as p:\n"
        "        print(cc.LOOP_LIMIT_DB, p.map(f, [0])[0])\n")
    env = {k: v for k, v in os.environ.items() if k != "FOURIER_CONFIG"}
    env["FOURIER_RESOLVED_CONFIG"] = str(resolved[0])
    out = subprocess.run([sys.executable, str(probe), str(ROOT / "src")], env=env,
                         capture_output=True, text=True, check=False, timeout=120)
    assert out.stdout.split() == ["4.5", "0.25"], out.stderr


def test_cli_bad_config_exits_2(tmp_path):
    r = _cli(["--set", "NOT_A_TUNABLE=1", "config", "show"], tmp_path)
    assert r.returncode == 2 and "unknown tunable" in r.stdout


def test_cli_without_config_changes_nothing(tmp_path):
    r = _cli(["config", "show"], tmp_path, {"FOURIER_CONFIG": "none"})
    assert r.returncode == 0 and "the code defaults" in r.stdout
    assert not (tmp_path / "home" / "run").exists()
