"""Overridable tunables (fourier/settings.py): defaults unchanged, overrides apply everywhere."""
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from fourier import settings
from fourier.settings import ConfigError, decode, encode, write_resolved

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((ROOT / "tests" / "data" / "tunables_snapshot.json").read_text())
CLASSES = yaml.safe_load((ROOT / "config" / "tunables.yaml").read_text())


def test_defaults_match_the_snapshot_in_value_and_type():
    """With no resolved config every tunable is exactly what it was before it became
    overridable (tests/data/tunables_snapshot.json; update it only for a deliberate change)."""
    assert not os.environ.get(settings.ENV)
    diffs = []
    for key, snap in SNAPSHOT.items():
        module, name = key.split(".")
        v = getattr(importlib.import_module(f"fourier.packs.{module}"), name)
        if type(v).__name__ != snap["type"] or encode(v) != snap["value"]:
            diffs.append(key)
    assert not diffs, diffs
    assert set(SNAPSHOT) == {f"{m}.{n}" for m, names in CLASSES.items() for n in names}


def test_every_non_internal_tunable_is_overridable():
    settings.load_all_and_check()
    took = {f"{m}.{n}" for m, names in settings._seen.items() for n in names}
    want = {f"{m}.{n}" for m, names in CLASSES.items() for n, c in names.items()
            if not str(c).startswith("internal")}
    assert took == want


@pytest.mark.parametrize("v", [
    3.0, 16, None, True, "x", [1, "a"], (1, 2, 4), {"a", "b"}, frozenset({"z"}),
    {"kick": -9.0, 3: [1, 2]}, re.compile(r"(?<![a-z])amen", re.IGNORECASE),
    [(re.compile("909"), "909")], {"CAT": {"phrases": ("a",), "noise": r"\bhat\b"}},
])
def test_encoding_round_trips(v):
    d = decode(json.loads(json.dumps(encode(v))))
    assert type(d) is type(v) and encode(d) == encode(v)


def test_hash_is_stable_and_order_free():
    a = {"curate_config.X": {"b", "a"}, "sets.Y": 1.0}
    b = {"sets.Y": 1.0, "curate_config.X": {"a", "b"}}
    assert settings.config_hash(a) == settings.config_hash(b)
    assert settings.config_hash(a) != settings.config_hash({**a, "sets.Y": 2.0})


_PROBE = r"""
import json, re, sys
sys.path.insert(0, sys.argv[1])
from fourier import settings
from fourier.packs import curate, curate_config as cc, naming, sets, audiocache
settings.load_all_and_check()
print(json.dumps(dict(
    loop_limit=[cc.LOOP_LIMIT_DB, type(cc.LOOP_LIMIT_DB).__name__, curate.LOOP_LIMIT_DB],
    acoustic_tags=cc.ACOUSTIC_ABLETON_TAGS[:2],
    limit_default=curate._limit.__defaults__[0],
    max_gain_default=curate._normalize.__defaults__[0],
    slice_min_hits=sets.SLICE_MIN_HITS, clap_z=naming.CLAP_Z,
    demo=[cc.DEMO_PATH.pattern, bool(cc.DEMO_PATH.flags & re.I)],
    fp=audiocache.dsp_fingerprint(), th=settings.tunables_hash(),
    meta=curate._build_meta()["overrides"],
)))
"""


def _probe(tmp_path, values):
    env = {k: v for k, v in os.environ.items() if k != settings.ENV}
    if values is not None:
        path = tmp_path / "resolved.json"
        write_resolved(path, values)
        env[settings.ENV] = str(path)
    r = subprocess.run([sys.executable, "-c", _PROBE, str(ROOT / "src")], env=env,
                       capture_output=True, text=True, check=False)
    return r


def test_overrides_reach_derived_values_defaults_and_other_modules(tmp_path):
    base = json.loads(_probe(tmp_path, None).stdout)
    r = _probe(tmp_path, {
        "curate_config.LOOP_LIMIT_DB": 6,                       # an int for a float default
        "curate_config.ORCH_ABLETON_TAGS": ["Violin", "Cello"],
        "curate_config.LIMIT_MS": 7.5,
        "curate.MAX_GAIN_DB": 12.0,
        "sets.SLICE_MIN_HITS": 0.25,
        "naming.CLAP_Z": 2.0,
        "curate_config.DEMO_PATH": re.compile("demo-only", re.IGNORECASE),
    })
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["loop_limit"] == [6.0, "float", 6.0]              # coerced; curate's import follows
    assert got["acoustic_tags"] == ["Violin", "Cello"]           # derived from the override
    assert got["limit_default"] == 7.5                           # default argument bound at def time
    assert got["max_gain_default"] == 12.0
    assert got["slice_min_hits"] == 0.25 and got["clap_z"] == 2.0
    assert got["demo"] == ["demo-only", True]
    assert got["fp"] != base["fp"]                               # the audio cache sees DSP edits


@pytest.mark.parametrize("values, message", [
    ({"curate_config.LOOP_LIMIT_DB": "loud"}, "expected a float"),
    ({"curate_config.DEMO_PATH": "demo"}, "expected a Pattern"),
    ({"curate_config.LOOP_LIMIT_DBB": 3.0}, "unknown or non-overridable"),
    ({"curate_config.ABLETON_TAGS_BY_CATEGORY": {}}, "unknown or non-overridable"),  # derived
])
def test_bad_overrides_fail_loudly(tmp_path, values, message):
    r = _probe(tmp_path, values)
    assert r.returncode != 0 and message in r.stderr


def test_bad_file(tmp_path, monkeypatch):
    p = tmp_path / "resolved.json"
    p.write_text(json.dumps({"version": 99, "values": {}}))
    monkeypatch.setenv(settings.ENV, str(p))
    settings.reset()
    try:
        with pytest.raises(ConfigError, match="version"):
            settings.overrides()
    finally:
        monkeypatch.delenv(settings.ENV)
        settings.reset()
    assert settings.overrides() == {}


def test_taste_edits_keep_the_audio_cache_warm_and_dsp_edits_miss_it(tmp_path):
    """The audio cache key moves only with what changes processed audio; the tunables
    hash and the manifest's overrides record every change."""
    base = json.loads(_probe(tmp_path, None).stdout)
    assert base["meta"] == {}
    taste = json.loads(_probe(tmp_path, {"naming.CLAP_Z": 2.0, "sets.KIT_COUNT": 12}).stdout)
    assert taste["fp"] == base["fp"] and taste["th"] != base["th"]
    assert set(taste["meta"]) == {"naming.CLAP_Z", "sets.KIT_COUNT"}
    dsp = json.loads(_probe(tmp_path, {"curate_config.EDGE_FADE_MS": 1.0}).stdout)
    assert dsp["fp"] != base["fp"]
    assert json.loads(_probe(tmp_path, None).stdout)["th"] == base["th"]   # stable
