"""The sealed golden harness (scripts/golden_harness.py), with a stand-in for the build."""
import hashlib
import importlib.util
import json
import os
import shlex
import sys
from pathlib import Path

import pytest

REF_RATINGS = b'{"keep": ["a"]}'


def _harness():
    p = Path(__file__).resolve().parents[1] / "scripts" / "golden_harness.py"
    spec = importlib.util.spec_from_file_location("golden_harness", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest(md5="m1"):
    return {"generated": "2000-01-27T05:13:08Z", "ratings_hash": hashlib.sha256(REF_RATINGS).hexdigest()[:16],
            "categories": {"KICKS": {"families": 1, "files": 1, "source_samples": 3, "entries": [
                {"family": "kick", "out": "kick/BD.wav", "src": "/lib/SampleLibrary/A/BD.wav", "out_md5": md5}]}},
            "sets": {"KITS": {"entries": []}, "SLICE": {"entries": []}}}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "cache" / "audio").mkdir(parents=True)
    (home / "cache" / "pitch").mkdir()
    (home / "cache" / "clap_text").mkdir()
    (home / "cache" / "audio" / "x.wav").write_bytes(b"audio")
    (home / "cache" / "render").mkdir()
    (home / "library.db").write_bytes(b"db")
    (home / "library.db-wal").write_bytes(b"recent writes")      # SQLite WAL: frozen with the DB
    (home / "clap_index.npz").write_bytes(b"idx")
    (home / "ratings.json").write_bytes(REF_RATINGS)
    (home / "builds").mkdir()
    (home / "builds" / "20000128T000000Z.json").write_text("{}")   # a newer build: must not be used
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "m8_tracker.lock.json").write_text('{"v": 1}')
    master = tmp_path / "master"
    master.mkdir()
    (master / "manifest.json").write_text(json.dumps(_manifest()))
    ref = master / "manifest.json"
    for k in ("FOURIER_HOME", "FOURIER_RATINGS", "FOURIER_NO_STICKY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(master))            # the "real" master
    return dict(home=home, locks=locks, ref=ref, work=tmp_path / "work", tmp=tmp_path)


def _args(s, *extra):
    return ["--reference", str(s["ref"]), "--home", str(s["home"]), "--work", str(s["work"]),
            "--lock-dir", str(s["locks"]), *extra]


def _standin(s, md5="m1", touch_real=False, fail=False, overrides=None):
    """A build that writes a manifest into $FOURIER_CURATED_DIR and records its env."""
    code = f"""
import json, os, sys
from pathlib import Path
Path(os.environ["FOURIER_HOME"], "env.json").write_text(json.dumps(dict(os.environ)))
m = json.loads(Path({str(s['ref'])!r}).read_text())
m["categories"]["KICKS"]["entries"][0]["out_md5"] = {md5!r}
m["overrides"] = json.loads({json.dumps(overrides or {})!r})
out = Path(os.environ["FOURIER_CURATED_DIR"]); out.mkdir(parents=True)
(out / "manifest.json").write_text(json.dumps(m))
if {touch_real!r}:
    Path({str(s['home'] / 'builds' / 'sneaky.json')!r}).write_text("{{}}")
sys.exit(1 if {fail!r} else 0)
"""
    return shlex.join([sys.executable, "-c", code])


def _tree(p):
    return {str(f.relative_to(p)): f.read_bytes() for f in sorted(p.rglob("*")) if f.is_file()}


def test_setup_freezes_inputs_and_isolates_the_build(setup, capsys):
    before = _tree(setup["home"])
    assert _harness().main(_args(setup, "--setup-only")) == 0
    fh = setup["work"] / "home"
    assert (fh / "library.db").read_bytes() == b"db"
    assert (fh / "library.db-wal").read_bytes() == b"recent writes"
    assert (fh / "clap_index.npz").exists() and (fh / "ratings.json").read_bytes() == REF_RATINGS
    assert (fh / "cache" / "audio" / "x.wav").exists() and (fh / "cache" / "pitch").is_dir()
    assert not (fh / "cache" / "render").exists()                     # not a build input
    assert [p.name for p in (fh / "builds").iterdir()] == ["20000127T051308Z.json"]   # only the reference
    assert (setup["work"] / "locks" / "m8_tracker.lock.json").exists()
    run = json.loads(capsys.readouterr().out)
    assert run["env"]["FOURIER_HOME"] == str(fh)
    assert run["env"]["FOURIER_CURATED_DIR"] == str(setup["work"] / "out" / "FourierCurated")
    assert run["env"]["FOURIER_LOCK_DIR"] == str(setup["work"] / "locks")
    assert "--no-describe" in run["cmd"] and run["cmd"][run["cmd"].index("--db") + 1] == str(fh / "library.db")
    assert _tree(setup["home"]) == before                             # the real home is untouched


def test_identical_build_passes(setup, monkeypatch):
    monkeypatch.setenv("FOURIER_RATINGS", "/somewhere/else.json")     # narrower overrides are dropped
    monkeypatch.setenv("FOURIER_RESOLVED_CONFIG", "/somewhere/resolved.json")   # ...and any config
    assert _harness().main(_args(setup, "--build-cmd", _standin(setup))) == 0
    env = json.loads((setup["work"] / "home" / "env.json").read_text())
    assert "FOURIER_RATINGS" not in env and "FOURIER_NO_STICKY" not in env
    assert env["FOURIER_CONFIG"] == "none" and "FOURIER_RESOLVED_CONFIG" not in env
    rep = json.loads((setup["work"] / "golden_report.json").read_text())
    assert rep["ok"] and rep["ratings_hash_ok"] and rep["build_exit"] == 0


def test_different_build_fails(setup):
    assert _harness().main(_args(setup, "--build-cmd", _standin(setup, md5="m2"))) == 1
    rep = json.loads((setup["work"] / "golden_report.json").read_text())
    assert rep["fields"] == {"out_md5": 1}


def test_no_sticky_and_cold_cache(setup):
    h = _harness()
    assert h.main(_args(setup, "--no-sticky", "--audio-cache", "cold", "--build-cmd", _standin(setup))) == 0
    env = json.loads((setup["work"] / "home" / "env.json").read_text())
    assert env["FOURIER_NO_STICKY"] == "1" and env["FOURIER_NO_AUDIO_CACHE"] == "1"
    assert not (setup["work"] / "home" / "cache" / "audio").exists()


def test_ratings_that_differ_from_the_reference_stop_it(setup):
    (setup["home"] / "ratings.json").write_bytes(b'{"keep": ["b"]}')
    assert _harness().main(_args(setup, "--build-cmd", _standin(setup))) == 3
    assert not setup["work"].exists()                                 # nothing was set up


def test_work_folder_must_be_fresh(setup):
    setup["work"].mkdir()
    (setup["work"] / "old").write_text("x")
    assert _harness().main(_args(setup, "--setup-only")) == 2


def test_touching_the_real_home_is_caught(setup):
    assert _harness().main(_args(setup, "--build-cmd", _standin(setup, touch_real=True))) == 5


def test_failed_build(setup):
    assert _harness().main(_args(setup, "--build-cmd", _standin(setup, fail=True))) == 4


def test_missing_reference(setup):
    assert _harness().main(["--reference", str(setup["tmp"] / "nope.json"), "--home", str(setup["home"]),
                            "--work", str(setup["work"])]) == 2
    assert os.path.exists(setup["home"] / "ratings.json")


def _config(s):
    (s["tmp"] / "mine.yaml").write_text("advanced:\n  LOOP_LIMIT_DB: 2.5\n")
    cfg = s["tmp"] / "fourier.toml"
    cfg.write_text('preset = "balanced"\noverlay = "mine.yaml"\n')          # the code's defaults + mine
    return cfg


def test_config_layers_must_be_exactly_what_the_build_used(setup):
    cfg = _config(setup)
    want = {"curate_config.LOOP_LIMIT_DB": 2.5}
    assert _harness().main(_args(setup, "--config", str(cfg), "--build-cmd", _standin(setup, overrides=want))) == 0
    env = json.loads((setup["work"] / "home" / "env.json").read_text())
    assert env["FOURIER_CONFIG"] == str(cfg.resolve())
    rep = json.loads((setup["work"] / "golden_report.json").read_text())
    assert rep["overrides_ok"] and rep["expected_overrides"] == ["curate_config.LOOP_LIMIT_DB"]


@pytest.mark.parametrize("config, used", [
    (True, {}),                                              # the overlay didn't reach the build
    (True, {"curate_config.LOOP_LIMIT_DB": 3.0}),            # ...or reached it with another value
    (False, {"curate_config.LOOP_LIMIT_DB": 2.5}),           # overrides without --config
])
def test_overrides_other_than_the_configs_fail(setup, config, used):
    extra = ("--config", str(_config(setup))) if config else ()
    assert _harness().main(_args(setup, *extra, "--build-cmd", _standin(setup, overrides=used))) == 1
    assert not json.loads((setup["work"] / "golden_report.json").read_text())["overrides_ok"]


def test_a_bad_config_stops_it_before_setup(setup):
    (setup["tmp"] / "bad.toml").write_text("[advanced]\nNOT_A_TUNABLE = 1\n")
    assert _harness().main(_args(setup, "--config", str(setup["tmp"] / "bad.toml"))) == 2
    assert not setup["work"].exists()
