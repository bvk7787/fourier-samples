"""fourier demo (fourier/demo.py): the pipeline on the synthetic library, in a sandbox folder
that is the only place it reads or writes."""
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from click.testing import CliRunner

from fourier import demo as D
from fourier.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _has(out, text):
    """Whether out says text, wherever Rich wrapped it (a long temporary path is folded
    mid-path at the console's width)."""
    return "".join(text.split()) in "".join(out.split())


def _tree(root: Path) -> dict:
    """Every path under root with its size and mtime (folders: None)."""
    return {str(p.relative_to(root)): (None if p.is_dir() else (p.stat().st_size, p.stat().st_mtime_ns))
            for p in sorted(root.rglob("*"))}


def test_prepare_refuses_a_folder_that_isnt_a_demo(tmp_path):
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "song.wav").write_bytes(b"x")
    with pytest.raises(D.DemoError, match="isn't an earlier demo"):
        D.prepare(tmp_path / "mine")
    assert [p.name for p in (tmp_path / "mine").iterdir()] == ["song.wav"]
    (tmp_path / "a-file").write_text("x")
    with pytest.raises(D.DemoError, match="is a file"):
        D.prepare(tmp_path / "a-file")


def test_prepare_takes_a_new_or_empty_folder_and_marks_it(tmp_path):
    for root in (tmp_path / "new" / "demo", tmp_path / "empty"):
        if root.name == "empty":
            root.mkdir()
        assert D.prepare(root) == root.resolve()
        assert [p.name for p in root.iterdir()] == [D.MARKER]


def test_a_rerun_replaces_the_demo_and_leaves_anything_else(tmp_path):
    root = D.prepare(tmp_path / "demo")
    for name in ("library", "master", "renders/m8_tracker", "home/cache"):
        (root / name).mkdir(parents=True)
        (root / name / "f.wav").write_bytes(b"x")
    (root / "fourier.toml").write_text("x")
    (root / "my notes.txt").write_text("keep me")
    D.prepare(root)
    assert sorted(p.name for p in root.iterdir()) == sorted([D.MARKER, "my notes.txt"])


def test_the_demo_config_names_only_the_sandbox(tmp_path):
    where = D.layout(tmp_path)
    cfg = tomllib.loads(D.config_text(where))
    assert cfg["library"] == [str(tmp_path / "library")]
    assert cfg["devices"] == list(D.DEVICES) and cfg["preset"] == D.PRESET and cfg["scale"] == D.SCALE
    assert all(Path(v).parent == tmp_path for v in cfg["output"].values())
    assert cfg["advanced"]["BUDGETS"] == D.BUDGETS
    env = D.env_text(where)
    assert f"FOURIER_HOME={where['home']}" in env and f"FOURIER_CONFIG={where['config']}" in env


def test_the_demo_builds_what_the_ci_golden_builds():
    """The demo's budgets and preset are the synthetic golden's, so CI's golden build covers it."""
    golden = tomllib.loads((ROOT / "tests" / "golden" / "synthetic.toml").read_text())
    assert golden["advanced"]["BUDGETS"] == D.BUDGETS
    assert golden.get("preset", D.PRESET) == D.PRESET
    assert golden["scale"] == D.SCALE == "off"


def test_the_sandbox_environment_is_put_back(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_LIBRARY", "/somewhere/else")
    monkeypatch.setenv("FOURIER_NO_AUDIO_CACHE", "1")
    before = dict(os.environ)
    where = D.layout(tmp_path)
    with D.sandbox_env(where):
        assert os.environ["FOURIER_HOME"] == str(where["home"])
        assert os.environ["FOURIER_CONFIG"] == str(where["config"])
        assert "FOURIER_LIBRARY" not in os.environ and "FOURIER_LOCK_DIR" not in os.environ
        assert os.environ["FOURIER_NO_AUDIO_CACHE"] == "1"           # a switch, not a place
    assert dict(os.environ) == before


def test_the_stand_in_clap_is_put_back():
    from fourier.analysis import clap_features as CF
    real = CF.embed_text
    with D.standin_clap():
        assert CF.embed_text is D.fake_embed_text
        v = CF.embed_text("dark kick")
        assert v.shape == (D.DIM,) and abs(float((v * v).sum()) - 1) < 1e-5
    assert CF.embed_text is real


def test_demo_refuses_a_foreign_folder_without_opening_the_users_home(tmp_path, monkeypatch):
    home = tmp_path / "fourier-home"
    monkeypatch.setenv("FOURIER_HOME", str(home))
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "keep.txt").write_text("x")
    r = CliRunner().invoke(main, ["demo", "--dir", str(tmp_path / "mine")])
    assert r.exit_code == 1 and "isn't an earlier demo" in r.output
    assert not home.exists()                 # the root didn't open the user's database
    assert [p.name for p in (tmp_path / "mine").iterdir()] == ["keep.txt"]


@pytest.mark.slow
def test_demo_builds_and_renders_inside_its_folder_only(tmp_path):
    """The real thing, in a fresh process with a fake HOME: a user config and a ./fourier.toml
    the demo must not read (both would fail a build), and nothing written outside --dir."""
    home, cwd, root = tmp_path / "home", tmp_path / "cwd", tmp_path / "demo"
    (home / ".config" / "fourier").mkdir(parents=True)
    (home / ".config" / "fourier" / "fourier.toml").write_text('library = ["/no/such/library"]\n'
                                                                "[advanced]\nNOT_A_TUNABLE = 1\n")
    (home / "Music").mkdir()
    cwd.mkdir()
    (cwd / "fourier.toml").write_text("[advanced]\nNOT_A_TUNABLE = 1\n")
    home_before, cwd_before = _tree(home), _tree(cwd)
    env = {k: v for k, v in os.environ.items() if not k.startswith("FOURIER_")}
    env.update(HOME=str(home), PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, "-m", "fourier", "demo", "--dir", str(root)], cwd=cwd,
                       env=env, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "stand-in" in r.stdout and "4/4" in r.stdout
    assert _has(r.stdout, str(root / "master")) and _has(r.stdout, f". {root / 'demo.env'}")
    assert (root / "master" / "manifest.json").exists()
    assert any((root / "master" / "KICKS").rglob("*.wav"))
    for device in D.DEVICES:
        assert sum(1 for _ in (root / "renders" / device).rglob("*.wav")) > 200
    assert (root / "home" / "library.db").exists() and (root / "demo.log").stat().st_size
    # nothing outside the demo folder: not the fake home (bar a library's own cache), not cwd
    ignore = lambda t: {k: v for k, v in t.items() if not k.startswith(".cache")}
    assert ignore(_tree(home)) == ignore(home_before)
    assert _tree(cwd) == cwd_before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cwd", "demo", "home"]
