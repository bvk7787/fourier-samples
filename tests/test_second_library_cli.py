"""A second library in the same home, through the real CLI (the sandbox and stand-in CLAP of
tests/test_first_run.py): another config (`--config`) with its own library and master scans
into the same database; each config's builds, library scale, doctor, `build --dry-run`, `why
--unrecognized`, search and build history read only the samples under its own library
folders."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_first_run import SR, Sandbox, _flat, _sounds  # noqa: E402

# one sandbox, built once, that the tests below change in turn: one worker runs them all, in
# order (pytest -n: --dist loadgroup, conftest.py)
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("second_library_cli")]

OTHER = "Zeta Works"            # the second library's only vendor


def make_other(root: Path) -> int:
    kick, snare, hat, clap, loop, bass, pad = _sounds(np.random.default_rng(23))
    n = 0
    for i in range(6):
        for rel, y in ((f"{OTHER}/Kicks/Thump {i + 1}.wav", kick(i + 3)),
                       (f"{OTHER}/Pads/Haze {i + 1} A.wav", pad(i + 3)),
                       (f"{OTHER}/Unsorted/{i + 1:03d}.wav", snare(i + 3))):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(p, (0.8 * y / np.max(np.abs(y))).astype("float32"), SR, subtype="PCM_16")
            n += 1
    return n


@pytest.fixture(scope="module")
def two(tmp_path_factory):
    b = Sandbox(tmp_path_factory.mktemp("second-library"))
    b.other = b.tmp / "other" / "SampleLibrary"
    b.n_other = make_other(b.other)
    b.master_b = b.tmp / "out-b" / "FourierCurated"
    b.cfg_b = b.tmp / "b.toml"
    b.cfg_b.write_text(f'library = ["{b.other}"]\ndevices = ["generic_folder"]\npreset = "balanced"\n'
                       f'[output]\nmaster = "{b.master_b}"\n')
    b.run("setup", "--yes", "--library", str(b.lib), "--device", "digitakt_2", "--preset", "balanced",
          "--no-clap")
    return b


def _run_b(b, *args, ok=True):
    """A command of the second library's config: its own master (the sandbox points
    $FOURIER_CURATED_DIR at the first one's)."""
    import subprocess
    env = b.env()
    env["FOURIER_CURATED_DIR"] = str(b.master_b)
    r = subprocess.run([sys.executable, str(b.runner), "--config", str(b.cfg_b), *args], env=env,
                       cwd=b.tmp, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, out[-3000:]
    if ok is not None:
        assert (r.returncode == 0) == ok, out[-3000:]
    return out


def _srcs(master):
    man = json.loads((master / "manifest.json").read_text())
    return [e["src"] for c in man["categories"].values() for e in c.get("entries") or ()]


def test_a_second_library_shares_the_home_and_stays_out(two):
    b = two
    out = _flat(_run_b(b, "tools", "scan"))
    assert f"{b.n_other} new" in out
    out = _flat(b.run("build", "--all", "-j", "1", "--no-describe"))
    assert f"{b.n_other} samples from other libraries left out" in out, out[-3000:]
    assert f"Your library has {b.n_files} usable samples" in out
    srcs = _srcs(b.master)
    assert srcs and all(s.startswith(str(b.lib) + "/") for s in srcs)
    out = _flat(b.run("doctor"))
    assert f"OK samples in the database: {b.n_files}" in out
    assert f"INFO other libraries: {b.n_other} more samples" in out
    out = _flat(_run_b(b, "doctor", ok=None))
    assert f"samples in the database: {b.n_other}" in out and "other libraries" in out
    out = _flat(b.run("build", "--dry-run"))
    assert f"{b.n_other} samples from other libraries left out" in out
    assert f"Your library has {b.n_files} usable samples" in out
    out = b.run("why", "--unrecognized")
    assert OTHER not in out and "Unsorted" not in out
    out = b.run("search", "--min-dur", "0.01", "--top", "200")
    assert OTHER not in out
    out = _run_b(b, "search", "--min-dur", "0.01", "--top", "200")
    assert OTHER in out and "Vendor A" not in out


def test_each_library_keeps_its_own_build_history(two):
    b = two
    if not (b.master / "manifest.json").exists():
        pytest.skip("needs the build above")
    out = _flat(_run_b(b, "build", "--all", "-j", "1", "--no-describe"))
    assert "New master in place" in out
    assert all(s.startswith(str(b.other) + "/") for s in _srcs(b.master_b))
    # the first library's next build compares with its own last build, not the other's
    before = sorted(_srcs(b.master))
    out = _flat(b.run("build", "--all", "-j", "1", "--no-describe", "--no-scan"))
    assert sorted(_srcs(b.master)) == before
    log = (b.master / "CHANGELOG.md").read_text()
    assert "0 files added, 0 removed" in log, log


def _run_c(b, cfg, *args, ok=True):
    """A command of a config written by `setup --to`: its own [output] (no $FOURIER_CURATED_DIR)."""
    import subprocess
    env = b.env()
    env["FOURIER_CURATED_DIR"] = ""
    r = subprocess.run([sys.executable, str(b.runner), "--config", str(cfg), *args], env=env,
                       cwd=b.tmp, capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    assert "Traceback" not in out, out[-3000:]
    if ok is not None:
        assert (r.returncode == 0) == ok, out[-3000:]
    return out


def test_setup_to_another_config_gives_it_its_own_folders_and_reads_nothing_of_the_other_library(two):
    """`setup --to` a config other than the default one: its master, renders and releases are
    named after it (two libraries never share a master), and until its first scan it reads
    none of the other library's samples: doctor says the first build scans it, `build
    --dry-run` sizes it from its files, `why --unrecognized` says it isn't scanned yet."""
    import tomllib
    b = two
    if not (b.master / "manifest.json").exists():
        pytest.skip("needs the first library's build above")
    third = b.tmp / "third" / "SampleLibrary"
    n = make_other(third)
    cfg = b.tmp / "second.toml"
    out = _flat(b.run("setup", "--yes", "--to", str(cfg), "--library", str(third), "--device",
                      "generic_folder", "--preset", "balanced", "--no-clap", "--no-build"))
    assert "isn't the default config" in out and "folders of their own" in out
    doc = tomllib.loads(cfg.read_text())
    music = b.user / "Music"
    assert doc["output"] == {"master": str(music / "FourierCurated-second"),
                             "renders": str(music / "FourierRenders-second"),
                             "publish": str(music / "Fourier-second")}
    out = _flat(_run_c(b, cfg, "doctor", ok=None))
    assert "NEXT samples in the database: none of this library's yet" in out, out
    assert "other libraries" in out, out
    assert f"OK output: master: {music / 'FourierCurated-second'}" in out, out
    assert f"analyzes {n} new samples first" in out, out
    out = _flat(_run_c(b, cfg, "build", "--dry-run"))
    assert "This config's library isn't scanned yet" in out
    assert f"Your library has {n} audio files" in out and "from the files in the library folders" in out, out
    out = _flat(_run_c(b, cfg, "why", "--unrecognized"))
    assert "isn't scanned yet" in out and "Vendor A" not in out and OTHER not in out
    out = _flat(_run_c(b, cfg, "tools", "analyze", "--status"))
    assert "Analysis of this library (0 samples)" in out
    out = _flat(_run_c(b, cfg, "build", "--all", "--no-scan", ok=False))
    assert "nothing scanned yet" in out
    assert not (music / "FourierCurated-second").exists()


def test_a_build_into_another_librarys_master_stops_before_anything_changes(two):
    """A config whose master is the first library's (an [output] left at the shared default):
    doctor FAILs it and its build stops before scanning, the first library's master as it was."""
    b = two
    if not (b.master / "manifest.json").exists():
        pytest.skip("needs the first library's build above")
    man = json.loads((b.master / "manifest.json").read_text())
    assert man["library"]["folders"] == [str(b.lib)]           # recorded by the build
    before = (b.master / "manifest.json").read_bytes()
    cfg = b.tmp / "shared.toml"
    cfg.write_text(f'library = ["{b.other}"]\ndevices = ["generic_folder"]\npreset = "balanced"\n'
                   f'[output]\nmaster = "{b.master}"\n')
    out = _flat(_run_c(b, cfg, "doctor", ok=False))
    assert "FAIL output: master" in out and "holds another library's build" in out
    out = _flat(_run_c(b, cfg, "build", "--all", "-j", "1", "--no-describe", ok=False))
    assert "holds another library's build" in out and "Build stopped: nothing in" in out
    assert "Scan the library" not in out
    assert (b.master / "manifest.json").read_bytes() == before
