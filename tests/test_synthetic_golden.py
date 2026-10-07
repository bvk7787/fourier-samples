"""The synthetic library (fourier/synthlib.py) and its golden build (tests/golden/)."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fourier import synthlib
from fourier.packs import manifests

ROOT = Path(__file__).resolve().parents[1]


def _digest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*.wav"))}


def test_the_library_is_the_same_every_time(tmp_path):
    a = synthlib.generate(tmp_path / "a")
    b = synthlib.generate(tmp_path / "b")
    assert [f.rel for f in a] == [f.rel for f in b]
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
    assert len(a) > 250 and len({f.rel for f in a}) == len(a)


def test_the_library_covers_every_category_and_the_defect_cases(tmp_path):
    files = synthlib.generate(tmp_path)
    cats = {c for f in files for c in f.categories}
    assert {"Perc Kicks", "Perc Hats & Shakers", "Tone Pads & Textures", "XFX Sweeps & Lasers"} <= cats
    assert {f.bpm for f in files if f.bpm} == {90.0, 128.0, 174.0}
    rels = {f.rel.rsplit("/", 1)[-1] for f in files}
    assert {"Flat Lead C.wav", "Hot Start Kick.wav", "Long Tail Pad.wav", "Phase Flipped Pad.wav"} <= rels
    assert next(f for f in files if "Phase Flipped" in f.rel).channels == 2
    # four packs per kind, so no vendor cap empties a folder
    assert len({f.rel.split("/")[0] for f in files if "/Kicks/" in f.rel}) == 4


@pytest.mark.skipif(not os.environ.get("FOURIER_GOLDEN"), reason="set FOURIER_GOLDEN=1 (CI runs it as its own step)")
@pytest.mark.slow
def test_synthetic_build_matches_the_golden(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "golden" / "synthetic_build.py"),
                        str(tmp_path / "work"), "--check"], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


@pytest.mark.slow
def test_the_synthetic_library_scaled_builds_every_category(tmp_path):
    """The scaled path end to end (synthetic_build.py --scale): the style's own budgets (about
    11,600 files) scaled to the 280-file library, so every category keeps what it has in one
    or two folders (a folder per tempo range for the loops), kits and slices are made, and
    verify passes against the recorded budgets. The golden above builds with scale = "off"."""
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "golden" / "synthetic_build.py"),
                        str(tmp_path / "work"), "--scale"], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "synthetic --scale: factor 0.00" in r.stdout and "19 of 19 categories" in r.stdout
    man = manifests.read(tmp_path / "work" / "out" / "FourierCurated" / "manifest.json")
    assert man["scale"]["samples"] == 280 and set(man["sets"]) == {"KITS", "SLICE"}
    folders = {c: len({e["family"] for e in cd["entries"]}) for c, cd in man["categories"].items()}
    assert max(folders.values()) <= 3 and sum(folders.values()) < 30, folders
    assert all(len(cd["entries"]) <= cd["budget"] <= 24 for cd in man["categories"].values())
