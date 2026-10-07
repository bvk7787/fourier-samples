"""Byte-identical copies, a renamed library folder and the read-only commands, through the real
CLI (the sandbox and stand-in CLAP of tests/test_first_run.py), without Sononym or Live:

- a byte-identical copy of a file the master holds: `why` names the file it repeats, `why
  --unrecognized` never lists it as placed nowhere, and the build log counts it;
- a library folder renamed: the scan counts its files as moved (not missing), `db-stats
  --missing` agrees, and the next build keeps the same files (the master doesn't churn; its
  CHANGELOG says the library moved, nothing else);
- `tools analyze --status` shows each step's coverage of the samples it covers, and
  `db-stats --disagreements` says Sononym isn't there;
- doctor and `build --dry-run` leave nothing in the home; `why` on a file rated Drop says
  that and no list of possible reasons."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_first_run import Sandbox, _flat, make_library  # noqa: E402
from fourier.packs import manifests

# one sandbox, built once, that the tests below change in turn: one worker runs them all, in
# order (pytest -n: --dist loadgroup, conftest.py)
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("copies_and_moves_cli")]

COPY = "Vendor B/Drums/Kicks/Kick 01 Copy.wav"


def make(lib: Path) -> int:
    n = make_library(lib)
    (lib / COPY).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lib / "Vendor A" / "Drums" / "Kicks" / "Kick 01.wav", lib / COPY)
    return n + 1


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    b = Sandbox(tmp_path_factory.mktemp("copies-moves"), make=make)
    b.run("setup", "--yes", "--library", str(b.lib), "--device", "digitakt_2", "--preset", "balanced",
          "--no-clap")
    b.build_log = _flat(b.run("build", "--all", "-j", "1", "--no-describe"))
    return b


def _entries(b):
    man = manifests.read(b.master / "manifest.json")
    return {e["src"]: (c, e["out"], e["out_md5"]) for c, cd in man["categories"].items()
            for e in cd.get("entries") or ()}


def test_a_byte_identical_copy_is_named_and_counted(box):
    b = box
    srcs = _entries(b)
    pair = [str(b.lib / "Vendor A/Drums/Kicks/Kick 01.wav"), str(b.lib / COPY)]
    held = [p for p in pair if p in srcs]
    assert len(held) == 1, held                         # one of the two copies in the master
    other = next(p for p in pair if p not in srcs)
    assert "DONE KICKS" in b.build_log and "1 twins collapsed" in b.build_log
    out = _flat(b.run("why", "--detail", other.split("SampleLibrary/")[1]))
    kept = held[0].split("SampleLibrary/")[1]
    assert f"a byte-identical copy of {kept}, which the master holds (KICKS/" in out, out
    assert "a duplicate copy, clipped, too long, or rated Drop" not in out
    out = _flat(b.run("why", "--detail", "--unrecognized"))
    assert other.rsplit("/", 1)[1] not in out, out


def test_a_renamed_library_folder_moves_files_and_the_master_keeps_them(box):
    b = box
    before = {s.replace("/Vendor C/", "/Vendor C Audio/"): v for s, v in _entries(b).items()}
    moved = sum(1 for s in _entries(b) if "/Vendor C/" in s)
    on_disk = sum(1 for p in (b.lib / "Vendor C").rglob("*.wav"))
    (b.lib / "Vendor C").rename(b.lib / "Vendor C Audio")
    b.print = __import__("synthetic_build").library_fingerprint(b.lib)    # the user's rename
    out = _flat(b.run("tools", "scan"))
    assert f"{on_disk} file(s) moved or renamed in the library" in out, out
    assert "weren't found" not in out
    out = _flat(b.run("tools", "db-stats", "--missing"))
    assert f"{on_disk} moved (old rows)" in out and "All" in out and "files present" in out, out
    assert "Missing files" not in out
    b.run("build", "--all", "-j", "1", "--no-describe")
    assert moved and _entries(b) == before              # the same files, names and audio
    log = (b.master / "CHANGELOG.md").read_text()
    assert f"{moved} files moved in the library, master unchanged." in log, log
    assert "new folder" not in log and "folder gone" not in log
    # a scan again: the old rows are still moved, not missing
    out = _flat(b.run("tools", "scan"))
    assert "weren't found" not in out


def test_status_and_disagreements_say_what_they_cover(box):
    b = box
    out = _flat(b.run("tools", "analyze", "--status"))
    assert "Phase" not in out and "Tier" not in out and "to go" not in out
    assert "Root notes" in out and "pYIN" in out and "Keys" in out          # (the table may wrap)
    assert "Analysis of this library" in out
    out = _flat(b.run("tools", "db-stats", "--disagreements"))
    assert "Sononym not found: nothing to compare" in out
    assert "both are needed" not in out and "fourier tools analyze" not in out


def test_read_only_commands_leave_nothing_in_the_home(box):
    b = box
    run = b.home / "run"
    shutil.rmtree(run, ignore_errors=True)
    for args in (("doctor",), ("build", "--dry-run"), ("build", "--all", "--dry-run"),
                 ("why", "Kick 01"), ("tools", "analyze", "--status"), ("render", "digitakt_2", "--check")):
        out = _flat(b.run(*args, ok=None))
        assert not run.exists(), (args, list(run.iterdir()), out[-500:])
    assert "Nothing was written" in _flat(b.run("build", "--dry-run"))


def test_why_gives_a_known_reason_alone(box):
    b = box
    name = "Clap 03"
    b.run("review", "rate", name, "drop")
    out = _flat(b.run("why", "--detail", name))
    assert "rated Drop" in out
    assert "not among the candidates the last build's gates saw" not in out, out
