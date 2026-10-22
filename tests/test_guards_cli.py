"""The guards, as the commands run them: `fourier build --out`, `fourier sync [--delete]` and
`fourier publish --to`. Each refusal is checked by its message AND by every file still being
there, so a guard call removed from the CLI fails a test here."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from fourier import safety
from fourier.cli import main
from fourier.packs import manifests


def _flat(text: str) -> str:
    return " ".join(text.split())


def _files(root) -> dict[str, bytes]:
    root = Path(root)
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    from fourier import places
    from fourier.db import session as sess
    home = tmp_path / "home"
    lib = home / "Music" / "Samples"
    (lib / "Acme").mkdir(parents=True)
    pub = tmp_path / "Pub"
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(f'preset = "balanced"\nlibrary = ["{lib}"]\n[output]\nmaster = "{home}/Music/FourierCurated"\n'
                   f'publish = "{pub}"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fourier-home"))
    monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    monkeypatch.setenv("FOURIER_LOCK_DIR", str(pub / "devices"))
    monkeypatch.setenv("FOURIER_RATINGS", str(tmp_path / "ratings.json"))
    monkeypatch.delenv("FOURIER_LIBRARY", raising=False)
    monkeypatch.delenv("FOURIER_CURATED_DIR", raising=False)
    # the tests' cards, as mounted volumes (sync refuses a folder that isn't one)
    import fourier.platforms as plat
    monkeypatch.setattr(plat, "mount_points", lambda: [tmp_path / n for n in ("CARD", "CARD_A", "CARD_B")])
    places.reset()
    old = (sess._engine, sess._SessionLocal)
    yield SimpleNamespace(tmp=tmp_path, home=home, lib=lib, pub=pub, db=str(tmp_path / "lib.db"),
                          master=home / "Music" / "FourierCurated")
    sess._engine, sess._SessionLocal = old
    places.reset()


def _run(env, *args):
    return CliRunner().invoke(main, ["--db", env.db, *args])


def _master(root, files=(("KICKS", "fam", "k1.wav"),), user=()):
    """A Fourier master (manifest entries with md5s), plus the files a user added."""
    cats = {}
    for cat, fam, fn in files:
        p = Path(root, cat, fam, fn)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"{cat}/{fn}".encode())
        cats.setdefault(cat, {"entries": []})["entries"].append(
            dict(family=fam, out=f"{fam}/{fn}", src=f"/lib/{fn}",
                 out_md5=hashlib.md5(p.read_bytes()).hexdigest()))
    for c in cats.values():
        c["files"] = len(c["entries"])
        c["families"] = len({e["family"] for e in c["entries"]})
    Path(root, "manifest.json").write_text(json.dumps(
        dict(fourier_manifest=2, generated="2000-01-02T00:00:00Z", categories=cats)))
    for rel in user:
        Path(root, rel).parent.mkdir(parents=True, exist_ok=True)
        Path(root, rel).write_bytes(b"the user's own work")
    return Path(root)


# ---------------------------------------------------------------------------
# fourier build --out
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("args", [["build", "KICKS"], ["build", "--all"], ["build", "--base", "v1"]])
def test_build_refuses_a_master_holding_a_user_file(env, args):
    m = _master(env.master, user=["KICKS/fam/my_resample.wav", "MY SET/live.wav"])
    for rel in ("KICKS/fam/Ableton Folder Info/x.xmp", "KICKS/fam/k1.wav.asd", ".DS_Store",
                "CHANGELOG.md", "KICKS/_manifest.json"):          # Live's and Fourier's own: fine
        (m / rel).parent.mkdir(parents=True, exist_ok=True)
        (m / rel).write_text("x")
    before = _files(m)
    res = _run(env, *args, "--out", str(m))
    out = _flat(res.output)
    assert res.exit_code == 1, res.output
    assert "2 file(s) Fourier didn't make" in out and "KICKS/fam/my_resample.wav" in out \
        and "MY SET/live.wav" in out and "k1.wav.asd" not in out
    assert _files(m) == before
    assert not Path(str(m) + ".next").exists() and not Path(str(m) + ".prev").exists()


def _says_release(text: str) -> bool:
    """The refusal names a release or the releases folder (not just a path that spells it)."""
    t = _flat(text)
    return "a release" in t or "the releases" in t


def test_build_refuses_a_release(env):
    rel = _master(env.pub / "releases" / "v1")
    (env.pub / "releases" / "LATEST.txt").write_text("v1\n")
    before = _files(env.pub)
    for out in (rel, rel / "KICKS", env.pub / "releases"):
        res = _run(env, "build", "KICKS", "--out", str(out))
        assert res.exit_code == 1 and _says_release(res.output), res.output
    assert _files(env.pub) == before


def test_a_release_published_elsewhere_stays_protected(env, monkeypatch):
    m = _master(env.master)
    elsewhere = env.tmp / "Elsewhere"
    res = _run(env, "publish", "--from", str(m), "--to", str(elsewhere), "--no-verify")
    assert res.exit_code == 0, res.output
    v1 = elsewhere / "releases" / "v1"
    assert manifests.read(v1 / "manifest.json")["release"] == "v1"
    assert str(v1.resolve()) in safety.published_record().read_text()
    moved = env.tmp / "Moved" / "v1"                          # a copy moved anywhere else
    moved.parent.mkdir()
    os.rename(v1, moved)
    os.makedirs(v1)
    for p in ("manifest.json", "KICKS/fam/k1.wav"):
        (v1 / p).parent.mkdir(parents=True, exist_ok=True)
        (v1 / p).write_bytes((moved / p).read_bytes())
    man = manifests.read(v1 / "manifest.json")
    del man["release"]                                        # only the publish record knows v1
    (v1 / "manifest.json").write_text(json.dumps(man))
    (elsewhere / "releases" / "LATEST.txt").unlink()
    before = (_files(v1), _files(moved))
    for out in (v1, v1 / "KICKS", moved):
        res = _run(env, "build", "KICKS", "--out", str(out))
        assert res.exit_code == 1 and _says_release(res.output), (out, res.output)
    render = _render(env)
    import fourier.platforms as plat
    monkeypatch.setattr(plat, "mount_points", lambda: [v1])   # as a card, so the release guard decides
    res = _run(env, "sync", "m8_tracker", str(v1), "--from", str(render), "--delete")
    assert res.exit_code == 1 and _says_release(res.output), res.output
    assert (_files(v1), _files(moved)) == before


def test_build_refuses_the_library(env):
    before = _files(env.home)
    for out in (env.lib / "Curated", env.lib, env.lib.parent):
        res = _run(env, "build", "KICKS", "--out", str(out))
        # "a library folder", not "library": the test's own tmp path spells library
        assert res.exit_code == 1 and "a library folder" in _flat(res.output), (out, res.output)
    assert _files(env.home) == before and not (env.lib / "Curated").exists()


def test_build_refuses_the_library_spelled_in_another_case(env, monkeypatch):
    from tests.test_safety import _ci_stat
    monkeypatch.setattr(safety, "_stat", _ci_stat)            # as APFS and FAT compare names
    monkeypatch.setattr(safety, "_case_insensitive", lambda d: True)
    res = _run(env, "build", "KICKS", "--out", str(env.home / "music" / "SAMPLES" / "Curated"))
    assert res.exit_code == 1 and "a library folder" in _flat(res.output), res.output
    assert not (env.home / "music" / "SAMPLES" / "Curated").exists()   # case-insensitive disks too


def test_build_out_a_file_is_refused_without_a_traceback(env):
    f = env.tmp / "notes.txt"
    f.write_text("mine")
    res = _run(env, "build", "KICKS", "--out", str(f))
    assert res.exit_code == 1 and "is a file" in _flat(res.output), res.output
    assert isinstance(res.exception, SystemExit) and f.read_text() == "mine"


def test_a_symlinked_master_keeps_next_and_prev_beside_the_link(env):
    from fourier.cli.build import _check_build_target
    real = _master(env.tmp / "Disk" / "Curated")
    link = env.home / "Music" / "FourierCurated"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(real)
    assert _check_build_target(str(link)) == str(link)       # .next/.prev go beside the link
    bad = env.lib / "Curated"                                 # a link inside the library: its
    bad.symlink_to(real)                                      # .next would land in the library
    with pytest.raises(SystemExit):
        _check_build_target(str(bad))


def test_swap_in_never_deletes_a_user_file(env):
    from fourier.cli.build import _swap_in
    m = _master(env.master, user=["KICKS/fam/mine.wav"])
    nxt = _master(Path(str(m) + ".next"))
    before = _files(m)
    with pytest.raises(SystemExit):
        _swap_in(str(nxt), str(m))
    assert _files(m) == before and nxt.exists() and not Path(str(m) + ".prev").exists()
    (m / "KICKS" / "fam" / "mine.wav").unlink()
    _swap_in(str(nxt), str(m))                                # nothing of the user's: goes ahead
    assert Path(str(m) + ".prev").exists() and not nxt.exists()


def _old_and_new(env):
    """A master (k1) and a verified build in .next that drops k1 and adds k2."""
    m = _master(env.master, files=(("KICKS", "fam", "k1.wav"), ("KICKS", "fam", "k0.wav")))
    nxt = _master(Path(str(m) + ".next"), files=(("KICKS", "fam", "k0.wav"), ("KICKS", "fam", "k2.wav")))
    man = json.loads((nxt / "manifest.json").read_text())
    man["generated"] = "2000-01-03T00:00:00Z"
    (nxt / "manifest.json").write_text(json.dumps(man))
    return m, nxt, _files(m), _files(nxt)


def _stop_at(monkeypatch, name):
    """Stop the swap (as Ctrl-C does) when it's about to put `name` in place."""
    real = os.replace

    def replace(src, dst, *a, **k):
        if str(dst).endswith(name):
            raise KeyboardInterrupt
        return real(src, dst, *a, **k)
    monkeypatch.setattr(os, "replace", replace)


def test_a_swap_that_stops_is_finished_by_the_next_build(env, monkeypatch):
    from fourier.cli.build import _check_build_target, _swap_in
    m, nxt, old, new = _old_and_new(env)
    with monkeypatch.context() as mp:
        _stop_at(mp, "k2.wav")
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    assert _check_build_target(str(m)) == str(m)              # finished, never "didn't make"
    assert _files(m) == new and not nxt.exists()
    assert _files(Path(str(m) + ".prev")) == old


def test_a_swap_an_earlier_version_left_half_way_is_finished(env):
    """An earlier version replaced manifest.json first and left no record."""
    from fourier.cli.build import _check_build_target
    m, nxt, old, new = _old_and_new(env)
    import shutil
    shutil.copytree(m, Path(str(m) + ".prev"))
    os.remove(m / "manifest.json")
    os.link(nxt / "manifest.json", m / "manifest.json")      # the new manifest, the old files
    assert _check_build_target(str(m)) == str(m)
    assert _files(m) == new and not nxt.exists()


def test_a_swap_that_cant_be_finished_names_the_way_back(env, monkeypatch):
    from fourier.cli.build import _swap_in
    m, nxt, old, new = _old_and_new(env)
    with monkeypatch.context() as mp:
        _stop_at(mp, "k2.wav")
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    (nxt / "KICKS" / "fam" / "k2.wav").unlink()              # the new build lost a file
    before = _files(m)
    res = _run(env, "build", "KICKS", "--out", str(m))
    out = _flat(res.output)
    assert res.exit_code == 1 and "can't be finished" in out and ".prev" in out, out
    assert "didn't make" not in out and _files(m) == before


@pytest.mark.parametrize("prev_left", ["empty", "part", "whole"])
def test_a_swap_that_stopped_while_copying_to_prev_never_points_at_prev(env, monkeypatch, prev_left):
    """Stopped while the master was being copied to .prev, the master is unchanged and .prev
    may be unfinished. When .next has since lost a file, the advice never says to restore
    .prev: the master is unchanged, the swap is dropped (an unfinished .prev with it) and the
    build goes ahead."""
    import shutil
    from fourier.cli.build import SWAP_MARK, _swap_in
    m, nxt, old, new = _old_and_new(env)
    prev = Path(str(m) + ".prev")
    real = shutil.copytree

    def stops(src, dst, *a, **k):
        if Path(src) != m:                                  # copytree's own calls for subfolders
            return real(src, dst, *a, **k)
        if prev_left == "whole":
            real(src, dst, *a, **k)                         # the copy finished, the mark not yet moved
        else:
            os.makedirs(dst)
            if prev_left == "part":
                (Path(dst) / "KICKS").mkdir()
                shutil.copy2(Path(src) / "KICKS" / "fam" / "k1.wav", Path(dst) / "KICKS" / "k1.wav")
        raise KeyboardInterrupt
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "copytree", stops)
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    assert _files(m) == old and json.loads((nxt / SWAP_MARK).read_text())["stage"] == "snapshot"
    (nxt / "KICKS" / "fam" / "k2.wav").unlink()              # the new build lost a file
    res = _run(env, "build", "KICKS", "--out", str(m), "--no-scan")
    out = _flat(res.output)
    assert f"{m} is unchanged" in out and f"rename {prev}" not in out and "go back" not in out, out
    assert _files(m) == old                                   # the swap never touched it
    assert not (nxt / SWAP_MARK).exists()                     # dropped: nothing pending
    assert prev.exists() == (prev_left == "whole"), out
    if prev_left == "whole":
        assert _files(prev) == old


@pytest.mark.parametrize("left_out", [("loops.csv", "phrases.csv"), ("CHANGELOG.md",)])
@pytest.mark.parametrize("stage", ["snapshot", "sync"])
def test_a_swap_whose_build_doesnt_write_a_file_the_master_has_is_finished(env, monkeypatch, stage, left_out):
    """A build may leave out files the master has from an earlier build (sets = "off" writes
    no loops.csv or phrases.csv; a home with no archived build writes no CHANGELOG.md). A
    swap of it that stopped is finished: only a file the build held when its swap began
    counts as lost."""
    import shutil
    from fourier.cli.build import _swap_in, pending_swap
    m, nxt, old, new = _old_and_new(env)
    for f in left_out:
        (m / f).write_text("old " + f)
    real = shutil.copytree

    def stops(src, dst, *a, **k):
        if Path(src) != m:
            return real(src, dst, *a, **k)
        os.makedirs(dst)
        raise KeyboardInterrupt
    with monkeypatch.context() as mp:
        if stage == "sync":
            _stop_at(mp, "k2.wav")                           # Ctrl-C while the master is updated
        else:
            mp.setattr(shutil, "copytree", stops)            # ... while it's copied to .prev
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    assert pending_swap(str(m)) == stage
    res = _run(env, "build", "KICKS", "--out", str(m), "--no-scan")
    out = _flat(res.output)
    assert "finishing that first" in out and "can't be finished" not in out, out
    assert _files(m) == new and not nxt.exists()


def test_a_file_the_build_held_when_its_swap_began_counts_when_lost(env, monkeypatch):
    from fourier.cli.build import _swap_in
    m, nxt, old, new = _old_and_new(env)
    (nxt / "loops.csv").write_text("new loops")
    with monkeypatch.context() as mp:
        _stop_at(mp, "k2.wav")
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    (nxt / "loops.csv").unlink()
    res = _run(env, "build", "KICKS", "--out", str(m), "--no-scan")
    out = _flat(res.output)
    assert res.exit_code == 1 and "can't be finished" in out and "loops.csv" in out, out


@pytest.mark.parametrize("mark", ["snapshot", "sync"])
def test_a_stop_while_a_swap_mark_is_written_says_what_happened(env, monkeypatch, mark):
    """Stopped before a stage's mark is on disk, the message names the stage before it: no
    mark yet means nothing for the next build to finish, and the master is untouched until the
    sync mark is written."""
    from fourier.cli.build import SWAP_MARK, _swap_in
    from fourier.cli import build as B
    m, nxt, old, new = _old_and_new(env)
    real = os.replace

    def replace(src, dst, *a, **k):
        if str(dst).endswith(SWAP_MARK) and (mark == "snapshot" or (nxt / SWAP_MARK).exists()):
            raise KeyboardInterrupt
        return real(src, dst, *a, **k)
    said = []
    monkeypatch.setattr(B.console, "print", lambda *a, **k: said.append(" ".join(map(str, a))))
    with monkeypatch.context() as mp:
        mp.setattr(os, "replace", replace)
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    out = _flat(" ".join(said))
    assert _files(m) == old and "only partly in place" not in out, out
    assert ("finishes putting it in place" in out) == (mark == "sync"), out


def test_a_folder_of_the_users_at_prev_stops_the_swap_before_it_starts(env):
    """A .prev that isn't a Fourier build stops the swap before its mark is written, so the
    next build has no swap to finish and never removes that folder."""
    from fourier.cli.build import SWAP_MARK, _check_build_target, _swap_in
    m, nxt, old, new = _old_and_new(env)
    prev = Path(str(m) + ".prev")
    (prev / "mine").mkdir(parents=True)
    (prev / "mine" / "notes.txt").write_text("mine")
    with pytest.raises(SystemExit):
        _swap_in(str(nxt), str(m))
    assert not (nxt / SWAP_MARK).exists()
    _check_build_target(str(m))
    assert (prev / "mine" / "notes.txt").read_text() == "mine" and _files(m) == old


def test_an_earlier_versions_half_swap_never_removes_files_written_beside_the_manifest(env):
    """An earlier version's removal of .next can stop after removing a category's
    _manifest.json or CHANGELOG.md and none of the listed audio. Finishing the swap from that
    .next would remove them from the master, so it isn't taken for a half swap."""
    import shutil
    from fourier.cli.build import _check_build_target, pending_swap
    m, nxt, old, new = _old_and_new(env)
    for f in ("KICKS/_manifest.json", "CHANGELOG.md", "loops.csv", "phrases.csv"):
        (nxt / f).write_text(f)
    shutil.copytree(m, Path(str(m) + ".prev"))
    shutil.rmtree(m)
    shutil.copytree(nxt, m)                                   # the old version's sync finished
    for gone in ("KICKS/_manifest.json", "CHANGELOG.md", "loops.csv", "phrases.csv"):
        (nxt / gone).unlink()                                 # then its rmtree(.next) stopped
        assert pending_swap(str(m)) is None, gone
        before = _files(m)
        _check_build_target(str(m))
        assert _files(m) == before
        (nxt / gone).write_text(gone)
    assert pending_swap(str(m)) == "sync"                     # whole again: finished as before


def test_a_finish_that_keeps_failing_names_the_error_to_fix(env, monkeypatch):
    """A permission error while the master is put in place says to fix it: building again
    without that fails the same way, so the advice never stops at "the next build finishes it"."""
    from fourier.cli.build import _swap_in
    m, nxt, old, new = _old_and_new(env)
    real = os.replace

    def denied(src, dst, *a, **k):
        if str(dst).endswith("k2.wav"):
            raise PermissionError(13, "Permission denied", str(dst))
        return real(src, dst, *a, **k)
    monkeypatch.setattr(os, "replace", denied)
    for first in (True, False):
        if first:
            with pytest.raises(SystemExit):
                _swap_in(str(nxt), str(m))
            continue
        res = _run(env, "build", "KICKS", "--out", str(m), "--no-scan")
        out = _flat(res.output)
        assert res.exit_code == 1 and "Permission denied" in out and "Fix that first" in out, out
        assert "The next `fourier build` finishes" not in out, out
    monkeypatch.setattr(os, "replace", real)                   # fixed: the next build finishes it
    from fourier.cli.build import _check_build_target
    _check_build_target(str(m))
    assert _files(m) == new and not nxt.exists()


def test_a_whole_build_left_in_next_after_its_swap_is_never_synced_again(env):
    """.next whose manifest the master already has, but missing files (its removal stopped):
    nothing is removed from the master."""
    from fourier.cli.build import _check_build_target
    m, nxt, old, new = _old_and_new(env)
    import shutil
    shutil.rmtree(m)
    shutil.copytree(nxt, m)
    (nxt / "KICKS" / "fam" / "k2.wav").unlink()
    _check_build_target(str(m))
    assert _files(m) == new


def test_a_swap_that_stopped_while_removing_next_only_removes_it(env, monkeypatch):
    from fourier.cli.build import _check_build_target, _swap_in
    import shutil
    m, nxt, old, new = _old_and_new(env)
    with monkeypatch.context() as mp:
        mp.setattr(shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    assert nxt.exists() and _files(m) == new
    _check_build_target(str(m))
    assert not nxt.exists() and _files(m) == new


def test_render_waits_for_a_swap_that_stopped(env, monkeypatch):
    from fourier.cli.build import _swap_in
    m, nxt, old, new = _old_and_new(env)
    with monkeypatch.context() as mp:
        _stop_at(mp, "k2.wav")
        with pytest.raises(KeyboardInterrupt):
            _swap_in(str(nxt), str(m))
    for args in (["render", "m8_tracker", "--out", str(env.tmp / "R")], ["publish", "--no-verify"]):
        res = _run(env, *args)
        assert res.exit_code == 1 and "finishes that first" in _flat(res.output), res.output
    assert not (env.tmp / "R").exists() and not (env.pub / "releases").exists()


def test_next_and_prev_must_be_fourier_builds(env, tmp_path):
    from fourier.cli.build import _check_own
    d = tmp_path / "M.prev"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"name": "someone else's"}))
    with pytest.raises(SystemExit):
        _check_own(str(d))
    (d / "manifest.json").write_text(json.dumps({"fourier_manifest": 2}))
    _check_own(str(d))
    (tmp_path / "file.next").write_text("x")
    with pytest.raises(SystemExit):
        _check_own(str(tmp_path / "file.next"))


# ---------------------------------------------------------------------------
# fourier sync [--delete]
# ---------------------------------------------------------------------------
def _render(env, files=("01_KICKS/a.wav", "01_KICKS/b.wav", "02_SNARES/c.wav"), name="render"):
    r = env.tmp / name
    if r.exists():
        import shutil
        shutil.rmtree(r)
    r.mkdir()
    for f in files:
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        (r / f).write_bytes(f.encode())
    from fourier.packs.render import write_render_marker
    write_render_marker(r, "m8_tracker")                      # a finished render
    return r


@pytest.fixture
def no_rsync(monkeypatch):
    """sync in Python, as on a system without rsync."""
    import fourier.platforms as plat
    real = plat.shutil.which
    monkeypatch.setattr(plat.shutil, "which", lambda n, *a, **k: None if n == "rsync" else real(n, *a, **k))


def test_sync_delete_refuses_a_user_file_on_the_card(env, no_rsync):
    r = _render(env)
    vol = env.tmp / "CARD"
    card = vol / "Samples" / "Fourier"
    (card / "01_KICKS").mkdir(parents=True)
    (card / "01_KICKS" / "mine.wav").write_bytes(b"mine")
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--delete")
    assert res.exit_code == 1 and "01_KICKS/mine.wav" in _flat(res.output), res.output
    assert _files(card) == {"01_KICKS/mine.wav": b"mine"}


def test_sync_prunes_what_earlier_syncs_put_there_in_python(env, no_rsync):
    r = _render(env)
    vol = env.tmp / "CARD"
    card = vol / "Samples" / "Fourier"
    vol.mkdir()
    assert _run(env, "sync", "m8_tracker", str(vol), "--from", str(r)).exit_code == 0
    r = _render(env, files=("01_KICKS/a.wav", "02_SNARES/c.wav"))
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r))      # b stays on the card
    assert res.exit_code == 0 and (card / "01_KICKS" / "b.wav").exists(), res.output
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--delete")
    assert res.exit_code == 0, res.output
    assert set(_files(card)) == {"01_KICKS/a.wav", "02_SNARES/c.wav", safety.CARD_MARK}
    assert safety.synced_files("m8_tracker", card, vol) == {"01_KICKS/a.wav", "02_SNARES/c.wav"}


def test_a_record_from_another_card_is_not_trusted(env, no_rsync):
    r = _render(env)
    a, b = env.tmp / "CARD_A", env.tmp / "CARD_B"
    a.mkdir()
    assert _run(env, "sync", "m8_tracker", str(a), "--from", str(r)).exit_code == 0
    import shutil
    shutil.copytree(a / "Samples", b / "Samples", ignore=shutil.ignore_patterns(safety.CARD_MARK))
    r = _render(env, files=("01_KICKS/a.wav",))
    before = _files(b)
    res = _run(env, "sync", "m8_tracker", str(b), "--from", str(r), "--delete")
    assert res.exit_code == 1 and "01_KICKS/b.wav" in _flat(res.output), res.output
    assert _files(b) == before
    assert _run(env, "sync", "m8_tracker", str(a), "--from", str(r), "--delete").exit_code == 0


def test_a_copy_that_stops_half_way_leaves_its_files_recorded(env, no_rsync, monkeypatch):
    import fourier.platforms as plat
    r = _render(env, files=("01_KICKS/a.wav", "01_KICKS/b.wav"))
    vol = env.tmp / "CARD"
    card = vol / "Samples" / "Fourier"
    vol.mkdir()

    real = plat.copy_file_whole

    def half(src, dst, *a):                     # a.wav is copied, then the card goes
        if Path(dst).name == "b.wav":
            raise OSError("card removed")
        real(src, dst, *a)
    with monkeypatch.context() as mp:
        mp.setattr(plat, "copy_file_whole", half)
        res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r))
    assert res.exit_code == 1, res.output
    r = _render(env, files=("01_KICKS/b.wav",))               # the next version retired a.wav
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--delete")
    assert res.exit_code == 0, res.output
    assert set(_files(card)) == {"01_KICKS/b.wav", safety.CARD_MARK}


def _synced(env, files=("01_KICKS/a.wav", "01_KICKS/b.wav")):
    """A render synced once to a card: (render, volume, card folder)."""
    r = _render(env, files=files)
    vol = env.tmp / "CARD"
    vol.mkdir(exist_ok=True)
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r))
    assert res.exit_code == 0, res.output
    return r, vol, vol / "Samples" / "Fourier"


def _rerender(r, rel, data):
    import time
    from fourier.packs.render import write_render_marker
    time.sleep(1.05)                                          # a newer file, as a later render writes
    (r / rel).write_bytes(data)
    write_render_marker(r, "m8_tracker")


def _device_edit(card, rel):
    """What the device does when it saves slice points: the same WAV, a chunk longer."""
    p = card / rel
    p.write_bytes(p.read_bytes() + b"cue \x04\x00\x00\x00edit")
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 30))
    return p.read_bytes()


@pytest.mark.parametrize("marker", [None, "m8_tracker\n", "m8_tracker (unfinished)\n", "changed"])
def test_sync_refuses_a_render_that_didnt_finish(env, no_rsync, marker):
    r, vol, card = _synced(env)
    before = _files(card)
    (r / "01_KICKS" / "a.wav").write_bytes(b"")               # what a full disk leaves
    if marker is None:
        (r / ".fourier-render").unlink()
    elif marker != "changed":
        (r / ".fourier-render").write_text(marker)
    for extra in ([], ["--delete"]):
        res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), *extra)
        out = _flat(res.output)
        assert res.exit_code == 1 and "fourier render m8_tracker" in out, out
    assert _files(card) == before


def test_sync_keeps_a_file_the_device_changed(env, no_rsync):
    r, vol, card = _synced(env)
    edited = _device_edit(card, "01_KICKS/a.wav")
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r))
    out = _flat(res.output)
    assert res.exit_code == 0 and "1 file(s) changed on the card" in out and "01_KICKS/a.wav" in out, out
    assert (card / "01_KICKS" / "a.wav").read_bytes() == edited
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r))   # and on every later sync
    assert res.exit_code == 0 and (card / "01_KICKS" / "a.wav").read_bytes() == edited


def test_sync_keeps_a_file_changed_on_the_card_and_in_the_render(env, no_rsync):
    r, vol, card = _synced(env)
    edited = _device_edit(card, "01_KICKS/a.wav")
    _rerender(r, "01_KICKS/a.wav", b"new audio")
    _rerender(r, "01_KICKS/b.wav", b"new b")
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r))
    out = _flat(res.output)
    assert res.exit_code == 1 and "in the render too" in out and "--overwrite-changed" in out, out
    assert (card / "01_KICKS" / "a.wav").read_bytes() == edited
    assert (card / "01_KICKS" / "b.wav").read_bytes() == b"new b"      # unchanged on the card: copied
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--overwrite-changed")
    assert res.exit_code == 0, res.output
    assert (card / "01_KICKS" / "a.wav").read_bytes() == b"new audio"


def test_a_device_edit_after_a_stopped_copy_stays_kept_on_later_syncs(env, no_rsync, monkeypatch):
    """A copy that stops right after writing a file leaves it recorded as pending. When the
    device then changes it, every later sync keeps it as the device's edit of the render's
    file: never a conflict that suggests --overwrite-changed."""
    import fourier.platforms as plat
    r, vol, card = _synced(env)
    _rerender(r, "01_KICKS/a.wav", b"new audio")
    real = plat.copy_file_whole

    def stops_after(src, dst, *a):              # a.wav is copied, then the sync is stopped
        real(src, dst, *a)
        raise OSError("card removed")
    with monkeypatch.context() as mp:
        mp.setattr(plat, "copy_file_whole", stops_after)
        assert _run(env, "sync", "m8_tracker", str(vol), "--from", str(r)).exit_code == 1
    assert (card / "01_KICKS" / "a.wav").read_bytes() == b"new audio"
    edited = _device_edit(card, "01_KICKS/a.wav")
    for _ in range(3):
        res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r))
        out = _flat(res.output)
        assert res.exit_code == 0 and "1 file(s) changed on the card" in out, out
        assert "--overwrite-changed" not in out, out
        assert (card / "01_KICKS" / "a.wav").read_bytes() == edited


def test_sync_delete_refuses_a_new_only_folder_or_another_devices_render(env, no_rsync):
    """A --new-only folder holds a release's new files only: --delete from it would remove
    every other file on the card. A render for another device is never synced."""
    from fourier.packs.render import write_render_marker
    r, vol, card = _synced(env, files=("01_KICKS/a.wav", "01_KICKS/b.wav", "02_SNARES/c.wav"))
    before = _files(card)
    new = _render(env, files=("02_SNARES/d.wav",), name="render-new-in-v2")
    write_render_marker(new, "m8_tracker", release="v2", new_only=True)
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(new), "--delete")
    out = _flat(res.output)
    assert res.exit_code == 1 and "--new-only folder" in out and "without --delete" in out, out
    assert _files(card) == before
    other = _render(env, name="render-dt")
    write_render_marker(other, "digitakt_2")
    for extra in ([], ["--delete"]):
        res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(other), *extra)
        out = _flat(res.output)
        assert res.exit_code == 1 and "a render for digitakt_2" in out, out
    assert _files(card) == before
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(new))    # adds its files
    assert res.exit_code == 0 and set(_files(card)) == set(before) | {"02_SNARES/d.wav"}, res.output


def test_sync_delete_removes_only_unchanged_files_a_sync_wrote(env, no_rsync):
    r, vol, card = _synced(env, files=("01_KICKS/a.wav", "01_KICKS/b.wav", "02_SNARES/c.wav"))
    edited = _device_edit(card, "01_KICKS/b.wav")
    r = _render(env, files=("01_KICKS/a.wav",))               # b and c retired
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--delete")
    out = _flat(res.output)
    assert res.exit_code == 0 and "01_KICKS/b.wav" in out and "stayed on the card" in out, out
    assert (card / "01_KICKS" / "b.wav").read_bytes() == edited
    assert not (card / "02_SNARES").exists()
    assert set(_files(card)) == {"01_KICKS/a.wav", "01_KICKS/b.wav", safety.CARD_MARK}
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--delete")   # still kept
    assert res.exit_code == 0 and (card / "01_KICKS" / "b.wav").read_bytes() == edited, res.output


def test_sync_takes_a_card_file_whose_time_alone_moved(env, no_rsync):
    """A card's file system can store another time than the one copied (FAT keeps even
    seconds): the same size and md5 is the file the sync wrote."""
    r, vol, card = _synced(env)
    p = card / "01_KICKS" / "a.wav"
    os.utime(p, (p.stat().st_atime, p.stat().st_mtime + 1))
    _rerender(r, "01_KICKS/a.wav", b"newer")
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--delete")
    assert res.exit_code == 0 and "changed on the card" not in res.output, res.output
    assert p.read_bytes() == b"newer"


def test_sync_keeps_a_differing_file_no_sync_recorded_writing(env, no_rsync):
    """A card synced by a version that recorded names only: a file that differs from the
    render may be the device's edit, so it stays until --overwrite-changed."""
    r, vol, card = _synced(env)
    rec = json.loads(safety.card_record("m8_tracker").read_text())
    for e in rec["cards"].values():
        e.pop("state"), e.pop("pending")
    safety.card_record("m8_tracker").write_text(json.dumps(rec))
    edited = _device_edit(card, "01_KICKS/a.wav")
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--delete")
    out = _flat(res.output)
    assert res.exit_code == 1 and "no sync recorded writing them" in out, out
    assert (card / "01_KICKS" / "a.wav").read_bytes() == edited
    res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--overwrite-changed")
    assert res.exit_code == 0 and (card / "01_KICKS" / "a.wav").read_bytes() == b"01_KICKS/a.wav"


def test_a_card_copy_that_stops_leaves_the_file_whole(tmp_path, monkeypatch):
    import fourier.platforms as plat
    src, dst = tmp_path / "a.wav", tmp_path / "card" / "a.wav"
    src.write_bytes(b"new" * 1000)
    dst.parent.mkdir()
    dst.write_bytes(b"old")
    monkeypatch.setattr(plat.os, "replace", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        plat.copy_file_whole(src, dst)
    assert dst.read_bytes() == b"old" and sorted(p.name for p in dst.parent.iterdir()) == ["a.wav"]


@pytest.mark.parametrize("where", ["master", "library", "renders"])
def test_sync_never_writes_into_fouriers_places_or_the_library(env, where):
    r = _render(env)
    vol = {"master": _master(env.master), "library": env.lib,
           "renders": env.home / "Music" / "FourierRenders"}[where]
    vol.mkdir(parents=True, exist_ok=True)
    before = _files(vol)
    for extra in ([], ["--delete"]):
        res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), *extra)
        assert res.exit_code == 1 and "doesn't look like a card" in _flat(res.output), res.output
        # --force takes a folder that isn't a card, never one of these
        res = _run(env, "sync", "m8_tracker", str(vol), "--from", str(r), "--force", *extra)
        assert res.exit_code == 1 and "can't be, hold or sit in" in _flat(res.output), res.output
    assert _files(vol) == before and not (vol / "Samples").exists()


def test_sync_wants_a_card(env, monkeypatch):
    """A folder that isn't a mounted volume, nor a card an earlier sync wrote, is refused (the
    home folder above all); --force takes it; a card synced before is taken again."""
    import fourier.platforms as plat
    r = _render(env)
    for where in (env.home, env.tmp / "Somewhere"):
        where.mkdir(parents=True, exist_ok=True)
        res = _run(env, "sync", "m8_tracker", str(where), "--from", str(r))
        out = _flat(res.output)
        assert res.exit_code == 1 and "doesn't look like a card" in out and "--force" in out, out
        assert not (where / "Samples").exists()
    res = _run(env, "sync", "m8_tracker", str(env.tmp / "Somewhere"), "--from", str(r), "--force")
    assert res.exit_code == 0, res.output
    assert (env.tmp / "Somewhere" / "Samples" / "Fourier" / safety.CARD_MARK).exists()
    assert _run(env, "sync", "m8_tracker", str(env.tmp / "Somewhere"), "--from", str(r)).exit_code == 0
    # the home folder never counts as a volume, even as a mount point
    monkeypatch.setattr(plat, "mount_points", lambda: [env.home])
    assert not plat.is_volume_root(env.home) and not plat.is_volume_root("/")


# ---------------------------------------------------------------------------
# fourier tools import-folder --to
# ---------------------------------------------------------------------------
def test_import_folder_never_writes_into_a_release_the_master_or_a_library_root(env):
    src = env.tmp / "Downloads" / "Kit"
    src.mkdir(parents=True)
    (src / "x.txt").write_text("x")
    rel = _master(env.tmp / "Else" / "releases" / "v2")
    (rel.parent / "LATEST.txt").write_text("v2\n")
    m = _master(env.master)
    tree = lambda: {k: v for k, v in _files(env.tmp).items() if not k.startswith("lib.db")}
    for dest in (rel / "Imported", m / "Imported", env.lib):
        before = tree()
        res = _run(env, "tools", "import-folder", str(src), "--to", str(dest))
        assert res.exit_code == 2, (dest, res.output)
        assert tree() == before and not (dest / ".fourier-import").exists()


def test_swap_in_never_replaces_a_prev_that_isnt_a_build(env):
    from fourier.cli.build import _swap_in
    m = _master(env.master)
    nxt = _master(Path(str(m) + ".next"))
    prev = Path(str(m) + ".prev")
    prev.mkdir()
    (prev / "manifest.json").write_text(json.dumps({"mine": True}))
    (prev / "set.wav").write_bytes(b"mine")
    with pytest.raises(SystemExit):
        _swap_in(str(nxt), str(m))
    assert (prev / "set.wav").read_bytes() == b"mine"


@pytest.mark.parametrize("args", [["build", "--all"], ["build", "--base", "v1"]])
def test_build_never_clears_a_next_that_isnt_a_build(env, args):
    m = _master(env.master)
    nxt = Path(str(m) + ".next")
    nxt.mkdir()
    (nxt / "manifest.json").write_text(json.dumps({"mine": True}))   # a manifest, not Fourier's
    (nxt / "set.wav").write_bytes(b"mine")
    res = _run(env, *args, "--out", str(m))
    assert res.exit_code != 0
    assert "isn't a Fourier build" in _flat(res.output), res.output
    assert (nxt / "set.wav").read_bytes() == b"mine"


# ---------------------------------------------------------------------------
# where renders and releases may go
# ---------------------------------------------------------------------------
def _config(env, **output):
    """fourier.toml with these [output] folders (the master and publish root as the env's)."""
    from fourier import places
    out = dict(master=str(env.master), publish=str(env.pub), **output)
    (env.tmp / "fourier.toml").write_text(f'preset = "balanced"\nlibrary = ["{env.lib}"]\n[output]\n'
                                         + "".join(f'{k} = "{v}"\n' for k, v in out.items()))
    places.reset()


def test_the_usual_layouts_are_fine(env):
    from fourier.safety import check_output_place
    _master(env.master)
    for out in (env.home / "Music" / "FourierRenders" / "m8_tracker", env.tmp / "R"):
        check_output_place(out, "the render folder", "renders")
    check_output_place(env.pub, "the publish root", "publish")
    one = env.tmp / "Fourier"                      # everything in one folder
    _config(env, renders=one / "renders")
    (env.tmp / "fourier.toml").write_text(
        f'library = ["{env.lib}"]\n[output]\nmaster = "{one}/master"\nrenders = "{one}/renders"\n'
        f'publish = "{one}"\n')
    from fourier import places
    places.reset()
    check_output_place(one / "renders" / "m8_tracker", "the render folder", "renders")
    check_output_place(one, "the publish root", "publish")


@pytest.mark.parametrize("where", ["master", "next", "prev", "home", "releases", "library"])
def test_render_refuses_a_folder_of_fouriers_own_or_in_the_library(env, where):
    m = _master(env.master)
    out = {"master": m / "Renders", "next": Path(str(m) + ".next") / "R",
           "prev": Path(str(m) + ".prev"), "home": env.tmp / "fourier-home" / "renders",
           "releases": env.pub / "releases" / "R", "library": env.lib / "Renders"}[where]
    before = _files(env.tmp)
    res = _run(env, "render", "m8_tracker", "--out", str(out))
    text = _flat(res.output)
    assert res.exit_code == 1 and ("can't be, hold or sit in" in text or "inside a library" in text), text
    assert not out.exists() and {k: v for k, v in _files(env.tmp).items() if "lib.db" not in k} == \
        {k: v for k, v in before.items() if "lib.db" not in k}


def test_renders_set_inside_the_library_are_skipped_by_the_scan_so_allowed(env):
    from fourier.safety import UnsafePath, check_output_place
    _config(env, renders=env.lib / "_Renders")
    check_output_place(env.lib / "_Renders" / "m8_tracker", "the render folder", "renders")
    with pytest.raises(UnsafePath, match="inside a library folder"):
        check_output_place(env.lib / "Elsewhere", "the render folder", "renders")
    with pytest.raises(UnsafePath, match="can't be or hold a library folder"):
        check_output_place(env.lib, "the render folder", "renders")


@pytest.mark.parametrize("where", ["master", "in-master", "prev", "home", "library"])
def test_publish_refuses_a_root_that_would_copy_the_master_into_itself(env, where):
    m = _master(env.master)
    to = {"master": m, "in-master": m / "Pub", "prev": Path(str(m) + ".prev"),
          "home": env.tmp / "fourier-home", "library": env.lib / "Pub"}[where]
    before = _files(m)
    res = _run(env, "publish", "--from", str(m), "--to", str(to), "--no-verify")
    text = _flat(res.output)
    assert res.exit_code == 1 and ("can't be, hold or sit in" in text or "inside a library" in text), text
    assert _files(m) == before and not (to / "releases").exists()


def test_a_publish_root_set_inside_the_library_is_allowed(env):
    _master(env.master)
    _config(env)
    (env.tmp / "fourier.toml").write_text((env.tmp / "fourier.toml").read_text().replace(
        f'publish = "{env.pub}"', f'publish = "{env.lib}/_Releases"'))
    from fourier import places
    places.reset()
    res = _run(env, "publish", "--no-verify")
    assert res.exit_code == 0 and (env.lib / "_Releases" / "releases" / "v1").is_dir(), res.output


# ---------------------------------------------------------------------------
# a publish that stopped
# ---------------------------------------------------------------------------
def test_a_publish_that_stops_leaves_nothing_in_the_way(env, monkeypatch):
    import fourier.platforms as plat
    m = _master(env.master)
    real = plat.copy_tree
    with monkeypatch.context() as mp:
        def stop(src, dst, **kw):
            real(src, dst, **kw)
            raise KeyboardInterrupt
        mp.setattr(plat, "copy_tree", stop)
        _run(env, "publish", "--from", str(m), "--no-verify")
    assert not (env.pub / "releases" / "v1.partial").exists()
    res = _run(env, "publish", "--from", str(m), "--no-verify")
    assert res.exit_code == 0 and (env.pub / "releases" / "v1").is_dir(), res.output


def test_a_publish_left_by_a_killed_run_is_removed_by_the_next(env):
    from fourier.cli.releases import PUBLISH_MARK
    m = _master(env.master)
    left = env.pub / "releases" / "v1.partial"                 # what a killed publish leaves
    (left / "KICKS").mkdir(parents=True)
    (left / PUBLISH_MARK).write_text("v1\n")
    res = _run(env, "publish", "--from", str(m), "--no-verify")
    assert res.exit_code == 0 and "left by a publish that stopped" in _flat(res.output), res.output
    assert (env.pub / "releases" / "v1").is_dir() and not left.exists()
    assert not (env.pub / "releases" / "v1" / PUBLISH_MARK).exists()


def test_leftovers_publish_cant_prove_its_own_name_the_fix(env):
    m = _master(env.master)
    rel = env.pub / "releases"
    (rel / "v1.partial" / "KICKS").mkdir(parents=True)            # an earlier version's
    res = _run(env, "publish", "--from", str(m), "--no-verify")
    text = _flat(res.output)
    assert res.exit_code == 1 and f"delete {rel / 'v1.partial'}" in text, text
    assert (rel / "v1.partial" / "KICKS").is_dir()               # never removed for the user
    import shutil
    shutil.rmtree(rel / "v1.partial")
    back = _master(rel / "v3.partial")                         # a release a sync client renamed back
    man = json.loads((back / "manifest.json").read_text())
    (back / "manifest.json").write_text(json.dumps(dict(man, release="v3")))
    res = _run(env, "publish", "--from", str(m), "--no-verify")
    assert res.exit_code == 1 and "rename it to v3" in _flat(res.output), res.output
    assert (back / "KICKS").is_dir()
    shutil.rmtree(back)
    (rel / "v2.replaced").mkdir()
    res = _run(env, "releases")
    assert "rename it back to v2" in _flat(res.output), res.output
    (rel / "v2").mkdir()
    res = _run(env, "releases")
    assert f"v2 is in place, so delete {rel / 'v2.replaced'}" in _flat(res.output), res.output


def _publish_force_stopped_between_renames(env, monkeypatch, m):
    """v1 published, then `publish --force --as-version 1` stopped right after it moved the old
    v1 aside (the kill window between its two renames)."""
    assert _run(env, "publish", "--from", str(m), "--no-verify").exit_code == 0
    real = os.rename

    def stop(a, b):
        real(a, b)
        if str(a).endswith(os.sep + "v1") and str(b).endswith("v1.replaced"):
            raise KeyboardInterrupt
    with monkeypatch.context() as mp:
        mp.setattr(os, "rename", stop)
        res = _run(env, "publish", "--from", str(m), "--no-verify", "--force", "--as-version", "1")
    assert res.exit_code == 130 and "`fourier releases` says which is which" in _flat(res.output), \
        res.output


@pytest.mark.parametrize("cloud", [False, True])
def test_a_publish_force_stopped_between_renames_names_one_way_out(env, monkeypatch, cloud):
    """Killed between moving the old vN aside and moving the new one in, publish --force
    leaves vN.partial (the new release) and vN.replaced (the old one). The advice says which
    is which and to keep one; in a cloud-synced publish root both sit in the staging folder
    and are still listed, and the next publish never removes them."""
    import shutil
    if cloud:
        pub = env.tmp / "Dropbox" / "Pub"
        st = env.tmp / "staging"
        monkeypatch.setenv("FOURIER_STAGING_DIR", str(st))
        cfg = env.tmp / "fourier.toml"
        cfg.write_text(cfg.read_text().replace(f'publish = "{env.pub}"', f'publish = "{pub}"'))
        from fourier import places
        places.reset()
    else:
        pub = env.pub
        st = pub / "releases"
    m = _master(env.master)
    _publish_force_stopped_between_renames(env, monkeypatch, m)
    rel = pub / "releases"
    assert not (rel / "v1").exists() and (st / "v1.partial").is_dir() and (st / "v1.replaced").is_dir()
    res = _run(env, "releases")
    text = _flat(res.output)
    assert f"{st / 'v1.partial'} is the new v1 and {st / 'v1.replaced'} the v1 it replaced" in text, text
    assert "Keep one" in text and "rename it back" not in text and "renamed back by a sync" not in text, text
    res = _run(env, "publish", "--from", str(m), "--no-verify")
    assert res.exit_code == 1 and "needs fixing first" in _flat(res.output), res.output
    assert (st / "v1.partial").is_dir() and (st / "v1.replaced").is_dir()      # never removed
    os.rename(st / "v1.partial", rel / "v1")                                 # the advice works
    shutil.rmtree(st / "v1.replaced")
    res = _run(env, "publish", "--from", str(m), "--no-verify")
    assert res.exit_code == 0 and (rel / "v2").is_dir(), res.output


def test_a_publish_out_of_room_says_how_much_it_needs(env, monkeypatch):
    import fourier.platforms as plat
    m = _master(env.master)
    monkeypatch.setattr(plat, "copy_tree", lambda *a, **k: (1, ["[Errno 28] No space left on device"]))
    res = _run(env, "publish", "--from", str(m), "--no-verify")
    text = _flat(res.output)
    assert res.exit_code == 1 and "No space left" in text and "A release needs about" in text, text
    assert not (env.pub / "releases" / "v1.partial").exists()


# ---------------------------------------------------------------------------
# db-stats --missing --prune
# ---------------------------------------------------------------------------
def _db(env, n_here, n_gone, where=None):
    from fourier.db import session as sess
    from fourier.db.models import Sample
    sess._engine = sess._SessionLocal = None
    sess.init_db(env.db)
    s = sess.get_session()
    where = where or env.lib / "Acme"
    for i in range(n_here):
        p = where / f"Kick {i:02}.wav"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF")
        s.add(Sample(path=str(p), filename=p.name))
    for i in range(n_gone):
        s.add(Sample(path=str(where / "Gone" / f"Snare {i:02}.wav"), filename=f"Snare {i:02}.wav"))
    s.commit()
    s.close()
    sess._engine = sess._SessionLocal = None


def _count(env):
    from sqlalchemy import text
    from fourier.db import session as sess
    sess._engine = sess._SessionLocal = None
    sess.init_db(env.db)
    s = sess.get_session()
    try:
        return s.execute(text("SELECT COUNT(*) FROM samples")).scalar()
    finally:
        s.close()
        sess._engine = sess._SessionLocal = None


def test_prune_refuses_while_a_library_folder_is_missing(env, monkeypatch):
    _db(env, 5, 0)
    import shutil
    shutil.move(str(env.lib), str(env.tmp / "unplugged"))       # the drive is gone
    res = _run(env, "tools", "db-stats", "--missing", "--prune")
    text = _flat(res.output)
    assert res.exit_code == 1 and "is missing" in text and "Nothing was removed" in text, text
    assert _count(env) == 5
    res = _run(env, "tools", "db-stats", "--missing", "--prune", "--force")
    assert res.exit_code == 1 and _count(env) == 5                # --force never prunes an absent drive


def test_prune_asks_before_removing_much_of_the_library(env):
    _db(env, 20, 12)
    res = _run(env, "tools", "db-stats", "--missing", "--prune")
    text = _flat(res.output)
    assert res.exit_code == 1 and "12 of 32 samples" in text and "--force" in text, text
    assert _count(env) == 32
    res = _run(env, "tools", "db-stats", "--missing", "--prune", "--force")
    assert res.exit_code == 0 and _count(env) == 20, res.output
    _db(env, 0, 2)                                              # a few: removed as before
    res = _run(env, "tools", "db-stats", "--missing", "--prune")
    assert res.exit_code == 0 and "Removed 2 samples" in res.output, res.output
