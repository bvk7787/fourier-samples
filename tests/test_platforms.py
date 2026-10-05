"""What differs between systems (fourier/platforms.py)."""
import os
import shutil
import time

import pytest

from fourier import platforms as P


def test_cloud_synced_folders():
    assert P.cloud_synced("/Users/x/Library/Mobile Documents/com~apple~CloudDocs/Samples/releases") == "iCloud Drive"
    assert P.cloud_synced("/Users/x/Library/CloudStorage/Dropbox/Fourier") is not None
    assert P.cloud_synced("/home/x/Dropbox/Fourier/releases") == "Dropbox"
    assert P.cloud_synced("C:\\Users\\x\\OneDrive\\Music\\Fourier") == "OneDrive"
    assert P.cloud_synced("/home/x/Music/Fourier/releases") is None


def _tree(root, files):
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _listing(root):
    return sorted(str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*") if p.is_file())


def test_sync_tree_without_rsync(tmp_path):
    src, dst = tmp_path / "render", tmp_path / "card"
    _tree(src, {"01_KICKS/deep/BD.wav": "a", "02_SNARES/x/SD.wav": "b", ".fourier-render": "m8",
                "01_KICKS/.DS_Store": "x"})
    _tree(dst, {"01_KICKS/deep/BD.wav": "old!", "09_GONE/old/y.wav": "z", "._BD.wav": "junk"})
    code, changes = P.sync_tree(src, dst, dry_run=True, use_rsync=False)
    assert code == 0 and any("SD.wav" in c for c in changes)
    assert _listing(dst) == ["._BD.wav", "01_KICKS/deep/BD.wav", "09_GONE/old/y.wav"]   # dry run
    P.sync_tree(src, dst, use_rsync=False)
    assert (dst / "01_KICKS/deep/BD.wav").read_text() == "a"
    assert _listing(dst) == ["._BD.wav", "01_KICKS/deep/BD.wav", "02_SNARES/x/SD.wav", "09_GONE/old/y.wav"]
    assert int((dst / "02_SNARES/x/SD.wav").stat().st_mtime) == int((src / "02_SNARES/x/SD.wav").stat().st_mtime)
    code, changes = P.sync_tree(src, dst, use_rsync=False)
    assert changes == []                                   # nothing new: nothing copied
    P.sync_tree(src, dst, delete=True, use_rsync=False)
    assert _listing(dst) == ["._BD.wav", "01_KICKS/deep/BD.wav", "02_SNARES/x/SD.wav"]  # dotfiles left alone
    assert not (dst / "09_GONE").exists()


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync")
def test_sync_tree_matches_rsync(tmp_path):
    files = {"a/1.wav": "1", "a/b/2.wav": "22", ".hidden": "h", "c/.x/3.wav": "3"}
    for use in (True, False):
        src, dst = tmp_path / f"s{use}", tmp_path / f"d{use}"
        _tree(src, files)
        _tree(dst, {"old/9.wav": "9", "a/1.wav": "stale"})
        time.sleep(0.01)
        P.sync_tree(src, dst, delete=True, use_rsync=use)
    assert _listing(tmp_path / "dTrue") == _listing(tmp_path / "dFalse")


def test_copy_tree_without_rsync(tmp_path):
    src, dst = tmp_path / "master", tmp_path / "v1.partial"
    _tree(src, {"KICKS/deep/BD.wav": "a", "_REVIEW/queue.wav": "q", "manifest.json": "{}"})
    code, _ = P.copy_tree(src, dst, exclude_top=("_REVIEW",), use_rsync=False)
    assert code == 0 and _listing(dst) == ["KICKS/deep/BD.wav", "manifest.json"]


def test_mount_points_are_folders():
    assert all(p.is_dir() for p in P.mount_points())


@pytest.mark.parametrize("mounted, card, inside", [
    ("/Volumes/M8", "/Volumes/M8", "/Volumes/M8/Samples"),               # macOS
    ("/media/x/M8", "/media/x/M8", "/media/x/M8/Samples"),               # Linux (udisks)
    ("/run/media/x/M8", "/run/media/x/M8", "/run/media/x/M8/Samples"),   # Linux (Fedora)
])
def test_a_volume_root_is_a_mounted_card_never_home_or_root(monkeypatch, mounted, card, inside):
    """sync's test for a card: where mount_points() says a volume is, or a mount point of its
    own; never the system root, the home folder or a folder holding it."""
    from pathlib import Path
    monkeypatch.setenv("HOME", "/home/x" if mounted.startswith(("/media", "/run")) else "/Users/x")
    monkeypatch.setattr(P, "mount_points", lambda: [Path(mounted)])
    monkeypatch.setattr(P.os.path, "ismount", lambda p: str(p) in ("/", mounted))
    assert P.is_volume_root(card)
    assert not P.is_volume_root(inside)
    home = Path(P.os.environ["HOME"])
    for p in ("/", home, home.parent, home / "Music"):
        assert not P.is_volume_root(p), p
