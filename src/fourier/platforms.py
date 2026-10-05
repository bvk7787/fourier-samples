"""What differs between macOS, Linux and Windows, in one place.

- cloud_synced(path): whether a folder is kept in sync by a cloud drive (iCloud Drive, a
  macOS File Provider folder such as Dropbox, OneDrive or Google Drive, or those apps' own
  folders). A release is staged outside it and moved in with one rename (publish).
- cloud_only(path) / materialize(paths, log): whether a file is a cloud drive's placeholder
  whose content isn't on this machine (iCloud's "Optimize Mac Storage", OneDrive Files
  On-Demand, Dropbox online-only), and fetching such files by reading them through. A read
  that starts the download can fail or see an unreadable file, so a build fetches its
  picks first instead of letting a read do it.
- mount_points(): where removable cards show up (/Volumes, /media/<user>, /run/media/<user>,
  drive letters).
- sync_tree(...) / copy_tree(...): rsync when it's installed (as before), else the same
  copy in Python, so publish and card sync work where rsync doesn't exist (Windows).
- dir_mtimes_reliable(path): whether the file system a folder is on updates a folder's
  modification time whenever an entry in it is added, removed or renamed (the library walk
  reuses an unchanged folder's listing only there).
- open_path(path, text): a folder in the file browser (Finder), or a text file in the
  system's text editor (`fourier open`, `fourier config edit`).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# path fragments of folders a cloud drive syncs: (fragment, provider)
CLOUD_MARKERS = (
    ("/Mobile Documents/", "iCloud Drive"),
    ("/Library/CloudStorage/", "a cloud drive (File Provider)"),
    ("/Dropbox/", "Dropbox"),
    ("/OneDrive", "OneDrive"),
    ("/Google Drive/", "Google Drive"),
    ("/My Drive/", "Google Drive"),
)


def cloud_synced(path) -> str | None:
    """The cloud drive that syncs this folder, or None."""
    p = str(path).replace("\\", "/")
    p = p if p.endswith("/") else p + "/"
    for frag, name in CLOUD_MARKERS:
        if frag in p:
            return name
    return None


# macOS: the file's content lives only in the cloud (sys/stat.h SF_DATALESS)
SF_DATALESS = 0x40000000
# Windows: a placeholder the cloud provider fills on access (winnt.h)
FILE_ATTRIBUTE_OFFLINE = 0x00001000
FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
_WIN_CLOUD_ONLY = (FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS | FILE_ATTRIBUTE_RECALL_ON_OPEN
                   | FILE_ATTRIBUTE_OFFLINE)


def stat_cloud_only(st, system: str | None = None) -> bool:
    """Whether a stat result (os.lstat, DirEntry.stat(follow_symlinks=False)) is a cloud-only
    placeholder: SF_DATALESS on macOS, a recall-on-access or offline attribute on Windows;
    never elsewhere."""
    system = system or sys.platform
    if system == "darwin":
        return bool(getattr(st, "st_flags", 0) & SF_DATALESS)
    if system.startswith("win"):
        return bool(getattr(st, "st_file_attributes", 0) & _WIN_CLOUD_ONLY)
    return False


def cloud_only(path) -> bool:
    """Whether the file at path is a cloud drive's placeholder (its content not on this
    machine). One lstat; False when the file can't be looked up."""
    try:
        return stat_cloud_only(os.stat(path, follow_symlinks=False))
    except (OSError, ValueError):
        return False


def _read_through(path, chunk: int = 1 << 20) -> None:
    """Read a whole file and drop the bytes: a cloud drive downloads a placeholder's
    content when it's read."""
    with open(path, "rb") as f:
        while f.read(chunk):
            pass


def materialize(paths, log=None) -> list:
    """Download the cloud-only files among paths by reading each through, then look again.
    Returns the paths that are still cloud-only or couldn't be read (in the order given)."""
    log = log or (lambda m: None)
    todo = [p for p in paths if cloud_only(p)]
    left = []
    for n, p in enumerate(todo, 1):
        try:
            _read_through(p)
        except OSError as e:
            log(f"can't read {p}: {e}")
            left.append(p)
            continue
        if cloud_only(p):
            left.append(p)
        if n % 500 == 0 and n < len(todo):
            log(f"  {n:,} of {len(todo):,} downloaded")
    return left


def mount_points() -> list[Path]:
    """Folders where removable volumes (SD cards, USB drives) appear on this system."""
    if sys.platform == "darwin":
        roots = [Path("/Volumes")]
    elif sys.platform.startswith("win"):
        return [Path(f"{c}:\\") for c in "DEFGHIJKLMNOPQRSTUVWXYZ" if os.path.exists(f"{c}:\\")]
    else:
        user = os.environ.get("USER") or ""
        roots = [Path("/media") / user, Path("/run/media") / user, Path("/media"), Path("/mnt")]
    out = []
    for r in roots:
        try:
            out += sorted(p for p in r.iterdir() if p.is_dir())
        except OSError:
            continue
    return list(dict.fromkeys(out))


def is_volume_root(path) -> bool:
    """Whether a folder is where a card or drive is mounted: one of mount_points(), or a mount
    point of its own; never the system's root or a folder that holds the home folder."""
    try:
        p = Path(os.path.realpath(os.path.expanduser(str(path))))
        home = Path(os.path.realpath(Path.home()))
    except (OSError, RuntimeError):
        return False
    if p == Path(p.anchor) or p == home or p in home.parents:
        return False
    if p in {Path(os.path.realpath(m)) for m in mount_points()}:
        return True
    try:
        return os.path.ismount(p)
    except OSError:
        return False


def _files(root: Path, skip) -> dict[str, Path]:
    out = {}
    for dp, dns, fns in os.walk(root):
        rel_dir = os.path.relpath(dp, root)
        dns[:] = sorted(d for d in dns if not skip(os.path.normpath(os.path.join(rel_dir, d))))
        for f in sorted(fns):
            rel = os.path.normpath(os.path.join(rel_dir, f))
            if not skip(rel):
                out[rel.replace(os.sep, "/")] = Path(dp) / f
    return out


def sync_tree(src, dst, delete=False, skip_dotfiles=True, dry_run=False,
              use_rsync: bool | None = None) -> tuple[int, list[str]]:
    """Copy src's files into dst the way `rsync -rt [--delete] --exclude=.*` does: a file
    is copied when it's new or its size or time differ, times are kept, extended
    attributes are not (so macOS writes no "._" twins onto a FAT/exFAT card); with delete,
    files and folders dst has and src doesn't go. Returns (return code, changes)."""
    src, dst = str(src).rstrip("/\\"), str(dst).rstrip("/\\")
    rsync = shutil.which("rsync") if use_rsync is not False else None
    if rsync:
        args = [rsync, "-rt"] + (["--exclude=.*"] if skip_dotfiles else []) \
            + (["--delete"] if delete else []) + (["--dry-run", "--itemize-changes"] if dry_run else []) \
            + [src + "/", dst + "/"]
        rr = subprocess.run(args, capture_output=True, text=True)
        return rr.returncode, (rr.stdout if rr.returncode == 0 else rr.stderr).splitlines()
    skip = (lambda rel: any(part.startswith(".") for part in Path(rel).parts)) if skip_dotfiles \
        else (lambda rel: False)
    have = _files(Path(src), skip)
    there = _files(Path(dst), skip) if os.path.isdir(dst) else {}
    changes = []
    for rel, p in have.items():
        q = Path(dst) / rel
        st = p.stat()
        old = there.get(rel)
        if old is not None:
            o = old.stat()
            if o.st_size == st.st_size and int(o.st_mtime) == int(st.st_mtime):
                continue
        changes.append(f">f {rel}")
        if not dry_run:
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, q)
            os.utime(q, (st.st_atime, st.st_mtime))
    if delete:
        for rel in sorted(set(there) - set(have), reverse=True):
            changes.append(f"*deleting {rel}")
            if not dry_run:
                (Path(dst) / rel).unlink()
        if not dry_run and os.path.isdir(dst):
            for dp, dns, fns in os.walk(dst, topdown=False):
                rel = os.path.relpath(dp, dst)
                if rel != "." and not skip(rel) and not os.listdir(dp) \
                        and not os.path.isdir(os.path.join(src, rel)):
                    os.rmdir(dp)
    return 0, changes


def copy_tree(src, dst, exclude_top=(), dry_run=False, use_rsync: bool | None = None) -> tuple[int, list[str]]:
    """Copy src to dst whole, like `rsync -a --exclude=/<name>`: permissions and times kept,
    the top-level names in exclude_top left out. Returns (return code, messages)."""
    src, dst = str(src).rstrip("/\\"), str(dst).rstrip("/\\")
    rsync = shutil.which("rsync") if use_rsync is not False else None
    if rsync:
        args = [rsync, "-a"] + (["--dry-run"] if dry_run else []) \
            + [f"--exclude=/{n}" for n in exclude_top] + [src + "/", dst + "/"]
        rr = subprocess.run(args, capture_output=True, text=True)
        return rr.returncode, (rr.stdout if rr.returncode == 0 else rr.stderr).splitlines()
    if dry_run:
        return 0, []
    top = os.path.normpath(src)
    ignore = lambda d, names: [n for n in names if os.path.normpath(d) == top and n in exclude_top]
    try:
        shutil.copytree(src, dst, symlinks=True, ignore=ignore, dirs_exist_ok=True)
    except (OSError, shutil.Error) as e:
        return 1, [str(e)]
    return 0, []


# file systems that update a folder's modification time on every entry added, removed or
# renamed in it. FAT and exFAT (most cards and many external drives) don't reliably, and a
# network or FUSE file system may cache it, so neither is trusted.
RELIABLE_DIR_MTIME_FS = frozenset({"apfs", "hfs", "ext2", "ext3", "ext4", "xfs", "btrfs", "zfs",
                                   "f2fs", "jfs", "reiserfs", "tmpfs", "overlay"})


def _mounts() -> list[tuple[str, str]]:
    """(mount point, file system type) for this system, longest mount point first; [] when
    it can't be read (Windows: never read)."""
    out = []
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/mounts") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        out.append((parts[1].replace("\\040", " "), parts[2].lower()))
        except OSError:
            return []
    elif sys.platform == "darwin":
        try:
            rr = subprocess.run(["/sbin/mount"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return []
        for line in rr.stdout.splitlines():     # "/dev/disk3s5 on /System/Volumes/Data (apfs, local, ...)"
            if " on " in line and " (" in line:
                point = line.split(" on ", 1)[1].rsplit(" (", 1)[0]
                fs = line.rsplit(" (", 1)[1].split(",")[0].strip(" )").lower()
                out.append((point, fs))
    return sorted(out, key=lambda m: len(m[0]), reverse=True)


_MOUNTS: list | None = None


def fs_type(path) -> str | None:
    """The file system type of the mount a folder is on ("apfs", "ext4", "exfat", ...), or
    None when unknown."""
    global _MOUNTS
    if _MOUNTS is None:
        _MOUNTS = _mounts()
    try:
        p = os.path.realpath(str(path))
    except (OSError, ValueError):
        return None
    for point, fs in _MOUNTS:
        if p == point or p.startswith(point.rstrip("/") + "/"):
            return fs
    return None


def dir_mtimes_reliable(path) -> bool:
    """Whether a folder's modification time changes whenever an entry in it is added,
    removed or renamed, on the file system it's on (RELIABLE_DIR_MTIME_FS)."""
    return fs_type(path) in RELIABLE_DIR_MTIME_FS


def file_browser() -> str:
    """What this system calls its file browser, for messages."""
    if sys.platform == "darwin":
        return "Finder"
    if sys.platform.startswith("win"):
        return "File Explorer"
    return "your file manager"


def open_path(path, text: bool = False) -> list[str] | None:
    """Show a folder in the file browser, or open a text file in the system's text editor
    (macOS: Finder, or TextEdit with `open -t`; Linux: xdg-open; Windows: the associated
    app). Returns the command run, or None when this system has no way to open it."""
    p = str(path)
    if sys.platform == "darwin":
        cmd = ["open", "-t", p] if text else ["open", p]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return cmd
    if sys.platform.startswith("win"):
        os.startfile(p)                                     # type: ignore[attr-defined]
        return ["start", p]
    if not shutil.which("xdg-open"):
        return None
    cmd = ["xdg-open", p]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)
    return cmd
