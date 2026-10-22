"""fourier render, sync, publish and releases, and fourier tools resolve: releases and devices."""
from __future__ import annotations


import click

from ..db.session import session_scope
from ..devices.loader import DeviceLoader
from ._app import _table_cols, console, log, main, tools  # noqa: F401
from ..places import master_dir as _master_dir, publish_root as _publish_root, renders_dir as _renders_dir
from .build import _run_verify

# ---------------------------------------------------------------------------
# fourier render: the master or a release, converted for one device
# ---------------------------------------------------------------------------
@main.command("render", short_help="Render the master or a release for a device.")
@click.argument("device_id")
@click.option("--from", "master_dir", default=None,
              help="Master dir to render from (default: the master)")
@click.option("--release", default=None,
              help="Render an immutable release (e.g. v1) instead of the master, and lock its paths")
@click.option("--no-lock", is_flag=True, default=False,
              help="With --release: render without recording new paths in the device lock")
@click.option("--out", "out_dir", default=None,
              help="Output dir (default: [output] renders, else FourierRenders beside the master)")
@click.option("--max-mb", default=None, type=float,
              help="Cap the rendered image size in MB (default: full)")
@click.option("--check", is_flag=True, default=False,
              help="Only check the on-card paths against the device's path-length limit")
@click.option("--card-dir", default=None,
              help="With --check: the folder the render is copied to (default: paths.card_dir)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Only preview what the render would change on the device (its path lock); "
                   "exits non-zero when a path on the device would lose its audio")
@click.option("--new-only", is_flag=True, default=False,
              help="With --release: also put the files this release adds to the device in a "
                   "folder of their own (<render>-new-in-vN), to copy across on their own")
def render(device_id, master_dir, release, no_lock, out_dir, max_mb, check, card_dir, dry_run,
           new_only=False):
    """Render the master (or a release) for a device: its sample rate, bit depth, channels,
    folder depth and path limits. Paths already on the device (its path lock, written when
    you render a release) never change: locked files keep their names and original audio,
    and retired ones stay in the image.

    \b
      fourier render m8_tracker                   # preview from the master
      fourier render m8_tracker --release v1      # render v1 for transfer; locks paths
      fourier render m8_tracker --dry-run         # what it would change on the device
      fourier render m8_tracker --check           # on-card paths vs the path-length limit
      fourier render digitakt_2 --release v2 --new-only   # and v2's new files on their own
    """
    import os as _os
    from ..packs.releases import RELEASES_ROOT

    if new_only and (not release or no_lock):
        raise click.UsageError("--new-only goes with --release (and the path lock it records)")
    if check or dry_run:            # reads only
        from ._app import read_only_config
        read_only_config()
    if release:
        master_dir = _os.path.join(RELEASES_ROOT, release)
    if check:
        _check_paths(device_id, master_dir, card_dir)
        return
    if dry_run:
        _device_preview(device_id, master_dir)
        return
    from ..packs.render import render_device
    from ..safety import UnsafePath, check_output_place
    master_dir = master_dir or _master_dir()
    if not release:
        _stop_on_pending_swap(master_dir)
    out_dir = out_dir or _renders_dir(device_id)
    try:
        check_output_place(out_dir, "the render folder", "renders", masters=(master_dir,))
        if new_only:
            check_output_place(out_dir.rstrip("/") + f"-new-in-{release}", "the --new-only folder",
                               "renders", masters=(master_dir,))
    except UnsafePath as e:
        console.print(str(e), style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1) from None
    from ._app import stop_note
    stop_note(f"The previous render in {out_dir} is as it was: `fourier render {device_id}"
              + (f" --release {release}" if release else "") + "` renders again.")
    try:
        # a render's lines name paths: never hard-wrapped (soft_wrap; the terminal folds them)
        r = render_device(device_id, master_dir, out_dir, max_mb=max_mb,
                          log=lambda m: console.print(m, markup=False, highlight=False, soft_wrap=True),
                          release=release, write_lock=bool(release) and not no_lock)
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)
    if r.get("failed"):
        console.print(f"[red]{r['failed']} files failed to convert (the FAILED lines above say why), "
                      f"so the render wasn't replaced[/red]"
                      + (":" if r.get("no_room") else ": fix what they name, then render again."))
        if r.get("no_room"):
            console.print(r["no_room"], style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    if r.get("missing"):
        console.print(f"[red]{r['missing']} locked files missing: this image is incomplete[/red] "
                      f"(a sync leaves them on the card)")
        raise SystemExit(1)
    if new_only:
        _new_only(device_id, release, out_dir, r["out"])


def _stop_on_pending_swap(master_dir) -> None:
    """Stop when the last build didn't finish putting the master in place (it holds files of
    the old build and the new one): the next build finishes that."""
    from .build import pending_swap
    if pending_swap(master_dir):
        console.print(f"The last build stopped while it put the new master in place at "
                      f"{master_dir}: run `fourier build` (it finishes that first), then try again.",
                      style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)


def _new_only(device_id, release, out_dir, base) -> None:
    """The files `release` added to the device's lock, copied out of its render into
    <render>-new-in-<release> (same folders), for a loader that can't sync (Transfer). Written
    beside it and moved in when finished, like a render, so a copy that stops leaves the
    previous one as it was."""
    import os as _os
    import shutil
    from pathlib import Path

    from ..packs.device_lock import load_lock
    from ..packs.render import RENDER_MARKER, put_render_in_place, start_render_folder, write_render_marker
    lock = load_lock(device_id)
    new = sorted(v["path"] for v in (lock.files.values() if lock else ()) if v.get("release") == release)
    out = Path(out_dir.rstrip("/") + f"-new-in-{release}")
    if out.exists() and not (out / RENDER_MARKER).is_file():
        console.print(f"{out} exists and isn't one Fourier made: left as it is.", style="red",
                      markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    try:
        stage = start_render_folder(out, f"{device_id} {release} new files")
    except RuntimeError as e:
        console.print(str(e), style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1) from None
    try:
        dest = stage / Path(base).relative_to(out_dir) if Path(base) != Path(out_dir) else stage
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for rel in new:
            src = Path(base) / rel
            if src.is_file():
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / rel)
                n += 1
        mb = sum(_os.path.getsize(p) for p in dest.rglob("*") if p.is_file() and p.name != RENDER_MARKER) / 2**20
        write_render_marker(stage, device_id, release=release, new_only=True)
        put_render_in_place(stage, out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    console.print(f"{n:,} files {release} adds to the device ({mb:.1f} MB), in folders of their "
                  f"own: {out}. Copy them into the same folders on the device (`fourier open "
                  f"renders {device_id}-new-in-{release}` shows them).", markup=False,
                  highlight=False, soft_wrap=True)

@main.command("sync", short_help="Copy a device render onto a mounted SD card.")
@click.argument("device_id")
@click.argument("volume")
@click.option("--from", "render_dir", default=None,
              help="Render to copy (default: [output] renders, else FourierRenders beside the master, /<device_id>)")
@click.option("--delete", is_flag=True, default=False,
              help="Also remove files in the card folder (<card_dir>) that the render doesn't have "
                   "(refused if it holds a file neither this render nor an earlier sync made)")
@click.option("--overwrite-changed", is_flag=True, default=False,
              help="Replace card files that changed since the last sync (edits or slice points the "
                   "device saved in them) with the render's")
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be copied")
@click.option("--force", is_flag=True, default=False,
              help="Sync into VOLUME even when it isn't a mounted card or drive (a folder you chose)")
def sync(device_id, volume, render_dir, delete, dry_run, force, overwrite_changed=False):
    """Copy a device render onto a mounted card, such as the M8's SD card.

    VOLUME must be where the card is mounted (/Volumes/M8, /media/<you>/M8), or a card an
    earlier sync wrote to; --force takes any other folder. Copies into <VOLUME><card_dir>
    (M8: /Samples/Fourier) file by file, each through a temporary file, and without
    extended attributes, so macOS writes no "._" AppleDouble twins onto the FAT/exFAT card
    (they show up in the device's browser as .wav files that fail to load). Dotfiles
    (the render marker, .DS_Store) are skipped, any "._" file under the target is removed,
    and the audio file count on the card is checked against the render.

    Only a finished render is copied. A card file that changed since the sync that wrote it
    (the device saves slice points and edits in the file) is kept, and listed, unless
    --overwrite-changed; --delete removes only files a sync wrote that are unchanged.

    \b
      fourier sync m8_tracker /Volumes/M8
    """
    import os as _os
    from ..devices.loader import DeviceLoader
    from ..packs.render import _card_prefix, read_render_marker, render_problem
    from ..platforms import copy_file_whole
    def stop(msg):
        console.print(msg, style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)

    try:
        dev = DeviceLoader().load(device_id)
    except Exception as e:
        stop(str(e))
    if not render_dir:
        render_dir = _renders_dir(device_id)
    if not _os.path.exists(render_dir):
        stop(f"no render yet: run `fourier render {device_id}` (it writes {render_dir})")
    problem = render_problem(render_dir)
    if problem:
        stop(f"{problem}. Nothing was copied: run `fourier render {device_id}` and sync once it "
             f"finishes.")
    marker = read_render_marker(render_dir) or {}
    if marker.get("device") and marker["device"] != device_id:
        stop(f"{render_dir} is a render for {marker['device']}. Nothing was copied: sync it with "
             f"`fourier sync {marker['device']}`, or render {device_id} with "
             f"`fourier render {device_id}`.")
    if delete and marker.get("new_only"):
        stop(f"{render_dir} holds only the files {marker.get('release') or 'a release'} adds (a "
             f"--new-only folder), so --delete would remove every other file Fourier put on the "
             f"card. Nothing was copied: sync it without --delete, or sync the full render with "
             f"--delete (leave out --from for the default one).")
    if dev.load != "card-sync":
        stop(f"{dev.name} doesn't load from a card, so there's nothing to sync: {_load_how(dev)}. "
             f"The render is in {render_dir}.")
    if not _os.path.isdir(volume):
        stop(f"{volume} isn't mounted")
    dest = _os.path.join(volume, _card_prefix(dev).lstrip("/"))
    from ..platforms import is_volume_root, mount_points
    from ..safety import (CARD_MARK, CARD_TMP, UnsafePath, check_card_dest, plan_sync, record_card,
                          synced_files, synced_state)
    if not (force or is_volume_root(volume) or _os.path.exists(_os.path.join(dest, CARD_MARK))):
        mounted = ", ".join(str(m) for m in mount_points()) or "none found"
        stop(f"{volume} doesn't look like a card: it isn't a mounted volume (mounted now: {mounted}) "
             f"and no earlier `fourier sync` wrote to it. Give the card's mount point, or pass "
             f"--force to sync into {dest} anyway.")
    try:
        dest = str(check_card_dest(dest, volume, render_dir, delete, device_id))
    except UnsafePath as e:
        stop(str(e))
    rendered = marker["files"]
    state, pending = synced_state(device_id, dest, volume)
    plan = plan_sync(render_dir, rendered, dest, state, pending, delete=delete,
                     overwrite_changed=overwrite_changed, protected=marker.get("missing") or ())
    console.print(f"Copying {render_dir} -> {dest}{' (dry run)' if dry_run else ''} ...",
                  markup=False, highlight=False, soft_wrap=True)
    if dry_run:
        lines = [f">f {r}" for r in plan.copy] + [f"*deleting {r}" for r in plan.delete]
        console.print("\n".join(lines)[-2000:] or "nothing to copy", markup=False, highlight=False)
        _sync_report(plan, dry_run=True)
        return
    _os.makedirs(dest, exist_ok=True)
    known = synced_files(device_id, dest, volume)
    rstat = lambda rel: _os.stat(_os.path.join(render_dir, rel))
    # recorded before copying (with what earlier syncs put on this card): a copy that stops
    # half way leaves every file Fourier wrote known, and each one as it was or as rendered
    record_card(device_id, render_dir, dest, volume, known | set(rendered), state,
                pending={**pending,
                         **{r: [rendered[r][0], int(rstat(r).st_mtime), rendered[r][1]] for r in plan.copy}})
    from ._app import stop_note
    stop_note(f"Some files are on the card and some aren't yet: run the same `fourier sync "
              f"{device_id}` again to finish before you eject it.")
    try:
        for rel in plan.copy:
            copy_file_whole(_os.path.join(render_dir, rel), _os.path.join(dest, rel), CARD_TMP)
        for rel in plan.delete:
            _os.remove(_os.path.join(dest, rel))
    except OSError as e:
        stop(f"copy failed: {e}")
    for dp, dns, _fns in _os.walk(dest, topdown=False):      # folders the delete emptied
        rel = _os.path.relpath(dp, dest)
        if delete and rel != "." and not any(p.startswith(".") for p in rel.split(_os.sep)) \
                and not _os.listdir(dp) and not _os.path.isdir(_os.path.join(render_dir, rel)):
            _os.rmdir(dp)
    junk = 0
    for dp, _, fns in _os.walk(dest):
        for fn in fns:
            if fn.startswith("._") or fn == ".DS_Store" or (fn.startswith(".") and fn.endswith(CARD_TMP)):
                _os.remove(_os.path.join(dp, fn))
                junk += 1
    is_audio = lambda f: f.lower().endswith((".wav", ".aif", ".aiff")) and not f.startswith(".")
    count = lambda root: sum(1 for dp, _, fns in _os.walk(root) for f in fns if is_audio(f))
    n_src, n_dst = count(render_dir), count(dest)
    left = sum(1 for r in plan.kept_extra + plan.locked if is_audio(_os.path.basename(r)))
    msg = f"{n_dst} audio files on the card, {n_src} in the render" + (f"; removed {junk} '._' / .DS_Store files" if junk else "")
    # what the card folder now holds of Fourier's, and each file's contents as a sync left it
    files = set(rendered) | set(plan.kept_extra) | set(plan.locked) | (set() if delete else known)
    new_state = {k: v for k, v in state.items() if k in files and k not in plan.delete}
    for rel in plan.copy + plan.same:
        st = _os.stat(_os.path.join(dest, rel))
        new_state[rel] = [st.st_size, int(st.st_mtime), plan.md5[rel]]
    # a file this sync left alone keeps what an earlier copy that stopped expected to write:
    # it may be on the card (the device's edit of it is kept, never a conflict next time)
    done = set(plan.copy) | set(plan.same) | set(plan.delete)
    left_pending = {k: v for k, v in pending.items() if k in files and k not in done}
    record_card(device_id, render_dir, dest, volume, files - set(plan.delete), new_state,
                pending=left_pending)
    if n_dst < n_src or (delete and n_dst != n_src + left):
        console.print(f"[red]{msg}[/red]")
        raise SystemExit(1)
    if not (plan.kept or plan.conflicts or plan.unknown or plan.kept_extra or plan.locked):
        console.print(f"[green]{msg}[/green]. Eject the card before removing it.")
        return
    console.print(f"[green]{msg}[/green].")
    if _sync_report(plan):
        raise SystemExit(1)
    console.print("Eject the card before removing it.")


def _sync_report(plan, dry_run: bool = False) -> bool:
    """The files a sync left as they were on the card, and why; True when the card doesn't
    hold what the render has (a file changed in both, or one no sync recorded writing)."""
    def say(paths, text, style):
        if paths:
            eg = ", ".join(paths[:5]) + (" ..." if len(paths) > 5 else "")
            console.print(f"{len(paths)} file(s) {text}: {eg}", style=style, markup=False,
                          highlight=False, soft_wrap=True)
    keep = "would keep" if dry_run else "kept"
    say(plan.kept, f"changed on the card since the last sync (the device saves edits and slice "
                   f"points in the file), {keep} as they are", "yellow")
    say(plan.conflicts, f"changed on the card since the last sync, and in the render too: {keep} "
                        f"the card's. `--overwrite-changed` replaces them with the render's", "red")
    say(plan.unknown, f"on the card differ from the render, and no sync recorded writing them (an "
                      f"older version of Fourier, or your own): {keep}. If they're Fourier's, "
                      f"`--overwrite-changed` replaces them with the render's", "red")
    say(plan.kept_extra, f"the render no longer has {'would stay' if dry_run else 'stayed'} on "
                         f"the card: they changed there since a sync wrote them, or no sync "
                         f"recorded their contents (a card synced by an older version of Fourier). "
                         f"Delete them from the card yourself if you don't need them", "yellow")
    say(plan.locked, f"locked on the device but missing from this render: {keep} on the card",
        "yellow")
    return bool(plan.conflicts or plan.unknown)


def _load_how(dev) -> str:
    """How a device that isn't synced from a card gets its samples (its profile's load)."""
    if dev.load == "transfer":
        app = "Elektron Transfer" if dev.name.startswith("Elektron") else "its maker's transfer app"
        return (f"it loads with {app}, which never overwrites a file of the same name: drag the "
                f"render's category folders into it")
    return "copy the render's category folders into a project or onto the device yourself"


def need_master(master_dir) -> str:
    """master_dir (default: the master), or a plain stop when nothing is built there yet."""
    import os as _os
    master_dir = master_dir or _master_dir()
    if not _os.path.isfile(_os.path.join(master_dir, "manifest.json")):
        console.print(f"Nothing built yet at {master_dir}: `fourier build` makes the master.",
                      style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    return master_dir


def _device_preview(device_id, master_dir) -> None:
    """render --dry-run: what rendering master_dir would do to the paths already on the
    device (its path lock); exits 1 when one would lose or change its audio."""
    from ..packs.device_lock import load_lock
    from ..packs.render import device_plan
    master_dir = need_master(master_dir)
    try:
        device = DeviceLoader().load(device_id)
    except Exception as e:
        console.print(str(e), style="red", markup=False, highlight=False)
        raise SystemExit(1) from None
    r = device_plan(master_dir, device, load_lock(device_id),
                    log=lambda m: console.print(m, markup=False, highlight=False))
    if not r["safe"]:
        raise SystemExit(1)


def _check_paths(device_id, master_dir, card_dir, show=10):
    """render --check: the rendered paths against the device's path-length limit (e.g. the
    M8: under 128 characters)."""
    from ..packs.device_lock import load_lock
    from ..packs.render import _assignment, card_paths, shortened_note

    device = DeviceLoader().load(device_id)
    if card_dir is not None:
        device.card_dir = card_dir
    limit = device.max_path_length
    if not limit:
        console.print(f"{device.name} has no path-length limit in its profile.")
        return
    master_dir = need_master(master_dir)
    lock = load_lock(device_id)
    got = _assignment(master_dir, device, lock)       # what the render names them
    paths = card_paths(master_dir, device, lock, assignment=got)
    bad = sorted(((p, len(p)) for p in paths if len(p) > limit), key=lambda x: -x[1])
    longest = max((len(p) for p in paths), default=0)
    console.print(f"{device.name}: {len(paths)} files under {device.card_dir or '/'}; "
                  f"limit {limit} chars; longest {longest}; over the limit: {len(bad)}", highlight=False)
    note = shortened_note(device, got[1].shortened)
    if note:
        console.print(note, markup=False, highlight=False, soft_wrap=True)
    for path, n in bad[:show]:
        console.print(f"  {n:4d}  {path}", markup=False, highlight=False)
    if bad:
        raise SystemExit(1)


def _release_staging(reldir, create: bool = True):
    """Where a release is copied and checked before it appears. Outside a cloud-synced
    folder (iCloud Drive, Dropbox, OneDrive, ...; platforms.cloud_synced) when the releases
    live in one, on the same volume so it moves in with one rename: a folder filled and
    renamed inside a synced folder can be renamed back by the sync client, which can also leave
    empty "vN 2" copies behind. create=False only looks (a staging folder not made yet is
    reldir's)."""
    import os as _os
    from ..platforms import cloud_synced
    if not cloud_synced(reldir):
        return reldir
    from fourier.paths import home_path
    st = _os.environ.get("FOURIER_STAGING_DIR") or str(home_path("staging"))
    if create:
        _os.makedirs(st, exist_ok=True)
    try:
        return st if _os.stat(st).st_dev == _os.stat(reldir).st_dev else reldir
    except OSError:
        return reldir


PUBLISH_MARK = ".fourier-publish"     # in v<N>.partial while publish fills it


def _own_partial(path) -> bool:
    """A v<N>.partial folder this version's publish made (it marks one before copying)."""
    import os as _os
    return _os.path.isdir(path) and _os.path.isfile(_os.path.join(path, PUBLISH_MARK))


def _partial_manifest(path) -> dict | None:
    """The manifest in a v<N>.partial folder, or None."""
    import os as _os
    from ..packs import manifests
    try:
        doc = manifests.read(_os.path.join(path, "manifest.json"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _remove_own_partials(*dirs) -> None:
    """Remove the v<N>.partial copies a publish that stopped left in dirs (only marked ones)."""
    import os as _os
    import re as _re
    import shutil
    for d in dict.fromkeys(dirs):
        for name in sorted(_os.listdir(d)) if _os.path.isdir(d) else ():
            p = _os.path.join(d, name)
            if _re.fullmatch(r"v\d+\.partial", name) and _own_partial(p):
                shutil.rmtree(p)
                console.print(f"removed {p}, left by a publish that stopped", markup=False,
                              highlight=False, soft_wrap=True)


def _release_problems(reldir):
    """What's wrong with a releases folder: leftover .partial / .replaced copies, a sync
    client's duplicate folders ("vN 2"), LATEST naming a missing release, and device locks whose
    files come from a missing release (they serve locked audio from it)."""
    import os as _os
    import re as _re
    import json as _json
    out = []
    if not _os.path.isdir(reldir):
        return out
    names = _os.listdir(reldir)
    have = {d for d in names if _re.fullmatch(r"v\d+", d) and _os.path.isdir(_os.path.join(reldir, d))}
    places = [(reldir, names)]
    staging = _release_staging(reldir, create=False)      # outside a cloud-synced publish root
    if staging != reldir and _os.path.isdir(staging):
        places.append((staging, _os.listdir(staging)))
    for where, ds in places:
        here = set(ds)
        for d in sorted(ds):
            p, v = _os.path.join(where, d), d.split(".")[0]
            # where vN goes: named as is in the releases folder, by its path from staging
            to_v = v if where == reldir else _os.path.join(reldir, v)
            new_whole = (f"{v}.partial" in here and not _own_partial(_os.path.join(where, f"{v}.partial"))
                         and "release" in (_partial_manifest(_os.path.join(where, f"{v}.partial")) or {}))
            if v not in have and new_whole and f"{v}.replaced" in here:
                # a publish --force stopped between its two renames: one line for the pair
                if d == f"{v}.partial":
                    old = _os.path.join(where, f"{v}.replaced")
                    out.append(f"leftover copies: {p} is the new {v} and {old} the {v} it replaced "
                               f"(a `publish --force` stopped before putting the new one in place). "
                               f"Keep one: rename {p} to {to_v} and delete {old}, or rename {old} "
                               f"to {to_v} and delete {p}")
                continue
            if _re.fullmatch(r"v\d+\.partial", d):
                if _own_partial(p):
                    out.append(f"leftover copy: {p if where != reldir else d} (a publish that "
                               f"stopped; the next publish removes it)")
                elif v in have:
                    out.append(f"leftover copy: {d} ({v} is in place, so delete {p})")
                elif "release" in (_partial_manifest(p) or {}):
                    out.append(f"leftover copy: {p if where != reldir else d} (the whole release "
                               f"{v}, which a publish stopped before putting in place or a sync "
                               f"client renamed back: rename it to {to_v})")
                else:
                    out.append(f"leftover copy: {d} (a publish that stopped before it finished; "
                               f"nothing uses it: delete {p}, then publish again)")
            elif _re.fullmatch(r"v\d+\.replaced", d):
                out.append(f"leftover copy: {d} (the copy of {v} a `publish --force` replaced; {v} is "
                           f"in place, so delete {p})" if v in have else
                           f"{p if where != reldir else d} is {v} as it was before a `publish "
                           f"--force` that stopped: rename it back to {to_v}")
    out += [f"sync duplicate: {d!r}" for d in names if _re.fullmatch(r"v\d+ \d+", d)]
    lp = _os.path.join(reldir, "LATEST.txt")
    if _os.path.exists(lp):
        latest = open(lp).read().strip()
        if latest and latest not in have:
            out.append(f"LATEST.txt says {latest}, which isn't there")
    from ..packs.device_lock import LOCK_DIR
    for lk in sorted(LOCK_DIR.glob("*.lock.json")) if LOCK_DIR.exists() else []:
        try:
            rels = {e.get("release") for e in _json.loads(lk.read_text()).get("files", {}).values()}
        except Exception:
            continue
        for r in sorted(x for x in rels if x and x not in have):
            out.append(f"{lk.name.split('.')[0]} lock serves files from {r}, which isn't there")
    return out


@main.command("publish", short_help="Cut an immutable release from the master.")
@click.option("--from", "src", default=None,
              help="Master dir (default: $FOURIER_CURATED_DIR or ~/Music/FourierCurated)")
@click.option("--to", "root", default=None, help="Publish root (default: [output] publish, else ~/Music/Fourier)")
@click.option("--notes", default="", help="Release notes")
@click.option("--as-version", "as_version", default=None, type=int, help="Force this release version number")
@click.option("--force", is_flag=True, default=False, help="Overwrite an existing release version")
@click.option("--release", "as_release", is_flag=True, default=False, hidden=True,
              help="(deprecated; publish always cuts a release now)")
@click.option("--dry-run", is_flag=True, default=False,
              help="Preview what the release would change against the last one (or --base); copy "
                   "nothing. Exits non-zero when it would move, change or remove a released file")
@click.option("--base", default=None,
              help="With --dry-run: the release to compare with (default: the latest)")
@click.option("--no-verify", is_flag=True, default=False,
              help="Publish even if `fourier verify` fails (a release is permanent: avoid)")
def publish(src, root, notes, as_version, force, as_release, dry_run, base, no_verify):
    """Cut an immutable release from the master (releases/v<N> under the publish root).

    Your working draft is the master ([output] master, ~/Music/FourierCurated by default):
    rebuild and iterate there freely; nothing touches the releases until you publish.
    Publishing copies the master to <publish root>/releases/v<N>, immutable and recorded in
    the database, for transfer to hardware. It never overwrites a release unless --force,
    and never one a device lock uses.

    \b
      fourier publish --dry-run          # what the release would change, against the last one
      fourier publish --notes "first set"
    """
    import os as _os
    import re as _re
    import json as _json
    import shutil
    src = (src or _master_dir()).rstrip("/")
    root = (root or _publish_root()).rstrip("/")
    if not _os.path.isdir(src):
        console.print(f"No master yet at {src}: `fourier build` makes it, then publish saves it "
                      "as a release.", style="red", markup=False, highlight=False)
        raise SystemExit(1)
    from ..safety import UnsafePath, check_output_place
    try:
        check_output_place(root, "the publish root", "publish", masters=(src,))
    except UnsafePath as err:
        console.print(str(err), style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1) from None
    _stop_on_pending_swap(src)
    reldir = _os.path.join(root, "releases")
    if not dry_run:                 # a dry run writes nothing, folders included
        _os.makedirs(reldir, exist_ok=True)
        _remove_own_partials(reldir, _release_staging(reldir))
    probs = _release_problems(reldir)
    if probs:
        console.print("the releases folder needs fixing first: " + "; ".join(probs)
                      + " (fourier releases)", style="red", markup=False, highlight=False,
                      soft_wrap=True)
        raise SystemExit(1)
    existing = sorted((d for d in (_os.listdir(reldir) if _os.path.isdir(reldir) else ())
                       if _re.fullmatch(r"v\d+", d) and _os.path.isdir(_os.path.join(reldir, d))),
                      key=lambda d: int(d[1:]))
    latest = existing[-1] if existing else None
    n = as_version if as_version else (int(latest[1:]) + 1 if latest else 1)
    target = _os.path.join(reldir, f"v{n}")
    if _os.path.exists(target):
        if not force:
            console.print(f"[red]release v{n} already exists[/red] (releases are immutable; --force to override)")
            raise SystemExit(1)
        # a device lock serves locked files from their release copy: never rewrite one in use
        from ..packs.device_lock import LOCK_DIR
        users = []
        for lp in sorted(LOCK_DIR.glob("*.lock.json")) if LOCK_DIR.exists() else []:
            try:
                if any(e.get("release") == f"v{n}" for e in _json.loads(lp.read_text()).get("files", {}).values()):
                    users.append(lp.name.split(".")[0])
            except Exception:
                pass
        if users:
            console.print(f"[red]v{n} is referenced by device locks ({', '.join(users)}); not overwriting[/red]")
            raise SystemExit(1)
    if not no_verify:
        console.print("Verifying the master before cutting a release ...")
        if not _run_verify(src, quiet=True):
            console.print("[red]not publishing:[/red] fix the FAILs above (or --no-verify)")
            raise SystemExit(1)
    from ..platforms import copy_tree
    # copy into v<N>.partial, check every file against the manifest, then rename: an
    # interrupted publish never leaves a half-copied "immutable" release. The review queue
    # (_REVIEW) is working state, never part of a release.
    if dry_run:
        console.print(f"Would publish {src} as {target} (dry run: nothing written)")
        code = (copy_tree(src, target, exclude_top=("_REVIEW",), dry_run=True)[0]
                if _os.path.isdir(reldir) else 0)        # rsync can't dry-run into a missing folder
        safe = _release_preview(base or latest, src)
        raise SystemExit(0 if code == 0 and safe else 1)
    staging = _release_staging(reldir)
    partial = _os.path.join(staging, f"v{n}.partial")
    if _os.path.exists(partial):
        shutil.rmtree(partial)
    console.print(f"Cutting RELEASE {src} -> {target} ...")
    from ._app import stop_note
    stop_note("Nothing was published: `fourier publish` starts again.")
    _os.makedirs(partial)
    open(_os.path.join(partial, PUBLISH_MARK), "w").write(f"v{n}\n")    # this copy is publish's
    try:
        code, lines = copy_tree(src, partial, exclude_top=("_REVIEW",))
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if code != 0:
        shutil.rmtree(partial, ignore_errors=True)
        need = sum(_os.path.getsize(_os.path.join(d_, f)) for d_, _d, fs in _os.walk(src)
                   if _os.path.relpath(d_, src).split(_os.sep)[0] != "_REVIEW" for f in fs)
        why = (lines[0] if lines else "")
        console.print(f"copy failed: {why[:300]}{' ...' if len(why) > 300 else ''}", style="red",
                      markup=False, highlight=False, soft_wrap=True)
        console.print(f"A release needs about {need / 2**20:,.0f} MB free in {staging}. Nothing was "
                      f"published.", style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    import hashlib as _h
    from ..packs import manifests
    man = manifests.read(_os.path.join(partial, "manifest.json"))
    bad = []
    for sect in ("categories", "sets"):
        for c, cd in (man.get(sect) or {}).items():
            for e in cd.get("entries", []):
                p = _os.path.join(partial, c, e["out"])
                if not _os.path.isfile(p) or (e.get("out_md5") and
                                              _h.md5(open(p, "rb").read()).hexdigest() != e["out_md5"]):
                    bad.append(f"{c}/{e['out']}")
    if bad:
        console.print(f"[red]the copy doesn't match the manifest ({len(bad)} files, e.g. {bad[0]}); "
                      f"left at {partial}[/red]")
        raise SystemExit(1)
    # the release's manifest says it's one, wherever it's moved or published (--to): no build
    # or sync ever writes into it (fourier/safety.py)
    _mp = _os.path.join(partial, "manifest.json")
    manifests.write(_mp, dict(man, release=f"v{n}"))
    _os.remove(_os.path.join(partial, PUBLISH_MARK))
    if _os.path.exists(target):
        old = _os.path.join(staging, f"v{n}.replaced")
        if _os.path.exists(old):
            shutil.rmtree(old)
        stop_note(f"v{n} was being replaced, and the new and the old copy may both be left "
                  f"aside: `fourier releases` says which is which and what to keep.")
        _os.rename(target, old)
        _os.rename(partial, target)
        shutil.rmtree(old)
    else:
        _os.rename(partial, target)
    stop_note(None)
    # the release must be there, whole, under its name (a sync can rename it back)
    want = sum(len(f) for d_, _d, f in _os.walk(src)
               if _os.path.relpath(d_, src).split(_os.sep)[0] != "_REVIEW")
    got = sum(len(f) for _, _, f in _os.walk(target)) if _os.path.isdir(target) else 0
    if got != want or _release_problems(reldir):
        console.print(f"[red]v{n} isn't in place after publishing[/red] ({got} of {want} files; "
                      f"{'; '.join(_release_problems(reldir)) or 'folder missing'})")
        raise SystemExit(1)
    from ..safety import record_release_path
    record_release_path(target, f"v{n}")
    # LATEST only moves forward (an --as-version on an older number doesn't rewind it)
    if not latest or n >= int(latest[1:]):
        open(_os.path.join(reldir, "LATEST.txt"), "w").write(f"v{n}\n")
    try:
        from ..packs.releases import record_release
        with session_scope() as _s:
            record_release(_s, f"v{n}", man, notes=notes or "released")
    except Exception as _e:
        console.print(f"[red]v{n} copied but not recorded in the DB: {_e}[/red] "
                      f"(fix, then: fourier releases import v{n})")
        raise SystemExit(1)
    from ..packs.releases import describe_counts, manifest_counts
    cnt = sum(len(f) for _, _, f in _os.walk(target))
    console.print(f"[green]Released[/green] v{n}: {describe_counts(manifest_counts(man))}; {cnt:,} files "
                  f"in all with the manifest and notes. In {target}", highlight=False)
    console.print(f"LATEST -> v{max(n, int(latest[1:]) if latest else n)}. Immutable; transfer to hardware. "
                  f"Working draft stays local at {src}.")


def _release_preview(base, master_dir) -> bool:
    """publish --dry-run: the master against the base release (added, moved, changed,
    removed). True when a new release would only add files."""
    if not base:
        console.print("No release yet: this would be the first.", highlight=False)
        return True
    from ..packs.releases import plan
    with session_scope() as session:
        r = plan(session, base, master_dir, log=lambda m: console.print(m, markup=False, highlight=False))
    return bool(r and r["safe"])


@main.group("releases", invoke_without_command=True,
            short_help="List the releases on disk and which is LATEST.")
@click.option("--to", "root", default=None,
              help="Publish root (default: [output] publish, else ~/Music/Fourier)")
@click.pass_context
def releases(ctx, root):
    """The releases on disk and which is LATEST, and any problem with the releases folder
    (subcommands: the database's record of them).

    \b
      fourier releases
      fourier releases recorded
      fourier releases import v1     # record a release the database doesn't know
    """
    if ctx.invoked_subcommand is None:
        _list_releases(root)


def _list_releases(root):
    """The releases under the publish root and which is LATEST."""
    import os as _os
    import re as _re
    root = (root or _publish_root()).rstrip("/")
    reldir = _os.path.join(root, "releases")
    if not _os.path.isdir(reldir):
        console.print("[yellow]no releases yet[/yellow] (fourier publish)"); return
    for p_ in _release_problems(reldir):
        console.print(f"  problem: {p_}", style="red", markup=False, highlight=False, soft_wrap=True)
    latest = None
    lp = _os.path.join(reldir, "LATEST.txt")
    if _os.path.exists(lp):
        latest = open(lp).read().strip()
    vs = sorted((d for d in _os.listdir(reldir)
                 if _re.fullmatch(r"v\d+", d) and _os.path.isdir(_os.path.join(reldir, d))),
                key=lambda d: int(d[1:]))
    if not vs:
        console.print("[yellow]no releases yet[/yellow] (fourier publish)"); return
    from ..packs.releases import describe_counts, manifest_counts
    for v in vs:
        mp = _os.path.join(reldir, v, "manifest.json"); meta = ""
        try:
            if _os.path.exists(mp):
                from ..packs import manifests
                d = manifests.read(mp)
                stamp = ", ".join(str(x) for x in (str(d.get("generated") or "")[:10],
                                                   (d.get("git_sha") and f"code {d['git_sha']}")
                                                   or (d.get("fourier_version")
                                                       and f"version {d['fourier_version']}")) if x)
                meta = f"  {describe_counts(manifest_counts(d))}" + (f"; {stamp}" if stamp else "")
        except Exception:
            pass
        console.print(f"  {v}{' *LATEST' if v == latest else ''}{meta}", highlight=False)


@tools.command("resolve", short_help="Store each sample's resolved category for queries.")
def resolve():
    """Store each sample's resolution between the two classifiers (Sononym and Live's tags)
    in the sample_resolution table: its home category, how well they agree, and both
    classifiers' labels, so the second opinion can be queried across the library.

    \b
      fourier tools resolve
    """
    from ..paths import clap_index_path
    if not clap_index_path().exists():
        console.print("Resolving categories needs the CLAP index: run `fourier build` first.",
                      style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    from ..analysis.resolution import persist_resolution
    with session_scope() as session:
        persist_resolution(session, log=lambda m: console.print(m))
    console.print("[green]Done.[/green] Query it via the sample_resolution table.")


@releases.command("import", short_help="Record a published release in the database.")
@click.argument("version")
def releases_import(version):
    """Record a published release (v1, v2, ...) in the database's release tables, for one
    published before the database knew it (publish records it)."""
    from ..packs.releases import import_version
    with session_scope() as session:
        import_version(session, version, log=lambda m: console.print(m))


@releases.command("recorded", short_help="List the releases the database records.")
def releases_recorded():
    """List the releases the database records."""
    from ..packs.releases import list_releases
    with session_scope() as session:
        rows = list_releases(session)
    if not rows:
        console.print("No releases recorded in the database yet (`fourier publish` records each "
                      "one; `fourier releases import vN` records one published elsewhere).",
                      markup=False, highlight=False); return
    for v, t, seed, sha, n, notes in rows:
        console.print(f"  {v:<5} {n or 0:>6} audio files  seed={seed}  {sha or '-':<8}  {(t or '')[:10]}  "
                      f"{notes or ''}", highlight=False)
