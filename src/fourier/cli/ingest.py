"""fourier tools scan and import-folder: samples and provider metadata into the library database."""
from __future__ import annotations

from pathlib import Path

import click

from ..ingest.importer import scan_filesystem, sync_from_sononym
from ._app import _table_cols, console, log, main, tools  # noqa: F401

# ---------------------------------------------------------------------------
# fourier tools scan (and the first stage of fourier build)
# ---------------------------------------------------------------------------
SCAN_STEPS = ("sononym", "ableton", "files")


def run_scan(only=(), prune=False, walk=False, allow_missing=False) -> None:
    """Bring the library into the database: Sononym's analysis when it's there (new and
    changed samples), otherwise a walk of the library folders, then Live's auto-tags when
    Live's file index is there (after the walk, so the files it found get their tags too).
    only: just these SCAN_STEPS; prune: drop samples Sononym no longer has; walk: walk the
    library folders after Sononym too (files Sononym hasn't indexed). Exits 2 when a library
    folder to walk is missing or empty (nothing is marked missing then), or when a linked
    folder in one is unavailable (its samples are marked missing), unless allow_missing."""
    from ..ingest.ableton_tags import latest_live_db
    from ..metadata import providers as P
    from ..places import library, library_roots, sononym_db
    names = P.configured()
    want = set(only) if only else None

    def use(step, available):
        if want is not None:
            return step in want
        if names is not None and step in ("sononym", "ableton"):
            return step in names
        return available

    sdb = sononym_db()
    synced = False
    UNAVAILABLE.clear()
    if use("sononym", sdb is not None and sdb.exists()):
        console.print("[bold]Sononym[/bold]")
        _sononym_sync(prune=prune)
        synced = True
    # without Sononym the walk is how files get in (Live's tags only label samples the
    # database has); with it, only when asked
    bad = []
    if use("files", not synced or walk):
        roots = library_roots()
        if not roots and library()[1]:
            console.print(f"The library names its folder by name only ({', '.join(library()[1])}), "
                          f"and a walk needs the folder's path: nothing to scan. Give the full path "
                          f"in fourier.toml to scan it.", style="yellow", markup=False, highlight=False)
        elif not roots:
            console.print("[red]No library folder: set library in fourier.toml (fourier setup)[/red]")
            raise SystemExit(2)
        for r in roots:
            console.print(f"[bold]Files under {r}[/bold]")
            if not _walk(r):
                bad.append(str(r))
    if use("ableton", latest_live_db() is not None):
        console.print("[bold]Live's auto-tags[/bold]")
        _ableton_tags()
    if bad or (UNAVAILABLE and not allow_missing):
        raise SystemExit(2)


@tools.command("scan", short_help="Bring new and changed library files into the database.")
@click.option("--only", "only", multiple=True, type=click.Choice(SCAN_STEPS),
              help="Just this source (repeatable)")
@click.option("--prune", is_flag=True, default=False,
              help="Sononym: also drop samples Sononym no longer has")
@click.option("--walk", is_flag=True, default=False,
              help="With Sononym: also walk the library folders (adds the files it hasn't "
                   "indexed, marks samples whose files are gone)")
def scan(only, prune, walk):
    """Bring the library into the database (`fourier build` does this first): Sononym's
    analysis when it's there (new and changed samples), otherwise a walk of the library
    folders (fourier.toml's library), then Live's auto-tags when Live's file index is there.

    The walk reads every audio file soundfile can (.wav .aif .aiff .aifc .flac .mp3 .ogg
    .opus .caf), follows folder symlinks that lead out of the library folder, says what it
    skipped, and marks samples it no longer finds missing (a build leaves them out). A
    library folder that is missing or empty (an unmounted drive) stops it, and nothing is
    marked. With Sononym, --walk walks too: the files it adds wait for Sononym's analysis
    before a build uses them, and samples whose files are gone are left out of builds.

    \b
      fourier tools scan
      fourier tools scan --only files
      fourier tools scan --walk          # Sononym, then a walk of the library folders
      fourier tools scan --prune
    """
    run_scan(only, prune, walk)


def _sononym_sync(prune=False) -> None:
    """Sononym's new and changed samples into the database (and with prune, drop the ones it
    no longer has)."""
    console.print("[cyan]Syncing from Sononym DB (incremental)...[/cyan]")
    if prune:
        console.print("  [yellow]--prune enabled: removed samples will be deleted[/yellow]")
    tick = {"last": 0}

    def progress(current, total, path):
        if current - tick["last"] >= 500 or current == total:
            console.print(f"  {current:,}/{total:,}: {Path(path).name}")
            tick["last"] = current

    counts = sync_from_sononym(progress_cb=progress, prune=prune)
    console.print(
        f"[green]Sync complete:[/green] "
        f"{counts['created']} new, {counts['updated']} updated, "
        f"{counts['unchanged']:,} unchanged, {counts['removed']} removed from Sononym"
        + (f", {counts.get('pruned', 0)} pruned" if prune else "")
    )
    if counts.get("cloud_only"):
        console.print(f"[yellow]{counts['cloud_only']:,} cloud-only file(s) not hashed yet (not on "
                      f"this machine); a later scan hashes them once they're downloaded[/yellow]")


def _walk(root) -> bool:
    """The audio files under root that the database doesn't have yet, what the walk skipped,
    and the samples it no longer finds. False when root is missing or empty (not walked)."""
    from ..ingest.walk import root_state
    why = root_state(root)
    if why:
        console.print(f"The library folder {root} is {why}: an unmounted drive or a renamed "
                      f"folder? Reconnect or fix it (fourier.toml's library), then scan again. "
                      f"Nothing was scanned there and no sample was marked missing.",
                      style="red", markup=False, highlight=False, soft_wrap=True)
        return False
    console.print(f"[cyan]Scanning:[/cyan] {root}")

    def progress(i, path):
        if i % 500 == 0 and i > 0:
            console.print(f"  {i:,} scanned: {Path(path).name}")

    counts = scan_filesystem(progress_cb=progress, root=Path(root))
    say = lambda m, style=None: console.print(m, style=style, markup=False, highlight=False,
                                              soft_wrap=True)
    say(f"Scan complete: {counts['created']:,} new, {counts['skipped']:,} already known, "
        f"{counts.get('errors', 0)} errors", "green")
    for line in scan_report(counts):
        say(*line)
    UNAVAILABLE.extend(counts.get("unavailable_links") or ())
    return True


# the linked folders the last walk in this process found unavailable [(link, target)]: a
# build stops on them unless --allow-missing (cli/build.py)
UNAVAILABLE: list = []


def scan_report(counts) -> list[tuple]:
    """The lines after a walk's summary: (text, style)."""
    out = []
    if counts.get("errors"):
        files = counts.get("error_files") or []
        more = counts["errors"] - len(files)
        out.append((f"  {counts['errors']:,} file(s) couldn't be read (corrupt or empty; left out):", "yellow"))
        out += [(f"    {f}", "yellow") for f in files]
        if more > 0:
            out.append((f"    ... and {more:,} more", "yellow"))
    skipped = counts.get("skipped_ext") or {}
    if skipped:
        from ..ingest.formats import AUDIO_EXTS
        audio = {e: n for e, n in skipped.items() if e in AUDIO_EXTS}
        other = {e: n for e, n in skipped.items() if e not in AUDIO_EXTS}
        if audio:
            out.append((f"  {sum(audio.values()):,} audio file(s) skipped, in a format soundfile can't "
                        f"read here: " + ", ".join(f"{n:,} {e}" for e, n in sorted(audio.items()))
                        + " (convert them to WAV or FLAC to use them)", "yellow"))
        if other:
            top = sorted(other.items(), key=lambda x: (-x[1], x[0]))
            shown = ", ".join(f"{n:,} {e}" for e, n in top[:8]) + (", ..." if len(top) > 8 else "")
            out.append((f"  {sum(other.values()):,} other file(s) skipped (not audio): {shown}", "dim"))
    if counts.get("links_followed") or counts.get("links_inside") or counts.get("loops"):
        out.append((f"  folder symlinks: {counts.get('links_followed', 0):,} followed out of the "
                    f"library folder, {counts.get('links_inside', 0):,} pointing inside it (walked "
                    f"there), {counts.get('loops', 0):,} leading back to a folder already walked "
                    f"(skipped)", "dim"))
    if counts.get("unlistable"):
        out.append((f"  {counts['unlistable']:,} folder(s) couldn't be listed; samples in them "
                    f"weren't marked missing", "yellow"))
    for link, target in counts.get("unavailable_links") or ():
        out.append((f"  A linked folder is unavailable: {link} -> {target} (a drive that isn't "
                    f"plugged in?). Its samples are left out; a build stops until it's back, or "
                    f"builds without them with --allow-missing.", "red"))
    if counts.get("moved"):
        out.append((f"  {counts['moved']:,} file(s) moved or renamed in the library (the same "
                    f"content at a new path): their analysis and ratings follow them", "dim"))
    gone = counts.get("missing") or 0          # the old rows of moved files aren't counted
    if gone > 0:
        out.append((f"  {gone:,} sample(s) in the database weren't found under it "
                    f"({counts.get('newly_missing', 0):,} newly): moved elsewhere, renamed or "
                    f"deleted; a build leaves them out. `fourier tools db-stats --missing --prune` "
                    f"removes them.", "yellow"))
    if counts.get("found_again"):
        out.append((f"  {counts['found_again']:,} sample(s) marked missing are back", "dim"))
    if counts.get("reused"):
        out.append((f"  {counts['reused']:,} unchanged folder(s) not listed again", "dim"))
    return out


def _ableton_tags(db_path=None) -> None:
    """Live's auto-tags, a second opinion next to Sononym: read from Live's file index and
    stored on each matching sample (Sample.ableton_tags, matched by pack and file name).
    Live's files in every format the walk reads (formats.readable_exts), or, when the
    database holds Sononym's samples, Live's list as it was (formats.LIVE_TAG_EXTS)."""
    from sqlalchemy import text

    from ..db.session import get_session
    from ..ingest.ableton_tags import import_ableton_tags
    from ..ingest.formats import LIVE_TAG_EXTS, readable_exts
    sess = get_session()
    extended = sess.execute(text("SELECT 1 FROM sononym_meta LIMIT 1")).first() is None
    import_ableton_tags(sess, db_path=db_path, log=lambda mm: console.print(mm),
                        exts=readable_exts() if extended else LIVE_TAG_EXTS)
    console.print("[green]Done.[/green] Ableton tags stored on Sample.ableton_tags.")


# ---------------------------------------------------------------------------
# fourier tools import-folder: copy a folder of audio into the library
# ---------------------------------------------------------------------------
@tools.command("import-folder", short_help="Copy a folder of audio into a library folder.")
@click.argument("src", type=click.Path(exists=True, file_okay=False))
@click.option("--to", "dest_root", default=None,
              help="Destination (default: <library>/<the source folder's name>)")
@click.option("--no-convert-ogg", is_flag=True, default=False,
              help="Copy .ogg as-is instead of decoding it to WAV")
@click.option("--no-preparers", is_flag=True, default=False,
              help="Don't offer files to the preparers fourier.toml enables (fourier/preparers.py)")
@click.option("--max-mb", type=float, default=0.0,
              help="Skip source files larger than N MB (0 = no cap).")
@click.option("--clean", is_flag=True, default=False,
              help="Wipe the destination first (default: incremental, skip unchanged)")
@click.option("--adopt", is_flag=True, default=False,
              help="Use an existing destination that holds only earlier imports (marks it as one)")
def import_folder(src, dest_root, no_convert_ogg, no_preparers, max_mb, clean, adopt):
    """Copy a folder of audio into a library folder (with several library folders, pass
    --to). The only command that writes into the library, and it only adds copies.

    Copies WAV/AIFF/FLAC verbatim and decodes OGG/Opus/MP3/CAF to WAV. A file whose extension an
    enabled preparer names (a codec soundfile lacks) is written as WAV. The source folder
    is never modified. Re-runnable: incremental by default, --clean wipes and reloads.

    \b
      fourier tools import-folder ~/Downloads/new-pack
    """
    from ..ingest.folder import ingest_folder
    from ..preparers import PreparerError

    if not dest_root:
        from ..places import library_roots
        if not library_roots():
            console.print("[red]No library folder: set library in fourier.toml, or pass --to[/red]")
            raise SystemExit(2)
        dest_root = str(library_roots()[0] / Path(src).expanduser().resolve().name)
    try:
        ingest_folder(src, dest_root, convert_ogg=not no_convert_ogg, max_mb=max_mb,
                      clean=clean, prepare=not no_preparers, adopt=adopt,
                      log=lambda m: console.print(m))
    except (PreparerError, ValueError, FileNotFoundError) as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(2) from None
    console.print("[green]Done.[/green] Next: `fourier build` scans and analyzes it.")
