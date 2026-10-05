"""fourier build, why, diff and verify, and fourier tools audit: the master."""
from __future__ import annotations

from collections import Counter

import click

from ..db.session import session_scope
from rich import box
from rich.table import Table

from ._app import _table_cols, console, log, main, tools  # noqa: F401
from ..places import master_dir as _master_dir
from ..safety import CloudOnlySources
from .config import _tunable_meta

# ---------------------------------------------------------------------------
# fourier build: scan, analyze, curate CATEGORY/family/file trees, verify
# ---------------------------------------------------------------------------
# the stages a build prints a header for (ctx.meta), and new samples worth an up-front estimate
STAGES_KEY = "fourier.build_stages"
QUIET_KEY = "fourier.build_quiet"     # a quiet build's capture (cli/quiet.py), started at Build
ESTIMATE_FROM = 200


@main.command("build", short_help="Scan and analyze what's new, build the master and verify it.")
@click.argument("category", required=False)
@click.option("--out", "out_dir", default=None,
              help="Output root (default: the master, $FOURIER_CURATED_DIR or [output] master)")
@click.option("--per-family", default=25, show_default=True, type=int,
              help="Samples per family folder")
@click.option("--clap-z", default=1.0, show_default=True, type=float,
              help="Min CLAP phrase z-score for a phrase to lead a folder name")
@click.option("--no-transcode", is_flag=True, default=False,
              help="Copy aif/mp3 verbatim instead of converting to 16-bit WAV")
@click.option("--describe/--no-describe", "describe", default=None,
              help="Folder descriptions from a local LLM (Ollama); default: DESCRIBE, off")
@click.option("--all", "do_all", is_flag=True, default=False,
              help="Build every category (what `fourier build` does when no CATEGORY is given)")
@click.option("--rebuild-index", is_flag=True, default=False,
              help="Rebuild the CLAP index from the DB first if it is stale")
@click.option("--base", default=None,
              help="Build ADDITIVELY as a superset of this published version (e.g. v2): copy it verbatim, add only new samples, never move/remove existing paths")
@click.option("--jobs", "-j", default=1, show_default=True, type=int,
              help="Parallel category workers for --all (process pool; per-category output is identical to serial)")
@click.option("--no-loudness", is_flag=True, default=False,
              help="Skip the per-kind trim/loudness export pass (copy/transcode verbatim)")
@click.option("--only-pack", default=None,
              help="With --base: add only samples whose library path contains this (a new pack)")
@click.option("--since", default=None,
              help="With --base: add only samples scanned on or after this date (YYYY-MM-DD)")
@click.option("--allowance", default=None, type=float,
              help="With --base: share of each category's budget it may grow by (default 0.15)")
@click.option("--resume", is_flag=True, default=False,
              help="--all: keep the categories an unfinished build already made, if nothing changed")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print files and disk space per category (every category unless one is given), "
                   "and what size = auto does; write nothing")
@click.option("--no-scan", is_flag=True, default=False,
              help="Skip the scan and the analysis: build from the database as it is")
@click.option("--allow-missing", is_flag=True, default=False,
              help="Build even when a linked folder in the library is unavailable (its samples are left out)")
@click.option("--verbose/--quiet", "verbose", default=None,
              help="Everything the Build and Verify stages say (default: on when the output isn't "
                   "a terminal); --quiet: a summary, with the rest in a log file")
@click.pass_context
def build(ctx, category, out_dir, per_family, clap_z, no_transcode, describe, do_all, rebuild_index, base,
          jobs, no_loudness, only_pack, since, allowance, resume=False, dry_run=False, no_scan=False,
          allow_missing=False, verbose=None):
    """Build the master, one step after another: scan the library for new and changed files
    (Sononym's analysis and Live's tags too, when they're there), analyze what's new, curate
    the CATEGORY/family/file folders and verify them. A whole build (no CATEGORY, or --all)
    replaces the master only when every category built and verify passed.

    Curating clusters each category's CLAP embeddings into families, names each family from
    what its sounds share, picks the samples that represent it best, and exports WAV. On a
    terminal the Build and Verify stages show a summary and keep the rest in a log
    (<home>/logs); --verbose prints it all.

    \b
      fourier build                     # everything (as --all); the first run analyzes the library
      fourier build -j 6                # the same, six categories at a time
      fourier build --dry-run           # files, disk space and time per category; writes nothing
      fourier build KICKS --out /tmp/k  # one category, into a scratch folder
      fourier build --all --no-scan     # from the database as it is (no scan, no analysis)
    """
    kw = {k: v for k, v in locals().items() if k not in ("ctx", "verbose")}
    from .quiet import Quiet, wants_quiet
    if verbose is None and ctx.find_root().params.get("verbose"):
        verbose = True                  # `fourier -v build`: the debug log is wanted on screen
    if dry_run or not wants_quiet(verbose):
        return _build(ctx, **kw)
    q = Quiet()
    ctx.meta[QUIET_KEY] = q
    ok = False
    try:
        _build(ctx, **kw)
        ok = True
    except SystemExit as e:
        ok = e.code in (0, None)
        raise
    finally:
        started = q.started
        q.stop()
        if started:
            for line in (q.summary() if ok else q.failure()):
                console.print(line, markup=False, highlight=False, soft_wrap=True)


def _build(ctx, category, out_dir, per_family, clap_z, no_transcode, describe, do_all, rebuild_index, base,
          jobs, no_loudness, only_pack, since, allowance, resume=False, dry_run=False, no_scan=False,
          allow_missing=False):
    """`fourier build` itself (build() decides whether its Build and Verify stages print
    everything or a summary, QUIET_KEY)."""
    import os as _os
    if dry_run:                     # reads only: no resolved config left in the home either
        from ._app import read_only_config
        read_only_config(ctx)
    from ..packs.curate import build_taxonomy
    from ..packs.curate_config import CATEGORIES, DESCRIBE
    no_describe = not (DESCRIBE if describe is None else describe)

    from ..packs.ratings import harvest, live_master_dir
    out_dir = out_dir or live_master_dir() or _master_dir()
    if not dry_run:
        out_dir = _check_build_target(out_dir)
    # no category builds them all (as --all does); an additive build and a dry run too
    if not category and not base and not dry_run:
        do_all = True
    if do_all or (base and not category) or (dry_run and not category):
        from ..packs.curate_config import CATEGORIES_OFF
        cats = [c for c in CATEGORIES if c not in CATEGORIES_OFF]      # the categories knob
    elif category and category.upper() in CATEGORIES:
        cats = [category.upper()]
    else:
        console.print(f"[red]Give a known category or --all. Known: {', '.join(CATEGORIES)}[/red]")
        raise SystemExit(1)
    _preset_note()
    if dry_run:
        from ..db.session import read_only
        read_only()                  # the database too (unless it's new, or its schema is behind)
        _dry_run(cats, jobs)
        return
    # curation reads each sample's path under the library (vendor/pack/...), and verify reads
    # the master's sources the same way: without a library they disagree about every pack rule
    from ..places import PlacesError, library as _library
    try:
        _roots, _names = _library()
    except PlacesError as e:
        console.print(f"[red]library: {e}[/red]")
        raise SystemExit(2) from None
    if not _roots and not _names:
        console.print("No sample folder set up yet: `fourier setup` asks where your samples are "
                      "(it writes library = [...] in fourier.toml).", style="red", markup=False,
                      highlight=False)
        raise SystemExit(2)
    # a device whose limits leave no room for the master's names (doctor FAILs it too)
    from ..knobs import device_problems
    for _d, _why in device_problems():
        console.print(f"device {_d}: {_why}.", style="red", markup=False, highlight=False, soft_wrap=True)
    if device_problems():
        raise SystemExit(2)
    if (base or do_all) and _os.path.exists(out_dir.rstrip("/") + ".next"):
        _check_own(out_dir.rstrip("/") + ".next")     # before a scan that can take a while
    _check_library_roots(out_dir)
    _check_master_library(out_dir)
    from ..demo import active as _demo
    if not no_scan and _demo():
        no_scan = True
        console.print("The demo rebuilds from its stand-in analysis (no scan, no model): "
                      "ratings and settings you change take effect.", markup=False,
                      highlight=False, soft_wrap=True)
    ctx.meta[STAGES_KEY] = (["Build", "Verify"] if no_scan else
                            ["Scan the library", "Analyze what's new", "Build", "Verify"])
    if not no_scan:
        _scan_and_analyze(ctx, out_dir, allow_missing)
    _stage("Build")
    _q = ctx.meta.get(QUIET_KEY)
    if _q is not None:                  # the detail from here on goes to the log
        _q.start(out_dir, len(cats))
    _check_missing_sources(out_dir)
    # the generic metadata tables must cover the library before workers read them
    from ..metadata.providers import ProviderError, active, describe
    from ..metadata.rows import ensure_current, outside_note
    from ..metadata.store import SOURCES, unmapped
    try:
        with session_scope() as _s:
            _act = active(_s)
            console.print(f"[dim]{describe(_act)}[/dim]")
            ensure_current(_s, log=lambda m: console.print(f"[dim]{m}[/dim]"))
            _other = outside_note(_s)
            if _other:                  # another config's library shares this home's database
                console.print(_other + ".", style="yellow", markup=False, highlight=False, soft_wrap=True)
            _un = {prov: unmapped(_s, prov) for prov in SOURCES if prov in _act.names}
    except ProviderError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(2) from None
    _un = {prov: u for prov, u in _un.items() if u}
    # with Sononym in use, the fallback providers run in shadow: recorded and scored against it
    if not _act.fallback:
        from ..metadata.shadow import SHADOW, agreement, format_agreement
        from ..metadata.shadow import ensure_current as _shadow_current
        with session_scope() as _s:
            _shadow_current(_s, log=lambda m: console.print(f"[dim]{m}[/dim]"))
            for _prov in SHADOW:
                console.print(f"[dim]shadow {format_agreement(agreement(_s, _prov))}[/dim]")
    if _un:   # a label the taxonomy can't see would drop its samples from every category
        for prov, u in _un.items():
            names = ", ".join(f"{k}:{lab} ({n:,})" for (k, lab), n in sorted(u.items()))
            console.print(f"[red]{prov} labels with no canonical mapping: {names}. Add them to "
                          f"config/providers/{prov}.yaml.[/red]")
        raise SystemExit(2)
    # the library rules (pack and folder names) ship empty: say so when no overlay set any
    from ..settings import overrides as _overrides
    library = [k for k, (cls, _) in _tunable_meta().items() if cls == "library"]
    if library and not any(k in _overrides() for k in library):
        console.print("[yellow]No library overlay: the library rules (config/tunables.yaml, class "
                      "\"library\") are empty for this build. `fourier config show` lists the layers.[/yellow]")
    # harvest Ableton "Fourier|..." ratings BEFORE a rebuild replaces the folders holding
    # them (Live keeps the tags in XMP sidecars inside the master): the master being
    # rebuilt, and the live master if that's a different folder. A failed harvest stops
    # the build, since continuing would throw the unharvested ratings away.
    targets = []
    for d in (out_dir, live_master_dir()):
        if d and _os.path.exists(_os.path.join(d, "manifest.json")) \
                and _os.path.realpath(d) not in {_os.path.realpath(t) for t in targets}:
            targets.append(d)
    for d in targets:
        try:
            harvest(d, log=lambda m: console.print(m, markup=False, highlight=False))
        except Exception as e:
            console.print(f"[red]ratings harvest failed for {d}: {e}[/red]\n"
                          "Not rebuilding: the rebuild would discard ratings that weren't harvested.")
            raise SystemExit(1)
    # ratings follow files the library moved (matched by file hash)
    try:
        from ..packs.ratings import rekey_by_hash
        with session_scope() as session:
            rekey_by_hash(session, log=lambda m: console.print(m, markup=False, highlight=False))
    except Exception as e:
        console.print(f"[yellow]ratings rekey skipped: {e}[/yellow]")
    _warn_gone_keeps()
    if base:
        # like a whole build: into <out>.next, swapped in only when verify passes
        import shutil as _sh
        from ..packs.builddiff import archive_build, write_changelog
        from ..packs.releases import additive_build
        build_dir = out_dir.rstrip("/") + ".next"
        if _os.path.exists(build_dir):
            _check_own(build_dir)
            _sh.rmtree(build_dir)
        try:
            with session_scope() as session:
                got = additive_build(session, base, build_dir, per_family=per_family,
                                     transcode=not no_transcode, describe=not no_describe,
                                     clap_z=clap_z, categories=cats, loudness=not no_loudness,
                                     only_pack=only_pack, since=since, allowance=allowance,
                                     log=lambda m: console.print(m))
        except CloudOnlySources as e:
            _stop_cloud_only(e, out_dir, build_dir)
        short = (got or {}).get("failed") or {} if isinstance(got, dict) else {}
        if short:                       # a pick it couldn't export: the master stays as it was
            console.print(_export_failure_message(short, out_dir) + f" The additive build is in {build_dir}.",
                          style="red", markup=False, highlight=False, soft_wrap=True)
            raise SystemExit(1)
        ok = _post_build_ratings(build_dir)
        write_changelog(build_dir, log=lambda m: console.print(m))
        if not ok:
            console.print(f"[red]verify failed: the master at {out_dir} is unchanged; the additive "
                          f"build is in {build_dir}.[/red]")
            raise SystemExit(1)
        prev = _swap_in(build_dir, out_dir)
        archive_build(out_dir)
        console.print(f"Additive build on {base} in place at {out_dir}{_kept_note(prev, out_dir)}. "
                      f"Preview: fourier publish --dry-run --base {base}", style="green", markup=False,
                      highlight=False, soft_wrap=True)
        return
    if do_all:
        # a whole build goes into <out>.next and replaces the master only when every
        # category built and verify passes; the old master is kept as <out>.prev
        import shutil as _sh
        from ..packs.builddiff import archive_build, write_changelog
        from pathlib import Path
        from ..packs.progress import PROGRESS_DIR, Progress, fingerprint, resumable
        import time as _time
        _t_build = _time.perf_counter()
        build_dir = out_dir.rstrip("/") + ".next"
        opts = dict(per_family=per_family, clap_z=clap_z, transcode=not no_transcode,
                    describe=not no_describe, loudness=not no_loudness)
        if _os.path.exists(build_dir):
            _check_own(build_dir)
        _preflight()
        _scaled = _library_scale()
        if _scaled:
            opts["scale"] = _scaled
        if not resume and resumable(build_dir):
            console.print(f"[yellow]{build_dir} holds an unfinished build; `--resume` would pick it "
                          f"up. Starting over.[/yellow]")
        with session_scope() as _s:
            progress = Progress(build_dir, fingerprint(_s, cats, opts))
        done = progress.start(resume, log=lambda m: console.print(m))
        failed, empty = [], []
        try:
            if jobs > 1 and len(cats) > 1:
                from ..db.session import get_engine
                from ..packs.curate import build_all_parallel
                dbp = get_engine().url.database
                console.print(f"[cyan]Curating {len(cats) - len(done)} categories with {jobs} workers...[/cyan]")
                res = build_all_parallel(dbp, cats, build_dir, jobs, opts, log=lambda m: console.print(m),
                                         progress=progress, done=done, empty=empty)
                failed = sorted(set(cats) - {r[0] for r in res} - set(empty))
            else:
                res = _build_serial(cats, done, build_dir, progress, failed, empty, rebuild_index,
                                    dict(opts))
        except CloudOnlySources as e:
            _stop_partial(build_dir, out_dir, str(e))
        except SystemExit:
            raise
        except Exception as e:      # before or between categories: nothing a resume needs, or a resume
            _stop_partial(build_dir, out_dir, f"build stopped: {e}")
        unrec = _unrecognized_note(build_dir) if (empty or not res) else []
        if empty:
            console.print(f"[yellow]{len(empty)} categor{'y' if len(empty) == 1 else 'ies'} left empty: "
                          + ("too few samples the rules recognize for them" if unrec else
                             "the library has too few samples for them")
                          + f" ({', '.join(sorted(empty))}).[/yellow]")
        for line in unrec:
            console.print(line, style="yellow", markup=False, highlight=False, soft_wrap=True)
        if not res and not failed:
            _remove_partial(build_dir)
            console.print("[red]Nothing to build: no category has enough samples for a folder. Add "
                          "samples to the library, then `fourier build` again.[/red]")
            raise SystemExit(1)
        short = _export_failures(res)
        if short and not failed:
            for cat in short:           # a resume builds them again (the rest it keeps)
                (Path(build_dir) / PROGRESS_DIR / f"{cat}.json").unlink(missing_ok=True)
            _stop_partial(build_dir, out_dir, _export_failure_message(short, out_dir))
        if not failed:
            progress.finish()
        if failed:
            console.print(f"[red]{len(failed)} categor{'y' if len(failed) == 1 else 'ies'} failed "
                          f"({', '.join(failed)}): the master at {out_dir} is unchanged.[/red]")
            _stop_partial(build_dir, out_dir)
        console.print(f"[green]Built {len(res)} categor{'y' if len(res) == 1 else 'ies'}.[/green]")
        ok = _post_build_ratings(build_dir)
        write_changelog(build_dir, log=lambda m: console.print(m))
        if not ok:
            console.print(f"[red]verify failed: the master at {out_dir} is unchanged; the new build "
                          f"is in {build_dir} (see its CHANGELOG.md).[/red]")
            raise SystemExit(1)
        # ratings made in Live while this build ran are in the old master's sidecars:
        # harvest them and write them into the new one before it replaces the old
        if _os.path.exists(_os.path.join(out_dir, "manifest.json")):
            try:
                from ..packs.ratings import apply_tags
                harvest(out_dir, log=lambda m: console.print(m, markup=False, highlight=False))
                apply_tags(build_dir, log=lambda m: None)
            except Exception as e:
                console.print(f"[red]ratings harvest before the swap failed: {e}[/red]\n"
                              f"The new build stays in {build_dir}; the master is unchanged.")
                raise SystemExit(1)
        prev = _swap_in(build_dir, out_dir)
        archive_build(out_dir)
        if not done:                    # a resumed build's wall time says little
            from ..timings import record_build
            record_build(res, jobs if len(cats) > 1 else 1, _time.perf_counter() - _t_build)
        try:
            from ..packs.audiocache import prune
            prune(log=lambda m: console.print(m))
        except Exception:
            pass
        console.print(f"New master in place at {out_dir}{_kept_note(prev, out_dir)}.", style="green",
                      markup=False, highlight=False, soft_wrap=True)
        return
    from ..packs.curate import TooFewSamples
    built = []
    _single_failures = {}
    _scaled = _library_scale()
    with session_scope() as session:
        for cat in cats:
            console.print(f"[cyan]Curating {cat}...[/cyan]")
            try:
                summary = build_taxonomy(session, cat, out_dir, per_family=per_family, clap_z=clap_z,
                                         transcode=not no_transcode, describe=not no_describe,
                                         rebuild_index=rebuild_index, loudness=not no_loudness,
                                         **({"scale": _scaled} if _scaled else {}),
                                         log=lambda m: console.print(m))
                built.append(cat)
                if isinstance(summary, dict):
                    _single_failures[cat] = int(summary.get("failed") or 0)
            except CloudOnlySources as e:
                _stop_cloud_only(e, out_dir)
            except TooFewSamples as e:
                console.print(f"{e}: left empty (`fourier why <file>` says why a sample was left out)",
                              style="yellow", markup=False, highlight=False)
            except Exception as e:
                console.print(str(e) if str(e).startswith(cat) else f"{cat}: {e}", style="red",
                              markup=False, highlight=False)
    if len(built) < len(cats):
        for line in _unrecognized_note(out_dir):
            console.print(line, style="yellow", markup=False, highlight=False, soft_wrap=True)
    if not built:
        console.print(f"Nothing built: {out_dir} is as it was.", style="red", markup=False,
                      highlight=False)
        raise SystemExit(1)
    _post_build_ratings(out_dir)
    short = {c: n for c, n in _single_failures.items() if n}
    if short:
        console.print(_export_failure_message(short, None), style="red", markup=False,
                      highlight=False, soft_wrap=True)
        raise SystemExit(1)


def _build_serial(cats, done, build_dir, progress, failed, empty, rebuild_index, kw):
    """A whole build's categories one after another (-j 1), then one surplus round (size =
    "auto"), the cross-category dedupe and the manifest: as build_all_parallel does. A
    category too few samples could fill is named in empty; one that fails, in failed."""
    from ..packs.curate import (POOL_SURPLUS, TooFewSamples, _dedupe_across, build_taxonomy,
                                merge_manifest, surplus_budgets)
    res = []
    with session_scope() as session:
        for cat in cats:
            if cat in done:
                res.append((cat, *done[cat]))
                continue
            console.print(f"[cyan]Curating {cat}...[/cyan]")
            try:
                summary, entries = build_taxonomy(
                    session, cat, build_dir, rebuild_index=rebuild_index,
                    return_entries=True, log=lambda m: console.print(m), **kw)
                res.append((cat, summary, entries))
                progress.done(cat, summary, entries)
            except CloudOnlySources:
                raise
            except TooFewSamples as e:
                console.print(f"{e}: left empty", style="yellow", markup=False, highlight=False)
                empty.append(cat)
            except Exception as e:
                console.print(str(e) if str(e).startswith(cat) else f"{cat}: {e}", style="red",
                              markup=False, highlight=False)
                failed.append(cat)
        # (none in a library-scaled build: its budgets already follow the pools)
        raised = surplus_budgets(res) if POOL_SURPLUS and not failed and not kw.get("scale") else {}
        for cat, budget in sorted(raised.items()):      # one surplus round (size = "auto")
            console.print(f"[cyan]Curating {cat} again with the surplus: {budget} files...[/cyan]")
            try:
                summary, entries = build_taxonomy(
                    session, cat, build_dir, return_entries=True,
                    budget_override=budget, log=lambda m: console.print(m), **kw)
                summary["budget"] = budget
                res = [(cat, summary, entries) if r[0] == cat else r for r in res]
            except CloudOnlySources:
                raise
            except Exception as e:
                console.print(f"{cat}: {e}", style="red", markup=False, highlight=False)
                failed.append(cat)
    # as the parallel path does: each source in one category, one manifest write
    _dedupe_across(build_dir, res)
    merge_manifest(build_dir, res)
    return res


# a build stops before it starts when more than this share of the previous master's sources
# are gone from disk while the database would still pick them (a renamed folder, deleted
# files Sononym's sync keeps); fewer are left to the export, which stops on any it can't read
MISSING_SOURCES_SHARE = 0.02
MISSING_SOURCES_MIN = 3
MISSING_LISTED = 10


def _check_library_roots(out_dir) -> None:
    """Stop (exit 1) when a library folder fourier.toml names is missing, empty or can't be
    read: an unmounted drive or a renamed folder. A build would otherwise read none of its
    samples, export nothing from it and replace the master with what's left. Folders named
    by name only can't be checked."""
    from ..ingest.walk import root_state
    from ..places import library_roots
    bad = [(r, why) for r in library_roots() for why in [root_state(r)] if why]
    if not bad:
        return
    for r, why in bad:
        console.print(f"The library folder {r} is {why}.", style="red", markup=False,
                      highlight=False, soft_wrap=True)
    console.print(f"An unmounted drive or a renamed folder? Reconnect it, or point library in "
                  f"fourier.toml at where it is now, then build again. Build stopped: nothing in "
                  f"{out_dir} changed.", style="red", markup=False, highlight=False, soft_wrap=True)
    raise SystemExit(1)


def _check_master_library(out_dir) -> None:
    """Stop (exit 1) before anything is scanned or written when the master at out_dir holds
    another library's build (its manifest's `library` shares no folder with this config's):
    a second config left at the default master would replace the first one's. A master
    from before manifests recorded their library is said once (this build records it)."""
    from ..places import master_library_problem
    got = master_library_problem(out_dir)
    if not got:
        return
    kind, why = got
    if kind == "old":
        console.print(f"Note: {why}.", style="yellow", markup=False, highlight=False, soft_wrap=True)
        return
    console.print(f"{why[0].upper()}{why[1:]}. Build stopped: nothing in {out_dir} changed.",
                  style="red", markup=False, highlight=False, soft_wrap=True)
    raise SystemExit(1)


def _check_missing_sources(out_dir) -> None:
    """Stop (exit 1) when more than MISSING_SOURCES_SHARE of the master's sources (its
    manifest) are gone from disk and still in the database unmarked: the build would pick
    them again and fail to export them. Sources a walk marked missing, or that the
    database no longer has, don't count: the build leaves them out on its own."""
    import json as _json
    import os as _os
    from sqlalchemy import text
    from ..metadata.rows import missing_marked
    try:
        with open(_os.path.join(out_dir, "manifest.json")) as f:
            doc = _json.load(f)
        srcs = sorted({e["src"] for c in (doc.get("categories") or {}).values()
                       for e in (c.get("entries") or []) if e.get("src")})
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return
    gone = [p for p in srcs if not _os.path.lexists(p)]
    if not gone or len(gone) < MISSING_SOURCES_MIN or len(gone) <= MISSING_SOURCES_SHARE * len(srcs):
        return
    with session_scope() as s:
        have = set()
        for i in range(0, len(gone), 500):
            chunk = gone[i:i + 500]
            have |= {p for (p,) in s.execute(text(
                "SELECT path FROM samples WHERE path IN ("
                + ", ".join(f":p{j}" for j in range(len(chunk))) + ")"),
                {f"p{j}": p for j, p in enumerate(chunk)})}
        if have and missing_marked(s):
            marked = {p for (p,) in s.execute(text(
                "SELECT m.path FROM missing_files m JOIN samples s ON s.id = m.sample_id "
                "AND s.path = m.path"))}
            have -= marked
    pickable = [p for p in gone if p in have]
    if len(pickable) < MISSING_SOURCES_MIN or len(pickable) <= MISSING_SOURCES_SHARE * len(srcs):
        return
    say = lambda m: console.print(m, style="red", markup=False, highlight=False, soft_wrap=True)
    say(f"{len(pickable):,} of the {len(srcs):,} files the master at {out_dir} was built from are "
        f"gone from disk, and the database still has them:")
    for p in pickable[:MISSING_LISTED]:
        say(f"  {p}")
    if len(pickable) > MISSING_LISTED:
        say(f"  ... and {len(pickable) - MISSING_LISTED:,} more")
    say("A drive that isn't mounted, or folders that were moved or renamed? Reconnect them and "
        "build again. If they were moved or deleted on purpose: `fourier tools scan` (with "
        "Sononym, after Sononym has rescanned: `fourier tools scan --prune`), or `fourier tools "
        f"db-stats --missing --prune`. Build stopped: nothing in {out_dir} changed.")
    raise SystemExit(1)


def _export_failures(res) -> dict:
    """{category: files it picked and couldn't export} from a build's results."""
    out = {}
    for r in res or ():
        summary = r[1] if len(r) > 1 else None
        n = int((summary or {}).get("failed") or 0) if isinstance(summary, dict) else 0
        if n:
            out[r[0]] = n
    return out


def _export_failure_message(short: dict, out_dir) -> str:
    n = sum(short.values())
    where = ", ".join(f"{c} {k}" for c, k in sorted(short.items()))
    msg = (f"{n:,} picked file{'' if n == 1 else 's'} couldn't be exported ({where}): a source "
           f"that's gone or can't be read (an unmounted drive, a moved or corrupt file), so the "
           f"build would be short. `fourier tools db-stats --missing` lists samples whose files "
           f"are gone (--prune removes them); `fourier tools scan` marks them for a library "
           f"Sononym doesn't index.")
    if out_dir:
        msg += f" The master at {out_dir} is unchanged."
    return msg


def _library_scale() -> dict | None:
    """{"factor": f, "samples": n} when the master scales down for this library (scale =
    "library" and fewer than LIBRARY_PER_MASTER usable samples a budgeted file; packs/scale.py),
    else None; either way the build log says how big the master will be."""
    from ..metadata.rows import usable_count
    from ..packs import scale as _sc
    with session_scope() as s:
        n = usable_count(s)
    f = _sc.factor(n)
    if f < 1:
        est = _sc.estimate(_sc.live_budgets(), n)
        console.print(_sc.describe(n, sum(est.values()), f=f) + ".", markup=False, highlight=False,
                      soft_wrap=True)
        return {"factor": f, "samples": n}
    console.print(_sc.describe(n, sum(_sc.live_budgets().values())) + ".", style="dim", markup=False,
                  highlight=False, soft_wrap=True)
    return None


def _built_any(build_dir) -> bool:
    """Whether an unfinished whole build has finished categories a --resume would keep."""
    from ..packs.progress import PROGRESS_DIR
    import glob as _glob
    import os as _os
    return bool(_glob.glob(_os.path.join(build_dir, PROGRESS_DIR, "*.json")))


def _remove_partial(build_dir) -> None:
    import shutil as _sh
    _sh.rmtree(build_dir, ignore_errors=True)


def _stop_partial(build_dir, out_dir, why=None):
    """Stop an unfinished whole build (exit 1). The master is untouched. <out>.next stays only
    when it holds finished categories a --resume would keep; else it's removed."""
    if why:
        console.print(why, style="red", markup=False, highlight=False)
    if _built_any(build_dir):
        console.print(f"Nothing in {out_dir} changed. The partial build is kept at {build_dir}; "
                      f"`fourier build --resume` continues it, or delete it.",
                      style="red", markup=False, highlight=False)
    else:
        _remove_partial(build_dir)
        console.print(f"Nothing in {out_dir} changed.", style="red", markup=False, highlight=False)
    raise SystemExit(1)


def _preset_note() -> None:
    """Say once when the config names no preset, so the default one is used."""
    ctx = click.get_current_context(silent=True)
    r = (ctx.find_root().obj or {}).get("resolved") if ctx is not None else None
    if r is not None and getattr(r, "preset_default", False):
        console.print(f"{r.config} names no preset: building with {r.preset} (set preset = \"...\" "
                      f"in it to choose; `fourier config show` lists the presets).",
                      style="yellow", markup=False, highlight=False)


def _preflight() -> None:
    """Stop with one line when the database can't make a build: nothing scanned, or no CLAP
    embeddings (they place every sample)."""
    from sqlalchemy import text
    from .enrich import CLAP_INSTALL, clap_extra_missing
    from ..metadata.rows import library_unscanned
    with session_scope() as s:
        n = s.execute(text("SELECT COUNT(*) FROM samples")).scalar() or 0
        if n and library_unscanned(s):      # only another config's library: not this one's
            n = 0
        emb = n and s.execute(text("SELECT 1 FROM sample_features WHERE clap_embedding IS NOT NULL "
                                   "LIMIT 1")).first()
    if not n:
        msg = ("nothing scanned yet: the library folders hold no audio files Fourier can read"
               if not _scanned_off() else
               "nothing scanned yet: run `fourier build` without --no-scan to scan the library")
    elif not emb and clap_extra_missing():
        msg = f"no CLAP embeddings: they need the CLAP model ({CLAP_INSTALL})"
    elif not emb:
        msg = ("no CLAP embeddings yet: none of the library's files could be analyzed"
               if not _scanned_off() else
               "no CLAP embeddings yet: run `fourier build` without --no-scan to analyze the library")
    else:
        return
    console.print(msg, style="red", markup=False, highlight=False)
    raise SystemExit(1)


def _scanned_off() -> bool:
    """Whether this build skipped the scan and the analysis (--no-scan)."""
    ctx = click.get_current_context(silent=True)
    return ctx is None or "Scan the library" not in (ctx.meta.get(STAGES_KEY) or ())


def _stage(name: str) -> None:
    """A build stage's header ("3/4  Build"), when this build runs that stage."""
    ctx = click.get_current_context(silent=True)
    stages = list(ctx.meta.get(STAGES_KEY) or ()) if ctx is not None else []
    if name in stages:
        console.print(f"\n[bold]{stages.index(name) + 1}/{len(stages)}  {name}[/bold]",
                      highlight=False)


def _warn_gone_keeps() -> None:
    """Say which Keep ratings point at files that are gone (moved or deleted): the build
    can't place them and passes over them, as verify does."""
    try:
        from ..packs.ratings import missing_keeps
        with session_scope() as s:
            gone = missing_keeps(s)
    except Exception:
        return
    if not gone:
        return
    names = ", ".join(_os_basename(p) for p, _c in gone[:5]) + (", ..." if len(gone) > 5 else "")
    console.print(f"{len(gone)} Keep rating{'' if len(gone) == 1 else 's'} point{'s' if len(gone) == 1 else ''} "
                  f"at files that are gone (moved or deleted): {names}. The build passes over them; "
                  f"`fourier review rate <name> clear` removes a rating.", style="yellow",
                  markup=False, highlight=False, soft_wrap=True)


def _os_basename(p):
    import os as _os
    return _os.path.basename(p)


def _scan_and_analyze(ctx, out_dir=None, allow_missing=False) -> None:
    """A build's first two stages: the library scan (as `fourier tools scan`), then every
    analysis step on what's new (as `fourier tools analyze`), with an estimate first when a
    lot is new. Without the CLAP model it stops before scanning: a build needs it."""
    from .enrich import clap_extra_missing, run_analysis
    from .ingest import run_scan
    from .setup import pending_analysis
    if clap_extra_missing():
        console.print("The CLAP model a build needs isn't installed: run `fourier setup` (or "
                      "build from the database as it is with --no-scan).",
                      style="red", markup=False, highlight=False)
        raise SystemExit(1)
    _stage("Scan the library")
    run_scan(allow_missing=True)
    from .ingest import UNAVAILABLE
    if UNAVAILABLE and not allow_missing:
        n = len(UNAVAILABLE)
        console.print(f"{n} linked folder{'' if n == 1 else 's'} in the library {'is' if n == 1 else 'are'} "
                      f"unavailable (above): a build without {'it' if n == 1 else 'them'} would leave "
                      f"out every sample there. Plug the drive in and build again, or build without "
                      f"them with --allow-missing. Build stopped: nothing in {out_dir} changed.",
                      style="red", markup=False, highlight=False, soft_wrap=True)
        raise SystemExit(1)
    _stage("Analyze what's new")
    pending = {k: v for k, v in pending_analysis().items() if v}
    # Fourier's own roots (pYIN) cover the tonal one-shots, not every new sample: counted apart
    own = pending.get("own", 0)
    new = max((v for k, v in pending.items() if k != "own"), default=0)
    if new >= ESTIMATE_FROM or own >= ESTIMATE_FROM:
        from ..timings import estimate_analyze, human
        secs, basis = estimate_analyze(pending, workers=4)
        roots = f"Fourier's own roots (pYIN) for {own:,} tonal one-shot{'s' if own != 1 else ''}"
        if new:
            console.print(f"{new:,} new samples to analyze, {human(secs)} (from {basis})"
                          + (f"; with {roots}." if own else "."), highlight=False)
        else:
            console.print(f"{roots} to read, {human(secs)} (from {basis}): once, then only new "
                          f"samples.", highlight=False)
    elif new:
        console.print(f"{new:,} new sample{'s' if new != 1 else ''} to analyze.", highlight=False)
    run_analysis(ctx)


def _stop_cloud_only(e, out_dir, build_dir=None):
    """Stop a build whose picks include cloud-only files that wouldn't download."""
    console.print(str(e), style="red", markup=False, highlight=False)
    console.print(f"Build stopped: nothing in {out_dir} changed"
                  + (f"; the partial build is in {build_dir}." if build_dir else "."),
                  style="red", markup=False, highlight=False)
    raise SystemExit(1)




@main.command("why", short_help="Show where a sample landed and which rule put it there.")
@click.argument("query", required=False)
@click.option("--master", "master_dir", default=None, help="Master to look in (default: your master).")
@click.option("--rules", is_flag=True, help="Print the routing precedence table and exit.")
@click.option("--limit", type=click.IntRange(1), default=None,
              help="Most samples to explain (default 5), or to list with --unrecognized (default 20).")
@click.option("--unrecognized", is_flag=True,
              help="List the samples no name rule recognized, and those recognized that no category took (without Sononym), and what to do about them.")
@click.option("--detail", is_flag=True,
              help="Every rule that matched, the classifiers' votes and labels, and the scores")
def why(query, master_dir, rules, limit, unrecognized, detail):
    """Where a sample landed and which routing rule put it there, or, for one the master left
    out, why the last build left it out (the per-vendor share, a near-duplicate of which
    file, a category's gate, the budget, or a category the library can't fill). QUERY is a
    full source path or a piece of a file name / library path.

    \b
      fourier why "Kick Round 03"
      fourier why --rules
      fourier why --unrecognized
    """
    import os as _os
    from ._app import read_only_config
    read_only_config()
    from ..packs.ratings import live_master_dir
    from ..packs.why import explain, format_why, precedence_table
    if unrecognized:
        _why_unrecognized(limit or 20, master_dir or _os.environ.get("FOURIER_CURATED_DIR")
                          or live_master_dir())
        return
    if rules or not query:
        for line in precedence_table():
            console.print(line, markup=False, highlight=False)
        return
    master_dir = master_dir or _os.environ.get("FOURIER_CURATED_DIR") or live_master_dir()
    want = limit or 5
    with session_scope() as session:
        ws = explain(session, query, master_dir=master_dir, limit=want + 1)
        if not ws and master_dir and _os.path.isfile(_os.path.join(master_dir, "manifest.json")):
            # a master path (KICKS/family/file.wav, as the report and Finder show it)
            from ..packs.ratings import find_rateable
            for t in find_rateable(master_dir, query)[:want + 1]:
                ws += explain(session, t["src"], master_dir=master_dir, limit=1)
    more = len(ws) > want
    ws = ws[:want]
    if not ws:
        from ..db.session import peek
        try:
            n = peek("SELECT COUNT(*) FROM samples") or 0
        except Exception:
            n = 0
        hint = ("" if n else ": nothing is scanned yet (`fourier build` scans the library first)")
        console.print(f"No sample matches {query!r}{hint}", style="red", markup=False,
                      highlight=False)
        raise SystemExit(1)
    from ..packs.why import format_short
    for w in ws:
        for line in (format_why(w) if detail else format_short(w)):
            console.print(line, markup=False, highlight=False, soft_wrap=True)
    if more:
        console.print(f"More samples match {query!r}: give more of the name, or --limit "
                      f"{want * 4} to see more.", markup=False, highlight=False)
    if not detail:
        console.print("`fourier why --detail` shows every rule, vote and score behind it.",
                      style="dim", markup=False, highlight=False)


def _clap_placed(master_dir) -> dict:
    """{sample id: category} the CLAP fallback or the sound model homed in a build of
    master_dir (its why logs)."""
    from ..packs.curate_config import CATEGORIES
    from ..packs.why_log import read
    out = {}
    if not master_dir:
        return out
    for cat in CATEGORIES:
        doc = read(master_dir, cat) or {}
        for sid in [*(doc.get("clap_homed") or ()), *(doc.get("sound_homed") or {})]:
            out[int(sid)] = cat
    return out


def _unrecognized_note(master_dir=None) -> list[str]:
    """Without Sononym, when a build left categories empty: how many samples no rule
    recognized and the CLAP fallback didn't place either, and what to do about it (none with
    Sononym, or when there are none)."""
    try:
        from ..metadata.providers import active
        from ..packs.curate import unrecognized
        with session_scope() as s:
            if not active(s).fallback:
                return []
            rows = unrecognized(s)
            twins = _master_twins(s, rows, master_dir)
    except Exception:
        return []
    placed = _clap_placed(master_dir)
    rows = [r for r in rows if r.id not in placed and r.id not in twins]
    return unrecognized_lines(rows) + (["`fourier why --unrecognized` lists them."] if rows else [])


def _master_twins(session, rows, master_dir) -> dict:
    """{sample id: category} of the rows not in the master whose bytes a master file's source
    has (a byte-identical copy: the master holds one of them)."""
    import json as _json
    import os as _os

    from sqlalchemy import text
    try:
        with open(_os.path.join(master_dir or "", "manifest.json")) as f:
            man = _json.load(f)
    except (OSError, ValueError, TypeError):
        return {}
    cat_of = {e.get("src"): c for c, cd in (man.get("categories") or {}).items()
              for e in (cd.get("entries") or ()) if e.get("src")}
    want = sorted({r.file_hash for r in rows if getattr(r, "file_hash", None) and r.path not in cat_of})
    if not want or not cat_of:
        return {}
    held = {}
    for i in range(0, len(want), 500):
        part = want[i:i + 500]
        params = {f"h{j}": h for j, h in enumerate(part)}
        for path, h in session.execute(text(
                f"SELECT path, file_hash FROM samples WHERE file_hash IN ({', '.join(':' + k for k in params)})"),
                params):
            if path in cat_of:
                held.setdefault(h, cat_of[path])
    return {r.id: held[r.file_hash] for r in rows
            if r.path not in cat_of and getattr(r, "file_hash", None) in held}


def unrecognized_lines(rows, limit=0, placed=None, twins=None) -> list[str]:
    """What `fourier why --unrecognized` and a build say about the samples no name rule
    recognized: how many (and how many of them the last build's sound model or CLAP fallback
    placed by sound, `placed`: {sample id: category}; how many are byte-identical copies of a file the
    master holds, `twins`: {sample id: category}, never listed as unplaced), the folders
    holding most of them, the first `limit` of them and the next step."""
    from ..packs.curate import top_folders
    from ..places import library_rel
    placed, twins = placed or {}, twins or {}
    rels = {r.id: r.rel_path or library_rel(r.path or "") for r in rows}
    if not rels:
        return []
    tops = ", ".join(f"{f} ({n:,})" for f, n in top_folders(list(rels.values())))
    n_placed = sum(1 for i in rels if i in placed and i not in twins)
    n_twins = sum(1 for i in rels if i in twins)
    out = [f"{len(rels):,} sample{'' if len(rels) == 1 else 's'} no rule recognized (top folders: {tops})"
           + (f"; the sound model or the CLAP fallback placed {n_placed:,} of them by sound" if n_placed else "")
           + (f"; {n_twins:,} {'is a byte-identical copy' if n_twins == 1 else 'are byte-identical copies'} "
              f"of files the master holds" if n_twins else "") + "."]
    for sid, x in sorted(rels.items(), key=lambda t: t[1])[:limit]:
        out.append(f"  {x}" + (f"  (a byte-identical copy of a file the master holds in {twins[sid]})"
                               if sid in twins else
                               f"  (placed by sound in {placed[sid]})" if sid in placed else ""))
    if limit and len(rels) > limit:
        out.append(f"  ... and {len(rels) - limit:,} more")
    out.append("Their names and folders hold no word Fourier knows for a category. The CLAP "
               "fallback places the ones that clearly sound like one category; for the rest, add "
               "your words with the `words` knob in fourier.toml (words = { KICKS = [\"bombo\"] }), "
               "rename the folders (Kicks, Snares, Pads, ...), or give a pack its home in a library "
               "overlay (sources.home).")
    return out


def _why_unrecognized(limit, master_dir=None):
    from ..metadata.providers import active
    from sqlalchemy import text as _sql

    from ..packs.curate import placed_nowhere, unrecognized
    with session_scope() as session:
        if not session.execute(_sql("SELECT COUNT(*) FROM samples")).scalar():
            console.print("No samples in the database yet: `fourier build` scans the library "
                          "first, then this lists what no rule recognized.", markup=False,
                          highlight=False)
            return
        if not active(session).fallback:
            console.print("Sononym classifies this library: every sample it analyzed has its "
                          "labels, so no name rule is needed.", markup=False, highlight=False)
            return
        rows = unrecognized(session)
        nowhere = placed_nowhere(session, master_dir)
        twins = _master_twins(session, rows, master_dir)
    say = lambda line: console.print(line, markup=False, highlight=False, soft_wrap=True)
    from ..metadata.rows import library_unscanned
    with session_scope() as session:
        unscanned = library_unscanned(session)
    if unscanned:
        say("This config's library isn't scanned yet (the database holds only other libraries' "
            "samples): `fourier tools scan`, or a build, reads it; then this lists what no rule "
            "recognized.")
        return
    if not rows:
        say("Every sample is recognized by a name rule (or is a phrase, a wave or an instrument by "
            "its name).")
    for line in unrecognized_lines(rows, limit, _clap_placed(master_dir), twins):
        say(line)
    for line in placed_nowhere_lines(nowhere, limit):
        say(line)


def placed_nowhere_lines(nowhere, limit=0) -> list[str]:
    """What `fourier why --unrecognized` says about the samples recognized by a name rule that
    the last build placed nowhere (curate.placed_nowhere): how many, by label, the first
    `limit`, and the next step. None (no build to read) says so."""
    from collections import Counter

    from ..places import library_rel
    if nowhere is None:
        return ["(Build first to see which recognized samples no category took.)"]
    if not nowhere:
        return []
    by = Counter(lab for _r, labs in nowhere for lab in labs)
    out = ["", f"{len(nowhere):,} recognized sample{'' if len(nowhere) == 1 else 's'} the last build placed "
               f"nowhere (labels: {', '.join(f'{k} {n:,}' for k, n in by.most_common(6))}):"]
    rows = sorted(((r.rel_path or library_rel(r.path or ""), labs) for r, labs in nowhere))
    for rel, labs in rows[:limit]:
        out.append(f"  {rel}  (recognized as {', '.join(labs)}, but no category took it)")
    if limit and len(rows) > limit:
        out.append(f"  ... and {len(rows) - limit:,} more")
    out.append("A rule kept them out of the categories their labels name (keys out of SYNTH, a "
               "drum word on a loop), or that category is off. `fourier why \"<name>\"` says which "
               "rules each one met; a rename or the `words` knob can point it at another category.")
    return out


@main.command("diff", short_help="Show what changed between two builds.")
@click.argument("old", required=False)
@click.argument("new", required=False)
def diff(old, new):
    """What changed between two builds: folders renamed / new / gone, files added,
    removed or moved. OLD and NEW are master folders, manifest files or archived build
    stamps (~/.fourier/builds); defaults: the previous archived build vs your current master.

    \b
      fourier diff
      fourier diff ~/Music/FourierCurated.prev ~/Music/FourierCurated
    """
    import os as _os
    from ..packs.builddiff import archived, diff_manifests, format_diff, load_manifest
    from ..packs.ratings import live_master_dir
    new = new or _os.environ.get("FOURIER_CURATED_DIR") or live_master_dir()
    if not old:
        arch = archived()
        if len(arch) < 2:
            console.print("[red]need OLD: fewer than two archived builds[/red]")
            raise SystemExit(1)
        old = str(arch[-2])
    console.print(format_diff(diff_manifests(load_manifest(old), load_manifest(new))),
                  markup=False, highlight=False)




# ---------------------------------------------------------------------------
# fourier tools audit: coverage blind spots, the master vs the whole library
# ---------------------------------------------------------------------------
@tools.command("audit", short_help="Find what the master leaves out of the library.")
@click.option("--from", "master_dir", default=None,
              help="Master to audit (default: $FOURIER_CURATED_DIR)")
@click.option("--out", "out_path", default=None,
              help="Write the markdown report to this path")
@click.option("--far", default=0.55, type=float, show_default=True,
              help="Nearest-curated-cosine threshold below which a sample counts as poorly covered")
def audit(master_dir, out_path, far):
    """Audit the master against the whole library for blind spots: kinds of sound the
    library has plenty of and the master barely covers.

    Three lenses: the classifier's categories the master leaves uncovered, CLAP
    nearest-neighbor coverage (regions far from every pick, clustered and named by a local
    LLM when Ollama runs), and packs big in the library but thin in the master.

    \b
      fourier tools audit
      fourier tools audit --out ~/audit.md
    """
    from ..packs.audit import run_audit

    from .releases import need_master
    master_dir = need_master(master_dir)
    with session_scope() as session:
        run_audit(session, master_dir, far=far, out_path=out_path, log=lambda m: console.print(m))


def _validate(master_dir, no_db):
    """The quick invariants (verify --quick): budgets, ceilings, vendor cap, hygiene, names."""
    from ..packs.validate import validate_master
    master_dir = master_dir or _master_dir()
    _need_master(master_dir)
    if no_db:
        ok, _res = validate_master(master_dir, session=None, log=lambda m: console.print(m))
    else:
        with session_scope() as session:
            ok, _res = validate_master(master_dir, session=session, log=lambda m: console.print(m))
    if not ok:
        raise SystemExit(1)


def _need_master(master_dir) -> None:
    """Stop with one line when there's no built master to check."""
    import os as _os
    if not master_dir or not _os.path.exists(_os.path.join(master_dir, "manifest.json")):
        console.print(f"no master at {master_dir} yet: run `fourier build` first",
                      style="red", markup=False, highlight=False)
        raise SystemExit(1)


def _run_verify(master_dir, renders=(), audio=True, use_db=True, keep=None, quiet=False):
    """Run verify; print every line (or, quiet, only WARN/FAIL and a summary).
    Returns True when nothing FAILed."""
    from ..packs.verify import render_and_verify, verify_master
    lines = []
    say = (lambda m: lines.append(m)) if quiet else (lambda m: console.print(m, markup=False, highlight=False))
    if use_db:
        with session_scope() as session:
            ok, res = verify_master(master_dir, session=session, audio=audio, log=say)
    else:
        ok, res = verify_master(master_dir, session=None, audio=audio, log=say)
    if renders:
        ok2, res2 = render_and_verify(master_dir, list(renders), keep_dir=keep, log=say)
        ok, res = ok and ok2, res + res2
    if quiet:
        for m in lines:
            if not m.startswith("PASS"):
                console.print(m, markup=False, highlight=False)
    n = Counter(r.level for r in res)
    console.print(f"{'[green]verify: PASS' if ok else '[red]verify: FAIL'}[/] -- "
                  f"{n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail")
    return ok


@main.command("verify", short_help="Check the master against every curation rule.")
@click.option("--from", "master_dir", default=None,
              help="Master to verify (default: your master)")
@click.option("--render", "renders", multiple=True,
              help="Also render for this device into a scratch dir and check it (repeatable)")
@click.option("--keep-renders", "keep", default=None, help="Keep the scratch renders in this dir")
@click.option("--no-audio", is_flag=True, default=False,
              help="Skip the checks that read every file's audio")
@click.option("--no-db", is_flag=True, default=False,
              help="Skip DB-backed checks (sample chains, twins, hygiene)")
@click.option("--quick", is_flag=True, default=False,
              help="Only the quick invariants (budgets, ceilings, vendor cap, hygiene, names)")
def verify(master_dir, renders, keep, no_audio, no_db, quick):
    """Check the master against every curation rule (it runs after each build and before
    publish), and with --render in scratch device renders: manifest against disk, budgets,
    names, ratings, routing, sample chains, length caps, identical audio, DC and more. Exits
    non-zero on any FAIL.

    \b
      fourier verify
      fourier verify --render m8_tracker --render digitakt_2
      fourier verify --quick
    """
    if quick:
        _validate(master_dir, no_db)
        return
    from ..packs.ratings import live_master_dir
    master_dir = master_dir or live_master_dir() or _master_dir()
    _need_master(master_dir)
    if not _run_verify(master_dir, renders, audio=not no_audio, use_db=not no_db, keep=keep):
        raise SystemExit(1)


def _post_build_ratings(master_dir):
    """After a build: rebuild the sets (KITS/, SLICE/), re-apply stored ratings as Live tags,
    print the scorecard, verify. Returns verify's verdict (True when nothing FAILed, or when
    verify couldn't run)."""
    try:
        from ..packs.sets import SETS_ON, build_sets
        if SETS_ON:
            build_sets(master_dir, log=lambda m: console.print(m))
        else:
            console.print("sets: off (sets = \"off\"): no 00_KITS or 00_SLICE", markup=False,
                          highlight=False)
    except Exception as e:
        console.print(f"[yellow]sets skipped: {e}[/yellow]")
    try:
        from ..packs.ratings import apply_tags
        apply_tags(master_dir, log=lambda m: console.print(m))
    except Exception as e:
        console.print(f"[yellow]ratings write-back skipped: {e}[/yellow]")
    _print_scorecard(master_dir)
    _stage("Verify")
    try:
        return _run_verify(master_dir, quiet=True)
    except Exception as e:
        # a verify that couldn't run hasn't passed: a whole build then stays in .next
        console.print(f"[red]verify couldn't run: {e}[/red]")
        return False


def _check_build_target(out_dir):
    """The master a build may write: clear of the library, releases and Fourier's own places,
    new, empty or a Fourier master holding only files Fourier made (a build replaces its
    contents; `fourier build CATEGORY` rewrites <out>/CATEGORY outright). <out>.next and
    <out>.prev sit beside the path as given (a symlinked master keeps them where they were)
    and must be clear too. Returns the path to build into; exits 1 on anything unsafe."""
    import os as _os
    from ..safety import UnsafePath, check_clear, check_master_contents, check_master_dir
    given = _os.path.abspath(_os.path.expanduser(str(out_dir))).rstrip("/\\") or str(out_dir)
    try:
        real = check_master_dir(given)
        for suffix in (".next", ".prev"):
            check_clear(given + suffix, f"the master's {suffix} folder")
        if real.is_dir() and _os.path.exists(_os.path.join(real, "manifest.json")):
            check_master_contents(real)
    except UnsafePath as e:
        console.print(str(e), style="red", markup=False, highlight=False)
        raise SystemExit(1) from None
    return given


def _check_own(d):
    """Refuse to remove a .next / .prev folder that isn't a Fourier build (a user's folder that
    happens to carry the name): a build has a Fourier manifest or its .progress folder."""
    import json as _json
    import os as _os
    if not _os.path.isdir(d):
        console.print(f"[red]{d} exists and isn't a folder: move it aside first[/red]")
        raise SystemExit(1)
    ok = not _os.listdir(d) or _os.path.isdir(_os.path.join(d, ".progress"))
    if not ok:
        try:
            with open(_os.path.join(d, "manifest.json")) as f:
                ok = "fourier_manifest" in _json.load(f)
        except (OSError, ValueError, TypeError):
            ok = False
    if not ok:
        console.print(f"[red]{d} exists and isn't a Fourier build: move it aside first[/red]")
        raise SystemExit(1)


def _kept_note(prev, out_dir) -> str:
    """The note after "New master in place at <out>": where the previous master is kept and
    that CHANGELOG.md says what changed, each only when it's there (a first build has neither)."""
    import os as _os
    parts = []
    if prev and _os.path.isdir(prev):
        parts.append(f"previous kept at {prev}")
    if _os.path.exists(_os.path.join(out_dir, "CHANGELOG.md")):
        parts.append("CHANGELOG.md says what changed")
    return f" ({'; '.join(parts)})" if parts else ""


def _swap_in(build_dir, out_dir):
    """Bring a finished build into place, touching only what changed. The old master is
    snapshotted as out_dir.prev (hardlinks, replacing an older .prev); then a file identical
    in the new build stays as it is (same inode, same mtime), changed and new files are
    linked in (each replaced atomically), and files the build dropped are removed. Live's
    indexer and Spotlight then see only the changes; swapping in a whole new folder made
    Live re-index every file (re-adding the folder doesn't help)."""
    import filecmp as _fc
    import os as _os
    import shutil as _sh
    prev = out_dir.rstrip("/") + ".prev"
    if not _os.path.exists(out_dir):
        _os.rename(build_dir, out_dir)
        return prev
    if _os.path.exists(_os.path.join(out_dir, "manifest.json")):
        # files the user added since the build started would be deleted below
        from ..safety import UnsafePath, check_master_contents
        try:
            check_master_contents(out_dir)
        except UnsafePath as e:
            console.print(f"{e}\nThe new build stays in {build_dir}; the master is unchanged.",
                          style="red", markup=False, highlight=False)
            raise SystemExit(1) from None
    if _os.path.exists(prev):
        _check_own(prev)
        _sh.rmtree(prev)
    _sh.copytree(out_dir, prev, copy_function=_link_or_copy)
    new_files, n = set(), dict(same=0, changed=0, added=0, removed=0)
    for root, _dirs, files in _os.walk(build_dir):
        rel_root = _os.path.relpath(root, build_dir)
        odir = _os.path.join(out_dir, rel_root)
        _os.makedirs(odir, exist_ok=True)
        # names on disk by lowercase: APFS (and FAT32) fold case, so "kick.wav" replacing
        # "Kick.wav" would otherwise match the old file and then be deleted as dropped
        actual = {x.lower(): x for x in _os.listdir(odir)}
        for f in files:
            rel = _os.path.normpath(_os.path.join(rel_root, f))
            new_files.add(rel)
            src, dst = _os.path.join(build_dir, rel), _os.path.join(out_dir, rel)
            if actual.get(f.lower(), f) != f:
                _os.remove(_os.path.join(odir, actual[f.lower()]))    # same name, other case
                actual[f.lower()] = f
                n["changed"] += 1
                _link_or_copy(src, dst)
                continue
            had = _os.path.exists(dst)
            if had and (_os.path.samefile(src, dst) or (
                    _os.path.getsize(src) == _os.path.getsize(dst) and _fc.cmp(src, dst, shallow=False))):
                n["same"] += 1
                continue
            n["changed" if had else "added"] += 1
            tmp = _os.path.join(_os.path.dirname(dst), f".{f}.fourier-swap")
            _link_or_copy(src, tmp)
            _os.replace(tmp, dst)
    for root, dirs, files in _os.walk(out_dir, topdown=False):
        for f in files:
            rel = _os.path.normpath(_os.path.relpath(_os.path.join(root, f), out_dir))
            if rel not in new_files:
                _os.remove(_os.path.join(root, f))
                n["removed"] += 1
        if root != out_dir and not _os.listdir(root):
            _os.rmdir(root)
    _sh.rmtree(build_dir)
    console.print(f"synced into {out_dir}: {n['same']:,} unchanged, {n['changed']:,} changed, "
                  f"{n['added']:,} added, {n['removed']:,} removed")
    return prev


def _link_or_copy(src, dst):
    import os as _os
    import shutil as _sh
    try:
        _os.link(src, dst)
    except OSError:
        _sh.copy2(src, dst)
    return dst


def _print_scorecard(master_dir, always=False):
    """Score a built master against the Ableton ratings store; never blocks a build."""
    import os as _os
    try:
        from ..packs.ratings import format_scorecard, save_scorecard, scorecard
        if not master_dir or not _os.path.exists(_os.path.join(master_dir, "manifest.json")):
            if always:
                console.print(f"[red]no manifest.json at {master_dir}[/red]")
            return
        sc = scorecard(master_dir)
        prev = save_scorecard(sc)
        t = sc["totals"]
        if always or t["rated"] or t["keeps_lost"] or t["misfiled_moved"]:
            for line in format_scorecard(sc, prev):
                console.print(line, markup=False, highlight=False)
    except Exception as e:
        console.print(f"[yellow]scorecard skipped: {e}[/yellow]")


def _library_files() -> int:
    """The audio files under the library folders (setup's count, capped), or 0."""
    try:
        from ..places import library_roots
        from .setup import COUNT_CAP, survey
        return survey(library_roots(), COUNT_CAP)[0]
    except Exception:
        return 0


def _dry_run(cats, jobs=1):
    from ..db.session import db_exists
    from ..packs.curate_config import CATEGORY_ORDER, LIBRARY_PER_MASTER
    from ..metadata.rows import usable_count
    from ..packs.dryrun import home_counts, plan
    cats = sorted(cats, key=lambda c: CATEGORY_ORDER.index(c) if c in CATEGORY_ORDER else 99)
    homes, n = None, 0
    if db_exists():                   # a dry run creates nothing, a database included
        from ..metadata.rows import outside_note
        with session_scope() as session:
            n = usable_count(session)
            other = outside_note(session)
            if other:
                console.print(other + ".", markup=False, highlight=False, soft_wrap=True)
            if n:
                homes = home_counts(session, log=lambda m: console.print(m, markup=False, highlight=False))
    from ..packs.scale import describe
    # before any analysis: size the master from the library's audio files (as doctor and
    # setup do), or say it can't be sized yet
    files = 0 if n else _library_files()
    p = plan(cats, homes, samples=n if n else (files or None))
    if not n and files:
        p.pop("picks", None)                          # what it fills: once it's analyzed
    scaled = p.get("scale", 1) < 1
    if not n and not files:
        console.print("Can't size the master before the first scan: no audio files found in the "
                      "library folders yet (the budgets below are the style's).", highlight=False)
    t = Table(title="fourier build --dry-run", box=box.SIMPLE)
    for col, j in (("Category", "left"), ("Budget", "right"), ("MB", "right"),
                   ("Home samples", "right"), ("", "left")):
        t.add_column(col, justify=j)
    for r in p["rows"]:
        h = r["homes"]
        if scaled:
            short = f"of the style's {r['style_budget']:,}" + ("" if r["budget"] else "; left empty")
        else:
            short = "may run short" if h is not None and h < r["budget"] else ""
        t.add_row(r["category"], f"{r['budget']:,}", f"{r['mb']:,.0f}", "-" if h is None else f"{h:,}", short)
    console.print(t)
    if scaled:
        line = describe(n or files, p["files"], p["categories"], p["scale"])
        if not n:
            line = line.replace("usable samples", "audio files", 1) + " (from the files in the library " \
                "folders; the first build's analysis says how many are usable)"
        console.print(line + ".", highlight=False)
        size = (f"{p['total_mb']:,.0f} MB" if p["total_mb"] < 1000 else f"{p['total_mb'] / 1000:,.1f} GB")
        console.print(f"Budgets, scaled to this library: {p['files']:,} files, about {size} with the kit "
                      f"and slice sets (+{p['sets_share']:.0%}); "
                      f"the style's own add up to {sum(r['style_budget'] for r in p['rows']):,} "
                      f"(scale = \"off\" builds those).", highlight=False)
    else:
        console.print(f"Budgets: {p['files']:,} files, about {p['total_mb'] / 1000:,.1f} GB with the kit and "
                      f"slice sets (+{p['sets_share']:.0%}).", highlight=False)
    if n and not scaled:
        console.print(describe(n, p["files"]) + ".", highlight=False)
        console.print(f"This library: up to about {round(p['picks']):,} files, {p['picks_mb'] / 1000:,.1f} GB, "
                      f"can be filled from its {n:,} analyzed samples (each category's budget or its "
                      f"home samples, whichever is fewer; none where they're under the category's "
                      f"minimum; a category's filters take more out).", highlight=False)
    if p["limit"]:
        mb, dev = p["limit"]
        fits = "fits" if p["total_mb"] <= mb else "doesn't fit: size = \"auto\" scales it down"
        console.print(f"Storage: {mb / 1000:,.1f} GB is the share of {dev}'s storage a master may use; "
                      f"the budgets' total {fits}.", highlight=False)
    if scaled:
        console.print("Each category's budget follows its pool: all of a small one, about one in "
                      f"{LIBRARY_PER_MASTER} of a large one (or its scaled share of the style's budget, "
                      "if more).", highlight=False)
    else:
        console.print("A category short of its budget hands the rest to the others (size = \"auto\")."
                      if p["surplus"] else "A category short of its budget stays short (size = \"fixed\").",
                      highlight=False)
    if n:
        from ..timings import estimate_build, human
        secs, basis = estimate_build({r["category"]: r["picks"] for r in p["rows"]}, jobs, samples=n)
        console.print(f"Time: {human(secs)} with {jobs} worker{'s' if jobs != 1 else ''} (-j), from {basis}.",
                      highlight=False)
    else:
        console.print("Time and what this library fills: can't estimate before the library is "
                      "analyzed (a build without --dry-run does it first).", highlight=False)
    if n:
        with session_scope() as session:
            from ..metadata.providers import active
            if active(session).fallback:
                from ..packs.curate import unrecognized
                un = unrecognized(session)
                if un:
                    for line in unrecognized_lines(un)[:1]:
                        console.print(line + " The CLAP fallback places the ones that clearly sound like "
                                      "one category; `fourier why --unrecognized` lists them.",
                                      markup=False, highlight=False, soft_wrap=True)
    console.print("Home samples: the samples that call a category home, before its filters (an "
                  "instrument category: the candidates its own rules keep; wave categories have "
                  "none). Nothing was written.", highlight=False)
