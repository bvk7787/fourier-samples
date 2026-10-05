"""fourier review rate, import, queue, ratings, score and misfiles: the listening loop."""
from __future__ import annotations

import sys

import click

from ._app import _table_cols, console, log, main  # noqa: F401
from .build import _print_scorecard


@main.group("review", short_help="The listening loop: rate files, import ratings, queue and score.")
def review():
    """The listening loop: rate the master's files Keep, Drop or Misfiled, and the next build
    follows the ratings (a Keep stays, a Drop goes, a Misfiled moves). Rate with `review
    rate` or `review import` (a CSV file), or in Ableton Live's browser with tags
    (Fourier|Keep / Drop / Misfiled / Move-<CAT>) that `review ratings` harvests. All of
    them write the same ratings store ($FOURIER_HOME/ratings.json).

    \b
    Without Live: rate, import, queue, score, misfiles.
    Needs Live:   ratings (it harvests Live's tags; queue, score and builds harvest them
                  too when Live has tagged the master).

    \b
      fourier review rate "Kick 01" keep
      fourier review rate "Snare Tight 03" misfiled --to CLAPS
      fourier review import ratings.csv   # columns: path (or name), rating, category
      fourier review queue                # the next files to listen to, in <master>/_REVIEW
      fourier review score                # a build against the ratings
      fourier review misfiles --dry-run
    """


def _master_or_stop(master_dir):
    import os as _os
    from ..packs.ratings import live_master_dir
    master_dir = master_dir or live_master_dir()
    if not master_dir or not _os.path.exists(_os.path.join(master_dir, "manifest.json")):
        console.print(f"no master with a manifest{' at ' + master_dir if master_dir else ''}: "
                      f"run `fourier build` first, or give --from", style="red",
                      markup=False, highlight=False)
        raise SystemExit(1)
    return master_dir


def _interactive() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


AFTER = {"keep": "the next build keeps it in {cat}",
         "drop": "the next build leaves it out",
         "misfiled": "the next build moves it out of {cat}"}


@review.command("rate", short_help="Rate one file Keep, Drop or Misfiled, without Live.")
@click.argument("query")
@click.argument("verdict", type=click.Choice(["keep", "drop", "misfiled", "clear"], case_sensitive=False))
@click.option("--to", "move_to", default=None, metavar="CATEGORY",
              help="With misfiled: the category it belongs in (the next build puts it there)")
@click.option("--from", "master_dir", default=None,
              help="Master the file is in (default: your master)")
def rate(query, verdict, move_to, master_dir):
    """Rate one of the master's files: keep (it stays), drop (the next build leaves it out) or
    misfiled (the next build moves it to another category; --to names which); clear removes
    a rating given here or imported (also one whose file is gone from the master or the
    library). QUERY is what `fourier why` takes: a source path, a master path
    (KICKS/family/file.wav) or a piece of a file name or library path. When it matches
    several files, you pick one.

    \b
      fourier review rate "Kick 01" keep
      fourier review rate KICKS/punchy/Kick_01.wav drop
      fourier review rate "Snare Tight 03" misfiled --to CLAPS
      fourier review rate "Kick 01" clear
    """
    from ..packs.ratings import apply_tags, find_rateable, harvest, set_ratings, uses_live
    say = lambda m: console.print(m, markup=False, highlight=False)
    master_dir = _master_or_stop(master_dir)
    if verdict.lower() == "clear":
        if move_to:
            say("--to goes with misfiled, not clear")
            raise SystemExit(2)
        _clear(query, master_dir, say)
        return
    hits = find_rateable(master_dir, query)
    if not hits:
        say(f"no file in the master matches {query!r} (`fourier why {query!r}` says where it went)")
        raise SystemExit(1)
    if len(hits) > 1:
        say(f"{query!r} matches {len(hits)} files:")
        for i, t in enumerate(hits[:30], 1):
            say(f"  {i:3d}  {t['path']}   ({t['short']})")
        if len(hits) > 30:
            say(f"  ... and {len(hits) - 30} more")
        if not _interactive():
            say("Give more of the name, or the master path (CATEGORY/family/file.wav).")
            raise SystemExit(2)
        pick = click.prompt("Which one (number)", type=click.IntRange(1, min(len(hits), 30)))
        hits = [hits[pick - 1]]
    t = hits[0]
    live = uses_live(master_dir)
    if live:             # Live's tags first, so the stored rating and the tag agree after
        harvest(master_dir, log=lambda m: None)
    try:
        set_ratings(master_dir, [(t, verdict, move_to)])
    except ValueError as e:
        say(str(e))
        raise SystemExit(2) from None
    if live:
        apply_tags(master_dir, log=lambda m: None)
    v = verdict.lower()
    what = AFTER[v].format(cat=t["category"])
    if v == "misfiled" and move_to:
        from ..packs.ratings import _category_named
        what = f"the next build moves it to {_category_named(move_to)}"
    say(f"{t['path']}: {v.capitalize()} ({what})")


# where a stored rating came from: Live's browser (its tag wins at every harvest) or here
FROM_LIVE = ("tag", "favorite")


def _clear(query, master_dir, say) -> None:
    """`review rate QUERY clear`: remove the stored rating of the one file QUERY names (in the
    master, or a stored rating whose file is gone from it). A rating from a Live tag is
    changed in Live: the tag would bring it back."""
    from ..packs.ratings import (_short_src, clear_ratings, find_rateable, load_store,
                                 stored_matching, uses_live)
    R = load_store()["ratings"]
    found = {t["src"]: R[t["src"]] for t in find_rateable(master_dir, query) if t["src"] in R}
    for src, r in stored_matching(query):
        found.setdefault(src, r)
    if not found:
        say(f"no stored rating matches {query!r} (`fourier review score` lists the rated files)")
        raise SystemExit(1)
    srcs = sorted(found)
    if len(srcs) > 1:
        say(f"{query!r} matches {len(srcs)} rated files:")
        for i, src in enumerate(srcs[:30], 1):
            say(f"  {i:3d}  {_short_src(src)}  ({found[src]['verdict']})")
        if not _interactive():
            say("Give more of the name, or the file's path.")
            raise SystemExit(2)
        pick = click.prompt("Which one (number)", type=click.IntRange(1, min(len(srcs), 30)))
        srcs = [srcs[pick - 1]]
    src = srcs[0]
    r = found[src]
    in_master = any(t["src"] == src for t in find_rateable(master_dir, src))
    if r.get("source") in FROM_LIVE and in_master and uses_live(master_dir):
        say(f"{_short_src(src)} is rated {r['verdict'].capitalize()} by its tag in Live's browser: "
            f"remove or change the Fourier tag there (the next build, or `fourier review "
            f"ratings`, reads it).")
        raise SystemExit(2)
    clear_ratings(master_dir, {src})
    say(f"{_short_src(src)}: {r['verdict'].capitalize()} rating cleared")


@review.command("import", short_help="Import ratings from a CSV file, without Live.")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--from", "master_dir", default=None,
              help="Master the files are in (default: your master)")
@click.option("--dry-run", is_flag=True, default=False, help="Match the rows and report, store nothing")
def import_ratings(csv_path, master_dir, dry_run):
    """Import ratings from a CSV file with a header row: `path` (or `name`: anything `review
    rate` takes), `rating` (keep, drop or misfiled) and an optional `category` (for
    misfiled: the category it belongs in). Rows that match no file, or several, are listed
    and skipped (exit 1); the rest are stored as `review rate` stores them.

    \b
      path,rating,category
      KICKS/punchy/Kick_01.wav,keep,
      Snare Tight 03,misfiled,CLAPS
    """
    from ..packs.ratings import apply_tags, harvest, read_ratings_csv, set_ratings, uses_live
    say = lambda m: console.print(m, markup=False, highlight=False)
    master_dir = _master_or_stop(master_dir)
    try:
        rated, skipped = read_ratings_csv(csv_path, master_dir)
    except (ValueError, OSError) as e:
        say(str(e))
        raise SystemExit(2) from None
    for line, q, why in skipped:
        say(f"  line {line}: {q!r}: {why}")
    if not dry_run and rated:
        live = uses_live(master_dir)
        if live:
            harvest(master_dir, log=lambda m: None)
        set_ratings(master_dir, rated, source="csv")
        if live:
            apply_tags(master_dir, log=lambda m: None)
    counts = {v: sum(1 for _t, x, _m in rated if x == v) for v in ("keep", "drop", "misfiled")}
    say(f"{'would rate' if dry_run else 'rated'} {len(rated)} files ({counts['keep']} keep, "
        f"{counts['drop']} drop, {counts['misfiled']} misfiled); {len(skipped)} rows skipped")
    if skipped:
        raise SystemExit(1)

@review.command("ratings", short_help="Harvest the rating tags from Live's browser (needs Live).")
@click.option("--from", "master_dir", default=None,
              help="Master to harvest (default: your master)")
@click.option("--apply", "do_apply", is_flag=True, default=False,
              help="Also write stored ratings back into the master as Live tags")
def ratings(master_dir, do_apply):
    """Harvest Fourier|Keep / Drop / Misfiled tags (Ableton browser) into the ratings store.
    Needs Ableton Live: without it, rate with `review rate` or `review import`.

    A Favorite (red Collection) on an untagged file counts as Keep; --apply then writes
    the Keep tag into Live right away."""
    from ..packs.ratings import apply_tags, harvest, live_master_dir
    master_dir = master_dir or live_master_dir()
    harvest(master_dir, log=lambda m: console.print(m))
    if do_apply:
        apply_tags(master_dir, log=lambda m: console.print(m))


@review.command("score", short_help="Score a build against the ratings.")
@click.option("--from", "master_dir", default=None,
              help="Built master to score (default: your master); e.g. a temp build before swapping it in")
def score(master_dir):
    """Score a built master against your ratings (harvests Live's tags from your master
    first, when it has any)."""
    from ..packs.ratings import harvest, live_master_dir, uses_live
    live = live_master_dir()
    master_dir = _master_or_stop(master_dir)
    if live and uses_live(live):
        harvest(live, log=lambda m: console.print(m))
    _print_scorecard(master_dir, always=True)


@review.command("queue", short_help="Queue the files whose rating teaches the most.")
@click.option("--n", "n", default=40, show_default=True, type=int, help="Files in the queue")
@click.option("--from", "master_dir", default=None,
              help="Master (default: your master)")
def queue(n, master_dir):
    """Build a review queue in <master>/_REVIEW: the samples whose rating teaches Fourier Samples most.

    Saves the previous queue's tags first (harvest + write-back into the master), then
    picks suspected misfiles, shaky placements, neighbors of your Drops, and random spot
    checks. Tag them in Live like anything else, or rate them with `review rate` (a queue
    file's name works as its QUERY)."""
    from ..packs.ratings import apply_tags, harvest, uses_live
    from ..packs.review import build_queue
    master_dir = _master_or_stop(master_dir)
    say = lambda m: console.print(m, markup=False, highlight=False)
    if uses_live(master_dir):                 # the queue's earlier tags, kept before it's replaced
        harvest(master_dir, log=say)
        apply_tags(master_dir, log=say)
    build_queue(master_dir, n=n, log=say)


@review.command("misfiles", short_help="Mark the misfile detector's confident calls.")
@click.option("--margin", default=None, type=float,
              help="Min margin (other-folder minus own-folder similarity); default 0.4")
@click.option("--from", "master_dir", default=None, help="Master (default: your master)")
@click.option("--dry-run", is_flag=True, default=False, help="List detections, store nothing")
@click.option("--clear", is_flag=True, default=False, help="Remove all automatic detections")
def misfiles(margin, master_dir, dry_run, clear):
    """Treat the misfile detector's most confident calls as Misfiled (next build re-homes them).

    Stored apart from your ratings; any rating you give a file wins."""
    from ..packs.ratings import live_master_dir
    from ..packs.review import AUTO_MISFILE_MARGIN, auto_misfile
    say = lambda m: console.print(m, markup=False, highlight=False)
    if not clear:
        master_dir = _master_or_stop(master_dir)
    auto_misfile(master_dir or live_master_dir(),
                 margin=AUTO_MISFILE_MARGIN if margin is None else margin,
                 dry_run=dry_run, clear=clear, log=say)
