"""fourier tools db-stats: what the library database holds, and its checks."""
from __future__ import annotations

import json
from collections import defaultdict

import click
from rich.table import Table

from ..db.models import Sample, SononymMeta
from ..db.session import get_session, session_scope
from ._app import _table_cols, console, log, main, tools  # noqa: F401

# what each of Sononym's categories is good for, beside its count
SONONYM_HINTS = {
    "Perc Kicks": "kicks",
    "Perc Snares": "snares",
    "Perc Claps": "claps / snares",
    "Perc Snips & Snaps": "short transients",
    "Perc Hats & Shakers": "hats (open or closed)",
    "Perc Cymbal Crashes": "open hats / cymbals",
    "Perc Toms": "toms / fills",
    "Perc Bongos & Congas": "toms / hand perc",
    "Perc Metal Hits": "industrial / metallic perc",
    "Perc Wood Hits": "wooden / ethnic perc",
    "Tone Bass & LowKeys": "bass one-shots",
    "Tone Leads & MidHiKeys": "melodic leads",
    "Tone Pads & Textures": "pads / atmosphere",
    "Tone Stabs & Orch. Hits": "stabs / cinematic",
    "XFX Sweeps & Lasers": "sweeps / transitions",
    "XFX Whooshes & Whips": "impacts / movement",
    "XFX Noise & Distortion": "noise / texture",
    "XFX Explosions & Shots": "impacts",
}


@tools.command("db-stats", short_help="Show what the database holds; check files and metadata.")
@click.option("--missing", is_flag=True,
              help="Also list samples whose files are no longer on disk")
@click.option("--prune", is_flag=True,
              help="With --missing: remove those samples from the database")
@click.option("--metadata", is_flag=True,
              help="Also check each provider's labels and descriptors against its source "
                   "(exits 1 on any difference)")
@click.option("--rebuild", "do_rebuild", is_flag=True,
              help="With --metadata: rebuild every provider's labels and descriptors first")
@click.option("--shadow", is_flag=True,
              help="With --metadata: also score the path and audio providers against Sononym")
@click.option("--disagreements", is_flag=True,
              help="Also list samples where Sononym and Fourier's own analysis disagree "
                   "(one-shot or loop, tempo beyond an octave, pitched, root note, and the sound "
                   "model's category): a report, nothing changes")
@click.option("--examples", default=10, show_default=True,
              help="With --metadata or --disagreements: differences to print (of each kind)")
def db_stats(missing, prune, metadata, do_rebuild, shadow, disagreements, examples):
    """Show what the library database holds: samples, favorites, and with Sononym's data its
    categories (count, average length, brightness and noisiness; the names `fourier search
    --categories` takes). --missing lists samples whose files are gone (--prune removes
    them); --metadata checks the classifiers' labels against their sources;
    --disagreements lists where Sononym's readings and Fourier's own differ, a place to look
    for misfiled samples.

    \b
      fourier tools db-stats
      fourier tools db-stats --missing --prune
      fourier tools db-stats --metadata --shadow
      fourier tools db-stats --disagreements --examples 30
    """
    if prune and not missing:
        raise click.UsageError("--prune goes with --missing")
    if (do_rebuild or shadow) and not metadata:
        raise click.UsageError("--rebuild and --shadow go with --metadata")
    _stats()
    if missing:
        _missing(prune)
    if disagreements:
        _disagreements(examples)
    if metadata and not _metadata(do_rebuild, examples, shadow):
        raise SystemExit(1)


_DISAGREE_WHAT = {"shape": "one-shot or loop", "tempo": "tempo (beyond an octave)",
                  "pitched": "pitched or not", "root": "root note"}


def _disagreements(examples) -> None:
    """Where Sononym's readings and Fourier's own disagree (metadata/resolve.disagreements):
    counts of each kind, and the first examples. Reading only."""
    from ..metadata.providers import SONONYM, ProviderError, has_data
    from ..metadata.resolve import disagreements
    with session_scope() as session:
        if not has_data(session, SONONYM):
            console.print("disagreements: Sononym not found: nothing to compare (Fourier's own "
                          "analysis is what the build reads).", markup=False, highlight=False)
            return
        try:
            got = disagreements(session)       # this config's library (rows.sample_select)
        except ProviderError as e:
            raise click.ClickException(str(e)) from None
    if not any(got["compared"].values()):
        console.print("disagreements: nothing to compare (none of this library's samples has both "
                      "Sononym's readings and Fourier's own analysis: `fourier tools analyze`).",
                      markup=False, highlight=False)
        return
    console.print(f"Sononym vs Fourier's own analysis, over {got['samples']:,} samples:",
                  markup=False, highlight=False)
    for what, label in _DISAGREE_WHAT.items():
        n, of = len(got["found"][what]), got["compared"][what]
        pct = f"{100 * n / of:.1f}%" if of else "-"
        console.print(f"  {label}: {n:,} of {of:,} compared disagree ({pct})", markup=False,
                      highlight=False)
        for rel, txt in got["found"][what][:examples]:
            console.print(f"    {rel}: {txt}", markup=False, highlight=False, soft_wrap=True)
    _sound_disagreements(examples)


def _sound_disagreements(examples) -> None:
    """Where the sound model's category (metadata/sound.py; a report while Sononym routes)
    and Sononym's differ, the model's most confident calls first. Reading only."""
    from sqlalchemy import text as _text

    from ..metadata import sound
    from ..metadata.rows import fetch, sample_select
    from ..places import library_rel
    try:
        mdl = sound.model()
    except sound.SoundModelError:
        return
    with session_scope() as session:
        got = sound.current(session, mdl) if mdl is not None else {}
        if not got:
            return
        son = defaultdict(set)
        for sid, lab in session.execute(_text(
                "SELECT sample_id, label FROM labels WHERE provider = 'sononym' AND kind = 'canonical'")):
            son[int(sid)].add(lab)
        rows = fetch(session, sample_select("id", "rel_path", "path", session=session))
    n, found = 0, []
    for r in rows:
        mine, labs = got.get(r.id), son.get(r.id)
        if not mine or not labs:
            continue
        loop = "class.loop" in labs and "class.oneshot" not in labs
        want = sound.category_for(sound.categories_of(labs), loop)
        if want is None:
            continue
        n += 1
        if mine[0] != want:
            found.append((-mine[1], r.rel_path or library_rel(r.path or ""),
                          f"{want} (Sononym) / {mine[0]} {mine[1]:.2f} (the sound model)"))
    if not n:
        return
    found.sort()
    console.print(f"  category, Sononym's vs the sound model's: {len(found):,} of {n:,} compared disagree "
                  f"({100 * len(found) / n:.1f}%)", markup=False, highlight=False)
    for _p, rel, txt in found[:examples]:
        console.print(f"    {rel}: {txt}", markup=False, highlight=False, soft_wrap=True)


def _stats() -> None:
    """Samples, favorites and Sononym's categories."""
    from sqlalchemy import text
    session = get_session()
    try:
        total = session.query(Sample).count()
        with_sononym = session.query(SononymMeta).count()
        favorites = session.query(Sample).filter_by(is_favorite=True).count()
        table = Table(title="Library database")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_row("Samples", f"{total:,}")
        table.add_row("With Sononym metadata", f"{with_sononym:,}")
        table.add_row("Favorites", f"{favorites:,}")
        console.print(table)
        if not with_sononym:
            return
        rows = session.execute(text("""
            SELECT sm.categories, sm.brightness, sm.noisiness, s.duration_s
            FROM sononym_meta sm
            JOIN samples s ON s.id = sm.sample_id
            WHERE sm.categories IS NOT NULL AND sm.categories != '[]'
              AND s.is_hidden = 0
        """)).fetchall()
        loops = session.execute(text(
            "SELECT COUNT(*) FROM sononym_meta WHERE classes LIKE '%Loop%'")).scalar() or 0
    finally:
        session.close()
    agg: dict[str, dict] = defaultdict(lambda: dict(count=0, dur=[], bright=[], noise=[]))
    for cats_raw, brightness, noisiness, duration in rows:
        try:
            cats = json.loads(cats_raw) if isinstance(cats_raw, str) else (cats_raw or [])
        except ValueError:
            continue
        if not cats:
            continue
        a = agg[cats[0]]
        a["count"] += 1
        for key, v in (("dur", duration), ("bright", brightness), ("noise", noisiness)):
            if v is not None:
                a[key].append(v)
    mean = lambda xs, fmt: fmt.format(sum(xs) / len(xs)) if xs else "?"
    table = Table(title="Sononym's categories")
    for col, justify in (("Category", "left"), ("Count", "right"), ("Avg length", "right"),
                         ("Avg bright", "right"), ("Avg noise", "right"), ("Use for", "left")):
        table.add_column(col, justify=justify, style="cyan" if col == "Category" else None)  # type: ignore[arg-type]
    for cat, a in sorted(agg.items(), key=lambda x: -x[1]["count"]):
        table.add_row(cat, f"{a['count']:,}", mean(a["dur"], "{:.2f}s"), mean(a["bright"], "{:.2f}"),
                      mean(a["noise"], "{:.2f}"), SONONYM_HINTS.get(cat, ""))
    if loops:
        table.add_row("Loop (class)", f"{loops:,}", "", "", "", "breakbeats / drum loops")
    console.print(table)
    console.print('[dim]Use these names with: fourier search --categories "Cat Name"[/dim]')


def _missing(prune) -> None:
    """Samples whose files are no longer on disk; with prune, removed from the database. The
    old rows of files a walk found moved (missing_files.moved_to) aren't missing: they're
    counted apart, as the scan counts them, and --prune removes them too."""
    import os
    import time

    from sqlalchemy import text as _text
    session = get_session()
    try:
        samples = session.query(Sample).all()
        console.print(f"[cyan]Checking {len(samples):,} samples...[/cyan]")
        t0 = time.monotonic()
        gone = [s for s in samples if not os.path.exists(s.path)]
        elapsed = time.monotonic() - t0
        try:
            moved_to = dict(session.execute(_text(  # type: ignore[arg-type]
                "SELECT sample_id, moved_to FROM missing_files WHERE moved_to IS NOT NULL")).all())
        except Exception:           # a database from before the column
            moved_to = {}
        moved = [s for s in gone if s.id in moved_to]
        gone = [s for s in gone if s.id not in moved_to]
        if moved:
            console.print(f"{len(moved):,} moved (old rows): files a scan found at a new path, whose "
                          f"analysis and ratings went with them (kept so a folder renamed back finds "
                          f"them; a build leaves them out)" + ("; --prune removes them too" if not prune
                                                               else ""), markup=False, highlight=False)
        if not gone:
            console.print(f"[green]All {len(samples) - len(moved):,} files present ({elapsed:.1f}s)[/green]")
            if prune and moved:
                from ..db.session import delete_samples
                n = delete_samples(session, [s.id for s in moved])
                console.print(f"[green]Removed {n:,} old rows of moved files from the database.[/green]")
            return
        table = Table(title=f"Missing files ({len(gone):,})")
        table.add_column("ID", style="dim", justify="right")
        table.add_column("Filename", style="yellow")
        table.add_column("Path", style="dim")
        for s in gone[:50]:
            table.add_row(str(s.id), s.filename, s.path[:80])
        if len(gone) > 50:
            table.add_row("...", f"({len(gone) - 50} more)", "")
        console.print(table)
        if prune:
            from ..db.session import delete_samples
            n = delete_samples(session, [s.id for s in gone + moved])
            console.print(f"[green]Removed {n:,} samples (and their analysis) from the database"
                          + (f", {len(moved):,} of them old rows of moved files" if moved else "")
                          + ".[/green]")
        else:
            console.print(f"\n[dim]--prune removes these {len(gone):,} samples from the database.[/dim]")
    finally:
        session.close()


def _metadata(do_rebuild, examples, shadow) -> bool:
    """Each provider's generic labels and descriptors against its source (Sononym's from
    sononym_meta, Live's auto-tags from samples.ableton_tags). The scan keeps them in step;
    rebuild does it by hand. True when every provider matches exactly."""
    from sqlalchemy import text as _text

    from ..metadata.store import SOURCES, parity, rebuild
    ok = True
    with session_scope() as session:
        for prov in SOURCES:
            if do_rebuild:
                n = rebuild(session, prov)
                console.print(f"{prov}: rebuilt {n['labels']:,} labels, {n['descriptors']:,} descriptors")
            rep = parity(session, prov, examples=examples)
            if rep["differences"]:
                ok = False
                console.print(f"[red]{prov}: {rep['samples']:,} samples, differences {rep['differences']}[/red]")
                for ex in rep["examples"]:
                    console.print(f"  {ex}")
            else:
                console.print(f"[green]{prov}: {rep['samples']:,} samples, parity exact[/green]")
        if shadow:
            from ..metadata import shadow as _sh
            for prov in _sh.SHADOW:
                if do_rebuild:
                    _sh.rebuild(session, prov)
                else:
                    _sh.ensure_current(session)
                a = _sh.agreement(session, prov, top=examples)
                console.print(f"shadow {_sh.format_agreement(a)}")
                for (ref, mine), n in a["confused"]:
                    console.print(f"  {n:>7,}  sononym {ref}  ->  {prov} {mine}")
        counts = dict(session.execute(_text(  # type: ignore[arg-type]
            "SELECT provider, count(*) FROM labels GROUP BY provider")).all())
    console.print(f"labels by provider: {counts or 'none'}")
    return ok
