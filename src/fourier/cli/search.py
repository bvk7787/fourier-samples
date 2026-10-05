"""fourier search, and fourier tools dedup: exploring the library."""
from __future__ import annotations

import click
from rich.table import Table

from ..db.session import get_session
from ..places import library_rel
from ._app import _table_cols, console, log, main, tools  # noqa: F401

# ---------------------------------------------------------------------------
# search command - free-form library explorer (no device context needed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

@main.command("search", short_help="Search the library by sound (CLAP) or by filters.")
@click.argument("query", nargs=-1)
@click.option("--class", "cls", default=None, type=click.Choice(["OneShot", "Loop"]),
              help="Sononym's class (OneShot or Loop)")
@click.option("--categories", "-c", default=None,
              help="Comma-separated Sononym category names (e.g. 'XFX Sweeps & Lasers,Tone Pads & Textures')")
@click.option("--min-dur", "min_dur", default=None, type=float, help="Min duration in seconds")
@click.option("--max-dur", "max_dur", default=None, type=float, help="Max duration in seconds")
@click.option("--brightness-min", "brightness_min", default=None, type=float, help="Sononym's brightness, 0 to 1")
@click.option("--brightness-max", "brightness_max", default=None, type=float)
@click.option("--noisiness-min", "noisiness_min", default=None, type=float, help="Sononym's noisiness, 0 to 1")
@click.option("--noisiness-max", "noisiness_max", default=None, type=float)
@click.option("--harmonicity-min", "harmonicity_min", default=None, type=float,
              help="Sononym's harmonicity, 0 to 1")
@click.option("--harmonicity-max", "harmonicity_max", default=None, type=float)
@click.option("--bpm-min", "bpm_min", default=None, type=float,
              help="Min BPM: a loop's tempo as a build resolves it (one-shots never match)")
@click.option("--bpm-max", "bpm_max", default=None, type=float, help="Max BPM")
@click.option("--bpm-octave", "bpm_octave", is_flag=True,
              help="Also match loops at half or double the tempo")
@click.option("--peak-min", "peak_min", default=None, type=float, help="Sononym's min peak dB (e.g. -12)")
@click.option("--favorites-only", is_flag=True, help="Only samples marked as favorites")
@click.option("--top", default=20, show_default=True, help="How many results to show")
@click.option("--like", "like_text", default=None,
              help="Text describing the sound (CLAP), the same as QUERY. E.g. 'dark sub kick'")
@click.option("--like-file", "like_file", default=None, type=click.Path(exists=True),
              help="An audio file to find similar sounds to (CLAP)")
def search(query, cls, categories, min_dur, max_dur, brightness_min, brightness_max,
           noisiness_min, noisiness_max, harmonicity_min, harmonicity_max,
           bpm_min, bpm_max, bpm_octave, peak_min, favorites_only, top, like_text, like_file):
    """Search the library: by words that describe the sound, by a reference file, or by
    filters.

    QUERY (or --like TEXT) and --like-file compare sounds with CLAP: they need the CLAP
    model and index (`fourier setup` installs the model; `fourier build` makes the index).
    --class, --categories and the
    brightness, noisiness, harmonicity and peak filters read Sononym's analysis and use its
    vocabulary, so they apply only when the library was scanned from Sononym; duration, BPM
    and favorites work without it.

    \b
    Examples:
      fourier search dark sub-heavy kick --top 20
      fourier search --like-file ~/Desktop/reference.wav --top 20
      fourier search --min-dur 1.0 --max-dur 4.0 --bpm-min 160 --bpm-max 180
    With Sononym:
      fourier search --class Loop --bpm-min 160 --bpm-max 180 --bpm-octave
      fourier search --categories "XFX Sweeps & Lasers,XFX Whooshes & Whips" --min-dur 1.0
      fourier search metal hit --categories "Perc Metal Hits"
    """
    from sqlalchemy import text as sql_text

    if query and like_text:
        raise click.UsageError("give the text once: as QUERY or as --like, not both")
    if query and like_file:
        raise click.UsageError("QUERY and --like-file both describe the sound: give one")
    like_text = " ".join(query) or like_text
    using_semantic = bool(like_text or like_file)
    from ..demo import REAL_ONLY, active
    if using_semantic and active():
        console.print(REAL_ONLY.format(what="searching by sound"), markup=False,
                      highlight=False, soft_wrap=True)
        raise SystemExit(1)
    if using_semantic:
        from ..paths import clap_index_path
        if not clap_index_path().exists():
            console.print("[red]Searching by sound needs the CLAP index: run `fourier build` "
                          "(with the CLAP model installed: `fourier setup`) first.[/red]")
            raise SystemExit(1)

    session = get_session()
    try:
        # --- Timbral filter pass (always run; may be the only pass) ---
        has_timbral_filters = any([
            cls, categories, min_dur, max_dur, brightness_min, brightness_max,
            noisiness_min, noisiness_max, harmonicity_min, harmonicity_max,
            bpm_min, bpm_max, peak_min, favorites_only, bpm_octave,
        ])

        where_clauses = ["s.is_hidden = 0", "s.file_format IN ('wav', 'aif', 'aiff')"]
        params: dict = {}
        # another config's library in this home's database (metadata/rows.py): left out
        from ..metadata.rows import _id_list, outside_library
        other = outside_library(session)
        if other:
            where_clauses.append(f"s.id NOT IN ({_id_list(other)})")

        if min_dur is not None:
            where_clauses.append(f"s.duration_s >= {min_dur}")
        if max_dur is not None:
            where_clauses.append(f"s.duration_s <= {max_dur}")
        if favorites_only:
            where_clauses.append("s.is_favorite = 1")

        if brightness_min is not None:
            where_clauses.append(f"sm.brightness >= {brightness_min}")
        if brightness_max is not None:
            where_clauses.append(f"sm.brightness <= {brightness_max}")
        if noisiness_min is not None:
            where_clauses.append(f"sm.noisiness >= {noisiness_min}")
        if noisiness_max is not None:
            where_clauses.append(f"sm.noisiness <= {noisiness_max}")
        if harmonicity_min is not None:
            where_clauses.append(f"sm.harmonicity >= {harmonicity_min}")
        if harmonicity_max is not None:
            where_clauses.append(f"sm.harmonicity <= {harmonicity_max}")
        if peak_min is not None:
            where_clauses.append(f"sm.peak_db >= {peak_min}")

        cat_class_clauses = []
        if categories:
            for i, cat in enumerate(categories.split(",")):
                cat = cat.strip()
                if cat:
                    key = f"cat_{i}"
                    cat_class_clauses.append(f"sm.categories LIKE :{key}")
                    params[key] = f"%{cat}%"
        if cls:
            cat_class_clauses.append("sm.classes LIKE :cls_filter")
            params["cls_filter"] = f"%{cls}%"

        if cat_class_clauses:
            where_clauses.append(f"({' OR '.join(cat_class_clauses)})")

        where_sql = " AND ".join(where_clauses)
        # Sononym's columns are NULL for a sample it never analyzed: its filters drop those
        _from = (
            "samples s "
            "LEFT JOIN sononym_meta sm ON sm.sample_id = s.id "
            "LEFT JOIN sample_features sf ON sf.sample_id = s.id"
        )
        # a tempo filter reads the tempo a build resolves (the name's, whole bars, octave
        # errors corrected), loops only: worked out here, not in SQL
        bpm_ids = None
        if bpm_min is not None or bpm_max is not None:
            bpm_ids = _bpm_matches(session, _from, where_sql, params, bpm_min, bpm_max, bpm_octave)
            total = len(bpm_ids)
        else:
            count_sql = sql_text(f"SELECT COUNT(*) FROM {_from} WHERE {where_sql}")
            total = session.execute(count_sql, params).scalar()

        # For semantic search we need a set of allowed IDs from timbral filters,
        # OR all IDs if no timbral filters applied.
        if using_semantic:
            if has_timbral_filters:
                # Fetch all matching IDs to pass as allowed_ids to CLAP search
                allowed_ids_sql = sql_text(f"SELECT s.id FROM {_from} WHERE {where_sql}")
                allowed_ids: set[int] | None = {
                    r[0] for r in session.execute(allowed_ids_sql, params).fetchall()
                    if bpm_ids is None or r[0] in bpm_ids
                }
            elif other:
                allowed_ids = {i for (i,) in session.execute(sql_text("SELECT id FROM samples"))} - other
            else:
                allowed_ids = None  # no restriction — search full index

            ids_ordered: list[int] = []  # will be set from CLAP results
        else:
            _limit = "" if bpm_ids is not None else f" LIMIT {top * 5}"
            ids_ordered = [
                r[0]
                for r in session.execute(
                    sql_text(
                        f"SELECT s.id FROM {_from} WHERE {where_sql} "
                        f"ORDER BY sm.peak_db DESC, s.id{_limit}"
                    ),
                    params,
                ).fetchall()
                if bpm_ids is None or r[0] in bpm_ids
            ][:top * 5]

        from ..db.models import Sample as SampleModel

        if using_semantic:
            # --- CLAP semantic search ---
            from fourier.analysis.clap_features import (
                embed_audio_file,
                embed_text,
                search_index,
            )

            try:
                if like_text:
                    console.print(f"[cyan]Encoding text query: '{like_text}'…[/cyan]")
                    query_vec = embed_text(like_text)
                else:
                    console.print(f"[cyan]Encoding reference audio: {like_file}…[/cyan]")
                    query_vec = embed_audio_file(like_file)
            except ImportError:
                console.print("[red]Searching by sound needs the CLAP model (the \\[clap] extra: "
                              "PyTorch and Transformers): `fourier setup` installs it.[/red]")
                raise SystemExit(1) from None

            try:
                clap_results = search_index(query_vec, top_k=top, allowed_ids=allowed_ids)
            except FileNotFoundError as exc:
                console.print(f"[red]{exc}[/red]")
                return

            if not clap_results:
                console.print("[yellow]No CLAP-indexed samples match your query.[/yellow]")
                console.print("  Run: fourier tools analyze --only clap")
                return

            ids_ordered = [sid for sid, _ in clap_results]
            clap_scores: dict[int, float] = {sid: score for sid, score in clap_results}
        else:
            clap_scores = {}

        from sqlalchemy.orm import joinedload as _joinedload
        id_to_sample = {}
        if ids_ordered:
            rows = (
                session.query(SampleModel)
                .options(_joinedload(SampleModel.sononym), _joinedload(SampleModel.features))
                .filter(SampleModel.id.in_(ids_ordered))
                .all()
            )
            # Force-access the lazy attrs while session is open
            for s in rows:
                _ = s.sononym, s.features  # trigger load (the table reads both)
                id_to_sample[s.id] = s
        samples_ordered = [id_to_sample[sid] for sid in ids_ordered if sid in id_to_sample]
        # without Sononym's category, the built-in path provider's label
        path_label = _path_labels(session, [x.id for x in samples_ordered[:top]
                                            if not (x.sononym and x.sononym.categories)])
        tempos = _loop_tempos(session, _from, [x.id for x in samples_ordered[:top]])

    finally:
        session.close()

    # --- Display ---
    filter_parts = []
    if like_text:
        filter_parts.append(f'like="{like_text[:40]}"')
    if like_file:
        import os
        filter_parts.append(f"like-file={os.path.basename(like_file)}")
    if cls:
        filter_parts.append(f"class={cls}")
    if categories:
        filter_parts.append(f"categories={categories[:50]}")
    if min_dur is not None:
        filter_parts.append(f"dur≥{min_dur}s")
    if max_dur is not None:
        filter_parts.append(f"dur≤{max_dur}s")
    if brightness_min is not None:
        filter_parts.append(f"bright≥{brightness_min}")
    if brightness_max is not None:
        filter_parts.append(f"bright≤{brightness_max}")
    if bpm_min is not None or bpm_max is not None:
        filter_parts.append(f"bpm {bpm_min or '?'}–{bpm_max or '?'}")
    filter_desc = "  ".join(filter_parts) or "(no filters)"

    console.print(f"\n[bold]Search:[/bold] {filter_desc}")
    if using_semantic:
        console.print(f"[green]{len(clap_results):,}[/green] semantic matches (from {total:,} candidates), showing top {min(top, len(samples_ordered))}\n")
    else:
        console.print(f"[green]{total:,}[/green] candidates found, showing top {min(top, len(samples_ordered))}\n")

    if not samples_ordered:
        console.print("[yellow]No samples found. Try relaxing your filters.[/yellow]")
        console.print("  Hint: `fourier tools db-stats` shows what's in your library.")
        return

    title = f"Search Results (top {min(top, len(samples_ordered))})"
    if using_semantic:
        title += " — semantic"
    table = Table(title=title)
    shown = samples_ordered[:top]
    # Sononym's measurements only when a result has them (without Sononym they'd be all "-")
    son = any(x.sononym for x in shown)
    if console.width < 100:           # a small window: no room for Sononym's measurements
        son = False
    table.add_column("#", justify="right", style="dim")
    table.add_column("Filename", style="green", min_width=12, overflow="fold")
    table.add_column("Folder", style="dim", max_width=28, no_wrap=True, overflow="ellipsis")
    table.add_column("Category", style="cyan", overflow="fold")
    table.add_column("Dur", justify="right", no_wrap=True)
    table.add_column("BPM", justify="right")
    if son:
        table.add_column("Bright", justify="right")
        table.add_column("Noise", justify="right")
    if using_semantic:
        table.add_column("Sim", justify="right", style="yellow")
    elif son:
        table.add_column("Peak dB", justify="right")

    for i, s in enumerate(shown, 1):
        meta = s.sononym
        cat = meta.categories[0] if (meta and meta.categories) else path_label.get(s.id, "-")
        where = _short_folder(s.rel_path or library_rel(s.path or ""))
        dur = f"{s.duration_s:.2f}s" if s.duration_s else "?"
        bpm_val = _tempo_text(tempos.get(s.id))
        row = [str(i), s.filename[:42], where, cat[:28], dur, bpm_val]
        if son:
            row += [f"{meta.brightness:.2f}" if (meta and meta.brightness is not None) else "—",
                    f"{meta.noisiness:.2f}" if (meta and meta.noisiness is not None) else "—"]
        if using_semantic:
            row.append(f"{clap_scores.get(s.id, 0):.3f}")
        elif son:
            row.append(f"{meta.peak_db:.1f}" if (meta and meta.peak_db is not None) else "—")
        table.add_row(*row)

    console.print(table)


LOOP_BPM_MIN_S = 2.0       # without features or a class, a file this long may be a loop


def _is_loop(classes, rel, duration_s, n_events, onset_rate_hz) -> bool:
    """Whether a sample is a loop, so its tempo means something (a snare's detected 117 BPM
    doesn't): Sononym's class, a loop word in its own name or folder, or its audio
    (metadata/shadow.py); with none of those, its length."""
    from ..metadata.shadow import audio_label, shape_labels
    classes = classes if isinstance(classes, str) else " ".join(map(str, classes or ()))
    if "Loop" in classes:
        return True
    if "OneShot" in classes:
        return False
    shape = shape_labels(rel)
    if shape:
        return "class.loop" in shape
    if n_events is not None:
        return audio_label(rel, duration_s, n_events, onset_rate_hz) == "class.loop"
    return (duration_s or 0) >= LOOP_BPM_MIN_S


def _loop_tempos(session, from_sql, ids=None, where_sql="1 = 1", params=None) -> dict:
    """{sample id: (tempo, resolved)} for the loops among `ids` (None: every sample `where_sql`
    matches); a one-shot isn't in it. The tempo is the one a build resolves
    (curate._resolve_tempo: a tempo the filename states; else Sononym's, else the measured one,
    each also at x2 and x0.5, whichever makes the loop whole bars), so a 120 BPM loop the
    analysis reads at 60 shows 120. A loop no tempo fits keeps its measured one, unresolved
    (None when it has none)."""
    from sqlalchemy import text as sql_text
    from ..packs.curate import _resolve_tempo
    if ids is not None:
        if not ids:
            return {}
        where_sql = f"s.id IN ({', '.join(str(int(i)) for i in ids)})"
    rows = session.execute(sql_text(
        "SELECT s.id, s.filename, s.rel_path, s.path, s.duration_s, sm.bpm, sm.classes, "
        "sf.tempo_bpm, sf.n_events, sf.onset_rate_hz "
        f"FROM {from_sql} WHERE {where_sql}"), params or {}).fetchall()
    from .enrich import fold_tempo
    out = {}
    for sid, fn, rel, path, dur, son_bpm, classes, tempo, n_ev, onsets in rows:
        rel = rel or library_rel(path or "")
        if not _is_loop(classes or "", rel, dur, n_ev, onsets):
            continue
        # Sononym's tempo, then Fourier's own (curation's chain)
        t, _src = _resolve_tempo(fn, dur, son_bpm, tempo)
        # unresolved: Sononym's tempo folded into 60-200, else Fourier's own (folded already)
        out[sid] = (t, True) if t else (((round(fold_tempo(son_bpm), 2) if son_bpm else None)
                                         or tempo), False)
    return out


def _tempo_text(got) -> str:
    """A loop's tempo for the table: "120", "~87" when no tempo makes it whole bars (the
    measurement, which can be off by an octave), "—" for a one-shot or none."""
    if not got or not got[0]:
        return "—"
    t, resolved = got
    return f"{t:.0f}" if resolved else f"~{t:.0f}"


def _bpm_matches(session, from_sql, where_sql, params, lo, hi, octave) -> set:
    """The loops matching the other filters whose tempo (_loop_tempos) is in [lo, hi]; with
    `octave`, at x2 or x0.5 too. One-shots never match a tempo filter."""
    lo = lo if lo is not None else 0
    hi = hi if hi is not None else 9999
    muls = (1, 2, 0.5) if octave else (1,)
    return {sid for sid, (t, _ok) in _loop_tempos(session, from_sql, None, where_sql, params).items()
            if t and any(lo <= t * m <= hi for m in muls)}


def _short_folder(rel: str, width: int = 28) -> str:
    """A file's folder under the library, cut from the left to `width` ("…/Pack/Kicks"), so
    two files of the same name from different packs tell apart."""
    import os
    d = os.path.dirname(rel or "")
    return d if len(d) <= width else "…" + d[-(width - 1):]


def _path_labels(session, ids) -> dict:
    """{sample id: the path provider's first category label} (metadata/shadow.py)."""
    from sqlalchemy import text
    if not ids:
        return {}
    out = {}
    rows = session.execute(text(
        "SELECT sample_id, label FROM labels WHERE provider = 'path' AND kind = 'canonical' "
        f"AND sample_id IN ({', '.join(str(int(i)) for i in ids)}) ORDER BY sample_id, rank"))
    for sid, label in rows:
        if not label.startswith("class.") and sid not in out:
            out[sid] = label
    return out


# ---------------------------------------------------------------------------
# fourier tools dedup: copies of the same file in the library
# ---------------------------------------------------------------------------
@tools.command("dedup", short_help="Find copies of the same file in the library.")
@click.option("--hide",    is_flag=True, help="Mark redundant copies as hidden (is_hidden=True)")
@click.option("--dry-run", is_flag=True, help="Report duplicates without making changes")
def dedup(hide, dry_run):
    """Find samples that share the same content hash (the first 8 KB of the file): the same
    file in more than one pack or folder. Nothing on disk changes.

    \b
    In each group, the one kept is chosen by:
      1. a favorite first
      2. the highest sample rate
      3. the longest
      4. the earliest scanned

    With --hide, the other copies are marked hidden in the database, so `fourier search`
    leaves them out; builds don't use the mark.

    \b
      fourier tools dedup --dry-run
      fourier tools dedup --hide
    """
    from sqlalchemy import text
    from fourier.db.models import Sample
    from fourier.db.session import get_session, session_scope

    session = get_session()
    try:
        # Find hashes that appear more than once
        dup_hashes = [
            row[0] for row in session.execute(
                text(
                    "SELECT file_hash FROM samples "
                    "WHERE file_hash IS NOT NULL "
                    "GROUP BY file_hash HAVING COUNT(*) > 1"
                )
            ).fetchall()
        ]

        if not dup_hashes:
            console.print("[green]No duplicates found.[/green]")
            return

        total_redundant = 0
        groups_info = []

        for h in dup_hashes:
            candidates = (
                session.query(Sample)
                .filter(Sample.file_hash == h)
                .order_by(
                    Sample.is_favorite.desc(),
                    Sample.sample_rate.desc(),
                    Sample.duration_s.desc(),
                    Sample.id.asc(),
                )
                .all()
            )
            keeper    = candidates[0]
            redundant = candidates[1:]
            total_redundant += len(redundant)
            groups_info.append((h, keeper, redundant))

    finally:
        session.close()

    console.print(
        f"[yellow]{len(groups_info)} duplicate group(s) found "
        f"({total_redundant} redundant copy/copies)[/yellow]"
    )
    for h, keeper, redundant in groups_info:
        console.print(f"\n  [dim]hash={h[:8]}[/dim]  keep: [green]{keeper.filename}[/green]")
        for s in redundant:
            _hidden = " [dim](already hidden)[/dim]" if s.is_hidden else ""
            console.print(f"    dup: [red]{s.filename}[/red]  {s.path}{_hidden}")

    if dry_run:
        console.print("\n[dim][dry-run: no changes made][/dim]")
        return

    if hide:
        ids_to_hide = [s.id for _, _, redundant in groups_info for s in redundant]
        with session_scope() as s:
            s.query(Sample).filter(Sample.id.in_(ids_to_hide)).update(
                {Sample.is_hidden: True}, synchronize_session=False
            )
        console.print(
            f"\n[green]Marked {len(ids_to_hide)} samples as hidden.[/green]"
        )
    else:
        console.print(
            "\n[dim]Run with --hide to mark redundant copies as hidden.[/dim]"
        )
