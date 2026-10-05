"""fourier tools analyze: the audio analysis curation reads (fourier build runs it)."""
from __future__ import annotations


import click
from rich import box
from rich.table import Table

from ._app import _table_cols, console, log, main, tools  # noqa: F401

# ---------------------------------------------------------------------------
# the analysis steps, run in order by fourier tools analyze and fourier build
# ---------------------------------------------------------------------------

# set by `fourier tools analyze --download`: the steps download cloud-only files instead of skipping them
DOWNLOAD_KEY = "fourier.download_cloud"


def _on_this_machine(items):
    """The (id, path) items an audio step can read now. A cloud drive's placeholder (a file
    whose content isn't on this machine) is skipped, and counted, since reading it starts a
    download that can fail and leave a wrong result; with `fourier tools analyze --download` the
    placeholders are downloaded first instead. One lstat per item."""
    from .. import platforms
    cloud = [p for _, p in items if p and platforms.cloud_only(p)]
    if not cloud:
        return items
    ctx = click.get_current_context(silent=True)
    if ctx is not None and ctx.meta.get(DOWNLOAD_KEY):
        console.print(f"[cyan]{len(cloud):,} cloud-only file(s): downloading them first[/cyan]")
        skip = set(platforms.materialize(cloud, log=lambda m: console.print(m, markup=False,
                                                                            highlight=False)))
        why = "couldn't be downloaded"
    else:
        skip, why = set(cloud), "are cloud-only (not on this machine)"
    if skip:
        console.print(f"[yellow]{len(skip):,} file(s) skipped: they {why}. `fourier tools "
                      f"analyze --download` downloads them first, or keep the library folder downloaded."
                      f"[/yellow]")
    return [it for it in items if it[1] not in skip]


# where to get the CLAP model (the [clap] extra: torch and transformers, then the model)
CLAP_INSTALL = "`fourier setup` installs it"


def clap_extra_missing() -> bool:
    """True when the [clap] extra (torch, transformers) isn't installed, so CLAP embeddings
    can't be computed here. A stand-in encoder put in its place (tests) needs no model."""
    import importlib.util

    from ..analysis import clap_features as CF
    if getattr(CF.embed_audio_file, "__module__", CF.__name__) != CF.__name__:
        return False
    return not all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


def clap_model_missing() -> bool:
    """True when the pinned CLAP model isn't downloaded yet (the first use downloads it). A
    stand-in encoder (tests) needs none."""
    from ..analysis import clap_features as CF
    if getattr(CF.embed_audio_file, "__module__", CF.__name__) != CF.__name__:
        return False
    return CF.cached_snapshot() is None


def sononym_classifies(session) -> bool:
    """Whether the build classifies with Sononym (metadata/providers.py). With it, the
    analysis steps cover the samples Sononym analyzed (the candidates); without it the
    built-in providers make every sample in the database a candidate, so every one is
    analyzed."""
    from ..metadata.providers import SONONYM, ProviderError, active, has_data
    try:
        return not active(session).fallback
    except ProviderError:
        return has_data(session, SONONYM)


def _candidates(session, Sample, SononymMeta, category_filter=None):
    """The samples an audio step covers (before its own "not done yet" filter): those with
    a Sononym row when Sononym classifies, else every sample, in id order. --category
    matches Sononym's categories, so it narrows to samples Sononym analyzed either way."""
    from sqlalchemy import String, cast
    if sononym_classifies(session):
        q = session.query(Sample).join(SononymMeta)
    else:
        q = session.query(Sample)
        if category_filter:
            q = q.join(SononymMeta)
        q = q.order_by(Sample.id)
    if category_filter:
        q = q.filter(cast(SononymMeta.categories, String).like(f"%{category_filter}%"))
    return q


def require_samples(session) -> None:
    """Stop with one line when the database has no samples: nothing to analyze yet."""
    from sqlalchemy import text
    if not session.execute(text("SELECT 1 FROM samples LIMIT 1")).first():
        raise click.ClickException("nothing scanned yet: run `fourier tools scan` (or `fourier build`, "
                                   "which scans first)")


def _fallback_loop(sample) -> bool:
    """Whether the built-in providers call a sample a loop (metadata/shadow.py): its path
    says so, or its length and event count do."""
    from ..metadata.shadow import audio_class, path_labels
    if "class.loop" in path_labels(sample.rel_path or sample.path or ""):
        return True
    feats = sample.features
    return audio_class(sample.duration_s, feats.n_events if feats else None,
                       feats.onset_rate_hz if feats else None,
                       feats.harmonic_percussive_ratio if feats else None) == "class.loop"


# the derived step's columns: legacy, computed from a Sononym row only (db/models.py)
DERIVED_COLUMNS = ("sub_weight", "transient_score", "loop_confidence", "is_pitched", "bpm_reliable",
                   "timbral_norm", "spectral_balance", "pitch_stability", "attack_class", "drum_subtype")


def fill_file_info(session, samples) -> int:
    """Duration, sample rate, channels, bit depth and format, from the file's header
    (soundfile.info), for samples that don't have them yet (a file walk from before the scan
    recorded them). Returns how many it filled."""
    from ..ingest.importer import file_info
    n = 0
    for s in samples:
        if s.duration_s is not None:
            continue
        info = file_info(s.path)
        if info:
            for k, v in info.items():
                if getattr(s, k) is None:
                    setattr(s, k, v)
            n += 1
    if n:
        session.commit()
    return n


# not a command of its own: the steps' options and defaults, in one place
steps = click.Group("steps", help="The analysis steps (fourier tools analyze).")


@steps.command("derived")
@click.option("--limit", default=0, type=int, help="Limit to N samples (0 = all)")
@click.option("--force", is_flag=True, help="Re-compute even if already enriched")
def enrich_derived(limit, force):
    """
    Compute Tier 1 derived features from existing Sononym metadata (no audio reads).

    Computes sub_weight, transient_score, loop_confidence, is_pitched, bpm_reliable,
    timbral_norm, spectral_balance, pitch_stability, attack_class and drum_subtype (legacy
    columns, Sononym's: db/models.py) for every sample with a Sononym row not done yet.
    Without Sononym every sample is marked done with those columns empty: they never hold
    a stand-in for Sononym's reading.

    Runs in seconds even for a large library. Safe to re-run.

    Runs in `fourier tools analyze` (--only derived).
    """
    import time
    from datetime import datetime

    from sqlalchemy import text

    from fourier.analysis.derived import compute_derived
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    from fourier.db.session import get_session

    session = get_session()
    try:
        require_samples(session)
        # Ensure new columns exist (lightweight schema migration for existing DBs).
        # Includes ALL the newer columns (both Tier 1 and Tier 2) so the ORM model
        # can SELECT without error even before enrich_librosa has been run.
        new_cols = [
            ("sub_weight", "REAL"),
            ("transient_score", "REAL"),
            ("loop_confidence", "REAL"),
            ("is_pitched", "INTEGER"),
            ("bpm_reliable", "INTEGER"),
            ("timbral_norm", "TEXT"),
            ("derived_computed_at", "DATETIME"),
            # Tier 1 additions
            ("spectral_balance", "REAL"),
            ("pitch_stability", "INTEGER"),
            ("attack_class", "TEXT"),
            ("drum_subtype", "TEXT"),
            # Tier 2 additions (populated by enrich_librosa, NULL until then)
            ("spectral_centroid_mean", "REAL"),
            ("spectral_rolloff_mean", "REAL"),
            ("zero_crossing_rate_mean", "REAL"),
            ("rms_mean", "REAL"),
            ("attack_time_ms", "REAL"),
            ("decay_time_ms", "REAL"),
            ("harmonic_percussive_ratio", "REAL"),
            ("chroma_concentration", "REAL"),
            # Quality gate
            ("is_clipped", "INTEGER"),
            ("dc_offset_ratio", "REAL"),
            # BPM correction
            ("bpm_corrected", "REAL"),
            # Loop trim (populated by the loop-trim step, NULL until then)
            ("trim_end_s", "REAL"),
            ("trim_computed_at", "DATETIME"),
        ]
        existing_col_names = _table_cols("sample_features")
        for col_name, col_type in new_cols:
            if col_name not in existing_col_names:
                session.execute(text(f"ALTER TABLE sample_features ADD COLUMN {col_name} {col_type}"))
        session.commit()

        # Find samples to process
        # joinedload(SononymMeta.sample) avoids N+1 queries when accessing
        # meta.sample.duration_s for attack_class and drum_subtype computation.
        from sqlalchemy.orm import joinedload as _joinedload
        if sononym_classifies(session):
            q = session.query(SononymMeta).options(_joinedload(SononymMeta.sample))
            if not force:
                q = (
                    q.outerjoin(SampleFeatures, SononymMeta.sample_id == SampleFeatures.sample_id)
                    .filter(
                        (SampleFeatures.id == None)  # noqa: E711
                        | (SampleFeatures.derived_computed_at == None)  # noqa: E711
                    )
                )
            metas = q.limit(limit).all() if limit else q.all()
            todo = [(m.sample_id, m, m.sample) for m in metas]
        else:
            # the built-in providers classify: every sample, with its Sononym row when it has one
            q = (session.query(Sample, SononymMeta)
                 .outerjoin(SononymMeta, SononymMeta.sample_id == Sample.id))
            if not force:
                q = (q.outerjoin(SampleFeatures, Sample.id == SampleFeatures.sample_id)
                     .filter((SampleFeatures.id == None)  # noqa: E711
                             | (SampleFeatures.derived_computed_at == None)))  # noqa: E711
            q = q.order_by(Sample.id)
            rows = q.limit(limit).all() if limit else q.all()
            fill_file_info(session, [s for s, _m in rows])
            todo = [(s.id, m, s) for s, m in rows]

        if not todo:
            console.print("[green]All samples already have derived features.[/green]")
            return

        console.print(f"[cyan]Computing derived features for {len(todo):,} samples...[/cyan]")
        t0 = time.monotonic()
        n_processed = 0
        n_created = 0

        for sample_id, meta, sample in todo:
            duration_s = sample.duration_s if sample else None
            # Sononym's values only: a sample without a Sononym row gets none
            derived = (compute_derived(meta, duration_s=duration_s) if meta is not None
                       else dict.fromkeys(DERIVED_COLUMNS))
            now = datetime.utcnow()

            # Upsert into SampleFeatures
            existing = (
                session.query(SampleFeatures)
                .filter(SampleFeatures.sample_id == sample_id)
                .first()
            )
            if existing is None:
                feat = SampleFeatures(sample_id=sample_id)     # computed_at: librosa's, when it runs
                session.add(feat)
                n_created += 1
            else:
                feat = existing

            feat.sub_weight = derived["sub_weight"]
            feat.transient_score = derived["transient_score"]
            feat.loop_confidence = derived["loop_confidence"]
            feat.is_pitched = derived["is_pitched"]
            feat.bpm_reliable = derived["bpm_reliable"]
            feat.timbral_norm = derived["timbral_norm"]
            feat.derived_computed_at = now
            # added later
            feat.spectral_balance = derived.get("spectral_balance")
            feat.pitch_stability = derived.get("pitch_stability")
            feat.attack_class = derived.get("attack_class")
            feat.drum_subtype = derived.get("drum_subtype")
            n_processed += 1

            # Batch commit every 5000 rows
            if n_processed % 5000 == 0:
                session.commit()

        session.commit()
        elapsed = time.monotonic() - t0
        console.print(
            f"[green]✓ Enriched {n_processed:,} samples "
            f"({n_created:,} new rows) in {elapsed:.1f}s[/green]"
        )

    finally:
        session.close()


def fold_tempo(bpm: float, lo: float = 60.0, hi: float = 200.0) -> float:
    """A tempo doubled while under lo and halved while over hi (half and double time are the
    same groove)."""
    if bpm <= 0:
        return bpm
    while bpm < lo:
        bpm *= 2.0
    while bpm > hi:
        bpm /= 2.0
    return bpm


@steps.command("bpm-fix")
@click.option("--force", is_flag=True, help="Re-compute even if bpm_corrected already set")
def enrich_bpm_fix(force):
    """
    Fold Sononym's tempo by octaves into [60, 200] (sononym_bpm_folded).

    Sononym's tempo only: Fourier's own (tempo_bpm, librosa) is folded when it's measured.
    The legacy bpm_corrected column, which held either, is no longer written (db/models.py).
    Without Sononym there is nothing to fold.

    Runs in seconds (pure SQL + math, no audio reads). Safe to re-run.

    Runs in `fourier tools analyze` (--only bpm-fix).
    """
    import time

    from fourier.db.models import SampleFeatures, SononymMeta
    from fourier.db.session import get_session

    session = get_session()
    try:
        require_samples(session)
        q = (session.query(SampleFeatures, SononymMeta)
             .join(SononymMeta, SampleFeatures.sample_id == SononymMeta.sample_id)
             .filter(SononymMeta.bpm != None)  # noqa: E711
             .order_by(SampleFeatures.sample_id))
        if not force:
            q = q.filter(SampleFeatures.sononym_bpm_folded == None)  # noqa: E711

        rows = q.all()
        if not rows:
            if session.query(SononymMeta.sample_id).filter(SononymMeta.bpm != None).first():  # noqa: E711
                console.print("[green]Every Sononym tempo is folded already.[/green]")
            else:
                console.print("No Sononym tempos to fold (Fourier's own tempos need no folding).")
            return

        console.print(f"[cyan]Folding Sononym's tempo for {len(rows):,} samples…[/cyan]")
        t0 = time.monotonic()
        n_fixed = 0

        for feat, meta in rows:
            feat.sononym_bpm_folded = round(fold_tempo(meta.bpm), 2)
            n_fixed += 1

            if n_fixed % 10000 == 0:
                session.commit()
                console.print(f"  {n_fixed:,} / {len(rows):,}…")

        session.commit()
        elapsed = time.monotonic() - t0
        console.print(f"[green]✓ Folded Sononym's tempo for {n_fixed:,} samples in {elapsed:.1f}s[/green]")

    finally:
        session.close()


@steps.command("librosa")
@click.option("--limit", default=0, type=int, help="Limit to N samples (0 = all)")
@click.option("--workers", default=4, type=int, help="Parallel worker threads")
@click.option("--force", is_flag=True, help="Re-compute even if already enriched")
@click.option("--category", "category_filter", default=None, help="Only enrich samples in this category")
def enrich_librosa(limit, workers, force, category_filter):
    """
    Compute Tier 2 librosa audio features (reads audio files).

    Computes mfcc_mean, onset_rate_hz, tempo_bpm, spectral_flatness_mean,
    attack_time_ms, decay_time_ms, harmonic_percussive_ratio, chroma_concentration,
    and wires in spectral_centroid_mean, spectral_rolloff_mean, zero_crossing_rate_mean,
    rms_mean. Runs incrementally — safe to stop and restart; already-enriched samples
    are skipped unless --force is used.

    Reads every audio file once; HPSS is the slow part.

    Runs in `fourier tools analyze` (--only librosa).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

    from sqlalchemy import text

    from fourier.analysis.librosa_features import compute_features
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    from fourier.db.session import get_session

    session = get_session()
    try:
        require_samples(session)
        # Ensure the newer Tier 2 columns exist (lightweight migration for existing DBs)
        new_cols = [
            ("spectral_centroid_mean", "REAL"),
            ("spectral_rolloff_mean", "REAL"),
            ("zero_crossing_rate_mean", "REAL"),
            ("rms_mean", "REAL"),
            ("attack_time_ms", "REAL"),
            ("decay_time_ms", "REAL"),
            ("harmonic_percussive_ratio", "REAL"),
            ("chroma_concentration", "REAL"),
            # Quality gate
            ("is_clipped", "INTEGER"),
            ("dc_offset_ratio", "REAL"),
            # BPM correction
            ("bpm_corrected", "REAL"),
        ]
        existing_col_names = _table_cols("sample_features")
        for col_name, col_type in new_cols:
            if col_name not in existing_col_names:
                session.execute(text(f"ALTER TABLE sample_features ADD COLUMN {col_name} {col_type}"))
        session.commit()

        q = _candidates(session, Sample, SononymMeta, category_filter)

        if not force:
            q = (
                q.outerjoin(SampleFeatures, Sample.id == SampleFeatures.sample_id)
                .filter(
                    (SampleFeatures.id == None)  # noqa: E711
                    | (SampleFeatures.mfcc_mean == None)  # noqa: E711
                )
            )

        samples = q.limit(limit).all() if limit else q.all()
        if not samples:
            console.print("[green]All samples already have librosa features.[/green]")
            return

        console.print(
            f"[cyan]Computing librosa features for {len(samples):,} samples "
            f"({workers} workers)...[/cyan]"
        )

        paths = _on_this_machine([(s.id, s.path) for s in samples])
        n_ok = 0
        n_err = 0
        t0 = time.monotonic()

        def process_one(item):
            sample_id, path = item
            feats = compute_features(path)
            return sample_id, feats

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console, disable=not console.is_terminal,
        ) as progress:
            task = progress.add_task("librosa", total=len(paths))

            # Use a fresh session per batch to avoid long transactions
            batch_session = get_session()
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(process_one, item): item for item in paths}
                    batch = []
                    for future in as_completed(futures):
                        sample_id, feats = future.result()
                        if feats:
                            batch.append((sample_id, feats))
                            n_ok += 1
                        else:
                            n_err += 1
                        progress.advance(task)

                        # Commit every 500 rows
                        if len(batch) >= 500:
                            _upsert_librosa_batch(batch_session, batch)
                            batch = []

                    if batch:
                        _upsert_librosa_batch(batch_session, batch)
            finally:
                batch_session.close()

        elapsed = time.monotonic() - t0
        console.print(
            f"[green]✓ Enriched {n_ok:,} samples[/green] "
            f"[dim]({n_err} skipped/errored) in {elapsed:.0f}s[/dim]"
        )

    finally:
        session.close()


@steps.command("quality")
@click.option("--workers", default=4, type=int, help="Parallel worker threads")
@click.option("--force", is_flag=True, help="Re-check even if already computed")
def enrich_quality(workers, force):
    """
    Check all samples for clipping and DC offset (fast, no heavy DSP).

    Populates is_clipped and dc_offset_ratio in sample_features.
    Much faster than a full librosa pass.

    After running, clipped or DC-offset samples are automatically penalised
    during pack building (is_clipped → -2.0 score; dc_offset > 0.02 → -0.5).

    Runs in `fourier tools analyze` (--only quality).
    """
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

    from fourier.db.models import Sample, SampleFeatures
    from fourier.db.session import get_session

    def _check_quality(item):
        sample_id, path = item
        if not os.path.exists(path):
            return sample_id, None
        try:
            import soundfile as sf

            from fourier.analysis.clap_features import read_seconds
            cap = read_seconds(path)            # a very long file: its start (as CLAP hears it)
            if cap is None:
                data, _ = sf.read(path, always_2d=False, dtype="float32")
            else:
                data, _ = sf.read(path, always_2d=False, dtype="float32",
                                  frames=int(cap * sf.info(path).samplerate))
            if data.ndim > 1:
                y = data.mean(axis=1)
            else:
                y = data
            if len(y) == 0:
                return sample_id, None
            clipped = int(np.sum(np.abs(y) >= 0.999))
            return sample_id, {
                "is_clipped": int(clipped / len(y) > 0.001),
                "dc_offset_ratio": round(float(abs(np.mean(y))), 6),
            }
        except Exception:
            return sample_id, None

    session = get_session()
    try:
        require_samples(session)
        from sqlalchemy import text
        # Ensure columns exist
        existing_cols = _table_cols("sample_features")
        for col, typ in [("is_clipped", "INTEGER"), ("dc_offset_ratio", "REAL")]:
            if col not in existing_cols:
                session.execute(text(f"ALTER TABLE sample_features ADD COLUMN {col} {typ}"))
        session.commit()

        q = session.query(Sample)
        if not force:
            q = (
                q.outerjoin(SampleFeatures, Sample.id == SampleFeatures.sample_id)
                .filter(
                    (SampleFeatures.id == None)  # noqa: E711
                    | (SampleFeatures.is_clipped == None)  # noqa: E711
                )
            )
        samples = q.all()

        if not samples:
            console.print("[green]All samples already have quality data.[/green]")
            return

        console.print(
            f"[cyan]Quality checking {len(samples):,} samples ({workers} workers)…[/cyan]"
        )
        paths = _on_this_machine([(s.id, s.path) for s in samples])
        t0 = time.monotonic()
        n_ok = n_err = n_clipped = n_dc = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console, disable=not console.is_terminal,
        ) as progress:
            task = progress.add_task("quality", total=len(paths))
            batch_session = get_session()
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(_check_quality, item): item for item in paths}
                    batch = []
                    for future in as_completed(futures):
                        sample_id, result = future.result()
                        if result is not None:
                            batch.append((sample_id, result))
                            n_ok += 1
                            if result["is_clipped"]:
                                n_clipped += 1
                            if result["dc_offset_ratio"] > 0.02:
                                n_dc += 1
                        else:
                            n_err += 1
                        progress.advance(task)

                        if len(batch) >= 500:
                            _upsert_quality_batch(batch_session, batch)
                            batch = []
                    if batch:
                        _upsert_quality_batch(batch_session, batch)
            finally:
                batch_session.close()

        elapsed = time.monotonic() - t0
        console.print(
            f"[green]✓ Checked {n_ok:,} samples in {elapsed:.0f}s[/green]"
            + (f" — [yellow]{n_clipped} clipped[/yellow]" if n_clipped else "")
            + (f", [yellow]{n_dc} with DC offset[/yellow]" if n_dc else "")
            + (f" [dim]({n_err} skipped)[/dim]" if n_err else "")
        )
    finally:
        session.close()


def _upsert_quality_batch(session, batch: list[tuple[int, dict]]) -> None:
    from fourier.db.models import SampleFeatures
    for sample_id, result in batch:
        feat = (
            session.query(SampleFeatures)
            .filter(SampleFeatures.sample_id == sample_id)
            .first()
        )
        if feat is None:
            feat = SampleFeatures(sample_id=sample_id)
            session.add(feat)
        feat.is_clipped = result["is_clipped"]
        feat.dc_offset_ratio = result["dc_offset_ratio"]
    session.commit()


def _upsert_loop_trim_batch(session, batch: list[tuple[int, dict]]) -> None:
    """Insert or update loop-trim metadata for a batch of (sample_id, result) pairs."""
    from fourier.db.models import SampleFeatures

    for sample_id, result in batch:
        feat = (
            session.query(SampleFeatures)
            .filter(SampleFeatures.sample_id == sample_id)
            .first()
        )
        if feat is None:
            feat = SampleFeatures(sample_id=sample_id)
            session.add(feat)
        feat.trim_end_s = result.get("trim_end_s")            # None is valid
        feat.trim_computed_at = result.get("trim_computed_at")
    session.commit()


def _upsert_librosa_batch(session, batch: list[tuple[int, dict]]) -> None:
    """Insert or update librosa features for a batch of (sample_id, feats) pairs."""
    from datetime import datetime

    from fourier.db.models import SampleFeatures

    now = datetime.utcnow()
    for sample_id, feats in batch:
        existing = (
            session.query(SampleFeatures)
            .filter(SampleFeatures.sample_id == sample_id)
            .first()
        )
        if existing is None:
            feat = SampleFeatures(sample_id=sample_id)
            session.add(feat)
        else:
            feat = existing

        if "mfcc_mean" in feats:
            feat.mfcc_mean = feats["mfcc_mean"]
        if "onset_rate_hz" in feats:
            feat.onset_rate_hz = feats["onset_rate_hz"]
        if "tempo_bpm" in feats:
            feat.tempo_bpm = feats["tempo_bpm"]
        if "spectral_flatness_mean" in feats:
            feat.spectral_flatness_mean = feats["spectral_flatness_mean"]
        # Previously-empty columns, now wired in
        if "spectral_centroid_mean" in feats:
            feat.spectral_centroid_mean = feats["spectral_centroid_mean"]
        if "spectral_rolloff_mean" in feats:
            feat.spectral_rolloff_mean = feats["spectral_rolloff_mean"]
        if "zero_crossing_rate_mean" in feats:
            feat.zero_crossing_rate_mean = feats["zero_crossing_rate_mean"]
        if "rms_mean" in feats:
            feat.rms_mean = feats["rms_mean"]
        # newer Tier 2 columns
        if "attack_time_ms" in feats:
            feat.attack_time_ms = feats["attack_time_ms"]
        if "decay_time_ms" in feats:
            feat.decay_time_ms = feats["decay_time_ms"]
        if "harmonic_percussive_ratio" in feats:
            feat.harmonic_percussive_ratio = feats["harmonic_percussive_ratio"]
        if "chroma_concentration" in feats:
            feat.chroma_concentration = feats["chroma_concentration"]
        # Quality gate
        if "is_clipped" in feats:
            feat.is_clipped = feats["is_clipped"]
        if "dc_offset_ratio" in feats:
            feat.dc_offset_ratio = feats["dc_offset_ratio"]
        feat.computed_at = now

    session.commit()


@steps.command("clap")
@click.option("--limit", default=0, type=int, help="Limit to N samples (0 = all)")
@click.option("--category", "category_filter", default=None, help="Only enrich samples in this category")
@click.option("--workers", default=1, type=int, help="Parallel threads (CPU-only; MPS/CUDA auto-detected)")
@click.option("--force", is_flag=True, help="Re-compute even if already enriched")
@click.option("--build-index", "build_index_flag", is_flag=True, help="Build/rebuild search index after enrichment")
def enrich_clap(limit, category_filter, workers, force, build_index_flag):
    """
    Compute Tier 3 CLAP audio embeddings (semantic search).

    Downloads laion/clap-htsat-unfused (about 600 MB) on first use unless `fourier setup`
    already has; after that it loads offline. Encodes each audio file into a 512-dim
    embedding stored as a BLOB in the database.

    With --build-index it then writes the search index `fourier search` reads.

    Runs in `fourier tools analyze` (--only clap). Needs the CLAP model (`fourier setup`
    installs it).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
    from sqlalchemy import text

    from fourier.analysis.clap_features import (
        CLAP_MODEL_ID,
        build_index,
        embed_audio_file,
        embedding_to_bytes,
    )
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    from fourier.db.session import get_session

    session = get_session()
    try:
        require_samples(session)
        # Inline schema migration — add CLAP columns to existing DBs
        new_cols = [
            ("clap_embedding", "BLOB"),
            ("clap_model", "TEXT"),
        ]
        existing_col_names = _table_cols("sample_features")
        for col_name, col_type in new_cols:
            if col_name not in existing_col_names:
                session.execute(text(f"ALTER TABLE sample_features ADD COLUMN {col_name} {col_type}"))
        session.commit()

        # --- Embedding pass ---
        q = _candidates(session, Sample, SononymMeta, category_filter)
        if not force:
            q = (
                q.outerjoin(SampleFeatures, Sample.id == SampleFeatures.sample_id)
                .filter(
                    (SampleFeatures.id == None)  # noqa: E711
                    | (SampleFeatures.clap_embedding == None)  # noqa: E711
                )
            )
        samples = q.limit(limit).all() if limit else q.all()

        if not samples:
            console.print("[green]All samples already have CLAP embeddings.[/green]")
            if build_index_flag:
                _write_index(session, build_index)
            return

        if clap_extra_missing():
            console.print(f"clap: skipped: CLAP embeddings need the [clap] extra (torch and "
                          f"transformers): {CLAP_INSTALL}, then `fourier build` analyzes them. "
                          f"A build needs them; the other steps don't.", style="yellow",
                          markup=False, highlight=False)
            return

        console.print(
            f"[cyan]Computing CLAP embeddings for {len(samples):,} samples "
            f"(model: {CLAP_MODEL_ID}, {workers} worker(s))…[/cyan]"
        )
        from fourier.analysis.clap_features import cached_snapshot
        if cached_snapshot() is None:
            console.print("[dim]The CLAP model isn't downloaded yet: about 600 MB from Hugging Face, "
                          "once (`fourier setup` does it ahead of time).[/dim]")

        paths = _on_this_machine([(s.id, s.path) for s in samples])
        n_ok = 0
        n_err = 0
        t0 = time.monotonic()

        def embed_one(item):
            sample_id, path = item
            vec = embed_audio_file(path)
            return sample_id, vec

        batch_session = get_session()
        try:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console, disable=not console.is_terminal,
            ) as progress:
                task = progress.add_task("clap", total=len(paths))

                if workers > 1:
                    # Multi-thread only makes sense on CPU; GPU enrichment is serial
                    ctx = ThreadPoolExecutor(max_workers=workers)
                    futures = {ctx.submit(embed_one, item): item for item in paths}
                    batch: list[tuple[int, bytes]] = []
                    with ctx:
                        for future in as_completed(futures):
                            sample_id, vec = future.result()
                            if vec is not None and vec.any():
                                batch.append((sample_id, embedding_to_bytes(vec)))
                                n_ok += 1
                            else:
                                n_err += 1
                            progress.advance(task)
                            if len(batch) >= 200:
                                _upsert_clap_batch(batch_session, batch, CLAP_MODEL_ID)
                                batch = []
                        if batch:
                            _upsert_clap_batch(batch_session, batch, CLAP_MODEL_ID)
                else:
                    batch = []
                    for item in paths:
                        sample_id, vec = embed_one(item)
                        if vec is not None and vec.any():
                            batch.append((sample_id, embedding_to_bytes(vec)))
                            n_ok += 1
                        else:
                            n_err += 1
                        progress.advance(task)
                        if len(batch) >= 200:
                            _upsert_clap_batch(batch_session, batch, CLAP_MODEL_ID)
                            batch = []
                    if batch:
                        _upsert_clap_batch(batch_session, batch, CLAP_MODEL_ID)
        finally:
            batch_session.close()

        elapsed = time.monotonic() - t0
        console.print(
            f"[green]✓ Embedded {n_ok:,} samples[/green] "
            f"[dim]({n_err} skipped/errored) in {elapsed:.0f}s[/dim]"
        )

        if build_index_flag or (limit == 0 and not category_filter):
            _write_index(session, build_index)

    finally:
        session.close()


def _write_index(session, build_index) -> None:
    """Build the CLAP search index; with no embeddings at all, say so in one line."""
    console.print("[cyan]Building search index…[/cyan]")
    session.commit()        # a fresh transaction: it sees the embeddings just written
    try:
        path = build_index(session)
    except RuntimeError:            # no embeddings: nothing was embedded, nothing to index
        console.print("[yellow]No CLAP embeddings yet, so no search index: none of the samples "
                      "could be embedded.[/yellow]")
        return
    if getattr(build_index, "wrote", True) is False:
        console.print(f"Search index unchanged: {path}", markup=False, highlight=False)
    else:
        console.print(f"[green]✓ Index written:[/green] {path}")


def _upsert_clap_batch(session, batch: list[tuple[int, bytes]], model_id: str) -> None:
    """Insert or update CLAP embeddings for a batch of (sample_id, blob) pairs."""
    from fourier.db.models import SampleFeatures

    for sample_id, blob in batch:
        existing = (
            session.query(SampleFeatures)
            .filter(SampleFeatures.sample_id == sample_id)
            .first()
        )
        if existing is None:
            feat = SampleFeatures(sample_id=sample_id, clap_embedding=blob, clap_model=model_id)
            session.add(feat)
        else:
            existing.clap_embedding = blob
            existing.clap_model = model_id
    session.commit()



@steps.command("loop-trim")
@click.option("--workers", default=4, type=int, show_default=True,
              help="Parallel worker threads")
@click.option("--force", is_flag=True,
              help="Re-compute even if trim_computed_at is already set")
@click.option("--limit", default=0, type=int,
              help="Cap at N samples for testing (0 = all)")
@click.option("--threshold-db", "threshold_db", default=-60.0, type=float,
              show_default=True,
              help="RMS threshold in dB below which audio is considered silent")
@click.option("--min-silence", "min_silence_s", default=0.05, type=float,
              show_default=True,
              help="Minimum trailing silence (seconds) required to record a trim point")
@click.option("--all-classes", "all_classes", is_flag=True, default=False,
              help="Process all samples, not just those classified as Loop")
def enrich_loop_trim(workers, force, limit, threshold_db, min_silence_s, all_classes):
    """
    Detect trailing silence trim points for loop samples (non-destructive).

    Computes the optimal trim endpoint for each loop by analyzing the RMS
    envelope, then stores it in sample_features.trim_end_s. The original audio
    file is NEVER modified. The trim is applied automatically at export time
    (device renders).

    By default only processes samples that Sononym classifies as Loop.
    Use --all-classes to include OneShots and unclassified samples.

    Runs in `fourier tools analyze` (--only loop-trim).
    """
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
    from sqlalchemy import text

    from fourier.analysis.loop_trim import compute_loop_trim
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    from fourier.db.session import get_session

    def _compute_trim(item):
        sample_id, path = item
        if not os.path.exists(path):
            return sample_id, None
        result = compute_loop_trim(
            path, threshold_db=threshold_db, min_silence_s=min_silence_s
        )
        return sample_id, result

    session = get_session()
    try:
        require_samples(session)
        existing_cols = _table_cols("sample_features")
        for col, typ in [("trim_end_s", "REAL"), ("trim_computed_at", "DATETIME")]:
            if col not in existing_cols:
                session.execute(
                    text(f"ALTER TABLE sample_features ADD COLUMN {col} {typ}")
                )
        session.commit()

        fallback = not sononym_classifies(session)
        if fallback:
            # the built-in providers' call: a loop by its name or folder, or by its length
            # and event count (events runs before this step)
            q = session.query(Sample).order_by(Sample.id)
        else:
            q = session.query(Sample).join(
                SononymMeta, Sample.id == SononymMeta.sample_id
            )
            if not all_classes:
                q = q.filter(SononymMeta.classes.like('%Loop%'))
        if not force:
            q = (
                q.outerjoin(SampleFeatures, Sample.id == SampleFeatures.sample_id)
                .filter(
                    (SampleFeatures.id == None)  # noqa: E711
                    | (SampleFeatures.trim_computed_at == None)  # noqa: E711
                )
            )
        if fallback and not all_classes:
            samples = [s for s in q.all() if _fallback_loop(s)]
            samples = samples[:limit] if limit else samples
        else:
            if limit:
                q = q.limit(limit)
            samples = q.all()
        if not samples:
            console.print("[green]All eligible loop samples already have trim data.[/green]")
            return

        console.print(
            f"[cyan]Loop-trim analysis: {len(samples):,} samples ({workers} workers)...[/cyan]"
        )
        paths = _on_this_machine([(s.id, s.path) for s in samples])
        t0 = time.monotonic()
        n_trimmed = n_skipped = n_err = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console, disable=not console.is_terminal,
        ) as progress:
            task = progress.add_task("loop-trim", total=len(paths))
            batch_session = get_session()
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_compute_trim, item): item for item in paths
                    }
                    batch = []
                    for future in as_completed(futures):
                        sample_id, result = future.result()
                        if result is not None:
                            batch.append((sample_id, result))
                            if result["trim_end_s"] is not None:
                                n_trimmed += 1
                            else:
                                n_skipped += 1
                        else:
                            n_err += 1
                        progress.advance(task)
                        if len(batch) >= 500:
                            _upsert_loop_trim_batch(batch_session, batch)
                            batch = []
                    if batch:
                        _upsert_loop_trim_batch(batch_session, batch)
            finally:
                batch_session.close()

        elapsed = time.monotonic() - t0
        console.print(
            f"[green]Done: {len(paths):,} loops in {elapsed:.0f}s[/green] -- "
            f"[cyan]{n_trimmed:,} with trim point[/cyan] . "
            f"[dim]{n_skipped:,} no silence . {n_err} errors[/dim]"
        )
    finally:
        session.close()


def _event_profile_job(item):
    sample_id, path = item
    import os as _os
    from fourier.analysis.events import event_profile
    if not path or not _os.path.exists(path):
        return sample_id, None
    return sample_id, event_profile(path)


@steps.command("events")
@click.option("--workers", default=6, type=int, show_default=True, help="Worker processes")
@click.option("--force", is_flag=True, help="Re-profile samples already done")
@click.option("--limit", default=0, type=int, help="Cap at N samples (0 = all)")
@click.option("--max-dur", "max_dur", default=60.0, type=float, show_default=True,
              help="Skip files longer than this (seconds); long files are never one-shots")
def enrich_events(workers, force, limit, max_dur):
    """
    Count the separate sound events in each sample (sample-chain detection).

    A one-shot is one event; a velocity or note ladder laid end to end in one file is
    many. Stores n_events, event_regularity and event_echo in sample_features; the
    curation chain guard uses them. Reads audio: the first run covers the whole library,
    later runs only new samples.

    Runs in `fourier tools analyze` (--only events).
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime

    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

    from fourier.db.models import Sample, SampleFeatures
    from fourier.db.session import get_session

    session = get_session()
    try:
        require_samples(session)
        q = session.query(Sample.id, Sample.path).filter(
            (Sample.duration_s == None) | (Sample.duration_s <= max_dur))  # noqa: E711
        if not force:
            q = q.outerjoin(SampleFeatures, Sample.id == SampleFeatures.sample_id).filter(
                (SampleFeatures.id == None) | (SampleFeatures.events_computed_at == None))  # noqa: E711
        q = q.order_by(Sample.id)
        if limit:
            q = q.limit(limit)
        items = _on_this_machine([(i, p) for i, p in q.all()])
        if not items:
            console.print("[green]Every sample already has an event profile.[/green]")
            return
        console.print(f"[cyan]Event profiles: {len(items):,} samples ({workers} workers)...[/cyan]")
        t0 = time.monotonic()
        n_chain = n_err = 0
        batch = []

        def flush():
            now = datetime.utcnow()
            ids = [sid for sid, _ in batch]
            feats = {f.sample_id: f for f in session.query(SampleFeatures)
                     .filter(SampleFeatures.sample_id.in_(ids)).all()}
            for sid, prof in batch:
                f = feats.get(sid)
                if f is None:
                    f = SampleFeatures(sample_id=sid)
                    session.add(f)
                f.n_events = prof["n_events"] if prof else None
                f.event_regularity = prof["event_regularity"] if prof else None
                f.event_echo = prof["event_echo"] if prof else None
                f.events_computed_at = now
            session.commit()
            batch.clear()

        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                      TaskProgressColumn(), TimeRemainingColumn(), console=console, disable=not console.is_terminal) as progress:
            task = progress.add_task("events", total=len(items))
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for sid, prof in ex.map(_event_profile_job, items, chunksize=64):
                    if prof is None:
                        n_err += 1
                    elif prof["n_events"] >= 2 and not prof["event_echo"]:
                        n_chain += 1
                    batch.append((sid, prof))
                    progress.advance(task)
                    if len(batch) >= 1000:
                        flush()
            if batch:
                flush()
        console.print(f"[green]Done: {len(items):,} samples in {time.monotonic() - t0:.0f}s[/green]"
                      f" -- {n_chain:,} with 2+ separate events, {n_err} unreadable")
    finally:
        session.close()


def _key_items(session, all_classes=False, force=False, limit=0) -> list:
    """(id, path) of the samples the key step reads: Fourier's own tonal ones
    (metadata/resolve.py KEY_CANDIDATE_SQL), whatever Sononym says; every sample with
    all_classes. Not yet read: no key_confidence (a sample read as unpitched keeps a
    confidence and no key, so it isn't read again)."""
    from sqlalchemy import text

    from ..metadata.resolve import KEY_CANDIDATE_SQL
    where = ["1 = 1" if all_classes else KEY_CANDIDATE_SQL]
    if not force:
        where.append("f.key_confidence IS NULL")
    sql = ("SELECT s.id, s.path FROM samples s JOIN sample_features f ON f.sample_id = s.id "
           f"WHERE {' AND '.join(where)} ORDER BY s.id" + (f" LIMIT {int(limit)}" if limit else ""))
    return [(i, p) for i, p in session.execute(text(sql))]


def _read_keys(paths, workers) -> None:
    """Estimate and store the key of each (id, path)."""
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from fourier.analysis.key_detection import estimate_key
    from fourier.db.models import SampleFeatures
    from fourier.db.session import session_scope

    console.print(f"[cyan]Key detection: {len(paths):,} samples, {workers} workers[/cyan]")
    t0 = time.monotonic()
    n_keyed = n_skipped = n_err = 0
    batch: list[tuple[int, dict]] = []
    BATCH_SIZE = 200

    def _process(args):
        sid, path = args
        try:
            return sid, estimate_key(path)
        except Exception as exc:
            return sid, {"error": str(exc)}

    def _flush(b):
        with session_scope() as s:
            for sid, result in b:
                feat = s.query(SampleFeatures).filter_by(sample_id=sid).first()
                if feat is None:
                    feat = SampleFeatures(sample_id=sid)
                    s.add(feat)
                feat.detected_key   = result.get("detected_key")
                feat.key_confidence = result.get("key_confidence")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_process, args): args[0] for args in paths}
        with console.status("[cyan]Estimating keys…[/cyan]") as status:
            done = 0
            for fut in as_completed(futs):
                done += 1
                sid, result = fut.result()
                if "error" in result:
                    n_err += 1
                else:
                    batch.append((sid, result))
                    if result.get("detected_key"):
                        n_keyed += 1
                    else:
                        n_skipped += 1
                if len(batch) >= BATCH_SIZE:
                    _flush(batch)
                    batch.clear()
                if done % 50 == 0:
                    status.update(
                        f"[cyan]{done:,}/{len(paths):,} — "
                        f"{n_keyed:,} keyed, {n_skipped:,} unpitched[/cyan]"
                    )

    if batch:
        _flush(batch)

    elapsed = time.monotonic() - t0
    console.print(
        f"[green]Done: {len(paths):,} samples in {elapsed:.0f}s[/green] — "
        f"[cyan]{n_keyed:,} keyed[/cyan] · "
        f"[dim]{n_skipped:,} unpitched · {n_err} errors[/dim]"
    )


@steps.command("key")
@click.option("--limit",   default=0,  type=int,  help="Limit to N samples (0 = all)")
@click.option("--workers", default=4,  type=int,  help="Parallel worker threads")
@click.option("--force",   is_flag=True,          help="Re-compute even if already enriched")
@click.option("--all-classes", is_flag=True,      help="Process all classes, not just pitched samples")
def enrich_key(limit, workers, force, all_classes):
    """
    Detect the musical key of tonal samples using Krumhansl-Schmuckler profile matching.

    By default only processes the samples Fourier's own analysis calls tonal (a pYIN root,
    or mostly harmonic with a pitch focus: metadata/resolve.py), with or without Sononym,
    which saves significant runtime on percussion-heavy libraries. Use --all-classes to
    process everything.

    Runs in `fourier tools analyze` (--only key).
    """
    from fourier.db.session import get_session

    session = get_session()
    try:
        require_samples(session)
        paths = _on_this_machine(_key_items(session, all_classes, force, limit))
    finally:
        session.close()

    if not paths:
        console.print("[yellow]No samples to process.[/yellow]")
        return
    _read_keys(paths, workers)


def _root_job(item):
    """(id, MIDI pitch or None) by pYIN for one (id, path): curation's own detector
    (packs/curate._detect_pitch), whose cache a build then reads."""
    sample_id, path = item
    import os as _os
    if not path or not _os.path.exists(path):
        return sample_id, None, False
    from fourier.packs.curate import _detect_pitch
    try:
        return sample_id, _detect_pitch(path), True
    except Exception:
        return sample_id, None, False


def own_pending(session) -> int:
    """Samples whose pYIN root `--only own` hasn't read yet (metadata/resolve.py's
    candidates)."""
    from sqlalchemy import text

    from ..metadata.resolve import ROOT_CANDIDATE_SQL
    try:
        return session.execute(text(
            "SELECT COUNT(*) FROM samples s JOIN sample_features f ON f.sample_id = s.id "
            f"WHERE {ROOT_CANDIDATE_SQL} AND f.own_root_at IS NULL")).scalar() or 0
    except Exception:
        return 0


@steps.command("own")
@click.option("--workers", default=6, type=int, show_default=True, help="Worker processes")
@click.option("--force", is_flag=True, help="Re-read samples already done")
@click.option("--limit", default=0, type=int, help="Cap at N samples (0 = all)")
def enrich_own(workers, force, limit):
    """
    Fourier's own readings the other steps don't make, with or without Sononym: the root
    note of each tonal one-shot by pYIN (own_root_midi), then the key of each sample Fourier
    calls tonal that has none yet. The tempo, onsets, harmonic share and events come from
    the librosa and events steps; metadata/resolve.py reads them all.

    Reads audio (pYIN is slow): the first run covers the library's tonal one-shots, later
    runs only new ones. Nothing a build picks changes: with Sononym the build uses
    Sononym's readings, and without it the fallback chain.

    Runs in `fourier tools analyze` (--only own).
    """
    import time
    from concurrent.futures import ProcessPoolExecutor
    from datetime import datetime

    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
    from sqlalchemy import text

    from fourier.db.models import SampleFeatures
    from fourier.db.session import get_session

    from ..metadata.resolve import ROOT_CANDIDATE_SQL

    session = get_session()
    try:
        require_samples(session)
        sql = ("SELECT s.id, s.path FROM samples s JOIN sample_features f ON f.sample_id = s.id "
               f"WHERE {ROOT_CANDIDATE_SQL}" + ("" if force else " AND f.own_root_at IS NULL")
               + " ORDER BY s.id" + (f" LIMIT {int(limit)}" if limit else ""))
        items = _on_this_machine([(i, p) for i, p in session.execute(text(sql))])
        if not items:
            console.print("[green]Every tonal one-shot already has Fourier's own root.[/green]")
        else:
            console.print(f"[cyan]Own roots (pYIN): {len(items):,} samples ({workers} workers)...[/cyan]")
            t0 = time.monotonic()
            n_pitched = n_err = 0
            batch = []

            def flush():
                now = datetime.utcnow()
                feats = {f.sample_id: f for f in session.query(SampleFeatures)
                         .filter(SampleFeatures.sample_id.in_([sid for sid, _m in batch])).all()}
                for sid, midi in batch:
                    f = feats.get(sid)
                    if f is not None:
                        f.own_root_midi, f.own_root_at = midi, now
                session.commit()
                batch.clear()

            with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(),
                          TaskProgressColumn(), TimeRemainingColumn(), console=console, disable=not console.is_terminal) as progress:
                task = progress.add_task("own", total=len(items))
                with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
                    for sid, midi, ok in ex.map(_root_job, items, chunksize=16):
                        if not ok:
                            n_err += 1          # unreadable: tried again next time
                        else:
                            n_pitched += midi is not None
                            batch.append((sid, midi))
                        progress.advance(task)
                        if len(batch) >= 500:
                            flush()
                if batch:
                    flush()
            console.print(f"[green]Done: {len(items):,} samples in {time.monotonic() - t0:.0f}s[/green]"
                          f" -- {n_pitched:,} with one clear pitch, {n_err} unreadable")
        keys = _on_this_machine(_key_items(session))
    finally:
        session.close()
    if keys:
        _read_keys(keys, max(1, workers))


@steps.command("sound")
@click.option("--force", is_flag=True, help="Relabel samples already done")
def enrich_sound(force):
    """
    The sound model's labels (provider fourier:sound, metadata/sound.py): a category and
    one-shot or loop, each with its probability, for every sample with a CLAP embedding,
    from the weights trained on this library (<home>/sound_model.npz, or
    $FOURIER_SOUND_MODEL). Without Sononym it first trains them when there are none yet and
    the library's own names label enough samples, or again when those have grown a lot
    (metadata/train.maybe_train; off with SOUND_TRAIN = false). Labelling is a matrix
    multiply over the embeddings: new samples only, every sample when the weights changed.
    Without weights, or for embeddings from another CLAP model than the weights', it labels
    nothing (and removes a removed model's labels).

    Runs in `fourier tools analyze` (--only sound).
    """
    from fourier.db.session import get_session

    from ..metadata import sound
    from ..metadata.train import maybe_train

    say = lambda m: console.print(m, markup=False, highlight=False, soft_wrap=True)
    session = get_session()
    try:
        maybe_train(session, log=say)
        try:
            mdl = sound.model()
        except sound.SoundModelError as e:
            raise click.ClickException(str(e)) from None
        got = sound.label(session, mdl, force=force)
        session.commit()
    finally:
        session.close()
    if got["model"] is None:
        console.print("No sound model: nothing to label.", markup=False, highlight=False)
        return
    other = (f"; not labelled: {got['other_clap']:,} with an embedding from another CLAP model "
             f"than its weights'" if got["other_clap"] else "")
    console.print(f"Sound model {got['model']}: {got['labelled']:,} samples labelled, "
                  f"{got['kept']:,} already done{other}.", markup=False, highlight=False)


@steps.command("status")
def enrich_status():
    """
    How far each analysis step has got over the samples it covers (its eligible samples:
    loop trim the loops, the keys the samples Fourier calls tonal, the pYIN roots the tonal
    one-shots), in this config's library (another library's samples in the same database
    left out, metadata/rows.outside_library).

    Shown by `fourier tools analyze --status`.
    """
    from sqlalchemy import text

    from fourier.db.session import get_session

    from ..metadata.resolve import KEY_CANDIDATE_SQL, ROOT_CANDIDATE_SQL
    from ..metadata.rows import _id_list, outside_library

    session = get_session()
    try:
        other = outside_library(session)
        scope = f" AND s.id NOT IN ({_id_list(other)})" if other else ""
        q = lambda sql: {i for (i,) in session.execute(text(sql))}
        base = ("SELECT s.id FROM samples s LEFT JOIN sample_features f ON f.sample_id = s.id "
                "WHERE 1 = 1" + scope)
        every = q(base)
        with_son = sononym_classifies(session)
        derived_of = q(base + " AND s.id IN (SELECT sample_id FROM sononym_meta)") if with_son else every
        if with_son:
            loops = q(base + " AND s.id IN (SELECT sample_id FROM sononym_meta WHERE classes LIKE '%Loop%')")
        else:
            from fourier.db.models import Sample
            loops = {smp.id for smp in session.query(Sample).filter(Sample.id.in_(every))
                     if _fallback_loop(smp)} if every else set()
        tonal = q(base + f" AND {KEY_CANDIDATE_SQL}")
        roots = q(base + f" AND {ROOT_CANDIDATE_SQL}")

        def done(col, among):
            return len(q(base + f" AND f.{col} IS NOT NULL") & among)

        table = Table(title=f"Analysis of this library ({len(every):,} samples)", box=box.SIMPLE)
        table.add_column("Step", style="bold")
        table.add_column("Command", style="dim")
        table.add_column("Done", justify="right")
        table.add_column("Coverage", justify="right")
        table.add_column("Covers", style="dim")

        def row(step, cmd, n, of, covers):
            table.add_row(step, cmd, f"{n:,} of {of:,}", f"{100 * n / of:.0f}%" if of else "-", covers)

        row("Derived features", "fourier tools analyze --only derived", done("derived_computed_at", derived_of),
            len(derived_of), "the samples Sononym analyzed" if with_son else "every sample")
        row("Audio features", "fourier tools analyze --only librosa", done("mfcc_mean", every), len(every),
            "every sample")
        row("CLAP embeddings", "fourier tools analyze --only clap", done("clap_embedding", every), len(every),
            "every sample")
        row("Loop trim", "fourier tools analyze --only loop-trim", done("trim_computed_at", loops), len(loops),
            "the loops")
        row("Keys", "fourier tools analyze --only key", done("key_confidence", tonal), len(tonal),
            "the samples Fourier calls tonal")
        row("Root notes (pYIN)", "fourier tools analyze --only own", done("own_root_at", roots), len(roots),
            "the tonal one-shots")
        from ..metadata import sound
        try:
            mdl = sound.model()
        except sound.SoundModelError:
            mdl = None
        if mdl is not None:            # the sound model's labels, over what its weights can read
            readable = {int(i) for i, cm in session.execute(text(
                "SELECT sample_id, clap_model FROM sample_features WHERE clap_embedding IS NOT NULL"))
                if mdl.compatible(cm)} & every
            marked = q("SELECT sample_id FROM descriptors WHERE provider = 'fourier:sound' AND name = "
                       f"'analysed' AND value = {mdl.marker!r}")
            row("Sound model", "fourier tools analyze --only sound", len(marked & readable), len(readable),
                "the samples with a CLAP embedding")
        console.print(table)
        if other:
            console.print(f"{len(other):,} samples of other libraries in this database aren't counted.",
                          markup=False, highlight=False)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# fourier tools analyze (and the second stage of fourier build)
# ---------------------------------------------------------------------------
# in order: each step reads what the ones before it wrote
ANALYZE_STEPS = ("derived", "librosa", "clap", "events", "quality", "bpm-fix", "loop-trim", "own",
                 "key", "sound")


def run_analysis(ctx, only=(), workers=None, force=False, download=False) -> None:
    """Every analysis step in order (or only these), each on the samples it hasn't done yet
    (all of them with force), recording each step's rate for the estimates."""
    import time

    from ..timings import record_analyze
    from .setup import pending_analysis
    ctx.meta[DOWNLOAD_KEY] = download
    for step in ANALYZE_STEPS:
        if only and step not in only:
            continue
        cmd = steps.commands[step]
        kw = {"force": force} if any(p.name == "force" for p in cmd.params) else {}
        if workers and any(p.name == "workers" for p in cmd.params):
            kw["workers"] = workers
        if step == "clap":
            kw["build_index_flag"] = True
        console.print(f"[bold]{step}[/bold]")
        before, t0 = pending_analysis().get(step), time.perf_counter()
        ctx.invoke(cmd, **kw)
        after = pending_analysis().get(step)
        if before is not None and after is not None and not force:
            used = kw.get("workers") or next((p.default for p in cmd.params if p.name == "workers"), 1)
            record_analyze(step, time.perf_counter() - t0, before - after, used or 1)  # type: ignore[arg-type]


@tools.command("analyze", short_help="Analyze the library's audio (incremental).")
@click.option("--only", "only", multiple=True, type=click.Choice(ANALYZE_STEPS),
              help="Just this step (repeatable)")
@click.option("--workers", default=None, type=int, help="Parallel workers for the steps that take them")
@click.option("--force", is_flag=True, default=False, help="Recompute samples already done")
@click.option("--status", is_flag=True, default=False, help="Show how far each step has got, and stop")
@click.option("--download", is_flag=True, default=False,
              help="Download cloud-only library files first (otherwise they're skipped and counted)")
@click.pass_context
def analyze(ctx, only, workers, force, status, download):
    """Analyze the audio curation reads, step by step (`fourier build` does this after its
    scan). Incremental: only samples not done yet. The steps: derived features (from
    Sononym's), librosa features, CLAP embeddings and their index, sample-chain events,
    quality, Sononym's folded tempo, loop trims, Fourier's own roots (pYIN) and keys, and the
    sound model's labels (without Sononym, trained on the library first when it has none).
    Fourier's own analysis runs with or without Sononym; `--only own` fills it in on a
    library analyzed before it existed.
    Files a cloud drive keeps only online are skipped unless --download is given.

    \b
      fourier tools analyze
      fourier tools analyze --only clap
      fourier tools analyze --only own
      fourier tools analyze --status
    """
    if status:
        from ._app import read_only_config
        read_only_config(ctx)
        ctx.invoke(enrich_status)
        return
    from ..demo import REAL_ONLY, active
    if active():
        console.print(REAL_ONLY.format(what="analyzing audio"), markup=False, highlight=False,
                      soft_wrap=True)
        raise SystemExit(1)
    run_analysis(ctx, only, workers, force, download)


@tools.command("train", short_help="Train the sound model on your library (it stays on this machine).")
@click.option("--out", default=None, type=click.Path(dir_okay=False),
              help="Where the weights go (default: <Fourier home>/sound_model.npz, where builds look)")
@click.option("--report", default=None, type=click.Path(dir_okay=False),
              help="Where the training report goes (default: beside the weights)")
@click.option("--clap-prior", is_flag=True, default=False,
              help="Pull the weights toward the taxonomy's CLAP prompts instead of zero")
@click.option("--keep", is_flag=True, default=False,
              help="Keep the model even when it isn't reliable on the packs it didn't learn from")
def train_sound(out, report, clap_prior, keep):
    """Train the sound model on your library: a small classifier over each sample's CLAP
    embedding and Fourier's own measurements, taught by how your sample makers named their
    files and folders (and your ratings), never by Sononym's or Live's analysis. About a
    minute; it reads the database only. Without Sononym a build trains one by itself once
    your library's names label enough samples, and places with it what no name rule
    recognizes; this trains it now (or again). A model is kept only when it's reliable on the
    packs it didn't learn from (--keep keeps it anyway). The weights and the report stay in
    your Fourier home: they're learned from your library, whose licences are yours.

    \b
      fourier tools train
      fourier tools analyze --only sound    # label the library with it (a build does this)
    """
    from fourier.db.session import get_session

    from ..metadata.train import summary_lines, train_and_keep

    say = lambda m: console.print(m, markup=False, highlight=False, soft_wrap=True)
    session = get_session()
    try:
        say("Training the sound model on your library (about a minute)...")
        try:
            got = train_and_keep(session, out, keep_anyway=keep, report=report, clap_prior_=clap_prior)
        except SystemExit as e:
            raise click.ClickException(f"no sound model: {e}") from None
    finally:
        session.close()
    for line in summary_lines(got):
        say(line)
    if not got["kept"]:
        return
    if out:
        say(f"Builds use {out} when $FOURIER_SOUND_MODEL names it.")
    else:
        say("The next build (or `fourier tools analyze --only sound`) labels the library with it.")
