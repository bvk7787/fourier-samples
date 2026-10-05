# Fourier Samples

Fourier Samples (the `fourier` command) curates a large sample library into one **master** of
`CATEGORY/family/file.wav` folders, then renders that master for each hardware sampler. This
file is for coding agents (and people) working on the code; [README.md](README.md) is for
users, with [docs/start.md](docs/start.md) for someone new to Terminal, a task guide in
[docs/guide.md](docs/guide.md) and its reference in
[docs/usage.md](docs/usage.md) (commands, configuration, styles and knobs);
[skills/fourier/SKILL.md](skills/fourier/SKILL.md) is for a user's coding assistant,
[docs/device-profiles.md](docs/device-profiles.md) covers drafting a device profile, and
[docs/curation.md](docs/curation.md) holds the curation rules (categories, families, budgets,
naming, routing, loudness, ratings). `fourier why --rules` prints the routing precedence.

## Ground rules

- **The library's files are never changed.** Fourier Samples writes the master, renders, releases
  and its own home (`$FOURIER_HOME`, default `~/.fourier`); only `fourier tools import-folder` adds
  copies to a library folder. Point in-progress code at a separate home
  (`FOURIER_HOME=~/.fourier-dev`) when you have a library you care about.
- **Device facts come from the manuals, not memory.** Before stating or changing a device limit
  (channels, formats, sample rates, file, folder and path rules), look it up in the device's
  manual and cite the PDF page with a verbatim quote. Every value in `config/devices/*.yaml` is
  `{value, cite: <id>}` or `{value, status: convention | unverified}`, and a profile's
  `manual:` names the PDF it quotes (title, version, url, sha256)
  (`src/fourier/devices/loader.py`). `tests/test_device_citations.py` checks each quote against
  its page when `FOURIER_MANUALS_DIR` holds the PDF. See docs/device-profiles.md.
- **Tunables.** Every module-level setting in the curation modules (`curate_config.py`,
  `curate.py`, `naming.py`, `sets.py`) is `NAME = _tunable("NAME", default)` and needs a class
  line in `config/tunables.yaml` (correctness, knob, taste, rules, engine, taxonomy, vendor,
  library, device, internal); `tests/test_tunables_inventory.py` fails without one, and
  `tests/data/tunables_snapshot.json` pins the defaults (update it only for a deliberate change).
  `$FOURIER_RESOLVED_CONFIG` replaces a default where it's defined, so derived values, default
  arguments and worker processes follow.
- **Config layers** (`src/fourier/layers.py`): defaults < preset (+ its `extends`) < overlay <
  `fourier.toml [advanced]` < `fourier --set KEY=VALUE`, each file's knobs
  (`src/fourier/knobs.py`) applied before its `[advanced]`. `fourier.toml` comes from
  `--config`, `$FOURIER_CONFIG`, `./fourier.toml` or `~/.config/fourier/fourier.toml`
  (`FOURIER_CONFIG=none` for none). A `fourier.toml` that names no preset gets `balanced`
  (`layers.DEFAULT_PRESET`, and a build says so); with no `fourier.toml` at all the code's
  defaults apply, which are `balanced`'s too (balanced.yaml resolves to no overrides).
  `breaks-acid` sets its budgets, drum-loop exclusions and no surplus round in `advanced:`, and
  the other genre styles `extend` it, so each resolves as it did when breaks-acid was the code's
  defaults; the synthetic golden and the demo name breaks-acid. `fourier config show [--all]` and `fourier config
  explain KEY` print what resolved and from where (without leaving a resolved file in the
  home); every manifest records `tunables_hash` and `overrides`.
- **No one's library in the code.** Rules that only match particular packs or vendors are
  class `library`: empty in the code, set by a user's overlay (`overlay = "my-library.yaml"`).
  A category only one library needs is added by its overlay (`add_categories`,
  `curate_config.ADDED_CATEGORIES`, `taxonomy.with_added`; docs/curation.md, "Adding a
  category"), never by code: the curation modules read the taxonomy through
  `curate_config.taxonomy_view`. Tests use invented vendors (Acme, Northwind, ...).
- **Where things are** comes from `fourier.toml` (`src/fourier/places.py`): `library`,
  `devices`, `sononym_db` and `[output]` (`master`, `renders`, `publish`), each with an
  environment override. Curation reads a sample's library path (`places.library_rel`:
  vendor/pack/...). Tests set `FOURIER_LIBRARY=SampleLibrary` (conftest.py).
- **The scan** (`cli/ingest.py` `run_scan`): Sononym's sync when it's there, else a walk of the
  library folders (`ingest/walk.py`, `importer.scan_filesystem`; with Sononym only under `tools
  scan --walk`), then Live's tags. The audio formats are one list (`ingest/formats.py`). The
  walk marks samples it didn't find missing (`missing_files`), which `rows.sample_select` leaves
  out only when any are marked; a build stops on a missing or empty library folder, on many of
  the master's sources gone while still pickable, and on a pick it couldn't export
  (`cli/build.py`). A Sononym library's scan never walks (without `--walk`), so its builds
  read the database as before; the stops act only where a build would otherwise come out short.
  The walk also matches moves (`importer._match_moves`: same content at a new path, or back
  where it was), which take their analysis and ratings along (`ratings.rekey_paths`), records
  the files it can't read (`scan_errors`, doctor's "unreadable files") and reports a folder
  symlink whose target is gone (`walk.unavailable`; a build stops on it unless
  `--allow-missing`; doctor FAILs it, `importer.unavailable_links`). A move's old row records
  where it went (`missing_files.moved_to`): it isn't counted missing (scan, `db-stats
  --missing`), and the previous build's picks follow it (`curate._moved_paths`). Another
  config's library may share the database: `rows.outside_library` leaves its samples out of
  sample_select, the scale count, doctor, search, `analyze --status` and the vendor layout,
  only when the database holds samples under none of the configured folders; when it holds
  none under them and a configured folder is on disk, the library isn't scanned yet
  (`rows.library_unscanned`: every sample is left out). `builddiff.latest_archived` picks this
  library's last build. Every manifest records its `library` (`places.library_record`); a
  build into a master whose recorded library shares no folder with the config's stops before
  anything (`places.master_library_problem`, a doctor FAIL), and `setup --to` a config other
  than the default one gives it `[output]` folders named after it (`setup.default_outputs`).
- **Metadata.** Curation reads samples through `src/fourier/metadata/rows.py`; code in
  `packs/` never names a provider's tables. Providers (`metadata/providers.py`): `sononym` and
  `ableton` when their data is there or `fourier.toml` names them, else the built-in `path`
  (generic words in folder and file names, `config/providers/path.yaml`; never a vendor or
  pack; one-shot / loop only from the file's name and folder, `shadow.shape_labels`) and
  `audio` providers (`metadata/shadow.py`; a change to their rules in code bumps
  `_PATH_RULES` / `_AUDIO_RULES` so the labels rebuild). With Sononym they route nothing (only
  the agreement report reads them), and without Live `NAME_TAGS` words stand in for its tags
  (`curate._name_tags`): changes there can't move a build that has Sononym and Live. Without
  Sononym, a loop guess from the audio alone yields to a sound named in the file's own name or
  folder (`shadow.audio_label`), and a loop's DRUMLOOPS vote needs drums
  (`curate._fallback_loop_votes`, `_drum_loop_evidence`; compute_homes and why); the phrase
  rule leaves such a loop to the drum loops (`_is_phrase(r, fallback)`, `_providers_drum_loop`)
  and DRUMLOOPS' drum-tag requirement takes the same evidence (`_select_records(fallback=)`).
  The taxonomy names canonical labels (`metadata/vocab.py`); a provider's own names map into
  them in `config/providers/*.yaml`.
- **Libraries without Sononym or Live, or laid out differently** (each gated so a build with
  Sononym and Live, on preset breaks-acid with first-folder vendors, is unchanged;
  tests/test_other_libraries.py):
  without Sononym the path words split letters from digits and camelCase
  (`shadow.words_of`: "kick01", "HiHat58", "BD01"), `config/providers/path.yaml` knows the
  common abbreviations, and the `words` knob adds a user's words (`PATH_WORDS`,
  `shadow.path_words`). A sample no name rule recognized but with a shape is homed by CLAP at
  the nearest category that it clears by a margin (`curate._clap_home`, recorded in the why
  log as `clap_homed`); `curate.unrecognized` and `fourier why --unrecognized` list the rest.
  Tempos a name, a number-only folder or a WAV's ACID chunk states (`curate._fallback_tempos`,
  `ingest/chunks.py`, read by the walk into `samples.acid_bpm/acid_beats/root_note`) are tried
  only when the usual chain finds nothing, except the ACID tempo (and a name's "bpm120"),
  which comes before librosa's estimate (`curate._resolve_tempo`). The vendor the per-vendor cap counts is
  `packs/vendors.py` (`vendors` knob: "first-folder", the code's default, or "auto", which
  setup writes: per library folder, vendor/pack, by sound type, umbrella or flat). The tempo
  knob derives the fold window from its range (`fold_window`; `TEMPO_FOLD_RANGE` only for a
  range narrower than its top octave, so 85-180 sets nothing) and `fold = "off"` turns
  folding off. Genre presets (`hiphop-lofi`, `house-techno`, `ambient-cinematic`, `trap`)
  carry a `description:` that setup and `config show` print, and extend `breaks-acid`. `sets = "off"` skips 00_KITS / 00_SLICE; `devices` lowers `SLICE_MAX` and
  `STEM_MAX` only for a profile that sets `max_slices` / `max_name_length` below them; a
  profile whose limits leave the names too little room sizes nothing (`knobs.limit_problem`:
  a doctor FAIL and a build stop, not a config error). Also without Sononym only: the path
  label `keys` (PIANO's, `curate._keys_labeled`), sustained sounds as one-shots
  (`shadow.audio_class`; a shapeless held sound offered the one-shot categories,
  `curate._held`, and so is a held, slow-onset, tonal "loop" no stated tempo fits,
  `curate._held_loop`), loops CLAP hears as the other loop category re-homed either way between
  PHRASES and the drum loops (`curate._clap_rehome`), busy loops in a folder named only by a
  tempo or "Loops" counted as drums (`_anonymous_loop`), a tempo from a loop's length
  (`_length_tempo`), recognized samples no category took listed by `why --unrecognized`
  (`placed_nowhere`), a bare note letter
  ending a name (`_BARE_NOTE`, "_from-<note>" for a second C), shakers in a percussion folder
  (`rules._drum_named_in`; their path label is `perc.hand`), loops at other tempos never
  near-duplicates (`_tempo_apart`) and "musical" held to `MUSICAL_HAR_MIN` and a pitch focus
  (`naming.trait_ok`). A render for a device not in `devices` cuts names per family folder
  (`render._fit_all`; `render --check` says what render does).
- **The taxonomy is `config/taxonomy.yaml`** (`src/fourier/taxonomy.py`): categories, kinds,
  labels, roles, bands and CLAP prompts. Prompts are compared as text by the CLAP text cache:
  change one on purpose.
- **Small pools.** A category's pool from fewer than three vendors isn't vendor-capped, and
  neither the cap nor the near-duplicate prune takes a pool below the category's folder
  minimum (`_cap_vendor_share`, `_prune_near_dups`: `keep`; an additive build, `--base`, caps
  as before). Both only act where a build would otherwise leave the category empty, so a
  library with many vendors per category builds the same. Likewise the gates' floor: a
  category its CLAP gate or harmonicity gate would leave below the minimum (TooFewSamples)
  readmits the loops that fail only one of them, best CLAP margin first, up to it
  (`_readmit_gated`; never a CLAP score under 0 or under the anti-prompts', `_clap_sane`; not
  in an additive build). Every build writes why it left each candidate out (each gate apart,
  with the file's value and the threshold), and each gated candidate's CLAP scores, to
  `$FOURIER_HOME/why/<master key>/` (`packs/why_log.py`, read by `fourier why`); never into the
  master or its manifest.
- **Library scale** (`packs/scale.py`, the `scale` knob, docs/curation.md 3.5): a library with
  fewer than `LIBRARY_PER_MASTER` usable samples for each budgeted file gets a master scaled by
  f = samples / (that x the budgets' total). The build passes `scale` to `build_taxonomy` only
  when f < 1, and everything it changes (budgets that follow the pool, the minimum and the
  floors, folder counts and sizes, tempo-band minimums, no surplus round) sits behind that, so
  a library at f = 1 builds exactly as before; the manifest records `scale` and each scaled
  `budget`, which verify reads. `files = N` and `scale = "off"` turn it off; the synthetic
  golden and the demo pin `scale = "off"`, and `synthetic_build.py --scale` builds it scaled.
- **Doctor's levels** (`cli/setup.py`): OK, WARN, NEXT (what the first build does itself: scan,
  analyze, index, the model download) and FAIL (what stops a build); doctor exits 1 only on a
  FAIL. It WARNs on a CUDA build of PyTorch on Linux without an NVIDIA GPU (`torch_has_cuda`;
  `fourier setup` switches it).
- **Installing and updating** (`cli/setup.py`): setup installs the `[clap]` extra with the
  installer this copy runs under (a uv tool is reinstalled with `--with`, which `uv tool
  upgrade` keeps), the CPU build of PyTorch on Linux without an NVIDIA GPU (`--torch-backend
  cpu`; a uv too old for it, before `UV_TORCH_BACKEND`, is offered `uv self update` (never
  under `--yes`), else PyTorch's CPU index with `--index-strategy unsafe-best-match`). uv's
  receipt keeps no `--torch-backend`, so in a uv tool setup then pins the CPU build it got
  (`torch==<v>+cpu` with the CPU index, both kept in the receipt: `pin_cpu_torch`, run
  quietly); `fourier setup` again updates PyTorch and pins the new one. `$FOURIER_HOME/install.json` records that setup
  installed CLAP, so doctor can say a `uv tool install --reinstall` removed it
  (`receipt_dropped_clap`). A failed install shows the installer's error and one next step
  (`install_failed`). `build --dry-run` opens the database read-only (`db.session.read_only`).
- **CLAP** loads from the pinned revision's folder in the Hugging Face cache with
  `HF_HUB_OFFLINE=1` and `DISABLE_SAFETENSORS_CONVERSION=true` once it's downloaded
  (`clap_features.cached_snapshot`); only setup or the first use downloads it. Never change
  `CLAP_REVISION`: embeddings are only comparable within one revision.
- **The CLI is `src/fourier/cli/`**: one module per command group; `cli/_app.py` holds the root
  group, its help sections (`SECTIONS`: `fourier --help` lists the commands under them) and the
  `tools` group, `cli/setup.py` the setup wizard, demo and doctor, `fourier/demo.py` the demo's
  sandbox. There are no hidden aliases. `tests/test_cli_tree.py` pins every command, option
  and help text; after a deliberate change run `python tests/test_cli_tree.py --write`. The
  root command only points the process at its database (`db.session.use_db`); the first query
  opens it, so `--help` and `doctor` on a new machine create nothing. `config` and `open` run
  with a config that doesn't load (`_app.CONFIG_ERROR_OK`: `config edit` is how it gets fixed;
  the others stop on it with `config_error_stop`); `layers._read` reads a fourier.toml typed
  with a text editor's smart quotes. `platforms.open_path` opens a folder in Finder or a file in
  the text editor (`fourier open`, `config edit`). An exception no command handles ends in
  one plain line and a traceback file in `<home>/logs/error-*.txt`
  (`_app.SectionedGroup.invoke`, `unhandled`; `fourier -v` or `FOURIER_TRACEBACK=1` re-raise).
  `fourier why` prints `why.format_short` (one sentence, `why.headline` and `PLAIN`) unless
  `--detail` (`format_why`). `fourier open report` writes `packs/report.py`'s page into
  `<home>/reports` (never the master): file:// players, the why log's left-out counts, rating
  buttons that export a `review import` CSV. `render --release vN --new-only` copies the lock's
  entries first locked by vN into `<render>-new-in-vN`. Setup's size, processing and category
  questions write knob lines (`setup.SIZES`, `AS_IS`); `knobs.PLAIN_HELP` is `config explain`'s
  plain text. In a demo folder (`demo.active`) `fourier build` builds with `--no-scan` and
  `search` by sound and `tools analyze` refuse: the demo's analysis and CLAP are stand-ins.
- **A quiet build.** On a terminal `fourier build` is quiet (`cli/quiet.py`): from the Build
  stage its output, worker processes' too, goes to `<home>/logs/build-*.log` and the terminal
  gets a spinner, then a summary read from the manifest and the log (or the log's last lines
  when it stops). Piped, or with `--verbose`, `fourier -v` or `FOURIER_VERBOSE=1`, it prints
  everything, which is what the tests, the goldens and the harness see. Keep the lines the
  summary reads stable: `verify: `, `WARN `, `... left empty`, `no rule recognized (top
  folders`, `New master in place at `.
- **What differs between systems** is in `src/fourier/platforms.py` (cloud-synced folders,
  cloud-only files, mount points, rsync or the same copy in Python). Keep
  system-specific commands out of the rest of the code. A build downloads its cloud-only picks
  before the first read of a source (`safety.ensure_local`) and stops if one won't download.
- **Database changes**: a nullable column a model gains is added by `init_db`; anything more is
  a numbered, additive migration in `src/fourier/db/migrations.py`. Migration 2 (schema 2) set
  the analysis apart by source (below); `tests/data/schema_v1` holds the tables before it, so
  tests can build an older database and migrate it (`synthetic_build.py --schema-v1`).
- **Data model: one source per column** (`src/fourier/db/models.py`; docs/curation.md,
  "Sources and resolution"). `samples` holds the file's facts (path, header, hash: read by
  Sononym's scan or the walk, one meaning either way) and what its own chunks state
  (`acid_bpm`, `acid_beats`, `root_note`). `sononym_meta` and the `sononym` labels and
  descriptors are Sononym's; `samples.ableton_tags` and the `ableton` labels are Live's; the
  `path` and `audio` labels and descriptors are Fourier's readings of the name and the audio.
  `sample_features` holds Fourier's own measurements (librosa, events, quality, CLAP, key,
  loop trim, the pYIN root `own_root_midi`), Sononym's folded tempo (`sononym_bpm_folded`),
  the legacy Sononym-derived columns of the derived step (`sub_weight`, `transient_score`,
  `loop_confidence`, `is_pitched`, `bpm_reliable`, `timbral_norm`, `spectral_balance`,
  `pitch_stability`, `attack_class`, `drum_subtype`: written only from a Sononym row, NULL
  without one) and the legacy `bpm_corrected` (no longer written or read). A new value goes
  beside its source's: never a column whose meaning depends on which providers are there.
  `rows.field_source` names each row field's source (tests/test_resolve.py checks every field
  has one). `src/fourier/metadata/resolve.py` chooses what the build uses (tempo, its
  reliability, one-shot or loop, root) and names its source: with Sononym exactly what
  curation chose before it, without it the fallback chain; the why log records each pick's
  `sources`. Fourier's own analysis runs with or without Sononym (`fourier tools analyze
  --only own`: pYIN roots of tonal one-shots, then the keys of what Fourier calls tonal) and
  picks nothing: `fourier why` shows both readings where they differ ("readings differ:
  tempo 174 (Sononym) / 87 (Fourier's own analysis)") and `fourier tools db-stats
  --disagreements` lists them across the library.
- **The sound model** (`src/fourier/metadata/sound.py`, provider `fourier:sound`;
  docs/curation.md, "The sound model"): a logistic regression over the CLAP embedding and
  Fourier's own measurements (`sound.OWN_FEATURES`) giving each sample a category (the built-in
  ones or "other") and one-shot or loop, with probabilities. Its labels come only from the pack
  makers' names (`sound.training_label`, the path provider's rules in the file's own name and
  folder) and the user's ratings (`sound.rating_label`); nothing of Sononym's or Live's is a
  label or a feature (`sound.FORBIDDEN`; `tests/test_sound_model.py` parses the module and the
  trainer, `metadata/train.py`, all but its `yardstick`, which measures the saved model against
  Sononym for the report afterwards). `fourier tools analyze --only sound` labels incrementally;
  without Sononym `curate.compute_homes` homes by it (`_sound_home`: `SOUND_MODEL_MIN` for what
  no name rule recognized, ahead of the CLAP fallback; `SOUND_MODEL_OVERRIDE_MIN` where only a
  pack's name or a weak word speaks, `_weak_path`; the why log's `sound_homed`), and with
  Sononym it routes nothing (`synthetic_build.py --check --sound`: the golden). Fourier ships no
  weights (they'd be learned from someone's licensed samples): each user's are trained on their
  own library into the Fourier home (`<home>/sound_model.npz` and its report), by the analysis's
  sound step without Sononym (`train.maybe_train`: none yet and `SOUND_TRAIN_MIN_LABELS` across
  `SOUND_TRAIN_MIN_PACKS`, or grown by `SOUND_RETRAIN_GROWTH`; kept only when reliable on the
  held-out packs, `train.train_and_keep` and the `SOUND_KEEP_MIN_*` tunables; off with
  `SOUND_TRAIN = false`) or by `fourier tools train` (about a minute; `python -m
  fourier.metadata.train --db ... --out ...` for another database). `$FOURIER_SOUND_MODEL` names
  other weights (`off` for none). No weights, or weights for another CLAP model or revision than
  a sample's embedding, and the provider labels nothing. Never commit weights or a report.
- **Preparers** (`src/fourier/preparers.py`) are plug-ins, registered under the entry-point
  group `fourier.preparers` and enabled in `fourier.toml`, that add a codec soundfile lacks for
  files with their own extensions (`fourier tools import-folder`). No code that circumvents copy
  protection, here or in a plug-in.
- Releases are immutable. Don't use `publish --force` or `--no-verify` unless asked.

## Checks

The suite never reads the machine it runs on: `conftest.py` clears the `FOURIER_*` variables
(but `TEST_INPUTS`) and sets `$HOME` to an empty folder, so a test that wants a user config,
a Live index or a platform makes it (`monkeypatch` `sys.platform`, `shutil.which`, the CLAP
checks); macOS and Linux must pass the same tests. Each pytest-xdist worker gets its own
home, temp folders and numba cache (`WORKER` in conftest.py), and `-n` distributes with
`--dist loadgroup`: a module whose tests share one sandbox, in order, is marked
`pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("<module>")]`.

```bash
pytest -q -n auto tests                                  # the suite in parallel (no library needed; serially without -n)
pytest -q -n auto -m "not slow" tests                    # ...without the end-to-end runs
pytest -q tests/test_first_run.py                        # a first run, setup to publish, without Sononym or Live
python tests/golden/synthetic_build.py /tmp/syn --check  # the pipeline on 280 generated files vs tests/golden/synthetic-<os>.json
python tests/golden/synthetic_build.py /tmp/syn2 --without sononym,ableton  # without either provider: every category fills
python tests/golden/synthetic_build.py /tmp/syn3 --preset balanced          # a preset builds and passes verify
python tests/golden/synthetic_build.py /tmp/syn5 --scale                     # the style's budgets scaled to the library
python tests/golden/synthetic_build.py /tmp/syn4 --check --interrupt KICKS,PADS  # a resumed build equals an uninterrupted one
python tests/golden/synthetic_build.py /tmp/syn6 --check --schema-v1 --own [--duckdb]  # an older database migrates, then Fourier's own analysis: the golden each time
python tests/golden/synthetic_build.py /tmp/syn7 --check --sound            # a sound model trained and labelled first: the golden still (it routes nothing with Sononym)
ruff check src tests
```

The synthetic golden compares every manifest entry field, category summary and set; `--update`
rewrites it after a deliberate change. With your own library, `scripts/golden_harness.py
--reference <releases/vN>` rebuilds a release from frozen copies of the database, CLAP index,
ratings and caches and compares it exactly (`scripts/golden_compare.py`), without touching the
real home, master, releases or locks. Both build with `--no-scan`: the database as it is, never
scanned or analyzed.

## The workflow

```bash
fourier setup && fourier doctor           # fourier.toml and the CLAP model; then what's missing (OK/WARN/NEXT/FAIL) and how long things take
uv tool upgrade fourier-samples           # an update keeps the CLAP extra and its PyTorch build; `fourier setup` again updates PyTorch
fourier build --all -j 8 [--dry-run]      # scan, analyze what's new, build into <master>.next, verify; synced in only when it all passed
fourier build --all -j 8 --resume         # pick up a build that stopped
fourier build --all --no-scan             # the database as it is (no scan, no analysis): the golden builds
fourier tools scan | analyze [--status]   # the first two stages on their own (incremental)
fourier tools analyze --only own          # Fourier's own roots and keys on a library analyzed before them
fourier tools db-stats --disagreements    # where Sononym's readings and Fourier's own differ, the sound model's too (a report)
fourier tools train                       # train the sound model on the library (into the Fourier home)
fourier verify [--render <device>]        # every curation rule, and in scratch device renders
fourier why "<name>"                      # where a sample landed and which rule put it there, or why the last build left it out
fourier review rate|import|queue|score    # the listening loop: rate a file or import a CSV (no Live needed), or tag in Live's browser and `review ratings`
fourier publish [--dry-run] --notes "..." # <publish root>/releases/v<N>, immutable; --dry-run: what it would change
fourier render <device> [--release vN]    # a device folder; a release records the path lock (--dry-run: what it would change)
fourier sync <device> <volume> [--delete] # an SD-card device
```

`fourier build` (without `--no-scan`) runs `fourier tools scan` and `fourier tools analyze`
first, so it stops before scanning when the CLAP model isn't installed (`fourier setup`
installs it); `--dry-run` reads only.

Builds are deterministic and **stable**: files and folders of the latest archived build stay
while still eligible, so a rule change moves only the files it touches (`FOURIER_NO_STICKY=1`
builds fresh). Each finished category is recorded in `<master>.next/.progress/` for
`--resume`; a build that stops keeps `<master>.next` only when it holds finished categories a
resume would keep, and removes it otherwise. A category the library has too few samples for is
left empty (and named), not a failure. Every build and analysis step records its times in
`<home>/timings.json`, which doctor and `--dry-run` estimate from.

Releases: a release is cut from a verified master and never changes. Rendering a release for a
device locks each file's path on that device; later builds keep locked paths and their audio,
never reuse a retired path, and `fourier render <device> --dry-run` exits non-zero on anything
that would break one (`fourier publish --dry-run`, on a release that would move, change or
remove a released file). Changes after a version is on a device go into the next version
(`fourier build --base vN`).

## Layout

```
src/fourier/
  cli/          the commands (_app.py: the root group, its help sections, tools; setup.py: setup, demo, doctor)
  demo.py       fourier demo: the pipeline on a synthetic library (synthlib.py) in a sandbox
  ingest/       Sononym, Ableton Live tags, folder scans and imports
  metadata/     providers, canonical labels, the rows curation reads, the resolver (resolve.py:
                what the build uses and its source), the sound model (sound.py) and its
                trainer (train.py)
  analysis/     derived, librosa and CLAP features
  packs/        curation (curate, curate_config, naming, sets; rules: the name, tag and path
                predicates routing, verify and why share), verify, render, releases, ratings,
                why, progress, audiocache
  devices/      device profile loader, audio export (conversion, naming)
  places.py platforms.py timings.py knobs.py layers.py settings.py taxonomy.py preparers.py
config/         devices/, presets/, providers/, taxonomy.yaml, tunables.yaml
docs/           start.md (getting started on a Mac, for someone new to Terminal), guide.md
                (the tasks after a first build), usage.md (the README's reference),
                curation.md (the curation rules),
                device-profiles.md (drafting a profile), design-history.md, images/ (the
                README's capture: scripts/capture_demo.py)
skills/fourier/ SKILL.md: a skill for a user's coding assistant (using Fourier, not changing it)
scripts/        golden harness and compare, render audit, manifest diff, tunables inventory,
                the README's demo capture
tests/          the suite; golden/ holds the synthetic build and its goldens
```
