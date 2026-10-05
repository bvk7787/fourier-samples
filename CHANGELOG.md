# Changelog

All notable changes to Fourier Samples are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (before 1.0, a minor version may
change the CLI or the config).

## [Unreleased]

### Changed

- CI runs on Python 3.11 to 3.14 (3.14 on Linux and macOS), measures test coverage on one
  job, and type-checks the source with mypy (a lenient start: the settings and the lines it
  first found are in pyproject.toml and marked in the code).
- The workflows pin each action to a commit, check out without keeping credentials, and an
  OpenSSF Scorecard workflow scores the repository's security practices weekly.
- Windows left the CI matrix: it doesn't work yet (fourier.toml paths, rsync, SQLite file
  locks; [#9](https://github.com/bvk7787/fourier-samples/issues/9)). The `Windows` workflow
  runs the suite there by hand.

## [0.1.0] - 2026-10-04

The first public release.

### Added

- **The command line, for people new to it.** `fourier --help` lists seventeen commands in
  sections (Get started, Build and load, Releases, Look inside, Settings, Listen and rate,
  Advanced). `fourier setup` is a step-by-step wizard (or `--yes` with flags): the sample
  folders (counted, with a warning for cloud-only files), devices and style ("What do you
  make?"), output folders, the config file, the CLAP model (it installs PyTorch and
  Transformers into the running environment with uv or pip, the CPU-only PyTorch on Linux
  without an NVIDIA GPU, and downloads the pinned model), an optional local LLM through Ollama,
  then doctor's checks and the first build. `fourier build` (every category; or one) scans the
  library and analyzes what's new before it builds (`--no-scan` skips both), with stage headers
  and an estimate up front; on a terminal its Build and Verify stages show progress and a
  summary (what it built, what it left empty, verify's result), with everything else in a log
  in `<home>/logs` (`--verbose` prints it all). The steps underneath are `fourier tools` (`scan`, `analyze`,
  `train`, `audit`, `import-folder`, `dedup`, `resolve`, `db-stats`); `publish --dry-run` and
  `render --dry-run` preview what a release or a render would change.
- **Curation.** `fourier build` turns a sample library into one master of
  `CATEGORY/family/file.wav` folders: nineteen categories in play order (an overlay can add
  more), families clustered from CLAP embeddings and named for what they sound like, drum kits
  (`KITS`) and slice-ready loops (`SLICE`), and a manifest recording where every file came from
  and why it's there. Builds are deterministic and stable (a rule change moves only the files
  it touches), resumable (`--resume`), and swapped into place only when every category built
  and verify passed. A category the library has too few samples for is left empty and named,
  not a failure.
- **The master's size follows the library** (`scale = "library"`): a style's budgets are for a
  large library, and a smaller one gets a master in proportion. `fourier doctor`, setup and
  `build --dry-run` say how big it will be. `scale = "off"` and `files = N` keep fixed budgets;
  `size` caps them to fit a device.
- **Export DSP.** DC offset, phase-flipped stereo, clicks at the end, hot starts, long tails,
  per-folder loudness, and notes a few cents off are fixed on the way into the master.
- **Any library layout, with or without other apps.** A build's scan reads Sononym's analysis
  and Ableton Live's auto-tags when they're there, and otherwise walks the library folders.
  Neither app is needed: built-in `path` and `audio` classifiers (folder and file names, WAV
  chunks, and the audio itself) stand in, with your own words (`words`), and `vendors = "auto"`
  reads each folder as vendor/pack folders, folders by sound type, an umbrella of packs or a
  flat folder. Files no name rule recognizes are placed by sound: by CLAP's prompts, and by a
  sound model trained on your own library (below). Fourier's own analysis (librosa features,
  CLAP embeddings, sample-chain events, loop trims, pYIN roots and keys) runs with or without
  Sononym and is stored apart from Sononym's; a resolver records which source each value a
  build uses came from. The library's own files are never changed.
- **A sound model trained on your library** (without Sononym). The first build trains a small
  classifier over each sample's CLAP embedding and Fourier's own measurements, taught by how
  your sample makers named their files and folders and by your ratings (never by Sononym's or
  Live's analysis), once the names label enough samples; it retrains when the library has grown
  a lot, and keeps it only when it's reliable on the packs it didn't learn from; `fourier tools
  train` trains it any time. It places what no name rule recognizes when it's confident, and
  its weights and report stay in your Fourier home. Fourier ships no
  weights: they'd be learned from licensed samples. `fourier setup --no-sound-model` turns it
  off.
- **Cloud-synced libraries.** `fourier setup` and `fourier doctor` count the files a cloud
  drive keeps only online, the analysis skips them (`fourier tools analyze --download` fetches
  them first), and a build downloads the ones it picks before reading any and stops, writing
  nothing, if one won't download.
- **A second library** can share the Fourier home: another config (`fourier setup --to
  other.toml`) gets its own master, renders and releases.
- **Checks and explanations.** `fourier verify` checks the master against every curation rule
  (and, with `--render`, scratch device renders); `fourier why` says where a sample landed and
  which rule put it there, or why it was left out, and `why --unrecognized` lists what no rule
  recognized; `fourier diff` compares two builds.
- **Releases and devices.** `fourier publish` cuts immutable releases; `fourier render` converts
  the master or a release for a device; a released render locks its paths on that device, so
  later releases never move them. `fourier sync` copies a render onto an SD card without
  macOS's `._` files.
- **Device profiles** for the Elektron Digitakt 2 and the Dirtywave M8, every value citing a
  page of the device's manual or marked as a convention or unverified, plus `generic_44k`,
  `generic_48k`, `generic_folder` and `generic_sd_card`. `fourier devices new` writes a profile
  for your own device; `fourier devices list|show`.
- **Configuration.** Style presets (`balanced`, the default and the code's own settings,
  `breaks-acid`, which the other genre styles build on, `hiphop-lofi`, `house-techno`,
  `ambient-cinematic`, `trap`), fifteen knobs (`devices`, `categories`, `tempo`,
  `fold`, `scale`, `files`, `size`, `loudness`, `retune`, `stereo`, `names`, `vendors`, `words`,
  `sets`, `sources`), overlays and `--set KEY=VALUE` layer over the code's defaults; `fourier
  config show|explain` prints what resolved and from where. `fourier doctor` checks what a
  build needs, fails on anything that would stop one, and estimates how long it takes.
- **The listening loop.** Rate the master's files Keep, Drop or Misfiled with `fourier review
  rate` or `review import` (a CSV file), or with tags in Ableton Live's browser; `fourier
  review` scores a build against the ratings and queues the files whose rating teaches the
  most. Every build follows the ratings.
- **Search.** `fourier search` by words or a reference file (CLAP), or by duration, tempo and
  descriptors.
- **`fourier demo`**: the whole pipeline on a generated library, in a sandbox folder with its own
  home and config, with no samples, model or download needed.
- **Safety guards.** Commands that write or delete refuse a library folder, a release, the
  home folder, a volume root and any folder they didn't make: a build stops on a master holding
  files it didn't make and lists them, renders and the demo replace only folders they marked as
  theirs, and `fourier sync --delete` removes only what it copied to that card. A first run's
  usual errors are one line saying what to run, not a traceback.
- **Preparer plug-ins** (`fourier.preparers` entry points) for open formats soundfile doesn't
  read.
- **For people new to Terminal**: a step-by-step getting-started page for the Mac
  (`docs/start.md`); setup takes a folder dragged in from Finder, devices and styles by number
  or name, and asks how big a set (starter, standard, full, with load times for Transfer),
  prepared for hardware or as-is, and which kinds of sound to leave out; `fourier open` shows the
  master, a render, the releases or the logs in your file browser; `fourier config edit` opens
  fourier.toml in a text editor (and runs with a config that doesn't load, which is when it's
  needed); a fourier.toml typed with smart quotes still loads, and a mistake in one names its
  line and the usual fix. `fourier config explain <setting>` says what a setting does in plain
  words, with an example.
- **A report page**: `fourier open report` opens a page in the web browser with every category
  and family, a play button for each file, where it came from, what the build left out and why,
  and Keep / Drop / Misfiled buttons that export a CSV for `fourier review import`.
- **Plain answers**: `fourier why` answers in one sentence (`--detail` for every rule, vote and
  score); a command that hits something unexpected says so in one line and keeps the details in
  a file for a bug report; commands with nothing built yet say what to run first.
- **Loading only what's new**: `fourier render <device> --release v2 --new-only` also copies the
  files v2 adds into a folder of their own, for a loader that can't sync (Elektron Transfer).
- **Documentation**: the README, a guide to what comes after the first build
  (`docs/guide.md`), the reference (`docs/usage.md`), the curation rules (`docs/curation.md`),
  and a skill for coding assistants (`skills/fourier/SKILL.md`) that teaches one to help with
  Fourier safely.
- `fourier --version`, and `python -m fourier`.
- Python 3.11 to 3.13 on macOS and Linux; Windows is experimental.

[Unreleased]: https://github.com/bvk7787/fourier-samples/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bvk7787/fourier-samples/releases/tag/v0.1.0
