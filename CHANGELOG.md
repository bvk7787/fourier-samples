# Changelog

All notable changes to Fourier Samples are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (before 1.0, a minor version may
change the CLI or the config).

## [Unreleased]

### Changed

- Documentation rewritten to be shorter and plainer, with consistent numbers and American
  spelling.
- Diagrams in the README (library to device), the guide (releases and path locks) and
  docs/curation.md (the build pipeline), drawn by GitHub from Mermaid text. The README's
  terminal capture shows the demo alone.
- CI runs on Python 3.11 to 3.14 (3.14 on Linux and macOS), measures test coverage on one job,
  and type-checks the source with mypy (lenient settings for now, in pyproject.toml; the lines
  it first flagged are marked in the code).
- The workflows pin each action to a commit and check out without keeping credentials. An
  OpenSSF Scorecard workflow scores the repository's security practices weekly.
- transformers 5.10.1 in uv.lock (Dependabot): its advisories are on no path Fourier uses, and
  CLAP embeds identically with it (400 library samples and every category phrase, compared with
  5.3.0).
- Windows left the CI matrix because it doesn't work yet (fourier.toml paths, rsync, SQLite file
  locks; [#9](https://github.com/bvk7787/fourier-samples/issues/9)). The `Windows` workflow runs
  the suite there by hand.

### Fixed

- Drums and drum loops whose stereo channels partly cancel are written mono (their louder
  channel) from a correlation of -0.3 or 4.5 dB lost summed to mono, where other categories
  still need -0.8 or 10 dB. A hat or snare no longer thins out on a mono system; ordinary
  wide stereo stays stereo.
- CLAP embedded a file over 10 seconds differently on every run, because its feature extractor
  crops a longer clip at a random offset. It now hears and decodes only the first 10 seconds, so
  the same file always gets the same embedding. A library analyzed with 0.1.0 keeps its earlier
  embeddings until `fourier tools analyze --only clap --force` redoes them.

## [0.1.0] - 2026-10-04

The first public release.

### Added

- `fourier build` curates a sample library into one master of `CATEGORY/family/file.wav`
  folders: 19 categories in play order (an overlay can add more), families clustered from CLAP
  embeddings and named for how they sound, drum kits (`KITS`), slice-ready loops (`SLICE`), and
  a manifest recording where each file came from and why it's there.
- Builds are deterministic, stable (a rule change moves only the files it touches) and
  resumable (`--resume`), swap in only when every category built and verify passed, and size
  the master to the library (`scale = "library"`; `files = N` and `scale = "off"` keep fixed
  budgets, `size` caps them for a device).
- The export fixes DC offset, phase-flipped stereo, clicks at the end, hot starts, long tails,
  per-folder loudness and notes a few cents off.
- Any library layout works, with or without Sononym and Ableton Live: built-in `path` and
  `audio` classifiers, your own `words` and `vendors = "auto"` stand in, cloud-only files are
  counted and downloaded when a build picks them, and the library's files are never changed.
- A sound model trained on your own library's file and folder names and your ratings (never on
  Sononym's or Live's analysis) places what no name rule recognizes, alongside CLAP's prompts;
  `fourier tools train` trains it any time, and Fourier ships no weights.
- `fourier setup` is a step-by-step wizard (or `--yes` with flags) that asks for the sample
  folders, devices and style, installs the CLAP model, can add a local LLM through Ollama, and
  runs doctor's checks and the first build; `fourier demo` runs the whole pipeline
  on a generated library.
- `fourier --help` lists 17 commands in sections, with the steps underneath in `fourier tools`;
  `doctor`, `verify`, `why`, `diff`, `search` and a report page (`fourier open report`) show
  what a build will do or did.
- Ratings (Keep, Drop, Misfiled) from `fourier review rate`, `review import` (a CSV, which the
  report page exports) or tags in Ableton Live's browser steer every build, and `fourier review`
  queues the files whose rating teaches the most.
- `fourier publish` cuts immutable releases, `fourier render` converts one for a device and
  locks its paths there, and `fourier sync` copies a render to an SD card; device profiles for
  the Elektron Digitakt 2 and Dirtywave M8 cite their manuals, with generic profiles and
  `fourier devices new` for others.
- Style presets, fifteen knobs, overlays and `--set` layer over the code's defaults (`fourier
  config show|explain|edit`); a second config can share the Fourier home; preparer plug-ins
  (`fourier.preparers`) add open formats.
- Commands that write or delete refuse folders they didn't make, and a first run's usual errors
  are one line saying what to run.
- Documentation (README, `docs/start.md` for people new to Terminal, `docs/guide.md`,
  `docs/usage.md`, `docs/curation.md`) and a skill for coding assistants
  (`skills/fourier/SKILL.md`); `fourier --version` and `python -m fourier`; Python 3.11 to 3.13
  on macOS and Linux, with Windows experimental.

The [README](README.md) has the full tour.

[Unreleased]: https://github.com/bvk7787/fourier-samples/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bvk7787/fourier-samples/releases/tag/v0.1.0
