# Design history

Fourier Samples curates a large sample library for hardware samplers; the Dirtywave M8 and
Elektron Digitakt 2 profiles are the reference devices. The curation rules and their reasons
are in [curation.md, section 14](curation.md#14-decision-log); this page covers how the tool
became something anyone can run, and why it's shaped the way it is.

## Why releases and path locks

Samplers keep track of samples in ways a reorganized card can break. M8 songs refer to samples
by path, so a renamed folder breaks old M8 projects; the Digitakt 2 follows a rename but drops a
deleted sample from every preset that used it. Curation, though, keeps improving (loop onsets,
pitch checks, names). The answer was to separate the two. The master is a working draft that
any rule change may reshape, and a release is an immutable copy of it. Rendering a release for a
device records each file's path there, and later builds keep those paths (and the audio behind
them) whatever the rules now say. A new version only adds.

## Why stable builds

Without stable builds, a rule change can reshuffle thousands of files, and every build means
relearning the library. Builds became stable (the previous build's files and folders stay while
still eligible), so a change moves only the files it touches, and a second build in a row
changes no audio at all. That also made a reproducible "golden" rebuild possible, which
everything after leaned on.

## How it became configurable

Early versions assumed one library and one machine: they needed one commercial sample
browser's database to classify anything, and their taste lived in about 300 constants in
Python. The work that followed had one hard rule: after every change, the reference release
had to rebuild with zero differences, checked by a sealed harness that rebuilds a release from
frozen copies of every input.

It went in phases. First the groundwork: a home folder that can move, that harness, CI and an
inventory of every setting. Then every setting became overridable in layers (preset, overlay,
`fourier.toml`, `--set`), with a synthetic library and a golden build in CI, so taste lives in
config and a test runs anywhere. Classification moved behind pluggable providers, with built-in
path and audio providers standing in when no sample browser's data is there. The taxonomy went
into one file, with knobs and presets for the choices people actually make. The command line
shrank to 17 top-level commands in sections (`setup`, `doctor` and `demo` among them), with
power tools under `fourier tools`, and devices got one profile schema that cites the manual for
every value. Last came portability (a platform adapter, resumable builds, measured build times,
numbered migrations, a pinned model) and this public repository.

## Decisions along the way

- **Library-specific rules leave the code, not the tool.** Rules that only match particular
  packs became settings of class `library`, empty by default and set by a user's overlay, and
  tests use invented vendor and pack names.
- **Every device fact cites the manual.** A limit written from memory can be wrong, so
  profiles carry a verbatim quote and page for each value, and a test checks them against the
  manual. A built-in manual search was removed in favor of citing pages directly.
- **Open formats.** The tool reads what soundfile reads (WAV, AIFF, FLAC, OGG, MP3). A preparer
  plug-in (entry-point group `fourier.preparers`) can add another open or openly documented
  format for files with their own extensions. Fourier Samples doesn't accept code that circumvents
  copy protection.
- **CLAP as an extra, pinned.** Clustering and naming use CLAP embeddings; the model is one
  pinned revision so every install embeds the same way.
