# A guide to Fourier Samples

The [README](../README.md) (or, new to Terminal, [getting started on a Mac](start.md)) gets you
installed and through a first build. This guide walks the
things you'll do after that: loading a device, keeping your projects working as the library
grows, listening and rating, and making the result yours. Every command and setting is in
[usage.md](usage.md); the rules a build follows are in [curation.md](curation.md).

## 1. Your first build

```bash
fourier setup        # where your samples are, your devices and style, the CLAP model
fourier doctor       # what's ready, what the first build will do, how long it should take
fourier build        # scan, analyze, build the master, verify it
```

The first build is the long one: it analyzes every sample (CLAP listens to each file, about
30 a second on a recent computer, so 100,000 files is roughly an hour). `fourier doctor` and the
build say how long before they start. Stopped part way? `fourier build --resume` picks up where
it left off. Later builds only analyze what's new.

On a terminal the build shows its progress, then a summary: the files in each category, any
category left empty, how many samples no name rule recognized, verify's result, and where the
master and the full log are (`~/.fourier/logs`). `fourier build --verbose` prints everything.

The master is `~/Music/FourierCurated` unless setup put it elsewhere: plain folders of WAV
files, `CATEGORY/family/file.wav`, with a `manifest.json` saying where each came from.
`fourier open report` shows it as a page in your web browser: every category and family with a
play button for each file, where each came from, what the build left out and why, and Keep /
Drop / Misfiled buttons (section 4). `fourier open` shows the folders in Finder, and `fourier
open logs` the build logs. Your library is only read, never changed.

**If a category is thin or empty**, the library has few samples Fourier recognizes for it.
`fourier why --unrecognized` lists the samples no rule recognized and the folders holding
them. Usually the fix is a word: packs name things their own way, and `words = { KICKS =
["bombo"] }` in `fourier.toml` teaches Fourier one (`fourier config edit` opens the file). Without Sononym, a large library also
trains a sound model on its own names once, and that places what names don't.

**If a sample is in the wrong place**, `fourier why "<part of its name>"` says which rule put it
there. Rate it Misfiled (section 4) and the next build moves it.

## 2. Loading a device

A render is the master converted for one device: its sample rate, bit depth, path limits and
folder layout. Load a **release** (section 3) for music you'll keep; the examples here render
the master itself, a preview to try a set on the device.

**Digitakt 2.** `fourier render digitakt_2`, then `fourier open renders digitakt_2` shows the
render in Finder. Open Elektron Transfer, make a folder on the +Drive (such as `/fourier`) and
drag the render's category folders into it. Transfer never
overwrites a file of the same name, so to replace a load, delete the folder in Transfer first.
Keep the Mac awake while it copies: a large load takes a while.

**Dirtywave M8.** Put the M8 in disk mode, then `fourier render m8_tracker` and `fourier sync
m8_tracker /Volumes/M8`. It copies into `/Samples/Fourier` on the card, without the `._` files
macOS leaves. `--delete` also removes files a newer version dropped (only files Fourier put
there; anything else stops it).

**Anything else.** `fourier devices list` shows the profiles; `generic_sd_card` and
`generic_folder` fit most samplers and DAWs, and `fourier devices new` writes one for yours
([usage.md](usage.md#your-own-device)).

A render of the master is a preview: the next build can move files. For a set you'll make
music with, load a release (next section).

## 3. Releases: keeping your projects working

A later build can move, rename or drop files. On the M8 a song finds its samples by path, so a
moved file goes missing from it; on the Digitakt 2 a project finds its samples by their content, so
a moved or renamed file is still found, but one deleted from the +Drive drops out of every
project (manual p.29). Releases keep
both safe: what a release put on the device stays, with the same name and audio.

```bash
fourier publish --notes "first set"        # releases/v1: a copy of the master that never changes
fourier render digitakt_2 --release v1     # render it, and lock its paths on that device
```

Load that render as above. The lock records every path on the device, and every later release
keeps them: a file you used stays where it is, with the same audio.

**The next release.** New packs, new settings or your ratings: build, publish v2 and render
it. The render keeps every path v1 put on the device with its audio, and adds the new files
beside them; `--new-only` also copies just those new files into a folder of their own
(`<render>-new-in-v2`), so Transfer loads only them, into the same folders as before:

```bash
fourier build
fourier publish --notes "new packs"
fourier render digitakt_2 --release v2 --dry-run   # what changes on the device
fourier render digitakt_2 --release v2 --new-only
fourier open renders digitakt_2-new-in-v2          # drag these into the same +Drive folder
```

On the M8, `fourier sync m8_tracker /Volumes/M8` copies only what changed (`--delete` also
removes files the new version dropped, only ever files Fourier put there). `fourier releases`
lists what you've published and which is latest.

**Only adding, nothing else changing.** An additive build copies the release and adds new
samples without touching anything already in it:

```bash
fourier build --base v1 --only-pack "New Pack"   # or --since 2026-10-01; --allowance sets how much it may grow
fourier publish --dry-run --base v1              # what v2 would change; it refuses to move a released path
fourier publish --notes "new pack"
```

## 4. Listening and rating

Rate the master's files and every later build follows: a **Keep** stays, a **Drop** goes, a
**Misfiled** moves to the category you name.

```bash
fourier review rate "Kick 01" keep
fourier review rate "Snare Tight 03" misfiled --to CLAPS
fourier review queue          # the files whose rating teaches the build most, in <master>/_REVIEW
fourier review score          # how a build does against your ratings
```

The easiest way is `fourier open report`: listen in your web browser and press K, D or M
beside a file (Misfiled asks which category it belongs in). **Export ratings** saves a
`ratings.csv`, and `fourier review import ~/Downloads/ratings.csv` hands them to the next build.

`review rate` takes a name, a library path or a master path, and asks which you mean when
several match. `fourier review import ratings.csv` rates many at once. With Ableton Live you can
tag files in Live's browser instead (`Fourier|Keep` and so on); `fourier review ratings`
harvests the tags. Rated files also teach the sound model.

## 5. Making it yours

- **A style.** `preset = "house-techno"` (or `balanced`, `breaks-acid`, `hiphop-lofi`,
  `ambient-cinematic`, `trap`) sets the budgets, tempo range and what the drum loops favor.
- **Knobs.** Plain settings in `fourier.toml` (`fourier config edit` opens it): `categories = { BLIPS = "off" }`, `tempo =
  "120-140"`, `size = "4GB"` to fit a card, `stereo`, `loudness`, `names`. `fourier config
  explain <knob>` says what each does; `fourier config show` prints what's in effect and from
  where.
- **Size and processing.** `size = "1GB"` (setup's starter), `"3GB"` (standard) or a card's size;
  `retune = "off"`, `loudness = "gentle"`, `stereo = "keep"` and `names = "keep"` keep every
  sound as it is (setup's as-is).
- **Words.** Your packs' own names for things: `words = { PADS = ["nappe"] }`.
- **A pack that's all one kind of sound**: `sources = { home = { "Acme Pads" = "PADS" } }`.
  `sources = { favor = ["Acme"] }` nudges a maker's sounds in.
- **Which packs.** Fourier reads every folder in `library`. To use only some packs, point
  `library` at the folders you want (`library = ["~/Samples/Drums", "~/Samples/Acme"]`).
- **An overlay** for rules beyond these (a new category, naming rules): `overlay =
  "my-library.yaml"`; see [curation.md](curation.md).
- **Your device.** `fourier devices new` (section 2).

`fourier build --dry-run` shows files, disk space and time per category before you commit to
a change.

## 6. When something goes wrong

- **`fourier doctor`** first: each line is OK, WARN, NEXT (what the first build does itself) or
  FAIL (what would stop a build), with what to do.
- **A build stopped.** It prints the last lines of its log and where the rest is. The master is
  unchanged until a whole build has verified; `fourier build --resume` continues.
- **Undo a build.** The previous master is kept beside it as `<master>.prev`. In Finder (`fourier
  open` shows the master), rename the master to something else, then rename the `.prev` folder
  to the master's name.
- **A cloud-synced library** (iCloud, Dropbox): keep it downloaded. Files only in the cloud are
  skipped by the analysis and downloaded before a build uses them.
- **More fixes** for specific symptoms: [usage.md, "Common fixes"](usage.md#common-fixes).

Something still odd? Ask in [Discussions](https://github.com/bvk7787/fourier-samples/discussions)
with `fourier doctor`'s output, or open an issue with it and the build log (or the error file a
stopped command names); the [contributing guide](../CONTRIBUTING.md) says what helps.
