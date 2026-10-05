# A guide to Fourier Samples

The [README](../README.md), or [getting started on a Mac](start.md) if you're new to Terminal,
gets you installed and through a first build. This guide walks through the things you'll do
after that: loading a device, keeping your projects working as the library grows, listening and
rating, and making the result yours. [usage.md](usage.md) has every command and setting, and
[curation.md](curation.md) has the rules a build follows.

## 1. Your first build

```bash
fourier setup        # where your samples are, your devices and style, the CLAP model
fourier doctor       # what's ready, what the first build will do, how long it should take
fourier build        # scan, analyze, build the master, verify it
```

The first build is the long one. CLAP listens to every sample, about 30 a second on a recent
computer, so 100,000 files take roughly an hour. `fourier doctor` and the build say how long
before they start. Later builds only analyze what's new.

On a terminal the build shows its progress, then a summary: the files in each category, any
category left empty, how many samples no name rule recognized, verify's result, and where the
master and the full log are (`~/.fourier/logs`). `fourier build --verbose` prints everything.

The master is `~/Music/FourierCurated` unless setup put it elsewhere. It's plain folders of WAV
files, `CATEGORY/family/file.wav`, with a `manifest.json` saying where each came from.
`fourier open report` shows it as a page in your web browser. It lists every category and
family, with a play button for each file, where each came from, what the build left out and
why, and Keep / Drop / Misfiled buttons (section 4). `fourier open` shows the folders in Finder,
and `fourier open logs` shows the build logs.

**If a category is thin or empty**, the library has few samples Fourier recognizes for it.
`fourier why --unrecognized` lists the samples no rule recognized and the folders holding
them. Usually the fix is a word. Packs name things their own way, and `words = { KICKS =
["bombo"] }` in `fourier.toml` teaches Fourier one (`fourier config edit` opens the file).
Without Sononym, a large library also trains a sound model on its own names once, and that
model places what names don't.

**If a sample is in the wrong place**, `fourier why "<part of its name>"` says which rule put it
there. Rate it Misfiled (section 4) and the next build moves it.

## 2. Loading a device

A render is the master converted for one device: its sample rate, bit depth, path limits and
folder layout. The examples here render the master itself. That's a preview to try a set on
the device, and the next build can move its files. For a set you'll make music with, load a
**release** (section 3).

**Digitakt 2.** Run `fourier render digitakt_2`, then `fourier open renders digitakt_2` to see
the render in Finder. Open Elektron Transfer, make a folder on the +Drive (such as `/fourier`)
and drag the render's category folders into it. Transfer never overwrites a file of the same
name, so to replace a load, delete the folder in Transfer first. A large load takes a while, so
keep the Mac awake while it copies.

**Dirtywave M8.** Put the M8 in disk mode, then run `fourier render m8_tracker` and `fourier
sync m8_tracker /Volumes/M8`. It copies into `/Samples/Fourier` on the card, without the `._`
files macOS leaves. Later syncs copy only what changed. `--delete` also removes files a newer
version dropped. It only ever removes files Fourier put there, and anything else stops it.

**Anything else.** `fourier devices list` shows the profiles. `generic_sd_card` and
`generic_folder` fit most samplers and DAWs, and `fourier devices new` writes one for yours
([usage.md](usage.md#your-own-device)).

## 3. Releases: keeping your projects working

A later build can move, rename or drop files. On the M8 a song finds its samples by path, so a
moved file goes missing from it. On the Digitakt 2 a project finds its samples by their content,
so a moved or renamed file is still found. But a file deleted from the +Drive drops out of every
project (manual p.29). Releases keep both safe.

```bash
fourier publish --notes "first set"        # releases/v1: a copy of the master that never changes
fourier render digitakt_2 --release v1     # render it, and lock its paths on that device
```

Load that render as in section 2. The lock records every path on the device, and every later
release keeps them. A file you used stays where it is, with the same name and audio.

**The next release.** For new packs, new settings or your ratings, build, publish v2 and render
it. The render keeps every path v1 put on the device, with its audio, and adds the new files
beside them. `--new-only` also copies just the new files into a folder of their own
(`<render>-new-in-v2`), so Transfer loads only them, into the same folders as before:

```bash
fourier build
fourier publish --notes "new packs"
fourier render digitakt_2 --release v2 --dry-run   # what changes on the device
fourier render digitakt_2 --release v2 --new-only
fourier open renders digitakt_2-new-in-v2          # drag these into the same +Drive folder
```

On the M8, sync as in section 2. `fourier releases` lists what you've published and which is
latest.

**Only adding, nothing else changing.** An additive build copies the release and adds new
samples without touching anything already in it:

```bash
fourier build --base v1 --only-pack "New Pack"   # or --since 2026-10-01; --allowance sets how much it may grow
fourier publish --dry-run --base v1              # what v2 would change; it refuses to move a released path
fourier publish --notes "new pack"
```

## 4. Listening and rating

Rate the master's files and every later build follows. A **Keep** stays, a **Drop** goes, and a
**Misfiled** moves to the category you name.

```bash
fourier review rate "Kick 01" keep
fourier review rate "Snare Tight 03" misfiled --to CLAPS
fourier review queue          # the files whose rating teaches the build most, in <master>/_REVIEW
fourier review score          # how a build does against your ratings
```

The easiest way is `fourier open report`. Listen in your web browser and press K, D or M beside
a file; Misfiled asks which category it belongs in. **Export ratings** saves a `ratings.csv`,
and `fourier review import ~/Downloads/ratings.csv` hands them to the next build.

`review rate` takes a name, a library path or a master path, and asks which you mean when
several match. `fourier review import ratings.csv` rates many at once. With Ableton Live you can
tag files in Live's browser instead (`Fourier|Keep` and so on), and `fourier review ratings`
harvests the tags. Rated files also teach the sound model.

## 5. Making it yours

- **A style.** `preset = "house-techno"` (or `balanced`, `breaks-acid`, `hiphop-lofi`,
  `ambient-cinematic`, `trap`) sets the budgets, tempo range and what the drum loops favor.
- **Knobs.** Plain settings in `fourier.toml` (`fourier config edit` opens it), such as
  `categories = { BLIPS = "off" }` and `tempo = "120-140"`. `size = "4GB"` fits a card;
  `"1GB"` is setup's starter and `"3GB"` its standard. Setup's as-is is `retune = "off"`,
  `loudness = "gentle"`, `stereo = "keep"` and `names = "keep"`, which keep every sound as it
  is. `fourier config explain <knob>` says what each does, and `fourier config show` prints
  what's in effect and where it came from.
- **Words.** Your packs' own names for things: `words = { PADS = ["nappe"] }`.
- **A pack that's all one kind of sound**: `sources = { home = { "Acme Pads" = "PADS" } }`.
  `sources = { favor = ["Acme"] }` nudges a maker's sounds in.
- **Which packs.** Fourier reads every folder in `library`. To use only some packs, point
  `library` at the folders you want: `library = ["~/Samples/Drums", "~/Samples/Acme"]`.
- **An overlay** for rules beyond these, such as a new category or naming rules: `overlay =
  "my-library.yaml"`. See [curation.md](curation.md).
- **Your device.** `fourier devices new` (section 2).

`fourier build --dry-run` shows files, disk space and time per category before you commit to
a change.

## 6. When something goes wrong

- **`fourier doctor`** first. Each line is OK, WARN, NEXT (the first build does it itself) or
  FAIL (it would stop a build), with what to do.
- **A build stopped.** It prints the last lines of its log and where the rest is. The master is
  unchanged until a whole build has verified, and `fourier build --resume` picks up where it
  left off.
- **Undo a build.** The previous master is kept beside it as `<master>.prev`. In Finder, rename
  the master to something else, then rename the `.prev` folder to the master's name. `fourier
  open` shows the master in Finder.
- **A cloud-synced library** (iCloud, Dropbox): keep it downloaded. Analysis skips files that
  are only in the cloud; a build downloads them first when it needs them.
- **More fixes** for specific symptoms: [usage.md, "Common fixes"](usage.md#common-fixes).

Something still odd? Ask in [Discussions](https://github.com/bvk7787/fourier-samples/discussions)
with `fourier doctor`'s output. Or open an issue with it and the build log, or the error file a
stopped command names. The [contributing guide](../CONTRIBUTING.md) says what helps.
