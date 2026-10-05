---
name: fourier
description: Help someone use Fourier Samples (the `fourier` command), which curates a sample library into device-ready folders for hardware samplers. Use when they ask about setting it up, a build's result, a sample in the wrong place, loading a Digitakt 2, M8 or other sampler, releases, ratings, or tuning fourier.toml.
---

# Fourier Samples, for an assistant helping its user

Fourier Samples reads a sample library (folders of WAV, AIFF, FLAC and the like) and builds a
**master**: `CATEGORY/family/file.wav` folders sized for a hardware sampler, with a
`manifest.json` recording where each file came from. `fourier render <device>` converts the
master for one device; `fourier publish` cuts an immutable release (`releases/vN`) that a
device render can lock, so projects on the device keep working as the library grows.

The user's guide is `docs/guide.md`, and every command and setting is in `docs/usage.md`
(https://github.com/bvk7787/fourier-samples/tree/main/docs). `fourier <command> --help` is
always current, so check it before suggesting an option you aren't sure of.

## Ground rules

- **The library is only read.** Fourier never changes the user's sample files, and you
  shouldn't either: no moving, renaming or deleting in the library to "fix" a build. Fix a
  misplaced sample with a rating, a `words` entry or an overlay (below).
- **Releases never change.** Don't run `fourier publish --force` or `--no-verify`, don't edit
  anything under `releases/` or the device lock files, and don't delete a release a device
  render was made from. Changes after a release is on a device go into the next release.
- **Ask before the long or the irreversible.** A first `fourier build` analyzes the whole
  library (about 30 files a second, so 100,000 files is about an hour); `fourier sync
  --delete` removes files from a card; `fourier publish` makes a release. Say what it will do
  and let the user start it. `fourier build --dry-run`, `publish --dry-run` and `render
  --dry-run` are safe previews.
- **Device facts come from the device's manual.** When writing or changing a device profile,
  cite the manual's page for each limit, or mark the value unverified. Don't state a sampler's
  limits from memory.
- **Sample licenses are the user's.** The sound model Fourier trains is learned from the
  user's own samples and stays in their Fourier home; never suggest sharing or committing it.

## Where things are

| What | Where |
|------|-------|
| Settings | `fourier.toml`: `--config`, `$FOURIER_CONFIG`, `./fourier.toml` or `~/.config/fourier/fourier.toml` |
| Fourier home | `~/.fourier` (`$FOURIER_HOME`): the database, ratings, build logs (`logs/`), the sound model, caches |
| Master | `[output] master` in fourier.toml, default `~/Music/FourierCurated`; the previous one is `<master>.prev` |
| Renders | `[output] renders`, default `FourierRenders/<device>` beside the master |
| Releases | `<publish root>/releases/vN`; locks in `<publish root>/devices/` |

- `fourier config show` prints the settings in effect and which file each came from.
- `fourier open <master|renders [device]|releases|logs|home> --print` prints a folder's path.
  Without `--print` it opens the folder in their file browser, which is what to suggest to
  someone who doesn't use a terminal much.
- `fourier open report` opens a page in their web browser to listen to the master, see why each
  sound is there and rate it.
- `fourier config edit` opens fourier.toml in their text editor.

If the user is new to Terminal, walk them through docs/start.md (getting started on a Mac)
rather than giving several commands at once, and say what they should see after each.

## Reading what Fourier says

- **`fourier doctor`**: one line per check, OK / WARN / NEXT / FAIL. NEXT is what the first
  build does by itself (scan, analyze, download the model). Only FAIL stops a build, and each
  FAIL says what to do. Start here when something's wrong.
- **A build** on a terminal prints its progress, then a summary: files per category,
  categories left empty, how many samples no rule recognized, verify's result with its first
  warnings, and the path of the full log (`~/.fourier/logs/build-*.log`). Read that log
  when you need the detail. `fourier build --verbose` prints everything instead, as a build
  piped to a file always does.
- **A build that stopped** shows the log's last lines. The master is unchanged until a whole
  build has verified; `fourier build --resume` continues where it stopped.
- **verify**: PASS with warnings is normal. Read the WARN lines; they say what to look at.
  `fourier verify` runs it again on the master.
- **A command that stopped** with "The details, for a bug report: <file>" hit something
  unexpected. That file is a Python traceback: read it to see what happened. `fourier -v
  <command>` shows it on screen.
- **`fourier why`** answers in one line; `fourier why --detail` gives every rule, vote and
  score behind it.

## Common requests

**"Category X is empty or thin."** The library has few samples Fourier recognizes for it.
`fourier why --unrecognized` lists the samples no rule recognized and their folders. If the
packs name the sound their own way, add the word to `fourier.toml`:

```toml
words = { KICKS = ["bombo"], PADS = ["nappe"] }
```

and build again. `fourier config explain words` says more. Some categories simply aren't in
every library, and an empty one is reported, not a failure.

**"This sample is in the wrong category."** `fourier why "<part of its name>"` says which
rule put it there. Rate it with `fourier review rate "<name>" misfiled --to <CATEGORY>` (or
`keep` / `drop`), and the next build follows. For many files, `fourier open report` has K / D /
M buttons and exports a `ratings.csv`, which `fourier review import ~/Downloads/ratings.csv`
reads.

**"Why isn't my favorite sample in the master?"** `fourier why "<name>"` says why the last
build left it out (a gate, the budget, a near-duplicate of another file, the per-vendor
share). A `keep` rating keeps it.

**"Make it smaller / fit my card / change the style."** Use the plain knobs in `fourier.toml`:
`preset` (`balanced`, `breaks-acid`, `house-techno`, `hiphop-lofi`, `ambient-cinematic`,
`trap`), `size = "4GB"`, `files = N`, `tempo = "120-140"`, `categories = { BLIPS = "off" }`,
`stereo`, `loudness`, `names`, `retune`. `fourier config explain <setting>` explains each, and
`fourier build --dry-run` shows the effect per category before building. For a pack that's all
one kind of sound, use `sources = { home = { "Acme Pads" = "PADS" } }`. Rules beyond the
settings, such as a new category, go in an overlay file (`overlay = "my-library.yaml"`, see
docs/curation.md), not in Fourier's code. "Keep my sounds as they are" is `retune = "off"`,
`loudness = "gentle"`, `stereo = "keep"`, `names = "keep"`.

**"Load it on my Digitakt 2."** Load a release, not the master: a render of the master is a
preview whose files can move with the next build. Run `fourier publish`, then `fourier render
digitakt_2 --release v1` and `fourier open renders digitakt_2`. In Elektron Transfer, drag the
render's category folders into a folder on the +Drive. Transfer doesn't overwrite a file of the
same name, so replacing a load means deleting that folder in Transfer first.

**"Load it on my M8."** `fourier publish`, `fourier render m8_tracker --release v1`, M8 in disk
mode, then `fourier sync m8_tracker /Volumes/<card>`.

**"My sampler isn't listed."** Run `fourier devices list`. `generic_sd_card`, `generic_44k`,
`generic_48k` or `generic_folder` often fit. Otherwise `fourier devices new` writes a profile
from a few questions, with every value marked unverified until the user cites the manual
(docs/device-profiles.md).

**"I'll make music with this set; keep it stable."** Publish and render the release:

```bash
fourier publish --notes "first set"
fourier render digitakt_2 --release v1      # locks its paths on that device
```

**"I bought new packs / changed settings; update the device without breaking my
projects."** Build, publish the next release and render it. The render keeps every path the
loaded release put on the device, with its audio. `--new-only` puts just the new files in a
folder of their own to copy across:

```bash
fourier build
fourier publish --notes "new packs"
fourier render digitakt_2 --release v2 --dry-run   # what changes on the device
fourier render digitakt_2 --release v2 --new-only
fourier open renders digitakt_2-new-in-v2          # drag into the same +Drive folder
```

To add packs and change nothing else, build additively first: `fourier build --base v1
--only-pack "New Pack"`, then `fourier publish --dry-run --base v1`, which refuses anything that
moves a released file.

**"Undo the last build."** The previous master is `<master>.prev`, and moving it back restores
it. Releases are never affected by a build.

**"Train the sound model."** Without Sononym, a build trains one on the library's own names
once there are enough, and keeps it only if it's reliable on packs it didn't learn from.
`fourier tools train` trains it now (about a minute, reads the database only) and writes a
report beside the weights in the Fourier home.

## Changing Fourier itself

For a change to Fourier's code rather than the user's settings, first read `CLAUDE.md` and
`AGENTS.md` in the repository. They hold the architecture, the rules a change follows and the
checks to run.
