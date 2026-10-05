# Using Fourier Samples

Reference for every command and setting. For step-by-step tasks, see [the guide](guide.md); for
an overview, the [README](../README.md).

## Commands

`fourier --help` lists them in these sections, and each command's `--help` says more.

| Section | Command | What it does |
|---------|---------|--------------|
| Get started | `setup` | Sets up for your library, step by step: samples, devices, style, size (`--size`), prepared or as-is (`--processing`), categories to leave out, the CLAP model, an optional local LLM, a check |
| | `demo` | Runs the whole pipeline on a generated library, in a sandbox folder |
| | `doctor` | Shows what a build needs, what's missing, and how long analysis and the build take |
| Build and load | `build` | Scans the library, analyzes what's new, builds the master and verifies it (`--all`, `CATEGORY`, `--dry-run`, `--resume`, `--base vN`, `--no-scan`) |
| | `render` | Converts the master or a release for one device (`--release vN`, `--dry-run`, `--check`). `--new-only` puts the files a release adds in a folder of their own |
| | `sync` | Copies a render onto a mounted SD card (`--delete`, `--dry-run`) |
| Releases | `publish` | Cuts an immutable release from the master (`--dry-run` shows what it would change) |
| | `releases` | Lists the releases on disk and which is latest (`recorded`, `import vN`) |
| Look inside | `open` | Opens the master, a device's render, the releases or the build logs in your file browser (`--print`: the path). `open report` opens a page in your web browser to listen, see why and rate |
| | `why` | Says where a sample landed and why, in one line (`--detail`: every rule, vote and score; `--rules`; `--unrecognized`) |
| | `verify` | Checks every curation rule on the master (`--render DEVICE`, `--quick`) |
| | `diff` | Shows what changed between two builds |
| | `search` | Searches the library by sound (words or a reference file) or by filters |
| Settings | `config` | Your settings: `edit` (opens fourier.toml in a text editor), `show`, `explain NAME` (`--detail`: the exact rule) |
| | `devices` | The device profiles: `list`, `show ID`, and `new` for a profile of your own device |
| Listen and rate | `review` | The listening loop: `rate` and `import` (no Live needed), `queue`, `score`, `misfiles`, and `ratings` (harvests Live's tags) |
| Advanced | `tools` | `scan` and `analyze` (the steps a build runs), `train` (the sound model), `audit` (what the master leaves out), `import-folder` (copies a folder of audio into the library), `dedup`, `resolve`, `db-stats` (the database, `--missing`, `--metadata`) |

## Words used here

| Word | Meaning |
|------|---------|
| library | your sample folders, as `fourier.toml` names them. Fourier only reads them |
| master | the curated folder a build makes (`~/Music/FourierCurated`) |
| category | a top-level folder of the master, such as `KICKS` or `DRUMLOOPS` |
| family | a folder inside a category: similar sounds, named for what they share |
| band | a split a category's folders never mix: closed and open hats, a loop's tempo range |
| render | a folder of the master or a release converted for one device |
| release | an immutable copy of the master, `releases/vN` |
| path lock | the record of the paths a released render put on a device. Later releases keep them |
| preset | a style: `balanced`, `breaks-acid`, `hiphop-lofi`, `house-techno`, `ambient-cinematic` or `trap` |
| knob | one of fifteen plain settings in `fourier.toml` (`categories`, `tempo`, `size`, `scale`, `words`, ...) |
| tunable | any of the 300+ settings underneath, reachable with `[advanced]` or `--set` |
| overlay | a private YAML file with rules for your own packs and categories |
| provider | a source of labels: Sononym, Live, or the built-in path and audio classifiers |

## Your own device

`fourier devices new` asks what the device plays and how it organizes files. It writes
`~/.config/fourier/devices/<id>.yaml` with every value marked unverified. From then on,
`fourier render <id>` and `devices = ["<id>"]` in `fourier.toml` use it. Each answer can be
given as an option instead (`fourier devices new --name "My Sampler" --load card --sample-rate
48000 --yes`).

```bash
fourier devices new
fourier devices show my_sampler
fourier render my_sampler
```

Your profiles survive an update of Fourier Samples. The [device profile
guide](device-profiles.md) covers the questions, path limits, where profiles are loaded from,
what a render does with each value, and how to check a profile against the device's manual.

## Installing and updating

```bash
uv tool upgrade fourier-samples       # keeps your setup: PyTorch, Transformers and their build
fourier doctor                        # then check
```

`fourier setup` installs PyTorch and Transformers into the uv tool as `--with` requirements,
which `uv tool upgrade` keeps. On Linux without an NVIDIA GPU, it pins the CPU build from
PyTorch's CPU index, so an upgrade never pulls the CUDA build (several GB the machine can't use).
To update PyTorch, run `fourier setup` again. It asks, then pins the new version. With pipx, run
`pipx upgrade fourier-samples`. In a virtual environment, `pip install -U` the address you
installed from.

`uv tool install --reinstall` drops PyTorch and Transformers, so run `fourier setup` afterwards
to put CLAP back (`fourier doctor` reminds you). uv picks Python 3.11 or later for the tool,
downloading one if needed. The suite runs on 3.11 to 3.14. An upgrade doesn't touch your config,
home, master or releases.

## Rating files

Rate the master's files and the next build follows: a **Keep** stays, a **Drop** goes, and a
**Misfiled** moves to another category. Three ways write the same ratings
(`~/.fourier/ratings.json`):

```bash
fourier review rate "Kick 01" keep                       # a name, a library path or KICKS/family/file.wav
fourier review rate "Snare Tight 03" misfiled --to CLAPS
fourier review import ratings.csv                        # header: path (or name), rating, category
```

`review rate` takes what `fourier why` takes, and asks which file you mean when a name matches
several. `review import` reads a CSV with a `path` or `name` column, a `rating` column (keep,
drop, misfiled) and an optional `category` column for a misfiled file. It lists the rows it
couldn't match.

With Ableton Live, tag files in its browser instead (Fourier|Keep, Fourier|Drop,
Fourier|Misfiled, Fourier|Move-CLAPS). `fourier review ratings` and every build harvest the tags.
`fourier review queue` puts the files whose rating teaches the most in `<master>/_REVIEW`, and
`fourier review score` scores a build against your ratings. Neither needs Live.

`fourier review rate "Kick 01" clear` removes a rating; change a Live tag's rating in Live.
Ratings follow a file the library walk sees moved, such as after a folder rename. A Keep whose
file is gone is passed over with a warning until you clear it.

## Configuration

`fourier setup` writes `~/.config/fourier/fourier.toml` (`--to` puts it elsewhere). A file named
by `--config` or `$FOURIER_CONFIG`, or a `fourier.toml` in the current folder, takes precedence.
The one thing a build needs is your library:

```toml
library = ["~/Samples", "/Volumes/Archive/Drums"]   # your sample folder(s); or $FOURIER_LIBRARY
devices = ["digitakt_2"]                             # what you render for
preset = "balanced"
[output]
master = "~/Music/FourierCurated"                    # the default
```

A library entry can be a bare folder name (`library = ["Samples"]`). It matches that folder
wherever it is, so one config works when the library moves between drives.

**A second library.** `fourier setup --to ~/other.toml --library ~/Other` writes a second config
and names its master, renders and releases after it (`~/Music/FourierCurated-other`,
`FourierRenders-other`, `Fourier-other`), so the two libraries never share a master. Use it with
`fourier --config ~/other.toml ...`.

Both configs share the home (`~/.fourier`) and its database. Each config's commands read only
the samples under its own library folders ("N samples from other libraries left out"), and a
sample both libraries hold is analyzed once. Before its first scan, the new library sees none of
the other's samples: doctor says the first build scans it, and `build --dry-run` sizes the
master from its files. Ratings and build history are kept apart by path.

Every build records its library in the master's manifest. A build into a master another
library's build made stops before changing anything, and doctor FAILs it. Give each config its
own `[output] master`. To separate the database, CLAP index, caches and ratings too, give the
other config its own home: `FOURIER_HOME=~/.fourier-other fourier --config ~/other.toml build`.

**A cloud-synced library** (iCloud Drive, OneDrive, Dropbox) works best kept downloaded. If it
isn't, `fourier setup` and `fourier doctor` count the cloud-only files, and a build's analysis
skips them (`fourier tools analyze --download` fetches them). A build downloads its picks first,
and stops without writing anything if one won't download.

**What the scan reads.** Without Sononym, or with `fourier tools scan --walk` beside it, a build's
scan walks the library folders for `.wav .aif .aiff .aifc .flac .mp3 .ogg .opus .caf` files,
whichever the installed soundfile reads (not `.m4a`). It reports what it skipped, by extension.

- A corrupt or empty audio file is listed as an error and left out. `fourier doctor` lists them,
  and a scan reads them again only once they change.
- Dot files and Fourier's own folders are skipped.
- A folder symlink that leads out of the library folder (a pack on another drive) is followed,
  and its files are filed where the link puts them. Each real folder is walked once.
- Samples the walk no longer finds (moved, renamed, deleted) are marked missing and left out of
  builds until they're back. `fourier tools db-stats --missing --prune` removes them.
- A file found at a new path with the same content (a renamed folder) is a move. Its analysis
  and ratings go with it, and the next build keeps it where the master had it, with the same
  name and audio. The scan, `db-stats --missing` ("moved (old rows)") and CHANGELOG.md say so.

**A drive that isn't there.** A build stops before anything changes when:

- a library folder is missing or empty (an unmounted drive);
- a folder symlink in it leads nowhere (a pack on a drive that isn't plugged in). Both the build
  and `fourier doctor` say "a linked folder is unavailable", and `--allow-missing` builds
  without it;
- many of the master's sources are gone from disk while the database would still pick them;
- a pick can't be exported.

The message says which files and what to run.

Rules for your own packs, and categories only your library needs, go in a private overlay
(`overlay = "my-library.yaml"`). See "Adding a category" in the
[curation reference](https://github.com/bvk7787/fourier-samples/blob/main/docs/curation.md).

## Styles and knobs

A preset decides what the master holds. `fourier setup` asks "What do you make?" and lists them,
and `fourier config show` prints each one's line:

| Preset | For |
|--------|-----|
| `balanced` | every category, budgets spread evenly, drum loops of every style from 70 to 180 BPM. The default, and what a `fourier.toml` without a `preset` line gets |
| `breaks-acid` | breaks, jungle, drum and bass and acid first, house after. Drum loops get the biggest budget (85-180 BPM, half time folded up), then synths and basses |
| `hiphop-lofi` | hip hop, boom bap and lo-fi: drum loops at 70-100 BPM, 8-bar loops up to 28 s, keys, vocals and textures |
| `house-techno` | house and techno: four-on-the-floor drum loops at 118-140 BPM, no breakbeat folding, stabs, chords and basses |
| `ambient-cinematic` | pads, textures, drones and FX first (pads up to 60 s), few drum loops |
| `trap` | 808s and hard drums, drum loops at 130-160 BPM (half time at 65-80 folded up), vocals and FX |

The knobs adjust any of them:

```toml
preset = "breaks-acid"
categories = { WAVES = "off", VOX = 1.5 }   # off, on, or a budget weight
tempo = "85-180"                            # where the 5-BPM loop folders lie, and where loops fold
fold = "auto"                               # half and double time fold into the tempo range; "off"
size = "auto"                               # fit the devices; "fixed", "5GB", or files = 6000
scale = "library"                           # the master follows the library's size; "off"
loudness = "standard"                       # gentle | standard | hot
retune = "detect"                           # tonal one-shots to C: off | named | detect
stereo = "fold-near-mono"                   # keep | fold-near-mono | mono
names = "canonical"                         # canonical | keep (the source names)
vendors = "auto"                            # a library's layout decides its vendors; "first-folder"
words = { KICKS = ["bombo"], PADS = ["nappe"] }   # your own words for a category (no Sononym)
sets = "on"                                 # 00_KITS and 00_SLICE; "off" for the folders only
```

**Tempo folding.** Half and double time are one tempo on a sampler, so a loop's tempo folds by
octaves into the top octave of the `tempo` range: with `85-180`, an 86 BPM break is filed at 172.
A fold never takes a loop out of a narrower range. With `house-techno`'s 118-140, a 174 BPM break
stays at 174 rather than 87, and with `hiphop-lofi`'s 70-100, a 120 BPM loop stays at 120.
`fold = "off"` files every loop at its own tempo.

**Vendors.** No vendor may supply more than 40% of a category (`sources.vendor_max`). With
`vendors = "auto"`, which `fourier setup` writes, each library folder's layout decides the
vendor:

- vendor or pack folders: the first folder;
- folders by sound type (`Drums/Kicks`, `Loops`): no vendor, so no cap;
- an umbrella folder of packs (`Downloads/`): the folder below it;
- a flat folder of files: no cap.

`fourier doctor` and setup print the layout in one line. `vendors = "first-folder"` always takes
the first folder.

**Names Fourier doesn't know.** Without Sononym, a file's category comes from the words in its
name and folders (`kick`, `BD`, `SD`, `CH`, `HiHat58`, `808s`, `pad`, `FX`, `vox`, ...; see
config/providers/path.yaml). A file with no such word (`001.wav`, `bombo_3.wav`) is placed by
sound when CLAP clearly hears one category, and `fourier why` says "placed by sound (CLAP), no
name rule matched". `words` adds your own words. `fourier why --unrecognized` lists what no rule
recognized.

**Without Sononym: a sound model trained on your library.** Your first build trains a small model
on your library, in about a minute. It learns from your sample makers' file and folder names and
your ratings, never from Sononym's or Live's analysis. It trains once your names label 1,000
samples or more across 8 packs, and again when your library has grown a lot. It keeps a model
only when it's reliable on packs it didn't learn from, which a library of a few packs usually
isn't. `fourier tools train` trains it any time.

It places a file no rule recognized when it's confident (0.70 or more), and a file only its
pack's name or a vague word ("FX", "Perc") describes when it's sure (0.90 or more). `fourier why`
then says, for example, "sound model: PADS 0.86". Drums are its strength. Synths, pads and basses
are harder, which is why it places only confident calls. The weights stay in your Fourier home.
They're learned from your samples, under your licenses, so Fourier ships none and you shouldn't
share yours.

With Sononym, it trains only when asked and places nothing: `fourier why` shows its call beside
Sononym's, and `fourier tools db-stats --disagreements` lists where they differ. `fourier setup
--no-sound-model` turns it off. The [curation
reference](https://github.com/bvk7787/fourier-samples/blob/main/docs/curation.md#23-the-sound-model)
has the details and how accurate it was on a large library.

**Library scaling.** The master scales with the library and stops growing at the style's size.

- With fewer than 8 usable samples for each file the budgets add up to, every budget is scaled
  by the library's share of that.
- A category keeps up to 24 of its usable files and about 1 in 8 beyond, or its scaled budget if
  that's more.
- A category builds from at least 2 usable samples.
- Folders hold about 24 files.

"Usable" is counted before the near-duplicate prune and the gates. Near-identical takes, files a
gate turns away and categories under their minimum take some out, so doctor, setup and `build
--dry-run` give the master's size as "up to about N files".

With `breaks-acid`, a 40-sample library gets a master of about 30 files, 400 samples about 340,
4,000 about 900 and 40,000 about 5,400, depending on how the sounds spread over the categories.
A large library's build is unchanged. `size` and the category weights still set the budgets the
master grows toward. `files = N` is an exact size and turns library scaling off, as
`scale = "off"` does.

`fourier config show` prints what the config sets and which layer set it. `fourier config explain
NAME` explains one setting (`fourier config explain scale`), and `--detail` adds the exact rule
and the advanced settings behind it. `words` takes the one-shot categories and the loop-free
ones. DRUMLOOPS, PHRASES, PIANO, WAVES and ACOUSTIC go by tempo, sound and tags.

### Other samplers

| Sampler | Profile to use |
|---------|----------------|
| Elektron Digitakt 2 | `digitakt_2` |
| Dirtywave M8 | `m8_tracker` |
| A sampler that reads WAV from an SD card (MPC, SP-404, 1010music Blackbox, Synthstrom Deluge, Teenage Engineering samplers, ...) | `generic_sd_card`, or better, a profile of your own (below) |
| Any other sampler, with files you copy over yourself | `generic_44k` or `generic_48k`, for the rate it plays: 16-bit stereo WAV, no limits claimed |
| A DAW or a computer folder | `generic_folder` |
| The first Elektron Digitakt, Model:Samples and other Elektron samplers | not profiled yet: `fourier devices new`. They hold less, so set `size` |

For a sampler without a profile, `fourier devices new` fits better than a generic one: it takes
the device's sample rate, folder depth and name limits, each marked unverified until you cite the
manual. Share the profile in Discussions.

A render writes only its own folder and never touches other samples on the device. On an SD
card, `fourier sync` copies into its own folder (`/Samples/Fourier` on the M8), and `--delete`
removes only files it put there. On the Digitakt 2, Transfer adds the folders you drag in beside
what's on the +Drive.

**Two sets from one library**, such as a breaks set and a house set: give each its own config
with its own `[output]` folders (`fourier setup --to ~/house.toml`, then `fourier --config
~/house.toml build`). On an SD card both would sync into the same folder, so give the second
device profile its own card folder (`fourier devices new --card-dir /Samples/House`).

### Common fixes

| What you see | Why, and what to do |
|--------------|---------------------|
| My kicks aren't showing up | Without Sononym, a file is a kick by the words in its name or folder (`kick`, `kik`, `kck`, `BD`, `Kick01`). `fourier why "<file>"` shows the words it found. Add yours with `words = { KICKS = ["bombo"] }`, or rename the folder `Kicks`. A file CLAP clearly hears as a kick is placed anyway. |
| A category is empty: "too few samples the rules recognize" | The library has the files, but no rule recognized them. The build prints how many and their folders, and `fourier why --unrecognized` lists them. Use `words`, rename the folders, or give a whole pack one home: `sources = { home = { "Acme Pads" = "PADS" } }`. |
| Everything ended up in FX | A folder named `FX` or `SFX` above your files puts them in FX when nothing nearer names them. Name the sounds in the folders below (`Kicks`, `Pads`) or with `words`. `fourier why "<file>"` shows which word decided. |
| My loops are in the wrong tempo folder | A loop's tempo comes from its name (`120bpm`, `_120_`, `bpm120`), then (without Sononym) the WAV's ACID chunk, then the analysis, then a folder named only by a number (`Loops/174/`). It then folds into the top octave of your `tempo` range. Set `tempo` to your style's range, pick a preset for it, or set `fold = "off"`. |
| My 8-bar loops are missing | Drum loops are at most 16 s by default (8 bars at 120 BPM). `preset = "hiphop-lofi"` allows up to 28 s (advanced setting `DRUMLOOP_DUR_MAX`). |
| My lo-fi drum loops are missing | Filtered drums, or drums under a chord, read as tonal: `fourier why` says "too harmonic". `hiphop-lofi`, `trap` and `ambient-cinematic` take more tonal drum loops. Without Sononym, a loop named like a phrase that CLAP hears as drums goes to the drum loops anyway, and the other way round ("fell back to ... by sound"). Busy loops in a folder named only by a tempo (`Loops/85/loop_17.wav`) count as drums. |
| My loops have no tempo | A loop needs a whole-bar tempo. Without Sononym, when nothing names one and the analysis reads a little off (89.1 for an 8-bar loop at 88), Fourier uses the tempo the loop's length implies, and `fourier why` says "tempo from length (8 bars)". Otherwise, put the tempo in the name or folder (`Loops/88/`). |
| Some files are in no category | `fourier why --unrecognized` lists files no word Fourier knows names. It also lists recognized files the last build placed nowhere ("recognized as lead, but no category took it"), because a rule kept them out or the category is off. `fourier why "<file>"` says which rule. |
| My keys aren't in PIANO | Without Sononym, keys are known by name or folder: `piano`, `rhodes`, `wurli`, `e-piano`, `EP`, `organ`, `clav`, or a `Keys` folder. PIANO takes single notes, and chords only as short stabs. `fourier why "<file>"` says which. |
| My 808 is called C but was an F | Tonal one-shots are retuned to C and renamed for the note they now play ("808_F" becomes "808_C"). A second C in the folder keeps the note it was ("808_C_from-f"). `retune = "off"` keeps every pitch and name. |
| My trap loops are missing | `breaks-acid`, and the styles built on it except `trap` and `hiphop-lofi`, leave trap, dubstep and future bass folders out of the drum loops (`fourier why`: "an off-style loop"). `balanced` and `trap` take them. |
| My sounds came out at a different pitch, level or name | The default prepares them for hardware: melodic one-shots tuned to C, loops evened out, near-mono files made mono, tidy names. `retune = "off"`, `loudness = "gentle"`, `stereo = "keep"` and `names = "keep"` keep them as they are (setup's "as-is"). |
| The set is too big for my card or takes too long to load | `size = "1GB"` (or a card's size) makes it fit. `fourier build --dry-run` shows the size per category first. |
| I only want some of my packs | Fourier reads every folder in `library`, so list just the ones you want (`library = ["~/Samples/Drums", "~/Samples/Acme"]`). There is no per-pack exclude. For Splice or Loopcloud downloads, add their sample folder (`~/Splice/sounds`) to `library`. |
| Builds are slow | `fourier build -j 4` (or more) builds categories in parallel. The first build's analysis is the long part, and it runs once. |
| A file is "too quiet" | It is near-silent once trimmed, or a soft velocity layer of a louder take, so export skipped it. |
| One pack fills a whole category | The vendor cap is reading the wrong folder. With `vendors = "first-folder"`, a library organized by sound type has one "vendor" per type folder, so the cap can't tell your packs apart. `vendors = "auto"` detects the layout. |
| I don't want the kits and slice folders | `sets = "off"` builds the category folders only. |
| My second library's build stopped: "holds another library's build" | Its config has no `[output] master` of its own, so it uses the first library's master. Set one in its config; `fourier setup --to` gives a new config its own. Nothing was changed. |
| A render cut my folder names | The device isn't in `devices`, so the master's names weren't sized for its path limit. Render and `render --check` say how many names were cut, one per folder. Add the device to `devices` and build again for names that fit. |
| My pads from an unsorted folder aren't in PADS | Without Sononym, a held, tonal sound with slow chord changes and no stated tempo is offered the one-shot categories and placed by sound. A pad with silences between its chords reads as several sounds (a sample chain) and is left out. |
| A copy of a file is "not in the master" | Byte-identical copies are kept once. `fourier why "<copy>"` names the file the master holds. |
