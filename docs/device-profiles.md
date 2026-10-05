# Device profiles

Fourier Samples renders for a device from its profile. A profile is only as good as its facts,
so every value either cites a page of the device's manual or says it's a convention or
unverified. Part 1 makes a profile for your own device in a minute, marked unverified. Part 2
checks one against the manual, which is how a profile joins `config/devices/` in the package.
Until a device has a profile, `generic_44k`, `generic_48k`, `generic_sd_card` or
`generic_folder` renders a usable library with no limits claimed.

## Part 1: making your own device

`fourier devices new` asks for:

- the device's name;
- how samples get on it: `card` (an SD card or drive that `fourier sync` copies to), `transfer`
  (its maker's app, which never overwrites a file) or `folder` (you copy the render yourself).
  The profile file stores these as `load: card-sync`, `transfer` and `copy`;
- its sample rate (44.1 or 48 kHz) and bit depth (16, 24, or 8 if it takes 8-bit files);
- mono or stereo, and WAV or AIFF;
- the folder on the card;
- folder levels: 2 for `CATEGORY/family`, or 1 for `CATEGORY` with the family in the file name;
- the most files in one folder;
- the longest file path, file or folder name, and file length it takes (0 for no limit);
- its storage;
- whether its screen shows only plain-ASCII names.

Each question has an option (`fourier devices new --help`), and `--yes` takes the default for
anything not given:

```bash
fourier devices new --name "My Sampler" --load card --card-dir /SAMPLES \
    --sample-rate 48000 --bit-depth 16 --files-per-folder 64 --max-path 255 --yes
```

It writes `~/.config/fourier/devices/<id>.yaml` (or into the first folder in `$FOURIER_DEVICES`)
with every value `status: unverified` and a comment on citing the manual, then loads it to check
it. Next:

```bash
fourier devices show my_sampler     # each value and where it comes from
fourier render my_sampler           # the master, converted for it
```

Add it to `devices = [...]` in `fourier.toml` so builds size the master's names for its path
limit. Use the manual's numbers when you have them, and leave a limit you don't know at 0 rather
than guess.

**Path limits.** Take the card folder, the longest category folder (`08_DRUMLOOPS/`), ".wav" and
a "_2" off the path limit. What's left must give the family folder and the file name at least 12
characters each. `devices new` refuses a smaller limit and says the least that works; a shorter
card folder leaves more room. It also refuses a name limit under 14. `fourier doctor` FAILs a
profile whose limit is too tight and names it. A build stops on it; every other command runs.

### Where profiles come from

- The package's `config/devices/*.yaml` load first. Then your own: each folder in
  `$FOURIER_DEVICES` (separated like `PATH`), then `~/.config/fourier/devices`. Your own survive
  an update of Fourier Samples. `FOURIER_DEVICES=none` turns them off, as the test suite does.
- A profile of your own never takes a package id by accident. One with a package profile's id is
  skipped, and `fourier devices list` says so. With `override: true` at the top, it replaces
  that profile instead. Among your own folders, the first to define an id wins.
- `extends: <id>` starts a profile from another one, with its manual, citations and values, and
  changes only what it sets. `paths`, `audio` and `citations` merge key by key:

```yaml
id: my_card_device
name: My card device, 64 files a folder
extends: generic_sd_card
paths:
  files_per_folder: {value: 64, status: unverified}
```

### What a render does with each value

| Key | A render |
|-----|----------|
| `audio.sample_rate`, `bit_depth`, `channels` | converts to them (8-bit: TPDF-dithered, unsigned in WAV, signed in AIFF) |
| `audio.formats` | writes WAV when the list has `wav`, else AIFF (`.aif`) |
| `audio.max_duration_s` | cuts a longer file there, with a 5 ms fade (never a single-cycle waveform) |
| `audio.max_slices` | leaves out of the `SLICE` set the loops whose 16th-note grid needs more slices |
| `paths.card_dir`, `root` | puts the render there on the card (`fourier sync`) |
| `paths.folder_depth` | 2: `CATEGORY/family/file`; 1: `CATEGORY/family__file` |
| `paths.files_per_folder` | continues a full folder in numbered siblings (`punchy`, `punchy-2`, ...); `null`: no limit |
| `paths.max_path_length` | the build sizes the master's names for it when the device is in `devices`; otherwise the render shortens what doesn't fit (below) |
| `paths.max_name_length` | cuts a longer file or folder name in the middle (a cut that makes two names equal adds `_2`) |
| `paths.ascii_names` | when set, names are NFC-normalized. `true` also makes them plain ASCII (accents dropped, `ß` as `ss`, anything else `_`) |
| `storage_mb` | warns when a render is bigger; `size = "auto"` fits the master to it |

**A device not in `devices`.** Rendered for it, a family folder holding a path over the limit
gets one shorter name, cut in the middle and sized from its longest path. Each file is then cut
as much as it still needs. `render --check` shows the same names, and both say how many were
cut. A render stops, writing nothing, if a path can't fit.

**Locked paths.** Once a release is rendered for a device, its paths are locked. A later change
to a name rule or the folder limit applies to new files only. A change to the audio format stops
the render until you restore the profile or start a new lock.

## Part 2: checking a profile against the manual

This part is for coding agents and people adding a device to `config/devices/` in the package,
or checking a profile of their own. An agent can draft one from the manual, and a person checks
it on the hardware.

### 1. Get the manual

Download the maker's manual PDF for the device's current firmware into a folder of your own.
Don't commit it: the manual is the maker's copyright. Note its title, version, URL and sha256
(`sha256sum manual.pdf`, or `shasum -a 256` on macOS). Don't state a device fact from memory.
Devices change between firmware versions, and the profile is checked against the manual.

### 2. Answer the questions a profile asks

Look each one up in the manual. For every answer, note:

- the PDF page: the page index in a PDF viewer, counting the first page as 1, not the printed
  page number;
- a short verbatim quote, at least a dozen characters, exactly as printed.


| Key | Question |
|-----|----------|
| `audio.sample_rate`, `bit_depth` | What sample rate and bit depth does the device play or convert to? |
| `audio.channels` | Mono only, or stereo too? |
| `audio.formats` | Which file formats does it read (WAV, AIFF, ...)? |
| `audio.max_slices`, `max_duration_s` | Slice markers, a length limit? |
| `paths.card_dir`, `paths.root` | Where on the card or drive do samples go? |
| `paths.max_path_length`, `max_name_length` | Limits on the whole path or a file name? |
| `paths.ascii_names` | Does its screen or file system show only plain-ASCII names? |
| `paths.folder_depth`, `files_per_folder` | How deep can folders go; how many files does the browser handle? |
| `storage_mb` | How much storage does it have for samples? |
| `load` | How do samples get onto it: an SD card (`card-sync`), a transfer app that never overwrites (`transfer`), or copying into a project (`copy`)? `devices new --load` calls these `card`, `transfer` and `folder`. |

When the manual doesn't say, the value is a Fourier Samples choice. Mark it `status: convention`
and say why in a comment. Mark a value nobody has checked `status: unverified`.

### 3. Write the profile

Start from the file `fourier devices new` wrote, or from `config/devices/m8_tracker.yaml` (an
SD-card device with a path limit) or `digitakt_2.yaml` (a transfer-app device). Values look like
this:

```yaml
id: my_sampler
name: Maker My Sampler
manual:                            # the manual every citation quotes
  title: My Sampler User Manual
  version: OS 1.2
  url: https://example.com/my-sampler-manual.pdf
  sha256: <the PDF's sha256>
load: card-sync
sample_refs: path                  # projects refer to samples by path: releases lock them
paths:
  card_dir: {value: /SAMPLES, cite: sample-folder}
  folder_depth: {value: 2, status: convention}         # CATEGORY/family
  max_path_length: {value: 255, cite: path-limit}
audio:
  sample_rate: {value: 48000, cite: native-format}
  bit_depth: {value: 16, cite: native-format}
  channels: {value: stereo, cite: native-format}
  formats: {value: [wav], cite: native-format}
storage_mb: {value: 16000, status: convention}           # the card that ships with it
citations:
  native-format: {claim: plays 16-bit 48 kHz WAV, mono or stereo, page: 31, quote: "..."}
  sample-folder: {claim: samples load from /SAMPLES, page: 12, quote: "..."}
  path-limit: {claim: paths up to 255 characters, page: 40, quote: "..."}
```

Fill `manual:` from step 1 with the title, version, URL and sha256 of the PDF you quoted. `page`
is the 1-based PDF page, not always the printed page number. A profile with `status: unverified`
at the top cites nothing, so remove that line once the values are cited. A value nobody has
checked can stay `{value, status: unverified}` on its own.

### 4. Check it

```bash
fourier devices show my_sampler                  # each value with its page or status
FOURIER_MANUALS_DIR=~/manuals pytest -q tests/test_device_citations.py   # every quote on its page
python tests/golden/synthetic_build.py /tmp/syn  # build the synthetic library, then:
fourier render my_sampler --from /tmp/syn/out/FourierCurated --out /tmp/render
fourier verify --from /tmp/syn/out/FourierCurated --render my_sampler
```

`FOURIER_MANUALS_DIR` is the folder holding the PDF. The test finds the manual there by its
sha256, whatever the file is called, and reads it with `pypdfium2` (in the `[dev]` extra).
Without the variable or a matching file, it skips the page check. The loader refuses a value
without a source, and the citation test fails when a quote isn't on its page. Then load a render
on the device and try it. Note in the pull request what you checked on the hardware.
