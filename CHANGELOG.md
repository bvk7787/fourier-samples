# Changelog

All notable changes to Fourier Samples are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) (before 1.0, a minor version may
change the CLI or the config).

## [Unreleased]

### Added

- `phrases.csv` beside `loops.csv` in the master: each phrase's folder, file and tempo, and
  the key its pack states for it, from its file name, the nearest folder that names one, or the
  files beside it when they agree, with where the key came from. Keys are never guessed from
  the audio: measured on loops whose names state one, no detector tried was right often enough
  (curation.md, section 13).
- One recording picked by two categories under two names (a pack shipped twice, renamed, or
  as mono and stereo, 44.1 and 48 kHz, 16 and 24 bit editions) now stays in one: picks in
  different categories at a CLAP cosine of 0.88 or more (`CROSS_COPY_COS`, a loose prefilter,
  since CLAP puts one recording's editions as low as 0.966) are compared as audio, a mono
  edition also against each side of a stereo one, and copies (cross-correlation 0.99 or more,
  `COPY_XCORR`) keep one, by the rules a source picked twice already follows. `fourier why`
  names the copy that stayed, and `fourier verify` checks the master for any left. Found by a QA sweep: a piano chord in both STABS and SYNTH,
  and an audit: chords in both PIANO and STABS, a guitar hit in both ACOUSTIC and BLIPS.

### Fixed

- Every rule that leaves a file out of a category now writes its own reason to the why log, so
  a build can say why each file left: the multisample set cap (`sibling_capped`, with the notes
  the folder keeps; its first pass is now counted, so the build log's "extra notes of
  multisample sets left out" line prints), one note per C (`one_note_per_c`, with the note kept
  and the C), long wavetables, the FX-folder rule, Drop and Misfiled ratings (the rated sibling
  named), routing to another category and the rest. `over_budget` now means only a candidate
  the budget passed over: the set cap's files were logged as over budget, which read as a
  category whose pool ran out. The build's CHANGELOG.md counts the files that left or joined
  by reason, the full list goes to `<stamp>.changes.csv` beside the archived build (never into
  the master), and `fourier verify` warns on a file that left with no logged reason. The
  files a build picks are unchanged.
- A rating reaches every edition of its recording: a Drop or Misfiled covers the file's byte
  copies (a pack's second copy) and its editions (a `.aif` beside the dropped `.wav`, a
  compilation's re-release), so a dropped file no longer comes back under another path. A
  dropped multisample note's slot no longer refills from its own set (a note the previous
  build held stays). A Misfiled covers its multisample set by name (identity's set tier: one
  source folder, notes, velocities, round robins) instead of any similar-sounding name of the
  pack (CLAP 0.75, `SIBLING_COS`), so it no longer reaches a numbered series. Plain drops stay
  note-level; `fourier review rate <file> drop --set` (and `misfiled --set`, or a `scope`
  column of `set` in `review import`) gives a rating to the whole set or patch.

- One meaning of "the same sound" (`packs/identity.py`, curation.md "Identity"): a recording,
  its editions, a multisample set and a numbered series, asked by the copy prune, the set cap,
  one note per C, round robins and kit takes. WAVES no longer loses single cycles as copies
  (a sine and a triangle cycle correlate at 0.99): it pairs editions by name only. A copy's
  keeper stays through its folder's new pick (two recordings left the master as copies of a
  keeper the pick then dropped). A raw take and its processed edition ("_Cleanup", "_Dirty",
  "-Clean", "-Tube") count as one recording. The set cap binds a described patch's variant
  letters ("A 01" to "C 09") and folds its editions, keeps to one source folder, and now holds
  in the drum categories for true sets (notes, velocities, round robins) only, never a
  numbered series. Bare numbers that track the measured pitch are notes for one note per C
  ("Lead 059" / "061" / "062" all on C3). Round robins of one note and velocity keep one, and a
  kit no longer takes two takes of one drum named by articulation or mic ("+OH").

- `fourier verify` no longer fails a file a Keep or a Move rating pinned into a banded
  category (FX, CYMBALS, PHRASES, ...) on its band: the build bands it on the labels the
  database holds, while its manifest entry records only the votes of where it came from, which
  can read as another band. Four files moved into FX by ratings stopped a build on it.

- Wavetables longer than 1 s (`WAVE_MAX_DUR`) under a wavetable / single-cycle folder no
  longer land in SUB, FX, PADS, SYNTH, STABS or ACOUSTIC, where a sampler played them as a
  buzz stepping through their frames: like impulse responses they are out of scope, in no
  category (Keeps included; `fourier why` says so, and verify fails on one). An audit found 35
  in a master, from a library holding about 1,300 more. And a wavetable pack's kit of single
  cycles (6 ms "kicks" and "hats" in a drum-kit folder, which kept them out of WAVES) now goes
  to WAVES: a file of 12 ms or less (`WAVE_KIT_CYCLE_MAX_S`) in such a folder is a cycle,
  while the kit's real hits still home as drums.
- Notes of one multisample patch are now one set however the name states the note: glued to
  the patch name ("warmpadF#2"), below octave 0 ("A-1"), before a pack's 4-character ID
  ("Pad_F#3-XQ1Z", "Pad-C3-127-86XQ") or with a sampler's hex code after it ("Pad_f#34b00ff").
  Before, each such note looked like a set of its own, so the one-note-per-C rule and the
  sibling cap never fired: up to 8 notes of one patch sat in a folder, and notes retuned onto
  one C in a category stayed side by side (an audit found 35 such groups). A detected root
  counts for the one-note-per-C rule when the name states the note, and a pYIN root never
  retunes a note onto a C its set already plays.
- A file retuned by Sononym's detected root now needs a steady voiced note in pYIN, as a pYIN
  root does (half the frames voiced, spread under 0.5 semitone); sung phrases, an unvoiced
  reversed shout and a cowbell were retuned by a pitch read off part of them.
- FX keeps at most 2 notes of one multisample set a folder (`SIBLING_CAP_BY_CAT`): 106 FX files
  were notes of 38 patches.
- Hats are banded open or closed by how long they sound, not by the length of their file: a
  hat no name calls open or closed was open from 0.35 s of source, silence included, so a
  10 ms tick or a 0.1 s closed hat in a 1.4 s file landed in an open-hat folder and could
  play a kit's open hat. The length is now the hat's attack plus its decay to -40 dB (Fourier's
  own analysis), within the file. "cl_hat" / "ClHH" names a closed hat. A required kit slot no
  longer falls back to a file its name rules out (a closed hat as the open hat; a kit without an
  open hat drops out), "+OH" and a pack prefix like `XX4d` no longer read as a hat word there,
  and a kit's two toms (two percs) aren't two takes of one drum ("_p2" / "_f2") when the pack
  has another.
- Routing where a name misleads (found by a sample-level audit): a pack prefix like `xx_yy000`
  ("sd_" and a code) yields to a folder that names the drum, so toms, hats and kicks of such a
  kit no longer land in SNARES; "Oh" (a word) is no open hat, nor is "oh" in a voice folder
  (`DRUM_ABBR_NOT_FOLDER_RE`); a drum word beside a bass, pad, string, lead or chord word
  (`DRUM_NAME_TONAL_RE`) decides nothing when the name carries a note or Live and the classifier
  agree on no drum (a C2 "snap" bass in CLAPS, a track title with "Snares" in SNARES). Bass,
  music and effects loops Live tags "Drum Loop" are out of DRUMLOOPS (`DRUMLOOP_MUSIC_DIR_RE`,
  `DRUMLOOP_KEY_NAME_RE`, `DRUMLOOP_BASS_NAME_RE`, `FX_FOLDER_RE`), and so are sample chains
  (`SAMPLE_CHAIN_PATH_RE`: slice-grid folders, g16 / g32 / g64 names), which reached 00_SLICE;
  the style exclusions read "Synth_Bass" and "FX_LOOPS" as well as "Synth Bass". A file under an
  FX folder the classifier labels FX only, or not at all, is no pad on Live's "Atmosphere" tag
  (`FX_FOLDER_NOT`), and a Live Set's `Samples/Recorded` takes are demo paths.
- Loops that are mostly silence are left out of DRUMLOOPS and PHRASES (and so of 00_SLICE):
  nothing checked how much of a loop sounds, so ending takes, fills that stop after a bar,
  4-bar loops with an empty last bar and one note followed by seconds of silence were picked.
  A loop whose silence before its first and after its last sound, plus its longest silent
  stretch inside, is over 45% of its length (`LOOP_SILENT_SHARE_MAX`, `LOOP_SILENT_DB`) is
  left out (`fourier why`: mostly silence). Found by an audit: 34 master loops at least 30%
  silent, the worst 83%.
- Named fills and rolls and swung drum loops that start late behind silence are now rotated to
  their downbeat: the quiet-head rule ran only after the name and swing checks, which skip
  rotation because such loops' off-grid hits are feel. A late start the loop is whole beats
  without (and not with) is cut instead of rotated, so it no longer runs long each time round
  (`trim_ms`). Found by an audit: 7 drum loops starting 16-27 ms late.
- One recording picked twice by one category stays once. A pack shipped in two editions
  (16 and 24 bit, an older sampler's and a newer one's) or as a long and a short cut of
  each hit put both copies in the master, because the near-duplicate prune only compares CLAP
  (above 0.985) and CLAP heard these pairs at 0.77 to 0.98. Each category now compares its
  picks above a CLAP cosine of 0.75 (`CATEGORY_COPY_COS`) as audio, after a quick look at their
  first second (`COPY_GATE_XCORR`, `COPY_GATE_SECONDS`, `COPY_GATE_RATE`; loops at other tempos
  never), keeps the longer sound, then the deeper bit depth (a Keep always stays), and refills
  the folder. The audio check now holds the shorter sound to the longer one from its start to
  where the shorter ends (at least half its energy, `COPY_SHARE`), so a shorter cut of a hit
  counts as a copy here, in the instrument categories' prune and across categories. `fourier why` names the
  copy that stayed, and `fourier verify` fails on any left. Found by an audit: 16 such pairs in
  one master, from hats, kicks and toms to an acid loop shipped under two names.
- Device path locks now follow a library that moved (another volume, a renamed library, vendor
  or pack folder). A lock names each source by its absolute path, so after a move a render
  found none of them: every file got a second copy on the card under a new name ("_2") beside
  its locked twin, and `render --dry-run` still said SAFE. A source the lock doesn't know now
  keeps the path of a locked source gone from the master when they are the same master file
  (master path and audio) or the same library path (vendor/pack/file), one to one; rendering a
  release re-keys the entry, and existing locks need no migration. `render --dry-run` is NOT
  SAFE when a new file repeats a retired locked file's audio in its category (DUPLICATE). Found
  by an audit: a library root change left 0 of 12,579 locked paths kept.
- Routing names that misread a pack's prefix, an ID token or a mic tag. A drum abbreviation
  glued to a number and more letters ("XX4d-...") no longer names a drum (`DRUM_NAME_RULES`,
  the hat bands and the drum words of `STAB_NAME_EXCLUDE` and FX's name filter; "CH01" and "BD909"
  still do), and "+OH" (an overhead mic) is no open hat. A chord quality inside a
  random ID token ("..._C1-M93Q", "..._C4-SUS7") no longer makes a stab (`STAB_CHORD_NAME_RE`).
  Hats named "Clsd" or "Foot" are closed (`HAT_CLOSED_RE`), and a kit's open-hat slot passes
  over a hat named closed (`ROLE_NAME`). A one-shot named as one kind of drum that a classifier
  also files with a drum homes in that drum category ahead of the acoustic and organ
  reservations (a flute-named kick, an organ maker's drum machine snare), and ACOUSTIC and
  PIANO leave it out; `fourier why` and `fourier verify` follow. `DEMO_PATH` also takes a pack's
  "Demo <name>" folder and the renders Ableton Live writes into a set's `Samples/Processed/Crop`,
  `Consolidate`, `Reverse` and `Freeze` folders. Found by an audit of a 246,000-sample library.
- Retune to C: an F# whose fine tuning is flat goes down to the C below (-5.51) instead of
  up 6.49, so no shift is over 6 semitones (verify now holds that at 6). Notes of one
  multisample set that retune onto the same C count as one in the retuned categories (one
  note per set and C across the category's folders, the smallest shift first), so a folder no
  longer holds several copies of one chord at C3. A tonal folder's note (`c3-`, `mallet-c4-`)
  is the median of the notes its files play after the retune, judged on the files it holds.
- A tempo written in a loop's name is taken only when the loop's length agrees: a whole number
  of beats within 10 ms (`NAME_TEMPO_TOL_MS`), measured from its first sound and with a pad of
  digital silence at its end left out (`LOOP_QUIET_DB`, `LOOP_PAD_DBFS`), else snapped to
  whole bars within 3% (a "96bpm" break cut 1% long is filed at 95.07), else the rest of the
  chain decides. Before, any
  "Nbpm" was taken as written, so a loop named 100 BPM that is 16 beats at 110, or a take
  number read as 72 BPM on a 128 BPM loop, was filed, banded and sliced at the wrong tempo
  (63 of 2,500 drum loops and phrases in one library). Tempo folding no longer halves an odd
  beat count, rotation skips a loop that isn't whole beats at its tempo, and verify checks
  each drum loop's and phrase's bar math.
- `phrases.csv` reads a bare note letter after the name's tempo as its key ("XX_124_E_Lead")
  and "C Major" as well as "Cmaj", and never borrows a key from neighbouring files when the
  file's own name holds a note letter (14 of 15 borrowed keys in one library contradicted it).
- A parallel build's log now has each category's own lines (its `DONE` line with failed and
  near-silent exports, multisample notes left out, contradicted note names, pYIN roots,
  cloud-only downloads), prefixed by the category, and each category's summary line says how
  many exports failed. They were dropped before.
- The why log matches the build: a source the cross-category dedupe takes out of a category is
  recorded there (`dup_source`, with the category that kept it), the dedupe lowers the
  category's `files` and no longer its `kept` (the candidate pool), and `counts` has
  `over_budget` and `too_quiet` like the other reasons.
- Folder levelling reads a wide stereo file at the level it plays (the level-keeping mono
  downmix kits already used); the plain L+R average read such files up to 3 dB quiet (more as
  the channels cancel), so levelling turned them up too far. Verify measures the same way.
- Single cycles (WAVES) are no longer held by the +18 dB gain cap (`MAX_GAIN_DB`) meant for
  noisy one-shots: a quiet cycle now reaches the -1 dBFS ceiling.
- A one-shot whose channels cancel keeps its louder channel at the file's own peak; it could
  sit 2 to 3 dB under it.
- Verify checks the sets' (00_KITS, 00_SLICE) files against their `out_md5` too, and its
  levelling check says the 4 dB it allows.

- PIANO and ACOUSTIC no longer keep two copies of one recording that reach the library under
  two names (a pack shipped twice, renamed): the near-duplicate prune those categories skip, so
  a multisample's notes stay, now drops copies the audio confirms (cross-correlation 0.99 or
  more, `COPY_XCORR`) among pairs above a CLAP cosine of 0.95 (`INSTRUMENT_COPY_COS`, a loose
  prefilter: one recording's editions sit as low as 0.966). Two notes, two takes and two
  different chords of one pack that CLAP scores as near-identical all stay. Found by a QA
  sweep of a 246,000-sample library (11 such pairs) and an audit (a note kept twice in one
  folder, as 16- and 24-bit editions, at CLAP 0.990).
- CI only: the test jobs precompile their venv and run without writing bytecode or reading a
  user site-packages, check that a child of the test Python runs in the venv too, and a test's
  child Python that can't import an installed package (a flake on macOS runners) is reported
  with what it saw and run once more.

## [0.1.2] - 2026-10-07

### Added

- The sound model's training report gives each category's precision at every confidence
  threshold, so a weak category (often SYNTH, PADS or FX) shows where its placements stop being
  trustworthy ([#3](https://github.com/bvk7787/fourier-samples/issues/3)).

### Changed

- Without Sononym, the count of samples the library's names label (which decides whether to
  train the sound model) is kept between builds and redone only when the samples, ratings,
  settings or naming rules change: about a minute saved per build on a large library
  ([#4](https://github.com/bvk7787/fourier-samples/issues/4)).
- `vendors = "auto"` is the default, as `fourier setup` already wrote it: a config without a
  `vendors` line now detects each library folder's layout instead of always taking the first
  folder as the vendor. A config that relied on the old default and wants to keep it says
  `vendors = "first-folder"`.
- The master's `manifest.json` (format 3) records each source path relative to its library
  folder, and the library folders once, with the home folder written as `~`: a manifest no
  longer repeats your home folder in every entry, so it can be shared. Fourier reads format 1
  and 2 manifests (absolute paths) as before, and builds are unchanged.

### Fixed

- Audio is read through soundfile only: a file it can't open is reported as unreadable instead
  of falling back to audioread, which is slow, decodes differently and warns about modules
  Python 3.13 removed. Every file soundfile opens decodes exactly as before
  ([#7](https://github.com/bvk7787/fourier-samples/issues/7)).
- With one worker (`--workers 1`), the pitch and event analysis steps run in this process
  instead of a one-process pool, so no worker can die under them.
- `fourier --help` lists `tools train` with the other power tools.

## [0.1.1] - 2026-10-06

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
- Saving ratings to a bare file name (`FOURIER_RATINGS=ratings.json`) failed; it saves in the
  current folder now ([#22](https://github.com/bvk7787/fourier-samples/issues/22)).

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

[Unreleased]: https://github.com/bvk7787/fourier-samples/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/bvk7787/fourier-samples/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/bvk7787/fourier-samples/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bvk7787/fourier-samples/releases/tag/v0.1.0
