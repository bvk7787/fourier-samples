# Getting started on a Mac

This page is for you if you make music and are at home in your DAW and Elektron Transfer, but
have never used Terminal. It goes from nothing to a set of sounds on your Digitakt 2 (or M8,
or another sampler), one step at a time, saying what you should see after each step.

Fourier Samples reads your sample packs and picks several thousand of the best sounds: about
8,000 to 11,500 from a big library, depending on the style, and fewer from a few packs. It sorts
them into folders named for what they sound like (kicks, snares, drum loops by tempo, pads, and
so on) and converts them for your sampler. **Your sample packs are never changed, moved or
renamed.** Fourier only reads them and writes its own folders next to your music.

## What you need

| | |
|---|---|
| **A Mac with Apple Silicon** (M1 or later) | That's what it's tested on. Intel Macs are untested. Linux works too, but this page is written for the Mac. Windows doesn't work yet. |
| **Free disk space** | About 25 GB for the full set from a large library: the curated set, a copy for your device, a saved release, and the downloads. About 5 GB for the starter set. |
| **An internet connection** for setup | It downloads about 800 MB once: the listening model and the software it runs on. After that it works offline. |
| **Time** | 10 to 20 minutes to install and set up. The first build listens to every sample: about an hour for each 100,000 files, and you can leave it running. Later builds take minutes. |
| **Your samples on this Mac** | On its own disk or an external drive. If they're in iCloud Drive, right-click the folder in Finder and choose **Keep Downloaded** first. |
| **Your sampler's own loader** | Elektron Transfer for a Digitakt 2. For an M8, its SD card. |

It's free and open source, and it sends nothing about you or your library anywhere.

## 1. Open Terminal

Terminal is an app that comes with every Mac. Open it from **Applications > Utilities >
Terminal**, or press **Cmd-Space**, type `Terminal` and press **Return**.

A window opens with a line ending in `%`. That's where you type. Every command on this page is
in a gray box. To run one:

1. Copy the line from this page (select it, Cmd-C).
2. Click in the Terminal window, paste (Cmd-V) and press **Return**.
3. Wait until the `%` line comes back. That means it's finished.

Anything after a `#` in a box is a note for you, and you don't need to type it. To stop a command
that's running, press **Ctrl-C** (the control key and C). Nothing on this page needs your Mac's
password.

## 2. Install it

First install uv, the installer Fourier uses. It also brings the Python that Fourier is written
in, so you don't need to install that yourself:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It ends with **everything's installed!** Now **quit Terminal (Cmd-Q) and open it again**, so
it finds what was just installed. Then:

```bash
uv tool install fourier-samples
```

It ends with **Installed 1 executable: fourier**. Check it works:

```bash
fourier --version
```

You should see `fourier (Fourier Samples) 0.1.0` (or a later version).

**`command not found`?** Quit Terminal and open it again, then retry. Still not found? Run
`uv tool update-shell`, then quit and reopen Terminal once more.

## 3. Try the demo (two minutes, nothing to lose)

```bash
fourier demo
```

It makes a small made-up library of test sounds, builds a curated set from it and converts it
for the Digitakt 2 and the M8. Everything goes into one new folder, `fourier-demo`, in your home
folder, and nothing else on your Mac is touched. It takes a minute or two and ends with a few
lines saying where everything is and what to try next. To open that folder in Finder:

```bash
open fourier-demo
```

The demo's sounds are simple test tones; your own build will use your own samples. When you're
done, drag the `fourier-demo` folder to the Trash.

## 4. Set it up for your samples

```bash
fourier setup
```

Setup goes through six steps, numbered on screen (1/6 to 6/6). Each question says what it's
for; pressing **Return** takes the suggested answer shown in `[brackets]`, when there is one.

**1/6 Your samples.** Drag your sample folder from Finder into the Terminal window and press
Return. To add several folders, drag them in together. It counts the audio files it finds, and asks
again if there are none.

**2/6 Devices and style.**

- **Which devices?** A numbered list. Type the number next to your sampler (or its name, such
  as `digitakt 2` or `m8`) and press Return. For both, separate them with a comma, as `1, 6`.
  For a sampler that isn't listed, pick `generic_sd_card` if it reads an SD card, or else
  `generic_folder`.
- **What do you make?** This is the style. It decides how much of each kind of sound you get
  and the drum loops' tempos. Type its number. `balanced` is a good start. The others are for
  ambient, breaks and acid, hip hop, house and techno, and trap.
- **How big a set?** `starter` (about 1,500 sounds, about 1 GB, quick to load), `standard`
  (about 4,500 sounds, 3 GB) or `full` (the style's whole set, about 8,000 to 11,500 sounds,
  5 to 7 GB). For a Digitakt 2 it says how long each takes to load with Transfer. Starting with
  `starter` is fine, since a bigger set later is one line in your settings.
- **Prepared or as-is?** `prepared` (the default) tunes melodic one-shots to C so they play in
  key across the keys. It also evens out loop levels, makes almost-mono files mono to save
  space and tidies the names. `as-is` keeps every sound at its own pitch, level and channels,
  with its original name.
- **Leave any kinds of sound out?** Press Return to keep them all, or type the numbers of any
  you never use.
- **Keep these folders?** Press Return (yes). Fourier's three folders go in your Music
  folder: `FourierCurated` (the curated set), `FourierRenders` (its copies converted for
  each sampler) and `Fourier` (your saved releases).

**3/6 The config file.** Your answers are saved. There's nothing to do.

**4/6 The CLAP model.** CLAP is an AI model that listens to each sample and describes what it
sounds like, so Fourier can sort a sound even when its file name says nothing (`Untitled
12.wav`). It runs on your Mac, not online. Say yes. It downloads about 800 MB (about 1.4 GB
once installed), which takes a few minutes.

**5/6 A local LLM.** This step asks something only if you have Ollama installed. If it does,
type `n` and press Return. It's optional and only writes folder descriptions.

**6/6 Check.** Every line is **OK** (fine), **WARN** (worth a look), **NEXT** (the first build
will do it) or **FAIL** (would stop a build, with what to do). A WARN doesn't stop anything, so
read it and carry on. Then setup offers to start the first build. Unless your library is small
(a few thousand files), type `n` and press Return. Then start the build as in section 5 below,
which keeps the Mac awake while it runs.

## 5. The first build

If setup didn't start it:

```bash
caffeinate -i fourier build
```

`caffeinate -i` keeps your Mac from going to sleep while the build runs. Keep a portable Mac
plugged in with the lid open. The build says how long it expects to take before it starts.
First it listens to every sample, the long part, with a progress bar. Then it builds and checks
the set. You can stop it with Ctrl-C at any point and carry on later with `fourier build
--resume`.

When it's done it shows a short summary:

```
Built 18 categories, 553 files:
  KICKS         74        SUB           26
  SNARES        88        SYNTH         33
  ...
1 category left empty: too few samples the rules recognize for them (WAVES).
verify: PASS -- 109 pass, 0 warn, 0 fail
New master in place at /Users/you/Music/FourierCurated
```

That one is from a single small pack; a big library gets several thousand files. **verify:
PASS** means every check passed. A category left empty just means your packs have few sounds of
that kind. To see the result:

```bash
fourier open report
```

That opens a page about the curated set (the **master**) in your web browser. It shows every
kind of sound, its families and their files, each with a play button and where it came from in
your packs. It also shows what the build left out and why. **K**, **D** and **M** next to a
file rate it Keep, Drop or Misfiled. **Export ratings** saves them as a file, and `fourier
review import ~/Downloads/ratings.csv` hands them to the next build. `fourier open` shows the
same folders in Finder, where you can select a file and press the spacebar to listen.

## 6. Put it on your sampler

First save this set as **release v1**. A release never changes, so once it's on your sampler,
the sounds your projects use stay where they are, even after later builds:

```bash
fourier publish --notes "first set"
```

Each of these steps copies or converts the whole set, so give it a few minutes on a big set.
It's done when the `%` line comes back.

**Digitakt 2:**

```bash
fourier render digitakt_2 --release v1
fourier open renders digitakt_2
```

The second line opens the converted folders in Finder. Check they fit: in your Music folder,
open `FourierRenders`, select the `digitakt_2` folder and press Cmd-I to see its size. (Or
select the numbered folders and press Ctrl-Cmd-I for one summary.) Compare it with the free
space Transfer shows on the +Drive: 20 GB in all, shared with whatever is on it already.

Then connect the Digitakt 2 over USB, open Elektron Transfer, make a folder on the +Drive (such
as `/fourier`) and drag all the numbered folders (`00_KITS`, `01_KICKS`, ...) into it.

Transfer copies over USB slowly, and a full set of several GB takes hours. The Mac must not
sleep until it finishes. In a Terminal window, run `caffeinate -i`. It shows nothing while it
runs, and that's expected. Leave it until Transfer is done, then press Ctrl-C.

On the Digitakt 2 the folders are in the +Drive's sample browser. `01_KICKS` to `19_VOX` hold
the categories, each split into families of similar sounds. `00_KITS` holds drum kits (a kick,
snare, hats and more that go together). `00_SLICE` holds drum loops cut to an even length,
ready for slicing. Load sounds into a project as you would any sample. A project holds up to
400 MB of samples (the Digitakt 2 manual, p.17), so you load the ones you use, not the whole set.

**M8:** put the M8 in disk mode so its card shows up in Finder's sidebar. In the second line
below, replace `M8` in `/Volumes/M8` with the card's name as Finder shows it:

```bash
fourier render m8_tracker --release v1
fourier sync m8_tracker /Volumes/M8
```

It copies into `/Samples/Fourier` on the card. Eject the card in Finder before you unplug.

**Another sampler:** `fourier render generic_sd_card --release v1` (or `generic_folder`),
then `fourier open renders generic_sd_card` and copy the folders the way your sampler expects.

**Later, to update it:** a release on your sampler stays as it is. Changes (new packs, new
settings, your ratings) go into the next release, v2. It keeps every sound v1 put on the
sampler where it was and adds the new ones:

```bash
fourier build
fourier publish --notes "new packs"
fourier render digitakt_2 --release v2 --new-only
fourier open renders digitakt_2-new-in-v2
```

`--new-only` also puts just the files v2 adds into a folder of their own, laid out in the same
folders as on the device. Drag those into the same `/fourier` folder in Transfer, so you never
copy the whole set again. On the M8, `fourier sync m8_tracker /Volumes/M8` copies only what
changed.
[The guide](guide.md#3-releases-keeping-your-projects-working) has more.

## 7. Change things

Your settings are in a small text file. To open it in TextEdit:

```bash
fourier config edit
```

Each line is `name = value`. Add a line like one of these, or change it if the file already
has a line with that name. Keep the quotes and brackets exactly as shown:

```toml
preset = "house-techno"            # the style
size = "4GB"                       # make the set fit a card or a drive
categories = { BLIPS = "off" }     # leave a category out
words = { KICKS = ["bombo"] }      # your packs' own word for a kick
```

Save it (Cmd-S), then run `fourier build` again. If a line has a mistake, the next command
says which line. `fourier config edit` opens the file again to fix it, and `fourier config show`
lists the settings in effect. To get the changes onto your sampler, publish the next release
(step 6, "Later, to update it").

## 8. If something goes wrong

- **`fourier doctor`** checks everything and says what to do about anything that's wrong.
- **A build stopped:** your curated set is unchanged until a build finishes. Run `fourier
  build --resume` to carry on.
- **Undo a build:** the previous set is kept next to it. In Finder, open your Music folder,
  rename `FourierCurated` to `FourierCurated-bad`, then rename `FourierCurated.prev` to
  `FourierCurated`.
- **A sound in the wrong folder:** `fourier why "part of its name"` says why it's there, and
  `fourier review rate "part of its name" misfiled --to SNARES` moves it in the next build.
- **Every command explains itself:** `fourier --help`, or `fourier build --help` for one.
- **Stuck?** Ask in [Discussions](https://github.com/bvk7787/fourier-samples/discussions)
  with what you ran and what `fourier doctor` says. Something broken? Open an
  [issue](https://github.com/bvk7787/fourier-samples/issues). If a command stopped with "The
  details, for a bug report", attach that file.
- **Your sets are copies of your samples.** The curated set, the renders and the releases hold
  the sounds from your packs, under their licenses. Keep them for your own use, and don't share
  or upload them.

## 9. Uninstall

```bash
uv tool uninstall fourier-samples
```

Your curated sets in your Music folder (`FourierCurated`, `FourierRenders`, `Fourier`) are
yours to keep or delete: they're plain WAV files. Fourier's own files are in hidden folders.
To remove them, in Finder choose **Go > Go to Folder** (Cmd-Shift-G), paste each of these and
drag the folder to the Trash:

- `~/.fourier` (its database and caches)
- `~/.config/fourier` (your settings)
- `~/.cache/huggingface/hub/models--laion--clap-htsat-unfused` (the CLAP model)

## Words used here

| Word | Means |
|---|---|
| **library** | Your sample packs: the folders you pointed setup at. Only ever read. |
| **master** | The curated set Fourier builds: `~/Music/FourierCurated`, one folder per kind of sound. |
| **family** | A folder of similar-sounding samples inside a category (`KICKS/tight-punchy`). |
| **render** | The master converted for one sampler (its sample rate, folders and name lengths). |
| **release** | A saved copy of the master that never changes (`v1`, `v2`, ...): what you load onto a device. |
| **CLAP** | An AI model that listens to a sound and describes it, so Fourier can sort sounds by how they sound, not just their names. It runs on your Mac. |
| **style** (preset) | Which kinds of sound and which tempos the master favors. |
| **`~`** | Your home folder (the one with your name, holding Music, Documents, ...). |
| **`fourier.toml`** | Your settings file. `fourier config edit` opens it. |

Next: [the guide](guide.md) covers adding packs to a release, rating sounds, and more.
