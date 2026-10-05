# Contributing

Thanks for looking. Fourier Samples is a small project, kept by one person in their spare time:
issues and pull requests are welcome, and answers are best effort. Questions and help getting
started go in [Discussions](https://github.com/bvk7787/fourier-samples/discussions) (Q&A);
issues are for bugs and feature requests. Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md); report
security problems privately, as [SECURITY.md](SECURITY.md) says.

## Setup

```bash
git clone https://github.com/bvk7787/fourier-samples.git && cd fourier-samples
uv sync --extra dev                # the locked versions (uv.lock); add --extra clap for real builds
uv run fourier demo                # the whole pipeline on a generated library, in ./fourier-demo
```

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"`, then
drop the `uv run` below.

## Running the checks

```bash
uv run pytest -q -n auto tests                                   # the suite, in parallel; needs no sample library
uv run pytest -q -n auto -m "not slow" tests                     # ...without the end-to-end runs (a quick pass)
uv run python tests/golden/synthetic_build.py /tmp/syn --check   # the synthetic golden build
uv run ruff check src tests                                      # lint
uv run mypy                                                       # types (settings in pyproject.toml)
```

The golden build generates 280 samples, builds them end to end with a stand-in CLAP (no model,
no download) and compares the result field by field with `tests/golden/synthetic-<os>.json`.
`--without sononym,ableton`, `--preset NAME`, `--scale` and `--interrupt KICKS,PADS,VOX` run
the variants CI runs (only the golden and the interrupted build are compared with the stored
result; the others must pass verify). `tests/test_first_run.py` is a first run end to end, one process per command, without
Sononym or Live.

`-n auto` runs one worker per CPU (pytest-xdist, in the dev extra); drop it to run serially,
which `--pdb` needs. `conftest.py` gives each worker its own `$HOME`, Fourier home, temp
folders and numba cache, and distributes with `--dist loadgroup`: the end-to-end modules that
share one sandbox across their tests are marked `xdist_group` so one worker runs them in order.
A test that keeps state in a module-scoped fixture its later tests depend on needs the same
mark. The end-to-end tests are marked `slow`.

[CLAUDE.md](CLAUDE.md) describes the architecture and the rules the code keeps; read it before
a larger change.

## What a pull request needs

- **Tests.** A behavior change comes with a test. Curation changes also run the synthetic
  golden (`python tests/golden/synthetic_build.py /tmp/syn --check`); if the change is meant to
  alter the output, regenerate the golden with `--update` and say in the pull request what moved
  and why.
- **The CLI tree.** A command, option or help-text change updates the snapshot
  (`python tests/test_cli_tree.py --write`).
- **Settings.** A new tunable gets a class line in `config/tunables.yaml` and a snapshot
  update; a setting users should reach is a knob, not only an `[advanced]` key.
- **No one's library.** Code, tests and docs name no real sample pack or vendor; rules that
  match particular packs belong in a user's overlay (class `library`, empty in the code). Tests
  use invented names (Acme, Northwind, ...).

## Device profiles

New devices are welcome: [docs/device-profiles.md](docs/device-profiles.md) walks through drafting a profile from the
manual, with a page citation for every value, and checking the quotes against the PDF
(`FOURIER_MANUALS_DIR`). Say in the pull request what you checked on the hardware. A profile
made with `fourier devices new` and tried on the hardware, but not yet cited, is welcome in
Discussions (Show and tell), so others with that sampler can use it and help cite it.

## Presets

A preset (`config/presets/<name>.yaml`) is a set of knobs and, if needed, `[advanced]` values,
optionally `extends` another. A new preset should build the synthetic library and pass verify
(`python tests/golden/synthetic_build.py /tmp/syn --preset <name>`), and say in a comment at its
top what music it's for and what it changes from `balanced` (the code's defaults). A genre style
can `extends: breaks-acid` and weight its budgets, as the others do.

## Releases

A maintainer bumps `version` in `pyproject.toml`, moves the changelog's entries under the new
version, and pushes a tag `v<version>`: `.github/workflows/release.yml` builds the package,
checks that the wheel installs and `fourier demo` runs, and publishes it to PyPI with Trusted
Publishing (no token is stored). The tag has to match the version.

Sound-model weights are never committed: they're learned from someone's licensed samples, so
each user trains their own (`fourier tools train`).

## What we won't take

Fourier Samples doesn't accept code that circumvents copy protection or DRM, in this repository or
as a plug-in, and requests for it are closed.

Preparer plug-ins (`fourier.preparers`) add open or openly documented formats that soundfile
doesn't read (WavPack, say), for files with their own extensions (`fourier tools import-folder`). A
preparer must not read encrypted or access-controlled audio.
