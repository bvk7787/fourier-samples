"""Where Fourier keeps its own state.

Everything Fourier writes for itself (the library database, ratings, build archive,
caches, the CLAP index, release staging) lives under one home
folder: $FOURIER_HOME, or ~/.fourier when it isn't set. The narrower overrides that
already exist (FOURIER_RATINGS, FOURIER_AUDIO_CACHE, FOURIER_RENDER_CACHE,
FOURIER_CLAP_TEXT_CACHE, FOURIER_STAGING_DIR, ...) still win over it,
so a harness or a test can relocate all of it with one variable and still pin a single
piece somewhere else.

Nothing here creates folders; callers do that when they write.
"""
from __future__ import annotations

import os
from pathlib import Path


def fourier_home() -> Path:
    """$FOURIER_HOME, or ~/.fourier. Read on every call, so a changed env var applies."""
    env = os.environ.get("FOURIER_HOME")
    return Path(env).expanduser() if env else Path.home() / ".fourier"


def config_dir() -> Path:
    """The shipped configuration (devices, presets, providers, taxonomy, tunables): <repo>/config
    in a source checkout, fourier/_config in an installed wheel."""
    here = Path(__file__).resolve()
    repo = here.parents[2] / "config"
    if (repo / "tunables.yaml").exists():
        return repo
    return here.parent / "_config"


def source_checkout() -> Path | None:
    """The repository root when running from a source checkout, else None."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "config" / "tunables.yaml").exists() and (root / "pyproject.toml").exists() else None


def home_path(*parts: str) -> Path:
    """A path under the Fourier home folder."""
    return fourier_home().joinpath(*parts)


def clap_index_path() -> Path:
    """The CLAP search index: $FOURIER_CLAP_INDEX, else <home>/clap_index.npz.

    Its memory-mapped .npy copies sit next to it, wherever it is.
    """
    env = os.environ.get("FOURIER_CLAP_INDEX")
    return Path(env).expanduser() if env else home_path("clap_index.npz")
