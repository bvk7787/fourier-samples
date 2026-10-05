"""Which metadata providers a build uses.

  sononym   a classifier: classes (one-shot / loop), categories and measurements, from
            Sononym's analysis (`fourier tools scan`)
  ableton   tags: Live's auto-tags (`fourier tools scan --only ableton`), extra category votes
  path      fallback classifier: labels from folder and file names (metadata/shadow.py)
  audio     fallback classifier: one-shot or loop from Fourier's own measurements
  fourier:sound  the sound model (metadata/sound.py): a category and a shape from the CLAP
            embedding and Fourier's own measurements, when there are weights trained on this
            library (metadata/train.py).
            Without Sononym it places what the names leave open (curate.compute_homes); with
            Sononym it is a report. It is on whenever its weights are, not named here

By default a build uses Sononym and Live when the library has their data, and the
fallback classifiers take Sononym's place when it doesn't (a library with neither still
builds, from its file names and audio). fourier.toml can name them instead:

    providers = ["sononym", "ableton"]      # or ["ableton"], or ["path", "audio"], ...

Without Sononym, path and audio are the classifier (a sample is a candidate when path has
looked at it, which is every file in the library).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text

SONONYM, ABLETON, PATH, AUDIO = "sononym", "ableton", "path", "audio"
SOUND = "fourier:sound"         # metadata/sound.py: on with its weights, never in `providers`
KNOWN = (SONONYM, ABLETON, PATH, AUDIO)
FALLBACK = (PATH, AUDIO)


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Active:
    names: tuple            # every provider in use
    classifiers: tuple      # the ones whose labels and measurements route samples, in priority order
    tags: tuple             # tag providers (extra category votes)
    source: str             # "fourier.toml" or "auto"

    @property
    def fallback(self) -> bool:
        return SONONYM not in self.classifiers


def configured() -> list[str] | None:
    """The names fourier.toml lists (`providers = [...]`), or None for automatic."""
    from ..layers import _read, find_config
    path = find_config()
    names = _read(path).get("providers") if path else None
    if names is None:
        return None
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ProviderError(f"{path}: providers must be a list of names")
    bad = [n for n in names if n not in KNOWN]
    if bad:
        raise ProviderError(f"{path}: unknown providers {', '.join(bad)} (known: {', '.join(KNOWN)})")
    return names


def has_data(session, name: str) -> bool:
    sql = {SONONYM: "SELECT 1 FROM sononym_meta LIMIT 1",
           ABLETON: "SELECT 1 FROM samples WHERE ableton_tags IS NOT NULL LIMIT 1"}.get(name)
    return True if sql is None else session.execute(text(sql)).first() is not None


_CACHE: dict = {}


def active(session) -> Active:
    """The providers this database and config use (worked out once per process)."""
    key = str(session.get_bind().url)
    if key not in _CACHE:
        names = configured()
        source = "fourier.toml"
        if names is None:
            source = "auto"
            names = [p for p in (SONONYM, ABLETON) if has_data(session, p)]
        classifiers = (SONONYM,) if SONONYM in names else FALLBACK
        tags = (ABLETON,) if ABLETON in names else ()
        _CACHE[key] = Active(tuple(dict.fromkeys([*classifiers, *tags, *names])), classifiers, tags,
                             source)
    return _CACHE[key]


def forget() -> None:
    """Tests, or after changing the config: work it out again."""
    _CACHE.clear()


def describe(a: Active) -> str:
    fb = " (no Sononym: file names and audio classify)" if a.fallback else ""
    return f"providers: {', '.join(a.names)} [{a.source}]{fb}"
