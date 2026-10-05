"""The canonical label vocabulary and each provider's mapping into it.

Taxonomy categories accept canonical labels ("kick", "fx.sweep", "class.oneshot"); a
provider's own names ("Perc Kicks") are mapped in config/providers/<provider>.yaml and
nowhere else. metadata/store.py writes the mapped labels (kind "canonical") next to the
provider's raw ones, so both can be read and compared.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from ..paths import config_dir  # noqa: E402

PROVIDERS_DIR = config_dir() / "providers"

# The vocabulary. Dotted names group a family (fx.*, perc.*, cymbal.*); "class.*" is the
# sample's shape.
CANONICAL = frozenset({
    "class.oneshot", "class.loop",
    # drums
    "kick", "snare", "clap", "snap", "hat", "tom", "cymbal.crash", "cymbal.ride",
    "perc.hand", "perc.wood", "perc.metal", "perc.scrape", "scratch", "zap",
    # tonal
    "bass", "lead", "blip", "pad", "stab", "bell", "vocal",
    # keys by name (rhodes, wurli, e-piano, organ): PIANO's, which no other category votes for
    "keys",
    # fx
    "fx.smash", "fx.rustle", "fx.impact", "fx.nature", "fx.noise", "fx.sweep", "fx.whoosh",
})
KINDS = {"classes": "class", "categories": "category"}      # mapping file section -> labels.kind


@lru_cache(maxsize=None)
def mapping(provider: str) -> dict:
    """{(labels.kind, raw label): canonical label} for a provider (empty if it has no file)."""
    path = PROVIDERS_DIR / f"{provider}.yaml"
    if not path.exists():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    out = {}
    for section, kind in KINDS.items():
        for raw, canon in (doc.get(section) or {}).items():
            if canon not in CANONICAL:
                raise ValueError(f"{path.name}: {raw!r} maps to {canon!r}, not a canonical label")
            out[(kind, str(raw))] = canon
    return out


def canonical(provider: str, kind: str, raw: str) -> str | None:
    return mapping(provider).get((kind, raw))
