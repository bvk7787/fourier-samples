"""Overridable tunables: one resolved config, applied where each tunable is defined.

Every tunable in the curation modules (config/tunables.yaml) is defined as

    LOOP_LIMIT_DB = _t("LOOP_LIMIT_DB", 3.0)

where _t = for_module("curate_config"). With no resolved config the default comes back
unchanged, so behavior is exactly the code's. With $FOURIER_RESOLVED_CONFIG pointing at a
resolved.json, an override for "curate_config.LOOP_LIMIT_DB" replaces the default at the
moment it is defined. So everything computed from it afterwards follows: derived values
(ACOUSTIC_ABLETON_TAGS from ORCH_ABLETON_TAGS), default arguments bound at def time
(_limit(ms=LIMIT_MS)), names other modules import, and spawned workers, which inherit the
environment variable.

resolved.json is {"version": 1, "values": {"<module>.<NAME>": <encoded value>}}. The
encoding keeps Python types JSON loses (tuples, sets, regexes, dict key types and order);
see encode(). An override must keep its default's kind: a number stays a number (an int
default stays int), a regex stays a regex, a set stays a set. An unknown or mistyped key
is an error, never a silent no-op.

(docs/design-history.md, "How it became configurable".)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache

ENV = "FOURIER_RESOLVED_CONFIG"
VERSION = 1
_MISSING = object()


class ConfigError(ValueError):
    """A resolved config that can't apply: bad file, unknown key, wrong kind."""


# ---------------------------------------------------------------------------
# Encoding: JSON that round-trips the tunables' Python values
# ---------------------------------------------------------------------------

def encode(v):
    """A JSON-safe form of v that decode() turns back into an equal value of the same type."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, re.Pattern):
        return {"$re": v.pattern, "flags": v.flags}
    if isinstance(v, tuple):
        return {"$tuple": [encode(x) for x in v]}
    if isinstance(v, (set, frozenset)):
        items = sorted((encode(x) for x in v), key=lambda e: json.dumps(e, sort_keys=True))
        return {"$frozenset" if isinstance(v, frozenset) else "$set": items}
    if isinstance(v, list):
        return [encode(x) for x in v]
    if isinstance(v, dict):
        return {"$dict": [[encode(k), encode(x)] for k, x in v.items()]}   # order kept
    raise ConfigError(f"can't encode a {type(v).__name__}")


def decode(e):
    if isinstance(e, list):
        return [decode(x) for x in e]
    if not isinstance(e, dict):
        return e
    if "$re" in e:
        return re.compile(e["$re"], e.get("flags", 0))
    if "$tuple" in e:
        return tuple(decode(x) for x in e["$tuple"])
    if "$set" in e:
        return {decode(x) for x in e["$set"]}
    if "$frozenset" in e:
        return frozenset(decode(x) for x in e["$frozenset"])
    if "$dict" in e:
        return {decode(k): decode(x) for k, x in e["$dict"]}
    raise ConfigError(f"unknown encoded value {e!r}")


def canonical(values: dict) -> str:
    """One string per resolved config: the same values always give the same text."""
    return json.dumps({k: encode(values[k]) for k in sorted(values)}, sort_keys=True,
                      separators=(",", ":"))


def config_hash(values: dict) -> str:
    return hashlib.sha256(canonical(values).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The resolved config
# ---------------------------------------------------------------------------

def write_resolved(path, values: dict) -> None:
    """Write a resolved.json holding these "<module>.<NAME>" overrides."""
    doc = {"version": VERSION, "values": {k: encode(v) for k, v in sorted(values.items())}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)


@lru_cache(maxsize=1)
def _load(path: str | None) -> dict:
    if not path:
        return {}
    try:
        with open(os.path.expanduser(path)) as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise ConfigError(f"{ENV}={path}: can't read it ({e})") from None
    if doc.get("version") != VERSION:
        raise ConfigError(f"{ENV}={path}: version {doc.get('version')!r}, expected {VERSION}")
    return {k: decode(v) for k, v in (doc.get("values") or {}).items()}


def overrides() -> dict:
    """The active overrides, "<module>.<NAME>" -> value (empty without the env var)."""
    return _load(os.environ.get(ENV) or None)


def reset() -> None:
    """Forget the loaded file (tests, or after changing the env var)."""
    _load.cache_clear()


_seen: dict[str, set] = {}   # module -> the tunable names it defined (for check_keys)


def _coerce(key: str, value, default):
    if default is None:
        return value
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, (int, float)):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            value = type(default)(value)
    elif isinstance(default, re.Pattern):
        ok = isinstance(value, re.Pattern)
    elif isinstance(default, (set, frozenset)):
        ok = isinstance(value, (set, frozenset))
        value = type(default)(value) if ok else value
    else:
        ok = isinstance(value, type(default))
    if not ok:
        raise ConfigError(f"{key}: expected a {type(default).__name__} like its default, "
                          f"got a {type(value).__name__}")
    return value


def for_module(module: str):
    """The _t(name, default) a curation module wraps its tunables in."""
    def _t(name: str, default):
        key = f"{module}.{name}"
        _seen.setdefault(module, set()).add(name)
        v = overrides().get(key, _MISSING)
        return default if v is _MISSING else _coerce(key, v, default)
    return _t


TUNABLE_MODULES = ("curate_config", "curate", "naming", "sets")


def load_all_and_check() -> None:
    """Import every curation module (so each tunable has been defined once), then fail on
    any override key none of them took."""
    import importlib
    for m in TUNABLE_MODULES:
        importlib.import_module(f"fourier.packs.{m}")
    check_keys()


def effective() -> dict:
    """"<module>.<NAME>" -> the value in use now, for every tunable of every curation
    module (overrides applied, derived values included)."""
    import importlib
    out = {}
    for m in TUNABLE_MODULES:
        mod = importlib.import_module(f"fourier.packs.{m}")
        for n in sorted(_seen.get(m, ())):
            out[f"{m}.{n}"] = getattr(mod, n)
    return out


def tunables_hash() -> str:
    """One hash of every tunable's effective value: equal hashes, equal settings."""
    return config_hash(effective())


def check_keys(known: set[str] | None = None) -> None:
    """Fail on override keys that no tunable took: a typo or a derived value."""
    if known is None:
        known = {f"{m}.{n}" for m, names in _seen.items() for n in names}
    unknown = sorted(set(overrides()) - known)
    if unknown:
        raise ConfigError(f"unknown or non-overridable tunables: {', '.join(unknown)}")
