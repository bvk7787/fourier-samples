"""Config layers: presets, an overlay, fourier.toml and --set flags, resolved once.

Precedence, lowest to highest:
  1. built-in defaults (the code)
  2. the preset named in fourier.toml, with every preset it `extends`
  3. the overlay named in fourier.toml (a private preset, e.g. your library rules)
  4. fourier.toml's own knobs and [advanced] table
  5. --set KEY=VALUE on the command line
Within a file, its add_categories table comes first (below), then its knobs (categories,
tempo, loudness, ...: fourier/knobs.py), then its [advanced] table.

Where fourier.toml is found: --config PATH, else $FOURIER_CONFIG, else ./fourier.toml,
else ~/.config/fourier/fourier.toml. FOURIER_CONFIG=none turns every layer off (the
golden harness uses it). With no file and no --set, nothing is resolved and the code's
defaults apply untouched. A fourier.toml that names no preset gets DEFAULT_PRESET
(balanced, the style for a first run); name one to choose (breaks-acid is the code's
defaults).

    # fourier.toml
    preset = "breaks-acid"              # a name in config/presets/, or a path
    overlay = "~/.config/fourier/presets/my-library.yaml"
    tempo = "85-180"                    # Tier 1 knobs (fourier/knobs.py), in any layer
    [advanced]
    LOOP_LIMIT_DB = 3.0                 # or "curate_config.LOOP_LIMIT_DB"

    # a preset or overlay (YAML)
    extends: balanced
    advanced:
      TEMPO_BANDS: [60, 75, 85]

    # an overlay adding a category of its own (docs/curation.md, "Adding a category")
    add_categories:
      FIELD:
        kind: gated
        labels: [fx.nature]
        noun: field
        prompts: [...]
        budget: 60

add_categories sets curate_config.ADDED_CATEGORIES (a later layer's entry of the same name
replaces an earlier one). The knobs of that layer and every later one already see the
added categories (their folder numbers, budgets and file sizes), and an entry's `presets`
({preset: off | on | weight}) is what the presets in use say about it, applied as that
layer's categories knob.

Keys are tunable names (config/tunables.yaml), module-qualified when a name exists in
more than one module. Plain TOML/YAML values are converted to each default's kind: a list
becomes a tuple or set where the default is one, a string becomes a regex (with the
default's flags), dict keys take the default's key type.

resolve() returns the overrides and where each came from; write() puts them in a
resolved.json under $FOURIER_HOME/run and returns its path for $FOURIER_RESOLVED_CONFIG.
(docs/design-history.md, "How it became configurable".)
"""
from __future__ import annotations

import importlib
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .settings import TUNABLE_MODULES, ConfigError, config_hash, write_resolved

from .paths import config_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = config_dir() / "presets"
ENV_CONFIG = "FOURIER_CONFIG"
USER_CONFIG = Path("~/.config/fourier/fourier.toml")
MAX_EXTENDS = 8
ADDED = "curate_config.ADDED_CATEGORIES"
DEFAULT_PRESET = "balanced"       # for a fourier.toml that names none


@dataclass
class Resolved:
    values: dict = field(default_factory=dict)     # "<module>.<NAME>" -> Python value
    sources: dict = field(default_factory=dict)    # "<module>.<NAME>" -> layer label
    files: list = field(default_factory=list)      # the files read, in order
    config: str | None = None                      # the fourier.toml, if any
    preset: str | None = None                      # the preset in use, if any
    preset_default: bool = False                   # True: the config named none (DEFAULT_PRESET)

    @property
    def hash(self) -> str:
        return config_hash(self.values)


def find_config(explicit: str | None = None) -> Path | None:
    """The fourier.toml to use, or None."""
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise ConfigError(f"--config {explicit}: no such file")
        return p
    env = os.environ.get(ENV_CONFIG)
    if env:
        if env.lower() == "none":
            return None
        p = Path(env).expanduser()
        if not p.exists():
            raise ConfigError(f"{ENV_CONFIG}={env}: no such file")
        return p
    for p in (Path.cwd() / "fourier.toml", USER_CONFIG.expanduser()):
        if p.exists():
            return p
    return None


def _defaults_here() -> dict:
    from . import settings
    mods = {m: importlib.import_module(f"fourier.packs.{m}") for m in TUNABLE_MODULES}
    if settings.overrides():
        raise ConfigError("the code's defaults can't be read with $FOURIER_RESOLVED_CONFIG set")
    return {f"{m}.{n}": getattr(mods[m], n) for m, names in settings._seen.items() for n in names}


def _dump_defaults() -> None:
    import json
    import sys

    from .settings import encode
    json.dump({k: encode(v) for k, v in _defaults_here().items()}, sys.stdout)


_DEFAULTS: dict | None = None


def defaults() -> dict:
    """"<module>.<NAME>" -> the code's default for every overridable tunable.

    Read in a child process when this one hasn't imported the curation modules yet, so it
    can still set $FOURIER_RESOLVED_CONFIG before they are first imported here."""
    global _DEFAULTS
    if _DEFAULTS is None:
        import json
        import subprocess
        import sys
        if any(f"fourier.packs.{m}" in sys.modules for m in TUNABLE_MODULES):
            _DEFAULTS = _defaults_here()
        else:
            from .settings import ENV, decode
            env = {k: v for k, v in os.environ.items() if k != ENV}
            src = str(Path(__file__).resolve().parents[1])
            env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            out = subprocess.run([sys.executable, "-c", "from fourier.layers import _dump_defaults; _dump_defaults()"],
                                 env=env, capture_output=True, text=True, check=False)
            if out.returncode:
                raise ConfigError(f"couldn't read the defaults: {out.stderr.strip()[-400:]}")
            _DEFAULTS = {k: decode(v) for k, v in json.loads(out.stdout).items()}
    return _DEFAULTS


def qualify(key: str, known: dict) -> str:
    """A tunable key as "<module>.<NAME>"; a bare NAME must be unique across modules."""
    if key in known:
        return key
    hits = [k for k in known if k.split(".", 1)[1] == key]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise ConfigError(f"{key} is in {len(hits)} modules; write one of {', '.join(sorted(hits))}")
    import difflib
    names = sorted({k.split(".", 1)[-1] for k in known})
    close = difflib.get_close_matches(key.split(".", 1)[-1].upper(), names, n=3)
    from .knobs import KNOBS
    raise ConfigError(f"unknown tunable {key!r}"
                      + (f" (did you mean {', '.join(close)}?)" if close else "")
                      + f"; the everyday settings are {', '.join(KNOBS)}: `fourier config "
                        f"explain <name>` says what each does")


def _record(v):
    """An item of a list or tuple whose default is empty (no item to copy the kind of):
    nested lists are records, so they become tuples, as they are written in the code."""
    return tuple(_record(x) for x in v) if isinstance(v, list) else v


def from_plain(key: str, value, default):
    """A TOML/YAML value converted to the kind of its default, all the way down: a list
    becomes a tuple or set where the default is one, a string a regex with the default's
    flags, dict keys the default's key type."""
    if isinstance(default, re.Pattern):
        if isinstance(value, re.Pattern):
            return value
        if not isinstance(value, str):
            raise ConfigError(f"{key}: expected a regex string")
        try:
            return re.compile(value, default.flags)
        except re.error as e:
            raise ConfigError(f"{key}: bad regex ({e})") from None
    if isinstance(default, tuple) and isinstance(value, (list, tuple)):
        return tuple(from_plain(key, v, default[0]) if default else _record(v) for v in value)
    if isinstance(default, (set, frozenset)) and isinstance(value, (list, set, frozenset)):
        return type(default)(value)
    if isinstance(default, list) and isinstance(value, list):
        return [from_plain(key, v, default[0]) if default else _record(v) for v in value]
    if isinstance(default, dict) and isinstance(value, dict) and default:
        first_k, first_v = next(iter(default.items()))
        kt = type(first_k)
        out = {}
        for k, v in value.items():
            try:
                k2 = k if isinstance(k, kt) else kt(k)
            except (TypeError, ValueError):
                raise ConfigError(f"{key}: keys must be {kt.__name__}") from None
            out[k2] = from_plain(key, v, default.get(k2, first_v))
        return out
    return value


# what a text editor's "smart quotes" type in place of straight ones (macOS TextEdit does by
# default): read as straight quotes when the file doesn't parse as written
SMART_QUOTES = (str.maketrans({"\u201c": '"', "\u201d": '"', "\u201e": '"'}),
                str.maketrans({"\u2018": "'", "\u2019": "'"}))


def _parse(path: Path, text: str):
    return tomllib.loads(text) if path.suffix == ".toml" else (yaml.safe_load(text) or {})


def _read(path: Path):
    text = path.read_text()
    try:
        return _parse(path, text)
    except (tomllib.TOMLDecodeError, yaml.YAMLError) as e:
        err = e
    for i in range(len(SMART_QUOTES)):          # curly double quotes first, then single
        fixed = text
        for t in SMART_QUOTES[:i + 1]:
            fixed = fixed.translate(t)
        if fixed == text:
            continue
        try:
            return _parse(path, fixed)
        except (tomllib.TOMLDecodeError, yaml.YAMLError):
            pass
    raise ConfigError(f"{path}: {plain_error(err, text)}") from None


def plain_error(err, text: str) -> str:
    """A TOML or YAML error in words someone who's never seen either can act on: the line it
    is about, as it reads, and the usual fix."""
    msg = str(err)
    m = re.search(r"line (\d+)", msg)
    if not m:
        return msg
    n = int(m.group(1))
    lines = text.splitlines()
    if "Unclosed" in msg or "Expected ']'" in msg or "Expected '}'" in msg:
        # the opening bracket is on an earlier line than where the parser noticed
        for i in range(min(n, len(lines)) - 1, -1, -1):
            if lines[i].count("[") > lines[i].count("]") or lines[i].count("{") > lines[i].count("}"):
                n = i + 1
                break
        what = "a list [ ] or table { } that isn't closed"
    elif "Invalid value" in msg or "Invalid initial character" in msg:
        what = 'a value that needs quotes, like preset = "balanced"'
    elif "Expected '=' after a key" in msg:
        what = "a line that isn't name = value"
    else:
        what = msg.split(" (at ")[0].lower()
    shown = lines[n - 1].strip() if 0 < n <= len(lines) else ""
    return f"line {n}" + (f" ({shown})" if shown else "") + f": {what}"


def _preset_path(ref: str, base: Path | None) -> Path:
    p = Path(ref).expanduser()
    if p.suffix in (".yaml", ".yml", ".toml") and not p.is_absolute() and base is not None:
        p = base.parent / p
    if p.exists():
        return p
    named = PRESETS_DIR / f"{ref}.yaml"
    if named.exists():
        return named
    raise ConfigError(f"preset {ref!r}: not a file and not in {PRESETS_DIR}")


def _preset_chain(ref: str, base: Path | None) -> list[tuple[Path, dict]]:
    """[(path, doc)] from the root preset down to ref."""
    chain, seen = [], set()
    while ref:
        p = _preset_path(ref, base).resolve()
        if p in seen or len(chain) >= MAX_EXTENDS:
            raise ConfigError(f"preset {ref!r}: an extends loop")
        seen.add(p)
        doc = _read(p)
        chain.append((p, doc))
        ref, base = doc.get("extends"), p
    return list(reversed(chain))


def known_with_added(values: dict) -> dict:
    """values ("<module>.<NAME>" -> value) as the knobs should see them: with the added
    categories (ADDED_CATEGORIES) in the folder order, budgets and file sizes, as the
    curation modules will derive them (fourier/taxonomy.py Taxonomy.add_values)."""
    added = values.get(ADDED) or {}
    if not added:
        return values
    out = dict(values)
    k = "curate_config.CATEGORY_ORDER"
    if k in out:
        out[k] = [*out[k], *(c for c in added if c not in out[k])]
    for key, field_ in (("curate_config.BUDGETS", "budget"), ("curate_config.AVG_FILE_MB", "avg_file_mb")):
        if key in out:
            out[key] = {**out[key], **{c: e[field_] for c, e in added.items() if field_ in e and c not in out[key]}}
    return out


def resolve(config: str | None = None, sets: tuple = (), known: dict | None = None) -> Resolved:
    """Every layer applied in order. known: the defaults (tests pass their own)."""
    known = defaults() if known is None else known
    r = Resolved()
    presets_in_use: set = set()     # the names in the preset chain (an added category's `presets`)

    def apply(table: dict, label: str):
        for key, value in (table or {}).items():
            k = qualify(key, known)
            r.values[k] = from_plain(k, value, known[k])
            r.sources[k] = label

    def add_categories(table, label: str) -> dict:
        """An add_categories table into ADDED_CATEGORIES; returns what the presets in use say
        about the new categories ({CATEGORY: off | on | weight})."""
        from .taxonomy import TaxonomyError, with_added
        if ADDED not in known:
            raise ConfigError(f"{label}: add_categories: this version has no {ADDED}")
        if not isinstance(table, dict):
            raise ConfigError(f"{label}: add_categories: a table of CATEGORY = {{settings}}")
        added = {**(r.values.get(ADDED, known[ADDED]) or {}), **table}
        try:
            with_added(added)
        except TaxonomyError as e:
            raise ConfigError(f"{label}: add_categories: {e}") from None
        r.values[ADDED] = added
        r.sources[ADDED] = f"{label} (add_categories)"
        return {c: v for c, e in table.items() for p, v in (e.get("presets") or {}).items()
                if p in presets_in_use}

    def knobs_(ldoc: dict, label: str, as_: str | None = None):
        from . import knobs
        for name, k, v in knobs.apply(ldoc, known_with_added({**known, **r.values})):   # on what's resolved so far
            r.values[k] = v
            r.sources[k] = f"{label} ({as_ or name})"

    def layer(ldoc: dict, label: str):
        """One file: its add_categories, its knobs (fourier/knobs.py), then its [advanced] table."""
        if ldoc.get("add_categories"):
            said = add_categories(ldoc["add_categories"], label)
            if said:
                knobs_({"categories": said}, label, "add_categories presets")
        knobs_(ldoc, label)
        apply(ldoc.get("advanced"), label)

    path = find_config(config)
    doc = _read(path) if path else {}
    r.config = str(path) if path else None
    preset = doc.get("preset")
    if path and not preset:
        preset, r.preset_default = DEFAULT_PRESET, True
    r.preset = preset or None
    for ref, label in ((preset, "preset"), (doc.get("overlay"), "overlay")):
        if ref:
            chain = _preset_chain(ref, path)
            if label == "preset":
                presets_in_use = {p.stem for p, _ in chain}
            for p, pdoc in chain:
                r.files.append(str(p))
                layer(pdoc, f"{label}:{p.stem}")
    if path:
        r.files.append(str(path))
        layer(doc, f"config:{path.name}")
    for s in sets:
        key, sep, raw = s.partition("=")
        if not sep:
            raise ConfigError(f"--set {s!r}: expected KEY=VALUE")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw                      # a bare word: a string
        apply({key.strip(): value}, "--set")
    for k, v in list(r.values.items()):     # the final kind check, as the modules will do it
        from .settings import _coerce
        r.values[k] = _coerce(k, v, known[k])
    return r


def write(r: Resolved, run_dir: Path | None = None) -> Path:
    """resolved.json for these values, named by their hash, under $FOURIER_HOME/run."""
    from .paths import home_path
    d = run_dir or home_path("run")
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"resolved-{r.hash}.json"
    if not p.exists():
        write_resolved(p, r.values)
    return p
