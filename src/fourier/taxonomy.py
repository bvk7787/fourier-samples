"""The taxonomy (config/taxonomy.yaml): which folders a build makes and what goes in each.

The curation modules read their taxonomy tunables from here (CATEGORY_ORDER, the categories'
kinds and labels, the CLAP prompts, KIT_CATS, MELODIC_CATS, ...), so a tunable keeps its
name and an [advanced] override by that name still wins; this file only supplies the
default (docs/design-history.md, "How it became configurable").

A library overlay can add whole categories of its own (`add_categories`, fourier/layers.py;
curate_config.ADDED_CATEGORIES): with_added() merges them into the file's taxonomy, and the
curation modules read that merged view (curate_config.taxonomy_view).
"""
from __future__ import annotations

import re
from functools import lru_cache

import yaml

from .paths import config_dir  # noqa: E402

TAXONOMY_PATH = config_dir() / "taxonomy.yaml"
KINDS = ("oneshot", "loop", "gated", "instrument", "waves")
ROLES = ("kit", "melodic", "tonal", "note_lead", "no_sets")
HOMES = ("phrases", "waves", "scratches", "key_chords")


class TaxonomyError(ValueError):
    pass


@lru_cache(maxsize=None)
def load() -> dict:
    """The taxonomy, checked: every category in order is defined and vice versa, kinds and
    roles are known, labels are canonical, homes name categories."""
    doc = yaml.safe_load(TAXONOMY_PATH.read_text()) or {}
    _check(doc, TAXONOMY_PATH.name)
    return doc


def _check(doc: dict, where: str) -> None:
    from .metadata.vocab import CANONICAL
    cats = doc.get("categories") or {}
    if set(doc.get("order") or ()) != set(cats) or len(doc["order"]) != len(set(doc["order"])):
        raise TaxonomyError(f"{where}: order must list every category exactly once")
    for name, c in cats.items():
        if c.get("kind") not in KINDS:
            raise TaxonomyError(f"{where}: {name}: kind {c.get('kind')!r} (one of {', '.join(KINDS)})")
        bad = sorted(set(c.get("labels") or ()) - CANONICAL)
        if bad:
            raise TaxonomyError(f"{where}: {name}: labels not in the canonical vocabulary: {bad}")
        bad = sorted(set(c.get("roles") or ()) - set(ROLES))
        if bad:
            raise TaxonomyError(f"{where}: {name}: unknown roles {bad} (known: {', '.join(ROLES)})")
    homes = doc.get("homes") or {}
    if set(homes) != set(HOMES):
        raise TaxonomyError(f"{where}: homes must name {', '.join(HOMES)}")
    for key, cat in homes.items():
        if cat is not None and cat not in cats:
            raise TaxonomyError(f"{where}: homes.{key}: {cat!r} is not a category")
    for old, new in (doc.get("renamed") or {}).items():
        if new not in cats:
            raise TaxonomyError(f"{where}: renamed.{old}: {new!r} is not a category")
    for name, c in cats.items():
        for p in c.get("prompts") or ():
            if not isinstance(p, str) and not (isinstance(p, dict) and list(p) == ["library"]):
                raise TaxonomyError(f"{where}: {name}: a prompt is text or {{library: NAME}}, not {p!r}")


class Taxonomy:
    """The taxonomy a build uses: config/taxonomy.yaml, plus the categories a library overlay
    adds (with_added). The module-level functions below read config/taxonomy.yaml alone."""

    def __init__(self, doc: dict, added: dict | None = None):
        self.doc = doc
        self.added = dict(added or {})

    def order(self) -> list:
        return list(self.doc["order"])

    def sets(self) -> dict:
        return dict(self.doc["sets"])

    def home(self, key: str):
        return self.doc["homes"][key]

    def piano_pool_labels(self) -> list:
        return list(self.doc["piano_pool_labels"])

    def renamed(self) -> dict:
        return dict(self.doc.get("renamed") or {})

    def bands(self) -> dict:
        """{band group: (band, ...)} in folder order."""
        return {k: tuple(v) for k, v in self.doc["bands"].items()}

    def phrase_roles(self) -> tuple:
        return tuple(self.doc["phrase_roles"])

    def categories(self) -> list:
        """Category names in the order they're defined (the curation modules' order)."""
        return list(self.doc["categories"])

    def kind(self, name: str) -> str:
        return self.doc["categories"][name]["kind"]

    def labels(self, name: str) -> tuple | None:
        got = self.doc["categories"][name].get("labels")
        return tuple(got) if got else None

    def prompts(self, name: str, library: dict | None = None) -> list:
        """A category's CLAP prompts. A {library: NAME} entry is where a library overlay's own
        prompts go (library[NAME], a library-class tunable), in that position; none by default."""
        out = []
        for p in self.doc["categories"][name].get("prompts") or ():
            if isinstance(p, dict):
                out.extend((library or {}).get(p["library"]) or ())
            else:
                out.append(p)
        return out

    def anti(self, name: str) -> list:
        return list(self.doc["categories"][name].get("anti") or ())

    def nouns(self) -> dict:
        return {n: c["noun"] for n, c in self.doc["categories"].items()}

    def with_role(self, role: str) -> set:
        if role not in ROLES:
            raise KeyError(role)
        return {n for n, c in self.doc["categories"].items() if role in (c.get("roles") or ())}

    def with_taxonomy(self, engine: dict) -> dict:
        """The categories as the curation code uses them (curate_config.CATEGORIES): each
        category's kind and labels from the taxonomy, then its engine settings from the code
        (an added category's from its entry: its prompts, anti and `engine`). Every taxonomy
        category needs engine settings and vice versa. An added category's `carve_from`
        ({CATEGORY: min}) keeps out of CATEGORY the files that sound more like it than like
        its anti prompts (by at least min): that category's exclude_* settings."""
        engine = {**engine, **{n: _added_engine(self, n, e) for n, e in self.added.items()}}
        names = self.categories()
        if set(engine) != set(names):
            raise TaxonomyError(f"{TAXONOMY_PATH.name}: categories {sorted(set(names) ^ set(engine))} "
                                "are in the taxonomy or the code, not both")
        out = {n: {"kind": self.kind(n), "labels": self.labels(n), **engine[n]} for n in names}
        for n, e in self.added.items():
            for target, least in (e.get("carve_from") or {}).items():
                if out[target].get("exclude_phrases"):
                    raise TaxonomyError(f"added category {n}: {target} is already carved from")
                out[target].update(exclude_phrases=out[n]["phrases"], exclude_anti=out[n]["anti"],
                                   exclude_min=float(least))
        return out

    def add_values(self, mapping: dict, field: str, convert=None) -> dict:
        """mapping ({CATEGORY: value}, a per-category tunable) plus each added category's
        `field` value, after the code's entries, for the categories mapping doesn't name yet.
        The same mapping when there is nothing to add."""
        extra = {n: (convert(e[field]) if convert else e[field]) for n, e in self.added.items()
                 if field in e and n not in mapping}
        return {**mapping, **extra} if extra else mapping

    def add_words(self, words: dict, field: str) -> dict:
        """words ({word: regex}) plus the added categories' `field` words it lacks, after them."""
        extra = {w: rx for e in self.added.values() for w, rx in (e.get(field) or {}).items() if w not in words}
        return {**words, **extra} if extra else words


# An added category's entry (a library overlay's add_categories; docs/curation.md, "Adding a
# category"): its taxonomy entry, where it goes, its engine settings and its share of the
# per-category tunables.
ADDED_TAXONOMY_KEYS = ("kind", "labels", "noun", "roles", "prompts", "anti")
ADDED_KEYS = ADDED_TAXONOMY_KEYS + ("after", "engine", "budget", "avg_file_mb", "rms_ceiling_db",
                                    "folder_words", "name_words", "carve_from", "family", "presets")
FAMILIES = ("drum", "tonal", "vocal", "fx")
# engine settings whose code values are lists (the rest are tuples, as the code writes them)
_ENGINE_LISTS = ("phrases", "anti", "dims", "clap_anti", "pool_labels", "ableton_exclude", "ableton_any")


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


def _added_engine(tax: Taxonomy, name: str, entry: dict) -> dict:
    """An added category's engine settings (curate_config.CATEGORIES): its prompts and anti
    prompts, then its `engine` table, in the kinds the code writes them in (dims as
    [(dimension, (low word, high word))], other lists as tuples)."""
    out = {"phrases": tax.prompts(name)}
    if tax.anti(name):
        out["anti"] = tax.anti(name)
    for k, v in (entry.get("engine") or {}).items():
        if isinstance(v, (list, tuple)):
            v = [_tuples(x) for x in v] if k in _ENGINE_LISTS else _tuples(v)
        out[k] = v
    return out


def _check_added(doc: dict, added: dict) -> None:
    code = set(doc["categories"])
    names = list(code)
    for name, e in added.items():
        where = f"added category {name}"
        if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise TaxonomyError(f"{where}: a name in capitals (KICKS, FIELD_FX)")
        if name in code:
            raise TaxonomyError(f"{where}: already a category")
        if not isinstance(e, dict):
            raise TaxonomyError(f"{where}: a table of its settings")
        bad = sorted(set(e) - set(ADDED_KEYS))
        if bad:
            raise TaxonomyError(f"{where}: unknown keys {bad} (known: {', '.join(ADDED_KEYS)})")
        for key in ("kind", "noun"):
            if not e.get(key):
                raise TaxonomyError(f"{where}: needs a {key}")
        if e.get("after") is not None and e["after"] not in names:
            raise TaxonomyError(f"{where}: after {e['after']!r}, not a category before it")
        for target in e.get("carve_from") or {}:
            if target not in code:
                raise TaxonomyError(f"{where}: carve_from {target!r} is not a category")
            if not e.get("anti"):
                raise TaxonomyError(f"{where}: carve_from needs anti prompts")
        if e.get("family") is not None and e["family"] not in FAMILIES:
            raise TaxonomyError(f"{where}: family {e['family']!r} (one of {', '.join(FAMILIES)})")
        if not isinstance(e.get("engine") or {}, dict):
            raise TaxonomyError(f"{where}: engine is a table of engine settings")
        num = (int, float)
        for key, ok in (("budget", lambda v: isinstance(v, int) and v >= 0),
                        ("avg_file_mb", lambda v: isinstance(v, num) and v > 0),
                        ("rms_ceiling_db", lambda v: isinstance(v, num)),
                        ("prompts", lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)),
                        ("anti", lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)),
                        ("folder_words", lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)),
                        ("name_words", lambda v: isinstance(v, dict) and all(isinstance(x, str) for x in v.values())),
                        ("carve_from", lambda v: isinstance(v, dict) and all(isinstance(x, num) for x in v.values())),
                        ("presets", lambda v: isinstance(v, dict))):
            if key in e and (isinstance(e[key], bool) or not ok(e[key])):
                raise TaxonomyError(f"{where}: {key} = {e[key]!r} (see docs/curation.md, Adding a category)")
        names.append(name)


def with_added(added: dict | None) -> Taxonomy:
    """The taxonomy with a library overlay's added categories (curate_config.ADDED_CATEGORIES,
    {NAME: entry}) merged in and checked like the file. Each is defined after its `after`
    category (else last), so the build reaches it there, and numbered after every category in
    config/taxonomy.yaml, in the order they're added: adding or removing one never renumbers
    another category's device folder."""
    doc = load()
    if not added:
        return Taxonomy(doc)
    _check_added(doc, added)
    cats = dict(doc["categories"])
    names = list(cats)
    placed: dict = {}
    for name, e in added.items():
        entry = {k: e[k] for k in ADDED_TAXONOMY_KEYS if k in e}
        cats[name] = entry
        after = e.get("after")
        if after is None:
            names.append(name)
            continue
        i = names.index(after) + 1
        while i < len(names) and placed.get(names[i]) == after:     # the overlay's order among them
            i += 1
        names.insert(i, name)
        placed[name] = after
    merged = {**doc, "categories": {n: cats[n] for n in names},
              "order": [*doc["order"], *(n for n in added if n not in doc["order"])]}
    _check(merged, f"{TAXONOMY_PATH.name} + added categories")
    return Taxonomy(merged, added)


def _public() -> Taxonomy:
    return Taxonomy(load())


def order() -> list:
    return _public().order()


def sets() -> dict:
    return _public().sets()


def home(key: str):
    return _public().home(key)


def piano_pool_labels() -> list:
    return _public().piano_pool_labels()


def renamed() -> dict:
    return _public().renamed()


def bands() -> dict:
    """{band group: (band, ...)} in folder order."""
    return _public().bands()


def phrase_roles() -> tuple:
    return _public().phrase_roles()


def categories() -> list:
    """Category names in the order they're defined (the curation modules' order)."""
    return _public().categories()


def kind(name: str) -> str:
    return _public().kind(name)


def labels(name: str) -> tuple | None:
    return _public().labels(name)


def prompts(name: str, library: dict | None = None) -> list:
    """A category's CLAP prompts (Taxonomy.prompts)."""
    return _public().prompts(name, library)


def anti(name: str) -> list:
    return _public().anti(name)


def nouns() -> dict:
    return _public().nouns()


def with_role(role: str) -> set:
    return _public().with_role(role)


def with_taxonomy(engine: dict) -> dict:
    """The categories as the curation code uses them (Taxonomy.with_taxonomy)."""
    return _public().with_taxonomy(engine)
