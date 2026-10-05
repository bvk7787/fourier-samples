"""The golden check: does a build reproduce a reference build exactly?

Compares two master manifests (manifest.json) entry by entry. Unlike
scripts/manifest_diff.py, which prints a stability summary, this is a gate: any
difference in any entry field, category summary or derived set fails it.

- Categories are keyed by the source path relative to the library, so the same
  library mounted somewhere else (a frozen copy, another machine) still compares.
- Sets (KITS, SLICE) are keyed by their output path.
- A field missing on one side equals None on the other (older manifests leave
  optional fields out).
- Build metadata that changes on every run (timestamps, git sha, code and config
  hashes) is not compared. The ratings hash is checked separately by the harness.
- Entry fields a later manifest format added (ADDED_FIELDS) aren't compared against a
  reference in an older format; the report lists them as skipped.

Used by scripts/golden_compare.py and scripts/golden_harness.py.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# top-level keys that describe the run, not its output
META_KEYS = frozenset({"generated", "git_sha", "code_hash", "config_hash", "ratings_hash",
                       "seed", "clap_model", "fourier_manifest", "tunables_hash", "overrides",
                       "providers", "fourier_version", "library"})
# per-category keys that describe the run
CATEGORY_META_KEYS = frozenset({"built", "entries"})
# manifest format ("fourier_manifest") -> the entry fields it added
ADDED_FIELDS = {2: frozenset({"labels"})}
EXAMPLES = 10


def rel_src(src: str | None, roots=()) -> str | None:
    """A source path relative to its library: the part after the first matching root,
    else its library path (fourier/places.py: the configured library), else unchanged."""
    if not src:
        return src
    for r in roots:
        r = r.rstrip("/") + "/"
        if src.startswith(r):
            return src[len(r):]
    from ..places import library_rel
    return library_rel(src)


@dataclass
class GoldenReport:
    gone: list = field(default_factory=list)          # (category, src) only in the reference
    new: list = field(default_factory=list)           # (category, src) only in the candidate
    fields: Counter = field(default_factory=Counter)  # field name -> entries that differ
    examples: dict = field(default_factory=dict)      # field name -> [(where, ref, cand)]
    summary: list = field(default_factory=list)       # (where, key, ref, cand)
    duplicates: list = field(default_factory=list)    # (group, key) seen twice on one side
    skipped: list = field(default_factory=list)       # entry fields newer than the reference's format

    @property
    def ok(self) -> bool:
        return not (self.gone or self.new or self.fields or self.summary or self.duplicates)

    def add(self, name, where, ref, cand):
        self.fields[name] += 1
        ex = self.examples.setdefault(name, [])
        if len(ex) < EXAMPLES:
            ex.append((where, ref, cand))

    def lines(self) -> list[str]:
        note = (f" (not compared, newer than the reference's format: {', '.join(self.skipped)})"
                if self.skipped else "")
        if self.ok:
            return ["golden: identical" + note]
        out = [f"golden: DIFFERENT ({self.total} differences)"]
        if self.duplicates:
            out.append(f"  duplicate keys: {self.duplicates[:EXAMPLES]}")
        if self.gone:
            out.append(f"  gone: {len(self.gone)}  e.g. {self.gone[:EXAMPLES]}")
        if self.new:
            out.append(f"  new: {len(self.new)}  e.g. {self.new[:EXAMPLES]}")
        for name, n in self.fields.most_common():
            out.append(f"  {name}: {n} differ")
            for where, a, b in self.examples[name]:
                out.append(f"    {where}: {a!r} -> {b!r}")
        for where, key, a, b in self.summary[:EXAMPLES * 2]:
            out.append(f"  {where}.{key}: {a!r} -> {b!r}")
        return out

    @property
    def total(self) -> int:
        return (len(self.gone) + len(self.new) + sum(self.fields.values())
                + len(self.summary) + len(self.duplicates))

    def to_json(self) -> dict:
        return dict(ok=self.ok, total=self.total, gone=self.gone, new=self.new,
                    fields=dict(self.fields), examples=self.examples,
                    summary=self.summary, duplicates=self.duplicates, skipped=self.skipped)


def _index(entries, key, report, group):
    out = {}
    for e in entries:
        k = key(e)
        if k in out:
            report.duplicates.append((group, k))
        out[k] = e
    return out


def _compare_entries(where, a, b, report, roots, ignore, skip=()):
    for name in sorted((set(a) | set(b)) - set(ignore) - set(skip)):
        va, vb = a.get(name), b.get(name)
        if name in ("src",):
            va, vb = rel_src(va, roots), rel_src(vb, roots)
        if va != vb:
            report.add(name, where, va, vb)


def compare_manifests(ref: dict, cand: dict, roots=(), ignore=()) -> GoldenReport:
    """Every difference between a reference manifest and a candidate."""
    r = GoldenReport()
    ref_format, cand_format = ref.get("fourier_manifest") or 1, cand.get("fourier_manifest") or 1
    newer = sorted(f for v, fs in ADDED_FIELDS.items() if ref_format < v <= cand_format for f in fs)
    r.skipped = newer
    ignore = frozenset(ignore) | frozenset(newer)

    for k in sorted((set(ref) | set(cand)) - META_KEYS - {"categories", "sets"} - ignore):
        if ref.get(k) != cand.get(k):
            r.summary.append(("manifest", k, ref.get(k), cand.get(k)))

    # categories: every source once, keyed by its library-relative path
    ents = {}
    for side, doc in (("ref", ref), ("cand", cand)):
        rows = [(cat, e) for cat, v in (doc.get("categories") or {}).items()
                for e in v.get("entries", [])]
        ents[side] = _index(rows, lambda ce: rel_src(ce[1].get("src"), roots), r, f"categories/{side}")
        for cat, v in sorted((doc.get("categories") or {}).items()):
            ents.setdefault(f"{side}_summary", {})[cat] = {
                k: x for k, x in v.items() if k not in CATEGORY_META_KEYS}
    for cat in sorted(set(ents["ref_summary"]) | set(ents["cand_summary"])):
        sa, sb = ents["ref_summary"].get(cat) or {}, ents["cand_summary"].get(cat) or {}
        for k in sorted((set(sa) | set(sb)) - ignore):
            if sa.get(k) != sb.get(k):
                r.summary.append((f"categories.{cat}", k, sa.get(k), sb.get(k)))
    A, B = ents["ref"], ents["cand"]
    r.gone += sorted((A[k][0], k) for k in A.keys() - B.keys())
    r.new += sorted((B[k][0], k) for k in B.keys() - A.keys())
    for k in sorted(A.keys() & B.keys()):
        (ca, ea), (cb, eb) = A[k], B[k]
        if ca != cb:
            r.add("category", k, ca, cb)
        _compare_entries(f"{cb}:{k}", ea, eb, r, roots, ignore)

    # derived sets: keyed by output path inside the set
    for s in sorted(set(ref.get("sets") or {}) | set(cand.get("sets") or {})):
        sa = _index((ref.get("sets") or {}).get(s, {}).get("entries", []), lambda e: e.get("out"),
                    r, f"sets.{s}/ref")
        sb = _index((cand.get("sets") or {}).get(s, {}).get("entries", []), lambda e: e.get("out"),
                    r, f"sets.{s}/cand")
        r.gone += sorted((s, k) for k in sa.keys() - sb.keys())
        r.new += sorted((s, k) for k in sb.keys() - sa.keys())
        for k in sorted(sa.keys() & sb.keys()):
            _compare_entries(f"{s}:{k}", sa[k], sb[k], r, roots, ignore)
    return r


def load(path) -> dict:
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "manifest.json"
    return json.loads(p.read_text())
