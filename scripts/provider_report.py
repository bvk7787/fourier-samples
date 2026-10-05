"""How a build with other metadata providers compares with a reference build (normally a release).

    python scripts/provider_report.py --reference <reference manifest.json> \
        --build "Live + fallback=<manifest.json>" --build "fallback only=<manifest.json>" \
        [--out report.md]

For each build: how many of the reference's files it kept in the same category, moved to
another, or left out, and per category the reference's size, the build's, the share kept,
where the reference's files went and where the build's new ones came from. Files are
matched by their library-relative path (fourier.packs.golden.rel_src). A report, not a
gate: builds without Sononym are expected to differ.

A category picks a budget from a much larger pool, so two good builds can share few files.
With --db (a library database with Sononym's labels), each build also gets "Sononym
agrees": the share of its files in a labelled category (KICKS, PADS, ...) that Sononym
gives one of the category's labels, the reference's included for comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fourier.packs.golden import rel_src  # noqa: E402

TOP = 3


def placements(manifest: dict) -> dict:
    """{library-relative source: category} (a source in two categories keeps the first)."""
    out = {}
    for cat, v in sorted((manifest.get("categories") or {}).items()):
        for e in v.get("entries") or ():
            out.setdefault(rel_src(e.get("src")), cat)
    return out


def sononym_labels(db: str) -> dict:
    """{library-relative path: {canonical label}} from Sononym's labels in this database."""
    import sqlite3
    con = sqlite3.connect(f"file:{Path(db).expanduser()}?mode=ro", uri=True)
    out = defaultdict(set)
    for rel, lab in con.execute(
            "SELECT s.rel_path, l.label FROM labels l JOIN samples s ON s.id = l.sample_id "
            "WHERE l.provider = 'sononym' AND l.kind = 'canonical'"):
        out[rel].add(lab)
    con.close()
    return out


def category_labels() -> dict:
    """{category: {canonical label}} for the categories the taxonomy names labels for."""
    from fourier.packs.curate_config import CATEGORIES
    return {c: set(v["labels"]) for c, v in CATEGORIES.items() if v.get("labels")}


def agreement(manifest: dict, son: dict, cats: dict) -> dict:
    """{category: (files Sononym labelled, of them with one of the category's labels)}."""
    out = {}
    for c, labs in cats.items():
        n = agree = 0
        for s in (e.get("src") for e in ((manifest.get("categories") or {}).get(c) or {}).get("entries") or ()):
            got = son.get(rel_src(s))
            if got:
                n += 1
                agree += bool(got & labs)
        out[c] = (n, agree)
    return out


def compare(ref: dict, cand: dict, son=None, labelled=None) -> dict:
    a, b = placements(ref), placements(cand)
    cats = sorted(set(a.values()) | set(b.values()))
    rows = {}
    for c in cats:
        mine = {s for s, k in a.items() if k == c}
        theirs = {s for s, k in b.items() if k == c}
        went = Counter(b.get(s, "(out)") for s in mine - theirs)
        came = Counter(a.get(s, "(not in ref)") for s in theirs - mine)
        rows[c] = dict(ref=len(mine), cand=len(theirs), kept=len(mine & theirs),
                       went=went.most_common(TOP), came=came.most_common(TOP))
    if son is not None:
        ra, ca = agreement(ref, son, labelled), agreement(cand, son, labelled)
        for c in rows:
            if c in labelled:
                rows[c]["agree_ref"], rows[c]["agree"] = ra[c], ca[c]
    kept = sum(1 for s, k in a.items() if b.get(s) == k)
    moved = sum(1 for s, k in a.items() if s in b and b[s] != k)
    return dict(ref=len(a), cand=len(b), kept=kept, moved=moved, out=len(a) - kept - moved,
                new=sum(1 for s in b if s not in a), providers=cand.get("providers"), rows=rows)


def _pct(x, n):
    return f"{100 * x / n:.0f}%" if n else "-"


def _list(pairs):
    return ", ".join(f"{k} {n}" for k, n in pairs) or "-"


def markdown(name: str, r: dict) -> str:
    agree = any("agree" in x for x in r["rows"].values())
    head = "| Category | Reference | This build | Kept | Kept % |"
    rule = "|---|---:|---:|---:|---:|"
    if agree:
        head += " Sononym agrees (ref) | Sononym agrees |"
        rule += "---:|---:|"
    lines = [f"## {name}", "",
             f"Providers: {', '.join(r['providers'] or ['?'])}. Of the reference's {r['ref']:,} files, "
             f"{r['kept']:,} ({_pct(r['kept'], r['ref'])}) stay in the same category, {r['moved']:,} move "
             f"to another and {r['out']:,} are left out; {r['new']:,} of this build's {r['cand']:,} files "
             "weren't in the reference.", "",
             head + " Reference files went to | New files came from |", rule + "---|---|"]
    tot = [0, 0, 0, 0]
    for c, x in r["rows"].items():
        cells = f"| {c} | {x['ref']} | {x['cand']} | {x['kept']} | {_pct(x['kept'], x['ref'])} |"
        if agree:
            (rn, ra), (cn, ca) = x.get("agree_ref", (0, 0)), x.get("agree", (0, 0))
            tot = [tot[0] + rn, tot[1] + ra, tot[2] + cn, tot[3] + ca]
            cells += f" {_pct(ra, rn) if rn else '-'} | {_pct(ca, cn) if cn else '-'} |"
        lines.append(f"{cells} {_list(x['went'])} | {_list(x['came'])} |")
    if agree:
        lines[2] += (f" Over the labelled categories Sononym agrees with {_pct(tot[3], tot[2])} of this "
                     f"build's files it labels ({_pct(tot[1], tot[0])} of the reference's).")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True)
    ap.add_argument("--build", action="append", required=True, metavar="NAME=MANIFEST")
    ap.add_argument("--out")
    ap.add_argument("--json", help="also write the numbers here")
    ap.add_argument("--db", help="a library database with Sononym's labels: adds 'Sononym agrees'")
    a = ap.parse_args(argv)
    ref = json.loads(Path(a.reference).expanduser().read_text())
    son = sononym_labels(a.db) if a.db else None
    cats = category_labels() if a.db else None
    parts, data = [], {}
    for spec in a.build:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--build {spec!r}: expected NAME=MANIFEST")
        r = compare(ref, json.loads(Path(path).expanduser().read_text()), son, cats)
        data[name] = r
        parts.append(markdown(name, r))
    text = "\n".join(parts)
    if a.out:
        Path(a.out).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(data, indent=1, default=list))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
