"""The golden check: exit 0 when a build reproduces a reference build exactly, 1 otherwise.

    python scripts/golden_compare.py <reference manifest.json | master dir> <candidate ...>
        [--root PREFIX ...]   library root(s) to strip from source paths
                              (default: fourier.toml's library, fourier/places.py)
        [--ignore FIELD ...]  entry or summary fields to leave out (for a documented drift)
        [--json REPORT.json]  write the full report

Compares every entry field, category summary and derived set (KITS, SLICE); see
src/fourier/packs/golden.py. A missing field equals None; run metadata (timestamps, git
sha, code/config/ratings hashes) is not compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fourier.packs.golden import compare_manifests, load


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--root", action="append", default=[])
    ap.add_argument("--ignore", action="append", default=[])
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args(argv)
    try:
        ref, cand = load(a.reference), load(a.candidate)
    except (OSError, ValueError) as e:
        print(f"golden: can't read a manifest: {e}", file=sys.stderr)
        return 2
    rep = compare_manifests(ref, cand, roots=a.root, ignore=a.ignore)
    print("\n".join(rep.lines()))
    if a.json_out:
        Path(a.json_out).expanduser().write_text(json.dumps(rep.to_json(), indent=2, default=list))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
