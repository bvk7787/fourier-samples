"""Every tunable in the curation code, with its value, class and DSP flag, as a CSV.

    python scripts/tunables_inventory.py [--out tunables.csv]

A tunable is a module-level UPPER_CASE name in the modules below. Its class (and the Tier 1
knob that sets it, if any) comes from config/tunables.yaml. "dsp" marks the ones named in
the export DSP the processed-audio cache fingerprints (packs/audiocache.py): changing one
of those changes the audio of files already built. Their values reach the cache key as
hashed constants, default arguments or cache-key arguments.

The input the typed config was built from (docs/design-history.md, "How it became configurable").
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULES = ("curate_config", "curate", "naming", "sets")
CLASSES = ("correctness", "knob", "taste", "rules", "engine", "taxonomy", "vendor", "library",
           "device", "internal")
KNOBS = ("loudness", "retune", "stereo", "tempo", "fold", "size", "scale", "sources", "names", "categories",
         "vendors", "words", "sets")
CLASSES_FILE = ROOT / "config" / "tunables.yaml"
NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
FIELDS = ("module", "name", "line", "class", "knob", "dsp", "type", "value")


def tunables(module: str) -> list[tuple[str, int]]:
    """(name, line) of each module-level UPPER_CASE assignment, first definition only."""
    src = (ROOT / "src" / "fourier" / "packs" / f"{module}.py").read_text()
    out, seen = [], set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        for t in targets:
            if NAME_RE.fullmatch(t) and t not in seen:
                seen.add(t)
                out.append((t, node.lineno))
    return out


def load_classes(path: Path = CLASSES_FILE) -> dict[tuple[str, str], tuple[str, str]]:
    import yaml
    doc = yaml.safe_load(path.read_text()) or {}
    out = {}
    for module, entries in doc.items():
        for name, spec in (entries or {}).items():
            cls, _, knob = str(spec).partition(":")
            out[(module, name)] = (cls, knob)
    return out


def dsp_names() -> set[str]:
    """Tunables named in the DSP code the audio cache fingerprints."""
    from fourier.packs import audiocache
    from fourier.packs import curate as C
    parts, seen = [], set()
    for f in (C._process_audio, C._clamp_rms, C._rms_db):
        audiocache._reach(f, C, seen, parts)
    hashed = {p.split("=", 1)[0] for p in parts if re.match(r"^[A-Z][A-Z0-9_]*=", p)}
    code = "\n".join(p for p in parts if not re.match(r"^[A-Z][A-Z0-9_]*=", p))
    return hashed | set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", code))


def _value(v) -> str:
    v = getattr(v, "pattern", v)          # a compiled regex shows its pattern
    s = repr(v).replace("\n", " ")
    return s if len(s) <= 160 else s[:157] + "..."


def rows() -> list[dict]:
    classes = load_classes()
    dsp = dsp_names()
    out = []
    for module in MODULES:
        mod = importlib.import_module(f"fourier.packs.{module}")
        for name, line in tunables(module):
            cls, knob = classes.get((module, name), ("", ""))
            v = getattr(mod, name, None)
            out.append(dict(module=module, name=name, line=line, **{"class": cls}, knob=knob,
                            dsp="yes" if name in dsp else "", type=type(v).__name__,
                            value=_value(v)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rs = rows()
    with (open(a.out, "w", newline="") if a.out else contextlib.nullcontext(sys.stdout)) as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rs)
    missing = [f"{r['module']}.{r['name']}" for r in rs if not r["class"]]
    if missing:
        print(f"unclassed (add to {CLASSES_FILE.relative_to(ROOT)}): {', '.join(missing)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
