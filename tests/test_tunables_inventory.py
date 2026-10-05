"""The tunables inventory (scripts/tunables_inventory.py, config/tunables.yaml) stays complete."""
import csv
import importlib.util
import io
from pathlib import Path


def _inv():
    p = Path(__file__).resolve().parents[1] / "scripts" / "tunables_inventory.py"
    spec = importlib.util.spec_from_file_location("tunables_inventory", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_tunable_has_exactly_one_class_line():
    inv = _inv()
    found = {(m, n) for m in inv.MODULES for n, _ in inv.tunables(m)}
    classes = inv.load_classes()
    assert not found - classes.keys(), f"add to config/tunables.yaml: {sorted(found - classes.keys())}"
    assert not classes.keys() - found, f"gone from the code: {sorted(classes.keys() - found)}"


def test_classes_and_knobs_are_known():
    inv = _inv()
    for (m, n), (cls, knob) in inv.load_classes().items():
        assert cls in inv.CLASSES, f"{m}.{n}: unknown class {cls!r}"
        assert not knob or knob in inv.KNOBS, f"{m}.{n}: unknown knob {knob!r}"
        assert (cls == "knob") <= bool(knob), f"{m}.{n}: a knob class names its knob"


def test_dsp_tunables_are_correctness_or_knobs():
    """Anything that changes the audio of built files is either a fixed safety value, a
    Tier 1 knob, or passed per category (and in the cache key)."""
    inv = _inv()
    classes = inv.load_classes()
    dsp = inv.dsp_names()
    bad = [(m, n, c) for (m, n), (c, _) in classes.items()
           if n in dsp and c not in ("correctness", "knob", "taste")]
    assert not bad, bad
    assert {"EDGE_FADE_MS", "LOOP_LIMIT_DB", "PEAK_CEILING_DB"} <= dsp


def test_csv_has_a_row_per_tunable(capsys):
    inv = _inv()
    assert inv.main([]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert len(rows) == sum(len(inv.tunables(m)) for m in inv.MODULES)
    assert all(r["class"] for r in rows)
    edge = next(r for r in rows if r["name"] == "EDGE_FADE_MS")
    assert edge["class"] == "correctness" and edge["dsp"] == "yes" and edge["type"] == "float"
    limit = next(r for r in rows if r["name"] == "LOOP_LIMIT_DB")
    assert (limit["class"], limit["knob"]) == ("knob", "loudness")
