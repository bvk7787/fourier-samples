"""scripts/provider_report.py: a build compared with a reference, category by category."""
import importlib.util
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "scripts" / "provider_report.py"
_spec = importlib.util.spec_from_file_location("provider_report", _p)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _m(cats, providers=None):
    return {"providers": providers, "categories": {
        c: {"entries": [{"src": f"/x/SampleLibrary/{s}"} for s in srcs]} for c, srcs in cats.items()}}


def test_kept_moved_out_and_new():
    ref = _m({"KICKS": ["a", "b", "c"], "SNARES": ["d"]})
    cand = _m({"KICKS": ["a", "e"], "SNARES": ["b"]}, ["path", "audio"])
    r = R.compare(ref, cand)
    assert (r["ref"], r["cand"], r["kept"], r["moved"], r["out"], r["new"]) == (4, 3, 1, 1, 2, 1)
    k = r["rows"]["KICKS"]
    assert (k["ref"], k["cand"], k["kept"]) == (3, 2, 1)
    assert dict(k["went"]) == {"SNARES": 1, "(out)": 1} and k["came"] == [("(not in ref)", 1)]
    assert r["rows"]["SNARES"]["came"] == [("KICKS", 1)]
    md = R.markdown("fallback", r)
    assert "Providers: path, audio" in md and "| KICKS | 3 | 2 | 1 | 33% |" in md


def test_sononym_agreement_per_category():
    ref = _m({"KICKS": ["a", "b"], "PADS": ["p"]})
    cand = _m({"KICKS": ["a", "c", "d"], "PADS": ["q"]})
    son = {"a": {"kick", "class.oneshot"}, "b": {"kick"}, "c": {"snare"}, "p": {"pad"}}
    r = R.compare(ref, cand, son, {"KICKS": {"kick"}, "PADS": {"pad"}})
    assert r["rows"]["KICKS"]["agree_ref"] == (2, 2) and r["rows"]["KICKS"]["agree"] == (2, 1)
    assert r["rows"]["PADS"]["agree"] == (0, 0)                       # "q": Sononym never saw it
    md = R.markdown("x", r)
    assert "| KICKS | 2 | 3 | 1 | 50% | 100% | 50% |" in md
    assert "Sononym agrees with 50% of this build's files it labels (100% of the reference's)" in md
