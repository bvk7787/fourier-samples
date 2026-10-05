"""Unit tests for the build validator (synthetic manifests, no DB)."""
import json
import os

from fourier.packs.validate import validate_master


def _write(tmp, cats):
    os.makedirs(str(tmp), exist_ok=True)
    json.dump({"categories": cats}, open(os.path.join(str(tmp), "manifest.json"), "w"))


def test_validate_pass(tmp_path):
    cats = {"KICKS": {"files": 80, "families": 2, "entries":
            [{"family": "a", "src": "/x/SampleLibrary/Acme/k.wav"}] * 40 +
            [{"family": "b", "src": "/x/SampleLibrary/Northwind/k.wav"}] * 40}}
    _write(tmp_path, cats)
    ok, res = validate_master(str(tmp_path), budgets={"KICKS": 80}, floor=6, ceil=60,
                              vendor_max=0.5, session=None, log=lambda m: None)
    assert ok
    assert not [r for r in res if r[0] == "FAIL"]


def test_validate_catches_budget_overshoot(tmp_path):
    cats = {"KICKS": {"files": 100, "families": 3,
                      "entries": [{"family": f"f{i%3}"} for i in range(100)]}}
    _write(tmp_path, cats)
    ok, res = validate_master(str(tmp_path), budgets={"KICKS": 80}, ceil=60,
                              session=None, log=lambda m: None)
    assert not ok
    assert any(c == "budget" and lvl == "FAIL" for lvl, c, d in res)


def test_validate_catches_ceiling(tmp_path):
    cats = {"KICKS": {"files": 70, "families": 1, "entries": [{"family": "a"}] * 70}}
    _write(tmp_path, cats)
    ok, res = validate_master(str(tmp_path), budgets={"KICKS": 70}, ceil=60,
                              session=None, log=lambda m: None)
    assert not ok
    assert any(c == "ceiling" and lvl == "FAIL" for lvl, c, d in res)


def test_validate_missing_manifest(tmp_path):
    ok, res = validate_master(str(tmp_path), session=None, log=lambda m: None)
    assert not ok


def test_validate_instrument_exempt_from_vendor(tmp_path):
    # ACOUSTIC is pack-scoped (instrument kind) -> 100% one vendor is expected, no warn
    cats = {"ACOUSTIC": {"files": 40, "families": 4, "entries":
            [{"family": f"f{i % 4}", "src": f"/x/SampleLibrary/Ableton/P{i % 5}/o{i}.wav"}
             for i in range(40)]}}
    _write(tmp_path, cats)
    ok, res = validate_master(str(tmp_path), budgets={"ACOUSTIC": 40}, ceil=60,
                              session=None, log=lambda m: None)
    assert ok
    assert not any(c == "vendor" for lvl, c, d in res)
    assert not any(c == "floor" for lvl, c, d in res)


def test_validate_instrument_dup_exempt(tmp_path):
    # PIANO (instrument, clap-sourced) sharing a source with SYNTH is by design, no dup warn
    src = "/x/SampleLibrary/Acme/s.wav"
    cats = {
        "SYNTH": {"files": 6, "families": 1, "entries": [{"family": "a", "src": src}] * 6},
        "PIANO": {"files": 6, "families": 1, "entries": [{"family": "a", "src": src}] * 6},
    }
    _write(tmp_path, cats)
    ok, res = validate_master(str(tmp_path), budgets={"SYNTH": 6, "PIANO": 6}, ceil=60,
                              session=None, log=lambda m: None)
    assert not any(c == "dup" for lvl, c, d in res)


def test_validate_real_dup_still_warns(tmp_path):
    # two home-managed (non-instrument) categories sharing a source IS a single-home
    # violation and must warn
    src = "/x/SampleLibrary/Acme/s.wav"
    cats = {
        "KICKS": {"files": 6, "families": 1, "entries": [{"family": "a", "src": src}] * 6},
        "SNARES": {"files": 6, "families": 1, "entries": [{"family": "a", "src": src}] * 6},
    }
    _write(tmp_path, cats)
    ok, res = validate_master(str(tmp_path), budgets={"KICKS": 6, "SNARES": 6}, ceil=60,
                              session=None, log=lambda m: None)
    assert any(c == "dup" for lvl, c, d in res)


def test_validate_override_quota(tmp_path):
    # 20 clavinets in a 100-file PIANO (20%) exceed the 10%+5% override share -> WARN
    ents = [{"family": f"f{i % 4}", "src": f"/x/SampleLibrary/Acme/P{i % 10}/Clavinet {i}.wav"}
            for i in range(20)] + \
           [{"family": f"f{i % 4}", "src": f"/x/SampleLibrary/Northwind/P{i % 10}/Piano {i}.wav"}
            for i in range(80)]
    _write(tmp_path, {"PIANO": {"files": 100, "families": 4, "entries": ents}})
    ok, res = validate_master(str(tmp_path), budgets={"PIANO": 100}, ceil=60,
                              session=None, log=lambda m: None)
    assert ok                                                   # WARN, not FAIL
    assert any(c == "override" and lvl == "WARN" for lvl, c, d in res)
    assert not any(c == "pack" for lvl, c, d in res)


def test_validate_instrument_pack_ceiling(tmp_path):
    # one pack supplying half of a 40-file ACOUSTIC exceeds 20%+5% -> WARN
    ents = [{"family": f"f{i % 4}", "src": "/x/SampleLibrary/Ableton/Big Pack/o%d.wav" % i}
            for i in range(20)] + \
           [{"family": f"f{i % 4}", "src": "/x/SampleLibrary/Ableton/P%d/o%d.wav" % (i % 5, i)}
            for i in range(20)]
    _write(tmp_path, {"ACOUSTIC": {"files": 40, "families": 4, "entries": ents}})
    ok, res = validate_master(str(tmp_path), budgets={"ACOUSTIC": 40}, ceil=60,
                              session=None, log=lambda m: None)
    assert ok
    assert any(c == "pack" and lvl == "WARN" for lvl, c, d in res)


def test_family_names_differing_only_by_case_fail(tmp_path):
    import json
    from fourier.packs.validate import validate_master
    ents = [dict(family="Dark", out="Dark/a.wav", src="/l/a.wav"),
            dict(family="dark", out="dark/b.wav", src="/l/b.wav")]
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"categories": {"KICKS": {"entries": ents, "files": 2, "families": 2}}}))
    ok, res = validate_master(str(tmp_path), session=None, log=lambda m: None)
    assert not ok and any(chk == "names" for lvl, chk, d in res if lvl == "FAIL")


def test_validate_vendor_share_only_with_three_vendors_or_more(tmp_path):
    """The build caps a category's vendors only when its pool has three or more; verify's
    vendor check follows it (one or two vendors: no warning; three with one over: a warning)."""
    def cats(vendors):
        return {"KICKS": {"files": len(vendors), "families": 1, "entries":
                [{"family": "a", "src": f"/x/SampleLibrary/{v}/k{i}.wav"} for i, v in enumerate(vendors)]}}
    for vendors, warned in (("A" * 12, False), ("A" * 9 + "B" * 3, False), ("A" * 10 + "BC", True)):
        _write(tmp_path, cats(vendors))
        budgets = {"KICKS": len(vendors)}
        ok, res = validate_master(str(tmp_path), budgets=budgets, ceil=60, session=None, log=lambda m: None)
        assert any(c == "vendor" for lvl, c, d in res) == warned, (vendors, res)
