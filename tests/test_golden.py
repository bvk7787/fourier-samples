"""The golden check (fourier.packs.golden, scripts/golden_compare.py)."""
import copy
import importlib.util
import json
from pathlib import Path

from fourier.packs.golden import compare_manifests, rel_src

LIB = "/data/SampleLibrary/"


def _manifest(lib=LIB):
    return {
        "fourier_manifest": 1, "generated": "2000-01-27T10:00:00Z", "git_sha": "abc",
        "code_hash": "c1", "config_hash": "k1", "ratings_hash": "r1", "seed": 0,
        "categories": {
            "KICKS": {"built": "2000-01-27T10:00:00Z", "families": 1, "files": 2,
                      "source_samples": 9, "entries": [
                          {"family": "kick-909", "out": "kick-909/BD_1.wav", "src": lib + "Pack A/BD_1.wav",
                           "out_md5": "m1", "support": 2, "band": None},
                          {"family": "kick-909", "out": "kick-909/BD_2.wav", "src": lib + "Pack A/BD_2.wav",
                           "out_md5": "m2", "support": 1, "band": None},
                      ]},
            "DRUMLOOPS": {"built": "2000-01-27T10:00:00Z", "families": 1, "files": 1,
                          "source_samples": 4, "entries": [
                              {"family": "jungle-breakbeat-170", "out": "jungle-breakbeat-170/amen.wav",
                               "src": lib + "Breaks/amen.wav", "out_md5": "m3", "bpm": 170.0,
                               "bars": 4, "rotate_ms": 12.5},
                          ]},
        },
        "sets": {
            "SLICE": {"entries": [{"family": "jungle-breakbeat-170", "out": "jungle-breakbeat-170/amen.wav",
                                   "src": lib + "Breaks/amen.wav", "out_md5": "m3",
                                   "came_from": "DRUMLOOPS/jungle-breakbeat-170/amen.wav"}]},
            "KITS": {"entries": [{"family": "kit-909", "out": "kit-909/BD_1.wav", "src": lib + "Pack A/BD_1.wav",
                                  "out_md5": "m1", "came_from": "KICKS/kick-909/BD_1.wav"}]},
        },
    }


def _entry(m, cat, i):
    return m["categories"][cat]["entries"][i]


def test_identical_manifests_pass():
    m = _manifest()
    rep = compare_manifests(m, copy.deepcopy(m))
    assert rep.ok and rep.total == 0 and rep.lines() == ["golden: identical"]


def test_run_metadata_is_not_compared():
    a, b = _manifest(), _manifest()
    b.update(generated="2000-01-28T09:00:00Z", git_sha="def", code_hash="c2", config_hash="k2",
             ratings_hash="r2", tunables_hash="t2", overrides={})
    b["categories"]["KICKS"]["built"] = "2000-01-28T09:00:00Z"
    assert compare_manifests(a, b).ok


def test_same_library_elsewhere_compares_equal():
    """A frozen copy of the library at another path is the same library."""
    a, b = _manifest(), _manifest(lib="/Volumes/Frozen/SampleLibrary/")
    assert compare_manifests(a, b).ok
    c = _manifest(lib="/tmp/golden/lib/")
    assert not compare_manifests(a, c).ok                       # no marker: paths differ
    assert compare_manifests(a, c, roots=["/tmp/golden/lib"]).ok  # ...unless the root is given


def test_changed_audio_fails_and_names_the_file():
    a, b = _manifest(), _manifest()
    _entry(b, "KICKS", 1)["out_md5"] = "m2x"
    rep = compare_manifests(a, b)
    assert not rep.ok and rep.fields["out_md5"] == 1
    assert "Pack A/BD_2.wav" in "\n".join(rep.lines())


def test_moved_renamed_recategorised_gone_new():
    a, b = _manifest(), _manifest()
    _entry(b, "KICKS", 0)["out"] = "kick-909/BD_one.wav"            # renamed
    moved = b["categories"]["KICKS"]["entries"].pop(1)               # BD_2 moves to PERC
    b["categories"]["PERC"] = {"families": 1, "files": 1, "source_samples": 1, "entries": [moved]}
    b["categories"]["DRUMLOOPS"]["entries"] = []                     # amen gone
    b["categories"]["DRUMLOOPS"]["entries"].append(dict(_entry(a, "DRUMLOOPS", 0), src=LIB + "Breaks/Loop 12.wav"))
    rep = compare_manifests(a, b)
    assert rep.fields["out"] == 1 and rep.fields["category"] == 1
    assert rep.gone == [("DRUMLOOPS", "Breaks/amen.wav")]
    assert rep.new == [("DRUMLOOPS", "Breaks/Loop 12.wav")]
    assert any(w == "categories.PERC" for w, *_ in rep.summary)     # a new category summary


def test_missing_field_equals_none():
    a, b = _manifest(), _manifest()
    del _entry(b, "KICKS", 0)["band"]                                # was None
    assert compare_manifests(a, b).ok
    _entry(b, "KICKS", 0)["retune"] = -1.0                           # new non-None field
    rep = compare_manifests(a, b)
    assert rep.fields["retune"] == 1


def test_loop_fields_and_summaries_are_compared():
    a, b = _manifest(), _manifest()
    _entry(b, "DRUMLOOPS", 0)["rotate_ms"] = 13.0
    b["categories"]["KICKS"]["source_samples"] = 10
    rep = compare_manifests(a, b)
    assert rep.fields["rotate_ms"] == 1
    assert ("categories.KICKS", "source_samples", 9, 10) in rep.summary


def test_sets_are_compared():
    a, b = _manifest(), _manifest()
    b["sets"]["KITS"]["entries"][0]["gain_db"] = -1.5
    b["sets"]["SLICE"]["entries"] = []
    rep = compare_manifests(a, b)
    assert rep.fields["gain_db"] == 1
    assert ("SLICE", "jungle-breakbeat-170/amen.wav") in rep.gone


def test_ignore_leaves_a_field_out():
    a, b = _manifest(), _manifest()
    _entry(b, "KICKS", 0)["support"] = 1
    assert not compare_manifests(a, b).ok
    assert compare_manifests(a, b, ignore=["support"]).ok


def test_duplicate_keys_fail():
    a, b = _manifest(), _manifest()
    b["categories"]["KICKS"]["entries"].append(dict(_entry(b, "KICKS", 0)))
    rep = compare_manifests(a, b)
    assert rep.duplicates and not rep.ok


def test_rel_src():
    assert rel_src(LIB + "A/b.wav") == "A/b.wav"
    assert rel_src("/x/lib/A/b.wav", roots=["/x/lib/"]) == "A/b.wav"
    assert rel_src("/elsewhere/b.wav") == "/elsewhere/b.wav"
    assert rel_src(None) is None


def _script():
    p = Path(__file__).resolve().parents[1] / "scripts" / "golden_compare.py"
    spec = importlib.util.spec_from_file_location("golden_compare", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exit_codes(tmp_path, capsys):
    main = _script().main
    ref, same, diff = tmp_path / "ref.json", tmp_path / "same", tmp_path / "diff.json"
    m = _manifest()
    ref.write_text(json.dumps(m))
    same.mkdir()
    (same / "manifest.json").write_text(json.dumps(m))            # a master dir works too
    m2 = _manifest()
    _entry(m2, "KICKS", 0)["out_md5"] = "zz"
    diff.write_text(json.dumps(m2))
    assert main([str(ref), str(same)]) == 0
    report = tmp_path / "report.json"
    assert main([str(ref), str(diff), "--json", str(report)]) == 1
    assert json.loads(report.read_text())["fields"] == {"out_md5": 1}
    assert main([str(ref), str(tmp_path / "missing.json")]) == 2
    assert "golden: DIFFERENT" in capsys.readouterr().out


def _v2(doc, labels=("class.oneshot", "kick")):
    doc = copy.deepcopy(doc)
    doc["fourier_manifest"] = 2
    for v in doc["categories"].values():
        for e in v["entries"]:
            e["labels"] = list(labels)
    return doc


def test_fields_newer_than_the_reference_format_are_skipped_not_failed():
    rep = compare_manifests(_manifest(), _v2(_manifest()))          # a format-1 reference, a format-2 build
    assert rep.ok and rep.skipped == ["labels"]
    assert rep.lines() == ["golden: identical (not compared, newer than the reference's format: labels)"]


def test_between_two_v2_manifests_labels_are_compared():
    rep = compare_manifests(_v2(_manifest()), _v2(_manifest(), labels=("class.oneshot", "snare")))
    assert not rep.ok and rep.fields["labels"] == 3 and rep.skipped == []
