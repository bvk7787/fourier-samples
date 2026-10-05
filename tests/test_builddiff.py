"""Build diffs (packs/builddiff.py): manifest diff, CHANGELOG.md, archived builds."""
import json

import pytest

from fourier.packs import builddiff as B


def _e(fam, src):
    return {"family": fam, "out": f"{fam}/{src}", "src": f"/l/SampleLibrary/V/{src}"}


OLD = {"generated": "2000-01-25T10:00:00", "categories": {
    "PADS": {"entries": [_e("gritty-bright", "p1.wav"), _e("gritty-bright", "p2.wav"),
                         _e("old", "p3.wav"), _e("riser", "Riser 1.wav")]},
    "FX": {"entries": [_e("impact-dark", "i1.wav")]}}}
NEW = {"generated": "2000-01-26T10:00:00", "categories": {
    "PADS": {"entries": [_e("bright-gritty-dark", "p1.wav"), _e("bright-gritty-dark", "p2.wav"),
                         _e("new", "p9.wav")]},
    "FX": {"entries": [_e("impact-dark", "i1.wav"), _e("riser-bright", "Riser 1.wav")]}}}


@pytest.fixture(autouse=True)
def _builds_home(tmp_path, monkeypatch):
    # builds_dir() sits next to the ratings store
    monkeypatch.setenv("FOURIER_RATINGS", str(tmp_path / "home" / "ratings.json"))


def _master(root, doc):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(doc))
    return root


def test_build_diff_renames_and_moves():
    d = B.diff_manifests(OLD, NEW)
    p = d["categories"]["PADS"]
    assert p["renamed"] == [("gritty-bright", "bright-gritty-dark", 1.0)]
    assert ("new", 1) in p["new_folders"] and ("old", 1) in p["gone_folders"]
    assert p["moved_out"] == [("FX", 1)] and d["categories"]["FX"]["moved_in"] == [("PADS", 1)]
    md = B.format_diff(d)
    assert "renamed: gritty-bright -> bright-gritty-dark" in md and "## FX" in md


def test_unchanged_category_is_left_out_of_the_changelog():
    d = B.diff_manifests(OLD, OLD)
    assert all(not c["renamed"] and not c["added"] and not c["removed"] for c in d["categories"].values())
    md = B.format_diff(d)
    assert "0 files added, 0 removed, 0 folders renamed." in md and "## " not in md


def test_archive_then_changelog_against_the_last_build(tmp_path):
    assert B.write_changelog(_master(tmp_path / "m0", OLD), log=lambda *a: None) is None   # nothing archived yet
    dest = B.archive_build(_master(tmp_path / "m1", OLD))
    assert dest.parent == B.builds_dir() and dest.name == "20000125T100000.json"
    new = _master(tmp_path / "m2", NEW)
    ch = B.write_changelog(new, log=lambda *a: None)
    assert ch == new / "CHANGELOG.md"
    assert "renamed: gritty-bright -> bright-gritty-dark" in ch.read_text()
    assert B.archive_build(tmp_path / "empty") is None                          # no manifest


def test_load_manifest_from_folder_file_or_stamp(tmp_path):
    m = _master(tmp_path / "m", OLD)
    B.archive_build(m)
    assert B.load_manifest(m) == OLD
    assert B.load_manifest(m / "manifest.json") == OLD
    assert B.load_manifest("20000125T100000") == OLD
    assert B.load_manifest("20000125T100000.json") == OLD


def test_an_installed_copys_build_is_named_by_its_version():
    """An installed copy has no git sha: the CHANGELOG and `fourier diff` name the version
    that built each side (the manifest's fourier_version), never "(?)"."""
    d = B.diff_manifests({"generated": "a", "fourier_version": "0.1.0", "categories": {}},
                         {"generated": "b", "git_sha": "abc1234", "categories": {}})
    text = B.format_diff(d)
    assert "(version 0.1.0)" in text and "(abc1234)" in text and "(?)" not in text
    assert "(an earlier version)" in B.format_diff(B.diff_manifests({"categories": {}}, {"categories": {}}))


def test_the_golden_compare_ignores_the_version():
    from fourier.packs.golden import compare_manifests
    cats = {"KICKS": {"files": 1, "entries": [{"family": "a", "out": "a/k.wav", "src": "/l/k.wav"}]}}
    assert compare_manifests({"categories": cats}, {"categories": cats, "fourier_version": "9.9"}).ok


def test_a_renamed_library_folder_renames_no_master_folder():
    """Every source of a folder moved in the library (another path, the same file at the same
    place in the master, its audio unchanged): the folder is the same one, and the CHANGELOG
    says only that files moved in the library."""
    def e(src, md5):
        return {"family": "kick", "out": f"kick/{src.rsplit('/', 1)[-1]}", "src": src, "out_md5": md5}
    old = {"categories": {"KICKS": {"entries": [e(f"/l/SampleLibrary/Acme/K{i}.wav", f"m{i}") for i in range(3)]}}}
    new = {"categories": {"KICKS": {"entries": [e(f"/l/SampleLibrary/Acme Audio/K{i}.wav", f"m{i}")
                                                for i in range(3)]}}}
    d = B.diff_manifests(old, new)
    k = d["categories"]["KICKS"]
    assert k["new_folders"] == [] and k["gone_folders"] == [] and k["renamed"] == []
    assert k["added"] == k["removed"] == 0 and k["library_moved"] == 3
    text = B.format_diff(d)
    assert "3 files moved in the library, master unchanged." in text and "new folder" not in text
