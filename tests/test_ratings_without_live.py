"""Ratings without Ableton Live: `fourier review rate` and `fourier review import` write the
ratings store Live's harvest writes, and builds, score and harvest read them the same way."""
import json
import os

import pytest
from click.testing import CliRunner

from fourier.cli import main
from fourier.cli import review as RV
from fourier.packs.ratings import (
    LIVE_USER_XMP, MANUAL_STAMP, drop_set, find_rateable, harvest, keep_pins, load_store,
    misfiled_map, move_targets, read_folder_tags, scorecard,
)

LIB = "/lib/SampleLibrary/Northwind/Pack One"
ENTRIES = {"KICKS": [("round", "Tone_Kick_01.wav", f"{LIB}/Tone Kick 01.wav"),
                     ("round", "Tone_Kick_02.wav", f"{LIB}/Tone Kick 02.wav")],
           "SNARES": [("tight", "Tone_Snare_01.wav", f"{LIB}/Tone Snare 01.wav")]}


def _master(root, stamp="2000-01-01T00:00:00Z"):
    cats = {c: {"entries": [{"family": f, "out": f"{f}/{n}", "src": s} for f, n, s in es]}
            for c, es in ENTRIES.items()}
    sets = {"SLICE": {"entries": [{"family": "round", "out": "round/Tone_Kick_02.wav",
                                   "src": f"{LIB}/Tone Kick 02.wav",
                                   "came_from": "KICKS/round/Tone_Kick_02.wav"}]}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"generated": stamp, "categories": cats, "sets": sets}))
    for c, es in ENTRIES.items():
        for f, n, _s in es:
            (root / c / f).mkdir(parents=True, exist_ok=True)
            (root / c / f / n).write_bytes(b"")
    return root


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = tmp_path / "ratings.json"
    monkeypatch.setenv("FOURIER_RATINGS", str(store))
    return _master(tmp_path / "master"), str(store)


def _cli(*args, input=None):
    return CliRunner().invoke(main, ["review", *args], input=input)


def test_find_rateable_takes_what_why_takes(env):
    m, _ = env
    paths = lambda q: sorted(t["path"] for t in find_rateable(str(m), q))
    assert paths("Tone Kick 01") == ["KICKS/round/Tone_Kick_01.wav"]           # spaces or underscores
    assert paths("tone_kick") == ["KICKS/round/Tone_Kick_01.wav", "KICKS/round/Tone_Kick_02.wav"]
    assert paths(f"{LIB}/Tone Snare 01.wav") == ["SNARES/tight/Tone_Snare_01.wav"]   # source path
    assert paths("SLICE/round/Tone_Kick_02.wav") == ["KICKS/round/Tone_Kick_02.wav"]  # a set copy
    assert paths(str(m / "KICKS" / "round" / "Tone_Kick_01.wav")) == ["KICKS/round/Tone_Kick_01.wav"]
    assert paths("nothing like it") == []


def test_rate_stores_what_a_build_reads(env):
    m, store = env
    r = _cli("rate", "Tone Kick 01", "keep", "--from", str(m))
    assert r.exit_code == 0, r.output
    assert "KICKS/round/Tone_Kick_01.wav: Keep" in r.output
    assert _cli("rate", "Tone Snare 01", "misfiled", "--to", "claps", "--from", str(m)).exit_code == 0
    assert _cli("rate", "KICKS/round/Tone_Kick_02.wav", "DROP", "--from", str(m)).exit_code == 0
    R = load_store(store)["ratings"]
    k = R[f"{LIB}/Tone Kick 01.wav"]
    assert (k["verdict"], k["category"], k["family"], k["out"], k["master"], k["source"]) == \
        ("keep", "KICKS", "round", "KICKS/round/Tone_Kick_01.wav", MANUAL_STAMP, "cli")
    assert keep_pins(store) == {f"{LIB}/Tone Kick 01.wav": "KICKS", f"{LIB}/Tone Snare 01.wav": "CLAPS"}
    assert drop_set(store) == {f"{LIB}/Tone Kick 02.wav"}
    assert misfiled_map(store) == {f"{LIB}/Tone Snare 01.wav": "SNARES"}
    assert move_targets(store) == {f"{LIB}/Tone Snare 01.wav": "CLAPS"}
    # a harvest of the same master (no Live tags in it) keeps every one of them
    harvest(str(m), store, log=lambda *a: None)
    assert len(load_store(store)["ratings"]) == 3
    sc = scorecard(str(m), store)
    assert sc["totals"]["rated"] == 3 and sc["categories"]["KICKS"]["keep"] == 1
    # no Live sidecar was written into the master
    assert not list(m.rglob("Ableton Folder Info"))


def test_rate_refuses_what_it_cannot_store(env):
    m, store = env
    r = _cli("rate", "Tone Kick 01", "keep", "--to", "CLAPS", "--from", str(m))
    assert r.exit_code == 2 and "misfiled" in r.output
    r = _cli("rate", "Tone Kick 01", "misfiled", "--to", "NOWHERE", "--from", str(m))
    assert r.exit_code == 2 and "NOWHERE" in r.output
    r = _cli("rate", "nothing like it", "keep", "--from", str(m))
    assert r.exit_code == 1 and "no file in the master" in r.output
    assert not os.path.exists(store)


def test_rate_asks_which_file_when_a_name_matches_several(env, monkeypatch):
    m, store = env
    r = _cli("rate", "tone kick", "keep", "--from", str(m))
    assert r.exit_code == 2 and "matches 2 files" in r.output and "master path" in r.output
    monkeypatch.setattr(RV, "_interactive", lambda: True)
    r = _cli("rate", "tone kick", "keep", "--from", str(m), input="2\n")
    assert r.exit_code == 0, r.output
    assert list(load_store(store)["ratings"]) == [f"{LIB}/Tone Kick 02.wav"]


def test_import_a_csv(env, tmp_path):
    m, store = env
    csv = tmp_path / "r.csv"
    csv.write_text("﻿Name,Rating,Category\n"
                   "Tone Kick 01,keep,\n"
                   "SNARES/tight/Tone_Snare_01.wav,Misfiled,CLAPS\n"
                   "tone kick,drop,\n"                      # two files: skipped
                   "nothing like it,keep,\n"                # none: skipped
                   "Tone Kick 02,meh,\n"                    # not a rating: skipped
                   ",,\n")
    r = _cli("import", str(csv), "--from", str(m), "--dry-run")
    assert r.exit_code == 1 and "would rate 2 files" in r.output and "3 rows skipped" in r.output
    assert "line 4" in r.output and "matches 2 files" in r.output and "line 5" in r.output
    assert not os.path.exists(store)
    r = _cli("import", str(csv), "--from", str(m))
    assert r.exit_code == 1 and "rated 2 files (1 keep, 0 drop, 1 misfiled)" in r.output
    R = load_store(store)["ratings"]
    assert {k: (v["verdict"], v.get("target"), v["source"]) for k, v in R.items()} == {
        f"{LIB}/Tone Kick 01.wav": ("keep", None, "csv"),
        f"{LIB}/Tone Snare 01.wav": ("misfiled", "CLAPS", "csv")}
    bad = tmp_path / "bad.csv"
    bad.write_text("file_name,stars\nx,5\n")
    r = _cli("import", str(bad), "--from", str(m))
    assert r.exit_code == 2 and "header row" in r.output


def test_with_live_the_tags_follow_a_rating_given_here(env):
    """A master Live has tagged: a rating given on the command line rewrites the file's tag,
    so the next harvest (every build runs one) doesn't bring the old verdict back."""
    m, store = env
    info = m / "KICKS" / "round" / "Ableton Folder Info"
    info.mkdir()
    (info / LIVE_USER_XMP).write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:ablFR="https://ns.ableton.com/xmp/fs-resources/1.0/">'
        '<ablFR:items><rdf:Bag><rdf:li rdf:parseType="Resource"><ablFR:filePath>Tone_Kick_01.wav</ablFR:filePath>'
        '<ablFR:keywords><rdf:Bag><rdf:li>Fourier|Keep</rdf:li></rdf:Bag></ablFR:keywords></rdf:li>'
        '</rdf:Bag></ablFR:items></rdf:Description></rdf:RDF></x:xmpmeta>')
    assert _cli("rate", "Tone Kick 01", "drop", "--from", str(m)).exit_code == 0
    assert read_folder_tags(str(m))["KICKS/round/Tone_Kick_01.wav"] == ["Fourier|Drop"]
    harvest(str(m), store, log=lambda *a: None)
    assert load_store(store)["ratings"][f"{LIB}/Tone Kick 01.wav"]["verdict"] == "drop"


def test_review_help_says_which_subcommands_need_live():
    out = CliRunner().invoke(main, ["review", "--help"], terminal_width=100).output
    assert "Without Live: rate, import, queue, score, misfiles." in out
    assert "Needs Live:   ratings" in out
    assert "(needs Live)" in CliRunner().invoke(main, ["review", "--help"]).output


def test_clear_removes_a_rating_given_here_or_one_whose_file_is_gone(env):
    m, store = env
    assert _cli("rate", "Tone Kick 01", "keep", "--from", str(m)).exit_code == 0
    r = _cli("rate", "Tone Kick 01", "clear", "--from", str(m))
    assert r.exit_code == 0 and "Keep rating cleared" in r.output, r.output
    assert f"{LIB}/Tone Kick 01.wav" not in load_store(store)["ratings"]
    # a stored rating whose file left the master (moved or deleted in the library)
    doc = json.loads(open(store).read())
    doc["ratings"][f"{LIB}/Old Pack/Tone Kick 77.wav"] = dict(verdict="drop", category="KICKS",
                                                             master="manual", source="cli")
    open(store, "w").write(json.dumps(doc))
    r = _cli("rate", "Tone Kick 77", "clear", "--from", str(m))
    assert r.exit_code == 0 and "Drop rating cleared" in r.output
    r = _cli("rate", "nothing like it", "clear", "--from", str(m))
    assert r.exit_code == 1 and "no stored rating matches" in r.output


def test_clear_points_a_live_tagged_rating_to_live(env):
    m, store = env
    info = m / "KICKS" / "round" / "Ableton Folder Info"
    info.mkdir()
    (info / LIVE_USER_XMP).write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:ablFR="https://ns.ableton.com/xmp/fs-resources/1.0/">'
        '<ablFR:items><rdf:Bag><rdf:li rdf:parseType="Resource"><ablFR:filePath>Tone_Kick_01.wav</ablFR:filePath>'
        '<ablFR:keywords><rdf:Bag><rdf:li>Fourier|Keep</rdf:li></rdf:Bag></ablFR:keywords></rdf:li>'
        '</rdf:Bag></ablFR:items></rdf:Description></rdf:RDF></x:xmpmeta>')
    harvest(str(m), store, log=lambda *a: None)
    r = _cli("rate", "Tone Kick 01", "clear", "--from", str(m))
    assert r.exit_code == 2 and "tag in Live's browser" in r.output
    assert load_store(store)["ratings"][f"{LIB}/Tone Kick 01.wav"]["verdict"] == "keep"
