"""Review queue: bucket selection (pure) and harvest of tags made in _REVIEW."""
import json
import os

import numpy as np

from fourier.packs.ratings import LIVE_USER_XMP, _render_xmp, harvest, load_store
from fourier.packs.review import REVIEW_DIR, REVIEW_INDEX, select_queue


def _unit(*xs):
    v = np.array(xs, dtype="float32")
    return v / np.linalg.norm(v)


def _entries():
    E, V = [], {}
    for i in range(8):                                   # KICKS cluster around x
        s = f"/k/{i}.wav"
        E.append(dict(src=s, category="KICKS", family=f"k{i % 4}", out=f"k{i % 4}/{i}.wav",
                      support=2, son_cats=["KICKS"], ab_cats=["KICKS"]))
        V[s] = _unit(1, 0.05 * i, 0)
    for i in range(8):                                   # SNARES cluster around y
        s = f"/s/{i}.wav"
        E.append(dict(src=s, category="SNARES", family=f"s{i % 4}", out=f"s{i % 4}/{i}.wav",
                      support=2, son_cats=["SNARES"], ab_cats=["SNARES"]))
        V[s] = _unit(0.05 * i, 1, 0)
    # a "kick" that sounds like a snare, and a kick placed on disagreeing votes
    E.append(dict(src="/k/odd.wav", category="KICKS", family="k9", out="k9/odd.wav",
                  support=2, son_cats=["KICKS"], ab_cats=["KICKS"]))
    V["/k/odd.wav"] = _unit(0.1, 1, 0)
    E.append(dict(src="/k/shaky.wav", category="KICKS", family="k8", out="k8/shaky.wav",
                  support=1, son_cats=["TOMS"], ab_cats=["KICKS"]))
    V["/k/shaky.wav"] = _unit(1, 0.2, 0.1)
    return E, V


def test_select_queue_buckets():
    E, V = _entries()
    q = select_queue(E, V, {}, n=6, seed=1)
    why = {e["src"]: b for e, b, _ in q}
    assert why["/k/odd.wav"] == "misfile"
    assert why["/k/shaky.wav"] == "shaky"
    assert len(q) == 6 and len(why) == 6


def test_select_queue_excludes_rated_and_is_deterministic():
    E, V = _entries()
    rated = {"/k/odd.wav": {"verdict": "keep", "category": "KICKS"}}
    q1 = select_queue(E, V, rated, n=6, seed=3)
    assert "/k/odd.wav" not in {e["src"] for e, _, _ in q1}
    assert q1 == select_queue(E, V, rated, n=6, seed=3)


def test_select_queue_neighbors_of_drops_and_caps():
    E, V = _entries()
    rated = {"/s/0.wav": {"verdict": "drop", "category": "SNARES"}}
    q = select_queue(E, V, rated, n=10, seed=0, cat_cap=6, fam_cap=2)
    nb = [e for e, b, _ in q if b == "neighbor"]
    assert nb and all(e["category"] == "SNARES" for e in nb)
    assert max(sum(1 for e, _, _ in q if e["category"] == c) for c in ("KICKS", "SNARES")) <= 6
    fams = [(e["category"], e["family"]) for e, _, _ in q]
    assert max(fams.count(f) for f in set(fams)) <= 2


def test_select_queue_skips_structural_single_vote():
    E = [dict(src=f"/b/{i}.wav", category="STABS", family="b", out=f"b/{i}.wav", support=1,
              son_cats=["STABS"], ab_cats=[]) for i in range(5)]
    V = {e["src"]: _unit(1, i, 0) for i, e in enumerate(E)}
    assert not [b for _, b, _ in select_queue(E, V, {}, n=5, seed=0) if b == "shaky"]


def _master_with_queue(root, qid, tag=None):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"generated": "t1", "categories": {"SNARES": {"entries": [
            {"family": "crack", "out": "crack/SD 1.wav", "src": "/lib/SD 1.wav"}]}}}, f)
    q = os.path.join(root, REVIEW_DIR)
    os.makedirs(q, exist_ok=True)
    name = "SNARES - crack - SD 1.wav"
    with open(os.path.join(q, REVIEW_INDEX), "w") as f:
        json.dump({"id": qid, "items": {name: {"src": "/lib/SD 1.wav", "category": "SNARES",
                   "family": "crack", "out": "crack/SD 1.wav", "bucket": "shaky"}}}, f)
    if tag:
        info = os.path.join(q, "Ableton Folder Info")
        os.makedirs(info, exist_ok=True)
        with open(os.path.join(info, LIVE_USER_XMP), "w") as f:
            f.write(_render_xmp({name: [tag]}, "Live"))
    return str(root)


def test_harvest_reads_review_queue_and_survives_regeneration(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master_with_queue(tmp_path / "m", "review-1", tag="Fourier|Drop")
    s = harvest(m, store, log=lambda x: None)
    r = load_store(store)["ratings"]["/lib/SD 1.wav"]
    assert (s["unmatched"], r["verdict"], r["via"], r["master"]) == (0, "drop", "shaky", "review-1")
    assert r["out"] == os.path.join("SNARES", "crack", "SD 1.wav")
    # queue regenerated (new id, sidecar gone): the rating stays
    import shutil
    shutil.rmtree(os.path.join(m, REVIEW_DIR))
    _master_with_queue(tmp_path / "m", "review-2")
    harvest(m, store, log=lambda x: None)
    assert load_store(store)["ratings"]["/lib/SD 1.wav"]["verdict"] == "drop"


def test_untag_in_current_queue_clears(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master_with_queue(tmp_path / "m", "review-1", tag="Fourier|Keep")
    harvest(m, store, log=lambda x: None)
    os.remove(os.path.join(m, REVIEW_DIR, "Ableton Folder Info", LIVE_USER_XMP))
    assert harvest(m, store, log=lambda x: None)["cleared"] == 1


def test_select_queue_skips_siblings_of_rated():
    E = [dict(src=f"/p/Chord Stab C 0{i}.wav", category="FX", family="f", out=f"f/{i}.wav",
              support=2, son_cats=["FX"], ab_cats=["FX"]) for i in range(4, 8)]
    E += [dict(src=f"/q/Wind {i:02d}.wav", category="FX", family="g", out=f"g/{i}.wav",
               support=2, son_cats=["FX"], ab_cats=["FX"]) for i in range(4)]
    V = {e["src"]: _unit(1, 0.01 * i, 0) for i, e in enumerate(E)}
    rated = {"/p/Chord Stab C 04.wav": {"verdict": "misfiled", "category": "FX"}}
    q = select_queue(E, V, rated, n=8, seed=0, cat_cap=8, fam_cap=8)
    assert not [e for e, _, _ in q if e["src"].startswith("/p/")]


def test_select_queue_scattered_set_shows_drum_placement():
    E = [dict(src=f"/x/SampleLibrary/C/Sax/Sax Long {i:02d}.wav",
              category=("SNARES" if i < 2 else "ACOUSTIC"), family=f"f{i}", out=f"f{i}/{i}.wav",
              support=2, son_cats=["X"], ab_cats=["X"]) for i in range(5)]
    V = {e["src"]: _unit(1, 0, 0) for e in E}          # identical: no misfile picks
    q = select_queue(E, V, {}, n=4, seed=0, cat_cap=4, fam_cap=4)
    sc = [(e, d) for e, b, d in q if b == "scattered"]
    assert len(sc) == 1                                   # one note per set
    assert sc[0][0]["category"] == "SNARES"               # shown in its drum-folder spot
    assert "2 folders" in sc[0][1]


def test_detect_and_auto_misfile(tmp_path):
    from fourier.packs.ratings import misfiled_map
    from fourier.packs.review import auto_misfile, detect_misfiles
    E, V = _entries()
    d = detect_misfiles(E, V)
    assert d[0]["entry"]["src"] == "/k/odd.wav" and d[0]["other"] == "SNARES"
    m = tmp_path / "m"; os.makedirs(m)
    cats = {}
    for e in E:
        cats.setdefault(e["category"], {"entries": []})["entries"].append(
            {"family": e["family"], "out": e["out"], "src": e["src"]})
    with open(m / "manifest.json", "w") as f:
        json.dump({"generated": "t1", "categories": cats}, f)
    store = str(tmp_path / "ratings.json")
    assert auto_misfile(str(m), margin=0.3, store_path=store, dry_run=True, vecs=V,
                        log=lambda x: None)
    assert misfiled_map(store) == {}                           # dry run stores nothing
    auto_misfile(str(m), margin=0.3, store_path=store, vecs=V, log=lambda x: None)
    assert misfiled_map(store) == {"/k/odd.wav": "KICKS"}
    # a human rating on the file wins
    s = json.load(open(store))
    s["ratings"]["/k/odd.wav"] = {"verdict": "keep", "category": "KICKS"}
    json.dump(s, open(store, "w"))
    assert misfiled_map(store) == {}
    auto_misfile(str(m), store_path=store, clear=True, log=lambda x: None)
    assert json.load(open(store))["auto_misfiled"] == {}


def test_build_queue_failure_keeps_the_previous_queue(tmp_path, monkeypatch):
    """Rebuilding the queue must not leave a _REVIEW without its index if it fails midway
    (e.g. a stale manifest entry whose file is gone): tags on the old queue stay mappable."""
    import json
    import pytest
    from fourier.packs import review as RV
    m = tmp_path / "m"
    (m / "KICKS" / "fam").mkdir(parents=True)
    (m / "KICKS" / "fam" / "a.wav").write_bytes(b"x")
    entries = [dict(family="fam", out="fam/a.wav", src="/l/a.wav"),
               dict(family="fam", out="fam/gone.wav", src="/l/gone.wav")]
    (m / "manifest.json").write_text(json.dumps(
        {"generated": "t1", "categories": {"KICKS": {"entries": entries}}}))
    old = m / RV.REVIEW_DIR
    old.mkdir()
    (old / RV.REVIEW_INDEX).write_text(json.dumps({"id": "review-old", "items": {}}))
    monkeypatch.setattr(RV, "_load_vectors", lambda srcs: {})
    monkeypatch.setattr(RV, "select_queue", lambda entries, vecs, ratings, n, seed: [
        (dict(e, category="KICKS"), "random", "") for e in entries])
    monkeypatch.setenv("FOURIER_RATINGS", str(tmp_path / "r.json"))
    with pytest.raises(OSError):
        RV.build_queue(str(m), n=2, log=lambda x: None)
    assert json.loads((old / RV.REVIEW_INDEX).read_text())["id"] == "review-old"
