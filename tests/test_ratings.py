"""Ableton-tag ratings: XMP parsing, verdicts, harvest lifecycle, scorecard."""
import json
import os

from fourier.packs.ratings import (
    LIVE_USER_XMP, apply_tags, keep_pins, misfiled_map, format_scorecard, harvest, load_store, read_folder_tags,
    scorecard, verdict_of,
)


def _xmp(items):
    lis = "".join(
        f'<rdf:li rdf:parseType="Resource"><ablFR:filePath>{f}</ablFR:filePath>'
        '<ablFR:keywords><rdf:Bag>' + "".join(f"<rdf:li>{k}</rdf:li>" for k in kws) +
        '</rdf:Bag></ablFR:keywords></rdf:li>' for f, kws in items)
    return ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
            'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" '
            'xmlns:ablFR="https://ns.ableton.com/xmp/fs-resources/1.0/">'
            f'<ablFR:items><rdf:Bag>{lis}</rdf:Bag></ablFR:items>'
            '</rdf:Description></rdf:RDF></x:xmpmeta>')


def _master(root, stamp, entries, tags=None):
    """entries: {cat: [(family, filename, src)]}; tags: {(cat, family): [(file, [kw])]}"""
    cats = {c: {"entries": [{"family": f, "out": f"{f}/{n}", "src": s} for f, n, s in es]}
            for c, es in entries.items()}
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump({"generated": stamp, "categories": cats}, fh)
    for (cat, fam), items in (tags or {}).items():
        d = os.path.join(root, cat, fam, "Ableton Folder Info")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "a.xmp"), "w") as fh:
            fh.write(_xmp(items))
    return str(root)


ENTRIES = {"KICKS": [("808-sub", "BD 808.wav", "/lib/SampleLibrary/A/BD 808.wav")],
           "SNARES": [("crack", "SD 1.wav", "/lib/SampleLibrary/A/SD 1.wav")]}


def test_verdict_of_group_and_priority():
    assert verdict_of(["Fourier|Keep"]) == "keep"
    assert verdict_of(["fourier|drop", "Fourier|Keep"]) == "drop"   # strongest wins
    assert verdict_of(["Drums|Kick"]) is None                       # other tag groups ignored
    assert verdict_of(["Fourier|Meh"]) is None                      # unknown values ignored
    assert verdict_of([]) is None


def test_read_folder_tags(tmp_path):
    m = _master(tmp_path / "m", "t1", ENTRIES,
                {("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])]})
    assert read_folder_tags(m) == {os.path.join("KICKS", "808-sub", "BD 808.wav"): ["Fourier|Keep"]}


def test_harvest_lifecycle_untag_clears_rebuild_does_not(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES, {
        ("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])],
        ("SNARES", "crack"): [("SD 1.wav", ["Fourier|Drop"])]})
    s = harvest(m, store, log=lambda x: None)
    assert s["added"] == 2
    R = load_store(store)["ratings"]
    assert R["/lib/SampleLibrary/A/BD 808.wav"]["verdict"] == "keep"
    assert R["/lib/SampleLibrary/A/SD 1.wav"]["category"] == "SNARES"

    # untag the snare in Live (same master) -> cleared
    os.remove(os.path.join(m, "SNARES", "crack", "Ableton Folder Info", "a.xmp"))
    s = harvest(m, store, log=lambda x: None)
    assert s["cleared"] == 1
    assert "/lib/SampleLibrary/A/SD 1.wav" not in load_store(store)["ratings"]

    # rebuild: new master, no sidecars -> the kick keeps its rating
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)
    s = harvest(m2, store, log=lambda x: None)
    assert s["cleared"] == 0
    assert load_store(store)["ratings"]["/lib/SampleLibrary/A/BD 808.wav"]["verdict"] == "keep"


def test_scorecard_counts_back_lost_and_rate(tmp_path):
    store = str(tmp_path / "ratings.json")
    entries1 = dict(ENTRIES, HATS=[("tick", "HH 1.wav", "/lib/SampleLibrary/A/HH 1.wav")])
    m1 = _master(tmp_path / "m1", "t1", entries1, {
        ("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])],
        ("SNARES", "crack"): [("SD 1.wav", ["Fourier|Drop"])],
        ("HATS", "tick"): [("HH 1.wav", ["Fourier|Keep"])]})
    harvest(m1, store, log=lambda x: None)

    # rebuild keeps the dropped snare, loses the kept hat
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)
    sc = scorecard(m2, store)
    assert sc["categories"]["KICKS"]["keep_rate"] == 1.0
    assert sc["categories"]["SNARES"]["drops_back"] == 1
    assert sc["categories"]["HATS"]["keeps_lost"] == 1
    assert sc["totals"]["rated"] == 2
    # a drop rated in the master being scored is not "back"
    assert scorecard(m1, store)["categories"]["SNARES"]["drops_back"] == 0
    assert any(line.startswith("TOTAL") for line in format_scorecard(sc))


BD_SRC = "/lib/SampleLibrary/A/BD 808.wav"
BD_REL = os.path.join("KICKS", "808-sub", "BD 808.wav")


def test_apply_tags_restores_rating_after_rebuild(tmp_path):
    store = str(tmp_path / "ratings.json")
    m1 = _master(tmp_path / "m1", "t1", ENTRIES,
                 {("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])]})
    harvest(m1, store, log=lambda x: None)
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)               # rebuild: no sidecars
    s = apply_tags(m2, store, log=lambda x: None)
    assert s["folders_written"] == 1
    assert os.path.exists(os.path.join(m2, "KICKS", "808-sub", "Ableton Folder Info", LIVE_USER_XMP))
    assert read_folder_tags(m2) == {BD_REL: ["Fourier|Keep"]}
    assert apply_tags(m2, store, log=lambda x: None)["folders_written"] == 0   # idempotent
    # harvesting the rebuilt master adopts it, so untagging there now clears
    harvest(m2, store, log=lambda x: None)
    assert load_store(store)["ratings"][BD_SRC]["master"] == "t2"


def test_apply_tags_preserves_other_keywords(tmp_path):
    from fourier.packs.ratings import _render_xmp
    store = str(tmp_path / "ratings.json")
    m1 = _master(tmp_path / "m1", "t1", ENTRIES,
                 {("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])]})
    harvest(m1, store, log=lambda x: None)
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)
    info = os.path.join(m2, "KICKS", "808-sub", "Ableton Folder Info")
    os.makedirs(info)
    with open(os.path.join(info, LIVE_USER_XMP), "w") as fh:     # Live's own sidecar
        fh.write(_render_xmp({"BD 808.wav": ["Drums|Kick", "Fourier|Drop"],
                              "Other.wav": ["Genres|Trap"]}, "Live"))
    apply_tags(m2, store, log=lambda x: None)
    tags = read_folder_tags(m2)
    assert tags[BD_REL] == ["Drums|Kick", "Fourier|Keep"]     # stored rating wins, other tag kept
    assert tags[os.path.join("KICKS", "808-sub", "Other.wav")] == ["Genres|Trap"]


OTHER_REL = os.path.join("KICKS", "808-sub", "Other.wav")


def _live_sidecar(master, items):
    from fourier.packs.ratings import _render_xmp
    info = os.path.join(master, "KICKS", "808-sub", "Ableton Folder Info")
    os.makedirs(info, exist_ok=True)
    p = os.path.join(info, LIVE_USER_XMP)
    with open(p, "w") as fh:
        fh.write(_render_xmp(items, "Updated by Ableton Index"))
    return p


def test_colors_survive_rebuild_and_clear_in_live(tmp_path):
    from fourier.packs.ratings import read_folder_items
    store = str(tmp_path / "ratings.json")
    m1 = _master(tmp_path / "m1", "t1", ENTRIES)
    _live_sidecar(m1, {"BD 808.wav": {"keywords": ["Fourier|Keep"], "colors": ["1"]}})
    harvest(m1, store, log=lambda x: None)
    assert load_store(store)["colors"][BD_SRC]["colors"] == ["1"]
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)               # rebuild
    apply_tags(m2, store, log=lambda x: None)
    assert read_folder_items(m2)[BD_REL] == {"keywords": ["Fourier|Keep"], "colors": ["1"]}
    harvest(m2, store, log=lambda x: None)                    # adopt the rebuilt master
    _live_sidecar(m2, {"BD 808.wav": {"keywords": ["Fourier|Keep"], "colors": []}})
    harvest(m2, store, log=lambda x: None)                    # color removed in Live
    assert BD_SRC not in load_store(store)["colors"]
    assert load_store(store)["ratings"][BD_SRC]["verdict"] == "keep"


def test_rewrite_preserves_live_colors_on_other_files(tmp_path):
    from fourier.packs.ratings import read_folder_items
    store = str(tmp_path / "ratings.json")
    m1 = _master(tmp_path / "m1", "t1", ENTRIES,
                 {("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])]})
    harvest(m1, store, log=lambda x: None)
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)
    _live_sidecar(m2, {"BD 808.wav": {"keywords": ["Fourier|Drop"], "colors": []},
                       "Other.wav": {"keywords": [], "colors": ["3"]}})
    s = apply_tags(m2, store, log=lambda x: None)
    assert s["folders_written"] == 1
    items = read_folder_items(m2)
    assert items[BD_REL]["keywords"] == ["Fourier|Keep"]
    assert items[OTHER_REL]["colors"] == ["3"]                # Live's color survives the rewrite


def test_unknown_sidecar_field_left_untouched(tmp_path):
    store = str(tmp_path / "ratings.json")
    m1 = _master(tmp_path / "m1", "t1", ENTRIES,
                 {("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])]})
    harvest(m1, store, log=lambda x: None)
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)
    p = _live_sidecar(m2, {"BD 808.wav": ["Fourier|Drop"]})
    raw = open(p).read().replace(
        "<ablFR:filePath>BD 808.wav</ablFR:filePath>",
        "<ablFR:filePath>BD 808.wav</ablFR:filePath><ablFR:rating>5</ablFR:rating>")
    with open(p, "w") as fh:
        fh.write(raw)
    s = apply_tags(m2, store, log=lambda x: None)
    assert s["folders_skipped"] == 1 and s["folders_written"] == 0
    assert open(p).read() == raw                               # byte-identical


def test_favorite_implies_keep_and_is_written_back(tmp_path):
    from fourier.packs.ratings import read_folder_items
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES)
    _live_sidecar(m, {"BD 808.wav": {"keywords": [], "colors": ["1"]}})   # red, untagged
    s = harvest(m, store, log=lambda x: None)
    r = load_store(store)["ratings"][BD_SRC]
    assert (r["verdict"], r["source"], s["from_favorites"]) == ("keep", "favorite", 1)
    apply_tags(m, store, log=lambda x: None)
    assert read_folder_items(m)[BD_REL] == {"keywords": ["Fourier|Keep"], "colors": ["1"]}
    # un-favorite in Live: the written Keep tag stays, so the rating stays
    _live_sidecar(m, {"BD 808.wav": {"keywords": ["Fourier|Keep"], "colors": []}})
    harvest(m, store, log=lambda x: None)
    assert load_store(store)["ratings"][BD_SRC]["verdict"] == "keep"


def test_explicit_tag_beats_favorite(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES)
    _live_sidecar(m, {"BD 808.wav": {"keywords": ["Fourier|Drop"], "colors": ["1"]}})
    harvest(m, store, log=lambda x: None)
    r = load_store(store)["ratings"][BD_SRC]
    assert (r["verdict"], r["source"]) == ("drop", "tag")


def test_other_colors_do_not_imply_keep(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES)
    _live_sidecar(m, {"BD 808.wav": {"keywords": [], "colors": ["4"]}})
    harvest(m, store, log=lambda x: None)
    assert BD_SRC not in load_store(store)["ratings"]
    assert load_store(store)["colors"][BD_SRC]["colors"] == ["4"]


def test_keep_pins_only_keeps(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES, {
        ("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Keep"])],
        ("SNARES", "crack"): [("SD 1.wav", ["Fourier|Drop"])]})
    harvest(m, store, log=lambda x: None)
    assert keep_pins(store) == {BD_SRC: "KICKS"}
    assert keep_pins(str(tmp_path / "missing.json")) == {}


def test_misfiled_map(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES, {
        ("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Misfiled"])],
        ("SNARES", "crack"): [("SD 1.wav", ["Fourier|Keep"])]})
    harvest(m, store, log=lambda x: None)
    assert misfiled_map(store) == {BD_SRC: "KICKS"}


def test_renamed_categories_map_on_load(tmp_path):
    import json
    from fourier.packs.ratings import keep_pins, load_store
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"version": 1, "ratings": {
        "/a.wav": {"category": "BELLS", "verdict": "keep", "out": "BELLS/x/a.wav"},
        "/b.wav": {"category": "ORCHESTRAL", "verdict": "misfiled"}}}))
    assert keep_pins(str(p)) == {"/a.wav": "ACOUSTIC"}
    st = load_store(str(p))
    assert st["ratings"]["/b.wav"]["category"] == "ACOUSTIC"
    assert st["ratings"]["/a.wav"]["out"] == "ACOUSTIC/x/a.wav"


# --- Move-<CAT> tags and hash rekeying (E6 / E4) ----------------------------------------------
def test_move_tag_is_misfiled_with_target(tmp_path):
    from fourier.packs.ratings import move_target, move_targets, tag_label
    assert verdict_of(["Fourier|Move-SNARES"]) == "misfiled"
    assert verdict_of(["Fourier|move to perc"]) == "misfiled"
    assert verdict_of(["Fourier|Move-NOWHERE"]) is None
    assert move_target(["Fourier|Move-snares"]) == "SNARES"
    assert move_target(["Fourier|Move-BELLS"]) == "ACOUSTIC"          # a renamed category
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES,
                {("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Move-PERC"])]})
    harvest(m, store, log=lambda x: None)
    r = load_store(store)["ratings"][BD_SRC]
    assert (r["verdict"], r["category"], r["target"]) == ("misfiled", "KICKS", "PERC")
    assert misfiled_map(store) == {BD_SRC: "KICKS"}
    assert keep_pins(store) == {BD_SRC: "PERC"}
    assert move_targets(store) == {BD_SRC: "PERC"}
    assert tag_label(r) == "Fourier|Move-PERC"
    # written back as the same tag while the file still sits where it was rated
    m2 = _master(tmp_path / "m2", "t2", ENTRIES)
    apply_tags(m2, store, log=lambda x: None)
    assert read_folder_tags(m2) == {BD_REL: ["Fourier|Move-PERC"]}


def test_move_to_own_category_is_plain_misfiled(tmp_path):
    store = str(tmp_path / "ratings.json")
    m = _master(tmp_path / "m", "t1", ENTRIES,
                {("KICKS", "808-sub"): [("BD 808.wav", ["Fourier|Move-KICKS"])]})
    harvest(m, store, log=lambda x: None)
    assert "target" not in load_store(store)["ratings"][BD_SRC]


class _Sess:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, q):
        class R:
            def __init__(s, rows):
                s._r = rows

            def all(s):
                return s._r
        return R(self.rows)


def test_rekey_by_hash_follows_moved_file(tmp_path):
    from fourier.packs.ratings import rekey_by_hash, save_store
    old = tmp_path / "old" / "BD 808.wav"
    new = tmp_path / "new" / "BD 808.wav"
    other = tmp_path / "x" / "SD.wav"
    for p in (old, new, other):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    store = str(tmp_path / "ratings.json")
    save_store({"version": 1, "ratings": {str(old): {"verdict": "keep", "category": "KICKS"},
                                          str(other): {"verdict": "drop", "category": "SNARES"}},
                "colors": {}, "auto_misfiled": {}}, store)
    # first pass: both files in the library -> hashes recorded
    s = rekey_by_hash(_Sess([(str(old), "h1"), (str(other), "h2")]), store, log=lambda x: None)
    assert s["hashed"] == 2 and s["moved"] == 0
    # the library moved the kick: its rating follows by hash
    old.unlink()
    s = rekey_by_hash(_Sess([(str(new), "h1"), (str(other), "h2")]), store, log=lambda x: None)
    R = load_store(store)["ratings"]
    assert s["moved"] == 1 and str(new) in R and str(old) not in R
    assert R[str(new)]["verdict"] == "keep"


def test_rekey_leaves_ambiguous_hash(tmp_path):
    from fourier.packs.ratings import rekey_by_hash, save_store
    a, b = tmp_path / "a" / "k1.wav", tmp_path / "b" / "k2.wav"
    for p in (a, b):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    store = str(tmp_path / "ratings.json")
    save_store({"version": 1, "ratings": {"/gone/k.wav": {"verdict": "keep", "category": "KICKS", "hash": "h"}},
                "colors": {}, "auto_misfiled": {}}, store)
    s = rekey_by_hash(_Sess([(str(a), "h"), (str(b), "h")]), store, log=lambda x: None)
    assert s["moved"] == 0 and s["orphaned"] == 1
    assert "/gone/k.wav" in load_store(store)["ratings"]
