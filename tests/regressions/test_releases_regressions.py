"""Regression tests for releases / publish / plan / additive builds.

Each test reproduces a defect the code must not bring back.
"""
import hashlib
import json
import os
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

from fourier.packs import curate as C
from fourier.packs import releases as R
from fourier.packs.curate_config import CATEGORIES
from fourier.packs.ratings import LIVE_USER_XMP, harvest, load_store


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_RSYNC_SHIM = r'''#!{py}
"""Minimal rsync stand-in: rsync -a [--delete] [--dry-run] SRC/ DST/"""
import os, shutil, sys
args = sys.argv[1:]
flags = [a for a in args if a.startswith("-")]
src, dst = [a for a in args if not a.startswith("-")][-2:]
if "--dry-run" in flags:
    sys.exit(0)
src, dst = src.rstrip("/"), dst.rstrip("/")
if os.path.realpath(src) == os.path.realpath(dst):
    sys.exit(0)
os.makedirs(dst, exist_ok=True)
if "--delete" in flags:
    for dp, dns, fns in os.walk(dst, topdown=False):
        rel = os.path.relpath(dp, dst)
        for f in fns:
            if not os.path.exists(os.path.join(src, rel, f)):
                os.remove(os.path.join(dp, f))
        for d in dns:
            p = os.path.join(dp, d)
            if not os.path.exists(os.path.join(src, rel, d)):
                shutil.rmtree(p, ignore_errors=True)
shutil.copytree(src, dst, dirs_exist_ok=True)
'''


@pytest.fixture
def fake_rsync(tmp_path, monkeypatch):
    b = tmp_path / "_bin"
    b.mkdir()
    p = b / "rsync"
    p.write_text(_RSYNC_SHIM.replace("{py}", sys.executable))
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ.get('PATH', '')}")
    return str(p)


@pytest.fixture
def db(tmp_path):
    from fourier.db import session as sess
    sess._engine = None
    sess._SessionLocal = None
    path = tmp_path / "lib.db"
    sess.init_db(path)
    s = sess.get_session()
    yield SimpleNamespace(session=s, path=str(path))
    s.close()
    sess._engine = None
    sess._SessionLocal = None


@pytest.fixture(autouse=True)
def _isolated_ratings(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_RATINGS", str(tmp_path / "ratings.json"))
    monkeypatch.delenv("FOURIER_CURATED_DIR", raising=False)


def _md5b(b):
    return hashlib.md5(b).hexdigest()


def _add_sample(session, path):
    from fourier.db.models import Sample
    s = Sample(path=path, filename=os.path.basename(path))
    session.add(s)
    session.commit()
    return s.id


def _write_tree(root, generated, files, extra_entry=None):
    """files: {(cat, family, filename): (bytes, src)} -> tree + manifest.json"""
    cats = {}
    for (cat, fam, fn), (data, src) in files.items():
        d = os.path.join(root, cat, fam)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, fn), "wb") as f:
            f.write(data)
        e = dict(family=fam, out=f"{fam}/{fn}", src=src, out_md5=_md5b(data), support=2)
        e.update(extra_entry or {})
        cats.setdefault(cat, {"entries": []})["entries"].append(e)
    for c in cats.values():
        c["files"] = len(c["entries"])
        c["families"] = len({e["family"] for e in c["entries"]})
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(dict(fourier_manifest=1, generated=generated, categories=cats), f)
    return str(root)


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


def _tag(root, cat, fam, fn, kw):
    d = os.path.join(root, cat, fam, "Ableton Folder Info")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, LIVE_USER_XMP), "w") as f:
        f.write(_xmp([(fn, [kw])]))


def _row(**kw):
    base = dict(id=1, rel_path="Pack/x.wav", path="/lib/Pack/x.wav", filename="x.wav",
                file_hash=None, file_format="wav", duration_s=0.5, brightness=0.5,
                noisiness=0.3, harmonicity=0.7, crest_factor=4.3, base_note=None,
                base_note_confidence=None, attack_time_ms=20.0, decay_time_ms=350.0,
                sub_weight=0.0, is_clipped=0, tempo_bpm=None, bpm_reliable=0,
                onset_rate_hz=None, chroma_concentration=None,
                spectral_flatness_mean=0.5, rms_mean=0.2, dc_offset_ratio=0.0,
                ableton_tags=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _no_additions(calls):
    def fake_build_taxonomy(session, category, out_dir, **kw):
        calls.append(dict(category=category, **kw))
        raise RuntimeError(f"{category}: only 0 samples after filtering")
    return fake_build_taxonomy


# ---------------------------------------------------------------------------
# Guards against: plan() says SAFE for a legacy (manifest-less) base whose audio changed
# ---------------------------------------------------------------------------
def test_plan_legacy_base_changed_audio_is_not_safe(tmp_path, db):
    root = tmp_path / "releases"
    v1 = root / "v1" / "KICKS" / "deep"
    v1.mkdir(parents=True)
    (v1 / "BD 1.wav").write_bytes(b"RIFF-old-audio")          # an older release format: no manifest
    R.import_version(db.session, "v1", root=str(root), log=lambda m: None)

    master = _write_tree(tmp_path / "master", "2000-01-02T00:00:00Z",
                         {("KICKS", "deep", "BD 1.wav"): (b"RIFF-NEW-audio", "/lib/BD 1.wav")})
    r = R.plan(db.session, "v1", master, log=lambda m: None)
    # same path, different bytes: a project pinned to v1 would play different audio
    assert r["safe"] is False
    assert r["changed"] == 1


# ---------------------------------------------------------------------------
# Guards against: additive exclusion runs before byte-hash/twin dedup -> duplicate audio added
# ---------------------------------------------------------------------------
def test_additive_exclusion_does_not_readd_byte_identical_duplicate():
    cfg = CATEGORIES["KICKS"]
    rows = [
        _row(id=1, filename="BD_808.wav", rel_path="PackA/BD_808.wav",
             path="/l/PackA/BD_808.wav", file_hash="h808"),       # already in base release
        _row(id=2, filename="BD_808.wav", rel_path="PackB/BD_808.wav",
             path="/l/PackB/BD_808.wav", file_hash="h808"),       # byte-identical copy
    ]
    emb_n = np.eye(4, dtype="float32")
    id2row = {1: 0, 2: 1}
    full, _, _ = C._select_records(rows, cfg, emb_n, id2row)
    assert [r["id"] for r in full] == [1]                    # full build: one copy
    add, _, _ = C._select_records(rows, cfg, emb_n, id2row, exclude_ids={1})
    # additive over a base holding id=1 must not add the identical file again
    assert add == []


# ---------------------------------------------------------------------------
# Guards against: additive exclusion is category-blind -> a Misfiled base sample never re-homes
# ---------------------------------------------------------------------------
def test_additive_exclusion_lets_misfiled_base_sample_rehome(tmp_path, db, fake_rsync,
                                                             monkeypatch):
    src = "/lib/Pack/Tom Low.wav"
    sid = _add_sample(db.session, src)
    root = tmp_path / "releases"
    _write_tree(root / "v2", "2000-01-01T00:00:00Z",
                {("KICKS", "deep", "Tom Low.wav"): (b"tom", src)})
    R.record_release(db.session, "v2", json.load(open(root / "v2" / "manifest.json")))
    # rated Misfiled in KICKS: the next build must re-home it elsewhere
    with open(os.environ["FOURIER_RATINGS"], "w") as f:
        json.dump({"version": 1, "ratings": {src: dict(verdict="misfiled", category="KICKS")}}, f)

    calls = []
    monkeypatch.setattr(R, "RELEASES_ROOT", str(root))
    monkeypatch.setattr(C, "build_taxonomy", _no_additions(calls))
    R.additive_build(db.session, "v2", str(tmp_path / "master"), log=lambda m: None)

    others = [c for c in calls if c["category"] != "KICKS"]
    assert others
    # it is only "already in the release" for KICKS; elsewhere it is a legitimate new add
    assert all(sid not in (c["exclude_ids"] or set()) for c in others)


# ---------------------------------------------------------------------------
# Guards against: superset manifest throws away base entries' fields (base_man loaded, unused)
# ---------------------------------------------------------------------------
def test_additive_superset_manifest_keeps_base_entry_fields(tmp_path, db, fake_rsync,
                                                            monkeypatch):
    root = tmp_path / "releases"
    _write_tree(root / "v2", "2000-01-01T00:00:00Z",
                {("DRUMLOOPS", "172bpm-amen", "amen.wav"): (b"amen", "/lib/amen.wav")},
                extra_entry=dict(son_cats=["DRUMLOOPS"], ab_cats=["PERC"], bpm=172.0,
                                 bpm_src="filename", bpm_fold=172.0,
                                 sononym=["loop"], ableton=["Drums|Break"]))
    R.record_release(db.session, "v2", json.load(open(root / "v2" / "manifest.json")))
    monkeypatch.setattr(R, "RELEASES_ROOT", str(root))
    monkeypatch.setattr(C, "build_taxonomy", _no_additions([]))
    out = tmp_path / "master"
    R.additive_build(db.session, "v2", str(out), log=lambda m: None)

    e = json.load(open(out / "manifest.json"))["categories"]["DRUMLOOPS"]["entries"][0]
    # review.py reads son_cats/ab_cats; loop entries carry bpm/bpm_src/bpm_fold
    for k in ("son_cats", "ab_cats", "bpm", "bpm_src", "bpm_fold", "sononym", "ableton"):
        assert k in e, f"superset manifest dropped base field {k!r}"


# ---------------------------------------------------------------------------
# Guards against: `build --base` never re-applies ratings -> stale release sidecars
#        resurrect an old verdict on the next harvest
# ---------------------------------------------------------------------------
def test_cli_additive_build_reapplies_ratings(tmp_path, db, fake_rsync, monkeypatch):
    from fourier import cli
    for mod in (cli.build, cli.releases):                            # fake audio bytes
        monkeypatch.setattr(mod, "_run_verify", lambda *a, **k: True)
    src = "/lib/Pack/BD 1.wav"
    root = tmp_path / "releases"
    files = {("KICKS", "deep", "BD 1.wav"): (b"kick", src)}
    _write_tree(root / "v2", "2000-01-01T00:00:00Z", files)
    _tag(str(root / "v2"), "KICKS", "deep", "BD 1.wav", "Fourier|Keep")   # at publish time
    R.record_release(db.session, "v2", json.load(open(root / "v2" / "manifest.json")))

    master = tmp_path / "master"
    _write_tree(master, "2000-01-01T00:00:00Z", files)
    _tag(str(master), "KICKS", "deep", "BD 1.wav", "Fourier|Drop")     # re-rated after publish
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(master))

    monkeypatch.setattr(R, "RELEASES_ROOT", str(root))
    monkeypatch.setattr(C, "build_taxonomy", _no_additions([]))
    res = CliRunner().invoke(cli.main, ["--db", db.path, "build", "--no-scan",
                                        "--base", "v2", "--out", str(master)])
    assert res.exit_code == 0, res.output
    assert load_store()["ratings"][src]["verdict"] == "drop"    # harvested before the build

    harvest(str(master), log=lambda m: None)                     # next build / pack ratings
    assert load_store()["ratings"][src]["verdict"] == "drop", \
        "stale Keep sidecar copied from v2 overrode the later Drop"


# ---------------------------------------------------------------------------
# Guards against: `build CATEGORY --base` ignores CATEGORY (and --no-loudness)
# ---------------------------------------------------------------------------
def _setup_min_release(tmp_path, db, monkeypatch):
    root = tmp_path / "releases"
    _write_tree(root / "v2", "2000-01-01T00:00:00Z",
                {("KICKS", "deep", "BD 1.wav"): (b"kick", "/lib/BD 1.wav")})
    R.record_release(db.session, "v2", json.load(open(root / "v2" / "manifest.json")))
    monkeypatch.setattr(R, "RELEASES_ROOT", str(root))
    calls = []
    monkeypatch.setattr(C, "build_taxonomy", _no_additions(calls))
    return calls


def test_cli_additive_build_honors_category(tmp_path, db, fake_rsync, monkeypatch):
    from fourier import cli
    for mod in (cli.build, cli.releases):                            # fake audio bytes
        monkeypatch.setattr(mod, "_run_verify", lambda *a, **k: True)
    calls = _setup_min_release(tmp_path, db, monkeypatch)
    res = CliRunner().invoke(cli.main, ["--db", db.path, "build", "KICKS", "--no-scan",
                                        "--base", "v2", "--out", str(tmp_path / "m")])
    assert res.exit_code == 0, res.output
    assert {c["category"] for c in calls} == {"KICKS"}


def test_cli_additive_build_honors_no_loudness(tmp_path, db, fake_rsync, monkeypatch):
    from fourier import cli
    for mod in (cli.build, cli.releases):                            # fake audio bytes
        monkeypatch.setattr(mod, "_run_verify", lambda *a, **k: True)
    calls = _setup_min_release(tmp_path, db, monkeypatch)
    res = CliRunner().invoke(cli.main, ["--db", db.path, "build", "--all", "--no-scan",
                                        "--base", "v2", "--no-loudness",
                                        "--out", str(tmp_path / "m")])
    assert res.exit_code == 0, res.output
    assert calls and all(c.get("loudness") is False for c in calls)


# ---------------------------------------------------------------------------
# Guards against: additive_build into the base release dir mutates the immutable release
# ---------------------------------------------------------------------------
def test_additive_build_refuses_to_write_into_a_release(tmp_path, db, fake_rsync, monkeypatch):
    _setup_min_release(tmp_path, db, monkeypatch)
    vdir = tmp_path / "releases" / "v2"
    before = (vdir / "manifest.json").read_bytes()
    try:
        R.additive_build(db.session, "v2", str(vdir), log=lambda m: None)
    except Exception:
        pass                                   # refusing is the fix
    assert (vdir / "manifest.json").read_bytes() == before, "immutable release v2 was rewritten"


# ---------------------------------------------------------------------------
# Guards against: `publish --force` over an existing version leaves stale files behind
# ---------------------------------------------------------------------------
def test_publish_force_does_not_leave_stale_files(tmp_path, db, fake_rsync):
    from fourier import cli
    root = tmp_path / "Fourier"
    _write_tree(root / "releases" / "v1", "2000-01-01T00:00:00Z",
                {("KICKS", "deep", "old.wav"): (b"old", "/lib/old.wav")})
    master = _write_tree(tmp_path / "master", "2000-01-02T00:00:00Z",
                         {("KICKS", "deep", "new.wav"): (b"new", "/lib/new.wav")})
    res = CliRunner().invoke(cli.main, ["--db", db.path, "publish", "--from", master,
                                        "--to", str(root), "--as-version", "1", "--force",
                                        "--no-verify"])     # fake audio bytes: tests the copy only
    assert res.exit_code == 0, res.output
    v1 = root / "releases" / "v1"
    assert (v1 / "KICKS" / "deep" / "new.wav").exists()
    man = json.load(open(v1 / "manifest.json"))
    on_disk = {os.path.relpath(os.path.join(dp, f), v1)
               for dp, _, fs in os.walk(v1) for f in fs if f.endswith(".wav")}
    in_man = {f"{c}/{e['out']}" for c, cd in man["categories"].items() for e in cd["entries"]}
    assert on_disk == in_man, f"files not in the release manifest: {on_disk - in_man}"


# ---------------------------------------------------------------------------
# Additive builds add only the new pack, within an allowance,
# into existing folders when they fit, and verify accepts base + allowance
# ---------------------------------------------------------------------------
def test_additive_only_pack_and_allowance(tmp_path, db, fake_rsync, monkeypatch):
    old = _add_sample(db.session, "/lib/SampleLibrary/Old/BD 9.wav")
    new = _add_sample(db.session, "/lib/SampleLibrary/NewPack/BD 1.wav")
    calls = _setup_min_release(tmp_path, db, monkeypatch)
    R.additive_build(db.session, "v2", str(tmp_path / "m"), categories=["KICKS"],
                     only_pack="NewPack", allowance=0.1, log=lambda m: None)
    c = calls[0]
    assert old in c["exclude_ids"] and new not in c["exclude_ids"]
    from fourier.packs.curate_config import BUDGETS
    assert c["budget_override"] == round(BUDGETS["KICKS"] * 0.1)


def _fake_additions(files):
    """build_taxonomy stand-in writing the given {family: [filename]} as a category tree."""
    def fake(session, category, out_dir, **kw):
        ents = []
        for fam, names in files.items():
            d = os.path.join(out_dir, category, fam)
            os.makedirs(d, exist_ok=True)
            for n in names:
                with open(os.path.join(d, n), "wb") as f:
                    f.write(n.encode())
                ents.append(dict(family=fam, out=f"{fam}/{n}", src=f"/lib/New/{n}"))
        with open(os.path.join(out_dir, category, "_manifest.json"), "w") as f:
            json.dump([dict(family=fam, copied=len(n)) for fam, n in files.items()], f)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump({"categories": {category: {"entries": ents}}}, f)
        return {}
    return fake


def test_additive_routes_close_group_into_existing_folder(tmp_path, db, fake_rsync, monkeypatch):
    _setup_min_release(tmp_path, db, monkeypatch)
    monkeypatch.setattr(C, "build_taxonomy", _fake_additions({"g1": ["BD a.wav", "BD b.wav"],
                                                              "g2": ["BD far.wav"]}))
    e = np.eye(3, dtype="float32")
    monkeypatch.setattr(R, "_centroids", lambda s, by: {f: (e[0] if f in ("deep", "g1") else e[1])
                                                        for f in by})
    out = tmp_path / "m"
    res = R.additive_build(db.session, "v2", str(out), categories=["KICKS"], log=lambda m: None)
    assert res["added"] == 3 and res["routed"] == 2
    assert sorted(os.listdir(out / "KICKS" / "deep")) == ["BD 1.wav", "BD a.wav", "BD b.wav"]
    assert (out / "KICKS" / "g2" / "BD far.wav").exists()
    man = json.load(open(out / "manifest.json"))
    assert man["base"] == "v2" and "add_allowance" in man
    fams = {e["family"] for e in man["categories"]["KICKS"]["entries"]}
    assert fams == {"deep", "g2"}
    meta = {f["family"]: f for f in json.load(open(out / "KICKS" / "_manifest.json"))}
    assert meta["g2"]["copied"] == 1


def test_verify_allows_base_plus_allowance(tmp_path):
    from fourier.packs.curate_config import BUDGETS
    from fourier.packs.verify import _allowed
    ctx = SimpleNamespace(man={"base": "v2", "add_allowance": 0.1})
    b, a = _allowed(ctx)
    assert a == 0.1 and b["KICKS"] == BUDGETS["KICKS"] + round(BUDGETS["KICKS"] * 0.1)
    b, a = _allowed(SimpleNamespace(man={}))
    assert a == 0.0 and b["KICKS"] == BUDGETS["KICKS"]


# ---------------------------------------------------------------------------
# Publish is atomic and checked
# ---------------------------------------------------------------------------
def test_publish_checks_the_copy_and_refuses_locked_versions(tmp_path, db, fake_rsync, monkeypatch):
    from fourier import cli
    from fourier.packs import device_lock
    root = tmp_path / "Fourier"
    master = _write_tree(tmp_path / "master", "2000-01-02T00:00:00Z",
                         {("KICKS", "deep", "k.wav"): (b"kick", "/lib/k.wav")})
    args = ["--db", db.path, "publish", "--from", master, "--to", str(root), "--no-verify"]
    res = CliRunner().invoke(cli.main, args)
    assert res.exit_code == 0, res.output
    rel = root / "releases"
    assert (rel / "v1" / "KICKS" / "deep" / "k.wav").exists() and not (rel / "v1.partial").exists()
    assert (rel / "LATEST.txt").read_text().strip() == "v1"
    # a copy that doesn't match the manifest never becomes a release
    man = json.load(open(os.path.join(master, "manifest.json")))
    man["categories"]["KICKS"]["entries"][0]["out_md5"] = "0" * 32
    json.dump(man, open(os.path.join(master, "manifest.json"), "w"))
    res = CliRunner().invoke(cli.main, args)
    assert res.exit_code == 1 and not (rel / "v2").exists() and (rel / "v2.partial").exists()
    # ...and the leftover copy blocks the next publish until it's dealt with
    res = CliRunner().invoke(cli.main, args)
    assert res.exit_code == 1 and "leftover copy: v2.partial" in res.output
    import shutil
    shutil.rmtree(rel / "v2.partial")
    # --force over a version a device lock uses is refused
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "m8_tracker.lock.json").write_text(json.dumps({"files": {"k": {"release": "v1"}}}))
    monkeypatch.setattr(device_lock, "LOCK_DIR", locks)
    res = CliRunner().invoke(cli.main, args + ["--as-version", "1", "--force"])
    assert res.exit_code == 1 and "device locks" in res.output


def test_releases_folder_problems_block_publish_and_show_in_versions(tmp_path, db, fake_rsync, monkeypatch):
    """A sync client can turn a release back into vN.partial and leave empty "vN 2" folders
    from re-cuts: versions lists them, publish refuses until they're
    fixed, and a device lock serving a missing release is a problem too."""
    from fourier import cli
    from fourier.packs import device_lock
    root = tmp_path / "Fourier"
    rel = root / "releases"
    (rel / "v1.partial").mkdir(parents=True)
    (rel / "v1 2").mkdir()
    (rel / "LATEST.txt").write_text("v1\n")
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "m8_tracker.lock.json").write_text(json.dumps({"files": {"k": {"release": "v1"}}}))
    monkeypatch.setattr(device_lock, "LOCK_DIR", locks)
    probs = cli._release_problems(str(rel))
    assert "leftover copy: v1.partial" in probs and "sync duplicate: 'v1 2'" in probs
    assert "LATEST.txt says v1, which isn't there" in probs
    assert "m8_tracker lock serves files from v1, which isn't there" in probs
    res = CliRunner().invoke(cli.main, ["--db", db.path, "releases", "--to", str(root)])
    assert "problem:" in res.output and "v1.partial" in res.output
    master = _write_tree(tmp_path / "master", "2000-01-02T00:00:00Z",
                         {("KICKS", "deep", "k.wav"): (b"kick", "/lib/k.wav")})
    res = CliRunner().invoke(cli.main, ["--db", db.path, "publish", "--from", master,
                                        "--to", str(root), "--no-verify"])
    assert res.exit_code == 1 and "needs fixing first" in res.output and not (rel / "v2").exists()


def test_release_staged_outside_a_cloud_synced_folder(tmp_path, db, fake_rsync, monkeypatch):
    """A release bound for a cloud-synced folder is copied and
    checked in a local staging folder on the same volume, then moved in with one rename, so a
    sync client never sees a half-written release; elsewhere it stages beside the target."""
    from fourier import cli
    synced = tmp_path / "Dropbox" / "releases"
    synced.mkdir(parents=True)
    st = tmp_path / "staging"
    monkeypatch.setenv("FOURIER_STAGING_DIR", str(st))
    assert cli._release_staging(str(synced)) == str(st)
    plain = tmp_path / "plain" / "releases"
    plain.mkdir(parents=True)
    assert cli._release_staging(str(plain)) == str(plain)
    # end to end: the release appears whole under its name, nothing is left in staging
    root = tmp_path / "Dropbox" / "Fourier"
    master = _write_tree(tmp_path / "master", "2000-01-02T00:00:00Z",
                         {("KICKS", "deep", "k.wav"): (b"kick", "/lib/k.wav")})
    res = CliRunner().invoke(cli.main, ["--db", db.path, "publish", "--from", master,
                                        "--to", str(root), "--no-verify"])
    assert res.exit_code == 0, res.output
    assert (root / "releases" / "v1" / "KICKS" / "deep" / "k.wav").exists()
    assert not list(st.iterdir()) and not list((root / "releases").glob("*.partial"))
