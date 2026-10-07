"""DB-tracked releases + plan/diff + additive builds.

A *release* is a published immutable version (v1, v2, ...). Its per-file manifest
is materialised into the DB (`release` / `release_file`) as the source of truth
for what has been transferred to hardware, so we can:

  - `publish --dry-run --base v2`  diff a prospective build against a release BEFORE
    transferring, and say plainly whether it is additive (safe) or would
    rename/replace/remove files a live device project references.
  - `fourier build --base v2`  build ADDITIVELY: copy the base verbatim,
    then add only source samples not already in the release, under new family
    folders. The result is a strict superset — re-transferring only adds files,
    so a project pinned to v2 can't break.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict

from sqlalchemy import select, text

from ..places import releases_root
from ..safety import UnsafePath
from . import manifests

# Canonical releases live here: <[output] publish>/releases (fourier/places.py), read once
RELEASES_ROOT = releases_root()

DDL = [
    """CREATE TABLE IF NOT EXISTS release (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT UNIQUE,
        created_at TEXT, seed INTEGER, git_sha TEXT, config_hash TEXT,
        clap_model TEXT, n_files INTEGER, notes TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS release_file (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        release_id INTEGER, category TEXT, family TEXT,
        out_path TEXT, src_path TEXT, src_sample_id INTEGER,
        out_md5 TEXT, support INTEGER
    )""",
    "CREATE INDEX IF NOT EXISTS ix_relfile_rel ON release_file(release_id)",
    "CREATE INDEX IF NOT EXISTS ix_relfile_src ON release_file(src_sample_id)",
]


def _md5_file(path):
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ddl(session):
    """The release tables. DuckDB has no AUTOINCREMENT: there an id comes from a sequence."""
    ddl = DDL
    if session.get_bind().dialect.name != "sqlite":
        for table in ("release", "release_file"):
            session.execute(text(f"CREATE SEQUENCE IF NOT EXISTS {table}_id_seq"))
        ddl = [s.replace("id INTEGER PRIMARY KEY AUTOINCREMENT",
                         f"id INTEGER PRIMARY KEY DEFAULT nextval('{t}_id_seq')")
               for s, t in zip(DDL, ("release", "release_file", None, None))]
    for s in ddl:
        session.execute(text(s))
    session.commit()


def manifest_counts(manifest: dict) -> dict:
    """What a master or release holds, counted the one way every command reports it:
    categories with files, the audio files in them (one per source), and the copies the
    kit and slice sets (KITS, SLICE) add."""
    cats = manifest.get("categories") or {}
    sets = manifest.get("sets") or {}
    n = {c: len(cd.get("entries") or ()) for c, cd in cats.items() if isinstance(cd, dict)}
    return {"categories": sum(1 for v in n.values() if v), "files": sum(n.values()),
            "set_files": sum(len(sd.get("entries") or ()) for sd in sets.values() if isinstance(sd, dict))}


def describe_counts(c: dict) -> str:
    """'7 categories, 66 audio files (+12 copies in the kit and slice sets)'."""
    out = (f"{c['categories']} categor{'y' if c['categories'] == 1 else 'ies'}, "
           f"{c['files']:,} audio files")
    if c.get("set_files"):
        out += f" (+{c['set_files']:,} copies in the kit and slice sets)"
    return out


def _path2id(session):
    from ..db.models import Sample
    return {p: i for i, p in session.execute(select(Sample.id, Sample.path)).all()}


def record_release(session, version, manifest, notes=""):
    """Insert a release + release_file rows from a build manifest dict (idempotent)."""
    _ddl(session)
    p2id = _path2id(session)
    rid = session.execute(text("SELECT id FROM release WHERE version=:v"), {"v": version}).scalar()
    if rid:
        session.execute(text("DELETE FROM release_file WHERE release_id=:r"), {"r": rid})
        session.execute(text("DELETE FROM release WHERE id=:r"), {"r": rid})
    cats = manifest.get("categories", {})
    nfiles = sum(len(c.get("entries", [])) for c in cats.values())
    session.execute(text(
        "INSERT INTO release(version,created_at,seed,git_sha,config_hash,clap_model,n_files,notes)"
        " VALUES(:v,:t,:s,:g,:c,:m,:n,:notes)"),
        dict(v=version, t=manifest.get("generated"), s=manifest.get("seed"),
             g=manifest.get("git_sha"), c=manifest.get("config_hash"),
             m=manifest.get("clap_model"), n=nfiles, notes=notes))
    rid = session.execute(text("SELECT id FROM release WHERE version=:v"), {"v": version}).scalar()
    for cat, cd in cats.items():
        for e in cd.get("entries", []):
            session.execute(text(
                "INSERT INTO release_file"
                "(release_id,category,family,out_path,src_path,src_sample_id,out_md5,support)"
                " VALUES(:r,:cat,:fam,:op,:sp,:sid,:md5,:sup)"),
                dict(r=rid, cat=cat, fam=e.get("family"), op=f"{cat}/{e['out']}",
                     sp=e.get("src"), sid=p2id.get(e.get("src")),
                     md5=e.get("out_md5"), sup=e.get("support")))
    session.commit()
    return dict(version=version, files=nfiles)


def import_version(session, version, root=RELEASES_ROOT, log=print):
    """Record a release from a published version's manifest.json, or scan its
    files when there is no manifest (an older release format: no source map)."""
    _ddl(session)
    vdir = os.path.join(root, version)
    mp = os.path.join(vdir, "manifest.json")
    if os.path.exists(mp):
        r = record_release(session, version, manifests.read(mp), notes="from manifest")
        log(f"{version}: recorded {r['files']} files from manifest")
        return r
    if not os.path.isdir(vdir):
        log(f"{version}: not found at {vdir}")
        return None
    session.execute(text("DELETE FROM release_file WHERE release_id IN "
                         "(SELECT id FROM release WHERE version=:v)"), {"v": version})
    session.execute(text("DELETE FROM release WHERE version=:v"), {"v": version})
    session.execute(text("INSERT INTO release(version,notes,n_files) VALUES(:v,:notes,0)"),
                    dict(v=version, notes="legacy scan (no source map)"))
    rid = session.execute(text("SELECT id FROM release WHERE version=:v"), {"v": version}).scalar()
    n = 0
    for dp, _, fs in os.walk(vdir):
        for f in fs:
            if f.lower().endswith(".wav"):
                rel = os.path.relpath(os.path.join(dp, f), vdir)
                parts = rel.split("/")
                session.execute(text(
                    "INSERT INTO release_file(release_id,category,family,out_path,out_md5) "
                    "VALUES(:r,:c,:f,:o,:m)"),
                    dict(r=rid, c=parts[0], f=(parts[1] if len(parts) > 2 else ""), o=rel,
                         m=_md5_file(os.path.join(dp, f))))
                n += 1
    session.execute(text("UPDATE release SET n_files=:n WHERE id=:r"), {"n": n, "r": rid})
    session.commit()
    log(f"{version}: recorded {n} files (legacy scan, no source map)")
    return dict(version=version, files=n)


def read_release(session, version):
    rid = session.execute(text("SELECT id FROM release WHERE version=:v"), {"v": version}).scalar()
    if not rid:
        return None
    return session.execute(text(
        "SELECT category,family,out_path,src_path,src_sample_id,out_md5,support "
        "FROM release_file WHERE release_id=:r"), {"r": rid}).all()


def list_releases(session):
    _ddl(session)
    return session.execute(text(
        "SELECT version,created_at,seed,git_sha,n_files,notes FROM release "
        "ORDER BY id")).all()


def _master_entries(master_dir):
    """{out_path: entry} from a master's manifest.json."""
    man = manifests.read(os.path.join(master_dir, "manifest.json"))
    out = {}
    for cat, cd in man.get("categories", {}).items():
        for e in cd.get("entries", []):
            out[f"{cat}/{e['out']}"] = e
    return man, out


def plan(session, base_version, master_dir, log=print):
    """Diff the current master against a base release. Reports adds / removes /
    content-changes / moves, and whether re-transferring would be additive (safe)."""
    base = read_release(session, base_version)
    if base is None:
        log(f"no release {base_version} in the DB — run: fourier releases import {base_version}")
        return None
    base_by_path = {r.out_path: r for r in base}
    base_md5 = {r.out_md5: r.out_path for r in base if r.out_md5}
    _, cur = _master_entries(master_dir)
    cur_paths, base_paths = set(cur), set(base_by_path)

    added = cur_paths - base_paths
    removed = base_paths - cur_paths
    unchanged, changed, unverified = 0, [], []
    for p in cur_paths & base_paths:
        bm, cm = base_by_path[p].out_md5, cur[p].get("out_md5")
        if not (bm and cm):
            unverified.append(p)      # no hash on one side: can't prove the audio is the same
        elif bm != cm:
            changed.append(p)
        else:
            unchanged += 1
    moved = [(base_md5[cur[p].get("out_md5")], p) for p in added
             if cur[p].get("out_md5") in base_md5 and base_md5[cur[p].get("out_md5")] in removed]
    moved_src = {a for a, _ in moved}
    moved_dst = {b for _, b in moved}
    pure_add = added - moved_dst
    pure_rem = removed - moved_src
    safe = not (pure_rem or changed or moved or unverified)

    log(f"plan: master vs {base_version}")
    log(f"  unchanged : {unchanged}")
    log(f"  ADDED     : {len(pure_add)}")
    log(f"  MOVED     : {len(moved)}   (renamed — breaks projects referencing the old path)")
    log(f"  CHANGED   : {len(changed)}   (same path, different audio — breaks projects)")
    log(f"  REMOVED   : {len(pure_rem)}   (breaks projects that used these)")
    if unverified:
        log(f"  UNVERIFIED: {len(unverified)}   (same path, no content hash to compare; "
            f"re-import the release or rebuild the manifest)")
    log("  => " + ("ADDITIVE — safe to re-transfer over the +Drive" if safe
                   else "NOT additive — transfer to a NEW +Drive folder, or use --base to build additively"))
    for a, b in list(moved)[:6]:
        log(f"       moved: {a}  ->  {b}")
    return dict(added=len(pure_add), moved=len(moved), changed=len(changed),
                removed=len(pure_rem), unchanged=unchanged, unverified=len(unverified), safe=safe)


def addition_ids(session, only_pack=None, since=None):
    """Sample ids an additive build may add: the ones whose library path matches only_pack
    (case-insensitive substring) and/or that were scanned on or after since (YYYY-MM-DD).
    None when neither is given (anything unreleased may be added)."""
    if not only_pack and not since:
        return None
    from datetime import datetime
    from ..db.models import Sample
    q = select(Sample.id)
    if only_pack:
        q = q.where(Sample.rel_path.ilike(f"%{only_pack}%") | Sample.path.ilike(f"%{only_pack}%"))
    if since:
        q = q.where(Sample.scanned_at >= datetime.fromisoformat(str(since)))
    return {int(i) for (i,) in session.execute(q).all()}


def _centroids(session, entries_by_family):
    """{family: unit CLAP centroid} from manifest entries (by their source samples)."""
    import numpy as np
    from ..db.models import Sample
    from .curate import load_index
    ids, emb = load_index()
    id2row = {int(i): r for r, i in enumerate(ids)}
    srcs = {e["src"] for es in entries_by_family.values() for e in es if e.get("src")}
    pid = {}
    srcs = list(srcs)
    for i in range(0, len(srcs), 900):
        pid.update(dict(session.execute(select(Sample.path, Sample.id)
                                        .where(Sample.path.in_(srcs[i:i + 900]))).all()))
    out = {}
    for fam, es in entries_by_family.items():
        rows = [id2row[pid[e["src"]]] for e in es if pid.get(e.get("src")) in id2row]
        if rows:
            v = emb[rows].astype("float32")
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            c = v.mean(0)
            out[fam] = c / (np.linalg.norm(c) + 1e-9)
    return out


def additive_build(session, base_version, out_dir, per_family=25, transcode=True,
                   describe=False, clap_z=1.0, log=print, categories=None, loudness=True,
                   only_pack=None, since=None, allowance=None):
    """Build a strict superset of `base_version`: copy the base verbatim, then add
    source samples not already in the release (per category).

    Additions are limited to `only_pack` / `since` when given
    (addition_ids), each category adds at most `allowance` of its budget
    (curate_config.ADD_ALLOWANCE), and a new group whose CLAP centroid sits close to an
    existing folder (ADD_ROUTE_MIN) joins that folder while it has room; the rest become
    new folders. verify allows a base-built master its budget plus the allowance."""
    from . import curate as _curate
    from .curate_config import ADD_ALLOWANCE, ADD_ROUTE_MIN, BUDGETS

    allowance = ADD_ALLOWANCE if allowance is None else float(allowance)
    base = read_release(session, base_version)
    if base is None:
        raise RuntimeError(f"no release {base_version} in DB; run: fourier releases import {base_version}")
    vdir = os.path.join(RELEASES_ROOT, base_version)
    if not os.path.isdir(vdir):
        raise RuntimeError(f"base version files not found at {vdir}")
    rel_root, out_real = os.path.realpath(RELEASES_ROOT), os.path.realpath(out_dir)
    if out_real == rel_root or out_real.startswith(rel_root + os.sep) \
            or rel_root.startswith(out_real + os.sep):
        raise RuntimeError(f"refusing to build into or over the releases folder ({out_dir}); "
                           f"releases are immutable. Build into the working master instead.")
    if os.path.isdir(out_dir) and os.listdir(out_dir) \
            and not os.path.exists(os.path.join(out_dir, "manifest.json")):
        raise RuntimeError(f"{out_dir} is not empty and holds no manifest.json; refusing to "
                           f"overwrite it (rsync --delete would remove its contents)")
    rsync = shutil.which("rsync")
    if not rsync:
        raise RuntimeError("rsync not found")
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([rsync, "-a", "--delete", vdir + "/", out_dir.rstrip("/") + "/"], check=True)
    log(f"copied base {base_version} verbatim -> {out_dir}")

    # "already in the release" is per category: a sample released in KICKS but rated
    # Misfiled there may be added to its next home, and PIANO/SYNTH share sources by design
    exclude = defaultdict(set)
    for r in base:
        if r.src_sample_id:
            exclude[r.category].add(r.src_sample_id)
    log(f"excluding {sum(map(len, exclude.values()))} (category, sample) pairs already in {base_version}")
    only = addition_ids(session, only_pack, since)
    if only is not None:
        from ..db.models import Sample
        others = {int(i) for (i,) in session.execute(select(Sample.id)).all()} - only
        log(f"additions limited to {len(only)} samples"
            + (f" matching {only_pack!r}" if only_pack else "") + (f" scanned since {since}" if since else ""))
    else:
        others = set()

    # start the superset manifest from the release's own manifest entries (every field the
    # rest of Fourier reads: son_cats/ab_cats, bpm, ...), falling back to the DB rows
    base_entries = {}
    bm = os.path.join(vdir, "manifest.json")
    if os.path.exists(bm):
        _, base_entries = _master_entries(vdir)
    sup_cats = defaultdict(list)
    for r in base:
        out_rel = r.out_path.split("/", 1)[1] if "/" in r.out_path else r.out_path
        e = dict(base_entries.get(r.out_path) or dict(
            family=r.family, out=out_rel, src=r.src_path, out_md5=r.out_md5, support=r.support))
        e["from_base"] = base_version
        sup_cats[r.category].append(e)

    from .curate_config import CATEGORIES
    tmp = out_dir.rstrip("/") + "__additions"
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    added_total, routed_total = 0, 0
    failed = {}                  # category -> picks it couldn't export (cli/build.py stops on any)
    for cat in (categories or list(CATEGORIES)):
        base_n = len(sup_cats.get(cat, []))
        allow = max(1, int(round(BUDGETS.get(cat, base_n) * allowance)))
        try:
            summary = _curate.build_taxonomy(session, cat, tmp, per_family=per_family, transcode=transcode,
                                             describe=describe, clap_z=clap_z,
                                             exclude_ids=exclude.get(cat, set()) | others, loudness=loudness,
                                             budget_override=allow, log=lambda m: None)
            if isinstance(summary, dict) and int(summary.get("failed") or 0):
                failed[cat] = int(summary["failed"])
        except UnsafePath:
            raise                           # cloud-only sources that won't download: stop
        except Exception as e:
            log(f"  {cat}: no additions ({e})")
            continue
        srcdir = os.path.join(tmp, cat)
        if not os.path.isdir(srcdir):
            continue
        try:
            _, tmp_entries = _master_entries(tmp)
        except Exception:
            tmp_entries = {}
        dstcat = os.path.join(out_dir, cat)
        os.makedirs(dstcat, exist_ok=True)
        # existing folders: where a close new group goes
        by_fam = defaultdict(list)
        for e in sup_cats.get(cat, []):  # type: ignore[misc]
            by_fam[e["family"]].append(e)
        try:
            base_c = _centroids(session, by_fam)
        except Exception:
            base_c = {}
        new_groups = defaultdict(list)
        for k, e in tmp_entries.items():  # type: ignore[misc]
            if k.startswith(cat + "/"):
                new_groups[e["family"]].append(e)
        new_c = _centroids(session, new_groups) if base_c else {}
        fam_meta = _read_fams(dstcat)
        tmp_meta = {f["family"]: f for f in _read_fams(srcdir)}
        cat_added = 0
        for fam in sorted(os.listdir(srcdir)):
            sp = os.path.join(srcdir, fam)
            if not os.path.isdir(sp):
                continue
            files = sorted(f for f in os.listdir(sp) if f.lower().endswith(".wav"))
            target, best = None, ADD_ROUTE_MIN
            if fam in new_c:
                for bf, bc in base_c.items():
                    room = _curate.MAX_PER_FAMILY - len(by_fam[bf])
                    sim = float(new_c[fam] @ bc)
                    if room >= len(files) and sim >= best:
                        target, best = bf, sim
            if target:
                name = target
                for f in files:
                    dst, k = f, 2
                    while os.path.exists(os.path.join(dstcat, name, dst)):
                        st, ext = os.path.splitext(f)
                        dst = f"{st[:_curate.STEM_MAX - len(str(k)) - 1]}_{k}{ext}"
                        k += 1
                    shutil.move(os.path.join(sp, f), os.path.join(dstcat, name, dst))
                    e = tmp_entries.get(f"{cat}/{fam}/{f}", {})  # type: ignore[misc]
                    ent = dict(e, family=name, out=f"{name}/{dst}", added_over=base_version)
                    sup_cats[cat].append(ent)
                    by_fam[name].append(ent)
                for m in fam_meta:
                    if m["family"] == name:
                        m["copied"] = m.get("copied", 0) + len(files)
                routed_total += len(files)
            else:
                name, k = fam, 2
                while os.path.exists(os.path.join(dstcat, name)):
                    name = f"{fam}-n{k}"; k += 1
                shutil.move(sp, os.path.join(dstcat, name))
                for f in files:
                    e = tmp_entries.get(f"{cat}/{fam}/{f}", {})  # type: ignore[misc]
                    sup_cats[cat].append(dict(e, family=name, out=f"{name}/{f}", added_over=base_version))
                meta = dict(tmp_meta.get(fam) or {"family": fam, "copied": len(files)})
                meta["family"] = name
                fam_meta.append(meta)
            cat_added += len(files)
        if fam_meta:
            with open(os.path.join(dstcat, "_manifest.json"), "w") as f:
                json.dump(fam_meta, f, indent=2)
        added_total += cat_added
        if cat_added:
            log(f"  +{cat}: {cat_added} new files (allowance {allow})")
    shutil.rmtree(tmp, ignore_errors=True)

    # write the superset manifest
    from .curate import _build_meta, _now_iso
    from .curate import _MANIFEST_VERSION
    doc = dict(fourier_manifest=_MANIFEST_VERSION, base=base_version, add_allowance=allowance)
    doc.update(_build_meta())
    doc["generated"] = _now_iso()
    doc["categories"] = {c: dict(families=len({e["family"] for e in ents}),
                                 files=len(ents), source_samples=len(ents), entries=ents)
                         for c, ents in sup_cats.items()}
    if os.path.exists(bm):
        # the base's kits and slice set are part of the release: carried over as they are
        base_sets = manifests.read(bm).get("sets")
        if base_sets:
            doc["sets"] = base_sets
    manifests.write(os.path.join(out_dir, "manifest.json"), doc)
    log(f"additive build: base {len(base)} files + {added_total} new "
        f"({routed_total} into existing folders) = superset in {out_dir}")
    return dict(base=len(base), added=added_total, routed=routed_total, failed=failed)


def _read_fams(cat_dir):
    p = os.path.join(cat_dir, "_manifest.json")
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []
