"""Review queue: a small folder of the samples whose rating would teach Fourier the most.

`fourier review queue` harvests existing ratings, writes them back into the master, and
rebuilds <master>/_REVIEW/ with ~40 hard links named "CATEGORY - family - file.wav".
You preview and tag them in Live exactly like the master (Fourier|Keep / Drop /
Misfiled, or a red Favorite); `ratings.harvest` maps them back to their source through
_REVIEW/_review.json. Renders skip "_" folders, so the queue never reaches a device.

Buckets (why a file was picked; kept out of the filename so it can't bias the call):
- misfile  sounds closer to another category's centroid than to its own (CLAP)
- shaky    placed on one classifier's vote, or on votes that disagree, in categories
           where both classifiers have vocabulary (for ACOUSTIC/STABS/... single-vote is
           structural, not a signal)
- neighbor nearest unrated sound to something rated Drop (same category)
- scattered one note from a multisample set whose notes span families (drum / tonal /
           vocal / fx), shown where it sits in a drum folder when it has one; a
           Misfiled verdict there clears the whole set from that folder
- random   stratified random picks, so keep rates stay honest
"""
from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

from .curate_config import ADDED_CATEGORIES as _ADDED_CATEGORIES

REVIEW_DIR = "_REVIEW"
REVIEW_INDEX = "_review.json"
SHARES = {"misfile": 0.25, "shaky": 0.25, "neighbor": 0.15, "scattered": 0.15}  # rest -> random
FAMILY = {c: "drum" for c in ("KICKS", "SNARES", "HATS", "CLAPS", "TOMS", "CYMBALS", "PERC",
                              "DRUMLOOPS")}
FAMILY.update({c: "tonal" for c in ("SUB", "SYNTH", "PADS", "STABS", "BLIPS", "PIANO",
                                    "ACOUSTIC", "PHRASES")})
FAMILY.update({"VOX": "vocal", "FX": "fx", "SCRATCHES": "fx"})
# ...and each category an overlay adds, by its entry's family
FAMILY.update({c: e.get("family") or "other" for c, e in _ADDED_CATEGORIES.items() if c not in FAMILY})
STRUCTURAL_SINGLE_VOTE = 0.60   # skip "shaky" in categories where >60% are single-vote


def review_index(master_dir):
    p = os.path.join(master_dir, REVIEW_DIR, REVIEW_INDEX)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def detect_misfiles(entries, vecs):
    """Entries that sound closer to another category's CLAP centroid than to their own:
    [{entry, category, other, margin, detail}], largest margin (other - own) first."""
    by_cat = defaultdict(list)
    for e in entries:
        by_cat[e["category"]].append(e)
    cents = {}
    for c, es in by_cat.items():
        V = [vecs[e["src"]] for e in es if e["src"] in vecs]
        if V:
            m = np.mean(V, axis=0)
            cents[c] = m / (np.linalg.norm(m) + 1e-9)
    out = []
    if len(cents) < 2:
        return out
    for e in entries:
        v = vecs.get(e["src"])
        if v is None or e["category"] not in cents:
            continue
        own = float(v @ cents[e["category"]])
        other_c, other = max(((c, float(v @ m)) for c, m in cents.items() if c != e["category"]),
                             key=lambda t: t[1])
        if other > own:
            out.append(dict(entry=e, category=e["category"], other=other_c, margin=other - own,
                            detail=f"closer to {other_c} ({other:.2f} vs {own:.2f})"))
    out.sort(key=lambda d: (-d["margin"], d["entry"]["src"]))
    return out


AUTO_MISFILE_MARGIN = 0.4   # a conservative margin


def _manifest_entries(master_dir):
    with open(os.path.join(master_dir, "manifest.json")) as f:
        man = json.load(f)
    entries = [dict(src=e["src"], category=cat, family=e.get("family"), out=e["out"],
                    support=e.get("support"), son_cats=e.get("son_cats"), ab_cats=e.get("ab_cats"))
               for cat, cd in man.get("categories", {}).items() for e in cd.get("entries", [])]
    return man, entries


def auto_misfile(master_dir, margin=AUTO_MISFILE_MARGIN, store_path=None, dry_run=False,
                 clear=False, vecs=None, log=print):
    """Record high-confidence misfile detections (margin > `margin`) in the ratings
    store's `auto_misfiled` map, kept apart from human ratings (a human rating on the
    file always wins; `clear` removes them all). The next build treats them like a
    Misfiled rating: excluded from that category (with multisample siblings) and
    re-homed."""
    from .ratings import load_store, save_store
    store = load_store(store_path)
    A = store["auto_misfiled"]
    if clear:
        n = len(A)
        A.clear()
        save_store(store, store_path)
        log(f"misfiles: cleared {n}")
        return []
    man, entries = _manifest_entries(master_dir)
    if vecs is None:
        vecs = _load_vectors([e["src"] for e in entries])
    R = store["ratings"]
    hits = [d for d in detect_misfiles(entries, vecs)
            if d["margin"] > margin and d["entry"]["src"] not in R]
    new = [d for d in hits if d["entry"]["src"] not in A]
    for d in hits:
        log(f"  {d['margin']:.2f}  {d['category']:10s} -> {d['other']:10s} "
            f"{os.path.basename(d['entry']['src'])}")
    if not dry_run:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for d in hits:
            A.setdefault(d["entry"]["src"], dict(category=d["category"], other=d["other"],
                                                 margin=round(d["margin"], 3),
                                                 master=man.get("generated"), at=now))
        save_store(store, store_path)
    log(f"misfiles: {len(hits)} above margin {margin} ({len(new)} new)"
        + (" [dry run]" if dry_run else f"; {len(A)} stored"))
    return hits


def select_queue(entries, vecs, ratings, n=40, seed=0, cat_cap=None, fam_cap=2):
    """Pick up to n review items. entries: [{src, category, family, out, support,
    son_cats, ab_cats}], vecs: {src: unit CLAP vector}, ratings: {src: {verdict,...}}.
    Returns [(entry, bucket, detail)] in pick order. Pure and deterministic per seed."""
    from .curate import SIBLING_COS, _sibling_key
    rng = random.Random(seed)
    cat_cap = cat_cap or max(3, -(-n // 6))
    # rating one note of a multisample rates the set: skip its siblings
    rated_keys = defaultdict(list)
    for s in ratings:
        k = _sibling_key(s)
        if k is not None:
            rated_keys[k].append(vecs.get(s))

    def sib_of_rated(e):
        rk = rated_keys.get(_sibling_key(e["src"]))
        if not rk:
            return False
        v = vecs.get(e["src"])
        return any(x is None or v is None or float(v @ x) >= SIBLING_COS for x in rk)
    todo = [e for e in entries if e["src"] not in ratings and not sib_of_rated(e)]
    todo_set = {id(e) for e in todo}
    by_cat = defaultdict(list)
    for e in entries:
        by_cat[e["category"]].append(e)

    # --- misfile suspects: own-centroid cosine vs best other centroid ---
    misfile = sorted(((-d["margin"], d["entry"], d["detail"]) for d in detect_misfiles(entries, vecs)
                      if id(d["entry"]) in todo_set), key=lambda t: (t[0], t[1]["src"]))

    # --- shaky placements ---
    single = {c: sum(1 for e in es if e.get("support") == 1) / len(es) for c, es in by_cat.items()}
    shaky = []
    for e in todo:
        if e.get("support") != 1 or single.get(e["category"], 1) > STRUCTURAL_SINGLE_VOTE:
            continue
        son, ab = set(e.get("son_cats") or []), set(e.get("ab_cats") or [])
        conflict = bool(son and ab and not (son & ab))
        shaky.append((0 if conflict else 1, rng.random(), e,
                      f"classifiers disagree: sononym {sorted(son)} / ableton {sorted(ab)}"
                      if conflict else "placed on one classifier's vote"))
    shaky.sort(key=lambda t: (t[0], t[1]))

    # --- neighbors of Drops ---
    neighbor = []
    drops = [s for s, r in ratings.items() if r.get("verdict") == "drop" and s in vecs]
    for s in sorted(drops):
        cat = ratings[s].get("category")
        cands = [e for e in todo if e["category"] == cat and e["src"] in vecs]
        cands.sort(key=lambda e: (-float(vecs[e["src"]] @ vecs[s]), e["src"]))
        for e in cands[:3]:
            neighbor.append((e, f"near Drop {os.path.basename(s)} "
                                f"({float(vecs[e['src']] @ vecs[s]):.2f})"))

    # --- scattered multisample sets: notes of one set across families ---
    by_set = defaultdict(list)
    for e in entries:
        k = _sibling_key(e["src"])
        if k is not None:
            by_set[k].append(e)
    scattered = []
    for k, es in by_set.items():
        fams = {FAMILY.get(e["category"], "other") for e in es}
        if len(fams) < 2:
            continue
        avail = [e for e in es if id(e) in todo_set]
        if not avail:
            continue
        drum = [e for e in avail if FAMILY.get(e["category"]) == "drum"]
        if drum:
            rep = min(drum, key=lambda e: e["src"])
        else:                                    # no drum member: show the minority family
            fc = Counter(FAMILY.get(e["category"], "other") for e in es)
            rep = min(avail, key=lambda e: (fc[FAMILY.get(e["category"], "other")], e["src"]))
        spread = dict(Counter(e["category"] for e in es))
        scattered.append((-len(spread), -len(es), rep["src"], rep,
                          f"set '{k[1]}' spans {len(spread)} folders: {spread}"))
    scattered.sort(key=lambda t: t[:3])

    # --- random, stratified by category size ---
    pool = sorted(todo, key=lambda e: e["src"])
    rng.shuffle(pool)

    picked, seen, cat_n, fam_n = [], set(), Counter(), Counter()

    def take(e, bucket, detail):
        if e["src"] in seen or cat_n[e["category"]] >= cat_cap \
                or fam_n[(e["category"], e["family"])] >= fam_cap:
            return False
        picked.append((e, bucket, detail))
        seen.add(e["src"]); cat_n[e["category"]] += 1; fam_n[(e["category"], e["family"])] += 1
        return True

    quota = {b: int(round(n * s)) for b, s in SHARES.items()}
    for b, items in (("misfile", [(e, d) for _, e, d in misfile]),
                     ("shaky", [(e, d) for _, _, e, d in shaky]),
                     ("neighbor", neighbor),
                     ("scattered", [(e, d) for *_, e, d in scattered])):
        got = 0
        for e, d in items:
            if got >= quota[b] or len(picked) >= n:
                break
            got += take(e, b, d)
    for e in pool:                      # random fills the rest (and any unused quota)
        if len(picked) >= n:
            break
        take(e, "random", "random spot check")
    return picked


def _load_vectors(srcs):
    """{src: unit CLAP vector} for the given source paths."""
    from sqlalchemy import select
    from ..db.models import Sample
    from ..db.session import session_scope
    from .curate import load_index
    ids, emb = load_index()
    row = {int(i): r for r, i in enumerate(ids)}
    want = set(srcs)
    out = {}
    with session_scope() as s:
        for sid, path in s.execute(select(Sample.id, Sample.path)):
            if path in want and int(sid) in row:
                v = emb[row[int(sid)]].astype("float32")
                out[path] = v / (np.linalg.norm(v) + 1e-9)
    return out


def build_queue(master_dir, n=40, store_path=None, seed=None, log=print):
    """Rebuild <master>/_REVIEW with n items. Call after harvest + apply_tags so the
    previous queue's tags are safe in the store and on the master copies."""
    from .ratings import load_store
    man, entries = _manifest_entries(master_dir)
    ratings = load_store(store_path)["ratings"]
    seed = len(ratings) if seed is None else seed
    vecs = _load_vectors([e["src"] for e in entries] + list(ratings))
    items = select_queue(entries, vecs, ratings, n=n, seed=seed)

    qdir = os.path.join(master_dir, REVIEW_DIR)
    final_qdir, qdir = qdir, qdir + ".new"    # build aside; the old queue stays until done
    if os.path.isdir(qdir):
        shutil.rmtree(qdir)
    os.makedirs(qdir)
    qid = "review-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    index = {"id": qid, "master": man.get("generated"), "seed": seed, "items": {}}
    for e, bucket, detail in items:
        src_file = os.path.join(master_dir, e["category"], e["out"])
        fam, fn = os.path.split(e["out"])
        name = f"{e['category']} - {fam} - {fn}"
        dst = os.path.join(qdir, name)
        try:
            os.link(src_file, dst)
        except OSError:
            shutil.copy2(src_file, dst)
        index["items"][name] = dict(src=e["src"], category=e["category"], family=e["family"],
                                    out=e["out"], bucket=bucket, detail=detail)
    with open(os.path.join(qdir, REVIEW_INDEX), "w") as f:
        json.dump(index, f, indent=1)
    if os.path.isdir(final_qdir):
        shutil.rmtree(final_qdir)
    os.replace(qdir, final_qdir)
    qdir = final_qdir
    b = Counter(bk for _, bk, _ in items)
    c = Counter(e["category"] for e, _, _ in items)
    log(f"review: {len(items)} files in {qdir}")
    log("  why:  " + ", ".join(f"{k} {v}" for k, v in b.most_common()))
    log("  from: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    return index
