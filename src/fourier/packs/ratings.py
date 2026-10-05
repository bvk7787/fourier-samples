"""Human ratings harvested from Ableton Live browser tags, and a per-build scorecard.

Workflow: preview the curated master in Live and tag files with a "Fourier" tag
group -- Fourier|Keep, Fourier|Drop, Fourier|Misfiled -- and optionally color them
with a Collection (Favorites is slot 1, red). Live persists both as XMP sidecars in
<folder>/Ableton Folder Info/*.xmp (ablFR:items -> filePath + colors + keywords).
A rebuild replaces the master folder and its sidecars, so:

- `harvest` copies the tags and colors into a durable store (~/.fourier/ratings.json)
  keyed by SOURCE path (resolved through the master's manifest). `fourier build`
  runs it against the live master before every build.
- A rating (or color) is cleared only when it disappears from the SAME master it was
  harvested from (you removed it in Live). A file that merely comes back untagged
  after a rebuild keeps it.
- A Favorite (Collection slot 1) on a file with no Fourier tag counts as Keep, and
  the write-back then puts a real Fourier|Keep tag on it. Explicit tags always win;
  un-favoriting later does not remove the Keep.
- `scorecard` scores any built master against the store: keep rate among rated
  files, drops a rebuild brought back, keeps a rebuild lost, misfiled still open.
- `apply_tags` writes stored ratings and colors back into a (rebuilt) master, so they
  show in Live's browser again. It preserves everything else Live stored and never
  rewrites a sidecar holding a field it does not understand.
"""
from __future__ import annotations

import glob
import json
import re
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone

TAG_GROUP = "Fourier"
PRIORITY = {"drop": 3, "misfiled": 2, "keep": 1}   # conflicting tags: the strongest wins
_NS = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
       "ablFR": "https://ns.ableton.com/xmp/fs-resources/1.0/"}
_XMP_NS = "http://ns.adobe.com/xap/1.0/"
_ITEM_FIELDS = {"filePath", "colors", "keywords"}  # per-item fields Fourier round-trips
# Live Collection slots that imply Keep on a file with no Fourier tag (slot 1 is
# Favorites, red). An explicit Fourier|... tag always wins.
FAVORITE_COLORS = {"1"}

# Live's sidecar names are fixed per sidecar type, not derived from the folder path
# (every factory pack's is c55d131f-...xmp). A user-tagged folder gets this one, so a
# sidecar written into a temp build still works after the build is moved into place.
LIVE_USER_XMP = "dc66a3fa-0fe1-5352-91cf-3ec237e9ee90.xmp"


def default_store():
    from ..paths import home_path
    return os.environ.get("FOURIER_RATINGS") or str(home_path("ratings.json"))


def default_scorecards():
    return os.path.join(os.path.dirname(default_store()), "scorecards")


def live_master_dir():
    """The master you preview in Live: $FOURIER_CURATED_DIR, else [output] master (or
    ~/Music/FourierCurated) when it holds a manifest."""
    from ..places import master_dir
    env = os.environ.get("FOURIER_CURATED_DIR")
    if env:
        return os.path.expanduser(env)
    p = master_dir()
    return p if os.path.exists(os.path.join(p, "manifest.json")) else None


# ---------------------------------------------------------------------------
# Sidecar parsing
# ---------------------------------------------------------------------------
def _bag(li, field):
    return [k.text for k in li.findall(f"ablFR:{field}/rdf:Bag/rdf:li", _NS) if k.text]


def _parse_sidecar(xmp_path):
    """({filePath: {"keywords": [...], "colors": [...]}}, CreatorTool, unknown) where
    unknown is True if any item carries a field Fourier does not round-trip."""
    items, unknown = {}, False
    try:
        root = ET.parse(xmp_path).getroot()
    except (ET.ParseError, OSError):
        return items, None, True
    ct = root.find(f".//{{{_XMP_NS}}}CreatorTool")
    for li in root.iter(f"{{{_NS['rdf']}}}li"):
        fp = li.find("ablFR:filePath", _NS)
        if fp is None or not fp.text:
            continue                                   # a keyword/color <li>, not an item
        for child in li:
            if child.tag.split("}")[-1] not in _ITEM_FIELDS:
                unknown = True
        it = items.setdefault(fp.text, {"keywords": [], "colors": []})
        it["keywords"].extend(_bag(li, "keywords"))
        it["colors"].extend(_bag(li, "colors"))
    return items, (ct.text if ct is not None else None), unknown


def read_folder_items(master_dir, unreadable=None):
    """{path relative to master: {"keywords": [...], "colors": [...]}} for every file
    Live tagged or colored. Folders whose sidecar can't be parsed are added to the
    `unreadable` set (paths relative to the master) so callers don't read them as
    'no tags'."""
    out = {}
    pattern = os.path.join(glob.escape(master_dir), "**", "Ableton Folder Info", "*.xmp")
    for x in glob.glob(pattern, recursive=True):
        folder = os.path.dirname(os.path.dirname(x))
        items, creator, _ = _parse_sidecar(x)
        if creator is None and not items and unreadable is not None and not _parses(x):
            unreadable.add(os.path.relpath(folder, master_dir))
            continue
        for fp, it in items.items():
            rel = os.path.relpath(os.path.join(folder, fp), master_dir)
            o = out.setdefault(rel, {"keywords": [], "colors": []})
            o["keywords"].extend(it["keywords"])
            o["colors"].extend(it["colors"])
    return out


def _parses(xmp_path):
    try:
        ET.parse(xmp_path)
        return True
    except (ET.ParseError, OSError):
        return False


def read_folder_tags(master_dir):
    """{path relative to master: [keyword, ...]} for every file Live tagged."""
    return {rel: it["keywords"] for rel, it in read_folder_items(master_dir).items()
            if it["keywords"]}


# "Fourier|Move-SNARES": misfiled here AND belongs in SNARES. Stored
# as a Misfiled with a target; curation pins the file in the target category.
MOVE_RE = re.compile(r"^move(?:[-_ ]to)?[-_ ]+([a-z0-9_]+)$", re.I)


def _category_named(word):
    """The current category a Move tag names (case-insensitive, renamed ones followed)."""
    from .curate_config import CATEGORIES, RENAMED_CATEGORIES
    w = (word or "").strip().upper()
    w = RENAMED_CATEGORIES.get(w, w)
    return w if w in CATEGORIES else None


def move_target(keywords):
    """The category a 'Fourier|Move-<CAT>' keyword names, else None."""
    for k in keywords or []:
        g, sep, v = k.partition("|")
        if sep and g.strip().lower() == TAG_GROUP.lower():
            m = MOVE_RE.match(v.strip())
            if m and _category_named(m.group(1)):
                return _category_named(m.group(1))
    return None


def verdict_of(keywords):
    """'keep' / 'drop' / 'misfiled' from Live keywords like 'Fourier|Keep', else None.
    A 'Fourier|Move-<CAT>' tag is a Misfiled (with a target, move_target)."""
    vs = []
    for k in keywords or []:
        g, sep, v = k.partition("|")
        v = v.strip().lower()
        if sep and g.strip().lower() == TAG_GROUP.lower():
            if v in PRIORITY:
                vs.append(v)
            elif MOVE_RE.match(v) and _category_named(MOVE_RE.match(v).group(1)):
                vs.append("misfiled")
    return max(vs, key=PRIORITY.get) if vs else None


def tag_label(r):
    """The Live keyword for a stored rating: Fourier|Keep, ..., Fourier|Move-SNARES."""
    if r.get("verdict") == "misfiled" and r.get("target"):
        return f"{TAG_GROUP}|Move-{r['target']}"
    return f"{TAG_GROUP}|{r['verdict'].capitalize()}"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def load_store(path=None):
    path = path or default_store()
    if os.path.exists(path):
        with open(path) as f:
            store = json.load(f)
    else:
        store = {"version": 1}
    store.setdefault("ratings", {})
    store.setdefault("colors", {})
    store.setdefault("auto_misfiled", {})
    # categories renamed or folded since a rating was stored (BELLS / MALLETS -> ACOUSTIC,
    # ...): a Keep or a Misfiled keeps pointing at the same folder under its new name
    from .curate_config import RENAMED_CATEGORIES
    for sect in ("ratings", "colors", "auto_misfiled"):
        for path, r in store[sect].items():
            if not isinstance(r, dict):
                continue
            new = None
            if r.get("category") in RENAMED_CATEGORIES:
                new = _renamed_category(r["category"], path, r.get("verdict"))
                r["category"] = new
            if r.get("other") in RENAMED_CATEGORIES:
                r["other"] = RENAMED_CATEGORIES[r["other"]]
            if isinstance(r.get("out"), str):
                head, _, rest = r["out"].partition("/")
                if head in RENAMED_CATEGORIES:
                    r["out"] = f"{new or RENAMED_CATEGORIES[head]}/{rest}"
    return store


def _renamed_category(category, path, verdict):
    """Where a rating in a retired category now points. MALLETS was folded into ACOUSTIC
    (acoustic tuned percussion) and SYNTH (synth mallets): a Keep on a synth or
    drum-machine pack's patch (an FM "Marimba") follows it to SYNTH."""
    from .curate_config import ORCH_PACK_EXCLUDE, RENAMED_CATEGORIES
    new = RENAMED_CATEGORIES[category]
    if category in ("MALLETS", "BELLS") and verdict == "keep":
        from .curate import _pack_of
        pack = _pack_of(None, path or "")
        if ORCH_PACK_EXCLUDE.search(pack if pack != "?" else ""):
            return "SYNTH"
    return new


def save_store(store, path=None):
    path = path or default_store()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(store, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _manifest(master_dir):
    with open(os.path.join(master_dir, "manifest.json")) as f:
        return json.load(f)


def harvest(master_dir=None, store_path=None, log=print):
    """Merge the master's Fourier|... tags and Collection colors into the store."""
    master_dir = master_dir or live_master_dir()
    if not master_dir or not os.path.exists(os.path.join(master_dir, "manifest.json")):
        log("ratings: no master with a manifest to harvest")
        return {}
    man = _manifest(master_dir)
    stamp = man.get("generated") or ""
    by_out = {}
    for cat, cd in man.get("categories", {}).items():
        for e in cd.get("entries", []):
            by_out[os.path.join(cat, e["out"])] = (cat, e)
    # a rating made on a set copy (KITS/, SLICE/) counts for the curated file it copies
    for sname, sd in (man.get("sets") or {}).items():
        for e in sd.get("entries", []):
            cat, _, out = (e.get("came_from") or "").partition("/")
            if cat and out:
                by_out[os.path.join(sname, e["out"])] = (cat, dict(e, family=out.split("/")[0], out=out))
    # the review queue (<master>/_REVIEW): map its links back to their sources
    from .review import REVIEW_DIR, review_index
    rix = review_index(master_dir) or {}
    qid = rix.get("id")
    for name, it in (rix.get("items") or {}).items():
        by_out[os.path.join(REVIEW_DIR, name)] = (
            it["category"], dict(src=it["src"], family=it.get("family"), out=it["out"]),
            qid, it.get("bucket"))
    store = load_store(store_path)
    R, C = store["ratings"], store["colors"]
    # files the library walk found moved since this master was built: a tag on the old
    # file's copy is the moved file's rating (rekey_paths)
    moved = store.get("moved") or {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen_r, seen_c = set(), set()
    s = dict(added=0, changed=0, cleared=0, unmatched=0, colored=0, from_favorites=0,
             unreadable=0)
    unreadable = set()
    items = read_folder_items(master_dir, unreadable)
    s["unreadable"] = len(unreadable)
    # gather every copy of a source's rating first (master copy + review-queue copy)
    cands = defaultdict(list)
    for rel, it in items.items():
        v = verdict_of(it["keywords"])
        colors = sorted(set(it["colors"]))
        source = "tag"
        if v is None and FAVORITE_COLORS & set(colors):
            v, source = "keep", "favorite"          # a Favorite implies Keep
        if v is None and not colors:
            continue
        hit = by_out.get(rel)
        if hit is None:
            s["unmatched"] += 1          # tagged file not in this manifest (stray/renamed)
            continue
        tgt = move_target(it["keywords"]) if v == "misfiled" else None
        cands[_moved_to(moved, hit[1]["src"])].append(dict(
            v=v, colors=colors, source=source, cat=hit[0], e=hit[1],
            target=tgt if tgt != hit[0] else None,
            hstamp=hit[2] if len(hit) > 2 else stamp, via=hit[3] if len(hit) > 3 else None,
            queue=rel.startswith(REVIEW_DIR + os.sep)))
    for src, cs in cands.items():
        old = R.get(src)
        rated = [c for c in cs if c["v"] is not None]
        if rated:
            # copies disagree (e.g. a stale queue Keep vs a new Drop on the master copy):
            # the copy that differs from the stored verdict is the new action; among those,
            # Drop > Misfiled > Keep, then the review queue (where active rating happens)
            new = [c for c in rated if old is None or c["v"] != old["verdict"]] or rated
            c = max(new, key=lambda c: (PRIORITY[c["v"]], c["queue"]))
            v, cat, e = c["v"], c["cat"], c["e"]
            seen_r.add(src)
            fresh = old is None or old["verdict"] != v or (v == "misfiled" and (
                old.get("target") != c["target"] or old.get("category") != cat))
            if old is None:
                s["added"] += 1
            elif fresh:
                s["changed"] += 1
            if fresh and c["source"] == "favorite":
                s["from_favorites"] += 1
            R[src] = dict(verdict=v, category=cat, family=e.get("family"),
                          out=os.path.join(cat, e["out"]), master=c["hstamp"],
                          rated_at=now if fresh else old.get("rated_at", now),
                          source=c["source"] if fresh else old.get("source", "tag"),
                          via=c["via"] if fresh else old.get("via"))
            if v == "misfiled" and c["target"]:
                R[src]["target"] = c["target"]
            if not fresh and old.get("hash"):
                R[src]["hash"] = old["hash"]
            # the master the verdict was given in (never re-stamped; the scorecard's "back")
            rated_in = ((rix.get("master") or stamp) if c["queue"] else c["hstamp"]) if fresh \
                else old.get("rated_in")
            if rated_in:
                R[src]["rated_in"] = rated_in
        colors = sorted({x for c in cs for x in c["colors"]})
        if colors:
            c = cs[0]
            seen_c.add(src)
            _h = (C.get(src) or {}).get("hash")
            C[src] = dict(colors=colors, category=c["cat"], family=c["e"].get("family"),
                          out=os.path.join(c["cat"], c["e"]["out"]), master=c["hstamp"],
                          **({"hash": _h} if _h else {}))
    # harvested from THIS master earlier but gone now: you removed it in Live. A folder
    # whose sidecar couldn't be read tells us nothing, so its ratings stay.
    def _unreadable(r):
        if r.get("master") == qid and REVIEW_DIR in unreadable:
            return True
        return os.path.dirname(r.get("out") or "") in unreadable
    for D, seen, key in ((R, seen_r, "cleared"), (C, seen_c, None)):
        for src, r in list(D.items()):
            if r.get("master") in ({stamp, qid} - {None}) and src not in seen \
                    and not _unreadable(r):
                del D[src]
                if key:
                    s[key] += 1
    s["colored"] = len(C)
    save_store(store, store_path)
    s["total"] = len(R)
    log(f"ratings: +{s['added']} new ({s['from_favorites']} from Favorites), {s['changed']} changed, "
        f"{s['cleared']} cleared, {s['unmatched']} unmatched; {s['total']} rated, {s['colored']} colored "
        f"in {store_path or default_store()}")
    if unreadable:
        log(f"ratings: WARNING {len(unreadable)} sidecar(s) couldn't be read, their ratings were kept "
            f"as stored: {', '.join(sorted(unreadable)[:3])}")
    return s


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
def _rated_before(r, stamp):
    """True if this rating was made in an earlier master than `stamp` (a later build had
    the chance to act on it). Uses rated_in, which harvest never re-stamps; older ratings
    without it fall back to comparing rated_at with the build time."""
    if r.get("rated_in"):
        return r["rated_in"] != stamp
    try:
        built = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        return datetime.fromisoformat(r["rated_at"].replace("Z", "+00:00")) < built
    except (KeyError, ValueError, AttributeError):
        return r.get("master") != stamp


def scorecard(master_dir, store_path=None):
    """Score a built master against the ratings store."""
    man = _manifest(master_dir)
    stamp = man.get("generated") or ""
    R = load_store(store_path)["ratings"]
    present = {e["src"]: cat for cat, cd in man.get("categories", {}).items()
               for e in cd.get("entries", [])}
    rows = {}

    def row(c):
        return rows.setdefault(c, dict(files=0, rated=0, keep=0, drop=0, misfiled=0,
                                       drops_back=0, keeps_lost=0, misfiled_moved=0))
    for cat, cd in man.get("categories", {}).items():
        row(cat)["files"] = len(cd.get("entries", []))
    for src, r in R.items():
        v, now_cat = r["verdict"], present.get(src)
        if now_cat is None:
            if v == "keep":
                row(r["category"])["keeps_lost"] += 1
            continue
        if v == "misfiled" and now_cat != r["category"]:
            row(r["category"])["misfiled_moved"] += 1      # a rebuild re-homed it
            continue
        x = row(now_cat)
        x["rated"] += 1
        x[v] += 1
        if v == "drop" and _rated_before(r, stamp):
            x["drops_back"] += 1                           # dropped earlier, rebuild kept it
    for x in rows.values():
        x["keep_rate"] = round(x["keep"] / x["rated"], 3) if x["rated"] else None
    tot = {k: sum(x[k] for x in rows.values())
           for k in ("files", "rated", "keep", "drop", "misfiled", "drops_back",
                     "keeps_lost", "misfiled_moved")}
    tot["keep_rate"] = round(tot["keep"] / tot["rated"], 3) if tot["rated"] else None
    return dict(master=stamp, master_dir=master_dir, categories=rows, totals=tot,
                scored_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def save_scorecard(sc, scorecards_dir=None):
    """Persist a scorecard (history keyed by master stamp); return the previous one."""
    d = scorecards_dir or default_scorecards()
    os.makedirs(d, exist_ok=True)
    key = "".join(ch if ch.isalnum() else "-" for ch in (sc.get("master") or "unknown"))
    prev = None
    others = sorted(p for p in glob.glob(os.path.join(d, "*.json"))
                    if os.path.basename(p) != key + ".json")
    if others:
        with open(max(others, key=os.path.getmtime)) as f:
            prev = json.load(f)
    with open(os.path.join(d, key + ".json"), "w") as f:
        json.dump(sc, f, indent=1)
    return prev


def format_scorecard(sc, prev=None):
    def pct(v):
        return "  -  " if v is None else f"{100 * v:4.0f}%"
    lines = [f"scorecard  master {sc['master']}",
             f"{'category':12s} {'files':>5s} {'rated':>5s} {'keep%':>6s} {'Δkeep':>6s} "
             f"{'drops':>5s} {'back':>4s} {'lost':>4s} {'misf':>4s}"]
    pc = (prev or {}).get("categories", {})
    for cat, x in list(sc["categories"].items()) + [("TOTAL", sc["totals"])]:
        p = pc.get(cat) if cat != "TOTAL" else (prev or {}).get("totals")
        d = ("" if (p is None or x["keep_rate"] is None or p.get("keep_rate") is None)
             else f"{100 * (x['keep_rate'] - p['keep_rate']):+5.0f}")
        if cat != "TOTAL" and not (x["rated"] or x["keeps_lost"] or x["misfiled_moved"]):
            continue
        lines.append(f"{cat:12s} {x['files']:5d} {x['rated']:5d} {pct(x['keep_rate']):>6s} {d:>6s} "
                     f"{x['drop']:5d} {x['drops_back']:4d} {x['keeps_lost']:4d} {x['misfiled']:4d}")
    lines.append("back = dropped earlier but kept by this build; lost = kept earlier, gone now; "
                 "misf = misfiled still in the same folder")
    return lines


# ---------------------------------------------------------------------------
# Write-back: re-apply stored ratings and colors as Live tags after a rebuild
# ---------------------------------------------------------------------------
def _render_xmp(items, creator):
    """Render a Live user sidecar. items: {filePath: {"keywords", "colors"}} (a bare
    list is taken as keywords)."""
    from xml.sax.saxutils import escape

    def bag(field, vals):
        if not vals:
            return ""
        lis = "".join(f"\n                        <rdf:li>{escape(v)}</rdf:li>" for v in vals)
        return (f"\n                  <ablFR:{field}>"
                f"\n                     <rdf:Bag>{lis}"
                "\n                     </rdf:Bag>"
                f"\n                  </ablFR:{field}>")
    lis = []
    for fp in sorted(items):
        it = items[fp]
        if isinstance(it, list):
            it = {"keywords": it, "colors": []}
        if not (it.get("keywords") or it.get("colors")):
            continue
        lis.append(
            '\n               <rdf:li rdf:parseType="Resource">'
            f"\n                  <ablFR:filePath>{escape(fp)}</ablFR:filePath>"
            + bag("colors", it.get("colors")) + bag("keywords", it.get("keywords")) +
            "\n               </rdf:li>")
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 6.0.0">\n'
        '   <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '      <rdf:Description rdf:about=""\n'
        '            xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
        '            xmlns:ablFR="https://ns.ableton.com/xmp/fs-resources/1.0/"\n'
        '            xmlns:xmp="http://ns.adobe.com/xap/1.0/">\n'
        '         <dc:format>application/vnd.ableton.folder</dc:format>\n'
        '         <ablFR:resource>folder</ablFR:resource>\n'
        '         <ablFR:platform>mac</ablFR:platform>\n'
        '         <ablFR:items>\n'
        f'            <rdf:Bag>{"".join(lis)}\n'
        '            </rdf:Bag>\n'
        '         </ablFR:items>\n'
        f'         <xmp:CreatorTool>{escape(creator or "Fourier ratings write-back")}</xmp:CreatorTool>\n'
        f'         <xmp:CreateDate>{now}</xmp:CreateDate>\n'
        f'         <xmp:MetadataDate>{now}</xmp:MetadataDate>\n'
        '      </rdf:Description>\n'
        '   </rdf:RDF>\n'
        '</x:xmpmeta>\n')


def apply_tags(master_dir, store_path=None, log=print):
    """Write stored ratings (Fourier|Keep etc.) and Collection colors back into a master
    so a rebuilt master shows them in Live's browser again. Only Fourier|... keywords
    and the stored files' colors change; everything else Live stored is preserved, and
    a sidecar with a field Fourier does not understand is left untouched. A 'misfiled'
    rating is not re-applied once a rebuild has moved the file to another category."""
    if not master_dir or not os.path.exists(os.path.join(master_dir, "manifest.json")):
        log("ratings: no master to apply tags to")
        return {}
    man = _manifest(master_dir)
    where = {e["src"]: (cat, e) for cat, cd in man.get("categories", {}).items()
             for e in cd.get("entries", [])}
    store = load_store(store_path)
    want = defaultdict(lambda: defaultdict(dict))   # folder -> filename -> {verdict, colors}
    for src, r in store["ratings"].items():
        hit = where.get(src)
        if hit is None:
            continue
        cat, e = hit
        if r["verdict"] == "misfiled" and cat != r.get("category"):
            continue
        fam, fn = os.path.split(e["out"])
        want[os.path.join(master_dir, cat, fam)][fn]["verdict"] = tag_label(r)
    for src, c in store["colors"].items():
        hit = where.get(src)
        if hit is None:
            continue
        cat, e = hit
        fam, fn = os.path.split(e["out"])
        want[os.path.join(master_dir, cat, fam)][fn]["colors"] = list(c["colors"])
    s = dict(files=0, folders_written=0, folders_skipped=0)
    for folder, files in want.items():
        info = os.path.join(folder, "Ableton Folder Info")
        path = os.path.join(info, LIVE_USER_XMP)
        items, creator, unknown = (_parse_sidecar(path) if os.path.exists(path)
                                   else ({}, None, False))
        if unknown:
            s["folders_skipped"] += 1
            log(f"ratings: left {path} untouched (unrecognized sidecar field)")
            continue
        changed = False
        for fn, w in files.items():
            it = items.setdefault(fn, {"keywords": [], "colors": []})
            new_kw = it["keywords"]
            if "verdict" in w:
                new_kw = [k for k in it["keywords"]
                          if k.partition("|")[0].strip().lower() != TAG_GROUP.lower()]
                new_kw = new_kw + [w["verdict"]]
            new_col = w.get("colors", it["colors"])
            s["files"] += 1
            if new_kw != it["keywords"] or new_col != it["colors"]:
                it["keywords"], it["colors"] = new_kw, new_col
                changed = True
        if changed:
            os.makedirs(info, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(_render_xmp(items, creator))
            os.replace(tmp, path)
            s["folders_written"] += 1
    log(f"ratings: applied {s['files']} files to {master_dir} "
        f"({s['folders_written']} sidecars written, {s['folders_skipped']} skipped)")
    return s


def _without_mirror_twins(m):
    """Drop a rating on a mirror-folder copy (a top-level folder of copies, curate_config
    MIRROR_ROOT_RE) when the same file name is rated outside the mirrors: the copy isn't
    curated, the original carries the rating (harvest can bring the copy's back from an
    older master's sidecars)."""
    from .curate_config import MIRROR_ROOT_RE
    from ..places import library_rel as rel
    outside = {os.path.basename(k).lower() for k in m if not MIRROR_ROOT_RE.search(rel(k))}
    return {k: v for k, v in m.items()
            if not (MIRROR_ROOT_RE.search(rel(k)) and os.path.basename(k).lower() in outside)}


def keep_pins(store_path=None):
    """{source path: category} for every file rated Keep (incl. Favorites). Curation
    always selects these, in the category they were rated in."""
    try:
        R = load_store(store_path)["ratings"]
    except (OSError, ValueError):
        return {}
    pins = {src: r["category"] for src, r in R.items() if r.get("verdict") == "keep"}
    # a Move-<CAT> pins the file in its target (and stays Misfiled where it was rated)
    pins.update({src: r["target"] for src, r in R.items()
                 if r.get("verdict") == "misfiled" and r.get("target")})
    return _without_mirror_twins(pins)


def move_targets(store_path=None):
    """{source path: target category} for every Move-<CAT> rating."""
    try:
        R = load_store(store_path)["ratings"]
    except (OSError, ValueError):
        return {}
    return {src: r["target"] for src, r in R.items()
            if r.get("verdict") == "misfiled" and r.get("target")}


def rekey_by_hash(session, store_path=None, log=print):
    """Ratings survive a library reorganisation. Every stored
    rating / color / auto-misfile records its source's file hash (Sample.file_hash); one
    whose source path is gone from the library moves to the one existing file with that
    hash (the same file name first). Returns {"hashed", "moved", "orphaned"}."""
    from sqlalchemy import select
    from ..db.models import Sample
    store = load_store(store_path)
    by_path, by_hash = {}, defaultdict(list)
    for p, h in session.execute(select(Sample.path, Sample.file_hash)).all():
        by_path[p] = h
        if h:
            by_hash[h].append(p)
    s = dict(hashed=0, moved=0, orphaned=0)
    for sect in ("ratings", "colors", "auto_misfiled"):
        D = store[sect]
        for src, r in list(D.items()):
            if not isinstance(r, dict):
                continue
            if src in by_path and os.path.exists(src):
                if by_path[src] and r.get("hash") != by_path[src]:
                    r["hash"] = by_path[src]
                    s["hashed"] += 1
                continue
            cands = [p for p in by_hash.get(r.get("hash"), []) if p != src and os.path.exists(p)]
            same = [p for p in cands if os.path.basename(p).lower() == os.path.basename(src).lower()]
            pick = same if same else cands
            if len(pick) == 1 and pick[0] not in D:
                D[pick[0]] = D.pop(src)
                s["moved"] += 1
            elif src not in by_path:
                s["orphaned"] += 1
    save_store(store, store_path)
    if s["hashed"] or s["moved"] or s["orphaned"]:
        log(f"ratings: {s['hashed']} hashed, {s['moved']} followed a moved file, "
            f"{s['orphaned']} point at files no longer in the library")
    return s


def rekey_paths(moves: dict, store_path=None) -> int:
    """Move the stored ratings, colors and automatic misfiles of these sources to their new
    paths ({old path: new path}: files the library walk found moved, the same content at a
    new path). A new path that already holds its own entry keeps it. Each move of a rated
    file is remembered (the store's "moved"), so a harvest of a master built before the move
    gives a tag on the old file's copy to the new path. Returns how many entries moved."""
    if not moves:
        return 0
    path = store_path or default_store()
    if not os.path.exists(path):
        return 0
    store = load_store(store_path)
    n = 0
    remember = store.setdefault("moved", {})
    for sect in ("ratings", "colors", "auto_misfiled"):
        D = store[sect]
        for old, new in moves.items():
            if old in D and new not in D and old != new:
                D[new] = D.pop(old)
                n += 1
                remember.pop(new, None)                  # a file is there now
                for k, v in list(remember.items()):      # moved twice: the latest place
                    if v == old:
                        remember[k] = new
                remember[old] = new
    for k in [k for k, v in remember.items() if k == v]:
        del remember[k]
    if n:
        save_store(store, store_path)
    return n


def _moved_to(moved: dict, src: str) -> str:
    """Where a source the walk found moved is now (moved: {old: new}), else the source."""
    seen = set()
    while src in moved and src not in seen:
        seen.add(src)
        src = moved[src]
    return src


def clear_ratings(master_dir, srcs, store_path=None) -> int:
    """Remove the stored ratings of these sources, and their Fourier|... tags from the
    master's sidecars (so a harvest doesn't bring them back). Returns how many were stored."""
    store = load_store(store_path)
    R = store["ratings"]
    n = sum(1 for src in srcs if R.pop(src, None) is not None)
    save_store(store, store_path)
    if not master_dir or not os.path.exists(os.path.join(master_dir, "manifest.json")):
        return n
    man = _manifest(master_dir)
    by_folder = defaultdict(set)
    for cat, cd in man.get("categories", {}).items():
        for e in cd.get("entries", []):
            if e.get("src") in srcs:
                fam, fn = os.path.split(e["out"])
                by_folder[os.path.join(master_dir, cat, fam)].add(fn)
    for folder, names in by_folder.items():
        path = os.path.join(folder, "Ableton Folder Info", LIVE_USER_XMP)
        if not os.path.exists(path):
            continue
        items, creator, unknown = _parse_sidecar(path)
        if unknown:
            continue
        changed = False
        for fn in names:
            it = items.get(fn)
            if it is None:
                continue
            kw = [k for k in it["keywords"] if k.partition("|")[0].strip().lower() != TAG_GROUP.lower()]
            if kw != it["keywords"]:
                it["keywords"], changed = kw, True
        if changed:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                f.write(_render_xmp(items, creator))
            os.replace(tmp, path)
    return n


def stored_matching(query, store_path=None) -> list:
    """[(source path, rating)] of the stored ratings whose source path (or its library path)
    holds the query, as find_rateable matches the master's files: an exact path, else a
    fragment (case, spaces and underscores aside)."""
    R = load_store(store_path)["ratings"]
    q = (query or "").strip()
    if not q:
        return []
    absq = os.path.abspath(os.path.expanduser(q))
    exact = [(src, r) for src, r in R.items() if src in (q, absq)]
    if exact:
        return exact
    nq = _norm(q)
    return [(src, r) for src, r in sorted(R.items())
            if nq in _norm(src) or nq in _norm(_short_src(src))]


def missing_keeps(session=None, store_path=None, pins=None) -> list:
    """[(source path, category)] of the Keep pins (keep_pins(), or these) whose source is
    gone: not on disk, or marked missing by the library walk. A build can't place them, so
    it and verify pass over them with a warning (`fourier review rate <name> clear` removes
    the rating)."""
    pins = keep_pins(store_path) if pins is None else pins
    gone = [(src, c) for src, c in sorted(pins.items()) if not os.path.lexists(src)]
    if session is not None:
        try:
            from sqlalchemy import text
            marked = {p for (p,) in session.execute(text("SELECT path FROM missing_files"))}
        except Exception:
            marked = set()
        have = {src for src, _ in gone}
        gone += [(src, c) for src, c in sorted(pins.items()) if src in marked and src not in have]
    return gone


def drop_set(store_path=None):
    """Source paths rated Drop. Curation leaves them out of every category (a Drop is a
    verdict on the sound, not its folder). Exact file only: siblings aren't dropped."""
    try:
        R = load_store(store_path)["ratings"]
    except (OSError, ValueError):
        return set()
    return set(_without_mirror_twins({src: 1 for src, r in R.items() if r.get("verdict") == "drop"}))


def misfiled_map(store_path=None):
    """{source path: category} for every file rated Misfiled, plus automatic
    high-confidence detections the human has not rated. Curation never puts the file
    back in that category; compute_homes re-homes it to its next-best candidate."""
    try:
        store = load_store(store_path)
    except (OSError, ValueError):
        return {}
    R = store["ratings"]
    m = {src: r["category"] for src, r in R.items() if r.get("verdict") == "misfiled"}
    # automatic detections (review.auto_misfile) apply unless a human rated the file
    for src, a in store.get("auto_misfiled", {}).items():
        if src not in R:
            m.setdefault(src, a["category"])
    return _without_mirror_twins(m)


# ---------------------------------------------------------------------------
# Ratings without Live: `fourier review rate` and `fourier review import`
# ---------------------------------------------------------------------------
VERDICTS = ("keep", "drop", "misfiled")
# the `master` stamp of a rating given on the command line or from a CSV file: harvest clears
# a rating only when it disappears from the master it was harvested from, which this never is
MANUAL_STAMP = "manual"


def _short_src(src):
    """A source's path under its library folder, else its last three parts."""
    try:
        from ..places import library_rel
        rel = library_rel(src)
    except Exception:  # noqa: BLE001 (no library configured: the tail of the path)
        rel = src
    return rel if rel != src else "/".join(src.replace("\\", "/").split("/")[-3:])


def rateable(master_dir):
    """Every file in the master a rating can name: [{src, category, family, out, path, also,
    short}], where path is CATEGORY/family/file.wav, also lists its copies (KITS/, SLICE/,
    the review queue's _REVIEW/ links) and short is its library path."""
    man = _manifest(master_dir)
    out, by_path = [], {}
    for cat, cd in man.get("categories", {}).items():
        for e in cd.get("entries", []):
            t = dict(src=e["src"], category=cat, family=e.get("family"), out=e["out"],
                     path=f"{cat}/{e['out']}", also=[], short=_short_src(e["src"]))
            out.append(t)
            by_path[t["path"]] = t
    for sname, sd in (man.get("sets") or {}).items():
        for e in sd.get("entries", []):
            t = by_path.get(e.get("came_from") or "")
            if t:
                t["also"].append(f"{sname}/{e['out']}")
    from .review import REVIEW_DIR, review_index
    for name, it in ((review_index(master_dir) or {}).get("items") or {}).items():
        t = by_path.get(f"{it.get('category')}/{it.get('out')}")
        if t:
            t["also"].append(f"{REVIEW_DIR}/{name}")
    return out


def _norm(s):
    """Lowercase, spaces and underscores alike: "Kick 01" finds "Kick_01.wav"."""
    return re.sub(r"[\s_]+", " ", (s or "").replace("\\", "/")).strip().lower()


def find_rateable(master_dir, query, targets=None):
    """The master's files a query names, as `fourier why` takes one: an exact source path,
    master path (CATEGORY/family/file.wav, a set copy or a review-queue link) or a file
    inside the master; else every file whose master path or library path holds the query
    as a fragment (case, spaces and underscores aside)."""
    ts = rateable(master_dir) if targets is None else targets
    q = (query or "").strip()
    if not q:
        return []
    md = os.path.abspath(master_dir)
    absq = os.path.abspath(os.path.expanduser(q))
    inside = os.path.relpath(absq, md).replace(os.sep, "/") if absq.startswith(md + os.sep) else None
    names = {q.replace("\\", "/"), inside} - {None}
    exact = [t for t in ts if q == t["src"] or absq == t["src"]
             or names & {t["path"], *t["also"]}]
    if exact:
        return exact
    nq = _norm(q)
    return [t for t in ts if nq in _norm(t["path"]) or nq in _norm(t["short"])
            or any(nq in _norm(a) for a in t["also"])]


def set_ratings(master_dir, rated, store_path=None, source="cli"):
    """Store ratings given without Live: rated is [(target, verdict, move_to)] with targets
    from rateable(). Each is stored as a harvested tag would be (keyed by source path, with
    its category, family and master path), so builds, `review score`, `queue` and
    `misfiles` read it the same way; its `master` is MANUAL_STAMP, so a harvest never
    clears it. move_to (misfiled only) is the category the file belongs in."""
    stamp = _manifest(master_dir).get("generated") or ""
    store = load_store(store_path)
    R = store["ratings"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for t, verdict, move_to in rated:
        v = (verdict or "").strip().lower()
        if v not in VERDICTS:
            raise ValueError(f"rating {verdict!r}: one of {', '.join(VERDICTS)}")
        tgt = None
        if move_to:
            if v != "misfiled":
                raise ValueError(f"a category to move to goes with misfiled, not {v}")
            tgt = _category_named(move_to)
            if tgt is None:
                raise ValueError(f"no category {move_to!r}")
        old = R.get(t["src"]) or {}
        R[t["src"]] = dict(verdict=v, category=t["category"], family=t.get("family"),
                           out=os.path.join(t["category"], t["out"]), master=MANUAL_STAMP,
                           rated_at=now, source=source, via=None, rated_in=stamp)
        if tgt and tgt != t["category"]:
            R[t["src"]]["target"] = tgt
        if old.get("hash"):
            R[t["src"]]["hash"] = old["hash"]
    save_store(store, store_path)
    return len(rated)


CSV_NAME = ("path", "name", "file")
CSV_RATING = ("rating", "verdict")
CSV_CATEGORY = ("category", "to", "move_to")


def read_ratings_csv(csv_path, master_dir):
    """Rows of a ratings CSV matched to the master's files: (rated, skipped) where rated is
    [(target, verdict, move_to)] and skipped [(line, value, why)]. Columns (a header row,
    any order, case aside): `path` or `name` (what `review rate` takes), `rating` (keep,
    drop or misfiled) and an optional `category` (misfiled: the category it belongs in)."""
    import csv
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {(c or "").strip().lower(): c for c in (reader.fieldnames or [])}
        name_col = next((cols[c] for c in CSV_NAME if c in cols), None)
        rating_col = next((cols[c] for c in CSV_RATING if c in cols), None)
        cat_col = next((cols[c] for c in CSV_CATEGORY if c in cols), None)
        if not name_col or not rating_col:
            raise ValueError(f"{csv_path}: needs a header row with a path (or name) column and a "
                             f"rating column; found {reader.fieldnames}")
        rows = [(i, r) for i, r in enumerate(reader, start=2)]
    targets = rateable(master_dir)
    rated, skipped, seen = [], [], {}
    for line, r in rows:
        q = (r.get(name_col) or "").strip()
        v = (r.get(rating_col) or "").strip().lower()
        to = (r.get(cat_col) or "").strip() if cat_col else ""
        if not q and not v:
            continue
        if v not in VERDICTS:
            skipped.append((line, q, f"rating {v or '(empty)'!r}: one of {', '.join(VERDICTS)}"))
            continue
        if to and v != "misfiled":
            skipped.append((line, q, f"a category goes with misfiled, not {v}"))
            continue
        if to and _category_named(to) is None:
            skipped.append((line, q, f"no category {to!r}"))
            continue
        hits = find_rateable(master_dir, q, targets)
        if not hits:
            skipped.append((line, q, "no file in the master matches"))
            continue
        if len(hits) > 1:
            skipped.append((line, q, f"matches {len(hits)} files ({', '.join(h['path'] for h in hits[:3])}"
                                     f"{', ...' if len(hits) > 3 else ''}); give its path"))
            continue
        key = (hits[0]["src"], hits[0]["category"])
        if key in seen:                       # a later row for the same file wins
            rated[seen[key]] = (hits[0], v, to or None)
        else:
            seen[key] = len(rated)
            rated.append((hits[0], v, to or None))
    return rated, skipped


def uses_live(master_dir):
    """True when the master holds Live sidecars (someone rates it in Live's browser)."""
    pattern = os.path.join(glob.escape(master_dir), "**", "Ableton Folder Info", "*.xmp")
    return next(iter(glob.iglob(pattern, recursive=True)), None) is not None
