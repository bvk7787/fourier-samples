"""What changed between two builds of the curated master.

Every --all build archives its manifest (~/.fourier/builds/<stamp>.json, next to the
ratings store) and writes CHANGELOG.md into the master: folders renamed, split, merged,
new or gone, and files added, removed or moved between categories. `fourier diff`
compares any two builds (a master folder, a manifest file, or an archived stamp).

A folder in the new build is the same folder as an old one when at least MATCH_SHARE
of its files came from it; a different name then is a rename.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path

from . import manifests

MATCH_SHARE = 0.5


def builds_dir() -> Path:
    from .ratings import default_store
    return Path(os.path.dirname(default_store())) / "builds"


def archived() -> list[Path]:
    """This library's archived builds, oldest first: the manifests in builds_dir() less those
    whose sources all sit under none of the configured library folders (another config's
    library sharing this home). One with no sources, or a config with no library, counts."""
    arch = sorted(builds_dir().glob("*.json")) if builds_dir().exists() else []
    return [p for p in arch if _of_this_library(p)]


def latest_archived() -> Path | None:
    """This library's latest archived build (archived()), or None."""
    arch = sorted(builds_dir().glob("*.json")) if builds_dir().exists() else []
    for p in reversed(arch):            # the newest first: usually the one
        if _of_this_library(p):
            return p
    return None


_SAMPLED = 50          # sources an archived build is judged by


def _of_this_library(p: Path) -> bool:
    from .. import places
    try:
        roots, names = places.library()
        if not roots and not names:
            return True
        doc = manifests.read(p)
    except Exception:
        return True
    srcs = [e.get("src") for c in (doc.get("categories") or {}).values()
            for e in (c.get("entries") or [])[:_SAMPLED] if e.get("src")][:_SAMPLED]
    if not srcs:
        return True
    return any(places.in_library(x) or places.library_root_of(x) is not None for x in srcs)


def load_manifest(ref) -> dict:
    """A manifest from a master folder, a manifest .json, or an archived build stamp."""
    p = Path(os.path.expanduser(str(ref)))
    if p.is_dir():
        p = p / "manifest.json"
    elif not p.exists():
        cand = builds_dir() / (str(ref) if str(ref).endswith(".json") else f"{ref}.json")
        if cand.exists():
            p = cand
    return manifests.read(p)


def _families(doc) -> dict:
    """{category: {family: set(src)}}"""
    out = defaultdict(lambda: defaultdict(set))
    for cat, cd in (doc.get("categories") or {}).items():
        for e in cd.get("entries", []):
            out[cat][e["family"]].add(e["src"])
    return out


def _entries_by_src(doc, cat) -> dict:
    return {e["src"]: e for e in (((doc.get("categories") or {}).get(cat) or {}).get("entries") or ())}


def _library_moves(old, new, cat, added, removed) -> list:
    """[(old src, new src)] in a category: a source that left and one that came, at the same
    master path with the same audio (out_md5), one to one."""
    if not added or not removed:
        return []
    eo, en = _entries_by_src(old, cat), _entries_by_src(new, cat)
    by_out = defaultdict(list)
    for src in removed:
        e = eo.get(src) or {}
        if e.get("out") and e.get("out_md5"):
            by_out[(e["out"], e["out_md5"])].append(src)
    out = []
    for src in sorted(added):
        e = en.get(src) or {}
        olds = by_out.get((e.get("out"), e.get("out_md5")))
        if olds and len(olds) == 1:
            out.append((olds.pop(), src))
    return out


def diff_manifests(old: dict, new: dict) -> dict:
    fo, fn = _families(old), _families(new)
    where_old = {s: c for c, fams in fo.items() for ss in fams.values() for s in ss}
    where_new = {s: c for c, fams in fn.items() for ss in fams.values() for s in ss}
    cats = {}
    for cat in sorted(set(fo) | set(fn)):
        o, n = fo.get(cat, {}), fn.get(cat, {})
        # a file the library moved (another source path, the same file at the same place in
        # the master, its audio unchanged): neither added nor removed, and its old folder is
        # matched by where it is now (a renamed library folder renames no master folder)
        so0 = set().union(*o.values()) if o else set()
        sn0 = set().union(*n.values()) if n else set()
        lib_moved = _library_moves(old, new, cat, sn0 - so0, so0 - sn0)
        if lib_moved:
            now = dict(lib_moved)
            o = {fam: {now.get(x, x) for x in srcs} for fam, srcs in o.items()}
        renamed, same, new_f, matched_old = [], [], [], set()
        for fam, srcs in sorted(n.items()):
            best, share = None, 0.0
            for ofam, osrcs in o.items():
                sh = len(srcs & osrcs) / max(1, len(srcs))
                if sh > share:
                    best, share = ofam, sh
            if best is not None and share >= MATCH_SHARE:
                matched_old.add(best)
                (same if best == fam else renamed).append((best, fam, round(share, 2)))
            else:
                new_f.append((fam, len(srcs)))
        gone = [(f, len(s)) for f, s in sorted(o.items()) if f not in matched_old]
        so = set().union(*o.values()) if o else set()
        sn = set().union(*n.values()) if n else set()
        added, removed = sn - so, so - sn
        cats[cat] = dict(
            files_old=len(so), files_new=len(sn), folders_old=len(o), folders_new=len(n),
            renamed=renamed, new_folders=new_f, gone_folders=gone,
            added=len(added), removed=len(removed),
            moved_in=sorted(Counter(where_old[s] for s in added if s in where_old).items()),
            moved_out=sorted(Counter(where_new[s] for s in removed if s in where_new).items()),
            added_examples=sorted(os.path.basename(s) for s in added)[:5],
            removed_examples=sorted(os.path.basename(s) for s in removed)[:5],
            library_moved=len(lib_moved),
        )
    return dict(old=dict(generated=old.get("generated"), git_sha=old.get("git_sha"),
                         version=old.get("fourier_version")),
                new=dict(generated=new.get("generated"), git_sha=new.get("git_sha"),
                         version=new.get("fourier_version")), categories=cats)


def built_by(b: dict) -> str:
    """What made a build: its git sha in a checkout, else the version that built it."""
    if b.get("git_sha"):
        return b["git_sha"]
    return f"version {b['version']}" if b.get("version") else "an earlier version"


def format_diff(d: dict) -> str:
    """The diff as a CHANGELOG-style markdown page."""
    o, n = d["old"], d["new"]
    lines = ["# What changed in this build", "",
             f"From the build of {o.get('generated') or '?'} ({built_by(o)}) "
             f"to {n.get('generated') or '?'} ({built_by(n)}).", ""]
    tot_add = sum(c["added"] for c in d["categories"].values())
    tot_rem = sum(c["removed"] for c in d["categories"].values())
    tot_ren = sum(len(c["renamed"]) for c in d["categories"].values())
    tot_mv = sum(c.get("library_moved", 0) for c in d["categories"].values())
    if tot_mv and not (tot_add or tot_rem or tot_ren):
        lines += [f"{tot_mv} file{'' if tot_mv == 1 else 's'} moved in the library, master unchanged.", ""]
    else:
        lines += [f"{tot_add} files added, {tot_rem} removed, {tot_ren} folders renamed"
                  + (f", {tot_mv} moved in the library (the same files in the master)" if tot_mv else "")
                  + ".", ""]
    for cat, c in d["categories"].items():
        if not (c["added"] or c["removed"] or c["renamed"] or c["new_folders"] or c["gone_folders"]):
            continue
        lines.append(f"## {cat}: {c['files_old']} -> {c['files_new']} files, "
                     f"{c['folders_old']} -> {c['folders_new']} folders")
        for a, b, sh in c["renamed"]:
            lines.append(f"- renamed: {a} -> {b} ({int(sh * 100)}% of its files carried over)")
        for f, k in c["new_folders"]:
            lines.append(f"- new folder: {f} ({k} files)")
        for f, k in c["gone_folders"]:
            lines.append(f"- folder gone: {f} (had {k} files)")
        if c["added"] or c["removed"]:
            mv_in = ", ".join(f"{k} from {cc}" for cc, k in c["moved_in"])
            mv_out = ", ".join(f"{k} to {cc}" for cc, k in c["moved_out"])
            lines.append(f"- files: +{c['added']}" + (f" ({mv_in})" if mv_in else "")
                         + f", -{c['removed']}" + (f" ({mv_out})" if mv_out else ""))
        lines.append("")
    return "\n".join(lines)


def write_changelog(out_dir, log=print) -> Path | None:
    """Write CHANGELOG.md into out_dir, against the latest archived build."""
    out = Path(out_dir)
    mp = out / "manifest.json"
    prev = latest_archived()
    if not mp.exists() or prev is None:
        return None
    try:
        d = diff_manifests(manifests.read(prev), manifests.read(mp))
        (out / "CHANGELOG.md").write_text(format_diff(d))
        return out / "CHANGELOG.md"
    except Exception as e:                         # a changelog never fails a build
        log(f"changelog skipped: {e}")
        return None


def archive_build(out_dir) -> Path | None:
    """Keep this build's manifest in builds_dir() (named by its build time)."""
    mp = Path(out_dir) / "manifest.json"
    if not mp.exists():
        return None
    new = manifests.read(mp)
    bd = builds_dir()
    bd.mkdir(parents=True, exist_ok=True)
    stamp = (new.get("generated") or "build").replace(":", "").replace("-", "")
    dest = bd / f"{stamp}.json"
    dest.write_text(mp.read_text())
    return dest
