"""Build validator: invariant checks over a curated master (+ optional DB), so
iterating on budgets/gates/naming can't silently regress. `fourier verify --quick`
runs these and exits non-zero on any FAIL.

Checks (per category, from manifest.json):
  budget    files must not exceed the category budget (FAIL); a reachable
            shortfall (families x ceiling >= budget) is a WARN.
  ceiling   no family folder may exceed MAX_PER_FAMILY (FAIL).
  floor     a folder below MIN_PER_FAMILY is a WARN (tiny cluster).
  vendor    no single source folder may exceed VENDOR_MAX_SHARE (+tolerance) of a
            category's files (WARN; pool cap != output cap exactly), in a category from
            three vendors or more (the build caps only those).
  names     family folder names unique within a category (FAIL).
  dup       a source sample, or identical audio under another name, appearing in >1
            category (WARN; instrument categories may overlap by design).
  hygiene   (DB) no clipped or >60s sample leaked into the output (FAIL).
  override  no NAME_OVERRIDES rule makes up > OVERRIDE_MAX_SHARE (+tol) of its target
            (WARN; pool cap != output cap exactly).
  pack      no vendor/pack makes up > INSTRUMENT_PACK_MAX_SHARE (+tol) of an
            instrument category (WARN).
"""
from __future__ import annotations

import os
from collections import Counter
from ..places import library_rel


def _vendor_of(src):
    """The vendor the build capped a source under (packs/vendors.py), or None (no cap)."""
    from .vendors import FIRST_FOLDER, mode, vendor_of
    if mode() == FIRST_FOLDER:
        seg = library_rel(src or "").split("/")
        return seg[0] if seg and seg[0] else "?"
    return vendor_of(library_rel(src or ""), src)


def validate_master(master_dir, budgets=None, floor=6, ceil=None, vendor_max=None,
                    vendor_tol=0.15, share_tol=0.05, session=None, log=print):
    """Return (ok, results). results is a list of (level, check, detail)."""
    from .curate_config import BUDGETS, VENDOR_MAX_SHARE, CATEGORIES
    budgets = BUDGETS if budgets is None else budgets
    vendor_max = VENDOR_MAX_SHARE if vendor_max is None else vendor_max
    if ceil is None:              # the build's own ceiling
        from .curate import MAX_PER_FAMILY as ceil

    mp = os.path.join(master_dir, "manifest.json")
    if not os.path.exists(mp):
        log(f"  [FAIL] manifest  no manifest.json at {master_dir}")
        return False, [("FAIL", "manifest", master_dir)]
    from .vendors import ensure
    ensure(session)               # the library folders' layouts (vendors = "auto" only)
    from . import manifests
    _doc = manifests.read(mp)
    cats = _doc.get("categories", {})
    if _doc.get("base") and budgets is BUDGETS:
        # an additive build: its base release plus the add allowance (releases.additive_build)
        from .curate_config import ADD_ALLOWANCE
        _a = float(_doc.get("add_allowance", ADD_ALLOWANCE))
        budgets = {c: b + max(1, int(round(b * _a))) for c, b in BUDGETS.items()}
    # a category rebuilt with a surplus (size = "auto"), or scaled to a small library (scale =
    # "library"), records its budget
    budgets = {**budgets, **{c: cd["budget"] for c, cd in cats.items() if "budget" in cd}}
    if _doc.get("scale"):        # a library-scaled master's folders may hold fewer files
        floor = min(floor, int(_doc["scale"].get("folder_min_files", floor)))
    res = []
    add = lambda lvl, chk, d: res.append((lvl, chk, d))

    seen_src = {}
    for cat, cd in cats.items():
        ents = cd.get("entries", [])
        files = cd.get("files", len(ents))
        fams = cd.get("families", len({e.get("family") for e in ents}))

        b = budgets.get(cat)
        if b is not None:
            if files > b:
                add("FAIL", "budget", f"{cat}: {files} exceeds budget {b}")
            elif files < b and fams * ceil >= b:
                add("WARN", "budget", f"{cat}: {files} < budget {b} (reachable: {fams} fams x {ceil})")

        kind = CATEGORIES.get(cat, {}).get("kind")
        per = Counter(e.get("family") for e in ents)
        # APFS and FAT32/exFAT cards fold case: "Dark" and "dark" would merge on disk
        folded = Counter((f or "").lower() for f in per)
        for f, n in folded.items():
            if n > 1:
                add("FAIL", "names", f"{cat}: {n} family folders differ only by case ({f!r})")
        under = 0
        for fam, n in per.items():
            if n > ceil:
                add("FAIL", "ceiling", f"{cat}/{fam}: {n} > {ceil}")
            elif n < floor:
                under += 1
        # floor is only actionable as systemic over-splitting; niche pools
        # (instrument / gated) inherently have a few tiny clusters, so exempt them
        if kind not in ("instrument", "gated") and under >= max(3, round(0.3 * max(len(per), 1))):
            add("WARN", "floor", f"{cat}: {under}/{len(per)} folders < {floor} (over-split; raise kdiv)")

        # vendor cap only runs for non-instrument kinds (_cap_vendor_share); instrument
        # categories are pack-scoped by design, so a single-vendor share is expected
        # (nor for a category from fewer than three vendors: the build doesn't cap those)
        vendors = Counter(v for v in (_vendor_of(e.get("src")) for e in ents) if v is not None)
        if files and kind != "instrument" and not CATEGORIES.get(cat, {}).get("no_vendor_cap") \
                and len(vendors) >= 3:
            top, n = vendors.most_common(1)[0]
            if n / files > vendor_max + vendor_tol:
                add("WARN", "vendor", f"{cat}: {top} {n/files:.0%} > {vendor_max:.0%}+{vendor_tol:.0%}")
        # override share + instrument per-pack share. Enforced on the candidate pool
        # during selection; pool share tracks output share, not exactly -> WARN.
        if files:
            from .curate import _name_override_rule, _NAME_OVERRIDES
            from .vendors import pack_key
            from .curate_config import OVERRIDE_MAX_SHARE, INSTRUMENT_PACK_MAX_SHARE
            rc = Counter()
            for e in ents:
                ri, rcat = _name_override_rule(os.path.basename(e.get("src") or ""))
                if ri is not None and rcat == cat:
                    rc[ri] += 1
            for ri, n in rc.items():
                if n / files > OVERRIDE_MAX_SHARE + share_tol:
                    add("WARN", "override", f"{cat}: override '{_NAME_OVERRIDES[ri][0][0]}' "
                        f"{n/files:.0%} > {OVERRIDE_MAX_SHARE:.0%}+{share_tol:.0%}")
            packs = Counter(k for k in (pack_key(e.get("src")) for e in ents) if k is not None)
            # (not on a library-scaled master: a small library's instruments come from a pack or two)
            if kind == "instrument" and packs and not _doc.get("scale"):
                top, n = packs.most_common(1)[0]
                if n / files > INSTRUMENT_PACK_MAX_SHARE + share_tol:
                    add("WARN", "pack", f"{cat}: {top} {n/files:.0%} > "
                        f"{INSTRUMENT_PACK_MAX_SHARE:.0%}+{share_tol:.0%}")

        # single-home applies only to the home-managed (non-instrument) categories;
        # instrument ones (PIANO clap_source, ACOUSTIC) legitimately share sources
        if kind != "instrument":
            for e in ents:
                s = e.get("src")
                if s:
                    if s in seen_src and seen_src[s] != cat:
                        add("WARN", "dup", f"{os.path.basename(s)}: {seen_src[s]} & {cat}")
                    seen_src[s] = cat

    # the same audio under different names (one sample shipped in two packs) in two
    # categories: compute_homes gives byte-identical copies one home, so this is a leak
    seen_md5 = {}
    for cat, cd in cats.items():
        if CATEGORIES.get(cat, {}).get("kind") == "instrument":
            continue
        for e in cd.get("entries", []):
            m = e.get("out_md5")
            if m:
                if m in seen_md5 and seen_md5[m][0] != cat:
                    add("WARN", "dup", f"identical audio: {seen_md5[m][1]} & {cat}/{e.get('out')}")
                seen_md5.setdefault(m, (cat, f"{cat}/{e.get('out')}"))

    if session is not None:
        try:
            from sqlalchemy import text
            srcs = [e.get("src") for cd in cats.values() for e in cd.get("entries", []) if e.get("src")]
            bad_clip = bad_dur = 0
            for k in range(0, len(srcs), 500):
                chunk = srcs[k:k + 500]
                ph = ",".join(f":p{i}" for i in range(len(chunk)))
                params = {f"p{i}": chunk[i] for i in range(len(chunk))}
                for dur, clip in session.execute(text(
                        f"SELECT s.duration_s, sf.is_clipped FROM samples s "
                        f"LEFT JOIN sample_features sf ON sf.sample_id = s.id "
                        f"WHERE s.path IN ({ph})").bindparams(**params)).all():
                    if clip:
                        bad_clip += 1
                    if dur and dur > 60:
                        bad_dur += 1
            if bad_clip:
                add("FAIL", "hygiene", f"{bad_clip} clipped samples in output")
            if bad_dur:
                add("FAIL", "hygiene", f"{bad_dur} samples > 60s in output")
        except Exception as e:
            add("WARN", "hygiene", f"DB check skipped: {e}")

    for lvl, chk, d in res:
        log(f"  [{lvl}] {chk:<8} {d}")
    fails = sum(1 for r in res if r[0] == "FAIL")
    warns = sum(1 for r in res if r[0] == "WARN")
    total = sum(cd.get("files", 0) for cd in cats.values())
    log(f"validate: {'PASS' if fails == 0 else 'FAIL'} -- {fails} fail, {warns} warn; "
        f"{len(cats)} categories, {total} files")
    return (fails == 0, res)
