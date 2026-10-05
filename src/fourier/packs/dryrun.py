"""fourier build --dry-run: what a build would make, without writing anything.

Per category: its budget, the estimated size (AVG_FILE_MB), and, when the library's CLAP
index is there, how many samples call it home before the category's filters (an upper
bound: a category whose home count is under its budget will run short). Then the total with
the derived sets (SETS_SHARE), the storage limit size = "auto" fits (STORAGE_SHARE of the
smallest configured device) and whether a short category's surplus moves (POOL_SURPLUS).
With the analysed library's size, also what it can fill at most: each category's budget or its
home count, whichever is smaller (none where the home count is under the category's minimum,
which a build leaves empty), and no more in all than the samples there are. A library small
enough for its master to scale down (scale = "library", packs/scale.py) shows the scaled
budgets instead: each category's budget for its home samples at the library's scale factor.
"""
from __future__ import annotations

from collections import Counter


def home_counts(session, log=lambda m: None) -> dict | None:
    """{category: samples whose home it is}, or None without a CLAP index or model. An
    instrument category (PIANO, ACOUSTIC) homes nothing: its count is the candidates its own
    selection takes (curate._select_records, as a build runs it), so the plan doesn't
    promise it samples its rules leave out."""
    try:
        import numpy as np

        from .curate import compute_homes, load_index
        ids, emb = load_index()
        emb = emb.astype("float32")
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        id2row = {int(i): r for r, i in enumerate(ids)}
        homes, support, votes = compute_homes(session, emb_n, id2row, log)
    except Exception as e:           # no index yet, no CLAP model installed
        log(f"(no home counts: {e})")
        return None
    out = dict(Counter(homes.values()))
    out.update(_instrument_pools(session, emb_n, id2row, homes, support))
    return out


def _instrument_pools(session, emb_n, id2row, homes, support) -> dict:
    """{instrument category: candidates its selection keeps}, as a build selects them."""
    from .curate import (_fallback, _fetch_rows, _mirror_dups, _select_records, _wave_hashes)
    from .curate_config import CATEGORIES, CATEGORIES_OFF
    out = {}
    for cat, cfg in CATEGORIES.items():
        if cfg.get("kind") != "instrument" or cat in CATEGORIES_OFF:
            continue
        try:
            rows = _fetch_rows(session, cfg, cat)
            rec, _fm, _st = _select_records(rows, cfg, emb_n, id2row, homes, support, cat,
                                            wave_hashes=_wave_hashes(session),
                                            mirrors=_mirror_dups(session), fallback=_fallback(session))
            out[cat] = len(rec)
        except Exception:
            continue
    return out


def plan(categories, homes: dict | None = None, samples: int | None = None) -> dict:
    """The dry-run numbers for these categories. samples: the analysed samples (with CLAP
    embeddings), when known: then "picks" and "picks_mb" say what this library can fill."""
    from .curate_config import (AVG_FILE_MB, BUDGETS, POOL_SURPLUS, SETS_SHARE,
                                STORAGE_SHARE)
    rows = []
    for c in categories:
        b = BUDGETS.get(c, 0)
        rows.append(dict(category=c, budget=b, mb=b * AVG_FILE_MB.get(c, 0.5),
                         homes=None if homes is None else homes.get(c)))
    files = sum(r["budget"] for r in rows)
    mb = sum(r["mb"] for r in rows)
    from .. import knobs
    known = {"curate_config.STORAGE_SHARE": STORAGE_SHARE}
    limit = knobs.storage_limit_mb(known)
    out = dict(rows=rows, files=files, mb=mb, total_mb=mb * (1 + SETS_SHARE),
               sets_share=SETS_SHARE, limit=limit, surplus=bool(POOL_SURPLUS))
    if samples is not None:
        from ..timings import picks
        from .curate import MIN_PER_FAMILY
        from .curate_config import CATEGORIES
        from .scale import factor
        need = lambda c: int(CATEGORIES.get(c, {}).get("min_per_family", MIN_PER_FAMILY))
        # a category with a count of its own: the homed ones, and an instrument category
        # whose selection was counted (home_counts)
        homed = lambda c: CATEGORIES.get(c, {}).get("kind") in ("oneshot", "loop", "gated") or (
            homes is not None and c in homes and CATEGORIES.get(c, {}).get("kind") == "instrument")
        f = factor(samples)
        if f < 1:
            return _scaled(out, rows, samples, homes, homed, f)
        if homes is None:
            got = picks({r["category"]: r["budget"] for r in rows}, samples)
        else:
            # a homed category fills at most its home samples, none under its minimum (a build
            # leaves it empty); the others (instruments, waves) draw on the samples no
            # category calls home, likewise
            got = {}
            for r in rows:
                h = homes.get(r["category"], 0)
                if homed(r["category"]):
                    r["homes"] = h
                    got[r["category"]] = 0 if h < need(r["category"]) else min(r["budget"], h)
            spare = max(0, samples - sum(homes.values()))
            rest = picks({r["category"]: r["budget"] for r in rows if r["category"] not in got}, spare)
            got.update({c: 0 if v < need(c) else v for c, v in rest.items()})
        for r in rows:
            r["picks"] = got[r["category"]]
        out["picks"] = sum(got.values())
        out["picks_mb"] = sum(r["picks"] * AVG_FILE_MB.get(r["category"], 0.5) for r in rows) * (1 + SETS_SHARE)
    return out


def _scaled(out: dict, rows: list, samples: int, homes, homed, f: float) -> dict:
    """plan() for a library its master scales down for: each row's budget is the scaled one
    for its pool (packs/scale.py), and that's also what it fills."""
    from .curate_config import AVG_FILE_MB, SETS_SHARE
    from .scale import estimate
    est = {c: int(round(v)) for c, v in
           estimate({r["category"]: r["budget"] for r in rows}, samples, homes, homed=homed).items()}
    for r in rows:
        c = r["category"]
        r["style_budget"] = r["budget"]
        r["budget"] = r["picks"] = est[c]
        r["mb"] = est[c] * AVG_FILE_MB.get(c, 0.5)
        if homes is not None and homed(c):
            r["homes"] = homes.get(c, 0)
    files = sum(est.values())
    mb = sum(r["mb"] for r in rows)
    out.update(files=files, mb=mb, total_mb=mb * (1 + SETS_SHARE), picks=files,
               picks_mb=mb * (1 + SETS_SHARE), scale=f, surplus=False,
               categories=sum(1 for v in est.values() if v), samples=samples)
    return out
