"""Library scale: the master's size follows the analyzed library (scale = "library", the
default; scale = "off" builds the budgets whatever the library's size).

    f = min(1, samples / (LIBRARY_PER_MASTER x budgets))

samples: the samples a build can place (those with a CLAP embedding, as doctor and `build
--dry-run` count them); budgets: the total of the categories a build makes (BUDGETS less
CATEGORIES_OFF, after the categories, files and size knobs). A library with at least
LIBRARY_PER_MASTER usable samples for each file the budgets add up to gets f = 1, and its
build is exactly what it was before scaling existed. Below 1, per category
(curate.build_taxonomy, its `scale` argument), with n its usable pool after the filters:

    budget   min(budget, n, max(round(budget x f), keep(n)))
    keep(n)  min(n, SCALED_KEEP_ALL) + round(max(0, n - SCALED_KEEP_ALL) / LIBRARY_PER_MASTER)
    minimum  SCALED_MIN_FILES usable files build a category (the gates' and caps' floors too)
    folders  round(budget / SCALED_FOLDER_FILES), 1 to the category's kmax, each with at least
             SCALED_FOLDER_MIN_FILES (a smaller folder merges into its nearest of the same band)

So a small pool keeps nearly all its usable files, a large one about one in LIBRARY_PER_MASTER
or its scaled budget, whichever is more, and the master tracks the library rather than a
fraction of a fraction of it. There's no surplus round (size = "auto") in a scaled build: the
scaled budgets already follow the pools. An additive build (--base) never scales.
"""
from __future__ import annotations

import math


def live_budgets() -> dict:
    """{category: budget} for the categories a build makes (the categories knob's off ones
    left out)."""
    from .curate_config import BUDGETS, CATEGORIES_OFF
    return {c: b for c, b in BUDGETS.items() if c not in CATEGORIES_OFF}


def factor(samples: int | None, total: int | None = None) -> float:
    """The library scale factor f in (0, 1]: 1.0 with scale = "off", for an unknown or empty
    library, and for a library of at least LIBRARY_PER_MASTER x total samples (total: the
    budgets' total, by default that of every category a build makes)."""
    from .curate_config import LIBRARY_PER_MASTER, LIBRARY_SCALE
    if not LIBRARY_SCALE or not samples or samples <= 0 or LIBRARY_PER_MASTER <= 0:
        return 1.0
    total = sum(live_budgets().values()) if total is None else total
    if total <= 0:
        return 1.0
    return min(1.0, samples / (LIBRARY_PER_MASTER * total))


def keep(n: float) -> int:
    """What a scaled category keeps of a usable pool of n at least: all of it up to
    SCALED_KEEP_ALL, then one in LIBRARY_PER_MASTER."""
    from .curate_config import LIBRARY_PER_MASTER, SCALED_KEEP_ALL
    n = max(0, int(round(n)))
    return min(n, SCALED_KEEP_ALL) + int(round(max(0, n - SCALED_KEEP_ALL) / max(LIBRARY_PER_MASTER, 1)))


def category_budget(budget: int, f: float, pool: float) -> int:
    """A category's budget at scale f with a usable pool of `pool` (the budget itself at f = 1)."""
    if f >= 1:
        return int(budget)
    pool = max(0, int(round(pool)))
    return int(min(budget, pool, max(int(round(budget * f)), keep(pool))))


def family_count(budget: int, pool: int, kmax: int) -> int:
    """A scaled category's folder count: about one per SCALED_FOLDER_FILES files, at least one,
    at most kmax, and none that couldn't hold SCALED_FOLDER_MIN_FILES."""
    from .curate_config import SCALED_FOLDER_FILES, SCALED_FOLDER_MIN_FILES
    files = max(1, min(int(budget), int(pool)))
    k = int(round(files / max(SCALED_FOLDER_FILES, 1)))
    return max(1, min(k, int(kmax), files // max(SCALED_FOLDER_MIN_FILES, 1)))


def tempo_band_min(pool: int, files: float) -> int:
    """The candidates a scaled tempo band needs to keep a folder of its own: enough that its
    share of the files is SCALED_FOLDER_MIN_FILES (TEMPO_BAND_MIN at most)."""
    from .curate_config import SCALED_FOLDER_MIN_FILES, TEMPO_BAND_MIN
    need = math.ceil(SCALED_FOLDER_MIN_FILES * max(pool, 1) / max(files, 1))
    return int(min(TEMPO_BAND_MIN, max(SCALED_FOLDER_MIN_FILES, need)))


def estimate(budgets: dict, samples: int, homes: dict | None = None, homed=None,
             total: int | None = None) -> dict:
    """{category: files} a scaled build of this library would make, about. With the samples'
    homes (homed categories; the others share the samples no category calls home, by budget):
    each category's scaled budget for its pool, none under SCALED_MIN_FILES. Without them a
    category's pool is its budget's share of the library's samples, an expected size (a
    fraction, and never counted out for being under the minimum on average). total: the
    budgets' total the factor is taken against (default: every category a build makes)."""
    from .curate_config import LIBRARY_PER_MASTER, SCALED_KEEP_ALL, SCALED_MIN_FILES
    live = live_budgets()
    total = sum(live.values()) if total is None else total
    f = factor(samples, total)
    if homes is None:
        out = {}
        for c, b in budgets.items():
            pool = b * samples / max(total, 1)
            kept = min(pool, SCALED_KEEP_ALL) + max(0.0, pool - SCALED_KEEP_ALL) / max(LIBRARY_PER_MASTER, 1)
            out[c] = b if f >= 1 else min(b, pool, max(b * f, kept))
        return out
    homed = homed or (lambda c: False)
    rest = {c: b for c, b in budgets.items() if not homed(c)}
    spare = max(0, samples - sum(homes.values()))
    pools = {c: homes.get(c, 0) if homed(c) else spare * b / max(sum(rest.values()), 1)
             for c, b in budgets.items()}
    return {c: 0 if pools[c] < SCALED_MIN_FILES else category_budget(b, f, pools[c])
            for c, b in budgets.items()}


def describe(samples: int, files: float, categories: int | None = None, f: float = 1.0) -> str:
    """One line on the master's size for this library: scaled ("Your library has 41 usable
    samples: the master will hold up to about 30 files in 9 categories") or at the budgets.
    Scaled, it is an upper bound: the build's near-duplicate prune and its gates, which an
    estimate can't run, take some out."""
    from .curate_config import LIBRARY_PER_MASTER, LIBRARY_SCALE
    if f < 1:
        cats = f" in up to {categories} categor{'y' if categories == 1 else 'ies'}" if categories is not None else ""
        return (f"Your library has {samples:,} usable samples: the master will hold up to about "
                f"{round(files):,} files{cats}, fewer once near-duplicates and the categories' "
                f"gates are taken out (scale = \"library\": it grows with the library up to the "
                f"style's budgets, reached at {LIBRARY_PER_MASTER} usable samples a file)")
    why = "scale = \"off\"" if not LIBRARY_SCALE else f"{samples:,} usable samples, enough for them"
    return f"The style's budgets: about {round(files):,} files ({why})"
