"""How long builds and analysis take, measured, so doctor and build --dry-run can estimate.

Every whole build writes its per-category seconds, files and processed-audio cache hits, and
analyze writes each step's seconds per sample, to <home>/timings.json. Estimates use the
last measurement on this machine when there is one, else DEFAULTS (rough defaults for a
cold-cache full rebuild and the analysis steps' rates).
"""
from __future__ import annotations

import json
import os
import time

from .paths import home_path

# Rough defaults for a cold-cache full rebuild: seconds per file per worker for
# each category, and the rest of a --all build beyond its longest category or its share of
# the work (homes, sets, ratings, verify). The processed-audio cache barely changes a
# build's time. Analysis: seconds per sample per worker.
DEFAULTS = {
    "build_s_per_file": {
        "ACOUSTIC": 0.064,
        "BLIPS": 0.024,
        "CLAPS": 0.02,
        "CYMBALS": 0.034,
        "DRUMLOOPS": 0.109,
        "FX": 0.072,
        "HATS": 0.035,
        "KICKS": 0.038,
        "PADS": 0.087,
        "PERC": 0.028,
        "PHRASES": 0.052,
        "PIANO": 0.083,
        "SNARES": 0.047,
        "STABS": 0.029,
        "SUB": 0.062,
        "SYNTH": 0.077,
        "TOMS": 0.035,
        "VOX": 0.037,
        "WAVES": 0.007,
    },
    "build_s_per_file_other": 0.061,
    "build_fixed_s": 149.0,
    # the rest of a build (homes, sets, verify) grows with the analysed library: the fixed
    # part above is for one this size, and scales down for a smaller one, to no less than
    # what any build spends (loading the CLAP model and embedding its prompts, nineteen
    # categories' set-up, the sets and verify)
    "build_fixed_samples": 200_000,
    "build_fixed_min_s": 45.0,
    "analyze_s_per_sample": {"derived": 0.0001, "librosa": 0.073, "clap": 0.042,
                             "events": 0.02, "quality": 0.01, "key": 0.08, "loop-trim": 0.06,
                             # pYIN on a tonal one-shot's first 3 s (the own step's candidates)
                             "own": 0.5},
    # what an analysis spends however few samples it has: starting its workers, compiling
    # the audio code, loading the CLAP model
    "analyze_fixed_s": 45.0,
}
DEFAULTS_SOURCE = "rough defaults"
KEEP = 5


def path():
    return home_path("timings.json")


def load() -> dict:
    try:
        return json.loads(path().read_text())
    except (OSError, ValueError):
        return {}


def _save(doc: dict) -> None:
    p = path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
        os.replace(tmp, p)
    except OSError:
        pass                                   # timings are a nicety, never a failure


def record_build(results, jobs: int, wall_s: float) -> None:
    """A whole build's times: results as the build returns them, (category, summary, entries)."""
    cats = {c: {"seconds": round(float(s.get("seconds", 0)), 1), "files": int(s.get("files", 0)),
                "cache_hits": s.get("cache_hits")}
            for c, s, _e in results if s is not None and s.get("seconds") is not None}
    if not cats:
        return
    doc = load()
    doc.setdefault("builds", []).append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), jobs=jobs,
                                             wall=round(wall_s, 1), categories=cats))
    doc["builds"] = doc["builds"][-KEEP:]
    _save(doc)


def record_analyze(step: str, seconds: float, samples: int, workers: int) -> None:
    if samples <= 0:
        return
    doc = load()
    doc.setdefault("analyze", {})[step] = dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                              s_per_sample=seconds * max(workers, 1) / samples,
                                              samples=samples, workers=workers)
    _save(doc)


def _wall(work: list, jobs: int, fixed: float) -> float:
    """A --all build's wall time: its longest category or its share of the work, whichever
    is longer, plus the rest of the build."""
    return max(max(work, default=0.0), sum(work) / jobs) + fixed


def picks(budgets: dict, samples: int) -> dict:
    """The files a build can pick per category: the budgets, scaled down (in their
    proportions) when the analysed library has fewer samples than they add up to, since
    each sample lands in one category at most."""
    total = sum(budgets.values())
    if samples >= total or total <= 0:
        return dict(budgets)
    return {c: b * samples / total for c, b in budgets.items()}


def estimate_build(budgets: dict, jobs: int, samples: int | None = None) -> tuple[float, str]:
    """(seconds, what it's based on) for a whole build of these budgets with jobs workers.
    samples: the analysed library's size, when known; the rough defaults' fixed part
    scales with it."""
    jobs = max(1, jobs)
    last = (load().get("builds") or [None])[-1]
    if last:
        cats = last["categories"]
        per_file = {c: v["seconds"] / v["files"] for c, v in cats.items() if v.get("files")}
        mean = sum(v["seconds"] for v in cats.values()) / max(1, sum(v["files"] for v in cats.values()))
        seen = [v["seconds"] for v in cats.values()]
        fixed = max(0.0, last["wall"] - _wall(seen, max(1, last["jobs"]), 0.0))
        work = [b * per_file.get(c, mean) for c, b in budgets.items()]
        return _wall(work, jobs, fixed), f"this machine's last build ({last['at'][:10]})"
    rates = DEFAULTS["build_s_per_file"]
    work = [b * rates.get(c, DEFAULTS["build_s_per_file_other"]) for c, b in budgets.items()]
    fixed = DEFAULTS["build_fixed_s"]
    if samples is not None:
        fixed = max(DEFAULTS["build_fixed_min_s"],
                    fixed * min(1.0, samples / DEFAULTS["build_fixed_samples"]))
    return _wall(work, jobs, fixed), DEFAULTS_SOURCE


ANALYZE_STEPS = tuple(DEFAULTS["analyze_s_per_sample"])     # what a first analysis runs
# the share of a library the own step reads (tonal one-shots: metadata/resolve.py
# ROOT_CANDIDATE_SQL), for an estimate before there is a database to count them in
OWN_SHARE = 0.3


def estimate_analyze(pending: dict, workers: int) -> tuple[float, str]:
    """(seconds, basis) to analyze the samples each step still has to do ({step: count}),
    with the start-up any analysis spends (analyze_fixed_s)."""
    measured = load().get("analyze") or {}
    total = DEFAULTS["analyze_fixed_s"] if any(pending.values()) else 0.0
    own = 0
    for step, n in pending.items():
        rate = (measured.get(step) or {}).get("s_per_sample")
        own += rate is not None
        rate = rate if rate is not None else DEFAULTS["analyze_s_per_sample"].get(step, 0.05)
        total += n * rate / (1 if step == "clap" else max(1, workers))
    basis = "this machine's last runs" if own == len(pending) and pending else \
        ("this machine's runs and " if own else "") + DEFAULTS_SOURCE
    return total, basis


def human(seconds: float) -> str:
    m = seconds / 60
    if m < 1:
        return "under a minute"
    return f"about {m:.0f} min" if m < 90 else f"about {m / 60:.1f} h"
