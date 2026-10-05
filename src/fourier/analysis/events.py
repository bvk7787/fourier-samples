"""Sample-chain detection: how many separate sound events a file holds.

A one-shot is one event. Sample chains are several: a velocity ladder (the same snare
hit 48 times, soft to loud), a note ladder (a multisample's notes laid end to end),
round robins. Their shared signature is loud events separated by real silence.

An event is a run of audio above FLOOR_DB (relative to the file's peak); a silence of
at least GAP_S ends it. Events quieter than LOUD_DB are ignored (a faint tail or a
click in the noise floor is not a second note). Silence-only splitting is deliberately
conservative: a snare roll, a flam or a pad with a dip never splits, because their
level never falls to silence between hits.

Delay/echo tails look like a chain too, so the profile also records whether every
event is quieter than the one before (`echo`), and how evenly the events are spaced
(`regularity`: coefficient of variation of the start-to-start times; ~0 for a chain
laid out on a fixed grid, larger for a performed phrase).
"""
from __future__ import annotations

import numpy as np

HOP_S = 0.005
FLOOR_DB = -45.0
GAP_S = 0.040
MIN_EVENT_S = 0.025
LOUD_DB = -30.0
ECHO_STEP_DB = 1.5


def envelope_db(x: np.ndarray, sr: int, hop_s: float = HOP_S) -> np.ndarray:
    """Peak envelope in dB relative to the file's peak, one value per hop."""
    if x.ndim == 2:
        x = np.abs(x).max(axis=1)
    else:
        x = np.abs(x)
    h = max(1, int(sr * hop_s))
    n = len(x) // h
    if n == 0:
        return np.zeros(0, dtype="float32")
    env = x[: n * h].reshape(n, h).max(axis=1)
    return (20 * np.log10(env / (float(env.max()) + 1e-12) + 1e-9)).astype("float32")


def find_events(db: np.ndarray, hop_s: float = HOP_S, floor_db: float = FLOOR_DB,
                gap_s: float = GAP_S, min_event_s: float = MIN_EVENT_S,
                loud_db: float = LOUD_DB) -> list[tuple[float, float, float]]:
    """(start_s, end_s, peak_db) for each loud event separated by silence."""
    if len(db) == 0:
        return []
    gap_f = max(1, int(round(gap_s / hop_s)))
    active = db > floor_db
    segs, start, quiet = [], None, 0
    for k, a in enumerate(active):
        if a:
            if start is None:
                start = k
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap_f:
                segs.append((start, k - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        segs.append((start, len(db)))
    out = []
    for a, b in segs:
        pk = float(db[a:b].max())
        if (b - a) * hop_s >= min_event_s and pk > loud_db:
            out.append((a * hop_s, b * hop_s, pk))
    return out


def profile(events: list[tuple[float, float, float]]) -> dict:
    """n_events, regularity (start spacing CV; None under 3 events), echo (bool)."""
    n = len(events)
    peaks = [p for _, _, p in events]
    echo = n >= 2 and all(peaks[i + 1] < peaks[i] - ECHO_STEP_DB for i in range(n - 1))
    reg = None
    if n >= 3:
        d = np.diff([s for s, _, _ in events])
        reg = float(d.std() / (d.mean() + 1e-9))
    return {"n_events": n, "event_regularity": reg, "event_echo": int(echo)}


def event_profile(path: str) -> dict | None:
    """Read a file and profile its events; None if it can't be read."""
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    return profile(find_events(envelope_db(x, sr)))
