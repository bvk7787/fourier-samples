"""Sample-chain detection (analysis/events.py) and the curation chain guard."""
import os
import sqlite3
import tempfile

import numpy as np

from fourier.analysis.events import envelope_db, find_events, profile
from fourier.packs import curate as C

SR = 48000


def _hit(amp=1.0, dur=0.25, decay=18.0, seed=0):
    t = np.arange(int(SR * dur)) / SR
    rng = np.random.default_rng(seed)
    return (amp * np.exp(-decay * t) * (0.6 * np.sin(2 * np.pi * 180 * t)
                                        + 0.4 * rng.standard_normal(len(t)))).astype("float32")


def _silence(s):
    return np.zeros(int(SR * s), dtype="float32")


def _prof(x):
    return profile(find_events(envelope_db(x, SR)))


def test_single_hit_is_one_event():
    assert _prof(np.concatenate([_hit(), _silence(0.3)]))["n_events"] == 1


def test_velocity_ladder_is_a_regular_chain():
    # a file of snare hits, soft to loud, sold as one "one-shot"
    x = np.concatenate([np.concatenate([_hit(a, seed=i), _silence(0.25)])
                        for i, a in enumerate(np.linspace(0.3, 1.0, 8))])
    p = _prof(x)
    assert p["n_events"] == 8 and not p["event_echo"]
    assert p["event_regularity"] < 0.05


def test_note_ladder_counts_every_note():
    notes = []
    for i, f in enumerate([110, 131, 147, 165]):
        t = np.arange(int(SR * 0.6)) / SR
        notes += [(0.8 * np.exp(-4 * t) * np.sin(2 * np.pi * f * t)).astype("float32"), _silence(0.2)]
    assert _prof(np.concatenate(notes))["n_events"] == 4


def test_echo_tail_is_flagged_as_echo():
    x = np.concatenate([np.concatenate([_hit(a), _silence(0.15)]) for a in (1.0, 0.6, 0.35, 0.2)])
    p = _prof(x)
    assert p["n_events"] == 4 and p["event_echo"] == 1


def test_roll_without_silence_is_one_event():
    # hits every 60 ms never decay to silence: a roll/flam, not a chain
    x = np.concatenate([_hit(0.9, dur=0.06, decay=8.0, seed=i) for i in range(12)] + [_silence(0.3)])
    assert _prof(x)["n_events"] == 1


def test_faint_blip_after_the_hit_is_not_an_event():
    x = np.concatenate([_hit(), _silence(0.2), _hit(0.01), _silence(0.2)])   # -40 dB
    assert _prof(x)["n_events"] == 1


def test_chain_guard_modes():
    assert C._chain_guard_mode("SNARES", {}) == "strict"
    assert C._chain_guard_mode("VOX", {}) == "regular"
    assert C._chain_guard_mode("DRUMLOOPS", {}) is None
    assert C._chain_guard_mode("BLIPS", {}) is None
    assert C._chain_guard_mode("SNARES", {"chain_guard": False}) is None
    s, r = "strict", "regular"
    assert C._is_sample_chain(s, 2, None, 0)
    assert not C._is_sample_chain(s, 1, None, 0)
    assert not C._is_sample_chain(s, 5, 0.0, 1)            # echo tail
    assert not C._is_sample_chain(s, None, None, None)     # not profiled yet
    assert C._is_sample_chain(r, 11, 0.1, 0)               # evenly spaced chain
    assert not C._is_sample_chain(r, 11, 0.8, 0)           # performed phrase
    assert not C._is_sample_chain(r, 2, None, 0)


def test_old_database_gets_new_columns():
    from fourier.db import session as sess
    path = tempfile.mktemp(suffix=".db")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE sample_features (id INTEGER PRIMARY KEY, sample_id INTEGER)")
    con.commit(); con.close()
    sess._engine = sess._SessionLocal = None
    try:
        sess.init_db(path)
        cols = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(sample_features)")}
        assert {"n_events", "event_regularity", "event_echo", "events_computed_at"} <= cols
    finally:
        sess._engine = sess._SessionLocal = None
        os.unlink(path)
