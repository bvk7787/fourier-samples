"""Small build fixes: a parallel build logs each category's lines, the why log records the
cross-category dedupe, levels read wide stereo as it plays, single cycles reach the ceiling,
a phase-fixed one-shot keeps its peak, and verify checks the sets' out_md5."""
import concurrent.futures as cf
import contextlib
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fourier.packs import curate as C
from fourier.packs import why_log


class _Scope:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _worker_env(monkeypatch):
    from fourier.db import session as S
    monkeypatch.setattr(S, "session_scope", lambda: _Scope())
    for k in ("homes", "support", "votes", "opts"):
        monkeypatch.setitem(C._WORKER, k, {})


def _summary(**kw):
    return dict(dict(families=1, files=1, near_dups_pruned=0, failed=0), **kw)


# --- 1: a parallel build's log -------------------------------------------------------------
def test_a_worker_hands_its_log_lines_to_the_parent(monkeypatch):
    _worker_env(monkeypatch)

    def built(session, category, out_dir, log=print, **k):
        log(f"{category}: 3 extra notes of multisample sets left out (at most 2 a set a folder)")
        log(f"DONE {category}: 1 families (from 1), 1 files (0 transcoded, 2 failed, 0 near-silent skipped)")
        return _summary(failed=2), []
    monkeypatch.setattr(C, "build_taxonomy", built)
    cat, summary, entries, err, lines = C._worker_build("KICKS", "/tmp/x")
    assert err is None and summary["failed"] == 2 and len(lines) == 2
    said = []
    C._log_worker(said.append, cat, lines)
    assert said == ["[KICKS] 3 extra notes of multisample sets left out (at most 2 a set a folder)",
                    "[KICKS] DONE KICKS: 1 families (from 1), 1 files (0 transcoded, 2 failed, 0 near-silent skipped)"]

    def broke(session, category, out_dir, log=print, **k):
        log(f"{category}: 1 file(s) whose named note the audio contradicts keep their pitch and name")
        raise ValueError("boom")
    monkeypatch.setattr(C, "build_taxonomy", broke)
    cat, summary, entries, err, lines = C._worker_build("KICKS", "/tmp/x")
    assert err == "boom" and summary is None and len(lines) == 1     # what it said before it failed


class _Pool:
    """A ProcessPoolExecutor that runs each task at once, in this process."""

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *a):
        f = cf.Future()
        f.set_result(fn(*a))
        return f

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_a_parallel_build_prints_each_categorys_lines_and_failed_exports(monkeypatch, tmp_path):
    from fourier.db import session as S
    _worker_env(monkeypatch)
    monkeypatch.setattr(S, "session_scope", lambda: _Scope())
    monkeypatch.setattr(S, "shared_with_workers", lambda path: contextlib.nullcontext())  # no database
    monkeypatch.setattr(cf, "ProcessPoolExecutor", _Pool)
    monkeypatch.setattr(C, "index_stale", lambda s: (False, 1, 1))
    monkeypatch.setattr(C, "load_index", lambda: (np.arange(1), np.ones((1, 4), dtype="float32")))
    monkeypatch.setattr(C, "compute_homes", lambda *a: ({}, {}, {}))
    monkeypatch.setattr(C, "_dedupe_across", lambda *a, **k: 0)
    monkeypatch.setattr(C, "merge_manifest", lambda *a: None)
    monkeypatch.setattr(C, "POOL_SURPLUS", False)
    monkeypatch.setattr(C, "_prompt_vectors", lambda *a, **k: {})   # no CLAP model in tests

    def built(session, category, out_dir, log=print, **k):
        log(f"{category}: 4 extra notes of multisample sets left out (at most 2 a set a folder)")
        return _summary(failed=3 if category == "KICKS" else 0), [{"src": f"/l/{category}.wav"}]
    monkeypatch.setattr(C, "build_taxonomy", built)
    said = []
    res = C.build_all_parallel("db", ["KICKS", "SNARES"], str(tmp_path), 2, {}, log=said.append)
    assert [r[0] for r in res] == ["KICKS", "SNARES"]
    assert "[KICKS] 4 extra notes of multisample sets left out (at most 2 a set a folder)" in said
    assert "[SNARES] 4 extra notes of multisample sets left out (at most 2 a set a folder)" in said
    assert any(m.startswith("[KICKS] 1 families, 1 files, 3 failed") for m in said)
    assert not any("failed" in m for m in said if m.startswith("[SNARES] 1 families"))
    # the quiet build's summary lines are untouched (cli/quiet.py)
    assert not any(m.startswith(("WARN ", "verify: ")) for m in said)


# --- 2: the why log follows the cross-category dedupe ----------------------------------------
def _two_picks(root, src):
    for cat, fam in (("PIANO", "chord-x"), ("STABS", "stab-x")):
        d = root / cat / fam
        d.mkdir(parents=True)
        (d / "a.wav").write_bytes(b"x")
        (root / cat / "_manifest.json").write_text(json.dumps([{"family": fam, "copied": 1, "n": 1}]))
    return [("PIANO", {"files": 1}, [{"family": "chord-x", "out": "chord-x/a.wav", "src": src}]),
            ("STABS", {"files": 1}, [{"family": "stab-x", "out": "stab-x/a.wav", "src": src}])]


def test_a_source_one_category_gave_up_is_in_its_why_log(tmp_path, monkeypatch):
    src = "/l/SampleLibrary/Acme/Qzv Pad (Chord) 01.wav"           # chord-named: STABS keeps it
    monkeypatch.setattr(C, "_sample_ids", lambda srcs: {src: 7} if src in srcs else {})
    why_log.write(tmp_path, "PIANO", {"category": "PIANO", "kept": 40, "files": 9, "out": {},
                                      "counts": {"over_budget": 31}})
    res = _two_picks(tmp_path, src)
    assert C._dedupe_across(tmp_path, res) == 1
    doc = why_log.read(tmp_path, "PIANO")
    assert doc["out"]["dup_source"] == {"7": "STABS"} and doc["counts"]["dup_source"] == 1
    assert doc["files"] == 8 and doc["kept"] == 40           # files exported; kept: the candidate pool
    assert why_log.reason_for(doc, 7) == ("dup_source", "STABS")
    line = why_log.describe("dup_source", "STABS", doc)
    assert "stays in STABS" in line
    assert "dup_source" in why_log.FILTER_WORDS and "over_budget" in why_log.FILTER_WORDS
    # a resumed build's second pass over the same category counts nothing twice
    C._why_dups(tmp_path, {"PIANO": {src: "STABS"}})
    doc = why_log.read(tmp_path, "PIANO")
    assert doc["counts"]["dup_source"] == 1 and doc["files"] == 8
    assert why_log.read(tmp_path, "STABS") is None            # the winner's log isn't made up


def test_an_old_why_log_still_reads(tmp_path, monkeypatch):
    # a doc from before: no files, kept lowered by a cross copy; the dedupe leaves files out
    monkeypatch.setattr(C, "_sample_ids", lambda srcs: {s: 3 for s in srcs})
    why_log.write(tmp_path, "PIANO", {"category": "PIANO", "kept": 5, "out": {"cross_copy": {"2": 1}},
                                      "counts": {"cross_copy": 1}})
    C._dedupe_across(tmp_path, _two_picks(tmp_path, "/l/SampleLibrary/Acme/Qzv Stb.wav"))
    doc = why_log.read(tmp_path, "PIANO")
    assert "files" not in doc and doc["kept"] == 5 and doc["out"]["dup_source"] == {"3": "STABS"}
    assert "over budget" in why_log.describe("over_budget", None, doc)
    from fourier.packs.report import _left_out
    assert "picked by another category too" in _left_out(tmp_path, "PIANO")


def test_a_build_counts_what_it_left_over_budget(monkeypatch, tmp_path):
    from .test_cloud_only import _cloud, _kicks
    _paths, _build = _kicks(monkeypatch, tmp_path, n=40)
    _cloud(monkeypatch, [])
    out = tmp_path / "master"
    C.build_taxonomy(None, "KICKS", str(out), homes={}, support={}, votes={}, describe=False,
                     loudness=False, return_entries=True, budget_override=6, log=lambda m: None)
    doc = why_log.read(out, "KICKS")
    n = len((doc.get("out") or {}).get("over_budget") or {})
    assert n > 0 and n == doc["counts"]["over_budget"]
    assert doc["kept"] - doc["files"] - len((doc.get("out") or {}).get("too_quiet") or {}) == n


# --- 3: levels read a wide stereo file as it plays -------------------------------------------
def test_short_term_level_reads_wide_stereo_at_its_channels_level():
    sr = 44100
    rng = np.random.default_rng(5)
    l, r = rng.standard_normal(sr) * 0.1, rng.standard_normal(sr) * 0.1    # uncorrelated: wide
    wide = np.stack([l, r], axis=1)
    mono = C._short_term_db(l, sr)
    assert abs(C._short_term_db(wide, sr) - mono) < 0.5                  # was about 3 dB under
    assert C._short_term_db(wide.mean(axis=1), sr) < mono - 2.5
    same = np.stack([l, l], axis=1)
    assert abs(C._short_term_db(same, sr) - mono) < 1e-9                 # mono in two channels
    from fourier.packs.sets import kit_level_db
    assert kit_level_db(wide, sr) == C._short_term_db(wide, sr)          # one downmix for both


# --- 4: a single cycle goes to the ceiling ---------------------------------------------------
def test_a_quiet_single_cycle_is_not_held_by_the_gain_cap():
    sr = 48000
    cyc = 0.02 * np.sin(2 * np.pi * np.arange(600) / 600)                # -34 dBFS: past the cap
    ceiling = 10 ** (C.PEAK_CEILING_DB / 20)
    y = C._process_audio(cyc.copy(), sr, "waves", False, False, "peak")
    assert len(y) == 600 and abs(float(np.abs(y).max()) - ceiling) < 1e-6
    hit = C._normalize(cyc.copy(), "peak", sr=sr)                        # a one-shot: still capped
    assert float(np.abs(hit).max()) == pytest.approx(0.02 * 10 ** (C.MAX_GAIN_DB / 20))


# --- 5: a phase-fixed one-shot keeps the file's peak -----------------------------------------
@pytest.mark.parametrize("kind, kept_peak", [("oneshot", "file"), ("loop", "channel")])
def test_a_phase_fixed_file_keeps_its_peak(tmp_path, kind, kept_peak):
    sf = pytest.importorskip("soundfile")
    sr = 44100
    t = np.arange(sr) / sr
    s = np.sin(2 * np.pi * 220 * t)
    left, right = 0.6 * s, -0.4 * s
    right[sr // 2] = 0.89                                                # the file's peak, in R
    p = tmp_path / "a.wav"
    sf.write(str(p), np.stack([left, right], axis=1), sr, subtype="FLOAT")
    e = {"out": "a.wav"}
    fixed = C._fix_phase(tmp_path, [e], kind=kind, category=None)       # no RMS ceiling
    assert fixed == 1
    y, _ = sf.read(str(p))
    assert y.ndim == 1 and e["phase_fix"] == "left"
    want = 0.89 if kept_peak == "file" else 0.6
    assert float(np.abs(y).max()) == pytest.approx(want, abs=1e-4)


# --- 6: verify checks the sets' out_md5 ------------------------------------------------------
def test_verify_checks_a_set_files_md5(tmp_path):
    from fourier.packs.verify import check_manifest
    f = tmp_path / "KITS" / "kit-x" / "a.wav"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    good = hashlib.md5(b"x").hexdigest()
    for md5, status in ((good, "PASS"), ("0" * 32, "FAIL")):
        ctx = SimpleNamespace(root=tmp_path, entries=[],
                              set_entries=[("KITS", {"out": "kit-x/a.wav", "out_md5": md5})],
                              path=lambda c, e: tmp_path / c / e["out"])
        got = {r.check: r for r in check_manifest(ctx)}
        assert got["out_md5 matches every file"].level == status


def test_a_phase_fixed_one_shot_stays_under_its_rms_ceiling(tmp_path):
    sf = pytest.importorskip("soundfile")
    sr = 44100
    s = np.sin(2 * np.pi * 220 * np.arange(sr) / sr)
    p = tmp_path / "a.wav"
    sf.write(str(p), np.stack([0.6 * s, -0.4 * s], axis=1), sr, subtype="FLOAT")
    monkey = dict(C.ONESHOT_RMS_CEIL_DB)
    try:
        C.ONESHOT_RMS_CEIL_DB["QXCAT"] = -20.0
        assert C._fix_phase(tmp_path, [{"out": "a.wav"}], kind="oneshot", category="QXCAT") == 1
    finally:
        C.ONESHOT_RMS_CEIL_DB.clear()
        C.ONESHOT_RMS_CEIL_DB.update(monkey)
    y, _ = sf.read(str(p))
    assert 20 * np.log10(np.sqrt(np.mean(y ** 2))) <= -20.0 + 1e-6
