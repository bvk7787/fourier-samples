"""What keeps a large library's build and analysis linear: the near-duplicate prune, the CLAP
index and its unit rows, the analysis steps' writes, the CQT filters and the pYIN cache. Each
does the same arithmetic as before, so picks and audio don't move."""
import numpy as np
import pytest
from click.testing import CliRunner
from sqlalchemy import event, text

from fourier.analysis import clap_features as CF
from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures
from fourier.packs import curate as C


def _old_prune(rec, emb_n, thr, keep=0, apart=None):
    """The prune as it read before: a list of the kept vectors, made into a matrix for each
    candidate. The reference the matrix made once must equal."""
    order = sorted(range(len(rec)), key=lambda i: (not rec[i].get("incumbent"), -rec[i]["qual"], rec[i]["id"]))
    keep_idx, kept, dropped = [], [], []
    for i in order:
        v = emb_n[rec[i]["row"]]
        if kept:
            sims = np.asarray(kept) @ v
            if apart is not None:
                for k in np.flatnonzero(sims > thr):
                    if apart(rec[i], rec[keep_idx[int(k)]]):
                        sims[k] = -1.0
            if float(np.max(sims)) > thr:
                dropped.append((i, keep_idx[int(np.argmax(sims))]))
                continue
        keep_idx.append(i)
        kept.append(v)
    if dropped and len(keep_idx) < keep:
        left = [i for i, _j in dropped]
        while left and len(keep_idx) < keep:
            K = np.asarray(kept)
            far = min(left, key=lambda i: (float(np.max(K @ emb_n[rec[i]["row"]])), i))
            left.remove(far)
            keep_idx.append(far)
            kept.append(emb_n[rec[far]["row"]])
    ks = set(keep_idx)
    return [r for j, r in enumerate(rec) if j in ks], len(rec) - len(ks), {
        rec[i]["id"]: rec[j]["id"] for i, j in dropped if i not in ks}


class _CountingNumpy:
    """numpy, counting the matrices made from a Python list."""

    def __init__(self):
        self.lists = 0

    def __getattr__(self, name):
        return getattr(np, name)

    def asarray(self, a, *args, **kw):
        self.lists += isinstance(a, list)
        return np.asarray(a, *args, **kw)


@pytest.mark.parametrize("keep", [0, 150])
def test_the_prune_keeps_what_it_kept_without_a_matrix_per_candidate(monkeypatch, keep):
    """The kept vectors sit in one matrix made once: building one from a list for every
    candidate copied all of them each time (quadratic in the pool). Same picks, same partners."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal((40, 64)).astype("float32")
    emb = base[rng.integers(0, 40, 300)] + 0.15 * rng.standard_normal((300, 64)).astype("float32")
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    rec = [{"id": 1000 + i, "row": i, "qual": float(rng.random()), "incumbent": i % 17 == 0,
            "bpm": float(rng.choice([90, 120, 174]))} for i in range(300)]
    apart = lambda a, b: a["bpm"] != b["bpm"]           # noqa: E731
    for ap in (None, apart):
        want, n_want, partners_want = _old_prune(rec, emb_n, 0.9, keep=keep, apart=ap)
        counting = _CountingNumpy()
        monkeypatch.setattr(C, "np", counting)
        got, n_got = C._prune_near_dups(rec, emb_n, 0.9, keep=keep, partners=(partners := {}), apart=ap)
        monkeypatch.setattr(C, "np", np)
        assert n_want > 0 and [r["id"] for r in got] == [r["id"] for r in want] and n_got == n_want
        assert partners == partners_want
        assert counting.lists == 0


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_CLAP_INDEX", str(tmp_path / "idx" / "clap_index.npz"))
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "t.duckdb")
    yield tmp_path / "t.duckdb"
    S._engine = S._SessionLocal = None


def _embedded(n, seed=1):
    rng = np.random.default_rng(seed)
    with S.session_scope() as s:
        for i in range(1, n + 1):
            s.add(Sample(id=i, path=f"/l/Acme/{i}.wav", filename=f"{i}.wav"))
            s.add(SampleFeatures(sample_id=i, clap_embedding=CF.embedding_to_bytes(
                rng.standard_normal(512).astype("float32"))))


def test_an_unchanged_index_is_known_by_its_digest(db, monkeypatch):
    """With nothing new, the index's files aren't read: the stored embeddings' digest matches
    the one kept beside the index (reading them both and sorting held about four copies)."""
    _embedded(30)
    with S.session_scope() as s:
        path = CF.build_index(s)
        assert CF.build_index.wrote is True
    monkeypatch.setattr(CF, "load_index_fast", lambda *a, **k: pytest.fail("the index was read"))
    with S.session_scope() as s:
        assert CF.build_index(s) == path and CF.build_index.wrote is False
    monkeypatch.undo()
    monkeypatch.setenv("FOURIER_CLAP_INDEX", str(path))
    # an index another version rewrote: its files no longer match the digest, so it's compared
    ids, emb = CF.load_index_fast()
    np.savez_compressed(path, ids=ids, embeddings=np.asarray(emb))
    with S.session_scope() as s:
        CF.build_index(s)
        assert CF.build_index.wrote is False
        s.execute(text("DELETE FROM sample_features WHERE sample_id = 4"))
    with S.session_scope() as s:
        CF.build_index(s)
        assert CF.build_index.wrote is True and 4 not in set(CF.load_index_fast()[0].tolist())


def test_the_unit_rows_are_shared_from_a_file_beside_the_index(db, monkeypatch):
    """Build workers memory-map the index's unit rows from one file (written once) where each
    made its own copy; the values are the ones computed in memory, bit for bit."""
    _embedded(50)
    with S.session_scope() as s:
        CF.build_index(s)
    ids, emb = C.load_index()
    assert isinstance(emb, np.memmap)
    in_memory = emb.astype("float32") / (np.linalg.norm(emb.astype("float32"), axis=1, keepdims=True) + 1e-9)
    got = C.unit_rows(emb)
    assert isinstance(got, np.memmap) and got.tobytes() == in_memory.tobytes()
    monkeypatch.setattr(CF, "_unit", lambda e: pytest.fail("computed again"))
    again = C.unit_rows(C.load_index()[1])
    assert isinstance(again, np.memmap) and again.tobytes() == in_memory.tobytes()
    plain = np.asarray(emb[:5])                           # not the index: computed as before
    monkeypatch.undo()
    assert C.unit_rows(plain).tobytes() == in_memory[:5].tobytes()


def test_unit_rows_follow_the_matrix_they_were_computed_from(db):
    """The index written again at the same size (an analysis beside a build) gets its own unit
    rows file: a build that mapped the new matrix reads the new rows, and one still holding
    the old matrix keeps rows of the old one."""
    _embedded(20)
    with S.session_scope() as s:
        CF.build_index(s)
    ids, old = C.load_index()
    unit = lambda e: e.astype("float32") / (np.linalg.norm(e.astype("float32"), axis=1, keepdims=True) + 1e-9)
    want_old = unit(np.asarray(old))
    assert C.unit_rows(old).tobytes() == want_old.tobytes()
    other = np.asarray(old)[::-1].copy()                  # same shape, other values
    CF._write_fast_index(CF._index_path(), ids, other)
    _, new = C.load_index()
    assert C.unit_rows(new).tobytes() == unit(other).tobytes()
    assert C.unit_rows(old).tobytes() == want_old.tobytes()
    files = list(CF._index_path().parent.glob(f"{CF._index_path().stem}.unit.*.npy"))
    assert len(files) == 1


@pytest.fixture
def feature_selects():
    """The SELECTs on sample_features any engine runs while the test does."""
    from sqlalchemy.engine import Engine
    seen = []

    def count(conn, cursor, statement, *a):
        if statement.lstrip().upper().startswith("SELECT") and "sample_features" in statement:
            seen.append(statement)
    event.listen(Engine, "before_cursor_execute", count)
    yield seen
    event.remove(Engine, "before_cursor_execute", count)


def test_analysis_steps_write_a_batch_with_one_query(db, feature_selects):
    """A step's batch of results finds its rows with one query (one a sample before)."""
    from fourier.cli import enrich as E
    with S.session_scope() as s:
        for i in range(1, 61):
            s.add(Sample(id=i, path=f"/l/Acme/{i}.wav", filename=f"{i}.wav"))
        for i in range(1, 31):
            s.add(SampleFeatures(sample_id=i, n_events=1))
    seen = feature_selects
    seen.clear()
    s = S.get_session()
    E._upsert_librosa_batch(s, [(i, {"mfcc_mean": [0.5], "onset_rate_hz": 2.0}) for i in range(1, 61)])
    E._upsert_clap_batch(s, [(i, b"\0" * 2048) for i in range(1, 61)], "stand-in")
    E._upsert_quality_batch(s, [(i, {"clip_share": 0.0, "is_clipped": False, "dc_offset_ratio": 0.0})
                                for i in range(1, 61)])
    s.close()
    assert len(seen) <= 3, len(seen)
    with S.session_scope() as s:
        rows = s.execute(text("SELECT count(*), count(onset_rate_hz), count(clap_embedding), "
                              "count(n_events), count(is_clipped) FROM sample_features")).one()
    assert tuple(rows) == (60, 60, 60, 30, 60)


def test_the_derived_step_computes_nothing_without_sononym(db, feature_selects):
    """Without Sononym the derived step's columns stay empty: it says so and marks every
    sample done in a few statements (a query and a write a sample before)."""
    from fourier.cli import main
    with S.session_scope() as s:
        for i in range(1, 201):
            s.add(Sample(id=i, path=f"/l/Acme/{i}.wav", filename=f"{i}.wav", duration_s=0.5))
        s.add(SampleFeatures(sample_id=7, n_events=1))
    S._engine = S._SessionLocal = None
    seen = feature_selects
    seen.clear()
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "analyze", "--only", "derived"])
    assert r.exit_code == 0, r.output
    out = " ".join(r.output.split())
    assert "none to compute without Sononym" in out and "200 samples marked done (199 new rows)" in out
    assert len(seen) < 50, len(seen)                    # a few counts, none a sample
    with S.session_scope() as s:
        got = s.execute(text("SELECT count(*), count(derived_computed_at), count(sub_weight), "
                             "count(n_events) FROM sample_features")).one()
    assert tuple(got) == (200, 200, 0, 1)
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "analyze", "--only", "derived"])
    assert "All samples already have derived features" in r.output


def test_chroma_builds_its_filters_once(monkeypatch):
    """chroma_cqt's constant-Q filters are built once per process for each tuning it
    estimates, and copied out, so every file's chroma is what librosa computes on its own."""
    import librosa
    from librosa.core import constantq as cq

    from fourier.analysis import librosa_features as LF
    rng = np.random.default_rng(3)
    y = rng.standard_normal(22050).astype("float32")
    ys = [y, 0.5 * y, 0.25 * y]                 # three files, one tuning (librosa estimates it)
    want = [librosa.feature.chroma_cqt(y=y, sr=22050) for y in ys]
    real = getattr(cq, "__vqt_filter_fft")
    built = []
    monkeypatch.setattr(cq, "__vqt_filter_fft", lambda *a, **k: (built.append(1), real(*a, **k))[1])
    monkeypatch.setattr(LF, "_FILTERS", {})
    LF.keep_cqt_filters()
    got = [librosa.feature.chroma_cqt(y=ys[0], sr=22050)]
    first = len(built)
    got += [librosa.feature.chroma_cqt(y=y, sr=22050) for y in ys[1:]]
    assert all(np.array_equal(a, b) for a, b in zip(want, got))
    assert first > 0 and len(built) == first               # the first file's filters serve the rest


def test_pyin_readings_go_into_one_file(tmp_path, monkeypatch):
    """pYIN readings are rows of one table in the pitch cache folder (a file a sample before,
    100,000 of them in a large library); a reading an older version kept in its own file is
    read and moved into the table."""
    import json

    import librosa
    import soundfile as sf
    monkeypatch.setenv("FOURIER_PITCH_CACHE", str(tmp_path / "pitch"))
    monkeypatch.setattr(C, "_PITCH_DB", {}, raising=False)
    t = np.arange(22050) / 22050
    srcs = []
    for i in range(4):
        p = tmp_path / f"Acme Tone {i}.wav"
        sf.write(p, (0.5 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 22050)
        srcs.append(str(p))
    calls = []
    real = librosa.pyin
    monkeypatch.setattr(librosa, "pyin", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    got = [C._detect_pitch(p) for p in srcs[:3]]
    assert all(g is not None and abs(g - 57.0) < 0.3 for g in got) and len(calls) == 3
    assert [C._detect_pitch(p) for p in srcs[:3]] == got and len(calls) == 3      # cached
    assert [C.cached_pitch(p) for p in srcs[:3]] == got
    files = sorted(p.name for p in (tmp_path / "pitch").rglob("*") if p.is_file())
    assert not [f for f in files if f.endswith(".json")] and "readings.sqlite" in files
    legacy = C._pitch_cache_file(srcs[3])                  # an older version's own file
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({"midi": 61.25}))
    assert C.cached_pitch(srcs[3]) == 61.25 and C._detect_pitch(srcs[3]) == 61.25 and len(calls) == 3
    legacy.unlink()
    assert C.cached_pitch(srcs[3]) == 61.25                 # moved into the table
