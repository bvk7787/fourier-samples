"""Which samples the analysis steps take (cli/enrich.py): with Sononym classifying, the ones
it analysed, as always; without it (the built-in path and audio providers), every sample in
the database, whether or not it has a Sononym row."""
import pytest
from click.testing import CliRunner

from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures, SononymMeta


@pytest.fixture
def db(tmp_path, monkeypatch):
    from fourier.metadata.rows import forget_current
    old = (S._engine, S._SessionLocal)
    path = tmp_path / "t.db"
    S.init_db(path)
    forget_current()
    yield path
    forget_current()
    S._engine, S._SessionLocal = old


def _add(s, i, sononym):
    s.add(Sample(id=i, path=f"/lib/SampleLibrary/V/P/Kick {i:02d}.wav", rel_path=f"V/P/Kick {i:02d}.wav",
                 filename=f"Kick {i:02d}.wav", duration_s=0.4))
    if sononym:
        s.add(SononymMeta(sample_id=i, classes=["OneShot"], categories=["Perc Kicks"], brightness=0.2,
                          harmonicity=0.6, noisiness=0.3, crest_factor=6.0, bpm=None))


def _derived(db):
    from fourier.cli import main
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "analyze", "--only", "derived"])
    assert r.exit_code == 0, r.output
    with S.session_scope() as s:
        return sorted(i for (i,) in s.query(SampleFeatures.sample_id)
                      .filter(SampleFeatures.derived_computed_at != None))  # noqa: E711


def test_with_sononym_the_steps_take_the_samples_it_analysed(db):
    from fourier.cli.enrich import _candidates
    with S.session_scope() as s:
        for i in (3, 1, 2):
            _add(s, i, sononym=i != 2)               # 2: a walked file Sononym never saw
    with S.session_scope() as s:
        got = [x.id for x in _candidates(s, Sample, SononymMeta).all()]
        want = [x.id for x in s.query(Sample).join(SononymMeta).all()]   # the query as it always was
        assert got == want and sorted(got) == [1, 3]
    assert _derived(db) == [1, 3]


def test_without_sononym_every_sample_is_analysed(db):
    from fourier.cli.enrich import _candidates, sononym_classifies
    with S.session_scope() as s:
        for i in (3, 1, 2):
            _add(s, i, sononym=False)
    with S.session_scope() as s:
        assert not sononym_classifies(s)
        assert [x.id for x in _candidates(s, Sample, SononymMeta).all()] == [1, 2, 3]
    assert _derived(db) == [1, 2, 3]
    r = CliRunner().invoke(__import__("fourier.cli", fromlist=["main"]).main,
                           ["--db", str(db), "tools", "analyze", "--only", "derived"])
    assert "All samples already have derived features" in r.output       # true now


def test_nothing_scanned_is_one_line(db):
    from fourier.cli import main
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "analyze", "--only", "derived"])
    assert r.exit_code == 1 and "nothing scanned yet: run `fourier tools scan`" in r.output
    assert "Traceback" not in r.output


def test_librosa_timestamp_is_naive_utc_without_deprecation(db):
    import warnings
    from datetime import datetime, timezone

    from fourier.cli.enrich import _upsert_librosa_batch

    with S.session_scope() as s:
        _add(s, 1, sononym=False)
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    with S.session_scope() as s, warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*utcnow.*", category=DeprecationWarning)
        _upsert_librosa_batch(s, [(1, {"tempo_bpm": 120.0})])
    after = datetime.now(timezone.utc).replace(tzinfo=None)
    with S.session_scope() as s:
        feat = s.query(SampleFeatures).filter_by(sample_id=1).one()
        assert feat.computed_at.tzinfo is None
        assert before <= feat.computed_at <= after
        assert feat.tempo_bpm == 120.0
