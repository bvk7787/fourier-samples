"""Samples a library walk marked missing aren't analyzed or counted as left to analyze: their
files aren't there (a drive unplugged, a folder moved away). A walk that finds one again
unmarks it, and the next analysis reads it."""
import pytest
from click.testing import CliRunner

from fourier.db import session as S
from fourier.db.models import MissingFile, Sample, SampleFeatures, SononymMeta

GONE = 2            # marked missing at its path
REUSED = 4          # a missing row from another file that had its id: not this one


@pytest.fixture(params=["t.db", "t.duckdb"])
def db(tmp_path, monkeypatch, request):
    from fourier.metadata.rows import forget_current
    old = (S._engine, S._SessionLocal)
    path = tmp_path / request.param
    S.init_db(path)
    forget_current()
    with S.session_scope() as s:
        for i in (1, 2, 3, 4):
            p = f"/lib/SampleLibrary/Acme/Qxvwz/Kick {i:02d}.wav"
            s.add(Sample(id=i, path=p, rel_path=p.split("SampleLibrary/")[1],
                         filename=f"Kick {i:02d}.wav", duration_s=0.4))
            s.add(SampleFeatures(sample_id=i))
        s.add(MissingFile(sample_id=GONE, path="/lib/SampleLibrary/Acme/Qxvwz/Kick 02.wav"))
        s.add(MissingFile(sample_id=REUSED, path="/lib/SampleLibrary/Acme/Old/Gone.wav"))
    yield path
    forget_current()
    S._engine, S._SessionLocal = old


def test_the_steps_leave_out_samples_marked_missing(db):
    from fourier.cli.enrich import _candidates, _key_items
    with S.session_scope() as s:
        assert [x.id for x in _candidates(s, Sample, SononymMeta).all()] == [1, 3, 4]
        assert [i for i, _p in _key_items(s, all_classes=True)] == [1, 3, 4]


@pytest.mark.parametrize("step", ["librosa", "quality", "events", "loop-trim"])
def test_no_step_reads_a_missing_file(db, monkeypatch, step):
    from fourier.cli import enrich as E
    from fourier.cli import main
    read = []

    def results(fn, items, workers, **_k):
        read.extend(str(it) for it in items)
        return iter(())
    monkeypatch.setattr(E, "_results", results)
    monkeypatch.setattr(E, "_on_this_machine", lambda items: items)
    args = ["--db", str(db), "tools", "analyze", "--only", step]
    if step == "loop-trim":
        monkeypatch.setattr(E, "_fallback_loop", lambda smp: True)
    CliRunner().invoke(main, args)
    assert read, f"{step} read nothing"
    assert not [r for r in read if "Kick 02" in r], read
    assert [r for r in read if "Kick 04" in r]


def test_derived_reads_no_missing_file(db, monkeypatch):
    from fourier.cli import enrich as E
    from fourier.cli import main
    with S.session_scope() as s:              # no header facts yet: the step reads the files
        s.query(Sample).update({Sample.duration_s: None})
    seen = []
    real = E.fill_file_info
    monkeypatch.setattr(E, "fill_file_info", lambda s, samples: seen.extend(x.id for x in samples)
                        or real(s, samples))
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "analyze", "--only", "derived"])
    assert r.exit_code == 0, r.output
    assert sorted(seen) == [1, 3, 4]


def test_missing_samples_arent_left_to_analyze(db):
    from fourier.cli.setup import check_analysis, pending_analysis
    with S.session_scope() as s:
        left = pending_analysis(s)
        assert left["librosa"] == left["clap"] == left["derived"] == 3, left
        assert left["events"] == left["quality"] == 3, left
        rows = {name: detail for _lvl, name, detail in check_analysis(s)}
        assert rows["analysis: librosa"].startswith("0 of 3"), rows
