"""SampleRow (fourier/metadata/rows.py): the selects curation reads samples through, now
over the generic labels/descriptors tables, returning what the sononym_meta selects did."""
import pytest
from sqlalchemy import select, text

from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures, SononymMeta
from fourier.metadata import rows as R


@pytest.fixture
def db(tmp_path):
    S._engine = S._SessionLocal = None
    R.forget_current()
    S.init_db(tmp_path / "t.db")
    with S.session_scope() as s:
        for i, (cls, cats, tags, feat) in enumerate([
            (["OneShot"], ["Perc Kicks"], ["Kick"], True),
            (["Loop"], ["Loops Drums"], None, False),       # no features row: outer join
            (["OneShot"], ["Tone Bass & LowKeys"], ["Bass", "Kick Bass"], True),
        ], 1):
            s.add(Sample(id=i, path=f"/l/SampleLibrary/P/{i}.wav", rel_path=f"P/{i}.wav", filename=f"{i}.wav",
                         file_hash=f"h{i}", duration_s=0.5 * i, file_format="wav", ableton_tags=tags))
            s.add(SononymMeta(sample_id=i, classes=cls, categories=cats, brightness=0.1 * i, bpm=120.0 * i))
            if feat:
                s.add(SampleFeatures(sample_id=i, tempo_bpm=121.0, n_events=i))
        # an `import scan` file nobody analysed: no classifier row, never a candidate
        s.add(Sample(id=4, path="/l/SampleLibrary/P/4.wav", rel_path="P/4.wav", filename="4.wav", file_hash="h4"))
    yield S.get_session()
    S._engine = S._SessionLocal = None


def test_same_rows_as_the_hand_written_selects(db):
    old = (select(Sample.id, Sample.rel_path, Sample.path, Sample.filename,
                  SononymMeta.categories, SononymMeta.classes, Sample.ableton_tags,
                  SononymMeta.harmonicity, SampleFeatures.chroma_concentration,
                  SampleFeatures.n_events, Sample.duration_s, Sample.file_hash,
                  SononymMeta.bpm, SampleFeatures.tempo_bpm, SampleFeatures.onset_rate_hz)
           .join(SononymMeta, SononymMeta.sample_id == Sample.id)
           .outerjoin(SampleFeatures, SampleFeatures.sample_id == Sample.id))
    new = R.sample_select(*R.HOME_FIELDS)
    a = db.execute(old.order_by(Sample.id)).all()
    b = R.fetch(db, new.order_by(Sample.id))            # builds labels/descriptors on first use
    assert [tuple(r) for r in a] == [tuple(r)[:-3] for r in b] and len(b) == 3     # not id 4 (nor canonical, acid_*)
    assert b[0].canonical == ["class.oneshot", "kick"]           # ...plus the canonical labels
    assert list(b[0]._fields) == list(R.HOME_FIELDS)
    assert b[1].n_events is None and b[1].bpm == 240.0                        # outer join on features
    assert b[0].categories == ["Perc Kicks"] and b[1].classes == ["Loop"]


def test_candidate_fields_all_resolve(db):
    rows = R.fetch(db, R.sample_select(*R.CANDIDATE_FIELDS).order_by(Sample.id))
    assert len(rows) == 3 and rows[2].brightness == pytest.approx(0.3)
    with pytest.raises(KeyError, match="nope"):
        R.sample_select("id", "nope")


def test_a_features_field_first_still_selects_from_samples(db):
    rows = R.fetch(db, R.sample_select("tempo_bpm", "path", "categories").order_by(text("path")))
    assert list(rows[0]._fields) == ["tempo_bpm", "path", "categories"]      # no helper id column
    assert rows[2].categories == ["Tone Bass & LowKeys"]
    assert [r.path.rsplit("/", 1)[1] for r in rows] == ["1.wav", "2.wav", "3.wav"]


def test_like_any_filters(db):
    def ids(sql, p):
        return sorted(r.id for r in R.fetch(db, R.sample_select("id").where(text(sql).bindparams(**p))))
    sql, p = R.like_any("categories", ["Kicks", "Bass"], "c")
    assert p == {"c0": "%Kicks%", "c1": "%Bass%"}
    assert ids(sql, p) == [1, 3]
    assert ids(*R.like_any("classes", ["Loop"], "k")) == [2]
    sql, p = R.like_any("ableton_tags", ["Kick"], "t", quoted=True)
    assert p == {"t0": "Kick"}                         # a whole tag: not "Kick Bass"
    assert ids(sql, p) == [1]
    assert ids(*R.like_any("categories", ["Perc Kicks"], "q", quoted=True)) == [1]


def test_ableton_tags_come_from_labels(db):
    rows = R.fetch(db, R.sample_select("id", "ableton_tags").order_by(Sample.id))
    assert [r.ableton_tags for r in rows] == [["Kick"], None, ["Bass", "Kick Bass"]]   # 2: never tagged
    assert db.execute(text("SELECT count(*) FROM labels WHERE provider = 'ableton'")).scalar() == 3
    sql, p = R.like_any("ableton_tags", ["kick"], "t", quoted=True)                  # case as LIKE did
    assert [r.id for r in R.fetch(db, R.sample_select("id").where(text(sql).bindparams(**p)))] == [1]


def test_a_stale_cache_is_rebuilt_once_per_process(db):
    R.fetch(db, R.sample_select("id"))
    db.add(SononymMeta(sample_id=4, classes=["OneShot"], categories=["Perc Snares"]))
    db.commit()
    assert len(R.fetch(db, R.sample_select("id"))) == 3          # checked once: not yet
    R.forget_current()
    rows = R.fetch(db, R.sample_select("id", "categories").order_by(Sample.id))
    assert len(rows) == 4 and rows[3].categories == ["Perc Snares"]


def test_samples_of_other_libraries_are_left_out_only_when_some_are(tmp_path, monkeypatch):
    """rows.outside_library: the samples under none of the configured library folders, when
    the database holds some under them too; a database whose samples are all under them (or
    none are) leaves nothing out."""
    from fourier import places
    from fourier.db import session as sess
    from fourier.db.models import Sample
    from fourier.metadata import rows
    sess._engine = sess._SessionLocal = None
    sess.init_db(tmp_path / "o.db")
    s = sess.get_session()
    try:
        mine = [Sample(path=f"/x/SampleLibrary/Acme/K{i}.wav", filename=f"K{i}.wav") for i in range(3)]
        s.add_all(mine)
        s.commit()
        monkeypatch.setenv("FOURIER_LIBRARY", "SampleLibrary")          # a bare folder name
        places.reset()
        assert rows.outside_library(s) == frozenset()                   # all of them under it
        other = Sample(path="/y/OtherLib/Zeta/K9.wav", filename="K9.wav")
        s.add(other)
        s.commit()
        assert rows.outside_library(s) == {other.id}
        assert "1 sample from other libraries left out" in rows.outside_note(s)
        q = rows.sample_select("id", session=None)
        assert "NOT IN" not in str(q)                                  # no session: as before
        monkeypatch.setenv("FOURIER_LIBRARY", "/z/Nowhere")             # none under it, not on disk
        places.reset()
        assert rows.outside_library(s) == frozenset() and not rows.library_unscanned(s)
        # a library folder on disk that holds none of them: not scanned yet, every sample in the
        # database is another library's
        new = tmp_path / "NewLib"
        new.mkdir()
        monkeypatch.setenv("FOURIER_LIBRARY", str(new))
        places.reset()
        assert rows.outside_library(s) == {x.id for x in mine} | {other.id}
        assert rows.library_unscanned(s)
        assert "isn't scanned yet" in rows.outside_note(s) and "4 samples" in rows.outside_note(s)
        assert rows.usable_count(s) == 0
        monkeypatch.setenv("FOURIER_LIBRARY", f"/y/OtherLib{__import__('os').pathsep}/x/SampleLibrary")
        places.reset()
        assert rows.outside_library(s) == frozenset()                   # both configured
    finally:
        s.close()
        sess._engine = sess._SessionLocal = None
        places.reset()
