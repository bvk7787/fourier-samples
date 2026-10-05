"""Generic provider metadata (fourier/metadata/store.py): Sononym's values in labels and
descriptors, rebuilt from sononym_meta, with a parity check that proves they match."""
import pytest
from sqlalchemy import text

from fourier.db import session as S
from fourier.db.models import Sample, SononymMeta
from fourier.metadata import store as M


@pytest.fixture(params=[".db", ".duckdb"])
def db(tmp_path, request):
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / f"t{request.param}")
    with S.session_scope() as s:
        rows = [
            (["OneShot"], ["Perc Vinyl Scratches", "Perc Kicks"], 0.2, 36.0),
            (["Loop"], [], 0.5, None),                  # no categories; no base note
            (["OneShot"], ["Tone Bass & LowKeys"], None, 40.5),   # no brightness
        ]
        for i, (cls, cats, br, note) in enumerate(rows, 1):
            s.add(Sample(id=i, path=f"/l/{i}.wav", rel_path=f"{i}.wav", filename=f"{i}.wav"))
            s.add(SononymMeta(sample_id=i, classes=cls, categories=cats, brightness=br, base_note=note,
                              bpm=120.0 + i, peak_db=-1.5, category_strengths=[0.0] * 29))
    yield S.get_session
    S._engine = S._SessionLocal = None


def test_rebuild_then_parity_is_exact(db):
    with S.session_scope() as s:
        n = M.rebuild_sononym(s)
        assert n["labels"] == 6 + 6                      # raw: classes 3, categories 3; canonical 6
        canon = s.execute(text("SELECT label FROM labels WHERE sample_id = 1 AND kind = 'canonical' "
                               "ORDER BY label")).all()
        assert [r[0] for r in canon] == ["class.oneshot", "kick", "scratch"]
        rep = M.sononym_parity(s)
        assert rep == {"samples": 3, "differences": {}, "examples": []}
        got = s.execute(text("SELECT label, rank FROM labels WHERE sample_id = 1 AND kind = 'category' "
                             "ORDER BY rank")).all()
        assert [tuple(r) for r in got] == [("Perc Vinyl Scratches", 0), ("Perc Kicks", 1)]
        names = {r[0] for r in s.execute(text("SELECT name FROM descriptors WHERE sample_id = 2"))}
        assert "analysed" in names and "base_note" not in names and "bpm" in names


def test_rebuild_is_idempotent(db):
    with S.session_scope() as s:
        a = M.rebuild_sononym(s)
        b = M.rebuild_sononym(s)
        assert a == b and not M.sononym_parity(s)["differences"]


@pytest.mark.parametrize("sql, field", [
    ("UPDATE labels SET rank = 5 WHERE sample_id = 1 AND label = 'Perc Vinyl Scratches'", "category"),
    ("DELETE FROM labels WHERE sample_id = 3 AND kind = 'class'", "class"),
    ("UPDATE descriptors SET value = 0.3 WHERE sample_id = 1 AND name = 'brightness'", "brightness"),
    ("INSERT INTO descriptors (sample_id, provider, name, value) VALUES (3, 'sononym', 'brightness', 0.1)",
     "brightness"),
    ("DELETE FROM descriptors WHERE sample_id = 2 AND name = 'analysed'", "analysed"),
])
def test_parity_catches_each_kind_of_drift(db, sql, field):
    with S.session_scope() as s:
        M.rebuild_sononym(s)
        s.execute(text(sql))
        rep = M.sononym_parity(s)
        assert set(rep["differences"]) == {field} and rep["examples"]


def test_importer_keeps_them_in_step(db, monkeypatch):
    from fourier.ingest import importer
    importer._rebuild_generic_metadata({"created": 0, "updated": 0})      # empty tables: builds
    with S.session_scope() as s:
        assert not M.sononym_parity(s)["differences"]
        s.execute(text("UPDATE sononym_meta SET categories = '[\"Perc Snares\"]' WHERE sample_id = 3"))
    importer._rebuild_generic_metadata({"created": 0, "updated": 0})      # nothing changed: kept
    with S.session_scope() as s:
        assert M.sononym_parity(s)["differences"] == {"category": 1}
    importer._rebuild_generic_metadata({"created": 0, "updated": 1})
    with S.session_scope() as s:
        assert not M.sononym_parity(s)["differences"]


def test_ableton_tags_as_labels(db):
    with S.session_scope() as s:
        s.execute(text("UPDATE samples SET ableton_tags = '[\"Kick\", \"Drums\"]' WHERE id = 1"))
        s.execute(text("UPDATE samples SET ableton_tags = '[]' WHERE id = 2"))   # tagged with nothing
        n = M.rebuild(s, M.ABLETON)                     # sample 3: never tagged
        assert n == {"labels": 2, "descriptors": 2}
        assert M.parity(s, M.ABLETON) == {"samples": 2, "differences": {}, "examples": []}
        s.execute(text("UPDATE labels SET label = 'Snare' WHERE provider = 'ableton' AND label = 'Kick'"))
        assert M.parity(s, M.ABLETON)["differences"] == {"tag": 1}
        s.execute(text("UPDATE samples SET ableton_tags = NULL WHERE id = 1"))
        assert "orphan rows" in M.parity(s, M.ABLETON)["differences"]


def test_an_unmapped_label_is_a_parity_difference(db):
    with S.session_scope() as s:
        s.execute(text("UPDATE sononym_meta SET categories = '[\"Perc Kazoos\"]' WHERE sample_id = 3"))
        M.rebuild(s, M.SONONYM)
        rep = M.sononym_parity(s)
        assert rep["differences"] == {"unmapped": 1}
        assert (None, "unmapped", "category:Perc Kazoos", 1) in rep["examples"]
