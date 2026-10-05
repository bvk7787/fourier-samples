"""Which metadata providers a build uses (fourier/metadata/providers.py), and how the sample
rows follow them: Sononym classifies when it's there, else the fallback (path and audio)."""
import pytest
from sqlalchemy import text

from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures, SononymMeta
from fourier.metadata import providers as P
from fourier.metadata import rows as R

ROWS = [  # rel, Sononym (classes, categories), Live tags, duration, events
    ("A/One Shots/Kicks/Kick 1.wav", (["OneShot"], ["Perc Kicks"]), ["Kick"], 0.4, 1),
    ("A/Loops/Break 170bpm.wav", (["Loop"], ["Perc Loops"]), None, 4.0, 16),
    ("A/Pads/Long Pad.wav", (["OneShot"], ["Tone Pads & Textures"]), None, 6.0, 1),
    ("A/Misc/thing.wav", (["OneShot"], ["Tone Leads & MidHiKeys"]), None, 1.2, 3),
]


def _db(tmp_path, monkeypatch, sononym=True, ableton=True, config=None):
    monkeypatch.setenv("FOURIER_CONFIG", "none")
    if config is not None:
        cfg = tmp_path / "fourier.toml"
        cfg.write_text(config)
        monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "t.db")
    R.forget_current()
    with S.session_scope() as s:
        for i, (rel, (cls, cats), tags, dur, ev) in enumerate(ROWS, 1):
            s.add(Sample(id=i, path=f"/l/SampleLibrary/{rel}", rel_path=rel, filename=rel.rsplit("/", 1)[1],
                         duration_s=dur, ableton_tags=tags if ableton else None))
            if sononym:
                s.add(SononymMeta(sample_id=i, classes=cls, categories=cats))
            s.add(SampleFeatures(sample_id=i, n_events=ev))
    s = S.get_session()
    R.ensure_current(s)
    return s


@pytest.fixture(autouse=True)
def _reset():
    yield
    R.forget_current()
    S._engine = S._SessionLocal = None


def test_auto_uses_sononym_and_live_when_they_have_data(tmp_path, monkeypatch):
    s = _db(tmp_path, monkeypatch)
    a = P.active(s)
    assert (a.names, a.classifiers, a.tags, a.source) == (("sononym", "ableton"), ("sononym",), ("ableton",), "auto")
    assert not a.fallback
    assert P.describe(a) == "providers: sononym, ableton [auto]"


def test_auto_falls_back_to_path_and_audio_without_sononym(tmp_path, monkeypatch):
    s = _db(tmp_path, monkeypatch, sononym=False, ableton=False)
    a = P.active(s)
    assert a.classifiers == ("path", "audio") and a.tags == () and a.fallback
    assert "no Sononym" in P.describe(a)


def test_config_names_them(tmp_path, monkeypatch):
    """fourier.toml wins over what the database has: Sononym's data is there, but unused."""
    s = _db(tmp_path, monkeypatch, config='providers = ["ableton"]\n')
    a = P.active(s)
    assert a.source == "fourier.toml" and a.fallback and a.tags == ("ableton",)
    assert a.names == ("path", "audio", "ableton")


@pytest.mark.parametrize("body, msg", [('providers = ["sononym", "echonest"]\n', "unknown providers echonest"),
                                       ('providers = "sononym"\n', "must be a list")])
def test_bad_config(tmp_path, monkeypatch, body, msg):
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(body)
    monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    with pytest.raises(P.ProviderError, match=msg):
        P.configured()


def test_fallback_rows(tmp_path, monkeypatch):
    s = _db(tmp_path, monkeypatch, sononym=False, ableton=False)
    q = R.sample_select("id", "classes", "categories", "canonical", "ableton_tags", session=s)
    got = {r.id: r for r in R.fetch(s, q.order_by(text("id")))}
    assert set(got) == {1, 2, 3, 4}                         # every file is a candidate
    assert got[1].canonical == ["class.oneshot", "kick"] and got[1].classes == ["OneShot"]
    assert got[2].canonical == ["class.loop"] and got[2].classes == ["Loop"]   # path's shape first
    assert got[3].canonical == ["class.oneshot", "pad"]     # one long event: audio says one-shot
    assert got[4].canonical == [] and got[4].classes == []  # nothing says anything
    assert all(r.categories == [] and r.ableton_tags is None for r in got.values())


def test_fallback_filters(tmp_path, monkeypatch):
    s = _db(tmp_path, monkeypatch, sononym=False, ableton=False)

    def ids(field, values):
        sql, p = R.like_any(field, values, "x", quoted=True, session=s)
        q = R.sample_select("id", session=s).where(text(sql).bindparams(**p))
        return sorted(r.id for r in R.fetch(s, q))

    assert ids("classes", ["OneShot"]) == [1, 3]           # the fallback's shape, by Sononym's name
    assert ids("canonical", ["pad"]) == [3]
    assert ids("categories", ["Perc Kicks"]) == []         # Sononym's own names: nobody gives them
    assert ids("ableton_tags", ["Kick"]) == []              # Live isn't in use


def test_live_without_sononym_still_votes(tmp_path, monkeypatch):
    s = _db(tmp_path, monkeypatch, sononym=False, ableton=True)
    assert P.active(s).names == ("path", "audio", "ableton")
    sql, p = R.like_any("ableton_tags", ["Kick"], "t", quoted=True, session=s)
    q = R.sample_select("id", "ableton_tags", session=s).where(text(sql).bindparams(**p))
    assert [(r.id, r.ableton_tags) for r in R.fetch(s, q)] == [(1, ["Kick"])]


def test_sononym_rows_are_as_before(tmp_path, monkeypatch):
    """With Sononym in use the rows are Sononym's, and the path labels route nothing."""
    s = _db(tmp_path, monkeypatch)
    got = R.fetch(s, R.sample_select("id", "classes", "categories", session=s).order_by(text("id")))
    assert [(r.classes, r.categories) for r in got] == [tuple(x[1]) for x in ROWS]
    assert s.execute(text("SELECT count(*) FROM labels WHERE provider = 'path'")).scalar() == 0


def test_cli_config_reaches_the_providers(tmp_path, monkeypatch):
    """--config is where every reader of fourier.toml looks, providers included."""
    from click.testing import CliRunner

    from fourier.cli import main
    cfg = tmp_path / "mine.toml"
    cfg.write_text('providers = ["path", "audio"]\n')
    monkeypatch.setenv("FOURIER_CONFIG", "none")
    monkeypatch.delenv("FOURIER_RESOLVED_CONFIG", raising=False)
    CliRunner().invoke(main, ["--db", str(tmp_path / "c.db"), "--config", str(cfg), "tools", "db-stats", "--metadata"])
    import os
    assert os.environ["FOURIER_CONFIG"] == str(cfg.resolve())
    assert P.configured() == ["path", "audio"]


def test_without_live_path_words_stand_in_for_its_tags(tmp_path, monkeypatch):
    """DRUMLOOPS needs a drum tag and ACOUSTIC's pool bowed-string tags: without Live, the
    path's words supply them (curate_config.NAME_TAGS); with Live, rows are left alone."""
    from fourier.packs import curate as C
    assert C._name_tags("Loops/Drum Loops/Break 170.wav") == ["Drum Loop"]
    assert C._name_tags("Strings/Cello Sustain C2.wav") == ["Cello"]
    assert C._name_tags("Leads/Saw 1.wav") == []
    # whole words of the file's name and folder, never the vendor's or pack's name: a breaks
    # pack's kicks and risers get no drum-loop tag, and "Beatbox" isn't "beat"
    assert C._name_tags("Acme/Acme Breaks and Hits/Kicks/Kick 01.wav") == []
    assert C._name_tags("Northwind/Jungle Breaks 174/FX/Riser 01.wav") == []
    assert C._name_tags("Acme/Pack/Beatbox/Vox 01.wav") == []
    assert C._name_tags("Acme/Pack/Breaks/Kick 01.wav") == []          # a drum name beats its folder
    assert C._name_tags("Acme/Breakbeats/WAV/Amen 01.wav") == ["Drum Loop"]   # WAV hands over
    # a file already a loop may take the loop tag from its pack's name
    assert C._name_tags("Northwind/Jungle Breaks 174/Loops/Roller 01.wav", loop=True) == ["Drum Loop"]
    assert C._name_tags("Northwind/Jungle Breaks 174/FX/Riser 01.wav", loop=False) == []
    # ...only when its own name or folder has a loop word: a riser, pad or bassline the audio
    # alone calls a loop (onsets, no silent gap) takes nothing from a breaks pack's name
    assert C._name_tags("Northwind/Jungle Breaks 174/FX/Riser 04.wav", loop=True) == []
    assert C._name_tags("Northwind/Jungle Breaks 174/Pads/Pad 01.wav", loop=True) == []
    assert C._name_tags("Northwind/Jungle Breaks 174/Roller 01 174bpm.wav", loop=True) == ["Drum Loop"]
    s = _db(tmp_path, monkeypatch, ableton=False)                   # Sononym only
    rows = C._with_name_tags(s, R.fetch(s, R.sample_select("id", "rel_path", "ableton_tags", session=s)))
    assert {r.rel_path: r.ableton_tags for r in rows}["A/Loops/Break 170bpm.wav"] == ["Drum Loop"]
    sql, p = C._name_tag_sql(["Cello", "Harp"], "n")
    assert "LIKE" in sql and set(p.values()) == {"%cello%"}
    assert C._name_tag_sql(["Harp"], "n") == ("1 = 0", {})
    R.forget_current()
    (tmp_path / "live").mkdir()
    s2 = _db(tmp_path / "live", monkeypatch)                         # Sononym and Live
    rows = R.fetch(s2, R.sample_select("id", "rel_path", "ableton_tags", session=s2))
    assert C._with_name_tags(s2, rows) is rows
