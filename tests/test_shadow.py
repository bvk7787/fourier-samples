"""The shadow providers (fourier/metadata/shadow.py): path and audio labels, scored against
the classifier's and changing no routing."""
import pytest
from sqlalchemy import text

from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures, SononymMeta
from fourier.metadata import shadow as SH
from fourier.metadata import store


@pytest.mark.parametrize("rel, labels", [
    ("Pack/One Shots/Kicks/BD_909_02.wav", {"kick"}),               # shape: the name and folder only
    ("Pack/Kicks/One Shots/BD_909_02.wav", {"class.oneshot", "kick"}),
    ("Pack/Loops/Amen Break 170bpm.wav", {"class.loop"}),
    ("Pack/Drums/DeepKick.wav", {"kick"}),                          # camelCase
    ("Pack/Hihat Closed 03.wav", {"hat"}),
    ("Pack/FX/Riser_Up_08.wav", {"fx.sweep"}),
    ("Pack/Vocals/Chant 01.wav", {"vocal"}),
    ("Pack/Synth Stabs/Stab C.wav", {"stab"}),
    ("Pack/Misc/untitled 4.wav", set()),
    ("Keys Pack/Grand Piano/Piano 01.wav", {"keys"}),               # keys by name: PIANO's
    ("Keys/rhodes_Cm.wav", {"keys"}),
    ("Keys/Soft E-Piano.wav", {"keys"}),
    ("Electric Piano/EP_C3.wav", {"keys"}),
    ("Synth Keys/Lead 01.wav", {"lead"}),
    ("Machines/808/Hat 808 01.wav", {"hat"}),                        # a bare 808 is a machine, not a bass
    ("Kits/TOM/Kick TOM Low.wav", {"kick"}),                         # an all-caps TOM is a machine
    ("Kits/Toms/Tom Hi 2.wav", {"tom"}),
    ("Claps/Noise Clap 2.wav", {"clap"}),                            # a weak word yields
    ("FX/Noise 01.wav", {"fx.noise"}),                               # ...and speaks when alone
    ("Leads/Metal Lead.wav", {"lead"}),
    ("Pack/Synths/Patch C3.wav", {"lead"}),                           # near_only: the folder names it
    ("Synths Vol 2/Misc/Patch C3.wav", set()),                        # ...a pack name doesn't
    ("Synth Drums A/Kit/Cymbals/Crash 01.wav", {"cymbal.crash"}),   # the pack name says nothing
    ("Bass Pack/One Shots/Hit 2.wav", {"class.oneshot", "bass"}),  # ...unless nothing nearer does
    ("Synth Loops/Drum Loops/Break 90 01.wav", {"class.loop"}),
    # a vendor's or pack's name never says one-shot or loop; a file's drum name beats its folder
    ("Acme Audio/Acme Breaks and Hits/Kicks/Kick 01.wav", {"kick"}),
    ("Northwind Loops/Jungle Breaks 174/FX/Riser 01.wav", {"fx.sweep"}),
    ("Vendor C/Drum Hits/Loops/Groove 01.wav", {"class.loop"}),
    ("Vendor C/Drum Hits/Loops/Kick 01.wav", {"kick"}),
    ("Vendor C/Breaks/One Shots/Kick 01.wav", {"class.oneshot", "kick"}),
    ("Vendor C/Jungle Breaks/WAV/Amen 01.wav", {"class.loop"}),       # a format folder hands over
    ("Vendor C/Breakbeats/24-bit WAV Files/Amen 01.wav", {"class.loop"}),
    ("Vendor C/Drumloops/Amen 01.wav", {"class.loop"}),
])
def test_path_labels(rel, labels):
    assert SH.path_labels(rel) == labels


def test_path_bpm():
    assert SH.path_bpm("Loops/Amen 170bpm.wav") == 170.0
    assert SH.path_bpm("Loops/Amen 170 BPM.wav") == 170.0
    assert SH.path_bpm("Kicks/BD 909.wav") is None and SH.path_bpm("x 12bpm.wav") is None


def test_audio_class():
    assert SH.audio_class(4.0, 16) == "class.loop"
    assert SH.audio_class(0.4, 1) == "class.oneshot"
    assert SH.audio_class(6.0, 1) == "class.oneshot"                # a pad: one long event
    assert SH.audio_class(0.4, None) == "class.oneshot" and SH.audio_class(6.0, None) is None
    assert SH.audio_class(1.2, 3) is None and SH.audio_class(None, 1) is None
    # a groove with no silent gap is one event, but its onsets say loop (over 1.5 s, at a
    # steady rate); a pad's or a crash's few onsets don't
    assert SH.audio_class(6.0, 1, onset_rate_hz=3.5) == "class.loop"
    assert SH.audio_class(6.0, 1, onset_rate_hz=0.3) == "class.oneshot"
    assert SH.audio_class(1.2, 1, onset_rate_hz=6.0) == "class.oneshot"     # short: a one-shot
    assert SH.audio_class(3.0, 1, onset_rate_hz=1.0) == "class.oneshot"     # 3 onsets: too few
    assert SH.audio_class(4.0, 1, onset_rate_hz=1.0) == "class.loop"


def test_a_named_sound_beats_a_loop_guess_from_the_audio():
    """A riser, pad or bassline has onsets and no silent gap, as a groove has: when the file's
    own name or folder names an instrument or FX sound and no loop, the audio provider calls
    it a one-shot; a loop word, a drum name or a pack name changes nothing."""
    groove = (3.0, 1, 5.3)              # 3 s, no silent gap, 16 onsets: a loop by the audio
    assert SH.audio_class(*groove) == "class.loop"
    for rel in ("Northwind/Jungle Breaks 174/FX/Riser 04.wav", "Acme/Breaks/Pads/Pad 01.wav",
                "Acme/Breaks/Bass/Bass 05.wav", "Acme/Synths/Arp 3.wav", "Acme/Misc/Riser Long.wav",
                "Acme/Breaks/Vocals/Chant 02.wav", "Acme/Breaks/Keys/Piano Demo 3.wav"):
        assert SH.names_a_sound(rel) and SH.audio_label(rel, *groove) == "class.oneshot", rel
    for rel in ("Acme/Bass Loops/Line 01.wav",             # a loop word: still a loop
                "Acme/Bass/Bass Loop 01.wav", "Acme/Loops/Pad 120bpm.wav",
                "Acme/Breaks/Beats/Four Floor 01.wav",     # no sound named
                "Acme/Bass Music Pack/Grooves/Roller 01.wav",   # a pack name names nothing
                "Acme/Breaks/Hats/Hat 01.wav"):            # a drum isn't an instrument or FX
        assert SH.audio_label(rel, *groove) == "class.loop", rel
    assert SH.audio_label("Acme/FX/Riser 01.wav", 2.0, 1, 1.0) == "class.oneshot"
    assert SH.audio_label("Acme/FX/Riser 01.wav", None, None) is None
    assert SH.near_labels("Vendor/Jungle Breaks/Pads/WAV/Pad 01.wav") == {"pad"}


def test_a_loop_cut_to_its_paths_tempo_stays_a_loop():
    """A bass line or synth riff whose tempo only its pack's folder states ("Acme 125bpm/Bass/
    Lines"): the audio's loop holds when it lasts whole beats at that tempo, four or more, and
    its name states no note (a note bounced at exactly a bar is still a note)."""
    groove = lambda dur: (dur, 1, 5.3)  # noqa: E731
    line = "Acme Qzv 125bpm/Bass/Lines/Qzv Bass A-04.wav"            # 4 beats at 125: 1.92 s
    riff = "Acme Qzv 130bpm/Synth Q1/Qzv Riff C-02.wav"             # 16 beats at 130
    assert SH.audio_label(line, *groove(1.92)) == SH.audio_label(riff, *groove(16 * 60 / 130)) == "class.loop"
    assert SH.on_path_grid(line, 1.92) and SH.on_path_grid(riff, 16 * 60 / 130)
    assert not SH.on_path_grid(line, 1.95)                           # off the grid
    assert not SH.on_path_grid(line, 1.44)                           # three beats
    assert SH.audio_label(line, *groove(1.95)) == "class.oneshot"     # (as before)
    assert not SH.on_path_grid("Acme Qzv/Bass/Lines/Qzv Bass A-04.wav", 1.92)   # no tempo stated
    note = "Acme Qzv 140bpm/Bass/Multisample/Qzv Bass-E3.wav"        # a note bounced at one bar
    assert not SH.on_path_grid(note, 240 / 140)
    assert SH.audio_label(note, *groove(240 / 140)) == "class.oneshot"


def test_a_note_named_as_one_sound_stays_a_one_shot_on_the_grid():
    """A held note bounced at whole bars of its pack's tempo in a sound folder, whose name
    calls it one sound ("Stab", "Hit", "Chord", "Note"), stays a one-shot even when tremolo
    or a filter made the audio hear a loop; a loop word in the name ("Stab Loop") and a plural
    ("Chords", "Chord Stabs": a phrase of them) keep the loop."""
    dur = 8 * 60 / 124
    groove = (dur, 3, 2.5)
    for name in ("Qzv Synth Stab 03", "Qzv_Synth_Hit_2", "QzvZorblChord C", "Qzv Bass Note 4", "Qzv Synth One Shot 7"):
        rel = f"Acme Qzv 124bpm/Synths/{name}.wav"
        assert not SH.on_path_grid(rel, dur), rel
        assert "class.oneshot" in SH.shape_labels(rel) or SH.audio_label(rel, *groove) == "class.oneshot", rel
    for name in ("Qzv Synth 03", "Qzv Stab Loop 03", "Qzv Zorbl Chords", "Qzv Notes Riff 1",
                 "Qzv House Stabs 01", "Qzv Chord Stabs Am"):
        rel = f"Acme Qzv 124bpm/Synths/{name}.wav"
        assert SH.on_path_grid(rel, dur), rel
        assert "class.oneshot" not in SH.shape_labels(rel) and SH.audio_label(rel, *groove) == "class.loop", rel


def test_every_path_rule_names_a_canonical_label():
    from fourier.metadata.vocab import CANONICAL
    assert {lab for lab, _, _ in SH.path_rules()} <= CANONICAL


@pytest.fixture(params=["t.db", "t.duckdb"])
def db(tmp_path, request):
    """SQLite and DuckDB alike (DuckDB stores a descriptor's value as a 32-bit float)."""
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / request.param)
    rows = [("A/One Shots/Kick 1.wav", ["OneShot"], ["Perc Kicks"], 0.4, 1),
            ("A/Loops/Break 170bpm.wav", ["Loop"], [], 4.0, 16),
            ("A/One Shots/Snare 2.wav", ["OneShot"], ["Perc Kicks"], 0.3, 1),     # names disagree
            ("A/Misc/thing.wav", ["OneShot"], ["Tone Pads & Textures"], 1.2, 3)]
    with S.session_scope() as s:
        for i, (rel, cls, cats, dur, ev) in enumerate(rows, 1):
            s.add(Sample(id=i, path=f"/l/SampleLibrary/{rel}", rel_path=rel, filename=rel.rsplit("/", 1)[1],
                         duration_s=dur))
            s.add(SononymMeta(sample_id=i, classes=cls, categories=cats))
            s.add(SampleFeatures(sample_id=i, n_events=ev, tempo_bpm=170.0 if ev > 4 else None,
                                 bpm_reliable=ev > 4))
    with S.session_scope() as s:
        store.rebuild(s, store.SONONYM)
    yield
    S._engine = S._SessionLocal = None


def test_rules_version_is_exact_in_a_32_bit_float():
    """The "analysed" marker must read back as written, or every build relabels everything."""
    import numpy as np
    for p in ("path", "audio"):
        v = SH.rules_version(p)
        assert float(np.float32(v)) == v


def test_rebuild_and_agreement(db):
    with S.session_scope() as s:
        assert SH.ensure_current(s) == ["path", "audio"]
        assert SH.ensure_current(s) == []                               # counts match now
        bpm = s.execute(text("SELECT value FROM descriptors WHERE provider = 'path' AND name = 'bpm'")).all()
        assert [r[0] for r in bpm] == [170.0]
        a = SH.agreement(s, "path")
        assert a["samples"] == 4 and a["labelled"] == 3
        assert (a["class_compared"], a["class_agree"]) == (3, 3)
        assert (a["label_compared"], a["label_agree"]) == (2, 1)
        assert a["confused"] == [(("kick", "snare"), 1)]
        au = SH.agreement(s, "audio")
        assert (au["class_compared"], au["class_agree"]) == (3, 3) and au["label_compared"] == 0
        assert "label agrees 50.0% of 2" in SH.format_agreement(a)


def test_shadow_labels_change_no_votes(db):
    """Routing reads only the classifier's canonical labels (provider sononym)."""
    from fourier.metadata import rows as R
    R.forget_current()
    s = S.get_session()
    before = [r.canonical for r in R.fetch(s, R.sample_select("id", "canonical").order_by(text("id")))]
    SH.ensure_current(s)
    after = [r.canonical for r in R.fetch(s, R.sample_select("id", "canonical").order_by(text("id")))]
    assert before == after
    s.close()


def test_changed_rules_rebuild_the_labels(db, monkeypatch):
    """The "analysed" marker carries the rules version: new rules, new labels."""
    with S.session_scope() as s:
        assert SH.ensure_current(s) == ["path", "audio"]
        assert SH.ensure_current(s) == []
        real = SH.rules_version
        monkeypatch.setattr(SH, "rules_version", lambda p: real(p) + (1.0 if p == "path" else 0.0))
        assert SH.ensure_current(s) == ["path"]
        assert SH.ensure_current(s) == []


def _rows(s, provider):
    return s.execute(text("SELECT sample_id, kind, label, rank FROM labels WHERE provider = :p "
                          "ORDER BY 1, 2, 4"), {"p": provider}).all(), \
        s.execute(text("SELECT sample_id, name, value FROM descriptors WHERE provider = :p "
                       "ORDER BY 1, 2"), {"p": provider}).all()


@pytest.mark.parametrize("provider", ["path", "audio"])
def test_rebuild_reads_every_row_while_it_writes_in_chunks(db, monkeypatch, provider):
    """The rebuild writes its rows in chunks while it reads the samples. Every sample must
    still be read, on DuckDB too, where a write on the connection ends an open result."""
    with S.session_scope() as s:
        whole = SH.rebuild(s, provider)
        want = _rows(s, provider)
    monkeypatch.setattr(SH, "_CHUNK", 2)                # a write after every sample or two
    with S.session_scope() as s:
        assert SH.rebuild(s, provider) == whole
        assert _rows(s, provider) == want


@pytest.mark.slow
def test_audio_rebuild_past_the_real_chunk_on_duckdb(tmp_path):
    """The audio provider writes 3 descriptors a sample: 7,000 samples cross the real chunk."""
    S._engine = S._SessionLocal = None
    S.init_db(tmp_path / "big.duckdb")
    n = 7000
    try:
        with S.session_scope() as s:
            s.execute(text(
                "INSERT INTO samples (id, path, rel_path, filename, duration_s, scanned_at, "
                "is_favorite, is_hidden) "
                "SELECT i, '/l/Acme/Kicks/Kick ' || i || '.wav', 'Acme/Kicks/Kick ' || i || '.wav', "
                "'Kick ' || i || '.wav', 0.4, now(), false, false FROM range(1, :n + 1) t(i)"),
                {"n": n})
            s.execute(text("INSERT INTO sample_features (sample_id, n_events, harmonic_percussive_ratio) "
                           "SELECT i, 1, 0.5 FROM range(1, :n + 1) t(i)"), {"n": n})
        with S.session_scope() as s:
            got = SH.rebuild(s, "audio")
        assert got["descriptors"] == 3 * n
    finally:
        S._engine = S._SessionLocal = None
