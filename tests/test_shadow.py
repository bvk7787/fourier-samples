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
