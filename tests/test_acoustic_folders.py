"""Without Live, an acoustic instrument its folder names is a vote for ACOUSTIC (a path word
standing in for Live's instrument tag): whole words, and below the file's own name and
folder. A ringing note of a named instrument is one note however many onsets it has.
Invented names only."""
from types import SimpleNamespace

import numpy as np
import pytest

from fourier.packs import curate as C
from fourier.packs import vendors
from fourier.packs.curate_config import CATEGORIES


def test_an_instrument_folder_above_stands_in_for_its_tag():
    assert C._name_tags("Acme/Reeds/Tenor Saxophone/Vibrato/Qzv Tnr C3 v2.wav") == ["Sax"]
    assert C._name_tags("Acme/Mallets/Vibraphone/Hard Mallets/Qzv_C3_v2_rr1.wav") == ["Vibraphone"]
    assert C._name_tags("Acme/Qzv/Folk Harp/QZHarp_Normal_C3.wav") == ["Harp"]       # its own folder
    # the file's own name or folder names another sound: that speaks first
    assert C._name_tags("Acme/Tenor Saxophone/Pads/Qzv Pad 01.wav") == []
    assert C._name_tags("Acme/Tenor Saxophone/Qzv Kick 01.wav") == []
    # a loop takes no instrument from the folders above; words are whole words
    assert C._name_tags("Acme/Tenor Saxophone/Loops/Qzv 01.wav", loop=True) == []
    assert C._name_tags("Saxonia Qzv/Notes/Qzv 01.wav") == []
    assert C._name_tags("Acme/Qzv Vibes/Hits/Qzv 01.wav") == []
    # what a Live tag does: ACOUSTIC's, out of the one-shot categories' homing
    tags = C._name_tags("Acme/Reeds/Tenor Saxophone/Vibrato/Qzv Tnr C3 v2.wav")
    assert C._acoustic_tagged(tags, "Qzv Tnr C3 v2.wav", "Reeds")


def test_a_vendors_or_packs_name_never_names_the_instrument():
    """Only the folder just above the file's own is read, never a vendor's or pack's folder,
    and a folder of drums or hits speaks first: a kick in a pack named for a sax is no sax."""
    assert C._name_tags("Qzv Sax Vibes/Drums/QSV_01.wav") == []
    assert C._name_tags("Qzv Sax Vibes/QSV_01.wav") == []
    assert C._name_tags("Harp Qzv Records/Deep Qzv 124/Misc/DQ124_Misc_01.wav") == []
    assert C._name_tags("Acme/Qzv Sax Vibes/Misc/QSV_01.wav") == []                 # (the pack's folder)
    assert C._name_tags("Acme/Qzv/Sax/Drums/QSV_01.wav") == []                      # drums speak first
    assert C._name_tags("Acme/Qzv/Sax/Hits/QSV_01.wav") == []
    assert C._name_tags("Acme/Qzv/Sax Hits/Misc/QSV_01.wav") == []
    assert C._name_tags("Acme/Qzv/Tenor Saxophone/Takes/Qzv Run/QSV_01.wav") == []         # two folders up
    assert C._name_tags("Acme/Qzv/Tenor Saxophone/Takes/QSV_01.wav") == ["Sax"]


def test_the_vendor_and_pack_folders_follow_the_librarys_layout(monkeypatch):
    monkeypatch.setattr(vendors, "mode", lambda: vendors.AUTO)
    for layout, depth in ((vendors.TYPES, 0), (vendors.PACKS, 2)):
        monkeypatch.setattr(vendors, "_layout_for", lambda p, lay=layout: {"layout": lay, "umbrellas": ["Store"]})
        assert vendors.pack_depth("Acme/Qzv/Note.wav") == depth
    monkeypatch.setattr(vendors, "_layout_for", lambda p: {"layout": vendors.UMBRELLA, "umbrellas": ["Store"]})
    assert vendors.pack_depth("Store/Acme/Qzv/Note.wav") == 3 and vendors.pack_depth("Acme/Qzv/Note.wav") == 2
    monkeypatch.setattr(vendors, "_layout_for", lambda p: {"layout": vendors.TYPES, "umbrellas": []})
    assert C._name_tags("Tenor Saxophone/Vibrato/Qzv 01.wav") == ["Sax"]


def test_a_synth_patch_or_a_loop_takes_no_instrument_tag():
    """A synth patch filed under the instrument it imitates, a loop named for an instrument:
    no instrument tag, so nothing reserves them for ACOUSTIC (which refuses a synth's name)."""
    assert C._name_tags("Acme Qzv Lib/Sax/Synth Lead 03.wav") == []
    assert C._name_tags("Acme/Qzv/Sax/Vibrato/Qzv FM Lead 03.wav") == []
    assert C._name_tags("Acme/Qzv/Synth Sax/Qzv 03.wav") == []
    assert C._name_tags("Acme/Sax Loops/Sax Loop 01 120bpm.wav", loop=True) == []
    assert C._name_tags("Acme/Loops/Sax Riff 01 120bpm.wav", loop=True) == []
    assert C._name_tags("Acme/Loops/Sax Riff 01 120bpm.wav") == ["Sax"]
    # a ringing note the audio alone calls a loop keeps its instrument when its name states
    # its note and its path no loop word or tempo
    assert C._name_tags("Acme/Qzv/Folk Harp/QZHarp_Normal_C4_v2.wav", loop=True) == ["Harp"]
    assert C._name_tags("Acme/Qzv/Tenor Saxophone/Vibrato/Qzv_Vib_C4_v2.wav", loop=True) == ["Sax"]
    assert C._name_tags("Acme/Qzv/Tenor Saxophone/Vibrato/Qzv Vib 01.wav", loop=True) == []
    assert C._name_tags("Acme/Qzv/Folk Harp/QZHarp 120bpm C4.wav", loop=True) == []
    assert C._name_tags("Acme/Qzv/Harp Loops/QZHarp C4.wav", loop=True) == []
    # bowed strings as before: a loop of them keeps its tag
    assert C._name_tags("Acme/Loops/Violin Phrase 01 120bpm.wav", loop=True) == ["Violin"]


def _row(i, rel, **kw):
    base = dict(id=i, rel_path=rel, path=f"/l/{rel}", filename=rel.rsplit("/", 1)[-1],
                file_hash=f"h{i}", file_format="wav", duration_s=4.0, brightness=0.5,
                noisiness=0.3, harmonicity=0.9, crest_factor=4.3, base_note=None,
                base_note_confidence=None, attack_time_ms=5.0, decay_time_ms=3000.0,
                sub_weight=0.0, is_clipped=0, tempo_bpm=None, bpm_reliable=0,
                onset_rate_hz=3.0, chroma_concentration=3.0, spectral_flatness_mean=0.1,
                rms_mean=0.2, dc_offset_ratio=0.0, n_events=1, classes=["OneShot"], categories=[],
                canonical=["class.oneshot"])
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def no_ratings(monkeypatch):
    from fourier.packs import ratings as R
    monkeypatch.setattr(R, "drop_set", lambda *a, **k: set())
    monkeypatch.setattr(R, "drop_categories", lambda *a, **k: {})
    monkeypatch.setattr(R, "rating_scopes", lambda *a, **k: {})
    monkeypatch.setattr(R, "misfiled_map", lambda *a, **k: {})
    C._REACH.clear()
    monkeypatch.setattr(C, "_misfiled_index", lambda m, *a, **k: C._MisfiledIndex(m))


def test_a_ringing_note_of_a_named_instrument_is_one_note(no_ratings):
    """A vibraphone's tremolo or a harp's ring gives a dozen onsets on one note: without
    Sononym, one event between silences of an instrument named by its path is one note; the
    same sound whose path names no instrument still meets the single-note gate."""
    rows = [_row(1, "Acme/Mallets/Vibraphone/Hard Mallets/Qzv_C3_v2_rr1.wav"),
            _row(2, "Acme/Mallets/Qzv/Hard Mallets/Qzv_D3_v2_rr1.wav")]
    rows = [r.__class__(**{**vars(r), "ableton_tags": C._name_tags(r.rel_path)}) for r in rows]
    E = np.eye(2, 8, dtype="float32")
    why: dict = {}
    rec, _fm, _st = C._select_records(rows, CATEGORIES["ACOUSTIC"], E, {1: 0, 2: 1}, category="ACOUSTIC",
                                      why=why, fallback=True)
    assert [r["id"] for r in rec] == [1]
    assert set(why.get("many_onsets", {})) == {2}


PHRASES = (  # (path, the audio's label, onsets): played phrases named for an instrument
    ("Qzv Harp Textures/Harp/Harp Gliss C3.wav", "loop", 9),
    ("Acme Studio/Qzv Strings 2/Harp/Phrases/Rising Figure C3.wav", "loop", 15),
    ("Acme Qzv Session/Sax/Sax Phrase A1.wav", "oneshot", 10),
    ("Acme/Qzv Mallets/Vibraphone/Qzv Arp C4.wav", "oneshot", 8),
    ("Acme/Qzv Mallets/Vibraphone/Vibraphone Arp C4.wav", "oneshot", 8),    # a name override
)


def test_a_phrase_named_for_an_instrument_is_no_ringing_note(no_ratings):
    """A glissando, a figure in a phrases folder, a sax line, an arpeggio: one event between
    silences, many onsets. A phrase word in the name or a folder keeps them out of the
    ringing-note exception, the one-note exemption and an instrument's name override, so the
    single-note gate drops them; a ringing note with the same onsets still goes in."""
    for rel, label, _n in PHRASES:
        if label == "loop":
            assert C._name_tags(rel, loop=True) == [], rel
    rows = [_row(i, rel, onset_rate_hz=n / 4.0,
                 classes=["Loop" if label == "loop" else "OneShot"], canonical=[f"class.{label}"],
                 ableton_tags=C._name_tags(rel, loop=label == "loop"))
            for i, (rel, label, n) in enumerate(PHRASES, 1)]
    note = "Acme/Qzv Mallets/Vibraphone/Qzv C4.wav"
    rows.append(_row(9, note, onset_rate_hz=2.0, ableton_tags=C._name_tags(note)))
    E = np.eye(len(rows), 8, dtype="float32")
    rec, _fm, _st = C._select_records(rows, CATEGORIES["ACOUSTIC"], E, {r.id: k for k, r in enumerate(rows)},
                                      category="ACOUSTIC", why={}, fallback=True)
    assert [r["id"] for r in rec] == [9]
