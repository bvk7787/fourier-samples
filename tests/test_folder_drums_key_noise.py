"""A drum hit named only by its folder ("Drum Hits/017.wav", rules._drum_folder_wins) isn't an
acoustic or organ instrument's, and a key or pedal noise recorded beside an instrument's notes
("Key Release", "Key Press", rules._key_noise_named) stays out of the tonal categories: the
build (compute_homes, _select_records), why and verify."""
import re

import numpy as np
import pytest

from fourier.packs import curate as C
from fourier.packs.curate_config import CATEGORIES, DRUM_HITS_FOLDER_RE

from .test_curate import _row as _srow
from .test_routing_names import _verify_fails
from .test_verify import _clean, _hit

TOM = ({"class.oneshot", "tom"}, ["One Shot", "Timpani"])


@pytest.mark.parametrize("folder, hit", [
    ("Drum Hits", True), ("Drumhit", True), ("DRUM_HITS", True), ("Drums", True), ("05. Drums", True),
    ("Drum One Shots", True), ("Drum Kits", True), ("QX DRUM HITS", True), ("One Shots Drums", True),
    # a tonal drum's folder, loops, a synth patch, a folder naming one kind of drum or a hand drum
    ("Steel Pan Hits", False), ("Steel Drum Hits", False), ("Tongue Drum", False), ("Slit Drum Hits", False),
    ("Drum Loops", False), ("Drum Machine", False), ("Kick Drums", False), ("Hand Drums", False),
    ("Drumulatqz", False),
])
def test_drum_hit_folders(folder, hit):
    # (read with "_" and "-" as spaces, as rules._drum_folder_wins does)
    assert bool(DRUM_HITS_FOLDER_RE.search(re.sub(r"[_-]+", " ", folder))) is hit, folder


def test_a_drum_hit_named_by_its_folder_only():
    W = C._drum_folder_wins
    canon, tags = TOM
    assert W("017_QXV.WAV", canon, tags, "Acme/Qzv Loops/Drumhit/017_QXV.WAV") == "TOMS"
    assert W("017.wav", canon, tags, "Acme/Qzv/Drum_Hits/WAV/017.wav") == "TOMS"    # above a generic folder
    # the name names the instrument, no classifier says drum, a vocal vote, or another folder
    assert W("Timpani Qzv 3.wav", canon, tags, "Acme/Perc/Drums/Timpani Qzv 3.wav") is None
    assert W("017.wav", {"class.oneshot", "lead"}, ["One Shot", "Timpani"], "Acme/Drums/017.wav") is None
    assert W("017.wav", {"class.oneshot", "tom", "vocal"}, tags, "Acme/Drums/017.wav") is None
    assert W("017.wav", canon, tags, "Acme/Qzv/Steel Drum Hits/017.wav") is None
    assert W("017.wav", canon, tags, "Acme/Qzv Hits/017.wav") is None
    # the more votes win among several drums
    assert W("05.wav", {"class.oneshot", "tom", "kick"}, ["One Shot", "Kick"], "Acme/Drums/05.wav") == "KICKS"


def test_acoustic_and_piano_leave_a_folder_named_drum_hit(monkeypatch):
    monkeypatch.setattr(C, "embed_text", lambda t: np.ones(512, dtype="float32"))     # no CLAP model
    canon, tags = TOM
    rows = [_srow(id=1, rel_path="Acme/Qzv/Drum Hits/017_QXV.WAV", path="/lib/Acme/Qzv/Drum Hits/017_QXV.WAV",
                  filename="017_QXV.WAV", classes=["OneShot"], categories=[], canonical=sorted(canon),
                  ableton_tags=tags, n_events=1, duration_s=0.6)]
    E = np.ones((1, 512), dtype="float32") / np.sqrt(512)
    for cat in ("ACOUSTIC", "PIANO"):
        why = {}
        C._select_records(rows, CATEGORIES[cat], E, {1: 0}, category=cat, why=why)
        assert why.get("drum_folder") == {1: "TOMS"}, why


def test_why_homes_a_folder_named_drum_hit_with_the_drums(tmp_path):
    from fourier.packs.why import explain_row

    from .test_routing_names import _Row, HOME_FIELDS
    canon, tags = TOM

    def row(rel):
        d = dict.fromkeys(HOME_FIELDS)
        d.update(id=1, path=f"/lib/SampleLibrary/{rel}", rel_path=rel, filename=rel.rsplit("/", 1)[-1],
                 classes=["OneShot"], canonical=sorted(canon), ableton_tags=list(tags), categories=[],
                 duration_s=0.4, n_events=1)
        return _Row(**d)
    w = explain_row(row("Acme/Qzv/Drum_Hits/017_QXV.WAV"), store_path=str(tmp_path / "none.json"))
    assert w.decided == ("drum_folder", "TOMS"), w.matched
    w = explain_row(row("Acme/Qzv/Timpani/017_QXV.WAV"), store_path=str(tmp_path / "none.json"))
    assert w.decided[1] == "ACOUSTIC", w.matched


DRUM_CHECK = "files named as a drum (or a scratch) live in that category"


def test_verify_a_folder_named_drum_hit_in_acoustic(tmp_path):
    labels = {"017_QXV.wav": TOM}
    bad = _clean() + [("ACOUSTIC", "mallet", "017_QXV.wav", _hit(), "V4/P/Drum Hits/017_QXV.wav")]
    assert DRUM_CHECK in _verify_fails(tmp_path / "a", bad, labels)
    ok = _clean() + [("ACOUSTIC", "mallet", "017_QXV.wav", _hit(), "V4/P/Timpani/017_QXV.wav")]
    assert DRUM_CHECK not in _verify_fails(tmp_path / "b", ok, labels)


# --- key and pedal noises ---------------------------------------------------------------------

@pytest.mark.parametrize("fn, noise", [
    ("Vqx Keyreleases 3.wav", True), ("Vqx_Keypresses_7.wav", True),
    ("QzvKeyreleases_C3.wav", True), ("Key Release 3.wav", True), ("Key_Press_2.wav", True), ("Grand Key Off 3.wav", True), ("Key Click 02.wav", True),
    ("Pedal Noise Up.wav", True), ("Qzv Rel Noise.wav", True), ("Vqx Release Noise b.wav", True),
    ("Damper_Noise.wav", True), ("Qzv KeyRel 1.wav", True),
    # a note's release take, an organ's pedal notes, a keyboard or a "monkey"
    ("Vqx Ocarina G4 rel.wav", False), ("Qzv Rel A5.wav", False), ("Vqx_Oboe_E5_rel.wav", False),
    ("Qzv_PedalSoft_41.wav", False), ("Vqx Pedal G1.wav", False), ("Keyboard Noise.wav", False),
    ("Monkey Press.wav", False), ("Key Lead C3.wav", False), ("Qzv Release C4.wav", False),
])
def test_key_noise_names(fn, noise):
    assert C._key_noise_named(fn) is noise, fn


def test_the_tonal_categories_leave_a_key_noise(monkeypatch):
    monkeypatch.setattr(C, "embed_text", lambda t: np.ones(512, dtype="float32"))     # no CLAP model
    rows = [_srow(id=1, rel_path="Acme/Qzv/Vqx Keyreleases 3.wav",
                  path="/lib/Acme/Qzv/Vqx Keyreleases 3.wav", filename="Vqx Keyreleases 3.wav",
                  classes=["OneShot"], categories=[], canonical=["class.oneshot"], ableton_tags=["Organ"],
                  n_events=1, duration_s=0.5)]
    E = np.ones((1, 512), dtype="float32") / np.sqrt(512)
    why = {}
    C._select_records(rows, CATEGORIES["PIANO"], E, {1: 0}, category="PIANO", why=why)
    assert why.get("key_noise") == {1: None}, why
    from fourier.packs.why_log import FILTER_WORDS, describe
    assert "key_noise" in FILTER_WORDS and "drum_folder" in FILTER_WORDS
    assert "not a note" in describe("key_noise", None, {"category": "PIANO"})
    assert "TOMS" in describe("drum_folder", "TOMS", {"category": "ACOUSTIC"})


@pytest.mark.parametrize("fn, rel, noise", [
    # release and pedal takes in a folder that holds them
    ("Qzv_rel_rr1_B3.wav", "Acme/Upright Qzv/Releases/Qzv_rel_rr1_B3.wav", True),
    ("Qzv Rel A5.wav", "Acme/Qzv Keys/Rel/Qzv Rel A5.wav", True),
    ("Qzv_rel_rr2_B3.wav", "Acme/Upright Qzv/Mechanical/Qzv_rel_rr2_B3.wav", True),
    # a "Pedal" or "Noise" folder alone doesn't say: an organ's pedal notes, a synth's noise
    # patch, and a piano's pedal takes (measured against the notes beside the folder instead)
    ("Ped_Subbass16_C2.wav", "Acme/Qzv Organ/Pedal/Ped_Subbass16_C2.wav", False),
    ("Pedal Bourdon C2.wav", "Acme/Qzv Organ/Pedal/Pedal Bourdon C2.wav", False),
    ("Noise Lead C4.wav", "Acme/Qzv Synth/Noise/Noise Lead C4.wav", False),
    ("Qzv_PedOff_002.wav", "Acme/Upright Qzv/Pedal/Off/Qzv_PedOff_002.wav", False),
    # beside the notes, a note's release take stays a note; a note in a "Releases" folder too
    ("Qzv Rel A5.wav", "Acme/Qzv Keys/Qzv Rel A5.wav", False),
    ("Vqx Ocarina G4 rel.wav", "Acme/Ocarina/Vqx Ocarina G4 rel.wav", False),
    ("Qzv_vl1_rr1_B3.wav", "Acme/Upright Qzv/Releases/Qzv_vl1_rr1_B3.wav", False),
])
def test_release_takes_in_their_folder_are_key_noise(fn, rel, noise):
    assert C._key_noise_named(fn, rel) is noise, rel


def test_a_release_take_far_under_its_folders_notes_is_key_noise(monkeypatch):
    """Beside the notes, a release take 20 dB or more under its folder's notes is the key's
    or damper's noise and stays out of PIANO; one at the notes' level is a note."""
    monkeypatch.setattr(C, "embed_text", lambda t: np.ones(512, dtype="float32"))
    notes = [_srow(id=i, rel_path=f"Acme/Qzv/Qzv_vl1_{n}.wav", path=f"/lib/Acme/Qzv/Qzv_vl1_{n}.wav",
                   filename=f"Qzv_vl1_{n}.wav", rms_mean=0.1, classes=["OneShot"], categories=[],
                   canonical=["class.oneshot"], ableton_tags=["Piano"], n_events=1, duration_s=2.0)
             for i, n in enumerate(("C3", "E3", "G3"), start=1)]
    quiet = _srow(id=4, rel_path="Acme/Qzv/Qzv_rel_B3.wav", path="/lib/Acme/Qzv/Qzv_rel_B3.wav",
                  filename="Qzv_rel_B3.wav", rms_mean=0.005, classes=["OneShot"], categories=[],
                  canonical=["class.oneshot"], ableton_tags=["Piano"], n_events=1, duration_s=2.0)
    loud = _srow(**{**vars(quiet), "id": 5, "rms_mean": 0.08, "filename": "Qzv_rel_D3.wav",
                    "path": "/lib/Acme/Qzv/Qzv_rel_D3.wav", "rel_path": "Acme/Qzv/Qzv_rel_D3.wav"})
    rows = notes + [quiet, loud]
    E = np.ones((5, 512), dtype="float32") / np.sqrt(512)
    why = {}
    C._select_records(rows, CATEGORIES["PIANO"], E, {i: i - 1 for i in range(1, 6)}, category="PIANO", why=why)
    assert why.get("key_noise") == {4: None}, why


def test_pedal_and_noise_folders_go_by_the_notes_beside_them(monkeypatch):
    """A take in a "Pedal" or "Noise" folder is key noise only when it's far under the notes
    of the folders beside that one: a piano's pedal noises leave PIANO, an organ's pedal
    notes (as loud as its manual's) and a synth's noise patch stay."""
    monkeypatch.setattr(C, "embed_text", lambda t: np.ones(512, dtype="float32"))

    def row(i, rel, rms):
        return _srow(id=i, rel_path=rel, path=f"/lib/{rel}", filename=rel.rsplit("/", 1)[1],
                     rms_mean=rms, classes=["OneShot"], categories=[], canonical=["class.oneshot"],
                     ableton_tags=["Piano"], n_events=1, duration_s=2.0)
    rows = [row(i, f"Acme/Upright Qzv/Sustain/Qzv_vl1_{n}.wav", 0.1) for i, n in enumerate(("C3", "E3", "G3"), 1)]
    rows += [row(4, "Acme/Upright Qzv/Pedal/Off/Qzv_PedOff_002.wav", 0.004)]
    rows += [row(i, f"Acme/Qzv Organ/Great/Qzv Great {n}.wav", 0.1) for i, n in zip((5, 6, 7), ("C3", "E3", "G3"))]
    rows += [row(8, "Acme/Qzv Organ/Pedal/Ped_Subbass16_C2.wav", 0.12)]
    rows += [row(i, f"Acme/Qzv Synth/Lead/Qzv Lead {n}.wav", 0.1) for i, n in zip((9, 10, 11), ("C3", "E3", "G3"))]
    rows += [row(12, "Acme/Qzv Synth/Noise/Noise Lead C4.wav", 0.09)]
    E = np.ones((len(rows), 512), dtype="float32") / np.sqrt(512)
    why = {}
    C._select_records(rows, CATEGORIES["PIANO"], E, {r.id: r.id - 1 for r in rows}, category="PIANO", why=why)
    assert why.get("key_noise") == {4: None}, why


KEY_CHECK = "no key or pedal noises in the tonal categories"


def test_verify_a_key_noise_in_piano(tmp_path):
    bad = _clean() + [("PIANO", "organ", "Vqx_Keypresses_7.wav", _hit(), "V4/P/Vqx Keypresses 7.wav")]
    assert KEY_CHECK in _verify_fails(tmp_path / "a", bad, {})
    ok = _clean() + [("PIANO", "organ", "Vqx_Ocarina_G4_rel.wav", _hit(), "V4/P/Vqx Ocarina G4 rel.wav")]
    assert KEY_CHECK not in _verify_fails(tmp_path / "b", ok, {})
