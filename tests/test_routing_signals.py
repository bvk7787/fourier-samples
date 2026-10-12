"""Routing where the name alone misleads, so other signals decide: a pack's drum-abbreviation
prefix against the folder that names the drum, the word "Oh" and a voice folder's "oh", a drum
word beside a tonal sound's word (rules._drum_named_in: compute_homes, why and verify); loops of
bass, music or effects that Live tags "Drum Loop", and sample chains, out of the drum loops
(rules._not_drum_loop, SAMPLE_CHAIN_PATH_RE); an FX folder's lasers and noise out of PADS
(rules._fx_folder_veto); and a Live Set's recordings as demo paths."""
import re

import pytest
import yaml

from fourier.packs import curate as C
from fourier.packs.curate_config import CATEGORIES, DEMO_PATH, DRUMLOOP_STYLE_EXCLUDE, SAMPLE_CHAIN_PATH_RE

from .test_curate import _loops
from .test_routing_names import DRUM_CHECK, _row, _verify_fails
from .test_verify import _clean, _hit


# --- a pack's prefix and the folder that names the drum ------------------------------------

@pytest.mark.parametrize("rel, want", [
    ("Acme/Kit/Vqx Tom/sd_qz030.wav", "TOMS"),
    ("Acme/Kit/Hat Closed Vqx/sd_qz105.wav", "HATS"),
    ("Acme/Kit/Kick (Vqx)/sd_zq090.wav", "KICKS"),
    ("Acme/Kit/Samples/Vqx Tom/sd_qz030.wav", "TOMS"),
    ("Acme/Kit/Snare Vqx/bd_qz12.wav", "SNARES"),
    # no drum in the folder: the abbreviation still names it
    ("Acme/Kit/Vqx/sd_qz030.wav", "SNARES"),
    # no letters+digits code after it: no prefix
    ("Acme/Kit/Vqx Tom/sd_01.wav", "SNARES"),
    ("Acme/Kit/Vqx Tom/SD Qzv 3.wav", "SNARES"),
])
def test_a_pack_prefix_yields_to_the_folder_that_names_the_drum(rel, want):
    fn = rel.rsplit("/", 1)[-1]
    assert C._drum_named_in(fn, rel) == want, rel


# --- "Oh" and a voice folder ---------------------------------------------------------------

def test_oh_is_a_word_unless_written_as_the_abbreviation():
    assert C._drum_named("Qzv_Oh_Vqx_D4.wav") is None
    assert C._drum_named("Vqx_Oh_Zq_1.wav") is None
    assert C._drum_named("OH Qzv 2.wav") == "HATS" and C._drum_named("oh_3.wav") == "HATS"
    assert C._drum_named("Qzv OH.wav") == "HATS"


@pytest.mark.parametrize("rel, want", [
    ("Acme/Pack/Speech Qzv/oh 7.wav", None),
    ("Acme/Pack/Vocals Vqx/SD 4.wav", None),
    ("Acme/Pack/Hats/oh 7.wav", "HATS"),
    ("Acme/Pack/Qzv/oh 7.wav", "HATS"),
    ("Acme/Pack/Speech Qzv/Hat Qzv Vqx.wav", "HATS"),        # a drum word, not an abbreviation
])
def test_an_abbreviation_in_a_voice_folder_is_a_word(rel, want):
    assert C._drum_named_in(rel.rsplit("/", 1)[-1], rel) == want, rel


# --- a drum word beside a tonal sound's word ---------------------------------------------------

BASS = ({"class.oneshot", "bass"}, ["One Shot", "Synth Bass"])
PAD = ({"class.oneshot", "lead"}, ["One Shot", "Pad"])
SNARE = ({"class.oneshot", "snare"}, ["One Shot", "Snare Hit"])


@pytest.mark.parametrize("fn, votes, want", [
    ("Qzv_D3_Clap_Bass_Vqx.wav", (None, None), None),          # a note and a bass word
    ("Qzv_D3_Clap_Bass_Vqx.wav", SNARE, None),
    ("Vqx String Snares Zq.wav", PAD, None),                        # both classifiers: no drum
    ("Vqx String Snares Zq.wav", (None, None), "SNARES"),           # ...not when they don't speak
    ("Vqx String Snares Zq.wav", SNARE, "SNARES"),
    ("Vqx String Snares Zq.wav", ({"class.oneshot", "lead"}, ["One Shot"]), "SNARES"),   # Live: no category
    # the drum's own word isn't the tonal one, and sub / synth aren't tonal words here
    ("Bass SDS 800 Qzv F1.wav", BASS, "KICKS"),
    ("Bass Drum Qzv C1.wav", BASS, "KICKS"),
    ("BD Vqx Sub A0.wav", BASS, "KICKS"),
    ("Tom Vqx Synth D2.wav", BASS, "TOMS"),
    ("BD Zq 7.wav", BASS, "KICKS"),                            # classifiers alone: the name wins
    ("Clap Vqx Pad 2.wav", SNARE, "CLAPS"),
])
def test_a_drum_word_beside_a_tonal_word(fn, votes, want):
    assert C._drum_named_in(fn, f"Acme/Pack/Vqx/{fn}", False, *votes) == want, fn


def test_why_and_verify_read_the_classifiers_as_the_build_does(tmp_path):
    from fourier.packs.why import explain_row
    w = explain_row(_row("Vqx String Snares Zq.wav", *PAD), store_path=str(tmp_path / "none.json"))
    assert "drum_name" not in {m[0] for m in w.matched}, w.matched
    w = explain_row(_row("Vqx String Snares Zq.wav", *SNARE), store_path=str(tmp_path / "none.json"))
    assert w.decided == ("drum_name", "SNARES"), w.matched
    labels = {"Vqx String Snares Zq.wav": (PAD[0], PAD[1])}
    files = _clean() + [("PADS", "dark", "Vqx String Snares Zq.wav", _hit(), "V4/P/Vqx String Snares Zq.wav")]
    assert DRUM_CHECK not in _verify_fails(tmp_path / "a", files, labels)
    assert DRUM_CHECK in _verify_fails(tmp_path / "b", files, {})       # no classifier word: the name


# --- loops of bass, music or effects; sample chains --------------------------------------------------

@pytest.mark.parametrize("rel, out", [
    ("Acme/Qzv/Bass Loops/Vqx_131_A#_Sub_Bass.wav", True),
    ("Acme/Qzv/Bass Loops Vqx/Vqx 133 F Bassline.wav", True),
    ("Acme/Drum & Bass/Bass Loops/Vqx_168_BMajor_Bassline.wav", True),
    ("Acme/Qzv/Music Loops/Vqx_118_Riff_Dmin.wav", True),
    ("Acme/Qzv/Synth_Loops/Vqx_127_Seq_A.wav", True),
    ("Acme/Qzv/Loops/Vqx_171_E_Bass_Wobble.wav", True),        # a key and a bass in the name
    ("Acme/Qzv/Effects Vqx/Vqx_121_Swell.wav", True),
    ("Acme/Qzv/Vqx Hits + FX/Vqx Ring Loop 4.wav", True),
    # a drum & bass pack's bass loops folder holding breaks; genre names; drum folders
    ("Acme/Drum & Bass/Bass Loops/Vqx_172_Brk_Roll.wav", False),
    ("Acme/Drum & Bass/Bass Loops/Vqx_169_Tom_Fill.wav", False),
    ("Acme/Qzv/Bass Loops/Vqx Bass Drum 121.wav", False),
    ("Acme/Qzv/Loops/Vqx Drum n Bass 166 A Break.wav", False),
    ("Acme/Qzv/Drum Loops/Vqx_122_G_Groove.wav", False),
    ("Acme/Qzv/Vqx FX Drums/Vqx_92_Beat.wav", False),
    ("Acme/Qzv/Drum Loops/Vqx Risers/Vqx Build 127bpm.wav", False),
])
def test_loops_that_are_no_drum_loops(rel, out):
    assert C._not_drum_loop(rel) is out, rel


@pytest.mark.parametrize("rel, chain", [
    ("Acme/Qzv/Vqx Slice Grid/vqx set 16.wav", True),
    ("Acme/Pack/Vqx/Grid/Qzv_g16.wav", True),
    ("Acme/Pack/Qzv_g32.wav", True),
    ("Acme/Qzv/Vqx Sliced Library/Vqx Rims 32 171bpm.wav", True),
    ("Acme/Qzv/Sample Chains/Vqx 64.wav", True),
    ("Acme/Qzv/Vqx (no grid)/vqx set.wav", False),
    ("Acme/Pack/Grid Kicks/Qzv.wav", False),
    ("Acme/Pack/Loops/Qzv_G16_break.wav", True),
    ("Acme/Pack/Loops/Qzv G4 Pad.wav", False),
    ("Acme/Pack/Sliced/Qzv 120/Qzv 001.wav", False),
])
def test_sample_chain_paths(rel, chain):
    assert bool(SAMPLE_CHAIN_PATH_RE.search(rel)) is chain, rel


def test_the_drum_loops_leave_out_bass_loops_and_sample_chains(monkeypatch):
    rows, E, id2row = _loops(["Break 1", "Break 2", "Break 3", "Break 4"], monkeypatch)
    rows[1].rel_path = "A/Vqx Bass Loops/Break 2 F# Sub Bass 124bpm.wav"
    rows[2].rel_path = "A/Vqx Slice Grid/Break 3 124bpm.wav"
    why = {}
    rec, _fm, st = C._select_records(rows, CATEGORIES["DRUMLOOPS"], E, id2row, category="DRUMLOOPS", why=why)
    assert {r["id"] for r in rec} == {1, 4}
    assert set(why["nondrum_gated"]) == {2} and set(why["chain_gated"]) == {3}
    assert st["nondrum_gated"] == 1 and st["chain_gated"] == 1


def test_style_exclusions_read_any_separator():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "config" / "presets"
    pats = [DRUMLOOP_STYLE_EXCLUDE] + [
        re.compile(yaml.safe_load((root / f"{p}.yaml").read_text())["advanced"]["DRUMLOOP_STYLE_EXCLUDE"], re.I)
        for p in ("breaks-acid", "trap", "hiphop-lofi")]
    for ex in pats:
        for rel in ("Acme/Qzv/Loops/Vqx_Synth_Bass_A.wav", "Acme/Qzv/Vqx_FX_Loops/Vqx 3.wav",
                    "Acme/Pack/Synth-Bass/Qzv.wav", "Acme/Pack/FX Loops/Riser.wav"):
            assert ex.search(rel), (ex.pattern, rel)
        assert not ex.search("Acme/Qzv/Drum Loops/Vqx Synthesis Bassline Break.wav")


# --- an FX folder's sounds out of PADS ---------------------------------------------------------

@pytest.mark.parametrize("canon, rel, pads", [
    ({"class.loop", "fx.sweep"}, "Acme/FX/Vqx Effects/Vqx Uplifters/Vqx Up 3.wav", False),
    ({"class.oneshot"}, "Acme/Qzv/Vqx FX/Vqx Drop 2.wav", False),
    ({"class.loop", "fx.noise", "fx.nature"}, "Acme/Qzv/Noise/Vqx Hiss 5.wav", False),
    # Sononym hears a pad, or the folder isn't an FX folder: an atmosphere stays
    ({"class.oneshot", "pad"}, "Acme/FX/Vqx Air/Vqx Air 4.wav", True),
    ({"class.oneshot", "fx.nature", "pad"}, "Acme/FX/Vqx Air/Vqx Air 5.wav", True),
    ({"class.oneshot", "fx.nature"}, "Acme/Qzv/Qzv Atmospheres/Vqx Haze.wav", True),
    ({"class.oneshot", "fx.noise"}, "Acme/Qzv/Vqx Effected/Vqx Wet 9.wav", True),
])
def test_an_fx_folder_sound_the_classifier_calls_fx_is_no_pad(canon, rel, pads):
    got = C._fx_folder_veto({"PADS", "FX"}, canon, rel)
    assert ("PADS" in got) is pads and "FX" in got, rel
    assert C._fx_folder_veto({"SYNTH"}, canon, rel) == {"SYNTH"}


# --- a Live Set's recordings -------------------------------------------------------------------

@pytest.mark.parametrize("rel, demo", [
    ("Acme/Qzv Set/Samples/Recorded/QZV 0004 [2021-05-06 101112].wav", True),
    ("Acme/Qzv Set Project/Samples/Recorded/1-Audio 0003.aif", True),
    ("Acme/Pack/Recorded Drums/Kick 01.wav", False),
    ("Acme/Pack/Samples/Recordings/Kick 01.wav", False),
])
def test_live_set_recordings_are_demo_paths(rel, demo):
    assert bool(DEMO_PATH.search(rel)) is demo, rel
