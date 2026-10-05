"""Routing and naming without Sononym, for libraries a real install brought: lo-fi drum
loops, loops whose tempo only their length states, keys named by their folder, sustained pads
in a flat folder, 808s named by a bare note, shakers in a percussion folder, near ties between
claps and hats, and drum-loop folders that aren't musical. Each acts only in a build without
Sononym (or behind a preset or a device limit), so a build with Sononym and Live is unchanged
(the synthetic golden proves it end to end)."""
from collections import namedtuple

import numpy as np
import pytest

from fourier.metadata import shadow as SH
from fourier.packs import curate as C
from fourier.packs import naming as N

Row = namedtuple("Row", "id path rel_path filename duration_s classes canonical bpm tempo_bpm "
                        "acid_bpm acid_beats ableton_tags categories")


def _row(rel, dur=10.0, tempo=None, canon=("class.loop",), classes=("Loop",)):
    return Row(1, f"/lib/SampleLibrary/{rel}", rel, rel.rsplit("/", 1)[-1], dur, list(classes),
               list(canon), None, tempo, None, None, None, [])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# --- 6: lo-fi drum loops -------------------------------------------------------------------
def test_hiphop_lofi_takes_more_tonal_drum_loops(tmp_path):
    from fourier import layers

    def values(preset):
        p = tmp_path / f"{preset}.toml"
        p.write_text(f'preset = "{preset}"\n')
        return layers.resolve(config=str(p)).values
    assert layers.defaults()["curate_config.DRUMLOOP_HAR_MAX"] == 0.68          # today's gate
    assert values("hiphop-lofi")["curate_config.DRUMLOOP_HAR_MAX"] > 0.7
    assert "curate_config.DRUMLOOP_HAR_MAX" not in values("breaks-acid")
    assert "curate_config.DRUMLOOP_HAR_MAX" not in values("balanced")


def test_a_phrase_marked_loop_that_sounds_like_drums_is_rehomed(monkeypatch):
    """A loop the phrase rule homes in PHRASES, whose CLAP scores fail PHRASES' gate (it is
    closer to the drum breaks) and clearly pass a drum-loop category's, homes there."""
    e = np.eye(6)
    anchors = {"PHRASES": e[0], "DRUMLOOPS": e[1]}
    anti = {"PHRASES": e[1], "DRUMLOOPS": e[2]}
    monkeypatch.setattr(C, "_clap_anti", lambda cat: anti.get(cat))
    drums = _unit(0.15 * e[0] + 0.6 * e[1] + 0.1 * e[2] + 0.6 * e[5])
    assert C._clap_rehome(drums, anchors) == "DRUMLOOPS"
    assert C._clap_rehome(drums, anchors, misfiled={"DRUMLOOPS"}) is None
    music = _unit(0.6 * e[0] + 0.3 * e[1] + 0.7 * e[5])               # PHRASES' gate takes it
    assert C._clap_rehome(music, anchors) is None
    unclear = _unit(0.1 * e[0] + 0.32 * e[1] + 0.3 * e[2] + 0.9 * e[5])  # not clear of its anti
    assert C._clap_rehome(unclear, anchors) is None


def test_a_rehomed_loop_skips_the_harmonicity_gate_only():
    r = namedtuple("R", "harmonicity filename duration_s tempo_bpm bpm")(0.75, "Loop 85 BPM.wav", 11.3, None, None)
    cfg = dict(har_max=0.68, bpm_min=60, bpm_max=200, require_tempo=True)
    assert C._after_gate(r, cfg)[0] == "too_harmonic"
    assert C._after_gate(r, cfg, har=False) is None


# --- 7: a tempo from the loop's length -------------------------------------------------------
@pytest.mark.parametrize("bpm, bars, read, want", [
    (88, 4, 89.1, 88.0),            # the analysis a little off: snapped to whole bars
    (90, 8, 89.1, 90.0),
    (88, 4, 44.6, 88.0),            # half time read: the double is tried
    (120, 2, 131.0, None),          # too far (over 3%)
])
def test_length_tempo(bpm, bars, read, want):
    dur = bars * 240.0 / bpm
    assert C._length_tempo(dur, read) == want


def test_fallback_tempos_add_the_length_tempo_last_and_resolve_marks_it():
    r = _row("Dump/201.wav", dur=4 * 240.0 / 88, tempo=89.1)
    got = C._fallback_tempos(r)
    assert got[-1] == (88.0, "length", True)
    # the usual chain finds nothing (89.1 isn't whole bars); the length's tempo is used
    assert C._resolve_tempo(r.filename, r.duration_s, None, 89.1) == (None, None)
    assert C._resolve_tempo(r.filename, r.duration_s, None, 89.1, got) == (88.0, "length")


# --- 8: keys named by their folder ---------------------------------------------------------
@pytest.mark.parametrize("rel, keys", [
    ("Keys/Am_01.wav", True), ("Keys/Rhodes C.wav", True), ("Keys/Tine EP 02.wav", True),
    ("Keys/Organ 3.wav", True), ("Synth Keys/Lead 01.wav", False), ("Keys/Bass 01.wav", False),
])
def test_keys_are_named_by_the_path_words(rel, keys):
    canon = SH.path_labels(rel) | {"class.oneshot"}
    assert C._keys_labeled(_row(rel, canon=canon, classes=("OneShot",))) is keys


def test_keys_are_left_out_of_the_unrecognized_and_named_in_why():
    from fourier.packs.why_log import describe
    doc = {"category": "PIANO"}
    assert "not a single note" in describe("many_onsets", [12.0, 2.0], doc)
    assert "a chord" in describe("chord_not_stab", None, doc)


# --- 9: sustained sounds -------------------------------------------------------------------
def test_a_sustained_pad_is_a_oneshot_even_with_onsets():
    # 6 s, one event, chords changing about once a second: a loop by its onsets alone
    assert SH.audio_class(6.0, 1, 1.2) == "class.loop"
    assert SH.audio_class(6.0, 1, 1.2, harmonicity=0.95) == "class.oneshot"
    # a groove (percussive) or a fast legato line stays a loop
    assert SH.audio_class(6.0, 1, 1.2, harmonicity=0.3) == "class.loop"
    assert SH.audio_class(6.0, 1, 4.0, harmonicity=0.95) == "class.loop"


# --- 10: 808s named by a bare note --------------------------------------------------------
@pytest.mark.parametrize("name, bare, pc", [
    ("808_F.wav", False, None), ("808_F.wav", True, 5), ("Sub G 02.wav", True, 7),
    ("808_Eb.wav", True, 3), ("Room B Thump.wav", True, None), ("C Bass.wav", True, None),
])
def test_a_bare_note_letter_reads_only_without_sononym(name, bare, pc):
    assert C._name_root_pc(name, bare=bare) == pc


def test_a_retuned_name_says_c_and_a_second_c_names_the_note_it_was():
    assert C._retuned_stem("808_F", -5.0) == "808_F"                       # with Sononym, as before
    assert C._retuned_stem("808_F", -5.0, bare=True) == "808_C"
    assert C._note_word("808_A#") == "as" and C._note_word("808_F", bare=True) == "f"
    taken = {"808_c.wav"}
    assert C._choose_outname("808_C", taken, alt="as") == "808_C_from-as.wav"
    assert C._choose_outname("808_C", taken | {"808_c_from-as.wav"}, alt="as") == "808_C_from-as_2.wav"
    assert C._choose_outname("808_C", taken) == "808_C_2.wav"              # with Sononym, as before
    # a lowercase note reads as no note: verify's "named for the note it plays" holds
    assert C._name_root_pc("808_C_from-as.wav", bare=True) in (0, None)
    assert C._name_root_pc("808_C_from-as.wav") is None


# --- 12a: loops at different tempos aren't duplicates ---------------------------------------
def test_near_duplicate_loops_at_different_tempos_both_stay():
    emb = np.array([[1.0, 0.0], [1.0, 0.001]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    rec = [dict(id=1, row=0, qual=1.0, bpm=120.0), dict(id=2, row=1, qual=0.5, bpm=124.0)]
    assert C._prune_near_dups(rec, emb, 0.99)[1] == 1                       # as before
    assert C._prune_near_dups(rec, emb, 0.99, apart=C._tempo_apart)[1] == 0
    rec[1]["bpm"] = 121.0                                                   # one tempo folder
    assert C._prune_near_dups(rec, emb, 0.99, apart=C._tempo_apart)[1] == 1


# --- 12i: shakers in a percussion folder, clap / hat ties ------------------------------------
@pytest.mark.parametrize("name, rel, fallback, want", [
    ("Grain Shaker.wav", "Perc/Grain Shaker.wav", True, "PERC"),
    ("Grain Shaker.wav", "Perc/Grain Shaker.wav", False, "HATS"),       # with Sononym, as before
    ("Grain Shaker.wav", "Hats/Grain Shaker.wav", True, "HATS"),
    ("Shaker Hat 01.wav", "Percussion/Shaker Hat 01.wav", True, "HATS"),
    ("Tambourine 2.wav", "Percussion/Tambourine 2.wav", True, "PERC"),
])
def test_a_shaker_in_a_percussion_folder(name, rel, fallback, want):
    assert C._drum_named_in(name, rel, fallback) == want


def test_a_near_tie_between_hats_and_claps_goes_to_claps(monkeypatch):
    monkeypatch.setattr(C, "_clap_anti", lambda cat: None)
    e = np.eye(8)
    cats = [c for c, cfg in C.CATEGORIES.items() if cfg["kind"] == "oneshot"]
    anchors = {c: e[i % 8] if c not in ("HATS", "CLAPS") else None for i, c in enumerate(cats)}
    anchors["HATS"], anchors["CLAPS"] = e[0], e[1]
    for c in cats:
        if c not in ("HATS", "CLAPS"):
            anchors[c] = _unit(e[7] + 0.01 * e[2])
    near = _unit(0.71 * e[0] + 0.69 * e[1])
    assert C._clap_home(near, anchors, oneshot=True, loop=False) == "CLAPS"
    clear = _unit(0.9 * e[0] + 0.3 * e[1])
    assert C._clap_home(clear, anchors, oneshot=True, loop=False) == "HATS"


# --- 12j: "musical" drum-loop folders ----------------------------------------------------------
def test_musical_needs_harmonic_loops_when_the_median_is_recorded():
    assert N.trait_ok("musical", {"med": {"atk": 5.0}})                     # with Sononym: as before
    assert not N.trait_ok("musical", {"med": {"har": 0.2}})
    assert N.trait_ok("musical", {"med": {"har": 0.7}})


def test_musical_needs_a_pitch_focus_too():
    """Filtered drums read half harmonic (0.57 to 0.68) but focus on no pitch class: not
    "musical"; a chord under them, or a line, does (Fourier's own pitched call)."""
    from fourier.metadata.resolve import OWN_PITCHED_CHROMA
    assert not N.trait_ok("musical", {"med": {"har": 0.62, "chroma": 1.4}})
    assert N.trait_ok("musical", {"med": {"har": 0.62, "chroma": OWN_PITCHED_CHROMA + 0.3}})
    assert N.trait_ok("musical", {"med": {"har": 0.62, "chroma": None}})     # not measured: as before


# --- a held, tonal "loop" with no tempo is a one-shot to the CLAP fallback ---------------------
HeldRow = namedtuple("HeldRow", "id path rel_path filename duration_s classes canonical bpm tempo_bpm "
                                "acid_bpm acid_beats n_events onset_rate_hz harmonicity chroma_concentration")


def _held(rel="Dump/007.wav", n_events=1, ons=1.2, har=0.73, chroma=1.7, tempo=80.0, dur=6.0):
    return HeldRow(1, f"/lib/SampleLibrary/{rel}", rel, rel.rsplit("/", 1)[-1], dur, ["Loop"],
                   ["class.loop"], None, tempo, None, None, n_events, ons, har, chroma)


def test_a_held_tonal_loop_without_a_stated_tempo_is_offered_the_one_shot_categories():
    assert C._held_loop(_held())                     # 80 BPM is only Fourier's estimate
    assert C._held_loop(_held(tempo=None))
    assert not C._held_loop(_held(rel="Dump/Pad 120bpm.wav", dur=8.0))   # a tempo its name states
    assert not C._held_loop(_held(rel="85/007.wav", dur=480 / 85))        # a tempo folder states one
    assert not C._held_loop(_held(rel="Loops/007.wav"))                   # a loop by its folder
    assert not C._held_loop(_held(ons=2.7))          # a beat's onsets: a groove
    assert not C._held_loop(_held(chroma=1.3))       # no pitch focus: not tonal
    assert not C._held_loop(_held(har=0.3))          # drums
    assert not C._held_loop(_held(n_events=4))       # silences in it: a chain, not one held sound
    one = _held()._replace(classes=["OneShot"])
    assert not C._held_loop(one)                     # already a one-shot


def test_shaken_percussion_is_labeled_percussion():
    """Shaken percussion reads as hand percussion (PERC's label) by its path words, as congas
    and bongos do; a hi-hat stays a hat. (The name rule still files a shaker named alone with
    the hats, DRUM_NAME_RULES; in a percussion folder, PERC.)"""
    for rel in ("Perc/Grain Shaker.wav", "Acme/Tambourine B.wav", "Northwind/Cabasa C.wav",
                "Acme/Maraca Pair D.wav", "Acme/Conga D.wav", "Northwind/Bongo Pair Q.wav"):
        assert "perc.hand" in SH.path_labels(rel) and "hat" not in SH.path_labels(rel), rel
    assert "hat" in SH.path_labels("Hats/Closed Hat 01.wav")


# --- 5: folder names within a device's limit -------------------------------------------------
def test_family_names_fit_a_device_set_cap(monkeypatch):
    from fourier.packs import curate_config as cc
    assert N.family_cap() is None                                           # the M8's 44: as before
    monkeypatch.setattr(cc, "FAMILY_NAME_MAX", 15)
    assert N.family_cap() == 15
    used = set()
    a = N.fit_name("open-analog-clicky", 15, used)
    assert len(a) <= 15 and a == "open-analog"
    used.add(a)
    b = N.fit_name("open-analog-dark", 15, used)
    assert len(b) <= 15 and b != a and b.lower() not in used
    assert len(N.fit_name("supercalifragilistic", 15, set())) <= 15


# --- 6: the other direction, and anonymous loop folders ------------------------------------
def test_a_drum_loop_that_sounds_like_a_phrase_falls_back_to_phrases(monkeypatch):
    """A loop homed in a drum-loop category whose CLAP scores fail that gate and clearly pass
    PHRASES' falls back to PHRASES (frm=DRUMLOOPS); one the drum gate takes stays."""
    e = np.eye(6)
    anchors = {"PHRASES": e[0], "DRUMLOOPS": e[1]}
    anti = {"PHRASES": e[1], "DRUMLOOPS": e[2]}
    monkeypatch.setattr(C, "_clap_anti", lambda cat: anti.get(cat))
    keys = _unit(0.6 * e[0] + 0.1 * e[1] + 0.3 * e[2] + 0.6 * e[5])
    assert C._clap_rehome(keys, anchors, frm="DRUMLOOPS") == "PHRASES"
    assert C._clap_rehome(keys, anchors, misfiled={"PHRASES"}, frm="DRUMLOOPS") is None
    drums = _unit(0.15 * e[0] + 0.6 * e[1] + 0.1 * e[2] + 0.6 * e[5])
    assert C._clap_rehome(drums, anchors, frm="DRUMLOOPS") is None


@pytest.mark.parametrize("rel, anon", [
    ("Loops/85/loop_17.wav", True), ("Loops/loop 3 90bpm.wav", True), ("90/Loop_04.wav", True),
    ("Loops/85/Keys Loop 01.wav", False), ("Keys/85/loop_17.wav", False),
    ("Loops/Chords/loop_17.wav", False), ("Loops/85/Dusty Groove.wav", False),
])
def test_an_anonymous_loop_folder(rel, anon):
    assert C._anonymous_loop(rel) is anon


def test_busy_audio_in_an_anonymous_loop_folder_is_drums():
    """A bare "Loops/85/loop_17" whose audio reads a little tonal (lo-fi drums under a chord)
    is a drum loop by its busy onsets; a slow, tonal one isn't, nor a named one."""
    R = namedtuple("R", "rel_path path harmonicity onset_rate_hz")
    cfg = {"har_max": 0.68}
    assert C._drum_loop_evidence(R("Loops/85/loop_17.wav", "", 0.70, 3.0), [], cfg)
    assert not C._drum_loop_evidence(R("Loops/85/loop_17.wav", "", 0.70, 1.2), [], cfg)
    assert not C._drum_loop_evidence(R("Loops/85/loop_17.wav", "", 0.86, 3.0), [], cfg)
    assert not C._drum_loop_evidence(R("Loops/85/Pad Loop 01.wav", "", 0.70, 3.0), [], cfg)


# --- 9: no shape, one sound ------------------------------------------------------------------
def test_a_shapeless_held_sound_is_offered_the_oneshot_categories():
    R = namedtuple("R", "n_events harmonicity onset_rate_hz")
    assert C._held(R(1, 0.3, 4.0))                       # one event between silences
    assert C._held(R(3, 0.9, 1.0))                       # a pad changing chords
    assert not C._held(R(3, 0.4, 3.0))                   # hits: no call
    assert not C._held(R(None, None, None))


# --- 7: a one-bar loop -----------------------------------------------------------------------
def test_length_tempo_takes_one_bar_too():
    assert C._length_tempo(240.0 / 124, 125.5) == 124.0
