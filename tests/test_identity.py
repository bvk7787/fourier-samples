"""What counts as the same sound (packs/identity.py): recordings (never two synthetic single
cycles), editions of one recording, multisample sets, numbered series, and the rules that ask
them (copy prune, set cap, one note per C, round robins, kit takes). Invented names only."""
import os

import numpy as np
import pytest
import soundfile as sf

from fourier.packs import curate as C
from fourier.packs import curate_config as CC
from fourier.packs import identity as I
from fourier.packs import sets as S

LIB = "/x/SampleLibrary"


def p(rel):
    return f"{LIB}/{rel}"


# --- recording: never two synthetic single cycles --------------------------------------------------

def _cycle(path, shape, n=1024, sr=44100):
    t = np.arange(n) / n
    y = np.sin(2 * np.pi * t) if shape == "sine" else 2 * np.abs(2 * ((t + 0.25) % 1) - 1) - 1
    sf.write(path, np.tile(0.89 * y, 4).astype("float32"), sr, subtype="PCM_16")
    return str(path)


def test_two_single_cycles_are_never_one_recording(tmp_path):
    C._copy_signal.cache_clear()
    srcs = [_cycle(tmp_path / "Qzv Sine.wav", "sine"), _cycle(tmp_path / "Qzv Triangle.wav", "tri")]
    V = np.ones((2, 2), dtype="float32") / np.sqrt(2)
    assert C._copies_within(srcs, V) == [(0, 1)]                 # the audio alone can't tell them apart
    assert not I.comparable("waves") and I.comparable("oneshot")
    assert C._copies_within(srcs, V, audio=False) == []          # so a waves category asks the names only
    rec = [dict(id=1 + i, row=i, path=s, qual=1.0) for i, s in enumerate(srcs)]
    clusters = [dict(idxs=[0, 1], _pick=[0, 1])]
    stats = {}
    C._drop_copies(clusters, [2], rec, V, {"kind": "waves"}, lambda idxs, want: list(idxs)[:want], stats,
                   log=lambda m: None)
    assert clusters[0]["_pick"] == [0, 1] and "copy" not in stats
    C._copy_signal.cache_clear()


def test_a_copys_keeper_stays_through_its_folders_new_pick(monkeypatch):
    """The keeper of a copy its folder picked stays picked when the folder picks again without
    the copy (a pick that would drop it left both out)."""
    rec = [dict(id=10 + i, row=i, path=p(f"Acme/Qzv/Qzv Hit {i}.wav"), qual=1.0) for i in range(4)]
    calls = []
    monkeypatch.setattr(C, "_copies_within", lambda srcs, V, **kw: [] if calls.append(1) or len(calls) > 1
                        else [(0, 1)])
    monkeypatch.setattr(C, "_copy_keepers", lambda g, rec: {0})
    clusters = [dict(idxs=[0, 1, 2, 3], _pick=[0, 1])]
    stats = {}
    C._drop_copies(clusters, [2], rec, np.ones((4, 2), dtype="float32"), {"kind": "oneshot"},
                   lambda idxs, want: [i for i in idxs if i != 0][:want], stats, log=lambda m: None)
    assert clusters[0]["_pick"] == [0, 2] and stats["copy"] == 1


def _copy_rounds(monkeypatch, rounds):
    """_copies_within answers each call with the next of these lists of pairs, then none; of
    one recording's copies the lowest index stays."""
    left = list(rounds)
    monkeypatch.setattr(C, "_copies_within", lambda srcs, V, **kw: left.pop(0) if left else [])
    monkeypatch.setattr(C, "_copy_keepers", lambda g, rec: {min(g)})
    return [dict(id=10 + i, row=i, path=p(f"Acme/Qzv Rim/Qzv Rim {i}.wav"), qual=1.0) for i in range(4)]


def test_a_copys_keeper_left_as_a_copy_in_a_later_round_hands_it_on(monkeypatch):
    """Round 1 leaves Rim 2 out as a copy of Rim 1; the folder picks Rim 0, and round 2 leaves
    Rim 1 out as a copy of Rim 0: Rim 2's keeper is Rim 0, the copy that stays (verify fails a
    copy whose keeper isn't in the master)."""
    rec = _copy_rounds(monkeypatch, [[(0, 1)], [(0, 1)]])        # (indices into the picks)
    clusters = [dict(idxs=[0, 1, 2, 3], _pick=[1, 2])]
    stats, why = {}, {}
    C._drop_copies(clusters, [2], rec, np.ones((4, 2), dtype="float32"), {"kind": "oneshot"},
                   lambda idxs, want: list(idxs)[:want], stats, why, log=lambda m: None)
    assert clusters[0]["_pick"] == [0, 3]
    assert why == {"copy": {12: 10, 11: 10}} and stats["copy"] == 2


def test_a_copys_keeper_left_as_a_copy_by_the_second_call_hands_it_on(monkeypatch):
    """The build prunes copies again after one note per C refilled a folder: a keeper of the
    first call left out by the second hands its copies on."""
    rec = _copy_rounds(monkeypatch, [[(0, 1)], [], [(0, 1)]])
    clusters = [dict(idxs=[1, 2, 3], _pick=[1, 2])]
    stats, why = {}, {}
    pick = lambda idxs, want: list(idxs)[:want]  # noqa: E731
    C._drop_copies(clusters, [2], rec, np.ones((4, 2), dtype="float32"), {"kind": "oneshot"}, pick, stats,
                   why, log=lambda m: None)
    assert clusters[0]["_pick"] == [1, 3] and why == {"copy": {12: 11}}
    clusters[0]["idxs"], clusters[0]["_pick"] = [0, 1, 3], [0, 1]   # a refill brings Rim 0
    C._drop_copies(clusters, [2], rec, np.ones((4, 2), dtype="float32"), {"kind": "oneshot"}, pick, stats,
                   why, log=lambda m: None)
    assert clusters[0]["_pick"] == [0, 3]
    assert why == {"copy": {12: 10, 11: 10}} and stats["copy"] == 2


def test_keepers_left_by_the_other_step_hand_their_files_on():
    """A copy of a note that one note per C left out leaves as that note did (for the note that
    stayed, on the same C), and a note left for a pick the copy prune later left out points at
    the copy that stayed; a chain that ends outside the picks stays as it was."""
    names = ["Qzv Pad C3 b", "Qzv Pad C3", "Qzv Pad A2", "Qzv Bell 1", "Qzv Bell 1 b", "Qzv Bell 1 c",
             "Qzv Sub E1", "Qzv Sub E1 b"]
    rec = [dict(id=20 + i, path=p(f"Acme/Ms/{n}.wav")) for i, n in enumerate(names)]
    why = {"copy": {20: 21, 24: 23, 27: 26},
           "one_note_per_c": {21: ["Qzv Pad A2.wav", "C3"], 25: ["Qzv Bell 1 b.wav", "C4"],
                              26: ["Qzv Pad A2.wav", "E1"]}}
    note_ids = {21: 22, 25: 24, 26: 29}            # (29: no such pick)
    stats = {"copy": 3}
    C._settle_keepers(why, note_ids, rec, picked={22, 23}, stats=stats)
    assert why["copy"] == {24: 23, 27: 26}
    assert why["one_note_per_c"] == {21: ["Qzv Pad A2.wav", "C3"], 20: ["Qzv Pad A2.wav", "C3"],
                                     25: ["Qzv Bell 1.wav", "C4"], 26: ["Qzv Pad A2.wav", "E1"]}
    assert stats["copy"] == 2


def test_a_note_left_for_a_note_a_later_call_left_out_points_at_the_one_that_stayed():
    """One note per C runs again after the copy prune refills: a note an earlier call kept and
    a later call left out hands the notes left for it on."""
    rec = [dict(id=30 + i, path=p(f"Acme/Ms/Qzv Key {n}.wav")) for i, n in enumerate(["A2", "C3", "D#3"])]
    why = {"one_note_per_c": {30: ["Qzv Key C3.wav", "C3"], 31: ["Qzv Key D#3.wav", "C3"]}}
    C._settle_keepers(why, {30: 31, 31: 32}, rec, picked={32}, stats={})
    assert why["one_note_per_c"] == {30: ["Qzv Key D#3.wav", "C3"], 31: ["Qzv Key D#3.wav", "C3"]}


# --- edition ------------------------------------------------------------------------------------

def test_editions_of_one_recording():
    raw = p("Northwind/Grit Pack/pt1/1 GRIT (raw)/QX Hit/QX Hit 887.wav")
    clean = p("Northwind/Grit Pack/pt2/GRIT Masters (MONO)/QX Hit/QX Hit 887 Cleanup.wav")
    dirty = p("Northwind/Grit Pack/pt1/2 GRIT (raw + heat)/QX Hit/QX Hit 887 Dirty.wav")
    assert I.editions(raw, clean, 0.175, 0.160) and I.editions(clean, dirty)   # trimmed differently
    assert I.editions(p("Acme/Found/Clean/QzvSlap-Clean.wav"), p("Acme/Found/Tube/QzvSlap-Tube.wav"))
    # a format twin side by side, and mono/stereo or 16/24-bit twin folders
    assert I.editions(p("Acme/Drums/Style A/117/Qzv_Brk_7_117.wav"), p("Acme/Drums/Style A/117/Qzv_Brk_7_117.aif"), 4.0, 4.0)
    assert I.editions(p("Acme/Pack 2/Pack 2 (WAV stereo)/Chord C 04.wav"),
                      p("Acme/Pack 2/Pack 2 (WAV mono)/Chord C 04.wav"), 1.0, 1.0)
    assert I.editions(p("Acme/Pack 2/Kicks/Qzv 01.wav"), p("Acme/Pack 2 - 24bit/Kicks/Qzv 01.wav"), 1.0, 1.0)
    # a compilation's re-release under another pack of the vendor, at one length
    assert I.editions(p("Acme/Qzv Synth/Lift/Vqx Lift Zap A 06.wav"),
                      p("Acme/Collection - 16bit/Kits/Vqx Lift Zap A 06.wav"), 2.0, 2.0)


def test_not_editions():
    raw = p("Northwind/Grit Pack/raw/QX Hit 887.wav")
    assert not I.editions(raw, p("Northwind/Grit Pack/proc/QX Hit 888 Cleanup.wav"))     # another hit
    assert not I.editions(p("Acme/Kit/808/Kick 01.wav"), p("Acme/Kit/909/Kick 01.wav"), 0.5, 0.5)
    assert not I.editions(p("Acme/Kit/Hit Dirty.wav"), p("Acme/Kit/Hit Clean.wav"))       # names nothing
    assert not I.editions(raw, p("Vqx/Grit Pack/raw/QX Hit 887 Cleanup.wav"))               # another vendor
    # one name in two style folders is two loops, whatever its format
    assert not I.editions(p("Acme/Drums/Style B/117/Qzv_Brk_7_117.wav"), p("Acme/Drums/Style A/117/Qzv_Brk_7_117.aif"),
                          2.0, 2.0)
    assert not I.editions(p("Acme/Drums/Style A/117/Qzv_Brk_7_117.wav"), p("Acme/Drums/Style A/117/Qzv_Brk_7_117.aif"),
                          2.0, 4.0)                                                       # other lengths
    assert not I.editions(p("Acme/Qzv Synth/Lift/Vqx Lift Zap A 06.wav"),
                          p("Acme/Collection/Kits/Vqx Lift Zap A 06.wav"), 2.0, 3.0)
    assert not I.editions(p("Acme/Loops A/Groove13.wav"), p("Acme/Loops B/Groove13.wav"), 2.0, 2.0)


def test_editions_are_copies_within_a_category():
    srcs = [p("Northwind/Grit/raw/QX Tom 731.wav"), p("Northwind/Grit/proc/QX Tom 731 Cleanup.wav"),
            p("Northwind/Grit/proc/QX Tom 732 Cleanup.wav")]
    W = np.eye(3, dtype="float32")
    W[1] = W[0]                                          # CLAP: the editions agree, 731 / 732 apart
    assert C._copies_within(srcs, W, durs=[0.63, 0.60, 0.6]) == [(0, 1)]   # (by name: no audio read)
    assert C._copies_within(srcs, W, new={2}) == []
    assert C._copies_within(srcs, np.eye(3, dtype="float32"), audio=False) == [(0, 1)]
    # an edition processed into another sound (CLAP under EDITION_COPY_COS) is its own sound
    W2 = np.eye(3, dtype="float32")
    W2[1] = [0.8, 0.6, 0.0]                              # cos 0.8 to the raw take
    assert C._copies_within(srcs, W2, durs=[0.63, 0.60, 0.6], cos=0.95) == []


def test_a_stereo_sounds_two_channels_are_editions():
    """Mono files of a stereo sound's two channels (a standalone L / R token, one folder, a
    name stating a note) are one recording, whatever their lengths; a copy prune keeps one
    while CLAP hears them at EDITION_COPY_COS."""
    left, right = p("Acme/Qzv/Keys/41_Vqx_L_QX9_A#2.wav"), p("Acme/Qzv/Keys/41_Vqx_R_QX9_A#2.wav")
    assert I.editions(left, right, 7.3, 8.9) and I.edition_key(left) == I.edition_key(right)
    assert I.editions(p("Acme/Qzv/Pad Left C3.wav"), p("Acme/Qzv/Pad Right C3.wav"))
    srcs = [left, right, p("Acme/Qzv/Keys/41_Vqx_R_QX9_E3.wav")]
    W = np.eye(3, dtype="float32")
    W[1] = W[0]
    assert C._copies_within(srcs, W, durs=[7.3, 8.9, 8.0]) == [(0, 1)]
    assert C._copies_within(srcs, np.eye(3, dtype="float32"), durs=[7.3, 8.9, 8.0]) == []   # CLAP apart
    # a rating on one channel covers the other
    assert I.Reach(drops={left}, facts={left: (None, 7.3)}).dropped(right, dur=8.9) == left


def test_l_and_r_that_arent_channels():
    org = "Acme/Qzv/Keys/41_Vqx_{}_QX9_A#2.wav"
    assert not I.editions(p(org.format("L")), p("Acme/Qzv/Keys 2/41_Vqx_R_QX9_A#2.wav"))      # another folder
    assert not I.editions(p(org.format("L")), p(org.format("L2")))
    assert not I.editions(p(org.format("L")), p("Acme/Qzv/Keys/41_Vqx_QX9_A#2.wav"))           # no channel
    assert not I.editions(p(org.format("L")), p(org.format("Left")))                          # one side
    assert not I.editions(p("Acme/Qzv/Keys/Vqx L C3.wav"), p("Acme/Qzv/Keys/Vqx R D3.wav"))   # two notes
    # no note named: a hand drum's left and right hand, a loop's two parts
    assert not I.editions(p("Acme/Perc/Qzvbongo L 05.wav"), p("Acme/Perc/Qzvbongo R 05.wav"), 0.4, 0.4)
    assert not I.editions(p("Acme/Loops/Qzphase L.wav"), p("Acme/Loops/Qzphase R.wav"), 2.0, 2.0)
    # a letter in a word or a note isn't a channel
    assert not I.editions(p("Acme/Qzv/Vqxl C3.wav"), p("Acme/Qzv/Vqxr C3.wav"))
    assert I.edition_name(p("Acme/Qzv/Lead Rr C3.wav")) == "lead rr c3"


# --- set and series ---------------------------------------------------------------------------------

def test_a_set_states_its_notes_velocities_or_round_robins():
    k = I.set_key
    assert k(p("Acme/Str/vqxP_C1_v2_RR1.wav")) == k(p("Acme/Str/vqxP_C3_v1_RR2.wav")) is not None
    assert k(p("Acme/Str/Qzv Pad C3.wav")) == k(p("Acme/Str/Qzv Pad G4.wav")) is not None
    assert k(p("Acme/Kit/Qzv Kick_v1.wav")) == k(p("Acme/Kit/Qzv Kick_v2.wav")) is not None
    assert k(p("Acme/Str/Qzv Pad C3.wav")) != k(p("Acme/Str B/Qzv Pad G4.wav"))        # one source folder
    assert k(p("Acme/Kit/MP_Qzv_KICK_05.wav")) is None           # a prefix, not a velocity
    assert k(p("Acme/Kit/Kick 01.wav")) is None and k(p("Acme/Lead/Qzv Lead 059.wav")) is None


def test_numbers_are_notes_only_when_they_track_the_pitch():
    lead = [(p(f"Acme/Ms/Qzv Env Lead {n}.wav"), r) for n, r in (("059", 58.81), ("061", 60.81), ("062", 61.8))]
    got = I.number_sets(lead)
    assert len(got) == 3 and len(set(got.values())) == 1
    vqx = [(p("Acme/Ms/QZ_Vqxbot-40-90.wav"), 50.01), (p("Acme/Ms/QZ_Vqxbot-44-90.wav"), 54.0),
             (p("Acme/Ms/QZ_Vqxbot-48-120.wav"), 58.02), (p("Acme/Ms/QZ_Vqxbot-52-120.wav"), None)]
    assert len(I.number_sets(vqx)) == 4                    # the first number; the second a velocity
    takes = [(p(f"Acme/Ms/Jq Lead {n:02d}.wav"), 60.0 + 0.02 * n) for n in (1, 2, 3)]
    assert I.number_sets(takes) == {}                         # three takes at one pitch: a series
    assert I.number_sets(lead[:2]) == {}                      # two numbers may agree by chance
    far = [(p(f"Acme/Ms/Qzv Hit {n}.wav"), n - 100.0) for n in (130, 135, 140)]
    assert I.number_sets(far) == {}                           # not MIDI notes
    assert I.number_sets([(p(f"Acme/Ms/Kick {n}.wav"), 30.0 + n) for n in (1, 2, 3)]) == {}   # generic


def test_a_numbered_set_survives_a_reading_off_by_an_octave_or_a_note():
    """A sampled instrument numbered by MIDI note plus a constant ("Vqx Gtr-73" plays 59): a
    root read an octave off, or one file the pack numbered a note off, doesn't unmake the set,
    as long as most of the rooted files agree."""
    gtr = [(p(f"Acme/Ms/Vqx_Gtr-{n}.wav"), n - 13.82) for n in (53, 57, 61, 65, 69, 73, 77, 81)]
    gtr += [(p("Acme/Ms/Vqx_Gtr-85.wav"), 85 - 13.82 + 12.0),        # an octave error
            (p("Acme/Ms/Vqx_Gtr-96.wav"), 96 - 12.86)]                 # numbered a note off
    got = I.number_sets(gtr)
    assert len(got) == 10 and len(set(got.values())) == 1
    # Sononym an octave off for a whole stretch of the low notes: still one set
    env = [(p(f"Acme/Ms/Qzv Env Pwm {n:03d}.wav"), float(n - 12 + (12 if n < 40 else 0))) for n in range(29, 50)]
    assert len(I.number_sets(env)) == 21
    # a number's editions are one vote: three editions of each of three numbers that don't
    # track the pitch are no set, however well each number's editions agree
    bells = [(p(f"Acme/Ms/Qzv Bell {n:02d}{ed}.wav"), r) for n, r in ((1, 32.0), (49, 56.0), (62, 53.6))
             for ed in ("", " Dirty", " Cleanup")]
    assert I.number_sets(bells) == {}
    # numbers that agree only octaves aside are no set (a fifth up each step, wrapping)
    assert I.number_sets([(p(f"Acme/Ms/Qzv Saw {n:02d}.wav"), r) for n, r in ((2, 67.0), (6, 71.0), (10, 63.0))]) == {}
    # most of them disagreeing is no set: three agree out of seven rooted
    few = gtr[:3] + [(p(f"Acme/Ms/Vqx_Gtr-{n}.wav"), 60.0 + k * 0.9) for k, n in enumerate((88, 90, 92, 94))]
    assert I.number_sets(few) == {}


def test_series_are_never_sets():
    k = I.patch_key
    assert k(p("Acme/Fx/Hinge07.wav")) is None and k(p("Acme/Fx/Lever01.wav")) is None
    assert k(p("Acme/Batch/FX/106_QZVX.wav")) is None and k(p("Acme/Kit/Kick 01.wav")) is None
    # a patch number glued to the synth's name: each its own patch
    assert k(p("Acme/Mini/Vqx 5/QX-Vqx5-27.wav")) != k(p("Acme/Mini/Vqx 5/QX-Vqx17-27.wav"))
    # one described patch's variant letters and takes are one patch...
    assert k(p("Acme/Qzv/Drop/Qzv Drop Zap A 01.wav")) == k(p("Acme/Qzv/Drop/Qzv Drop Zap C 09.wav")) is not None
    assert k(p("Acme/Qzv/Drop/Qzv Drop Zap A 01.wav")) != k(p("Acme/Qzv/Drop/Qzv Drop Hiss A 01.wav"))
    # ...a one-word name's letter is a bank, its number a preset
    assert k(p("Acme/Arch/Qzvarch B 57.wav")) != k(p("Acme/Arch/Qzvarch A 23.wav"))
    # a processing word is an edition, not another patch
    assert k(p("Acme/Grit/QX Tom 741 Dirty.wav")) == k(p("Acme/Grit/QX Tom 742.wav"))


def test_drum_categories_cap_true_sets_only():
    assert {"KICKS", "PERC"} <= CC.SIBLING_SETS_ONLY <= CC.SIBLING_CAP_CATS
    series = p("Acme/Kit/BD Qzv 07.wav")
    assert I.cap_key(series, "KICKS") is None and I.cap_key(series, "FX") is not None
    notes = [p(f"Acme/Ticks/61_Qzv_Tick_ZQ_{n}_0001.wav") for n in ("C1", "C3", "C#4", "D4", "D#5")]
    assert len({I.cap_key(x, "PERC") for x in notes}) == 1
    rec = [dict(path=x, tune=t, pin=False) for x, t in zip(notes, (24, 48, 61, 62, 75))]
    rec += [dict(path=p(f"Acme/Perc/Qzv Block {n:02d}.wav"), tune=None, pin=False) for n in range(1, 6)]
    key = lambda x: I.cap_key(x, "PERC")                                      # noqa: E731
    got = C._cap_siblings(list(range(10)), rec, cap=CC.sibling_cap("PERC"), key=key)
    assert len([i for i in got if i < 5]) == 3 and [i for i in got if i >= 5] == [5, 6, 7, 8, 9]
    numbered = {x: ("acme/perc", "qzv block") for x in (r["path"] for r in rec[5:])}
    assert I.cap_key(rec[5]["path"], "PERC", numbered) == ("acme/perc", "qzv block")


def test_the_fx_cap_binds_a_patchs_variant_letters():
    rec = [dict(path=p(f"Acme/Qzv/Drop/Qzv Drop Zap {v}.wav"), tune=None, pin=False)
           for v in ("A 01", "A 03", "B 03", "B 06", "C 03", "C 05")]
    assert len(C._cap_siblings(list(range(6)), rec, cap=CC.sibling_cap("FX"),
                               key=lambda x: I.cap_key(x, "FX"))) == 2


# --- the rules that ask -------------------------------------------------------------------------------

def _rt(name, tune):
    """A note retuned by its measured root onto the nearest C (the name states none)."""
    return dict(path=p(f"Acme/Ms/{name}"), tune=tune, retune=round(12 * round(tune / 12) - tune, 2),
                root_src="detect", name_pc=None)


def test_one_note_per_c_reads_numbers_that_track_the_pitch():
    rec = [_rt("Qzv Env Lead 059.wav", 58.81), _rt("Qzv Env Lead 061.wav", 60.81),
           _rt("Qzv Env Lead 062.wav", 61.8)]
    assert [r["retune"] for r in rec] == [1.19, -0.81, -1.8]
    groups, n = C._one_per_retune_target([[0, 1, 2]], rec)
    assert n == 2 and groups == [[1]]                     # 061's shift is the smallest
    # numbered takes at one pitch are a series: untouched
    rec = [_rt(f"Jq Lead {i:02d}.wav", 61.8 + 0.01 * i) for i in (1, 2, 3)]
    assert C._one_per_retune_target([[0, 1, 2]], rec) == ([[0, 1, 2]], 0)



def test_one_note_per_c_reads_the_whole_folder():
    """A numbered set is read on its whole source folder (curate._folder_number_roots), so the
    three of its notes a category's candidates hold are the set its other notes say it is."""
    rec = [_rt(f"Qzv Env Pwm {n:03d}.wav", n - 12.0 + d) for n, d in ((59, 0.02), (61, 0.04), (62, 0.04))]
    rec += [dict(path=p(f"Acme/Ms/Qzv Env Pwm {n:03d}.wav"), tune=None, own_root=None, retune=None)
            for n in (20, 21, 22)]                 # roots the candidates don't carry
    roots = {p(f"Acme/Ms/Qzv Env Pwm {n:03d}.wav"): n - 12.0 for n in range(20, 40)}
    assert C._one_per_retune_target([[0, 1, 2, 3, 4, 5]], rec)[1] == 2   # three agree: a set already
    rec[2]["tune"] = 49.5                                                  # one off: two of three
    rec[2]["retune"] = round(48 - 49.5, 2)
    assert C._one_per_retune_target([[0, 1, 2, 3, 4, 5]], rec)[1] == 0
    groups, n = C._one_per_retune_target([[0, 1, 2, 3, 4, 5]], rec, roots=roots)
    assert n == 2 and 0 in groups[0] and {1, 2}.isdisjoint(groups[0])    # the folder says: a set


def test_one_note_per_c_reads_pyin_where_sononym_isnt_sure():
    """Notes of a numbered set Sononym reads with little confidence (no root the build retunes
    from) are still the set by pYIN's roots, so two of its notes never land on one C."""
    rec = [_rt(f"Vqx_Gtr-{n}.wav", n - 13.82) for n in (73, 77)]
    rec += [dict(path=p(f"Acme/Ms/Vqx_Gtr-{n}.wav"), tune=None, own_root=n - 13.82, retune=None)
            for n in (53, 57, 61)]
    groups, n = C._one_per_retune_target([[0, 1, 2, 3, 4]], rec)
    assert n == 1 and groups == [[0, 2, 3, 4]]            # 73 (+0.82) keeps C4 over 77 (-3.18)


def test_one_round_robin_a_note_and_velocity():
    names = ["vqxP_C1_v2_RR1", "vqxP_C1_v2_RR2", "vqxP_C1_v3_RR2", "vqxP_C3_v2_RR2", "Qzv Pad C3"]
    rec = [dict(path=p(f"Acme/Str/{n}.wav")) for n in names]
    assert C._one_per_round_robin([[0, 1], [2, 3, 4]], rec) == ([[0], [2, 3, 4]], 1)
    rec[1]["incumbent"] = True                            # the previous build's twin stays
    assert C._one_per_round_robin([[0, 1, 2, 3, 4]], rec) == ([[1, 2, 3, 4]], 1)
    rec[0]["pin"] = True                                  # and a Keep always does
    assert C._one_per_round_robin([[0, 1, 2, 3, 4]], rec) == ([[0, 2, 3, 4]], 1)
    assert I.rr_number(rec[1]["path"]) == 2 and I.rr_key(rec[4]["path"]) is None


def test_a_bare_round_robin_after_a_velocity():
    """A bare number right after a velocity, ending a name that states a note, is its round
    robin ("_v4_1" / "_v4_2"): one stays."""
    names = ["Vqx_Stac_D2_v4_1", "Vqx_Stac_D2_v4_2", "Vqx_Stac_D2_v2_1", "Vqx_Stac_E2_v4_2",
             "Vqx Tuba A#2 PP 004", "Vqx Tuba A#2 PP 005"]
    rec = [dict(path=p(f"Acme/Qzv/Stac/{n}.wav")) for n in names]
    assert C._one_per_round_robin([[0, 1, 2, 3, 4, 5]], rec) == ([[0, 2, 3, 4]], 2)
    assert I.rr_key(rec[0]["path"]) == I.rr_key(rec[1]["path"]) != I.rr_key(rec[2]["path"])
    assert I.rr_number(rec[1]["path"]) == 2 and I.rr_number(rec[5]["path"]) == 5
    # only then: no note (a version's numbered patches), no velocity, a number not last
    for n in ("Qzv Lead v2 01", "Qzv_Lead_C2_7", "Qzv_C2_v3_1_Wet", "Qzv_v3_1", "Qzv_Pad_C2_v3_rr1_2"):
        assert not I._bare_rr(p(f"Acme/X/{n}.wav"), [t for t in I._SPLIT.split(n.lower()) if t]), n
    assert I.rr_key(p("Acme/X/Qzv Lead v2 01.wav")) is None
    assert I.rr_key(p("Acme/X/Qzv_Pad_C2_v3_rr1_2.wav")) == (I._where(p("Acme/X/a.wav"))[2], "qzv pad c2 v3 2")


def test_a_kit_takes_one_take_of_a_drum():
    ce = lambda name: ("TOMS", {"src": p(f"Acme/Kit/{name}.wav")})            # noqa: E731
    assert S._stem(ce("QzvDrumLo Brush A 4")) == S._stem(ce("QzvDrumLo Stick A +OH 1"))
    assert S._stem(ce("Tom_Hi_A_QX7_05")) != S._stem(ce("Tom_Mid_QX7_Tone_06"))
    assert S._stem(ce("Qzv-Tom-T12_RQ_T20")) != S._stem(ce("Qzv-Tom-T14_FQ_T51"))
    c = [ce(n) for n in ("QzvDrumLo Brush A 4", "QzvDrumLo Stick A +OH 1", "QzvDrumHi Brush A 2")]
    got = S._spread_pick(c, 2)
    assert len(got) == 2 and len({S._stem(x) for x in got}) == 2


def test_names_stay_library_neutral():
    assert os.path.basename(I.__file__) == "identity.py"
    assert I.edition_name(p("Acme/X/Hit 87 Cleanup.wav")) == I.edition_name(p("Acme/Y/Hit 87.aif")) == "hit 87"


@pytest.mark.parametrize("db", ["t.db", "t.duckdb"])
def test_folder_number_roots_reads_the_whole_folder(tmp_path, db):
    """The build's and verify's numbered-set roots come from every file of a numbered file's
    source folder: Sononym's sure root, else pYIN's; other folders and unnumbered names aside."""
    from fourier.db import session as S
    from fourier.db.models import Sample, SampleFeatures, SononymMeta
    from fourier.metadata.rows import forget_current
    old = (S._engine, S._SessionLocal)
    S._engine = S._SessionLocal = None
    try:
        S.init_db(tmp_path / db)
        forget_current()
        lib = "/l/SampleLibrary/Acme/Ms_100%"           # (LIKE's own wildcards in a folder name)
        with S.session_scope() as s:
            for i, (n, son, conf, own) in enumerate(((40, 52.0, 0.9, 52.1), (41, 70.0, 0.1, 53.0),
                                                     (42, None, None, None))):
                s.add(Sample(id=i + 1, path=f"{lib}/Qzv Env Pwm {n:03d}.wav", rel_path=f"Acme/Ms/Qzv Env Pwm {n:03d}.wav",
                             filename=f"Qzv Env Pwm {n:03d}.wav", duration_s=1.0))
                if son is not None:
                    s.add(SononymMeta(sample_id=i + 1, base_note=son, base_note_confidence=conf))
                s.add(SampleFeatures(sample_id=i + 1, own_root_midi=own))
            s.add(Sample(id=9, path="/l/SampleLibrary/Acme/Other/Qzv Env Pwm 043.wav",
                         rel_path="Acme/Other/Qzv Env Pwm 043.wav", filename="Qzv Env Pwm 043.wav", duration_s=1.0))
            s.add(Sample(id=10, path="/l/SampleLibrary/Acme/MsX100%/Qzv Env Pwm 044.wav",       # "_" isn't a wildcard
                         rel_path="Acme/MsX100%/Qzv Env Pwm 044.wav", filename="Qzv Env Pwm 044.wav", duration_s=1.0))
            s.add(SampleFeatures(sample_id=10, own_root_midi=55.0))
        with S.session_scope() as s:
            got = C._folder_number_roots(s, [f"{lib}/Qzv Env Pwm 040.wav", f"{lib}/Pad.wav"])
            assert C._folder_number_roots(s, [f"{lib}/Pad.wav"]) == {}         # no numbered name
        assert {k: v for k, v in got.items() if v is not None} == {f"{lib}/Qzv Env Pwm 040.wav": 52.0,
                                                                  f"{lib}/Qzv Env Pwm 041.wav": 53.0}
        assert C._folder_number_roots(None, [f"{lib}/Qzv Env Pwm 040.wav"]) == {}
    finally:
        forget_current()
        if S._engine is not None:
            S._engine.dispose()
        S._engine, S._SessionLocal = old


def test_one_note_per_c_among_the_picks_leaves_only_for_a_note_that_ships():
    """The build runs one note per C on its picks (curate._one_note_per_c_picked): a note leaves
    only for one its folders keep, and its folder picks again without it."""
    rec = [dict(path=p("Acme/Ms/Qzv Pad A2.wav"), retune=3.0, root_src="name", tune=None, id=1),
           dict(path=p("Acme/Ms/Qzv Pad C3.wav"), retune=0.0, root_src="name", tune=None, id=2),
           dict(path=p("Acme/Ms/Qzv Bell 1.wav"), retune=None, tune=None, id=3),
           dict(path=p("Acme/Ms/Qzv Pad D3.wav"), retune=-2.0, root_src="name", tune=None, id=4)]
    clusters = [dict(idxs=[0, 2], _pick=[0]), dict(idxs=[1, 3], _pick=[1])]
    why: dict = {}
    pick = lambda idxs, want: list(idxs)[:want]  # noqa: E731
    assert C._one_note_per_c_picked(clusters, [1, 1], rec, pick, why=why) == 1
    assert clusters[0]["_pick"] == [2] and clusters[1]["_pick"] == [1]    # A2 left; its folder refilled
    assert why == {1: ["Qzv Pad C3.wav", "C3"]}
    # a note the folders don't pick (D3, not in a _pick) never unseats one
    assert C._one_note_per_c_picked(clusters, [1, 1], rec, pick) == 0


def test_an_automatic_misfile_covers_its_file_not_its_set():
    a, b = p("Acme/Ms/Qzv Marimba C5.wav"), p("Acme/Ms/Qzv Marimba C3.wav")
    assert I.Reach(misfiled={a: "ACOUSTIC"}).misfiled(b) == {"ACOUSTIC"}            # a rating: its set
    r = I.Reach(misfiled={a: "ACOUSTIC"}, file_only={a})
    assert r.misfiled(b) == set() and r.misfiled(a) == {"ACOUSTIC"}               # a detection: its file
    from fourier.packs import ratings as R
    store = {"ratings": {}, "auto_misfiled": {a: {"category": "ACOUSTIC", "other": "ACOUSTIC"},
                                              b: {"category": "FX", "other": "SUB"}}}
    assert R._auto_misfiled(store) == {b: "FX"}                                  # (one that says nothing: skipped)


def test_a_number_whose_files_disagree_is_no_vote_and_parts_are_one_pack():
    """A series of different kicks whose numbers' own files read ten semitones apart is no
    set, however its medians fall; a download part's number in an edition's brackets is the
    pack ("(WAV stereo 1)" and "(WAV mono)" are one pack's editions)."""
    kicks = [(p(f"Acme/Ms/Qzv Kick Zap {n:02d}{ed}.wav"), r) for n, ed, r in
             ((9, "", 28.8), (9, " Dirty", 39.0), (11, "", 36.8), (11, " Dirty", 29.1), (13, "", 30.0), (15, "", 33.0))]
    assert I.number_sets(kicks) == {}
    assert I._clean("Qzv Pack 2 (WAV stereo 1)") == I._clean("Qzv Pack 2 (WAV mono)") == "qzv pack 2"
    assert I._clean("Drums (2)") != I._clean("Drums")          # a folder's own number stays
    from fourier.packs import sets as K
    assert K._stem(("TOMS", {"src": p("Acme/Kit/Qzv Tom Low A_5b.wav")})) == \
        K._stem(("TOMS", {"src": p("Acme/Kit/Qzv Tom Low A_5.wav")}))      # "_5b" is a take too


def test_a_velocity_read_off_doesnt_void_its_numbers_vote():
    """Three velocities of each number, the loudest read a semitone sharp (a brighter layer):
    most of each number's files agree, so the numbers still vote and the set holds."""
    lyr = [(p(f"Acme/Ms/Qzv Strx-{n}-{v}.wav"), n - 2.0 + (1.15 if v == 127 else 0.03))
           for n in (42, 46, 50, 54) for v in (89, 109, 127)]
    assert len(I.number_sets(lyr)) == 12


def test_a_pyin_fill_an_octave_off_takes_the_octave_its_number_names(monkeypatch):
    """A numbered set's note with no root the build retunes from, whose pYIN pitch reads an
    octave low, is judged at the octave its number names: it never lands on a C another note
    of its set already plays."""
    monkeypatch.delenv("FOURIER_NO_PITCH_CHECK", raising=False)
    name = "Acme/Ms/Qzv Strx-{}-127.wav"
    rec = [dict(path=p(name.format(n)), tune=n - 2.0, retune=None) for n in (50, 54, 58)]
    rec.append(dict(path=p(name.format(70)), tune=68.0, retune=4.0, root_src="detect"))   # plays C (72)
    rec.append(dict(path=p(name.format(74)), tune=None, own_root=60.0, retune=None))      # 72, read as 60
    heard = {name.format(74).rsplit("/", 1)[-1]: 60.0}
    monkeypatch.setattr(C, "_detect_pitch", lambda q: heard.get(q.rsplit("/", 1)[-1]))
    assert C._fill_roots("SYNTH", list(range(len(rec))), rec) == 0
    assert rec[4]["retune"] is None                    # its C is taken: it keeps its pitch
