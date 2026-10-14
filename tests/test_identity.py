"""What counts as the same sound (packs/identity.py): recordings (never two synthetic single
cycles), editions of one recording, multisample sets, numbered series, and the rules that ask
them (copy prune, set cap, one note per C, round robins, kit takes). Invented names only."""
import os

import numpy as np
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
    W = np.eye(3, dtype="float32")                       # CLAP: apart, so no audio is read
    assert C._copies_within(srcs, W, durs=[0.63, 0.60, 0.6]) == [(0, 1)]
    assert C._copies_within(srcs, W, new={2}) == []
    assert C._copies_within(srcs, W, audio=False) == [(0, 1)]


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


def test_one_round_robin_a_note_and_velocity():
    names = ["vqxP_C1_v2_RR1", "vqxP_C1_v2_RR2", "vqxP_C1_v3_RR2", "vqxP_C3_v2_RR2", "Qzv Pad C3"]
    rec = [dict(path=p(f"Acme/Str/{n}.wav")) for n in names]
    assert C._one_per_round_robin([[0, 1], [2, 3, 4]], rec) == ([[0], [2, 3, 4]], 1)
    rec[1]["incumbent"] = True                            # the previous build's twin stays
    assert C._one_per_round_robin([[0, 1, 2, 3, 4]], rec) == ([[1, 2, 3, 4]], 1)
    rec[0]["pin"] = True                                  # and a Keep always does
    assert C._one_per_round_robin([[0, 1, 2, 3, 4]], rec) == ([[0, 2, 3, 4]], 1)
    assert I.rr_number(rec[1]["path"]) == 2 and I.rr_key(rec[4]["path"]) is None


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
