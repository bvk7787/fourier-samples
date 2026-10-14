"""`fourier verify`'s invariants (packs/verify.py): each fails on a master that breaks it, built
here from invented names, so a regression in the build can't pass verify again."""
import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from fourier.packs import curate as C
from fourier.packs import why_log
from fourier.packs.verify import (FAIL, PASS, WARN, _Ctx, _check_kit_slots, _late_start, check_audio,
                                  check_copies, check_one_note_kept, check_pitch, check_ratings, check_routing,
                                  verify_master)

SR = 44100
LIB = "/lib/SampleLibrary"


def _tone(seconds=0.5, hz=220.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.8 * np.exp(-3 * t) * np.sin(2 * np.pi * hz * t)).astype("float32")


def _master(tmp_path, files, ratings=None, sets=None):
    """files: [dict(cat, fam, fn, y, src, **entry fields)] -> (master_dir, store_path); sets:
    [dict(fam, fn, came_from)] kit files (copies of the curated file)."""
    root = tmp_path / "master"
    cats, fams = {}, {}
    for f in files:
        f = dict(f)
        cat, fam, fn, y, src = (f.pop(k) for k in ("cat", "fam", "fn", "y", "src"))
        d = root / cat / fam
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / fn, y, SR, subtype="PCM_24")
        md5 = hashlib.md5((d / fn).read_bytes()).hexdigest()
        cats.setdefault(cat, {"families": 0, "files": 0, "source_samples": 0, "entries": []})
        cats[cat]["entries"].append({"family": fam, "out": f"{fam}/{fn}", "src": f"{LIB}/{src}",
                                     "out_md5": md5, **f})
        fams.setdefault(cat, {}).setdefault(fam, 0)
        fams[cat][fam] += 1
    for cat, fl in fams.items():
        cats[cat]["families"] = len(fl)
        cats[cat]["files"] = len(cats[cat]["entries"])
        (root / cat / "_manifest.json").write_text(json.dumps(
            [{"family": f, "copied": n, "n": n * 3, "clap": None, "traits": []} for f, n in fl.items()]))
    man = {"categories": cats}
    if sets:
        ents = []
        by = {f"{c}/{e['out']}": e for c, cd in cats.items() for e in cd["entries"]}
        for s in sets:
            src = root / s["came_from"]
            dst = root / "KITS" / s["fam"] / s["fn"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
            ents.append({"family": s["fam"], "out": f"{s['fam']}/{s['fn']}", "src": by[s["came_from"]]["src"],
                         "out_md5": by[s["came_from"]]["out_md5"], "came_from": s["came_from"]})
        man["sets"] = {"KITS": {"entries": ents}}
    (root / "manifest.json").write_text(json.dumps(man))
    store = tmp_path / "ratings.json"
    store.write_text(json.dumps({"version": 1, "ratings": ratings or {}}))
    return str(root), str(store)


def _ctx(tmp_path, files, ratings=None, sets=None):
    root, store = _master(tmp_path, files, ratings, sets)
    return _Ctx(root, store_path=store)


def _got(results, check):
    """The results of the checks whose names start with `check`."""
    return [r for r in results if r.check.startswith(check)]


def _level(results, check):
    got = _got(results, check)
    assert got, f"no check {check!r} in {[r.check for r in results]}"
    return got[0].level


# --- 1. copies: never a single cycle by its audio, and never without its keeper -----------------

def _wave(tmp_path, ids):
    files = [dict(cat="WAVES", fam="cycle-qzv", fn="Qzv_Tri.wav", y=_tone(0.02), src="Acme/Waves/Qzv Tri.wav")]
    ctx = _ctx(tmp_path, files)
    ctx.id_path = {k: f"{LIB}/{v}" for k, v in ids.items()}
    return ctx


def test_a_single_cycle_left_out_as_a_copy_of_another_wave_fails(tmp_path):
    ctx = _wave(tmp_path / "a", {1: "Acme/Waves/Qzv Sine.wav", 2: "Acme/Waves/Qzv Tri.wav"})
    why_log.write(ctx.root, "WAVES", {"category": "WAVES", "out": {"copy": {"1": 2}}})
    res = list(check_copies(ctx))
    assert _level(res, "no single cycle left out as a copy") == FAIL
    assert "Qzv Sine.wav" in _got(res, "no single cycle")[0].detail
    assert _level(res, "every copy left out has its keeper") == PASS
    # an edition of the same cycle (a .wav beside its .aif) may leave
    ctx = _wave(tmp_path / "b", {1: "Acme/Waves/Qzv Tri.aif", 2: "Acme/Waves/Qzv Tri.wav"})
    why_log.write(ctx.root, "WAVES", {"category": "WAVES", "out": {"copy": {"1": 2}}})
    assert _level(list(check_copies(ctx)), "no single cycle left out as a copy") == PASS


def test_a_copy_whose_keeper_left_too_fails(tmp_path):
    files = [dict(cat="SNARES", fam="tight", fn="Vqxw_Snare_Bzt_2.wav", y=_tone(), src="Acme/Kit/Vqxw Snare Bzt 2.wav")]
    ctx = _ctx(tmp_path, files)
    ctx.id_path = {1: f"{LIB}/Acme/Kit/Vqxw Snare Bzt 1.wav", 2: f"{LIB}/Northwind/Kit/Vqxw Snare Bzt 1.wav",
                   3: f"{LIB}/Acme/Kit/Vqxw Snare Bzt 3.wav", 4: f"{LIB}/Acme/Kit/Vqxw Snare Bzt 2.wav"}
    why_log.write(ctx.root, "SNARES", {"category": "SNARES", "out": {"copy": {"1": 2, "3": 4}}})
    res = list(check_copies(ctx))
    assert _level(res, "every copy left out has its keeper") == FAIL
    det = _got(res, "every copy left out")[0].detail
    assert det.startswith("1:") and "Vqxw Snare Bzt 1.wav" in det and "Vqxw Snare Bzt 3.wav" not in det
    # without the database the ids can't be read: a warning, never a pass or a fail
    ctx.id_path = None
    assert [r.level for r in check_copies(ctx)] == [WARN]


# --- 2. the set cap -------------------------------------------------------------------------------

def test_a_folder_over_its_set_cap_fails(tmp_path):
    files = [dict(cat="FX", fam="riser-bright", fn=f"Qxvwz_Rise_{v}_0{i}.wav", y=_tone(1.0),
                  src=f"Acme/Fx/Qxvwz Rise {v} 0{i}.wav") for i, v in enumerate("ABC")]
    ctx = _ctx(tmp_path, files)
    res = list(check_routing(ctx))
    assert _level(res, "at most 3 notes of one multisample set a folder") == FAIL
    assert "qxvwz rise" in _got(res, "at most 3 notes")[0].detail


# --- 3. one note of a set on each C ---------------------------------------------------------------

def _synth(fn, src, **kw):
    return dict(dict(cat="SYNTH", fam="bright-pad", fn=fn, y=_tone(), src=src), **kw)


def test_two_notes_of_one_set_on_one_c_fail(tmp_path):
    files = [_synth("Qzv_Pad_C3.wav", "Acme/Pads/Qzv Pad A2.wav", retune=3.0, root_src="name"),
             _synth("Qzv_Pad_C3_2.wav", "Acme/Pads/Qzv Pad C3.wav", retune=0.0, root_src="name"),
             _synth("Qzv_Pad_C3_3.wav", "Acme/Pads/Qzv Pad C3 v2.wav", retune=0.0, root_src="name")]
    res = list(check_pitch(_ctx(tmp_path / "a", files)))
    assert _level(res, "one note of a multisample set on each C") == FAIL
    assert _got(res, "one note of")[0].detail.startswith("3:")
    # the velocity layers of one note are that note
    assert _level(list(check_pitch(_ctx(tmp_path / "b", files[1:]))), "one note of") == PASS
    # a Keep stays, as in the build, and so does the note it keeps
    keep = {f"{LIB}/Acme/Pads/Qzv Pad A2.wav": {"category": "SYNTH", "verdict": "keep"},
            f"{LIB}/Acme/Pads/Qzv Pad C3.wav": {"category": "SYNTH", "verdict": "keep"},
            f"{LIB}/Acme/Pads/Qzv Pad C3 v2.wav": {"category": "SYNTH", "verdict": "keep"}}
    assert _level(list(check_pitch(_ctx(tmp_path / "c", files, keep))), "one note of") == PASS


def test_numbers_that_track_the_pitch_are_notes_on_one_c(tmp_path):
    # "Qzv Lead 059 / 061 / 062": detected roots 58.8, 60.8, 61.8, all retuned onto C4
    files = [_synth(f"Qzv_Lead_{n:03d}.wav", f"Acme/Leads/Qzv Lead {n:03d}.wav", retune=round(60 - r, 3),
                    root_src="detect", root_midi=r) for n, r in ((59, 58.8), (61, 60.8), (62, 61.8))]
    res = list(check_pitch(_ctx(tmp_path / "a", files)))
    assert _level(res, "one note of") == FAIL
    # numbers that don't track the roots are a series: other patches
    files = [_synth(f"Qzv_Lead_{n:03d}.wav", f"Acme/Leads/Qzv Lead {n:03d}.wav", retune=round(60 - r, 3),
                    root_src="detect", root_midi=r) for n, r in ((1, 58.8), (2, 60.8), (3, 61.8))]
    assert _level(list(check_pitch(_ctx(tmp_path / "b", files))), "one note of") == PASS


def test_numbers_read_by_pyin_where_sononym_isnt_sure(tmp_path):
    """verify reads a numbered set's roots as the build does (curate._number_root): a detected
    root, else Sononym's sure root, else pYIN's, so a set whose low notes Sononym can't read
    is still one set, and one file numbered a note off doesn't unmake it."""
    gtr = "Acme/Ms/Vqx_Gtr-{}.wav"
    files = [_synth(f"Vqx_Gtr_{n}.wav", gtr.format(n), retune=round(60 - (n - 13.82), 2), root_src="detect",
                    root_midi=n - 13.82, fam=fam) for n, fam in ((73, "warm"), (77, "plucky"))]
    files += [_synth("Vqx_Gtr_96.wav", gtr.format(96), retune=0.86, root_src="detect", root_midi=96 - 12.86,
                     fam="warm")]
    assert _level(list(check_pitch(_ctx(tmp_path / "a", files))), "one note of") == PASS   # two of three agree
    low = [_synth(f"Vqx_Gtr_{n}.wav", gtr.format(n), retune=round(36 - (n - 13.82), 2), fam="warm")
           for n in (53, 57)]
    ctx = _ctx(tmp_path / "b", files + low)
    ctx.pyin = {f"{LIB}/{gtr.format(n)}": n - 13.82 for n in (53, 57)}
    assert _level(list(check_pitch(ctx)), "one note of") == FAIL       # 73 and 77 both on C4
    ctx.pyin = None
    ctx.son_root = {f"{LIB}/{gtr.format(n)}": n - 13.82 for n in (53, 57)}
    assert _level(list(check_pitch(ctx)), "one note of") == FAIL       # (Sononym's sure root)


# --- 4. the category's retune cap -----------------------------------------------------------------

@pytest.mark.parametrize("cat, retune, level", [("VOX", 3.4, PASS), ("VOX", 4.0, FAIL), ("SYNTH", 5.9, PASS),
                                                ("PIANO", -3.6, FAIL)])
def test_a_retune_over_the_category_cap_fails(tmp_path, cat, retune, level):
    files = [dict(cat=cat, fam="bright", fn="Qzv_Ah_C3.wav", y=_tone(), src="Acme/Vox/Qzv Ah A2.wav",
                  retune=retune, root_src="name")]
    res = list(check_audio(_ctx(tmp_path, files)))
    assert _level(res, "tonal one-shots with a named root retuned to C") == level


# --- 5. a file named for C plays C ------------------------------------------------------------------

def test_a_file_named_for_c_that_plays_another_note_fails(tmp_path):
    files = [_synth("Qzv_Bell_C3.wav", "Acme/Bells/Qzv Bell C3.wav", retune=0.0, root_src="name")]
    ctx = _ctx(tmp_path / "a", files)
    src = f"{LIB}/Acme/Bells/Qzv Bell C3.wav"
    for heard, son, level in ((48.3, None, PASS), (36.0, None, PASS),       # an octave is the same note
                              (61.0, 61.2, FAIL), (61.0, 73.1, FAIL)):      # pYIN and Sononym agree: C#
        ctx.pyin, ctx.son_root = {src: heard}, ({src: son} if son is not None else None)
        assert _level(list(check_pitch(ctx)), "files retuned to C or named for it play C") == level, heard
    # one reading alone against the name, which the build's pitch trust leaves be (a retune or a
    # name needs two of name, Sononym and pYIN): a warning, whichever reading it is
    for py, son in ((61.0, None), (None, 61.0), (61.0, 60.1), (60.1, 61.0), (61.0, 63.0)):
        ctx.pyin, ctx.son_root = ({src: py} if py is not None else None), ({src: son} if son is not None else None)
        res = list(check_pitch(ctx))
        assert _level(res, "files retuned to C or named for it play C") == PASS, (py, son)
        assert _level(res, "files named for C that one pitch reading") == WARN, (py, son)
    ctx.son_root = None
    # pYIN a fifth over the name: maybe its 3rd harmonic, which the build leaves be
    ctx.pyin = {src: 55.0}
    res = list(check_pitch(ctx))
    assert _level(res, "files retuned to C or named for it play C") == PASS
    assert _level(res, "files named for C whose pitch reads a fifth") == WARN
    ctx.son_root = {src: 55.1}                       # ...even with Sononym hearing the fifth too
    res = list(check_pitch(ctx))
    assert _level(res, "files retuned to C or named for it play C") == PASS
    assert _level(res, "files named for C whose pitch reads a fifth") == WARN
    ctx.son_root = None
    # a chord ("Cmin"): pYIN and Sononym read one of its notes, and the build checks none
    files = [_synth("Qzv_Chop_Cmin_01.wav", "Acme/Gtr/Qzv Chop Cmin 01.wav", retune=0.0, root_src="name")]
    ctx = _ctx(tmp_path / "chord", files)
    ctx.pyin = ctx.son_root = {f"{LIB}/Acme/Gtr/Qzv Chop Cmin 01.wav": 51.1}
    res = list(check_pitch(ctx))
    assert _level(res, "files retuned to C or named for it play C") == PASS
    assert _level(res, "files named for a C chord or key") == WARN
    # a detected root retuned to C whose pYIN pitch says otherwise
    files = [_synth("Qzv_Chime.wav", "Acme/Bells/Qzv Chime.wav", retune=2.0, root_src="detect", root_midi=58.0)]
    ctx = _ctx(tmp_path / "b", files)
    assert _level(list(check_pitch(ctx)), "files retuned to C") == PASS        # by its own root: C
    ctx.pyin = {f"{LIB}/Acme/Bells/Qzv Chime.wav": 55.0}
    assert _level(list(check_pitch(ctx)), "files retuned to C") == FAIL       # pYIN: G + 2 = A


def test_verify_reads_the_builds_pitch_cache(tmp_path):
    """No pitch tracker in verify: the build's own pYIN readings, from its cache."""
    src = tmp_path / "lib" / "Qzv Bell C3.wav"
    src.parent.mkdir()
    sf.write(src, _tone(), SR)
    files = [_synth("Qzv_Bell_C3.wav", "x.wav", retune=0.0, root_src="name")]
    root, store = _master(tmp_path, files)
    m = json.loads((tmp_path / "master" / "manifest.json").read_text())
    m["categories"]["SYNTH"]["entries"][0]["src"] = str(src)
    (tmp_path / "master" / "manifest.json").write_text(json.dumps(m))
    cf = C._pitch_cache_file(str(src))
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps({"midi": 49.0}))
    assert C.cached_pitch(str(src)) == 49.0
    _ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    assert _level(res, "files named for C that one pitch reading") == WARN        # pYIN's alone


# --- 6. loops: whole beats within 10 ms, inside their folder's tempo -------------------------------

def _loop(seconds, sound=None, lead=0.0, fam="120-125bpm-clean", bpm=120.0, fn="Qzv_Groove.wav"):
    n = int(round(SR * seconds))
    y = np.zeros(n, dtype="float32")
    rng = np.random.default_rng(1)
    beat = int(SR * 60 / bpm / 2)
    end = int(SR * (sound if sound is not None else seconds))
    for k in range(int(SR * lead), end - 2000, beat):
        hit = rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300.0)
        y[k:k + 2000] += 0.7 * hit.astype("float32")
    y[-200:] *= np.linspace(1, 0, 200, dtype="float32")
    return dict(cat="DRUMLOOPS", fam=fam, fn=fn, y=y, src=f"Acme/Loops/{fn}", bpm=bpm, bpm_src="sononym",
                bpm_fold=bpm, loop_row=True)


def test_a_loop_20_ms_off_whole_beats_fails(tmp_path):
    res = list(check_routing(_ctx(tmp_path / "a", [_loop(2.0)])))
    assert _level(res, "loops are whole beats at their tempo") == PASS
    res = list(check_routing(_ctx(tmp_path / "b", [_loop(2.02)])))          # 0.04 beats: under BAR_TOL_BEATS
    assert _level(res, "loops are whole beats at their tempo") == FAIL


def test_a_loop_outside_its_folders_tempo_fails(tmp_path):
    assert _level(list(check_routing(_ctx(tmp_path / "a", [_loop(2.0)]))), "loops inside their folder") == PASS
    files = [_loop(2.0, bpm=120.0, fam="125-130bpm-clean")]
    assert _level(list(check_routing(_ctx(tmp_path / "b", files))), "loops inside their folder") == FAIL
    files = [_loop(2.0, bpm=120.0, fam="124bpm-clean")]                    # one tempo: within its spread
    assert _level(list(check_routing(_ctx(tmp_path / "c", files))), "loops inside their folder") == FAIL
    files = [_loop(2.0, bpm=120.0, fam="122bpm-clean")]
    assert _level(list(check_routing(_ctx(tmp_path / "d", files))), "loops inside their folder") == PASS


# --- 7. loops: not mostly silence, no late start on digital silence --------------------------------

def test_a_mostly_silent_loop_fails(tmp_path):
    res = list(check_audio(_ctx(tmp_path / "a", [_loop(2.0)])))
    assert _level(res, "no loop more than") == PASS
    res = list(check_audio(_ctx(tmp_path / "b", [_loop(2.0, sound=0.6)])))   # one bar, then nothing
    assert _level(res, "no loop more than") == FAIL


def test_a_loop_starting_late_on_digital_silence_fails(tmp_path):
    res = list(check_audio(_ctx(tmp_path / "a", [_loop(2.0, lead=0.025)])))
    assert _level(res, "no loop starts") == FAIL
    assert _level(list(check_audio(_ctx(tmp_path / "b", [_loop(2.0)]))), "no loop starts") == PASS


def test_late_start_needs_digital_silence_in_the_rotation_window():
    y = np.zeros((SR, 1))
    y[int(SR * 0.025):] = 0.5
    assert _late_start(y, SR) == pytest.approx(0.024, abs=0.002)
    y2 = y.copy()
    y2[:int(SR * 0.025)] = 0.001                  # a faint floor (-60 dBFS), not digital silence
    assert _late_start(y2, SR) == 0.0
    y3 = np.zeros((SR, 1))
    y3[int(SR * 0.080):] = 0.5                    # past ROTATE_MAX_MS: a rest, not a late cut
    assert _late_start(y3, SR) == 0.0


# --- 8, 9. waves and hats, as the build bands them -------------------------------------------------

def test_a_frame_chain_in_a_cycle_band_and_a_wave_outside_waves_fail(tmp_path):
    files = [dict(cat="WAVES", fam="cycle-qzv", fn="Qzv_(256x8).wav", y=_tone(2048 / SR), band="cycle",
                  src="Acme/Single Cycle/Qzv (256x8).wav"),
             dict(cat="SYNTH", fam="bright-pad", fn="Qzv_Saw.wav", y=_tone(0.02), src="Acme/Single Cycle/Qzv Saw.wav")]
    ctx = _ctx(tmp_path, files)
    from fourier.packs.verify import check_bands
    assert _level(list(check_bands(ctx)), "WAVES files are in the right wave band") == FAIL
    assert _level(list(check_routing(ctx)), "no waves outside WAVES") == FAIL


def test_a_hat_banded_against_its_folder_fails(tmp_path):
    files = [dict(cat="HATS", fam="open-qzv", fn="zq_hx005.wav", y=_tone(1.5, 6000), band="open",
                  src="Acme/Kit/Closed Hat/zq_hx005.wav")]
    from fourier.packs.verify import check_bands
    assert _level(list(check_bands(_ctx(tmp_path, files))), "HATS files are in the right hat band") == FAIL


# --- 10. kit slots ------------------------------------------------------------------------------------

def _drum(cat, fn, band=None, db=-1.0, fam="f"):
    t = np.arange(int(SR * 0.4)) / SR
    y = np.random.default_rng(len(fn)).standard_normal(t.size) * np.exp(-8 * t)
    y = (y / np.abs(y).max() * 10 ** (db / 20)).astype("float32")
    return dict(cat=cat, fam=fam, fn=fn.replace(" ", "_") + ".wav", y=y, src=f"Acme/Qzv Kit/{fn}.wav",
                **({"band": band} if band else {}))


def _kit_files():
    return [_drum("KICKS", "Qzv Kick 1"), _drum("SNARES", "Qzvwx Snare Bzt 1"),
            _drum("HATS", "Qzv Closed Hat 1", "closed"), _drum("HATS", "Qzv Open Hat 1", "open"),
            _drum("HATS", "Qzv Hat PC 2", "open"),
            _drum("PERC", "Qzvwx Shaker Bzt 1", db=-40.0), _drum("PERC", "Qzv Conga 1"),
            _drum("TOMS", "Qzv Tom Brush A 1"), _drum("TOMS", "Qzv Tom Stick A 2"), _drum("TOMS", "Qzv Tom Low 1")]


def _kit_results(tmp_path, picks):
    files = _kit_files()
    sets = [dict(fam="kit-qzv", fn=f["fn"], came_from=f"{f['cat']}/{f['fam']}/{f['fn']}")
            for f in files if f["fn"].replace("_", " ")[:-4] in picks]
    ctx = _ctx(tmp_path, files, sets=sets)
    ent = {f"{c}/{e['out']}": (c, e) for c, e in ctx.entries}
    return list(_check_kit_slots(ctx, ent))


def test_kit_slots_by_role_level_and_take(tmp_path):
    good = {"Qzv Kick 1", "Qzvwx Snare Bzt 1", "Qzv Closed Hat 1", "Qzv Open Hat 1", "Qzv Conga 1",
            "Qzv Tom Brush A 1", "Qzv Tom Low 1"}
    res = _kit_results(tmp_path / "a", good)
    assert all(r.level == PASS for r in res), [(r.check, r.detail) for r in res if r.level != PASS]
    # a pack-coded closed hat ("PC") in the open-hat slot
    res = _kit_results(tmp_path / "b", good - {"Qzv Open Hat 1"} | {"Qzv Hat PC 2"})
    assert _level(res, "kit slots filled by their role") == FAIL
    # a percussion take 25 dB under its role's level, when the pack has another
    res = _kit_results(tmp_path / "c", good - {"Qzv Conga 1"} | {"Qzvwx Shaker Bzt 1"})
    assert _level(res, "no kit slot more than") == FAIL
    # two takes of one tom (brush and stick of "Tom A"), when the pack has another tom
    res = _kit_results(tmp_path / "d", good - {"Qzv Tom Low 1"} | {"Qzv Tom Stick A 2"})
    assert _level(res, "no kit holds two takes of one drum") == FAIL


# --- 12. a rating reaches the recording's editions and the set --------------------------------------

def test_an_edition_of_a_drop_and_a_set_mate_of_a_misfile_fail(tmp_path):
    files = [dict(cat="KICKS", fam="punchy", fn="Qx_Hit_87.wav", y=_tone(0.3, 60), src="Acme/Kit/Qx Hit 87.aif"),
             dict(cat="PADS", fam="warm", fn="Qzv_Pad_E3.wav", y=_tone(2.0), src="Acme/Pads/Qzv Pad E3.wav")]
    r = {f"{LIB}/Acme/Kit/Qx Hit 87.wav": {"category": "KICKS", "verdict": "drop"},
         f"{LIB}/Acme/Pads/Qzv Pad C3.wav": {"category": "PADS", "verdict": "misfiled"}}
    res = list(check_ratings(_ctx(tmp_path / "a", files, r)))
    assert _level(res, "no Drop in the master") == FAIL
    assert "Qx Hit 87.wav" in _got(res, "no Drop")[0].detail
    assert _level(res, "no Misfiled file back in its folder") == FAIL
    # a Keep on the set mate wins, as in the build
    r[f"{LIB}/Acme/Pads/Qzv Pad E3.wav"] = {"category": "PADS", "verdict": "keep"}
    res = list(check_ratings(_ctx(tmp_path / "b", files, r)))
    assert _level(res, "no Misfiled file back in its folder") == PASS


def test_a_pyin_fill_never_lands_on_a_c_its_numbered_set_plays(monkeypatch):
    """The build side of one note per C: a file with no root, retuned by its pYIN pitch, keeps
    its pitch where another note of its set (numbers that track the roots) already plays that C."""
    monkeypatch.delenv("FOURIER_NO_PITCH_CHECK", raising=False)
    lead = f"{LIB}/Acme/Leads/Qzv Lead"
    rec = [dict(path=f"{lead} {n:03d}.wav", retune=round(60 - r, 3), root_src="detect", tune=r)
           for n, r in ((59, 58.8), (61, 60.8), (62, 61.8))]
    rec += [dict(path=f"{lead} 060.wav", retune=None, tune=None), dict(path=f"{lead} 070.wav", retune=None, tune=None)]
    heard = {"Qzv Lead 060.wav": 59.9, "Qzv Lead 070.wav": 69.9}
    monkeypatch.setattr(C, "_detect_pitch", lambda p: heard.get(p.rsplit("/", 1)[-1]))
    assert C._fill_roots("SYNTH", list(range(len(rec))), rec) == 1
    assert rec[3]["retune"] is None                         # C4: Lead 059 / 061 / 062 play it
    assert rec[4]["retune"] == pytest.approx(2.1) and rec[4]["root_src"] == "pyin"   # C5: free


def test_a_bare_c_the_build_cant_check_warns(tmp_path):
    """A name's note without an octave ("Qzv Bass 01 C", read without Sononym) isn't checked
    against the audio by the build: a pitch reading elsewhere warns instead of failing."""
    files = [dict(cat="SUB", fam="bass", fn="Qzv_Bass_01_C.wav", y=_tone(), src="Acme/Bass/Qzv Bass 01 C.wav",
                  retune=0.0, root_src="name")]
    ctx = _ctx(tmp_path, files)
    ctx.fallback = True
    ctx.pyin = {f"{LIB}/Acme/Bass/Qzv Bass 01 C.wav": 33.0}
    res = list(check_pitch(ctx))
    assert _level(res, "files retuned to C or named for it play C") == PASS
    assert _level(res, "files named for a C chord or key, or a C without an octave") == WARN


def test_a_note_left_for_its_set_mate_left_for_one_the_master_holds(tmp_path):
    files = [_synth("Qzv_Pad_C3.wav", "Acme/Pads/Qzv Pad C3.wav", retune=0.0, root_src="name")]
    ctx = _ctx(tmp_path, files)
    why_log.write(ctx.root, "SYNTH", {"category": "SYNTH", "out": {"one_note_per_c": {"7": ["Qzv Pad C3.wav", "C3"]}}})
    assert _level(list(check_one_note_kept(ctx)), "a note left out for another note") == PASS
    why_log.write(ctx.root, "SYNTH", {"category": "SYNTH", "out": {"one_note_per_c": {"7": ["Qzv Pad A2.wav", "C3"]}}})
    res = list(check_one_note_kept(ctx))
    assert _level(res, "a note left out for another note") == FAIL
    assert "Qzv Pad A2.wav" in _got(res, "a note left out")[0].detail
