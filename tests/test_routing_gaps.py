"""Routing gaps a whole-library review found: named risers a classifier or Live also calls a
loop, or Live tags as a guitar, or whose name repeats a "Pad" pack's name (rules._is_fx_transition:
compute_homes, why and verify); a hat banded by its folder; a kit's open-hat slot and slots far
under their role's level (sets); a pack prefix in a folder naming two drums; frame chains
named "(256xN)"; snare folders without a tune word; drums-only loops in a style-excluded
pack (rules._style_excluded)."""
import json
import os
import re

import numpy as np
import pytest
import soundfile as sf

from fourier.packs import curate as C
from fourier.packs import naming as N
from fourier.packs import sets as S
from fourier.packs.curate_config import CATEGORIES

from .test_curate import _loops
from .test_curate import _row as _srow
from .test_routing_names import _row
from .test_verify import _clean, _hit, _master

TRN_CHECK = "named risers / downlifters / sweeps only in FX"


# --- named transitions ------------------------------------------------------------------------

def test_a_pack_name_in_the_file_name_isnt_a_sweep_pad():
    rel = "Acme/Qzv Pad Works by Vqx Zed/Samples/One-Shots/riser 1 - Northwind X - Vqx Zed Qzv Pad Works.wav"
    fn = rel.rsplit("/", 1)[-1]
    assert C._is_fx_transition(fn, rel)
    assert not C._is_fx_transition(fn)                                 # without its folders: a pad
    assert not C._is_fx_transition("Sweep Pad 3.wav", "Acme/Sweep Pad/Sweep Pad 3.wav")
    assert not C._is_fx_transition("Riser - Murk Pad.wav", "Acme/Pad Risers/Riser - Murk Pad.wav")
    assert not C._is_fx_transition("Riser - Pad.wav", "Acme/Pad/Riser - Pad.wav")   # one word: kept


@pytest.mark.parametrize("fn, classes, canon, tags", [
    # a classifier hears a one-shot and a loop, Live a loop
    ("Qzv Uplifter 7.wav", ["OneShot", "Loop"], {"class.oneshot", "class.loop", "pad"}, ["Atmosphere", "Loop"]),
    # a one-shot Live tags a loop, named as a chord too (not a stab)
    ("Riser Qzv Chord.wav", ["OneShot"], {"class.oneshot", "pad"}, ["Atmosphere", "Loop"]),
    # a one-shot Live tags as a guitar (an acoustic reservation)
    ("Qzv Downlifter 12.wav", ["OneShot"], {"class.oneshot", "lead"}, ["Electric Guitar", "Loop"]),
])
def test_why_puts_a_named_transition_in_fx(tmp_path, fn, classes, canon, tags):
    from fourier.packs.why import explain_row
    r = _row(fn, canon, tags)._replace(classes=classes)
    w = explain_row(r, store_path=str(tmp_path / "none.json"))
    assert w.decided == ("fx_transition", "FX"), w.matched


def test_why_leaves_a_loop_by_its_class_alone(tmp_path):
    from fourier.packs.why import explain_row
    r = _row("Qzv Riser Loop.wav", {"class.loop", "pad"}, ["Loop"])._replace(classes=["Loop"])
    w = explain_row(r, store_path=str(tmp_path / "none.json"))
    assert "fx_transition" not in {m[0] for m in w.matched}, w.matched


def _verify_with(tmp_path, files, meta):
    """Verify FAILs for a master whose entries carry {filename: (labels, ableton, loop_row)}."""
    from fourier.packs.verify import FAIL, verify_master
    root, store = _master(tmp_path, files)
    p = os.path.join(root, "manifest.json")
    man = json.load(open(p))
    for cd in man["categories"].values():
        for e in cd["entries"]:
            lab, ab, lr = meta.get(os.path.basename(e["src"]), ((), (), False))
            e["labels"], e["ableton"] = sorted(lab), list(ab)
            if lr:
                e["loop_row"] = True
    json.dump(man, open(p, "w"))
    _ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    return {r.check for r in res if r.level == FAIL}


def test_verify_holds_the_transition_rule_on_the_shapes_the_build_takes(tmp_path):
    meta = {"Qzv Uplifter 3.wav": ({"class.oneshot", "class.loop", "pad"}, ["Loop"], True),
            "Qzv Downlifter 4.wav": ({"class.oneshot"}, ["Electric Guitar", "Loop"], True),
            "Qzv Riser Loop.wav": ({"class.loop", "pad"}, ["Loop"], True)}
    bad = _clean() + [("PADS", "warm", "Qzv Uplifter 3.wav", _hit(), "V4/P/Qzv Uplifter 3.wav")]
    assert TRN_CHECK in _verify_with(tmp_path / "a", bad, meta)
    bad = _clean() + [("ACOUSTIC", "plucked", "Qzv Downlifter 4.wav", _hit(), "V4/P/Qzv Downlifter 4.wav")]
    assert TRN_CHECK in _verify_with(tmp_path / "b", bad, meta)
    ok = _clean() + [("PADS", "warm", "Qzv Riser Loop.wav", _hit(), "V4/P/Qzv Riser Loop.wav")]
    assert TRN_CHECK not in _verify_with(tmp_path / "c", ok, meta)


def test_acoustic_leaves_a_named_transition_to_fx():
    rows = [_srow(id=1, rel_path="Acme/Qzv FX/Qzv Downlifter 12.wav", path="/lib/Acme/Qzv FX/Qzv Downlifter 12.wav",
                  filename="Qzv Downlifter 12.wav", classes=["OneShot"], categories=[], canonical=["class.oneshot"],
                  ableton_tags=["Electric Guitar", "Loop"], n_events=1, duration_s=5.0)]
    E = np.ones((1, 512), dtype="float32") / np.sqrt(512)
    why = {}
    C._select_records(rows, CATEGORIES["ACOUSTIC"], E, {1: 0}, category="ACOUSTIC", why=why)
    assert why.get("named_elsewhere") == {1: "FX"}, why


# --- a hat's folder ---------------------------------------------------------------------------

def test_a_hat_takes_its_folders_band_when_its_name_says_neither():
    assert C._hat_band("zq_qx005.wav", 0.8, "Acme/Kit/Qzv Takes/Closed Hat/zq_qx005.wav") == "closed"
    assert C._hat_band("Qx_3-02.wav", 0.6, "Acme/Qx/Hats/Tuned/Closed/Qx_3-02.wav") == "closed"
    assert C._hat_band("zq_qx006.wav", 0.1, "Acme/Kit/Open_Hat/zq_qx006.wav") == "open"
    # the name first; a folder naming both or neither: the length
    assert C._hat_band("Qzv Open 2.wav", 0.1, "Acme/Kit/Closed Hat/Qzv Open 2.wav") == "open"
    assert C._hat_band("zq_qx007.wav", 0.8, "Acme/Kit/Open and Closed Hats/zq_qx007.wav") == "open"
    assert C._hat_band("zq_qx007.wav", 0.8, "Acme/Kit/Hats/zq_qx007.wav") == "open"
    assert C._hat_band("zq_qx007.wav", 0.8) == "open"


# --- kit slots --------------------------------------------------------------------------------

H = lambda n: ("HATS", dict(src=f"/x/{n}.wav", out=f"f/{n}.wav", band="open"))
_names = lambda cs: [os.path.basename(ce[1]["src"]) for ce in cs]


@pytest.mark.parametrize("pedal", ["Hat Pedal Qz", "Qz Foot Hat 2", "Qz Hat PC 3", "zq hc017", "Qz_PC_Hat"])
def test_the_open_hat_slot_skips_a_pedal_or_closed_take(pedal):
    assert _names(S._slot_candidates("hat_open", [H(pedal), H("Hat Open Qz")])) == ["Hat Open Qz.wav"], pedal


def test_a_slot_passes_over_a_take_far_under_its_role_level(tmp_path):
    sr = 44100
    t = np.arange(int(sr * 0.5)) / sr
    noise = np.random.default_rng(3).standard_normal(t.size) * np.exp(-4 * t)
    cands = []
    for n, db in (("Hat A", -2.0), ("Hat B", -30.0), ("Hat C", -3.0)):
        d = tmp_path / "HATS" / "f"
        d.mkdir(parents=True, exist_ok=True)
        y = noise / np.abs(noise).max() * 10 ** (db / 20)
        sf.write(d / f"{n}.wav", y, sr)
        cands.append(H(n))
    quiet = S._quiet_check(tmp_path)
    assert [quiet(c) for c in cands] == [False, True, False]
    assert _names(S._spread_pick(cands, 1)) == ["Hat B.wav"]                  # the middle one, by name
    assert _names(S._spread_pick(cands, 1, quiet)) == ["Hat A.wav"]           # ...under its role: the nearest other
    only = [cands[1]]
    assert _names(S._spread_pick(only, 1, quiet)) == ["Hat B.wav"]            # no other: it stays
    assert S._quiet_check(None) is None


# --- a pack prefix in a folder naming two drums ----------------------------------------------

@pytest.mark.parametrize("rel, canon, want", [
    ("Acme/Kit/Kick (snare rattle)/sd_zq050.wav", None, "KICKS"),          # the drum named first
    ("Acme/Kit/Snare and Kick Qzv/bd_zq051.wav", None, "SNARES"),
    ("Acme/Kit/Kick (snare rattle)/sd_zq050.wav", {"class.oneshot", "snare"}, "SNARES"),   # the labels single one out
    ("Acme/Kit/Kick (snare rattle)/sd_zq050.wav", {"class.oneshot", "kick"}, "KICKS"),
    ("Acme/Kit/Vqx/sd_zq050.wav", None, "SNARES"),                             # no drum in the folder: the prefix
])
def test_a_prefix_in_a_folder_naming_two_drums(rel, canon, want):
    assert C._drum_named_in(rel.rsplit("/", 1)[-1], rel, False, canon, None) == want, rel


# --- frame chains -----------------------------------------------------------------------------

@pytest.mark.parametrize("fn, frames, band", [
    ("Qzv (128x8).wav", 1023, "table"),
    ("Qzv_(512x7).wav", 3583, "table"),
    ("Qzv (256x11).wav", 2816, "table"),
    ("Qzv (512x13).wav", 512, "cycle"),          # one frame of a table whose name the folder took
    ("Qzv_03.wav", 512, "cycle"),
    ("Qzv (256x1).wav", 256, "cycle"),
    ("Qzv 128x128.wav", 2048, "cycle"),          # the length doesn't hold the frames it names
])
def test_a_named_frame_chain_is_a_table(fn, frames, band):
    assert C._wave_band(frames / 44100, 44100, fn) == band, fn


# --- snare folder names ----------------------------------------------------------------------

def test_snare_folders_carry_no_tune_word():
    assert "tune" not in dict(N.naming_dims("SNARES", CATEGORIES["SNARES"]))
    for cat in ("KICKS", "TOMS"):
        assert "tune" in dict(N.naming_dims(cat, CATEGORIES[cat])), cat


# --- drums-only loops in a style-excluded pack -------------------------------------------------

STYLE = re.compile(r"(?<![a-z])(?:dubstep|brostep|trap|future[ _-]?bass|fx[ _-]?loops?|impact[ _-]?fx"
                   r"|synth[ _-]?bass|brass[ _-]?stab)(?![a-z])", re.I)


@pytest.mark.parametrize("rel, out", [
    # drums by the loop's own folder or name (or the folder above): the style word doesn't hold
    ("Acme/Qzv Trap/Drum Loops/Vqx_140_Kick_Snare.wav", None),
    ("Acme/Qzv/Dubstep Drum Loops Qzv/zq_drm150_vq_hats.wav", None),
    ("Acme/Qzv/Dubstep Drum Loops Qzv/zq_drm150_vq_full.wav", None),           # "full": the whole kit
    ("Acme/Qzv Live Kit Loops/Dubstep Qzv/Vqx 3.wav", None),
    ("Acme/Qzv Trap/QX_PERC_TOPS/Vqx 1.wav", None),
    ("Acme/Future Bass/Drums/Vqx Beat 3.wav", None),
    # no drums named near the loop, or a sound that isn't drums: still out
    ("Acme/Qzv Trap/Loops/Vqx 140 Full.wav", "Trap"),
    ("Acme/Qzv Trap/Drum Loops/Vqx_Synth_Bass_140.wav", "Trap"),
    ("Acme/Qzv Trap/Drum Loops/Vqx_Brass_Stab_Loop.wav", "Trap"),
    ("Acme/Qzv Trap/Drum Loops/Vqx Full Mix 140.wav", "Trap"),
    ("Acme/Qzv Trap/Music Loops/Vqx 140.wav", "Trap"),
    ("Acme/Qzv/FX Loops/Drum Fill 2.wav", "FX Loops"),
    ("Acme/Qzv/Drums/Synth Bass/Vqx 4.wav", "Synth Bass"),
    ("Acme/Qzv/Drum Loops/Vqx 4.wav", None),                                   # nothing to exclude
])
def test_a_style_exclusion_spares_drums_only_loops(rel, out):
    assert C._style_excluded(rel, STYLE) == out, rel


def test_the_drum_loops_take_a_drums_only_loop_from_a_style_excluded_pack(monkeypatch):
    rows, E, id2row = _loops(["Break 1", "Break 2", "Break 3"], monkeypatch)
    rows[0].rel_path = "A/Qzv Trap/Drum Loops/Break 1 Kick Snare 124bpm.wav"
    rows[1].rel_path = "A/Qzv Trap/Loops/Qzv 2 124bpm.wav"          # (its sound passes the gate)
    why = {}
    cfg = {**CATEGORIES["DRUMLOOPS"], "path_exclude": STYLE}
    rec, _fm, st = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", why=why)
    assert {r["id"] for r in rec} == {1, 3}
    assert why["style_excluded"] == {2: "Trap"} and st["style_excluded"] == 1
