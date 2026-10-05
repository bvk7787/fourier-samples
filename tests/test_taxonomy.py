"""Category taxonomy: renamed categories (MALLETS / BELLS / ORCHESTRAL into ACOUSTIC,
SCRATCHES into FX), ACOUSTIC plucked strings, bands (HATS closed/open, FX types, DRUMLOOPS
full/tops/classic), routing and naming rules, and the verify checks for each."""
import re
import json
from pathlib import Path

import numpy as np

from fourier.packs import curate as C
from fourier.packs import naming as N
from fourier.packs.curate_config import CATEGORIES, RENAMED_CATEGORIES

from .test_verify import LIB, _clean, _fails, _hit, _master

# --- categories ---------------------------------------------------------------------------

def test_renamed_categories():
    assert RENAMED_CATEGORIES == {"BELLS": "ACOUSTIC", "ORCHESTRAL": "ACOUSTIC", "MALLETS": "ACOUSTIC",
                                  "SCRATCHES": "FX"}
    assert "ACOUSTIC" in CATEGORIES
    assert not {"BELLS", "ORCHESTRAL", "MALLETS"} & set(CATEGORIES)


def test_tuned_percussion_overrides_to_acoustic_and_cowbell_to_perc():
    for fn in ("Marimba C3.wav", "Xylophone_A4.wav", "Vibraphone Soft.wav", "Glockenspiel 01.wav",
               "Steel Drum G.wav", "Celesta C5.wav", "Gamelan Gong.wav", "Toy Piano E4.wav"):
        assert C._name_override(fn) == "ACOUSTIC", fn
    for fn in ("Cowbell 808.wav", "Agogo Hi.wav"):
        assert C._name_override(fn) == "PERC", fn
    assert C._name_override("Synth Marimba.wav") is None          # a synth patch keeps its home


def test_plucked_strings_by_name():
    for fn in ("Guitar Pluck C3.wav", "Zither A2.wav", "Lyre 04.wav", "Harp Gliss.wav",
               "Cigar Box Pluck C2.wav", "Banjo Roll.wav", "Koto D4.wav"):
        assert C._is_plucked(fn), fn
    for fn in ("Bass Guitar E1.wav", "Harpsichord C4.wav", "Synth Guitar.wav", "Sharp Kick.wav",
               "Absolute Pad.wav"):
        assert not C._is_plucked(fn), fn
    assert C._plucked_label("Cigar Box Pluck C2") == "cigar-box"
    assert C._plucked_label("Acoustic Guitar A") == "guitar"


# --- bands ----------------------------------------------------------------------------------

def test_hat_band_name_first_then_length():
    assert C._hat_band("OH 909.wav", 0.1) == "open"
    assert C._hat_band("Closed Hat 03.wav", 0.8) == "closed"
    assert C._hat_band("Pedal Hat.wav", 0.6) == "closed"
    assert C._hat_band("Hat 12.wav", 0.12) == "closed"
    assert C._hat_band("Hat 12.wav", 0.6) == "open"


def test_fx_band_follows_the_source_library():
    assert C._fx_band(["XFX Explosions & Shots"], []) == "impact"
    assert C._fx_band(["XFX Nature & Athmospheric", "XFX Sweeps & Lasers"], []) == "ambience"
    assert C._fx_band([], ["Sweep"]) == "sweep"
    assert C._fx_band([], []) == "misc"


def test_loop_band_full_vs_tops():
    assert C._loop_band("Shaker Loop 120.wav", []) == "tops"
    assert C._loop_band("Hat Loop 03.wav", []) == "tops"
    assert C._loop_band("Amen Break 170.wav", []) == "classic"            # the classic breaks
    assert C._loop_band("Classic Break 03.wav", []) == "classic"
    assert C._loop_band("Break 170.wav", []) == "full"
    assert C._loop_band("Full Kit Hats 124.wav", []) == "full"            # "full"/"kit" wins
    assert C._loop_band("Loop 07.wav", ["Kick", "Closed Hihat"]) == "full"
    assert C._loop_band("Loop 07.wav", ["Conga"]) == "tops"
    assert C._loop_band("Loop 07.wav", []) == "full"


def test_band_of_dispatch():
    assert C._band_of("hat", "OH.wav") == "open"
    assert C._band_of("fx", "x.wav", son_labels=["XFX Cracks & Rustle"]) == "foley"
    assert C._band_of("loop", "Perc Loop.wav") == "tops"
    assert C._band_of("wave", "SAW.wav", duration=2048 / 44100, sample_rate=44100) == "cycle"
    assert C._band_of(None, "x.wav") is None
    for bt, bands in C.BAND_ORDER.items():
        assert len(bands) == len(set(bands)), bt


def test_band_phrase_masks():
    assert not C._band_phrase_ok("hat", "closed", "909 open hat")
    assert not C._band_phrase_ok("hat", "open", "tight closed hat")
    assert C._band_phrase_ok("hat", "open", "909 open hat")
    assert C._band_phrase_ok("fx", "impact", "boom explosion")
    assert not C._band_phrase_ok("fx", "impact", "ocean waves")
    assert C._band_phrase_ok("fx", "misc", "ocean waves")                 # misc: any FX phrase
    assert not C._band_phrase_ok("loop", "tops", "amen break")
    assert C._band_phrase_ok("loop", "full", "amen break")
    assert not C._band_phrase_ok("chord", "chord", "grand piano note")


def test_band_categories_configured():
    assert CATEGORIES["HATS"]["band"] == "hat"
    assert CATEGORIES["FX"]["band"] == "fx"
    assert CATEGORIES["DRUMLOOPS"]["band"] == "loop"
    assert CATEGORIES["PIANO"]["band"] == "chord"


# --- naming leads ---------------------------------------------------------------------------

def _cl(**kw):
    c = dict(idxs=[0], tune=None, bpm=None, bpm_range=None, free_tempo=False, source=None,
             br=0.5, noi=0.3, har=0.7, atk=20.0, dec=350.0, sub=0.0, cr=4.3)
    c.update(kw)
    return c


def _names(cat, clusters, Zp, phrases):
    cfg = CATEGORIES[cat]
    dims = N.naming_dims(cat, cfg)
    N.rank_traits(clusters, dims, cfg["kind"])
    return N.name_families(clusters, cat, cfg, np.array(Zp, float), phrases, dims=dims)


def test_hat_and_fx_folders_lead_with_their_band():
    cl = [_cl(band="closed", band_type="hat"), _cl(band="open", band_type="hat", dec=900.0)]
    names = _names("HATS", cl, [[2.0, 0], [0, 2.0]], ["tight closed hat", "909 open hat"])
    assert names[0].startswith("closed-") and names[1].startswith("open-")
    assert names[1].count("open") == 1                                    # not "open-909-open-hat"
    cl = [_cl(band="impact", band_type="fx"), _cl(band="ambience", band_type="fx")]
    names = _names("FX", cl, [[2.0, 0], [0, 2.0]], ["boom explosion", "ocean waves"])
    assert names[0].startswith("impact-boom") and names[1].startswith("ambience-ocean")


def test_tops_loops_keep_tempo_first():
    cl = [_cl(bpm=124.0, bpm_range=[122, 126], band="tops", band_type="loop"),
          _cl(bpm=124.0, bpm_range=[122, 126], band="full", band_type="loop"),
          _cl(free_tempo=True, band="tops", band_type="loop")]
    names = _names("DRUMLOOPS", cl, [[0, 2.0], [2.0, 0], [0, 0]], ["amen break", "shaker groove"])
    assert re.match(r"^\d{3}(-\d{3})?bpm-tops", names[0])
    assert "tops" not in names[1].split("-")
    assert names[2].startswith("freetempo-tops")


# --- verify -----------------------------------------------------------------------------------

def _with_bands(root, cat, bands, fam_band):
    mp, fp = Path(root) / "manifest.json", Path(root) / cat / "_manifest.json"
    m = json.loads(mp.read_text())
    for e in m["categories"][cat]["entries"]:
        e.update(bands.get(e["out"].split("/")[-1], {}))
    mp.write_text(json.dumps(m))
    fams = json.loads(fp.read_text())
    for f in fams:
        f["band"] = fam_band[f["family"]]
    fp.write_text(json.dumps(fams))


def _verify(root, store):
    from fourier.packs.verify import FAIL, verify_master
    _ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    return {r.check for r in res if r.level == FAIL}


def test_verify_hat_bands(tmp_path):
    files = _clean() + [("HATS", "closed-tight", "CH 01.wav", _hit(0.1), "V1/P/CH 01.wav"),
                        ("HATS", "open-909", "OH 01.wav", _hit(0.6), "V1/P/OH 01.wav")]
    root, store = _master(tmp_path, files)
    _with_bands(root, "HATS", {"CH 01.wav": {"band": "closed"}, "OH 01.wav": {"band": "open"}},
                {"closed-tight": "closed", "open-909": "open"})
    assert not {f for f in _verify(root, store) if f.startswith("HATS")}
    # an open hat recorded as closed, in a folder that leads with the wrong band
    root, store = _master(tmp_path / "b", files)
    _with_bands(root, "HATS", {"CH 01.wav": {"band": "closed"}, "OH 01.wav": {"band": "closed"}},
                {"closed-tight": "closed", "open-909": "closed"})
    fails = _verify(root, store)
    assert {"HATS files are in the right hat band", "HATS folder names lead with their band"} <= fails


def test_verify_fx_one_band_per_folder(tmp_path):
    files = _clean() + [("FX", "impact-boom", "a.wav", _hit(1.0), "V1/P/a.wav"),
                        ("FX", "impact-boom", "b.wav", _hit(1.0), "V1/P/b.wav")]
    root, store = _master(tmp_path, files)
    _with_bands(root, "FX", {"a.wav": {"band": "impact", "sononym": ["XFX Explosions & Shots"]},
                             "b.wav": {"band": "sweep", "sononym": ["XFX Sweeps & Lasers"]}},
                {"impact-boom": "impact"})
    assert "FX folders hold one fx band each" in _verify(root, store)


def test_verify_routing_rules_for_the_new_taxonomy(tmp_path):
    files = _clean() + [
        ("SYNTH", "c3-pluck-bright", "Acoustic Guitar C3.wav", _hit(), "V1/P/Acoustic Guitar C3.wav"),
        ("SYNTH", "c3-pluck-dark", "Marimba C3.wav", _hit(), "V1/P/Marimba C3.wav"),
        ("ACOUSTIC", "mallet-e5-bell", "Cowbell 808.wav", _hit(), "V1/P/Cowbell 808.wav"),
        ("FX", "misc-rumble", "Clap Verb.wav", _hit(1.0), "V1/P/Clap Verb.wav"),
    ]
    fails = _fails(tmp_path, files)[1]
    assert {"no plucked-string or tuned-percussion files in SYNTH (they live in ACOUSTIC)",
            "name overrides hold (tuned percussion in ACOUSTIC, cowbell/agogo in PERC, ...)",
            "name filters hold (no drum hits in FX, no kicks in SNARES, ...)"} <= fails
    keep = {f"{LIB}/V1/P/{n}": {"category": c, "verdict": "keep"} for c, n in
            [("SYNTH", "Acoustic Guitar C3.wav"), ("SYNTH", "Marimba C3.wav"),
             ("ACOUSTIC", "Cowbell 808.wav"), ("FX", "Clap Verb.wav")]}
    fails = _fails(tmp_path / "k", files, keep)[1]
    assert not any(f.startswith(("no plucked", "name overrides", "name filters")) for f in fails)


def test_verify_retired_category_and_plucked_names(tmp_path):
    files = _clean() + [("BELLS", "e5-bell", "Bell.wav", _hit(), "V1/P/Bell.wav"),
                        ("SYNTH", "c3-guitar-pluck", "s.wav", _hit(), "V1/P/s.wav")]
    fails = _fails(tmp_path, files)[1]
    assert "no plucked-string names in SYNTH" in fails
    assert any(f.startswith("no retired category folders") for f in fails)


def test_verify_loop_band_lead_and_loop_override_exemption(tmp_path):
    files = _clean() + [
        ("DRUMLOOPS", "090-100bpm-tops-tribal", "Shaker Loop 95.wav", _hit(2.0), "V1/P/Shaker Loop 95.wav"),
        ("DRUMLOOPS", "124bpm-full-break", "Full Break 124.wav", _hit(2.0), "V1/P/Full Break 124.wav"),
        ("VOX", "a3-sung-note", "Xylophone Texture 07.wav", _hit(), "V1/P/Xylophone Texture 07.wav"),
    ]
    root, store = _master(tmp_path, files)
    _with_bands(root, "DRUMLOOPS", {"Shaker Loop 95.wav": {"band": "tops"},
                                    "Full Break 124.wav": {"band": "full"}},
                {"090-100bpm-tops-tribal": "tops", "124bpm-full-break": "full"})
    mp = Path(root) / "manifest.json"
    m = json.loads(mp.read_text())
    m["categories"]["VOX"]["entries"][0]["ableton"] = ["Loop", "Solo Voice"]   # a loop: exempt
    mp.write_text(json.dumps(m))
    fails = _verify(root, store)
    assert not {f for f in fails if f.startswith(("DRUMLOOPS", "name overrides"))}, fails



# --- tops tempo bands, FX synth band, name vetoes, ACOUSTIC labels --------------------------

def test_tops_get_wide_tempo_bands():
    from fourier.packs.curate_config import TOPS_TEMPO_BANDS
    assert CATEGORIES["DRUMLOOPS"]["band_tempo"]["tops"] == TOPS_TEMPO_BANDS
    bpms = [92, 96, 104, 112, 121, 123, 126, 128, 134, 146, 160, 172, None]
    groups = C._tempo_band_groups(bpms, TOPS_TEMPO_BANDS, 1)
    assert [len(g) for g in groups] == [4, 2, 2, 2, 2, 1]                 # 5 tempo bands + freetempo


def test_tops_named_only_by_tops_phrases():
    assert C._band_phrase_ok("loop", "tops", "shaker loop")
    assert not C._band_phrase_ok("loop", "tops", "techno drum loop")
    assert not C._band_phrase_ok("loop", "full", "shaker loop")
    assert C._band_phrase_ok("loop", "full", "techno drum loop")


def test_fx_synth_band_and_drum_leak():
    assert C._fx_band(["Tone Leads & MidHiKeys"], ["Sound FX"]) == "synth"
    assert C._fx_band(["Perc Zaps & Blips"], []) == "synth"
    assert C._fx_band(["XFX Sweeps & Lasers", "Tone Leads & MidHiKeys"], []) == "sweep"   # library type wins
    assert C._fx_band(["Tone Pads & Textures"], ["Sound FX"]) == "misc"
    assert C._fx_drum_leak(["Perc Cymbal Crashes"], ["Sound FX", "One Shot"])
    assert not C._fx_drum_leak(["Perc Snares", "XFX Explosions & Shots"], [])
    assert not C._fx_drum_leak(["Perc Snares"], ["Impact"])
    assert not C._fx_drum_leak(["Tone Leads & MidHiKeys"], ["Sound FX"])


def test_name_filter_veto_moves_the_home():
    assert C._name_filter_veto({"SNARES", "PERC"}, "Kick 808.wav") == {"PERC"}
    assert C._name_filter_veto({"SNARES"}, "Kick 808.wav") == {"SNARES"}              # never empty
    assert C._name_filter_veto({"SNARES", "PERC"}, "Snare 01.wav") == {"SNARES", "PERC"}


def test_mallet_labels_and_acoustic_names():
    assert C._mallet_label("Xylophone B4 v2 rr1.wav") == "xylophone"
    assert C._mallet_label("Vibes 02 D2.wav") == "vibraphone"
    assert C._mallet_label("Arimba G2.wav") == "marimba"
    assert C._mallet_label("Toy Piano Gb3 v3.wav") == "toy-piano"
    assert C._mallet_label("Thumb Piano C3.wav") == "kalimba"
    assert C._mallet_label("Vibrato Lead.wav") is None
    for fn in ("Xylophone A4.wav", "Kalimba Tone C4.wav", "Church Bell C3.wav",
               "Tibetan Bowl Soft.wav", "Glockenspiel F5.wav"):
        assert C._is_acoustic_named(fn), fn
    for fn, tags in (("FM Bell C3.wav", []), ("Bell Lead 03.wav", []), ("Synth Marimba.wav", []),
                     ("Marimba Patch E5.wav", ["Synth Mallets"])):
        assert not C._is_acoustic_named(fn, tags), fn
    assert C._acoustic_name_label("Cigar Box A2") == "cigar-box"
    assert C._acoustic_name_label("Xylophone C3") == "xylophone"


def test_acoustic_bands(monkeypatch):
    b = C._acoustic_band
    # a pack of mallet sounds is a library rule (ACOUSTIC_MALLET_PACKS, empty in the code)
    monkeypatch.setitem(b.__globals__, "ACOUSTIC_MALLET_PACKS", re.compile("mallet pack", re.I))
    assert b("Xylophone C3.wav") == "mallet" and b("Timpani Hit 01.wav") == "mallet"
    assert b("Zither Hit C1.wav") == "plucked" and b("x.wav", ["Harp"]) == "plucked"
    assert b("Bb Clarinet Solo.wav") == "wind" and b("x.wav", ["Trumpet"]) == "wind"
    assert b("Violin Solo Staccato.wav") == "string" and b("x.wav", ["Cello"]) == "string"
    assert b("Kalimba C4.wav", ["Acoustic Guitar"]) == "mallet"           # the name decides first
    assert b("Tile Hit B3 01.wav", [], "Mallet Pack A") == "mallet"
    assert b("Long Sustain 01.wav", [], "Brass Pack A") == "wind"
    assert C.BAND_ORDER["acoustic"] == ("string", "wind", "plucked", "mallet")
    assert CATEGORIES["ACOUSTIC"]["band"] == "acoustic"


def test_acoustic_label_prefers_specific_tag_then_name_then_catch_all():
    tags = CATEGORIES["ACOUSTIC"]["ableton_any"]
    assert C._acoustic_label({"ab": ["Misc Plucked"], "plk": "zither"}, tags) == "zither"
    assert C._acoustic_label({"ab": ["Misc Plucked", "Harp"], "plk": "zither"}, tags) == "Harp"
    assert C._acoustic_label({"ab": ["Misc Strings"]}, tags) == "Misc Strings"
    assert C._acoustic_label({"ab": []}, tags) is None


def test_instrument_named_folders_keep_a_phrase_that_names_them():
    # name_instruments (generic): the instrument most files name, unless the phrase names it
    cfg = dict(CATEGORIES["STABS"], name_instruments=True)
    dims = N.naming_dims("STABS", cfg)
    cl = [_cl(tune=72, inst="xylophone"), _cl(tune=64, inst="bell"), _cl(tune=60)]
    N.rank_traits(cl, dims, cfg["kind"])
    names = N.name_families(cl, "STABS", cfg, np.array([[2.0, 0], [0, 2.0], [0, 0]], float),
                            ["music box", "church bell"], dims=dims)
    assert names[0].startswith("xylophone")                           # not "music box"
    assert names[1].startswith("church-bell")                         # the phrase names the bell


def test_verify_followup_checks(tmp_path):
    files = _clean() + [
        ("FX", "impact-boom", "Cymbal Hit 01.wav", _hit(1.0), "V1/P/Cymbal Hit 01.wav"),
        ("ACOUSTIC", "mallet-g7-bell", "Fat Bass 01.wav", _hit(0.5), "V1/P/Fat Bass 01.wav"),
        ("ACOUSTIC", "mallet-c5-music-box", "Xylophone C5.wav", _hit(0.6), "V1/P/Xylophone C5.wav"),
        ("ACOUSTIC", "misc-plucked", "Chord Chop 01.wav", _hit(), "V1/P/Chord Chop 01.wav"),
        ("ACOUSTIC", "misc-plucked", "Zither Hit C1.wav", _hit(), "V1/P/Zither Hit C1.wav"),
        ("ACOUSTIC", "misc-plucked", "Zither Hit D1.wav", _hit(), "V1/P/Zither Hit D1.wav"),
    ]
    root, store = _master(tmp_path, files)
    mp = Path(root) / "manifest.json"
    m = json.loads(mp.read_text())
    lab = {"Cymbal Hit 01.wav": (["Perc Cymbal Crashes"], ["Sound FX"]),
           "Fat Bass 01.wav": (["Tone Triangles & Bells"], ["Synth Bass"]),
           "Chord Chop 01.wav": ([], ["Misc Plucked"]),
           "Zither Hit C1.wav": ([], ["Misc Plucked"]), "Zither Hit D1.wav": ([], ["Misc Plucked"])}
    for cd in m["categories"].values():
        for e in cd["entries"]:
            son, ab = lab.get(e["out"].split("/")[-1], ([], []))
            e["sononym"], e["ableton"] = son, ab
    mp.write_text(json.dumps(m))
    fails = _verify(root, store)
    assert {"no kit drums in FX on Ableton's \"Sound FX\" alone",
            "no synth-tagged voices in ACOUSTIC",
            "ACOUSTIC catch-all tags backed by an instrument name or pack",
            "ACOUSTIC folders named for the instrument most files name"} <= fails



def test_override_cannot_route_a_wave_out_of_waves(monkeypatch):
    # a name override can't force a wave out of WAVES (a wavetable frame named for a gamelan)
    from tests.regressions.test_curate_regressions import _emb, _pool, _row, _run_select
    rows = _pool()
    wave = _row(100, "Vendor A", "Gamelan 05.wav"); wave.duration_s = 0.046
    wave.rel_path = "Vendor A/Wavetable Pack/General/Gamelan 05.wav"
    note = _row(101, "V1", "Gamelan Gong C3.wav")
    rows += [wave, note]
    got = _run_select(monkeypatch, rows, "ACOUSTIC", {r.id: "ACOUSTIC" for r in rows}, keeps={},
                      misfiled={}, emb=_emb(len(rows)))
    assert wave.path not in got and note.path in got



def test_impulse_responses_are_blocked(monkeypatch, tmp_path):
    for rel in ("Vendor A/Reverb Pack/IRs/Large Rooms/Hall 01.aif",
                "Vendor B/Effects Pack/Audio Effect/Reverb/IRs/x.aif",
                "Vendor C/Impulse Responses/Plate 01.wav", "Pack/IR/Room.wav"):
        assert C._is_ir(rel), rel
    for rel in ("Vendor A/Synth Kit/WAV/Bass/Sub/Sub Bass F1-IR.wav",
                "Vendor B/Synth Pack A/Impulse/Impulse 01.wav",
                "Vendor A/Sampler Kit/WAV/Synths/FX/Space/Space FX E3.wav",
                "Pack/Irish Whistle/C4.wav"):
        assert not C._is_ir(rel), rel
    # out of selection even as a Keep
    from tests.regressions.test_curate_regressions import _emb, _pool, _row, _run_select
    rows = _pool()
    ir = _row(100, "Vendor A", "Small Room 01.aif")
    ir.rel_path = "Vendor A/Reverb Pack/IRs/Small Rooms/Small Room 01.aif"
    ir.path = "/lib/SampleLibrary/" + ir.rel_path
    rows.append(ir)
    got = _run_select(monkeypatch, rows, "PERC", {r.id: "PERC" for r in rows}, keeps={ir.path: "PERC"},
                      misfiled={}, emb=_emb(len(rows)))
    assert ir.path not in got and len(got) == len(rows) - 1
    # and verify fails on one
    rel = "Vendor A/Reverb Pack/IRs/Large Rooms/Hall 01.aif"
    files = _clean() + [("FX", "impact-boom", "Hall 01.wav", _hit(1.0), rel)]
    assert "no reverb impulse responses" in _fails(tmp_path, files)[1]


def test_cap_folders_merges_nearest_within_a_band():
    rng = np.random.default_rng(0)
    P = rng.standard_normal((40, 8))
    P[20:] += 5.0                                                   # two clearly separate blobs
    groups = [list(range(0, 10)), list(range(10, 12)), list(range(12, 20)),
              list(range(20, 30)), list(range(30, 40))]
    keys = ["closed", "closed", "closed", "open", "open"]
    out = C._cap_folders(groups, keys, P, 3)
    assert len(out) == 3 and sorted(i for g in out for i in g) == list(range(40))
    assert all(set(g) <= set(range(20)) or set(g) <= set(range(20, 40)) for g in out)   # bands never mix
    # a band down to one folder and tempo folders (None) are never merged
    assert len(C._cap_folders(groups, ["a", "b", None, None, "c"], P, 2)) == 5


def test_folder_cap_config():
    from fourier.packs.curate_config import FOLDER_MAX
    assert FOLDER_MAX == 12 and C.MAX_PER_FAMILY >= 80
    assert CATEGORIES["DRUMLOOPS"]["folder_max"] == 32          # room for every 5-BPM band


def test_verify_flags_too_many_folders(tmp_path):
    files = _clean() + [("PADS", f"pad-{i}", f"p{i}.wav", _hit(1.0), f"V1/P/p{i}.wav") for i in range(14)]
    assert "folders per category within the cap (12)" in _fails(tmp_path, files)[1]


def test_category_folders_are_numbered_drums_first():
    from fourier.packs.curate_config import BUDGETS, CATEGORY_ORDER, category_dir
    from fourier.packs.render import _rel_dest
    assert set(CATEGORY_ORDER) == set(CATEGORIES) == set(BUDGETS)            # every category, once
    assert len(CATEGORY_ORDER) == len(set(CATEGORY_ORDER))
    assert category_dir("KICKS") == "01_KICKS" and category_dir("VOX") == "19_VOX" and category_dir("PHRASES") == "09_PHRASES"
    assert category_dir("SOMETHING") == "SOMETHING"
    assert CATEGORY_ORDER[:7] == ["KICKS", "SNARES", "CLAPS", "HATS", "CYMBALS", "TOMS", "PERC"]
    assert _rel_dest("KICKS", "909-house", "BD 1", 2).as_posix() == "01_KICKS/909-house/BD_1.wav"


def test_preset_and_kit_previews_are_blocked(monkeypatch, tmp_path):
    for rel in ("Vendor A/Orchestra Pack/Ableton Folder Info/Previews/Sounds/Tuba/Tuba Soft.adg.wav",
                "Vendor B/Kit Pack/[Previews]/Kit 01.xpm.mp3",
                "Vendor C/Guitar Pack/Samples/Guitar-Preset 01.adg.wav",
                "Kit Pack 2/Preview/Demo Loop.wav"):
        assert C._is_preview(rel), rel
    for rel in ("Vendor A/Orchestra Pack/Samples/Tuba/Tuba C2.wav",
                "Pack/Previewed Hits/Kick 1.wav", "Pack/Kicks/Kick Preview Style.wav"):
        assert not C._is_preview(rel), rel
    from tests.regressions.test_curate_regressions import _emb, _pool, _row, _run_select
    rows = _pool()
    pv = _row(100, "Vendor A", "Tuba Soft.adg.wav")
    pv.rel_path = "Vendor A/Orchestra Pack/Ableton Folder Info/Previews/Sounds/Tuba/Tuba Soft.adg.wav"
    pv.path = "/lib/SampleLibrary/" + pv.rel_path
    rows.append(pv)
    got = _run_select(monkeypatch, rows, "PERC", {r.id: "PERC" for r in rows}, keeps={pv.path: "PERC"},
                      misfiled={}, emb=_emb(len(rows)))
    assert pv.path not in got
    files = _clean() + [("ACOUSTIC", "tuba", "Tuba Soft.adg.wav", _hit(), pv.rel_path)]
    assert "no preset or kit previews" in _fails(tmp_path, files)[1]


def test_waves_come_from_wave_folders_only():
    # a bare "cycle" in a pack title doesn't make its drums waves; only wave folders count
    assert not C._is_wave("Vendor A/Breaks Pack/Samples/Loops/Cycle Drums/"
                          "Hits/Kick/Kick 03.wav", 0.14)
    assert not C._is_wave("Vendor B/Synth Pack A/Samples/One Shots/Misc/Glass Wavetable.wav", 0.5)
    assert C._is_wave("Vendor B/Synth Pack B/Samples/Warm Wavetable/Warm Wavetable 01.wav", 0.05)
    assert C._is_wave("Vendor C/Wavetable Pack/WAVETABLES/Synth A/Vox-C1.wav", 0.02)
    assert C._is_wave("Vendor D/Wavetable Pack/Wavetable Pack/Misc/Wave_31.wav", 0.05)
    assert C._is_wave("Free/Single Cycle Waves/Set 01/Wave 01.wav", 0.02)


def test_a_wave_twin_outside_a_wave_folder_stays_out(monkeypatch):
    from tests.regressions.test_curate_regressions import _emb, _pool, _row
    import fourier.packs.ratings as ratings
    rows = _pool()
    twin = _row(100, "Vendor D", "Wave_31.wav"); twin.duration_s = 0.046; twin.file_hash = "hw"
    rows.append(twin)
    monkeypatch.setattr(ratings, "keep_pins", lambda *a, **k: {})
    monkeypatch.setattr(ratings, "misfiled_map", lambda *a, **k: {})
    from tests.regressions.test_curate_regressions import _MisfiledIndex
    monkeypatch.setattr(C, "_misfiled_index", lambda m, e, i, pid=None: _MisfiledIndex(m))
    id2row = {r.id: j for j, r in enumerate(rows)}
    def sel(wh):
        rec, _, _ = C._select_records(rows, CATEGORIES["BLIPS"], _emb(len(rows)), id2row,
                                      homes={r.id: "BLIPS" for r in rows}, support={},
                                      category="BLIPS", wave_hashes=wh)
        return {d["path"] for d in rec}
    assert twin.path in sel(set())            # not under a wave folder: fine on its own
    assert twin.path not in sel({"hw"})       # but its twin is a wave: WAVES only


def test_verify_override_exempts_sononym_loops(tmp_path):
    # a vibraphone-named patch Sononym classes OneShot + Loop: the build exempts it
    # from the vibraphone override (as a loop) and it homes in SUB
    rel = "Vendor A/Synth Pack A/WAV/Patches/Vibraphone Patch 01.wav"
    files = _clean() + [("SUB", "saw-growl", "Vibraphone Patch 01.wav", _hit(0.7), rel)]
    name = "name overrides hold (tuned percussion in ACOUSTIC, cowbell/agogo in PERC, ...)"
    rel2 = "V9/P/Vibraphone 02.wav"             # (a synth-pack patch is exempt on its own)
    files = _clean() + [("SUB", "saw-growl", "Vibraphone 02.wav", _hit(0.7), rel2)]
    assert name in _fails(tmp_path, files)[1]
    root, store = _master(tmp_path / "l", files)
    mp = Path(root) / "manifest.json"
    m = json.loads(mp.read_text())
    m["categories"]["SUB"]["entries"][0]["loop_row"] = True
    mp.write_text(json.dumps(m))
    assert name not in _verify(root, store)


def test_acoustic_band_shares():
    cfg = CATEGORIES["ACOUSTIC"]
    assert abs(sum(cfg["band_share"].values()) - 1.0) < 1e-9 and cfg["band_min_folders"] == 2
    from fourier.packs.curate_config import BUDGETS
    assert BUDGETS["ACOUSTIC"] == 455
    # a small string pool still gets its share of the budget; pool size doesn't decide
    sizes = [300, 300, 40, 40, 60, 60, 50, 50]
    keys = ["plucked", "plucked", "string", "string", "wind", "wind", "mallet", "mallet"]
    a = C._allocate_banded(sizes, keys, cfg["band_share"], 400, 6, 120)
    got = {k: sum(x for x, kk in zip(a, keys) if kk == k) for k in set(keys)}
    assert sum(a) == 400
    assert got["string"] == 80 and got["plucked"] >= 100      # strings full (pool 80), rest spills
    assert all(x <= min(120, s) for x, s in zip(a, sizes))


def test_cap_folders_keeps_the_band_minimum():
    rng = np.random.default_rng(1)
    P = rng.standard_normal((40, 8))
    groups = [list(range(i * 5, i * 5 + 5)) for i in range(8)]
    keys = ["a"] * 6 + ["b"] * 2
    out = C._cap_folders(groups, keys, P, 4, min_per_key=2)
    assert len(out) == 4
    assert sum(1 for g in out if set(g) <= set(range(30, 40))) == 2    # band b kept both


def test_acoustic_keeps_synth_named_files_out():
    rx = CATEGORIES["ACOUSTIC"]["name_exclude"]
    for fn in ("Fm_Bass_01.wav", "Synth Strings C3.wav", "Saw Brass 01.wav", "Grand Piano C3.wav"):
        assert rx.search(fn), fn
    for fn in ("Upright Bass 01.wav", "Violin Solo Staccato.wav", "Xylophone C3.wav",
               "Seesaw Creak.wav", "Formant Flute.wav"):
        assert not rx.search(fn), fn


def test_nylon_string_is_plucked_and_tempo_key_names_are_phrases():
    from fourier.packs.curate_config import BPM_NAME_RE
    assert C._acoustic_band("string_nylon_01.wav") == "plucked"
    assert C._is_plucked("Steel String Gtr C3.wav")
    for fn in ("Strings_120_Amin.wav", "Loop_133_Dmin_Strings.wav",
               "Pad 90 F#m.wav", "Loop 120bpm.wav"):
        assert BPM_NAME_RE.search(fn), fn
    for fn in ("Violin C4.wav", "Upright Bass 01.wav", "808 Kick A1.wav", "Clap 01.wav",
               "Bass 7 C.wav", "Sub 100 Hz.wav", "take_100_amp.wav"):
        assert not BPM_NAME_RE.search(fn), fn


def test_name_words_name_acoustic_folders_before_catch_all_tags():
    assert C._acoustic_name_label("Bowed Metal D1 v1") == "bowed"
    assert C._acoustic_name_label("Strings Pizzicato F#2") == "string"
    assert C._acoustic_name_label("Bb Clarinet Solo") == "clarinet"
    tags = CATEGORIES["ACOUSTIC"]["ableton_any"]
    assert C._acoustic_label({"ab": ["Misc Plucked"], "plk": "bowed"}, tags) == "bowed"


# --- PHRASES: musical loops kept apart from the one-shots ----------------------------------

def _ph(rel, dur=7.742, classes='["Loop"]', ableton=None, categories=None, bpm=None, tempo=None, i=1):
    from types import SimpleNamespace as NS
    return NS(id=i, rel_path=rel, path=f"{LIB}/{rel}", filename=rel.split("/")[-1], duration_s=dur,
              ableton_tags=ableton, classes=classes, categories=categories, bpm=bpm, tempo_bpm=tempo)


def test_phrases_config_and_order():
    from fourier.packs.curate_config import (BUDGETS, CATEGORY_ORDER, PHRASE_ROLE_PHRASES,
                                             PHRASE_ROLE_SHARE, PHRASE_ROLES, category_dir)
    cfg = CATEGORIES["PHRASES"]
    assert cfg["kind"] == "loop" and cfg["band"] == "phrase" and cfg["tempo_bands"] == (60, 90, 120, 140, 201)
    assert CATEGORY_ORDER.index("PHRASES") == CATEGORY_ORDER.index("DRUMLOOPS") + 1
    assert category_dir("PHRASES") == "09_PHRASES" and BUDGETS["PHRASES"] == 720
    assert C.BAND_ORDER["phrase"] == PHRASE_ROLES == ("acid", "bass", "chords", "lead", "live")
    assert abs(sum(PHRASE_ROLE_SHARE.values()) - 1.0) < 1e-9 and set(PHRASE_ROLE_SHARE) == set(PHRASE_ROLES)
    allp = [p for ps in PHRASE_ROLE_PHRASES.values() for p in ps]
    assert len(allp) == len(set(allp))                  # a phrase names one role only
    assert C._band_phrase_ok("phrase", "bass", "deep house bassline")
    assert C._band_phrase_ok("phrase", "acid", "303 bassline")
    assert not C._band_phrase_ok("phrase", "bass", "string swell")
    assert not C._band_phrase_ok("phrase", "bass", "303 bassline")
    assert not C._band_phrase_ok("phrase", "live", "303 bassline")


def test_phrase_marked_by_name_or_loop_folder():
    assert C._phrase_marked("V/P/One Shots/Bass 124bpm.wav", "Bass 124bpm.wav", False) == "name"
    assert C._phrase_marked("V/P/x.wav", "Loop_120_Amin_Keys.wav", False) == "name"
    assert C._phrase_marked("L/Music Loops/Bass Loops/riff_Am.wav", "riff_Am.wav", True) == "folder"
    assert C._phrase_marked("L/WAV_LOOPS/MUSIC_LPS/x.wav", "x.wav", True) == "folder"
    assert C._phrase_marked("F/Loops (120BPM)/WAV/Bass - Synth A/Bass_01.wav", "Bass_01.wav", True) == "folder"
    # a loop folder narrowed below by a one-shot folder is not
    assert C._phrase_marked("A/Sample Loops/Chords/Electric Piano/EP C7.wav", "EP C7.wav", True) is None
    assert C._phrase_marked("A/Sample Loops/Music Loops/Keys/EP.wav", "EP.wav", True) == "folder"
    assert C._phrase_marked("A/Loops/One Shots/Bass/B1.wav", "B1.wav", True) is None
    assert C._phrase_marked("A/Chord Progressions/Prog 1.wav", "Prog 1.wav", True) == "folder"
    # a loop folder alone doesn't make a one-shot-classed file a phrase
    assert C._phrase_marked("L/Music Loops/x.wav", "x.wav", False) is None
    # synth-emulation note samples: a leading MIDI note number is not a tempo
    assert C._phrase_marked("F/Vendor A/Synth Kit/WAV/Saw/69_Saw_A4.wav",
                            "69_Saw_A4.wav", True) is None


def test_phrase_roles():
    R = C._phrase_role
    assert R("Loop_124_C_low_bass.wav") == "bass"
    # a 303 / acid line is its own role, ahead of bass
    assert R("Line_303_01.wav") == "acid" and R("Loop_125_F_DeepAcid_01.wav") == "acid"
    assert R("Acid Bass E 130bpm.wav") == "acid" and R("Seq_TB3_02.wav") == "acid"
    assert R("Pad_Placid_120.wav") != "acid"
    assert R("Loop_107_C_Guitar_01.wav") == "live" and R("Cello Phrase Dmin 118bpm 01.wav") == "live"
    assert R("Bassoon Line 90bpm.wav") == "live"                     # a bassoon is no bass
    assert R("Warm Chords 122bpm Em.wav") == "chords" and R("Loop_Rhodes_91_C#_01.wav") == "chords"
    assert R("Synth Lead Melody Fmin 117 bpm.wav") == "lead"
    assert R("Synth Strings 120bpm.wav") == "lead"                   # synth strings aren't played live
    # then the folder, then Ableton, then Sononym
    assert R("Loop_123_Gm_01.wav", "O/Techno/Bass Loops/Loop_123_Gm_01.wav") == "bass"
    assert R("x_120.wav", "P/Music Loops/x_120.wav", ab_tags=["Electric Guitar"]) == "live"
    assert R("Loop_127_Dm_SynthRiff.wav", "P/Synth Loops/x.wav", ab_tags=["Electric Guitar"]) == "lead"
    assert R("x_120.wav", "P/Music Loops/x_120.wav", ab_tags=["Electric Piano"]) == "chords"
    assert R("x_120.wav", "P/Music Loops/x_120.wav", son_labels=["Tone Bass & LowKeys"]) == "bass"
    assert R("x_120.wav", "P/Music Loops/x_120.wav") == "lead"
    assert C._band_of("phrase", "x_120.wav", rel="P/Bass Loops/x_120.wav") == "bass"


def test_phrase_folder_names_lead_with_role_then_tempo():
    c = dict(band="chords", band_type="phrase", bpm=124.0, bpm_range=[120, 132])
    body, lead = N._body_and_lead(c, "PHRASES", CATEGORIES["PHRASES"], ["deep", "house", "chords"], "phrase")
    assert lead == ["chords", "120-135bpm"]
    c = dict(band="bass", band_type="phrase", bpm=172.0, bpm_range=[170, 174])
    assert N._body_and_lead(c, "PHRASES", CATEGORIES["PHRASES"], [], "phrase")[1] == ["bass", "170-175bpm"]
    c = dict(band="bass", band_type="phrase", bpm=172.0, bpm_range=[172, 172])
    assert N._body_and_lead(c, "PHRASES", CATEGORIES["PHRASES"], [], "phrase")[1] == ["bass", "172bpm"]


def test_twin_of_a_phrase_homes_with_it():
    homes = {1: "SUB", 2: "PHRASES", 3: "PADS"}
    assert C._unify_twin_homes(homes, {1: "h", 2: "h", 3: "g"}) == 1
    assert homes == {1: "PHRASES", 2: "PHRASES", 3: "PADS"}


def test_phrases_stay_out_of_one_shots_and_in_phrases(monkeypatch):
    from tests.regressions.test_curate_regressions import _emb, _pool, _row, _run_select
    from fourier.packs.curate_config import PHRASE_PHRASES
    rows = _pool()
    ph = _row(100, "V1", "Loop_124_C_low_bass.wav")
    ph.rel_path = "V1/House/Bass Loops/Loop_124_C_low_bass.wav"; ph.path = f"{LIB}/{ph.rel_path}"
    ph.classes, ph.duration_s = '["Loop"]', 7.742
    kept = _row(101, "V2", "Loop_125_D_keys_riff.wav")
    kept.rel_path = "V2/House/Music Loops/Loop_125_D_keys_riff.wav"; kept.path = f"{LIB}/{kept.rel_path}"
    kept.classes, kept.duration_s = '["Loop"]', 7.68
    rows += [ph, kept]
    # a phrase is guarded out of a one-shot category even when homed there; a Keep wins
    got = _run_select(monkeypatch, rows, "SUB", {r.id: "SUB" for r in rows}, keeps={kept.path: "SUB"},
                      misfiled={}, emb=_emb(len(rows)))
    assert ph.path not in got and kept.path in got and len(got) == len(rows) - 1
    # ...and a phrase rated Misfiled in PHRASES may go back to a one-shot home
    got = _run_select(monkeypatch, rows, "SUB", {r.id: "SUB" for r in rows}, keeps={},
                      misfiled={ph.path: "PHRASES"}, emb=_emb(len(rows)))
    assert ph.path in got
    # PHRASES takes phrases only (CLAP text anchors stubbed: every row reads as musical)
    mus, brk = np.eye(16, dtype="float32")[0], np.eye(16, dtype="float32")[1]
    monkeypatch.setattr(C, "embed_text", lambda p: mus if p in PHRASE_PHRASES else brk)
    emb = np.stack([mus + 0.5 * np.eye(16, dtype="float32")[2 + i % 14] for i in range(len(rows))])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)            # distinct: no near-dup prune
    got = _run_select(monkeypatch, rows, "PHRASES", {r.id: "PHRASES" for r in rows}, keeps={},
                      misfiled={}, emb=emb)
    assert got == {ph.path, kept.path}


def test_verify_phrases(tmp_path):
    from fourier.packs import verify as V
    fn = "Loop_124_C_low_bass.wav"
    files = _clean() + [("PHRASES", "bass-120-125bpm-acid-bassline", fn, _hit(2.0), f"V1/Bass Loops/{fn}"),
                        ("SUB", "dark", "Bass Loop 2.wav", _hit(1.0), "V1/Bass Loops/Bass Loop 2.wav")]
    root, store = _master(tmp_path, files)
    _with_bands(root, "PHRASES", {fn: {"band": "bass", "bpm": 124.0}}, {"bass-120-125bpm-acid-bassline": "bass"})
    fails = _verify(root, store)
    assert not {f for f in fails if "PHRASES" in f or "phrase" in f or "tempo ranges" in f}, fails
    ctx = V._Ctx(root, store_path=store)
    ctx.phrase = {f"{LIB}/V1/Bass Loops/{fn}", f"{LIB}/V1/Bass Loops/Bass Loop 2.wav"}
    lv = {r.check: r.level for r in V.check_routing(ctx)}
    assert lv["no musical phrases outside PHRASES (one-shot folders hold one-shots)"] == V.FAIL
    assert lv["PHRASES holds only musical phrases"] == V.PASS
    ctx.phrase = set()
    lv = {r.check: r.level for r in V.check_routing(ctx)}
    assert lv["PHRASES holds only musical phrases"] == V.FAIL
    # a folder that doesn't lead with its role, and an off-grid tempo range
    root, store = _master(tmp_path / "b", files)
    _with_bands(root, "PHRASES", {fn: {"band": "bass", "bpm": 124.0}}, {"bass-120-125bpm-acid-bassline": "lead"})
    assert "PHRASES folders hold one phrase band each" in _verify(root, store)


def test_phrase_folders_named_for_their_instrument(tmp_path):
    L = C._phrase_inst_label
    assert L("Loop_130_E_Gritty_Guitar_01.wav") == "guitar" and L("Loop_146_G#m_Git_Organ_01.wav") == "guitar"
    assert L("Cello Phrase Dmin 118bpm 01.wav") == "cello" and L("Loop_127_F_rhodes_01.wav") == "rhodes"
    assert L("Line_303_01.wav") == "acid" and L("Loop_122_Em_BassArp.wav") == "arp"
    assert L("Loop_124_C_low_bass.wav") is None
    assert CATEGORIES["PHRASES"]["name_instruments"] == "phrase"
    c = dict(band="live", band_type="phrase", bpm=100.0, bpm_range=[92, 118], inst="guitar")
    body, lead = N._body_and_lead(c, "PHRASES", CATEGORIES["PHRASES"], ["orchestral", "string", "phrase"],
                                  "phrase")
    assert body == ["guitar"] and lead == ["live", "090-120bpm"]
    files = _clean() + [("PHRASES", "live-090-120bpm-orchestral-string-phrase", f"G{i}_100_Guitar.wav",
                         _hit(1.0), f"V1/Guitar Loops/G{i}_100_Guitar.wav") for i in range(3)]
    root, store = _master(tmp_path, files)
    assert "PHRASES folders named for the instrument most files name" in _verify(root, store)


def test_synth_emulation_note_samples_are_not_tempo_named():
    from fourier.packs.curate_config import BPM_NAME_RE
    assert not BPM_NAME_RE.search("84 AM Pad C6.wav")           # MIDI note, patch, note
    assert not BPM_NAME_RE.search("68 Am Keys G#4_01.wav")
    assert BPM_NAME_RE.search("Loop_120_Amin_Pad.wav") and BPM_NAME_RE.search("Bass 124bpm C4.wav")
    assert CATEGORIES["PHRASES"]["break_min"] == 0.0                # contrastive gate only


# --- names that describe, one note a set, fewer small folders ------------------------------

def test_phrase_support_and_genre_backers():
    T = np.eye(4, dtype="float32")
    V = np.array([[1, .5, .1, 0], [1, .4, .2, 0], [0, .1, 1, .5]], dtype="float32")
    sup = C._phrase_support(V, T, np.array([True, True, True, True]), topk=1)
    assert sup == {0: 2 / 3, 1: 0.0, 2: 1 / 3, 3: 0.0}
    assert C._phrase_support(V, T, np.array([False, True, False, True]), topk=1) == {1: 2 / 3, 3: 1 / 3}
    rx = C._genre_backers("909 house kick")
    assert len(rx) == 2 and all(r.search("Vendor A/909 Kit/House BD.wav") for r in rx)
    assert C._genre_backers("punchy kick") == [] and C._genre_backers("soft round kick") == []
    assert C._genre_backers("drum and bass hat")[0].search("Loops/Drum & Bass/x.wav")


def test_sibling_cap_keeps_low_mid_high_and_keeps():
    rec = [dict(path=f"/lib/SampleLibrary/V/P/Warm Pad {n}.wav", tune=t, pin=False)
           for n, t in (("C2", 36), ("C3", 48), ("C4", 60), ("C5", 72), ("C6", 84))]
    rec.append(dict(path="/lib/SampleLibrary/V/P/Kick.wav", tune=None, pin=False))   # no set: untouched
    assert C._cap_siblings(list(range(6)), rec, cap=3) == [0, 2, 4, 5]
    rec[3]["pin"] = True                                                          # a Keep stays
    assert C._cap_siblings(list(range(6)), rec, cap=3) == [0, 3, 4, 5]


def test_small_categories_get_fewer_folders():
    assert C._folder_kmax(12, 90, 500) == 3          # a small category
    assert C._folder_kmax(12, 225, 500) == 5         # BLIPS
    assert C._folder_kmax(12, 375, 285) == 6         # STABS: its pool, not its budget
    assert C._folder_kmax(12, 900, 5000) == 12       # never above the cap
    assert CATEGORIES["DRUMLOOPS"]["require_tempo"] and CATEGORIES["DRUMLOOPS"]["kmin"] == 12


def test_note_leads_only_in_piano():
    for cat in ("SUB", "SYNTH", "PADS", "STABS", "VOX"):
        body, lead = N._body_and_lead(dict(tune=38), cat, CATEGORIES[cat], ["saw", "bass"], "bass")
        assert lead == [], cat
    assert N._body_and_lead(dict(tune=60), "PIANO", dict(kind="oneshot"), ["piano"], "piano")[1] == ["C4"]


def test_verify_checks_backed_names_note_leads_and_sets(tmp_path):
    files = _clean() + [
        ("KICKS", "909-house-dark", "BD 1.wav", _hit(), "V1/Pack/BD 1.wav"),        # no 909 / house
        ("SUB", "d2-saw-short", "Saw Bass D2.wav", _hit(), "V1/P/Saw Bass D2.wav"),  # note lead
    ] + [("SYNTH", "bell-tone", f"Warm Pad C{o}.wav", _hit(), f"V1/P/Warm Pad C{o}.wav")
         for o in range(2, 7)]                                                           # 5 notes of a set
    fails = _fails(tmp_path, files)[1]
    assert "genre and drum-machine words in names are backed by the files" in fails
    assert "note leads only in PIANO" in fails
    assert "at most 3 notes of one multisample set a folder" in fails
    ok = _clean() + [("KICKS", "909-dark", "BD 1.wav", _hit(), "V1/909 Kit/BD 1.wav")]
    fails = _fails(tmp_path / "ok", ok)[1]
    assert not {f for f in fails if "genre" in f or "note leads" in f or "multisample" in f}, fails
    # a weakly supported naming phrase, a freetempo DRUMLOOPS folder, a long loop in PADS
    files = _clean() + [("DRUMLOOPS", "freetempo-bright", "L.wav", _hit(2.0), "V1/P/L.wav"),
                        ("PADS", "warm", "Music Loop 3.wav", _hit(6.0), "V1/Music Loops/Music Loop 3.wav")]
    root, store = _master(tmp_path / "w", files)
    fp = Path(root) / "KICKS" / "_manifest.json"
    fams = json.loads(fp.read_text()); fams[0].update(clap="punchy kick", clap_support=0.1); fp.write_text(json.dumps(fams))
    fails = _verify(root, store)
    assert "naming phrases fit at least 40% of their folder's files" in fails
    assert "loops without a whole-bar tempo are left out where required" in fails
    assert "no long loop-folder files in the tonal one-shot folders" in fails


def test_chord_named_one_shots_are_stabs():
    for fn in ("C Tone (Chord).wav", "28 Saw Chords E1.wav", "Chord Hit Gm 01.wav",
               "Chord Soft Dbmaj7.wav", "Pad Cm7.wav", "Synth Hit Fsus4.wav"):
        assert C._is_named_stab(fn, 1.2, 1), fn
    for fn in ("Harpsichord_01.wav",                  # a harpsichord, not a chord
               "Bass Fm.wav", "Lead Am 02.wav",       # a bare key labels one note
               "Chord Progression 120bpm.wav"):       # a phrase
        assert not C._is_named_stab(fn, 1.2, 1), fn
    assert not C._is_named_stab("Soul Chord.wav", 6.0, 1)       # too long for a stab
    assert not C._is_named_stab("Soul Chord.wav", 1.0, 3)       # several hits
    from fourier.packs.curate_config import STAB_CHORD_FROM
    assert STAB_CHORD_FROM == {"SYNTH", "SUB", "FX", "BLIPS", "PADS", "PERC"}


def test_stabs_holds_named_stabs_only(monkeypatch, tmp_path):
    for fn in ("Synth Stab 03.wav", "Brass Stab 01.wav", "30_OrchestraHit_F#1.wav",
               "Hoover Lead C2.wav", "C Tone (Chord).wav", "Gtr Stb 01.wav", "Lead_G_Synth_Hit_01.wav"):
        assert C._stab_named(fn), fn
    for fn in ("Zaps_07_03.wav", "Muted-Hit-01.wav", "Perc Stab 2.wav", "SoftKick_07.wav",
               "Sax_Growl_01.wav", "Beep Tone 01.wav", "Kick Hit.wav"):
        assert not C._stab_named(fn), fn
    # a bass or vocal stab isn't moved out of SUB / VOX; a chord is, whatever else it says
    assert not C._is_named_stab("Bass_Stab_Short.wav", 0.5, 1)
    assert not C._is_named_stab("Vox Stab 03.wav", 0.5, 1)
    assert C._is_named_stab("Bass Chords 01.wav", 0.5, 1)
    assert C._is_named_stab("Synth Stab 3.wav", 0.5, 1) and not C._is_named_stab("Synth Stab 3.wav", 6.0, 1)
    assert CATEGORIES["STABS"]["stab_names"]
    # selection drops an unnamed file Sononym calls a stab; a Keep stays
    from tests.regressions.test_curate_regressions import _emb, _pool, _row, _run_select
    rows = _pool()
    for r in rows:
        r.filename = r.filename.replace("Texture", "Synth Stab")
        r.path = r.path.replace("Texture", "Synth Stab"); r.rel_path = r.rel_path.replace("Texture", "Synth Stab")
    junk, kept = _row(100, "V1", "Muted-Hit-01.wav"), _row(101, "V2", "Beep Tone 01.wav")
    rows += [junk, kept]
    got = _run_select(monkeypatch, rows, "STABS", {r.id: "STABS" for r in rows}, keeps={kept.path: "STABS"},
                      misfiled={}, emb=_emb(len(rows)))
    assert junk.path not in got and kept.path in got and len(got) == len(rows) - 1
    # verify flags an unnamed file in STABS
    files = _clean() + [("STABS", "dark", "Muted-Hit-01.wav", _hit(), "V1/P/Muted-Hit-01.wav"),
                        ("STABS", "dark", "Synth Stab 3.wav", _hit(), "V1/P/Synth Stab 3.wav")]
    fails = _fails(tmp_path, files)[1]
    assert "STABS holds only files named as chords or stabs" in fails


# --- named scratches, drum names, quiet files, lead floors, tiny folders ---

def test_scratch_and_drum_names():
    for fn in ("Vinyl-Scratch11.wav", "Chirp Scratch 03.wav", "Transformer 01.wav", "Record Stop.wav", "Scr_04.wav"):
        assert C._scratch_named(fn), fn
    assert C._scratch_named("x 01.wav", "Pack/DJ Scratches/x 01.wav")          # by its folder
    for fn in ("piano scratch.wav", "Perc Scratch Long.wav", "Laser TOM Low.wav", "Hihat Closed Wide.wav",
               "Duck Call 01.wav"):
        assert not C._scratch_named(fn), fn
    D = C._drum_named
    assert D("BD Round Low Thud.wav") == "KICKS" and D("SD Tone 01.wav") == "SNARES" and D("Clap 07.wav") == "CLAPS"
    assert D("HH_SDX_Tight.wav") == "HATS" and D("Synth Tom Low.wav") == "TOMS"
    assert D("Ride 06.wav") == "CYMBALS" and D("Cowbell 808.wav") == "PERC" and D("Zaps_05_08.wav") == "BLIPS"
    assert D("Metal TOM 01.wav") is None                   # a drum machine's name, not a tom
    assert D("Kick Snare Layer.wav") is None               # two kinds decide nothing
    assert D("Layered Hat Crash.wav") == "HATS"  # a hat named with a cymbal word is a hat
    assert D("Warm Pad C3.wav") is None


def test_machine_labels_and_patch_sets():
    assert C._machine_label("/x/SampleLibrary/Vendor A/909 Kit/BD 909 01.wav") == "909"
    assert C._machine_label("/x/SampleLibrary/Vendor B/Kit Pack 1/808/BD 1.wav") == "808"
    assert C._machine_label("/x/SampleLibrary/V/P/Kick 1.wav") is None
    k = C._sibling_key
    assert k("/p/marimba 47.wav") == k("/p/marimba 52.wav") is not None      # a patch and a note number
    assert k("/p/SoftPiano_12.wav") == k("/p/SoftPiano_19.wav") is not None
    assert k("/p/Kick 01.wav") is None and k("/p/Pad 3.wav") is None and k("/p/Dim_17.wav") is None


def test_merge_small_groups_within_band():
    P = np.array([[1, 0], [1, 0.1], [0.9, 0.2], [0, 1], [0.1, 1], [1, 0.05]], dtype="float32")
    groups = [[0, 1], [2], [3, 4], [5]]
    out = C._merge_small_groups(groups, ["a", "a", "b", "a"], P, 3)
    assert sorted(map(sorted, out)) == [[0, 1, 2, 5], [3, 4]]                  # "b" has no mate: kept
    assert C._merge_small_groups([[0], [1]], ["a", "a"], P, 3, keep_per_band=2) == [[0], [1]]


def test_lead_floor_trim_and_quiet_files():
    sr = 44100
    rng = np.random.default_rng(0)
    room = 0.005 * rng.standard_normal(sr)                                       # 1 s of room, ~-41 dB
    tone = 0.8 * np.sin(2 * np.pi * 3000 * np.arange(sr // 2) / sr)
    y = np.concatenate([room, tone, 0.005 * rng.standard_normal(sr // 4)])
    plain = C._trim_edges(y, sr, True, False)                                    # -50 dB floor: room stays
    floored = C._trim_edges(y, sr, True, False, lead_floor_db=-30.0)             # a category's trim_lead_db
    assert len(plain) > len(y) - 100 and abs(len(y) - len(floored) - sr) < sr * 0.02
    assert C._rms_db(np.zeros(100)) == float("-inf") and abs(C._rms_db(np.full(100, 0.1)) + 20) < 1e-6
    assert CATEGORIES["KICKS"]["trim_lead_db"] == -40.0 and "SCRATCHES" not in CATEGORIES


def test_verify_named_scratches_drums_and_quiet(tmp_path):
    files = _clean() + [
        ("PERC", "dark", "Vinyl-Scratch04.wav", _hit(), "V1/P/Vinyl-Scratch04.wav"),        # a scratch: FX's
        ("TOMS", "dark", "BD Round Low Thud.wav", _hit(), "V1/P/BD Round Low Thud.wav"),
        ("PERC", "soft", "Silent.wav", np.zeros(4410, dtype="float32"), "V1/P/Silent.wav"),
    ]
    fails = _fails(tmp_path, files)[1]
    assert "files named as a drum (or a scratch) live in that category" in fails
    assert "no near-silent one-shots (RMS under -38 dBFS)" in fails
    ok = _clean() + [("FX", "scratch-dark", "Vinyl-Scratch01.wav", _hit(), "V1/P/Vinyl-Scratch01.wav"),
                     ("KICKS", "dark", "BD Round Low Thud.wav", _hit(), "V1/P/BD Round Low Thud.wav")]
    fails = _fails(tmp_path / "ok", ok)[1]
    assert not {f for f in fails if "scratch" in f or "named as a drum" in f or "near-silent" in f}, fails



def test_scratches_folded_into_fx():
    # read here, not at import: a fixture may reload curate_config
    from fourier.packs.curate_config import BUDGETS, CATEGORIES, CATEGORY_ORDER, FX_BAND_SHARE
    assert "SCRATCHES" not in CATEGORIES and "SCRATCHES" not in CATEGORY_ORDER and "SCRATCHES" not in BUDGETS
    assert "scratch" in C.BAND_ORDER["fx"] and abs(sum(FX_BAND_SHARE.values()) - 1) < 1e-9
    assert CATEGORIES["FX"]["band_share"] is FX_BAND_SHARE and BUDGETS["FX"] == 660
    assert C._fx_band(["XFX Sweeps & Lasers"], [], "Vinyl-Scratch11.wav") == "scratch"      # the name wins
    assert C._fx_band(["XFX Sweeps & Lasers"], [], "Laser Beam 01.wav") == "sweep"
    assert C._band_of("fx", "Chirp Scratch 03.wav") == "scratch"
    assert C._band_phrase_ok("fx", "scratch", "baby scratch") and not C._band_phrase_ok("fx", "sweep", "baby scratch")
    assert not C._scratch_named("Shaker Scratch Soft.wav") and C._drum_named("Shaker Scratch Soft.wav") == "HATS"
    assert not C._scratch_named("scratchy_lead_Dm_120.wav")


# --- drum names and wave folders -------------------------------------------------------------

def test_drum_names_machine_bass_camelcase_cymb_and_rs_rimshot():
    assert C._drum_named("Bass SDS 800 Low.wav") == "KICKS"       # a kit that calls its kicks "Bass"
    assert C._drum_named("Tom SDS 800 Low.wav") == "TOMS"         # "SDS" isn't a snare
    assert C._drum_named("SoftKick_04.wav") == "KICKS"
    assert C._drum_named("Plate Cymb3.wav") == "CYMBALS" and C._drum_named("softCymb_p_rr1.wav") == "CYMBALS"
    assert C._drum_named("RS 09.wav") == "SNARES"              # rims live in SNARES
    assert C._drum_named("sd_01.wav") == "SNARES" and C._drum_named("Kick Snare.wav") is None
    assert C._drum_named("Bass Guitar C2.wav") is None and C._drum_named("Burst.wav") is None


def test_drumkit_in_a_wavetable_pack_is_not_waves():
    assert C._is_wave("Vendor A/Wavetable Pack/Synth A/Wavetables/Saw/01.wav", 0.05)
    assert not C._is_wave("Vendor A/Wavetable Pack/Synth A/Drumkit/"
                          "Kit 01/Cymb 07.wav", 0.05)


def test_verify_winds_outside_acoustic(tmp_path):
    files = _clean() + [("PERC", "dark", "Sax_Growl_02.wav", _hit(), "V1/P/Sax_Growl_02.wav"),
                        ("VOX", "dark", "Viola_D4_v1_rr1.wav", _hit(1.0),
                         "Orchestra Pack/Strings/Viola/Viola_D4_v1_rr1.wav")]
    fails = _fails(tmp_path, files)[1]
    assert "named wind / brass one-shots and orchestra folders only in ACOUSTIC" in fails


def test_hot_one_shots_turned_down():
    from fourier.packs.curate_config import ONESHOT_RMS_CEIL_DB
    sq = np.sign(np.sin(2 * np.pi * 200 * np.arange(4410) / 44100)) * 0.89
    y = C._clamp_rms(sq, ONESHOT_RMS_CEIL_DB["HATS"])
    assert abs(C._rms_db(y) - ONESHOT_RMS_CEIL_DB["HATS"]) < 0.01
    quiet = sq * 0.01
    assert np.array_equal(C._clamp_rms(quiet, -12.0), quiet)                        # never turned up


def test_verify_hot_one_shot(tmp_path):
    sq = (np.sign(np.sin(2 * np.pi * 200 * np.arange(8820) / 44100)) * 0.89).astype("float32")
    fails = _fails(tmp_path, _clean() + [("HATS", "closed-dark", "hot.wav", sq, "V1/P/hot.wav")])[1]
    assert "no one-shot hotter than its category's RMS ceiling" in fails


def test_sibling_cap_prefers_loud_layers():
    assert C._velocity_rank("Strings_A4_v1_rr2.wav") == 1 and C._velocity_rank("crash_hit_ff_short.wav") == 7
    assert C._velocity_rank("Kick 01.wav") is None
    rec = [dict(path=f"/lib/SampleLibrary/Orchestra Pack/Strings/Strings_{n}_{v}_rr1.wav", tune=t, pin=False)
           for n, t in (("C3", 48), ("C4", 60), ("C5", 72)) for v in ("v1", "v3")]
    got = C._cap_siblings(list(range(6)), rec, cap=3)
    assert [C._velocity_rank(rec[i]["path"]) for i in got] == [3, 3, 3]              # one loud layer a note
    # no velocity words: the register spread as before
    rec2 = [dict(path=f"/lib/SampleLibrary/V/P/Warm Pad {n}.wav", tune=t, pin=False)
            for n, t in (("C2", 36), ("C3", 48), ("C4", 60), ("C5", 72), ("C6", 84))]
    assert C._cap_siblings(list(range(5)), rec2, cap=3) == [0, 2, 4]


def test_names_collide_on_word_set_and_near_tempo():
    assert N.name_collides("pad-bright-gritty", ["pad-gritty-bright"])
    assert N.name_collides("127bpm-loop-bright", ["126bpm-loop-bright"])
    assert not N.name_collides("130bpm-loop-bright", ["126bpm-loop-bright"])
    assert not N.name_collides("stab-sustained-dark", ["stab-sustained"])
    assert N.unique_name("pad-bright-gritty", {"pad", "bright", "gritty"}, ["dark"], {"pad-gritty-bright"}) \
        == "pad-bright-gritty-dark"


def test_synth_folders_not_named_stab():
    from fourier.packs.curate_config import NAME_EXCLUDE_WORDS, NAME_SUPPORT_FULL
    assert NAME_EXCLUDE_WORDS["SYNTH"] == {"stab", "stabs"} and "FX" in NAME_SUPPORT_FULL


def test_merge_small_groups_by_planned_files():
    P = np.array([[1, 0], [1, .1], [0, 1], [.1, 1], [.2, 1]], dtype="float32")
    groups = [[0, 1], [2, 3, 4]]
    # by count nothing is under 2; by planned files the first gets 1 and merges
    assert C._merge_small_groups(groups, [None, None], P, 2) == groups
    got = C._merge_small_groups(groups, [None, None], P, 2, measure=lambda gs, bs: [1, 3][:len(gs)] if len(gs) == 2 else [5])
    assert got == [[0, 1, 2, 3, 4]]


def test_verify_duplicate_and_stab_names(tmp_path):
    files = _clean() + [("PADS", "gritty-bright", "p2.wav", _hit(2.0), "V1/P/p2.wav"),
                        ("PADS", "bright-gritty", "p3.wav", _hit(2.0), "V2/P/p3.wav"),
                        ("SYNTH", "organ-stab-sustained", "s1.wav", _hit(1.0), "V1/P/s1.wav")]
    fails = _fails(tmp_path, files)[1]
    assert any(f.startswith("folder names unique per category") for f in fails)
    assert "no stab names outside STABS (SYNTH)" in fails


def test_fx_folder_without_phrase_is_band_and_trait():
    body, lead = N._body_and_lead({"band_type": "fx", "band": "foley"}, "FX", CATEGORIES["FX"], [], "fx")
    assert body == [] and lead == ["foley"]
    body, lead = N._body_and_lead({"band_type": "hat", "band": "closed"}, "HATS", CATEGORIES["HATS"], [], "hat")
    assert body == ["hat"]


# --- tempo, machine labels, tails and routing -----------------------------------------------

def test_tempo_snaps_to_the_length():
    dur = 32 * 60 / 140.0                                   # 32 beats at exactly 140
    assert C._snap_tempo(dur, 140.5) == 140.0
    assert C._snap_tempo(dur, 150.0) == 150.0               # >3% off: not snapped
    assert C._resolve_tempo("Breaks 140.bpm.wav", dur, 140.5, None)[0] == 140.0


def test_machine_label_reads_the_filename_first():
    # a file named for one machine, in a folder named for another, is the file's machine
    assert C._machine_label("/l/SampleLibrary/Vendor A/808 Kit/Kick 909 A.wav") == "909"
    assert C._machine_label("/l/SampleLibrary/Vendor A/Drum Kit/Kick 01.wav") is None


def test_fx_transitions_by_name():
    assert C._is_fx_transition("Uplifter 01.wav") and C._is_fx_transition("Sweep_Down_01.wav")
    assert C._is_fx_transition("Riser_03.wav") and C._is_fx_transition("FXUP_Phase_01.wav")
    assert not C._is_fx_transition("Sweep Pad C3.wav") and not C._is_fx_transition("Sunrise Chord.wav")


def test_end_fade_only_when_the_end_is_loud():
    sr = 44100
    t = np.arange(sr) / sr
    sus = 0.8 * np.sin(2 * np.pi * 55 * t)                  # a sub cut off at full level
    y = C._end_fade(sus, sr)
    assert abs(y[-1]) < 1e-3 and np.allclose(y[:-500], sus[:-500])
    dec = 0.8 * np.exp(-20 * t) * np.sin(2 * np.pi * 55 * t)  # already decayed to silence
    assert np.array_equal(C._end_fade(dec, sr), dec)


def test_edition_twins_share_a_content_key(monkeypatch):
    from fourier.packs import curate_config as cc
    monkeypatch.setattr(cc, "UMBRELLA_VENDORS", ("Vendor A",))      # one folder per pack below it
    a = C._content_key(C._pack_of("Vendor A/Synth Pack A - 16bit/Synths/x.wav", "/x"), "Clav C3.wav", 1.0)
    b = C._content_key(C._pack_of("Vendor A/Synth Pack A/Synths/x.wav", "/x"), "Clav C3.wav", 1.0)
    assert a != C._content_key(C._pack_of("Vendor A/Synth Pack B/Synths/x.wav", "/x"), "Clav C3.wav", 1.0)
    assert a == b
    assert C._content_key("808", "x.wav", 1.0) != C._content_key("909", "x.wav", 1.0)


def test_cross_category_dedupe(tmp_path):
    import json as _j
    src_ch = "/l/SampleLibrary/Vendor A/Synth Pack A/C Pad (Chord) 01.wav"
    src_pp = "/l/SampleLibrary/Vendor B/Keys Pack/Samples/Piano Knock 01.wav"
    assert C._dup_winner(src_ch, ["PIANO", "STABS"]) == "STABS"
    assert C._dup_winner(src_pp, ["PERC", "PIANO"]) == "PIANO"
    assert C._dup_winner("/l/SampleLibrary/X/Chorus_02.wav", ["PIANO", "SUB"]) == "SUB"
    for cat, fam in (("PIANO", "chord-x"), ("STABS", "stab-x")):
        d = tmp_path / cat / fam
        d.mkdir(parents=True)
        (d / "a.wav").write_bytes(b"x")
        (tmp_path / cat / "_manifest.json").write_text(_j.dumps([{"family": fam, "copied": 1, "n": 1}]))
    res = [("PIANO", {"files": 1}, [{"family": "chord-x", "out": "chord-x/a.wav", "src": src_ch}]),
           ("STABS", {"files": 1}, [{"family": "stab-x", "out": "stab-x/a.wav", "src": src_ch}])]
    assert C._dedupe_across(tmp_path, res) == 1
    assert res[0][2] == [] and res[0][1]["files"] == 0 and len(res[1][2]) == 1
    assert not (tmp_path / "PIANO" / "chord-x" / "a.wav").exists()
    assert _j.loads((tmp_path / "PIANO" / "_manifest.json").read_text())[0]["copied"] == 0


def test_verify_checks_end_fades_transitions_keys_and_one_home(tmp_path):
    sr = 44100
    t = np.arange(sr // 2) / sr
    cut = (0.8 * np.sin(2 * np.pi * 55 * t)).astype("float32")     # ends at full level
    files = _clean() + [
        ("SUB", "dark", "cut.wav", cut, "V1/P/cut.wav"),
        ("PADS", "dark", "Uplifter 01.wav", _hit(2.0), "V1/P/Uplifter 01.wav"),
        ("KICKS", "dark", "piano thud.wav", _hit(), "V1/P/piano thud.wav"),
        ("SNARES", "dark", "k1.wav", _hit(), "V1/Pack/k1.wav"),          # same source as a KICKS file
    ]
    fails = _fails(tmp_path, files)[1]
    for chk in ("one-shots end quietly (faded, no click)", "named risers / downlifters / sweeps only in FX",
                "no keys-named files in the drum categories", "every source file in one category only"):
        assert chk in fails, (chk, fails)


# --- names -----------------------------------------------------------------------------------

def test_instrument_words_need_backing():
    assert [r.pattern for r in C._inst_backers("djembe hit")] == ["djembe"]
    assert C._inst_backers("punchy kick") == []
    rx = C._inst_backers("conga groove")[0]
    assert rx.search("Vendor A/Perc Pack/Tumba 01.wav") and not rx.search("Vendor A/Perc Pack/Bongo 01.wav")
    assert C._inst_backers("clavinet note")[0].search("Vendor A/Keys Pack/Clavinet/Clav C3.wav")


def test_trait_gates():
    assert not N.trait_ok("soft", {"med": {"atk": 15.0, "dur": 0.4}})
    assert N.trait_ok("soft", {"med": {"atk": 40.0, "dur": 0.4}})
    assert not N.trait_ok("short", {"med": {"atk": 1400.0, "dur": 6.4}})
    assert not N.trait_ok("long-tail", {"med": {"atk": 10.0, "dur": 0.39}})
    assert N.trait_ok("dark", {"med": {"atk": 10.0, "dur": 0.39}}) and N.trait_ok("soft", {})


def test_pairwise_tiebreak_points_the_right_way():
    dims = [("noi", ("clean", "gritty")), ("br", ("dark", "bright"))]
    a = {"z": {"noi": 0.8, "br": 0.1}}
    b = {"z": {"noi": 0.2, "br": 0.3}}
    assert N.pairwise_traits(a, b, dims)[0] == "gritty"       # a is the grittier one
    assert N.pairwise_traits(b, a, dims)[0] == "clean"


def test_risers_and_downlifters_by_name():
    assert C._fx_band(["XFX Sweeps & Lasers"], [], "Uplifter 01.wav") == "riser"
    assert C._fx_band([], ["Sweep"], "Sweep_Down_01.wav") == "downlifter"
    assert C._fx_band(["XFX Sweeps & Lasers"], [], "Rise and Fall 01.wav") == "sweep"   # says both
    from fourier.packs.curate_config import FX_BAND_SHARE
    assert abs(sum(FX_BAND_SHARE.values()) - 1.0) < 1e-9 and FX_BAND_SHARE["riser"] == 0.15


# --- pipeline safety ---------------------------------------------------------------------------

def test_names_cut_in_the_middle_and_boilerplate_stripped(monkeypatch):
    from fourier.devices.exporter import _sanitize_filename, _truncate_middle
    from fourier.packs import curate_config
    from fourier.packs.render import _strip_boilerplate
    a = _sanitize_filename("Acme 909 Kit_Kick Tune Max Decay Min", 32)
    b = _sanitize_filename("Acme 909 Kit_Kick Tune Max Decay Max", 32)
    assert len(a) <= 32 and "~" in a and a != b and a.endswith("Decay_Min")
    assert _strip_boilerplate("Vendor A 909 Sample Pack_Kick A") == "Vendor A 909_Kick A"      # the default
    monkeypatch.setattr(curate_config, "NAME_BOILERPLATE", ["^vendor a[ _-]+", "^vendor b[ _-]+",
                                                            *curate_config.NAME_BOILERPLATE])
    assert _strip_boilerplate("Vendor A Vendor B 909 Sample Pack_Kick A") == "909_Kick A"
    assert _strip_boilerplate("Kick 01") == "Kick 01"
    assert _truncate_middle("abcdefghij", 5) == "ab~ij" and _truncate_middle("abc", 5) == "abc"


def test_swap_in_keeps_previous(tmp_path):
    from fourier.cli import _swap_in
    out, nxt = tmp_path / "M", tmp_path / "M.next"
    out.mkdir(); (out / "old").write_text("o")
    nxt.mkdir(); (nxt / "new").write_text("n")
    prev = _swap_in(str(nxt), str(out))
    assert (out / "new").exists() and (tmp_path / "M.prev" / "old").exists() and not nxt.exists()
    assert prev.endswith("M.prev")


def test_swap_in_touches_only_what_changed(tmp_path):
    import os
    from fourier.cli import _swap_in
    out, nxt = tmp_path / "M", tmp_path / "M.next"
    for d in (out / "KICKS" / "a", out / "KICKS" / "gone", nxt / "KICKS" / "a", nxt / "SNARES" / "b"):
        d.mkdir(parents=True)
    (out / "KICKS" / "a" / "same.wav").write_bytes(b"same")
    (out / "KICKS" / "a" / "changed.wav").write_bytes(b"old")
    (out / "KICKS" / "gone" / "dropped.wav").write_bytes(b"x")
    (nxt / "KICKS" / "a" / "same.wav").write_bytes(b"same")          # equal bytes, other inode
    (nxt / "KICKS" / "a" / "changed.wav").write_bytes(b"new")
    (nxt / "SNARES" / "b" / "added.wav").write_bytes(b"n")
    st = os.stat(out / "KICKS" / "a" / "same.wav")
    _swap_in(str(nxt), str(out))
    st2 = os.stat(out / "KICKS" / "a" / "same.wav")
    assert (st.st_ino, st.st_mtime_ns) == (st2.st_ino, st2.st_mtime_ns)       # untouched
    assert (out / "KICKS" / "a" / "changed.wav").read_bytes() == b"new"
    assert (out / "SNARES" / "b" / "added.wav").exists()
    assert not (out / "KICKS" / "gone").exists() and not nxt.exists()
    prev = tmp_path / "M.prev"
    assert (prev / "KICKS" / "a" / "changed.wav").read_bytes() == b"old"      # the old master kept
    assert (prev / "KICKS" / "gone" / "dropped.wav").exists()
    assert not list(out.rglob("*.fourier-swap"))


def test_build_meta_hashes_code():
    m = C._build_meta()
    assert m.get("code_hash") and len(m["code_hash"]) == 16 and "ratings_hash" in m


def test_verify_dsp_fallback(tmp_path):
    import json as _j
    from pathlib import Path as _P
    root, store = _master(tmp_path, _clean())
    root = _P(root)
    mp = root / "manifest.json"
    doc = _j.loads(mp.read_text())
    doc["categories"]["KICKS"]["entries"][0]["dsp_fallback"] = True
    mp.write_text(_j.dumps(doc))
    from fourier.packs.verify import verify_master, FAIL
    ok, res = verify_master(root, session=None, store_path=store, log=lambda *a: None)
    assert "every file went through export processing (no verbatim fallbacks)" in {r.check for r in res if r.level == FAIL}


def test_verify_canonical_names(tmp_path):
    files = _clean() + [("KICKS", "punchy-sub-heavy", "Kick 01.wav", _hit(), "V1/P/Kick 01.wav"),
                        ("KICKS", "k" * 45, "k3.wav", _hit(), "V1/P/k3.wav")]
    fails = _fails(tmp_path, files)[1]
    assert "file names canonical and within 46 characters (the same on every device)" in fails
    assert "folder names within 44 characters" in fails


# --- cymbal bands, rims, mono, folder splits ---------------------------------------------------

def test_cymbal_bands_and_rims():
    assert C._cymbal_band("Ride 909 Tone Test.wav") == "ride" and C._cymbal_band("Crash_01.wav") == "crash"
    assert C._cymbal_band("Cymbal 02.wav", ["Perc Cymbal Rides"]) == "ride"
    assert C._cymbal_band("Cym Short.wav") == "cymbal"
    assert C._band_phrase_ok("cymbal", "ride", "bright ride") and not C._band_phrase_ok("cymbal", "ride", "crash cymbal")
    assert C._drum_named("Rim 808 01.wav") == "SNARES" and C._drum_named("RS 808 Dry.wav") == "SNARES"


def test_organs_are_piano_but_organ_stabs_stay():
    assert C._is_organ_named("Drawbar_Organ_C3.wav") and C._is_organ_named("Hammond Slow Swell.wav")
    assert not C._is_organ_named("Organ Stab 3.wav") and not C._is_organ_named("Piano B3.wav")


def test_mono_kicks_and_near_mono():
    t = np.arange(4410) / 44100
    l = np.sin(2 * np.pi * 60 * t)
    wide = np.stack([l, np.sin(2 * np.pi * 61 * t)], axis=1)
    assert C._to_mono(wide).ndim == 2                         # real stereo stays
    assert C._to_mono(wide, force=True).ndim == 1             # KICKS / SUB forced
    near = np.stack([l, l * 0.9999], axis=1)
    assert C._to_mono(near).ndim == 1                         # sides under -40 dB: mono
    inv = np.stack([l, -l], axis=1)
    assert np.allclose(C._to_mono(inv, force=True), l)        # cancelling channels: keep one


def test_slow_non_breakbeat_loops_not_folded():
    from fourier.packs.curate_config import NO_FOLD_RE, FOLD_ANYWAY_RE
    assert NO_FOLD_RE.search("Vendor A/Trip Hop Pack/083 Drum Loop.wav")
    assert FOLD_ANYWAY_RE.search("Vendor B/Jungle Pack/Dub Loop 01.wav")
    from fourier.packs.curate_config import CATEGORIES
    assert CATEGORIES["DRUMLOOPS"]["no_fold"] is NO_FOLD_RE and "classic" in CATEGORIES["DRUMLOOPS"]["keep_all_bands"]


def test_classic_breaks_folder_name():
    body, lead = N._body_and_lead({"band_type": "loop", "band": "classic", "bpm": 100.0},
                                  "DRUMLOOPS", CATEGORIES["DRUMLOOPS"], [], "loop")
    assert lead == ["classic"] and body == ["breaks"]


def test_split_capped_groups():
    rng = np.random.default_rng(0)
    P = np.vstack([rng.normal(0, .1, (150, 3)) + [1, 0, 0], rng.normal(0, .1, (150, 3)) + [0, 1, 0],
                   rng.normal(0, .1, (40, 3))])
    rec = [{"band": None} for _ in range(len(P))]
    groups = [list(range(300)), list(range(300, 340))]
    planned = lambda gs, bs: [min(120, len(g)) for g in gs]
    out = C._split_capped_groups(groups, P, planned, 120, 12, rec)
    assert len(out) == 3 and sorted(len(g) for g in out) == [40, 150, 150]
    # a previous build's folder, kept whole, isn't split again
    rec = [{"band": None, "prev_family": "big" if i < 300 else None} for i in range(len(P))]
    out = C._split_capped_groups(groups, P, planned, 120, 12, rec)
    assert sorted(len(g) for g in out) == [40, 300]


# --- retune to C -------------------------------------------------------------------------------

def test_name_root_and_retune_shift():
    assert C._name_root_pc("trombone_C#3_v2.wav") == 1 and C._name_root_pc("Cs4.wav") == 1
    assert C._name_root_pc("Chord Soft Dbmaj7.wav") == 1 and C._name_root_pc("Em.wav") == 4
    assert C._name_root_pc("C Pad (Chord) 01.wav") is None and C._name_root_pc("Kit B Snare.wav") is None
    assert C._retune_shift("SUB", "Bass G1.wav", None) == (5, "name")          # G up 5 to C
    assert C._retune_shift("SUB", "Bass A1.wav", None) == (3, "name")
    assert C._retune_shift("SUB", "Bass E1.wav", None) == (-4, "name")
    assert C._retune_shift("SUB", "Bass.wav", 43.0) == (None, None)           # SUB: detection not trusted
    assert C._retune_shift("SYNTH", "Lead.wav", 43.2) == (4.8, "detect")      # G +0.2 -> up 4.8
    assert C._retune_shift("KICKS", "BD C1.wav", 24.0) == (None, None)


def test_repitch_changes_pitch_and_length():
    sr = 44100
    t = np.arange(sr) / sr
    y = np.sin(2 * np.pi * 440 * t)
    up = C._repitch(y, sr, 12)
    assert abs(len(up) - sr / 2) < 5                                          # an octave up: half as long
    f = np.argmax(np.abs(np.fft.rfft(up))) * sr / len(up)
    assert abs(f - 880) < 5


def test_sibling_cap_prefers_c_notes():
    rec = [dict(path=f"/lib/SampleLibrary/V/P/Strings Pad {n}.wav", tune=t, pin=False, name_pc=C._name_root_pc(n + ".wav"))
           for n, t in (("C2", 36), ("E2", 40), ("G2", 43), ("C3", 48), ("E3", 52), ("C4", 60))]
    got = C._cap_siblings(list(range(6)), rec, cap=3)
    assert [rec[i]["path"].split()[-1] for i in got] == ["C2.wav", "C3.wav", "C4.wav"]


def test_sticky_names(monkeypatch):
    doc = {"categories": {"KICKS": {"entries": [
        {"family": "tight-punchy-gritty", "src": f"/l/SampleLibrary/V/k{i}.wav"} for i in range(10)] + [
        {"family": "soft-round", "src": f"/l/SampleLibrary/V/s{i}.wav"} for i in range(10)]}}}
    monkeypatch.setitem(C._PREV_FAMS, "doc", doc)
    rec = [dict(path=f"/l/SampleLibrary/V/k{i}.wav", ab=[]) for i in range(8)] + \
          [dict(path=f"/l/SampleLibrary/V/s{i}.wav", ab=[]) for i in range(8)]
    clusters = [dict(name="bright-gritty", idxs=list(range(8)), med={"atk": 10.0, "dur": 0.3}),
                dict(name="dark-round", idxs=list(range(8, 16)), med={"atk": 10.0, "dur": 0.3})]
    n = C._apply_sticky_names(clusters, "KICKS", rec, {"kind": "oneshot"})
    assert n == 1 and clusters[0]["name"] == "tight-punchy-gritty"
    assert clusters[1]["name"] == "dark-round"         # "soft" no longer true (10 ms attack)


def test_sticky_names_drop_the_category_noun(monkeypatch):
    doc = {"categories": {"KICKS": {"entries": [
        {"family": "cardboard-box-kick-gritty", "src": f"/l/SampleLibrary/V/k{i}.wav"} for i in range(10)]}}}
    monkeypatch.setitem(C._PREV_FAMS, "doc", doc)
    rec = [dict(path=f"/l/SampleLibrary/V/k{i}.wav", ab=[]) for i in range(8)]
    clusters = [dict(name="cardboard-box-gritty", idxs=list(range(8)), med={"atk": 10.0, "dur": 0.3})]
    assert C._apply_sticky_names(clusters, "KICKS", rec, {"kind": "oneshot"}) == 0


def test_gather_sets_moves_stragglers_within_band():
    from fourier.packs.curate import _gather_sets
    P = "/lib/SampleLibrary/V/Pack/"
    rec = [{"path": P + f"Glass {n}.wav"} for n in (1, 2, 3, 4)] + \
          [{"path": P + f"Other {k} Hit.wav"} for k in "abcdefgh"]
    clusters = [
        dict(band="x", _pick=[0, 1, 2, 4, 5], idxs=[0, 1, 2, 4, 5]),
        dict(band="x", _pick=[3, 6, 7, 8, 9], idxs=[3, 6, 7, 8, 9]),
        dict(band="y", _pick=[10, 11], idxs=[10, 11]),
    ]
    n = _gather_sets(clusters, rec, ceil=120, min_left=2)
    assert n == 1
    assert sorted(clusters[0]["_pick"]) == [0, 1, 2, 3, 4, 5] and 3 in clusters[0]["idxs"]
    assert 3 not in clusters[1]["_pick"]


def test_gather_sets_respects_cap_and_min_left():
    from fourier.packs.curate import _gather_sets
    P = "/lib/SampleLibrary/V/Pack/"
    rec = [{"path": P + f"Glass {n}.wav"} for n in (1, 2, 3)] + [{"path": P + "Solo Thing.wav"}]
    clusters = [dict(band=None, _pick=[0, 1], idxs=[0, 1]), dict(band=None, _pick=[2, 3], idxs=[2, 3])]
    assert _gather_sets(clusters, rec, ceil=2, min_left=1) == 0        # home is full
    assert _gather_sets(clusters, rec, ceil=120, min_left=2) == 0      # source would fall under 2
    assert _gather_sets(clusters, rec, ceil=120, min_left=1) == 1


def test_gather_sets_respects_set_cap():
    from fourier.packs.curate import _gather_sets
    P = "/lib/SampleLibrary/V/Pack/"
    rec = [{"path": P + f"Glass {n}.wav"} for n in range(1, 6)] + \
          [{"path": P + f"Filler {k} Tone.wav"} for k in "abcdef"]
    clusters = [dict(band=None, _pick=[0, 1, 2, 5, 6, 7], idxs=[0, 1, 2, 5, 6, 7]),
                dict(band=None, _pick=[3, 4, 8, 9, 10], idxs=[3, 4, 8, 9, 10])]
    assert _gather_sets(clusters, rec, ceil=120, min_left=1, set_cap=3) == 0
    assert _gather_sets(clusters, rec, ceil=120, min_left=1, set_cap=4) == 1


def test_retune_holds_length_cap():
    import numpy as np
    from fourier.packs.curate import _process_audio
    sr = 8000
    y = np.sin(np.arange(sr * 2) * 0.05) * 0.5
    out = _process_audio(y, sr, "oneshot", False, False, "peak", retune=-5, dur_cap=2.0)
    assert len(out) <= 2 * sr and abs(out[-1]) < 1e-3
    free = _process_audio(y, sr, "oneshot", False, False, "peak", retune=-5)
    assert len(free) > 2 * sr


def test_name_lead_includes_tempo():
    from fourier.packs.curate import _name_lead
    assert _name_lead("live-120-135bpm-gritty") == "live-120-135bpm"
    assert _name_lead("live-120-180bpm-gritty") != _name_lead("live-120-135bpm-gritty")
    assert _name_lead("125-130bpm-tops-hi-hat-bright") == "125-130bpm-tops"
    assert _name_lead("172bpm-rolling-dnb-break") == "172bpm"
    assert _name_lead("mallet-c5-xylophone") == "mallet-c5"
    assert _name_lead("mallet-cs5-bright") == "mallet-cs5"
    assert _name_lead("ride-bright-snappy") == "ride"
    assert _name_lead("classic-breaks-gritty") == "classic-breaks"


def test_name_root_reads_glued_and_bare_notes():
    f = C._name_root_pc
    assert f("ToneF#3-01.wav") == 6 and f("warmpadC3-4.wav") == 0    # glued to a patch name
    assert f("strings_01_g#3.wav") == 8                             # lowercase with a sharp
    assert f("Saw_Stab_C#.wav") == 1 and f("Chord_F#_01.wav") == 6      # bare key
    for n in ("ZA7 Pad.wav", "ZD40 Kick.wav", "ZE800 Saw.wav", "B50 Pad.wav"):
        assert f(n) is None                                               # model numbers aren't notes


def test_sticky_names_need_backed_source_words(monkeypatch):
    # "cardboard" is a source word (INSTRUMENT_WORDS): no file says box, so the old name goes
    doc = {"categories": {"KICKS": {"entries": [
        {"family": "cardboard-box-gritty", "src": f"/l/SampleLibrary/V/k{i}.wav"} for i in range(10)]}}}
    monkeypatch.setitem(C._PREV_FAMS, "doc", doc)
    rec = [dict(path=f"/l/SampleLibrary/V/k{i}.wav", ab=[]) for i in range(8)]
    clusters = [dict(name="bright-gritty", idxs=list(range(8)), med={"atk": 10.0, "dur": 0.3})]
    assert C._apply_sticky_names(clusters, "KICKS", rec, {"kind": "oneshot"}) == 0


def test_synth_machine_pattern_reads_303_sources():
    rx = dict(C._SYNTH_MACHINE_RX)["303"]
    assert rx.search("Vendor A/Acid Pack/303/saw-01.wav")
    assert rx.search("Vendor A/Acid Pack/WAV/TB-3_Saw-C0.wav")
    assert not rx.search("Vendor/Pack/Bass 3033 Hz.wav") and not rx.search("Vendor B/Sub Bass 30.wav")


def test_retuned_names_say_c():
    r = C._retuned_stem
    assert r("Strings_Pizz_A2_v2", 3) == "Strings_Pizz_C3_v2"
    assert r("ToneF#3-01", 6) == "ToneC4-01"
    assert r("Saw_Stab_C#", -1) == "Saw_Stab_C"
    assert r("Chord Soft Dbmaj7", -1) == "Chord Soft Cmaj7"
    assert r("No Note Here", 2) == "No Note Here" and r("Pad_A2", None) == "Pad_A2"
    assert C._name_root_midi("Pad_A2.wav") == 45 and C._name_root_midi("Pad.wav") is None


def test_note_codes_are_not_roots():
    # a vendor code after the note isn't the root, though it starts like one ("C1" before a
    # capital); nor is a model code before it (a note letter after a capital)
    assert C._name_root_pc("Pad-D#3-127-C1ZQ.wav") == 3
    assert C._name_root_pc("ZA7 Pad-D#3-127-C1ZQ.wav") == 3
    assert C._name_root_pc("warmpadF#3.wav") == 6 and C._name_root_pc("ZA7 Pad.wav") is None


def test_several_notes_no_retune():
    assert C._name_multi_note("Vox B2-D3-E2 Amin.wav")
    assert C._retune_shift("VOX", "Vox B2-D3-E2 Amin.wav", 52.0) == (None, None)
    assert C._name_root_midi("Vox B2-D3-E2 Amin.wav") is None
    assert not C._name_multi_note("Rhodes C3 v2 C3.wav")


def test_voices_move_at_most_three_semitones():
    assert C._retune_shift("VOX", "Vox F#2 Ah.wav", None) == (None, None)   # +6: a chipmunk
    assert C._retune_shift("VOX", "Vox G#2 Ah.wav", None) == (None, None)   # +4
    assert C._retune_shift("VOX", "Vox D3 Oh.wav", None) == (-2.0, "name")
    assert C._retune_shift("PIANO", "Clavinet F#0.wav", None) == (None, None)
    assert C._retune_shift("PIANO", "51 Clav D#3.wav", 50.894) == (-2.894, "name")   # 3 plus fine tuning
    assert C._retune_shift("ACOUSTIC", "piccolo_A#6_01.wav", None) == (2.0, "name")
    assert C._retune_shift("SYNTH", "warmpadF#3.wav", None) == (6.0, "name")          # a synth may move 6


def _pick_rec(n, inc=()):
    return [dict(clip=0, sup=1, qual=0.0, fav=0, incumbent=(i in inc)) for i in range(n)]


def test_stable_pick_keeps_incumbents_through_a_pool_change():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(200, 8))
    rec = _pick_rec(200)
    first = C._medoids(list(range(199)), 20, rec, P)            # a pool of 199
    rec = _pick_rec(200, inc=set(first))
    plain = C._medoids(list(range(200)), 20, rec, P)            # one file more: re-picked
    stable = C._stable_pick(list(range(200)), 20, rec, P)
    assert len(stable) == 20 and len(set(stable)) == 20
    assert set(first) <= set(stable)
    assert len(set(first) & set(plain)) < 20                    # what it guards against


def test_stable_pick_fills_gaps_and_trims_to_want():
    rng = np.random.default_rng(1)
    P = np.vstack([rng.normal(0, .1, (50, 4)), rng.normal(5, .1, (50, 4))])
    rec = _pick_rec(100, inc={0, 1, 2})                         # all incumbents in one blob
    out = C._stable_pick(list(range(100)), 6, rec, P)
    assert {0, 1, 2} <= set(out) and any(i >= 50 for i in out)  # the other blob gets a newcomer
    rec = _pick_rec(100, inc=set(range(10)))
    out = C._stable_pick(list(range(100)), 4, rec, P)
    assert len(out) == 4 and set(out) <= set(range(10))         # more incumbents than slots


def test_spread_caps_keep_incumbents():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(60, 6))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rec = [dict(row=i, id=i, qual=0.0, grp="a", incumbent=i in (5, 17, 42)) for i in range(60)]
    out = C._cap_group_spread(rec, emb, lambda d: d["grp"], 8)
    assert len(out) == 8 and {5, 17, 42} <= {d["id"] for d in out}
    # a near-duplicate of an incumbent loses to it even with better quality
    emb2 = np.vstack([emb[:1], emb[:1]])
    rec2 = [dict(row=0, id=0, qual=0.0, incumbent=True), dict(row=1, id=1, qual=1.0)]
    kept, n = C._prune_near_dups(rec2, emb2, 0.985)
    assert [d["id"] for d in kept] == [0] and n == 1


def test_allocations_follow_incumbents_within_a_band():
    rec = [dict(clip=0, incumbent=i < 30) for i in range(100)]
    clusters = [dict(idxs=list(range(0, 40)), band="a"), dict(idxs=list(range(40, 100)), band="a"),
                dict(idxs=[], band="b")]
    out = C._allocs_for_incumbents([20, 30, 5], clusters, rec, [40, 60, 0], 5, 120)
    assert out == [30, 20, 5] and sum(out) == 55              # band totals kept


def test_cluster_starts_from_previous_folders():
    rng = np.random.default_rng(3)
    P = np.vstack([rng.normal(0, .3, (40, 3)) + 6 * np.eye(3)[c] for c in range(3)])
    prev = [list(range(0, 40)), list(range(40, 80)), list(range(80, 120))]
    init = np.array([P[m].mean(0) for m in prev])
    groups, k = C._cluster(P, kmax=5, kmin=2, kdiv=1, init=init)
    assert k == 3 and sorted(sorted(g) for g in groups) == prev
    _, k = C._cluster(P, kmax=2, kmin=2, kdiv=1, init=init)          # seeds that don't fit: ignored
    assert k == 2


def test_cluster_keeps_previous_members_together():
    rng = np.random.default_rng(4)
    P = np.vstack([rng.normal(0, .3, (30, 3)) + 6 * np.eye(3)[c] for c in range(3)])
    fixed = np.full(90, -1)
    fixed[:30], fixed[30:60], fixed[60:] = 0, 1, 2
    fixed[5] = 2                                   # a file the previous build put elsewhere
    init = np.array([P[fixed == k].mean(0) for k in range(3)])
    groups, k = C._cluster(P, kmax=5, kmin=2, kdiv=1, init=(init, fixed))
    assert k == 3 and any(5 in g and 70 in g for g in groups)



def test_swap_in_same_name_other_sample(tmp_path):
    """A name that now belongs to a different sample (same size, other bytes) is replaced,
    and a name changing only in case isn't deleted as dropped on a case-folding disk."""
    from fourier.cli import _swap_in
    out, nxt = tmp_path / "M", tmp_path / "M.next"
    (out / "K").mkdir(parents=True); (nxt / "K").mkdir(parents=True)
    (out / "K" / "Kick.wav").write_bytes(b"AAAA")
    (out / "K" / "Snare.wav").write_bytes(b"s1")
    (nxt / "K" / "Kick.wav").write_bytes(b"BBBB")            # same size, a different sample
    (nxt / "K" / "snare.wav").write_bytes(b"s2")             # only the case differs
    _swap_in(str(nxt), str(out))
    assert (out / "K" / "Kick.wav").read_bytes() == b"BBBB"
    names = sorted(p.name for p in (out / "K").iterdir())
    assert names == ["Kick.wav", "snare.wav"] and (out / "K" / "snare.wav").read_bytes() == b"s2"
    assert (tmp_path / "M.prev" / "K" / "Kick.wav").read_bytes() == b"AAAA"


def test_file_names_stay_with_their_samples():
    """Two different samples share a stem: the one the previous build named "Kick.wav"
    keeps it and the newcomer gets "Kick_2.wav", whichever comes first; names compare
    case-insensitively; a previous name that isn't the file's own is ignored."""
    taken = set()
    old = C._choose_outname("Kick", taken, prev_name="Kick.wav")
    taken.add(old.lower())
    new = C._choose_outname("Kick", taken)
    assert (old, new) == ("Kick.wav", "Kick_2.wav")
    # the previous "_2" owner keeps "_2" even when the plain name is free
    assert C._choose_outname("Kick", set(), prev_name="Kick_2.wav") == "Kick_2.wav"
    assert C._choose_outname("KICK", {"kick.wav"}) == "KICK_2.wav"
    assert C._choose_outname("Kick", set(), prev_name="Snare.wav") == "Kick.wav"
    assert C._choose_outname("Kick", {"kick.wav"}, prev_name="Kick.wav") == "Kick_2.wav"   # taken: no reuse


def test_previous_names_claim_first(monkeypatch, tmp_path):
    doc = {"categories": {"KICKS": {"entries": [{"src": "/a/Kick.wav", "out": "f/Kick.wav", "family": "f"}]}}}
    monkeypatch.setitem(C._PREV_FAMS, "doc", doc)
    assert C._previous_outnames("KICKS") == {"/a/Kick.wav": "Kick.wav"}
    assert C._previous_outnames("SNARES") == {}


def test_breaks_at_jungle_tempo_back_jungle_only():
    """A 155-180 BPM break backs "jungle" in a folder name (a jungle break, whatever its
    path says); tops loops and slower breaks don't, and the tempo backs no other genre word."""
    from fourier.packs.curate import _GENRE_RX, genre_evidence
    j, dnb, ragga = _GENRE_RX["jungle"][1], _GENRE_RX["dnb"][1], _GENRE_RX["ragga"][1]
    ev = genre_evidence("DRUMLOOPS", "Pack/Breaks/Break_01.wav", 168.0, "full")
    assert j.search(ev) and not dnb.search(ev) and not ragga.search(ev)
    assert not j.search(genre_evidence("DRUMLOOPS", "Pack/Tops/Hat_01.wav", 168.0, "tops"))
    assert not j.search(genre_evidence("DRUMLOOPS", "Pack/Breaks/Break_01.wav", 140.0, "full"))
    assert not j.search(genre_evidence("PHRASES", "Pack/Bass/B_170.wav", 170.0, "bass"))
    from fourier.packs.curate_config import BREAK_PHRASES, NAME_EXCLUDE_WORDS
    assert "fast breakbeat" not in BREAK_PHRASES and "fast" in NAME_EXCLUDE_WORDS["DRUMLOOPS"]


def test_twin_names():
    from fourier.packs.naming import twin_names
    assert twin_names("bright-sustained", "bright-swelling-sustained")
    assert twin_names("gritty-bright", "bright-gritty")
    assert not twin_names("bright-steady-sustained", "bright-swelling-sustained")
    assert not twin_names("120-125bpm-gritty", "120-125bpm-tops-gritty")      # tempo-led

