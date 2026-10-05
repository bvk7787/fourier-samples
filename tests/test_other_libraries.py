"""Libraries that aren't laid out vendor/pack/... with English names: the path words and
their tokenizer, the `words` knob, the CLAP fallback, vendor detection (`vendors`), tempo
folding from the tempo range (`tempo`, `fold`), the genre presets, tempos from folders, names
and WAV chunks, the `sets` knob and the device-driven caps. Each is gated so a library with
Sononym and Live, on breaks-acid with first-folder vendors, builds as before (the synthetic
golden proves it end to end)."""
from collections import namedtuple

import numpy as np
import pytest
import soundfile as sf

from fourier import knobs, layers
from fourier.metadata import shadow as SH
from fourier.packs import vendors as V
from fourier.settings import ConfigError


@pytest.fixture(scope="module")
def known():
    return layers.defaults()


# --- the path words -------------------------------------------------------------------------
@pytest.mark.parametrize("rel, labels", [
    ("Drums/kick01.wav", {"kick"}),                  # letters split from digits
    ("Drums/HiHat58.wav", {"hat"}),                  # ...and camelCase
    ("x/BD01.wav", {"kick"}),
    ("x/CH_01.wav", {"hat"}),                         # closed / open hat abbreviations
    ("x/OH 47.wav", {"hat"}),
    ("x/CP_43.wav", {"clap"}),
    ("x/kck_3.wav", {"kick"}),
    ("x/808Kick.wav", {"kick"}),
    ("Bass/808s/808 01.wav", {"bass"}),              # a folder of 808s; "808s" stays one word
    ("Machines/808/CB.wav", set()),                  # a bare 808 is a machine
    ("TOM37.wav", set()),                            # an all-caps TOM, digits or not
    ("x/Tom37.wav", {"tom"}),
    ("Perc/Perc 01.wav", {"perc.hand"}),
    ("Perc/Kick Perc.wav", {"kick"}),                # a weak word yields
    ("FX/Thing 01.wav", {"fx.noise"}),
    ("Chords/Chord Am.wav", {"stab"}),
    ("Strings/Strings C3.wav", {"pad"}),
    ("Brass/Hit 01.wav", {"class.oneshot", "lead"}),
    ("x/Sub 39.wav", {"bass"}),
    ("Fills/Fill 01.wav", {"class.loop"}),
    ("Loops/Groove96.wav", {"class.loop"}),
    ("Drum Loops/Beat_120BPM.wav", {"class.loop"}),
])
def test_path_words_and_tokenizer(rel, labels):
    assert SH.path_labels(rel) == labels


def test_words_knob(known, monkeypatch):
    got = dict((k, v) for _, k, v in knobs.apply({"words": {"kicks": ["Bombo", "kck"], "PADS": "nappe"}}, known))
    assert got == {"curate_config.PATH_WORDS": {"KICKS": ["bombo", "kck"], "PADS": ["nappe"]}}
    assert knobs.apply({"words": {}}, known) == []
    for bad in ({"DRUMLOOPS": ["groove"]}, {"NOPE": ["x"]}, {"KICKS": []}, ["bombo"]):
        with pytest.raises(ConfigError, match="words"):
            knobs.apply({"words": bad}, known)
    from fourier.packs import curate_config as cc
    before = SH.rules_version(SH.PATH)
    assert SH.path_labels("Muestras/Bombo 01.wav") == set()
    monkeypatch.setattr(cc, "PATH_WORDS", {"KICKS": ["bombo"], "PADS": ["nappe"]})
    assert SH.path_labels("Muestras/Bombo 01.wav") == {"kick"}
    assert SH.path_labels("Nappes/Nappe Douce.wav") == {"pad"}       # plural and the folder too
    assert SH.rules_version(SH.PATH) != before                         # the labels rebuild


# --- the CLAP fallback ------------------------------------------------------------------------
def test_clap_fallback_homes_only_a_clear_match(monkeypatch):
    from fourier.packs import curate as C
    rng = np.random.default_rng(0)
    cats = [c for c, cfg in C.CATEGORIES.items() if cfg["kind"] in ("oneshot", "loop")]
    anchors = {c: (lambda v: v / np.linalg.norm(v))(rng.standard_normal(16)) for c in cats}
    monkeypatch.setattr(C, "_clap_anti", lambda cat: None)
    kick = anchors["KICKS"]
    assert C._clap_home(kick, anchors, oneshot=True, loop=False) == "KICKS"
    assert C._clap_home(kick, anchors, oneshot=True, loop=False, misfiled={"KICKS"}) != "KICKS"
    halfway = (anchors["KICKS"] + anchors["SNARES"]) / np.linalg.norm(anchors["KICKS"] + anchors["SNARES"])
    assert C._clap_home(halfway, anchors, oneshot=True, loop=False) is None     # no clear winner
    assert C._clap_home(anchors["STABS"], anchors, oneshot=True, loop=False) != "STABS"   # named only
    assert C._clap_home(np.zeros(16), anchors, oneshot=True, loop=False) is None   # like nothing
    assert C._clap_home(anchors["DRUMLOOPS"], anchors, oneshot=False, loop=True) == "DRUMLOOPS"
    assert C._clap_home(kick, anchors, oneshot=False, loop=False) is None       # no shape, no home


# --- vendors -------------------------------------------------------------------------------
def test_layouts_are_detected():
    assert V.detect(["Acme/Pack A/Kick 01.wav", "Northwind/Loops/Groove.wav", "Vendor C/x/y.wav"])["layout"] == V.PACKS
    types = ["Drums/Kicks/Kick 01.wav", "Drums/Snares/SD 01.wav", "Loops/174/Groove 01.wav",
             "Bass/808s/808 01.wav", "One Shots/Hits/Hit.wav", "Pads/Pad 01.wav"]
    assert V.detect(types)["layout"] == V.TYPES
    umb = [f"Downloads/Pack {i}/Kicks/Kick {j}.wav" for i in range(5) for j in range(3)]
    got = V.detect(umb)
    assert got["layout"] == V.UMBRELLA and got["umbrellas"] == ["Downloads"]
    assert V.detect([f"{i:03d}.wav" for i in range(20)] + ["Kicks/Kick 01.wav"])["layout"] == V.FLAT


def test_vendor_of(monkeypatch):
    from fourier.packs import curate_config as cc
    assert V.vendor_of("Acme/Pack/Kick.wav") == "Acme" and V.vendor_of("") == "?"      # first-folder
    assert V.pack_key("Acme/Pack/Kits/Kick.wav") == "Acme/Pack"
    monkeypatch.setattr(cc, "VENDORS", "auto")
    for lay, rel, vendor, pack in ((V.TYPES, "Drums/Kicks/Kick.wav", None, None),
                                   (V.FLAT, "001.wav", None, None),
                                   (V.UMBRELLA, "Downloads/Pack A/Kicks/Kick.wav", "Downloads/Pack A",
                                    "Downloads/Pack A/Kicks"),
                                   (V.PACKS, "Acme/Pack/Kits/Kick.wav", "Acme", "Acme/Pack")):
        monkeypatch.setattr(V, "_LAYOUTS", {"": {"layout": lay, "umbrellas": ["Downloads"]}})
        assert (V.vendor_of(rel, rel), V.pack_key(rel)) == (vendor, pack), lay


def test_samples_without_a_vendor_are_never_capped():
    from fourier.packs import curate as C
    rec = [dict(id=i, vendor=None, qual=0.5, row=i) for i in range(10)]
    rec += [dict(id=10 + i, vendor=v, qual=0.5, row=10 + i) for i, v in enumerate("AAAAAABC")]
    out = C._cap_vendor_share(rec, None, 0.4)
    assert [r["id"] for r in out if r["vendor"] is None] == list(range(10))
    assert sum(1 for r in out if r["vendor"] == "A") < 6                       # the rest are capped


def test_vendors_knob(known):
    assert knobs.apply({"vendors": "first-folder"}, known) == []
    assert knobs.apply({"vendors": "auto"}, known) == [("vendors", "curate_config.VENDORS", "auto")]
    with pytest.raises(ConfigError, match="vendors"):
        knobs.apply({"vendors": "second-folder"}, known)


# --- tempo folding --------------------------------------------------------------------------
def test_the_tempo_range_folds_breaks_acid_as_before(known):
    assert knobs.apply({"tempo": "70-180", "fold": "auto"}, known) == []          # balanced's: nothing set
    got = dict((k, v) for _, k, v in knobs.apply({"tempo": "85-180"}, known))
    assert set(got) == {"curate_config.TEMPO_BANDS"}                               # breaks-acid: its bands only
    assert knobs.fold_window(85, 180) == ((90.0, 180.0), None)
    assert knobs.fold_window(70, 180) == ((90.0, 180.0), None)                    # balanced's too
    got = dict((k, v) for _, k, v in knobs.apply({"tempo": "118-140"}, known))
    assert got["curate_config.TEMPO_FOLD"] == (70.0, 140.0)
    assert got["curate_config.TEMPO_FOLD_RANGE"] == (118.0, 140.0)
    assert got["curate_config.TEMPO_BANDS"][:4] == (60, 75, 118, 120)              # 5-BPM inner edges
    assert dict((k, v) for _, k, v in knobs.apply({"fold": "off"}, known)) == {"curate_config.TEMPO_FOLDING": False}


@pytest.mark.parametrize("rng, bpm, filed", [
    (None, 86, 172), (None, 200, 100),                       # no range: as always
    ((118.0, 140.0), 174, 174), ((118.0, 140.0), 64, 128),   # house: a break isn't halved
    ((70.0, 100.0), 170, 85), ((70.0, 100.0), 120, 120),     # hip hop: never out of the range
    ((130.0, 160.0), 70, 140),                               # trap: half time folds up
])
def test_folds_stay_in_the_range(monkeypatch, rng, bpm, filed):
    from fourier.packs import curate as C
    monkeypatch.setattr(C, "TEMPO_FOLD_RANGE", rng)
    hi = rng[1] if rng else 180.0
    assert C._fold_in_range(bpm, C._fold_tempo(bpm, (hi / 2, hi))) == filed


# --- presets --------------------------------------------------------------------------------
@pytest.fixture
def resolve(tmp_path, monkeypatch):
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "none.toml")

    def run(text):
        p = tmp_path / "fourier.toml"
        p.write_text(text)
        return layers.resolve(config=str(p))
    return run


@pytest.mark.parametrize("name", ["balanced", "hiphop-lofi", "house-techno", "ambient-cinematic", "trap"])
def test_genre_presets(resolve, name):
    from fourier.cli.setup import preset_line, presets
    v = resolve(f'preset = "{name}"\n').values
    assert name in presets() and preset_line(name)
    ex = v.get("curate_config.DRUMLOOP_STYLE_EXCLUDE", layers.defaults()["curate_config.DRUMLOOP_STYLE_EXCLUDE"])
    if name in ("balanced", "trap", "hiphop-lofi"):            # no breaks-only exclusions
        assert not ex.search("Trap Drums/Loops/Trap Loop 01.wav") and ex.search("FX Loops/Riser.wav")
    if name == "hiphop-lofi":
        assert v["curate_config.DRUMLOOP_DUR_MAX"] >= 8 * 4 * 60 / 85              # 8 bars at 85 BPM
        assert v["curate_config.TEMPO_FOLD_RANGE"] == (70.0, 100.0)
    if name == "ambient-cinematic":
        assert v["curate_config.DUR_CAP"]["PADS"] > layers.defaults()["curate_config.DUR_CAP"]["PADS"]


def test_balanced_resolves_to_nothing_and_breaks_acid_to_its_own(resolve):
    from fourier.cli.setup import preset_line
    assert resolve('preset = "balanced"\n').values == {}                           # the code's defaults
    ba = resolve('preset = "breaks-acid"\n').values
    assert set(ba) == {f"curate_config.{n}" for n in
                       ("BUDGETS", "TEMPO_BANDS", "POOL_SURPLUS", "DRUMLOOP_STYLE_EXCLUDE")}
    assert ba["curate_config.BUDGETS"]["DRUMLOOPS"] == 1900 and ba["curate_config.POOL_SURPLUS"] is False
    assert preset_line("breaks-acid").startswith("breaks, jungle")


# --- tempo from names, folders and WAV chunks ---------------------------------------------------
def test_wav_chunks(tmp_path):
    from fourier.ingest.chunks import acid_chunk, add_chunks, read_wav_chunks, smpl_chunk
    p = tmp_path / "Beat.wav"
    sf.write(p, np.zeros(4410, dtype="float32"), 44100, subtype="PCM_16")
    assert read_wav_chunks(p) == {}
    add_chunks(p, acid_chunk(128.0, 16), smpl_chunk(57))
    assert read_wav_chunks(p) == {"acid_bpm": 128.0, "acid_beats": 16, "root_note": 57}
    q = tmp_path / "Hit.wav"
    sf.write(q, np.zeros(4410, dtype="float32"), 44100, subtype="PCM_16")
    add_chunks(q, acid_chunk(120.0, 4, root=48, one_shot=True))
    assert read_wav_chunks(q) == {"root_note": 48}                                # a one-shot: no tempo
    assert sf.info(str(q)).frames == 4410                                        # still a valid WAV
    (tmp_path / "x.wav").write_bytes(b"RIFF\x04\x00\x00\x00WAVEjunk")
    assert read_wav_chunks(tmp_path / "x.wav") == {}


Row = namedtuple("Row", "filename rel_path path duration_s acid_bpm acid_beats")


@pytest.mark.parametrize("row, want", [
    (Row("Beat bpm120.wav", "A/Beat bpm120.wav", "/l/A/Beat bpm120.wav", 3.3, None, None), (120.0, "name")),
    (Row("Groove96.wav", "A/Groove96.wav", "/l/A/Groove96.wav", 10.0, None, None), (96.0, "name")),
    (Row("T120.wav", "A/T120.wav", "/l/A/T120.wav", 7.0, None, None), (None, None)),   # not whole bars
    (Row("Groove 01.wav", "Loops/174/Groove 01.wav", "/l/Loops/174/Groove 01.wav", 4 * 4 * 60 / 174, None, None),
     (174.0, "folder")),
    (Row("Beat A.wav", "Loops/Beat A.wav", "/l/Loops/Beat A.wav", 7.5, 128.0, 16), (128.0, "acid")),
    (Row("Beat B.wav", "Loops/Beat B.wav", "/l/Loops/Beat B.wav", 8.0, None, 16), (120.0, "acid")),
])
def test_tempo_sources_without_sononym(row, want):
    from fourier.packs import curate as C
    assert C._resolve_tempo(row.filename, row.duration_s, None, None, C._fallback_tempos(row)) == want
    assert C._resolve_tempo(row.filename, row.duration_s, None, None) == (None, None)   # with Sononym: as before


def test_the_name_still_wins():
    from fourier.packs import curate as C
    row = Row("Beat 125bpm.wav", "Loops/174/Beat 125bpm.wav", "/l/x", 7.68, 128.0, 16)
    assert C._resolve_tempo(row.filename, row.duration_s, None, None, C._fallback_tempos(row)) == (125.0, "name")


# --- sets and the devices' caps ------------------------------------------------------------------
def test_sets_knob(known):
    assert knobs.apply({"sets": "on"}, known) == []
    assert knobs.apply({"sets": "off"}, known) == [("sets", "sets.SETS_ON", False)]


def test_device_caps_only_from_a_profile_that_sets_them(known, tmp_path, monkeypatch):
    assert knobs.apply({"devices": ["m8_tracker", "digitakt_2"]}, known) == []      # as before
    d = tmp_path / "devices"
    d.mkdir()
    (d / "tiny_sampler.yaml").write_text(
        "id: tiny_sampler\nname: Tiny Sampler\nstatus: unverified\nload: copy\n"
        "audio:\n  sample_rate: {value: 44100, status: unverified}\n  bit_depth: {value: 16, status: unverified}\n"
        "  max_slices: {value: 32, status: unverified}\n"
        "paths:\n  max_name_length: {value: 24, status: unverified}\n")
    monkeypatch.setenv("FOURIER_DEVICES", str(d))
    got = dict((k, v) for _, k, v in knobs.apply({"devices": ["digitakt_2", "tiny_sampler"]}, known))
    assert got == {"sets.SLICE_MAX": 32, "curate_config.STEM_MAX": 24 - knobs.NAME_ROOM}


# --- reasons ----------------------------------------------------------------------------------
def test_why_names_off_style_and_quiet_files():
    from fourier.packs.why_log import describe
    doc = {"category": "DRUMLOOPS"}
    assert "off-style loop" in describe("style_excluded", "trap", doc) and "trap" in describe("style_excluded", "trap", doc)
    assert describe("too_quiet", -38.0, doc).startswith("too quiet")
    assert describe("soft_layers", -30.0, {"category": "KICKS"}).startswith("too quiet")


def test_eight_bit_clipping_takes_a_run():
    from fourier.packs.verify import _clipped
    y = np.zeros((100, 1))
    y[[10, 40, 70]] = 1.0 - 2.0 / 256          # three peaks at the top code: not a clip at 8 bits
    assert not _clipped(y, 8) and _clipped(np.where(y > 0, 1.0, 0.0), 16)   # ...at 16, as before
    y[50:53] = 1.0 - 2.0 / 256
    assert _clipped(y, 8)


def test_the_survey_follows_links_out_of_the_library(tmp_path):
    from fourier.cli.setup import survey
    lib, other = tmp_path / "lib", tmp_path / "drive" / "Pack B"
    (lib / "Pack A").mkdir(parents=True)
    other.mkdir(parents=True)
    for p in (lib / "Pack A" / "Kick 01.wav", lib / "Pack A" / "Snare 01.flac", other / "Hat 01.wav"):
        sf.write(p, np.zeros(441, dtype="float32"), 44100)
    (lib / "Pack A" / "notes.txt").write_text("x")
    (lib / "Pack B").symlink_to(other)
    (lib / "Again").symlink_to(lib / "Pack A")              # inside the library: walked at its target
    rels: list = []
    assert survey([lib], 100, rels=rels) == (3, 0)
    assert sorted(rels) == ["Pack A/Kick 01.wav", "Pack A/Snare 01.flac", "Pack B/Hat 01.wav"]
