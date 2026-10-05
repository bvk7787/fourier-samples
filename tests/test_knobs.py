"""Tier 1 knobs (fourier/knobs.py): plain settings that stand for tunables, in any layer."""
import pytest

from fourier import knobs, layers
from fourier.settings import ConfigError


@pytest.fixture(scope="module")
def known():
    return layers.defaults()


def test_the_defaults_are_the_knobs_standard_values(known):
    doc = {"tempo": "70-180", "loudness": "standard", "retune": "detect", "stereo": "fold-near-mono",
           "categories": {"KICKS": "on"}, "sources": {"vendor_max": 0.4}}
    assert knobs.apply(doc, known) == []


def test_tempo(known):
    assert knobs.tempo_bands(70, 180) == known["curate_config.TEMPO_BANDS"]   # balanced's
    got = dict((k, v) for _, k, v in knobs.apply({"tempo": "70-170"}, known))
    assert got["curate_config.TEMPO_BANDS"][:4] == (60, 70, 75, 80)
    assert got["curate_config.TEMPO_BANDS"][-3:] == (170, 180, 201)
    for bad in ("fast", "180-85", "85-195", "50-120"):
        with pytest.raises(ConfigError, match="tempo"):
            knobs.apply({"tempo": bad}, known)


def test_categories(known):
    got = dict((k, v) for _, k, v in knobs.apply({"categories": {"waves": "off", "VOX": 0.5}}, known))
    assert got["curate_config.CATEGORIES_OFF"] == {"WAVES"}
    assert got["curate_config.BUDGETS"]["VOX"] == round(known["curate_config.BUDGETS"]["VOX"] / 2)
    assert got["curate_config.BUDGETS"]["KICKS"] == known["curate_config.BUDGETS"]["KICKS"]
    with pytest.raises(ConfigError, match="not a category"):
        knobs.apply({"categories": {"COWBELLS": "off"}}, known)
    with pytest.raises(ConfigError, match="off, on or a weight"):
        knobs.apply({"categories": {"VOX": "loud"}}, known)


@pytest.mark.parametrize("doc, want", [
    ({"loudness": "hot"}, {"curate_config.LOOP_LIMIT_DB": 6.0}),
    ({"loudness": "gentle"}, {"curate_config.LOOP_LIMIT_DB": 0.0}),
    ({"retune": "off"}, {"curate_config.RETUNE_CATS": set(), "curate_config.RETUNE_DETECT_CATS": set()}),
    ({"retune": "named"}, {"curate_config.RETUNE_DETECT_CATS": set()}),
    ({"stereo": "keep"}, {"curate_config.MONO_CATS": set(), "curate_config.NEAR_MONO_SIDE_DB": -200.0}),
])
def test_choices(known, doc, want):
    assert dict((k, v) for _, k, v in knobs.apply(doc, known)) == want


def test_bad_choice(known):
    with pytest.raises(ConfigError, match="loudness = 'loud'"):
        knobs.apply({"loudness": "loud"}, known)


def test_sources(known):
    got = dict((k, v) for _, k, v in knobs.apply(
        {"sources": {"favor": ["classic breaks", "my kits/"], "home": {"one-shot drums": "kicks"},
                     "vendor_max": 0.3}}, known))
    assert got["curate_config.FAVORED_SOURCES"].search("Packs/Classic Breaks/amen.wav")
    assert got["curate_config.PACK_HOME"] == {"one-shot drums": "KICKS"}
    assert got["curate_config.VENDOR_MAX_SHARE"] == 0.3
    with pytest.raises(ConfigError, match="unknown keys"):
        knobs.apply({"sources": {"avoid": ["x"]}}, known)
    with pytest.raises(ConfigError, match="sources.favor"):
        knobs.apply({"sources": {"favor": ["(unclosed"]}}, known)


def test_knobs_in_the_layers(tmp_path, monkeypatch, known):
    """In each file its knobs apply before its [advanced]; a later file beats an earlier one."""
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "none.toml")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "style.yaml").write_text("tempo: 70-170\nloudness: hot\nstereo: keep\n")
    (tmp_path / "fourier.toml").write_text(
        'preset = "style.yaml"\nloudness = "gentle"\n[advanced]\nMONO_CATS = ["KICKS"]\n')
    r = layers.resolve(known=known)
    assert r.values["curate_config.LOOP_LIMIT_DB"] == 0.0                     # fourier.toml's knob
    assert r.sources["curate_config.LOOP_LIMIT_DB"] == "config:fourier.toml (loudness)"
    assert r.values["curate_config.TEMPO_BANDS"][1] == 70                      # the preset's
    assert r.sources["curate_config.TEMPO_BANDS"] == "preset:style (tempo)"
    assert r.values["curate_config.MONO_CATS"] == {"KICKS"}                    # [advanced] wins
    assert r.values["curate_config.NEAR_MONO_SIDE_DB"] == -200.0


def test_a_weight_scales_the_budget_an_earlier_layer_set(tmp_path, monkeypatch, known):
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "none.toml")
    monkeypatch.chdir(tmp_path)
    budgets = dict(known["curate_config.BUDGETS"], VOX=100)
    (tmp_path / "style.yaml").write_text("advanced:\n  BUDGETS: " + repr(budgets).replace("'", '"') + "\n")
    (tmp_path / "fourier.toml").write_text('preset = "style.yaml"\ncategories = { VOX = 0.5 }\n')
    r = layers.resolve(known=known)
    assert r.values["curate_config.BUDGETS"]["VOX"] == 50


def test_on_brings_back_a_category_an_earlier_layer_switched_off(known):
    got = dict((k, v) for _, k, v in knobs.apply(
        {"categories": {"WAVES": "on"}}, {**known, "curate_config.CATEGORIES_OFF": {"WAVES", "VOX"}}))
    assert got["curate_config.CATEGORIES_OFF"] == {"VOX"}


def test_files_and_size(known):
    got = dict((k, v) for _, k, v in knobs.apply({"files": 6000}, known))
    b, b0 = got["curate_config.BUDGETS"], known["curate_config.BUDGETS"]
    assert sum(b.values()) == 6000 and abs(b["DRUMLOOPS"] / b["KICKS"] - b0["DRUMLOOPS"] / b0["KICKS"]) < 0.02
    seven = dict((k, v) for _, k, v in knobs.apply({"size": "7GB"}, known))["curate_config.BUDGETS"]
    assert 10500 < sum(seven.values()) < 12500                  # the preset's budgets
    two = dict((k, v) for _, k, v in knobs.apply({"size": "2000MB"}, known))["curate_config.BUDGETS"]
    assert sum(two.values()) < sum(seven.values()) / 3
    assert knobs.apply({"size": "fixed"}, known) == [("size", "curate_config.POOL_SURPLUS", False)]
    for bad in ({"size": "big"}, {"size": "1MB"}, {"files": 5}, {"files": 5000, "size": "5GB"}):
        with pytest.raises(ConfigError):
            knobs.apply(bad, known)


def test_size_after_the_category_weights_in_one_file(known):
    """The knobs in one file apply in order, so size scales the weighted budgets, and a
    category switched off takes no share."""
    got = dict((k, v) for _, k, v in knobs.apply({"categories": {"WAVES": "off", "VOX": 2}, "files": 5000}, known))
    b = got["curate_config.BUDGETS"]
    live = {c: n for c, n in b.items() if c != "WAVES"}
    assert sum(live.values()) == 5000 and b["WAVES"] == known["curate_config.BUDGETS"]["WAVES"]
    assert b["VOX"] / b["PADS"] > 1.1


def test_size_auto_fits_the_devices_and_moves_the_pools_surplus(known, monkeypatch):
    monkeypatch.setattr(knobs, "storage_limit_mb", lambda known: None)
    assert knobs.apply({"size": "auto"}, known) == []          # the surplus round is the default
    monkeypatch.setattr(knobs, "storage_limit_mb", lambda known: (3000.0, "tiny"))
    got = dict((k, v) for _, k, v in knobs.apply({"size": "auto"}, known))
    b, b0 = got["curate_config.BUDGETS"], known["curate_config.BUDGETS"]
    assert got.get("curate_config.POOL_SURPLUS", known["curate_config.POOL_SURPLUS"]) is True
    # whole files: rounding the budgets can overshoot the cap by about a file's size
    slack = max(known["curate_config.AVG_FILE_MB"].values()) * (1 + known["curate_config.SETS_SHARE"])
    assert knobs.master_mb(b, known) <= 3000 + slack and 3000 < knobs.master_mb(b0, known)
    assert all(b[c] <= b0[c] for c in b0)                       # scaled down, never up
    monkeypatch.setattr(knobs, "storage_limit_mb", lambda known: (10 ** 6, "huge"))
    assert "curate_config.BUDGETS" not in dict((k, v) for _, k, v in knobs.apply({"size": "auto"}, known))
    after_auto = {**known, "curate_config.POOL_SURPLUS": True}
    assert knobs.apply({"size": "fixed"}, after_auto) == [("size", "curate_config.POOL_SURPLUS", False)]


def test_storage_limit_reads_the_configured_devices(known, tmp_path, monkeypatch):
    from fourier import places
    cfg = tmp_path / "fourier.toml"
    cfg.write_text('devices = ["m8_tracker", "digitakt_2", "generic_48k"]\n')
    monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    places.reset()
    assert knobs.storage_limit_mb(known) == (20000 * 0.5, "digitakt_2")
    cfg.write_text('devices = ["generic_48k"]\n')
    assert knobs.storage_limit_mb(known) is None


def test_surplus_budgets_shares_what_the_short_categories_left():
    from fourier.packs.curate import surplus_budgets
    s = lambda n: {"files": n}
    budgets = {"A": 100, "B": 50, "C": 30, "D": 20}
    res = [("A", s(100), []), ("B", s(50), []), ("C", s(20), []), ("D", s(15), []), ("E", s(9), [])]
    got = surplus_budgets(res, budgets)
    assert got == {"A": 110, "B": 55}                    # 15 files short, shared 2:1
    assert surplus_budgets([("A", s(100), []), ("B", s(50), [])], budgets) == {}
    assert surplus_budgets([("C", s(1), []), ("D", s(2), [])], budgets) == {}
    odd = surplus_budgets([("A", s(100), []), ("B", s(50), []), ("C", s(29), [])], budgets)
    assert odd == {"A": 101}                             # 1 file short: the largest remainder


def test_dry_run_plan(monkeypatch):
    from fourier import knobs as K
    from fourier.packs import dryrun
    from fourier.packs.curate_config import BUDGETS
    monkeypatch.setattr(K, "storage_limit_mb", lambda known: (500.0, "tiny"))
    p = dryrun.plan(["KICKS", "PADS"], {"KICKS": 5})
    assert [r["category"] for r in p["rows"]] == ["KICKS", "PADS"]
    assert p["files"] == BUDGETS["KICKS"] + BUDGETS["PADS"] and p["total_mb"] > p["mb"]
    assert p["rows"][0]["homes"] == 5 and p["rows"][1]["homes"] is None
    assert p["limit"] == (500.0, "tiny") and p["surplus"] is True
    # what a library fills at most: a category with fewer home samples than its minimum (6)
    # fills none, as a build leaves it empty (scale = "off"; tests/test_scale.py has the
    # scaled plan)
    from fourier.packs import curate_config as cc
    monkeypatch.setattr(cc, "LIBRARY_SCALE", False)
    p = dryrun.plan(["KICKS", "SNARES", "FX"], {"KICKS": 5, "SNARES": 8, "FX": 6}, samples=19)
    assert {r["category"]: r["picks"] for r in p["rows"]} == {"KICKS": 0, "SNARES": 8, "FX": 6}
    assert p["picks"] == 14


def test_names_knob_and_kept_names(known):
    from fourier.packs import curate as C
    assert knobs.apply({"names": "canonical"}, known) == []
    assert knobs.apply({"names": "keep"}, known) == [("names", "curate_config.NAMES", "keep")]
    with pytest.raises(ConfigError):
        knobs.apply({"names": "fancy"}, known)
    assert C._kept_stem("Break Loop Warm", 46, bpm=172.4) == \
        "Break_Loop_Warm_172bpm"
    assert C._kept_stem("Amen 170 BPM", 46, bpm=170) == "Amen_170_BPM"          # already says it
    assert len(C._kept_stem("x" * 80, 46)) == 46


def test_devices_size_the_names_for_the_tightest_path(known, tmp_path, monkeypatch):
    from fourier.devices import loader as L
    assert knobs.apply({"devices": ["m8_tracker", "digitakt_2"]}, known) == []     # the defaults: 46 / 44
    assert knobs.apply({"devices": ["generic_48k"]}, known) == []                  # no path limit
    d = tmp_path / "devices"
    d.mkdir()
    for f in L.DEFAULT_DEVICES_DIR.glob("*.yaml"):
        (d / f.name).write_text(f.read_text())
    (d / "tight.yaml").write_text(
        "id: tight\nstatus: unverified\npaths:\n  card_dir: {value: /SAMPLES, status: unverified}\n"
        "  max_path_length: {value: 100, status: unverified}\n")
    monkeypatch.setattr(L, "DEFAULT_DEVICES_DIR", d)
    got = dict((k, v) for _, k, v in knobs.apply({"devices": ["m8_tracker", "tight"]}, known))
    room = 100 - len("/SAMPLES/") - len("08_DRUMLOOPS/") - 1 - 4 - 2
    assert got["curate_config.STEM_MAX"] + got["curate_config.FAMILY_NAME_MAX"] == room
    with pytest.raises(ConfigError):
        knobs.apply({"devices": ["walkman"]}, known)
