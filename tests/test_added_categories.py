"""Categories a library overlay adds (add_categories; fourier/taxonomy.py with_added,
fourier/layers.py, curate_config.ADDED_CATEGORIES): a whole category from an overlay, taxonomy
entry, engine settings, budget and routing values, numbered after the code's categories."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fourier import layers, taxonomy
from fourier.settings import ConfigError, encode, write_resolved

ROOT = Path(__file__).resolve().parents[1]

FIELD = {
    "after": "BLIPS", "kind": "gated", "labels": ["fx.nature"], "noun": "field", "roles": ["no_sets"],
    "family": "fx", "prompts": ["rain on a roof", "street noise", "room tone"],
    "anti": ["synth pad", "drum hit"],
    "engine": {"dims": [["br", ["dark", "bright"]]], "noise": "(?!x)x", "break_min": 0.3,
               "dur_max": 30.0, "no_vendor_cap": True, "trim_lead_db": -30.0, "keep_all_bands": ["x"]},
    "carve_from": {"FX": 0.25}, "budget": 40, "avg_file_mb": 2.0, "rms_ceiling_db": -18.0,
    "folder_words": ["field"], "name_words": {"rain": "rain|drizzle"}, "presets": {"balanced": "off"},
}


def test_with_added_merges_the_category():
    t = taxonomy.with_added({"FIELD": FIELD})
    code = taxonomy.order()
    assert t.order() == [*code, "FIELD"]                      # numbered after the code's categories
    names = t.categories()
    assert names[names.index("BLIPS") + 1] == "FIELD"         # defined after its `after`
    assert t.kind("FIELD") == "gated" and t.labels("FIELD") == ("fx.nature",) and t.nouns()["FIELD"] == "field"
    assert "FIELD" in t.with_role("no_sets") and t.prompts("FIELD")[0] == "rain on a roof"
    engine = {n: {} for n in taxonomy.categories()}
    cats = t.with_taxonomy(engine)
    f = cats["FIELD"]
    assert list(f)[:4] == ["kind", "labels", "phrases", "anti"]
    assert f["dims"] == [("br", ("dark", "bright"))] and f["keep_all_bands"] == ("x",)     # the code's kinds
    assert f["trim_lead_db"] == -30.0 and f["no_vendor_cap"] is True
    assert cats["FX"]["exclude_phrases"] == f["phrases"] and cats["FX"]["exclude_anti"] == f["anti"]
    assert cats["FX"]["exclude_min"] == 0.25
    assert t.add_values({"KICKS": 1}, "budget") == {"KICKS": 1, "FIELD": 40}
    assert t.add_values({"FIELD": 5}, "budget") == {"FIELD": 5}             # an override wins
    assert t.add_values({"KICKS": set()}, "folder_words", set)["FIELD"] == {"field"}
    assert t.add_words({"conga": "conga"}, "name_words") == {"conga": "conga", "rain": "rain|drizzle"}
    none = taxonomy.with_added({})
    assert none.order() == code and none.add_values(d := {"KICKS": 1}, "budget") is d


@pytest.mark.parametrize("entry, msg", [
    ({"KICKS": {"kind": "oneshot", "noun": "kick"}}, "already a category"),
    ({"field": {"kind": "gated", "noun": "field"}}, "a name in capitals"),
    ({"FIELD": {"kind": "gated"}}, "needs a noun"),
    ({"FIELD": {"kind": "thing", "noun": "field"}}, "kind 'thing'"),
    ({"FIELD": {"kind": "gated", "noun": "f", "labels": ["field"]}}, "not in the canonical vocabulary"),
    ({"FIELD": {"kind": "gated", "noun": "f", "after": "NOPE"}}, "not a category before it"),
    ({"FIELD": {"kind": "gated", "noun": "f", "carve_from": {"FX": 0.2}}}, "needs anti prompts"),
    ({"FIELD": {"kind": "gated", "noun": "f", "budget": "lots"}}, "budget"),
    ({"FIELD": {"kind": "gated", "noun": "f", "colour": "red"}}, "unknown keys"),
])
def test_a_bad_entry_says_what(entry, msg):
    with pytest.raises(taxonomy.TaxonomyError, match=msg):
        taxonomy.with_added(entry)


@pytest.fixture
def resolve(tmp_path, monkeypatch):
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "none.toml")
    import yaml
    (tmp_path / "mine.yaml").write_text(yaml.safe_dump({"add_categories": {"FIELD": FIELD}}))

    def run(text):
        p = tmp_path / "fourier.toml"
        p.write_text(text)
        return layers.resolve(config=str(p))
    return run


def test_an_overlay_adds_a_category_the_later_knobs_see(resolve):
    r = resolve('overlay = "mine.yaml"\ncategories = { FIELD = 2 }\n')
    assert r.values["curate_config.ADDED_CATEGORIES"] == {"FIELD": FIELD}
    assert r.sources["curate_config.ADDED_CATEGORIES"] == "overlay:mine (add_categories)"
    assert r.values["curate_config.BUDGETS"]["FIELD"] == 80                 # its budget, weighted
    r = resolve('overlay = "mine.yaml"\nfiles = 5000\ncategories = { FIELD = "off" }\n')
    b = r.values["curate_config.BUDGETS"]
    assert r.values["curate_config.CATEGORIES_OFF"] == {"FIELD"}
    assert sum(v for c, v in b.items() if c != "FIELD") == 5000 and b["FIELD"] == 40
    with pytest.raises(ConfigError, match="not a category"):
        resolve('categories = { FIELD = "off" }\n')                        # not without the overlay


def test_what_the_preset_in_use_says_about_it(resolve):
    r = resolve('preset = "balanced"\noverlay = "mine.yaml"\n')
    assert r.values["curate_config.CATEGORIES_OFF"] == {"FIELD"}
    assert r.sources["curate_config.CATEGORIES_OFF"] == "overlay:mine (add_categories presets)"
    r = resolve('preset = "balanced"\noverlay = "mine.yaml"\ncategories = { FIELD = "on" }\n')
    assert r.values["curate_config.CATEGORIES_OFF"] == set()
    assert "curate_config.CATEGORIES_OFF" not in resolve('preset = "breaks-acid"\noverlay = "mine.yaml"\n').values
    # a config naming no preset uses balanced, and the entry's word for it
    assert resolve('overlay = "mine.yaml"\n').values["curate_config.CATEGORIES_OFF"] == {"FIELD"}


def test_a_bad_entry_is_a_config_error(resolve, tmp_path):
    (tmp_path / "bad.yaml").write_text("add_categories:\n  KICKS: {kind: oneshot, noun: kick}\n")
    with pytest.raises(ConfigError, match="add_categories: added category KICKS: already a category"):
        resolve('overlay = "bad.yaml"\n')


_PROBE = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from fourier import settings
from fourier.packs import curate, curate_config as cc, naming, review
settings.load_all_and_check()
print(json.dumps(dict(
    order=cc.CATEGORY_ORDER, names=list(cc.CATEGORIES), dir=cc.category_dir("FIELD"), vox=cc.category_dir("VOX"),
    field=settings.encode(cc.CATEGORIES["FIELD"]), fx_min=cc.CATEGORIES["FX"].get("exclude_min"),
    budget=cc.BUDGETS["FIELD"], mb=cc.AVG_FILE_MB["FIELD"], ceil=cc.ONESHOT_RMS_CEIL_DB["FIELD"],
    words=cc.INSTRUMENT_WORDS.get("rain"), noun=naming.CATEGORY_NOUN["FIELD"],
    drop=sorted(naming.NOUN_DROP["FIELD"]), family=review.FAMILY["FIELD"],
    skip="FIELD" in cc.SET_TOGETHER_SKIP, kit="FIELD" in curate.KIT_CATS,
)))
"""


def test_the_curation_modules_take_it(tmp_path):
    path = tmp_path / "resolved.json"
    write_resolved(path, {"curate_config.ADDED_CATEGORIES": {"FIELD": FIELD}})
    env = {**os.environ, "FOURIER_RESOLVED_CONFIG": str(path)}
    r = subprocess.run([sys.executable, "-c", _PROBE, str(ROOT / "src")], env=env,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["order"][-1] == "FIELD" and got["dir"] == "20_FIELD" and got["vox"] == "19_VOX"
    assert got["names"][got["names"].index("BLIPS") + 1] == "FIELD"
    assert (got["budget"], got["mb"], got["ceil"], got["words"]) == (40, 2.0, -18.0, "rain|drizzle")
    assert (got["noun"], got["drop"], got["family"], got["skip"], got["kit"]) == ("field", ["field"], "fx", True, False)
    assert got["fx_min"] == 0.25
    field = dict(got["field"]["$dict"])
    assert field["kind"] == "gated" and field["phrases"] == FIELD["prompts"] and field["trim_lead_db"] == -30.0
    assert field["dims"] == [encode(("br", ("dark", "bright")))]
