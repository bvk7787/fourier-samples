"""config/taxonomy.yaml (fourier/taxonomy.py): the folders a build makes, read into the
curation tunables they always were."""
import pytest
import yaml

from fourier import taxonomy as T


def test_the_taxonomy_feeds_the_tunables():
    from fourier.packs import curate as C
    from fourier.packs import curate_config as cc
    from fourier.packs import naming as N
    assert cc.CATEGORY_ORDER == T.order() and set(cc.CATEGORIES) == set(T.order())
    for name, cfg in cc.CATEGORIES.items():
        assert (cfg["kind"], cfg["labels"]) == (T.kind(name), T.labels(name))
    assert cc.KICK_PHRASES == T.prompts("KICKS") and cc.BREAK_ANTI_PHRASES == T.anti("DRUMLOOPS")
    assert C.KIT_CATS == T.with_role("kit") and N.MELODIC_CATS == T.with_role("melodic")
    assert cc.SET_TOGETHER_SKIP == T.with_role("no_sets") and N.CATEGORY_NOUN["SUB"] == "bass"
    assert C.WAVES_CATEGORY == "WAVES" and cc.PHRASES_CATEGORY == "PHRASES"


@pytest.fixture
def broken(tmp_path, monkeypatch):
    doc = yaml.safe_load(T.TAXONOMY_PATH.read_text())

    def write(change):
        change(doc)
        p = tmp_path / "taxonomy.yaml"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
        monkeypatch.setattr(T, "TAXONOMY_PATH", p)
        T.load.cache_clear()
    yield write
    T.load.cache_clear()


@pytest.mark.parametrize("change, msg", [
    (lambda d: d["order"].pop(), "order must list every category"),
    (lambda d: d["categories"]["KICKS"].update(kind="thing"), "kind 'thing'"),
    (lambda d: d["categories"]["KICKS"].update(labels=["kick", "boom"]), "not in the canonical vocabulary"),
    (lambda d: d["categories"]["KICKS"].update(roles=["kit", "loud"]), "unknown roles"),
    (lambda d: d["homes"].update(waves="WAVEZ"), "homes.waves"),
])
def test_a_broken_taxonomy_says_what(broken, change, msg):
    broken(change)
    with pytest.raises(T.TaxonomyError, match=msg):
        T.load()


def test_engine_settings_and_taxonomy_agree():
    with pytest.raises(T.TaxonomyError, match="not both"):
        T.with_taxonomy({"KICKS": {}})


def test_explain_categories(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from fourier.cli import main
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "none"))
    r = CliRunner().invoke(main, ["--db", str(tmp_path / "t.db"), "config", "explain", "categories"])
    assert r.exit_code == 0, r.output
    assert "01_KICKS  oneshot; labels: kick; roles: kit" in r.output
    assert "00_KITS" in r.output and "19_VOX" in r.output
