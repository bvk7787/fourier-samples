"""The shipped style presets (config/presets/): breaks-acid is the reference style, balanced the
default for a first run. Both build and pass verify on the synthetic library (CI:
synthetic_build.py --preset)."""
import pytest

from fourier import layers


@pytest.fixture
def resolve(tmp_path, monkeypatch):
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "none.toml")

    def run(text):
        p = tmp_path / "fourier.toml"
        p.write_text(text)
        return layers.resolve(config=str(p))
    return run


def test_balanced_is_the_code_defaults(resolve):
    r = resolve('preset = "balanced"\n')
    assert r.values == {} and r.files[0].endswith("config/presets/balanced.yaml")
    known = layers.defaults()
    assert known["curate_config.TEMPO_BANDS"][:3] == (60, 70, 75)


def test_breaks_acid(resolve):
    v = resolve('preset = "breaks-acid"\n').values
    known = layers.defaults()
    assert "curate_config.CATEGORIES_OFF" not in v                  # every category
    b, b0 = v["curate_config.BUDGETS"], known["curate_config.BUDGETS"]
    assert b["DRUMLOOPS"] > b0["DRUMLOOPS"] and b["PADS"] < b0["PADS"] and b["FX"] == b0["FX"]
    assert v["curate_config.TEMPO_BANDS"][:3] == (60, 75, 85)
    assert "trap" in v["curate_config.DRUMLOOP_STYLE_EXCLUDE"].pattern


def test_your_knobs_go_over_the_preset(resolve):
    v = resolve('preset = "breaks-acid"\ncategories = { VOX = "off" }\ntempo = "70-180"\n').values
    assert v["curate_config.TEMPO_BANDS"] == layers.defaults()["curate_config.TEMPO_BANDS"]
    assert v["curate_config.CATEGORIES_OFF"] == {"VOX"}
