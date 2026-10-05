"""Library scale (packs/scale.py, the scale knob): the master's size follows the analyzed
library, and a library large enough for the style's budgets builds exactly as before."""
import pytest

from fourier import knobs, layers
from fourier.packs import curate_config as cc
from fourier.packs import scale as S
from fourier.settings import ConfigError

TOTAL = sum(cc.BUDGETS.values())          # breaks-acid's budgets (the code's defaults)
SIZES = (40, 400, 4_000, 40_000, 250_000)


def _homes(n):
    """A library of n usable samples whose sounds fall into the categories as the budgets
    weigh them (each sample calls one category home at most)."""
    return {c: int(round(b * n / TOTAL)) for c, b in cc.BUDGETS.items()}


def _est(n):
    return S.estimate(dict(cc.BUDGETS), n, _homes(n), homed=lambda c: True)


def test_the_factor_follows_the_library_and_saturates_at_one():
    fs = [S.factor(n) for n in SIZES]
    assert fs == sorted(fs) and all(0 < f < 1 for f in fs[:-1])
    assert fs[-1] == 1.0                                  # 250,000 samples: the style's budgets
    assert S.factor(cc.LIBRARY_PER_MASTER * TOTAL) == 1.0 and S.factor(cc.LIBRARY_PER_MASTER * TOTAL - 1) < 1
    assert S.factor(0) == S.factor(None) == 1.0           # nothing analyzed: nothing to scale by
    assert S.factor(40) == pytest.approx(40 / (cc.LIBRARY_PER_MASTER * TOTAL))


def test_a_large_library_is_far_from_scaling():
    """A library of a few hundred thousand samples is well over the threshold (the style's
    budgets x LIBRARY_PER_MASTER), so its builds can't move when it shrinks a little."""
    threshold = cc.LIBRARY_PER_MASTER * TOTAL
    assert 80_000 < threshold < 100_000
    for n in (200_000, 250_000):
        assert n / threshold > 2 and S.factor(n) == 1.0
    # a budget at f = 1 is the budget itself, whatever the pool
    assert all(S.category_budget(b, 1.0, pool) == b for b in (6, 825, 1900) for pool in (0, 3, 10 ** 6))


def test_scale_off_or_files_keep_the_budgets(monkeypatch):
    monkeypatch.setattr(cc, "LIBRARY_SCALE", False)
    assert S.factor(40) == 1.0
    assert S.describe(40, TOTAL).startswith(f"The style's budgets: about {TOTAL:,} files (scale = \"off\")")


@pytest.mark.parametrize("n", SIZES)
def test_category_budgets_minimums_and_folders(n):
    f = S.factor(n)
    homes = _homes(n)
    est = _est(n)
    for c, b in cc.BUDGETS.items():
        pool = homes[c]
        got = est[c]
        if f == 1:
            assert got == b                                # untouched
            continue
        assert got <= b and got <= pool                    # never over the budget or the pool
        if pool < cc.SCALED_MIN_FILES:
            assert got == 0                                # too few to build: left empty
            continue
        assert got >= min(pool, cc.SCALED_KEEP_ALL)        # a small pool keeps all it has
        k = S.family_count(got, pool, 12)
        assert 1 <= k <= 12 and got // k >= cc.SCALED_FOLDER_MIN_FILES or k == 1
    files = sum(est.values())
    if f < 1:
        # about one file in LIBRARY_PER_MASTER of the library, plus what small pools keep
        assert n / cc.LIBRARY_PER_MASTER * 0.9 <= files <= n / cc.LIBRARY_PER_MASTER + 19 * cc.SCALED_KEEP_ALL
        assert files < TOTAL
    else:
        assert files == TOTAL


def test_what_each_library_size_gets():
    """The master grows with the library: more files, more categories, more folders."""
    rows = []
    for n in SIZES:
        est = _est(n)
        cats = [c for c, v in est.items() if v]
        folders = sum(S.family_count(est[c], est[c], 12) for c in cats) if S.factor(n) < 1 else None
        rows.append((n, len(cats), sum(est.values()), folders))
    files = [r[2] for r in rows]
    assert files == sorted(files) and [r[1] for r in rows] == sorted(r[1] for r in rows)
    assert rows[0][1] < 19 and rows[2][1] == 19 and rows[-1][2] == TOTAL
    assert rows[0][3] == rows[0][1]                        # a tiny master: a folder a category


def test_keep_and_the_scaled_budget():
    assert [S.keep(n) for n in (0, 2, 24, 25, 32, 104)] == [0, 2, 24, 24, 25, 34]
    # a 41-sample library's categories keep what they have
    f = S.factor(41)
    assert S.category_budget(825, f, 6) == 6 and S.category_budget(1900, f, 30) == 25
    # a 40,000-sample library: the scaled budget or one in LIBRARY_PER_MASTER of a large pool
    f = S.factor(40_000)
    assert S.category_budget(825, f, 3_000) == S.keep(3_000) < 825
    assert S.category_budget(1900, f, 1_500) == round(1900 * f)


def test_folders_and_tempo_bands():
    assert S.family_count(2, 2, 12) == 1 and S.family_count(24, 30, 12) == 1
    assert S.family_count(60, 60, 12) == 2 and S.family_count(500, 500, 12) == 12
    assert S.family_count(60, 60, 1) == 1
    # a tempo band keeps its own folder from enough candidates for SCALED_FOLDER_MIN_FILES files
    assert S.tempo_band_min(12, 12) == cc.SCALED_FOLDER_MIN_FILES
    assert S.tempo_band_min(400, 40) == 30 and S.tempo_band_min(1000, 40) == cc.TEMPO_BAND_MIN


def test_estimate_without_home_counts():
    """Before the samples' homes are known (doctor, the build log), a category's pool is its
    budget's share of the library: a tiny library's estimate is about all of it."""
    for n in (3, 41, 400):
        est = S.estimate(dict(cc.BUDGETS), n)
        assert 0.8 * n <= sum(est.values()) <= n + 1e-9
    assert S.estimate(dict(cc.BUDGETS), 250_000) == dict(cc.BUDGETS)


def test_estimate_with_home_counts():
    homed = lambda c: c in ("KICKS", "SNARES")        # noqa: E731
    est = S.estimate({"KICKS": 825, "SNARES": 750, "PIANO": 350}, 41,
                     {"KICKS": 6, "SNARES": 1}, homed=homed)
    assert est["KICKS"] == 6 and est["SNARES"] == 0
    assert est["PIANO"] == 0 or est["PIANO"] >= cc.SCALED_MIN_FILES      # the samples no category calls home


def test_describe():
    line = S.describe(41, 30, 9, S.factor(41))
    assert line.startswith("Your library has 41 usable samples: the master will hold up to about 30 files in up to 9 categories")
    assert S.describe(250_000, TOTAL).startswith(f"The style's budgets: about {TOTAL:,} files")


def test_dry_run_plan_scales_with_the_library():
    from fourier.packs import dryrun
    p = dryrun.plan(["KICKS", "SNARES", "FX"], {"KICKS": 5, "SNARES": 30, "FX": 1}, samples=41)
    rows = {r["category"]: r for r in p["rows"]}
    assert p["scale"] < 1 and p["surplus"] is False
    assert (rows["KICKS"]["budget"], rows["SNARES"]["budget"], rows["FX"]["budget"]) == (5, 25, 0)
    assert rows["KICKS"]["style_budget"] == cc.BUDGETS["KICKS"] and p["files"] == p["picks"] == 30
    assert p["categories"] == 2
    # a large library: the plan as it always was
    big = dryrun.plan(["KICKS"], {"KICKS": 5000}, samples=250_000)
    assert "style_budget" not in big["rows"][0] and big["rows"][0]["budget"] == cc.BUDGETS["KICKS"]


@pytest.fixture(scope="module")
def known():
    return layers.defaults()


def test_the_scale_knob(known):
    assert knobs.apply({"scale": "library"}, known) == []               # the default
    assert knobs.apply({"scale": "off"}, known) == [("scale", "curate_config.LIBRARY_SCALE", False)]
    with pytest.raises(ConfigError, match="scale"):
        knobs.apply({"scale": "auto"}, known)
    # files = N is an explicit size: library scaling off, even with scale = "library" beside it
    got = dict((k, v) for _, k, v in knobs.apply({"files": 3000, "scale": "library"}, known))
    assert got["curate_config.LIBRARY_SCALE"] is False and sum(got["curate_config.BUDGETS"].values()) == 3000
    # size caps from above and scaling stays on (the factor is taken against the capped budgets)
    got = dict((k, v) for _, k, v in knobs.apply({"size": "2000MB"}, known))
    assert "curate_config.LIBRARY_SCALE" not in got


def test_the_presets_scale_by_default(tmp_path, monkeypatch):
    """Both presets leave scaling on, and a fourier.toml can turn it off."""
    monkeypatch.delenv(layers.ENV_CONFIG, raising=False)
    monkeypatch.setattr(layers, "USER_CONFIG", tmp_path / "none.toml")
    p = tmp_path / "fourier.toml"
    assert layers.defaults()["curate_config.LIBRARY_SCALE"] is True
    for preset in ("breaks-acid", "balanced"):
        p.write_text(f'preset = "{preset}"\n')
        assert "curate_config.LIBRARY_SCALE" not in layers.resolve(config=str(p)).values
    p.write_text('preset = "balanced"\nscale = "off"\n')
    assert layers.resolve(config=str(p)).values["curate_config.LIBRARY_SCALE"] is False


def test_config_explains_the_scale_knob():
    import os
    import subprocess
    import sys
    env = {**os.environ, "FOURIER_CONFIG": "none"}
    out = subprocess.run([sys.executable, "-m", "fourier", "config", "explain", "scale"], env=env,
                         capture_output=True, text=True, check=True).stdout
    flat = " ".join(out.split())
    assert "scale a setting for fourier.toml" in flat and 'For example: scale = "off"' in flat
    out = subprocess.run([sys.executable, "-m", "fourier", "config", "explain", "scale", "--detail"],
                         env=env, capture_output=True, text=True, check=True).stdout
    flat = " ".join(out.split())
    assert '"off": the budgets whatever the library\'s size' in flat
    assert "sets curate_config.LIBRARY_SCALE (default True)" in flat
    assert "uses curate_config.LIBRARY_PER_MASTER (default 8)" in flat
    out = subprocess.run([sys.executable, "-m", "fourier", "config", "show"], env=env,
                         capture_output=True, text=True, check=True).stdout
    assert "scale: library (the master follows the library's size" in " ".join(out.split())
