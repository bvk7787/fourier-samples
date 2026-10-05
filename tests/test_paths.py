"""FOURIER_HOME relocates every piece of Fourier's own state (fourier.paths)."""
import re
from pathlib import Path

import pytest

# the narrower overrides that would otherwise pin a piece somewhere else
_OVERRIDES = ("FOURIER_HOME", "FOURIER_RATINGS", "FOURIER_AUDIO_CACHE", "FOURIER_RENDER_CACHE",
              "FOURIER_NO_RENDER_CACHE", "FOURIER_CLAP_TEXT_CACHE", "FOURIER_CLAP_INDEX",
              "FOURIER_STAGING_DIR", "FOURIER_PITCH_CACHE", "FOURIER_DESCRIBE_CACHE")


@pytest.fixture
def clean_env(monkeypatch):
    for k in _OVERRIDES:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _all_state_paths():
    """Every place Fourier keeps its own state, as the code resolves it now."""
    from fourier import cli
    from fourier.analysis import clap_features
    from fourier.db import session
    from fourier.packs import audiocache, builddiff, curate, ratings, render
    return {
        "db": session.get_db_path(),
        "ratings": Path(ratings.default_store()),
        "scorecards": Path(ratings.default_scorecards()),
        "builds": builddiff.builds_dir(),
        "audio_cache": audiocache.cache_dir(),
        "render_cache": render._render_cache_dir(),
        "pitch_cache": curate._pitch_cache_dir(),
        "describe_cache": curate._describe_cache_dir(),
        "clap_text_cache": clap_features._text_cache_path("kick").parent,
        "clap_index": clap_features._index_path(),
        "staging": Path(cli._release_staging(str(Path(ratings.default_store()).parent
                                                   / "Dropbox"))),
    }


def test_default_home_is_dot_fourier(clean_env, tmp_path):
    from fourier.paths import clap_index_path, fourier_home
    clean_env.setattr(Path, "home", lambda: tmp_path)
    assert fourier_home() == tmp_path / ".fourier"
    assert clap_index_path() == tmp_path / ".fourier" / "clap_index.npz"


def test_default_layout_is_unchanged(clean_env, tmp_path):
    """With nothing set, every path is exactly where it was before FOURIER_HOME existed."""
    clean_env.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".fourier"
    (home / "Dropbox").mkdir(parents=True)
    paths = _all_state_paths()
    assert paths["db"] == home / "library.duckdb"
    assert paths["ratings"] == home / "ratings.json"
    assert paths["builds"] == home / "builds"
    assert paths["audio_cache"] == home / "cache" / "audio"
    assert paths["render_cache"] == home / "cache" / "render"
    assert paths["clap_text_cache"] == home / "cache" / "clap_text"
    assert paths["clap_index"] == home / "clap_index.npz"
    assert paths["staging"] == home / "staging"


def test_fourier_home_moves_every_path(clean_env, tmp_path):
    home = tmp_path / "elsewhere"
    home.mkdir()
    (home / "Dropbox").mkdir()
    clean_env.setenv("FOURIER_HOME", str(home))
    paths = _all_state_paths()
    outside = {k: p for k, p in paths.items() if home not in Path(p).parents}
    assert not outside, f"not under FOURIER_HOME: {outside}"


def test_narrow_overrides_still_win(clean_env, tmp_path):
    clean_env.setenv("FOURIER_HOME", str(tmp_path / "home"))
    clean_env.setenv("FOURIER_CLAP_INDEX", str(tmp_path / "idx" / "clap.npz"))
    clean_env.setenv("FOURIER_RATINGS", str(tmp_path / "r" / "ratings.json"))
    from fourier.analysis import clap_features
    from fourier.packs import builddiff, ratings
    assert clap_features._index_path() == tmp_path / "idx" / "clap.npz"
    assert ratings.default_store() == str(tmp_path / "r" / "ratings.json")
    assert builddiff.builds_dir() == tmp_path / "r" / "builds"   # follows the ratings store


def test_no_hardcoded_home_paths_left():
    """Code resolves its state through fourier.paths; only docs and comments may say ~/.fourier."""
    src = Path(__file__).parents[1] / "src" / "fourier"
    bad = re.compile(r"""expanduser\(\s*["']~/\.fourier|home\(\)\s*/\s*["']\.fourier""")
    hits = [f"{p.relative_to(src)}:{i}" for p in src.rglob("*.py") if p.name != "paths.py"
            for i, line in enumerate(p.read_text().splitlines(), 1) if bad.search(line)]
    assert not hits, hits
