"""Shared test setup (loaded before any test module is imported)."""
import os
import sys
from pathlib import Path

# src layout: import fourier from this checkout's src/, not an installed copy
sys.path.insert(0, str(Path(__file__).parent / "src"))

import atexit  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402

# Each test process has places of its own. Under pytest-xdist (`pytest -n auto`) every worker
# is a process that loads this file with $PYTEST_XDIST_WORKER set (gw0, gw1, ...; "main"
# without xdist), so the folders below are one worker's, named after it, and removed when it
# exits. Subprocesses a test starts inherit them.
WORKER = os.environ.get("PYTEST_XDIST_WORKER", "main")
# the FOURIER_* variables a test run may take from outside: test inputs, never the user's setup
TEST_INPUTS = {"FOURIER_MANUALS_DIR", "FOURIER_GOLDEN", "FOURIER_TEST_SONONYM_DB"}


def _isolate() -> Path:
    """Point the environment at this process's own places; returns their folder."""
    places = Path(tempfile.mkdtemp(prefix=f"fourier-test-{WORKER}-"))
    atexit.register(shutil.rmtree, places, ignore_errors=True)

    def _place(name: str) -> str:
        p = places / name
        p.mkdir()
        return str(p)

    # The machine the suite runs on stays out of it. No FOURIER_* variable the user exported
    # reaches a test (only the test inputs in TEST_INPUTS), and $HOME is an empty folder of the
    # worker's own, so the user's ~/.config/fourier (fourier.toml, the overlay, device
    # profiles), Live's file index (~/Library/Application Support/Ableton), ~/Music and the
    # Hugging Face cache (the CLAP model) are never read. A test that wants one makes it.
    for k in [k for k in os.environ if k.startswith("FOURIER_") and k not in TEST_INPUTS]:
        del os.environ[k]
    os.environ["HOME"] = _place("user")
    for k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
        os.environ.pop(k, None)

    # builds in tests never read or fill the real processed-audio cache (~/.fourier/cache);
    # tests/test_audiocache.py points it at a temp folder and turns it on
    os.environ["FOURIER_NO_AUDIO_CACHE"] = "1"
    os.environ["FOURIER_NO_RENDER_CACHE"] = "1"   # ...nor the render cache
    # ...nor the real device path locks (a test that publishes or renders would write them)
    os.environ["FOURIER_LOCK_DIR"] = _place("locks")
    # ...nor the real pitch cache
    os.environ["FOURIER_PITCH_CACHE"] = _place("pitch")
    # ...nor any real Fourier home (~/.fourier, or one $FOURIER_HOME names):
    # a test that falls through to the default home gets an empty one of its own
    os.environ["FOURIER_HOME"] = _place("home")
    # ...nor the user's own fourier.toml (~/.config/fourier/fourier.toml, the overlay): a test
    # that wants config layers sets FOURIER_CONFIG itself
    os.environ["FOURIER_CONFIG"] = "none"
    # ...nor the user's own device profiles (~/.config/fourier/devices, devices/loader.py)
    os.environ["FOURIER_DEVICES"] = "none"
    # test libraries sit in a folder named SampleLibrary, wherever it is (fourier/places.py: a
    # library entry that is a bare folder name)
    os.environ["FOURIER_LIBRARY"] = "SampleLibrary"
    return places


# Once per process: another conftest (an overlay's own test suite) may execute this file
# again when it can't find it loaded (pytest drops a conftest from sys.modules once imported).
if "fourier_test_places" not in sys.modules:
    import types  # noqa: E402
    _mod = sys.modules["fourier_test_places"] = types.ModuleType("fourier_test_places")
    _mod.worker, _mod.root = WORKER, _isolate()

import pytest  # noqa: E402


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    """`pytest -n auto` distributes by group (`--dist loadgroup`): the end-to-end modules share
    one sandbox across their tests, in order, and are marked `xdist_group` to keep them on one
    worker. `--dist load` would split them, so it becomes `loadgroup` too (the same scheduling
    for every other test); `loadfile` and `loadscope` keep a module together and stay as given
    (`worksteal` would split them too). Runs before pytest-xdist's own hook, which would pick
    `load`; without pytest-xdist the options don't exist and this does nothing."""
    if getattr(config.option, "numprocesses", None) and getattr(config.option, "dist", None) in ("no", "load"):
        config.option.dist = "loadgroup"
        config.option.distload = False


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """A worker parses only the command line, so it's told the distribution mode chosen above
    (it names each grouped test after its group, which is how the groups reach the scheduler)."""
    node.workerinput["fourier_dist"] = node.config.option.dist


def pytest_configure(config):
    worker = getattr(config, "workerinput", None)
    if worker is None and getattr(config.option, "dist", "no") != "no":
        return                  # pytest-xdist's controller runs no test; its workers inherit its env
    # librosa's numba functions are compiled once and cached on disk; workers compiling the
    # same function at once can race on one cache (a half-written index fails the load). Each
    # worker keeps its own, which outlives the run (warm next time): under $NUMBA_CACHE_DIR when
    # the user set one, else the system temp folder. Set before any test module imports numba.
    numba = os.environ.get("NUMBA_CACHE_DIR") or Path(tempfile.gettempdir()) / "fourier-test-numba"
    os.environ["NUMBA_CACHE_DIR"] = str(Path(numba) / WORKER)
    if worker and worker.get("fourier_dist") == "loadgroup":
        config.option.loadgroup = True


def pytest_collection_modifyitems(config, items):
    """In a worker the slow tests come first (in their order), so the long end-to-end runs start
    at once, spread over the workers, and the quick tests fill in around them."""
    if hasattr(config, "workerinput"):
        items.sort(key=lambda item: item.get_closest_marker("slow") is None)


@pytest.fixture
def umbrella_vendor(monkeypatch):
    """An invented umbrella vendor, "Acme", holding one folder per pack ("Acme/<pack> Acme/..."),
    as a library overlay would set it (UMBRELLA_VENDORS, PACK_SUFFIX_RE, NAMEABLE_VENDORS;
    read at call time, so patching curate_config is enough)."""
    import re
    from fourier.packs import curate_config as cc
    monkeypatch.setattr(cc, "UMBRELLA_VENDORS", ("Acme",))
    monkeypatch.setattr(cc, "PACK_SUFFIX_RE", re.compile("Acme", re.I))
    monkeypatch.setattr(cc, "NAMEABLE_VENDORS", ("Acme",))


@pytest.fixture(autouse=True)
def _config_env_stays_put():
    """An in-process CLI call resolves config into the environment
    ($FOURIER_RESOLVED_CONFIG, $FOURIER_CONFIG); put both back after every test."""
    keep = {k: os.environ.get(k) for k in ("FOURIER_RESOLVED_CONFIG", "FOURIER_CONFIG")}
    yield
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def _fresh_vendor_layouts():
    """The library layouts vendors = "auto" detected (packs/vendors.py) are kept per process:
    forget them before every test, so one test's library never decides another's vendors."""
    from fourier.packs import vendors
    vendors.forget()
    yield
