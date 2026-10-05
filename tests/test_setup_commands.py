"""fourier setup (the wizard) and fourier doctor (fourier/cli/setup.py). Nothing here touches
the network: the installers, the model download and Ollama are stand-ins."""
import os
import sys
import tomllib

import pytest
from click.testing import CliRunner

from fourier import places
from fourier.cli import enrich as E
from fourier.cli import main
from fourier.cli import setup as S

PY = sys.executable
REAL_OLLAMA_STATUS = S.ollama_status
REAL_UV_TOOL_RECEIPT = S.uv_tool_receipt
REAL_TORCH_HAS_CUDA = S.torch_has_cuda


@pytest.fixture(autouse=True)
def no_ollama(monkeypatch, tmp_path):
    """The same machine everywhere: no Ollama, no CLAP extra or model, and a home folder of
    the test's own (so Live's index, the default output folders and anything else under ~
    come from the test, not from the machine running it)."""
    from fourier.cli import enrich as E
    monkeypatch.setattr(S, "ollama_status", lambda: (False, False, []))
    monkeypatch.setattr(E, "clap_extra_missing", lambda: ["torch", "transformers"])
    monkeypatch.setattr(S, "clap_model_cached", lambda: False)
    monkeypatch.setattr(S, "uv_tool_receipt", lambda prefix=None: None)    # not a uv tool
    monkeypatch.setattr(S, "torch_has_cuda", lambda: False)
    monkeypatch.setattr(S, "installed_torch", lambda exe=None: None)      # nothing to pin
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fourier-home"))     # install.json


def _run(tmp_path, monkeypatch, *args, input=None, config=None):
    monkeypatch.setenv("FOURIER_CONFIG", str(config or tmp_path / "fourier.toml"))
    monkeypatch.delenv("FOURIER_LIBRARY", raising=False)
    places.reset()
    try:
        return CliRunner().invoke(main, ["--db", str(tmp_path / "t.duckdb"), *args], input=input)
    finally:
        places.reset()


def _flat(out):
    return " ".join(out.split())


def _has(out, text):
    """Whether out says text, wherever a terminal or Rich wrapped either (a long temporary
    path, as macOS's /private/var/folders/..., is folded mid-path at the console's width)."""
    return "".join(text.split()) in "".join(out.split())


def _lib(tmp_path, n=3):
    lib = tmp_path / "Samples"
    (lib / "Acme" / "Kicks").mkdir(parents=True)
    for i in range(n):
        (lib / "Acme" / "Kicks" / f"Kick {i}.wav").write_bytes(b"")
    (lib / "notes.txt").write_text("not audio")
    return lib


def test_setup_yes_writes_the_answers(tmp_path, monkeypatch):
    lib, cfg = _lib(tmp_path), tmp_path / "fourier.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--library", str(lib),
               "--device", "m8_tracker, generic_48k", "--no-clap")
    assert res.exit_code == 0, res.output
    music, tag = tmp_path / "home" / "Music", tmp_path.name      # a fourier.toml: its folder's name
    assert tomllib.loads(cfg.read_text()) == {
        "library": [str(lib)], "devices": ["m8_tracker", "generic_48k"], "preset": "balanced",
        "vendors": "auto", "fold": "auto",
        "output": {"master": str(music / f"FourierCurated-{tag}"), "renders": str(music / f"FourierRenders-{tag}"),
                   "publish": str(music / f"Fourier-{tag}")}}
    out = _flat(res.output)
    assert "isn't the default config" in out
    assert "Found 3 audio files." in out and "1/6 Your samples" in out and "6/6 Check" in out
    assert "Skipped. A build needs it" in out and "https://ollama.com" in out
    assert "CLAP model" in out and "would stop a build" in out       # no model: no build offered
    assert "fourier devices show <id>" in cfg.read_text() and "config/devices" not in cfg.read_text()


def test_the_default_config_keeps_the_default_folders_and_another_gets_its_own(tmp_path, monkeypatch):
    """Setup writing the default config (~/.config/fourier/fourier.toml) writes no [output]: the
    master is ~/Music/FourierCurated. Another config is a second library's: its master,
    renders and releases are named after it, so a build of one never replaces the other's."""
    lib = _lib(tmp_path)
    home_cfg = tmp_path / "home" / ".config" / "fourier" / "fourier.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--library", str(lib), "--device", "m8_tracker",
               "--no-clap", config=home_cfg)
    assert res.exit_code == 0, res.output
    assert "output" not in tomllib.loads(home_cfg.read_text()) and "isn't the default config" not in res.output
    assert S.config_tag(home_cfg) is None and S.config_tag(tmp_path / "drums.toml") == "drums"
    assert S.config_tag(tmp_path / "Two Libs" / "fourier.toml") == "Two-Libs"
    other = tmp_path / "drums.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(other), "--library", str(lib),
               "--device", "m8_tracker", "--no-clap", "--master", str(tmp_path / "Mine"), config=other)
    assert res.exit_code == 0, res.output
    out = tomllib.loads(other.read_text())["output"]
    assert out["master"] == str(tmp_path / "Mine")                      # as given
    assert out["publish"] == str(tmp_path / "home" / "Music" / "Fourier-drums")
    # the same answers again: kept (the folders it would name are the ones it wrote)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(other), "--library", str(lib),
               "--device", "m8_tracker", "--no-clap", "--publish", out["publish"], config=other)
    assert res.exit_code == 0 and "Kept the config" in _flat(res.output), res.output


def test_setup_refuses_to_overwrite_a_config(tmp_path, monkeypatch):
    lib, cfg = _lib(tmp_path), tmp_path / "fourier.toml"
    base = ("setup", "--yes", "--to", str(cfg), "--library", str(lib), "--device", "m8_tracker", "--no-clap")
    assert _run(tmp_path, monkeypatch, *base).exit_code == 0
    before = cfg.read_text()
    res = _run(tmp_path, monkeypatch, *base, "--preset", "breaks-acid")
    out = _flat(res.output)
    assert res.exit_code == 1 and cfg.read_text() == before
    assert _has(out, f"A config exists at {cfg}; pass --force to replace it, or drop the options to keep it")
    # the same answers again (as the README's line, run twice): kept, and said so
    res = _run(tmp_path, monkeypatch, *base)
    assert res.exit_code == 0 and cfg.read_text() == before and "Kept the config at" in _flat(res.output)
    # asked: no, keep it
    res = _run(tmp_path, monkeypatch, "setup", "--to", str(cfg), "--library", str(lib),
               "--device", "m8_tracker", "--preset", "breaks-acid", "--no-clap",
               input="\n\n\ny\nn\n")                 # size, processing, categories as they are
    assert res.exit_code == 0 and "Replace it?" in res.output and cfg.read_text() == before, res.output
    # --yes with nothing to change keeps it and runs the other steps
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--no-clap")
    assert res.exit_code == 0 and cfg.read_text() == before and "2/6" not in res.output
    assert _has(res.output, f"Kept the config at {cfg}")
    res = _run(tmp_path, monkeypatch, *base, "--preset", "breaks-acid", "--force")
    assert res.exit_code == 0 and tomllib.loads(cfg.read_text())["preset"] == "breaks-acid"


def test_setup_names_the_config_as_given(tmp_path, monkeypatch):
    """Messages show the config's path as the user typed it (not expanded or resolved, and
    never folded mid-path however long it is)."""
    lib = _lib(tmp_path)
    base = ("setup", "--yes", "--to", "~/cfg/fourier.toml", "--library", str(lib), "--device", "m8_tracker",
            "--no-clap")
    cfg = tmp_path / "home" / "cfg" / "fourier.toml"
    res = _run(tmp_path, monkeypatch, *base, config=cfg)
    assert res.exit_code == 0 and "Wrote ~/cfg/fourier.toml." in res.output and cfg.exists()
    res = _run(tmp_path, monkeypatch, *base, "--preset", "breaks-acid", config=cfg)
    assert res.exit_code == 1 and "A config exists at ~/cfg/fourier.toml; pass --force" in res.output
    deep = tmp_path / ("a-rather-long-folder-name-" * 4) / "fourier.toml"
    base = ("setup", "--yes", "--to", str(deep), "--library", str(lib), "--device", "m8_tracker", "--no-clap")
    _run(tmp_path, monkeypatch, *base, config=deep)
    res = _run(tmp_path, monkeypatch, *base, "--preset", "breaks-acid", config=deep)
    assert _has(res.output, f"A config exists at {deep}; pass --force")


def test_setup_refuses_what_it_cant_use(tmp_path, monkeypatch):
    lib, cfg = _lib(tmp_path), tmp_path / "fourier.toml"
    for args, why in (([f"--library={tmp_path}/nope", "--device=m8_tracker"], "not folders"),
                      ([f"--library={lib}", "--device=walkman"], "unknown devices"),
                      ([f"--library={lib}", "--device=m8_tracker", "--preset=polka"], "unknown preset")):
        res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--no-clap", *args)
        assert res.exit_code == 1 and why in res.output
        assert not cfg.exists()
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg))
    assert res.exit_code == 2 and "needs --library" in res.output and not cfg.exists()


def test_setup_asks_and_asks_again(tmp_path, monkeypatch):
    lib, cfg = _lib(tmp_path), tmp_path / "fourier.toml"
    answers = [f"{tmp_path}/nope", str(lib),          # not a folder, then one
               "walkman", "m8_tracker, generic_48k",  # not a device, then two
               "",                                    # the default style
               "", "", "",                            # the full size, prepared, every category
               "n", str(tmp_path / "Curated"), "", "",  # change the master, keep the others
               "n"]                                   # no CLAP for now
    res = _run(tmp_path, monkeypatch, "setup", "--to", str(cfg), input="\n".join(answers) + "\n")
    assert res.exit_code == 0, res.output
    out = _flat(res.output)
    assert "not folders" in out and "unknown devices: walkman" in out
    assert "balanced" in out and "breaks-acid" in out and "Elektron Digitakt 2" in out
    doc = tomllib.loads(cfg.read_text())
    music, tag = tmp_path / "home" / "Music", tmp_path.name      # the others: named after the config
    assert doc == {"library": [str(lib)], "devices": ["m8_tracker", "generic_48k"], "preset": "balanced",
                   "vendors": "auto", "fold": "auto",
                   "output": {"master": str(tmp_path / "Curated"), "renders": str(music / f"FourierRenders-{tag}"),
                              "publish": str(music / f"Fourier-{tag}")}}
    assert "What do you make?" in out and "hiphop-lofi" in out and "fourier devices new" in out


def test_setup_keeps_a_config_when_asked(tmp_path, monkeypatch):
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(f'library = ["{_lib(tmp_path)}"]\n')
    res = _run(tmp_path, monkeypatch, "setup", "--to", str(cfg), input="y\nn\n")
    assert res.exit_code == 0 and "Keep it" in res.output and "1/6" not in res.output, res.output
    assert cfg.read_text().startswith("library = ")
    # blank for the library skips writing a config at all
    other = tmp_path / "other.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--to", str(other), input="\nn\n")
    assert res.exit_code == 0 and "no config written" in res.output and not other.exists()


def test_setup_warns_about_cloud_only_files(tmp_path, monkeypatch):
    from fourier import platforms
    lib = _lib(tmp_path)
    monkeypatch.setattr(platforms, "cloud_only", lambda p: "Kick 1" in str(p))
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(lib), "--no-clap")
    out = _flat(res.output)
    assert "1 of them are only in the cloud" in out and "Keep Downloaded" in out


# --- the CLAP install --------------------------------------------------------------------
def _which(monkeypatch, **found):
    monkeypatch.setattr(S.shutil, "which", lambda name: found.get(name.replace("-", "_")))


def test_clap_requirements_are_the_extras():
    with open(S.Path(__file__).parents[1] / "pyproject.toml", "rb") as f:
        extra = tomllib.load(f)["project"]["optional-dependencies"]["clap"]
    assert sorted(S.clap_requirements()) == sorted(extra) == sorted(S.CLAP_REQUIREMENTS)


def test_cpu_index_on_linux_without_nvidia(monkeypatch):
    monkeypatch.setattr(S.sys, "platform", "linux")
    monkeypatch.setattr(S, "installed_from_url", lambda: True)
    _which(monkeypatch, uv="/bin/uv")
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": True)
    tool, cmds = S.clap_install(PY)
    assert tool == "uv" and cmds == [["/bin/uv", "pip", "install", "--python", PY, "--torch-backend", "cpu",
                                      *S.clap_requirements()]]
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": False)    # an older uv
    # PyTorch's CPU index beside PyPI, each package's best version from either: by default uv
    # takes a package from the first index that has it (an old packaging or urllib3)
    tool, cmds = S.clap_install(PY)
    assert cmds == [["/bin/uv", "pip", "install", "--python", PY, "--extra-index-url", S.TORCH_CPU_INDEX,
                     "--index-strategy", "unsafe-best-match", *S.clap_requirements()]]
    _which(monkeypatch)                                                 # pip, no uv
    monkeypatch.setattr(S, "has_pip", lambda: True)
    tool, cmds = S.clap_install(PY)
    assert tool == "pip" and cmds == [[PY, "-m", "pip", "install", "--index-url", S.TORCH_CPU_INDEX, "torch>=2.2"],
                                      [PY, "-m", "pip", "install", *S.clap_requirements()]]
    _which(monkeypatch, nvidia_smi="/bin/nvidia-smi")                   # a GPU: the default index
    tool, cmds = S.clap_install(PY)
    assert cmds == [[PY, "-m", "pip", "install", *S.clap_requirements()]]


def test_clap_install_in_a_uv_tool_reinstalls_the_tool_with_it(monkeypatch, tmp_path):
    """In a uv tool environment the [clap] requirements go in as --with on a reinstall from
    the receipt's source, so `uv tool upgrade` keeps them."""
    (tmp_path / S.UV_RECEIPT).write_text(
        '[tool]\nrequirements = [{ name = "fourier-samples", git = "https://example.org/acme/fs?rev=abc" },'
        ' { name = "rich" }]\n')
    monkeypatch.setattr(S, "uv_tool_receipt", lambda prefix=None: REAL_UV_TOOL_RECEIPT(tmp_path))
    monkeypatch.setattr(S.sys, "platform", "darwin")
    _which(monkeypatch, uv="/bin/uv")
    tool, (cmd,) = S.clap_install(PY)
    py = f"{S.sys.version_info.major}.{S.sys.version_info.minor}"
    assert tool == "uv tool" and cmd == [
        "/bin/uv", "tool", "install", "--reinstall", "--python", py,
        *[a for r in S.clap_requirements() for a in ("--with", r)], "--with", "rich",
        "fourier-samples @ git+https://example.org/acme/fs@abc"]
    # Linux without a GPU: the CPU build, by --torch-backend or the CPU index
    monkeypatch.setattr(S.sys, "platform", "linux")
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": group == "tool")
    assert S.clap_install(PY)[1][0][6:8] == ["--torch-backend", "cpu"]
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": False)
    assert S.clap_install(PY)[1][0][6:10] == ["--index", S.TORCH_CPU_INDEX, "--index-strategy",
                                              "unsafe-best-match"]
    # a receipt's other sources
    assert S._requirement({"name": "fourier-samples", "specifier": ">=0.3"}) == ["fourier-samples>=0.3"]
    assert S._requirement({"name": "fourier-samples", "editable": "/src/fs"}) == ["--editable", "/src/fs"]
    assert S._requirement({"name": "fourier-samples", "directory": "/src/fs"}) == ["fourier-samples @ file:///src/fs"]


def test_setup_says_only_what_it_downloads(tmp_path, monkeypatch, clap_env):
    """The model already downloaded: setup says so, installs only PyTorch and Transformers,
    and loads the model without downloading it."""
    _which(monkeypatch)
    monkeypatch.setattr(S, "has_pip", lambda: True)
    monkeypatch.setattr(S, "clap_model_cached", lambda: True)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert res.exit_code == 0, res.output
    assert "the model is downloaded already" in out and "600 MB" not in out and "Downloading" not in out
    assert clap_env["downloaded"] == 0 and clap_env["loaded"] == 1 and "The CLAP model is ready." in out


def test_clap_install_from_an_index_installs_the_package(monkeypatch):
    monkeypatch.setattr(S.sys, "platform", "darwin")
    monkeypatch.setattr(S, "installed_from_url", lambda: False)
    _which(monkeypatch, uv="/bin/uv")
    tool, cmds = S.clap_install(PY)
    assert cmds == [["/bin/uv", "pip", "install", "--python", PY, f"fourier-samples[clap]=={S._version()}"]]


def test_installed_from_url_reads_direct_url(monkeypatch):
    class Dist:
        def __init__(self, text):
            self.text = text

        def read_text(self, name):
            return self.text if name == "direct_url.json" else None
    monkeypatch.setattr(S.importlib.metadata, "distribution", lambda name: Dist('{"url": "git+https://x"}'))
    assert S.installed_from_url()
    monkeypatch.setattr(S.importlib.metadata, "distribution", lambda name: Dist(None))
    assert not S.installed_from_url()


@pytest.fixture
def clap_env(monkeypatch):
    """The CLAP extra missing until an install runs; the download and load recorded."""
    state = {"installed": False, "ran": [], "downloaded": 0, "loaded": 0}
    monkeypatch.setattr(E, "clap_extra_missing", lambda: not state["installed"])
    monkeypatch.setattr(S, "clap_model_cached", lambda: False)

    def run(cmd, keep=None, quiet=False):
        state["ran"].append(cmd)
        state.setdefault("quiet", []).append(quiet)
        state["installed"] = True
        return 0
    monkeypatch.setattr(S, "_run", run)
    monkeypatch.setattr(S, "download_clap_model", lambda: state.__setitem__("downloaded", state["downloaded"] + 1))
    monkeypatch.setattr(S, "load_clap_model", lambda: state.__setitem__("loaded", state["loaded"] + 1))
    monkeypatch.setattr(S, "cpu_only_torch", lambda: False)
    monkeypatch.setattr(S, "installed_from_url", lambda: True)
    monkeypatch.setattr(S.sys, "platform", "linux")       # the same wording on every machine
    return state


def test_setup_installs_clap_and_downloads_the_model(tmp_path, monkeypatch, clap_env):
    _which(monkeypatch)
    monkeypatch.setattr(S, "has_pip", lambda: True)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    assert res.exit_code == 0, res.output
    assert clap_env["ran"] == [[PY, "-m", "pip", "install", *S.clap_requirements()]]
    assert clap_env["downloaded"] == 1 and clap_env["loaded"] == 1
    out = _flat(res.output)
    assert "PyTorch with CUDA (for the NVIDIA GPU here) is several GB" in out    # (cpu_only_torch: False)
    assert "The CLAP model is ready." in out
    assert "Ready to build" in out and "Next: fourier build (it can stop" in out    # --yes: no build


def test_setup_prints_the_command_when_nothing_can_install(tmp_path, monkeypatch, clap_env):
    _which(monkeypatch)
    monkeypatch.setattr(S, "has_pip", lambda: False)
    monkeypatch.setattr(S, "cpu_only_torch", lambda: True)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    assert res.exit_code == 0, res.output
    assert clap_env["ran"] == [] and clap_env["downloaded"] == 0
    out = _flat(res.output)
    assert "Neither uv nor pip" in out and _has(out, f"uv pip install --python {PY} --torch-backend cpu")
    assert "install.sh" in out and "6/6 Check" in out


# what uv 0.8 printed for `uv tool install --index <PyTorch CPU index>` (no --index-strategy):
# its build backend's requirements came from the first index, which has an old packaging
UV_FAILED = """Resolved 3 packages in 1.20s
  \x1b[31m×\x1b[0m Failed to build `fourier-samples @ file:///tmp/src/fourier-samples`
  ├─▶ Failed to resolve requirements from `build-system.requires`
  ├─▶ No solution found when resolving: `hatchling>=1.27`
  ╰─▶ Because only packaging<=24.1 is available and all of:
          hatchling>=1.27.0,<=1.29.0
      depend on packaging>=24.2, we can conclude that all of:
       cannot be used.

      hint: `packaging` was found on https://download.pytorch.org/whl/cpu,
      but not at the requested version (packaging>=24.2). A compatible version
      may be available on a subsequent index (e.g., https://pypi.org/simple).
"""


def test_setup_says_so_when_the_install_fails(tmp_path, monkeypatch, clap_env):
    """A failed install: what the installer said that matters (its error and hint, not the
    whole log), and the exact command to retry."""
    _which(monkeypatch, uv="/bin/uv")
    monkeypatch.setattr(S, "_run", lambda cmd, keep=None: clap_env["ran"].append(cmd)
                        or keep.extend(UV_FAILED.splitlines()) or 1)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert res.exit_code == 0 and clap_env["downloaded"] == 0
    assert "The install didn't finish. What it said: × Failed to build" in out and "Fix what" not in out
    assert "╰─▶ Because only packaging<=24.1 is available" in out and "hatchling>=1.27.0,<=1.29.0" not in out
    assert "hint: `packaging` was found on https://download.pytorch.org/whl/cpu, but not at" in out
    assert "Resolved 3 packages" not in out and "\x1b[" not in res.output
    assert _has(out, f"Retry with this command, then `fourier setup` again: {S.shlex.join(clap_env['ran'][0])}")
    assert not (tmp_path / "fourier-home" / "install.json").exists()


def test_error_lines():
    assert S.error_lines(["Collecting torch", "ERROR: No matching distribution found for torch>=2.2"]) == [
        "ERROR: No matching distribution found for torch>=2.2"]
    assert S.error_lines([f"line {i}" for i in range(30)], n=3) == ["line 27", "line 28", "line 29"]
    got = S.error_lines(UV_FAILED.splitlines())
    assert got[0] == "× Failed to build `fourier-samples @ file:///tmp/src/fourier-samples`"
    assert got[-1] == "  may be available on a subsequent index (e.g., https://pypi.org/simple)."


def test_an_old_uv_is_offered_an_update(monkeypatch, tmp_path):
    """uv before 0.9.19 can't name the CPU build in `uv tool install`: setup offers `uv self
    update`; declined (or failed), it installs from PyTorch's CPU index beside PyPI."""
    (tmp_path / S.UV_RECEIPT).write_text('[tool]\nrequirements = [{ name = "fourier-samples" }]\n')
    monkeypatch.setattr(S, "uv_tool_receipt", lambda prefix=None: REAL_UV_TOOL_RECEIPT(tmp_path))
    monkeypatch.setattr(S.sys, "platform", "linux")
    _which(monkeypatch, uv="/bin/uv")
    state = {"new": False, "ran": [], "ok": True}
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": state["new"] or group == "pip")
    monkeypatch.setattr(S, "uv_version", lambda uv: (0, 9, 30) if state["new"] else (0, 8, 17))

    def run(cmd, keep=None):
        state["ran"].append(cmd)
        if cmd[1:] == ["self", "update"] and state["ok"]:
            state["new"] = True
        return 0 if state["ok"] else 1
    monkeypatch.setattr(S, "_run", run)

    class Wizard:
        def __init__(self, answer):
            self.answer, self.asked = answer, []

        def confirm(self, text, default):
            self.asked.append(text)
            return self.answer
    assert S.uv_too_old("uv tool") == ((0, 8, 17), (0, 9, 19)) and S.uv_too_old("uv") is None
    w = Wizard(True)
    assert S.offer_uv_update(w, "uv tool") and state["ran"] == [["/bin/uv", "self", "update"]]
    assert w.asked == ["Run `uv self update` now?"]
    assert S.clap_install(PY)[1][0][6:8] == ["--torch-backend", "cpu"]
    # declined: the CPU index, each package's best version from either index
    state.update(new=False, ran=[])
    assert not S.offer_uv_update(Wizard(False), "uv tool") and state["ran"] == []
    assert S.clap_install(PY)[1][0][6:10] == ["--index", S.TORCH_CPU_INDEX, "--index-strategy",
                                              "unsafe-best-match"]
    # a uv that can't update itself (from a package manager): the CPU index too
    state["ok"] = False
    assert not S.offer_uv_update(Wizard(True), "uv tool") and state["ran"] == [["/bin/uv", "self", "update"]]
    # a GPU, or a uv new enough: nothing to offer
    _which(monkeypatch, uv="/bin/uv", nvidia_smi="/bin/nvidia-smi")
    assert S.uv_too_old("uv tool") is None


def test_setup_yes_never_updates_uv(tmp_path, monkeypatch, clap_env):
    """--yes never updates uv (a tool on the machine): it says how, and installs PyTorch's CPU
    build from its index; install.json records that setup installed it."""
    (tmp_path / S.UV_RECEIPT).write_text('[tool]\nrequirements = [{ name = "fourier-samples" }]\n')
    monkeypatch.setattr(S, "uv_tool_receipt", lambda prefix=None: REAL_UV_TOOL_RECEIPT(tmp_path))
    monkeypatch.setattr(S, "cpu_only_torch", lambda: True)
    _which(monkeypatch, uv="/bin/uv")
    new = []
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": bool(new))
    monkeypatch.setattr(S, "uv_version", lambda uv: (0, 8, 17))
    real_run = S._run

    def run(cmd, keep=None, quiet=False):
        if cmd[1:] == ["self", "update"]:
            new.append(1)
            clap_env["ran"].append(cmd)
            return 0
        return real_run(cmd, keep, quiet)
    monkeypatch.setattr(S, "_run", run)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert res.exit_code == 0, res.output
    assert "uv 0.8.17 is older than 0.9.19" in out and "$ uv self update" not in out
    assert "Kept this uv (--yes doesn't update it): installing PyTorch from its CPU index instead." in out
    assert "`uv self update`, then `fourier setup` again, uses the newer uv." in out
    assert ["/bin/uv", "self", "update"] not in clap_env["ran"] and not new
    assert "--torch-backend" not in clap_env["ran"][0] and S.TORCH_CPU_INDEX in clap_env["ran"][0]
    import json
    assert json.loads((tmp_path / "fourier-home" / "install.json").read_text())["torch"] == "cpu"


def _fake_uv(monkeypatch, version: str):
    """`uv --version` and `uv pip|tool install --help` as a uv of this version prints them:
    --torch-backend in `uv pip install` from 0.6.9, in `uv tool install` from 0.9.19."""
    v = tuple(int(x) for x in version.split("."))

    def run(cmd, **kw):
        if cmd[1:] == ["--version"]:
            out = f"uv {version} (abc1234 2025-09-10)\n"
        elif cmd[2:] == ["install", "--help"]:
            has = v >= S.UV_TORCH_BACKEND["tool" if cmd[1] == "tool" else "pip"]
            out = "Usage: uv install\n      --index-strategy <INDEX_STRATEGY>\n" + (
                "      --torch-backend <TORCH_BACKEND>\n" if has else "")
        else:
            raise AssertionError(cmd)
        return S.subprocess.CompletedProcess(cmd, 0, out, "")
    monkeypatch.setattr(S.subprocess, "run", run)


@pytest.mark.parametrize("version, tool_args, pip_args", [
    ("0.8.17", ["--index", S.TORCH_CPU_INDEX, "--index-strategy", "unsafe-best-match"],
     ["--torch-backend", "cpu"]),
    ("0.6.0", ["--index", S.TORCH_CPU_INDEX, "--index-strategy", "unsafe-best-match"],
     ["--extra-index-url", S.TORCH_CPU_INDEX, "--index-strategy", "unsafe-best-match"]),
    ("0.9.30", ["--torch-backend", "cpu"], ["--torch-backend", "cpu"]),
])
def test_cpu_build_commands_for_old_and_new_uv(monkeypatch, tmp_path, version, tool_args, pip_args):
    """What this uv's help says it takes decides the CPU build's flags: --torch-backend, else
    PyTorch's CPU index beside PyPI with each package's best version from either (uv 0.8's
    `uv tool install` has no --torch-backend, and by default takes a package from the first
    index that has it: an old packaging the build backend can't use)."""
    _fake_uv(monkeypatch, version)
    _which(monkeypatch, uv="/bin/uv")
    monkeypatch.setattr(S.sys, "platform", "linux")
    monkeypatch.setattr(S, "installed_from_url", lambda: True)
    assert S.uv_version("/bin/uv") == tuple(int(x) for x in version.split("."))
    assert S.clap_install(PY)[1] == [["/bin/uv", "pip", "install", "--python", PY, *pip_args,
                                      *S.clap_requirements()]]
    (tmp_path / S.UV_RECEIPT).write_text('[tool]\nrequirements = [{ name = "fourier-samples" }]\n')
    monkeypatch.setattr(S, "uv_tool_receipt", lambda prefix=None: REAL_UV_TOOL_RECEIPT(tmp_path))
    py = f"{S.sys.version_info.major}.{S.sys.version_info.minor}"
    assert S.clap_install(PY)[1] == [["/bin/uv", "tool", "install", "--reinstall", "--python", py, *tool_args,
                                      *[a for r in S.clap_requirements() for a in ("--with", r)],
                                      "fourier-samples"]]
    old = S.uv_too_old("uv tool")
    assert (old is None) == (tool_args[0] == "--torch-backend")


def test_a_failed_install_with_an_old_uv_says_to_update_it(tmp_path, monkeypatch, clap_env):
    """With a uv older than the known-good one, the one next step is updating uv (not the
    command that just failed)."""
    _which(monkeypatch, uv="/bin/uv")
    monkeypatch.setattr(S, "uv_version", lambda uv: (0, 8, 17))
    monkeypatch.setattr(S, "_run", lambda cmd, keep=None: keep.extend(UV_FAILED.splitlines()) or 1)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert "The install didn't finish. What it said: × Failed to build" in out
    assert "Your uv (0.8.17) is old: run `uv self update` and then `fourier setup` again." in out
    assert "Retry with this command" not in out


def _tool_receipt(tmp_path, monkeypatch, *extra):
    """This copy as a uv tool, its receipt with these requirements besides its own."""
    reqs = ['{ name = "fourier-samples" }', *extra]
    (tmp_path / S.UV_RECEIPT).write_text(f'[tool]\nrequirements = [{", ".join(reqs)}]\n')
    monkeypatch.setattr(S, "uv_tool_receipt", lambda prefix=None: REAL_UV_TOOL_RECEIPT(tmp_path))


def test_setup_pins_the_cpu_build_in_a_uv_tool(tmp_path, monkeypatch, clap_env):
    """uv's receipt keeps no --torch-backend, so once a newer PyTorch is out `uv tool upgrade`
    would bring the CUDA build: after the install, setup pins the CPU build it got (the exact
    version and PyTorch's CPU index, both kept in the receipt)."""
    _tool_receipt(tmp_path, monkeypatch)
    _which(monkeypatch, uv="/bin/uv")
    monkeypatch.setattr(S, "cpu_only_torch", lambda: True)
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": True)
    monkeypatch.setattr(S, "installed_torch", lambda exe=None: "2.5.1+cpu")
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert res.exit_code == 0, res.output
    first, pin = clap_env["ran"]
    assert first[6:8] == ["--torch-backend", "cpu"] and "torch>=2.2" in first
    assert pin[6:10] == ["--index", S.TORCH_CPU_INDEX, "--index-strategy", "unsafe-best-match"]
    assert "torch==2.5.1+cpu" in pin and "torch>=2.2" not in pin and "--torch-backend" not in pin
    assert pin[-1] == "fourier-samples" and pin[pin.index("--with") + 1:].count("transformers>=4.40") == 1
    assert "Pinning PyTorch at 2.5.1+cpu (the CPU build), so `uv tool upgrade` keeps it." in out
    assert "Pinned: uv's receipt names torch==2.5.1+cpu and PyTorch's CPU index." in out
    assert clap_env["quiet"] == [False, True]          # the second reinstall shows no package list
    assert "The CLAP model is ready." in out
    # a default build (a GPU machine), or a pin already in place: nothing more to do
    for installed in ("2.5.1", None):
        clap_env.update(installed=False, ran=[])
        monkeypatch.setattr(S, "installed_torch", lambda exe=None, v=installed: v)
        _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"))
        assert len(clap_env["ran"]) == 1


def test_setup_again_updates_a_pinned_pytorch(tmp_path, monkeypatch, clap_env):
    """Installed and pinned: `fourier setup` asks whether to update PyTorch (--yes doesn't,
    --clap does), then pins the new version."""
    _tool_receipt(tmp_path, monkeypatch, '{ name = "torch", specifier = "==2.5.1+cpu" }',
                  '{ name = "transformers", specifier = ">=4.40" }')
    clap_env["installed"] = True
    monkeypatch.setattr(S, "clap_model_cached", lambda: True)
    _which(monkeypatch, uv="/bin/uv")
    monkeypatch.setattr(S, "cpu_only_torch", lambda: True)
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": True)
    monkeypatch.setattr(S, "installed_torch", lambda exe=None: "2.6.0+cpu")
    cfg = tmp_path / "fourier.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert "Installed; PyTorch is pinned at 2.5.1+cpu (the CPU build), so `uv tool upgrade` keeps it." in out
    assert clap_env["ran"] == [] and clap_env["loaded"] == 0
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--clap")
    out = _flat(res.output)
    assert res.exit_code == 0, res.output
    update, pin = clap_env["ran"]
    assert "torch>=2.2" in update and "torch==2.5.1+cpu" not in update and "--torch-backend" in update
    assert "torch==2.6.0+cpu" in pin and "Pinning PyTorch at 2.6.0+cpu" in out
    assert clap_env["downloaded"] == 0 and clap_env["loaded"] == 1 and "The CLAP model is ready." in out
    # asked, and declined
    clap_env["ran"] = []
    res = _run(tmp_path, monkeypatch, "setup", "--to", str(cfg), "--no-llm", "--no-build",
               input="y\nn\n")
    assert "Update PyTorch and Transformers to their newest now?" in res.output and clap_env["ran"] == []


def test_setup_switches_a_cuda_build_to_the_cpu_build(tmp_path, monkeypatch, clap_env):
    """A CUDA build of PyTorch on Linux without an NVIDIA GPU (doctor WARNs): setup reinstalls
    PyTorch's CPU build."""
    clap_env["installed"] = True
    monkeypatch.setattr(S, "clap_model_cached", lambda: True)
    monkeypatch.setattr(S, "torch_has_cuda", lambda: True)
    monkeypatch.setattr(S, "cpu_only_torch", lambda: True)
    _which(monkeypatch, uv="/bin/uv")
    monkeypatch.setattr(S, "uv_has_torch_backend", lambda uv, group="pip": True)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert res.exit_code == 0, res.output
    assert "PyTorch here is a CUDA build" in out and "The CLAP model is ready." in out
    assert clap_env["ran"] == [["/bin/uv", "pip", "install", "--python", PY, "--torch-backend", "cpu",
                                "--reinstall-package", "torch", *S.clap_requirements()]]
    _which(monkeypatch)
    monkeypatch.setattr(S, "has_pip", lambda: True)
    assert S.clap_install(PY, reinstall_torch=True)[1][0] == [
        PY, "-m", "pip", "install", "--index-url", S.TORCH_CPU_INDEX, "--force-reinstall", "torch>=2.2"]


def test_doctor_says_a_reinstall_removed_clap(monkeypatch, tmp_path):
    """`uv tool install --reinstall` with only the package drops the --with requirements setup
    gave it: doctor names that, and the one command that puts them back."""
    _tool_receipt(tmp_path, monkeypatch)
    assert S.check_clap(None)[0] == (S.FAIL, "CLAP model", "not installed: `fourier setup` installs it "
                                                           "(a build needs it)")
    S.save_install("uv tool", "cpu")
    assert S.check_clap(None)[0] == (S.FAIL, "CLAP model", "CLAP was removed by a reinstall: run `fourier setup`")
    _tool_receipt(tmp_path, monkeypatch, '{ name = "torch", specifier = ">=2.2" }')
    assert "not installed" in S.check_clap(None)[0][2]


def test_torch_pin_reads_the_receipt():
    assert S.torch_pin({"requirements": [{"name": "torch", "specifier": "==2.5.1+cpu"}]}) == "2.5.1+cpu"
    assert S.torch_pin({"requirements": [{"name": "torch", "specifier": ">=2.2"}]}) is None
    assert S.torch_pin({"requirements": [{"name": "fourier-samples"}]}) is None and S.torch_pin(None) is None


def test_doctor_warns_about_a_cuda_build_without_a_gpu(monkeypatch):
    monkeypatch.setattr(E, "clap_extra_missing", lambda: False)
    monkeypatch.setattr(S, "clap_model_cached", lambda: True)
    monkeypatch.setattr(S, "cpu_only_torch", lambda: True)
    monkeypatch.setattr(S, "torch_has_cuda", lambda: True)
    rows = S.check_clap(None)
    assert (S.WARN, "PyTorch build", "a CUDA build, with NVIDIA libraries this machine can't use (no "
            "NVIDIA GPU): run `fourier setup` to switch to the smaller CPU build") in rows
    monkeypatch.setattr(S, "cpu_only_torch", lambda: False)             # a GPU: it's the right one
    assert not [r for r in S.check_clap(None) if r[1] == "PyTorch build"]


def test_torch_has_cuda_reads_the_installed_packages(monkeypatch):
    class D:
        def __init__(self, name):
            self.metadata = {"Name": name}
    md = S.importlib.metadata
    for version, dists, cuda in (("2.5.1+cpu", ["torch"], False), ("2.5.1", ["torch", "nvidia-cublas-cu12"], True),
                                 ("2.5.1+cu121", ["torch"], True), ("2.5.1", ["torch"], False)):
        monkeypatch.setattr(md, "version", lambda name, v=version: v)
        monkeypatch.setattr(md, "distributions", lambda d=dists: [D(n) for n in d])
        assert REAL_TORCH_HAS_CUDA() is cuda, (version, dists)
    monkeypatch.setattr(md, "version", lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name)))
    assert REAL_TORCH_HAS_CUDA() is False



def test_setup_starts_the_first_build_when_asked(tmp_path, monkeypatch, clap_env):
    clap_env["installed"] = True
    monkeypatch.setattr(S, "clap_model_cached", lambda: True)
    monkeypatch.setattr(S, "doctor_rows", lambda: [(S.OK, "config", "x"),
                                                   (S.NEXT, "samples in the database", "none")])
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--build", "--to", str(tmp_path / "fourier.toml"),
               "--library", str(_lib(tmp_path)))
    assert res.exit_code == 0, res.output
    (cmd,) = clap_env["ran"]
    assert cmd[:3] == [PY, "-m", "fourier"] and cmd[3:7] == ["--db", str(tmp_path / "t.duckdb"), "build", "--all"]
    assert "analyzes 3 new samples" in _flat(res.output)
    # asked, a small library defaults to yes
    clap_env["ran"].clear()
    res = _run(tmp_path, monkeypatch, "setup", "--to", str(tmp_path / "fourier.toml"), input="y\n\n")
    assert res.exit_code == 0 and "Start it now? [Y/n]" in res.output and len(clap_env["ran"]) == 1, res.output


# --- the local LLM ------------------------------------------------------------------------
def test_setup_llm_with_ollama_turns_descriptions_on(tmp_path, monkeypatch):
    cfg = tmp_path / "fourier.toml"
    pulled = []
    monkeypatch.setattr(S, "ollama_status", lambda: (True, True, []))
    monkeypatch.setattr(S, "describe_model", lambda: "tiny:1b")
    monkeypatch.setattr(S, "ollama_pull", lambda model, progress=None: pulled.append(model) or
                        (progress and progress(5, 10, "pulling")))
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--llm", "--no-clap", "--to", str(cfg),
               "--library", str(_lib(tmp_path)))
    assert res.exit_code == 0, res.output
    assert pulled == ["tiny:1b"] and tomllib.loads(cfg.read_text())["advanced"] == {"DESCRIBE": True}
    assert "Folder descriptions are on" in res.output
    # the model there already: no pull; asked, the default is no
    pulled.clear()
    monkeypatch.setattr(S, "ollama_status", lambda: (True, True, ["tiny:1b"]))
    res = _run(tmp_path, monkeypatch, "setup", "--to", str(cfg), "--no-clap", input="y\n\n")
    assert res.exit_code == 0 and "[y/N]" in res.output and "Skipped." in res.output and pulled == []


def test_setup_llm_installed_but_not_running(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "ollama_status", lambda: (True, False, []))
    monkeypatch.setattr(S, "describe_model", lambda: "tiny:1b")
    cfg = tmp_path / "fourier.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--llm", "--no-clap", "--to", str(cfg),
               "--library", str(_lib(tmp_path)))
    assert res.exit_code == 0 and "not running" in res.output and "ollama pull tiny:1b" in _flat(res.output)
    assert "advanced" not in tomllib.loads(cfg.read_text())


def test_ollama_status(monkeypatch):
    import io
    import json

    from fourier import net

    class Opener:
        def __init__(self, models=None):
            self.models = models

        def open(self, url, timeout=None):
            assert url == "http://localhost:11434/api/tags"
            if self.models is None:
                raise ConnectionRefusedError("refused")
            return io.BytesIO(json.dumps({"models": [{"name": m} for m in self.models]}).encode())
    _which(monkeypatch)
    monkeypatch.setattr(net, "local_opener", lambda: Opener())
    assert REAL_OLLAMA_STATUS() == (False, False, [])
    _which(monkeypatch, ollama="/bin/ollama")
    assert REAL_OLLAMA_STATUS() == (True, False, [])
    monkeypatch.setattr(net, "local_opener", lambda: Opener(["tiny:1b"]))
    assert REAL_OLLAMA_STATUS() == (True, True, ["tiny:1b"])


def test_turn_on_describe_keeps_the_rest(tmp_path):
    p = tmp_path / "f.toml"
    p.write_text('library = ["/x"]\n[advanced]\nCLAP_Z = 1.5\n')
    S.turn_on_describe(p)
    assert tomllib.loads(p.read_text()) == {"library": ["/x"], "advanced": {"DESCRIBE": True, "CLAP_Z": 1.5}}
    p.write_text('library = ["/x"]\n[advanced]\nDESCRIBE = false\n')
    S.turn_on_describe(p)
    assert tomllib.loads(p.read_text())["advanced"] == {"DESCRIBE": True}
    p.write_text('library = ["/x"]\n[output]\nmaster = "/m"')
    S.turn_on_describe(p)
    assert tomllib.loads(p.read_text()) == {"library": ["/x"], "output": {"master": "/m"},
                                            "advanced": {"DESCRIBE": True}}


# --- doctor ------------------------------------------------------------------------------
def test_doctor_passes_a_fresh_install_and_fails_what_blocks_a_build(tmp_path, monkeypatch):
    lib = _lib(tmp_path)
    monkeypatch.setattr(E, "clap_extra_missing", lambda: False)        # a correct, fresh install
    monkeypatch.setattr(S, "clap_model_cached", lambda: True)
    cfg = tmp_path / "fourier.toml"
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "out" / "master"))
    cfg.write_text(f'library = ["{lib}"]\ndevices = ["generic_48k"]\npreset = "balanced"\n'
                   f'[output]\npublish = "{tmp_path}/pub"\n')
    res = _run(tmp_path, monkeypatch, "doctor")
    out = _flat(res.output)
    # what the first build does itself is NEXT, not a failure: doctor passes
    assert res.exit_code == 0, res.output
    assert "OK    config:" in res.output and "OK    library folder" in res.output
    assert "style balanced" in out and "tunables differ" not in out
    # a new machine: doctor says there's no database yet, and doesn't create one
    assert "NEXT samples in the database: none yet" in out and "NEXT CLAP index: none yet" in out
    assert not (tmp_path / "t.duckdb").exists() and "FAIL" not in out
    assert "NEXT lines are what the first `fourier build` does itself" in out
    # an estimate from the file count, not "can't estimate"
    assert "OK build time: about 2 min for the first" in out and "analyzes 3 new samples first" in out
    assert "can't estimate" not in out
    # a generic profile is unverified by design: for information, not a warning
    assert "INFO  device: generic_48k" in res.output and "unverified by design" in out
    # three files in one pack: a small library, and what to expect
    assert "WARN library size: a small library (3 audio files in 1 pack folder)" in out
    assert "the master scales down to it" in out and "names the categories it leaves empty" in out
    assert "OK master size: Your library has 3 audio files: the master will hold up to about" in out
    # the model not downloaded yet: NEXT too
    monkeypatch.setattr(S, "clap_model_cached", lambda: False)
    res = _run(tmp_path, monkeypatch, "doctor")
    assert res.exit_code == 0 and "NEXT CLAP model: the software is installed; the model" in _flat(res.output)


def test_doctor_fails_what_blocks_a_build(tmp_path, monkeypatch):
    lib = tmp_path / "Samples"
    lib.mkdir()
    cfg = tmp_path / "fourier.toml"
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(tmp_path / "out" / "master"))
    cfg.write_text(f'library = ["{lib}"]\ndevices = ["generic_48k"]\n')
    res = _run(tmp_path, monkeypatch, "doctor")       # no CLAP install, no audio files
    out = _flat(res.output)
    assert res.exit_code == 1, res.output
    assert "FAIL CLAP model: not installed" in out and "FAIL audio files: none in the library" in out
    cfg.write_text(f'library = ["{tmp_path}/gone"]\ndevices = ["generic_48k"]\nmystery = 1\n')
    res = _run(tmp_path, monkeypatch, "doctor")
    assert res.exit_code == 1
    assert "FAIL  library folder" in res.output and "not keys fourier reads: mystery" in res.output
    # a folder it can't read
    cfg.write_text(f'library = ["{_lib(tmp_path)}"]\ndevices = ["generic_48k"]\n')
    real = S.os.access
    monkeypatch.setattr(S.os, "access", lambda p, mode: False if str(p).endswith("Samples") else real(p, mode))
    res = _run(tmp_path, monkeypatch, "doctor")
    assert res.exit_code == 1 and "can't be read" in _flat(res.output)


# --- messages and warnings ----------------------------------------------------------------
def test_no_message_says_pip_install():
    """What to install is `fourier setup`'s job (or a reinstall, for a library every install
    has): no message tells a user to pip install something. setup.py builds the installer's
    own commands."""
    root = S.Path(__file__).parents[1] / "src" / "fourier"
    hits = [f"{p.relative_to(root)}:{i}" for p in root.rglob("*.py") if p.name != "setup.py"
            for i, line in enumerate(p.read_text().splitlines(), 1) if "pip install" in line]
    assert not hits, hits


@pytest.mark.parametrize("verbose", [False, True])
def test_librosa_and_numba_warnings_stay_out_unless_verbose(verbose):
    import os
    import subprocess
    code = ("import fourier, warnings\n"
            "warnings.warn_explicit('n_fft=1024 is too large', UserWarning, 'spectrum.py', 1, "
            "module='librosa.core.spectrum')\n"
            "warnings.warn_explicit('reflected list', Warning, 'core.py', 1, module='numba.core')\n"
            "warnings.warn_explicit('a real one', UserWarning, 'x.py', 1, module='fourier.x')\n")
    env = {k: v for k, v in os.environ.items() if k != "FOURIER_VERBOSE"}
    if verbose:
        env["FOURIER_VERBOSE"] = "1"
    err = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env).stderr
    assert "a real one" in err
    assert ("n_fft" in err and "reflected list" in err) == verbose, err


def test_the_pin_runs_quietly_and_shows_its_output_only_when_it_fails(monkeypatch, capsys):
    """The pin is a second reinstall of the same packages: its output is kept back, a line
    says what it did, and the whole output shows only when it fails."""
    receipt = {"requirements": [{"name": "fourier-samples"}]}
    monkeypatch.setattr(S, "cpu_only_torch", lambda: True)
    monkeypatch.setattr(S, "installed_torch", lambda exe=None: "2.5.1+cpu")
    monkeypatch.setattr(S, "uv_tool_receipt", lambda prefix=None: receipt)
    _which(monkeypatch, uv="/bin/uv")
    lines = ["Resolved 40 packages", " + torch==2.5.1+cpu",
             "warning: The `--torch-backend` option is experimental and may change without warning."]
    got = {}

    def run(cmd, keep=None, quiet=False):
        got["quiet"] = quiet
        keep.extend(S.shown_lines(lines) + (["error: no space left"] if got.get("fail") else []))
        return 1 if got.get("fail") else 0
    monkeypatch.setattr(S, "_run", run)
    S.pin_cpu_torch("uv tool")
    out = _flat(capsys.readouterr().out)
    assert got["quiet"] and "Pinned: uv's receipt names torch==2.5.1+cpu" in out and "Resolved 40" not in out
    got["fail"] = True
    S.pin_cpu_torch("uv tool")
    out = _flat(capsys.readouterr().out)
    assert "Resolved 40 packages" in out and "+ torch==2.5.1+cpu" in out and "experimental" not in out
    assert "PyTorch isn't pinned (the CLAP install itself is fine): error: no space left" in out


def test_run_hides_uvs_preview_notice(tmp_path, capfd):
    """uv's "`--torch-backend` option is experimental" notice is left out of what the user
    sees (and of the lines kept for a summary)."""
    script = tmp_path / "noisy.py"
    script.write_text("print('Resolved 3 packages')\n"
                      "print('warning: The `--torch-backend` option is experimental and may change "
                      "without warning. Pass `--preview-features torch-backend` to disable this warning.')\n"
                      "print('Installed 3 packages')\n")
    keep = []
    assert S._run([S.sys.executable, str(script)], keep=keep) == 0
    out = capfd.readouterr().out
    assert "Resolved 3 packages" in out and "Installed 3 packages" in out and "experimental" not in out
    assert keep == ["Resolved 3 packages", "Installed 3 packages"]
    keep = []
    assert S._run([S.sys.executable, str(script)], keep=keep, quiet=True) == 0
    assert capfd.readouterr().out == "" and len(keep) == 2


@pytest.mark.parametrize("platform, nvidia, says", [
    ("darwin", False, "PyTorch is about 200 MB (about 750 MB installed)"),
    ("linux", False, "PyTorch's CPU build is about 200 MB (about 750 MB installed)"),
    ("linux", True, "PyTorch with CUDA (for the NVIDIA GPU here) is several GB"),
    ("win32", True, "PyTorch is about 200 MB (about 750 MB installed)"),
])
def test_torch_size_per_platform(monkeypatch, platform, nvidia, says):
    monkeypatch.setattr(S.sys, "platform", platform)
    _which(monkeypatch, **({"nvidia_smi": "/bin/nvidia-smi"} if nvidia else {}))
    assert S.torch_size() == says


def test_check_lines_never_break_mid_path(monkeypatch, capsys):
    """Doctor's and setup's check lines aren't hard-wrapped: a long path stays whole in narrow
    or piped output."""
    path = "/Users/someone/Library/Mobile Documents/com~apple~CloudDocs/Music/SampleLibrary/Some Vendor/Pack"
    monkeypatch.setattr(S.console, "width", 40)
    S.print_rows([(S.OK, "library", f"1,234 audio files in {path}")])
    assert path in capsys.readouterr().out


# --- setup --to, unreadable files, the layout line, a small library's words ---------------
def test_setup_to_checks_the_config_it_wrote(tmp_path, monkeypatch):
    lib = _lib(tmp_path)
    other = tmp_path / "Elsewhere"
    other.mkdir()
    env_cfg = tmp_path / "fourier.toml"                      # the config the run starts with
    env_cfg.write_text(f'library = ["{other}"]\ndevices = ["generic_48k"]\n')
    target = tmp_path / "new" / "fourier.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(target), "--library", str(lib),
               "--device", "generic_48k", "--no-clap", config=env_cfg)
    assert res.exit_code == 0 and target.exists(), res.output
    out = _flat(res.output)
    assert f"library folder: {lib}" in out and f"library folder: {other}" not in out
    assert f"config: {target}" in out


def _wav(p, seconds=0.2):
    import numpy as np
    import soundfile as sf
    p.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(44100 * seconds)) / 44100
    sf.write(p, (0.5 * np.sin(2 * np.pi * 110 * t) * np.exp(-t / 0.05)).astype("float32"), 44100)


def test_unreadable_files_are_errors_not_files_waiting_for_analysis(tmp_path, monkeypatch):
    lib = tmp_path / "SampleLibrary"
    for i in range(3):
        _wav(lib / "Kicks" / f"Kick {i}.wav")
    (lib / "Kicks" / "Kick Empty.wav").write_bytes(b"")
    (lib / "Kicks" / "Kick Broken.wav").write_bytes(b"RIFF....WAVEjunk")
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(f'library = ["{lib}"]\ndevices = ["generic_48k"]\npreset = "balanced"\n')
    monkeypatch.setattr(E, "clap_extra_missing", lambda: False)
    monkeypatch.setattr(S, "clap_model_cached", lambda: True)
    res = _run(tmp_path, monkeypatch, "tools", "scan")
    assert res.exit_code == 0 and "2 file(s) couldn't be read" in res.output, res.output
    from fourier.ingest.importer import scan_errors
    assert {p.rsplit("/", 1)[-1] for p in scan_errors([lib])} == {"Kick Empty.wav", "Kick Broken.wav"}
    out = _flat(_run(tmp_path, monkeypatch, "doctor").output)
    assert "WARN unreadable files: 2 audio files the last scan couldn't read" in out
    assert "Kick Empty.wav" in out
    # the estimate counts the three readable files as the ones to analyze, not five
    assert "analyzes 3 new samples first" in out
    # a second scan doesn't read them again while they're unchanged, and still counts them
    from fourier.ingest import importer as I
    read = []
    real = I.file_info
    monkeypatch.setattr(I, "file_info", lambda p, *a, **k: (read.append(os.path.basename(str(p))), real(p, *a, **k))[1])
    res = _run(tmp_path, monkeypatch, "tools", "scan")
    assert "2 file(s) couldn't be read" in res.output and not {"Kick Empty.wav", "Kick Broken.wav"} & set(read)
    monkeypatch.setattr(I, "file_info", real)
    # fixed: no longer listed
    _wav(lib / "Kicks" / "Kick Empty.wav")
    (lib / "Kicks" / "Kick Broken.wav").unlink()
    _run(tmp_path, monkeypatch, "tools", "scan")
    assert scan_errors([lib]) == {}


def test_the_layout_line_is_the_same_before_and_after_the_scan(tmp_path, monkeypatch):
    lib = tmp_path / "SampleLibrary"
    for kind in ("Kicks", "Snares", "Hats"):
        for i in range(2):
            _wav(lib / kind / f"{kind[:-1]} {i}.wav")
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(f'library = ["{lib}"]\ndevices = ["generic_48k"]\n')
    from fourier.packs import curate_config as cc
    monkeypatch.setattr(cc, "VENDORS", "auto")             # vendors = "auto", as setup writes
    line = lambda out: next(x for x in out.splitlines() if "library layout" in x)
    before = line(_run(tmp_path, monkeypatch, "doctor").output)
    assert "folders by sound type" in before
    _run(tmp_path, monkeypatch, "tools", "scan")
    assert line(_run(tmp_path, monkeypatch, "doctor").output) == before
    # a small library by sound type is counted in its own words, not "pack folders"
    out = _flat(_run(tmp_path, monkeypatch, "doctor").output)
    assert "a small library (6 audio files in 3 sound-type folders)" in out


def test_a_dry_run_before_the_first_scan_sizes_from_the_files(tmp_path, monkeypatch):
    lib = _lib(tmp_path, n=5)
    cfg = tmp_path / "fourier.toml"
    cfg.write_text(f'library = ["{lib}"]\ndevices = ["generic_48k"]\npreset = "balanced"\n')
    out = _flat(_run(tmp_path, monkeypatch, "build", "--dry-run").output)
    assert "Your library has 5 audio files: the master will hold up to about" in out
    assert "from the files in the library folders" in out and "11,388" not in out
    empty = tmp_path / "Empty"
    empty.mkdir()
    cfg.write_text(f'library = ["{empty}"]\ndevices = ["generic_48k"]\npreset = "balanced"\n')
    monkeypatch.delenv("FOURIER_RESOLVED_CONFIG", raising=False)   # (the dry run above removed its own)
    out = _flat(_run(tmp_path, monkeypatch, "build", "--dry-run").output)
    assert "Can't size the master before the first scan" in out


def test_why_limit_help_and_rated_when():
    from fourier.packs.why import _rated_when
    res = CliRunner().invoke(main, ["why", "--help"])
    flat = _flat(res.output)
    assert "(default 5)" in flat and "--unrecognized (default 20)" in flat
    assert _rated_when({"rated_at": "2026-01-02T03:04:05+00:00", "source": "cli"}) == \
        " (on 2026-01-02, by fourier review rate)"
    assert _rated_when({"source": "tag"}) == " (by a Live tag)"


def test_setup_says_the_first_build_trains_a_sound_model_and_can_turn_it_off(tmp_path, monkeypatch):
    """Setup says what the first build does with CLAP (without Sononym, a sound model trained
    on the library's own names, on this machine); --no-sound-model writes SOUND_TRAIN = false
    to [advanced], and --sound-model turns it back on."""
    import tomllib
    cfg = tmp_path / "fourier.toml"
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--library", str(_lib(tmp_path)))
    out = _flat(res.output)
    assert res.exit_code == 0, res.output
    assert "the first build also trains a sound model" in out and "It stays on this machine" in out
    assert "SOUND_TRAIN" not in cfg.read_text()
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--no-sound-model")
    assert res.exit_code == 0, res.output
    assert "The sound model is off: SOUND_TRAIN = false in" in _flat(res.output)
    assert tomllib.loads(cfg.read_text())["advanced"]["SOUND_TRAIN"] is False
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg))
    assert "The sound model is off (SOUND_TRAIN = false" in _flat(res.output)
    res = _run(tmp_path, monkeypatch, "setup", "--yes", "--to", str(cfg), "--sound-model")
    assert tomllib.loads(cfg.read_text())["advanced"]["SOUND_TRAIN"] is True
    assert "the first build also trains a sound model" in _flat(res.output)
