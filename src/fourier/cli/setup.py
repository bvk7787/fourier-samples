"""fourier setup (a first-run wizard that writes the fourier.toml), fourier demo and fourier
doctor (check what a build needs)."""
from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import click

from ._app import console, main

# doctor's levels: NEXT is what the first `fourier build` does itself (scan, analyze, index,
# download the model); only FAIL stops a build
OK, WARN, NEXT, FAIL, INFO = "OK", "WARN", "NEXT", "FAIL", "INFO"
LEVEL_STYLE = {OK: "green", WARN: "yellow", NEXT: "cyan", FAIL: "red", INFO: "blue"}


# --- demo --------------------------------------------------------------------------------
@main.command("demo", short_help="Try Fourier Samples on a generated library, in a sandbox folder.")
@click.option("--dir", "root", default=None, type=click.Path(file_okay=False),
              help="The sandbox folder: new, empty, or an earlier demo's (default: ./fourier-demo)")
def demo(root):
    """A first run that can't touch anything of yours: generate a small synthetic library,
    analyze it with stand-in data, build the master and render it for the Digitakt 2 and the
    M8, all inside one folder with its own Fourier home and config. Needs no samples, no
    model and no download; takes a minute or two.

    \b
      fourier demo
      fourier demo --dir /tmp/fourier-demo
    """
    from .. import demo as D
    try:
        r = D.run(root or D.DEFAULT_DIR, say=click.echo)
    except D.DemoError as e:
        click.secho(str(e), fg="red", err=True)
        raise SystemExit(1) from None
    for line in D.summary(r):
        click.echo(line)


# --- the config setup writes -------------------------------------------------------------
OUTPUTS = ("master", "renders", "publish")


def presets() -> list[str]:
    """The shipped styles, balanced (the default) first."""
    from ..layers import DEFAULT_PRESET, PRESETS_DIR
    names = sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))
    return sorted(names, key=lambda n: n != DEFAULT_PRESET)


def preset_line(name: str) -> str:
    """One line saying what a preset is for: its `description`, else its first comment."""
    from ..layers import PRESETS_DIR, _preset_path, _read
    try:
        p = _preset_path(name, None)
        doc = _read(p)
    except Exception:
        return ""
    if isinstance(doc, dict) and isinstance(doc.get("description"), str):
        return doc["description"].strip()
    try:
        first = next(ln for ln in p.read_text().splitlines() if ln.startswith("#"))
    except (OSError, StopIteration):
        return ""
    return first.lstrip("# ").split(": ", 1)[-1].split(". ")[0] if p.parent == PRESETS_DIR else ""


def default_outputs(target: Path | None = None) -> dict[str, str]:
    """Where the master, renders and releases go when the config doesn't say. For a config
    setup writes somewhere else than the default config (`--to`: a second library), folders
    of its own, named after it (FourierCurated-<name>, ...): two libraries never share a
    master."""
    from ..places import DEFAULT_MASTER, DEFAULT_PUBLISH
    master = os.path.expanduser(DEFAULT_MASTER)
    publish = os.path.expanduser(DEFAULT_PUBLISH)
    renders = os.path.join(os.path.dirname(master), "FourierRenders")
    tag = config_tag(target) if target is not None else None
    if tag:
        master, renders, publish = f"{master}-{tag}", f"{renders}-{tag}", f"{publish}-{tag}"
    return {"master": master, "renders": renders, "publish": publish}


def config_tag(target: Path) -> str | None:
    """The name a config other than the default one gives its own output folders: its file
    name without the extension, or its folder's name for a file named fourier.toml; None for
    the default config (~/.config/fourier/fourier.toml)."""
    from ..layers import USER_CONFIG
    t = Path(target).expanduser()
    try:
        if t.resolve() == USER_CONFIG.expanduser().resolve():
            return None
    except OSError:
        pass
    name = t.stem if t.stem.lower() not in ("fourier", "") else t.resolve().parent.name
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return tag or "other"


def config_text(library: list[str], devices: list[str], preset: str,
                output: dict | None = None, describe: bool = False,
                extra: list[str] | None = None) -> str:
    q = lambda xs: "[" + ", ".join(json.dumps(x) for x in xs) + "]"
    text = (f"# Fourier config, written by fourier setup on {date.today().isoformat()}.\n"
            "# fourier config show prints what it resolves to; fourier doctor checks it.\n"
            f"library = {q(library)}     # where your samples are (read, never written)\n"
            f"devices = {q(devices)}     # fourier devices list; fourier devices show <id>\n"
            f'preset = "{preset}"        # the style: {", ".join(presets())}, or a file\n'
            'vendors = "auto"         # a library\'s layout decides its vendors (first-folder: vendor/pack/...)\n'
            'fold = "auto"            # half- and double-time loops fold into the style\'s tempo range (or "off")\n')
    text += "".join(ln + "\n" for ln in extra or ())
    if output:
        text += "\n[output]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in output.items())
    if describe:
        text += "\n[advanced]\n" + DESCRIBE_LINE
    return text


DESCRIBE_LINE = "DESCRIBE = true     # folder descriptions from a local LLM (Ollama)\n"


def turn_on_describe(path: Path) -> None:
    """DESCRIBE = true in the config's [advanced] table (made when it has none)."""
    set_advanced(path, "DESCRIBE", True, DESCRIBE_LINE)


SOUND_OFF_LINE = "SOUND_TRAIN = false   # no sound model trained on this library\n"


def set_advanced(path: Path, key: str, value, line: str) -> None:
    """`key` = value in the config's [advanced] table (made when it has none), as `line`."""
    import tomllib
    text = path.read_text()
    adv = tomllib.loads(text).get("advanced") or {}
    if key in adv and adv[key] == value and type(adv[key]) is type(value):
        return
    lines = text.splitlines(keepends=True)
    if key in adv:
        lines = [line if re.match(rf"\s*{key}\s*=", ln) else ln for ln in lines]
    elif any(ln.strip() == "[advanced]" for ln in lines):
        at = next(i for i, ln in enumerate(lines) if ln.strip() == "[advanced]")
        lines.insert(at + 1, line)
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ["\n[advanced]\n", line]
    path.write_text("".join(lines))


def _split(values) -> list[str]:
    return [v.strip() for x in values for v in str(x).split(",") if v.strip()]


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    return s


def _unescape(s: str) -> str:
    """A path as a terminal pastes a folder dragged into it (macOS Terminal: a backslash
    before each space and bracket, and a space after), as it is on disk."""
    return s if os.name == "nt" else re.sub(r"\\(.)", r"\1", s).strip()


def folder_answers(values) -> list[str]:
    """The folders in setup's answers or --library values, as typed, pasted or dragged in
    from Finder: a whole answer that is a folder (quoted, backslash-escaped, with a comma in
    its name) is one folder; else several dragged in at once, or a comma-separated list."""
    out: list[str] = []
    for raw in values:
        raw = str(raw or "").strip()
        if raw:
            out += _one_answer(raw)
    return out


def _one_answer(raw: str) -> list[str]:
    def is_dir(p):
        return bool(p) and Path(p).expanduser().is_dir()
    for cand in (raw, _unquote(raw), _unescape(_unquote(raw))):
        if is_dir(cand):
            return [cand]
    if os.name != "nt":
        try:
            parts = shlex.split(raw)
        except ValueError:
            parts = []
        if len(parts) > 1 and all(is_dir(p) for p in parts):
            return parts
    pieces = raw.split(",") if "," in raw else [raw]
    return [q for q in (_unescape(_unquote(x)) for x in pieces) if q]


# setup's size choice: (name, what it writes, what it says)
SIZES = (
    ("starter", 'size = "1GB"', "about 1,500 sounds, about 1 GB: quick to load, the best of each kind"),
    ("standard", 'size = "3GB"', "about 4,500 sounds, about 3 GB"),
    ("full", None, "the style's full set: about 8,000 to 11,500 sounds, 5 to 7 GB (fewer from a "
                   "smaller library)"),
)
GB_PER_TRANSFER_HOUR = 2            # Elektron Transfer over USB, roughly (for the estimate only)
AS_IS = ('retune = "off"           # melodic one-shots keep their own pitch',
         'loudness = "gentle"      # loops aren\'t pushed louder',
         'stereo = "keep"          # every file keeps its channels',
         'names = "keep"           # the original file names')


def size_line(answer: str | None) -> tuple[str | None, str]:
    """(the line setup writes, or None for the style's own size; the name) for a size
    answer: a number in the list, starter / standard / full, or a card size ("4GB")."""
    a = (answer or "").strip().lower()
    if not a or a in ("3", "full"):
        return None, "full"
    for i, (name, line, _what) in enumerate(SIZES, 1):
        if a in (str(i), name):
            return line, name
    if re.fullmatch(r"\d+(\.\d+)?\s*(gb|mb)", a):
        return f'size = "{a.replace(" ", "").upper()}"', a.upper()
    raise ValueError(a)


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def pick_devices(values, known: list[str], names: dict) -> tuple[list[str], list[str]]:
    """(device ids, answers that match none) from answers given as a number in setup's
    list, an id (digitakt_2) or a name ("Digitakt 2", "Elektron Digitakt 2", "m8")."""
    ids, bad = [], []
    for a in _split(values):
        if a.isdigit() and 1 <= int(a) <= len(known):
            ids.append(known[int(a) - 1])
            continue
        if a in known:
            ids.append(a)
            continue
        k = _key(a)
        hits = [d for d in known if k and (k == _key(d) or k == _key(names.get(d, ""))
                                           or _key(names.get(d, "")).endswith(k)
                                           or _key(d).startswith(k))]
        if len(hits) == 1:
            ids.append(hits[0])
        else:
            bad.append(a)
    return list(dict.fromkeys(ids)), bad


# --- the library -------------------------------------------------------------------------
COUNT_CAP = 100_000      # audio files setup counts before it says "more than"


def survey(roots, cap: int, packs: set | None = None, rels: list | None = None) -> tuple[int, int]:
    """(audio files, how many of them a cloud drive keeps only online) under roots, the
    first `cap` audio files at most (one lstat each), as the library walk finds them
    (ingest/walk.py): the readable formats (ingest/formats.py), dot files and folders left
    out, a folder symlink followed when it leads out of the library folder (each real folder
    once). `packs`, a set, gets each file's vendor/pack folders (the first two under its
    root); `rels`, a list, each file's path under its root ("/"-separated)."""
    from .. import platforms
    from ..ingest.formats import AUDIO_EXTS, readable_exts
    exts = tuple(readable_exts() or AUDIO_EXTS)
    stack = []
    for r in roots:
        if os.path.isdir(r):
            stack.append((str(r), os.path.realpath(str(r)), str(r), ""))
    seen = cloud = 0
    visited: set = set()
    while stack and seen < cap:
        root, real_root, here, rel = stack.pop()
        try:
            st = os.stat(here)
            ident = (st.st_dev, st.st_ino)
            if ident in visited and (st.st_dev or st.st_ino):
                continue
            visited.add(ident)
            with os.scandir(here) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        sub = rel + "/" if rel else ""
        for e in entries:
            if e.name.startswith(".") or seen >= cap:
                continue
            try:
                if e.is_symlink() and e.is_dir():
                    target = os.path.realpath(e.path)
                    if target == real_root or target.startswith(real_root.rstrip(os.sep) + os.sep):
                        continue                # walked at its target, inside the library
                    stack.append((root, real_root, e.path, sub + e.name))
                elif e.is_dir(follow_symlinks=False):
                    stack.append((root, real_root, e.path, sub + e.name))
                elif e.name.lower().endswith(exts) and e.is_file():
                    seen += 1
                    cloud += platforms.cloud_only(e.path)
                    if packs is not None:
                        packs.add(tuple(rel.split("/")[:2]) if rel else (".",))
                    if rels is not None:
                        rels.append(sub + e.name)
            except OSError:
                continue
    return seen, cloud


CLOUD_HINT = ('Finder: "Keep Downloaded" on the library folder; OneDrive: "Always keep on this '
              'device"')
# a library this small (files, or packs) fills only some categories: each needs at least a
# folder's minimum of usable samples (curate.MIN_PER_FAMILY, or SCALED_MIN_FILES when the
# master scales to the library), and one or two packs rarely cover more than a few kinds of
# sound
THIN_FILES, THIN_PACKS = 300, 3


def _folders_phrase(layout, rels, packs) -> str:
    """How the library's folders read in its detected layout (packs/vendors.py): " in 13
    pack folders", " in 6 sound-type folders", ", most of them loose in the library folder"."""
    from ..packs import vendors as V
    plural = lambda n, w: f"{n} {w}{'' if n == 1 else 's'}"
    if layout == V.FLAT:
        return ", most of them loose in the library folder"
    if layout == V.TYPES:
        tops = {r.split("/")[0] for r in rels if "/" in r}
        return f" in {plural(len(tops), 'sound-type folder')}"
    if layout == V.UMBRELLA:
        inner = {tuple(r.split("/")[:3]) for r in rels if r.count("/") >= 3}
        return f" in {plural(len(inner) or len(packs), 'pack folder')}"
    return f" in {plural(len(packs), 'pack folder')}"


def check_thin(roots) -> list[tuple]:
    """A WARN, with what to expect, when the library is small: fewer than THIN_FILES audio
    files or THIN_PACKS packs (vendor/pack folders)."""
    packs: set = set()
    rels: list = []
    files, _cloud = survey(roots, THIN_FILES * 20, packs, rels)
    from ..packs import vendors as V
    layout = V.detect(rels)["layout"]
    by_type = layout in (V.TYPES, V.FLAT)        # no packs in the path: only the file count says
    if not files or (files >= THIN_FILES and (files >= THIN_FILES * 20 or by_type or len(packs) >= THIN_PACKS)):
        return []
    from ..packs.curate import MIN_PER_FAMILY
    from ..packs.curate_config import LIBRARY_SCALE, SCALED_MIN_FILES
    lib = f"a small library ({files:,} audio files{_folders_phrase(layout, rels, packs)})"
    if LIBRARY_SCALE:
        return [(WARN, "library size",
                 f"{lib}: the master scales down to it (scale = \"library\"), and a category builds "
                 f"from {SCALED_MIN_FILES} usable samples; the build names the categories it leaves "
                 f"empty, and `fourier why <file>` says why a sample was left out. More packs fill "
                 f"more categories.")]
    return [(WARN, "library size",
             f"{lib}: a build fills only the categories that have at least {MIN_PER_FAMILY} usable "
             f"samples (a folder's minimum) and leaves the rest empty, naming them; "
             f"`fourier why <file>` says why a sample was left out. More packs fill more categories.")]


# --- CLAP ---------------------------------------------------------------------------------
DIST = "fourier-samples"
CLAP_EXTRA = "clap"
# pyproject's [clap] extra, for a checkout run without installing it (no metadata to read)
CLAP_REQUIREMENTS = ("transformers>=4.40", "torch>=2.2")
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
UV_INSTALL = "curl -LsSf https://astral.sh/uv/install.sh | sh"


def clap_requirements() -> list[str]:
    """The [clap] extra's requirements (transformers and torch), from the installed
    package's metadata."""
    try:
        reqs = importlib.metadata.requires(DIST) or []
    except importlib.metadata.PackageNotFoundError:
        return list(CLAP_REQUIREMENTS)
    out = [spec.strip() for r in reqs for spec, _sep, marker in [r.partition(";")]
           if re.search(rf"""extra\s*==\s*['"]{CLAP_EXTRA}['"]""", marker)]
    return out or list(CLAP_REQUIREMENTS)


def installed_from_url() -> bool:
    """Whether this copy came from a git URL or a local folder (or isn't installed at all, a
    checkout): then the index has no matching package, so the extra's requirements are
    installed instead of the package."""
    try:
        dist = importlib.metadata.distribution(DIST)
    except importlib.metadata.PackageNotFoundError:
        return True
    return dist.read_text("direct_url.json") is not None


def cpu_only_torch() -> bool:
    """Linux without an NVIDIA GPU: the CPU build of PyTorch, without the CUDA libraries."""
    return sys.platform.startswith("linux") and shutil.which("nvidia-smi") is None


# the first uv whose `uv pip install` / `uv tool install` takes --torch-backend
UV_TORCH_BACKEND = {"pip": (0, 6, 9), "tool": (0, 9, 19)}
# an older uv gets PyTorch's CPU index beside PyPI instead; uv then takes each package's best
# version from either (by default it takes every package from the first index that has it,
# and PyTorch's index has old copies of packaging, urllib3 and others)
CPU_INDEX_ARGS = ("--index-strategy", "unsafe-best-match")


def uv_version(uv: str) -> tuple | None:
    """uv's version, (0, 8, 17), from `uv --version`; None when it can't be read."""
    try:
        out = subprocess.run([uv, "--version"], capture_output=True, text=True, timeout=30,
                             check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out or "")
    return tuple(int(x) for x in m.groups()) if m else None


def _dotted(v) -> str:
    return ".".join(map(str, v)) if v else "?"


def uv_has_torch_backend(uv: str, group: str = "pip") -> bool:
    """Whether this uv's `uv pip install` (or `uv tool install`) knows --torch-backend."""
    try:
        out = subprocess.run([uv, group, "install", "--help"], capture_output=True, text=True,
                             timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return "--torch-backend" in out.stdout


def has_pip() -> bool:
    """Whether the running Python has pip (a uv tool environment doesn't)."""
    import importlib.util
    return importlib.util.find_spec("pip") is not None


UV_RECEIPT = "uv-receipt.toml"


def uv_tool_receipt(prefix: str | None = None) -> dict | None:
    """The receipt uv keeps in a tool's environment (`uv tool install`), when this Python is
    one: what the tool was installed from and with."""
    import tomllib
    p = Path(prefix or sys.prefix) / UV_RECEIPT
    try:
        return tomllib.loads(p.read_text()).get("tool") or {}
    except (OSError, ValueError):
        return None


def _requirement(req: dict) -> list[str]:
    """A uv receipt's requirement as `uv tool install` arguments: "name>=1", "name @ git+URL@rev",
    "name @ file:///path", or --editable PATH."""
    from urllib.parse import parse_qs, urlsplit, urlunsplit
    name = req["name"] + (f"[{','.join(req['extras'])}]" if req.get("extras") else "")
    if req.get("editable"):
        return ["--editable", req["editable"]]
    if req.get("git"):
        u = urlsplit(req["git"])
        q = parse_qs(u.query)
        ref = next((q[k][0] for k in ("rev", "tag", "branch") if q.get(k)), None)
        url = urlunsplit((u.scheme, u.netloc, u.path + (f"@{ref}" if ref else ""), "", u.fragment))
        return [f"{name} @ git+{url}"]
    if req.get("url"):
        return [f"{name} @ {req['url']}"]
    for key in ("directory", "path"):
        if req.get(key):
            return [f"{name} @ {Path(req[key]).expanduser().resolve().as_uri()}"]
    return [name + (req.get("specifier") or "")]


def _cpu_args(uv: str, group: str) -> list[str]:
    """How this uv installs the CPU build of PyTorch: --torch-backend cpu, else PyTorch's CPU
    index beside PyPI, best version from either (CPU_INDEX_ARGS)."""
    if uv_has_torch_backend(uv, group):
        return ["--torch-backend", "cpu"]
    return ["--index" if group == "tool" else "--extra-index-url", TORCH_CPU_INDEX, *CPU_INDEX_ARGS]


def _req_name(req: str) -> str:
    return re.split(r"[\s<>=!~\[;@]", req, maxsplit=1)[0].lower()


def uv_tool_install(uv: str, receipt: dict, reqs: list[str], cpu: bool, pin: str | None = None) -> list[str]:
    """`uv tool install --reinstall` of this tool from the source its receipt names, with the
    [clap] requirements as --with: uv records them, so `uv tool upgrade` keeps them (installed
    into the tool's environment any other way, an upgrade would drop them). `pin`, a CPU
    build's version ("2.5.1+cpu"): PyTorch exactly that, from PyTorch's CPU index, both of
    which uv records too (pin_cpu_torch)."""
    mine = [r for r in receipt.get("requirements") or () if r.get("name") == DIST]
    source = _requirement(mine[0]) if mine else [DIST]
    if pin:
        reqs = [f"torch=={pin}" if _req_name(r) == "torch" else r for r in reqs]
    names = {_req_name(r) for r in reqs}
    other = [a for r in receipt.get("requirements") or ()
             if r.get("name") != DIST and r.get("name", "").lower() not in names
             for a in ("--with", " ".join(_requirement(r)))]
    cmd = [uv, "tool", "install", "--reinstall",
           "--python", f"{sys.version_info.major}.{sys.version_info.minor}"]
    if pin:     # the index goes in the receipt with the pin, so upgrades find that version
        cmd += ["--index", TORCH_CPU_INDEX, *CPU_INDEX_ARGS]
    elif cpu:   # the CPU build of PyTorch (uv's receipt keeps no --torch-backend: pin_cpu_torch)
        cmd += _cpu_args(uv, "tool")
    for r in reqs:
        cmd += ["--with", r]
    return cmd + other + source


def clap_install(exe: str | None = None, reinstall_torch: bool = False) -> tuple[str | None, list[list[str]]]:
    """(the installer, the commands) that add the [clap] extra to the Python running this:
    in a uv tool environment, `uv tool install --reinstall --with ...` (so upgrades keep it);
    else uv when it's on PATH, else this Python's pip; (None, the commands to run by hand) with
    neither. On Linux without an NVIDIA GPU, PyTorch comes from its CPU index.
    `reinstall_torch`: replace the PyTorch that's there (a CUDA build where it can't run)."""
    exe = exe or sys.executable
    reqs = clap_requirements()
    receipt = uv_tool_receipt()
    uv = shutil.which("uv")
    if receipt is not None and uv:
        return "uv tool", [uv_tool_install(uv, receipt, reqs, cpu_only_torch())]
    targets = reqs if installed_from_url() else [f"{DIST}[{CLAP_EXTRA}]=={_version()}"]
    torch = [r for r in reqs if re.match(r"torch(?![\w.-])", r)]
    cpu = cpu_only_torch()
    if uv:
        base = [uv, "pip", "install", "--python", exe]
        again = ["--reinstall-package", "torch"] if reinstall_torch else []
        return "uv", [base + (_cpu_args(uv, "pip") if cpu else []) + again + targets]
    if has_pip():
        base = [exe, "-m", "pip", "install"]
        if not cpu:
            return "pip", [base + targets]
        again = ["--force-reinstall"] if reinstall_torch else []
        return "pip", [base + ["--index-url", TORCH_CPU_INDEX, *again, *torch], base + targets]
    base = ["uv", "pip", "install", "--python", exe]
    return None, [base + (["--torch-backend", "cpu"] if cpu else []) + targets]


def uv_too_old(tool: str | None, cpu: bool | None = None) -> tuple | None:
    """With uv as the installer of the CPU build of PyTorch (`cpu`, default: Linux without an
    NVIDIA GPU): (its version, the version whose --torch-backend this install would use)
    when it's older, else None."""
    uv = shutil.which("uv")
    if tool not in ("uv", "uv tool") or not uv or not (cpu_only_torch() if cpu is None else cpu):
        return None
    group = "tool" if tool == "uv tool" else "pip"
    if uv_has_torch_backend(uv, group):
        return None
    return uv_version(uv), UV_TORCH_BACKEND[group]


def offer_uv_update(w, tool: str | None, cpu: bool | None = None) -> bool:
    """When uv is too old to name the CPU build (uv_too_old), offer `uv self update`. True
    when uv updated itself (work out the commands again); False when it's not needed, was
    declined or failed (the CPU index then, CPU_INDEX_ARGS)."""
    old = uv_too_old(tool, cpu)
    if old is None:
        return False
    have, need = old
    console.print(f"uv {_dotted(have)} is older than {_dotted(need)}, the first uv that installs "
                  f"the CPU build of PyTorch by name here (--torch-backend cpu).", highlight=False)
    if getattr(w, "yes", False):      # never update a tool on the machine unasked
        console.print("Kept this uv (--yes doesn't update it): installing PyTorch from its CPU "
                      "index instead. `uv self update`, then `fourier setup` again, uses the "
                      "newer uv.", highlight=False)
        return False
    if not w.confirm("Run `uv self update` now?", default=True):
        console.print("Kept this uv: installing PyTorch from its CPU index instead.", highlight=False)
        return False
    console.print("$ uv self update", markup=False, highlight=False)
    if _run([shutil.which("uv"), "self", "update"]) != 0 or uv_too_old(tool, cpu) is not None:
        console.print("uv didn't update itself (a uv from a package manager updates there): "
                      "installing PyTorch from its CPU index instead.", style="yellow", highlight=False)
        return False
    return True


# uv's notice that --torch-backend is a preview feature: nothing for the user to act on
_UV_NOISE = re.compile(r"`--torch-backend`.{0,20}\bexperimental\b", re.I)


def shown_lines(lines) -> list[str]:
    """An installer's output as the user sees it: without uv's preview-feature notice."""
    return [ln for ln in lines if not _UV_NOISE.search(ln)]


def _run(cmd: list[str], keep: list | None = None, quiet: bool = False) -> int:
    """Run a command with its output on the terminal (less uv's preview notice, shown_lines);
    its exit code. `keep`, a list, gets the output's last lines too (for a summary when it
    fails). `quiet`: nothing shown while it runs (the caller shows `keep`)."""
    if keep is None and not quiet:
        return subprocess.run(cmd, check=False).returncode
    from collections import deque
    tail: deque = deque(maxlen=200)
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace", bufsize=1)
    except OSError as e:
        if keep is not None:
            keep.append(f"error: {e}")
        return 127
    for line in p.stdout:
        if _UV_NOISE.search(line):
            continue
        if not quiet:
            sys.stdout.write(line)
            sys.stdout.flush()
        tail.append(line.rstrip("\n"))
    code = p.wait()
    if keep is not None:
        keep.extend(tail)
    return code


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ERROR_START = re.compile(r"^\s*(?:×|error:|ERROR:|Traceback)")
# the lines of an error block that say what went wrong: uv's "×" and its causes ("├─▶",
# "╰─▶"), "error:", pip's "ERROR:", and a "hint:" / "help:" (with its paragraph)
_ERROR_MARK = re.compile(r"^\s*(?:×|├─▶|╰─▶|error:|ERROR:|caused by:|hint:|help:)", re.I)


def error_lines(lines: list[str], n: int = 10, hint: int = 3) -> list[str]:
    """What an installer's output says went wrong: its last error block (uv's "×" or
    "error:", pip's "ERROR:"), as its cause lines and the start of its hint (`hint` lines
    more), at most n lines; else its last n lines."""
    clean = [_ANSI.sub("", ln).rstrip() for ln in lines]
    starts = [i for i, ln in enumerate(clean) if _ERROR_START.match(ln)]
    if not starts:
        return [ln.strip() for ln in clean if ln.strip()][-n:]
    out, more = [], 0
    for ln in clean[starts[-1]:]:
        if _ERROR_MARK.match(ln):
            out.append(ln.strip())
            more = hint if ln.strip().lower().startswith(("hint:", "help:")) else 0
        elif more and ln.strip():
            out.append("  " + ln.strip())            # the start of the hint
            more -= 1
        else:
            more = 0
    return out[:n]


# the oldest uv setup's commands are known to work with (`uv tool install --torch-backend`)
UV_KNOWN_GOOD = UV_TORCH_BACKEND["tool"]


def install_failed(cmd: list[str], lines: list[str]) -> None:
    """A short summary of a failed install: what the installer said (its error, not the whole
    log) and one next step: with a uv older than UV_KNOWN_GOOD, updating it; else the
    command to retry."""
    console.print("The install didn't finish. What it said:", style="red", highlight=False)
    for ln in error_lines(lines) or ["(nothing)"]:
        console.print(f"  {ln}", markup=False, highlight=False, soft_wrap=True)
    uv = cmd[0] if cmd and Path(cmd[0]).name.startswith("uv") else None
    have = uv_version(uv) if uv else None
    if have is not None and have < UV_KNOWN_GOOD:
        console.print(f"Your uv ({_dotted(have)}) is old: run `uv self update` and then "
                      f"`fourier setup` again.", highlight=False)
        return
    console.print("Retry with this command, then `fourier setup` again:", highlight=False)
    _say(f"  {shlex.join(cmd)}")


INSTALL_FILE = "install.json"


def save_install(installer: str | None, torch: str) -> None:
    """That setup installed the [clap] extra, and how (installer, and "cpu" or "default"
    PyTorch), in the Fourier home's install.json: doctor tells a reinstall that dropped it
    from a first run. A home that can't be written is skipped."""
    from ..paths import home_path
    p = home_path(INSTALL_FILE)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"installer": installer, "torch": torch,
                                 "date": date.today().isoformat()}, indent=1) + "\n")
    except OSError:
        pass


def saved_install() -> dict:
    from ..paths import home_path
    try:
        return json.loads(home_path(INSTALL_FILE).read_text())
    except (OSError, ValueError):
        return {}


def installed_torch(exe: str | None = None) -> str | None:
    """The version of PyTorch installed for this Python ("2.5.1+cpu"), read by a new process
    (its environment may have just been reinstalled); None without one."""
    try:
        out = subprocess.run([exe or sys.executable, "-c", "import importlib.metadata as m; "
                              "print(m.version('torch'))"], capture_output=True, text=True,
                             timeout=120, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    v = out.stdout.strip()
    return v if out.returncode == 0 and v else None


def torch_pin(receipt: dict | None) -> str | None:
    """The PyTorch version a uv tool's receipt pins ("torch==2.5.1+cpu": "2.5.1+cpu"), else None."""
    for r in (receipt or {}).get("requirements") or ():
        spec = (r.get("specifier") or "").strip()
        if r.get("name", "").lower() == "torch" and spec.startswith("==") and "," not in spec:
            return spec[2:].strip()
    return None


def receipt_dropped_clap(receipt: dict | None) -> bool:
    """A uv tool reinstalled without the [clap] requirements setup gave it (`uv tool install
    --reinstall` with only the package): its receipt has no torch or transformers while
    install.json says setup installed them."""
    if receipt is None or not saved_install().get("installer"):
        return False
    names = {(r.get("name") or "").lower() for r in receipt.get("requirements") or ()}
    return not names & {"torch", "transformers"}


def pin_cpu_torch(tool: str | None) -> None:
    """After an install into a uv tool on Linux without an NVIDIA GPU: pin the CPU build of
    PyTorch it got (`--with torch==2.5.1+cpu`, and PyTorch's CPU index beside PyPI), both kept
    in uv's receipt, so `uv tool upgrade` keeps that build. uv's receipt keeps no
    --torch-backend, and an unpinned upgrade would bring the CUDA build once a newer PyTorch
    is out. `fourier setup` again updates it (and pins the new one)."""
    if tool != "uv tool" or not cpu_only_torch():
        return
    v = installed_torch()
    receipt, uv = uv_tool_receipt(), shutil.which("uv")
    if not v or not v.endswith("+cpu") or receipt is None or not uv or torch_pin(receipt) == v:
        return
    cmd = uv_tool_install(uv, receipt, clap_requirements(), True, pin=v)
    console.print(f"Pinning PyTorch at {v} (the CPU build), so `uv tool upgrade` keeps it.",
                  highlight=False)
    # the same packages again: run quietly, its whole output only when it fails
    out: list = []
    if _run(cmd, keep=out, quiet=True) != 0:
        _say(f"$ {shlex.join(cmd)}")
        for ln in shown_lines(out):
            _say(ln)
        console.print("PyTorch isn't pinned (the CLAP install itself is fine): "
                      f"{(error_lines(out, n=1) or ['no reason given'])[0]}. After a `uv tool "
                      "upgrade`, `fourier doctor` says whether it brought the CUDA build.",
                      style="yellow", markup=False, highlight=False)
        return
    console.print(f"Pinned: uv's receipt names torch=={v} and PyTorch's CPU index.", highlight=False)


def torch_size() -> str:
    """What PyTorch takes here: the build setup installs is the CPU one on macOS, on Windows
    (PyPI's) and on Linux without an NVIDIA GPU, about 200 MB to download; on Linux with an
    NVIDIA GPU, the CUDA build with NVIDIA's libraries, several GB."""
    if sys.platform.startswith("linux") and not cpu_only_torch():
        return "PyTorch with CUDA (for the NVIDIA GPU here) is several GB"
    if sys.platform.startswith("linux"):
        return "PyTorch's CPU build is about 200 MB (about 750 MB installed)"
    return "PyTorch is about 200 MB (about 750 MB installed)"


def torch_has_cuda() -> bool:
    """Whether the installed PyTorch is a CUDA build (it carries NVIDIA's libraries), read from
    the installed packages without importing it: not a "+cpu" version, and nvidia-* packages
    beside it."""
    try:
        v = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return False
    if "+cpu" in v:
        return False
    return "+cu" in v or any((d.metadata.get("Name") or "").lower().startswith("nvidia-")
                             for d in importlib.metadata.distributions())


def _version() -> str:
    try:
        return importlib.metadata.version(DIST)
    except importlib.metadata.PackageNotFoundError:
        return "0"


def clap_model_cached() -> bool:
    """Whether the pinned CLAP model is already downloaded (its files in the Hugging Face
    cache; looked up without contacting the Hub)."""
    from .enrich import clap_model_missing
    return not clap_model_missing()


def download_clap_model() -> str:
    """The pinned CLAP model from Hugging Face (about 600 MB, once), with one progress bar of
    ours: the Hub's own bars (one per file, and its downloader's) are off, and ours follows
    the bytes arriving in the cache."""
    import threading

    from ..analysis.clap_features import CLAP_MODEL_ID, CLAP_REVISION, hf_hub_cache, quiet_hub
    quiet_hub()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass
    total = None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(CLAP_MODEL_ID, revision=CLAP_REVISION, files_metadata=True)
        total = sum(f.size or 0 for f in info.siblings or ()) or None
    except Exception:            # the bar then counts bytes without a total
        pass
    blobs = hf_hub_cache() / ("models--" + CLAP_MODEL_ID.replace("/", "--")) / "blobs"
    have = _folder_bytes(blobs)
    got: dict = {}

    def fetch():
        try:
            got["path"] = snapshot_download(repo_id=CLAP_MODEL_ID, revision=CLAP_REVISION)
        except BaseException as e:      # handed to the caller below
            got["error"] = e
    t = threading.Thread(target=fetch, daemon=True)
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn
    with Progress(TextColumn("CLAP model"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                  console=console, transient=False) as bar:
        task = bar.add_task("model", total=total)
        t.start()
        while t.is_alive():
            t.join(0.25)
            bar.update(task, completed=max(0, _folder_bytes(blobs) - have))
        if "path" in got and total:
            bar.update(task, completed=total)
    if "error" in got:
        raise got["error"]
    return got["path"]


def _folder_bytes(folder: Path) -> int:
    """The bytes of the files in a folder (a download's partial files too)."""
    try:
        return sum(e.stat().st_size for e in os.scandir(folder) if e.is_file())
    except OSError:
        return 0


def load_clap_model() -> None:
    """Load the CLAP model once, as analysis and builds do (raises if it can't)."""
    from ..analysis import clap_features as CF
    CF._load_model_and_processor()


# --- the local LLM -------------------------------------------------------------------------
OLLAMA = "http://localhost:11434"


def ollama_status() -> tuple[bool, bool, list[str]]:
    """(the ollama program is on PATH, its server answers on localhost, the models it has)."""
    from ..net import local_opener
    binary = shutil.which("ollama") is not None
    try:
        with local_opener().open(f"{OLLAMA}/api/tags", timeout=2) as r:
            models = [m.get("name") or m.get("model") for m in json.loads(r.read()).get("models") or ()]
        return binary, True, [m for m in models if m]
    except (OSError, ValueError):
        return binary, False, []


def ollama_pull(model: str, progress=None) -> None:
    """Pull a model through Ollama's local API; progress(done_bytes, total_bytes, status)
    after each update. Raises RuntimeError when Ollama reports an error."""
    import urllib.request

    from ..net import local_opener
    req = urllib.request.Request(f"{OLLAMA}/api/pull", method="POST",
                                 data=json.dumps({"model": model, "name": model, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    sizes: dict[str, tuple[int, int]] = {}
    with local_opener().open(req, timeout=60) as r:
        for line in r:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("error"):
                raise RuntimeError(d["error"])
            if d.get("digest") and d.get("total"):
                sizes[d["digest"]] = (d.get("completed") or 0, d["total"])
            if progress:
                progress(sum(c for c, _t in sizes.values()), sum(t for _c, t in sizes.values()),
                         d.get("status") or "")


def describe_model() -> str:
    """The model folder descriptions use (the DEFAULT_MODEL tunable, as the config sets it)."""
    from .. import layers
    try:
        r = layers.resolve()
        return r.values.get("curate.DEFAULT_MODEL") or layers.defaults()["curate.DEFAULT_MODEL"]
    except Exception:
        return "qwen3.5:9b"


# --- setup -------------------------------------------------------------------------------
BIG_BUILD_S = 15 * 60    # a first build estimated longer than this isn't started by default


class _Wizard:
    """The questions, asked or (with --yes) answered with the defaults."""

    def __init__(self, yes: bool):
        self.yes = yes

    def step(self, n: int, title: str, why: str) -> None:
        console.print(f"\n[bold]{n}/6  {title}[/bold]", highlight=False)
        console.print(why, highlight=False)

    def ask(self, text: str, default=None, **kw):
        if self.yes:
            return default
        return click.prompt(text, default=default, **kw)

    def confirm(self, text: str, default: bool) -> bool:
        if self.yes:
            return default
        return click.confirm(text, default=default)


@main.command("setup", short_help="Set Fourier Samples up for your library, step by step.")
@click.option("--library", "library", multiple=True,
              help="A folder of samples (repeatable, or comma-separated)")
@click.option("--device", "devices", multiple=True,
              help="A device id (repeatable, or comma-separated; fourier devices list)")
@click.option("--preset", default=None,
              help="The style: balanced (the default), breaks-acid, hiphop-lofi, house-techno, "
                   "ambient-cinematic, trap, or a preset file")
@click.option("--master", default=None, help="Where the master goes (default: ~/Music/FourierCurated; "
                                            "FourierCurated-<config name> for a config --to puts elsewhere)")
@click.option("--renders", default=None, help="Where device renders go (default: beside the master)")
@click.option("--publish", default=None, help="Where releases go (default: ~/Music/Fourier; "
                                              "Fourier-<config name> for a config --to puts elsewhere)")
@click.option("--to", "path", default=None,
              help="Where to write the config (default: ~/.config/fourier/fourier.toml); another "
                   "path is a second library's, with its own master, renders and releases")
@click.option("--force", is_flag=True, default=False, help="Replace an existing config")
@click.option("--clap/--no-clap", "clap", default=None,
              help="Install the CLAP model a build needs (default: ask; yes with --yes); "
                   "when it's installed, update PyTorch and Transformers (default: ask; no "
                   "with --yes)")
@click.option("--llm/--no-llm", "llm", default=None,
              help="Set up folder descriptions with a local LLM (default: ask; no with --yes)")
@click.option("--sound-model/--no-sound-model", "sound_model", default=None,
              help="Without Sononym, let a build train a sound model on your library's own names "
                   "(default: on; --no-sound-model writes SOUND_TRAIN = false to the config)")
@click.option("--size", default=None,
              help="How big a set: starter (about 1 GB), standard (about 3 GB), full (the style's "
                   "own size; the default), or a card size like 4GB")
@click.option("--processing", type=click.Choice(["prepared", "as-is"]), default=None,
              help="prepared: melodic one-shots tuned to C, loops levelled, tidy names (the "
                   "default); as-is: every sound as it is, with its own name")
@click.option("--build/--no-build", "build", default=None,
              help="Start `fourier build` at the end (default: ask; no with --yes)")
@click.option("--yes", "-y", is_flag=True, default=False,
              help="Ask nothing: the options given, defaults for the rest")
@click.pass_context
def setup(ctx, library, devices, preset, master, renders, publish, path, force, clap, llm,
          build, yes, sound_model=None, size=None, processing=None):
    """Set Fourier Samples up for your library, step by step: where your samples are, your
    devices and style, the config file, the CLAP model a build needs, an optional local LLM,
    and a check of everything. Each step says what it's for and can be skipped; run it
    again any time.

    \b
      fourier setup
      fourier setup --yes --library ~/Samples --device digitakt_2 --preset balanced
      fourier setup --yes --force --library ~/Samples --device m8_tracker   # replace a config
    """
    from ..layers import USER_CONFIG
    w = _Wizard(yes)
    target = Path(path).expanduser() if path else USER_CONFIG.expanduser()
    shown = path or str(USER_CONFIG)          # the path as given, in messages (not resolved)
    console.print("Fourier Samples setup. It only reads your sample folders; everything it "
                  "writes is listed as it goes. Press Ctrl-C to stop at any point.", highlight=False)
    given = bool(library or devices or preset or master or renders or publish or size or processing)
    outputs = {"master": master, "renders": renders, "publish": publish}
    if yes and target.exists() and given and not force:
        if not _same_answers(target, library, devices, preset, outputs):
            _say(f"A config exists at {shown}; pass --force to replace it, or drop the options "
                 f"to keep it.", style="red")
            raise SystemExit(1)
        given = False                          # it says that already: keep it
    keep = target.exists() and not force and not given and (
        yes or w.confirm(f"You already have a config at {shown}. Keep it (skips steps 1 to 3)?",
                         default=True))
    if keep and yes:
        _say(f"Kept the config at {shown} (steps 1 to 3 skipped; `--force` with options "
             f"replaces it).")
    if yes and not keep and not library:
        raise click.UsageError("setup --yes needs --library: where your samples are")
    if not keep:
        # declining to keep it is the go-ahead to replace it
        _config_steps(w, target, library, devices, preset, outputs,
                      replace=force or (target.exists() and not given), shown=shown,
                      size=size, processing=processing)
    from .. import places
    places.reset()
    _clap_step(w, clap)
    _sound_model_note(sound_model, target if target.exists() else None, shown)
    _llm_step(w, llm, target if target.exists() else None)
    _check_step(ctx, w, build, target if path else None)


def _same_answers(target: Path, library, devices, preset, outputs: dict) -> bool:
    """Whether a config already says what these setup options would write (the options
    given; the ones left out don't count)."""
    import tomllib
    try:
        doc = tomllib.loads(target.read_text())
    except (OSError, ValueError):
        return False
    norm = lambda xs: [str(Path(x).expanduser()) for x in xs]
    if library and norm(folder_answers(library)) != norm(doc.get("library") or []):
        return False
    if devices and _split(devices) != list(doc.get("devices") or []):
        return False
    if preset and preset.strip() != doc.get("preset"):
        return False
    have = doc.get("output") or {}
    return all(str(Path(v).expanduser()) == str(Path(have.get(k, default_outputs(target)[k])).expanduser())
               for k, v in outputs.items() if v)


def _say(text: str, **kw) -> None:
    """A line with a path in it: printed as it is (no markup, never folded mid-path; the
    terminal wraps it)."""
    console.print(text, markup=False, highlight=False, soft_wrap=True, **kw)


def _choice_steps(w: _Wizard, devs: list[str], size, processing) -> list[str]:
    """How big a set, prepared or as-is, and categories to leave out: the lines they add to
    the config (none for the defaults: the style's size, prepared for hardware, every
    category)."""
    lines: list[str] = []
    transfer = "digitakt_2" in devs
    if size is None and not w.yes:
        console.print("How big a set? (you can change it later)", highlight=False)
        for i, (name, _line, what) in enumerate(SIZES, 1):
            gb = {"starter": 1, "standard": 3}.get(name)
            t = (f"; about {max(1, round(gb / GB_PER_TRANSFER_HOUR * 60 / 15) * 15)} minutes "
                 f"to load with Transfer" if transfer and gb else "")
            console.print(f"  {i}  {name:<9} {what}{t}", markup=False, highlight=False, soft_wrap=True)
        while True:
            try:
                line, name = size_line(w.ask("Size (a number, or a card size like 4GB)", default="3"))
                break
            except ValueError as e:
                console.print(f"not a size: {e} (1, 2, 3 or a size like 4GB)", style="red",
                              markup=False, highlight=False)
    else:
        try:
            line, name = size_line(size)
        except ValueError as e:
            raise click.UsageError(f"--size {e}: starter, standard, full or a size like 4GB") from None
    if line:
        lines.append(f"{line:<24} # the set's size (setup's {name}): a card size, \"auto\" or \"fixed\"")
    if processing is None and not w.yes:
        console.print("Prepare the sounds for playing on hardware, or keep them as they are?",
                      highlight=False)
        console.print("  1  prepared  melodic one-shots tuned to C, loops at an even level, near-mono "
                      "files made mono, tidy names (the default)", markup=False, highlight=False,
                      soft_wrap=True)
        console.print("  2  as-is     every sound at its own pitch, level and channels, with its "
                      "original name", markup=False, highlight=False, soft_wrap=True)
        while True:
            a = (w.ask("Prepared or as-is? (1 or 2)", default="1") or "1").strip().lower()
            if a in ("1", "prepared", "2", "as-is", "as is", "asis"):
                processing = "as-is" if a in ("2", "as-is", "as is", "asis") else "prepared"
                break
    if processing == "as-is":
        lines += list(AS_IS)
    if not w.yes:
        from ..packs.curate_config import CATEGORY_ORDER
        cats = list(CATEGORY_ORDER)
        console.print("Leave any kinds of sound out? Every one is in unless you say:",
                      highlight=False)
        console.print("  " + "  ".join(f"{i}:{c}" for i, c in enumerate(cats, 1)), markup=False,
                      highlight=False)
        while True:
            a = w.ask("Numbers or names to leave out (Return keeps them all)", default="",
                      show_default=False) or ""
            picked, bad = [], []
            for t in _split([a]):
                if t.isdigit() and 1 <= int(t) <= len(cats):
                    picked.append(cats[int(t) - 1])
                elif t.upper() in cats:
                    picked.append(t.upper())
                else:
                    bad.append(t)
            if not bad:
                break
            console.print(f"not categories: {', '.join(bad)}", style="red", markup=False,
                          highlight=False)
        if picked:
            off = ", ".join(f'{c} = "off"' for c in dict.fromkeys(picked))
            lines.append(f"categories = {{ {off} }}   # left out at setup (\"on\" brings one back)")
    return lines


def _config_steps(w: _Wizard, target: Path, library, devices, preset, outputs: dict, replace: bool,
                  shown: str | None = None, size=None, processing=None):
    """Steps 1 to 3: the library, the devices and style, the config file. Returns the answers
    written, or None when skipped. `shown`: the config's path as the user gave it."""
    shown = shown or str(target)
    from ..devices.loader import DeviceLoader
    w.step(1, "Your samples", "Fourier reads your sample folders (and never changes them) to "
                              "pick the best of them. Give the folder or folders that hold "
                              "your packs.")
    libs = folder_answers(library)
    while True:
        if not libs:
            from ..platforms import file_browser
            console.print(f"Drag the folder from {file_browser()} into this window (or type its "
                          "path), then press Return. Several folders: drag them in together, or "
                          "separate them with commas.", markup=False, highlight=False)
            answer = w.ask("Where are your samples? (blank skips setting up the config)",
                           default="", show_default=False)
            libs = folder_answers([answer or ""])
            if not libs:
                console.print("Skipped steps 1 to 3: no config written.", highlight=False)
                return None
        missing = [d for d in libs if not Path(d).expanduser().is_dir()]
        if missing:
            console.print(f"not folders: {', '.join(missing)} (no folder by that name; dragging "
                          "it in from your file browser gives its exact path)", style="red",
                          markup=False, highlight=False, soft_wrap=True)
            if w.yes:
                raise SystemExit(1)
            libs = []
            continue
        libs = [str(Path(d).expanduser()) for d in libs]
        n, cloud = survey([Path(d) for d in libs], COUNT_CAP)
        more = "+" if n >= COUNT_CAP else ""
        console.print(f"Found {n:,}{more} audio file{'' if n == 1 else 's'}.", highlight=False)
        if n or w.yes:
            break
        from ..ingest.formats import readable_exts
        console.print(f"No audio files there ({', '.join(readable_exts())}): give the folder that "
                      "holds your sample packs.", style="yellow", markup=False, highlight=False,
                      soft_wrap=True)
        libs = []
    if not n:
        from ..ingest.formats import readable_exts
        console.print(f"No audio files there yet ({', '.join(readable_exts())}): a build needs "
                      "some.", style="yellow", markup=False, highlight=False)
    else:
        from ..packs.vendors import describe, detect_folders
        _say(f"Layout: {describe(detect_folders([Path(d) for d in libs]))} (vendors = \"auto\").")
    if cloud:
        console.print(f"{cloud:,} of them are only in the cloud, not on this machine: a build "
                      f"downloads the ones it picks and analysis skips them. Keeping the folder "
                      f"downloaded is faster and safer ({CLOUD_HINT}).",
                      style="yellow", markup=False, highlight=False)

    loader = DeviceLoader()
    known = loader.list_devices()
    w.step(2, "Devices and style", "Fourier renders the master for each of your samplers, in "
                                   "the format and folder rules of each. The style decides "
                                   "what the master holds.")
    console.print("Devices:", highlight=False)
    names = {}
    for i, d in enumerate(known, 1):
        try:
            names[d] = loader.load(d).name
        except Exception:
            names[d] = "?"
        console.print(f"  {i:>2}  {d:<16} {names[d]}", markup=False, highlight=False)
    console.print("Your device isn't listed? Pick generic_folder or run `fourier devices new` "
                  "later.", markup=False, highlight=False)
    devs, bad = pick_devices(devices or (), known, names)
    while True:
        if not devs and not bad:
            devs, bad = pick_devices([w.ask("Which devices? (numbers or names, comma-separated)",
                                            default="generic_48k" if w.yes else None)],
                                     known, names)
        if not bad:
            break
        console.print(f"unknown devices: {', '.join(bad)} (known: {', '.join(known)})",
                      style="red", markup=False, highlight=False)
        if w.yes:
            raise SystemExit(1)
        devs, bad = [], []
    names = presets()
    console.print("What do you make? Each style decides how much of each kind of sound the "
                  "master holds and the drum loops' tempos:", highlight=False)
    width = max(len(p) for p in names) + 2
    for i, p in enumerate(names, 1):
        console.print(f"  {i:>2}  {p:<{width}} {preset_line(p)}", markup=False, highlight=False)
    while True:
        if preset is None:
            preset = w.ask("What do you make? (a number or a style's name)", default="balanced")
        preset = (preset or "").strip() or "balanced"
        if preset.isdigit() and 1 <= int(preset) <= len(names):
            preset = names[int(preset) - 1]
        if preset in names or Path(preset).expanduser().is_file():
            break
        console.print(f"unknown preset {preset!r} (known: {', '.join(names)})", style="red",
                      markup=False, highlight=False)
        if w.yes:
            raise SystemExit(1)
        preset = None
    extra = _choice_steps(w, devs, size, processing)
    defaults = default_outputs(target)
    own = config_tag(target) is not None       # not the default config: folders of its own
    chosen = {k: str(Path(v).expanduser()) for k, v in outputs.items() if v}
    console.print("Where Fourier writes (the master is the curated folder; renders are its "
                  "copies for each device; releases are fixed versions of it):", highlight=False)
    for k in OUTPUTS:
        _say(f"  {k:<8} {chosen.get(k, defaults[k])}")
    if own and len(chosen) < len(OUTPUTS):
        _say(f"{shown} isn't the default config (~/.config/fourier/fourier.toml), so its master, renders and "
             f"releases get folders of their own, named after it: a build for one library never "
             f"replaces another's master.")
    if not chosen and not w.confirm("Keep these folders?", default=True):
        for k in OUTPUTS:
            v = str(Path(click.prompt(f"  {k}", default=defaults[k])).expanduser())
            if v != defaults[k]:
                chosen[k] = v
    if own:                                    # written out: the code's defaults are the shared ones
        chosen = {k: chosen.get(k, defaults[k]) for k in OUTPUTS}

    w.step(3, "The config file", f"Your answers go in {shown}, a plain text file you can edit "
                                 f"(`fourier config show` prints what it sets).")
    if target.exists() and not replace and not w.confirm(f"{shown} exists. Replace it?", default=False):
        _say(f"Kept {shown} as it was.")
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(config_text(libs, devs, preset, {k: chosen[k] for k in OUTPUTS if k in chosen},
                                  extra=extra))
    _say(f"Wrote {shown}.")
    from ..layers import find_config
    from ..settings import ConfigError
    try:
        found = find_config()
    except ConfigError:
        found = None
    if found is None or found.resolve() != target.resolve():
        console.print(f"Note: Fourier reads {found or 'no config'} first ($FOURIER_CONFIG or a "
                      f"fourier.toml in this folder), not this file.", style="yellow",
                      markup=False, highlight=False)
    return dict(library=libs, devices=devs, preset=preset, output=chosen)


def _clap_step(w: _Wizard, clap) -> None:
    """Step 4: the [clap] extra (PyTorch and Transformers) and the pinned CLAP model. When
    they're installed: the CPU build of PyTorch in place of a CUDA build that can't run here,
    or, in a uv tool whose PyTorch setup pinned (pin_cpu_torch), an update when asked."""
    from .enrich import clap_extra_missing
    w.step(4, "The CLAP model", "A build sorts and names sounds with CLAP, a model that "
                                "listens to each sample. It needs PyTorch and Hugging Face "
                                "Transformers, and the model itself.")
    missing = clap_extra_missing()
    cached = clap_model_cached()
    cuda = not missing and cpu_only_torch() and torch_has_cuda()
    if not missing and not cuda:
        pinned = torch_pin(uv_tool_receipt())
        if pinned:
            console.print(f"Installed; PyTorch is pinned at {pinned} (the CPU build), so `uv tool "
                          f"upgrade` keeps it.", highlight=False)
            if clap if clap is not None else (not w.yes and w.confirm(
                    "Update PyTorch and Transformers to their newest now?", default=True)):
                if _install_clap(w):
                    _model_ready(cached)
                return
        if cached:
            if not pinned:
                console.print("Installed, and the model is downloaded.", highlight=False)
            return
        console.print("PyTorch and Transformers are installed; the model (about 600 MB, once) "
                      "isn't downloaded yet.", highlight=False)
        if clap is None:
            clap = w.confirm("Download the model now?", default=True)
        if not clap:
            console.print("Skipped: the first build downloads it (or `fourier setup` again).",
                          highlight=False)
            return
        _model_ready(cached)
        return
    if cuda:
        console.print("PyTorch here is a CUDA build, with NVIDIA libraries this machine can't use "
                      "(no NVIDIA GPU): the CPU build is several GB smaller.", highlight=False)
    else:
        console.print(f"Not installed yet. It's a one-time download: {torch_size()}"
                      + (", and the model about 600 MB." if not cached else
                         " (the model is downloaded already)."), highlight=False)
    if clap is None:
        clap = w.confirm("Switch to the CPU build now?" if cuda else "Install it now?", default=True)
    if not clap:
        console.print("Kept it. `fourier setup` again switches it." if cuda else
                      "Skipped. A build needs it: run `fourier setup` again when you're ready.",
                      highlight=False)
        return
    if _install_clap(w, reinstall_torch=cuda):
        _model_ready(cached)


def _sound_model_note(sound_model, config: Path | None, shown: str) -> None:
    """After the CLAP step: what the first build does with it, the sound model (without
    Sononym: trained on the library's own names, metadata/train.maybe_train), and
    --no-sound-model / --sound-model writing SOUND_TRAIN to the config."""
    import tomllib
    off = False
    if config is not None:
        try:
            off = (tomllib.loads(config.read_text()).get("advanced") or {}).get("SOUND_TRAIN") is False
        except (OSError, tomllib.TOMLDecodeError):
            off = False
    if sound_model is False and config is not None:
        set_advanced(config, "SOUND_TRAIN", False, SOUND_OFF_LINE)
        _say(f"The sound model is off: SOUND_TRAIN = false in {shown}.")
        return
    if sound_model is True and off and config is not None:
        set_advanced(config, "SOUND_TRAIN", True, "SOUND_TRAIN = true\n")
        off = False
    if off:
        _say(f"The sound model is off (SOUND_TRAIN = false in {shown}; `fourier setup "
             f"--sound-model` turns it on).")
        return
    from ..packs.curate_config import SOUND_TRAIN_MIN_LABELS
    _say(f"Without Sononym, the first build also trains a sound model on CLAP and your library's "
         f"own folder and file names (about a minute, once your names label "
         f"{SOUND_TRAIN_MIN_LABELS:,} samples or more), and places with it what no name rule "
         f"recognizes. It stays on this machine; `fourier setup --no-sound-model` turns it off.")


def _model_ready(cached: bool) -> None:
    """Download the pinned CLAP model unless it's `cached`, and load it once to check it."""
    try:
        if not cached:
            console.print("Downloading the CLAP model (pinned to one revision)...", highlight=False)
            download_clap_model()
        load_clap_model()
    except Exception as e:      # a network error, a full disk, a broken install
        console.print(f"The model didn't {'download or ' if not cached else ''}load: {e}. Run "
                      f"`fourier setup` again to retry.", style="red", markup=False, highlight=False)
        return
    console.print("The CLAP model is ready.", style="green", highlight=False)


def _install_clap(w: _Wizard, reinstall_torch: bool = False) -> bool:
    """Install (or update) the [clap] extra with the installer this copy runs under
    (clap_install), offering a too-old uv an update first; then pin the CPU build of PyTorch
    in a uv tool (pin_cpu_torch) and record the install (install.json). True when PyTorch
    and Transformers import afterwards."""
    from .enrich import clap_extra_missing
    tool, cmds = clap_install(reinstall_torch=reinstall_torch)
    if offer_uv_update(w, tool):
        tool, cmds = clap_install(reinstall_torch=reinstall_torch)
    if tool is None:
        console.print("Neither uv nor pip can install into the Python running Fourier "
                      f"({sys.executable}). Run this, then `fourier setup` again:",
                      highlight=False)
        for c in cmds:
            console.print(f"  {shlex.join(c)}", markup=False, highlight=False)
        console.print(f"(uv installs with: {UV_INSTALL})", markup=False, highlight=False)
        return False
    if cpu_only_torch():
        console.print("No NVIDIA GPU here: installing the CPU build of PyTorch.", highlight=False)
    if tool == "uv tool":
        console.print("Fourier runs as a uv tool: reinstalling it with PyTorch and Transformers, "
                      "so `uv tool upgrade` keeps them.", highlight=False)
    for c in cmds:
        _say(f"$ {shlex.join(c)}")
        out: list = []
        if _run(c, keep=out) != 0:
            install_failed(c, out)
            return False
    importlib.invalidate_caches()
    pin_cpu_torch(tool)
    save_install(tool, "cpu" if cpu_only_torch() else "default")
    if clap_extra_missing():
        console.print("PyTorch or Transformers still can't be imported here; run "
                      "`fourier setup` again in a new terminal.", style="red", markup=False,
                      highlight=False)
        return False
    return True


def _llm_step(w: _Wizard, llm, config: Path | None) -> None:
    """Step 5: an optional local LLM (Ollama) for folder descriptions and the audit."""
    w.step(5, "A local LLM (optional)", "A local LLM only writes a short description of each "
                                        "folder into the manifest and powers `fourier tools "
                                        "audit`. Builds don't need it.")
    binary, running, models = ollama_status()
    if not (binary or running):
        console.print("Ollama isn't installed, and that's fine. To add it later, get it from "
                      "https://ollama.com, then run `fourier setup` again.", highlight=False)
        return
    model = describe_model()
    if llm is None:
        llm = w.confirm(f"Ollama is here. Use it for folder descriptions (model {model}, a few "
                        f"GB the first time)?", default=False)
    if not llm:
        console.print("Skipped.", highlight=False)
        return
    if not running:
        console.print(f"Ollama is installed but not running: start it (open the app, or run "
                      f"`ollama serve`), then `ollama pull {model}` and `fourier setup` again.",
                      markup=False, highlight=False)
        return
    if model not in models:
        from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn
        try:
            with Progress(TextColumn("{task.description}"), BarColumn(), DownloadColumn(),
                          console=console) as bar:
                task = bar.add_task(f"ollama pull {model}", total=None)
                ollama_pull(model, lambda done, total, status: bar.update(
                    task, completed=done, total=total or None))
        except (OSError, RuntimeError, ValueError) as e:
            console.print(f"The pull didn't finish: {e}. Try `ollama pull {model}` yourself.",
                          style="red", markup=False, highlight=False)
            return
    if config is None:
        console.print("No config of yours to turn descriptions on in: set DESCRIBE = true under "
                      "[advanced] in your fourier.toml, or build with --describe.", markup=False,
                      highlight=False)
        return
    turn_on_describe(config)
    console.print(f"Folder descriptions are on (DESCRIBE = true in {config}).", markup=False,
                  highlight=False)


def setup_estimate(n_files: int, session=None) -> tuple[float, int]:
    """(seconds, new samples) for a first `fourier build` of a library of n_files:
    analyzing what the database doesn't have yet, then the build."""
    from sqlalchemy import text

    from ..packs.curate_config import BUDGETS, CATEGORIES_OFF
    from ..timings import ANALYZE_STEPS, OWN_SHARE, estimate_analyze, estimate_build
    have = (session.execute(text("SELECT COUNT(*) FROM samples")).scalar() or 0) if session else 0
    if have:                    # this config's library: another's samples in the database don't count
        from ..metadata.rows import outside_library
        have -= len(outside_library(session))
    n_files = max(0, n_files - len(_unreadable_files()))   # never samples: nothing to analyze
    analysed = analysed_count(session)
    new = max(0, max(n_files, have) - analysed)
    # every analysis step a first build runs, and the start-up they spend
    # (the own step reads only the tonal one-shots: OWN_SHARE of them)
    secs, _ = estimate_analyze({s: (round(new * OWN_SHARE) if s == "own" else new) for s in ANALYZE_STEPS},
                               workers=4)
    jobs = max(1, (os.cpu_count() or 2) - 1)
    budgets = {c: b for c, b in BUDGETS.items() if c not in CATEGORIES_OFF}
    n = max(n_files, analysed)
    if n:
        secs += estimate_build(_master_files(budgets, n), jobs, samples=n)[0]
    return secs, new


def _unreadable_files() -> dict:
    """{file: why} under the library folders the last walk couldn't read (still there)."""
    try:
        from ..ingest.importer import scan_errors
        from ..places import library_roots
        return {p: w for p, w in scan_errors(library_roots()).items() if os.path.lexists(p)}
    except Exception:
        return {}


def _master_files(budgets: dict, n: int) -> dict:
    """{category: files} a build of a library of n usable samples makes, about: the budgets
    scaled to the library (packs/scale.py), else as many of them as the samples fill."""
    from ..packs import scale
    from ..timings import picks
    return scale.estimate(budgets, n) if scale.factor(n) < 1 else picks(budgets, n)


def master_size_row(n: int, analyzed: bool = True) -> tuple:
    """The doctor line on how big the master will be for a library of n samples (analyzed,
    else audio files counted before any analysis): scaled down to a small library (scale =
    "library", packs/scale.py), else the style's budgets."""
    from ..packs import scale
    budgets = scale.live_budgets()
    f = scale.factor(n)
    if f < 1:
        # (how many categories it fills takes the samples' homes: `fourier build --dry-run`)
        line = scale.describe(n, sum(scale.estimate(budgets, n).values()), f=f)
    else:
        line = scale.describe(n, sum(budgets.values()))
    if not analyzed:
        line = line.replace("usable samples", "audio files", 1)
    return (OK, "master size", line)


def _check_step(ctx, w: _Wizard, build, target: Path | None = None) -> None:
    """Step 6: doctor's checks, then the first build if everything it needs is there."""
    from ..db.session import db_exists, session_scope
    from ..places import PlacesError, library_roots
    from ..settings import ConfigError
    from ..timings import human
    from ._app import _resolve_config
    from .enrich import clap_extra_missing
    w.step(6, "Check", "What a build needs, and what's still missing.")
    root = ctx.find_root()
    cfg_path, sets = (root.obj or {}).get("config_args", (None, ()))
    if target is not None and target.exists() and not cfg_path:
        # the config setup wrote (`--to`) is the one checked, not the one this run started with
        import os as _os
        from .. import layers, places
        from ..settings import ENV
        cfg_path = str(target)
        _os.environ.pop(ENV, None)
        _os.environ[layers.ENV_CONFIG] = str(target.resolve())
        places.reset()
    try:
        _resolve_config(root, cfg_path, sets)
    except SystemExit:      # the config's error is printed, and doctor's config line says it
        pass
    # (the build time is the estimate below)
    rows = [r for r in doctor_rows() if r[1] != "build time"]
    print_rows(rows)
    blocking = [r for r in rows if r[0] == FAIL]
    n = {lv: sum(1 for r in rows if r[0] == lv) for lv in (OK, WARN, NEXT)}
    n[OK] += sum(1 for r in rows if r[0] == INFO)          # for information: nothing to do
    console.print(f"{n[OK]} OK, {n[WARN]} to look at, {n[NEXT]} for the first build to do"
                  + (f", {len(blocking)} that would stop a build ({', '.join(r[1] for r in blocking)})"
                     if blocking else ", nothing that would stop a build") + ".", highlight=False)
    if blocking or clap_extra_missing():
        console.print("Fix those, then run `fourier setup` again (or `fourier doctor` to "
                      "check).", highlight=False)
        return
    try:
        files, _cloud = survey(library_roots(), COUNT_CAP)
    except (PlacesError, ConfigError):
        files = 0
    if db_exists():
        with session_scope() as s:
            secs, new = setup_estimate(files, s)
    else:
        secs, new = setup_estimate(files)
    big = secs > BIG_BUILD_S
    console.print(f"Ready to build. The first `fourier build` scans the library, analyzes "
                  f"{new:,} new samples and builds: {human(secs)} on this machine. Later builds "
                  f"only analyze what's new.", highlight=False)
    if build is None:
        build = w.confirm("Start it now?", default=not big and not w.yes)
    if not build:
        console.print("Next: fourier build (it can stop and resume), then fourier render "
                      "<device> or fourier sync <device> <card>, and fourier publish.", highlight=False)
        return
    args = [sys.executable, "-m", "fourier"]
    db = root.params.get("db")
    if db:
        args += ["--db", db]
    args += ["build", "--all", "-j", str(max(1, (os.cpu_count() or 2) - 1))]
    console.print(f"$ fourier {' '.join(args[3:])}", markup=False, highlight=False)
    raise SystemExit(_run(args))


# --- doctor ------------------------------------------------------------------------------
def check_config() -> list[tuple]:
    from .. import layers
    from ..settings import ConfigError
    try:
        path = layers.find_config()
    except ConfigError as e:
        return [(FAIL, "config", str(e))]
    if path is None:
        return [(WARN, "config", "no fourier.toml (fourier setup writes one); the code's defaults")]
    try:
        r = layers.resolve()
    except ConfigError as e:
        return [(FAIL, "config", f"{path}: {e}")]
    doc = layers._read(path)
    preset = doc.get("preset") or f"{layers.DEFAULT_PRESET} (named by none: the default)"
    known = {"library", "devices", "preset", "overlay", "providers", "preparers", "sononym_db",
             "output", "advanced", "add_categories"} | set(__import__("fourier.knobs", fromlist=["KNOBS"]).KNOBS)
    extra = sorted(set(doc) - known)
    own = sum(1 for src in r.sources.values() if not str(src).startswith("preset"))
    out = [(OK, "config", f"{path} (style {preset}"
                          + (f", and {own} setting{'' if own == 1 else 's'} of your own" if own else "")
                          + "; `fourier config show` prints what it sets)")]
    if extra:
        out.append((WARN, "config keys", f"not keys fourier reads: {', '.join(extra)}"))
    return out


def check_library(session) -> list[tuple]:
    """The library folders, and with a database (session) its schema and samples; without
    one (a new machine), that nothing is scanned yet."""
    from sqlalchemy import text

    from ..places import PlacesError, library, library_roots
    try:
        roots, names = library()
    except PlacesError as e:
        return [(FAIL, "library", str(e))]
    if not roots and not names:
        return [(FAIL, "library", "no library in fourier.toml (fourier setup)")]
    out = []
    roots = library_roots()
    for r in roots:
        if not r.is_dir():
            out.append((FAIL, "library folder", f"{r}: missing"))
        elif not os.access(r, os.R_OK | os.X_OK):
            out.append((FAIL, "library folder", f"{r}: can't be read (its permissions)"))
        else:
            out.append((OK, "library folder", str(r)))
    from ..ingest.importer import unavailable_links
    for link, target in unavailable_links([r for r in roots if r.is_dir()], session):
        # what the scan says, and why a build stops (cli/ingest.scan_report, cli/build.py)
        out.append((FAIL, "linked folder",
                    f"A linked folder is unavailable: {link} -> {target} (a drive that isn't plugged "
                    f"in?). A build without it would leave out every sample there, so it stops until "
                    f"it's back, or builds without it with --allow-missing."))
    has_audio = survey(roots, 1)[0] > 0
    if not has_audio and any(r.is_dir() for r in roots):
        from ..ingest.formats import readable_exts
        out.append((FAIL, "audio files", f"none in the library folders ({', '.join(readable_exts())})"))
    elif has_audio:
        out += check_thin(roots)
        out.append(check_layout(session, roots))
    if session is None:
        from ..db.session import db_path_in_use
        return out + [(NEXT, "samples in the database",
                       f"none yet at {db_path_in_use()}: the first `fourier build` scans the library")]
    from ..db.migrations import LATEST, current
    v = current(session.get_bind())
    out.append((OK if v == LATEST else WARN, "database schema",
                f"version {v}" + ("" if v == LATEST else f"; this code knows {LATEST}: update fourier"
                                  if v > LATEST else f"; {LATEST} is current")))
    from ..metadata.rows import library_unscanned, outside_library
    n = session.execute(text("SELECT COUNT(*) FROM samples")).scalar() or 0
    other = len(outside_library(session)) if n else 0
    n -= other
    if not n and other and library_unscanned(session):
        out.append((NEXT, "samples in the database",
                    "none of this library's yet (the database holds another library's): the first "
                    "`fourier build` scans it"))
    else:
        out.append((OK if n else NEXT, "samples in the database",
                    f"{n:,}" + ("" if n else ": the first `fourier build` scans the library")))
    bad = _unreadable_files()
    if bad:
        shown = "; ".join(f"{p} ({w})" for p, w in list(bad.items())[:3])
        out.append((WARN, "unreadable files",
                    f"{len(bad):,} audio file{'' if len(bad) == 1 else 's'} the last scan couldn't read "
                    f"(corrupt or empty), left out of every build: {shown}"
                    + (f"; ... and {len(bad) - 3:,} more" if len(bad) > 3 else "")))
    if other:
        out.append((INFO, "other libraries",
                    f"{other:,} more sample{'' if other == 1 else 's'} in this home's database "
                    f"{'is' if other == 1 else 'are'} under none of this config's library folders "
                    f"(another config's library): builds, these counts and search leave them out"))
    return out


def check_layout(session, roots) -> tuple:
    """One line on whose vendor cap a sample counts toward (packs/vendors.py): with vendors =
    "auto" the layout detected in each library folder (from the database once scanned, else
    from the files), with "first-folder" that the first folder is the vendor."""
    from ..packs import vendors as V
    if V.mode() != V.AUTO:
        return (OK, "library layout", "vendors = \"first-folder\": the first folder under the library "
                                      "folder is each sample's vendor (\"auto\" detects the layout)")
    return (OK, "library layout", V.describe(V.layouts(session, roots)))


CLOUD_SCAN_MAX = 300_000      # audio files doctor checks for cloud-only placeholders


def check_cloud_only(roots=None, cap: int = CLOUD_SCAN_MAX) -> list[tuple]:
    """How many of the library's audio files a cloud drive keeps only online (one lstat each,
    the first `cap` files of a larger library)."""
    from ..places import PlacesError, library_roots
    if roots is None:
        try:
            roots = library_roots()
        except PlacesError:
            return []
    if not any(os.path.isdir(r) for r in roots):
        return []
    seen, cloud = survey(roots, cap)
    part = f" (the first {seen:,} checked)" if seen >= cap else ""
    if not cloud:
        return [(OK, "cloud-only files", f"none of {seen:,} audio files{part}")]
    return [(WARN, "cloud-only files",
             f"{cloud:,} of {seen:,} audio files{part} aren't on this machine: a build downloads "
             f"the ones it picks and stops if it can't, analysis skips them (fourier tools "
             f"analyze --download fetches them). {CLOUD_HINT}")]


def check_providers(session) -> list[tuple]:
    from ..ingest.ableton_tags import latest_live_db
    from ..metadata import providers as P
    from ..places import sononym_db
    out = []
    sdb = sononym_db()
    out.append((OK, "Sononym", f"{sdb}" if sdb and sdb.exists() else "not found (optional)"))
    live = latest_live_db()
    out.append((OK, "Live's file index", live or "not found (optional)"))
    if session is None:
        return out
    try:
        a = P.active(session)
        out.append((OK, "classifier", "the built-in path and audio providers" if a.fallback
                    else ", ".join(a.classifiers)))
    except Exception as e:           # a provider named in fourier.toml without its data
        out.append((FAIL, "providers", str(e)))
    return out


# the analysis steps whose coverage a column shows (the others run on a subset of samples)
ANALYSIS_COLUMNS = (("derived", "derived_computed_at"), ("librosa", "mfcc_mean"),
                    ("clap", "clap_embedding"))


def pending_analysis(session=None) -> dict:
    """{step: samples it hasn't analyzed yet}, for the steps a column shows."""
    from sqlalchemy import text

    from ..db.session import session_scope
    def count(s):
        total = s.execute(text("SELECT COUNT(*) FROM samples")).scalar() or 0
        out = {}
        for step, col in ANALYSIS_COLUMNS:
            try:
                n = s.execute(text(f"SELECT COUNT(*) FROM sample_features WHERE {col} IS NOT NULL")).scalar() or 0
            except Exception:
                n = 0
            out[step] = max(0, total - n)
        from .enrich import own_pending           # a share of the library: its own count
        out["own"] = own_pending(s)
        return out
    if session is not None:
        return count(session)
    try:
        with session_scope() as s:
            return count(s)
    except Exception:
        return {}


def check_analysis(session) -> list[tuple]:
    """Each step's coverage: NEXT while it isn't done (a build analyzes what's new first)."""
    from sqlalchemy import text

    from ..timings import estimate_analyze, human
    if session is None:
        return []
    total = session.execute(text("SELECT COUNT(*) FROM samples")).scalar() or 0
    if not total:
        return []
    pending = pending_analysis(session)

    out = [(OK if pending[step] == 0 else NEXT, f"analysis: {step}",
            f"{total - pending[step]:,} of {total:,}"
            + ("" if pending[step] == 0 else ": `fourier build` analyzes the rest first"))
           for step, _col in ANALYSIS_COLUMNS]
    left = {k: v for k, v in pending.items() if v}
    if left:
        secs, basis = estimate_analyze(left, workers=4)
        out.append((OK, "analysis time left", f"{human(secs)} with 4 workers, from {basis}"))
    return out


def analysed_count(session) -> int:
    """Samples with a CLAP embedding, less the ones a walk marked missing: the ones a build
    can place (metadata.rows.usable_count, what the library scale counts)."""
    if session is None:
        return 0
    from ..metadata.rows import usable_count
    return usable_count(session)


def check_build_time(session=None) -> list[tuple]:
    """How long a whole build takes: the budgets, or the files the analyzed library can
    fill when it holds fewer samples, at this machine's last rates (else rough defaults).
    Before any analysis, the first build's time from the library's file count (what setup
    says)."""
    import os as _os

    from ..packs.curate_config import BUDGETS, CATEGORIES_OFF
    from ..places import PlacesError, library_roots
    from ..settings import ConfigError
    from ..timings import estimate_build, human
    n = analysed_count(session)
    if not n:
        try:
            files, _cloud = survey(library_roots(), COUNT_CAP)
        except (PlacesError, ConfigError):
            files = 0
        if not files:
            return []
        secs, new = setup_estimate(files, session)
        return [(OK, "build time", f"{human(secs)} for the first `fourier build`, which "
                                   f"analyzes {new:,} new samples first (from rough reference rates)"),
                master_size_row(files, analyzed=False)]
    jobs = max(1, (_os.cpu_count() or 2) - 1)
    budgets = {c: b for c, b in BUDGETS.items() if c not in CATEGORIES_OFF}
    can = _master_files(budgets, n)
    secs, basis = estimate_build(can, jobs, samples=n)
    size = ("" if can == budgets else
            f" (up to {round(sum(can.values())):,} files from {n:,} analyzed samples)")
    return [(OK, "build time", f"{human(secs)} for fourier build -j {jobs}{size}, from {basis}"),
            master_size_row(n)]


def check_clap(session) -> list[tuple]:
    """The [clap] extra (a build embeds its prompts with the model, and analyze its samples)
    and the index: FAIL without either; WARN when the index is behind the database (a
    build refreshes it)."""
    from ..paths import clap_index_path
    from .enrich import CLAP_INSTALL, clap_extra_missing
    have = not clap_extra_missing()
    if not have and receipt_dropped_clap(uv_tool_receipt()):
        out = [(FAIL, "CLAP model", "CLAP was removed by a reinstall: run `fourier setup`")]
    elif not have:
        out = [(FAIL, "CLAP model", f"not installed: {CLAP_INSTALL} (a build needs it)")]
    elif not clap_model_cached():
        out = [(NEXT, "CLAP model", "the software is installed; the model (about 600 MB, once) "
                                    "downloads on the first build, or now with `fourier setup`")]
    else:
        out = [(OK, "CLAP model", "installed")]
    if have and cpu_only_torch() and torch_has_cuda():
        out.append((WARN, "PyTorch build", "a CUDA build, with NVIDIA libraries this machine can't "
                                           "use (no NVIDIA GPU): run `fourier setup` to switch "
                                           "to the smaller CPU build"))
    idx = clap_index_path()
    n_db = analysed_count(session)
    if idx.exists():
        from ..analysis.clap_features import load_index_fast
        n_idx = len(load_index_fast(ids_only=True)[0])
        out.append((OK if n_idx >= n_db else WARN, "CLAP index",
                    (f"{n_idx:,} of {n_db:,} embeddings" if n_idx <= n_db else
                     f"{n_idx:,} embeddings, {n_db:,} of them this library's usable samples")
                    + ("" if n_idx >= n_db else " (a build refreshes it)")))
    elif n_db:
        out.append((WARN, "CLAP index", f"none at {idx} yet; a build makes it from the {n_db:,} embeddings"))
    else:
        out.append((NEXT, "CLAP index", f"none yet at {idx}: the first `fourier build` makes it"))
    return out


# the package's generic profiles: unverified on purpose (no device, no manual), so doctor
# names them as INFO, where a device's own unverified profile is a WARN
GENERIC_PROFILES = ("generic_folder", "generic_sd_card", "generic_44k", "generic_48k")


def check_devices(devices: list[str]) -> list[tuple]:
    from ..devices.loader import DeviceLoader
    from ..knobs import limit_problem
    from ..packs.curate_config import CATEGORY_ORDER
    if not devices:
        return [(WARN, "devices", "none in fourier.toml (devices = [...])")]
    out = []
    for d in devices:
        try:
            p = DeviceLoader().load(d)
        except Exception as e:
            out.append((FAIL, f"device: {d}", str(e)))
            continue
        why = limit_problem(p, CATEGORY_ORDER)
        if why:                      # a build stops on it (cli/build.py)
            out.append((FAIL, f"device: {d}", why))
            continue
        detail = f"{p.name}, {p.sample_rate} Hz / {p.bit_depth}-bit, load: {p.load}"
        generic = d in GENERIC_PROFILES
        if not p.verified:
            detail += ("; a generic starting point, unverified by design (it describes no device)"
                       if generic else "; unverified profile (no manual behind it)")
        if p.load == "card-sync" and p.card_dir:
            from ..platforms import mount_points
            mounted = [v for v in mount_points() if (v / p.card_dir.lstrip("/")).is_dir()]
            detail += f"; mounted at {mounted[0]}" if mounted else "; card not mounted"
        out.append((OK if p.verified else INFO if generic else WARN, f"device: {d}", detail))
        for msg in p.warnings:
            out.append((WARN, f"device: {d}", msg))
    return out


def check_output() -> list[tuple]:
    from ..places import master_dir, publish_root
    out = []
    for label, p in (("master", master_dir()), ("publish root", publish_root())):
        near = Path(p)
        while not near.exists() and near != near.parent:      # the nearest folder that exists
            near = near.parent
        ok = near.is_dir() and os.access(near, os.W_OK)
        note = "" if Path(p).exists() else " (will be created)"
        out.append((OK if ok else FAIL, f"output: {label}", p + (note if ok else f": can't write in {near}")))
        if label == "master":
            from ..places import master_library_problem
            got = master_library_problem(p)
            if got and got[0] == "fail":       # a build stops on it before anything is written
                out.append((FAIL, "output: master", got[1]))
    return out


def doctor_rows() -> list[tuple]:
    """(level, what, detail) for everything a build needs, without creating a database on a
    new machine."""
    from ..db.session import db_exists, session_scope
    from ..places import PlacesError
    from ..places import devices as configured_devices
    try:
        devs = configured_devices()
    except PlacesError as e:
        devs = []
        console.print(f"[red]{e}[/red]")
    rows = check_config()
    if db_exists():
        with session_scope() as session:
            rows += check_library(session) + check_cloud_only() + check_providers(session)
            rows += check_analysis(session)
            rows += check_clap(session)
            rows += check_build_time(session)
    else:       # a new machine: say what's missing without creating the database
        rows += check_library(None) + check_cloud_only() + check_providers(None)
        rows += check_clap(None) + check_build_time(None)
    return rows + check_devices(devs) + check_output()


@main.command("doctor", short_help="Check what a build needs and estimate how long it takes.")
@click.pass_context
def doctor(ctx):
    """Check what a build needs: the config, the library and its database (and how many of
    its files a cloud drive keeps only online), the providers (Sononym, Live, the built-in
    fallback), the analysis, the CLAP model and index, and the devices, and estimate the
    analysis and build time (from this machine's last runs, else built-in reference rates).
    OK is ready, WARN worth a look, NEXT what the first build does itself (scan, analyze, index,
    download the model) and FAIL what would stop a build; it exits non-zero only on a FAIL.

    \b
      fourier doctor
    """
    err = (ctx.find_root().obj or {}).get("config_error")
    if err:                        # nothing else can be checked until the config loads
        print_rows([(FAIL, "config", f"{err}. `fourier config edit` opens it to fix")])
        raise SystemExit(1)
    from ._app import read_only_config
    read_only_config()
    rows = doctor_rows()
    print_rows(rows)
    if any(r[0] == NEXT for r in rows) and not any(r[0] == FAIL for r in rows):
        console.print("NEXT lines are what the first `fourier build` does itself.", highlight=False)
    if any(r[0] == FAIL for r in rows):
        raise SystemExit(1)


def print_rows(rows) -> None:
    """One line per check, never hard-wrapped (a path in it isn't broken mid-path when the
    output is narrow or piped; the terminal folds a long line)."""
    from rich.markup import escape
    for level, what, detail in rows:
        c = LEVEL_STYLE[level]
        console.print(f"[{c}]{level:<4}[/{c}]  {escape(what)}: {escape(str(detail))}", highlight=False,
                      soft_wrap=True)
