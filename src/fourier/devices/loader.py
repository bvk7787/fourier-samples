"""Device profiles: config/devices/<id>.yaml, one schema for every device.

    id: m8_tracker
    name: Dirtywave M8 Tracker
    manual:                             # the manual the citations quote (none: unverified)
      title: M8 Operation Manual
      version: v20260421
      url: https://...
      sha256: 0063...                   # of the PDF, so a test can find it under any file name
    load: card-sync                     # card-sync | transfer | copy
    sample_refs: path                   # path: a release locks its on-card paths
    paths:
      card_dir:         {value: /Samples/Fourier, status: convention}
      folder_depth:     {value: 2, status: convention}
      files_per_folder: {value: 128, status: convention}
      max_path_length:  {value: 127, cite: path-length}
    audio:
      sample_rate: {value: 44100, cite: render-format}
      ...
    storage_mb: {value: 32000, status: convention}
    citations:
      path-length: {claim: ..., page: 74, quote: "..."}   # page: the PDF page (1-based)

Every value says where it comes from: `cite` (a citation id: a PDF page and a verbatim quote
from the manual, checked by tests/test_device_citations.py), or `status` (`convention`: a
Fourier choice, not a device limit; `unverified`: not checked against a manual). A profile
with `status: unverified` at the top (the generic ones) makes no claim about any device.

Where profiles come from: the package's config/devices, then your own: each folder in
$FOURIER_DEVICES (os.pathsep-separated; `none` turns your own off), then
~/.config/fourier/devices (`fourier devices new` writes there). A profile of your own
never shadows a package id by accident: it takes a package id only with `override: true`
at the top, and is otherwise reported (`fourier devices list`) and skipped. Among your own
folders the first one wins. `extends: <id>` starts a profile from another one (its manual,
citations and values) and changes only what it sets (docs/device-profiles.md).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default location for device profiles shipped with the project
from ..paths import config_dir  # noqa: E402

DEFAULT_DEVICES_DIR = config_dir() / "devices"
USER_DEVICES_ENV = "FOURIER_DEVICES"
USER_DEVICES_DIR = Path("~/.config/fourier/devices")


def user_devices_dirs() -> list[Path]:
    """Your own profile folders, in the order they win: each of $FOURIER_DEVICES
    (os.pathsep-separated), then ~/.config/fourier/devices. FOURIER_DEVICES=none: none."""
    env = os.environ.get(USER_DEVICES_ENV)
    if env is not None and env.strip().lower() == "none":
        return []
    dirs = [Path(x).expanduser() for x in (env or "").split(os.pathsep) if x.strip()]
    default = USER_DEVICES_DIR.expanduser()
    return dirs + ([default] if default not in dirs else [])


def user_devices_dir() -> Path:
    """Where `fourier devices new` writes: the first of $FOURIER_DEVICES, else
    ~/.config/fourier/devices."""
    dirs = user_devices_dirs()
    return dirs[0] if dirs else USER_DEVICES_DIR.expanduser()


LOADS = ("card-sync", "transfer", "copy")
SAMPLE_REFS = ("path", "embedded")
STATUSES = ("convention", "unverified")
BIT_DEPTHS = (8, 16, 24, 32)
SAMPLE_RATE_RANGE = (8000, 192000)       # Hz: what a profile may set
# the rates converters and devices commonly use; another rate in the range loads with a warning
COMMON_SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000,
                       176400, 192000)
WRITABLE_FORMATS = ("wav", "aiff", "aif")     # what a render can write (packs/render.py)
TOP_KEYS = {"id", "name", "description", "manual", "load", "sample_refs", "status", "paths",
            "audio", "storage_mb", "citations", "override", "extends"}
MANUAL_KEYS = ("title", "version", "url", "sha256")
# section -> {key: (DeviceProfile field, default)}
FACTS = {
    "paths": {"root": ("root", ""), "card_dir": ("card_dir", ""),
              "folder_depth": ("folder_depth", 2), "files_per_folder": ("files_per_folder", 128),
              "max_path_length": ("max_path_length", None),
              "max_name_length": ("max_name_length", None),
              "ascii_names": ("ascii_names", None)},
    "audio": {"sample_rate": ("sample_rate", 44100), "bit_depth": ("bit_depth", 16),
              "channels": ("channels", "stereo"), "formats": ("formats", ["wav"]),
              "max_slices": ("max_slices", None), "max_duration_s": ("max_duration_s", None),
              "dither": ("dither", True)},
    None: {"storage_mb": ("storage_mb", None)},
}


class DeviceProfileError(ValueError):
    pass


@dataclass
class DeviceProfile:
    device_id: str
    name: str = ""
    description: str = ""
    manual: dict[str, str] | None = None   # {title, version, url, sha256} of the cited manual
    load: str = "copy"
    sample_refs: str = "path"
    verified: bool = True
    # paths
    root: str = ""                    # the folder the render sits in, under card_dir ("" = none)
    card_dir: str = ""                # where a render goes on the card, e.g. "/Samples/Fourier"
    folder_depth: int = 2             # folder levels under card_dir, root included
    files_per_folder: int | None = 128
    max_path_length: int | None = None    # the longest file path the device takes, card_dir included
    max_name_length: int | None = None    # a render cuts longer file and folder names (in the middle)
    # None: names as the master has them; False: NFC-normalized; True: NFC and plain ASCII
    ascii_names: bool | None = None
    # audio
    sample_rate: int = 44100
    bit_depth: int = 16
    channels: str = "stereo"          # stereo keeps each file's channels; mono folds everything
    formats: list[str] = field(default_factory=lambda: ["wav"])
    max_slices: int | None = None
    max_duration_s: float | None = None
    dither: bool = True
    storage_mb: float | None = None
    # where each value comes from: {"paths.max_path_length": {"cite": ...} | {"status": ...}}
    sources: dict[str, dict] = field(default_factory=dict)
    citations: dict[str, dict] = field(default_factory=dict)
    yaml_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    origin: str = "package"           # package | user (a profile of your own)
    extends: str | None = None        # the profile this one starts from
    warnings: list[str] = field(default_factory=list)   # values that load but look unusual

    @property
    def mono(self) -> bool:
        return self.channels == "mono"

    @classmethod
    def from_dict(cls, raw: dict, path: Path | None = None) -> DeviceProfile:
        where = path.name if path else raw.get("id", "profile")
        if not isinstance(raw, dict) or not raw.get("id"):
            raise DeviceProfileError(f"{where}: a profile needs an id")
        unknown = set(raw) - TOP_KEYS
        if unknown:
            raise DeviceProfileError(f"{where}: unknown keys {sorted(unknown)}")
        if raw.get("load", "copy") not in LOADS:
            raise DeviceProfileError(f"{where}: load {raw.get('load')!r} (one of {', '.join(LOADS)})")
        if raw.get("sample_refs", "path") not in SAMPLE_REFS:
            raise DeviceProfileError(f"{where}: sample_refs {raw.get('sample_refs')!r}")
        cites = raw.get("citations") or {}
        verified = raw.get("status") != "unverified"
        manual = raw.get("manual")
        if manual is not None:
            if not isinstance(manual, dict) or set(manual) != set(MANUAL_KEYS):
                raise DeviceProfileError(f"{where}: manual: give {{{', '.join(MANUAL_KEYS)}}}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(manual["sha256"])):
                raise DeviceProfileError(f"{where}: manual: sha256 must be 64 lowercase hex digits")
            manual = {k: str(manual[k]) for k in MANUAL_KEYS}
        if cites and manual is None:
            raise DeviceProfileError(f"{where}: citations need the manual they quote (manual:)")
        kw, sources = {}, {}
        for section, keys in FACTS.items():
            block = raw if section is None else (raw.get(section) or {})
            if section is not None:
                extra = set(block) - set(keys)
                if extra:
                    raise DeviceProfileError(f"{where}: {section}: unknown keys {sorted(extra)}")
            for key, (attr, _default) in keys.items():
                if key not in block:
                    continue
                fact = block[key]
                name = f"{section}.{key}" if section else key
                if not isinstance(fact, dict) or not ({"cite", "status"} & set(fact)):
                    raise DeviceProfileError(f"{where}: {name}: give {{value, cite}} or {{value, status}}")
                if "cite" in fact and fact["cite"] not in cites:
                    raise DeviceProfileError(f"{where}: {name}: no citation {fact['cite']!r}")
                if "status" in fact and fact["status"] not in STATUSES:
                    raise DeviceProfileError(f"{where}: {name}: status {fact['status']!r} "
                                             f"(one of {', '.join(STATUSES)})")
                if "cite" in fact and not verified:
                    raise DeviceProfileError(f"{where}: {name}: an unverified profile cites nothing")
                if "value" in fact:
                    kw[attr] = fact["value"]
                sources[name] = {k: v for k, v in fact.items() if k != "value"}
        if kw.get("channels", "stereo") not in ("stereo", "mono"):
            raise DeviceProfileError(f"{where}: audio.channels must be stereo or mono")
        bd = kw.get("bit_depth", 16)
        if isinstance(bd, bool) or bd not in BIT_DEPTHS:
            raise DeviceProfileError(f"{where}: audio.bit_depth {bd!r}: one of "
                                     f"{', '.join(map(str, BIT_DEPTHS))}")
        warnings = []
        sr = kw.get("sample_rate", 44100)
        lo, hi = SAMPLE_RATE_RANGE
        if isinstance(sr, bool) or not isinstance(sr, int) or not lo <= sr <= hi:
            raise DeviceProfileError(f"{where}: audio.sample_rate {sr!r}: a whole number of Hz from "
                                     f"{lo} to {hi} (44100 and 48000 are the usual ones)")
        if sr not in COMMON_SAMPLE_RATES:
            warnings.append(f"audio.sample_rate {sr} Hz is unusual (the usual ones: "
                            f"{', '.join(map(str, COMMON_SAMPLE_RATES))}): check the manual")
        fd = kw.get("folder_depth", 2)
        if isinstance(fd, bool) or not isinstance(fd, int) or fd < 1:
            raise DeviceProfileError(f"{where}: paths.folder_depth {fd!r}: a whole number of folder "
                                     f"levels, 1 or more")
        for key in ("root", "card_dir"):
            if key in kw and not isinstance(kw[key], str):
                raise DeviceProfileError(f"{where}: paths.{key} {kw[key]!r}: a folder path (text)")
        if "dither" in kw and not isinstance(kw["dither"], bool):
            raise DeviceProfileError(f"{where}: audio.dither {kw['dither']!r}: true or false")
        v = kw.get("storage_mb")
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
            raise DeviceProfileError(f"{where}: storage_mb {v!r}: a number of MB above 0 (or null)")
        fmts = kw.get("formats", ["wav"])
        if not isinstance(fmts, list) or not any(str(f).lower() in WRITABLE_FORMATS for f in fmts):
            raise DeviceProfileError(f"{where}: audio.formats must list wav or aiff "
                                     f"(what a render writes)")
        for key in ("max_name_length", "max_path_length", "files_per_folder", "max_slices"):
            v = kw.get(key)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                raise DeviceProfileError(f"{where}: {key} must be a whole number above 0 (or null)")
        if kw.get("max_name_length") is not None and kw["max_name_length"] < 8:
            raise DeviceProfileError(f"{where}: max_name_length under 8 leaves no room for a name")
        v = kw.get("max_duration_s")
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
            raise DeviceProfileError(f"{where}: max_duration_s must be a number of seconds above 0 "
                                     f"(or null)")
        if kw.get("ascii_names") is not None and not isinstance(kw["ascii_names"], bool):
            raise DeviceProfileError(f"{where}: paths.ascii_names must be true or false")
        if "extends" in raw:
            raise DeviceProfileError(f"{where}: extends: {raw['extends']!r} is resolved by "
                                     f"DeviceLoader; load the profile through it")
        return cls(device_id=raw["id"], name=raw.get("name", raw["id"]),
                   description=raw.get("description", ""), manual=manual,
                   load=raw.get("load", "copy"), sample_refs=raw.get("sample_refs", "path"),
                   verified=verified, sources=sources, citations=cites, yaml_path=path, raw=raw,
                   warnings=warnings, **kw)

    @classmethod
    def from_yaml(cls, path: Path) -> DeviceProfile:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f), path)

    def source(self, name: str) -> str:
        """Where a value comes from, in words: 'manual PDF p.74', 'convention', 'unverified'."""
        s = self.sources.get(name)
        if not s:
            return "default"
        if "cite" in s:
            c = self.citations[s["cite"]]
            return f"manual PDF p.{c['page']}" + (" (convention)" if s.get("status") == "convention" else "")
        return s["status"]

    def summary(self) -> str:
        def fmt(name, v, unit=""):
            v = ", ".join(v) if isinstance(v, list) else v
            shown = "-" if v in (None, "") else f"{v}{unit}"
            return f"    {name:<24} {shown:<18} {self.source(name)}"
        manual = f"{self.manual['title']} {self.manual['version']}" if self.manual else "-"
        lines = [f"Device: {self.name} ({self.device_id})" + ("" if self.verified else "  UNVERIFIED"),
                 f"  Manual: {manual}   Load: {self.load}   Sample refs: {self.sample_refs}"]
        if self.origin != "package" or self.extends:
            lines.append(f"  From: {'your own profile ' + str(self.yaml_path) if self.origin == 'user' else 'the package'}"
                         + (f" (extends {self.extends})" if self.extends else ""))
        lines.append("  Paths:")
        for key, (attr, _d) in FACTS["paths"].items():
            lines.append(fmt(f"paths.{key}", getattr(self, attr)))
        lines.append("  Audio:")
        for key, (attr, _d) in FACTS["audio"].items():
            lines.append(fmt(f"audio.{key}", getattr(self, attr)))
        lines.append(fmt("storage_mb", self.storage_mb, " MB").replace("    storage_mb", "  storage_mb  ", 1))
        return "\n".join(lines)


def _merge(base: dict, child: dict) -> dict:
    """A profile that `extends` another: the child's top-level values win, and its paths,
    audio and citations add to or replace the base's key by key."""
    out = {k: v for k, v in base.items() if k not in ("override", "extends")}
    for k, v in child.items():
        if k in ("paths", "audio", "citations") and isinstance(v, dict):
            out[k] = {**(base.get(k) or {}), **v}
        elif k != "extends":
            out[k] = v
    return out


class DeviceLoader:
    """Discovers and loads device profile YAML files: the package's, then your own
    (user_devices_dirs; see the module docstring for which one wins)."""

    def __init__(self, devices_dir: Path | None = None, user_dirs: list[Path] | None = None):
        self.devices_dir = devices_dir or DEFAULT_DEVICES_DIR
        # an explicit devices_dir (a test's folder) reads only that folder unless user_dirs
        # are given too
        if user_dirs is None:
            user_dirs = [] if devices_dir is not None else user_devices_dirs()
        self.user_dirs = [Path(d) for d in user_dirs]

    @staticmethod
    def _yaml_files(folder: Path):
        if not folder.is_dir():
            return []
        return [p for p in sorted(folder.glob("*.yaml")) if not p.name.startswith(("_", "."))]

    def _paths(self):
        return self._yaml_files(self.devices_dir)

    @staticmethod
    def _read(p: Path):
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception:
            return None

    def _index(self) -> tuple[dict[str, tuple[Path, str]], list[tuple[Path, str]]]:
        """({id: (yaml path, "package" | "user")}, [(yaml path, problem)])."""
        index, problems, package = {}, [], set()
        for p in self._paths():
            raw = self._read(p)
            did = (raw or {}).get("id", p.stem) if isinstance(raw, dict) else p.stem
            if did not in index:
                index[did] = (p, "package")
                package.add(did)
        mine: dict[str, Path] = {}
        for folder in self.user_dirs:
            for p in self._yaml_files(folder):
                raw = self._read(p)
                if not isinstance(raw, dict):
                    problems.append((p, "not a YAML mapping (a device profile)"))
                    continue
                did = raw.get("id") or p.stem
                if did in mine:
                    problems.append((p, f"id {did!r} is already defined by {mine[did]}; this one is skipped"))
                    continue
                if did in package and raw.get("override") is not True:
                    problems.append((p, f"id {did!r} is a package profile; give yours another id, "
                                        f"or put `override: true` at the top to replace it"))
                    continue
                mine[did] = p
                index[did] = (p, "user")
        return index, problems

    def problems(self) -> list[tuple[Path, str]]:
        """Profiles of your own that are skipped, and why."""
        return self._index()[1]

    def list_devices(self) -> list[str]:
        """Return available device IDs: the package's, then your own."""
        index, _ = self._index()
        return list(index)

    def _raw(self, device_id: str, index, seen=()) -> tuple[dict, Path, str, str | None]:
        if device_id not in index:
            raise ValueError(f"No device profile {device_id!r}. The profiles: {', '.join(index)}; "
                             f"`fourier devices list` describes them, and `fourier devices new` "
                             f"makes one for your sampler.")
        p, origin = index[device_id]
        raw = self._read(p)
        if not isinstance(raw, dict):
            raise DeviceProfileError(f"{p.name}: not a YAML mapping (a device profile)")
        base_id = raw.get("extends")
        if base_id is None:
            return raw, p, origin, None
        if base_id in seen or base_id == device_id and origin == "package":
            raise DeviceProfileError(f"{p.name}: extends {base_id!r} goes round in a circle")
        base_index = index
        if base_id == device_id:            # an override that extends the package profile
            base_index = {k: v for k, v in index.items()}
            base_index[device_id] = next(((q, "package") for q in self._paths()
                                          if (self._read(q) or {}).get("id", q.stem) == device_id),
                                         (None, None))
            if base_index[device_id][0] is None:
                raise DeviceProfileError(f"{p.name}: extends {base_id!r}: no package profile has that id")
        if base_id not in base_index:
            raise DeviceProfileError(f"{p.name}: extends {base_id!r}: no profile has that id")
        base, _bp, _bo, _be = self._raw(base_id, base_index, (*seen, device_id))
        return _merge(base, raw), p, origin, base_id

    def load(self, device_id: str) -> DeviceProfile:
        """Load a device profile by ID."""
        index, _ = self._index()
        raw, p, origin, base = self._raw(device_id, index)
        prof = DeviceProfile.from_dict(raw, p)
        prof.origin, prof.extends = origin, base
        return prof

    def load_all(self) -> dict[str, DeviceProfile]:
        """Load all device profiles."""
        out = {}
        for device_id in self.list_devices():
            try:
                out[device_id] = self.load(device_id)
            except Exception:
                pass
        return out
