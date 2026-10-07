"""Render the curated master to a device-ready image (Stage B).

The master (CATEGORY/family/file WAV tree from `fourier build`) is the
device-agnostic source of truth. This renders a disposable, regenerable copy
tailored to one device's constraints: sample rate, bit depth, mono downmix,
FAT32-safe names, folder-depth limits, and an optional size budget.

Conversion reuses the existing `devices/exporter._convert_and_copy` (load ->
resample -> mono -> dither -> write). No time-stretch (the master is one-shots
and whole loops; slicing is done on the device).

Profile fields a render follows only when the profile sets them (so a profile that leaves
them unset renders exactly as before, byte for byte and path for path):

  audio.formats       without wav: AIFF files (.aif) (devices/convert.py)
  audio.bit_depth 8   8-bit PCM, TPDF-dithered (devices/convert.py)
  audio.max_duration_s  longer files are cut there with a 5 ms fade (never a WAVES cycle)
  audio.max_slices    below the master's slice grid (sets.SLICE_MAX): the SLICE set leaves
                      out the loops whose grid needs more slices than the device takes
  paths.max_name_length  longer file and folder names are cut in the middle; a name a
                      cut makes equal to another's gets "_2"
  paths.ascii_names   set (true or false): names NFC-normalized; true: also plain ASCII
                      (NFKD, accents dropped, a few letters spelled out, the rest "_")
  paths.files_per_folder  a folder over the limit continues in numbered siblings
                      ("punchy", "punchy-2", ...), filled in the master's order; a path lock's
                      paths never move, and new files fill the room its folders have left
"""
from __future__ import annotations

import hashlib
import os
import shutil
import time
from functools import lru_cache
from pathlib import Path

from ..devices.exporter import _convert_and_copy, _sanitize_filename, _strip_boilerplate  # noqa: F401
from ..devices.loader import DeviceLoader
from . import manifests

# Render cache (~/.fourier/cache/render): a converted file is kept under a key of its source
# audio (md5), the conversion settings and the exporter's code, and hardlinked into the next
# render. Conversion is deterministic (dither is seeded from the content), so a hit is the
# same bytes a fresh conversion would write; a render that changes few files takes seconds.
# Misses convert in RENDER_JOBS processes. FOURIER_NO_RENDER_CACHE=1 turns the cache off.
RENDER_JOBS = min(8, os.cpu_count() or 1)
RENDER_POOL_MIN = 64           # fewer misses than this convert in-process (pool start-up costs more)
RENDER_PRUNE_DAYS = 14


def _render_cache_dir() -> Path | None:
    if os.environ.get("FOURIER_NO_RENDER_CACHE", "") in ("1", "true", "yes"):
        return None
    from ..paths import home_path
    return Path(os.environ.get("FOURIER_RENDER_CACHE") or home_path("cache", "render"))


@lru_cache(maxsize=1)
def _exporter_fingerprint() -> str:
    import inspect
    import numpy
    import scipy
    import soundfile
    from ..devices import exporter
    parts = [inspect.getsource(exporter), numpy.__version__, scipy.__version__, soundfile.__version__]
    for mod in ("soxr", "librosa"):
        try:
            parts.append(__import__(mod).__version__)
        except Exception:
            parts.append(f"{mod}=none")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_md5s(master: Path) -> dict:
    """{absolute master path: out_md5} from the master's manifest (categories and sets)."""
    try:
        man = manifests.read(master / "manifest.json")
    except (OSError, ValueError):
        return {}
    out = {}
    for group in ("categories", "sets"):
        for name, cd in (man.get(group) or {}).items():
            for e in cd.get("entries", []):
                if e.get("out_md5"):
                    out[str(master / name / e["out"])] = e["out_md5"]
    return out


def _render_key(digest: str, opts: dict) -> str:
    return hashlib.sha256(repr((digest, sorted(opts.items()), _exporter_fingerprint())).encode()).hexdigest()


def _cache_put(dest: Path, cpath: Path) -> None:
    try:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_name(f".{cpath.name}.{os.getpid()}.tmp")
        try:
            os.link(dest, tmp)
        except OSError:
            shutil.copy2(dest, tmp)
        os.replace(tmp, cpath)
    except OSError:
        pass


def _convert_job(args):
    """Convert one file (a pool worker); returns (index, error or None)."""
    i, src, dest, opts, cpath = args
    try:
        if "convert" in opts:          # 8-bit, AIFF or a length limit (devices/convert.py)
            from ..devices.convert import convert
            convert(src, dest, **{k: v for k, v in opts.items() if k != "convert"})
        else:
            _convert_and_copy(src, dest, **opts)
    except Exception as e:  # reported by the caller
        return i, str(e) or type(e).__name__
    if cpath:
        _cache_put(Path(dest), Path(cpath))
    return i, None


def prune_render_cache(days=RENDER_PRUNE_DAYS, log=print) -> int:
    """Drop cached renders no render links to any more that weren't used for `days` days."""
    d = _render_cache_dir()
    if not d or not d.exists():
        return 0
    cut, n = time.time() - days * 86400, 0
    for p in d.glob("*/*.wav"):
        try:
            st = p.stat()
            if st.st_nlink == 1 and st.st_atime < cut and st.st_mtime < cut:
                p.unlink()
                n += 1
        except OSError:
            pass
    if n:
        log(f"render cache: pruned {n} unused entries")
    return n


def _device_root(device) -> str:
    """The folder the render sits in under card_dir (e.g. 'SAMPLES'; paths.root), or ""."""
    return (device.root or "").strip("/")


def _allowed_sub(device) -> int:
    """Subfolder levels left for CATEGORY/family under the device root (root counts as one)."""
    return max(1, (device.folder_depth or 2) - (1 if _device_root(device) else 0))


_ASCII_SPELLED = {"ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O",
                  "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D",
                  "ı": "i", "‐": "-", "–": "-", "—": "-", "‘": "'", "’": "'", "“": "'", "”": "'",
                  "×": "x", "µ": "u", "°": "deg"}


def _ascii(name: str) -> str:
    """name in plain ASCII: NFKD splits accented letters (the accents are dropped), a few
    letters without a decomposition are spelled out (_ASCII_SPELLED), anything else is "_"."""
    import unicodedata
    out = []
    for ch in unicodedata.normalize("NFKD", name):
        if ord(ch) < 128:
            out.append(ch)
        elif unicodedata.combining(ch):
            continue
        else:
            out.append(_ASCII_SPELLED.get(ch, "_"))
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "_"


def _names_apply(device) -> bool:
    """Whether the profile sets a name rule (paths.ascii_names, paths.max_name_length)."""
    return device is not None and (device.ascii_names is not None or bool(device.max_name_length))


def _device_name(name: str, device, ext: str = "") -> str:
    """A file or folder name under the profile's name rules: NFC (and ASCII with
    ascii_names: true), then cut in the middle to max_name_length, extension included."""
    import unicodedata
    from ..devices.exporter import _truncate_middle
    if device.ascii_names is not None:
        name = unicodedata.normalize("NFC", name)
        if device.ascii_names:
            name = _ascii(name)
    if device.max_name_length and len(name) + len(ext) > device.max_name_length:
        name = _truncate_middle(name, device.max_name_length - len(ext))
    return name + ext


def _out_format(device) -> str:
    from ..devices.convert import out_format
    return out_format(device.formats)


def _ext(device) -> str:
    from ..devices.convert import extension
    return extension(_out_format(device)) if device is not None else ".wav"


def _rel_dest(category: str, family: str, stem: str, allowed_sub: int, device=None) -> Path:
    """Where a master file lands in the render, relative to the device root. The category
    folder is numbered in play order (curate_config.category_dir: '01_KICKS')."""
    from .curate_config import category_dir
    ext = _ext(device)
    if ext != ".wav" or _names_apply(device):
        cdir = _device_name(_sanitize_filename(category_dir(category)), device)
        if allowed_sub >= 2:
            return (Path(cdir) / _device_name(_sanitize_filename(family, None), device)
                    / _device_name(_sanitize_filename(stem, None), device, ext))
        return Path(cdir) / _device_name(_sanitize_filename(f"{family}__{stem}", None), device, ext)
    cdir = _sanitize_filename(category_dir(category))
    # the file's name is the master's (curate: exporter.canonical_stem), the same on every
    # device: a render never shortens it ("learn the library once")
    if allowed_sub >= 2:
        return Path(cdir) / _sanitize_filename(family, None) / f"{_sanitize_filename(stem, None)}.wav"
    # fold family into the filename to respect the depth limit
    return Path(cdir) / f"{_sanitize_filename(f'{family}__{stem}', None)}.wav"


def _card_prefix(device) -> str:
    """card_dir + device root, e.g. '/Samples/Fourier' or 'SAMPLES'."""
    return "/".join(x for x in (device.card_dir.rstrip("/"), _device_root(device)) if x)


RENDER_MARKER = ".fourier-render"
# single-cycle waveforms: rendered sample-exact (relabelled, never resampled) so they
# loop seamlessly in the M8's OSC/FWDLOOP play modes (manual p. 61) and the Digitakt II's
# FORWARD LOOP (manual p. 70)
CYCLE_CATEGORIES = {"WAVES"}


def _check_out_dir(base: Path, master: Path):
    """Refuse to wipe anything that isn't a previous render: the master itself, a folder that
    contains or sits inside it, or a non-empty folder without the render marker."""
    b, m = base.resolve(), master.resolve()
    if b == m or m.is_relative_to(b) or b.is_relative_to(m):
        raise RuntimeError(f"render output {base} overlaps the source {master}; pick a separate --out")
    if b.exists() and any(b.iterdir()) and not (b / RENDER_MARKER).exists():
        raise RuntimeError(f"{base} is not empty and isn't a Fourier render (no {RENDER_MARKER}); "
                           f"refusing to delete it. Remove it yourself or pick another --out")


def _render_fmt(device) -> dict:
    """The audio format a path lock records; the format and length limit only when the
    profile sets them (a lock made without them stays valid)."""
    c = device
    fmt = dict(sample_rate=c.sample_rate, bit_depth=c.bit_depth,
               mono=device.mono)
    if _out_format(device) != "wav":
        fmt["format"] = _out_format(device)
    if device.max_duration_s:
        fmt["max_duration_s"] = device.max_duration_s
    return fmt


def expected_subtype(device) -> str:
    """The soundfile subtype every rendered file has (PCM_16, PCM_24, PCM_U8 ...)."""
    from ..devices.convert import subtype
    return subtype(device.bit_depth, _out_format(device))


def _convert_opts(device, cat: str) -> dict:
    """The conversion settings for one file. A profile that asks for 8-bit, AIFF or a length
    limit adds keys (and the convert module's fingerprint, so the render cache follows its
    code); any other profile gets exactly the settings it always had."""
    opts = dict(target_sr=device.sample_rate, target_bit_depth=device.bit_depth,
                convert_to_mono=device.mono, dither=device.dither,
                collapse_dual_mono=not device.mono, preserve_length=cat in CYCLE_CATEGORIES)
    extra: dict[str, object] = {}
    if _out_format(device) != "wav":
        extra["out_format"] = _out_format(device)
    if device.max_duration_s and cat not in CYCLE_CATEGORIES:
        extra["max_duration_s"] = float(device.max_duration_s)
    if extra or device.bit_depth not in (16, 24, 32):
        opts.update(extra, convert=_convert_fingerprint())
    return opts


@lru_cache(maxsize=1)
def _convert_fingerprint() -> str:
    import inspect
    from ..devices import convert
    return hashlib.sha256(inspect.getsource(convert).encode()).hexdigest()[:16]


def _slice_overflow(master: Path, device) -> set[str]:
    """SLICE set files (master paths) whose loop needs a finer slice grid than the device
    takes: the set judged each loop on min(SLICE_MAX, its 16ths) equal slices
    (sets.slice_metrics), so only a profile with max_slices under SLICE_MAX can lose any."""
    from . import sets
    if not device.max_slices or device.max_slices >= sets.SLICE_MAX:
        return set()
    try:
        man = manifests.read(master / "manifest.json")
    except (OSError, ValueError):
        return set()
    bars = {f"{cat}/{e['out']}": e.get("bars") for cat, cd in (man.get("categories") or {}).items()
            for e in cd.get("entries", [])}
    out = set()
    for e in ((man.get("sets") or {}).get("SLICE") or {}).get("entries", []):
        b = bars.get(e.get("came_from") or "")
        if not b:
            continue
        n = int(min(sets.SLICE_MAX, max(1, int(round(float(b) * 4)) * sets.SLICE_STEP // 4)))
        if n > device.max_slices:
            out.add(f"SLICE/{e['out']}")
    return out


def _lock_mismatch(lock, device) -> str | None:
    """Why this lock's files would not come out the same (path or audio format) under the
    profile as it is now, if they wouldn't."""
    from .device_lock import norm_card_dir

    if lock.fmt and lock.fmt != _render_fmt(device):
        return f"audio format: locked {lock.fmt}, profile {_render_fmt(device)}"
    if lock.card_dir != norm_card_dir(device.card_dir):
        return f"card_dir: locked {lock.card_dir!r}, profile {norm_card_dir(device.card_dir)!r}"
    if (lock.root or "") != _device_root(device):
        return f"device root: locked {lock.root!r}, profile {_device_root(device)!r}"
    return None


FIT_MIN = 8          # the fewest characters a folder or file name is cut to for a path limit


def _cuts(names: list[str], over: int) -> list[int] | None:
    """How many characters to cut from each name (to FIT_MIN at least) to take `over` off
    their total, the longer name first and the rest in proportion to what each can spare;
    None when they can't spare that many."""
    spare = [max(0, len(n) - FIT_MIN) for n in names]
    if sum(spare) < over:
        return None
    cuts, left = [0] * len(names), over
    for i in sorted(range(len(names)), key=lambda i: -spare[i]):      # the longer name first
        share = min(spare[i], -(-over * spare[i] // max(sum(spare), 1)), left)
        cuts[i], left = share, left - share
    for i in range(len(names)):
        extra = min(left, spare[i] - cuts[i])
        cuts[i], left = cuts[i] + extra, left - extra
    return cuts


def _fit_all(rels: list[Path], prefix: str, limit: int | None) -> list[Path]:
    """A render keeps the master's names on every device the master was sized for
    (`devices` in fourier.toml: STEM_MAX and FAMILY_NAME_MAX fit the tightest path limit),
    so a folder whose paths are all within the limit is never changed. A device the master
    wasn't sized for (one rendered without being in `devices`) still gets paths within its
    limit, cut per folder: a family folder over the limit gets one shorter name, sized from
    its longest path (the folder and that file name cut in the middle in proportion, to
    FIT_MIN characters at least), and each of its files is then cut as much as it still
    needs, the extension kept. A folder holding a path that can't fit is left as it is, and
    the render stops on it (render_master_to_device)."""
    if not limit:
        return list(rels)
    from ..devices.exporter import _truncate_middle
    pre = len(prefix) + 1 if prefix else 0
    out = list(rels)
    groups: dict[tuple, list[int]] = {}
    for i, rel in enumerate(rels):
        groups.setdefault(rel.parts[:-1], []).append(i)
    for parent, idx in groups.items():
        over = {i: pre + len(rels[i].as_posix()) - limit for i in idx}
        if max(over.values()) <= 0:
            continue
        split = [os.path.splitext(rels[i].parts[-1]) for i in idx]
        stems = {i: st for i, (st, _e) in zip(idx, split)}
        exts = {i: e for i, (_s, e) in zip(idx, split)}
        if len(parent) < 2:                  # no family folder: the file name alone
            for i in idx:
                if over[i] > 0:
                    c = _cuts([stems[i]], over[i])
                    if c:
                        out[i] = Path(*parent, _truncate_middle(stems[i], len(stems[i]) - c[0]) + exts[i])
            continue
        folder = parent[-1]
        spare_f = max(0, len(folder) - FIT_MIN)
        if any(over[i] > spare_f + max(0, len(stems[i]) - FIT_MIN) for i in idx):
            continue                 # a path here can't fit: the folder stays whole, the render stops
        top = max(idx, key=lambda i: (over[i], -i))          # the longest path sizes the folder
        c = _cuts([folder, stems[top]], over[top])
        cut_f = c[0] if c else spare_f
        # enough off the folder that every file can fit with its own name cut to FIT_MIN
        cut_f = min(spare_f, max([cut_f] + [over[i] - max(0, len(stems[i]) - FIT_MIN) for i in idx]))
        name_f = _truncate_middle(folder, len(folder) - cut_f)
        for i in idx:
            need = over[i] - cut_f
            st = stems[i]
            if need > 0:
                st = _truncate_middle(st, len(st) - min(need, max(0, len(st) - FIT_MIN)))
            out[i] = Path(*parent[:-1], name_f, st + exts[i])
    return out


def _assignment(master_dir, device, lock=None):
    """(files, Assignment) for rendering master_dir to device, honoring its path lock. The
    Assignment's `shortened` is (family folders, file names) cut to fit the path limit."""
    from .device_lock import assign, master_files

    sub, prefix, limit = _allowed_sub(device), _card_prefix(device), device.max_path_length
    files = master_files(master_dir)
    over = _slice_overflow(Path(master_dir), device)
    if over:
        files = [f for f in files if f.master not in over]
    # the profile's own name rules and extension (None/".wav" for a profile that sets neither)
    rules = device if (_names_apply(device) or _ext(device) != ".wav") else None
    whole = {}
    for f in files:
        whole.setdefault(f.key, _rel_dest(f.category, f.family, f.stem, sub, rules))
    keys = list(whole)
    fitted = dict(zip(keys, _fit_all([whole[k] for k in keys], prefix, limit)))
    fresh = lambda f: fitted[f.key].as_posix()
    a = assign(files, lock, fresh, limit_len=limit,
               prefix_len=len(prefix) + 1 if prefix else 0,
               name_len=device.max_name_length, folder_limit=device.files_per_folder)
    cut = [k for k in a.new if fitted[k] != whole[k]]
    a.shortened = (len({whole[k].parent for k in cut if fitted[k].parent != whole[k].parent}),
                   sum(1 for k in cut if fitted[k].name != whole[k].name))
    return files, a


def shortened_note(device, shortened) -> str | None:
    """The line render and `render --check` print when names were cut to fit the device's
    path limit, else None."""
    folders, names = shortened or (0, 0)
    if not folders and not names:
        return None
    what = " and ".join(x for x in (
        f"{folders:,} family folder{'' if folders == 1 else 's'}" if folders else "",
        f"{names:,} file name{'' if names == 1 else 's'}" if names else "") if x)
    return (f"{what} cut to fit {device.device_id}'s {device.max_path_length}-character path limit "
            f"(each folder one name): the master's names weren't sized for it. Add "
            f"{device.device_id} to devices in fourier.toml and build again for names that fit "
            f"without cuts.")


def _full(prefix: str, rel: str) -> str:
    return f"{prefix}/{rel}" if prefix else rel


def card_paths(master_dir, device, lock=None, assignment=None) -> list[str]:
    """Full on-card path of every file the render would write: card_dir + device root +
    CATEGORY/family/name, exactly as the render names them (cut per folder to the path
    limit on a device the master wasn't sized for, _fit_all; a path still over it is
    reported), plus retired locked files, which stay on the card."""
    files, a = assignment or _assignment(master_dir, device, lock)
    prefix = _card_prefix(device)
    rels = [a.paths[f.key] for f in files if f.key in a.paths]
    rels += [lock.files[k]["path"] for k in a.retired] if lock else []
    return sorted({_full(prefix, r) for r in rels})


def path_violations(master_dir, device, lock=None) -> list[tuple[str, int]]:
    """On-card paths longer than the device's max_path_length, longest first."""
    limit = device.max_path_length
    if not limit:
        return []
    bad = [(p, len(p)) for p in card_paths(master_dir, device, lock) if len(p) > limit]
    return sorted(bad, key=lambda x: -x[1])


def _release_copy(entry: dict) -> Path | None:
    """The original audio for a locked file: its master path inside the release it was locked from."""
    from .releases import RELEASES_ROOT

    if not entry.get("release"):
        return None
    p = Path(RELEASES_ROOT) / entry["release"] / entry["master"]
    return p if p.exists() else None


def render_master_to_device(master_dir, device, out_dir, max_mb=None, log=print, lock=None,
                            release=None, write_lock=False, lock_dir=None):
    """Render every WAV under master_dir into a device image under out_dir. Returns a summary dict.

    With a path lock (see packs/device_lock.py), locked files keep their on-card path, keep
    their original audio (taken from the release they were locked from if the master's copy
    changed), and retired locked files are rendered from their release too, so the image is
    a superset of what the device already has. write_lock records new paths (use it when
    rendering a release you're about to transfer)."""
    from .device_lock import record, save_lock

    master = Path(master_dir)
    if not master.is_dir():
        raise RuntimeError(f"no master at {master} yet: run `fourier build` first"
                           if release is None else f"release not found at {master}: `fourier releases` lists them")
    c = device
    why = _lock_mismatch(lock, device) if lock else None
    if why:
        raise RuntimeError(f"{device.device_id}: files already on the device would change ({why}). "
                           f"Restore the profile or start a new lock.")
    root = _device_root(device)
    base = Path(out_dir) / root if root else Path(out_dir)
    _check_out_dir(base, master)
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    (base / RENDER_MARKER).write_text(f"{device.device_id}\n")

    budget =int(max_mb * 1024 * 1024) if max_mb else None  # SD holds more than RAM; no cap unless asked

    files, a = _assignment(master, device, lock)
    prefix = _card_prefix(device)
    too_long = sorted((p for p in (_full(prefix, r) for r in a.paths.values())
                       if c.max_path_length and len(p) > c.max_path_length), key=lambda p: -len(p))
    if too_long:          # never written: a device can't load a path over its limit
        shutil.rmtree(base, ignore_errors=True)
        raise RuntimeError(
            f"{device.device_id}: {len(too_long)} path(s) would exceed its {c.max_path_length}-character "
            f"limit on {device.card_dir or 'the card root'} even with their names cut (longest "
            f"{len(too_long[0])}): " + "; ".join(too_long[:5]) + (" ..." if len(too_long) > 5 else "")
            + ". Raise the profile's paths.max_path_length if the device allows it, or put a "
            "shorter card folder in it; nothing was rendered.")

    note = shortened_note(device, a.shortened)
    if note:
        log(f"  {note}")

    # jobs: (key, source audio, device-relative path, category, locked?)
    jobs, kept_orig, missing, queued = [], 0, [], set()
    for f in files:
        if f.key in queued:  # same source twice in one category: assign() gave it one path
            continue
        queued.add(f.key)
        src = master / f.master
        if f.key in a.changed:
            orig = _release_copy(lock.files[f.key])
            if orig:
                src, kept_orig = orig, kept_orig + 1
            else:
                log(f"  WARNING: {a.paths[f.key]} is locked but its audio changed and the original "
                    f"release copy is missing; rendering the new audio")
        jobs.append((f.key, src, a.paths[f.key], f.category, f.key not in a.new))
    for k in a.retired:
        e = lock.files[k]
        orig = _release_copy(e)
        if orig:
            # the master category ("WAVES"), not the numbered device folder ("16_WAVES"):
            # a retired cycle must stay sample-exact too
            jobs.append((k, orig, e["path"], (e.get("master") or e["path"]).split("/")[0], True))
        else:
            missing.append(e["path"])
    jobs.sort(key=lambda j: (not j[4], j[2]))  # locked first: a size budget never drops them

    total = 0; nfiles = 0; skipped = 0; failed = 0
    per_cat: dict[str, int] = {}
    written: set[str] = set()
    # convert (cache hits are hardlinked, misses converted in parallel), then apply the
    # size budget in job order, as a one-by-one render would
    cdir = _render_cache_dir()
    md5s = _manifest_md5s(master) if cdir else {}
    errors, todo, hits = {}, [], 0
    for i, (key, src, rel, cat, locked) in enumerate(jobs):
        dest = base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        opts = _convert_opts(device, cat)
        cpath = None
        if cdir:
            try:
                digest = md5s.get(str(src)) or _md5_file(src)
                k = _render_key(digest, opts)
                cpath = cdir / k[:2] / f"{k}.wav"
                if cpath.exists():
                    try:
                        os.link(cpath, dest)
                    except OSError:
                        shutil.copy2(cpath, dest)
                    hits += 1
                    continue
            except OSError:
                cpath = None
        todo.append((i, str(src), str(dest), opts, str(cpath) if cpath else None))
    if len(todo) >= RENDER_POOL_MIN and RENDER_JOBS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=RENDER_JOBS) as ex:
            results = list(ex.map(_convert_job, todo, chunksize=16))
    else:
        results = [_convert_job(t) for t in todo]
    errors = {i: e for i, e in results if e}
    if cdir:
        log(f"  render cache: {hits} reused, {len(todo)} converted")
    for i, (key, src, rel, cat, locked) in enumerate(jobs):
        dest = base / rel
        if i in errors:
            failed += 1
            log(f"  {'ERROR (locked path, projects on the device use it)' if locked else 'FAILED'}: "
                f"{rel} from {src}: {errors[i]}")
            continue
        sz = dest.stat().st_size
        if budget is not None and not locked and total + sz > budget:
            dest.unlink(missing_ok=True); skipped += 1
            continue
        total += sz; nfiles += 1; per_cat[cat] = per_cat.get(cat, 0) + 1
        written.add(key)
    for cat in sorted(per_cat):
        log(f"  {cat}: {per_cat[cat]} files")
    if lock:
        log(f"  lock: {len(a.kept)} locked paths kept, {len(a.new)} new, {len(a.retired)} retired "
            f"(kept on the card), {len(a.changed)} with changed audio ({kept_orig} rendered from "
            f"their original release)")
    if missing:
        log(f"  WARNING: {len(missing)} locked files are gone from the master and their release copy "
            f"is missing, so they are NOT in this image (e.g. {missing[0]})")
    if write_lock:
        if not release:
            raise RuntimeError("write_lock needs the release being rendered (render from a release)")
        new_written = [k for k in a.new if k in written]
        p = save_lock(record(lock, device.device_id, device.card_dir, files, a, release,
                             written=written, root=root, fmt=_render_fmt(device)), lock_dir)
        log(f"  locked {len(new_written)} new paths -> {p}")

    if cdir:
        prune_render_cache(log=log)
    mb = total / 1024 / 1024
    room = c.storage_mb
    note = ""
    if room and mb > room:
        note = (f" (image is {mb:.0f}MB; {device.name} holds {room:.0f}MB, "
                f"so load a subset)")
    n_sets = sum(n for cat, n in per_cat.items() if cat in ("KITS", "SLICE"))
    split = (f" ({nfiles - n_sets:,} from the categories, {n_sets:,} in the kit and slice sets)"
             if n_sets else "")
    log(f"DONE {device.device_id}: {nfiles:,} audio files{split}, {mb:.1f} MB -> {base}{note}")
    return dict(device=device.device_id, files=nfiles, mb=round(mb, 1), skipped=skipped, failed=failed,
                fmt=f"{c.sample_rate}Hz/{c.bit_depth}bit/{c.channels}", out=str(base), ram_note=note.strip(),
                new=len(a.new), kept=len(a.kept), retired=len(a.retired), changed=len(a.changed),
                missing=len(missing))


def device_plan(master_dir, device, lock, log=print) -> dict:
    """What rendering master_dir would do to the paths already on the device (per its lock)."""
    prefix = _card_prefix(device)
    files, a = _assignment(master_dir, device, lock)
    if lock is None:
        where = f"under {prefix} on the card" if prefix else "on the device"
        log(f"{device.device_id}: nothing locked on this device yet; rendering a release locks "
            f"the {len(a.new):,} file paths it puts {where}, so later releases keep them")
        return dict(device=device.device_id, locked=0, new=len(a.new), safe=True)
    mismatch = _lock_mismatch(lock, device)
    card_ok = mismatch is None
    changed_lost = [k for k in a.changed if not _release_copy(lock.files[k])]
    retired_lost = [k for k in a.retired if not _release_copy(lock.files[k])]
    safe = card_ok and not changed_lost and not retired_lost
    log(f"{device.device_id}: {len(lock.files)} locked paths under {lock.card_dir or '/'}")
    log(f"  kept      : {len(a.kept)}   (same path; unchanged audio unless listed below)")
    log(f"  NEW       : {len(a.new)}   (fresh names that avoid every locked path)")
    log(f"  CHANGED   : {len(a.changed)}   (master audio differs; {len(a.changed) - len(changed_lost)} "
        f"will keep their original audio from the release they were locked from)")
    log(f"  RETIRED   : {len(a.retired)}   (gone from the master; {len(a.retired) - len(retired_lost)} "
        f"stay on the card from their release)")
    if not card_ok:
        log(f"  PROFILE   : {mismatch} -> every file on the device would change")
    for k in (changed_lost + retired_lost)[:6]:
        log(f"       at risk: {_full(prefix, lock.files[k]['path'])} (original release copy missing)")
    log("  => " + ("SAFE: every path on the device keeps its audio" if safe
                   else "NOT SAFE: some device paths would lose or change their audio"))
    return dict(device=device.device_id, locked=len(lock.files), kept=len(a.kept), new=len(a.new),
                changed=len(a.changed), retired=len(a.retired),
                at_risk=len(changed_lost) + len(retired_lost), card_dir_ok=card_ok, safe=safe)


def render_device(device_id, master_dir, out_dir, max_mb=None, log=print, release=None,
                  write_lock=False, lock_dir=None):
    """Convenience wrapper: load the device profile and its path lock, then render."""
    from .device_lock import load_lock

    device = DeviceLoader().load(device_id)
    lock = load_lock(device_id, lock_dir)
    log(f"Rendering {'release ' + release if release else 'master'} -> {device.name} "
        f"({device.sample_rate}Hz/{device.bit_depth}bit/"
        f"{device.channels}){'; path lock: ' + str(len(lock.files)) + ' files' if lock else ''}")
    for msg in device.warnings:
        log(f"  WARNING: {device.device_id}: {msg}")
    return render_master_to_device(master_dir, device, out_dir, max_mb=max_mb, log=log, lock=lock,
                                   release=release, write_lock=write_lock, lock_dir=lock_dir)
