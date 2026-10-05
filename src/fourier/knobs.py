"""Knobs: the plain settings in fourier.toml (and in presets and overlays), each turned
into the tunables it stands for (docs/design-history.md, "How it became configurable").

    # fourier.toml
    preset = "balanced"
    devices = ["m8_tracker"]                    # also sizes the names for the tightest path limit
    categories = { WAVES = "off", VOX = 0.5 }   # off, on, or a budget weight
    tempo = "85-180"                            # the 5-BPM loop bands span this range, and
                                                # loops fold by octaves into its top octave
    fold = "auto"                               # auto (as tempo says) | off (no octave folding)
    vendors = "auto"                            # first-folder | auto (each library's layout)
    words = { KICKS = ["bombo"] }               # your words for a category (without Sononym)
    sets = "on"                                 # on | off (00_KITS and 00_SLICE)
    size = "auto"                               # auto (fits the devices, pools' surplus moves), fixed, "5GB"
    scale = "library"                           # the master follows the library's size (or "off")
    files = 6000                                # or a file count (instead of size; scale off)
    loudness = "standard"                       # gentle | standard | hot
    retune = "detect"                           # off | named | detect
    stereo = "fold-near-mono"                   # keep | fold-near-mono | mono
    names = "canonical"                         # canonical | keep
    [sources]
    favor = ["classic breaks", "(?:^|/)my kits/"]   # regexes on the file path
    home = { "one-shot drums" = "KICKS" }           # pack -> its only category
    vendor_max = 0.4

In each layer (the preset chain, the overlay, fourier.toml) the knobs apply first and its
[advanced] table after, so an [advanced] tunable beats a knob in the same file. A knob at
its default value sets nothing. `fourier config show` names the knob behind each value.
"""
from __future__ import annotations

import re

from .settings import ConfigError

KNOBS: dict = {}


def knob(name):
    def wrap(fn):
        KNOBS[name] = fn
        return fn
    return wrap


def _choice(name, value, choices):
    if value not in choices:
        raise ConfigError(f"{name} = {value!r}: one of {', '.join(map(repr, choices))}")
    return value


@knob("devices")
def devices(value, known) -> dict:
    """The devices a build is for (config/devices/<id>.yaml). Their tightest path limit sizes
    the master's names (STEM_MAX, FAMILY_NAME_MAX), which every device shares: the limit
    less the card folder, the longest category folder, "/", ".wav" and room for a "_2",
    split between family and file name as the defaults split it (44:46 for the M8's 127).
    A profile whose limits leave too little room for names (device_problems) sizes nothing:
    doctor names it as a FAIL and a build stops on it, while every other command runs."""
    from .devices.loader import DeviceLoader
    ids = [value] if isinstance(value, str) else value
    if not isinstance(ids, list) or not all(isinstance(d, str) for d in ids):
        raise ConfigError("devices: a list of device ids (fourier devices list)")
    loader = DeviceLoader()
    known_ids = loader.list_devices()
    bad = [d for d in ids if d not in known_ids]
    if bad:
        raise ConfigError(f"devices: unknown {bad} (known: {', '.join(known_ids)})")
    order = known["curate_config.CATEGORY_ORDER"]
    room, names, slices = [], [], []
    for d in ids:
        p = loader.load(d)
        if limit_problem(p, order):
            continue
        if p.max_path_length:
            room.append(path_room(p, order))
        if p.max_name_length:
            names.append(p.max_name_length)
        if p.max_slices:
            slices.append(p.max_slices)
    out = {}
    if room:
        family, stem = split_room(min(room))
        if stem < known["curate_config.STEM_MAX"]:
            out["curate_config.STEM_MAX"] = stem
        if family < known["curate_config.FAMILY_NAME_MAX"]:
            out["curate_config.FAMILY_NAME_MAX"] = family
    # a profile's longest file name (paths.max_name_length): the master's names fit it with
    # ".wav" and a "_2" (NAME_ROOM); only ever down, and only when a profile sets one
    if names:
        stem = min(names) - NAME_ROOM
        if stem < out.get("curate_config.STEM_MAX", known["curate_config.STEM_MAX"]):
            out["curate_config.STEM_MAX"] = stem
    # the SLICE set's grid follows the device with the fewest slices (audio.max_slices), when
    # one takes fewer than the grid; never up
    k = "sets.SLICE_MAX"
    if slices and k in known and min(slices) < known[k]:
        out[k] = int(min(slices))
    return out


def longest_category(order) -> int:
    """The longest numbered category folder ("08_DRUMLOOPS") and its "/"."""
    return max((len(f"{i + 1:02d}_{c}") for i, c in enumerate(order)), default=0) + 1


def card_prefix(card_dir: str, root: str = "") -> str:
    return "/".join(x for x in ((card_dir or "").rstrip("/"), (root or "").strip("/")) if x)


def path_room(p, order, limit: int | None = None) -> int:
    """The characters a profile's path limit (or `limit`) leaves for a family folder and a
    file name: the limit less the card folder and device root, the longest category folder,
    "/", ".wav" and a "_2"."""
    prefix = card_prefix(p.card_dir, getattr(p, "root", ""))
    limit = p.max_path_length if limit is None else limit
    return limit - (len(prefix) + 1 if prefix else 0) - longest_category(order) - 1 - 4 - 2


def split_room(total: int) -> tuple[int, int]:
    """(family, file name) characters of a path room, split as the defaults split it."""
    fam0, stem0 = FAMILY_SPLIT
    family = int(round(total * fam0 / (fam0 + stem0)))
    return family, total - family


NAME_MIN = 12                # the fewest characters a path limit may leave a family folder or file name
STEM_MIN = 8                 # the fewest characters a name limit may leave a file name


def min_room() -> int:
    """The smallest path room that leaves NAME_MIN characters to both names."""
    total = 2 * NAME_MIN
    while min(split_room(total)) < NAME_MIN:
        total += 1
    return total


def limit_problem(p, order) -> str | None:
    """Why a profile's limits leave too little room for the master's names, with the fix, or
    None. The path limit must leave min_room() characters past its card folder and the
    longest category folder; the name limit STEM_MIN past ".wav" and a "_2"."""
    where = getattr(p, "yaml_path", None) or f"{p.device_id}.yaml"
    if p.max_path_length:
        total = path_room(p, order)
        if total < min_room():
            need = p.max_path_length + min_room() - total
            prefix = card_prefix(p.card_dir, getattr(p, "root", ""))
            return (f"its path limit of {p.max_path_length} leaves {max(total, 0)} characters for "
                    f"folder and file names (after {prefix or 'the card root'} and the category "
                    f"folders, up to {longest_category(order) - 1} characters); the names need "
                    f"{min_room()}. Set paths.max_path_length to at least {need} (or 0: no limit) "
                    f"in {where}, or take {p.device_id} out of devices = [...] in fourier.toml")
    if p.max_name_length and p.max_name_length - NAME_ROOM < STEM_MIN:
        return (f"its name limit of {p.max_name_length} leaves {p.max_name_length - NAME_ROOM} "
                f"characters for a name (after \".wav\" and a \"_2\"); the names need {STEM_MIN}. "
                f"Set paths.max_name_length to at least {STEM_MIN + NAME_ROOM} (or 0: no limit) in "
                f"{where}, or take {p.device_id} out of devices = [...] in fourier.toml")
    return None


def device_problems(ids=None) -> list[tuple[str, str]]:
    """[(device id, problem)] for the configured devices (or these) whose limits leave too
    little room for the master's names (limit_problem): what doctor FAILs and a build stops
    on."""
    from .devices.loader import DeviceLoader
    from .packs.curate_config import CATEGORY_ORDER
    if ids is None:
        from .places import PlacesError
        from .places import devices as configured
        try:
            ids = configured()
        except PlacesError:
            return []                 # the config's own error says so
    loader = DeviceLoader()
    out = []
    for d in ids:
        try:
            p = loader.load(d)
        except Exception:
            continue                  # doctor's device line says why it didn't load
        why = limit_problem(p, CATEGORY_ORDER)
        if why:
            out.append((d, why))
    return out


NAME_ROOM = 6                # ".wav" and a "_2" after a master file name


FAMILY_SPLIT = (44, 46)      # family : file name, the defaults' split


@knob("categories")
def categories(value, known) -> dict:
    """{CATEGORY: "off" | "on" | weight}: off leaves the category out of the build (its
    folder number stays free), on brings back one an earlier layer switched off, a weight
    scales its budget."""
    if not isinstance(value, dict):
        raise ConfigError("categories: a table of CATEGORY = off, on or a weight")
    order = known["curate_config.CATEGORY_ORDER"]
    budgets = dict(known["curate_config.BUDGETS"])
    was = set(known["curate_config.CATEGORIES_OFF"])
    off = set(was)
    for cat, v in value.items():
        c = str(cat).upper()
        if c not in order:
            raise ConfigError(f"categories: {cat!r} is not a category ({', '.join(order)})")
        if v in ("off", False):
            off.add(c)
        elif v in ("on", True):
            off.discard(c)                  # on again, over an earlier layer's off
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
            if c in budgets:
                budgets[c] = int(round(budgets[c] * float(v)))
        else:
            raise ConfigError(f"categories: {cat} = {v!r}: off, on or a weight (0 or more)")
    out = {}
    if off != was:
        out["curate_config.CATEGORIES_OFF"] = off
    if budgets != known["curate_config.BUDGETS"]:
        out["curate_config.BUDGETS"] = budgets
    return out


TEMPO_FLOOR, TEMPO_CEIL = 60, 201      # the outermost band edges (DRUMLOOPS keeps 60-200 BPM)


def tempo_bands(lo: int, hi: int) -> tuple:
    """5-BPM bands from lo to hi, two coarse ones below (60, 75) and one above (hi + 10). A
    range that isn't a multiple of 5 BPM wide ("118-140") has its inner edges on multiples of
    5 (118, 120, 125, ... 140)."""
    below = [e for e in (TEMPO_FLOOR, 75) if e < lo]
    above = [e for e in (hi + 10,) if e < TEMPO_CEIL]
    if (hi - lo) % 5:
        mid = [lo] + [b for b in range(lo - lo % 5 + 5, hi, 5)] + [hi]
    else:
        mid = list(range(lo, hi + 1, 5))
    return tuple(below + mid + above + [TEMPO_CEIL])


def fold_window(lo: int, hi: int) -> tuple:
    """(window, range) the loops' tempos fold into: the range's top octave [hi / 2, hi), and
    the range itself when it doesn't hold that whole octave (lo above hi / 2): a fold that
    would leave the range doesn't happen, so a loop stays at its own tempo (a 174 BPM loop
    in a 118-140 style isn't halved to 87; in 70-100 a 120 isn't halved to 60). A range that
    holds the octave (85-180, 70-180) folds exactly as the window says, and gives no range."""
    window = (hi / 2.0, float(hi))
    return window, ((float(lo), float(hi)) if lo > hi / 2.0 else None)


@knob("tempo")
def tempo(value, known) -> dict:
    """"lo-hi" (BPM): where the 5-BPM drum-loop tempo bands lie (outside it, coarse bands),
    and where loop tempos fold by octaves (half and double time): into the range's top octave,
    never out of the range (fold_window; the fold knob turns folding off)."""
    m = re.fullmatch(r"\s*(\d{2,3})\s*-\s*(\d{2,3})\s*", str(value))
    if not m:
        raise ConfigError(f"tempo = {value!r}: a range like \"85-180\"")
    lo, hi = int(m.group(1)), int(m.group(2))
    if not (TEMPO_FLOOR < lo < hi < TEMPO_CEIL - 10):
        raise ConfigError(f"tempo = {value!r}: {TEMPO_FLOOR} < low < high < {TEMPO_CEIL - 10}")
    out = {}
    bands = tempo_bands(lo, hi)
    if bands != tuple(known["curate_config.TEMPO_BANDS"]):
        out["curate_config.TEMPO_BANDS"] = bands
    window, rng = fold_window(lo, hi)
    if window != tuple(known["curate_config.TEMPO_FOLD"]):
        out["curate_config.TEMPO_FOLD"] = window
    k = "curate_config.TEMPO_FOLD_RANGE"
    if k in known and rng != (tuple(known[k]) if known[k] else None):
        out[k] = rng
    return out


@knob("fold")
def fold(value, known) -> dict:
    """auto (loop tempos fold by octaves as the tempo range says: half and double time are
    one tempo) or off (every loop keeps the tempo it has, its own name's or its measured
    one)."""
    on = _choice("fold", value, ("auto", "off")) == "auto"
    return {} if on == known["curate_config.TEMPO_FOLDING"] else {"curate_config.TEMPO_FOLDING": on}


def _scaled(budgets: dict, total: int) -> dict:
    """The budgets in the same proportions, summing to total (largest remainders)."""
    s = sum(budgets.values())
    raw = {c: b * total / s for c, b in budgets.items()}
    out = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: raw[c] - out[c], reverse=True)[:total - sum(out.values())]:
        out[c] += 1
    return out


def _live_budgets(known) -> dict:
    off = known["curate_config.CATEGORIES_OFF"]
    return {c: b for c, b in known["curate_config.BUDGETS"].items() if c not in off}


@knob("scale")
def scale(value, known) -> dict:
    """"library": the master's size follows the analyzed library (a library with fewer than
    LIBRARY_PER_MASTER usable samples for each file the budgets add up to gets a smaller
    master, with smaller and fewer folders; a larger one the budgets as they are); "off": the
    budgets whatever the library's size. files = N turns it off (packs/scale.py)."""
    on = _choice("scale", value, ("library", "off")) == "library"
    return {} if on == known["curate_config.LIBRARY_SCALE"] else {"curate_config.LIBRARY_SCALE": on}


@knob("files")
def files(value, known) -> dict:
    """The master's file count: the budgets keep their proportions and sum to this (the
    categories switched off left out), whatever the library's size (library scaling off)."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 100:
        raise ConfigError(f"files = {value!r}: a file count, at least 100")
    live = _live_budgets(known)
    new = {**known["curate_config.BUDGETS"], **_scaled(live, value)}
    out = {} if new == known["curate_config.BUDGETS"] else {"curate_config.BUDGETS": new}
    if known["curate_config.LIBRARY_SCALE"]:
        out["curate_config.LIBRARY_SCALE"] = False
    return out


def master_mb(budgets: dict, known) -> float:
    """The master's estimated size in MB for these budgets, its derived sets included."""
    avg = known["curate_config.AVG_FILE_MB"]
    return sum(b * avg.get(c, 0.5) for c, b in budgets.items()) * (1 + known["curate_config.SETS_SHARE"])


def storage_limit_mb(known) -> tuple[float, str] | None:
    """(STORAGE_SHARE of the smallest storage among fourier.toml's devices, that device), or
    None when no configured device states its storage."""
    from .devices.loader import DeviceLoader
    from .places import PlacesError, devices
    try:
        ids = devices()
    except PlacesError:
        return None
    sizes = []
    for d in ids:
        try:
            p = DeviceLoader().load(d)
        except Exception:
            continue
        if p.storage_mb:
            sizes.append((p.storage_mb, d))
    if not sizes:
        return None
    mb, dev = min(sizes)
    return mb * known["curate_config.STORAGE_SHARE"], dev


@knob("size")
def size(value, known) -> dict:
    """"auto": the preset's budgets, scaled down (never up) so the master fits STORAGE_SHARE
    of the smallest configured device's storage, and a category its pool can't fill hands
    its surplus to the others (POOL_SURPLUS); "fixed": the preset's budgets as they are; or a
    card size ("5GB", "800MB"): the budgets keep their proportions and are scaled so the
    master, with its derived sets, fits (AVG_FILE_MB per category, SETS_SHARE for 00_KITS
    and 00_SLICE)."""
    if value == "fixed":
        return {"curate_config.POOL_SURPLUS": False} if known["curate_config.POOL_SURPLUS"] else {}
    live = _live_budgets(known)
    if value == "auto":
        out = {} if known["curate_config.POOL_SURPLUS"] else {"curate_config.POOL_SURPLUS": True}
        cap = storage_limit_mb(known)
        if cap and master_mb(live, known) > cap[0]:
            total = max(100, int(sum(live.values()) * cap[0] / master_mb(live, known)))
            out["curate_config.BUDGETS"] = {**known["curate_config.BUDGETS"], **_scaled(live, total)}
        return out
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(GB|MB)\s*", str(value), re.I)
    if not m:
        raise ConfigError(f"size = {value!r}: auto, fixed, or a size like \"5GB\" or \"800MB\"")
    mb = float(m.group(1)) * (1000 if m.group(2).upper() == "GB" else 1)
    total = int(sum(live.values()) * mb / master_mb(live, known))
    if total < 100:
        raise ConfigError(f"size = {value!r}: too small for a master (about {total} files)")
    new = {**known["curate_config.BUDGETS"], **_scaled(live, total)}
    return {} if new == known["curate_config.BUDGETS"] else {"curate_config.BUDGETS": new}


LOUDNESS_LIMIT_DB = {"gentle": 0.0, "standard": 3.0, "hot": 6.0}


@knob("loudness")
def loudness(value, known) -> dict:
    """How hard loops are limited toward their RMS target: gentle (not at all), standard
    (up to 3 dB), hot (up to 6 dB)."""
    db = LOUDNESS_LIMIT_DB[_choice("loudness", value, tuple(LOUDNESS_LIMIT_DB))]
    return {} if db == known["curate_config.LOOP_LIMIT_DB"] else {"curate_config.LOOP_LIMIT_DB": db}


@knob("retune")
def retune(value, known) -> dict:
    """Tonal one-shots to C: off, named (only when the file name says the note), detect
    (also by pitch detection)."""
    _choice("retune", value, ("off", "named", "detect"))
    if value == "off":
        return {"curate_config.RETUNE_CATS": set(), "curate_config.RETUNE_DETECT_CATS": set()}
    if value == "named":
        return {"curate_config.RETUNE_DETECT_CATS": set()}
    return {}


@knob("stereo")
def stereo(value, known) -> dict:
    """keep (every file's channels), fold-near-mono (stereo files with almost no width, and
    KICKS / SUB, to mono), mono (everything)."""
    _choice("stereo", value, ("keep", "fold-near-mono", "mono"))
    if value == "keep":
        return {"curate_config.MONO_CATS": set(), "curate_config.NEAR_MONO_SIDE_DB": -200.0}
    if value == "mono":
        return {"curate_config.MONO_CATS": set(known["curate_config.CATEGORY_ORDER"])}
    return {}


@knob("names")
def names(value, known) -> dict:
    """canonical (the source name minus vendor boilerplate, cut to the name limit) or keep
    (the source name as it is, FAT-safe, with a loop's BPM when the name lacks it)."""
    _choice("names", value, ("canonical", "keep"))
    return {} if value == known["curate_config.NAMES"] else {"curate_config.NAMES": value}


@knob("vendors")
def vendors(value, known) -> dict:
    """first-folder (the first folder under the library folder is a sample's vendor, for the
    per-vendor cap) or auto (each library folder's layout decides: vendor/pack folders, folders
    by sound type with no vendor cap, an umbrella folder whose packs are the vendors, a flat
    folder of files; packs/vendors.py)."""
    from .packs.vendors import MODES
    v = _choice("vendors", value, MODES)
    return {} if v == known["curate_config.VENDORS"] else {"curate_config.VENDORS": v}


@knob("words")
def words(value, known) -> dict:
    """{CATEGORY: [word, ...]}: words your library names a category's sounds with that
    Fourier doesn't know ("bombo" for KICKS, "nappe" for PADS), matched as whole words in
    file and folder names, added to the built-in words (config/providers/path.yaml). They
    route only without Sononym, which classifies by sound."""
    if not isinstance(value, dict):
        raise ConfigError('words: a table of CATEGORY = ["word", ...]')
    cats = known["curate_config.CATEGORIES"]
    added = known.get("curate_config.ADDED_CATEGORIES") or {}
    order = known["curate_config.CATEGORY_ORDER"]
    got = {}
    for cat, ws in value.items():
        c = str(cat).upper()
        if c not in order:
            raise ConfigError(f"words: {cat!r} is not a category ({', '.join(order)})")
        labels = (cats.get(c) or {}).get("labels") or (added.get(c) or {}).get("labels")
        if not labels:
            raise ConfigError(f"words: {c} is chosen by other rules (loops, instruments, waves), "
                              f"not by name words")
        ws = [ws] if isinstance(ws, str) else ws
        if not isinstance(ws, (list, tuple)) or not ws or not all(isinstance(w, str) and w.strip() for w in ws):
            raise ConfigError(f"words: {c} = {ws!r}: a list of words")
        got[c] = sorted({w.strip().lower() for w in ws})
    new = {**known["curate_config.PATH_WORDS"], **got}
    return {} if new == known["curate_config.PATH_WORDS"] else {"curate_config.PATH_WORDS": new}


@knob("sets")
def sets(value, known) -> dict:
    """on (a build adds 00_KITS, ready drum kits, and 00_SLICE, loops ready to slice, beside
    the categories) or off (only the category folders)."""
    on = _choice("sets", value, ("on", "off")) == "on"
    return {} if on == known["sets.SETS_ON"] else {"sets.SETS_ON": on}


@knob("sources")
def sources(value, known) -> dict:
    """favor: regexes on the file path (their files are nudged toward selection); home:
    {pack: CATEGORY}, a pack whose every file has one home; vendor_max: the most of a
    category one vendor may supply (0 to 1)."""
    if not isinstance(value, dict):
        raise ConfigError("sources: a table with favor, home and vendor_max")
    extra = set(value) - {"favor", "home", "vendor_max"}
    if extra:
        raise ConfigError(f"sources: unknown keys {sorted(extra)} (favor, home, vendor_max)")
    out = {}
    if value.get("favor"):
        pats = [value["favor"]] if isinstance(value["favor"], str) else list(value["favor"])
        for p in pats:
            try:
                re.compile(p)
            except re.error as e:
                raise ConfigError(f"sources.favor: {p!r}: {e}") from None
        out["curate_config.FAVORED_SOURCES"] = re.compile("|".join(f"(?:{p})" for p in pats), re.I)
    if value.get("home"):
        order = known["curate_config.CATEGORY_ORDER"]
        bad = {k: v for k, v in value["home"].items() if str(v).upper() not in order}
        if bad:
            raise ConfigError(f"sources.home: not categories: {bad}")
        out["curate_config.PACK_HOME"] = {k: str(v).upper() for k, v in value["home"].items()}
    if "vendor_max" in value:
        v = value["vendor_max"]
        if not isinstance(v, (int, float)) or not 0 < v <= 1:
            raise ConfigError(f"sources.vendor_max = {v!r}: a share, above 0 and at most 1")
        if float(v) != known["curate_config.VENDOR_MAX_SHARE"]:
            out["curate_config.VENDOR_MAX_SHARE"] = float(v)
    return out


def apply(doc: dict, known: dict) -> list[tuple[str, str, object]]:
    """[(knob, qualified tunable, value)] for the knobs a config document sets. known: the
    values so far (the defaults, with the earlier layers' overrides), which a knob adjusts
    (a category weight scales the budget the preset set) or compares with."""
    if "files" in doc and "size" in doc and doc["size"] not in ("auto", "fixed"):
        raise ConfigError("files and size both set: give one")
    out, now = [], dict(known)
    for name, fn in KNOBS.items():           # in order: categories' weights before size, scale before files
        if name in doc:
            for k, v in fn(doc[name], now).items():
                out.append((name, k, v))
                now[k] = v
    return out


# each knob in plain words, for `fourier config explain <knob>` (the docstrings above are the
# exact rules, shown with --detail)
PLAIN_HELP = {
    "devices": ('The samplers you build for, by profile name (`fourier devices list`). File and '
                'folder names are cut to fit the strictest one.', 'devices = ["digitakt_2", "m8_tracker"]'),
    "categories": ("Leave a category out, bring one back, or give it more or fewer files than the "
                   "style does (1.5 is half as many again).", 'categories = { BLIPS = "off", PADS = 1.5 }'),
    "tempo": ("The drum-loop tempos you play at. Loops are sorted into 5-BPM folders across this "
              "range, and half- or double-time loops are filed at the tempo they'd play at in it.",
              'tempo = "120-140"'),
    "fold": ('Whether a half- or double-time loop is filed at the tempo it would play at in your '
             'range ("auto") or at its own ("off").', 'fold = "off"'),
    "scale": ('"library": a small library gets a smaller set, in proportion, so every category '
              'still has a fair share. "off": the style\'s full size whatever the library.',
              'scale = "off"'),
    "files": ("Make the whole set this many files, keeping the style's proportions.", "files = 3000"),
    "size": ('Make the whole set fit a card or a drive ("4GB", "800MB"). "auto" fits half of your '
             'smallest device\'s storage; "fixed" keeps the style\'s size.', 'size = "4GB"'),
    "loudness": ('How hard loops are pushed toward an even level: "gentle" (not at all), "standard", '
                 'or "hot".', 'loudness = "gentle"'),
    "retune": ('Tune melodic one-shots to C so they play in key across the keyboard: "detect" '
               '(listens for the note), "named" (only when the file name says the note), or "off" '
               '(as they are).', 'retune = "off"'),
    "stereo": ('"keep" every file as it is, fold almost-mono files (and kicks and subs) to mono '
               '("fold-near-mono", saves space), or make everything "mono".', 'stereo = "keep"'),
    "names": ('"canonical": tidy names (the pack maker\'s prefixes taken off, cut to fit the '
              'device). "keep": the original file names.', 'names = "keep"'),
    "vendors": ("How Fourier tells which sample maker a file comes from, so no one maker fills a "
                'category: "auto" works it out from your folders.', 'vendors = "auto"'),
    "words": ("Your packs' own words for a kind of sound, when Fourier doesn't know them. Matched "
              "as whole words in file and folder names. Not for the loop, piano, acoustic and "
              "wavetable categories, which go by sound and tags.",
              'words = { KICKS = ["bombo"], PADS = ["nappe"] }'),
    "sets": ('Whether a build also makes 00_KITS (ready drum kits) and 00_SLICE (loops ready to '
             'slice).', 'sets = "off"'),
    "sources": ("Favor some packs, send a whole pack to one category, or change how much of a "
                "category one maker may supply (0 to 1).",
                'sources = { favor = ["Acme"], home = { "Acme Pads" = "PADS" }, vendor_max = 0.5 }'),
}
