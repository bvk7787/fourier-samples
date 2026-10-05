"""Derived sets in the master: copies of curated files grouped for playing,
beside the categories. Built after a whole build from its manifest.

  KITS/<kit>/    ready drum kits: one pack (and drum machine) each, kick / snare / clap /
                 closed and open hat / cymbal / toms / percussion;
                 every drum machine's best kit first
  SLICE/<family>/  drum loops that slice cleanly on an equal 16th grid (the Digitakt 2
                 Slice machine's CREATE SLICE GRID, the M8's even slices), not swung
  loops.csv      every drum loop's tempo, bars, swing and slice verdict

Renders number them 00_KITS / 00_SLICE (curate_config.DERIVED_DIRS). The manifest lists
them under "sets" (not "categories"), so category rules don't apply to the copies. SLICE
files are byte-identical to the curated file they came from; a kit file is too, or that
file turned down to its role's level (KIT_LEVEL_DB, `gain_db` in the manifest).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..settings import for_module as _for_module

_tunable = _for_module("sets")   # overridable: fourier/settings.py, config/tunables.yaml

# the sets knob: "off" builds the category folders only (no 00_KITS, 00_SLICE or loops.csv)
SETS_ON = _tunable("SETS_ON", True)
SLICE_STEP = _tunable("SLICE_STEP", 16)            # slices per bar (16ths)
SLICE_MAX = _tunable("SLICE_MAX", 64)             # the Digitakt 2's largest grid (p.99)
SLICE_TOL_S = _tunable("SLICE_TOL_S", 0.010)        # an onset this close to a slice start counts as on it
SLICE_READY = _tunable("SLICE_READY", 0.80)         # share of slices that start on a hit or hold none
SLICE_MIN_HITS = _tunable("SLICE_MIN_HITS", 0.125)     # ...and at least this share start on a hit: 2 a bar (not a crash or FX loop)
SLICE_BARS = _tunable("SLICE_BARS", (1, 2, 4))     # 16ths on the Digitakt's largest grid (64) need 4 bars or fewer
SLICE_BAR_ERR = _tunable("SLICE_BAR_ERR", 0.1)        # ...and whole bars (beats off), so the grid lines up
SWING_FLAG = _tunable("SWING_FLAG", 56.0)          # swing % (MPC-style) from which a loop is "swung"
KIT_COUNT = _tunable("KIT_COUNT", 24)             # kits kept: acoustic kits, then drum machines, then the most complete packs
# Kits (so the sets don't lean too hard on particular drum machines, the library's
# acoustic kits come first): every acoustic kit
# pack that has the parts comes first (each gets its slots pinned into the build, see
# curate._kit_source_pins), then at most KIT_MACHINE_MAX drum machines, then pack kits.
KIT_ACOUSTIC_PACKS = _tunable("KIT_ACOUSTIC_PACKS", re.compile(r"(?!)", re.I))   # library: acoustic kit packs (none by default)
KIT_ACOUSTIC_MAX = _tunable("KIT_ACOUSTIC_MAX", 10)
KIT_MACHINE_MAX = _tunable("KIT_MACHINE_MAX", 10)
KIT_NAME_DROP = _tunable("KIT_NAME_DROP", {"the", "sample", "samples", "pack", "kit", "kits", "by"})
KIT_GENERIC = _tunable("KIT_GENERIC", {"presets", "samples", "sample-pack", "kits", "drums", "one-shots", "oneshots", "wav"})
KIT_ROLES = _tunable("KIT_ROLES", ("kick", "snare", "clap", "hat_closed", "hat_open", "cymbal", "tom", "perc"))
KIT_REQUIRED = _tunable("KIT_REQUIRED", ("kick", "snare", "hat_closed", "hat_open"))
KIT_PER_ROLE = _tunable("KIT_PER_ROLE", {"tom": 2, "perc": 2})
# Kit levelling (kits play at one common level): every drum one-shot is
# peak-normalized, so a kit's short-term level depends on the pack, and a kick can sit as
# loud as the loops' peaks. Each kit slot is turned down to its role's
# loudness (the loudest 75 ms window, curate's LEVEL_WINDOW_MS; set low, so most slots are
# turned down to it), so every kit plays at one level with one balance. Only down: the files
# already peak at -1 dBFS, so a slot quieter than its target stays as it is.
KIT_LEVEL_DB = _tunable("KIT_LEVEL_DB", {"kick": -9.0, "snare": -13.0, "clap": -14.0, "hat_closed": -16.0,
                "hat_open": -15.0, "cymbal": -16.0, "tom": -11.0, "perc": -15.0})
KIT_LEVEL_TOL_DB = _tunable("KIT_LEVEL_TOL_DB", 0.5)


# ---------------------------------------------------------------------------
# Slice metrics (per exported drum loop)
# ---------------------------------------------------------------------------
ONSET_MIN_RISE_DB = _tunable("ONSET_MIN_RISE_DB", 3.0)     # an onset's envelope rises at least this much within 2 ms


def _onsets(y, sr):
    """Attack times (s): librosa onsets refined to the steepest rise of the 0.5 ms envelope
    near each (the point 30% of the way up that rise), plus a hit at 0 when the file opens
    loud and the detector missed it. Refining by the rise, not by the first sample at 30% of
    the window's peak, keeps a hit over a ringing ride or cymbal from reading 25-35 ms early
    (which would rotate a ride loop or a break that is on its grid)."""
    import librosa
    hop = 64
    env = librosa.onset.onset_strength(y=y.astype("float32"), sr=sr, hop_length=hop, center=False)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop, units="frames")
    w = max(1, int(sr * 0.0005))
    k = max(1, int(sr * 0.002))
    lev = np.sqrt(np.convolve(y * y, np.ones(w) / w, "same"))
    ldb = 20 * np.log10(lev + 1e-7)
    out = []
    for f in fr:
        s0, s1 = max(k, int((f - 12) * hop)), min(len(y), int((f + 12) * hop) + 2048)
        if s1 - s0 < 2:
            continue
        rise = ldb[s0:s1] - ldb[s0 - k:s1 - k]
        i = s0 + int(np.argmax(rise))                                 # end of the steepest 2 ms rise
        if rise[i - s0] < ONSET_MIN_RISE_DB:                          # no attack to place here
            continue
        lo, hi = float(lev[i - k]), float(lev[i])
        j = i - k + int(np.argmax(lev[i - k:i + 1] >= lo + 0.3 * (hi - lo)))
        out.append(j / sr)
    out = sorted(out)
    out = [t for n, t in enumerate(out) if n == 0 or t - out[n - 1] > 0.005]   # one per hit
    w = max(1, int(sr * 0.01))
    if len(y) > w and (not out or out[0] > 0.015):
        head = float(np.sqrt(np.mean(y[:w] ** 2)))
        body = float(np.sqrt(np.mean(y ** 2)))
        if body > 0 and head >= body * 10 ** (-12 / 20):
            out.insert(0, 0.0)
    return np.array(sorted(out))


def slice_metrics(path, bpm):
    """{bars, slice_clean, swing} for a drum loop at its tempo, or {} when it can't be judged.
    slice_clean: share of equal 16th slices (at most SLICE_MAX) that start on a hit or hold
    none; slice_hits: share that start on a hit; swing: MPC-style swing % of the offbeat 16ths (50 = straight)."""
    import soundfile as sf
    if not bpm:
        return {}
    try:
        y, sr = sf.read(str(path), always_2d=True)
    except Exception:
        return {}
    y = y.mean(axis=1)
    dur = len(y) / sr
    exact = dur * bpm / 60.0
    beats = int(round(exact))
    if beats <= 0:
        return {}
    on = _onsets(y, sr)
    n = int(min(SLICE_MAX, max(1, beats * SLICE_STEP // 4)))
    edges = np.arange(n) * dur / n
    good = hits = 0
    for i, e in enumerate(edges):
        nxt = edges[i + 1] if i + 1 < n else dur
        near = on[np.abs(on - e) <= SLICE_TOL_S] if len(on) else on
        inside = on[(on > e + SLICE_TOL_S) & (on < nxt - SLICE_TOL_S)] if len(on) else on
        if len(near) or not len(inside):
            good += 1
        hits += bool(len(near))
    six = dur / beats / 4
    sw = None
    if len(on) >= 8:
        idx = np.round(on / six).astype(int)
        dev = on / six - idx
        odd, even = dev[(idx % 2) == 1], dev[(idx % 2) == 0]
        if len(odd) >= 3:
            # the offbeat 16ths' delay against the on-beat ones: a loop whose hits are all
            # late (cut early) isn't swung
            base = float(np.median(even)) if len(even) >= 3 else 0.0
            sw = round(50.0 * (1.0 + float(np.median(odd)) - base), 1)
            if sw < 45.0:
                sw = None          # offbeats "early" by that much is a misread, not swing
    return {"bars": round(beats / 4, 2), "slice_clean": round(good / n, 3),
            "slice_hits": round(hits / n, 3), "swing": sw,
            "bar_err": round(abs(exact - beats), 3)}


# ---------------------------------------------------------------------------
# Sets
# ---------------------------------------------------------------------------
def slice_ready(e):
    """A drum loop for 00_SLICE: cleanly sliceable on a 16th grid, busy enough to chop
    (SLICE_MIN_HITS), straight, whole bars, and 1, 2 or 4 bars long (SLICE_BARS)."""
    hits = e.get("slice_hits")
    return ((e.get("slice_clean") or 0) >= SLICE_READY and (e.get("swing") or 50.0) < SWING_FLAG
            and (1.0 if hits is None else hits) >= SLICE_MIN_HITS
            and e.get("bars") in SLICE_BARS and (e.get("bar_err") or 0.0) <= SLICE_BAR_ERR)


def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)            # same volume: no extra space
    except OSError:
        shutil.copy2(src, dst)


def kit_level_db(y, sr):
    """A kit slot's level as it plays: the loudest window of a level-keeping mono downmix
    (for stereo, about the channels' own level). The plain L+R average reads a wide hat or
    crash up to 3 dB quiet, so its slot would play over its role on a mono device."""
    from ..devices.exporter import _mono_downmix
    from .curate import _short_term_db
    if not y.size:
        return float("-inf")
    m = _mono_downmix(np.asarray(y).T, keep_level=True) if y.ndim == 2 and y.shape[1] > 1 else y.reshape(-1)
    return _short_term_db(np.asarray(m), sr)


def kit_gain_db(y, sr, role):
    """The cut (dB, <= 0) that brings a kit slot to its role's level; 0 when it's already
    at or under it."""
    t = KIT_LEVEL_DB.get(role)
    st = kit_level_db(y, sr)
    if t is None or not np.isfinite(st) or st <= t:
        return 0.0
    return round(float(t - st), 2)


def _write_kit_file(src, dst, role):
    """Copy src to dst at its role's level; returns the gain applied (dB)."""
    import soundfile as sf
    y, sr = sf.read(str(src), always_2d=True)
    g = kit_gain_db(y, sr, role)
    if g == 0.0:
        _copy(src, dst)
        return 0.0
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = y * (10 ** (g / 20.0))
    sf.write(str(dst), out if out.shape[1] > 1 else out[:, 0], sr, subtype=sf.info(str(src)).subtype)
    return g


def _role(cat, e):
    band = e.get("band")
    return {"KICKS": "kick", "SNARES": "snare", "CLAPS": "clap", "CYMBALS": "cymbal",
            "TOMS": "tom", "PERC": "perc",
            "HATS": "hat_open" if band == "open" else "hat_closed"}.get(cat)


def _slug(s):
    from .curate_config import PACK_SUFFIX_RE
    s = re.sub(r"\b(?:" + PACK_SUFFIX_RE.pattern + r")\b", "", s.lower(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def kit_name(pack, mach):
    """'kit-<machine>-<pack>' without repeats: the machine alone when the pack is named for
    it ("909 Kit"), the vendor when the pack's own name is generic or long
    ("Vendor A/Presets", or a pack name of many words)."""
    if pack == "*":
        return f"kit-{_slug(mach)}" if mach else "kit"
    vendor, _, name = pack.partition("/")
    toks = [t for t in _slug(name or vendor).split("-") if t not in KIT_NAME_DROP]
    ns = "-".join(toks)
    if len(ns) > 30 and not mach:
        # a long pack name keeps its first words
        while len("-".join(toks)) > 24 and len(toks) > 1:
            toks.pop()
        ns = "-".join(toks)
    if not ns or ns in KIT_GENERIC or len(ns) > 30:
        ns = _slug(vendor)
    m = _slug(mach) if mach else ""
    if m:
        # drop pack words spelling the machine another way ("drumtrax", "lindrum"; a model
        # like "lm1" stays)
        from difflib import SequenceMatcher
        ns = "-".join(t for t in ns.split("-")
                      if t and SequenceMatcher(None, t, m).ratio() < 0.8)
    base = "-".join(x for x in ("kit", m, ns) if x)
    return re.sub(r"-+", "-", base)[:40].strip("-")


# what a file's name must say to fill a kit slot (when the pack has such a file; else any
# file of the role's category): no tom as the kick, no tambourine as the closed hat
_W = lambda p: re.compile(p, re.I)
ROLE_NAME = _tunable("ROLE_NAME", {
    "kick": (_W(r"kick|(?<![a-z])bd|bass ?drum"), _W(r"(?<![a-z])(lt|mt|ht|tom|toms)(?![a-z])|low ?tune")),
    "snare": (_W(r"snare|(?<![a-z])sd(?![a-z])|(?<![a-z])sn"), _W(r"rim|(?<![a-z])rs(?![a-z])|stick|perc")),
    "clap": (_W(r"clap|(?<![a-z])cp(?![a-z])|handclap"), None),
    "hat_closed": (_W(r"hat|(?<![a-z])(hh|ch|chh)(?![a-z])|closed"), _W(r"tamb|shak|marac|cabasa|open|(?<![a-z])oh")),
    "hat_open": (_W(r"hat|(?<![a-z])(oh|ohh)(?![a-z])|open"), _W(r"tamb|shak|marac|cabasa")),
    "cymbal": (_W(r"crash|ride|cym"), _W(r"(?<![a-z])rev|reverse")),
    "tom": (_W(r"tom"), None),
    "perc": (None, _W(r"fx|synth|flute|pad")),
})


def _slot_candidates(role, cands):
    """The candidates whose names fit the slot (ROLE_NAME), else all of them."""
    want, avoid = ROLE_NAME.get(role, (None, None))
    fn = lambda ce: os.path.splitext(os.path.basename(ce[1]["src"]))[0].replace("_", " ")
    ok = [ce for ce in cands if (want is None or want.search(fn(ce))) and not (avoid and avoid.search(fn(ce)))]
    return ok or cands


def _kits(man):
    """[(kit name, [(category, entry)])]: per (pack, drum machine), one file a role
    (two toms, two percs). The best kit of each drum machine comes first (acid and house
    reach for a 909 or an 808 kit), then the kits with the most roles; KIT_COUNT of them."""
    from .curate import _machine_label, _pack_key2
    groups = defaultdict(lambda: defaultdict(list))
    for cat, cd in (man.get("categories") or {}).items():
        for e in cd.get("entries", []):
            r = _role(cat, e)
            if r:
                mach = _machine_label(e["src"])
                # a drum machine's kit draws on every source of that machine (the master
                # keeps a 909 pack's kicks but maybe not its hats); other kits, one pack
                key = ("*", mach) if mach else (_pack_key2(e["src"]), None)
                groups[key][r].append((cat, e))
    kits = []
    for (pack, mach), roles in groups.items():
        if not all(roles.get(r) for r in KIT_REQUIRED):
            continue
        have = [r for r in KIT_ROLES if roles.get(r)]
        if len(have) < 6:
            continue
        pick = []
        for r in KIT_ROLES:
            c = sorted(_slot_candidates(r, roles.get(r, [])), key=lambda ce: os.path.basename(ce[1]["out"]).lower())
            k = KIT_PER_ROLE.get(r, 1)
            if c:
                step = max(1, len(c) // (k + 1))
                pick += [c[min(len(c) - 1, step * (i + 1))] for i in range(min(k, len(c)))]
        kits.append((len(have), sum(len(v) for v in roles.values()), pack, mach, pick))
    kits.sort(key=lambda t: (-t[0], -t[1], t[2], t[3] or ""))
    acoustic = [t for t in kits if not t[3] and KIT_ACOUSTIC_PACKS.search(t[2])][:KIT_ACOUSTIC_MAX]
    machines, seen = [], set()
    for t in sorted(kits, key=lambda t: (-t[1], t[2])):   # each machine's biggest kit
        if t[3] and t[3] not in seen:
            seen.add(t[3])
            machines.append(t)
    machines = machines[:KIT_MACHINE_MAX]
    machines.sort(key=lambda t: (-t[0], -t[1], t[2], t[3] or ""))
    first = acoustic + machines
    kits = first + [t for t in kits if t not in first and not t[3]]
    out, used = [], Counter()
    for _, _, pack, mach, pick in kits[:KIT_COUNT]:
        base = kit_name(pack, mach)
        used[base] += 1
        out.append((base if used[base] == 1 else f"{base}-{used[base]}", pick))
    return out


def build_sets(master_dir, log=print):
    """Write KITS/, SLICE/ and loops.csv into a built master and list them in its manifest."""
    root = Path(master_dir)
    mp = root / "manifest.json"
    man = json.loads(mp.read_text())
    additive = bool(man.get("base"))
    if additive:
        # a v2 on top of a release keeps the release's kits and slice set as they are
        # (they're on devices); only new slice-ready loops are added
        old = man.get("sets") or {}
        sets = {k: {"entries": list((old.get(k) or {}).get("entries", []))} for k in ("KITS", "SLICE")}
    else:
        for d in ("KITS", "SLICE"):
            if (root / d).exists():
                shutil.rmtree(root / d)
        sets = {"KITS": {"entries": []}, "SLICE": {"entries": []}}
    have = {e["out"] for e in sets["SLICE"]["entries"]}
    for kit, pick in ([] if additive else _kits(man)):
        for cat, e in pick:
            src = root / cat / e["out"]
            name = os.path.basename(e["out"])
            dst = root / "KITS" / kit / name
            if dst.exists():
                continue
            g = _write_kit_file(src, dst, _role(cat, e))
            sets["KITS"]["entries"].append(dict(family=kit, out=f"{kit}/{name}", src=e["src"],
                                                out_md5=(e.get("out_md5") if g == 0.0 else None) or _md5(dst),
                                                came_from=f"{cat}/{e['out']}",
                                                **({"gain_db": g} if g else {})))
    rows = []
    for e in (man.get("categories", {}).get("DRUMLOOPS") or {}).get("entries", []):
        ready = slice_ready(e)
        fb = e.get("bpm_fold") or e.get("bpm")
        _bb = e.get("bpm_bars") or e.get("bpm")              # the tempo "bars" was counted at
        bars_f = round(e["bars"] * fb / _bb, 2) if e.get("bars") and _bb and fb else None
        rows.append([e["family"], os.path.basename(e["out"]), e.get("bpm"), e.get("bars"), fb, bars_f,
                     e.get("swing"), e.get("slice_clean"), "yes" if ready else "",
                     "swung" if (e.get("swing") or 50.0) >= SWING_FLAG else ""])
        if ready and e["out"] not in have and not (additive and not e.get("added_over")):
            dst = root / "SLICE" / e["out"]
            _copy(root / "DRUMLOOPS" / e["out"], dst)
            sets["SLICE"]["entries"].append(dict(family=e["family"], out=e["out"], src=e["src"],
                                                 out_md5=e.get("out_md5") or _md5(dst),
                                                 came_from=f"DRUMLOOPS/{e['out']}"))
    with open(root / "loops.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "file", "bpm", "bars", "folder_bpm", "bars_at_folder_bpm", "swing_pct",
                    "slice_clean", "slice_ready", "swung"])
        w.writerows(rows)
    man["sets"] = sets
    mp.write_text(json.dumps(man, indent=2))
    log(f"sets: {len({e['family'] for e in sets['KITS']['entries']})} kits "
        f"({len(sets['KITS']['entries'])} files), {len(sets['SLICE']['entries'])} slice-ready loops")
    return sets
