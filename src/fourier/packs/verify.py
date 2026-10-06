"""`fourier verify`: prove a built master (and, optionally, real device renders)
obeys every curation rule, not just the structural ones `verify --quick` covers.

Each check reads the rule from the same config the build uses (curate_config, the
curate helpers), so a rule change is verified the day it ships. Results are
(level, check, detail) with level PASS / WARN / FAIL; any FAIL makes the run fail.
A human Keep is exempt from the selection rules, as it is in the build.

  master   manifest <-> disk, out_md5, budgets and folder ceiling, allocation follows
           cluster size, naming vocabulary, ratings honoured, PACK_HOME/keys/waves routing,
           name overrides and filters, plucked strings, band purity (hats, FX, loops),
           sample chains, length caps, bpm-named phrases, musical phrases only in PHRASES,
           identical audio, DC, WAVES exactness, instrument tails, plus everything in
           `verify --quick`.
  render   (--render DEVICE) renders into a scratch dir and checks file count, device
           format, card path length, WAVES sample-exact, and that stereo files which
           cancel when summed keep their level in a mono render.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ..places import library_rel

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
DC_MAX_SHARE = 0.02          # |mean| / peak
DC_FAIL_COUNT = 0            # export re-checks DC after the trim, so any file over is a bug
TAIL_SLACK_S = 0.06          # instrument tail past -60 dB: pad + fade + envelope rounding
TAIL_CHECK_DB = -63.0        # "silence" for the tail check: clearly under the -60 dB trim floor


def _build_duration(ctx, src):
    """A source's length as the build judged it (the database's, when there is one): a
    stored length can exceed the frame count in the last digit, and a 4 s limit must fall
    the same way in both."""
    d = (getattr(ctx, "dur", None) or {}).get(src)
    if d is not None:
        return d
    fr, sr = _src_frames(src)
    return None if fr is None else fr / sr


def _tail_sound(env, rel, sr):
    """Where an exported tail's sound is, as the trim judged it: runs of `env` above `rel` x
    its max lasting 20 ms, and also a run that reaches the file's end. The trim cut there,
    after a run long enough in the source; the fade and the envelope's window at the edge
    can shorten it under 20 ms here (a short blip the trim rightly kept can then look like
    quiet)."""
    from .curate import TAIL_FADE_MS, TRIM_PAD_MS, _sustained_above
    nz = _sustained_above(env, rel, int(sr * 0.02))
    above = np.where(env > float(env.max()) * rel)[0]
    edge = int(sr * (TRIM_PAD_MS + TAIL_FADE_MS + 10.0) / 1000.0)
    if above.size and above[-1] >= len(env) - 1 - edge and (not len(nz) or above[-1] > nz[-1]):
        return above
    return nz
MONO_OK_DB = -3.0            # a phase-inverted stereo file may lose at most this in a mono render


@dataclass
class Result:
    level: str
    check: str
    detail: str = ""


class _Ctx:
    def __init__(self, master_dir, store_path=None, prof=None):
        from .ratings import drop_set, keep_pins, misfiled_map
        self.root = Path(master_dir)
        self.man = json.loads((self.root / "manifest.json").read_text())
        self.cats = self.man.get("categories", {})
        self.entries = [(c, e) for c, cd in self.cats.items() for e in cd.get("entries", [])]
        self.set_entries = [(s, e) for s, sd in (self.man.get("sets") or {}).items()
                            for e in sd.get("entries", [])]
        self.fams = {}
        for c in self.cats:
            p = self.root / c / "_manifest.json"
            self.fams[c] = json.loads(p.read_text()) if p.exists() else []
        self.pins = keep_pins(store_path)
        self.drops = drop_set(store_path)
        self.misf = misfiled_map(store_path)
        self.prof = prof          # {src path: (n_events, regularity, echo, file_hash)} or None
        self.extra = None         # {src path: (chroma_concentration, harmonicity)} or None
        self.phrase = None        # {src paths that curate._is_phrase calls a phrase} or None
        self.fallback = False     # the build classified without Sononym (names read bare notes)

    def path(self, c, e):
        return self.root / c / e["out"]

    def pinned(self, c, e):
        return self.pins.get(e.get("src")) == c


def _rel(src):
    return library_rel(src or "")


def _item(x) -> str:
    """One item of a check's detail as a person reads it: ('PADS', 9, 30) -> PADS (9, 30)."""
    if isinstance(x, tuple) and x:
        head, *rest = x
        return f"{head} ({', '.join(str(r) for r in rest)})" if rest else str(head)
    return str(x)


def _fmt(items, n=3):
    items = list(items)
    if not items:
        return ""
    more = f"; and {len(items) - n} more" if len(items) > n else ""
    return f"{len(items)}: " + "; ".join(_item(x) for x in items[:n]) + more


# --- structural -----------------------------------------------------------------------------
def check_manifest(ctx):
    on_disk = {p.relative_to(ctx.root).as_posix() for p in ctx.root.rglob("*")
               if p.suffix.lower() == ".wav" and p.is_file()
               and not p.relative_to(ctx.root).parts[0].startswith(("_", "."))}
    listed = {f"{c}/{e['out']}" for c, e in ctx.entries}
    listed |= {f"{s}/{e['out']}" for s, e in ctx.set_entries}
    miss, extra = listed - on_disk, on_disk - listed
    yield Result(FAIL if (miss or extra) else PASS, "manifest matches disk",
                 f"{len(listed)} files" + (f"; missing {_fmt(miss)}" if miss else "")
                 + (f"; unlisted {_fmt(extra)}" if extra else ""))
    bad = [f"{c}/{e['out']}" for c, e in ctx.entries if e.get("out_md5") and ctx.path(c, e).exists()
           and hashlib.md5(ctx.path(c, e).read_bytes()).hexdigest() != e["out_md5"]]
    yield Result(FAIL if bad else PASS, "out_md5 matches every file", _fmt(bad))
    nomd5 = [f"{c}/{e['out']}" for c, e in ctx.entries if not e.get("out_md5")]
    nomd5 += [f"{c}/{e['out']}" for c, e in ctx.set_entries if not e.get("out_md5")]
    yield Result(FAIL if nomd5 else PASS, "every file has an out_md5 (releases and locks rely on it)",
                 _fmt(nomd5))


def check_sets(ctx):
    """The derived sets (packs/sets.py) are copies of curated files (kit files maybe turned
    down to their role's level), and every kit has the parts a kit needs."""
    if not ctx.set_entries:
        return
    import soundfile as sf
    from .sets import KIT_LEVEL_DB, KIT_LEVEL_TOL_DB, KIT_REQUIRED, _role, kit_level_db
    ent = {f"{c}/{e['out']}": (c, e) for c, e in ctx.entries}
    bad, loud = [], []
    for s, e in ctx.set_entries:
        a, b = ctx.root / s / e["out"], ctx.root / (e.get("came_from") or "")
        if not (a.is_file() and b.is_file()):
            bad.append(f"{s}/{e['out']}")
            continue
        g = e.get("gain_db") if s == "KITS" else None
        if not g and a.read_bytes() != b.read_bytes():
            bad.append(f"{s}/{e['out']}")
            continue
        if s != "KITS":
            continue
        ya, sra = sf.read(str(a), always_2d=True)
        if g:
            # the curated file times the gain, to within a few 16-bit steps
            yb, srb = sf.read(str(b), always_2d=True)
            if sra != srb or ya.shape != yb.shape or (
                    ya.size and float(np.max(np.abs(ya - yb * 10 ** (g / 20.0)))) > 3.0 / 32768):
                bad.append(f"{s}/{e['out']}")
                continue
        r = _role(*ent[e["came_from"]]) if e.get("came_from") in ent else None
        if r in KIT_LEVEL_DB and ya.size:
            v = kit_level_db(ya, sra)
            if v > KIT_LEVEL_DB[r] + KIT_LEVEL_TOL_DB:
                loud.append((f"{s}/{e['out']}", round(v, 1)))
    yield Result(FAIL if bad else PASS, "set files are copies of curated files (kit files maybe turned down)",
                 _fmt(bad))
    yield Result(FAIL if loud else PASS, "kit slots at or under their role's level (KIT_LEVEL_DB)", _fmt(loud))
    by_kit = defaultdict(set)
    for s, e in ctx.set_entries:
        if s == "KITS" and e.get("came_from") in ent:
            by_kit[e["family"]].add(_role(*ent[e["came_from"]]))
    thin = [(k, sorted(set(KIT_REQUIRED) - r)) for k, r in by_kit.items() if not set(KIT_REQUIRED) <= r]
    yield Result(FAIL if thin else PASS, "every kit has kick, snare, closed and open hat", _fmt(thin))
    from .sets import slice_ready
    loose = [f"SLICE/{e['out']}" for s, e in ctx.set_entries if s == "SLICE"
             and ((e.get("came_from") not in ent) or not slice_ready(ent[e["came_from"]][1]))]
    yield Result(FAIL if loose else PASS, "SLICE holds only straight, cleanly slicing loops of 1, 2 or 4 bars",
                 _fmt(loose))


def _allowed(ctx):
    """Budgets and folder caps a master is held to: an additive build (manifest "base")
    may exceed them by its add allowance (releases.additive_build)."""
    from .curate_config import ADD_ALLOWANCE, BUDGETS
    # a category rebuilt with a surplus (size = "auto") records its raised budget
    raised = {c: cd["budget"] for c, cd in ctx.man.get("categories", {}).items() if "budget" in cd}
    if not ctx.man.get("base"):
        return {**BUDGETS, **raised}, 0.0
    a = float(ctx.man.get("add_allowance", ADD_ALLOWANCE))
    return {c: b + max(1, int(round(b * a))) for c, b in {**BUDGETS, **raised}.items()}, a


def check_budgets(ctx):
    from .curate import MAX_PER_FAMILY
    BUDGETS, _add = _allowed(ctx)
    over = [(c, len(cd["entries"]), BUDGETS[c]) for c, cd in ctx.cats.items()
            if c in BUDGETS and len(cd["entries"]) > BUDGETS[c]]
    yield Result(FAIL if over else PASS, "no category over budget", _fmt(over))
    under = [(c, len(cd["entries"]), BUDGETS[c]) for c, cd in ctx.cats.items()
             if c in BUDGETS and len(cd["entries"]) < BUDGETS[c]]
    if under:
        yield Result(WARN, "categories under budget (pool ran out)", _fmt(under, 5))
    big = [(c, f["family"], f["copied"]) for c, fl in ctx.fams.items() for f in fl
           if f["copied"] > MAX_PER_FAMILY]
    yield Result(FAIL if big else PASS, f"no folder over {MAX_PER_FAMILY} files", _fmt(big))
    from .curate_config import CATEGORIES, FOLDER_MAX
    many, many_t = [], []
    for c, fl in ctx.fams.items():
        cfg = CATEGORIES.get(c, {})
        cap = int(cfg.get("folder_max", FOLDER_MAX))
        cap += int(np.ceil(cap * _add))
        if len(fl) > cap:
            (many_t if cfg.get("tempo_bands") else many).append((c, len(fl), cap))
    yield Result(FAIL if many else PASS, f"folders per category within the cap ({FOLDER_MAX})", _fmt(many))
    if many_t:        # tempo ranges each keep a folder, so these can run over
        yield Result(WARN, "tempo-banded categories over their folder cap", _fmt(many_t))
    from .curate_config import BUDGETS as _B, FOLDER_MIN, FOLDER_TARGET_FILES
    sc = ctx.man.get("scale")
    if sc:
        # a library-scaled master (packs/scale.py): its recorded budgets, about one folder per
        # folder_files files, and a band keeps its own folder
        _B = {**_B, **{c: cd["budget"] for c, cd in ctx.man.get("categories", {}).items() if "budget" in cd}}
        most = lambda c, fl: max(1, int(round(_B[c] / max(sc["folder_files"], 1))),  # noqa: E731
                                 len({f.get("band") for f in fl}))
        per = sc["folder_files"]
    else:
        most = lambda c, fl: max(FOLDER_MIN, int(round(_B[c] / FOLDER_TARGET_FILES)))  # noqa: E731
        per = FOLDER_TARGET_FILES
    small = [(c, len(fl), most(c, fl))
             for c, fl in ctx.fams.items() if c in _B and not CATEGORIES.get(c, {}).get("tempo_bands")
             and not CATEGORIES.get(c, {}).get("band_min_folders")
             and len(fl) > most(c, fl)]
    yield Result(WARN if small else PASS, f"small categories keep about {per} files a folder",
                 _fmt(small, 5))
    from .curate_config import FOLDER_MIN_FILES
    fmin = int(sc["folder_min_files"]) if sc else FOLDER_MIN_FILES
    bmf = (lambda c: 1) if sc else (lambda c: int(CATEGORIES.get(c, {}).get("band_min_folders", 1)))
    tiny = [(c, f["family"], f["copied"]) for c, fl in ctx.fams.items() for f in fl
            if not CATEGORIES.get(c, {}).get("tempo_bands") and f["copied"] < fmin
            and sum(1 for g in fl if g.get("band") == f.get("band")) > bmf(c)]
    yield Result(WARN if tiny else PASS, f"no folders under {fmin} files", _fmt(tiny, 5))
    flat = [(c, len(fl)) for c, fl in ctx.fams.items()
            if not _add and len(fl) > 1 and sum(f["copied"] >= MAX_PER_FAMILY for f in fl) > len(fl) / 2]
    yield Result(FAIL if flat else PASS, "allocation follows cluster size (most folders below the ceiling)",
                 _fmt(flat))


def check_naming(ctx):
    from . import naming as N
    from .curate_config import CATEGORIES, KEYS_EXCLUDE
    names = [(c, f["family"]) for c, fl in ctx.fams.items() for f in fl]
    drum = {"gated", "clicky", "snappy", "punchy", "squashed", "sub-heavy", "long-tail",
            "low-tuned", "high-tuned"}
    bad = [(c, n) for c, n in names if c in N.TONAL_CATS and any(w in n.split("-") or w in n for w in drum)]
    yield Result(FAIL if bad else PASS, "tonal folders use tonal words", _fmt(bad))
    # a fragment of a hyphenated trait ("long" of "long-tail") that isn't a whole trait itself
    frag = []
    for c, n in names:
        dims = N.naming_dims(c, CATEGORIES.get(c, {"kind": "oneshot"}))
        whole = {w for _, pair in dims for w in pair if w}
        parts = {p for w in whole if "-" in w for p in w.split("-")} - whole
        last = n.split("-")[-1]
        if last in parts and not any(n.endswith(w) for w in whole):
            frag.append((c, n))
    yield Result(FAIL if frag else PASS, "traits added whole (no half traits)", _fmt(frag))
    # one word per dimension, and no sibling that only reorders or drops another's words
    both, twins = [], []
    for c in {c for c, _ in names}:
        dims = N.naming_dims(c, CATEGORIES.get(c, {"kind": "oneshot"}))
        fam = [n for cc, n in names if cc == c]
        for n in fam:
            t = "-" + n + "-"
            both += [(c, n) for _, pair in dims if pair[0] and pair[1]
                     and f"-{pair[0]}-" in t and f"-{pair[1]}-" in t]
        for i, a in enumerate(fam):
            twins += [(c, a, b) for b in fam[i + 1:] if N.twin_names(a, b)]
    yield Result(FAIL if both else PASS, "no folder name holds both sides of a trait", _fmt(both))
    yield Result(WARN if twins else PASS, "sibling folder names differ by more than order or one missing word",
                 _fmt(twins, 5))
    filler = [(c, n) for c, n in names if "neutral" in n.split("-") or (c == "DRUMLOOPS" and n.endswith("-drums"))]
    yield Result(FAIL if filler else PASS, "no filler traits", _fmt(filler))
    rng = [n for _, n in names if (m := re.search(r"(?:^|-)(\d{3})-(\d{3})bpm", n)) and any(int(x) % 5 for x in m.groups())]
    yield Result(FAIL if rng else PASS, "tempo ranges on multiples of 5", _fmt(rng))
    dup = []
    for c, fl in ctx.fams.items():
        seen_ = []
        for f in fl:
            if N.name_collides(f["family"], seen_):
                dup.append((c, f["family"]))
            seen_.append(f["family"].lower())
    yield Result(FAIL if dup else PASS, "folder names unique per category (word order ignored, "
                 f"tempos within {N.SAME_TEMPO_BPM} BPM the same)", _fmt(dup))
    # one name per file on every device: master names are canonical and sized for the
    # tightest device path, family names too
    from ..devices.exporter import _sanitize_filename, canonical_stem
    from .curate_config import NAMES, STEM_MAX, FAMILY_NAME_MAX
    longfam = sorted({(c, n) for c, n in names if len(n) > FAMILY_NAME_MAX})
    yield Result(FAIL if longfam else PASS, f"folder names within {FAMILY_NAME_MAX} characters", _fmt(longfam))
    badname = []
    for c, e in ctx.entries:
        st = os.path.splitext(os.path.basename(e["out"]))[0]
        base = re.sub(r"_\d+$", "", st) if len(st) > STEM_MAX else st
        # names = "keep" leaves the source's words; either way a name is FAT-safe and short
        ok = _sanitize_filename(base, STEM_MAX) == base if NAMES == "keep" \
            else canonical_stem(base, STEM_MAX) == base
        if len(st) > STEM_MAX + 3 or not ok:
            badname.append((c, e["out"]))
    yield Result(FAIL if badname else PASS, f"file names {NAMES} and within {STEM_MAX} characters "
                 "(the same on every device)", _fmt(badname))
    from .curate_config import NAME_EXCLUDE_WORDS
    xw = [(c, n) for c, n in names if set(n.split("-")) & NAME_EXCLUDE_WORDS.get(c, set())]
    yield Result(FAIL if xw else PASS, "no stab names outside STABS (SYNTH)", _fmt(xw))
    kp = [(c, n) for c, n in names if c in KEYS_EXCLUDE
          and set(n.split("-")) & {"piano", "rhodes", "wurlitzer", "clav", "clavinet", "keys", "key"}]
    yield Result(FAIL if kp else PASS, "no keys names in SYNTH/SUB/FX", _fmt(kp))
    from .curate import _acoustic_label, _acoustic_name_label, _is_synth_tagged
    from .curate_config import ACOUSTIC_ABLETON_TAGS

    def _ac_label(e):                   # the label the build names an ACOUSTIC folder by
        fn = os.path.splitext(os.path.basename(e["src"]))[0]
        plk = None if _is_synth_tagged(e.get("ableton")) else _acoustic_name_label(fn)
        lab = _acoustic_label({"ab": e.get("ableton") or [], "plk": plk, "path": e["src"]},
                              ACOUSTIC_ABLETON_TAGS)
        return re.sub(r"[^a-z0-9]+", "-", lab.lower()).strip("-") if lab else None
    from .curate import _phrase_inst_label

    def _ph_label(e):                   # PHRASES: the instrument most filenames name
        return _phrase_inst_label(os.path.basename(e["src"]))
    for cat, lab in (("ACOUSTIC", _ac_label), ("PHRASES", _ph_label)):
        if cat not in ctx.cats:
            continue
        by = defaultdict(list)
        for c, e in ctx.entries:
            if c == cat:
                by[e["family"]].append(lab(e))
        miss = []
        for fam, labs in by.items():
            top, n = Counter(x for x in labs if x).most_common(1)[0] if any(labs) else (None, 0)
            if top and n / len(labs) > 0.5 and not set(top.split("-")) <= set(fam.lower().split("-")):
                miss.append((fam, top))
        yield Result(FAIL if miss else PASS, f"{cat} folders named for the instrument most files name",
                     _fmt(miss))
    # names describe their folders: the naming phrase fits most files,
    # a genre / drum-machine word is backed by the files' paths, notes lead only in PIANO
    from .curate import _GENRE_RX
    from .curate_config import GENRE_SUPPORT_MIN, NAME_SUPPORT_MIN, NOTE_LEAD_CATS
    weak = [(c, f["family"], f["clap_support"]) for c, fl in ctx.fams.items() for f in fl
            if f.get("clap") and f.get("clap_support") is not None
            and f["clap_support"] < NAME_SUPPORT_MIN]
    yield Result(FAIL if weak else PASS, f"naming phrases fit at least {NAME_SUPPORT_MIN:.0%} of their "
                 "folder's files", _fmt(weak))
    from .curate import genre_evidence
    by_fam = defaultdict(list)
    for c, e in ctx.entries:
        by_fam[(c, e["family"])].append(genre_evidence(c, _rel(e["src"]), e.get("bpm_fold") or e.get("bpm"),
                                                       e.get("band")))
    genre = []
    for (c, fam), rels in by_fam.items():
        text = fam.lower().replace("-", " ")
        for w, (word_rx, path_rx) in _GENRE_RX.items():
            if word_rx.search(text):
                share = float(np.mean([bool(path_rx.search(r)) for r in rels]))
                if share < GENRE_SUPPORT_MIN:
                    genre.append((c, fam, w, round(share, 2)))
    yield Result(FAIL if genre else PASS, "genre and drum-machine words in names are backed by the files",
                 _fmt(genre))
    # instrument / source nouns in a naming phrase: backed by paths or Ableton tags
    from .curate import _INST_RX
    txt = defaultdict(list)
    for c, e in ctx.entries:
        txt[(c, e["family"])].append(_rel(e["src"]) + " | " + " ".join(e.get("ableton") or []))
    inst = []
    for c, fl in ctx.fams.items():
        for f in fl:
            ph = (f.get("clap") or "").lower()
            ts = txt.get((c, f["family"]))
            if not ph or not ts:
                continue
            for w, (word_rx, rx) in _INST_RX.items():
                if word_rx.search(ph):
                    share = float(np.mean([bool(rx.search(t)) for t in ts]))
                    if share < GENRE_SUPPORT_MIN:
                        inst.append((c, f["family"], w, round(share, 2)))
    yield Result(FAIL if inst else PASS, "instrument words in naming phrases are backed by the files",
                 _fmt(inst))
    lead = [(c, n) for c, n in names if c in CATEGORIES and c not in NOTE_LEAD_CATS
            and CATEGORIES[c].get("kind") == "oneshot" and re.match(r"^[a-g]s?-?\d-", n)]
    yield Result(FAIL if lead else PASS, "note leads only in " + "/".join(sorted(NOTE_LEAD_CATS)), _fmt(lead))
    from .curate_config import PLUCKED_NAME_RE, RENAMED_CATEGORIES
    pl = [(c, n) for c, n in names if c == "SYNTH" and PLUCKED_NAME_RE.search(n.replace("-", " "))]
    yield Result(FAIL if pl else PASS, "no plucked-string names in SYNTH", _fmt(pl))
    old = sorted(set(RENAMED_CATEGORIES) & set(ctx.cats))
    yield Result(FAIL if old else PASS, "no retired category folders ("
                 + ", ".join(f"{a}->{b}" for a, b in RENAMED_CATEGORIES.items()) + ")", _fmt(old))


def check_ratings(ctx):
    src_cat = defaultdict(set)
    for c, e in ctx.entries:
        src_cat[e.get("src")].add(c)
    leak = [p for p in ctx.drops if p in src_cat]
    yield Result(FAIL if leak else PASS, "no Drop in the master", _fmt(map(os.path.basename, leak)))
    back = [(os.path.basename(p), c) for p, c in ctx.misf.items()
            if c in src_cat.get(p, set()) and ctx.pins.get(p) != c]
    yield Result(FAIL if back else PASS, "no Misfiled file back in its folder", _fmt(back))
    lost = {p: c for p, c in ctx.pins.items() if c not in src_cat.get(p, set())}
    # a Keep whose source is gone (moved or deleted, or marked missing by the library walk)
    # can't be placed: passed over with a warning, as the build does
    gone = _gone_keeps(lost) if lost else {}
    lost_ = [(os.path.basename(p), c) for p, c in lost.items() if p not in gone]
    yield Result(FAIL if lost_ else PASS, f"every Keep in its category ({len(ctx.pins)} keeps)", _fmt(lost_))
    if gone:
        yield Result(WARN, "Keeps whose source is gone (moved or deleted; `fourier review rate <name> "
                           "clear` removes the rating)", _fmt(os.path.basename(p) for p in sorted(gone)))


def _gone_keeps(pins: dict) -> dict:
    """{source: category} of these Keep pins whose source is gone (ratings.missing_keeps)."""
    from .ratings import missing_keeps
    try:
        from ..db.session import db_exists, session_scope
        if db_exists():
            with session_scope() as s:
                return dict(missing_keeps(s, pins=pins))
    except Exception:
        pass
    return dict(missing_keeps(pins=pins))


def check_routing(ctx):
    import soundfile as sf
    from . import curate as C
    from .curate_config import (BPM_NAME_GUARD, BPM_NAME_RE, CATEGORIES, DUR_CAP, KEYBOARD_NAME_RE,
                                KEYS_CHORD_NAME_RE, KEYS_EXCLUDE, ORCH_PACK_EXCLUDE, PACK_HOME)
    wrong_home = [(c, e["out"]) for c, e in ctx.entries
                  if PACK_HOME.get(C._pack_of(_rel(e["src"]), e["src"]), c) != c and not ctx.pinned(c, e)]
    yield Result(FAIL if wrong_home else PASS, "PACK_HOME packs only in their home",
                 _fmt(wrong_home))
    keys = [(c, e["out"]) for c, e in ctx.entries if c in KEYS_EXCLUDE and not ctx.pinned(c, e)
            and KEYBOARD_NAME_RE.search(os.path.basename(e["src"]))]
    yield Result(FAIL if keys else PASS, f"no keys-named files in {'/'.join(sorted(KEYS_EXCLUDE))}", _fmt(keys))
    for cat, cfg in CATEGORIES.items():
        if not cfg.get("chord_band") or cat not in ctx.cats:
            continue
        mixed = []
        for c, e in ctx.entries:
            if c != cat:
                continue
            chroma, har = (ctx.extra or {}).get(e["src"], (None, None))
            if chroma is None and har is None:      # no DB: judge by name only
                chord = bool(KEYS_CHORD_NAME_RE.search(os.path.splitext(os.path.basename(e["src"]))[0]))
            else:
                chord = C._is_chord(os.path.basename(e["src"]), chroma, cfg.get("min_chroma", 1.6), har)
            if chord != e["family"].startswith("chord-"):
                mixed.append((e["family"], os.path.basename(e["out"]), "chord" if chord else "note"))
        yield Result(FAIL if mixed else PASS, f"{cat} chord and note folders kept apart", _fmt(mixed))
    irs = [(c, e["out"]) for c, e in ctx.entries if C._is_ir(_rel(e["src"]), e["src"])]
    yield Result(FAIL if irs else PASS, "no reverb impulse responses", _fmt(irs))
    prv = [(c, e["out"]) for c, e in ctx.entries if C._is_preview(_rel(e["src"]), e["src"])]
    yield Result(FAIL if prv else PASS, "no preset or kit previews", _fmt(prv))
    loopish = {c for c, cfg in CATEGORIES.items() if cfg.get("kind") == "loop"}
    # loops are exempt, as in the build (_row_override): by Sononym's class (recorded as
    # loop_row) or Ableton's "Loop" tag
    ov = [(c, e["out"], C._name_override(os.path.basename(e["src"]))) for c, e in ctx.entries
          if c not in loopish and not ctx.pinned(c, e) and "Loop" not in (e.get("ableton") or ())
          and not e.get("loop_row") and not C._is_synth_tagged(e.get("ableton"))
          and not ORCH_PACK_EXCLUDE.search(C._pack_of(_rel(e["src"]), e["src"]))
          and C._name_override(os.path.basename(e["src"])) not in (None, c)]
    yield Result(FAIL if ov else PASS, "name overrides hold (tuned percussion in ACOUSTIC, "
                 "cowbell/agogo in PERC, ...)", _fmt(ov))
    # (a patch from a synth / drum-machine pack is a synth whatever its name: an FM "Harp")
    # (loops too: the build reserves only one-shots for ACOUSTIC by name)
    plk = [(c, e["out"]) for c, e in ctx.entries if c == "SYNTH" and not ctx.pinned(c, e)
           and not e.get("loop_row")
           and C._is_acoustic_named(os.path.basename(e["src"]), e.get("ableton"))
           and not ORCH_PACK_EXCLUDE.search(C._pack_of(_rel(e["src"]), e["src"]))]
    ps = [(c, e["out"]) for c, e in ctx.entries if c == "SYNTH" and not ctx.pinned(c, e)
          and not e.get("loop_row") and C._perc_sourced(_rel(e["src"]), os.path.basename(e["src"]))]
    yield Result(FAIL if ps else PASS, "no acoustic percussion in SYNTH (drum / percussion folders and packs)",
                 _fmt(ps))
    yield Result(FAIL if plk else PASS, "no plucked-string or tuned-percussion files in SYNTH "
                 "(they live in ACOUSTIC)", _fmt(plk))
    nf = []
    for c, e in ctx.entries:
        pat = CATEGORIES.get(c, {}).get("noise")
        if pat and not ctx.pinned(c, e) and re.search(pat, os.path.basename(e["src"]), re.I):
            nf.append((c, e["out"]))
    yield Result(FAIL if nf else PASS, "name filters hold (no drum hits in FX, no kicks in SNARES, ...)",
                 _fmt(nf))
    leak = [e["out"] for c, e in ctx.entries if c == "FX" and not ctx.pinned(c, e)
            and not C._scratch_named(os.path.basename(e["src"]), _rel(e["src"]))   # a named scratch is FX's
            and not C._is_fx_transition(os.path.basename(e["src"]))                # ...and a named sweep
            and C._fx_drum_leak(e.get("sononym"), e.get("ableton"))]
    yield Result(FAIL if leak else PASS, "no kit drums in FX on Ableton's \"Sound FX\" alone", _fmt(leak))
    syn_ac = [e["out"] for c, e in ctx.entries if c == "ACOUSTIC" and not ctx.pinned(c, e)
              and C._is_synth_tagged(e.get("ableton"))
              and not C._acoustic_tagged(e.get("ableton"), os.path.basename(e["src"]),
                                         C._pack_of(_rel(e["src"]), e["src"]))]
    yield Result(FAIL if syn_ac else PASS, "no synth-tagged voices in ACOUSTIC", _fmt(syn_ac))
    ac = [e["out"] for c, e in ctx.entries if c == "ACOUSTIC" and not ctx.pinned(c, e)
          and e.get("ableton") and not C._acoustic_tagged(e["ableton"], os.path.basename(e["src"]),
                                                          C._pack_of(_rel(e["src"]), e["src"]),
                                                          CATEGORIES["ACOUSTIC"]["ableton_any"])
          and not C._acoustic_reserved(os.path.basename(e["src"]), e.get("ableton"), _rel(e["src"]))
          and not CATEGORIES["ACOUSTIC"]["pack_re"].search(C._pack_of(_rel(e["src"]), e["src"]))]
    yield Result(FAIL if ac else PASS, "ACOUSTIC catch-all tags backed by an instrument name or pack", _fmt(ac))
    # named winds and the orchestra folders (ORCH_PATH_RE) are ACOUSTIC's (one-shots, as in the build)
    wind = [(c, e["out"]) for c, e in ctx.entries if c != "ACOUSTIC" and c not in loopish
            and not ctx.pinned(c, e)
            and not ORCH_PACK_EXCLUDE.search(C._pack_of(_rel(e["src"]), e["src"]))
            and ((C._is_wind_named(os.path.basename(e["src"]), e.get("ableton"))
                  and not e.get("loop_row") and "Loop" not in (e.get("ableton") or ()))
                 or C._orch_path(_rel(e["src"])) is not None)]
    yield Result(FAIL if wind else PASS, "named wind / brass one-shots and orchestra folders only in ACOUSTIC",
                 _fmt(wind))
    from .curate_config import FX_TRANSITION_FROM, MIRROR_ROOT_RE
    both = defaultdict(set)
    for c, e in ctx.entries:
        both[e["src"]].add(c)
    two = [(os.path.basename(k), "/".join(sorted(v))) for k, v in both.items() if len(v) > 1]
    yield Result(FAIL if two else PASS, "every source file in one category only", _fmt(two))
    org = [(c, e["out"]) for c, e in ctx.entries if c != "PIANO" and c not in loopish and not ctx.pinned(c, e)
           and not e.get("loop_row") and C._is_organ_named(os.path.basename(e["src"]))]
    yield Result(FAIL if org else PASS, "named organs only in PIANO (organ stabs in STABS)", _fmt(org))
    from .curate_config import VOX_CHOIR_RE, VOX_CHOIR_SHARE, BUDGETS
    nch = sum(1 for c, e in ctx.entries if c == "VOX" and VOX_CHOIR_RE.search(_rel(e["src"])))
    lim = int(np.ceil(VOX_CHOIR_SHARE * BUDGETS.get("VOX", 375)))
    if "VOX" in ctx.cats:
        yield Result(FAIL if nch > lim else PASS, f"choirs at most {VOX_CHOIR_SHARE:.0%} of VOX",
                     f"{nch} of {lim} allowed")
    trn = [(c, e["out"]) for c, e in ctx.entries if c in FX_TRANSITION_FROM and not ctx.pinned(c, e)
           and not e.get("loop_row") and C._is_fx_transition(os.path.basename(e["src"]))]
    yield Result(FAIL if trn else PASS, "named risers / downlifters / sweeps only in FX", _fmt(trn))
    kk = [(c, e["out"]) for c, e in ctx.entries if c in C.KIT_CATS and not ctx.pinned(c, e)
          and KEYBOARD_NAME_RE.search(os.path.basename(e["src"])) and not C._drum_named(os.path.basename(e["src"]))]
    yield Result(FAIL if kk else PASS, "no keys-named files in the drum categories", _fmt(kk))
    sty = [e["out"] for c, e in ctx.entries if CATEGORIES.get(c, {}).get("path_exclude") is not None
           and not ctx.pinned(c, e) and CATEGORIES[c]["path_exclude"].search(_rel(e["src"]))]
    yield Result(FAIL if sty else PASS, "no dubstep / trap loops in DRUMLOOPS", _fmt(sty))
    mir = [e["out"] for c, e in ctx.entries if MIRROR_ROOT_RE.search(_rel(e["src"])) and not e.get("mirror_only")]
    yield Result(WARN if mir else PASS, "no files from mirror folders (copies of packs kept elsewhere) unless only there",
                 _fmt(mir))
    syn = [e["out"] for c, e in ctx.entries if c == "ACOUSTIC" and not ctx.pinned(c, e)
           and ORCH_PACK_EXCLUDE.search(C._pack_of(_rel(e["src"]), e["src"]))]
    yield Result(FAIL if syn else PASS, "no synth or drum-machine packs in ACOUSTIC", _fmt(syn))
    durs = {}
    for c, e in ctx.entries:
        try:
            durs[(c, e["out"])] = sf.info(str(ctx.path(c, e))).duration
        except Exception:
            durs[(c, e["out"])] = None
    wout = [(c, e["out"]) for c, e in ctx.entries if c != C.WAVES_CATEGORY and not ctx.pinned(c, e)
            and C._is_wave(_rel(e["src"]), (durs[(c, e["out"])] or 0) + 0.01)]
    yield Result(FAIL if wout else PASS, "no waves outside WAVES", _fmt(wout))
    if C.WAVES_CATEGORY in ctx.cats:
        nonw = [e["out"] for c, e in ctx.entries if c == C.WAVES_CATEGORY and not C.WAVE_PATH_RE.search(os.path.dirname(e["src"]))]
        yield Result(FAIL if nonw else PASS, "WAVES holds only waves", _fmt(nonw))
    longf = [(c, e["out"], round(durs[(c, e["out"])], 1)) for c, e in ctx.entries
             if DUR_CAP.get(c) and durs[(c, e["out"])] and durs[(c, e["out"])] > DUR_CAP[c] + 0.05
             and not ctx.pinned(c, e)]
    yield Result(FAIL if longf else PASS, "length caps hold", _fmt(longf))
    bpm = [(c, e["out"]) for c, e in ctx.entries if c in BPM_NAME_GUARD and not ctx.pinned(c, e)
           and BPM_NAME_RE.search(os.path.basename(e["src"]))]
    yield Result(FAIL if bpm else PASS, "no bpm-named phrases in one-shot folders", _fmt(bpm))
    # one note of an instrument is enough: at most SIBLING_CAP files of a multisample set
    # a folder (a Keep is exempt), in the categories where it applies
    from .curate_config import (LOOP_DIR_ONESHOT_CATS, LOOP_DIR_ONESHOT_MAX_S, SIBLING_CAP,
                                SIBLING_CAP_CATS)
    sets = Counter((c, e["family"], C._sibling_key(e["src"])) for c, e in ctx.entries
                   if c in SIBLING_CAP_CATS and not ctx.pinned(c, e) and C._sibling_key(e["src"]))
    many = [(c, f, k[1], n) for (c, f, k), n in sets.items() if n > SIBLING_CAP]
    yield Result(FAIL if many else PASS, f"at most {SIBLING_CAP} notes of one multisample set a folder",
                 _fmt(many))
    # (a rated Keep still lands there: pins win over require_tempo, as the build has it)
    unpinned = {(c, e["family"]) for c, e in ctx.entries if not ctx.pinned(c, e)}
    free = [(c, f["family"]) for c, fl in ctx.fams.items() for f in fl
            if CATEGORIES.get(c, {}).get("require_tempo") and f["family"].startswith("freetempo")
            and (c, f["family"]) in unpinned]
    yield Result(FAIL if free else PASS, "loops without a whole-bar tempo are left out where required",
                 _fmt(free))
    lf = [(c, e["out"]) for c, e in ctx.entries if c in LOOP_DIR_ONESHOT_CATS and not ctx.pinned(c, e)
          and (durs.get((c, e["out"])) or 0) > LOOP_DIR_ONESHOT_MAX_S + 0.05
          and C._in_loop_folder(_rel(e["src"])) and ctx.misf.get(e["src"]) != "PHRASES"]
    yield Result(FAIL if lf else PASS, "no long loop-folder files in the tonal one-shot folders", _fmt(lf))
    # a file named as one kind of drum lives in that category, and a named scratch in FX
    # (a Keep, a loop or a Misfiled verdict there are exempt)
    from .curate_config import DRUM_NAME_FROM, SCRATCH_HOME
    # (a Misfiled verdict, yours or the detector's, covers its multisample siblings too)
    _msib = {(C._sibling_key(p), cat) for p, cat in ctx.misf.items() if C._sibling_key(p)}
    wrong = []
    for c, e in ctx.entries:
        if c not in DRUM_NAME_FROM or CATEGORIES.get(c, {}).get("kind") != "oneshot" \
                or ctx.pinned(c, e) or e.get("loop_row"):
            continue
        fn = os.path.basename(e["src"])
        want = SCRATCH_HOME if C._scratch_named(fn, _rel(e["src"])) else C._drum_named_in(
            fn, _rel(e["src"]), ctx.fallback)
        # (a name override wins, as in the build: "Kick_Cowbell_02" is a cowbell)
        if want and want != c and ctx.misf.get(e["src"]) != want \
                and (C._sibling_key(e["src"]), want) not in _msib and C._name_override(fn) != c:
            wrong.append((c, e["out"], want))
    yield Result(FAIL if wrong else PASS, "files named as a drum (or a scratch) live in that category",
                 _fmt(wrong))
    # chord stabs live in STABS: a short single-hit one-shot named as a chord
    from .curate_config import STAB_CHORD_FROM
    if ctx.prof is None:
        yield Result(WARN, "chord-named stabs live in STABS", "skipped (no DB)")
    else:
        cs = []
        for c, e in ctx.entries:
            if c not in STAB_CHORD_FROM or ctx.pinned(c, e) or ctx.misf.get(e["src"]) == "STABS":
                continue
            if not C._stab_named(os.path.basename(e["src"])):
                continue
            dur = _build_duration(ctx, e["src"])
            n_ev = (ctx.prof.get(e["src"]) or (None,))[0]
            if dur is not None and C._is_named_stab(os.path.basename(e["src"]), dur, n_ev) \
                    and not e.get("loop_row") \
                    and not C._is_fx_transition(os.path.basename(e["src"])):
                cs.append((c, e["out"]))
        yield Result(FAIL if cs else PASS, "chord-named stabs live in STABS", _fmt(cs))
    ns = [e["out"] for c, e in ctx.entries if CATEGORIES.get(c, {}).get("stab_names")
          and not ctx.pinned(c, e) and not C._stab_named(os.path.basename(e["src"]))]
    yield Result(FAIL if ns else PASS, "STABS holds only files named as chords or stabs", _fmt(ns))
    from .curate_config import PHRASES_CATEGORY as PH
    if PH in CATEGORIES:
        if ctx.phrase is None:
            yield Result(WARN, "musical phrases only in PHRASES", "skipped (no DB)")
        else:
            # the build's own test (curate._is_phrase) on each source row; a Keep is exempt, and
            # a phrase rated Misfiled in PHRASES may live in a one-shot folder
            # (a phrase-marked loop CLAP heard as a drum loop, without Sononym, is the drum loops')
            out = [(c, e["out"]) for c, e in ctx.entries if c != PH and not ctx.pinned(c, e)
                   and e["src"] in ctx.phrase and ctx.misf.get(e["src"]) != PH and not e.get("rehomed")]
            yield Result(FAIL if out else PASS, "no musical phrases outside PHRASES "
                         "(one-shot folders hold one-shots)", _fmt(out))
            # (a drum-loop candidate CLAP heard as a phrase, without Sononym, is PHRASES')
            non = [e["out"] for c, e in ctx.entries if c == PH and not ctx.pinned(c, e)
                   and e["src"] not in ctx.phrase and not e.get("rehomed")]
            yield Result(FAIL if non else PASS, "PHRASES holds only musical phrases", _fmt(non))
        notempo = [e["out"] for c, e in ctx.entries if c == PH and not ctx.pinned(c, e) and not e.get("bpm")]
        yield Result(FAIL if notempo else PASS, "every phrase has a whole-bar tempo", _fmt(notempo))
    md = defaultdict(set)
    for c, e in ctx.entries:
        if CATEGORIES.get(c, {}).get("kind") != "instrument" and e.get("out_md5"):
            md[e["out_md5"]].add(c)
    twins = [sorted(cs) for cs in md.values() if len(cs) > 1]
    if ctx.prof is not None:
        fh = defaultdict(set)
        for c, e in ctx.entries:
            h = (ctx.prof.get(e["src"]) or (None,) * 4)[3]
            if h and CATEGORIES.get(c, {}).get("kind") != "instrument":
                fh[h].add(c)
        twins += [sorted(cs) for cs in fh.values() if len(cs) > 1]
    yield Result(FAIL if twins else PASS, "no identical audio in two categories", _fmt(twins))
    if ctx.prof is None:
        yield Result(WARN, "sample-chain guard", "skipped (no DB)")
        return
    unprof = sum(1 for c, e in ctx.entries if (ctx.prof.get(e["src"]) or (None,))[0] is None)
    chains = []
    for c, e in ctx.entries:
        n, rg, ec, _ = ctx.prof.get(e["src"]) or (None,) * 4
        if not ctx.pinned(c, e) and C._is_sample_chain(C._chain_guard_mode(c, CATEGORIES.get(c, {})), n, rg, ec):
            chains.append((c, e["out"], n))
    yield Result(FAIL if chains else PASS, "no sample chains in guarded categories", _fmt(chains))
    if unprof:
        yield Result(WARN, "files without an event profile (run `fourier tools analyze --only events`)", str(unprof))


def _src_frames(src):
    import soundfile as sf
    try:
        i = sf.info(src)
        return i.frames, i.samplerate
    except Exception:
        return None, None


def check_bands(ctx):
    """Band categories (HATS closed/open, FX types, DRUMLOOPS full/tops, WAVES cycle/table,
    PIANO note/chord): every folder holds one band, leads with it, and each file's band
    recomputed from the manifest (name, length, Sononym labels, Ableton tags) agrees."""
    import soundfile as sf
    from . import curate as C
    from .curate_config import CATEGORIES
    for cat, cfg in CATEGORIES.items():
        bt = cfg.get("band")
        if not bt or cat not in ctx.cats:
            continue
        ents = [e for c, e in ctx.entries if c == cat]
        if not any("band" in e for e in ents):
            yield Result(WARN, f"{cat} bands recorded", "manifest predates bands (rebuild)")
            continue
        fam_bands = defaultdict(set)
        for e in ents:
            fam_bands[e["family"]].add(e.get("band"))
        mixed = [(f, sorted(map(str, b))) for f, b in fam_bands.items() if len(b) > 1]
        fb = {f["family"]: f.get("band") for f in ctx.fams.get(cat, [])}
        mixed += [(f, fb.get(f), next(iter(b))) for f, b in fam_bands.items()
                  if len(b) == 1 and fb.get(f) != next(iter(b))]
        yield Result(FAIL if mixed else PASS, f"{cat} folders hold one {bt} band each", _fmt(mixed))
        lead = []
        for f, band in fb.items():
            parts = f.lower().split("-")
            if bt in ("hat", "fx", "wave", "acoustic", "phrase"):
                ok = parts[0] == band
            elif bt == "chord":
                ok = (parts[0] == "chord") == (band == "chord")
            elif bt == "loop":             # after the tempo lead: 124bpm-tops-, 090-100bpm-tops-
                rest = re.sub(r"^(\d{3}(-\d{3})?bpm|freetempo)-", "", f.lower()).split("-")
                ok = (rest[0] == "tops") == (band == "tops")
            else:
                ok = True
            if not ok:
                lead.append((f, band))
        yield Result(FAIL if lead else PASS, f"{cat} folder names lead with their band", _fmt(lead))
        if cfg.get("band_share"):
            nb = Counter(fb.values())
            # (a library-scaled master guarantees a band one folder: packs/scale.py)
            bmin = 1 if ctx.man.get("scale") else cfg.get("band_min_folders", 1)
            few = [(b, nb.get(b, 0)) for b in cfg["band_share"] if nb.get(b, 0) < bmin]
            if ctx.man.get("scale"):        # a small library may have none of a band at all
                few = [(b, n) for b, n in few if n]
            files = Counter(e.get("band") for e in ents)
            # a band short of its share because its whole pool is in is not a problem
            spent = {b for b in cfg["band_share"]
                     if all(f.get("copied", 0) >= f.get("n", 0) for f in ctx.fams.get(cat, [])
                            if f.get("band") == b)}
            # (bands over their share take what an exhausted band couldn't hold)
            off = [(b, files.get(b, 0), int(round(sh * len(ents)))) for b, sh in cfg["band_share"].items()
                   if files.get(b, 0) < sh * len(ents) - 0.05 * len(ents) and b not in spent]
            yield Result(WARN if (few or off) else PASS, f"{cat} band shares and folder minimums hold",
                         _fmt(few + off, 4))
        if bt == "chord":
            continue                        # recomputed from the DB in check_routing
        wrong = []
        for e in ents:
            fn = os.path.basename(e["src"])
            if bt in ("hat", "wave"):
                fr, sr = _src_frames(e["src"])
                if fr is None:
                    try:
                        i = sf.info(str(ctx.root / cat / e["out"]))
                        fr, sr = i.frames, i.samplerate
                    except Exception:
                        continue
                d = fr / sr
                if bt == "wave":
                    got = {C._wave_band(d, sr)}
                else:                       # a file on the length boundary may read either way
                    got = {C._hat_band(fn, d - 0.02), C._hat_band(fn, d + 0.02)}
            elif bt == "fx":
                if not e.get("sononym") and not e.get("ableton"):
                    continue                # no labels recorded (pinned / re-homed)
                got = {C._fx_band(e.get("sononym"), e.get("ableton"), fn, _rel(e["src"]))}
                if got == {"misc"}:         # placed by CLAP among the library FX types
                    got = set(C.BAND_ORDER["fx"]) - {"synth", "scratch", "misc", "riser", "downlifter"}
            else:
                got = {C._band_of(bt, fn, son_labels=e.get("sononym") or (), ab_tags=e.get("ableton") or (),
                                  pack=C._pack_of(_rel(e["src"]), e["src"]), rel=_rel(e["src"]))}
            if e.get("band") not in got:
                wrong.append((e["out"], e.get("band"), "/".join(sorted(got))))
        yield Result(FAIL if wrong else PASS, f"{cat} files are in the right {bt} band", _fmt(wrong))


def check_audio(ctx):
    import soundfile as sf
    from .curate_config import CATEGORIES
    from .curate import WAVES_CATEGORY, _rms_env
    from . import curate as C
    from .curate_config import QUIET_RMS_DB, ONESHOT_RMS_CEIL_DB, END_HOT_DB
    dc, wdc, wlen, tails, quiet, leads, hot, ends, stereo_bad = [], [], [], [], [], [], [], [], []
    hard, seams, tiny, phase_bad, phase_warn = [], [], [], [], []
    from .curate_config import LEVEL_CATS, LEVEL_TOL_DB
    st = defaultdict(list)            # (category, folder) -> [(file, short-term dB, room to lift)]
    for c, e in ctx.entries:
        p = ctx.path(c, e)
        try:
            x, sr = sf.read(str(p), always_2d=True)
        except Exception as ex:
            dc.append((c, e["out"], f"unreadable: {ex}"))
            continue
        pk = float(np.abs(x).max()) if x.size else 0.0
        mean = float(x.mean()) if x.size else 0.0
        if c == WAVES_CATEGORY:
            if abs(mean) > 0.002:
                wdc.append((e["out"], round(mean, 4)))
            try:
                if sf.info(e["src"]).frames != len(x):
                    wlen.append((e["out"], sf.info(e["src"]).frames, len(x)))
            except Exception:
                pass
        elif pk > 0 and abs(mean) > DC_MAX_SHARE * pk:
            dc.append((c, e["out"], round(abs(mean) / pk, 3)))
        _k = CATEGORIES.get(c, {}).get("kind")
        if x.shape[1] == 2 and _k != "waves" and x.size:
            from .curate_config import MONO_CATS, NEAR_MONO_SIDE_DB
            _mid, _side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
            _rm, _rs = float(np.sqrt(np.mean(_mid ** 2))), float(np.sqrt(np.mean(_side ** 2)))
            if c in MONO_CATS or _rs <= _rm * 10 ** ((NEAR_MONO_SIDE_DB - 1) / 20):
                stereo_bad.append((c, e["out"]))
        if x.size and _k not in ("loop", "waves") and not ctx.pinned(c, e):
            _r = float(np.sqrt(np.mean(np.square(x))))
            if _r <= 0 or 20 * np.log10(_r) < QUIET_RMS_DB - 0.5:
                quiet.append((c, e["out"], round(20 * np.log10(_r), 1) if _r > 0 else "silent"))
            if _r > 0 and c in ONESHOT_RMS_CEIL_DB and 20 * np.log10(_r) > ONESHOT_RMS_CEIL_DB[c] + 0.5:
                hot.append((c, e["out"], round(20 * np.log10(_r), 1)))
            _n1 = max(1, int(sr * 0.001))
            if len(x) >= 16 and pk > 0:
                _end = float(np.abs(x[-4:]).max())       # the last samples: a fade ends near zero
                if _end > pk * 10 ** (END_HOT_DB / 20):
                    ends.append((c, e["out"], round(20 * np.log10(_end / pk), 1)))
        if c in LEVEL_CATS and x.size and not ctx.pinned(c, e):
            _v = C._short_term_db(x.mean(axis=1), sr)
            _rr = float(np.sqrt(np.mean(np.square(x))))
            _rdb = 20 * np.log10(_rr) if _rr > 0 else -120.0
            # room to lift: under the peak ceiling and the category's RMS ceiling
            _room = pk < 10 ** (-1.1 / 20) and _rdb < ONESHOT_RMS_CEIL_DB.get(c, 0.0) - 0.5
            if np.isfinite(_v):
                st[(c, e["family"])].append((e["out"], _v, _room))
        _lf = CATEGORIES.get(c, {}).get("trim_lead_db")
        if _lf is not None and x.size:
            env = _rms_env(np.abs(x).mean(axis=1), sr)
            nzl = np.where(env > float(env.max()) * 10 ** (_lf / 20))[0]
            if len(nzl) and nzl[0] / sr > 0.2:     # (the second DC pass can shift it a little)
                leads.append((c, e["out"], round(nzl[0] / sr, 2)))
        if _k not in ("loop", "waves") and x.size and pk > 0 and not ctx.pinned(c, e):
            _a = np.abs(x).max(axis=1)
            if _a[0] > pk * 10 ** (C.EDGE_HOT_DB / 20):
                hard.append((c, e["out"], round(float(_a[0]), 3)))
            _env = _rms_env(np.abs(x).mean(axis=1), sr)
            _nz = _tail_sound(_env, pk * 10 ** ((C.TAIL_PEAK_FLOOR_DB - 3) / 20) / max(float(_env.max()), 1e-12), sr)
            if len(_nz) and (len(_env) - 1 - _nz[-1]) / sr > TAIL_SLACK_S:
                tails.append((c, e["out"], round((len(_env) - 1 - _nz[-1]) / sr, 2)))
        if _k == "loop" and x.size and pk > 0:
            _a = np.abs(x).max(axis=1)
            if max(_a[0], _a[-1]) > pk * 10 ** (C.EDGE_HOT_DB / 20):
                seams.append((c, e["out"], round(float(_a[0]), 3), round(float(_a[-1]), 3)))
        if x.shape[1] == 2 and _k != "waves":
            from .curate_config import PHASE_WARN_CORR, phase_fix_limits
            _corr, _loss = C.phase_stats(x)
            _fc, _fl = phase_fix_limits(c)
            if (_corr is not None and _corr < _fc) or _loss > _fl:
                phase_bad.append((c, e["out"], None if _corr is None else round(_corr, 2), round(_loss, 1)))
            elif _corr is not None and _corr < PHASE_WARN_CORR:
                phase_warn.append((c, e["out"], round(_corr, 2), round(_loss, 1)))
        if c == WAVES_CATEGORY and len(x) < C.WAVE_MIN_SAMPLES:
            tiny.append((e["out"], len(x)))
        if _k == "instrument" and x.size:
            env = _rms_env(np.abs(x).mean(axis=1), sr)            # the envelope the trim uses
            # 3 dB under the trim floor: a noise floor hovering right at -60 dB flickers
            # across it, so only real silence past the trim point counts
            nz = _tail_sound(env, 10 ** (TAIL_CHECK_DB / 20), sr)
            if len(nz) and (len(env) - 1 - nz[-1]) / sr > TAIL_SLACK_S:
                tails.append((c, e["out"], round((len(env) - 1 - nz[-1]) / sr, 2)))
    unread = [d for d in dc if isinstance(d[2], str)]
    dc = [d for d in dc if not isinstance(d[2], str)]
    if unread:
        yield Result(FAIL, "every file readable", _fmt(unread))
    lvl = FAIL if len(dc) > DC_FAIL_COUNT else (WARN if dc else PASS)
    yield Result(lvl, f"DC offset under {DC_MAX_SHARE:.0%} of peak", _fmt(dc))
    if WAVES_CATEGORY in ctx.cats:
        yield Result(FAIL if wdc else PASS, "WAVES carry no DC (loop cleanly)", _fmt(wdc))
        yield Result(FAIL if wlen else PASS, "WAVES exported sample-exact", _fmt(wlen))
    yield Result(FAIL if tails else PASS, "tails trimmed (no quiet past -60 dB of the envelope or -50 dB of the peak)", _fmt(tails))
    yield Result(FAIL if hard else PASS, "one-shots start quietly (faded in, no click)", _fmt(hard))
    yield Result(FAIL if phase_bad else PASS, "no stereo file cancels in mono (one channel flipped)", _fmt(phase_bad))
    yield Result(WARN if phase_warn else PASS, "stereo files partly out of phase (check by ear)", _fmt(phase_warn))
    yield Result(FAIL if seams else PASS, "loops start and end quietly (no click at the loop point)", _fmt(seams))
    if C.WAVES_CATEGORY in ctx.cats:
        yield Result(FAIL if tiny else PASS, f"single cycles at least {C.WAVE_MIN_SAMPLES} samples", _fmt(tiny))
    yield Result(FAIL if quiet else PASS, f"no near-silent one-shots (RMS under {QUIET_RMS_DB:.0f} dBFS)", _fmt(quiet))
    yield Result(FAIL if hot else PASS, "no one-shot hotter than its category's RMS ceiling", _fmt(hot))
    yield Result(FAIL if ends else PASS, "one-shots end quietly (faded, no click)", _fmt(ends))
    yield Result(FAIL if stereo_bad else PASS, "stereo only where it's real (KICKS / SUB mono, no near-mono stereo)",
                 _fmt(stereo_bad))
    from .curate_config import RETUNE_CATS
    rt = [(c, e["out"], e.get("retune")) for c, e in ctx.entries if c in RETUNE_CATS and not e.get("loop_row")
          and (e.get("retune") is None and not e.get("pitch_conflict")
               and C._retune_shift(c, os.path.basename(e["src"]), None, bare=ctx.fallback)[0] is not None
               or (e.get("retune") is not None and abs(e["retune"]) > 6.5))]
    yield Result(FAIL if rt else PASS, "tonal one-shots with a named root retuned to C (at most 6 semitones)",
                 _fmt(rt))
    stale = [(c, e["out"]) for c, e in ctx.entries if c in RETUNE_CATS and e.get("retune") is not None
             and e.get("root_src") == "name"
             and C._name_root_pc(os.path.basename(e["out"]), bare=ctx.fallback) not in (0, None)]
    for c, e in ctx.entries:          # ...and a detected root's note left in the name ("Saw2D1")
        if c in RETUNE_CATS and e.get("retune") is not None and e.get("root_midi") is not None:
            pc = int(round(e["root_midi"])) % 12
            stem = os.path.splitext(os.path.basename(e["out"]))[0]
            if pc != 0 and any((C._PC[g.group(1)] + (1 if g.group(2) in ("#", "s") else -1 if g.group(2) == "b" else 0)) % 12 == pc
                               for g in C._LOOSE_NOTE.finditer(stem)):
                stale.append((c, e["out"]))
    yield Result(FAIL if stale else PASS, "retuned files named for the note they play (C)", _fmt(stale))
    far = []
    for (c, fam), vs in st.items():
        if len(vs) < 3:
            continue
        med = float(np.median([v for _, v, _ in vs]))
        far += [(c, o, round(v - med, 1)) for o, v, room in vs
                if v > med + LEVEL_TOL_DB + 1.0 or (v < med - LEVEL_TOL_DB - 1.0 and room)]
    yield Result(WARN if far else PASS, f"melodic folders levelled (within {LEVEL_TOL_DB:.0f} dB of the "
                 "folder median, short-term loudness)", _fmt(far, 5))
    fb = [(c, e["out"]) for c, e in ctx.entries if e.get("dsp_fallback")]
    yield Result(FAIL if fb else PASS, "every file went through export processing (no verbatim fallbacks)",
                 _fmt(fb))
    yield Result(FAIL if leads else PASS, "files start at the sound (leads trimmed at their category's trim_lead_db)", _fmt(leads))


def verify_master(master_dir, session=None, audio=True, store_path=None, log=print):
    """Run every master check. Returns (ok, [Result])."""
    from .validate import validate_master
    prof = None
    if session is not None:
        from sqlalchemy import select
        from ..db.models import Sample, SampleFeatures
        prof = {p: (n, rg, ec, fh) for p, n, rg, ec, fh in session.execute(
            select(Sample.path, SampleFeatures.n_events, SampleFeatures.event_regularity,
                   SampleFeatures.event_echo, Sample.file_hash)
            .outerjoin(SampleFeatures, SampleFeatures.sample_id == Sample.id)).all()}
    ctx = _Ctx(master_dir, store_path=store_path, prof=prof)
    if session is not None:
        from ..metadata.rows import fetch, sample_select
        ctx.extra = {p: (ch, h) for p, ch, h in fetch(
            session, sample_select("path", "chroma_concentration", "harmonicity", session=session))}
        from . import curate as C
        srcs = {e.get("src") for _, e in ctx.entries}
        rows = C._with_name_tags(session, fetch(session, sample_select(      # as the build saw them
            "path", "rel_path", "filename", "duration_s", "ableton_tags",
            "classes", "categories", "bpm", "tempo_bpm", "harmonicity", "onset_rate_hz",
            session=session)))
        _fb = C._fallback(session)
        ctx.fallback = _fb
        ctx.phrase = {r.path for r in rows if r.path in srcs and C._is_phrase(r, _fb)}
        ctx.dur = {r.path: r.duration_s for r in rows if r.path in srcs and r.duration_s is not None}
    out = []
    for fn in (check_manifest, check_sets, check_budgets, check_naming, check_ratings, check_routing, check_bands,
               *( [check_audio] if audio else [])):
        out.extend(fn(ctx))
    _ok, vres = validate_master(str(master_dir), session=session, log=lambda *a: None)
    for level, check, detail in vres:
        out.append(Result(level, f"validate: {check}", detail))
    if not any(r.check.startswith("validate") and r.level == FAIL for r in out):
        out.append(Result(PASS, "validate (budget, ceiling, vendor, names, dup, hygiene)"))
    for r in out:
        log(f"{r.level}  {r.check}" + (f"  -- {r.detail}" if r.detail else ""))
    return not any(r.level == FAIL for r in out), out


def verify_render(master_dir, render_out, device, log=print):
    """Check a render of master_dir for device under render_out (as render_master_to_device
    wrote it). Returns (ok, [Result])."""
    import soundfile as sf
    from .curate import WAVES_CATEGORY
    from .render import CYCLE_CATEGORIES, _assignment, _card_prefix, _device_root, _full, expected_subtype
    c = device
    subtype = expected_subtype(device)
    base = Path(render_out) / _device_root(device) if _device_root(device) else Path(render_out)
    files, a = _assignment(master_dir, device, None)
    master = Path(master_dir)
    out = []
    got = {k: base / rel for k, rel in a.paths.items()}
    missing = [rel for k, rel in a.paths.items() if not got[k].exists()]
    out.append(Result(FAIL if missing else PASS, f"{device.device_id}: every master file rendered",
                      f"{len(a.paths)} expected" + (f"; missing {_fmt(missing)}" if missing else "")))
    mono = device.mono
    fmt_bad, wave_bad, quiet, fmts, clipped, too_long = [], [], [], Counter(), [], []
    max_s = float(device.max_duration_s) if device.max_duration_s else None
    for f in files:
        dst = got.get(f.key)
        if dst is None or not dst.exists():
            continue
        info = sf.info(str(dst))
        fmts[(info.samplerate, info.subtype, info.channels)] += 1
        if info.samplerate != c.sample_rate or info.subtype != subtype or (mono and info.channels != 1):
            fmt_bad.append(f.master)
        if max_s and f.category not in CYCLE_CATEGORIES and info.frames > (max_s + 0.01) * info.samplerate:
            too_long.append((f.master, round(info.frames / info.samplerate, 2)))
        if f.category != WAVES_CATEGORY:
            _y, _ = sf.read(str(dst), always_2d=True)
            if _clipped(_y, c.bit_depth):
                clipped.append(f.master)       # resampling overshoot clipped at full scale
        src = master / f.master
        # the rule itself, not render's constant, so a render regression can't hide here
        if f.category == WAVES_CATEGORY and sf.info(str(src)).frames != info.frames:
            wave_bad.append((f.master, sf.info(str(src)).frames, info.frames))
        if mono:
            x, _ = sf.read(str(src), always_2d=True)
            if x.shape[1] == 2 and x.size:
                m = x.mean(axis=1)
                if 10 * np.log10(float((m ** 2).mean()) / (float((x ** 2).mean()) + 1e-12) + 1e-12) < -6:
                    y, _ = sf.read(str(dst))
                    lvl = 20 * np.log10(np.sqrt(float((y ** 2).mean())) / np.sqrt(float((x ** 2).mean())) + 1e-12)
                    if lvl < MONO_OK_DB:
                        quiet.append((f.master, round(lvl, 1)))
    out.append(Result(FAIL if fmt_bad else PASS,
                      f"{device.device_id}: device format {c.sample_rate} Hz / {c.bit_depth}-bit"
                      + (" / mono" if mono else ""), _fmt(fmt_bad) or str(dict(fmts))))
    if max_s:
        out.append(Result(FAIL if too_long else PASS,
                          f"{device.device_id}: files within {max_s:g} s (audio.max_duration_s)", _fmt(too_long)))
    out.append(Result(FAIL if wave_bad else PASS, f"{device.device_id}: WAVES keep every sample", _fmt(wave_bad)))
    out.append(Result(FAIL if clipped else PASS, f"{device.device_id}: no clipped files (full-scale runs)",
                      _fmt(clipped)))
    if mono:
        out.append(Result(FAIL if quiet else PASS,
                          f"{device.device_id}: phase-inverted stereo keeps its level in mono", _fmt(quiet)))
    if c.max_path_length:
        prefix = _card_prefix(device)
        longest = max((len(_full(prefix, rel)) for rel in a.paths.values()), default=0)
        out.append(Result(FAIL if longest > c.max_path_length else PASS,
                          f"{device.device_id}: card paths within {c.max_path_length} chars", f"longest {longest}"))
    if c.files_per_folder:
        per_dir = Counter(Path(rel).parent.as_posix() for rel in a.paths.values())
        big = sorted(((d, n) for d, n in per_dir.items() if n > c.files_per_folder), key=lambda t: -t[1])
        out.append(Result(FAIL if big else PASS,
                          f"{device.device_id}: folders within {c.files_per_folder} files", _fmt(big)))
    mb = sum(p.stat().st_size for p in got.values() if p.exists()) / 1e6
    out.append(Result(PASS, f"{device.device_id}: image size", f"{mb:,.0f} MB"))
    for r in out:
        log(f"{r.level}  {r.check}" + (f"  -- {r.detail}" if r.detail else ""))
    return not any(r.level == FAIL for r in out), out


def _clipped(y, bit_depth) -> bool:
    """Whether a rendered file was clipped at full scale: 3 samples or more at the top code
    (16 bits and up). At 8 bits one code is 1/128 of full scale, so a peak normalized to
    within a code of it lands there without clipping: there it takes a run of 3 consecutive
    samples at the top or bottom code in one channel."""
    top = 1.0 - 2.0 / (2 ** bit_depth)
    hot = np.abs(y) >= top
    if bit_depth > 8:
        return int(hot.sum()) >= 3
    for ch in range(hot.shape[1]):
        h = hot[:, ch].astype(np.int8)
        if h.size >= 3 and int((h[:-2] & h[1:-1] & h[2:]).sum()):
            return True
    return False


def render_and_verify(master_dir, device_ids, keep_dir=None, log=print):
    """Render master_dir for each device into a scratch dir (no path lock is read or
    written) and verify it. Returns (ok, [Result])."""
    import shutil
    from ..devices.loader import DeviceLoader
    from .render import render_master_to_device
    ok, out = True, []
    scratch = keep_dir or tempfile.mkdtemp(prefix="fourier-verify-")
    try:
        for did in device_ids:
            dev = DeviceLoader().load(did)
            dst = os.path.join(scratch, did)
            s = render_master_to_device(master_dir, dev, dst, log=lambda *a: None, lock=None)
            if s.get("failed"):
                out.append(Result(FAIL, f"{did}: render failures", str(s["failed"])))
                log(f"FAIL  {did}: {s['failed']} files failed to render")
                ok = False
            o, r = verify_render(master_dir, dst, dev, log=log)
            ok &= o
            out += r
    finally:
        if not keep_dir:
            shutil.rmtree(scratch, ignore_errors=True)
    return ok, out
