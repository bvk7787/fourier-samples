"""The sound model: provider "fourier:sound", a category and a shape for every sample from
Fourier's own measurements of its audio.

A small classifier (multinomial logistic regression) over the CLAP embedding
(analysis/clap_features.py) and a few of Fourier's own librosa, onset and event measurements
(OWN_FEATURES). It gives each sample with a CLAP embedding

  labels       kind "category"  the most likely category (the taxonomy's built-in ones, or
                                "other"), confidence its probability
               kind "class"     class.oneshot or class.loop, confidence its probability
  descriptors  "p_category" and "p_loop" (the same probabilities), and "analysed", which
               carries the model's version, so a new model relabels every sample

What it learns from (metadata/train.py, `fourier tools train`, run against a library database):
  - the pack makers' own folder and file names, read by the path provider's rules
    (config/providers/path.yaml, metadata/shadow.py): a category word in the file's own name
    or its folder, never a pack's or vendor's name alone (training_label)
  - the user's own ratings (ratings.json): Keep in a category, Misfiled with a category to
    move it to (rating_label)
  - optionally, the taxonomy's CLAP prompts as a prior (the trainer's --clap-prior)
What it never uses, as a target or as a feature: another product's analysis or tags (the
labels and descriptors of the "sononym" and "ableton" providers, sononym_meta, the
sample_features columns computed from Sononym's readings, samples.ableton_tags). FORBIDDEN
names them; tests/test_sound_model.py checks this module and the trainer never read them.

Routing (packs/curate.py compute_homes) reads the labels only without Sononym: a sample no
name rule recognized, or one only the folders above its own (a pack's or vendor's name)
describe, is homed where the model is confident (curate.SOUND_MODEL_MIN,
SOUND_MODEL_OVERRIDE_MIN). With Sononym the labels are a report (`fourier why`, `fourier
tools db-stats --disagreements`) and route nothing.

The weights are the user's own: trained on their library (the analysis's sound step trains
them without Sononym when the names label enough samples, metadata/train.maybe_train; or
`fourier tools train`) into the Fourier home (<home>/sound_model.npz), since a sample
library's licence is its owner's and Fourier ships none. $FOURIER_SOUND_MODEL names another
file, or "off" for none. Without weights, or for a sample whose embedding came from another
CLAP model than the one the weights were trained on, the provider labels nothing and
everything works as it did without it.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import text

PROVIDER = "fourier:sound"
MODEL_ENV = "FOURIER_SOUND_MODEL"
FORMAT = 1                       # the weights file's layout (SoundModel.save / load)
OTHER = "other"                  # none of the built-in categories
SHAPES = ("class.oneshot", "class.loop")
ANALYSED = "analysed"
P_CATEGORY, P_LOOP = "p_category", "p_loop"
HOME_NAME = "sound_model.npz"     # <Fourier home>/sound_model.npz: the weights trained here
# a package's own weights, if a release ever ships some (one trained on openly licensed audio)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "sound_model.npz"
_CHUNK = 20000

# Fourier's own measurements the model reads beside the CLAP embedding: (feature, column,
# transform). Each is a sample_features column Fourier computes from the audio (librosa,
# events) or the file's own header (samples.duration_s); never a column computed from
# another product's analysis (FORBIDDEN).
OWN_FEATURES = (
    ("log_duration", "s.duration_s", "log"),
    ("log_onset_rate", "f.onset_rate_hz", "log1p"),
    ("harmonic_share", "f.harmonic_percussive_ratio", None),
    ("log_events", "f.n_events", "log1p"),
    ("log_centroid", "f.spectral_centroid_mean", "log"),
    ("flatness", "f.spectral_flatness_mean", None),
    ("chroma_focus", "f.chroma_concentration", None),
    ("log_attack_ms", "f.attack_time_ms", "log1p"),
    ("log_decay_ms", "f.decay_time_ms", "log1p"),
)
FEATURE_NAMES = tuple(f[0] for f in OWN_FEATURES)
FEATURE_SQL = ("SELECT s.id, f.clap_model, " + ", ".join(c for _, c, _ in OWN_FEATURES)
               + " FROM samples s JOIN sample_features f ON f.sample_id = s.id "
               "WHERE f.clap_embedding IS NOT NULL")

# What the model never reads: another product's tables, providers and the columns derived
# from them (db/models.py SampleFeatures, Tier 1). tests/test_sound_model.py checks that
# none of these appears in this module's code or the trainer's training code.
FORBIDDEN = ("sononym", "ableton", "sub_weight", "transient_score", "loop_confidence",
             "is_pitched", "bpm_reliable", "timbral_norm", "spectral_balance", "pitch_stability",
             "attack_class", "drum_subtype", "bpm_corrected", "derived_computed_at")


class SoundModelError(RuntimeError):
    pass


# --- the categories ------------------------------------------------------------------------

@lru_cache(maxsize=None)
def builtin_categories() -> tuple:
    """The taxonomy's own categories (config/taxonomy.yaml), in folder order; a library
    overlay's added ones aren't among them (a rating in one is "other")."""
    from .. import taxonomy
    return tuple(taxonomy.load()["order"])


def model_categories() -> tuple:
    return builtin_categories() + (OTHER,)


@lru_cache(maxsize=None)
def _label_categories() -> dict:
    """{canonical label: built-in category} from the taxonomy's labels, plus the labels that
    route by name (keys: PIANO; a scratch: the scratches' home)."""
    from .. import taxonomy
    doc = taxonomy.load()
    out = {}
    for cat in doc["order"]:
        for lab in doc["categories"][cat].get("labels") or ():
            out.setdefault(lab, cat)
    if "PIANO" in doc["categories"]:
        out.setdefault("keys", "PIANO")
    scr = (doc.get("homes") or {}).get("scratches")
    if scr:
        out.setdefault("scratch", scr)
    return out


def categories_of(labels) -> set:
    """The built-in categories these canonical labels name (shapes ignored)."""
    m = _label_categories()
    return {m[lab] for lab in labels or () if lab in m}


@lru_cache(maxsize=None)
def _groups() -> tuple:
    """(drum hit categories: the kit role; categories whose loops are phrases: the tonal
    ones and the instruments, but the voice)."""
    from .. import taxonomy
    doc = taxonomy.load()
    cats = doc["categories"]
    drums = frozenset(c for c in doc["order"] if "kit" in (cats[c].get("roles") or ()))
    tonal = frozenset(c for c in doc["order"]
                      if cats[c].get("kind") in ("oneshot", "instrument") and c not in drums
                      and not set(cats[c].get("labels") or ()) & {"vocal"}
                      and not any(str(lab).startswith("fx.") for lab in cats[c].get("labels") or ()))
    return drums, tonal


def _loop_category(kind: str) -> str | None:
    """The built-in category of a drum loop ("drums") or a musical phrase ("phrase")."""
    from .. import taxonomy
    doc = taxonomy.load()
    homes = doc.get("homes") or {}
    if kind == "phrase":
        return homes.get("phrases")
    loops = [c for c in doc["order"] if doc["categories"][c].get("kind") == "loop"
             and c != homes.get("phrases")]
    return loops[0] if loops else None


# a loop's own name or folder calls it drums ("Breaks", "Drum Loops", "Groove 03")
_DRUM_LOOP_WORDS = re.compile(r"(?<![a-z])(?:drums?|breaks?|breakbeats?|beats?|grooves?|tops?)(?![a-z])",
                              re.I)


def category_for(cats, loop: bool, drum_words: bool = False) -> str | None:
    """The one category these built-in categories (categories_of) make a sample: for a
    one-shot the one they name (none when they name two); for a loop the drum loops' (drum
    sounds, or `drum_words` and no sound named) or the phrases' (tonal sounds), and "other"
    for a voice or FX loop, which no loop category holds."""
    cats = set(cats or ())
    if not loop:
        return next(iter(cats)) if len(cats) == 1 else None
    drums, tonal = _groups()
    if (cats and cats <= drums) or (not cats and drum_words):
        return _loop_category("drums")
    if cats and cats <= tonal:
        return _loop_category("phrase")
    return OTHER if len(cats) == 1 else None


def training_label(rel_path: str, duration_s=None) -> tuple:
    """(category or None, shape or None) the pack maker's own names give a sample, read by
    the path provider's rules: a category word in the file's own name or its folder (not the
    folders above, where a pack's or vendor's name is), and one-shot or loop from the same
    place. Waves, acoustic instruments (by name, and the bowed strings the path words stand
    in for, curate._name_tags) and organs by their name rules (packs/rules.py), ahead of the
    words, as routing has them. A loop is a drum loop or a phrase (category_for). A drum hit
    named as one with no loop word is a one-shot. Ambiguous names give no category."""
    from ..packs import rules as R
    from ..packs.curate import _name_tags
    from .shadow import near_labels, shape_labels, shape_scope
    rel = rel_path or ""
    if R._is_ir(rel, rel) or R._is_preview(rel, rel):
        return None, None
    shapes = shape_labels(rel)
    shape = next(iter(shapes)) if len(shapes) == 1 else None
    name, folder = shape_scope(rel)
    builtin = builtin_categories()
    if R._is_wave(rel, duration_s):
        return ("WAVES" if "WAVES" in builtin else None), SHAPES[0]
    loop = shape == SHAPES[1]
    if not loop and "ACOUSTIC" in builtin and (
            R._acoustic_tagged(_name_tags(rel), name, R._pack_of(rel, rel)) or R._is_acoustic_named(name)
            or R._is_wind_named(name) or R._orch_path(rel) is not None):
        return "ACOUSTIC", shape
    if not loop and R._is_organ_named(name) and "PIANO" in builtin:
        return "PIANO", shape
    cats = categories_of(near_labels(rel))
    cat = category_for(cats, loop, bool(_DRUM_LOOP_WORDS.search(f"{name} {folder}")))
    if cat is None or loop:
        return cat, shape
    return cat, shape or (SHAPES[0] if cat in _groups()[0] else None)


def rating_label(rating: dict) -> tuple:
    """(category or None, shape or None) the user's own rating gives a sample: Keep in a
    category, or Misfiled with a category to move it to; a category the taxonomy doesn't
    have (a library overlay's own) is "other". A Drop, or a Misfiled with no target, says
    nothing about where a sample belongs (rating_excludes)."""
    from .. import taxonomy
    rating = rating or {}
    v = rating.get("verdict")
    cat = rating.get("category") if v == "keep" else rating.get("target") if v == "misfiled" else None
    if not cat:
        return None, None
    if cat not in builtin_categories():
        return OTHER, None
    kind = taxonomy.load()["categories"][cat].get("kind")
    return cat, {"loop": SHAPES[1], "oneshot": SHAPES[0], "waves": SHAPES[0]}.get(kind)


def rating_excludes(rating: dict) -> str | None:
    """The category a Misfiled rating says a sample isn't (its names may say it is)."""
    rating = rating or {}
    return rating.get("category") if rating.get("verdict") == "misfiled" else None


# --- features ------------------------------------------------------------------------------

def own_matrix(raw) -> np.ndarray:
    """(n, len(OWN_FEATURES)) raw measurements (None or NaN where missing) in the model's
    units: logs where a measurement spans decades. Missing stays NaN."""
    out = np.array(raw, dtype=np.float64).reshape(-1, len(OWN_FEATURES))
    for j, (_n, _c, how) in enumerate(OWN_FEATURES):
        x = out[:, j]
        with np.errstate(invalid="ignore", divide="ignore"):
            if how == "log":
                out[:, j] = np.log(np.clip(x, 1e-4, None))
            elif how == "log1p":
                out[:, j] = np.log1p(np.clip(x, 0.0, None))
    return out


def design(emb, own_raw, mean=None, std=None) -> np.ndarray:
    """The model's inputs: the CLAP embedding L2-normalized, then the own features
    (own_matrix), each column standardized by mean and std (missing values at the mean)."""
    emb = np.asarray(emb, dtype=np.float32)
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    X = np.hstack([e.astype(np.float64), own_matrix(own_raw)])
    if mean is not None:
        X = (X - mean) / std
    X[~np.isfinite(X)] = 0.0 if mean is not None else np.nan
    return X.astype(np.float32)


def standardizer(X) -> tuple:
    """(mean, std) per column of an unstandardized design matrix (NaN where missing; a
    column with no value at all gets mean 0, std 1)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    mean[~np.isfinite(mean)] = 0.0
    std[~(std > 1e-6)] = 1.0
    return mean.astype(np.float64), std.astype(np.float64)


def feature_rows(session, ids=None) -> list:
    """[(sample id, clap model, *own measurements)] for samples with a CLAP embedding (only
    these ids, when given), in id order."""
    sql = FEATURE_SQL
    rows = []
    if ids is None:
        return list(session.execute(text(sql + " ORDER BY s.id")))
    ids = sorted({int(i) for i in ids})
    for i in range(0, len(ids), 900):
        chunk = ", ".join(str(x) for x in ids[i:i + 900])
        rows += list(session.execute(text(sql + f" AND s.id IN ({chunk}) ORDER BY s.id")))
    return rows


def _index_arrays(path: Path):
    """(ids, embeddings) of the CLAP index at path, read only: its .npy copies when they're
    as new as the .npz, else the .npz; (None, None) without one."""
    pi, pe = path.with_suffix(".ids.npy"), path.with_suffix(".emb.npy")
    try:
        if min(pi.stat().st_mtime_ns, pe.stat().st_mtime_ns) >= path.stat().st_mtime_ns:
            return np.load(pi), np.load(pe, mmap_mode="r")
    except OSError:
        pass
    if path.exists():
        d = np.load(path)
        return d["ids"], d["embeddings"]
    return None, None


def embeddings(session, ids, index_path=None) -> np.ndarray:
    """(len(ids), 512) CLAP embeddings for these sample ids: from the CLAP index where it
    holds them, else the database's own copy."""
    from ..analysis.clap_features import CLAP_EMBEDDING_DIM, bytes_to_embedding
    from ..paths import clap_index_path
    ids = [int(i) for i in ids]
    out = np.zeros((len(ids), CLAP_EMBEDDING_DIM), dtype=np.float32)
    got = np.zeros(len(ids), dtype=bool)
    try:
        idx_ids, idx_emb = _index_arrays(Path(index_path) if index_path else clap_index_path())
    except Exception:              # an unreadable index: the database's copy
        idx_ids = idx_emb = None
    if idx_ids is not None and len(ids):
        pos = {int(s): r for r, s in enumerate(idx_ids)}
        rows = np.array([pos.get(s, -1) for s in ids])
        have = rows >= 0
        if have.any():
            out[have] = np.asarray(idx_emb[rows[have]], dtype=np.float32)
            got[have] = True
    rest = [i for i, ok in zip(ids, got) if not ok]
    if rest:
        at = {s: k for k, s in enumerate(ids)}
        for i in range(0, len(rest), 900):
            chunk = ", ".join(str(x) for x in rest[i:i + 900])
            for sid, blob in session.execute(text(
                    f"SELECT sample_id, clap_embedding FROM sample_features WHERE sample_id IN ({chunk}) "
                    "AND clap_embedding IS NOT NULL")):
                out[at[int(sid)]] = bytes_to_embedding(blob)
    return out


# --- the model -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SoundModel:
    categories: tuple           # the category outputs, in order (the built-in ones, "other")
    w_cat: np.ndarray           # (inputs, categories)
    b_cat: np.ndarray
    w_cls: np.ndarray           # (inputs, 2): one-shot, loop
    b_cls: np.ndarray
    mean: np.ndarray            # (inputs,): the design matrix's standardization
    std: np.ndarray
    t_cat: float                # temperatures (calibration)
    t_cls: float
    clap_model: str             # the CLAP model (and revision) the embeddings came from
    clap_revision: str
    features: tuple = FEATURE_NAMES
    version: str = ""
    path: str = ""
    trained_on: int = 0         # the samples with a category label it learned from (0: unknown)

    @property
    def marker(self) -> float:
        """The version as the "analysed" descriptor's value: six hex digits, exact in a 32-bit
        float (DuckDB's FLOAT)."""
        return float(int(self.version[:6], 16))

    def compatible(self, clap_model) -> bool:
        """Whether an embedding made by this CLAP model can be read by these weights: the same
        model, and for the real one, the same pinned revision."""
        from ..analysis.clap_features import CLAP_MODEL_ID, CLAP_REVISION
        if not clap_model or clap_model != self.clap_model:
            return False
        return clap_model != CLAP_MODEL_ID or self.clap_revision == CLAP_REVISION

    def probabilities(self, X) -> tuple:
        """(category probabilities (n, categories), loop probability (n,)) for a standardized
        design matrix."""
        return (softmax((X @ self.w_cat + self.b_cat) / self.t_cat),
                softmax((X @ self.w_cls + self.b_cls) / self.t_cls)[:, 1])

    def predict(self, emb, own_raw) -> tuple:
        return self.probabilities(design(emb, own_raw, self.mean, self.std))

    def save(self, path) -> "SoundModel":
        """Write the weights (no file names, no per-sample data: weights, biases, the
        features' means and standard deviations, and how many samples it learned from) and
        return the model with its version: a hash of everything it holds."""
        arrays = dict(
            format=np.array(FORMAT), categories=np.array(self.categories),
            features=np.array(self.features), w_cat=self.w_cat.astype(np.float32),
            b_cat=self.b_cat.astype(np.float32), w_cls=self.w_cls.astype(np.float32),
            b_cls=self.b_cls.astype(np.float32), mean=self.mean.astype(np.float32),
            std=self.std.astype(np.float32), t_cat=np.array(float(self.t_cat), dtype=np.float32),
            t_cls=np.array(float(self.t_cls), dtype=np.float32),
            clap_model=np.array(self.clap_model), clap_revision=np.array(self.clap_revision))
        if self.trained_on:
            arrays["trained_on"] = np.array(int(self.trained_on), dtype=np.int64)
        version = _version_of(arrays)
        arrays["version"] = np.array(version)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
        os.replace(tmp, path)
        return load(path)


def softmax(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _version_of(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.asarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:12]


def load(path) -> SoundModel:
    """The model in a weights file, checked: its layout, its features (this code's own) and
    its shapes."""
    path = Path(path)
    try:
        d = np.load(path, allow_pickle=False)
        got = {k: d[k] for k in d.files}
    except Exception as e:
        raise SoundModelError(f"the sound model {path} can't be read ({e})") from None
    if int(got.get("format", -1)) != FORMAT:
        raise SoundModelError(f"the sound model {path} is in another format "
                              f"({got.get('format')}, this code reads {FORMAT}): train it again")
    feats = tuple(str(x) for x in got["features"])
    if feats != FEATURE_NAMES:
        raise SoundModelError(f"the sound model {path} reads other features than this code makes: "
                              f"train it again")
    cats = tuple(str(x) for x in got["categories"])
    w_cat, w_cls = got["w_cat"].astype(np.float32), got["w_cls"].astype(np.float32)
    n_in = w_cat.shape[0]
    if (w_cat.shape[1] != len(cats) or got["b_cat"].shape != (len(cats),) or w_cls.shape != (n_in, 2)
            or got["mean"].shape != (n_in,) or got["std"].shape != (n_in,)
            or n_in != 512 + len(FEATURE_NAMES)):
        raise SoundModelError(f"the sound model {path} is malformed: its shapes don't agree")
    return SoundModel(categories=cats, w_cat=w_cat, b_cat=got["b_cat"].astype(np.float32),
                      w_cls=w_cls, b_cls=got["b_cls"].astype(np.float32),
                      mean=got["mean"].astype(np.float64), std=got["std"].astype(np.float64),
                      t_cat=float(got["t_cat"]), t_cls=float(got["t_cls"]),
                      clap_model=str(got["clap_model"]), clap_revision=str(got["clap_revision"]),
                      features=feats, version=str(got["version"]), path=str(path),
                      trained_on=int(got["trained_on"]) if "trained_on" in got else 0)


def model_path() -> Path | None:
    """The weights in use: $FOURIER_SOUND_MODEL ("off" or "none": none), else the ones
    trained on this library (<Fourier home>/sound_model.npz), else a package's own if it
    ships some (none does)."""
    env = os.environ.get(MODEL_ENV)
    if env is not None and env.strip():
        if env.strip().lower() in ("off", "none", "0"):
            return None
        p = Path(env).expanduser()
        if not p.is_file():
            raise SoundModelError(f"${MODEL_ENV}: no weights file at {p}")
        return p
    from ..paths import fourier_home
    home = fourier_home() / HOME_NAME
    if home.is_file():
        return home
    return DEFAULT_PATH if DEFAULT_PATH.is_file() else None


_LOADED: dict = {}


def model() -> SoundModel | None:
    """The sound model in use, or None (no weights: the provider is off)."""
    p = model_path()
    if p is None:
        return None
    key = (str(p), p.stat().st_mtime_ns)
    if key not in _LOADED:
        _LOADED.clear()
        _LOADED[key] = load(p)
    return _LOADED[key]


def forget() -> None:
    """Tests: load the weights again on the next model()."""
    _LOADED.clear()


# --- the provider's labels ----------------------------------------------------------------

def _markers(session) -> dict:
    """{marker value: samples} of the provider's "analysed" descriptors."""
    return {float(v): int(n) for v, n in session.execute(text(
        "SELECT value, count(*) FROM descriptors WHERE provider = :p AND name = :n GROUP BY value"),
        {"p": PROVIDER, "n": ANALYSED})}


def clear(session) -> int:
    """Remove every label and descriptor of the provider. Returns the samples it had."""
    n = sum(_markers(session).values())
    session.execute(text("DELETE FROM labels WHERE provider = :p"), {"p": PROVIDER})
    session.execute(text("DELETE FROM descriptors WHERE provider = :p"), {"p": PROVIDER})
    return n


def pending(session, mdl: SoundModel | None = None) -> int:
    """Samples the model can read (an embedding from its CLAP model) that it hasn't labelled
    yet; 0 without a model."""
    mdl = mdl or model()
    if mdl is None:
        return 0
    done = {int(s) for (s,) in session.execute(text(
        "SELECT sample_id FROM descriptors WHERE provider = :p AND name = :n AND value = :v"),
        {"p": PROVIDER, "n": ANALYSED, "v": mdl.marker})}
    return sum(1 for sid, cm in session.execute(text(
        "SELECT sample_id, clap_model FROM sample_features WHERE clap_embedding IS NOT NULL"))
        if int(sid) not in done and mdl.compatible(cm))


def label(session, mdl: SoundModel | None = None, force: bool = False, log=lambda m: None) -> dict:
    """Write the provider's labels for the samples the model hasn't labelled yet (every
    sample with force, or when the weights changed). Without a model, its old labels are
    removed. Returns {"labelled": n, "kept": n, "other_clap": n, "model": version or None}."""
    from sqlalchemy import insert

    from ..db.models import Descriptor, Label
    mdl = mdl if mdl is not None else model()
    if mdl is None:
        gone = clear(session)
        if gone:
            log(f"sound model: none in use; removed its labels of {gone:,} samples")
        return {"labelled": 0, "kept": 0, "other_clap": 0, "model": None}
    marks = _markers(session)
    if force or set(marks) - {mdl.marker}:
        clear(session)                      # another model's labels (or relabel all): all again
        marks = {}
    done = set()
    if marks:
        done = {int(s) for (s,) in session.execute(text(
            "SELECT sample_id FROM descriptors WHERE provider = :p AND name = :n"),
            {"p": PROVIDER, "n": ANALYSED})}
    todo, other = [], 0
    for sid, cm in session.execute(text(
            "SELECT sample_id, clap_model FROM sample_features WHERE clap_embedding IS NOT NULL "
            "ORDER BY sample_id")):
        if int(sid) in done:
            continue
        if mdl.compatible(cm):
            todo.append(int(sid))
        else:
            other += 1
    for i in range(0, len(todo), _CHUNK):
        rows = feature_rows(session, todo[i:i + _CHUNK])
        ids = [int(r[0]) for r in rows]
        own = [[np.nan if v is None else float(v) for v in r[2:]] for r in rows]
        p_cat, p_loop = mdl.predict(embeddings(session, ids), own)
        best = p_cat.argmax(axis=1)
        labels, descs = [], []
        for k, sid in enumerate(ids):
            pc, pl = float(p_cat[k, best[k]]), float(p_loop[k])
            shape, ps = (SHAPES[1], pl) if pl >= 0.5 else (SHAPES[0], 1.0 - pl)
            labels.append(dict(sample_id=sid, provider=PROVIDER, kind="category",
                               label=mdl.categories[best[k]], rank=0, confidence=round(pc, 4)))
            labels.append(dict(sample_id=sid, provider=PROVIDER, kind="class", label=shape, rank=0,
                               confidence=round(ps, 4)))
            descs += [dict(sample_id=sid, provider=PROVIDER, name=ANALYSED, value=mdl.marker),
                      dict(sample_id=sid, provider=PROVIDER, name=P_CATEGORY, value=round(pc, 4)),
                      dict(sample_id=sid, provider=PROVIDER, name=P_LOOP, value=round(pl, 4))]
        if labels:
            session.execute(insert(Label), labels)
            session.execute(insert(Descriptor), descs)
    if todo:
        log(f"sound model: labelled {len(todo):,} samples ({len(done):,} already done)")
    return {"labelled": len(todo), "kept": len(done), "other_clap": other, "model": mdl.version}


def current(session, mdl: SoundModel | None = None, ids=None) -> dict:
    """{sample id: (category, probability, loop probability)} the model in use has labelled
    (only these ids, when given); {} without one."""
    mdl = mdl if mdl is not None else model()
    if mdl is None:
        return {}
    sql = ("SELECT l.sample_id, l.label, l.confidence, d2.value FROM labels l "
           "JOIN descriptors d ON d.sample_id = l.sample_id AND d.provider = l.provider "
           "AND d.name = :a AND d.value = :v "
           "LEFT JOIN descriptors d2 ON d2.sample_id = l.sample_id AND d2.provider = l.provider "
           "AND d2.name = :pl "
           "WHERE l.provider = :p AND l.kind = 'category'")
    params = {"p": PROVIDER, "a": ANALYSED, "v": mdl.marker, "pl": P_LOOP}
    if ids is None:
        parts = [sql]
    else:
        ids = sorted({int(i) for i in ids})
        parts = [sql + f" AND l.sample_id IN ({', '.join(str(x) for x in ids[i:i + 900])})"
                 for i in range(0, len(ids), 900)]
    out = {}
    for q in parts:
        for sid, cat, p, pl in session.execute(text(q), params):
            out[int(sid)] = (str(cat), float(p or 0.0), None if pl is None else float(pl))
    return out


def describe_label(got) -> str:
    """'PADS 0.86, one-shot 0.97' for a sample's (category, probability, loop probability)."""
    cat, p, pl = got
    shape = "" if pl is None else (f", loop {pl:.2f}" if pl >= 0.5 else f", one-shot {1 - pl:.2f}")
    return f"{cat} {p:.2f}{shape}"
