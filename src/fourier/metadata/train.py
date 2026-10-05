"""Train the sound model (metadata/sound.py) on a library, reading its database only.

    fourier tools train                  # the library's database -> <home>/sound_model.npz
    python -m fourier.metadata.train --db <database> [--ratings ratings.json] [--index clap_index.npz]
        [--out sound_model.npz] [--report sound_model_report.md]

Without Sononym, the analysis's sound step trains one by itself (maybe_train): when there's
none yet and the library's own names label enough samples across enough packs
(curate_config.SOUND_TRAIN_MIN_LABELS, SOUND_TRAIN_MIN_PACKS), and again when the labelled
samples have grown (SOUND_RETRAIN_GROWTH). The weights stay in the Fourier home: they are
learned from the user's own library, whose licences are the user's, so Fourier ships none.

What it learns from (its labels): the pack makers' own folder and file names, read by the path
provider's rules (sound.training_label: a category word in the file's own name or folder), and
the user's own ratings (sound.rating_label: Keep in a category, Misfiled with a category to
move it to), which win where both speak. With --clap-prior the taxonomy's CLAP prompts are the
prior the weights are pulled toward (instead of zero). What it reads as features: the CLAP
embedding and Fourier's own measurements (sound.OWN_FEATURES). Nothing of another product's
analysis or tags is a label or a feature: tests/test_sound_model.py checks the code of every
function here but the yardstick's.

How: the labelled samples are split by pack (vendor/pack folders: no pack on both sides), a
share for the holdout; a multinomial logistic regression (L2, class-balanced weights,
L-BFGS stopped early on the holdout's loss) learns the category (the built-in categories and
"other") and a second one the shape (one-shot or loop); a temperature for each is fitted on
the holdout (calibration).

Writes the weights (--out; no file names and no per-sample data: weights, biases and the
features' means and standard deviations over the library) and a report (--report, kept in
the Fourier home): label counts, holdout accuracy per category against the labels,
the confusion matrix, coverage at confidence thresholds, and the yardstick: agreement with
Sononym's category and class where a sample has both, measured after the model is saved and
never fed back into it.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import sound as S

REPORT_NAME = "sound_model_report.md"   # beside the weights, in the Fourier home
SKIPPED_NAME = "sound_model.skipped.json"   # the last model discarded as unreliable: its count
RETRAIN_MIN_NEW = 500         # a retrain also needs this many more labelled samples
HOLDOUT_SHARE = 0.2           # of the labelled samples, by whole packs
L2 = 1e-3                     # on the weights (not the biases), per sample's mean loss
MAX_ITER = 400
PATIENCE = 25                 # L-BFGS iterations without a better holdout loss: stop
PRIOR_SCALE = 20.0            # --clap-prior: the logit a cosine of 1 with a category's prompts gives
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
_CHUNK = 20000


# --- reading the database ------------------------------------------------------------------

def use_config(config=None):
    """Resolve the config layers (fourier.toml, its preset and library overlay: the path
    words, the name rules) for this process, as `fourier` does, before the curation modules
    are first imported, into a scratch folder (nothing is written to the home). Returns what
    undoes it."""
    import shutil
    import tempfile

    from .. import layers, settings
    if os.environ.get(settings.ENV) or not (config or layers.find_config()):
        return lambda: None
    r = layers.resolve(config)
    if not r.values:
        return lambda: None
    tmp = tempfile.mkdtemp(prefix="fourier-train-")
    os.environ[settings.ENV] = str(layers.write(r, Path(tmp)))
    settings.reset()

    def undo():
        os.environ.pop(settings.ENV, None)
        settings.reset()
        shutil.rmtree(tmp, ignore_errors=True)
    return undo


def open_db(path):
    """A session on the database, opened read only (nothing created, migrated or written)."""
    from sqlalchemy.orm import Session

    from ..db.session import _open
    path = Path(path).expanduser()
    if not path.exists():
        raise SystemExit(f"no database at {path}")
    return Session(_open(path, True))


def _missing(session) -> set:
    """Samples a walk marked missing (or moved): no label, no feature."""
    from sqlalchemy import text
    try:
        return {int(s) for (s,) in session.execute(text("SELECT sample_id FROM missing_files"))}
    except Exception:
        return set()


def library_rows(session):
    """(ids, clap model, own measurements (n, k), rel paths, durations, paths) of the samples
    with a CLAP embedding from the library's main CLAP model."""
    from sqlalchemy import text

    from ..places import library_rel
    rows = S.feature_rows(session)
    gone = _missing(session)
    rows = [r for r in rows if int(r[0]) not in gone]
    if not rows:
        raise SystemExit("no sample has a CLAP embedding yet: `fourier tools analyze` first")
    clap = Counter(r[1] for r in rows if r[1]).most_common(1)
    clap = clap[0][0] if clap else None
    rows = [r for r in rows if r[1] == clap]
    ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
    own = np.array([[np.nan if v is None else float(v) for v in r[2:]] for r in rows], dtype=np.float64)
    where = {int(i): (rel, path, dur) for i, rel, path, dur in session.execute(text(
        "SELECT id, rel_path, path, duration_s FROM samples"))}
    rels = [where[int(i)][0] or library_rel(where[int(i)][1] or "") for i in ids]
    paths = [where[int(i)][1] for i in ids]
    durs = [where[int(i)][2] for i in ids]
    return ids, clap, own, rels, durs, paths


def pack_key(rel: str) -> str:
    """A sample's pack: its first two folders (vendor/pack), or the one it has."""
    folders = [p for p in (rel or "").split("/") if p][:-1]
    return "/".join(folders[:2]) or "(top)"


def split_packs(keys, share=HOLDOUT_SHARE) -> np.ndarray:
    """A holdout mask over samples whose packs are `keys`: whole packs, taken in an order
    their names hash to, until the holdout holds `share` of the samples. With two packs or
    more, both sides get at least one."""
    keys = list(keys)
    count = Counter(keys)
    order = sorted(count, key=lambda k: hashlib.sha1(k.encode()).hexdigest())
    total, held, out = len(keys), 0, set()
    for k in order:
        if len(out) + 1 >= len(order):
            break
        if held + count[k] <= share * total or not out:
            out.add(k)
            held += count[k]
    return np.array([k in out for k in keys], dtype=bool)


def labels(ids, rels, durs, paths, ratings_path=None) -> tuple:
    """(category labels, shape labels, sources): for each sample the user's rating when there
    is one, else the pack maker's names; None where neither says."""
    from ..packs.ratings import load_store
    try:
        rated = load_store(ratings_path)["ratings"] if ratings_path or ratings_path is None else {}
    except (OSError, ValueError):
        rated = {}
    cats, shapes, src = [], [], []
    for sid, rel, dur, path in zip(ids, rels, durs, paths):
        c = s = None
        how = None
        r = rated.get(path)
        if r:
            c, s = S.rating_label(r)
            how = "rating" if c else None
        if c is None or s is None:
            pc, ps = S.training_label(rel, dur)
            if c is None and pc is not None and pc != S.rating_excludes(r):  # type: ignore[arg-type]
                c, how = pc, "names"
            if s is None:
                s = ps
        cats.append(c)
        shapes.append(s)
        src.append(how)
    return cats, shapes, src


# --- the model -----------------------------------------------------------------------------

def _loss_grad(theta, X, Y, sw, l2, W0, d, K):
    W = theta[:d * K].reshape(d, K)
    b = theta[d * K:]
    Z = (X @ W.astype(np.float32)).astype(np.float64) + b
    P = S.softmax(Z)
    swn = sw / sw.sum()
    loss = -float(np.sum(swn * np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, None))))
    R = P
    R[np.arange(len(Y)), Y] -= 1.0
    R *= swn[:, None]
    gW = (X.T @ R.astype(np.float32)).astype(np.float64) + l2 * (W - W0)
    gb = R.sum(axis=0)
    loss += 0.5 * l2 * float(np.sum((W - W0) ** 2))
    return loss, np.concatenate([gW.ravel(), gb])


def _nll(Z, Y, w=None) -> float:
    P = S.softmax(Z)
    lp = -np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, None))
    return float(np.sum(lp * w) / np.sum(w)) if w is not None else float(np.mean(lp))


def train_softmax(X, y, K, Xh, yh, l2=L2, W0=None, b0=None, max_iter=MAX_ITER, log=print):
    """(W, b, iterations) of a multinomial logistic regression with class-balanced weights,
    an L2 pull toward W0 (zero without a prior), fitted by L-BFGS and stopped at its best
    holdout loss (balanced the same way) when the holdout has samples."""
    from scipy.optimize import minimize
    n, d = X.shape
    counts = np.bincount(y, minlength=K).astype(np.float64)
    present = counts > 0
    cw = np.where(present, n / (present.sum() * np.maximum(counts, 1.0)), 0.0)
    sw = cw[y]
    W0 = np.zeros((d, K)) if W0 is None else W0
    b0 = np.zeros(K) if b0 is None else b0
    theta0 = np.concatenate([W0.ravel(), b0])
    hw = cw[yh] if len(yh) else None
    use_holdout = hw is not None and hw.sum() > 0
    best = {"loss": np.inf, "theta": theta0.copy(), "it": 0, "since": 0, "it_now": 0}

    def holdout_loss(theta):
        W, b = theta[:d * K].reshape(d, K), theta[d * K:]
        return _nll((Xh @ W.astype(np.float32)).astype(np.float64) + b, yh, hw)

    def cb(intermediate_result):
        best["it_now"] += 1
        if not use_holdout:
            return
        th = intermediate_result.x
        hl = holdout_loss(th)
        if hl < best["loss"] - 1e-6:
            best.update(loss=hl, theta=th.copy(), it=best["it_now"], since=0)
        else:
            best["since"] += 1
            if best["since"] >= PATIENCE:
                raise StopIteration

    res = minimize(_loss_grad, theta0, args=(X, y, sw, l2, W0, d, K), jac=True, method="L-BFGS-B",
                   callback=cb, options={"maxiter": max_iter})
    theta = best["theta"] if use_holdout and best["it"] else res.x
    W, b = theta[:d * K].reshape(d, K), theta[d * K:]
    return W, b, best["it"] if use_holdout else best["it_now"]


def fit_temperature(Z, y) -> float:
    """The temperature that minimizes the holdout's negative log-likelihood (1 without one)."""
    if not len(y):
        return 1.0
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda lt: _nll(Z / np.exp(lt), y), bounds=(np.log(0.05), np.log(20.0)),
                        method="bounded")
    return float(np.exp(r.x))


def clap_prior(categories, mean, std) -> tuple:
    """(W0, b0): each category's logit PRIOR_SCALE x the cosine of the (unstandardized)
    embedding with the mean of its taxonomy prompts' CLAP text embeddings, written over the
    standardized inputs; zero for a category without prompts and for the own features."""
    from .. import taxonomy
    from ..analysis.clap_features import embed_text
    doc = taxonomy.load()
    d, K = len(mean), len(categories)
    W0, b0 = np.zeros((d, K)), np.zeros(K)
    for k, cat in enumerate(categories):
        prompts = [p for p in ((doc["categories"].get(cat) or {}).get("prompts") or ()) if isinstance(p, str)]
        if not prompts:
            continue
        a = np.mean([embed_text(p) for p in prompts], axis=0).astype(np.float64)
        a /= np.linalg.norm(a) + 1e-9
        W0[:512, k] = PRIOR_SCALE * std[:512] * a
        b0[k] = PRIOR_SCALE * float(mean[:512] @ a)
    return W0, b0


# --- predicting over the library -----------------------------------------------------------

def predict_all(mdl, session, ids, own, index=None):
    """(category probabilities (n, K), loop probabilities (n,)) for these samples, in chunks."""
    pc, pl = [], []
    for i in range(0, len(ids), _CHUNK):
        emb = S.embeddings(session, ids[i:i + _CHUNK], index)
        a, b = mdl.predict(emb, own[i:i + _CHUNK])
        pc.append(a)
        pl.append(b)
    if not pc:
        return np.zeros((0, len(mdl.categories))), np.zeros(0)
    return np.vstack(pc), np.concatenate(pl)


# --- the report ----------------------------------------------------------------------------

def _pct(a, b) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "-"


def _table(head, rows) -> list:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


def report_training(cats, y_cat, hold, P_h, y_h, shape_y, shape_hold, pl_h, sy_h, src, P_lib,
                    pl_lib, info) -> list:
    """The report's sections on the labels and the holdout (everything but the yardstick)."""
    out = ["# Sound model report", "",
           f"Model {info['version']}, trained on {info['clap']} embeddings; "
           f"{info['n']:,} samples with an embedding, {info['labelled']:,} with a category label, "
           f"{info['shaped']:,} with a shape label; {info['packs']:,} packs ({info['hold_packs']:,} in "
           f"the holdout). Category model: {info['it_cat']} iterations, temperature {info['t_cat']:.2f}; "
           f"shape model: {info['it_cls']} iterations, temperature {info['t_cls']:.2f}. "
           f"{info['seconds']:.0f} s.", "", "## Labels", ""]
    by = defaultdict(Counter)
    for c, h, s in zip(y_cat, hold, src):
        if c is not None:
            by[c]["holdout" if h else "train"] += 1
            by[c][s] += 1
    out += _table(["category", "train", "holdout", "from names", "from ratings"],
                  [(c, by[c]["train"], by[c]["holdout"], by[c]["names"], by[c]["rating"])
                   for c in cats if by[c]])
    sc = Counter((s, "holdout" if h else "train") for s, h in zip(shape_y, shape_hold) if s)
    out += ["", f"Shapes: one-shot {sc[('class.oneshot', 'train')]:,} train / "
                f"{sc[('class.oneshot', 'holdout')]:,} holdout; loop {sc[('class.loop', 'train')]:,} train / "
                f"{sc[('class.loop', 'holdout')]:,} holdout.", "", "## Holdout: category", ""]
    if not len(y_h):
        out.append("No holdout samples with a category label.")
    else:
        pred = P_h.argmax(axis=1)
        conf = np.zeros((len(cats), len(cats)), dtype=int)
        for t, p in zip(y_h, pred):
            conf[t, p] += 1
        rows, recalls = [], []
        for k, c in enumerate(cats):
            n, hit, said = conf[k].sum(), conf[k, k], conf[:, k].sum()
            if n or said:
                rows.append((c, n, _pct(hit, n), _pct(hit, said)))
            if n:
                recalls.append(hit / n)
        out += [f"Accuracy {_pct(np.trace(conf), conf.sum())} of {conf.sum():,}; balanced "
                f"(mean recall over the categories it has) {100 * np.mean(recalls):.1f}%.", ""]
        out += _table(["category", "holdout", "recall", "precision"], rows)
        used = [k for k in range(len(cats)) if conf[k].sum() or conf[:, k].sum()]
        out += ["", "Confusion (rows: the label, columns: the model's answer):", ""]
        out += _table(["label \\ model"] + [cats[k] for k in used],
                      [[cats[i]] + [conf[i, j] or "" for j in used] for i in used])
        out += ["", "Coverage (holdout: share at or above the threshold, accuracy there; library: "
                    "share of every sample with an embedding):", ""]
        pmax = P_h.max(axis=1)
        lib = P_lib.max(axis=1) if len(P_lib) else np.zeros(0)
        out += _table(["threshold", "holdout covered", "holdout accuracy", "library covered"],
                      [(t, _pct((pmax >= t).sum(), len(pmax)),
                        _pct(((pmax >= t) & (pred == y_h)).sum(), (pmax >= t).sum()),
                        _pct((lib >= t).sum(), len(lib))) for t in THRESHOLDS])
    out += ["", "## Holdout: one-shot or loop", ""]
    if not len(sy_h):
        out.append("No holdout samples with a shape label.")
    else:
        pr = (pl_h >= 0.5).astype(int)
        conf = np.zeros((2, 2), dtype=int)
        for t, p in zip(sy_h, pr):
            conf[t, p] += 1
        out += [f"Accuracy {_pct(np.trace(conf), conf.sum())} of {conf.sum():,}.", ""]
        out += _table(["label \\ model", "one-shot", "loop"],
                      [("one-shot", conf[0, 0], conf[0, 1]), ("loop", conf[1, 0], conf[1, 1])])
        pc = np.maximum(pl_h, 1 - pl_h)
        lib = np.maximum(pl_lib, 1 - pl_lib) if len(pl_lib) else np.zeros(0)
        out += ["", ""] + _table(["threshold", "holdout covered", "holdout accuracy", "library covered"],
                                 [(t, _pct((pc >= t).sum(), len(pc)),
                                   _pct(((pc >= t) & (pr == sy_h)).sum(), (pc >= t).sum()),
                                   _pct((lib >= t).sum(), len(lib))) for t in THRESHOLDS])
    return out


def yardstick(session, mdl, ids, P_lib, pl_lib) -> list:
    """The report's yardstick: where the saved model and Sononym's analysis both speak, how
    often they agree (category, by Sononym's category, and class). Read after the model is
    saved; nothing here goes back into it."""
    import json

    from sqlalchemy import text

    from ..metadata.vocab import mapping
    out = ["", "## Yardstick: Sononym (measured only, never trained on)", ""]
    try:
        son = {int(s): (json.loads(c or "[]") if isinstance(c, str) else (c or []),
                        json.loads(k or "[]") if isinstance(k, str) else (k or []))
               for s, c, k in session.execute(text("SELECT sample_id, classes, categories FROM sononym_meta"))}
    except Exception:
        son = {}
    if not son:
        return out + ["No Sononym analysis in this database."]
    m = mapping("sononym")
    cats = list(mdl.categories)
    pos = {int(i): r for r, i in enumerate(ids)}
    cls_n = cls_same = 0
    per = defaultdict(Counter)
    at = {t: [0, 0] for t in THRESHOLDS}
    for sid, (classes, categories) in son.items():
        r = pos.get(sid)
        if r is None:
            continue
        shape = {m.get(("class", c)) for c in classes} & set(S.SHAPES)
        if len(shape) == 1:
            cls_n += 1
            cls_same += (pl_lib[r] >= 0.5) == ("class.loop" in shape)
        canon = {m.get(("category", c)) for c in categories} - {None}
        want = S.category_for(S.categories_of(canon), "class.loop" in shape)
        if want is None:
            continue
        got = cats[int(P_lib[r].argmax())]
        per[want]["n"] += 1
        per[want]["same" if got == want else got] += 1
        p = float(P_lib[r].max())
        for t in THRESHOLDS:
            if p >= t:
                at[t][0] += 1
                at[t][1] += got == want
    n = sum(c["n"] for c in per.values())
    same = sum(c["same"] for c in per.values())
    out += [f"Category: agrees on {_pct(same, n)} of {n:,} samples Sononym files under a built-in "
            f"category; class (one-shot or loop): agrees on {_pct(cls_same, cls_n)} of {cls_n:,}.", ""]
    rows = []
    for c in cats:
        if per[c]["n"]:
            other = Counter({k: v for k, v in per[c].items() if k not in ("n", "same")}).most_common(2)
            rows.append((c, per[c]["n"], _pct(per[c]["same"], per[c]["n"]),
                         ", ".join(f"{k} {v}" for k, v in other) or "-"))
    out += _table(["Sononym's category", "samples", "sound model agrees", "else, most often"], rows)
    out += ["", ""] + _table(["sound model confidence", "samples", "agree"],
                             [(f">= {t}", at[t][0], _pct(at[t][1], at[t][0])) for t in THRESHOLDS])
    return out


# --- training -------------------------------------------------------------------------------

def default_out() -> Path:
    """Where trained weights go and are looked for first: <Fourier home>/sound_model.npz."""
    from ..paths import fourier_home
    return fourier_home() / S.HOME_NAME


def label_counts(session) -> tuple:
    """(samples with a category label from their own names or a rating, packs they're in, all
    samples with an embedding): what the training gate reads. (0, 0, 0) without embeddings."""
    try:
        ids, _clap, _own, rels, durs, paths = library_rows(session)
    except SystemExit:
        return 0, 0, 0
    y_cat, _ys, _src = labels(ids, rels, durs, paths, None)
    keys = {pack_key(r) for r, c in zip(rels, y_cat) if c is not None}
    return sum(1 for c in y_cat if c is not None), len(keys), len(ids)


def train(session, out=None, report=None, ratings=None, index=None, clap_prior_=False, l2=L2,
          holdout=HOLDOUT_SHARE, log=print) -> dict:
    """Train on the library in `session` (read only), write the weights to `out` (default:
    the Fourier home) and the report to `report` (default: beside them). Returns a summary:
    the model, its version, the counts, the holdout's accuracy overall and at SOUND_MODEL_MIN
    (and how much of it that covers), and the shape's accuracy."""
    t0 = time.monotonic()
    from ..analysis.clap_features import CLAP_MODEL_ID, CLAP_REVISION
    from ..packs.curate import SOUND_MODEL_MIN
    out = Path(out).expanduser() if out else default_out()
    report = Path(report).expanduser() if report else out.with_name(REPORT_NAME)
    ids, clap, own, rels, durs, paths = library_rows(session)
    log(f"{len(ids):,} samples with {clap} embeddings")
    y_cat, y_shape, src = labels(ids, rels, durs, paths, ratings)
    cats = list(S.model_categories())
    ck = {c: k for k, c in enumerate(cats)}
    keys = [pack_key(r) for r in rels]
    hold = split_packs(keys, holdout)
    has_cat = np.array([c is not None for c in y_cat])
    has_shape = np.array([s is not None for s in y_shape])
    use = has_cat | has_shape
    if not has_cat.any():
        raise SystemExit("no sample has a category label from its names or a rating")
    log(f"labels: {has_cat.sum():,} categories, {has_shape.sum():,} shapes; holdout: "
        f"{len({k for k, h in zip(keys, hold) if h}):,} of {len(set(keys)):,} packs")
    emb = S.embeddings(session, ids[use], index)
    X_raw = S.design(emb, own[use])
    train_rows = ~hold[use]
    mean, std = S.standardizer(X_raw[train_rows])
    X = ((X_raw - mean) / std).astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    del X_raw, emb
    sub_cat, sub_shape = has_cat[use], has_shape[use]
    yc = np.array([ck[c] if c is not None else -1 for c in np.array(y_cat, dtype=object)[use]])
    ys = np.array([{"class.oneshot": 0, "class.loop": 1}.get(s, -1)
                   for s in np.array(y_shape, dtype=object)[use]])
    tr, ho = train_rows, ~train_rows
    if not (ho & sub_cat).any() or not (tr & sub_cat).any():
        raise SystemExit("the labelled samples sit in too few packs to hold some out")
    W0 = b0 = None
    if clap_prior_:
        W0, b0 = clap_prior(cats, mean, std)
    Wc, bc, it_c = train_softmax(X[tr & sub_cat], yc[tr & sub_cat], len(cats), X[ho & sub_cat],
                                 yc[ho & sub_cat], l2, W0, b0, log=log)
    t_cat = fit_temperature((X[ho & sub_cat] @ Wc.astype(np.float32)).astype(np.float64) + bc,
                            yc[ho & sub_cat])
    if len(set(ys[tr & sub_shape])) < 2:
        raise SystemExit("the shape labels hold only one shape: nothing to learn")
    Ws, bs, it_s = train_softmax(X[tr & sub_shape], ys[tr & sub_shape], 2, X[ho & sub_shape],
                                 ys[ho & sub_shape], l2, log=log)
    t_cls = fit_temperature((X[ho & sub_shape] @ Ws.astype(np.float32)).astype(np.float64) + bs,
                            ys[ho & sub_shape])
    mdl = S.SoundModel(categories=tuple(cats), w_cat=Wc, b_cat=bc, w_cls=Ws, b_cls=bs, mean=mean,
                       std=std, t_cat=t_cat, t_cls=t_cls, clap_model=clap,
                       clap_revision=CLAP_REVISION if clap == CLAP_MODEL_ID else "",
                       trained_on=int(has_cat.sum()))
    mdl = mdl.save(out)
    log(f"wrote {out} (model {mdl.version})")
    P_h, _ = mdl.probabilities(X[ho & sub_cat])
    _, pl_h = mdl.probabilities(X[ho & sub_shape])
    del X
    P_lib, pl_lib = predict_all(mdl, session, ids, own, index)
    info = dict(version=mdl.version, clap=clap, n=len(ids), labelled=int(has_cat.sum()),
                shaped=int(has_shape.sum()), packs=len(set(keys)),
                hold_packs=len({k for k, h in zip(keys, hold) if h}), it_cat=it_c, it_cls=it_s,
                t_cat=t_cat, t_cls=t_cls, seconds=time.monotonic() - t0)
    y_h = yc[ho & sub_cat]
    lines = report_training(cats, y_cat, hold, P_h, y_h, y_shape, hold, pl_h,
                            ys[ho & sub_shape], src, P_lib, pl_lib, info)
    lines += yardstick(session, mdl, ids, P_lib, pl_lib)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    log(f"wrote {report} ({time.monotonic() - t0:.0f} s)")
    conf, right = P_h.max(axis=1), P_h.argmax(axis=1) == y_h
    sure = conf >= SOUND_MODEL_MIN
    sy = ys[ho & sub_shape]
    return dict(model=mdl, version=mdl.version, out=out, report=report, labelled=info["labelled"],
                packs=len({k for k, c in zip(keys, y_cat) if c is not None}),
                hold_packs=len({k for k, h, c in zip(keys, hold, y_cat) if h and c is not None}),
                held=int(len(y_h)),
                accuracy=float(right.mean()) if len(y_h) else 0.0,
                at_min=SOUND_MODEL_MIN, covered=float(sure.mean()) if len(y_h) else 0.0,
                accuracy_at_min=float(right[sure].mean()) if sure.any() else 0.0,
                shape_accuracy=float(((pl_h >= 0.5) == (sy == 1)).mean()) if len(sy) else 0.0,
                seconds=info["seconds"])


def summary_lines(got: dict) -> list:
    """What `fourier tools train` and a training build say about a model they trained."""
    c = got["covered"]
    if c >= 0.005:
        sure = (f"{100 * got['accuracy_at_min']:.0f}% when at least {got['at_min']:.2f} sure "
                f"({100 * c:.0f}% of them)")
    elif c > 0:
        sure = f"at least {got['at_min']:.2f} sure of under 1% of them"
    else:
        sure = f"never {got['at_min']:.2f} sure"
    out = [f"Sound model {got['version']}: trained on {got['labelled']:,} samples your library's "
           f"own names label, across {got['packs']:,} packs.",
           f"On {got['hold_packs']:,} packs it didn't learn from, its category agreed with the "
           f"names {100 * got['accuracy']:.0f}% of the time, {sure}; one-shot or loop "
           f"{100 * got['shape_accuracy']:.0f}%."]
    if got.get("kept", True):
        out.append(f"Weights: {got['out']} (they stay on this machine); report: {got['report']}.")
    else:
        out.append(f"Not kept: {got['why']}. The CLAP fallback places what no name rule "
                   f"recognizes; a build tries again once your library has grown. Report: "
                   f"{got['report']}.")
    return out


def reliable(got: dict) -> tuple:
    """(keep it, why not): the holdout's calls at SOUND_MODEL_MIN or more agree with the names
    often enough (SOUND_KEEP_MIN_ACCURACY), there are enough of them (SOUND_KEEP_MIN_COVERAGE),
    and enough samples were held out to tell (SOUND_KEEP_MIN_HELD)."""
    from ..packs import curate_config as C
    if got["held"] < C.SOUND_KEEP_MIN_HELD:
        return False, (f"only {got['held']:,} samples in packs it didn't learn from, too few to "
                       f"tell how reliable it is ({C.SOUND_KEEP_MIN_HELD:,} needed)")
    if got["covered"] < C.SOUND_KEEP_MIN_COVERAGE:
        share = f"{100 * got['covered']:.0f}%" if got["covered"] >= 0.005 else "under 1%"
        return False, (f"it was at least {got['at_min']:.2f} sure of only {share} of the samples "
                       f"in packs it didn't learn from ({100 * C.SOUND_KEEP_MIN_COVERAGE:.0f}% needed)")
    if got["accuracy_at_min"] < C.SOUND_KEEP_MIN_ACCURACY:
        return False, (f"its confident calls agreed with the names only "
                       f"{100 * got['accuracy_at_min']:.0f}% of the time "
                       f"({100 * C.SOUND_KEEP_MIN_ACCURACY:.0f}% needed)")
    return True, ""


def train_and_keep(session, out=None, keep_anyway=False, log=lambda m: None, **kw) -> dict:
    """train() into a trial file beside `out` (default: the Fourier home's weights), then keep
    it (replacing the weights there) when it's reliable (or `keep_anyway`); otherwise discard
    it, keep whatever weights were there, and record its count (SKIPPED_NAME) so builds try
    again only once the library has grown. The report says which, and why."""
    import json
    out = Path(out).expanduser() if out else default_out()
    trial = out.with_name(f".{out.stem}.trial.npz")
    report = kw.pop("report", None)
    report = Path(report).expanduser() if report else out.with_name(REPORT_NAME)
    try:
        got = train(session, trial, report, log=log, **kw)
    except BaseException:
        trial.unlink(missing_ok=True)
        raise
    ok, why = reliable(got)
    skipped = out.with_name(SKIPPED_NAME)
    if ok or keep_anyway:
        os.replace(trial, out)
        skipped.unlink(missing_ok=True)
        got.update(out=out, model=S.load(out), kept=True, why=why)
        verdict = "Kept." if ok else f"Kept on request, though {why}."
    else:
        trial.unlink(missing_ok=True)
        skipped.write_text(json.dumps({"labelled": got["labelled"], "version": got["version"]}) + "\n")
        got.update(kept=False, why=why)
        verdict = f"Not kept: {why}."
    with report.open("a") as f:
        f.write(f"\n## Verdict\n\n{verdict}\n")
    return got


def maybe_train(session, log=print) -> dict | None:
    """Train when a build should (the analysis's sound step): without Sononym, with training
    on (SOUND_TRAIN) and no $FOURIER_SOUND_MODEL, when the Fourier home has no weights yet
    and the library's own names label SOUND_TRAIN_MIN_LABELS samples across
    SOUND_TRAIN_MIN_PACKS packs, or when the labelled samples have grown by
    SOUND_RETRAIN_GROWTH (and by RETRAIN_MIN_NEW) since the home's weights were trained, or
    since a model was last discarded as unreliable. It keeps a model only when it's reliable
    (train_and_keep). Returns the summary, or None (and says why when it matters)."""
    from ..packs import curate_config as C
    from .providers import active
    if os.environ.get(S.MODEL_ENV, "").strip() or not C.SOUND_TRAIN:
        return None
    if not active(session).fallback:            # Sononym classifies: the model routes nothing
        return None
    import json
    have = default_out()
    trained_on = None
    if have.is_file():
        try:
            trained_on = S.load(have).trained_on
        except S.SoundModelError:
            trained_on = None                    # unreadable: train a new one
        if trained_on is not None and not trained_on:
            return None                          # made by hand (no count): left alone
    try:                                         # the last model discarded as unreliable
        tried = int(json.loads(have.with_name(SKIPPED_NAME).read_text())["labelled"])
    except (OSError, ValueError, KeyError, TypeError):
        tried = 0
    base = max(trained_on or 0, tried)
    n, packs, n_emb = label_counts(session)
    if not n_emb:
        return None
    if base:
        if n < max(base * (1 + C.SOUND_RETRAIN_GROWTH), base + RETRAIN_MIN_NEW):
            return None
        log(f"Training the sound model again: {n:,} samples your library's names label now, "
            f"{base:,} when it was last trained.")
    elif n < C.SOUND_TRAIN_MIN_LABELS or packs < C.SOUND_TRAIN_MIN_PACKS:
        log(f"No sound model yet: your library's own names label {n:,} samples across {packs:,} "
            f"packs, and training needs {C.SOUND_TRAIN_MIN_LABELS:,} across "
            f"{C.SOUND_TRAIN_MIN_PACKS}. The CLAP fallback places what no name rule recognizes.")
        return None
    else:
        log(f"Training a sound model on your library ({n:,} samples its own names label, "
            f"{packs:,} packs): about a minute, once.")
    try:
        got = train_and_keep(session)
    except SystemExit as e:
        log(f"No sound model: {e}")
        return None
    for line in summary_lines(got):
        log(line)
    return got


# --- main ----------------------------------------------------------------------------------

def main(argv=None, log=print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", help="the library database (default: the Fourier home's)")
    ap.add_argument("--config", help="the fourier.toml whose words and name rules label the samples "
                                     "(default: the one `fourier` finds)")
    ap.add_argument("--ratings", help="ratings.json (default: the Fourier home's; 'none' for none)")
    ap.add_argument("--index", help="the CLAP index (default: the Fourier home's clap_index.npz); "
                                    "embeddings it lacks come from the database")
    ap.add_argument("--out", help="the weights file to write (default: <home>/sound_model.npz)")
    ap.add_argument("--report", help="the report to write (default: beside the weights)")
    ap.add_argument("--clap-prior", action="store_true",
                    help="pull the weights toward the taxonomy's CLAP prompts instead of zero")
    ap.add_argument("--l2", type=float, default=L2, help="L2 strength on the weights")
    ap.add_argument("--holdout", type=float, default=HOLDOUT_SHARE, help="share of samples held out, by pack")
    a = ap.parse_args(argv)
    undo = use_config(a.config)
    try:
        from ..db.session import default_db_path
        session = open_db(a.db or default_db_path())
        try:
            ratings = None if a.ratings is None else ("" if a.ratings.lower() == "none" else a.ratings)
            train(session, a.out, a.report, ratings, a.index, a.clap_prior, a.l2, a.holdout, log=log)
        finally:
            session.close()
            session.get_bind().dispose()        # (a DuckDB file then opens read-write again)
        return 0
    finally:
        undo()


if __name__ == "__main__":
    sys.exit(main())
