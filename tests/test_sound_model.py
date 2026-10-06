"""The sound model (fourier/metadata/sound.py, fourier/metadata/train.py): what it learns
from (the pack makers' names, the user's ratings; never another product's analysis or tags),
the pack-split holdout, the provider's labels, and routing: without Sononym it places what
the names leave open; with Sononym it routes nothing (and the synthetic golden is unchanged:
synthetic_build.py --check --sound)."""
from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from sqlalchemy import text

from fourier.db import session as S
from fourier.db.models import Sample, SampleFeatures, SononymMeta
from fourier.demo import fake_embed_text, unit
from fourier.metadata import rows as R
from fourier.metadata import sound as SM
from fourier.metadata import train as T
from fourier.packs import curate_config as CC

ROOT = Path(__file__).resolve().parents[1]
LIB = "/l/SampleLibrary"


# kind: (folder, file word, stand-in CLAP concept, harmonic share, length in s)
KINDS = {
    "kick": ("Kicks", "Kick", "kick", 0.6, 0.4), "snare": ("Snares", "Snare", "snare", 0.3, 0.3),
    "hat": ("Hats", "Hat", "hat", 0.1, 0.2), "pad": ("Pads", "Pad", "pad", 0.9, 4.0),
    "bass": ("Bass", "Bass", "bass", 0.85, 1.0), "break": ("Drum Loops", "Break", "break", 0.3, 8.0),
}
VENDORS = ("Acme", "Northwind", "Vendor C", "Vendor D")


def _emb(kind, key, concept=0.3):
    """A stand-in CLAP embedding that sounds like its kind (a timbre all of the kind's files
    share) but whose words CLAP hears only faintly (a little of the kind's text concept), so
    the CLAP fallback's prompt anchors can't place it and a model trained on named files can."""
    v = concept * unit("concept", KINDS[kind][2]) + unit("timbre", kind) + 0.35 * unit("file", key)
    return (v / np.linalg.norm(v)).astype("float32")


def _plan(dump=6, weak=()):
    """[(rel path, kind)]: four vendors' named packs, a flat dump of numbered files, and
    `weak`: (rel, kind) files only a pack's name describes."""
    out = []
    for v in VENDORS:
        for kind, (folder, word, *_r) in KINDS.items():
            for i in range(6):
                out.append((f"{v}/{word} Pack/{folder}/{word} {i + 1:02d}.wav", kind))
    n = 0
    for kind in KINDS:
        for _ in range(dump):
            n += 1
            out.append((f"Dump/{n:03d}.wav", kind))
    return out + list(weak)


def _fill(db, plan, sononym=False, ableton=False, clap_model="stand-in"):
    """A database of these files: the file, Fourier's own measurements and a stand-in CLAP
    embedding each; with `sononym` / `ableton`, rows of another product's analysis and tags
    that say something else entirely (random categories and scores)."""
    rng = np.random.default_rng(7)
    S._engine = S._SessionLocal = None
    S.init_db(db)
    R.forget_current()
    son_cats = ["Perc Kicks", "Tone Pads & Textures", "Perc Snares", "XFX Sweeps & Lasers"]
    with S.session_scope() as s:
        for i, (rel, kind) in enumerate(plan, 1):
            _f, _w, _c, har, dur = KINDS[kind]
            s.add(Sample(id=i, path=f"{LIB}/{rel}", rel_path=rel, filename=rel.rsplit("/", 1)[1],
                         duration_s=dur, ableton_tags=(["Kick", "Pad"][i % 2:i % 2 + 1] if ableton else None)))
            s.add(SampleFeatures(
                sample_id=i, clap_embedding=_emb(kind, rel).tobytes(), clap_model=clap_model,
                harmonic_percussive_ratio=har, onset_rate_hz=4.0 if kind == "break" else 1.0,
                n_events=16 if kind == "break" else 1,
                spectral_centroid_mean=500 + 3000 * (1 - har), spectral_flatness_mean=1 - har,
                chroma_concentration=1 + 2 * har,
                **({"loop_confidence": float(rng.random()), "is_pitched": int(rng.integers(2)),
                    "bpm_reliable": 1, "transient_score": float(rng.random()),
                    "timbral_norm": [0.1, 0.2, 0.3]} if sononym else {})))
            if sononym:
                s.add(SononymMeta(sample_id=i, classes=["OneShot"], categories=[son_cats[int(rng.integers(4))]],
                                  harmonicity=float(rng.random()), brightness=float(rng.random())))
    return db


@pytest.fixture(autouse=True)
def _reset(monkeypatch, tmp_path):
    monkeypatch.setenv("FOURIER_CONFIG", "none")
    monkeypatch.setenv("FOURIER_RATINGS", str(tmp_path / "no-ratings.json"))
    monkeypatch.setattr(SM, "DEFAULT_PATH", tmp_path / "not-shipped.npz")
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "fourier-home"))   # no weights trained here yet
    monkeypatch.setattr(CC, "SOUND_KEEP_MIN_HELD", 0)    # these test libraries hold few packs out
    import fourier.analysis.clap_features as CF
    monkeypatch.setattr(CF, "embed_text", fake_embed_text)
    SM.forget()
    yield
    SM.forget()
    R.forget_current()
    S._engine = S._SessionLocal = None


def _train(db, out, *extra):
    lines = []
    assert T.main(["--db", str(db), "--ratings", "none", "--out", str(out),
                   "--report", str(Path(out).with_suffix(".md")), *extra], log=lines.append) == 0
    return SM.load(out)


# --- labels -------------------------------------------------------------------------------

@pytest.mark.parametrize("rel, want", [
    ("Acme/Drum Kit/Kicks/Kick 01.wav", ("KICKS", "class.oneshot")),
    ("Acme/Drum Kit/Kicks/BD01.wav", ("KICKS", "class.oneshot")),
    ("Acme/Synth Pack/Pads/Warm 03.wav", ("PADS", None)),
    ("Acme/Loops/Drum Loops/Break 174 01.wav", ("DRUMLOOPS", "class.loop")),
    ("Acme/Loops/Bass Loops/Bassline 120 01.wav", ("PHRASES", "class.loop")),
    ("Acme/Loop Pack/Vox/Shout Loop 04.wav", ("other", "class.loop")),
    ("Acme/Grand Piano/Piano C3.wav", ("PIANO", None)),
    ("Acme/Strings/Violin/Violin Sustain 01.wav", ("ACOUSTIC", None)),
    ("Acme/Kick Pack/Misc/thing 01.wav", (None, None)),        # only the pack's name says kick
    ("Acme/Mixed/Kick and Snare 01.wav", (None, None)),        # two categories: no label
    ("Dump/001.wav", (None, None)),
])
def test_training_labels_come_from_the_files_own_name_and_folder(rel, want):
    assert SM.training_label(rel, 0.5) == want


def test_ratings_label_and_exclude():
    assert SM.rating_label({"verdict": "keep", "category": "HATS"}) == ("HATS", "class.oneshot")
    assert SM.rating_label({"verdict": "misfiled", "category": "HATS", "target": "DRUMLOOPS"}) == \
        ("DRUMLOOPS", "class.loop")
    assert SM.rating_label({"verdict": "keep", "category": "BIRDS"}) == ("other", None)
    assert SM.rating_label({"verdict": "drop", "category": "HATS"}) == (None, None)
    assert SM.rating_label({"verdict": "misfiled", "category": "HATS"}) == (None, None)
    assert SM.rating_excludes({"verdict": "misfiled", "category": "HATS"}) == "HATS"


def test_a_misfiled_rating_overrides_the_name(tmp_path):
    rated = tmp_path / "ratings.json"
    rated.write_text(json.dumps({"version": 1, "ratings": {
        f"{LIB}/a/Hats/Hat 01.wav": {"verdict": "misfiled", "category": "HATS"},
        f"{LIB}/a/Hats/Hat 02.wav": {"verdict": "misfiled", "category": "HATS", "target": "SNARES"},
        f"{LIB}/a/Dump/003.wav": {"verdict": "keep", "category": "PADS"}}}))
    rels = ["a/Hats/Hat 01.wav", "a/Hats/Hat 02.wav", "a/Dump/003.wav", "a/Hats/Hat 04.wav"]
    cats, shapes, src = T.labels([1, 2, 3, 4], rels, [0.2] * 4, [f"{LIB}/{r}" for r in rels], str(rated))
    assert cats == [None, "SNARES", "PADS", "HATS"]
    assert src == [None, "rating", "rating", "names"]


def test_the_holdout_is_split_by_pack():
    keys = [T.pack_key(r) for r, _k in _plan()]
    hold = T.split_packs(keys, 0.2)
    assert {k for k, h in zip(keys, hold) if h}.isdisjoint({k for k, h in zip(keys, hold) if not h})
    assert 0 < hold.sum() < len(keys)
    assert (T.split_packs(keys, 0.2) == hold).all()                          # the same every time
    assert T.pack_key("Acme/Kick Pack/Kicks/Kick 01.wav") == "Acme/Kick Pack"
    assert T.pack_key("Dump/001.wav") == "Dump"


# --- never another product's analysis -----------------------------------------------------

def _tokens(node, skip=()):
    """Every name, attribute and string a piece of code uses, but its docstrings and the
    top-level assignments and functions named in skip."""
    out = []
    body = getattr(node, "body", [])
    docs = {id(b) for b in body[:1] if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant)}
    for b in body:
        if id(b) in docs:
            continue
        if isinstance(b, (ast.FunctionDef, ast.Assign)) and (
                getattr(b, "name", None) in skip
                or any(isinstance(t, ast.Name) and t.id in skip for t in getattr(b, "targets", ()))):
            continue
        for n in ast.walk(b):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body \
                    and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                n.body[0].value.value = ""                        # a docstring says why, not what
            if isinstance(n, ast.Name):
                out.append(n.id)
            elif isinstance(n, ast.Attribute):
                out.append(n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.append(n.value)
            elif isinstance(n, ast.alias):
                out.append(n.name)
    return out


def test_training_never_reads_another_products_analysis():
    """The sound model's code and the trainer's (but the yardstick, which measures the
    saved model against Sononym afterwards) name none of another product's tables,
    providers or the columns computed from them, and its features are Fourier's own."""
    def hits(path, skip):
        toks = _tokens(ast.parse(Path(path).read_text()), skip)
        return sorted({(t[:60], w) for t in toks for w in SM.FORBIDDEN if w in t.lower()})
    assert not hits(inspect.getsourcefile(SM), {"FORBIDDEN"})
    trainer = inspect.getsourcefile(T)
    assert not hits(trainer, {"yardstick"})
    # (the guard itself works: the yardstick does read Sononym)
    assert hits(trainer, set())
    cols = [c for _n, c, _t in SM.OWN_FEATURES]
    assert all(c.split(".")[0] in ("s", "f") for c in cols)
    assert not [c for c in cols + [SM.FEATURE_SQL.lower()] for w in SM.FORBIDDEN if w in c.lower()]
    # the yardstick is read after the weights are written, and its result goes only to the report
    src = inspect.getsource(T.train)
    assert src.index("mdl.save(") < src.index("yardstick(")
    assert "lines += yardstick(" in src


def test_another_products_analysis_and_tags_change_nothing_in_training(tmp_path):
    """The same files trained with and without Sononym's analysis and Live's tags in the
    database (saying something else entirely): the same weights, to the bit."""
    plan = _plan()
    a = _train(_fill(tmp_path / "a.db", plan, sononym=True, ableton=True), tmp_path / "a.npz")
    b = _train(_fill(tmp_path / "b.db", plan), tmp_path / "b.npz")
    assert a.version == b.version
    da, db_ = np.load(tmp_path / "a.npz"), np.load(tmp_path / "b.npz")
    assert set(da.files) == set(db_.files)
    assert all(np.array_equal(da[k], db_[k]) for k in da.files)
    # the report measures Sononym afterwards, where it's there
    assert "agrees on" in (tmp_path / "a.md").read_text()
    assert "No Sononym analysis" in (tmp_path / "b.md").read_text()


# --- the model and its provider ------------------------------------------------------------

def test_the_weights_hold_no_file_names_and_load_back(tmp_path):
    db = _fill(tmp_path / "t.db", _plan())
    m = _train(db, tmp_path / "m.npz")
    raw = (tmp_path / "m.npz").read_bytes()
    assert b"Acme" not in raw and b"Kick 01" not in raw and b"SampleLibrary" not in raw
    d = np.load(tmp_path / "m.npz")
    assert {str(x) for x in d["categories"]} >= {"KICKS", "PADS", "other"}
    assert m.clap_model == "stand-in" and m.clap_revision == "" and len(m.version) == 12
    assert m.w_cat.shape == (512 + len(SM.OWN_FEATURES), len(SM.model_categories()))
    assert m.compatible("stand-in") and not m.compatible("laion/clap-htsat-unfused") and not m.compatible(None)
    rep = (tmp_path / "m.md").read_text()
    for part in ("## Labels", "## Holdout: category", "Confusion", "Coverage", "## Yardstick"):
        assert part in rep


def test_the_provider_labels_every_sample_once(tmp_path, monkeypatch):
    from fourier.cli import main
    plan = _plan()
    db = _fill(tmp_path / "t.db", plan)
    _train(db, tmp_path / "m.npz")
    with S.session_scope() as s:              # one sample embedded by another CLAP model
        s.execute(text("UPDATE sample_features SET clap_model = 'another' WHERE sample_id = 1"))
    S._engine = S._SessionLocal = None
    args = ["--db", str(db), "tools", "analyze", "--only", "sound"]
    r = CliRunner().invoke(main, args, catch_exceptions=False)
    out = " ".join(r.output.split())
    assert "No sound model: nothing to label" in out                     # no weights: nothing
    assert "No sound model yet: your library's own names label 143 samples across 24 packs" in out   # (one embedded by another CLAP model)
    monkeypatch.setenv("FOURIER_SOUND_MODEL", str(tmp_path / "m.npz"))
    r = CliRunner().invoke(main, args, catch_exceptions=False)
    out = " ".join(r.output.split())
    assert f"{len(plan) - 1:,} samples labelled, 0 already done" in out
    assert "not labelled: 1 with an embedding from another CLAP model" in out
    s = S.get_session()
    n = dict(s.execute(text("SELECT kind, count(*) FROM labels WHERE provider = 'fourier:sound' GROUP BY kind")).all())
    assert n == {"category": len(plan) - 1, "class": len(plan) - 1}
    got = SM.current(s)
    assert 1 not in got and len(got) == len(plan) - 1
    named = {i: kind for i, (rel, kind) in enumerate(plan, 1) if not rel.startswith("Dump/")}
    right = sum(got[i][0] == KINDS_CAT[k] for i, k in named.items() if i in got)
    assert right >= 0.95 * (len(named) - 1)
    assert all(0.0 <= p <= 1.0 and 0.0 <= pl <= 1.0 for _c, p, pl in got.values())
    s.close()
    r = CliRunner().invoke(main, args, catch_exceptions=False)          # incremental
    assert f"0 samples labelled, {len(plan) - 1:,} already done" in " ".join(r.output.split())
    monkeypatch.setenv("FOURIER_SOUND_MODEL", "off")                    # removed: its labels go
    CliRunner().invoke(main, args, catch_exceptions=False)
    s = S.get_session()
    assert s.execute(text("SELECT count(*) FROM labels WHERE provider = 'fourier:sound'")).scalar() == 0
    s.close()


KINDS_CAT = {"kick": "KICKS", "snare": "SNARES", "hat": "HATS", "pad": "PADS", "bass": "SUB",
             "break": "DRUMLOOPS"}


def test_a_bad_weights_file_is_one_error(tmp_path, monkeypatch):
    bad = tmp_path / "bad.npz"
    np.savez(bad, format=np.array(99))
    monkeypatch.setenv("FOURIER_SOUND_MODEL", str(bad))
    with pytest.raises(SM.SoundModelError, match="another format"):
        SM.model()
    monkeypatch.setenv("FOURIER_SOUND_MODEL", str(tmp_path / "missing.npz"))
    with pytest.raises(SM.SoundModelError, match="no weights file"):
        SM.model()


# --- routing --------------------------------------------------------------------------------

def _homes(db):
    """compute_homes over the database, as a build runs it."""
    from fourier.packs import curate as C
    S._engine = S._SessionLocal = None
    S.use_db(db)
    R.forget_current()
    C._HOMES_CACHE.clear()
    s = S.get_session()
    R.ensure_current(s)
    ids, emb = [], []
    for sid, blob in s.execute(text("SELECT sample_id, clap_embedding FROM sample_features ORDER BY sample_id")):
        ids.append(int(sid))
        emb.append(np.frombuffer(blob, dtype=np.float32))
    emb_n = np.stack(emb)
    emb_n = emb_n / np.linalg.norm(emb_n, axis=1, keepdims=True)
    lines = []
    got = C.compute_homes(s, emb_n, {i: r for r, i in enumerate(ids)}, log=lines.append)
    return s, got, lines


WEAK = [(f"Acme/Snare Pack/Misc/one {i + 1:02d}.wav", "hat") for i in range(4)]      # hats in a snare pack
STRONG = [(f"Acme/Snare Pack/Snares/Snare X{i + 1}.wav", "hat") for i in range(2)]    # named snares


def test_without_sononym_the_sound_model_places_a_flat_dump(tmp_path, monkeypatch):
    """A flat folder of numbered files: no word for any rule, and sounds the CLAP prompts
    describe too faintly for the CLAP fallback. A model trained on the named packs places
    them; without it, as before, they stay unplaced. Files only a pack's name describes
    follow a confident model; a file named as a snare stays one."""
    plan = _plan(weak=WEAK + STRONG)
    db = _fill(tmp_path / "t.db", plan)
    _train(db, tmp_path / "m.npz")
    want = {i: KINDS_CAT[k] for i, (rel, k) in enumerate(plan, 1) if rel.startswith("Dump/")}
    weak = [i for i, (rel, _k) in enumerate(plan, 1) if (rel, _k) in WEAK]
    strong = [i for i, (rel, _k) in enumerate(plan, 1) if (rel, _k) in STRONG]

    s, (homes, _sup, votes), lines = _homes(db)          # today: no weights
    before = sum(homes.get(i) == c for i, c in want.items())
    assert before <= len(want) // 4
    assert all(homes.get(i) == "SNARES" for i in weak + strong)
    s.close()

    monkeypatch.setenv("FOURIER_SOUND_MODEL", str(tmp_path / "m.npz"))
    s = S.get_session()
    SM.label(s)
    s.commit()
    s.close()
    s, (homes, _sup, votes), lines = _homes(db)
    after = sum(homes.get(i) == c for i, c in want.items())
    assert after >= 0.9 * len(want) and after > before
    i = next(iter(want))
    assert votes[i]["sound"][0] == homes[i] and votes[i]["sound"][1] >= 0.7
    assert all(homes.get(i) == "HATS" and votes[i]["sound"][0] == "HATS" for i in weak)
    assert all(homes.get(i) == "SNARES" and "sound" not in votes[i] for i in strong)
    assert any("the sound model placed" in ln for ln in lines)
    s.close()


def test_why_says_sound_model(tmp_path, monkeypatch):
    from fourier.metadata.providers import active
    from fourier.packs import why as W
    plan = _plan(dump=1)
    db = _fill(tmp_path / "t.db", plan)
    _train(db, tmp_path / "m.npz")
    monkeypatch.setenv("FOURIER_SOUND_MODEL", str(tmp_path / "m.npz"))
    S._engine = S._SessionLocal = None
    S.use_db(db)
    s = S.get_session()
    R.ensure_current(s)
    SM.label(s)
    s.commit()
    r = W._rows(s, f"{LIB}/Dump/001.wav")[0]
    got = SM.current(s, ids=[r.id])[r.id]
    docs = {"KICKS": {"sound_homed": {str(r.id): 0.86}}}
    w = W.explain_row(r, act=active(s), why_docs=docs.get, sound=got)
    out = "\n".join(W.format_why(w))
    assert w.decided == ("sound", "KICKS")
    assert "sound model: KICKS 0.86, no name rule matched" in out
    assert f"  sound model: {SM.describe_label(got)}" in out and "a report" not in out
    assert "sound" in W.RULE_IDS
    s.close()


def test_with_sononym_the_sound_model_routes_nothing(tmp_path, monkeypatch):
    """Sononym classifies: the sound model's labels are written (a report) and every home is
    what it is without them."""
    plan = _plan(weak=WEAK)
    db = _fill(tmp_path / "t.db", plan, sononym=True)
    _train(db, tmp_path / "m.npz")
    s, (before, _s1, _v1), _l = _homes(db)
    s.close()
    monkeypatch.setenv("FOURIER_SOUND_MODEL", str(tmp_path / "m.npz"))
    s = S.get_session()
    SM.label(s)
    s.commit()
    assert len(SM.current(s)) == len(plan)
    s.close()
    s, (after, _s2, votes), lines = _homes(db)
    assert after == before
    assert not any("sound" in (v or {}) for v in votes.values())
    assert not any("sound model" in ln for ln in lines)
    s.close()


@pytest.mark.skipif(not __import__("os").environ.get("FOURIER_GOLDEN"),
                    reason="set FOURIER_GOLDEN=1 (CI runs the golden builds as their own step)")
def test_the_synthetic_golden_is_unchanged_with_a_sound_model(tmp_path):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "golden" / "synthetic_build.py"),
                        str(tmp_path / "work"), "--check", "--sound"], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "golden: identical" in r.stdout


# --- trained on the user's own library ---------------------------------------------------------

def _analyze_sound(db):
    from fourier.cli import main
    S._engine = S._SessionLocal = None
    SM.forget()
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "analyze", "--only", "sound"],
                           catch_exceptions=False)
    return " ".join(r.output.split())


def test_the_weights_are_looked_for_in_the_home_then_the_package(tmp_path, monkeypatch):
    home = tmp_path / "fourier-home"
    assert SM.model_path() is None                                       # none anywhere
    pkg = tmp_path / "pkg.npz"
    pkg.write_bytes(b"x")
    monkeypatch.setattr(SM, "DEFAULT_PATH", pkg)
    assert SM.model_path() == pkg
    home.mkdir()
    (home / SM.HOME_NAME).write_bytes(b"x")
    assert SM.model_path() == home / SM.HOME_NAME                        # the library's own first
    monkeypatch.setenv("FOURIER_SOUND_MODEL", "off")
    assert SM.model_path() is None
    other = tmp_path / "other.npz"
    other.write_bytes(b"x")
    monkeypatch.setenv("FOURIER_SOUND_MODEL", str(other))
    assert SM.model_path() == other


def test_a_library_without_sononym_trains_its_own_model_once(tmp_path, monkeypatch):
    """The analysis's sound step trains the model on the library's own names when there's
    none yet and they label enough samples across enough packs, writes it to the Fourier home,
    labels with it, and leaves it alone until the labelled samples have grown."""
    monkeypatch.setattr(CC, "SOUND_TRAIN_MIN_LABELS", 100)
    plan = _plan()
    db = _fill(tmp_path / "t.db", plan)
    home = tmp_path / "fourier-home" / SM.HOME_NAME
    out = _analyze_sound(db)
    assert "Training a sound model on your library (144 samples its own names label, 24 packs)" in out
    assert "trained on 144 samples your library's own names label, across 24 packs" in out
    assert "agreed with the names" in out and "they stay on this machine" in out
    assert f"{len(plan):,} samples labelled, 0 already done" in out
    mdl = SM.load(home)
    assert mdl.trained_on == 144 and (home.parent / T.REPORT_NAME).is_file()
    out = _analyze_sound(db)                                             # again: nothing new
    assert "Training" not in out and f"0 samples labelled, {len(plan):,} already done" in out
    assert SM.load(home).version == mdl.version
    # more named samples than it learned from: trained again, and everything relabelled
    monkeypatch.setattr(T, "RETRAIN_MIN_NEW", 10)
    more = [(f"Northwind/Extra Pack/Kicks/Kick X{i:02d}.wav", "kick") for i in range(40)]
    db2 = _fill(tmp_path / "t2.db", plan + more)
    out = _analyze_sound(db2)
    assert "Training the sound model again: 184 samples your library's names label now, 144" in out
    assert SM.load(home).trained_on == 184 and SM.load(home).version != mdl.version


def test_no_model_is_trained_when_it_shouldnt_be(tmp_path, monkeypatch):
    """Not with Sononym (the model would route nothing), not when training is off or
    $FOURIER_SOUND_MODEL names weights, not below the gate (it says what's missing), and a
    model the user made by hand in the home is left alone."""
    home = tmp_path / "fourier-home" / SM.HOME_NAME
    plan = _plan()
    db = _fill(tmp_path / "t.db", plan)
    out = _analyze_sound(db)                                             # 144 < 1,000
    assert "No sound model yet" in out and "needs 1,000 across 8" in out and not home.exists()
    monkeypatch.setattr(CC, "SOUND_TRAIN_MIN_LABELS", 100)
    monkeypatch.setattr(CC, "SOUND_TRAIN_MIN_PACKS", 30)                 # 24 packs < 30
    assert "across 24 packs" in _analyze_sound(db) and not home.exists()
    monkeypatch.setattr(CC, "SOUND_TRAIN_MIN_PACKS", 8)
    monkeypatch.setattr(CC, "SOUND_TRAIN", False)
    assert "Training" not in _analyze_sound(db) and not home.exists()
    monkeypatch.setattr(CC, "SOUND_TRAIN", True)
    monkeypatch.setenv("FOURIER_SOUND_MODEL", "off")
    assert "Training" not in _analyze_sound(db) and not home.exists()
    monkeypatch.delenv("FOURIER_SOUND_MODEL")
    son = _fill(tmp_path / "s.db", plan, sononym=True)
    assert "Training" not in _analyze_sound(son) and not home.exists()
    # a model made by hand (no count of what it learned from) stays as it is
    made = _train(db, tmp_path / "m.npz")
    home.parent.mkdir(parents=True, exist_ok=True)
    by_hand = dataclasses.replace(made, trained_on=0).save(home)
    out = _analyze_sound(db)
    assert "Training" not in out and SM.load(home).version == by_hand.version


def test_tools_train_trains_now(tmp_path):
    from fourier.cli import main
    db = _fill(tmp_path / "t.db", _plan())
    S._engine = S._SessionLocal = None
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "train"], catch_exceptions=False)
    out = " ".join(r.output.split())
    assert r.exit_code == 0, out
    home = tmp_path / "fourier-home"
    assert (home / SM.HOME_NAME).is_file() and (home / T.REPORT_NAME).is_file()
    assert "trained on 144 samples" in out and "The next build" in out
    empty = _fill(tmp_path / "e.db", [])
    S._engine = S._SessionLocal = None
    r = CliRunner().invoke(main, ["--db", str(empty), "tools", "train"])
    assert r.exit_code == 1 and "no sound model: no sample has a CLAP embedding yet" in " ".join(r.output.split())


def test_an_unreliable_model_is_discarded_and_tried_again_only_after_growth(tmp_path, monkeypatch):
    """A model that isn't reliable on the packs it didn't learn from isn't kept: the home
    gets no weights, the report says why, and builds don't retrain until the library has
    grown as much as a retrain needs; `tools train --keep` keeps one anyway."""
    from fourier.cli import main
    monkeypatch.setattr(CC, "SOUND_TRAIN_MIN_LABELS", 100)
    monkeypatch.setattr(CC, "SOUND_KEEP_MIN_ACCURACY", 1.01)            # nothing is good enough
    plan = _plan()
    db = _fill(tmp_path / "t.db", plan)
    home = tmp_path / "fourier-home"
    out = _analyze_sound(db)
    assert "Not kept: its confident calls agreed with the names only" in out and "(101% needed)" in out
    assert "No sound model: nothing to label" in out
    assert not (home / SM.HOME_NAME).exists() and not list(home.glob(".*trial*"))
    assert json.loads((home / T.SKIPPED_NAME).read_text())["labelled"] == 144
    assert "## Verdict" in (home / T.REPORT_NAME).read_text() and "Not kept" in (home / T.REPORT_NAME).read_text()
    assert "Training" not in _analyze_sound(db)                         # not every build
    monkeypatch.setattr(T, "RETRAIN_MIN_NEW", 10)
    monkeypatch.setattr(CC, "SOUND_KEEP_MIN_ACCURACY", 0.8)
    more = [(f"Northwind/Extra Pack/Kicks/Kick X{i:02d}.wav", "kick") for i in range(40)]
    out = _analyze_sound(_fill(tmp_path / "t2.db", plan + more))
    assert "Training the sound model again: 184 samples your library's names label now, 144" in out
    assert (home / SM.HOME_NAME).is_file() and not (home / T.SKIPPED_NAME).exists()
    # by hand: --keep keeps one that isn't reliable
    monkeypatch.setattr(CC, "SOUND_KEEP_MIN_ACCURACY", 1.01)
    (home / SM.HOME_NAME).unlink()
    S._engine = S._SessionLocal = None
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "train"], catch_exceptions=False)
    assert "Not kept" in " ".join(r.output.split()) and not (home / SM.HOME_NAME).exists()
    S._engine = S._SessionLocal = None
    r = CliRunner().invoke(main, ["--db", str(db), "tools", "train", "--keep"], catch_exceptions=False)
    assert (home / SM.HOME_NAME).is_file() and "Kept on request" in (home / T.REPORT_NAME).read_text()


def test_reliable_needs_enough_held_out_and_confident_calls(monkeypatch):
    monkeypatch.setattr(CC, "SOUND_KEEP_MIN_HELD", 200)
    got = dict(held=150, covered=0.5, accuracy_at_min=0.95, at_min=0.7)
    assert T.reliable(got) == (False, "only 150 samples in packs it didn't learn from, too few "
                                      "to tell how reliable it is (200 needed)")
    got.update(held=500, covered=0.0)
    assert "of only under 1% of the samples" in T.reliable(got)[1]
    got.update(covered=0.05)
    assert "of only 5% of the samples" in T.reliable(got)[1]
    base = dict(version="v", labelled=10, packs=3, hold_packs=1, accuracy=0.2, shape_accuracy=0.8,
                out="o", report="r", kept=True)
    assert "at least 0.70 sure of under 1% of them" in T.summary_lines({**base, "covered": 0.001, "accuracy_at_min": 1.0, "at_min": 0.7})[1]
    assert "never 0.70 sure" in T.summary_lines({**base, "covered": 0.0, "accuracy_at_min": 0.0, "at_min": 0.7})[1]
    got.update(covered=0.4, accuracy_at_min=0.5)
    assert "agreed with the names only 50% of the time (80% needed)" in T.reliable(got)[1]
    got.update(accuracy_at_min=0.92)
    assert T.reliable(got) == (True, "")


def test_the_report_gives_precision_per_category_at_each_threshold():
    """Of the holdout samples placed in a category at or above a threshold, the share whose label
    agrees, with how many: here KICKS is right 2 of 3 times at 0.5 and 1 of 1 at 0.9."""
    import numpy as np

    from fourier.metadata.train import THRESHOLDS, report_training
    cats = ["KICKS", "SNARES"]
    P_h = np.array([[0.95, 0.05], [0.6, 0.4], [0.55, 0.45], [0.2, 0.8]])
    y_h = np.array([0, 0, 1, 1])
    info = dict(version="t", clap="c", n=4, labelled=4, shaped=0, packs=2, hold_packs=1,
                it_cat=1, t_cat=1.0, it_cls=0, t_cls=1.0, seconds=0.0)
    lines = report_training(cats, ["KICKS", "KICKS", "SNARES", "SNARES"], [True] * 4, P_h, y_h,
                            [None] * 4, [True] * 4, np.zeros(0), np.zeros(0, dtype=int),
                            ["names"] * 4, P_h, np.zeros(0), info)
    text = "\n".join(lines)
    assert "Precision per category" in text
    row = next(l for l in lines if l.startswith("| KICKS |") and "(3)" in l)
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1 + THRESHOLDS.index(0.5)] == "66.7% (3)"
    assert cells[1 + THRESHOLDS.index(0.9)] == "100.0% (1)"
    assert cells[1 + THRESHOLDS.index(0.95)] == "100.0% (1)"
    snares = next(l for l in lines if l.startswith("| SNARES |") and "(1)" in l)
    assert "100.0% (1)" in snares and "-" in snares        # 0.8 only; nothing at 0.9


def test_the_label_count_is_kept_until_what_it_depends_on_changes(tmp_path, monkeypatch):
    """label_counts reads every sample's names once; a second build with the same samples,
    ratings and settings reuses the count, and a renamed sample counts again."""
    import numpy as np

    from fourier.metadata import train as T
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FOURIER_RATINGS", str(tmp_path / "ratings.json"))
    rels = ["Northwind/Drums/Kick Qxvwz.wav", "Northwind/Drums/Snare Qxvwz.wav", "Other/Pads/Pad Qxvwz.wav"]
    rows = [np.array([1, 2, 3]), "clap", None, list(rels), [0.5, 0.4, 3.0], [f"/l/{r}" for r in rels]]
    monkeypatch.setattr(T, "library_rows", lambda session: tuple(rows))
    calls = []
    real = T.labels
    monkeypatch.setattr(T, "labels", lambda *a, **k: calls.append(1) or real(*a, **k))
    first = T.label_counts(None)
    assert T.label_counts(None) == first and len(calls) == 1
    assert (tmp_path / "home" / "run" / "sound_labels.json").is_file()
    rows[3][2] = "Other/Pads/Pad Qxvwz Two.wav"
    T.label_counts(None)
    assert len(calls) == 2
