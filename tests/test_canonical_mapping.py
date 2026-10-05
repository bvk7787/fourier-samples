"""The canonical mapping (config/providers/sononym.yaml, metadata/vocab.py) routes exactly
as the old Sononym-name predicates did.

The combinations are generated from Sononym's own labels (config/providers/sononym.yaml):
every class list, with no category, one category, or any two. Both the old predicates and
the new ones vote when any of a file's labels matches, so single labels and pairs cover
every combination a library can hold. For each one and
each category, the old predicates (substring matches on Sononym's names, frozen below as
they were) and the new ones (canonical labels) must agree, in both places curation asks:
the classifier votes in compute_homes / pack why, and the SQL that fetches a category's
candidates."""
import itertools

import pytest
from sqlalchemy import text

from fourier.db import session as S
from fourier.db.models import Sample, SononymMeta
from fourier.metadata import rows as R
from fourier.metadata import vocab
from fourier.packs import curate as C
from fourier.packs.curate_config import CATEGORIES

def _combos():
    m = vocab.mapping("sononym")
    cats = sorted(raw for kind, raw in m if kind == "category")
    classes = [[], ["OneShot"], ["Loop"], ["OneShot", "Loop"], ["Loop", "OneShot"]]
    labelsets = [[]] + [[c] for c in cats] + [list(p) for p in itertools.combinations(cats, 2)]
    return [{"classes": cl, "categories": ls} for cl in classes for ls in labelsets]


COMBOS = _combos()

# the old taxonomy, in Sononym's names (curate_config CATEGORIES "cat", PIANO_POOL_CATS)
OLD_CAT = {
    "KICKS": ["Perc Kicks"], "SNARES": ["Perc Snares"], "HATS": ["Perc Hats & Shakers"],
    "SUB": ["Tone Bass & LowKeys"], "SYNTH": ["Tone Leads & MidHiKeys", "Tone Triangles & Bells"],
    "CLAPS": ["Perc Claps"], "TOMS": ["Perc Toms"],
    "PERC": ["Perc Bongos & Congas", "Perc Wood Hits", "Perc Metal Hits", "Perc Vibraslap & Guiro",
             "Perc Snips & Snaps"],
    "CYMBALS": ["Perc Cymbal Crashes", "Perc Cymbal Rides"],
    "FX": ["XFX Nature & Athmospheric", "XFX Noise & Distortion", "XFX Explosions & Shots",
           "XFX Sweeps & Lasers", "XFX Whooshes & Whips", "XFX Cracks & Rustle"],
    "VOX": ["Tone Voice & Acapella"], "PADS": ["Tone Pads & Textures"],
    "STABS": ["Tone Stabs & Orch. Hits"], "BLIPS": ["Tone Blips & HighKeys", "Perc Zaps & Blips"],
    "DRUMLOOPS": [], "PHRASES": [], "WAVES": [],
}
OLD_PIANO_POOL = ["Tone Leads & MidHiKeys", "Tone Bass & LowKeys", "Tone Blips & HighKeys",
                 "Tone Stabs & Orch. Hits"]
VOTING = {c: cfg for c, cfg in CATEGORIES.items() if cfg["kind"] in ("oneshot", "gated", "loop")}


def _canon(classes, categories):
    m = vocab.mapping("sononym")
    return {m[("class", c)] for c in classes} | {m[("category", c)] for c in categories}


def _old_votes(classes, categories):
    """compute_homes' classifier votes, the old way."""
    is_oneshot = any("OneShot" in c for c in classes)
    is_loop = any("Loop" in c for c in classes)
    out = set()
    for cat, cfg in VOTING.items():
        k = cfg["kind"]
        if (k == "oneshot" and is_oneshot) or k == "gated":
            if any(any(sub in sc for sc in categories) for sub in OLD_CAT[cat]):
                out.add(cat)
        elif k == "loop" and is_loop and cat != C.PHRASES_CATEGORY:
            out.add(cat)
    return out


def test_sononyms_labels_are_all_mapped_to_canonical_labels():
    m = vocab.mapping("sononym")
    assert len([k for k, _ in m if k == "class"]) == 2 and len([k for k, _ in m if k == "category"]) == 28
    assert set(m.values()) <= vocab.CANONICAL
    assert len({(tuple(c["classes"]), tuple(c["categories"])) for c in COMBOS}) == len(COMBOS)


def test_the_taxonomy_names_only_canonical_labels():
    for cat, cfg in CATEGORIES.items():
        assert "cat" not in cfg, cat
        assert set(cfg.get("labels") or ()) <= vocab.CANONICAL, cat
        assert set(cfg.get("pool_labels") or ()) <= vocab.CANONICAL, cat


def test_votes_match_the_old_predicates_for_every_combination():
    diffs = [(x["classes"], x["categories"], sorted(_old_votes(x["classes"], x["categories"])),
              sorted(C._classifier_votes(VOTING, _canon(x["classes"], x["categories"]))))
             for x in COMBOS]
    diffs = [d for d in diffs if d[2] != d[3]]
    assert not diffs, diffs[:5]


@pytest.fixture(scope="module")
def combos_db(tmp_path_factory):
    """One sample per combination, fetched through metadata/rows.py."""
    S._engine = S._SessionLocal = None
    R.forget_current()
    S.init_db(tmp_path_factory.mktemp("combos") / "c.db")
    with S.session_scope() as s:
        for i, x in enumerate(COMBOS, 1):
            s.add(Sample(id=i, path=f"/l/{i}.wav", rel_path=f"{i}.wav", filename=f"{i}.wav"))
            s.add(SononymMeta(sample_id=i, classes=x["classes"], categories=x["categories"]))
    s = S.get_session()
    R.fetch(s, R.sample_select("id"))                  # builds labels and descriptors
    yield s
    s.close()
    S._engine = S._SessionLocal = None


def _old_sql(session, kind, cats, cls=None):
    """The old candidate filter on sononym_meta (curate._fetch_rows)."""
    where, p = [], {}
    if cls:
        where.append("classes LIKE :cls")
        p["cls"] = f"%{cls}%"
    if cats:
        where.append("(" + " OR ".join(f"categories LIKE :c{i}" for i in range(len(cats))) + ")")
        p.update({f"c{i}": f"%{c}%" for i, c in enumerate(cats)})
    sql = "SELECT sample_id FROM sononym_meta" + (" WHERE " + " AND ".join(where) if where else "")
    return {r[0] for r in session.execute(text(sql), p)}


def _new_sql(session, labels, cls=None):
    q = R.sample_select("id")
    for sql, p in ([R.like_any("canonical", [cls], "k", quoted=True)] if cls else []) + \
                  ([R.like_any("canonical", labels, "c", quoted=True)] if labels else []):
        q = q.where(text(sql).bindparams(**p))
    return {r.id for r in R.fetch(session, q)}


@pytest.mark.parametrize("cat", [c for c, cfg in CATEGORIES.items() if cfg["kind"] in ("oneshot", "loop", "gated")])
def test_candidate_sql_matches_the_old_filter(combos_db, cat):
    cfg = CATEGORIES[cat]
    kind = cfg["kind"]
    if kind in ("oneshot", "loop"):
        cls_old = "OneShot" if kind == "oneshot" else "Loop"
        cls_new = "class.oneshot" if kind == "oneshot" else "class.loop"
        old = _old_sql(combos_db, kind, OLD_CAT[cat], cls_old)
        new = _new_sql(combos_db, cfg.get("labels"), cls_new)
    else:
        old = _old_sql(combos_db, kind, OLD_CAT[cat])
        new = _new_sql(combos_db, cfg.get("labels"))
    assert old == new and (old or not OLD_CAT[cat])


def test_piano_pool_matches_the_old_filter(combos_db):
    old = _old_sql(combos_db, "oneshot", OLD_PIANO_POOL, "OneShot")
    new = _new_sql(combos_db, CATEGORIES["PIANO"]["pool_labels"], "class.oneshot")
    assert old == new and old
