"""What a rating covers (identity.Reach, curate._rating_reach): a Drop or Misfiled reaches its
file's byte copies and editions, and the byte copies of those editions; a Drop stays note-level and its set never refills the
dropped note's slot; a Misfiled covers its multisample set; `--set` (or a `scope` column)
widens either to the set or patch. Invented names only."""
import pytest

from fourier.packs import curate as C
from fourier.packs import identity as I
from fourier.packs import ratings as R

LIB = "/x/SampleLibrary"


def p(rel):
    return f"{LIB}/{rel}"


WAV = p("Acme/Drums/Style A/117/Qzv_Brk_7_117.wav")
AIF = p("Acme/Drums/Style A/117/Qzv_Brk_7_117.aif")
LEAD = [p(f"Acme/Ms/Qzv Env Lead {n}.wav") for n in ("059", "061", "062")]
NOTES = [p(f"Acme/Str/vqxP_{n}_v2.wav") for n in ("C1", "E1", "G1")]
TAKES = [p(f"Acme/Fx/Drop/Qzv Drop Zap {v}.wav") for v in ("A 01", "B 03", "C 05")]


def test_a_drop_reaches_its_editions_and_byte_copies():
    reach = I.Reach({WAV, NOTES[0]}, facts={WAV: ("h1", 4.0)})
    assert reach.dropped(AIF, None, 4.0) == WAV                         # the .aif edition
    assert reach.dropped(AIF, None, 2.0) is None                        # another length: another loop
    assert reach.dropped(p("Vqx/Other/Copy Of Brk.wav"), "h1", 4.0) == WAV     # the same bytes
    assert reach.dropped(p("Acme/Drums/Style B/117/Qzv_Brk_7_117.wav"), None, 4.0) is None
    # note-level: the set's other notes stay
    assert reach.dropped(NOTES[1]) is None


def test_a_set_wide_drop_covers_the_set_or_patch():
    reach = I.Reach({NOTES[0], TAKES[0]}, scopes={NOTES[0]: "set", TAKES[0]: "set"})
    assert reach.dropped(NOTES[2]) == NOTES[0]
    assert reach.dropped(TAKES[2]) == TAKES[0]                           # a described patch's takes
    assert reach.dropped(p("Acme/Str B/vqxP_E1_v2.wav")) is None        # another source folder
    assert I.Reach({p("Acme/Kit/Kick 01.wav")}, scopes={p("Acme/Kit/Kick 01.wav"): "set"}).dropped(
        p("Acme/Kit/Kick 02.wav")) is None                              # a series is never a set


def test_a_dropped_notes_slot_never_refills_from_its_set():
    reach = I.Reach({NOTES[0]}, drop_cats={NOTES[0]: "ACOUSTIC"})
    assert reach.refill_of(NOTES[1], "ACOUSTIC") == NOTES[0]
    assert reach.refill_of(NOTES[1], "PIANO") is None                   # rated in another category
    assert reach.refill_of(p("Acme/Str/Qzv Other C1.wav"), "ACOUSTIC") is None
    # numbers that track the pitch are a set (members: read with their roots)
    members = {x: ("acme/ms", "qzv env lead") for x in LEAD}
    reach = I.Reach({LEAD[0]}, members=members, drop_cats={LEAD[0]: "SUB"})
    assert reach.refill_of(LEAD[2], "SUB") == LEAD[0] and reach.dropped(LEAD[2]) is None
    assert I.Reach({LEAD[0]}, drop_cats={LEAD[0]: "SUB"}).refill_of(LEAD[2], "SUB") is None


def test_a_misfiled_covers_its_set_editions_and_with_set_its_patch():
    idx = C._MisfiledIndex({NOTES[0]: "PIANO", WAV: "PHRASES", TAKES[0]: "BLIPS"})
    assert idx.categories(NOTES[2]) == {"PIANO"}
    assert idx.categories(AIF, dur=4.0) == {"PHRASES"}
    assert idx.categories(TAKES[1]) == set()                            # a patch's take: not its set
    reach = I.Reach(misfiled={TAKES[0]: "BLIPS"}, scopes={TAKES[0]: "set"})
    assert C._MisfiledIndex({TAKES[0]: "BLIPS"}, reach).category(TAKES[1]) == "BLIPS"


# --- the store and the CSV --------------------------------------------------------------------------

def _target(src, cat="ACOUSTIC"):
    return dict(src=src, category=cat, family="fam", out="fam/x.wav", path=f"{cat}/fam/x.wav")


def test_set_ratings_stores_a_scope(tmp_path):
    store = tmp_path / "ratings.json"
    (tmp_path / "manifest.json").write_text('{"generated": "t", "categories": {}}')
    R.set_ratings(str(tmp_path), [(_target(NOTES[0]), "drop", None, "set"), (_target(WAV, "PHRASES"), "drop", None)],
                  store_path=str(store))
    assert R.rating_scopes(str(store)) == {NOTES[0]: "set"}
    assert R.drop_categories(str(store)) == {NOTES[0]: "ACOUSTIC", WAV: "PHRASES"}
    assert R.drop_set(str(store)) == {NOTES[0], WAV}
    with pytest.raises(ValueError, match="Keep is for one file"):
        R.set_ratings(str(tmp_path), [(_target(LEAD[0]), "keep", None, "set")], store_path=str(store))
    with pytest.raises(ValueError, match="scope"):
        R.set_ratings(str(tmp_path), [(_target(LEAD[0]), "drop", None, "most")], store_path=str(store))


def test_the_ratings_csv_takes_an_optional_scope(tmp_path, monkeypatch):
    targets = [_target(NOTES[0]), _target(WAV, "PHRASES"), _target(LEAD[0], "SUB")]
    monkeypatch.setattr(R, "rateable", lambda m: targets)
    monkeypatch.setattr(R, "find_rateable", lambda m, q, t=None: [x for x in targets if q in x["src"]])
    csv = tmp_path / "r.csv"
    csv.write_text("path,rating,category,scope\nvqxP_C1,drop,,set\nQzv_Brk,drop,,\n"
                   "Env Lead 059,keep,,set\nEnv Lead 059,drop,,most\n")
    rated, skipped = R.read_ratings_csv(str(csv), str(tmp_path))
    assert [(t["src"], v, s) for t, v, _m, s in rated] == [(NOTES[0], "drop", "set"), (WAV, "drop", None)]
    assert [line for line, _q, _w in skipped] == [4, 5]


# --- from the database --------------------------------------------------------------------------------

def test_rating_reach_reads_hashes_lengths_and_numbered_sets(tmp_path):
    from fourier.db import session as sess
    from fourier.db.models import Sample
    sess._engine = sess._SessionLocal = None
    sess.init_db(tmp_path / "library.db")
    s = sess.get_session()
    rows = [(LEAD[0], 59, None, 1.0), (LEAD[1], 61, None, 1.0), (LEAD[2], 62, None, 1.0),
            (WAV, None, "h1", 4.0), (p("Vqx/Other/Copy Of Brk.wav"), None, "h1", 4.0)]
    for path, root, h, d in rows:
        s.add(Sample(path=path, filename=path.rsplit("/", 1)[1], root_note=root, file_hash=h, duration_s=d))
    s.commit()
    s.close()
    try:
        C._REACH.clear()
        reach = C._rating_reach(misfiled={}, drops={LEAD[0]: "SUB", WAV: "DRUMLOOPS"}, scopes={})
        assert reach.facts[WAV] == ("h1", 4.0)
        assert reach.refill_of(LEAD[1], "SUB") == LEAD[0]               # 059/061/062 rooted 59/61/62
        assert reach.dropped(p("Vqx/Other/Copy Of Brk.wav"), "h1", 4.0) == WAV
    finally:
        C._REACH.clear()
        sess._engine = sess._SessionLocal = None


# a rated file, its compilation edition (another pack of the vendor, same name and length), a
# byte copy of that edition under another name, and an edition of the copy by the copy's name
RATED = p("Acme/Qzv Essentials 16bit/Kits/Zap Kit/Qzv Boom Thud Lo.wav")
EDITION = p("Acme/Zap Kit/WAV/Qzv Boom Thud Lo.wav")
COPY = p("Acme/Qzv Essentials/Drums/FX/BD Qzv Vrrb 07.wav")
COPY_ED = p("Acme/Qzv Essentials 16bit/Drums/FX/BD Qzv Vrrb 07.wav")


def test_a_rating_reaches_byte_copies_of_its_editions_only_one_step():
    reach = I.Reach({RATED}, facts={RATED: ("hR", 0.6)}, copies={"hE": [RATED], "hX": [p("Acme/Other.wav")]})
    assert reach.dropped(EDITION, "hE", 0.6) == RATED                   # the edition (by name and bytes)
    assert reach.dropped(COPY, "hE", 0.6) == RATED                      # its bytes under another name
    assert reach.dropped(COPY_ED, "hD", 0.6) is None                    # the copy's own edition: not
    assert reach.dropped(COPY, "hX", 0.6) is None                       # an unrated source's hash
    assert I.Reach({RATED}).dropped(COPY, "hE", 0.6) is None            # no copies read: names only
    idx = C._MisfiledIndex({RATED: "KICKS"}, I.Reach(misfiled={RATED: "KICKS"}, copies={"hE": [RATED]}))
    assert idx.categories(COPY, file_hash="hE") == {"KICKS"}
    assert idx.rated_by(COPY, "KICKS", "hE", 0.6) == "Qzv Boom Thud Lo.wav"
    assert idx.categories(COPY_ED, file_hash="hD") == set()
    for x in (RATED, COPY, p("WAV/Acme (WAV)/Kit/Qzv Hit.wav"), "/elsewhere/Acme (WAV)/Qzv Hit.wav"):
        assert I.vendor_of(x) == I.edition_key(x)[0]                    # the cheap pass's vendor


def _library_db(tmp_path, rows):
    from fourier.db import session as sess
    from fourier.db.models import Sample
    sess._engine = sess._SessionLocal = None
    sess.init_db(tmp_path / "library.db")
    s = sess.get_session()
    for path, h, d in rows:
        s.add(Sample(path=path, filename=path.rsplit("/", 1)[1], file_hash=h, duration_s=d))
    s.commit()
    s.close()


EDITION_ROWS = [(RATED, "hR", 0.6), (EDITION, "hE", 0.6), (COPY, "hE", 0.6), (COPY_ED, "hD", 0.6),
                (p("Acme/Zap Kit/WAV/Qzv Boom Thud Hi.wav"), "hH", 0.6),          # not an edition
                (p("Northwind/Hits/Qzv Hit 9.wav"), "hH", 0.6)]


def test_rating_reach_reads_the_hashes_of_the_rated_files_editions(tmp_path):
    from fourier.db import session as sess
    _library_db(tmp_path, EDITION_ROWS + [(p("Acme/Zap Kit/Short/Qzv Boom Thud Lo.wav"), "hS", 0.3),
                                          (p("Vqx/Short Copy.wav"), "hS", 0.3)])
    try:
        C._REACH.clear()
        assert C._rated_facts([RATED])[2] == {"hE": [RATED]}            # another length: no edition
        reach = C._rating_reach(misfiled={}, drops={RATED: "STABS"}, scopes={})
        assert reach.dropped(COPY, "hE", 0.6) == RATED
        assert reach.dropped(COPY_ED, "hD", 0.6) is None
        assert reach.dropped(p("Northwind/Hits/Qzv Hit 9.wav"), "hH", 0.6) is None
        assert reach.dropped(p("Vqx/Short Copy.wav"), "hS", 0.3) is None
        C._REACH.clear()
        reach = C._rating_reach(misfiled={RATED: "KICKS"}, drops={}, scopes={}, auto=())
        assert reach.misfiled(COPY, "hE", 0.6) == {"KICKS"}
    finally:
        C._REACH.clear()
        sess._engine = sess._SessionLocal = None


def test_verify_reads_the_same_reach(tmp_path):
    """check_ratings and the drum-name check see a byte copy of a rated file's edition as the
    build does."""
    import json

    import numpy as np
    import soundfile as sf

    from fourier.db import session as sess
    from fourier.packs.verify import FAIL, PASS, _Ctx, check_ratings, check_routing
    _library_db(tmp_path, EDITION_ROWS)

    def ctx(d, cat, verdict):
        root = d / "master"
        (root / cat / "dark").mkdir(parents=True)
        sf.write(root / cat / "dark" / "BD_Qzv_Vrrb_07.wav", np.zeros(2205, dtype="float32") + 0.1, 44100)
        e = {"family": "dark", "out": "dark/BD_Qzv_Vrrb_07.wav", "src": COPY}
        (root / "manifest.json").write_text(json.dumps({"categories": {cat: {"entries": [e]}}}))
        store = d / "ratings.json"
        store.write_text(json.dumps({"version": 1, "ratings": {RATED: {"category": "KICKS", "verdict": verdict}}}))
        return _Ctx(str(root), store_path=str(store), prof={COPY: (1, 0.0, 0.0, "hE")})

    def level(res, check):
        return [r.level for r in res if r.check.startswith(check)][0]
    try:
        C._REACH.clear()
        res = list(check_ratings(ctx(tmp_path / "a", "STABS", "drop")))
        assert level(res, "no Drop in the master") == FAIL
        assert "Qzv Boom Thud Lo.wav" in [r for r in res if r.check.startswith("no Drop")][0].detail
        C._REACH.clear()
        # named as a kick, outside KICKS: its edition's Misfiled out of KICKS says it's no kick
        res = list(check_routing(ctx(tmp_path / "b", "TOMS", "misfiled")))
        assert level(res, "files named as a drum") == PASS
    finally:
        C._REACH.clear()
        sess._engine = sess._SessionLocal = None


def test_review_rate_set_flag(tmp_path):
    from click.testing import CliRunner

    from fourier.cli import main
    r = CliRunner().invoke(main, ["review", "rate", "x", "keep", "--set", "--from", str(tmp_path)])
    assert r.exit_code == 2 and "--set goes with drop or misfiled" in r.output
