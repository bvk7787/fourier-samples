"""One note of a multisample set on each C across categories (curate._cross_one_note, run by the
cross-category dedupe after every category has picked), its why log reason and verify's mirror."""
import json

from fourier.packs import curate as C
from fourier.packs import why_log
from fourier.packs.verify import FAIL, PASS, check_one_note_kept, check_pitch

LIB = "/l/SampleLibrary"


def _e(name, folder="Acme/Qzv Strings", **kw):
    """A retuned pick of a name-rooted note (the shift _retune_shift gives it)."""
    rt, src = C._retune_shift("SYNTH", name, None)
    return dict({"family": "x", "out": f"x/{name}", "src": f"{LIB}/{folder}/{name}", "retune": rt,
                 "root_src": src}, **kw)


def _cross(results, pins=None, prev=None):
    return C._cross_one_note(results, pins=pins, roots=({}, {}), prev=prev or {})


def test_two_categories_on_one_c_keep_the_smaller_shift():
    # F#1 +6 and D#2 -3 both play C2: the smaller shift stays, in its category
    res = [("SUB", {}, [_e("Qzv Strings F#1.wav")]), ("SYNTH", {}, [_e("Qzv Strings D#2.wav")])]
    got = _cross(res)
    assert got == {"SUB": {f"{LIB}/Acme/Qzv Strings/Qzv Strings F#1.wav":
                           ("SYNTH", f"{LIB}/Acme/Qzv Strings/Qzv Strings D#2.wav", "C2")}}


def test_a_keep_stays_and_the_previous_pick_breaks_a_tie():
    sub, syn = _e("Qzv Strings F#1.wav"), _e("Qzv Strings D#2.wav")
    res = [("SUB", {}, [sub]), ("SYNTH", {}, [syn])]
    assert set(_cross(res, pins={sub["src"]: "SUB"})) == {"SYNTH"}
    # a Keep in another category isn't one here
    assert set(_cross(res, pins={sub["src"]: "PADS"})) == {"SUB"}
    # both Keeps: both stay
    assert _cross(res, pins={sub["src"]: "SUB", syn["src"]: "SYNTH"}) == {}
    # A#1 +2 and D2 -2: the same shift, so the previous build's pick stays
    a, d = _e("Qzv Strings A#1.wav"), _e("Qzv Strings D2.wav")
    res = [("PADS", {}, [a]), ("SYNTH", {}, [d])]
    assert set(_cross(res, prev={"SYNTH": {d["src"]}})) == {"PADS"}
    assert set(_cross(res, prev={"PADS": {a["src"]}})) == {"SYNTH"}


def test_only_true_sets_retuned_notes_and_the_retuned_categories():
    f, d = _e("Qzv Strings F#1.wav"), _e("Qzv Strings D#2.wav")
    # another source folder: another set
    other = _e("Qzv Strings D#2.wav", folder="Acme/Qzv Strings Two")
    assert _cross([("SUB", {}, [f]), ("SYNTH", {}, [other])]) == {}
    # a category that doesn't retune (FX), and a note not retuned
    assert _cross([("FX", {}, [f]), ("SYNTH", {}, [d])]) == {}
    plain = {k: v for k, v in d.items() if k not in ("retune", "root_src")}
    assert _cross([("SUB", {}, [f]), ("SYNTH", {}, [plain])]) == {}
    # the same note (another velocity) in two categories: one note, both stay
    v2 = _e("Qzv Strings F#1 V2.wav")
    assert _cross([("SUB", {}, [f]), ("PADS", {}, [v2])]) == {}
    # two Cs apart: both stay
    assert _cross([("SUB", {}, [f]), ("SYNTH", {}, [_e("Qzv Strings D#3.wav")])]) == {}
    # a numbered series whose numbers don't track the roots (takes of one note) is no set
    n1 = dict(_e("Qzv Lead 01.wav"), retune=0.0, root_src="detect", root_midi=60.0)
    n5 = dict(_e("Qzv Lead 05.wav"), retune=-2.0, root_src="detect", root_midi=62.0)
    n9 = dict(_e("Qzv Lead 09.wav"), retune=1.0, root_src="detect", root_midi=59.0)
    assert _cross([("SYNTH", {}, [n1, n9]), ("PADS", {}, [n5])]) == {}


def test_numbers_that_track_the_pitch_are_a_set():
    ents = [dict(_e(f"Qzv Lead {n:03d}.wav"), retune=float(-(n % 12) if n % 12 <= 6 else 12 - n % 12),
                 root_src="detect", root_midi=float(n)) for n in (58, 59, 61, 62)]
    roots = ({e["src"]: (e["root_midi"], None) for e in ents}, {})
    got = C._cross_one_note([("SYNTH", {}, ents[:2]), ("PADS", {}, ents[2:])], roots=roots, prev={})
    # 058 +2 and 059 +1 (SYNTH), 061 -1 and 062 -2 (PADS) all play C4 (60): SYNTH's 059 and
    # PADS' 061 tie on a semitone, so the lower root breaks it; the other category's notes leave
    assert set(got) == {"PADS"} and len(got["PADS"]) == 2


def test_the_dedupe_drops_the_note_and_logs_it(tmp_path, monkeypatch):
    sub, syn = _e("Qzv Strings F#1.wav"), _e("Qzv Strings D#2.wav")
    for cat, e in (("SUB", sub), ("SYNTH", syn)):
        d = tmp_path / cat / "x"
        d.mkdir(parents=True)
        (d / e["out"].split("/")[1]).write_bytes(b"x")
        (tmp_path / cat / "_manifest.json").write_text(json.dumps([{"family": "x", "copied": 1, "n": 1}]))
    why_log.write(tmp_path, "SUB", {"category": "SUB", "kept": 9, "files": 1, "out": {}, "counts": {}})
    monkeypatch.setattr(C, "_sample_ids", lambda srcs: {sub["src"]: 7} if sub["src"] in srcs else {})
    monkeypatch.setattr(C, "_entry_roots", lambda paths: ({}, {}))
    monkeypatch.setattr(C, "_previous_families", lambda c: {})
    res = [("SUB", {"files": 1}, [sub]), ("SYNTH", {"files": 1}, [syn])]
    said = []
    assert C._dedupe_across(tmp_path, res, log=said.append) == 1
    assert res[0][2] == [] and res[0][1]["files"] == 0 and len(res[1][2]) == 1
    assert not (tmp_path / "SUB" / sub["out"]).exists() and (tmp_path / "SYNTH" / syn["out"]).exists()
    assert json.loads((tmp_path / "SUB" / "_manifest.json").read_text())[0]["copied"] == 0
    assert any("same C" in m for m in said)
    doc = why_log.read(tmp_path, "SUB")
    assert doc["out"]["cross_note"] == {"7": ["SYNTH", "Qzv Strings D#2.wav", "C2"]}
    assert doc["counts"]["cross_note"] == 1 and doc["files"] == 0
    assert "cross_note" in why_log.FILTER_WORDS
    line = why_log.describe("cross_note", doc["out"]["cross_note"]["7"], doc)
    assert "already plays in SYNTH as Qzv Strings D#2.wav" in line and "C2" in line


def test_a_copy_on_its_way_out_keeps_no_note_from_leaving():
    sub, syn = _e("Qzv Strings F#1.wav"), _e("Qzv Strings D#2.wav")
    res = [("SUB", {}, [sub]), ("SYNTH", {}, [syn])]
    assert C._cross_one_note(res, skip={("SYNTH", syn["src"])}, roots=({}, {}), prev={}) == {}


# --- verify ---------------------------------------------------------------------------------

def _master(root, picks, ratings=None):
    cats = {}
    for cat, e in picks:
        cats.setdefault(cat, {"families": 1, "files": 0, "source_samples": 0, "entries": []})["entries"].append(e)
    for cat, cd in cats.items():
        cd["files"] = len(cd["entries"])
        (root / cat).mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"categories": cats}))
    store = root.parent / f"{root.name}-ratings.json"
    store.write_text(json.dumps({"version": 1, "ratings": ratings or {}}))
    from fourier.packs.verify import _Ctx
    return _Ctx(str(root), store_path=str(store))


def _level(res, start):
    got = [r for r in res if r.check.startswith(start)]
    assert got, [r.check for r in res]
    return got[0].level


ACROSS = "one note of a multisample set on each C across categories"


def test_verify_fails_a_set_on_one_c_in_two_categories(tmp_path):
    sub, syn = _e("Qzv Strings F#1.wav"), _e("Qzv Strings D#2.wav")
    res = list(check_pitch(_master(tmp_path / "a", [("SUB", sub), ("SYNTH", syn)])))
    assert _level(res, ACROSS) == FAIL
    assert _level(res, "one note of a multisample set on each C (") == PASS     # each category alone
    assert "SUB/x/Qzv Strings F#1.wav" in [r for r in res if r.check.startswith(ACROSS)][0].detail
    # one of them gone, or two Cs apart: a pass
    assert _level(list(check_pitch(_master(tmp_path / "b", [("SYNTH", syn)]))), ACROSS) == PASS
    far = _e("Qzv Strings D#3.wav")
    assert _level(list(check_pitch(_master(tmp_path / "c", [("SUB", sub), ("SYNTH", far)]))), ACROSS) == PASS
    # a note Sononym or Live classes as a loop is retuned all the same: it counts
    loopy = dict(sub, loop_row=True)
    assert _level(list(check_pitch(_master(tmp_path / "d", [("SUB", loopy), ("SYNTH", syn)]))), ACROSS) == FAIL
    # Keeps stay (both rated Keep where they are)
    keeps = {sub["src"]: {"verdict": "keep", "category": "SUB"}, syn["src"]: {"verdict": "keep", "category": "SYNTH"}}
    assert _level(list(check_pitch(_master(tmp_path / "e", [("SUB", sub), ("SYNTH", syn)], keeps))), ACROSS) == PASS


def test_verify_fails_a_note_left_for_one_another_category_doesnt_hold(tmp_path):
    syn = _e("Qzv Strings D#2.wav")
    ctx = _master(tmp_path / "m", [("SYNTH", syn), ("SUB", _e("Vqx Bass C1.wav", folder="Northwind/Bass"))])
    why_log.write(ctx.root, "SUB", {"category": "SUB", "out": {"cross_note": {"7": ["SYNTH", "Qzv Strings D#2.wav", "C2"]}}})
    assert _level(list(check_one_note_kept(ctx)), "a note left out for another note") == PASS
    why_log.write(ctx.root, "SUB", {"category": "SUB", "out": {"cross_note": {"7": ["PADS", "Qzv Strings D#2.wav", "C2"]}}})
    res = list(check_one_note_kept(ctx))
    got = next(r for r in res if r.check.startswith("a note left out for another note"))
    assert got.level == FAIL and "PADS's Qzv Strings D#2.wav" in got.detail


# --- a keeper the dedupe drops hands its files on -------------------------------------------------

def _dedupe(tmp_path, monkeypatch, res, docs, ids, copies=None):
    """Run the cross-category dedupe on res with these why logs and {source: sample id}; copies:
    what _cross_copies returns, else nothing is a copy."""
    for cat, _s, es in res:
        for e in es:
            d = tmp_path / cat / e["out"].split("/")[0]
            d.mkdir(parents=True, exist_ok=True)
            (d / e["out"].split("/")[1]).write_bytes(b"x")
        (tmp_path / cat / "_manifest.json").write_text(json.dumps([{"family": "x", "copied": len(es), "n": 1}]))
    for cat, doc in docs.items():
        why_log.write(tmp_path, cat, doc)
    monkeypatch.setattr(C, "_sample_ids", lambda srcs: {s: i for s, i in ids.items() if s in srcs})
    monkeypatch.setattr(C, "_entry_roots", lambda paths: ({}, {}))
    monkeypatch.setattr(C, "_previous_families", lambda c: {})
    monkeypatch.setattr(C, "_cross_copies", lambda results, skip=(), pins=None: (copies or {}, {}))
    C._dedupe_across(tmp_path, res)
    return {cat: why_log.read(tmp_path, cat) for cat in docs}


def _verify(tmp_path, res, docs, ids):
    from fourier.packs.verify import check_copies
    ctx = _master(tmp_path / "m", [(c, e) for c, _s, es in res for e in es])
    for cat in docs:                      # (a category the dedupe emptied stays in the master)
        ctx.cats.setdefault(cat, {"entries": []})
    for cat, doc in docs.items():
        why_log.write(ctx.root, cat, doc)
    ctx.id_path = {i: s for s, i in ids.items()}
    got = list(check_copies(ctx)) + list(check_one_note_kept(ctx))
    return {r.check.split(" (")[0]: r.level for r in got}


def test_a_keeper_that_leaves_as_a_note_across_categories_hands_its_files_on(tmp_path, monkeypatch):
    # SUB kept F#1 over a copy of it (11) and over another note of its set on C2 (12); then F#1
    # leaves SUB because SYNTH plays the set on C2 too: both now point at SYNTH's note
    sub, syn = _e("Qzv Strings F#1.wav"), _e("Qzv Strings D#2.wav")
    ids = {sub["src"]: 7, syn["src"]: 8, f"{LIB}/Acme/Qzv Strings/Qzv Strings F#1 B.wav": 11}
    res = [("SUB", {"files": 1}, [sub]), ("SYNTH", {"files": 1}, [syn])]
    docs = {"SUB": {"category": "SUB", "files": 1, "counts": {"copy": 1, "one_note_per_c": 1},
                    "out": {"copy": {"11": 7}, "one_note_per_c": {"12": ["Qzv Strings F#1.wav", "C2"]}}},
            "SYNTH": {"category": "SYNTH", "files": 1, "out": {}, "counts": {}}}
    got = _dedupe(tmp_path / "a", monkeypatch, res, docs, ids)
    out = got["SUB"]["out"]
    assert not out.get("copy") and not out.get("one_note_per_c")
    assert out["cross_note"] == {"7": ["SYNTH", "Qzv Strings D#2.wav", "C2"],
                                 "11": ["SYNTH", "Qzv Strings D#2.wav", "C2"],
                                 "12": ["SYNTH", "Qzv Strings D#2.wav", "C2"]}
    assert got["SUB"]["counts"] == {"copy": 0, "one_note_per_c": 0, "cross_note": 3}
    assert set(_verify(tmp_path, res, got, ids).values()) == {PASS}


def test_a_keeper_another_category_keeps_hands_its_copies_on_to_the_end_of_the_chain(tmp_path, monkeypatch):
    # PADS and SYNTH both picked X; X stays in one of them, which then drops it as a copy of
    # STABS' Y: SYNTH's copy of X (21), and X itself, point at Y
    x, y = f"{LIB}/Acme/Qzv Pad 01.wav", f"{LIB}/Acme/Qzv Chord 01.wav"
    ids = {x: 20, y: 30, f"{LIB}/Acme/Qzv Pad 01 B.wav": 21}

    def picks():
        return [("PADS", {"files": 1}, [{"family": "x", "out": "x/a.wav", "src": x}]),
                ("STABS", {"files": 1}, [{"family": "x", "out": "x/b.wav", "src": y}]),
                ("SYNTH", {"files": 1}, [{"family": "x", "out": "x/c.wav", "src": x}])]
    res = picks()
    win = C._dup_winner(x, ["PADS", "SYNTH"])
    lose = ({"PADS", "SYNTH"} - {win}).pop()
    docs = {c: {"category": c, "files": 1, "out": {}, "counts": {}} for c in ("PADS", "STABS", "SYNTH")}
    docs[lose] = {"category": lose, "files": 1, "counts": {"copy": 1}, "out": {"copy": {"21": 20}}}
    got = _dedupe(tmp_path / "a", monkeypatch, res, docs, ids, copies={win: {x: ("STABS", y)}})
    assert got[lose]["out"]["cross_copy"] == {"20": 30, "21": 30}
    assert not got[lose]["out"].get("copy") and not got[lose]["out"].get("dup_source")
    assert got[win]["out"]["cross_copy"] == {"20": 30}
    assert set(_verify(tmp_path, res, got, ids).values()) == {PASS}
    # with no later step the copy points at X where the other category keeps it
    docs[lose] = {"category": lose, "files": 1, "counts": {"copy": 1}, "out": {"copy": {"21": 20}}}
    got = _dedupe(tmp_path / "b", monkeypatch, picks(), docs, ids)
    assert got[lose]["out"]["cross_copy"] == {"21": 20} and got[lose]["out"]["dup_source"] == {"20": win}
