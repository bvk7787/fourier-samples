"""fourier why: the routing precedence table and per-sample explanations."""
from types import SimpleNamespace

from fourier.packs import why as W


def _row(path, cats="", classes='["OneShot"]', tags="[]", dur=0.4, n_events=1):
    return SimpleNamespace(id=1, rel_path=path.split("SampleLibrary/")[-1], path=path,
                           filename=path.rsplit("/", 1)[-1], categories=cats, classes=classes,
                           ableton_tags=tags, harmonicity=0.2, chroma_concentration=None,
                           n_events=n_events, duration_s=dur, file_hash="h", bpm=None, tempo_bpm=None)


def test_precedence_ids_unique_and_ordered():
    assert len(set(W.RULE_IDS)) == len(W.RULE_IDS)
    assert W.RULE_IDS.index("pin") < W.RULE_IDS.index("override") < W.RULE_IDS.index("drum_name") \
        < W.RULE_IDS.index("votes")
    assert len(W.precedence_table()) == len(W.PRECEDENCE)


def test_drum_name_decides(tmp_path):
    r = _row("/lib/SampleLibrary/V/P/Snare 04.wav", cats='["Perc Kicks"]')
    w = W.explain_row(r, store_path=str(tmp_path / "none.json"))
    assert w.decided == ("drum_name", "SNARES")
    assert "KICKS" in w.votes["sononym"]


def test_keep_pin_beats_name(tmp_path):
    from fourier.packs.ratings import save_store
    p = "/lib/SampleLibrary/V/P/Snare 04.wav"
    store = str(tmp_path / "r.json")
    save_store({"version": 1, "ratings": {p: {"verdict": "keep", "category": "PERC"}},
                "colors": {}, "auto_misfiled": {}}, store)
    man = {"categories": {"PERC": {"entries": [{"src": p, "out": "perc-x/Snare 04.wav"}]}}}
    w = W.explain_row(_row(p, cats='["Perc Snares"]'), manifest=man, store_path=store)
    assert w.decided == ("pin", "PERC")
    assert w.in_master == [("PERC", "perc-x/Snare 04.wav")]
    text = "\n".join(W.format_why(w))
    assert "decided by: pin -> PERC" in text and "in the master: PERC/" in text


def test_not_selected_is_said(tmp_path):
    w = W.explain_row(_row("/lib/SampleLibrary/V/P/Kick 1.wav", cats='["Perc Kicks"]'),
                      manifest={"categories": {}}, store_path=str(tmp_path / "n.json"))
    text = "\n".join(W.format_why(w))
    assert "not in the master" in text and "the next `fourier build` records why" in text


def _doc(status="built", **out):
    from fourier.packs.curate import left_out_doc
    return left_out_doc("KICKS", status, 6 if status == "built" else 1, 6,
                        dict(total=8, near_dup=1, vendor_capped=0, gated_out=1),
                        {k: {1: v} for k, v in out.items()}, files=6, budget=825)


def _why(tmp_path, doc, **kw):
    w = W.explain_row(_row("/lib/SampleLibrary/V/P/Kick 1.wav", cats='["Perc Kicks"]'),
                      manifest={"categories": {}}, store_path=str(tmp_path / "n.json"),
                      why_docs=lambda cat: doc if cat == "KICKS" else None,
                      name_of=lambda sid: f"Acme/Pack/Item {sid}.wav", **kw)
    return "\n".join(W.format_why(w))


def test_why_says_what_the_build_recorded(tmp_path):
    """The real reason a build left a sample out (packs/why_log.py), not a list of guesses."""
    assert "left out of KICKS: a near-duplicate of Acme/Pack/Item 7.wav (CLAP), which KICKS kept" in \
        _why(tmp_path, _doc(near_dup=7))
    assert "over the per-vendor share: V has more than its share" in _why(tmp_path, _doc(vendor_capped="V"))
    assert "failed KICKS's CLAP gate" in _why(tmp_path, _doc(gated_out="clap"))
    assert "over budget: KICKS kept 6 of its 6 candidates (budget 825)" in \
        _why(tmp_path, _doc(over_budget=None))
    text = _why(tmp_path, _doc("too_few"))
    assert "left out of KICKS: KICKS was left empty: 8 found; 1 near-duplicates, 0 over the " \
           "per-vendor share, 1 failed the category's CLAP gate; need 6" in " ".join(text.split())
    # recorded, but not among the candidates the gates saw
    assert "not among the candidates the last build's gates saw" in _why(tmp_path, _doc())


def test_why_gives_each_files_clap_gate_result(tmp_path):
    """A gated category records each candidate's CLAP scores: why says the file failed (and
    by what), passed and was left out later (and why), or was readmitted to fill the minimum."""
    def doc(score, anti, out, readmitted=False, status="built"):
        d = _doc(status, **out)
        d.update(gate={"1": [score, anti] + ([1] if readmitted else [])}, gate_min=0.30)
        return " ".join(_why(tmp_path, d).split())
    assert "left out of KICKS: failed its CLAP gate (score 0.21 < 0.30)" in \
        doc(0.21, 0.1, {"gated_out": "clap"})
    assert "failed its CLAP gate (score 0.35 not above its anti-prompts' 0.40)" in \
        doc(0.35, 0.40, {"gated_out": "clap"})
    assert ("left out of KICKS: passed its CLAP gate (score 0.45 >= 0.30, anti-prompts 0.20); "
            "left out because a near-duplicate of Acme/Pack/Item 7.wav") in doc(0.45, 0.2, {"near_dup": 7})
    assert "passed its CLAP gate (score 0.45 >= 0.30, anti-prompts 0.20); left out because " \
           "failed KICKS's harmonicity gate" in doc(0.45, 0.2, {"gated_out": "harmonic"})
    assert "KICKS was left empty: 8 found;" in doc(0.1, 0.2, {}, status="too_few")
    assert "need 6; this file failed its CLAP gate (score 0.10 < 0.30)" in doc(0.1, 0.2, {}, status="too_few")
    # in the master: how it got past the gate
    from fourier.packs.curate import left_out_doc
    d = left_out_doc("KICKS", "built", 6, 6, dict(total=8), {}, gate={"1": [0.1, 0.2, 1]}, gate_min=0.3)
    p = "/lib/SampleLibrary/V/P/Kick 1.wav"
    w = W.explain_row(_row(p, cats='["Perc Kicks"]'), store_path=str(tmp_path / "n.json"),
                      manifest={"categories": {"KICKS": {"entries": [{"src": p, "out": "k/Kick 1.wav"}]}}},
                      why_docs=lambda cat: d if cat == "KICKS" else None)
    assert "KICKS: readmitted below its CLAP gate (score 0.10 < 0.30) to reach the minimum of 6: " \
           "a loop by its own name, folder or audio" in " ".join("\n".join(W.format_why(w)).split())


def test_why_names_the_near_duplicates_kept_partner_and_where_it_is(tmp_path):
    """A near-duplicate's line names the kept file it repeats (the build's partner, by id) and
    where the master holds it, or that it was kept as a candidate only."""
    assert ("left out of KICKS: a near-duplicate of Acme/Pack/Item 7.wav (CLAP), which KICKS "
            "kept (in the master: KICKS/punchy/Item 7.wav)") in " ".join(_why(
                tmp_path, _doc(near_dup=7), place_of=lambda sid: "KICKS/punchy/Item 7.wav" if sid == 7 else None
            ).split())
    text = " ".join(_why(tmp_path, _doc(near_dup=7), place_of=lambda sid: None).split())
    assert "which KICKS kept as a candidate; `fourier why` on it says where it went" in text


def test_why_gives_a_files_own_reason_in_a_category_left_empty(tmp_path):
    """A category left empty: its counts (each gate apart), then this file's own gates, with
    the value and threshold of the one that left it out."""
    from fourier.packs.curate import left_out_doc
    d = left_out_doc("DRUMLOOPS", "too_few", 3, 6, dict(total=5, gated_out=2, too_harmonic=3),
                     {"too_harmonic": {1: [0.74, 0.68]}}, gate={"1": [0.85, 0.5]}, gate_min=0.3)
    w = W.explain_row(_row("/lib/SampleLibrary/V/Loops/Loop 1.wav", classes='["Loop"]', dur=8.0),
                      manifest={"categories": {}}, store_path=str(tmp_path / "n.json"),
                      why_docs=lambda cat: d)
    w.left_out = []
    W._left_out(w, 1, ["DRUMLOOPS"], lambda cat: d, str)
    text = " ".join(" ".join(x for _c, x in w.left_out).split())
    assert ("DRUMLOOPS was left empty: 5 found; 0 near-duplicates, 0 over the per-vendor share, 2 "
            "failed the category's CLAP gate, 3 too harmonic for DRUMLOOPS; need 6; this file passed "
            "its CLAP gate (score 0.85 >= 0.30, anti-prompts 0.50), then failed DRUMLOOPS's "
            "harmonicity gate (harmonicity 0.74 > 0.68: too tonal)") in text, text


def test_why_log_round_trip(tmp_path, monkeypatch):
    from fourier.packs import why_log
    monkeypatch.setenv("FOURIER_HOME", str(tmp_path / "home"))
    master = tmp_path / "out" / "FourierCurated"
    why_log.write(str(master) + ".next", "KICKS", _doc(near_dup=7))     # a whole build's .next
    assert why_log.read(master, "KICKS")["out"] == {"near_dup": {"1": 7}}
    assert why_log.reason_for(why_log.read(master, "KICKS"), 1) == ("near_dup", 7)
    assert why_log.read(tmp_path / "elsewhere", "KICKS") is None
    assert not list(master.parent.glob("*")) if master.parent.exists() else True   # nothing in the master


def test_keys_name_is_reported_but_never_decides(tmp_path):
    w = W.explain_row(_row("/lib/SampleLibrary/V/P/Wurli C3.wav", cats='["Tone Leads & MidHiKeys"]'),
                      store_path=str(tmp_path / "n.json"))
    assert "keys" in [m[0] for m in w.matched] and w.decided == ("votes", None)


def test_drop_and_move_ratings(tmp_path):
    from fourier.packs.ratings import save_store
    p = "/lib/SampleLibrary/V/P/Snare 04.wav"
    store = str(tmp_path / "r.json")
    save_store({"version": 1, "ratings": {p: {"verdict": "drop", "category": "SNARES"}},
                "colors": {}, "auto_misfiled": {}}, store)
    assert W.explain_row(_row(p, cats='["Perc Snares"]'), store_path=store).decided == ("drop", None)
    save_store({"version": 1, "ratings": {p: {"verdict": "misfiled", "category": "SNARES", "target": "PERC"}},
                "colors": {}, "auto_misfiled": {}}, store)
    man = {"categories": {}, "sets": {"KITS": {"entries": [{"src": p, "out": "kit-909/Snare 04.wav"}]}}}
    w = W.explain_row(_row(p, cats='["Perc Snares"]'), manifest=man, store_path=store, mirrors={p})
    assert w.decided == ("pin", "PERC")
    assert {"mirror", "misfiled"} <= {m[0] for m in w.matched}
    assert w.in_master == [("KITS", "kit-909/Snare 04.wav")]
    assert "rating: misfiled in SNARES -> PERC" in "\n".join(W.format_why(w))


# --- rule precedence edge cases -----------------------------------------------------------

def test_a_transition_name_beats_a_chord_name(tmp_path):
    # a riser named with a chord word is FX's, not STABS'; verify wants named risers in FX
    w = W.explain_row(_row("/lib/SampleLibrary/V/P/FX/Riser Chord Soft.wav", cats='["Tone Pads & Textures"]',
                           dur=2.0), store_path=str(tmp_path / "n.json"))
    assert w.decided == ("fx_transition", "FX")
    assert "stab" not in [m[0] for m in w.matched]


def test_a_shapeless_file_from_a_percussion_folder_is_percs(tmp_path):
    # with no classifier shape (Live only), a SYNTH vote from a Foley/Metallic folder is still PERC's
    w = W.explain_row(_row("/lib/SampleLibrary/V/P/Foley/Metallic/Metallic 5.wav",
                           tags='["Bell", "One Shot"]', classes="[]"),
                      store_path=str(tmp_path / "n.json"))
    assert w.decided == ("perc_source", "PERC")


def test_a_loop_tagged_riser_chord_keeps_the_stab_rule(tmp_path):
    # verify holds neither rule on a file Live or a classifier calls a loop; the stab rule keeps it in STABS
    w = W.explain_row(_row("/lib/SampleLibrary/V/P/FX/Riser Chord Soft.wav", cats='["Tone Pads & Textures"]',
                           tags='["Atmosphere", "Loop"]', dur=2.0), store_path=str(tmp_path / "n.json"))
    assert "stab" in [m[0] for m in w.matched]


def test_the_votes_name_the_providers_that_cast_them(tmp_path):
    from fourier.metadata.providers import FALLBACK, Active
    r = _row("/lib/SampleLibrary/V/P/Kick 1.wav")
    r.canonical = ["class.oneshot", "kick"]           # the built-in providers' labels, merged
    w = W.explain_row(r, store_path=str(tmp_path / "n.json"),
                      act=Active(FALLBACK, FALLBACK, (), "auto"))
    assert w.votes == {"path+audio": ["KICKS"], "path words": []}
    assert w.labels["path"] == ["kick"] and w.labels["audio"] == ["class.oneshot"]
    text = "\n".join(W.format_why(w))
    assert "votes: path + audio: KICKS | path words (no Live): -" in text and "sononym" not in text
