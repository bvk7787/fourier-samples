"""Family naming rules (packs/naming.py): each case is a way a folder name can mislead."""
import numpy as np

from fourier.packs import naming as N
from fourier.packs.curate_config import CATEGORIES, NAME_PHRASES

DRUM_WORDS = {"gated", "clicky", "snappy", "punchy", "squashed", "sub-heavy", "long-tail",
              "low-tuned", "high-tuned", "thin"}


def _words_of(dims):
    return {w for _, pair in dims for w in pair if w}


# --- vocabulary per category -------------------------------------------------

def test_tonal_categories_use_tonal_vocabulary():
    for cat in N.TONAL_CATS:
        dims = N.naming_dims(cat, CATEGORIES[cat])
        assert not (_words_of(dims) & DRUM_WORDS), cat      # "a3-vocal-pad-gated"
        assert "tune" not in dict(dims)                      # note lead carries pitch


def test_weight_word_is_per_category():
    assert dict(N.naming_dims("KICKS", CATEGORIES["KICKS"]))["sub"] == ("thin", "sub-heavy")
    assert dict(N.naming_dims("SNARES", CATEGORIES["SNARES"]))["sub"] == ("thin", "full")
    for cat in ("HATS", "CYMBALS", "CLAPS"):                 # "pedal-hat" sub-heavy
        d = dict(N.naming_dims(cat, CATEGORIES[cat]))
        assert "sub" not in d and "tune" not in d            # "bright-ride-low-tuned"


def test_drum_loops_never_say_drums():
    dims = N.naming_dims("DRUMLOOPS", CATEGORIES["DRUMLOOPS"])
    assert "drums" not in _words_of(dims)
    assert "musical" in _words_of(dims)


def test_instrument_categories_keep_their_configured_dims():
    assert N.naming_dims("ACOUSTIC", CATEGORIES["ACOUSTIC"]) == CATEGORIES["ACOUSTIC"]["dims"]


def test_naming_phrases_are_separate_from_gating_phrases():
    for cat, extra in NAME_PHRASES.items():
        assert cat in CATEGORIES
        assert not set(extra) & set(CATEGORIES[cat]["phrases"]), cat
        assert all(p == p.lower() for p in extra)


# --- traits --------------------------------------------------------------------

def _oneshot(**kw):
    base = dict(tune=None, br=0.5, noi=0.3, har=0.7, atk=20.0, dec=350.0, sub=0.0, cr=4.3)
    base.update(kw)
    return base


def test_flat_category_gets_no_filler_trait():
    cl = [_oneshot(), _oneshot(), _oneshot()]
    N.rank_traits(cl, N.ONESHOT_DIMS, "oneshot")
    assert all(c["traits"] == [] for c in cl)                # no more "neutral"


def test_one_sided_dimension_says_nothing_on_the_silent_side():
    cl = [dict(br=0.5, noi=0.3, har=0.1), dict(br=0.5, noi=0.3, har=0.9)]
    N.rank_traits(cl, N.LOOP_DIMS, "loop")
    assert cl[0]["traits"] == [] and cl[1]["traits"] == ["musical"]


def test_body_blocks_repeating_or_contradicting_dimensions():
    dims = N.naming_dims("KICKS", CATEGORIES["KICKS"])
    assert {"br", "sub"} <= N.blocked_dims(["bright", "thin", "kick"], dims)
    assert "atk" in N.blocked_dims(["soft", "round", "kick"], dims)
    assert "tune" in N.blocked_dims(["high", "tom"], N.naming_dims("TOMS", CATEGORIES["TOMS"]))
    assert "dec" in N.blocked_dims(["short", "crash"], N.naming_dims("CYMBALS", CATEGORIES["CYMBALS"]))


# --- phrase choice ---------------------------------------------------------------

def test_choose_phrases_strong_then_unique_fallback():
    phrases = ["a", "b", "c"]
    Zp = np.array([[1.5, 0.0, 0.0],     # strong a
                   [0.9, 0.2, 0.7],     # weak: a is taken, c (0.7) is free
                   [0.6, 0.1, 0.8],     # weak: c wanted but row 1... strongest first wins
                   [0.2, 0.3, 0.1]])    # nothing >= 0.5
    out = N.choose_phrases(Zp, phrases)
    assert out[0] == "a"
    assert out[2] == "c" and out[1] is None                  # c goes to the stronger (0.8)
    assert out[3] is None
    assert N.choose_phrases(Zp, phrases, fallback_z=None) == ["a", None, None, None]


def test_category_phrases_outrank_naming_only_phrases():
    # "808-snare" became "rave-snare" when genre words competed on equal terms
    phrases = ["808 snare", "rave snare"]
    Zp = np.array([[1.2, 2.5],      # both strong: the category's own phrase wins
                   [0.1, 1.4],      # only the genre phrase is strong: it names the family
                   [0.6, 0.9]])     # neither strong: weak fallback prefers the primary tier
    assert N.choose_phrases(Zp, phrases, n_primary=1) == ["808 snare", "rave snare", None]
    Zp2 = np.array([[1.2, 0.0], [0.1, 0.2], [0.6, 0.9]])
    assert N.choose_phrases(Zp2, phrases, n_primary=1) == ["808 snare", None, "rave snare"]


# --- assembly ----------------------------------------------------------------------

def test_traits_are_whole_units_never_split():
    # "high-tom-tuned": "high" is already in the body, so only "tuned" is added
    nm, _ = N.assemble_name([], ["high", "tom"], ["high-tuned", "snappy"], ntr=1)
    assert nm == "high-tom-snappy"
    # "cs2-noise-wash-pad-low": the cap cut "low-tuned" in half
    nm, _ = N.assemble_name(["cs2"], ["noise", "wash", "pad"], ["long-tail"], ntr=1)
    assert nm == "cs2-noise-wash-pad-long-tail"
    nm, _ = N.assemble_name(["gs1"], ["acid", "squelch", "bass", "x"], ["long-tail"], ntr=1)
    assert nm == "gs1-acid-squelch-bass-x"                   # full: no half trait


def test_tempo_range_lead_is_one_unit():
    # "110-120bpm-ghost-note-funk": the range counted as two words and cut the phrase
    nm, _ = N.assemble_name(["110-120bpm"], ["ghost", "note", "funk", "break"], ["bright"], ntr=1)
    assert nm == "110-120bpm-ghost-note-funk-break"


def test_name_length_limit_skips_long_traits():
    nm, _ = N.assemble_name([], ["x" * 30], ["long-tail", "dark"], ntr=1, max_chars=36)
    assert nm == "x" * 30 + "-dark"


def test_unique_name_is_case_insensitive_and_prefers_a_trait():
    used = {"kick-dark"}
    assert N.unique_name("Kick-Dark", {"kick", "dark"}, ["dark", "gritty"], used) == "Kick-Dark-gritty"
    assert N.unique_name("kick-dark", {"kick", "dark"}, ["dark"], used) == "kick-dark-2"


# --- end to end -----------------------------------------------------------------------

def _cluster(**kw):
    c = dict(idxs=[0], tune=None, bpm=None, bpm_range=None, free_tempo=False, source=None,
             br=0.5, noi=0.3, har=0.7, atk=20.0, dec=350.0, sub=0.0, cr=4.3)
    c.update(kw)
    return c


def _name(cat, clusters, Zp, phrases):
    cfg = CATEGORIES[cat]
    dims = N.naming_dims(cat, cfg)
    N.rank_traits(clusters, dims, cfg["kind"])
    return N.name_families(clusters, cat, cfg, np.array(Zp, float), phrases, dims=dims)


def test_siblings_sharing_a_body_are_told_apart_first():
    # three "reese" families told apart only by note and a category-relative trait read as
    # one; siblings lead with what separates them from each other
    cl = [_cluster(tune=37, br=0.3, atk=10.0), _cluster(tune=39, br=0.8, atk=10.0),
          _cluster(tune=40, br=0.5, atk=200.0, dec=2000.0)]
    names = _name("SUB", cl, [[2.0, 0], [2.0, 0], [0, 2.0]], ["reese bass", "fm bass"])
    # no note lead outside PIANO, no category noun
    assert names[0].startswith("reese-dark")
    assert names[1].startswith("reese-bright")
    assert not any(set(n.split("-")) & DRUM_WORDS for n in names)


def test_acoustic_band_leads_then_note_then_instrument():
    # ACOUSTIC: band word first (mallet / string / wind / plucked), then the note, then the instrument
    cl = [_cluster(tune=76, br=0.9, source="xylophone", band="mallet", band_type="acoustic"),
          _cluster(tune=48, br=0.1, source="cello", band="string", band_type="acoustic"),
          _cluster(tune=None, source="flute", band="wind", band_type="acoustic")]
    names = _name("ACOUSTIC", cl, [[0, 0], [0, 0], [0, 0]], ["church bell", "marimba"])
    assert names[0].startswith("mallet-e5-xylophone") and "tuned" not in names[0]
    assert names[1].startswith("string-c3-cello")
    assert names[2].startswith("wind-flute")


def test_body_never_gets_a_contradicting_trait():
    # "bright-thin-kick" measured sub-heavy; "bright-ride" measured low-tuned
    cl = [_cluster(br=0.1, sub=0.9, dec=100.0), _cluster(br=0.9, sub=0.0), _cluster()]
    names = _name("KICKS", cl, [[2.0, 0], [0, 0], [0, 0]], ["bright thin kick", "tight kick"])
    assert names[0].startswith("bright-thin") and "kick" not in names[0]
    assert not set(names[0].split("-")) & {"dark", "sub", "heavy"}


def test_stab_and_staccato_bodies_never_swell():
    cl = [_cluster(tune=60, atk=400.0, dec=3000.0), _cluster(tune=62, atk=5.0, dec=100.0),
          _cluster(tune=64)]
    names = _name("STABS", cl, [[2.0, 0], [0, 2.0], [0, 0]], ["staccato strings", "rave stab"])
    assert not {"swelling", "sustained"} & set(cl[0]["traits"]), names
    assert "swelling" not in names[0] and "sustained" not in names[0]
    assert cl[0]["clap_z"] == 2.0 and cl[2]["clap_z"] is None


def test_names_are_unique_within_a_category():
    cl = [_cluster(), _cluster(), _cluster()]
    names = _name("KICKS", cl, [[0], [0], [0]], ["x kick"])
    assert len({n.lower() for n in names}) == 3


def test_piano_chord_band_leads_with_chord():
    cl = [_cluster(tune=60, band="chord", band_type="chord", source=None),
          _cluster(tune=48, band="note", band_type="chord", source=None)]
    names = _name("PIANO", cl, [[2.0, 0], [0, 2.0]], ["rhodes chord stab", "grand piano note"])
    assert names[0].startswith("chord-rhodes") and names[1].startswith("c3-grand-piano-note")


def test_category_noun_dropped():
    from fourier.packs.naming import drop_noun
    assert drop_noun(["tight punchy kick"], "KICKS") == ["tight punchy"]
    assert drop_noun(["kick"], "KICKS") == []
    assert drop_noun(["open acoustic hi hat"], "HATS") == ["open acoustic"]
    assert drop_noun(["grand piano"], "PIANO") == ["grand piano"]      # piano vs organ


def test_traits_in_one_order_and_one_per_dimension():
    # two folders with the same two traits read the same way round
    cl = [_cluster(atk=300.0, dec=3000.0, br=0.2, med={"atk": 300.0, "dur": 3.0}),
          _cluster(atk=320.0, dec=3200.0, br=0.9, med={"atk": 320.0, "dur": 3.2}),
          _cluster(atk=5.0, dec=100.0, br=0.5, med={"atk": 5.0, "dur": 0.2})]
    names = _name("PADS", cl, [[0, 0], [0, 0], [0, 0]], ["warm pad", "glass pad"])
    for n in names:
        ws = n.split("-")
        assert not ({"dark", "bright"} <= set(ws)) and not ({"clean", "gritty"} <= set(ws))
    order = ["dark", "bright", "clean", "gritty", "plucky", "swelling", "short", "sustained"]
    for n in names:
        idx = [order.index(w) for w in n.split("-") if w in order]
        assert idx == sorted(idx), n


def test_subset_sibling_says_the_other_side():
    # "short-plucky" beside "short-plucky-dark": the brighter one says so
    cl = [_cluster(atk=5.0, dec=100.0, br=0.2, med={"atk": 5.0, "dur": 0.3}),
          _cluster(atk=6.0, dec=110.0, br=0.8, med={"atk": 6.0, "dur": 0.3}),
          _cluster(atk=400.0, dec=4000.0, br=0.5, med={"atk": 400.0, "dur": 4.0})]
    names = _name("SUB", cl, [[0, 0], [0, 0], [0, 0]], ["reese bass", "fm bass"])
    a, b = set(names[0].split("-")), set(names[1].split("-"))
    assert not a < b and not b < a, names


def test_trait_only_names_carry_no_category_noun():
    cl = [_cluster(br=0.1, sub=0.9), _cluster(br=0.9, sub=0.0), _cluster(br=0.5, noi=0.9)]
    names = _name("KICKS", cl, [[0, 0], [0, 0], [0, 0]], ["tight kick", "boomy kick"])
    assert names and not any("kick" in n.split("-") for n in names), names


def test_a_half_trait_ending_becomes_the_whole_trait():
    # "long hat" named an open-hat folder "open-long": half of "long-tail"
    dims = N.naming_dims("HATS", CATEGORIES["HATS"])
    assert N.whole_trait_ending("open-long", dims) == "open-long-tail"
    assert N.whole_trait_ending("open-long-tail", dims) == "open-long-tail"
    assert N.whole_trait_ending("open-dark", dims) == "open-dark"
    assert N.whole_trait_ending("open-tail-long", dims) == "open-long-tail"


def test_hats_named_by_a_long_phrase_pass_the_half_trait_rule():
    cl = [_cluster(band_type="hat", band="open", dec=900.0), _cluster(band_type="hat", band="open", dec=100.0)]
    names = _name("HATS", cl, [[2.0, 0], [0, 2.0]], ["long hat", "washy hat"])
    assert names[0].startswith("open-long-tail"), names
    assert not any(n.split("-")[-1] in {"long", "tail"} and not n.endswith("long-tail") for n in names)
