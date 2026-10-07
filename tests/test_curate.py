"""Unit tests for the taxonomy curation logic (fourier.packs.curate)."""
from types import SimpleNamespace

import numpy as np
import pytest

from fourier.packs import curate as C
from fourier.packs.curate_config import CATEGORIES
from fourier.packs import manifests


@pytest.mark.usefixtures("umbrella_vendor")
def test_pack_of_umbrella_and_plain():
    # an umbrella vendor's packs are nested one level deeper; the pack, minus the vendor's
    # suffix, is the pack name
    assert C._pack_of("Acme/Kit Pack 1 Acme/Kicks/Kick.wav", "x") == "Kit Pack 1"
    assert C._pack_of("Vendor B/kick.wav", "x") == "Vendor B"
    assert C._pack_of(None, "/lib/SampleLibrary/Acme/MPC60 Kit/Kick.wav") == "MPC60 Kit"
    assert C._is_nameable_vendor("Acme/909 Kit/Kick.wav", "x") is True
    assert C._is_nameable_vendor("Vendor B/Kick.wav", "x") is False


def test_pack_of_without_umbrella_vendors():
    # the code's defaults: no umbrella folder (no folder name is special), no vendor is nameable
    assert C._pack_of("Acme/909 Kit/Kicks/Kick.wav", "x") == "Acme"
    assert C._pack_of("Ableton/Kit Pack 1/Samples/Kick.wav", "x") == "Ableton"
    assert C._is_nameable_vendor("Acme/909 Kit/Kick.wav", "x") is False


def test_instrument_roots_are_umbrellas_and_instrument_pools(monkeypatch):
    """INSTRUMENT_ROOTS (a library overlay's): their packs sit one level down, and their files
    are in the instrument categories' pools; empty, no folder is (the SQL is false)."""
    import fourier.packs.curate_config as CC
    import fourier.packs.curate as CU
    assert CU._roots_sql() == "(1 = 0)" and CU._roots_params() == {}
    monkeypatch.setattr(CC, "INSTRUMENT_ROOTS", ("Collections",))
    monkeypatch.setattr(CU, "INSTRUMENT_ROOTS", ("Collections",))
    assert C._pack_of("Collections/Grand Piano/Samples/C3.wav", "x") == "Grand Piano"
    assert CU._roots_sql() == "(samples.rel_path LIKE :iroot0)"
    assert CU._roots_params() == {"iroot0": "Collections/%"}


def test_content_key_collapses_format_twins():
    # same pack + stem + duration => same key regardless of extension
    a = C._content_key("909", "BD_808.wav", 0.5031)
    b = C._content_key("909", "BD_808.aif", 0.5029)   # rounds to same 0.50
    assert a == b
    # trailing separators in the stem are stripped
    assert C._content_key("p", "kick_ .wav", 1.0)[1] == "kick"
    # different duration => different key
    assert C._content_key("909", "BD_808.wav", 1.5) != a


def test_cluster_partitions_all_and_respects_cap():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(120, 6))
    groups, k0 = C._cluster(P, kmax=5, kmin=5, kdiv=1_000_000)  # forces k0 == 5
    assert k0 == 5
    flat = sorted(i for g in groups for i in g)
    assert flat == list(range(120))          # every sample lands in exactly one family
    assert 1 <= len(groups) <= 5             # merge can only reduce the count


def test_rank_traits_picks_the_extreme_dimension():
    clusters = [
        dict(tune=None, br=0.9, noi=0.3, har=0.7, atk=20, dec=350, sub=0.0, cr=4.3),  # brightest
        dict(tune=None, br=0.5, noi=0.3, har=0.7, atk=20, dec=350, sub=0.0, cr=4.3),
        dict(tune=None, br=0.1, noi=0.3, har=0.7, atk=20, dec=350, sub=0.0, cr=4.3),  # darkest
    ]
    C._rank_traits(clusters, C.ONESHOT_DIMS, "oneshot")
    assert clusters[0]["traits"][0] == "bright"
    assert clusters[2]["traits"][0] == "dark"


def _row(**kw):
    base = dict(id=1, rel_path="Pack/x.wav", path="/lib/Pack/x.wav", filename="x.wav",
                file_hash=None, file_format="wav", duration_s=0.5, brightness=0.5,
                noisiness=0.3, harmonicity=0.7, crest_factor=4.3, base_note=None,
                base_note_confidence=None, attack_time_ms=20.0, decay_time_ms=350.0,
                sub_weight=0.0, is_clipped=0, tempo_bpm=None, bpm_reliable=0,
                onset_rate_hz=None, chroma_concentration=None,
                spectral_flatness_mean=0.5, rms_mean=0.2, dc_offset_ratio=0.0,
                ableton_tags=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_select_records_dedup_and_noise_filter():
    cfg = CATEGORIES["KICKS"]
    rows = [
        _row(id=1, filename="BD_808.wav", rel_path="P/BD_808.wav", path="/l/P/BD_808.wav", duration_s=0.50),
        _row(id=2, filename="BD_808.aif", rel_path="P/BD_808.aif", path="/l/P/BD_808.aif", duration_s=0.50),
        _row(id=3, filename="snare_01.wav", rel_path="P/snare_01.wav", path="/l/P/snare_01.wav"),
    ]
    emb_n = np.eye(4, dtype="float32")
    id2row = {1: 0, 2: 1, 3: 2}
    why = {}
    rec, fm, stats = C._select_records(rows, cfg, emb_n, id2row, category="KICKS", why=why)
    paths = [r["path"] for r in rec]
    assert len(rec) == 1                       # aif twin collapsed, snare out by KICKS' noise words
    assert paths[0].endswith("BD_808.wav")     # the real wav is the one kept
    assert stats["twins"] == 1
    assert why == {}                           # neither was a candidate the gates left out


def test_unknown_category_raises():
    with pytest.raises(Exception):
        C.build_taxonomy(session=None, category="NOPE", out_dir="/tmp/x")


def test_manifest_merge_keeps_other_categories(tmp_path):
    summ = dict(families=2, files=5, source_samples=40)
    C.update_build_manifest(tmp_path, "KICKS", summ, [{"family": "a", "out": "a/k.wav", "src": "/l/k.wav"}])
    C.merge_manifest(tmp_path, [("SNARES", dict(summ, chain_gated=3), [{"family": "b"}]),
                                ("HATS", None, None)])                    # a failed build: skipped
    doc = manifests.read(tmp_path / "manifest.json")
    assert set(doc["categories"]) == {"KICKS", "SNARES"} and doc["fourier_manifest"] == manifests.FORMAT == 3
    assert doc["categories"]["KICKS"]["entries"][0]["out"] == "a/k.wav"
    assert doc["categories"]["SNARES"]["chain_gated"] == 3 and "built" in doc["categories"]["SNARES"]
    assert doc.get("generated") and doc.get("code_hash")


def test_stereo_that_is_only_dc_comes_out_mono():
    """The mono call follows the DC pass: a DC difference between channels isn't width."""
    import numpy as np
    sr = 44100
    t = np.arange(sr // 4) / sr
    tone = 0.5 * np.sin(2 * np.pi * 220 * t) * np.exp(-t * 8)
    y = np.stack([tone + 0.08, tone - 0.08], axis=1)          # same sound, opposite DC
    out = C._process_audio(y, sr, "oneshot", True, True, "peak")
    assert out.ndim == 1
    wide = np.stack([tone, np.roll(tone, 300)], axis=1)       # real width stays stereo
    assert C._process_audio(wide, sr, "oneshot", True, True, "peak").ndim == 2


def _kicks(n, vendors, sims=None):
    """n KICKS-named rows from these vendors (cycled), and CLAP vectors: near-identical
    (cosine > NEAR_DUP_COS) unless sims gives each one's distance from the first."""
    rows = [_row(id=i + 1, filename=f"Kick {i + 1:02d}.wav", rel_path=f"{vendors[i % len(vendors)]}/Kicks/Kick {i + 1:02d}.wav",
                 path=f"/l/{vendors[i % len(vendors)]}/Kicks/Kick {i + 1:02d}.wav", duration_s=0.4,
                 file_hash=f"h{i}") for i in range(n)]
    E = np.zeros((n, 8), dtype="float32")
    E[:, 0] = 1.0
    for i in range(n):
        E[i, 1 + i % 7] = (sims or {}).get(i, 0.01 * (i + 1))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return rows, E, {i + 1: i for i in range(n)}


def test_a_small_pool_keeps_its_minimum():
    """One vendor's six near-identical kicks: no vendor cap (one vendor is the whole pool) and
    the near-duplicate prune stops at the minimum, so KICKS fills; without the minimum the
    prune leaves one (what used to empty a small library's categories)."""
    rows, E, id2row = _kicks(6, ["Acme"])
    rec, _fm, st = C._select_records(rows, CATEGORIES["KICKS"], E, id2row, category="KICKS", floor=6)
    assert len(rec) == 6 and st["vendor_capped"] == 0 and st["near_dup"] == 0
    rec, _fm, st = C._select_records(rows, CATEGORIES["KICKS"], E, id2row, category="KICKS")
    assert len(rec) == 1 and st["near_dup"] == 5


def test_the_minimum_brings_back_the_least_similar_first():
    rec = [dict(id=i, row=i, qual=1.0 - 0.01 * i) for i in range(4)]
    E = np.array([[1, 0, 0], [1, .02, 0], [1, 0, .15], [1, .05, 0]], dtype="float32")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    partners = {}
    out, pruned = C._prune_near_dups(rec, E, 0.95, keep=2, partners=partners)
    # every one is a near-duplicate of id 0; id 2 is the least like it, so it comes back
    assert [r["id"] for r in out] == [0, 2] and pruned == 2 and partners == {1: 0, 3: 0}
    out0, pruned0 = C._prune_near_dups(rec, E, 0.95)
    assert [r["id"] for r in out0] == [0] and pruned0 == 3


def test_the_minimum_changes_nothing_in_a_pool_that_stays_above_it():
    """The gate: with three or more vendors and a pool still at or above the minimum after the
    caps (every category of a build that fills), floor changes nothing."""
    rng = np.random.default_rng(3)
    for n, vendors in ((40, ["A", "B", "C"]), (60, ["A", "A", "A", "B", "C", "D"]), (25, list("ABCDE"))):
        rows, _E, id2row = _kicks(n, vendors)
        E = rng.standard_normal((n, 16)).astype("float32")
        E[: n // 3] = E[0] + 0.01 * rng.standard_normal((n // 3, 16)).astype("float32")   # a dup cluster
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        why = {}
        a = C._select_records(rows, CATEGORIES["KICKS"], E, id2row, category="KICKS", why=why)
        b = C._select_records(rows, CATEGORIES["KICKS"], E, id2row, category="KICKS", floor=6)
        assert len(a[0]) >= 6 and [r["id"] for r in a[0]] == [r["id"] for r in b[0]] and a[2] == b[2]
        # and what each step left out is recorded by id, for fourier why
        assert set(why.get("near_dup", {})) | set(why.get("vendor_capped", {})) == \
            {r.id for r in rows} - {r["id"] for r in a[0]}


def test_too_few_names_the_filters():
    doc = C.left_out_doc("KICKS", "too_few", 1, 6, dict(total=6, near_dup=5, vendor_capped=0, twins=0), {})
    assert C.too_few_line(doc) == "6 found; 5 near-duplicates, 0 over the per-vendor share; need 6"
    doc = C.left_out_doc("TOMS", "too_few", 0, 6, dict(total=0), {})
    assert C.too_few_line(doc) == "none found; need 6"


def test_an_additive_build_caps_as_before():
    """An additive build's picks join a release: its caps stay as they were (any vendor
    count, no minimum), so a later additive build adds what an earlier one would have."""
    rows, E, id2row = _kicks(6, ["Acme"], sims={i: 1.0 for i in range(6)})
    rec, _fm, st = C._select_records(rows, CATEGORIES["KICKS"], E, id2row, category="KICKS", floor=6,
                                     exclude_ids=set())
    assert len(rec) == 1 and st["vendor_capped"] == 5


def _loops(names, monkeypatch, scores=None, **kw):
    """DRUMLOOPS candidates (drum-tagged, 124 BPM by name, percussive) with stand-in CLAP
    vectors: a name with a break word passes the gate; the others sit far from the prompts
    (their `scores` order them)."""
    from fourier.demo import fake_embed_text, unit
    monkeypatch.setattr(C, "embed_text", fake_embed_text)
    rows, vecs = [], []
    for i, name in enumerate(names):
        rel = f"{'AB'[i % 2]}/Loops/{name} 124bpm.wav"
        fields = dict(id=i + 1, filename=f"{name} 124bpm.wav", rel_path=rel, path=f"/l/{rel}",
                      duration_s=7.74, harmonicity=0.3, ableton_tags=["Drum Loop"], n_events=1,
                      onset_rate_hz=4.1, bpm=None, file_hash=f"h{i}")
        rows.append(_row(**{**fields, **kw.get(name, {})}))
        v = fake_embed_text(rel) if "Break" in name else unit("file", rel)
        if scores and name in scores:      # a little of the prompts: a better score
            v = v + scores[name] * fake_embed_text("funk drum break")
        vecs.append(v / np.linalg.norm(v))
    return rows, np.array(vecs, dtype="float32"), {i + 1: i for i in range(len(names))}


def test_the_clap_gate_floor_readmits_the_best_scoring_loops(monkeypatch):
    """A category the CLAP gate alone would leave below its minimum takes back its best-
    scoring gated loops, up to the minimum and no more; each one's scores are recorded."""
    names = ["Break 1", "Break 2", "Floor 1", "Floor 2", "Floor 3", "Floor 4", "Floor 5"]
    rows, E, id2row = _loops(names, monkeypatch, scores={"Floor 2": 0.3, "Floor 4": 0.2,
                                                          "Floor 5": 0.1, "Floor 3": 0.05})
    cfg = CATEGORIES["DRUMLOOPS"]
    why, gate = {}, {}
    rec, _fm, st = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=6,
                                     why=why, gate=gate)
    got = {r["path"].rsplit("/", 1)[-1].split(" 124")[0] for r in rec}
    assert got == {"Break 1", "Break 2", "Floor 2", "Floor 4", "Floor 5", "Floor 3"}, got
    assert st["clap_readmitted"] == 4 and st["gated_out"] == 1
    assert set(why["gated_out"]) == {3}                         # Floor 1: still out, with its scores
    assert all(len(gate[i]) == 2 for i in (1, 2, 3)) and all(gate[i][2] == 1 for i in (4, 5, 6, 7))
    assert gate[1][0] >= 0.30 > gate[3][0]


def test_the_clap_gate_floor_changes_nothing_above_the_minimum(monkeypatch):
    """The gate: a category with enough loops past the CLAP gate (every category of a build
    that fills), an additive build, and a loop out for another reason (too tonal) as well."""
    cfg = CATEGORIES["DRUMLOOPS"]
    names = [f"Break {i}" for i in range(7)] + ["Floor 1", "Floor 2"]
    rows, E, id2row = _loops(names, monkeypatch)
    a = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=6)
    b = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS")
    assert [r["id"] for r in a[0]] == [r["id"] for r in b[0]] and len(a[0]) == 7
    assert a[2] == b[2] and "clap_readmitted" not in a[2]
    rows, E, id2row = _loops(["Break 1", "Floor 1", "Floor 2"], monkeypatch)
    rec, _fm, st = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=6,
                                     exclude_ids=set())
    assert len(rec) == 1 and "clap_readmitted" not in st      # an additive build: as before
    rows, E, id2row = _loops(["Break 1", "Floor 1", "Floor 2"], monkeypatch,
                             scores={"Floor 1": 0.2, "Floor 2": 0.2}, **{"Floor 2": dict(harmonicity=0.9)})
    rec, _fm, st = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=6)
    assert len(rec) == 2 and st["clap_readmitted"] == 1        # the tonal one stays out (two gates)


def test_the_floor_never_readmits_what_sounds_like_the_anti_prompts(monkeypatch):
    """A gated loop scoring below 0, or no more for the category's prompts than for its
    anti-prompts, never comes back, whatever the minimum (a drum loop's score for PHRASES)."""
    from fourier.demo import fake_embed_text
    cfg = CATEGORIES["DRUMLOOPS"]
    rows, E, id2row = _loops(["Break 1", "Floor 1", "Floor 2", "Floor 3"], monkeypatch,
                             scores={"Floor 3": 0.2})
    anti = np.mean([fake_embed_text(p) for p in cfg["anti"]], axis=0)
    E[1] = anti / np.linalg.norm(anti)                         # Floor 1: like the anti-prompts
    E[2] = -E[0]                                               # Floor 2: a score below 0
    why, gate = {}, {}
    rec, _fm, st = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=6,
                                     why=why, gate=gate)
    assert {r["id"] for r in rec} == {1, 4} and st["clap_readmitted"] == 1
    assert gate[2][0] <= gate[2][1] and gate[3][0] < 0 and set(why["gated_out"]) == {2, 3}
    assert not C._clap_sane(-0.09, -0.2) and not C._clap_sane(0.25, 0.3) and C._clap_sane(0.25, 0.1)


def test_the_floor_readmits_loops_only_their_harmonicity_keeps_out(monkeypatch):
    """A category its harmonicity gate would leave below its minimum takes back the loops that
    passed the CLAP gate and fail only that one, best CLAP margin first, up to the minimum;
    the build records the gate, each file's value and the threshold, and what came back."""
    from fourier.packs import why_log
    cfg = CATEGORIES["DRUMLOOPS"]
    names = [f"Break {i}" for i in range(1, 9)]
    tonal = {f"Break {i}": dict(harmonicity=0.7 + 0.01 * i) for i in range(2, 9)}
    rows, E, id2row = _loops(names, monkeypatch, **tonal)
    why, gate = {}, {}
    rec, _fm, st = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=6,
                                     why=why, gate=gate)
    assert len(rec) == 6 and st["too_harmonic"] == 2 and st["too_harmonic_readmitted"] == 5
    assert st["gated_out"] == 0 and "clap_readmitted" not in st
    out = why["too_harmonic"]
    assert len(out) == 2 and all(v[1] == cfg["har_max"] and v[0] > cfg["har_max"] for v in out.values())
    back = why[why_log.READMITTED]
    assert len(back) == 5 and all(v[0] == "too_harmonic" for v in back.values())
    doc = C.left_out_doc("DRUMLOOPS", "built", len(rec), 6, st, why)
    assert set(doc["readmitted"]) == {str(i) for i in back} and why_log.READMITTED not in doc["out"]
    sid = next(iter(back))
    assert why_log.readmitted_result(doc, sid).startswith(
        "readmitted past DRUMLOOPS's harmonicity gate (harmonicity 0.7")
    left = next(iter(out))
    assert why_log.describe("too_harmonic", out[left], doc) == (
        f"failed DRUMLOOPS's harmonicity gate (harmonicity {out[left][0]:.2f} > 0.68: too tonal)")
    # above the minimum, and in an additive build: as before
    a = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=1)
    b = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS")
    assert [r["id"] for r in a[0]] == [r["id"] for r in b[0]] == [1] and a[2] == b[2]
    rec, _fm, st = C._select_records(rows, cfg, E, id2row, category="DRUMLOOPS", floor=6,
                                     exclude_ids=set())
    assert len(rec) == 1 and "too_harmonic_readmitted" not in st


def test_too_few_counts_each_gate_apart():
    """A category left empty says how many each gate took: the CLAP gate and the harmonicity
    gate are counted apart."""
    doc = dict(category="DRUMLOOPS", found=5, need=6,
               counts=dict(total=5, gated_out=2, too_harmonic=3))
    line = C.too_few_line(doc)
    assert "2 failed the category's CLAP gate" in line and "3 too harmonic for DRUMLOOPS" in line


def test_a_loop_votes_drum_loops_only_with_drums():
    """Without Sononym, class.loop alone isn't a drum loop: a drum tag or word (in the file's
    own name or folder) or percussive audio is needed for DRUMLOOPS' vote."""
    cfg = CATEGORIES["DRUMLOOPS"]
    ev = lambda rel, tags=(), **kw: C._drum_loop_evidence(_row(rel_path=rel, **kw), list(tags), cfg)
    sustained = dict(harmonicity=0.9, onset_rate_hz=5.3)
    assert not ev("Northwind/Jungle Breaks 174/FX/Riser 04.wav", **sustained)
    assert not ev("Northwind/Jungle Breaks 174/Misc/Thing 1.wav", **sustained)
    assert ev("Northwind/Jungle Breaks 174/FX/Riser 04.wav", ["Drum Loop"], **sustained)
    assert ev("Acme/Loops/Beat 01.wav", **sustained) and ev("Acme/Perc Loops/Conga 2.wav", **sustained)
    assert ev("Acme/Misc/Thing 1.wav", harmonicity=0.3, onset_rate_hz=4.0)       # percussive audio
    assert not ev("Acme/Misc/Thing 1.wav", harmonicity=0.3, onset_rate_hz=0.5)
    assert not ev("Acme/Misc/Thing 1.wav", harmonicity=None)
    r = _row(rel_path="Acme/Misc/Thing 1.wav", **sustained)
    assert C._fallback_loop_votes({"DRUMLOOPS", "PADS"}, r, []) == {"PADS"}


def _phrase_row(rel, **kw):
    fields = dict(rel_path=rel, path=f"/l/{rel}", filename=rel.rsplit("/", 1)[-1], duration_s=8.0,
                  classes=["Loop"], categories=[], ableton_tags=[], bpm=None, tempo_bpm=None,
                  harmonicity=0.8, onset_rate_hz=4.0)
    fields.update(kw)
    return _row(**fields)


def test_without_sononym_the_phrase_rule_leaves_drum_loops_to_the_drum_loops():
    """Without Sononym, a loop the built-in providers call a drum loop (a drum word in its own
    name or folder or in its pack's name, or percussive audio) isn't a phrase; with Sononym
    (fallback False) the phrase rule is as it was. A loop whose own name or folder names an
    instrument stays a phrase, however percussive its audio."""
    pack_loop = _phrase_row("Vendor C/Drum Hits/Loops/WAV/House 120 Loop 01.wav")
    perc_loop = _phrase_row("Vendor C/Grooves/Loops/Shuffle 120 01.wav", harmonicity=0.3)
    synth = _phrase_row("Vendor C/Synth Loops/Loops/Lead 120 01.wav")
    bassline = _phrase_row("Vendor C/Bass Loops/Bassline 01 120bpm.wav", harmonicity=0.3)
    for r in (pack_loop, perc_loop, synth, bassline):
        assert C._is_phrase(r), r.rel_path                     # with Sononym: unchanged
    assert not C._is_phrase(pack_loop, True) and not C._is_phrase(perc_loop, True)
    assert C._is_phrase(synth, True) and C._is_phrase(bassline, True)
    # the pack's name counts for a loop by its own name or folder only (never a riser)
    assert C._pack_drum_named("Vendor C/Drum Hits/Loops/WAV/House 120 Loop 01.wav")
    assert not C._pack_drum_named("Vendor C/Drum Hits/FX/Riser 01.wav")
    assert not C._pack_drum_named("Vendor C/Drum & Bass Pack/Loops/Lead 174 01.wav")
    assert not C._pack_drum_named("Loops/Lead 120 01.wav")
    cfg = CATEGORIES["DRUMLOOPS"]
    assert C._drum_loop_evidence(pack_loop, [], cfg)        # and its DRUMLOOPS vote stands


def test_without_sononym_drum_loops_take_a_loop_its_providers_call_drums(monkeypatch):
    """DRUMLOOPS' drum-tag requirement: without Sononym, the drums its vote needed
    (_drum_loop_evidence) stand in for a tag; with Sononym a tag is needed, as before."""
    from fourier.demo import fake_embed_text
    monkeypatch.setattr(C, "embed_text", fake_embed_text)
    cfg = CATEGORIES["DRUMLOOPS"]
    rel = "Vendor C/Drum Hits/Loops/WAV/House 120 Loop 01.wav"
    r = _phrase_row(rel, id=1, duration_s=8.0, harmonicity=0.3, file_hash="h1",
                    categories=["Drum Loops"])           # (a classifier label: no phrase)
    E = (fake_embed_text("funk drum break amen break")[None, :]).astype("float32")
    E /= np.linalg.norm(E)
    rec, _fm, st = C._select_records([r], cfg, E, {1: 0}, category="DRUMLOOPS")
    assert not rec and st["nondrum_gated"] == 1
    rec, _fm, st = C._select_records([r], cfg, E, {1: 0}, category="DRUMLOOPS", fallback=True)
    assert [d["id"] for d in rec] == [1] and st["nondrum_gated"] == 0


def test_near_duplicate_partner_is_the_kept_file_it_repeats():
    """The prune names, for each file it drops, the kept file it repeats (the most similar
    one kept before it), never itself or another dropped file; the floor's readmissions have
    none."""
    E = np.array([[1, 0, 0], [0.999, 0.045, 0], [0, 1, 0], [0.02, 0.9998, 0], [0, 0, 1]], dtype="float32")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    rec = [dict(id=i + 1, row=i, qual=q, incumbent=False) for i, q in enumerate((0.9, 0.5, 0.4, 0.8, 0.7))]
    partners = {}
    kept, pruned = C._prune_near_dups(rec, E, 0.985, partners=partners)
    assert [d["id"] for d in kept] == [1, 4, 5] and pruned == 2
    assert partners == {2: 1, 3: 4}
    partners = {}
    kept, _ = C._prune_near_dups(rec, E, 0.985, keep=4, partners=partners)
    assert len(kept) == 4 and set(partners) | {d["id"] for d in kept} == {1, 2, 3, 4, 5}
    assert all(p in {d["id"] for d in kept} and p != s for s, p in partners.items())
