"""Unit tests for quality scoring, near-dup pruning, and pitch windows (pure, no DB)."""
import numpy as np

from fourier.packs.curate import (
    _quality, _prune_near_dups, _note_window, NOTE_WINDOW_DEFAULT,
    _loop_guard_mode, _is_mislabeled_loop, _has_drum_tag, _family_veto,
    _name_override, _override_subs_for, _name_override_rule, _pack_key2,
    _cap_group_spread, _filename_bpms, _bar_fit, _resolve_tempo, _tempo_band_groups,
    _with_pins, _restore_pins, _tempo_lead, _fold_tempo, _is_loop_row, _row_override,
    _sibling_key, _MisfiledIndex,
)


def test_quality_full_when_clean():
    assert _quality({"clip": 0, "dc": 0.0, "rms": 0.2}) == 1.0


def test_quality_handles_missing_signals():
    # no defect signals present -> full score (never crashes on absent keys/None)
    assert _quality({}) == 1.0
    assert _quality({"clip": 0, "dc": None, "rms": None}) == 1.0


def test_quality_penalizes_each_defect():
    clean = {"clip": 0, "dc": 0.0, "rms": 0.2}
    assert _quality({"clip": 1, "dc": 0.0, "rms": 0.2}) < _quality(clean)     # clipping
    assert _quality({"clip": 0, "dc": 0.1, "rms": 0.2}) < _quality(clean)     # DC offset
    assert _quality({"clip": 0, "dc": 0.0, "rms": 0.001}) < _quality(clean)   # near-silent


def test_quality_bounded_0_1():
    worst = {"clip": 1, "dc": 1.0, "rms": 0.0}
    assert 0.0 <= _quality(worst) <= 1.0


def test_prune_keeps_higher_quality_of_identical_pair():
    # rows 0 & 1 are identical embeddings; row 2 is orthogonal
    emb = np.array([[1., 0.], [1., 0.], [0., 1.]], dtype="float32")
    rec = [
        {"id": 1, "row": 0, "qual": 0.9},
        {"id": 2, "row": 1, "qual": 0.5},
        {"id": 3, "row": 2, "qual": 1.0},
    ]
    out, pruned = _prune_near_dups(rec, emb, 0.99)
    assert pruned == 1
    assert sorted(r["id"] for r in out) == [1, 3]   # dropped the lower-quality twin (id 2)


def test_prune_noop_when_distinct():
    emb = np.array([[1., 0.], [0., 1.]], dtype="float32")
    rec = [{"id": 1, "row": 0, "qual": 1.0}, {"id": 2, "row": 1, "qual": 1.0}]
    out, pruned = _prune_near_dups(rec, emb, 0.99)
    assert pruned == 0 and len(out) == 2


def test_prune_preserves_original_order():
    emb = np.array([[1., 0.], [0., 1.], [0.7071, 0.7071]], dtype="float32")
    rec = [{"id": 10, "row": 0, "qual": 1.0},
           {"id": 20, "row": 1, "qual": 1.0},
           {"id": 30, "row": 2, "qual": 1.0}]
    out, _ = _prune_near_dups(rec, emb, 0.99)
    assert [r["id"] for r in out] == [10, 20, 30]   # survivors keep input order


def test_prune_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, 8)).astype("float32")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    rec = [{"id": i, "row": i, "qual": float((i * 7 % 5) / 5)} for i in range(30)]
    a, pa = _prune_near_dups([dict(r) for r in rec], emb, 0.9)
    b, pb = _prune_near_dups([dict(r) for r in rec], emb, 0.9)
    assert pa == pb and [r["id"] for r in a] == [r["id"] for r in b]


def test_note_window_drums_ungated():
    # drum categories are not pitched -> no window (base_note is meaningless there)
    assert _note_window("KICKS", {"kind": "oneshot"}) == (None, None)
    assert _note_window("HATS", {"kind": "oneshot"}) == (None, None)


def test_note_window_pitched_defaults():
    assert _note_window("SYNTH", {"kind": "oneshot"}) == (24, 96)   # C1..C7, drops B7=107
    assert _note_window("SUB", {"kind": "oneshot"})[1] == 64        # bass ceiling
    assert _note_window("STABS", {"kind": "oneshot"}) == NOTE_WINDOW_DEFAULT


def test_note_window_instrument_kind_gated():
    lo, hi = _note_window("PIANO", {"kind": "instrument"})
    assert lo is not None and hi is not None


def test_note_window_cfg_override_wins():
    assert _note_window("SYNTH", {"kind": "oneshot", "note_min": 30, "note_max": 90}) == (30, 90)
    # an override on a normally-ungated category still applies
    assert _note_window("KICKS", {"kind": "oneshot", "note_max": 84}) == (None, 84)


# --- mislabeled-loop guard --------------------------------------------------

def test_loop_guard_mode_by_category():
    assert _loop_guard_mode("KICKS", {"kind": "oneshot"}) == "tag_or_onset"
    assert _loop_guard_mode("HATS", {"kind": "oneshot"}) == "tag_or_onset"
    assert _loop_guard_mode("PERC", {"kind": "oneshot"}) == "tag_and_onset"
    assert _loop_guard_mode("CYMBALS", {"kind": "oneshot"}) == "tag_and_onset"
    assert _loop_guard_mode("SYNTH", {"kind": "oneshot"}) is None   # tonal: never guarded
    assert _loop_guard_mode("DRUMLOOPS", {"kind": "loop"}) is None


def test_loop_guard_mode_cfg_override():
    assert _loop_guard_mode("KICKS", {"kind": "oneshot", "loop_guard": False}) is None
    assert _loop_guard_mode("PERC", {"kind": "oneshot", "loop_guard": "tag_or_onset"}) == "tag_or_onset"


def test_loop_guard_tight_tag_or_onset():
    m = "tag_or_onset"
    assert _is_mislabeled_loop(m, ["One Shot", "Kick"], 1, False) is False   # clean single hit
    assert _is_mislabeled_loop(m, ["Loop"], 1, False) is True                # Ableton loop tag
    assert _is_mislabeled_loop(m, ["Drum Loop"], 30, True) is True
    assert _is_mislabeled_loop(m, ["One Shot"], 12, True) is True            # onset + tempo
    assert _is_mislabeled_loop(m, ["One Shot"], 12, False) is False          # onsets but no tempo (reverb tail)


def test_loop_guard_handperc_requires_both():
    m = "tag_and_onset"
    # a conga articulation Ableton mis-tagged Loop but only a few onsets: KEEP
    assert _is_mislabeled_loop(m, ["Loop"], 3, False) is False
    # a real conga loop: loop tag AND many onsets -> drop
    assert _is_mislabeled_loop(m, ["Loop"], 22, False) is True
    # many onsets but no loop tag (a busy tom fill): KEEP under hand-perc rule
    assert _is_mislabeled_loop(m, ["One Shot"], 20, True) is False


def test_loop_guard_mode_none_keeps_everything():
    assert _is_mislabeled_loop(None, ["Loop"], 99, True) is False


def test_has_drum_tag():
    assert _has_drum_tag(["Drum Loop"]) is True
    assert _has_drum_tag(["Loop", "Closed Hihat"]) is True          # drum-instrument tag
    assert _has_drum_tag(["Loop", "Sound FX", "Sweep"]) is False    # an FX sweep loop
    assert _has_drum_tag(["Loop", "Synth Bass"]) is False           # bassline loop
    assert _has_drum_tag(["Lead", "Loop"]) is False                 # synth lead loop
    assert _has_drum_tag([]) is False


# --- cross-family home veto -------------------------------------------------

def test_family_veto_bass_out_of_kicks():
    # Sononym says KICKS, Ableton says Synth Bass -> KICKS (drum) vetoed, SUB kept
    assert _family_veto({"KICKS", "SUB"}, ["Synth Bass"]) == {"SUB"}


def test_family_veto_fx_out_of_drums_and_tonal():
    assert _family_veto({"SNARES", "FX"}, ["Sound FX"]) == {"FX"}     # FX exempt, drum vetoed
    assert _family_veto({"SYNTH", "FX"}, ["Sound FX"]) == {"FX"}      # synth-FX -> FX
    assert _family_veto({"HATS"}, ["Solo Voice"]) == {"HATS"}         # veto empties -> keep original


def test_family_veto_vocal_to_vox():
    assert _family_veto({"HATS", "VOX"}, ["Solo Voice"]) == {"VOX"}   # a vocal in a hat pool -> VOX


def test_family_veto_guards_stabs_blips():
    # STABS/BLIPS are now guarded: drum & vocal contamination is evicted
    assert _family_veto({"STABS", "CYMBALS"}, ["Crash"]) == {"CYMBALS"}
    assert _family_veto({"BLIPS", "VOX"}, ["Solo Voice"]) == {"VOX"}
    assert _family_veto({"STABS", "SYNTH"}, ["Lead"]) == {"STABS", "SYNTH"}   # tonal stab stays


def test_family_veto_keeps_when_family_matches():
    assert _family_veto({"KICKS"}, ["Kick"]) == {"KICKS"}             # drum tag, drum cat: fine
    assert _family_veto({"PERC"}, ["Wood"]) == {"PERC"}               # wood is a drum-family tag


def test_family_veto_exempts_coarse_categories():
    # PIANO/WAVES/ACOUSTIC aren't in CATEGORY_AB_FAMILY, so Ableton's coarse tag
    # (Solo Voice / Sound FX) never vetoes their finer Fourier home
    assert _family_veto({"PIANO"}, ["Solo Voice"]) == {"PIANO"}
    assert _family_veto({"WAVES"}, ["Sound FX"]) == {"WAVES"}
    assert _family_veto({"ACOUSTIC"}, ["Field & Foley"]) == {"ACOUSTIC"}


def test_family_veto_no_ableton_tags_noop():
    assert _family_veto({"KICKS", "SUB"}, []) == {"KICKS", "SUB"}


def test_name_override_routes_mis_slotted_instruments():
    assert _name_override("Music Box C4.wav") == "ACOUSTIC"
    assert _name_override("Kalimba C3.wav") == "ACOUSTIC"
    assert _name_override("Thumb Piano E3.wav") == "ACOUSTIC"
    assert _name_override("Church Bell.wav") == "ACOUSTIC"
    assert _name_override("Bell Pluck C3 07.wav") is None     # a synth bell: homes normally
    assert _name_override("Cello Harmonics 2.wav") == "ACOUSTIC"
    assert _name_override("Funk Keys The Clav A-2.wav") == "PIANO"
    assert _name_override("Ah Vox 2.wav") == "VOX"


def test_name_override_exempts_synth_versions():
    # a 'Synth <x>' keeps its synth home, not the acoustic instrument's
    assert _name_override("Synth Kalimba 7.wav") is None
    assert _name_override("Synth Bell Pad.wav") is None


def test_name_override_leaves_normal_files_alone():
    assert _name_override("Kick 12.wav") is None
    assert _name_override("Grand Piano C3.wav") is None
    assert _name_override("") is None
    assert _name_override(None) is None


def test_override_subs_for_by_category():
    assert set(_override_subs_for("ACOUSTIC")) >= {"music box", "kalimba", "thumb piano"}
    assert _override_subs_for("VOX") == ["ah vox"]
    assert _override_subs_for("KICKS") == []
    assert _override_subs_for(None) == []


def test_name_override_rule_groups_synonyms():
    i1, c1 = _name_override_rule("Clavinet C3.wav")
    i2, c2 = _name_override_rule("Funk Keys The Clav A-2.wav")
    assert c1 == c2 == "PIANO" and i1 == i2          # one rule, one shared quota
    assert _name_override_rule("Kick 01.wav") == (None, None)
    assert _name_override_rule("Synth Kalimba.wav") == (None, None)


def test_pack_key2_splits_vendor_packs():
    assert _pack_key2("/x/SampleLibrary/Vendor A/Keys Pack/a.wav") == "Vendor A/Keys Pack"
    assert _pack_key2("/x/SampleLibrary/Vendor A/Piano Pack/b.wav") == "Vendor A/Piano Pack"
    assert _pack_key2(None) == "?"


def test_cap_group_spread_caps_preserves_order_deterministic():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(30, 8)).astype("float32")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    rec = [{"row": i, "g": ("a" if i < 20 else None)} for i in range(30)]
    out = _cap_group_spread(rec, E, lambda d: d["g"], 5)
    assert sum(1 for d in out if d["g"] == "a") == 5
    assert sum(1 for d in out if d["g"] is None) == 10          # None is never capped
    rows = [d["row"] for d in out]
    assert rows == sorted(rows)                                # input order preserved
    assert out == _cap_group_spread(rec, E, lambda d: d["g"], 5)


def test_cap_group_spread_noop_under_cap():
    E = np.eye(4, dtype="float32")
    rec = [{"row": i, "g": "a"} for i in range(4)]
    assert _cap_group_spread(rec, E, lambda d: d["g"], 10) == rec


def test_filename_bpm_forms():
    assert _filename_bpms("Drum Loop 75bpm.wav") == [75.0]
    assert _filename_bpms("Beat_100_4B_Dry.wav") == [100.0]
    assert _filename_bpms("Groove(100).WAV") == [100.0]
    assert _filename_bpms("Break_175_Beat_009.wav") == [175.0]
    assert _filename_bpms("Kick 808 Short 03.wav") == []       # 808 out of range, 03 too small
    assert _filename_bpms(None) == []


def test_bar_fit():
    assert _bar_fit(3.2, 75.0)            # 1 bar at 75
    assert _bar_fit(9.6, 100.0)           # 4 bars at 100
    assert not _bar_fit(3.2, 99.4)        # the detector's wrong tempo
    assert not _bar_fit(None, 120.0)


def test_resolve_tempo_priority():
    # filename confirmed by length wins over a wrong detector
    assert _resolve_tempo("Drum Loop 75bpm.wav", 3.2, 99.4, 99.4) == (75.0, "name")
    # filename number that the length contradicts is ignored (kit number, not tempo)
    assert _resolve_tempo("Kit_150_Loop.wav", 4.0, 120.0, 90.0) == (120.0, "sononym")
    # sononym octave error corrected by x2 (half a bar at 60 -> one bar at 120)
    assert _resolve_tempo("loop.wav", 2.0, 60.0, None) == (120.0, "sononym")
    # librosa as last resort
    assert _resolve_tempo("loop.wav", 4.0, 97.0, 120.0) == (120.0, "librosa")
    # nothing confirms: unknown, not a guess
    assert _resolve_tempo("loop.wav", 3.7, 97.0, 101.0) == (None, None)


def test_tempo_band_groups_order_merge_unknown():
    edges = (60, 80, 90, 120, 201)
    bpms = [85.0] * 5 + [70.0] * 1 + [100.0] * 5 + [None] * 2 + [150.0] * 5
    g = _tempo_band_groups(bpms, edges, 3)
    # the lone 70 merges into 80-90; bands stay in tempo order; unknown last
    assert [sorted(bpms[i] for i in grp if bpms[i] is not None) for grp in g[:-1]] == \
        [[70.0] + [85.0] * 5, [100.0] * 5, [150.0] * 5]
    assert [bpms[i] for i in g[-1]] == [None, None]


def test_with_pins_always_selects_pinned():
    pick = lambda rest, n: rest[:n]
    assert _with_pins([1, 2, 3, 4, 5], 3, {4}, pick) == [4, 1, 2]
    assert _with_pins([1, 2, 3], 1, {2, 3}, pick) == [2, 3]        # pins exceed want: all kept
    assert _with_pins([1, 2], 2, {1, 2}, pick) == [1, 2]


def test_restore_pins_readds_removed():
    a, b, c = {"id": 1}, {"id": 2, "pin": True}, {"id": 3, "pin": True}
    assert _restore_pins([a, c], [b, c]) == [a, c, b]
    assert _restore_pins([a, b, c], [b, c]) == [a, b, c]


def test_tempo_lead_range_when_wide():
    # ranges widen to multiples of 5 so sibling folders read consistently
    assert _tempo_lead({"bpm": 162.0, "bpm_range": [150, 162]}) == "150-165bpm"
    assert _tempo_lead({"bpm": 115.0, "bpm_range": [110, 119]}) == "110-120bpm"
    assert _tempo_lead({"bpm": 115.0, "bpm_range": [110, 120]}) == "110-120bpm"
    assert _tempo_lead({"bpm": 150.0, "bpm_range": [141, 158]}) == "140-160bpm"
    assert _tempo_lead({"bpm": 127.4, "bpm_range": [125, 128]}) == "125-130bpm"
    assert _tempo_lead({"bpm": 127.4, "bpm_range": [126, 128]}) == "127bpm"
    assert _tempo_lead({"bpm": 86.0, "bpm_range": None}) == "086bpm"


def test_fold_tempo_one_octave():
    w = (90.0, 180.0)
    assert _fold_tempo(82.0, w) == 164.0      # half-time amen -> jungle tempo
    assert _fold_tempo(86.0, w) == 172.0      # half-time dnb -> dnb
    assert _fold_tempo(172.0, w) == 172.0
    assert _fold_tempo(124.0, w) == 124.0
    assert _fold_tempo(60.0, w) == 120.0
    assert _fold_tempo(190.0, w) == 95.0
    assert _fold_tempo(None, w) is None
    assert _fold_tempo(82.0, None) == 82.0    # no window: unchanged


def test_overrides_skip_loops():
    from types import SimpleNamespace as NS
    loop = NS(filename="Kalimba Loop 120.wav", classes=["Loop"],
              ableton_tags=["Drum Loop", "Loop"])
    loop_ab_only = NS(filename="Kalimba Groove.wav", classes=[], ableton_tags=["Loop"])
    note = NS(filename="Kalimba C3.wav", classes=["OneShot"], ableton_tags=["One Shot"])
    assert _is_loop_row(loop) and _is_loop_row(loop_ab_only) and not _is_loop_row(note)
    assert _row_override(loop) == (None, None)
    assert _row_override(loop_ab_only) == (None, None)
    assert _row_override(note)[1] == "ACOUSTIC"


def test_sibling_key_groups_multisample_sets():
    k = _sibling_key
    assert k("/p/Chord Stab C 04.wav") == k("/p/Chord Stab C 06.wav") is not None
    assert k("/p/Toy Piano A#3 v1.wav") == k("/p/Toy Piano D#4 v2.wav")
    assert k("/p/Tom Short D 15.wav") == k("/p/Tom Short D 17.wav")
    assert k("/p/Tom Long A 12.wav") != k("/p/Tom Short D 15.wav")
    assert k("/a/Piano C3.wav") != k("/b/Piano C3.wav")          # different folders
    assert k("/p/Kick 01.wav") is None                           # too generic to be a set
    assert k("/p/Piano C3.wav") == k("/p/Piano G4.wav")          # note names make it a set
    # a MIDI note glued to the name, and a trailing ID token after the velocity
    assert k("/p/112Pad_Chord_Warm_E7.wav") == k("/p/57Pad_Chord_Warm_A2.wav")
    assert k("/p/Pad-D#3-127-C1ZQ.wav") == k("/p/Pad-E2-127-C2ZQ.wav")
    assert k("/p/Organ-G#1-127-C3ZQ.wav") != k("/p/Pad-E2-127-C2ZQ.wav")
    assert k("/p/Sub Bass 1234.wav") == k("/p/Sub Bass 1234.wav")


def test_misfiled_index_expands_to_siblings():
    near, far = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    idx = _MisfiledIndex({"/p/Chord Stab C 04.wav": "FX"}, lambda s: near)
    assert idx.category("/p/Chord Stab C 04.wav") == "FX"
    assert idx.category("/p/Chord Stab C 06.wav", near) == "FX"
    assert idx.category("/p/Chord Stab C 06.wav", far) is None      # sounds unrelated
    assert idx.category("/p/Other 06.wav", near) is None


def test_sibling_key_spans_pack_twin_folders():
    st = "/x/SampleLibrary/Vendor A/Synth Pack 2/Synth Pack 2 (WAV stereo)/Chord Stab C 04.wav"
    mo = "/x/SampleLibrary/Vendor A/Synth Pack 2/Synth Pack 2 (WAV mono)/Chord Stab C 04.wav"
    other = "/x/SampleLibrary/Vendor A/Synth Pack 3/Chord Stab C 04.wav"
    assert _sibling_key(st) == _sibling_key(mo) is not None
    assert _sibling_key(st) != _sibling_key(other)             # different pack


def test_filename_tempo_wins():
    from fourier.packs.curate import _resolve_tempo
    # explicit bpm: as written, never snapped to the length (a reverb tail)
    assert _resolve_tempo("Shuffle Drum Loop 92 bpm.wav", 10.66, 90.0, None) == (92.0, "name")
    # explicit bpm even when the length doesn't fit bars
    assert _resolve_tempo("Piano Chords Emin 155 bpm.wav", 7.0, 99.0, None) == (155.0, "name")
    # a bare number counts when it gives whole bars, any count (6 bars at 120 = 12 s)
    assert _resolve_tempo("Seq_120_6x4_Loop.wav", 12.0, None, 160.0) == (120.0, "name")
    # a bare number that doesn't fit is a take number: detection decides
    assert _resolve_tempo("Funk Loop 072.wav", 10.0, 96.0, None)[1] == "sononym"


def test_level_folder_pulls_outliers_to_the_edge(tmp_path):
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import _level_folder, _short_term_db
    sr = 8000
    fam = tmp_path / "f"
    fam.mkdir()
    ents = []
    for k, amp in enumerate([0.5, 0.5, 0.5, 0.05, 0.5]):
        y = np.sin(np.arange(sr) * 0.05) * amp
        sf.write(str(fam / f"{k}.wav"), y, sr, subtype="PCM_16")
        ents.append(dict(family="f", out=f"f/{k}.wav"))
    n = _level_folder(tmp_path, ents)
    # 20 dB under the median: lifted, but only by LEVEL_MAX_UP_DB
    assert n == 1 and ents[3]["level_db"] == 6.0
    y3, _ = sf.read(str(fam / "3.wav"))
    y0, _ = sf.read(str(fam / "0.wav"))
    assert abs((_short_term_db(y0, sr) - _short_term_db(y3, sr)) - 14.0) < 0.3
    # a file 5 dB under comes to the 3 dB edge
    ents2 = []
    for k, amp in enumerate([0.5, 0.5, 0.5, 0.5 * 10 ** (-5 / 20)]):
        sf.write(str(fam / f"b{k}.wav"), np.sin(np.arange(sr) * 0.05) * amp, sr, subtype="PCM_16")
        ents2.append(dict(family="f", out=f"f/b{k}.wav"))
    assert _level_folder(tmp_path, ents2) == 1 and abs(ents2[3]["level_db"] - 2.0) < 0.2


def test_grid_offset_and_rotation(tmp_path):
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import _grid_offset, _rotate_and_slice
    sr, bpm = 44100, 120
    six = 60 / bpm / 4
    n = int(32 * six * sr)
    y = np.zeros(n)
    for k in range(32):
        i = int((k * six + 0.025) * sr) % n                 # every hit 25 ms late
        m = min(n - i, 1500)
        y[i:i + m] += np.exp(-np.arange(m) / 150.0) * np.sin(np.arange(m) * 0.4)
    assert abs(_grid_offset(y, sr, bpm) - 0.025) < 0.004
    p = tmp_path / "loop.wav"
    sf.write(str(p), y * 0.5, sr, subtype="PCM_16")
    out = _rotate_and_slice(p, bpm, "/lib/SampleLibrary/V/Break 120.wav")
    assert 20 < out["rotate_ms"] < 30 and sf.info(str(p)).frames == n
    assert abs(_grid_offset(sf.read(str(p))[0], sr, bpm)) == 0.0
    # a fill is left alone
    q = tmp_path / "fill.wav"
    sf.write(str(q), y * 0.5, sr, subtype="PCM_16")
    assert "rotate_ms" not in _rotate_and_slice(q, bpm, "/lib/SampleLibrary/V/Snare Fill 120.wav")


def test_thin_tempo_band_joins_the_nearer_tempo():
    bpms = [127.0] * 10 + [140.0] * 2 + [170.0] * 5
    g = _tempo_band_groups(bpms, (125, 130, 150, 201), 3)
    assert sorted(bpms[i] for i in g[0]) == [127.0] * 10 + [140.0] * 2


def _hits(n_six, six, sr, shift_of):
    import numpy as np
    n = int(n_six * six * sr)
    y = np.zeros(n)
    for k in range(n_six):
        i = int((k * six + shift_of(k)) * sr) % n
        m = min(n - i, 1500)
        y[i:i + m] += np.exp(-np.arange(m) / 150.0) * np.sin(np.arange(m) * 0.4)
    return y


def test_early_hits_rotate_onto_the_grid(tmp_path):
    """A loop cut late (every hit 25 ms early) rotates the other way and lands on the grid."""
    import soundfile as sf
    from fourier.packs.curate import _grid_offset, _rotate_and_slice
    sr, bpm = 44100, 120
    six = 60 / bpm / 4
    y = _hits(32, six, sr, lambda k: -0.025)
    assert abs(_grid_offset(y, sr, bpm) + 0.025) < 0.004
    p = tmp_path / "early.wav"
    sf.write(str(p), y * 0.5, sr, subtype="PCM_16")
    out = _rotate_and_slice(p, bpm, "/lib/SampleLibrary/V/Break 120.wav")
    assert -30 < out["rotate_ms"] < -20
    assert _grid_offset(sf.read(str(p))[0], sr, bpm) == 0.0


def test_pushed_groove_is_not_rotated():
    """A break that opens on its downbeat with some hits pushed ahead of the beat is feel,
    not a bad cut: it stays as it is (rotating it would move the downbeat late)."""
    from fourier.packs.curate import _grid_offset
    sr, bpm = 44100, 167
    six = 60 / bpm / 4
    y = _hits(32, six, sr, lambda k: 0.0 if k % 8 == 0 else -0.026)   # 3 in 4 beats pushed
    assert _grid_offset(y, sr, bpm) == 0.0


def test_late_start_rotates_to_the_downbeat_not_past_it(tmp_path):
    """A loop whose downbeat comes 15 ms in, with the later hits sitting 22 ms late, rotates
    to the downbeat (about 14 ms), not by the hits' median, which would cut the front of the
    downbeat off and move it to the end."""
    import soundfile as sf
    from fourier.packs.curate import _grid_offset, _rotate_and_slice
    sr, bpm = 44100, 120
    six = 60 / bpm / 4
    import numpy as np
    y = _hits(32, six, sr, lambda k: 0.015 if k == 0 else 0.022)
    y[:int(0.014 * sr)] += np.random.default_rng(0).normal(0, 0.01, int(0.014 * sr))  # room tone
    off = _grid_offset(y, sr, bpm)
    assert 0.0135 <= off <= 0.0145
    p = tmp_path / "late.wav"
    sf.write(str(p), y * 0.5, sr, subtype="PCM_16")
    out = _rotate_and_slice(p, bpm, "/lib/SampleLibrary/V/Break 120.wav")
    assert 13.5 <= out["rotate_ms"] <= 14.5
    y2 = sf.read(str(p))[0]
    lead = int(0.0005 * sr)                                   # the attack starts inside 1.5 ms
    assert abs(y2[:lead]).max() < 0.03 and abs(y2[lead:int(0.003 * sr)]).max() > 0.1


def test_silent_lead_in_rotates_past_the_sixteenth_cap():
    """30 ms of silence before the downbeat at 175 BPM is more than 0.3 of a 16th, but a
    silent lead-in is a late cut, not an early hit: it rotates to the downbeat."""
    import numpy as np
    from fourier.packs.curate import _grid_offset
    sr, bpm = 44100, 175
    six = 60 / bpm / 4
    y = _hits(32, six, sr, lambda k: 0.030)
    y[-int(0.030 * sr):] = 0.0                       # the last hit's tail isn't wrapped in
    off = _grid_offset(y, sr, bpm)
    assert 0.028 <= off <= 0.030
    # the same shift under a ringing tail is left alone: no silent lead-in, and over the cap
    z = y + 0.2 * np.sin(np.arange(len(y)) * 0.05)
    assert _grid_offset(z, sr, bpm) == 0.0


def test_loops_are_counted_at_their_folder_tempo(tmp_path):
    """A break Sononym read at half time is filed at the folded tempo; its bars are counted
    there too (2 bars at 160, not 1 at 80)."""
    import soundfile as sf
    from fourier.packs.curate import _loop_tempo, _slice_fields
    sr, bpm = 44100, 160.0
    six = 60 / bpm / 4
    p = tmp_path / "Break Loop.wav"
    sf.write(str(p), _hits(32, six, sr, lambda k: 0.0) * 0.5, sr, subtype="PCM_16")
    r = {"bpm": 80.0, "bpm_fold": 160.0}
    assert _loop_tempo(r) == 160.0 and _loop_tempo({"bpm": 120.0}) == 120.0
    assert _slice_fields(p, _loop_tempo(r))["bars"] == 2.0
    assert _slice_fields(p, r["bpm"])["bars"] == 1.0          # what the half-time count said


def test_named_note_contradicted_by_the_audio_keeps_pitch_and_name(tmp_path):
    """A file named D#4 that plays A4 isn't retuned by
    its name; one that plays its named note is; a chord name isn't checked."""
    import os
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import _check_name_pitches, _name_root_pc, _pitch_conflicts, _retune_shift
    sr = 44100
    t = np.arange(int(1.5 * sr)) / sr

    def tone(name, hz):
        p = tmp_path / name
        y = 0.5 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(4 * np.pi * hz * t)
        sf.write(str(p), y * np.minimum(1, t / 0.01), sr, subtype="PCM_16")
        return str(p)

    def rec(path):
        n = os.path.basename(path)
        rt, src = _retune_shift("SYNTH", n, None)
        return dict(path=path, name_pc=_name_root_pc(n), retune=rt, root_src=src)

    wrong, right = tone("Saw D#4.wav", 440.0), tone("Lead D#4.wav", 311.13)
    chord = tone("Stab Am.wav", 261.63)
    rs = [rec(wrong), rec(right), rec(chord)]
    assert all(r["retune"] is not None for r in rs)
    assert _check_name_pitches([0, 1, 2], rs) == 1
    assert rs[0]["retune"] is None and abs(rs[0]["pitch_conflict"] - 69) < 0.3
    assert rs[1]["retune"] == -3.0 and rs[2]["retune"] is not None
    assert not _pitch_conflicts(0, 55.0) and not _pitch_conflicts(0, 43.0) and _pitch_conflicts(0, 45.0)


def test_hard_start_faded_in_and_loop_point_joins_through_zero():
    """A hit that starts at full level is faded in over half a millisecond; a loop fades both
    ends, so it neither clicks when triggered nor at its loop point."""
    import numpy as np
    from fourier.packs.curate import _edge_fades, _process_audio
    sr = 44100
    t = np.arange(sr // 2) / sr
    hit = 0.9 * np.cos(2 * np.pi * 50 * t) * np.exp(-8 * t)            # sample 0 at 0.9
    out = _process_audio(hit, sr, "oneshot", True, True, "peak")
    assert abs(out[0]) < 0.01 and float(np.abs(out[:200]).max()) > 0.5
    loop = 0.5 * np.sin(2 * np.pi * 3 * t + 1.0)                          # both ends hot
    lo = _process_audio(loop, sr, "loop", False, False, "rms")
    assert len(lo) == len(loop) and abs(lo[0]) < 0.01 and abs(lo[-1]) < 0.05
    quiet = np.r_[np.zeros(100), hit]
    assert _edge_fades(quiet, sr) is quiet                                # a quiet start is left alone


def test_long_quiet_tail_trimmed_at_minus_50_of_peak():
    """A decay's last stretch under -50 dB of the peak is cut (a long decay would keep
    seconds of inaudible tail)."""
    import numpy as np
    from fourier.packs.curate import _process_audio
    sr = 22050
    t = np.arange(sr * 6) / sr
    y = 0.8 * np.sin(2 * np.pi * 220 * t) * np.exp(-2.0 * t)             # -50 dB at ~2.9 s
    out = _process_audio(y, sr, "instrument", True, True, "peak")
    assert 2.6 < len(out) / sr < 3.2


def test_band_overflow_follows_the_previous_build():
    """Overflow slots on a tie between two bands' folders (a wind and a plucked one): a one-file
    change in one pool (which near-dup the previous build kept) would tip it and swap a file
    every build. With the previous build's counts the split holds either way."""
    from fourier.packs.curate import _allocate_banded
    from fourier.packs.curate_config import CATEGORIES
    sh = CATEGORIES["ACOUSTIC"]["band_share"]
    keys = ["string", "string", "wind", "wind", "plucked", "plucked", "mallet", "mallet"]
    s93, s94 = [45, 43, 70, 82, 68, 93, 80, 72], [45, 43, 70, 82, 68, 94, 80, 72]
    a, b = [45, 43, 34, 46, 37, 57, 46, 42], [45, 43, 33, 46, 37, 58, 46, 42]
    assert _allocate_banded(s93, keys, sh, 350, 20, 120) != _allocate_banded(s94, keys, sh, 350, 20, 120)
    for prev in (a, b):
        for s in (s93, s94):
            assert _allocate_banded(s, keys, sh, 350, 20, 120, prefer=prev) == prev


def test_peaky_loop_limited_up_to_3db_toward_minus_20():
    """A loop whose one hot transient stops it short of -20 dBFS RMS gets up to 3 dB of that
    peak limited; the peak stays at the -1 dBFS ceiling."""
    import numpy as np
    from fourier.packs.curate import _normalize, _rms_db
    sr = 44100
    t = np.arange(sr * 2) / sr
    y = 0.08 * np.sin(2 * np.pi * 110 * t)
    y[sr // 2:sr // 2 + 200] += 0.9 * np.hanning(200)                   # one spike
    plain = _rms_db(y * (10 ** (-1 / 20) / np.abs(y).max()))            # peak-bound, no limiter
    out = _normalize(y, "rms", sr=sr)
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-9
    assert 2.0 < _rms_db(out) - plain <= 3.05                           # (the spike's own energy is cut)
    easy = 0.1 * np.sin(2 * np.pi * 110 * t)                            # reaches -20 unaided
    assert abs(_rms_db(_normalize(easy, "rms", sr=sr)) + 20) < 0.05


def test_drum_folder_loudest_hits_come_down_to_median_plus_3(tmp_path):
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import _level_folder, _short_term_db
    sr = 44100
    t = np.arange(int(sr * 0.3)) / sr
    fam = tmp_path / "f"
    fam.mkdir()
    ents = []
    for k, (amp, dec) in enumerate([(0.9, 40), (0.9, 40), (0.9, 40), (0.9, 5), (0.2, 40)]):
        y = amp * np.sin(2 * np.pi * 200 * t) * np.exp(-dec * t)          # k=3 rings long: louder
        sf.write(str(fam / f"{k}.wav"), y, sr, subtype="PCM_16")
        ents.append(dict(family="f", out=f"f/{k}.wav"))
    before = [_short_term_db(sf.read(str(fam / f"{k}.wav"))[0], sr) for k in range(5)]
    med = float(np.median(before))
    n = _level_folder(tmp_path, ents, tol_db=3.0, up=False)
    after = [_short_term_db(sf.read(str(fam / f"{k}.wav"))[0], sr) for k in range(5)]
    assert n == 1 and ents[3]["level_db"] < 0 and abs(after[3] - (med + 3)) < 0.2
    assert abs(after[4] - before[4]) < 0.01                            # quiet hits aren't lifted


def test_rootless_tonal_one_shot_retuned_by_pyin(tmp_path):
    """A pad with no note in its name that clearly plays A3 is retuned 3 semitones up to C
    by its pYIN pitch; a noise hit stays as it is."""
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import _fill_roots
    sr = 44100
    t = np.arange(int(1.5 * sr)) / sr
    pad = tmp_path / "Warm Pad 3.wav"
    sf.write(str(pad), 0.4 * np.sin(2 * np.pi * 220 * t) * np.minimum(1, t / 0.02), sr, subtype="PCM_16")
    noise = tmp_path / "Air Texture.wav"
    sf.write(str(noise), 0.3 * np.random.default_rng(0).normal(size=len(t)), sr, subtype="PCM_16")
    rs = [dict(path=str(pad), name_pc=None, retune=None, root_src=None),
          dict(path=str(noise), name_pc=None, retune=None, root_src=None)]
    assert _fill_roots("PADS", [0, 1], rs) == 1
    assert abs(rs[0]["retune"] - 3.0) < 0.1 and rs[0]["root_src"] == "pyin"
    assert rs[1]["retune"] is None


def test_flipped_stereo_keeps_its_louder_channel(tmp_path):
    """A clap whose right channel is the left flipped vanishes summed to mono: it keeps one
    channel. Ordinary wide stereo is left alone."""
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import _fix_phase, phase_stats
    sr = 44100
    t = np.arange(sr // 4) / sr
    s = 0.7 * np.exp(-20 * t) * np.sin(2 * np.pi * 900 * t)
    fam = tmp_path / "f"
    fam.mkdir()
    sf.write(str(fam / "flip.wav"), np.stack([s, -0.95 * s], 1), sr, subtype="PCM_16")
    rng = np.random.default_rng(0)
    wide = np.stack([rng.normal(0, 0.2, len(t)), rng.normal(0, 0.2, len(t))], 1) * np.exp(-5 * t)[:, None]
    sf.write(str(fam / "wide.wav"), wide, sr, subtype="PCM_16")
    corr, loss = phase_stats(sf.read(str(fam / "flip.wav"))[0])
    assert corr < -0.9 and loss > 20
    ents = [dict(family="f", out="f/flip.wav"), dict(family="f", out="f/wide.wav")]
    assert _fix_phase(tmp_path, ents) == 1
    assert sf.info(str(fam / "flip.wav")).channels == 1 and ents[0]["phase_fix"] == "left"
    assert sf.info(str(fam / "wide.wav")).channels == 2 and "phase_fix" not in ents[1]



def test_drums_are_held_to_a_stricter_phase_line(tmp_path):
    """A hat whose channels partly cancel (correlation about -0.5, about 6 dB lost in mono)
    keeps one channel in HATS and DRUMLOOPS; the same file stays stereo in PADS, where only a
    near flip is fixed. Uncorrelated wide noise stays stereo even as a drum."""
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import _fix_phase, phase_stats
    from fourier.packs.curate_config import phase_fix_limits
    sr = 44100
    n = sr // 4
    t = np.arange(n) / sr
    rng = np.random.default_rng(1)
    env = np.exp(-12 * t)
    a, b = rng.normal(0, 0.2, n), rng.normal(0, 0.2, n)
    part = np.stack([a, -0.5 * a + 0.75 * b], 1) * env[:, None]       # partly flipped
    wide = np.stack([a, b], 1) * env[:, None]                         # uncorrelated: about 3 dB
    corr, loss = phase_stats(part)
    assert -0.7 < corr < -0.3 and 4.5 < loss < 10
    assert phase_stats(wide)[1] < 4.5
    for cat, fixed in (("HATS", True), ("DRUMLOOPS", True), ("PADS", False)):
        fam = tmp_path / cat / "f"
        fam.mkdir(parents=True)
        sf.write(str(fam / "part.wav"), part, sr, subtype="PCM_24")
        sf.write(str(fam / "wide.wav"), wide, sr, subtype="PCM_24")
        ents = [dict(family="f", out="f/part.wav"), dict(family="f", out="f/wide.wav")]
        kind = "loop" if cat == "DRUMLOOPS" else "oneshot"
        assert _fix_phase(tmp_path / cat, ents, kind=kind, category=cat) == int(fixed)
        assert sf.info(str(fam / "part.wav")).channels == (1 if fixed else 2)
        assert sf.info(str(fam / "wide.wav")).channels == 2
    assert phase_fix_limits("HATS") != phase_fix_limits("PADS") == phase_fix_limits(None)

def test_detected_root_renames_the_note_in_the_name():
    """Saw2D1-env5-mod8, retuned -2 by its detected D, plays C: the name says C1; a token
    that isn't the detected note (mod8, env5) stays."""
    from fourier.packs.curate import _renamed_detected_note
    assert _renamed_detected_note("Saw2D1-env5-mod8", -2.0, 38.1) == "Saw2C1-env5-mod8"
    assert _renamed_detected_note("Tri2D2-env1", -2.0, 50.0) == "Tri2C2-env1"
    assert _renamed_detected_note("Bass_E1_x", -4.0, 38.0) == "Bass_E1_x"          # E isn't the D it plays
    assert _renamed_detected_note("Pluck_B0", 1.0, 35.0) == "Pluck_C1"                # B0 up one is C1


def test_kick_lead_of_low_noise_is_trimmed(tmp_path):
    """20 ms of low noise above the -50 dB floor before a kick is trimmed
    on the RMS envelope at HIT_LEAD_FLOOR_DB."""
    import numpy as np
    from fourier.packs.curate import _process_audio
    from fourier.packs.curate_config import CATEGORIES
    sr = 44100
    t = np.arange(int(sr * 0.3)) / sr
    kick = 0.9 * np.sin(2 * np.pi * 60 * t) * np.exp(-12 * t)
    y = np.r_[np.random.default_rng(0).normal(0, 0.004, int(0.02 * sr)), kick]    # about -47 dB
    out = _process_audio(y, sr, "oneshot", True, True, "peak",
                         lead_floor_db=CATEGORIES["KICKS"]["trim_lead_db"])
    lead = int(np.argmax(np.abs(out) > 0.1)) / sr
    assert lead < 0.012                                         # envelope + 3 ms pad, not the 20 ms of noise


def test_phase_fix_trims_the_kept_channels_tail(tmp_path):
    """A flipped stereo one-shot whose tail was trimmed on both channels is trimmed again on
    the one it keeps: no quiet past -50 dB of its peak."""
    import numpy as np
    import soundfile as sf
    from fourier.packs.curate import TAIL_PEAK_FLOOR_DB, _fix_phase, _rms_env, _sustained_above
    sr = 44100
    t = np.arange(int(sr * 1.5)) / sr
    loud = 0.8 * np.sin(2 * np.pi * 800 * t) * np.exp(-6 * t)          # -50 dB at ~0.96 s: 0.5 s of quiet
    fam = tmp_path / "f"
    fam.mkdir()
    sf.write(str(fam / "p.wav"), np.stack([loud, -0.99 * loud], 1), sr, subtype="PCM_24")
    ents = [dict(family="f", out="f/p.wav")]
    assert _fix_phase(tmp_path, ents, kind="instrument") == 1
    x, _ = sf.read(str(fam / "p.wav"))
    env = _rms_env(np.abs(x), sr)
    nz = _sustained_above(env, float(np.abs(x).max()) * 10 ** ((TAIL_PEAK_FLOOR_DB - 3) / 20) / float(env.max()),
                          int(sr * 0.02))
    assert (len(env) - 1 - nz[-1]) / sr < 0.06
