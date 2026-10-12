"""Hat bands go by how long a hat sounds, and the kits' hat and tom slots (packs/sets.py)."""
import os
from types import SimpleNamespace

from fourier.metadata.rows import CANDIDATE_FIELDS
from fourier.packs import curate as C
from fourier.packs import sets as S


def test_a_hat_sounds_for_its_attack_and_decay_within_the_file():
    assert C._hat_seconds(1.36, 9.0, 95.0) == 0.104            # a short hit in a long file
    assert C._hat_seconds(0.5, 20.0, 900.0) == 0.5             # never past the file's end
    assert C._hat_seconds(0.8, None, None) == 0.8              # not analyzed: the file's length
    assert C._hat_seconds(None, None, 120.0) == 0.12


def test_a_short_hat_in_a_long_file_is_closed():
    assert C._hat_band("Qzv Hat 3.wav", C._hat_seconds(1.36, 9.0, 95.0)) == "closed"
    assert C._hat_band("Qzv Hat 3.wav", C._hat_seconds(1.36, 9.0, 600.0)) == "open"
    assert C._hat_band("Qzv Open Hat Vqx.wav", C._hat_seconds(1.36, 9.0, 95.0)) == "open"   # the name decides


def _row(**kw):
    base = {f: None for f in CANDIDATE_FIELDS}
    base.update(id=1, filename="Qzv Hat 3.wav", rel_path="Acme/Qzv/Qzv Hat 3.wav",
                path="/l/Acme/Qzv/Qzv Hat 3.wav", duration_s=1.36, classes=["One Shot"])
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_build_bands_a_hat_on_its_sounding_length():
    rec = lambda r: C._mkrec(r, "Acme/Qzv", r.rel_path, "HATS", None, {1: 0})
    assert rec(_row(attack_time_ms=9.0, decay_time_ms=95.0))["band"] == "closed"
    assert rec(_row(attack_time_ms=9.0, decay_time_ms=700.0))["band"] == "open"
    assert rec(_row())["band"] == "open"                       # no decay: the file's length, as before


def test_a_cl_hat_is_closed():
    for n in ("zq_cl_hat_b2", "Qx_clhat_loose", "Vqx ClHH 2", "Acme Cl HiHat 3", "QX cl-hi-hat"):
        assert C._hat_band(f"{n}.wav", 1.0) == "closed", n
    for n in ("Clap Hat Vqx", "Click Hat Qzv", "Cyclhat Vqx", "Uncl_hat Qzv"):    # not the token
        assert C._hat_band(f"{n}.wav", 1.0) == "open", n


H = lambda n: ("HATS", dict(src=f"/x/{n}.wav", out=f"f/{n}.wav"))      # noqa: E731
names = lambda cs: [os.path.basename(ce[1]["src"]) for ce in cs]        # noqa: E731


def test_a_required_slot_never_falls_back_to_the_other_hat():
    assert names(S._slot_candidates("hat_open", [H("Hat Closed Qz"), H("Qzv 7")])) == ["Qzv 7.wav"]
    assert names(S._slot_candidates("hat_open", [H("zq cl hat b2")])) == []
    assert names(S._slot_candidates("hat_closed", [H("Qzv Open 2")])) == []
    # a pack prefix and an overhead mic say neither hat
    assert names(S._slot_candidates("hat_open", [H("QX4d-Vqx_05")])) == ["QX4d-Vqx_05.wav"]
    assert names(S._slot_candidates("hat_closed", [H("Qzv Hat Closed +OH 3")])) == ["Qzv Hat Closed +OH 3.wav"]


def test_a_kit_without_an_open_hat_drops_out():
    def ent(cat, name, band=None):
        return cat, dict(family="f", out=f"f/{name}", src=f"/x/SampleLibrary/Acme/Qzv/{name}", band=band)
    base = [ent("KICKS", "Qzv Kick Vqx.wav"), ent("SNARES", "Qzv Snare Vqx.wav"), ent("CLAPS", "Qzv Clap Vqx.wav"),
            ent("HATS", "Qzv Hat Closed Vqx.wav", "closed"), ent("CYMBALS", "Qzv Crash Vqx.wav"), ent("TOMS", "Qzv Tom Vqx.wav")]

    def man(extra):
        m: dict = {"categories": {}}
        for cat, e in base + extra:
            m["categories"].setdefault(cat, {"entries": []})["entries"].append(e)
        return m
    assert len(S._kits(man([ent("HATS", "Qzv 9.wav", "open")]))) == 1
    assert S._kits(man([ent("HATS", "zq_cl_hat_9.wav", "open")])) == []


def test_two_toms_are_two_drums_when_the_pack_has_them():
    T = lambda n: ("TOMS", dict(src=f"/x/{n}.wav", out=f"f/{n}.wav"))   # noqa: E731
    c = [T(n) for n in ("Qzv Tom Low 1", "Qzv Tom Low 4", "Qzv Tom Low 7", "Qzv Tom Mid 2", "Qzv Tom Mid 5")]
    assert names(S._spread_pick(c, 2)) == ["Qzv Tom Low 4.wav", "Qzv Tom Mid 2.wav"]
    v = [T(n) for n in ("qx_tom_hi_p3", "qx_tom_low_f2", "qx_tom_low_p2")]     # velocity layers
    assert names(S._spread_pick(v, 2)) == ["qx_tom_low_f2.wav", "qx_tom_hi_p3.wav"]
    # one drum only: its takes, as before
    same = [T(f"Qzv Tom Vqx {i}") for i in range(1, 6)]
    assert names(S._spread_pick(same, 2)) == ["Qzv Tom Vqx 2.wav", "Qzv Tom Vqx 3.wav"]
    assert names(S._spread_pick(c, 1)) == ["Qzv Tom Low 7.wav"]


def _hat_master(tmp_path, band, src_seconds):
    import json

    import soundfile as sf

    from .test_taxonomy import _with_bands
    from .test_verify import SR, _clean, _hit, _master
    root, store = _master(tmp_path, _clean() + [("HATS", f"{band}-qzv", "Hat 05.wav", _hit(0.1), "V1/P/Hat 05.wav")])
    src = tmp_path / "Hat 05.wav"                   # the source: the same hit, then silence
    y = _hit(0.1)
    sf.write(src, list(y) + [0.0] * int(SR * (src_seconds - 0.1)), SR)
    _with_bands(root, "HATS", {"Hat 05.wav": {"band": band}}, {f"{band}-qzv": band})
    mp = tmp_path / "master" / "manifest.json"
    m = json.loads(mp.read_text())
    for e in m["categories"]["HATS"]["entries"]:
        e["src"] = str(src)
    mp.write_text(json.dumps(m))
    return root, store, str(src)


def _band_fails(root, store, hat_s=None):
    from fourier.packs.verify import FAIL, _Ctx, check_bands
    ctx = _Ctx(root, store_path=store)
    ctx.hat_s = hat_s
    return [r.detail for r in check_bands(ctx) if r.level == FAIL and r.check == "HATS files are in the right hat band"]


def test_verify_judges_a_hat_by_how_long_it_sounds(tmp_path):
    # with the database: as the build judged it
    root, store, src = _hat_master(tmp_path / "a", "open", 1.4)
    assert _band_fails(root, store, {src: C._hat_seconds(1.4, 5.0, 95.0)})
    root, store, src = _hat_master(tmp_path / "b", "closed", 1.4)
    assert not _band_fails(root, store, {src: C._hat_seconds(1.4, 5.0, 95.0)})
    # without it: somewhere between the trimmed export and the source
    assert not _band_fails(root, store)
    root, store, src = _hat_master(tmp_path / "c", "open", 0.1)
    assert _band_fails(root, store)


def test_acoustic_kit_pins_follow_the_same_pick(monkeypatch):
    import re
    monkeypatch.setattr(S, "KIT_ACOUSTIC_PACKS", re.compile(r"^Acme/Qzv"))
    toms = ("Qzv Tom Low 1", "Qzv Tom Low 4", "Qzv Tom Low 7", "Qzv Tom Mid 2", "Qzv Tom Mid 5")
    rec = [dict(path=f"/x/SampleLibrary/Acme/Qzv/{n}.wav") for n in toms]
    assert C._kit_source_pins(rec, "TOMS") == 2
    assert [os.path.basename(d["path"]) for d in rec if d.get("pin")] == ["Qzv Tom Low 4.wav", "Qzv Tom Mid 2.wav"]
