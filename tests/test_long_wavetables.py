"""Wave folders: a multi-frame wavetable over WAVE_MAX_DUR is in no category (out of scope,
like an impulse response), and a single small cycle in a wavetable pack's kit folder is a
wave, while the kit's real drum hits still home as drums."""
from fourier.metadata import sound
from fourier.packs import curate as C
from fourier.packs import why as W

from .test_verify import _clean, _fails, _hit

TABLE = "Acme/Samples/WAVETABLES/Qxvwz Tables/Qzv Table 01.wav"         # 64 frames of 2048
TABLE_S = 2048 * 64 / 44100
KIT = "Acme/Qxvwz Wavetable Pack/For Synth Q/Drumkit/Qzv Kit (frames)/"


def test_long_wavetable_is_neither_a_wave_nor_a_sample():
    assert C._is_long_wavetable(TABLE, TABLE_S) and not C._is_wave(TABLE, TABLE_S)
    assert C._is_long_wavetable("Free/Single Cycle Waves/Set 01/Pad Table.wav", 1.5)
    assert C._is_wave("Acme/Samples/WAVETABLES/Qxvwz Tables/Saw.wav", 2048 / 44100)
    assert not C._is_long_wavetable("Acme/Samples/WAVETABLES/Qxvwz Tables/Saw.wav", 2048 / 44100)
    assert not C._is_wave(TABLE, C.WAVE_MAX_DUR + 0.001)            # the cap itself is WAVES'
    assert C._is_wave(TABLE, C.WAVE_MAX_DUR)
    # not a wave folder, or a drum kit shipped in one: an ordinary sample
    assert not C._is_long_wavetable("Acme/Northwind Pads/Pads/Wave Pad 01.wav", 4.0)
    assert not C._is_long_wavetable("Acme/Wavetable Pack/Synth A/Drumkit/Kit 01/Crash 03.wav", 2.0)
    assert not C._is_long_wavetable(TABLE, None)


def test_tiny_cycles_in_a_wavetable_packs_kit_are_waves():
    assert C._is_wave(KIT + "07 BD4.wav", 264 / 44100)
    assert C._is_wave(KIT + "13 QM.wav", 256 / 22050)               # a 256-sample frame at 22.05 kHz
    # a real drum hit in a kit beside a 'wave' word stays a drum, even a short one
    assert not C._is_wave(KIT + "Hat Closed 03.wav", 0.05)
    assert not C._is_wave("Acme/Wavetable Pack/Synth A/Drumkit/Kit 01/Cymb 07.wav", 0.05)
    assert not C._is_wave("Acme/Drum Kits/Kit 01/Click 01.wav", 0.006)    # no wave folder above


def test_long_wavetable_out_of_selection_even_as_a_keep(monkeypatch):
    from tests.regressions.test_curate_regressions import _emb, _pool, _row, _run_select
    rows = _pool()
    wt = _row(100, "Acme", "Qzv Table 01.wav")
    wt.rel_path, wt.duration_s = TABLE, TABLE_S
    wt.path = "/lib/SampleLibrary/" + TABLE
    rows.append(wt)
    got = _run_select(monkeypatch, rows, "PADS", {r.id: "PADS" for r in rows}, keeps={wt.path: "PADS"},
                      misfiled={}, emb=_emb(len(rows)))
    assert wt.path not in got and len(got) == len(rows) - 1


def test_long_wavetable_gives_no_training_label_and_why_says_so():
    assert sound.training_label(TABLE, TABLE_S) == (None, None)
    assert sound.training_label(KIT + "07 BD4.wav", 264 / 44100)[0] == "WAVES"
    assert "long wavetable" in W.PLAIN["ir_preview"]


def test_verify_fails_on_a_long_wavetable(tmp_path):
    files = _clean() + [("PADS", "warm-analog-dark", "Qzv_Table_01.wav", _hit(1.6), TABLE)]
    assert "no wavetables longer than 1 s (out of scope)" in _fails(tmp_path, files)[1]
    short = _clean() + [("PADS", "warm-analog-dark", "Pad_01.wav", _hit(1.6),
                         "Acme/Northwind Pads/Pads/Pad 01.wav")]
    assert "no wavetables longer than 1 s (out of scope)" not in _fails(tmp_path / "b", short)[1]
