"""What someone new to a terminal does: drags a folder in from Finder, answers with a device's
name or its number in the list, edits fourier.toml in TextEdit (smart quotes on), and wants a
folder shown in Finder rather than a path to find."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from fourier.cli import main
from fourier.cli.setup import folder_answers, pick_devices


def _esc(p) -> str:
    """How macOS Terminal pastes a folder dragged into it."""
    s = str(p)
    for ch in " ()&'":
        s = s.replace(ch, "\\" + ch)
    return s + " "


@pytest.mark.skipif(os.name == "nt", reason="a POSIX terminal's escaping")
def test_a_dragged_folder_is_the_folder(tmp_path):
    a = tmp_path / "Sample Packs (Bought)"
    b = tmp_path / "Hits, Set Two"
    c = tmp_path / "Loops"
    for d in (a, b, c):
        d.mkdir()
    assert folder_answers([_esc(a)]) == [str(a)]
    assert folder_answers([f"'{a}'"]) == [str(a)]
    assert folder_answers([str(b)]) == [str(b)]                  # a comma in its name
    assert folder_answers([_esc(a) + _esc(c)]) == [str(a), str(c)]  # two dragged in at once
    assert folder_answers([f"{c}, {a}"]) == [str(c), str(a)]
    assert folder_answers(["", "  "]) == []


def test_devices_by_number_or_name():
    known = ["digitakt_2", "m8_tracker", "generic_48k"]
    names = {"digitakt_2": "Elektron Digitakt 2", "m8_tracker": "Dirtywave M8",
             "generic_48k": "Generic 48 kHz"}
    assert pick_devices(["1"], known, names) == (["digitakt_2"], [])
    assert pick_devices(["Digitakt 2, m8"], known, names) == (["digitakt_2", "m8_tracker"], [])
    assert pick_devices(["digitakt_2"], known, names) == (["digitakt_2"], [])
    assert pick_devices(["9, polysampler"], known, names) == ([], ["9", "polysampler"])


def test_setup_takes_a_dragged_folder_and_a_device_number(tmp_path, monkeypatch):
    empty = tmp_path / "Empty"
    empty.mkdir()
    lib = tmp_path / "My Samples"
    lib.mkdir()
    _wav(lib / "Kick 01.wav")
    target = tmp_path / "fourier.toml"
    drag = (lambda p: _esc(p) if os.name != "nt" else str(p))
    answers = "\n".join([drag(empty), drag(lib),         # an empty folder is asked again
                         "", "1",                         # no device default: asked again
                         "1",                             # the first style
                         "1", "2", "BLIPS, 16",           # starter size, as-is, two left out
                         "y"]) + "\n"
    r = CliRunner().invoke(main, ["setup", "--to", str(target), "--no-clap", "--no-llm",
                                  "--no-build", "--no-sound-model"], input=answers)
    text = target.read_text() if target.exists() else ""
    assert str(lib) in text, r.output
    assert "digitakt_2" in text, r.output
    import tomllib
    doc = tomllib.loads(text)
    assert doc["size"] == "1GB" and doc["retune"] == "off" and doc["names"] == "keep"
    assert doc["categories"] == {"BLIPS": "off", "WAVES": "off"}
    assert "minutes to load with Transfer" in " ".join(r.output.split())
    assert "Drag the folder" in r.output
    assert "No audio files there" in r.output


def _wav(path):
    import numpy as np
    import soundfile as sf
    sf.write(str(path), np.zeros(4800, dtype="float32"), 48000)


def test_a_config_typed_with_smart_quotes_still_loads(tmp_path):
    from fourier import layers
    p = tmp_path / "fourier.toml"
    p.write_text('preset = “balanced”\nwords = { KICKS = [“bombo”] }\n')
    doc = layers._read(p)
    assert doc["preset"] == "balanced" and doc["words"] == {"KICKS": ["bombo"]}


def test_a_broken_config_names_config_edit(tmp_path):
    from fourier import layers
    from fourier.settings import ConfigError
    p = tmp_path / "fourier.toml"
    p.write_text("preset = balanced\n")
    with pytest.raises(ConfigError, match=r'line 1 \(preset = balanced\): a value that needs quotes'):
        layers._read(p)
    p.write_text('words = { KICKS = ["bombo" }\n')
    with pytest.raises(ConfigError, match=r"line 1 .*isn't closed"):
        layers._read(p)


def _recording_editor(tmp_path) -> tuple[str, Path]:
    log = tmp_path / "edited.txt"
    script = tmp_path / "ed.py"
    script.write_text(f"import sys; open({str(log)!r}, 'w').write(sys.argv[1])\n")
    return f"{sys.executable} {script}", log


def test_config_edit_opens_the_file_even_when_it_doesnt_load(tmp_path, monkeypatch):
    cfg = tmp_path / "fourier.toml"
    cfg.write_text("preset = balanced\n")                      # not valid TOML
    monkeypatch.setenv("FOURIER_CONFIG", str(cfg))
    editor, log = _recording_editor(tmp_path)
    monkeypatch.setenv("EDITOR", editor)
    r = CliRunner().invoke(main, ["config", "edit"])
    assert r.exit_code == 0, r.output
    assert log.read_text() == str(cfg)
    assert "doesn't load as it is" in r.output
    r = CliRunner().invoke(main, ["config", "show"])            # show still says what's wrong
    assert r.exit_code == 2 and "fourier config edit" in r.output


def test_config_edit_without_a_config_points_at_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(main, ["config", "edit"])
    assert r.exit_code == 1 and "fourier setup" in r.output


def test_open_shows_the_folder_or_says_what_makes_it(tmp_path, monkeypatch):
    master = tmp_path / "Curated"
    monkeypatch.setenv("FOURIER_CURATED_DIR", str(master))
    opened = []
    monkeypatch.setattr("fourier.platforms.open_path", lambda p, text=False: opened.append(str(p)) or ["open"])
    r = CliRunner().invoke(main, ["open"])
    assert r.exit_code == 1 and "fourier build" in r.output and not opened
    master.mkdir()
    r = CliRunner().invoke(main, ["open"])
    assert r.exit_code == 0 and opened == [str(master)]
    r = CliRunner().invoke(main, ["open", "renders", "digitakt_2", "--print"])
    assert r.output.strip() == str(tmp_path / "FourierRenders" / "digitakt_2")
    r = CliRunner().invoke(main, ["open", "master", "digitakt_2"])
    assert r.exit_code == 2


def test_open_platforms(monkeypatch, tmp_path):
    from fourier import platforms
    ran = []
    monkeypatch.setattr(platforms.subprocess, "run", lambda cmd, **k: ran.append(cmd))
    monkeypatch.setattr(platforms.sys, "platform", "darwin")
    assert platforms.open_path(tmp_path) == ["open", str(tmp_path)]
    assert platforms.open_path(tmp_path / "f.toml", text=True)[:2] == ["open", "-t"]
    monkeypatch.setattr(platforms.sys, "platform", "linux")
    monkeypatch.setattr(platforms.shutil, "which", lambda name: None)
    assert platforms.open_path(tmp_path) is None


def test_every_setting_has_plain_help():
    from fourier.knobs import KNOBS, PLAIN_HELP
    assert set(PLAIN_HELP) == set(KNOBS)
    for name, (plain, example) in PLAIN_HELP.items():
        assert example.startswith(name + " = ") and "_" not in plain.replace("00_", ""), name


def _master(tmp_path):
    import json
    m = tmp_path / "Curated"
    for cat, fam, name in (("KICKS", "punchy", "Kick_01.wav"), ("DRUMLOOPS", "120-125bpm", "Loop_01.wav")):
        (m / cat / fam).mkdir(parents=True)
        _wav(m / cat / fam / name)
    (m / "manifest.json").write_text(json.dumps({"generated": "2026-10-04T10:00:00Z", "categories": {
        "DRUMLOOPS": {"entries": [{"family": "120-125bpm", "out": "120-125bpm/Loop_01.wav",
                                   "src": str(tmp_path / "lib" / "Acme" / "Loop 01.wav")}]},
        "KICKS": {"entries": [{"family": "punchy", "out": "punchy/Kick_01.wav",
                               "src": str(tmp_path / "lib" / "Acme" / "Kick 01.wav")}]}}}))
    (m / "loops.csv").write_text("folder,file,bpm,bars\n120-125bpm,Loop_01.wav,122.0,4.0\n")
    return m


def test_the_report_lists_every_file_with_a_player(tmp_path, monkeypatch):
    from fourier.packs.report import build_report
    monkeypatch.setenv("FOURIER_LIBRARY", str(tmp_path / "lib"))
    m = _master(tmp_path)
    r = build_report(m, tmp_path / "r.html")
    page = (tmp_path / "r.html").read_text()
    assert r["files"] == 2 and r["categories"] == 2
    assert (m / "KICKS" / "punchy" / "Kick_01.wav").as_uri() in page
    assert page.index('id="KICKS"') < page.index('id="DRUMLOOPS"')        # play order
    assert "122 BPM, 4 bars" in page and "from Acme/Kick 01.wav" in page
    assert 'data-path="KICKS/punchy/Kick_01.wav"' in page and "fourier review import" in page


def test_a_report_csv_imports(tmp_path, monkeypatch):
    """What the report's Export button writes is what `review import` reads."""
    from fourier.packs.ratings import read_ratings_csv
    monkeypatch.setenv("FOURIER_LIBRARY", str(tmp_path / "lib"))
    m = _master(tmp_path)
    csv = tmp_path / "ratings.csv"
    csv.write_text("path,rating,category\nKICKS/punchy/Kick_01.wav,misfiled,PERC\n"
                   "DRUMLOOPS/120-125bpm/Loop_01.wav,keep,\n")
    rated, skipped = read_ratings_csv(str(csv), str(m))
    assert not skipped and len(rated) == 2


def test_new_only_copies_what_the_release_added(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from fourier.cli import releases as R
    render = tmp_path / "renders" / "digitakt_2"
    for rel in ("01_KICKS/punchy/Kick_01.wav", "01_KICKS/punchy/Kick_02.wav"):
        (render / rel).parent.mkdir(parents=True, exist_ok=True)
        (render / rel).write_bytes(b"x")
    lock = SimpleNamespace(files={"a": {"path": "01_KICKS/punchy/Kick_01.wav", "release": "v1"},
                                  "b": {"path": "01_KICKS/punchy/Kick_02.wav", "release": "v2"}})
    monkeypatch.setattr("fourier.packs.device_lock.load_lock", lambda d: lock)
    R._new_only("digitakt_2", "v2", str(render), str(render))
    out = tmp_path / "renders" / "digitakt_2-new-in-v2"
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.wav")) == ["01_KICKS/punchy/Kick_02.wav"]
    R._new_only("digitakt_2", "v2", str(render), str(render))             # again: replaced, not refused
    (tmp_path / "renders" / "digitakt_2-new-in-v3").mkdir()
    import pytest
    with pytest.raises(SystemExit):                                        # a folder of the user's
        R._new_only("digitakt_2", "v3", str(render), str(render))
