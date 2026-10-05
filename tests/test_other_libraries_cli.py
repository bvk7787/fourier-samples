"""Libraries that don't look like a vendor/pack library, through the real CLI (one process per
command, as from a shell; the sandbox and stand-in CLAP of tests/test_first_run.py, here
hearing each file as what it is rather than what its name says):

  Types   folders by sound type ("Drums/Snares/SD_01.wav"), names with digits glued on
          ("kick01", "HiHat58"), drum loops whose tempo only a WAV ACID chunk states, and a
          tempo folder ("Loops/174/")
  Dump    a flat folder of numbered files ("001.wav"): no word for any rule
  HipHop  vendor/pack folders of 8-bar boom-bap loops at 85 BPM (22.6 s each)

built with `balanced` (what setup writes, with vendors and fold "auto"), then with
`hiphop-lofi` from the same database."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_first_run import SR, Sandbox, _flat, _has, _sounds  # noqa: E402

# one sandbox, built once, that the tests below change in turn: one worker runs them all, in
# order (pytest -n: --dist loadgroup, conftest.py)
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("other_libraries_cli")]

ROOT = Path(__file__).resolve().parents[1]

# the stand-in CLAP hears a file as the sound it is (CONCEPTS, a map beside the library, not in
# it), whatever its name says
RUNNER = """
import json, os, sys
sys.path[:0] = [{src!r}, {golden!r}]
import numpy as np
import synthetic_build as SB
import fourier.analysis.clap_features as CF
HEARS = json.load(open({concepts!r}))


def embed_audio_file(path):
    p = str(path)
    rel = p.split("SampleLibrary/")[-1]
    v = SB.fake_embed_text(HEARS.get(rel, rel)) + 0.3 * SB.unit("file", rel)
    return (v / np.linalg.norm(v)).astype("float32")


CF.embed_text = SB.fake_embed_text
CF.embed_audio_file = embed_audio_file
from fourier.cli import main
if __name__ == "__main__":
    main(sys.argv[1:], prog_name="fourier")
"""


def _loop(kick, snare, hat, bpm, i, bars):
    beat = int(round(60 / bpm * SR))
    y = 0.02 * np.random.default_rng(i).standard_normal(beat * 4 * bars)
    for b in range(4 * bars):
        hit = (kick(i) if b % 2 == 0 else snare(i))[:beat // 2]
        y[b * beat:b * beat + len(hit)] += hit
        h = hat(i)[:beat // 4]
        y[b * beat + beat // 2:b * beat + beat // 2 + len(h)] += 0.5 * h
    return y


def make_libraries(lib: Path) -> dict:
    """{library path: what CLAP hears} for the three libraries under lib (a SampleLibrary
    folder holding them)."""
    from fourier.ingest.chunks import acid_chunk, add_chunks
    kick, snare, hat, clap, _loop0, bass, pad = _sounds(np.random.default_rng(17))
    hears = {}

    def w(rel, y, sound, chunks=()):
        p = lib / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (0.8 * y / np.max(np.abs(y))).astype("float32"), SR, subtype="PCM_16")
        if chunks:
            add_chunks(p, *chunks)
        hears[rel] = f"{sound} {rel.rsplit('/', 1)[-1]}"

    for i in range(6):
        w(f"Types/Samples/kick{i + 1:02d}.wav", kick(i), "kick drum")
        w(f"Types/Drums/Snares/SD_{i + 1:02d}.wav", snare(i), "snare drum")
        w(f"Types/Drums/HiHats/HiHat{i + 1:02d}.wav", hat(i), "closed hi-hat")
        w(f"Dump/{i + 1:03d}.wav", kick(i + 10), "kick drum")
        w(f"Dump/{i + 7:03d}.wav", snare(i + 10), "snare drum")
        w(f"Dump/{i + 13:03d}.wav", pad(i), "warm analog pad")
    for i in range(4):
        w(f"Types/Loops/Acid/Beat {chr(65 + i)}.wav", _loop(kick, snare, hat, 128, i, 4), "house drum loop",
          (acid_chunk(128.0, 16),))
        w(f"Types/Loops/174/Groove {i + 1:02d}.wav", _loop(kick, snare, hat, 174, i + 4, 4), "jungle breakbeat")
    for v in ("Acme Beats", "Northwind Audio", "Vendor C"):
        for i in range(2):
            w(f"HipHop/{v}/Dusty Kit/Drum Loops/Dusty Loop {i + 1:02d} 85 BPM.wav",
              _loop(kick, snare, hat, 85, i + 20 + len(v), 8), "boom bap break hip hop drum break")
    return hears


class Libraries(Sandbox):
    def __init__(self, tmp: Path):
        holder = {}

        def make(lib):
            holder.update(make_libraries(lib))
            return len(holder)
        super().__init__(tmp, make=make)
        concepts = tmp / "hears.json"
        concepts.write_text(json.dumps(holder))
        self.runner.write_text(RUNNER.format(src=str(ROOT / "src"), golden=str(ROOT / "tests" / "golden"),
                                             concepts=str(concepts)))

    def manifest(self):
        return json.loads((self.master / "manifest.json").read_text())


@pytest.fixture(scope="module")
def libs(tmp_path_factory):
    return Libraries(tmp_path_factory.mktemp("other-libraries"))


def _per(man):
    return {c: [e["src"].split("SampleLibrary/")[1] for e in v.get("entries") or ()]
            for c, v in man["categories"].items() if v.get("entries")}


def test_libraries_unlike_the_reference_one(libs):
    b = libs
    roots = [str(b.lib / r) for r in ("Dump", "HipHop", "Types")]
    out = b.run("setup", "--yes", "--no-clap", "--device", "generic_48k",
                *[x for r in roots for x in ("--library", r)])
    cfg = b.user / ".config" / "fourier" / "fourier.toml"
    text = cfg.read_text()
    assert 'vendors = "auto"' in text and 'fold = "auto"' in text and 'preset = "balanced"' in text
    assert _has(out, "files with no folders") and _has(out, "folders by sound type")
    assert "What do you make?" in out and "hiphop-lofi" in out
    out = b.run("build", "--all", "-j", "1", "--no-describe")
    flat = _flat(out)
    assert "no name rule recognized; the CLAP fallback placed" in flat
    per = _per(b.manifest())
    # names with digits glued on, and a flat dump placed by sound
    assert {f"Types/Samples/kick{i:02d}.wav" for i in range(1, 7)} <= set(per["KICKS"])
    assert sum(x.startswith("Dump/") for x in per["KICKS"]) >= 4
    assert sum(x.startswith("Dump/") for x in per["SNARES"]) >= 4
    assert sum(x.startswith("Dump/") for x in per["PADS"]) >= 4
    # loops whose tempo only their ACID chunk states
    loops = {e["src"].split("SampleLibrary/")[1]: e for e in b.manifest()["categories"]["DRUMLOOPS"]["entries"]}
    acid = [e for k, e in loops.items() if "/Acid/" in k]
    assert acid and all(e["bpm"] == 128.0 and e["bpm_src"] == "acid" for e in acid)
    # 22.6 s loops: longer than balanced's drum loops take
    assert not any("Dusty Loop" in k for k in loops)
    out = _flat(b.run("why", "--detail", "Dusty Loop 01"))
    assert "its length is outside DRUMLOOPS" in out
    out = _flat(b.run("why", "--detail", str(b.lib / "Dump" / "001.wav")))
    assert "placed by sound (CLAP), no name rule matched" in out
    out = _flat(b.run("doctor"))
    assert "library layout" in out and "folders by sound type" in out
    b.run("why", "--detail", "--unrecognized")


def test_hiphop_lofi_takes_the_long_loops_at_their_tempo(libs):
    b = libs
    cfg = b.user / ".config" / "fourier" / "fourier.toml"
    if not cfg.exists():
        pytest.skip("needs the build above")
    cfg.write_text(cfg.read_text().replace('preset = "balanced"', 'preset = "hiphop-lofi"'))
    b.run("build", "--all", "-j", "1", "--no-describe", "--no-scan")
    ents = b.manifest()["categories"]["DRUMLOOPS"]["entries"]
    dusty = [e for e in ents if "Dusty Loop" in e["src"]]
    assert dusty and all(e["bpm_fold"] == 85.0 and e["family"].startswith("085bpm") for e in dusty)
    # the 174 BPM grooves fold into the range (87), the 128 BPM ones would leave it: they stay
    by = {e["src"].split("SampleLibrary/")[1]: e for e in ents}
    assert all(e["bpm_fold"] == 87.0 for k, e in by.items() if "/174/" in k)
    assert all(e["bpm_fold"] == 128.0 for k, e in by.items() if "/Acid/" in k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
