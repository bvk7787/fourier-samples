"""Resumable builds (fourier/packs/progress.py)."""
from fourier.packs.progress import Progress, resumable


def test_a_build_resumes_only_with_the_same_fingerprint(tmp_path):
    nxt = tmp_path / "FourierCurated.next"
    p = Progress(nxt, "fp1")
    assert p.start(resume=True, log=lambda m: None) == {}          # nothing to resume
    (nxt / "KICKS").mkdir()
    p.done("KICKS", {"files": 3, "families": 1}, [{"out": "a/k.wav"}])
    (nxt / "SNARES").mkdir()                                         # built, never recorded
    assert resumable(nxt)
    again = Progress(nxt, "fp1").start(resume=True, log=lambda m: None)
    assert again == {"KICKS": ({"files": 3, "families": 1}, [{"out": "a/k.wav"}])}
    assert (nxt / "SNARES").is_dir()                                 # kept, rebuilt by the build
    msgs = []
    assert Progress(nxt, "fp2").start(resume=True, log=msgs.append) == {}
    assert "starting over" in msgs[0] and not (nxt / "KICKS").exists()
    p2 = Progress(nxt, "fp2")
    p2.start(resume=False, log=lambda m: None)
    p2.finish()
    assert not resumable(nxt)


def test_a_recorded_category_whose_folder_is_gone_is_rebuilt(tmp_path):
    nxt = tmp_path / "m.next"
    p = Progress(nxt, "fp")
    p.start(resume=False, log=lambda m: None)
    p.done("PADS", {"files": 1}, [])
    assert Progress(nxt, "fp").start(resume=True, log=lambda m: None) == {}
