"""Loading the CLAP model makes no network request once it's downloaded
(analysis/clap_features.py): it loads from the pinned revision's folder in the Hugging Face
cache with the Hub offline and Transformers' safetensors conversion off. Every test here
fails any socket connection."""
import importlib.util
import logging
import socket
import sys
import types

import pytest

from fourier.analysis import clap_features as CF


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError(f"network access: {a[1:] if len(a) > 1 else a}")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def hf_cache(tmp_path, monkeypatch):
    """An empty Hugging Face cache of the test's own, and a clean process state."""
    for var in ("HF_HUB_OFFLINE", "DISABLE_SAFETENSORS_CONVERSION", "HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.setenv(var, "")          # restored (or removed) after the test, whatever the
        monkeypatch.delenv(var)              # loader sets meanwhile
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    CF._load_model_and_processor.cache_clear()
    yield tmp_path / "hub"
    CF._load_model_and_processor.cache_clear()


def _snapshot(hub):
    snap = hub / "models--laion--clap-htsat-unfused" / "snapshots" / CF.CLAP_REVISION
    snap.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "vocab.json", "merges.txt",
                 "tokenizer_config.json", "pytorch_model.bin"):
        (snap / name).write_text("{}")
    return snap


@pytest.fixture
def fake_clap(monkeypatch):
    """Stand-ins for transformers and torch that record how the model was loaded."""
    calls = []

    class Loader:
        def __init__(self, name):
            self.name = name

        def from_pretrained(self, what, **kw):
            import os
            calls.append((self.name, str(what), kw, os.environ.get("HF_HUB_OFFLINE"),
                          os.environ.get("DISABLE_SAFETENSORS_CONVERSION")))
            return types.SimpleNamespace(eval=lambda: None, to=lambda device: "model")

    tf = types.ModuleType("transformers")
    tf.ClapModel, tf.ClapProcessor = Loader("model"), Loader("processor")
    torch = types.ModuleType("torch")
    no = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(mps=no)
    torch.cuda = no
    torch.device = lambda name: name
    tlog = types.ModuleType("transformers.utils.logging")
    tlog.set_verbosity_error = lambda: calls.append(("quiet", "verbosity"))
    tlog.disable_progress_bar = lambda: calls.append(("quiet", "bars"))
    tu = types.ModuleType("transformers.utils")
    tu.logging = tlog
    tf.utils = tu
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "transformers.utils", tu)
    monkeypatch.setitem(sys.modules, "transformers.utils.logging", tlog)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_a_downloaded_model_loads_offline(hf_cache, fake_clap, no_network):
    snap = _snapshot(hf_cache)
    assert CF.cached_snapshot() == snap
    CF._load_model_and_processor()
    # Transformers' progress bars ("Loading weights" on every load) and notices are off
    assert fake_clap[:2] == [("quiet", "verbosity"), ("quiet", "bars")]
    del fake_clap[:2]
    # both from the revision's folder, local files only, the Hub offline, no conversion thread
    assert [(name, what, kw) for name, what, kw, _o, _d in fake_clap] == [
        ("processor", str(snap), {"local_files_only": True}),
        ("model", str(snap), {"local_files_only": True})]
    assert all(off == "1" and conv == "true" for *_x, off, conv in fake_clap)
    # the Hub's token notice stays out of the output
    assert logging.getLogger("huggingface_hub.utils._http").level == logging.ERROR


def test_the_first_use_downloads_the_pinned_revision(hf_cache, fake_clap, monkeypatch):
    assert CF.cached_snapshot() is None
    CF._load_model_and_processor()
    del fake_clap[:2]                                              # the quieting
    assert [(name, what, kw) for name, what, kw, _o, _d in fake_clap] == [
        ("processor", CF.CLAP_MODEL_ID, {"revision": CF.CLAP_REVISION}),
        ("model", CF.CLAP_MODEL_ID, {"revision": CF.CLAP_REVISION})]
    assert all(off is None for *_x, off, _conv in fake_clap)       # online, for the download


def test_an_incomplete_download_is_not_a_cached_model(hf_cache):
    snap = _snapshot(hf_cache)
    (snap / "pytorch_model.bin").unlink()
    assert CF.cached_snapshot() is None


def test_setup_and_doctor_find_the_model_without_the_hub(hf_cache, no_network, monkeypatch):
    from fourier.cli import enrich as E
    from fourier.cli import setup as S
    def real(path):                          # the encoder itself, not a test's stand-in
        pass
    real.__module__ = CF.__name__
    monkeypatch.setattr(CF, "embed_audio_file", real)
    assert not S.clap_model_cached()
    _snapshot(hf_cache)
    assert S.clap_model_cached() and not E.clap_model_missing()


@pytest.mark.skipif(not (importlib.util.find_spec("transformers") and importlib.util.find_spec("torch")
                         and CF.cached_snapshot()),
                    reason="needs the [clap] extra and the downloaded model")
def test_the_real_model_loads_with_no_network(no_network, monkeypatch):
    CF._load_model_and_processor.cache_clear()
    try:
        model, processor, device = CF._load_model_and_processor()
        assert model is not None and processor is not None
    finally:
        CF._load_model_and_processor.cache_clear()


def test_setup_downloads_the_model_with_one_progress_bar(hf_cache, monkeypatch, capsys):
    """setup's download: the Hub's own bars off (one per file, and its downloader's), and
    one bar of ours following the bytes in the cache; a failed download raises."""
    from fourier.cli import setup as S
    seen = {}
    hub = types.ModuleType("huggingface_hub")
    hu = types.ModuleType("huggingface_hub.utils")
    hu.disable_progress_bars = lambda: seen.setdefault("bars_off", True)
    hub.utils = hu

    class Api:
        def model_info(self, repo, revision=None, files_metadata=False):
            return types.SimpleNamespace(siblings=[types.SimpleNamespace(size=3000),
                                                  types.SimpleNamespace(size=1000)])
    hub.HfApi = Api

    def snapshot_download(repo_id, revision):
        blobs = hf_cache / ("models--" + repo_id.replace("/", "--")) / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "abc.incomplete").write_bytes(b"x" * 4000)
        seen["env"] = __import__("os").environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        if seen.get("fail"):
            raise OSError("connection reset")
        return str(hf_cache / "snap")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", hu)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    assert S.download_clap_model() == str(hf_cache / "snap")
    assert seen["bars_off"] and seen["env"] == "1"
    out = capsys.readouterr().out
    assert out.count("CLAP model") == 1 and "Reconstructing" not in out and "4.0" in out
    import shutil
    shutil.rmtree(hf_cache)
    seen["fail"] = True
    with pytest.raises(OSError, match="connection reset"):
        S.download_clap_model()


def test_an_unchanged_index_is_not_written_again(tmp_path, monkeypatch):
    """A build that embedded nothing new leaves the CLAP index's files as they were (the same
    ids and vectors, in any order); a changed embedding writes them again."""
    import numpy as np

    from fourier.db import session as DS
    from fourier.db.models import Sample, SampleFeatures
    monkeypatch.setenv("FOURIER_CLAP_INDEX", str(tmp_path / "idx" / "clap_index.npz"))
    DS._engine = DS._SessionLocal = None
    DS.init_db(tmp_path / "t.db")
    rng = np.random.default_rng(1)
    with DS.session_scope() as s:
        for i in range(1, 6):
            s.add(Sample(id=i, path=f"/l/{i}.wav", filename=f"{i}.wav"))
            s.add(SampleFeatures(sample_id=i, clap_embedding=CF.embedding_to_bytes(rng.standard_normal(512))))
    with DS.session_scope() as s:
        path = CF.build_index(s)
        assert CF.build_index.wrote is True
        files = [path, *CF._fast_paths(path)]
        before = [p.stat().st_mtime_ns for p in files]
        assert CF.build_index(s) == path and CF.build_index.wrote is False
        assert [p.stat().st_mtime_ns for p in files] == before
        s.query(SampleFeatures).filter_by(sample_id=3).update(
            {"clap_embedding": CF.embedding_to_bytes(rng.standard_normal(512))})
    with DS.session_scope() as s:
        CF.build_index(s)
        assert CF.build_index.wrote is True
        ids, emb = CF.load_index()
        got = {int(i): np.asarray(e) for i, e in zip(ids, emb)}
        want = CF.bytes_to_embedding(s.query(SampleFeatures).filter_by(sample_id=3).one().clap_embedding)
        assert np.array_equal(got[3], want)
    DS._engine = DS._SessionLocal = None
