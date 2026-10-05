"""Build-speed caches: folder descriptions, CLAP text embeddings, the memory-mapped index."""
import json

import numpy as np

from fourier.analysis import clap_features as CF
from fourier.packs import curate as C


def test_describe_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_DESCRIBE_CACHE", str(tmp_path))
    calls = []

    class R:
        def read(self):
            return json.dumps({"response": json.dumps({"d": {"dark-clean": "dark clean kicks"}})}).encode()

    def fake(req, timeout=0):
        calls.append(json.loads(req.data)["options"])
        return R()
    import fourier.net as NET
    monkeypatch.setattr(NET, "local_opener", lambda: type("O", (), {"open": staticmethod(fake)})())
    cl = [dict(name="dark-clean", source=None, share=0.2, clap_phrase="dark kick", traits=["dark"])]
    assert C._describe(cl, "KICKS", "m") == {"dark-clean": "dark clean kicks"}
    assert C._describe(cl, "KICKS", "m") == {"dark-clean": "dark clean kicks"}
    assert len(calls) == 1 and calls[0]["seed"] == C.CURATION_SEED
    cl[0]["traits"] = ["bright"]                          # a changed prompt asks again
    C._describe(cl, "KICKS", "m")
    assert len(calls) == 2


def test_text_embeddings_cached_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_CLAP_TEXT_CACHE", str(tmp_path))
    CF._embed_text_cached.cache_clear()
    n = []
    monkeypatch.setattr(CF, "_embed_text_model", lambda t: (n.append(t), np.full(512, .5, np.float32))[1])
    a = CF.embed_text("dark kick")
    CF._embed_text_cached.cache_clear()                   # a new process: disk, not the model
    b = CF.embed_text("dark kick")
    assert n == ["dark kick"] and np.array_equal(a, b) and a.dtype == np.float32
    CF._embed_text_cached.cache_clear()


def test_fast_index_follows_the_npz(tmp_path, monkeypatch):
    monkeypatch.setattr(CF, "_index_path", lambda: tmp_path / "clap_index.npz")
    ids, emb = np.arange(4, dtype=np.int64), np.eye(4, 512, dtype=np.float32)
    np.savez_compressed(tmp_path / "clap_index.npz", ids=ids, embeddings=emb)
    i2, e2 = CF.load_index_fast()
    assert np.array_equal(i2, ids) and np.array_equal(np.asarray(e2), emb)
    assert isinstance(e2, np.memmap) and CF.load_index_fast(ids_only=True)[1] is None
    import os
    import time
    time.sleep(0.01)
    np.savez_compressed(tmp_path / "clap_index.npz", ids=ids[:2], embeddings=emb[:2])   # rebuilt
    os.utime(tmp_path / "clap_index.npz")
    assert len(CF.load_index_fast()[0]) == 2
