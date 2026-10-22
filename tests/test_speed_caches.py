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


def test_a_builds_prompts_are_encoded_in_batches_first(tmp_path, monkeypatch):
    """A first build encodes its few hundred prompts in batches, with progress, before any
    category reads one; the categories then read them from the cache, and the next build
    has nothing to encode."""
    monkeypatch.setenv("FOURIER_CLAP_TEXT_CACHE", str(tmp_path))
    monkeypatch.delenv("FOURIER_CONFIG", raising=False)
    monkeypatch.delenv("FOURIER_HOME", raising=False)
    CF._embed_text_cached.cache_clear()
    batches = []

    def stand_in(texts):                                  # the model, encoding a batch
        batches.append(list(texts))
        return [np.full(512, (hash(t) % 97) / 97, np.float32) for t in texts]
    monkeypatch.setattr(CF, "_embed_text_batch", stand_in)
    said = []
    prompts = C.text_prompts()
    assert len(prompts) > 100
    assert C.prepare_prompts(log=said.append) == len(prompts)
    assert len(batches) == -(-len(prompts) // CF.TEXT_BATCH)
    assert all(1 <= len(b) <= CF.TEXT_BATCH for b in batches)
    assert sorted(t for b in batches for t in b) == sorted(prompts)
    assert said[0].startswith(f"CLAP text prompts: encoding {len(prompts)}")
    assert f"{len(prompts)} of {len(prompts)} prompts" in " ".join(said)
    assert CF.TEXT_PROGRESS["total"] == 0                 # the spinner's line is cleared
    monkeypatch.setattr(CF, "_embed_text_batch", lambda texts: 1 / 0)    # no model from here on
    CF._embed_text_cached.cache_clear()
    v = C.embed_text(prompts[0])
    assert v.dtype == np.float32 and v.shape == (512,)
    assert C.prepare_prompts(log=said.append) == 0


def test_prompts_encoded_where_the_cache_cant_be_written_are_kept(tmp_path, monkeypatch):
    """A home the text cache can't be written to: the batch's vectors stay in memory, so the
    build doesn't encode every prompt a second time, one at a time."""
    monkeypatch.setenv("FOURIER_CLAP_TEXT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("FOURIER_CONFIG", raising=False)
    monkeypatch.delenv("FOURIER_HOME", raising=False)
    monkeypatch.setattr(CF, "_UNSAVED", {})
    monkeypatch.setattr(CF, "_save_text", lambda p, vec: False)          # read-only or full
    CF._embed_text_cached.cache_clear()
    monkeypatch.setattr(CF, "_embed_text_batch",
                        lambda texts: [np.full(512, 0.5, np.float32) for _ in texts])
    prompts = C.text_prompts()[:40]
    assert CF.embed_texts(prompts) == 40
    monkeypatch.setattr(CF, "_embed_text_batch", lambda texts: 1 / 0)    # no model from here on
    v = C.embed_text(prompts[0])
    assert v.shape == (512,) and float(v[0]) == 0.5
    assert CF.embed_texts(prompts) == 0
    CF._embed_text_cached.cache_clear()


def test_a_stand_in_text_encoder_has_no_prompts_to_prepare(tmp_path, monkeypatch):
    monkeypatch.setenv("FOURIER_CLAP_TEXT_CACHE", str(tmp_path))
    monkeypatch.setattr(C, "embed_text", lambda t: np.ones(512, dtype="float32"))
    monkeypatch.setattr(CF, "_embed_text_batch", lambda texts: 1 / 0)
    assert C.prepare_prompts(log=lambda m: None) == 0 and not any(tmp_path.iterdir())


def test_the_build_spinner_shows_the_prompts_being_encoded(monkeypatch):
    from fourier.cli import quiet as Q
    assert Q._text_progress() is None
    monkeypatch.setitem(CF.TEXT_PROGRESS, "total", 600)
    monkeypatch.setitem(CF.TEXT_PROGRESS, "done", 64)
    assert Q._text_progress() == {"done": 64, "total": 600}
