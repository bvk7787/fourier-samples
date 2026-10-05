"""Test the instrument-library cap: orchestral/keys multisample packs are capped
to a spread of INSTRUMENT_CAP samples; other packs pass through untouched."""
import numpy as np

from fourier.packs import curate as C


def _make(pack, n, start):
    return [dict(pack=pack, row=start + i) for i in range(n)]


def test_small_instrument_pack_not_upsampled():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((50, 16)).astype(np.float32)
    rec = _make("Keys Pack", 12, 0)  # below cap
    out = C._cap_instrument_packs(rec, emb)
    assert len(out) == 12
