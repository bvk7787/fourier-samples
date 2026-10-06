"""
Tier 3 — CLAP semantic audio embeddings.

Uses laion/clap-htsat-unfused (via HuggingFace transformers) to produce 512-dim
float32 embeddings for both audio files and text queries, enabling:

  fourier tools analyze --only clap         # embed all samples
  fourier search --like "dark kick" # text → audio cosine search
  fourier search --like-file ref.wav # audio → audio similarity

The search index is cached at ~/.fourier/clap_index.npz as a dense
(N, 512) float32 matrix + matching ID array for fast in-process cosine search.

Optional dependency group [clap] (`fourier setup` installs it; in a checkout of the source: uv sync --extra clap)
  - transformers>=4.40
  - torch>=2.2
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

CLAP_MODEL_ID = "laion/clap-htsat-unfused"
# the model files pinned to one Hugging Face revision, so every install embeds with the same
# weights (the embeddings and the text cache are only comparable within one revision)
CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
CLAP_SAMPLE_RATE = 48_000  # CLAP expects 48 kHz
CLAP_EMBEDDING_DIM = 512
# CLAP hears 10 s: its feature extractor crops anything longer at a random offset
# (`rand_trunc`), so a long file embedded twice came out differently. Fourier gives it the
# first 10 s instead (CLAP_WINDOW_S), and decodes only those: the same file, the same embedding.
CLAP_WINDOW_S = 10.0
CLAP_WINDOW = int(CLAP_WINDOW_S * CLAP_SAMPLE_RATE)
INDEX_FILENAME = "clap_index.npz"
# A file longer than LONG_FILE_S is longer than any category's length limit takes (they run
# to seconds; a phrase or a loop to well under a minute): the quality check reads its first
# LONG_FILE_READ_S instead of decoding all of it (a long recording held whole at 48 kHz runs
# to gigabytes). CLAP reads only its first CLAP_WINDOW_S of any file.
LONG_FILE_S = 600.0
LONG_FILE_READ_S = 60.0


# ---------------------------------------------------------------------------
# Lazy model loader — avoids importing torch/transformers at import time
# ---------------------------------------------------------------------------

def hf_hub_cache() -> Path:
    """The Hugging Face hub cache folder, as huggingface_hub works it out (read here without
    importing it: it reads HF_HUB_OFFLINE when it's imported)."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache"), "huggingface")
    return Path(home).expanduser() / "hub"


def cached_snapshot() -> Path | None:
    """The pinned revision's folder in the Hugging Face cache, when what loading the model and
    its processor reads is there (`fourier setup` downloads it, or the first use)."""
    snap = (hf_hub_cache() / ("models--" + CLAP_MODEL_ID.replace("/", "--")) / "snapshots"
            / CLAP_REVISION)
    have = lambda *names: any((snap / n).is_file() for n in names)
    if (have("config.json") and have("preprocessor_config.json")
            and have("vocab.json", "tokenizer.json") and have("pytorch_model.bin", "model.safetensors")):
        return snap
    return None


def quiet_hub() -> None:
    """Hugging Face's own notices stay out of the output: the Hub's request for a token
    (X-HF-Warning: a download works without one) and Transformers' background conversion of
    the pinned revision's pytorch_model.bin to safetensors, which would ask the Hub to convert
    the model on every load (DISABLE_SAFETENSORS_CONVERSION)."""
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)


def _quiet_transformers() -> None:
    """Transformers' own output stays out of ours: its progress bars ("Loading weights" on
    every load, the Hub's per-file bars) and its notices below errors."""
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        tlog.disable_progress_bar()
    except Exception:            # an older Transformers without these: its bars show
        pass


@lru_cache(maxsize=1)
def _load_model_and_processor():
    """Load the CLAP model + processor once and cache in process memory. Once the pinned
    revision is downloaded it loads from that folder with the Hub offline (HF_HUB_OFFLINE),
    so a load makes no network request at all; the first use downloads it."""
    quiet_hub()
    snap = cached_snapshot()
    if snap is not None:
        # before transformers and huggingface_hub are imported: they read it then
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from transformers import ClapModel, ClapProcessor  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "CLAP needs PyTorch and Transformers (the [clap] extra): `fourier setup` installs them"
        ) from exc
    _quiet_transformers()

    import torch

    if snap is not None:
        try:
            processor = ClapProcessor.from_pretrained(str(snap), local_files_only=True)
            model = ClapModel.from_pretrained(str(snap), local_files_only=True)
        except OSError as e:
            raise OSError(f"the CLAP model in {snap} doesn't load ({e}): run `fourier setup` to "
                          f"download it again") from e
    else:                        # first use: download that revision (about 600 MB)
        logger.info("Downloading CLAP model %s (once, about 600 MB)", CLAP_MODEL_ID)
        processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID, revision=CLAP_REVISION)
        model = ClapModel.from_pretrained(CLAP_MODEL_ID, revision=CLAP_REVISION)
    model.eval()

    # a GPU (MPS or CUDA) when available, else the CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    logger.info("CLAP model loaded on %s", device)
    return model, processor, device


# ---------------------------------------------------------------------------
# Public embedding API
# ---------------------------------------------------------------------------

def _text_cache_path(text: str) -> Path:
    import hashlib
    from ..paths import home_path
    base = os.environ.get("FOURIER_CLAP_TEXT_CACHE") or str(home_path("cache", "clap_text"))
    return Path(base) / (hashlib.sha256(f"{CLAP_MODEL_ID}\n{text}".encode()).hexdigest() + ".npy")


@lru_cache(maxsize=4096)
def _embed_text_cached(text: str) -> bytes:
    p = _text_cache_path(text)
    try:
        return np.load(p).astype(np.float32).tobytes()
    except (OSError, ValueError):
        pass
    vec = _embed_text_model(text)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.stem}.{os.getpid()}.npy")
        np.save(tmp, vec)
        os.replace(tmp, p)
    except OSError:
        pass
    return vec.tobytes()


def embed_text(text: str) -> np.ndarray:
    """
    Encode a natural-language description into a 512-dim L2-normalised embedding.

    Cached on disk (~/.fourier/cache/clap_text) by model and text: a build's few hundred
    phrases then never load the model (no build worker needs to), and the vectors stay
    identical across runs (MPS text encoding isn't bit-stable).

    Args:
        text: e.g. "dark neurofunk kick with sub bass"

    Returns:
        np.ndarray of shape (512,) dtype float32
    """
    return np.frombuffer(_embed_text_cached(text), dtype=np.float32).copy()


def _embed_text_model(text: str) -> np.ndarray:
    import torch

    model, processor, device = _load_model_and_processor()
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    # transformers ≥5.x returns BaseModelOutputWithPooling; ≤4.x returns a tensor
    if hasattr(out, "pooler_output"):
        vec = out.pooler_output[0].cpu().float().numpy()  # (512,)
    else:
        vec = out[0].cpu().float().numpy()
    return _normalise(vec)


def embed_audio_file(path: str | Path) -> np.ndarray:
    """
    Encode an audio file into a 512-dim L2-normalised embedding.

    Resamples to 48 kHz mono using librosa and hears the first CLAP_WINDOW_S (10 s) of the
    file, so a longer one embeds the same every time. Returns zeros on load failure.

    Args:
        path: absolute path to WAV/AIFF/etc.

    Returns:
        np.ndarray of shape (512,) dtype float32
    """
    import torch

    try:
        from ..audioio import load as load_audio
        y, _ = load_audio(path, sr=CLAP_SAMPLE_RATE, mono=True, duration=CLAP_WINDOW_S)
        y = y[:CLAP_WINDOW]           # resampling can leave a sample over: the extractor would crop it
    except Exception as exc:
        logger.warning("CLAP: could not load %s: %s", path, exc)
        return np.zeros(CLAP_EMBEDDING_DIM, dtype=np.float32)

    model, processor, device = _load_model_and_processor()
    # transformers ≥5.0 uses `audio=` (singular); older versions used `audios=`.
    try:
        inputs = processor(
            audio=[y],
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
        ).to(device)
    except TypeError:
        inputs = processor(  # type: ignore[call-arg]
            audios=[y],
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
        ).to(device)
    with torch.no_grad():
        out = model.get_audio_features(**inputs)
    # transformers ≥5.x returns BaseModelOutputWithPooling; ≤4.x returns a tensor
    if hasattr(out, "pooler_output"):
        vec = out.pooler_output[0].cpu().float().numpy()  # (512,)
    else:
        vec = out[0].cpu().float().numpy()
    return _normalise(vec)


def read_seconds(path) -> float | None:
    """How much of a file the quality check reads: LONG_FILE_READ_S of one longer than
    LONG_FILE_S (by its header), else None (all of it; also when the header can't be read)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        seconds = info.frames / info.samplerate
    except Exception:
        return None
    return LONG_FILE_READ_S if seconds > LONG_FILE_S else None


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    """Serialise a float32 embedding to raw bytes for DB BLOB storage."""
    return vec.astype(np.float32).tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialise raw bytes back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Search index — .npz on disk for fast bulk cosine search
# ---------------------------------------------------------------------------

def _index_path() -> Path:
    """$FOURIER_CLAP_INDEX, else <FOURIER_HOME>/clap_index.npz (fourier.paths)."""
    from ..paths import clap_index_path
    path = clap_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_index(session) -> Path:
    """
    Build (or rebuild) the cosine search index from all stored CLAP embeddings.

    Reads (sample_id, clap_embedding) from the DB, stacks into a (N, 512) float32
    matrix, and saves to ~/.fourier/clap_index.npz.

    Returns the path to the written index file.
    """
    from sqlalchemy import text

    rows = session.execute(
        text("SELECT sf.sample_id, sf.clap_embedding FROM sample_features sf "
             "WHERE sf.clap_embedding IS NOT NULL")
    ).fetchall()

    if not rows:
        raise RuntimeError("No CLAP embeddings found. Run `fourier build` (or `fourier tools analyze --only clap`) first.")

    ids = []
    embeddings = []
    for sample_id, blob in rows:
        if blob is None:
            continue
        ids.append(sample_id)
        embeddings.append(bytes_to_embedding(blob))

    ids_arr = np.array(ids, dtype=np.int64)
    emb_arr = np.stack(embeddings, axis=0).astype(np.float32)  # (N, 512)

    path = _index_path()
    if _same_index(path, ids_arr, emb_arr):
        build_index.wrote = False            # type: ignore[attr-defined] # nothing changed: the files stay as they are
        logger.info("CLAP index unchanged: %s (%d embeddings)", path, len(ids))
        return path
    np.savez_compressed(path, ids=ids_arr, embeddings=emb_arr)
    _write_fast_index(path, ids_arr, emb_arr)
    build_index.wrote = True  # type: ignore[attr-defined]
    logger.info("CLAP index written: %s (%d embeddings)", path, len(ids))
    return path


build_index.wrote = None     # type: ignore[attr-defined] # whether the last build_index() wrote the files (False: identical)


def _same_index(path: Path, ids_arr, emb_arr) -> bool:
    """Whether the index on disk holds exactly these embeddings (the same ids, each with the
    same vector, in any order), read from its .npz and its fast copies alike."""
    if not path.exists():
        return False
    try:
        old_ids, old_emb = load_index_fast()
        if len(old_ids) != len(ids_arr):
            return False
        a, b = np.argsort(old_ids, kind="stable"), np.argsort(ids_arr, kind="stable")
        return bool(np.array_equal(old_ids[a], ids_arr[b])
                    and np.array_equal(np.asarray(old_emb)[a], emb_arr[b]))
    except Exception:            # unreadable: write it again
        return False


def _fast_paths(path: Path) -> tuple[Path, Path]:
    return path.with_suffix(".ids.npy"), path.with_suffix(".emb.npy")


def _write_fast_index(path: Path, ids_arr, emb_arr) -> None:
    """Raw .npy copies of the index beside the .npz: memory-mapped in milliseconds, where
    decompressing the .npz is slow (and happens in every build category)."""
    for p, a in zip(_fast_paths(path), (ids_arr, emb_arr)):
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp.npy")
        np.save(tmp, a)
        os.replace(tmp, p)


def load_index_fast(ids_only: bool = False):
    """(ids, embeddings) from the raw .npy copies (embeddings memory-mapped), written from
    the .npz the first time or whenever the .npz is newer. ids_only skips the embeddings."""
    path = _index_path()
    if not path.exists():
        raise FileNotFoundError(
            f"CLAP index not found at {path}. Run `fourier build` (or `fourier tools analyze --only clap`)."
        )
    pi, pe = _fast_paths(path)
    try:
        fresh = min(pi.stat().st_mtime_ns, pe.stat().st_mtime_ns) >= path.stat().st_mtime_ns
    except OSError:
        fresh = False
    if not fresh:
        data = np.load(path, allow_pickle=True)
        ids_arr, emb_arr = data["ids"], data["embeddings"]
        try:
            _write_fast_index(path, ids_arr, emb_arr)
        except OSError:
            return ids_arr, (None if ids_only else emb_arr)
    ids = np.load(pi)
    return ids, (None if ids_only else np.load(pe, mmap_mode="r"))


def load_index() -> tuple[np.ndarray, np.ndarray]:
    """
    Load the cached index from disk.

    Returns:
        (ids, embeddings) where ids is (N,) int64 and embeddings is (N, 512) float32.

    Raises:
        FileNotFoundError if the index hasn't been built yet.
    """
    return load_index_fast()


def search_index(
    query_embedding: np.ndarray,
    top_k: int = 20,
    *,
    allowed_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    """
    Cosine-similarity search over the pre-built index.

    Args:
        query_embedding: (512,) float32, already L2-normalised.
        top_k:           Number of results to return.
        allowed_ids:     If given, restrict results to this set of sample IDs.

    Returns:
        List of (sample_id, cosine_score) sorted descending by score.
    """
    ids, embeddings = load_index()

    if allowed_ids is not None:
        mask = np.isin(ids, list(allowed_ids))
        ids = ids[mask]
        embeddings = embeddings[mask]

    if len(ids) == 0:
        return []

    # embeddings are already L2-normalised; dot product == cosine similarity
    q = _normalise(query_embedding)
    scores = embeddings @ q  # (N,) — fast matmul

    if top_k >= len(scores):
        idx = np.argsort(scores)[::-1]
    else:
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

    return [(int(ids[i]), float(scores[i])) for i in idx]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)
