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


# vectors embed_texts encoded but couldn't write to the disk cache (a read-only or full
# home), by cache path: embed_text reads them from here then
_UNSAVED: dict[str, bytes] = {}


def _save_text(p: Path, vec: np.ndarray) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.stem}.{os.getpid()}.npy")
        np.save(tmp, vec)
        os.replace(tmp, p)
        return True
    except OSError:
        return False


def text_cached(text: str) -> bool:
    """Whether embed_text has this text's vector without the model."""
    p = _text_cache_path(text)
    return str(p) in _UNSAVED or p.is_file()


@lru_cache(maxsize=4096)
def _embed_text_cached(text: str) -> bytes:
    p = _text_cache_path(text)
    try:
        return np.load(p).astype(np.float32).tobytes()
    except (OSError, ValueError):
        pass
    if str(p) in _UNSAVED:
        return _UNSAVED[str(p)]
    vec = _embed_text_model(text)
    _save_text(p, vec)
    return vec.tobytes()


TEXT_BATCH = 32                         # prompts encoded at once: about 5x cheaper than one by one
TEXT_PROGRESS = {"done": 0, "total": 0}     # embed_texts' progress, for a build's spinner


def embed_texts(texts, log=None) -> int:
    """Encode the texts the disk cache doesn't hold yet into it, TEXT_BATCH at a time, so
    embed_text then reads each from the cache (or from memory when the cache can't be
    written); log(done, total) after each batch. Returns
    how many it encoded (none in a demo, whose text encoder is a stand-in)."""
    from ..demo import active
    if active():
        return 0
    todo = [t for t in dict.fromkeys(texts) if not text_cached(t)]
    if not todo:
        return 0
    one_by_one = _embed_text_model is not _EMBED_TEXT_MODEL      # a stand-in model: its own way
    TEXT_PROGRESS.update(done=0, total=len(todo))
    try:
        for i in range(0, len(todo), TEXT_BATCH):
            part = todo[i:i + TEXT_BATCH]
            vecs = [_embed_text_model(t) for t in part] if one_by_one else _embed_text_batch(part)
            for t, vec in zip(part, vecs):
                p = _text_cache_path(t)
                if not _save_text(p, vec):
                    _UNSAVED[str(p)] = np.asarray(vec, np.float32).tobytes()
            TEXT_PROGRESS["done"] = i + len(part)
            if log is not None:
                log(i + len(part), len(todo))
    finally:
        TEXT_PROGRESS.update(done=0, total=0)
    return len(todo)


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
    from ..demo import active, fake_embed_text
    if active():            # a demo's analysis is stand-in data, so its text encoder is too
        return fake_embed_text(text)
    return np.frombuffer(_embed_text_cached(text), dtype=np.float32).copy()


_EMBED_TEXT = embed_text


def _embed_text_batch(texts: list[str]) -> list[np.ndarray]:
    """The model's embedding of each text, encoded together (padded to the longest)."""
    import torch

    model, processor, device = _load_model_and_processor()
    inputs = processor(text=list(texts), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    # transformers ≥5.x returns BaseModelOutputWithPooling; ≤4.x returns a tensor
    if hasattr(out, "pooler_output"):
        out = out.pooler_output
    return [_normalise(v) for v in out.cpu().float().numpy()]       # each (512,)


def _embed_text_model(text: str) -> np.ndarray:
    return _embed_text_batch([text])[0]


_EMBED_TEXT_MODEL = _embed_text_model


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


_INDEX_SQL = ("SELECT sf.sample_id, sf.clap_embedding FROM sample_features sf "
              "WHERE sf.clap_embedding IS NOT NULL")
_INDEX_CHUNK = 4096


def _index_rows(session):
    """(sample_id, blob) of every stored embedding, a chunk at a time (nothing is written on
    the session while they're read)."""
    from sqlalchemy import text
    result = session.execute(text(_INDEX_SQL))
    while True:
        rows = result.fetchmany(_INDEX_CHUNK)
        if not rows:
            break
        for sample_id, blob in rows:
            if blob is not None:
                yield sample_id, blob


def _rows_digest(session) -> tuple[str, int, int]:
    """(a digest of every stored (sample id, embedding) in the database's order, how many,
    the embedding's length), read a chunk at a time: what the index was last built from, if
    it matches the digest kept beside it."""
    import hashlib
    h, n, dim = hashlib.sha256(), 0, 0
    for sample_id, blob in _index_rows(session):
        h.update(int(sample_id).to_bytes(8, "little", signed=True))
        h.update(blob)
        n += 1
        dim = dim or len(blob) // 4
    return h.hexdigest(), n, dim


def _digest_path(path: Path) -> Path:
    return path.with_suffix(".digest.json")


def _index_stat(path: Path) -> list[int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return [st.st_size, st.st_mtime_ns]


def _digest_matches(path: Path, digest: str) -> bool:
    """The digest kept beside the index is this one, and the index files are the ones it was
    written with (an older version that rewrote them leaves no matching digest)."""
    import json
    try:
        kept = json.loads(_digest_path(path).read_text())
    except (OSError, ValueError):
        return False
    files = [path, *_fast_paths(path)]
    return (kept.get("digest") == digest
            and kept.get("files") == [_index_stat(p) for p in files]
            and all(st is not None for st in kept.get("files") or [None]))


def _write_digest(path: Path, digest: str) -> None:
    import json
    files = [path, *_fast_paths(path)]
    tmp = _digest_path(path).with_name(f".{_digest_path(path).name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps({"digest": digest, "files": [_index_stat(p) for p in files]}))
        os.replace(tmp, _digest_path(path))
    except OSError:
        pass


def build_index(session) -> Path:
    """
    Build (or rebuild) the cosine search index from all stored CLAP embeddings.

    Reads (sample_id, clap_embedding) from the DB, stacks into a (N, 512) float32
    matrix, and saves to ~/.fourier/clap_index.npz.

    The embeddings are read a chunk at a time, first only to compare a digest of them with
    the one kept beside the index: when they're the ones it holds, nothing more is read or
    written. Otherwise they're read again into one matrix made once.

    Returns the path to the written index file.
    """
    path = _index_path()
    digest, n, dim = _rows_digest(session)
    if not n:
        raise RuntimeError("No CLAP embeddings found. Run `fourier build` (or `fourier tools analyze --only clap`) first.")
    if _digest_matches(path, digest):
        build_index.wrote = False            # type: ignore[attr-defined] # nothing changed: the files stay as they are
        logger.info("CLAP index unchanged: %s (%d embeddings)", path, n)
        return path

    ids_arr = np.empty(n, dtype=np.int64)
    emb_arr = np.empty((n, dim), dtype=np.float32)  # (N, 512)
    k = 0
    for sample_id, blob in _index_rows(session):
        if k == n:                 # more rows than the first read counted: grow
            ids_arr = np.concatenate([ids_arr, np.empty(n, dtype=np.int64)])
            emb_arr = np.concatenate([emb_arr, np.empty((n, dim), dtype=np.float32)])
            n *= 2
        ids_arr[k] = sample_id
        emb_arr[k] = bytes_to_embedding(blob)
        k += 1
    ids_arr, emb_arr = ids_arr[:k], emb_arr[:k]

    if _same_index(path, ids_arr, emb_arr):
        build_index.wrote = False            # type: ignore[attr-defined] # nothing changed: the files stay as they are
        _write_digest(path, digest)
        logger.info("CLAP index unchanged: %s (%d embeddings)", path, k)
        return path
    try:
        _digest_path(path).unlink()
    except OSError:
        pass
    np.savez_compressed(path, ids=ids_arr, embeddings=emb_arr)
    _write_fast_index(path, ids_arr, emb_arr)
    _write_digest(path, digest)
    build_index.wrote = True  # type: ignore[attr-defined]
    logger.info("CLAP index written: %s (%d embeddings)", path, k)
    return path


build_index.wrote = None     # type: ignore[attr-defined] # whether the last build_index() wrote the files (False: identical)

_COMPARE_ROWS = 8192


def _same_index(path: Path, ids_arr, emb_arr) -> bool:
    """Whether the index on disk holds exactly these embeddings (the same ids, each with the
    same vector, in any order), read from its .npz and its fast copies alike. The vectors are
    compared a block at a time (each side's sorted copy of the whole matrix doubled the peak)."""
    if not path.exists():
        return False
    try:
        old_ids, old_emb = load_index_fast()
        if len(old_ids) != len(ids_arr) or old_emb.shape != emb_arr.shape:
            return False
        a, b = np.argsort(old_ids, kind="stable"), np.argsort(ids_arr, kind="stable")
        if not np.array_equal(old_ids[a], ids_arr[b]):
            return False
        return all(np.array_equal(np.asarray(old_emb[a[i:i + _COMPARE_ROWS]]), emb_arr[b[i:i + _COMPARE_ROWS]])
                   for i in range(0, len(a), _COMPARE_ROWS))
    except Exception:            # unreadable: write it again
        return False


def _fast_paths(path: Path) -> tuple[Path, Path]:
    return path.with_suffix(".ids.npy"), path.with_suffix(".emb.npy")


def _unit_path(path: Path, ident: str) -> Path:
    """The unit rows' file for the index matrix file whose identity is `ident` (_file_ident):
    a matrix written again gets another file, so a file from an older matrix is never read."""
    return path.with_name(f"{path.stem}.unit.{ident}.npy")


def _file_ident(p: Path) -> str:
    st = p.stat()
    return f"{st.st_ino}-{st.st_size}-{st.st_mtime_ns}"


def _write_fast_index(path: Path, ids_arr, emb_arr) -> None:
    """Raw .npy copies of the index beside the .npz: memory-mapped in milliseconds, where
    decompressing the .npz is slow (and happens in every build category)."""
    for p, a in zip(_fast_paths(path), (ids_arr, emb_arr)):
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp.npy")
        np.save(tmp, a)
        os.replace(tmp, p)


def unit_rows(emb) -> np.ndarray:
    """emb's rows as unit vectors, emb / (norm + 1e-9) in float32 as curation computes them.
    For the index's own memory-mapped matrix (load_index_fast) they're kept in a file beside
    it, written the first time and memory-mapped after that: build workers then share one
    copy through the page cache where each made its own."""
    got = _unit_index(emb) if isinstance(emb, np.memmap) else None
    if got is not None:
        return got
    return _unit(np.asarray(emb, dtype=np.float32))


def _unit(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)


def _unit_index(emb: np.memmap) -> np.ndarray | None:
    """unit_rows for the fast index matrix emb, from the file for the very matrix file it
    maps (load_index_fast records that file's identity on it; written the first time), or
    None when emb isn't that matrix, its identity isn't known, or the file can't be written
    or read."""
    try:
        ident = getattr(emb, "_fourier_ident", None)
        src = Path(str(emb.filename))
        path = _index_path()
        if not ident or src.resolve() != _fast_paths(path)[1].resolve() or emb.dtype != np.float32:
            return None
        unit = _unit_path(path, ident)
        if unit.exists():
            got = np.load(unit, mmap_mode="r")
            if got.shape == emb.shape and got.dtype == np.float32:
                return got
        tmp = unit.with_name(f".{unit.name}.{os.getpid()}.tmp.npy")
        np.save(tmp, _unit(np.asarray(emb)))
        os.replace(tmp, unit)
        for old in unit.parent.glob(f"{path.stem}.unit.*.npy"):    # an older matrix's
            if old != unit:
                try:
                    old.unlink()
                except OSError:
                    pass
        return np.load(unit, mmap_mode="r")
    except (OSError, ValueError, AttributeError, TypeError):
        return None


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
    if ids_only:
        return ids, None
    # the identity of the file mapped (unchanged while it was mapped), for its unit rows
    try:
        before = _file_ident(pe)
        emb = np.load(pe, mmap_mode="r")
        if _file_ident(pe) == before:
            emb._fourier_ident = before        # type: ignore[attr-defined]
    except OSError:
        emb = np.load(pe, mmap_mode="r")
    return ids, emb


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
