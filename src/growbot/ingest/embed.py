"""Phase 3 - Embedding (ingestion step 3).

MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, 384-d) encodes **chunks only**
- never a whole page - because five HDFC schemes share almost identical wording
and a page-level vector blurs them together (architecture §5.3).

This module is the single embedding helper for the whole project. Ingestion
embeds chunks here and Phase 6's retriever embeds the user question through the
same `embed_query`, so the two paths cannot drift onto different models or
different normalisation settings - the classic cause of a query that silently
stops matching its own index.

The model is loaded once per process and cached, so a chat UI does not pay the
load cost on every question (architecture §11 / Phase 11).

**Two runtimes, one vector.** `config.EMBEDDING_BACKEND` picks which one
executes the encoder: `onnx` (onnxruntime) or `torch` (sentence-transformers).
The weights are identical and so is the output - measured cosine 1.0000000000,
max elementwise difference 1.0e-07, `allclose(atol=1e-5)` - in the raw
un-normalised form `embed_texts` actually uses. That is float32 rounding, far
below the precision `SIMILARITY_FLOOR` is specified to, so switching backends
does not move a score across the floor and does not require re-ingesting the
corpus. `retrieve.checks` re-measures this on every run rather than leaving it
as a comment.

`onnx` is the default because of memory, not speed. Loading torch to evaluate a
22M-parameter model costs ~500 MB of runtime before a single weight is used;
measured peak RSS for a full `ask()` is 579.7 MB on torch versus 238.7 MB on
onnxruntime, which is the difference between fitting a 512 MB container and
being OOM-killed inside it. torch stays selectable because it is the reference
the onnx path is checked against, and because it batches faster at ingest time.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Sequence

from growbot.config import EMBEDDING_BACKEND, EMBEDDING_DIM, EMBEDDING_MODEL

#: The only model chromadb's bundled onnxruntime encoder implements. Its
#: `__init__` accepts only `preferred_providers` - there is no model argument -
#: so it cannot honour a custom `EMBEDDING_MODEL`. See `_load_onnx`.
_ONNX_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Set before sentence-transformers is imported anywhere, so the weight-loading
# progress bar does not scribble over the demo output. The model is loaded
# lazily, so doing it here covers every entry point - CLIs, the UI and the
# tests - from one place. HF_HUB_DISABLE_TELEMETRY is unrelated to the bar but
# is silenced here for the same reason.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

log = logging.getLogger("growbot.ingest.embed")

#: Chunks per forward pass. Small enough to stay responsive on CPU.
DEFAULT_BATCH_SIZE = 32

_model = None
_model_lock = threading.Lock()


def _model_is_cached() -> bool:
    """True when the model is already in the local HuggingFace cache.

    Checked on the filesystem rather than with ``snapshot_download`` on purpose:
    asking the hub anything constructs an HTTP client, which prints an
    unauthenticated-access notice to stdout on every run. A directory check
    answers the same question with no side effects.

    Offline is the right default once the model is on disk: startup is faster,
    and it makes the Phase 6 claim that retrieval never touches the network true
    in practice rather than by convention.
    """
    cache_root = os.getenv("HF_HUB_CACHE")
    if not cache_root:
        home = os.getenv("HF_HOME")
        base = Path(home) if home else Path.home() / ".cache" / "huggingface"
        cache_root = str(base / "hub")
    snapshots = Path(cache_root) / f"models--{EMBEDDING_MODEL.replace('/', '--')}"
    try:
        return any(snapshots.joinpath("snapshots").iterdir())
    except OSError:
        return False


def _prefer_offline_if_cached() -> bool:
    """Set ``HF_HUB_OFFLINE`` when the model is already on disk.

    Must run *before* ``sentence_transformers``/``huggingface_hub`` are first
    imported. The unauthenticated-access notice is emitted while the hub
    initialises on import, not when the model is fetched, so setting the flag
    after the import is too late to suppress it. Offline also means startup
    makes no network call, which is what lets Phase 6 honestly claim retrieval
    never touches the network.
    """
    if not _model_is_cached():
        return False
    os.environ["HF_HUB_OFFLINE"] = "1"
    return True


def _load_sentence_transformer():
    """Import and construct the model, preferring the local cache over the network."""
    offline = _prefer_offline_if_cached()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - setup guidance
        raise ImportError(
            "sentence-transformers is not installed. "
            'Run: pip install -e ".[rag]"'
        ) from exc

    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001 - fall back to a real fetch
        if not offline:
            raise
        # The cache directory existed but the model was not actually usable
        # (partial download, missing weights). Clear offline and retry online.
        log.warning("offline load failed (%s); retrying online", exc)
        os.environ["HF_HUB_OFFLINE"] = "0"
        model = SentenceTransformer(EMBEDDING_MODEL)

    if offline:
        log.info("loaded %s from the local cache (offline)", EMBEDDING_MODEL)
    else:
        log.info("downloaded and loaded %s", EMBEDDING_MODEL)
    return model


class _OnnxEncoder:
    """Presents chroma's onnxruntime encoder with the `.encode()` shape used here.

    chromadb ships `ONNXMiniLM_L6_V2` because onnxruntime is already one of its
    own dependencies - so this backend needs no package that is not installed
    alongside chromadb, and no `optimum`. It is called as `ef(texts)` rather
    than `.encode(...)`, hence this adapter. `batch_size` is accepted and
    ignored: chroma batches internally.
    """

    def __init__(self, function) -> None:
        self._function = function

    def encode(self, texts, **_kwargs):
        import numpy as np

        vectors = self._function(list(texts))
        # float32 explicitly: cos similarity is computed against an index built
        # in float32, and a widened float64 here would spend memory for nothing.
        return np.asarray(vectors, dtype="float32")

    def get_sentence_embedding_dimension(self) -> int:
        return EMBEDDING_DIM

    get_embedding_dimension = get_sentence_embedding_dimension


def _load_onnx():
    """Load MiniLM through onnxruntime, without importing torch.

    Falls back to torch when `EMBEDDING_MODEL` names something other than the
    one model chroma implements. That is a correctness guard, not a
    convenience: chroma's encoder takes no model argument, so obeying
    `EMBEDDING_BACKEND=onnx` with a custom model would silently encode every
    chunk with MiniLM while `config` reported a different model - the precise
    drift this module exists to prevent, and invisible until a similarity score
    quietly stopped matching its index. Memory is worth less than being right.
    """
    if EMBEDDING_MODEL != _ONNX_MODEL_NAME:
        log.warning(
            "EMBEDDING_BACKEND=onnx cannot serve EMBEDDING_MODEL=%r (chroma's "
            "encoder implements %r only); using torch for this process",
            EMBEDDING_MODEL,
            _ONNX_MODEL_NAME,
        )
        return _load_sentence_transformer()

    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    except ImportError as exc:  # pragma: no cover - setup guidance
        raise ImportError(
            "the onnx embedding backend needs chromadb. "
            'Run: pip install -e ".[rag]"'
        ) from exc

    model = _OnnxEncoder(ONNXMiniLM_L6_V2())
    log.info("loaded %s via onnxruntime (torch not imported)", EMBEDDING_MODEL)
    return model


def _load_model():
    """Construct the encoder named by `config.EMBEDDING_BACKEND`."""
    if EMBEDDING_BACKEND == "onnx":
        return _load_onnx()
    return _load_sentence_transformer()


def get_model():
    """Return the cached encoder, loading it on first use.

    Double-checked locking keeps a Streamlit rerender from loading two copies of
    the model into the same process.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = _load_model()

                # sentence-transformers 6.x renamed this accessor; support both
                # so the code does not break on the next major release.
                get_dimension = getattr(_model, "get_embedding_dimension", None)
                if get_dimension is None:
                    get_dimension = _model.get_sentence_embedding_dimension
                dimension = get_dimension()
                if dimension != EMBEDDING_DIM:
                    raise ValueError(
                        f"{EMBEDDING_MODEL} produced {dimension}-d vectors but "
                        f"config expects {EMBEDDING_DIM}-d. Change EMBEDDING_MODEL "
                        "and rebuild the index."
                    )
    return _model


#: The torch encoder, cached apart from `_model` so that asking for the
#: reference implementation never displaces the one the app is using.
_torch_model = None


def get_torch_model():
    """The torch encoder specifically, whatever `EMBEDDING_BACKEND` says.

    Exists so `retrieve.checks` can hold the two runtimes against each other on
    demand. Nothing on the serving path calls this.
    """
    global _torch_model
    if _torch_model is None:
        with _model_lock:
            if _torch_model is None:
                _torch_model = _load_sentence_transformer()
    return _torch_model


def embed_texts(
    texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
) -> list[list[float]]:
    """Encode chunk texts into 384-d vectors.

    Vectors are returned raw (not L2-normalised) because the collection is
    created with cosine distance, which already normalises internally. Phase 6
    therefore reads similarity as ``1 - distance``.
    """
    if not texts:
        return []

    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def embed_query(question: str) -> list[float]:
    """Encode a single user question with the same model as ingestion."""
    if not question or not question.strip():
        raise ValueError("cannot embed an empty question")
    return embed_texts([question])[0]
