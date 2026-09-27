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
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Sequence

from growbot.config import EMBEDDING_DIM, EMBEDDING_MODEL

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


def get_model():
    """Return the cached SentenceTransformer, loading it on first use.

    Double-checked locking keeps a Streamlit rerender from loading two copies of
    the model into the same process.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = _load_sentence_transformer()

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
