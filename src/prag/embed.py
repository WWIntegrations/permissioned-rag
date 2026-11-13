"""Local embedding and reranking models (loaded lazily, offline only)."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from . import config

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def local_model_path(repo: str) -> str:
    """Resolve a cached Hugging Face repo to its snapshot directory.

    Falls back to the repo id (normal hub resolution) when no snapshot is on
    disk. Some caches lack refs/main, which breaks offline resolution by name.
    """
    from pathlib import Path

    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    snaps = sorted((root / ("models--" + repo.replace("/", "--")) / "snapshots").glob("*/config.json"))
    return str(snaps[-1].parent) if snaps else repo


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBED_MODEL, device=_device())
    model.max_seq_length = 512
    return model


def embed_passages(texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    # E5 models are trained with these exact prefixes.
    return embedder().encode(
        ["passage: " + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    ).astype(np.float32)


def embed_queries(texts: list[str], batch_size: int = 64) -> np.ndarray:
    return embedder().encode(
        ["query: " + t for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


@lru_cache(maxsize=1)
def reranker():
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        local_model_path(config.RERANK_MODEL),
        device=_device(),
        max_length=384,
        model_kwargs={"torch_dtype": torch.float16},
    )


def rerank_scores(query: str, passages: list[str], batch_size: int = 16) -> list[float]:
    if not passages:
        return []
    scores = reranker().predict([(query, p) for p in passages], batch_size=batch_size)
    return [float(s) for s in scores]
