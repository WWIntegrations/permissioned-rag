"""Lexical, vector and hybrid retrieval. All filtering happens in Postgres.

Every query here runs on a connection whose transaction already carries the
caller identity (see db.as_principal). There is no access check in Python:
rows the caller may not read never leave the database, so they cannot take a
rank position, influence fusion, or reach the reranker.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import psycopg

from . import config

# OR-query over the stemmed terms of the question. plainto_tsquery would AND
# them, which returns almost nothing for natural-language questions. The cast
# from text keeps the already-stemmed lexemes as they are.
LEXICAL_SQL = """
WITH q AS (
    SELECT replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery AS q
)
SELECT c.id, ts_rank_cd(c.tsv, q.q, %(norm)s) AS score
FROM chunks c, q
WHERE c.tsv @@ q.q
ORDER BY score DESC, c.id
LIMIT %(k)s
"""

# relaxed_order iterative HNSW scans (set in db.as_principal) keep scanning the
# graph until `k` rows pass the RLS filter; results can be slightly out of
# order, so they are re-sorted here.
VECTOR_SQL = """
SELECT id, dist FROM (
    SELECT c.id, c.embedding <=> %(v)s AS dist
    FROM chunks c
    ORDER BY c.embedding <=> %(v)s
    LIMIT %(k)s
) t
ORDER BY dist, id
"""

CHUNK_TEXT_SQL = "SELECT id, doc_id, content FROM chunks WHERE id = ANY(%(ids)s)"


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    score: float


def lexical(conn: psycopg.Connection, query: str, k: int = config.RETRIEVAL.lexical_k) -> list[Hit]:
    rows = conn.execute(
        LEXICAL_SQL, {"q": query, "k": k, "norm": config.RETRIEVAL.ts_rank_normalization}
    ).fetchall()
    return [Hit(int(i), float(s)) for i, s in rows]


def vector(conn: psycopg.Connection, qvec: np.ndarray, k: int = config.RETRIEVAL.vector_k) -> list[Hit]:
    rows = conn.execute(VECTOR_SQL, {"v": qvec, "k": k}).fetchall()
    return [Hit(int(i), 1.0 - float(d)) for i, d in rows]


def rrf(rankings: Sequence[Sequence[int]], k: int = config.RETRIEVAL.rrf_k) -> list[Hit]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank), rank from 1.

    Ties are broken by the smallest id so the output is deterministic.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        seen: set[int] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # a list may not vote twice for the same item
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return [Hit(i, s) for i, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def hybrid(conn: psycopg.Connection, query: str, qvec: np.ndarray) -> list[Hit]:
    lex = lexical(conn, query)
    vec = vector(conn, qvec)
    return rrf([[h.chunk_id for h in lex], [h.chunk_id for h in vec]])


def chunk_texts(conn: psycopg.Connection, ids: Sequence[int]) -> dict[int, tuple[str, str]]:
    """id -> (doc_id, content). Still under RLS: a forged id returns nothing."""
    rows = conn.execute(CHUNK_TEXT_SQL, {"ids": list(ids)}).fetchall()
    return {int(i): (d, c) for i, d, c in rows}


def rerank(conn: psycopg.Connection, query: str, hits: list[Hit], top: int = config.RETRIEVAL.rerank_top) -> list[Hit]:
    """Cross-encoder rerank of the first `top` hits; the tail keeps its order below them."""
    from . import embed

    head, tail = hits[:top], hits[top:]
    texts = chunk_texts(conn, [h.chunk_id for h in head])
    head = [h for h in head if h.chunk_id in texts]
    scores = embed.rerank_scores(query, [texts[h.chunk_id][1] for h in head])
    order = sorted(zip(head, scores), key=lambda hs: (-hs[1], hs[0].chunk_id))
    return [Hit(h.chunk_id, s) for h, s in order] + tail


def to_documents(conn: psycopg.Connection, hits: list[Hit], limit: int = 100) -> list[str]:
    """Collapse a chunk ranking to a document ranking (first occurrence wins)."""
    if not hits:
        return []
    ids = [h.chunk_id for h in hits]
    doc_of = dict(conn.execute("SELECT id, doc_id FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall())
    out: list[str] = []
    seen: set[str] = set()
    for i in ids:
        d = doc_of.get(i)
        if d is not None and d not in seen:
            seen.add(d)
            out.append(d)
            if len(out) == limit:
                break
    return out
