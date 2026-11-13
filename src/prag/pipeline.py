"""Retrieve under the caller's identity, generate, verify citations."""

from __future__ import annotations

import time
from dataclasses import asdict

import psycopg

from . import citations, config, db, embed, retrieval
from .db import Principal
from .generation import Generator, get

CONTEXT_CHUNKS = 5
MAX_ANSWER_TOKENS = 300

SYSTEM = f"""You answer questions about personal finance using only the numbered sources you are given.

Rules:
1. Every sentence that states a fact ends with the id of the source that supports it, written exactly like [#123].
2. Include at least one short exact quote copied character for character from a source, in double quotes, followed by its id: "exact words" [#123].
3. If the sources do not contain enough information to answer, reply with exactly {citations.REFUSAL} and nothing else.
4. Do not use outside knowledge. Do not invent source ids.
Keep the answer under 120 words."""


def build_user_prompt(question: str, context: list[tuple[int, str]]) -> str:
    blocks = "\n\n".join(f"[#{cid}]\n{text}" for cid, text in context)
    return f"Sources:\n\n{blocks}\n\nQuestion: {question}"


def retrieve_context(
    conn: psycopg.Connection, question: str, principal: Principal, rerank: bool = True, k: int = CONTEXT_CHUNKS
) -> tuple[list[tuple[int, str]], float]:
    t0 = time.perf_counter()
    qv = embed.embed_queries([question])[0]
    with db.as_principal(conn, principal):
        hits = retrieval.hybrid(conn, question, qv)
        if rerank:
            hits = retrieval.rerank(conn, question, hits)
        top = [h.chunk_id for h in hits[:k]]
        texts = retrieval.chunk_texts(conn, top)
    return [(cid, texts[cid][1]) for cid in top if cid in texts], time.perf_counter() - t0


def answer(gen: Generator, question: str, context: list[tuple[int, str]]) -> dict:
    g = gen.generate(SYSTEM, build_user_prompt(question, context), MAX_ANSWER_TOKENS)
    verdict = citations.verify(g.text, dict(context))
    return {
        "model": g.model,
        "answer": g.text,
        "verdict": verdict.status,
        "problems": [asdict(p) for p in verdict.problems],
        "cited_ids": verdict.cited_ids,
        "quotes_checked": verdict.quotes_checked,
        "sentence_coverage": verdict.sentence_coverage,
        "input_tokens": g.input_tokens,
        "output_tokens": g.output_tokens,
        "generation_s": round(g.latency_s, 3),
        "stop_reason": g.stop_reason,
    }


def ask(question: str, tenant: int, principal: str, model: str = "qwen3-1.7b", rerank: bool = False) -> dict:
    gen = get(model)
    ok, why = gen.configured()
    if not ok:
        return {"model": model, "error": why}
    with db.connect(config.READER_ROLE) as conn:
        context, rsecs = retrieve_context(conn, question, Principal(tenant, principal), rerank=rerank)
    out = answer(gen, question, context)
    out["retrieval_s"] = round(rsecs, 3)
    out["context_ids"] = [cid for cid, _ in context]
    return out
