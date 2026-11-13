"""Retrieval quality on FiQA: nDCG@10, Recall@100, MRR@10 per method.

Two scopes:

* corpus: the BYPASSRLS auditor role sees all 57,638 documents, so numbers are
  comparable with published BEIR FiQA results.
* permissioned: each query is asked by its assigned principal through the
  RLS-restricted reader role. Relevance judgments are cut down to documents
  that principal may read (computed by the independent Python oracle);
  queries left with no readable relevant document are excluded and counted.
"""

from __future__ import annotations

import time

from .. import config, corpus, db, embed, retrieval
from ..db import Principal
from . import hardware, write_result
from .metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k

METHODS = ("lexical", "vector", "hybrid", "hybrid_rerank")


def _doc_ranking(hits, doc_of, limit=100):
    out, seen = [], set()
    for h in hits:
        d = doc_of[h.chunk_id]
        if d not in seen:
            seen.add(d)
            out.append(d)
            if len(out) == limit:
                break
    return out


def run(split: str = "test", scopes=("corpus", "permissioned"), methods=METHODS) -> dict:
    queries = corpus.load_queries()
    qrels = corpus.load_qrels(split)
    qids = sorted(qrels, key=lambda q: (len(q), q))
    docs = corpus.load_corpus()
    layout = corpus.build_layout([d.id for d in docs])
    del docs

    t0 = time.perf_counter()
    qvecs = embed.embed_queries([queries[q] for q in qids])
    embed_ms = (time.perf_counter() - t0) * 1000 / len(qids)

    with db.connect(config.AUDITOR_ROLE) as aud:
        doc_of = dict(aud.execute("SELECT id, doc_id FROM chunks").fetchall())

    out: dict = {
        "dataset": f"BEIR FiQA-2018 {split}",
        "queries_total": len(qids),
        "query_embedding_ms_mean_batched": round(embed_ms, 2),
        "settings": {
            "lexical_k": config.RETRIEVAL.lexical_k,
            "vector_k": config.RETRIEVAL.vector_k,
            "rrf_k": config.RETRIEVAL.rrf_k,
            "rerank_top": config.RETRIEVAL.rerank_top,
            "ef_search": config.RETRIEVAL.ef_search,
            "ts_rank_cd_normalization": config.RETRIEVAL.ts_rank_normalization,
            "embedding_model": config.EMBED_MODEL,
            "reranker": config.RERANK_MODEL,
        },
        "hardware": hardware(),
        "scopes": {},
    }

    for scope in scopes:
        role = config.AUDITOR_ROLE if scope == "corpus" else config.READER_ROLE
        per = {m: {"ndcg@10": [], "recall@100": [], "mrr@10": [], "ms": []} for m in methods}
        excluded = 0
        with db.connect(role) as conn:
            for qid, qv in zip(qids, qvecs):
                rel = qrels[qid]
                principal = None
                if scope == "permissioned":
                    t, p = corpus.assign_query_principal(qid, rel, layout)
                    rel = {d: g for d, g in rel.items() if layout.visible(t, p, d)}
                    if not rel:
                        excluded += 1
                        continue
                    principal = Principal(t, p)
                text = queries[qid]
                with db.as_principal(conn, principal):
                    a = time.perf_counter()
                    lex = retrieval.lexical(conn, text)
                    b = time.perf_counter()
                    vec = retrieval.vector(conn, qv)
                    c = time.perf_counter()
                    hyb = retrieval.rrf([[h.chunk_id for h in lex], [h.chunk_id for h in vec]])
                    d = time.perf_counter()
                    rr = retrieval.rerank(conn, text, hyb) if "hybrid_rerank" in methods else None
                    e = time.perf_counter()
                runs = {
                    "lexical": (lex, b - a),
                    "vector": (vec, c - b),
                    "hybrid": (hyb, d - a),
                    "hybrid_rerank": (rr, e - a),
                }
                for m in methods:
                    hits, secs = runs[m]
                    ranked = _doc_ranking(hits, doc_of)
                    per[m]["ndcg@10"].append(ndcg_at_k(ranked, rel, 10))
                    per[m]["recall@100"].append(recall_at_k(ranked, rel, 100))
                    per[m]["mrr@10"].append(mrr_at_k(ranked, rel, 10))
                    per[m]["ms"].append(secs * 1000)
        summary = {}
        for m in methods:
            n = len(per[m]["ms"])
            summary[m] = {
                "queries": n,
                "ndcg@10": round(sum(per[m]["ndcg@10"]) / n, 4),
                "recall@100": round(sum(per[m]["recall@100"]) / n, 4),
                "mrr@10": round(sum(per[m]["mrr@10"]) / n, 4),
                "latency_ms_p50": round(percentile(per[m]["ms"], 50), 1),
                "latency_ms_p95": round(percentile(per[m]["ms"], 95), 1),
            }
        out["scopes"][scope] = {"excluded_no_readable_relevant_doc": excluded, "methods": summary}
        print(scope, summary, flush=True)
    return out


def main(args) -> None:
    scopes = ("corpus", "permissioned") if args.scope == "both" else (args.scope,)
    methods = tuple(m for m in args.methods.split(",") if m)
    res = run(args.split, scopes, methods)
    print("wrote", write_result(f"retrieval_{args.split}", res))
