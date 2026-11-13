"""Concurrent retrieval load test.

Each client is a thread with its own prag_reader connection. A request is one
permissioned hybrid retrieval: set identity, lexical top-200, vector top-200,
RRF, in one transaction, for a FiQA test query asked by its assigned principal.
Query embeddings are computed before the clock starts, so the numbers measure
the database path; query embedding cost is reported separately. Reranking and
generation are not part of this test.
"""

from __future__ import annotations

import threading
import time

from .. import config, corpus, db, embed, retrieval
from ..db import Principal
from . import hardware, write_result
from .metrics import percentile


def _worker(stop_at: float, warm_until: float, jobs, start_idx: int, lat: list, errors: list) -> None:
    conn = db.connect(config.READER_ROLE)
    i = start_idx
    try:
        while True:
            now = time.perf_counter()
            if now >= stop_at:
                break
            text, qv, principal = jobs[i % len(jobs)]
            i += 1
            t0 = time.perf_counter()
            try:
                with db.as_principal(conn, principal):
                    retrieval.hybrid(conn, text, qv)
            except Exception as exc:  # noqa: BLE001 - counted and reported
                errors.append(repr(exc))
                continue
            t1 = time.perf_counter()
            if t0 >= warm_until:
                lat.append((t1 - t0) * 1000)
    finally:
        conn.close()


def run(levels: list[int], duration: float, warmup: float = 3.0) -> dict:
    queries = corpus.load_queries()
    qrels = corpus.load_qrels("test")
    docs = corpus.load_corpus()
    layout = corpus.build_layout([d.id for d in docs])
    del docs
    qids = sorted(qrels, key=lambda q: (len(q), q))
    vecs = embed.embed_queries([queries[q] for q in qids])
    # Single-query embedding latency, the way a live request would pay it.
    single = []
    for q in qids[:50]:
        t0 = time.perf_counter()
        embed.embed_queries([queries[q]])
        single.append((time.perf_counter() - t0) * 1000)
    jobs = [
        (queries[q], v, Principal(*corpus.assign_query_principal(q, qrels[q], layout)))
        for q, v in zip(qids, vecs)
    ]
    out = {
        "request": "permissioned hybrid retrieval (lexical 200 + vector 200 + RRF), one transaction",
        "duration_s_per_level": duration,
        "warmup_s": warmup,
        "query_embedding_ms_single": {"p50": round(percentile(single, 50), 2), "p95": round(percentile(single, 95), 2)},
        "hardware": hardware(),
        "postgres": {},
        "levels": {},
    }
    with db.connect(config.OWNER_ROLE) as c:
        for name in ("server_version", "shared_buffers", "work_mem", "max_connections"):
            out["postgres"][name] = c.execute(f"SHOW {name}").fetchone()[0]
    for n in levels:
        lat: list[float] = []
        errors: list[str] = []
        start = time.perf_counter()
        warm_until = start + warmup
        stop_at = warm_until + duration
        threads = [
            threading.Thread(target=_worker, args=(stop_at, warm_until, jobs, k * 37, lat, errors))
            for k in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out["levels"][str(n)] = {
            "clients": n,
            "requests": len(lat),
            "errors": len(errors),
            "qps": round(len(lat) / duration, 1),
            "p50_ms": round(percentile(lat, 50), 1),
            "p95_ms": round(percentile(lat, 95), 1),
            "p99_ms": round(percentile(lat, 99), 1),
        }
        print(out["levels"][str(n)], flush=True)
    return out


def main(args) -> None:
    levels = [int(x) for x in args.concurrency.split(",")]
    res = run(levels, args.duration)
    print("wrote", write_result("load", res))
