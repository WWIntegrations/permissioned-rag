"""Access-control evaluation against the live corpus.

1. Leakage: every test query is run as four principals (one per tenant,
   rotating through all 24). Every chunk returned by lexical, vector and
   hybrid retrieval is checked against the Python oracle built from the seed,
   which never consults the database. Forged chunk ids from other tenants are
   also requested directly.
2. Exactness: for a sample, the RLS result must equal what the BYPASSRLS
   auditor gets when it applies the same predicate by hand. This catches
   over-filtering, not only leaks.
3. Default deny: with no identity set, every query returns nothing.
4. Revocation: remove a membership (or a group from a document's ACL),
   COMMIT, and query again at once. The chunk must be gone on that first query.
"""

from __future__ import annotations

import random
import time

from .. import config, corpus, db, embed, retrieval
from ..db import Principal
from . import write_result
from .metrics import percentile

USERS = ["admin", "user1", "user2", "user3", "user4", "user5"]

EXACT_LEXICAL_SQL = """
WITH q AS (SELECT replace(plainto_tsquery('english', %(q)s)::text, ' & ', ' | ')::tsquery AS q)
SELECT c.id FROM chunks c, q
WHERE c.tsv @@ q.q AND c.tenant_id = %(t)s AND c.acl && %(g)s::int[]
ORDER BY ts_rank_cd(c.tsv, q.q, %(norm)s) DESC, c.id
LIMIT %(k)s
"""

EXACT_VECTOR_SQL = """
SELECT c.id FROM chunks c
WHERE c.tenant_id = %(t)s AND c.acl && %(g)s::int[]
ORDER BY c.embedding <=> %(v)s, c.id
LIMIT %(k)s
"""


def leakage(conn, aud_doc_of, layout, qids, queries, qvecs) -> dict:
    checked = violations_tenant = violations_group = runs = 0
    forged_requested = forged_returned_bad = 0
    rng = random.Random(7)
    all_ids = list(aud_doc_of)
    for i, (qid, qv) in enumerate(zip(qids, qvecs)):
        for t in layout.tenants:
            p = USERS[(i + t) % len(USERS)]
            with db.as_principal(conn, Principal(t, p)):
                lex = retrieval.lexical(conn, queries[qid])
                vec = retrieval.vector(conn, qv)
                hyb = retrieval.rrf([[h.chunk_id for h in lex], [h.chunk_id for h in vec]])
                forged = rng.sample(all_ids, 20)
                got = retrieval.chunk_texts(conn, forged)
            runs += 1
            forged_requested += len(forged)
            for h in [*lex, *vec, *hyb]:
                d = aud_doc_of[h.chunk_id]
                checked += 1
                if layout.doc_tenant[d] != t:
                    violations_tenant += 1
                elif not layout.visible(t, p, d):
                    violations_group += 1
            for cid, (d, _) in got.items():
                if not layout.visible(t, p, d):
                    forged_returned_bad += 1
    return {
        "queries": len(qids),
        "principal_runs": runs,
        "returned_chunks_checked": checked,
        "cross_tenant_chunks": violations_tenant,
        "same_tenant_unauthorized_chunks": violations_group,
        "forged_ids_requested": forged_requested,
        "forged_ids_returned_unauthorized": forged_returned_bad,
    }


def exactness(conn, aud, layout, qids, queries, qvecs, n=100) -> dict:
    lex_equal = vec_equal = 0
    vec_overlap = []
    for i, (qid, qv) in enumerate(zip(qids[:n], qvecs[:n])):
        t = layout.tenants[i % len(layout.tenants)]
        p = USERS[i % len(USERS)]
        groups = list(layout.members[(t, p)])
        with db.as_principal(conn, Principal(t, p)):
            lex = [h.chunk_id for h in retrieval.lexical(conn, queries[qid])]
            vec = [h.chunk_id for h in retrieval.vector(conn, qv)]
        with aud.transaction():
            aud.execute("SET LOCAL enable_indexscan = off")  # force exact distance ordering
            want_lex = [r[0] for r in aud.execute(EXACT_LEXICAL_SQL, {"q": queries[qid], "t": t, "g": groups, "norm": config.RETRIEVAL.ts_rank_normalization, "k": config.RETRIEVAL.lexical_k}).fetchall()]
            want_vec = [r[0] for r in aud.execute(EXACT_VECTOR_SQL, {"v": qv, "t": t, "g": groups, "k": config.RETRIEVAL.vector_k}).fetchall()]
        lex_equal += lex == want_lex
        vec_equal += vec == want_vec
        vec_overlap.append(len(set(vec) & set(want_vec)) / max(1, len(want_vec)))
    return {
        "sample": n,
        "lexical_identical_to_hand_filtered": lex_equal,
        "vector_identical_to_exact_hand_filtered": vec_equal,
        "vector_overlap_mean": round(sum(vec_overlap) / len(vec_overlap), 4),
    }


def default_deny(conn, queries, qids, qvecs) -> dict:
    rows = 0
    with db.as_principal(conn, None):
        rows += conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        rows += conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        for qid, qv in zip(qids[:20], qvecs[:20]):
            rows += len(retrieval.lexical(conn, queries[qid])) + len(retrieval.vector(conn, qv))
    return {"rows_returned_without_identity": rows}


def revocation(conn, owner, layout, trials=50) -> dict:
    """Membership revocation and document ACL edits, measured on the live data."""
    rng = random.Random(11)
    cands = []
    for (t, p), groups in layout.members.items():
        if p == "admin":
            continue
        for g in groups:
            cands.append((t, p, g))
    commit_ms, first_query_ms = [], []
    hidden_first = restored = 0
    doc_hidden_first = 0
    for k in range(trials):
        t, p, g = rng.choice(cands)
        cid, content = owner.execute(
            "SELECT id, content FROM chunks WHERE tenant_id = %s AND acl = ARRAY[%s]::int[] "
            "ORDER BY id OFFSET %s LIMIT 1",
            (t, g, rng.randrange(200)),
        ).fetchone()
        owner.commit()
        probe = " ".join(content.split()[:25])
        who = Principal(t, p)
        with db.as_principal(conn, who):
            before = conn.execute("SELECT count(*) FROM chunks WHERE id = %s", (cid,)).fetchone()[0]
        if before != 1:
            raise RuntimeError(f"chunk {cid} should be visible to {who} before the revoke")
        try:
            a = time.perf_counter()
            owner.execute(
                "DELETE FROM memberships WHERE tenant_id = %s AND principal_id = %s AND group_id = %s", (t, p, g)
            )
            owner.commit()
            b = time.perf_counter()
            with db.as_principal(conn, who):
                ids = {h.chunk_id for h in retrieval.lexical(conn, probe)}
                direct = conn.execute("SELECT count(*) FROM chunks WHERE id = %s", (cid,)).fetchone()[0]
            c = time.perf_counter()
            commit_ms.append((b - a) * 1000)
            first_query_ms.append((c - b) * 1000)
            hidden_first += cid not in ids and direct == 0
        finally:
            owner.execute(
                "INSERT INTO memberships (tenant_id, principal_id, group_id) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (t, p, g),
            )
            owner.commit()
        with db.as_principal(conn, who):
            restored += conn.execute("SELECT count(*) FROM chunks WHERE id = %s", (cid,)).fetchone()[0] == 1

        # Same chunk, but this time the document loses the group instead.
        other = next(x for x in layout.groups[t] if x not in layout.members[(t, p)])
        doc_id = owner.execute("SELECT doc_id FROM chunks WHERE id = %s", (cid,)).fetchone()[0]
        try:
            owner.execute("UPDATE documents SET acl = ARRAY[%s]::int[] WHERE id = %s", (other, doc_id))
            owner.commit()
            with db.as_principal(conn, who):
                doc_hidden_first += conn.execute("SELECT count(*) FROM chunks WHERE id = %s", (cid,)).fetchone()[0] == 0
        finally:
            owner.execute("UPDATE documents SET acl = ARRAY[%s]::int[] WHERE id = %s", (g, doc_id))
            owner.commit()
    return {
        "trials": trials,
        "membership_revoke_hidden_on_first_query": hidden_first,
        "membership_restore_visible_again": restored,
        "document_acl_change_hidden_on_first_query": doc_hidden_first,
        "revoke_commit_ms_p50": round(percentile(commit_ms, 50), 2),
        "revoke_commit_ms_p95": round(percentile(commit_ms, 95), 2),
        "first_query_after_commit_ms_p50": round(percentile(first_query_ms, 50), 2),
        "first_query_after_commit_ms_p95": round(percentile(first_query_ms, 95), 2),
        "note": "No cache sits between the policy and the data, so the bound is one query: "
        "the first query that starts after COMMIT returns sees the change.",
    }


def main(args) -> None:
    queries = corpus.load_queries()
    qrels = corpus.load_qrels(args.split)
    qids = sorted(qrels, key=lambda q: (len(q), q))
    docs = corpus.load_corpus()
    layout = corpus.build_layout([d.id for d in docs])
    del docs
    qvecs = embed.embed_queries([queries[q] for q in qids])
    with db.connect(config.AUDITOR_ROLE) as aud, db.connect() as conn, db.connect(config.OWNER_ROLE) as owner:
        aud_doc_of = dict(aud.execute("SELECT id, doc_id FROM chunks").fetchall())
        out = {
            "split": args.split,
            "leakage": leakage(conn, aud_doc_of, layout, qids, queries, qvecs),
            "exactness": exactness(conn, aud, layout, qids, queries, qvecs),
            "default_deny": default_deny(conn, queries, qids, qvecs),
            "revocation": revocation(conn, owner, layout),
        }
    print(out)
    print("wrote", write_result("acl", out))
