"""Choose the lexical ranking variant on the FiQA *dev* split (test is never used here).

Postgres full-text ranking has no inverse document frequency, so common
finance words ("money", "tax", "account") dominate an OR query. This compares
ts_rank_cd normalization flags and an IDF-style pruning of the query: terms
whose document frequency in the chunk table exceeds a threshold are dropped
before the OR query is built. Corpus scope (auditor role), lexical only.

    python scripts/tune_lexical_dev.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from prag import config, corpus, db  # noqa: E402
from prag.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402

SQL = """
WITH terms AS (
    SELECT t.lexeme FROM unnest(to_tsvector('english', %(q)s)) t
    LEFT JOIN lexeme_df s ON s.lexeme = t.lexeme
    WHERE coalesce(s.df_frac, 0) <= %(max_df)s
), q AS (
    SELECT coalesce(string_agg(quote_literal(lexeme), ' | '), '')::tsquery AS q FROM terms
)
SELECT c.doc_id FROM chunks c, q
WHERE c.tsv @@ q.q
ORDER BY {rank} DESC, c.id
LIMIT 200
"""


def main() -> None:
    queries = corpus.load_queries()
    qrels = corpus.load_qrels("dev")
    qids = sorted(qrels)
    with db.connect(config.OWNER_ROLE) as own:
        own.execute(
            "CREATE TEMP TABLE lexeme_df AS SELECT word AS lexeme, ndoc::float8 / (SELECT count(*) FROM chunks) AS df_frac "
            "FROM ts_stat('SELECT tsv FROM chunks')"
        )
        own.execute("CREATE INDEX ON lexeme_df (lexeme)")
        # norm=32 maps rank to rank/(rank+1), a monotone change that cannot alter the order.
        variants = [(f"ts_rank_cd norm={n}", f"ts_rank_cd(c.tsv, q.q, {n})", 1.0) for n in (0, 1, 2)]
        variants += [
            (f"ts_rank_cd norm=1 max_df={m}", "ts_rank_cd(c.tsv, q.q, 1)", m) for m in (0.01, 0.03, 0.1)
        ]
        out = {}
        for name, rank, max_df in variants:
            nd, rc = [], []
            for qid in qids:
                rows = own.execute(SQL.format(rank=rank), {"q": queries[qid], "max_df": max_df}).fetchall()
                ranked, seen = [], set()
                for (d,) in rows:
                    if d not in seen:
                        seen.add(d)
                        ranked.append(d)
                nd.append(ndcg_at_k(ranked, qrels[qid], 10))
                rc.append(recall_at_k(ranked, qrels[qid], 100))
            out[name] = {"ndcg@10": round(sum(nd) / len(nd), 4), "recall@100": round(sum(rc) / len(rc), 4)}
            print(name, out[name], flush=True)
        own.rollback()
    (config.RESULTS_DIR / "lexical_tuning_dev.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
