"""Load FiQA into Postgres: tenancy layout, documents, chunks, embeddings."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict

import numpy as np
import psycopg

from . import config, corpus
from .corpus import Chunk, Document, Layout


def write_layout(conn: psycopg.Connection, layout: Layout) -> None:
    with conn.transaction():
        for t in layout.tenants:
            conn.execute("INSERT INTO tenants (id, name) VALUES (%s, %s)", (t, f"tenant-{t}"))
            for g in layout.groups[t]:
                conn.execute(
                    "INSERT INTO acl_groups (id, tenant_id, name) VALUES (%s, %s, %s)",
                    (g, t, f"t{t}-g{g % 100}"),
                )
        for (t, p), groups in sorted(layout.members.items()):
            conn.execute("INSERT INTO principals (tenant_id, id) VALUES (%s, %s)", (t, p))
            for g in groups:
                conn.execute(
                    "INSERT INTO memberships (tenant_id, principal_id, group_id) VALUES (%s, %s, %s)",
                    (t, p, g),
                )


def write_documents(conn: psycopg.Connection, docs: list[Document], layout: Layout) -> None:
    with conn.transaction(), conn.cursor().copy(
        "COPY documents (id, tenant_id, acl, title) FROM STDIN"
    ) as cp:
        for d in docs:
            cp.write_row((d.id, layout.doc_tenant[d.id], list(layout.doc_acl[d.id]), d.title))


def write_chunks(conn: psycopg.Connection, chunks: list[Chunk], emb: np.ndarray, batch: int = 5000) -> None:
    """COPY chunks in batches. tenant_id and acl are filled by the chunks_inherit_acl trigger."""
    assert len(chunks) == len(emb)
    for start in range(0, len(chunks), batch):
        with conn.transaction(), conn.cursor().copy(
            "COPY chunks (id, doc_id, ord, tenant_id, acl, content, embedding) FROM STDIN WITH (FORMAT BINARY)"
        ) as cp:
            cp.set_types(["int8", "text", "int4", "int4", "int4[]", "text", "vector"])
            for c, e in zip(chunks[start : start + batch], emb[start : start + batch]):
                cp.write_row((c.id, c.doc_id, c.ord, 0, [0], c.content, e))


def chunk_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    h.update(config.EMBED_MODEL.encode())
    for c in chunks:
        h.update(f"{c.id}|{c.doc_id}|{c.ord}|".encode())
        h.update(c.content.encode())
    return h.hexdigest()[:16]


def embeddings_for(chunks: list[Chunk]) -> tuple[np.ndarray, dict]:
    """Embed every chunk, caching the matrix on disk keyed by a content fingerprint."""
    from . import embed

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fp = chunk_fingerprint(chunks)
    path = config.CACHE_DIR / f"chunk_embeddings_{fp}.npy"
    if path.exists():
        return np.load(path), {"cached": True, "fingerprint": fp}
    t0 = time.perf_counter()
    emb = embed.embed_passages([c.content for c in chunks], batch_size=64, show_progress=True)
    secs = time.perf_counter() - t0
    np.save(path, emb)
    return emb, {"cached": False, "fingerprint": fp, "seconds": round(secs, 1), "per_second": round(len(chunks) / secs, 1)}


def run_ingest() -> dict:
    docs = corpus.load_corpus()
    layout = corpus.build_layout([d.id for d in docs])
    chunks = list(corpus.iter_chunks(docs))
    emb, emb_info = embeddings_for(chunks)
    stats = {
        "documents": len(docs),
        "chunks": len(chunks),
        "docs_without_text": sum(1 for d in docs if not d.text.strip()),
        "embedding": emb_info,
        "spec": asdict(config.CORPUS),
    }
    with psycopg.connect(config.dsn(config.OWNER_ROLE)) as conn:
        from pgvector.psycopg import register_vector

        register_vector(conn)
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if n:
            raise RuntimeError(f"chunks already holds {n} rows; ingest expects an empty database")
        t0 = time.perf_counter()
        write_layout(conn, layout)
        write_documents(conn, docs, layout)
        t1 = time.perf_counter()
        write_chunks(conn, chunks, emb)
        t2 = time.perf_counter()
        conn.execute("ANALYZE")
        conn.commit()
        stats["load_seconds"] = {"layout_and_documents": round(t1 - t0, 1), "chunks_with_indexes": round(t2 - t1, 1)}
        stats["table_sizes"] = dict(
            conn.execute(
                "SELECT relname, pg_size_pretty(pg_total_relation_size(oid)) FROM pg_class "
                "WHERE relname IN ('chunks', 'documents', 'chunks_embedding_hnsw', 'chunks_tsv_gin')"
            ).fetchall()
        )
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "ingest.json").write_text(json.dumps(stats, indent=2))
    return stats
