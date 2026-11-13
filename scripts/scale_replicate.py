"""Grow the chunk table to about 1M rows by synthetic replication. NOT run for the published results.

How it works
------------
The 76,026 FiQA chunks are copied `copies` times. Copy r (1-based) goes to
new tenants (original tenant + 4r), with group ids shifted the same way, new
principals mirroring the original six per tenant, and document ids suffixed
`~r`. Each copied embedding gets Gaussian noise (sigma 0.02 per dimension)
and is re-normalized, so copies are near-duplicates instead of exact ties,
which would otherwise make HNSW and RRF behave unrealistically. Text is
copied unchanged, so lexical scores tie across copies; the RLS filter keeps
each tenant's view the same size as before, which is the point of the test:
how latency changes when the table grows 13x while the per-tenant slice
does not.

With copies=12 the table holds 76,026 * 13 = 988,338 chunks.

Disk
----
At ingest, 76k chunks took about 0.6 GB including indexes (table 406 MB with
TOAST and GIN, HNSW 148 MB). Linear extrapolation gives about 7.2 GB for
988k chunks, plus WAL during the load. The script refuses to start unless
free space minus that estimate stays above --min-free-gb (default 20), and it
re-checks free space after every batch and stops cleanly if the margin is
crossed.

Usage
-----
    python scripts/scale_replicate.py --copies 12 --dry-run   # estimate only
    python scripts/scale_replicate.py --copies 12             # write
"""

from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np

sys.path.insert(0, "src")

from prag import config, db  # noqa: E402

BYTES_PER_CHUNK = 0.6e9 / 76_026  # measured: table + TOAST + GIN + HNSW at ingest
NOISE = 0.02
BATCH = 5000


def free_gb() -> float:
    return shutil.disk_usage(config.ROOT).free / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=12)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()

    with db.connect(config.OWNER_ROLE) as conn:
        base = conn.execute("SELECT count(*) FROM chunks WHERE doc_id NOT LIKE '%%~%%'").fetchone()[0]
        n_tenants = conn.execute("SELECT count(*) FROM tenants WHERE id <= 4").fetchone()[0]
    new_rows = base * args.copies
    need_gb = new_rows * BYTES_PER_CHUNK / 1e9
    print(f"base chunks {base}, copies {args.copies}, new rows {new_rows}, total {base + new_rows}")
    print(f"estimated extra disk {need_gb:.1f} GB, free now {free_gb():.1f} GB, floor {args.min_free_gb} GB")
    if free_gb() - need_gb < args.min_free_gb:
        sys.exit("refusing: the estimate would take free space below the floor")
    if args.dry_run:
        return

    rng = np.random.default_rng(args.seed)
    with db.connect(config.OWNER_ROLE) as conn:
        for r in range(1, args.copies + 1):
            shift = n_tenants * r
            with conn.transaction():
                conn.execute(
                    "INSERT INTO tenants SELECT id + %(s)s, name || '-copy' || %(r)s FROM tenants WHERE id <= %(n)s",
                    {"s": shift, "r": r, "n": n_tenants},
                )
                conn.execute(
                    "INSERT INTO acl_groups SELECT id + 100 * %(s)s, tenant_id + %(s)s, name || '-copy' || %(r)s "
                    "FROM acl_groups WHERE tenant_id <= %(n)s",
                    {"s": shift, "r": r, "n": n_tenants},
                )
                conn.execute(
                    "INSERT INTO principals SELECT tenant_id + %(s)s, id FROM principals WHERE tenant_id <= %(n)s",
                    {"s": shift, "n": n_tenants},
                )
                conn.execute(
                    "INSERT INTO memberships SELECT tenant_id + %(s)s, principal_id, group_id + 100 * %(s)s "
                    "FROM memberships WHERE tenant_id <= %(n)s",
                    {"s": shift, "n": n_tenants},
                )
                conn.execute(
                    "INSERT INTO documents SELECT id || '~' || %(r)s, tenant_id + %(s)s, "
                    "(SELECT array_agg(g + 100 * %(s)s) FROM unnest(acl) g), title "
                    "FROM documents WHERE id NOT LIKE '%%~%%'",
                    {"s": shift, "r": r},
                )
            offset = 10_000_000 * r
            last = 0
            while True:
                rows = conn.execute(
                    "SELECT id, doc_id, ord, content, embedding FROM chunks "
                    "WHERE doc_id NOT LIKE '%%~%%' AND id > %s ORDER BY id LIMIT %s",
                    (last, BATCH),
                ).fetchall()
                if not rows:
                    break
                last = rows[-1][0]
                with conn.transaction(), conn.cursor().copy(
                    "COPY chunks (id, doc_id, ord, tenant_id, acl, content, embedding) FROM STDIN WITH (FORMAT BINARY)"
                ) as cp:
                    cp.set_types(["int8", "text", "int4", "int4", "int4[]", "text", "vector"])
                    for cid, doc, ord_, content, emb in rows:
                        v = np.asarray(emb, dtype=np.float32) + rng.normal(0, NOISE, config.EMBED_DIM).astype(np.float32)
                        v /= np.linalg.norm(v)
                        cp.write_row((cid + offset, f"{doc}~{r}", ord_, 0, [0], content, v))
                if free_gb() < args.min_free_gb:
                    sys.exit(f"stopping: free space {free_gb():.1f} GB fell below {args.min_free_gb} GB (copy {r})")
            print(f"copy {r} done, free {free_gb():.1f} GB", flush=True)
        conn.execute("ANALYZE chunks")
        conn.commit()


if __name__ == "__main__":
    main()
