"""Connections and the per-transaction caller identity.

Identity is set with set_config(..., is_local => true), so it lives only until
the end of the transaction. A pooled connection that is handed to another
caller therefore starts with no identity, and the RLS policy matches nothing
until a new identity is set.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

from . import config


@dataclass(frozen=True)
class Principal:
    tenant_id: int
    principal_id: str


def connect(role: str = config.READER_ROLE, dsn: str | None = None, **kw) -> psycopg.Connection:
    conn = psycopg.connect(dsn or config.dsn(role), **kw)
    register_vector(conn)
    return conn


@contextmanager
def as_principal(
    conn: psycopg.Connection,
    principal: Principal | None,
    ef_search: int = config.RETRIEVAL.ef_search,
) -> Iterator[psycopg.Connection]:
    """Run the block in one transaction with the caller identity and search settings set.

    `principal=None` is only meaningful for the BYPASSRLS auditor role.
    """
    with conn.transaction():
        conn.execute(
            "SELECT set_config('hnsw.ef_search', %s, true),"
            " set_config('hnsw.iterative_scan', 'relaxed_order', true)",
            (str(ef_search),),
        )
        if principal is not None:
            conn.execute(
                "SELECT set_config('app.tenant_id', %s, true),"
                " set_config('app.principal_id', %s, true)",
                (str(principal.tenant_id), principal.principal_id),
            )
        yield conn
