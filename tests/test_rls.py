"""Row-level security: isolation, default deny, and that the filter runs in the database."""

from __future__ import annotations

import psycopg
import pytest

from prag import config, db, retrieval
from prag.db import Principal

from .conftest import DOCS

TENANT_OF = {d: t for d, t, _, _ in DOCS}


def visible_docs(conn, principal):
    with db.as_principal(conn, principal):
        return {r[0] for r in conn.execute("SELECT doc_id FROM chunks").fetchall()}


def test_tenant_isolation_full_scan(reader):
    assert visible_docs(reader, Principal(1, "alice")) == {"a1", "a2", "a3", "a4"}
    assert visible_docs(reader, Principal(2, "dave")) == {"b1", "b2", "b3"}


def test_group_filtering_within_tenant(reader):
    assert visible_docs(reader, Principal(1, "bob")) == {"a1", "a3"}
    assert visible_docs(reader, Principal(1, "carol")) == {"a2", "a3", "a4"}
    assert visible_docs(reader, Principal(2, "erin")) == {"b2", "b3"}


def test_lexical_never_returns_other_tenant(reader):
    # "index funds" and "Roth IRA" appear in both tenants.
    for query, expect_hits in [
        ("index funds low fees retirement", True),
        ("Roth IRA tax", True),
        ("secret merger plan competitor", False),  # only tenant 2 has these words
    ]:
        with db.as_principal(reader, Principal(1, "alice")):
            hits = retrieval.lexical(reader, query)
            docs = retrieval.chunk_texts(reader, [h.chunk_id for h in hits])
        assert bool(docs) == expect_hits, query
        assert all(TENANT_OF[d] == 1 for d, _ in docs.values())


def test_vector_search_respects_policy_even_for_exact_match(reader, seeded):
    # Query with tenant 2's secret document embedding: the nearest neighbour
    # overall is b2, but tenant 1 must not get it, and erin's colleague with
    # only group 200 must not either.
    q = seeded["b2"]
    with db.as_principal(reader, Principal(1, "alice")):
        ids = [h.chunk_id for h in retrieval.vector(reader, q, k=10)]
        docs = {d for d, _ in retrieval.chunk_texts(reader, ids).values()}
    assert docs and "b2" not in docs and all(TENANT_OF[d] == 1 for d in docs)
    with db.as_principal(reader, Principal(2, "erin")):
        top = retrieval.vector(reader, q, k=1)
    assert top[0].chunk_id == 6  # b2 is chunk 6 and erin holds group 201


def test_vector_returns_k_rows_when_filter_is_selective(reader):
    # bob sees 2 of 7 chunks; asking for 2 must return both, not fewer.
    with db.as_principal(reader, Principal(1, "bob")):
        hits = retrieval.vector(reader, retrieval_np_zero(), k=2)
    assert len(hits) == 2


def retrieval_np_zero():
    import numpy as np

    v = np.ones(config.EMBED_DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_forged_ids_return_nothing(reader):
    with db.as_principal(reader, Principal(1, "bob")):
        got = retrieval.chunk_texts(reader, [1, 2, 3, 4, 5, 6, 7])
    assert {d for d, _ in got.values()} == {"a1", "a3"}


def test_default_deny_without_identity(reader):
    with db.as_principal(reader, None):
        assert reader.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert reader.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
        assert retrieval.lexical(reader, "index funds") == []


def test_unknown_principal_sees_nothing(reader):
    assert visible_docs(reader, Principal(1, "mallory")) == set()
    # Right name, wrong tenant.
    assert visible_docs(reader, Principal(2, "alice")) == set()


def test_identity_does_not_outlive_transaction(reader):
    """A pooled connection reused by another caller starts with no identity."""
    assert visible_docs(reader, Principal(1, "alice"))
    with reader.transaction():
        assert reader.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0
        assert reader.execute("SELECT current_setting('app.tenant_id', true)").fetchone()[0] in (None, "")


def test_reader_cannot_read_memberships_or_write(reader):
    for stmt in [
        "SELECT * FROM memberships",
        "INSERT INTO memberships VALUES (1, 'bob', 101)",
        "UPDATE chunks SET acl = '{100}'",
        "DELETE FROM chunks",
    ]:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with reader.transaction():
                reader.execute(stmt)


def test_reader_cannot_bypass_rls(reader):
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with reader.transaction():
            reader.execute("SET row_security = off")
            reader.execute("SELECT count(*) FROM chunks")
    row = reader.execute(
        "SELECT rolbypassrls, rolsuper FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    reader.rollback()
    assert row == (False, False)


def test_auditor_sees_everything(cluster, seeded):
    with cluster.connect(config.AUDITOR_ROLE) as aud:
        assert aud.execute("SELECT count(*) FROM chunks").fetchone()[0] == len(DOCS)


def test_policy_is_applied_inside_the_plan(reader):
    """The RLS predicate appears in the query plan, i.e. Postgres filters before ranking."""
    with db.as_principal(reader, Principal(1, "bob")):
        plan = "\n".join(
            r[0]
            for r in reader.execute(
                "EXPLAIN " + retrieval.LEXICAL_SQL, {"q": "index funds", "k": 10, "norm": 1}
            ).fetchall()
        )
    assert "current_groups" in plan or "InitPlan" in plan
    assert "tenant_id" in plan
