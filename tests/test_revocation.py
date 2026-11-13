"""Revoking access applies to the next query, with no cache to expire."""

from __future__ import annotations

import psycopg
import pytest

from prag import db, retrieval
from prag.db import Principal


def sees(conn, principal, doc):
    with db.as_principal(conn, principal):
        return conn.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc,)).fetchone()[0] > 0


def test_membership_revoke_applies_to_next_query(reader, owner):
    carol = Principal(1, "carol")
    assert sees(reader, carol, "a2")
    try:
        owner.execute("DELETE FROM memberships WHERE tenant_id = 1 AND principal_id = 'carol' AND group_id = 101")
        owner.commit()
        # Same reader connection, very next transaction.
        assert not sees(reader, carol, "a2")
        with db.as_principal(reader, carol):
            assert retrieval.lexical(reader, "Roth IRA after tax dollars") == []
    finally:
        owner.execute("INSERT INTO memberships VALUES (1, 'carol', 101) ON CONFLICT DO NOTHING")
        owner.commit()
    assert sees(reader, carol, "a2")


def test_uncommitted_revoke_is_not_visible_yet(reader, owner):
    carol = Principal(1, "carol")
    owner.execute("DELETE FROM memberships WHERE tenant_id = 1 AND principal_id = 'carol' AND group_id = 101")
    try:
        assert sees(reader, carol, "a2")  # not committed: still visible
    finally:
        owner.rollback()
    assert sees(reader, carol, "a2")


def test_document_acl_change_propagates_to_chunks(reader, owner):
    bob = Principal(1, "bob")
    assert sees(reader, bob, "a1")
    try:
        owner.execute("UPDATE documents SET acl = '{101}' WHERE id = 'a1'")
        owner.commit()
        assert owner.execute("SELECT acl FROM chunks WHERE doc_id = 'a1'").fetchone()[0] == [101]
        owner.commit()
        assert not sees(reader, bob, "a1")
    finally:
        owner.execute("UPDATE documents SET acl = '{100}' WHERE id = 'a1'")
        owner.commit()
    assert sees(reader, bob, "a1")


def test_deleting_a_group_revokes_everyone(reader, owner):
    erin = Principal(2, "erin")
    assert sees(reader, erin, "b2")
    dave = Principal(2, "dave")
    try:
        # Memberships cascade away with the group; documents still list 201,
        # but nobody holds it any more.
        with owner.transaction():
            owner.execute("DELETE FROM acl_groups WHERE id = 201")
        assert not sees(reader, erin, "b2")
        assert not sees(reader, erin, "b3")
        assert not sees(reader, dave, "b2")
        assert sees(reader, dave, "b3")  # still reachable through group 200
    finally:
        with owner.transaction():
            owner.execute("INSERT INTO acl_groups VALUES (201, 2, 'g201') ON CONFLICT DO NOTHING")
            owner.execute("INSERT INTO memberships VALUES (2, 'dave', 201), (2, 'erin', 201) ON CONFLICT DO NOTHING")
    assert sees(reader, erin, "b2")


def test_acl_cannot_name_another_tenants_group(owner):
    with pytest.raises(psycopg.errors.RaiseException):
        with owner.transaction():
            owner.execute("UPDATE documents SET acl = '{200}' WHERE id = 'a1'")


def test_chunk_insert_inherits_document_acl(owner):
    with owner.transaction():
        owner.execute(
            "INSERT INTO chunks (id, doc_id, ord, tenant_id, acl, content) VALUES (999, 'a2', 5, 2, '{200}', 'x')"
        )
        row = owner.execute("SELECT tenant_id, acl FROM chunks WHERE id = 999").fetchone()
        assert row == (1, [101])
        owner.execute("DELETE FROM chunks WHERE id = 999")
