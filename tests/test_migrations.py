from __future__ import annotations

import shutil
import uuid

import psycopg
import pytest
from psycopg import sql

from prag import config, migrate


@pytest.fixture
def fresh_db(cluster):
    name = "prag_mig_" + uuid.uuid4().hex[:8]
    with psycopg.connect(cluster.admin_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(sql.Identifier(name), sql.Identifier(config.OWNER_ROLE)))
    with psycopg.connect(cluster.admin_dsn.replace("dbname=postgres", f"dbname={name}"), autocommit=True) as conn:
        conn.execute(config.EXTENSIONS_SQL.read_text())
    yield cluster.dsn(config.OWNER_ROLE, dbname=name)
    with psycopg.connect(cluster.admin_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


def test_applies_all_in_order_and_is_idempotent(fresh_db):
    with psycopg.connect(fresh_db) as conn:
        first = migrate.migrate(conn)
        assert first == [m.version for m in migrate.discover()]
        assert migrate.migrate(conn) == []
        rows = conn.execute("SELECT version, checksum FROM schema_migrations ORDER BY version").fetchall()
        assert [(m.version, m.checksum) for m in migrate.discover()] == rows


def test_schema_has_rls_enabled(fresh_db):
    with psycopg.connect(fresh_db) as conn:
        migrate.migrate(conn)
        rls = dict(
            conn.execute(
                "SELECT relname, relrowsecurity FROM pg_class WHERE relname IN ('chunks', 'documents')"
            ).fetchall()
        )
        assert rls == {"chunks": True, "documents": True}
        idx = {r[0] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'").fetchall()}
        assert {"chunks_tsv_gin", "chunks_embedding_hnsw"} <= idx


def test_edited_migration_is_refused(fresh_db, tmp_path):
    for m in migrate.discover():
        shutil.copy(m.path, tmp_path / m.path.name)
    with psycopg.connect(fresh_db) as conn:
        migrate.migrate(conn, tmp_path)
        target = tmp_path / "0001_core.sql"
        target.write_text(target.read_text() + "\n-- edited later\n")
        with pytest.raises(migrate.MigrationError, match="modified after it was applied"):
            migrate.migrate(conn, tmp_path)


def test_failed_migration_rolls_back(fresh_db, tmp_path):
    for m in migrate.discover():
        shutil.copy(m.path, tmp_path / m.path.name)
    (tmp_path / "0099_broken.sql").write_text("CREATE TABLE half_done (id int);\nSELECT no_such_function();\n")
    with psycopg.connect(fresh_db) as conn:
        with pytest.raises(psycopg.errors.UndefinedFunction):
            migrate.migrate(conn, tmp_path)
        conn.rollback()
        assert conn.execute("SELECT to_regclass('half_done')").fetchone()[0] is None
        versions = {r[0] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
        assert "0099_broken" not in versions and "0003_row_level_security" in versions
