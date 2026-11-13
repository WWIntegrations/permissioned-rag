"""Real-Postgres fixtures.

By default a throwaway cluster is created with initdb in a temp directory on a
free port and stopped at the end of the session. Set PRAG_TEST_ADMIN_DSN to a
superuser DSN (dbname=postgres) to use an existing server instead, as CI does
with its Postgres service container. Each session creates a fresh database.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

from prag import config, migrate, pgcluster

DIM = config.EMBED_DIM


@dataclass
class Cluster:
    host: str
    port: int
    dbname: str
    admin_dsn: str  # superuser, maintenance db
    password: str = ""

    def dsn(self, role: str, dbname: str | None = None) -> str:
        # The application roles have no password; the test server trusts
        # connections from the loopback interface (CI sets the same).
        return f"host={self.host} port={self.port} dbname={dbname or self.dbname} user={role}"

    def connect(self, role: str, **kw) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn(role), **kw)
        register_vector(conn)
        return conn


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# A tiny corpus with a known layout. Two tenants; tenant 1 has groups 100, 101,
# tenant 2 has 200, 201. Several documents share vocabulary across tenants so a
# leak would actually be ranked.
DOCS = [
    # id, tenant, acl, text
    ("a1", 1, [100], "Index funds have low fees and broad diversification for retirement savings."),
    ("a2", 1, [101], "Roth IRA contributions are made with after tax dollars and grow tax free."),
    ("a3", 1, [100, 101], "Dollar cost averaging into index funds reduces timing risk."),
    ("a4", 1, [101], "Capital gains on index funds held over a year are taxed at a lower rate."),
    ("b1", 2, [200], "Index funds with low fees are popular for retirement in tenant two."),
    ("b2", 2, [201], "Secret merger plan: tenant two acquires a competitor next quarter."),
    ("b3", 2, [200, 201], "Roth IRA rules for tenant two employees and their tax treatment."),
]
PRINCIPALS = {
    (1, "alice"): [100, 101],
    (1, "bob"): [100],
    (1, "carol"): [101],
    (2, "dave"): [200, 201],
    (2, "erin"): [201],
}


def load_fixture_data(conn: psycopg.Connection) -> dict[str, np.ndarray]:
    vecs = {}
    with conn.transaction():
        conn.execute("INSERT INTO tenants VALUES (1, 'tenant-1'), (2, 'tenant-2')")
        conn.execute(
            "INSERT INTO acl_groups VALUES (100, 1, 'g100'), (101, 1, 'g101'), (200, 2, 'g200'), (201, 2, 'g201')"
        )
        for (t, p), groups in PRINCIPALS.items():
            conn.execute("INSERT INTO principals VALUES (%s, %s)", (t, p))
            for g in groups:
                conn.execute("INSERT INTO memberships VALUES (%s, %s, %s)", (t, p, g))
        for i, (doc, t, acl, text) in enumerate(DOCS):
            conn.execute("INSERT INTO documents (id, tenant_id, acl) VALUES (%s, %s, %s)", (doc, t, acl))
            vecs[doc] = unit(i)
            conn.execute(
                "INSERT INTO chunks (id, doc_id, ord, tenant_id, acl, content, embedding) "
                "VALUES (%s, %s, 0, 0, '{0}', %s, %s)",
                (i + 1, doc, text, vecs[doc]),
            )
    return vecs


@pytest.fixture(scope="session")
def cluster(tmp_path_factory) -> Cluster:
    admin = os.environ.get("PRAG_TEST_ADMIN_DSN")
    dbname = "prag_test_" + uuid.uuid4().hex[:8]
    if admin:
        params = psycopg.conninfo.conninfo_to_dict(admin)
        c = Cluster(params.get("host", "127.0.0.1"), int(params.get("port", 5432)), dbname, admin, params.get("password", ""))
        migrate.bootstrap(admin, dbname)
        yield c
        return
    data_dir: Path = tmp_path_factory.mktemp("pg") / "data"
    port = pgcluster.free_port()
    settings = dict(pgcluster.SERVER_SETTINGS, shared_buffers="'128MB'", maintenance_work_mem="'64MB'")
    pgcluster.initdb(data_dir, settings)
    pgcluster.start(data_dir, port, data_dir.parent / "pg.log")
    try:
        c = Cluster("127.0.0.1", port, dbname, f"host=127.0.0.1 port={port} dbname=postgres user=postgres")
        migrate.bootstrap(c.admin_dsn, dbname)
        yield c
    finally:
        pgcluster.stop(data_dir)


@pytest.fixture(scope="session")
def seeded(cluster: Cluster) -> dict[str, np.ndarray]:
    with cluster.connect(config.OWNER_ROLE) as conn:
        migrate.migrate(conn)
        return load_fixture_data(conn)


@pytest.fixture
def reader(cluster, seeded):
    with cluster.connect(config.READER_ROLE) as conn:
        yield conn


@pytest.fixture
def owner(cluster, seeded):
    with cluster.connect(config.OWNER_ROLE) as conn:
        yield conn
