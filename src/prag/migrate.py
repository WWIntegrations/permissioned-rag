"""Plain-SQL migrations with checksums.

Each file in migrations/ is applied once, in filename order, inside its own
transaction, and recorded with its SHA-256. If an applied file is later edited
the runner refuses to continue instead of silently drifting.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import sql

from . import config


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str


def discover(directory: Path = config.MIGRATIONS_DIR) -> list[Migration]:
    out = []
    for path in sorted(directory.glob("*.sql")):
        body = path.read_bytes()
        out.append(Migration(path.stem, path, hashlib.sha256(body).hexdigest()))
    return out


def applied(conn: psycopg.Connection) -> dict[str, str]:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version    text PRIMARY KEY,
            checksum   text NOT NULL,
            applied_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    conn.commit()
    rows = conn.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return dict(rows)


def migrate(conn: psycopg.Connection, directory: Path = config.MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations. Returns the versions applied in this call."""
    done = applied(conn)
    pending = []
    for m in discover(directory):
        if m.version in done:
            if done[m.version] != m.checksum:
                raise MigrationError(
                    f"migration {m.version} was modified after it was applied "
                    f"(recorded {done[m.version][:12]}, file {m.checksum[:12]})"
                )
            continue
        pending.append(m)
    applied_now = []
    conn.commit()
    for m in pending:
        # One transaction per file: a failing file leaves earlier ones applied
        # and itself fully rolled back.
        try:
            conn.execute(m.path.read_text())
            conn.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                (m.version, m.checksum),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        applied_now.append(m.version)
    return applied_now


def bootstrap(admin_dsn_db: str, dbname: str) -> None:
    """Create roles, database and extension. Needs a superuser connection.

    `admin_dsn_db` is a superuser DSN for the maintenance database (postgres).
    """
    with psycopg.connect(admin_dsn_db, autocommit=True) as conn:
        # Roles first: CREATE DATABASE ... OWNER needs the owner role to exist.
        conn.execute(config.ROLES_SQL.read_text())
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone()
        if not exists:
            conn.execute(
                sql.SQL("CREATE DATABASE {} OWNER {}").format(
                    sql.Identifier(dbname), sql.Identifier(config.OWNER_ROLE)
                )
            )
    target = admin_dsn_db.replace("dbname=postgres", f"dbname={dbname}")
    with psycopg.connect(target, autocommit=True) as conn:
        conn.execute(config.EXTENSIONS_SQL.read_text())
