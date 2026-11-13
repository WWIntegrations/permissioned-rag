"""Paths, connection settings and fixed experiment parameters.

Everything that changes a measured number lives here or in config/prices.toml,
so a results file can be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = ROOT / "migrations"
ROLES_SQL = ROOT / "sql" / "roles.sql"
EXTENSIONS_SQL = ROOT / "sql" / "extensions.sql"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FIQA_DIR = RAW_DIR / "fiqa"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
PRICES_FILE = ROOT / "config" / "prices.toml"

PG_BIN = Path(os.environ.get("PRAG_PG_BIN", "/opt/homebrew/opt/postgresql@17/bin"))
PG_DATA = Path(os.environ.get("PRAG_PG_DATA", str(ROOT / "pgdata")))
PG_LOG = ROOT / "pglog" / "postgres.log"
PG_HOST = "127.0.0.1"
PG_PORT = int(os.environ.get("PRAG_PG_PORT", "5433"))
DB_NAME = os.environ.get("PRAG_DB", "prag")

# Roles created by sql/bootstrap.sql.
OWNER_ROLE = "prag_owner"  # owns tables, runs migrations and ACL changes
READER_ROLE = "prag_reader"  # what the application uses; subject to RLS
AUDITOR_ROLE = "prag_auditor"  # BYPASSRLS, used only for corpus-wide benchmark numbers

FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
FIQA_SHA256 = "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02"

EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
QWEN_MODEL = "mlx-community/Qwen3-1.7B-4bit"


def dsn(role: str = READER_ROLE, dbname: str | None = None, host: str = PG_HOST, port: int | None = None) -> str:
    return f"host={host} port={port or PG_PORT} dbname={dbname or DB_NAME} user={role}"


@dataclass(frozen=True)
class CorpusSpec:
    """Synthetic tenancy and ACL layout. Changing any field changes every ACL."""

    seed: int = 20261112
    n_tenants: int = 4
    groups_per_tenant: int = 8
    users_per_tenant: int = 6
    second_group_prob: float = 0.3
    chunk_words: int = 180
    chunk_overlap: int = 30


CORPUS = CorpusSpec()


@dataclass(frozen=True)
class RetrievalSpec:
    rrf_k: int = 60
    lexical_k: int = 200
    vector_k: int = 200
    ef_search: int = 200
    rerank_top: int = 30
    ts_rank_normalization: int = 1


RETRIEVAL = RetrievalSpec()
