"""FiQA loading, chunking, and the deterministic synthetic tenancy layout.

Tenancy layout (all derived from CorpusSpec.seed with SHA-256, so it is the same
on every machine and Python version):

* 4 tenants. Each document belongs to exactly one tenant.
* 8 ACL groups per tenant, ids tenant*100 + g (g = 0..7).
* Each document gets one primary group and, with probability 0.3, a second
  distinct group of the same tenant. A reader needs at least one of them.
* 6 principals per tenant: `admin` holds all 8 groups; `user1`..`user5` hold a
  seeded random subset of 1 to 4 groups.
* Chunks inherit the tenant and ACL of their document.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from . import config
from .config import CorpusSpec


def _unit(seed: int, *parts: object) -> float:
    h = hashlib.sha256(("|".join([str(seed), *map(str, parts)])).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: int
    doc_id: str
    ord: int
    content: str


@dataclass(frozen=True)
class Layout:
    tenants: list[int]
    groups: dict[int, list[int]]  # tenant -> group ids
    doc_tenant: dict[str, int]
    doc_acl: dict[str, tuple[int, ...]]
    members: dict[tuple[int, str], tuple[int, ...]]  # (tenant, principal) -> groups

    def visible(self, tenant: int, principal: str, doc_id: str) -> bool:
        """Independent oracle used by the leakage test (does not ask the database)."""
        if self.doc_tenant[doc_id] != tenant:
            return False
        groups = self.members.get((tenant, principal), ())
        return any(g in groups for g in self.doc_acl[doc_id])

    def principals(self) -> list[tuple[int, str]]:
        return sorted(self.members)


def load_corpus(path: Path = config.FIQA_DIR / "corpus.jsonl") -> list[Document]:
    docs = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            docs.append(Document(str(row["_id"]), row.get("title") or "", row.get("text") or ""))
    return docs


def load_queries(path: Path = config.FIQA_DIR / "queries.jsonl") -> dict[str, str]:
    out = {}
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            out[str(row["_id"])] = row["text"]
    return out


def load_qrels(split: str, root: Path = config.FIQA_DIR) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / f"{split}.tsv").open() as fh:
        next(fh)
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    stride = size - overlap
    out = []
    start = 0
    while True:
        end = min(start + size, len(words))
        out.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += stride
    return out


def iter_chunks(docs: list[Document], spec: CorpusSpec = config.CORPUS) -> Iterator[Chunk]:
    next_id = 1
    for d in docs:
        body = (d.title + "\n" + d.text).strip() if d.title else d.text
        for i, piece in enumerate(chunk_words(body, spec.chunk_words, spec.chunk_overlap)):
            yield Chunk(next_id, d.id, i, piece)
            next_id += 1


def build_layout(doc_ids: list[str], spec: CorpusSpec = config.CORPUS) -> Layout:
    tenants = list(range(1, spec.n_tenants + 1))
    groups = {t: [t * 100 + g for g in range(spec.groups_per_tenant)] for t in tenants}
    doc_tenant: dict[str, int] = {}
    doc_acl: dict[str, tuple[int, ...]] = {}
    for d in doc_ids:
        t = tenants[int(_unit(spec.seed, "tenant", d) * spec.n_tenants)]
        g1 = int(_unit(spec.seed, "group1", d) * spec.groups_per_tenant)
        acl = [groups[t][g1]]
        if _unit(spec.seed, "second", d) < spec.second_group_prob:
            g2 = int(_unit(spec.seed, "group2", d) * spec.groups_per_tenant)
            if g2 != g1:
                acl.append(groups[t][g2])
        doc_tenant[d] = t
        doc_acl[d] = tuple(sorted(acl))
    members: dict[tuple[int, str], tuple[int, ...]] = {}
    for t in tenants:
        members[(t, "admin")] = tuple(groups[t])
        for i in range(1, spec.users_per_tenant):
            rng = random.Random(f"{spec.seed}-{t}-user{i}")
            k = rng.randint(1, 4)
            members[(t, f"user{i}")] = tuple(sorted(rng.sample(groups[t], k)))
    return Layout(tenants, groups, doc_tenant, doc_acl, members)


def assign_query_principal(
    qid: str, relevant: dict[str, int], layout: Layout, spec: CorpusSpec = config.CORPUS
) -> tuple[int, str]:
    """Each evaluation query is asked by one deterministic (tenant, principal).

    The tenant is the owner of the query's first relevant document (by id), so
    the question is plausible for that tenant. The principal is drawn uniformly
    from that tenant's six principals, so some callers cannot see any relevant
    document; those queries become "answer not accessible" cases.
    """
    first = min(relevant, key=lambda d: (len(d), d))
    t = layout.doc_tenant[first]
    i = int(_unit(spec.seed, "qprincipal", qid) * spec.users_per_tenant)
    return t, ("admin" if i == 0 else f"user{i}")
