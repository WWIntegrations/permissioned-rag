"""Ranking metrics with the BEIR conventions (binary gains for FiQA)."""

from __future__ import annotations

import math
from collections.abc import Sequence


def ndcg_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int = 10) -> float:
    dcg = sum(relevant.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int = 100) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if relevant.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
