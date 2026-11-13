from __future__ import annotations

import math

import pytest

from prag.eval.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k
from prag.retrieval import rrf


def scores(hits):
    return {h.chunk_id: h.score for h in hits}


def test_rrf_hand_computed():
    # a: ranks 1 and 3, b: ranks 2 and 1, c: rank 3 only, d: rank 2 only
    out = rrf([[1, 2, 3], [2, 4, 1]], k=60)
    s = scores(out)
    assert s[1] == pytest.approx(1 / 61 + 1 / 63)
    assert s[2] == pytest.approx(1 / 62 + 1 / 61)
    assert s[3] == pytest.approx(1 / 63)
    assert s[4] == pytest.approx(1 / 62)
    assert [h.chunk_id for h in out] == [2, 1, 4, 3]


def test_rrf_single_list_preserves_order():
    assert [h.chunk_id for h in rrf([[9, 3, 7]])] == [9, 3, 7]


def test_rrf_ties_broken_by_id():
    # Each item is rank 1 in one list and absent from the other.
    assert [h.chunk_id for h in rrf([[5], [2]])] == [2, 5]


def test_rrf_duplicate_in_one_list_votes_once():
    assert scores(rrf([[1, 1, 2]], k=0))[1] == pytest.approx(1.0)


def test_rrf_small_k_rewards_top_ranks_more():
    lists = [[1, 2], [2, 1]]
    assert scores(rrf(lists, k=0))[1] == pytest.approx(1 + 1 / 2)
    assert rrf([], k=60) == []


def test_ndcg_perfect_and_partial():
    rel = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "x"], rel) == pytest.approx(1.0)
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a"], {"a": 1, "b": 1}) == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    ) == pytest.approx(expected)
    assert ndcg_at_k(["x"], {}) == 0.0


def test_recall_and_mrr():
    rel = {"a": 1, "b": 1, "c": 1, "d": 1}
    assert recall_at_k(["a", "x", "c"], rel, 100) == 0.5
    assert recall_at_k(["a", "x", "c"], rel, 1) == 0.25
    assert mrr_at_k(["x", "y", "b"], rel) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"] * 10 + ["a"], rel, 10) == 0.0


def test_percentile_interpolates():
    xs = [1, 2, 3, 4]
    assert percentile(xs, 50) == 2.5
    assert percentile(xs, 0) == 1 and percentile(xs, 100) == 4
