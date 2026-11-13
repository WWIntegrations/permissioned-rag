import pytest

from prag.eval.answer_eval import parse_score


@pytest.mark.parametrize("text,expected", [
    ("SCORE: 4", 4),
    ("SCORE: 5\nEvery claim is in source #12.", 5),
    ("The answer cites #3 correctly.\nSCORE: 2", 2),
    ("**SCORE:** 3", 3),
    ("SCORE: **3**", 3),
    ("score: 1", 1),
    ("4", 4),
    ("The answer accurately reflects source #4, which is", None),
    ("The answer directly addresses the question", None),
    ("SCORE: 7", None),
])
def test_parse_score(text, expected):
    assert parse_score(text) == expected
