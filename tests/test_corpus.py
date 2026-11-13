from __future__ import annotations

from prag import corpus
from prag.config import CorpusSpec


def test_chunk_words_overlap_and_coverage():
    words = [f"w{i}" for i in range(25)]
    chunks = corpus.chunk_words(" ".join(words), size=10, overlap=3)
    assert chunks[0].split() == words[:10]
    assert chunks[1].split()[:3] == words[7:10]  # overlap
    assert chunks[-1].split()[-1] == "w24"
    assert set(" ".join(chunks).split()) == set(words)


def test_chunk_words_short_and_empty():
    assert corpus.chunk_words("one two", 10, 3) == ["one two"]
    assert corpus.chunk_words("   ", 10, 3) == []


def test_layout_is_deterministic_and_well_formed():
    ids = [str(i) for i in range(2000)]
    a, b = corpus.build_layout(ids), corpus.build_layout(ids)
    assert a.doc_tenant == b.doc_tenant and a.doc_acl == b.doc_acl and a.members == b.members
    for d in ids:
        t = a.doc_tenant[d]
        assert all(g // 100 == t for g in a.doc_acl[d])
        assert 1 <= len(a.doc_acl[d]) <= 2
    for (t, p), groups in a.members.items():
        assert all(g // 100 == t for g in groups)
        assert (len(groups) == 8) == (p == "admin")
    # A different seed gives a different layout.
    c = corpus.build_layout(ids, CorpusSpec(seed=1))
    assert c.doc_tenant != a.doc_tenant


def test_oracle_visibility():
    layout = corpus.build_layout([str(i) for i in range(500)])
    d = "7"
    t = layout.doc_tenant[d]
    assert layout.visible(t, "admin", d)
    other = 1 + (t % 4)
    assert not layout.visible(other, "admin", d)
    assert not layout.visible(t, "nobody", d)
