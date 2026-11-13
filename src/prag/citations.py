"""Citation verification for generated answers.

Answer contract (stated in the prompt, enforced here):

* A citation is `[#<chunk id>]`; several may share brackets: `[#12, #40]`.
* A quote is text in double quotes followed by a citation: `"exact words" [#12]`.
* A model that cannot answer from the sources replies `INSUFFICIENT_CONTEXT`.

An answer is `valid` only when it has at least one citation, every cited id is
one of the chunks that were actually in the prompt, and every quote appears in
the text of a chunk it cites. Matching is exact after two normalizations:
runs of whitespace collapse to one space, and typographic quotes/apostrophes
map to their ASCII forms. An ellipsis inside a quote splits it into segments
that must each appear, in order, in the same chunk. Nothing else is forgiven:
changed words, case, punctuation or numbers make the quote fail.

Everything that is not valid is flagged with the reasons; nothing is dropped
or repaired silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

REFUSAL = "INSUFFICIENT_CONTEXT"

_CITE = re.compile(r"\[\s*(#\s*\d+(?:\s*[,;]\s*#?\s*\d+)*)\s*\]")
_ID = re.compile(r"\d+")
# Bracket groups that look like an attempt to cite but do not follow the contract.
_MALFORMED = re.compile(
    r"\[\s*(?:#|source|src|doc|chunk|s(?=\s*\d)|c(?=\s*\d)|id|(?=\d))[^\]]{0,40}\]", re.IGNORECASE
)
_QUOTE = re.compile(r"\"([^\"]{3,600}?)\"(\s*(?:\[[^\]]*\]\s*)*)")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|…)\s*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")

_TRANSLATE = str.maketrans(
    {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'", "′": "'",
        " ": " ",
    }
)

# Short quoted phrases without a citation are treated as emphasis, not as quotes.
MIN_UNCITED_QUOTE_WORDS = 4

Status = Literal["valid", "invalid", "uncited", "refusal"]


def normalize(text: str) -> str:
    return " ".join(text.translate(_TRANSLATE).split())


@dataclass(frozen=True)
class Problem:
    kind: Literal["not_in_context", "quote_mismatch", "malformed_citation", "refusal_with_content"]
    detail: str


@dataclass
class Verdict:
    status: Status
    cited_ids: list[int] = field(default_factory=list)
    quotes_checked: int = 0
    problems: list[Problem] = field(default_factory=list)
    sentence_coverage: float = 0.0  # share of answer sentences that carry a citation

    @property
    def ok(self) -> bool:
        return self.status in ("valid", "refusal")


def _ids_in(fragment: str) -> list[int]:
    out = []
    for m in _CITE.finditer(fragment):
        out.extend(int(x) for x in _ID.findall(m.group(1)))
    return out


def _quote_in(quote: str, text: str) -> bool:
    segments = [s for s in _ELLIPSIS.split(normalize(quote)) if s]
    if not segments:
        return False
    hay = normalize(text)
    pos = 0
    for seg in segments:
        i = hay.find(seg, pos)
        if i < 0:
            return False
        pos = i + len(seg)
    return True


def verify(answer: str, context: dict[int, str]) -> Verdict:
    """Check `answer` against the chunks that were sent to the model (id -> text)."""
    text = answer.strip()
    if text.strip(" .\n\t`*\"'") == REFUSAL:
        return Verdict(status="refusal")

    norm = text.translate(_TRANSLATE)
    problems: list[Problem] = []
    if REFUSAL in norm:
        # Answering and refusing at once is a contract violation; the answer
        # part is still checked below so fabricated citations are reported.
        problems.append(Problem("refusal_with_content", "answer text and the refusal marker together"))
        norm = norm.replace(REFUSAL, "")

    cited = _ids_in(norm)
    for cid in dict.fromkeys(cited):
        if cid not in context:
            problems.append(Problem("not_in_context", f"[#{cid}] was not among the {len(context)} chunks sent"))

    valid_spans = [m.span() for m in _CITE.finditer(norm)]
    for m in _MALFORMED.finditer(norm):
        if not any(a <= m.start() < b for a, b in valid_spans):
            problems.append(Problem("malformed_citation", m.group(0)))

    quotes_checked = 0
    for m in _QUOTE.finditer(norm):
        quote, trailer = m.group(1), m.group(2)
        ids = _ids_in(trailer)
        if ids:
            quotes_checked += 1
            sources = [context[i] for i in ids if i in context]
            if not any(_quote_in(quote, s) for s in sources):
                problems.append(Problem("quote_mismatch", f'"{quote[:80]}" not found in cited chunk(s) {ids}'))
        elif len(quote.split()) >= MIN_UNCITED_QUOTE_WORDS:
            quotes_checked += 1
            if not any(_quote_in(quote, s) for s in context.values()):
                problems.append(Problem("quote_mismatch", f'uncited quote "{quote[:80]}" not found in any chunk'))

    sentences = [s for s in _SENTENCE.split(norm) if s.strip()]
    covered = sum(1 for s in sentences if _CITE.search(s))
    coverage = covered / len(sentences) if sentences else 0.0

    if problems:
        status: Status = "invalid"
    elif not cited:
        status = "uncited"
    else:
        status = "valid"
    return Verdict(status, list(dict.fromkeys(cited)), quotes_checked, problems, round(coverage, 3))
