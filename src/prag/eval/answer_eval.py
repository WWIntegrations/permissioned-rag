"""Answer evaluation: citation validity, refusals, judged quality, cost, latency.

Case types, drawn from FiQA test queries with a fixed seed:

* answerable (60% of the sample): the asking principal can read a relevant
  document and the retrieved context contains a chunk of one.
* acl_blocked (20%): the asking principal can read no relevant document. The
  model should refuse. Noisy label: FiQA judgments are incomplete, so the
  readable corpus may still hold an answer.
* swapped (20%): the context retrieved for an unrelated question (word
  overlap below 0.1) is paired with this question. The model should refuse.

The judge scores answerable, non-refused answers 1 to 5 against the question
and the sources. With the local model the judge is the same Qwen3-1.7B that
wrote the answers; that score is a weak signal and reported as such.
"""

from __future__ import annotations

import json
import random
import re
import time

from .. import config, corpus, db, embed, pipeline, retrieval
from ..db import Principal
from ..generation import cost_usd, get, prices
from . import hardware, write_result
from .metrics import percentile

JUDGE_SYSTEM = """You grade answers to personal finance questions. You are given the question, the sources the answer was allowed to use, and the answer.
Score from 1 to 5:
5 = correct, directly answers the question, every claim supported by the sources
4 = mostly correct and supported, minor gaps
3 = partly answers the question or partly unsupported
2 = mostly unhelpful or mostly unsupported
1 = wrong, off-topic, or contradicts the sources
Start your reply with the line SCORE: <number>. You may add one short reason after it."""

# Room for a model that explains before scoring; the parser reads only the SCORE line.
# 16 tokens cut the first hosted judge run off before the score (29 of 30 unparsed).
JUDGE_MAX_TOKENS = 200
# The published local run skipped the cross-encoder to keep memory (and swap)
# down on a 16 GB machine; see README.
RERANK = False


def judge_prompt(question: str, context: list[tuple[int, str]], answer: str) -> str:
    return f"{pipeline.build_user_prompt(question, context)}\n\nAnswer to grade:\n{answer}"


def parse_score(text: str) -> int | None:
    """The score from a 'SCORE: n' line, or a reply that is only the digit. A bare digit
    inside prose (a source id such as #4) is never taken as the score."""
    m = re.search(r"SCORE:\s*\**\s*([1-5])\b", text, re.I) or re.fullmatch(r"\s*([1-5])\s*", text)
    return int(m.group(1)) if m else None


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", s.lower()))


def build_cases(n: int, seed: int = 7) -> list[dict]:
    queries = corpus.load_queries()
    qrels = corpus.load_qrels("test")
    docs = corpus.load_corpus()
    layout = corpus.build_layout([d.id for d in docs])
    del docs
    qids = sorted(qrels, key=lambda q: (len(q), q))
    rng = random.Random(seed)
    rng.shuffle(qids)
    want = {"answerable": round(n * 0.6), "acl_blocked": round(n * 0.2), "swapped": n - round(n * 0.6) - round(n * 0.2)}
    cases: list[dict] = []
    with db.connect(config.READER_ROLE) as conn, db.connect(config.AUDITOR_ROLE) as aud:
        doc_of = dict(aud.execute("SELECT id, doc_id FROM chunks").fetchall())
        counts = dict.fromkeys(want, 0)
        for qid in qids:
            if all(counts[k] >= want[k] for k in want):
                break
            t, p = corpus.assign_query_principal(qid, qrels[qid], layout)
            readable = {d for d in qrels[qid] if layout.visible(t, p, d)}
            principal = Principal(t, p)
            if readable and counts["answerable"] < want["answerable"]:
                ctx, rsecs = pipeline.retrieve_context(conn, queries[qid], principal, rerank=RERANK)
                if any(doc_of[c] in readable for c, _ in ctx):
                    cases.append(dict(kind="answerable", qid=qid, tenant=t, principal=p, question=queries[qid], context=ctx, retrieval_s=rsecs, gold_docs=sorted(readable)))
                    counts["answerable"] += 1
            elif not readable and counts["acl_blocked"] < want["acl_blocked"]:
                ctx, rsecs = pipeline.retrieve_context(conn, queries[qid], principal, rerank=RERANK)
                cases.append(dict(kind="acl_blocked", qid=qid, tenant=t, principal=p, question=queries[qid], context=ctx, retrieval_s=rsecs, gold_docs=[]))
                counts["acl_blocked"] += 1
            elif readable and counts["swapped"] < want["swapped"]:
                other = next(
                    o for o in qids
                    if o != qid and len(_words(queries[o]) & _words(queries[qid])) / max(1, len(_words(queries[o]) | _words(queries[qid]))) < 0.1
                    and not (set(qrels[o]) & set(qrels[qid]))
                )
                ctx, rsecs = pipeline.retrieve_context(conn, queries[other], principal, rerank=RERANK)
                if not any(doc_of[c] in readable for c, _ in ctx):
                    cases.append(dict(kind="swapped", qid=qid, tenant=t, principal=p, question=queries[qid], context=ctx, retrieval_s=rsecs, gold_docs=[], context_from=other))
                    counts["swapped"] += 1
    return cases


def run(model: str, n: int, judge_model: str) -> dict:
    gen = get(model)
    ok, why = gen.configured()
    if not ok:
        raise SystemExit(f"{model}: {why}")
    judge = get(judge_model)
    cases = build_cases(n)
    rows = []
    for i, c in enumerate(cases):
        out = pipeline.answer(gen, c["question"], c["context"])
        row = {k: v for k, v in c.items() if k != "context"}
        row["context_ids"] = [cid for cid, _ in c["context"]]
        row["retrieval_s"] = round(c["retrieval_s"], 3)
        row.update(out)
        if c["kind"] == "answerable" and out["verdict"] != "refusal":
            j = judge.generate(JUDGE_SYSTEM, judge_prompt(c["question"], c["context"], out["answer"]), JUDGE_MAX_TOKENS)
            row["judge_model"] = judge.name
            row["judge_raw"] = j.text
            row["judge_score"] = parse_score(j.text)
            row["judge_input_tokens"] = j.input_tokens
        rows.append(row)
        print(i, c["kind"], out["verdict"], out["output_tokens"], round(out["generation_s"], 2), flush=True)
    return summarize(model, judge.name, rows)


def summarize(model: str, judge_name: str, rows: list[dict]) -> dict:
    by = lambda k: [r for r in rows if r["kind"] == k]  # noqa: E731
    ans, blocked, swapped = by("answerable"), by("acl_blocked"), by("swapped")
    non_refusal = [r for r in rows if r["verdict"] != "refusal"]
    problems: dict[str, int] = {}
    for r in non_refusal:
        for p in r["problems"]:
            problems[p["kind"]] = problems.get(p["kind"], 0) + 1
    scores = [r["judge_score"] for r in ans if r.get("judge_score") is not None]
    gen_s = [r["generation_s"] for r in rows]
    e2e_s = [r["generation_s"] + r["retrieval_s"] for r in rows]
    tin = sum(r["input_tokens"] for r in rows) / len(rows)
    tout = sum(r["output_tokens"] for r in rows) / len(rows)
    rate = lambda xs, f: round(sum(1 for r in xs if f(r)) / len(xs), 3) if xs else None  # noqa: E731
    summary = {
        "model": model,
        "cases": {"answerable": len(ans), "acl_blocked": len(blocked), "swapped": len(swapped)},
        "citation": {
            "non_refusal_answers": len(non_refusal),
            "valid": sum(r["verdict"] == "valid" for r in non_refusal),
            "invalid": sum(r["verdict"] == "invalid" for r in non_refusal),
            "uncited": sum(r["verdict"] == "uncited" for r in non_refusal),
            "validity_rate": rate(non_refusal, lambda r: r["verdict"] == "valid"),
            "problem_counts": problems,
            "answers_with_checked_quote": sum(r["quotes_checked"] > 0 for r in non_refusal),
            "mean_sentence_coverage": round(sum(r["sentence_coverage"] for r in non_refusal) / max(1, len(non_refusal)), 3),
        },
        "refusal": {
            "answerable_false_refusal_rate": rate(ans, lambda r: r["verdict"] == "refusal"),
            "acl_blocked_refusal_rate": rate(blocked, lambda r: r["verdict"] == "refusal"),
            "swapped_refusal_rate": rate(swapped, lambda r: r["verdict"] == "refusal"),
        },
        "judge": {
            "model": judge_name,
            "scored": len(scores),
            "mean_score_1_to_5": round(sum(scores) / len(scores), 2) if scores else None,
            "distribution": {s: scores.count(s) for s in range(1, 6)},
            "caveat": "same model as the generator" if judge_name == model else "",
        },
        "tokens_mean": {"input": round(tin, 1), "output": round(tout, 1)},
        "cost_per_1k_queries_usd": round(cost_usd(model, tin * 1000, tout * 1000), 4),
        "latency_s": {
            "generation_p50": round(percentile(gen_s, 50), 2),
            "generation_p95": round(percentile(gen_s, 95), 2),
            "end_to_end_p50": round(percentile(e2e_s, 50), 2),
            "end_to_end_p95": round(percentile(e2e_s, 95), 2),
        },
        "settings": {"context_chunks": pipeline.CONTEXT_CHUNKS, "max_answer_tokens": pipeline.MAX_ANSWER_TOKENS, "rerank": RERANK},
        "hardware": hardware(),
    }
    return {"summary": summary, "rows": rows}


def plan(models: list[str], n: int, judge_model: str = "claude-sonnet-5-5") -> dict:
    """Exact call count and token-based cost estimate for a hosted run. Calls nothing remote.

    Input tokens are counted with the Qwen3 tokenizer on the exact prompts that
    would be sent (Claude's tokenizer is not available offline, so this is a
    proxy; the margin column adds 35%). Expected output tokens come from the
    measured local run when present; the upper bound assumes every call hits
    max_tokens.
    """
    from ..generation.mlx_qwen import QwenMLX

    cases_path = config.RESULTS_DIR / "answers_qwen3-1.7b_cases.jsonl"
    if not cases_path.exists():
        raise SystemExit("run `prag eval answers --model qwen3-1.7b` first; the plan reuses its cases")
    rows = [json.loads(line) for line in cases_path.read_text().splitlines()]
    with db.connect(config.READER_ROLE) as conn:
        contexts = []
        for r in rows:
            with db.as_principal(conn, Principal(r["tenant"], r["principal"])):
                texts = retrieval.chunk_texts(conn, r["context_ids"])
            contexts.append([(cid, texts[cid][1]) for cid in r["context_ids"]])
    tok = QwenMLX()
    overhead = 10  # per-message framing tokens, rough
    answer_in = [tok.count_tokens(pipeline.SYSTEM) + tok.count_tokens(pipeline.build_user_prompt(r["question"], c)) + overhead for r, c in zip(rows, contexts)]
    mean_out = sum(r["output_tokens"] for r in rows) / len(rows)
    n_judge = sum(1 for r in rows if r["kind"] == "answerable")
    judge_in = [
        tok.count_tokens(JUDGE_SYSTEM) + tok.count_tokens(judge_prompt(r["question"], c, r["answer"])) + overhead
        for r, c in zip(rows, contexts) if r["kind"] == "answerable"
    ]
    jp = prices()["models"][judge_model]
    out = {"checked": prices()["checked"], "cases": len(rows), "models": {}}
    for m in models:
        p = prices()["models"][m]
        total_in = sum(answer_in)
        exp = (total_in * p["input"] + mean_out * len(rows) * p["output"]) / 1e6
        ub = (total_in * 1.35 * p["input"] + pipeline.MAX_ANSWER_TOKENS * len(rows) * p["output"]) / 1e6
        jin = sum(judge_in)
        jexp = (jin * jp["input"] + 6 * len(judge_in) * jp["output"]) / 1e6
        jub = (jin * 1.35 * jp["input"] + JUDGE_MAX_TOKENS * len(judge_in) * jp["output"]) / 1e6
        out["models"][m] = {
            "answer_calls": len(rows),
            "answer_input_tokens_total": total_in,
            "answer_output_tokens_expected_total": round(mean_out * len(rows)),
            "answer_cost_usd_expected": round(exp, 4),
            "answer_cost_usd_upper_bound": round(ub, 4),
            "judge_model": judge_model,
            "judge_calls_upper_bound": n_judge,
            "judge_input_tokens_total": jin,
            "judge_cost_usd_expected": round(jexp, 4),
            "judge_cost_usd_upper_bound": round(jub, 4),
            "cost_per_1k_queries_usd_expected": round(exp / len(rows) * 1000, 4),
            "price_per_mtok": {"input": p["input"], "output": p["output"]},
        }
    out["total_usd_expected"] = round(sum(v["answer_cost_usd_expected"] + v["judge_cost_usd_expected"] for v in out["models"].values()), 4)
    out["total_usd_upper_bound"] = round(sum(v["answer_cost_usd_upper_bound"] + v["judge_cost_usd_upper_bound"] for v in out["models"].values()), 4)
    return out


def rejudge(model: str, judge_model: str = "claude-sonnet-5-5") -> dict:
    """Judge the saved answers of a finished run again, without generating them again.
    The context each answer saw is read back by chunk id."""
    path = config.RESULTS_DIR / f"answers_{model}_cases.jsonl"
    rows = [json.loads(line) for line in path.open()]
    judge = get(judge_model)
    ok, why = judge.configured()
    if not ok:
        raise SystemExit(f"{judge_model}: {why}")
    ids = sorted({cid for r in rows for cid in r["context_ids"]})
    with db.connect(config.AUDITOR_ROLE) as aud:
        text = {int(i): c for i, c in aud.execute("SELECT id, content FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()}
    for r in rows:
        if r["kind"] == "answerable" and r["verdict"] != "refusal":
            ctx = [(cid, text[cid]) for cid in r["context_ids"]]
            j = judge.generate(JUDGE_SYSTEM, judge_prompt(r["question"], ctx, r["answer"]), JUDGE_MAX_TOKENS)
            r.update(judge_model=judge.name, judge_raw=j.text, judge_score=parse_score(j.text), judge_input_tokens=j.input_tokens)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return summarize(model, judge.name, rows)


def main(args) -> None:
    if args.cmd == "rejudge":
        res = rejudge(args.model, args.judge if args.judge != "qwen3-1.7b" else "claude-sonnet-5-5")
        prev = config.RESULTS_DIR / f"answers_{args.model}.json"
        if prev.exists():  # keep the wall clock of the generating run
            old = json.loads(prev.read_text())
            if "wall_clock_s" in old:
                res["summary"]["wall_clock_s"] = old["wall_clock_s"]
        print(json.dumps(res["summary"], indent=2))
        print("wrote", write_result(f"answers_{args.model}", res["summary"]))
        return
    hosted = args.model.startswith("claude")
    if hosted and not args.plan_only:
        ok, why = get(args.model).configured()
        if not ok:
            print(f"{args.model}: {why}. Printing the plan instead; nothing was sent.")
            args.plan_only = True
    if args.plan_only:
        models = [args.model] if hosted else ["claude-haiku-5-5", "claude-sonnet-5-5"]
        p = plan(models, args.sample)
        print(json.dumps(p, indent=2))
        write_result("claude_cost_plan", p)
        return
    judge = args.judge
    if hosted and judge == "qwen3-1.7b":
        judge = "claude-sonnet-5-5"  # a hosted run is judged by the stronger hosted model
    t0 = time.perf_counter()
    res = run(args.model, args.sample, judge)
    res["summary"]["wall_clock_s"] = round(time.perf_counter() - t0, 1)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with (config.RESULTS_DIR / f"answers_{args.model}_cases.jsonl").open("w") as fh:
        for r in res["rows"]:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(res["summary"], indent=2))
    print("wrote", write_result(f"answers_{args.model}", res["summary"]))
