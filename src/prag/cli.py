"""Command line entry point: `prag <group> <command>`."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile

from . import config


def _db(args) -> None:
    from . import migrate, pgcluster

    if args.cmd in ("up", "start"):
        if args.cmd == "up":
            pgcluster.initdb(config.PG_DATA)
        pgcluster.start(config.PG_DATA, config.PG_PORT, config.PG_LOG)
        if args.cmd == "up":
            migrate.bootstrap(config.dsn("postgres", dbname="postgres"), config.DB_NAME)
            import psycopg

            with psycopg.connect(config.dsn(config.OWNER_ROLE)) as conn:
                print("applied:", migrate.migrate(conn) or "nothing pending")
        print(f"postgres running on {config.PG_HOST}:{config.PG_PORT}, data dir {config.PG_DATA}")
    elif args.cmd == "stop":
        pgcluster.stop(config.PG_DATA)
        print("postgres stopped")
    elif args.cmd == "migrate":
        import psycopg

        with psycopg.connect(config.dsn(config.OWNER_ROLE)) as conn:
            print("applied:", migrate.migrate(conn) or "nothing pending")
    elif args.cmd == "status":
        print("running" if pgcluster.is_running(config.PG_DATA) else "stopped")


def _data(args) -> None:
    target = config.RAW_DIR / "fiqa.zip"
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print("downloading", config.FIQA_URL)
        with urllib.request.urlopen(config.FIQA_URL) as resp, target.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != config.FIQA_SHA256:
        sys.exit(f"checksum mismatch for {target}: {digest}")
    with zipfile.ZipFile(target) as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in name.split("/"):
                sys.exit(f"refusing suspicious archive member {name}")
        zf.extractall(config.RAW_DIR)
    print("FiQA ready in", config.FIQA_DIR)


def _ingest(args) -> None:
    from . import ingest

    print(json.dumps(ingest.run_ingest(), indent=2))


def _eval(args) -> None:
    if args.cmd == "retrieval":
        from .eval import retrieval_eval

        retrieval_eval.main(args)
    elif args.cmd == "acl":
        from .eval import acl_eval

        acl_eval.main(args)
    elif args.cmd in ("answers", "rejudge"):
        from .eval import answer_eval

        answer_eval.main(args)
    elif args.cmd == "load":
        from .eval import load_test

        load_test.main(args)


def _ask(args) -> None:
    from . import pipeline

    out = pipeline.ask(args.question, args.tenant, args.principal, model=args.model, rerank=args.rerank)
    print(json.dumps(out, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="prag")
    sub = p.add_subparsers(dest="group", required=True)

    db = sub.add_parser("db", help="manage the local Postgres cluster")
    db.add_argument("cmd", choices=["up", "start", "stop", "migrate", "status"])
    db.set_defaults(func=_db)

    data = sub.add_parser("data", help="download and verify FiQA")
    data.add_argument("cmd", choices=["fetch"])
    data.set_defaults(func=_data)

    ing = sub.add_parser("ingest", help="chunk, embed and load FiQA")
    ing.set_defaults(func=_ingest)

    ev = sub.add_parser("eval", help="run an evaluation")
    ev.add_argument("cmd", choices=["retrieval", "acl", "answers", "rejudge", "load"])
    ev.add_argument("--split", default="test")
    ev.add_argument("--scope", choices=["corpus", "permissioned", "both"], default="both")
    ev.add_argument("--methods", default="lexical,vector,hybrid,hybrid_rerank")
    ev.add_argument("--model", default="qwen3-1.7b", help="answers: qwen3-1.7b | claude-haiku-5-5 | claude-sonnet-5-5")
    ev.add_argument("--judge", default="qwen3-1.7b")
    ev.add_argument("--sample", type=int, default=100)
    ev.add_argument("--plan-only", action="store_true", help="answers: print the planned calls and cost, run nothing")
    ev.add_argument("--concurrency", default="1,8,32")
    ev.add_argument("--duration", type=float, default=20.0)
    ev.set_defaults(func=_eval)

    ask = sub.add_parser("ask", help="answer one question as a given principal")
    ask.add_argument("question")
    ask.add_argument("--tenant", type=int, required=True)
    ask.add_argument("--principal", required=True)
    ask.add_argument("--model", default="qwen3-1.7b")
    ask.add_argument("--rerank", action="store_true")
    ask.set_defaults(func=_ask)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
