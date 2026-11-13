# Design notes

## Request path

```
caller (tenant, principal)
   |
   v
BEGIN
  set_config('app.tenant_id', ..., local)      -- identity lives only in this transaction
  set_config('app.principal_id', ..., local)
  lexical top-200  ----\                        -- both queries run as prag_reader;
  vector  top-200  -----+--> RRF (k=60)         -- the RLS policy is part of each plan
  chunk text fetch (RLS again)                  -- a forged id returns nothing
COMMIT
   |
   v
optional cross-encoder rerank -> top 5 chunks -> prompt -> model -> citation verifier
```

## Why the policy looks up groups in the database

The session carries the tenant and the principal, not the group list. The
policy calls `app.current_groups()`, a `SECURITY DEFINER` function that reads
`memberships`, which the reader role cannot read directly. Revoking access is a
single `DELETE` or `UPDATE`, and it applies to the next query because nothing
caches the groups between the policy and the data. If groups came from a token,
revocation would wait for the token to expire.

The function is wrapped as `(SELECT app.current_groups())` in the policy so it
runs once per query as an InitPlan, not once per row.

## Denormalized ACL on chunks

Each chunk stores its document's `tenant_id` and `acl` so the policy needs no
join. Two triggers keep the copy honest: chunks take the document's values on
insert (whatever the writer passes is overwritten), and a document ACL change
rewrites its chunks in the same transaction. A third trigger rejects an ACL that
names another tenant's group.

## What the planner does under RLS (measured on this data)

`tsvector @@ tsquery` (`ts_match_vq`) is not marked LEAKPROOF. Postgres will
not evaluate a non-leakproof user predicate before the security predicate, so
under RLS it does not use the GIN index for the text match. With four tenants of about 19,000 chunks each, the planner instead uses
the b-tree on `tenant_id` and filters the tenant's rows. For vector search the plan depends on how the statement is executed:

* A one-off (custom) plan uses the same tenant b-tree and computes exact
  distances over the tenant slice. Exact, about 45 ms per query here, linear
  in tenant size.
* psycopg prepares a statement after its fifth execution, and Postgres then
  switches to a generic plan that walks the HNSW graph, applies the policy as
  a filter, and keeps scanning (`hnsw.iterative_scan = relaxed_order`) until
  200 rows pass. This is approximate.

Measured on 30 test queries: with auto-prepare disabled the RLS result set was
identical to exact hand-filtered kNN in 30/30 queries; with the default
(prepare after 5) mean overlap was 0.9605 and 10/30 were identical. The
published numbers use the psycopg default. Setting
`conn.prepare_threshold = None` on the reader connection gives exact results at
this scale; at larger scale the HNSW path is the one that keeps latency flat.

The BYPASSRLS auditor role, which has no policy, gets the GIN and HNSW plans.

## Citation contract

See `src/prag/citations.py` for the full rules. In short: ids must be in the
prompt, quotes must appear in the cited chunk after whitespace and quote-glyph
normalization only, an answer with zero citations is `uncited`, and an answer
that both answers and emits the refusal marker is `invalid`.
