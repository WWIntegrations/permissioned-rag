-- Row-level security. The application connects as prag_reader and, inside
-- each transaction, sets two transaction-local settings:
--   app.tenant_id     the tenant of the caller
--   app.principal_id  the caller's id within that tenant
-- Group membership is looked up inside the database, so revoking a membership
-- is a single DELETE that applies to the next query. When a setting is
-- missing the functions return NULL / an empty array and the policy matches
-- nothing: the default is deny.

CREATE SCHEMA app;

CREATE FUNCTION app.current_tenant() RETURNS integer
LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT nullif(current_setting('app.tenant_id', true), '')::integer
$$;

-- SECURITY DEFINER so prag_reader can resolve its own groups without being
-- able to read the memberships table.
CREATE FUNCTION app.current_groups() RETURNS integer[]
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $$
    SELECT coalesce(array_agg(m.group_id ORDER BY m.group_id), '{}'::integer[])
    FROM public.memberships m
    WHERE m.tenant_id = nullif(current_setting('app.tenant_id', true), '')::integer
      AND m.principal_id = current_setting('app.principal_id', true)
$$;

REVOKE ALL ON FUNCTION app.current_groups() FROM PUBLIC;

ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- The (SELECT ...) wrappers make Postgres evaluate each function once per
-- query as an InitPlan instead of once per row.
CREATE POLICY chunks_reader ON chunks
    FOR SELECT TO prag_reader
    USING (tenant_id = (SELECT app.current_tenant())
           AND acl && (SELECT app.current_groups()));

CREATE POLICY documents_reader ON documents
    FOR SELECT TO prag_reader
    USING (tenant_id = (SELECT app.current_tenant())
           AND acl && (SELECT app.current_groups()));

GRANT USAGE ON SCHEMA app TO prag_reader;
GRANT EXECUTE ON FUNCTION app.current_tenant(), app.current_groups() TO prag_reader;
GRANT SELECT ON chunks, documents TO prag_reader;

GRANT SELECT ON chunks, documents, tenants, acl_groups, principals, memberships TO prag_auditor;
