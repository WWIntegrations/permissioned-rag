-- Run once per cluster as a superuser. Roles are cluster-wide objects, so they
-- live here and not in migrations/.

DO $$
BEGIN
    -- Owns the schema, runs migrations, changes ACLs.
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'prag_owner') THEN
        CREATE ROLE prag_owner LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
    END IF;
    -- The application role. It can only SELECT, and every row it sees has
    -- passed the row-level security policy.
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'prag_reader') THEN
        CREATE ROLE prag_reader LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
    END IF;
    -- Benchmark-only role that sees the whole corpus, so retrieval quality
    -- can be compared with published BEIR numbers. Never used to serve answers.
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'prag_auditor') THEN
        CREATE ROLE prag_auditor LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS;
    END IF;
END
$$;
