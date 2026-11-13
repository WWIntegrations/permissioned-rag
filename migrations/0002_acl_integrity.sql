-- Keep chunk ACL copies equal to their document, and reject ACLs that name a
-- group from another tenant. Both run inside the writing transaction, so an ACL
-- change is visible to the very next query after COMMIT.

CREATE FUNCTION documents_check_acl() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    bad integer;
BEGIN
    SELECT g INTO bad
    FROM unnest(NEW.acl) AS g
    WHERE NOT EXISTS (
        SELECT 1 FROM acl_groups ag WHERE ag.id = g AND ag.tenant_id = NEW.tenant_id
    )
    LIMIT 1;
    IF bad IS NOT NULL THEN
        RAISE EXCEPTION 'group % does not belong to tenant %', bad, NEW.tenant_id;
    END IF;
    RETURN NEW;
END
$$;

CREATE TRIGGER documents_check_acl
    BEFORE INSERT OR UPDATE OF acl, tenant_id ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_check_acl();

CREATE FUNCTION documents_sync_chunks() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    UPDATE chunks
       SET acl = NEW.acl, tenant_id = NEW.tenant_id
     WHERE doc_id = NEW.id;
    RETURN NEW;
END
$$;

CREATE TRIGGER documents_sync_chunks
    AFTER UPDATE OF acl, tenant_id ON documents
    FOR EACH ROW EXECUTE FUNCTION documents_sync_chunks();

-- Chunks inherit the ACL of their document at insert time; a mismatch is an error.
CREATE FUNCTION chunks_inherit_acl() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    d documents%ROWTYPE;
BEGIN
    SELECT * INTO d FROM documents WHERE id = NEW.doc_id;
    NEW.tenant_id := d.tenant_id;
    NEW.acl := d.acl;
    RETURN NEW;
END
$$;

CREATE TRIGGER chunks_inherit_acl
    BEFORE INSERT ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_inherit_acl();
