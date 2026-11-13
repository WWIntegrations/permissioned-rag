-- Core tables. Chunks carry a denormalized copy of their document's tenant and
-- ACL so the row-level security check needs no join (see 0002 for the trigger
-- that keeps the copy in sync).

CREATE TABLE tenants (
    id   integer PRIMARY KEY,
    name text    NOT NULL UNIQUE
);

-- Group ids are globally unique, so an integer array of group ids is
-- unambiguous across tenants.
CREATE TABLE acl_groups (
    id        integer PRIMARY KEY,
    tenant_id integer NOT NULL REFERENCES tenants (id),
    name      text    NOT NULL,
    UNIQUE (tenant_id, name)
);

CREATE TABLE principals (
    tenant_id integer NOT NULL REFERENCES tenants (id),
    id        text    NOT NULL,
    PRIMARY KEY (tenant_id, id)
);

CREATE TABLE memberships (
    tenant_id    integer NOT NULL,
    principal_id text    NOT NULL,
    group_id     integer NOT NULL REFERENCES acl_groups (id) ON DELETE CASCADE,
    PRIMARY KEY (tenant_id, principal_id, group_id),
    FOREIGN KEY (tenant_id, principal_id) REFERENCES principals (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE documents (
    id        text      PRIMARY KEY,
    tenant_id integer   NOT NULL REFERENCES tenants (id),
    acl       integer[] NOT NULL CHECK (cardinality(acl) > 0),
    title     text      NOT NULL DEFAULT ''
);

CREATE TABLE chunks (
    id        bigint    PRIMARY KEY,
    doc_id    text      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ord       integer   NOT NULL,
    tenant_id integer   NOT NULL,
    acl       integer[] NOT NULL,
    content   text      NOT NULL,
    tsv       tsvector  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding vector(384),
    UNIQUE (doc_id, ord)
);

CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_tenant ON chunks (tenant_id);
CREATE INDEX memberships_group ON memberships (group_id);
