"""Create passages table and search function

Revision ID: 20241025_01
Revises: 
Create Date: 2024-10-25 14:45:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "20241025_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE passages (
          id bigserial PRIMARY KEY,
          kind text NOT NULL,
          ref text NOT NULL UNIQUE,
          collection text,
          arabic text NOT NULL,
          english text NOT NULL,
          narrator text,
          grade text,
          graded_by text,
          url text,
          embedding vector(384),
          fts tsvector GENERATED ALWAYS AS (to_tsvector('english', english)) STORED
        )
        """
    )
    op.execute(
        """
        CREATE INDEX passages_embedding_hnsw_idx
        ON passages
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    op.execute("CREATE INDEX passages_fts_idx ON passages USING gin (fts)")
    op.execute("CREATE INDEX passages_kind_idx ON passages (kind)")
    op.execute(
        """
        CREATE FUNCTION search_passages(q text, qe vector(384), p_kind text, k int default 5)
        returns table (ref text, kind text, collection text, arabic text, english text,
                       narrator text, grade text, graded_by text, url text)
        language sql stable as $$
          with v as (
            select id, row_number() over (order by embedding <=> qe) as r
            from passages
            where kind = p_kind
            order by embedding <=> qe
            limit 20
          ),
          t as (
            select id, row_number() over (order by ts_rank(fts, websearch_to_tsquery('english', q)) desc) as r
            from passages
            where kind = p_kind and fts @@ websearch_to_tsquery('english', q)
            limit 20
          ),
          scored as (
            select id, sum(1.0 / (60 + r)) as score
            from (select * from v union all select * from t) x
            group by id
          )
          select p.ref, p.kind, p.collection, p.arabic, p.english,
                 p.narrator, p.grade, p.graded_by, p.url
          from scored s
          join passages p using (id)
          order by s.score desc
          limit k;
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS search_passages(text, vector(384), text, integer)")
    op.execute("DROP INDEX IF EXISTS passages_kind_idx")
    op.execute("DROP INDEX IF EXISTS passages_fts_idx")
    op.execute("DROP INDEX IF EXISTS passages_embedding_hnsw_idx")
    op.execute("DROP TABLE IF EXISTS passages")
    op.execute("DROP EXTENSION IF EXISTS vector")
