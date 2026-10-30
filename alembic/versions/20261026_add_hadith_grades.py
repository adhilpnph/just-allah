"""Add source grade arrays to passages

Revision ID: 20261026_01
Revises: 20241025_01
Create Date: 2026-10-26 15:30:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20261026_01"
down_revision = "20241025_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "passages",
        sa.Column("grades", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.execute(
        "DROP FUNCTION search_passages(text, vector(384), text, integer)"
    )
    op.execute(
        """
        CREATE FUNCTION search_passages(q text, qe vector(384), p_kind text, k int default 5)
        returns table (ref text, kind text, collection text, arabic text, english text,
                       narrator text, grade text, graded_by text, grades jsonb, url text)
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
                 p.narrator, p.grade, p.graded_by, p.grades, p.url
          from scored s
          join passages p using (id)
          order by s.score desc
          limit k;
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION search_passages(text, vector(384), text, integer)")
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
    op.drop_column("passages", "grades")
