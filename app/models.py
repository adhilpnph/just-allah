from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Passage(Base):
    __tablename__ = "passages"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    ref: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    collection: Mapped[str | None] = mapped_column(String, nullable=True)
    arabic: Mapped[str] = mapped_column(Text, nullable=False)
    english: Mapped[str] = mapped_column(Text, nullable=False)
    narrator: Mapped[str | None] = mapped_column(Text, nullable=True)
    grade: Mapped[str | None] = mapped_column(String, nullable=True)
    graded_by: Mapped[str | None] = mapped_column(String, nullable=True)
    grades: Mapped[list[dict[str, str]] | None] = mapped_column(
        JSONB(none_as_null=True),
        nullable=True,
    )
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    fts: Mapped[str] = mapped_column(
        Computed("to_tsvector('english', english)", persisted=True),
        nullable=False,
    )


__table_args__ = (
    Index("passages_kind_idx", Passage.kind),
    Index("passages_fts_idx", Passage.fts, postgresql_using="gin"),
)
