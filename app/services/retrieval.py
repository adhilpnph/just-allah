from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import bindparam, text

from app.database import SessionLocal
from app.services.embeddings import embed_query


def _normalized_arabic(text: str) -> str:
    text = "".join(
        character
        for character in text
        if not (
            "\u0610" <= character <= "\u061a"
            or "\u064b" <= character <= "\u065f"
            or character in "\u0670\u06d6\u06d7\u06d8\u06d9\u06da\u06db\u06dc\u06df\u06e0\u06e1\u06e2\u06e3\u06e4\u06e5\u06e6\u06e7\u06e8\u06ea\u06eb\u06ec\u06ed"
        )
    )
    text = text.replace("\u0640", "")
    return text.translate(str.maketrans("أإآٱىؤئ", "اااايوي"))


def _arabic_text_matches(session, question: str, k: int) -> list:
    normalized_question = _normalized_arabic(question)
    if not normalized_question:
        return []
    candidates = session.execute(
        text(
            "SELECT ref, kind, collection, arabic, english, narrator, grade, graded_by, grades, url "
            "FROM passages WHERE kind = 'quran'"
        )
    ).mappings().all()
    matches = []
    for candidate in candidates:
        arabic = _normalized_arabic(str(candidate["arabic"] or ""))
        if normalized_question in arabic:
            matches.append(dict(candidate))
            if len(matches) >= k:
                break
    return matches


def search(question: str, kind: str, k: int = 5):
    query_embedding = embed_query(question)
    with SessionLocal() as session:
        session.execute(text("SET LOCAL hnsw.ef_search = 200"))
        rows = session.execute(
            text(
                "SELECT ref, kind, collection, arabic, english, narrator, grade, graded_by, grades, url "
                "FROM search_passages(:q, :qe, :kind, :k)"
            ).bindparams(bindparam("qe", type_=Vector(384))),
            {
                "q": question,
                "qe": query_embedding,
                "kind": kind,
                "k": k,
            },
        ).mappings().all()

        if question and kind == "quran" and any("\u0600" <= char <= "\u06ff" for char in question):
            arabic_matches = _arabic_text_matches(session, question, k)
            for arabic_match in arabic_matches:
                arabic_match.setdefault("similarity", 0.0)
            seen = {row["ref"] for row in rows}
            rows = list(rows) + [row for row in arabic_matches if row["ref"] not in seen][:k]
    return rows


def search_with_similarity(question: str, kind: str, k: int = 5):
    query_embedding = embed_query(question)
    with SessionLocal() as session:
        session.execute(text("SET LOCAL hnsw.ef_search = 200"))
        rows = session.execute(
            text(
                """
                WITH v AS (
                    SELECT id, row_number() OVER (ORDER BY embedding <=> :qe) AS r
                    FROM passages
                    WHERE kind = :kind
                    ORDER BY embedding <=> :qe
                    LIMIT 20
                ),
                t AS (
                    SELECT id, row_number() OVER (
                        ORDER BY ts_rank(fts, websearch_to_tsquery('english', :q)) DESC
                    ) AS r
                    FROM passages
                    WHERE kind = :kind
                      AND fts @@ websearch_to_tsquery('english', :q)
                    LIMIT 20
                ),
                scored AS (
                    SELECT id, sum(1.0 / (60 + r)) AS score
                    FROM (SELECT * FROM v UNION ALL SELECT * FROM t) ranked
                    GROUP BY id
                )
                SELECT p.ref, p.kind, p.collection, p.arabic, p.english,
                       p.narrator, p.grade, p.graded_by, p.grades, p.url,
                       1 - (p.embedding <=> :qe) AS similarity
                FROM scored
                JOIN passages p USING (id)
                ORDER BY scored.score DESC
                LIMIT :k
                """
            ).bindparams(bindparam("qe", type_=Vector(384))),
            {"q": question, "qe": query_embedding, "kind": kind, "k": k},
        ).mappings().all()
        if question and kind == "quran" and any("\u0600" <= char <= "\u06ff" for char in question):
            arabic_matches = _arabic_text_matches(session, question, k)
            for arabic_match in arabic_matches:
                arabic_match.setdefault("similarity", 0.0)
            seen = {row["ref"] for row in rows}
            rows = list(rows) + [row for row in arabic_matches if row["ref"] not in seen][:k]
    return rows
