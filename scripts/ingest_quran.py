from __future__ import annotations

import sqlite3
import sys
from time import perf_counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy.dialects.postgresql import insert

from app.database import SessionLocal
from app.models import Passage
from app.services.embeddings import embed_passages

DB_PATH = Path(__file__).resolve().parents[1] / "app" / "quran.db" / "quran.db"
INSERT_CHUNK_SIZE = 500


def load_bismillah_prefix() -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT a.text
            FROM ayahs a
            JOIN surahs s ON s.id = a.surah_id
            WHERE s.number = 1 AND a.number_in_surah = 1
            """
        ).fetchone()
        if row is None:
            raise RuntimeError("Could not find Qur'an ayah 1:1 in the source database")
        return row[0]


def build_ayah_rows() -> list[dict[str, object]]:
    bismillah = load_bismillah_prefix()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT
                s.number AS surah_number,
                a.number_in_surah AS ayah_number,
                a.text AS arabic_text,
                ae.data AS english_text,
                e.name AS collection_name,
                a.id AS ayah_id
            FROM ayahs a
            JOIN surahs s ON s.id = a.surah_id
            JOIN ayah_edition ae ON ae.ayah_id = a.id
            JOIN editions e ON e.id = ae.edition_id
            WHERE e.identifier = 'en.sahih'
              AND ae.is_audio = 0
            ORDER BY s.number, a.number_in_surah
            """
        ).fetchall()

    payload: list[dict[str, object]] = []
    for surah_number, ayah_number, arabic_text, english_text, collection_name, ayah_id in rows:
        cleaned_arabic = arabic_text
        if int(surah_number) > 1:
            cleaned_arabic = cleaned_arabic.removeprefix(bismillah)
        ref = f"quran:{surah_number}:{ayah_number}"
        payload.append(
            {
                "kind": "quran",
                "ref": ref,
                "collection": collection_name,
                "arabic": cleaned_arabic,
                "english": english_text,
                "narrator": None,
                "grade": None,
                "graded_by": None,
                "url": f"https://quran.com/{surah_number}/{ayah_number}",
            }
        )

    embeddings = embed_passages([str(row["english"]) for row in payload])
    for row, embedding in zip(payload, embeddings, strict=True):
        row["embedding"] = embedding

    return payload


def ingest() -> None:
    started_at = perf_counter()
    rows = build_ayah_rows()
    print(f"Embedded {len(rows)} Quran ayahs", flush=True)
    total_chunks = (len(rows) + INSERT_CHUNK_SIZE - 1) // INSERT_CHUNK_SIZE
    with SessionLocal() as session:
        for chunk_index, start in enumerate(range(0, len(rows), INSERT_CHUNK_SIZE), start=1):
            chunk = rows[start : start + INSERT_CHUNK_SIZE]
            statement = insert(Passage).values(chunk).on_conflict_do_nothing(
                index_elements=[Passage.ref]
            )
            session.execute(statement)
            session.commit()
            print(
                f"Inserted chunk {chunk_index}/{total_chunks} "
                f"({min(start + len(chunk), len(rows))}/{len(rows)} ayahs)",
                flush=True,
            )

    print(f"Total ingestion time: {perf_counter() - started_at:.2f} seconds", flush=True)


if __name__ == "__main__":
    ingest()
