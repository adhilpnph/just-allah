from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy.dialects.postgresql import insert

from app.database import SessionLocal
from app.models import Passage
from app.services.embeddings import embed_passages

DATA_DIR = ROOT / "data" / "hadith"
INSERT_CHUNK_SIZE = 500
COLLECTIONS = ("bukhari", "muslim", "abudawud")


def _hadith_number(value: int | float | str) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _read_records(path: Path) -> tuple[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["metadata"]["name"], data["hadiths"]


def _duplicate_numbers(records: list[dict[str, Any]]) -> list[int | float | str]:
    numbers = [row["hadithnumber"] for row in records]
    return [number for number, count in Counter(numbers).items() if count > 1]


def build_rows(
    collection: str,
) -> tuple[list[dict[str, Any]], set[int | float | str], set[int | float | str], int | float]:
    arabic_name = f"ara-{collection}.json"
    english_name = f"eng-{collection}.json"
    arabic_path = DATA_DIR / arabic_name
    english_path = DATA_DIR / english_name
    if not english_path.exists():
        raise FileNotFoundError(f"Required English source file is missing: {english_path}")

    arabic_collection, arabic_records = _read_records(arabic_path)
    english_collection, english_records = _read_records(english_path)
    if arabic_collection != english_collection:
        raise ValueError(f"Collection metadata differs between {arabic_name} and {english_name}")

    arabic_duplicates = _duplicate_numbers(arabic_records)
    english_duplicates = _duplicate_numbers(english_records)
    if arabic_duplicates or english_duplicates:
        raise ValueError(
            f"Duplicate hadithnumber values in {collection}: "
            f"Arabic={arabic_duplicates[:10]}, English={english_duplicates[:10]}"
        )

    arabic_by_number = {row["hadithnumber"]: row for row in arabic_records}
    english_by_number = {row["hadithnumber"]: row for row in english_records}
    arabic_numbers = set(arabic_by_number)
    english_numbers = set(english_by_number)
    if arabic_numbers != english_numbers:
        raise ValueError(
            f"Arabic and English hadithnumber sets differ for {collection}: "
            f"Arabic-only={sorted(arabic_numbers - english_numbers)[:10]}, "
            f"English-only={sorted(english_numbers - arabic_numbers)[:10]}"
        )

    rows: list[dict[str, Any]] = []
    skipped_arabic: set[int | float | str] = set()
    skipped_english: set[int | float | str] = set()
    max_hadithnumber = 0
    for arabic in arabic_records:
        number = arabic["hadithnumber"]
        max_hadithnumber = max(max_hadithnumber, number)
        english = english_by_number[number]
        arabic_text = arabic["text"].strip()
        english_text = english["text"].strip()
        if not arabic_text:
            skipped_arabic.add(number)
        if not english_text:
            skipped_english.add(number)
        if not arabic_text or not english_text:
            continue

        number_ref = _hadith_number(number)
        source_grades = arabic["grades"]
        rows.append(
            {
                "kind": "hadith",
                "ref": f"{collection}:{number_ref}",
                "collection": arabic_collection,
                "arabic": arabic_text,
                "english": english_text,
                "narrator": None,
                "grade": None,
                "graded_by": None,
                "grades": source_grades or None,
                "url": f"https://sunnah.com/{collection}:{number_ref}",
            }
        )

    embeddings = embed_passages([row["english"] for row in rows])
    for row, embedding in zip(rows, embeddings, strict=True):
        row["embedding"] = embedding
    return rows, skipped_arabic, skipped_english, max_hadithnumber


def ingest(collection: str) -> None:
    started_at = perf_counter()
    rows, skipped_arabic, skipped_english, max_hadithnumber = build_rows(collection)
    skipped_numbers = skipped_arabic | skipped_english
    if skipped_numbers:
        skipped_path = DATA_DIR / f"{collection}_skipped.txt"
        skipped_path.write_text(
            "".join(f"{_hadith_number(number)}\n" for number in sorted(skipped_numbers)),
            encoding="utf-8",
        )
        print(
            f"Arabic/English skipped hadithnumber sets identical: "
            f"{skipped_arabic == skipped_english}",
            flush=True,
        )
        print(f"Wrote skipped hadithnumbers to {skipped_path}", flush=True)

    print(
        f"Prepared {len(rows)} {collection} records; skipped "
        f"{len(skipped_numbers)} records with blank Arabic or English text "
        f"(Arabic blank={len(skipped_arabic)}, English blank={len(skipped_english)}); "
        f"last_hadithnumber={_hadith_number(max_hadithnumber)}",
        flush=True,
    )
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
                f"Processed chunk {chunk_index}/{total_chunks} "
                f"({min(start + len(chunk), len(rows))}/{len(rows)} records)",
                flush=True,
            )

    print(f"Total ingestion time: {perf_counter() - started_at:.2f} seconds", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Sahih al-Bukhari or Sahih Muslim")
    parser.add_argument("collection", choices=COLLECTIONS)
    args = parser.parse_args()
    ingest(args.collection)


if __name__ == "__main__":
    main()
