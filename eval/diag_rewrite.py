from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import llm


QUESTIONS = (
    "What does Islam say about Prophet Lut?",
    "What does Islam say about Prophet Ibrahim?",
    "What did the prophet say about jokes?",
)


def _top_five(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda passage: float(passage.get("similarity", 0.0)),
        reverse=True,
    )[:5]


async def main() -> None:
    model_name = os.getenv("REWRITE_MODEL")
    for question in QUESTIONS:
        print(f"\nQuestion: {question}")
        print(f"REWRITE_MODEL: {'set' if model_name else 'not set'}")

        try:
            phrases = await llm._request_rewrite_search_phrases(question)
        except Exception as error:
            phrases = []
            fallback = True
            print(
                "Rewrite call failed: "
                f"{type(error).__name__}: {error}"
            )
        else:
            fallback = len(phrases) < 3

        if phrases:
            for phrase in phrases:
                print(phrase)
        else:
            print("empty")

        print(f"Original question used as fallback: {'yes' if fallback else 'no'}")
        retrieval_phrases = [question] if fallback else phrases

        for phrase in retrieval_phrases:
            print(f"\nPhrase retrieval: {phrase}")
            for kind in ("quran", "hadith"):
                candidates = llm.search_with_similarity(
                    llm._retrieval_query(phrase), kind, k=10
                )
                matches = _top_five(candidates)
                if kind == "quran":
                    print(f"  Quran result count: {len(candidates)}")
                refs_and_scores = [
                    (passage["ref"], float(passage.get("similarity", 0.0)))
                    for passage in matches
                ]
                print(f"  Top 5 {kind} refs and similarity: {refs_and_scores}")

        merged = llm.retrieve(question, retrieval_phrases)
        print("\nMerged passages sent to the LLM:")
        for kind in ("quran", "hadith"):
            kind_passages = [passage for passage in merged if passage.get("kind") == kind]
            print(f"  Top 5 {kind}:")
            for passage in kind_passages:
                similarity = float(passage.get("similarity", 0.0))
                english = str(passage.get("english") or "")[:60]
                print(f"    {passage['ref']} (similarity {similarity:.6f}): {english}")


if __name__ == "__main__":
    asyncio.run(main())
