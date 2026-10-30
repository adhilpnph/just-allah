from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.llm import _retrieval_query, lookup_claim_passages


CLAIMS = (
    "The Prophet advised travelling by night",
    "Keep to travelling by night",
)


def diagnose(claim: str) -> None:
    normalized = _retrieval_query(claim)
    print(f"CLAIM={claim!r}")
    print(f"NORMALIZED_QUERY={normalized!r}")

    passages = lookup_claim_passages(claim)
    quran_count = sum(str(row.get("kind", "")).lower() == "quran" for row in passages)
    hadith = sorted(
        (row for row in passages if str(row.get("kind", "")).lower() == "hadith"),
        key=lambda row: float(row.get("similarity", 0.0)),
        reverse=True,
    )
    print(f"TOTAL_PASSAGES={len(passages)} QURAN={quran_count} HADITH={len(hadith)}")
    print("TOP_10_HADITH:")
    for rank, row in enumerate(hadith[:10], start=1):
        excerpt = " ".join(str(row.get("english") or "").split())[:80]
        print(f"  rank={rank} ref={row.get('ref')} similarity={float(row.get('similarity', 0.0)):.6f} text={excerpt!r}")

    target = next((index for index, row in enumerate(hadith, start=1) if row.get("ref") == "abudawud:2571"), None)
    if target is None:
        print("ABUDAWUD_2571=not present in returned hadith candidates")
    else:
        row = hadith[target - 1]
        print(f"ABUDAWUD_2571 rank={target} similarity={float(row.get('similarity', 0.0)):.6f}")

    print(
        "LIMITS=lookup_claim_passages requests k=20 independently for Quran and hadith; "
        "it applies no additional filter or similarity threshold. "
        "The later claim judge sorts the returned passages, then caps its prompt at 3 Quran "
        "and 2 hadith passages (no similarity threshold)."
    )


if __name__ == "__main__":
    for claim in CLAIMS:
        diagnose(claim)
