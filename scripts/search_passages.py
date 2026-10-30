from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.retrieval import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Quran passages")
    parser.add_argument("question")
    parser.add_argument("--kind", default="quran")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    for result in search(args.question, args.kind, args.k):
        print(result["ref"])


if __name__ == "__main__":
    main()
