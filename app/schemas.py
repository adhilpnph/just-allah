from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class VerifyRequest(BaseModel):
    text: str
    mode: Literal["hadith", "claim"] = "hadith"


class PassageCard(BaseModel):
    ref: str
    kind: str
    collection: str | None
    arabic: str
    english: str
    narrator: str | None
    grade: str | None
    graded_by: str | None
    grades: list[dict[str, str]] | None
    url: str | None


class VerifyResponse(BaseModel):
    mode: Literal["hadith", "claim"]
    status: Literal["matched", "unverified", "claim"]
    answer: str
    passages: list[PassageCard]
    similarity: float | None = None
