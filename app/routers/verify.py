from __future__ import annotations

import re
from difflib import SequenceMatcher

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.schemas import PassageCard, VerifyRequest, VerifyResponse
from app.services.llm import (
    lookup_claim_passages,
    retrieve_hadith_with_similarity,
    filter_supported_claim_passages,
    summarize_claim,
)

router = APIRouter()
VERIFY_MATCH_THRESHOLD = 0.85
VERIFY_LEXICAL_OVERLAP_MINIMUM = 4
NEGATIONS = {"not", "no", "never", "dont", "don't", "cannot", "neither", "nor", "forbid", "forbade", "forbidden", "prohibit", "prohibited"}
_VERIFY_STOPWORDS = {
    "all", "and", "are", "but", "for", "has", "have", "he", "her", "his",
    "not", "of", "on", "our", "reported", "said", "the", "that", "this",
    "to", "was", "were", "who", "with", "you",
}


def _content_words(text: str) -> set[str]:
    words = set()
    for word in re.findall(r"[a-z]+", text.lower(), flags=re.ASCII):
        if word not in _VERIFY_STOPWORDS:
            word = re.sub(r"(?:ing|ed|es|s)$", "", word)
            words.add(word)
    return words


def _word_sequence(text: str) -> list[str]:
    sequence: list[str] = []
    for word in re.findall(r"[a-z]+(?:'?[a-z]+)?", text.lower(), flags=re.ASCII):
        normalized = word.replace("'", "")
        if normalized in _VERIFY_STOPWORDS or normalized in NEGATIONS:
            continue
        sequence.append(re.sub(r"(?:ing|ed|es|s)$", "", normalized))
    return sequence


def _is_negated(text: str) -> bool:
    normalized = re.sub(r"\bdo\s+not\b", "dont", text.lower())
    words = re.findall(r"[a-z]+(?:'?[a-z]+)?", normalized, flags=re.ASCII)
    return any(word.replace("'", "") in NEGATIONS for word in words)


def _has_one_word_swap(query: str, passage: str) -> bool:
    query_words = _word_sequence(query)
    passage_words = _word_sequence(passage)
    if len(query_words) < 4 or len(passage_words) < 4:
        return False

    replacements = []
    for tag, query_start, query_end, passage_start, passage_end in SequenceMatcher(
        None, query_words, passage_words
    ).get_opcodes():
        if tag == "replace" and query_end - query_start == 1 and passage_end - passage_start == 1:
            query_word = query_words[query_start]
            passage_word = passage_words[passage_start]
            if query_word != passage_word:
                replacements.append((query_start, passage_start, query_word, passage_word))
    unique_replacements = {(query_word, passage_word) for _, _, query_word, passage_word in replacements}
    return len(unique_replacements) == 1


def _has_lexical_evidence(query: str, passage: dict) -> bool:
    passage_text = passage.get("english") or passage.get("arabic") or ""
    query_words = _content_words(query)
    passage_words = _content_words(passage_text)
    overlap = len(query_words.intersection(passage_words))
    required_overlap = VERIFY_LEXICAL_OVERLAP_MINIMUM if len(query_words) >= VERIFY_LEXICAL_OVERLAP_MINIMUM else 3
    if overlap < required_overlap:
        return False

    if _is_negated(query) != _is_negated(passage_text):
        return False

    common = query_words.intersection(passage_words)
    query_only = query_words - common
    passage_only = passage_words - common
    if len(common) >= VERIFY_LEXICAL_OVERLAP_MINIMUM and query_only and passage_only and _has_one_word_swap(query, passage_text):
        return False

    return True


@router.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest) -> VerifyResponse:
    if request.mode == "hadith":
        results = await run_in_threadpool(retrieve_hadith_with_similarity, request.text)
        matched = max(results, key=lambda row: float(row["similarity"]), default=None)
        if (
            matched is None
            or float(matched["similarity"]) < VERIFY_MATCH_THRESHOLD
            or not _has_lexical_evidence(request.text, matched)
        ):
            return VerifyResponse(
                mode="hadith",
                status="unverified",
                answer="This hadith cannot be verified in the sources.",
                passages=[],
            )
        matched["similarity"] = float(matched["similarity"])
        return VerifyResponse(
            mode="hadith",
            status="matched",
            answer=f"Matched source: [{matched['ref']}]",
            passages=[PassageCard(**matched)],
            similarity=matched["similarity"],
        )

    sources = await run_in_threadpool(lookup_claim_passages, request.text)
    supported_sources = await filter_supported_claim_passages(request.text, sources)
    if not supported_sources:
        return VerifyResponse(
            mode="claim",
            status="claim",
            answer="The sources do not address this claim.",
            passages=[],
        )
    answer, cited_passages = await summarize_claim(request.text, supported_sources)
    return VerifyResponse(
        mode="claim",
        status="claim",
        answer=answer,
        passages=[PassageCard(**passage) for passage in cited_passages],
    )
