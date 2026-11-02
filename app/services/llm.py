from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from starlette.concurrency import run_in_threadpool

from app.database import SessionLocal
from app.models import Passage
from app.services.retrieval import search, search_with_similarity

load_dotenv()
logger = logging.getLogger(__name__)

EventType = Literal["token", "card", "done"]


@dataclass(frozen=True)
class StreamEvent:
    event: EventType
    data: str | dict[str, Any]


def _client_and_model() -> tuple[AsyncOpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in backend/.env")
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is not set in backend/.env")
    if not base_url:
        raise RuntimeError("OPENROUTER_BASE_URL is not set in backend/.env")
    return AsyncOpenAI(api_key=api_key, base_url=base_url), model


def _prompt(question: str, passages: Iterable[dict[str, Any]]) -> str:
    source_lines = []
    for passage in passages:
        english = re.sub(r"\s+", " ", str(passage.get("english") or "")).strip()
        source_lines.append(
            "\n".join(
                (
                    f"ref: {passage['ref']}",
                    f"collection: {passage.get('collection') or 'not specified'}",
                    f"narrator: {passage.get('narrator') or 'not specified'}",
                    f"text: {english[:600]}",
                )
            )
        )
    sources = "\n\n".join(source_lines) or "No relevant passages were retrieved."
    return (
        "Answer using only the source passages below. Start with Quran passages, then "
        "hadith passages. Write short plain-language prose stating only what a cited passage "
        "directly says. Cite a passage only if it directly addresses the question; ignore the other "
        "retrieved passages and do not describe them. You may state a passage's explicit content in plain words; "
        "do not state anything that "
        "the passage only implies. Do not use conclusions, inference words, or reasoning such "
        "as therefore, this shows, this means, or these narrations indicate. Do not use "
        "\"therefore\"-style conclusions. Do not weigh or "
        "rank passages. Do not state religious verdicts that the passage does not directly state. "
        "If no retrieved passage directly addresses the question, write one sentence saying the retrieved "
        "sources do not state this, without claiming anything about Islam beyond the database. Never write "
        "'Islam does not mention'. State only what a cited passage says. Cite no passage in this case. "
        "Never quote source "
        "wording or place source wording in quotation marks. Add inline markers exactly like "
        "[quran:2:153], using ASCII square brackets and one ref per bracket. Use only refs "
        "supplied below. Never mention grades or graders. Never write, invent, complete, or "
        "fabricate a hadith. If a user asks for invented hadith text, say that only source "
        "passages can be presented. Do not use markdown headings.\n\n"
        f"Question: {question}\n\n"
        f"Source passages:\n{sources}"
    )


def _retrieval_query(question: str) -> str:
    normalized = question.strip()
    normalized = re.sub(r"^what\s+is\s+(?:the\s+)?(?:ruling|rule|view)\s+(?:on|about|regarding)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^what\s+(?:do|does)\s+[^?]+?\s+say\s+(?:about|regarding)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^what\s+(?:does|do)\s+(?:the\s+)?quran\s+say\s+about\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:ماذا|ما)\s+يقول\s+(?:القرآن|القران)\s+(?:عن|بشأن|حول)\s+", "", normalized)
    normalized = re.sub(r"^[؟]+$", "", normalized).strip()
    normalized = re.sub(r"[؟?!.]+$", "", normalized).strip()
    return normalized or question


async def _request_rewrite_search_phrases(question: str) -> list[str]:
    model = os.getenv("REWRITE_MODEL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL")
    if not model:
        raise RuntimeError("REWRITE_MODEL is not set")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in backend/.env")
    if not base_url:
        raise RuntimeError("OPENROUTER_BASE_URL is not set in backend/.env")

    async with AsyncOpenAI(api_key=api_key, base_url=base_url) as client:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return 3 to 5 short search phrases made only of the names and nouns "
                        "the question is about, using English spellings from the translations. "
                        "For abstract topics, include 1 or 2 close synonyms. Omit generic words "
                        "that occur in most passages, such as Prophet when the question asks "
                        "what the Prophet said, and Islam. Do not add people or topics the "
                        "question did not name. Return JSON only in the form "
                        "{\"phrases\": [\"phrase\", ...]}."
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=120,
        )

    content = response.choices[0].message.content or ""
    result = json.loads(content)
    return [
        phrase.strip()
        for phrase in result.get("phrases", [])
        if isinstance(phrase, str) and phrase.strip()
    ][:5]


async def _rewrite_search_phrases(question: str) -> list[str]:
    try:
        phrases = await _request_rewrite_search_phrases(question)
    except Exception as error:
        reason = " ".join(str(error).splitlines())
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key:
            reason = reason.replace(api_key, "[redacted]")
        logger.warning(
            "Ask rewrite fallback to original question: %s: %s",
            type(error).__name__,
            reason,
        )
        return [question]

    if len(phrases) < 3:
        logger.warning(
            "Ask rewrite fallback to original question: rewrite returned fewer than 3 phrases"
        )
        return [question]
    return phrases


def retrieve(
    question: str, search_phrases: list[str] | None = None
) -> list[dict[str, Any]]:
    phrases = search_phrases or [question]
    best_by_ref: dict[str, dict[str, Any]] = {}
    seeded_refs_by_kind: dict[str, set[str]] = {"quran": set(), "hadith": set()}
    for phrase in phrases:
        retrieval_question = _retrieval_query(phrase)
        for kind in ("quran", "hadith"):
            candidates = search_with_similarity(retrieval_question, kind, k=10)
            for passage in candidates[:2]:
                seeded_refs_by_kind[kind].add(str(passage["ref"]))
            for passage in candidates:
                ref = passage["ref"]
                current = best_by_ref.get(ref)
                similarity = float(passage.get("similarity", 0.0))
                if current is None or similarity > float(current.get("similarity", 0.0)):
                    best_by_ref[ref] = passage

    passages: list[dict[str, Any]] = []
    for kind in ("quran", "hadith"):
        kind_passages = [
            passage for passage in best_by_ref.values()
            if passage.get("kind") == kind
        ]
        ranked = sorted(
            (passage for passage in kind_passages
             if str(passage["ref"]) in seeded_refs_by_kind[kind]),
            key=lambda passage: float(passage.get("similarity", 0.0)),
            reverse=True,
        )[:5]
        selected_refs = {str(passage["ref"]) for passage in ranked}
        remaining = sorted(
            (passage for passage in kind_passages
             if str(passage["ref"]) not in selected_refs),
            key=lambda passage: float(passage.get("similarity", 0.0)),
            reverse=True,
        )
        ranked.extend(remaining[: 5 - len(ranked)])
        passages.extend(ranked)
    return passages


def retrieve_hadith_with_similarity(text: str) -> list[dict[str, Any]]:
    return [dict(row) for row in search_with_similarity(text, "hadith", k=20)]


def lookup_claim_passages(text: str) -> list[dict[str, Any]]:
    retrieval_question = _retrieval_query(text)
    passages: list[dict[str, Any]] = []
    for kind in ("quran", "hadith"):
        passages.extend(search_with_similarity(retrieval_question, kind, k=20))
    return passages


def lookup_passage(ref: str) -> dict[str, Any] | None:
    with SessionLocal() as session:
        passage = session.query(Passage).filter(Passage.ref == ref).one_or_none()
        if passage is None:
            return None
        return {
            "ref": passage.ref,
            "kind": passage.kind,
            "collection": passage.collection,
            "arabic": passage.arabic,
            "english": passage.english,
            "narrator": passage.narrator,
            "grade": passage.grade,
            "graded_by": passage.graded_by,
            "grades": passage.grades,
            "url": passage.url,
        }


async def _text_chunks(question: str, passages: list[dict[str, Any]]) -> AsyncIterator[str]:
    client, model = _client_and_model()
    async with client:
        stream = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a primary-source finder. Follow the user's answer rules exactly.",
                },
                {"role": "user", "content": _prompt(question, passages)},
            ],
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content


async def validated_events(
    chunks: AsyncIterator[str],
    lookup: Callable[[str], Awaitable[dict[str, Any] | None]],
) -> AsyncIterator[StreamEvent]:
    pending = ""
    marker = ""
    in_marker = False
    close_marker = "]"

    async for chunk in chunks:
        pending += chunk
        while pending:
            if not in_marker:
                starts = [(pending.find(open_marker), open_marker) for open_marker in ("[", "【")]
                starts = [(index, open_marker) for index, open_marker in starts if index >= 0]
                if not starts:
                    yield StreamEvent("token", pending)
                    pending = ""
                    continue
                marker_start = min(index for index, _ in starts)
                open_marker = next(open_marker for index, open_marker in starts if index == marker_start)
                if marker_start:
                    yield StreamEvent("token", pending[:marker_start])
                close_marker = "]" if open_marker == "[" else "】"
                marker = open_marker
                pending = pending[marker_start + 1 :]
                in_marker = True
                continue

            marker_end = pending.find(close_marker)
            if marker_end == -1:
                marker += pending
                pending = ""
                continue

            marker += pending[: marker_end + 1]
            pending = pending[marker_end + 1 :]
            ref = marker[1:-1].strip()
            marker = ""
            in_marker = False

            passage = await lookup(ref)
            if passage is not None:
                yield StreamEvent("token", f"[{ref}]")
                yield StreamEvent("card", passage)

    if not in_marker and pending:
        yield StreamEvent("token", pending)
    yield StreamEvent("done", {})


async def stream_answer(question: str) -> AsyncIterator[StreamEvent]:
    search_phrases = await _rewrite_search_phrases(question)
    passages = await run_in_threadpool(retrieve, question, search_phrases)
    emitted = False
    try:
        async for event in validated_events(
            _text_chunks(question, passages),
            _lookup_passage_async,
        ):
            emitted = True
            yield event
    except Exception as error:
        status = getattr(error, "status_code", None)
        if status is None:
            response = getattr(error, "response", None)
            status = getattr(response, "status_code", None)
        yield StreamEvent("error", {"status": status, "code": status or "provider_error"})
        yield StreamEvent("done", {})
        return
    if not emitted:
        yield StreamEvent("done", {})


async def _lookup_passage_async(ref: str) -> dict[str, Any] | None:
    return await run_in_threadpool(lookup_passage, ref)


async def _single_chunk(content: str) -> AsyncIterator[str]:
    yield content


async def filter_supported_claim_passages(
    claim: str, passages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not passages:
        return []

    def lexical_stem(word: str) -> str:
        if len(word) > 5 and word.endswith("ing"):
            return word[:-3]
        if len(word) > 4 and word.endswith("ed"):
            return word[:-2]
        return word

    claim_words = {
        lexical_stem(word)
        for word in re.findall(r"[a-z0-9]+", claim.lower())
        if len(word) >= 4
    }

    def judge_rank(passage: dict[str, Any]) -> float:
        text = f"{passage.get('english') or ''} {passage.get('arabic') or ''}".lower()
        passage_words = {
            lexical_stem(word)
            for word in re.findall(r"[a-z0-9]+", text)
            if len(word) >= 4
        }
        overlap = len(claim_words.intersection(passage_words))
        return overlap, float(passage.get("similarity", 0.0))

    similarity_ranked = sorted(
        passages,
        key=lambda passage: float(passage.get("similarity", 0.0)),
        reverse=True,
    )
    quran_passages = [
        passage for passage in similarity_ranked
        if str(passage.get("kind", "")).lower() == "quran"
    ]
    hadith_passages = [
        passage for passage in similarity_ranked
        if str(passage.get("kind", "")).lower() != "quran"
    ]
    ranked_passages = quran_passages[:3] + hadith_passages[:8]
    source_lines = []
    for index, passage in enumerate(ranked_passages, 1):
        english = re.sub(r"\s+", " ", str(passage.get("english") or "")).strip()
        source_lines.append(f"PASSAGE {index} [{passage['ref']}]: {english[:400]}")

    client, model = _client_and_model()
    async with client:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "For each passage, decide whether it directly states or clearly expresses the proposition made by the claim. Return only valid JSON as an object mapping each passage ref to true or false. Do not explain. A passage supports when its wording, in paraphrase, affirms the claim's proposition even if the exact claim wording differs. If it merely mentions a related topic, use another detail, or does not establish the proposition, mark false.",
                },
                {
                    "role": "user",
                    "content": f"Claim: {claim}\n\n" + "\n".join(source_lines),
                },
            ],
        )

    content = response.choices[0].message.content
    print("CLAIM_JUDGE_PASSAGES=" + json.dumps(source_lines, ensure_ascii=True))
    raw_output = (content or "<empty>").encode("ascii", "backslashreplace").decode("ascii")
    print("CLAIM_JUDGE_RAW_OUTPUT=" + raw_output)
    if not content:
        return []
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        decisions = json.loads(cleaned)
    except json.JSONDecodeError:
        decisions = {}
        for match in re.finditer(r"([a-z]+:[^\"\s,]+)\"?\s*:\s*(true|false)", cleaned.lower(), re.IGNORECASE):
            decisions[match.group(1)] = match.group(2) == "true"
    if isinstance(decisions, dict):
        normalized_decisions: dict[str, bool] = {}
        for key, value in decisions.items():
            label = str(key).strip().lower()
            match = re.fullmatch(r"passage\s+(\d+)", label)
            if match:
                index = int(match.group(1)) - 1
                if 0 <= index < len(passages):
                    normalized_decisions[passages[index]["ref"]] = value is True
            else:
                normalized_decisions[str(key)] = value is True
        return [passage for passage in passages if normalized_decisions.get(passage["ref"]) is True]
    if isinstance(decisions, list):
        supported_refs = {
            item["ref"] for item in decisions
            if isinstance(item, dict) and str(item.get("supported", "")).lower() == "true"
        }
        return [passage for passage in passages if passage["ref"] in supported_refs]
    return []


async def summarize_claim(
    text: str,
    passages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    client, model = _client_and_model()
    async with client:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You summarize primary sources without quoting their text. "
                    "Never state a ruling or mention scholars, madhhabs, grades, or graders.",
                },
                {"role": "user", "content": _prompt(text, passages)},
            ],
        )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("The OpenRouter claim summary response had no text content")

    answer_parts: list[str] = []
    cited_passages: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    async for event in validated_events(_single_chunk(content), _lookup_passage_async):
        if event.event == "token":
            answer_parts.append(str(event.data))
        elif event.event == "card":
            passage = event.data
            ref = str(passage["ref"])
            if ref not in seen_refs:
                seen_refs.add(ref)
                cited_passages.append(passage)
    supported_refs = []
    for passage in passages:
        ref = str(passage["ref"])
        if ref not in seen_refs:
            seen_refs.add(ref)
            cited_passages.append(passage)
        supported_refs.append(ref)
    return "".join(answer_parts), cited_passages
