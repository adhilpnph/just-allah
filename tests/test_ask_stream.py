from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import app.services.llm as llm
from app.services.llm import StreamEvent, validated_events


async def _chunks(*values: str) -> AsyncIterator[str]:
    for value in values:
        yield value


def _collect(
    chunks: AsyncIterator[str],
    lookup: Any,
) -> list[StreamEvent]:
    return asyncio.run(_collect_async(chunks, lookup))


async def _collect_async(
    chunks: AsyncIterator[str],
    lookup: Any,
) -> list[StreamEvent]:
    return [event async for event in validated_events(chunks, lookup)]


def test_marker_split_across_chunks_emits_verified_card_and_keeps_marker() -> None:
    async def lookup(ref: str) -> dict[str, Any] | None:
        assert ref == "quran:2:153"
        return {"ref": ref, "english": "Database text"}

    events = _collect(_chunks("Trust [quran:2", ":153] in patience."), lookup)

    assert [(event.event, event.data) for event in events] == [
        ("token", "Trust "),
        ("token", "[quran:2:153]"),
        ("card", {"ref": "quran:2:153", "english": "Database text"}),
        ("token", " in patience."),
        ("done", {}),
    ]


def test_full_width_marker_emits_verified_card_and_ascii_marker() -> None:
    async def lookup(ref: str) -> dict[str, Any] | None:
        assert ref == "muslim:2154"
        return {"ref": ref, "english": "Database text"}

    events = _collect(_chunks("Trust 【muslim:2154】 in patience."), lookup)

    assert [(event.event, event.data) for event in events] == [
        ("token", "Trust "),
        ("token", "[muslim:2154]"),
        ("card", {"ref": "muslim:2154", "english": "Database text"}),
        ("token", " in patience."),
        ("done", {}),
    ]


def test_unknown_marker_is_dropped() -> None:
    async def lookup(ref: str) -> None:
        assert ref == "quran:999:999"
        return None

    events = _collect(_chunks("Before [quran:999:999] after"), lookup)

    assert "".join(event.data for event in events if event.event == "token") == "Before  after"
    assert all(event.event != "card" for event in events)
    assert events[-1] == StreamEvent("done", {})


def test_plain_text_without_markers_is_emitted_unchanged() -> None:
    async def lookup(ref: str) -> None:
        raise AssertionError(f"Unexpected lookup for {ref}")

    events = _collect(_chunks("A plain answer.", " More text."), lookup)

    assert "".join(event.data for event in events if event.event == "token") == (
        "A plain answer. More text."
    )
    assert all(event.event == "token" for event in events[:-1])
    assert events[-1] == StreamEvent("done", {})


def test_retrieval_requests_ten_and_sends_top_five_each(monkeypatch: Any) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_search_with_similarity(question: str, kind: str, k: int) -> list[dict[str, Any]]:
        calls.append((question, kind, k))
        return [
            {"ref": f"{kind}:{i}", "kind": kind, "similarity": 0.9 - i / 100, "english": "text"}
            for i in range(6)
        ]

    monkeypatch.setattr(llm, "search_with_similarity", fake_search_with_similarity)

    passages = llm.retrieve("question")
    assert calls == [("question", "quran", 10), ("question", "hadith", 10)]
    assert [passage["kind"] for passage in passages].count("quran") == 5
    assert [passage["kind"] for passage in passages].count("hadith") == 5


def test_retrieval_seeds_two_per_phrase_then_fills_by_similarity(monkeypatch: Any) -> None:
    calls: list[tuple[str, str, int]] = []
    results = {
        ("Lut", "quran"): [
            {"ref": f"quran:{index}", "kind": "quran", "similarity": score}
            for index, score in enumerate((0.9, 0.8, 0.99, 0.6, 0.5), start=1)
        ],
        ("Lot", "quran"): [
            {"ref": "quran:1", "kind": "quran", "similarity": 0.95},
            {"ref": "quran:6", "kind": "quran", "similarity": 0.94},
        ],
        ("Lut", "hadith"): [
            {"ref": "hadith:1", "kind": "hadith", "similarity": 0.4}
        ],
        ("Lot", "hadith"): [
            {"ref": "hadith:1", "kind": "hadith", "similarity": 0.7}
        ],
    }

    def fake_search_with_similarity(query: str, kind: str, k: int) -> list[dict[str, Any]]:
        calls.append((query, kind, k))
        return results.get((query, kind), [])

    monkeypatch.setattr(llm, "search_with_similarity", fake_search_with_similarity)

    passages = llm.retrieve("Prophet Lut", ["Lut", "Lot"])

    assert calls == [
        ("Lut", "quran", 10), ("Lut", "hadith", 10),
        ("Lot", "quran", 10), ("Lot", "hadith", 10),
    ]
    assert [passage["ref"] for passage in passages if passage["kind"] == "quran"] == [
        "quran:1", "quran:6", "quran:2", "quran:3", "quran:4"
    ]
    assert len([passage for passage in passages if passage["kind"] == "quran"]) == 5
    assert next(p["similarity"] for p in passages if p["ref"] == "quran:1") == 0.95
    assert next(p["similarity"] for p in passages if p["ref"] == "hadith:1") == 0.7


def test_rewrite_uses_separate_model_and_falls_back_on_provider_failure(monkeypatch: Any) -> None:
    monkeypatch.setenv("REWRITE_MODEL", "rewrite-test-model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://test.invalid/v1")
    observed_models: list[str] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            observed_models.append(kwargs["model"])
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": '{"phrases":["Lut","Lot","Prophet Lot"]}'},
                                )()
                            },
                        )()
                    ]
                },
            )()

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

    monkeypatch.setattr(llm, "AsyncOpenAI", FakeClient)

    assert asyncio.run(llm._rewrite_search_phrases("Prophet Lut")) == [
        "Lut", "Lot", "Prophet Lot"
    ]
    assert observed_models == ["rewrite-test-model"]

    class FailingClient(FakeClient):
        async def __aenter__(self) -> "FailingClient":
            raise RuntimeError("mock provider failure")

    monkeypatch.setattr(llm, "AsyncOpenAI", FailingClient)
    assert asyncio.run(llm._rewrite_search_phrases("Prophet Lut")) == ["Prophet Lut"]


def test_stream_answer_emits_error_and_done_for_provider_error(monkeypatch: Any) -> None:
    async def failing_chunks(question: str, passages: list[dict[str, Any]]) -> AsyncIterator[str]:
        error = RuntimeError("provider failed")
        error.status_code = 413
        raise error
        yield ""

    monkeypatch.setattr(llm, "retrieve", lambda question, search_phrases=None: [])
    async def original_query(question: str) -> list[str]:
        return [question]

    monkeypatch.setattr(llm, "_rewrite_search_phrases", original_query)
    monkeypatch.setattr(llm, "_text_chunks", failing_chunks)

    events = asyncio.run(_collect_stream_events("question"))
    assert events == [
        llm.StreamEvent("error", {"status": 413, "code": 413}),
        llm.StreamEvent("done", {}),
    ]


async def _collect_stream_events(question: str) -> list[StreamEvent]:
    return [event async for event in llm.stream_answer(question)]
