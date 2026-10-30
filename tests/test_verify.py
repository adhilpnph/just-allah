from __future__ import annotations

import asyncio
from typing import Any

import app.services.llm as llm
import app.routers.verify as verify_router
from app.schemas import VerifyRequest


def _verify(request: VerifyRequest) -> Any:
    return asyncio.run(verify_router.verify(request))


def _passage(ref: str, similarity: float = 0.93) -> dict[str, Any]:
    return {
        "ref": ref,
        "kind": "hadith",
        "collection": "Test collection",
        "arabic": "Arabic from database",
        "english": "Known hadith text and paraphrased narration by the Prophet about Allah",
        "narrator": None,
        "grade": None,
        "graded_by": None,
        "grades": [{"name": "grader", "grade": "Sahih"}],
        "url": f"https://example.test/{ref}",
        "similarity": similarity,
    }


def test_real_hadith_match_returns_database_passage(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        verify_router,
        "retrieve_hadith_with_similarity",
        lambda _: [_passage("abudawud:407")],
    )

    response = _verify(VerifyRequest(text="known hadith text"))

    assert response.status == "matched"
    assert response.passages[0].ref == "abudawud:407"
    assert response.passages[0].english.startswith("Known hadith text")
    assert response.passages[0].grades == [{"name": "grader", "grade": "Sahih"}]
    assert response.similarity == 0.93


def test_paraphrased_hadith_above_threshold_returns_match(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        verify_router,
        "retrieve_hadith_with_similarity",
        lambda _: [_passage("abudawud:407", 0.8983)],
    )

    response = _verify(VerifyRequest(text="paraphrased known hadith"))

    assert response.status == "matched"
    assert response.passages[0].ref == "abudawud:407"


def test_selects_highest_similarity_candidate(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        verify_router,
        "retrieve_hadith_with_similarity",
        lambda _: [
            _passage("abudawud:407", 0.86),
            _passage("bukhari:1", 0.93),
        ],
    )

    response = _verify(VerifyRequest(text="known hadith text"))

    assert response.status == "matched"
    assert response.passages[0].ref == "bukhari:1"
    assert response.similarity == 0.93


def test_low_similarity_is_reported_as_unverified(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        verify_router,
        "retrieve_hadith_with_similarity",
        lambda _: [_passage("abudawud:407", 0.4)],
    )

    response = _verify(VerifyRequest(text="unmatched text"))

    assert response.status == "unverified"
    assert response.answer == "This hadith cannot be verified in the sources."
    assert response.passages == []


def test_negated_daytime_travel_does_not_match_night_travel_source(monkeypatch: Any) -> None:
    passage = _passage("abudawud:2571", 0.942647)
    passage["english"] = "Keep to travelling by night, for the earth is traversed more easily by night"
    monkeypatch.setattr(verify_router, "retrieve_hadith_with_similarity", lambda _: [passage])

    response = _verify(
        VerifyRequest(
            text=(
                "Anas reported the Apostle of Allah as saying do not travel in the day, "
                "for the earth is easier to travel by night"
            )
        )
    )

    assert response.status == "unverified"


def test_fast_continuously_day_and_night_is_not_rejected_as_travel(monkeypatch: Any) -> None:
    passage = _passage("bukhari:1967", 0.93)
    passage["english"] = "Do not fast continuously day and night and continue until Suhur."
    monkeypatch.setattr(verify_router, "retrieve_hadith_with_similarity", lambda _: [passage])

    response = _verify(
        VerifyRequest(
            text="Do not fast continuously day and night and continue until Suhur"
        )
    )

    assert response.status == "matched"


def test_material_contradiction_is_unverified(monkeypatch: Any) -> None:
    passage = _passage("bukhari:3974", 0.95)
    passage["english"] = "The sword was decorated with silver."
    monkeypatch.setattr(verify_router, "retrieve_hadith_with_similarity", lambda _: [passage])

    response = _verify(VerifyRequest(text="The sword was decorated with gold"))

    assert response.status == "unverified"


def test_supplied_paraphrased_night_claim_with_unrelated_source_is_unverified(monkeypatch: Any) -> None:
    passage = _passage("muslim:1627", 0.879)
    passage["english"] = (
        "Anas reported that when the Messenger of Allah had to set out on a journey "
        "hurriedly, he delayed the noon prayer and combined it with the afternoon prayer."
    )
    monkeypatch.setattr(verify_router, "retrieve_hadith_with_similarity", lambda _: [passage])

    response = _verify(
        VerifyRequest(
            text=(
                "Anas narrated that the Prophet advised making journeys during the night, "
                "since distances are shorter and easier to cover after dark."
            )
        )
    )

    assert response.status == "unverified"


def test_claim_mode_returns_only_database_validated_citations(monkeypatch: Any) -> None:
    sources = [_passage("quran:2:153")]
    monkeypatch.setattr(verify_router, "lookup_claim_passages", lambda _: sources)
    async def filter_sources(_: str, received_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        assert received_sources == sources
        return sources

    monkeypatch.setattr(verify_router, "filter_supported_claim_passages", filter_sources)

    async def summarize(
        _: str, received_sources: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        assert received_sources == sources
        return "The sources describe patience [quran:2:153].", [sources[0]]

    monkeypatch.setattr(verify_router, "summarize_claim", summarize)

    response = _verify(VerifyRequest(text="a claim", mode="claim"))

    assert response.status == "claim"
    assert response.answer.endswith("[quran:2:153].")
    assert [passage.ref for passage in response.passages] == ["quran:2:153"]


def test_claim_judge_selects_top_three_quran_and_eight_hadith_with_400_char_text(
    monkeypatch: Any,
) -> None:
    captured_messages: list[dict[str, str]] = []

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            captured_messages.extend(kwargs["messages"])
            return type("Response", (), {"choices": [type("Choice", (), {"message": type("Message", (), {"content": "{}"})()})()]})()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *_: Any) -> None:
            return None

    monkeypatch.setattr(llm, "_client_and_model", lambda: (FakeClient(), "mock-model"))
    passages = [
        {
            "ref": f"quran:2:{number}",
            "kind": "quran",
            "similarity": 1.0 - number / 100,
            "english": f"Quran excerpt {number} " + "q" * 500,
            "arabic": "",
        }
        for number in range(1, 6)
    ] + [
        {
            "ref": f"bukhari:{number}",
            "kind": "hadith",
            "similarity": 1.0 - number / 100,
            "english": f"Hadith excerpt {number} " + "h" * 500,
            "arabic": "",
        }
        for number in range(1, 11)
    ]

    asyncio.run(llm.filter_supported_claim_passages("some claim", passages))

    user_message = captured_messages[1]["content"]
    source_lines = user_message.split("\n\n", 1)[1].splitlines()
    expected_refs = [f"quran:2:{number}" for number in range(1, 4)] + [
        f"bukhari:{number}" for number in range(1, 9)
    ]
    assert len(source_lines) == 11
    assert [line.split(" [", 1)[1].split("]", 1)[0] for line in source_lines] == expected_refs
    assert all(len(line.split("]: ", 1)[1]) == 400 for line in source_lines)
