from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

import app.routers.ask as ask_router
import app.routers.verify as verify_router
from app.main import app
from app.rate_limit import limiter


@pytest.fixture(autouse=True)
def reset_limits() -> None:
    limiter.reset()
    yield
    limiter.reset()


@pytest.mark.anyio
async def test_ask_limit_is_per_ip(monkeypatch: Any) -> None:
    from app.services.llm import StreamEvent

    async def stream_answer(_: str) -> AsyncIterator[StreamEvent]:
        yield StreamEvent("done", {})

    monkeypatch.setattr(ask_router, "stream_answer", stream_answer)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(5):
            response = await client.post(
                "/ask",
                json={"question": "A question"},
                headers={"X-Forwarded-For": "198.51.100.10, 10.0.0.1"},
            )
            assert response.status_code == 200

        limited = await client.post(
            "/ask",
            json={"question": "A question"},
            headers={"X-Forwarded-For": "198.51.100.10, 10.0.0.1"},
        )
        different_ip = await client.post(
            "/ask",
            json={"question": "A question"},
            headers={"X-Forwarded-For": "198.51.100.11"},
        )

    assert limited.status_code == 429
    assert limited.json() == {"detail": "Too many requests. Please wait and try again."}
    assert different_ip.status_code == 200


@pytest.mark.anyio
async def test_verify_limit_is_per_ip(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        verify_router,
        "retrieve_hadith_with_similarity",
        lambda _: [],
    )
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=True)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(10):
            response = await client.post(
                "/verify",
                json={"text": "Unknown hadith", "mode": "hadith"},
                headers={"X-Forwarded-For": "198.51.100.20"},
            )
            assert response.status_code == 200

        limited = await client.post(
            "/verify",
            json={"text": "Unknown hadith", "mode": "hadith"},
            headers={"X-Forwarded-For": "198.51.100.20"},
        )
        different_ip = await client.post(
            "/verify",
            json={"text": "Unknown hadith", "mode": "hadith"},
            headers={"X-Forwarded-For": "198.51.100.21"},
        )

    assert limited.status_code == 429
    assert limited.json() == {"detail": "Too many requests. Please wait and try again."}
    assert different_ip.status_code == 200
