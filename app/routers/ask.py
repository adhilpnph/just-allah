from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.schemas import AskRequest
from app.services.llm import StreamEvent, stream_answer

router = APIRouter()


def _sse(event: StreamEvent) -> str:
    return f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"


async def _response_events(question: str) -> AsyncIterator[str]:
    async for event in stream_answer(question):
        yield _sse(event)


@router.post("/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        _response_events(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
