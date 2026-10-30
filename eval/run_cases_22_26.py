from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from eval.run import load_questions
from app.main import app


async def run_case(question: dict) -> bool | None:
    case_id = question["id"]
    expected_refs = set(question.get("expected_refs", [])) if case_id in (22, 23) else set()
    print(f"CASE {case_id} INPUT={question['input']!r}")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/verify", json={"text": question["input"], "mode": "claim"}, timeout=60
        )

    print(f"CASE {case_id} HTTP_STATUS={response.status_code}")
    if response.status_code >= 400:
        print(f"CASE {case_id} HTTP_ERROR={response.text}")
        return None

    if "text/event-stream" in response.headers.get("content-type", ""):
        event_type = ""
        for line in response.text.splitlines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = payload
                if event_type == "error":
                    print(f"CASE {case_id} ERROR_EVENT={data!r}")
                    return None
                event_type = ""

    try:
        result = response.json()
    except ValueError:
        print(f"CASE {case_id} RESPONSE_ERROR={response.text!r}")
        return None

    status = result.get("status")
    refs = [passage.get("ref") for passage in result.get("passages", []) if passage.get("ref")]
    answer = result.get("answer", "")
    print(f"CASE {case_id} STATUS={status!r}")
    print(f"CASE {case_id} CITED_REFS={refs!r}")
    if not answer.strip():
        print(f"CASE {case_id} EMPTY_ANSWER raw_response={result!r}")
        return None

    if case_id in (22, 23):
        passed = bool(expected_refs.intersection(refs))
        expected = f"cite at least one of {sorted(expected_refs)!r}"
    else:
        passed = not refs
        expected = "cite nothing"
    print(f"CASE {case_id} EXPECTED_OUTCOME={expected}; {'PASS' if passed else 'FAIL'}")
    return passed


async def main() -> None:
    questions = {question["id"]: question for question in load_questions()}
    for index, case_id in enumerate((22, 23, 24, 25, 26)):
        result = await run_case(questions[case_id])
        if result is None:
            return
        if index < 4:
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
