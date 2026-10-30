from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from eval.run import evaluate_ask, load_questions
from app.main import app


async def run_case(question: dict) -> bool:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/ask", json={"question": question["input"]}, timeout=60)
        print(f"CASE {question['id']} HTTP_STATUS={response.status_code}")
        if response.status_code >= 400:
            print(f"CASE {question['id']} HTTP_ERROR={response.text}")
            return False

        tokens: list[str] = []
        cards: list[dict] = []
        events: list[tuple[str, object]] = []
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
                events.append((event_type, data))
                if event_type == "token":
                    tokens.append(str(data))
                elif event_type == "card":
                    cards.append(data if isinstance(data, dict) else {})
                elif event_type == "error":
                    print(f"CASE {question['id']} ERROR_EVENT={data!r}")
                    print(f"CASE {question['id']} RAW_EVENTS={events!r}")
                    return False
                event_type = ""

    result = {"text": "".join(tokens), "cards": cards}
    print(f"CASE {question['id']} RAW_ANSWER={result['text']!r}")
    refs = [card.get("ref") for card in cards if card.get("ref")]
    print(f"CASE {question['id']} CITED_REFS={refs!r}")
    if not result["text"].strip():
        print(f"CASE {question['id']} EMPTY_ANSWER raw_events={events!r}")
        return False

    passed, failures = evaluate_ask(question, result)
    failed = set(failures)
    for check in question.get("checks", []):
        status = "FAIL" if check in failed else "PASS"
        if check == "no_ruling_or_madhhab_language" and check in failed:
            lower = result["text"].lower()
            patterns = (
                r"\bis\s+(?:obligatory|haram|halal|permissible|forbidden)\b",
                r"\b(?:must|should)\b(?=.*\b(?:do|observe|pray|fast|read|perform|avoid)\b)",
                r"\b(?:hanafi|maliki|shafi['i]*|hanbali)\b",
                r"\b(?:scholars?|madhhab(?:s)?|school\s+of\s+(?:thought|jurisprudence))\b",
                r"\b(?:scholars?\s+say|according\s+to)\b",
                r"\b(?:therefore|thus|this\s+(?:shows|means|indicates)|these\s+narrations?\s+(?:show|mean|indicate)|it\s+(?:follows|can be concluded))\b",
                r"\b(?:in\s+conclusion|we\s+can\s+conclude)\b",
            )
            match = next((re.search(pattern, lower, re.IGNORECASE) for pattern in patterns if re.search(pattern, lower, re.IGNORECASE)), None)
            matched_phrase = repr(match.group(0)) if match else "unknown"
            print(f"CASE {question['id']} CHECK {check}=FAIL matched_phrase={matched_phrase}")
        else:
            print(f"CASE {question['id']} CHECK {check}={status}")
    return passed


async def main() -> None:
    questions = {q["id"]: q for q in load_questions()}
    for index, case_id in enumerate((28, 29, 30)):
        ok = await run_case(questions[case_id])
        if not ok and not questions[case_id].get("checks"):
            return
        if index < 2:
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
