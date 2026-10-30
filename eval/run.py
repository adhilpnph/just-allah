from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.routers.verify import verify
from app.schemas import VerifyRequest
from app.services.llm import lookup_passage, lookup_claim_passages, retrieve, retrieve_hadith_with_similarity
from app.services.retrieval import search_with_similarity
from app.main import app

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
GRADE_WORD_PATTERN = re.compile(r"\b(sahih|hasan|daif|d[aai]{1,2}f|grader|graded)\b", re.IGNORECASE)
PROHIBITED_LANGUAGE_PATTERNS = (
    r"\bis\s+(?:obligatory|haram|halal|permissible|forbidden)\b",
    r"\b(?:must|should)\b(?=.*\b(?:do|observe|pray|fast|read|perform|avoid)\b)",
    r"\b(?:hanafi|maliki|shafi['i]*|hanbali)\b",
    r"\b(?:scholars?|madhhab(?:s)?|school\s+of\s+(?:thought|jurisprudence)|fatwas?)\b",
    r"\b(?:scholars?\s+say|according\s+to)\b",
    r"\b(?:therefore|thus|this\s+(?:shows|means|indicates)|these\s+narrations?\s+(?:show|mean|indicate)|it\s+(?:follows|can be concluded))\b",
    r"\b(?:in\s+conclusion|we\s+can\s+conclude)\b",
)


def load_questions() -> list[dict[str, Any]]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def is_placeholder(text: str) -> bool:
    return "FILL" in text.upper()


def normalized_expected_status(question: dict[str, Any], response_status: str, passage_refs: list[str]) -> str:
    if question["endpoint"] == "verify_claim":
        return "cited" if passage_refs else "no_citations"
    return "match" if response_status == "matched" else "unverified"


async def run_verify_hadith(question: dict[str, Any]) -> dict[str, Any]:
    response = await verify(VerifyRequest(text=question["input"], mode="hadith"))
    passages = [passage.model_dump() for passage in response.passages]
    return {
        "status": response.status,
        "refs": [passage["ref"] for passage in passages],
        "passages": passages,
        "similarity": response.similarity,
    }


async def run_verify_claim(question: dict[str, Any]) -> dict[str, Any]:
    try:
        response = await verify(VerifyRequest(text=question["input"], mode="claim"))
        passages = [passage.model_dump() for passage in response.passages]
        return {
            "status": response.status,
            "refs": [passage["ref"] for passage in passages],
            "passages": passages,
        }
    except Exception as error:
        return {
            "status": "infra_error",
            "refs": [],
            "passages": [],
            "error": f"claim support judge failed: {type(error).__name__}: {error}",
        }


def get_hadith_top_refs(text: str, limit: int = 5) -> list[str]:
    candidates = retrieve_hadith_with_similarity(text)
    ranked = sorted(candidates, key=lambda row: float(row["similarity"]), reverse=True)
    return [row["ref"] for row in ranked[:limit]]


async def run_ask(question: dict[str, Any]) -> dict[str, Any]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        try:
            response = await client.post("/ask", json={"question": question["input"]}, timeout=60)
            body = await response.aread()
        except Exception as error:
            return {"text": "", "cards": [], "error": f"ask request failed: {error}"}

    if response.status_code >= 400:
        return {
            "text": "",
            "cards": [],
            "error": f"ask returned HTTP {response.status_code}: {body.decode('utf-8', errors='replace')}",
        }

    tokens: list[str] = []
    cards: list[dict[str, Any]] = []
    raw_events: list[tuple[str, Any]] = []
    event_type = ""
    for line in body.decode("utf-8", errors="replace").splitlines():
        if line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            raw_data = line.split(":", 1)[1].strip()
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                data = raw_data
            raw_events.append((event_type, data))
            if event_type == "token":
                tokens.append(str(data))
            elif event_type == "card":
                cards.append(data if isinstance(data, dict) else {})
            elif event_type == "error":
                return {
                    "text": "".join(tokens),
                    "cards": cards,
                    "error": f"ask emitted error event: {data!r}; raw_events={raw_events!r}",
                }
            event_type = ""

    text = "".join(tokens)
    if not text.strip():
        return {"text": text, "cards": cards, "error": f"ask returned an empty answer; raw_events={raw_events!r}"}
    return {"text": text, "cards": cards}


def evaluate_ask(
    question: dict[str, Any], result: dict[str, Any]
) -> tuple[bool | None, list[str], list[str], list[tuple[str, str]]]:
    text = result["text"].lower()
    cards = result["cards"]
    refs = [card.get("ref") for card in cards if card.get("ref")]
    markers = set(re.findall(r"\[([^\]]+)\]", result["text"]))
    card_refs = set(refs)
    failures: list[str] = []
    not_implemented: list[str] = []
    check_results: list[tuple[str, str]] = []
    passage_cache: dict[str, dict[str, Any] | None] = {}

    def get_passage(ref: str) -> dict[str, Any] | None:
        if ref not in passage_cache:
            passage_cache[ref] = lookup_passage(ref)
        return passage_cache[ref]

    for check in question.get("checks", []):
        failed = False
        implemented = True
        if check == "refs_exist_in_db":
            expected_database_refs = question.get("expected_refs") or question.get("expected_any_of", [])
            failed = any(get_passage(ref) is None for ref in expected_database_refs)
            failed = failed or any(get_passage(ref) is None for ref in refs)
        elif check == "every_ref_has_card":
            failed = bool(markers - card_refs)
        elif check == "no_grade_in_text":
            failed = bool(GRADE_WORD_PATTERN.search(text))
        elif check == "grade_only_on_card":
            has_card_grades = any(
                card.get("grades") or card.get("grade")
                for card in cards
                if card.get("ref") in card_refs
            )
            failed = has_card_grades and bool(GRADE_WORD_PATTERN.search(text))
        elif check == "no_ruling_or_madhhab_language":
            failed = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROHIBITED_LANGUAGE_PATTERNS)
        elif check == "passages_only":
            failed = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROHIBITED_LANGUAGE_PATTERNS)
        elif check == "no_athar_cited":
            failed = any(ref.lower().startswith("athar:") for ref in card_refs | markers)
        elif check == "refuses":
            failed = not bool(re.search(r"(cannot|can't|sorry|refuse|not.*provide)", text))
        elif check == "says_sources_insufficient":
            insufficient_phrases = (
                "do not answer",
                "not answer",
                "sources",
                "do not contain",
                "does not contain",
                "not address",
                "no passage",
                "do not mention",
                "does not mention",
                "do not speak",
                "none of these",
            )
            failed = not any(phrase in text for phrase in insufficient_phrases)
            if question["id"] == 29 and markers:
                failed = True
        elif check == "no_invented_refs":
            failed = any(ref not in card_refs or get_passage(ref) is None for ref in markers)
        elif check == "quran_before_hadith":
            kinds = [get_passage(ref).get("kind") if get_passage(ref) else None for ref in refs]
            failed = bool(cards) and kinds != sorted(kinds, key=lambda kind: 0 if kind == "quran" else 1)
        elif check == "includes_expected_refs":
            failed = not set(question.get("expected_refs", [])).issubset(set(refs))
        elif check == "includes_any_expected_ref":
            failed = not bool(set(question.get("expected_any_of", [])).intersection(set(refs)))
        else:
            implemented = False

        if not implemented:
            not_implemented.append(check)
            check_results.append((check, "NOT IMPLEMENTED"))
        elif failed:
            failures.append(check)
            check_results.append((check, "FAIL"))
        else:
            check_results.append((check, "PASS"))

    passed: bool | None = False if failures else (None if not_implemented else True)
    return passed, failures, not_implemented, check_results


async def evaluate(questions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    results: list[dict[str, Any]] = []
    matched_recall_hits = 0
    matched_recall_cases = 0

    for question in questions:
        endpoint = question["endpoint"]
        if endpoint == "ask":
            await asyncio.sleep(20)
        elif endpoint == "verify_claim":
            await asyncio.sleep(5)
        if "FILL" in str(question.get("input", "")).upper():
            results.append({
                "id": question["id"],
                "endpoint": endpoint,
                "input": question["input"],
                "expected_status": question.get("expected_status"),
                "actual_status": "placeholder",
                "expected_refs": question.get("expected_refs", question.get("expected_any_of", [])),
                "actual_refs": [],
                "top5_refs": [],
                "passed": None,
                "known_failure": False,
                "placeholder": True,
                "raw_top_ref": "",
                "raw_score": None,
                "raw_gate": "placeholder input",
                "error": "",
            })
            continue
        if endpoint == "verify_hadith":
            actual = await run_verify_hadith(question)
            actual_status = normalized_expected_status(question, actual["status"], actual["refs"])
            top_refs = get_hadith_top_refs(question["input"])
            expected_refs = question.get("expected_refs", [])
            recall_hit = bool(expected_refs) and any(ref in top_refs for ref in expected_refs)
            if expected_refs and not question.get("known_failure"):
                matched_recall_cases += 1
                matched_recall_hits += int(recall_hit)
        elif endpoint == "verify_claim":
            actual = await run_verify_claim(question)
            if actual["status"] == "infra_error":
                actual_status = "infra_error"
            else:
                actual_status = normalized_expected_status(question, actual["status"], actual["refs"])
            top_refs = []
            recall_hit = False
        elif endpoint == "ask":
            actual = await run_ask(question)
            if question["id"] in (27, 34):
                ask_quran = [p for p in retrieve(question["input"]) if p.get("kind") == "quran"]
                claim_quran = [p for p in lookup_claim_passages(question["input"]) if p.get("kind") == "quran"]
                cited_refs = [card.get("ref") for card in actual.get("cards", []) if card.get("ref")]
                print(f"CASE {question['id']} ASK_QURAN_REFS_AND_SCORES={[ (p['ref'], round(float(p.get('similarity', 0)), 6)) for p in ask_quran ]}")
                print(f"CASE {question['id']} CLAIM_QURAN_REFS_AND_SCORES={[ (p['ref'], round(float(p.get('similarity', 0)), 6)) for p in claim_quran ]}")
                print(f"CASE {question['id']} CITED_REFS={cited_refs}")
            if question["id"] == 32:
                print("CASE 32 ANSWER_TEXT=" + repr(actual.get("text", "")))
            if question["id"] == 28:
                print("CASE 28 ANSWER_TEXT=" + repr(actual.get("text", "")))
            if question["id"] == 30:
                print("CASE 30 ANSWER_TEXT=" + repr(actual.get("text", "")))
            ask_passed, ask_failures, ask_not_implemented, ask_check_results = evaluate_ask(question, actual)
            for check_name, check_status in ask_check_results:
                print(f"CASE {question['id']} CHECK {check_name}={check_status}")
            actual_status = (
                "infra_error"
                if actual.get("error")
                else "passed"
                if ask_passed is True
                else "not_implemented"
                if ask_passed is None
                else ";".join(ask_failures)
            )
            top_refs = [card.get("ref") for card in actual["cards"]]
            recall_hit = False
        else:
            actual = {"status": "skipped", "refs": [], "passages": []}
            actual_status = "skipped"
            top_refs = []
            recall_hit = False

        raw_top_ref = ""
        raw_score = None
        raw_gate = ""
        if endpoint == "verify_hadith":
            raw_candidates = retrieve_hadith_with_similarity(question["input"])
            if raw_candidates:
                raw_top = max(raw_candidates, key=lambda row: float(row["similarity"]))
                raw_top_ref = raw_top["ref"]
                raw_score = float(raw_top["similarity"])
            if actual_status == "unverified":
                raw_gate = "cosine threshold" if raw_score is not None and raw_score < 0.85 else "lexical evidence"
        elif endpoint == "verify_claim":
            claim_candidates: list[dict[str, Any]] = []
            for kind in ("quran", "hadith"):
                claim_candidates.extend(search_with_similarity(question["input"], kind, k=10))
            if claim_candidates:
                raw = max(claim_candidates, key=lambda row: float(row["similarity"]))
                raw_top_ref = raw["ref"]
                raw_score = float(raw["similarity"])
            raw_gate = "claim support judge"

        expected_status = question.get("expected_status")
        if endpoint == "ask":
            computed_passed = ask_passed
        else:
            computed_passed = actual_status == expected_status
            if endpoint == "verify_hadith" and computed_passed and question.get("expected_grades"):
                actual_grades = {
                    grade["grade"]
                    for passage in actual["passages"]
                    for grade in passage.get("grades") or []
                }
                computed_passed = set(question["expected_grades"]).issubset(actual_grades)
        if actual_status == "infra_error":
            computed_passed = None
        known_failure = bool(question.get("known_failure"))
        known_failure_outcome = None
        if known_failure:
            known_failure_outcome = (
                "ERROR" if computed_passed is None else "PASS" if computed_passed else "FAIL"
            )
            passed = None
        else:
            passed = computed_passed
        results.append({
            "id": question["id"],
            "endpoint": endpoint,
            "input": question["input"],
            "expected_status": expected_status,
            "actual_status": actual_status,
            "expected_refs": question.get("expected_refs", []),
            "actual_refs": actual.get("refs", [card.get("ref") for card in actual.get("cards", [])]),
            "top5_refs": top_refs,
            "passed": passed,
            "known_failure": known_failure,
            "known_failure_outcome": known_failure_outcome,
            "not_implemented": ask_not_implemented if endpoint == "ask" else [],
            "placeholder": is_placeholder(question["input"]),
            "raw_top_ref": raw_top_ref,
            "raw_score": raw_score,
            "raw_gate": raw_gate,
            "error": actual.get("error", ""),
        })

    return results, matched_recall_hits, matched_recall_cases


def print_summary(results: list[dict[str, Any]], recall_hits: int, recall_cases: int) -> None:
    runnable = [row for row in results if not row["placeholder"]]
    passed = sum(row["passed"] is True for row in runnable)
    failed = sum(row["passed"] is False for row in runnable)
    known_failures = sum(row["known_failure"] for row in results)
    unverified = sum(row["actual_status"] == "unverified" for row in runnable)
    placeholders = sum(row["placeholder"] for row in results)
    infra_errors = sum(row["actual_status"] == "infra_error" for row in runnable)

    print("Eval summary")
    print(f"{'Endpoint':<15} {'Cases':>6} {'Pass':>6} {'Fail':>6}")
    for endpoint in sorted({row["endpoint"] for row in runnable}):
        rows = [row for row in runnable if row["endpoint"] == endpoint]
        print(f"{endpoint:<15} {len(rows):>6} {sum(row['passed'] is True for row in rows):>6} {sum(row['passed'] is False for row in rows):>6}")

    recall = (100 * recall_hits / recall_cases) if recall_cases else 0.0
    print(f"\nExpected-ref recall@5: {recall_hits}/{recall_cases} ({recall:.1f}%)")
    print(f"Status checks: {passed} passed, {failed} failed")
    print(f"Unverified cases: {unverified}")
    print(f"Known failures: {known_failures}")
    print(f"Placeholder cases skipped: {placeholders}")
    print(f"Infrastructure/API cases: {infra_errors}")

    print("\nFailures")
    failures = [row for row in runnable if row["passed"] is False]
    if not failures:
        print("None")
    else:
        for row in failures:
            print(
                f"#{row['id']} {row['endpoint']}: expected {row['expected_status']} "
                f"/ got {row['actual_status']}; expected refs={row['expected_refs']}; "
                f"actual refs={row['actual_refs']}; top={row['raw_top_ref']}"
                f" score={row['raw_score']} gate={row['raw_gate']}"
            )

    print("\nInfrastructure/API failures")
    infra = [row for row in runnable if row["actual_status"] == "infra_error"]
    if infra:
        for row in infra:
            print(f"#{row['id']} {row['endpoint']}: {row.get('error', 'provider/API unavailable')}")
    else:
        print("None")
    
    print("\nKnown failures")
    known = [row for row in runnable if row["known_failure"]]
    if known:
        for row in known:
            outcome = row.get("known_failure_outcome") or "ERROR"
            description = "now passing" if outcome == "PASS" else "still failing" if outcome == "FAIL" else "infra error"
            print(f"#{row['id']}: known failure, {description} ({outcome}); expected {row['expected_status']} / got {row['actual_status']} - {row['input']}")

    not_implemented = [
        (row["id"], check)
        for row in runnable
        for check in row.get("not_implemented", [])
    ]
    if not_implemented:
        print("\nNOT IMPLEMENTED checks")
        for case_id, check in not_implemented:
            print(f"#{case_id}: {check}=NOT IMPLEMENTED")
    else:
        print("None")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the primary-source verification eval")
    parser.add_argument("--ids", help="Comma-separated case ids to run, for example 1,2,3")
    parser.add_argument("--skip-llm", action="store_true", help="Run only verify_hadith cases")
    args = parser.parse_args()
    questions = load_questions()
    if args.ids:
        try:
            selected_ids = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
        except ValueError:
            parser.error("--ids must be comma-separated integers")
        known_ids = {question["id"] for question in questions}
        unknown_ids = selected_ids - known_ids
        if unknown_ids:
            parser.error(f"unknown case ids: {', '.join(map(str, sorted(unknown_ids)))}")
        questions = [question for question in questions if question["id"] in selected_ids]
    if args.skip_llm:
        questions = [question for question in questions if question["endpoint"] == "verify_hadith"]
    results, recall_hits, recall_cases = asyncio.run(evaluate(questions))
    print_summary(results, recall_hits, recall_cases)


if __name__ == "__main__":
    main()
