"""Load and validate the manually verified Phase 3 golden dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REQUIRED_FIELDS = {
    "question",
    "ground_truth_answer",
    "ground_truth_sources",
}
_OPTIONAL_FIELDS = {"should_decline"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One manually verified expected behavior for the RAG pipeline."""

    question: str
    ground_truth_answer: str
    ground_truth_sources: tuple[str, ...]
    should_decline: bool = False


def _nonempty_string(value: Any, *, field: str, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"line {line_number}: {field} must be a nonempty string")
    return value.strip()


def _parse_case(record: Any, *, line_number: int) -> GoldenCase:
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: each record must be a JSON object")

    fields = set(record)
    unknown_fields = sorted(fields - _ALLOWED_FIELDS)
    if unknown_fields:
        raise ValueError(
            f"line {line_number}: unknown fields: {', '.join(unknown_fields)}"
        )

    missing_fields = sorted(_REQUIRED_FIELDS - fields)
    if missing_fields:
        raise ValueError(
            f"line {line_number}: missing required fields: "
            f"{', '.join(missing_fields)}"
        )

    question = _nonempty_string(
        record["question"],
        field="question",
        line_number=line_number,
    )
    answer = _nonempty_string(
        record["ground_truth_answer"],
        field="ground_truth_answer",
        line_number=line_number,
    )

    raw_sources = record["ground_truth_sources"]
    if not isinstance(raw_sources, list) or any(
        not isinstance(source, str) or not source.strip() for source in raw_sources
    ):
        raise ValueError(
            f"line {line_number}: ground_truth_sources must be a list of "
            "nonempty strings"
        )
    sources = tuple(source.strip() for source in raw_sources)
    if len(set(sources)) != len(sources):
        raise ValueError(
            f"line {line_number}: ground_truth_sources contains duplicate sources"
        )

    should_decline = record.get("should_decline", False)
    if type(should_decline) is not bool:
        raise ValueError(f"line {line_number}: should_decline must be a boolean")
    if should_decline and sources:
        raise ValueError(f"line {line_number}: decline cases must not list sources")
    if not should_decline and not sources:
        raise ValueError(
            f"line {line_number}: supported cases require ground_truth_sources"
        )

    return GoldenCase(
        question=question,
        ground_truth_answer=answer,
        ground_truth_sources=sources,
        should_decline=should_decline,
    )


def load_golden_dataset(
    path: str | Path,
    *,
    min_cases: int = 1,
) -> list[GoldenCase]:
    """Load JSONL records and reject invalid or ambiguous evaluation data."""

    if type(min_cases) is not int or min_cases < 1:
        raise ValueError("min_cases must be a positive integer")

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    cases: list[GoldenCase] = []
    seen_questions: set[str] = set()
    for line_number, raw_line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"line {line_number}: invalid JSON: {error.msg}"
            ) from error

        case = _parse_case(record, line_number=line_number)
        normalized_question = " ".join(case.question.split()).casefold()
        if normalized_question in seen_questions:
            raise ValueError(f"line {line_number}: duplicate question: {case.question}")
        seen_questions.add(normalized_question)
        cases.append(case)

    if len(cases) < min_cases:
        raise ValueError(
            f"golden dataset must contain at least {min_cases} cases; "
            f"found {len(cases)}"
        )
    return cases
