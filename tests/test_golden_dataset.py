"""Behavioral tests for the Phase 3 golden-dataset contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eval.dataset import GoldenCase, load_golden_dataset


class GoldenDatasetTests(unittest.TestCase):
    def write_dataset(self, lines: list[str]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "golden_dataset.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_loads_supported_and_decline_cases_from_jsonl(self) -> None:
        path = self.write_dataset(
            [
                '{"question":"What is hybrid search?",'
                '"ground_truth_answer":"It combines semantic and keyword search.",'
                '"ground_truth_sources":["docs/retrieval.md"]}',
                '{"question":"What is the capital of Mars?",'
                '"ground_truth_answer":"The corpus does not contain this information.",'
                '"ground_truth_sources":[],"should_decline":true}',
            ]
        )

        cases = load_golden_dataset(path)

        self.assertEqual(
            cases,
            [
                GoldenCase(
                    question="What is hybrid search?",
                    ground_truth_answer=(
                        "It combines semantic and keyword search."
                    ),
                    ground_truth_sources=("docs/retrieval.md",),
                    should_decline=False,
                ),
                GoldenCase(
                    question="What is the capital of Mars?",
                    ground_truth_answer=(
                        "The corpus does not contain this information."
                    ),
                    ground_truth_sources=(),
                    should_decline=True,
                ),
            ],
        )

    def test_rejects_invalid_json_with_the_line_number(self) -> None:
        path = self.write_dataset(
            [
                '{"question":"What is RAG?","ground_truth_answer":"Answer",'
                '"ground_truth_sources":["rag.pdf"]}',
                '{"question":',
            ]
        )

        with self.assertRaisesRegex(ValueError, r"line 2.*invalid JSON"):
            load_golden_dataset(path)

    def test_supported_case_requires_a_nonempty_answer_and_sources(self) -> None:
        missing_answer = self.write_dataset(
            [
                '{"question":"What is RAG?","ground_truth_answer":" ",'
                '"ground_truth_sources":["rag.pdf"]}'
            ]
        )
        missing_sources = self.write_dataset(
            [
                '{"question":"What is RAG?",'
                '"ground_truth_answer":"Retrieval-augmented generation.",'
                '"ground_truth_sources":[]}'
            ]
        )

        with self.assertRaisesRegex(ValueError, "ground_truth_answer"):
            load_golden_dataset(missing_answer)
        with self.assertRaisesRegex(ValueError, "ground_truth_sources"):
            load_golden_dataset(missing_sources)

    def test_decline_case_requires_an_expected_answer_and_no_sources(self) -> None:
        missing_answer = self.write_dataset(
            [
                '{"question":"Unknown?","ground_truth_answer":"",'
                '"ground_truth_sources":[],"should_decline":true}'
            ]
        )
        invented_source = self.write_dataset(
            [
                '{"question":"Unknown?","ground_truth_answer":"Decline.",'
                '"ground_truth_sources":["not-a-source.md"],'
                '"should_decline":true}'
            ]
        )

        with self.assertRaisesRegex(ValueError, "ground_truth_answer"):
            load_golden_dataset(missing_answer)
        with self.assertRaisesRegex(ValueError, "must not list sources"):
            load_golden_dataset(invented_source)

    def test_rejects_duplicate_questions_and_duplicate_or_blank_sources(self) -> None:
        duplicate_questions = self.write_dataset(
            [
                '{"question":"What is RAG?","ground_truth_answer":"Answer",'
                '"ground_truth_sources":["rag.pdf"]}',
                '{"question":"  what IS rag?  ","ground_truth_answer":"Answer",'
                '"ground_truth_sources":["rag.pdf"]}',
            ]
        )
        duplicate_sources = self.write_dataset(
            [
                '{"question":"What is RAG?","ground_truth_answer":"Answer",'
                '"ground_truth_sources":["rag.pdf","rag.pdf"]}'
            ]
        )
        blank_source = self.write_dataset(
            [
                '{"question":"What is RAG?","ground_truth_answer":"Answer",'
                '"ground_truth_sources":[" "]}'
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate question"):
            load_golden_dataset(duplicate_questions)
        with self.assertRaisesRegex(ValueError, "duplicate sources"):
            load_golden_dataset(duplicate_sources)
        with self.assertRaisesRegex(ValueError, "nonempty strings"):
            load_golden_dataset(blank_source)

    def test_rejects_unknown_fields_and_non_boolean_decline_flags(self) -> None:
        typo = self.write_dataset(
            [
                '{"question":"What is RAG?","ground_truth_answer":"Answer",'
                '"ground_truth_sources":["rag.pdf"],"ground_truth_source":[]}'
            ]
        )
        invalid_flag = self.write_dataset(
            [
                '{"question":"Unknown?","ground_truth_answer":"Decline.",'
                '"ground_truth_sources":[],"should_decline":"yes"}'
            ]
        )

        with self.assertRaisesRegex(ValueError, "unknown fields"):
            load_golden_dataset(typo)
        with self.assertRaisesRegex(ValueError, "should_decline"):
            load_golden_dataset(invalid_flag)

    def test_enforces_a_configurable_minimum_dataset_size(self) -> None:
        path = self.write_dataset(
            [
                '{"question":"What is RAG?","ground_truth_answer":"Answer",'
                '"ground_truth_sources":["rag.pdf"]}'
            ]
        )

        with self.assertRaisesRegex(ValueError, "at least 2"):
            load_golden_dataset(path, min_cases=2)
        with self.assertRaisesRegex(ValueError, "min_cases"):
            load_golden_dataset(path, min_cases=0)

    def test_missing_or_empty_dataset_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jsonl"
            empty = Path(directory) / "empty.jsonl"
            empty.write_text("\n  \n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                load_golden_dataset(missing)
            with self.assertRaisesRegex(ValueError, "at least 1"):
                load_golden_dataset(empty)

    def test_project_dataset_meets_phase3_size_breadth_and_source_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]

        cases = load_golden_dataset(
            project_root / "eval" / "golden_dataset.jsonl",
            min_cases=50,
        )

        supported = [case for case in cases if not case.should_decline]
        declines = [case for case in cases if case.should_decline]
        referenced_sources = {
            source for case in supported for source in case.ground_truth_sources
        }

        self.assertGreaterEqual(len(supported), 45)
        self.assertGreaterEqual(len(declines), 3)
        self.assertGreaterEqual(len(referenced_sources), 15)
        for source in referenced_sources:
            self.assertTrue((project_root / source).is_file(), source)


if __name__ == "__main__":
    unittest.main()
