"""Behavioral tests for collecting Phase 3 evaluation inputs."""

from __future__ import annotations

import unittest

from eval.dataset import GoldenCase
from eval.runner import evaluate_cases
from src.embed import RetrievedChunk
from src.generate import Citation, GeneratedAnswer, NO_CONTEXT_ANSWER
from src.pipeline import RAGRun


def chunk(record_id: str, text: str, source: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        id=record_id,
        text=text,
        metadata={"source": source, "chunk_index": rank - 1},
        distance=rank / 10,
        rank=rank,
    )


class FakePipeline:
    def __init__(self, runs: dict[str, RAGRun]) -> None:
        self.runs = runs
        self.calls: list[tuple[str, int]] = []

    def run(self, question: str, *, top_k: int) -> RAGRun:
        self.calls.append((question, top_k))
        return self.runs[question]


class EvaluationRunnerTests(unittest.TestCase):
    def test_collects_answer_contexts_retrieved_sources_and_citations(self) -> None:
        expected = chunk(
            "expected",
            "Hybrid search combines semantic and keyword retrieval.",
            "data/raw/docs/hybrid.mdx",
            1,
        )
        supporting = chunk(
            "supporting",
            "RRF combines ranked result lists.",
            "data/raw/docs/rrf.mdx",
            2,
        )
        answer = GeneratedAnswer(
            text="Hybrid search combines retrieval methods [1].",
            citations=[
                Citation(
                    number=1,
                    record_id="expected",
                    source="data/raw/docs/hybrid.mdx",
                    page_number=None,
                    chunk_index=0,
                )
            ],
        )
        case = GoldenCase(
            question="What is hybrid search?",
            ground_truth_answer="It combines semantic and keyword retrieval.",
            ground_truth_sources=("docs/hybrid.mdx",),
        )
        pipeline = FakePipeline(
            {
                case.question: RAGRun(
                    answer=answer,
                    retrieved_chunks=(expected, supporting),
                )
            }
        )

        results = evaluate_cases([case], pipeline=pipeline, top_k=2)

        self.assertEqual(pipeline.calls, [(case.question, 2)])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].case, case)
        self.assertEqual(results[0].actual_answer, answer.text)
        self.assertEqual(
            results[0].contexts,
            (expected.text, supporting.text),
        )
        self.assertEqual(
            results[0].retrieved_sources,
            ("data/raw/docs/hybrid.mdx", "data/raw/docs/rrf.mdx"),
        )
        self.assertEqual(
            results[0].cited_sources,
            ("data/raw/docs/hybrid.mdx",),
        )
        self.assertTrue(results[0].expected_source_retrieved)
        self.assertFalse(results[0].declined)

    def test_expected_decline_is_recorded_without_context_or_citations(self) -> None:
        case = GoldenCase(
            question="What is the capital of Mars?",
            ground_truth_answer="The corpus cannot answer this.",
            ground_truth_sources=(),
            should_decline=True,
        )
        pipeline = FakePipeline(
            {
                case.question: RAGRun(
                    answer=GeneratedAnswer(
                        text=NO_CONTEXT_ANSWER,
                        citations=[],
                    ),
                    retrieved_chunks=(),
                )
            }
        )

        result = evaluate_cases([case], pipeline=pipeline)[0]

        self.assertTrue(result.declined)
        self.assertFalse(result.expected_source_retrieved)
        self.assertTrue(result.retrieval_expectation_met)
        self.assertEqual(result.contexts, ())
        self.assertEqual(result.retrieved_sources, ())
        self.assertEqual(result.cited_sources, ())

    def test_missing_expected_source_is_visible_in_the_result(self) -> None:
        wrong = chunk("wrong", "Unrelated evidence.", "other.mdx", 1)
        case = GoldenCase(
            question="What is RAG?",
            ground_truth_answer="Retrieval-augmented generation.",
            ground_truth_sources=("rag.pdf",),
        )
        pipeline = FakePipeline(
            {
                case.question: RAGRun(
                    answer=GeneratedAnswer(
                        text="An unrelated answer [1].",
                        citations=[
                            Citation(
                                number=1,
                                record_id="wrong",
                                source="other.mdx",
                                page_number=None,
                                chunk_index=0,
                            )
                        ],
                    ),
                    retrieved_chunks=(wrong,),
                )
            }
        )

        result = evaluate_cases([case], pipeline=pipeline)[0]

        self.assertFalse(result.expected_source_retrieved)
        self.assertFalse(result.retrieval_expectation_met)
        self.assertFalse(result.declined)

    def test_preserves_dataset_order_and_runs_each_case_once(self) -> None:
        first = GoldenCase("First?", "First answer.", ("first.mdx",))
        second = GoldenCase("Second?", "Second answer.", ("second.mdx",))
        first_chunk = chunk("first", "First context.", "first.mdx", 1)
        second_chunk = chunk("second", "Second context.", "second.mdx", 1)
        pipeline = FakePipeline(
            {
                first.question: RAGRun(
                    GeneratedAnswer("First [1].", []),
                    (first_chunk,),
                ),
                second.question: RAGRun(
                    GeneratedAnswer("Second [1].", []),
                    (second_chunk,),
                ),
            }
        )

        results = evaluate_cases([first, second], pipeline=pipeline, top_k=3)

        self.assertEqual([result.case for result in results], [first, second])
        self.assertEqual(
            pipeline.calls,
            [(first.question, 3), (second.question, 3)],
        )

    def test_invalid_inputs_are_rejected_before_running_the_pipeline(self) -> None:
        pipeline = FakePipeline({})
        case = GoldenCase("Question?", "Answer.", ("source.mdx",))

        with self.assertRaisesRegex(ValueError, "cases"):
            evaluate_cases([], pipeline=pipeline)
        with self.assertRaisesRegex(ValueError, "top_k"):
            evaluate_cases([case], pipeline=pipeline, top_k=0)

        self.assertEqual(pipeline.calls, [])


if __name__ == "__main__":
    unittest.main()
