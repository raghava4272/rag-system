"""Contract tests for the Phase 3 pull-request evaluation workflow."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "rag-eval.yml"


class Phase3CIWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_uses_safe_pull_request_trigger_and_read_only_permissions(self) -> None:
        self.assertIn("pull_request:", self.workflow)
        self.assertNotIn("pull_request_target", self.workflow)
        self.assertIn("permissions:\n  contents: read", self.workflow)

    def test_runs_tests_then_builds_index_then_evaluates(self) -> None:
        test_position = self.workflow.index("python -m unittest discover -s tests -v")
        index_position = self.workflow.index("python scripts/build_index.py data/raw")
        eval_position = self.workflow.index("python eval/run_eval.py")

        self.assertLess(test_position, index_position)
        self.assertLess(index_position, eval_position)
        self.assertIn("--dataset eval/golden_dataset.jsonl", self.workflow)
        self.assertIn("--threshold 0.8", self.workflow)
        self.assertIn("--report artifacts/rag-evaluation.md", self.workflow)
        self.assertIn("GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}", self.workflow)

    def test_always_uploads_the_markdown_report(self) -> None:
        self.assertRegex(
            self.workflow,
            r"uses: actions/upload-artifact@[0-9a-f]{40}",
        )
        self.assertIn("if: ${{ always() }}", self.workflow)
        self.assertIn("name: rag-evaluation-report", self.workflow)
        self.assertIn("path: artifacts/rag-evaluation.md", self.workflow)
        self.assertIn("if-no-files-found: warn", self.workflow)

    def test_evaluation_failure_is_not_suppressed(self) -> None:
        evaluation_step = self.workflow.split(
            "- name: Run the golden RAG evaluation",
            maxsplit=1,
        )[1].split("- name: Upload the evaluation report", maxsplit=1)[0]

        self.assertNotIn("continue-on-error", evaluation_step)
        self.assertNotIn("|| true", evaluation_step)
        self.assertNotIn("exit 0", evaluation_step)

    def test_every_external_action_is_pinned_to_an_immutable_commit(self) -> None:
        uses = re.findall(
            r"^\s*uses:\s*([^\s#]+)",
            self.workflow,
            flags=re.MULTILINE,
        )
        self.assertGreaterEqual(len(uses), 3)
        for action in uses:
            with self.subTest(action=action):
                self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
