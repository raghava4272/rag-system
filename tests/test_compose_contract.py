"""Static contracts for the persistent Docker Compose workflow."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = PROJECT_ROOT / "compose.yaml"


class ComposeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compose = COMPOSE_FILE.read_text(encoding="utf-8")

    def test_defines_one_shot_jobs_with_stable_entrypoints(self) -> None:
        for service in ("index", "ask", "evaluate", "report"):
            self.assertRegex(self.compose, rf"(?m)^  {service}:$")

        self.assertIn(
            'entrypoint: ["python", "scripts/container.py", "index"]',
            self.compose,
        )
        self.assertIn(
            'entrypoint: ["python", "scripts/container.py", "ask"]',
            self.compose,
        )
        self.assertIn(
            'entrypoint: ["python", "scripts/container.py", "evaluate", '
            '"--report", "/app/artifacts/eval-report.md"]',
            self.compose,
        )
        self.assertIn(
            'entrypoint: ["cat", "/app/artifacts/eval-report.md"]',
            self.compose,
        )

    def test_defines_a_health_checked_long_running_api(self) -> None:
        self.assertRegex(self.compose, r"(?m)^  api:$")
        self.assertIn(
            'entrypoint: ["python", "-m", "uvicorn", "src.api:app"]',
            self.compose,
        )
        self.assertIn('command: ["--host", "0.0.0.0", "--port", "8000"]', self.compose)
        self.assertIn('- "8000:8000"', self.compose)
        self.assertIn("healthcheck:", self.compose)
        self.assertIn("restart: unless-stopped", self.compose)

    def test_builds_the_shared_runtime_image_once(self) -> None:
        self.assertEqual(self.compose.count("  build:\n"), 1)
        index_service = self.compose.split("  index:\n", maxsplit=1)[1].split(
            "\n  ask:\n",
            maxsplit=1,
        )[0]
        self.assertIn("    build:\n", index_service)
        self.assertIn("      target: runtime", index_service)

    def test_uses_the_hosts_native_architecture_by_default(self) -> None:
        self.assertNotRegex(self.compose, r"(?m)^\s*platform:")

    def test_persists_index_models_and_evaluation_report(self) -> None:
        required_mounts = {
            "./data/raw:/app/data/raw:ro",
            "rag-chroma:/app/data/chroma",
            "rag-cache:/home/rag/.cache",
            "rag-artifacts:/app/artifacts",
        }
        for mount in required_mounts:
            self.assertIn(f"- {mount}", self.compose)

        for volume in ("rag-chroma", "rag-cache", "rag-artifacts"):
            self.assertRegex(self.compose, rf"(?m)^  {volume}:$")

    def test_injects_groq_configuration_only_at_runtime(self) -> None:
        self.assertIn("GROQ_API_KEY: ${GROQ_API_KEY:-}", self.compose)
        self.assertIn(
            "GROQ_MODEL: ${GROQ_MODEL:-openai/gpt-oss-20b}",
            self.compose,
        )
        self.assertNotRegex(self.compose, re.compile(r"gsk_[A-Za-z0-9_-]+"))

    def test_injects_hashed_api_identity_configuration_only_into_api(self) -> None:
        setting = "RAG_API_KEY_HASHES: ${RAG_API_KEY_HASHES:-}"
        self.assertIn(setting, self.compose)
        self.assertEqual(self.compose.count(setting), 1)

    def test_injects_document_permissions_only_into_api(self) -> None:
        setting = "RAG_DOCUMENT_PERMISSIONS: ${RAG_DOCUMENT_PERMISSIONS:-}"
        self.assertIn(setting, self.compose)
        self.assertEqual(self.compose.count(setting), 1)

    def test_injects_bounded_rate_limit_configuration_only_into_api(self) -> None:
        requests = "RAG_RATE_LIMIT_REQUESTS: ${RAG_RATE_LIMIT_REQUESTS:-30}"
        window = (
            "RAG_RATE_LIMIT_WINDOW_SECONDS: ${RAG_RATE_LIMIT_WINDOW_SECONDS:-60}"
        )
        self.assertIn(requests, self.compose)
        self.assertIn(window, self.compose)
        self.assertEqual(self.compose.count(requests), 1)
        self.assertEqual(self.compose.count(window), 1)

    def test_injects_log_level_only_into_api(self) -> None:
        setting = "RAG_LOG_LEVEL: ${RAG_LOG_LEVEL:-INFO}"
        self.assertIn(setting, self.compose)
        self.assertEqual(self.compose.count(setting), 1)

    def test_jobs_use_a_hardened_read_only_runtime(self) -> None:
        self.assertIn("read_only: true", self.compose)
        self.assertIn("init: true", self.compose)
        self.assertIn("- ALL", self.compose)
        self.assertIn("- no-new-privileges:true", self.compose)
        self.assertIn(
            "- /tmp:rw,noexec,nosuid,size=1g,uid=10001,gid=10001",
            self.compose,
        )


if __name__ == "__main__":
    unittest.main()
