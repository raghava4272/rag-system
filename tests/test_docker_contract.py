"""Static safety and runtime contracts for the Docker image."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
REQUIREMENTS_IN = PROJECT_ROOT / "requirements.in"
LOCKFILES = {
    "amd64": PROJECT_ROOT / "requirements-linux-amd64.lock",
    "arm64": PROJECT_ROOT / "requirements-linux-arm64.lock",
}


class DockerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        cls.dockerignore = {
            line.strip()
            for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    def test_build_context_excludes_secrets_and_local_artifacts(self) -> None:
        required_patterns = {
            ".env",
            ".git",
            ".venv",
            "__pycache__",
            "*.py[cod]",
            ".DS_Store",
            "data/chroma",
            "data/processed",
            "eval/report.md",
        }

        self.assertTrue(required_patterns.issubset(self.dockerignore))
        self.assertIn("!.env.example", self.dockerignore)

    def test_uses_python_312_multistage_build(self) -> None:
        self.assertRegex(
            self.dockerfile,
            r"(?m)^FROM python:3\.12-slim-bookworm AS dependencies$",
        )
        self.assertIn("FROM dependencies AS test", self.dockerfile)
        self.assertIn("FROM dependencies AS runtime", self.dockerfile)
        self.assertIn("python -m unittest discover -s tests -v", self.dockerfile)
        self.assertIn("python -m pip check", self.dockerfile)

    def test_test_stage_contains_every_contract_input(self) -> None:
        test_stage = self.dockerfile.split(
            "FROM dependencies AS test",
            maxsplit=1,
        )[1].split("FROM dependencies AS runtime", maxsplit=1)[0]

        self.assertIn(
            "COPY Dockerfile .dockerignore compose.yaml requirements.in "
            "requirements-linux-*.lock ./",
            test_stage,
        )
        self.assertIn("COPY .github ./.github", test_stage)

    def test_selects_an_exact_cpu_lock_for_each_target_architecture(self) -> None:
        for architecture, lockfile in LOCKFILES.items():
            with self.subTest(architecture=architecture):
                lock_text = lockfile.read_text(encoding="utf-8")
                lock_lines = [
                    line.strip()
                    for line in lock_text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]

                self.assertTrue(lock_lines)
                self.assertTrue(
                    all("==" in line or line.startswith("--") for line in lock_lines),
                    "Every Linux runtime dependency must be pinned exactly",
                )
                normalized_lock = lock_text.lower()
                self.assertRegex(normalized_lock, r"(?m)^torch==[^\n]+\+cpu$")
                self.assertNotRegex(
                    normalized_lock,
                    r"(?m)^(?:cuda-|nvidia-|triton==)",
                )

        self.assertIn("ARG TARGETARCH", self.dockerfile)
        self.assertIn(
            "COPY requirements-linux-${TARGETARCH}.lock ./requirements-linux.lock",
            self.dockerfile,
        )
        self.assertIn("-r requirements-linux.lock", self.dockerfile)
        self.assertNotIn("-r requirements.txt", self.dockerfile)
        self.assertIn("uv==0.12.1", self.dockerfile)
        self.assertIn("--torch-backend cpu", self.dockerfile)

    def test_runtime_is_non_root_and_keeps_the_safe_default_command(self) -> None:
        runtime = self.dockerfile.split("FROM dependencies AS runtime", maxsplit=1)[1]

        self.assertIn("USER rag", runtime)
        self.assertIn(
            'ENTRYPOINT ["python", "scripts/container.py"]',
            runtime,
        )
        self.assertIn('CMD ["--help"]', runtime)

    def test_api_runtime_dependencies_are_declared_and_locked(self) -> None:
        direct_requirements = REQUIREMENTS_IN.read_text(encoding="utf-8")
        self.assertRegex(direct_requirements, r"(?m)^fastapi\b")
        self.assertRegex(direct_requirements, r"(?m)^uvicorn\b")

        for architecture, lockfile in LOCKFILES.items():
            with self.subTest(architecture=architecture):
                lock_text = lockfile.read_text(encoding="utf-8")
                self.assertRegex(lock_text, r"(?m)^fastapi==[^\n]+$")
                self.assertRegex(lock_text, r"(?m)^uvicorn==[^\n]+$")

    def test_groq_secret_is_never_baked_into_the_image(self) -> None:
        self.assertNotRegex(
            self.dockerfile,
            re.compile(r"(?mi)^\s*(?:ARG|ENV)\s+GROQ_API_KEY\b"),
        )
        self.assertNotIn("COPY .env", self.dockerfile)
        self.assertNotRegex(
            self.dockerfile,
            re.compile(r"(?mi)^\s*(?:ARG|ENV)\s+RAG_API_KEY_HASHES\b"),
        )
        self.assertNotRegex(
            self.dockerfile,
            re.compile(r"(?mi)^\s*(?:ARG|ENV)\s+RAG_DOCUMENT_PERMISSIONS\b"),
        )
        self.assertNotRegex(
            self.dockerfile,
            re.compile(r"(?mi)^\s*(?:ARG|ENV)\s+RAG_RATE_LIMIT_"),
        )
        self.assertNotRegex(
            self.dockerfile,
            re.compile(r"(?mi)^\s*(?:ARG|ENV)\s+RAG_LOG_LEVEL\b"),
        )


if __name__ == "__main__":
    unittest.main()
