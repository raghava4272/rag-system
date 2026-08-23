"""Security and publishing contracts for the container image workflow."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "container-image.yml"
DEPENDABOT_PATH = PROJECT_ROOT / ".github" / "dependabot.yml"


class ContainerImageWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        cls.verify_job = cls.workflow.split("  verify:", maxsplit=1)[1].split(
            "  publish:", maxsplit=1
        )[0]
        cls.publish_job = cls.workflow.split("  publish:", maxsplit=1)[1]

    def test_prs_verify_but_only_trusted_pushes_publish(self) -> None:
        self.assertIn("pull_request:", self.workflow)
        self.assertIn("push:", self.workflow)
        self.assertIn("branches: [main]", self.workflow)
        self.assertIn("tags: [\"v*\"]", self.workflow)
        self.assertNotIn("pull_request_target", self.workflow)
        self.assertIn("permissions:\n  contents: read", self.workflow)
        self.assertNotIn("packages: write", self.verify_job)
        self.assertIn("needs: verify", self.publish_job)
        self.assertIn("if: ${{ github.event_name == 'push' }}", self.publish_job)
        self.assertIn("packages: write", self.publish_job)

    def test_verification_builds_tests_and_a_scannable_runtime_image(self) -> None:
        self.assertIn("target: test", self.verify_job)
        self.assertIn("target: runtime", self.verify_job)
        self.assertIn("load: true", self.verify_job)
        self.assertIn("tags: rag-system:scan-${{ matrix.arch }}", self.verify_job)
        self.assertLess(
            self.verify_job.index("target: test"),
            self.verify_job.index("target: runtime"),
        )

    def test_each_published_architecture_is_tested_and_scanned_natively(self) -> None:
        self.assertIn("runner: ubuntu-latest", self.verify_job)
        self.assertIn("platform: linux/amd64", self.verify_job)
        self.assertIn("arch: amd64", self.verify_job)
        self.assertIn("runner: ubuntu-24.04-arm", self.verify_job)
        self.assertIn("platform: linux/arm64", self.verify_job)
        self.assertIn("arch: arm64", self.verify_job)
        self.assertIn("runs-on: ${{ matrix.runner }}", self.verify_job)
        self.assertIn("platforms: ${{ matrix.platform }}", self.verify_job)
        self.assertIn(
            "name: container-vulnerability-report-${{ matrix.arch }}",
            self.verify_job,
        )

    def test_vulnerability_report_is_uploaded_and_fixable_findings_gate_builds(self) -> None:
        self.assertRegex(
            self.verify_job,
            r"uses: aquasecurity/trivy-action@[0-9a-f]{40}",
        )
        self.assertIn("output: trivy-results.json", self.verify_job)
        self.assertIn("severity: HIGH,CRITICAL", self.verify_job)
        self.assertIn("exit-code: 0", self.verify_job)
        gate = self.verify_job.split("- name: Enforce vulnerability policy", maxsplit=1)[
            1
        ].split("- name: Upload vulnerability report", maxsplit=1)[0]
        self.assertIn("exit-code: 1", gate)
        self.assertIn("ignore-unfixed: true", gate)
        self.assertIn("if: ${{ always() }}", self.verify_job)
        self.assertIn("name: container-vulnerability-report", self.verify_job)

    def test_publishes_native_amd64_and_arm64_images_with_stable_tags(self) -> None:
        self.assertIn("REGISTRY: ghcr.io", self.workflow)
        self.assertIn("IMAGE_NAME: ${{ github.repository }}", self.workflow)
        self.assertIn("type=sha,format=long", self.publish_job)
        self.assertIn("type=ref,event=branch", self.publish_job)
        self.assertIn("type=semver,pattern={{version}}", self.publish_job)
        self.assertIn("platforms: linux/amd64,linux/arm64", self.publish_job)
        self.assertIn("push: true", self.publish_job)
        self.assertIn("provenance: mode=max", self.publish_job)
        self.assertIn("sbom: true", self.publish_job)

    def test_published_digest_receives_a_github_attestation(self) -> None:
        self.assertIn("attestations: write", self.publish_job)
        self.assertIn("id-token: write", self.publish_job)
        self.assertIn("subject-digest: ${{ steps.push.outputs.digest }}", self.publish_job)
        self.assertIn("push-to-registry: true", self.publish_job)

    def test_every_external_action_is_pinned_to_an_immutable_commit(self) -> None:
        uses = re.findall(r"^\s*uses:\s*([^\s#]+)", self.workflow, flags=re.MULTILINE)
        self.assertGreaterEqual(len(uses), 10)
        for action in uses:
            with self.subTest(action=action):
                self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$")

    def test_dependabot_keeps_action_pins_and_base_image_current(self) -> None:
        dependabot = DEPENDABOT_PATH.read_text(encoding="utf-8")

        self.assertIn('package-ecosystem: "github-actions"', dependabot)
        self.assertIn('package-ecosystem: "docker"', dependabot)
        self.assertEqual(dependabot.count('interval: "weekly"'), 2)


if __name__ == "__main__":
    unittest.main()
