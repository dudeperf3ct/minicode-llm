"""Run generated Python against public tests in isolated Modal Sandboxes."""

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import modal

APP_NAME = "qwen35-kodcode-rlvr-verifier"
TEST_TIMEOUT_SECONDS = 10
SANDBOX_TIMEOUT_SECONDS = 30
MAX_WORKERS = 8
MAX_ATTEMPTS = 3
OUTPUT_LIMIT = 4_000


@dataclass(frozen=True)
class VerificationRequest:
    question_id: str
    code: str
    tests: str


@dataclass(frozen=True)
class VerificationResult:
    question_id: str
    status: str
    passed: bool
    duration_seconds: float
    output: str = ""


class Verifier(Protocol):
    def verify_batch(self, requests: list[VerificationRequest]) -> list[VerificationResult]: ...


class ModalVerifier:
    """Create one network-isolated Modal Sandbox per completion."""

    def __init__(self) -> None:
        self.app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self.image = modal.Image.debian_slim(python_version="3.12").pip_install("pytest==8.4.2")

    def verify_batch(self, requests: list[VerificationRequest]) -> list[VerificationResult]:
        if not requests:
            return []
        workers = min(MAX_WORKERS, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._verify_with_retries, requests))

    def _verify_with_retries(self, request: VerificationRequest) -> VerificationResult:
        for attempt in range(MAX_ATTEMPTS):
            try:
                return self._verify_once(request)
            except (modal.Error, TimeoutError, ConnectionError) as error:
                if attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"Modal verification failed for {request.question_id} after "
                        f"{MAX_ATTEMPTS} attempts"
                    ) from error
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _verify_once(self, request: VerificationRequest) -> VerificationResult:
        started = time.monotonic()
        sandbox = None
        try:
            sandbox = modal.Sandbox.create(
                app=self.app,
                image=self.image,
                cpu=1.0,
                memory=1_024,
                timeout=SANDBOX_TIMEOUT_SECONDS,
                block_network=True,
            )
            sandbox.filesystem.write_text(request.code, "/tmp/work/solution.py")
            sandbox.filesystem.write_text(request.tests, "/tmp/work/test_solution.py")
            try:
                process = sandbox.exec(
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "--disable-warnings",
                    "--tb=short",
                    "--show-capture=no",
                    "--maxfail=1",
                    "test_solution.py",
                    workdir="/tmp/work",
                    timeout=TEST_TIMEOUT_SECONDS,
                    env={"HOME": "/tmp/work", "PYTHONDONTWRITEBYTECODE": "1"},
                )
                stdout = process.stdout.read()
                stderr = process.stderr.read()
                return_code = process.wait()
            except modal.exception.ExecTimeoutError:
                return VerificationResult(
                    question_id=request.question_id,
                    status="test_timeout",
                    passed=False,
                    duration_seconds=time.monotonic() - started,
                )
            if return_code == -1:
                return VerificationResult(
                    question_id=request.question_id,
                    status="test_timeout",
                    passed=False,
                    duration_seconds=time.monotonic() - started,
                )

            passed = return_code == 0
            return VerificationResult(
                question_id=request.question_id,
                status="passed" if passed else "test_failed",
                passed=passed,
                duration_seconds=time.monotonic() - started,
                output=(stdout + stderr)[-OUTPUT_LIMIT:],
            )
        finally:
            if sandbox is not None:
                with suppress(modal.Error):
                    sandbox.terminate()


def main() -> None:
    requests = [
        VerificationRequest(
            "passes",
            "def add(left, right):\n    return left + right\n",
            "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        ),
        VerificationRequest(
            "fails",
            "def add(left, right):\n    return left - right\n",
            "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        ),
        VerificationRequest(
            "times-out",
            "while True:\n    pass\n",
            "def test_import():\n    import solution  # noqa: F401\n",
        ),
        VerificationRequest(
            "network-blocked",
            (
                "import socket\n\n"
                "def network_is_blocked():\n"
                "    try:\n"
                "        socket.create_connection(('1.1.1.1', 80), timeout=1)\n"
                "    except OSError:\n"
                "        return True\n"
                "    return False\n"
            ),
            (
                "from solution import network_is_blocked\n\n"
                "def test_network():\n"
                "    assert network_is_blocked()\n"
            ),
        ),
    ]
    results = ModalVerifier().verify_batch(requests)
    expected = ["passed", "test_failed", "test_timeout", "passed"]
    statuses = [result.status for result in results]
    for result in results:
        print(f"{result.question_id}: {result.status} ({result.duration_seconds:.2f}s)")
    if statuses != expected:
        raise RuntimeError(f"Unexpected smoke results: {statuses}")


if __name__ == "__main__":
    main()
