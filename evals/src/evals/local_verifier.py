"""Legacy local verifier for offline evaluation on disposable machines."""

import ast
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from evals.code import ONLINE_JUDGE_STYLE

JsonObject = dict[str, Any]


def run_private_tests(code: str, tests: str, style: str, timeout: int) -> JsonObject:
    if style == ONLINE_JUDGE_STYLE:
        return run_online_judge_tests(code, tests, timeout)
    return run_pytest_tests(code, tests, timeout)


def run_online_judge_tests(code: str, tests: str, timeout: int) -> JsonObject:
    cases = ast.literal_eval(tests)
    if (
        not isinstance(cases, dict)
        or not isinstance(cases.get("stdin"), list)
        or not isinstance(cases.get("stdout"), list)
        or len(cases["stdin"]) != len(cases["stdout"])
    ):
        raise ValueError("Online-judge tests must contain matched stdin/stdout lists")

    with tempfile.TemporaryDirectory(prefix="kodcode-test-") as directory:
        workdir = Path(directory)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        env = test_environment(directory)
        deadline = time.monotonic() + timeout

        for index, (stdin, expected) in enumerate(
            zip(cases["stdin"], cases["stdout"], strict=True), start=1
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"status": "test_timeout", "passed": False}
            try:
                process = subprocess.run(
                    [sys.executable, "solution.py"],
                    cwd=workdir,
                    env=env,
                    input=f"{stdin}\n",
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {"status": "test_timeout", "passed": False}

            actual = process.stdout.strip()
            if process.returncode != 0 or actual != expected.strip():
                details = f"case {index}: expected {expected!r}, got {actual!r}\n{process.stderr}"
                return {
                    "status": "test_failed",
                    "passed": False,
                    "test_output": details[-4_000:],
                }

    return {"status": "passed", "passed": True, "test_output": ""}


def run_pytest_tests(code: str, tests: str, timeout: int) -> JsonObject:
    with tempfile.TemporaryDirectory(prefix="kodcode-test-") as directory:
        workdir = Path(directory)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "test_solution.py").write_text(tests, encoding="utf-8")
        env = test_environment(directory)

        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--disable-warnings",
                    "--tb=short",
                    "test_solution.py",
                ],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "test_timeout", "passed": False}

    return {
        "status": "passed" if process.returncode == 0 else "test_failed",
        "passed": process.returncode == 0,
        "test_output": (process.stdout + process.stderr)[-4_000:],
    }


def test_environment(home: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(HOME=home, PYTHONDONTWRITEBYTECODE="1")
    env.pop("HF_TOKEN", None)
    env.pop("WANDB_API_KEY", None)
    return env
