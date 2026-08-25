from evals.local_verifier import run_private_tests


def test_pytest_verifier_passes_correct_code() -> None:
    result = run_private_tests(
        "def add(left, right):\n    return left + right\n",
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        "instruct",
        10,
    )

    assert result["status"] == "passed"
    assert result["passed"] is True


def test_online_judge_verifier_detects_failure() -> None:
    result = run_private_tests(
        "print(0)",
        repr({"stdin": ["2"], "stdout": ["2"]}),
        "online_judge",
        10,
    )

    assert result["status"] == "test_failed"
    assert result["passed"] is False
