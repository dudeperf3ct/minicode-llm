from evals.code import extract_code


def test_extracts_fenced_python() -> None:
    assert extract_code("```python\ndef answer():\n    return 42\n```", "instruct") == (
        "def answer():\n    return 42"
    )


def test_rejects_unmarked_prose() -> None:
    assert extract_code("The answer is forty-two.", "instruct") == ""


def test_online_judge_accepts_script_body() -> None:
    assert extract_code("print(input())<|im_end|>", "online_judge") == "print(input())"
