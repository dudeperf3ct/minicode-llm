"""Evaluate one trained checkpoint on the untouched 500-example test split.

The script sends each user prompt to an already-running vLLM OpenAI-compatible
server, extracts the final Python implementation, and runs the matching private
tests in a temporary directory. Per-example results are written as they finish
so an interrupted evaluation can resume without regenerating completed rows.
The final summary records correctness and the reasoning-output diagnostics used
to compare the four 10K experiment runs.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from datasets import load_dataset

from json_utils import read_json, read_jsonl, write_json, write_jsonl, write_jsonl_line
from pipeline_utils import PROJECT_DIR
from wandb_utils import WandbRun, log_evaluation

EVALUATION_MANIFEST = PROJECT_DIR / "manifests/evaluation.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
CODE_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
CODE_MARKERS = re.compile(r"(^|\n)\s*(?:async\s+def|def|class|from|import)\s+", re.MULTILINE)
MAIN_CONFIGS = (
    "configs/direct-lora.yml",
    "configs/reasoning-lora.yml",
    "configs/direct-fft.yml",
    "configs/reasoning-fft.yml",
)
RETRYABLE_STATUSES = {"api_error"}

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class TestExample:
    id: str
    messages: list[dict[str, str]]
    test: str
    difficulty: str
    subset: str
    style: str


@dataclass(frozen=True)
class TestDataset:
    repo_id: str
    revision: str
    split: str


@dataclass(frozen=True)
class EvaluationConfig:
    wandb: WandbRun
    mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=MAIN_CONFIGS, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--test-timeout", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = load_evaluation_config(Path(args.config))
    output_dir = PROJECT_DIR / "reports/test" / run_config.wandb.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = load_test_dataset()
    examples = load_examples(test_dataset)
    results_path = output_dir / "results.jsonl"
    completed = read_completed_results(results_path)
    pending = [example for example in examples if example.id not in completed]

    with httpx.Client(base_url=args.base_url, timeout=args.request_timeout) as client:
        model = get_served_model(client)
        if model != run_config.wandb.name:
            raise RuntimeError(
                f"vLLM serves {model!r}; expected --served-model-name {run_config.wandb.name!r}"
            )
        print(f"Evaluating {len(pending)} pending examples with model {model!r}")

        with results_path.open("a", encoding="utf-8") as output:
            for result in generate_results(
                pending,
                client,
                model,
                mode=run_config.mode,
                max_tokens=args.max_tokens,
                workers=args.workers,
                test_timeout=args.test_timeout,
            ):
                write_jsonl_line(output, result)
                output.flush()
                completed[result["id"]] = result
                print_progress(completed, len(examples))

    ordered_results = [completed[example.id] for example in examples]
    summary_path = output_dir / "summary.json"
    summary = build_summary(run_config, model, test_dataset, ordered_results)
    write_jsonl(results_path, ordered_results)
    write_json(summary_path, summary)
    log_evaluation(run_config.wandb, summary, results_path, summary_path)
    print(f"Evaluation complete: {summary_path}")


def load_evaluation_config(path: Path) -> EvaluationConfig:
    config = yaml.safe_load((PROJECT_DIR / path).read_text(encoding="utf-8"))
    mode = "reasoning" if config["datasets"][0]["split_thinking"] else "direct"
    return EvaluationConfig(wandb=WandbRun.from_axolotl_config(config), mode=mode)


def load_test_dataset() -> TestDataset:
    config = read_json(EVALUATION_MANIFEST)["test_dataset"]
    return TestDataset(**config)


def load_examples(source: TestDataset) -> list[TestExample]:
    dataset = load_dataset(source.repo_id, revision=source.revision, split=source.split)
    return [
        TestExample(
            id=row["id"],
            messages=row["messages"],
            test=row["test"],
            difficulty=row["gpt_difficulty"],
            subset=row["subset"],
            style=row["style"],
        )
        for row in dataset
    ]


def generate_results(
    examples: list[TestExample],
    client: httpx.Client,
    model: str,
    *,
    mode: str,
    max_tokens: int,
    workers: int,
    test_timeout: int,
) -> Iterator[JsonObject]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_example,
                example,
                client,
                model,
                mode,
                max_tokens,
                test_timeout,
            ): example
            for example in examples
        }
        for future in as_completed(futures):
            try:
                yield future.result()
            except Exception as error:
                yield {
                    **example_metadata(futures[future]),
                    "status": "api_error",
                    "error": str(error),
                }


def evaluate_example(
    example: TestExample,
    client: httpx.Client,
    model: str,
    mode: str,
    max_tokens: int,
    test_timeout: int,
) -> JsonObject:
    response = request_completion(
        client=client,
        model=model,
        messages=example.messages,
        enable_thinking=mode == "reasoning",
        max_tokens=max_tokens,
    )
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    code = extract_code(content)

    result = {
        **example_metadata(example),
        "finish_reason": choice.get("finish_reason"),
        "output_tokens": response.get("usage", {}).get("completion_tokens"),
        "has_think": bool(reasoning or "<think>" in content),
        "has_final_code": False,
        "content": content,
        "reasoning": reasoning,
    }
    if not code:
        result["status"] = "extraction_failed"
        result["code_present_but_not_extracted"] = bool(CODE_MARKERS.search(content))
        return result

    try:
        ast.parse(code)
    except SyntaxError as error:
        result.update(status="syntax_error", code=code, error=f"{error.msg} at line {error.lineno}")
        return result

    result["has_final_code"] = True
    result["code"] = code
    result.update(run_private_tests(code, example.test, test_timeout))
    return result


def example_metadata(example: TestExample) -> JsonObject:
    return {
        "id": example.id,
        "difficulty": example.difficulty,
        "subset": example.subset,
        "style": example.style,
    }


def request_completion(
    *,
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
    enable_thinking: bool,
    max_tokens: int,
) -> JsonObject:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    response = client.post("/chat/completions", json=payload)
    response.raise_for_status()
    return response.json()


def get_served_model(client: httpx.Client) -> str:
    response = client.get("/models")
    response.raise_for_status()
    models = response.json()["data"]
    if len(models) != 1:
        raise RuntimeError(f"Expected one served model, found {len(models)}")
    return models[0]["id"]


def extract_code(content: str) -> str:
    content = content.replace("<|im_end|>", "").strip()
    fenced = CODE_FENCE.findall(content)
    if fenced:
        return "\n\n".join(block.strip() for block in fenced)
    return content if CODE_MARKERS.search(content) else ""


def run_private_tests(code: str, tests: str, timeout: int) -> JsonObject:
    with tempfile.TemporaryDirectory(prefix="kodcode-test-") as directory:
        workdir = Path(directory)
        (workdir / "solution.py").write_text(code, encoding="utf-8")
        (workdir / "test_solution.py").write_text(tests, encoding="utf-8")
        env = os.environ.copy()
        env.update(HOME=directory, PYTHONDONTWRITEBYTECODE="1")
        env.pop("HF_TOKEN", None)
        env.pop("WANDB_API_KEY", None)

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


def build_summary(
    run_config: EvaluationConfig, model: str, test_dataset: TestDataset, results: list[JsonObject]
) -> JsonObject:
    output_tokens = [
        result["output_tokens"]
        for result in results
        if isinstance(result.get("output_tokens"), int)
    ]
    return {
        "wandb_run_id": run_config.wandb.run_id,
        "wandb_name": run_config.wandb.name,
        "model": model,
        "mode": run_config.mode,
        "test_dataset": {
            "repo_id": test_dataset.repo_id,
            "revision": test_dataset.revision,
            "split": test_dataset.split,
        },
        **pass_summary(results),
        "average_output_tokens": (
            sum(output_tokens) / len(output_tokens) if output_tokens else None
        ),
        "contains_think": count_and_rate(results, "has_think"),
        "contains_final_code": count_and_rate(results, "has_final_code"),
        "truncations": sum(result.get("finish_reason") == "length" for result in results),
        "extraction_failures": sum(
            result.get("status") == "extraction_failed" for result in results
        ),
        "code_present_but_not_extracted": sum(
            result.get("code_present_but_not_extracted") is True for result in results
        ),
        "syntax_errors": sum(result.get("status") == "syntax_error" for result in results),
        "test_timeouts": sum(result.get("status") == "test_timeout" for result in results),
        "api_errors": sum(result.get("status") == "api_error" for result in results),
        "pass_rate_by_difficulty": grouped_pass_rates(results, "difficulty"),
        "pass_rate_by_subset": grouped_pass_rates(results, "subset"),
        "pass_rate_by_style": grouped_pass_rates(results, "style"),
    }


def count_and_rate(results: list[JsonObject], field: str) -> JsonObject:
    count = sum(result.get(field) is True for result in results)
    return {"count": count, "rate": count / len(results)}


def grouped_pass_rates(results: list[JsonObject], field: str) -> JsonObject:
    groups = defaultdict(list)
    for result in results:
        if field in result:
            groups[result[field]].append(result)
    return {group: pass_summary(rows) for group, rows in sorted(groups.items())}


def pass_summary(results: list[JsonObject]) -> JsonObject:
    passed = sum(result.get("passed") is True for result in results)
    return {"examples": len(results), "passed": passed, "pass_rate": passed / len(results)}


def print_progress(results: dict[str, JsonObject], total: int) -> None:
    passed = sum(result.get("passed") is True for result in results.values())
    print(f"{len(results)}/{total} complete, {passed} passed")


def read_completed_results(path: Path) -> dict[str, JsonObject]:
    if not path.exists():
        return {}
    return {
        row["id"]: row for row in read_jsonl(path) if row.get("status") not in RETRYABLE_STATUSES
    }


if __name__ == "__main__":
    main()
