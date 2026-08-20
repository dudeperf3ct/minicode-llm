"""Greedy LM/FIM evaluation for HF-compatible checkpoints."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
END_OF_TEXT = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LM/FIM generation on JSONL eval samples")
    parser.add_argument("--model", required=True, help="HF model repo id or local model path")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer repo id or local path")
    parser.add_argument(
        "--samples", default="eval/eval_samples.jsonl", help="Path to eval sample JSONL"
    )
    parser.add_argument(
        "--output-jsonl", default=None, help="Optional path to write per-sample outputs"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="<=0 means greedy decoding")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stop-at-eot", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show-prompts", action="store_true")
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def build_fim_prompt(prefix: str, suffix: str, fim_format: str) -> str:
    if fim_format == "psm":
        return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    if fim_format == "spm":
        return f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}"
    raise ValueError(f"Unsupported fim_format: {fim_format}")


def resolve_dtype(device: str, dtype_arg: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_stop_token_id(tokenizer, stop_at_eot: bool) -> int | None:
    if not stop_at_eot:
        return None
    for attr in ("eos_token_id",):
        token_id = getattr(tokenizer, attr, None)
        if isinstance(token_id, int):
            return token_id
    token_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    if isinstance(token_id, int) and token_id >= 0:
        return token_id
    return None


def build_prompt(sample: dict[str, Any]) -> tuple[str, str]:
    if sample.get("prompt") is not None:
        return sample["prompt"], "lm"
    prefix = sample.get("fim_prefix", "")
    suffix = sample.get("fim_suffix", "")
    fim_format = sample.get("fim_format", "psm")
    return build_fim_prompt(prefix, suffix, fim_format), "fim"


def evaluate_expected(
    sample: dict[str, Any], generated_text: str
) -> tuple[bool | None, str | None]:
    expected = sample.get("expected_middle")
    if expected is not None:
        return generated_text.startswith(expected), "expected_middle_prefix"

    expected_lm = sample.get("expected_lm")
    if expected_lm is not None:
        return generated_text.startswith(expected_lm), "expected_lm_prefix"

    expected_any = sample.get("expected_any_of")
    if isinstance(expected_any, list) and expected_any:
        return any(
            generated_text.startswith(item) for item in expected_any
        ), "expected_any_of_prefix"

    return None, None


def main() -> None:
    args = parse_args()
    sample_path = Path(args.samples)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    samples = load_samples(sample_path)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No samples loaded")

    device = torch.device(args.device)
    dtype = resolve_dtype(args.device, args.dtype)
    tokenizer_name = args.tokenizer or args.model

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=args.trust_remote_code
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
    model.to(device)
    model.eval()

    stop_token_id = resolve_stop_token_id(tokenizer, args.stop_at_eot)
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    if stop_token_id is not None:
        generation_kwargs["eos_token_id"] = stop_token_id
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        if args.top_k is not None:
            generation_kwargs["top_k"] = args.top_k

    results: list[dict[str, Any]] = []
    for sample in samples:
        name = sample.get("name", "sample")
        prompt, mode = build_prompt(sample)

        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        prompt_len = input_ids.shape[1]
        gen_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        expected_match, expected_type = evaluate_expected(sample, generated_text)

        result = {
            "name": name,
            "mode": mode,
            "generated": generated_text,
            "prompt_tokens": int(prompt_len),
            "generated_tokens": int(gen_ids.shape[0]),
            "expected_type": expected_type,
            "expected_prefix_match": expected_match,
        }
        results.append(result)

        status = "n/a" if expected_match is None else ("pass" if expected_match else "fail")
        print(f"[{name}] mode={mode} expected={status}")
        if args.show_prompts:
            print("prompt:")
            print(prompt)
        print("generated:")
        print(generated_text)
        print("-" * 80)

    with_expected = [r for r in results if r["expected_prefix_match"] is not None]
    passed = sum(1 for r in with_expected if r["expected_prefix_match"])
    summary = {
        "total": len(results),
        "with_expected": len(with_expected),
        "expected_prefix_match_pass": passed,
        "expected_prefix_match_rate": (passed / len(with_expected)) if with_expected else None,
    }
    print(json.dumps(summary, indent=2))

    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
