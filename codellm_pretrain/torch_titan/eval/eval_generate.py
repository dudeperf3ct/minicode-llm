"""Evaluation script for TorchTitan DCP checkpoints."""

# Modified from: https://github.com/pytorch/torchtitan/blob/main/scripts/generate/test_generate.py
# and https://github.com/pytorch/torchtitan/blob/main/scripts/generate/_generation.py

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torchtitan.config import ConfigManager
from torchtitan.protocols.train_spec import get_train_spec
from torchtitan.tools.logging import init_logger, logger


def multinomial_sample_one(probs: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
    # probs: (B, vocab_size)
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    # logits: (B, vocab_size)
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    return torch.nn.functional.softmax(logits, dim=-1)


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    if hasattr(model, "max_seq_len"):
        # Align TorchTitan's position_ids with the current prompt length to avoid RoPE shape mismatches.
        model.max_seq_len = x.shape[1]
    logits = model(x)  # (B, T, vocab_size)
    if temperature <= 0:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(dtype=torch.long)

    # Sample from the last-step distribution only: (B, vocab_size).
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    return multinomial_sample_one(probs, rng=rng)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    stop_token_id: int | None = None,
) -> torch.Tensor:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    # generated_tokens: (B, T)
    generated_tokens = input_ids.clone()

    for _ in range(max_new_tokens):
        next_token = generate_next_token(
            model,
            x=generated_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

        if stop_token_id is not None and (next_token == stop_token_id).all():
            break

    return generated_tokens


def build_fim_prompt(
    prefix: str,
    suffix: str,
    fim_format: str,
    fim_prefix_token: str,
    fim_suffix_token: str,
    fim_middle_token: str,
) -> str:
    if fim_format == "psm":
        return f"{fim_prefix_token}{prefix}{fim_suffix_token}{suffix}{fim_middle_token}"
    if fim_format == "spm":
        return f"{fim_suffix_token}{suffix}{fim_prefix_token}{prefix}{fim_middle_token}"

    raise ValueError(f"Unsupported FIM format: {fim_format}")


def load_samples(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_stop_token_id(tokenizer, stop_at_eos: bool) -> int | None:
    if not stop_at_eos:
        return None

    for attr in ("eos_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int):
            return value

    try:
        ids = tokenizer.encode("<|endoftext|>", add_bos=False, add_eos=False)
    except TypeError:
        ids = tokenizer.encode("<|endoftext|>")

    if isinstance(ids, list) and ids:
        return ids[0]

    return None


def clean_byte_level_text(text: str) -> str:
    """Normalize byte-level BPE markers to spaces/newlines for readability."""
    return text.replace("\u0120", " ").replace("\u010a", "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from a TorchTitan DCP checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="TOML config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--prompt", type=str, default="", help="Input prompt")
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Optional JSONL with prompts/FIM fields",
    )

    parser.add_argument(
        "--fim_prefix",
        type=str,
        default=None,
        help="FIM prefix text (enables FIM if set)",
    )
    parser.add_argument(
        "--fim_suffix",
        type=str,
        default="<|fim_suffix|>",
        help="FIM suffix text (enables FIM if set)",
    )
    parser.add_argument(
        "--fim_format",
        choices=["psm", "spm"],
        default="psm",
        help="FIM ordering (default: psm)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (<=0 for greedy)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples to run")
    parser.add_argument("--top_k", type=int, help="Top-k sampling")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--stop_at_eos",
        action="store_true",
        help="Stop early if EOS token is generated",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--add_bos",
        action="store_true",
        help="Add BOS token to prompt",
    )
    parser.add_argument(
        "--add_eos",
        action="store_true",
        help="Add EOS token to prompt",
    )
    parser.add_argument(
        "--custom_import",
        type=str,
        default=None,
        help="Override experimental.custom_import (e.g. custom_spec)",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="Override hf_transformers.model with a local path or repo id",
    )
    parser.add_argument(
        "--out",
        action="store_true",
        default=False,
        help="Print JSON report to stdout",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    # Ensure custom modules like `custom_spec.py` at repo root are discoverable.
    for path in (project_root, script_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    init_logger()

    if args.custom_import:
        importlib.import_module(args.custom_import)

    if args.samples is None and not args.prompt and args.fim_prefix is None:
        raise SystemExit("Provide --prompt or --samples or --fim_prefix/--fim_suffix.")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_str == "cuda" and torch.cuda.is_available():
        device_str = f"cuda:{local_rank}"

    device = torch.device(device_str)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config_manager = ConfigManager()
    config_args = [f"--job.config_file={args.config}"]
    if args.custom_import:
        config_args.append(f"--experimental.custom_import={args.custom_import}")
    if args.hf_model:
        config_args.append(f"--hf_transformers.model={args.hf_model}")
    config = config_manager.parse_args(config_args)

    train_spec = get_train_spec(config.model.name)

    tokenizer = train_spec.build_tokenizer_fn(config)

    model_args = train_spec.model_args[config.model.flavor]
    model_args.update_from_config(config)

    init_device = "meta" if device.type == "cuda" else device_str
    with torch.device(init_device):
        model = train_spec.model_cls(model_args)

    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights()
    model.eval()

    state_dict = model.state_dict()
    logger.info("Loading checkpoint: %s", args.checkpoint)
    begin = time.monotonic()
    dcp.load(state_dict, checkpoint_id=args.checkpoint)
    logger.info("Checkpoint loaded in %.2f seconds", time.monotonic() - begin)

    stop_token_id = resolve_stop_token_id(tokenizer, args.stop_at_eos)

    samples = []
    if args.samples:
        samples.extend(load_samples(Path(args.samples)))
    else:
        samples.append(
            {
                "name": "prompt",
                "prompt": args.prompt,
                "fim_prefix": args.fim_prefix,
                "fim_suffix": args.fim_suffix,
                "fim_format": args.fim_format,
            }
        )

    output_data = {"metadata": {}, "responses": []}

    for sample in samples:
        name = sample.get("name", "sample")
        fim_prefix = sample.get("fim_prefix", args.fim_prefix)
        fim_suffix = sample.get("fim_suffix", args.fim_suffix)
        fim_format = sample.get("fim_format", args.fim_format)

        if fim_prefix is not None or fim_suffix is not None:
            prompt = build_fim_prompt(
                fim_prefix or "",
                fim_suffix or "",
                fim_format,
                fim_prefix_token="<|fim_prefix|>",
                fim_suffix_token="<|fim_suffix|>",
                fim_middle_token="<|fim_middle|>",
            )
        else:
            prompt = sample.get("prompt", args.prompt)

        if not prompt:
            logger.warning("Sample '%s' has empty prompt.", name)

        input_ids = tokenizer.encode(prompt, add_bos=args.add_bos, add_eos=args.add_eos)
        input_ids = (
            torch.tensor(input_ids, dtype=torch.long)
            .view(1, -1)
            .repeat(args.batch_size, 1)
            .to(device)
        )

        start = time.monotonic()
        responses = generate(
            model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            stop_token_id=stop_token_id,
        )
        elapsed = time.monotonic() - start

        input_n_tokens = input_ids.size(1)

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()
            input_text_raw = tokenizer.decode(inp_tok)
            output_text_raw = tokenizer.decode(out_tok)
            input_text = clean_byte_level_text(input_text_raw)
            output_text = clean_byte_level_text(output_text_raw)
            output_data["responses"].append(
                {
                    "name": name,
                    "response_idx": i,
                    "input_text_raw": input_text_raw,
                    "output_text_raw": output_text_raw,
                    "input_text": input_text,
                    "output_text": output_text,
                    "generation_time_sec": elapsed,
                }
            )

    output_data["metadata"] = {
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device_str,
        "torch_version": torch.__version__,
    }

    if args.out:
        print(json.dumps(output_data, indent=4))
    else:
        for item in output_data["responses"]:
            logger.info(
                "[%s] raw: %s%s", item["name"], item["input_text_raw"], item["output_text_raw"]
            )
            logger.info("[%s] clean: %s%s", item["name"], item["input_text"], item["output_text"])


if __name__ == "__main__":
    main()
