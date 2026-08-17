import argparse
import json
from importlib.metadata import version
from pathlib import Path

DATASETS = ("humaneval", "mbpp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=("direct", "thinking"))
    parser.add_argument("--backend", required=True, choices=("vllm", "openai"))
    parser.add_argument("--base-url")
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "openai" and not args.base_url:
        raise ValueError("--base-url is required for the OpenAI backend")
    if args.backend == "vllm" and args.base_url:
        raise ValueError("--base-url is only valid for the OpenAI backend")

    args.result_dir.mkdir(parents=True, exist_ok=True)
    write_protocol(args)
    install_model_factory(args.max_new_tokens)

    from evalplus.evaluate import evaluate

    evaluate(
        dataset=args.dataset,
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        root=str(args.result_dir),
        greedy=args.greedy,
        temperature=args.temperature,
        n_samples=1,
        bs=1,
        tp=1,
        dtype="bfloat16",
        max_new_tokens=args.max_new_tokens,
    )


def install_model_factory(max_new_tokens: int) -> None:
    from evalplus import codegen

    def make_model(**kwargs):
        backend = kwargs["backend"]
        if backend == "vllm":
            from evalplus.provider.vllm import VllmDecoder

            return VllmDecoder(
                name=kwargs["model"],
                batch_size=kwargs["batch_size"],
                temperature=kwargs["temperature"],
                dataset=kwargs["dataset"],
                force_base_prompt=kwargs["force_base_prompt"],
                tensor_parallel_size=kwargs["tp"],
                instruction_prefix=kwargs["instruction_prefix"],
                response_prefix=kwargs["response_prefix"],
                trust_remote_code=kwargs["trust_remote_code"],
                enable_prefix_caching=kwargs["enable_prefix_caching"],
                enable_chunked_prefill=kwargs["enable_chunked_prefill"],
                dtype=kwargs["dtype"],
                gguf_file=kwargs["gguf_file"],
                max_new_tokens=max_new_tokens,
            )

        if backend == "openai":
            from evalplus.provider.openai import OpenAIChatDecoder

            class NullSafeOpenAIChatDecoder(OpenAIChatDecoder):
                def codegen(
                    self, prompt: str, do_sample: bool = True, num_samples: int = 200
                ) -> list[str]:
                    outputs = super().codegen(prompt, do_sample, num_samples)
                    return [output or "" for output in outputs]

            return NullSafeOpenAIChatDecoder(
                name=kwargs["model"],
                batch_size=kwargs["batch_size"],
                temperature=kwargs["temperature"],
                base_url=kwargs["base_url"],
                verify_certificate=kwargs["verify_certificate"],
                instruction_prefix=kwargs["instruction_prefix"],
                response_prefix=kwargs["response_prefix"],
                max_new_tokens=max_new_tokens,
            )

        raise ValueError(f"Unsupported backend: {backend}")

    codegen.make_model = make_model


def write_protocol(args: argparse.Namespace) -> None:
    protocol_path = args.result_dir / "protocol.json"
    protocol = {
        "profile": args.profile,
        "model": args.model,
        "backend": args.backend,
        "base_url": args.base_url,
        "decoding": {
            "greedy": args.greedy,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "n_samples": 1,
            "batch_size": 1,
        },
        "runtime": runtime_protocol(args),
        "environment": {"evalplus": version("evalplus")},
    }

    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        comparable_keys = (
            "profile",
            "model",
            "backend",
            "base_url",
            "decoding",
            "runtime",
        )
        if any(existing.get(key) != protocol[key] for key in comparable_keys):
            raise RuntimeError(
                f"{protocol_path} describes a different evaluation protocol; "
                "use a separate result directory"
            )
        return

    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")


def runtime_protocol(args: argparse.Namespace) -> dict:
    if args.backend == "vllm":
        return {
            "executor": "embedded-vllm",
            "max_model_len": 2048,
        }

    return {
        "executor": "openai-compatible-vllm-server",
        "required_chat_template_kwargs": {
            "enable_thinking": args.profile == "thinking"
        },
        "required_generation_config": "auto",
        "required_max_model_len": 65536 if args.profile == "thinking" else 2048,
    }


if __name__ == "__main__":
    main()
