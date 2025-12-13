"""Training and evaluating the tokenizer from the command line."""

from argparse import ArgumentParser
from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from config import TokenizerConfig
from eval import TokenizerEvaluator
from logging_config import setup_logging


def build_arg_parser() -> ArgumentParser:
    """Build arguments from CLI."""
    parser = ArgumentParser(description="Train a byte-level BPE tokenizer for code.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run lightweight evaluation on canned code snippets after training",
    )
    parser.add_argument(
        "--hub-token",
        default=None,
        help="Hugging Face token. If omitted, uses cached auth (huggingface-cli login) or env HF_TOKEN.",
    )
    return parser


def load_config(path: Path) -> TokenizerConfig:
    """Load and create configuration."""
    if not path.exists():
        msg = f"Config file {path} not found"
        raise FileNotFoundError(msg)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    try:
        config = TokenizerConfig.from_dict(raw)
    except ValidationError as exc:
        msg = f"Invalid config in {path}: {exc}"
        raise ValueError(msg) from exc
    logger.info(f"Loaded config from {path}")
    return config


def run_evaluation(tokenizer_path: Path) -> None:
    """Run evaluation suite."""
    evaluator = TokenizerEvaluator(str(tokenizer_path))
    evaluator.full_evaluation()


def main() -> None:
    """Main entrypoint."""
    setup_logging(Path.cwd())
    args = build_arg_parser().parse_args()
    config = load_config(Path(args.config))
    logger.info(f"Configuration: {config}")

    # tokenizer = CodeTokenizer(config)
    # st = perf_counter()
    # dataset_iterator = tokenizer.create_dataset_iterator()
    # logger.info(f"Time taken to setup dataset: {perf_counter() - st} seconds")

    # logger.info("Training a custom tokenizer")
    # st = perf_counter()
    # tokenizer.train_tokenizer(dataset_iterator)
    # logger.info(f"Time taken to train tokenizer: {perf_counter() - st} seconds")

    if args.run_eval:
        tokenizer_path = Path(config.output.output_dir) / "tokenizer.json"
        run_evaluation(tokenizer_path)

    # if config.output.push_to_hf_hub:
    #     logger.info("Pushing the tokenizer to huggingface hub")
    #     tokenizer.push_to_huggingface_hub(token=args.hub_token)


if __name__ == "__main__":
    main()
