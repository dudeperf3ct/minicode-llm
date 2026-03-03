"""Train entrypoint with quieter third-party HTTP logs.

Use with torchrun:
  torchrun --nproc-per-node=1 -m train_entry --config <yaml>
"""

import logging


def _quiet_noisy_loggers() -> None:
    for name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    _quiet_noisy_loggers()
    from nemo_automodel.recipes.llm.train_ft import main as train_main

    train_main()


if __name__ == "__main__":
    main()
