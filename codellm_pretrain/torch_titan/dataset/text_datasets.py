# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import Any

import torch
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.tools.logging import logger


def _load_c4_dataset(dataset_path: str, split: str):
    """Load C4 dataset with default configuration."""
    return load_dataset(dataset_path, name="en", split=split, streaming=True)


def _process_c4_text(sample: dict[str, Any]) -> str:
    """Process C4 dataset sample text."""
    return sample["text"]


def _load_swallowcode_v2_dataset(dataset_path: str, split: str, subset: str):
    """Load SwallowCode-v2 dataset with the specified subset."""
    return load_dataset(dataset_path, subset, split=split, streaming=True)


def _process_swallowcode_text(sample: dict[str, Any]) -> str:
    """Process SwallowCode-v2 dataset sample text."""
    return sample["improved_code"]


class ProcessSwallowCodeDataset:
    def __init__(
        self, rank: int, seed: int = 42, fim_rate: float = 0.5, min_code_length: int = 100
    ):
        self.rng = random.Random(seed + rank)
        self.fim_rate = fim_rate
        self.min_code_length = min_code_length

        # FIM tokens
        self.fim_prefix = "<|fim_prefix|>"
        self.fim_middle = "<|fim_middle|>"
        self.fim_suffix = "<|fim_suffix|>"
        self.endoftext = "<|endoftext|>"

    def _select_fim_format(self) -> str:
        """
        Select FIM format according to StarCoder2 strategy:
        - PSM (Prefix-Suffix-Middle): 50%
        - SPM (Suffix-Prefix-Middle): 25%
        - Middle-only: 25%
        """
        rand = self.rng.random()
        if rand < 0.5:
            return "PSM"  # Prefix-Suffix-Middle
        if rand < 0.75:
            return "SPM"  # Suffix-Prefix-Middle
        return "M"  # Middle-only

    def apply_fim_to_text(self, code):
        # No FIM (50% of time)
        if self.rng.random() > self.fim_rate:
            return code + self.endoftext

        fim_type = self._select_fim_format()

        # Select span (character-based)
        code_len = len(code)
        if code_len < self.min_code_length:
            return code + self.endoftext

        # Middle span: 10-50% of code
        min_middle = code_len // 10
        max_middle = code_len // 2

        middle_start = self.rng.randint(0, code_len - min_middle)
        middle_len = self.rng.randint(min_middle, min(max_middle, code_len - middle_start))
        middle_end = middle_start + middle_len

        prefix = code[:middle_start]
        middle = code[middle_start:middle_end]
        suffix = code[middle_end:]

        # Format based on type
        if fim_type == "PSM":
            return (
                f"{self.fim_prefix}{prefix}"
                f"{self.fim_suffix}{suffix}"
                f"{self.fim_middle}{middle}"
                f"{self.endoftext}"
            )
        if fim_type == "SPM":
            return (
                f"{self.fim_suffix}{suffix}"
                f"{self.fim_prefix}{prefix}"
                f"{self.fim_middle}{middle}"
                f"{self.endoftext}"
            )
        return f"{self.fim_middle}{middle}{self.endoftext}"


# Add your dataset here - more information at docs/datasets.md
DATASETS = {
    "c4": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_test": DatasetConfig(
        path="tests/assets/c4_test",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
    "c4_validation": DatasetConfig(
        path="allenai/c4",
        loader=partial(_load_c4_dataset, split="validation"),
        sample_processor=_process_c4_text,
    ),
    "swallowcode": DatasetConfig(
        path="tokyotech-llm/swallow-code-v2",
        loader=partial(_load_swallowcode_v2_dataset, split="train", subset="swallowcode-v2"),
        sample_processor=_process_swallowcode_text,
    ),
}


def _validate_dataset(
    dataset_name: str, dataset_path: str | None = None
) -> tuple[str, Callable, Callable]:
    """Validate dataset name and path."""
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Dataset {dataset_name} is not supported. "
            f"Supported datasets are: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    path = dataset_path or config.path
    logger.info(f"Preparing {dataset_name} dataset from {path}")
    return path, config.loader, config.sample_processor


class HuggingFaceTextDataset(IterableDataset, Stateful):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int = 2048,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        infinite: bool = False,
    ) -> None:
        # Force lowercase for consistent comparison
        dataset_name = dataset_name.lower()

        path, dataset_loader, text_processor = _validate_dataset(dataset_name, dataset_path)
        ds = dataset_loader(path)

        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite

        self._text_processor = text_processor
        self._fim_processor = None
        self._add_bos = True
        self._add_eos = True
        if self.dataset_name == "swallowcode":
            self._fim_processor = ProcessSwallowCodeDataset(rank=dp_rank).apply_fim_to_text
            self._add_bos = False
            self._add_eos = False

        # Variables for checkpointing
        self._sample_idx = 0
        self._token_buffer: list[int] = []

    def _get_data_iter(self):
        # For map-style datasets, resume by skipping to the correct index
        # For iterable-style datasets, the underlying iterator already points to the correct index
        if isinstance(self._data, Dataset):
            if self._sample_idx == len(self._data):
                return iter([])
            return iter(self._data.skip(self._sample_idx))

        return iter(self._data)

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = self._text_processor(sample)
                if self._fim_processor is not None:
                    sample_text = self._fim_processor(sample_text)

                sample_tokens = self._tokenizer.encode(
                    sample_text, add_bos=self._add_bos, add_eos=self._add_eos
                )

                self._token_buffer.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._token_buffer) >= max_buffer_token_len:
                    x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._token_buffer = self._token_buffer[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield {"input": input}, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data")
                break

            # Reset offset for the next iteration
            self._sample_idx = 0
            logger.warning(f"Dataset {self.dataset_name} is being re-looped")
            # Ensures re-looping a dataset loaded from a checkpoint works correctly
            if not isinstance(self._data, Dataset):
                if hasattr(self._data, "set_epoch") and hasattr(self._data, "epoch"):
                    self._data.set_epoch(self._data.epoch + 1)

    def load_state_dict(self, state_dict):
        self._token_buffer = state_dict["token_buffer"]

        if isinstance(self._data, Dataset):
            self._sample_idx = state_dict["sample_idx"]
        else:
            assert "data" in state_dict
            self._data.load_state_dict(state_dict["data"])

    def state_dict(self):
        _state_dict: dict[str, Any] = {"token_buffer": self._token_buffer}

        if isinstance(self._data, Dataset):
            _state_dict["sample_idx"] = self._sample_idx
        else:
            # Save the iterable dataset's state to later efficiently resume from it
            # https://huggingface.co/docs/datasets/v3.5.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration
            _state_dict["data"] = self._data.state_dict()

        return _state_dict


def build_text_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """Build a data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.training.dataset
    dataset_path = job_config.training.dataset_path
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.training.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )


def build_text_validation_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = False,
) -> ParallelAwareDataloader:
    """Build a validation data loader for HuggingFace datasets.

    Args:
        dp_world_size: Data parallelism world size.
        dp_rank: Data parallelism rank.
        tokenizer: Tokenizer to use for encoding text.
        job_config: Job configuration containing dataset and DataLoader settings.
        infinite: Whether to loop the dataset infinitely.
    """
    dataset_name = job_config.validation.dataset
    dataset_path = job_config.validation.dataset_path
    batch_size = job_config.validation.local_batch_size
    seq_len = job_config.validation.seq_len

    hf_ds = HuggingFaceTextDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=infinite,
    )

    dataloader_kwargs = {
        **asdict(job_config.validation.dataloader),
        "batch_size": batch_size,
    }

    return ParallelAwareDataloader(
        hf_ds,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        **dataloader_kwargs,
    )
