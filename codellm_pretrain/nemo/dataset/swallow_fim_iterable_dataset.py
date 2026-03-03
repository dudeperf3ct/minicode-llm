"""Streaming SwallowCode-v2 dataset with on-the-fly FIM transforms."""

import random
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerBase

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
END_OF_TEXT = "<|endoftext|>"


@dataclass
class FIMResult:
    text: str
    mode: str
    middle_chars: int
    skipped_short: bool


@dataclass
class FIMTelemetry:
    total_samples: int = 0
    fim_applied: int = 0
    psm_count: int = 0
    spm_count: int = 0
    skipped_short: int = 0
    skipped_missing_text: int = 0
    total_middle_chars: int = 0

    def update(self, result: FIMResult) -> None:
        self.total_samples += 1
        self.total_middle_chars += result.middle_chars
        if result.skipped_short:
            self.skipped_short += 1
        if result.mode == "psm":
            self.fim_applied += 1
            self.psm_count += 1
        elif result.mode == "spm":
            self.fim_applied += 1
            self.spm_count += 1

    def mark_missing_text(self) -> None:
        self.skipped_missing_text += 1

    def to_dict(self) -> dict[str, float | int]:
        avg_middle_chars = self.total_middle_chars / self.fim_applied if self.fim_applied else 0.0
        effective_fim_rate = self.fim_applied / self.total_samples if self.total_samples else 0.0
        return {
            "total_samples": self.total_samples,
            "fim_applied": self.fim_applied,
            "effective_fim_rate": effective_fim_rate,
            "psm_count": self.psm_count,
            "spm_count": self.spm_count,
            "avg_middle_chars": avg_middle_chars,
            "skipped_short": self.skipped_short,
            "skipped_missing_text": self.skipped_missing_text,
        }


@dataclass(frozen=True)
class FIMConfig:
    fim_rate: float = 0.8
    psm_prob: float = 0.5
    spm_prob: float = 0.5
    min_code_length: int = 64


@dataclass(frozen=True)
class HFDatasetConfig:
    path: str = "tokyotech-llm/swallow-code-v2"
    subset: str = "swallowcode-v2"
    split: str = "train"
    text_field: str = "improved_code"
    revision: str | None = None


class FIMTransform:
    """Character-level FIM transform with PSM/SPM formats only."""

    def __init__(self, config: FIMConfig | None = None) -> None:
        self.config = config or FIMConfig()
        if not 0.0 <= self.config.fim_rate <= 1.0:
            raise ValueError(f"fim_rate must be in [0, 1], got {self.config.fim_rate}")
        if self.config.psm_prob < 0.0 or self.config.spm_prob < 0.0:
            raise ValueError("psm_prob and spm_prob must be non-negative")
        total_prob = self.config.psm_prob + self.config.spm_prob
        if abs(total_prob - 1.0) > 1e-6:
            raise ValueError(f"psm_prob + spm_prob must equal 1.0, got {total_prob}")
        if self.config.min_code_length < 0:
            raise ValueError("min_code_length must be >= 0")

    def _choose_mode(self, rng: random.Random) -> str:
        return "psm" if rng.random() < self.config.psm_prob else "spm"

    def apply(self, text: str, rng: random.Random) -> FIMResult:
        if len(text) < self.config.min_code_length:
            return FIMResult(
                text=f"{text}{END_OF_TEXT}", mode="none", middle_chars=0, skipped_short=True
            )

        if rng.random() >= self.config.fim_rate:
            return FIMResult(
                text=f"{text}{END_OF_TEXT}", mode="none", middle_chars=0, skipped_short=False
            )

        cut_a = rng.randint(0, len(text))
        cut_b = rng.randint(0, len(text))
        start = min(cut_a, cut_b)
        end = max(cut_a, cut_b)

        prefix = text[:start]
        middle = text[start:end]
        suffix = text[end:]
        mode = self._choose_mode(rng)

        if mode == "psm":
            transformed = (
                f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}{END_OF_TEXT}"
            )
        else:
            transformed = (
                f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}{END_OF_TEXT}"
            )

        return FIMResult(
            text=transformed,
            mode=mode,
            middle_chars=len(middle),
            skipped_short=False,
        )


class SwallowCodeFIMIterableDataset(IterableDataset):
    """Streaming SwallowCode-v2 dataset yielding fixed-length token training samples."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        seq_len: int,
        *,
        hf_config: HFDatasetConfig | None = None,
        fim_config: FIMConfig | None = None,
        seed: int = 42,
        infinite: bool = True,
        samples_per_epoch: int | None = None,
        max_rows_per_epoch: int | None = None,
        shuffle_buffer_size: int = 0,
        vary_fim_across_epochs: bool = True,
        dataset: Any = None,
    ) -> None:
        super().__init__()
        if seq_len <= 1:
            raise ValueError(f"seq_len must be > 1, got {seq_len}")

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.hf_config = hf_config or HFDatasetConfig()
        self.seed = seed
        self.infinite = infinite
        self.samples_per_epoch = samples_per_epoch
        self.max_rows_per_epoch = max_rows_per_epoch
        self.shuffle_buffer_size = shuffle_buffer_size
        self.vary_fim_across_epochs = vary_fim_across_epochs
        self.rank = 0
        self.world_size = 1
        self.fim_config = fim_config or FIMConfig()

        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be > 0 when provided")
        if self.max_rows_per_epoch is not None and self.max_rows_per_epoch <= 0:
            raise ValueError("max_rows_per_epoch must be > 0 when provided")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be >= 0")

        self.transform = FIMTransform(config=self.fim_config)
        self.telemetry = FIMTelemetry()

        self.dataset = dataset if dataset is not None else self._load_streaming_dataset()

    def _load_streaming_dataset(self):
        dataset = load_dataset(
            self.hf_config.path,
            self.hf_config.subset,
            split=self.hf_config.split,
            streaming=True,
            revision=self.hf_config.revision,
        )
        if self.shuffle_buffer_size > 0 and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer_size)
        return dataset

    def _clone_with_dataset(self, dataset: Any) -> "SwallowCodeFIMIterableDataset":
        clone = SwallowCodeFIMIterableDataset(
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            hf_config=self.hf_config,
            fim_config=self.fim_config,
            seed=self.seed,
            infinite=self.infinite,
            samples_per_epoch=self.samples_per_epoch,
            max_rows_per_epoch=self.max_rows_per_epoch,
            shuffle_buffer_size=self.shuffle_buffer_size,
            vary_fim_across_epochs=self.vary_fim_across_epochs,
            dataset=dataset,
        )
        clone.rank = self.rank
        clone.world_size = self.world_size
        return clone

    def shard(self, num_shards: int, index: int) -> "SwallowCodeFIMIterableDataset":
        if num_shards <= 1:
            return self
        if not hasattr(self.dataset, "shard"):
            return self
        sharded = self.dataset.shard(num_shards=num_shards, index=index)
        clone = self._clone_with_dataset(sharded)
        clone.rank = index
        clone.world_size = num_shards
        return clone

    def shuffle(self, buffer_size: int = 10_000, seed: int = 42) -> "SwallowCodeFIMIterableDataset":
        if not hasattr(self.dataset, "shuffle"):
            return self
        shuffled = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        return self._clone_with_dataset(shuffled)

    def get_telemetry(self) -> dict[str, float | int]:
        return self.telemetry.to_dict()

    def __len__(self) -> int:
        if self.samples_per_epoch is None:
            raise TypeError(
                "SwallowCodeFIMIterableDataset has no static length. "
                "Set dataset.samples_per_epoch when using lr_scheduler."
            )
        return self.samples_per_epoch

    def _iter_worker_stream(self):
        stream = self.dataset
        worker = get_worker_info()
        if worker is not None and worker.num_workers > 1 and hasattr(stream, "shard"):
            stream = stream.shard(num_shards=worker.num_workers, index=worker.id)
        return stream

    def _build_rng(self, epoch: int) -> random.Random:
        worker = get_worker_info()
        worker_offset = worker.id if worker is not None else 0
        epoch_seed = epoch if self.vary_fim_across_epochs else 0
        seed = self.seed + (epoch_seed * 10_007) + (self.rank * 1_000_003) + worker_offset
        return random.Random(seed)

    def __iter__(self):
        max_tokens = self.seq_len + 1
        token_buffer: list[int] = []
        epoch = 0

        while True:
            yielded_in_epoch = 0
            rng = self._build_rng(epoch)
            worker_stream = self._iter_worker_stream()
            row_iter = (
                islice(worker_stream, self.max_rows_per_epoch)
                if self.max_rows_per_epoch is not None
                else worker_stream
            )
            for row in row_iter:
                raw_text = row.get(self.hf_config.text_field)
                if not isinstance(raw_text, str) or not raw_text:
                    self.telemetry.mark_missing_text()
                    continue

                fim_result = self.transform.apply(raw_text, rng)
                self.telemetry.update(fim_result)

                token_ids = self.tokenizer.encode(fim_result.text, add_special_tokens=False)
                if not token_ids:
                    continue

                token_buffer.extend(token_ids)
                while len(token_buffer) >= max_tokens:
                    chunk = token_buffer[:max_tokens]
                    token_buffer = token_buffer[max_tokens:]
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                    }
                    yielded_in_epoch += 1
                    if (
                        self.samples_per_epoch is not None
                        and yielded_in_epoch >= self.samples_per_epoch
                    ):
                        break

                if (
                    self.samples_per_epoch is not None
                    and yielded_in_epoch >= self.samples_per_epoch
                ):
                    break

            if not self.infinite:
                break
            epoch += 1
