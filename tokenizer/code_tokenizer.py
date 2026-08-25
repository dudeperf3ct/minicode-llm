"""Custom tokenizer training."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from huggingface_hub import HfApi, get_token, upload_folder
from loguru import logger
from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors, trainers
from tokenizers.normalisers import NFKC

from config import TokenizerConfig


class CodeTokenizer:
    """High-level helper to train and save a code tokenizer."""

    def __init__(self, config: TokenizerConfig):
        """Codetokenizer init."""
        self.config = config
        self.tokenizer: Tokenizer | None = None

    def get_dataset(self) -> IterableDataset:
        """Get the dataset from Hugging Face Hub.

        Returns:
            Hugging Face Dataset
        """
        logger.info(f"Loading dataset {self.config.dataset.hf_path}")
        return load_dataset(
            self.config.dataset.hf_path,
            self.config.dataset.subset,
            split=self.config.dataset.split,
            streaming=self.config.dataset.streaming,
        )  # pyrefly: ignore[bad-return]

    def create_tokenizer(self):
        """Create a byte level BPE tokenizer.

        This follows StarCoder2/SantaCoder approach:
        1. Digit splitter (splits individual digits)
        2. GPT-2 regex for pre-tokenization
        3. ByteLevel encoding
        """
        self.tokenizer = Tokenizer(models.BPE())

        # GPT-2 pre-tokenization regex
        # This splits on:
        # - English contractions ('s, 't, 'll, 've, 're)
        # - Optional space + letters
        # - Optional space + numbers
        # - Optional space + non-letter/non-number characters
        # - Whitespace patterns
        gpt2_pattern = (
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        # Pre-tokenization sequence (ORDER MATTERS!):
        # 1. First split on individual digits (important for code!)
        # 2. Then apply GPT-2 regex
        # 3. Finally convert to bytes
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),  # Split each digit separately
                pre_tokenizers.Split(
                    pattern=Regex(gpt2_pattern),  # Apply GPT-2 regex pattern
                    behaviour="removed",  # Remove the delimiter (it's already captured)
                    invert=True,  # Split on things that DON'T match (keep what matches)
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False),  # Convert to bytes
            ]
        )

        # Normalise to NFKC (compatibility normalisation)
        # This handles things like superscript numbers: ² → 2
        self.tokenizer.normaliser = NFKC()

        return self.tokenizer

    def _special_tokens(self) -> list[str]:
        """List of special tokens."""
        return [
            "<|endoftext|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<|fim_pad|>",
            "<|file_separator|>",
            "<pad>",
            "<unk>",
            "<|repo_name|>",
            "<|file_name|>",
        ]

    def train_tokenizer(self, dataset_iterator: Iterable[str]) -> Tokenizer:
        """Train tokenizer and persist it under output_dir/tokenizer.json."""
        if self.tokenizer is None:
            self.create_tokenizer()

        trainer = trainers.BpeTrainer(
            vocab_size=self.config.training.vocab_size,
            special_tokens=self._special_tokens(),
            min_frequency=self.config.training.min_frequency,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        assert self.tokenizer is not None
        self.tokenizer.train_from_iterator(dataset_iterator, trainer)

        # Add post-processor for proper decoding
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Save tokenizer
        output_path = Path(self.config.output.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / "tokenizer.json"
        self.tokenizer.save(str(save_path))
        logger.info(f"Tokenizer saved to {save_path}")

        return self.tokenizer

    def create_dataset_iterator(self):
        """Create a iterator for dataset."""
        dataset = self.get_dataset()
        if self.config.dataset.shuffle:
            dataset = dataset.shuffle(
                seed=self.config.dataset.seed, buffer_size=self.config.dataset.shuffle_buffer
            )

        def text_iterator() -> Iterator[str]:
            for idx, item in enumerate(dataset):
                if (
                    self.config.dataset.max_samples is not None
                    and idx >= self.config.dataset.max_samples
                ):
                    logger.info(f"Reached sample cap of {self.config.dataset.max_samples} items")
                    break
                yield item[self.config.dataset.text_field]

        return text_iterator()

    def push_to_huggingface_hub(
        self, repo_id: str | None = None, token: str | None = None, private: bool | None = None
    ) -> None:
        """Push the trained tokenizer artefacts to Hugging Face Hub."""
        repo = repo_id or self.config.output.hub_repo_id
        if repo is None:
            msg = "hub_repo_id must be provided either in config or as an argument"
            raise ValueError(msg)

        privacy = private if private is not None else self.config.output.hub_private
        hf_token = token or get_token()
        if hf_token is None:
            msg = "No Hugging Face token found; login with `huggingface-cli login` or pass --hub-token."
            raise ValueError(msg)

        output_path = Path(self.config.output.output_dir)
        if not output_path.exists():
            msg = f"Output directory {output_path} does not exist; train before pushing."
            raise FileNotFoundError(msg)

        tokenizer_file = output_path / "tokenizer.json"
        if not tokenizer_file.exists():
            msg = f"{tokenizer_file} not found; train and save tokenizer before pushing."
            raise FileNotFoundError(msg)

        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo, repo_type="model", private=privacy, exist_ok=True)
        logger.info(f"Uploading tokenizer assets from {output_path} to {repo} (private={privacy})")
        upload_folder(folder_path=str(output_path), repo_id=repo, repo_type="model", token=hf_token)
