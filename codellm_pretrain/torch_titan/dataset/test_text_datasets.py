"""Tests for SwallowCode FIM formatting and tokenization."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest
from abc import ABC, abstractmethod


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


class BaseTokenizer(ABC):
    # base tokenizer interface, for typing purpose mainly
    def __init__(self):
        self.eos_id = 0

    @abstractmethod
    def encode(self, *args, **kwargs) -> list[int]: ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str: ...

    @abstractmethod
    def get_vocab_size(self) -> int: ...


class DummyTokenizer(BaseTokenizer):
    """A dummy tokenizer for testing that implements BaseTokenizer interface."""

    def __init__(self):
        super().__init__()
        self.eos_id = 2

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = [ord(c) for c in text]
        if add_bos:
            tokens.insert(0, 1)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t > 2)

    def get_vocab_size(self) -> int:
        return 256


class DeterministicProcessSwallowCode(ProcessSwallowCodeDataset):
    """Force a deterministic FIM format for tests."""

    def __init__(self, fim_format: str, **kwargs):
        super().__init__(**kwargs)
        self._fim_format = fim_format

    def _select_fim_format(self) -> str:
        return self._fim_format


class TestSwallowCodeFim(unittest.TestCase):
    def setUp(self):
        self.code = "def add(x, y):\n    return x + y\n" * 20

    def test_no_fim_for_short_code(self):
        processor = ProcessSwallowCodeDataset(rank=0, seed=0, fim_rate=1.0, min_code_length=1000)
        result = processor.apply_fim_to_text("print('hi')")
        self.assertEqual(result, "print('hi')" + processor.endoftext)
        self.assertNotIn(processor.fim_prefix, result)
        self.assertNotIn(processor.fim_suffix, result)
        self.assertNotIn(processor.fim_middle, result)

    def test_no_fim_when_rate_zero(self):
        processor = ProcessSwallowCodeDataset(rank=0, seed=0, fim_rate=0.0, min_code_length=10)
        result = processor.apply_fim_to_text(self.code)
        self.assertEqual(result, self.code + processor.endoftext)
        self.assertNotIn(processor.fim_prefix, result)
        self.assertNotIn(processor.fim_suffix, result)
        self.assertNotIn(processor.fim_middle, result)

    def test_fim_psm_reconstructs_code(self):
        processor = DeterministicProcessSwallowCode(
            fim_format="PSM", rank=0, seed=0, fim_rate=1.0, min_code_length=10
        )
        result = processor.apply_fim_to_text(self.code)
        self.assertTrue(result.endswith(processor.endoftext))

        prefix = result.split(processor.fim_prefix, 1)[1].split(processor.fim_suffix, 1)[0]
        suffix = result.split(processor.fim_suffix, 1)[1].split(processor.fim_middle, 1)[0]
        middle = result.split(processor.fim_middle, 1)[1].split(processor.endoftext, 1)[0]
        self.assertEqual(prefix + middle + suffix, self.code)

    def test_fim_spm_reconstructs_code(self):
        processor = DeterministicProcessSwallowCode(
            fim_format="SPM", rank=0, seed=0, fim_rate=1.0, min_code_length=10
        )
        result = processor.apply_fim_to_text(self.code)
        self.assertTrue(result.endswith(processor.endoftext))

        suffix = result.split(processor.fim_suffix, 1)[1].split(processor.fim_prefix, 1)[0]
        prefix = result.split(processor.fim_prefix, 1)[1].split(processor.fim_middle, 1)[0]
        middle = result.split(processor.fim_middle, 1)[1].split(processor.endoftext, 1)[0]
        self.assertEqual(prefix + middle + suffix, self.code)

    def test_fim_middle_only_is_substring(self):
        processor = DeterministicProcessSwallowCode(
            fim_format="M", rank=0, seed=0, fim_rate=1.0, min_code_length=10
        )
        result = processor.apply_fim_to_text(self.code)
        self.assertTrue(result.startswith(processor.fim_middle))
        self.assertTrue(result.endswith(processor.endoftext))
        self.assertNotIn(processor.fim_prefix, result)
        self.assertNotIn(processor.fim_suffix, result)
        middle = result[len(processor.fim_middle) : -len(processor.endoftext)]
        self.assertIn(middle, self.code)

    def test_tokenization_round_trip_for_fim(self):
        processor = DeterministicProcessSwallowCode(
            fim_format="PSM", rank=0, seed=1, fim_rate=1.0, min_code_length=10
        )
        result = processor.apply_fim_to_text(self.code)
        tokenizer = DummyTokenizer()
        token_ids = tokenizer.encode(result, add_bos=False, add_eos=False)
        self.assertEqual(tokenizer.decode(token_ids), result)
        self.assertNotIn(tokenizer.eos_id, token_ids)


if __name__ == "__main__":
    unittest.main()
