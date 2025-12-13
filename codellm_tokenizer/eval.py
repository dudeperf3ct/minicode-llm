"""Evaluation suite."""

import keyword

import numpy as np
from loguru import logger
from tokenizers import Tokenizer


class TokenizerEvaluator:
    """Evaluation suite for tokenizer."""

    def __init__(self, tokenizer_path: str):
        """Load trained tokenizer."""
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def calculate_compression_ratio(self, texts):
        """Calculate compression ratio: chars per token.

        Higher is better for efficiency
        """
        total_chars = 0
        total_tokens = 0

        for text in texts:
            encoded = self.tokenizer.encode(text)
            total_chars += len(text)
            total_tokens += len(encoded.tokens)

        ratio = total_chars / total_tokens if total_tokens > 0 else 0
        return ratio

    def calculate_fertility(self, texts):
        """Calculate fertility: average tokens per word.

        Lower is generally better (but not always predictive!)
        """
        total_tokens = 0
        total_words = 0

        for text in texts:
            encoded = self.tokenizer.encode(text)
            words = text.split()  # Simple whitespace splitting

            total_tokens += len(encoded.tokens)
            total_words += len(words)

        fertility = total_tokens / total_words if total_words > 0 else 0
        return fertility

    def evaluate_indentation(self, indent_samples):
        """Evaluate how efficiently indentation is encoded."""
        results = []

        for indent_type, sample in indent_samples.items():
            encoded = self.tokenizer.encode(sample)
            tokens = encoded.tokens

            # Count tokens used just for indentation
            indent_tokens = [t for t in tokens if t.strip() == ""]

            results.append(
                {
                    "type": indent_type,
                    "text": sample,
                    "total_tokens": len(tokens),
                    "indent_tokens": len(indent_tokens),
                    "efficiency": len(indent_tokens) / len(tokens) if len(tokens) > 0 else 0,
                }
            )

        return results

    def evaluate_comments(self, comment_samples):
        """Evaluate comment tokenization."""
        results = []

        for comment in comment_samples:
            encoded = self.tokenizer.encode(comment)
            tokens = encoded.tokens

            results.append(
                {
                    "comment": comment,
                    "tokens": tokens[:10],  # First 10 tokens
                    "num_tokens": len(tokens),
                    "compression": len(comment) / len(tokens) if len(tokens) > 0 else 0,
                }
            )

        return results

    def evaluate_identifiers(self, identifier_samples):
        """Evaluate tokenization of common code patterns.

        - snake_case
        - CamelCase
        - SCREAMING_SNAKE_CASE
        """
        results = []

        for pattern_type, identifiers in identifier_samples.items():
            pattern_results = []

            for identifier in identifiers:
                encoded = self.tokenizer.encode(identifier)
                tokens = encoded.tokens

                pattern_results.append(
                    {
                        "identifier": identifier,
                        "tokens": tokens,
                        "num_tokens": len(tokens),
                        "chars_per_token": len(identifier) / len(tokens) if len(tokens) > 0 else 0,
                    }
                )

            avg_tokens = np.mean([r["num_tokens"] for r in pattern_results])
            avg_cpt = np.mean([r["chars_per_token"] for r in pattern_results])

            results.append(
                {
                    "pattern": pattern_type,
                    "samples": pattern_results,
                    "avg_tokens": avg_tokens,
                    "avg_chars_per_token": avg_cpt,
                }
            )

        return results

    def evaluate_numbers(self, number_samples):
        """Evaluate number tokenization.

        Check if digits are split appropriately
        """
        results = []

        for num_str in number_samples:
            encoded = self.tokenizer.encode(num_str)
            tokens = encoded.tokens

            results.append(
                {
                    "number": num_str,
                    "tokens": tokens,
                    "num_tokens": len(tokens),
                    "split_correctly": len(tokens)
                    >= len(num_str.replace(".", "").replace("-", "")),
                }
            )

        return results

    def evaluate_special_tokens(self):
        """Check if special tokens are properly registered."""
        special_tokens = []
        vocab = self.tokenizer.get_vocab()

        common_special = [
            "<|endoftext|>",  # End of text/document
            "<|fim_prefix|>",  # Fill-in-middle: prefix
            "<|fim_middle|>",  # Fill-in-middle: middle (what to generate)
            "<|fim_suffix|>",  # Fill-in-middle: suffix
            "<|fim_pad|>",  # Fill-in-middle: padding
            "<|file_separator|>",  # Separate different files
            "<pad>",  # Padding token
            "<unk>",  # Unknown token (rare with byte-level)
            "<|repo_name|>",  # Repository metadata
            "<|file_name|>",  # File name metadata
        ]

        for token in common_special:
            if token in vocab:
                special_tokens.append({"token": token, "id": vocab[token], "present": True})
            else:
                special_tokens.append({"token": token, "id": None, "present": False})

        return special_tokens

    def evaluate_python_keywords(self, keywords):
        """Evaluate tokenization of Python reserved keywords."""
        results = []
        for kw in keywords:
            encoded = self.tokenizer.encode(kw)
            tokens = encoded.tokens
            results.append(
                {
                    "keyword": kw,
                    "tokens": tokens,
                    "num_tokens": len(tokens),
                    "chars_per_token": len(kw) / len(tokens) if len(tokens) > 0 else 0,
                }
            )
        return results

    def compare_with_baseline(self, texts, baseline_tokenizer: str = "gpt2"):
        """Compare current tokenizer with a GPT-2 pretrained tokenizer."""
        baseline = Tokenizer.from_pretrained(baseline_tokenizer)

        current_lengths = []
        baseline_lengths = []

        for text in texts:
            current_enc = self.tokenizer.encode(text)
            baseline_enc = baseline.encode(text)

            current_lengths.append(len(current_enc.tokens))
            baseline_lengths.append(len(baseline_enc.tokens))

        nsl = np.mean([y / b for y, b in zip(current_lengths, baseline_lengths)])

        return {
            "current_avg_length": np.mean(current_lengths),
            "baseline_avg_length": np.mean(baseline_lengths),
            "nsl": nsl,  # < 1.0 means you're more efficient
            "improvement": (1 - nsl) * 100,  # Percentage improvement
        }

    def evaluate_unseen_identifiers(self, identifiers):
        """Evaluate tokenizer behaviour on unseen snake_case identifiers."""
        results = []

        all_tokens = []
        for ident in identifiers:
            enc = self.tokenizer.encode(ident)
            all_tokens.extend(enc.tokens)

            results.append(
                {
                    "identifier": ident,
                    "tokens": enc.tokens,
                    "num_tokens": len(enc.tokens),
                    "chars_per_token": len(ident) / len(enc.tokens),
                }
            )

        unique_tokens = len(set(all_tokens))
        reuse_ratio = 1 - unique_tokens / len(all_tokens) if all_tokens else 0

        return {
            "samples": results,
            "avg_tokens": sum(r["num_tokens"] for r in results) / len(results),
            "avg_chars_per_token": sum(r["chars_per_token"] for r in results) / len(results),
            "token_reuse_ratio": reuse_ratio,
        }

    def full_evaluation(self, baseline_tokenizer: str = "gpt2"):
        """Run comprehensive evaluation."""
        logger.info("=" * 60)
        logger.info("TOKENIZER EVALUATION REPORT")
        logger.info("=" * 60)

        # 1. Test samples
        code_samples = [
            "def hello_world():\n    print('Hello, world!')",
            "class MyClass:\n    def __init__(self):\n        pass",
            "for i in range(100):\n    if i % 2 == 0:\n        print(i)",
            "import numpy as np\nimport pandas as pd\n\ndata = pd.read_csv('file.csv')",
        ]

        logger.info("\n1. COMPRESSION RATIO")
        logger.info("-" * 60)
        compression = self.calculate_compression_ratio(code_samples)
        logger.info("Compression Ratio: {:.2f} chars/token", compression)
        logger.info("Interpretation: Higher is better. Code typically: 3-5")

        logger.info("\n2. FERTILITY")
        logger.info("-" * 60)
        fertility = self.calculate_fertility(code_samples)
        logger.info("Fertility: {:.2f} tokens/word", fertility)
        logger.info("Note: Lower is generally better, but not always predictive!")

        # 3. Indentation
        logger.info("\n3. INDENTATION HANDLING")
        logger.info("-" * 60)
        indent_samples = {
            "4_spaces": "    def foo():",
            "2_spaces": "  def foo():",
            "tabs": "\tdef foo():",
            "nested_4": "        nested_function()",
        }

        indent_results = self.evaluate_indentation(indent_samples)
        for result in indent_results:
            logger.info(
                "{:<20}: {:2d} tokens ({} for indent)",
                result["type"],
                result["total_tokens"],
                result["indent_tokens"],
            )

        # 4. Identifier patterns
        logger.info("\n4. IDENTIFIER TOKENIZATION")
        logger.info("-" * 60)
        identifier_samples = {
            "snake_case": ["print_hello_world", "calculate_total_sum", "get_user_data"],
            "CamelCase": ["MyClassName", "DataProcessor", "HTTPConnection"],
            "SCREAMING": ["MAX_VALUE", "API_KEY", "DEFAULT_TIMEOUT"],
        }

        id_results = self.evaluate_identifiers(identifier_samples)
        for result in id_results:
            logger.info(
                "{:<15}: avg {:.1f} tokens, {:.2f} chars/token",
                result["pattern"],
                result["avg_tokens"],
                result["avg_chars_per_token"],
            )

        # 5. Numbers
        logger.info("\n5. NUMBER TOKENIZATION")
        logger.info("-" * 60)
        number_samples = ["42", "3.14159", "127", "0xFF", "1e-10", "1000000"]

        num_results = self.evaluate_numbers(number_samples)
        for result in num_results:
            correct = "ok" if result["split_correctly"] else "split_issue"
            logger.info("{:<10}: {} {}", result["number"], result["tokens"], correct)

        # 6. Comments
        logger.info("\n6. COMMENT TOKENIZATION")
        logger.info("-" * 60)
        comment_samples = [
            "# This is a simple comment",
            "# TODO: Implement this feature",
            "# FIXME: Bug in line 42",
            '"""This is a docstring with multiple words"""',
        ]

        comment_results = self.evaluate_comments(comment_samples)
        for result in comment_results:
            logger.info(
                "Tokens: {:3d}, Compression: {:.2f}",
                result["num_tokens"],
                result["compression"],
            )

        # 7. Special tokens
        logger.info("\n7. SPECIAL TOKENS")
        logger.info("-" * 60)
        special_results = self.evaluate_special_tokens()
        for result in special_results:
            status = "present" if result["present"] else "missing"
            logger.info("{:<20}: {} (ID: {})", result["token"], status, result["id"])

        keyword_results = self.evaluate_python_keywords(keyword.kwlist)
        logger.info("\n8. PYTHON KEYWORDS")
        logger.info("-" * 60)
        for result in keyword_results:
            logger.info(
                "{:<10}: {} tokens={}",
                result["keyword"],
                result["tokens"],
                result["num_tokens"],
            )

        logger.info("\n10. IDENTIFIER GENERALISATION")
        logger.info("-" * 60)
        unseen_identifiers = [
            "compute_normalised_attention_scores",
            "apply_multi_head_projection",
            "build_incremental_decoding_cache",
            "update_exponential_moving_average",
            "serialise_model_checkpoint_state",
        ]
        gen_results = self.evaluate_unseen_identifiers(unseen_identifiers)
        logger.info(
            "Unseen identifiers: avg_tokens={:.2f}, reuse_ratio={:.2f}",
            gen_results["avg_tokens"],
            gen_results["token_reuse_ratio"],
        )

        logger.info("\n10. BASELINE COMPARISON")
        logger.info("-" * 60)
        try:
            baseline_results = self.compare_with_baseline(code_samples, baseline_tokenizer)
            logger.info(
                "Baseline ({}): current_avg_len={:.2f}, baseline_avg_len={:.2f}, NSL={:.3f}, improvement={:.1f}%",  # noqa: E501
                baseline_tokenizer,
                baseline_results["current_avg_length"],
                baseline_results["baseline_avg_length"],
                baseline_results["nsl"],
                baseline_results["improvement"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping baseline comparison with {}: {}", baseline_tokenizer, exc)

        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
