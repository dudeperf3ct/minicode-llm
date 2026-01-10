# Eval Output Debugging Notes

## Symptoms Seen
- FIM completions look noisy or unrelated to the prefix/suffix.
- Model rarely emits EOS in short gaps; continues generating code-like noise.
- SPM outputs are worse than PSM.
- Outputs show odd spacing around punctuation/underscores.

## Likely Causes (ordered by likelihood)
- FIM exposure is sparse: `fim_rate=0.5` and PSM is ~25% of total samples.
- Short snippets rarely see FIM: `min_code_length=100` excludes small functions.
- Middle-length mismatch: training uses 10-50% of code as middle; tiny gaps are out-of-distribution.
- Undertrained: 9.8B tokens for 1B params is early for clean code generation.
- Sampling noise: high temperature/top-k hides the model's best guess.
- Byte-level BPE decoding: low-confidence token choices surface as extra spaces (e.g., `@ _ . append`).

## Fast Validations Before Long Runs
- Overfit a tiny dataset (1-5 files) with `fim_rate=1.0`; verify loss goes near zero and the model memorizes.
- Evaluate on a training sample and verify the exact snippet can be reconstructed.
- Run greedy decoding (`temperature=0`, `top_k=1`) to remove sampling noise.
- Confirm FIM tokens are present in prompts and that `<|endoftext|>` exists in the tokenizer.
- Use long FIM examples with a substantial missing middle (closer to training distribution).
- Use `--show_raw` or `--out` to confirm spacing is from model output, not display.
- Verify the model can do basic LM completion on a longer prompt (`--mode lm`) before judging FIM quality.
- Log the effective FIM rate in the dataloader (count FIM tokens per batch) to confirm the training mix.
- Check that the tokenizer vocab size matches the model config (custom vocab size should be consistent).

## Knobs That Most Affect FIM Behavior
- `fim_rate`: raise to 1.0 for FIM-only training.
- `min_code_length`: lower (e.g., 40) so short functions get FIM exposure.
- FIM format mix: reduce or disable middle-only; bias toward PSM.
- Sampling: use greedy for evaluation; keep `max_new_tokens` small for short gaps.
- Prompt length: use longer code blocks with meaningful missing middles to match training distribution.

## Recommended Debugging Sequence
1. Greedy LM eval (`--mode lm`) on a simple prompt to see base LM quality.
2. Greedy FIM eval on a long function with a large missing middle.
3. Overfit tiny dataset; confirm loss collapse and exact reconstruction.
4. Increase `fim_rate` and lower `min_code_length`, then re-run FIM eval.

## Interpreting "Bad" Outputs
- Code-like noise is typical early in training and for OOD FIM gaps.
- If LM outputs are also poor on long prompts, the model is still undertrained.
- If the model cannot overfit a tiny dataset, investigate the data pipeline, tokenizer, or checkpoint loading.
- Odd spacing generally reflects low-confidence decoding with a byte-level tokenizer; it is not a logging artifact.

## Why Are There Extra Spaces?
- This tokenizer is byte-level BPE, which represents word boundaries as explicit space tokens.
- When the model is uncertain, it emits fragmented tokens that decode into spaces around punctuation or underscores (e.g., `@ _ . append`).
- This is a model-quality symptom, not a tokenizer bug or display issue.
- You can post-process output for readability, but the real fix is better training signal or longer training.

## Takeaways
- The tokenizer is not broken; spacing artifacts come from low-confidence byte-level decoding.
- Model quality is the bottleneck; 9.8B tokens is early for clean code and FIM behavior.
- FIM quality depends heavily on training distribution (rate, gap size, and code length).

## Pre-Run Checklist
- Overfit a tiny dataset (1-5 files) with `fim_rate=1.0` and confirm near-zero loss.
- Run an LM sanity check on a longer prompt (`--mode lm`) to confirm basic completion.
- Run a FIM sanity check on a long snippet with a large missing middle.
- Verify the effective FIM rate in the dataloader (count FIM tokens per batch).
- Use greedy decoding (`temperature=0`, `top_k=1`) for evaluation.
