# Step 3 FIM Evaluation Checklist

Use this checklist before launching long remote training runs.

## 1) Data/Tokenizer sanity
```bash
.venv/bin/python -m dataset.sanity_check_swallow_fim \
  --num-samples 2000 \
  --shuffle-buffer-size 10000 \
  --tokenizer dudeperf3ct/codellm-tokenizer \
  --validate
```

Expect all checks to pass, especially:

- `fim_rate` close to target
- balanced `psm_spm_balance`
- `format_layout.pass=true`
- `tokenizer_special_tokens.pass=true`

## 2) Overfit single example (remote run)

```bash
torchrun --nproc-per-node=1 -m nemo_automodel.recipes.llm.train_ft \
  --config configs/pretrain_swallow_fim_overfit_single.yaml
```

Expected signal:

- train loss should rapidly drop toward near-zero
- generated FIM continuations should closely match the seen pattern

## 3) Overfit tiny subset (remote run)

```bash
torchrun --nproc-per-node=1 -m nemo_automodel.recipes.llm.train_ft \
  --config configs/pretrain_swallow_fim_overfit_tiny.yaml
```

Expected signal:

- loss continues dropping and stabilizes low
- model can reconstruct training-style infills for multiple samples

## 4) Greedy LM/FIM generation gate

Run eval with deterministic decoding (`temperature=0`):

```bash
.venv/bin/python -m eval.eval_fim_generate \
  --model <hf_or_local_consolidated_model_path> \
  --tokenizer dudeperf3ct/codellm-tokenizer \
  --samples eval/eval_samples.jsonl \
  --max-new-tokens 128 \
  --temperature 0 \
  --stop-at-eot \
  --output-jsonl eval/results_eval_fim_generate.jsonl
```

Expected signal:

- FIM samples: sensible infill between prefix/suffix for both PSM and SPM
- LM samples: coherent continuation for simple function tasks
- summary `expected_prefix_match_rate` improves as training progresses

## 5) Smoke train readiness

Proceed to smoke training (`configs/pretrain_swallow_fim_stream_smoke.yaml`) only after:

- sections 1-4 are healthy
- no tokenizer mismatch issues
- no dataset format errors
