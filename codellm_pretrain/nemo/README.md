
# Pretraining Code LLMs with NeMO framework


Write up:

Training an LLM from scratch using the custom tokenizer and dataset prepared in previous steps.

Custom tokenizer: https://dudeperf3ct.github.io/projects/train_llm_part1/
Dataset: [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)


> [!WARNING]
> I am using Lambda Labs GPU instance with 1xA100 GPU (40 GB SMX4) for running Step 1 and Step 2. It costs about **$1.48/hr**.


## Step 1: Streaming FIM data pipeline

- Dataset implementation: `dataset/swallow_fim_iterable_dataset.py`
- Sanity checker: `dataset/sanity_check_swallow_fim.py`


Example sanity run:

```bash
.venv/bin/python -m dataset.sanity_check_swallow_fim \
  --num-samples 2000 \
  --fim-rate 0.8 \
  --tokenizer dudeperf3ct/codellm-tokenizer \
  --validate
```

## Step 2: AutoModel config check

- Smoke config: `configs/pretrain_swallow_fim_stream_smoke.yaml`
- Full config template: `configs/pretrain_swallow_fim_stream_full.yaml`

Quick local config check (no `torchrun`, no full train):

```bash
.venv/bin/python - <<'PY'
from nemo_automodel.components.config.loader import load_yaml_config

cfg = load_yaml_config("configs/pretrain_swallow_fim_stream_smoke.yaml")
ds = cfg.dataset.instantiate()
batch = next(iter(ds))
print(sorted(batch.keys()), tuple(batch["input_ids"].shape), tuple(batch["labels"].shape))
PY
```

## Step 3: Evaluation

- Eval script: `eval/eval_fim_generate.py`
- Eval samples: `eval/eval_samples.jsonl`
- Gate checklist: `docs/fim_eval_checklist.md`
- Overfit configs:
  - `configs/pretrain_swallow_fim_overfit_single.yaml`
  - `configs/pretrain_swallow_fim_overfit_tiny.yaml`

Overfit gate commands (remote machine):

```bash
# Overfit single example
torchrun --nproc-per-node=1 -m nemo_automodel.recipes.llm.train_ft --config configs/pretrain_swallow_fim_overfit_single.yaml

# Overfit tiny subset
torchrun --nproc-per-node=1 -m nemo_automodel.recipes.llm.train_ft --config configs/pretrain_swallow_fim_overfit_tiny.yaml
```

Evaluate overfit checkpoints (run from `nemo/`):

```bash
# Evaluate single-example overfit checkpoint
.venv/bin/python -m eval.eval_fim_generate \
  --model checkpoints/swallow_fim_overfit_single/LATEST/model/consolidated \
  --tokenizer dudeperf3ct/codellm-tokenizer \
  --samples eval/eval_samples.jsonl \
  --max-new-tokens 128 \
  --temperature 0 \
  --stop-at-eot \
  --output-jsonl eval/results_overfit_single.jsonl

# Evaluate tiny-subset overfit checkpoint
.venv/bin/python -m eval.eval_fim_generate \
  --model checkpoints/swallow_fim_overfit_tiny/LATEST/model/consolidated \
  --tokenizer dudeperf3ct/codellm-tokenizer \
  --samples eval/eval_samples.jsonl \
  --max-new-tokens 128 \
  --temperature 0 \
  --stop-at-eot \
  --output-jsonl eval/results_overfit_tiny.jsonl
```

If `LATEST` is unavailable, replace it with the actual checkpoint step directory, for example:
`checkpoints/swallow_fim_overfit_single/epoch_0_step_300/model/consolidated`.

What to expect from evaluation before scale-up:

- `dataset.sanity_check_swallow_fim --validate` reports passing checks (`fim_rate`, `psm_spm_balance`, `format_layout`, `tokenizer_special_tokens`)
- single-example overfit run shows rapid loss collapse and structured FIM completions
- tiny-subset overfit run shows low/stable loss and better multi-sample reconstruction than single-example run
- `eval.eval_fim_generate` summary shows `expected_prefix_match_rate` trending upward from single -> tiny

## Step 4: Run full setup after gates pass

Recommended order on remote machine:

```bash
# 1) Smoke pretraining (stability check)
torchrun --nproc-per-node=4 -m nemo_automodel.recipes.llm.train_ft \
  --config configs/pretrain_swallow_fim_stream_smoke.yaml

# 2) Full pretraining
torchrun --nproc-per-node=4 -m nemo_automodel.recipes.llm.train_ft \
  --config configs/pretrain_swallow_fim_stream_full.yaml
```

Evaluate full-run checkpoint:

```bash
.venv/bin/python -m eval.eval_fim_generate \
  --model checkpoints/swallow_fim_full/LATEST/model/consolidated \
  --tokenizer dudeperf3ct/codellm-tokenizer \
  --samples eval/eval_samples.jsonl \
  --max-new-tokens 128 \
  --temperature 0 \
  --stop-at-eot \
  --output-jsonl eval/results_full.jsonl
```

If needed, swap `LATEST` with an explicit checkpoint path such as
`checkpoints/swallow_fim_full/epoch_0_step_5000/model/consolidated`.
