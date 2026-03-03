# Pretraining Code LLMs with NeMO framework

For this pretraining recipe, we will use Nemo Automodel project.

[Torchtitan](../torchtitan/README.md) is another library used for pretraining LLM.

Write up:

Training an LLM from scratch using the custom tokenizer and dataset prepared in previous steps.

Custom tokenizer: https://dudeperf3ct.github.io/projects/train_llm_part1/

Dataset: [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)

Model Architecture: Llama 3.2 1B (1 billion parameter)


> [!WARNING]
> I am using Lambda Labs GPU instance with 1xA100 GPU (40 GB SMX4) for running Step 1 and Step 2. It costs about **$1.48/hr**.

## Dependencies setup

One remote machine, set up the environment and authenticate with Hugging Face and Weights & Biases:

```bash
uv sync
source .venv/bin/activate
hf auth login
wandb login
```

## Step 1: Data/Tokenizer sanity check

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

## Step 3: Overfitting and evaluation sanity check

- Eval script: `eval/eval_fim_generate.py`
- Eval samples: `eval/eval_samples.jsonl`
- Gate checklist: `docs/fim_eval_checklist.md`
- Overfit configs:
  - `configs/pretrain_swallow_fim_overfit_single.yaml`
  - `configs/pretrain_swallow_fim_overfit_tiny.yaml`

Overfit single and tiny subset:

```bash
# Reduce HF/httpx network log noise
export HF_HUB_DISABLE_XET=1

# Overfit single example
torchrun --nproc-per-node=1 -m nemo_automodel.recipes.llm.train_ft --config configs/pretrain_swallow_fim_overfit_single.yaml

# Overfit tiny subset
torchrun --nproc-per-node=1 -m nemo_automodel.recipes.llm.train_ft --config configs/pretrain_swallow_fim_overfit_tiny.yaml
```

Quiet log entrypoint (suppresses noisy `httpx/httpcore/huggingface_hub` INFO logs):

```bash
mkdir -p logs
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
PYTHONUNBUFFERED=1 torchrun --nproc-per-node=2 -m nemo_automodel.recipes.llm.train_ft \
  --config configs/pretrain_swallow_fim_overfit_single.yaml \
  2>&1 | tee "logs/overfit_single_$(date +%Y%m%d_%H%M%S).log"
```

Make sure to run a rclone sync of the `checkpoints/` directory to your local machine after the overfit runs to access the checkpoint files for evaluation. Here's an example `rclone` command:

```bash
 rclone -P --stats=1s --transfers 8 --checkers 16 sync \
  ":sftp,host=<host>,user=<remote-address>" \
  ./checkpoints
```

Evaluate overfit checkpoints:

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

## Step 4: Run smoke test and full pretraining


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
