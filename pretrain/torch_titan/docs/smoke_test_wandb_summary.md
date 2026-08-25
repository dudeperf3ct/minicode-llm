# Smoke Test W&B Summary (98M tokens)

- Source: W&B group "Smoke run - 98M tokens", rank0 run (`ez8idg0a`).
- Metrics are logged per-rank; aggregate throughput uses `data_parallel_replicate_degree=2`.

> [!NOTE]
> W&B plots for experiment: [Plots](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/workspace?nw=nwuserdudeperf3ct), [Logs](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/logs), [Summary](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/overview) and [Report](https://wandb.ai/dudeperf3ct/torchtitan/reports/Pretraining-LLM-experiment--VmlldzoxNTU4NTA1NQ)

## Run Overview
- Model: Llama 3.2 1B, custom tokenizer (32k), dataset SwallowCode v2
- Config: `seq_len=8192`, `local_batch_size=6`, `global_batch_size=12`, `steps=1,000`
- Hardware: 2 x H100 80GB (SMX5)
- Total tokens seen: 98,304,000 (98.3M)
- Runtime: 924 s (15.4 min)

## Training Dynamics (from plots)
- Loss: global avg loss fell from 10.88 at step 1 to 4.69 at step 1k, with a steady downward trend.
- LR: warmup from 3e-7 to 3e-4 by step 1k; no decay during this smoke run.
- Grad norm: mean ~51, median ~2.16, rare spikes up to 600; no instability or OOMs observed.

## Throughput and Efficiency
- Step time: ~0.90 s after the first ~100 steps.
- Throughput per rank: ~54.4k tokens/s average.
- Global throughput: ~109k tokens/s (98,304 tokens/step / 0.90 s), ~3,980 steps/hour.
- MFU: ~41% average.
- TFLOPS: ~405 TFLOPS.

## Memory and Input Pipeline
- Peak active memory: 70.17 GiB (88.6% of 80GB).
- Peak reserved memory: 72.82 GiB (92.0% of 80GB).
- Memory stayed flat after warmup; 0 OOMs and 0 alloc retries.
- Data loading overhead: ~0.03% of step time (~0.00028 s/step).

## Notes
- W&B plots are per-rank; multiply throughput by 2 to estimate total throughput.
- Tokens per step: `12 * 8192 = 98,304`, matching the 98.3M total tokens at 1k steps.
