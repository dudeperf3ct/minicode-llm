# Full Training W&B Summary (9.8B tokens)

- Source: W&B group "Full run - 9.8B tokens", rank0 run (`d1t24qnh`).
- Metrics are logged per-rank; aggregate throughput uses `data_parallel_replicate_degree=4`.

> [!NOTE]
> W&B plots for experiment: [Plots](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/workspace), [Logs](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/logs), [Summary](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/overview) and [Report](https://wandb.ai/dudeperf3ct/torchtitan/reports/Pretraining-LLM-experiment--VmlldzoxNTU4NTA1NQ)

## Run Overview

- Model: Llama 3.2 1B, custom tokenizer (32k), dataset SwallowCode v2
- Config: `seq_len=8192`, `local_batch_size=6`, `global_batch_size=24`, `steps=50,000`
- Hardware: 4 x H100 80GB (SMX5)
- Total tokens seen: 9,830,400,000 (9.83B)
- Runtime: 44,127 s (12.26 h), cost approx $151.5 at $12.36/hr

## Training Dynamics (from plots)

- Loss: global avg loss fell from 10.86 at step 1 to 2.91 at step 50k. The last 10% of steps averaged ~3.02. Loss shows occasional spikes (p90 ~4.28, p99 ~6.51) but a clear downward trend overall.
- LR: warmup to 3e-4 by step 800, then cosine decay to ~3.06e-13 at step 50k.
- Grad norm: median ~1.76k, p90 ~11.3k, rare spikes up to 622k. Despite spikes, loss stayed stable and no OOMs occurred.

## Throughput and Efficiency

- Step time: median 0.880 s after warmup (p90 ~0.886 s).
- Throughput per rank: 55.7k tokens/s average (p10 ~55.5k, p90 ~55.9k).
- Global throughput: ~223k tokens/s (196,608 tokens/step / 0.88 s), roughly 4,090 steps/hour.
- MFU: 42.06% average, steady after warmup.
- TFLOPS: ~416 TFLOPS.

## Memory and Input Pipeline

- Peak active memory: 70.17 GiB (88.6% of 80GB).
- Peak reserved memory: 70.85 GiB (89.5% of 80GB).
- Memory stayed flat after warmup; 0 OOMs and 0 alloc retries.
- Data loading overhead: ~0.03% of step time (~0.00027 s/step), not a bottleneck.

## Notes

- The first ~1k steps include warmup and compile overhead; metrics stabilize after that window.
- W&B plots are per-rank; multiply throughput by 4 to estimate total throughput.
- Tokens per step: `24 * 8192 = 196,608`, matching the 9.83B total tokens at 50k steps.
