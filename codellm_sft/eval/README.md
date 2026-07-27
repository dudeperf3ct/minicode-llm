# Evaluation

We are performing SFT on [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base) base model. To study what SFT changes, we evaluate the following checkpoints:

* `Qwen/Qwen3.5-4B-Base`
* `Qwen/Qwen3.5-4B-Base` after KodCode SFT
* `Qwen/Qwen3.5-4B` used as a post-trained reference

To measure the effectiveness of the training on Python coding questions, we will benchmark both the base model and trained SFT model on the following benchmarks.

1. LiveCodeBench (3 modes - Easy, Medium, Hard variants) version 5
2. HumanEval (Base and Plus versions)
3. MBPP (Base and Plus variants)

To run evaluation benchmark, we will use [evalplus](https://github.com/evalplus/evalplus) library for HumanEval and MBPP and [Skythoughts-Eval](https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/evals) for LiveCodeBench.

Hardware: 1 x A100 40 GB SXM4 ($1.99/hr July 2026 on Lambda Labs)

For fair evaluation, we will use greedy decoding, official Qwen-3.5 model chat template and 16,384 maximum generation length [^1].

## Running evaluation benchmark

To run evaluation, we use `vllm` to host the model for local inference as backend.

On a `1 x A100` machine, run the following command for evaluation. The results are stored under `results` folder.

### Evalplus

Install the library,

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --upgrade "evalplus[vllm] @ git+https://github.com/evalplus/evalplus.git"
```

Running the HumanEval and MBPP benchmarks,

Base Qwen model (about 30 mins)

```bash
./run_evalplus.sh \
  Qwen/Qwen3.5-4B-Base \
  results/qwen3.5-4b-base/evalplus
```

Post trained Qwen model (about 30 mins)

```bash
./run_evalplus.sh \
  "Qwen/Qwen3.5-4B" \
  "results/qwen3.5-4b-post-trained/evalplus"
```

Change the `MODEL` to `"/path/to/qwen3.5-4b-kodcode-sft"` to run evaluations on for SFT models.

> [!WARNING]
> HumanEval and MBPP inference will be restricted to a 2048-token total context [^1].

### SkyThought Evals

The current SkyThought LiveCodeBench task files use `release_v2` and pins vllm to `0.7.0`. I have created a fork that upgrades the vllm to `0.25.1` and uses `release_v5` for livecodebench yaml files.

Install the library

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install "skythought @ git+https://github.com/dudeperf3ct/SkyThought.git@feat/qwen35-livecodebench-v5"
```

Run the LiveCodeBench benchmark,

Base Qwen model (about 30 mins)

```bash
./run_skythought.sh \
  Qwen/Qwen3.5-4B-Base \
  results/qwen3.5-4b-base/livecodebench
```

Post trained Qwen model (about 30 mins)

```bash
./run_skythought.sh \
  "Qwen/Qwen3.5-4B" \
  "results/qwen3.5-4b-post-trained/livecodebench"
```


There are multiple backends supported in the [official guide](https://github.com/NovaSky-AI/SkyThought/tree/main/skythought/evals). For example, `ray` backend on top of `vllm` is recommended for high throughput.

[^1] : `evalplus` library [hardcodes](https://github.com/evalplus/evalplus/blob/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e/evalplus/provider/vllm.py#L45) `vllm` maximum model length to 2048.
