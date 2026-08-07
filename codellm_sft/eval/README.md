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

For fair comparison with the completed base and SFT evaluations, the `direct`
profile retains the original EvalPlus protocol: greedy decoding, a 768-token
generation limit, and EvalPlus's 2,048-token embedded-vLLM context limit [^1].
The separate `thinking` profile follows the Qwen3.5 thinking-mode recommendation
with temperature `0.6` and a 32,768-token generation limit [^2]. Thinking scores
are a higher-compute capability reference rather than an equal-compute comparison.

## Running evaluation benchmark

To run evaluation, we use `vllm` to host the model for local inference as backend.

On a `1 x A100` machine, run the following command for evaluation. The results are stored under `results` folder.

### Evalplus

Install the library,

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --upgrade \
  "evalplus[vllm] @ git+https://github.com/evalplus/evalplus.git@26d6d00bb1fd0fa37f39c99d5290da67891d1c5e"
```

`run_evalplus.sh` runs both HumanEval and MBPP. Existing two-argument commands
default to the `direct` profile and remain compatible with previously generated
results. Each result directory receives a `protocol.json`; the runner refuses to
reuse that directory with incompatible settings.

Running the HumanEval and MBPP benchmarks,

Base Qwen model (about 30 mins)

```bash
./run_evalplus.sh \
  Qwen/Qwen3.5-4B-Base \
  results/qwen3.5-4b-base/evalplus
```

Direct SFT model

```bash
./run_evalplus.sh \
  /path/to/qwen3.5-4b-kodcode-sft \
  results/qwen3.5-4b-direct-fft/evalplus
```

An explicit generation limit can be supplied without changing the profile:

```bash
./run_evalplus.sh \
  /path/to/model \
  results/model/evalplus \
  --profile direct \
  --max-new-tokens 768
```

### Post-trained direct mode

The post-trained checkpoint requires an OpenAI-compatible vLLM server so that
thinking can be explicitly disabled. Start the server in one terminal:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen35-4b-post-direct \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --language-model-only \
  --generation-config auto \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90
```

Run the matched direct evaluation in another terminal:

```bash
OPENAI_API_KEY=EMPTY ./run_evalplus.sh \
  qwen35-4b-post-direct \
  results/qwen3.5-4b-post-trained/direct/evalplus \
  --profile direct \
  --base-url http://127.0.0.1:8000/v1
```

### Post-trained thinking mode

Start a fresh server with thinking enabled and sufficient context:

```bash
vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen35-4b-post-thinking \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --language-model-only \
  --generation-config auto \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90
```

Run the thinking evaluation in another terminal:

```bash
OPENAI_API_KEY=EMPTY ./run_evalplus.sh \
  qwen35-4b-post-thinking \
  results/qwen3.5-4b-post-trained/thinking/evalplus \
  --profile thinking \
  --base-url http://127.0.0.1:8000/v1
```

> [!WARNING]
> EvalPlus's embedded vLLM backend remains restricted to a 2,048-token total
> context [^1]. The thinking profile therefore requires a separately configured
> vLLM server. Do not reuse a result directory between profiles.

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

[^1]: EvalPlus's decoder [defaults generation to 768 tokens](https://github.com/evalplus/evalplus/blob/master/evalplus/provider/base.py#L11), while its embedded vLLM provider [hardcodes the maximum model length to 2,048](https://github.com/evalplus/evalplus/blob/master/evalplus/provider/vllm.py#L41).
[^2]: The [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B) recommends `temperature=0.6`, `top_p=0.95`, and `top_k=20` for precise coding tasks in thinking mode, with 32,768 output tokens for most queries.
