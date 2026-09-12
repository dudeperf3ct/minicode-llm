# Axolotl 0.18.0 Compatibility Patches

The RLVR environment combines Axolotl 0.18.0 with TRL 1.8.0. Four small
compatibility fixes were needed while bringing up the two-H100 experiment.
They modify the installed Axolotl package and are lost whenever Axolotl is
reinstalled.

## Apply on the GPU server

Activate the RLVR environment and run the checked-in patcher:

```bash
cd rlvr
source .venv/bin/activate
python scripts/patch_axolotl_018.py
```

The script is idempotent. Run it again after every Axolotl reinstall, then
restart both the vLLM server and trainer. Confirm the important changes:

```bash
python - <<'PY'
from pathlib import Path

import axolotl

root = Path(axolotl.__file__).parent
serve = (root / "cli" / "vllm_serve.py").read_text()
merge = (root / "cli" / "merge_lora.py").read_text()
merge_utils = (root / "cli" / "utils" / "lora_merge.py").read_text()
sync = (root / "monkeypatch" / "trainer" / "trl_vllm.py").read_text()
assert "revision=cfg.revision_of_model" in serve
assert "revision=cfg.revision_of_model" in merge
assert "snapshot_download(str(base_model_path), revision=revision)" in merge_utils
assert "self._dist.is_fsdp" in sync
assert "self._dist.is_zero3" in sync
print("Axolotl compatibility patches verified")
PY
```

## Patch inventory

### 1. Forward the model revision

**Symptom:** `axolotl vllm-serve` loads the Hugging Face `main` revision even
when `revision_of_model` is pinned in the config.

**Change:** pass `revision=cfg.revision_of_model` when Axolotl constructs the
vLLM server arguments.

**Upstream status:** fixed on the fork branch
`dudeperf3ct/axolotl:fix/vllm-serve-model-revision` at commit `938036cc5`.

### 2. Import `extract_logprobs` from TRL 1.8

**Symptom:** the native-LoRA server fails during import:

```text
ImportError: cannot import name 'extract_logprobs' from 'trl.scripts.vllm_serve'
```

**Change:** import it from `trl.generation.vllm_generation`, with the older
location retained only as a fallback. Some Axolotl 0.18.0 builds already
contain this fix; the patcher leaves those installations unchanged.

### 3. Use TRL's distributed backend

**Symptom:** weight synchronization fails with:

```text
AttributeError: 'VLLMGeneration' object has no attribute 'is_fsdp_enabled'
```

**Cause:** TRL 1.8 removed `VLLMGeneration.is_fsdp_enabled` and introduced its
`DistributedBackend` wrapper.

**Change:** use `self._dist.is_fsdp` and `self._dist.is_zero3` inside Axolotl's
patched `sync_weights` method.

### 4. Preserve the revision during efficient LoRA merging

**Symptom:** `axolotl merge-lora` downloads the default Hugging Face revision
and then reports that no model shards were found, even though
`revision_of_model` points to a valid branch or commit containing the weights.

**Change:** forward `cfg.revision_of_model` to the efficient merge helper and
pass it to `snapshot_download`. This keeps merging on the same immutable SFT
snapshot used by training and vLLM.

## Async prefetch

The experiment configs deliberately keep `async_prefetch: false`. This guide
does not patch Axolotl's stale `AsyncGRPOTrainer` method signature. If a
`num_tiles` argument error still occurs, `use_data_producer: true` is selecting
that trainer even though background prefetch is disabled; turning prefetch off
alone does not repair the underlying Axolotl and TRL incompatibility.

## Project-level fixes

These changes live in this repository and do not modify `site-packages`:

- `rlvr.verifier` uses Modal's current Sandbox filesystem API.
- Modal transport `TimeoutError` and `ConnectionError` failures are retried.
- Persistent Modal infrastructure failures abort training rather than being
  recorded as incorrect model completions.

## Remove the patches

Reinstalling Axolotl restores the packaged files:

```bash
uv pip install --reinstall --no-build-isolation 'axolotl==0.18.0'
```

Reapply `xformers`, Modal, vLLM, and the compatibility patcher afterward using
the order documented in the main RLVR README.
