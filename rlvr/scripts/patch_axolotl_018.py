"""Apply the Axolotl 0.18.0 compatibility fixes required by this experiment."""

import argparse
import importlib.util
from pathlib import Path


def replace_once(source: str, old: str, new: str, description: str) -> str:
    if new in source:
        print(f"already applied: {description}")
        return source
    if source.count(old) != 1:
        raise RuntimeError(f"cannot apply {description}: expected source was not found once")
    print(f"applied: {description}")
    return source.replace(old, new, 1)


def package_root() -> Path:
    spec = importlib.util.find_spec("axolotl")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("Axolotl is not installed in the active environment")
    return Path(next(iter(spec.submodule_search_locations)))


def patch_vllm_serve(root: Path) -> None:
    path = root / "cli" / "vllm_serve.py"
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "        model=model,\n        tensor_parallel_size=tensor_parallel_size,",
        "        model=model,\n        revision=cfg.revision_of_model,\n"
        "        tensor_parallel_size=tensor_parallel_size,",
        "forward revision_of_model to vLLM",
    )
    path.write_text(source, encoding="utf-8")


def patch_lora_server(root: Path) -> None:
    path = root / "scripts" / "vllm_serve_lora.py"
    source = path.read_text(encoding="utf-8")
    current_import = "from trl.generation.vllm_generation import extract_logprobs"
    if current_import in source:
        print("already applied: import extract_logprobs from its TRL 1.8 location")
        return
    source = replace_once(
        source,
        "from trl.scripts.vllm_serve import extract_logprobs",
        current_import,
        "import extract_logprobs from its TRL 1.8 location",
    )
    path.write_text(source, encoding="utf-8")


def patch_vllm_sync(root: Path) -> None:
    path = root / "monkeypatch" / "trainer" / "trl_vllm.py"
    source = path.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "        is_fsdp_enabled = self.is_fsdp_enabled",
        "        is_fsdp_enabled = self._dist.is_fsdp",
        "use TRL 1.8 FSDP state during vLLM synchronization",
    )
    source = replace_once(
        source,
        "        deepspeed_plugin = accelerator.state.deepspeed_plugin\n"
        "        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3",
        "        zero_stage_3 = self._dist.is_zero3",
        "use TRL 1.8 ZeRO-3 state during vLLM synchronization",
    )
    path.write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        help="Axolotl package directory; defaults to the active environment",
    )
    args = parser.parse_args()
    root = args.root.resolve() if args.root else package_root()
    print(f"patching {root}")

    patch_vllm_serve(root)
    patch_lora_server(root)
    patch_vllm_sync(root)

    print("Axolotl compatibility patches are ready. Restart vLLM and training.")


if __name__ == "__main__":
    main()
