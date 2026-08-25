# Label Audit

This directory receives the direct and reasoning JSON reports generated after
Axolotl preprocesses the fixed 32-example datasets. No report is committed
until it was produced by the pinned H100 environment.

Run the commands from `sft/`:

```bash
axolotl preprocess configs/audit/direct.yml \
  --debug \
  --debug-num-examples 5

uv run python scripts/audit_labels.py configs/audit/direct.yml
```

```bash
axolotl preprocess configs/audit/reasoning.yml \
  --debug \
  --debug-num-examples 5

uv run python scripts/audit_labels.py configs/audit/reasoning.yml
```

The first command in each pair exposes Axolotl's debug rendering for manual
inspection. The second independently checks every prepared row and writes
`direct.json` or `reasoning.json`.

Qwen's official template renders an empty `<think></think>` block in direct
assistant turns. Axolotl masks that template block while training the code and
assistant EOT. The reasoning audit separately requires the complete source
reasoning to be trainable.

Delete only the corresponding `prepared/label-audit/<variant>/` directory
before intentionally regenerating an audit with changed inputs or settings.
