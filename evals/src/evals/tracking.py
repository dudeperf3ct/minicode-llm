"""Log evaluation results to an existing W&B training run.

Axolotl owns training-time W&B logging. Evaluation reopens that exact run by
its stable ID, adds summary metrics, and attaches the detailed result files as
an evaluation artifact.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb


@dataclass(frozen=True)
class WandbRun:
    entity: str
    project: str
    name: str
    run_id: str

    @classmethod
    def from_axolotl_config(cls, config: dict[str, Any]) -> "WandbRun":
        return cls(
            entity=config["wandb_entity"],
            project=config["wandb_project"],
            name=config["wandb_name"],
            run_id=config["wandb_run_id"],
        )


def log_evaluation(
    config: WandbRun, summary: dict[str, Any], results_path: Path, summary_path: Path
) -> None:
    with wandb.init(
        entity=config.entity, project=config.project, id=config.run_id, resume="must"
    ) as run:
        run.summary.update(
            {
                "held_out_test": summary,
                "test_pass_rate": summary["pass_rate"],
                "test_average_output_tokens": summary["average_output_tokens"],
            }
        )
        artifact = wandb.Artifact(
            name=f"{config.run_id}-held-out-test", type="evaluation", metadata=summary
        )
        artifact.add_file(str(results_path))
        artifact.add_file(str(summary_path))
        run.log_artifact(artifact, aliases=["held-out-test"])
