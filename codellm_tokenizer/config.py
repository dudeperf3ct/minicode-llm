"""Configuration for training tokenizer."""

from pydantic import BaseModel, ConfigDict


class DatasetConfig(BaseModel):
    """Dataset-related settings."""

    model_config = ConfigDict(extra="forbid")

    hf_path: str
    subset: str | None = None
    split: str
    text_field: str
    streaming: bool
    shuffle: bool
    shuffle_buffer: int
    seed: int
    max_samples: int | None = None


class TrainingConfig(BaseModel):
    """Tokenizer training hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    vocab_size: int
    min_frequency: int


class OutputConfig(BaseModel):
    """Output locations and publishing flags."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str
    push_to_hf_hub: bool | None = None
    hub_repo_id: str | None = None
    hub_private: bool = False


class TokenizerConfig(BaseModel):
    """Top-level configuration, loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    dataset: DatasetConfig
    training: TrainingConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, data: dict) -> "TokenizerConfig":
        """Build config directly from the nested YAML mapping."""
        return cls.model_validate(data)
