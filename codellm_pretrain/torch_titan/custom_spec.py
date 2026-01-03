# Custom TorchTitan train spec for overriding tokenizer vocab size.

from types import SimpleNamespace

import torchtitan.experiments.transformers_modeling_backend as base_backend
from torchtitan.experiments.transformers_modeling_backend.model.args import (
    HFTransformerModelArgs,
)
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

TRAIN_SPEC_NAME = "transformers_modeling_backend_custom"
FLAVOUR_NAME = "llama32_1b_tok32k"
VOCAB_SIZE = 32768


def _vocab_only_args(vocab_size: int) -> SimpleNamespace:
    return SimpleNamespace(vocab_size=vocab_size)


base_spec = base_backend.get_train_spec()
custom_model_args = dict(base_spec.model_args)
custom_model_args[FLAVOUR_NAME] = HFTransformerModelArgs(
    titan_dense_args=_vocab_only_args(VOCAB_SIZE)
)

custom_spec = TrainSpec(
    model_cls=base_spec.model_cls,
    model_args=custom_model_args,
    parallelize_fn=base_spec.parallelize_fn,
    pipelining_fn=base_spec.pipelining_fn,
    build_optimisers_fn=base_spec.build_optimisers_fn,
    build_lr_schedulers_fn=base_spec.build_lr_schedulers_fn,
    build_dataloader_fn=base_spec.build_dataloader_fn,
    build_tokenizer_fn=base_spec.build_tokenizer_fn,
    build_loss_fn=base_spec.build_loss_fn,
    build_validator_fn=base_spec.build_validator_fn,
    build_metrics_processor_fn=base_spec.build_metrics_processor_fn,
    state_dict_adapter=base_spec.state_dict_adapter,
)

register_train_spec(TRAIN_SPEC_NAME, custom_spec)
