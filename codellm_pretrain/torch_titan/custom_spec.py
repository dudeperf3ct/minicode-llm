# Custom TorchTitan train spec for overriding tokenizer vocab size.

import torchtitan.experiments.transformers_modeling_backend as base_backend
from torchtitan.experiments.transformers_modeling_backend.model.args import (
    HFTransformerModelArgs,
    TitanDenseModelArgs,
)
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

TRAIN_SPEC_NAME = "transformers_modeling_backend_custom"
FLAVOUR_NAME = "llama32_1b_tok32k"
VOCAB_SIZE = 32768


def _vocab_only_args(vocab_size: int) -> TitanDenseModelArgs:
    # Let HF config define core shape params; only override vocab size (and MLP width).
    args = TitanDenseModelArgs()
    for attr in (
        "dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "norm_eps",
        "rope_theta",
        "max_seq_len",
    ):
        setattr(args, attr, None)
    args.vocab_size = vocab_size
    # Llama 3.2 1B uses intermediate_size=4*hidden_size; this matches via 1.5x on 2/3*4.
    args.ffn_dim_multiplier = 1.5
    return args


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
    build_optimizers_fn=base_spec.build_optimizers_fn,
    build_lr_schedulers_fn=base_spec.build_lr_schedulers_fn,
    build_dataloader_fn=base_spec.build_dataloader_fn,
    build_tokenizer_fn=base_spec.build_tokenizer_fn,
    build_loss_fn=base_spec.build_loss_fn,
    build_validator_fn=base_spec.build_validator_fn,
    build_metrics_processor_fn=base_spec.build_metrics_processor_fn,
    state_dict_adapter=base_spec.state_dict_adapter,
)

register_train_spec(TRAIN_SPEC_NAME, custom_spec)


"""
>>> cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
>>> print(cfg)
LlamaConfig {
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "dtype": "bfloat16",
  "eos_token_id": 128001,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 32.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "transformers_version": "4.57.3",
  "use_cache": true,
  "vocab_size": 128256
}

>>> print("hidden_size", cfg.hidden_size)
hidden_size 2048
>>> print("num_hidden_layers", cfg.num_hidden_layers)
num_hidden_layers 16
>>> print("num_attention_heads", cfg.num_attention_heads)
num_attention_heads 32
"""
