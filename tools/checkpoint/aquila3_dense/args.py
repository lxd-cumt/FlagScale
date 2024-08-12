import os
import sys
import json


def load_args_hf2mg(args):

    # Read transformers args.
    hf_args_path = os.path.join(args.load, "config.json")
    with open(hf_args_path) as f:
        hf_args = json.load(f)

    # Update Megatron args.
    args.attention_dropout = hf_args["attention_dropout"]
    args.hidden_dropout = hf_args["attention_dropout"]
    args.hidden_size = hf_args["hidden_size"]
    args.add_qkv_bias = hf_args.get("attention_bias", True)
    args.swiglu = hf_args["hidden_act"] == "silu"
    args.init_method_std = hf_args["initializer_range"]
    args.ffn_hidden_size = hf_args["intermediate_size"]
    args.max_position_embeddings = hf_args["max_position_embeddings"]
    args.model_type = hf_args["model_type"]
    args.num_attention_heads = hf_args["num_attention_heads"]
    args.num_layers = hf_args["num_hidden_layers"]
    args.num_query_groups = hf_args["num_key_value_heads"]
    args.norm_epsilon = hf_args["rms_norm_eps"]
    args.rotary_base = hf_args["rope_theta"]
    args.untie_embeddings_and_output_weights = not hf_args["tie_word_embeddings"]
    args.bf16 = hf_args["torch_dtype"] == "bfloat16"
    args.fp16 = hf_args["torch_dtype"] == "float16"
    args.vocab_size = hf_args["vocab_size"]
    args.padded_vocab_size = hf_args["vocab_size"]

    args.seq_length = 2048
    args.global_batch_size = 1024
    args.iteration = 1 # '0', 'release' don't work
    args.add_position_embedding = False
    args.group_query_attention = True
    args.normalization = "RMSNorm"
    args.use_rotary_position_embeddings = True
    args.add_bias_linear = False
    args.make_vocab_size_divisible_by = 64
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.norm_has_bias = False

    return args


def save_args_mg2hf(args):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from examples.aquila.aquila3_dense.configuration_llama import LlamaConfig

    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.ffn_hidden_size,
        num_hidden_layers=args.encoder_num_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_query_groups,
        hidden_act="silu",
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=args.init_method_std,
        rms_norm_eps=args.norm_epsilon,
        use_cache=True,
        pad_token_id=151643,
        bos_token_id=151849,
        eos_token_id=151850,
        pretraining_tp=1,
        tie_word_embeddings=(not args.untie_embeddings_and_output_weights),
        rope_theta=args.rotary_base,
        attention_dropout=args.attention_dropout,
        torch_dtype=args.params_dtype,
    )
    config.architectures = ["LlamaForCausalLM"]
    auto_map = dict()
    auto_map['AutoConfig'] = 'configuration_llama.LlamaConfig'
    auto_map['AutoModelForCausalLM'] = 'modeling_llama.LlamaForCausalLM'
    config.auto_map = auto_map
    config.save_pretrained(args.save)

    return config
