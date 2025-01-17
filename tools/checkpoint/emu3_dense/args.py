import os
import sys
import json


def load_args_hf2mg(args):
    pass


def save_args_mg2hf(args):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from examples.emu.emu3_dense.configuration_emu3 import Emu3Config

    config = Emu3Config(
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
        pretraining_tp=1,
        tie_word_embeddings=(not args.untie_embeddings_and_output_weights),
        rope_theta=args.rotary_base,
        attention_dropout=args.attention_dropout,
        torch_dtype=args.params_dtype,
    )
    config.architectures = ["Emu3ForCausalLM"]
    auto_map = dict()
    auto_map['AutoConfig'] = 'configuration_emu3.Emu3Config'
    auto_map['AutoModelForCausalLM'] = 'modeling_emu3.Emu3ForCausalLM'
    config.auto_map = auto_map
    config.save_pretrained(args.save)

    return config
