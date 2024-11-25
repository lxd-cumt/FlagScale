import sys
import os


def load_args_hf2mg(args):
    pass


def save_args_mg2hf(args):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from examples.emu3.emu3_moe.configuration_llama import LlamaConfig

    LlamaConfig.model_type = "emu3"
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        language_vocab_size=args.true_language_vocab_size,
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
        hidden_dropout=args.hidden_dropout,
        add_qkv_bias=args.add_qkv_bias,
        qk_layernorm=args.qk_layernorm,
        torch_dtype=args.params_dtype,
        num_experts_per_tok = args.moe_router_topk if not args.multimodal_num_experts_split else args.multimodal_moe_router_topk,
        num_local_experts = args.num_experts,
        router_aux_loss_coef = args.moe_aux_loss_coeff if not args.multimodal_num_experts_split else args.multimodal_moe_aux_loss_coeff,
        use_multimodal_router_attn = args.use_multimodal_router_attn,
        use_multimodal_router_mlp = args.use_multimodal_router_mlp,
        multimodal_num_experts_split = args.multimodal_num_experts_split
    )
    config.architectures = ["LlamaForCausalLM"]
    auto_map = dict()
    auto_map['AutoConfig'] = 'configuration_llama.LlamaConfig'
    auto_map['AutoModelForCausalLM'] = 'modeling_llama.LlamaForCausalLM'
    config.auto_map = auto_map
    config.save_pretrained(args.save)

    return config