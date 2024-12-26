import sys
import copy
import torch
sys.path.append("..")
from utils import padding_vocab_size
from aquila3_dense.ckpt import (
    get_hf_attn_ckpt,
    get_hf_mlp_ckpt,
    set_hf_attn_ckpt,
    set_hf_mlp_ckpt,
    set_hf_final_norm_ckpt,
    get_mlp_ckpt,
    get_final_norm_ckpt,
    get_output_layer_ckpt,
    set_mlp_ckpt,
    set_final_norm_ckpt,
)


def set_hf_embedding_ckpt(message, model, md, args):
    print(f"Warning: set_hf_embedding will change embedding shape-0 to true_vocab_size[{md.true_vocab_size}] .")

    orig_word_embed = message.pop("word embeddings")
    full_word_embed = padding_vocab_size(orig_word_embed, md, args, "language_padded_vocab_size")
    full_word_embed = full_word_embed[:md.true_language_vocab_size, :]

    orig_vision_embed = message.pop("vision embeddings")
    full_vision_embed = padding_vocab_size(orig_vision_embed, md, args, "vision_padded_vocab_size")
    true_vision_vocab_size = md.true_vocab_size - md.true_language_vocab_size
    full_vision_embed = full_vision_embed[:true_vision_vocab_size, :]

    model.model.embed_tokens.weight.data.copy_(torch.cat([full_word_embed, full_vision_embed], dim=0))


def set_hf_output_layer_ckpt(message, model, md, args):
    print(f"Warning: set_hf_output_layer will change output_layer shape-0 to true_vocab_size[{md.true_vocab_size}] .")
    orig_output_layer_weight = message.pop("weight")
    full_output_layer_weight = padding_vocab_size(orig_output_layer_weight, md, args)[:md.true_vocab_size, :]
    model.lm_head.weight.data.copy_(full_output_layer_weight)


def _get_parallel_size(args):
    return args.tensor_model_parallel_size, \
        args.pipeline_model_parallel_size, \
        args.expert_model_parallel_size, \
        args.virtual_pipeline_model_parallel_size or 1


def get_embedding_ckpt(message, models, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    word_embeddings = []
    vision_embeddings = []
    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)
        word_embeddings.append(model.embedding.word_embeddings.weight.data)
        vision_embeddings.append(model.embedding.vision_embeddings.weight.data)
    message["word embeddings"] = torch.cat(word_embeddings, dim=0)
    message["vision embeddings"] = torch.cat(vision_embeddings, dim=0)
    if args.position_embedding_type == 'learned_absolute':
        message["position embeddings"] = models[0].embedding.position_embeddings.weight.data
    else:
        assert not hasattr(models[0].embedding, 'position_embeddings')


def get_attn_ckpt(message, models, layer_id, args):
    tp_size, _, ep_size, _ = _get_parallel_size(args)

    # parallel tensor
    qkv_weight = []
    qkv_bias = []
    proj_weight = []
    # non-parallel tensor
    proj_bias = None
    input_norm_weight = None
    input_norm_bias = None
    post_norm_weight = None
    post_norm_bias = None

    assert len(models) == tp_size * ep_size
    complete_tp_ranks = []
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        if tp_rank in complete_tp_ranks:
            continue
        complete_tp_ranks.append(tp_rank)

        tf_layer = model.decoder.layers[layer_id]
        # weight
        qkv_weight.append(tf_layer.self_attention.linear_qkv.weight.data)
        proj_weight.append(tf_layer.self_attention.linear_proj.weight.data)
        input_norm_weight = tf_layer.input_layernorm.weight.data
        post_norm_weight = tf_layer.pre_mlp_layernorm.weight.data
        # bias
        if args.norm_has_bias:
            input_norm_bias = tf_layer.input_layernorm.bias.data
            post_norm_bias = tf_layer.pre_mlp_layernorm.bias.data
        if args.add_qkv_bias or args.add_bias_linear:
            qkv_bias.append(tf_layer.self_attention.linear_qkv.bias.data)
        if args.add_bias_linear:
            proj_bias = tf_layer.self_attention.linear_proj.bias.data

    # weight
    message["qkv weight"] = torch.cat(qkv_weight, dim=0)
    message["proj weight"] = torch.cat(proj_weight, dim=1)
    message["input norm weight"] = input_norm_weight
    message["post norm weight"] = post_norm_weight
    # bias
    if args.norm_has_bias:
        message["input norm bias"] = input_norm_bias
        message["post norm bias"] = post_norm_bias
    if args.add_qkv_bias or args.add_bias_linear:
        message["qkv bias"] = torch.cat(qkv_bias, dim=0)
    if args.add_bias_linear:
        message["proj bias"] = proj_bias



def set_embedding_ckpt(message, models, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)
    # embedding
    pos_embed = None
    if md.position_embedding_type == 'learned_absolute':
        pos_embed = message.pop("position embeddings")

    orig_word_embed = message.pop("word embeddings")
    full_word_embed = padding_vocab_size(orig_word_embed, md, args, "language_padded_vocab_size")
    out_word_embed = torch.chunk(full_word_embed, tp_size, dim=0)

    out_vision_embed = None
    if "vision embeddings" in message:
        orig_vision_embed = message.pop("vision embeddings")
        full_vision_embed = padding_vocab_size(orig_vision_embed, md, args, "vision_padded_vocab_size")
        out_vision_embed = torch.chunk(full_vision_embed, tp_size, dim=0)

    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        if out_vision_embed is not None:
            model.embedding.vision_embeddings.weight.data.copy_(out_vision_embed[tp_rank])
        if pos_embed is not None:
            model.embedding.position_embeddings.weight.data.copy_(pos_embed)
        else:
            assert not hasattr(model.embedding, "position_embeddings")


def set_attn_ckpt(message, models, layer_id, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)

    # weight
    qkv_weight = torch.chunk(message.pop("qkv weight"), tp_size, dim=0)
    proj_weight = torch.chunk(message.pop("proj weight"), tp_size, dim=1)
    input_norm_weight = message.pop("input norm weight")
    post_norm_weight = message.pop("post norm weight")
    # bias
    if md.norm_has_bias:
        input_norm_bias = message.pop("input norm bias")
        post_norm_bias = message.pop("post norm bias")
    if md.add_qkv_bias or md.add_bias_linear:
        qkv_bias = torch.chunk(message.pop("qkv bias"), tp_size, dim=0)
    if md.add_bias_linear:
        proj_bias = message.pop("proj bias")

    # set data to transformer layer's self-attention
    for tp_ep_rank, model in enumerate(models):
        tp_rank = tp_ep_rank % tp_size
        tf_layer = model.decoder.layers[layer_id]

        tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)
        tf_layer.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)
        if md.norm_has_bias:
            tf_layer.input_layernorm.bias.data.copy_(input_norm_bias)
            tf_layer.pre_mlp_layernorm.bias.data.copy_(post_norm_bias)

        tf_layer.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
        tf_layer.self_attention.linear_proj.weight.data.copy_(proj_weight[tp_rank])
        if md.add_qkv_bias or md.add_bias_linear:
            tf_layer.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
        if md.add_bias_linear:
            tf_layer.self_attention.linear_proj.bias.data.copy_(proj_bias)


def set_output_layer_ckpt(message, models, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)

    if message["weight"].shape[0] > args.language_padded_vocab_size:
        # language embedding and vision embedding are all loaded
        from aquila3_dense.ckpt import set_output_layer_ckpt
        set_output_layer_ckpt(message, models, md, args)
    else:
        # language embedding is loaded and vision embedding is random initialized
        orig_output_layer_weight = message.pop("weight")
        full_output_layer_weight = padding_vocab_size(orig_output_layer_weight, md, args)
        output_layer_weight = torch.chunk(full_output_layer_weight, tp_size, dim=0)
        part_v = output_layer_weight[0].shape[0]
        for tp_ep_rank, model in enumerate(models):
            tp_rank = tp_ep_rank % tp_size

            # only set language_padded_vocab_size's data in output_layer.weight
            # the remian data is random init
            output_layer_w = output_layer_weight[tp_rank]
            start_v = tp_rank * part_v
            if start_v + part_v < args.language_padded_vocab_size:
                output_layer_w = output_layer_w
            elif start_v < args.language_padded_vocab_size and start_v + part_v > args.language_padded_vocab_size:
                output_layer_w = output_layer_w[:args.language_padded_vocab_size - start_v]
            else:
                continue

            dim = output_layer_w.shape[0]
            model.output_layer.weight.data[:dim, :] = output_layer_w
