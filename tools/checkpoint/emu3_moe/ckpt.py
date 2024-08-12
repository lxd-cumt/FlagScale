import sys
import copy
import torch
sys.path.append("..")
from utils import padding_vocab_size
from mixtral.ckpt import (
    get_final_norm_ckpt, 
    get_output_layer_ckpt,
    set_final_norm_ckpt,
)


# def get_hf_attn_ckpt(message, model, layer_id, args):
#     raise NotImplementedError()


# def get_hf_mlp_ckpt(message, model, layer_id, args):
#     raise NotImplementedError()


# def set_hf_attn_ckpt(message, model, layer_id, md, args):
#     qkv_weight = message.pop("qkv weight")
#     proj_weight = message.pop("proj weight")
#     input_norm_weight = message.pop("input norm weight")
#     post_norm_weight = message.pop("post norm weight")
#     # bias
#     if md.norm_has_bias:
#         input_norm_bias = message.pop("input norm bias")
#         post_norm_bias = message.pop("post norm bias")
#     if md.add_qkv_bias or md.add_bias_linear:
#         qkv_bias = message.pop("qkv bias")
#     if md.add_bias_linear:
#         proj_bias = message.pop("proj bias")

#     nh = args.num_attention_heads
#     ng = args.num_query_groups if args.group_query_attention else args.num_attention_heads
#     dim = args.hidden_size
#     assert nh % ng == 0

#     tf_layer = model.model.layers[layer_id]
#     # weight
#     qkv_weight = qkv_weight.view(ng, -1, dim)
#     qkv_weight = torch.split(qkv_weight, [dim//ng, dim//nh, dim//nh], dim=1)
#     tf_layer.self_attn.q_proj.weight.data.copy_(qkv_weight[0].reshape(-1, dim))
#     tf_layer.self_attn.k_proj.weight.data.copy_(qkv_weight[1].reshape(-1, dim))
#     tf_layer.self_attn.v_proj.weight.data.copy_(qkv_weight[2].reshape(-1, dim))
#     tf_layer.self_attn.o_proj.weight.data.copy_(proj_weight)
#     tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)
#     tf_layer.post_attention_layernorm.weight.data.copy_(post_norm_weight)
#     # bias
#     if md.norm_has_bias:
#         tf_layer.input_layernorm.bias.data.copy_(input_norm_bias)
#         tf_layer.post_attention_layernorm.bias.data.copy_(post_norm_bias)
#     if md.add_qkv_bias or md.add_bias_linear:
#         qkv_bias = qkv_bias.view(ng, -1, 1)
#         qkv_bias = torch.split(qkv_bias, [dim//ng, dim//nh, dim//nh], dim=1)
#         tf_layer.self_attn.q_proj.bias.data.copy_(qkv_bias[0].reshape(-1))
#         tf_layer.self_attn.k_proj.bias.data.copy_(qkv_bias[1].reshape(-1))
#         tf_layer.self_attn.v_proj.bias.data.copy_(qkv_bias[2].reshape(-1))
#     if md.add_bias_linear:
#         tf_layer.self_attn.o_proj.bias.data.copy_(proj_bias)


# def set_hf_mlp_ckpt(message, model, layer_id, md, args):
#     assert md.swiglu is True
#     assert args.num_experts is not None

#     if md.previous_num_experts is not None:
#         tf_layer = model.model.layers[layer_id]
#         tf_layer.block_sparse_moe.gate.weight.data.copy_(message.pop("router weight"))
#         for id in range(md.previous_num_experts):
#             expert = tf_layer.block_sparse_moe.experts[id]
#             expert.w1.weight.data.copy_(message.pop(f"expert{id} l0 weight W"))
#             expert.w3.weight.data.copy_(message.pop(f"expert{id} l0 weight V"))
#             expert.w2.weight.data.copy_(message.pop(f"expert{id} l1 weight"))

#             if md.add_bias_linear:
#                 expert.w1.bias.data.copy_(message.pop(f"expert{id} l0 bias W"))
#                 expert.w3.bias.data.copy_(message.pop(f"expert{id} l0 bias V"))
#                 expert.w2.bias.data.copy_(message.pop(f"expert{id} l1 bias"))
#     else:
#         tf_layer = model.model.layers[layer_id]
#         # tf_layer.block_sparse_moe.gate.weight.data.copy_(message.pop("router weight"))
#         mlp_l0_weight_W = message.pop("mlp l0 weight W")
#         mlp_l0_weight_V = message.pop("mlp l0 weight V")
#         mlp_l1_weight = message.pop("mlp l1 weight")
#         if md.add_bias_linear:
#             mlp_l0_bias_W = message.pop("mlp l0 bias W")
#             mlp_l0_bia_V = message.pop("mlp l0 bias V")
#             mlp_l1_bias = message.pop("mlp l1 bias")

#         for id in range(args.num_experts):
#             expert = tf_layer.block_sparse_moe.experts[id]
#             expert.w1.weight.data.copy_(mlp_l0_weight_W)
#             expert.w3.weight.data.copy_(mlp_l0_weight_V)
#             expert.w2.weight.data.copy_(mlp_l1_weight)

#             if md.add_bias_linear:
#                 expert.w1.bias.data.copy_(mlp_l0_bias_W)
#                 expert.w3.bias.data.copy_(mlp_l0_bia_V)
#                 expert.w2.bias.data.copy_(mlp_l1_bias)


def _get_parallel_size(args):
    return args.tensor_model_parallel_size, \
        args.pipeline_model_parallel_size, \
        args.expert_model_parallel_size, \
        args.virtual_pipeline_model_parallel_size or 1


# def get_embedding_ckpt(message, models, args):
#     tp_size, _, _, _ = _get_parallel_size(args)
#     word_embeddings = []
#     vision_embeddings = []
#     complete_tp_ranks = []
#     for tp_ep_rank, model in enumerate(models):
#         tp_rank = tp_ep_rank % tp_size
#         if tp_rank in complete_tp_ranks:
#             continue
#         complete_tp_ranks.append(tp_rank)
#         word_embeddings.append(model.embedding.word_embeddings.weight.data)
#         vision_embeddings.append(model.embedding.vision_embeddings.weight.data)
#     message["word embeddings"] = torch.cat(word_embeddings, dim=0)
#     message["vision embeddings"] = torch.cat(vision_embeddings, dim=0)
#     if args.position_embedding_type == 'learned_absolute':
#         message["position embeddings"] = models[0].embedding.position_embeddings.weight.data
#     else:
#         assert not hasattr(models[0].embedding, 'position_embeddings')


# def get_attn_ckpt(message, models, layer_id, args):
#     tp_size, _, _, _ = _get_parallel_size(args)

#     # common
#     input_norm_weight = None
#     input_norm_bias = None
#     post_norm_weight = None
#     post_norm_bias = None

#     # language
#     # parallel tensor
#     qkv_weight = []
#     qkv_bias = []
#     proj_weight = []
#     # non-parallel tensor
#     proj_bias = None

#     if args.use_multimodal_router_attn:
#         # vision
#         # parallel tensor
#         qkv_weight_vs = []
#         qkv_bias_vs = []
#         proj_weight_vs = []
#         # non-parallel tensor
#         proj_bias_vs = None

#     complete_tp_ranks = []
#     for tp_ep_rank, model in enumerate(models):
#         tp_rank = tp_ep_rank % tp_size
#         if tp_rank in complete_tp_ranks:
#             continue
#         complete_tp_ranks.append(tp_rank)
        
#         tf_layer = model.decoder.layers[layer_id]

#         # common
#         input_norm_weight = tf_layer.input_layernorm.weight.data
#         post_norm_weight = tf_layer.pre_mlp_layernorm.weight.data
#         if args.norm_has_bias:
#             input_norm_bias = tf_layer.input_layernorm.bias.data
#             post_norm_bias = tf_layer.pre_mlp_layernorm.bias.data
#         if args.use_multimodal_router_attn:
#             # language
#             qkv_weight.append(tf_layer.self_attention.linear_qkv.language.weight.data)
#             proj_weight.append(tf_layer.self_attention.linear_proj.language.weight.data)
#             if args.add_qkv_bias or args.add_bias_linear:
#                 qkv_bias.append(tf_layer.self_attention.linear_qkv.language.bias.data)
#             if args.add_bias_linear:
#                 proj_bias = tf_layer.self_attention.linear_proj.language.bias.data
#             # vision
#             qkv_weight_vs.append(tf_layer.self_attention.linear_qkv.vision.weight.data)
#             proj_weight_vs.append(tf_layer.self_attention.linear_proj.vision.weight.data)
#             if args.add_qkv_bias or args.add_bias_linear:
#                 qkv_bias_vs.append(tf_layer.self_attention.linear_qkv.vision.bias.data)
#             if args.add_bias_linear:
#                 proj_bias_vs = tf_layer.self_attention.linear_proj.vision.bias.data
#         else:
#             qkv_weight.append(tf_layer.self_attention.linear_qkv.weight.data)
#             proj_weight.append(tf_layer.self_attention.linear_proj.weight.data)
#             if args.add_qkv_bias or args.add_bias_linear:
#                 qkv_bias.append(tf_layer.self_attention.linear_qkv.bias.data)
#             if args.add_bias_linear:
#                 proj_bias = tf_layer.self_attention.linear_proj.bias.data

#     # common
#     message["input norm weight"] = input_norm_weight
#     message["post norm weight"] = post_norm_weight
#     if args.norm_has_bias:
#         message["input norm bias"] = input_norm_bias
#         message["post norm bias"] = post_norm_bias
#     # language & vision
#     if args.use_multimodal_router_attn:
#         message["qkv weight"] = {"lm": torch.cat(qkv_weight, dim=0), "vs": torch.cat(qkv_weight_vs, dim=0)}
#         message["proj weight"] = {"lm": torch.cat(proj_weight, dim=1), "vs": torch.cat(proj_weight_vs, dim=1)}
#         if args.add_qkv_bias or args.add_bias_linear:
#             message["qkv bias"] = {"lm": torch.cat(qkv_bias, dim=0), "vs": torch.cat(qkv_bias_vs, dim=0)}
#         if args.add_bias_linear:
#             message["proj bias"] = {"lm": proj_bias, "vs": proj_bias_vs}
#     else:
#         message["qkv weight"] = torch.cat(qkv_weight, dim=0)
#         message["proj weight"] = torch.cat(proj_weight, dim=1)
#         if args.add_qkv_bias or args.add_bias_linear:
#             message["qkv bias"] = torch.cat(qkv_bias, dim=0)
#         if args.add_bias_linear:
#             message["proj bias"] = proj_bias


# def get_mlp_ckpt(message, models, layer_id, args):
#     tp_size, _, ep_size, _ = _get_parallel_size(args)

#     if args.use_multimodal_router_mlp:
#         assert args.num_experts is not None and args.num_experts % ep_size == 0
#         num_lm_experts = args.multimodal_num_experts_split
#         num_vs_experts = args.num_experts - num_lm_experts

#         message["router weight"] = dict()
#         num_lm_local_experts = num_lm_experts // ep_size
#         num_vs_local_experts = num_vs_experts // ep_size
#         for expert_id in range(args.num_experts // ep_size):
#             for ep_rank in range(ep_size):
#                 # language
#                 # parallel tensor
#                 l0_weight = []
#                 l0_bias = []
#                 l1_weight = []
#                 # non-parallel tensor
#                 l1_bias = None
#                 router_weight = None

#                 if expert_id < num_lm_local_experts:
#                     mode = "lm"
#                     module = "language"
#                     eid = expert_id
#                     global_expert_id = num_lm_local_experts * ep_rank + eid
#                 else:
#                     mode = "vs"
#                     module = "vision"
#                     eid = expert_id - num_lm_local_experts
#                     global_expert_id = num_vs_local_experts * ep_rank + eid + num_lm_experts

#                 for tp_rank in range(tp_size):
#                     tp_ep_rank = ep_rank * tp_size + tp_rank
#                     tf_layer = models[tp_ep_rank].decoder.layers[layer_id]
#                     router_weight = eval(f"tf_layer.mlp.{module}").router.weight.data
#                     expert = eval(f"tf_layer.mlp.{module}").experts.local_experts[eid]
#                     # weight
#                     l0_weight.append(expert.linear_fc1.weight.data)
#                     l1_weight.append(expert.linear_fc2.weight.data)
#                     # bias
#                     if args.add_bias_linear:
#                         l0_bias.append(expert.linear_fc1.bias.data)
#                         l1_bias = expert.linear_fc2.bias.data

#                 # weight
#                 message["router weight"][mode] = router_weight
#                 message[f"expert{global_expert_id} l1 weight"] = torch.cat(l1_weight, dim=1)
#                 if args.swiglu:
#                     for tp_rank in range(tp_size):
#                         l0_weight[tp_rank] = torch.chunk(l0_weight[tp_rank], 2, dim=0)
#                     message[f"expert{global_expert_id} l0 weight W"] = torch.cat([w[0] for w in l0_weight], dim=0)
#                     message[f"expert{global_expert_id} l0 weight V"] = torch.cat([w[1] for w in l0_weight], dim=0)
#                 else:
#                     message[f"expert{global_expert_id} l0 weight"] = torch.cat(l0_weight, dim=0)
#                 # bias
#                 if args.add_bias_linear:
#                     message[f"expert{global_expert_id} l1 bias"] = l1_bias
#                     if args.swiglu:
#                         for tp_rank in range(tp_size):
#                             l0_bias[tp_rank] = torch.chunk(l0_bias[tp_rank], 2, dim=0)
#                         message[f"expert{global_expert_id} l0 bias W"] = torch.cat([b[0] for b in l0_bias],dim=0)
#                         message[f"expert{global_expert_id} l0 bias V"] = torch.cat([b[1] for b in l0_bias],dim=0)
#                     else:
#                         message[f"expert{global_expert_id} l0 bias"] = torch.cat(l0_bias, dim=0)
#     else:
#         assert args.num_experts is not None and args.num_experts % ep_size == 0
#         num_local_experts = args.num_experts // ep_size
#         for expert_id in range(num_local_experts):
#             for ep_rank in range(ep_size):
#                 global_expert_id = num_local_experts * ep_rank + expert_id

#                 # parallel tensor
#                 l0_weight = []
#                 l0_bias = []
#                 l1_weight = []
#                 # non-parallel tensor
#                 l1_bias = None
#                 router_weight = None
#                 for tp_rank in range(tp_size):
#                     tp_ep_rank = ep_rank * tp_size + tp_rank
#                     tf_layer = models[tp_ep_rank].decoder.layers[layer_id]
#                     expert = tf_layer.mlp.experts.local_experts[expert_id]
#                     # weight
#                     l0_weight.append(expert.linear_fc1.weight.data)
#                     l1_weight.append(expert.linear_fc2.weight.data)
#                     router_weight = tf_layer.mlp.router.weight.data
#                     # bias
#                     if args.add_bias_linear:
#                         l0_bias.append(expert.linear_fc1.bias.data)
#                         l1_bias = expert.linear_fc2.bias.data

#                 # weight
#                 message["router weight"] = router_weight
#                 message[f"expert{global_expert_id} l1 weight"] = torch.cat(l1_weight, dim=1)
#                 if args.swiglu:
#                     for tp_rank in range(tp_size):
#                         l0_weight[tp_rank] = torch.chunk(l0_weight[tp_rank], 2, dim=0)
#                     message[f"expert{global_expert_id} l0 weight W"] = torch.cat([w[0] for w in l0_weight], dim=0)
#                     message[f"expert{global_expert_id} l0 weight V"] = torch.cat([w[1] for w in l0_weight], dim=0)
#                 else:
#                     message[f"expert{global_expert_id} l0 weight"] = torch.cat(l0_weight, dim=0)
#                 # bias
#                 if args.add_bias_linear:
#                     message[f"expert{global_expert_id} l1 bias"] = l1_bias
#                     if args.swiglu:
#                         for tp_rank in range(tp_size):
#                             l0_bias[tp_rank] = torch.chunk(l0_bias[tp_rank], 2, dim=0)
#                         message[f"expert{global_expert_id} l0 bias W"] = torch.cat([b[0] for b in l0_bias],dim=0)
#                         message[f"expert{global_expert_id} l0 bias V"] = torch.cat([b[1] for b in l0_bias],dim=0)
#                     else:
#                         message[f"expert{global_expert_id} l0 bias"] = torch.cat(l0_bias, dim=0)


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
        origin_vision_embed = message.pop("vision embeddings")
        full_vision_embed = padding_vocab_size(origin_vision_embed, md, args, "vision_padded_vocab_size")
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

        # for common
        tf_layer.input_layernorm.weight.data.copy_(input_norm_weight)
        tf_layer.pre_mlp_layernorm.weight.data.copy_(post_norm_weight)
        if md.norm_has_bias:
            tf_layer.input_layernorm.bias.data.copy_(input_norm_bias)
            tf_layer.pre_mlp_layernorm.bias.data.copy_(post_norm_bias)

        # for linear
        if args.use_multimodal_router_attn:
            tf_layer.self_attention.linear_qkv.language.weight.data.copy_(qkv_weight[tp_rank])
            tf_layer.self_attention.linear_proj.language.weight.data.copy_(proj_weight[tp_rank])
            if md.add_qkv_bias or md.add_bias_linear:
                tf_layer.self_attention.linear_qkv.language.bias.data.copy_(qkv_bias[tp_rank])
            if md.add_bias_linear:
                tf_layer.self_attention.linear_proj.language.bias.data.copy_(proj_bias)

            tf_layer.self_attention.linear_qkv.vision.weight.data.copy_(qkv_weight[tp_rank])
            tf_layer.self_attention.linear_proj.vision.weight.data.copy_(proj_weight[tp_rank])
            if md.add_qkv_bias or md.add_bias_linear:
                tf_layer.self_attention.linear_qkv.vision.bias.data.copy_(qkv_bias[tp_rank])
            if md.add_bias_linear:
                tf_layer.self_attention.linear_proj.vision.bias.data.copy_(proj_bias)
        else:
            tf_layer.self_attention.linear_qkv.weight.data.copy_(qkv_weight[tp_rank])
            tf_layer.self_attention.linear_proj.weight.data.copy_(proj_weight[tp_rank])
            if md.add_qkv_bias or md.add_bias_linear:
                tf_layer.self_attention.linear_qkv.bias.data.copy_(qkv_bias[tp_rank])
            if md.add_bias_linear:
                tf_layer.self_attention.linear_proj.bias.data.copy_(proj_bias)


def set_mlp_ckpt(message, models, layer_id, md, args):
    tp_size, _, ep_size, _ = _get_parallel_size(args)

    # for dense
    if args.num_experts is None:
        from mistral.ckpt import set_mlp_ckpt
        set_mlp_ckpt(message, models, layer_id, md, args)
        return

    # for moe
    assert md.previous_num_experts is not None
    if not args.use_multimodal_router_mlp:
        from mixtral.ckpt import set_mlp_ckpt
        assert md.previous_num_experts == args.num_experts
        set_mlp_ckpt(message, models, layer_id, md, args)
        return

    # for multimodal moe
    assert md.previous_num_experts <= args.num_experts
    if md.previous_num_experts < args.num_experts:
        assert md.previous_num_experts == args.multimodal_num_experts_split
        num_lm_experts = args.multimodal_num_experts_split
        num_lm_local_experts = num_lm_experts // ep_size

        router_weight = message.pop("router weight")
        for expert_id in range(num_lm_local_experts):
            for ep_rank in range(ep_size):
                global_expert_id = num_lm_local_experts * ep_rank + expert_id

                # weight
                l1_weight = torch.chunk(message.pop(f"expert{global_expert_id} l1 weight"), tp_size, dim=1)
                if md.swiglu:
                    l0_weight_W = torch.chunk(message.pop(f"expert{global_expert_id} l0 weight W"), tp_size, dim=0)
                    l0_weight_V = torch.chunk(message.pop(f"expert{global_expert_id} l0 weight V"), tp_size, dim=0)
                    l0_weight = [torch.cat(weights, dim=0) for weights in zip(l0_weight_W, l0_weight_V)]
                else:
                    l0_weight = torch.chunk(message.pop(f"expert{global_expert_id} l0 weight"), tp_size, dim=0)
                # bias
                if md.add_bias_linear:
                    l1_bias = message.pop(f"expert{global_expert_id} l1 bias")
                    if md.swiglu:
                        l0_bias_W = torch.chunk(message.pop(f"expert{global_expert_id} l0 bias W"), tp_size, dim=0)
                        l0_bias_V = torch.chunk(message.pop(f"expert{global_expert_id} l0 bias V"), tp_size, dim=0)
                        l0_bias = [torch.cat(bias, dim=0) for bias in zip(l0_bias_W, l0_bias_V)]
                    else:
                        l0_bias = torch.chunk(message.pop(f"expert{global_expert_id} l0 bias"), tp_size, dim=0)

                # set data for language module
                # vison module weights are random-initialization
                for tp_rank in range(tp_size):
                    tp_ep_rank = ep_rank * tp_size + tp_rank
                    tf_layer = models[tp_ep_rank].decoder.layers[layer_id]
                    tf_layer.mlp.language.router.weight.data.copy_(router_weight)

                    expert = tf_layer.mlp.language.experts.local_experts[expert_id]
                    expert.linear_fc1.weight.data.copy_(l0_weight[tp_rank])
                    expert.linear_fc2.weight.data.copy_(l1_weight[tp_rank])
                    if md.add_bias_linear:
                        expert.linear_fc1.bias.data.copy_(l0_bias[tp_rank])
                        expert.linear_fc2.bias.data.copy_(l1_bias)

    else:
        num_lm_experts = args.multimodal_num_experts_split
        num_vs_experts = args.num_experts - num_lm_experts
        num_lm_local_experts = num_lm_experts // ep_size
        num_vs_local_experts = num_vs_experts // ep_size

        router_weight = message.pop("router weight")
        num_local_experts = args.num_experts // ep_size
        for expert_id in range(num_local_experts):
            for ep_rank in range(ep_size):
                if expert_id < num_lm_local_experts:
                    mode = "lm"
                    module = "language"
                    eid = expert_id
                    global_expert_id = num_lm_local_experts * ep_rank + eid
                else:
                    mode = "vs"
                    module = "vision"
                    eid = expert_id - num_lm_local_experts
                    global_expert_id = num_vs_local_experts * ep_rank + eid + num_lm_experts

                # weight
                l1_weight = torch.chunk(message.pop(f"expert{global_expert_id} l1 weight"), tp_size, dim=1)
                if md.swiglu:
                    l0_weight_W = torch.chunk(message.pop(f"expert{global_expert_id} l0 weight W"), tp_size, dim=0)
                    l0_weight_V = torch.chunk(message.pop(f"expert{global_expert_id} l0 weight V"), tp_size, dim=0)
                    l0_weight = [torch.cat(weights, dim=0) for weights in zip(l0_weight_W, l0_weight_V)]
                else:
                    l0_weight = torch.chunk(message.pop(f"expert{global_expert_id} l0 weight"), tp_size, dim=0)
                # bias
                if md.add_bias_linear:
                    l1_bias = message.pop(f"expert{global_expert_id} l1 bias")
                    if md.swiglu:
                        l0_bias_W = torch.chunk(message.pop(f"expert{global_expert_id} l0 bias W"), tp_size, dim=0)
                        l0_bias_V = torch.chunk(message.pop(f"expert{global_expert_id} l0 bias V"), tp_size, dim=0)
                        l0_bias = [torch.cat(bias, dim=0) for bias in zip(l0_bias_W, l0_bias_V)]
                    else:
                        l0_bias = torch.chunk(message.pop(f"expert{global_expert_id} l0 bias"), tp_size, dim=0)

                for tp_rank in range(tp_size):
                    tp_ep_rank = ep_rank * tp_size + tp_rank
                    tf_layer = models[tp_ep_rank].decoder.layers[layer_id]
                    eval(f"tf_layer.mlp.{module}").router.weight.data.copy_(router_weight[mode])

                    expert = eval(f"tf_layer.mlp.{module}").experts.local_experts[eid]
                    expert.linear_fc1.weight.data.copy_(l0_weight[tp_rank])
                    expert.linear_fc2.weight.data.copy_(l1_weight[tp_rank])
                    if md.add_bias_linear:
                        expert.linear_fc1.bias.data.copy_(l0_bias[tp_rank])
                        expert.linear_fc2.bias.data.copy_(l1_bias)


def set_output_layer_ckpt(message, models, md, args):
    tp_size, _, _, _ = _get_parallel_size(args)
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
