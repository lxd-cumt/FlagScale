# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_with_cos_sin,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_context_parallel_group,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import deprecate_inference_params, divide, is_fa_min_version

from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MagiAttentionTransformerConfig
from megatron.core.utils import deprecate_inference_params

from magi_attention.common.ranges import AttnRanges
from magi_attention.common.enum import AttnMaskType as MagiAttnMaskType
from magi_attention.common.enum import AttnOverlapMode as MagiAttnOverlapMode
from magi_attention.common.enum import AttnRole as MagiAttnRole
from magi_attention.api.functools import compute_pad_size, squash_batch_dim, pad_at_dim, unpad_at_dim
from magi_attention.api.magi_attn_interface import magi_attn_flex_dispatch, calc_attn, undispatch, get_position_ids, magi_attn_flex_key
from magi_attention.config import DistAttnConfig
from magi_attention.meta.solver.dispatch_solver import (
    DispatchConfig,
    LBDispatchAlg,
    DPDispatchAlg,
    BSDispatchAlg,
    MinHeapDispatchAlg,
    BTPDispatchAlg,
    ToppHeapDispatchAlg,
)
from magi_attention.meta.solver.overlap_solver import (
    OverlapConfig,
    UniformOverlapAlg,
    GreedyOverlapAlg,
)
from megatron.core.transformer.attention import SelfAttentionSubmodules

type_to_dispatch_alg = {
    "lower_bound": LBDispatchAlg,
    "dynamic_programming": DPDispatchAlg,
    "binary_search": BSDispatchAlg,
    "min_heap": MinHeapDispatchAlg,
    "topp_heap": ToppHeapDispatchAlg,
    "backtracing_pruning": BTPDispatchAlg,
}

type_to_overlap_mode = {
    "static": MagiAttnOverlapMode.STATIC,
    "dynamic": MagiAttnOverlapMode.DYNAMIC,
}

type_to_overlap_alg = {
    "uniform": UniformOverlapAlg,
    "greedy": GreedyOverlapAlg,
}

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward
    from flashattn_hopper.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )

    HAVE_FA3 = True
except:
    HAVE_FA3 = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


def get_ranges_on_this_cp_rank(
    q_ranges,
    k_ranges,
    magi_attn_mask_type,
    cp_rank,
    seq_len,# total_seq_len / cp_size
):
    global_seq_start = cp_rank * seq_len
    global_seq_end = (cp_rank+1) * seq_len
    cur_q_ranges = []
    cur_k_ranges = []
    cur_magi_attn_mask_type = []

    for idx, q_range in enumerate(q_ranges):
        start = max(q_range[0], global_seq_start) - cp_rank*seq_len
        end = min(q_range[1], global_seq_end) - cp_rank*seq_len
        if start < end:
            cur_q_ranges.append([start, end])
            cur_k_ranges.append(k_ranges[idx])
            cur_magi_attn_mask_type.append(magi_attn_mask_type[idx])

    return cur_q_ranges, cur_k_ranges, cur_magi_attn_mask_type


def get_global_k(
    k,
    seq_len, # total_seq_len / cp_size
    bsz,
    pad_size,
    batch_id,
):
    cp_rank = k // seq_len
    offset = k % seq_len
    global_k = cp_rank * (seq_len*bsz+pad_size) + batch_id * seq_len + offset
    return global_k

def split_into_intervals(lst):
    if not lst:
        return []
    result = []
    current_start = lst[0]
    current_end = lst[0] + 1

    for i in range(1, len(lst)):
        if lst[i] == lst[i-1] + 1:
            current_end = lst[i] + 1
        else:
            result.append([current_start, current_end])
            current_start = lst[i]
            current_end = lst[i] + 1

    result.append([current_start, current_end])
    return result

def get_global_k_list(
    k_range_start,
    k_range_end,
    seq_len, # total_seq_len / cp_size
    bsz,
    pad_size,
    batch_id,
):
    k_list = []
    for k in range(k_range_start, k_range_end):
        global_k = get_global_k(k, seq_len, bsz, pad_size, batch_id)
        k_list.append(global_k)
    
    return split_into_intervals(k_list)

class MagiAttentionSlices:
    def __init__(
        self,
        q_ranges: List[List[Union[int]]] = None,
        k_ranges: List[List[Union[int]]] = None,
        magi_attn_mask_type: List[Union[MagiAttnMaskType]] = None,
        seq_length: int = None,
    ):
        self.seq_length = seq_length
        
        if q_ranges is not None and k_ranges is not None and magi_attn_mask_type is not None:
            assert q_ranges[0][0] == 0 and q_ranges[-1][1] == self.seq_length, "invalid attention mask"
            self.q_ranges = q_ranges
            self.k_ranges = k_ranges
            self.magi_attn_mask_type = magi_attn_mask_type
        else:
            self.q_ranges = [[0, seq_length]]
            self.k_ranges = [[0, seq_length]]
            self.magi_attn_mask_type = [MagiAttnMaskType.CAUSAL]


    def update_ranges(
            self,
            seq_len, # total_seq_len / cp_size
            bsz,
            cp_size,
            pad_size,
        ):

            global_q_ranges = []
            global_k_ranges = []
            global_magi_attn_mask_type = []

            global_q_range_offset = 0
            for cp_rank in range(cp_size):
                # get mask on this cp rank
                cur_q_ranges, cur_k_ranges, cur_magi_attn_mask_type = get_ranges_on_this_cp_rank(
                        self.q_ranges,
                        self.k_ranges,
                        self.magi_attn_mask_type,
                        cp_rank,
                        seq_len,
                    )
                assert len(cur_q_ranges) == len(cur_k_ranges) and len(cur_q_ranges) == len(cur_magi_attn_mask_type), "invalid ranges after slicing cp"

                new_q_ranges = []
                new_k_ranges = []
                new_magi_attn_mask_type = []
                # expand mask to more batch
                num_valid_range_each_batch = len(cur_q_ranges)
                for batch_id in range(0, bsz):
                    for range_id in range(num_valid_range_each_batch):
                        q_range_start = cur_q_ranges[range_id][0]
                        q_range_end = cur_q_ranges[range_id][1]
                        k_range_start = cur_k_ranges[range_id][0]
                        k_range_end = cur_k_ranges[range_id][1]
                        attn_mask_type = cur_magi_attn_mask_type[range_id]

                        k_ranges_list = get_global_k_list(k_range_start, k_range_end, seq_len, bsz, pad_size, batch_id)
                        for k_id in range(len(k_ranges_list)):
                            new_q_ranges.append([q_range_start+batch_id*seq_len, q_range_end+batch_id*seq_len])
                            new_k_ranges.append(k_ranges_list[k_id])
                            new_magi_attn_mask_type.append(attn_mask_type)
                assert len(new_q_ranges) == len(new_k_ranges) and len(new_q_ranges) == len(new_magi_attn_mask_type), "invalid ranges after expanding batch"

                # add padding mask
                padding_start = seq_len * bsz
                padding_end = seq_len * bsz + pad_size
                new_q_ranges.append([padding_start, padding_end])
                new_k_ranges.append([0, 0])
                new_magi_attn_mask_type.append(MagiAttnMaskType.FULL)

                # append ranges of this cp rank to global ranges
                for range_id, new_q_range in enumerate(new_q_ranges):
                    q_range_start = new_q_range[0] + global_q_range_offset
                    q_range_end = new_q_range[1] + global_q_range_offset
                    k_range = new_k_ranges[range_id]
                    attn_mask_type = new_magi_attn_mask_type[range_id]
                    global_q_ranges.append([q_range_start, q_range_end])
                    global_k_ranges.append(k_range)
                    global_magi_attn_mask_type.append(attn_mask_type)

                assert len(global_q_ranges) == len(global_k_ranges) and len(global_q_ranges) == len(global_magi_attn_mask_type), "invalid ranges of global ranges"
                
                # update q range offset for next cp rank
                global_q_range_offset += (seq_len*bsz+pad_size)

            updated_ranges = tuple([global_q_ranges, global_k_ranges, global_magi_attn_mask_type])
            return updated_ranges


class MagiAttention(Attention):
    def __init__(
        self,
        config: MagiAttentionTransformerConfig,
        submodules: Union[SelfAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        cp_comm_type: str = None,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
        )

        self.config = config

        self.magi_dist_attn_config = DistAttnConfig(
            DispatchConfig(
                alg=type_to_dispatch_alg[self.config.magi_dispatch_alg](),
            ),
            OverlapConfig(
                enable=self.config.magi_overlap_enable,
                mode=type_to_overlap_mode[self.config.magi_overlap_mode],
                degree=self.config.magi_degree,
                dynamic_max_degree=self.config.magi_dynamic_max_degree,
                min_chunk_size=self.config.magi_min_chunk_size,
                max_num_chunks=self.config.magi_max_num_chunks,
                alg=type_to_overlap_alg[self.config.magi_overlap_alg](),
                calc_cost_factor=self.config.magi_calc_cost_factor,
                comm_cost_factor=self.config.magi_comm_cost_factor,
            ),
            high_bandwith_domain_size=self.config.magi_high_bandwidth_domain_size,
            deterministic=self.config.magi_deterministic,
        )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            tp_group=self.model_comm_pgs.tp,
        )

        if submodules.q_layernorm is not None:
            if not self.config.qk_layernorm_hidden_dim:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    hidden_size=self.hidden_size_per_attention_head,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                tp_world_size = get_tensor_model_parallel_world_size()
                assert tp_world_size <= 1, "TP world size must be less than 1 for qk_layernorm_hidden_dim"
                # nums_head_cur_rank = divide(self.config.num_attention_heads, tp_world_size)
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    hidden_size=self.query_projection_size,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            if not self.config.qk_layernorm_hidden_dim:
                self.k_layernorm = build_module(
                    submodules.k_layernorm,
                    hidden_size=self.hidden_size_per_attention_head,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                tp_world_size = get_tensor_model_parallel_world_size()
                assert tp_world_size <= 1, "TP world size must be less than 1 for qk_layernorm_hidden_dim"
                # nums_head_cur_rank = divide(self.config.num_attention_heads, tp_world_size)
                self.k_layernorm = build_module(
                    submodules.k_layernorm,
                    hidden_size=self.kv_projection_size,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
        else:
            self.k_layernorm = None

        self.updated_ranges_cache = None
        self.position_ids_cache = None


    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [b*s/cp+pad, h] --> [b*s/cp+pad, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [b*s/cp+pad, hp] --> [b*s/cp+pad, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [b*s/cp+pad, ng, (np/ng + 2) * hn]
            # --> [b*s/cp+pad, ng, np/ng * hn], [b*s/cp+pad, ng, hn], [b*s/cp+pad, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 2, split_arg_list)
        else:

            # [b*s/cp+pad, ng, (np/ng + 2) * hn]
            # --> [b*s/cp+pad, ng, np/ng * hn], [b*s/cp+pad, ng, hn], [b*s/cp+pad, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=2)

        # [b*s/cp+pad, ng, np/ng * hn] -> [b*s/cp+pad, np, hn]
        query = query.reshape(query.size(0), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            if not self.config.qk_layernorm_hidden_dim:
                query = self.q_layernorm(query)
            else:
                # [b*s/cp+pad, np, hn] -> [b*s/cp+pad, 1, np * hn]
                query_shape = list(query.shape)
                query = query.reshape(query.size(0), 1, -1)
                query = self.q_layernorm(query)
                query = query.reshape(*query_shape)

        if self.k_layernorm is not None:
            if not self.config.qk_layernorm_hidden_dim:
                key = self.k_layernorm(key)
            else:
                # [b*s/cp+pad, ng, hn] -> [b*s/cp+pad, 1, ng * hn]
                key_shape = list(key.shape)
                key = key.reshape(key.size(0), 1, -1)
                key = self.k_layernorm(key)
                key = key.reshape(*key_shape)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        magi_attn_runtime_key,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            magi_attn_runtime_key = inputs[3]
            output, _ = calc_attn(
                local_query,
                local_key,
                local_value,
                magi_attn_runtime_key,
            )
            return output

        hidden_states = tensor_parallel.checkpoint(
            custom_forward, False, query, key, value, magi_attn_runtime_key
        )

        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        magi_attention_slices: MagiAttentionSlices = None,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:

        assert not self.config.sequence_parallel, "currently, magi attention in flagscale is not support with sequence parallel"
        # here, seq_len = total_seq_len / cp_size
        seq_len, batch_size, hidden_size = hidden_states.shape
        
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        # hidden_states: [s/cp, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        hidden_states = hidden_states.permute(1, 0, 2).contiguous() # [s/cp, b, h] -> [b, s/cp, h]
        # squash batch dim
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1]) # [b, s/cp, h] -> [b*s/cp, h]

        # # cal padding size
        original_seqlen = batch_size * seq_len
        head_dim = self.config.kv_channels
        pad_size, _ = compute_pad_size(original_seqlen, 1, head_dim)
        cp_size = self.config.context_parallel_size
        total_seqlen = (original_seqlen + pad_size) * cp_size

        # # padding hidden_states
        new_hidden_states = pad_at_dim(hidden_states, 0, pad_size)
        
        # # update q_ranges, k_ranges, attn_mask, fake here            
        if self.updated_ranges_cache is None:
            update_ranges = magi_attention_slices.update_ranges(seq_len, batch_size, cp_size, pad_size)
            self.updated_ranges_cache = update_ranges
        q_ranges = self.updated_ranges_cache[0]
        k_ranges = self.updated_ranges_cache[1]
        magi_attn_mask_type = self.updated_ranges_cache[2]
        print(f"new q ranges is {q_ranges}")
        print(f"new k ranges is {k_ranges}")
        print(f"new magi attn mask is {magi_attn_mask_type}")

        cur_pad_size, _ = compute_pad_size(total_seqlen, cp_size, head_dim)
        assert cur_pad_size == 0, "invalid pad size"

        # [b*s/cp+pad, h]
        local_hidden_states, magi_attn_runtime_key = magi_attn_flex_key(
                new_hidden_states,
                q_ranges=AttnRanges.from_ranges(q_ranges),
                k_ranges=AttnRanges.from_ranges(k_ranges),
                attn_mask_type=magi_attn_mask_type,
                total_seqlen_q=total_seqlen,
                total_seqlen_k=total_seqlen,
                head_dim=head_dim,
                pad_size=cur_pad_size,
                cp_group=get_context_parallel_group(),
                is_same_source=True,
                is_q_permutable=True,
                is_k_permutable=True,
                dist_attn_config=self.magi_dist_attn_config,
          )
        # local_query: [b*s/cp+pad, nh, hd]
        local_query, local_key, local_value = self.get_query_key_value_tensors(local_hidden_states, key_value_states)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if self.position_ids_cache is None:
            position_ids = torch.cat(
                (torch.arange(seq_len).repeat(batch_size),
                torch.full((pad_size,), seq_len-1),)
            )
            self.position_ids_cache = position_ids
        
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    local_query = apply_rotary_pos_emb(
                        local_query,
                        q_pos_emb,
                        config=self.config,
                        cu_seqlens=cu_seqlens_q,
                        cp_group=self.model_comm_pgs.cp,
                        position_ids=self.position_ids_cache,
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q, self.model_comm_pgs.cp
                    )
            if k_pos_emb is not None:
                local_key = apply_rotary_pos_emb(
                    local_key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    cp_group=self.model_comm_pgs.cp,
                    position_ids=self.position_ids_cache,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                local_query,
                local_key,
                local_value,
                magi_attn_runtime_key,
            )
        else:
            core_attn_out, _ = calc_attn(
                local_query,
                local_key,
                local_value,
                magi_attn_runtime_key,
            )

        core_attn_out = unpad_at_dim(core_attn_out, 0, pad_size).contiguous() # [b*s/cp+pad, h] -> [b*s/cp, h]
        core_attn_out = core_attn_out.view(batch_size, seq_len, -1).permute(1, 0, 2) # [b*s/cp, h] -> [b, s/cp, h] -> [s/cp, b, h]
        
        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)

        return output, bias