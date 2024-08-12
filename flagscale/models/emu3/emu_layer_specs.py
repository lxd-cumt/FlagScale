from dataclasses import dataclass
from typing import Union

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TENorm,
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm


@dataclass
class MultiModalSubmodules:
    language: Union[ModuleSpec, type] = None
    vision: Union[ModuleSpec, type] = None


# Use this spec to use lower level Transformer Engine modules (required for fp8 training)
def get_emu_with_transformer_engine_spec(
    num_experts: int = None, 
    use_multimodal_router_mlp: bool = False, 
    use_multimodal_router_attn: bool = False,
    qk_layernorm: bool = False, 
) -> ModuleSpec:

    mlp = _get_mlp_module_spec(num_experts, use_multimodal_router_mlp)
    self_attn = _get_self_attn_module_spec(use_multimodal_router_attn, qk_layernorm)

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=TENorm,
            self_attention=self_attn,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=TENorm,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def _get_self_attn_module_spec(
    use_multimodal_router_attn: bool = False,
    qk_layernorm: bool = False,
):
    if use_multimodal_router_attn:
        from flagscale.transformer.attention import MultiModalLinearQKV, MultiModalLinearProj
        return ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=ModuleSpec(
                    module=MultiModalLinearQKV,
                    submodules=MultiModalSubmodules(
                        language=TEColumnParallelLinear,
                        vision=TEColumnParallelLinear,
                    ),
                ),
                core_attention=TEDotProductAttention,
                linear_proj=ModuleSpec(
                    module=MultiModalLinearProj,
                    submodules=MultiModalSubmodules(
                        language=TERowParallelLinear,
                        vision=TERowParallelLinear,
                    ),
                ),
                q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
            ),
        )
    else:
        return ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=TENorm if qk_layernorm else IdentityOp,
                k_layernorm=TENorm if qk_layernorm else IdentityOp,
            ),
        )


# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    num_experts: int = None, 
    use_multimodal_router_mlp: bool = False 
) -> ModuleSpec:

    if num_experts is None: 
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        )
    else:
        # Mixture of experts with modules in megatron core.
        if use_multimodal_router_mlp:
            from flagscale.transformer.moe.moe_layer import MultiModalMoELayer
            return ModuleSpec(
                module=MultiModalMoELayer,
                submodules=MultiModalSubmodules(
                    language=ModuleSpec(
                        module=MoELayer,
                        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
                    ),
                    vision=ModuleSpec(
                        module=MoELayer,
                        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
                    )
                )
            )
        else:
            return ModuleSpec(
                module=MoELayer,
                submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            )
