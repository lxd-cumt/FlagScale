import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.utils import divide
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.module import MegatronModule

from flagscale.models.emu3.emu_layer_specs import MultiModalSubmodules


class MultiModalLinear(MegatronModule):
    def __init__(self, *args, **kwargs):
        assert kwargs.get("config", None)
        super().__init__(kwargs["config"])

        assert kwargs.get("submodules", None)
        submodules=kwargs.pop("submodules")
        assert isinstance(submodules, MultiModalSubmodules)

        self.language = build_module(
            submodules.language,
            *args,
            **kwargs)

        self.vision = build_module(
            submodules.vision,
            *args,
            **kwargs)

        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()


class MultiModalLinearQKV(MultiModalLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor, multimodal_mask: torch.Tensor):
        # hidden_states: [s/tp b h]
        ng = divide(self.config.num_query_groups, self.tp_size)
        np = divide(self.config.num_attention_heads, self.tp_size)
        query_projection_size = self.config.kv_channels * self.config.num_attention_heads
        hn = divide(query_projection_size, self.config.num_attention_heads)
        attn_dim = ng * (np//ng + 2) * hn

        # [b s] --> [s b]
        global_mask = multimodal_mask.transpose(0, 1).contiguous()

        s_tp, b, h = hidden_states.shape
        # output: [s b attn_h]
        output = torch.empty(
            (s_tp*self.tp_size, b, attn_dim), 
            dtype=hidden_states.dtype, 
            device=torch.cuda.current_device()
        )
        # bias: [s b 1]
        bias = torch.empty(
            (s_tp*self.tp_size, b, 1),
            dtype=hidden_states.dtype, 
            device=torch.cuda.current_device()
        ) if self.config.add_bias_linear else None,

        for mask, layer in zip(
            [global_mask, ~global_mask],
            [self.language, self.vision],
        ):
            # hidden_states: [s/tp b h]
            qkv_output, qkv_bias = layer(hidden_states)
            # qkv_output: [s b attn_h]
            # global_mask: [s b]
            output.view(-1, attn_dim)[mask.view(-1), :] = qkv_output.view(-1, attn_dim)[mask.view(-1), :]
            if self.config.add_bias_linear:
                bias.view(-1, 1)[mask.view(-1), :] = qkv_bias.view(-1, 1)[mask.view(-1), :]

            # # NOTE: cannot turn on sequence parallel
            # # hidden_states: [s b h]
            # partial_hidden_state = hidden_states.view(-1, h)[mask.view(-1), :].view(-1, b, h)
            # # partial_hidden_state: [s' b h]
            # qkv_output, qkv_bias = layer(partial_hidden_state)
            # # qkv_output: [s' b attn_h]
            # # mask: [s b]
            # # output: [s b attn_h]
            # output.view(-1, attn_dim)[mask.view(-1), :] = qkv_output.view(-1, attn_dim)
            # if self.config.add_bias_linear:
            #     bias.view(-1, 1)[mask.view(-1), :] = qkv_bias.view(-1, 1)

        # output: [s b attn_h]
        return output, bias


class MultiModalLinearProj(MultiModalLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, hidden_states: torch.Tensor, multimodal_mask: torch.Tensor):
        # hidden_states: [s b attn_h]
        # [b s] --> [s b]
        global_mask = multimodal_mask.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            # [s b] --> [s/tp b]
            local_mask = tensor_parallel.scatter_to_sequence_parallel_region(global_mask)
        else:
            local_mask = global_mask

        s, b, attn_dim = hidden_states.shape
        hidden_dim = self.config.hidden_size
        # output: [s/tp b h]
        output = torch.empty(
            (s//self.tp_size, b, hidden_dim),
            dtype=hidden_states.dtype, 
            device=torch.cuda.current_device()
        )
        # output: [s/tp b 1]
        bias = torch.empty(
            (s//self.tp_size, b, 1),
            dtype=hidden_states.dtype, 
            device=torch.cuda.current_device()
        ) if self.config.add_bias_linear else None

        for mask, layer in zip(
            [local_mask, ~local_mask],
            [self.language, self.vision],
        ):
            # hidden_states: [s b attn_h]
            proj_output, proj_bias = layer(hidden_states)
            # proj_output: [s/tp b h]
            # local_mask: [s/tp b]
            output.view(-1, hidden_dim)[mask.view(-1), :] = proj_output.view(-1, hidden_dim)[mask.view(-1), :]
            if self.config.add_bias_linear:
                bias.view(-1, 1)[mask.view(-1), :] = proj_bias.view(-1, 1)[mask.view(-1), :]

            # # NOTE: cannot turn on sequence parallel
            # # hidden_states: [s b attn_h]
            # partial_hidden_state = hidden_states.view(-1, attn_dim)[mask.view(-1), :].view(-1, b, attn_dim)
            # # partial_hidden_state: [s' b attn_h]
            # proj_output, proj_bias = layer(partial_hidden_state)
            # # proj_output: [s' b h]
            # # mask: [s b]
            # # output: [s b h]
            # output.view(-1, hidden_dim)[mask.view(-1), :] = proj_output.view(-1, hidden_dim)
            # if self.config.add_bias_linear:
            #     bias.view(-1, 1)[mask.view(-1), :] = proj_bias.view(-1, 1)

        # output: [s/tp b h]
        return output, bias
