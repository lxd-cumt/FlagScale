from copy import deepcopy

import torch

from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.emu3.emu_layer_specs import MultiModalSubmodules


class MultiModalMoELayer(MegatronModule):
    def __init__(self, config: TransformerConfig, submodules: MultiModalSubmodules):
        super().__init__(config=config)

        args = get_args()
        def _set_moe_config(config, idx=0):
            split = args.multimodal_num_experts_split
            config.num_moe_experts = abs(idx * args.num_experts - split) if split  else args.num_experts
            config.moe_router_load_balancing_type = args.multimodal_moe_router_load_balancing_type[idx]
            config.moe_router_topk = args.multimodal_moe_router_topk[idx]
            config.moe_aux_loss_coeff = args.multimodal_moe_aux_loss_coeff[idx]
            config.moe_z_loss_coeff = args.multimodal_moe_z_loss_coeff[idx]
            config.moe_input_jitter_eps = args.multimodal_moe_input_jitter_eps[idx]
            config.moe_token_dropping = args.multimodal_moe_token_dropping[idx]

        _set_moe_config(config, 0)
        self.language = build_module(submodules.language, config=deepcopy(config))
        _set_moe_config(config, 1)
        self.vision = build_module(submodules.vision, config=deepcopy(config))

    def forward(self, hidden_states: torch.Tensor, multimodal_mask: torch.Tensor):

        def _padding(hidden_state, padded_size):
            origin_size = hidden_state.shape[0]
            assert origin_size <= padded_size
            if origin_size < padded_size:
                return torch.nn.functional.pad(
                    hidden_state,
                    (0, 0, 0, 0, 0, padded_size - origin_size),
                    "constant", 
                    0
                )
            return hidden_state

        # [b s] --> [s b]
        global_mask = multimodal_mask.transpose(0, 1).contiguous()
        if self.config.sequence_parallel:
            # [s b] --> [s/tp b]
            local_mask = tensor_parallel.scatter_to_sequence_parallel_region(global_mask)
        else:
            local_mask = global_mask

        s, b, h = hidden_states.shape # [s/tp b h]
        output = torch.empty_like(hidden_states)
        bias = torch.empty_like(hidden_states) if self.config.add_bias_linear else None

        for mask, layer in zip(
            [local_mask, ~local_mask], 
            [self.language, self.vision]
        ):
            # hidden_states: [s/tp b h]
            # mask: [s/tp b]
            # partial_hidden_state: [s' 1 h]
            partial_hidden_state = hidden_states.view(-1, h)[mask.view(-1), :].view(-1, 1, h)
            seq_len = partial_hidden_state.shape[0]

            with torch.no_grad():
                global_seq_lens = tensor_parallel.gather_from_sequence_parallel_region_to_moe(
                    torch.tensor([seq_len], 
                    device=torch.cuda.current_device())
                )
                global_max_seq_len = torch.max(global_seq_lens).item()

            if global_max_seq_len > 0:
                # padded_hidden_state: [padded_s 1 h]
                padded_hidden_state = _padding(partial_hidden_state, global_max_seq_len)
                partial_output, partial_bias = layer(padded_hidden_state)
                output.view(-1, h)[mask.view(-1), :] = partial_output[:seq_len, :].view(-1, h)
                if self.config.add_bias_linear:
                    bias.view(-1, 1)[mask.view(-1), :] = partial_bias[:seq_len, :].view(-1, 1)

        return output, bias
