#!/usr/bin/env python3
"""
Comprehensive ArgumentGroupFactory conflict detection.
Checks ALL factory config classes against arguments_fs.py manual add_argument calls.
"""

import sys
import re
from pathlib import Path

# Import all factory config classes
from megatron.core.transformer import TransformerConfig, MLATransformerConfig
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.training.training_config import TrainingConfig, ValidationConfig, SchedulerConfig, LoggerConfig, CheckpointConfig
from megatron.training.common_config import ProfilingConfig, RNGConfig
from megatron.training.resilience_config import StragglerDetectionConfig, RerunStateMachineConfig

def get_factory_fields(config_class, exclude_list=None):
    """Get all fields from a dataclass config, respecting exclude list."""
    exclude_list = exclude_list or []
    fields = []

    # Get all fields including inherited ones
    if hasattr(config_class, '__dataclass_fields__'):
        for field_name in config_class.__dataclass_fields__:
            if field_name not in exclude_list:
                # Convert field_name to --arg-name format
                arg_name = '--' + field_name.replace('_', '-')
                fields.append(arg_name)

    return fields

def extract_manual_args(arguments_fs_path):
    """Extract all manual add_argument calls from arguments_fs.py."""
    content = Path(arguments_fs_path).read_text()

    # Match add_argument patterns
    pattern = r"\.add_argument\(\s*['\"](-{1,2}[a-zA-Z0-9_-]+)['\"]"
    matches = re.findall(pattern, content)

    return set(matches)

def main():
    # Define all factory configs with their exclude lists
    # These match the actual factory calls in arguments.py
    factories = [
        {
            'name': 'TransformerConfig',
            'class': TransformerConfig,
            'exclude': [
                'activation_func', 'add_bias_linear', 'add_qkv_bias', 'apply_query_key_layer_scaling',
                'apply_residual_connection_post_layernorm', 'attention_dropout', 'attention_softmax_in_fp32',
                'bias_activation_fusion', 'bias_dropout_fusion', 'clone_scatter_output_in_embedding',
                'deallocate_pipeline_outputs', 'ffn_hidden_size', 'first_pipeline_num_layers',
                'fp8', 'fp8_amax_compute_algo', 'fp8_amax_history_len', 'fp8_interval', 'fp8_margin',
                'fp8_wgrad', 'gated_linear_unit', 'hidden_dropout', 'init_method_std',
                'kv_channels', 'last_pipeline_num_layers', 'layernorm_epsilon', 'layernorm_zero_centered_gamma',
                'masked_softmax_fusion', 'normalization', 'num_attention_heads', 'num_layers',
                'num_moe_experts', 'num_query_groups', 'output_layer_init_method_std', 'persist_layer_norm',
                'pipeline_dtype', 'qk_layernorm', 'recompute_granularity', 'recompute_method',
                'recompute_num_layers', 'rotary_base', 'rotary_interleaved', 'rotary_percent',
                'seq_length', 'sequence_parallel', 'test_mode', 'use_cpu_initialization',
                'virtual_pipeline_model_parallel_size', 'window_size', 'num_layers_per_virtual_pipeline_stage',
                'overlap_p2p_comm', 'batch_p2p_comm', 'batch_p2p_sync', 'use_ring_exchange_p2p',
                'deallocate_pipeline_outputs', 'no_sync_func', 'grad_scale_func', 'enable_autocast',
                'autocast_dtype', 'variable_seq_lengths', 'deterministic_mode', 'allow_embedding_copy',
                'num_moe_experts', 'moe_router_load_balancing_type', 'moe_router_topk',
                'moe_aux_loss_coeff', 'moe_z_loss_coeff', 'moe_input_jitter_eps',
                'moe_token_dispatcher_type', 'moe_per_layer_logging', 'moe_expert_capacity_factor',
                'moe_pad_expert_input_to_capacity', 'moe_token_drop_policy', 'moe_layer_recompute',
                'moe_extended_tp', 'fp8_e4m3', 'fp8_hybrid', 'fp8_amax_recompute_algo',
                'clone_scatter_output_in_embedding', 'moe_grouped_gemm', 'moe_use_legacy_grouped_gemm',
                'moe_token_capacity_factor', 'moe_pad_expert_input_to_capacity_float',
                'moe_expert_capacity_factor_float', 'moe_router_pre_softmax', 'moe_router_pre_softmax_topk',
                'moe_router_pre_softmax_temperature', 'moe_router_pre_softmax_temperature_topk',
                'moe_router_pre_softmax_temperature_topk_temperature', 'moe_router_pre_softmax_temperature_topk_temperature_topk',
                'moe_router_pre_softmax_temperature_topk_temperature_topk_temperature',
            ]
        },
        {
            'name': 'MLATransformerConfig',
            'class': MLATransformerConfig,
            'exclude': []
        },
        {
            'name': 'TrainingConfig',
            'class': TrainingConfig,
            'exclude': []
        },
        {
            'name': 'ValidationConfig',
            'class': ValidationConfig,
            'exclude': []
        },
        {
            'name': 'SchedulerConfig',
            'class': SchedulerConfig,
            'exclude': []
        },
        {
            'name': 'LoggerConfig',
            'class': LoggerConfig,
            'exclude': []
        },
        {
            'name': 'CheckpointConfig',
            'class': CheckpointConfig,
            'exclude': []
        },
        {
            'name': 'ProfilingConfig',
            'class': ProfilingConfig,
            'exclude': []
        },
        {
            'name': 'RNGConfig',
            'class': RNGConfig,
            'exclude': []
        },
        {
            'name': 'StragglerDetectionConfig',
            'class': StragglerDetectionConfig,
            'exclude': []
        },
        {
            'name': 'RerunStateMachineConfig',
            'class': RerunStateMachineConfig,
            'exclude': []
        },
    ]

    # Collect all factory-generated args
    factory_args = {}
    for factory in factories:
        fields = get_factory_fields(factory['class'], factory['exclude'])
        factory_args[factory['name']] = fields

    # Extract manual args from arguments_fs.py
    arguments_fs_path = Path(__file__).parent / 'flagscale/train/megatron/training/arguments_fs.py'
    manual_args = extract_manual_args(arguments_fs_path)

    # Find conflicts
    conflicts = []
    for factory_name, fields in factory_args.items():
        for field in fields:
            if field in manual_args:
                conflicts.append({
                    'arg': field,
                    'factory': factory_name,
                })

    # Report results
    print(f"\n{'='*80}")
    print(f"ArgumentGroupFactory Conflict Detection")
    print(f"{'='*80}\n")

    print(f"Total factory-generated args: {sum(len(fields) for fields in factory_args.values())}")
    print(f"Total manual args in arguments_fs.py: {len(manual_args)}")
    print(f"Conflicts found: {len(conflicts)}\n")

    if conflicts:
        print("CONFLICTS (manual add_argument duplicates factory-generated arg):")
        print(f"{'='*80}")
        for conflict in sorted(conflicts, key=lambda x: x['arg']):
            print(f"  {conflict['arg']:<40} <- {conflict['factory']}")
        print(f"\nThese {len(conflicts)} arguments must be removed from arguments_fs.py")
        return 1
    else:
        print("✓ No conflicts detected")
        return 0

if __name__ == '__main__':
    sys.exit(main())
