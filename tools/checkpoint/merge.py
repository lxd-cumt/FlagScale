import os
import sys
import argparse
import importlib
import torch
import re

root_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                    os.path.pardir,
                    os.path.pardir))
sys.path.append(os.path.join(root_path, "megatron"))
sys.path.append(root_path)

from megatron.core import mpu
from megatron.training.arguments import parse_args, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.checkpointing import _load_base_checkpoint, load_args_from_checkpoint
from megatron.training.checkpointing import save_checkpoint, get_checkpoint_name

from flagscale.train.arguments import add_flagscale_args


def loader_args(load_dir):
    sys.argv = [
        'script.py',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--use-mcore-models',
        '--transformer-impl', 'transformer_engine',
        '--load', load_dir
    ]
    margs = parse_args(add_flagscale_args)
    margs, checkpoint_args = load_args_from_checkpoint(margs)

    def _set_arg(arg_name):
        ckpt_value = getattr(checkpoint_args, arg_name, None)
        setattr(margs, arg_name, ckpt_value)

    _set_arg("expert_model_parallel_size")
    _set_arg("num_experts")
    _set_arg("sequence_parallel")
    _set_arg("bf16")
    _set_arg("fp16")
    _set_arg("params_dtype")
    return margs, checkpoint_args


def saver_args(args, save_dir):
    sys.argv = [
        'script.py',
        '--no-masked-softmax-fusion',
        '--no-bias-gelu-fusion',
        '--no-bias-dropout-fusion',
        '--no-async-tensor-model-parallel-allreduce',
        '--use-cpu-initialization',
        '--micro-batch-size', '1',
        '--no-load-optim',
        '--no-load-rng',
        '--no-save-optim',
        '--no-save-rng',
        '--no-initialization',
        '--use-mcore-models',
        '--transformer-impl', 'transformer_engine',
        '--save-interval', '1',
        '--save', save_dir
    ]
    if args.use_attn_type == 'aquila3_dense':
        sys.argv.append('--add-qkv-bias')
    margs = parse_args(add_flagscale_args) 
    return margs


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Merge checkpoint")
    # convert args
    parser.add_argument('--src-model-types', type=str, default=['aquila3_dense', 'emu3_dense'], nargs=2, required=True,
                        choices=['aquila3_dense', 'emu3_dense'],
                        help='Type of the model.')
    parser.add_argument('--dst-model-type', type=str, default=['emu3_moe'], required=True,
                        choices=['emu3_moe'])
    parser.add_argument('--load-dirs', type=str, nargs=2, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--use-attn-type', type=str, required=True,
                        choices=['aquila3_dense', 'emu3_dense', 'mean'],
                        help='Attention checkpoint use aquila3_dense or emu3_dense')
    parser.add_argument("--target-num-experts", type=int, required=True)
    parser.add_argument("--target-num-experts-split", type=int, required=True)
    args = parser.parse_args()

    margs0, checkpoint_args0 = loader_args(args.load_dirs[0])
    print("margs0:", margs0)
    margs1, checkpoint_args1 = loader_args(args.load_dirs[1])
    print("margs1:", margs1)

    for attr in vars(margs0):
        if attr in [
            'load', 'padded_vocab_size', 'add_qkv_bias', 'seq_length', 
            'tokenizer_type', 'iteration', 'max_position_embeddings'
        ]:
            continue
        assert getattr(margs0, attr) == getattr(margs1, attr), f"margs0.{attr} != margs1.{attr}"

    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    assert margs1.expert_model_parallel_size == 1
    os.environ["WORLD_SIZE"] = f'{margs1.tensor_model_parallel_size * margs1.pipeline_model_parallel_size}'
    dst_model_plugin = importlib.import_module(args.dst_model_type + ".model")
    saver_margs = saver_args(args, args.save_dir)
    saver_margs.num_experts = args.target_num_experts
    saver_margs.params_dtype = margs1.params_dtype
    args_to_keep = [
        "num_experts", 'add_qkv_bias', 'params_dtype', 'world_size', 
        'num_layers_per_virtual_pipeline_stage', 'virtual_pipeline_model_parallel_size',
        'masked_softmax_fusion', 'bias_gelu_fusion', 'bias_dropout_fusion',
        'sequence_parallel', 'async_tensor_model_parallel_allreduce',
        'no_load_optim', 'no_load_rng', 'no_save_optim', 'no_save_rng',
        'vocab_file', 'tokenizer_model',
        'save_interval', 'save',
        'perform_initialization', 'use_cpu_initialization',
        'recompute_granularity', 'recompute_num_layers', 'recompute_method',
        'encoder_num_layers', 'encoder_seq_length',
        'distribute_saved_activations',
        'train_iters', 'lr_decay_iters', 'lr_warmup_iters', 'lr_warmup_fraction',
        'start_weight_decay', 'end_weight_decay'
    ]
    for arg, value in vars(checkpoint_args1).items():
        if arg in args_to_keep:
            continue
        if not hasattr(saver_margs, arg):
            print(f"Checkpoint had argument {arg} but new arguments does not have this.")
            setattr(saver_margs, arg, value)
            continue
        if getattr(saver_margs, arg) != value:
            print(f"Overwriting default {arg} value {getattr(saver_margs, arg)} with value from checkpoint {value}.")
            setattr(saver_margs, arg, value)

    print("saver_margs:", saver_margs)
    saver_margs = validate_args(saver_margs)
    set_global_variables(saver_margs, build_tokenizer=False)

    tp_size = saver_margs.tensor_model_parallel_size
    pp_size = saver_margs.pipeline_model_parallel_size
    ep_size = saver_margs.expert_model_parallel_size
    mpu.set_tensor_model_parallel_world_size(tp_size)
    mpu.set_pipeline_model_parallel_world_size(pp_size)
    mpu.set_expert_model_parallel_world_size(ep_size)

    def get_models(count, dtype):
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        models = [dst_model_plugin.get_mg_model(dtype, pre_process, post_process) for _ in range(count)]
        return models

    mpu.set_pipeline_model_parallel_rank(0)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_expert_model_parallel_rank(0)
    models = get_models(tp_size * ep_size, saver_margs.params_dtype)

    for pp_rank in range(pp_size):
        mpu.set_pipeline_model_parallel_rank(pp_rank)
        if pp_rank > 0:
            models = get_models(tp_size * ep_size, saver_margs.params_dtype)

        for tp_rank in range(tp_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)

            for ep_rank in range(ep_size):
                mpu.set_expert_model_parallel_rank(ep_rank)

                state_dict0, _, _ = _load_base_checkpoint(args.load_dirs[0], margs0, rank0=False)
                state_dict1, _, _ = _load_base_checkpoint(args.load_dirs[1], margs1, rank0=False)
                state_dict0 = state_dict0["model"]
                state_dict1 = state_dict1["model"]

                model = models[ep_rank * tp_size + tp_rank]
                # embedding and output_layer
                if pp_rank == 0:
                    model.embedding.word_embeddings.weight.data.copy_(state_dict1["embedding.word_embeddings.weight"])
                    model.embedding.vision_embeddings.weight.data.copy_(state_dict1["embedding.vision_embeddings.weight"])
                if pp_rank == pp_size - 1:
                    model.output_layer.weight.data.copy_(state_dict1["output_layer.weight"])

                # transformer layer
                for i, layer in enumerate(model.decoder.layers):
                    print(f"process transformer layer {i}")

                    # common
                    layer.input_layernorm.weight.data.copy_(state_dict1[f"decoder.layers.{i}.input_layernorm.weight"])
                    layer.pre_mlp_layernorm.weight.data.copy_(state_dict1[f"decoder.layers.{i}.pre_mlp_layernorm.weight"])

                    # self-attn
                    if args.use_attn_type == "aquila3_dense":
                        layer.self_attention.linear_qkv.weight.data.copy_(state_dict0[f'decoder.layers.{i}.self_attention.linear_qkv.weight'])
                        layer.self_attention.linear_proj.weight.data.copy_(state_dict0[f'decoder.layers.{i}.self_attention.linear_proj.weight'])
                        layer.self_attention.linear_qkv.bias.data.copy_(state_dict0[f'decoder.layers.{i}.self_attention.linear_qkv.bias'])
                    elif args.use_attn_type == "emu3_dense":
                        layer.self_attention.linear_qkv.weight.data.copy_(state_dict1[f'decoder.layers.{i}.self_attention.linear_qkv.weight'])
                        layer.self_attention.linear_proj.weight.data.copy_(state_dict1[f'decoder.layers.{i}.self_attention.linear_proj.weight'])
                    else: # mean
                        layer.self_attention.linear_qkv.weight.data.copy_((state_dict0[f'decoder.layers.{i}.self_attention.linear_qkv.weight'] + state_dict1[f'decoder.layers.{i}.self_attention.linear_qkv.weight']) / 2)
                        layer.self_attention.linear_proj.weight.data.copy_((state_dict0[f'decoder.layers.{i}.self_attention.linear_qkv.bias'] + state_dict1[f'decoder.layers.{i}.self_attention.linear_qkv.bias']) / 2)

                    # mlp (layer.mlp.router.weight is random initialized in get_model func)
                    for j, expert in enumerate(layer.mlp.experts.local_experts):
                        if j < args.target_num_experts_split:
                            expert.linear_fc1.weight.data.copy_(state_dict0[f"decoder.layers.{i}.mlp.linear_fc1.weight"])
                            expert.linear_fc2.weight.data.copy_(state_dict0[f"decoder.layers.{i}.mlp.linear_fc2.weight"])
                        else:
                            expert.linear_fc1.weight.data.copy_(state_dict1[f"decoder.layers.{i}.mlp.linear_fc1.weight"])
                            expert.linear_fc2.weight.data.copy_(state_dict1[f"decoder.layers.{i}.mlp.linear_fc2.weight"])

                checkpoint_name = get_checkpoint_name(saver_margs.save, iteration=1)
                print(f"megtron model is saving to {checkpoint_name} ...")
                # set iteration to 1
                save_checkpoint(1, [model], None, None, num_floating_point_operations_so_far=0)
