# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain GPT."""

import os
import math
import torch
from functools import partial
from contextlib import nullcontext
import inspect

from typing import List, Optional, Tuple, Union
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import StragglerDetector
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

from flagscale.models.emu3.emu_layer_specs import get_emu_with_transformer_engine_spec
from flagscale.datasets.emu_dataset import EmuDataset
from flagscale.models.emu3.emu_model import EmuModel
from flagscale.train.arguments import add_flagscale_args
from flagscale.train.train import pretrain


stimer = StragglerDetector()

def small_init_init_method(dim):
    """Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2010), using a normal distribution."""
    std = math.sqrt(2 / (5 * dim))

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def wang_init_method(n_layers, dim):
    std = 2 / n_layers / math.sqrt(dim)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    if args.split == "None":
        args.split = None
    use_te = args.transformer_impl == "transformer_engine"
    assert use_te

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    config.init_method = small_init_init_method(config.hidden_size)
    config.output_layer_init_method = wang_init_method(config.num_layers, config.hidden_size)

    assert args.use_legacy_models is False

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_emu_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.multi_latent_attention, args.fp8)

    build_model_context = nullcontext
    build_model_context_args = {}
    if args.fp8_param_gather:
        try:
            from transformer_engine.pytorch import fp8_model_init

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True

            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True
        except:
            raise RuntimeError("--fp8-param-gather requires `fp8_model_init` from TransformerEngine, but not found.")

    with build_model_context(**build_model_context_args):
        model = EmuModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            language_vocab_size=args.language_padded_vocab_size,
            vision_vocab_size=args.vision_padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling
        )

    if args.multimodal_freeze_language_parameters:
        model.freeze_language_model()
        no_grad_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                no_grad_params.append(name)

        print_rank_0(f" > Parameters [{no_grad_params}] are frozen and will not updated.")

    return model


def get_batch_on_this_tp_rank(data_iterator):
    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if data_iterator is not None:
            data = next(data_iterator)
            while "loss_mask" in data and data["loss_mask"].sum() == 0:
                print_rank_0("loss_mask is zero, no valid tokens, read next data")
                data = next(data_iterator)
        else:
            data = None

        batch = {
            'tokens': data["tokens"].cuda(non_blocking = True),
            'labels': data["labels"].cuda(non_blocking = True),
            'loss_mask': data["loss_mask"].cuda(non_blocking = True),
            'attention_mask': None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True),
            'position_ids': data["position_ids"].cuda(non_blocking = True),
            "multimodal_mask": data['tokens'].cuda(non_blocking = True) < args.language_vocab_size
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast(batch['multimodal_mask'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast(batch['multimodal_mask'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['multimodal_mask'])

        else:
            _broadcast(batch['multimodal_mask'])

    else:

        tokens=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
        labels=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
        loss_mask=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.float32 , device = torch.cuda.current_device())
        if args.create_attention_mask_in_dataloader:
            attention_mask=torch.empty(
                (args.micro_batch_size,1,args.seq_length,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device()
            )
        else:
            attention_mask=None
        position_ids=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.int64 , device = torch.cuda.current_device())
        multimodal_mask=torch.empty((args.micro_batch_size,args.seq_length), dtype = torch.bool , device = torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            _broadcast(multimodal_mask)

        elif mpu.is_pipeline_first_stage():
            labels=None
            loss_mask=None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            _broadcast(multimodal_mask)

        elif mpu.is_pipeline_last_stage():
            tokens=None
            position_ids=None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(multimodal_mask)

        else:
            tokens=None
            labels=None
            loss_mask=None
            attention_mask=None
            position_ids=None

            _broadcast(multimodal_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'multimodal_mask': multimodal_mask
        }

    return batch


def get_batch(data_iterator):
    """Generate a batch."""

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def multimodal_loss_func(labels: torch.Tensor, loss_mask: torch.Tensor, logits: torch.Tensor):
    """Multimodal loss function.

    Args:

        labels  (torch.Tensor): Used to compute loss with logits
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        logits (torch.Tensor): The output tensor of transformer

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()

    labels = labels.transpose(0, 1).contiguous() # [b s] => [s b]
    logits = logits.transpose(0, 1).contiguous() # [b s h] => [s b h]

    losses = tensor_parallel.vocab_parallel_cross_entropy(logits.float(), labels)
    losses = losses.transpose(0, 1).contiguous().float()

    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if args.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss[0].isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * args.context_parallel_size,
        local_num_tokens,
        {'lm loss': (reporting_loss[0], reporting_loss[1])},
    )


def get_parallel_video_tokens(tokens):
    if tokens is None:
        return tokens
    args = get_args()
    img_id, boi_id, eoi_id = args.multimodal_visual_start_end_tokens

    for i, sample_tokens in enumerate(tokens):
        num_boi = len(torch.where(sample_tokens == boi_id)[0])
        if num_boi != 1:
            continue
        img_start, vid_start = torch.where(sample_tokens == img_id)[0][-2:]
        frame0 = sample_tokens[img_start + 1:vid_start]
        frame0 = frame0[frame0 > eoi_id]
        frame_size = len(frame0)
        frames = sample_tokens[vid_start + 1:]
        frames = frames[frames > eoi_id]

        tokens[i, vid_start:vid_start + len(frames) + 1] = \
            torch.cat([
                frame0, 
                frames[:len(frames) - frame_size],
                frame0.new([img_id])
            ], dim=0)

    return tokens


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids, multimodal_mask = get_batch(
            data_iterator)
        # tokens = get_parallel_video_tokens(tokens)
    timers('batch-generator').stop()

    with stimer:
        # NOTE: When labels is None, the output_tensor is logits of last transformer layer and its shape is [b, s, h]
        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=None, multimodal_mask=multimodal_mask)

    return output_tensor, partial(multimodal_loss_func, labels, loss_mask)


def is_dataset_built_on_rank():
    # NOTE: every tp_rank=0 build dataset
    return mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        renormalize_blend_weights=args.renormalize_blend_weights,
        split=args.split,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        s3_cache_path=args.s3_cache_path,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    assert args.mock_data is False
    dataset_type = EmuDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_flagscale_args,
    )
