## built-in
from typing import Optional, Dict
from torch import Tensor
import torch

## megatron-core
from megatron.core.models.gpt import GPTModel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import deprecate_inference_params

## engram
from .engram_transformer_layer import EngramTransformerBlock
from .ngram_hash import get_or_create_hash_mapping

class EngramModel(GPTModel):
    def __init__(
        self, *args, **kwargs
    ):
        # NOTE: We temporarily replace TransformerBlock with EngramTransformerBlock
        # during super().__init__() to avoid creating decoder twice.
        # This is necessary because GPTModel.__init__ hardcodes TransformerBlock.
        # The replacement is scoped to this initialization only.
        from megatron.core.transformer.transformer_block import TransformerBlock
        import megatron.core.models.gpt.gpt_model as gpt_module
        
        original_block = gpt_module.TransformerBlock
        gpt_module.TransformerBlock = EngramTransformerBlock
        
        try:
            super().__init__(*args, **kwargs)
            # self.decoder is now EngramTransformerBlock, no need to recreate
        finally:
            gpt_module.TransformerBlock = original_block

        self.engram_hash = get_or_create_hash_mapping(
            engram_vocab_size=self.config.engram_vocab_size,
            max_ngram_size = self.config.max_ngram_size,
            n_embed_per_ngram = self.config.n_embed_per_ngram,
            n_head_per_ngram = self.config.n_head_per_ngram,
            layer_ids = self.config.engram_layer_ids,
            tokenizer_name_or_path=self.config.engram_tokenizer_name_or_path,
            pad_id = self.config.engram_pad_id,
            seed = self.config.engram_seed,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        
        assert input_ids is not None, "Input ids can not be None for EngramModel"
        inference_context = deprecate_inference_params(inference_context, inference_params)

        preproc_output = self._preprocess(
            input_ids=input_ids,
            position_ids=position_ids,
            decoder_input=decoder_input,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
        )

        (decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset) = (
            preproc_output[:5]
        )

        rotary_pos_cos_sin = preproc_output[5] if len(preproc_output) == 6 else None

        engram_hash_input_ids = self.engram_hash.hash(input_ids)

        # Run decoder with engram
        hidden_states = self.decoder(
            input_ids=input_ids,
            engram_hash_input_ids=engram_hash_input_ids,
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ):
        raise NotImplementedError("Sharded state dict is not supported for EngramModel")
