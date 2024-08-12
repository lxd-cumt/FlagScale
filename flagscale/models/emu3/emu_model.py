from typing import Literal, Optional

import torch
from torch import Tensor

from megatron.training import get_args
from megatron.core import InferenceParams
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor

from flagscale.models.emu3.language_model_embedding import LanguageModelEmbedding


class EmuModel(GPTModel):
    """Emu Transformer multimodal language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        language_vocab_size (int): Language vocabulary size
        vision_vocab_size (int): Vision vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        vision_vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
    ) -> None:

        super().__init__(
            config,
            transformer_layer_spec,
            language_vocab_size + vision_vocab_size,
            max_sequence_length,
            pre_process,
            post_process,
            fp16_lm_cross_entropy,
            parallel_output,
            share_embeddings_and_output_weights,
            position_embedding_type,
            rotary_percent,
            rotary_base,
            seq_len_interpolation_factor,
        )

        if self.pre_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                language_vocab_size=language_vocab_size,
                vision_vocab_size=vision_vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        multimodal_mask: Tensor = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids, multimodal_mask=multimodal_mask)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        args = get_args()
        if args.use_multimodal_router_attn and args.use_multimodal_router_mlp:
            # Run decoder.
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                multimodal_mask=multimodal_mask,
                **(extra_block_kwargs or {}),
            )
        elif not args.use_multimodal_router_attn and not args.use_multimodal_router_mlp:
            # Run decoder.
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                **(extra_block_kwargs or {}),
            )
        else:
            raise ValueError("use_multimodal_router_attn and use_multimodal_router_mlp must be set all true or all false currently.")

        if not self.post_process:
            return hidden_states

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def freeze_language_model(self):
        self.requires_grad_(False)

        if self.pre_process:
            self.embedding.vision_embeddings.requires_grad_(True)

        if self.post_process:
            self.output_layer.requires_grad_(True)

        for layer in self.decoder.layers:
            layer.self_attention.linear_qkv.vision.requires_grad_(True)
            layer.self_attention.linear_proj.vision.requires_grad_(True)
            layer.mlp.vision.router.requires_grad_(True)
            for expert in layer.mlp.vision.experts.local_experts:
                expert.requires_grad_(True)


class EmuTransformerLayer(TransformerLayer):
    def forward(
        self,
        hidden_states,
        attention_mask,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        inference_params=None,
        packed_seq_params=None,
        multimodal_mask=None,
    ):
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Self attention.
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            multimodal_mask=multimodal_mask,
        )

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, torch.zeros_like(residual), self.hidden_dropout
            )

        # Optional Input Layer norm
        hidden_states = self.input_layernorm(hidden_states) + residual

        # Residual connection.
        residual = hidden_states

        # MLP.
        if multimodal_mask is not None:
            mlp_output_with_bias = self.mlp(hidden_states, multimodal_mask=multimodal_mask)
        else:
            mlp_output_with_bias = self.mlp(hidden_states)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, torch.zeros_like(residual), self.hidden_dropout
            )

        # Optional Layer norm post the cross-attention.
        hidden_states = self.pre_mlp_layernorm(hidden_states) + residual

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context
