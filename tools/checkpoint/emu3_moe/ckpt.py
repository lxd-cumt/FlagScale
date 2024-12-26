import sys
import copy
import torch
sys.path.append("..")
from utils import padding_vocab_size
from emu3_dense.ckpt import (
    get_hf_attn_ckpt,
    set_hf_embedding_ckpt,
    set_hf_attn_ckpt,
    set_hf_output_layer_ckpt,
    set_hf_final_norm_ckpt,
    get_embedding_ckpt,
    get_attn_ckpt,
    get_output_layer_ckpt,
    get_final_norm_ckpt,
    set_embedding_ckpt,
    set_attn_ckpt,
    set_output_layer_ckpt,
    set_final_norm_ckpt,
)

from mixtral.ckpt import (
    get_hf_mlp_ckpt,
    set_hf_mlp_ckpt,
    get_mlp_ckpt,
    set_mlp_ckpt,
)

