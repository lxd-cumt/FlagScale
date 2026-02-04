## built-in

## third-party
import torch
import torch.nn as nn

# megatron-core
from megatron.core import tensor_parallel
from megatron.core.utils import get_tensor_model_parallel_group_if_none

# engram
from .engram_config import EngramConfig


class MultiHeadEmbedding(nn.Module):
    def __init__(self, engram_cfg: EngramConfig, list_of_N: list[int], D: int):
        super().__init__()
        self.engram_cfg = engram_cfg
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)

        # embeddings (parallel).
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
        self.reduce_scatter_embeddings = self.engram_cfg.sequence_parallel
        self.embedding = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=total_N,
            embedding_dim=D,
            init_method=self.engram_cfg.embedding_init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.engram_cfg,
            tp_group=self.tp_group,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids)

        if not self.reduce_scatter_embeddings:
            output = output.transpose(0, 1).contiguous()
        return output
