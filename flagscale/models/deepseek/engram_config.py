## built-in
from typing import List, Optional
from dataclasses import field, dataclass

## megatron-core
from megatron.core.transformer import MLATransformerConfig

@dataclass
class EngramConfig(MLATransformerConfig):
    engram_tokenizer_name_or_path: Optional[str] = None
    engram_vocab_size: Optional[List[int]] = None
    max_ngram_size: int = 1
    n_embed_per_ngram: Optional[int] = None
    n_head_per_ngram: int = 1
    engram_layer_ids: Optional[List[int]] = None
    engram_pad_id: int = 0
    engram_seed: int = 0
    engram_kernel_size: int = 1
    engram_hc_mult: int = 1
