# Copyright (c) 2025, FlagScale Team. All rights reserved.

"""FlagScale-specific tokenizer types and build_tokenizer wrapper.

Standard tokenizer types are delegated to the upstream factory at
megatron.core.tokenizers.utils.build_tokenizer. This module only handles
FlagScale-specific tokenizer types that have no upstream equivalent.
"""

from megatron.core.tokenizers.base_tokenizer import MegatronTokenizerBase
from megatron.core.tokenizers.utils.build_tokenizer import (
    build_tokenizer as _upstream_build_tokenizer,
    vocab_size_with_padding,
)

from .gpt2_tokenization import AquilaTokenizer
from .rwkv_tokenization import RWKVTokenizer


# FlagScale-specific tokenizer type names
_FLAGSCALE_TOKENIZER_TYPES = {
    'AquilaTokenizerFS',
    'HFTokenizerFS',
    'Llama3TokenizerFS',
    'QwenTokenizerFS',
    'HFTokenizersTokenizerFS',
    'Qwen2TokenizerFS',
    'Qwen2VLTokenizer',
    'RWKVTokenizer',
}


def build_tokenizer(args, **kwargs):
    """Initialize tokenizer.

    Delegates standard types to upstream megatron.core.tokenizers.utils.build_tokenizer.
    Handles FlagScale-specific types locally.
    """
    if args.tokenizer_type not in _FLAGSCALE_TOKENIZER_TYPES:
        return _upstream_build_tokenizer(args, **kwargs)

    from megatron.training.utils import print_rank_0
    print_rank_0('> building {} tokenizer ...'.format(args.tokenizer_type))

    if args.tokenizer_type == 'AquilaTokenizerFS':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        assert args.special_tokens_file is not None
        tokenizer = _AquilaTokenizerFS(args.vocab_file, args.merge_file,
                                       args.special_tokens_file)
    elif args.tokenizer_type == "HFTokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _HFTokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "Llama3TokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _Llama3TokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "QwenTokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _QwenTokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "HFTokenizersTokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _HFTokenizersTokenizerFS(args.tokenizer_path)
    elif args.tokenizer_type == "Qwen2TokenizerFS":
        assert args.tokenizer_path is not None
        tokenizer = _Qwen2TokenizerFS(args.tokenizer_path, args)
    elif args.tokenizer_type == 'Qwen2VLTokenizer':
        assert args.tokenizer_path is not None
        tokenizer = _Qwen2VLTokenizer(args.tokenizer_path, args.extra_vocab_size)
        args.padded_vocab_size = tokenizer.vocab_size  # no padding
    elif args.tokenizer_type == "RWKVTokenizer":
        assert args.tokenizer_path is not None, "tokenizer_path must be provided for RWKV tokenizer"
        tokenizer = _RWKVTokenizerFS(args.tokenizer_path)
    else:
        raise NotImplementedError('{} tokenizer is not implemented.'.format(args.tokenizer_type))

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


# ---------------------------------------------------------------------------
# FlagScale-specific tokenizer classes
#
# All inherit from MegatronTokenizerBase (upstream 0.17.0 ABC).
# The base class signature is __init__(self, path, config, **kwargs).
# ---------------------------------------------------------------------------


class _FlagScaleTokenizerBase(MegatronTokenizerBase):
    """Convenience base for FlagScale tokenizers.

    Provides default no-op implementations for abstract methods that
    some FlagScale tokenizers don't need (e.g. apply_chat_template).
    """

    def __init__(self, path, config=None, **kwargs):
        if config is None:
            config = {}
        super().__init__(path=path, config=config, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("This tokenizer does not support chat templates.")


class _AquilaTokenizerFS(_FlagScaleTokenizerBase):
    """Aquila tokenizer using GPT2 BPE with custom special tokens."""

    def __init__(self, vocab_file, merge_file, special_tokens_file):
        super().__init__(path=vocab_file)

        special_tokens = []
        if special_tokens_file:
            special_tokens = open(special_tokens_file, encoding='utf-8').read().split('\n')[:-1]

        self.tokenizer = AquilaTokenizer(vocab_file, merge_file, errors='replace',
                                         special_tokens=special_tokens, max_len=None)
        self.eod_id = self.tokenizer.encoder['</s>']
        self.cls_id = self.tokenizer.encoder['[CLS]']
        self.pad_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _HFTokenizerFS(_FlagScaleTokenizerBase):
    """HuggingFace AutoTokenizer wrapper."""

    def __init__(self, tokenizer_path):
        super().__init__(path=tokenizer_path)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.eod_id = self.tokenizer.eos_token_id
        self.cls_id = self.tokenizer.bos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self._inv_vocab = None

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        vocab = self.vocab
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in vocab.items()}
        return self._inv_vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _Llama3TokenizerFS(_HFTokenizerFS):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + len(self.tokenizer.get_added_vocab())


class _QwenTokenizerFS(_HFTokenizerFS):
    """Adapted Qwen tokenizer."""

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode('<|extra_204|>')[0]
        self.cls_id = self.tokenizer.encode('<|extra_203|>')[0]
        self.pad_id = self.tokenizer.encode('<|endoftext|>')[0]


class _HFTokenizersTokenizerFS(_FlagScaleTokenizerBase):
    """Tokenizer using HuggingFace tokenizers library (not transformers)."""

    def __init__(self, json_file):
        super().__init__(path=json_file)

        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(json_file)

        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")

        self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = self.tokenizer.token_to_id("<|padding|>")

        self._inv_vocab = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        vocab = self.vocab
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in vocab.items()}
        return self._inv_vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def pad(self):
        return self.pad_id


class _Qwen2TokenizerFS(_HFTokenizerFS):
    """Adapted Qwen2 tokenizer with explicit vocab_size from args."""

    def __init__(self, tokenizer_path, args):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode('<|extra_204|>')[0]
        self.cls_id = self.tokenizer.encode('<|extra_203|>')[0]
        self.pad_id = self.tokenizer.encode('<|endoftext|>')[0]
        assert args.vocab_size is not None
        self._vocab_size = args.vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size


class _Qwen2VLTokenizer(_FlagScaleTokenizerBase):
    """Full Qwen2-VL tokenizer with AutoProcessor and chat template support."""

    def __init__(self, tokenizer_path, extra_vocab_size):
        super().__init__(path=tokenizer_path)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            use_fast=True,
            split_special_tokens=False,
            trust_remote_code=True,
            revision="main",
            token=None,
        )
        self.extra_vocab_size = extra_vocab_size
        self.special_tokens_map = {
            k: v for k, v in zip(
                self.tokenizer.all_special_tokens,
                self.tokenizer.all_special_ids,
            )
        }
        self.image_token = '<|image_pad|>'
        self.video_token = '<|video_pad|>'
        self.vision_start_token = '<|vision_start|>'
        self.vision_end_token = '<|vision_end|>'

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            revision="main",
            token=None,
        )
        self.chat_template = self.processor.chat_template

    def __call__(self, text, return_tensors=None,
                 padding=None, max_length=None, truncation=None, add_special_tokens=None):
        return self.tokenizer(
            text, return_tensors=return_tensors, padding=padding,
            max_length=max_length, truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

    def apply_chat_template(self, conversations, tokenize: bool = True, **kwargs):
        return self.tokenizer.apply_chat_template(
            conversations, tokenize=tokenize, chat_template=self.chat_template, **kwargs
        )

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def image_token_id(self):
        return self.special_tokens_map[self.image_token]

    @property
    def video_token_id(self):
        return self.special_tokens_map[self.video_token]

    @property
    def vision_start_token_id(self):
        return self.special_tokens_map[self.vision_start_token]

    @property
    def vision_end_token_id(self):
        return self.special_tokens_map[self.vision_end_token]

    def encode(self, x):
        return self.tokenizer.encode(x)


class _RWKVTokenizerFS(_FlagScaleTokenizerBase):
    """RWKV Trie-based tokenizer, wrapped for MegatronTokenizerBase compatibility."""

    def __init__(self, tokenizer_path):
        super().__init__(path=tokenizer_path)
        self._rwkv = RWKVTokenizer(tokenizer_path)
        self.eod = self._rwkv.eod

    @property
    def vocab_size(self):
        return self._rwkv.vocab_size

    @property
    def vocab(self):
        return self._rwkv.token2idx

    @property
    def inv_vocab(self):
        return self._rwkv.idx2token

    def tokenize(self, text):
        return self._rwkv.encode(text)

    def detokenize(self, token_ids):
        return self._rwkv.decode(token_ids)

    def apply_chat_template(self, *args, **kwargs):
        raise NotImplementedError("RWKVTokenizer does not support chat templates.")
