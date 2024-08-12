from typing import Dict, List, Union, Tuple, Optional
import os
import glob
import numpy
import torch

from megatron.training import get_args
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, _get_ltor_masks_and_position_ids
from megatron.core.datasets.indexed_dataset import IndexedDataset


class ConcatedIndexedDataset(IndexedDataset):
    def __init__(self, datasets) -> None:
        self.path_prefix = datasets[-1].path_prefix
        self.datasets = datasets
        self.offsets = [0]
        for dataset in datasets:
            self.offsets.append(self.offsets[-1] + len(dataset))

    def __del__(self) -> None:
        for dataset in self.datasets:
            del dataset

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(
        self, idx: Union[int, numpy.integer, slice]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]:
        for i, size in enumerate(self.offsets[1:]):
            if idx < size:
                break
        return self.datasets[i][idx - self.offsets[i]]

    def get(self, idx: int, offset: int = 0, length: Optional[int] = None) -> numpy.ndarray:
        for i, size in enumerate(self.offsets[1:]):
            if idx < size:
                break
        
        return self.datasets[i].get(idx - self.offsets[i], offset, length)

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        return numpy.concatenate([dataset.sequence_lengths for dataset in self.datasets])

    @property
    def document_indices(self) -> numpy.ndarray:
        return numpy.concatenate([
            dataset.document_indices[:-1] + offset
            for dataset, offset in zip(self.datasets, self.offsets)
        ] + [numpy.array([len(self),])])


class EmuDataset(GPTDataset):
    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        if os.path.isfile(dataset_path + '.bin'):
            return IndexedDataset(
                dataset_path, multimodal=False, mmap=config.mmap_bin_files,
            )
        datasets = []

        dataset_paths = glob.glob(os.path.join(dataset_path, '*.bin'))
        dataset_paths = sorted(
            dataset_paths, key=lambda x:os.path.basename(x),
        )
        for file in dataset_paths:
            datasets.append(
                IndexedDataset(
                    file[:-4], multimodal=False, mmap=config.mmap_bin_files,
                )
            )
        return ConcatedIndexedDataset(datasets)

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        if idx is None:
            # Batch padding sequence so the index does not matter
            text, _ = self._query_document_sample_shuffle_indices(0)
        else:
            text, _ = self._query_document_sample_shuffle_indices(idx)

        args = get_args()
        if args.multimodal_visual_start_end_tokens is not None:
            _, sov, eov = args.multimodal_visual_start_end_tokens
            text = text.tolist()
            if text.count(sov) <= 1:
                if text[-1] == self.config.tokenizer.cls:
                    text[-1] = self.config.tokenizer.pad
                text = numpy.array(text)
                # print('video', text[:10], '...', text[-10:])
            else:
                text_len = len(text)
                first_sov = text.index(sov) if sov in text else text_len
                first_eov = text.index(eov) if eov in text else text_len
                if first_sov > first_eov:
                    text = text[first_eov + 1:]
                text = text[::-1]
                last_sov = text.index(sov) if sov in text else text_len
                last_eov = text.index(eov) if eov in text else text_len
                if last_sov < last_eov:
                    text = text[last_sov + 1:]
                text = text[::-1]

                if text[0] == self.config.tokenizer.eod:
                    text = text[1:]

                text = numpy.pad(
                    text,
                    [(0, text_len - len(text))],
                    'constant',
                    constant_values=self.config.tokenizer.pad, # NOTE: current pad is <endoftext>
                )
                # print('image', text[:10], '...', text[-10:])

        text = torch.from_numpy(text).long()
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        assert not torch.any(
            tokens >= self.config.tokenizer.vocab_size
        ), "An input token is out of bounds of the tokenizer vocabulary"

        attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )

        # For padded sequences, mask the loss
        pad_id = self._pad_token_id
        img_token_id, soi_token_id, eoi_token_id = args.multimodal_visual_start_end_tokens

        if args.multimodal_token_dropping > 0:
            random_tensor = torch.rand(tokens.shape, device=tokens.device)
            mask = (tokens > eoi_token_id) & (random_tensor < args.multimodal_token_dropping)
            tokens[mask] = self._pad_token_id

        for token_id in [pad_id, img_token_id, soi_token_id, eoi_token_id]:
            loss_mask[labels == token_id] = 0.0

        _cnt_soi = (labels == soi_token_id).long().cumsum(dim=-1)
        _cnt_eoi = (labels == eoi_token_id).long().cumsum(dim=-1)
        within_visual_context = _cnt_soi > _cnt_eoi

        loss_mask[within_visual_context] = 0.
        # for no lang loss
        # loss_mask[labels <= eoi_token_id] = 0.
        loss_mask[labels > eoi_token_id] = args.multimodal_visual_loss_weight

        loss_mask[tokens == self._pad_token_id] = 0.0
        loss_mask[labels == self._pad_token_id] = 0.0

        # for SFT:
        # loss_mask[_cnt_soi==0] = 0

        # # For padded sequences, ensure the embedding layer can map the token ID
        # tokens[tokens == self._pad_token_id] = 0
        # labels[labels == self._pad_token_id] = 0

        # Batch padding sequence so we mask the loss
        if idx is None:
            loss_mask = torch.zeros_like(loss_mask)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
