import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from pathlib import Path
from transformers import BartTokenizer
from transformers.file_utils import cached_property
from seq2seq.utils import (
    DistributedSortishSampler,
    SortishSampler,
)
try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


class Seq2SeqDataset_ET5(Dataset):
    """A dataset that calls prepare_seq2seq_batch."""
    def __init__(
        self,
        tokenizer,
        data_dir,
        processor,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        filetype="json",
        **dataset_kwargs
    ):
        super().__init__()
        ##########
        if filetype == "json":
            self.data_file = Path(data_dir).joinpath(type_path + ".json")
        elif filetype == "tsv":
            self.data_file = Path(data_dir).joinpath(type_path + ".tsv")
        elif filetype == "jsonl":
            self.data_file = Path(data_dir).joinpath(type_path + ".jsonl")
        (list_id, src_texts, tgt_texts) = processor.get_seq2seq_examples(filename=self.data_file)
        self.data = {'list_id':list_id, "src_texts": src_texts, "tgt_texts": tgt_texts}
        self.src_lens = self.get_char_lens(self.data['src_texts'])
        self.used_char_len = True
        ##########

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        # return [len(x) for x in Path(data_file).open().readlines()]
        return [len(x) for x in data_file]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, index) -> Dict[str, str]:
        source_line = self.data['src_texts'][index]
        tgt_line = self.data['tgt_texts'][index]

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


