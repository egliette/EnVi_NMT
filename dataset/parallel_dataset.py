import torch
from torch.utils.data import Dataset


class ParallelDataset(Dataset):
    """Load dataset from source and target text files"""

    def __init__(self, src_tokenized_sents, tgt_tokenized_sents,
                 src_tokenizer, tgt_tokenizer, is_sorted=True):
        if is_sorted:
            self.src_tokenized_sents = sorted(src_tokenized_sents, key=len)
            self.src_tokenized_sents = sorted(tgt_tokenized_sents, key=len)
        else:
            self.src_tokenized_sents = src_tokenized_sents
            self.tgt_tokenized_sents = tgt_tokenized_sents

        self.src_mapped_sents = src_tokenizer.vocab.sents2tensors(src_tokenized_sents,
                                                                  add_bos_eos=True)
        self.tgt_mapped_sents = tgt_tokenizer.vocab.sents2tensors(tgt_tokenized_sents,
                                                                  add_bos_eos=True)

    def __len__(self):
        return len(self.src_mapped_sents)

    def __getitem__(self, id):
        if isinstance(id, int):
            return {"src": self.src_mapped_sents[id],
                    "tgt": self.tgt_mapped_sents[id]}
        elif isinstance(id, slice):
            start = 0 if id.start == None else id.start
            step = 1 if id.step == None else id.step
            return [self[i] for i in range(start, id.stop, step)]
        elif isinstance(id, list):
            return [self[i] for i in id]
        else:
            raise TypeError("Invalid argument type.")