from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from utils.data_utils import pad_tensor


class ParallelDataset(Dataset):
    """Load dataset from source and target text files"""

    def __init__(self, src_fpath, tgt_fpath, max_sent_len, parallel_vocab=None,
                 tokenized_fpath=None):
        self.parallel_vocab = parallel_vocab
        self.src_sents = self._read_data(src_fpath)
        self.tgt_sents = self._read_data(tgt_fpath)
        self.src_tokenized, self.tgt_tokenized = self._remove_invalid_sent(max_sent_len)

    def __len__(self):
        return len(self.src_tokenized)

    def __getitem__(self, id):
        if isinstance(id, int):
            return {"src": self.src_tokenized[id],
                    "tgt": self.tgt_tokenized[id]}
            # return self.tokenize_pair(id)
            # return self.src_sents[id], self.tgt_sents[id]
        elif isinstance(id, slice):
            start = 0 if id.start == None else id.start
            step = 1 if id.step == None else id.step
            return [self[i] for i in range(start, id.stop, step)]
        elif isinstance(id, list):
            return [self[i] for i in id]
        else:
            raise TypeError("Invalid argument type.")

    def _read_data(cls, fpath):
        sents = list()
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                sents.append(line.rstrip("\n"))

        return sents

    def tokenize_pair(self, id):
        src = self.parallel_vocab.tokenize_corpus([self.src_sents[id]])
        src = ["<s>"] + src[0] + ["</s>"]
        tgt = self.parallel_vocab.tokenize_corpus([self.tgt_sents[id]], is_source=False)
        tgt = ["<s>"] + tgt[0] + ["</s>"]
        return {"src": src, "tgt": tgt}

    def _remove_invalid_sent(self, max_sent_len):
        src_tokenized = list()
        tgt_tokenized = list()
        total_lines = len(self.src_sents)
        for i in tqdm(range(total_lines)):
            tokenized = self.tokenize_pair(i)
            src_len = len(tokenized["src"])
            tgt_len = len(tokenized["tgt"])
            if (src_len < max_sent_len and src_len > 0 and
                tgt_len < max_sent_len and tgt_len > 0):
                src_tokenized.append(tokenized["src"])
                tgt_tokenized.append(tokenized["tgt"])
        print(f"Remove {total_lines - len(src_tokenized)} ivalid sentences")
        return src_tokenized, tgt_tokenized

    def collate_fn(self, examples):
        src_sents = [pair["src"] for pair in examples]
        processed_src = self.parallel_vocab.corpus_to_tensor(src_sents,
                                                             is_tokenized=True)
        tgt_sents = [pair["tgt"] for pair in examples]
        processed_tgt = self.parallel_vocab.corpus_to_tensor(tgt_sents,
                                                             is_tokenized=True,
                                                             is_source=False)

        return {"src": pad_tensor(processed_src, self.parallel_vocab.src.pad_id),
                "tgt": pad_tensor(processed_tgt, self.parallel_vocab.tgt.pad_id)}
