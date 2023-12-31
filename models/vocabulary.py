from itertools import chain
from collections import Counter

import torch

from utils.data_utils import pad_tensor


class Vocabulary:
    ''' 
        The Vocabulary class heavily inspired by machine translation exercise 
        of cs224n is used to record words from corpus. The recorded words are 
        used to convert text to numbers and vice versa.
    '''

    def __init__(self):
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.word2id = dict()
        self.word2id[self.unk_token] = 0
        self.word2id[self.pad_token] = 1
        self.word2id[self.bos_token] = 2
        self.word2id[self.eos_token] = 3
        self.unk_id = self.word2id[self.unk_token]
        self.pad_id = self.word2id[self.pad_token]
        self.bos_id = self.word2id[self.bos_token]
        self.eos_id = self.word2id[self.eos_token]
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def id2word(self, word_index):
        return self.id2word[word_index]

    def add(self, word):
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    def words2tensor(self, words, add_bos_eos=False):
        if add_bos_eos:
            processed_words = [self.bos_token] + words + [self.eos_token]
        else:
            processed_words = words
        return torch.tensor(list(map(lambda w: self[w], processed_words)), dtype=torch.int64)

    def sents2tensors(self, tokenized_sents, add_bos_eos=False):
        tensors = list()
        for s in tokenized_sents:
            tensors.append(self.words2tensor(s, add_bos_eos))
        return tensors

    def tensor2words(self, tensor):
        return list(map(lambda index: self.id2word[index.item()], tensor))

    def tensors2sents(self, tensors):
        tokenized_sents = list()
        for t in tensors:
            tokenized_sents.append(self.tensor2words)
        return tokenized_sents

    def add_words(self, tokenized_sents, min_freq=1, vocab_size=None):
        word_freq = Counter(chain(*tokenized_sents))
        non_singletons = [w for w in word_freq if word_freq[w] >= min_freq]
        print(f"Total number of tokens in the corpus: {len(word_freq)}")
        print(f"Number of tokens appearing at least {min_freq} times: {len(non_singletons)}")
        if vocab_size is not None:
            non_singletons = sorted(non_singletons, key=lambda w: word_freq[w], reverse=True)[:vocab_size]
        print(f"Total number of Vocabulary tokens (excluding special tokens): {len(non_singletons)}")
        for word in non_singletons:
            self.add(word)


class ParallelVocabulary:
    '''
        Created from two Vocabulary of source and target languages
        Used to create collate_fn for DataLoader
    '''
    def __init__(self, src_vocab, tgt_vocab, is_sorted, device):
        self.src = src_vocab
        self.tgt = tgt_vocab
        self.is_sorted = is_sorted
        self.device = device

    def collate_fn(self, examples, is_sorted=False):
        src_sents = [pair["src"] for pair in examples]
        tgt_sents = [pair["tgt"] for pair in examples]
        if self.is_sorted:
            pairs = zip(src_sents, tgt_sents)
            sorted_pairs = zip(*sorted(pairs, key= lambda x: (len(x[0]), len(x[1])), reverse=True))
            src_sents, tgt_sents = tuple(list(sorted_sents) for sorted_sents in sorted_pairs)
        return {"src": pad_tensor(src_sents, self.src.pad_id).to(self.device),
                "tgt": pad_tensor(tgt_sents, self.tgt.pad_id).to(self.device)}