from itertools import chain
from collections import Counter

import torch


class Vocabulary:
    """ The Vocabulary class heavily inspired by machine translation exercise of
        cs224n is used to record words, which are used to convert text to numbers
        and vice versa.
    """

    def __init__(self, tokenizer):
        self.word2id = dict()
        self.word2id['<pad>'] = 0   # Pad Token
        self.word2id['<s>'] = 1     # Start Token
        self.word2id['</s>'] = 2    # End Token
        self.word2id['<unk>'] = 3   # Unknown Token
        self.pad_id = self.word2id['<pad>']
        self.bos_id = self.word2id['<s>']
        self.eos_id = self.word2id['</s>']
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.tokenizer = tokenizer

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def id2word(self, word_index):
        """
        @param word_index (int)
        @return word (str)
        """
        return self.id2word[word_index]

    def add(self, word):
        """ Add word to vocabulary
        @param word (str)
        @return index (str): index of the word just added
        """
        if word not in self:
            word_index = self.word2id[word] = len(self.word2id)
            self.id2word[word_index] = word
            return word_index
        else:
            return self[word]

    def tokenize_corpus(self, corpus):
        """Split the documents of the corpus into words
        @param corpus (list(str)): list of documents
        @return tokenized_corpus (list(list(str))): list of words
        """
        tokenized_corpus = list()
        for document in corpus:
            tokenized_document = self.tokenizer.tokenize(document)
            tokenized_corpus.append(tokenized_document)

        return tokenized_corpus

    def corpus_to_tensor(self, corpus, is_tokenized=False):
        """ Convert corpus to a list of indices tensor
        @param corpus (list(str) if is_tokenized==False else list(list(str)))
        @param is_tokenized (bool)
        @return indicies_corpus (list(tensor))
        """
        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus)
        indicies_corpus = list()
        for document in tokenized_corpus:
            indicies_document = torch.tensor(list(map(lambda word: self[word], document)),
                                             dtype=torch.int64)
            indicies_corpus.append(indicies_document)

        return indicies_corpus

    def tensor_to_corpus(self, tensor):
        """ Convert list of indices tensor to a list of tokenized documents
        @param indicies_corpus (list(tensor))
        @return corpus (list(list(str)))
        """
        corpus = list()
        for indicies in tensor:
            document = list(map(lambda index: self.id2word[index.item()], indicies))
            corpus.append(document)

        return corpus

    def add_words_from_corpus(self, corpus=None, corpus_fpath=None,
                              is_tokenized=False, min_freq=1, lowercase=False):
        print("Add tokens from the corpus...")

        if corpus_fpath is not None:
            corpus = list()
            with open(corpus_fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if lowercase:
                        corpus.append(line.rstrip("\n").lower())
                    else:
                        corpus.append(line.rstrip("\n"))

        if is_tokenized:
            tokenized_corpus = corpus
        else:
            tokenized_corpus = self.tokenize_corpus(corpus)

        word_freq = Counter(chain(*tokenized_corpus))
        non_singletons = [w for w in word_freq if word_freq[w] >= min_freq]
        print(f"Number of tokens in the corpus: {len(word_freq)}")
        print(f"Number of tokens appearing at least {min_freq} times: {len(non_singletons)}")
        for word in non_singletons:
            self.add(word)


class ParallelVocabulary:

    def __init__(self, src_vocab, tgt_vocab):
        self.src = src_vocab
        self.tgt = tgt_vocab

    def tokenize_corpus(self, corpus, is_source=True):
        if is_source:
            return self.src.tokenize_corpus(corpus)
        else:
            return self.tgt.tokenize_corpus(corpus)

    def corpus_to_tensor(self, corpus, is_tokenized=False, is_source=True):
        if is_source:
            return self.src.corpus_to_tensor(corpus, is_tokenized)
        else:
            return self.tgt.corpus_to_tensor(corpus, is_tokenized)

    def tensor_to_corpus(self, tensor, is_source=True):
        if is_source:
            return self.src.tensor_to_corpus(tensor)
        else:
            return self.tgt.tensor_to_corpus(tensor)