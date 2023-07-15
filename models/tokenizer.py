from abc import ABC, abstractmethod

import spacy
from pyvi import ViTokenizer as PyViTokenizer

from models.vocabulary import Vocabulary


class BaseTokenizer(ABC):

    def __init__(self, vocab_fpath=None):
        self.vocab = Vocabulary()
        if vocab_fpath:
            self.build_vocab(vocab_fpath)

    @abstractmethod
    def tokenize(self, sent):
        pass
    
    @abstractmethod
    def detokenize(self, tokens):
        pass

    def build_vocab(self, sents, is_tokenized=False, min_freq=1, vocab_fpath=None):
        tokenized_sents = list()
        if vocab_fpath is not None:
            with open(vocab_fpath, "r") as f:
                for token in f: 
                    tokenized_sents.append(token.rstrip("\n"))
            min_freq = 1
        elif not is_tokenized:
            tokenized_sents = self.tokenize(sents)
        else:
            tokenized_sents = sents
        self.vocab.add_words(tokenized_sents, min_freq)

    def save_vocab(self, vocab_fpath):
        with open(vocab_fpath, "w") as f:
            for token in self.vocab.word2id.keys(): 
                f.wwrite(token + ("\n"))


class ViTokenizer(BaseTokenizer):

    def tokenize(self, sentence):
        if len(sentence) == 0:
            return list()
        else:
            return PyViTokenizer.spacy_tokenize(sentence)[0]
        
    def detokenize(self, tokens):
        return " ".join(tokens)


class EnTokenizer(BaseTokenizer):

    def __init__(self):
        super().__init__()
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize(self, sentence):
        return [tok.text for tok in self.spacy_en.tokenizer(sentence)]
    
    def detokenize(self, tokens):
        return " ".join(tokens)