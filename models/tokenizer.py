from abc import ABC, abstractmethod

import spacy
from pyvi import ViTokenizer as PyViTokenizer

from models.vocabulary import Vocabulary


class BaseTokenizer(ABC):

    def __init__(self):
        self.vocab = Vocabulary()

    @abstractmethod
    def tokenize(self):
        pass

    def train_vocab(self, sents, is_tokenized=False, min_freq=1):
        if not is_tokenized:
            tokenized_sents = self.tokenize(sents)
        else:
            tokenized_sents = sents
        self.vocab.add_words(tokenized_sents, min_freq)


class ViTokenizer(BaseTokenizer):

    def tokenize(self, sentence):
        if len(sentence) == 0:
            return list()
        else:
            return PyViTokenizer.spacy_tokenize(sentence)[0]


class EnTokenizer(BaseTokenizer):

    def __init__(self):
        super().__init__()
        self.spacy_en = spacy.load('en_core_web_sm')

    def tokenize(self, sentence):
        return [tok.text for tok in self.spacy_en.tokenizer(sentence)]