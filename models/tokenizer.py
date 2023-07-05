from abc import ABC, abstractmethod

import nltk
from pyvi import ViTokenizer as PyViTokenizer


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class BaseTokenizer(ABC):

    @abstractmethod
    def tokenize(self):
        pass


class ViTokenizer(BaseTokenizer):

    def tokenize(self, sentence):
        if len(sentence) == 0:
            return list()
        else:
            return PyViTokenizer.spacy_tokenize(sentence)[0]


class EnTokenizer(BaseTokenizer):

    def tokenize(self, sentence):
        return nltk.tokenize.word_tokenize(sentence)