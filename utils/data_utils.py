import yaml
from tqdm import tqdm

import torch
from torch.utils.data import random_split


def pad_tensor(sents, pad_id):
    # lengths = torch.tensor([len(s) for s in sents])
    padded_tensor = torch.nn.utils.rnn.pad_sequence(sents,
                                                    batch_first=True,
                                                    padding_value=pad_id)
    # return padded_tensor, lengths
    return padded_tensor

def split_dataset(dataset, split_rate):
    full_size = len(dataset)
    train_size = (int)(split_rate * full_size)
    valid_size = (int)((full_size - train_size)/2)
    test_size = full_size - train_size - valid_size
    return random_split(dataset, lengths=[train_size, valid_size, test_size])

def get_config(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def read_sents(fpath, is_lowercase=False):
    sents = list()
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            sents.append(line.rstrip("\n"))

    if is_lowercase:
        return [s.lower() for s in sents]
    else:
        return sents

def is_sent_valid(tokenized_sent, max_len):
    sent_len = len(tokenized_sent)
    return (sent_len < max_len and sent_len > 0)

def tokenize_and_remove_invalid_sents(src_sents, tgt_sents, max_len,
                                      src_tokenize_func, tgt_tokenize_func):
    '''
        Tokenize sentences, count number of tokens to remove invalid sentences
        Parameters:
            src_sents (list(str)): list of source sentence strings
            tgt_sents (list(str)): list of target sentence strings
            max_len (int): maximum number of tokens for each sentence
            src_tokenize_func (function): tokenizing function for source language
            tgt_tokenize_func (function): tokenizing function for target language
        Returns:
            src_tokenized_sents (list(list(str))): list of valid source sentences,
                                                   each sentence is a list of tokens
            tgt_tokenized_sents (list(list(str))): list of valid target sentences,
                                                   each sentence is a list of tokens
    '''
    src_tokenized_sents = list()
    tgt_tokenized_sents = list()

    total_lines = len(src_sents)
    for i in tqdm(range(total_lines), desc="Read lines"):
        src_tokenized_s = src_tokenize_func(src_sents[i])
        tgt_tokenized_s = tgt_tokenize_func(tgt_sents[i])
        if (is_sent_valid(src_tokenized_s, max_len) and
            is_sent_valid(tgt_tokenized_s, max_len)):
            src_tokenized_sents.append(src_tokenized_s)
            tgt_tokenized_sents.append(tgt_tokenized_s)

    print(f"Remove {total_lines - len(src_tokenized_sents)} invalid sentences")

    return src_tokenized_sents, tgt_tokenized_sents