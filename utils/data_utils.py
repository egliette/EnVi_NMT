import yaml

import torch
from torch.utils.data import random_split


def pad_tensor(sents, pad_id):
    sorted_sents = sorted(sents, key=lambda s: len(s), reverse=True)
    # lengths = torch.tensor([len(s) for s in sents])
    padded_tensor = torch.nn.utils.rnn.pad_sequence(sents,
                                                    batch_first=True,
                                                    padding_value=pad_id)
    # return padded_tensor, lengths
    return padded_tensor

def add_bos_eos(tensor, bos_id, eos_id):
    bos_tensor = torch.tensor([bos_id])
    eos_tensor = torch.tensor([eos_id])
    return torch.cat((bos_tensor, tensor, eos_tensor))

def collate_fn(parallel_vocab, examples):
    src_sents = [src for src, __ in examples]
    src_tensors = parallel_vocab.corpus_to_tensor(src_sents)
    processed_src = list(map(lambda tensor: add_bos_eos(tensor,
                                                        parallel_vocab.src.bos_id,
                                                        parallel_vocab.src.eos_id),
                             src_tensors))

    tgt_sents = [tgt for __, tgt in examples]
    processed_tgt = parallel_vocab.corpus_to_tensor(tgt_sents, is_source=False)

    return {"src": pad_tensor(processed_src, parallel_vocab.src.pad_id),
            "tgt": pad_tensor(processed_tgt, parallel_vocab.tgt.pad_id)}

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