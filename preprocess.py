import torch
from torch.utils.data import DataLoader

from models.parallel_dataset import ParallelDataset
from models.tokenizer import EnTokenizer, ViTokenizer
from models.vocabulary import ParallelVocabulary
import utils.data_utils as data_utils
import utils.other_utils as other_utils


def load_dataloader_from_fpath(pair_fpath, src_tok, tgt_tok, batch_size, max_len, 
                               device, is_lowercase=True, is_train=False, is_sorted=True):
    src_sents = data_utils.read_sents(pair_fpath["src"], is_lowercase)
    tgt_sents = data_utils.read_sents(pair_fpath["tgt"], is_lowercase)
    src_tok_sents, tgt_tok_sents = data_utils.tokenize_and_remove_invalid_sents(src_sents,
                                                                                tgt_sents,
                                                                                max_len,
                                                                                src_tok.tokenize,
                                                                                tgt_tok.tokenize)

    if is_train:
        print(f"Create vocabulary from {pair_fpath['src']}...")
        src_tok.train_vocab(src_tok_sents, is_tokenized=True)
        print(f"Create vocabulary from {pair_fpath['tgt']}...")
        tgt_tok.train_vocab(tgt_tok_sents, is_tokenized=True)

    dataset = ParallelDataset(src_tok_sents, tgt_tok_sents, src_tok, tgt_tok, is_sorted=True)
    parallel_vocab = ParallelVocabulary(src_tok.vocab, tgt_tok.vocab, device)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=parallel_vocab.collate_fn,
                        shuffle=True)

    if is_train:
        return loader, src_tok, tgt_tok
    else:
        return loader


print("Load config file...")
config = data_utils.get_config("config.yml")
for key, value in config.items():
    globals()[key] = value

print("Load tokenizers...")
src_tok = EnTokenizer()
tgt_tok = ViTokenizer()

print("Load DataLoaders")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, src_tok, tgt_tok = load_dataloader_from_fpath(path["train"],
                                                            src_tok,
                                                            tgt_tok,
                                                            batch_size,
                                                            max_len,
                                                            device,
                                                            is_lowercase=True,
                                                            is_train=True)
valid_loader = load_dataloader_from_fpath(path["valid"],
                                          src_tok,
                                          tgt_tok,
                                          batch_size,
                                          max_len,
                                          device,
                                          is_lowercase=True,
                                          is_train=False)
test_loader = load_dataloader_from_fpath(path["test"],
                                         src_tok,
                                         tgt_tok,
                                         batch_size,
                                         max_len,
                                         device,
                                         is_lowercase=True,
                                         is_train=False)

dataloaders = {"train_loader": train_loader,
               "valid_loader": valid_loader,
               "test_loader": test_loader}

print("Load file paths and save...")
other_utils.create_dir(checkpoint["dir"])
src_tok_fpath = "/".join([checkpoint["dir"], checkpoint["tokenizer"]["src"]])
tgt_tok_fpath = "/".join([checkpoint["dir"], checkpoint["tokenizer"]["tgt"]])
dataloaders_fpath = "/".join([checkpoint["dir"], checkpoint["dataloaders"]])

torch.save(src_tok, src_tok_fpath)
torch.save(tgt_tok, tgt_tok_fpath)
torch.save(dataloaders, dataloaders_fpath)