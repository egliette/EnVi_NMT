# Stop printing tensorflow's logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse

import torch
from torch import nn

import utils.data_utils as data_utils
import utils.model_utils as model_utils
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.transformer.seq2seq import Seq2Seq
from models.tokenizer import EnTokenizer, ViTokenizer


def main(config_fpath="config.yml"):
    print(f"Load config file {config_fpath}...")
    config = data_utils.get_config(config_fpath)
    for key, value in config.items():
        globals()[key] = value


    print("Load prepared dataloader & tokenizers...")
    dataloaders_fpath = "/".join([checkpoint["dir"], checkpoint["dataloaders"]])
    dataloaders = torch.load(dataloaders_fpath)
    test_loader = dataloaders["test_loader"]

    src_vocab_fpath = "/".join([checkpoint["dir"], checkpoint["vocab"]["src"]])
    tgt_vocab_fpath = "/".join([checkpoint["dir"], checkpoint["vocab"]["tgt"]])
    src_tok = EnTokenizer(src_vocab_fpath)
    tgt_tok = EnTokenizer(tgt_vocab_fpath)


    print("Load model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc_voc_size = len(src_tok.vocab)
    dec_voc_size = len(tgt_tok.vocab)

    enc = Encoder(input_dim=enc_voc_size,
                hid_dim=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                pf_dim=ffn_hidden,
                dropout=drop_prob,
                device=device,
                max_length=max_len)

    dec = Decoder(output_dim=dec_voc_size,
                hid_dim=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                pf_dim=ffn_hidden,
                dropout=drop_prob,
                device=device,
                max_length=max_len)

    src_pad_id = src_tok.vocab.pad_id
    tgt_pad_id = tgt_tok.vocab.pad_id

    model = Seq2Seq(enc, dec, src_pad_id, tgt_pad_id, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)


    print("Load the best checkpoint...")
    best_checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["best"]])
    checkpoint_dict = torch.load(best_checkpoint_fpath)
    model.load_state_dict(checkpoint_dict["model_state_dict"])

    print(f"The model has {model_utils.count_parameters(model):,} trainable parameters")


    print("Start testing...")
    model_utils.test(model, test_loader, criterion, src_tok, tgt_tok, max_len, beam_size)

    print("Finish testing!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the NMT model")

    parser.add_argument("--config", 
                        default="config.yml", 
                        help="path to config file",
                        dest="config_fpath")
    
    args = parser.parse_args()

    main(**vars(args))