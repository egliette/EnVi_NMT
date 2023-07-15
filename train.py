# Stop printing tensorflow's logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import math
import argparse

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

import utils.data_utils as data_utils
import utils.model_utils as model_utils
import utils.other_utils as other_utils
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.transformer.seq2seq import Seq2Seq
from models.tokenizer import EnTokenizer, ViTokenizer


def main(config_fpath="config.yml"):
    print(f"Load config file {config_fpath}...")
    config = data_utils.get_config(config_fpath)
    for key, value in config.items():
        globals()[key] = value


    print("Load prepared dataloaders & tokenizers...")
    dataloaders_fpath = "/".join([checkpoint["dir"], checkpoint["dataloaders"]])
    dataloaders = torch.load(dataloaders_fpath)
    train_loader = dataloaders["train_loader"]
    valid_loader = dataloaders["valid_loader"]

    src_tok_fpath = "/".join([checkpoint["dir"], checkpoint["tokenizer"]["src"]])
    tgt_tok_fpath = "/".join([checkpoint["dir"], checkpoint["tokenizer"]["tgt"]])
    src_tok = EnTokenizer(src_tok_fpath)
    tgt_tok = EnTokenizer(tgt_tok_fpath)

    print("Load model & optimizer & criterion...")
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


    print("Load checkpoint...")
    begin_epoch = 1
    best_loss = float("inf")
    best_checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["best"]])
    last_checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["last"]])

    if other_utils.exist(last_checkpoint_fpath):
        checkpoint_dict = torch.load(last_checkpoint_fpath)
        best_loss = checkpoint_dict["loss"]
        begin_epoch = checkpoint_dict["epoch"] + 1
        model.load_state_dict(checkpoint_dict["model_state_dict"])
        optimizer = Adam(params=model.parameters(), lr=init_lr)
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        if begin_epoch <= total_epoch:
            print(f"Continue from the last epoch {begin_epoch}...")
    else:
        print(f"Last checkpoint is not found, continue from epoch 1...")
        model.apply(model_utils.initialize_weights)
        optimizer = Adam(params=model.parameters(), lr=init_lr)

    print(f"The model has {model_utils.count_parameters(model):,} trainable parameters")


    columns = ["epoch", "train_loss", "valid_loss", "valid_BLEU"]
    results_df = pd.DataFrame(columns=columns)
    results_fpath = "/".join([checkpoint["dir"], checkpoint["results"]])
    if other_utils.exist(results_fpath):
        results_df = pd.read_csv(results_fpath)


    print(f"Start training & evaluating...")
    for epoch in range(begin_epoch, total_epoch+1):
        with other_utils.TimeContextManager() as timer:
            train_loss = model_utils.train(epoch, model, train_loader, optimizer, 
                                            criterion, clip)
            valid_loss = model_utils.evaluate(model, valid_loader, criterion)
            valid_BLEU = model_utils.calculate_dataloader_bleu(valid_loader, 
                                                                src_tok, tgt_tok, 
                                                                model, device, 
                                                                max_len=max_len, 
                                                                teacher_forcing=True) * 100
        epoch_mins, epoch_secs = timer.get_time()


        results_df.loc[len(results_df), columns] = epoch, train_loss, valid_loss, valid_BLEU
        results_df.to_csv(results_fpath, index=False)

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({"epoch": epoch,
                        "loss": best_loss,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                        best_checkpoint_fpath)

        torch.save({"epoch": epoch,
                    "loss": best_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                    last_checkpoint_fpath)

        print(f"Epoch: {epoch} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}")
        print(f"\tBLEU Score: {valid_BLEU:.3f}")

    print("Finish training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate the NMT model")

    parser.add_argument("--config", 
                        default="config.yml", 
                        help="path to config file",
                        dest="config_fpath")
    
    args = parser.parse_args()

    main(**vars(args))