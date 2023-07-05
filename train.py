import math

import pandas as pd
import torch
from torch import nn, optim
from torch.optim import Adam

import utils.data_utils as data_utils
import utils.model_utils as model_utils
import utils.other_utils as other_utils
from models.model.transformer import Transformer


print("Load config file...")
config = data_utils.get_config("config.yml")
for key, value in config.items():
    globals()[key] = value


print("Ensure directory and load file paths...")
other_utils.create_dir(checkpoint["dir"])
best_checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["best"]])
last_checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["last"]])
results_fpath = "/".join([checkpoint["dir"], checkpoint["results"]])
vocab_fpath = "/".join([checkpoint["dir"], checkpoint["parallel_vocab"]])
dataloaders_fpath = "/".join([checkpoint["dir"], checkpoint["dataloaders"]])


print("Load prepared dataloaders and vocabulary...")
envi_vocab = torch.load(vocab_fpath)
dataloaders = torch.load(dataloaders_fpath)
train_loader = dataloaders["train_loader"]
valid_loader = dataloaders["valid_loader"]


print("Load model & optimizer & criterion...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc_voc_size = len(envi_vocab.src)
dec_voc_size = len(envi_vocab.tgt)

model = Transformer(src_pad_idx=envi_vocab.src.pad_id,
                    tgt_pad_idx=envi_vocab.tgt.pad_id,
                    tgt_sos_idx=envi_vocab.tgt.bos_id,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

optimizer = Adam(params=model.parameters(),
                lr=init_lr,
                weight_decay=weight_decay,
                eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=envi_vocab.src.pad_id)


print("Load checkpoint...")
begin_epoch = 1
best_loss = float("inf")
if other_utils.exist(last_checkpoint_fpath):
    checkpoint_dict = torch.load(last_checkpoint_fpath)
    best_loss = checkpoint_dict["loss"]
    begin_epoch = checkpoint_dict["epoch"] + 1
    model.load_state_dict(checkpoint_dict["model_state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    if begin_epoch <= total_epoch:
        print(f"Continue from the last epoch {begin_epoch}...")
else:
    print(f"Last checkpoint is not found, continue from epoch 1...")
    model.apply(model_utils.initialize_weights)

print(f"The model has {model_utils.count_parameters(model):,} trainable parameters")

columns = ["epoch", "train_loss", "valid_loss", "valid_BLEU"]
results_df = pd.DataFrame(columns=columns)
if other_utils.exist(results_fpath):
    results_df = pd.read_csv(results_fpath)


print(f"Start training & evaluating...")
for epoch in range(begin_epoch, total_epoch+1):
    with other_utils.TimeContextManager() as timer:
        train_loss = model_utils.train(model, train_loader, optimizer, criterion, clip, epoch)
        valid_loss, valid_BLEU = model_utils.evaluate(model, valid_loader, criterion, envi_vocab)
        epoch_mins, epoch_secs = timer.get_time()

    if epoch > warmup:
        scheduler.step(valid_loss)

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