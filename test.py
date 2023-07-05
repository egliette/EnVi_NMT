import torch

import utils.data_utils as data_utils
import utils.model_utils as model_utils
from models.model.transformer import Transformer


print("Load config file...")
config = data_utils.get_config("config.yml")
for key, value in config.items():
    globals()[key] = value


print("Ensure directory and load file paths...")
checkpoint_fpath = "/".join([checkpoint["dir"], checkpoint["best"]])
vocab_fpath = "/".join([checkpoint["dir"], checkpoint["parallel_vocab"]])
dataloaders_fpath = "/".join([checkpoint["dir"], checkpoint["dataloaders"]])

envi_vocab = torch.load(vocab_fpath)
dataloaders = torch.load(dataloaders_fpath)
test_loader = dataloaders["test_loader"]


print("Load model...")
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
                    drop_prob=0.00,
                    device=device).to(device)


print("Load checkpoint...")
checkpoint_dict = torch.load(checkpoint_fpath)
model.load_state_dict(checkpoint_dict["model_state_dict"])

print(f"The model has {model_utils.count_parameters(model):,} trainable parameters")


print("Start testing...")
model_utils.test(model, test_loader, envi_vocab)
print("Finish testing!")