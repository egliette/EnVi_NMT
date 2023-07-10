import math

from tqdm import tqdm
import torch
import torch.nn as nn

import utils.bleu as bleu
import utils.model_utils as model_utils


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def translate_sentence(sent, src_tok, tgt_tok, model, device, max_len = 256, tgt_sent=None):
    model.eval()

    if isinstance(sent, str):
        tokens = [token.lower() for token in src_tok.tokenize(sent)]
    else:
        tokens = [token.lower() for token in sent]

    tokens = [src_tok.vocab.bos_token] + tokens + [src_tok.vocab.eos_token]
    src_tensor = src_tok.vocab.words2tensor(tokens).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    tgt_indexes = [tgt_tok.vocab.bos_id]

    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indexes).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)
        with torch.no_grad():
            output, attention = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        tgt_indexes.append(pred_token)
        if pred_token == tgt_tok.vocab.eos_id:
            break

    tgt_tokens = [tgt_tok.vocab.id2word[i] for i in tgt_indexes]

    return tgt_tokens[1:], attention

def translate_tensor_teacher_forcing(src_tensor, tgt_tensor, tgt_tok, 
                                     model, device, max_len = 256):
    model.eval()
    src = src_tensor.unsqueeze(0).to(device)
    tgt = tgt_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output, _ = model(src, tgt[:,:-1])

    pred_tokens = tgt_tok.vocab.tensor2words(output.argmax(2).squeeze(0))

    if tgt_tok.vocab.eos_token in pred_tokens:
        eos_position = pred_tokens.index(tgt_tok.vocab.eos_token)
        pred_tokens = pred_tokens[:eos_position]

    return pred_tokens

def train(epoch, model, loader, optimizer, criterion, clip):

    model.train()
    device = model.device
    epoch_loss = 0

    with tqdm(enumerate(loader), total=len(loader)) as pbar:
        pbar.set_description(f"Epoch {epoch}")
        for i, batch in pbar:
            src = batch["src"]
            tgt = batch["tgt"]

            optimizer.zero_grad()
            output, _ = model(src, tgt[:,:-1])

            #output = [batch size, tgt len - 1, output dim]
            #tgt = [batch size, tgt len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[:,1:].contiguous().view(-1)

            #output = [batch size * tgt len - 1, output dim]
            #tgt = [batch size * tgt len - 1]

            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": epoch_loss/(i+1)})

    return epoch_loss / len(loader)

def evaluate(model, loader, criterion):

    model.eval()
    device = model.device
    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(loader):

            src = batch["src"]
            tgt = batch["tgt"]

            output, _ = model(src, tgt[:,:-1])

            #output = [batch size, tgt len - 1, output dim]
            #tgt = [batch size, tgt len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[:,1:].contiguous().view(-1)

            #output = [batch size * tgt len - 1, output dim]
            #tgt = [batch size * tgt len - 1]

            loss = criterion(output, tgt)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)

def test(model, test_loader, criterion, src_tok, tgt_tok, max_len):
    device = model.device

    test_loss = model_utils.evaluate(model, test_loader, criterion)
    test_BLEU = bleu.calculate_dataloader_bleu(test_loader, src_tok, tgt_tok, 
                                               model, device, max_len=max_len, 
                                               teacher_forcing=False, 
                                               print_pair=True) * 100
    
    print(f"Test Loss: {test_loss:.3f} |  Test PPL: {math.exp(test_loss):7.3f}")
    print(f"TOTAL BLEU SCORE = {test_BLEU}")