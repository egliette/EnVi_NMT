import math

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.model_utils as model_utils
import utils.bleu as bleu


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def translate_sentence(sent, src_tok, tgt_tok, model, device, max_len=256):
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

def translate_sentence_beam_search(sent, src_tok, tgt_tok, model, device, 
                                   max_len=256, beam_size=1):
    model.eval()

    if isinstance(sent, str):
        src_tokens = [token.lower() for token in src_tok.tokenize(sent)]
    else:
        src_tokens = [token.lower() for token in sent]

    src_tokens = [src_tok.vocab.bos_token] + src_tokens + [src_tok.vocab.eos_token]

    src_tensor = src_tok.vocab.words2tensor(src_tokens).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    candidate_list = [([tgt_tok.vocab.bos_id], 0.0)]
    completed_candidates = list()

    for i in range(max_len):
        num_candidates = len(candidate_list)
        new_candidate_list = list()
        for j in range(num_candidates):
            pred_indexes, candidate_score = candidate_list[j]
            pred_tensor = torch.LongTensor(pred_indexes).unsqueeze(0).to(device)
            pred_mask = model.make_tgt_mask(pred_tensor)
            with torch.no_grad():
                output, attention = model.decoder(pred_tensor, enc_src, pred_mask, src_mask)

            output = F.log_softmax(output, dim=-1)
            probs, indexes = output[:, -1].data.topk(beam_size)
            probs = probs.squeeze(0).tolist()
            indexes = indexes.type(torch.int64).squeeze(0).tolist()

            for expand_id in range(beam_size):
                score = probs[expand_id]
                token_id = indexes[expand_id]
                new_candidate = (pred_indexes + [token_id], candidate_score + score)
                new_candidate_list.append(new_candidate)

        sorted_candidate_list = sorted(new_candidate_list, key=lambda c: c[1], reverse=True)
        sorted_candidate_list = sorted_candidate_list[:beam_size-len(completed_candidates)]
        candidate_list = list()
        for candidate in sorted_candidate_list:
            if candidate[0][-1] == tgt_tok.vocab.eos_id:
                completed_candidates.append(candidate)
                if (len(completed_candidates) == beam_size):
                    break
            else:
                candidate_list.append(candidate)

    completed_candidates += candidate_list
    completed_candidates = sorted(completed_candidates, key=lambda c: c[1], reverse=True)

    pred_sents = list()
    for token_indexes, log in completed_candidates:
        pred_tensor = torch.Tensor(token_indexes)
        pred_tokens = tgt_tok.vocab.tensor2words(pred_tensor)
        pred_sents.append((pred_tokens, log))

    pred_sents = sorted(pred_sents, key=lambda c: c[1], reverse=True)
    return pred_sents

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

def test(model, test_loader, criterion, src_tok, tgt_tok, max_len, beam_size=1):
    device = model.device

    test_loss = model_utils.evaluate(model, test_loader, criterion)
    test_BLEU = bleu.calculate_dataloader_bleu(test_loader, src_tok, tgt_tok,
                                               model, device, max_len,
                                               teacher_forcing=False,
                                               print_pair=True,
                                               beam_size=beam_size) * 100

    print(f"Test Loss: {test_loss:.3f} |  Test PPL: {math.exp(test_loss):7.3f}")
    print(f"TOTAL BLEU SCORE = {test_BLEU}")