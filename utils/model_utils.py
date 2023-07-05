from tqdm import tqdm
import torch
import torch.nn as nn

from utils.bleu import idx_to_word, get_bleu


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)

def train(model, iterator, optimizer, criterion, clip, batch_id):
    model.train()
    device = model.device
    epoch_loss = 0
    with tqdm(enumerate(iterator), total=len(iterator)) as pbar:
        pbar.set_description(f"Epoch {batch_id}")
        for i, batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, parallel_vocab):
    model.eval()
    device = model.device
    epoch_loss = 0
    batch_bleu = list()
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            output = model(src, tgt[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, tgt)
            epoch_loss += loss.item()
            total_bleu = list()
            for j in range(batch["tgt"].shape[0]):
                  tgt_words = idx_to_word(batch["tgt"].to(device)[j], parallel_vocab.tgt)
                  output_words = output[j].max(dim=1)[1]
                  output_words = idx_to_word(output_words, parallel_vocab.tgt)
                  bleu = get_bleu(hypotheses=output_words.split(), reference=tgt_words.split())
                  total_bleu.append(bleu)

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu

def test(model, iterator, parallel_vocab):
    device = model.device

    with torch.no_grad():
        batch_bleu = []
        for i, batch in enumerate(iterator):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            output = model(src, tgt[:, :-1])

            total_bleu = []
            for j in range(batch["tgt"].shape[0]):
                src_words = idx_to_word(src[j], parallel_vocab.src)
                tgt_words = idx_to_word(tgt[j], parallel_vocab.tgt)

                output_words = output[j].max(dim=1)[1]
                output_words = idx_to_word(output_words, parallel_vocab.tgt)

                print("source :", src_words)
                print("target :", tgt_words)
                print("predicted :", output_words)
                print()
                bleu = get_bleu(hypotheses=output_words.split(), reference=tgt_words.split())
                total_bleu.append(bleu)


            total_bleu = sum(total_bleu) / len(total_bleu)
            print(f"BLEU SCORE = {total_bleu}")
            batch_bleu.append(total_bleu)

        batch_bleu = sum(batch_bleu) / len(batch_bleu)
        print(f"TOTAL BLEU SCORE = {batch_bleu}")