import math

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    '''
        Translate sentence used trained model with autoregression decoding strategy
        Parameters:
            sent (str or list(str)): sentence string or list of tokens of 
                                     source language
            src_tok (BaseTokenizer): tokenizer for source language
            tgt_tok (BaseTokenizer): tokenizer for target language
            model (Seq2Seq): trained NMT model
            device (torch.device): cpu or cuda
            max_len (int): maximum number of tokens for each sentence
        Returns:
            pred_tokens (list(str)): list of translated tokens 
            attention (Tensor): attention score from the last layer, can be used
                                to create attention alignment matrix
    '''
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

    pred_indexes = [tgt_tok.vocab.bos_id]

    # Loop to predict next index from list of previous indexes until reach the 
    # end-of-string (eos) token or reach the maximum length
    for i in range(max_len):
        pred_tensor = torch.LongTensor(pred_indexes).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(pred_tensor)
        with torch.no_grad():
            output, attention = model.decoder(pred_tensor, enc_src, tgt_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        pred_indexes.append(pred_token)
        if pred_token == tgt_tok.vocab.eos_id:
            break

    pred_tokens = [tgt_tok.vocab.id2word[i] for i in pred_indexes]

    return pred_tokens, attention

def display_attention(src_tokens, pred_tokens, attention, n_heads = 8, 
                      n_rows=4, n_cols=2, fig_size=(15,25)):

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=fig_size)

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i+1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + src_tokens, rotation=90)
        ax.set_yticklabels([''] + pred_tokens)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    return fig

def translate_sentence_beam_search(sent, src_tok, tgt_tok, model, device, 
                                   max_len=256, beam_size=1):
    '''
        Translate sentence used trained model with beam search decoding strategy
        Parameters:
            sent (str or list(str)): sentence string or list of tokens of 
                                     source language
            src_tok (BaseTokenizer): tokenizer for source language
            tgt_tok (BaseTokenizer): tokenizer for target language
            model (Seq2Seq): trained NMT model
            device (torch.device): cpu or cuda
            max_len (int): maximum number of tokens for each sentence
            beam_size (int): maximum number of candidate sentences
        Returns:
            pred_sents (list(list(str))): list of translated sentences, each
                                          sentence is a list of tokens 
    ''' 
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

    # candidate_list: where each candidate is a tuple(indexes_list, log_score)
    candidate_list = [([tgt_tok.vocab.bos_id], 0.0)]

    # completed_candidate: the chosen candidates
    completed_candidates = list()

    # loop to predict next index from list of previous indexes until reach the 
    # end-of-string (eos) token or reach the maximum length
    for i in range(max_len):
        num_candidates = len(candidate_list)
        new_candidate_list = list()

        # for each candidate, concatnate it with beam_size best next tokens
        # then add them to new_candidate_list
        for j in range(num_candidates):
            pred_indexes, candidate_score = candidate_list[j]
            pred_tensor = torch.LongTensor(pred_indexes).unsqueeze(0).to(device)
            pred_mask = model.make_tgt_mask(pred_tensor)
            with torch.no_grad():
                output, attention = model.decoder(pred_tensor, enc_src, pred_mask, src_mask)

            output = F.log_softmax(output, dim=-1)

            # get beam_size indexes having lowest log_softmax values
            probs, indexes = output[:, -1].data.topk(beam_size)
            probs = probs.squeeze(0).tolist()
            indexes = indexes.type(torch.int64).squeeze(0).tolist()

            # add beam_size new candidates
            for expand_id in range(beam_size):
                score = probs[expand_id]
                token_id = indexes[expand_id]
                new_candidate = (pred_indexes + [token_id], candidate_score + score)
                new_candidate_list.append(new_candidate)

        # sort and keep n best candidates, where n = beam_size - len(completed_candidates)
        sorted_candidate_list = sorted(new_candidate_list, key=lambda c: c[1], reverse=True)
        sorted_candidate_list = sorted_candidate_list[:beam_size-len(completed_candidates)]

        # from n best candidates, find completed candidates 
        # and add them into completed_candidates, 
        # other candidates will keep for the next loop
        candidate_list = list()
        for candidate in sorted_candidate_list:
            if candidate[0][-1] == tgt_tok.vocab.eos_id:
                completed_candidates.append(candidate)
                if (len(completed_candidates) == beam_size):
                    break
            else:
                candidate_list.append(candidate)

    # if len(completed_candidates) < beam_size, 
    # add remaining ones from candidate_list which reach the max_len
    completed_candidates += candidate_list

    pred_sents = list()
    for token_indexes, log in completed_candidates:
        pred_tensor = torch.Tensor(token_indexes)
        pred_tokens = tgt_tok.vocab.tensor2words(pred_tensor)
        pred_sents.append((pred_tokens, log))

    pred_sents = sorted(pred_sents, key=lambda c: c[1], reverse=True)
    return pred_sents

def translate_tensor_teacher_forcing(src_tensor, tgt_tensor, tgt_tok,
                                     model, device, max_len=256):
    '''
        Translate sentence used trained model with teacher forcing strategy,
        which uses tgt_tensor to get the prediction faster
        Parameters:
            src_tensor (Tensor): Tensor of source sentence indexes
            tgt_tensor (Tensor): Tensor of target sentece indexes
            tgt_tok (BaseTokenizer): tokenizer for target language
            model (Seq2Seq): trained NMT model
            device (torch.device): cpu or cuda
            max_len (int): maximum number of tokens for each sentence
        Returns:
            pred_tokens (list(str)): list of translated tokens 
    ''' 
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
    '''
        Train step 
        Parameters:
            epoch (int): the index of current epoch
            model (Seq2Seq): trained NMT model
            loader (DataLoader): train DataLoader
            optimizer (torch.optim)
            criterion (torch.nn)
        Returns:
            epoch_loss / len(loader) (float): average loss of this epoch 
    '''
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
    '''
        Evaluate the model
        Parameters:
            model (Seq2Seq): trained NMT model
            loader (DataLoader): test/valid DataLoader
            criterion (torch.nn)
        Returns:
            epoch_loss / len(loader) (float): average loss
    '''
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

def calculate_dataloader_bleu(loader, src_tok, tgt_tok, model, device, max_len=256, 
                              teacher_forcing=False, print_pair=False, beam_size=1):
    '''
        Calculate BLEU score for all sentence pairs of a dataloader
        Parameters:
            loader (DataLoader)
            src_tok (BaseTokenizer): tokenizer for source language
            tgt_tok (BaseTokenizer): tokenizer for target language
            model (Seq2Seq): trained NMT model
            device (torch.device): cpu or cuda
            max_len (int): maximum number of tokens for each sentence
            teacher_forcing (bool): use teacher forcing decoding strategy
            print_pair (bool): print source, target and predicted sentences or not
            beam_size (bool): maximum number of candidate sentences. Beam search
                              decoding is used when teacher_forcing == False
        Returns:
            bleu (float): BLEU score for all sentence pairs of a dataloader
    '''
    dataset = loader.dataset
    total = len(dataset)
    pred_sents = list()
    tgt_sents = list()

    for i, data in tqdm(enumerate(dataset), desc ="BLEU calculating", disable=print_pair):
        src_tokens = src_tok.vocab.tensor2words(data["src"])
        tgt_tokens = tgt_tok.vocab.tensor2words(data["tgt"])
        
        # choose decoding strategy
        if teacher_forcing:
            pred_tokens = translate_tensor_teacher_forcing(data["src"], data["tgt"],
                                                           tgt_tok, model, device, 
                                                           max_len)
        else:
            candidates = translate_sentence_beam_search(src_tokens,
                                                        src_tok, tgt_tok, model, 
                                                        device, max_len, beam_size)
            # cut off <bos> and <eos> tokens
            candidates = [(tokens[1:-1], score) for tokens, score in candidates]
            pred_tokens = candidates[0][0]

        # these below commented lines use to debug beam search strategy 
        # where beam_size == 1. 
        # Please change above 'else:' into 'if beam_size > 1' if you want to debug.
        
        # else:
        #     pred_tokens, _ = translate_sentence(src_tokens, src_tok, tgt_tok,
        #                                         model, device, max_len)
        #     # cut off <bos> and <eos> tokens
        #     pred_tokens = pred_tokens[1:-1]

        # cut off <bos> and <eos> tokens
        tgt_tokens = tgt_tokens[1:-1]
        src_tokens = src_tokens[1:-1]

        pred_sents.append(pred_tokens)
        tgt_sents.append(tgt_tokens)

        if print_pair:
            print(f"{i}/{total}")
            print("Source:", src_tok.detokenize(src_tokens))
            print("Target:", tgt_tok.detokenize(tgt_tokens))
            if beam_size > 1:
                for i, (tokens, log) in enumerate(candidates):
                    print(f"Predict {i+1} - log={log:.2f}:", tgt_tok.detokenize(tokens))
            else:
                print("Predict:", tgt_tok.detokenize(pred_tokens))
            print("BLEU:", bleu.calculate_bleu([pred_tokens], [tgt_tokens]) * 100)
            print("-" * 79)


    return bleu.calculate_bleu(pred_sents, tgt_sents)

def test(model, loader, criterion, src_tok, tgt_tok, max_len, beam_size=1):
    '''
        Print loss, translated sentences and BLEU score.
        Parameters:
            model (Seq2Seq): trained NMT model
            loader (DataLoader): test/valid DataLoader
            criterion (torch.nn)
            src_tok (BaseTokenizer): tokenizer for source language
            tgt_tok (BaseTokenizer): tokenizer for target language
            max_len (int): maximum number of tokens for each sentence
            beam_size (bool): maximum number of candidate sentences
        Returns:
            None
    '''
    device = model.device

    loss = model_utils.evaluate(model, loader, criterion)
    BLEU = calculate_dataloader_bleu(loader, src_tok, tgt_tok,
                                          model, device, max_len,
                                          teacher_forcing=False,
                                          print_pair=True,
                                          beam_size=beam_size) * 100

    print(f"Test Loss: {loss:.3f} |  Test PPL: {math.exp(loss):7.3f}")
    print(f"TOTAL BLEU SCORE = {BLEU}")