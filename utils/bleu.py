import math
from collections import Counter

from tqdm import tqdm

import utils.model_utils as model_utils


def count_n_gram(tokens, n):
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) + 1 - n)])

def precision_i(pred_sents, tgt_sents, i):
    total_matched_i_grams = 0
    total_pred_i_grams = 0
    for pred_tokens, tgt_tokens in zip(pred_sents, tgt_sents):
        pred_i_grams = count_n_gram(pred_tokens, i)
        tgt_i_grams = count_n_gram(tgt_tokens, i)

        total_pred_i_grams += len(pred_tokens) - i + 1
        for pred_i_gram, pred_count in pred_i_grams.items():
            tgt_matched_count = tgt_i_grams.get(pred_i_gram, 0)
            total_matched_i_grams += min(tgt_matched_count, pred_count)

    if total_pred_i_grams == 0:
        return 0
    return total_matched_i_grams / total_pred_i_grams

def calculate_bleu(pred_sents, tgt_sents):
    n_gram_overlap = 0

    for i in range(1, 5):
        prec_i = precision_i(pred_sents,  tgt_sents, i)
        if prec_i == 0:
            return 0.0
        else:
            n_gram_overlap += 0.25 * math.log(prec_i)

    geometric_mean = math.exp(n_gram_overlap)

    r = sum(len(tgt_tokens) for tgt_tokens in pred_sents)
    c = sum(len(pred_tokens) for pred_tokens in pred_sents)

    brevity_penalty = math.exp(min(1 - r/c, 0))

    bleu = brevity_penalty * geometric_mean
    return bleu

def calculate_dataloader_bleu(dataloader, src_tok, tgt_tok, model, device,
                              max_len=256, teacher_forcing=False,
                              print_pair=False, beam_size=1, acceptable_delta=2):

    dataset = dataloader.dataset
    total = len(dataset)
    pred_sents = list()
    tgt_sents = list()

    for i, data in tqdm(enumerate(dataset), desc ="BLEU calculating", disable=print_pair):
        src_tokens = src_tok.vocab.tensor2words(data["src"])
        tgt_tokens = tgt_tok.vocab.tensor2words(data["tgt"])
        
        if teacher_forcing:
            pred_tokens = model_utils.translate_tensor_teacher_forcing(data["src"], 
                                                                       data["tgt"],
                                                                       tgt_tok, 
                                                                       model, 
                                                                       device, 
                                                                       max_len)
        elif beam_size > 1:
            candidates = model_utils.translate_sentence_beam_search(src_tokens,
                                                                    src_tok, 
                                                                    tgt_tok,
                                                                    model, 
                                                                    device, 
                                                                    max_len, 
                                                                    beam_size,
                                                                    acceptable_delta)
            # cut off <bos> and <eos> tokens
            candidates = [(tokens[1:-1], score) for tokens, score in candidates]
            pred_tokens = candidates[0][0]
        else:
            pred_tokens, _ = model_utils.translate_sentence(src_tokens,
                                                            src_tok, 
                                                            tgt_tok,
                                                            model, 
                                                            device, 
                                                            max_len)
            # cut off <eos> token
            pred_tokens = pred_tokens[:-1]

        # cut off <bos> and <eos> tokens
        tgt_tokens = tgt_tokens[1:-1]
        src_tokens = src_tokens[1:-1]

        pred_sents.append(pred_tokens)
        tgt_sents.append(tgt_tokens)

        if print_pair:
            print(f"{i}/{total}")
            print("Source:", " ".join(src_tokens))
            print("Target:", " ".join(tgt_tokens))
            if beam_size > 1:
                for i, (tokens, score) in enumerate(candidates):
                    print(f"Predict {i+1} - score={score:.2f}:", " ".join(tokens))
            else:
                print("Predict:", " ".join(pred_tokens))
            print("BLEU:", calculate_bleu([pred_tokens], [tgt_tokens]) * 100)
            print("-" * 79)

    return calculate_bleu(pred_sents, tgt_sents)