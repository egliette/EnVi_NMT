import math
from collections import Counter

from tqdm import tqdm

from utils.model_utils import translate_sentence, translate_tensor_teacher_forcing


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
                              max_len=256, teacher_forcing=False, print_pair=False):

    dataset = dataloader.dataset
    pred_sents = list()
    tgt_sents = list()

    for data in tqdm(dataset, desc ="BLEU calculating"):
        src_tokens = src_tok.vocab.tensor2words(data["src"])
        tgt_tokens = tgt_tok.vocab.tensor2words(data["tgt"])

        if teacher_forcing:
            pred_tokens = translate_tensor_teacher_forcing(data["src"], data["tgt"],
                                                        tgt_tok, model, device, max_len)
        else:
            pred_tokens, _ = translate_sentence(src_tokens, src_tok, tgt_tok, 
                                             model, device, max_len)
            # cut off <eos> token
            pred_tokens = pred_tokens[:-1]

        # cut off <bos> and <eos> tokens
        tgt_tokens = tgt_tokens[1:-1]
        src_tokens = src_tokens[1:-1]

        pred_sents.append(pred_tokens)
        tgt_sents.append(tgt_tokens)

        if print_pair:
            print("Source:", " ".join(src_tokens))
            print("Target:", " ".join(tgt_tokens))
            print("Predict:", " ".join(pred_tokens))
            print("BLEU:", calculate_bleu([pred_tokens], [tgt_tokens]))

    return calculate_bleu(pred_sents, tgt_sents)