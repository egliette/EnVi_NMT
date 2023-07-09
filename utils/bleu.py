import math
from collections import Counter


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

    return total_matched_i_grams / total_pred_i_grams

def calculate_bleu(pred_sents, tgt_sents):
    n_gram_overlap = 0

    for i in range(1, 5):
        n_gram_overlap += 0.25 * math.log(precision_i(pred_sents,  tgt_sents, i))

    geometric_mean = math.exp(n_gram_overlap)

    r = sum(len(tgt_tokens) for tgt_tokens in pred_sents)
    c = sum(len(pred_tokens) for pred_tokens in pred_sents)
        
    brevity_penalty = math.exp(min(1 - r/c, 0))

    bleu = brevity_penalty * geometric_mean
    return bleu