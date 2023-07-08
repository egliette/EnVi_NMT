import math
from collections import Counter


def count_n_gram(tokens, n):
    return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) + 1 - n)])

def precision_i(pred_tokens, tgt_tokens, i):
    pred_i_grams = count_n_gram(pred_tokens, i)
    tgt_i_grams = count_n_gram(tgt_tokens, i)

    total_matched_i_grams = 0
    total_pred_i_grams = 0
    for pred_i_gram, pred_count in pred_i_grams.items():
        tgt_matched_count = tgt_i_grams.get(pred_i_gram, 0)
        total_matched_i_grams += min(tgt_matched_count, pred_count)
        total_pred_i_grams += pred_count
    
    return total_matched_i_grams / total_pred_i_grams

def calculate_bleu(pred_tokens, tgt_tokens):
    brevity_penalty = min(1, math.exp(1 - len(tgt_tokens)/len(pred_tokens)))
    n_gram_overlap = 1

    for i in range(1, 5):
        n_gram_overlap *= precision_i(pred_tokens, tgt_tokens, i)
    
    bleu = brevity_penalty * math.pow(n_gram_overlap, 1/4)
    return bleu