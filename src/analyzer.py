import numpy as np
import pandas as pd

def calculatet_log_odds(pos_counts, neg_counts, alpha = 1.0):

    words = list(set(pos_counts.keys()) | set (neg_counts.keys()))

    pos_arr = np.array([pos_counts.get(w,0) + alpha for w in words])
    neg_arr = np.array([neg_counts.get(w,0) + alpha for w in words])

    log_odds = np.log(pos_arr / pos_arr.sum()) - np.log(neg_arr / neg_arr.sum())

    return dict(zip(words, log_odds))

def mask_extreme_tokens(log_odds_dict):

    words = list(log_odds_dict.keys())
    scores = np.array(list(log_odds_dict.values()))

    pos_mask = scores > 2.0
    neg_mask = scores < -2.0

    pos_words = {words[i] for i in range (len(words)) if pos_mask[i]}
    neg_words = {words[i] for i in range (len(words)) if neg_mask[i]}

    return pos_words, neg_words