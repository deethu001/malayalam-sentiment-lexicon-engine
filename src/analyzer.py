import numpy as np
import pandas as pd
from collections import Counter

class MalayalamSentimentEngine:
    def __init__(self):
        self.df = None
        self.lexicon = None

    def load_from_csv(self,path):
        df = pd.read_csv(path)

        df['token'] = df['review_text'].str.split()
        df['token_count'] = df['token'].str.len()
        df['char_count'] = df ['review_text'].str.len()

        self.df = df
        return self

    def build_lexicon(self,min_df = 1, laplace_alpha = 1.0):

        if laplace_alpha <= 0:
            raise ValueError("laplace_alpha must be greater than 0")
        
        counts = {
            'positive': Counter(),
            'negative': Counter(),
            'neutral': Counter()
        }


        for tokens, label in zip(self.df['token'], self.df['label']):
            counts[label].update(set(tokens))

        pos, neg, neu = counts['positive'],counts['negative'],counts['neutral']

        vocab = sorted(
            t for t in set(pos) | set(neg) | set(neu)
            if pos[t] + neg[t] + neu[t] >= min_df
        )

        p = np.array([pos[t] for t in vocab], dtype = float)

        n = np.array([neg[t] for t in vocab], dtype = float)

        u = np.array([neu[t] for t in vocab], dtype = float)

        a, v = laplace_alpha, len(vocab)

        log_odds = (
            np.log((p + a) / (p.sum()+a *v))
            - np.log((n + a) / (n.sum() + a * v))      
        )
        self.lexicon = pd.DataFrame({
            'token': vocab,
            'pos': p,
            'neg': n,
            'neu': u,
            'freq': p + n + u,
            'log_odds': log_odds,
        }).sort_values(
            'log_odds', ascending = False
        ).reset_index(drop = True)

        return self
    
