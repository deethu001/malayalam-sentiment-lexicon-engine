import numpy as np
import pandas as pd
from collections import Counter

class MalayalamSentimentEngine:
    def __init__(self):
        self.df = None
        self.lexicon = None

    def load_from_csv(self,path):
        df = pd.read_csv(path)

        df['tokens'] = df['review_text'].str.split()
        df['token_count'] = df['tokens'].str.len()
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


        for tokens, label in zip(self.df['tokens'], self.df['label']):
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


    ## N4 - verification , prediction, masks and report(Q3,Q4,Q5)
    
    def verify_log_odds(self, laplace_alpha = 1.0):
        lex, a = self.lexicon, laplace_alpha
        v,P,N = len(lex), lex['pos'].sum(), lex['neg'].sum()

        loop = [
            np.log((r.pos+a) / (P + a * v))
            - np.log((r.neg + a) / (N + a * v))
            for r in lex.itertuples()
        ]

        return np.allclose(lex['log_odds'].to_numpy(), loop)


    def predict(self,threshold = 0.25):
        w = dict(zip(self.lexicon['token'], self.lexicon['log_odds']))

        scores = self.df['tokens'].apply(
            lambda ts: np.mean([w.get(t, 0.0) for t in ts]) if ts else 0.0
        )

        self.df['lexicon_sentiment_score'] = scores 

        self .df['predicted_polarity'] = np.select(
            [scores > threshold, scores < - threshold],
            ['positive', 'negative'],
            default = 'neutral'
        )

        return self


    def extreme_tokens(self, cut =2.0):
        lo = self.lexicon['log_odds']. to_numpy()
        toks = self.lexicon['token']. to_numpy()

        pos_set, neg_set = set(toks[lo > cut]), set(toks[lo < - cut])

        return pos_set, neg_set, pos_set & neg_set

    def summary_report(self):
        return (
            f"Reviews: {len(self.df)} | "
            f"Lexicon size: {len(self.lexicon)} | "
            f"Mean tokens per review: "
            f"{self.df['token_count']. mean():2f}"
        )


    #Q6 to Q9

    def ols_length_vs_strength(self):

        y = self.df['lexicon_sentiment_score'].abs().to_numpy()

        X = np.column_stack([
            np.ones(len(y)),
            self.df['token_count'],
            self.df['char_count']
        ])

        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        ss_res = ((y - X @ beta)**2).sum()
        ss_tot = ((y - y.mean()) **2).sum()

        return beta, (1-ss_res / ss_tot if ss_tot else float('nan'))

    def domain_tables(self):

        d = self.df.assign(
            word_len = self.df['char_count'] / self.df['token_count'].replace(0,np.nan)
        )

        by_domain = d.groupby('domain').agg(
            reviews = ('label', 'size'),
            mean_tokens = ('token_count', 'mean'),
            mean_word_len = ('word_len', 'mean')
        )

        label_share = d['label'].value_counts(normalize = True)

        cross = pd.crosstab(d['domain'], d['label'])

        pivot = d.pivot_table(
            index ='domain',
            columns = 'label',
            values = 'char_count',
            aggfunc = 'mean',
            margins = True

        )

        return by_domain, label_share,cross,pivot


    def top_tokens_long(self, k = 5):

        wide = self.lexicon[['token', 'pos', 'neg', 'neu']].rename(
            columns = {
                'pos': 'positive',
                'neg': 'negative',
                'neu': 'neutral'
            }
        )

        long = wide.melt(
            id_vars ='token',
            var_name = 'Sentiment_Class',
            value_name = 'count'
        )

        idx = long.groupby('Sentiment_Class')['count'].nlargest(k).index.get_level_values(1)

        return long.loc[idx].sort_values(
            ['Sentiment_Class','count'],
            ascending = [True, False]
        )