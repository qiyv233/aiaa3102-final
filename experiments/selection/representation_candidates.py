"""Fold-local feature variants for controlled comparisons of text pairs and categorical tables."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from paired_text import pair_texts


class PairDifference(TransformerMixin, BaseEstimator):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        a, b = pair_texts(X)
        self.vectorizer_ = clone(self.vectorizer).fit(np.concatenate([a, b]))
        return self

    def transform(self, X):
        a, b = pair_texts(X)
        return self.vectorizer_.transform(a) - self.vectorizer_.transform(b)


def estimator(name, x, seed):
    if name in TABLE:
        onehot = name in {"hgb_onehot", "hgb_onehot_wide"}
        wide = name.endswith("_wide")
        if onehot:
            if 8 * len(x) * sum(x.nunique(dropna=False)) > 256 * 1024**2:
                raise ValueError("Estimated dense one-hot matrix exceeds 256 MiB")
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        tree = HistGradientBoostingClassifier(max_iter=250 if wide else 80, max_leaf_nodes=63 if wide else 31,
                                              min_samples_leaf=20 if wide else 8, categorical_features=None,
                                              random_state=seed)
        return make_pipeline(encoder, tree)
    if name not in TEXT:
        raise ValueError(f"Unknown representation: {name}")
    boundary = name != "pair_char_raw"
    long_ngrams = name in {"pair_wb_long", "pair_word_wb_long"}
    options = dict(min_df=2, sublinear_tf=True)
    chars = PairDifference(TfidfVectorizer(analyzer="char_wb" if boundary else "char",
                           ngram_range=(3,5) if long_ngrams else (2,4),
                           max_features=80000 if long_ngrams else 45000, **options))
    representation = chars
    if name == "pair_word_wb_long":
        word = PairDifference(TfidfVectorizer(ngram_range=(1,2), max_features=60000, **options))
        representation = FeatureUnion([("word", word), ("character", chars)])
    return make_pipeline(representation, LogisticRegression(fit_intercept=False, max_iter=1000, random_state=seed))


TEXT = ("pair_char_raw", "pair_wb_matched", "pair_wb_long", "pair_word_wb_long")
TABLE = ("hgb_ordinal", "hgb_onehot", "hgb_ordinal_wide", "hgb_onehot_wide")
