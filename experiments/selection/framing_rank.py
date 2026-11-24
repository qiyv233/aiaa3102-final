"""Extra antisymmetric feature sets for binary text preferences."""

import re
from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from focus_candidates import concepts
from paired_text import pair_estimator, pair_texts


TOKENS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?|[^\w\s]")


@lru_cache(maxsize=16384)
def frames(text):
    words = TOKENS.findall(text.casefold())
    if not words:
        return ("empty",)
    tags = []
    output = []
    for word in words:
        tags_for_word = concepts(word)
        tag = tags_for_word[0] if tags_for_word else word
        tags.append(tag)
        if tags_for_word:
            output.extend(tags_for_word)
    for width in (1, 2, 3):
        output.append("open:" + " ".join(words[:width]))
        output.append("close:" + " ".join(words[-width:]))
        for pos in range(len(tags) - width + 1):
            output.append("tag:" + " ".join(tags[pos:pos + width]))
    for i in range(1, len(words)):
        output.extend(("before:" + words[i-1] + "/" + tags[i],
                       "after:" + tags[i-1] + "/" + words[i]))
    return tuple(output)


class FrameDifference(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.vector_ = TfidfVectorizer(analyzer=frames, min_df=2, max_features=35000,
                                       sublinear_tf=True).fit(np.concatenate(pair_texts(X)))
        return self

    def transform(self, X):
        a, b = pair_texts(X)
        return self.vector_.transform(a) - self.vector_.transform(b)


class ContrastDifference(TransformerMixin, BaseEstimator):
    """Weight binary word and character counts by training-side label association."""

    def fit(self, X, y):
        a, b = pair_texts(X)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("Expected two ordered classes")
        positive = np.asarray(y) == classes[1]
        texts = np.concatenate((a, b))
        wins = np.concatenate((positive, ~positive))
        self.blocks_ = []
        for params in (dict(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b"),
                       dict(analyzer="char_wb", ngram_range=(3, 5))):
            vec = CountVectorizer(binary=True, min_df=3, max_features=50000, **params)
            counts = vec.fit_transform(texts)
            won = np.asarray(counts[wins].sum(0)).ravel()
            lost = np.asarray(counts[~wins].sum(0)).ravel()
            weights = np.minimum(2.5, np.abs(np.log((won + 2.) / (lost + 2.))))
            self.blocks_.append((vec, weights))
        return self

    def transform(self, X):
        from scipy.sparse import hstack
        a, b = pair_texts(X)
        blocks = []
        for vec, weights in self.blocks_:
            delta = vec.transform(a) - vec.transform(b)
            blocks.append(normalize(delta.multiply(weights).tocsr(), copy=False))
        return hstack(blocks, format="csr")


def framing_estimator(name):
    model = pair_estimator("pair_hybrid", 113)
    extra = FrameDifference() if name == "framing_rank" else ContrastDifference()
    model.named_steps["featureunion"].transformer_list.append(("extra", extra))
    return model
