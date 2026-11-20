"""Represent binary text comparisons with features that preserve the order of the pair."""

import re

import numpy as np
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted


PAIR_CANDIDATES = {
    "pair_tfidf": "Compare the texts in a shared word and character vocabulary.",
    "pair_style": "Try nonlinear comparisons of length, casing, digits and punctuation.",
    "pair_hybrid": "Combine lexical differences with writing-style differences.",
}

WORD = re.compile(r"[^\W_]+")
PUNCTUATION = ("?", "!", ".", ",", ":", ";", "-", "'", '"', "(", "%", "…")


def pair_texts(frame):
    if frame.shape[1] != 2:
        raise ValueError("A text comparison needs exactly two ordered columns")
    a, b = frame.fillna("").astype(str).to_numpy().T
    return a, b


class PairTfidf(TransformerMixin, BaseEstimator):
    """Fit vocabularies on both sides and return A minus B in the shared space."""

    def fit(self, X, y=None):
        a, b = pair_texts(X)
        texts = np.concatenate((a, b))
        self.word_ = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
                                     min_df=2, sublinear_tf=True, max_features=45000).fit(texts)
        self.char_ = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                                     sublinear_tf=True, max_features=45000).fit(texts)
        return self

    def transform(self, X):
        check_is_fitted(self, ("word_", "char_"))
        a, b = pair_texts(X)
        return hstack([tfidf.transform(a) - tfidf.transform(b) for tfidf in (self.word_, self.char_)],
                      format="csr")


def text_style(text):
    words = WORD.findall(text)
    count = max(len(words), 1)
    letters = max(sum(char.isalpha() for char in text), 1)
    digits = sum(char.isdigit() for char in text)
    return (
        len(text), len(words), sum(map(len, words)) / count,
        max(map(len, words), default=0), len(set(w.casefold() for w in words)) / count,
        sum(char.isupper() for char in text) / letters,
        sum(word.istitle() for word in words) / count,
        sum(word.isupper() for word in words) / count,
        digits, sum(word.isdigit() for word in words), float(text[:1].isdigit()),
        *(text.count(mark) for mark in PUNCTUATION),
    )


def style_sides(frame):
    a, b = pair_texts(frame)
    return np.asarray([text_style(text) for text in a]), np.asarray([text_style(text) for text in b])


def style_difference(frame):
    a, b = style_sides(frame)
    return a - b


class PairStyleClassifier(ClassifierMixin, BaseEstimator):
    """Learn interactions and average both orders to make the comparison consistent."""

    def __init__(self, seed=17):
        self.seed = seed

    def fit(self, X, y):
        labels = np.asarray(y)
        self.classes_ = np.unique(labels)
        if len(self.classes_) != 2:
            raise ValueError("Pairwise text classification requires two classes")
        a, b = style_sides(X)
        forward, reverse = np.column_stack((a, b, a - b)), np.column_stack((b, a, b - a))
        flipped = np.where(labels == self.classes_[0], self.classes_[1], self.classes_[0])
        self.model_ = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=self.seed,
        ).fit(np.concatenate((forward, reverse)), np.concatenate((labels, flipped)))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        a, b = style_sides(X)
        forward = self.model_.predict_proba(np.column_stack((a, b, a - b)))
        reverse = self.model_.predict_proba(np.column_stack((b, a, b - a)))
        return (forward + reverse[:, ::-1]) / 2

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def pair_estimator(name, seed):
    if name == "pair_style":
        return PairStyleClassifier(seed)
    features = PairTfidf()
    if name == "pair_hybrid":
        style = make_pipeline(FunctionTransformer(style_difference), StandardScaler(with_mean=False))
        features = FeatureUnion([("tfidf", features), ("style", style)])
    return make_pipeline(features, LogisticRegression(fit_intercept=False, max_iter=500, random_state=seed))
