"""A supervised margin classifier with a local vote for short-text classification."""

import numpy as np
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from features import text_values


class TextMarginNeighbors:
    def fit(self, x, y):
        self.classes_, self.labels_ = np.unique(y, return_inverse=True)
        if len(self.classes_) < 3:
            raise ValueError("Multiclass margin voting requires at least three classes")
        self.vocabulary_ = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
                                     sublinear_tf=True, max_features=50000, dtype=np.float32)),
            ("subword", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                       sublinear_tf=True, max_features=50000, dtype=np.float32)),
        ])
        features = normalize(self.vocabulary_.fit_transform(text_values(x)))
        self.examples_ = features.T.tocsc()
        self.margin_ = LinearSVC(C=2., dual=True, max_iter=1600, random_state=113).fit(features,self.labels_)
        return self

    def predict_proba(self, x):
        features = normalize(self.vocabulary_.transform(text_values(x)))
        margins = self.margin_.decision_function(features)
        k = min(8,len(self.labels_))
        for offset in range(0,len(x),96):
            block = slice(offset,offset+96)
            similarity = (features[block] @ self.examples_).toarray()
            closest = np.argpartition(similarity,-k,axis=1)[:,-k:]
            weights = np.take_along_axis(similarity,closest,axis=1) ** 3
            votes = np.zeros_like(margins[block])
            np.add.at(votes,(np.arange(len(weights))[:,None],self.labels_[closest]),weights)
            # Weak lexical matches contribute little. Identical examples get a vote of one.
            margins[block] += votes / np.maximum(1.,weights.sum(axis=1,keepdims=True))
        return softmax(4.*margins,axis=1)
