"""Two pre-specified alternatives for the Agent's similar-wording slice."""
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.preprocessing import normalize

from develop import diagnostics


def near_cases(frame):
    return diagnostics("pair", None, frame)["similar_wording"]


class SliceCandidate:
    """Use the locked hybrid outside the slice; estimate only the alternative head."""

    def __init__(self, name, baseline):
        self.name = name
        self.baseline = baseline
        self.classes_ = baseline.classes_

    def matrix(self, x):
        union = self.baseline.steps[0][1]
        if self.name == "domain_features":
            design = union.transform(x)
            near = near_cases(x)[:, None]
            return hstack([design, design.multiply(near), design.multiply(~near)], format="csr")
        if self.name == "unit_lexical_edit":
            blocks = dict(union.transformer_list)
            lexical = normalize(blocks["tfidf"].transform(x), norm="l2")
            return hstack([lexical, blocks["style"].transform(x)], format="csr")
        raise ValueError(f"Unknown candidate {self.name}")

    def fit(self, x, y):
        self.head = clone(self.baseline.steps[-1][1]).fit(self.matrix(x), y)
        return self

    def predict_proba(self, x):
        scores = self.baseline.predict_proba(x)
        near = near_cases(x)
        if near.any():
            scores[near] = self.head.predict_proba(self.matrix(x.loc[near]))
        return scores
