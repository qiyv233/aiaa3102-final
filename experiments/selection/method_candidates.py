"""Fold-local learners for ordered text pairs and ordinal views of categorical data."""
import re

import numpy as np
from sklearn.preprocessing import normalize

from paired_text import PairTfidf


class EditNeighbors:
    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.features = PairTfidf().fit(x)
        self.examples = normalize(self.features.transform(x))
        self.signs = np.where(y == self.classes_[1], 1., -1.)
        return self

    def predict_proba(self, x):
        query = normalize(self.features.transform(x))
        positive = np.full(len(x), .5)
        for start in range(0, len(x), 64):
            scores = (query[start:start + 64] @ self.examples.T).toarray()
            # Stable tie-break uses training input order; it is invariant under swapping A and B.
            neighbors = np.argsort(-np.abs(scores), axis=1, kind="stable")[:, :7]
            similarity = np.take_along_axis(scores, neighbors, axis=1)
            total = np.abs(similarity).sum(axis=1)
            signed_sum = (similarity * self.signs[neighbors]).sum(axis=1)
            vote = np.divide(signed_sum, total, out=np.zeros_like(total), where=total > 0)
            positive[start:start + len(scores)] = (1 + np.clip(vote, -1, 1)) / 2
        return np.column_stack((1 - positive, positive))


def ordinal_token(value):
    """Return domain, lower endpoint, upper endpoint; open ends stay missing."""
    s = str(value).strip().casefold()
    if s in ('', 'nan', 'none', 'null', 'n/a'):
        return None
    number = r'(?:0|[1-9]\d*)(?:\.\d+)?'
    if s == 'never':
        return 'band', 0., 0.
    if re.fullmatch(number, s):
        return 'scalar', float(s), float(s)
    clock = re.fullmatch(r'(1[0-2]|[1-9])\s*(am|pm)', s)
    if clock:
        hour = int(clock[1]) % 12 + (12 if clock[2] == 'pm' else 0)
        return 'clock', float(hour), float(hour)
    duration = re.fullmatch('(' + number + r')\s*(min|h|d)', s)
    if duration:
        minutes = float(duration[1]) * {'min': 1, 'h': 60, 'd': 1440}[duration[2]]
        return 'duration', minutes, minutes
    money = '$' in s
    if money:
        s = s.replace('$', '').replace(',', '')
    domain = 'money' if money else 'band'
    closed = re.fullmatch('(' + number + r')\s*(?:~|–|-)\s*(' + number + ')', s)
    if closed and float(closed[1]) <= float(closed[2]):
        return domain, float(closed[1]), float(closed[2])
    below = re.fullmatch(r'(?:less(?:\s+than)?|below|lt|<)\s*(' + number + ')', s)
    if below:
        return domain, np.nan, float(below[1])
    above = re.fullmatch(r'(?:greater\s+than|above|gt|>)\s*(' + number + ')', s)
    plus = re.fullmatch('(' + number + r')\s*(?:plus|or more|\+)', s)
    if above or plus:
        return domain, float((above or plus)[1]), np.nan
    return None


class OrdinalCatBoost:
    def fit(self, x, y):
        from catboost import CatBoostClassifier
        self.views = {}
        for col in x:
            raw = [v for v in x[col].astype(str).unique() if v.strip().casefold() not in ('','nan','none','null','n/a')]
            parsed = [ordinal_token(v) for v in raw]
            if len(raw) < 2 or any(v is None for v in parsed):
                continue
            domains = {v[0] for v in parsed}
            if len(domains) > 1 and domains != {'scalar','band'}:
                continue
            # A column of unexplained digit codes might be an identifier.
            # Add pure scalar views only when separately declared numeric.
            if domains == {'scalar'}:
                continue
            self.views[col] = domains
        if not self.views:
            raise ValueError('No supported ordered category found in training')
        self.model = CatBoostClassifier(cat_features=list(x), iterations=250, depth=6, learning_rate=.08,
                                        random_seed=113, thread_count=2, verbose=False, allow_writing_files=False)
        self.model.fit(self.transform(x), y)
        self.classes_ = self.model.classes_
        return self

    def transform(self, x):
        result = x.copy()
        for i, (col, domains) in enumerate(self.views.items()):
            parsed = {v: ordinal_token(v) for v in x[col].unique()}
            for j, side in enumerate(('lo','hi'), start=1):
                name = f'__ordinal_view_{i}_{side}'
                while name in result:
                    name += '_'
                mapping = {v: token[j] for v, token in parsed.items() if token and token[0] in domains}
                result[name] = x[col].map(mapping).astype(float)
        return result

    def predict_proba(self, x):
        return self.model.predict_proba(self.transform(x))
