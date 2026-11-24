"""Models that adjust predictions only on a training-supported subset."""
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import re

from catboost import CatBoostClassifier
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

from paired_text import pair_estimator
from profile_boost import ProfileBackoff, row_keys


TOKEN = re.compile(r"[a-z0-9]+|[^\w\s]", re.IGNORECASE)
WORD = re.compile(r"[a-z0-9]+")
DICT_HASH = "ea49afb616e5778e539bd0cb89140bd5b4b773b5cc251625b2426034d0939022"


@lru_cache(maxsize=1)
def dictionary():
    raw = (Path(__file__).with_name("wordnet-data") / "semantic.json.gz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DICT_HASH:
        raise ValueError("Unexpected lexical resource")
    return json.loads(gzip.decompress(raw))


@lru_cache(maxsize=32768)
def concepts(word):
    lookup = dictionary()
    candidates = [word]
    endings = [("ies","y"),("es",""),("s",""),("ing",""),("ing","e"),("ed",""),("ed","e")]
    for suffix, replacement in endings:
        if word.endswith(suffix) and len(word) >= len(suffix)+3:
            candidates.append(word[:-len(suffix)]+replacement)
    for form in candidates:
        if form in lookup:
            lexical_class, parent = lookup[form]
            return tuple(v for v in ("lex="+lexical_class, "parent="+parent if parent else "") if v)
    if word.isdigit():
        return ("number",)
    return ()


def overlap(left, right):
    a = set(WORD.findall(left.casefold()))
    b = set(WORD.findall(right.casefold()))
    return len(a & b) / max(1, len(a | b))


def semantic_edit(left, right):
    # Canonical ordering makes ambiguous alignments identical when the two sides swap.
    a, b = (tuple(TOKEN.findall(v.casefold())) for v in (left, right))
    orientation = 1 if a <= b else -1
    if orientation < 0:
        a, b = b, a
    features = Counter()
    for op, i, j, k, m in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        for tokens, begin, end, sign in ((a, i, j, orientation), (b, k, m, -orientation)):
            loc = "first" if begin == 0 else ("last" if end == len(tokens) else "internal")
            span = tokens[begin:end]
            if not span:
                continue
            features["span_len="+str(min(6,len(span)))] += sign
            features["position="+loc] += sign
            for token in span:
                features["word="+token] += sign * .65
                for tag in concepts(token):
                    features[tag] += sign
                    features[loc+"|"+tag] += sign * .5
            for t, u in zip(span,span[1:]):
                features["phrase="+t+" "+u] += sign * .5
    return {k:v for k,v in features.items() if v}


class SemanticEdits:
    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.base_ = pair_estimator("pair_hybrid",113).fit(x,y)
        pairs = list(x.itertuples(index=False, name=None))
        self.vocab_ = DictVectorizer()
        matrix = normalize(self.vocab_.fit_transform(semantic_edit(a,b) for a,b in pairs))
        weights = [3. if overlap(a,b) >= .25 else 1. for a,b in pairs]
        self.expert_ = None
        if matrix.shape[1]:
            self.expert_ = LogisticRegression(C=.65,fit_intercept=False,max_iter=500,random_state=113)
            self.expert_.fit(matrix,y,sample_weight=weights)
        return self

    def predict_proba(self, x):
        p = self.base_.predict_proba(x)
        pairs = list(x.itertuples(index=False, name=None))
        indices = [i for i,(a,b) in enumerate(pairs) if overlap(a,b) >= .5]
        if indices and self.expert_ is not None:
            matrix = self.vocab_.transform(semantic_edit(*pairs[i]) for i in indices)
            usable = np.flatnonzero(matrix.getnnz(axis=1))
            if len(usable):
                target = np.asarray(indices)[usable]
                local = self.expert_.predict_proba(normalize(matrix[usable]))
                p[target] = .45 * p[target] + .55 * local
        return p


def small_tree(iterations, depth):
    return CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=.06,
                              l2_leaf_reg=6., one_hot_max_size=32, random_seed=113,
                              thread_count=2, verbose=False, allow_writing_files=False)


class SegmentExperts:
    def fit(self, x, y):
        x = x.reset_index(drop=True)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.base_ = ProfileBackoff().fit(x,y)
        profile = self.base_.profile_.profile_
        self.field_ = None
        self.experts_ = {}
        self.screen_ = []
        eligible = [c for c in x if c not in profile and 3 <= x[c].nunique() <= 12]
        group = row_keys(x,x.columns).factorize()[0]
        if not profile or not eligible or len(np.unique(group)) < 4:
            return self
        # Use out-of-fold errors from a cheap global model as the screening signal.
        oof = np.empty(len(y),dtype=bool)
        importances = np.zeros(x.shape[1])
        for train, check in GroupKFold(4, shuffle=True, random_state=113).split(x,groups=group):
            if len(np.unique(y[train])) < 2:
                return self
            probe = small_tree(125,5).fit(x.iloc[train],y[train],cat_features=list(x))
            oof[check] = self.classes_[probe.predict_proba(x.iloc[check]).argmax(1)] == y[check]
            importances += probe.get_feature_importance()
        self.field_ = max(eligible,key=lambda c: (importances[x.columns.get_loc(c)],str(c)))
        global_rate = oof.mean()
        for val, idx in x.groupby(self.field_,sort=True).indices.items():
            size = len(idx)
            score = (oof[idx].sum()+80*global_rate)/(size+80)
            if size >= 150 and len(np.unique(y[idx]))==2 and score < global_rate - .015:
                self.screen_.append((float(score),str(val),size))
        self.screen_.sort()
        for _,val,size in self.screen_[:2]:
            index = np.flatnonzero(x[self.field_].to_numpy() == val)
            subset = x.iloc[index]
            columns = [c for c in x if subset[c].nunique()>1]
            if columns:
                expert = small_tree(320,4).fit(subset[columns],y[index],cat_features=columns)
                self.experts_[val] = (columns,expert,min(.65,size/(size+450.)))
        return self

    def predict_proba(self, x):
        p = self.base_.predict_proba(x)
        if self.experts_:
            known = self.base_.known(x)
            for val,(columns,expert,weight) in self.experts_.items():
                rows = np.flatnonzero(known & (x[self.field_].to_numpy() == val))
                if len(rows):
                    p[rows] = (1-weight)*p[rows] + weight*expert.predict_proba(x.iloc[rows][columns])
        return p

