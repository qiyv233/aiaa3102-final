"""Augment preference learning with low-weight, two-hop comparisons inside each fitting fold."""
from collections import defaultdict
import unicodedata

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.svm import LinearSVC

from paired_text import pair_estimator


def key(text):
    return " ".join(unicodedata.normalize("NFKC",str(text)).casefold().split())


def paths(x, y, classes):
    successors=defaultdict(set)
    predecessors=defaultdict(set)
    raw={}
    for (a,b),label in zip(x.itertuples(index=False,name=None),y):
        left,right=key(a),key(b)
        raw.setdefault(left,a)
        raw.setdefault(right,b)
        if left==right:
            continue
        win,lose=(left,right) if label==classes[1] else (right,left)
        successors[win].add(lose)
        predecessors[lose].add(win)
    links=set()
    ambiguous={(a,b) for a in successors for b in successors[a] if a in successors.get(b,())}
    for middle in sorted(predecessors.keys() & successors.keys()):
        before,after=predecessors[middle],successors[middle]
        if len(before)*len(after)>25000:
            continue
        for a in before:
            if (a,middle) in ambiguous:
                continue
            for b in after:
                if (middle,b) in ambiguous or a==b or b in successors.get(a,()) or a in successors.get(b,()):
                    continue
                links.add((a,b))
        if len(links)>max(50000,4*len(x)):
            break
    pairs=sorted((a,b) for a,b in links if (b,a) not in links)
    if len(pairs)>len(x):
        positions=np.random.default_rng(113).choice(len(pairs),len(x),replace=False)
        pairs=[pairs[i] for i in sorted(positions)]
    return [(raw[a],raw[b]) for a,b in pairs]


class GraphRank:
    def __init__(self, margin=False):
        self.margin=margin

    def fit(self,x,y):
        self.classes_=np.unique(y)
        if len(self.classes_)!=2:
            raise ValueError("Expected two preference labels")
        inferred=paths(x,y,self.classes_)
        self.extra_pairs_=len(inferred)
        base=pair_estimator("pair_hybrid",113)
        self.transform_=base.steps[0][1]
        old=self.transform_.fit_transform(x,y)
        weights=np.ones(len(x))
        if inferred:
            from scipy.sparse import vstack
            extra=self.transform_.transform(pd.DataFrame(inferred,columns=x.columns))
            old=vstack((old,extra),format="csr")
            y=np.r_[y,np.repeat(self.classes_[1],len(inferred))]
            weights=np.r_[weights,np.full(len(inferred),.5)]
        self.model_=(LinearSVC(C=.15,fit_intercept=False,dual=False,tol=1e-6,random_state=113,max_iter=2000)
                     if self.margin else base.steps[1][1])
        self.model_.fit(old,y,sample_weight=weights)
        return self

    def predict_proba(self,x):
        features=self.transform_.transform(x)
        if not self.margin:
            return self.model_.predict_proba(features)
        pos=expit(2*self.model_.decision_function(features))
        return np.column_stack([1-pos,pos])
