"""Tabular candidates using bounded category interactions and cross-fitted repeated profiles."""
from itertools import combinations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold


def tree():
    return CatBoostClassifier(iterations=650, depth=7, learning_rate=.065, one_hot_max_size=32,
                              random_seed=113, thread_count=2, verbose=False, allow_writing_files=False)


def row_keys(x, columns):
    # MultiIndex equality preserves raw strings, tuple lengths and field order.
    return pd.MultiIndex.from_frame(x[list(columns)])


def discover(x):
    active = [c for c in x if 1 < x[c].nunique() <= 64]
    if len(active) < 5:
        return []
    encoded = {c:pd.factorize(x[c],sort=True)[0] for c in active}
    sizes = {c:int(encoded[c].max()+1) for c in active}
    probs = {c:np.bincount(encoded[c])/len(x) for c in active}
    base = {c:float(p @ p) for c,p in probs.items()}
    entropy = {c:float(-p @ np.log(p)) for c,p in probs.items()}
    seeds = sorted(active,key=lambda c:(-entropy[c],str(c)))[:8]

    def expand(group):
        codes = row_keys(x,group).factorize()[0]
        sizes_g = np.bincount(codes)
        pairs = sizes_g @ (sizes_g-1)
        if pairs < len(x):
            return group
        result = set(group)
        for c in active:
            bins = np.bincount(codes.astype(np.int64)*sizes[c]+encoded[c])
            same = float(bins @ (bins-1))/pairs
            if same >= base[c]+.60*(1-base[c]):
                result.add(c)
        return tuple(c for c in active if c in result)

    blocks = set()
    for seed in combinations(seeds,3):
        first = expand(seed)
        if len(first)>3:
            blocks.add(expand(first))
    scored = []
    for block in blocks:
        if len(block)<4 or len(block)==len(active):
            continue
        ids = row_keys(x,block).factorize()[0]
        freq = np.bincount(ids)
        coverage = float((freq[ids]>=3).mean())
        if coverage>=.40 and len(freq)>=12:
            scored.append((len(block)*coverage,coverage,tuple(sorted(block))))
    return list(max(scored)[2]) if scored else []


class ProfileBoost:
    """Target statistics from other training rows and other exact-feature groups only."""
    def make_maps(self, x, y):
        global_mean=(y.sum()+1)/(len(y)+2)
        maps=[]
        for cols in self.keys_:
            frame=x[list(cols)].copy()
            frame["__response__"]=y
            table=frame.groupby(list(cols),sort=False,dropna=False)["__response__"].agg(["sum","count"])
            maps.append(table)
        return global_mean,maps

    def encode(self,x,lookup):
        prior,maps=lookup
        out={}
        for i,(cols,table) in enumerate(zip(self.keys_,maps)):
            keys=row_keys(x,cols)
            target=table.reindex(keys)
            counts=target["count"].fillna(0).to_numpy()
            sums=target["sum"].fillna(0).to_numpy()
            rate=(sums + (4*prior if i==0 else 2*out["profile_rate_0"]))/(counts+(4 if i==0 else 2))
            out[f"profile_rate_{i}"]=rate
            out[f"profile_support_{i}"]=np.log1p(counts)
        return pd.DataFrame(out,index=x.index)

    def fit(self,x,y):
        x=x.reset_index(drop=True)
        self.classes_=np.unique(y)
        if len(self.classes_)!=2:
            raise ValueError("Profile boosting expects binary labels")
        numeric=(np.asarray(y)==self.classes_[1]).astype(float)
        self.profile_=discover(x)
        self.keys_=[]
        self.tree_=tree()
        if not self.profile_:
            self.tree_.fit(x,y,cat_features=list(x))
            return self
        context=[c for c in x if c not in self.profile_ and 1<x[c].nunique()<=64]
        self.keys_=[self.profile_,*[self.profile_+[c] for c in context]]
        row_group=row_keys(x,x.columns).factorize()[0]
        encoded=[]
        for a,b in GroupKFold(4,shuffle=True,random_state=113).split(x,groups=row_group):
            lookup=self.make_maps(x.iloc[a],numeric[a])
            encoded.append(self.encode(x.iloc[b],lookup))
        stacked=pd.concat(encoded).sort_index()
        assert stacked.index.equals(x.index) and not stacked.isna().any().any()
        self.maps_=self.make_maps(x,numeric)
        self.tree_.fit(pd.concat([x,stacked],axis=1),y,cat_features=list(x))
        return self

    def predict_proba(self,x):
        target=x.reset_index(drop=True)
        if self.keys_:
            target=pd.concat([target,self.encode(target,self.maps_)],axis=1)
        return self.tree_.predict_proba(target)


class ProfileBackoff:
    """Blend two new learners for seen attribute tuples; otherwise use the reference ensemble."""
    def fit(self,x,y):
        from models import fit_one
        self.classes_=np.unique(y)
        self.profile_=ProfileBoost().fit(x,y)
        self.reference_=[fit_one(name,"categorical",x,y,self.classes_) for name in ("catboost","trees")]
        self.onehot_=tree().fit(x,y,cat_features=list(x)) if self.profile_.keys_ else None
        return self

    def known(self,x):
        if self.onehot_ is None:
            return np.zeros(len(x),dtype=bool)
        index=self.profile_.maps_[1][0].index
        return row_keys(x,self.profile_.profile_).isin(index)

    def predict_proba(self,x):
        from models import probabilities
        known=self.known(x)
        result=np.empty((len(x),len(self.classes_)),dtype=float)
        if known.any():
            rows=x.iloc[np.flatnonzero(known)]
            result[known]=np.mean([probabilities(model,rows,self.classes_) for model in (self.onehot_,self.profile_)],axis=0)
        if (~known).any():
            rows=x.iloc[np.flatnonzero(~known)]
            result[~known]=np.mean([probabilities(model,rows,self.classes_) for model in self.reference_],axis=0)
        return result
