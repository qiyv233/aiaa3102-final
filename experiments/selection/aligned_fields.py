"""Pool structurally related ordered survey responses across context categories."""
from collections import Counter, defaultdict
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from profile_boost import ProfileBackoff, row_keys, tree


def ordered_level(value):
    text = str(value).casefold().strip()
    if text in {"never", "none", "zero"}:
        return 0.
    digits = re.findall(r"\d+(?:\.\d+)?", text)
    if not digits:
        return None
    nums = list(map(float, digits))
    if len(nums) == 2 and re.fullmatch(r"[\d.]+\s*(?:~|-|–|to)\s*[\d.]+", text):
        return sum(nums)/2
    if len(nums) != 1:
        return None
    if re.fullmatch(r"(?:lt|less(?:\s*than)?|<)\s*[\d.]+", text):
        return nums[0]/2
    if re.fullmatch(r"(?:gt|more(?:\s*than)?|>)\s*[\d.]+|[\d.]+\+", text):
        return nums[0]+1
    return nums[0] if re.fullmatch(r"[\d.]+", text) else None


def name_tokens(text):
    split = re.sub(r"([a-z])([A-Z])",r"\1 \2",str(text)).casefold()
    split = re.sub(r"less\s*than|<"," lt ",split)
    split = re.sub(r"greater\s*than|>"," gt ",split)
    return set(re.findall(r"[a-z]+|\d+",split)) - {"to","and"}


class AlignedFields:
    def fit(self, frame):
        scales = defaultdict(list)
        for col in frame:
            values = set(frame[col].unique()) - {""}
            if not 3 <= len(values) <= 16:
                continue
            numeric = [ordered_level(v) for v in values]
            if all(v is not None for v in numeric) and len(set(numeric)) == len(values):
                scales[tuple(sorted(numeric))].append(col)
        best = None
        for columns in scales.values():
            if len(columns)<3:
                continue
            name_sets = {col:name_tokens(col) for col in columns}
            frequencies = Counter(k for words in name_sets.values() for k in words)
            def weight(words):
                return sum(1. + np.log((len(columns)+1)/(frequencies[k]+1)) for k in words)
            for context in sorted(set(frame.columns)-set(columns)):
                counts = frame[context].value_counts()
                if not 2<=len(counts)<=2*len(columns)+2:
                    continue
                matches = {}
                quality = 0.
                for value, count in counts.items():
                    query = name_tokens(value)
                    scores = []
                    for col,words in name_sets.items():
                        common = weight(query & words)
                        score = 2*common / max(1.,weight(query)+weight(words))
                        if {k for k in query if k.isdigit()} != {k for k in words if k.isdigit()}:
                            score *= .35
                        scores.append((score,col))
                    scores.sort(reverse=True)
                    if scores[0][0] >= .42 and scores[0][0]-scores[1][0] >= .12:
                        score,col = scores[0]
                        matches[value] = col
                        quality += score*count/len(frame)
                if len(set(matches.values()))<3 or len(set(matches.values()))!=len(matches):
                    continue
                if best is None or quality > best[0]:
                    best = (quality,context,tuple(columns),matches)
        self.context_ = None if best is None else best[1]
        self.columns_ = () if best is None else best[2]
        self.mapping_ = {} if best is None else best[3]
        self.ranks_ = {}
        for col in self.columns_:
            values = set(frame[col].unique())-{""}
            self.ranks_[col] = {v:i for i,v in enumerate(sorted(values,key=ordered_level))}
        self.names_ = []
        for label in ("selected_rank","relative_rank","mean_rank","selected_missing"):
            name = "__aligned_"+label
            while name in frame.columns:
                name = "_"+name
            self.names_.append(name)
        return self

    def transform(self,frame):
        if self.context_ is None:
            return frame, np.zeros(len(frame),dtype=bool)
        encoded = np.column_stack([frame[c].map(self.ranks_[c]).to_numpy(dtype=float) for c in self.columns_])
        n = np.isfinite(encoded).sum(axis=1)
        mean = np.divide(np.nansum(encoded,axis=1),n,out=np.full(len(frame),np.nan),where=n>0)
        selected = np.full(len(frame),np.nan)
        values = frame[self.context_].to_numpy()
        known_context = np.zeros(len(frame),dtype=bool)
        for val,col in self.mapping_.items():
            hit = values==val
            selected[hit] = encoded[hit,self.columns_.index(col)]
            known_context |= hit
        numeric = np.column_stack([selected,selected-mean,mean,~np.isfinite(selected)])
        return pd.concat([frame,pd.DataFrame(numeric,index=frame.index,columns=self.names_)],axis=1), known_context


class AlignedBackoff:
    def fit(self,x,y):
        self.base_ = ProfileBackoff().fit(x,y)
        self.classes_ = self.base_.classes_
        self.fields_ = AlignedFields().fit(x)
        self.fitted_ = False
        profile = self.base_.profile_
        if not self.fields_.mapping_ or not profile.keys_:
            return self
        raw = x.reset_index(drop=True)
        expanded,_ = self.fields_.transform(raw)
        positive = (np.asarray(y)==self.classes_[1]).astype(float)
        groups = row_keys(raw,raw.columns).factorize()[0]
        pieces = []
        for train,held in GroupKFold(4,shuffle=True,random_state=113).split(raw,groups=groups):
            maps = profile.make_maps(raw.iloc[train],positive[train])
            pieces.append(profile.encode(raw.iloc[held],maps))
        stats = pd.concat(pieces).sort_index()
        assert expanded.index.equals(stats.index) and not stats.isna().any().any()
        self.direct_ = tree().fit(expanded,y,cat_features=list(raw))
        self.profiled_ = tree().fit(pd.concat([expanded,stats],axis=1),y,cat_features=list(raw))
        self.fitted_ = True
        return self

    def predict_proba(self,x):
        if not self.fitted_:
            return self.base_.predict_proba(x)
        raw = x.reset_index(drop=True)
        expanded,matched = self.fields_.transform(raw)
        use = self.base_.known(raw) & matched
        p = np.empty((len(x),2))
        if use.any():
            subset = expanded.loc[use]
            profile = self.base_.profile_
            stats = profile.encode(raw.loc[use],profile.maps_)
            full = pd.concat([subset,stats],axis=1)
            p[use] = (self.direct_.predict_proba(subset)+self.profiled_.predict_proba(full))/2
        if (~use).any():
            p[~use] = self.base_.predict_proba(raw.loc[~use])
        return p

