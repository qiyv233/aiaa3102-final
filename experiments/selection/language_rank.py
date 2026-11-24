"""Linguistic comparison and a nonlinear pair ranker. Language resources are label-free."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize

from paired_text import pair_texts, pair_estimator, text_style

RESOURCE_SHA = "ec636f884f36b5de6a4d681b2f61c08c2ec90eeb440472033f3307d0d6fa8bc9"
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
CATEGORIES = [set(line.split()) for line in (
    "i me my mine myself i'm i've i'll", "we us our ours let's", "you your yours you're you've you'll",
    "he him his she her hers they their theirs them", "this these that those here there",
    "who what where when why how which whose", "can could may might must should would will",
    "can't don't didn't never not no nothing nobody neither without cannot won't isn't wasn't",
    "all always every everything everyone only just even really actually ever most more least",
    "but although though yet instead however despite unless", "because since so therefore reason reasons",
    "if then before after while until once now still", "a an the", "and or nor",
    "see saw seen look looks watch hear heard feel felt think know learn believe",
    "is are was were am be been being", "has have had do does did", "in on at by of for from with to"
)]


class LanguageDifference(TransformerMixin, BaseEstimator):
    def fit(self,x,y=None):
        path=Path(__file__).with_name("vader-data")/"vader_lexicon.txt"
        data=path.read_bytes()
        if sha256(data).hexdigest()!=RESOURCE_SHA:
            raise ValueError("VADER lexicon fingerprint mismatch")
        self.valence_={}
        for line in data.decode("utf-8-sig").splitlines():
            fields=line.split("\t")
            if len(fields)>=2:
                self.valence_[fields[0]]=float(fields[1])
        self.df_=Counter()
        corpus=set(map(str,np.concatenate(pair_texts(x))))
        for doc in corpus:
            self.df_.update(set(WORD.findall(doc.casefold())))
        self.n_=len(corpus)
        return self

    def one(self,text):
        words=WORD.findall(text.casefold().replace("’","'"))
        count=max(1,len(words))
        tally=Counter(words)
        polar=np.asarray([self.valence_.get(w,0) for w in words] or [0.])
        rarity=np.array([np.log((self.n_+1)/(self.df_.get(w,0)+1)) for w in words] or [0.])
        sizes=np.array(list(map(len,words)) or [0])
        sentences=[s for s in re.split(r"[.!?]+",text) if s.strip()]
        feats=[len(sentences),len(re.findall(r"[,;:–—]",text)),count/max(len(sentences),1),
               float(bool(words and words[0].isdigit())),np.mean(sizes>=7),np.mean(sizes<=3),
               float((polar>0).sum()),float((polar<0).sum()),polar.mean(),np.abs(polar).mean(),polar.min(),polar.max(),
               rarity.mean(),rarity.std(),rarity.max(),np.mean(rarity>=6),rarity[0],rarity[-1]]
        for group in CATEGORIES:
            n=sum(tally.get(w,0) for w in group)
            feats.extend((n, n/count, float(bool(words and words[0] in group))))
        return np.array(feats)

    def sides(self,x):
        a,b=pair_texts(x)
        cache={s:self.one(s) for s in set(a)|set(b)}
        return np.vstack([cache[s] for s in a]),np.vstack([cache[s] for s in b])

    def transform(self,x):
        a,b=self.sides(x)
        return a-b


def linear():
    model=pair_estimator("pair_hybrid",113)
    model.named_steps["featureunion"].transformer_list.append(("language",make_pipeline(LanguageDifference(),StandardScaler(with_mean=False))))
    model.named_steps["featureunion"].set_params(transformer_weights={"tfidf":1.,"style":1.,"language":.5})
    return model


class TopicStyleRanker:
    def arrays(self,x):
        a,b=pair_texts(x)
        l,r=self.language_.sides(x)
        first=np.column_stack([l,[text_style(t) for t in a],normalize(self.topics_.transform(self.vector_.transform(a)))])
        second=np.column_stack([r,[text_style(t) for t in b],normalize(self.topics_.transform(self.vector_.transform(b)))])
        return np.column_stack([first,second,first-second]),np.column_stack([second,first,second-first])

    def fit(self,x,y):
        self.classes_=np.unique(y)
        corpus=np.unique(np.concatenate(pair_texts(x)))
        self.language_=LanguageDifference().fit(x)
        self.vector_=TfidfVectorizer(ngram_range=(1,2),max_features=25000,min_df=2,strip_accents="unicode",sublinear_tf=True)
        matrix=self.vector_.fit_transform(corpus)
        self.topics_=TruncatedSVD(n_components=min(24,matrix.shape[1]-1,len(corpus)-1),random_state=113).fit(matrix)
        f,r=self.arrays(x)
        yi=(np.asarray(y)==self.classes_[1]).astype(int)
        self.model_=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,learning_rate=.065,
                  l2_regularization=5,min_samples_leaf=30,early_stopping=False,random_state=113)
        self.model_.fit(np.vstack([f,r]),np.r_[yi,1-yi])
        return self

    def predict_proba(self,x):
        f,r=self.arrays(x)
        return (self.model_.predict_proba(f)+self.model_.predict_proba(r)[:,::-1])/2
