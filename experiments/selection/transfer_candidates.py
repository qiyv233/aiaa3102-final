"""Fold-local distributional word classes for preferences and a mixed categorical kernel."""
from collections import Counter
import re

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, OrdinalEncoder

from paired_text import pair_estimator,pair_texts
from profile_boost import discover


TOKEN=re.compile(r"\w+(?:['’]\w+)?|[?!]",re.UNICODE)


class WordClassDifference(TransformerMixin,BaseEstimator):
    def fit(self,x,y=None):
        docs=[TOKEN.findall(t.casefold()) for t in sorted(set(np.concatenate(pair_texts(x))))]
        counts=Counter(w for words in docs for w in words)
        vocab={w:i for i,w in enumerate(sorted(w for w,n in counts.items() if n>=3))}
        n=len(vocab)
        if n<6:
            raise ValueError("Insufficient training vocabulary for distributional classes")
        rows=[]; cols=[]
        # Separate left and right contexts to distinguish syntactically different words.
        for words in docs:
            ids=[vocab.get(w,-1) for w in words]
            for i,word in enumerate(ids):
                if word<0:
                    continue
                for j in range(max(0,i-3),min(len(ids),i+4)):
                    if i!=j and ids[j]>=0:
                        rows.append(word); cols.append(ids[j]+(n if j>i else 0))
        mat=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,2*n)).tocsr().tocoo()
        r=np.bincount(mat.row,weights=mat.data,minlength=n)
        c=np.bincount(mat.col,weights=mat.data,minlength=2*n)**.75
        mat.data=np.maximum(0,np.log(mat.data*c.sum()/(r[mat.row]*c[mat.col])))
        mat=mat.tocsr();mat.eliminate_zeros()
        embed=normalize(TruncatedSVD(n_components=min(n-1,40),n_iter=5,random_state=113).fit_transform(mat))
        labels=MiniBatchKMeans(n_clusters=min(64,max(3,n//12)),batch_size=512,n_init=3,random_state=113).fit_predict(embed)
        self.classes_={w:"c"+str(labels[i]) for w,i in vocab.items()}
        transformed=[self.encode(doc) for doc in docs]
        self.vector_=TfidfVectorizer(token_pattern=r"\S+",lowercase=False,min_df=2,sublinear_tf=True).fit(transformed)
        return self

    def encode(self,words):
        groups=[self.classes_.get(w,"unk") for w in words]
        if not groups:return "empty"
        return " ".join([*groups,*[a+"_"+b for a,b in zip(groups,groups[1:])],"first_"+groups[0],"last_"+groups[-1]])

    def transform(self,x):
        a,b=pair_texts(x)
        cache={t:self.encode(TOKEN.findall(t.casefold())) for t in set(a)|set(b)}
        return self.vector_.transform([cache[t] for t in a])-self.vector_.transform([cache[t] for t in b])


def distributional():
    model=pair_estimator("pair_hybrid",113)
    union=model.named_steps["featureunion"]
    union.transformer_list.append(("distributional",WordClassDifference()))
    union.set_params(transformer_weights={"tfidf":1.,"style":1.,"distributional":.65})
    return model


class CategoricalKernel:
    """Ridge smoother with separate kernels for recurring profiles and changing context."""
    def kernel(self,left,right):
        def distance(cols):
            out=np.zeros((len(left),len(right)),dtype=np.float64)
            for c in cols:
                out+=left[:,c,None]!=right[None,:,c]
            return out/max(1,len(cols))
        if not self.profile_ or not self.context_:
            return np.exp(-4*distance(range(left.shape[1])))
        dp=distance(self.profile_)
        dc=distance(self.context_)
        kp=np.exp(-4*dp)
        kc=np.exp(-4*dc)
        return .10*kp+.20*kc+.35*kp*kc+.35*(dp==0)*(.20+.80*kc)

    def fit(self,x,y):
        y=np.asarray(y)
        self.classes_=np.unique(y)
        if len(self.classes_)!=2:raise ValueError("Binary kernel requires two classes")
        if len(x)>8200:
            keep,_=train_test_split(np.arange(len(x)),train_size=8200,stratify=y,random_state=113)
            x=x.iloc[keep];y=y[keep]
        self.columns_=[c for c in x if x[c].nunique()>1]
        if not self.columns_:raise ValueError("No varying attributes")
        fields=discover(x[self.columns_])
        self.profile_=[i for i,c in enumerate(self.columns_) if c in fields]
        self.context_=[i for i,c in enumerate(self.columns_) if c not in fields]
        self.encoder_=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=np.int32)
        self.train_=self.encoder_.fit_transform(x[self.columns_])
        n=len(x)
        gram=np.empty((n,n),dtype=np.float64,order="F")
        for start in range(0,n,192):
            stop=min(start+192,n)
            gram[:,start:stop]=self.kernel(self.train_[start:stop],self.train_).T
        gram.flat[::n+1]+=.70
        self.mean_=np.mean(y==self.classes_[1])
        self.coef_=cho_solve(cho_factor(gram,lower=True,overwrite_a=True,check_finite=False),
                            (y==self.classes_[1]).astype(float)-self.mean_,check_finite=False)
        return self

    def predict_proba(self,x):
        codes=self.encoder_.transform(x[self.columns_])
        pos=np.empty(len(codes))
        for start in range(0,len(x),192):
            stop=min(start+192,len(x))
            pos[start:stop]=self.mean_+self.kernel(codes[start:stop],self.train_)@self.coef_
        pos=np.clip(pos,.005,.995)
        return np.column_stack([1-pos,pos])
