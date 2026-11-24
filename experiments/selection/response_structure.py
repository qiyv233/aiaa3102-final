"""Binary categorical response models using repeated profiles, situations and aligned preferences."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier

from aligned_fields import AlignedFields
from profile_boost import ProfileBackoff, discover


def combine(frame, columns):
    # Length prefixes make composite values unambiguous for arbitrary category strings.
    value=None
    for column in columns:
        s=frame[column].astype(str)
        encoded=s.str.len().astype(str)+":"+s
        value=encoded if value is None else value+encoded
    return value


class Structure:
    def fit(self,x):
        self.alignment=AlignedFields().fit(x)
        self.columns=[c for c in x if x[c].nunique()>1]
        stable=set(discover(x))|set(self.alignment.columns_)
        self.stable=[c for c in self.columns if c in stable]
        self.context=[c for c in self.columns if c not in stable]
        self.anchor=self.alignment.context_
        return self

    def transform(self,x):
        raw=x.reset_index(drop=True)
        ext,_=self.alignment.transform(raw)
        columns={f"f{i}":raw[c] for i,c in enumerate(self.columns)}
        for i,c in enumerate(self.alignment.names_ if self.anchor else []):
            columns[f"a{i}"]=ext[c].fillna(-99).round(2).astype(str)
        if self.stable: columns["profile"]=combine(raw,self.stable)
        if self.context: columns["situation"]=combine(raw,self.context)
        if self.anchor and self.stable: columns["profile_anchor"]=combine(raw,[*self.stable,self.anchor])
        base=pd.DataFrame(columns)
        crossed={}
        if self.anchor:
            for i,c in enumerate(self.stable):
                crossed[f"anchor{i}"]=combine(raw,[self.anchor,c])
            for i,c in enumerate(self.alignment.names_):
                crossed[f"aligned{i}"]=combine(base.assign(anchor_value=raw[self.anchor]),["anchor_value",f"a{i}"])
        for i,(a,b) in enumerate(combinations(self.context,2)):
            crossed[f"s{i}"]=combine(raw,[a,b])
        for i,c in enumerate(self.context):
            if self.stable and c!=self.anchor:
                crossed[f"pc{i}"]=combine(raw,[*self.stable,c])
        # Pool population/context interactions separately from more granular profile effects.
        for i,c in enumerate(self.stable):
            for j,q in enumerate(self.context):
                if q!=self.anchor:
                    crossed[f"pop{i}_{j}"]=combine(raw,[c,q])
        return base,pd.DataFrame(crossed,index=raw.index)


class Factorized:
    def fit(self,x,y):
        self.encoder=OneHotEncoder(handle_unknown="ignore")
        a=self.encoder.fit_transform(x).tocsr()
        a2=a.multiply(a).tocsr()
        d=a.shape[1]
        rng=np.random.default_rng(113)
        initial=np.r_[0.,np.zeros(d),rng.normal(0,.035,(d,6)).ravel()]
        def objective(params):
            w=params[1:d+1]; v=params[d+1:].reshape(d,6)
            av=a@v
            eta=params[0]+a@w+.5*((av**2).sum(1)-a2@(v**2).sum(1))
            err=expit(eta)-y
            loss=np.sum(np.logaddexp(0,eta)-y*eta)+1.5*(w@w)+6.5*np.sum(v*v)
            gv=a.T@(err[:,None]*av)-(a2.T@err)[:,None]*v+13*v
            grad=np.r_[err.sum(),a.T@err+3*w,gv.ravel()]
            return loss,grad
        opt=minimize(objective,initial,method="L-BFGS-B",jac=True,options={"maxiter":180,"ftol":1e-8,"maxcor":8})
        if not np.isfinite(opt.fun) or (not opt.success and opt.status!=1):raise ValueError(opt.message)
        self.iterations=int(opt.nit)
        self.converged=bool(opt.success)
        self.bias=opt.x[0];self.w=opt.x[1:d+1];self.v=opt.x[d+1:].reshape(d,6)
        return self

    def probability(self,x):
        a=self.encoder.transform(x).tocsr()
        eta=self.bias+a@self.w+.5*(((a@self.v)**2).sum(1)-a.multiply(a)@(self.v**2).sum(1))
        return expit(eta)


class ResponseModel:
    def __init__(self,seen="baseline",unseen="baseline"):
        self.routes=(seen,unseen)

    def fit(self,x,y,research=False):
        self.classes_=np.unique(y)
        if len(self.classes_)!=2:raise ValueError("Response model requires binary labels")
        self.base=ProfileBackoff().fit(x,y)
        needed=set().union(*(MENU[name] for name in self.routes))
        if research:needed.update(["logit","boost","factor"])
        self.fitted=sorted(needed&{"logit","boost","factor"})
        if not self.fitted:return self
        self.structure=Structure().fit(x)
        base,extra=self.structure.transform(x)
        if "logit" in self.fitted:
            self.encoder=OneHotEncoder(handle_unknown="ignore")
            combined=pd.concat([base,extra],axis=1)
            design=self.encoder.fit_transform(combined)
            # Heavier shrinkage for numerous population interactions than for profile preferences.
            scales=[.38 if key.startswith("pop") else .50 if key.startswith("pc") else
                    1.45 if key=="profile_anchor" else 1.0 for key in combined]
            self.scale=np.repeat(scales,[len(v) for v in self.encoder.categories_])
            self.linear=LogisticRegression(C=.7,solver="lbfgs",max_iter=400).fit(design.multiply(self.scale),y)
        if "boost" in self.fitted:
            self.boost=CatBoostClassifier(iterations=650,depth=6,learning_rate=.055,one_hot_max_size=32,
                        verbose=False,allow_writing_files=False,thread_count=2,random_seed=113)
            self.boost.fit(base,y,cat_features=list(base))
        if "factor" in self.fitted:
            self.factor=Factorized().fit(base,(np.asarray(y)==self.classes_[1]).astype(float))
        return self

    def components(self,x):
        from models import probabilities
        known=self.base.known(x)
        p={"baseline":probabilities(self.base,x,self.classes_)[:,1]}
        for name,model in zip(["cat","trees"],self.base.reference_):
            p[name]=probabilities(model,x,self.classes_)[:,1]
        if self.fitted:
            base,extra=self.structure.transform(x)
            if "logit" in self.fitted:
                design=self.encoder.transform(pd.concat([base,extra],axis=1))
                p["logit"]=self.linear.predict_proba(design.multiply(self.scale))[:,1]
            if "boost" in self.fitted:p["boost"]=self.boost.predict_proba(base)[:,1]
            if "factor" in self.fitted:p["factor"]=self.factor.probability(base)
        return known,p

    def predict_proba(self,x):
        known,p=self.components(x)
        pos=np.where(known,blend(p,self.routes[0]),blend(p,self.routes[1]))
        return np.column_stack((1-pos,pos))


MENU={"baseline":{"baseline":1},"ridge_blend":{"baseline":.5,"logit":.5},
      "boost_blend":{"baseline":.5,"boost":.5},"factor_blend":{"baseline":.5,"factor":.5},
      "pool":{"logit":1/3,"boost":1/3,"factor":1/3},
      "pool_blend":{"baseline":.5,"logit":1/6,"boost":1/6,"factor":1/6},
      "cat_only":{"cat":1},"trees_only":{"trees":1},"cat_heavy":{"cat":.75,"trees":.25},
      "trees_heavy":{"cat":.25,"trees":.75}}


def blend(p,name):
    return sum(p[key]*weight for key,weight in MENU[name].items())
