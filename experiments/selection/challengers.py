"""Task-independent competitors for ordered text pairs and all-categorical tables."""
from itertools import combinations

import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, normalize


def sides(x):
    a,b=x.fillna("").astype(str).to_numpy().T
    return a,b


class EditOdds:
    def delta(self,x):
        a,b=sides(x)
        return (self.vocab.transform(a)-self.vocab.transform(b)).tocsr()

    def fit(self,x,y):
        a,b=sides(x)
        self.vocab=FeatureUnion([
            ("word",CountVectorizer(ngram_range=(1,2),min_df=2,binary=True,max_features=50000,
                                    token_pattern=r"(?u)\b\w+\b")),
            ("char",CountVectorizer(analyzer="char",ngram_range=(3,4),min_df=2,binary=True,max_features=50000))
        ]).fit(np.concatenate([a,b]))
        delta=self.delta(x)
        oriented=delta.multiply((2*np.asarray(y)-1)[:,None]).tocsr()
        win=np.asarray(oriented.maximum(0).sum(axis=0)).ravel()+1.
        lose=np.asarray((-oriented).maximum(0).sum(axis=0)).ravel()+1.
        self.weights=np.log((win/win.sum())/(lose/lose.sum()))
        self.model=LogisticRegression(C=4,fit_intercept=False,max_iter=500,random_state=113)
        self.model.fit(normalize(delta.multiply(self.weights)),y)
        return self

    def prob(self,x):
        return self.model.predict_proba(normalize(self.delta(x).multiply(self.weights)))[:,1]


class LatentPair:
    def features(self,x):
        a,b=sides(x)
        def encode(t):
            return normalize(self.svd.transform(self.words.transform(t)))
        l,r=encode(a),encode(b)
        return np.column_stack([l,r,l-r]),np.column_stack([r,l,r-l])

    def fit(self,x,y):
        a,b=sides(x)
        self.words=TfidfVectorizer(ngram_range=(1,2),min_df=2,sublinear_tf=True,max_features=45000)
        matrix=self.words.fit_transform(np.concatenate([a,b]))
        self.svd=TruncatedSVD(n_components=64,random_state=113).fit(matrix)
        f,r=self.features(x)
        self.model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1.,
                                                   early_stopping=False,random_state=113)
        self.model.fit(np.row_stack([f,r]),np.concatenate([y,1-y]))
        return self

    def prob(self,x):
        f,r=self.features(x)
        return .5*(self.model.predict_proba(f)[:,1]+self.model.predict_proba(r)[:,0])


class CrossedLogit:
    def data(self,x):
        codes=self.ordinal.transform(x)+1
        # Independent column vocabularies ensure that fields never collide.
        pair=[codes[:,i]*self.radix[j]+codes[:,j] for i,j in combinations(range(codes.shape[1]),2)]
        return np.column_stack([codes,*pair])

    def fit(self,x,y):
        self.ordinal=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=np.int64).fit(x)
        self.radix=[len(cat)+1 for cat in self.ordinal.categories_]
        self.encoder=OneHotEncoder(handle_unknown="ignore")
        matrix=self.encoder.fit_transform(self.data(x))
        self.model=LogisticRegression(C=.1,solver="liblinear",max_iter=400,random_state=113).fit(matrix,y)
        return self

    def prob(self,x):
        return self.model.predict_proba(self.encoder.transform(self.data(x)))[:,1]


class TabularNet:
    def net(self):
        from tabm import TabM
        return TabM.make(n_num_features=0,cat_cardinalities=self.sizes,d_out=2,n_blocks=2,
                         d_block=128,k=8,dropout=.15)

    def indices(self,x):
        import torch
        return torch.tensor(self.encoder.transform(x)+1,dtype=torch.long)

    def fit(self,x,y):
        import torch
        from torch.nn.functional import cross_entropy
        torch.set_num_threads(2)
        torch.manual_seed(113)
        torch.use_deterministic_algorithms(True)
        self.encoder=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=np.int64).fit(x)
        self.sizes=[len(col)+1 for col in self.encoder.categories_]
        device="cuda" if torch.cuda.is_available() else "cpu"
        self.fit_device=device
        model=self.net().to(device).train()
        features=self.indices(x).to(device)
        labels=torch.tensor(np.asarray(y),dtype=torch.long,device=device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        for epoch in range(32):
            for batch in torch.randperm(len(y),device=device).split(256):
                optimizer.zero_grad()
                logits=model(x_cat=features[batch])
                loss=cross_entropy(logits.flatten(0,1),labels[batch].repeat_interleave(model.k))
                loss.backward()
                optimizer.step()
        self.weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
        return self

    def prob(self,x):
        import torch
        torch.set_num_threads(2)
        model=self.net().eval()
        model.load_state_dict(self.weights)
        with torch.no_grad():
            return torch.cat([model(x_cat=batch).softmax(dim=-1)[:,:,1].mean(dim=1)
                              for batch in self.indices(x).split(256)]).numpy().astype(float)


FACTORIES={"edit_odds":EditOdds,"latent_pair":LatentPair,"crossed_logit":CrossedLogit,"tabm":TabularNet}
