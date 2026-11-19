"""Research candidate library. All learned transforms are fitted within each fold."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import build, text_values, image_features


class NBLogRatio(BaseEstimator, ClassifierMixin):
    """One-vs-rest NB log-count ratios with discriminative logistic heads."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.heads_ = []
        for label in self.classes_:
            positive = np.asarray(y == label)
            a = 1 + np.asarray(X[positive].sum(axis=0)).ravel()
            b = 1 + np.asarray(X[~positive].sum(axis=0)).ravel()
            ratio = np.log((a/a.sum()) / (b/b.sum()))
            head = LogisticRegression(C=2, max_iter=500).fit(X.multiply(ratio), positive)
            self.heads_.append((ratio, head))
        return self

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        scores = np.column_stack([m.predict_proba(X.multiply(r))[:, 1] for r,m in self.heads_])
        return scores / scores.sum(axis=1, keepdims=True)


def pixels(frame):
    images = []
    for file in frame.iloc[:, 0]:
        with Image.open(file) as image:
            patch = ImageOps.exif_transpose(image).convert("L").resize((24,24), Image.Resampling.BILINEAR)
            images.append(np.asarray(patch, dtype=np.float32).ravel()/255)
    return np.stack(images)


class SmallCNN(ClassifierMixin, BaseEstimator):
    def __init__(self, seed=17, epochs=12):
        self.seed, self.epochs = seed, epochs

    @staticmethod
    def network(k):
        from torch import nn
        return nn.Sequential(nn.Conv2d(1,24,3,padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(24,48,3,padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
                             nn.Conv2d(48,64,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((3,3)),
                             nn.Flatten(), nn.Linear(576,k))

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_used_ = device
        self.classes_, labels = np.unique(y, return_inverse=True)
        model = self.network(len(self.classes_)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        data = torch.from_numpy(X.reshape(-1,1,24,24))
        targets = torch.as_tensor(labels, dtype=torch.long)
        for _ in range(self.epochs):
            for batch in torch.randperm(len(X)).split(64):
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(data[batch].to(device)), targets[batch].to(device))
                loss.backward()
                optimizer.step()
        self.weights_ = {k:v.detach().cpu() for k,v in model.state_dict().items()}
        return self

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        import torch
        model = self.network(len(self.classes_))
        model.load_state_dict(self.weights_)
        model.eval()
        with torch.inference_mode():
            batches = torch.from_numpy(X.reshape(-1,1,24,24)).split(128)
            return torch.cat([model(b).softmax(1) for b in batches]).numpy()


POOLS = {"table":["linear","trees","hist","rbf"], "text":["char","word","union","nb_ratio"],
         "image":["color","gradient","neighbors","cnn"]}


def candidate(mode, name, X, seed, pilot=False):
    if name in {"linear","trees","char","word","color","gradient"}:
        return build(mode,name,X,seed)
    if name == "gradient_svm":
        return make_pipeline(FunctionTransformer(image_features,
                                                 kw_args={"gradients":True}), StandardScaler(),
                             SVC(C=4, probability=True, random_state=seed))
    if mode=="table":
        nums = X.select_dtypes(include="number").columns.tolist()
        cats = [c for c in X if c not in nums]
        if name=="rbf":
            # Reuse the one-hot and numeric pipeline of the linear baseline.
            prep = build(mode,"linear",X,seed).steps[0][1]
            return make_pipeline(prep,SVC(C=4, probability=True, random_state=seed))
        prep = ColumnTransformer([("nums",SimpleImputer(strategy="median",keep_empty_features=True),nums),
                                  ("cats",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1),cats)])
        return make_pipeline(prep,HistGradientBoostingClassifier(max_iter=80, min_samples_leaf=8,random_state=seed))
    if mode=="text":
        flatten = FunctionTransformer(text_values)
        if name=="nb_ratio":
            return make_pipeline(flatten, CountVectorizer(binary=True, ngram_range=(1,2),max_features=60000), NBLogRatio())
        union = FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),max_features=40000)),
                              ("char",TfidfVectorizer(analyzer="char",ngram_range=(2,4),max_features=40000))])
        return make_pipeline(flatten,union,LogisticRegression(C=3,max_iter=400))
    transform = FunctionTransformer(pixels)
    if name=="neighbors":
        return make_pipeline(transform,KNeighborsClassifier(n_neighbors=5,weights="distance"))
    return make_pipeline(transform,SmallCNN(seed,epochs=4 if pilot else 12))
