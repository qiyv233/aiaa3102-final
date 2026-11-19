"""Offline ImageNet ResNet-18 features; saved models include the frozen encoder."""
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class FrozenResNet18(ClassifierMixin, BaseEstimator):
    def __init__(self, seed=17):
        self.seed = seed

    def _features(self, frame):
        import torch
        from torchvision.models import resnet18, ResNet18_Weights

        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = resnet18(weights=None)
        encoder.fc = torch.nn.Identity()
        encoder.load_state_dict(self.encoder_weights_)
        encoder.to(device).eval()
        preprocess = ResNet18_Weights.IMAGENET1K_V1.transforms()
        features = []
        with torch.inference_mode():
            for start in range(0, len(frame), 32):
                batch = []
                for path in frame.iloc[start:start+32, 0]:
                    with Image.open(path) as image:
                        batch.append(preprocess(ImageOps.exif_transpose(image).convert("RGB")))
                features.append(encoder(torch.stack(batch).to(device)).cpu().numpy())
        return np.concatenate(features)

    def fit(self, X, y):
        import torch
        default = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
        path = Path(os.environ.get("AIAA_RESNET18_WEIGHTS", default))
        with path.open("rb") as stream:
            if not hashlib.file_digest(stream, "sha256").hexdigest().startswith("f37072fd"):
                raise ValueError("Unexpected ResNet18 ImageNet V1 checkpoint")
        weights = torch.load(path, map_location="cpu", weights_only=True)
        self.encoder_weights_ = {k:v for k,v in weights.items() if not k.startswith("fc.")}
        X = self._features(X)
        self.scaler_ = StandardScaler().fit(X)
        self.head_ = LogisticRegression(C=0.1, max_iter=500, random_state=self.seed)
        self.head_.fit(self.scaler_.transform(X), y)
        self.classes_ = self.head_.classes_
        return self

    def predict_proba(self, X):
        return self.head_.predict_proba(self.scaler_.transform(self._features(X)))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
