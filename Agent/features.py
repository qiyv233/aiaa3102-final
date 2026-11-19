from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def text_values(frame):
    return frame.fillna("").astype(str).agg(" ".join, axis=1).to_numpy()


def image_features(frame, *, gradients=False):
    rows = []
    for path in frame.iloc[:, 0]:
        with Image.open(path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB").resize((32, 32))
            rgb = np.asarray(img, dtype=float) / 255
        base = rgb.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3)).ravel()
        if gradients:
            dy, dx = np.gradient(rgb.mean(axis=2))
            bins = ((np.arctan2(dy, dx) % np.pi) * 8 / np.pi).astype(int) % 8
            mag = np.hypot(dx, dy)
            hist = [np.bincount(bins[i:i+8, j:j+8].ravel(),
                                weights=mag[i:i+8, j:j+8].ravel(), minlength=8)
                    for i in range(0, 32, 8) for j in range(0, 32, 8)]
            base = np.concatenate([base, *hist])
        rows.append(base)
    return np.stack(rows)


CANDIDATES = {
    "table": ["linear", "trees", "balanced_linear"],
    "text": ["char", "word", "balanced_char"],
    "image": ["color", "gradient", "balanced_gradient"],
}


def infer_modality(frame):
    if len(frame.columns) == 1:
        values = frame.iloc[:, 0].dropna().astype(str).head(64)
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        if len(values) and all(Path(x).suffix.lower() in extensions for x in values):
            return "image"
        if len(values) and (values.str.len().median() >= 24 or
                            values.str.count(r"\s").median() >= 2):
            return "text"
    return "table"


def build(mode, name, frame, seed):
    weight = "balanced" if name.startswith("balanced") else None
    classifier = LogisticRegression(class_weight=weight, max_iter=500, random_state=seed)
    if mode == "text":
        char = "char" in name
        vectorizer = TfidfVectorizer(analyzer="char" if char else "word",
                                     ngram_range=(1, 3) if char else (1, 2),
                                     sublinear_tf=True, max_features=60000)
        return make_pipeline(FunctionTransformer(text_values), vectorizer, classifier)
    if mode == "image":
        transform = FunctionTransformer(image_features, kw_args={"gradients": "gradient" in name})
        return make_pipeline(transform, StandardScaler(), classifier)
    nums = frame.select_dtypes(include="number").columns.tolist()
    cats = [c for c in frame.columns if c not in nums]
    preprocess = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                  StandardScaler()), nums),
        ("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                                      OneHotEncoder(handle_unknown="ignore")), cats),
    ])
    if name == "trees":
        classifier = ExtraTreesClassifier(n_estimators=80, n_jobs=2, random_state=seed)
    return make_pipeline(preprocess, classifier)
