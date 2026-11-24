"""Candidate families shared by checkpoint replay, timed training and prediction."""
import hashlib
import os
from pathlib import Path
import sys
import unicodedata

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "2"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.append(str(Path(__file__).resolve().parents[2] / "Agent"))

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from advanced.learners import candidate
from challengers import FACTORIES
from features import build
from paired_text import pair_estimator, pair_groups
from method_candidates import EditNeighbors, OrdinalCatBoost
import representation_candidates as representations
from objective import Objective, deployed_indices
from task_data import image_inputs, load_features, load_labeled

POOLS = {
    "pair": ("edit_odds", ["pair_hybrid", "edit_odds", "pair_tfidf", "latent_pair"]),
    "categorical": ("crossed_logit", ["catboost", "tabm", "crossed_logit", "trees"]),
    "image": ("color", ["gradient", "gradient_svm", "color", "neighbors"]),
    "table": ("linear", ["linear", "trees", "hist", "balanced_linear"]),
    "text": ("word", ["char", "word", "union", "nb_ratio"]),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_xy(spec, root, split):
    if split not in ("train", "validation"):
        raise ValueError("Experiment data access is limited to development splits")
    path = root / spec[split]
    labels = root / spec[split + "_labels"] if split + "_labels" in spec else None
    frame = load_labeled((path, labels), spec)
    return features(frame, spec, path.parent), frame[spec["target"]].to_numpy(str)


def features(frame, spec, directory):
    cols = spec["features"]
    if len(set(cols)) != len(cols) or spec["target"] in cols:
        raise ValueError("Feature list contains a duplicate or the target")
    x = frame[cols].copy()
    for key in spec.get("numeric", []):
        x[key] = pd.to_numeric(x[key], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if spec["modality"] == "image":
        return image_inputs(x, directory)
    return x


def family(spec, y):
    labels = np.unique(y)
    Objective(spec, labels)
    if spec.get("pairwise"):
        if len(labels) != 2 or len(spec["features"]) != 2 or spec["modality"] != "text":
            raise ValueError("An ordered comparison requires two text fields and two classes")
        return "pair"
    if spec["modality"] == "table" and len(labels) == 2 and spec.get("numeric") == []:
        return "categorical"
    return spec["modality"]


def groups(x, kind, group_field=None):
    if group_field is not None:
        if group_field not in x or kind in {"image", "pair"}:
            raise ValueError("Declared group field is incompatible with the input or grouping strategy")
        values = x[group_field]
        if values.isna().any():
            raise ValueError("Missing group key")
        keys = [" ".join(unicodedata.normalize("NFKC", str(value)).casefold().split()) for value in values]
        if not all(keys):
            raise ValueError("Empty group key")
        return pd.factorize(pd.Index(keys), sort=True)[0]
    if kind == "image":
        def pixel_key(path):
            with Image.open(path) as img:
                return hashlib.sha256(str((img.size,img.mode)).encode() + img.tobytes()).hexdigest()
        return pd.factorize(pd.Index([pixel_key(p) for p in x.iloc[:, 0]]), sort=True)[0]
    if kind != "pair":
        return pd.MultiIndex.from_frame(x).factorize(sort=True)[0]
    return pair_groups(x)


def fold_ids(y, group, seed):
    folds = np.full(len(y), -1)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for fold, (train, valid) in enumerate(splitter.split(np.zeros(len(y)), y, group)):
        if set(group[train]) & set(group[valid]):
            raise ValueError("Groups overlap")
        folds[valid] = fold
    if min(folds) < 0:
        raise ValueError("Incomplete folds")
    return folds


def fit_one(name, kind, x, y, classes):
    if set(y) != set(classes):
        raise ValueError("Training subset missing a class")
    if name=="profile_response_pool" and kind=="categorical":
        from response_structure import ResponseModel
        return ResponseModel("pool_blend","pool_blend").fit(x,y)
    if name=="text_margin_neighbors" and kind=="text":
        from text_neighbors import TextMarginNeighbors
        return TextMarginNeighbors().fit(x,y)
    if name in {"framing_rank","contrast_rank"}:
        from framing_rank import framing_estimator
        return framing_estimator(name).fit(x,y)
    if name=="aligned_fields":
        from aligned_fields import AlignedBackoff
        return AlignedBackoff().fit(x,y)
    if name in {"semantic_edits","segment_experts"}:
        from focus_candidates import SemanticEdits, SegmentExperts
        model=SemanticEdits() if name=="semantic_edits" else SegmentExperts()
        return model.fit(x,y)
    if name in {"word_classes","category_kernel"}:
        from transfer_candidates import distributional, CategoricalKernel
        model=distributional() if name=="word_classes" else CategoricalKernel()
        return model.fit(x,y)
    if name in {"graph_logit","graph_margin"}:
        from graph_rank import GraphRank
        return GraphRank(name=="graph_margin").fit(x,y)
    if name=="profile_backoff":
        from profile_boost import ProfileBackoff
        return ProfileBackoff().fit(x,y)
    if name in {"language_linear","topic_style_ranker"}:
        from language_rank import linear, TopicStyleRanker
        model = linear() if name == "language_linear" else TopicStyleRanker()
        return model.fit(x,y)
    if name in {"onehot_boost","profile_boost"}:
        from profile_boost import tree, ProfileBoost
        if name == "onehot_boost":
            return tree().fit(x,y,cat_features=list(x))
        return ProfileBoost().fit(x,y)
    if name in FACTORIES:
        return FACTORIES[name]().fit(x, (y == classes[1]).astype(int))
    if name in representations.TEXT or name in representations.TABLE:
        return representations.estimator(name,x,113).fit(x,y)
    if name == "edit_neighbors":
        model = EditNeighbors()
    elif name == "ordered_catboost":
        model = OrdinalCatBoost()
    elif name == "pair_hybrid_stronger_penalty":
        model = pair_estimator("pair_hybrid", 113)
        model.set_params(logisticregression__C=0.25)
    elif name.startswith("pair_"):
        model = pair_estimator(name, 113)
    elif name in {"catboost", "catboost_shallow"}:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(cat_features=list(x), iterations=250, depth=4 if name == "catboost_shallow" else 6, learning_rate=.08,
                                   random_seed=113, thread_count=2, verbose=False, allow_writing_files=False)
    elif name == "trees_min_leaf4":
        model = build("table", "trees", x, 113)
        model.steps[-1][1].set_params(min_samples_leaf=4)
    elif name in {"trees","linear","balanced_linear","word","char","gradient","color"}:
        model = build("table" if kind == "categorical" else kind, name, x, 113)
    else:
        model = candidate(kind, name, x, 113)
    return model.fit(x, y)


def probabilities(model, x, classes):
    if hasattr(model, "prob"):
        positive = model.prob(x)
        p = np.column_stack((1 - positive, positive))
    else:
        if not np.array_equal(np.asarray(model.classes_, dtype=str), classes):
            raise ValueError("Class order mismatch")
        p = model.predict_proba(x)
    if p.shape != (len(x), len(classes)) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError("Invalid class probabilities")
    return p


def infer(bundle, x):
    return np.mean([probabilities(m, x, bundle["classes"]) for m in bundle["members"]], axis=0)


def save_bundle(path, spec, classes, models, stage, threshold=None):
    keep = {k:spec[k] for k in ("metric","positive_class","modality","features","numeric","pairwise","target","allow_pickle_features","native_format","tsv_columns","json_types","group_field") if k in spec}
    bundle = dict(spec=keep, classes=classes, names=list(models), members=list(models.values()), stage=stage, decision_threshold=threshold)
    temp = path.with_suffix(".pending")
    joblib.dump(bundle, temp, compress=2)
    os.replace(temp, path)


def predict_file(path, input_file, output, *, bundle=None):
    if output.exists():
        raise FileExistsError(output)
    if bundle is None:
        bundle = joblib.load(path)
    spec = bundle["spec"]
    x = features(load_features(input_file, spec), spec, input_file.parent)
    prob = infer(bundle, x)
    frame = pd.DataFrame(prob, columns=["p_" + str(c) for c in bundle["classes"]])
    frame.insert(0, "prediction", bundle["classes"][deployed_indices(bundle, prob)])
    frame.to_csv(output, index=False)
