"""Three-trial policy. All observations come from the training side."""
import importlib.util
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def use_rgb_transfer(frame):
    if not all(importlib.util.find_spec(k) is not None for k in ("torch","torchvision")):
        return False
    from torch.hub import get_dir
    default=Path(get_dir())/"checkpoints/resnet18-f37072fd.pth"
    if not Path(os.environ.get("AIAA_RESNET18_WEIGHTS",default)).is_file():
        return False
    sampled=frame.iloc[:24,0]
    colorful=0
    for path in sampled:
        with Image.open(path) as img:
            rgb=np.asarray(ImageOps.exif_transpose(img).convert("RGB").resize((32,32)),dtype=float)
            colorful+=np.ptp(rgb,axis=2).mean()>=8
    return bool(len(sampled) and colorful*2>=len(sampled))


def initial_choices(mode, frame):
    if mode=="table":
        return ["linear","rbf" if len(frame)<=5000 else "trees","hist"]
    if mode=="text":
        return ["char","word","balanced_char","union"]
    return ["color","gradient_svm","resnet18","cnn","gradient"]


def action(mode, choices, history, frame, labels):
    if not history or mode=="table":
        return choices[0], "Use the predeclared baseline or tabular model order."
    if mode=="text" and "word" in choices:
        return "word", "Compare word features with the character baseline."
    if mode=="text":
        counts=labels.value_counts()
        last=next((r for r in reversed(history) if r.get("status")=="ok"),None)
        imbalance=counts.max()>=2*counts.min()
        low_recall=last is not None and min(last["recall"].values())<0.6
        target="balanced_char" if imbalance or low_recall else "union"
        return (target if target in choices else choices[0]), f"Imbalanced counts={imbalance}, low validation recall={low_recall}."
    if len(history)==1:
        try:
            transfer=use_rgb_transfer(frame)
        except (ImportError,OSError,RuntimeError):
            transfer=False
        if transfer:
            return "resnet18", "Training images contain color and a local pretrained encoder is available."
        return "gradient_svm","Try a local gradient representation."
    if "cnn" in choices and importlib.util.find_spec("torch") is not None:
        return "cnn","Compare a full small CNN under the remaining training time."
    return ("gradient" if "gradient" in choices else choices[0]), "Use the remaining image candidate."
