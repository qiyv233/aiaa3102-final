"""Create reproducible development tasks and verify prediction in a fresh process."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


def make_task(folder, mode):
    folder.mkdir(parents=True)
    if mode == "table":
        data = load_breast_cancer(as_frame=True)
        df = data.data.copy()
        df["label"] = data.target.astype(str)
        df["radius_band"] = np.where(df.iloc[:, 0] < 15, "small", "large")
        df.loc[::13, df.columns[1]] = np.nan
        metric, output = "roc_auc", "positive_probability"
    elif mode == "image":
        data = load_digits()
        rows = []
        for i in range(360):
            name = f"digit_{i}.png"
            Image.fromarray((data.images[i] * 15).astype("uint8")).save(folder / name)
            rows.append((name, str(data.target[i])))
        df = pd.DataFrame(rows, columns=["path", "label"])
        metric, output = "log_loss", "probabilities"
    else:
        rows = [(f"{prefix} {noun} with {adj} {i}", label)
                for i in range(8) for prefix in ("request", "issue", "check")
                for adj in ("new", "old") for label, noun in
                (("account", "password"), ("network", "router"), ("display", "screen"))]
        df = pd.DataFrame(rows, columns=["description", "label"])
        metric, output = "macro_f1", "label"
    df.insert(0, "id", [f"{i:05d}" for i in range(len(df))])
    train, test = train_test_split(df, test_size=0.2, stratify=df.label, random_state=27)
    train.to_csv(folder / "train.csv", index=False)
    test.drop(columns="label").to_csv(folder / "test.csv", index=False)
    spec = dict(train="train.csv", test="test.csv", id="id", target="label", metric=metric, output=output)
    if mode == "table":
        spec.update(positive_class="1", prediction_column="probability")
    (folder / "task.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=False)
    agent = str(Path(__file__).with_name("agent.py"))
    for mode in ("table", "text", "image"):
        task = args.folder / mode
        test = make_task(task, mode)
        for strategy in ("baseline", "rule"):
            out = task / strategy
            subprocess.run([sys.executable, agent, "run", "--task", str(task / "task.json"),
                            "--out", str(out), "--strategy", strategy], check=True)
            replay = out / "replay.csv"
            subprocess.run([sys.executable, agent, "predict", "--model", str(out / "model.joblib"),
                            "--input", str(task / "test.csv"), "--output", str(replay)], check=True)
            assert replay.read_bytes() == (out / "predictions.csv").read_bytes()
            pred = pd.read_csv(replay, dtype={"id": str})
            assert pred.id.tolist() == test.id.tolist() and not pred.isna().any().any()
            if mode == "image":
                assert list(pred.columns[1:]) == list("0123456789")
                assert np.allclose(pred.iloc[:, 1:].sum(axis=1), 1)
            assert json.loads((out / "run.jsonl").read_text(encoding="utf-8").splitlines()[0])["mode"] == mode
    print("Verified all six runs, saved models, ordered IDs and image class probabilities.")
