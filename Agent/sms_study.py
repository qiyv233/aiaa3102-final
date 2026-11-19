"""Reproduce the offline study. Expects the original UCI SMS Spam Collection ZIP."""
import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


ARCHIVE_SHA = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
ARMS = [("baseline", "baseline", 1), ("fixed2", "fixed", 2),
        ("rule2", "rule", 2), ("rule3", "rule", 3)]


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def study(archive, root):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("Archive hash differs from the pinned UCI release.")
    with ZipFile(archive) as z:
        lines = z.read("SMSSpamCollection").decode("utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        label, sms = line.split("\t", 1)
        key = " ".join(unicodedata.normalize("NFKC", sms).casefold().split())
        if label not in ("ham", "spam") or not key:
            raise ValueError(f"Invalid row {i+1}")
        rows.append((f"s{i+1:05d}", label, sms, key))
    raw = pd.DataFrame(rows, columns=["id", "label", "text", "key"])
    conflicts = raw.groupby("key").label.transform("nunique") > 1
    clean = raw.loc[~conflicts].drop_duplicates("key").drop(columns="key")
    train, hold = train_test_split(clean, test_size=0.2, stratify=clean.label, random_state=20261110)
    root.mkdir(parents=True, exist_ok=False)
    data = root / "data"
    data.mkdir()
    train.to_csv(data / "train.csv", index=False)
    hold.drop(columns="label").to_csv(data / "test.csv", index=False)
    dump(data / "task.json", dict(train="train.csv", test="test.csv", target="label", id="id",
                                  metric="macro_f1", seed=17, output="label"))
    agent = str(Path(__file__).with_name("agent.py"))
    for name, strategy, cap in ARMS:
        out = root / name
        subprocess.run([sys.executable, agent, "run", "--task", str(data / "task.json"), "--out", str(out),
                        "--strategy", strategy, "--max-trials", str(cap)], check=True)
        replay = out / "reload.csv"
        subprocess.run([sys.executable, agent, "predict", "--model", str(out / "model.joblib"),
                        "--input", str(data / "test.csv"), "--output", str(replay)], check=True)
        if replay.read_bytes() != (out / "predictions.csv").read_bytes():
            raise AssertionError("Prediction replay differs.")
    # Score all frozen outputs together after all four runs finish.
    results, preds = {}, {}
    truth = hold.label.to_numpy()
    for name, _, _ in ARMS:
        out = root / name
        pred = pd.read_csv(out / "predictions.csv")
        assert pred.id.tolist() == hold.id.tolist() and pred.label.isin(["ham", "spam"]).all()
        decisions = [json.loads(x) for x in (out / "run.jsonl").read_text(encoding="utf-8").splitlines()]
        chosen = decisions[-1]
        p = pred.label.to_numpy()
        tn, fp, fn, tp = confusion_matrix(truth, p, labels=["ham","spam"]).ravel().tolist()
        results[name] = {**chosen, "macro_f1": f1_score(truth, p, average="macro"),
                         "spam_precision": tp/(tp+fp) if tp+fp else 0, "spam_recall": tp/(tp+fn),
                         "false_spam": fp, "missed_spam": fn}
        preds[name] = p
    # Paired stratified bootstrap, conditional on this single split and the fitted models.
    rng = np.random.default_rng(20261110)
    groups = [np.flatnonzero(truth == k) for k in ["ham", "spam"]]
    y = truth == "spam"
    def macro(indices, name):
        a, b = y[indices], preds[name][indices] == "spam"
        tp, tn = (a & b).sum(), (~a & ~b).sum()
        err = (a != b).sum()
        return tp / (2*tp + err) + tn / (2*tn + err)
    diffs = []
    for _ in range(3000):
        ix = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        diffs.append(macro(ix,"rule3")-macro(ix,"baseline"))
    dump(root / "summary.json", {"source_sha256": ARCHIVE_SHA, "raw": len(raw), "clean": len(clean),
         "conflict_rows_removed": int(conflicts.sum()), "train": train.label.value_counts().to_dict(),
         "holdout": hold.label.value_counts().to_dict(), "results": results,
         "rule3_minus_baseline_bootstrap_95": np.quantile(diffs,[0.025,0.975]).tolist(),
         "llm_calls": 0})
    hold[["id","label"]].to_csv(root / "evaluation_labels.csv", index=False)
    print((root / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("new_folder", type=Path)
    args = p.parse_args()
    study(args.archive, args.new_folder)
