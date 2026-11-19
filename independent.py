"""Run the locked three-way comparison on one previously unused dataset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def prepare(name, raw):
    if name == "wdbc":
        ds = load_breast_cancer(as_frame=True)
        frame = ds.frame.rename(columns={"target": "label"})
        frame.insert(0, "id", ["w" + str(n) for n in frame.index])
        train, test = train_test_split(frame, stratify=frame.label, test_size=.25, random_state=11919)
        return train, test, "table"
    with (raw / "movie_reviews.zip").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != "a41211ae685019137410268134db6a1a14428c89b671eb83056151a878539008":
            raise ValueError("NLTK corpus hash mismatch")
    rows = []
    with ZipFile(raw / "movie_reviews.zip") as archive:
        for member in sorted(archive.namelist()):
            path = Path(member)
            if path.suffix == ".txt" and path.parent.name in {"pos", "neg"}:
                rows.append((f"r{len(rows)}", archive.read(member).decode("utf-8"), path.parent.name,
                             int(path.name[2:5])))
    frame = pd.DataFrame(rows, columns=["id", "text", "label", "fold"])
    train = frame.loc[frame.fold < 800].drop(columns="fold")
    test = frame.loc[frame.fold >= 800].drop(columns="fold")
    assert (len(train), len(test)) == (1600, 400)
    return train, test, "text"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["wdbc", "movies"])
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=False)
    train, test, modality = prepare(args.name, args.raw)
    train.to_csv(folder / "train.csv", index=False)
    test.drop(columns="label").to_csv(folder / "test.csv", index=False)
    test[["id", "label"]].to_csv(folder / "evaluation_labels.csv", index=False)
    spec = dict(train="train.csv", test="test.csv", target="label", id="id", output="probabilities",
                metric="macro_f1", modality=modality, seed=113)
    task = folder / "task-113.json"
    task.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    solutions = [("adaptive", "fixed"), ("adaptive", "strong_fixed"), ("reviewed", "rule")]
    cmd = [sys.executable, str(Path(__file__).parent / "Agent" / "agent.py")]
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1")
    for pool, strategy in solutions:
        out = folder / f"113-{pool}-{strategy}"
        subprocess.run(cmd + ["run", "--task", str(task), "--pool", pool, "--strategy", strategy,
                              "--max-trials", "3", "--seconds", "180", "--out", str(out)], check=True, env=env)
        subprocess.run(cmd + ["predict", "--model", str(out / "model.joblib"), "--input", str(folder / "test.csv"),
                              "--output", str(out / "replay.csv")], check=True, env=env)
        assert (out / "predictions.csv").read_bytes() == (out / "replay.csv").read_bytes()

    rows = []
    for pool, strategy in solutions:
        out = folder / f"113-{pool}-{strategy}"
        events = [json.loads(s) for s in (out / "experiments.jsonl").read_text(encoding="utf-8").splitlines()]
        run = [json.loads(s) for s in (out / "run.jsonl").read_text(encoding="utf-8").splitlines()]
        pred = pd.read_csv(out / "predictions.csv")
        assert pred.id.tolist() == test.id.tolist()
        prob = pred.drop(columns="id")
        assert set(prob.columns) == set(train.label.astype(str)) and np.isfinite(prob).all().all()
        assert (prob >= 0).all().all() and np.allclose(prob.sum(axis=1), 1)
        rows.append(dict(pool=pool, strategy=strategy, score=f1_score(test.label.astype(str), prob.idxmax(axis=1),
                         average="macro"), selected=run[-1]["selected"], seconds=run[-1]["seconds"],
                         status=run[-1]["status"], attempts=[e["candidate"] for e in events],
                         failures=[e for e in events if e["status"] != "ok"], split=run[0]["validation_sha256"]))
    assert len({r["split"] for r in rows}) == 1
    columns = [c for c in train if c not in {"label", "id"}]
    keys = set(train[columns].itertuples(index=False, name=None))
    duplicates = sum(x in keys for x in test[columns].itertuples(index=False, name=None))
    report = dict(train=len(train), test=len(test), overlapping_test_rows=duplicates, runs=rows)
    (folder / "independent.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report, flush=True)


if __name__ == "__main__":
    main()
