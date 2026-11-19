"""Compare locked policies and replay each model before reading development labels."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path, help="JSON list with name, manifest and evaluation_labels")
    parser.add_argument("out", type=Path)
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--pools", nargs="+", choices=["classic","expanded","adaptive","transfer","categorical"],
                        default=["classic","expanded"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    tasks = json.loads(args.tasks.read_text(encoding="utf-8-sig"))
    pending = []
    agent = str(Path(__file__).with_name("agent.py"))
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for item in tasks:
        manifest = args.tasks.parent / item["manifest"]
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        for pool in args.pools:
            out = args.out / item["name"] / pool
            tick = time.monotonic()
            subprocess.run([sys.executable, agent, "run", "--task", str(manifest),
                            "--out", str(out), "--strategy", "rule", "--pool", pool,
                            "--max-trials", "3", "--seconds", str(args.seconds)], check=True,env=env)
            wall = time.monotonic()-tick
            subprocess.run([sys.executable,agent,"predict","--model",str(out/"model.joblib"),
                            "--input",str(manifest.parent/spec["test"]),"--output",str(out/"replay.csv")],
                           check=True,env=env)
            content = (out/"predictions.csv").read_bytes()
            if content != (out/"replay.csv").read_bytes():
                raise RuntimeError("Replay mismatch")
            logs=[json.loads(s) for s in (out/"run.jsonl").read_text(encoding="utf-8").splitlines()]
            trials=[json.loads(s) for s in (out/"experiments.jsonl").read_text(encoding="utf-8").splitlines()]
            pending.append((item, spec, out, dict(task=item["name"],pool=pool,model=logs[-1]["selected"],
                           status=logs[-1]["status"],seconds=round(wall,2),attempts=len(trials),
                           actions=[s["candidate"] for s in trials], failed=sum(s["status"]=="failed" for s in trials),
                           split=logs[0]["validation_sha256"],replay_sha256=hashlib.sha256(content).hexdigest())))
    results=[]
    for item,spec,out,row in pending:
        labels=pd.read_csv(args.tasks.parent/item["evaluation_labels"],dtype=str,keep_default_na=False)
        pred=pd.read_csv(out/"predictions.csv",dtype=str,keep_default_na=False)
        if not labels[spec["id"]].equals(pred[spec["id"]]):
            raise ValueError("Evaluation IDs out of order")
        y=labels[spec["target"]]
        if spec.get("output","label")=="probabilities":
            probs=pred.drop(columns=[spec["id"]]).astype(float)
            guesses=probs.idxmax(axis=1)
        else:
            guesses=pred[spec.get("prediction_column",spec["target"])]
        if spec["metric"]=="log_loss":
            totals = probs.sum(axis=1)
            if not np.allclose(totals,1,atol=1e-5) or (probs<0).any().any():
                raise ValueError("Invalid exported probabilities")
            value=log_loss(y,probs.div(totals,axis=0),labels=probs.columns.tolist())
        elif spec["metric"]=="macro_f1":
            value=f1_score(y,guesses,average="macro")
        else:
            raise ValueError("Development suite supports Macro-F1 and log loss")
        results.append(dict(row,metric=spec["metric"],score=round(float(value),6),rows=len(y)))
    for item in tasks:
        assert len({r["split"] for r in results if r["task"]==item["name"]})==1
    (args.out/"summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
