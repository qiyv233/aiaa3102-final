"""Train one example baseline and verify that a new process reproduces its predictions."""
import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def read_ids(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row["id"] for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", choices=["table", "text", "image"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    mode = args.example
    data = root / "examples" / mode
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = root / "runs" / f"{mode}-{stamp}"
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "2"
    agent = [sys.executable, str(root / "agent.py")]
    subprocess.run(agent + ["run", "--task", str(data / "task.json"), "--strategy",
                   "baseline", "--out", str(out)], check=True, env=env)
    replay = out / "replay.csv"
    subprocess.run(agent + ["predict", "--model", str(out / "model.joblib"),
                   "--input", str(data / "test.csv"), "--output", str(replay)], check=True, env=env)
    ids = read_ids(replay)
    if ids != read_ids(data / "test.csv") or replay.read_bytes() != (out / "predictions.csv").read_bytes():
        raise RuntimeError("Prediction replay or test ID order differs.")
    print(f"PASS {mode}: {len(ids)} predictions, matching IDs and replay. Output: {out}")


if __name__ == "__main__":
    main()
