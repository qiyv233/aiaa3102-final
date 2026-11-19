"""Leave one review source out. Run from the SuperAgent project with its requirements."""
import argparse
import hashlib
import json
import subprocess
import sys
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import joblib
import pandas as pd
from sklearn.metrics import f1_score, recall_score

import agent


DOMAINS = {"amazon": "amazon_cells", "imdb": "imdb", "yelp": "yelp"}
ARCHIVE = "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343"


def read_data(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE:
        raise ValueError("Different UCI archive version.")
    rows = []
    with ZipFile(path) as z:
        for source, stem in DOMAINS.items():
            text_file = z.read(f"sentiment labelled sentences/{stem}_labelled.txt").decode()
            # IMDb contains Unicode line separators inside two reviews; LF delimits records.
            lines = text_file.rstrip("\r\n").split("\n")
            assert len(lines) == 1000
            for i, line in enumerate(lines):
                text, label = line.rstrip("\r").rsplit("\t", 1)
                key = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
                if label not in {"0", "1"} or not key:
                    raise ValueError("Invalid observation.")
                rows.append((f"{source}-{i}", source, text, label, key))
    raw = pd.DataFrame(rows, columns=["id","source","text","label","key"])
    groups = raw.groupby("key")
    # Exclude cross-domain duplicates to avoid assigning an ambiguous source.
    ok = (groups.source.transform("nunique") == 1) & (groups.label.transform("nunique") == 1)
    return raw[ok].drop_duplicates("key")


def study(archive, out):
    clean = read_data(archive)
    out.mkdir(parents=True, exist_ok=False)
    scored = []
    for domain in DOMAINS:
        test = clean[clean.source == domain]
        other = clean[clean.source != domain]
        for scenario in ("natural", "stress"):
            train = other
            if scenario == "stress":
                negatives = other[other.label == "0"]
                positives = other[other.label == "1"].sample(n=len(negatives)//5, random_state=17)
                train = pd.concat([negatives, positives]).sort_index()
            root = out / domain / scenario
            root.mkdir(parents=True)
            train[["id","text","label"]].to_csv(root/"train.csv", index=False)
            test[["id","text"]].to_csv(root/"test.csv", index=False)
            spec = dict(train="train.csv", test="test.csv", target="label", id="id",
                        metric="macro_f1", seed=17, features=["text"], output="label")
            (root/"task.json").write_text(json.dumps(spec), encoding="utf-8")
            for strategy in ("fixed","rule"):
                dest = root / strategy
                subprocess.run([sys.executable, str(Path(__file__).with_name("agent.py")), "run",
                                "--task", str(root/"task.json"), "--out", str(dest), "--pool", "classic",
                                "--strategy", strategy, "--max-trials", "2", "--seconds", "600"], check=True)
                scored.append((domain, scenario, strategy, train.label.value_counts().to_dict(),
                               test, root, dest))
    # Open evaluator only after all model selection and prediction have finished.
    results = []
    for domain, scenario, strategy, counts, test, root, dest in scored:
        pred = pd.read_csv(dest/"predictions.csv", dtype=str)
        assert pred.id.tolist() == test.id.tolist() and pred.label.isin(["0","1"]).all()
        agent.predict(joblib.load(dest/"model.joblib"), root/"test.csv", dest/"replay.csv")
        assert (dest/"predictions.csv").read_bytes() == (dest/"replay.csv").read_bytes()
        events = [json.loads(x) for x in (dest/"experiments.jsonl").read_text(encoding="utf-8").splitlines()]
        runs = [json.loads(x) for x in (dest/"run.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(events)==2 and all(e["status"]=="ok" for e in events)
        results.append(dict(domain=domain, scenario=scenario, strategy=strategy,
                            training_counts=counts, test_n=len(test), split=runs[0]["validation_sha256"],
                            first_recall=events[0]["recall"], second=events[1]["candidate"],
                            selected=runs[-1]["selected"], validation=runs[-1]["validation_score"],
                            macro_f1=f1_score(test.label, pred.label, average="macro"),
                            positive_recall=recall_score(test.label,pred.label,pos_label="1")))
        test[["id","label"]].to_csv(root/"evaluation_labels.csv", index=False)
    for a,b in zip(results[::2],results[1::2]):
        assert a["split"] == b["split"]
    (out/"summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for r in results:
        print(r["domain"],r["scenario"],r["strategy"],r["second"],r["selected"],round(r["macro_f1"],4))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("new_output",type=Path)
    args=p.parse_args()
    study(args.archive,args.new_output)
