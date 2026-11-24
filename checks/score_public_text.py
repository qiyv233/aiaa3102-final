"""Evaluate a frozen intent classifier against its separately supplied public validation data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/selection"))
from models import np, pd, sha
from sklearn.metrics import f1_score

VALIDATION_HASHES={
    "hwu":"eb01a1b121c26f713d97a92e464012bbffe50d79e0111930e0dceadbc5e3addb",
    "banking_77":"b3dccbc70f6dcb4fd7d8d57921e2a1e96be768b4e30a1062d7259c287599c426",
}


def score(name, run, data, validation):
    def read(file):
        return json.loads((run/file).read_text(encoding="utf-8-sig"))
    result=read("result.json")
    spec=read("task.json")
    assert result["stage"]=="full_refit" and result["run_status"]=="completed"
    assert result["model_sha256"]==sha(run/"model.joblib") and result["contract_sha256"]==sha(run/"task.json")
    assert result["challenger_lock_sha256"]==sha(run/"challenger.json")
    assert spec["features"]==["text"] and sha(validation)==VALIDATION_HASHES[name]
    held=pd.read_parquet(validation)
    train=pd.read_parquet(data/spec["train"])
    with TemporaryDirectory() as folder:
        inputs=Path(folder)/"features.csv"
        held[spec["features"]].to_csv(inputs,index=False)
        out=run/"public-predictions.csv"
        subprocess.run([sys.executable,"-B",str(ROOT/"Agent/agent.py"),"predict",
                        "--model",str(run/"model.joblib"),"--input",str(inputs),"--output",str(out)],check=True)
    labels=np.unique(train[spec["target"]].astype(str))
    y=held[spec["target"]].astype(str).to_numpy()
    pred=pd.read_csv(out,keep_default_na=False)
    prob=pred[["p_"+c for c in labels]].to_numpy()
    assert np.array_equal(pred.prediction,labels[prob.argmax(1)]) and np.allclose(prob.sum(1),1)
    def normalize(s):
        return " ".join(unicodedata.normalize("NFKC",s).casefold().split())
    hits=pred.prediction.to_numpy()==y
    summary=dict(rows=len(y),correct=int(hits.sum()),accuracy=float(hits.mean()),classes=len(labels),
                 macro_f1=float(f1_score(y,pred.prediction,average="macro")),
                 train_text_overlap=int(held.text.map(normalize).isin(set(train.text.map(normalize))).sum()),
                 validation_sha256=sha(validation),predictions_sha256=sha(out))
    with (run/"public-metrics.json").open("x",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name",choices=VALIDATION_HASHES)
    for name in ("run","data","validation"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    score(args.name,args.run,args.data,args.validation)
