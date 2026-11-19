"""Paired post-design check on the same held-out rows, with a new internal seed."""
import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score, log_loss


def main():
    base=Path(sys.argv[1])
    agent=Path(__file__).parent/"Agent"/"agent.py"
    env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
    rows=[]
    for name in sys.argv[2:]:
        folder=base/name
        spec=json.loads((folder/"task-17.json").read_text())
        spec["seed"]=113
        task=folder/"task-113.json"
        task.write_text(json.dumps(spec),encoding="utf-8")
        outputs=[]
        for pool,strategy in [("adaptive","strong_fixed"),("reviewed","rule")]:
            out=folder/f"113-{pool}-{strategy}"
            subprocess.run([sys.executable,str(agent),"run","--task",str(task),"--pool",pool,"--strategy",strategy,
                            "--max-trials","3","--seconds","180","--out",str(out)],check=True,env=env)
            subprocess.run([sys.executable,str(agent),"predict","--model",str(out/"model.joblib"),"--input",
                            str(folder/"test.csv"),"--output",str(out/"replay.csv")],check=True,env=env)
            assert (out/"predictions.csv").read_bytes()==(out/"replay.csv").read_bytes()
            outputs.append((pool,strategy,out))
        labels=pd.read_csv(folder/"evaluation_labels.csv",dtype=str)
        for pool,strategy,out in outputs:
            pred=pd.read_csv(out/"predictions.csv",dtype={"id":str})
            assert labels.id.equals(pred.id)
            p=pred.drop(columns="id")
            value=(f1_score(labels.label,p.idxmax(axis=1),average="macro") if spec["metric"]=="macro_f1"
                   else log_loss(labels.label,p,labels=list(p.columns)))
            run=[json.loads(s) for s in (out/"run.jsonl").read_text(encoding="utf-8").splitlines()]
            trials=[json.loads(s) for s in (out/"experiments.jsonl").read_text(encoding="utf-8").splitlines()]
            rows.append(dict(task=name,pool=pool,seed=113,strategy=strategy,score=value,
                 selected=run[-1]["selected"],split=run[0]["validation_sha256"],seconds=run[-1]["seconds"],
                 actions=[s["candidate"] for s in trials],failed=sum(s["status"]!="ok" for s in trials)))
        assert rows[-1]["split"]==rows[-2]["split"]
        (folder/"confirmation.json").write_text(json.dumps(rows[-2:],indent=2),encoding="utf-8")
        print(rows[-2:],flush=True)


if __name__=="__main__":
    main()
