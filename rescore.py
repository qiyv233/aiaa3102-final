"""Independently rescore the previous 18 held-out prediction files and audit row overlap."""
from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd
from sklearn.metrics import f1_score,log_loss


def main():
    p=argparse.ArgumentParser()
    p.add_argument("previous",type=Path)
    p.add_argument("new_tasks",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    old=[]
    for suite in ("five","car"):
        folder=a.previous/"results"/suite
        for row in json.loads((folder/"summary.json").read_text()):
            task=row["task"]
            directory=folder/task/row["pool"]
            y=pd.read_csv(a.previous/"comparison"/(task+"-labels.csv"),dtype=str)
            pred=pd.read_csv(directory/"predictions.csv",dtype={"id":str})
            assert y.id.equals(pred.id)
            assert (directory/"predictions.csv").read_bytes()==(directory/"replay.csv").read_bytes()
            if row["metric"]=="log_loss":
                prob=pred.drop(columns="id")
                v=log_loss(y.label,prob.div(prob.sum(axis=1),axis=0),labels=prob.columns.tolist())
            else:
                guess=pred.label.astype(str) if "label" in pred else pred.drop(columns="id").idxmax(axis=1)
                v=f1_score(y.label,guess,average="macro")
            assert abs(row["score"]-v)<=0.00000051
            old.append(dict(task=task,pool=row["pool"],recalculated=float(v)))
    overlap={}
    for name in ("bank","youtube","cifar"):
        folder=a.new_tasks/name
        train=pd.read_csv(folder/"train.csv").drop(columns=["label","id"])
        test=pd.read_csv(folder/"test.csv").drop(columns="id")
        if name=="cifar":
            def keys(series):
                return series.map(lambda f:hashlib.sha256((folder/f).read_bytes()).hexdigest())
            matched=keys(test.image).isin(set(keys(train.image)))
        else:
            set_train=set(train.itertuples(index=False,name=None))
            matched=pd.Series([t in set_train for t in test.itertuples(index=False,name=None)])
        info=dict(train=len(train),test=len(test),exact_matches=int(matched.sum()))
        if matched.any() and name=="bank":
            clean={}
            y=pd.read_csv(folder/"evaluation_labels.csv",dtype=str)
            for r in json.loads((folder/"summary.json").read_text()):
                pr=pd.read_csv(folder/f"{r['seed']}-{r['strategy']}"/"predictions.csv").drop(columns="id").idxmax(axis=1)
                clean[f"{r['seed']}-{r['strategy']}"]=f1_score(y.label[~matched],pr[~matched],average="macro")
            info["scores_without_matches"]=clean
        overlap[name]=info
    a.output.write_text(json.dumps(dict(prior=old,overlap=overlap),indent=2),encoding="utf-8")
    print(len(old),overlap)


if __name__=="__main__":
    main()
