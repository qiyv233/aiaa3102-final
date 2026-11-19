"""Prospective controls on Banknote, YouTube and a four-class CIFAR-10 subset."""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tarfile
import unicodedata
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split


def dump(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def cifar(raw, folder):
    archive = raw / "cifar.tar.gz"
    if (raw/"cifar-mirror.tar.gz").is_file():
        archive = raw/"cifar-mirror.tar.gz"
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce":
            raise ValueError("CIFAR-10 official Python archive checksum mismatch")
    rng = np.random.default_rng(1119)
    (folder/"images").mkdir()
    splits=[]
    with tarfile.open(archive,"r:gz") as t:
        for split,count in [("train",180),("test",120)]:
            names=[f"data_batch_{i}" for i in range(1,6)] if split=="train" else ["test_batch"]
            batches=[pickle.load(t.extractfile("cifar-10-batches-py/"+n),encoding="bytes") for n in names]
            y=np.concatenate([b[b"labels"] for b in batches])
            x=np.concatenate([b[b"data"] for b in batches]).reshape(-1,3,32,32)
            ids=np.concatenate([rng.choice(np.flatnonzero(y==c),count,replace=False) for c in range(4)])
            rows=[]
            for i in rng.permutation(ids):
                fid=f"{split}-{i}"
                name=f"images/{fid}.png"
                Image.fromarray(x[i].transpose(1,2,0)).save(folder/name)
                rows.append((fid,name,str(y[i])))
            splits.append(pd.DataFrame(rows,columns=["id","image","label"]))
    return splits


def prepare(name, raw, folder):
    folder.mkdir(parents=True)
    if name=="bank":
        with ZipFile(raw/"bank.zip") as z:
            df=pd.read_csv(z.open("data_banknote_authentication.txt"),names=["variance","skew","curtosis","entropy","label"])
        df.insert(0,"id",["b"+str(i) for i in df.index])
        train,test=train_test_split(df,test_size=.25,stratify=df.label,random_state=1119)
    elif name=="youtube":
        with ZipFile(raw/"youtube.zip") as z:
            sources=sorted(n for n in z.namelist() if n.endswith(".csv") and "/" not in n)
            df=pd.concat([pd.read_csv(z.open(n),encoding="utf-8-sig").assign(source=n) for n in sources],ignore_index=True)
        df=df.rename(columns={"CONTENT":"text","CLASS":"label"})
        df.insert(0,"id",["y"+str(i) for i in df.index])
        df["key"]=df.text.map(lambda t:" ".join(unicodedata.normalize("NFKC",t).casefold().split()))
        df=df.loc[df.key.ne("") & df.groupby("key").source.transform("nunique").eq(1)
                  & df.groupby("key").label.transform("nunique").eq(1)].drop_duplicates("key")
        train=df.loc[df.source!=sources[-1],["id","text","label"]]
        test=df.loc[df.source==sources[-1],["id","text","label"]]
    else:
        train,test=cifar(raw,folder)
    train.to_csv(folder/"train.csv",index=False)
    test.drop(columns="label").to_csv(folder/"test.csv",index=False)
    test[["id","label"]].to_csv(folder/"evaluation_labels.csv",index=False)
    spec=dict(train="train.csv",test="test.csv",target="label",id="id",output="probabilities",
              metric="log_loss" if name=="cifar" else "macro_f1",
              modality={"bank":"table","youtube":"text","cifar":"image"}[name])
    for seed in (17,43,71):
        dump(folder/f"task-{seed}.json",dict(spec,seed=seed))
    return spec


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("name",choices=["bank","youtube","cifar"])
    parser.add_argument("raw",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    folder=args.out/args.name
    spec=prepare(args.name,args.raw,folder)
    agent=Path(__file__).with_name("Agent")/"agent.py"
    env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2",PYTHONUTF8="1")
    rows=[]
    for seed in (17,43,71):
        for strategy in ["fixed","strong_fixed","rule"]+(["transfer"] if args.name=="cifar" else []):
            out=folder/f"{seed}-{strategy}"
            transfer=strategy=="transfer"
            subprocess.run([sys.executable,str(agent),"run","--task",str(folder/f"task-{seed}.json"),
                  "--out",str(out),"--pool","transfer" if transfer else "adaptive",
                  "--strategy","fixed" if transfer else strategy,"--max-trials","2" if transfer else "3",
                  "--seconds","180"],check=True,env=env)
            log=[json.loads(s) for s in (out/"run.jsonl").read_text(encoding="utf-8").splitlines()]
            trials=[json.loads(s) for s in (out/"experiments.jsonl").read_text(encoding="utf-8").splitlines()]
            rows.append(dict(seed=seed,strategy=strategy,selected=log[-1]["selected"],split=log[0]["validation_sha256"],
                   status=log[-1]["status"],actions=[s["candidate"] for s in trials],
                   failures=[s for s in trials if s["status"]!="ok"],seconds=log[-1]["seconds"]))
        assert len({r["split"] for r in rows if r["seed"]==seed})==1
    truth=pd.read_csv(folder/"evaluation_labels.csv",dtype=str)
    for row in rows:
        pred=pd.read_csv(folder/f"{row['seed']}-{row['strategy']}"/"predictions.csv",dtype={"id":str})
        assert pred.id.equals(truth.id)
        probs=pred.drop(columns="id")
        assert np.isfinite(probs.values).all() and np.allclose(probs.sum(axis=1),1) and (probs>=0).all().all()
        val=(log_loss(truth.label,probs,labels=list(probs.columns)) if spec["metric"]=="log_loss"
             else f1_score(truth.label,probs.idxmax(axis=1),average="macro"))
        row.update(metric=spec["metric"],score=float(val),rows=len(truth))
    dump(folder/"summary.json",rows)
    print(pd.DataFrame(rows).groupby("strategy").score.median(),flush=True)


if __name__=="__main__":
    main()

