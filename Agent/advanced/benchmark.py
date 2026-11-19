"""Fixed development protocol for the advanced research engine."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from engine import run, probabilities, score
from reviews_study import read_data


def prepare(mode,root,images,reviews):
    if mode=="table":
        d=load_wine(as_frame=True)
        df=d.data.copy()
        df["label"]=d.target
        tr,te=train_test_split(df,test_size=.30,random_state=43,stratify=df.label)
    elif mode=="image":
        if hashlib.sha256(images.read_bytes()).hexdigest()!="6fb091394714cddda5751d4e1c2781ab094e7cf15de07917fb40e581f19efc75":
            raise ValueError("Unexpected Semeion archive")
        with ZipFile(images) as z:
            raw=np.loadtxt(z.open("semeion.data"))
        assert raw.shape==(1593,266) and np.all(raw[:,256:].sum(axis=1)==1)
        files=[]
        (root/"images").mkdir()
        for i,img in enumerate(raw[:,:256]):
            name=f"images/{i}.png"
            Image.fromarray((img.reshape(16,16)*255).astype("uint8")).save(root/name)
            files.append(name)
        df=pd.DataFrame(dict(file=files,label=raw[:,256:].argmax(axis=1)))
        cut=int(len(df)*.6)
        tr,te=df.iloc[:cut],df.iloc[cut:]
    else:
        df=read_data(reviews)
        tr=df.loc[df.source!="yelp",["text","label"]]
        te=df.loc[df.source=="yelp",["text","label"]]
    tr=tr.copy()
    te=te.copy()
    tr.insert(0,"id",[f"tr{i}" for i in range(len(tr))])
    te.insert(0,"id",[f"te{i}" for i in range(len(te))])
    tr.to_csv(root/"train.csv",index=False)
    te.drop(columns="label").to_csv(root/"test.csv",index=False)
    spec=dict(train="train.csv",test="test.csv",id="id",target="label",modality=mode,
              metric="log_loss" if mode=="image" else "macro_f1",output="probabilities",seed=43)
    (root/"task.json").write_text(json.dumps(spec),encoding="utf8")
    return te


def benchmark(root,images,reviews,modes=("table","text","image")):
    if "text" in modes and not reviews.is_file():
        raise FileNotFoundError(f"Missing reviews archive: {reviews}")
    if "image" in modes and not images.is_file():
        raise FileNotFoundError(f"Missing Semeion archive: {images}")
    root.mkdir(parents=True,exist_ok=False)
    scoring=[]
    for mode in modes:
        task=root/mode
        task.mkdir()
        y=prepare(mode,task,images,reviews)
        start=time.monotonic()
        result=run(task/"task.json",task/"run",audit=True)
        scoring.append((mode,task,y,result,time.monotonic()-start))
        print(mode,result["finalists"],result["policies"]["selected"],flush=True)
    report=[]
    for mode,task,te,r,sec in scoring:
        labels=np.array(r["classes"])
        folder=task/"run"
        bundle=joblib.load(folder/"model.joblib")
        assert np.allclose(probabilities(bundle,task/"test.csv"),np.load(folder/"selected.npy"),atol=1e-7)
        export=pd.read_csv(folder/"predictions.csv",dtype={"id":str})
        assert export.id.tolist()==te.id.tolist() and np.allclose(export[labels].values,np.load(folder/"selected.npy"))
        vals={name:score(r,te.label.astype(str),np.load(folder/(name+".npy")),labels) for name in r["policies"]}
        te[["id","label"]].to_csv(task/"evaluation_labels.csv",index=False)
        report.append(dict(modality=mode,test_n=len(te),metric=r["metric"],full_audit_seconds=sec,
                           pilot=r["pilot"],finalists=r["finalists"],oof=r["oof"],policies=r["policies"],
                           external=vals))
    (root/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("semeion_zip",type=Path,nargs="?",default=Path("evidence/source/semeion.zip"))
    p.add_argument("reviews_zip",type=Path,nargs="?",default=Path("evidence/source/reviews.zip"))
    p.add_argument("--only",choices=("table","text","image"),help="Run one independent development track.")
    a=p.parse_args()
    benchmark(a.out,a.semeion_zip,a.reviews_zip,(a.only,) if a.only else ("table","text","image"))
