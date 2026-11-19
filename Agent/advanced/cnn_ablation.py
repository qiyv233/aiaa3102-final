"""Follow-up analysis on training-side validation only; never opens test.csv."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold,train_test_split

from engine import score
from agent import inputs,load_csv,read_json
from learners import SmallCNN,pixels


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("task",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    spec=read_json(args.task)
    file=args.task.parent/spec["train"]
    frame=load_csv(file)
    X=pixels(inputs(frame,dict(mode="image",features=["file"],numeric=[]),file.parent))
    y=frame[spec["target"]].to_numpy()
    tr,va=next(StratifiedKFold(3,shuffle=True,random_state=43).split(X,y))
    short,_=train_test_split(tr,train_size=.5,stratify=y[tr],random_state=43)
    rows=[]
    for seed in [17,43,71]:
        for idx in [short,tr]:
            for epochs in [4,12]:
                model=SmallCNN(seed=seed,epochs=epochs).fit(X[idx],y[idx])
                val=score(spec,y[va],model.predict_proba(X[va]),model.classes_)
                rows.append(dict(seed=seed,train_n=len(idx),valid_n=len(va),epochs=epochs,
                                 device=model.device_used_,log_loss=val))
    args.output.write_text(json.dumps(rows,indent=2),encoding="utf-8")
    for n in (len(short),len(tr)):
        for e in (4,12):
            print(n,e,np.mean([r["log_loss"] for r in rows if r["train_n"]==n and r["epochs"]==e]))
