"""Two-stage search with out-of-fold ensemble selection and separate audit arms."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

if __package__:
    from .learners import POOLS, candidate
else:
    from learners import POOLS, candidate
from agent import load_csv, inputs, read_json, evaluate
from features import infer_modality
from planning import record, ask
from budget import task_key
from submission import write_submission


def normalized(probs):
    p=np.asarray(probs,dtype=np.float64)
    total=p.sum(axis=1,keepdims=True)
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(total,1,atol=1e-5):
        raise ValueError("Invalid probabilities")
    return p/total


def score(spec, y, probs, labels):
    return evaluate(spec["metric"],np.asarray(y),normalized(probs),labels,spec.get("positive_class"))[0]


def find_mix(spec, y, labels, names, oof):
    direction = -1 if spec["metric"]=="log_loss" else 1
    rank = sorted(names,key=lambda n: -direction*score(spec,y,oof[n],labels))
    if not rank:
        raise ValueError("No valid candidates")
    if len(rank) == 1:
        return {rank[0]:1.0},rank[0]
    a,b=rank[:2]
    options=[(a,1.0,b,0.0)]
    options.extend((a,w,b,1-w) for w in (0.75,0.5,0.25))
    def value(option):
        a,w,b,v=option
        return direction*score(spec,y,w*oof[a]+v*oof[b],labels)
    a,w,b,v=max(options,key=value)
    return {n:s for n,s in ((a,w),(b,v)) if s},rank[0]


def probabilities(package, file):
    frame=load_csv(file)
    X=inputs(frame,package["schema"],file.parent)
    scores={n:m.predict_proba(X) for n,m in package["models"].items()}
    result=sum(w*scores[n] for n,w in package["weights"].items())
    return normalized(result)


def run(task, out, audit=False, planner=None):
    out.mkdir(parents=True,exist_ok=False)
    spec=read_json(task)
    if "validation" in spec or spec.get("split","stratified")!="stratified" or spec.get("group_column") or spec.get("time_column"):
        raise ValueError("Research engine currently supports internal stratified folds only.")
    train_file=task.parent/spec["train"]
    frame=load_csv(train_file)
    cols=spec.get("features",[c for c in frame if c not in (spec["target"],spec.get("id"))])
    if not cols or spec["target"] in cols or spec.get("id") in cols:
        raise ValueError("Invalid features")
    mode=spec.get("modality") or infer_modality(frame[cols])
    num=[]
    if mode=="table":
        for c in cols:
            observed=frame.loc[frame[c].ne(""),c]
            if len(observed) and pd.to_numeric(observed,errors="coerce").notna().all():
                num.append(c)
        num=spec.get("numeric",num)
    schema=dict(mode=mode,numeric=num,features=cols)
    X=inputs(frame,schema,train_file.parent)
    y=frame[spec["target"]].to_numpy()
    classes=np.unique(y)
    seed=int(spec.get("seed",17))
    folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(X,y))
    direction=-1 if spec["metric"]=="log_loss" else 1
    pilot, val=folds[0]
    pilot,_=train_test_split(pilot,train_size=0.5,stratify=y[pilot],random_state=seed)
    pool=POOLS[mode]
    log=out/"experiments.jsonl"
    def fit(name, idx, stage):
        tick=time.monotonic()
        m=candidate(mode,name,X.iloc[idx],seed,pilot=stage=="pilot").fit(X.iloc[idx],y[idx])
        if not np.array_equal(m.classes_,classes):
            raise ValueError("Class alignment error")
        record(log,dict(stage=stage,candidate=name,rows=len(idx),seconds=time.monotonic()-tick))
        return m
    pilots={}
    recalls={}
    pilot_times={}
    for n in pool:
        started=time.monotonic()
        try:
            m=fit(n,pilot,"pilot")
            p=normalized(m.predict_proba(X.iloc[val]))
            pilots[n],recalls[n]=evaluate(spec["metric"],y[val],p,classes,spec.get("positive_class"))
            pilot_times[n]=round(time.monotonic()-started,4)
        except Exception as error:
            record(log,dict(stage="pilot",candidate=n,status="failed",error=str(error)[:300]))
    if not pilots:
        raise ValueError("All pilot fits failed")
    ranked=sorted(pilots,key=lambda n:-direction*pilots[n])
    finalists=ranked[:2]
    decision=dict(policy="pilot_rank",reason="Promote the two highest pilot ranks.")
    if planner is not None and len(ranked)>2:
        evidence=dict(mode=mode,metric=spec["metric"],maximize=direction==1,
                      pilot_train_rows=len(pilot),pilot_validation_rows=len(val),
                      class_counts={str(c):int((y==c).sum()) for c in classes},
                      pilot_scores=pilots,per_class_recall=recalls,seconds=pilot_times,
                      retained=ranked[0],note="Second slot may preserve a complementary or slower-starting family. CNN pilot uses 4 epochs, full fits 12.")
        try:
            cfg=read_json(planner)
            if not cfg.get("model","").strip() or cfg["model"].lower() in {"deepseek-flash","deepseek-v4-flash","deepseek-v4-flash-vision-exp"}:
                raise ValueError("Configure a verified dated deployment for the pre-October-17 project.")
            second,reason=ask(cfg,evidence,ranked[1:],out/"ledger.jsonl",task_key(train_file,spec))
            finalists=[ranked[0],second]
            decision=dict(policy="llm",reason=reason,model=cfg["model"])
        except Exception as error:
            decision=dict(policy="rule_fallback",reason=f"{type(error).__name__}: {error}")
    record(log,dict(stage="promotion",finalists=finalists,decision=decision,pilot_scores=pilots))
    oof={}
    def crossfit(n):
        p=np.empty((len(y),len(classes)))
        for k,(tr,va) in enumerate(folds):
            m=fit(n,tr,f"fold_{k}")
            p[va]=m.predict_proba(X.iloc[va])
        oof[n]=p
    def try_folds(n):
        try:
            crossfit(n)
        except Exception as error:
            record(log,dict(stage="crossfit",candidate=n,status="failed",error=str(error)[:300]))
    for n in finalists + [n for n in ranked if n not in finalists]:
        try_folds(n)
        if len(oof) >= 2:
            break
    finalists=list(oof)
    mix,single=find_mix(spec,y,classes,finalists,oof)
    policies={"selected":mix}
    if audit:
        # These fits occur AFTER selected was fixed. They never modify it.
        for n in ranked:
            if n not in oof:
                try_folds(n)
        all_mix,all_single=find_mix(spec,y,classes,list(oof),oof)
        if pool[0] in oof:
            policies["baseline"]={pool[0]:1.0}
        policies.update(promoted_single={single:1.0},
                        exhaustive_single={all_single:1.0},exhaustive_mix=all_mix)
    fitted={n:fit(n,np.arange(len(y)),"refit") for n in pool if any(n in w for w in policies.values())}
    for policy,weights in policies.items():
        package=dict(spec=spec,schema=schema,classes=classes,models={n:fitted[n] for n in weights},weights=weights)
        if policy=="selected":
            joblib.dump(package,out/"model.joblib")
        if "test" in spec:
            file=task.parent/spec["test"]
            pred=probabilities(package,file)
            np.save(out/(policy+".npy"),pred)
            if policy=="selected":
                template=task.parent/spec["submission_template"] if spec.get("submission_template") else None
                export(package,file,pred,spec,out/"predictions.csv",template)
    result=dict(mode=mode,classes=classes.tolist(),seed=seed,metric=spec["metric"],pilot=pilots,decision=decision,
                finalists=finalists,policies=policies,oof={n:score(spec,y,p,classes) for n,p in oof.items()})
    (out/"selection.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(out/"oof.npz",labels=y.astype(str),**oof)
    return result


def export(package,file,probs,spec,dest,template=None):
    import pandas as pd
    result=pd.DataFrame(index=range(len(probs)))
    if spec.get("id"):
        result[spec["id"]]=load_csv(file)[spec["id"]]
    labels=package["classes"]
    form=spec.get("output","label")
    outputs=([spec.get("probability_columns",{}).get(c,c) for c in labels] if form=="probabilities"
             else [spec.get("prediction_column",spec["target"])])
    if len(outputs)!=len(set(outputs)) or spec.get("id") in outputs:
        raise ValueError("Duplicate submission column")
    if form=="label":
        result[spec.get("prediction_column",spec["target"])]=labels[probs.argmax(axis=1)]
    elif form=="positive_probability":
        result[spec["prediction_column"]]=probs[:,labels.tolist().index(str(spec["positive_class"]))]
    elif form=="probabilities":
        for i,c in enumerate(labels):
            result[spec.get("probability_columns",{}).get(c,c)]=probs[:,i]
    else:
        raise ValueError("Unknown output format")
    write_submission(result,dest,template,spec.get("id"))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--task",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--audit",action="store_true",help="Run additional comparison arms after selection")
    p.add_argument("--planner",type=Path,help="Optional verified historical DeepSeek endpoint configuration")
    args=p.parse_args()
    run(args.task,args.out,args.audit,args.planner)
