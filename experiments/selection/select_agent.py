"""Timed, group-aware offline selection for classification tasks scored by accuracy."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from models import (POOLS, family, fit_one, fold_ids, groups, joblib, load_xy, np, pd,
                    predict_file, probabilities, save_bundle, sha)


def write_json(path, value):
    scratch = path.with_suffix(".new")
    scratch.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),encoding="utf-8")
    scratch.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def log(path, **event):
    with path.open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+"\n")


def worker(payload, arm, out, seconds):
    start = time.monotonic()
    x,y,grp,spec = joblib.load(payload)
    classes = np.unique(y)
    kind = family(spec,y)
    anchor, priority = POOLS[kind]
    ids = fold_ids(y,grp,311)
    path = out/"model.joblib"
    events = out/"events.jsonl"
    log(events,event="start",arm=arm,seed=311,family=kind,rows=len(y),classes=classes.tolist(),
        anchor=anchor,priority=priority,remaining_seconds=round(seconds,3),llm_calls=0)
    pd.DataFrame({"position":np.arange(len(y)),"group":grp,"inner_fold":ids}).to_csv(out/"inner-split.csv",index=False)

    def attempt(name, fit, hold, stage):
        begin = time.monotonic()
        try:
            m = fit_one(name,kind,x.iloc[fit],y[fit],classes)
            prob = probabilities(m,x.iloc[hold],classes)
            ok = classes[prob.argmax(1)] == y[hold]
            elapsed = max(.0001, time.monotonic()-begin)
            log(events,event=stage,candidate=name,train_rows=len(fit),hold_rows=len(hold),
                correct=int(ok.sum()),seconds=elapsed)
            return m,prob,ok,elapsed
        except Exception as exc:
            log(events,event="candidate_failed",candidate=name,stage=stage,error=repr(exc))
            return None

    pilot_fit = np.flatnonzero(ids>=2)
    pilot_hold = np.flatnonzero(ids==1)
    screening = {}
    pilot_probabilities = {}
    best_pilot = -1
    for name in dict.fromkeys([anchor,*priority]):
        trial = attempt(name,pilot_fit,pilot_hold,"pilot")
        if trial is None:
            continue
        model,prob,ok,elapsed = trial
        screening[name] = (ok,elapsed)
        if arm in ("D", "E"):
            pilot_probabilities[name] = prob
        if int(ok.sum()) > best_pilot:
            best_pilot = int(ok.sum())
            save_bundle(path,spec,classes,{name:model},"pilot_fallback")
    if not screening:
        raise RuntimeError("No successful pilot")
    base = screening.get(anchor,next(iter(screening.values())))[0]
    scores = {n:dict(corrected=int((v & ~base).sum()),introduced=int((~v & base).sum()),
                     net=int(v.sum()-base.sum()),seconds=t) for n,(v,t) in screening.items()}
    ordered = [n for n in priority if n in scores]
    if arm in ("C","D","E"):
        ordered.sort(key=lambda n:(-scores[n]["net"],scores[n]["seconds"],priority.index(n)))
    fit = np.flatnonzero(ids!=0)
    hold = np.flatnonzero(ids==0)
    # Approximate two shortlisted fits plus a possible final refit for each.
    remaining = seconds-(time.monotonic()-start)
    reserved = 0.
    shortlist = []
    partner_scores = {}
    def affordable(name):
        return scores[name]["seconds"]*(len(fit)+len(y))/len(pilot_fit)
    for name in ordered:
        estimate = scores[name]["seconds"]*(len(fit)+len(y))/len(pilot_fit)
        if estimate+reserved <= remaining:
            shortlist.append(name)
            reserved += estimate
            if arm=="E":
                break
        if len(shortlist)==2:
            break
    if arm=="E" and shortlist:
        first=shortlist[0]
        for n in ordered:
            if n==first or affordable(n)+reserved>remaining:
                continue
            combined=(pilot_probabilities[first]+pilot_probabilities[n])*.5
            ok=classes[combined.argmax(1)]==y[pilot_hold]
            first_ok=screening[first][0]
            partner_scores[n]=dict(corrected=int((ok & ~first_ok).sum()),introduced=int((~ok & first_ok).sum()),
                                   ensemble_correct=int(ok.sum()))
        if partner_scores:
            # Keep D's order on a pair-score tie, isolating incremental value as the only change.
            second=max(partner_scores,key=lambda n:partner_scores[n]["ensemble_correct"])
            shortlist.append(second)
            reserved+=affordable(second)
    if pilot_probabilities:
        np.savez_compressed(out/"pilot-probabilities.npz",position=pilot_hold,actual=y[pilot_hold],
                            classes=classes,**pilot_probabilities)
    log(events,event="shortlist",pilot_errors=scores,selected=shortlist,remaining=remaining,
        pair_evidence=partner_scores,
        reason=("best single then best equal-weight partner" if arm=="E" else
                "pilot net corrections, then time" if arm in ("C","D") else "fixed priority"),
        estimated_reserved_seconds=reserved)
    predictions,checked = {},{}
    for name in shortlist:
        trial = attempt(name,fit,hold,"selection")
        if trial is not None:
            checked[name],predictions[name] = trial[:2]
    if not checked:
        log(events,event="fallback",reason="No successful candidate reached the selection stage")
        return
    options = { (n,):p for n,p in predictions.items() }
    if arm in ("B","D","E") and len(checked)==2:
        options[tuple(checked)] = np.mean(list(predictions.values()),axis=0)
    correct = {names:int(np.sum(classes[p.argmax(1)]==y[hold])) for names,p in options.items()}
    # Tie order is independent of ordering by pilot score; singles beat combinations on ties.
    winner = min(correct,key=lambda ns:(-correct[ns],len(ns),tuple(priority.index(n) for n in ns)))
    save_bundle(path,spec,classes,{n:checked[n] for n in winner},"selection_fallback")
    log(events,event="selection_lock",winner=winner,inner_correct={"+".join(k):v for k,v in correct.items()},
        hold_rows=len(hold),reason="highest inner accuracy; ties prefer single then fixed priority")
    final = {}
    try:
        for n in winner:
            tick = time.monotonic()
            final[n] = fit_one(n,kind,x,y,classes)
            probabilities(final[n],x.iloc[:min(16,len(x))],classes)
            log(events,event="full_fit",candidate=n,rows=len(y),seconds=time.monotonic()-tick)
        save_bundle(path,spec,classes,final,"full_refit")
    except Exception as exc:
        log(events,event="fallback",reason=repr(exc))


def supervise(payload, out, arm, seconds):
    if not np.isfinite(seconds) or seconds<=0:
        raise ValueError("Positive finite timeout required")
    out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"_worker","--payload",str(payload.resolve()),
             "--arm",arm,"--out",str(out.resolve()),"--seconds",str(seconds)]
    tick=time.monotonic()
    status="completed"
    with (out/"console.txt").open("w",encoding="utf-8") as stream:
        try:
            subprocess.run(command,check=True,timeout=seconds,stdout=stream,stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired:
            status="timeout"
        except subprocess.CalledProcessError:
            status="worker_error"
    elapsed=time.monotonic()-tick
    model_path=out/"model.joblib"
    if not model_path.exists():
        raise RuntimeError(f"No usable checkpoint after {status}: {out/'console.txt'}")
    bundle=joblib.load(model_path)
    record=dict(arm=arm,seconds=elapsed,cap_seconds=seconds,status=status,stage=bundle["stage"],
                models=bundle["names"],model_sha256=sha(model_path),llm_calls=0,cost_rmb=0)
    write_json(out/"result.json",record)
    return record


def main():
    import tempfile
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="action",required=True)
    for cmd in ("train","_worker"):
        sub=subs.add_parser(cmd)
        sub.add_argument("--arm",choices=list("ABCDE"),required=True)
        sub.add_argument("--seconds",type=float,default=600)
        sub.add_argument("--out",type=Path,required=True)
        if cmd=="train":
            sub.add_argument("--task",type=Path,required=True)
            sub.add_argument("--data",type=Path,required=True)
        else:
            sub.add_argument("--payload",type=Path,required=True)
    predict=subs.add_parser("predict")
    for key in ("model","input","output"):
        predict.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args()
    if args.action=="predict":
        predict_file(args.model,args.input,args.output)
    elif args.action=="_worker":
        worker(args.payload,args.arm,args.out,args.seconds)
    else:
        spec=read(args.task)
        x,y=load_xy(spec,args.data.resolve(),"train")
        grp=groups(x,family(spec,y),spec.get("group_field"))
        with tempfile.TemporaryDirectory() as tmp:
            payload=Path(tmp)/"train.joblib"
            joblib.dump((x,y,grp,spec),payload)
            supervise(payload,args.out,args.arm,args.seconds)


if __name__=="__main__":
    main()
