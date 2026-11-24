"""Repeat the positive slice result on new folds of the original ranking rows only."""
import argparse
import json
from pathlib import Path

from run import (BASE, TRAIN, data_root, features, groups, guarded_worker, load_labeled,
                 near_cases, np, pd, read, scores, sha, task_files, train_and_replay, write)
from models import fold_ids


def repeat(data, frozen, first, out):
    original = read(first/"ranking-decision.json")
    assert original["nominee"] is None and read(first/"result.json")["selected"] == BASE
    positive = [n for n, s in original["results"].items() if s["all_rows"]["net"] > 0]
    if len(positive)!=1:
        raise ValueError("This diagnostic requires a single positive but ineligible candidate")
    name = positive[0]
    write(out/"protocol.json", dict(candidate=name, seed=7919, source_hash=sha(Path(__file__)),
          original_lock=sha(first/"ranking-decision.json"), role="diagnostic only; no promotion or confirmation"))
    spec=read(frozen/"task.json")
    assert spec["train"]==TRAIN and not spec.get("train_labels")
    assert sha(frozen/"partition.csv")==read(first/"protocol.json")["roles_sha256"]
    roles=pd.read_csv(frozen/"partition.csv")
    rank=np.flatnonzero(roles.ranking_fold.to_numpy()>=0)
    with data_root(data) as root:
        tr=task_files(root,spec)["train"]
        frame=load_labeled(tr,spec).iloc[rank]
        x=features(frame,spec,tr[0].parent)
    y=frame[spec["target"]].to_numpy(str)
    g=roles.group.to_numpy()[rank]
    assert np.array_equal(pd.factorize(g,sort=True)[0],groups(x,"pair"))
    folds=fold_ids(y,g,7919)
    actual=np.searchsorted(np.unique(y),y)
    preds={n:np.empty((len(y),2)) for n in (BASE,name)}
    by_fold=[]
    for fold in range(4):
        fit,held=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not set(g[fit]) & set(g[held])
        r=train_and_replay(x,y,fit,held,[name],out/f"fold-{fold}.joblib")
        for n,p in r.items():
            preds[n][held]=p
        by_fold.append(scores(r[BASE],r[name],actual[held]))
        print(f"repeat fold {fold+1}/4",flush=True)
    near=near_cases(x)
    np.savez_compressed(out/"ranking.npz",row=rank+1,group=g,near=near,fold=folds,actual_index=actual,**preds)
    result=dict(seed=7919, candidate=name, all_rows=scores(preds[BASE],preds[name],actual),
                near=scores(preds[BASE][near],preds[name][near],actual[near]),folds=by_fold,selected=BASE)
    write(out/"result.json",result)
    print(json.dumps(result["all_rows"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("data","frozen","first","out"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:
        repeat(a.data.resolve(),a.frozen,a.first,a.out)
    else:
        guarded_worker(__file__, a.data, a.out)
