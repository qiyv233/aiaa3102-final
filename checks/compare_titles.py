"""Report the locked models' paired results on training folds and the training confirmation split."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/selection"))
from models import features, fit_one, np, pd, probabilities, sha
from task_data import data_file, data_root, load_labeled


def compare(old, other, y):
    a=old.argmax(axis=1)==y
    b=other.argmax(axis=1)==y
    return {"rows":len(y),"hybrid_correct":int(a.sum()),"tfidf_correct":int(b.sum()),
            "tfidf_corrected":int((b & ~a).sum()),"tfidf_lost":int((a & ~b).sum())}


def main(run, source, out):
    contract=json.loads((run/"task.json").read_text(encoding="utf-8"))
    result=json.loads((run/"result.json").read_text(encoding="utf-8"))
    assert result["contract_sha256"]==sha(run/"task.json") and result["stage"]=="full_refit"
    assert contract["train"]=="final_project_tasks/headlines_ctr_clean/train.csv"
    with data_root(source) as root:
        key=data_file(root,contract["train"])
        data=load_labeled((key,None),contract)
        x=features(data,contract,key.parent)
    y=data[contract["target"]].to_numpy(str)
    roles=pd.read_csv(run/"partition.csv").ranking_fold.to_numpy()
    fit=np.flatnonzero(roles>=0)
    hold=np.flatnonzero(roles==-1)
    classes=np.unique(y)
    with np.load(run/"ranking.npz",allow_pickle=False) as pack:
        assert np.array_equal(pack["row"],fit+1) and np.array_equal(pack["actual_index"],np.searchsorted(classes,y[fit]))
        ranking=compare(pack["pair_hybrid"],pack["pair_tfidf"],pack["actual_index"])
        by_fold=[]
        for fold in range(4):
            mask=pack["fold"]==fold
            by_fold.append(compare(pack["pair_hybrid"][mask],pack["pair_tfidf"][mask],pack["actual_index"][mask]))
    forecasts=[]
    for name in ("pair_hybrid","pair_tfidf"):
        estimator=fit_one(name,"pair",x.iloc[fit],y[fit],classes)
        forecasts.append(probabilities(estimator,x.iloc[hold],classes))
    summary=dict(ranking=ranking,by_fold=by_fold,training_confirmation=compare(*forecasts,np.searchsorted(classes,y[hold])))
    with out.open("x",encoding="utf-8") as f:
        json.dump(summary,f,indent=2)
    print(json.dumps(summary))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("run","data","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    main(args.run,args.data,args.out)
