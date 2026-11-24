"""Score an already locked checkpoint on the validation split."""
import argparse
from pathlib import Path

from models import features, infer, joblib, np, pd, sha
from objective import Objective, deployed_indices
from select_agent import read, write_json
from task_data import data_root, load_labeled, task_files


def evaluate(task, source, out, split="validation"):
    if split not in {"validation", "test"}:
        raise ValueError("Choose a validation or final test split")
    result=read(out/"result.json")
    nomination=result["challenger_lock_sha256"]
    if (sha(out/"model.joblib")!=result["model_sha256"] or sha(task)!=result["contract_sha256"]
            or (nomination is None and result.get("run_status")!="timeout_checkpoint")
            or (nomination is not None and sha(out/"challenger.json")!=nomination)):
        raise ValueError("The model, selection lock or task manifest differs from the frozen run")
    bundle=joblib.load(out/"model.joblib")
    if bundle["names"]!=result["selected"]:
        raise ValueError("Checkpoint names differ from the delivered result")
    spec=read(task)
    with data_root(source.resolve()) as root:
        paths=task_files(root,spec)[split]
        frame=load_labeled(paths,spec)
        x=features(frame,spec,paths[0].parent)
        p=infer(bundle,x)
    classes=bundle["classes"]
    actual=frame[spec["target"]].to_numpy(str)
    if not len(actual) or not np.isin(actual,classes).all():
        raise ValueError(f"{split} labels do not match training classes")
    prediction=deployed_indices(bundle,p)
    labels=classes[prediction]
    pred=pd.DataFrame(p,columns=["p_"+c for c in classes])
    pred.insert(0,"prediction",labels)
    metrics={f"{split}_metric":spec["metric"],
             f"{split}_score":Objective(spec,classes).score(np.searchsorted(classes,actual),prediction),
             f"{split}_correct":int((labels==actual).sum()),f"{split}_rows":len(actual),
             f"{split}_accuracy":float((labels==actual).mean())}
    target=out/f"{split}-predictions.csv"
    tmp=target.with_suffix(".new")
    pred.to_csv(tmp,index=False)
    tmp.replace(target)
    result.update(metrics)
    write_json(out/"result.json",result)
    print(f"{split} {metrics[f'{split}_correct']}/{len(actual)}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--task",type=Path,help="Manifest, defaults to the one generated in the run")
    parser.add_argument("--split",choices=["validation","test"],default="validation",
                        help="Use test only for final scoring of the locked checkpoint")
    args=parser.parse_args()
    evaluate(args.task or args.out/"task.json",args.data,args.out,args.split)
