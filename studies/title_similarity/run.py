"""Screen two near-wording models on the frozen training groups of the title task."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"experiments/selection"))
from models import features, fit_one, groups, joblib, np, pd, probabilities, sha
from task_data import data_root, load_labeled, task_files
from candidates import near_cases, SliceCandidate

BASE = "pair_hybrid"
NAMES = ("domain_features", "unit_lexical_edit")
TRAIN = "final_project_tasks/headlines_ctr_clean/train.csv"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)
        out.write("\n")


def scores(baseline, candidate, truth):
    a = baseline.argmax(axis=1) == truth
    b = candidate.argmax(axis=1) == truth
    return dict(rows=len(a), baseline=int(a.sum()), correct=int(b.sum()),
                corrected=int((~a & b).sum()), lost=int((a & ~b).sum()),
                net=int(b.sum())-int(a.sum()))


def group_interval(baseline, candidate, actual, group):
    # Resample complete connected-title groups within the reported slice.
    delta = (candidate.argmax(1)==actual).astype(int) - (baseline.argmax(1)==actual)
    _, membership = np.unique(group, return_inverse=True)
    total = np.bincount(membership)
    net = np.bincount(membership, weights=delta)
    rng = np.random.default_rng(113)
    estimates = []
    for _ in range(50):
        draw = rng.integers(len(total), size=(100,len(total)))
        estimates.extend(net[draw].sum(1) / total[draw].sum(1))
    return list(np.quantile(estimates,[.025,.975]))


def train_and_replay(x, y, fit, held, names, path):
    baseline = fit_one(BASE, "pair", x.iloc[fit], y[fit], np.unique(y[fit]))
    models = {BASE:baseline, **{n:SliceCandidate(n, baseline).fit(x.iloc[fit], y[fit]) for n in names}}
    joblib.dump(models, path, compress=2)
    rows = x.iloc[held]
    probabilities_by_name = {}
    for n, restored in joblib.load(path).items():
        p = probabilities(restored, rows, baseline.classes_)
        assert np.array_equal(p, models[n].predict_proba(rows))
        reversed_scores = restored.predict_proba(rows.iloc[:,::-1].set_axis(rows.columns,axis=1))
        np.testing.assert_allclose(p, reversed_scores[:,::-1], rtol=0, atol=1e-10)
        probabilities_by_name[n] = p
    outside = ~near_cases(rows)
    for n in names:
        assert np.array_equal(probabilities_by_name[n][outside], probabilities_by_name[BASE][outside])
    return probabilities_by_name


def guarded_worker(script, data, out):
    out.mkdir(parents=True,exist_ok=False)
    audit=out.resolve()/"access.jsonl"
    env=dict(os.environ,AIAA_STUDY_ZIP=str(data.resolve()),AIAA_STUDY_MEMBERS=json.dumps([TRAIN]),
             AIAA_STUDY_ACCESS=str(audit),PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT/"checks"),env.get("PYTHONPATH","")])
    subprocess.run([sys.executable,"-B",script,*sys.argv[1:],"--worker"],env=env,check=True)
    reads=[json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert {r["file"] for r in reads}=={TRAIN} and all(r["allowed"] for r in reads)


def run(data, frozen, out):
    spec = read(frozen/"task.json")
    assert spec["train"] == TRAIN and spec["pairwise"] and not spec.get("train_labels")
    prior = read(frozen/"protocol.json")
    assert sha(frozen/"partition.csv") == prior["roles_sha256"]
    assert sha(frozen/"ranking.npz") == read(frozen/"challenger.json")["ranking_sha256"]
    floor, min_wins = prior["minimum_gain"], prior["min_winning_folds"]
    assert (floor,min_wins) == (.005,3)
    write(out/"protocol.json",dict(candidates=list(NAMES),baseline=BASE,similarity="Agent similar_wording, Jaccard >= 0.65",
          source_hashes={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name("candidates.py"))},
          roles_sha256=prior["roles_sha256"],minimum_gain=floor,min_winning_folds=min_wins,
          selection="best eligible raw accuracy, alphabetical tie break; confirm only that candidate",
          confirmation="at least the same 0.005 absolute accuracy gain", official_splits_opened=[]))
    with data_root(data) as root:
        tr = task_files(root,spec)["train"]
        frame = load_labeled(tr,spec)
        x = features(frame,spec,tr[0].parent)
    y = frame[spec["target"]].to_numpy(str)
    part = pd.read_csv(frozen/"partition.csv")
    roles = part.ranking_fold.to_numpy()
    group = groups(x,"pair")
    assert np.array_equal(part.row,np.arange(1,len(x)+1)) and np.array_equal(group,part.group)
    assert part.groupby("group").ranking_fold.nunique().eq(1).all()
    rank = np.flatnonzero(roles>=0)
    folds = roles[rank]
    classes = np.unique(y[rank])
    truth = np.searchsorted(classes,y[rank])
    with np.load(frozen/"ranking.npz",allow_pickle=False) as old:
        assert np.array_equal(old["row"],rank+1) and np.array_equal(old["actual_index"],truth)
        assert np.array_equal(old["fold"],folds) and np.array_equal(old["classes"],classes)
        recorded = old[BASE].copy()
    all_p = {n:np.empty_like(recorded) for n in (BASE,*NAMES)}
    for fold in range(4):
        fit, held = rank[folds!=fold], rank[folds==fold]
        fp = out/f"fold-{fold}.joblib"
        result = train_and_replay(x,y,fit,held,NAMES,fp)
        for n,p in result.items():
            all_p[n][folds==fold] = p
        print(f"completed grouped fold {fold+1}/4",flush=True)
    np.testing.assert_allclose(all_p[BASE], recorded, rtol=0, atol=1e-10)
    assert np.array_equal(all_p[BASE].argmax(1),recorded.argmax(1))
    near = near_cases(x.iloc[rank])
    np.savez_compressed(out/"ranking.npz",row=rank+1,fold=folds,group=group[rank],
                        near=near,actual_index=truth,classes=classes,**all_p)
    summary = {}
    eligible = []
    for n in NAMES:
        p = all_p[n]
        s = scores(recorded,p,truth)
        by_fold = [scores(recorded[folds==f],p[folds==f],truth[folds==f]) for f in range(4)]
        slice_stats = scores(recorded[near],p[near],truth[near])
        slice_stats["group_bootstrap_95_interval"] = group_interval(recorded[near],p[near],truth[near],group[rank][near])
        summary[n] = dict(all_rows=s,near=slice_stats,other=scores(recorded[~near],p[~near],truth[~near]),
                          folds=by_fold)
        if s["net"]/len(rank)>=floor and sum(f["net"]>0 for f in by_fold)>=min_wins:
            eligible.append(n)
    best = min(eligible,key=lambda n:(-summary[n]["all_rows"]["correct"],n)) if eligible else None
    write(out/"ranking-decision.json",dict(results=summary,nominee=best,ranking_sha256=sha(out/"ranking.npz")))
    confirmation = None
    selected = BASE
    if best:
        hold = np.flatnonzero(roles==-1)
        check = train_and_replay(x,y,rank,hold,[best],out/"confirmation.joblib")
        actual = np.searchsorted(classes,y[hold])
        confirmation = scores(check[BASE],check[best],actual)
        np.savez_compressed(out/"confirmation.npz",row=hold+1,actual_index=actual,**check)
        if confirmation["net"]/len(hold)>=floor:
            selected = best
    result = dict(results=summary,selected=selected,nominee=best,confirmation=confirmation,
                  ranking_rows=len(rank),near_rows=int(near.sum()),reserved_confirmation_rows=int((roles==-1).sum()),
                  baseline_class_predictions_identical=True,
                  max_baseline_probability_difference=float(np.max(np.abs(all_p[BASE]-recorded))),
                  fold_model_hashes=[sha(out/f"fold-{f}.joblib") for f in range(4)],paid_calls=0)
    write(out/"result.json",result)
    print(json.dumps({n:v["all_rows"] for n,v in summary.items()}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ("data","frozen","out"):
        ap.add_argument("--"+k,type=Path,required=True)
    ap.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    a=ap.parse_args()
    if a.worker:
        run(a.data.resolve(),a.frozen.resolve(),a.out.resolve())
    else:
        guarded_worker(__file__, a.data, a.out)
