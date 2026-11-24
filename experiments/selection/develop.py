"""Select, train and replay a classifier. Official validation is opened after selection."""
import argparse
from decimal import Decimal
import hashlib
import json
from math import sqrt
from pathlib import Path
import shutil
import subprocess
import sys
import time

from models import (POOLS, family, features, fit_one, fold_ids, groups, infer, joblib,
                    np, pd, probabilities, save_bundle, sha)
from task_data import data_root, load_labeled, task_files
from select_agent import log, read, write_json
from selection_policy import overlap_report, partition, select_candidate
from objective import Objective, METRICS
from planning import ask
import budget


def summarize(p, y, classes, slices, objective):
    predicted=classes[p.argmax(1)]
    ok=predicted==y
    breakdown={}
    for label, mask in slices.items():
        if mask.any():
            breakdown[label]={"rows":int(mask.sum()), "correct":int(ok[mask].sum()),
                              "accuracy":round(float(ok[mask].mean()),4)}
    return {"rows":len(y),"correct":int(ok.sum()),"accuracy":round(float(ok.mean()),4),
            "metric":objective.metric,"score":objective.score(np.searchsorted(classes,y),p.argmax(1)),
            "recall_by_class_index":[round(float(ok[y==c].mean()),4) if (y==c).any() else None for c in classes],
            "slices":breakdown}


def diagnostics(kind, fit, held):
    if kind=="pair":
        def similar(pair):
            a,b=(set(str(s).casefold().split()) for s in pair)
            return len(a & b)/max(1,len(a | b))>=.65
        near=np.fromiter((similar(pair) for pair in held.itertuples(index=False,name=None)),dtype=bool)
        return {"similar_wording":near,"different_wording":~near}
    if kind in {"table","categorical"}:
        rare=np.zeros(len(held),dtype=bool)
        for col in fit.select_dtypes(exclude="number").columns:
            counts=fit[col].value_counts()
            rare |= held[col].map(counts).fillna(0).to_numpy() < max(3,.01*len(fit))
        return {"contains_rare_category":rare,"common_categories":~rare}
    return {}


DETAIL = {
    "profile_response_pool":"estimate preferences from repeated profiles and situations using complementary models",
    "text_margin_neighbors":"compare word and subword margins with votes from similar training texts",
    "framing_rank":"compare phrase openings, endings and sequences of lexical categories",
    "contrast_rank":"learn smoothed winner/loser weights for word and subword differences",
    "edit_neighbors":"seven most similar training edits, direction-aware cosine voting",
    "ordered_catboost":"keep categorical values and append train-inferred interval, clock and duration views",
    "pair_hybrid":"word and character differences plus writing style, logistic regression",
    "edit_odds":"learn which added or removed words and characters distinguish winners",
    "pair_tfidf":"word and character differences without style",
    "pair_style":"nonlinear model on paired writing-style features, symmetric under swapped sides",
    "latent_pair":"latent text representation with nonlinear comparisons",
    "pair_hybrid_stronger_penalty":"same lexical and style features, stronger L2 regularization, C=0.25",
    "catboost":"boosted trees with native categorical features, depth 6",
    "catboost_shallow":"same categorical boosting, reduce interaction depth to 4",
    "crossed_logit":"regularized logistic regression with categorical pair interactions",
    "trees":"extremely randomized trees on encoded features",
    "trees_min_leaf4":"extremely randomized trees with at least four training rows per leaf for regularization",
    "tabm":"neural tabular ensemble on categorical embeddings",
}


def weaker_slice(first, second, slices):
    """Require separated Wilson intervals before spending a pilot on a slice."""
    bounds = {}
    for key in (first, second):
        sample = slices.get(key, {})
        n = sample.get("rows", 0)
        if not n or "correct" not in sample:
            return None
        p = sample["correct"] / n
        z = 1.96
        denominator = 1 + z*z/n
        center = (p + z*z/(2*n)) / denominator
        radius = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
        bounds[key] = (center-radius, center+radius)
    if bounds[first][1] < bounds[second][0]:
        return first
    if bounds[second][1] < bounds[first][0]:
        return second
    return None


def local_decision(kind, actions, recent):
    slices=recent["slices"]
    # These are hypotheses for the next pilot. Four-fold ranking and the
    # separate training confirmation still decide which model to deploy.
    routes = {
        "pair": {"similar_wording": "contrast_rank", "different_wording": "framing_rank"},
        "categorical": {"contains_rare_category": "profile_response_pool", "common_categories": "onehot_boost"},
    }
    if recent["metric"] == "accuracy" and kind in routes:
        first, second = routes[kind]
        weak = weaker_slice(first, second, slices)
        action = routes[kind].get(weak)
        if action in actions:
            counts = ", ".join(f"{key} {slices[key]['correct']}/{slices[key]['rows']}" for key in (first,second))
            evidence = f"{recent['action']} on the training feedback fold: {counts}. Lower slice: {weak} (separated 95% Wilson intervals)."
            return action, evidence, "Test this representation against the observed errors: " + actions[action]
    for name in ("profile_response_pool","text_margin_neighbors"):
        if name in actions:
            return name,f"Reference pilot scored {recent['score']:.4f}.",DETAIL[name]
    if kind=="pair":
        for name in ("pair_tfidf","framing_rank","contrast_rank"):
            if name in actions:
                return name,f"Prior paired-text pilot scored {recent['score']:.4f}.",DETAIL[name]
    if kind=="categorical" and "onehot_boost" in actions:
        if "trees" in actions:
            return "trees",f"Baseline pilot accuracy was {recent['accuracy']}.","Measure the other reference learner before proposing an upgrade."
        return "onehot_boost",f"Pilot accuracy was {recent['accuracy']} on categorical data.","Compare deeper boosting with direct indicators for small categories."
    if kind=="categorical" and "profile_backoff" in actions:
        return "profile_backoff",f"Pilot accuracy was {recent['accuracy']}; recurring attribute blocks were found in fitting rows.","Test cross-fitted response rates for seen profiles and the reference ensemble for unseen profiles."
    if kind=="pair" and "graph_logit" in actions:
        return "graph_logit",f"Pilot accuracy was {recent['accuracy']}; fitting text has comparable two-hop paths.","Learn with half-weight comparisons implied by two observed wins."
    if kind=="pair" and "graph_margin" in actions:
        return "graph_margin",f"Augmented comparison pilot scored {recent['accuracy']}.","Check a regularized margin loss on the same within-fold evidence."
    if kind=="pair" and "edit_odds" in actions:
        evidence=f"Correctness on similar wording was {slices.get('similar_wording')}"
        return "edit_odds",evidence,"Check whether changed words distinguish more winners."
    if kind=="categorical" and "catboost" in actions:
        return "catboost",f"Rare-category results were {slices.get('contains_rare_category')}.","Try nonlinear categorical interactions."
    if kind=="categorical" and "trees" in actions:
        return "trees",f"Last pilot accuracy was {recent['accuracy']}.","Compare randomized trees for complementary errors."
    if kind=="pair" and "pair_hybrid_stronger_penalty" in actions:
        return "pair_hybrid_stronger_penalty",f"Last pilot accuracy was {recent['accuracy']}.","Test stronger regularization."
    return next(iter(actions)),f"Last pilot {recent['metric']} was {recent['score']:.4f}.","Evaluate a different remaining representation."


def parent_evidence(folder, task_id, folds, grp, classes, spec, label_digest):
    if folder is None:
        return [], [], None
    result=read(folder/"result.json")
    trace=[json.loads(l) for l in (folder/"events.jsonl").read_text(encoding="utf-8").splitlines()]
    first=next((entry for entry in trace if entry.get("event")=="start"),{})
    if first.get("task_hash")!=task_id or result["model_sha256"]!=sha(folder/"model.joblib"):
        raise ValueError("Previous model or training features differ")
    if label_digest is not None:
        expected=first.get("train_labels_sha256")
        if expected is None:
            legacy=read(Path(__file__).with_name("legacy-training-inputs.json"))
            expected=legacy.get(result["version"],{}).get(task_id)
        if expected!=label_digest:
            raise ValueError("Training label fingerprint differs from the prior run")
    partition=pd.read_csv(folder/"folds.csv")
    if not (np.array_equal(partition["row"],np.arange(1,len(folds)+1)) and
            np.array_equal(partition["group"],grp) and np.array_equal(partition["inner_fold"],folds)):
        raise ValueError("Previous training folds differ")
    archive=joblib.load(folder/"model.joblib")
    if archive["names"]!=result["selected"] or not np.array_equal(archive["classes"],classes):
        raise ValueError("Prior checkpoint metadata differs")
    if any(archive["spec"].get(k)!=spec.get(k) for k in ("features","modality","target","numeric","pairwise","group_field","json_types")):
        raise ValueError("Input contract differs from the prior checkpoint")
    records=[]
    for e in trace:
        if e["event"]=="inherited_training":
            records.extend(e["pilot_history"])
        if e["event"]=="pilot_result":
            records.append({k:v for k,v in e.items() if k!="event"})
    if not records:
        raise ValueError("Missing previous training feedback")
    return records, archive["names"], result["model_sha256"]


def accounting(history):
    charges=[r for r in history if r["event"]=="settled"]
    pending={r["call"] for r in history if r["event"]=="reserved"}-{r["call"] for r in charges}
    return dict(charged_calls=len(charges),pending_reservations=sorted(pending),
                recorded_cost_rmb=str(sum((Decimal(r["amount"]) for r in charges),Decimal(0))))


def run(args, root, out):
    spec=read(args.task)
    files=task_files(root,spec)
    train_path=files["train"][0]
    frame=load_labeled(files["train"],spec)
    x=features(frame,spec,train_path.parent)
    y=frame[spec["target"]].to_numpy(str)
    classes=np.unique(y)
    objective=Objective(spec,classes)
    kind=family(spec,y)
    grp=groups(x,kind,spec.get("group_field"))
    folds=fold_ids(y,grp,311)
    roles=partition(y,grp)
    write_json(out/"group-overlap.json",overlap_report(grp,roles,kind,spec.get("group_field")))
    fit=np.flatnonzero(roles>=2)
    screen=np.flatnonzero(roles==1)
    task_id=budget.task_key(train_path,spec)
    label_path=files["train"][1]
    label_digest=budget.task_key(label_path,{}) if label_path else None
    prior_trials,incumbent,parent_hash=parent_evidence(args.prior,task_id,folds,grp,classes,spec,label_digest)
    version=hashlib.sha256()
    code_root=Path(__file__).resolve().parents[2]
    for path in sorted([*(code_root/"Agent").rglob("*.py"),*Path(__file__).resolve().parent.glob("*.py")]):
        version.update(path.relative_to(code_root).as_posix().encode())
        version.update(path.read_bytes())
    events=out/"events.jsonl"
    def record(**event):
        log(events,**event)
    pd.DataFrame({"row":np.arange(1,len(x)+1),"group":grp,"inner_fold":folds}).to_csv(out/"folds.csv",index=False)
    pd.DataFrame({"row":np.arange(1,len(x)+1),"group":grp,"ranking_fold":roles}).to_csv(out/"partition.csv",index=False)
    anchor, names=POOLS[kind]
    if kind=="pair":
        anchor="pair_hybrid"
        names=[*names,"pair_hybrid_stronger_penalty","pair_style"]
    if kind=="categorical":
        names=[*names,"catboost_shallow","trees_min_leaf4"]
    scheduled=[]
    allowed=None
    repeat_seed=7919 if args.repeat_ranking else None
    if repeat_seed is not None and objective.metric!="accuracy":
        raise ValueError("Repeated ranking currently supports accuracy")
    short_multiclass=(kind=="text" and 2<len(classes)<=150 and 800<=len(x)<=20000 and
                      x.iloc[fit].astype(str).apply(lambda s:s.str.len()).sum(axis=1).median()<=250)
    upgrade=args.strengthen and objective.metric=="accuracy" and (kind in {"pair","categorical"} or short_multiclass)
    if upgrade:
        if kind=="pair":
            from graph_rank import paths
            anchor="pair_hybrid"
            incumbent=[anchor]
            count=len(paths(x.iloc[fit],y[fit],classes))
            names=[anchor,"pair_tfidf","framing_rank","contrast_rank",*(["graph_logit","graph_margin"] if count else [])]
            record(event="training_structure",pilot_implied_comparisons=count)
        elif short_multiclass:
            anchor="union"
            names=[anchor,"text_margin_neighbors","char","word"]
            incumbent=[anchor]
        else:
            from profile_boost import discover
            anchor="catboost"
            incumbent=[anchor,"trees"]
            block=discover(x.iloc[fit])
            names=[anchor,"trees","onehot_boost",*(["profile_backoff"] if block else [])]
            if block and x.shape[1]<=32 and len(x)<=15000:
                anchor="profile_backoff"
                incumbent=[anchor]
                names=[anchor,"profile_response_pool","onehot_boost","trees"]
            # The backoff candidate already includes both reference learners.
            allowed={frozenset([n]) for n in names} | {frozenset(incumbent)}
            record(event="training_structure",pilot_recurring_features=block)
    if args.method_study in {"attainable","focus","aligned"}:
        if objective.metric!="accuracy" or kind not in {"pair","categorical"}:
            raise ValueError("This study requires paired text or a categorical binary task, with accuracy")
        anchor,challenger=("pair_hybrid","word_classes") if kind=="pair" else ("profile_backoff","category_kernel")
        if args.method_study in {"focus","aligned"}:
            challenger="semantic_edits" if kind=="pair" else "segment_experts"
        if args.method_study=="aligned" and kind=="categorical":
            challenger="aligned_fields"
        incumbent=[anchor]
        names=[anchor,challenger]
        scheduled=[challenger]
    elif args.method_study=="quality":
        if objective.metric!="accuracy" or kind not in {"pair","categorical"}:
            raise ValueError("This comparison requires accuracy on text pairs or binary categorical tables")
        if kind=="pair":
            anchor="pair_hybrid"
            incumbent=[anchor]
            scheduled=["language_linear","topic_style_ranker"]
        else:
            anchor="catboost"
            incumbent=[anchor,"trees"]
            scheduled=["trees","onehot_boost","profile_boost"]
        names=[anchor,*scheduled]
    elif args.method_study=="representations":
        from representation_candidates import TEXT, TABLE
        if objective.metric!="accuracy" or kind not in {"pair","categorical"}:
            raise ValueError("Representation comparison supports accuracy on text preferences and categorical tables")
        if kind=="pair":
            anchor="pair_hybrid"
            incumbent=[anchor]
            scheduled=["pair_tfidf",*TEXT]
        else:
            anchor="catboost"
            incumbent=[anchor,"trees"]
            scheduled=["trees",*TABLE]
        names=[anchor,*scheduled]
        allowed={frozenset([n]) for n in names} | {frozenset(incumbent)}
        repeat_seed=7919
    elif args.method_study:
        if kind=="pair":
            anchor="pair_hybrid"
            scheduled=["edit_neighbors"]
            incumbent=[anchor]
        elif kind=="categorical":
            anchor="catboost"
            scheduled=["trees","ordered_catboost"]
            incumbent=[anchor,"trees"]
            allowed={frozenset(m) for m in [["catboost"],["trees"],["ordered_catboost"],["catboost","trees"],["ordered_catboost","trees"]]}
        else:
            raise ValueError("This controlled comparison requires ordered text pairs or an all-categorical table")
        names=[anchor,*scheduled]
    search = roles >= 0
    profile=dict(family=kind,train_rows=int(search.sum()),features=x.shape[1],groups=len(np.unique(grp[search])),
                 class_counts=[int((y[search]==c).sum()) for c in classes],classes=len(classes))
    record(event="start",version=version.hexdigest(),task_hash=task_id,profile=profile,train_rows=len(x),
           provider=args.provider,metric=spec["metric"],available=names,train_labels_sha256=label_digest)
    slices=diagnostics(kind,x.iloc[fit],x.iloc[screen])
    actions={name:DETAIL.get(name,name) for name in names}
    choices=list(incumbent)
    summaries=[]
    predictions={}
    api_enabled=args.provider!="local"
    checkpoint=out/"model.joblib"
    if args.prior:
        old_bundle=joblib.load(args.prior/"model.joblib")
        if (old_bundle["spec"].get("metric","accuracy")!=objective.metric
                or old_bundle["spec"].get("positive_class")!=spec.get("positive_class")):
            raise ValueError("Inherited model was developed under a different objective")
        objective.threshold=old_bundle.get("decision_threshold")
        scratch=checkpoint.with_suffix(".pending")
        shutil.copyfile(args.prior/"model.joblib",scratch)
        scratch.replace(checkpoint)
        record(event="inherited_training",parent_model_sha256=parent_hash,incumbent=incumbent,pilot_history=prior_trials)
        # Include the preceding incumbent as a candidate even when an older selection removed it.
        for line in (args.prior/"events.jsonl").read_text(encoding="utf-8").splitlines():
            event=json.loads(line)
            if event["event"]=="inherited_training":
                choices.extend(n for n in event["incumbent"] if n not in choices and n in names)
        for trial in prior_trials:
            actions.pop(trial["action"],None)
    baseline_name=incumbent[0] if incumbent else anchor
    actions.pop(baseline_name,None)
    trial_count=1+min(len(scheduled) if scheduled else args.max_actions,len(actions))
    floor=.005 if args.method_study or upgrade or repeat_seed is not None else 0.
    initial_threshold=objective.threshold
    if args.method_study:
        (out/"trial-models").mkdir()
    write_json(out/"protocol.json",dict(metric=spec["metric"],roles_sha256=sha(out/"partition.csv"),
               planning_fit_folds=[2,3],planning_feedback_fold=1,ranking_folds=[0,1,2,3],
               confirmation_fold=-1,min_winning_folds=3,confirmation_ties="retain incumbent",
               initial_reference=list(incumbent) if incumbent else [anchor],max_new_planning_actions=trial_count-1,
               repeat_seed=repeat_seed,decision_check="locked_pair_repeated_group_cv" if repeat_seed else "confirmation",
               minimum_gain=floor,scheduled=scheduled,allowed_options=sorted(sorted(m) for m in allowed) if allowed else None))
    for turn in range(trial_count):
        if turn==0:
            name=baseline_name
            record(event="decision",origin="baseline",action=name)
        elif scheduled:
            name=scheduled[turn-1]
            record(event="decision",origin="preregistered",action=name)
        elif args.provider=="local" and len(summaries)>=3 and max(t["score"] for t in summaries[-2:])<=max(t["score"] for t in summaries[:-2])+.005:
            record(event="planning_stop",reason="two_pilots_without_material_gain",metric=objective.metric)
            break
        elif summaries and ((objective.metric=="accuracy" and summaries[0]["score"]==1.) or not actions):
            record(event="planning_stop",reason="perfect_screen_score" if actions else "no_remaining_action")
            break
        elif not summaries:
            name=next(iter(actions))
            record(event="decision",origin="recovery" if not summaries else "only_available",action=name)
        else:
            obs=dict(metric=objective.metric,profile=profile,trials=summaries,incumbent=incumbent,turn=turn+1,trials_left=trial_count-turn)
            decision=None
            if api_enabled:
                offered={**actions,"stop":"Stop search and confirm the best available candidate."}
                record(event="planner_observation",observation=obs,options=offered)
                try:
                    action, reason=ask(read(args.planner),obs,list(offered),events,task_id)
                    decision=dict(action=action,evidence=reason,expectation=offered[action])
                except Exception as exc:
                    api_enabled=False
                    record(event="planner_unavailable",error_type=type(exc).__name__,
                           detail=str(exc)[:180] if isinstance(exc,RuntimeError) else type(exc).__name__)
            if decision is not None:
                name=decision["action"]
                record(event="decision",origin="configured",**decision)
                if name=="stop":
                    record(event="planning_stop",reason="planner_stop")
                    break
            else:
                name,evidence,expected=local_decision(kind,actions,summaries[-1])
                record(event="decision",origin="local",action=name,evidence=evidence,expectation=expected)
        actions.pop(name,None)
        start=time.monotonic()
        try:
            model=fit_one(name,kind,x.iloc[fit],y[fit],classes)
            p=probabilities(model,x.iloc[screen],classes)
        except Exception as exc:
            record(event="pilot_failure",action=name,error=type(exc).__name__,detail=str(exc)[:300])
            continue
        diag=summarize(p,y[screen],classes,slices,objective)
        if name=="segment_experts":
            record(event="pilot_segments",field=model.field_,screen=model.screen_,selected=sorted(model.experts_))
        if name=="aligned_fields":
            record(event="column_alignment",context=model.fields_.context_,matches=model.fields_.mapping_,fit=model.fitted_)
        if args.method_study:
            save_bundle(out/"trial-models"/(name+".joblib"),spec,classes,{name:model},"planning_pilot")
            if hasattr(model,"views"):
                diag["ordered_views"]={k:sorted(v) for k,v in model.views.items()}
        diag.update(action=name,seconds=round(time.monotonic()-start,3))
        if predictions:
            base=classes[next(iter(predictions.values())).argmax(1)]==y[screen]
            ok=classes[p.argmax(1)]==y[screen]
            diag.update(comparison_action=next(iter(predictions)),corrected=int((ok & ~base).sum()),lost=int((~ok & base).sum()))
        record(event="pilot_result",**diag)
        if name not in choices:
            choices.append(name)
        summaries.append(diag)
        predictions[name]=p
        if not args.prior and not checkpoint.exists():
            save_bundle(checkpoint,spec,classes,{name:model},"pilot")
    else:
        record(event="planning_stop",reason="registered_candidates_exhausted" if scheduled else "action_budget_exhausted")
    if args.method_study and set(names)!=set(predictions):
        raise RuntimeError("Controlled comparison incomplete; inspect pilot failures before interpreting scores")
    if not choices:
        raise RuntimeError("All candidate fits failed")
    reference=list(incumbent) or [choices[0]]
    winner=select_candidate(x,y,classes,roles,grp,kind,choices,reference,out,record,objective,floor,allowed,repeat_seed)
    if args.prior and winner==incumbent and objective.threshold==initial_threshold:
        record(event="retained_checkpoint",model_sha256=parent_hash)
    else:
        try:
            final={n:fit_one(n,kind,x,y,classes) for n in winner}
            for m in final.values(): probabilities(m,x.iloc[:16],classes)
            save_bundle(checkpoint,spec,classes,final,"full_refit",objective.threshold)
        except Exception as exc:
            record(event="refit_failure",error=type(exc).__name__,detail=str(exc)[:200])
    saved=joblib.load(checkpoint)
    history=[json.loads(l) for l in events.read_text(encoding="utf-8").splitlines()]
    result=dict(version=version.hexdigest(),selected=saved["names"],stage=saved["stage"],train_rows=len(x),
                model_sha256=sha(checkpoint),planner=args.provider,
                planner_decisions=sum(r.get("origin") in {"local","configured"} for r in history),
                parent_model_sha256=parent_hash,new_pilots=[n for n in predictions if n!=baseline_name],
                run_status="refit_fallback" if any(r["event"]=="refit_failure" for r in history) else "completed",
                **accounting(history))
    result["proposed"]=winner
    result.update(metric=objective.metric,decision_threshold=saved.get("decision_threshold"))
    result["challenger_lock_sha256"]=sha(out/"challenger.json")
    result["contract_sha256"]=sha(args.task)
    write_json(out/"result.json",result)
    print(json.dumps(result,ensure_ascii=True),flush=True)


def deliver_timeout(args, out):
    checkpoint=out/"model.joblib"
    events=out/"events.jsonl"
    if not checkpoint.is_file() or not events.is_file():
        raise RuntimeError("Time limit reached before a usable checkpoint was saved")
    # Only a complete, readable atomic checkpoint is eligible for partial delivery.
    saved=joblib.load(checkpoint)
    history=[]
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            history.append(json.loads(line))
        except json.JSONDecodeError:
            break
    start=next(e for e in history if e["event"]=="start")
    locked=list(saved["names"])
    challenger=out/"challenger.json"
    write_json(out/"result.json",dict(run_status="timeout_checkpoint",version=start["version"],
               selected=locked,stage=saved["stage"],train_rows=start["train_rows"],
               proposed=None,planner=args.provider,planner_decisions=sum(e.get("origin") in {"local","configured"} for e in history),
               model_sha256=sha(checkpoint),contract_sha256=sha(args.task),
               challenger_lock_sha256=sha(challenger) if challenger.is_file() else None,**accounting(history)))
    write_json(out/"timeout.json",dict(limit_seconds=args.seconds,checkpoint_stage=saved["stage"],selected=locked))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data","out"): parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--task",type=Path,help="Existing manifest; omit to read the task automatically")
    parser.add_argument("--readme",help="README path inside the data root or ZIP")
    parser.add_argument("--trust-object-features",action="store_true",help="Allow training on trusted NumPy object feature arrays")
    parser.add_argument("--reference-cache",type=Path,help="Directory for versioned paper evidence")
    parser.add_argument("--fetch-citations",action="store_true",help="Allow retrieving an explicitly cited paper to resolve a missing metric")
    parser.add_argument("--prior",type=Path,help="Continue from an existing run's training feedback and saved model")
    parser.add_argument("--provider",choices=["local","configured"],default="local")
    parser.add_argument("--planner",type=Path,help="Existing course planner configuration for --provider configured")
    parser.add_argument("--train-only",action="store_true",help="Stop at the saved model, before validation scoring")
    parser.add_argument("--strengthen",action="store_true",help="Try structure-supported upgrades from a cold local accuracy baseline")
    parser.add_argument("--objective",choices=sorted(METRICS),help="Assessment objective explicitly supplied with the task")
    parser.add_argument("--method-study",nargs="?",const="original",choices=["original","representations","quality","attainable","focus","aligned"],
                        help="Run a local, preregistered method comparison from a cold start")
    parser.add_argument("--repeat-ranking",action="store_true",
                        help="Confirm a locked accuracy challenger on new grouped folds of the ranking rows")
    parser.add_argument("--max-actions",type=int,choices=range(0,9),default=4,help="Maximum new adaptive actions after the baseline")
    parser.add_argument("--seconds",type=int,default=600)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if (args.provider=="configured") != bool(args.planner):
        parser.error("--provider configured requires --planner; local planning needs neither")
    if args.strengthen and (args.prior or args.method_study or args.provider!="local"):
        parser.error("--strengthen uses the local planner from a cold start")
    if args.task and args.objective and read(args.task).get("metric")!=args.objective:
        parser.error("Explicit objective disagrees with task manifest")
    if args.method_study and (args.prior or args.provider!="local"):
        parser.error("--method-study uses a cold start and --provider local")
    if args.method_study=="original" and args.repeat_ranking:
        parser.error("The original study has a fixed holdout protocol")
    if args.seconds<=0: parser.error("positive --seconds required")
    out=args.out.resolve()
    if args.worker:
        with data_root(args.data.resolve()) as root:
            run(args,root,out)
        return
    if args.task and args.readme: parser.error("Choose --task or --readme")
    out.mkdir(parents=True,exist_ok=False)
    worker_args=list(sys.argv[1:])
    if args.task is None:
        from task_intake import generate, IntakeError
        try:
            args.task=generate(args.data,args.readme,out,args.trust_object_features,args.reference_cache,args.fetch_citations,args.objective)
        except IntakeError as exc:
            parser.exit(2,f"{exc}\n")
        if read(args.task)["metric"] not in METRICS:
            parser.exit(2,"Metric recognised but unsupported for training. See intake.json.\n")
        if read(args.task).get("allow_pickle_features") is False:
            parser.exit(2,"Object array recognised. For trusted course data, pass --trust-object-features.\n")
        worker_args.extend(["--task",str(args.task)])
    try:
        subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),*worker_args,"--worker"],
                       timeout=args.seconds,check=True)
    except subprocess.TimeoutExpired:
        if not (out/"result.json").is_file():
            deliver_timeout(args,out)
    if not args.train_only and read(args.task).get("validation"):
        from validate_model import evaluate
        evaluate(args.task,args.data,out)


if __name__=="__main__":
    main()
