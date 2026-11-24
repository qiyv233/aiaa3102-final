"""Rank on grouped out-of-fold forecasts, then confirm one locked decision."""
from itertools import combinations
from time import monotonic

from models import fit_one, fold_ids, np, pd, probabilities, sha
from select_agent import write_json
from objective import Objective


def partition(y, groups):
    # Role -1 is the confirmation set. Only the ranking subset is visible to the planner.
    roles = np.full(len(y), -1, dtype=int)
    ranking = fold_ids(y, groups, 1907) != 0
    roles[ranking] = fold_ids(y[ranking], groups[ranking], 2047)
    if set(roles.tolist()) != {-1, 0, 1, 2, 3}:
        raise ValueError("Cannot form four ranking folds and a separate confirmation set")
    if not pd.DataFrame({"group": groups, "role": roles}).groupby("group").role.nunique().eq(1).all():
        raise ValueError("A group appears in multiple roles")
    return roles


def overlap_report(groups, roles, kind, field):
    ranking = roles >= 0
    boundaries = [("pilot", roles >= 2, roles == 1)]
    boundaries.extend((f"ranking_fold_{fold}", ranking & (roles != fold), roles == fold)
                      for fold in range(4))
    boundaries.append(("confirmation", ranking, roles == -1))
    checks = {}
    for name, fit, held in boundaries:
        fit_keys, held_keys = set(groups[fit]), set(groups[held])
        shared = fit_keys & held_keys
        if not fit_keys or not held_keys or shared:
            raise ValueError(f"Empty split or overlapping groups at {name}")
        checks[name] = dict(fit_rows=int(fit.sum()), evaluated_rows=int(held.sum()),
                            fit_groups=len(fit_keys), evaluated_groups=len(held_keys),
                            shared_groups=len(shared))
    strategy = "normalized_field" if field else {"pair":"connected_pairs", "image":"pixel_hash"}.get(kind, "complete_feature_row")
    return dict(strategy=strategy, field=field, groups=len(set(groups)), rows=len(groups), checks=checks)


def option_stats(members, predictions, actual, folds, objective, threshold):
    p = np.mean([predictions[m] for m in members], axis=0)
    indices = objective.labels(p, threshold)
    correct = indices == actual
    return {"members": list(members), "correct": int(correct.sum()), "threshold":threshold,
            "score":objective.score(actual,indices),
            "fold_scores":[objective.score(actual[folds==f],indices[folds==f]) for f in range(4)],
            "by_fold": [int(correct[folds == f].sum()) for f in range(4)]}


def nominate(predictions, incumbent, actual, folds, objective=None, floor=0., allowed=None):
    if objective is None:
        objective=Objective({"metric":"accuracy"},np.arange(2))
    if not all(m in predictions for m in incumbent):
        return list(incumbent), [], "incumbent_incomplete"
    options = [(name,) for name in predictions] + list(combinations(predictions, 2))
    if allowed:
        options=[o for o in options if frozenset(o) in allowed or set(o)==set(incumbent)]
    thresholds = [None, .25, .35, .45, .50, .55, .65, .75] if objective.metric=="f1" else [None]
    if objective.threshold not in thresholds:
        thresholds.append(objective.threshold)
    stats = [option_stats(o, predictions, actual, folds,objective,t) for o in options for t in thresholds]
    reference = next(s for s in stats if set(s["members"]) == set(incumbent) and s["threshold"]==objective.threshold)
    eligible = []
    for s in stats:
        delta = [a-b for a,b in zip(s["by_fold"], reference["by_fold"])]
        s["net_by_fold"] = delta
        s["net"] = s["correct"] - reference["correct"]
        s["score_gain"] = s["score"] - reference["score"]
        fold_gains=np.subtract(s["fold_scores"],reference["fold_scores"])
        if s["score_gain"] > 1e-12 and s["score_gain"] + 1e-12 >= floor and sum(fold_gains > 1e-12) >= 3:
            eligible.append(s)
    if not eligible:
        return list(incumbent), stats, "no_consistent_gain"
    best = min(eligible, key=lambda s: (-s["score"], len(s["members"]), tuple(sorted(s["members"])),
                                       abs((s["threshold"] if s["threshold"] is not None else .5)-.5)))
    objective.threshold=best["threshold"]
    return best["members"], stats, "positive_total_and_three_folds"


def grouped_forecasts(x, y, classes, rank, folds, kind, names, record, stage, strict):
    forecasts = {}
    for name in dict.fromkeys(names):
        start = monotonic()
        f = None
        try:
            probability = np.empty((len(rank),len(classes)))
            for f in range(4):
                fit = rank[folds != f]
                val = rank[folds == f]
                learner = fit_one(name, kind, x.iloc[fit], y[fit], classes)
                probability[folds == f] = probabilities(learner, x.iloc[val], classes)
            forecasts[name] = probability
            record(event=stage+"_completed", action=name, folds=4, rows=len(rank), seconds=round(monotonic()-start,3))
        except Exception as exc:
            record(event=stage+"_failed", action=name, fold=f, error=type(exc).__name__, detail=str(exc)[:200])
            if strict:
                raise
    return forecasts


def select_candidate(x, y, classes, roles, groups, kind, names, incumbent, out, record, objective=None, floor=0., allowed=None,repeat_seed=None):
    objective=objective or Objective({"metric":"accuracy"},classes)
    incumbent_threshold=objective.threshold
    rank = np.flatnonzero(roles >= 0)
    final = np.flatnonzero(roles == -1)
    folds = roles[rank]
    forecasts=grouped_forecasts(x,y,classes,rank,folds,kind,[*incumbent,*names],record,"ranking",repeat_seed is not None)
    actual = np.searchsorted(classes, y[rank])
    archive = out/"ranking.npz"
    np.savez_compressed(archive, row=rank+1, fold=folds, group=groups[rank],
                        actual_index=actual, classes=classes, **forecasts)
    challenger, stats, reason = nominate(forecasts, incumbent, actual, folds, objective, floor, allowed)
    lock = {"incumbent":list(incumbent), "challenger":challenger, "reason":reason,
            "ranking_rows_by_fold":np.bincount(folds,minlength=4).tolist(),
            "options":stats, "ranking_sha256":sha(archive),"metric":objective.metric,
            "minimum_gain":floor,"incumbent_threshold":incumbent_threshold,"challenger_threshold":objective.threshold}
    write_json(out/"challenger.json",lock)
    record(event="challenger_locked",challenger=challenger, reason=reason, lock_sha256=sha(out/"challenger.json"))
    if set(challenger)==set(incumbent) and objective.threshold==incumbent_threshold:
        record(event="selection_lock",winner=incumbent,reason=reason,confirmation_scored=False)
        return list(incumbent)
    if repeat_seed is not None:
        if objective.metric!="accuracy":
            raise ValueError("Repeated comparison requires accuracy")
        repeat_folds=fold_ids(y[rank],groups[rank],repeat_seed)
        repeated=grouped_forecasts(x,y,classes,rank,repeat_folds,kind,[*incumbent,*challenger],record,"repeat",True)
        np.savez_compressed(out/"repeat.npz",row=rank+1,fold=repeat_folds,group=groups[rank],actual_index=actual,
                            classes=classes,**repeated)
        old=option_stats(incumbent,repeated,actual,repeat_folds,objective,None)
        new=option_stats(challenger,repeated,actual,repeat_folds,objective,None)
        wins=int((np.subtract(new["by_fold"],old["by_fold"])>0).sum())
        gain=new["score"]-old["score"]
        winner=challenger if gain>0 and gain+1e-12>=floor and wins>=3 else incumbent
        record(event="selection_lock",winner=list(winner),confirmation_scored=False,seed=repeat_seed,
               incumbent_score=old["score"],challenger_score=new["score"],winning_folds=wins,
               reason="repeat_passed" if winner==challenger else "repeat_did_not_confirm",
               repeat_sha256=sha(out/"repeat.npz"))
        return list(winner)
    try:
        check = {}
        for name in dict.fromkeys([*incumbent,*challenger]):
            model = fit_one(name,kind,x.iloc[rank],y[rank],classes)
            check[name] = probabilities(model,x.iloc[final],classes)
        old = objective.labels(np.mean([check[n] for n in incumbent],axis=0),incumbent_threshold)
        new = objective.labels(np.mean([check[n] for n in challenger],axis=0),objective.threshold)
        truth = np.searchsorted(classes,y[final])
        gained = int(((old!=truth)&(new==truth)).sum())
        lost = int(((old==truth)&(new!=truth)).sum())
        np.savez_compressed(out/"confirmation.npz",row=final+1,group=groups[final],actual_index=truth,
                            incumbent_prediction=old,challenger_prediction=new)
        old_score, new_score = objective.score(truth,old), objective.score(truth,new)
        gain=new_score-old_score
        promote=gain>1e-12 and gain+1e-12>=floor
        winner = challenger if promote else incumbent
        if not promote:
            objective.threshold=incumbent_threshold
        record(event="selection_lock",winner=list(winner),confirmation_scored=True,rows=len(truth),
               incumbent_score=old_score,challenger_score=new_score,decision_threshold=objective.threshold,
               incumbent_correct=int((old==truth).sum()),challenger_correct=int((new==truth).sum()),
               corrected=gained,lost=lost,reason="positive_confirmation_gain" if promote else "confirmation_did_not_improve")
    except Exception as exc:
        winner = incumbent
        objective.threshold=incumbent_threshold
        record(event="selection_lock",winner=list(winner),confirmation_scored=False,
               reason="confirmation_failed",error=type(exc).__name__,detail=str(exc)[:200])
    return list(winner)
