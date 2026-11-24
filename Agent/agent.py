import argparse
import importlib.util
import subprocess
import sys
import math
import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score, roc_auc_score
from advanced.learners import candidate
from budget import task_key
from task_data import (
    data_file, data_root, image_inputs, load_csv, load_features, load_labeled, load_labels, task_files,
)
from validation import split_rows
from submission import write_submission, validate_probabilities

from features import CANDIDATES, build, infer_modality
from paired_text import PAIR_CANDIDATES, pair_estimator
from planning import ask, record, rule


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def inputs(raw, bundle, root):
    x = raw[bundle["features"]].copy()
    for col in bundle["numeric"]:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if bundle["mode"] == "image":
        return image_inputs(x, root)
    return x


def evaluate(metric, y, probs, classes, positive):
    probs = validate_probabilities(probs, classes)
    pred = classes[probs.argmax(axis=1)]
    if metric == "accuracy":
        score = accuracy_score(y, pred)
    elif metric == "macro_f1":
        score = f1_score(y, pred, labels=classes, average="macro", zero_division=0)
    elif metric == "log_loss":
        score = log_loss(y, probs, labels=classes)
    elif metric == "roc_auc":
        if len(classes) == 2:
            pos = str(positive) if positive is not None else classes[1]
            score = roc_auc_score(y == pos, probs[:, classes.tolist().index(pos)])
        else:
            score = roc_auc_score(y, probs, labels=classes, multi_class="ovr", average="macro")
    else:
        raise ValueError(f"Unsupported metric {metric}")
    recalls = recall_score(y, pred, labels=classes, average=None, zero_division=0)
    return float(score), dict(zip(classes.tolist(), recalls.tolist()))


def save(bundle, out):
    tmp = out / "model.tmp"
    joblib.dump(bundle, tmp)
    os.replace(tmp, out / "model.joblib")


def predict(bundle, file, dest, template=None, *, score_test=False, label_file=None):
    model, cfg = bundle["model"], bundle["spec"]
    raw = load_features(file, cfg)
    x = inputs(raw, bundle, file.parent)
    result = pd.DataFrame(index=raw.index)
    if cfg.get("id"):
        result[cfg["id"]] = raw[cfg["id"]]
    output = cfg.get("output", "label")
    if output in ("label", "positive_probability") and cfg.get("prediction_column", cfg["target"]) in result:
        raise ValueError("Prediction column collides with ID")
    probs = validate_probabilities(model.predict_proba(x), model.classes_)
    if output == "label":
        result[cfg.get("prediction_column", cfg["target"])] = model.classes_[probs.argmax(axis=1)]
    elif output in ("probabilities", "positive_probability"):
        labels = model.classes_.tolist()
        if output == "positive_probability":
            pos = str(cfg["positive_class"])
            result[cfg["prediction_column"]] = probs[:, labels.index(pos)]
        else:
            for i, label in enumerate(labels):
                name = cfg.get("probability_columns", {}).get(label, label)
                if name in result:
                    raise ValueError(f"Duplicate output column {name}")
                result[name] = probs[:, i]
    else:
        raise ValueError("Unknown output format.")
    metrics = None
    if score_test:
        y = load_labels(label_file, len(raw)) if label_file is not None else raw[cfg["target"]]
        if not y.isin(model.classes_).all():
            raise ValueError("Test labels contain a missing or unknown class")
        metric = cfg["metric"]
        score, _ = evaluate(metric, y, probs, model.classes_, cfg.get("positive_class"))
        metrics = dict(test_metric=metric, test_score=score, test_rows=len(y))
    write_submission(result,dest,template,cfg.get("id"))
    return metrics


def run(args):
    cfg = read_json(args.task)
    with data_root(getattr(args, "data", None) or args.task.parent) as root:
        run_trials(args, cfg, root)


def run_trials(args, cfg, root):
    start = time.monotonic()
    files = task_files(root, cfg)
    train_file, _ = files["train"]
    raw = load_labeled(files["train"], cfg)
    target = cfg["target"]
    reserved = {target, cfg.get("id"), cfg.get("group_column"), cfg.get("time_column")}
    features = cfg.get("features", [c for c in raw if c not in reserved])
    if reserved.intersection(features) or len(set(features)) != len(features):
        raise ValueError("Invalid or duplicated input features")
    if target in features or cfg.get("id") in features or not features:
        raise ValueError("Features must exclude target and ID and be nonempty.")
    y = raw[target]
    if y.eq("").any() or y.nunique() < 2:
        raise ValueError("Missing labels or fewer than two classes.")
    seed = int(cfg.get("seed", 17))
    validation = files.get("validation")
    train, valid = split_rows(raw, {**cfg, "features":features}, seed, load_labeled(validation, cfg) if validation else None)
    valid_root = validation[0].parent if validation else train_file.parent
    pairwise = cfg.get("pairwise", False)
    mode = cfg.get("modality") or ("text" if pairwise else infer_modality(train[features]))
    if pairwise and (mode != "text" or len(features) != 2 or y.nunique() != 2):
        raise ValueError("Pairwise classification requires two ordered text features and two classes")
    numeric = cfg.get("numeric", [col for col in features if len(v := train[col].replace("", np.nan).dropna())
                                  and pd.to_numeric(v, errors="coerce").notna().all()]) if mode == "table" else []
    bundle = {"spec": cfg, "mode": mode, "features": features, "numeric": numeric}
    x = inputs(train, bundle, train_file.parent)
    v = inputs(valid, bundle, valid_root)
    metric = cfg["metric"]
    if metric not in ("accuracy", "macro_f1", "log_loss", "roc_auc"):
        raise ValueError("Unsupported official metric; implement it before running.")
    out = args.out

    history, choices = [], list(CANDIDATES[mode])
    if pairwise:
        choices = list(PAIR_CANDIDATES)
    elif args.pool == "reviewed":
        from reviewed import initial_choices, action
        choices = initial_choices(mode,x)
    elif args.pool == "transfer" and mode == "image":
        choices.insert(1,"resnet18")
    elif args.pool == "categorical" and mode == "table":
        choices.insert(1,"catboost")
    elif args.pool in ("expanded","adaptive"):
        extra = {"table":["rbf","hist"], "text":["union","nb_ratio"], "image":["gradient_svm","cnn"]}
        choices.extend(extra[mode])
        if mode == "table" and len(train) > 5000:
            choices.remove("rbf")
        if mode == "image" and importlib.util.find_spec("torch") is None:
            choices.remove("cnn")
    best, best_name = None, None
    planner = read_json(args.planner) if args.planner else None
    if args.strategy == "llm" and not planner:
        raise ValueError("--strategy llm requires --planner.")
    budget_key = task_key(train_file, cfg) if planner else None
    record(out / "run.jsonl", {"task": cfg, "strategy": args.strategy, "pool": args.pool, "split": cfg.get("split","stratified"), "candidates": list(choices), "mode": mode,
           "validation_source": "supplied" if validation else "internal",
           "seed": seed, "train_rows": len(train), "valid_rows": len(valid),
           "validation_sha256": hashlib.sha256(valid.to_csv(index=False).encode()).hexdigest()})
    while choices:
        if args.max_trials is not None and len(history) >= args.max_trials:
            break
        if best is not None and (args.strategy == "baseline" or time.monotonic()-start >= args.seconds):
            break
        name, why = rule(mode, choices, history)
        if pairwise:
            why = PAIR_CANDIDATES[name]
        elif args.pool == "reviewed":
            name, why = action(mode,choices,history,x,train[target])
        if args.pool == "transfer" and history and "resnet18" in choices:
            name, why = "resnet18", "Evaluate pretrained RGB features on the same validation rows."
        if args.pool in ("expanded","adaptive") and history and not name.startswith("balanced"):
            priority = {"table":["rbf","hist"], "text":["union","nb_ratio"],
                        "image":["gradient_svm","cnn"]}[mode]
            alt = next((n for n in priority if n in choices), None)
            if alt is not None:
                name, why = alt, "Try a complementary model family using full training epochs."
        if args.pool == "adaptive" and mode == "text" and len(history) >= 2 and "balanced_char" in choices:
            counts = train[target].value_counts()
            if counts.max() >= 2 * counts.min():
                name, why = "balanced_char", "Reserve a trial for inverse-frequency weighting on imbalanced text."
        if args.strategy == "fixed":
            name, why = choices[0], "Use the predeclared candidate order."
        if args.strategy == "strong_fixed":
            order = {"table": ["linear", "rbf", "hist"],
                     "text": ["char", "union", "balanced_char"],
                     "image": ["color", "gradient_svm", "cnn"]}[mode]
            name = next((item for item in order if item in choices), choices[0])
            why = "Use the predeclared modality-specific order."
        if planner and args.strategy == "llm" and history:
            try:
                name, why = ask(planner, {"mode": mode, "metric": metric, "train_rows": len(x),
                                "class_counts": train[target].value_counts().to_dict(),
                                "experiments": history, "seconds_left": int(args.seconds-time.monotonic()+start)},
                                choices, out / "ledger.jsonl", budget_key)
            except Exception as exc:
                record(out / "run.jsonl", {"planner_fallback": type(exc).__name__, "detail": str(exc)[:300]})
                planner = None
        choices.remove(name)
        event = {"candidate": name, "reason": why}
        tick = time.monotonic()
        try:
            model = estimator(mode, name, x, seed).fit(x, train[target])
            score, recalls = evaluate(metric, valid[target], model.predict_proba(v), model.classes_,
                                      cfg.get("positive_class"))
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation score")
            event.update(status="ok", score=score, recall=recalls)
            if best is None or (score < best if metric == "log_loss" else score > best):
                save(dict(bundle, model=model, candidate=name, validation_score=score), out)
                best, best_name = score, name
        except Exception as exc:
            event.update(status="failed", error=str(exc)[:500])
        event["seconds"] = round(time.monotonic()-tick, 4)
        history.append(event)
        record(out / "experiments.jsonl", event)
    if best is None:
        raise RuntimeError("No valid candidate; see experiments.jsonl.")
    # Refit on train + internal validation only; respect a supplied validation split.
    if "validation" not in cfg:
        try:
            full = inputs(raw, bundle, train_file.parent)
            model = estimator(mode, best_name, full, seed).fit(full, y)
            save(dict(bundle, model=model, candidate=best_name, validation_score=best), out)
        except Exception as exc:
            record(out / "run.jsonl", {"refit_fallback": str(exc)[:500]})

def estimator(mode, name, frame, seed):
    if name in PAIR_CANDIDATES:
        return pair_estimator(name, seed)
    if name == "catboost":
        from catboost import CatBoostClassifier
        categories = frame.select_dtypes(exclude="number").columns.tolist()
        return CatBoostClassifier(cat_features=categories,iterations=250,depth=6,learning_rate=0.08,
                                  random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False)
    if name == "resnet18":
        from transfer import FrozenResNet18
        return FrozenResNet18(seed)
    if name in CANDIDATES[mode]:
        return build(mode, name, frame, seed)
    return candidate(mode, name, frame, seed)


def supervise(args):
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Training time limit must be positive and finite")
    score_test = getattr(args, "evaluate_test", False)
    if score_test and not read_json(args.task).get("test"):
        raise ValueError("--evaluate-test requires a test split in the task manifest")
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
        env.setdefault(key, "2")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    status = "completed"
    try:
        subprocess.run(command, timeout=args.seconds, check=True, env=env)
    except subprocess.TimeoutExpired:
        status = "training_timeout"
        record(args.out/"run.jsonl", {"event":status,"training_limit":args.seconds})
    except subprocess.CalledProcessError as error:
        if not (args.out/"model.joblib").exists():
            raise
        status = "worker_failed_with_checkpoint"
        record(args.out/"run.jsonl", {"event":status,"returncode":error.returncode})
    model_file = args.out/"model.joblib"
    if not model_file.is_file():
        raise RuntimeError("Training ended without a valid checkpoint: " + status)
    bundle = joblib.load(model_file)
    cfg = bundle["spec"]
    if score_test:
        record(args.out / "run.jsonl", {"event": "model_selected", "candidate": bundle["candidate"],
                                       "validation_score": bundle["validation_score"]})
    test_metrics = None
    if cfg.get("test") and not getattr(args, "validation_only", False):
        with data_root(getattr(args, "data", None) or args.task.parent) as root:
            test_file, label_file = task_files(root, cfg)["test"]
            template = data_file(root, cfg["submission_template"]) if cfg.get("submission_template") else None
            test_metrics = predict(bundle, test_file, args.out/"predictions.csv", template,
                                   score_test=score_test, label_file=label_file)
    result = dict(selected=bundle["candidate"], validation_score=bundle["validation_score"],
                  status=status, seconds=round(time.monotonic()-start,4))
    if test_metrics is not None:
        result.update(test_metrics)
        if args.strategy != "llm":
            result.update(llm_calls=0, llm_cost_rmb=0)
        (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    record(args.out/"run.jsonl", result)
    print(json.dumps(result))


if __name__ == "__main__":
    selection_dir = Path(__file__).resolve().parents[1] / "experiments" / "selection"
    commands = {"inspect":"task_intake.py", "develop":"develop.py", "validate":"validate_model.py"}
    if sys.argv[1:2] and sys.argv[1] in commands:
        raise SystemExit(subprocess.call([sys.executable,"-B",str(selection_dir/commands[sys.argv[1]]),*sys.argv[2:]]))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Infer the task from its documentation and training files")
    sub.add_parser("develop", help="Select a model using grouped training folds and confirmation")
    sub.add_parser("validate", help="Evaluate a locked model on its supplied validation split")
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, type=Path)
    r.add_argument("--data", type=Path, help="Data directory or ZIP archive for task-relative paths")
    evaluation = r.add_mutually_exclusive_group()
    evaluation.add_argument("--evaluate-test", action="store_true",
                            help="Score a labeled test split after model selection")
    evaluation.add_argument("--validation-only", action="store_true",
                            help="Select and save a model without reading the test split")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--strategy", choices=["baseline", "fixed", "strong_fixed", "rule", "llm"], default="rule")
    r.add_argument("--max-trials", type=int, choices=range(1, 9),
                   help="Maximum attempted candidates, including failures (1 to 8).")
    r.add_argument("--pool", choices=["classic","expanded","adaptive","transfer","categorical","reviewed"], default="classic")
    r.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    r.add_argument("--planner", type=Path)
    r.add_argument("--seconds", type=float, default=600)
    study = sub.add_parser("study", help="Two-stage stratified cross-validation research run")
    study.add_argument("--task", required=True,type=Path)
    study.add_argument("--out", required=True,type=Path)
    study.add_argument("--audit",action="store_true")
    study.add_argument("--planner",type=Path)
    p = sub.add_parser("predict")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--task", type=Path, help="Original task manifest for archived research ensembles")
    p.add_argument("--template", type=Path, help="Match sample submission ID and column order")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--data", type=Path, help="Data directory or ZIP archive for the prediction input")
    p.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.command == "run":
        if a.pool == "reviewed" and a.max_trials is None:
            a.max_trials = 3
        if a.worker:
            run(a)
        else:
            supervise(a)
    elif a.command=="study":
        from advanced.engine import run as study_run
        study_run(a.task,a.out,a.audit,a.planner)
    else:
        from advanced import learners
        sys.modules.setdefault("learners", learners)
        sys.path.append(str(selection_dir))
        bundle = joblib.load(a.model)
        with data_root(a.data or Path.cwd()) as root:
            file = data_file(root, a.input) if a.data else a.input
            template = data_file(root, a.template) if a.data and a.template else a.template
            if "members" in bundle and "names" in bundle and "classes" in bundle:
                if template:
                    parser.error("Grouped checkpoints export prediction and per-class probabilities without a template")
                from models import predict_file
                predict_file(a.model,file,a.output,bundle=bundle)
            elif "weights" in bundle:
                from advanced.engine import probabilities, export
                spec = read_json(a.task) if a.task else bundle.get("spec")
                if spec is None:
                    raise ValueError("Pass --task for this historical ensemble")
                export(bundle,file,probabilities(bundle,file,spec),spec,a.output,template)
            else:
                predict(bundle, file, a.output, template)
