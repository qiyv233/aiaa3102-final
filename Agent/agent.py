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
from validation import split_rows
from submission import write_submission, validate_probabilities

from features import CANDIDATES, build, infer_modality
from planning import ask, record, rule


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_csv(path):
    # Preserve leading zeros in IDs and label names. Numeric features are converted later.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def inputs(raw, bundle, root):
    x = raw[bundle["features"]].copy()
    for col in bundle["numeric"]:
        x[col] = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if bundle["mode"] == "image":
        if len(x.columns) != 1:
            raise ValueError("Image manifests require one path column.")
        col = x.columns[0]
        x[col] = x[col].map(lambda p: str((root / p).resolve()))
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


def predict(bundle, file, dest, template=None):
    raw = load_csv(file)
    model, cfg = bundle["model"], bundle["spec"]
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
    write_submission(result,dest,template,cfg.get("id"))


def run(args):
    start = time.monotonic()
    cfg = read_json(args.task)
    root = args.task.parent
    train_file = root / cfg["train"]
    raw = load_csv(train_file)
    target = cfg["target"]
    reserved = {target, cfg.get("id"), cfg.get("group_column"), cfg.get("time_column")}
    features = cfg.get("features", [c for c in raw if c not in reserved])
    if reserved.intersection(features) or len(set(features)) != len(features):
        raise ValueError("Invalid or duplicated input features")
    if cfg.get("test") and (root/cfg["test"]).resolve() in {train_file.resolve(), (root/cfg.get("validation",cfg["train"])).resolve()}:
        raise ValueError("Test file overlaps training or validation")
    if target in features or cfg.get("id") in features or not features:
        raise ValueError("Features must exclude target and ID and be nonempty.")
    y = raw[target]
    if y.eq("").any() or y.nunique() < 2:
        raise ValueError("Missing labels or fewer than two classes.")
    seed = int(cfg.get("seed", 17))
    valid_file = root / cfg["validation"] if "validation" in cfg else None
    if valid_file and valid_file.resolve() == train_file.resolve():
        raise ValueError("Training and validation files coincide")
    train, valid = split_rows(raw, cfg, seed, load_csv(valid_file) if valid_file else None)
    valid_root = valid_file.parent if valid_file else train_file.parent
    mode = cfg.get("modality") or infer_modality(train[features])
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
    if args.pool == "reviewed":
        from reviewed import initial_choices, action
        choices = initial_choices(mode,x)
    if args.pool == "transfer" and mode == "image":
        choices.insert(1,"resnet18")
    if args.pool == "categorical" and mode == "table":
        choices.insert(1,"catboost")
    if args.pool in ("expanded","adaptive"):
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
           "seed": seed, "train_rows": len(train), "valid_rows": len(valid),
           "validation_sha256": hashlib.sha256(valid.to_csv(index=False).encode()).hexdigest()})
    while choices:
        if args.max_trials is not None and len(history) >= args.max_trials:
            break
        if best is not None and (args.strategy == "baseline" or time.monotonic()-start >= args.seconds):
            break
        name, why = rule(mode, choices, history)
        if args.pool == "reviewed":
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
    if cfg.get("test"):
        template = args.task.parent/cfg["submission_template"] if cfg.get("submission_template") else None
        predict(bundle, args.task.parent/cfg["test"], args.out/"predictions.csv",template)
    result = dict(selected=bundle["candidate"], validation_score=bundle["validation_score"],
                  status=status, seconds=round(time.monotonic()-start,4))
    record(args.out/"run.jsonl", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, type=Path)
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
        bundle = joblib.load(a.model)
        if "weights" in bundle:
            from advanced.engine import probabilities, export
            spec = read_json(a.task) if a.task else bundle.get("spec")
            if spec is None:
                raise ValueError("Pass --task for this historical ensemble")
            export(bundle,a.input,probabilities(bundle,a.input),spec,a.output,a.template)
        else:
            predict(bundle, a.input, a.output,a.template)
